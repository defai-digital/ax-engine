import Foundation
import Combine
import Darwin

public struct ProcessResult: Sendable {
    public let status: Int32
    public let stdout: String
    public let stderr: String
    public let truncated: Bool
}

// Pipe reads occur off the main actor. The lock protects the bounded byte stores;
// a polling consumer coalesces UI notifications instead of scheduling per line.
private final class CapturedOutput: @unchecked Sendable {
    private let lock = NSLock()
    private var stdout = Data()
    private var stderr = Data()
    private var overflow = false
    private let limit = 4 * 1024 * 1024
    func append(_ data: Data, isError: Bool) {
        lock.lock(); defer { lock.unlock() }
        if isError {
            stderr.append(data)
            if stderr.count > limit { stderr.removeFirst(stderr.count - limit); overflow = true }
        } else {
            stdout.append(data)
            if stdout.count > limit { stdout.removeFirst(stdout.count - limit); overflow = true }
        }
    }
    func snapshot(status: Int32 = -1) -> ProcessResult {
        lock.lock(); defer { lock.unlock() }
        return ProcessResult(status: status, stdout: String(decoding: stdout, as: UTF8.self),
                             stderr: String(decoding: stderr, as: UTF8.self), truncated: overflow)
    }
}

@MainActor
public final class ProcessJob: ObservableObject {
    @Published public private(set) var running = false
    @Published public private(set) var cancelled = false
    @Published public private(set) var output = ""
    public private(set) var processID: Int32?
    private var process: Process?
    private var stopTask: Task<Void, Never>?

    public init() {}

    public func run(executable: URL, arguments: [String], environment: [String: String] = [:],
                    timeout: TimeInterval? = nil,
                    update: ((ProcessResult) -> Void)? = nil) async throws -> ProcessResult {
        guard !running else { throw AppFailure.message("This operation is already running.") }
        try Task.checkCancellation()
        let child = Process()
        child.executableURL = executable
        child.arguments = arguments
        var env = ProcessInfo.processInfo.environment
        let developmentRuntime = Bundle.main.object(forInfoDictionaryKey: "AXEngineDevelopmentRuntime") as? String
        // A development helper environment belongs only to the source-built
        // runtime. Let installed runtimes choose their own download environment.
        let pythonBin = executable.path == developmentRuntime
            ? Bundle.main.object(forInfoDictionaryKey: "AXEngineDevelopmentPythonBin") as? String : nil
        let localBins = [executable.deletingLastPathComponent().path, pythonBin].compactMap { $0 }
        env["PATH"] = localBins.joined(separator: ":") + ":/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + (env["PATH"] ?? "")
        env.merge(environment) { _, value in value }
        child.environment = env
        child.standardInput = FileHandle.nullDevice
        let out = Pipe(), err = Pipe()
        child.standardOutput = out; child.standardError = err
        let capture = CapturedOutput()
        let readers = DispatchGroup()
        try child.run()
        process = child; processID = child.processIdentifier
        let pid = child.processIdentifier
        let ownsGroup = getpgid(pid) == pid && pid != getpgrp()
        running = true; cancelled = false; output = ""
        for (pipe, isError) in [(out, false), (err, true)] {
            // Release parent's write ends so EOF follows the child's exit.
            try? pipe.fileHandleForWriting.close()
            readers.enter()
            DispatchQueue.global(qos: .utility).async {
                defer { readers.leave() }
                // FileHandle.read(upToCount:) may wait to fill a buffer on a
                // pipe. A POSIX read delivers the bytes currently available,
                // allowing small progress events to reach the UI before exit.
                let descriptor = pipe.fileHandleForReading.fileDescriptor
                var bytes = [UInt8](repeating: 0, count: 16_384)
                while true {
                    let count = Darwin.read(descriptor, &bytes, 16_384)
                    if count < 0 && errno == EINTR { continue }
                    guard count > 0 else { break }
                    capture.append(Data(bytes.prefix(count)), isError: isError)
                }
            }
        }
        let started = Date()
        var timedOut = false
        while child.isRunning {
            if Task.isCancelled { cancel() }
            if let timeout, Date().timeIntervalSince(started) > timeout { timedOut = true; cancel() }
            let value = capture.snapshot()
            output = String((value.stderr + "\n" + value.stdout).suffix(32_768))
            update?(value)
            // Cancellation is handled explicitly above; do not spin on a cancelled Task.sleep.
            await Task.detached { try? await Task.sleep(for: .milliseconds(100)) }.value
        }
        // A descendant may hold a pipe after its leader exits. Only touch the
        // process group whose ID is the child we spawned, never the app's group.
        if ownsGroup { kill(-pid, SIGTERM) }
        let drained = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(returning: readers.wait(timeout: .now() + 2) == .success)
            }
        }
        // Descendants can also close both pipes before outliving their leader.
        // Remember group ownership while the leader is alive, then clean up
        // independently of whether pipe drainage succeeded.
        if ownsGroup && kill(-pid, 0) == 0 { kill(-pid, SIGKILL) }
        if !drained {
            try? out.fileHandleForReading.close(); try? err.fileHandleForReading.close()
        }
        stopTask?.cancel(); stopTask = nil
        let result = capture.snapshot(status: child.terminationStatus)
        output = String((result.stderr + "\n" + result.stdout).suffix(32_768))
        update?(result)
        running = false; process = nil; processID = nil
        if timedOut { throw AppFailure.message("Operation timed out. Check the runtime and its output.") }
        if cancelled || Task.isCancelled { throw CancellationError() }
        guard drained else { throw AppFailure.message("Runtime exited with open output pipes; its child group was stopped.") }
        return result
    }

    public func cancel() {
        guard let child = process, child.isRunning, !cancelled else { return }
        cancelled = true
        let pid = child.processIdentifier
        let ownsGroup = getpgid(pid) == pid && pid != getpgrp()
        if ownsGroup { kill(-pid, SIGTERM) } else { child.terminate() }
        stopTask = Task {
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            if ownsGroup { kill(-pid, SIGKILL) }
            else if child.isRunning { kill(pid, SIGKILL) }
        }
    }
}
