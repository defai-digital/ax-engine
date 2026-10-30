import Darwin
import Foundation
import Testing

struct FixtureLaunchFailure: Error, CustomStringConvertible {
    var description: String
}

/// Starts `chat_performance.py` and waits until it publishes its port.
///
/// The first loopback bind on a cold runner can exit or stall before the port
/// file exists. A failed attempt is stopped and replaced; stderr is kept so
/// the test failure names the interpreter error.
@MainActor
func startChatPerformanceFixture(in root: URL) async throws -> (process: Process, port: Int) {
    let script = try #require(
        Bundle.module.url(forResource: "chat_performance", withExtension: "py", subdirectory: "Fixtures"))
    let portFile = root.appendingPathComponent("port")
    var notes: [String] = []
    for attempt in 1...2 {
        try? FileManager.default.removeItem(at: portFile)
        let stderr = root.appendingPathComponent("fixture-stderr-\(attempt).txt")
        FileManager.default.createFile(atPath: stderr.path, contents: nil)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [script.path, portFile.path]
        // Process closes the descriptors it is given. A dedicated file keeps
        // FileHandle.nullDevice, which is process-wide, intact.
        process.standardOutput = try FileHandle(forWritingTo: stderr)
        process.standardError = try FileHandle(forWritingTo: stderr)
        try process.run()
        let deadline = ContinuousClock.now.advanced(by: .seconds(15))
        while ContinuousClock.now < deadline {
            if let text = try? String(contentsOf: portFile, encoding: .utf8),
               let port = Int(text.trimmingCharacters(in: .whitespacesAndNewlines)) {
                return (process, port)
            }
            if !process.isRunning { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        await stopFixture(process)
        let detail = (try? String(contentsOf: stderr, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        notes.append("attempt \(attempt) status \(process.terminationStatus): \(detail)")
    }
    throw FixtureLaunchFailure(description: notes.joined(separator: "\n"))
}

@MainActor
private func stopFixture(_ process: Process) async {
    if process.isRunning { process.terminate() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while process.isRunning, ContinuousClock.now < deadline {
        try? await Task.sleep(for: .milliseconds(20))
    }
    if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    process.waitUntilExit()
}
