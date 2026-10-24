import Foundation
import Combine

public struct StorageSnapshot: Codable, Sendable {
    public let totalBytes: Int64
    public let availableBytes: Int64
    public let volumeName: String
    public let requestedPath: String
    public let resolvedPath: String
    public var valid: Bool { totalBytes > 0 && availableBytes >= 0 && availableBytes <= totalBytes }
    public var usedBytes: Int64 { valid ? totalBytes - availableBytes : 0 }
    public init(totalBytes: Int64, availableBytes: Int64, volumeName: String, requestedPath: String, resolvedPath: String) {
        self.totalBytes = totalBytes; self.availableBytes = availableBytes
        self.volumeName = volumeName; self.requestedPath = requestedPath; self.resolvedPath = resolvedPath
    }
    // Called only in the cancellable metadata helper, never on the UI thread.
    public static func read(path: String) throws -> Self {
        guard path.hasPrefix("/"), !path.contains("\0") else { throw AppFailure.message("Choose an absolute model directory.") }
        var url = URL(fileURLWithPath: path).standardizedFileURL
        while !FileManager.default.fileExists(atPath: url.path) {
            if (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil {
                throw AppFailure.message("The model volume link is unavailable.")
            }
            // A missing mounted volume must not silently report the startup disk.
            if url.path == "/" || url.deletingLastPathComponent().path == "/Volumes" {
                throw AppFailure.message("The model volume is unavailable.")
            }
            url.deleteLastPathComponent()
        }
        let values = try url.resourceValues(forKeys: [.volumeTotalCapacityKey, .volumeAvailableCapacityKey, .volumeNameKey])
        guard let total = values.volumeTotalCapacity, let free = values.volumeAvailableCapacity else {
            throw AppFailure.message("This volume does not report capacity.")
        }
        let snapshot = Self(totalBytes: Int64(total), availableBytes: Int64(free),
                            volumeName: values.volumeName ?? "Model volume", requestedPath: path, resolvedPath: url.path)
        guard snapshot.valid else { throw AppFailure.message("The volume reported invalid capacity.") }
        return snapshot
    }
}

@MainActor
public final class StorageMonitor: ObservableObject {
    @Published public private(set) var snapshot: StorageSnapshot?
    @Published public private(set) var updatedAt: Date?
    @Published public private(set) var error: String?
    @Published public private(set) var refreshing = false
    private var generation = 0
    private var job: ProcessJob?
    private let helper: URL
    private let timeout: TimeInterval
    public init(helper: URL? = nil, timeout: TimeInterval = 3) {
        self.helper = helper ?? Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("AXEngineModelScanner")
        self.timeout = timeout
    }
    public func refresh(path: String) async {
        generation += 1
        let identity = generation, previous = job
        previous?.cancel()
        if snapshot?.requestedPath != path { snapshot = nil; updatedAt = nil }
        error = nil; refreshing = true
        while previous?.running == true { await Task.detached { try? await Task.sleep(for: .milliseconds(50)) }.value }
        guard identity == generation, !Task.isCancelled else {
            if identity == generation { refreshing = false }
            return
        }
        let current = ProcessJob(); job = current
        defer { if identity == generation { refreshing = false; job = nil } }
        do {
            let result = try await current.run(executable: helper, arguments: ["--volume", path], timeout: timeout)
            guard identity == generation, !Task.isCancelled else { return }
            guard result.status == 0, !result.truncated,
                  let value = try? JSONDecoder().decode(StorageSnapshot.self, from: Data(result.stdout.utf8)),
                  value.valid, value.requestedPath == path else {
                throw AppFailure.message("Storage information is unavailable. Check that the model volume is connected.")
            }
            snapshot = value; updatedAt = Date()
        } catch {
            guard identity == generation, !Task.isCancelled else { return }
            snapshot = nil; updatedAt = nil
            self.error = error.localizedDescription
        }
    }
    public func shutdown() async {
        generation += 1; job?.cancel()
        while job?.running == true { await Task.detached { try? await Task.sleep(for: .milliseconds(50)) }.value }
        job = nil; refreshing = false
    }
}

public struct ModelInventory: Sendable {
    public let complete: Int
    public let incomplete: Int
    public let unchecked: Int
    public var total: Int { complete + incomplete + unchecked }
    public init(library: [LocalModel], checked: Set<String>) {
        complete = library.filter { checked.contains($0.path) && $0.ready }.count
        incomplete = library.filter { checked.contains($0.path) && !$0.ready }.count
        unchecked = library.filter { !checked.contains($0.path) }.count
    }
}
