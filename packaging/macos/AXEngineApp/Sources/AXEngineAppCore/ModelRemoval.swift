import Foundation
import Darwin

/// What deleting a library row removes. Every target is fenced to a folder the
/// app or the AX Engine downloader owns; arbitrary paths are never deletable.
public enum ModelRemovalTarget: Equatable, Sendable {
    /// A model folder directly under the app models folder (has model-manifest.json).
    case managedFolder(URL)
    /// A whole `models--AutomatosX--<name>` repository in the Hugging Face cache.
    /// The library shows one snapshot per repository, so removal frees every
    /// cached revision of that repository.
    case cacheRepository(URL)

    public var url: URL {
        switch self { case .managedFolder(let url), .cacheRepository(let url): url }
    }
    public var isCacheRepository: Bool { if case .cacheRepository = self { true } else { false } }
}

/// An interrupted native download left on disk. These stage folders are hidden
/// and survive app restarts; a stale one can also block a later download of a
/// different revision into the same destination.
public struct PartialDownload: Codable, Identifiable, Sendable, Equatable {
    public let path: String
    public let label: String
    public var sizeBytes: Int64?
    public var id: String { path }
    public init(path: String, label: String, sizeBytes: Int64?) {
        self.path = path; self.label = label; self.sizeBytes = sizeBytes
    }
}

public enum ModelFiles {
    static let stageSuffix = ".ax-download"
    static let stateFile = ".ax-native-state.json"
    static let cacheRepositoryPrefix = "models--AutomatosX--"

    /// Mirrors `ax_engine_download::cache_root` so the app and runtime agree.
    public static func hubCacheRoot(environment: [String: String] = ProcessInfo.processInfo.environment,
                                    home: URL = FileManager.default.homeDirectoryForCurrentUser) -> URL {
        if let path = environment["HF_HUB_CACHE"] ?? environment["HUGGINGFACE_HUB_CACHE"], path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        if let path = environment["HF_HOME"], path.hasPrefix("/") {
            return URL(fileURLWithPath: path).appendingPathComponent("hub")
        }
        if let path = environment["XDG_CACHE_HOME"], path.hasPrefix("/") {
            return URL(fileURLWithPath: path).appendingPathComponent("huggingface/hub")
        }
        return home.appendingPathComponent(".cache/huggingface/hub")
    }

    static func isLink(_ url: URL) -> Bool {
        (try? FileManager.default.destinationOfSymbolicLink(atPath: url.path)) != nil
    }

    static func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }

    /// Resolve which folder removing `path` would delete, or nil when the row
    /// is not app-removable (imported, configured, linked or foreign folders).
    public static func removalTarget(path: String, modelsRoot: String, cacheRoot: URL) -> ModelRemovalTarget? {
        if ModelStore.canRemove(path: path, root: modelsRoot) {
            return .managedFolder(URL(fileURLWithPath: path).standardizedFileURL)
        }
        let snapshot = URL(fileURLWithPath: path).standardizedFileURL
        let snapshots = snapshot.deletingLastPathComponent()
        let repository = snapshots.deletingLastPathComponent()
        let cache = cacheRoot.resolvingSymlinksInPath().standardizedFileURL
        guard snapshots.lastPathComponent == "snapshots",
              repository.lastPathComponent.hasPrefix(cacheRepositoryPrefix),
              repository.lastPathComponent.count > cacheRepositoryPrefix.count,
              repository.deletingLastPathComponent() == cache,
              !isLink(repository), !isLink(snapshots), isDirectory(repository),
              isDirectory(snapshot) else { return nil }
        return .cacheRepository(repository)
    }

    /// Allocated bytes under `url`, without following links. Nil when unreadable.
    public static func allocatedSize(of url: URL) -> Int64? {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey, .isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys,
                                                              options: [], errorHandler: { _, _ in true }) else { return nil }
        var total: Int64 = 0
        for case let item as URL in enumerator {
            guard let values = try? item.resourceValues(forKeys: Set(keys)), values.isRegularFile == true else { continue }
            total += Int64(values.totalFileAllocatedSize ?? values.fileAllocatedSize ?? 0)
        }
        return total
    }

    static func isStageName(_ name: String) -> Bool {
        name.hasPrefix(".") && name.hasSuffix(stageSuffix) && name.count > 1 + stageSuffix.count
    }

    /// Interrupted download stages in the app models folder and in AutomatosX
    /// repositories of the Hugging Face cache.
    public static func partialDownloads(modelsRoot: String, cacheRoot: URL) -> [PartialDownload] {
        let fm = FileManager.default
        var found: [PartialDownload] = []
        func collect(in directory: URL, label: (String) -> String) {
            guard let names = try? fm.contentsOfDirectory(atPath: directory.path) else { return }
            for name in names.sorted() where isStageName(name) {
                let stage = directory.appendingPathComponent(name)
                guard !isLink(stage), isDirectory(stage),
                      fm.fileExists(atPath: stage.appendingPathComponent(stateFile).path) else { continue }
                let base = String(name.dropFirst().dropLast(stageSuffix.count))
                found.append(PartialDownload(path: stage.path, label: label(base), sizeBytes: allocatedSize(of: stage)))
            }
        }
        let root = URL(fileURLWithPath: modelsRoot).resolvingSymlinksInPath()
        collect(in: root) { $0.replacingOccurrences(of: "--", with: "/") }
        let cache = cacheRoot.resolvingSymlinksInPath()
        for repo in ((try? fm.contentsOfDirectory(atPath: cache.path)) ?? []).sorted()
        where repo.hasPrefix(cacheRepositoryPrefix) {
            let repository = cache.appendingPathComponent(repo)
            guard !isLink(repository) else { continue }
            let name = repo.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
            collect(in: repository.appendingPathComponent(".ax-downloads")) { name + " · " + String($0.prefix(8)) }
        }
        return found
    }

    /// True when `path` is a stage folder that `partialDownloads` would report.
    public static func isPartialDownload(path: String, modelsRoot: String, cacheRoot: URL) -> Bool {
        let stage = URL(fileURLWithPath: path).standardizedFileURL
        let parent = stage.deletingLastPathComponent()
        guard isStageName(stage.lastPathComponent), !isLink(stage), isDirectory(stage),
              FileManager.default.fileExists(atPath: stage.appendingPathComponent(stateFile).path) else { return false }
        if parent == URL(fileURLWithPath: modelsRoot).resolvingSymlinksInPath().standardizedFileURL { return true }
        let repository = parent.deletingLastPathComponent()
        return parent.lastPathComponent == ".ax-downloads"
            && repository.lastPathComponent.hasPrefix(cacheRepositoryPrefix) && !isLink(repository)
            && repository.deletingLastPathComponent() == cacheRoot.resolvingSymlinksInPath().standardizedFileURL
    }

    /// Run `body` only while holding the downloader's exclusive lock for `stage`,
    /// so a stage owned by a running CLI or app download is never deleted.
    static func withStageLock<T>(_ stage: URL, _ body: () throws -> T) throws -> T {
        let lock = stage.deletingLastPathComponent().appendingPathComponent(stage.lastPathComponent + ".lock")
        guard !isLink(lock) else { throw AppFailure.message("The download lock is not a regular file.") }
        let descriptor = open(lock.path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else { throw AppFailure.message("Could not check whether this download is still running.") }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            throw AppFailure.message("This download is still running in AX Engine. Cancel it first.")
        }
        defer { flock(descriptor, LOCK_UN) }
        return try body()
    }

    /// Delete a fenced folder. It is first renamed to a hidden sibling so that
    /// other tools stop seeing it at once, even if a large delete is slow.
    static func delete(_ url: URL) throws {
        let fm = FileManager.default
        let hidden = url.deletingLastPathComponent()
            .appendingPathComponent("." + url.lastPathComponent + ".ax-removing-" + UUID().uuidString.prefix(8))
        try fm.moveItem(at: url, to: hidden)
        try fm.removeItem(at: hidden)
    }

    public static func removeCacheRepository(_ repository: URL) throws {
        try delete(repository)
    }

    public static func discardPartialDownload(_ stage: URL) throws {
        try withStageLock(stage) { try delete(stage) }
    }
}

public enum PortProbe {
    /// Whether a loopback listener could bind `port` now. Uses SO_REUSEADDR like
    /// the server, so a recently stopped server's TIME_WAIT is not a conflict.
    public static func isAvailable(port: Int) -> Bool {
        guard (1...65535).contains(port) else { return false }
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return true }
        defer { close(descriptor) }
        var reuse: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bound == 0
    }
}
