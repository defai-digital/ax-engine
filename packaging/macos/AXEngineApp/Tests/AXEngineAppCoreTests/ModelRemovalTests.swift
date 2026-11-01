import Foundation
import Testing
import Darwin
@testable import AXEngineAppCore

private func temporaryRoot() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("ax-removal-test-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url.resolvingSymlinksInPath()
}

private func write(_ text: String, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(text.utf8).write(to: url)
}

/// A cache snapshot laid out as the native downloader writes it.
@discardableResult
private func cacheSnapshot(_ cache: URL, repo: String, revision: String = "0123456789abcdef") throws -> URL {
    let snapshot = cache.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
        .appendingPathComponent("snapshots").appendingPathComponent(revision)
    try write("{}", to: snapshot.appendingPathComponent("config.json"))
    try write(String(repeating: "w", count: 4096), to: snapshot.appendingPathComponent("model.safetensors"))
    return snapshot
}

private func emptyScanner(in root: URL) throws -> URL {
    let url = root.appendingPathComponent("empty-scanner")
    try write("#!/bin/sh\necho '{\"complete\":true}'\n", to: url)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    return url
}

@Test func cacheSnapshotsOfAutomatosXReposResolveToTheirRepository() throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let cache = root.appendingPathComponent("hub"), models = root.appendingPathComponent("models")
    let own = try cacheSnapshot(cache, repo: "AutomatosX/Pack")
    let foreign = try cacheSnapshot(cache, repo: "community/Pack")
    let target = ModelFiles.removalTarget(path: own.path, modelsRoot: models.path, cacheRoot: cache)
    #expect(target == .cacheRepository(cache.appendingPathComponent("models--AutomatosX--Pack")))
    #expect(ModelFiles.removalTarget(path: foreign.path, modelsRoot: models.path, cacheRoot: cache) == nil)
    // A snapshot under some other cache root is not fenced to this one.
    let elsewhere = try cacheSnapshot(root.appendingPathComponent("other-hub"), repo: "AutomatosX/Pack")
    #expect(ModelFiles.removalTarget(path: elsewhere.path, modelsRoot: models.path, cacheRoot: cache) == nil)
    // A missing snapshot or a non-snapshot folder is never a target.
    #expect(ModelFiles.removalTarget(path: own.deletingLastPathComponent().appendingPathComponent("gone").path,
                                     modelsRoot: models.path, cacheRoot: cache) == nil)
    #expect(ModelFiles.removalTarget(path: cache.appendingPathComponent("models--AutomatosX--Pack").path,
                                     modelsRoot: models.path, cacheRoot: cache) == nil)
}

@Test func linkedCacheRepositoriesAreNotRemovable() throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let cache = root.appendingPathComponent("hub"), outside = root.appendingPathComponent("outside")
    let real = try cacheSnapshot(outside, repo: "AutomatosX/Pack")
    try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
    let link = cache.appendingPathComponent("models--AutomatosX--Pack")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real.deletingLastPathComponent().deletingLastPathComponent())
    let unresolved = link.appendingPathComponent("snapshots").appendingPathComponent(real.lastPathComponent)
    #expect(ModelFiles.removalTarget(path: unresolved.path, modelsRoot: root.path, cacheRoot: cache) == nil)
    #expect(ModelFiles.removalTarget(path: real.path, modelsRoot: root.path, cacheRoot: cache) == nil)
}

@Test func managedFoldersStillResolveAsManaged() throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let folder = root.appendingPathComponent("models/AutomatosX--Pack--rev")
    try write("{}", to: folder.appendingPathComponent("model-manifest.json"))
    let target = ModelFiles.removalTarget(path: folder.path, modelsRoot: root.appendingPathComponent("models").path,
                                          cacheRoot: root.appendingPathComponent("hub"))
    #expect(target?.isCacheRepository == false)
    #expect(target?.url.path == folder.path)
}

@Test func scannerReportsTheSizeThatRemovalFrees() throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let cache = root.appendingPathComponent("hub")
    let older = try cacheSnapshot(cache, repo: "AutomatosX/Pack", revision: "1111111111111111")
    try cacheSnapshot(cache, repo: "AutomatosX/Pack", revision: "2222222222222222")
    try FileManager.default.setAttributes([.modificationDate: Date.distantPast], ofItemAtPath: older.path)
    let rows = ModelStore.scanLibrary(root: root.appendingPathComponent("models").path, imported: [],
                                      environment: ["HF_HUB_CACHE": cache.path])
    #expect(rows.count == 1)
    let repository = ModelFiles.allocatedSize(of: cache.appendingPathComponent("models--AutomatosX--Pack"))
    let snapshot = ModelFiles.allocatedSize(of: older)
    #expect(rows.first?.sizeBytes == repository)
    #expect((repository ?? 0) > (snapshot ?? 0))
}

@Test @MainActor func removingACacheRepositoryDeletesEveryRevisionAndNothingElse() async throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let cache = root.appendingPathComponent("hub")
    let snapshot = try cacheSnapshot(cache, repo: "AutomatosX/Pack")
    try cacheSnapshot(cache, repo: "AutomatosX/Pack", revision: "fedcba9876543210")
    let sibling = try cacheSnapshot(cache, repo: "AutomatosX/Other")
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"),
                           scannerURL: try emptyScanner(in: root), scanEnvironment: ["HF_HUB_CACHE": cache.path])
    var settings = AppSettings(); settings.modelsDirectory = root.appendingPathComponent("models").path
    let row = LocalModel(path: snapshot.path, ready: true, managed: false, source: "Hugging Face cache")
    #expect(store.removalTarget(for: row, settings: settings)?.isCacheRepository == true)
    #expect(await store.remove(row, settings: settings))
    #expect(store.error == nil)
    #expect(!FileManager.default.fileExists(atPath: cache.appendingPathComponent("models--AutomatosX--Pack").path))
    #expect(FileManager.default.fileExists(atPath: sibling.path))
    #expect(try FileManager.default.contentsOfDirectory(atPath: cache.path) == ["models--AutomatosX--Other"])
}

@Test @MainActor func nonAutomatosXRowsAreRefused() async throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let cache = root.appendingPathComponent("hub")
    let snapshot = try cacheSnapshot(cache, repo: "community/Pack")
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"),
                           scannerURL: try emptyScanner(in: root), scanEnvironment: ["HF_HUB_CACHE": cache.path])
    var settings = AppSettings(); settings.modelsDirectory = root.appendingPathComponent("models").path
    let row = LocalModel(path: snapshot.path, ready: true, managed: false)
    #expect(!(await store.remove(row, settings: settings)))
    #expect(store.error != nil)
    #expect(FileManager.default.fileExists(atPath: snapshot.path))
}

@Test func partialDownloadsAreFoundInModelsFolderAndCache() throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let models = root.appendingPathComponent("models"), cache = root.appendingPathComponent("hub")
    let appStage = models.appendingPathComponent(".AutomatosX--Pack--rev.ax-download")
    try write("{}", to: appStage.appendingPathComponent(".ax-native-state.json"))
    try write("partial", to: appStage.appendingPathComponent("model-00001.safetensors"))
    let cacheStage = cache.appendingPathComponent("models--AutomatosX--Pack/.ax-downloads/.0123456789abcdef.ax-download")
    try write("{}", to: cacheStage.appendingPathComponent(".ax-native-state.json"))
    // Not a downloader stage: no state file, or a foreign repository.
    try FileManager.default.createDirectory(at: models.appendingPathComponent(".unrelated.ax-download"), withIntermediateDirectories: true)
    try write("{}", to: cache.appendingPathComponent("models--community--Pack/.ax-downloads/.abc.ax-download/.ax-native-state.json"))
    let found = ModelFiles.partialDownloads(modelsRoot: models.path, cacheRoot: cache)
    #expect(found.map(\.path) == [appStage.path, cacheStage.path])
    #expect(found.map(\.label) == ["AutomatosX/Pack/rev", "AutomatosX/Pack · 01234567"])
    #expect(found.allSatisfy { ModelFiles.isPartialDownload(path: $0.path, modelsRoot: models.path, cacheRoot: cache) })
    #expect(!ModelFiles.isPartialDownload(path: models.appendingPathComponent(".unrelated.ax-download").path,
                                          modelsRoot: models.path, cacheRoot: cache))
}

@Test func discardingAPartialDownloadRespectsTheDownloaderLock() throws {
    let root = try temporaryRoot(); defer { try? FileManager.default.removeItem(at: root) }
    let stage = root.appendingPathComponent(".AutomatosX--Pack--rev.ax-download")
    try write("{}", to: stage.appendingPathComponent(".ax-native-state.json"))
    let lock = open(root.appendingPathComponent(".AutomatosX--Pack--rev.ax-download.lock").path, O_RDWR | O_CREAT, 0o644)
    #expect(lock >= 0)
    #expect(flock(lock, LOCK_EX | LOCK_NB) == 0)
    #expect(throws: AppFailure.self) { try ModelFiles.discardPartialDownload(stage) }
    #expect(FileManager.default.fileExists(atPath: stage.path))
    flock(lock, LOCK_UN); close(lock)
    try ModelFiles.discardPartialDownload(stage)
    #expect(!FileManager.default.fileExists(atPath: stage.path))
}

@Test func portProbeSeesAnActiveListener() throws {
    let descriptor = socket(AF_INET, SOCK_STREAM, 0)
    #expect(descriptor >= 0)
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = 0
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    var size = socklen_t(MemoryLayout<sockaddr_in>.size)
    let bound = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { pointer in
            bind(descriptor, pointer, size) == 0 && listen(descriptor, 1) == 0 && getsockname(descriptor, pointer, &size) == 0
        }
    }
    #expect(bound)
    let port = Int(UInt16(bigEndian: address.sin_port))
    #expect(!PortProbe.isAvailable(port: port))
    close(descriptor)
    #expect(PortProbe.isAvailable(port: port))
    #expect(!PortProbe.isAvailable(port: 0))
}

@Test @MainActor func startRefusesAnOccupiedPortBeforeLaunching() throws {
    let runtime = RuntimeController(portAvailable: { _ in false })
    var settings = AppSettings(); settings.executable = "/bin/sh"
    runtime.start(model: "/models/AutomatosX--Pack", settings: settings)
    #expect(runtime.state == "Failed")
    #expect(!runtime.owned && !runtime.busy && !runtime.job.running)
    #expect(runtime.error?.contains("already in use") == true)
}

@Test func runtimeExitMessageUsesTheLastDiagnosticLine() {
    #expect(RuntimeController.lastDiagnostic("loading\nerror: model is not supported\n\n") == "error: model is not supported")
    #expect(RuntimeController.lastDiagnostic(" \n") == nil)
    #expect(RuntimeController.lastDiagnostic(String(repeating: "x", count: 500))?.count == 401)
}
