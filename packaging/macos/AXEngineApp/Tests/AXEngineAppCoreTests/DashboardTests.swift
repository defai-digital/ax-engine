import Foundation
import Testing
@testable import AXEngineAppCore

private func scanFixture() throws -> (URL, URL) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("ax-library-test-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let fixture = root.appendingPathComponent("scanner")
    let script = #"""
#!/usr/bin/python3
import json, sys, time, signal
request = json.loads(sys.argv[1]); root = request['root']; mode = root.rsplit('/', 1)[-1]
if mode == 'stubborn': signal.signal(signal.SIGTERM, signal.SIG_IGN)
def emit(value):
    print(json.dumps(value), flush=True)
if mode == 'malformed':
    print('not-json', flush=True)
elif mode == 'warning':
    emit({'warning': 'Storage is unavailable.', 'complete': False})
elif mode != 'empty':
    emit({'model': {'path': root + '/model', 'ready': True, 'managed': False, 'source': 'Fixture'}, 'complete': False})
    time.sleep(10 if mode in ('slow', 'stubborn') else 0.5)
emit({'complete': True})
"""#
    try Data(script.utf8).write(to: fixture)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fixture.path)
    return (root, fixture)
}

@MainActor
private func until(_ condition: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(5)
    while !condition() && Date() < deadline { try await Task.sleep(for: .milliseconds(30)) }
    #expect(condition())
}

@Test @MainActor func libraryPublishesRowsBeforeWorkerCompletesAndKeepsPartialResults() async throws {
    let (root, fixture) = try scanFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"), scannerURL: fixture, scanTimeout: 5)
    var settings = AppSettings(); settings.modelsDirectory = root.appendingPathComponent("slow").path
    let scan = Task { await store.scan(settings: settings) }
    try await until { !store.library.isEmpty }
    #expect(store.scanning)
    #expect(store.checkedPaths.count == 1)
    await scan.value
    #expect(!store.scanning)
    #expect(store.library.count == 1)
    #expect(store.libraryIssue?.contains("incomplete") == true)
    #expect(store.lastScanAt == nil)
}

@Test @MainActor func librarySupersedesAndCancelsOnlyOwnedScan() async throws {
    let (root, fixture) = try scanFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"), scannerURL: fixture)
    var old = AppSettings(); old.modelsDirectory = root.appendingPathComponent("slow").path
    let first = Task { await store.scan(settings: old) }
    try await until { !store.library.isEmpty }
    var new = AppSettings(); new.modelsDirectory = root.appendingPathComponent("new").path
    await store.scan(settings: new)
    await first.value
    #expect(store.library.map(\.path) == [new.modelsDirectory + "/model"])
    #expect(store.checkedPaths == [new.modelsDirectory + "/model"])
    #expect(store.lastScanAt != nil)
    let last = Task { await store.scan(settings: old) }
    try await until { store.checkedPaths.contains(old.modelsDirectory + "/model") }
    store.cancelScan(); await last.value
    #expect(!store.scanning)
    #expect(store.libraryIssue?.contains("stopped") == true)
    #expect(store.library.contains(where: { $0.path == new.modelsDirectory + "/model" }))
}

@Test @MainActor func malformedOrUnavailableScanCannotEraseKnownRows() async throws {
    let (root, fixture) = try scanFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"), scannerURL: fixture)
    var settings = AppSettings(); settings.modelsDirectory = root.appendingPathComponent("good").path
    await store.scan(settings: settings)
    let known = store.library
    for mode in ["malformed", "warning"] {
        settings.modelsDirectory = root.appendingPathComponent(mode).path
        await store.scan(settings: settings)
        #expect(store.library == known)
        #expect(store.checkedPaths.isEmpty)
        #expect(store.libraryIssue != nil)
    }
    settings.modelsDirectory = root.appendingPathComponent("empty").path
    await store.scan(settings: settings)
    #expect(store.library.isEmpty)
    #expect(store.libraryIssue == nil)
}

@Test @MainActor func realLibraryHelperReadsMetadataWithoutLoadingWeights() async throws {
    let (root, _) = try scanFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("models/Local")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: model.appendingPathComponent("config.json"))
    try Data("fixture".utf8).write(to: model.appendingPathComponent("model.safetensors"))
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"), scannerURL: testScannerURL(), scanEnvironment: ["HF_HUB_CACHE": root.appendingPathComponent("empty").path, "AX_ENGINE_MODELS_DIR": root.appendingPathComponent("models").path])
    var settings = AppSettings(); settings.modelsDirectory = root.appendingPathComponent("models").path
    await store.scan(settings: settings)
    #expect(store.libraryIssue == nil)
    #expect(store.library.count == 1)
    #expect(store.library.first?.ready == true)
    #expect(store.library.first?.source == "App folder")
    #expect(store.lastScanAt != nil)
    await store.shutdown()
}

@Test func catalogProfilesHaveDistinctIdentity() {
    let first = CatalogModel(alias: "first", repoId: "Owner/Repo", revision: "a", mtpIncluded: false)
    let second = CatalogModel(alias: "second", repoId: "Owner/Repo", revision: "b", mtpIncluded: false)
    #expect(first.id != second.id)
}

@Test @MainActor func cancellingSupersedingScanDoesNotLaunchPendingWorker() async throws {
    let (root, fixture) = try scanFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"), scannerURL: fixture)
    var old = AppSettings(); old.modelsDirectory = root.appendingPathComponent("stubborn").path
    let first = Task { await store.scan(settings: old) }
    try await until { !store.library.isEmpty }
    var new = AppSettings(); new.modelsDirectory = root.appendingPathComponent("new").path
    let pending = Task { await store.scan(settings: new) }
    try await Task.sleep(for: .milliseconds(100))
    store.cancelScan()
    await pending.value; await first.value
    #expect(!store.scanning)
    #expect(store.library.map(\.path) == [old.modelsDirectory + "/model"])
    #expect(store.lastScanAt == nil)
    #expect(store.libraryIssue?.contains("stopped") == true)
}

@Test func linkedRepositoryIsDeferredWithoutLosingLocalRows() throws {
    let (root, _) = try scanFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let local = root.appendingPathComponent("models/Local")
    let repository = root.appendingPathComponent("external/models--Tests--Linked")
    let snapshot = repository.appendingPathComponent("snapshots/revision")
    let cache = root.appendingPathComponent("cache")
    for folder in [local, snapshot, cache] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
    for folder in [local, snapshot] {
        try Data("{}".utf8).write(to: folder.appendingPathComponent("config.json"))
        try Data("fixture".utf8).write(to: folder.appendingPathComponent("model.safetensors"))
    }
    try FileManager.default.createSymbolicLink(at: cache.appendingPathComponent("models--Tests--Linked"), withDestinationURL: repository)
    var emitted: [String] = []
    let rows = ModelStore.scanLibrary(root: local.deletingLastPathComponent().path, imported: [],
        environment: ["HF_HUB_CACHE": cache.path], emit: { emitted.append($0.path) })
    #expect(emitted == [local.path, snapshot.path])
    #expect(rows.count == 2)
}
