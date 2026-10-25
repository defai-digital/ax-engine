import Foundation
import Testing
import Darwin
import AppKit
@testable import AXEngineAppCore

private func temporary() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("ax-engine-gui-test-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@Test func settingsExcludeSecretsAndValidate() throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("settings.json")
    var settings = AppSettings(); settings.port = 34567
    try settings.save(to: url)
    #expect(try AppSettings.load(from: url) == settings)
    let data = String(decoding: try Data(contentsOf: url), as: UTF8.self)
    #expect(!data.contains("apiKey"))
    settings.port = 0
    #expect(throws: AppFailure.self) { try settings.validate() }
}

@Test func malformedSettingsRemainVisible() throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("settings.json")
    try Data("broken".utf8).write(to: url)
    #expect(throws: (any Error).self) { try AppSettings.load(from: url) }
}

@Test func progressNeedsTerminalSuccess() {
    let progress = DownloadProgress.parse("{\"event\":\"progress\",\"done\":100,\"total\":100,\"file\":\"Ready\"}\n")
    #expect(progress.fraction == 1)
    #expect(!progress.terminalReady)
    let failure = DownloadProgress.parse("{\"schema_version\":\"ax.download_model.v1\",\"status\":\"download_failed\",\"errors\":[\"disk full\"]}\n")
    #expect(!failure.terminalReady)
    #expect(failure.detail == "disk full")
}

@Test func progressHandlesNoiseAndZeroTotal() {
    let value = DownloadProgress.parse("not json\n{\"event\":\"progress\",\"done\":8,\"total\":0}\n{\"schema_version\":\"ax.download_model.v1\",\"status\":\"ready\",\"dest\":\"/model\"}\n")
    #expect(value.fraction == nil)
    #expect(value.terminalReady)
    #expect(value.destination == "/model")
}

@Test func catalogPreservesRuntimePin() throws {
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    let model = try decoder.decode(CatalogModel.self, from: Data(#"{"alias":"Model","repo_id":"Owner/Repo","revision":"abc123","mtp_included":true}"#.utf8))
    #expect(model.target == "Owner/Repo@abc123")
    #expect(model.folderName == "Owner--Repo--abc123")
}

@Test func metricLabelsDoNotDoubleCount() {
    let parsed = EngineConnection.parseMetrics("# comment\nax_engine_tokens 12\nax_engine_tokens{model=\"a\"} 10\ninvalid NaN\n")
    #expect(parsed["ax_engine_tokens"] == 12)
    #expect(parsed["invalid"] == nil)
    #expect(parsed.count == 1)
    #expect(EngineConnection.requestID("chatcmpl-123") == 123)
    #expect(EngineConnection.requestID("chatcmpl-../1") == nil)
    #expect(EngineConnection.requestID("cmpl-123") == nil)
}

@Test func removeCannotEscapeManagedRoot() throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let models = root.appendingPathComponent("models"), outside = root.appendingPathComponent("outside")
    try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: outside.appendingPathComponent("model-manifest.json"))
    let link = models.appendingPathComponent("escape")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)
    #expect(!ModelStore.canRemove(path: link.path, root: models.path))
    #expect(!ModelStore.canRemove(path: outside.path, root: models.path))
    #expect(!ModelStore.canRemove(path: models.path, root: models.path))
}

@Test func boundedLogsCapEntriesAndMessageSize() async {
    let logs = BoundedLogStore(capacity: 2)
    await logs.append("first", stream: .system)
    await logs.append("second", stream: .system)
    await logs.append(String(repeating: "x", count: 9000), stream: .stderr)
    let snapshot = await logs.snapshot()
    #expect(snapshot.count == 2)
    #expect(snapshot.first?.message == "second")
    #expect(snapshot.last?.message.count == 4096)
}

@Test @MainActor func processDrainsBothPipesAndKeepsArgumentBoundaries() async throws {
    let job = ProcessJob()
    let result = try await job.run(executable: URL(fileURLWithPath: "/usr/bin/python3"),
        arguments: ["-c", "import sys; print(sys.argv[1]); sys.stderr.write('e'*200000); print('o'*200000)", "hello; no shell"], timeout: 10)
    #expect(result.status == 0)
    #expect(result.stdout.hasPrefix("hello; no shell\n"))
    #expect(result.stderr.count == 200000)
    #expect(!job.running)
}

@Test @MainActor func processBoundsLargeOutput() async throws {
    let result = try await ProcessJob().run(executable: URL(fileURLWithPath: "/usr/bin/python3"),
        arguments: ["-c", "print('x'*5000000)"], timeout: 15)
    #expect(result.status == 0)
    #expect(result.truncated)
    #expect(result.stdout.utf8.count <= 4 * 1024 * 1024)
}

@Test @MainActor func processCancellationIsBounded() async throws {
    let job = ProcessJob()
    let task = Task { try await job.run(executable: URL(fileURLWithPath: "/usr/bin/python3"), arguments: ["-c", "import time; time.sleep(60)"]) }
    while !job.running { try await Task.sleep(for: .milliseconds(20)) }
    let started = Date(); job.cancel()
    do { _ = try await task.value; Issue.record("Cancelled job returned success") } catch is CancellationError {} catch { throw error }
    #expect(Date().timeIntervalSince(started) < 6)
    #expect(!job.running)
}

@Test @MainActor func processTimeoutIsAnError() async throws {
    await #expect(throws: AppFailure.self) {
        _ = try await ProcessJob().run(executable: URL(fileURLWithPath: "/usr/bin/python3"),
            arguments: ["-c", "import time; time.sleep(60)"], timeout: 0.2)
    }
}

@Test @MainActor func modelStoreShutsDownCatalogProcess() async throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let executable = root.appendingPathComponent("slow-runtime")
    try Data("#!/usr/bin/python3\nimport time\ntime.sleep(60)\n".utf8).write(to: executable)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
    var settings = AppSettings(); settings.executable = executable.path
    settings.modelsDirectory = root.appendingPathComponent("models").path
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"), scannerURL: testScannerURL(), scanEnvironment: ["HF_HUB_CACHE": root.appendingPathComponent("empty-cache").path])
    let task = Task { await store.refresh(settings: settings) }
    while !store.refreshing { try await Task.sleep(for: .milliseconds(20)) }
    let started = Date()
    await store.shutdown()
    await task.value
    #expect(!store.refreshing)
    #expect(Date().timeIntervalSince(started) < 6)
}

@Test(arguments: [false, true]) @MainActor func exitedLeaderCannotLeaveDescendant(closePipes: Bool) async throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let pidFile = root.appendingPathComponent("descendant.pid")
    let script = """
    import os, signal, sys, time
    pid = os.fork()
    if pid:
        time.sleep(0.1)
        os._exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(sys.argv[1], 'w') as f:
        f.write(str(os.getpid()))
    if sys.argv[2] == 'close':
        os.close(1)
        os.close(2)
    time.sleep(60)
    """
    let started = Date()
    let job = ProcessJob()
    if closePipes {
        let result = try await job.run(executable: URL(fileURLWithPath: "/usr/bin/python3"),
            arguments: ["-c", script, pidFile.path, "close"], timeout: 10)
        #expect(result.status == 0)
    } else {
        await #expect(throws: AppFailure.self) {
            _ = try await job.run(executable: URL(fileURLWithPath: "/usr/bin/python3"),
                arguments: ["-c", script, pidFile.path, "keep"], timeout: 10)
        }
    }
    let pid = try #require(Int32(String(contentsOf: pidFile, encoding: .utf8)))
    defer { if kill(pid, 0) == 0 { kill(pid, SIGKILL) } }
    let deadline = Date().addingTimeInterval(2)
    while kill(pid, 0) == 0 && Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
    #expect(kill(pid, 0) != 0)
    #expect(Date().timeIntervalSince(started) < 6)
}

@Test @MainActor func fixtureRuntimeLifecycleChatAndDownloads() async throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let fixture = root.appendingPathComponent("runtime")
    try FileManager.default.copyItem(at: #require(Bundle.module.url(forResource: "runtime", withExtension: "py", subdirectory: "Fixtures")), to: fixture)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fixture.path)
    // Obtain a short-lived test port. Runtime readiness is authenticated, so a
    // bind race fails visibly rather than attaching to another listener.
    let portResult = try await ProcessJob().run(executable: URL(fileURLWithPath: "/usr/bin/python3"),
        arguments: ["-c", "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])"])
    var settings = AppSettings(); settings.executable = fixture.path
    settings.port = try #require(Int(portResult.stdout.trimmingCharacters(in: .whitespacesAndNewlines)))
    settings.modelsDirectory = root.appendingPathComponent("models").path
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"), scannerURL: testScannerURL(), scanEnvironment: ["HF_HUB_CACHE": root.appendingPathComponent("empty-cache").path])
    await store.refresh(settings: settings)
    #expect(store.catalog.count == 1)
    store.enqueue(try #require(store.catalog.first), settings: settings)
    while store.downloading { try await Task.sleep(for: .milliseconds(50)) }
    #expect(store.downloads.first?.state == "Ready")
    #expect(store.library.contains(where: { $0.managed && $0.ready }))
    for name in ["Failure", "Missing"] {
        store.enqueue(CatalogModel(alias: name, repoId: "AutomatosX/" + name, revision: "abc123", mtpIncluded: false), settings: settings)
    }
    while store.downloading { try await Task.sleep(for: .milliseconds(50)) }
    #expect(store.downloads.suffix(2).allSatisfy { $0.state == "Failed" })
    let runtime = RuntimeController()
    runtime.start(model: "fixture", settings: settings)
    let deadline = Date().addingTimeInterval(15)
    while !runtime.ready && Date() < deadline && runtime.owned { try await Task.sleep(for: .milliseconds(50)) }
    #expect(runtime.ready)
    #expect(runtime.startedAt != nil)
    #expect(runtime.lastSuccessfulCheck != nil)
    #expect(runtime.selectedModel == "fixture")
    let client = try #require(runtime.connection)
    let attached = RuntimeController()
    await attached.attach(port: settings.port, apiKey: "wrong")
    #expect(!attached.ready)
    #expect(attached.error?.contains("API key") == true)
    #expect(runtime.job.running)
    // The generated runtime key reaches an independent client only through the copy action.
    let pasteboard = NSPasteboard.withUniqueName()
    let clipboard = SensitiveClipboard(pasteboard: pasteboard)
    defer { clipboard.clearIfOwned(); pasteboard.releaseGlobally() }
    #expect(client.copyAPIKey(to: clipboard))
    let copiedKey = pasteboard.string(forType: .string) ?? ""
    await attached.attach(port: settings.port, apiKey: copiedKey)
    #expect(attached.ready)
    #expect(attached.connection?.apiURL == client.apiURL)
    clipboard.clearIfOwned()
    #expect(pasteboard.string(forType: .string) == nil)
    let logsExcludeKey = !runtime.logs.contains(copiedKey)
    #expect(logsExcludeKey)
    var redirectRejected = false
    do { _ = try await client.data("redirect") } catch { redirectRejected = error.localizedDescription.contains("HTTP 302") }
    #expect(redirectRejected)
    var cancellationRejected = false
    do { try await client.cancel(requestID: 999) } catch { cancellationRejected = error.localizedDescription.contains("API key") }
    #expect(cancellationRejected)
    let chat = ChatStore(storage: root.appendingPathComponent("chat.json"))
    for (prompt, expected) in [("hello", "Complete"), ("truncate", "Failed"), ("error", "Failed"), ("redirect", "Failed")] {
        chat.clear(); chat.send(prompt, model: "fixture", connection: client, settings: settings)
        while chat.generating { try await Task.sleep(for: .milliseconds(30)) }
        #expect(chat.messages.last?.state == expected)
    }
    // The last turn failed; its orphan user prompt must not be replayed.
    chat.send("history", model: "fixture", connection: client, settings: settings)
    while chat.generating { try await Task.sleep(for: .milliseconds(30)) }
    #expect(chat.messages.last?.content.contains("error") == false)
    #expect(chat.messages.last?.content.contains("redirect") == false)
    #expect(chat.messages.last?.content.contains("history") == true)
    let restored = ChatStore(storage: root.appendingPathComponent("chat.json"))
    let retainedCount = restored.messages.count
    restored.send("keep my history", model: "different", connection: client, settings: settings)
    #expect(!restored.generating)
    #expect(restored.messages.count == retainedCount)
    #expect(restored.error?.contains("changing models") == true)
    restored.retry(model: "different", connection: client, settings: settings, supportsTemperature: true)
    #expect(restored.messages.count == retainedCount)
    // Prove profile defaults reach the HTTP request, including omission semantics.
    for profile in [GenerationProfile.coding, .chatbot, .media] {
        var profiled = settings; profiled.apply(profile)
        chat.clear(); chat.send("parameters", model: "fixture", connection: client, settings: profiled)
        while chat.generating { try await Task.sleep(for: .milliseconds(30)) }
        let text = try #require(chat.messages.last?.content)
        let payload = try #require(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        #expect(payload["max_tokens"] as? Int == profile.outputLimit)
        #expect(payload["temperature"] == nil)
        #expect(runtime.ready && runtime.job.running)
    }
    chat.clear(); chat.send("slow", model: "fixture", connection: client, settings: settings)
    try await Task.sleep(for: .milliseconds(350)); await chat.stop()
    #expect(!chat.generating)
    #expect(chat.messages.last?.state == "Stopped")
    await attached.stop()
    #expect(runtime.job.running)
    await runtime.stop()
    #expect(!runtime.job.running)
    #expect(!runtime.owned)
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["AX_ENGINE_GUI_REAL_PORT"] != nil, "Requires an explicitly provisioned model test service")) @MainActor func realRuntimeSmokeWhenExplicitlyRequested() async throws {
    guard let raw = ProcessInfo.processInfo.environment["AX_ENGINE_GUI_REAL_PORT"], let port = Int(raw) else { return }
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let runtime = RuntimeController()
    await runtime.attach(port: port, apiKey: ProcessInfo.processInfo.environment["AX_ENGINE_GUI_REAL_API_KEY"] ?? "")
    #expect(runtime.ready)
    let connection = try #require(runtime.connection)
    let model = try #require(runtime.models.first)
    var settings = AppSettings(); settings.maxTokens = 32; settings.temperature = 0
    let chat = ChatStore(storage: root.appendingPathComponent("chat.json"))
    chat.send("Reply with the word ready.", model: model.id, connection: connection, settings: settings,
              supportsTemperature: model.capabilities?.temperature == true)
    let deadline = Date().addingTimeInterval(120)
    while chat.generating && Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
    if chat.generating { await chat.stop(); Issue.record("Real runtime smoke exceeded its deadline") }
    #expect(chat.error == nil)
    #expect(chat.messages.last?.state == "Complete")
    #expect(chat.messages.last?.content.isEmpty == false)
    await runtime.stop()
    // Attached-mode stop is lifecycle-neutral: the external server still answers.
    #expect(try await connection.models().contains(where: { $0.id == model.id }))
}

@Test func runtimeDiscoveryFindsUserAndDevelopmentInstalls() throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let userBin = root.appendingPathComponent(".local/bin/ax-engine")
    try FileManager.default.createDirectory(at: userBin.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("#!/bin/sh\nexit 0\n".utf8).write(to: userBin)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: userBin.path)
    #expect(try RuntimeLocator.find(environment: [:], home: root, systemDirectories: []) == userBin)
    let development = root.appendingPathComponent("checkout-runtime")
    try FileManager.default.copyItem(at: userBin, to: development)
    #expect(try RuntimeLocator.find(development: development.path, environment: [:], home: root, systemDirectories: []) == userBin)
    #expect(throws: AppFailure.self) {
        try RuntimeLocator.find(explicit: root.appendingPathComponent("missing").path,
                                development: development.path, environment: [:], home: root, systemDirectories: [])
    }
    try FileManager.default.removeItem(at: userBin)
    let pathBin = root.appendingPathComponent("ax-engine")
    try FileManager.default.copyItem(at: development, to: pathBin)
    #expect(try RuntimeLocator.find(environment: ["PATH": root.path + ":relative"], home: root, systemDirectories: []) == pathBin)
    #expect(try RuntimeLocator.find(environment: ["AX_ENGINE_BIN": development.path], home: root, systemDirectories: []) == development)
}

@Test func libraryPrefersCompleteSnapshotAndValidatesShardFiles() throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let cache = root.appendingPathComponent("hub")
    let snapshots = cache.appendingPathComponent("models--Tests--Model/snapshots")
    let complete = snapshots.appendingPathComponent("old-complete")
    let metadata = snapshots.appendingPathComponent("new-metadata")
    for folder in [complete, metadata] {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: folder.appendingPathComponent("config.json"))
    }
    let blob = root.appendingPathComponent("blob")
    try Data("fixture".utf8).write(to: blob)
    try FileManager.default.createSymbolicLink(at: complete.appendingPathComponent("model.safetensors"), withDestinationURL: blob)
    try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 10)], ofItemAtPath: complete.path)
    let models = ModelStore.scanLibrary(root: root.appendingPathComponent("managed").path, imported: [], environment: ["HF_HUB_CACHE": cache.path])
    #expect(models.count == 1)
    #expect(models.first?.path == complete.path)
    #expect(models.first?.ready == true)
    #expect(!ModelStore.modelReady(metadata))
    let index = complete.appendingPathComponent("model.safetensors.index.json")
    try Data(#"{"weight_map":{"a":"model.safetensors","b":"missing.safetensors"}}"#.utf8).write(to: index)
    #expect(!ModelStore.modelReady(complete))
    try Data(#"{"weight_map":{"a":"../blob"}}"#.utf8).write(to: index)
    #expect(!ModelStore.modelReady(complete))
    try Data(#"{"weight_map":{"a":"model.safetensors"}}"#.utf8).write(to: index)
    #expect(ModelStore.modelReady(complete))
}

@Test func runtimeDiscoveryUsesCompleteAxCodeManagedInstall() throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let payload = root.appendingPathComponent(".cache/ax-code/ax-engine/bin/7.7.0")
    let marker = root.appendingPathComponent(".local/state/ax-code/ax-engine/install.json")
    try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
    let binary = payload.appendingPathComponent("ax-engine")
    try Data("#!/bin/sh\nexit 0\n".utf8).write(to: binary)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
    try JSONSerialization.data(withJSONObject: ["path": binary.path, "version": "7.7.0"]).write(to: marker)
    #expect(throws: AppFailure.self) { try RuntimeLocator.find(environment: [:], home: root, systemDirectories: []) }
    for name in ["ax-engine-server", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"] {
        try Data("fixture".utf8).write(to: payload.appendingPathComponent(name))
    }
    #expect(try RuntimeLocator.find(environment: [:], home: root, systemDirectories: []) == binary)
    try FileManager.default.removeItem(at: marker)
    #expect(throws: AppFailure.self) { try RuntimeLocator.find(environment: [:], home: root, systemDirectories: []) }
}

func testScannerURL() -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent(".build/debug/AXEngineModelScanner")
}

@Test @MainActor func attachedServiceFailurePreservesOwnershipAndHealthFreshness() async throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let fixture = root.appendingPathComponent("runtime")
    try FileManager.default.copyItem(at: #require(Bundle.module.url(forResource: "runtime", withExtension: "py", subdirectory: "Fixtures")), to: fixture)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fixture.path)
    let portResult = try await ProcessJob().run(executable: URL(fileURLWithPath: "/usr/bin/python3"),
        arguments: ["-c", "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])"])
    let port = try #require(Int(portResult.stdout.trimmingCharacters(in: .whitespacesAndNewlines)))
    let service = ProcessJob()
    defer { service.cancel() }
    let running = Task { try await service.run(executable: fixture, arguments: ["serve", "fixture", "--port", String(port)],
                                                environment: ["AX_ENGINE_API_KEY": "fixture-key"]) }
    let attached = RuntimeController()
    let startDeadline = Date().addingTimeInterval(8)
    while !attached.ready && Date() < startDeadline {
        await attached.attach(port: port, apiKey: "fixture-key")
        if !attached.ready { try await Task.sleep(for: .milliseconds(100)) }
    }
    #expect(attached.ready)
    #expect(!attached.owned)
    let marker = fixture.appendingPathExtension("fail-metrics")
    try Data().write(to: marker)
    let failureDeadline = Date().addingTimeInterval(6)
    while attached.state != "Connection degraded" && Date() < failureDeadline { try await Task.sleep(for: .milliseconds(50)) }
    #expect(attached.state == "Connection degraded")
    #expect(!attached.ready)
    #expect(attached.metrics.isEmpty)
    #expect(attached.history.samples.last?.values.isEmpty == true)
    #expect(attached.lastSuccessfulCheck != nil)
    let lastCheck = attached.lastSuccessfulCheck
    #expect(service.running)
    #expect(!attached.owned)
    try FileManager.default.removeItem(at: marker)
    let recoveryDeadline = Date().addingTimeInterval(6)
    while !attached.ready && Date() < recoveryDeadline { try await Task.sleep(for: .milliseconds(50)) }
    #expect(attached.ready)
    #expect(attached.lastSuccessfulCheck != lastCheck)
    #expect(attached.history.latest("ax_engine_generation_active_streams") == 0)
    await attached.stop()
    #expect(attached.lastSuccessfulCheck == nil)
    #expect(attached.history.latest("ax_engine_generation_active_streams") == nil)
    #expect(service.running)
    service.cancel()
    _ = try? await running.value
}

@Test func runtimeDiscoveryPrefersBundleAndFindsCurlInstallWithoutShellPath() throws {
    let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
    let curl = root.appendingPathComponent(".local/ax-engine/ax-engine")
    let bundle = root.appendingPathComponent("AX Engine.app/Contents/Helpers/AXEngineRuntime.bundle/Contents/MacOS/ax-engine")
    for binary in [curl, bundle] {
        try FileManager.default.createDirectory(at: binary.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("#!/bin/sh\nexit 0\n".utf8).write(to: binary)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binary.path)
    }
    #expect(try RuntimeLocator.find(environment: [:], home: root, systemDirectories: []) == curl)
    #expect(try RuntimeLocator.find(bundled: bundle.path, environment: ["PATH": curl.deletingLastPathComponent().path], home: root, systemDirectories: []) == bundle)
    #expect(try RuntimeLocator.find(explicit: curl.path, bundled: bundle.path, environment: [:], home: root, systemDirectories: []) == curl)
}
