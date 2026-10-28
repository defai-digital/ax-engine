import Foundation
import Testing
import AxEngine
@testable import AXEngineAppCore

private func performance(_ fields: [String: Any]) throws -> ChatPerformance? {
    let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
    let report = try decoder.decode(OpenAiInferencePerformance.self,
        from: JSONSerialization.data(withJSONObject: fields))
    return ChatPerformance(report)
}

@Test func measuredChatRatesUseNativeCountsAndMicroseconds() throws {
    let metrics = try #require(try performance([
        "metrics_version": 1, "prefill_tokens": 200, "prefill_time_us": 400_000,
        "decode_tokens": 80, "decode_time_us": 2_000_000
    ]))
    #expect(metrics.prefillTokensPerSecond == 500)
    #expect(metrics.decodeTokensPerSecond == 40)
    let restored = try JSONDecoder().decode(ChatPerformance.self, from: JSONEncoder().encode(metrics))
    #expect(restored == metrics)
}

@Test func unavailableOrInvalidChatMeasurementsNeverBecomeZeroOrInfiniteRates() throws {
    for duration in [0.0, -1.0, 1e-310] {
        let metrics = try #require(try performance([
            "metrics_version": 1, "prefill_tokens": 200, "prefill_time_us": duration,
            "decode_tokens": 80, "decode_time_us": duration
        ]))
        #expect(metrics.prefillTokensPerSecond == nil)
        #expect(metrics.decodeTokensPerSecond == nil)
    }
    let partial = try #require(try performance(["metrics_version": 1, "prefill_time_us": 100]))
    #expect(partial.prefillTokensPerSecond == nil)
    let cached = try #require(try performance(["metrics_version": 1, "prefill_tokens": 0, "prefill_time_us": 100]))
    #expect(cached.promptFullyCached)
    #expect(cached.prefillTokensPerSecond == nil)
    let negative = try #require(try performance(["metrics_version": 1, "decode_tokens": -1, "decode_time_us": 100]))
    #expect(negative.decodeTokensPerSecond == nil)
    #expect(try performance(["metrics_version": 2]) == nil)
}

@Test func oldConversationsWithoutSpeedMeasurementsStillDecode() throws {
    let data = Data("""
    {"id":"D9E6D3D3-1072-4FA1-92D9-DF2124A22EFE","role":"assistant","content":"Earlier reply","reasoning":"","state":"Complete"}
    """.utf8)
    let message = try JSONDecoder().decode(ChatMessage.self, from: data)
    #expect(message.performance == nil)
    #expect(message.content == "Earlier reply")
}

@Test @MainActor func interruptedHistoryDoesNotRestoreAFinalSpeedReading() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    var message = ChatMessage(role: "assistant", content: "Partial reply", state: "Streaming")
    message.performance = try performance(["metrics_version": 1, "decode_tokens": 80, "decode_time_us": 2_000_000])
    try JSONEncoder().encode([message]).write(to: file)
    let restored = ChatStore(storage: file)
    #expect(restored.messages.first?.state == "Interrupted")
    #expect(restored.messages.first?.performance == nil)
}

@Test @MainActor func chatSSEPersistsPerTurnRatesAndClearsFailedOrLegacyMeasurements() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try #require(Bundle.module.url(forResource: "chat_performance", withExtension: "py", subdirectory: "Fixtures"))
    let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
    let portFile = root.appendingPathComponent("port")
    process.arguments = [fixture.path, portFile.path]
    process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
    try process.run()
    defer { if process.isRunning { process.terminate() }; process.waitUntilExit() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while !FileManager.default.fileExists(atPath: portFile.path), ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
    let port = try #require(Int(try String(contentsOf: portFile, encoding: .utf8)))
    let connection = EngineConnection(port: port)
    let storage = root.appendingPathComponent("chat.json")
    let chat = ChatStore(storage: storage)
    for prompt in ["measured", "legacy", "truncated", "stopped"] {
        chat.send(prompt, model: "speed-fixture", connection: connection, settings: AppSettings())
        let finish = ContinuousClock.now.advanced(by: .seconds(5))
        if prompt == "stopped" {
            while chat.messages.last?.performance == nil, chat.generating, ContinuousClock.now < finish {
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(chat.generating)
            await chat.stop()
        }
        while chat.generating, ContinuousClock.now < finish {
            try await Task.sleep(for: .milliseconds(10))
        }
        if chat.generating { await chat.stop(); Issue.record("Fixture chat timed out") }
        let message = try #require(chat.messages.last)
        if prompt == "measured" {
            #expect(message.state == "Complete")
            #expect(message.performance?.prefillTokensPerSecond == 500)
            #expect(message.performance?.decodeTokensPerSecond == 40)
        } else {
            #expect(message.performance == nil)
            #expect(message.state == (prompt == "legacy" ? "Complete" : prompt == "stopped" ? "Stopped" : "Failed"))
        }
    }
    let restored = ChatStore(storage: storage)
    #expect(restored.messages[1].performance?.decodeTokensPerSecond == 40)
    #expect(restored.messages.last?.performance == nil)
    restored.clear()
    #expect(restored.messages.isEmpty)
}

@Test(arguments: ["oversized", "exact", "advertised"]) @MainActor
func serviceResponseLimitStopsAnOversizedBodyBeforeEOF(mode: String) async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let fixture = try #require(Bundle.module.url(forResource: "chat_performance", withExtension: "py", subdirectory: "Fixtures"))
    let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
    let portFile = root.appendingPathComponent("port")
    process.arguments = [fixture.path, portFile.path]
    process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
    try process.run()
    defer { if process.isRunning { process.terminate() }; process.waitUntilExit() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while !FileManager.default.fileExists(atPath: portFile.path), ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
    let port = try #require(Int(try String(contentsOf: portFile, encoding: .utf8)))
    let started = ContinuousClock.now
    do {
        let data = try await EngineConnection(port: port).data(mode, timeout: 15)
        #expect(mode == "exact")
        #expect(data.count == 4 * 1024 * 1024)
    } catch {
        #expect(mode != "exact")
        #expect(error.localizedDescription.contains("display limit"))
    }
    #expect(ContinuousClock.now - started < .seconds(6))
}
