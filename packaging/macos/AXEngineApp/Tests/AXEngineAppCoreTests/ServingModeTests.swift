import Foundation
import Testing
@testable import AXEngineAppCore

@Test func servingModeDefaultsToAutomaticAndRoundTrips() throws {
    let old = Data(#"{"executable":"","modelsDirectory":"/tmp/models","port":31418,"maxTokens":1024,"temperature":0.7}"#.utf8)
    let legacy = try JSONDecoder().decode(AppSettings.self, from: old)
    #expect(legacy.servingMode == .auto && legacy.concurrentRequests == 4)
    #expect(AppSettings().servingMode == .auto && AppSettings().concurrentRequests == 4)
    var settings = AppSettings()
    let saved = settings
    for mode in ServingMode.allCases {
        settings.servingMode = mode
        try settings.validate()
        // Switching the mode needs a server restart, not a rescan or a runtime lock.
        #expect(!settings.hasRuntimeChanges(comparedTo: saved))
        let decoded = try JSONDecoder().decode(AppSettings.self, from: JSONEncoder().encode(settings))
        #expect(decoded == settings && decoded.servingMode == mode)
    }
    for count in [0, 9] {
        settings.concurrentRequests = count
        #expect(throws: AppFailure.self) { try settings.validate() }
    }
    settings.concurrentRequests = 8
    try settings.validate()
    #expect(!settings.hasRuntimeChanges(comparedTo: saved))
    #expect(Set(ServingMode.allCases.map(\.title)).count == ServingMode.allCases.count)
    #expect(ServingMode.allCases.map(\.runtimeValue) == ["auto", "coding", "agentic", "chatbot"])
}

@Test func servingModeIsForwardedAfterTheServeSeparator() {
    #expect(ServingMode.auto.serverOptions.isEmpty)
    #expect(ServingMode.coding.serverOptions == ["--speculation-profile", "coding"])
    var settings = AppSettings(); settings.port = 31500
    let plain = RuntimeController.launchArguments(model: "/models/qwen", settings: settings)
    // The default cap is always forwarded; Automatic adds no profile flag.
    #expect(plain == ["serve", "/models/qwen", "--offline", "--host", "127.0.0.1", "--port", "31500", "--", "--max-concurrent-requests", "4"])
    settings.servingMode = .agentic; settings.concurrentRequests = 2
    let tuned = RuntimeController.launchArguments(model: "/models/qwen", settings: settings)
    #expect(tuned.prefix(7).elementsEqual(plain.prefix(7)))
    #expect(tuned.suffix(5).elementsEqual(["--", "--speculation-profile", "agentic", "--max-concurrent-requests", "2"]))
    #expect(tuned.filter { $0 == "--" }.count == 1)
    let separator = tuned.firstIndex(of: "--") ?? -1
    let port = tuned.firstIndex(of: "--port") ?? Int.max
    #expect(separator > port)
    #expect(ServingMode.auto.commandLineHint == "ax-engine serve <model>")
    #expect(ServingMode.chatbot.commandLineHint.hasSuffix("-- --speculation-profile chatbot"))
}

@Test func restartTextsNameWhatChanges() {
    #expect(ServerGate.restartTitle(modelChanged: false, modeChanged: false, mode: .auto) == "Restart Server")
    #expect(ServerGate.restartTitle(modelChanged: true, modeChanged: false, mode: .auto) == "Restart with Selected Model")
    #expect(ServerGate.restartTitle(modelChanged: false, modeChanged: true, mode: .coding) == "Restart in Coding Mode")
    #expect(ServerGate.restartTitle(modelChanged: true, modeChanged: true, mode: .chatbot) == "Restart with Selected Model in Chatbot Mode")
    #expect(ServerGate.pendingRestartNotice(modelChanged: false, modeChanged: false, running: .auto, selected: .auto) == nil)
    let modeOnly = ServerGate.pendingRestartNotice(modelChanged: false, modeChanged: true, running: .auto, selected: .coding) ?? ""
    #expect(modeOnly.contains("Coding mode is saved") && modeOnly.contains("Automatic mode"))
    let both = ServerGate.pendingRestartNotice(modelChanged: true, modeChanged: true, running: .chatbot, selected: .agentic) ?? ""
    #expect(both.contains("model and serving mode") && both.contains("Chatbot mode"))
    #expect(ServerGate.pendingRestartNotice(modelChanged: true, modeChanged: false, running: .auto, selected: .auto)?.contains("selected model differs") == true)
    #expect(ServerGate.restartTitle(modelChanged: false, modeChanged: false, capChanged: true, mode: .auto) == "Restart with New Request Cap")
    #expect(ServerGate.restartTitle(modelChanged: true, modeChanged: false, capChanged: true, mode: .auto) == "Restart with Selected Model and New Request Cap")
    let capOnly = ServerGate.pendingRestartNotice(modelChanged: false, modeChanged: false, running: .auto, selected: .auto, runningCap: 4, selectedCap: 6) ?? ""
    #expect(capOnly.contains("Up to 6 concurrent requests") && capOnly.contains("admits 4"))
    #expect(ServerGate.pendingRestartNotice(modelChanged: false, modeChanged: false, running: .auto, selected: .auto, runningCap: 4, selectedCap: 4) == nil)
    let modeAndCap = ServerGate.pendingRestartNotice(modelChanged: false, modeChanged: true, running: .auto, selected: .coding, runningCap: 4, selectedCap: 2) ?? ""
    #expect(modeAndCap.contains("Coding mode is saved") && modeAndCap.contains("from 4 to 2"))
}
