import Foundation
import Testing
import AxEngine
@testable import AXEngineAppCore

@Test func legacySettingsKeepTheirGenerationBehavior() throws {
    let old = Data(#"{"executable":"","modelsDirectory":"/tmp/models","port":31418,"maxTokens":1024,"temperature":0.7}"#.utf8)
    let settings = try JSONDecoder().decode(AppSettings.self, from: old)
    #expect(settings.effectiveProfile == .custom)
    #expect(!settings.useRuntimeSampling)
    #expect(settings.maxTokens == 1024)
    #expect(settings.requestTemperature(supportsTemperature: true) == 0.7)
    #expect(try JSONDecoder().decode(AppSettings.self, from: JSONEncoder().encode(settings)) == settings)
    #expect(AppSettings().effectiveProfile == .chatbot)
}

@Test func profilesApplyWithoutChangingRuntimeOrOverwritingCustomTemperature() throws {
    var settings = AppSettings()
    settings.port = 32000; settings.modelsDirectory = "/tmp/profile-models"; settings.temperature = 0.85
    let saved = settings
    for profile in [GenerationProfile.coding, .chatbot, .media] {
        settings.apply(profile)
        #expect(settings.effectiveProfile == profile)
        #expect(settings.maxTokens == profile.outputLimit)
        #expect(settings.useRuntimeSampling)
        #expect(settings.temperature == 0.85)
        #expect(!settings.hasRuntimeChanges(comparedTo: saved))
        try settings.validate()
        #expect(try JSONDecoder().decode(AppSettings.self, from: JSONEncoder().encode(settings)) == settings)
    }
    settings.maxTokens = 5000
    #expect(settings.effectiveProfile == .custom)
    settings.apply(.custom)
    #expect(settings.maxTokens == 5000)
    settings.apply(.coding)
    settings.useRuntimeSampling = false
    #expect(settings.effectiveProfile == .custom)
    settings.port += 1
    #expect(settings.hasRuntimeChanges(comparedTo: saved))
}

@Test func profileRequestsOmitAutomaticAndUnsupportedSampling() throws {
    var settings = AppSettings()
    let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase
    func requestJSON(_ settings: AppSettings, temperatureSupported: Bool = true) throws -> [String: Any] {
        let request = settings.chatRequest(model: "test", messages: [.init(role: "user", content: "test")],
                                          supportsTemperature: temperatureSupported)
        return try #require(JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any])
    }
    for profile in [GenerationProfile.coding, .chatbot, .media] {
        settings.apply(profile)
        let request = try requestJSON(settings)
        #expect(request["max_tokens"] as? Int == profile.outputLimit)
        #expect(request["temperature"] == nil)
        #expect(request["top_p"] == nil)
        #expect(request["top_k"] == nil)
        #expect(request["multimodal_inputs"] == nil)
        let exported = try #require(JSONSerialization.jsonObject(with: Data(settings.requestDefaultsJSON().utf8)) as? [String: Any])
        #expect(Set(exported.keys) == ["max_tokens"])
        #expect(exported["max_tokens"] as? Int == profile.outputLimit)
    }
    settings.useRuntimeSampling = false; settings.temperature = 0.6
    #expect(try requestJSON(settings)["temperature"] as? Double == 0.6)
    #expect(try requestJSON(settings, temperatureSupported: false)["temperature"] == nil)
    let exported = try #require(JSONSerialization.jsonObject(with: Data(settings.requestDefaultsJSON().utf8)) as? [String: Any])
    #expect(Set(exported.keys) == ["max_tokens", "temperature"])
    settings.maxTokens = 0
    #expect(throws: AppFailure.self) { try settings.requestDefaultsJSON() }
}
