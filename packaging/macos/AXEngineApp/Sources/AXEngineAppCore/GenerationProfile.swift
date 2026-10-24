import Foundation
import AxEngine

public enum GenerationProfile: String, Codable, CaseIterable, Sendable {
    case coding, chatbot, media, custom
    public var title: String {
        switch self {
        case .coding: "Coding"
        case .chatbot: "Chatbot"
        case .media: "Media"
        case .custom: "Custom"
        }
    }
    public var summary: String {
        switch self {
        case .coding: "More room for code, explanations and tests."
        case .chatbot: "A shorter response budget for everyday conversation."
        case .media: "A text response budget for image and document analysis."
        case .custom: "Your own response limit and sampling preference."
        }
    }
    public var outputLimit: Int? {
        switch self {
        case .coding: 8192
        case .chatbot: 2048
        case .media: 4096
        case .custom: nil
        }
    }
}

extension AppSettings {
    public var effectiveProfile: GenerationProfile {
        guard useRuntimeSampling, generationProfile.outputLimit == maxTokens else { return .custom }
        return generationProfile
    }
    public mutating func apply(_ profile: GenerationProfile) {
        generationProfile = profile
        if let limit = profile.outputLimit {
            maxTokens = limit
            useRuntimeSampling = true
        }
    }
    public func requestTemperature(supportsTemperature: Bool) -> Double? {
        !useRuntimeSampling && supportsTemperature ? temperature : nil
    }
    public func chatRequest(model: String, messages: [OpenAiChatMessage], supportsTemperature: Bool) -> OpenAiChatCompletionRequest {
        .init(model: model, messages: messages, maxTokens: maxTokens,
              temperature: requestTemperature(supportsTemperature: supportsTemperature))
    }
    /// Only request parameters: never credentials, paths, system prompts or load settings.
    public func requestDefaultsJSON(supportsTemperature: Bool = true) throws -> String {
        try validate()
        struct Defaults: Encodable {
            let maxTokens: Int
            let temperature: Double?
            enum CodingKeys: String, CodingKey { case maxTokens = "max_tokens", temperature }
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(Defaults(maxTokens: maxTokens,
            temperature: requestTemperature(supportsTemperature: supportsTemperature)))
        return String(decoding: data, as: UTF8.self)
    }
}
