import Foundation
import AxEngine

/// Persisted per-turn native evaluation measurements, independent of UI/network pacing.
public struct ChatPerformance: Codable, Equatable, Sendable {
    public let prefillTokens: Int?
    public let prefillTimeUs: Double?
    public let decodeTokens: Int?
    public let decodeTimeUs: Double?

    public init?(_ report: OpenAiInferencePerformance) {
        guard report.metricsVersion == 1 else { return nil }
        prefillTokens = report.prefillTokens.flatMap { $0 >= 0 ? $0 : nil }
        decodeTokens = report.decodeTokens.flatMap { $0 >= 0 ? $0 : nil }
        prefillTimeUs = Self.validDuration(report.prefillTimeUs)
        decodeTimeUs = Self.validDuration(report.decodeTimeUs)
    }

    public var prefillTokensPerSecond: Double? { Self.rate(prefillTokens, prefillTimeUs) }
    public var decodeTokensPerSecond: Double? { Self.rate(decodeTokens, decodeTimeUs) }
    public var promptFullyCached: Bool { prefillTokens == 0 && prefillTimeUs != nil }

    private static func validDuration(_ value: Double?) -> Double? {
        value.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
    }

    private static func rate(_ tokens: Int?, _ duration: Double?) -> Double? {
        guard let tokens, tokens > 0, let duration = validDuration(duration) else { return nil }
        let value = Double(tokens) / duration * 1_000_000
        return value.isFinite && value > 0 ? value : nil
    }
}
