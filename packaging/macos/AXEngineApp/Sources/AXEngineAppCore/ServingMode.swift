import Foundation

/// The speculative-decode posture an app-owned server is started with.
/// Mirrors the runtime's `--speculation-profile` selector (ADR-022): it bundles
/// the MTP gate and n-gram policy for one workload. It never changes which
/// tokens are committed, only how aggressively drafts are proposed, so it is
/// safe to switch between restarts without affecting output quality.
public enum ServingMode: String, Codable, CaseIterable, Sendable {
    case auto, coding, agentic, chatbot

    public var title: String {
        switch self {
        case .auto: "Automatic"
        case .coding: "Coding"
        case .agentic: "Agentic"
        case .chatbot: "Chatbot"
        }
    }

    public var summary: String {
        switch self {
        case .auto: "Follows each request's temperature: throughput defaults for low-temperature requests, diversity-preserving drafting for sampled chat."
        case .coding: "Low-temperature code and explanations. Drafts more aggressively on Qwen models for higher decode throughput."
        case .agentic: "Tool calls, JSON and reasoning at low temperature. The most aggressive draft gate on Qwen models."
        case .chatbot: "Higher-temperature conversation. Conservative drafting that protects reply diversity."
        }
    }

    /// Short text for pills and menus.
    public var badge: String { title + " mode" }

    /// The value accepted by `ax-engine-server --speculation-profile`.
    public var runtimeValue: String { rawValue }

    /// Arguments appended to `ax-engine serve …`. Automatic adds nothing, so
    /// existing installs keep the runtime default and any explicit
    /// `AX_MLX_SPECULATION_PROFILE` in the environment still applies.
    public var serverArguments: [String] {
        self == .auto ? [] : ["--", "--speculation-profile", runtimeValue]
    }

    /// The equivalent command-line flag, for users who run the CLI themselves.
    public var commandLineHint: String {
        self == .auto ? "ax-engine serve <model>" : "ax-engine serve <model> -- --speculation-profile \(runtimeValue)"
    }
}
