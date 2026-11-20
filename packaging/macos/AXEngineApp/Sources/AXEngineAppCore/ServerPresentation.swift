import Foundation

/// What the toolbar capsule, sidebar and menu bar say about the server.
/// One vocabulary everywhere: green means this app's own server is serving,
/// blue means a service the app attached to (it never stops that one).
public enum ServerStatusKind: String, Sendable, Equatable {
    case serving, external, transition, degraded, failed, stopped

    public static func from(state: String, owned: Bool, ready: Bool) -> Self {
        switch state {
        case "Starting", "Stopping", "Connecting": return .transition
        case "Connection degraded", "Not ready": return .degraded
        case "Failed": return .failed
        case "Running", "Attached":
            guard ready else { return .transition }
            return owned ? .serving : .external
        default: return .stopped
        }
    }

    /// Short label for the capsule; the full state string stays in help text.
    public var label: String {
        switch self {
        case .serving: "Serving"
        case .external: "Attached"
        case .transition: "Working"
        case .degraded: "Degraded"
        case .failed: "Failed"
        case .stopped: "Stopped"
        }
    }
}

/// Per-second rate of a monotonic runtime counter between successful polls.
/// Resets, gaps longer than `maxGap` and missing counters yield nil, never a
/// guessed value.
public struct CounterRate: Sendable {
    public let key: String
    public let maxGap: TimeInterval
    private var previous: (time: Date, value: Double)?
    public init(key: String, maxGap: TimeInterval = 12) { self.key = key; self.maxGap = maxGap }
    public mutating func reset() { previous = nil }
    public mutating func observe(_ metrics: [String: Double], at time: Date = Date()) -> Double? {
        guard let value = metrics[key], value.isFinite, value >= 0 else { reset(); return nil }
        defer { previous = (time, value) }
        guard let previous, time > previous.time, time.timeIntervalSince(previous.time) <= maxGap,
              value >= previous.value else { return nil }
        return (value - previous.value) / time.timeIntervalSince(previous.time)
    }
}

/// Ready-to-paste client setups for an OpenAI-compatible endpoint. Snippets
/// never contain a key: when one is required they reference an environment
/// variable the user fills in.
public enum ClientSnippet: String, CaseIterable, Identifiable, Sendable {
    case curl, python, environment
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .curl: "curl"
        case .python: "Python (OpenAI SDK)"
        case .environment: "Environment variables"
        }
    }
    public func text(apiURL: String, model: String, requiresKey: Bool) -> String {
        let id = model.isEmpty ? "default" : model
        switch self {
        case .curl:
            return "curl \(apiURL)/chat/completions \\\n"
                + "  -H 'Content-Type: application/json' \\\n"
                + (requiresKey ? "  -H \"Authorization: Bearer $AX_ENGINE_API_KEY\" \\\n" : "")
                + "  -d '{\"model\": \"\(id)\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
        case .python:
            return "from openai import OpenAI\n"
                + (requiresKey ? "import os\n\n" : "\n")
                + "client = OpenAI(base_url=\"\(apiURL)\", api_key="
                + (requiresKey ? "os.environ[\"AX_ENGINE_API_KEY\"])\n" : "\"not-needed\")\n")
                + "reply = client.chat.completions.create(\n"
                + "    model=\"\(id)\",\n"
                + "    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n"
                + ")\n"
                + "print(reply.choices[0].message.content)"
        case .environment:
            return "export OPENAI_BASE_URL=\(apiURL)\n"
                + (requiresKey ? "export OPENAI_API_KEY=$AX_ENGINE_API_KEY" : "export OPENAI_API_KEY=not-needed")
        }
    }
}

/// Model memory as the runtime reports it. Parts the runtime does not export
/// stay nil and are not drawn; nothing is estimated.
public struct ModelMemoryReport: Equatable, Sendable {
    public let active: Double?
    public let cache: Double?
    public let prefixCache: Double?
    public let peak: Double?
    public init(metrics: [String: Double]) {
        func bytes(_ key: String) -> Double? { metrics[key].flatMap { $0.isFinite && $0 >= 0 ? $0 : nil } }
        active = bytes("ax_engine_memory_mlx_active_bytes")
        cache = bytes("ax_engine_memory_mlx_cache_bytes")
        prefixCache = bytes("ax_engine_model_memory_prefix_cache_payload_bytes")
        peak = bytes("ax_engine_memory_mlx_peak_bytes")
    }
    public var total: Double? {
        guard active != nil || cache != nil else { return nil }
        return (active ?? 0) + (cache ?? 0)
    }
}

/// One entry of the ⌘K palette. Matching is case-insensitive over the title
/// and keywords; every word of the query must match.
public struct PaletteEntry: Identifiable, Equatable, Sendable {
    public let id: String
    public let title: String
    public let section: String
    public let keywords: [String]
    public init(id: String, title: String, section: String, keywords: [String] = []) {
        self.id = id; self.title = title; self.section = section; self.keywords = keywords
    }
    public func matches(_ query: String) -> Bool {
        let words = query.lowercased().split(whereSeparator: \.isWhitespace)
        guard !words.isEmpty else { return true }
        let haystack = ([title] + keywords).joined(separator: " ").lowercased()
        return words.allSatisfy { haystack.contains($0) }
    }
    public static func filter(_ entries: [PaletteEntry], query: String) -> [PaletteEntry] {
        entries.filter { $0.matches(query) }
    }
}

public enum TokenRateFormat {
    /// "48 tok/s", "1.2k tok/s"; a dash when there is no reading.
    public static func short(_ value: Double?) -> String {
        guard let value, value.isFinite, value >= 0 else { return "—" }
        if value >= 1000 { return String(format: "%.1fk tok/s", value / 1000) }
        return String(format: value >= 100 ? "%.0f tok/s" : "%.1f tok/s", value)
    }
}
