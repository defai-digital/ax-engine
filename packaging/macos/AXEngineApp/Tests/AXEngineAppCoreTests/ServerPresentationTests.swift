import Foundation
import Testing
@testable import AXEngineAppCore

@Test func statusKindKeepsGreenForOwnedServingOnly() {
    #expect(ServerStatusKind.from(state: "Running", owned: true, ready: true) == .serving)
    #expect(ServerStatusKind.from(state: "Attached", owned: false, ready: true) == .external)
    #expect(ServerStatusKind.from(state: "Running", owned: true, ready: false) == .transition)
    for state in ["Starting", "Stopping", "Connecting"] {
        #expect(ServerStatusKind.from(state: state, owned: true, ready: false) == .transition)
    }
    #expect(ServerStatusKind.from(state: "Connection degraded", owned: true, ready: false) == .degraded)
    #expect(ServerStatusKind.from(state: "Not ready", owned: true, ready: false) == .degraded)
    #expect(ServerStatusKind.from(state: "Failed", owned: false, ready: false) == .failed)
    #expect(ServerStatusKind.from(state: "Stopped", owned: false, ready: false) == .stopped)
}

@Test func counterRateUsesOnlyConsecutiveMonotonicPolls() {
    var rate = CounterRate(key: "tokens")
    let start = Date(timeIntervalSince1970: 1_000)
    #expect(rate.observe(["tokens": 100], at: start) == nil)
    #expect(rate.observe(["tokens": 300], at: start.addingTimeInterval(2)) == 100)
    // A counter reset (server restart) never reports a negative rate.
    #expect(rate.observe(["tokens": 10], at: start.addingTimeInterval(4)) == nil)
    #expect(rate.observe(["tokens": 30], at: start.addingTimeInterval(6)) == 10)
    // A long gap is not averaged over.
    #expect(rate.observe(["tokens": 90], at: start.addingTimeInterval(30)) == nil)
    // A missing counter clears the window.
    #expect(rate.observe([:], at: start.addingTimeInterval(32)) == nil)
    #expect(rate.observe(["tokens": 100], at: start.addingTimeInterval(34)) == nil)
    #expect(rate.observe(["tokens": .nan], at: start.addingTimeInterval(36)) == nil)
}

@Test func snippetsNeverEmbedAKey() {
    let url = "http://127.0.0.1:31418/v1"
    for snippet in ClientSnippet.allCases {
        let open = snippet.text(apiURL: url, model: "qwen", requiresKey: false)
        let keyed = snippet.text(apiURL: url, model: "qwen", requiresKey: true)
        #expect(open.contains(url) && keyed.contains(url))
        #expect(!open.contains("AX_ENGINE_API_KEY"))
        #expect(keyed.contains("AX_ENGINE_API_KEY"))
        #expect(!keyed.contains("Bearer axe_"))
    }
    #expect(ClientSnippet.curl.text(apiURL: url, model: "", requiresKey: false).contains("\"model\": \"default\""))
    #expect(ClientSnippet.python.text(apiURL: url, model: "qwen", requiresKey: false).contains("model=\"qwen\""))
}

@Test func memoryReportOnlyUsesReportedValues() {
    let empty = ModelMemoryReport(metrics: [:])
    #expect(empty.total == nil && empty.prefixCache == nil)
    let report = ModelMemoryReport(metrics: ["ax_engine_memory_mlx_active_bytes": 10, "ax_engine_memory_mlx_cache_bytes": 5,
                                             "ax_engine_model_memory_prefix_cache_payload_bytes": -1])
    #expect(report.total == 15)
    #expect(report.prefixCache == nil)
    #expect(ModelMemoryReport(metrics: ["ax_engine_memory_mlx_cache_bytes": 4]).total == 4)
}

@Test func paletteMatchesEveryWordCaseInsensitively() {
    let entries = [PaletteEntry(id: "a", title: "Serve Qwen 3.8 27B", section: "Models", keywords: ["AutomatosX/qwen"]),
                   PaletteEntry(id: "b", title: "Go to System", section: "Navigate")]
    #expect(PaletteEntry.filter(entries, query: "").count == 2)
    #expect(PaletteEntry.filter(entries, query: "serve 27b").map(\.id) == ["a"])
    #expect(PaletteEntry.filter(entries, query: "automatosx").map(\.id) == ["a"])
    #expect(PaletteEntry.filter(entries, query: "SYSTEM").map(\.id) == ["b"])
    #expect(PaletteEntry.filter(entries, query: "serve system").isEmpty)
}

@Test func tokenRateFormatting() {
    #expect(TokenRateFormat.short(nil) == "—")
    #expect(TokenRateFormat.short(.nan) == "—")
    #expect(TokenRateFormat.short(47.34) == "47.3 tok/s")
    #expect(TokenRateFormat.short(312.6) == "313 tok/s")
    #expect(TokenRateFormat.short(1540) == "1.5k tok/s")
}
