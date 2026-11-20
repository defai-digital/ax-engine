import Testing
@testable import AXEngineAppCore

private func side(total: Double?, first: Double?, draft: Double?, phase: CompareSide.Phase = .done) -> CompareSide {
    var value = CompareSide(path: "/m", title: "M")
    value.phase = phase; value.totalSeconds = total; value.firstTextSeconds = first; value.draftAcceptedPercent = draft
    return value
}

@Test func compareSummaryWaitsForBothRuns() {
    #expect(CompareSummary(a: side(total: 4, first: 0.3, draft: nil), b: side(total: 5, first: 0.4, draft: nil, phase: .running)) == nil)
    #expect(CompareSummary(a: side(total: 4, first: 0.3, draft: nil, phase: .failed("x")), b: side(total: 5, first: 0.4, draft: nil)) == nil)
}

@Test func compareSummaryNamesTheSoonerSideAndDeltas() throws {
    let summary = try #require(CompareSummary(a: side(total: 4.8, first: 0.312, draft: 79), b: side(total: 6.8, first: 0.486, draft: 64)))
    #expect(summary.finished?.winner == .a)
    #expect(abs((summary.finished?.seconds ?? 0) - 2.0) < 1e-9)
    #expect(abs((summary.firstTextDeltaMs ?? 0) - 174) < 1e-6)
    #expect(summary.draftDeltaPoints == 15)
    #expect(summary.decodeRatio == nil) // no runtime measurement on either side
    let tie = try #require(CompareSummary(a: side(total: 5, first: nil, draft: nil), b: side(total: 5.01, first: nil, draft: nil)))
    #expect(tie.finished?.winner == .tie)
    #expect(tie.draftDeltaPoints == nil)
}

@Test func tokenCurveNeedsTheRuntimeTokenCount() {
    var value = side(total: 3, first: 1, draft: nil)
    value.samples = [.init(seconds: 1, characters: 10), .init(seconds: 2, characters: 40)]
    #expect(value.tokenCurve.isEmpty) // no runtime measurement: no curve, not characters
    #expect(CompareSide().tokenCurve.isEmpty)
}

@Test func draftAcceptanceUsesTheCounterDeltaOnly() {
    let before = ["ax_engine_mtp_draft_tokens_total": 100.0, "ax_engine_mtp_accepted_tokens_total": 60.0]
    let after = ["ax_engine_mtp_draft_tokens_total": 300.0, "ax_engine_mtp_accepted_tokens_total": 210.0]
    #expect(CompareStore.draftAcceptance(before: before, after: after) == 75)
    #expect(CompareStore.draftAcceptance(before: before, after: before) == nil) // no drafts this run
    #expect(CompareStore.draftAcceptance(before: after, after: before) == nil) // counter reset
    #expect(CompareStore.draftAcceptance(before: nil, after: after) == nil)
}
