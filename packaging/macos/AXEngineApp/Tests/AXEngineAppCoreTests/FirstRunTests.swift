import Testing
@testable import AXEngineAppCore

private func entry(_ name: String, _ size: Int64?) -> CatalogModel {
    var model = CatalogModel(alias: name, repoId: "AutomatosX/" + name, revision: "abc", mtpIncluded: true)
    model.approxSizeBytes = size
    return model
}

@Test func firstRunOrdersWithinMemorySmallestFirst() {
    let gib: Int64 = 1 << 30
    let catalog = [entry("Huge", 200 * gib), entry("Mid", 30 * gib), entry("Unknown", nil), entry("Small", 10 * gib), entry("Big", 70 * gib)]
    #expect(FirstRun.ordered(catalog, memoryBytes: 64 * gib).map(\.alias) == ["Small", "Mid", "Unknown", "Big", "Huge"])
    #expect(FirstRun.ordered([], memoryBytes: 64 * gib).isEmpty)
}

@Test func firstRunIsOfferedOnlyAfterACompleteEmptyScan() {
    func offer(completed: Bool = false, attempted: Bool = true, scanning: Bool = false, count: Int = 0,
               failed: Bool = false, active: Bool = false) -> Bool {
        FirstRun.shouldOffer(completed: completed, discoveryAttempted: attempted, scanning: scanning,
                             libraryCount: count, scanFailed: failed, serverActive: active)
    }
    #expect(offer())
    #expect(!offer(completed: true))
    #expect(!offer(attempted: false))
    #expect(!offer(scanning: true))
    #expect(!offer(count: 2))
    #expect(!offer(failed: true))
    #expect(!offer(active: true))
}
