import Foundation
import Testing
@testable import AXEngineAppCore

@Test func metricHistoryBoundsAgeCapacityAndMissingSegments() {
    var history = MetricHistory(capacity: 3, duration: 10)
    let start = Date(timeIntervalSince1970: 100)
    history.append(["x": 1, "invalid": .nan], at: start)
    history.append([:], at: start.addingTimeInterval(1))
    history.append(["x": 2], at: start.addingTimeInterval(2))
    let points = history.points("x", now: start.addingTimeInterval(2))
    #expect(points.count == 2)
    #expect(points[0].segment != points[1].segment)
    #expect(history.latest("invalid", now: start.addingTimeInterval(2)) == nil)
    #expect(history.latest("x", now: start.addingTimeInterval(20)) == nil)
    history.append(["x": 3], at: start.addingTimeInterval(3))
    #expect(history.samples.count == 3)
    history.append(["x": 4], at: start.addingTimeInterval(30))
    #expect(history.samples.count == 1)
    history.append(["x": 5], at: start)
    #expect(history.samples.count == 1)
    #expect(history.latest("x", now: start.addingTimeInterval(-1)) == nil)
}

@Test func mtpIntervalsRejectIdleResetsGapsAndImpossibleCounters() {
    var window = MTPCounterWindow()
    let start = Date(timeIntervalSince1970: 100)
    func counters(_ draft: Double, _ accepted: Double) -> [String: Double] {
        ["ax_engine_mtp_draft_tokens_total": draft, "ax_engine_mtp_accepted_tokens_total": accepted]
    }
    #expect(window.observe(counters(100, 50), at: start) == nil)
    #expect(window.observe(counters(110, 58), at: start.addingTimeInterval(2)) == 80)
    #expect(window.observe(counters(110, 58), at: start.addingTimeInterval(4)) == nil)
    #expect(window.observe(counters(10, 5), at: start.addingTimeInterval(6)) == nil)
    #expect(window.observe(counters(20, 10), at: start.addingTimeInterval(8)) == 50)
    #expect(window.observe(counters(30, 31), at: start.addingTimeInterval(10)) == nil)
    #expect(window.observe(counters(40, 20), at: start.addingTimeInterval(12)) == nil)
    #expect(window.observe(counters(50, 30), at: start.addingTimeInterval(30)) == nil)
    #expect(window.observe([:], at: start.addingTimeInterval(32)) == nil)
    #expect(window.observe(counters(60, 40), at: start.addingTimeInterval(34)) == nil)
    #expect(window.observe(counters(70, 40), at: start.addingTimeInterval(36)) == 0)
}

@Test func hostParsersDoNotFabricateUnknownValues() throws {
    #expect(HostMetricParser.cpu("50.0\n150.0\n", cores: 4) == 50)
    #expect(HostMetricParser.cpu("", cores: 4) == nil)
    #expect(HostMetricParser.cpu("nan", cores: 4) == nil)
    #expect(HostMetricParser.cpu("1.0 bad", cores: 4) == nil)
    let vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages active: 2.\nPages wired down: 1.\nPages occupied by compressor: 1.\n"
    #expect(HostMetricParser.memory(vm, total: 131072) == 50)
    #expect(HostMetricParser.memory("Pages active: 2.", total: 131072) == nil)
    #expect(HostMetricParser.memory(vm, total: 1) == nil)
    func gpu(_ value: Any) throws -> String {
        String(decoding: try PropertyListSerialization.data(fromPropertyList: [["PerformanceStatistics": ["Device Utilization %": value]]], format: .xml, options: 0), as: UTF8.self)
    }
    #expect(HostMetricParser.gpu(try gpu(37)) == 37)
    #expect(HostMetricParser.gpu(try gpu(101)) == nil)
    #expect(HostMetricParser.gpu(try gpu(true)) == nil)
    #expect(HostMetricParser.gpu("missing") == nil)
}

@Test @MainActor func hostPollingIsOptInAndCancellable() async throws {
    #expect(HostMonitor.shouldSample(enabled: true))
    #expect(!HostMonitor.shouldSample(enabled: false))
    let probe = HostProbe(kind: "cpu", executable: URL(fileURLWithPath: "/usr/bin/python3"), arguments: ["-c", "print('80.0')"])
    let monitor = HostMonitor(probes: [probe], interval: .milliseconds(60), cores: 4)
    #expect(monitor.history.samples.isEmpty)
    monitor.setActive(true)
    let deadline = Date().addingTimeInterval(4)
    while monitor.history.samples.count < 2 && Date() < deadline { try await Task.sleep(for: .milliseconds(30)) }
    #expect(monitor.history.latest("cpu") == 20)
    await monitor.shutdown()
    let stoppedCount = monitor.history.samples.count
    try await Task.sleep(for: .milliseconds(200))
    #expect(monitor.history.samples.count == stoppedCount)
    #expect(monitor.history.latest("cpu") == nil)
    #expect(!monitor.active)
    #expect(!monitor.collecting)
    let slow = HostMonitor(probes: [.init(kind: "cpu", executable: URL(fileURLWithPath: "/usr/bin/python3"),
        arguments: ["-c", "import time; time.sleep(20); print('80.0')"])], timeout: 0.2)
    slow.setActive(true)
    let timeout = Date().addingTimeInterval(4)
    while slow.history.samples.isEmpty && Date() < timeout { try await Task.sleep(for: .milliseconds(30)) }
    #expect(slow.history.samples.count == 1)
    #expect(slow.history.latest("cpu") == nil)
    await slow.shutdown()
}

@Test func libraryAssociationDoesNotInferCompatibility() {
    let catalog = CatalogModel(alias: "test", repoId: "Owner/Repo", revision: "abc", mtpIncluded: false)
    let matched = LocalModel(path: "/cache/models--Owner--Repo/snapshots/abc", ready: false, managed: false)
    let other = LocalModel(path: "/cache/models--Other--Repo/snapshots/def", ready: true, managed: false)
    #expect(matched.isCatalogAssociated([catalog]))
    #expect(!matched.ready)
    #expect(!other.isCatalogAssociated([catalog]))
    #expect(other.ready)

}

@Test func chartInspectionDoesNotBridgeMissingOrExpiredSamples() {
    var history = MetricHistory()
    let start = Date(timeIntervalSince1970: 100)
    history.append(["cpu": 25], at: start)
    history.append([:], at: start.addingTimeInterval(5))
    history.append(["cpu": 75], at: start.addingTimeInterval(10))
    let now = start.addingTimeInterval(20)
    #expect(history.sample("cpu", near: start.addingTimeInterval(1), now: now)?.values["cpu"] == 25)
    #expect(history.sample("cpu", near: start.addingTimeInterval(5), now: now) == nil)
    #expect(history.sample("cpu", near: start.addingTimeInterval(9), now: now)?.values["cpu"] == 75)
    #expect(history.sample("cpu", near: now, now: now) == nil)
    #expect(history.sample("cpu", near: now.addingTimeInterval(1), now: now) == nil)
    #expect(history.sample("cpu", near: start, now: start.addingTimeInterval(301)) == nil)
    #expect(history.sample("gpu", near: start, now: now) == nil)
}

@Test func hostHistoryRetainsFourHoursAndExpiresOlderSamples() {
    var history = MetricHistory.hostActivity()
    let start = Date(timeIntervalSince1970: 100_000)
    for seconds in stride(from: 0, through: 6 * 3600, by: 5) {
        history.append(["cpu": Double(seconds % 100)], at: start.addingTimeInterval(Double(seconds)))
    }
    let now = start.addingTimeInterval(6 * 3600)
    #expect(history.samples.count == 2881)
    #expect(history.samples.count <= history.capacity)
    #expect(history.samples.first?.time == now.addingTimeInterval(-4 * 3600))
    #expect(history.sample("cpu", near: now.addingTimeInterval(-3 * 3600), now: now) != nil)
    #expect(history.sample("cpu", near: now.addingTimeInterval(-4 * 3600 - 5), now: now) == nil)
    #expect(MetricHistory().duration == 300) // Runtime history keeps its own retention.
    #expect(ActivityHistoryRange.allCases.map(\.rawValue) == [60, 300, 900, 1800, 3600, 14400])
    for range in ActivityHistoryRange.allCases {
        let plot = history.plotPoints("cpu", window: range.rawValue, now: now)
        #expect(!plot.isEmpty)
        #expect(plot.count <= 600)
        #expect(plot.first!.time >= now.addingTimeInterval(-range.rawValue))
        #expect(plot.last?.time == now)
    }
}

@Test func reducedHistoryPreservesExtremaRealSamplesAndGaps() {
    var history = MetricHistory.hostActivity()
    let start = Date(timeIntervalSince1970: 100_000)
    for seconds in stride(from: 0, through: 14400, by: 5) {
        let value: Double = seconds == 1010 ? 100 : seconds == 1020 ? 0 : 50
        history.append(seconds == 7200 ? [:] : ["cpu": value], at: start.addingTimeInterval(Double(seconds)))
    }
    let now = start.addingTimeInterval(14400)
    let raw = history.points("cpu", now: now)
    let plot = history.plotPoints("cpu", window: 14400, now: now)
    #expect(plot.count <= 600)
    #expect(plot.contains { $0.value == 100 })
    #expect(plot.contains { $0.value == 0 })
    #expect(Set(plot.map(\.id)).count == plot.count)
    let originals = Dictionary(uniqueKeysWithValues: raw.map { ($0.id, $0) })
    for point in plot {
        #expect(originals[point.id]?.time == point.time)
        #expect(originals[point.id]?.value == point.value)
        #expect(originals[point.id]?.segment == point.segment)
    }
    #expect(plot.last(where: { $0.time < start.addingTimeInterval(7200) })?.segment !=
            plot.first(where: { $0.time > start.addingTimeInterval(7200) })?.segment)
    #expect(history.sample("cpu", near: start.addingTimeInterval(7200), now: now) == nil)
    let short = history.plotPoints("cpu", window: 60, now: now)
    #expect(short.map(\.id) == raw.filter { $0.time >= now.addingTimeInterval(-60) }.map(\.id))
}

@Test func guiLibraryRestrictsPublisherWithoutRemovingOtherModels() {
    let own = LocalModel(path: "/cache/models--AutomatosX--Pack/snapshots/rev", ready: true, managed: false)
    let other = LocalModel(path: "/cache/models--community--Pack/snapshots/rev", ready: true, managed: false)
    let managed = LocalModel(path: "/models/AutomatosX--Pack--rev", ready: true, managed: true)
    let imported = LocalModel(path: "/downloads/other-model", ready: true, managed: false)
    #expect(own.isAutomatosX)
    #expect(managed.isAutomatosX)
    #expect(!other.isAutomatosX)
    #expect(!imported.isAutomatosX)
}

@Test @MainActor func guiRejectsThirdPartyDownloadBeforeStartingRuntime() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let store = ModelStore(importsURL: root.appendingPathComponent("imports.json"))
    store.enqueue(CatalogModel(alias: "Other", repoId: "community/Pack", revision: nil, mtpIncluded: false), settings: AppSettings())
    #expect(store.downloads.isEmpty)
    #expect(!store.downloading)
    #expect(store.error != nil)
    #expect(!store.canSelectInGUI("/downloads/other-model"))
    await store.shutdown()
}
