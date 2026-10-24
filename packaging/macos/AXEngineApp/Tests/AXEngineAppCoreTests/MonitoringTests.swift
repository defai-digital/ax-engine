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

@Test @MainActor func hostPollingIsOptInVisibleAndCancellable() async throws {
    #expect(HostMonitor.shouldSample(enabled: true, foreground: true, visible: true))
    #expect(!HostMonitor.shouldSample(enabled: false, foreground: true, visible: true))
    #expect(!HostMonitor.shouldSample(enabled: true, foreground: false, visible: true))
    #expect(!HostMonitor.shouldSample(enabled: true, foreground: true, visible: false))
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
