import Foundation
import Testing
@testable import AXEngineAppCore

@Test func diskCapacityUsesExistingParentAndRejectsMissingMounts() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let path = root.appendingPathComponent("not-created/models").path
    let snapshot = try StorageSnapshot.read(path: path)
    #expect(snapshot.valid)
    #expect(snapshot.requestedPath == path)
    #expect(snapshot.resolvedPath == root.path)
    #expect(snapshot.usedBytes + snapshot.availableBytes == snapshot.totalBytes)
    #expect(throws: (any Error).self) { try StorageSnapshot.read(path: "/Volumes/AX-Missing-" + UUID().uuidString + "/Models") }
    let link = root.appendingPathComponent("offline")
    try FileManager.default.createSymbolicLink(atPath: link.path, withDestinationPath: root.appendingPathComponent("missing").path)
    #expect(throws: (any Error).self) { try StorageSnapshot.read(path: link.appendingPathComponent("Models").path) }
    #expect(!StorageSnapshot(totalBytes: 100, availableBytes: 101, volumeName: "", requestedPath: "", resolvedPath: "").valid)
    #expect(!StorageSnapshot(totalBytes: 100, availableBytes: -1, volumeName: "", requestedPath: "", resolvedPath: "").valid)
}

@Test func inventorySeparatesFilesFromUncheckedAndIncompleteModels() {
    let rows = [LocalModel(path: "/complete", ready: true, managed: false),
                LocalModel(path: "/partial", ready: false, managed: true),
                LocalModel(path: "/previous", ready: true, managed: false)]
    let summary = ModelInventory(library: rows, checked: ["/complete", "/partial", "/not-in-library"])
    #expect(summary.total == 3)
    #expect(summary.complete == 1)
    #expect(summary.incomplete == 1)
    #expect(summary.unchecked == 1)
}

@Test @MainActor func realCapacityHelperAndFailureDoNotFabricateFreeSpace() async throws {
    let monitor = StorageMonitor(helper: testScannerURL())
    await monitor.refresh(path: FileManager.default.temporaryDirectory.path)
    #expect(monitor.snapshot?.valid == true)
    #expect(monitor.updatedAt != nil)
    await monitor.refresh(path: "/Volumes/AX-Missing-" + UUID().uuidString)
    #expect(monitor.snapshot == nil)
    #expect(monitor.updatedAt == nil)
    #expect(monitor.error != nil)
    #expect(!monitor.refreshing)
    await monitor.shutdown()
}

@Test @MainActor func capacityProbeTimeoutAndCancellationAreBounded() async throws {
    let fixture = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try Data("#!/usr/bin/python3\nimport time\ntime.sleep(20)\n".utf8).write(to: fixture)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: fixture.path)
    defer { try? FileManager.default.removeItem(at: fixture) }
    let monitor = StorageMonitor(helper: fixture, timeout: 0.2)
    let started = Date()
    await monitor.refresh(path: "/tmp")
    #expect(Date().timeIntervalSince(started) < 5)
    #expect(monitor.snapshot == nil)
    #expect(monitor.error != nil)
    let pending = Task { await monitor.refresh(path: "/tmp") }
    try await Task.sleep(for: .milliseconds(50))
    await monitor.shutdown()
    await pending.value
    #expect(!monitor.refreshing)
    #expect(monitor.snapshot == nil)
}
