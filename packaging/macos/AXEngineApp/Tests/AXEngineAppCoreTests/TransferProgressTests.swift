import Foundation
import Testing
@testable import AXEngineAppCore

@Test func progressParsesPhaseAndByteCountsWhenPresent() {
    let output = """
    {"event":"progress","done":0,"total":100,"file":"Checking public model files","phase":"inventory","bytes_done":null,"bytes_total":null}
    {"event":"progress","done":180,"total":400,"file":"model.safetensors","phase":"transfer","bytes_done":2,"bytes_total":4}
    """
    let progress = DownloadProgress.parse(output)
    #expect(progress.phase == .transfer)
    #expect(progress.bytesDone == 2)
    #expect(progress.bytesTotal == 4)
    #expect(progress.fraction == 0.45)
    #expect(progress.detail == "model.safetensors")
    // A later phase without byte counts clears them instead of keeping stale sizes.
    let validating = DownloadProgress.parse(output + "\n" + #"{"event":"progress","done":95,"total":100,"file":"Validating model manifest","phase":"validate"}"#)
    #expect(validating.phase == .validate)
    #expect(validating.bytesDone == nil && validating.bytesTotal == nil)
    // Unknown phases and malformed byte counts are ignored, not fatal.
    let odd = DownloadProgress.parse(#"{"event":"progress","done":1,"total":2,"file":"x","phase":"mystery","bytes_done":-5,"bytes_total":0}"#)
    #expect(odd.phase == nil && odd.bytesDone == nil && odd.fraction == 0.5)
}

@Test func legacyRatioInfersPhases() {
    #expect(DownloadPhase.inferred(fromFraction: nil) == nil)
    #expect(DownloadPhase.inferred(fromFraction: 0.3) == .transfer)
    #expect(DownloadPhase.inferred(fromFraction: 0.95) == .validate)
    #expect(DownloadPhase.inferred(fromFraction: 1) == .ready)
}

@Test func rateEstimatorUsesAShortWindowAndDetectsStalls() {
    var estimator = TransferRateEstimator()
    let start = Date(timeIntervalSince1970: 1_000)
    #expect(estimator.observe(bytes: 0, at: start) == nil)
    #expect(estimator.observe(bytes: 1_000_000, at: start.addingTimeInterval(0.5)) == nil)
    let rate = estimator.observe(bytes: 4_000_000, at: start.addingTimeInterval(2))
    #expect(rate == 2_000_000)
    // Samples older than the window drop out, so the rate follows the recent speed.
    _ = estimator.observe(bytes: 4_000_000 + 20_000_000, at: start.addingTimeInterval(22))
    let recent = estimator.observe(bytes: 4_000_000 + 30_000_000, at: start.addingTimeInterval(32))
    #expect(recent == 1_000_000)
    #expect(estimator.stalledFor(at: start.addingTimeInterval(40)) == nil)
    #expect(estimator.stalledFor(at: start.addingTimeInterval(32 + TransferRateEstimator.stallAfter)) == TransferRateEstimator.stallAfter)
    // A regression (resumed transfer) resets the history.
    #expect(estimator.observe(bytes: 10, at: start.addingTimeInterval(70)) == nil)
    #expect(TransferRateEstimator.remainingSeconds(remainingBytes: 10_000_000, rate: 1_000_000) == 10)
    #expect(TransferRateEstimator.remainingSeconds(remainingBytes: 10, rate: 1) == nil)
}

@Test func transferTextIsCompactAndMarksEstimates() {
    #expect(TransferFormat.remaining(nil) == nil)
    #expect(TransferFormat.remaining(5) == "a few seconds left")
    #expect(TransferFormat.remaining(45) == "about 45 s left")
    #expect(TransferFormat.remaining(600) == "about 10 min left")
    #expect(TransferFormat.remaining(7_200) == "about 2.0 h left")
    #expect(TransferFormat.rate(500) == nil)
    #expect(TransferFormat.duration(3_725) == "1 h 2 min")
    let exact = TransferFormat.summary(done: 2_000_000_000, total: 4_000_000_000, rate: 100_000_000, remaining: 20)
    #expect(exact?.hasPrefix("2 GB of 4 GB · ") == true)
    #expect(exact?.hasSuffix("about 20 s left") == true)
    let estimated = TransferFormat.summary(done: 1, total: 2, rate: nil, remaining: nil, estimated: true)
    #expect(estimated?.hasPrefix("about ") == true)
    #expect(TransferFormat.summary(done: nil, total: nil, rate: nil, remaining: nil) == nil)
}

@Test func downloadItemSummarisesTransferAndReadyStates() throws {
    var item = DownloadItem(model: CatalogModel(alias: "A", repoId: "AutomatosX/A", revision: "r", mtpIncluded: true, approxSizeBytes: 4_000),
                            destination: "/tmp/a")
    #expect(item.progressSummary == nil)
    item.state = "Downloading"; item.phase = .transfer; item.bytesDone = 1_000; item.bytesTotal = 4_000; item.bytesPerSecond = 1_500
    #expect(item.phaseTitle == "Downloading")
    let summary = try #require(item.progressSummary)
    #expect(summary.contains(" of ") && summary.contains("/s · ") && summary.hasSuffix("a few seconds left"))
    item.phase = .validate
    #expect(item.phaseTitle == "Verifying model files")
    #expect(item.progressSummary?.contains("left") == false)
    item.state = "Ready"; item.bytesEstimated = true
    #expect(item.progressSummary == "about 4 KB")
    #expect(item.isFinished && !item.isActive)
}

@Test func removalGateExplainsPermanentAndTransientBlocks() {
    let imported = RemovalGate.State(removable: false, source: "Imported", downloading: true)
    #expect(RemovalGate.isPermanent(imported))
    #expect(RemovalGate.reason(imported)?.contains("Imported folders") == true)
    #expect(RemovalGate.reason(.init(removable: false, source: "Configured folder"))?.contains("AX_ENGINE_MODELS_DIR") == true)
    #expect(RemovalGate.reason(.init(removable: false, source: "Hugging Face cache"))?.contains("AutomatosX") == true)
    var managed = RemovalGate.State(removable: true, source: "App folder")
    #expect(RemovalGate.reason(managed) == nil)
    managed.downloading = true; managed.servedByThisApp = true
    #expect(RemovalGate.reason(managed) == "Stop the server to remove the model it is serving.")
    managed.servedByThisApp = false
    #expect(RemovalGate.reason(managed) == "Wait for downloads to finish before removing models.")
    managed.downloading = false; managed.attachedToExternalService = true
    #expect(RemovalGate.reason(managed)?.contains("Disconnect") == true)
}

@Test func serverGateExplainsWhyStartIsUnavailable() {
    #expect(ServerGate.startBlockReason(.init(selectedPath: "/m", selectable: true, libraryEmpty: false)) == nil)
    #expect(ServerGate.startBlockReason(.init(selectedPath: "", selectable: false, libraryEmpty: true))?.contains("Catalog") == true)
    #expect(ServerGate.startBlockReason(.init(selectedPath: "", selectable: false, libraryEmpty: false)) == "Choose an installed model to start.")
    #expect(ServerGate.startBlockReason(.init(selectedPath: "/m", selectable: false, libraryEmpty: false, scanning: true))?.contains("scan") == true)
    #expect(ServerGate.startBlockReason(.init(selectedPath: "/m", selectable: false, libraryEmpty: false))?.contains("incomplete") == true)
    #expect(ServerGate.startBlockReason(.init(selectedPath: "/m", selectable: true, libraryEmpty: false, attached: true))?.contains("Disconnect") == true)
    #expect(ServerGate.startBlockReason(.init(selectedPath: "/m", selectable: true, libraryEmpty: false, runtimeBusy: true))?.contains("starting or stopping") == true)
    let started = Date(timeIntervalSince1970: 0)
    #expect(ServerGate.stateDetail(state: "Starting", startedAt: started, now: started.addingTimeInterval(75)) == "Loading the model · 1 min 15 s")
    #expect(ServerGate.stateDetail(state: "Running", startedAt: started, now: started.addingTimeInterval(3_600)) == "Up for 1 h 0 min")
    #expect(ServerGate.stateDetail(state: "Stopped", startedAt: started) == nil)
    #expect(ServerGate.stateDetail(state: "Starting", startedAt: nil) == nil)
}

@Test func catalogRowsMatchPartialStagesAndOnlyAutomatosXRows() {
    let item = CatalogModel(alias: "Fixture", repoId: "AutomatosX/Fixture", revision: "abc123", mtpIncluded: false)
    #expect(item.folderName == "AutomatosX--Fixture--abc123")
    let stage = PartialDownload(path: "/models/.AutomatosX--Fixture--abc123.ax-download", label: "AutomatosX/Fixture", sizeBytes: 1)
    #expect(URL(fileURLWithPath: stage.path).lastPathComponent == "." + item.folderName + ".ax-download")
}
