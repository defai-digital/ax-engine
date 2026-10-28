import AppKit
import Combine
import Foundation
import Testing
@testable import AXEngineApp

@MainActor
private final class TestUpdateDriver: AppUpdateDriver {
    let ready = CurrentValueSubject<Bool, Never>(true)
    let automatic = CurrentValueSubject<Bool, Never>(false)
    var canCheck: AnyPublisher<Bool, Never> { ready.eraseToAnyPublisher() }
    var automaticChecks: AnyPublisher<Bool, Never> { automatic.eraseToAnyPublisher() }
    var starts = 0
    var checks = 0
    var failure: Error?
    func start() throws { starts += 1; if let failure { throw failure } }
    func check() { checks += 1; ready.send(false) }
    func setAutomaticChecks(_ enabled: Bool) { automatic.send(enabled) }
}

private var releaseInfo: [String: Any] {
    ["AXEngineDistributionBuild": true,
     "SUFeedURL": "https://download.automatosx.com/ax-engine-desktop/appcast.xml",
     "SUPublicEDKey": Data(repeating: 1, count: 32).base64EncodedString(),
     "SUAllowsAutomaticUpdates": false, "SUEnableSystemProfiling": false]
}

@Test @MainActor func updaterStartsOnceTracksBusyStateAndPersistsUserChoiceThroughDriver() {
    let driver = TestUpdateDriver()
    let updates = AppUpdateController(info: releaseInfo, makeDriver: { driver })
    #expect(!updates.canCheck)
    updates.start(); updates.start()
    #expect(driver.starts == 1)
    #expect(updates.canCheck)
    #expect(!updates.automaticChecks)
    updates.setAutomaticChecks(true)
    #expect(updates.automaticChecks)
    updates.check(); updates.check()
    #expect(driver.checks == 1)
    #expect(!updates.canCheck)
    driver.ready.send(true)
    updates.check()
    #expect(driver.checks == 2)
    updates.setAutomaticChecks(false)
    #expect(!updates.automaticChecks)
}

@Test @MainActor func developmentAndInvalidTrustConfigurationNeverStartUpdater() {
    var invalid = releaseInfo
    invalid["SUPublicEDKey"] = "not-a-key"
    var wrongFeed = releaseInfo
    wrongFeed["SUFeedURL"] = "https://example.invalid/appcast.xml"
    var silent = releaseInfo
    silent["SUAllowsAutomaticUpdates"] = true
    for info in [[:], invalid, wrongFeed, silent] {
        let driver = TestUpdateDriver()
        let updates = AppUpdateController(info: info, makeDriver: { driver })
        updates.start(); updates.check(); updates.setAutomaticChecks(true)
        #expect(driver.starts == 0)
        #expect(driver.checks == 0)
        #expect(!updates.canCheck)
        #expect(updates.unavailableReason != nil)
    }
}

@Test @MainActor func updaterStartupFailureDoesNotEnableUpdateActions() {
    let driver = TestUpdateDriver()
    driver.failure = NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "fixture unavailable"])
    let updates = AppUpdateController(info: releaseInfo, makeDriver: { driver })
    updates.start(); updates.check()
    #expect(!updates.canCheck)
    #expect(driver.checks == 0)
    #expect(updates.unavailableReason?.contains("fixture unavailable") == true)
    #expect(updates.canRetryStart)
    driver.failure = nil
    updates.retryStart()
    #expect(updates.canCheck)
    #expect(updates.unavailableReason == nil)
    #expect(driver.starts == 2)
}

@Test @MainActor func cancelledUpdateRestartCanRetryAndDuplicateQuitWaitsForCleanup() async {
    let coordinator = AppQuitCoordinator()
    var shutdowns = 0
    var replies = 0
    var resume: CheckedContinuation<Void, Never>?
    let shutdown: @MainActor () async -> Void = {
        shutdowns += 1
        await withCheckedContinuation { resume = $0 }
    }
    let reply: @MainActor () -> Void = { replies += 1 }
    #expect(coordinator.request(needsConfirmation: true, confirm: { false }, shutdown: shutdown, reply: reply) == .terminateCancel)
    await Task.yield()
    #expect(shutdowns == 0)
    #expect(replies == 0)
    #expect(coordinator.request(needsConfirmation: true, confirm: { true }, shutdown: shutdown, reply: reply) == .terminateLater)
    #expect(coordinator.request(needsConfirmation: false, confirm: { true }, shutdown: shutdown, reply: reply) == .terminateLater)
    for _ in 0..<100 where resume == nil { await Task.yield() }
    #expect(shutdowns == 1)
    #expect(replies == 0)
    resume?.resume()
    for _ in 0..<100 where replies == 0 { await Task.yield() }
    #expect(replies == 1)
}

@Test @MainActor func updaterRestartWarnsBeforeDiscardingSettingsDraft() {
    let model = ApplicationModel(settingsURL: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
    #expect(model.quitWarning == nil)
    model.settingsDraft.port += 1
    #expect(model.quitWarning?.contains("Unsaved settings") == true)
    model.revertSettings()
    #expect(model.quitWarning == nil)
}

@Test @MainActor func quitConfirmationCannotReenterCleanupThroughTheModalRunLoop() async {
    let coordinator = AppQuitCoordinator()
    var shutdowns = 0
    var replies = 0
    let reply: @MainActor () -> Void = { replies += 1 }
    let shutdown: @MainActor () async -> Void = { shutdowns += 1 }
    let result = coordinator.request(needsConfirmation: true, confirm: {
        let nested = coordinator.request(needsConfirmation: true, confirm: {
            Issue.record("Nested quit must not open another confirmation")
            return true
        }, shutdown: shutdown, reply: reply)
        #expect(nested == .terminateCancel)
        return false
    }, shutdown: shutdown, reply: reply)
    #expect(result == .terminateCancel)
    await Task.yield()
    #expect(shutdowns == 0)
    #expect(replies == 0)
}
