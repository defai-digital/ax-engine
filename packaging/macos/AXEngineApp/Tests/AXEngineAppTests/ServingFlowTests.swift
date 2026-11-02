import Foundation
import Testing
import Combine
import AXEngineAppCore
@testable import AXEngineApp

private func tempSettings() throws -> (URL, URL) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let file = root.appendingPathComponent("settings.json")
    var original = AppSettings()
    original.modelsDirectory = root.appendingPathComponent("models").path
    try original.save(to: file)
    return (root, file)
}

@Test @MainActor func serverOptionsSaveAtOnceAndKeepPendingSettingsEdits() throws {
    let (root, file) = try tempSettings(); defer { try? FileManager.default.removeItem(at: root) }
    let app = ApplicationModel(settingsURL: file)
    app.settingsDraft.port = 32003
    app.settingsDraft.maxTokens = 5000
    app.selectServingMode(.coding)
    app.selectConcurrentRequests(6)
    app.selectRequireAPIKey(true)
    // No page bounce, and the server options are saved without the pending edits.
    #expect(app.page == .overview)
    let saved = try AppSettings.load(from: file)
    #expect(saved.servingMode == .coding && saved.concurrentRequests == 6 && saved.requireAPIKey)
    #expect(saved.port == AppSettings().port && saved.maxTokens == AppSettings().maxTokens)
    // The draft keeps its edits and mirrors the saved options, so Revert cannot undo them.
    #expect(app.settingsDraft.port == 32003 && app.settingsDraft.maxTokens == 5000)
    #expect(app.settingsDraft.servingMode == .coding && app.settingsDraft.concurrentRequests == 6 && app.settingsDraft.requireAPIKey)
    app.revertSettings()
    #expect(app.settingsDraft == saved && app.settings == saved)
    app.selectConcurrentRequests(99)
    #expect(app.settings.concurrentRequests == 6)
}

@Test @MainActor func serveIgnoresModelsThatAreNotServable() throws {
    let (root, file) = try tempSettings(); defer { try? FileManager.default.removeItem(at: root) }
    let app = ApplicationModel(settingsURL: file)
    app.page = .models
    app.serve("/not/in/the/library")
    #expect(app.page == .models && app.selectedPath.isEmpty && !app.runtime.owned)
    #expect(!app.canStartServer)
    #expect(app.startBlockReason != nil)
}

@Test func codeFenceDropsOnlyALanguageTag() {
    #expect(CodeFence.body("swift\nlet x = 1\n") == "let x = 1\n")
    #expect(CodeFence.body("\nplain\n") == "plain\n")
    #expect(CodeFence.body("let x = 1\nprint(x)") == "let x = 1\nprint(x)")
    #expect(CodeFence.body("single line") == "single line")
}

@Test @MainActor func menuBarInsertionDoesNotRepublishUnchangedValues() throws {
    let (root, file) = try tempSettings(); defer { try? FileManager.default.removeItem(at: root) }
    let app = ApplicationModel(settingsURL: file)
    let original = app.showsMenuBarItem
    defer { app.showsMenuBarItem = original }
    var changes = 0
    let sink = app.objectWillChange.sink { _ in changes += 1 }
    defer { sink.cancel() }
    // SwiftUI writes the current value back on every scene update.
    for _ in 0..<5 { app.menuBarInsertion.wrappedValue = app.showsMenuBarItem }
    #expect(changes == 0)
    app.menuBarInsertion.wrappedValue = !original
    #expect(changes == 1 && app.showsMenuBarItem == !original)
}
