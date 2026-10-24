import Foundation
import Testing
import AXEngineAppCore
@testable import AXEngineApp

@Test @MainActor func quickProfileSelectionPersistsAndKeepsRuntimeSettings() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("settings.json")
    var original = AppSettings()
    original.port = 32001
    original.modelsDirectory = root.appendingPathComponent("models").path
    try original.save(to: file)
    let app = ApplicationModel(settingsURL: file)
    for profile in [GenerationProfile.coding, .chatbot, .media] {
        app.selectPlaygroundProfile(profile)
        let persisted = try AppSettings.load(from: file)
        #expect(persisted == app.settings)
        #expect(persisted.effectiveProfile == profile)
        #expect(!persisted.hasRuntimeChanges(comparedTo: original))
        #expect(!app.hasSettingsChanges)
        #expect(app.settingsError == nil)
    }
    let saved = app.settings
    app.selectPlaygroundProfile(.custom)
    #expect(app.page == .settings)
    #expect(app.settingsDraft.generationProfile == .custom)
    #expect(app.settingsDraft.maxTokens == saved.maxTokens)
    #expect(try AppSettings.load(from: file) == saved)
}

@Test @MainActor func quickProfileSelectionPreservesUnsavedEdits() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("settings.json")
    let original = AppSettings()
    try original.save(to: file)
    let app = ApplicationModel(settingsURL: file)
    app.settingsDraft.maxTokens = 5000
    app.settingsDraft.port = 32002
    let draft = app.settingsDraft
    app.selectPlaygroundProfile(.coding)
    #expect(app.page == .settings)
    #expect(app.settingsDraft == draft)
    #expect(app.settings == original)
    #expect(try AppSettings.load(from: file) == original)
}

@Test @MainActor func failedProfileSaveKeepsThePreviouslySelectedProfile() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let file = root.appendingPathComponent("settings.json")
    let original = AppSettings()
    try original.save(to: file)
    let app = ApplicationModel(settingsURL: file)
    // Replace the parent directory with a file, producing a deterministic
    // persistence failure without relying on user permissions or credentials.
    try FileManager.default.removeItem(at: root)
    try Data("blocked".utf8).write(to: root)
    app.selectPlaygroundProfile(.coding)
    #expect(app.settings == original)
    #expect(app.settingsDraft == original)
    #expect(app.settingsError != nil)
}
