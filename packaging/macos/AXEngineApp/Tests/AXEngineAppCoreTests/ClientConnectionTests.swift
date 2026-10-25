import AppKit
import Foundation
import Testing
@testable import AXEngineAppCore

@Test @MainActor func clientCopyUsesTheConnectedPortAndKeepsDescriptionsRedacted() {
    let key = UUID().uuidString
    let connection = EngineConnection(port: 34567, apiKey: key)
    #expect(connection.apiURL.absoluteString == "http://127.0.0.1:34567/v1")
    #expect(connection.hasAPIKey)
    let descriptionsAreRedacted = !String(describing: connection).contains(key)
        && !String(reflecting: connection).contains(key)
    #expect(descriptionsAreRedacted)
    let reflectedNames = Mirror(reflecting: connection).children.compactMap(\.label)
    #expect(reflectedNames == ["baseURL", "hasAPIKey"])
    #expect(EngineConnection(port: 34567, apiKey: key).id != connection.id)
    let pasteboard = NSPasteboard.withUniqueName()
    let clipboard = SensitiveClipboard(pasteboard: pasteboard)
    defer { clipboard.clearIfOwned(); pasteboard.releaseGlobally() }
    #expect(connection.copyAPIKey(to: clipboard))
    let keyMatches = pasteboard.string(forType: .string) == key
    #expect(keyMatches)
    #expect(pasteboard.types?.contains(.init("org.nspasteboard.ConcealedType")) == true)
    #expect(pasteboard.types?.contains(.init("org.nspasteboard.TransientType")) == true)
    clipboard.clearIfOwned()
    #expect(pasteboard.string(forType: .string) == nil)
}

@Test @MainActor func absentOrInvalidClientKeysLeaveClipboardUnchanged() {
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    pasteboard.setString("User text", forType: .string)
    let clipboard = SensitiveClipboard(pasteboard: pasteboard)
    let connection = EngineConnection(port: 34567)
    #expect(!connection.hasAPIKey)
    #expect(!connection.copyAPIKey(to: clipboard))
    #expect(!clipboard.copy("\n"))
    #expect(!clipboard.copy(String(repeating: "x", count: 16_385)))
    #expect(pasteboard.string(forType: .string) == "User text")
}

@Test @MainActor func secretExpiryDoesNotClearANewerOwnerEvenWithIdenticalText() {
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    let clipboard = SensitiveClipboard(pasteboard: pasteboard)
    let key = UUID().uuidString
    #expect(clipboard.copy(key))
    pasteboard.clearContents()
    pasteboard.setString(key, forType: .string)
    clipboard.clearIfOwned()
    let newerValuePreserved = pasteboard.string(forType: .string) == key
    #expect(newerValuePreserved)
    pasteboard.clearContents()
}

@Test @MainActor func recopyReplacesOwnershipAndClearRemovesOnlyTheLatestSecret() {
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    let clipboard = SensitiveClipboard(pasteboard: pasteboard)
    let first = UUID().uuidString
    let second = UUID().uuidString
    #expect(clipboard.copy(first))
    #expect(clipboard.copy(second))
    let latestMatches = pasteboard.string(forType: .string) == second
    #expect(latestMatches)
    clipboard.clearIfOwned()
    #expect(pasteboard.string(forType: .string) == nil)
    pasteboard.setString("Later user text", forType: .string)
    clipboard.clearIfOwned()
    #expect(pasteboard.string(forType: .string) == "Later user text")
}

@Test @MainActor func scheduledSecretExpiryClearsOwnedContent() async throws {
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    let clipboard = SensitiveClipboard(pasteboard: pasteboard, lifetime: .milliseconds(10))
    defer { clipboard.clearIfOwned() }
    #expect(clipboard.copy(UUID().uuidString))
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while pasteboard.string(forType: .string) != nil && ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
    #expect(pasteboard.string(forType: .string) == nil)
}

@Test @MainActor func scheduledSecretExpiryPreservesOtherAppsText() async throws {
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    let clipboard = SensitiveClipboard(pasteboard: pasteboard, lifetime: .milliseconds(10))
    #expect(clipboard.copy(UUID().uuidString))
    pasteboard.clearContents()
    pasteboard.setString("New user text", forType: .string)
    try await Task.sleep(for: .milliseconds(60))
    #expect(pasteboard.string(forType: .string) == "New user text")
}

@Test @MainActor func secretExpirySurvivesItsViewOwnerBeingReleased() async throws {
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    var clipboard: SensitiveClipboard? = SensitiveClipboard(pasteboard: pasteboard, lifetime: .milliseconds(10))
    #expect(clipboard?.copy(UUID().uuidString) == true)
    weak var lifetimeProbe = clipboard
    clipboard = nil
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while (pasteboard.string(forType: .string) != nil || lifetimeProbe != nil) && ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(10))
    }
    #expect(pasteboard.string(forType: .string) == nil)
    #expect(lifetimeProbe == nil)
}
