import SwiftUI
import AppKit
import AXEngineAppCore

/// Launch flags for screenshot and end-to-end QA of source builds. They do
/// nothing in a distribution build (no development runtime in Info.plist).
///   -AXStartServer YES            start the preselected model once the scan allows it
///   -AXCompareModelB <substring>  with -AXComparePrompt <text>: run A/B compare
///                                 against the first servable model whose path matches
///   -AXSettingsPane <general|server|models|playground|advanced>  first Settings tab (with -AXOpenSettings YES)
///   -AXShowMenuPanel YES          also show the menu bar panel in a window for screenshots
@MainActor
enum QALaunch {
    nonisolated static var enabled: Bool { Bundle.main.object(forInfoDictionaryKey: "AXEngineDevelopmentRuntime") != nil }

    static func run(_ model: ApplicationModel) async {
        guard enabled else { return }
        let defaults = UserDefaults.standard
        if defaults.bool(forKey: "AXShowMenuPanel") { showMenuPanel(model) }
        if defaults.bool(forKey: "AXStartServer") {
            guard await wait(seconds: 180, { model.canStartServer }) else { return }
            model.startServer()
        }
        if let match = defaults.string(forKey: "AXCompareModelB"), let prompt = defaults.string(forKey: "AXComparePrompt") {
            guard await wait(seconds: 900, { model.runtime.ready && model.runtime.owned }),
                  let path = model.servableModels.first(where: { $0.path.contains(match) && $0.path != model.runtime.selectedModel })?.path
            else { return }
            model.compare.prompt = prompt
            model.runCompare(bPath: path, confirmed: true)
        }
    }

    private static var menuPanel: NSPanel?
    private static func showMenuPanel(_ model: ApplicationModel) {
        let host = NSHostingView(rootView: MenuBarContent(model: model, runtime: model.runtime).tint(Brand.accent))
        let panel = NSPanel(contentRect: NSRect(origin: .zero, size: host.fittingSize),
                            styleMask: [.titled, .utilityWindow], backing: .buffered, defer: false)
        panel.title = "Menu Bar Panel (QA)"
        panel.contentView = host
        panel.setFrameOrigin(NSPoint(x: 40, y: 200))
        panel.orderFrontRegardless()
        menuPanel = panel
    }

    private static func wait(seconds: Double, _ condition: () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while Date() < deadline && !Task.isCancelled {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(500))
        }
        return !Task.isCancelled && condition()
    }
}
