import SwiftUI
import AppKit
import AXEngineAppCore

@MainActor
final class ApplicationModel: ObservableObject {
    @Published var settings = AppSettings()
    @Published var settingsDraft = AppSettings() {
        didSet { settingsSaved = false }
    }
    @Published var settingsSaved = false
    @Published var settingsError: String?
    @Published var page: Page? = .overview
    @Published var selectedPath = ""
    @Published var modelSearch = ""
    @Published var catalogSearch = ""
    @Published var modelsSection = "library"
    @Published var searchFocusRequest = 0
    @Published var dashboardHistoryWindow: Double = 60
    @Published var hostMonitoringEnabled = UserDefaults.standard.bool(forKey: "hostMonitoringEnabled") {
        didSet { UserDefaults.standard.set(hostMonitoringEnabled, forKey: "hostMonitoringEnabled") }
    }
    @Published var showsMenuBarItem = UserDefaults.standard.object(forKey: "showsMenuBarItem") as? Bool ?? true {
        didSet { UserDefaults.standard.set(showsMenuBarItem, forKey: "showsMenuBarItem") }
    }
    /// Binding for MenuBarExtra(isInserted:). SwiftUI writes it back on every
    /// scene update; republishing an unchanged value would rebuild the main
    /// menu and write again, an unbounded update loop (GBs per minute).
    var menuBarInsertion: Binding<Bool> {
        Binding(get: { self.showsMenuBarItem },
                set: { if $0 != self.showsMenuBarItem { self.showsMenuBarItem = $0 } })
    }
    /// First-run guide: finished or skipped once, then only from Help.
    @Published var onboardingCompleted = UserDefaults.standard.bool(forKey: "onboardingCompleted") {
        didSet { UserDefaults.standard.set(onboardingCompleted, forKey: "onboardingCompleted") }
    }
    @Published var showsOnboarding = false
    func offerOnboardingIfNeeded() {
        if FirstRun.shouldOffer(completed: onboardingCompleted, discoveryAttempted: models.discoveryAttempted,
                                scanning: models.scanning, libraryCount: models.productLibrary.count,
                                scanFailed: models.libraryIssue != nil,
                                serverActive: runtime.owned || runtime.connection != nil) {
            showsOnboarding = true
        }
    }
    /// Serve-model picker state, keyed by model path; per-user convenience.
    @Published var favoriteModels = UserDefaults.standard.stringArray(forKey: "favoriteModels") ?? [] {
        didSet { UserDefaults.standard.set(favoriteModels, forKey: "favoriteModels") }
    }
    @Published var recentModels = UserDefaults.standard.stringArray(forKey: "recentModels") ?? [] {
        didSet { UserDefaults.standard.set(recentModels, forKey: "recentModels") }
    }
    func toggleFavoriteModel(_ path: String) { favoriteModels = ModelPicker.toggling(path, in: favoriteModels) }
    func recordRecentModel(_ path: String) { recentModels = ModelPicker.recording(path, in: recentModels) }
    let updates = AppUpdateController()
    let hostMonitor = HostMonitor()
    let storageMonitor = StorageMonitor()
    let runtime = RuntimeController()
    let models = ModelStore()
    let chat = ChatStore()
    private let settingsURL: URL
    init(settingsURL: URL = AppPaths.settings) {
        self.settingsURL = settingsURL
        do { settings = try AppSettings.load(from: settingsURL) }
        catch { settingsError = error.localizedDescription }
        settingsDraft = settings
    }
    var hasSettingsChanges: Bool { settingsDraft != settings }
    var quitWarning: String? {
        let active = runtime.owned || runtime.busy || models.downloading || models.removing || chat.generating
        guard active || hasSettingsChanges else { return nil }
        var messages: [String] = []
        if active { messages.append("App-owned inference, downloads and model operations will stop. Attached services will keep running.") }
        if hasSettingsChanges { messages.append("Unsaved settings will be discarded.") }
        return messages.joined(separator: "\n\n")
    }
    func saveSettings() -> Bool {
        do {
            try settingsDraft.save(to: settingsURL)
            settings = settingsDraft; settingsError = nil; settingsSaved = true
            return true
        } catch {
            settingsError = error.localizedDescription; settingsSaved = false
            return false
        }
    }
    func selectPlaygroundProfile(_ profile: GenerationProfile) {
        guard !chat.generating else { return }
        // Keep pending edits intact; Settings owns their save/revert flow.
        guard !hasSettingsChanges else { page = .settings; return }
        if profile == .custom {
            settingsDraft.apply(.custom)
            page = .settings
            return
        }
        var updated = settings
        updated.apply(profile)
        do {
            try updated.save(to: settingsURL)
            settings = updated
            settingsDraft = updated
            settingsError = nil
            settingsSaved = true
        } catch {
            settingsError = error.localizedDescription
        }
    }
    /// Server options (serving mode, request cap, API key requirement) save
    /// at once from every page, Settings included. Only that field changes:
    /// other saved values stay, and unsaved Settings edits stay pending.
    private func applyServerOption(_ change: (inout AppSettings) -> Void) {
        var updated = settings
        change(&updated)
        guard updated != settings else { return }
        do {
            try updated.save(to: settingsURL)
            let pending = hasSettingsChanges
            settings = updated
            change(&settingsDraft)
            settingsError = nil
            if !pending { settingsSaved = true }
        } catch {
            settingsError = error.localizedDescription
        }
    }
    /// Serving mode for the next app-owned launch; a running server keeps its
    /// mode until it is restarted.
    func selectServingMode(_ mode: ServingMode) { applyServerOption { $0.servingMode = mode } }
    /// Request admission cap for the next app-owned launch.
    func selectConcurrentRequests(_ count: Int) {
        guard AppSettings.concurrentRequestRange.contains(count) else { return }
        applyServerOption { $0.concurrentRequests = count }
    }
    /// API key requirement for the next app-owned launch.
    func selectRequireAPIKey(_ required: Bool) { applyServerOption { $0.requireAPIKey = required } }

    // MARK: - Serving

    /// Downloaded models the scan has checked; the only ones Start accepts.
    var servableModels: [LocalModel] { models.productLibrary.filter { $0.ready && models.checkedPaths.contains($0.path) } }
    var attachedToService: Bool { runtime.connection != nil && !runtime.owned }
    var inferenceActive: Bool {
        chat.generating || (runtime.metrics["ax_engine_generation_active_streams"] ?? 0) > 0
    }
    /// Why Start Server is unavailable for the selected model, or nil.
    var startBlockReason: String? {
        ServerGate.startBlockReason(.init(selectedPath: selectedPath, selectable: models.canSelectInGUI(selectedPath),
                                          libraryEmpty: servableModels.isEmpty, scanning: models.scanning,
                                          runtimeBusy: runtime.busy, removing: models.removing, attached: attachedToService))
    }
    var canStartServer: Bool { !runtime.owned && runtime.connection == nil && startBlockReason == nil }
    /// The model a restart uses: the selection when it differs and is valid.
    var restartTarget: String {
        selectedPath != runtime.selectedModel && models.canSelectInGUI(selectedPath) ? selectedPath : runtime.selectedModel
    }
    /// A changed selection that cannot be served blocks Restart instead of
    /// silently restarting the running model.
    var canRestartServer: Bool {
        let unservableSwitch = !selectedPath.isEmpty && selectedPath != runtime.selectedModel && !models.canSelectInGUI(selectedPath)
        return runtime.owned && !runtime.busy && runtime.state != "Stopping" && !models.removing && !unservableSwitch
    }
    /// Connecting to an existing service, before it answers.
    var connecting: Bool { !runtime.owned && runtime.busy && runtime.connection == nil && runtime.state == "Connecting" }

    func startServer() {
        guard canStartServer else { return }
        recordRecentModel(selectedPath)
        runtime.start(model: selectedPath, settings: settings)
    }
    func restartServer(model target: String? = nil) {
        guard canRestartServer else { return }
        let path = target ?? restartTarget
        recordRecentModel(path)
        Task { await chat.stop(); await runtime.restart(model: path, settings: settings) }
    }
    func stopServer() {
        Task { await chat.stop(); if runtime.owned { await runtime.stop() } else { runtime.detach() } }
    }
    /// Menu and shortcut entry points: confirm before ending a reply in progress.
    func requestStopServer() {
        if connecting { runtime.detach(); return }
        guard runtime.owned || runtime.connection != nil else { return }
        if inferenceActive && !confirm("Stop the server?", "A reply is being generated. Stopping ends it.", action: "Stop") { return }
        stopServer()
    }
    func requestStartOrRestart() {
        if runtime.owned {
            guard canRestartServer else { return }
            if inferenceActive && !confirm("Restart the server?", "A reply is being generated. Restarting ends it.", action: "Restart") { return }
            restartServer()
        } else if canStartServer {
            startServer()
        } else {
            page = .overview
        }
    }
    private func confirm(_ title: String, _ message: String, action: String) -> Bool {
        NSApp.activate()
        let alert = NSAlert()
        alert.messageText = title; alert.informativeText = message
        alert.addButton(withTitle: action); alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }
    /// One action for "use this model": select it, start it when nothing is
    /// serving, and show Dashboard where the server is controlled. A running
    /// server is never replaced silently; Dashboard offers the restart.
    func serve(_ path: String) {
        guard models.canSelectInGUI(path), !attachedToService, !connecting else { return }
        selectedPath = path
        page = .overview
        if canStartServer { startServer() }
    }

    func revertSettings() {
        settingsDraft = settings; settingsError = nil
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: ApplicationModel?
    private let quitCoordinator = AppQuitCoordinator()
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model else { return .terminateNow }
        return quitCoordinator.request(
            needsConfirmation: model.quitWarning != nil,
            confirm: {
                let alert = NSAlert()
                alert.messageText = "Quit AX Engine?"
                alert.informativeText = model.quitWarning ?? ""
                alert.addButton(withTitle: "Stop and Quit"); alert.addButton(withTitle: "Cancel")
                return alert.runModal() == .alertFirstButtonReturn
            }, shutdown: {
                await model.hostMonitor.shutdown()
                await model.storageMonitor.shutdown()
                await model.chat.stop()
                await model.models.shutdown()
                await model.runtime.stop()
            }, reply: { sender.reply(toApplicationShouldTerminate: true) })
    }
}

@main
struct AXEngineApplication: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = ApplicationModel()
    var body: some Scene {
        Window("AX Engine", id: "main") {
            MainView(model: model, runtime: model.runtime, models: model.models, chat: model.chat)
                .tint(Brand.accentChrome)
                .frame(minWidth: 940, minHeight: 680)
                .onAppear {
                    delegate.model = model
                    model.updates.start()
                    if let logo = Brand.logo { NSApplication.shared.applicationIconImage = logo }
                }
        }
        .defaultSize(width: 1180, height: 800)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About AX Engine") { showAbout() }
            }
            CommandGroup(after: .appInfo) {
                CheckForAppUpdates(updates: model.updates)
            }
            ThirdPartyNoticeCommands()
            SettingsNavigationCommands(model: model)
            NavigationCommands(model: model)
            ServerCommands(model: model, runtime: model.runtime)
            CommandGroup(replacing: .help) {
                Button("Getting Started") { model.showsOnboarding = true }
            }
        }
        MenuBarExtra(isInserted: model.menuBarInsertion) {
            MenuBarContent(model: model, runtime: model.runtime)
        } label: {
            MenuBarLabel(runtime: model.runtime)
        }
        Window("Third-Party Notices", id: "third-party-notices") {
            ThirdPartyNoticesView()
        }
        .windowResizability(.contentSize)
    }
    private func showAbout() {
        let credits = NSMutableAttributedString(string: "Native local inference for Apple Silicon.\n\nManage your models, run local inference and test responses in Playground. This app sends no analytics or telemetry.\n\nOpen Third-Party Notices from the AX Engine menu for open-source acknowledgements.")
        NSApplication.shared.orderFrontStandardAboutPanel(options: [.credits: credits])
    }
}

struct SettingsNavigationCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var model: ApplicationModel
    var body: some Commands {
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") {
                model.page = .settings
                openWindow(id: "main")
            }
            .keyboardShortcut(",", modifiers: .command)
        }
    }
}

struct ThirdPartyNoticeCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    var body: some Commands {
        CommandGroup(after: .appInfo) {
            Button("Third-Party Notices") { openWindow(id: "third-party-notices") }
        }
    }
}

struct NavigationCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var model: ApplicationModel
    var body: some Commands {
        CommandMenu("Navigate") {
            ForEach(Array(Page.allCases.enumerated()), id: \.offset) { index, page in
                Button(page.rawValue) { model.page = page; openWindow(id: "main") }
                    .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
            }
            Divider()
            Button("Search Models") {
                model.page = .models
                if model.modelsSection == "transfers" { model.modelsSection = "library" }
                model.searchFocusRequest += 1
                openWindow(id: "main")
            }.keyboardShortcut("f", modifiers: .command)
            Button("Refresh Models") {
                Task { await model.models.refresh(settings: model.settings) }
            }.keyboardShortcut("r", modifiers: .command)
        }
    }
}

struct ServerCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    var body: some Commands {
        CommandMenu("Server") {
            Button(runtime.owned ? "Restart Server" : "Start Server") { model.requestStartOrRestart(); openWindow(id: "main") }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(runtime.owned ? !model.canRestartServer : !model.canStartServer)
            Button(runtime.state == "Starting" || model.connecting ? "Cancel" : runtime.owned ? "Stop Server" : "Disconnect") { model.requestStopServer() }
                .keyboardShortcut(".", modifiers: [.command, .shift])
                .disabled(!(runtime.owned || runtime.connection != nil || model.connecting) || runtime.state == "Stopping")
            Divider()
            Button("Copy API URL") { if let url = runtime.connection?.apiURL.absoluteString { copyText(url) } }
                .disabled(runtime.connection == nil)
            Button("Open Playground") { model.page = .chat; openWindow(id: "main") }
                .disabled(!runtime.ready)
        }
    }
}

/// Status glyph in the menu bar: filled while serving, outlined otherwise.
struct MenuBarLabel: View {
    @ObservedObject var runtime: RuntimeController
    var body: some View {
        Image(systemName: runtime.ready ? "bolt.horizontal.circle.fill"
              : ["Starting", "Stopping", "Connecting"].contains(runtime.state) ? "hourglass.circle" : "bolt.horizontal.circle")
            .accessibilityLabel("AX Engine, " + runtime.state)
    }
}

/// Menu-bar menu: state, the model it serves, and one-click lifecycle.
struct MenuBarContent: View {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    private var selected: LocalModel? { model.models.productLibrary.first { $0.path == model.selectedPath } }
    private var running: LocalModel? { model.models.productLibrary.first { $0.path == runtime.selectedModel } }

    var body: some View {
        Text("Server: " + runtime.state + (runtime.connection.map { " · " + $0.apiURL.absoluteString } ?? ""))
        if runtime.owned, let running { Text("Serving " + running.displayName) }
        else if !runtime.owned, runtime.connection == nil { Text(selected.map { "Model: " + $0.displayName } ?? "No model selected") }
        Divider()
        if runtime.owned {
            Button("Restart Server") { model.requestStartOrRestart() }.disabled(!model.canRestartServer)
            Button(runtime.state == "Starting" ? "Cancel Start" : "Stop Server") { model.requestStopServer() }
                .disabled(runtime.state == "Stopping")
        } else if runtime.connection != nil || model.connecting {
            Button(model.connecting ? "Cancel Connecting" : "Disconnect") { model.requestStopServer() }
        } else {
            Button("Start Server") { model.startServer() }.disabled(!model.canStartServer)
            if !model.servableModels.isEmpty {
                Menu("Start with…") {
                    ForEach(model.servableModels) { item in
                        Button(item.displayName) { model.selectedPath = item.path; model.startServer() }
                    }
                }.disabled(runtime.busy || model.models.scanning || model.models.removing)
            }
        }
        Divider()
        Button("Copy API URL") { if let url = runtime.connection?.apiURL.absoluteString { copyText(url) } }
            .disabled(runtime.connection == nil)
        Button("Open Playground") { model.page = .chat; openWindow(id: "main"); NSApp.activate() }
            .disabled(!runtime.ready)
        Button("Open AX Engine") { openWindow(id: "main"); NSApp.activate() }
        Divider()
        Button("Quit AX Engine") { NSApp.terminate(nil) }.keyboardShortcut("q")
    }
}
