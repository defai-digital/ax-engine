import SwiftUI
import AppKit
import Combine
import AXEngineAppCore

@MainActor
final class ApplicationModel: ObservableObject {
    @Published var settings = AppSettings()
    @Published var settingsDraft = AppSettings() {
        didSet { settingsSaved = false }
    }
    @Published var settingsSaved = false
    @Published var settingsError: String?
    /// `-AXOpenPage home|chat|models|api|system` opens a page at launch (QA screenshots).
    @Published var page: Page? = Page.launchPage ?? .home
    /// Each increment asks the window to open the Settings window; the model
    /// cannot call `openSettings` itself.
    @Published var settingsRequests = 0
    @Published var showsPalette = false
    @Published var showsTelemetry = UserDefaults.standard.object(forKey: "showsTelemetry") as? Bool ?? true {
        didSet { UserDefaults.standard.set(showsTelemetry, forKey: "showsTelemetry") }
    }
    func openSettings() { settingsRequests += 1 }
    @Published var selectedPath = ""
    @Published var modelSearch = ""
    @Published var catalogSearch = QALaunch.enabled ? UserDefaults.standard.string(forKey: "AXCatalogSearch") ?? "" : ""
    @Published var modelsSection = QALaunch.enabled ? UserDefaults.standard.string(forKey: "AXModelsSection") ?? "library" : "library"
    @Published var searchFocusRequest = 0
    @Published var dashboardHistoryWindow: Double = ActivityHistoryRange.fiveMinutes.rawValue
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
    /// Which discovered models every chooser lists: AutomatosX packs, or every
    /// model folder found on disk. Narrowing never stops or hides the serving
    /// model; a hidden selection falls back and pins return with the scope.
    @Published var libraryScope = LibraryScope(rawValue: UserDefaults.standard.string(forKey: "libraryScope") ?? "") ?? .automatosX {
        didSet {
            guard libraryScope != oldValue else { return }
            UserDefaults.standard.set(libraryScope.rawValue, forKey: "libraryScope")
            models.scope = libraryScope
            preselectModel()
        }
    }
    private var servedModelSync: AnyCancellable?
    func toggleFavoriteModel(_ path: String) { favoriteModels = ModelPicker.toggling(path, in: favoriteModels) }
    func recordRecentModel(_ path: String) { recentModels = ModelPicker.recording(path, in: recentModels) }
    let updates = AppUpdateController()
    let hostMonitor = HostMonitor()
    let storageMonitor = StorageMonitor()
    let runtime = RuntimeController()
    let models = ModelStore()
    let chat = ChatStore()
    let compare = CompareStore()
    private let settingsURL: URL
    init(settingsURL: URL = AppPaths.settings) {
        self.settingsURL = settingsURL
        do { settings = try AppSettings.load(from: settingsURL) }
        catch { settingsError = error.localizedDescription }
        settingsDraft = settings
        models.scope = libraryScope
        servedModelSync = runtime.$selectedModel.combineLatest(runtime.$owned).sink { [models] path, owned in
            let served = owned ? path : ""
            if models.servedPath != served { models.servedPath = served }
        }
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
        guard !hasSettingsChanges else { openSettings(); return }
        if profile == .custom {
            settingsDraft.apply(.custom)
            openSettings()
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
        takeOverFromCompare()
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
        cancelPendingStop()
        Task { await chat.stop(); if runtime.owned { await runtime.stop() } else { runtime.detach() } }
    }
    // MARK: - A/B compare

    /// Why Compare cannot run now, or nil.
    func compareBlockReason(bPath: String) -> String? {
        if !runtime.owned { return attachedToService ? "Compare restarts the server, so it needs a server this app started."
                                                     : "Start the server on model A first." }
        if !runtime.ready || runtime.busy { return "Wait until the server is ready." }
        if chat.generating { return "Wait for the Playground reply to finish." }
        if stopPending { return "The server stops after the current reply." }
        if settings.servingMode != runtime.servingMode || settings.concurrentRequests != runtime.concurrentRequests
            || settings.requireAPIKey != runtime.requiresAPIKey {
            return "Restart to apply the saved serving options first, so A and B run with the same options."
        }
        if bPath.isEmpty { return "Choose model B." }
        if bPath == runtime.selectedModel { return "Choose a different model for B." }
        if !models.canSelectInGUI(bPath) { return "Model B is not ready to serve." }
        if compare.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a prompt." }
        return nil
    }

    /// Runs the prompt on the serving model (A), restarts the server on B,
    /// runs it again, then optionally restarts on A. Each restart reloads a
    /// model and cuts off other clients, so it asks first. Settings are
    /// snapshotted so both runs and every launch use the same values.
    func runCompare(bPath: String, confirmed: Bool = false) {
        guard compareBlockReason(bPath: bPath) == nil, !compare.running else { return }
        let aPath = runtime.selectedModel
        let title = { (path: String) in self.models.productLibrary.first { $0.path == path }?.label.title ?? (path as NSString).lastPathComponent }
        let inFlight = Int(runtime.metrics["ax_engine_http_requests_in_flight"] ?? 0)
        guard confirmed || confirm("Run A/B compare?",
                      "AX Engine runs the prompt on \(title(aPath)), restarts the server on \(title(bPath)) and runs it again"
                      + (compare.returnToA ? ", then restarts on \(title(aPath))." : ".")
                      + " Loading a model can take a minute or more."
                      + (inFlight > 0 ? " \(inFlight) request\(inFlight == 1 ? " is" : "s are") in flight and will be cut off." : " Other apps lose their connection during each restart."),
                      action: "Run Compare") else { return }
        compare.resetSides(aPath: aPath, aTitle: title(aPath), bPath: bPath, bTitle: title(bPath))
        compareTakenOver = false
        let prompt = compare.prompt, system = chat.systemPrompt, returnToA = compare.returnToA
        let snapshot = settings
        compare.begin { [self] in
            for (isA, path) in [(true, aPath), (false, bPath)] {
                guard !Task.isCancelled else { break }
                if runtime.selectedModel != path || !runtime.ready {
                    compare.update(isA) { $0.phase = .loading }
                    compare.status = "Loading \(title(path))…"
                    guard await switchServer(to: path, settings: snapshot) else {
                        compare.update(isA) { $0.phase = Task.isCancelled ? .stopped : .failed(self.runtime.error ?? "The server did not become ready.") }
                        break
                    }
                }
                guard let connection = runtime.connection, let id = runtime.models.first?.id else {
                    compare.update(isA) { $0.phase = .failed("The server is not reachable.") }; break
                }
                compare.status = "Running \(isA ? "A" : "B")…"
                let temperature = runtime.models.first(where: { $0.id == id })?.capabilities?.temperature == true
                await compare.stream(isA: isA, prompt: prompt, system: system, model: id, connection: connection,
                                     settings: snapshot, supportsTemperature: temperature)
                if compare.a.phase != .done && isA { break }
            }
            if compare.a.phase == .waiting { compare.a.phase = .stopped }
            if compare.b.phase == .waiting { compare.b.phase = .stopped }
            // Restore A even after a failed B load or a cancel, unless the
            // user took over the server (Stop, Start, Restart or Serve).
            if returnToA && !compareTakenOver && !(runtime.ready && runtime.selectedModel == aPath) {
                compare.status = "Restoring \(title(aPath))…"
                // An unstructured task does not inherit this task's cancellation.
                let restored = await Task { await self.switchServer(to: aPath, settings: snapshot) }.value
                if !restored && !compareTakenOver {
                    compare.notice = "Could not restart on \(title(aPath)): " + (runtime.error ?? "the server did not become ready.") + " Start it from Home."
                }
            } else if !(runtime.ready && runtime.selectedModel == aPath) && !compareTakenOver {
                compare.notice = runtime.ready ? "The server is now serving \(title(runtime.selectedModel))." : "The server is stopped."
            }
            if !compareTakenOver { selectedPath = runtime.owned ? runtime.selectedModel : aPath }
            compare.status = nil
        }
    }

    /// Set when the user starts, stops, restarts or switches the server while
    /// a compare runs; the compare is cancelled and does not restore A.
    private var compareTakenOver = false
    /// Every user-initiated server action calls this first.
    private func takeOverFromCompare() {
        guard compare.running else { return }
        compareTakenOver = true
        compare.cancel()
    }

    /// Starts (or restarts) the app-owned server on `path` and waits until it
    /// is ready. Selection and recent models change only after success.
    private func switchServer(to path: String, settings snapshot: AppSettings) async -> Bool {
        guard models.canSelectInGUI(path), runtime.owned || runtime.connection == nil else { return false }
        await chat.stop()
        // A cancel or takeover during either await must not launch the next model.
        guard !Task.isCancelled, !compareTakenOver else { return false }
        if runtime.owned {
            await runtime.stop()
            guard !Task.isCancelled, !compareTakenOver else { return false }
        }
        runtime.start(model: path, settings: snapshot)
        // A cold load of a large pack can take several minutes.
        let deadline = Date().addingTimeInterval(900)
        while Date() < deadline && !Task.isCancelled && !compareTakenOver {
            if runtime.ready && runtime.selectedModel == path {
                selectedPath = path; recordRecentModel(path)
                return true
            }
            if !runtime.owned && !runtime.busy { return false }
            try? await Task.sleep(for: .milliseconds(500))
        }
        // Cancelled or timed out: do not leave a load running that nobody
        // asked for. A user takeover owns the server now, so leave it alone.
        if !compareTakenOver && runtime.owned { await runtime.stop() }
        return false
    }

    /// Menu and shortcut entry points: confirm before ending a reply in progress.
    func requestStopServer() {
        if connecting { runtime.detach(); return }
        guard runtime.owned || runtime.connection != nil else { return }
        guard inferenceActive, runtime.owned else {
            if inferenceActive && !confirm("Stop the server?", "A reply is being generated. Stopping ends it.", action: "Stop") { return }
            takeOverFromCompare(); stopServer(); return
        }
        // 12-states.svg "Stop with apps connected": finish, stop now or keep serving.
        let streams = Int(runtime.metrics["ax_engine_generation_active_streams"] ?? (chat.generating ? 1 : 0))
        let title = models.productLibrary.first { $0.path == runtime.selectedModel }?.label.title ?? "the model"
        NSApp.activate()
        let alert = NSAlert()
        alert.messageText = "Stop serving \(title)?"
        alert.informativeText = (streams > 1 ? "\(streams) replies are streaming. Stopping now ends them."
                                             : "A reply is streaming. Stopping now ends it.")
            + " Finish Reply, Then Stop waits up to 10 minutes."
            + (stopPending ? " A stop is already scheduled; Keep Serving cancels it." : "")
        alert.addButton(withTitle: "Keep Serving")
        alert.addButton(withTitle: "Finish Reply, Then Stop")
        alert.addButton(withTitle: "Stop Now")
        switch alert.runModal() {
        // Keep Serving leaves a running compare alone; the stop choices take over.
        case .alertSecondButtonReturn: takeOverFromCompare(); stopWhenIdle()
        case .alertThirdButtonReturn: takeOverFromCompare(); stopServer()
        default: cancelPendingStop()
        }
    }
    /// Stops once no reply is streaming (up to ten minutes), so connected
    /// apps get their current answer. New requests are still accepted meanwhile.
    @Published private(set) var stopPending = false
    /// Replaced to disarm a scheduled stop.
    private var stopToken = UUID()
    private func cancelPendingStop() { stopToken = UUID(); stopPending = false }
    func prepareForQuit() {
        if compare.running { compareTakenOver = true; compare.cancel() }
        cancelPendingStop()
    }
    /// "Finish Reply, Then Stop". Internal so source-build QA can drive it.
    func stopWhenIdle() {
        guard !stopPending else { return }
        stopPending = true
        let server = runtime.startedAt, token = UUID()
        stopToken = token
        Task {
            defer { if stopToken == token { stopPending = false } }
            let deadline = Date().addingTimeInterval(600)
            while Date() < deadline && stopToken == token && runtime.owned && runtime.startedAt == server && inferenceActive {
                try? await Task.sleep(for: .milliseconds(500))
            }
            if stopToken == token && runtime.owned && runtime.startedAt == server { stopServer() }
        }
    }
    func requestStartOrRestart() {
        takeOverFromCompare()
        cancelPendingStop()
        if runtime.owned {
            guard canRestartServer else { return }
            if inferenceActive && !confirm("Restart the server?", "A reply is being generated. Restarting ends it.", action: "Restart") { return }
            restartServer()
        } else if canStartServer {
            startServer()
        } else {
            page = .home
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
    /// serving, and show Home where the server is controlled. A running
    /// server is never replaced silently; Home offers the restart.
    func serve(_ path: String) {
        guard models.canSelectInGUI(path), !attachedToService, !connecting else { return }
        takeOverFromCompare()
        selectedPath = path
        page = .home
        if canStartServer { startServer() }
    }

    /// Keep a startable model selected so Start works in one click: the
    /// running model while the app owns a server, otherwise the most recently
    /// served, a favorite, or the first downloaded model. An explicit valid
    /// choice is never replaced.
    /// While the library scan runs, only a recent or favorite model that is
    /// already confirmed servable is picked; the first-model fallback waits
    /// for the full list so it does not pick an arbitrary early result.
    func preselectModel() {
        guard selectedPath.isEmpty || !models.canSelectInGUI(selectedPath) else { return }
        if runtime.owned, models.canSelectInGUI(runtime.selectedModel) { selectedPath = runtime.selectedModel; return }
        if models.scanning {
            if let known = ModelPicker.preferred(models: servableModels, favorites: favoriteModels, recent: recentModels),
               recentModels.contains(known.path) || favoriteModels.contains(known.path) {
                selectedPath = known.path
            }
            return
        }
        if let preferred = ModelPicker.preferred(models: servableModels, favorites: favoriteModels, recent: recentModels) {
            selectedPath = preferred.path
        }
    }

    func revertSettings() {
        settingsDraft = settings; settingsError = nil
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: ApplicationModel?
    private let quitCoordinator = AppQuitCoordinator()
    static let quitCountdownSeconds = 30

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model else { return .terminateNow }
        return quitCoordinator.request(
            needsConfirmation: model.quitWarning != nil,
            confirm: {
                let alert = NSAlert()
                alert.messageText = "Quit AX Engine?"
                alert.informativeText = model.quitWarning ?? ""
                // 12-states.svg "Quit while serving": the menu bar keeps the server reachable.
                let canKeep = model.runtime.owned && model.showsMenuBarItem
                if canKeep { alert.addButton(withTitle: "Keep Serving in Menu Bar") }
                alert.addButton(withTitle: canKeep ? "Stop Server and Quit" : "Stop and Quit")
                alert.addButton(withTitle: "Cancel")
                // Unattended quits (logout, scripts, nobody at the Mac) must
                // not hold the model in memory forever: quit after 30 s unless
                // someone chooses Cancel or Keep Serving.
                let quitResponse: NSApplication.ModalResponse = canKeep ? .alertSecondButtonReturn : .alertFirstButtonReturn
                let countdown = QuitCountdown(seconds: Self.quitCountdownSeconds, response: quitResponse,
                                      action: model.runtime.owned ? "Stopping the server and quitting" : "Quitting")
                alert.accessoryView = countdown.label
                countdown.start()
                let choice = alert.runModal()
                countdown.stop()
                if canKeep && choice == .alertFirstButtonReturn { NSApp.hide(nil); return false }
                return choice == (canKeep ? .alertSecondButtonReturn : .alertFirstButtonReturn)
            }, shutdown: {
                // No compare restart or scheduled stop may act after this.
                model.prepareForQuit()
                await model.hostMonitor.shutdown()
                await model.storageMonitor.shutdown()
                await model.chat.stop()
                await model.models.shutdown()
                await model.runtime.stop()
            }, reply: { sender.reply(toApplicationShouldTerminate: true) })
    }
}

/// The quit dialog's countdown: a label that ticks once a second and, at
/// zero, ends the modal alert with the quit response.
@MainActor
final class QuitCountdown {
    let label: NSTextField
    private let seconds: Int
    /// System uptime, which a wall-clock change does not move.
    private var deadline: TimeInterval = 0
    private let response: NSApplication.ModalResponse
    private var timer: Timer?
    /// Keeps App Nap from stretching the timer while the app is in the background.
    private var activity: NSObjectProtocol?

    /// What happens at zero, named on the label.
    private let action: String

    init(seconds: Int, response: NSApplication.ModalResponse, action: String) {
        self.seconds = seconds
        self.action = action
        self.response = response
        label = NSTextField(labelWithString: "")
        label.font = .systemFont(ofSize: NSFont.smallSystemFontSize)
        label.textColor = .secondaryLabelColor
        label.frame = NSRect(x: 0, y: 0, width: 300, height: 18)
        label.stringValue = "\(action) in \(seconds) s…"
    }

    private var remaining: Int { max(0, Int((deadline - ProcessInfo.processInfo.systemUptime).rounded(.up))) }

    func start() {
        deadline = ProcessInfo.processInfo.systemUptime + TimeInterval(seconds)
        activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Quit countdown")
        render()
        // Ticks often; the label and the deadline come from the clock, not the tick count.
        let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        }
        // .modalPanel keeps it firing while runModal() owns the run loop.
        RunLoop.main.add(timer, forMode: .modalPanel)
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stop() {
        timer?.invalidate(); timer = nil
        if let activity { ProcessInfo.processInfo.endActivity(activity) }
        activity = nil
    }

    private func tick() {
        render()
        if remaining == 0 { stop(); NSApp.stopModal(withCode: response) }
    }

    private func render() {
        let text = "\(action) in \(remaining) s…"
        if label.stringValue != text { label.stringValue = text }
    }
}

@main
struct AXEngineApplication: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = ApplicationModel()
    var body: some Scene {
        Window("AX Engine", id: "main") {
            MainView(model: model, runtime: model.runtime, models: model.models, chat: model.chat)
                .tint(Brand.accent)
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
        .menuBarExtraStyle(.window)
        Settings {
            SettingsView(model: model)
                .tint(Brand.accent)
                .frame(width: 720, height: 620)
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
            Button("Command Palette…") { model.showsPalette = true; openWindow(id: "main") }
                .keyboardShortcut("k", modifiers: .command)
            Button(model.showsTelemetry ? "Hide Telemetry" : "Show Telemetry") { model.showsTelemetry.toggle() }
                .keyboardShortcut("i", modifiers: [.command, .option])
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

/// Status glyph in the menu bar, plus the live token rate while the
/// runtime reports one (Settings can hide the item entirely).
struct MenuBarLabel: View {
    @ObservedObject var runtime: RuntimeController
    var body: some View {
        let rate = runtime.ready ? runtime.history.latest(RuntimeController.tokenRateKey) : nil
        HStack(spacing: 3) {
            Image(systemName: runtime.ready ? "bolt.horizontal.circle.fill"
                  : ["Starting", "Stopping", "Connecting"].contains(runtime.state) ? "hourglass.circle" : "bolt.horizontal.circle")
            if let rate, rate >= 1 { Text(TokenRateFormat.short(rate)).monospacedDigit() }
        }
        .accessibilityLabel("AX Engine, " + runtime.state)
    }
}

/// Menu-bar panel (`10-menubar-and-command-palette.svg`): state, the model
/// it serves, live speed, one-click lifecycle and model switching.
struct MenuBarContent: View {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    private var selected: LocalModel? { model.models.productLibrary.first { $0.path == model.selectedPath } }
    private var running: LocalModel? { model.models.productLibrary.first { $0.path == runtime.selectedModel } }
    private var kind: ServerStatusKind { .from(state: runtime.state, owned: runtime.owned, ready: runtime.ready) }

    var body: some View {
        let now = Date()
        let rate = runtime.history.latest(RuntimeController.tokenRateKey, now: now)
        let acceptance = runtime.history.latest("mtpAcceptedPercent", now: now)
        let shown = runtime.owned ? running : runtime.connection == nil ? selected : nil
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                StatusPill(text: kind == .serving ? "Serving" : kind == .external ? "Attached" : runtime.state, tint: Brand.tint(kind))
                Spacer()
                if let url = runtime.connection?.baseURL { Text(url.host.map { host in url.port.map { "\(host):\($0)" } ?? host } ?? "")
                    .font(.system(size: 11, design: .monospaced)).foregroundStyle(Brand.text3) }
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(shown?.label.title ?? runtime.models.first?.id ?? "No model selected").font(.system(size: 16, weight: .bold)).lineLimit(1)
                Text(([shown?.label.tags.joined(separator: " · ")].compactMap { $0 } + [Format.uptime(since: runtime.connection == nil ? nil : runtime.startedAt).map { "up " + $0 }].compactMap { $0 })
                        .filter { !$0.isEmpty }.joined(separator: " · "))
                    .font(.system(size: 11.5)).foregroundStyle(Brand.text3)
            }
            if runtime.connection != nil {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(Format.rate(rate) ?? "–").font(.system(size: 24, weight: .bold)).monospacedDigit()
                        Text("tok/s").font(.system(size: 16, weight: .medium)).foregroundStyle(Brand.text2)
                        Spacer()
                        if let acceptance { Text(String(format: "Draft %.0f%%", acceptance)).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(Brand.boost) }
                    }
                    Sparkline(points: runtime.history.plotPoints(RuntimeController.tokenRateKey, window: 300, now: now),
                              start: now.addingTimeInterval(-300), end: now).frame(height: 32)
                }
                .padding(12)
                .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
            }
            HStack(spacing: 8) {
                if runtime.owned {
                    Button(runtime.state == "Starting" ? "Cancel Start" : "Stop Server", systemImage: "stop.fill") { model.requestStopServer() }
                        .buttonStyle(.ax(.danger)).disabled(runtime.state == "Stopping").frame(maxWidth: .infinity)
                    Button("Restart", systemImage: "arrow.clockwise") { model.requestStartOrRestart() }
                        .buttonStyle(.ax(.secondary)).disabled(!model.canRestartServer).frame(maxWidth: .infinity)
                } else if runtime.connection != nil || model.connecting {
                    Button(model.connecting ? "Cancel Connecting" : "Disconnect") { model.requestStopServer() }.buttonStyle(.ax(.secondary))
                } else {
                    Button("Start Server", systemImage: "play.fill") { model.startServer() }
                        .buttonStyle(.ax(.serve)).disabled(!model.canStartServer)
                }
            }
            let candidates = model.servableModels.filter { $0.path != runtime.selectedModel }
            let others = candidates.prefix(6)
            if !others.isEmpty && !model.attachedToService {
                SectionLabel(runtime.owned ? "Switch model" : "Serve a model")
                    .help(runtime.owned ? "Selects the model and restarts the server on it." : "Starts the server on the model.")
                VStack(spacing: 1) {
                    ForEach(Array(others)) { item in
                        Button {
                            model.serve(item.path)
                            // serve() never replaces a running server; here the row says Switch.
                            if runtime.owned { model.requestStartOrRestart() }
                            openMain()
                        } label: {
                            HStack {
                                Circle().strokeBorder(Brand.text3, lineWidth: 1.2).frame(width: 7, height: 7)
                                Text(item.label.title).font(.system(size: 12.5, weight: .medium)).lineLimit(1)
                                Spacer()
                                Text(item.sizeBytes.map(TransferFormat.bytes) ?? "").font(.system(size: 11)).foregroundStyle(Brand.text3)
                            }
                            .padding(.horizontal, 8).padding(.vertical, 6).contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(runtime.busy || model.models.scanning || model.models.removing || !model.models.canSelectInGUI(item.path))
                    }
                    if candidates.count > others.count {
                        Button("All \(candidates.count) models…") { model.modelsSection = "library"; model.page = .models; openMain() }
                            .buttonStyle(.plain).font(.system(size: 12)).foregroundStyle(Brand.accent)
                            .padding(.horizontal, 8).padding(.vertical, 4)
                    }
                }
            }
            Divider().overlay(Brand.hairlineColor)
            VStack(spacing: 1) {
                action("Copy OpenAI base URL", "doc.on.doc", hint: nil) {
                    if let url = runtime.connection?.apiURL.absoluteString { copyText(url) }
                }.disabled(runtime.connection == nil)
                action("Open Playground", "bubble.left.and.bubble.right", hint: "⌘2") { model.page = .chat; openMain() }
                    .disabled(!runtime.ready)
                action("Open AX Engine", "house", hint: nil) { openMain() }
                action("Quit AX Engine", "power", hint: "⌘Q") { NSApp.terminate(nil) }.keyboardShortcut("q")
            }
        }
        .padding(16)
        .frame(width: 320)
    }

    private func openMain() { openWindow(id: "main"); NSApp.activate() }

    private func action(_ title: String, _ icon: String, hint: String?, perform: @escaping () -> Void) -> some View {
        Button(action: perform) {
            HStack(spacing: 10) {
                Image(systemName: icon).font(.system(size: 12)).frame(width: 16).foregroundStyle(Brand.text2)
                Text(title).font(.system(size: 12.5, weight: .medium))
                Spacer()
                if let hint { Text(hint).font(.system(size: 11)).foregroundStyle(Brand.text3) }
            }
            .padding(.horizontal, 6).padding(.vertical, 5).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
