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
    @Published var modelsSection = "library"
    @Published var searchFocusRequest = 0
    @Published var dashboardHistoryWindow: Double = 60
    @Published var hostMonitoringEnabled = UserDefaults.standard.bool(forKey: "hostMonitoringEnabled") {
        didSet { UserDefaults.standard.set(hostMonitoringEnabled, forKey: "hostMonitoringEnabled") }
    }
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
