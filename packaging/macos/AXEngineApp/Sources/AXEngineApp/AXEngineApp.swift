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
    @Published var libraryScope = "all"
    @Published var hostMonitoringEnabled = UserDefaults.standard.bool(forKey: "hostMonitoringEnabled") {
        didSet { UserDefaults.standard.set(hostMonitoringEnabled, forKey: "hostMonitoringEnabled") }
    }
    let hostMonitor = HostMonitor()
    let runtime = RuntimeController()
    let models = ModelStore()
    let chat = ChatStore()
    init() {
        do { settings = try AppSettings.load() }
        catch { settingsError = error.localizedDescription }
        settingsDraft = settings
    }
    var hasSettingsChanges: Bool { settingsDraft != settings }
    func saveSettings() -> Bool {
        do {
            try settingsDraft.save()
            settings = settingsDraft; settingsError = nil; settingsSaved = true
            return true
        } catch {
            settingsError = error.localizedDescription; settingsSaved = false
            return false
        }
    }
    func revertSettings() {
        settingsDraft = settings; settingsError = nil
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: ApplicationModel?
    private var quitting = false
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !quitting, let model else { return .terminateNow }
        if model.runtime.owned || model.models.downloading || model.models.removing || model.chat.generating {
            let alert = NSAlert()
            alert.messageText = "Quit AX Engine?"
            alert.informativeText = "App-owned inference, downloads and model operations will stop. Attached services will keep running."
            alert.addButton(withTitle: "Stop and Quit"); alert.addButton(withTitle: "Cancel")
            guard alert.runModal() == .alertFirstButtonReturn else { return .terminateCancel }
        }
        quitting = true
        Task {
            await model.hostMonitor.shutdown()
            await model.chat.stop()
            await model.models.shutdown()
            await model.runtime.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
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
                    if let logo = Brand.logo { NSApplication.shared.applicationIconImage = logo }
                }
        }
        .defaultSize(width: 1180, height: 800)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About AX Engine") { showAbout() }
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
        let credits = NSMutableAttributedString(string: "Native local inference for Apple Silicon.\n\nManage your models, run local inference and chat on your Mac.\n\nOpen Third-Party Notices from the AX Engine menu for open-source acknowledgements.")
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
