import SwiftUI
import AppKit
import AXEngineAppCore

@MainActor
final class ApplicationModel: ObservableObject {
    @Published var settings = AppSettings()
    @Published var settingsError: String?
    let runtime = RuntimeController()
    let models = ModelStore()
    let chat = ChatStore()
    init() {
        do { settings = try AppSettings.load() }
        catch { settingsError = error.localizedDescription }
    }
    func saveSettings() {
        do { try settings.save(); settingsError = nil }
        catch { settingsError = error.localizedDescription }
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
        WindowGroup("AX Engine", id: "main") {
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
        }
        Settings { SettingsView(model: model).frame(width: 620).padding(28) }
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

struct ThirdPartyNoticeCommands: Commands {
    @Environment(\.openWindow) private var openWindow
    var body: some Commands {
        CommandGroup(after: .appInfo) {
            Button("Third-Party Notices") { openWindow(id: "third-party-notices") }
        }
    }
}
