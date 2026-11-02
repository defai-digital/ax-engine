import SwiftUI
import AppKit
import AXEngineAppCore

enum Page: String, CaseIterable, Identifiable {
    case overview = "Dashboard", models = "Models", server = "Server", chat = "Playground", settings = "Settings"
    var id: String { rawValue }
    var icon: String {
        switch self {
        case .overview: "square.grid.2x2"
        case .models: "square.stack.3d.up"
        case .server: "server.rack"
        case .chat: "bubble.left.and.bubble.right"
        case .settings: "gearshape"
        }
    }
}

struct MainView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var chat: ChatStore
    @FocusState private var searchFocused: Bool
    @State private var clientClipboard = SensitiveClipboard()
    @State private var clientCopyNotice = ""

    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 10) {
                    if let logo = Brand.logo { Image(nsImage: logo).resizable().frame(width: 44, height: 44) }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("AX Engine").font(.headline)
                        Text("Local inference").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.horizontal, 14).padding(.top, 20)
                List(Page.allCases, selection: $model.page) { item in
                    Label(item.rawValue, systemImage: item.icon).tag(item)
                        .padding(.vertical, 5)
                }.listStyle(.sidebar)
                VStack(alignment: .leading, spacing: 10) {
                    PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : "circle", tint: runtime.ready ? Brand.accentChrome : .secondary)
                    Text("Apple Silicon · \(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)) unified memory")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding(18)
            }
            .navigationSplitViewColumnWidth(min: 210, ideal: 230, max: 280)
        } detail: {
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 5) {
                        Text((model.page ?? .overview).rawValue).font(.system(size: 27, weight: .semibold))
                        Text(subtitle).font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if runtime.ready && model.page != .chat { Button("Open Playground", systemImage: "bubble.left") { model.page = .chat } }
                }.padding(26)
                if [.overview, .server, .chat].contains(model.page) {
                    ModelProfileHeader(model: model, runtime: runtime, chat: chat)
                        .padding(.horizontal, 26).padding(.bottom, 16)
                }
                Divider()
                Group {
                    switch model.page ?? .overview {
                    case .overview: DashboardView(model: model, models: models, monitor: model.hostMonitor, storage: model.storageMonitor)
                    case .models: ModelsPage(model: model, runtime: runtime, models: models, storage: model.storageMonitor, searchFocused: $searchFocused)
                    case .server: ServerPage(model: model, runtime: runtime, models: models, chat: chat,
                                             clientCopyNotice: $clientCopyNotice, clientClipboard: clientClipboard)
                    case .chat: ChatView(chat: chat, runtime: runtime, settings: model.settings, openServer: { model.page = .server })
                    case .settings: SettingsView(model: model)
                    }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .task { if !models.discoveryAttempted && !models.refreshing { await models.refresh(settings: model.settings) } }
        .onAppear { syncHostMonitoring() }
        .onChange(of: model.hostMonitoringEnabled) { _, _ in syncHostMonitoring() }
        .onDisappear { model.hostMonitor.setActive(false); clientClipboard.clearIfOwned() }
        .onChange(of: runtime.ready) { _, ready in
            if !ready { clientClipboard.clearIfOwned(); clientCopyNotice = "" }
        }
        .onChange(of: runtime.connection?.id) { _, _ in
            clientClipboard.clearIfOwned()
            clientCopyNotice = ""
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            clientClipboard.clearIfOwned()
        }
        .onChange(of: model.searchFocusRequest) { _, _ in
            Task { @MainActor in await Task.yield(); searchFocused = true }
        }
    }

    private func syncHostMonitoring() {
        model.hostMonitor.setActive(HostMonitor.shouldSample(enabled: model.hostMonitoringEnabled))
    }
    private var subtitle: String {
        switch model.page ?? .overview {
        case .overview: "Memory, storage and the models on your Mac."
        case .models: "Download public AutomatosX models, manage local folders and follow transfers."
        case .server: "Start a local model in the serving mode that fits your workload, or connect to an existing AX Engine service."
        case .chat: "Test your model’s responses and inference speed."
        case .settings: "Configure your local runtime and generation defaults."
        }
    }
}

struct ErrorBanner: View {
    let text: String
    var body: some View {
        Label(text, systemImage: "exclamationmark.triangle")
            .font(.callout).foregroundStyle(.orange).textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading).padding(12)
            .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }
}

@MainActor
func copyText(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}
