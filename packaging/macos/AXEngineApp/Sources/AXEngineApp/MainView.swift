import SwiftUI
import AppKit
import AXEngineAppCore

/// Sidebar order is the ⌘1–⌘5 order. Settings is its own window (⌘,).
enum Page: String, CaseIterable, Identifiable {
    case home = "Home", chat = "Playground", models = "Models", api = "API & Logs", system = "System"
    var id: String { rawValue }
    var icon: String {
        switch self {
        case .home: "house"
        case .chat: "bubble.left.and.bubble.right"
        case .models: "square.stack.3d.up"
        case .api: "chevron.left.forwardslash.chevron.right"
        case .system: "gauge.with.dots.needle.67percent"
        }
    }
    var tileColor: Color {
        switch self {
        case .home: .indigo
        case .chat: .purple
        case .models: .orange
        case .api: .teal
        case .system: .gray
        }
    }
}

struct MainView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var chat: ChatStore
    @Environment(\.openSettings) private var openSettings
    @FocusState private var searchFocused: Bool
    @State private var clientClipboard = SensitiveClipboard()
    @State private var clientCopyNotice = ""

    var body: some View {
        NavigationSplitView {
            Sidebar(model: model, runtime: runtime, models: models)
                .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 280)
        } detail: {
            Group {
                switch model.page ?? .home {
                case .home: HomeView(model: model, models: models, runtime: runtime, chat: chat,
                                     clipboard: clientClipboard, copyNotice: $clientCopyNotice)
                case .chat: ChatView(chat: chat, runtime: runtime, model: model)
                case .models: ModelsPage(model: model, runtime: runtime, models: models, storage: model.storageMonitor, searchFocused: $searchFocused)
                case .api: ServerPage(model: model, runtime: runtime, models: models, chat: chat,
                                      clientCopyNotice: $clientCopyNotice, clientClipboard: clientClipboard)
                case .system: SystemView(model: model, models: models, monitor: model.hostMonitor, storage: model.storageMonitor)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(nsColor: .windowBackgroundColor))
            .navigationTitle((model.page ?? .home).rawValue)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    ServerCapsule(model: model, runtime: runtime, models: models)
                }
                ToolbarItemGroup(placement: .primaryAction) {
                    DownloadsToolbarButton(model: model, models: models)
                    Button { model.showsPalette = true } label: { Label("Command Palette", systemImage: "command") }
                        .help("Command palette (⌘K)")
                    if model.page == .chat {
                        Button { model.showsTelemetry.toggle() } label: { Label("Telemetry", systemImage: "sidebar.trailing") }
                            .help(model.showsTelemetry ? "Hide telemetry (⌥⌘I)" : "Show telemetry (⌥⌘I)")
                    }
                }
            }
        }
        .task {
            if !models.discoveryAttempted && !models.refreshing { await models.refresh(settings: model.settings) }
            model.offerOnboardingIfNeeded()
        }
        .onChange(of: models.scanning) { _, scanning in if !scanning { model.offerOnboardingIfNeeded() } }
        .onChange(of: model.settingsRequests) { _, _ in openSettings() }
        .sheet(isPresented: $model.showsOnboarding, onDismiss: { model.onboardingCompleted = true }) {
            OnboardingView(model: model, models: models, runtime: runtime) {
                model.onboardingCompleted = true; model.showsOnboarding = false
            }
        }
        .sheet(isPresented: $model.showsPalette) {
            CommandPalette(model: model, runtime: runtime, models: models)
        }
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
}

/// Edge-to-edge sidebar: pages with colored tiles, then pinned models that
/// serve in one click, then a compact host line.
private struct Sidebar: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore

    private var pinned: [LocalModel] {
        model.favoriteModels.compactMap { path in model.servableModels.first { $0.path == path } }
    }

    var body: some View {
        List(selection: $model.page) {
            Section {
                ForEach(Array(Page.allCases.enumerated()), id: \.element) { index, page in
                    HStack(spacing: 9) {
                        Image(systemName: page.icon)
                            .font(.system(size: 11, weight: .semibold)).foregroundStyle(.white)
                            .frame(width: 22, height: 22)
                            .background(page.tileColor.gradient, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                        Text(page.rawValue)
                        Spacer()
                        if page == .models, models.activeDownloadCount > 0 {
                            Text("\(models.activeDownloadCount)").font(.caption2.monospacedDigit())
                                .padding(.horizontal, 6).padding(.vertical, 1)
                                .background(.quaternary, in: Capsule())
                                .accessibilityLabel("\(models.activeDownloadCount) downloads")
                        }
                    }
                    .tag(page)
                    .help("\(page.rawValue) (⌘\(index + 1))")
                }
            }
            if !pinned.isEmpty {
                Section("Pinned") {
                    ForEach(pinned) { item in
                        let serving = runtime.owned && runtime.selectedModel == item.path
                        Button { model.serve(item.path) } label: {
                            HStack(spacing: 8) {
                                Circle().fill(serving ? Brand.serve : Color.secondary.opacity(0.35)).frame(width: 7, height: 7)
                                Text(item.displayName).lineLimit(1).truncationMode(.middle)
                            }
                        }
                        .buttonStyle(.plain)
                        .disabled(serving || model.attachedToService || model.connecting)
                        .help(serving ? "Serving" : runtime.owned ? "Select \(item.displayName); Home offers the restart"
                                                                  : "Serve \(item.displayName)")
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) {
            HStack(spacing: 8) {
                if let logo = Brand.logo { Image(nsImage: logo).resizable().frame(width: 20, height: 20) }
                VStack(alignment: .leading, spacing: 1) {
                    Text("AX Engine").font(.caption.weight(.semibold))
                    Text(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)
                         + " unified memory").font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 16).padding(.vertical, 12)
        }
    }
}

/// The single server control shown on every page: state, model, endpoint
/// and the one lifecycle action that applies. Restart lives in the menu
/// (⇧⌘R) and on Home; Stop Server is ⇧⌘.
struct ServerCapsule: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @State private var copied = false

    private var kind: ServerStatusKind { .from(state: runtime.state, owned: runtime.owned, ready: runtime.ready) }
    private var title: String {
        if runtime.owned || runtime.busy {
            if let item = models.productLibrary.first(where: { $0.path == runtime.selectedModel }) { return item.displayName }
            if !runtime.selectedModel.isEmpty { return URL(fileURLWithPath: runtime.selectedModel).lastPathComponent }
        }
        if runtime.connection != nil { return runtime.models.first?.id ?? "External service" }
        if let item = models.productLibrary.first(where: { $0.path == model.selectedPath }) { return item.displayName }
        return model.servableModels.isEmpty ? "No model downloaded" : "No model selected"
    }

    var body: some View {
        HStack(spacing: 10) {
            Circle().fill(Brand.tint(kind)).frame(width: 8, height: 8)
                .overlay { if kind == .transition { ProgressView().controlSize(.mini).scaleEffect(0.6) } }
            VStack(alignment: .leading, spacing: 0) {
                Text(title).font(.callout.weight(.semibold)).lineLimit(1).truncationMode(.middle)
                Text(subtitle).font(.caption2.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
            }
            .frame(minWidth: 140, maxWidth: 300, alignment: .leading)
            if let url = runtime.connection?.apiURL.absoluteString {
                Button {
                    copyText(url); copied = true
                    Task { try? await Task.sleep(for: .seconds(2)); copied = false }
                } label: { Image(systemName: copied ? "checkmark" : "doc.on.doc") }
                .buttonStyle(.borderless).help(copied ? "Copied" : "Copy API URL " + url)
                .accessibilityLabel("Copy API URL")
            }
            action
        }
        .padding(.leading, 12).padding(.trailing, 6).padding(.vertical, 3)
        .help(runtime.error ?? ("Server: " + runtime.state))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Server, " + runtime.state + ", " + title)
    }

    private var subtitle: String {
        switch kind {
        case .serving, .external:
            let rate = runtime.history.latest(RuntimeController.tokenRateKey)
            let endpoint = runtime.connection.map { "127.0.0.1:" + String($0.baseURL.port ?? model.settings.port) } ?? ""
            return [kind.label, endpoint, rate.map(TokenRateFormat.short) ?? ""].filter { !$0.isEmpty }.joined(separator: " · ")
        case .transition: return runtime.state + "…"
        case .degraded, .failed: return runtime.state
        case .stopped: return "Stopped · 127.0.0.1:" + String(model.settings.port)
        }
    }

    private var action: some View {
        actionButton.labelStyle(.titleAndIcon)
    }

    @ViewBuilder private var actionButton: some View {
        if runtime.owned {
            Button(runtime.state == "Starting" ? "Cancel" : "Stop", systemImage: "stop.fill") { model.requestStopServer() }
                .controlSize(.small).disabled(runtime.state == "Stopping")
                .help(runtime.state == "Starting" ? "Cancel the launch (⇧⌘.)" : "Stop Server (⇧⌘.)")
        } else if runtime.connection != nil || model.connecting {
            Button(model.connecting ? "Cancel" : "Disconnect") { model.requestStopServer() }
                .controlSize(.small).help("Stop using the attached service. It keeps running.")
        } else {
            Button("Start", systemImage: "play.fill") { model.requestStartOrRestart() }
                .controlSize(.small).buttonStyle(.borderedProminent)
                .disabled(!model.canStartServer)
                .help(model.startBlockReason ?? "Start Server (⇧⌘R)")
        }
    }
}

/// Downloads in progress, one click from any page.
private struct DownloadsToolbarButton: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @State private var open = false
    var body: some View {
        let active = models.downloads.filter(\.isActive)
        Button { open.toggle() } label: {
            Label("Downloads", systemImage: active.isEmpty ? "arrow.down.circle" : "arrow.down.circle.fill")
        }
        .help(active.isEmpty ? "Downloads" : "\(active.count) downloading")
        .popover(isPresented: $open, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Downloads").font(.headline)
                if models.downloads.isEmpty {
                    Text("Nothing downloading. Find models in the Catalog.").font(.callout).foregroundStyle(.secondary)
                } else {
                    ForEach(models.downloads.prefix(6)) { item in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(item.model.repoId).font(.callout).lineLimit(1).truncationMode(.middle)
                                Spacer()
                                Text(item.state).font(.caption).foregroundStyle(.secondary)
                            }
                            if item.isActive {
                                if let fraction = item.fraction { ProgressView(value: fraction) } else { ProgressView().progressViewStyle(.linear) }
                            }
                        }
                    }
                }
                Divider()
                HStack {
                    Button("Open Catalog") { open = false; model.catalogSearch = ""; model.modelsSection = "catalog"; model.page = .models }
                    Spacer()
                    Button("Show All") { open = false; model.modelsSection = "transfers"; model.page = .models }
                        .buttonStyle(.borderedProminent)
                }
            }
            .padding(16).frame(width: 340)
        }
    }
}

struct ErrorBanner: View {
    let text: String
    var body: some View {
        Label(text, systemImage: "exclamationmark.triangle")
            .font(.callout).foregroundStyle(Brand.warn).textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading).padding(12)
            .background(Brand.warn.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }
}

@MainActor
func copyText(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
}
