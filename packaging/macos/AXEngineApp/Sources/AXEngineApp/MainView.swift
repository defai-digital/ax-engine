import SwiftUI
import AppKit
import AXEngineAppCore

/// Sidebar order is the ⌘1–⌘5 order. Settings is its own window (⌘,).
enum Page: String, CaseIterable, Identifiable {
    case home = "Home", chat = "Playground", models = "Models", api = "API & Logs", system = "System"
    var id: String { rawValue }
    static var launchPage: Page? {
        switch UserDefaults.standard.string(forKey: "AXOpenPage") {
        case "home": .home
        case "chat": .chat
        case "models": .models
        case "api": .api
        case "system": .system
        default: nil
        }
    }
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
                .navigationSplitViewColumnWidth(min: 220, ideal: 236, max: 290)
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
            .background(Brand.canvas)
            .navigationTitle((model.page ?? .home).rawValue)
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    ServerCapsule(model: model, runtime: runtime, models: models)
                }
                .sharedBackgroundVisibility(.hidden)
                ToolbarItem(placement: .primaryAction) {
                    PaletteField { model.showsPalette = true }
                }
                .sharedBackgroundVisibility(.hidden)
                ToolbarItem(placement: .primaryAction) {
                    DownloadsToolbarButton(model: model, models: models)
                }
                .sharedBackgroundVisibility(.hidden)
                if model.page == .chat {
                    ToolbarItem(placement: .primaryAction) {
                        ToolbarCircleButton(systemImage: "sidebar.trailing", active: model.showsTelemetry,
                                            help: model.showsTelemetry ? "Hide telemetry (⌥⌘I)" : "Show telemetry (⌥⌘I)") {
                            model.showsTelemetry.toggle()
                        }
                    }
                    .sharedBackgroundVisibility(.hidden)
                }
            }
            .toolbarBackground(Brand.canvas, for: .windowToolbar)
        }
        .task {
            if !models.discoveryAttempted && !models.refreshing { await models.refresh(settings: model.settings) }
            model.offerOnboardingIfNeeded()
        }
        .onChange(of: models.scanning) { _, scanning in if !scanning { model.offerOnboardingIfNeeded(); model.preselectModel() } }
        .onChange(of: model.servableModels.map(\.path)) { _, _ in model.preselectModel() }
        .onAppear { model.preselectModel() }
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

/// Edge-to-edge sidebar from the boards: brand header, pages with colored
/// tiles and their ⌘ shortcuts, pinned models that serve in one click, and
/// Settings at the foot.
private struct Sidebar: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @Environment(\.openSettings) private var openSettings

    private var pinned: [LocalModel] {
        model.favoriteModels.compactMap { path in model.servableModels.first { $0.path == path } }
    }
    /// Without favorites, the most recently served models stand in.
    private var shortcuts: (title: String, items: [LocalModel]) {
        if !pinned.isEmpty { return ("Pinned", pinned) }
        let recent = model.recentModels.compactMap { path in model.servableModels.first { $0.path == path } }
        return ("Recent", Array((recent.isEmpty ? model.servableModels : recent).prefix(4)))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header.padding(.horizontal, 18).padding(.top, 6).padding(.bottom, 18)
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(Page.allCases.enumerated()), id: \.element) { index, page in
                        pageRow(page, shortcut: index + 1)
                    }
                    let list = shortcuts
                    SectionLabel(list.title).padding(.horizontal, 12).padding(.top, 18).padding(.bottom, 6)
                    ForEach(list.items) { item in pinnedRow(item) }
                    Button {
                        model.modelSearch = ""; model.catalogSearch = ""
                        model.modelsSection = model.servableModels.isEmpty ? "catalog" : "library"
                        model.page = .models
                    } label: {
                        Text(model.servableModels.isEmpty ? "+ Download a model" : "+ Pin a model").font(.system(size: 12))
                            .foregroundStyle(Brand.text3).padding(.leading, 30).padding(.vertical, 5)
                            .frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help(model.servableModels.isEmpty ? "Find a model in the Catalog" : "Star a model in the model picker or Models to pin it here")
                }
                .padding(.horizontal, 10)
            }
            Spacer(minLength: 0)
            Button { openSettings() } label: {
                HStack(spacing: 9) {
                    Image(systemName: "gearshape").font(.system(size: 13))
                    Text("Settings").font(.system(size: 13))
                    Spacer()
                    Text("⌘,").font(.system(size: 11)).foregroundStyle(Brand.text3)
                }
                .foregroundStyle(Brand.text2).padding(.horizontal, 12).padding(.vertical, 8).contentShape(Rectangle())
            }
            .buttonStyle(.plain).padding(.horizontal, 10).padding(.bottom, 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .clipped()
        .background(Brand.sidebar)
    }

    private var header: some View {
        HStack(spacing: 10) {
            Text("AX").font(.system(size: 12, weight: .heavy)).foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(LinearGradient(colors: [Brand.accent, Brand.boost], startPoint: .topLeading, endPoint: .bottomTrailing),
                            in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text("AX Engine").font(.system(size: 14, weight: .bold)).foregroundStyle(Brand.text1)
                Text(HostInfo.summary).font(.system(size: 11)).foregroundStyle(Brand.text3).lineLimit(1).truncationMode(.tail)
            }
        }
    }

    private func pageRow(_ page: Page, shortcut: Int) -> some View {
        let selected = (model.page ?? .home) == page
        return Button { model.page = page } label: {
            HStack(spacing: 10) {
                Image(systemName: page.icon)
                    .font(.system(size: 11, weight: .semibold)).foregroundStyle(.white)
                    .frame(width: 22, height: 22)
                    .background(page.tileColor.gradient, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                Text(page.rawValue).font(.system(size: 13, weight: selected ? .semibold : .regular))
                    .foregroundStyle(Brand.text1)
                Spacer()
                if page == .models, models.activeDownloadCount > 0 {
                    Text("\(models.activeDownloadCount)").font(.system(size: 10.5, weight: .semibold).monospacedDigit())
                        .foregroundStyle(.white).padding(.horizontal, 6).padding(.vertical, 1)
                        .background(Brand.accent, in: Capsule())
                        .accessibilityLabel("\(models.activeDownloadCount) downloads")
                }
                Text("⌘\(shortcut)").font(.system(size: 11)).foregroundStyle(Brand.text3)
            }
            .padding(.horizontal, 8).padding(.vertical, 6)
            .background(selected ? Brand.control : .clear, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .help("\(page.rawValue) (⌘\(shortcut))")
    }

    private func pinnedRow(_ item: LocalModel) -> some View {
        let serving = runtime.owned && runtime.selectedModel == item.path
        return Button { model.serve(item.path) } label: {
            HStack(spacing: 10) {
                Circle().strokeBorder(serving ? Brand.serve : Brand.text3, lineWidth: serving ? 0 : 1.2)
                    .background(Circle().fill(serving ? Brand.serve : .clear))
                    .frame(width: 7, height: 7).frame(width: 22)
                Text(item.label.title).font(.system(size: 12.5, weight: serving ? .semibold : .regular))
                    .foregroundStyle(Brand.text1).lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8).padding(.vertical, 5).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(serving || model.attachedToService || model.connecting)
        .help(serving ? "Serving" : runtime.owned ? "Select \(item.displayName); Home offers the restart"
                                                  : "Serve \(item.displayName)")
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
        HStack(spacing: 9) {
            Circle().fill(Brand.tint(kind)).frame(width: 8, height: 8)
                .overlay { if kind == .transition { ProgressView().controlSize(.mini).scaleEffect(0.55) } }
            Text(title).font(.system(size: 12.5, weight: .semibold)).foregroundStyle(Brand.text1)
                .lineLimit(1).truncationMode(.middle).frame(maxWidth: 220, alignment: .leading)
            Text(subtitle).font(.system(size: 12).monospacedDigit()).foregroundStyle(Brand.text2).lineLimit(1)
            if let url = runtime.connection?.apiURL.absoluteString {
                Button {
                    copyText(url); copied = true
                    Task { try? await Task.sleep(for: .seconds(2)); copied = false }
                } label: { Image(systemName: copied ? "checkmark" : "doc.on.doc").font(.system(size: 10.5)) }
                .buttonStyle(.plain).foregroundStyle(Brand.text3).help(copied ? "Copied" : "Copy API URL " + url)
                .accessibilityLabel("Copy API URL")
            }
            action
        }
        .padding(.leading, 13).padding(.trailing, 4).padding(.vertical, 4)
        .background(Brand.panel, in: Capsule())
        .overlay(Capsule().strokeBorder(Brand.hairlineColor))
        .help(runtime.error ?? ("Server: " + runtime.state))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Server, " + runtime.state + ", " + title)
    }

    private var subtitle: String {
        switch kind {
        case .serving, .external:
            let rate = runtime.history.latest(RuntimeController.tokenRateKey)
            let endpoint = runtime.connection.map { "127.0.0.1:" + String($0.baseURL.port ?? model.settings.port) } ?? ""
            return [endpoint, rate.map(TokenRateFormat.short) ?? ""].filter { !$0.isEmpty }.joined(separator: " · ")
        case .transition: return runtime.state + "…"
        case .degraded, .failed: return runtime.state
        case .stopped: return "Not serving"
        }
    }

    private var action: some View {
        actionButton.labelStyle(.titleAndIcon)
    }

    @ViewBuilder private var actionButton: some View {
        if runtime.owned {
            circle(systemImage: "stop.fill", tint: Brand.danger,
                   help: runtime.state == "Starting" ? "Cancel the launch (⇧⌘.)" : "Stop Server (⇧⌘.)") { model.requestStopServer() }
                .disabled(runtime.state == "Stopping")
                .accessibilityLabel(runtime.state == "Starting" ? "Cancel start" : "Stop server")
        } else if runtime.connection != nil || model.connecting {
            circle(systemImage: "xmark", tint: Brand.boost, help: "Stop using the attached service. It keeps running.") { model.requestStopServer() }
                .accessibilityLabel(model.connecting ? "Cancel connecting" : "Disconnect")
        } else {
            circle(systemImage: "play.fill", tint: Brand.serve, help: model.startBlockReason ?? "Start Server (⇧⌘R)") { model.requestStartOrRestart() }
                .disabled(!model.canStartServer)
                .accessibilityLabel("Start server")
        }
    }

    private func circle(systemImage: String, tint: Color, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage).font(.system(size: 8.5, weight: .bold)).foregroundStyle(tint)
                .frame(width: 22, height: 22).background(tint.opacity(0.18), in: Circle())
        }
        .buttonStyle(.plain).help(help)
    }
}

/// The toolbar's "Search or run a command ⌘K" field; it opens the palette.
private struct PaletteField: View {
    let open: () -> Void
    var body: some View {
        Button(action: open) {
            HStack(spacing: 7) {
                Image(systemName: "magnifyingglass").font(.system(size: 12))
                Text("Search or run a command").font(.system(size: 12.5))
                Spacer(minLength: 12)
                Text("⌘K").font(.system(size: 10.5, weight: .semibold)).foregroundStyle(Brand.text2)
                    .padding(.horizontal, 5).padding(.vertical, 1.5)
                    .background(Brand.control, in: RoundedRectangle(cornerRadius: 4, style: .continuous))
            }
            .foregroundStyle(Brand.text3)
            .padding(.leading, 10).padding(.trailing, 6).padding(.vertical, 5.5)
            .frame(width: 228)
            .background(Brand.panel, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Brand.hairlineColor))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Command palette (⌘K)")
        .accessibilityLabel("Search or run a command")
    }
}

/// Round toolbar button (downloads, inspector).
struct ToolbarCircleButton: View {
    let systemImage: String
    var active = false
    var badge: Double? = nil
    let help: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().strokeBorder(active ? Brand.accent : Brand.hairlineColor, lineWidth: 1)
                    .background(Circle().fill(active ? Brand.accent.opacity(0.16) : Brand.panel))
                if let badge {
                    Circle().trim(from: 0, to: max(0.03, min(1, badge))).stroke(Brand.accent, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                        .rotationEffect(.degrees(-90)).padding(1)
                }
                Image(systemName: systemImage).font(.system(size: 12, weight: .medium))
                    .foregroundStyle(active ? Brand.accent : Brand.text2)
            }
            .frame(width: 28, height: 28).contentShape(Circle())
        }
        .buttonStyle(.plain).help(help).accessibilityLabel(help)
    }
}

/// Downloads in progress, one click from any page.
private struct DownloadsToolbarButton: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @State private var open = false
    var body: some View {
        let active = models.downloads.filter(\.isActive)
        ToolbarCircleButton(systemImage: "arrow.down", active: !active.isEmpty,
                            badge: active.first(where: { $0.state == "Downloading" })?.fraction,
                            help: active.isEmpty ? "Downloads" : "\(active.count) downloading") { open.toggle() }
        .popover(isPresented: $open, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Downloads").font(Brand.Font.cardTitle)
                    Spacer()
                    Button("Show All") { open = false; model.modelsSection = "transfers"; model.page = .models }
                        .buttonStyle(.axLink)
                }
                if models.downloads.isEmpty {
                    Text("Nothing downloading. Find models in the Catalog.").font(Brand.Font.body).foregroundStyle(Brand.text2)
                } else {
                    ForEach(models.downloads.prefix(6)) { item in row(item) }
                }
                Divider()
                Button("Open Catalog", systemImage: "books.vertical") {
                    open = false; model.catalogSearch = ""; model.modelsSection = "catalog"; model.page = .models
                }
                .buttonStyle(.ax(.secondary))
            }
            .padding(18).frame(width: 360)
        }
    }

    @ViewBuilder private func row(_ item: DownloadItem) -> some View {
        let label = ModelLabel(repositoryName: item.model.repoId.components(separatedBy: "/").last ?? item.model.repoId)
        let failed = item.state == "Failed"
        if item.state == "Ready" {
            HStack(spacing: 8) {
                Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(Brand.serve)
                Text(label.text + " finished · verified").font(.system(size: 12)).foregroundStyle(Brand.text2).lineLimit(1)
                Spacer()
                Button("Serve") { open = false; model.serve(item.destination) }.buttonStyle(.axLink)
                    .disabled(!models.canSelectInGUI(item.destination) || model.attachedToService || model.connecting)
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                Text(label.text).font(.system(size: 12.5, weight: .semibold)).lineLimit(1).truncationMode(.middle)
                if item.isActive || failed {
                    StackedBar(segments: [.init(id: "done", value: item.fraction ?? 0.02,
                                                color: failed ? Brand.danger : Brand.accent)], total: 1, height: 5)
                }
                Text(failed ? item.detail : item.state == "Queued" ? "Queued · starts after the current download"
                     : [item.phaseTitle, item.progressSummary].compactMap { $0 }.joined(separator: " · "))
                    .font(.system(size: 11)).foregroundStyle(failed ? Brand.danger : Brand.text3).lineLimit(1).truncationMode(.middle)
            }
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
