import SwiftUI
import AppKit
import AXEngineAppCore

private enum Page: String, CaseIterable, Identifiable {
    case overview = "Overview", models = "Models", downloads = "Downloads", server = "Server", chat = "Chat"
    var id: String { rawValue }
    var icon: String {
        switch self {
        case .overview: "square.grid.2x2"
        case .models: "square.stack.3d.up"
        case .downloads: "arrow.down.circle"
        case .server: "server.rack"
        case .chat: "bubble.left.and.bubble.right"
        }
    }
}

struct MainView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var chat: ChatStore
    @State private var page: Page? = .overview
    @State private var selectedPath = ""
    @State private var search = ""
    @State private var removeTarget: LocalModel?
    @State private var stopConfirmation = false
    @State private var apiKey = ""

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
                List(Page.allCases, selection: $page) { item in
                    Label(item.rawValue, systemImage: item.icon).tag(item)
                        .padding(.vertical, 5)
                }.listStyle(.sidebar)
                VStack(alignment: .leading, spacing: 10) {
                    PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : "circle", tint: runtime.ready ? Brand.accentChrome : .secondary)
                    Text("Apple Silicon · \(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)) unified memory")
                        .font(.caption).foregroundStyle(.secondary)
                    SettingsLink { Label("Settings", systemImage: "gearshape") }
                        .buttonStyle(.plain)
                }.padding(18)
            }
            .navigationSplitViewColumnWidth(min: 210, ideal: 230, max: 280)
        } detail: {
            VStack(spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 5) {
                        Text((page ?? .overview).rawValue).font(.system(size: 27, weight: .semibold))
                        Text(subtitle).font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if runtime.ready && page != .chat { Button("Open Chat", systemImage: "bubble.left") { page = .chat } }
                }.padding(26)
                Divider()
                Group {
                    switch page ?? .overview {
                    case .overview: overview
                    case .models: modelsPage
                    case .downloads: downloadsPage
                    case .server: serverPage
                    case .chat: ChatView(chat: chat, runtime: runtime, settings: model.settings)
                    }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .task { await models.refresh(settings: model.settings) }
        .alert("Remove downloaded model?", isPresented: Binding(get: { removeTarget != nil }, set: { if !$0 { removeTarget = nil } })) {
            Button("Remove", role: .destructive) {
                if let target = removeTarget, !runtime.owned, runtime.connection == nil, !runtime.busy {
                    Task { await models.remove(target, settings: model.settings) }
                }
                removeTarget = nil
            }
            Button("Cancel", role: .cancel) { removeTarget = nil }
        } message: { Text("This deletes the app-managed model folder. Hugging Face cache and imported folders are not deleted.") }
        .alert("Stop the local server?", isPresented: $stopConfirmation) {
            Button("Stop", role: .destructive) { Task { await chat.stop(); await runtime.stop() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("Active inference on this app-owned server will stop.") }
    }

    private var subtitle: String {
        switch page ?? .overview {
        case .overview: "Your models. Your Mac. Your control."
        case .models: "Browse runtime profiles and manage local model folders."
        case .downloads: "A resumable queue powered by the AX Engine downloader."
        case .server: "Start a local model or connect to an existing AX Engine service."
        case .chat: "Private conversations with your running model."
        }
    }

    private var overview: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HStack(spacing: 25) {
                    if let logo = Brand.logo { Image(nsImage: logo).resizable().frame(width: 116, height: 116) }
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Make room for your next idea.").font(.system(size: 29, weight: .semibold))
                        Text("Choose a model, start the engine, and keep the conversation on your Mac.")
                            .foregroundStyle(.secondary)
                        HStack {
                            Button("Browse Models", systemImage: "square.stack.3d.up") { page = .models }.buttonStyle(.borderedProminent)
                            Button("Connect a Server") { page = .server }
                        }
                    }
                }.padding(.vertical, 12)
                if let error = model.settingsError { ErrorBanner(text: error) }
                HStack(alignment: .top, spacing: 16) {
                    Card("This Mac") {
                        MetricRow(label: "Unified memory", value: ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory))
                        MetricRow(label: "Processor cores", value: String(ProcessInfo.processInfo.processorCount))
                        MetricRow(label: "Thermal state", value: thermalLabel)
                    }
                    Card("Your workspace") {
                        MetricRow(label: "Local models", value: String(models.library.count))
                        MetricRow(label: "Server", value: runtime.state)
                        MetricRow(label: "Active downloads", value: String(models.downloads.filter { ["Queued", "Downloading"].contains($0.state) }.count))
                    }
                }
                Card("Get started") {
                    Label("1. Choose a model from the runtime catalog, or add a local folder.", systemImage: "square.stack.3d.up")
                    Label("2. Start its server. Model loading can take a few minutes.", systemImage: "power")
                    Label("3. Chat here, or copy the API endpoint into another client.", systemImage: "bubble.left")
                }
                Text("CLI and TUI remain available. Model compatibility, memory admission, MTP and certification are determined by the AX Engine runtime.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(26)
        }
    }
    private var thermalLabel: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unavailable"
        }
    }

    private var modelsPage: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                if let error = models.error { ErrorBanner(text: error) }
                HStack {
                    TextField("Search models", text: $search).textFieldStyle(.roundedBorder)
                    Button("Refresh", systemImage: "arrow.clockwise") { Task { await models.refresh(settings: model.settings) } }.disabled(models.refreshing)
                    Button("Add Local Folder", systemImage: "folder.badge.plus") { importFolder() }
                }
                Card("Local library", subtitle: "Imported and cached folders stay where they are.") {
                    if models.library.isEmpty { Text("No local models found. Download a model below or add an existing folder.").foregroundStyle(.secondary) }
                    ForEach(models.library.filter { search.isEmpty || $0.path.localizedCaseInsensitiveContains(search) || $0.name.localizedCaseInsensitiveContains(search) }) { item in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.name).font(.headline).lineLimit(1)
                                Text(item.path).font(.caption).foregroundStyle(.secondary).lineLimit(1).help(item.path)
                                Text(item.ready ? "Manifest present · runtime validates at load" : "Manifest missing · prepare with AX Engine CLI").font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Use") { selectedPath = item.path; page = .server }.disabled(!item.ready)
                            if item.managed {
                                Button(role: .destructive) { removeTarget = item } label: { Image(systemName: "trash") }
                                    .disabled(runtime.owned || runtime.connection != nil || runtime.busy || models.downloading || models.removing)
                                    .help("Remove app-managed model")
                            }
                        }.padding(.vertical, 6)
                        Divider()
                    }
                }
                Card("Runtime catalog", subtitle: "Pinned profiles come from your installed AX Engine runtime. A catalog entry is not a hardware qualification.") {
                    if models.refreshing { ProgressView("Reading runtime catalog…") }
                    if models.catalog.isEmpty && !models.refreshing { Text("Select your AX Engine runtime in Settings, then refresh.").foregroundStyle(.secondary) }
                    ForEach(models.catalog.filter { search.isEmpty || $0.alias.localizedCaseInsensitiveContains(search) || $0.repoId.localizedCaseInsensitiveContains(search) }) { item in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.alias).font(.headline)
                                Text(item.repoId).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                                Text(item.revision.map { "Revision " + $0 } ?? "Unpinned runtime profile").font(.caption2).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Download", systemImage: "arrow.down") { models.enqueue(item, settings: model.settings); page = .downloads }
                        }.padding(.vertical, 7)
                        Divider()
                    }
                }
            }.padding(26)
        }
    }
    private var downloadsPage: some View {
        ScrollView {
            VStack(spacing: 16) {
                if models.downloads.isEmpty {
                    ContentUnavailableView("No downloads yet", systemImage: "arrow.down.circle", description: Text("Choose a model from Models to begin."))
                    Button("Browse Models") { page = .models }
                }
                ForEach(models.downloads) { item in
                    Card(item.model.alias, subtitle: item.state) {
                        if let progress = item.fraction { ProgressView(value: progress) }
                        else if item.state == "Downloading" { ProgressView() }
                        Text(item.detail).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        HStack {
                            Text(item.destination).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                            Spacer()
                            if ["Queued", "Downloading"].contains(item.state) { Button("Cancel", role: .destructive) { models.cancel(item.id) } }
                            if ["Failed", "Cancelled"].contains(item.state) { Button("Retry") { models.retry(item.id, settings: model.settings) } }
                            if item.state == "Ready" { Button("Use Model") { selectedPath = item.destination; page = .server } }
                        }
                    }
                }
            }.padding(26)
        }
    }
    private var serverPage: some View {
        ScrollView {
            VStack(spacing: 18) {
                if let error = runtime.error { ErrorBanner(text: error) }
                Card("Local server", subtitle: "Listening on loopback only. Existing external services are never stopped by this app.") {
                    HStack {
                        PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : "circle", tint: runtime.ready ? Brand.accentChrome : .secondary)
                        Spacer()
                        Text("127.0.0.1:\(String(model.settings.port))").font(.system(.callout, design: .monospaced))
                        Button("Copy API URL", systemImage: "doc.on.doc") { copyText("http://127.0.0.1:\(model.settings.port)/v1") }
                    }
                    TextField("Installed model path or cached model alias", text: $selectedPath).textFieldStyle(.roundedBorder)
                        .disabled(runtime.owned || runtime.connection != nil)
                    HStack {
                        Menu("Choose Installed Model") {
                            ForEach(models.library.filter(\.ready)) { item in Button(item.name) { selectedPath = item.path } }
                        }.disabled(runtime.owned || runtime.connection != nil)
                        Spacer()
                        if runtime.owned {
                            Button("Stop Server", role: .destructive) { stopConfirmation = true }.disabled(runtime.state == "Stopping")
                        } else if runtime.connection != nil {
                            Button("Disconnect") { Task { await chat.stop(); runtime.detach() } }
                        } else {
                            Button("Start Server", systemImage: "play.fill") { runtime.start(model: selectedPath, settings: model.settings) }
                                .buttonStyle(.borderedProminent).disabled(selectedPath.isEmpty || runtime.busy || models.removing)
                        }
                    }
                    Text("Startup is offline. MTP policy and expert streaming follow AX Engine defaults. No model performance or certification setting is changed.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if runtime.connection == nil && !runtime.owned {
                    Card("Connect to an existing service", subtitle: "Uses the port in Settings. Authentication is held in memory for this connection only.") {
                        SecureField("API key (if configured)", text: $apiKey).textFieldStyle(.roundedBorder)
                        Button("Connect") {
                            let key = apiKey; apiKey = ""
                            Task { await runtime.attach(port: model.settings.port, apiKey: key) }
                        }.disabled(runtime.busy || models.removing)
                    }
                }
                Card("Runtime metrics", subtitle: "Live server counters; unavailable metrics are shown as a dash.") {
                    MetricRow(label: "Active generation streams", value: metric("ax_engine_generation_active_streams"))
                    MetricRow(label: "Scheduled tokens", value: metric("ax_engine_scheduled_tokens_total"))
                    MetricRow(label: "MLX active memory", value: memoryMetric("ax_engine_memory_mlx_active_bytes"))
                    MetricRow(label: "MLX cache memory", value: memoryMetric("ax_engine_memory_mlx_cache_bytes"))
                    MetricRow(label: "Process resident memory", value: memoryMetric("ax_engine_memory_host_resident_bytes"))
                    MetricRow(label: "MTP draft tokens", value: metric("ax_engine_mtp_draft_tokens_total"))
                    MetricRow(label: "MTP accepted tokens", value: metric("ax_engine_mtp_accepted_tokens_total"))
                    ForEach(runtime.models) { item in MetricRow(label: item.id, value: item.contextLength.map { "\($0) context" } ?? "Connected") }
                }
                if !runtime.logs.isEmpty {
                    Card("Runtime output", subtitle: "Bounded local diagnostic output") {
                        ScrollView([.horizontal, .vertical]) {
                            Text(runtime.logs).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }.frame(height: 190)
                    }
                }
            }.padding(26)
        }
    }
    private func memoryMetric(_ key: String) -> String {
        guard let bytes = runtime.metrics[key], bytes >= 0, bytes < Double(Int64.max) else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
    private func metric(_ key: String) -> String { runtime.metrics[key].map { String(format: "%.0f", $0) } ?? "—" }
    private func importFolder() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.prompt = "Add Model Folder"
        if panel.runModal() == .OK, let url = panel.url { Task { await models.importFolder(url, settings: model.settings) } }
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
