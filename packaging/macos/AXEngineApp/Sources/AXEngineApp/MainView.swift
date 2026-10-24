import SwiftUI
import AppKit
import AXEngineAppCore

enum Page: String, CaseIterable, Identifiable {
    case overview = "Overview", models = "Models", server = "Server", chat = "Chat", settings = "Settings"
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
    @State private var selectedPath = ""
    @State private var search = ""
    @State private var modelsSection = "library"
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
                    if runtime.ready && model.page != .chat { Button("Open Chat", systemImage: "bubble.left") { model.page = .chat } }
                }.padding(26)
                Divider()
                Group {
                    switch model.page ?? .overview {
                    case .overview: overview
                    case .models: modelsPage
                    case .server: serverPage
                    case .chat: ChatView(chat: chat, runtime: runtime, settings: model.settings)
                    case .settings: SettingsView(model: model)
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
        switch model.page ?? .overview {
        case .overview: "Your models. Your Mac. Your control."
        case .models: "Browse runtime profiles and manage local model folders."
        case .server: "Start a local model or connect to an existing AX Engine service."
        case .chat: "Private conversations with your running model."
        case .settings: "Configure your local runtime and generation defaults."
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
                            Button("Browse Models", systemImage: "square.stack.3d.up") { model.page = .models }.buttonStyle(.borderedProminent)
                            Button("Connect a Server") { model.page = .server }
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
                        MetricRow(label: "Local models", value: models.scanning || models.refreshing ? "Checking…" : String(models.library.count))
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
        VStack(spacing: 0) {
            VStack(spacing: 14) {
                if let error = models.error { ErrorBanner(text: error) }
                Picker("Models view", selection: $modelsSection) {
                    Text("Local Library").tag("library")
                    Text("Catalog").tag("catalog")
                    Text("Transfers").tag("transfers")
                }.pickerStyle(.segmented).frame(maxWidth: 460)
                if modelsSection != "transfers" {
                    HStack {
                        TextField("Search models", text: $search).textFieldStyle(.roundedBorder)
                        Button("Refresh", systemImage: "arrow.clockwise") { Task { await models.refresh(settings: model.settings) } }
                            .disabled(models.refreshing || models.scanning)
                        if modelsSection == "library" {
                            Button("Add Local Folder", systemImage: "folder.badge.plus") { importFolder() }
                        }
                    }
                }
            }.padding(26)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    switch modelsSection {
                    case "catalog": runtimeCatalog
                    case "transfers": transfers
                    default: localLibrary(filter: search)
                    }
                }.padding([.horizontal, .bottom], 26)
            }
        }
        .task { await models.scan(settings: model.settings) }
    }
    private var runtimeCatalog: some View {
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
                    Button("Download", systemImage: "arrow.down") { models.enqueue(item, settings: model.settings); modelsSection = "transfers"; model.page = .models }
                }.padding(.vertical, 7)
                Divider()
            }
        }
    }
    private func localLibrary(filter: String = "") -> some View {
        Card("Local library", subtitle: models.scanning ? "Checking model folders and cached snapshots…" : "\(models.library.count) local models · \(models.library.filter { $0.ready }.count) with complete weights") {
            if models.scanning { ProgressView("Scanning local library…") }
            if models.library.isEmpty && !models.scanning { Text("No local models found. Browse Catalog to download one, or add an existing folder.").foregroundStyle(.secondary) }
            ForEach(models.library.filter { filter.isEmpty || $0.path.localizedCaseInsensitiveContains(filter) || $0.name.localizedCaseInsensitiveContains(filter) }) { item in
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(item.name).font(.headline).lineLimit(1)
                        Text(item.path).font(.caption).foregroundStyle(.secondary).lineLimit(1).help(item.path)
                        Text(item.ready ? "Downloaded · runtime validates at load" : "Incomplete download or metadata only").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Use") { selectedPath = item.path; model.page = .server }.disabled(!item.ready)
                    if item.managed {
                        Button(role: .destructive) { removeTarget = item } label: { Image(systemName: "trash") }
                            .disabled(runtime.owned || runtime.connection != nil || runtime.busy || models.downloading || models.removing)
                            .help("Remove app-managed model")
                    }
                }.padding(.vertical, 6)
                Divider()
            }
        }
    }
    private var transfers: some View {
        VStack(spacing: 16) {
            if models.downloads.isEmpty {
                ContentUnavailableView("No transfers in this session", systemImage: "arrow.down.circle", description: Text("Previously downloaded models are available in Local Library."))
                Button("Browse Catalog") { search = ""; modelsSection = "catalog" }
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
                        if item.state == "Ready" { Button("Use Model") { selectedPath = item.destination; model.page = .server } }
                    }
                }
            }
        }.frame(maxWidth: .infinity)
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
