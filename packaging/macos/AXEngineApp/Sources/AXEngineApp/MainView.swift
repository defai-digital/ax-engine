import SwiftUI
import AppKit
import AXEngineAppCore

enum Page: String, CaseIterable, Identifiable {
    case overview = "Dashboard", models = "Models", server = "Server", chat = "Chat", settings = "Settings"
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
    @Environment(\.scenePhase) private var scenePhase
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
                    case .overview: DashboardView(model: model, models: models, monitor: model.hostMonitor, storage: model.storageMonitor)
                    case .models: modelsPage
                    case .server: serverPage
                    case .chat: ChatView(chat: chat, runtime: runtime, settings: model.settings, openServer: { model.page = .server })
                    case .settings: SettingsView(model: model)
                    }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .task { if !models.discoveryAttempted && !models.refreshing { await models.refresh(settings: model.settings) } }
        .onAppear { syncHostMonitoring() }
        .onChange(of: model.page) { _, _ in syncHostMonitoring() }
        .onChange(of: model.hostMonitoringEnabled) { _, _ in syncHostMonitoring() }
        .onChange(of: scenePhase) { _, _ in syncHostMonitoring() }
        .onDisappear { model.hostMonitor.setActive(false) }
        .onChange(of: model.searchFocusRequest) { _, _ in
            Task { @MainActor in await Task.yield(); searchFocused = true }
        }
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

    private func syncHostMonitoring() {
        model.hostMonitor.setActive(HostMonitor.shouldSample(enabled: model.hostMonitoringEnabled,
            foreground: scenePhase == .active, visible: model.page == .overview || model.page == .server))
    }
    private var subtitle: String {
        switch model.page ?? .overview {
        case .overview: "Memory, storage and the models on your Mac."
        case .models: "Browse runtime profiles and manage local model folders."
        case .server: "Start a local model or connect to an existing AX Engine service."
        case .chat: "Private conversations with your running model."
        case .settings: "Configure your local runtime and generation defaults."
        }
    }

    private var modelsPage: some View {
        VStack(spacing: 0) {
            VStack(spacing: 14) {
                if let error = models.error {
                    ErrorBanner(text: error)
                    Button("Review runtime settings") { model.page = .settings }
                }
                Picker("Models view", selection: $model.modelsSection) {
                    Text("Local Library").tag("library")
                    Text("Catalog").tag("catalog")
                    Text("Transfers").tag("transfers")
                }.pickerStyle(.segmented).frame(maxWidth: 460)
                if model.modelsSection != "transfers" {
                    HStack {
                        TextField("Search models", text: $model.modelSearch).textFieldStyle(.roundedBorder).focused($searchFocused)
                        Button("Refresh", systemImage: "arrow.clockwise") { Task { await models.refresh(settings: model.settings) } }
                            .disabled(models.refreshing || models.scanning)
                        if model.modelsSection == "library" {
                            Button("Add Local Folder", systemImage: "folder.badge.plus") { importFolder() }
                        }
                    }
                }
            }.padding(26)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    switch model.modelsSection {
                    case "catalog": runtimeCatalog
                    case "transfers": transfers
                    default: localLibrary(filter: model.modelSearch)
                    }
                }.padding([.horizontal, .bottom], 26)
            }
        }
        .task { if !models.scanning && !models.discoveryAttempted { await models.scan(settings: model.settings) } }
    }
    private var runtimeCatalog: some View {
        Card("Runtime catalog", subtitle: "Pinned profiles come from your installed AX Engine runtime. A catalog entry is not a hardware qualification.") {
            if models.refreshing { ProgressView("Reading runtime catalog…") }
            if models.catalog.isEmpty && !models.refreshing { Text("Select your AX Engine runtime in Settings, then refresh.").foregroundStyle(.secondary) }
            let matches = models.catalog.filter { model.modelSearch.isEmpty || $0.alias.localizedCaseInsensitiveContains(model.modelSearch) || $0.repoId.localizedCaseInsensitiveContains(model.modelSearch) }
            if matches.isEmpty && !model.modelSearch.isEmpty && !models.refreshing {
                Text("No catalog profiles match your search.").foregroundStyle(.secondary)
                Button("Clear search") { model.modelSearch = "" }
            }
            ForEach(matches) { item in
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(item.alias).font(.headline)
                        Text(item.repoId).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        Text(item.revision.map { "Revision " + $0 } ?? "Unpinned runtime profile").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(models.downloads.contains { $0.model.target == item.target && ["Queued", "Downloading"].contains($0.state) } ? "View Transfer" : "Download", systemImage: "arrow.down") {
                        models.enqueue(item, settings: model.settings); model.modelsSection = "transfers"; model.page = .models
                    }
                }.padding(.vertical, 7)
                Divider()
            }
        }
    }
    private func localLibrary(filter: String = "") -> some View {
        Card("Local library", subtitle: "\(models.library.count) folders · \(models.library.filter { $0.ready && models.checkedPaths.contains($0.path) }.count) with checked, complete weight files") {
            Picker("Library filter", selection: $model.libraryScope) {
                Text("All folders").tag("all")
                Text("Catalog matches").tag("catalog")
                Text("Other folders").tag("other")
            }.pickerStyle(.segmented)
            if model.libraryScope != "all" {
                Text("Catalog association is not a compatibility check. Other folders may still be supported by AX Engine.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if models.scanning {
                HStack {
                    ProgressView("Checking folders · \(models.checkedPaths.count) found")
                    Spacer()
                    Button("Stop scan") { models.cancelScan() }
                }
            }
            if let issue = models.libraryIssue {
                ErrorBanner(text: issue)
                Button("Retry scan") { Task { await models.scan(settings: model.settings) } }.disabled(models.scanning)
            }
            if let date = models.lastScanAt {
                Text("Last complete scan: " + date.formatted(date: .omitted, time: .standard)).font(.caption).foregroundStyle(.secondary)
            }
            if models.library.isEmpty && !models.scanning && models.libraryIssue == nil {
                Text("No local models found. Browse Catalog to download one, or add an existing folder.")
                Button("Browse Catalog") { model.modelSearch = ""; model.modelsSection = "catalog" }
            }
            let matches = models.library.filter {
                let associated = $0.isCatalogAssociated(models.catalog)
                let included = model.libraryScope == "all" || (model.libraryScope == "catalog" ? associated : !associated)
                return included && (filter.isEmpty || $0.path.localizedCaseInsensitiveContains(filter) || $0.name.localizedCaseInsensitiveContains(filter))
            }.sorted { left, right in
                let l = left.ready && models.checkedPaths.contains(left.path)
                let r = right.ready && models.checkedPaths.contains(right.path)
                return l == r ? left.path < right.path : l && !r
            }
            if matches.isEmpty && (!filter.isEmpty || model.libraryScope != "all") {
                Text("No folders match this filter.").foregroundStyle(.secondary)
                Button("Show all folders") { model.libraryScope = "all"; model.modelSearch = "" }
            }
            ForEach(matches) { item in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.displayName).font(.headline).lineLimit(2)
                            Text(item.source).font(.caption).foregroundStyle(.secondary)
                            Text(runtime.ready && runtime.owned && runtime.selectedModel == item.path
                                 ? "Running in this app"
                                 : models.checkedPaths.contains(item.path)
                                    ? (item.ready ? "Downloaded · compatibility unverified" : "Weights incomplete or metadata only")
                                    : "Not checked in this scan · previous result")
                                .font(.callout).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Select Model") { model.selectedPath = item.path; model.page = .server }
                            .disabled(!item.ready || !models.checkedPaths.contains(item.path))
                            .accessibilityLabel("Select " + item.displayName)
                        if item.managed {
                            Button(role: .destructive) { removeTarget = item } label: { Image(systemName: "trash") }
                                .disabled(runtime.owned || runtime.connection != nil || runtime.busy || models.downloading || models.removing)
                                .help("Remove app-managed model").accessibilityLabel("Remove " + item.displayName)
                        }
                    }
                    DisclosureGroup("Location and details") {
                        Text(item.name).font(.callout).textSelection(.enabled)
                        Text(item.path).font(.caption).textSelection(.enabled)
                        Button("Copy path", systemImage: "doc.on.doc") { copyText(item.path) }
                        Text("File completeness does not establish compatibility. AX Engine validates the model when you start it.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.vertical, 8)
                Divider()
            }
        }
    }
    private var transfers: some View {
        VStack(spacing: 16) {
            if models.downloads.isEmpty {
                ContentUnavailableView("No transfers in this session", systemImage: "arrow.down.circle", description: Text("Previously downloaded models are available in Local Library."))
                Button("Browse Catalog") { model.modelSearch = ""; model.modelsSection = "catalog" }
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
                        if item.state == "Ready" { Button("Use Model") { model.selectedPath = item.destination; model.page = .server } }
                    }
                }
            }
        }.frame(maxWidth: .infinity)
    }
    private var serverPage: some View {
        ScrollView {
            VStack(spacing: 18) {
                if let error = runtime.error {
                    ErrorBanner(text: error)
                    Button("Review settings") { model.page = .settings }
                }
                if let checked = runtime.lastSuccessfulCheck {
                    Text("Last successful check: " + checked.formatted(date: .omitted, time: .standard)).font(.caption).foregroundStyle(.secondary)
                }
                Card("Local server", subtitle: "Listening on loopback only. Existing external services are never stopped by this app.") {
                    HStack {
                        PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : "circle", tint: runtime.ready ? Brand.accentChrome : .secondary)
                        Spacer()
                        Text("127.0.0.1:\(String(model.settings.port))").font(.system(.callout, design: .monospaced))
                        Button("Copy API URL", systemImage: "doc.on.doc") { copyText("http://127.0.0.1:\(model.settings.port)/v1") }
                    }
                    TextField("Installed model path or cached model alias", text: $model.selectedPath).textFieldStyle(.roundedBorder)
                        .disabled(runtime.owned || runtime.connection != nil)
                    HStack {
                        Menu("Choose Installed Model") {
                            ForEach(models.library.filter { $0.ready && models.checkedPaths.contains($0.path) }) { item in Button(item.name) { model.selectedPath = item.path } }
                        }.disabled(runtime.owned || runtime.connection != nil)
                        Spacer()
                        if runtime.owned {
                            Button("Stop Server", role: .destructive) { stopConfirmation = true }.disabled(runtime.state == "Stopping")
                        } else if runtime.connection != nil {
                            Button("Disconnect") { Task { await chat.stop(); runtime.detach() } }
                        } else {
                            Button("Start Server", systemImage: "play.fill") { runtime.start(model: model.selectedPath, settings: model.settings) }
                                .buttonStyle(.borderedProminent).disabled(model.selectedPath.isEmpty || runtime.busy || models.removing)
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
                RuntimeHistoryView(runtime: runtime)
                HostMonitoringView(model: model, monitor: model.hostMonitor)
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
