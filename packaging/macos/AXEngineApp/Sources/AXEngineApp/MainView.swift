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
    @State private var removeTarget: LocalModel?
    @State private var discardTarget: PartialDownload?
    @State private var stopConfirmation = false
    @State private var apiKey = ""
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
                if model.page == .server || model.page == .chat {
                    ModelProfileHeader(model: model, runtime: runtime, chat: chat)
                        .padding(.horizontal, 26).padding(.bottom, 16)
                }
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
        .alert(removeTarget.map { "Remove \($0.displayName)?" } ?? "Remove model?",
               isPresented: Binding(get: { removeTarget != nil }, set: { if !$0 { removeTarget = nil } })) {
            Button("Remove", role: .destructive) {
                if let target = removeTarget, removalBlockReason(target) == nil {
                    Task {
                        if await models.remove(target, settings: model.settings), model.selectedPath == target.path {
                            model.selectedPath = ""
                        }
                    }
                }
                removeTarget = nil
            }
            Button("Cancel", role: .cancel) { removeTarget = nil }
        } message: { Text(removeTarget.map(removalMessage) ?? "") }
        .alert("Discard partial download?", isPresented: Binding(get: { discardTarget != nil }, set: { if !$0 { discardTarget = nil } })) {
            Button("Discard", role: .destructive) {
                if let target = discardTarget { Task { await models.discardPartial(target, settings: model.settings) } }
                discardTarget = nil
            }
            Button("Cancel", role: .cancel) { discardTarget = nil }
        } message: {
            Text(discardTarget.map { "Deletes the incomplete files at \($0.path)\($0.sizeBytes.map { ", freeing about " + bytes($0) } ?? ""). A later download starts over." } ?? "")
        }
        .alert("Stop the local server?", isPresented: $stopConfirmation) {
            Button("Stop", role: .destructive) { Task { await chat.stop(); await runtime.stop() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("Active inference on this app-owned server will stop.") }
    }

    private func bytes(_ value: Int64) -> String { ByteCountFormatter.string(fromByteCount: value, countStyle: .file) }

    /// Why a library row cannot be removed right now, or nil when it can.
    private func removalBlockReason(_ item: LocalModel) -> String? {
        if models.downloading { return "Wait for downloads to finish before removing models." }
        if models.removing { return "Another model operation is in progress." }
        if runtime.busy { return "Wait for the server to finish starting or stopping." }
        if runtime.owned && runtime.selectedModel == item.path { return "Stop the server to remove the model it is serving." }
        if runtime.connection != nil && !runtime.owned { return "Disconnect from the attached service first; it may be serving this model." }
        return nil
    }

    private func removalMessage(_ item: LocalModel) -> String {
        let size = item.sizeBytes.map { " This frees about " + bytes($0) + "." } ?? ""
        switch models.removalTarget(for: item, settings: model.settings) {
        case .cacheRepository(let url):
            return "Deletes every cached revision of \(item.displayName) from the Hugging Face cache at \(url.path).\(size) Other apps that use this cache must download it again. This can't be undone."
        case .managedFolder(let url):
            return "Deletes the downloaded files at \(url.path).\(size) This can't be undone."
        case nil:
            return "This model cannot be removed by AX Engine."
        }
    }

    private func syncHostMonitoring() {
        model.hostMonitor.setActive(HostMonitor.shouldSample(enabled: model.hostMonitoringEnabled))
    }
    private var subtitle: String {
        switch model.page ?? .overview {
        case .overview: "Memory, storage and the models on your Mac."
        case .models: "Browse runtime profiles and manage local model folders."
        case .server: "Start a local model or connect to an existing AX Engine service."
        case .chat: "Test your model’s responses and inference speed."
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
        Card("Public AutomatosX models", subtitle: "Pinned profiles come from your installed AX Engine runtime. A catalog entry is not a hardware qualification.") {
            Label("No Hugging Face account or token required", systemImage: "arrow.down.circle")
                .font(.headline)
            Text("Model downloads connect to Hugging Face and its delivery hosts.")
                .font(.caption).foregroundStyle(.secondary)
            Text("AX Engine provides downloads for public AutomatosX models only. To use other models, download them separately and run them with the AX Engine CLI.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Divider()
            if models.refreshing { ProgressView("Reading runtime catalog…") }
            if !models.refreshing && !models.supportsNativeDownloads {
                Text("Update your AX Engine runtime for downloads without Python, then refresh the catalog.")
                    .font(.callout).foregroundStyle(.secondary)
            }
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
                    }.disabled(!models.supportsNativeDownloads)
                }.padding(.vertical, 7)
                Divider()
            }
        }
    }
    private func localLibrary(filter: String = "") -> some View {
        Card("Local library", subtitle: "\(models.productLibrary.count) folders · \(models.productLibrary.filter { $0.ready && models.checkedPaths.contains($0.path) }.count) with checked, complete weight files") {
            Text("Public AutomatosX models downloaded on this Mac. Other models can be used through the AX Engine CLI.")
                .font(.caption).foregroundStyle(.secondary)
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
            if models.productLibrary.isEmpty && !models.scanning && models.libraryIssue == nil {
                Text("No local models found. Browse Catalog to download a public AutomatosX model.")
                Button("Browse Catalog") { model.modelSearch = ""; model.modelsSection = "catalog" }
            }
            let matches = models.productLibrary.filter {
                filter.isEmpty || $0.path.localizedCaseInsensitiveContains(filter) || $0.name.localizedCaseInsensitiveContains(filter)
            }.sorted { left, right in
                let l = left.ready && models.checkedPaths.contains(left.path)
                let r = right.ready && models.checkedPaths.contains(right.path)
                return l == r ? left.path < right.path : l && !r
            }
            if matches.isEmpty && !filter.isEmpty {
                Text("No folders match this filter.").foregroundStyle(.secondary)
                Button("Clear search") { model.modelSearch = "" }
            }
            ForEach(matches) { item in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(item.displayName).font(.headline).lineLimit(2)
                            Text(item.source + (item.sizeBytes.map { " · " + bytes($0) } ?? "")).font(.caption).foregroundStyle(.secondary)
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
                        if models.removalTarget(for: item, settings: model.settings) != nil {
                            let blocked = removalBlockReason(item)
                            Button("Remove", systemImage: "trash", role: .destructive) { removeTarget = item }
                                .disabled(blocked != nil)
                                .help(blocked ?? "Delete this downloaded model from disk")
                                .accessibilityLabel("Remove " + item.displayName)
                        }
                    }
                    DisclosureGroup("Location and details") {
                        Text(item.name).font(.callout).textSelection(.enabled)
                        Text(item.path).font(.caption).textSelection(.enabled)
                        HStack {
                            Button("Copy path", systemImage: "doc.on.doc") { copyText(item.path) }
                            Button("Show in Finder", systemImage: "folder") {
                                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.path)])
                            }
                        }
                        if runtime.owned && runtime.selectedModel == item.path {
                            Text("Stop the server to remove this model.").font(.caption).foregroundStyle(.secondary)
                        }
                        Text("File completeness does not establish compatibility. AX Engine validates the model when you start it.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(.vertical, 8)
                Divider()
            }
            if !models.partialDownloads.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Interrupted downloads").font(.headline)
                    Text("Incomplete files kept so a retry can resume. Discard them to free space or to download a different revision.")
                        .font(.caption).foregroundStyle(.secondary)
                    ForEach(models.partialDownloads) { partial in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(partial.label).font(.callout)
                                Text(partial.sizeBytes.map(bytes) ?? "Size unknown").font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Discard", systemImage: "trash", role: .destructive) { discardTarget = partial }
                                .disabled(models.downloading || models.removing)
                                .help(models.downloading ? "Wait for downloads to finish." : "Delete these incomplete files")
                        }
                    }
                }.padding(.top, 6)
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
                        Text(runtime.connection?.apiURL.absoluteString ?? "127.0.0.1:\(String(model.settings.port))")
                            .font(.system(.callout, design: .monospaced))
                    }
                    if let connection = runtime.connection {
                        HStack {
                            Button("Copy API URL", systemImage: "doc.on.doc") {
                                guard let current = runtime.connection else { return }
                                copyText(current.apiURL.absoluteString)
                                clientCopyNotice = "API URL copied."
                            }
                            Button("Copy API key", systemImage: "key") {
                                guard runtime.ready, let current = runtime.connection else { return }
                                clientCopyNotice = current.copyAPIKey(to: clientClipboard)
                                    ? "API key copied for up to 60 seconds. Paste it into your client."
                                    : "The API key could not be copied. Try again."
                            }.disabled(!runtime.ready || !connection.hasAPIKey)
                        }
                        Text("Copy the URL and API key into your local client. The copied key stays on this Mac and clears after 60 seconds if unchanged.")
                            .font(.caption).foregroundStyle(.secondary)
                        if !clientCopyNotice.isEmpty {
                            Text(clientCopyNotice).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Text(model.selectedPath.isEmpty ? "Choose an installed AutomatosX model below." : model.selectedPath)
                        .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                    HStack {
                        Menu("Choose Installed Model") {
                            ForEach(models.productLibrary.filter { $0.ready && models.checkedPaths.contains($0.path) }) { item in Button(item.name) { model.selectedPath = item.path } }
                        }.disabled(runtime.connection != nil && !runtime.owned)
                        Spacer()
                        if runtime.owned {
                            let switching = !model.selectedPath.isEmpty && model.selectedPath != runtime.selectedModel
                            Button(switching ? "Restart with Selected Model" : "Restart Server", systemImage: "arrow.clockwise") {
                                let target = switching && models.canSelectInGUI(model.selectedPath) ? model.selectedPath : runtime.selectedModel
                                Task { await chat.stop(); await runtime.restart(model: target, settings: model.settings) }
                            }
                            .disabled(runtime.busy || runtime.state == "Stopping" || models.removing
                                      || (switching && !models.canSelectInGUI(model.selectedPath)))
                            .help("Stop this server and start it again. Active inference stops.")
                            Button("Stop Server", role: .destructive) { stopConfirmation = true }.disabled(runtime.state == "Stopping")
                        } else if runtime.connection != nil {
                            Button("Disconnect") { Task { await chat.stop(); runtime.detach() } }
                        } else {
                            Button("Start Server", systemImage: "play.fill") { if models.canSelectInGUI(model.selectedPath) { runtime.start(model: model.selectedPath, settings: model.settings) } }
                                .buttonStyle(.borderedProminent).disabled(!models.canSelectInGUI(model.selectedPath) || runtime.busy || models.removing)
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
