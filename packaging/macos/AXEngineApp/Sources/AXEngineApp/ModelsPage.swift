import SwiftUI
import AppKit
import AXEngineAppCore

/// Models workspace: Local Library, Catalog and Transfers. Removal, download
/// and discard confirmations live here so every entry point shares them.
struct ModelsPage: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var storage: StorageMonitor
    @FocusState.Binding var searchFocused: Bool
    @State private var removeTarget: LocalModel?
    @State private var discardTarget: PartialDownload?
    @State private var spaceWarning: CatalogModel?
    @State private var redownloadTarget: CatalogModel?
    @State private var cancelTarget: DownloadItem?
    @State private var largerExpanded = false

    private static let spaceMargin: Int64 = 64 * 1024 * 1024

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 14) {
                if let error = models.error {
                    ErrorBanner(text: error)
                    HStack {
                        Button("Review runtime settings") { model.page = .settings }
                        Button("Dismiss") { models.dismissError() }
                    }
                }
                Picker("Models view", selection: $model.modelsSection) {
                    Text("Your Models").tag("library")
                    Text("Catalog").tag("catalog")
                    Text(models.activeDownloadCount > 0 ? "Downloads (\(models.activeDownloadCount))" : "Downloads").tag("transfers")
                }.pickerStyle(.segmented).frame(maxWidth: 460)
                if model.modelsSection != "transfers" {
                    HStack {
                        TextField(model.modelsSection == "catalog" ? "Search catalog" : "Search your models",
                                  text: model.modelsSection == "catalog" ? $model.catalogSearch : $model.modelSearch)
                            .textFieldStyle(.roundedBorder).focused($searchFocused)
                        Button("Refresh", systemImage: "arrow.clockwise") { Task { await models.refresh(settings: model.settings) } }
                            .disabled(models.refreshing || models.scanning)
                    }
                }
            }.padding(26)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    switch model.modelsSection {
                    case "catalog": catalog
                    case "transfers": transfers
                    default: library(filter: model.modelSearch)
                    }
                }.padding([.horizontal, .bottom], 26)
            }
        }
        .task { if !models.scanning && !models.discoveryAttempted { await models.scan(settings: model.settings) } }
        .task(id: model.modelsSection) {
            if model.modelsSection == "catalog" { await storage.refresh(path: model.settings.modelsDirectory) }
        }
        .alert(removeTarget.map { "Remove \($0.displayName)?" } ?? "Remove model?",
               isPresented: Binding(get: { removeTarget != nil }, set: { if !$0 { removeTarget = nil } })) {
            Button("Remove", role: .destructive) {
                if let target = removeTarget, removalReason(target) == nil {
                    Task {
                        let removal = models.removalTarget(for: target, settings: model.settings)
                        if await models.remove(target, settings: model.settings),
                           removal?.containsModel(path: model.selectedPath) == true {
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
        .alert("Probably not enough free space", isPresented: Binding(get: { spaceWarning != nil }, set: { if !$0 { spaceWarning = nil } })) {
            Button("Download Anyway") { if let item = spaceWarning { startDownload(item) }; spaceWarning = nil }
            Button("Change Download Folder…") { spaceWarning = nil; model.page = .settings }
            Button("Cancel", role: .cancel) { spaceWarning = nil }
        } message: { Text(spaceWarning.map(spaceMessage) ?? "") }
        .alert(cancelTarget.map { "Stop downloading \($0.model.alias)?" } ?? "Stop download?",
               isPresented: Binding(get: { cancelTarget != nil }, set: { if !$0 { cancelTarget = nil } })) {
            Button("Stop Download", role: .destructive) { if let item = cancelTarget { models.cancel(item.id) }; cancelTarget = nil }
            Button("Keep Downloading", role: .cancel) { cancelTarget = nil }
        } message: { Text("Files already verified stay on disk, so Retry or Resume continues from there.") }
        .alert(redownloadTarget.map { "Download \($0.alias) again?" } ?? "Download again?",
               isPresented: Binding(get: { redownloadTarget != nil }, set: { if !$0 { redownloadTarget = nil } })) {
            Button("Download") { if let item = redownloadTarget { requestDownload(item) }; redownloadTarget = nil }
            Button("Cancel", role: .cancel) { redownloadTarget = nil }
        } message: {
            Text("This model is already in your library. AX Engine re-verifies the pinned files and reuses the ones that are unchanged.")
        }
    }

    // MARK: - Shared helpers

    private func bytes(_ value: Int64) -> String { TransferFormat.bytes(value) }

    private func removalState(_ item: LocalModel) -> RemovalGate.State {
        RemovalGate.State(removable: models.removalTarget(for: item, settings: model.settings) != nil, source: item.source,
                          downloading: models.downloading, removing: models.removing, runtimeBusy: runtime.busy,
                          servedByThisApp: removalAffectsOwnedRuntime(item),
                          attachedToExternalService: runtime.connection != nil && !runtime.owned)
    }
    private func removalReason(_ item: LocalModel) -> String? { RemovalGate.reason(removalState(item)) }

    private func removalAffectsOwnedRuntime(_ item: LocalModel) -> Bool {
        runtime.owned && (runtime.selectedModel == item.path
            || models.removalTarget(for: item, settings: model.settings)?.containsModel(path: runtime.selectedModel) == true)
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

    private func availableSpace() -> StorageSnapshot? {
        guard let snapshot = storage.snapshot, snapshot.requestedPath == model.settings.modelsDirectory else { return nil }
        return snapshot
    }

    /// Free space the catalog estimate would not fit into, or nil when it fits or is unknown.
    private func spaceShortfall(_ item: CatalogModel) -> Int64? {
        guard let needed = item.approxSizeBytes, let snapshot = availableSpace() else { return nil }
        let shortfall = needed + Self.spaceMargin - snapshot.availableBytes
        return shortfall > 0 ? shortfall : nil
    }

    private func spaceMessage(_ item: CatalogModel) -> String {
        guard let needed = item.approxSizeBytes, let snapshot = availableSpace() else { return "" }
        return "The catalog estimates \(item.alias) at about \(bytes(needed)); \(snapshot.volumeName) has \(bytes(snapshot.availableBytes)) free. Before downloading, AX Engine checks the exact remaining size and stops without touching existing models if it does not fit; a resumed download may need less. Free up space or choose another download folder."
    }

    private func requestDownload(_ item: CatalogModel) {
        if spaceShortfall(item) != nil { spaceWarning = item } else { startDownload(item) }
    }

    private func startDownload(_ item: CatalogModel) {
        models.enqueue(item, settings: model.settings)
        Task { await storage.refresh(path: model.settings.modelsDirectory) }
    }

    private func showInFinder(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    /// The single "use this model" action shared by Library, Catalog and
    /// Downloads. Starts at once when nothing is serving; otherwise selects the
    /// model on Dashboard, where Restart switches to it.
    private func serveButton(path: String, ready: Bool, prominent: Bool = false) -> some View {
        let busy = runtime.owned
        let reason: String? = !ready ? "Weights are incomplete or not checked yet. Refresh, or download it again from Catalog."
            : model.attachedToService ? "Disconnect from the attached service on Server Settings to serve your own model."
            : model.connecting ? "Wait for the connection to the existing service to finish."
            : model.models.canSelectInGUI(path) ? nil : "The library scan has not checked this model yet."
        return Group {
            if prominent {
                Button(busy ? "Select on Dashboard" : "Serve", systemImage: "play.fill") { model.serve(path) }.buttonStyle(.borderedProminent)
            } else {
                Button(busy ? "Select on Dashboard" : "Serve", systemImage: "play.fill") { model.serve(path) }
            }
        }
        .disabled(reason != nil)
        .help(reason ?? (busy ? "Select this model on Dashboard; Restart there switches the server to it." : "Start the server on this model"))
        .accessibilityHint(reason ?? "")
    }

    // MARK: - Local Library

    private func library(filter: String) -> some View {
        Card("Your models", subtitle: "\(models.productLibrary.count) on this Mac · \(model.servableModels.count) ready to serve") {
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
                ContentUnavailableView {
                    Label("No models yet", systemImage: "square.stack.3d.up")
                } description: {
                    Text("Download a public AutomatosX model from the Catalog. No account or token needed.")
                } actions: {
                    Button("Browse Catalog") { model.catalogSearch = ""; model.modelsSection = "catalog" }.buttonStyle(.borderedProminent)
                }
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
                libraryRow(item)
                Divider()
            }
            if !models.partialDownloads.isEmpty {
                HStack {
                    Label("\(models.partialDownloads.count) interrupted download\(models.partialDownloads.count == 1 ? "" : "s") on disk", systemImage: "arrow.down.circle.dotted")
                        .font(.callout).foregroundStyle(.secondary)
                    Button("Review in Downloads") { model.modelsSection = "transfers" }.buttonStyle(.link)
                }
            }
        }
    }

    private func libraryRow(_ item: LocalModel) -> some View {
        let running = runtime.ready && runtime.owned && runtime.selectedModel == item.path
        let checked = models.checkedPaths.contains(item.path)
        let state = removalState(item)
        let reason = RemovalGate.reason(state)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        ModelTitle(label: item.label, font: .headline).help(item.name)
                        if running { PillBadge(text: "Serving", systemImage: "circle.fill", tint: Brand.accentChrome, emphasized: true) }
                        else if model.selectedPath == item.path { PillBadge(text: "Selected", systemImage: "checkmark") }
                    }
                    Text(item.source + (item.sizeBytes.map { " · " + bytes($0) } ?? "")).font(.caption).foregroundStyle(.secondary)
                    Label(checked ? (item.ready ? "Ready to serve" : "Weights incomplete or metadata only") : "Not checked in this scan",
                          systemImage: checked ? (item.ready ? "checkmark.circle" : "exclamationmark.triangle") : "questionmark.circle")
                        .font(.callout).foregroundStyle(checked && !item.ready ? .orange : .secondary)
                }
                Spacer()
                if !running { serveButton(path: item.path, ready: item.ready && checked) }
                if state.removable {
                    Button("Remove", systemImage: "trash", role: .destructive) { removeTarget = item }
                        .disabled(reason != nil)
                        .help(reason ?? "Delete this downloaded model from disk")
                        .accessibilityLabel("Remove " + item.displayName)
                }
            }
            if let reason {
                Label(reason, systemImage: state.removable ? "clock" : "lock")
                    .font(.caption).foregroundStyle(.secondary)
            }
            DisclosureGroup("Location and details") {
                Text(item.name).font(.callout).textSelection(.enabled)
                Text(item.path).font(.caption).textSelection(.enabled)
                HStack {
                    Button("Copy path", systemImage: "doc.on.doc") { copyText(item.path) }
                    Button("Show in Finder", systemImage: "folder") { showInFinder(item.path) }
                }
                Text("File completeness does not establish compatibility. AX Engine validates the model when you start it.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 8)
        .contentShape(Rectangle())
        .contextMenu {
            Button("Serve") { model.serve(item.path) }.disabled(!item.ready || !checked || running)
            Button("Show in Finder") { showInFinder(item.path) }
            Button("Copy Path") { copyText(item.path) }
            if state.removable {
                Divider()
                Button("Remove…", role: .destructive) { removeTarget = item }.disabled(reason != nil)
            }
        }
    }

    private var partialDownloads: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Interrupted downloads").font(.headline)
            Text("Incomplete files kept on disk so a retry resumes where it stopped. Discard them to free space or to download a different revision.")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(models.partialDownloads) { partial in
                let entry = models.catalogEntry(forPartial: partial)
                let resuming = entry.flatMap(models.activeDownload(for:)) != nil
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(partial.label).font(.callout)
                        Text((partial.sizeBytes.map(bytes) ?? "Size unknown") + (resuming ? " · resuming in Downloads" : "")).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let entry, !resuming {
                        Button("Resume", systemImage: "arrow.down.circle") { requestDownload(entry) }
                            .disabled(models.downloadBlockReason(for: entry) != nil)
                            .help("Queue this download again; verified files are kept")
                    }
                    Button("Discard", systemImage: "trash", role: .destructive) { discardTarget = partial }
                        .disabled(models.downloading || models.removing)
                        .help(models.downloading ? "Wait for downloads to finish." : "Delete these incomplete files")
                }
            }
            if models.downloading || models.removing {
                Label("Discard is available once downloads and model operations finish.", systemImage: "clock").font(.caption).foregroundStyle(.secondary)
            }
        }.padding(.top, 6)
    }

    // MARK: - Catalog

    private var memoryBytes: Int64 { Int64(ProcessInfo.processInfo.physicalMemory) }

    private var catalog: some View {
        let search = model.catalogSearch
        let matches = models.catalog.filter { search.isEmpty || $0.alias.localizedCaseInsensitiveContains(search) || $0.repoId.localizedCaseInsensitiveContains(search) }
        // Download size against unified memory: a size comparison, not a fit claim.
        let fits = matches.filter { ($0.approxSizeBytes ?? 0) <= memoryBytes }
        let larger = matches.filter { ($0.approxSizeBytes ?? 0) > memoryBytes }
        return Card("Catalog", subtitle: "Public AutomatosX models. Pinned profiles come from your AX Engine runtime; an entry is not a hardware qualification.") {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Label("No Hugging Face account or token required", systemImage: "arrow.down.circle").font(.headline)
                    Text("Downloads connect to Hugging Face and its delivery hosts. Use other models through the AX Engine CLI.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                if let snapshot = availableSpace() {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(bytes(snapshot.availableBytes) + " free").font(.callout.weight(.medium))
                        Text("on " + snapshot.volumeName).font(.caption).foregroundStyle(.secondary)
                    }
                } else if storage.refreshing {
                    ProgressView().controlSize(.small)
                } else {
                    Button("Free space unknown · Check") { Task { await storage.refresh(path: model.settings.modelsDirectory) } }
                        .buttonStyle(.link).font(.caption)
                }
            }
            Divider()
            if models.refreshing { ProgressView("Reading runtime catalog…") }
            if let blocked = models.downloadsBlockReason, !models.refreshing {
                Label(blocked, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
            }
            if models.catalog.isEmpty && !models.refreshing {
                ContentUnavailableView {
                    Label("Catalog unavailable", systemImage: "books.vertical")
                } description: {
                    Text("The catalog comes from your installed AX Engine runtime. Check the runtime location in Settings, then refresh.")
                } actions: {
                    HStack {
                        Button("Open Settings") { model.page = .settings }
                        Button("Refresh Catalog") { Task { await models.refresh(settings: model.settings) } }.buttonStyle(.borderedProminent)
                    }
                }
            }
            if matches.isEmpty && !search.isEmpty && !models.refreshing {
                Text("No catalog models match “\(search)”.").foregroundStyle(.secondary)
                Button("Clear search") { model.catalogSearch = "" }
            }
            if !fits.isEmpty && !larger.isEmpty {
                Text("Download size under this Mac's \(ByteCountFormatter.string(fromByteCount: memoryBytes, countStyle: .memory)) memory")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            ForEach(fits) { item in
                catalogRow(item)
                Divider()
            }
            if !larger.isEmpty {
                DisclosureGroup(isExpanded: $largerExpanded) {
                    ForEach(larger) { item in
                        catalogRow(item)
                        Divider()
                    }
                } label: {
                    Text("Download size over this Mac's memory (\(larger.count))")
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                }
            }
        }
    }

    private func catalogRow(_ item: CatalogModel) -> some View {
        let installed = models.installedModel(for: item)
        let active = models.activeDownload(for: item)
        let blocked = models.downloadBlockReason(for: item)
        let shortfall = spaceShortfall(item)
        let memory = memoryBytes
        let label = ModelLabel(repositoryName: item.repoId.components(separatedBy: "/").last ?? item.repoId)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    ModelTitle(label: label, font: .headline).help(item.repoId)
                    Text(item.alias + " · " + item.repoId).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    HStack(spacing: 6) {
                        if let size = item.approxSizeBytes { PillBadge(text: "about " + bytes(size), systemImage: "internaldrive") }
                        if item.mtpIncluded == true { PillBadge(text: "MTP included", systemImage: "bolt") }
                        if let certification = item.certification { PillBadge(text: certification.capitalized, systemImage: "checkmark.seal") }
                        Text(item.revision.map { "Revision " + String($0.prefix(12)) } ?? "Unpinned runtime profile").font(.caption).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                if let active {
                    VStack(alignment: .trailing, spacing: 4) {
                        if let fraction = active.fraction { ProgressView(value: fraction).frame(width: 150) } else { ProgressView().controlSize(.small) }
                        Text(active.state == "Queued" ? "Queued" : active.phaseTitle + (active.fraction.map { " · \(Int($0 * 100)) %" } ?? ""))
                            .font(.caption).foregroundStyle(.secondary)
                        Button("View Progress") { model.modelsSection = "transfers" }
                    }
                } else if let installed {
                    HStack(spacing: 8) {
                        PillBadge(text: "Downloaded", systemImage: "checkmark.circle.fill", tint: Brand.accentChrome, emphasized: true)
                        serveButton(path: installed.path, ready: models.canSelectInGUI(installed.path))
                        Menu {
                            Button("Show in Your Models") { model.modelSearch = installed.repositoryName; model.modelsSection = "library" }
                            Button("Show in Finder") { showInFinder(installed.path) }
                            Button("Download Again…") { redownloadTarget = item }.disabled(blocked != nil)
                            Divider()
                            Button("Remove…", role: .destructive) { removeTarget = installed }.disabled(removalReason(installed) != nil)
                        } label: { Image(systemName: "ellipsis.circle") }
                        .menuStyle(.borderlessButton).fixedSize()
                        .accessibilityLabel("More actions for " + item.alias)
                    }
                } else {
                    Button("Download", systemImage: "arrow.down") { requestDownload(item) }
                        .buttonStyle(.borderedProminent).disabled(blocked != nil)
                        .help(blocked ?? "Download this public AutomatosX pack")
                }
            }
            if let shortfall, installed == nil, active == nil {
                Label("Needs about \(bytes(shortfall)) more free space on the model volume.", systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            if let size = item.approxSizeBytes, size > memory {
                Label("Larger than this Mac's \(ByteCountFormatter.string(fromByteCount: memory, countStyle: .memory)) unified memory.", systemImage: "memorychip")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.padding(.vertical, 7)
    }

    // MARK: - Transfers

    private var transfers: some View {
        VStack(spacing: 16) {
            if models.downloads.isEmpty {
                ContentUnavailableView {
                    Label("No downloads in this session", systemImage: "arrow.down.circle")
                } description: {
                    Text("Downloads you start from Catalog appear here with their progress. The list clears when the app quits; downloaded models stay in Your Models.")
                } actions: {
                    HStack {
                        Button("Browse Catalog") { model.catalogSearch = ""; model.modelsSection = "catalog" }.buttonStyle(.borderedProminent)
                        Button("Your Models (\(models.productLibrary.count))") { model.modelSearch = ""; model.modelsSection = "library" }
                    }
                }
            } else {
                HStack {
                    Text(transferSummary).font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear Finished") { models.clearFinishedDownloads() }.disabled(models.finishedDownloadCount == 0)
                }
                Text("Downloads run one at a time and resume interrupted files on retry. Downloaded means complete, verified weight files; the runtime still validates compatibility on start.")
                    .font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading)
            }
            ForEach(models.downloads) { item in transferCard(item) }
            if !models.partialDownloads.isEmpty {
                Card { partialDownloads }
            }
        }.frame(maxWidth: .infinity)
    }

    private var transferSummary: String {
        let downloading = models.downloads.filter { $0.state == "Downloading" }.count
        let queued = models.downloads.filter { $0.state == "Queued" }.count
        var parts: [String] = []
        if downloading > 0 { parts.append("\(downloading) downloading") }
        if queued > 0 { parts.append("\(queued) queued") }
        if models.finishedDownloadCount > 0 { parts.append("\(models.finishedDownloadCount) finished") }
        return parts.joined(separator: " · ")
    }

    private func transferTint(_ state: String) -> Color {
        switch state {
        case "Ready", "Downloading": Brand.accentChrome
        case "Failed": .orange
        default: .secondary
        }
    }

    /// "Ready" is reserved for a serving server; a finished download reads
    /// "Downloaded" everywhere.
    private func transferState(_ item: DownloadItem) -> String {
        switch item.state {
        case "Ready": return "Downloaded"
        case "Queued":
            let queued = models.downloads.filter { $0.state == "Queued" }
            return (queued.firstIndex { $0.id == item.id }).map { "Queued · #\($0 + 1) of \(queued.count)" } ?? "Queued"
        default: return item.state
        }
    }

    private func transferCard(_ item: DownloadItem) -> some View {
        let label = ModelLabel(repositoryName: item.model.repoId.components(separatedBy: "/").last ?? item.model.repoId)
        return Card(label.text, subtitle: item.model.alias + " · " + item.model.repoId, trailing: {
            PillBadge(text: transferState(item), systemImage: item.state == "Downloading" ? "arrow.down.circle" : item.state == "Ready" ? "checkmark.circle.fill" : "circle",
                      tint: transferTint(item.state), emphasized: item.state != "Queued")
        }) {
            if item.state == "Downloading" {
                HStack(spacing: 10) {
                    if let fraction = item.fraction { ProgressView(value: fraction) } else { ProgressView().controlSize(.small); Spacer() }
                    Text(item.fraction.map { "\(Int($0 * 100)) %" } ?? "").font(.callout.monospacedDigit()).frame(minWidth: 44, alignment: .trailing)
                }
                HStack(spacing: 6) {
                    Text(item.phaseTitle).font(.callout.weight(.medium))
                    if let summary = item.progressSummary { Text("· " + summary).font(.callout).foregroundStyle(.secondary) }
                }
                if let stalled = item.stalledSeconds {
                    Label("No data for \(Int(stalled)) s. The transfer continues to wait for the server; cancel to stop it.", systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.orange)
                }
                if item.phase == .transfer || item.phase == nil {
                    Text(item.detail).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                }
            } else if item.state == "Failed" {
                ErrorBanner(text: item.detail)
            } else {
                Text(item.detail).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            HStack(spacing: 10) {
                if let summary = item.progressSummary, item.state == "Ready" { Text(summary).font(.caption).foregroundStyle(.secondary) }
                if let started = item.startedAt {
                    Text("Started " + started.formatted(date: .omitted, time: .shortened)
                         + (item.finishedAt.map { " · took " + TransferFormat.duration($0.timeIntervalSince(started)) } ?? ""))
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            HStack {
                Text(item.destination).font(.caption2).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                Spacer()
                if item.isActive {
                    Button(item.state == "Queued" ? "Remove from Queue" : "Cancel", role: .destructive) {
                        if item.state == "Queued" { models.cancel(item.id) } else { cancelTarget = item }
                    }
                }
                if ["Failed", "Cancelled"].contains(item.state) { Button("Retry") { models.retry(item.id, settings: model.settings) } }
                if item.state == "Ready" {
                    Button("Show in Finder", systemImage: "folder") { showInFinder(item.destination) }
                    serveButton(path: item.destination, ready: models.canSelectInGUI(item.destination), prominent: true)
                }
                if item.isFinished {
                    Menu {
                        if item.state == "Ready", let installed = models.installedModel(for: item.model) {
                            Button("Remove from Disk…", role: .destructive) { removeTarget = installed }
                                .disabled(removalReason(installed) != nil)
                            Divider()
                        }
                        Button("Dismiss") { models.clearFinishedDownloads(id: item.id) }
                    } label: { Image(systemName: "ellipsis.circle") }
                    .menuStyle(.borderlessButton).fixedSize()
                    .accessibilityLabel("More actions for " + label.title)
                }
            }
        }
    }
}
