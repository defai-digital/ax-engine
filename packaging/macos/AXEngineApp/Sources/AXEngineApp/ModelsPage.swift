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
                    Text("Local Library").tag("library")
                    Text("Catalog").tag("catalog")
                    Text(models.activeDownloadCount > 0 ? "Transfers (\(models.activeDownloadCount))" : "Transfers").tag("transfers")
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
        .alert("Not enough free space", isPresented: Binding(get: { spaceWarning != nil }, set: { if !$0 { spaceWarning = nil } })) {
            Button("Download Anyway") { if let item = spaceWarning { startDownload(item) }; spaceWarning = nil }
            Button("Cancel", role: .cancel) { spaceWarning = nil }
        } message: { Text(spaceWarning.map(spaceMessage) ?? "") }
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
        return "\(item.alias) needs about \(bytes(needed)) but \(snapshot.volumeName) has \(bytes(snapshot.availableBytes)) available. The runtime refuses to stage a download it cannot verify, so free up space or choose another model folder in Settings first."
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

    // MARK: - Local Library

    private func library(filter: String) -> some View {
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
                libraryRow(item)
                Divider()
            }
            if !models.partialDownloads.isEmpty { partialDownloads }
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
                        Text(item.displayName).font(.headline).lineLimit(2)
                        if running { PillBadge(text: "Running in this app", systemImage: "circle.fill", tint: Brand.accentChrome, emphasized: true) }
                        else if model.selectedPath == item.path { PillBadge(text: "Selected for Server", systemImage: "checkmark") }
                    }
                    Text(item.source + (item.sizeBytes.map { " · " + bytes($0) } ?? "")).font(.caption).foregroundStyle(.secondary)
                    Text(checked ? (item.ready ? "Downloaded · compatibility unverified" : "Weights incomplete or metadata only")
                                 : "Not checked in this scan · previous result")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Select Model") { model.selectedPath = item.path; model.page = .server }
                    .disabled(!item.ready || !checked)
                    .accessibilityLabel("Select " + item.displayName)
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
            Button("Select Model") { model.selectedPath = item.path; model.page = .server }.disabled(!item.ready || !checked)
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
                        Text((partial.sizeBytes.map(bytes) ?? "Size unknown") + (resuming ? " · resuming in Transfers" : "")).font(.caption).foregroundStyle(.secondary)
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

    private var catalog: some View {
        Card("Public AutomatosX models", subtitle: "Pinned profiles come from your installed AX Engine runtime. A catalog entry is not a hardware qualification.") {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Label("No Hugging Face account or token required", systemImage: "arrow.down.circle").font(.headline)
                    Text("Downloads connect to Hugging Face and its delivery hosts. Only public AutomatosX packs are offered; use other models through the AX Engine CLI.")
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
                }
            }
            Divider()
            if models.refreshing { ProgressView("Reading runtime catalog…") }
            if let blocked = models.downloadsBlockReason, !models.refreshing {
                Label(blocked, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
            }
            if models.catalog.isEmpty && !models.refreshing { Text("Select your AX Engine runtime in Settings, then refresh.").foregroundStyle(.secondary) }
            let matches = models.catalog.filter { model.modelSearch.isEmpty || $0.alias.localizedCaseInsensitiveContains(model.modelSearch) || $0.repoId.localizedCaseInsensitiveContains(model.modelSearch) }
            if matches.isEmpty && !model.modelSearch.isEmpty && !models.refreshing {
                Text("No catalog profiles match your search.").foregroundStyle(.secondary)
                Button("Clear search") { model.modelSearch = "" }
            }
            ForEach(matches) { item in
                catalogRow(item)
                Divider()
            }
        }
    }

    private func catalogRow(_ item: CatalogModel) -> some View {
        let installed = models.installedModel(for: item)
        let active = models.activeDownload(for: item)
        let blocked = models.downloadBlockReason(for: item)
        let shortfall = spaceShortfall(item)
        let memory = Int64(ProcessInfo.processInfo.physicalMemory)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.alias).font(.headline)
                    Text(item.repoId).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
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
                        Button("View Transfer") { model.modelsSection = "transfers" }
                    }
                } else if let installed {
                    HStack(spacing: 8) {
                        PillBadge(text: "Downloaded", systemImage: "checkmark.circle.fill", tint: Brand.accentChrome, emphasized: true)
                        Button("Use Model") { model.selectedPath = installed.path; model.page = .server }
                            .disabled(!models.canSelectInGUI(installed.path))
                        Menu {
                            Button("Show in Library") { model.modelSearch = installed.displayName; model.modelsSection = "library" }
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
                ContentUnavailableView("No transfers in this session", systemImage: "arrow.down.circle",
                                       description: Text("Downloads you start from Catalog appear here with their progress. The list clears when the app quits; downloaded models stay in Local Library."))
                HStack {
                    Button("Browse Catalog") { model.modelSearch = ""; model.modelsSection = "catalog" }
                    Button("View Local Library (\(models.productLibrary.count))") { model.modelSearch = ""; model.modelsSection = "library" }
                }
            } else {
                HStack {
                    Text(transferSummary).font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear Finished") { models.clearFinishedDownloads() }.disabled(models.finishedDownloadCount == 0)
                }
                Text("Transfers run one at a time and resume interrupted files on retry. Ready means complete, verified weight files; the runtime still validates compatibility on start.")
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

    private func transferCard(_ item: DownloadItem) -> some View {
        Card(item.model.alias, subtitle: item.model.repoId, trailing: {
            PillBadge(text: item.state, systemImage: item.state == "Downloading" ? "arrow.down.circle" : item.state == "Ready" ? "checkmark.circle.fill" : "circle",
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
                if item.isActive { Button("Cancel", role: .destructive) { models.cancel(item.id) } }
                if ["Failed", "Cancelled"].contains(item.state) { Button("Retry") { models.retry(item.id, settings: model.settings) } }
                if item.state == "Ready" {
                    Button("Show in Finder", systemImage: "folder") { showInFinder(item.destination) }
                    if let installed = models.installedModel(for: item.model) {
                        Button("Remove…", systemImage: "trash", role: .destructive) { removeTarget = installed }
                            .disabled(removalReason(installed) != nil)
                            .help(removalReason(installed) ?? "Delete this downloaded model from disk")
                    }
                    Button("Use Model") { model.selectedPath = item.destination; model.page = .server }
                        .buttonStyle(.borderedProminent).disabled(!models.canSelectInGUI(item.destination))
                }
                if item.isFinished { Button("Dismiss") { models.clearFinishedDownloads(id: item.id) } }
            }
        }
    }
}
