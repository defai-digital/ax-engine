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
    @State private var communityWarning: CatalogModel?
    @AppStorage("communityDownloadAcknowledged") private var communityAcknowledged = false
    @State private var cancelTarget: DownloadItem?
    @State private var largerExpanded = false
    @State private var inspected: String?
    @State private var libraryWidth: CGFloat = 1000

    private static let spaceMargin: Int64 = 64 * 1024 * 1024

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 14) {
                if let error = models.error {
                    ErrorBanner(text: error)
                    HStack {
                        Button("Review runtime settings") { model.openSettings() }
                        Button("Dismiss") { models.dismissError() }
                    }
                }
                HStack(spacing: 12) {
                    AXSegmented(options: [("library", "Your Models \(models.productLibrary.count)"),
                                          ("catalog", models.catalog.isEmpty ? "Catalog" : "Catalog \(models.catalog.count + models.visibleCommunityCatalog.count)"),
                                          ("transfers", models.activeDownloadCount > 0 ? "Downloads \(models.activeDownloadCount)" : "Downloads")],
                                selection: $model.modelsSection)
                    Spacer()
                    if model.modelsSection != "transfers" {
                        HStack(spacing: 6) {
                            Image(systemName: "magnifyingglass").font(.system(size: 12)).foregroundStyle(Brand.text3)
                            TextField(model.modelsSection == "catalog" ? "Search catalog" : "Filter your models",
                                      text: model.modelsSection == "catalog" ? $model.catalogSearch : $model.modelSearch)
                                .textFieldStyle(.plain).font(.system(size: 12.5)).focused($searchFocused)
                        }
                        .padding(.horizontal, 10).padding(.vertical, 6).frame(maxWidth: 260)
                        .background(Brand.panel, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Brand.hairlineColor))
                        Button { Task { await models.refresh(settings: model.settings) } } label: { Image(systemName: "arrow.clockwise") }
                            .buttonStyle(.ax(.secondary)).help("Refresh the library and catalog")
                            .accessibilityLabel("Refresh")
                            .disabled(models.refreshing || models.scanning)
                    }
                }
            }.padding(.horizontal, 24).padding(.top, 8).padding(.bottom, 14)
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    switch model.modelsSection {
                    case "catalog": catalog
                    case "transfers": transfers
                    default: library(filter: model.modelSearch)
                    }
                }.padding([.horizontal, .bottom], 24)
            }
        }
        .background(Brand.canvas)
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
            Button("Change Download Folder…") { spaceWarning = nil; model.openSettings() }
            Button("Cancel", role: .cancel) { spaceWarning = nil }
        } message: { Text(spaceWarning.map(spaceMessage) ?? "") }
        .alert("Download a model without official support?", isPresented: Binding(get: { communityWarning != nil }, set: { if !$0 { communityWarning = nil } })) {
            Button("Download") {
                communityAcknowledged = true
                if let item = communityWarning { requestDownload(item) }
                communityWarning = nil
            }
            Button("Cancel", role: .cancel) { communityWarning = nil }
        } message: {
            Text((communityWarning.map { "\($0.repoId) is published by \($0.repoId.components(separatedBy: "/").first ?? "another publisher"), not AutomatosX. " } ?? "")
                 + "AutomatosX has not tested it. The download is anonymous and every file is checksum-verified; the runtime checks compatibility when you start the model and may refuse it. You can remove it from Your Models. This is asked once.")
        }
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
        if !item.isOfficial && !communityAcknowledged { communityWarning = item; return }
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
    /// model on Home, where Restart switches to it.
    private func serveButton(path: String, ready: Bool, prominent: Bool = false) -> some View {
        let busy = runtime.owned
        let reason: String? = !ready ? "Weights are incomplete or not checked yet. Refresh, or download it again from Catalog."
            : model.attachedToService ? "Disconnect from the attached service on API & Logs to serve your own model."
            : model.connecting ? "Wait for the connection to the existing service to finish."
            : model.models.canSelectInGUI(path) ? nil
            : models.scope == .automatosX && !models.productLibrary.contains(where: { $0.path == path })
                && models.library.contains(where: { $0.path == path && $0.ready }) && models.checkedPaths.contains(path)
                ? "Switch Your Models to All models to serve a model without official support."
            : "The library scan has not checked this model yet."
        return Group {
            if prominent {
                Button(busy ? "Select" : "Serve", systemImage: "play.fill") { model.serve(path) }.buttonStyle(.ax(.serve))
            } else {
                Button(busy ? "Select on Home" : "Serve", systemImage: "play.fill") { model.serve(path) }.buttonStyle(.ax(.secondary))
            }
        }
        .disabled(reason != nil)
        .help(reason ?? (busy ? "Select this model on Home; Restart there switches the server to it." : "Start the server on this model"))
        .accessibilityHint(reason ?? "")
    }

    // MARK: - Local Library

    private func library(filter: String) -> some View {
        let matches = models.productLibrary.filter {
            filter.isEmpty || $0.path.localizedCaseInsensitiveContains(filter) || $0.name.localizedCaseInsensitiveContains(filter)
        }.sorted { left, right in
            let l = left.ready && models.checkedPaths.contains(left.path)
            let r = right.ready && models.checkedPaths.contains(right.path)
            return l == r ? left.path < right.path : l && !r
        }
        let selected = matches.first { $0.path == inspected } ?? matches.first
        // Width decides the layout: titles are long, so ideal-size fitting
        // would always fall back to the stacked form.
        return libraryLayout(matches: matches, selected: selected, filter: filter,
                             side: libraryWidth >= 700, format: libraryWidth >= 960 || libraryWidth < 700)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { libraryWidth = $0 }
    }

    @ViewBuilder private func libraryLayout(matches: [LocalModel], selected: LocalModel?, filter: String, side: Bool, format: Bool) -> some View {
        let layout = side ? AnyLayout(HStackLayout(alignment: .top, spacing: 16)) : AnyLayout(VStackLayout(alignment: .leading, spacing: 16))
        layout {
            VStack(alignment: .leading, spacing: 0) {
                if models.scanning {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text("Checking folders · \(models.checkedPaths.count) found").font(Brand.Font.caption).foregroundStyle(Brand.text2)
                        Spacer()
                        Button("Stop scan") { models.cancelScan() }.buttonStyle(.axLink)
                    }.padding(.horizontal, 20).padding(.vertical, 10)
                }
                if let issue = models.libraryIssue {
                    VStack(alignment: .leading) {
                        ErrorBanner(text: issue)
                        Button("Retry scan") { Task { await models.scan(settings: model.settings) } }.disabled(models.scanning)
                    }.padding(16)
                }
                // Inside the table, not the page bar: the bar has no room at the minimum width.
                HStack(spacing: 10) {
                    AXSegmented(options: LibraryScope.allCases.map { ($0, $0.title) }, selection: $model.libraryScope, compact: true)
                        .accessibilityLabel("Models shown")
                    Text(model.libraryScope == .automatosX ? "Packs published by AutomatosX."
                         : "Every model folder found on this Mac. The runtime checks compatibility at start.")
                        .font(.system(size: 11.5)).foregroundStyle(Brand.text3).lineLimit(2)
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 20).padding(.top, 12)
                .help("All models also lists models AutomatosX did not publish, and the Catalog adds community models. AX Engine removes only folders it downloaded and AutomatosX repositories in the Hugging Face cache.")
                HStack(spacing: 0) {
                    Text("Model").frame(maxWidth: .infinity, alignment: .leading)
                    if format { Text("Format").frame(width: 120, alignment: .leading) }
                    Text("Size").frame(width: 80, alignment: .leading)
                    Text("Status").frame(width: 116, alignment: .leading)
                }
                .font(.system(size: 11.5, weight: .semibold)).foregroundStyle(Brand.text3)
                .padding(.horizontal, 20).padding(.vertical, 12)
                Divider().overlay(Brand.hairlineColor)
                if models.productLibrary.isEmpty && !models.scanning && models.libraryIssue == nil {
                    ContentUnavailableView {
                        Label("No models yet", systemImage: "square.stack.3d.up")
                    } description: {
                        Text(models.hiddenCount > 0
                             ? "Download a public AutomatosX model from the Catalog, or show the \(models.hiddenCount) other model\(models.hiddenCount == 1 ? "" : "s") found on this Mac."
                             : "Download a public AutomatosX model from the Catalog. No account or token needed.")
                    } actions: {
                        Button("Browse Catalog") { model.catalogSearch = ""; model.modelsSection = "catalog" }.buttonStyle(.axPrimary)
                        if models.hiddenCount > 0 { Button("Show All Models") { model.libraryScope = .all }.buttonStyle(.ax(.secondary)) }
                    }.padding(.vertical, 30)
                }
                if matches.isEmpty && !filter.isEmpty {
                    HStack {
                        Text("No folders match this filter.").foregroundStyle(Brand.text3)
                        Button("Clear search") { model.modelSearch = "" }.buttonStyle(.axLink)
                    }.padding(20)
                }
                ForEach(matches) { item in
                    libraryRow(item, selected: item.path == selected?.path, format: format)
                }
                Spacer(minLength: 0)
                HStack {
                    Text(libraryFooter).font(.system(size: 11.5)).foregroundStyle(Brand.text3)
                    Spacer()
                    if models.hiddenCount > 0 {
                        Button("\(models.hiddenCount) other model\(models.hiddenCount == 1 ? "" : "s") on disk hidden · Show") { model.libraryScope = .all }
                            .buttonStyle(.axLink)
                            .help("Model folders found on this Mac that AutomatosX did not publish")
                    }
                    if !models.partialDownloads.isEmpty {
                        Button("\(models.partialDownloads.count) interrupted download\(models.partialDownloads.count == 1 ? "" : "s") on disk") {
                            model.modelsSection = "transfers"
                        }.buttonStyle(.axLink)
                    }
                }
                .padding(.horizontal, 20).padding(.vertical, 14)
            }
            .frame(maxWidth: .infinity, minHeight: 420, alignment: .top)
            .panelSurface()
            if let selected {
                libraryInspector(selected).frame(width: side ? (format ? 300 : 270) : nil)
            }
        }
    }

    private var libraryFooter: String {
        let total = models.productLibrary.compactMap(\.sizeBytes).reduce(0, +)
        return "\(models.productLibrary.count) model\(models.productLibrary.count == 1 ? "" : "s") · \(model.servableModels.count) ready to serve"
            + (total > 0 ? " · " + bytes(total) : "")
            + (models.lastScanAt.map { " · checked " + $0.formatted(date: .omitted, time: .shortened) } ?? "")
    }

    private func libraryStatus(_ item: LocalModel) -> (String, Color) {
        let running = runtime.ready && runtime.owned && runtime.selectedModel == item.path
        let checked = models.checkedPaths.contains(item.path)
        if running { return ("Serving", Brand.serve) }
        if runtime.owned && runtime.selectedModel == item.path { return (runtime.state, Brand.accent) }
        if !checked { return ("Not checked", Brand.text3) }
        return item.ready ? (model.selectedPath == item.path ? "Selected" : "Ready", Brand.text2) : ("Incomplete", Brand.warn)
    }

    private func libraryRow(_ item: LocalModel, selected: Bool, format: Bool) -> some View {
        let checked = models.checkedPaths.contains(item.path)
        let status = libraryStatus(item)
        let running = status.0 == "Serving"
        let state = removalState(item)
        let reason = RemovalGate.reason(state)
        return HStack(spacing: 0) {
            HStack(spacing: 12) {
                ModelAvatar(name: item.label.title)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(item.label.title).font(.system(size: 13, weight: .semibold)).foregroundStyle(Brand.text1).lineLimit(1)
                        if model.favoriteModels.contains(item.path) {
                            Image(systemName: "star.fill").font(.system(size: 9)).foregroundStyle(Brand.accent)
                        }
                        if !item.isAutomatosX { Tag(LibraryScope.unofficialTag).help(LibraryScope.unofficialNote) }
                    }
                    Text(sourceLine(item)).font(.system(size: 11.5)).foregroundStyle(Brand.text3).lineLimit(1)
                }
            }
            .frame(minWidth: 180, maxWidth: .infinity, alignment: .leading)
            if format {
                Text(item.label.tags.isEmpty ? "—" : item.label.tags.joined(separator: " · "))
                    .font(.system(size: 12.5)).foregroundStyle(Brand.text2).lineLimit(1).frame(width: 120, alignment: .leading)
            }
            Text(item.sizeBytes.map(bytes) ?? "—").font(.system(size: 12.5, weight: .semibold)).monospacedDigit()
                .lineLimit(1).frame(width: 80, alignment: .leading)
            HStack {
                if selected && !(runtime.owned && runtime.selectedModel == item.path) && item.ready && checked {
                    serveButton(path: item.path, ready: true, prominent: true).controlSize(.small)
                } else {
                    StatusPill(text: status.0, tint: status.1,
                               systemImage: status.0 == "Incomplete" ? "exclamationmark.triangle.fill" : nil)
                }
            }
            .frame(width: 116, alignment: .leading)
        }
        .padding(.horizontal, 20).padding(.vertical, 11)
        .background(selected ? Brand.accent.opacity(0.10) : .clear)
        .contentShape(Rectangle())
        .onTapGesture { inspected = item.path }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(item.label.title + (item.isAutomatosX ? "" : ", not officially supported") + ", " + status.0)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityAction(named: "Inspect") { inspected = item.path }
        .contextMenu {
            Button("Serve") { model.serve(item.path) }.disabled(!item.ready || !checked || running)
            Button(model.favoriteModels.contains(item.path) ? "Unpin" : "Pin to Sidebar") { model.toggleFavoriteModel(item.path) }
            Button("Show in Finder") { showInFinder(item.path) }
            Button("Copy Path") { copyText(item.path) }
            if state.removable {
                Divider()
                Button("Remove…", role: .destructive) { removeTarget = item }.disabled(reason != nil)
            }
        }
    }

    /// Source plus the publishing organization for rows AutomatosX did not publish.
    private func sourceLine(_ item: LocalModel) -> String {
        guard !item.isAutomatosX, let organization = item.organization else { return item.source }
        return item.source + " · " + organization
    }

    private func libraryInspector(_ item: LocalModel) -> some View {
        let checked = models.checkedPaths.contains(item.path)
        let status = libraryStatus(item)
        let state = removalState(item)
        let reason = RemovalGate.reason(state)
        let catalogEntry = models.catalog.first { models.installedModel(for: $0)?.path == item.path }
        return VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text(item.label.title).font(.system(size: 17, weight: .bold)).textSelection(.enabled)
                WrapLayout(spacing: 6) {
                    StatusPill(text: status.0, tint: status.1)
                    ForEach(item.label.tags, id: \.self) { Tag($0, tint: $0 == "MTP" ? Brand.boost : nil) }
                    if !item.isAutomatosX { Tag(LibraryScope.unofficialTag) }
                    if let certification = catalogEntry?.certification { Tag(certification.capitalized, tint: Brand.accent) }
                }
            }
            HStack(spacing: 8) {
                if status.0 != "Serving" { serveButton(path: item.path, ready: item.ready && checked, prominent: true) }
                Button(model.favoriteModels.contains(item.path) ? "Unpin" : "Pin", systemImage: model.favoriteModels.contains(item.path) ? "star.slash" : "star") {
                    model.toggleFavoriteModel(item.path)
                }.buttonStyle(.ax(.secondary))
                .help("Pinned models appear in the sidebar and serve in one click")
            }
            SectionLabel("Size")
            VStack(alignment: .leading, spacing: 6) {
                Text(item.sizeBytes.map { bytes($0) + " on disk" } ?? "Size unknown").font(.system(size: 13, weight: .semibold))
                if let size = item.sizeBytes {
                    StackedBar(segments: [.init(id: "size", value: Double(size), color: Brand.accent)],
                               total: Double(ProcessInfo.processInfo.physicalMemory), height: 6)
                    Text("Against " + HostInfo.memory + " unified memory · size comparison only, not a fit guarantee")
                        .font(.system(size: 11)).foregroundStyle(Brand.text3).fixedSize(horizontal: false, vertical: true)
                }
            }
            SectionLabel("Details")
            VStack(spacing: 9) {
                detail("Repository", item.repositoryName)
                if let revision = catalogEntry?.revision { detail("Revision", String(revision.prefix(8))) }
                detail("Source", sourceLine(item))
                detail("Files", checked ? (item.ready ? "Complete" : "Incomplete") : "Not checked")
                detail("Location", (item.path as NSString).abbreviatingWithTildeInPath)
            }
            Text(item.isAutomatosX ? "File completeness does not establish compatibility. AX Engine validates the model when you start it."
                                   : LibraryScope.unofficialNote)
                .font(.system(size: 11)).foregroundStyle(Brand.text3).fixedSize(horizontal: false, vertical: true)
            Divider().overlay(Brand.hairlineColor)
            HStack(spacing: 4) {
                Button("Show in Finder", systemImage: "folder") { showInFinder(item.path) }.buttonStyle(.axLink)
                Button("Copy Path", systemImage: "doc.on.doc") { copyText(item.path) }.buttonStyle(.axLink)
            }
            HStack {
                Spacer()
                if state.removable {
                    Button("Remove…", systemImage: "trash") { removeTarget = item }
                        .buttonStyle(.ax(.danger)).disabled(reason != nil)
                        .help(reason ?? "Delete this downloaded model from disk")
                        .accessibilityLabel("Remove " + item.displayName)
                }
            }
            if let reason {
                Label(reason, systemImage: state.removable ? "clock" : "lock").font(.system(size: 11)).foregroundStyle(Brand.text3)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .panelSurface()
    }

    private func detail(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).foregroundStyle(Brand.text3)
            Spacer(minLength: 10)
            Text(value).fontWeight(.medium).lineLimit(1).truncationMode(.middle).textSelection(.enabled).help(value)
        }
        .font(.system(size: 12))
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
        let matching = { (item: CatalogModel) in search.isEmpty || item.alias.localizedCaseInsensitiveContains(search) || item.repoId.localizedCaseInsensitiveContains(search) }
        let matches = models.catalog.filter(matching)
        let community = models.visibleCommunityCatalog.filter(matching)
        // Download size against unified memory: a size comparison, not a fit claim.
        let fits = matches.filter { ($0.approxSizeBytes ?? 0) <= memoryBytes }
        let larger = matches.filter { ($0.approxSizeBytes ?? 0) > memoryBytes }
        return Card("Catalog", subtitle: "Public AutomatosX models, plus community models under All models. Profiles come from your AX Engine runtime; an entry is not a hardware qualification.") {
            HStack(spacing: 10) {
                AXSegmented(options: LibraryScope.allCases.map { ($0, $0.title) }, selection: $model.libraryScope, compact: true)
                    .accessibilityLabel("Models shown")
                Text(model.libraryScope == .automatosX ? "Officially supported AutomatosX packs."
                     : "Also lists other publishers' models the runtime has a profile for, without official support.")
                    .font(.system(size: 11.5)).foregroundStyle(Brand.text3).lineLimit(2)
                Spacer(minLength: 0)
            }
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Label("No Hugging Face account or token required", systemImage: "arrow.down.circle").font(.headline)
                    Text("Downloads connect to Hugging Face and its delivery hosts. Models from other publishers that are already on this Mac appear under Your Models › All models.")
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
                        Button("Open Settings") { model.openSettings() }
                        Button("Refresh Catalog") { Task { await models.refresh(settings: model.settings) } }.buttonStyle(.borderedProminent)
                    }
                }
            }
            if matches.isEmpty && community.isEmpty && !search.isEmpty && !models.refreshing {
                Text("No catalog models match “\(search)”.").foregroundStyle(.secondary)
                Button("Clear search") { model.catalogSearch = "" }
            }
            if !fits.isEmpty && !larger.isEmpty {
                Text("Download size under this Mac's \(ByteCountFormatter.string(fromByteCount: memoryBytes, countStyle: .memory)) memory")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 340), spacing: 14)], spacing: 14) {
                ForEach(fits) { item in catalogCard(item) }
            }
            if !larger.isEmpty {
                DisclosureGroup(isExpanded: $largerExpanded) {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 340), spacing: 14)], spacing: 14) {
                        ForEach(larger) { item in catalogCard(item) }
                    }.padding(.top, 10)
                } label: {
                    Text("Download size over this Mac's memory (\(larger.count))")
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                }
            }
            if model.libraryScope == .all && !models.refreshing && (models.communityCatalog.isEmpty ? !models.catalog.isEmpty : !community.isEmpty) {
                Divider()
                VStack(alignment: .leading, spacing: 4) {
                    Text("Community models · not officially supported")
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary).textCase(.uppercase)
                    Text(models.communityCatalog.isEmpty
                         ? "This AX Engine runtime does not list community models. Update the runtime, then refresh."
                         : "Published by others and not tested by AutomatosX. You are free to try them: downloads use the same anonymous, checksum-verified path, and the runtime checks compatibility at start and may refuse a model.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 340), spacing: 14)], spacing: 14) {
                    ForEach(community) { item in catalogCard(item) }
                }
            }
        }
    }

    private func catalogCard(_ item: CatalogModel) -> some View {
        catalogRow(item)
            .padding(16)
            .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
            .background(Brand.inset, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Brand.hairlineColor))
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
                    HStack(spacing: 8) {
                        ModelAvatar(name: label.title)
                        ModelTitle(label: label, font: .system(size: 13.5, weight: .semibold)).help(item.repoId)
                    }
                    Text(item.alias + " · " + item.repoId + (!item.isOfficial && item.revision == nil ? " · publisher's latest revision" : ""))
                        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    HStack(spacing: 6) {
                        if !item.isOfficial {
                            PillBadge(text: LibraryScope.unofficialTag, systemImage: "person.2", tint: .orange).fixedSize()
                                .help(LibraryScope.unofficialNote)
                        }
                        if let size = item.approxSizeBytes { PillBadge(text: "about " + bytes(size), systemImage: "internaldrive").fixedSize() }
                        if item.mtpIncluded == true { PillBadge(text: "MTP included", systemImage: "bolt") }
                        if item.isOfficial, let certification = item.certification { PillBadge(text: certification.capitalized, systemImage: "checkmark.seal") }
                        // Community entries say "publisher's latest revision" on the line above; this row has no room.
                        if item.isOfficial || item.revision != nil {
                            Text(item.revision.map { "Revision " + String($0.prefix(12)) } ?? "Unpinned runtime profile").font(.caption).foregroundStyle(.secondary)
                        }
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
                        PillBadge(text: "Downloaded", systemImage: "checkmark.circle.fill", tint: Brand.accent, emphasized: true)
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
                        .buttonStyle(.ax(.primary)).disabled(blocked != nil)
                        .help(blocked ?? (item.isOfficial ? "Download this public AutomatosX pack" : "Download this community model. " + LibraryScope.unofficialNote))
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
        case "Ready", "Downloading": Brand.accent
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
                    HStack {
                        Label("No data for \(Int(stalled)) s. It keeps waiting for the server; Restart Download reconnects and keeps verified files.", systemImage: "exclamationmark.triangle")
                            .font(.caption).foregroundStyle(.orange)
                        Spacer()
                        Button("Restart Download", systemImage: "arrow.clockwise") { models.restartDownload(item.id) }.controlSize(.small)
                    }
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
