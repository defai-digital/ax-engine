import SwiftUI
import AXEngineAppCore

/// First-run guide: what AX Engine does, one model to download, then Start.
/// Downloads keep running if the guide is closed; nothing here is required.
struct OnboardingView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @ObservedObject var runtime: RuntimeController
    let finish: () -> Void
    @State private var step = 0
    @State private var chosen: CatalogModel?

    private var memoryBytes: Int64 { Int64(ProcessInfo.processInfo.physicalMemory) }
    private var memoryText: String { ByteCountFormatter.string(fromByteCount: memoryBytes, countStyle: .memory) }
    private var installed: LocalModel? { chosen.flatMap(models.installedModel(for:)) }
    private var active: DownloadItem? { chosen.flatMap(models.activeDownload(for:)) }
    private var lastAttempt: DownloadItem? { chosen.flatMap { item in models.downloads.last { $0.model.target == item.target } } }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Step \(step + 1) of 3").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Skip") { finish() }.buttonStyle(.link)
            }.padding([.horizontal, .top], 22)
            Group {
                switch step {
                case 0: welcome
                case 1: pick
                default: serve
                }
            }
            .padding(22)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .frame(width: 560, height: 520)
        .task { if models.catalog.isEmpty && !models.refreshing { await models.refresh(settings: model.settings) } }
    }

    private var welcome: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let logo = Brand.logo { Image(nsImage: logo).resizable().frame(width: 56, height: 56) }
            Text("Welcome to AX Engine").font(.largeTitle.weight(.semibold))
            Text("Run AutomatosX models on this Mac and use them from your apps through an OpenAI- and Anthropic-compatible API on 127.0.0.1.")
                .font(.title3).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            VStack(alignment: .leading, spacing: 10) {
                Label("Download one public model. No account or token needed.", systemImage: "arrow.down.circle")
                Label("Start the server with one click. No API key needed by default; it listens on this Mac only.", systemImage: "play.circle")
                Label("Try it in Playground, or point your client at the URL.", systemImage: "bubble.left.and.bubble.right")
            }.font(.callout)
            Spacer()
            HStack {
                Spacer()
                Button("Get Started") { step = 1 }.buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            }
        }
    }

    private var pick: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Choose a model").font(.title2.weight(.semibold))
            Text("Models whose download is smaller than this Mac's \(memoryText) memory are listed first, smallest first. Download size is not a guarantee that a model runs well.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if models.refreshing { ProgressView("Reading the catalog…") }
            if let blocked = models.downloadsBlockReason, !models.refreshing {
                Label(blocked, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
            }
            ScrollView {
                VStack(spacing: 6) {
                    ForEach(FirstRun.ordered(models.catalog, memoryBytes: memoryBytes)) { item in row(item) }
                }
            }
            if models.catalog.isEmpty && !models.refreshing {
                Text("The catalog comes from your AX Engine runtime. Check Settings › Runtime, then try again.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            HStack {
                Button("Back") { step = 0 }
                Spacer()
                Button(installed != nil ? "Continue" : "Download and Continue") {
                    if let chosen, installed == nil, active == nil { models.enqueue(chosen, settings: model.settings) }
                    step = 2
                }
                .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                .disabled(chosen == nil || (installed == nil && (models.downloadsBlockReason != nil
                                                                  || chosen.flatMap(models.downloadBlockReason(for:)) != nil)))
            }
        }
    }

    private func row(_ item: CatalogModel) -> some View {
        let label = ModelLabel(repositoryName: item.repoId.components(separatedBy: "/").last ?? item.repoId)
        let selected = chosen?.id == item.id
        let downloaded = models.installedModel(for: item) != nil
        let larger = (item.approxSizeBytes ?? 0) > memoryBytes
        return Button { chosen = item } label: {
            HStack(spacing: 10) {
                Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(selected ? Brand.accentChrome : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    ModelTitle(label: label, font: .callout.weight(.medium))
                    Text(item.alias).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if downloaded { PillBadge(text: "Downloaded", systemImage: "checkmark.circle.fill", tint: Brand.accentChrome) }
                if let size = item.approxSizeBytes {
                    Text("about " + TransferFormat.bytes(size)).font(.caption.monospacedDigit())
                        .foregroundStyle(larger ? Color.orange : Color.secondary)
                        .help(larger ? "Download size is larger than this Mac's memory." : "Approximate download size")
                }
            }
            .padding(10).contentShape(Rectangle())
            .background(RoundedRectangle(cornerRadius: 8).fill(selected ? Brand.accentChrome.opacity(0.1) : Color.secondary.opacity(0.05)))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? [.isSelected] : [])
    }

    private var serve: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(installed != nil ? "Ready to serve" : "Downloading").font(.title2.weight(.semibold))
            if let chosen {
                ModelTitle(label: ModelLabel(repositoryName: chosen.repoId.components(separatedBy: "/").last ?? chosen.repoId), font: .headline)
            }
            if let active {
                VStack(alignment: .leading, spacing: 6) {
                    if let fraction = active.fraction { ProgressView(value: fraction) } else { ProgressView().controlSize(.small) }
                    Text(active.state == "Queued" ? "Waiting for another download" : active.phaseTitle + (active.progressSummary.map { " · " + $0 } ?? ""))
                        .font(.callout).foregroundStyle(.secondary)
                    Text("You can close this guide; the download continues in Models › Downloads.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else if installed == nil, let attempt = lastAttempt, ["Failed", "Cancelled"].contains(attempt.state) {
                ErrorBanner(text: attempt.detail)
                Button("Try Again") { models.retry(attempt.id, settings: model.settings) }
            } else if installed == nil, lastAttempt?.state == "Ready" {
                HStack {
                    Text(models.scanning ? "Downloaded. Checking your library…" : "Downloaded, but the library has not listed it yet.")
                        .foregroundStyle(.secondary)
                    if !models.scanning { Button("Check Again") { Task { await models.scan(settings: model.settings) } } }
                }
            } else if installed == nil && !models.scanning {
                Text("Waiting for the download to start…").foregroundStyle(.secondary)
            }
            if installed != nil {
                Text("Start Server loads the model and listens on http://127.0.0.1:\(String(model.settings.port))/v1. No API key is needed unless you turn one on in Server Settings.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            HStack {
                Button("Back") { step = 1 }
                Spacer()
                if installed == nil { Button("Close and Keep Downloading") { finish(); model.modelsSection = "transfers"; model.page = .models } }
                Button(runtime.owned ? "Select on Dashboard" : "Start Server", systemImage: "play.fill") {
                    if let installed { model.serve(installed.path) }
                    finish()
                }
                .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                .disabled(installed.map { !models.canSelectInGUI($0.path) } ?? true || model.attachedToService || model.connecting)
                .help(runtime.owned ? "A server is already running; Dashboard's Restart switches it to this model." : "Start serving this model")
            }
        }
    }
}
