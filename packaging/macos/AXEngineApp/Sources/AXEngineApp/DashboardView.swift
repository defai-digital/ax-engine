import SwiftUI
import AXEngineAppCore

struct DashboardView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore

    private var status: DashboardStatus {
        DashboardStatus.resolve(state: runtime.state, ready: runtime.ready, owned: runtime.owned,
            hasConnection: runtime.connection != nil,
            hasFiles: models.library.contains { $0.ready && models.checkedPaths.contains($0.path) },
            discovering: models.scanning, discoveryAttempted: models.discoveryAttempted,
            discoveryIssue: models.libraryIssue != nil,
            hasSelection: models.library.contains { $0.path == model.selectedPath && $0.ready && models.checkedPaths.contains($0.path) })
    }
    private var modelName: String {
        if let running = runtime.models.first { return running.id }
        let path = runtime.owned ? runtime.selectedModel : model.selectedPath
        return models.library.first(where: { $0.path == path })?.displayName
            ?? (path.isEmpty ? "No model selected" : URL(fileURLWithPath: path).lastPathComponent)
    }
    private var ownership: String {
        runtime.owned ? "Managed by this app" : runtime.connection != nil ? "External service · disconnect only" : "Not connected"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Card {
                    HStack {
                        PillBadge(text: runtime.state, systemImage: runtime.ready ? "checkmark.circle" : "circle", tint: runtime.ready ? Brand.accentChrome : .secondary)
                        Spacer()
                        Text(ownership).font(.callout).foregroundStyle(.secondary)
                    }
                    Text(status.title).font(.title2.weight(.semibold))
                    Text(status.detail).foregroundStyle(.secondary)
                    Label(modelName, systemImage: "cube").font(.headline).textSelection(.enabled)
                    if runtime.state == "Starting" || runtime.state == "Connecting", let started = runtime.startedAt {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            HStack {
                                ProgressView().controlSize(.small)
                                Text("Elapsed \(max(0, Int(context.date.timeIntervalSince(started))))s · waiting for readiness")
                                    .monospacedDigit().font(.callout)
                            }
                        }
                    }
                    if let error = runtime.error { ErrorBanner(text: error) }
                    if let checked = runtime.lastSuccessfulCheck {
                        Text("Last successful check: " + checked.formatted(date: .omitted, time: .standard))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    HStack {
                        Button(status.actionTitle, systemImage: status.action == .chat ? "bubble.left" : "arrow.right") { perform(status.action) }
                            .buttonStyle(.borderedProminent)
                        if status.action != .server { Button("Server details") { model.page = .server } }
                        if runtime.ready {
                            Button("Copy API URL", systemImage: "doc.on.doc") { copyText("http://127.0.0.1:\(model.settings.port)/v1") }
                        }
                    }
                }
                if let error = model.settingsError {
                    Card("Settings need attention") {
                        ErrorBanner(text: error)
                        Button("Review Settings") { model.page = .settings }
                    }
                }
                if let error = models.error {
                    Card("Model catalog needs attention") {
                        ErrorBanner(text: error)
                        Button("Review runtime settings") { model.page = .settings }
                    }
                }
                HStack(alignment: .top, spacing: 16) {
                    Card("Runtime snapshot", subtitle: "Reported by the connected service. Unavailable values show a dash.") {
                        MetricRow(label: "Active streams", value: number("ax_engine_generation_active_streams"))
                        MetricRow(label: "MLX active memory", value: memory("ax_engine_memory_mlx_active_bytes"))
                        MetricRow(label: "Process resident memory", value: memory("ax_engine_memory_host_resident_bytes"))
                        Button("View diagnostics") { model.page = .server }
                    }
                    Card("Model library", subtitle: "Downloaded files still require runtime compatibility checks.") {
                        MetricRow(label: "Discovered folders", value: String(models.library.count))
                        MetricRow(label: "Checked this scan", value: String(models.checkedPaths.count))
                        if models.scanning { ProgressView("Finding models…") }
                        else if models.libraryIssue != nil { Label("Some folders need another check", systemImage: "exclamationmark.triangle").font(.callout) }
                        else if let date = models.lastScanAt {
                            Text("Checked " + date.formatted(date: .omitted, time: .standard)).font(.caption).foregroundStyle(.secondary)
                        }
                        Button("Open library") { model.modelsSection = "library"; model.page = .models }
                    }
                }
                if let transfer = models.downloads.first(where: { ["Queued", "Downloading", "Failed", "Cancelled"].contains($0.state) }) {
                    Card("Model transfer", subtitle: transfer.model.alias) {
                        Text(transfer.state).font(.headline)
                        if let fraction = transfer.fraction { ProgressView(value: fraction) }
                        Text(transfer.detail).font(.callout).lineLimit(2)
                        Button("View transfers") { model.modelsSection = "transfers"; model.page = .models }
                    }
                }
                HStack {
                    Label("This Mac", systemImage: "desktopcomputer")
                    Text(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory) + " unified memory")
                    Spacer()
                    Text("Model fit is checked by AX Engine.")
                }.font(.caption).foregroundStyle(.secondary)
            }.padding(26)
        }
    }
    private func perform(_ action: DashboardStatus.Action) {
        switch action {
        case .models: model.modelsSection = "library"; model.page = .models
        case .catalog: model.modelSearch = ""; model.modelsSection = "catalog"; model.page = .models
        case .server: model.page = .server
        case .chat: model.page = .chat
        }
    }
    private func number(_ key: String) -> String { runtime.metrics[key].map { String(format: "%.0f", $0) } ?? "—" }
    private func memory(_ key: String) -> String {
        guard let bytes = runtime.metrics[key], bytes >= 0, bytes < Double(Int64.max) else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}
