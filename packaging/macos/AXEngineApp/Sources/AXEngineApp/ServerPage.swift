import SwiftUI
import AppKit
import AXEngineAppCore

/// Server page: app-owned server lifecycle, attach to an existing service,
/// client connection copying and runtime metrics.
struct ServerPage: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var chat: ChatStore
    @Binding var clientCopyNotice: String
    let clientClipboard: SensitiveClipboard
    @State private var apiKey = ""
    @State private var stopConfirmation = false
    @State private var restartTarget: String?

    private var attached: Bool { runtime.connection != nil && !runtime.owned }
    private var selectable: [LocalModel] { models.productLibrary.filter { $0.ready && models.checkedPaths.contains($0.path) } }
    private var startBlockReason: String? {
        ServerGate.startBlockReason(.init(selectedPath: model.selectedPath, selectable: models.canSelectInGUI(model.selectedPath),
                                          libraryEmpty: selectable.isEmpty, scanning: models.scanning,
                                          runtimeBusy: runtime.busy, removing: models.removing, attached: attached))
    }
    private var inferenceActive: Bool {
        chat.generating || (runtime.metrics["ax_engine_generation_active_streams"] ?? 0) > 0
    }
    private var transitioning: Bool { ["Starting", "Stopping", "Connecting"].contains(runtime.state) }

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                if let error = runtime.error {
                    ErrorBanner(text: error)
                    HStack {
                        if !runtime.owned && runtime.connection == nil && !runtime.selectedModel.isEmpty && models.canSelectInGUI(runtime.selectedModel) {
                            Button("Start Again", systemImage: "play.fill") { runtime.start(model: runtime.selectedModel, settings: model.settings) }
                                .disabled(runtime.busy || models.removing)
                        }
                        if !runtime.logs.isEmpty {
                            Button("Copy Diagnostics", systemImage: "doc.on.clipboard") {
                                copyText("AX Engine runtime error: " + (runtime.error ?? "") + "\n\n" + runtime.logs)
                                clientCopyNotice = "Diagnostics copied. They contain no API key."
                            }
                        }
                        Button("Review settings") { model.page = .settings }
                    }
                }
                if let checked = runtime.lastSuccessfulCheck {
                    Text("Last successful check: " + checked.formatted(date: .omitted, time: .standard)).font(.caption).foregroundStyle(.secondary)
                }
                serverCard
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
                    Card("Runtime output", subtitle: "Bounded local diagnostic output", trailing: {
                        Button("Copy", systemImage: "doc.on.doc") { copyText(runtime.logs) }.controlSize(.small)
                    }) {
                        ScrollView([.horizontal, .vertical]) {
                            Text(runtime.logs).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }.frame(height: 190)
                    }
                }
            }.padding(26)
        }
        .alert("Stop \(runningTitle)?", isPresented: $stopConfirmation) {
            Button("Stop", role: .destructive) { Task { await chat.stop(); await runtime.stop() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("A reply is being generated. Stopping the app-owned server ends it. You can start the server again at any time.") }
        .alert("Restart \(runningTitle)?", isPresented: Binding(get: { restartTarget != nil }, set: { if !$0 { restartTarget = nil } })) {
            Button("Restart", role: .destructive) {
                if let target = restartTarget { Task { await chat.stop(); await runtime.restart(model: target, settings: model.settings) } }
                restartTarget = nil
            }
            Button("Cancel", role: .cancel) { restartTarget = nil }
        } message: { Text("A reply is being generated. Restarting ends it; the server comes back with a new API key.") }
    }

    private var serverCard: some View {
        Card("Local server", subtitle: "Listening on loopback only. Existing external services are never stopped by this app.") {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : transitioning ? "clock" : "circle",
                              tint: runtime.ready ? Brand.accentChrome : runtime.state == "Failed" || runtime.state == "Connection degraded" ? .orange : .secondary,
                              emphasized: runtime.ready || transitioning)
                    if transitioning { ProgressView().controlSize(.small) }
                    if let detail = ServerGate.stateDetail(state: runtime.state, startedAt: runtime.startedAt, now: context.date) {
                        Text(detail).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(runtime.connection?.apiURL.absoluteString ?? "127.0.0.1:\(String(model.settings.port))")
                        .font(.system(.callout, design: .monospaced))
                }
            }
            if runtime.state == "Starting" {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Large models can take a few minutes from a cold start. Stop Server cancels the launch.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let line = RuntimeController.lastDiagnostic(runtime.logs) {
                        Text(line).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(2).truncationMode(.middle)
                    }
                }
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
            }
            if !clientCopyNotice.isEmpty {
                Text(clientCopyNotice).font(.caption).foregroundStyle(.secondary)
            }
            Divider()
            modelSelection
            actions
            Text("Startup is offline. MTP policy and expert streaming follow AX Engine defaults. No model performance or certification setting is changed.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var modelSelection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Model").font(.caption).foregroundStyle(.secondary)
                Menu {
                    if selectable.isEmpty {
                        Text("No installed models")
                        Button("Browse Catalog") { model.modelsSection = "catalog"; model.page = .models }
                    }
                    ForEach(selectable) { item in
                        Button {
                            model.selectedPath = item.path
                        } label: {
                            if model.selectedPath == item.path { Label(menuTitle(item), systemImage: "checkmark") } else { Text(menuTitle(item)) }
                        }
                    }
                } label: {
                    Text(selectedTitle).lineLimit(1).truncationMode(.middle)
                }.disabled(attached).frame(maxWidth: 420)
                if !model.selectedPath.isEmpty { Button("Show in Library") { model.modelSearch = selectedModel?.displayName ?? ""; model.modelsSection = "library"; model.page = .models } }
            }
            if !model.selectedPath.isEmpty {
                Text(model.selectedPath).font(.caption).foregroundStyle(.secondary).textSelection(.enabled).lineLimit(1).truncationMode(.middle)
            }
        }
    }

    private var selectedModel: LocalModel? { models.productLibrary.first { $0.path == model.selectedPath } }
    private var runningTitle: String {
        models.productLibrary.first { $0.path == runtime.selectedModel }?.displayName
            ?? (runtime.selectedModel.isEmpty ? "the local server" : URL(fileURLWithPath: runtime.selectedModel).lastPathComponent)
    }
    private var selectedTitle: String {
        if let selectedModel { return menuTitle(selectedModel) }
        return model.selectedPath.isEmpty ? "Choose an installed model" : URL(fileURLWithPath: model.selectedPath).lastPathComponent
    }
    private func menuTitle(_ item: LocalModel) -> String {
        item.displayName + (item.sizeBytes.map { " · " + TransferFormat.bytes($0) } ?? "")
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                if runtime.owned {
                    let switching = !model.selectedPath.isEmpty && model.selectedPath != runtime.selectedModel
                    Button(switching ? "Restart with Selected Model" : "Restart Server", systemImage: "arrow.clockwise") {
                        let target = switching && models.canSelectInGUI(model.selectedPath) ? model.selectedPath : runtime.selectedModel
                        if inferenceActive { restartTarget = target }
                        else { Task { await chat.stop(); await runtime.restart(model: target, settings: model.settings) } }
                    }
                    .disabled(runtime.busy || runtime.state == "Stopping" || models.removing
                              || (switching && !models.canSelectInGUI(model.selectedPath)))
                    .help(switching ? "Stop this server and start it again on the selected model. Active inference stops."
                                    : "Stop this server and start it again. Active inference stops.")
                    Button(runtime.state == "Starting" ? "Cancel Start" : "Stop Server", systemImage: "stop.fill", role: .destructive) {
                        if inferenceActive { stopConfirmation = true } else { Task { await chat.stop(); await runtime.stop() } }
                    }.disabled(runtime.state == "Stopping")
                    .help(runtime.state == "Starting" ? "Cancel the launch" : "Stop the app-owned server")
                } else if runtime.connection != nil {
                    Button("Disconnect") { Task { await chat.stop(); runtime.detach() } }
                } else {
                    Button("Start Server", systemImage: "play.fill") {
                        if models.canSelectInGUI(model.selectedPath) { runtime.start(model: model.selectedPath, settings: model.settings) }
                    }
                    .buttonStyle(.borderedProminent).disabled(startBlockReason != nil)
                    .help(startBlockReason ?? "Start a loopback server on the selected model")
                }
                Spacer()
            }
            if runtime.owned, !model.selectedPath.isEmpty, model.selectedPath != runtime.selectedModel {
                Label("The selected model differs from the running one. Restart to switch; the current server keeps serving until then.", systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            } else if !runtime.owned, runtime.connection == nil, let reason = startBlockReason {
                Label(reason, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func memoryMetric(_ key: String) -> String {
        guard let bytes = runtime.metrics[key], bytes >= 0, bytes < Double(Int64.max) else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
    private func metric(_ key: String) -> String { runtime.metrics[key].map { String(format: "%.0f", $0) } ?? "—" }
}
