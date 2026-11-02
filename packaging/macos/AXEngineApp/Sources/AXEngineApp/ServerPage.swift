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

    private var attached: Bool { runtime.connection != nil && !runtime.owned }
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
                    MetricRow(label: "Requests in current step", value: metric("ax_engine_step_scheduled_requests"))
                    MetricRow(label: "Batch capacity", value: metric("ax_runtime_max_batch_size"))
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
    }

    private var serverCard: some View {
        Card("Local server", subtitle: "Listening on loopback only. Existing external services are never stopped by this app.") {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : transitioning ? "clock" : "circle",
                              tint: runtime.ready ? Brand.accentChrome : runtime.state == "Failed" || runtime.state == "Connection degraded" ? .orange : .secondary,
                              emphasized: runtime.ready || transitioning)
                    if transitioning { ProgressView().controlSize(.small) }
                    if runtime.owned {
                        PillBadge(text: runtime.servingMode.badge, systemImage: "bolt")
                            .help("Speculative-decode posture this server was started with.")
                        PillBadge(text: "Up to \(runtime.concurrentRequests) requests", systemImage: "person.2")
                            .help("Requests this server decodes at once before new ones receive 429.")
                    }
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
                    .help(connection.hasAPIKey ? "Copy the API key for an external local client" : "This server does not use an API key")
                }
                Text(connection.hasAPIKey
                     ? "Copy the URL and API key into your local client. The copied key stays on this Mac and clears after 60 seconds if unchanged."
                     : "No API key: local clients on this Mac can call the URL directly.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !clientCopyNotice.isEmpty {
                Text(clientCopyNotice).font(.caption).foregroundStyle(.secondary)
            }
            Divider()
            ServerControls(model: model, runtime: runtime, models: models, chat: chat)
            Divider()
            apiKeySetting
            Text("Startup is offline. MTP policy and expert streaming follow AX Engine defaults; the serving mode only tunes speculative drafting. No model certification setting is changed.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var keyAwaitsRestart: Bool {
        runtime.owned && runtime.connection.map { $0.hasAPIKey != model.settings.requireAPIKey } == true
    }

    private var apiKeySetting: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle("Require an API key", isOn: Binding(get: { model.settings.requireAPIKey },
                                                         set: { model.selectRequireAPIKey($0) }))
                .toggleStyle(.switch).controlSize(.small)
                .disabled(attached || model.hasSettingsChanges)
            Text(apiKeyGuidance).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var apiKeyGuidance: String {
        if attached { return "An attached service keeps its own authentication. Enter its key under Connect." }
        if model.hasSettingsChanges { return "You have unsaved Settings changes. Save or revert them before changing authentication." }
        if keyAwaitsRestart {
            return model.settings.requireAPIKey
                ? "Saved for the next launch. Restart the server to require a key; it stays open until then."
                : "Saved for the next launch. Restart the server to drop the key; it still requires one until then."
        }
        return model.settings.requireAPIKey
            ? "Each launch gets a new in-memory key. Copy it into your client after the server starts or restarts."
            : "Off by default. The server listens on this Mac only (127.0.0.1), so local clients can call it without a key. Turn on if other local apps should need one."
    }

    private func memoryMetric(_ key: String) -> String {
        guard let bytes = runtime.metrics[key], bytes >= 0, bytes < Double(Int64.max) else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
    private func metric(_ key: String) -> String { runtime.metrics[key].map { String(format: "%.0f", $0) } ?? "—" }
}
