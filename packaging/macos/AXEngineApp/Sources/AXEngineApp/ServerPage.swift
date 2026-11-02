import SwiftUI
import AppKit
import AXEngineAppCore

/// Server Settings: the app-owned server in detail (lifecycle, serving
/// options, API key, client connection), attaching to an existing service,
/// and runtime metrics and output once something is serving.
struct ServerPage: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var chat: ChatStore
    @Binding var clientCopyNotice: String
    let clientClipboard: SensitiveClipboard
    @State private var apiKey = ""
    @State private var connectExpanded = false
    @State private var outputExpanded = false
    @State private var diagnosticsNotice = ""
    @State private var logQuery = ""
    @State private var problemsOnly = false
    @State private var logLines = 500

    private var attached: Bool { runtime.connection != nil && !runtime.owned }

    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                if let error = runtime.error { errorCard(error) }
                serverCard
                if runtime.connection == nil && !runtime.owned { connectCard }
                if runtime.connection != nil {
                    metricsCard
                    RuntimeHistoryView(runtime: runtime)
                } else {
                    Card {
                        Label("Runtime metrics and history appear here while a server is running. Mac CPU, GPU and memory charts are on Dashboard.",
                              systemImage: "chart.xyaxis.line")
                            .font(.callout).foregroundStyle(.secondary)
                        Button("Open Dashboard") { model.page = .overview }.buttonStyle(.link)
                    }
                }
                if !runtime.logs.isEmpty { outputCard }
            }.padding(26)
        }
        .onAppear { if runtime.error != nil { outputExpanded = true } }
        .onChange(of: runtime.error) { _, error in if error != nil { outputExpanded = true }; diagnosticsNotice = "" }
    }

    private func errorCard(_ error: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ErrorBanner(text: error)
            HStack {
                if !runtime.owned && runtime.connection == nil && !runtime.selectedModel.isEmpty && models.canSelectInGUI(runtime.selectedModel) {
                    Button("Start Again", systemImage: "play.fill") {
                        model.selectedPath = runtime.selectedModel
                        model.startServer()
                    }.disabled(runtime.busy || models.removing || models.scanning)
                }
                if !runtime.logs.isEmpty {
                    Button("Copy Diagnostics", systemImage: "doc.on.clipboard") {
                        copyText("AX Engine runtime error: " + (runtime.error ?? "") + "\n\n" + runtime.logs)
                        diagnosticsNotice = "Diagnostics copied. They contain no API key."
                    }
                }
                Button("Review Settings") { model.page = .settings }
            }
            if !diagnosticsNotice.isEmpty { Text(diagnosticsNotice).font(.caption).foregroundStyle(.secondary) }
        }
    }

    private var serverCard: some View {
        Card("Local server", subtitle: "Listens on this Mac only (127.0.0.1). Services started elsewhere are never stopped by this app.") {
            ServerStatusRow(model: model, runtime: runtime)
            if let checked = runtime.lastSuccessfulCheck {
                Text("Last successful check: " + checked.formatted(date: .omitted, time: .standard)).font(.caption).foregroundStyle(.secondary)
            }
            ClientCopyButtons(runtime: runtime, clipboard: clientClipboard, notice: $clientCopyNotice)
            Divider()
            ServerControls(model: model, runtime: runtime, models: models, chat: chat)
            Divider()
            ServingOptions(model: model, runtime: runtime)
            Divider()
            apiKeySetting
            Text("Startup is offline. MTP policy and expert streaming follow AX Engine defaults; the serving mode only tunes speculative drafting.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var connectCard: some View {
        Card {
            DisclosureGroup("Already running AX Engine on this Mac? Connect to it", isExpanded: $connectExpanded) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Attaches to the service on port \(String(model.settings.port)) (change it in Settings). A key you enter is held in memory for this connection only.")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        SecureField("API key, if that service requires one", text: $apiKey).textFieldStyle(.roundedBorder)
                        Button("Connect") {
                            let key = apiKey; apiKey = ""
                            Task { await runtime.attach(port: model.settings.port, apiKey: key) }
                        }.disabled(runtime.busy || models.removing)
                    }
                }.padding(.top, 6)
            }
        }
    }

    private var metricsCard: some View {
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
    }

    private var outputCard: some View {
        let filtered = LogFilter.apply(runtime.logs, query: logQuery, problemsOnly: problemsOnly, lastLines: logLines == 0 ? nil : logLines)
        return Card("Runtime output", subtitle: "Bounded local diagnostic output; filters change this view only", trailing: {
            Button("Copy All", systemImage: "doc.on.doc") { copyText(runtime.logs) }.controlSize(.small)
        }) {
            DisclosureGroup(outputExpanded ? "Hide output" : "Show output", isExpanded: $outputExpanded) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        TextField("Filter lines", text: $logQuery).textFieldStyle(.roundedBorder).frame(maxWidth: 240)
                        Toggle("Warnings and errors", isOn: $problemsOnly).toggleStyle(.checkbox)
                        Picker("Show", selection: $logLines) {
                            Text("Last 100").tag(100); Text("Last 500").tag(500); Text("All").tag(0)
                        }.pickerStyle(.menu).fixedSize()
                        Spacer()
                        Text("\(min(filtered.matched, logLines == 0 ? filtered.matched : logLines)) of \(filtered.total) lines")
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    ScrollViewReader { proxy in
                        ScrollView([.horizontal, .vertical]) {
                            Text(filtered.text.isEmpty ? "No lines match." : filtered.text)
                                .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                                .foregroundStyle(filtered.text.isEmpty ? .secondary : .primary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Color.clear.frame(height: 1).id("logEnd")
                        }
                        .frame(height: 220)
                        // Follow new output only in the unfiltered view, so reading matches is not interrupted.
                        .onChange(of: runtime.logs) { _, _ in if logQuery.isEmpty && !problemsOnly { proxy.scrollTo("logEnd", anchor: .bottom) } }
                        .onAppear { proxy.scrollTo("logEnd", anchor: .bottom) }
                    }
                }.padding(.top, 6)
            }
        }
    }

    private var keyAwaitsRestart: Bool { runtime.owned && runtime.requiresAPIKey != model.settings.requireAPIKey }

    private var apiKeySetting: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Toggle("Require an API key", isOn: Binding(get: { model.settings.requireAPIKey },
                                                             set: { model.selectRequireAPIKey($0) }))
                    .toggleStyle(.switch).controlSize(.small)
                    .disabled(attached)
                if keyAwaitsRestart {
                    PillBadge(text: "Restart required", systemImage: "arrow.clockwise", tint: .orange)
                    Button("Restart Now") { model.requestStartOrRestart() }
                    .controlSize(.small).disabled(!model.canRestartServer)
                    .help("Restart the server so the API key setting applies. Asks first if a reply is being generated.")
                }
            }
            Text(apiKeyGuidance).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var apiKeyGuidance: String {
        if attached { return "An attached service keeps its own authentication. Enter its key under Connect." }
        if keyAwaitsRestart {
            return model.settings.requireAPIKey
                ? "Saved for the next launch. The running server stays open to local clients until it restarts."
                : "Saved for the next launch. The running server still requires its key until it restarts."
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
