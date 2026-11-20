import SwiftUI
import AppKit
import AXEngineAppCore

/// API & Logs: how apps connect (endpoint, ready-to-paste snippets, API
/// key), the app-owned server in detail, attaching to an existing service,
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
    @State private var followLogs = true
    @State private var diagnosticsNotice = ""
    @State private var logQuery = ""
    @State private var problemsOnly = false
    @State private var logLines = 500

    private var attached: Bool { runtime.connection != nil && !runtime.owned }

    var body: some View {
        GeometryReader { geometry in
            PageScroll {
                if let error = runtime.error { errorCard(error) }
                if geometry.size.width >= 900 {
                    HStack(alignment: .top, spacing: 16) {
                        endpointPanel.frame(maxWidth: .infinity)
                        ConnectAppPanel(runtime: runtime, port: model.settings.port, keyWhenStopped: model.settings.requireAPIKey).frame(maxWidth: .infinity)
                    }
                } else {
                    endpointPanel
                    ConnectAppPanel(runtime: runtime, port: model.settings.port, keyWhenStopped: model.settings.requireAPIKey)
                }
                logsPanel
                if runtime.connection == nil && !runtime.owned { connectCard }
                if runtime.connection != nil {
                    metricsCard
                    RuntimeHistoryView(runtime: runtime)
                }
            }
        }
        .onChange(of: runtime.error) { _, _ in diagnosticsNotice = "" }
    }

    /// Endpoint panel from `08-api-and-logs.svg`: URLs, key, serving profile.
    private var endpointPanel: some View {
        let url = runtime.connection?.apiURL.absoluteString ?? "http://127.0.0.1:\(model.settings.port)/v1"
        return Card("Endpoint", trailing: {
            StatusPill(text: "Local only", tint: Brand.serve, systemImage: "lock.fill")
        }) {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                GridRow {
                    fieldLabel("OpenAI"); CodeField(text: url)
                }
                GridRow {
                    fieldLabel("Anthropic"); CodeField(text: ClientSnippet.anthropicBase(url))
                }
                GridRow {
                    fieldLabel("Model ID"); CodeField(text: runtime.models.first?.id ?? "", placeholder: "Appears when a model is serving")
                }
            }
            .opacity(runtime.connection == nil ? 0.6 : 1)
            if runtime.connection?.hasAPIKey == true {
                ClientCopyButtons(runtime: runtime, clipboard: clientClipboard, notice: $clientCopyNotice)
            }
            Divider().overlay(Brand.hairlineColor)
            apiKeySetting
            Divider().overlay(Brand.hairlineColor)
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Serving profile").font(.system(size: 13, weight: .semibold))
                    Spacer()
                    if runtime.owned && model.settings.servingMode != runtime.servingMode {
                        Text("Applies on restart").font(Brand.Font.caption).foregroundStyle(Brand.warn)
                    }
                }
                AXSegmented(options: ServingMode.allCases.map { ($0, $0.title) },
                            selection: Binding(get: { model.settings.servingMode }, set: { model.selectServingMode($0) }))
                    .disabled(attached)
                Text(attached ? "An attached service keeps the serving mode it was started with."
                              : model.settings.servingMode.summary)
                    .font(Brand.Font.caption).foregroundStyle(Brand.text3).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 14) {
                    Text("Parallel requests").font(.system(size: 13))
                    Stepper(value: Binding(get: { model.settings.concurrentRequests }, set: { model.selectConcurrentRequests($0) }),
                            in: AppSettings.concurrentRequestRange) {
                        Text("\(model.settings.concurrentRequests)").font(.system(size: 13, weight: .semibold).monospacedDigit())
                            .frame(minWidth: 18)
                    }
                    .disabled(attached).accessibilityLabel("Parallel requests")
                    Spacer()
                    Text("Port").font(.system(size: 13))
                    Button(String(model.settings.port)) { model.openSettings() }
                        .buttonStyle(.ax(.secondary)).monospacedDigit().help("Change the port in Settings")
                }
                if runtime.owned && model.settings.concurrentRequests != runtime.concurrentRequests {
                    Label("Applies on restart. The running server still accepts \(runtime.concurrentRequests) at once.", systemImage: "arrow.clockwise")
                        .font(Brand.Font.caption).foregroundStyle(Brand.warn)
                }
                Text("Startup is offline. Extra requests beyond the limit receive 429 until a slot frees.")
                    .font(Brand.Font.caption).foregroundStyle(Brand.text3)
            }
        }
    }

    private func fieldLabel(_ text: String) -> some View {
        Text(text).font(.system(size: 12.5)).foregroundStyle(Brand.text2).frame(width: 72, alignment: .leading)
    }

    /// Logs panel: level tabs, filter, follow, colourized lines.
    private var logsPanel: some View {
        let filtered = LogFilter.apply(runtime.logs, query: logQuery, problemsOnly: problemsOnly, lastLines: logLines == 0 ? nil : logLines)
        let problems = LogFilter.apply(runtime.logs, problemsOnly: true).matched
        return Card {
            HStack(spacing: 12) {
                Text("Logs").font(Brand.Font.cardTitle)
                AXSegmented(options: [(false, "All"), (true, problems > 0 ? "Problems \(problems)" : "Problems")],
                            selection: $problemsOnly, compact: true)
                Spacer()
                Button("Copy", systemImage: "doc.on.doc") { copyText(runtime.logs) }.buttonStyle(.axLink)
                    .disabled(runtime.logs.isEmpty)
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass").font(.system(size: 11)).foregroundStyle(Brand.text3)
                    TextField("Filter logs", text: $logQuery).textFieldStyle(.plain).font(.system(size: 12.5))
                }
                .padding(.horizontal, 9).padding(.vertical, 5).frame(maxWidth: 220)
                .background(Brand.inset, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).strokeBorder(Brand.hairlineColor))
                Toggle("Follow", isOn: $followLogs).toggleStyle(.switch).controlSize(.mini).font(.system(size: 12.5))
            }
            ScrollViewReader { proxy in
                ScrollView([.vertical, .horizontal]) {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(filtered.text.isEmpty ? AttributedString(runtime.logs.isEmpty ? "Runtime output appears here while the app-owned server runs." : "No lines match.")
                                                   : colourized(filtered.text))
                            .font(.system(size: 11.5, design: .monospaced)).lineSpacing(4).textSelection(.enabled)
                            .foregroundStyle(filtered.text.isEmpty ? Brand.text3 : Brand.text1)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Color.clear.frame(height: 1).id("logEnd")
                    }
                    .padding(14)
                }
                .frame(height: 300)
                .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
                .onChange(of: runtime.logs) { _, _ in if followLogs && logQuery.isEmpty && !problemsOnly { proxy.scrollTo("logEnd", anchor: .bottom) } }
                .onAppear { proxy.scrollTo("logEnd", anchor: .bottom) }
            }
            HStack {
                Text("\(min(filtered.matched, logLines == 0 ? filtered.matched : logLines)) of \(filtered.total) lines · local diagnostic output; filters change this view only")
                Spacer()
                Picker("Show", selection: $logLines) {
                    Text("Last 100").tag(100); Text("Last 500").tag(500); Text("All").tag(0)
                }.pickerStyle(.menu).labelsHidden().fixedSize()
            }
            .font(Brand.Font.caption).foregroundStyle(Brand.text3)
        }
    }

    private func colourized(_ text: String) -> AttributedString {
        var result = AttributedString()
        for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            if index > 0 { result += AttributedString("\n") }
            var piece = AttributedString(String(line))
            switch LogFilter.level(line) {
            case .error: piece.foregroundColor = Brand.danger
            case .warning: piece.foregroundColor = Brand.warn
            case .info: piece.foregroundColor = Brand.text1
            }
            result += piece
        }
        return result
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
                Button("Review Settings") { model.openSettings() }
            }
            if !diagnosticsNotice.isEmpty { Text(diagnosticsNotice).font(.caption).foregroundStyle(.secondary) }
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

    private var keyAwaitsRestart: Bool { runtime.owned && runtime.requiresAPIKey != model.settings.requireAPIKey }

    private var apiKeySetting: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Require API key").font(.system(size: 13, weight: .semibold))
                Spacer()
                if keyAwaitsRestart {
                    PillBadge(text: "Restart required", systemImage: "arrow.clockwise", tint: .orange)
                    Button("Restart Now") { model.requestStartOrRestart() }
                    .controlSize(.small).disabled(!model.canRestartServer)
                    .help("Restart the server so the API key setting applies. Asks first if a reply is being generated.")
                }
                Toggle("Require API key", isOn: Binding(get: { model.settings.requireAPIKey },
                                                         set: { model.selectRequireAPIKey($0) }))
                    .toggleStyle(.switch).controlSize(.small).labelsHidden()
                    .disabled(attached)
            }
            Text(apiKeyGuidance).font(Brand.Font.caption).foregroundStyle(Brand.text3).fixedSize(horizontal: false, vertical: true)
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
