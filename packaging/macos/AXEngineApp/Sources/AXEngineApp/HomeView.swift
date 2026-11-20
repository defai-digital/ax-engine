import SwiftUI
import Charts
import AXEngineAppCore

/// Home (`.internal/ui/02-home-dark.svg`): what is serving, how fast, and how
/// to connect. Every number comes from the runtime's own counters, the host
/// monitor or a completed Playground reply; a dash means "not reported".
struct HomeView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var chat: ChatStore
    let clipboard: SensitiveClipboard
    @Binding var copyNotice: String

    var body: some View {
        GeometryReader { geometry in
            let wide = geometry.size.width >= 760
            let threeUp = geometry.size.width >= 940
            PageScroll {
                if let error = runtime.error { errorBanner(error) }
                TimelineView(.periodic(from: .now, by: 2)) { context in
                    VStack(spacing: 16) {
                        HeroPanel(model: model, models: models, runtime: runtime, chat: chat, clipboard: clipboard,
                                  copyNotice: $copyNotice, now: context.date, wide: wide)
                        let row = Group {
                            MemoryPanel(runtime: runtime, monitor: model.hostMonitor, now: context.date)
                            ThroughputPanel(runtime: runtime, now: context.date)
                            RequestsPanel(runtime: runtime)
                        }
                        if threeUp {
                            HStack(alignment: .top, spacing: 16) { row }.fixedSize(horizontal: false, vertical: true)
                        } else if wide {
                            HStack(alignment: .top, spacing: 16) {
                                MemoryPanel(runtime: runtime, monitor: model.hostMonitor, now: context.date)
                                ThroughputPanel(runtime: runtime, now: context.date)
                            }.fixedSize(horizontal: false, vertical: true)
                            RequestsPanel(runtime: runtime)
                        } else {
                            VStack(spacing: 16) { row }
                        }
                    }
                }
                ConnectAppPanel(runtime: runtime, port: model.settings.port, keyWhenStopped: model.settings.requireAPIKey)
            }
        }
    }

    private func errorBanner(_ error: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            ErrorBanner(text: error)
            Button("Details") { model.page = .api }.buttonStyle(.axLink)
        }
    }
}

// MARK: - Hero

/// Now serving (left) and live token speed (right), in one panel.
private struct HeroPanel: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var chat: ChatStore
    let clipboard: SensitiveClipboard
    @Binding var copyNotice: String
    let now: Date
    let wide: Bool
    @State private var pickerOpen = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var kind: ServerStatusKind { .from(state: runtime.state, owned: runtime.owned, ready: runtime.ready) }
    private var shown: LocalModel? {
        // An attached service runs its own model; never title it with the local selection.
        if !runtime.owned && !runtime.busy && runtime.connection != nil { return nil }
        let path = runtime.owned || runtime.busy ? runtime.selectedModel : model.selectedPath
        return models.productLibrary.first { $0.path == path }
    }
    private var selected: LocalModel? { models.productLibrary.first { $0.path == model.selectedPath } }
    private var switching: Bool { runtime.owned && !model.selectedPath.isEmpty && model.selectedPath != runtime.selectedModel }
    private var needsFirstModel: Bool {
        model.servableModels.isEmpty && !models.scanning && !runtime.owned && runtime.connection == nil
    }

    var body: some View {
        let layout = wide ? AnyLayout(HStackLayout(alignment: .top, spacing: 28)) : AnyLayout(VStackLayout(alignment: .leading, spacing: 22))
        layout {
            serving.frame(maxWidth: .infinity, alignment: .leading)
            if wide { Rectangle().fill(Brand.hairlineColor).frame(width: 1).padding(.vertical, 4) }
            speed.frame(maxWidth: wide ? 520 : .infinity, alignment: .leading)
        }
        .padding(24)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous)
                .fill(LinearGradient(colors: [Brand.panel, Brand.accent.opacity(kind == .serving ? 0.07 : 0.03)],
                                     startPoint: .leading, endPoint: .trailing))
                .overlay(RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous).strokeBorder(Brand.hairlineColor))
        }
    }

    // Left: state, model, specs, endpoint and the actions that apply.
    @ViewBuilder private var serving: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                StatusPill(text: statusWord, tint: Brand.tint(kind),
                           systemImage: kind == .failed || kind == .degraded ? "exclamationmark.triangle.fill" : nil)
                if let line = statusLine { Text(line).font(.system(size: 12.5)).foregroundStyle(Brand.text2) }
                if kind == .transition { ProgressView().controlSize(.small) }
            }
            if needsFirstModel {
                Text("No model downloaded yet").font(Brand.Font.display).foregroundStyle(Brand.text1)
                Text("Download a public AutomatosX model once; then Start serves it on this Mac. No account or token needed.")
                    .font(Brand.Font.body).foregroundStyle(Brand.text2).fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 10) {
                    Button("Download a Model", systemImage: "arrow.down") {
                        model.catalogSearch = ""; model.modelsSection = "catalog"; model.page = .models
                    }.buttonStyle(.axPrimary)
                    if models.activeDownloadCount > 0 {
                        Button("View Downloads (\(models.activeDownloadCount))") { model.modelsSection = "transfers"; model.page = .models }
                            .buttonStyle(.axSecondary)
                    }
                }
            } else {
                Text(title).font(Brand.Font.display).foregroundStyle(Brand.text1).lineLimit(1).minimumScaleFactor(0.6)
                    .textSelection(.enabled)
                tags
                HStack(spacing: 12) {
                    CodeField(text: endpoint).frame(minWidth: 250, maxWidth: 400).layoutPriority(1)
                    Text("OpenAI · Anthropic compatible").font(.system(size: 12)).foregroundStyle(Brand.text3)
                        .lineLimit(1)
                }
                .opacity(runtime.connection == nil ? 0.55 : 1)
                .help(runtime.connection == nil ? "Clients connect here once the server is running." : "API endpoint")
                actions
                if runtime.connection?.hasAPIKey == true {
                    ClientCopyButtons(runtime: runtime, clipboard: clipboard, notice: $copyNotice).controlSize(.small)
                }
                if let notice { Label(notice, systemImage: "info.circle").font(Brand.Font.caption).foregroundStyle(Brand.text3) }
                if runtime.state == "Starting", let line = RuntimeController.lastDiagnostic(runtime.logs) {
                    Text(line).font(.system(size: 11, design: .monospaced)).foregroundStyle(Brand.text3)
                        .lineLimit(1).truncationMode(.middle)
                }
            }
        }
    }

    private var statusWord: String {
        switch kind {
        case .serving: "Serving"
        case .external: "Attached"
        case .transition: runtime.state
        case .degraded: "Needs attention"
        case .failed: "Stopped with error"
        case .stopped: "Not serving"
        }
    }

    private var statusLine: String? {
        if runtime.connection != nil {
            let uptime = Format.uptime(since: runtime.startedAt, now: now).map { "Up " + $0 }
            let requests = runtime.metrics["ax_engine_generation_completed_requests_total"]
                .map { Format.count($0)! + ($0 == 1 ? " request" : " requests") }
            return [uptime, requests].compactMap { $0 }.joined(separator: " · ")
        }
        if runtime.state == "Starting" { return Format.uptime(since: runtime.startedAt, now: now).map { "Loading · " + $0 } }
        if model.servableModels.isEmpty { return nil }
        return "\(model.servableModels.count) downloaded model\(model.servableModels.count == 1 ? "" : "s")"
    }

    private var title: String {
        if let shown { return shown.label.title }
        if runtime.connection != nil { return runtime.models.first?.id ?? "External service" }
        return model.servableModels.isEmpty ? "Checking your models…" : "Choose a model"
    }

    private var tags: some View {
        HStack(spacing: 6) {
            if let shown {
                ForEach(shown.label.tags, id: \.self) { tag in Tag(tag, tint: tag == "MTP" ? Brand.boost : nil) }
                if let size = shown.sizeBytes { Tag(TransferFormat.bytes(size)) }
            }
            if let context = runtime.models.first?.contextLength { Tag(context >= 1024 ? "\(context / 1024)k context" : "\(context) context") }
            if runtime.owned { Tag(runtime.servingMode.badge) }
        }
    }

    private var endpoint: String {
        runtime.connection?.apiURL.absoluteString ?? "http://127.0.0.1:\(model.settings.port)/v1"
    }

    @ViewBuilder private var actions: some View {
        HStack(spacing: 10) {
            if runtime.owned {
                if runtime.ready {
                    Button { model.page = .chat } label: {
                        HStack(spacing: 7) { Text("Open Playground"); Text("⌘2").font(.system(size: 11)).opacity(0.7) }
                    }.buttonStyle(.axPrimary)
                }
                switchButton
                let modeChanged = model.settings.servingMode != runtime.servingMode
                let capChanged = model.settings.concurrentRequests != runtime.concurrentRequests
                let keyChanged = model.settings.requireAPIKey != runtime.requiresAPIKey
                let changed = switching || modeChanged || capChanged || keyChanged
                Button(changed ? ServerGate.restartTitle(modelChanged: switching, modeChanged: modeChanged, capChanged: capChanged,
                                                         keyChanged: keyChanged, mode: model.settings.servingMode) : "Restart",
                       systemImage: "arrow.clockwise") { model.requestStartOrRestart() }
                    .buttonStyle(changed ? .axPrimary : .axSecondary)
                    .disabled(!model.canRestartServer)
                    .help(changed ? "Stop this server and start it again with the saved changes. Active inference stops. (⇧⌘R)"
                                  : "Stop this server and start it again. Active inference stops. (⇧⌘R)")
                Button(runtime.state == "Starting" ? "Cancel Start" : "Stop", systemImage: "stop.fill") { model.requestStopServer() }
                    .buttonStyle(.axDanger).disabled(runtime.state == "Stopping")
                    .help("Stop the server and free its memory. (⇧⌘.)")
            } else if runtime.connection != nil {
                if runtime.ready { Button("Open Playground") { model.page = .chat }.buttonStyle(.axPrimary) }
                Button("Disconnect") { model.requestStopServer() }.buttonStyle(.axSecondary)
                    .help("Stop using the attached service. It keeps running.")
            } else if model.connecting {
                Button("Cancel Connecting") { runtime.detach() }.buttonStyle(.axSecondary)
            } else {
                Button("Start Server", systemImage: "play.fill") { model.startServer() }
                    .buttonStyle(.axServe).disabled(model.startBlockReason != nil)
                    .help(model.startBlockReason ?? "Start a loopback server on the selected model (⇧⌘R)")
                switchButton
            }
        }
    }

    private var switchButton: some View {
        Button { pickerOpen.toggle() } label: {
            Label(runtime.owned ? "Switch Model" : "Choose Model", systemImage: "chevron.down")
        }
        .buttonStyle(.axSecondary)
        .disabled(model.attachedToService || model.servableModels.isEmpty)
        .popover(isPresented: $pickerOpen, arrowEdge: .bottom) {
            ModelPickerPopover(models: model.servableModels, selectedPath: model.selectedPath,
                               favorites: model.favoriteModels, recent: model.recentModels, scanning: models.scanning,
                               choose: { model.selectedPath = $0; pickerOpen = false },
                               toggleFavorite: { model.toggleFavoriteModel($0) },
                               browseCatalog: { pickerOpen = false; model.catalogSearch = ""; model.modelsSection = "catalog"; model.page = .models })
        }
    }

    private var notice: String? {
        if runtime.owned {
            return ServerGate.pendingRestartNotice(
                modelChanged: switching, modeChanged: model.settings.servingMode != runtime.servingMode,
                running: runtime.servingMode, selected: model.settings.servingMode,
                runningCap: runtime.concurrentRequests, selectedCap: model.settings.concurrentRequests,
                runningRequiresKey: runtime.requiresAPIKey, selectedRequiresKey: model.settings.requireAPIKey)
                .map { switching ? "Switch to \(selected?.displayName ?? "the selected model"): " + $0 : $0 }
        }
        if runtime.connection == nil, !model.connecting { return model.startBlockReason }
        return nil
    }

    // Right: live token speed with a five-minute sparkline.
    private var speed: some View {
        let rate = runtime.connection == nil ? nil : runtime.history.latest(RuntimeController.tokenRateKey, now: now)
        let acceptance = runtime.history.latest("mtpAcceptedPercent", now: now)
        let active = runtime.metrics["ax_engine_generation_active_streams"]
        let last = chat.messages.last { $0.role == "assistant" && $0.state == "Complete" }?.performance
        return VStack(alignment: .leading, spacing: 12) {
            SectionLabel("Token speed · live" + (acceptance != nil ? " · MTP on" : ""))
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(Format.rate(rate) ?? "–").font(rate == nil ? .system(size: 56, weight: .thin) : Brand.Font.hero)
                    .foregroundStyle(rate == nil ? Brand.text3 : Brand.text1)
                    .contentTransition(reduceMotion ? .identity : .numericText())
                Text("tok/s").font(.system(size: 18, weight: .medium)).foregroundStyle(Brand.text2)
                Spacer(minLength: 0)
            }
            .help("Tokens the runtime scheduled per second between polls, prompt and output together.")
            Sparkline(points: runtime.history.plotPoints(RuntimeController.tokenRateKey, window: 300, now: now),
                      start: now.addingTimeInterval(-300), end: now)
                .frame(height: 50)
                .accessibilityLabel("Token speed, last five minutes")
            HStack(alignment: .top, spacing: 18) {
                StatReading(title: "First text", value: chat.firstTokenSeconds.map { String(format: "%.0f", $0 * 1000) }, unit: "ms",
                            help: "Last Playground reply: time from Send to the first visible text.")
                StatReading(title: "Prefill", value: last.flatMap { $0.promptFullyCached ? "Cached" : Format.rate($0.prefillTokensPerSecond) },
                            unit: last?.promptFullyCached == true ? nil : "tok/s",
                            help: "Last Playground reply, native prefill average.")
                StatReading(title: "Draft accepted", value: acceptance.map { String(format: "%.0f", $0) }, unit: "%",
                            tint: Brand.boost,
                            help: "MTP drafts the model accepted in the last poll interval. Accepted drafts are output it did not decode one by one; not a speedup measurement.")
                StatReading(title: "In flight", value: active.map { String(format: "%.0f", $0) },
                            unit: runtime.owned ? "of \(runtime.concurrentRequests)" : nil,
                            help: "Requests generating now.")
            }
            if runtime.connection == nil {
                Text("Speed appears while the server runs requests.").font(Brand.Font.caption).foregroundStyle(Brand.text3)
            }
        }
    }
}

// MARK: - Row panels

/// Unified memory: what AX Engine uses, from runtime counters, against the
/// host's used pages when live activity sampling is on.
private struct MemoryPanel: View {
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var monitor: HostMonitor
    let now: Date

    var body: some View {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        let report = ModelMemoryReport(metrics: runtime.metrics)
        let used = monitor.history.latest("memory", now: now, maxAge: 30).map { $0 / 100 * total }
        let engine = report.total
        let other = used.flatMap { used in engine.map { max(0, used - $0) } ?? used }
        Card("Unified memory", trailing: {
            Text(used.map { (Format.bytes(total - $0) ?? "—") + " free of " + HostInfo.memory } ?? HostInfo.memory + " total")
                .font(Brand.Font.caption).foregroundStyle(Brand.text3)
        }) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(Format.bytes(engine) ?? "—").font(.system(size: 24, weight: .bold)).monospacedDigit()
                    .foregroundStyle(engine == nil ? Brand.text3 : Brand.text1)
                Text("used by AX Engine").font(.system(size: 22, weight: .semibold)).foregroundStyle(Brand.text3)
                    .lineLimit(1).minimumScaleFactor(0.6)
            }
            StackedBar(segments: [
                .init(id: "active", value: report.active ?? 0, color: Brand.accent),
                .init(id: "cache", value: report.cache ?? 0, color: Brand.boost),
                .init(id: "other", value: other ?? 0, color: Brand.text3.opacity(0.7)),
            ], total: total)
            VStack(spacing: 7) {
                legend("MLX active · weights and KV", report.active, Brand.accent)
                legend("MLX cache", report.cache, Brand.boost)
                legend("Rest of system", other, Brand.text3.opacity(0.7))
                if let prefix = report.prefixCache {
                    HStack {
                        Text("Prefix cache payload").font(.system(size: 11.5)).foregroundStyle(Brand.text3)
                        Spacer()
                        Text(Format.bytes(prefix) ?? "—").font(.system(size: 11.5)).monospacedDigit().foregroundStyle(Brand.text3)
                    }.help("Reported separately by the model runner; not a slice of the bar.")
                }
            }
            Text(used == nil ? "From runtime counters. Turn on live activity in System for the Mac's total."
                             : "From runtime counters and macOS page counts.")
                .font(.system(size: 11)).foregroundStyle(Brand.text3)
        }
    }

    private func legend(_ title: String, _ value: Double?, _ color: Color) -> some View {
        HStack(spacing: 8) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 9, height: 9)
            Text(title).font(.system(size: 12.5)).foregroundStyle(Brand.text2)
            Spacer()
            Text(Format.bytes(value) ?? "—").font(.system(size: 12.5, weight: .semibold)).monospacedDigit()
                .foregroundStyle(value == nil ? Brand.text3 : Brand.text1)
        }
    }
}

/// Tokens the runtime scheduled per ten seconds over the last five minutes.
private struct ThroughputPanel: View {
    @ObservedObject var runtime: RuntimeController
    let now: Date

    private struct Bucket: Identifiable { let id: Int; let start: Date; let tokens: Double }

    var body: some View {
        let buckets = Self.buckets(runtime.history.points(RuntimeController.tokenRateKey, now: now), now: now)
        Card("Throughput", trailing: { Tag("5 min") }) {
            Chart(buckets) { bucket in
                BarMark(x: .value("Time", bucket.start, unit: .second), y: .value("Tokens", bucket.tokens), width: .ratio(0.62))
                    .foregroundStyle(Brand.accent).cornerRadius(2)
            }
            .chartXScale(domain: now.addingTimeInterval(-300)...now)
            .chartXAxis(.hidden)
            .chartYAxis {
                AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) { value in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 1, dash: [2, 4])).foregroundStyle(Brand.hairlineColor)
                    AxisValueLabel {
                        if let number = value.as(Double.self) {
                            Text(number >= 1000 ? String(format: "%.0fk", number / 1000) : String(format: "%.0f", number))
                                .font(.system(size: 10)).foregroundStyle(Brand.text3)
                        }
                    }
                }
            }
            .frame(height: 128)
            .overlay {
                if buckets.allSatisfy({ $0.tokens == 0 }) {
                    Text(runtime.connection == nil ? "No server running" : "Idle · no tokens scheduled")
                        .font(Brand.Font.caption).foregroundStyle(Brand.text3)
                }
            }
            .accessibilityLabel("Scheduled tokens per ten seconds, last five minutes")
            HStack { Text("5 min ago"); Spacer(); Text("now") }.font(.system(size: 10.5)).foregroundStyle(Brand.text3)
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 2).fill(Brand.accent).frame(width: 9, height: 9)
                Text("Tokens scheduled per 10 s · prompt and output").font(.system(size: 11)).foregroundStyle(Brand.text2)
            }
        }
    }

    /// Integrate the per-poll rate into fixed ten-second buckets.
    private static func buckets(_ points: [MetricPlotPoint], now: Date) -> [Bucket] {
        let width: TimeInterval = 10, count = 30
        let start = now.addingTimeInterval(-width * Double(count))
        var totals = [Double](repeating: 0, count: count)
        var previous: MetricPlotPoint?
        for point in points where point.time >= start {
            defer { previous = point }
            guard let previous, previous.segment == point.segment else { continue }
            let index = min(count - 1, max(0, Int(point.time.timeIntervalSince(start) / width)))
            totals[index] += point.value * point.time.timeIntervalSince(previous.time)
        }
        return totals.enumerated().map { Bucket(id: $0.offset, start: start.addingTimeInterval(width * Double($0.offset) + width / 2), tokens: $0.element) }
    }
}

/// Request counters since the server started. The runtime does not report
/// which app sent a request, so this lists outcomes, not clients.
private struct RequestsPanel: View {
    @ObservedObject var runtime: RuntimeController
    var body: some View {
        let metric = { (key: String) in runtime.metrics[key] }
        Card("Requests", trailing: {
            Text("Since the server started").font(Brand.Font.caption).foregroundStyle(Brand.text3)
        }) {
            VStack(spacing: 12) {
                row("bolt.horizontal", "Generating now", "Streams in progress", metric("ax_engine_generation_active_streams"), tint: Brand.serve)
                row("checkmark.circle", "Completed", "Finished generations", metric("ax_engine_generation_completed_requests_total"))
                row("arrow.left.arrow.right", "HTTP requests", httpDetail, metric("ax_engine_http_requests_total"))
                row("exclamationmark.triangle", "Errors", "4xx client · 5xx server",
                    sum(metric("ax_engine_http_status_4xx_total"), metric("ax_engine_http_status_5xx_total")),
                    tint: (sum(metric("ax_engine_http_status_4xx_total"), metric("ax_engine_http_status_5xx_total")) ?? 0) > 0 ? Brand.warn : Brand.text1)
            }
            Spacer(minLength: 0)
            Text("Local only · per-app history is not reported by the runtime").font(.system(size: 11)).foregroundStyle(Brand.text3)
        }
    }
    private var httpDetail: String {
        Format.count(runtime.metrics["ax_engine_http_status_2xx_total"]).map { $0 + " succeeded" } ?? "OpenAI and Anthropic routes"
    }
    private func sum(_ a: Double?, _ b: Double?) -> Double? {
        a == nil && b == nil ? nil : (a ?? 0) + (b ?? 0)
    }
    private func row(_ icon: String, _ title: String, _ detail: String, _ value: Double?, tint: Color = Brand.text1) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon).font(.system(size: 13)).foregroundStyle(Brand.text2)
                .frame(width: 32, height: 32)
                .background(Brand.inset, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Brand.hairlineColor))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(size: 13, weight: .semibold)).foregroundStyle(Brand.text1)
                Text(detail).font(.system(size: 11.5)).foregroundStyle(Brand.text3).lineLimit(1)
            }
            Spacer()
            Text(Format.count(value) ?? "—").font(.system(size: 15, weight: .semibold)).monospacedDigit()
                .foregroundStyle(value == nil ? Brand.text3 : tint)
        }
        .accessibilityElement(children: .combine)
    }
}
