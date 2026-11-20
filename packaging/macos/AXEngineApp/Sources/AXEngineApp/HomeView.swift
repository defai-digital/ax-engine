import SwiftUI
import Charts
import AXEngineAppCore

/// Home answers three questions at a glance: what is serving, how fast,
/// and how to connect. Every number comes from the runtime's own counters
/// or a completed Playground reply; a dash means "not reported", never zero.
struct HomeView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var chat: ChatStore
    let clipboard: SensitiveClipboard
    @Binding var copyNotice: String

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if runtime.connection != nil {
                        TimelineView(.periodic(from: .now, by: 2)) { context in
                            SpeedHero(runtime: runtime, now: context.date)
                        }
                    }
                    adaptive(wide: geometry.size.width >= 820) {
                        ServeCard(model: model, runtime: runtime, models: models, chat: chat,
                                  clipboard: clipboard, copyNotice: $copyNotice)
                    } trailing: {
                        VStack(spacing: 16) {
                            if runtime.connection != nil { MemoryCard(report: ModelMemoryReport(metrics: runtime.metrics)) }
                            LastReplyCard(model: model, chat: chat)
                        }
                    }
                }
                .frame(maxWidth: 1240).padding(24).frame(maxWidth: .infinity)
            }
        }
    }

    @ViewBuilder private func adaptive<A: View, B: View>(wide: Bool, @ViewBuilder leading: () -> A,
                                                          @ViewBuilder trailing: () -> B) -> some View {
        if wide {
            HStack(alignment: .top, spacing: 16) {
                leading().frame(maxWidth: .infinity)
                trailing().frame(width: 340)
            }
        } else {
            VStack(spacing: 16) { leading(); trailing() }
        }
    }
}

/// Live speed: the runtime's scheduled-token rate between polls, request
/// counters and MTP draft acceptance, with a five-minute history.
private struct SpeedHero: View {
    @ObservedObject var runtime: RuntimeController
    let now: Date
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let rate = runtime.history.latest(RuntimeController.tokenRateKey, now: now)
        let acceptance = runtime.history.latest("mtpAcceptedPercent", now: now)
        let active = runtime.metrics["ax_engine_generation_active_streams"]
        let completed = runtime.metrics["ax_engine_generation_completed_requests_total"]
        let kind = ServerStatusKind.from(state: runtime.state, owned: runtime.owned, ready: runtime.ready)
        Card {
            HStack(alignment: .top, spacing: 28) {
                VStack(alignment: .leading, spacing: 6) {
                    Label(kind == .external ? "Attached service" : "Live speed", systemImage: "bolt.fill")
                        .font(.caption.weight(.semibold)).foregroundStyle(Brand.tint(kind))
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(rate.map { String(format: $0 >= 100 ? "%.0f" : "%.1f", $0) } ?? "—")
                            .font(.system(size: 52, weight: .semibold, design: .rounded)).monospacedDigit()
                            .contentTransition(reduceMotion ? .identity : .numericText())
                        Text("tok/s").font(.title3).foregroundStyle(.secondary)
                    }
                    Text(rate == nil ? "Idle. Speed appears while requests run." : "Tokens scheduled per second · prompt and output")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(minWidth: 220, alignment: .leading)
                Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 10) {
                    GridRow {
                        stat("Active requests", active.map { String(format: "%.0f", $0) })
                        stat("Completed", completed.map { $0.formatted(.number.precision(.fractionLength(0))) })
                    }
                    GridRow {
                        stat("MTP drafts accepted", acceptance.map { String(format: "%.0f%%", $0) }, tint: Brand.boost,
                             help: "Share of speculative draft tokens the model accepted in the last poll interval. Accepted drafts are output the model did not have to decode one by one. This is not a speedup measurement.")
                        stat("Context", runtime.models.first?.contextLength.map { $0.formatted() + " tokens" })
                    }
                }
                Spacer(minLength: 0)
            }
            Chart {
                ForEach(runtime.history.plotPoints(RuntimeController.tokenRateKey, window: 300, now: now)) { point in
                    AreaMark(x: .value("Time", point.time), y: .value("tok/s", point.value),
                             series: .value("Segment", point.segment))
                        .foregroundStyle(Brand.accent.opacity(0.18).gradient)
                    LineMark(x: .value("Time", point.time), y: .value("tok/s", point.value),
                             series: .value("Segment", point.segment))
                        .foregroundStyle(Brand.accent).lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))
                }
            }
            .chartXScale(domain: now.addingTimeInterval(-300)...now)
            .chartXAxis(.hidden)
            .chartYAxis { AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) {
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [3, 4])).foregroundStyle(.secondary.opacity(0.2))
                AxisValueLabel().font(.system(size: 10))
            } }
            .frame(height: 90)
            .accessibilityLabel("Tokens per second, last five minutes. Gaps mean no measurement.")
            HStack {
                Text("5 min ago"); Spacer(); Text("Now")
            }.font(.system(size: 10)).foregroundStyle(.secondary)
        }
    }

    private func stat(_ title: String, _ value: String?, tint: Color = .primary, help: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value ?? "—").font(.system(.title3, design: .rounded).weight(.semibold)).monospacedDigit()
                .foregroundStyle(value == nil ? .secondary : tint)
        }
        .help(help ?? title)
        .accessibilityElement(children: .combine)
    }
}

/// Pick a downloaded model, then Start, Restart or Stop. Serving options
/// fold away; API key and attach live on API & Logs.
private struct ServeCard: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var chat: ChatStore
    let clipboard: SensitiveClipboard
    @Binding var copyNotice: String

    var body: some View {
        Card(runtime.connection == nil ? "Serve a model" : "Now serving",
             subtitle: "Apps on this Mac connect to one OpenAI-compatible endpoint.", trailing: {
            Button("API & Logs", systemImage: "chevron.left.forwardslash.chevron.right") { model.page = .api }.controlSize(.small)
                .help("Connection snippets, API key, attach to an existing service, metrics and runtime output")
        }) {
            ServerStatusRow(model: model, runtime: runtime)
            if let error = runtime.error {
                HStack(alignment: .firstTextBaseline) {
                    Label(error, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(Brand.warn)
                        .lineLimit(3).textSelection(.enabled)
                    Spacer()
                    Button("Details") { model.page = .api }.buttonStyle(.link).font(.caption)
                }
            }
            ServerControls(model: model, runtime: runtime, models: models, chat: chat, showsPath: false)
            ClientCopyButtons(runtime: runtime, clipboard: clipboard, notice: $copyNotice,
                              showsPlayground: true, openPlayground: { model.page = .chat })
            if !model.servableModels.isEmpty || runtime.owned {
                Divider()
                ServingOptionsDisclosure(model: model, runtime: runtime)
            }
        }
    }
}

/// Memory the runtime reports for the loaded model. Parts it does not
/// export are left out rather than estimated.
private struct MemoryCard: View {
    let report: ModelMemoryReport
    var body: some View {
        Card("Model memory", subtitle: "Reported by the runtime") {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(report.total.map(bytes) ?? "—")
                    .font(.system(size: 28, weight: .semibold, design: .rounded)).monospacedDigit()
                Text("MLX active + cache").font(.caption).foregroundStyle(.secondary)
            }
            if let total = report.total, total > 0 {
                GeometryReader { geometry in
                    HStack(spacing: 2) {
                        segment(report.active, of: total, width: geometry.size.width, color: Brand.accent)
                        segment(report.cache, of: total, width: geometry.size.width, color: Brand.accent.opacity(0.4))
                    }
                }
                .frame(height: 8).clipShape(Capsule())
                .accessibilityHidden(true)
            }
            VStack(spacing: 6) {
                MetricRow(label: "Active", value: report.active.map(bytes) ?? "—", valueTint: .primary)
                MetricRow(label: "Cache", value: report.cache.map(bytes) ?? "—", valueTint: .primary)
                MetricRow(label: "Prefix cache", value: report.prefixCache.map(bytes) ?? "—", valueTint: .primary)
                MetricRow(label: "Peak", value: report.peak.map(bytes) ?? "—", valueTint: .secondary)
            }
        }
    }
    private func segment(_ value: Double?, of total: Double, width: CGFloat, color: Color) -> some View {
        Rectangle().fill(color).frame(width: max(0, width * CGFloat((value ?? 0) / total)))
    }
    private func bytes(_ value: Double) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(min(value, Double(Int64.max))), countStyle: .memory)
    }
}

/// The most recent completed Playground reply's native speed.
private struct LastReplyCard: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var chat: ChatStore
    var body: some View {
        let last = chat.messages.last { $0.role == "assistant" && $0.state == "Complete" }
        let performance = last?.performance
        Card("Last Playground reply", subtitle: last == nil ? "No replies yet" : "Measured by the runtime, excludes display time") {
            if let performance {
                HStack(spacing: 18) {
                    reading("Decode", TokenRateFormat.short(performance.decodeTokensPerSecond))
                    reading("Prefill", performance.promptFullyCached ? "Cached" : TokenRateFormat.short(performance.prefillTokensPerSecond))
                    if let tokens = performance.decodeTokens { reading("Output", "\(tokens) tok") }
                }
            } else {
                Text(last == nil ? "Send a prompt in Playground to measure this model on your Mac."
                                 : "The runtime reported no timing for the last reply.")
                    .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Button("Open Playground", systemImage: "bubble.left.and.bubble.right") { model.page = .chat }
                .controlSize(.small)
        }
    }
    private func reading(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.callout, design: .rounded).weight(.semibold)).monospacedDigit()
        }.accessibilityElement(children: .combine)
    }
}
