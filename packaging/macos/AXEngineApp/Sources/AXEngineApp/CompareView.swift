import SwiftUI
import Charts
import AXEngineAppCore

/// Playground › Compare A/B (`.internal/ui/07-playground-compare.svg`): the
/// same prompt and settings on A, then on B. Speed only, and the result is
/// declared after both runs end.
struct CompareView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var compare: CompareStore
    @Binding var mode: String
    @AppStorage("compareModelB") private var bPath = ""

    private let colorA = Brand.accent
    private let colorB = Brand.boost

    var body: some View {
        let reason = model.compareBlockReason(bPath: bPath)
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                AXSegmented(options: [("single", "Single"), ("compare", "Compare A/B")], selection: $mode, compact: true)
                    .disabled(compare.running)
                Text("Runs A, then B, with the same prompt and settings. Compares speed, not answer quality.")
                    .font(.system(size: 12)).foregroundStyle(Brand.text3).lineLimit(2)
                Spacer()
            }
            promptBar(reason: reason)
            if let status = compare.status {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(status).font(.system(size: 12, weight: .medium)).foregroundStyle(Brand.text2)
                    Text("Each switch reloads a model; large packs can take minutes.").font(.system(size: 11.5)).foregroundStyle(Brand.text3)
                }
            }
            if let notice = compare.notice, !compare.running {
                Label(notice, systemImage: "exclamationmark.triangle").font(.system(size: 12)).foregroundStyle(Brand.warn)
            }
            HStack(alignment: .top, spacing: 14) {
                sidePanel(isA: true)
                sidePanel(isA: false)
            }
            .frame(maxHeight: .infinity)
            chartPanel.frame(minHeight: 250)
        }
        .padding(.horizontal, 24).padding(.top, 6).padding(.bottom, 20)
        .background(Brand.canvas)
        .onAppear(perform: pickDefaultB)
        .onChange(of: model.servableModels.map(\.path)) { _, _ in pickDefaultB() }
        .onChange(of: runtime.selectedModel) { _, _ in if !compare.running { pickDefaultB() } }
    }

    private func pickDefaultB() {
        let valid = model.servableModels.contains { $0.path == bPath }
        if !valid || bPath == runtime.selectedModel { bPath = defaultB }
    }

    private var defaultB: String {
        model.servableModels.first { $0.path != runtime.selectedModel }?.path ?? ""
    }

    private func promptBar(reason: String?) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "text.cursor").foregroundStyle(Brand.text3)
            TextField("Prompt for both runs", text: $compare.prompt, axis: .vertical)
                .textFieldStyle(.plain).font(.system(size: 13, weight: .medium)).lineLimit(1...4)
                .disabled(compare.running)
            Toggle("Return to A", isOn: $compare.returnToA).toggleStyle(.checkbox).font(.system(size: 12))
                .foregroundStyle(Brand.text2).disabled(compare.running)
                .help("After B, restart the server on A so it serves what it served before.")
            if compare.running {
                Button("Cancel", systemImage: "stop.fill") { compare.cancel() }.buttonStyle(.ax(.danger))
            } else {
                Button("Run Compare", systemImage: "play.fill") { model.runCompare(bPath: bPath) }
                    .buttonStyle(.ax(.primary)).disabled(reason != nil).help(reason ?? "Run A, then B")
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
        .panelSurface(radius: 12)
        .overlay(alignment: .bottomLeading) {
            if let reason, !compare.running {
                Text(reason).font(.system(size: 11)).foregroundStyle(Brand.text3).offset(x: 16, y: 18)
            }
        }
        .padding(.bottom, reason != nil && !compare.running ? 10 : 0)
    }

    // MARK: Sides

    private func side(_ isA: Bool) -> CompareSide { isA ? compare.a : compare.b }

    private func sidePanel(isA: Bool) -> some View {
        let value = side(isA)
        let summary = compare.summary
        let faster: Double? = summary?.decodeRatio.map { isA ? $0 : 1 / $0 }
        return VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Text(isA ? "A" : "B").font(.system(size: 12, weight: .bold)).foregroundStyle(isA ? colorA : colorB)
                    .frame(width: 24, height: 24)
                    .background((isA ? colorA : colorB).opacity(0.15), in: RoundedRectangle(cornerRadius: 6))
                if isA {
                    Text(value.title.isEmpty ? servingTitle : value.title).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                        .help("A is the model the server is serving when Compare starts.")
                } else {
                    Menu {
                        ForEach(model.servableModels.filter { $0.path != runtime.selectedModel }) { item in
                            Button(item.label.title + (item.label.tags.isEmpty ? "" : " · " + item.label.tags.joined(separator: " · "))) { bPath = item.path }
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Text(value.phase == .idle || value.title.isEmpty ? bTitle : value.title).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                            Image(systemName: "chevron.down").font(.system(size: 10, weight: .semibold))
                        }
                    }
                    .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize().disabled(compare.running)
                }
                Spacer()
                if let faster, faster > 1.005 {
                    StatusPill(text: String(format: "%.2f× faster", faster), tint: Brand.serve)
                }
                if value.draftAcceptedPercent != nil { Tag("MTP on", tint: Brand.boost) }
            }
            // Four across when they fit at full size, otherwise two by two.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) { stats(value) }
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                    let cells = stats(value).value
                    GridRow { cells.0; cells.1 }
                    GridRow { cells.2; cells.3 }
                }
            }
            .padding(12)
            .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
            ScrollView {
                Text(value.text.isEmpty ? placeholder(value.phase) : value.text)
                    .font(.system(size: 13)).lineSpacing(3)
                    .foregroundStyle(value.text.isEmpty ? Brand.text3 : Brand.text1)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 120)
            Divider().overlay(Brand.hairlineColor)
            HStack {
                Text(footer(value, isA: isA)).font(.system(size: 11.5)).foregroundStyle(phaseColor(value.phase)).lineLimit(2)
                Spacer()
                Button("Copy") { copyText(value.text) }.buttonStyle(.axLink).disabled(value.text.isEmpty)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .panelSurface()
    }

    private var servingTitle: String {
        model.models.productLibrary.first { $0.path == runtime.selectedModel }?.label.title
            ?? (runtime.owned ? "Serving model" : "Start a server first")
    }
    private var bTitle: String {
        model.models.productLibrary.first { $0.path == bPath }?.label.title ?? "Choose model B"
    }

    private func stats(_ value: CompareSide) -> TupleView<(some View, some View, some View, some View)> {
        TupleView((
            stat("Decode", value.performance?.decodeTokensPerSecond.flatMap(Format.rate), "tok/s"),
            stat("TTFT", value.firstTextSeconds.map { String(format: "%.0f", $0 * 1000) }, "ms")
                .help("Time to first text, from Send to the first streamed text."),
            stat("Draft", value.draftAcceptedPercent.map { String(format: "%.0f", $0) }, "%")
                .help("MTP drafts accepted during this run, from the server's counters; it includes any other traffic the server handled meanwhile."),
            stat("Total", value.totalSeconds.map { String(format: "%.1f", $0) }, "s")
        ))
    }

    private func stat(_ title: String, _ value: String?, _ unit: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.system(size: 11)).foregroundStyle(Brand.text3).lineLimit(1)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value ?? "—").font(.system(size: 17, weight: .bold)).monospacedDigit()
                    .foregroundStyle(value == nil ? Brand.text3 : Brand.text1)
                if value != nil { Text(unit).font(.system(size: 12)).foregroundStyle(Brand.text2) }
            }
            .lineLimit(1)
        }
        .fixedSize()
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func placeholder(_ phase: CompareSide.Phase) -> String {
        switch phase {
        case .idle: "The reply appears here."
        case .waiting: "Waiting for its turn."
        case .loading: "Loading this model on the server…"
        case .running: "Waiting for the first text…"
        case .stopped: "Not run."
        case .done: "The model returned no text."
        case .failed(let message): message
        }
    }

    private func footer(_ value: CompareSide, isA: Bool) -> String {
        let order = isA ? "ran first" : "ran second"
        let tokens = value.performance?.decodeTokens.map { " · \($0) output tokens" } ?? ""
        switch value.phase {
        case .idle: return isA ? "Runs first on the serving model" : "Runs second after a restart"
        case .waiting: return "Waiting"
        case .loading: return "Loading model"
        case .running: return "Generating"
        case .done: return "Done · " + order + tokens
        case .stopped: return "Stopped"
        case .failed(let message): return "Failed · " + message
        }
    }

    private func phaseColor(_ phase: CompareSide.Phase) -> Color {
        if case .failed = phase { return Brand.danger }
        return Brand.text3
    }

    // MARK: Chart + result

    private var chartPanel: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 24) {
                chart.frame(minWidth: 380)
                resultColumn.frame(width: 240)
            }
            VStack(alignment: .leading, spacing: 14) {
                chart
                resultColumn
            }
        }
        .padding(20)
        .panelSurface()
    }

    private var chart: some View {
            VStack(alignment: .leading, spacing: 4) {
                Text("Tokens over time").font(Brand.Font.cardTitle)
                Text("Steeper is faster. A flat start is time to first text. Streamed text scaled to the runtime's token count.")
                    .font(.system(size: 11.5)).foregroundStyle(Brand.text3)
                Chart {
                    ForEach(Array(compare.a.tokenCurve.enumerated()), id: \.offset) { _, point in
                        LineMark(x: .value("Seconds", point.seconds), y: .value("Tokens", point.tokens), series: .value("Side", "A"))
                            .foregroundStyle(colorA)
                    }
                    ForEach(Array(compare.b.tokenCurve.enumerated()), id: \.offset) { _, point in
                        LineMark(x: .value("Seconds", point.seconds), y: .value("Tokens", point.tokens), series: .value("Side", "B"))
                            .foregroundStyle(colorB)
                    }
                }
                .chartXAxisLabel("seconds")
                .chartYAxis { AxisMarks(position: .leading) { _ in AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [3, 3])); AxisValueLabel() } }
                .chartXAxis { AxisMarks { _ in AxisValueLabel() } }
                .frame(minHeight: 150)
                .overlay {
                    if compare.a.tokenCurve.isEmpty && compare.b.tokenCurve.isEmpty {
                        Text("No runs here yet").font(.system(size: 12)).foregroundStyle(Brand.text3)
                    }
                }
            }
    }

    private var resultColumn: some View {
        let summary = compare.summary
        return VStack(alignment: .leading, spacing: 10) {
            SectionLabel("Result")
            if let summary {
                resultRow(winnerTitle(summary.finished?.winner),
                          summary.finished.map { $0.winner == .tie ? "Same time" : String(format: "%.1f s sooner", $0.seconds) },
                          tint: Brand.serve)
                resultRow("Decode", summary.decodeRatio.map { ratio in
                    abs(ratio - 1) < 0.005 ? "Same" : ratio > 1 ? String(format: "A %.2f× faster", ratio) : String(format: "B %.2f× faster", 1 / ratio) })
                resultRow("TTFT", summary.firstTextDeltaMs.map { delta in
                    abs(delta) < 50 ? "Within 50 ms" : String(format: "%@ %.0f ms sooner", delta > 0 ? "A" : "B", abs(delta)) })
                resultRow("Draft accepted", summary.draftDeltaPoints.map { String(format: "A %+.0f pts", $0) }, tint: Brand.boost)
            } else {
                Text(compare.running ? "Declared after both runs finish." : "Run both sides to see a result.")
                    .font(.system(size: 12)).foregroundStyle(Brand.text3)
            }
            Spacer()
            HStack(spacing: 14) {
                legend(colorA, compare.a.title.isEmpty ? "A" : "A " + compare.a.title)
                legend(colorB, compare.b.title.isEmpty ? "B" : "B " + compare.b.title)
            }
            Text("One run per side on this Mac; not a benchmark. Draft counts all server traffic during each run.")
                .font(.system(size: 10.5)).foregroundStyle(Brand.text3).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func winnerTitle(_ winner: CompareSummary.Winner?) -> String {
        switch winner {
        case .a: "A finished"
        case .b: "B finished"
        case .tie, nil: "Finished"
        }
    }

    private func resultRow(_ title: String, _ value: String?, tint: Color = Brand.text1) -> some View {
        HStack {
            Text(title).font(.system(size: 12.5)).foregroundStyle(Brand.text2)
            Spacer()
            Text(value ?? "—").font(.system(size: 12.5, weight: .semibold)).monospacedDigit()
                .foregroundStyle(value == nil ? Brand.text3 : tint)
        }
    }

    private func legend(_ color: Color, _ title: String) -> some View {
        HStack(spacing: 5) {
            RoundedRectangle(cornerRadius: 1).fill(color).frame(width: 10, height: 2.5)
            Text(title).font(.system(size: 11)).foregroundStyle(Brand.text2).lineLimit(1)
        }
    }
}
