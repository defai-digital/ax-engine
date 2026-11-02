import SwiftUI
import Charts
import AXEngineAppCore

struct MetricHistoryChart: View {
    let title: String
    let history: MetricHistory
    let key: String
    let now: Date
    var percent = false
    var divisor: Double = 1
    var unit = ""
    var tint: Color = Brand.accentChrome
    var symbol = "waveform.path"
    var detail = "Runtime measurement"
    var emptyDetail = "Awaiting measurements"
    var window: TimeInterval = 300
    var readoutMultiplier: Double = 1
    var readoutUnit: String? = nil
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var contrast
    @State private var selectedDate: Date?

    var body: some View {
        let range = ActivityHistoryRange(rawValue: window) ?? .fiveMinutes
        let start = now.addingTimeInterval(-window)
        let points = history.plotPoints(key, window: window, now: now)
        let selected = selectedDate.flatMap { history.sample(key, near: $0, now: now) }.flatMap { $0.time >= start ? $0 : nil }
        let latest = selectedDate == nil ? history.latest(key, now: now) : selected?.values[key]
        let segmentCounts = Dictionary(grouping: points, by: \.segment).mapValues(\.count)
        let ceiling = percent ? 100 : max(1, (points.map { $0.value / divisor }.max() ?? 1) * 1.1)
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: symbol).foregroundStyle(tint).accessibilityHidden(true)
                Text(title).font(.system(size: 13, weight: .semibold))
            }
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(latest.map { String(format: "%.1f", $0 / divisor * readoutMultiplier) } ?? "—")
                    .font(.system(size: 32, weight: .semibold, design: .rounded))
                    .monospacedDigit().contentTransition(reduceMotion ? .identity : .numericText())
                if latest != nil {
                    Text((readoutUnit ?? unit).trimmingCharacters(in: .whitespaces))
                        .font(.system(size: 16, weight: .medium)).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .accessibilityElement(children: .combine)
            Text(selectedDate != nil
                 ? selected.map { $0.time.formatted(date: .omitted, time: .standard) } ?? "No sample at this time"
                 : latest == nil ? (points.isEmpty ? emptyDetail : "No current sample") : detail)
                .font(.caption).foregroundStyle(.secondary)
            Chart {
                ForEach(points) { point in
                    AreaMark(x: .value("Time", point.time), y: .value(title, point.value / divisor),
                             series: .value("Continuous samples", point.segment))
                        .interpolationMethod(.linear)
                        .foregroundStyle(LinearGradient(colors: [tint.opacity(0.22), tint.opacity(0.01)], startPoint: .top, endPoint: .bottom))
                        .accessibilityHidden(true)
                    LineMark(x: .value("Time", point.time), y: .value(title, point.value / divisor),
                             series: .value("Continuous samples", point.segment))
                        .interpolationMethod(.linear).foregroundStyle(tint)
                        .lineStyle(StrokeStyle(lineWidth: contrast == .increased ? 3 : 2, lineCap: .round, lineJoin: .round))
                    // Keep isolated samples visible without dotting every point on a line.
                    if point.id == points.last?.id || segmentCounts[point.segment] == 1 {
                        PointMark(x: .value("Time", point.time), y: .value(title, point.value / divisor))
                            .symbolSize(18).foregroundStyle(tint).accessibilityHidden(true)
                    }
                }
            }
            .chartOverlay { chart in
                GeometryReader { geometry in
                    if let selected, let anchor = chart.plotFrame, let x = chart.position(forX: selected.time) {
                        let frame = geometry[anchor]
                        Rectangle().fill(.secondary.opacity(0.6))
                            .frame(width: 1, height: frame.height)
                            .position(x: frame.minX + x, y: frame.midY)
                            .allowsHitTesting(false).accessibilityHidden(true)
                    }
                    Color.clear.contentShape(Rectangle())
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let location): inspect(location, chart: chart, geometry: geometry)
                            case .ended: selectedDate = nil
                            }
                        }
                        .gesture(SpatialTapGesture().onEnded { inspect($0.location, chart: chart, geometry: geometry) })
                        .accessibilityHidden(true)
                }
            }
            .chartXScale(domain: start...now)
            .chartYScale(domain: 0...ceiling)
            .chartXAxis(.hidden)
            .chartYAxis {
                AxisMarks(position: .leading, values: [0, ceiling / 2, ceiling]) { value in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [3, 4]))
                        .foregroundStyle(.secondary.opacity(0.2))
                    AxisValueLabel {
                        if let number = value.as(Double.self) {
                            Text(number.formatted(.number.precision(.fractionLength(0...1))) + (percent ? "%" : ""))
                                .font(.system(size: 10)).foregroundStyle(.secondary)
                                .accessibilityHidden(true)
                        }
                    }
                }
            }
            .chartPlotStyle { plot in
                // Rasterize only the plot through SwiftUI's Metal-backed path.
                // Keep controls, labels and chart accessibility outside this layer.
                plot.padding(.vertical, 4).padding(.horizontal, 4)
                    .drawingGroup(opaque: false, colorMode: .linear)
            }
            .frame(height: 108)
            .accessibilityLabel(title + ". Last \(range.spokenLabel). Missing samples are gaps.")
            .accessibilityValue(percent ? "Vertical scale: zero to one hundred percent." : "Values in " + (unit.isEmpty ? "count" : unit))
            HStack {
                Text("\(range.label) ago")
                Spacer()
                Text("Now")
            }.font(.system(size: 10)).foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Brand.cardSurface, in: RoundedRectangle(cornerRadius: 16))
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Brand.separator.opacity(0.6), lineWidth: 0.5))
        .accessibilityElement(children: .contain)
        .onDisappear { selectedDate = nil }
        .onChange(of: window) { _, _ in selectedDate = nil }
    }
    private func inspect(_ location: CGPoint, chart: ChartProxy, geometry: GeometryProxy) {
        guard let anchor = chart.plotFrame else { selectedDate = nil; return }
        let frame = geometry[anchor]
        guard frame.contains(location) else { selectedDate = nil; return }
        selectedDate = chart.value(atX: location.x - frame.minX, as: Date.self)
    }
}

struct RuntimeHistoryView: View {
    @ObservedObject var runtime: RuntimeController
    @State private var expanded = false
    var body: some View {
        Card("Runtime history", subtitle: "Existing service polls · bounded to the last five minutes · kept in memory only") {
            DisclosureGroup("Show history and MTP diagnostics", isExpanded: $expanded) {
                TimelineView(.periodic(from: .now, by: 2)) { context in
                    VStack(alignment: .leading, spacing: 18) {
                        MetricHistoryChart(title: "MLX active memory", history: runtime.history,
                            key: "ax_engine_memory_mlx_active_bytes", now: context.date, divisor: 1_073_741_824, unit: " GiB")
                        MetricHistoryChart(title: "Active streams", history: runtime.history,
                            key: "ax_engine_generation_active_streams", now: context.date)
                        MetricHistoryChart(title: "MTP accepted drafts · sample interval", history: runtime.history,
                            key: "mtpAcceptedPercent", now: context.date, percent: true, unit: "%")
                    }.padding(.top, 12)
                }
                Text("MTP acceptance uses changes in accepted/draft counters between successful polls. Idle intervals, missing counters and counter resets show a gap. Acceptance is not a speedup measurement. Histories reset when the connection or reported model changes; probe failures leave gaps.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
