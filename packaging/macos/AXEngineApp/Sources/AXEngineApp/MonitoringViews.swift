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

    var body: some View {
        let points = history.points(key, now: now)
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title).font(.callout.weight(.medium))
                Spacer()
                Text(history.latest(key, now: now).map { String(format: "%.1f", $0 / divisor) + unit } ?? "—")
                    .monospacedDigit()
            }
            Chart(points) { point in
                LineMark(x: .value("Time", point.time), y: .value(title, point.value / divisor),
                         series: .value("Continuous samples", point.segment))
                    .interpolationMethod(.linear).foregroundStyle(Brand.accentChrome)
                PointMark(x: .value("Time", point.time), y: .value(title, point.value / divisor))
                    .symbolSize(8).foregroundStyle(Brand.accentChrome)
            }
            .chartXScale(domain: now.addingTimeInterval(-300)...now)
            .chartYScale(domain: 0...(percent ? 100 : max(1, (points.map { $0.value / divisor }.max() ?? 1) * 1.1)))
            .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) { AxisValueLabel(format: .dateTime.hour().minute()) } }
            .frame(height: 110)
            .accessibilityLabel(title + ". Last five minutes. Missing samples are gaps.")
            if points.isEmpty { Text("No measurements available").font(.caption).foregroundStyle(.secondary) }
        }
        .accessibilityElement(children: .contain)
    }
}

struct HostMonitoringView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var monitor: HostMonitor
    var body: some View {
        Card("This Mac · monitoring", subtitle: "Optional host-wide measurements, separate from the inference runtime.") {
            Toggle("Show CPU, GPU and memory history", isOn: $model.hostMonitoringEnabled)
            if model.hostMonitoringEnabled {
                Text(monitor.active ? (monitor.collecting ? "Collecting a sample…" : "Sampling while this page is active") : "Paused · sampling resumes when this page is active")
                    .font(.callout).foregroundStyle(.secondary)
                TimelineView(.periodic(from: .now, by: 2)) { context in
                    VStack(spacing: 18) {
                        ViewThatFits(in: .horizontal) {
                            HStack(alignment: .top, spacing: 20) {
                                hostChart("CPU estimate", key: "cpu", at: context.date).frame(minWidth: 190)
                                hostChart("GPU utilization", key: "gpu", at: context.date).frame(minWidth: 190)
                                hostChart("Memory estimate", key: "memory", at: context.date).frame(minWidth: 190)
                            }
                            VStack(spacing: 20) {
                                hostChart("CPU estimate", key: "cpu", at: context.date)
                                hostChart("GPU utilization", key: "gpu", at: context.date)
                                hostChart("Memory estimate", key: "memory", at: context.date)
                            }
                        }
                    }
                }
                HStack {
                    Text("Last OS memory-pressure event: " + (monitor.pressure ?? "Not reported"))
                    if let date = monitor.pressureAt { Text(date.formatted(date: .omitted, time: .standard)) }
                }.font(.caption)
                Text("CPU: process averages normalized by logical cores. Memory: active + wired + compressed pages / unified memory. GPU: IOAccelerator when exposed. These are whole-Mac estimates, not model throughput or a memory-fit guarantee. Missing or old samples show a dash; gaps are not filled.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
    private func hostChart(_ title: String, key: String, at date: Date) -> some View {
        MetricHistoryChart(title: title, history: monitor.history, key: key, now: date, percent: true, unit: "%")
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
