import SwiftUI
import Charts
import AXEngineAppCore

/// System: this Mac's memory, CPU and GPU history, model storage and the
/// model inventory. Moved off Home so Home can lead with the model's speed.
struct SystemView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @ObservedObject var monitor: HostMonitor
    @ObservedObject var storage: StorageMonitor
    @Environment(\.scenePhase) private var scenePhase
    @State private var detailsExpanded = false

    var body: some View {
        GeometryReader { geometry in
            let wide = geometry.size.width >= 820
            PageScroll {
                HStack(spacing: 12) {
                    Text(monitor.active ? "Your Mac, as the runtime sees it. Live while this window is open; history resets on quit."
                                        : model.hostMonitoringEnabled ? "Activity monitoring is paused." : "Live activity is off. Turn it on to record Mac history.")
                        .font(.system(size: 12.5)).foregroundStyle(Brand.text2)
                    Spacer()
                    Toggle("Live activity", isOn: $model.hostMonitoringEnabled)
                        .toggleStyle(.switch).controlSize(.mini).font(.system(size: 12.5)).fixedSize()
                    AXSegmented(options: [(ActivityHistoryRange.fiveMinutes.rawValue, "5m"), (ActivityHistoryRange.hour.rawValue, "1h"),
                                          (ActivityHistoryRange.fourHours.rawValue, "4h")],
                                selection: $model.dashboardHistoryWindow, compact: true)
                        .accessibilityLabel("Activity history range")
                    Button {
                        Task {
                            async let capacity: Void = storage.refresh(path: model.settings.modelsDirectory)
                            async let library: Void = models.scan(settings: model.settings)
                            _ = await (capacity, library)
                        }
                    } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.ax(.secondary, compact: true))
                    .help("Refresh storage and model inventory").accessibilityLabel("Refresh storage and model inventory")
                    .disabled(storage.refreshing || models.scanning)
                }
                MemoryPressureBanner(model: model, monitor: monitor, runtime: model.runtime, showsSystemLink: false)
                TimelineView(.periodic(from: .now, by: 2)) { context in
                    let grid = wide ? [GridItem(.flexible(), spacing: 16), GridItem(.flexible(), spacing: 16)] : [GridItem(.flexible())]
                    LazyVGrid(columns: grid, spacing: 16) {
                        ramChart(at: context.date)
                        activityChart("GPU", key: "gpu", tint: Brand.boost, symbol: "cpu", detail: "busy", at: context.date)
                        activityChart("CPU", key: "cpu", tint: Brand.serve, symbol: "cpu.fill", detail: "average", at: context.date)
                        StorageCapacityCard(storage: storage, models: models, downloadFolder: model.settings.modelsDirectory)
                    }
                }
                FactsPanel(model: model, models: models, monitor: monitor, storage: storage, detailsExpanded: $detailsExpanded)
            }
        }
        .task(id: "\(scenePhase == .active):\(model.settings.modelsDirectory)") {
            guard scenePhase == .active else { return }
            while !Task.isCancelled {
                await storage.refresh(path: model.settings.modelsDirectory)
                do { try await Task.sleep(for: .seconds(60)) } catch { return }
            }
        }
    }
    private var historyRange: ActivityHistoryRange {
        ActivityHistoryRange(rawValue: model.dashboardHistoryWindow) ?? .minute
    }
    private var memoryTotal: String {
        ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)
    }
    private func ramChart(at date: Date) -> some View {
        MetricHistoryChart(title: "Memory", history: monitor.history, key: "memory", now: date,
            percent: true, unit: "%", tint: .purple, symbol: "memorychip",
            detail: "of " + HostInfo.memory,
            emptyDetail: model.hostMonitoringEnabled ? "Awaiting measurements" : "Live activity off",
            window: model.dashboardHistoryWindow,
            readoutMultiplier: Double(ProcessInfo.processInfo.physicalMemory) / 100 / 1_073_741_824,
            readoutUnit: "GB")
    }
    private func activityChart(_ title: String, key: String, tint: Color, symbol: String, detail: String, at date: Date) -> some View {
        MetricHistoryChart(title: title, history: monitor.history, key: key, now: date,
            percent: true, unit: "%", tint: tint, symbol: symbol, detail: detail,
            emptyDetail: model.hostMonitoringEnabled ? (key == "gpu" ? "Not reported by this Mac yet" : "Awaiting measurements") : "Live activity off",
            window: model.dashboardHistoryWindow)
    }
}

/// Chip, memory, OS and app facts, pressure, and how the numbers are measured.
private struct FactsPanel: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @ObservedObject var monitor: HostMonitor
    @ObservedObject var storage: StorageMonitor
    @Binding var detailsExpanded: Bool
    var body: some View {
        let inventory = ModelInventory(library: models.productLibrary, checked: models.checkedPaths)
        Card {
            HStack(alignment: .top, spacing: 0) {
                fact("Chip", HostInfo.chip)
                fact("Memory", HostInfo.memory + " unified")
                fact("macOS", HostInfo.macOS)
                fact("AX Engine", HostInfo.version ?? "Development")
                fact("Models", models.lastScanAt == nil && models.productLibrary.isEmpty ? "—" : "\(inventory.complete) ready")
                    .help("\(inventory.complete) ready · \(inventory.incomplete) incomplete · \(inventory.unchecked) unchecked")
                fact("Pressure", monitor.pressure ?? "Not reported", tint: pressureTint)
                    .help("macOS memory pressure from the last system event; none has arrived yet when it says Not reported.")
            }
            Divider().overlay(Brand.hairlineColor)
            HStack {
                Button("Copy diagnostics", systemImage: "doc.on.clipboard") { copyText(diagnostics) }.buttonStyle(.axLink)
                    .foregroundStyle(Brand.text2)
                Spacer()
                Button(detailsExpanded ? "Hide how these numbers are measured" : "How these numbers are measured ›") {
                    detailsExpanded.toggle()
                }.buttonStyle(.axLink)
            }
            if detailsExpanded {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Memory is an estimate from active, wired and compressed pages, not model memory or an admission guarantee. CPU uses process averages normalized by logical cores. GPU is available when IOAccelerator exposes it. Missing and stale samples leave gaps. Activity history stays in memory for up to four hours and resets when the app quits. Longer charts keep each time bucket's low and high readings; hover reads an original sample.")
                    Text("Model storage shows the volume holding your download folder. Model bytes come from the library index; other files on the volume are shown in gray. Model caches can live on other volumes.")
                    Text("Download folder: " + model.settings.modelsDirectory).textSelection(.enabled)
                }
                .font(Brand.Font.caption).foregroundStyle(Brand.text3)
            }
        }
    }
    private var pressureTint: Color {
        switch monitor.pressure {
        case "Critical": Brand.danger
        case "Warning": Brand.warn
        case nil: Brand.text3
        default: Brand.serve
        }
    }
    private var diagnostics: String {
        ["AX Engine " + (HostInfo.version ?? "development"), HostInfo.chip, HostInfo.memory + " unified memory",
         "macOS " + HostInfo.macOS, "Memory pressure: " + (monitor.pressure ?? "not reported"),
         "Models: \(models.productLibrary.count) discovered"].joined(separator: "\n")
    }
    private func fact(_ title: String, _ value: String, tint: Color = Brand.text1) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 11.5)).foregroundStyle(Brand.text3)
            Text(value).font(.system(size: 13, weight: .semibold)).foregroundStyle(tint).lineLimit(1).minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}

/// Model storage: the download folder's volume, with model bytes from the
/// library index in the accent and other files in gray.
private struct StorageCapacityCard: View {
    @ObservedObject var storage: StorageMonitor
    @ObservedObject var models: ModelStore
    let downloadFolder: String
    var body: some View {
        let managedBytes = models.productLibrary.filter(\.managed).compactMap(\.sizeBytes).reduce(0, +)
        let cachedBytes = models.productLibrary.filter { !$0.managed }.compactMap(\.sizeBytes).reduce(0, +)
        Card("Model storage", trailing: {
            Text(storage.snapshot.map { bytes($0.availableBytes) + " free" } ?? (storage.refreshing ? "Reading…" : "—"))
                .font(.system(size: 12.5, weight: .semibold)).monospacedDigit()
        }) {
            if let value = storage.snapshot {
                volume(name: value.volumeName, role: "Download folder", models: managedBytes,
                       used: value.usedBytes, total: value.totalBytes, tint: Brand.accent)
                if cachedBytes > 0 {
                    HStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 2).fill(Brand.boost).frame(width: 9, height: 9)
                        Text("Hugging Face cache").font(.system(size: 12.5)).foregroundStyle(Brand.text2)
                        Spacer()
                        Text(bytes(cachedBytes) + " models").font(.system(size: 12)).foregroundStyle(Brand.text3).monospacedDigit()
                    }
                }
            } else {
                Text(storage.error ?? "Volume containing your model downloads")
                    .font(Brand.Font.body).foregroundStyle(Brand.text3).frame(maxWidth: .infinity, minHeight: 80)
            }
            Spacer(minLength: 0)
            Text("Model bytes come from the library index; other files are shown in gray.")
                .font(.system(size: 11)).foregroundStyle(Brand.text3)
            if let updated = storage.updatedAt {
                Text("Checked " + updated.formatted(date: .omitted, time: .shortened)).font(.system(size: 11)).foregroundStyle(Brand.text3)
            }
        }
        .frame(minHeight: 236)
    }
    private func volume(name: String, role: String, models: Int64, used: Int64, total: Int64, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(name).font(.system(size: 13, weight: .semibold)).lineLimit(1).truncationMode(.middle).layoutPriority(2)
                Text(role).font(.system(size: 11.5)).foregroundStyle(Brand.text3).lineLimit(1)
                Spacer(minLength: 6)
                Text((models > 0 ? bytes(models) + " models · " : "") + bytes(total)).font(.system(size: 11.5)).foregroundStyle(Brand.text3)
                    .monospacedDigit().lineLimit(1).fixedSize()
            }
            // Indexed model bytes can exceed this volume's used space (shared
            // blocks, other volumes); then show used versus free only.
            StackedBar(segments: models <= used ? [
                .init(id: "models", value: Double(models), color: tint),
                .init(id: "other", value: Double(used - models), color: Brand.text3.opacity(0.6)),
            ] : [.init(id: "used", value: Double(used), color: Brand.text3.opacity(0.6))], total: Double(max(total, 1)), height: 8)
        }
        .help(name)
    }
    private func bytes(_ value: Int64) -> String { ByteCountFormatter.string(fromByteCount: value, countStyle: .file) }
}
