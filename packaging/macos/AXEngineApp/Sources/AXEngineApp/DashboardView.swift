import SwiftUI
import Charts
import AXEngineAppCore

struct DashboardView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var models: ModelStore
    @ObservedObject var monitor: HostMonitor
    @ObservedObject var storage: StorageMonitor
    @Environment(\.scenePhase) private var scenePhase
    @State private var detailsExpanded = false

    var body: some View {
        GeometryReader { geometry in
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    Label("This Mac", systemImage: "desktopcomputer").font(.headline)
                    Text(memoryTotal + " unified memory").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Picker("History range", selection: $model.dashboardHistoryWindow) {
                        ForEach(ActivityHistoryRange.allCases, id: \.rawValue) { range in
                            Text(range.label).tag(range.rawValue)
                        }
                    }.pickerStyle(.menu).labelsHidden().frame(width: 112)
                        .accessibilityLabel("Activity history range")
                    Toggle("Live activity", isOn: $model.hostMonitoringEnabled)
                        .toggleStyle(.switch).controlSize(.small).fixedSize()
                    Button {
                        Task {
                            async let capacity: Void = storage.refresh(path: model.settings.modelsDirectory)
                            async let library: Void = models.scan(settings: model.settings)
                            _ = await (capacity, library)
                        }
                    } label: { Image(systemName: "arrow.clockwise") }
                    .help("Refresh storage and model inventory").accessibilityLabel("Refresh storage and model inventory")
                    .disabled(storage.refreshing || models.scanning)
                }.frame(height: 36)
                TimelineView(.periodic(from: .now, by: 2)) { context in
                    VStack(spacing: 20) {
                    if geometry.size.width >= 640 {
                        HStack(alignment: .top, spacing: 14) {
                            ramChart(at: context.date).frame(minWidth: 260)
                            StorageCapacityCard(storage: storage).frame(minWidth: 260)
                        }
                    } else {
                        VStack(spacing: 14) {
                            ramChart(at: context.date)
                            StorageCapacityCard(storage: storage)
                        }
                    }
                    if geometry.size.width >= 720 {
                        HStack(alignment: .top, spacing: 14) {
                            activityChart("CPU", key: "cpu", tint: Brand.accentChrome, symbol: "cpu", at: context.date).frame(minWidth: 180)
                            activityChart("GPU", key: "gpu", tint: .blue, symbol: "square.3.layers.3d", at: context.date).frame(minWidth: 180)
                            ModelInventoryCard(models: models).frame(minWidth: 210)
                        }
                    } else {
                        VStack(spacing: 14) {
                            HStack(spacing: 14) {
                                activityChart("CPU", key: "cpu", tint: Brand.accentChrome, symbol: "cpu", at: context.date)
                                activityChart("GPU", key: "gpu", tint: .blue, symbol: "square.3.layers.3d", at: context.date)
                            }
                            ModelInventoryCard(models: models)
                        }
                    }
                    }
                }
                HStack(spacing: 7) {
                    Image(systemName: monitor.active ? "circle.fill" : "pause.circle")
                        .foregroundStyle(monitor.active ? Brand.accentChrome : .secondary)
                    Text(model.hostMonitoringEnabled
                         ? monitor.active ? "Live activity · last \(historyRange.label)" : "Activity monitoring paused"
                         : "Activity monitoring is off")
                    Spacer()
                    Text("Hover over a history to inspect a sample")
                }.font(.caption).foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    Label("Playground profile: " + model.settings.effectiveProfile.title, systemImage: "slider.horizontal.3")
                    Button("Change") { model.page = .settings }
                        .buttonStyle(.borderless)
                    Spacer()
                }
                .font(.caption)
                .help("Saved defaults for the next Playground request. Other clients use their own parameters.")
                DisclosureGroup("How these measurements work", isExpanded: $detailsExpanded) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("RAM is an estimate from active, wired and compressed pages, not model memory or an admission guarantee. CPU uses process averages normalized by logical cores. GPU is available when IOAccelerator exposes it. Missing and stale samples leave gaps. Live activity continues across pages and in the background while this window is open. Turn it off to stop recording. Activity history stays in memory for up to four hours and resets when the app quits. Longer charts preserve each time bucket’s low and high readings; hover reads an original sample.")
                        Text("Storage shows the volume containing your download folder, including other files and APFS allocations. Model caches can live on other volumes. The app does not scan model weights to estimate disk usage.")
                        Text("Model counts cover discovered AutomatosX folders, not verified runtime support. One cached snapshot per repository is counted. Incomplete includes metadata-only folders; unchecked folders are retained from earlier scans.")
                        Text("Last OS memory-pressure event: " + (monitor.pressure ?? "Not reported"))
                        Text("Download folder: " + model.settings.modelsDirectory).textSelection(.enabled)
                        if let snapshot = storage.snapshot {
                            Text("Storage read from: " + snapshot.resolvedPath).textSelection(.enabled)
                        }
                    }.font(.callout).foregroundStyle(.secondary).padding(.top, 8)
                }.font(.caption).foregroundStyle(.secondary)
            }
            .frame(maxWidth: 1240).padding(26).frame(maxWidth: .infinity)
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
        let percent = monitor.history.latest("memory", now: date)
        return MetricHistoryChart(title: "RAM usage", history: monitor.history, key: "memory", now: date,
            percent: true, unit: "%", tint: .purple, symbol: "memorychip",
            detail: percent.map { String(format: "%.1f%%", $0) + " of " + memoryTotal } ?? "Estimated unified memory",
            emptyDetail: model.hostMonitoringEnabled ? "Awaiting measurements" : "Enable Live activity",
            window: model.dashboardHistoryWindow,
            readoutMultiplier: Double(ProcessInfo.processInfo.physicalMemory) / 100 / 1_073_741_824,
            readoutUnit: "GB")
    }
    private func activityChart(_ title: String, key: String, tint: Color, symbol: String, at date: Date) -> some View {
        MetricHistoryChart(title: title, history: monitor.history, key: key, now: date,
            percent: true, unit: "%", tint: tint, symbol: symbol,
            detail: key == "cpu" ? "Estimated utilization" : "Device utilization",
            emptyDetail: model.hostMonitoringEnabled ? "Awaiting measurements" : "Enable Live activity",
            window: model.dashboardHistoryWindow)
    }
}

private struct StorageCapacityCard: View {
    @ObservedObject var storage: StorageMonitor
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Disk space", systemImage: "internaldrive").font(.system(size: 13, weight: .semibold))
            if let value = storage.snapshot {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(bytes(value.availableBytes)).font(.system(size: 32, weight: .semibold, design: .rounded)).monospacedDigit()
                    Text("available").font(.callout).foregroundStyle(.secondary)
                }
                Text(value.volumeName + " · " + bytes(value.totalBytes) + " capacity")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1).help(value.volumeName)
                Chart {
                    SectorMark(angle: .value("Bytes", value.usedBytes), innerRadius: .ratio(0.74), angularInset: 2)
                        .foregroundStyle(Color.secondary.opacity(0.25))
                        .accessibilityLabel("Used or reserved").accessibilityValue(bytes(value.usedBytes))
                    SectorMark(angle: .value("Bytes", value.availableBytes), innerRadius: .ratio(0.74), angularInset: 2)
                        .foregroundStyle(Color.teal)
                        .accessibilityLabel("Available disk space").accessibilityValue(bytes(value.availableBytes))
                }
                .chartPlotStyle { $0.drawingGroup(opaque: false, colorMode: .linear) }
                .chartBackground { _ in
                    VStack(spacing: 3) {
                        Text(String(format: "%.0f%%", Double(value.availableBytes) / Double(value.totalBytes) * 100))
                            .font(.system(size: 23, weight: .semibold, design: .rounded))
                        Text("available").font(.caption).foregroundStyle(.secondary)
                    }.accessibilityHidden(true)
                }
                .frame(height: 108)
                HStack {
                    Label("Used / reserved " + bytes(value.usedBytes), systemImage: "circle.fill").foregroundStyle(.secondary)
                    Spacer()
                    if let updated = storage.updatedAt { Text(updated.formatted(date: .omitted, time: .shortened)).foregroundStyle(.secondary) }
                }.font(.system(size: 10))
            } else {
                Text("—").font(.system(size: 32, weight: .semibold, design: .rounded))
                Text(storage.refreshing ? "Reading model volume…" : "Storage unavailable")
                    .font(.callout).foregroundStyle(.secondary)
                Image(systemName: "internaldrive").font(.system(size: 40)).foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, minHeight: 108)
                Text(storage.error ?? "Volume containing your model downloads")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
        }.padding(18).frame(maxWidth: .infinity, minHeight: 278, alignment: .topLeading)
            .background(Brand.cardSurface, in: RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Brand.separator.opacity(0.6), lineWidth: 0.5))
    }
    private func bytes(_ value: Int64) -> String { ByteCountFormatter.string(fromByteCount: value, countStyle: .file) }
}

private struct ModelInventoryCard: View {
    @ObservedObject var models: ModelStore
    var body: some View {
        let inventory = ModelInventory(library: models.productLibrary, checked: models.checkedPaths)
        let hasInventory = models.lastScanAt != nil || !models.productLibrary.isEmpty
        VStack(alignment: .leading, spacing: 14) {
            Label("Models on disk", systemImage: "square.stack.3d.up").font(.system(size: 13, weight: .semibold))
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(hasInventory ? String(inventory.total) : "—")
                    .font(.system(size: 32, weight: .semibold, design: .rounded)).monospacedDigit()
                Text("folders").font(.callout).foregroundStyle(.secondary)
            }
            Text(models.scanning ? "Checking your library…" : models.libraryIssue != nil ? "Partial scan · counts may be incomplete" : "Discovered in your library")
                .font(.caption).foregroundStyle(.secondary)
            if hasInventory {
                Chart {
                BarMark(x: .value("Folders", inventory.complete), y: .value("Status", "Complete"))
                    .foregroundStyle(Brand.accentChrome).cornerRadius(3)
                    .annotation(position: .trailing) { Text(String(inventory.complete)).font(.caption.monospacedDigit()) }
                BarMark(x: .value("Folders", inventory.incomplete), y: .value("Status", "Incomplete"))
                    .foregroundStyle(Color.orange).cornerRadius(3)
                    .annotation(position: .trailing) { Text(String(inventory.incomplete)).font(.caption.monospacedDigit()) }
                BarMark(x: .value("Folders", inventory.unchecked), y: .value("Status", "Unchecked"))
                    .foregroundStyle(Color.secondary).cornerRadius(3)
                    .annotation(position: .trailing) { Text(String(inventory.unchecked)).font(.caption.monospacedDigit()) }
            }
            .chartXScale(domain: 0...max(1, inventory.total))
            .chartXAxis(.hidden)
            .chartYAxis { AxisMarks { AxisValueLabel().font(.system(size: 10)) } }
            .chartPlotStyle { $0.padding(.trailing, 20).drawingGroup(opaque: false, colorMode: .linear) }
            .frame(height: 108)
                .accessibilityLabel("Discovered model folders by file status")
            } else {
                Text(models.scanning ? "Discovering model folders…" : "Model inventory unavailable")
                    .font(.callout).foregroundStyle(.secondary).frame(maxWidth: .infinity, minHeight: 108)
            }
            if let date = models.lastScanAt, !models.scanning, models.libraryIssue == nil {
                Text("Checked " + date.formatted(date: .omitted, time: .shortened))
                    .font(.system(size: 10)).foregroundStyle(.secondary)
            } else {
                Text("Complete files ≠ runtime compatibility").font(.system(size: 10)).foregroundStyle(.secondary)
            }
        }.padding(18).frame(maxWidth: .infinity, minHeight: 278, alignment: .topLeading)
            .background(Brand.cardSurface, in: RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Brand.separator.opacity(0.6), lineWidth: 0.5))
    }
}
