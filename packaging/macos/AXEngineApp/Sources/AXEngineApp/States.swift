import SwiftUI
import AXEngineAppCore

/// Home's runtime problem panel (`12-states.svg`: "Runtime stopped" and
/// "Port busy"): what happened, the last problem lines, and the next step.
struct RuntimeIssuePanel: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    let error: String
    @State private var copied = false

    private var portBusy: Bool { error.contains("already in use") }
    private var problemLines: [String] {
        runtime.logs.split(separator: "\n")
            .filter { LogFilter.level($0) != .info }.suffix(3).map(String.init)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if portBusy { portBusyBody } else { stoppedBody }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .panelSurface()
    }

    private var stoppedBody: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle").foregroundStyle(Brand.danger)
                VStack(alignment: .leading, spacing: 3) {
                    Text("The server stopped").font(.system(size: 14, weight: .semibold)).foregroundStyle(Brand.danger)
                    Text(error).font(.system(size: 12)).foregroundStyle(Brand.text2).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Brand.danger.opacity(0.12), in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
            if !problemLines.isEmpty {
                VStack(alignment: .leading, spacing: 5) {
                    ForEach(Array(problemLines.enumerated()), id: \.offset) { _, line in
                        Text(line).font(.system(size: 11.5, design: .monospaced)).lineLimit(1).truncationMode(.middle)
                            .foregroundStyle(LogFilter.level(Substring(line)) == .error ? Brand.danger : Brand.warn)
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
            }
            HStack(spacing: 10) {
                if !runtime.owned && runtime.connection == nil && models.canSelectInGUI(runtime.selectedModel) {
                    Button("Start Again", systemImage: "play.fill") {
                        model.selectedPath = runtime.selectedModel
                        model.startServer()
                    }
                    .buttonStyle(.axPrimary).disabled(runtime.busy || models.removing || models.scanning)
                }
                Button(copied ? "Copied" : "Copy Report", systemImage: copied ? "checkmark" : "doc.on.doc") {
                    copyText("AX Engine runtime error: " + error + "\n\n" + runtime.logs)
                    copied = true
                }
                .buttonStyle(.axSecondary)
                .help("Copies the error and the bounded runtime output. It contains no API key.")
                Button("Open Logs") { model.page = .api }.buttonStyle(.axLink)
            }
            if let hint = memoryHint {
                Text(hint).font(.system(size: 11.5)).foregroundStyle(Brand.text3)
            }
            if models.library.contains(where: { $0.path == runtime.selectedModel && !$0.isAutomatosX }) {
                Text("This model was not published by AutomatosX. The runtime checks compatibility at start, so complete files can still be refused; the files stay on disk.")
                    .font(.system(size: 11.5)).foregroundStyle(Brand.text3).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var models: ModelStore { model.models }

    /// Only when the host monitor has a recent reading.
    private var memoryHint: String? {
        guard let percent = model.hostMonitor.history.latest("memory", now: Date(), maxAge: 30) else { return nil }
        let used = percent / 100 * Double(ProcessInfo.processInfo.physicalMemory)
        guard percent > 70, let text = Format.bytes(used) else { return nil }
        return "This Mac has \(text) in use. If the model ran out of memory, close other apps or pick a smaller model."
    }

    private var portBusyBody: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle").foregroundStyle(Brand.warn)
                Text("Port \(String(model.settings.port)) is already in use").font(.system(size: 15, weight: .semibold))
                Spacer()
                Tag("Conflict", tint: Brand.warn)
            }
            Text("Another program, possibly another AX Engine server, is listening there.")
                .font(.system(size: 12.5)).foregroundStyle(Brand.text2)
            option("Attach", "Use it here. This app never stops a server it did not start.", suggested: true)
            option("Use another port", "Change the port in Settings, then update the port in your apps.", suggested: false)
            HStack {
                Spacer()
                Button("Change Port…") { model.openSettings() }.buttonStyle(.axSecondary)
                Button("Attach") { Task { await runtime.attach(port: model.settings.port, apiKey: "") } }
                    .buttonStyle(.axPrimary).disabled(runtime.busy || models.removing)
                    .help("Attach to the service on this port. If it needs a key, connect from API & Logs.")
            }
        }
    }

    private func option(_ title: String, _ detail: String, suggested: Bool) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(size: 13, weight: .semibold))
                Text(detail).font(.system(size: 11.5)).foregroundStyle(Brand.text3)
            }
            Spacer()
            if suggested { Text("Suggested").font(.system(size: 11, weight: .semibold)).foregroundStyle(Brand.accent) }
        }
        .padding(12)
        .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
    }
}

/// `12-states.svg` "Memory pressure": shown on Home and System only after
/// macOS reports Warning or Critical pressure.
struct MemoryPressureBanner: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var monitor: HostMonitor
    @ObservedObject var runtime: RuntimeController
    var showsSystemLink = true

    var body: some View {
        if let pressure = monitor.pressure, pressure == "Warning" || pressure == "Critical" {
            let critical = pressure == "Critical"
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle").foregroundStyle(critical ? Brand.danger : Brand.warn)
                    Text(critical ? "Memory pressure is critical" : "Memory pressure is high").font(.system(size: 14, weight: .semibold))
                    Spacer()
                    Tag(pressure, tint: critical ? Brand.danger : Brand.warn)
                }
                Text("Replies may slow down, and loading a larger model may fail. Free memory or serve a smaller model.")
                    .font(.system(size: 12.5)).foregroundStyle(Brand.text2)
                let report = ModelMemoryReport(metrics: runtime.metrics)
                if let engine = Format.bytes(report.total) {
                    Text("AX Engine reports \(engine) of MLX memory.").font(.system(size: 11.5)).foregroundStyle(Brand.text3)
                }
                HStack(spacing: 8) {
                    Button("Choose a Smaller Model") { model.modelsSection = "library"; model.page = .models }.buttonStyle(.axSecondary)
                    if showsSystemLink { Button("See memory on System") { model.page = .system }.buttonStyle(.axLink) }
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background((critical ? Brand.danger : Brand.warn).opacity(0.10), in: RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Brand.Radii.l, style: .continuous).strokeBorder((critical ? Brand.danger : Brand.warn).opacity(0.3)))
        }
    }
}
