import SwiftUI
import AppKit
import AXEngineAppCore

struct SettingsView: View {
    @Environment(\.openWindow) private var openWindow
    @ObservedObject var model: ApplicationModel
    @ObservedObject private var runtime: RuntimeController
    @ObservedObject private var models: ModelStore
    init(model: ApplicationModel) {
        self.model = model
        _runtime = ObservedObject(wrappedValue: model.runtime)
        _models = ObservedObject(wrappedValue: model.models)
    }
    private var locked: Bool {
        runtime.owned || runtime.connection != nil || runtime.busy || models.downloading || models.removing || models.refreshing
    }
    var body: some View {
        VStack(spacing: 0) {
            if let error = model.settingsError { ErrorBanner(text: error).padding([.horizontal, .top], 26) }
            Form {
                Section("Runtime") {
                    HStack {
                        TextField("Executable (automatic if empty)", text: $model.settingsDraft.executable)
                        Button("Choose…") { choose(directory: false) }
                    }
                    HStack {
                        TextField("Download folder", text: $model.settingsDraft.modelsDirectory)
                        Button("Choose…") { choose(directory: true) }
                    }
                    TextField("Local port", value: $model.settingsDraft.port, format: .number.grouping(.never))
                    if let executable = try? model.settings.runtime() {
                        LabeledContent("Current runtime", value: executable.path)
                            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                }.disabled(locked)
                Section("Generation") {
                    TextField("Maximum output tokens", value: $model.settingsDraft.maxTokens, format: .number.grouping(.never))
                    Slider(value: $model.settingsDraft.temperature, in: 0...2, step: 0.05) {
                        Text("Temperature \(model.settingsDraft.temperature, specifier: "%.2f")")
                    }
                    Text("Temperature is sent only when the running model advertises support. Model memory admission and MTP policy remain in AX Engine.")
                        .font(.caption).foregroundStyle(.secondary)
                }.disabled(locked)
                Section("Privacy and licenses") {
                    Text("Your settings and conversation are stored on this Mac. No telemetry is sent by this app.")
                        .font(.callout).foregroundStyle(.secondary)
                    Button("Third-Party Notices") { openWindow(id: "third-party-notices") }
                }
            }
            .formStyle(.grouped)
            .frame(maxWidth: 860)
            .frame(maxWidth: .infinity, alignment: .leading)
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                if locked {
                    Label(models.refreshing ? "Refreshing the model library…" : "Stop or disconnect the server and model operations before changing settings.", systemImage: "lock")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    if model.hasSettingsChanges {
                        Text("Unsaved changes").foregroundStyle(.secondary)
                    } else if model.settingsSaved {
                        Label("Saved", systemImage: "checkmark.circle").foregroundStyle(.green)
                    } else {
                        Text("Changes apply when you save.").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Revert") { model.revertSettings() }
                        .disabled(locked || !model.hasSettingsChanges)
                    Button("Save") {
                        if model.saveSettings() {
                            let settings = model.settings
                            Task { await models.refresh(settings: settings) }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(locked)
                }
            }.padding(26)
        }
    }
    private func choose(directory: Bool) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = directory; panel.canChooseFiles = !directory
        if panel.runModal() == .OK, let url = panel.url {
            if directory { model.settingsDraft.modelsDirectory = url.path }
            else { model.settingsDraft.executable = url.path }
        }
    }
}
