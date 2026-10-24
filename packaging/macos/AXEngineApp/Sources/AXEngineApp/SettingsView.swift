import SwiftUI
import AppKit
import AXEngineAppCore

struct SettingsView: View {
    @Environment(\.openWindow) private var openWindow
    @State private var advancedGeneration = false
    @State private var copiedDefaults = false
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
    private var runtimeChanged: Bool { model.settingsDraft.hasRuntimeChanges(comparedTo: model.settings) }
    var body: some View {
        VStack(spacing: 0) {
            if let error = model.settingsError { ErrorBanner(text: error).padding([.horizontal, .top], 26) }
            Form {
                Section("Usage profile") {
                    Picker("Use AX Engine for", selection: Binding(
                        get: { model.settingsDraft.effectiveProfile },
                        set: { model.settingsDraft.apply($0) }
                    )) {
                        ForEach(GenerationProfile.allCases, id: \.self) { profile in
                            Text(profile.title).tag(profile)
                        }
                    }
                    Text(model.settingsDraft.effectiveProfile.summary)
                        .font(.callout).foregroundStyle(.secondary)
                    LabeledContent("Response limit", value: "\(model.settingsDraft.maxTokens.formatted()) tokens")
                    LabeledContent("Sampling", value: model.settingsDraft.useRuntimeSampling
                        ? "Runtime defaults" : "Custom temperature")
                    if model.settingsDraft.generationProfile == .media {
                        Label("Media analysis needs a compatible model and client. The built-in Chat currently accepts text only; this profile does not enable image, audio or video generation.", systemImage: "info.circle")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    DisclosureGroup("Advanced generation", isExpanded: $advancedGeneration) {
                        TextField("Maximum output tokens", value: $model.settingsDraft.maxTokens, format: .number.grouping(.never))
                        Text("This is a response ceiling, not a target or a context-window size. Reasoning may use part of it; increase it if replies are cut short.")
                            .font(.caption).foregroundStyle(.secondary)
                        Toggle("Use runtime sampling defaults", isOn: $model.settingsDraft.useRuntimeSampling)
                        if !model.settingsDraft.useRuntimeSampling {
                            Slider(value: $model.settingsDraft.temperature, in: 0...2, step: 0.05) {
                                Text("Temperature \(model.settingsDraft.temperature, specifier: "%.2f")")
                            }
                        }
                        Text("Runtime defaults omit sampling overrides. A custom temperature is sent only when the model supports it. Context limits, memory admission, thinking and MTP remain controlled by the model and runtime.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Text("Saved profiles apply to new requests in the built-in Chat. Other apps keep their own request parameters; copy the saved defaults to configure them.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button(copiedDefaults ? "Copied request defaults" : "Copy saved request defaults") {
                        do {
                            let json = try model.settings.requestDefaultsJSON()
                            NSPasteboard.general.clearContents()
                            copiedDefaults = NSPasteboard.general.setString(json, forType: .string)
                        } catch { model.settingsError = error.localizedDescription }
                    }.disabled(model.hasSettingsChanges)
                }
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
                Section("Privacy and licenses") {
                    Text("Your settings and conversation are stored on this Mac. No telemetry is sent by this app.")
                        .font(.callout).foregroundStyle(.secondary)
                    Button("Third-Party Notices") { openWindow(id: "third-party-notices") }
                }
            }
            .onChange(of: model.settingsDraft) { _, _ in copiedDefaults = false }
            .formStyle(.grouped)
            .frame(maxWidth: 860)
            .frame(maxWidth: .infinity, alignment: .leading)
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                if locked {
                    Label("Runtime locations and port are locked during server or model operations. Profile changes apply to the next Chat request.", systemImage: "lock")
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
                        .disabled(!model.hasSettingsChanges)
                    Button("Save") {
                        let refreshRuntime = runtimeChanged
                        guard !locked || !refreshRuntime else { return }
                        if model.saveSettings(), refreshRuntime {
                            let settings = model.settings
                            Task { await models.refresh(settings: settings) }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled((locked && runtimeChanged) || !model.hasSettingsChanges)
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
