import SwiftUI
import AppKit
import AXEngineAppCore

/// Tabs follow `.internal/ui/13-settings.svg`.
private enum SettingsPane: String, Hashable {
    case general, server, models, playground, advanced
    /// Panes with fields that wait for Save.
    var editable: Bool { self != .general }
}

struct SettingsView: View {
    // QA launch value (source builds only) picks the first tab for screenshots.
    @State private var pane: SettingsPane = QALaunch.enabled
        ? SettingsPane(rawValue: UserDefaults.standard.string(forKey: "AXSettingsPane") ?? "") ?? .server : .server
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
            TabView(selection: $pane) {
                Tab("General", systemImage: "gearshape", value: SettingsPane.general) {
                    settingsForm {
                        Section("Running in the background") {
                            Toggle("Show AX Engine in the menu bar", isOn: $model.showsMenuBarItem)
                            Text("The menu bar shows live speed and lets you start, stop, switch models and copy the URL without the window. Quit offers to keep serving there.")
                                .font(.caption).foregroundStyle(Brand.text3)
                        }
                        AppUpdatesSettings(updates: model.updates)
                        privacySection
                    }
                }
                Tab("Server", systemImage: "powerplug", value: SettingsPane.server) {
                    settingsForm {
                        networkSection
                        servingSection
                    }
                }
                Tab("Models", systemImage: "shippingbox", value: SettingsPane.models) {
                    settingsForm { storageSection }
                }
                Tab("Playground", systemImage: "bubble.left", value: SettingsPane.playground) {
                    settingsForm { generationSection }
                }
                Tab("Advanced", systemImage: "terminal", value: SettingsPane.advanced) {
                    settingsForm {
                        runtimeSection
                        Section("About") {
                            LabeledContent("AX Engine", value: HostInfo.version ?? "development")
                            LabeledContent("This Mac", value: HostInfo.chip + " · " + HostInfo.memory + " · macOS " + HostInfo.macOS)
                        }
                        Section("Open-source licenses") {
                            Text("AX Engine uses selected open-source components. These acknowledgements identify their authors and licenses.")
                            ThirdPartyCreditsView()
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .onChange(of: model.settingsDraft) { _, _ in copiedDefaults = false }
            .onAppear { if model.settingsDraft.effectiveProfile == .custom { advancedGeneration = true } }
            if pane.editable {
                Divider()
                settingsFooter
            } else if model.hasSettingsChanges {
                Divider()
                HStack {
                    Text("Unsaved changes on another tab").foregroundStyle(Brand.text3)
                    Spacer()
                    Button("Review Changes") { pane = changedPane }
                }.padding(26)
            }
        }
        .background(Brand.canvas)
    }

    /// The first tab with an unsaved field.
    private var changedPane: SettingsPane {
        let draft = model.settingsDraft, saved = model.settings
        if draft.port != saved.port { return .server }
        if draft.modelsDirectory != saved.modelsDirectory { return .models }
        if draft.executable != saved.executable { return .advanced }
        return .playground
    }

    private func settingsForm<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        Form { content() }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
            .background(Brand.canvas)
            .scrollBounceBehavior(.basedOnSize)
    }

    private var generationSection: some View {
        Section("Playground profile") {
            Picker("Profile", selection: Binding(
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
                Label("Media analysis needs a compatible model and client. Playground currently accepts text only; this profile does not enable image, audio or video generation.", systemImage: "info.circle")
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
            Text("Saved profiles apply to new requests in Playground. Other apps keep their own request parameters; copy the saved defaults to configure them.")
                .font(.caption).foregroundStyle(.secondary)
            Button(copiedDefaults ? "Copied request defaults" : "Copy Request Defaults") {
                do {
                    let json = try model.settings.requestDefaultsJSON()
                    NSPasteboard.general.clearContents()
                    copiedDefaults = NSPasteboard.general.setString(json, forType: .string)
                } catch { model.settingsError = error.localizedDescription }
            }.disabled(model.hasSettingsChanges)
            .help(model.hasSettingsChanges ? "Save or revert your changes first; this copies the saved values." : "Copy the saved request parameters as JSON")
        }
    }

    private var lockNote: some View {
        Label("Unlocks after the server stops and model operations finish.", systemImage: "lock")
            .font(.caption).foregroundStyle(Brand.text3)
    }

    private var networkSection: some View {
        Section("Network") {
            TextField("Port", value: $model.settingsDraft.port, format: .number.grouping(.never))
                .disabled(locked)
            Text(verbatim: "Apps connect to http://127.0.0.1:\(String(model.settingsDraft.port))/v1").font(.caption).foregroundStyle(Brand.text3)
            LabeledContent("Listen address") {
                Label("127.0.0.1", systemImage: "lock").font(.system(size: 12.5, design: .monospaced)).foregroundStyle(Brand.serve)
            }
            Text("Loopback only. Other devices cannot connect.").font(.caption).foregroundStyle(Brand.text3)
            if locked { lockNote }
        }
    }

    private var storageSection: some View {
        Section("Model storage") {
            HStack {
                TextField("Download folder", text: $model.settingsDraft.modelsDirectory)
                Button("Choose…") { choose(directory: true) }
            }
            .disabled(locked)
            Text("New downloads go here. The library also lists AutomatosX models in the Hugging Face cache; that cache is read-only to the app except when you remove a model explicitly.")
                .font(.caption).foregroundStyle(Brand.text3)
            if locked { lockNote }
        }
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            HStack {
                TextField("Executable (automatic if empty)", text: $model.settingsDraft.executable)
                Button("Choose…") { choose(directory: false) }
            }
            .disabled(locked)
            if let executable = try? model.settings.runtime() {
                LabeledContent("Current runtime", value: executable.path)
                    .font(.caption).foregroundStyle(Brand.text3).textSelection(.enabled)
            }
            if locked { lockNote }
        }
    }

    private var servingSection: some View {
        Section("Serving") {
            Picker("Serving mode", selection: Binding(get: { model.settings.servingMode }, set: { model.selectServingMode($0) })) {
                ForEach(ServingMode.allCases, id: \.self) { Text($0.title).tag($0) }
            }
            Text(model.settings.servingMode.summary).font(.callout).foregroundStyle(.secondary)
            Text("Command line: " + model.settings.servingMode.commandLineHint)
                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            Stepper("Concurrent requests: \(model.settings.concurrentRequests)",
                    value: Binding(get: { model.settings.concurrentRequests }, set: { model.selectConcurrentRequests($0) }),
                    in: AppSettings.concurrentRequestRange)
            Text("How many requests one server decodes at once before new ones receive 429 (`--max-concurrent-requests`). 4 keeps each request near full speed on batched models; 8 is the runtime's batch limit.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Require an API key", isOn: Binding(get: { model.settings.requireAPIKey }, set: { model.selectRequireAPIKey($0) }))
            Text(model.attachedToService
                 ? "An attached service keeps its own options. Disconnect to change the options for your own server."
                 : "Serving options save as soon as you change them, here or on Home and API & Logs, and apply at the next start or restart.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .disabled(model.attachedToService)
    }

    private var privacySection: some View {
        Section("Privacy") {
            Label("No analytics or telemetry uploads", systemImage: "hand.raised.fill")
                .font(.headline)
            Text("This app does not upload your conversations or usage analytics to AutomatosX. Playground sends requests to the AX Engine service on this Mac.")
            Text("Settings and conversations are saved locally. Use Clear in Playground to remove the saved conversation. Activity history stays in memory and resets when the app quits.")
                .foregroundStyle(.secondary)
            Text("Optional update checks connect to the AX Engine release service. Update downloads include version and platform requests, but no conversations or model data.")
                .foregroundStyle(.secondary)
            Text("Model downloads connect to Hugging Face and its delivery hosts. Downloads and external links are subject to those services’ privacy practices.")
                .foregroundStyle(.secondary)
            Button("Show local data folder") { NSWorkspace.shared.open(AppPaths.support) }
        }
    }

    private var settingsFooter: some View {
        VStack(alignment: .leading, spacing: 10) {
            if locked {
                Label("Port, download folder and runtime are locked during server or model operations. Playground changes apply to the next request.", systemImage: "lock")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                if model.hasSettingsChanges {
                    Text("Unsaved changes").foregroundStyle(.secondary)
                } else if model.settingsSaved {
                    Label("Saved", systemImage: "checkmark.circle").foregroundStyle(.green)
                } else {
                    Text("Changes on this tab apply when you save.").foregroundStyle(Brand.text3)
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

    private func choose(directory: Bool) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = directory; panel.canChooseFiles = !directory
        if panel.runModal() == .OK, let url = panel.url {
            if directory { model.settingsDraft.modelsDirectory = url.path }
            else { model.settingsDraft.executable = url.path }
        }
    }
}
