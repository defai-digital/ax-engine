import SwiftUI
import AppKit
import AXEngineAppCore

struct SettingsView: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject private var runtime: RuntimeController
    @ObservedObject private var models: ModelStore
    @State private var draft: AppSettings
    init(model: ApplicationModel) {
        self.model = model
        _draft = State(initialValue: model.settings)
        _runtime = ObservedObject(wrappedValue: model.runtime)
        _models = ObservedObject(wrappedValue: model.models)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                if let logo = Brand.logo { Image(nsImage: logo).resizable().frame(width: 56, height: 56) }
                VStack(alignment: .leading) { Text("AX Engine").font(.title2.bold()); Text("Native inference settings").foregroundStyle(.secondary) }
            }
            if let error = model.settingsError { ErrorBanner(text: error) }
            Form {
                Section("Runtime") {
                    HStack {
                        TextField("Executable (automatic if empty)", text: $draft.executable)
                        Button("Choose…") { choose(directory: false) }
                    }
                    HStack {
                        TextField("Download folder", text: $draft.modelsDirectory)
                        Button("Choose…") { choose(directory: true) }
                    }
                    TextField("Local port", value: $draft.port, format: .number.grouping(.never))
                }
                Section("Generation") {
                    TextField("Maximum output tokens", value: $draft.maxTokens, format: .number.grouping(.never))
                    Slider(value: $draft.temperature, in: 0...2, step: 0.05) { Text("Temperature \(draft.temperature, specifier: "%.2f")") }
                    Text("Temperature is sent only when the running model advertises support. Model memory admission and MTP policy remain in AX Engine.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }.formStyle(.grouped).frame(height: 310)
                .disabled(runtime.owned || runtime.connection != nil || runtime.busy || models.downloading || models.removing)
            HStack {
                Text("Stop or disconnect the server and downloads before changing settings.").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Save") {
                    do {
                        try draft.save()
                        model.settings = draft; model.settingsError = nil
                        Task { await models.refresh(settings: draft) }
                    } catch { model.settingsError = error.localizedDescription }
                }.buttonStyle(.borderedProminent)
                    .disabled(runtime.owned || runtime.connection != nil || runtime.busy || models.downloading || models.removing)
            }
            Divider()
            Text("Powered by MTPLX").font(.headline)
            Link("MTPLX by Youssof Altoukhi · Apache-2.0", destination: URL(string: "https://github.com/youssofal/mtplx")!)
            Button("View Third-Party Notices") {
                if let url = Bundle.main.url(forResource: "MTPLX-NOTICE", withExtension: "txt") { NSWorkspace.shared.open(url) }
            }
            Text("Selected native app components are adapted from MTPLX. Inference is provided by AX Engine. No telemetry is sent by this app.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
    private func choose(directory: Bool) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = directory; panel.canChooseFiles = !directory
        if panel.runModal() == .OK, let url = panel.url {
            if directory { draft.modelsDirectory = url.path } else { draft.executable = url.path }
        }
    }
}
