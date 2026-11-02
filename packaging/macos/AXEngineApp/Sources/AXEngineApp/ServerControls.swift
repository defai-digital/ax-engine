import SwiftUI
import AXEngineAppCore

/// Installed-model picker plus Start / Restart / Stop for the app-owned
/// server. Shared by Dashboard and Server so both pages drive the same
/// lifecycle with the same gates and confirmations.
struct ServerControls: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var models: ModelStore
    @ObservedObject var chat: ChatStore
    /// Show the selected model's folder under the picker.
    var showsPath = true
    @State private var stopConfirmation = false
    @State private var restartTarget: String?
    @State private var pickerOpen = false

    private var attached: Bool { runtime.connection != nil && !runtime.owned }
    private var selectable: [LocalModel] { models.productLibrary.filter { $0.ready && models.checkedPaths.contains($0.path) } }
    private var startBlockReason: String? {
        ServerGate.startBlockReason(.init(selectedPath: model.selectedPath, selectable: models.canSelectInGUI(model.selectedPath),
                                          libraryEmpty: selectable.isEmpty, scanning: models.scanning,
                                          runtimeBusy: runtime.busy, removing: models.removing, attached: attached))
    }
    private var inferenceActive: Bool {
        chat.generating || (runtime.metrics["ax_engine_generation_active_streams"] ?? 0) > 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            modelSelection
            actions
        }
        .onAppear(perform: preselect)
        .onChange(of: selectable.map(\.path)) { _, _ in preselect() }
        .onChange(of: models.scanning) { _, _ in preselect() }
        .alert("Stop \(runningTitle)?", isPresented: $stopConfirmation) {
            Button("Stop", role: .destructive) { Task { await chat.stop(); await runtime.stop() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("A reply is being generated. Stopping the app-owned server ends it. You can start the server again at any time.") }
        .alert("Restart \(runningTitle)?", isPresented: Binding(get: { restartTarget != nil }, set: { if !$0 { restartTarget = nil } })) {
            Button("Restart", role: .destructive) {
                if let target = restartTarget { restart(target) }
                restartTarget = nil
            }
            Button("Cancel", role: .cancel) { restartTarget = nil }
        } message: {
            Text(model.settings.requireAPIKey
                 ? "A reply is being generated. Restarting ends it; the server comes back with a new API key."
                 : "A reply is being generated. Restarting ends it.")
        }
    }

    /// Keep a startable model selected so Start works in one click: the
    /// running model while the app owns a server, otherwise the most recently
    /// served, a favorite, or the first downloaded model. An explicit valid
    /// choice is never replaced.
    private func preselect() {
        guard !models.scanning, model.selectedPath.isEmpty || !models.canSelectInGUI(model.selectedPath) else { return }
        if runtime.owned, models.canSelectInGUI(runtime.selectedModel) { model.selectedPath = runtime.selectedModel }
        else if let preferred = ModelPicker.preferred(models: selectable, favorites: model.favoriteModels, recent: model.recentModels) {
            model.selectedPath = preferred.path
        }
    }

    private func start(_ path: String) {
        model.recordRecentModel(path)
        runtime.start(model: path, settings: model.settings)
    }

    private func restart(_ path: String) {
        model.recordRecentModel(path)
        Task { await chat.stop(); await runtime.restart(model: path, settings: model.settings) }
    }

    private var modelSelection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Model").font(.caption).foregroundStyle(.secondary)
                Button { pickerOpen.toggle() } label: {
                    HStack(spacing: 6) {
                        if model.favoriteModels.contains(model.selectedPath) {
                            Image(systemName: "star.fill").foregroundStyle(Brand.accentChrome)
                        }
                        if let selectedModel {
                            ModelTitle(label: selectedModel.label)
                            if let size = selectedModel.sizeBytes {
                                Text(TransferFormat.bytes(size)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                        } else {
                            Text(selectedTitle).lineLimit(1).truncationMode(.middle)
                        }
                        Image(systemName: "chevron.up.chevron.down").font(.caption).foregroundStyle(.secondary)
                    }
                }
                .disabled(attached).frame(maxWidth: 420, alignment: .leading)
                .help(selectedModel.map { $0.name + "\n" + $0.path } ?? "Choose a downloaded model")
                .popover(isPresented: $pickerOpen, arrowEdge: .bottom) {
                    ModelPickerPopover(models: selectable, selectedPath: model.selectedPath,
                                       favorites: model.favoriteModels, recent: model.recentModels, scanning: models.scanning,
                                       choose: { model.selectedPath = $0; pickerOpen = false },
                                       toggleFavorite: { model.toggleFavoriteModel($0) },
                                       browseCatalog: { pickerOpen = false; model.modelsSection = "catalog"; model.page = .models })
                }
                    .accessibilityLabel("Model to serve")
                if !model.selectedPath.isEmpty {
                    Button("Show in Library") { model.modelSearch = selectedModel?.repositoryName ?? ""; model.modelsSection = "library"; model.page = .models }
                }
            }
            if showsPath && !model.selectedPath.isEmpty {
                Text(model.selectedPath).font(.caption).foregroundStyle(.secondary).textSelection(.enabled).lineLimit(1).truncationMode(.middle)
            }
        }
    }

    private var selectedModel: LocalModel? { models.productLibrary.first { $0.path == model.selectedPath } }
    private var runningTitle: String {
        models.productLibrary.first { $0.path == runtime.selectedModel }?.displayName
            ?? (runtime.selectedModel.isEmpty ? "the local server" : URL(fileURLWithPath: runtime.selectedModel).lastPathComponent)
    }
    private var selectedTitle: String {
        if let selectedModel { return menuTitle(selectedModel) }
        if model.selectedPath.isEmpty { return selectable.isEmpty ? "No downloaded models" : "Choose a downloaded model" }
        return URL(fileURLWithPath: model.selectedPath).lastPathComponent
    }
    private func menuTitle(_ item: LocalModel) -> String {
        item.displayName + (item.sizeBytes.map { " · " + TransferFormat.bytes($0) } ?? "")
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                if runtime.owned {
                    let switching = !model.selectedPath.isEmpty && model.selectedPath != runtime.selectedModel
                    let switchingMode = model.settings.servingMode != runtime.servingMode
                    let switchingCap = model.settings.concurrentRequests != runtime.concurrentRequests
                    Button(ServerGate.restartTitle(modelChanged: switching, modeChanged: switchingMode, capChanged: switchingCap,
                                                   mode: model.settings.servingMode),
                           systemImage: "arrow.clockwise") {
                        let target = switching && models.canSelectInGUI(model.selectedPath) ? model.selectedPath : runtime.selectedModel
                        if inferenceActive { restartTarget = target }
                        else { restart(target) }
                    }
                    .disabled(runtime.busy || runtime.state == "Stopping" || models.removing
                              || (switching && !models.canSelectInGUI(model.selectedPath)))
                    .help(switching || switchingMode || switchingCap
                          ? "Stop this server and start it again with the selected model, serving mode and request cap. Active inference stops."
                          : "Stop this server and start it again. Active inference stops.")
                    Button(runtime.state == "Starting" ? "Cancel Start" : "Stop Server", systemImage: "stop.fill", role: .destructive) {
                        if inferenceActive { stopConfirmation = true } else { Task { await chat.stop(); await runtime.stop() } }
                    }.disabled(runtime.state == "Stopping")
                    .help(runtime.state == "Starting" ? "Cancel the launch" : "Stop the app-owned server")
                } else if runtime.connection != nil {
                    Button("Disconnect") { Task { await chat.stop(); runtime.detach() } }
                } else {
                    Button("Start Server", systemImage: "play.fill") {
                        if models.canSelectInGUI(model.selectedPath) { start(model.selectedPath) }
                    }
                    .buttonStyle(.borderedProminent).disabled(startBlockReason != nil)
                    .help(startBlockReason ?? "Start a loopback server on the selected model")
                }
                Spacer()
            }
            if runtime.owned, let notice = ServerGate.pendingRestartNotice(
                modelChanged: !model.selectedPath.isEmpty && model.selectedPath != runtime.selectedModel,
                modeChanged: model.settings.servingMode != runtime.servingMode,
                running: runtime.servingMode, selected: model.settings.servingMode,
                runningCap: runtime.concurrentRequests, selectedCap: model.settings.concurrentRequests) {
                Label(notice, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
            } else if !runtime.owned, runtime.connection == nil, let reason = startBlockReason {
                Label(reason, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
