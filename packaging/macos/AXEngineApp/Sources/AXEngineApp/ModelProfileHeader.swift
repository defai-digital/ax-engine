import SwiftUI
import AXEngineAppCore

/// One shared inference context on Dashboard, Server and Playground: the
/// running model and the mode it drafts with. Dashboard and Server carry the
/// serving mode selector (a server-level setting); Playground carries the
/// request profile selector, which only shapes Playground requests.
struct ModelProfileHeader: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var chat: ChatStore

    private var attached: Bool { runtime.connection != nil && !runtime.owned }
    private var inferenceActive: Bool {
        chat.generating || (runtime.metrics["ax_engine_generation_active_streams"] ?? 0) > 0
    }
    private var modeAwaitsRestart: Bool { runtime.owned && runtime.servingMode != model.settings.servingMode }
    private var showsServingMode: Bool { model.page != .chat }

    var body: some View {
        Card {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Label(runtime.ready ? "Running model" : "No model running", systemImage: "cpu")
                    .font(.caption).foregroundStyle(.secondary)
                if runtime.ready {
                    if model.page == .chat && runtime.models.count > 1 {
                        Picker("Running model", selection: $chat.selectedModel) {
                            Text("Default model").tag("")
                            ForEach(runtime.models) { Text($0.id).tag($0.id) }
                        }.labelsHidden().disabled(chat.generating)
                    } else {
                        Text(runtime.models.map(\.id).joined(separator: " · "))
                            .font(.callout.weight(.medium)).lineLimit(2)
                            .textSelection(.enabled)
                    }
                } else if model.page == .overview {
                    Button("Open Server") { model.page = .server }.buttonStyle(.link).font(.caption)
                }
                Spacer(minLength: 0)
                if runtime.owned {
                    PillBadge(text: runtime.servingMode.badge, systemImage: "bolt")
                        .help("Speculative-decode posture the running app-owned server was started with.")
                }
                PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : "circle",
                          tint: runtime.ready ? Brand.accentChrome : .secondary)
            }
            if showsServingMode {
                selectorRow("Serving mode", chips: servingModes)
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text(servingGuidance).font(.caption).foregroundStyle(.secondary)
                    if modeAwaitsRestart {
                        Button("Restart in \(model.settings.servingMode.title) Mode", systemImage: "arrow.clockwise") {
                            let settings = model.settings
                            Task { await chat.stop(); await runtime.restart(settings: settings) }
                        }
                        .controlSize(.small)
                        .disabled(runtime.busy || runtime.state == "Stopping" || inferenceActive)
                        .help(inferenceActive ? "Finish or stop the current reply first, or restart from the Server page."
                                              : "Stop this server and start it again in the saved serving mode.")
                    }
                }
            } else {
                selectorRow("Playground profile", chips: profiles)
                Text(guidance).font(.caption).foregroundStyle(.secondary)
            }
            if let error = model.settingsError { ErrorBanner(text: error) }
        }
    }

    private func selectorRow(_ title: String, chips: some View) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) {
                Text(title).font(.caption).foregroundStyle(.secondary).frame(width: 112, alignment: .leading)
                chips
                Spacer(minLength: 0)
            }
            VStack(alignment: .leading, spacing: 8) {
                Text(title).font(.caption).foregroundStyle(.secondary)
                chips
            }
        }
    }

    private var servingModes: some View {
        HStack(spacing: 6) {
            ForEach(ServingMode.allCases, id: \.self) { mode in
                SelectionChip(title: mode.title, selected: model.settings.servingMode == mode,
                              accessibilityLabel: mode.title + " serving mode", help: mode.summary) {
                    model.selectServingMode(mode)
                }.disabled(attached || model.hasSettingsChanges)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Serving mode")
    }

    private var profiles: some View {
        HStack(spacing: 6) {
            ForEach(GenerationProfile.allCases, id: \.self) { profile in
                SelectionChip(title: profile.title, selected: model.settings.effectiveProfile == profile,
                              accessibilityLabel: profile.title + " profile",
                              help: profile == .custom ? "Edit your own generation parameters in Settings." : profile.summary) {
                    model.selectPlaygroundProfile(profile)
                }.disabled(chat.generating)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Playground request profile")
    }

    private var servingGuidance: String {
        if attached { return "An attached service keeps the serving mode it was started with." }
        if model.hasSettingsChanges { return "You have unsaved Settings changes. Save or revert them before changing the serving mode." }
        if modeAwaitsRestart {
            return "\(model.settings.servingMode.title) mode is saved for the next launch. The server is still running in \(runtime.servingMode.title) mode."
        }
        return model.settings.servingMode.summary + " Applies when the server starts and only tunes speculative drafting."
    }

    private var guidance: String {
        if chat.generating { return "Finish or stop the current reply before changing its profile." }
        if model.hasSettingsChanges { return "You have unsaved Settings changes. Choose a profile to review them first." }
        if model.settings.effectiveProfile == .media {
            return "Media sets text response parameters. Playground accepts text only; your model and client determine media support."
        }
        return "Applies to new Playground requests. Other clients use their own parameters. Custom opens Settings."
    }
}
