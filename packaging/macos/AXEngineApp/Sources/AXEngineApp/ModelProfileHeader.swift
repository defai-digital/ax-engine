import SwiftUI
import AXEngineAppCore

/// Playground's context bar: the running model and the request profile,
/// which only shapes Playground requests. Server-level options (serving mode,
/// request cap) live in the server card on Dashboard and Server Settings.
struct ModelProfileHeader: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var chat: ChatStore


    var body: some View {
        Card {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Label(runtime.ready ? "Running model" : "No model running", systemImage: "cpu")
                    .font(.caption).foregroundStyle(.secondary)
                if runtime.ready {
                    if runtime.models.count > 1 {
                        Picker("Running model", selection: $chat.selectedModel) {
                            Text("Default model").tag("")
                            ForEach(runtime.models) { Text($0.id).tag($0.id) }
                        }.labelsHidden().disabled(chat.generating)
                    } else {
                        Text(runtime.models.map(\.id).joined(separator: " · "))
                            .font(.callout.weight(.medium)).lineLimit(2)
                            .textSelection(.enabled)
                    }
                }
                Spacer(minLength: 0)
                if runtime.owned {
                    PillBadge(text: runtime.servingMode.badge, systemImage: "bolt")
                        .help("Speculative-decode posture the running app-owned server was started with.")
                    PillBadge(text: "Up to \(runtime.concurrentRequests) requests", systemImage: "person.2")
                        .help("Requests the running server decodes at once before new ones receive 429.")
                }
                PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : "circle",
                          tint: runtime.ready ? Brand.accentChrome : .secondary)
            }
            selectorRow("Playground profile", chips: profiles)
            Text(guidance).font(.caption).foregroundStyle(.secondary)
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

    private var guidance: String {
        if chat.generating { return "Finish or stop the current reply before changing its profile." }
        if model.hasSettingsChanges { return "You have unsaved Settings changes. Choose a profile to review them first." }
        if model.settings.effectiveProfile == .media {
            return "Media sets text response parameters. Playground accepts text only; your model and client determine media support."
        }
        return "Applies to new Playground requests. Other clients use their own parameters. Custom opens Settings."
    }
}

/// Serving mode chips and the request cap. Both save at once and apply at the
/// next start or restart. Shared by the Dashboard card and Server Settings.
struct ServingOptions: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    private var attached: Bool { runtime.connection != nil && !runtime.owned }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            row("Serving mode") {
                HStack(spacing: 6) {
                    ForEach(ServingMode.allCases, id: \.self) { mode in
                        SelectionChip(title: mode.title, selected: model.settings.servingMode == mode,
                                      accessibilityLabel: mode.title + " serving mode", help: mode.summary) {
                            model.selectServingMode(mode)
                        }.disabled(attached)
                    }
                }
                .accessibilityElement(children: .contain)
                .accessibilityLabel("Serving mode")
            }
            Text(attached ? "An attached service keeps the serving mode it was started with."
                          : model.settings.servingMode.summary + " Applies when the server starts and only tunes speculative drafting.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            row("Concurrent requests") {
                HStack(spacing: 12) {
                    Stepper(value: Binding(get: { model.settings.concurrentRequests }, set: { model.selectConcurrentRequests($0) }),
                            in: AppSettings.concurrentRequestRange) {
                        Text("\(model.settings.concurrentRequests)").font(.callout.weight(.medium).monospacedDigit()).frame(minWidth: 18)
                    }
                    .disabled(attached)
                    .accessibilityLabel("Concurrent requests")
                    if let batchHeadroom { Text(batchHeadroom).font(.caption).foregroundStyle(.secondary) }
                }
            }
            Text(attached ? "An attached service keeps its own request limits."
                          : "Requests one server decodes at once; extra requests receive 429 until a slot frees. 4 keeps each request near full speed; 8 is the runtime's batch limit.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func row<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) {
                Text(title).font(.caption).foregroundStyle(.secondary).frame(width: 124, alignment: .leading)
                content()
                Spacer(minLength: 0)
            }
            VStack(alignment: .leading, spacing: 8) {
                Text(title).font(.caption).foregroundStyle(.secondary)
                content()
            }
        }
    }

    /// Live batch headroom from the runtime's own gauges, when it exports them.
    private var batchHeadroom: String? {
        guard runtime.ready, let capacity = runtime.metrics["ax_runtime_max_batch_size"], capacity > 0 else { return nil }
        let scheduled = runtime.metrics["ax_engine_step_scheduled_requests"].map { Int($0) } ?? 0
        return "\(scheduled) of \(Int(capacity)) batch slots in use"
    }
}
