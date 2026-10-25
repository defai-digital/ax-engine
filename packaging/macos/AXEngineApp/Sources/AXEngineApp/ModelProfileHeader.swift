import SwiftUI
import AXEngineAppCore

/// One shared model context and saved request profile on Server and Playground.
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
                }
                Spacer(minLength: 0)
                PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : "circle",
                          tint: runtime.ready ? Brand.accentChrome : .secondary)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) {
                    Text("Playground profile").font(.caption).foregroundStyle(.secondary)
                    profiles
                    Spacer(minLength: 0)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Playground profile").font(.caption).foregroundStyle(.secondary)
                    profiles
                }
            }
            Text(guidance).font(.caption).foregroundStyle(.secondary)
            if let error = model.settingsError { ErrorBanner(text: error) }
        }
    }

    private var profiles: some View {
        HStack(spacing: 6) {
            ForEach(GenerationProfile.allCases, id: \.self) { profile in
                let selected = model.settings.effectiveProfile == profile
                Button { model.selectPlaygroundProfile(profile) } label: {
                    HStack(spacing: 5) {
                        if selected { Image(systemName: "checkmark.circle.fill") }
                        Text(profile.title)
                    }
                    .font(.callout.weight(selected ? .semibold : .regular))
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .foregroundStyle(selected ? Brand.accentChrome : Color.primary)
                    .background(selected ? Brand.accentChrome.opacity(0.14) : Color.secondary.opacity(0.08), in: Capsule())
                    .overlay(Capsule().strokeBorder(selected ? Brand.accentChrome.opacity(0.5) : Color.clear))
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain).disabled(chat.generating)
                .accessibilityLabel(profile.title + " profile")
                .accessibilityAddTraits(selected ? [.isSelected] : [])
                .help(profile == .custom ? "Edit your own generation parameters in Settings." : profile.summary)
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
