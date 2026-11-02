import SwiftUI
import AXEngineAppCore

/// Server state, elapsed time and the listening address. Before anything is
/// listening it says so instead of showing an address clients cannot reach.
struct ServerStatusRow: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    private var transitioning: Bool { ["Starting", "Stopping", "Connecting"].contains(runtime.state) }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                PillBadge(text: runtime.state, systemImage: runtime.ready ? "circle.fill" : transitioning ? "clock" : "circle",
                          tint: runtime.ready ? Brand.accentChrome
                              : runtime.state == "Failed" || runtime.state == "Connection degraded" ? .orange : .secondary,
                          emphasized: runtime.ready || transitioning)
                if transitioning { ProgressView().controlSize(.small) }
                if let detail = ServerGate.stateDetail(state: runtime.state, startedAt: runtime.startedAt, now: context.date) {
                    Text(detail).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
                if let connection = runtime.connection {
                    Text(connection.apiURL.absoluteString).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                } else {
                    Text(transitioning ? "Not listening yet" : "Not listening").font(.callout).foregroundStyle(.secondary)
                        .help("Clients connect to http://127.0.0.1:\(String(model.settings.port))/v1 once the server is running.")
                }
            }
            .accessibilityElement(children: .combine)
        }
        if runtime.state == "Starting" {
            VStack(alignment: .leading, spacing: 4) {
                Text("Large models can take a few minutes from a cold start. Cancel Start stops the launch.")
                    .font(.caption).foregroundStyle(.secondary)
                if let line = RuntimeController.lastDiagnostic(runtime.logs) {
                    Text(line).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(2).truncationMode(.middle)
                }
            }
        }
    }
}

/// Copy API URL and, when the server uses one, Copy API key. The same
/// buttons and wording on every page that shows a connection.
struct ClientCopyButtons: View {
    @ObservedObject var runtime: RuntimeController
    let clipboard: SensitiveClipboard
    @Binding var notice: String
    var showsPlayground = false
    var openPlayground: () -> Void = {}

    var body: some View {
        if let connection = runtime.connection {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Button("Copy API URL", systemImage: "doc.on.doc") {
                        guard let current = runtime.connection else { return }
                        copyText(current.apiURL.absoluteString)
                        notice = "API URL copied."
                        Task { try? await Task.sleep(for: .seconds(4)); if notice == "API URL copied." { notice = "" } }
                    }
                    if connection.hasAPIKey {
                        Button("Copy API Key", systemImage: "key") {
                            guard runtime.ready, let current = runtime.connection else { return }
                            notice = current.copyAPIKey(to: clipboard)
                                ? "API key copied for up to 60 seconds. Paste it into your client."
                                : "The API key could not be copied. Try again."
                        }.disabled(!runtime.ready)
                    }
                    if showsPlayground && runtime.ready {
                        Button("Open Playground", systemImage: "bubble.left") { openPlayground() }
                            .buttonStyle(.borderedProminent)
                    }
                }
                Text(notice.isEmpty
                     ? (connection.hasAPIKey
                        ? "Clients need the URL and the API key. A copied key clears from the clipboard after 60 seconds."
                        : "No API key: local clients on this Mac can call the URL directly.")
                     : notice)
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// Serving mode and request cap behind a one-line summary. They save at once
/// and apply at the next start or restart.
struct ServingOptionsDisclosure: View {
    @ObservedObject var model: ApplicationModel
    @ObservedObject var runtime: RuntimeController
    @State private var expanded = false

    private var summary: String {
        let settings = model.settings
        return settings.servingMode.badge + " · up to \(settings.concurrentRequests) requests · "
            + (settings.requireAPIKey ? "API key required" : "no API key")
    }

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            ServingOptions(model: model, runtime: runtime).padding(.top, 6)
        } label: {
            HStack(spacing: 8) {
                Text("Serving options").font(.callout)
                Text(summary).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .accessibilityLabel("Serving options, " + summary)
    }
}
