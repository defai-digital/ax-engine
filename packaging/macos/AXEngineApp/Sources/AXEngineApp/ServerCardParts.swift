import SwiftUI
import AXEngineAppCore

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
