import SwiftUI
import AXEngineAppCore

/// "Connect an app": ready-to-paste setups for the local endpoint. Shown on
/// Home and API & Logs. Snippets reference $AX_ENGINE_API_KEY and never
/// contain a key.
struct ConnectAppPanel: View {
    @ObservedObject var runtime: RuntimeController
    let port: Int
    /// The saved API-key setting, used before a server is listening.
    var keyWhenStopped = false
    @AppStorage("connectSnippet") private var snippetID = ClientSnippet.claudeCode.rawValue
    @State private var copied = false

    private var snippet: ClientSnippet { ClientSnippet(rawValue: snippetID) ?? .claudeCode }
    private var url: String { runtime.connection?.apiURL.absoluteString ?? "http://127.0.0.1:\(port)/v1" }
    private var requiresKey: Bool { runtime.connection?.hasAPIKey ?? keyWhenStopped }

    var body: some View {
        let text = snippet.text(apiURL: url, model: runtime.models.first?.id ?? "", requiresKey: requiresKey)
        Card("Connect an app", trailing: {
            Text(runtime.connection == nil ? "Start the server to connect" : "OpenAI and Anthropic compatible")
                .font(Brand.Font.caption).foregroundStyle(Brand.text3)
        }) {
            AXSegmented(options: ClientSnippet.allCases.map { ($0.rawValue, $0.title) }, selection: $snippetID, compact: true)
            VStack(alignment: .trailing, spacing: 10) {
                ScrollView(.horizontal, showsIndicators: false) {
                    Text(highlighted(text)).font(Brand.Font.code).lineSpacing(5).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                Button(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc") {
                    copyText(text); copied = true
                    Task { try? await Task.sleep(for: .seconds(2)); copied = false }
                }
                .buttonStyle(.ax(.primary))
            }
            .padding(16)
            .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous).strokeBorder(Brand.hairlineColor))
            Text(requiresKey ? "Copies with your current port and model. Set AX_ENGINE_API_KEY from Copy API Key; snippets never include the key."
                             : "Copies with your current port and model. Never includes a stored key.")
                .font(Brand.Font.caption).foregroundStyle(Brand.text3)
            Label("Only apps on this Mac can connect (127.0.0.1).", systemImage: "lock")
                .font(Brand.Font.caption).foregroundStyle(Brand.text3)
        }
    }

    /// Comments dim; `export` and shell/SDK keywords take the accent.
    private func highlighted(_ text: String) -> AttributedString {
        var result = AttributedString()
        let keywords = ["export ", "from ", "import ", "const ", "await ", "curl "]
        for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            if index > 0 { result += AttributedString("\n") }
            var piece = AttributedString(String(line))
            if line.hasPrefix("#") {
                piece.foregroundColor = Brand.text3
            } else if let keyword = keywords.first(where: { line.hasPrefix($0) }),
                      let range = piece.range(of: keyword) {
                piece.foregroundColor = Brand.text1
                piece[range].foregroundColor = Brand.accent
            } else {
                piece.foregroundColor = Brand.text1
            }
            result += piece
        }
        return result
    }
}
