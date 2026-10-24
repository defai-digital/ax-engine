import SwiftUI
import AppKit
import AXEngineAppCore

struct ChatView: View {
    @ObservedObject var chat: ChatStore
    @ObservedObject var runtime: RuntimeController
    let settings: AppSettings
    var openServer: () -> Void
    @State private var clearConfirmation = false
    private var modelID: String { runtime.models.contains(where: { $0.id == chat.selectedModel }) ? chat.selectedModel : runtime.models.first?.id ?? "" }
    private var supportsTemperature: Bool { runtime.models.first(where: { $0.id == modelID })?.capabilities?.temperature == true }
    private var canSend: Bool { !chat.generating && runtime.ready && !chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Picker("Model", selection: $chat.selectedModel) {
                    Text(runtime.models.isEmpty ? "No running model" : "Default model").tag("")
                    ForEach(runtime.models) { Text($0.id).tag($0.id) }
                }.frame(maxWidth: 450).disabled(chat.generating)
                Spacer()
                Toggle("Follow output", isOn: $chat.followOutput).toggleStyle(.checkbox)
                Button("Clear", systemImage: "trash") { clearConfirmation = true }.disabled(chat.generating || chat.messages.isEmpty)
            }.padding(.horizontal, 26).padding(.vertical, 14)
            if let error = chat.error { ErrorBanner(text: error).padding(.horizontal, 26) }
            if !runtime.ready {
                HStack {
                    ErrorBanner(text: runtime.error ?? "Start or connect to a server to send messages.")
                    Button("Review Server") { openServer() }
                }.padding(.horizontal, 26)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 20) {
                        if chat.messages.isEmpty {
                            ContentUnavailableView("A conversation, on your Mac", systemImage: "bubble.left.and.bubble.right",
                                description: Text("Your chat history stays in your local application support folder."))
                                .frame(maxWidth: .infinity).padding(.vertical, 45)
                        }
                        ForEach(chat.messages) { message in
                            VStack(alignment: .leading, spacing: 10) {
                                HStack {
                                    Label(message.role == "user" ? "You" : "AX Engine", systemImage: message.role == "user" ? "person.crop.circle" : "sparkle")
                                        .font(.headline)
                                    Spacer()
                                    if message.state != "Complete" { Text(message.state).font(.caption).foregroundStyle(.secondary) }
                                    Button { copyText(message.content) } label: { Image(systemName: "doc.on.doc") }
                                        .buttonStyle(.plain).help("Copy message").accessibilityLabel("Copy message")
                                }
                                if !message.reasoning.isEmpty {
                                    DisclosureGroup("Reasoning") { Text(message.reasoning).font(.callout).foregroundStyle(.secondary).textSelection(.enabled) }
                                }
                                if message.content.isEmpty && message.state == "Streaming" { ProgressView("Waiting for the model…") }
                                else if message.state == "Streaming" { Text(message.content).textSelection(.enabled) }
                                else { MarkdownMessage(text: message.content) }
                                if message.role == "assistant" {
                                    ChatSpeedSummary(message: message)
                                }
                            }
                            .padding(18)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(message.role == "user" ? Brand.accentChrome.opacity(0.06) : Brand.cardSurface,
                                        in: RoundedRectangle(cornerRadius: 14))
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }.padding(26)
                }
                .onChange(of: chat.messages.last?.content) { _, _ in if chat.followOutput { proxy.scrollTo("bottom", anchor: .bottom) } }
            }
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                TextField("Message your model…", text: $chat.draft, axis: .vertical)
                    .onSubmit { send() }
                    .onKeyPress(.return, phases: .down) { key in
                        // Keep native selection, undo and IME handling. Let
                        // unmodified Return reach the field's submit action.
                        guard key.modifiers.contains(.shift),
                              let editor = NSApp.keyWindow?.firstResponder as? NSTextView,
                              !editor.hasMarkedText() else { return .ignored }
                        editor.insertNewlineIgnoringFieldEditor(nil)
                        return .handled
                    }
                    .lineLimit(2...7).textFieldStyle(.plain).padding(12)
                    .background(Brand.cardSurface, in: RoundedRectangle(cornerRadius: 10))
                    .disabled(chat.generating)
                Text("Enter to send · Shift+Enter for a new line")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    if let seconds = chat.firstTokenSeconds { Text(String(format: "First text %.2fs", seconds)).font(.caption).foregroundStyle(.secondary) }
                    if let seconds = chat.elapsedSeconds { Text(String(format: "Elapsed %.1fs", seconds)).font(.caption).foregroundStyle(.secondary) }
                    Spacer()
                    Button("Retry", systemImage: "arrow.clockwise") {
                        if let connection = runtime.connection { chat.retry(model: modelID, connection: connection, settings: settings, supportsTemperature: supportsTemperature) }
                    }.disabled(chat.generating || chat.messages.isEmpty || !runtime.ready)
                    if chat.generating { Button("Stop", systemImage: "stop.fill") { Task { await chat.stop() } } }
                    else { Button("Send", systemImage: "arrow.up") { send() }.buttonStyle(.borderedProminent)
                            .keyboardShortcut(.return, modifiers: .command)
                            .disabled(!canSend) }
                }
            }.padding(20)
        }
        .onChange(of: runtime.models.map(\.id)) { _, ids in
            if !ids.contains(chat.selectedModel) { chat.selectedModel = "" }
        }
        .alert("Clear this conversation?", isPresented: $clearConfirmation) {
            Button("Clear", role: .destructive) { chat.clear() }
            Button("Cancel", role: .cancel) {}
        } message: { Text("This removes the saved local conversation.") }
    }
    private func send() {
        guard canSend, let connection = runtime.connection else { return }
        chat.send(chat.draft, model: modelID, connection: connection, settings: settings, supportsTemperature: supportsTemperature)
        if chat.generating { chat.draft = "" }
    }
}

private struct ChatSpeedSummary: View {
    let message: ChatMessage
    private var performance: ChatPerformance? {
        message.state == "Complete" ? message.performance : nil
    }
    private func rate(_ value: Double?) -> String {
        value.map { String(format: "%.1f tok/s", $0) } ?? "—"
    }
    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 16) { readings }
            VStack(alignment: .leading, spacing: 6) { readings }
        }
        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
        .accessibilityElement(children: .combine)
        .help("Native evaluation averages for this reply. Prefill counts uncached prompt tokens. Decode excludes the first output batch. These exclude network/display time; first-text latency and elapsed time are shown separately. A dash means no measurement was reported, not zero speed.")
    }
    @ViewBuilder private var readings: some View {
        Text("Prefill \(performance?.promptFullyCached == true ? "Cached" : rate(performance?.prefillTokensPerSecond))")
        Text("Decode \(rate(performance?.decodeTokensPerSecond))")
        if message.state == "Streaming" {
            Text("Measuring · rates shown when complete")
        } else if performance?.prefillTokensPerSecond != nil
                    || performance?.decodeTokensPerSecond != nil
                    || performance?.promptFullyCached == true {
            Text("Model average")
        } else {
            Text("Timing unavailable")
        }
    }
}

// Settled replies receive Markdown formatting; live replies stay plain text.
// This follows MTPLX's streaming-cost boundary without executing generated HTML.
private struct MarkdownMessage: View {
    let text: String
    var body: some View {
        let parts = text.components(separatedBy: "```")
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(parts.enumerated()), id: \.offset) { index, part in
                if index % 2 == 1 {
                    ScrollView(.horizontal) { Text(part).font(.system(.callout, design: .monospaced)).textSelection(.enabled).padding(12) }
                        .background(.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                } else {
                    Text((try? AttributedString(markdown: part, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(part))
                        .textSelection(.enabled)
                }
            }
        }
        .environment(\.openURL, OpenURLAction { url in
            guard ["https", "http"].contains(url.scheme?.lowercased() ?? "") else { return .discarded }
            return .systemAction
        })
    }
}
