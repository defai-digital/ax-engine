import SwiftUI
import AppKit
import UniformTypeIdentifiers
import AXEngineAppCore

struct ChatView: View {
    @ObservedObject var chat: ChatStore
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var model: ApplicationModel
    private var settings: AppSettings { model.settings }
    @State private var clearConfirmation = false
    private var modelID: String { runtime.models.contains(where: { $0.id == chat.selectedModel }) ? chat.selectedModel : runtime.models.first?.id ?? "" }
    private var supportsTemperature: Bool { runtime.models.first(where: { $0.id == modelID })?.capabilities?.temperature == true }
    private var canSend: Bool { !chat.generating && !chat.recoveryRequired && runtime.ready && !chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Spacer()
                Toggle("Follow output", isOn: $chat.followOutput).toggleStyle(.checkbox)
                if chat.recoveryRequired {
                    Button("Reveal Saved Conversation", systemImage: "folder") {
                        NSWorkspace.shared.activateFileViewerSelecting([chat.storage])
                    }
                }
                Button("Export", systemImage: "square.and.arrow.up") { exportConversation() }
                    .disabled(chat.generating || chat.messages.isEmpty || chat.recoveryRequired)
                Button("Clear", systemImage: "trash") { clearConfirmation = true }.disabled(chat.generating || (chat.messages.isEmpty && !chat.recoveryRequired))
            }.padding(.horizontal, 26).padding(.vertical, 14)
            if let error = chat.error { ErrorBanner(text: error).padding(.horizontal, 26) }
            if !runtime.ready && !chat.messages.isEmpty { notServingBanner.padding(.horizontal, 26) }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 20) {
                        if !runtime.ready && chat.messages.isEmpty {
                            notServing.frame(maxWidth: .infinity).padding(.vertical, 30)
                        } else if chat.messages.isEmpty {
                            ContentUnavailableView("Try your model", systemImage: "bubble.left.and.bubble.right",
                                description: Text("Send a prompt to inspect response quality and inference speed. History stays on this Mac."))
                                .frame(maxWidth: .infinity).padding(.vertical, 45)
                        }
                        ForEach(chat.messages) { message in
                            VStack(alignment: .leading, spacing: 10) {
                                HStack {
                                    Label(message.role == "user" ? "You" : "AX Engine", systemImage: message.role == "user" ? "person.crop.circle" : "sparkle")
                                        .font(.headline)
                                    Spacer()
                                    if message.state != "Complete" { Text(message.state).font(.caption).foregroundStyle(.secondary) }
                                    CopyButton(text: message.content, label: "Copy message")
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
                TextField(placeholder, text: $chat.draft, axis: .vertical)
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
                    .disabled(chat.recoveryRequired)
                HStack(alignment: .firstTextBaseline) {
                    Text("Enter to send · Shift+Enter for a new line")
                    Spacer(minLength: 12)
                    Label("No analytics or telemetry uploads", systemImage: "lock.shield")
                        .help("Playground sends requests to the AX Engine service on this Mac. Conversations are saved locally. Model downloads use the network.")
                }
                .font(.caption).foregroundStyle(.secondary)
                HStack {
                    if let seconds = chat.firstTokenSeconds { Text(String(format: "First text %.2fs", seconds)).font(.caption).foregroundStyle(.secondary) }
                    if let seconds = chat.elapsedSeconds { Text(String(format: "Elapsed %.1fs", seconds)).font(.caption).foregroundStyle(.secondary) }
                    Spacer()
                    Button("Retry", systemImage: "arrow.clockwise") {
                        if let connection = runtime.connection { chat.retry(model: modelID, connection: connection, settings: settings, supportsTemperature: supportsTemperature) }
                    }.disabled(chat.generating || chat.messages.isEmpty || !runtime.ready)
                    if chat.generating {
                        Button("Stop", systemImage: "stop.fill") { Task { await chat.stop() } }
                            .keyboardShortcut(".", modifiers: .command).help("Stop this reply (⌘.)")
                    }
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
    private var placeholder: String {
        if chat.generating { return "Write your next message; Send is available when this reply finishes…" }
        if runtime.state == "Starting" { return "Loading the model…" }
        return runtime.ready ? "Message your model…" : "Start the server to send messages"
    }

    /// One state for "nothing to talk to", with the next step that applies.
    @ViewBuilder private var notServing: some View {
        if runtime.state == "Starting" || runtime.state == "Connecting" {
            ContentUnavailableView {
                Label("Loading the model…", systemImage: "hourglass")
            } description: {
                Text("Large models can take a few minutes from a cold start. You can type your message meanwhile.")
            }
        } else if let error = runtime.error {
            ContentUnavailableView {
                Label("The server is not available", systemImage: "exclamationmark.triangle")
            } description: { Text(error) } actions: {
                Button("Review Server") { model.page = .server }.buttonStyle(.borderedProminent)
            }
        } else if model.attachedToService {
            ContentUnavailableView {
                Label("Waiting for the connected service", systemImage: "network")
            } description: {
                Text("AX Engine is attached to a service it did not start. Playground opens once that service reports ready.")
            } actions: {
                Button("Review Server Settings") { model.page = .server }
            }
        } else if model.models.scanning && model.servableModels.isEmpty {
            ContentUnavailableView {
                Label("Checking your models…", systemImage: "magnifyingglass")
            } description: { Text("Playground can start a server once the library scan finds a downloaded model.") }
        } else if model.servableModels.isEmpty {
            ContentUnavailableView {
                Label("Download a model to try", systemImage: "arrow.down.circle")
            } description: {
                Text("Playground talks to a model served on this Mac. Download one from the Catalog first.")
            } actions: {
                Button("Download a Model") { model.catalogSearch = ""; model.modelsSection = "catalog"; model.page = .models }
                    .buttonStyle(.borderedProminent)
            }
        } else {
            ContentUnavailableView {
                Label("Start the server to try your model", systemImage: "play.circle")
            } description: {
                Text(model.models.productLibrary.first { $0.path == model.selectedPath }.map { "Starts " + $0.displayName + " on this Mac." }
                     ?? "Choose a downloaded model on Dashboard.")
            } actions: {
                if model.canStartServer {
                    Button("Start Server", systemImage: "play.fill") { model.startServer() }.buttonStyle(.borderedProminent)
                }
                Button("Choose Model") { model.page = .overview }
            }
        }
    }

    /// Compact form of the not-serving state above an existing conversation.
    private var notServingBanner: some View {
        HStack(spacing: 10) {
            Label(runtime.state == "Starting" ? "Loading the model…"
                  : runtime.error ?? (model.attachedToService ? "Waiting for the connected service to report ready."
                                     : "The server is not running. Start it to continue this conversation."),
                  systemImage: runtime.error == nil ? "info.circle" : "exclamationmark.triangle")
                .font(.callout).foregroundStyle(runtime.error == nil ? Color.secondary : Color.orange).lineLimit(2)
            Spacer()
            if runtime.error != nil {
                Button("Review Server") { model.page = .server }
            } else if model.canStartServer {
                Button("Start Server", systemImage: "play.fill") { model.startServer() }.buttonStyle(.borderedProminent)
            } else if runtime.state != "Starting" {
                Button("Open Dashboard") { model.page = .overview }
            }
        }
        .padding(10)
        .background(Brand.cardSurface, in: RoundedRectangle(cornerRadius: 10))
    }

    private func exportConversation() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "AX Engine Conversation.json"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        chat.export(to: destination)
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
                    let code = CodeFence.body(part)
                    ScrollView(.horizontal) { Text(code).font(.system(.callout, design: .monospaced)).textSelection(.enabled).padding(12) }
                        .background(.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(alignment: .topTrailing) { CopyButton(text: code, label: "Copy code").padding(8) }
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

/// Copies text and confirms with a checkmark for two seconds.
struct CopyButton: View {
    let text: String
    var label = "Copy"
    @State private var copied = false
    var body: some View {
        Button {
            copyText(text); copied = true
            Task { try? await Task.sleep(for: .seconds(2)); copied = false }
        } label: { Image(systemName: copied ? "checkmark" : "doc.on.doc") }
        .buttonStyle(.plain).help(copied ? "Copied" : label)
        .accessibilityLabel(copied ? "Copied" : label)
    }
}

enum CodeFence {
    /// The code inside a fence, without a language tag on its first line.
    static func body(_ fenced: String) -> String {
        guard let newline = fenced.firstIndex(of: "\n") else { return fenced }
        let info = String(fenced[..<newline])
        let isLanguageTag = info.count <= 24 && info.range(of: #"^[A-Za-z0-9+#._-]*$"#, options: .regularExpression) != nil
        return isLanguageTag ? String(fenced[fenced.index(after: newline)...]) : fenced
    }
}
