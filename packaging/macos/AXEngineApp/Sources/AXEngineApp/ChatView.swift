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
    @State private var systemPromptOpen = false
    private var modelID: String { runtime.models.contains(where: { $0.id == chat.selectedModel }) ? chat.selectedModel : runtime.models.first?.id ?? "" }
    private var supportsTemperature: Bool { runtime.models.first(where: { $0.id == modelID })?.capabilities?.temperature == true }
    private var canSend: Bool { !chat.generating && !chat.recoveryRequired && runtime.ready && !chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

    var body: some View {
        VStack(spacing: 0) {
            topBar.padding(.horizontal, 24).padding(.top, 6).padding(.bottom, 10)
            if let error = chat.error { ErrorBanner(text: error).padding(.horizontal, 24).padding(.bottom, 8) }
            if let error = model.settingsError { ErrorBanner(text: error).padding(.horizontal, 24).padding(.bottom, 8) }
            if !runtime.ready && !chat.messages.isEmpty { notServingBanner.padding(.horizontal, 24).padding(.bottom, 8) }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 22) {
                        if !runtime.ready && chat.messages.isEmpty {
                            notServing.frame(maxWidth: .infinity).padding(.vertical, 30)
                        } else if chat.messages.isEmpty {
                            ContentUnavailableView("Try your model", systemImage: "bubble.left.and.bubble.right",
                                description: Text("Send a prompt to inspect response quality and inference speed. History stays on this Mac."))
                                .frame(maxWidth: .infinity).padding(.vertical, 45)
                        }
                        // Equatable rows: a streaming flush or a poll re-renders
                        // only the row that changed, so settled replies are not
                        // re-parsed as Markdown 20 times a second.
                        ForEach(chat.messages) { message in
                            MessageRow(message: message, retry: message.id == lastAssistantID && !chat.generating ? retryAction : nil)
                                .equatable()
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(.horizontal, 40).padding(.vertical, 18)
                    .frame(maxWidth: 900).frame(maxWidth: .infinity)
                }
                .onChange(of: chat.messages.last?.content) { _, _ in if chat.followOutput { proxy.scrollTo("bottom", anchor: .bottom) } }
            }
            composer.padding(.horizontal, 24).padding(.bottom, 20).padding(.top, 6)
                .frame(maxWidth: 900).frame(maxWidth: .infinity)
        }
        .background(Brand.canvas)
        .inspector(isPresented: $model.showsTelemetry) {
            TelemetryInspector(runtime: runtime, chat: chat)
                .inspectorColumnWidth(min: 300, ideal: 330, max: 400)
        }
        .onChange(of: runtime.models.map(\.id)) { _, ids in
            if !ids.contains(chat.selectedModel) { chat.selectedModel = "" }
        }
        .alert("Clear this conversation?", isPresented: $clearConfirmation) {
            Button("Clear", role: .destructive) { chat.clear() }
            Button("Cancel", role: .cancel) {}
        } message: { Text("This removes the saved local conversation.") }
    }

    private var lastAssistantID: UUID? { chat.messages.last { $0.role == "assistant" }?.id }
    private var retryAction: (() -> Void)? {
        guard runtime.ready, runtime.connection != nil else { return nil }
        // Rows are Equatable and keep this closure across re-renders, so it
        // reads the connection (per-launch key), model and settings on click.
        return {
            guard runtime.ready, let connection = runtime.connection else { return }
            chat.retry(model: modelID, connection: connection, settings: settings, supportsTemperature: supportsTemperature)
        }
    }

    /// Model chip, then conversation actions, as on the board.
    private var topBar: some View {
        HStack(spacing: 10) {
            HStack(spacing: 8) {
                Circle().fill(Brand.tint(.from(state: runtime.state, owned: runtime.owned, ready: runtime.ready))).frame(width: 7, height: 7)
                if runtime.ready && runtime.models.count > 1 {
                    Picker("Running model", selection: $chat.selectedModel) {
                        Text("Default model").tag("")
                        ForEach(runtime.models) { Text($0.id).tag($0.id) }
                    }.labelsHidden().pickerStyle(.menu).fixedSize().disabled(chat.generating)
                } else {
                    Text(runtime.ready ? modelID : runtime.state == "Starting" ? "Loading…" : "No model running")
                        .font(.system(size: 12.5, weight: .semibold)).lineLimit(1).truncationMode(.middle)
                }
            }
            .padding(.horizontal, 11).padding(.vertical, 6)
            .background(Brand.panel, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Brand.hairlineColor))
            if runtime.owned { Tag(runtime.servingMode.badge, tint: Brand.boost) }
            Spacer()
            Toggle("Follow", isOn: $chat.followOutput).toggleStyle(.checkbox).font(.system(size: 12)).foregroundStyle(Brand.text2)
            if chat.recoveryRequired {
                Button("Reveal Saved", systemImage: "folder") { NSWorkspace.shared.activateFileViewerSelecting([chat.storage]) }
                    .buttonStyle(.ax(.secondary))
            }
            Button { clearConfirmation = true } label: {
                HStack(spacing: 6) { Text("New Chat"); Text("⌘N").font(.system(size: 10.5)).foregroundStyle(Brand.text3) }
            }
            .buttonStyle(.ax(.secondary))
            .keyboardShortcut("n", modifiers: .command)
            .disabled(chat.generating || (chat.messages.isEmpty && !chat.recoveryRequired))
            .help("Clear this conversation and start a new one")
            Button("Export") { exportConversation() }.buttonStyle(.ax(.plain))
                .foregroundStyle(Brand.text2)
                .disabled(chat.generating || chat.messages.isEmpty || chat.recoveryRequired)
        }
    }

    /// Composer card: prompt, request chips, Send or Stop.
    private var composer: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
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
                    .lineLimit(2...8).textFieldStyle(.plain).font(.system(size: 13.5))
                    .disabled(chat.recoveryRequired)
                Text("↩ send · ⇧↩ line").font(.system(size: 10.5)).foregroundStyle(Brand.text3).padding(.top, 2)
            }
            HStack(spacing: 8) {
                ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                Menu {
                    ForEach(GenerationProfile.allCases, id: \.self) { profile in
                        Button(profile.title + (model.settings.effectiveProfile == profile ? " ✓" : "")) { model.selectPlaygroundProfile(profile) }
                    }
                } label: { chip("Profile " + model.settings.effectiveProfile.title) }
                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                .disabled(chat.generating)
                .help(model.settings.effectiveProfile.summary + " Applies to new Playground requests only.")
                .accessibilityLabel("Playground request profile")
                Button { systemPromptOpen.toggle() } label: { chip(chat.systemPrompt.isEmpty ? "System prompt" : "System prompt ✓") }
                    .buttonStyle(.plain)
                    .popover(isPresented: $systemPromptOpen, arrowEdge: .top) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("System prompt").font(Brand.Font.cardTitle)
                            TextEditor(text: $chat.systemPrompt).font(.system(size: 12.5)).frame(width: 360, height: 140)
                                .scrollContentBackground(.hidden).padding(6)
                                .background(Brand.inset, in: RoundedRectangle(cornerRadius: 8))
                            Text("Sent before every Playground message. Other clients set their own.").font(Brand.Font.caption).foregroundStyle(Brand.text3)
                        }.padding(16).disabled(chat.generating)
                    }
                Button { model.openSettings() } label: {
                    chip(settings.useRuntimeSampling ? "Runtime sampling" : String(format: "Temp %.1f", settings.temperature))
                }.buttonStyle(.plain).help("Sampling for Playground requests. Opens Settings.")
                Button { model.openSettings() } label: { chip("Max " + settings.maxTokens.formatted()) }
                    .buttonStyle(.plain).help("Maximum output tokens for Playground requests. Opens Settings.")
                }
                }
                Spacer(minLength: 6)
                if chat.generating {
                    Button("Stop", systemImage: "stop.fill") { Task { await chat.stop() } }
                        .buttonStyle(.ax(.danger, compact: false))
                        .keyboardShortcut(".", modifiers: .command).help("Stop this reply (⌘.)")
                } else {
                    Button("Send", systemImage: "arrow.up") { send() }
                        .buttonStyle(.ax(.primary, compact: false))
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!canSend)
                }
            }
        }
        .padding(16)
        .panelSurface(radius: 14)
    }

    private func chip(_ text: String) -> some View {
        Text(text).font(.system(size: 11.5, weight: .medium)).foregroundStyle(Brand.text2).lineLimit(1).fixedSize()
            .padding(.horizontal, 9).padding(.vertical, 4.5)
            .background(Brand.control, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
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
                Button("Review Server") { model.page = .api }.buttonStyle(.borderedProminent)
            }
        } else if model.attachedToService {
            ContentUnavailableView {
                Label("Waiting for the connected service", systemImage: "network")
            } description: {
                Text("AX Engine is attached to a service it did not start. Playground opens once that service reports ready.")
            } actions: {
                Button("Review API & Logs") { model.page = .api }
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
                     ?? "Choose a downloaded model on Home.")
            } actions: {
                if model.canStartServer {
                    Button("Start Server", systemImage: "play.fill") { model.startServer() }.buttonStyle(.borderedProminent)
                }
                Button("Choose Model") { model.page = .home }
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
                Button("Review Server") { model.page = .api }
            } else if model.canStartServer {
                Button("Start Server", systemImage: "play.fill") { model.startServer() }.buttonStyle(.borderedProminent)
            } else if runtime.state != "Starting" {
                Button("Open Home") { model.page = .home }
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

private struct MessageRow: View, Equatable {
    let message: ChatMessage
    var retry: (() -> Void)? = nil
    private var canRetry: Bool { retry != nil }
    nonisolated static func == (lhs: MessageRow, rhs: MessageRow) -> Bool {
        MainActor.assumeIsolated { lhs.message == rhs.message && lhs.canRetry == rhs.canRetry }
    }
    var body: some View {
        if message.role == "user" {
            HStack {
                Spacer(minLength: 80)
                Text(message.content).font(.system(size: 13.5, weight: .medium)).textSelection(.enabled)
                    .foregroundStyle(Brand.text1)
                    .padding(.horizontal, 16).padding(.vertical, 10)
                    .background(Brand.accent.opacity(0.22), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .contextMenu { Button("Copy") { copyText(message.content) } }
            }
        } else {
            VStack(alignment: .leading, spacing: 10) {
                if !message.reasoning.isEmpty {
                    DisclosureGroup("Reasoning") {
                        Text(message.reasoning).font(.system(size: 12.5)).foregroundStyle(Brand.text2).textSelection(.enabled)
                    }.font(.system(size: 12)).foregroundStyle(Brand.text3)
                }
                if message.content.isEmpty && message.state == "Streaming" {
                    HStack(spacing: 8) { ProgressView().controlSize(.small); Text("Waiting for the model…").foregroundStyle(Brand.text3) }
                } else if message.state == "Streaming" {
                    Text(message.content).font(.system(size: 13.5)).textSelection(.enabled)
                } else {
                    MarkdownMessage(text: message.content)
                }
                HStack(spacing: 14) {
                    if message.state == "Streaming" {
                        StatusPill(text: "Generating", tint: Brand.serve)
                        Text("⌘. cancels").font(.system(size: 11.5)).foregroundStyle(Brand.text3)
                    } else if message.state != "Complete" {
                        StatusPill(text: message.state, tint: message.state == "Failed" ? Brand.danger : Brand.text3)
                    }
                    ChatSpeedSummary(message: message)
                    Spacer()
                    CopyButton(text: message.content, label: "Copy reply")
                    if let retry {
                        Button(action: retry) { Image(systemName: "arrow.clockwise") }
                            .accessibilityLabel("Retry this reply")
                            .buttonStyle(.plain).foregroundStyle(Brand.text3).help("Retry this reply")
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Per-turn speed line under a reply, e.g. "71.2 tok/s decode · 412 tokens".
private struct ChatSpeedSummary: View {
    let message: ChatMessage
    var body: some View {
        let performance = message.state == "Complete" ? message.performance : nil
        HStack(spacing: 14) {
            if let performance {
                reading(performance.decodeTokensPerSecond.map { String(format: "%.1f tok/s", $0) }, "decode")
                reading(performance.promptFullyCached ? "Cached" : performance.prefillTokensPerSecond.map { String(format: "%.0f tok/s", $0) }, "prefill")
                reading(performance.decodeTokens.map { "\($0)" }, "tokens")
            } else if message.state == "Complete" {
                Text("Timing unavailable").foregroundStyle(Brand.text3)
            }
        }
        .font(.system(size: 11.5)).monospacedDigit()
        .accessibilityElement(children: .combine)
        .help("Native evaluation averages for this reply. Prefill counts uncached prompt tokens. Decode excludes the first output batch. A dash means no measurement was reported, not zero speed.")
    }
    @ViewBuilder private func reading(_ value: String?, _ label: String) -> some View {
        HStack(spacing: 4) {
            Text(value ?? "—").fontWeight(.semibold).foregroundStyle(Brand.text1)
            Text(label).foregroundStyle(Brand.text3)
        }
    }
}

// Settled replies receive Markdown formatting; live replies stay plain text.
// This follows MTPLX's streaming-cost boundary without executing generated HTML.
private struct MarkdownMessage: View {
    let text: String
    private func inline(_ text: String) -> AttributedString {
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
    }
    var body: some View {
        let parts = text.components(separatedBy: "```")
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(parts.enumerated()), id: \.offset) { index, part in
                if index % 2 == 1 {
                    let code = CodeFence.body(part)
                    let language = CodeFence.language(part)
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(language ?? "code").font(.system(size: 11)).foregroundStyle(Brand.text3)
                            Spacer()
                            CopyButton(text: code, label: "Copy code")
                        }
                        ScrollView(.horizontal) {
                            Text(code).font(.system(size: 12.5, design: .monospaced)).lineSpacing(4).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(14)
                    .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous).strokeBorder(Brand.hairlineColor))
                } else {
                    ForEach(Array(MarkdownBlocks.split(part).enumerated()), id: \.offset) { _, block in
                        switch block {
                        case .heading(let level, let text):
                            Text(inline(text)).font(.system(size: level == 1 ? 18 : level == 2 ? 15.5 : 14, weight: .bold))
                                .padding(.top, 4).textSelection(.enabled)
                        case .text(let text):
                            Text(inline(text)).font(.system(size: 13.5)).lineSpacing(3).textSelection(.enabled)
                        }
                    }
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
        } label: { Image(systemName: copied ? "checkmark" : "doc.on.doc").font(.system(size: 11.5)) }
        .buttonStyle(.plain).foregroundStyle(Brand.text3).help(copied ? "Copied" : label)
        .accessibilityLabel(copied ? "Copied" : label)
    }
}

enum CodeFence {
    /// The fence's language tag ("rust"), when it has one.
    static func language(_ fenced: String) -> String? {
        guard let newline = fenced.firstIndex(of: "\n") else { return nil }
        let info = String(fenced[..<newline]).trimmingCharacters(in: .whitespaces)
        return body(fenced) != fenced && !info.isEmpty ? info : nil
    }
    /// The code inside a fence, without a language tag on its first line.
    static func body(_ fenced: String) -> String {
        guard let newline = fenced.firstIndex(of: "\n") else { return fenced }
        let info = String(fenced[..<newline])
        let isLanguageTag = info.count <= 24 && info.range(of: #"^[A-Za-z0-9+#._-]*$"#, options: .regularExpression) != nil
        return isLanguageTag ? String(fenced[fenced.index(after: newline)...]) : fenced
    }
}

/// Playground's right inspector (`06-playground.svg`): live runtime speed,
/// the last reply's native timings, draft acceptance, prefix-cache reuse and
/// per-turn speeds. A dash means "not reported".
private struct TelemetryInspector: View {
    @ObservedObject var runtime: RuntimeController
    @ObservedObject var chat: ChatStore
    var body: some View {
        TimelineView(.periodic(from: .now, by: 2)) { context in
            let now = context.date
            let last = chat.messages.last { $0.role == "assistant" }
            let performance = last?.state == "Complete" ? last?.performance : nil
            let rate = runtime.history.latest(RuntimeController.tokenRateKey, now: now)
            let acceptance = runtime.history.latest("mtpAcceptedPercent", now: now)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    SectionLabel(chat.generating ? "This reply · live" : "Live")
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .firstTextBaseline) {
                            Text(Format.rate(rate).map { $0 + " tok/s" } ?? "— tok/s").font(.system(size: 15, weight: .bold)).monospacedDigit()
                                .foregroundStyle(rate == nil ? Brand.text3 : Brand.text1)
                            Spacer()
                            Text("active \(Format.count(runtime.metrics["ax_engine_generation_active_streams"]) ?? "—")")
                                .font(.system(size: 11)).foregroundStyle(Brand.text3)
                        }
                        Sparkline(points: runtime.history.plotPoints(RuntimeController.tokenRateKey, window: 120, now: now),
                                  start: now.addingTimeInterval(-120), end: now, tint: Brand.serve)
                            .frame(height: 64)
                        HStack { Text("2 min ago"); Spacer(); Text("now") }.font(.system(size: 10)).foregroundStyle(Brand.text3)
                    }
                    .padding(12)
                    .background(Brand.inset, in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
                    .help("Tokens the runtime scheduled per second, prompt and output together.")
                    SectionLabel(chat.generating ? "This reply" : "Last reply")
                    Grid(horizontalSpacing: 8, verticalSpacing: 8) {
                        GridRow {
                            StatTile(title: "First text", value: chat.firstTokenSeconds.map { String(format: "%.0f", $0 * 1000) }, unit: "ms",
                                     help: "Time from Send to the first visible text, including network and display.")
                            StatTile(title: "Prefill", value: performance.flatMap { $0.promptFullyCached ? "Cached" : Format.rate($0.prefillTokensPerSecond) },
                                     unit: performance?.promptFullyCached == true ? nil : "tok/s")
                            StatTile(title: "Decode", value: performance.flatMap { Format.rate($0.decodeTokensPerSecond) }, unit: "tok/s",
                                     help: "Native decode average; excludes the first output batch.")
                        }
                        GridRow {
                            StatTile(title: "Output", value: performance?.decodeTokens.map { "\($0)" }, unit: "tok")
                            StatTile(title: "Prompt", value: performance?.prefillTokens.map { "\($0)" }, unit: "tok",
                                     help: "Uncached prompt tokens the model processed.")
                            StatTile(title: "Elapsed", value: chat.elapsedSeconds.map { String(format: "%.1f", $0) }, unit: "s")
                        }
                    }
                    if acceptance != nil { mtpCallout(acceptance) }
                    SectionLabel("Prefix cache")
                    VStack(spacing: 8) {
                        let hits = runtime.metrics["ax_engine_mlx_prefix_cache_hits_total"]
                        let misses = runtime.metrics["ax_engine_mlx_prefix_cache_misses_total"]
                        let rateText = hits.flatMap { h in misses.flatMap { m in h + m > 0 ? String(format: "%.0f%%", 100 * h / (h + m)) : nil } }
                        row("Hit rate", rateText, tint: Brand.serve, help: "Since the server started. Reused prompt prefixes skip prefill.")
                        row("Reused tokens", Format.count(runtime.metrics["ax_engine_mlx_prefix_cache_reused_tokens_total"]))
                    }
                    SectionLabel("Model")
                    VStack(spacing: 8) {
                        row("Context", runtime.models.first?.contextLength.map { $0.formatted() + " tokens" })
                    }
                    let completed = chat.messages.filter { $0.role == "assistant" && $0.state == "Complete" && $0.performance != nil }
                    let turns = completed.suffix(8)
                    let firstTurn = completed.count - turns.count + 1
                    if !turns.isEmpty {
                        SectionLabel("Turns")
                        VStack(spacing: 7) {
                            ForEach(Array(turns.enumerated()), id: \.element.id) { index, turn in
                                HStack {
                                    Text("Turn \(firstTurn + index)").foregroundStyle(Brand.text2)
                                    Spacer()
                                    Text(turn.performance?.decodeTokens.map { "\($0) tok" } ?? "—").foregroundStyle(Brand.text3)
                                        .frame(width: 70, alignment: .trailing)
                                    Text(Format.rate(turn.performance?.decodeTokensPerSecond).map { $0 + " tok/s" } ?? "—")
                                        .fontWeight(.semibold).frame(width: 84, alignment: .trailing)
                                }
                                .font(.system(size: 12)).monospacedDigit()
                            }
                        }
                    }
                }
                .padding(18)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Brand.panel)
        }
    }

    private func mtpCallout(_ acceptance: Double?) -> some View {
        let acceptance = acceptance ?? 0
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "sparkle").font(.system(size: 11, weight: .bold))
                Text(String(format: "Draft accepted %.0f%%", acceptance)).font(.system(size: 12.5, weight: .semibold))
                Spacer()
                Text("MTP").font(.system(size: 11, weight: .bold))
            }
            .foregroundStyle(Brand.boost)
            StackedBar(segments: [.init(id: "accepted", value: acceptance, color: Brand.boost)], total: 100, height: 5)
            Text("MTP predicts tokens ahead and checks them in one pass. Accepted drafts save decode steps; this is not a speedup measurement.")
                .font(.system(size: 11.5)).foregroundStyle(Brand.text2).fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(Brand.boost.opacity(0.10), in: RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: Brand.Radii.m, style: .continuous).strokeBorder(Brand.boost.opacity(0.25)))
    }

    private func row(_ title: String, _ value: String?, tint: Color = Brand.text1, help: String? = nil) -> some View {
        HStack {
            Text(title).foregroundStyle(Brand.text2)
            Spacer()
            Text(value ?? "—").fontWeight(.semibold).monospacedDigit().foregroundStyle(value == nil ? Brand.text3 : tint)
        }
        .font(.system(size: 12.5))
        .help(help ?? title)
    }
}
