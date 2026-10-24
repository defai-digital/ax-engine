import Foundation
import Combine
import AxEngine

public struct ChatMessage: Codable, Identifiable, Sendable {
    public let id: UUID
    public let role: String
    public var content: String
    public var reasoning: String
    public var state: String
    public init(role: String, content: String, reasoning: String = "", state: String = "Complete") {
        id = UUID(); self.role = role; self.content = content; self.reasoning = reasoning; self.state = state
    }
}

private struct ConversationArchive: Codable {
    let schemaVersion: Int
    let model: String?
    let messages: [ChatMessage]
}

@MainActor
public final class ChatStore: ObservableObject {
    @Published public private(set) var messages: [ChatMessage] = []
    @Published public private(set) var generating = false
    @Published public private(set) var error: String?
    @Published public private(set) var firstTokenSeconds: Double?
    @Published public private(set) var elapsedSeconds: Double?
    @Published public var systemPrompt = ""
    private var task: Task<Void, Never>?
    private var requestID: Int?
    private var activeConnection: EngineConnection?
    private let storage: URL
    private var content = "", reasoning = ""
    private var conversationModel: String?
    private var flushTask: Task<Void, Never>?

    public init(storage: URL = AppPaths.conversation) {
        self.storage = storage
        do {
            if FileManager.default.fileExists(atPath: storage.path) {
                let data = try Data(contentsOf: storage)
                guard data.count <= 8 * 1024 * 1024 else { throw AppFailure.message("Saved conversation exceeds 8 MiB. Move it aside before loading.") }
                if let archive = try? JSONDecoder().decode(ConversationArchive.self, from: data), archive.schemaVersion == 1 {
                    messages = archive.messages
                    conversationModel = archive.model
                } else {
                    messages = try JSONDecoder().decode([ChatMessage].self, from: data)
                    conversationModel = messages.isEmpty ? nil : "unknown saved model"
                }
                for index in messages.indices where messages[index].state == "Streaming" { messages[index].state = "Interrupted" }
            }
        } catch { self.error = "Conversation recovery: " + error.localizedDescription }
    }

    public func send(_ prompt: String, model: String, connection: EngineConnection, settings: AppSettings,
                     supportsTemperature: Bool = true) {
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !generating, !prompt.isEmpty else { return }
        guard prompt.utf8.count <= 256 * 1024 else { error = "The message exceeds 256 KiB."; return }
        guard !model.isEmpty else { error = "Select a running model."; return }
        do { try settings.validate() } catch { self.error = error.localizedDescription; return }
        if let previous = conversationModel, previous != model, !messages.isEmpty {
            error = "Clear the conversation before changing models. Your saved messages have been kept."
            return
        }
        conversationModel = model
        let client = connection.sdk
        activeConnection = connection; requestID = nil; error = nil
        firstTokenSeconds = nil; elapsedSeconds = nil; content = ""; reasoning = ""
        // Failed/stopped turns are visible locally but must not leave orphaned
        // user prompts in the next request's history.
        var history: [OpenAiChatMessage] = []
        var pendingUser: ChatMessage?
        for message in messages {
            if message.role == "user" { pendingUser = message }
            else if message.role == "assistant", message.state == "Complete", let user = pendingUser {
                history.append(.init(role: "user", content: user.content))
                history.append(.init(role: "assistant", content: message.content))
                pendingUser = nil
            }
        }
        if !systemPrompt.isEmpty { history.insert(.init(role: "system", content: systemPrompt), at: 0) }
        history.append(.init(role: "user", content: prompt))
        messages.append(.init(role: "user", content: prompt))
        messages.append(.init(role: "assistant", content: "", state: "Streaming"))
        generating = true
        let started = Date()
        flushTask = Task {
            while !Task.isCancelled {
                flush()
                try? await Task.sleep(for: .milliseconds(50))
            }
        }
        task = Task {
            defer {
                flushTask?.cancel(); flushTask = nil; flush()
                generating = false; elapsedSeconds = Date().timeIntervalSince(started)
                requestID = nil; activeConnection = nil; persist()
            }
            do {
                let request = OpenAiChatCompletionRequest(model: model, messages: history, maxTokens: settings.maxTokens,
                    temperature: supportsTemperature ? settings.temperature : nil)
                for try await chunk in client.streamChatCompletion(request) {
                    try Task.checkCancellation()
                    if requestID == nil { requestID = EngineConnection.requestID(chunk.id) }
                    for choice in chunk.choices {
                        if let delta = choice.delta.content, !delta.isEmpty {
                            if firstTokenSeconds == nil { firstTokenSeconds = Date().timeIntervalSince(started) }
                            content += delta
                        }
                        if let delta = choice.delta.reasoningContent { reasoning += delta }
                        if choice.finishReason == "error" { throw AppFailure.message("The runtime reported a generation error.") }
                        if choice.finishReason == "tool_calls" { throw AppFailure.message("This conversation requested tools. Tool execution is not enabled in AX Engine GUI.") }
                    }
                    if content.utf8.count + reasoning.utf8.count > 2 * 1024 * 1024 {
                        if let requestID { try? await connection.cancel(requestID: requestID) }
                        throw AppFailure.message("Reply exceeded the 2 MiB display limit and was stopped.")
                    }
                }
                try Task.checkCancellation()
                if !messages.isEmpty { messages[messages.count - 1].state = "Complete" }
            } catch {
                if !messages.isEmpty { messages[messages.count - 1].state = Task.isCancelled || error is CancellationError ? "Stopped" : "Failed" }
                if !Task.isCancelled && !(error is CancellationError) { self.error = error.localizedDescription }
            }
        }
    }
    private func flush() {
        guard let last = messages.indices.last, messages[last].role == "assistant" else { return }
        if messages[last].content != content || messages[last].reasoning != reasoning {
            messages[last].content = content; messages[last].reasoning = reasoning
        }
    }
    public func stop() async {
        let id = requestID, connection = activeConnection
        task?.cancel()
        // Cancel the transport immediately; the server also cancels on disconnect.
        // Do not hold UI cancellation hostage to the SDK's longer inference timeout.
        if let id, let connection { try? await connection.cancel(requestID: id) }
        await task?.value
    }
    public func clear() {
        guard !generating else { return }
        messages = []; conversationModel = nil; error = nil
        firstTokenSeconds = nil; elapsedSeconds = nil; persist()
    }
    public func retry(model: String, connection: EngineConnection, settings: AppSettings, supportsTemperature: Bool) {
        guard !generating, let index = messages.lastIndex(where: { $0.role == "user" }) else { return }
        let prompt = messages[index].content
        // Validate before removing anything: a rejected retry must preserve the
        // original turn just as a rejected Send does.
        guard !model.isEmpty, conversationModel == nil || conversationModel == model else {
            error = "Clear the conversation before changing models. Your saved messages have been kept."
            return
        }
        do { try settings.validate() } catch { self.error = error.localizedDescription; return }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              prompt.utf8.count <= 256 * 1024 else { error = "The saved message cannot be retried."; return }
        messages.removeSubrange(index...)
        send(prompt, model: model, connection: connection, settings: settings, supportsTemperature: supportsTemperature)
    }
    private func persist() {
        do {
            let data = try JSONEncoder().encode(ConversationArchive(schemaVersion: 1, model: conversationModel, messages: messages))
            guard data.count <= 8 * 1024 * 1024 else { throw AppFailure.message("Conversation exceeds the 8 MiB local history limit. Export or clear it.") }
            try FileManager.default.createDirectory(at: storage.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: storage, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: storage.path)
        } catch { self.error = "Could not save conversation: " + error.localizedDescription }
    }
}
