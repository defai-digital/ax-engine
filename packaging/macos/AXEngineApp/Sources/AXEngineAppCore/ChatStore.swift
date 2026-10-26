import Foundation
import Combine
import AxEngine

public struct ChatMessage: Codable, Identifiable, Sendable {
    public let id: UUID
    public let role: String
    public var content: String
    public var reasoning: String
    public var state: String
    public var performance: ChatPerformance?
    public init(role: String, content: String, reasoning: String = "", state: String = "Complete") {
        id = UUID(); self.role = role; self.content = content; self.reasoning = reasoning; self.state = state
    }
}

enum ConversationLimits {
    static let messages = 200
    static let textBytes = 4 * 1024 * 1024
    static let archiveBytes = 8 * 1024 * 1024
    static let promptBytes = 256 * 1024
    static let replyBytes = 2 * 1024 * 1024
    // Space for terminal state and the small, fixed-shape timing record.
    static let terminalMetadataBytes = 16 * 1024
}

private struct BoundedMessages: Decodable {
    let values: [ChatMessage]
    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        guard (container.count ?? 0) <= ConversationLimits.messages else {
            throw AppFailure.message("Saved conversation exceeds 200 messages. The file has been kept.")
        }
        var values: [ChatMessage] = []
        var textBytes = 0
        while !container.isAtEnd {
            guard values.count < ConversationLimits.messages else {
                throw AppFailure.message("Saved conversation exceeds 200 messages. The file has been kept.")
            }
            let message = try container.decode(ChatMessage.self)
            textBytes += message.content.utf8.count + message.reasoning.utf8.count
            guard textBytes <= ConversationLimits.textBytes else {
                throw AppFailure.message("Saved conversation exceeds 4 MiB of text. The file has been kept.")
            }
            values.append(message)
        }
        self.values = values
    }
}

private struct ConversationArchive: Codable {
    let schemaVersion: Int
    let model: String?
    let messages: [ChatMessage]
    init(schemaVersion: Int, model: String?, messages: [ChatMessage]) {
        self.schemaVersion = schemaVersion; self.model = model; self.messages = messages
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, model, messages }
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        guard schemaVersion == 1 else { throw AppFailure.message("Unsupported saved conversation version. The file has been kept.") }
        model = try container.decodeIfPresent(String.self, forKey: .model)
        messages = try container.decode(BoundedMessages.self, forKey: .messages).values
    }
}

/// Counts escaped JSON text as well as UTF-8 bytes before retaining a stream delta.
/// This prevents a control-character-heavy reply from overflowing the archive
/// even when its unescaped display text still fits.
struct ConversationReplyBudget {
    var remainingTextBytes: Int
    var remainingArchiveBytes: Int
    private(set) var replyBytes = 0

    mutating func admit(_ text: String) throws -> Bool {
        let bytes = text.utf8.count
        guard bytes <= remainingTextBytes,
              bytes <= ConversationLimits.replyBytes - replyBytes else { return false }
        let encodedBytes = try JSONEncoder().encode(text).count - 2
        guard encodedBytes <= remainingArchiveBytes else { return false }
        remainingTextBytes -= bytes; remainingArchiveBytes -= encodedBytes; replyBytes += bytes
        return true
    }
}

@MainActor
public final class ChatStore: ObservableObject {
    @Published public private(set) var messages: [ChatMessage] = []
    @Published public private(set) var generating = false
    @Published public private(set) var error: String?
    @Published public private(set) var firstTokenSeconds: Double?
    @Published public private(set) var elapsedSeconds: Double?
    @Published public var systemPrompt = ""
    @Published public var draft = ""
    @Published public var selectedModel = ""
    @Published public var followOutput = true
    private var task: Task<Void, Never>?
    private var requestID: Int?
    private var activeConnection: EngineConnection?
    public let storage: URL
    @Published public private(set) var recoveryRequired = false
    private var content = "", reasoning = ""
    private var conversationModel: String?
    private var flushTask: Task<Void, Never>?

    public init(storage: URL = AppPaths.conversation) {
        self.storage = storage
        do {
            if FileManager.default.fileExists(atPath: storage.path) {
                let file = try FileHandle(forReadingFrom: storage)
                defer { try? file.close() }
                let data = try file.read(upToCount: ConversationLimits.archiveBytes + 1) ?? Data()
                guard data.count <= ConversationLimits.archiveBytes else {
                    throw AppFailure.message("Saved conversation exceeds 8 MiB. The file has been kept.")
                }
                let decoder = JSONDecoder()
                if data.first(where: { ![9, 10, 13, 32].contains($0) }) == 91 {
                    messages = try decoder.decode(BoundedMessages.self, from: data).values
                    conversationModel = messages.isEmpty ? nil : "unknown saved model"
                } else {
                    let archive = try decoder.decode(ConversationArchive.self, from: data)
                    messages = archive.messages; conversationModel = archive.model
                }
                for index in messages.indices {
                    if messages[index].state == "Streaming" { messages[index].state = "Interrupted" }
                    if messages[index].state != "Complete" { messages[index].performance = nil }
                }
            }
        } catch {
            messages = []; conversationModel = nil; recoveryRequired = true
            self.error = "Conversation recovery: " + error.localizedDescription + " Reveal the saved file or clear it to start a new conversation."
        }
    }

    public func send(_ prompt: String, model: String, connection: EngineConnection, settings: AppSettings,
                     supportsTemperature: Bool = true) {
        beginSend(prompt, model: model, connection: connection, settings: settings,
                  supportsTemperature: supportsTemperature, previousMessages: messages)
    }

    private func beginSend(_ prompt: String, model: String, connection: EngineConnection, settings: AppSettings,
                           supportsTemperature: Bool, previousMessages: [ChatMessage]) {
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !generating, !prompt.isEmpty else { return }
        guard prompt.utf8.count <= ConversationLimits.promptBytes else { error = "The message exceeds 256 KiB."; return }
        guard !model.isEmpty else { error = "Select a running model."; return }
        do { try settings.validate() } catch { self.error = error.localizedDescription; return }
        if let previous = conversationModel, previous != model, !messages.isEmpty {
            error = "Clear the conversation before changing models. Your saved messages have been kept."
            return
        }
        guard !recoveryRequired else { error = "Reveal or clear the saved conversation before sending. The original file has been kept."; return }
        guard systemPrompt.utf8.count <= ConversationLimits.promptBytes else { error = "The system prompt exceeds 256 KiB."; return }
        let nextMessages = previousMessages + [.init(role: "user", content: prompt), .init(role: "assistant", content: "", state: "Streaming")]
        let textBytes = nextMessages.reduce(0) { $0 + $1.content.utf8.count + $1.reasoning.utf8.count }
        guard nextMessages.count <= ConversationLimits.messages, textBytes < ConversationLimits.textBytes else {
            error = "This conversation has reached its limit (200 messages or 4 MiB of text). Export it, then clear it to continue."
            return
        }
        let archiveBytes: Int
        do { archiveBytes = try encodedArchive(messages: nextMessages, model: model).count }
        catch { self.error = error.localizedDescription; return }
        guard archiveBytes < ConversationLimits.archiveBytes - ConversationLimits.terminalMetadataBytes else {
            error = "This conversation has reached its saved history limit. Export it, then clear it to continue."
            return
        }
        var replyBudget = ConversationReplyBudget(remainingTextBytes: ConversationLimits.textBytes - textBytes,
            remainingArchiveBytes: ConversationLimits.archiveBytes - ConversationLimits.terminalMetadataBytes - archiveBytes)
        conversationModel = model
        let client = connection.sdk
        activeConnection = connection; requestID = nil; error = nil
        firstTokenSeconds = nil; elapsedSeconds = nil; content = ""; reasoning = ""
        // Failed/stopped turns are visible locally but must not leave orphaned
        // user prompts in the next request's history.
        var history: [OpenAiChatMessage] = []
        var pendingUser: ChatMessage?
        for message in previousMessages {
            if message.role == "user" { pendingUser = message }
            else if message.role == "assistant", message.state == "Complete", let user = pendingUser {
                history.append(.init(role: "user", content: user.content))
                history.append(.init(role: "assistant", content: message.content))
                pendingUser = nil
            }
        }
        if !systemPrompt.isEmpty { history.insert(.init(role: "system", content: systemPrompt), at: 0) }
        history.append(.init(role: "user", content: prompt))
        messages = nextMessages
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
                if let last = messages.indices.last, messages[last].state != "Complete" {
                    messages[last].performance = nil
                }
                generating = false; elapsedSeconds = Date().timeIntervalSince(started)
                requestID = nil; activeConnection = nil; persist()
                content = ""; reasoning = ""; task = nil
            }
            do {
                var request = settings.chatRequest(model: model, messages: history,
                    supportsTemperature: supportsTemperature)
                request.streamOptions = .init(includeUsage: true)
                for try await chunk in client.streamChatCompletion(request) {
                    try Task.checkCancellation()
                    if requestID == nil { requestID = EngineConnection.requestID(chunk.id) }
                    if let measurement = chunk.usage?.axEngine, let last = messages.indices.last {
                        messages[last].performance = ChatPerformance(measurement)
                    }
                    for choice in chunk.choices {
                        if let delta = choice.delta.content, !delta.isEmpty {
                            if firstTokenSeconds == nil { firstTokenSeconds = Date().timeIntervalSince(started) }
                            guard try replyBudget.admit(delta) else { throw AppFailure.message("Conversation memory limit reached. Export it, then clear it to continue.") }
                            content += delta
                        }
                        if let delta = choice.delta.reasoningContent {
                            guard try replyBudget.admit(delta) else { throw AppFailure.message("Conversation memory limit reached. Export it, then clear it to continue.") }
                            reasoning += delta
                        }
                        if choice.finishReason == "error" { throw AppFailure.message("The runtime reported a generation error.") }
                        if choice.finishReason == "tool_calls" { throw AppFailure.message("This conversation requested tools. Tool execution is not enabled in AX Engine GUI.") }
                    }
                }
                try Task.checkCancellation()
                if !messages.isEmpty { messages[messages.count - 1].state = "Complete" }
            } catch {
                if let requestID { try? await connection.cancel(requestID: requestID) }
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
        messages = []; conversationModel = nil; error = nil; recoveryRequired = false
        content = ""; reasoning = ""; draft = ""
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
        beginSend(prompt, model: model, connection: connection, settings: settings,
                  supportsTemperature: supportsTemperature, previousMessages: Array(messages[..<index]))
    }
    private func encodedArchive(messages: [ChatMessage], model: String?) throws -> Data {
        try JSONEncoder().encode(ConversationArchive(schemaVersion: 1, model: model, messages: messages))
    }

    public func export(to destination: URL) {
        guard !generating, !recoveryRequired else { return }
        do {
            let data = try encodedArchive(messages: messages, model: conversationModel)
            guard data.count <= ConversationLimits.archiveBytes else { throw AppFailure.message("Conversation exceeds the export limit.") }
            try data.write(to: destination, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path)
        } catch { self.error = "Could not export conversation: " + error.localizedDescription }
    }

    private func persist() {
        guard !recoveryRequired else { return }
        do {
            let data = try encodedArchive(messages: messages, model: conversationModel)
            guard data.count <= ConversationLimits.archiveBytes else { throw AppFailure.message("Conversation exceeds the 8 MiB local history limit. Export or clear it.") }
            try FileManager.default.createDirectory(at: storage.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: storage, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: storage.path)
        } catch { self.error = "Could not save conversation: " + error.localizedDescription }
    }
}
