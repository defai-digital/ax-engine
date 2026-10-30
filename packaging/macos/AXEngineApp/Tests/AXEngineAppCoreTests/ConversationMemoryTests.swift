import Foundation
import Testing
@testable import AXEngineAppCore

private func archive(_ messages: [ChatMessage], model: String = "speed-fixture") throws -> Data {
    struct Saved: Encodable {
        let schemaVersion = 1
        let model: String
        let messages: [ChatMessage]
    }
    return try JSONEncoder().encode(Saved(model: model, messages: messages))
}

@Test func replyBudgetChargesUTF8AndJSONEscapesBeforeRetainingText() throws {
    var budget = ConversationReplyBudget(remainingTextBytes: 5, remainingArchiveBytes: 10)
    #expect(try budget.admit("\u{0001}")) // One UTF-8 byte, six JSON bytes.
    #expect(budget.remainingTextBytes == 4)
    #expect(budget.remainingArchiveBytes == 4)
    #expect(try budget.admit("😀"))
    #expect(budget.remainingTextBytes == 0)
    #expect(budget.remainingArchiveBytes == 0)
    #expect(try !budget.admit("x"))
    #expect(budget.replyBytes == 5)
    var escaped = ConversationReplyBudget(remainingTextBytes: 100, remainingArchiveBytes: 5)
    #expect(try !escaped.admit("\u{0001}"))
    #expect(escaped.replyBytes == 0)
    #expect(escaped.remainingTextBytes == 100)
}

@Test func replyBudgetCapsCombinedReasoningAndAnswer() throws {
    var budget = ConversationReplyBudget(remainingTextBytes: ConversationLimits.textBytes,
                                        remainingArchiveBytes: ConversationLimits.archiveBytes)
    #expect(try budget.admit(String(repeating: "r", count: ConversationLimits.replyBytes - 1)))
    #expect(try budget.admit("a"))
    #expect(try !budget.admit("b"))
    #expect(budget.replyBytes == ConversationLimits.replyBytes)
}

@Test @MainActor func historyLimitRejectsNewTurnsWithoutChangingSavedOrVisibleMessages() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    let messages = (0..<ConversationLimits.messages).map {
        ChatMessage(role: $0.isMultiple(of: 2) ? "user" : "assistant", content: "Message \($0)")
    }
    let original = try archive(messages)
    try original.write(to: file)
    let chat = ChatStore(storage: file)
    #expect(!chat.recoveryRequired)
    chat.send("next", model: "speed-fixture", connection: EngineConnection(port: 1), settings: AppSettings())
    #expect(!chat.generating)
    #expect(chat.messages.map(\.id) == messages.map(\.id))
    #expect(chat.error?.contains("200 messages") == true)
    #expect(try Data(contentsOf: file) == original)
    let exported = file.appendingPathExtension("export.json")
    defer { try? FileManager.default.removeItem(at: exported) }
    chat.export(to: exported)
    #expect(ChatStore(storage: exported).messages.count == ConversationLimits.messages)
    chat.clear()
    #expect(chat.messages.isEmpty)
    #expect(ChatStore(storage: file).messages.isEmpty)
}

@Test @MainActor func byteLimitAndRejectedRetryKeepTheOriginalConversation() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    let messages = [ChatMessage(role: "user", content: "x"),
                    ChatMessage(role: "assistant", content: String(repeating: "a", count: ConversationLimits.textBytes - 1))]
    let original = try archive(messages)
    try original.write(to: file)
    let chat = ChatStore(storage: file)
    #expect(!chat.recoveryRequired)
    chat.send("next", model: "speed-fixture", connection: EngineConnection(port: 1), settings: AppSettings())
    #expect(!chat.generating)
    #expect(chat.messages.map(\.id) == messages.map(\.id))
    chat.systemPrompt = String(repeating: "s", count: ConversationLimits.promptBytes + 1)
    chat.retry(model: "speed-fixture", connection: EngineConnection(port: 1), settings: AppSettings(), supportsTemperature: true)
    #expect(!chat.generating)
    #expect(chat.messages.map(\.id) == messages.map(\.id))
    #expect(try Data(contentsOf: file) == original)
}

@Test @MainActor func oversizedArchiveReadsAreBoundedAndCannotBeOverwrittenBySend() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    try Data().write(to: file)
    let handle = try FileHandle(forWritingTo: file)
    try handle.truncate(atOffset: UInt64(ConversationLimits.archiveBytes * 4))
    try handle.close()
    let chat = ChatStore(storage: file)
    #expect(chat.recoveryRequired)
    #expect(chat.messages.isEmpty)
    chat.send("next", model: "speed-fixture", connection: EngineConnection(port: 1), settings: AppSettings())
    #expect(!chat.generating)
    let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
    #expect((attributes[.size] as? NSNumber)?.intValue == ConversationLimits.archiveBytes * 4)
    chat.clear() // Explicit user action is required to replace the unread archive.
    #expect(!chat.recoveryRequired)
    #expect(!ChatStore(storage: file).recoveryRequired)
}

@Test @MainActor func excessiveMessageCountsAndUnknownVersionsRequireExplicitRecovery() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: file) }
    let tooMany = try archive((0...ConversationLimits.messages).map { _ in ChatMessage(role: "user", content: "") })
    for original in [tooMany, Data("{\"schemaVersion\":2,\"messages\":[]}".utf8)] {
        try original.write(to: file)
        let chat = ChatStore(storage: file)
        #expect(chat.recoveryRequired)
        #expect(chat.messages.isEmpty)
        chat.send("next", model: "speed-fixture", connection: EngineConnection(port: 1), settings: AppSettings())
        #expect(!chat.generating)
        #expect(try Data(contentsOf: file) == original)
    }
}

@Test @MainActor func streamedHistoryOverflowKeepsABoundedRecoverableArchive() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let launched = try await startChatPerformanceFixture(in: root)
    let process = launched.process
    defer { if process.isRunning { process.terminate() }; process.waitUntilExit() }
    let port = launched.port
    let file = root.appendingPathComponent("conversation.json")
    try archive([ChatMessage(role: "user", content: "x"),
                 ChatMessage(role: "assistant", content: String(repeating: "a", count: ConversationLimits.textBytes - 21))]).write(to: file)
    var chat: ChatStore? = ChatStore(storage: file)
    weak var lifetime = chat
    chat?.send("x", model: "speed-fixture", connection: EngineConnection(port: port), settings: AppSettings())
    let finished = ContinuousClock.now.advanced(by: .seconds(5))
    while chat?.generating == true, ContinuousClock.now < finished {
        try await Task.sleep(for: .milliseconds(10))
    }
    if chat?.generating == true { await chat?.stop() }
    #expect(chat?.messages.last?.state == "Failed")
    #expect(chat?.error?.contains("memory limit") == true)
    #expect((chat?.messages.reduce(0) { $0 + $1.content.utf8.count + $1.reasoning.utf8.count } ?? 0) <= ConversationLimits.textBytes)
    let restored = ChatStore(storage: file)
    #expect(!restored.recoveryRequired)
    #expect(restored.messages.last?.state == "Failed")
    chat = nil
    await Task.yield()
    #expect(lifetime == nil, "Finished stream and flush tasks must release their store")
}
