import Foundation
import Testing
@testable import AXEngineAppCore

@Test func replyBudgetCountsEscapesLikeJSONEncoder() throws {
    var samples = (0..<0x80).map { String(UnicodeScalar(UInt8($0))) }
    samples += ["", "plain text", "a/b\\c\"d", "tab\tline\nfeed\u{0C}\r\u{08}", "\u{1B}[31mred", "é😀\u{2028}\u{FEFF}\u{7F}",
                String(repeating: "x/\n\u{01}", count: 64)]
    for text in samples {
        let expected = try JSONEncoder().encode(text).count - 2
        #expect(ConversationReplyBudget.encodedJSONBytes(text) == expected, "\(Array(text.utf8))")
    }
}

@Test func capturedOutputTailSkipsUnchangedAndKeepsUTF8Boundaries() {
    let capture = CapturedOutput()
    #expect(capture.tail(8, since: 0) == nil)
    capture.append(Data("ab".utf8), isError: true)
    let first = capture.tail(8, since: 0)
    #expect(first?.result.stderr == "ab")
    #expect(capture.tail(8, since: first!.version) == nil)
    // A tail cut must not start inside the two-byte "é".
    capture.append(Data("cdéfghi".utf8), isError: true)
    let second = capture.tail(5, since: first!.version)
    #expect(second?.result.stderr == "fghi")
    #expect(second?.result.truncated == true)
    #expect(capture.snapshot().stderr == "abcdéfghi")
}

@MainActor @Test func liveUpdatesStopWhileTheChildIsSilent() async throws {
    let job = ProcessJob()
    var updates = 0
    _ = try await job.run(executable: URL(fileURLWithPath: "/bin/sh"),
                          arguments: ["-c", "echo ready; sleep 1"]) { _ in updates += 1 }
    // One live update for "ready" plus the final one, not one per 100 ms poll.
    #expect(updates <= 3)
}
