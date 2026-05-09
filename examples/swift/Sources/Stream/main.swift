// Example: streaming with the AX Engine Swift SDK.
//
// Requires a running ax-engine-server on http://127.0.0.1:8080.
//
// Run:
//   cd examples/swift && swift run Stream

import AxEngine
import Foundation

let client = AxEngineClient()

// ── Streaming chat completion ─────────────────────────────────────────────────
print("--- streaming chat ---")
for try await chunk in client.streamChatCompletion(.init(
    messages: [.init(role: "user", content: "Count from 1 to 5.")],
    maxTokens: 64
)) {
    if let text = chunk.choices.first?.delta.content {
        print(text, terminator: "")
        fflush(stdout)
    }
}
print()

// ── Streaming text completion ─────────────────────────────────────────────────
print("\n--- streaming completion ---")
for try await chunk in client.streamCompletion(.init(
    prompt: "The quick brown fox",
    maxTokens: 16
)) {
    print(chunk.choices.first?.text ?? "", terminator: "")
    fflush(stdout)
}
print()

// ── Native ax-engine streaming (request / step / response events) ─────────────
print("\n--- stream_generate ---")
for try await event in client.streamGenerate(.init(inputTokens: [1, 2, 3], maxOutputTokens: 4)) {
    switch event.event {
    case "step":
        if let tokens = event.step?.deltaTokens, !tokens.isEmpty {
            print("  step tokens:", tokens)
        }
    case "response":
        print("  finish_reason:", event.response?.response.finishReason ?? "nil")
    default:
        break
    }
}
