// Example: chat completion with the AX Engine Swift SDK.
//
// Requires a running ax-engine-server on http://127.0.0.1:8080.
//
// Run:
//   cd examples/swift && swift run Chat

import AxEngine

let client = AxEngineClient()

// ── Health check ──────────────────────────────────────────────────────────────
let health = try await client.health()
print("status:", health.status)

// ── Chat completion ───────────────────────────────────────────────────────────
let response = try await client.chatCompletion(.init(
    messages: [
        .init(role: "system", content: "You are AX Engine."),
        .init(role: "user",   content: "Say hello in one sentence."),
    ],
    maxTokens: 64,
    temperature: 0.7
))
print("chat:", response.choices[0].message.content)
