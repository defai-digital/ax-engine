# Swift SDK

`sdk/swift/` is the native Swift client for AX Engine v6, distributed as a
Swift Package.

It is intentionally thin:

- it speaks to `ax-engine-server`, not directly to `ax-engine-core`
- zero external dependencies (Foundation and `URLSession` only)
- Swift 5.9+ with `async/await` throughout
- `AsyncThrowingStream` for SSE streaming — no callback plumbing

## Package

```text
name: "ax-engine-swift"
```

Located at `sdk/swift/`. Add it to your project through Swift Package Manager.

### Local development

In your `Package.swift`:

```swift
.package(path: "../../sdk/swift")
```

Or use the repository root example targets directly.

### Version

The package version is kept in lockstep with `Cargo.toml` and the other SDKs
by the CI "Verify version consistency" check. The current version is `6.12.0`.

## Quick Start

```swift
import AxEngine

let client = AxEngineClient()  // default: http://127.0.0.1:31418

let response = try await client.chatCompletion(.init(
    messages: [
        .init(role: "system", content: "You are AX Engine."),
        .init(role: "user",   content: "Say hello in one sentence."),
    ],
    maxTokens: 128,
    temperature: 0.7
))
print(response.choices[0].message.content ?? "")
```

## Client Configuration

```swift
let client = AxEngineClient(
    baseURL: URL(string: "http://127.0.0.1:31418")!,  // default
    session: .shared                                  // default
)
```

For authenticated servers, set a custom `URLSession` with a delegate or add the
`Authorization` header through a custom `URLRequest` interceptor.

## API Reference

### Health and models

```swift
// GET /health
let health = try await client.health()

// GET /v1/models
let models = try await client.models()
```

### Native ax-engine endpoints

```swift
// POST /v1/generate — blocking native generate
let resp = try await client.generate(.init(
    inputTokens: [1, 2, 3],
    maxOutputTokens: 32
))

// Stepwise lifecycle
let report   = try await client.submit(.init(inputTokens: [1, 2, 3], maxOutputTokens: 32))
let snapshot = try await client.requestSnapshot(id: report.requestId)
let step     = try await client.step()
let cancelled = try await client.cancel(id: report.requestId)

// GET /v1/runtime — resolved backend and host report
let info = try await client.runtime()
```

> **Known limitation:** the typed request does not yet expose the free-form
> `tools` / `tool_choice` / `response_format` / `reasoning` fields.
> Foundation's `.convertToSnakeCase` also rewrites dictionary keys, so
> arbitrary schema keys (for example `"userId"` inside tool parameters)
> would be silently corrupted; typed support is pending an
> explicit-CodingKeys refactor. Tool-call *responses*, streamed tool-call
> deltas, assistant tool-call echo, and `role: "tool"` result messages are
> fully supported.

### OpenAI-compatible endpoints

```swift
// Text completion
let resp = try await client.completion(.init(
    prompt: "Hello from Swift",
    maxTokens: 64
))
print(resp.choices[0].text)

// Chat completion
let chatResp = try await client.chatCompletion(.init(
    messages: [
        .init(role: "system", content: "You are AX Engine."),
        .init(role: "user",   content: "Hello!"),
    ],
    maxTokens: 128,
    temperature: 0.7
))
print(chatResp.choices[0].message.content ?? "")

// Embeddings
let embed = try await client.embeddings(.init(
    input: [1, 2, 3],
    pooling: "last",
    normalize: true
))
print(embed.data[0].embedding.count)
```

### Streaming

All streaming methods return `AsyncThrowingStream`. Iterate with `for try await`:

```swift
// Streaming chat
for try await chunk in client.streamChatCompletion(.init(
    messages: [.init(role: "user", content: "Count from 1 to 5.")],
    maxTokens: 64
)) {
    if let text = chunk.choices.first?.delta.content {
        print(text, terminator: "")
    }
}
print()

// Streaming text completion
for try await chunk in client.streamCompletion(.init(
    prompt: "Once upon a time",
    maxTokens: 64
)) {
    print(chunk.choices.first?.text ?? "", terminator: "")
}
print()

// Native ax-engine streaming (request / step / response events)
for try await event in client.streamGenerate(.init(
    inputTokens: [1, 2, 3],
    maxOutputTokens: 32
)) {
    switch event.event {
    case "step":
        if let tokens = event.step?.deltaTokens, !tokens.isEmpty {
            print("step tokens:", tokens)
        }
    case "response":
        print("finish_reason:", event.response?.response.finishReason ?? "nil")
    default:
        break
    }
}
```

### Error handling

```swift
do {
    let resp = try await client.chatCompletion(.init(messages: []))
} catch let error as AxEngineHTTPError {
    print("status:", error.statusCode)
    print("message:", error.message)
}
```

### Gemma 4 multimodal

The Swift SDK supports the Gemma 4 unified multimodal contract. Pass
preprocessed tensor data through `multimodalInputs` on chat requests:

```swift
let resp = try await client.chatCompletion(.init(
    messages: [
        .init(role: "user", content: "Describe this image. <image>"),
    ],
    maxTokens: 128,
    multimodalInputs: .init(gemma4Unified: .init(
        images: [/* preprocessed tensor data */]
    ))
))
```

## Running Examples

```bash
cd sdk/swift

# Chat completion (requires ax-engine-server on http://127.0.0.1:31418)
swift run ChatExample

# Streaming chat, completion, and native generate
swift run StreamExample
```

## Running Tests

```bash
cd sdk/swift && swift test
```

Tests run fully offline — no server required.
