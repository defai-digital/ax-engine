# Go SDK

`sdk/go/axengine` is the Go HTTP client for AX Engine v4.

It is intentionally thin:

- it speaks to `ax-engine-server`, not directly to `ax-engine-core`
- zero external dependencies (stdlib only)
- typed generics for streaming responses (Go 1.22+)
- channel-based streaming — no iterator interface to implement

## Module

```
module github.com/ax-engine/ax-engine-go
```

Located at `sdk/go/axengine/`. For local development use a `replace` directive:

```
require github.com/ax-engine/ax-engine-go v0.0.0
replace github.com/ax-engine/ax-engine-go => ../../sdk/go/axengine
```

See `examples/go/go.mod` for a working example.

## Install

Once published:

```bash
go get github.com/ax-engine/ax-engine-go
```

From the repository root (local development):

```bash
cd examples/go && go build ./...
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/ax-engine/ax-engine-go"
)

func main() {
    client := axengine.NewClient(nil)

    resp, err := client.ChatCompletion(context.Background(), axengine.OpenAiChatCompletionRequest{
        Messages: []axengine.OpenAiChatMessage{
            {Role: "system", Content: "You are AX Engine."},
            {Role: "user", Content: "Say hello in one sentence."},
        },
        MaxTokens:   axengine.Ptr(128),
        Temperature: axengine.Ptr(0.7),
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(resp.Choices[0].Message.Content)
}
```

## Client Configuration

```go
client := axengine.NewClient(&axengine.ClientOptions{
    BaseURL:    "http://127.0.0.1:8080",  // default
    HTTPClient: &http.Client{Timeout: 30 * time.Second},
    Headers:    http.Header{"Authorization": {"Bearer token"}},
})
```

## API Reference

### Standard JSON endpoints

```go
// GET /health
resp, err := client.Health(ctx)

// GET /v1/runtime (ax-engine server info)
// (call via Generate or ChatCompletion to get runtime in response)

// Native ax-engine generate (token-based)
resp, err := client.Generate(ctx, axengine.PreviewGenerateRequest{
    InputTokens:     []int{1, 2, 3},
    MaxOutputTokens: axengine.Ptr(32),
})

// Stepwise lifecycle
report, err := client.Submit(ctx, req)
snap,   err := client.RequestSnapshot(ctx, report.RequestID)
step,   err := client.Step(ctx)
snap,   err := client.Cancel(ctx, report.RequestID)

// OpenAI-compatible
resp, err := client.Completion(ctx, axengine.OpenAiCompletionRequest{
    Prompt:    "Hello from Go",
    MaxTokens: axengine.Ptr(64),
})

resp, err := client.ChatCompletion(ctx, axengine.OpenAiChatCompletionRequest{
    Messages: []axengine.OpenAiChatMessage{
        {Role: "user", Content: "Hello!"},
    },
    MaxTokens: axengine.Ptr(64),
})

resp, err := client.Embeddings(ctx, axengine.OpenAiEmbeddingRequest{
    Input:     []int{1, 2, 3},
    Pooling:   axengine.Ptr("last"),
    Normalize: axengine.Ptr(true),
})
```

### Streaming

Both streaming methods return a data channel and an error channel. The data
channel is closed when the stream ends; the error channel carries at most one
error.

```go
// Streaming chat
ch, errCh := client.StreamChatCompletion(ctx, axengine.OpenAiChatCompletionRequest{
    Messages: []axengine.OpenAiChatMessage{
        {Role: "user", Content: "Count from 1 to 5."},
    },
    MaxTokens: axengine.Ptr(64),
})

for chunk := range ch {
    if len(chunk.Choices) > 0 {
        if text := chunk.Choices[0].Delta.Content; text != nil {
            fmt.Print(*text)
        }
    }
}
fmt.Println()

if err := <-errCh; err != nil {
    log.Fatal(err)
}

// Streaming text completion
ch, errCh := client.StreamCompletion(ctx, axengine.OpenAiCompletionRequest{
    Prompt:    "Once upon a time",
    MaxTokens: axengine.Ptr(64),
})
for chunk := range ch {
    fmt.Print(chunk.Choices[0].Text)
}
```

Context cancellation stops the stream and unblocks the channel:

```go
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()
ch, errCh := client.StreamChatCompletion(ctx, req)
```

### Error handling

```go
resp, err := client.ChatCompletion(ctx, req)
if err != nil {
    var httpErr *axengine.HTTPError
    if errors.As(err, &httpErr) {
        fmt.Println("status:", httpErr.Status)
        fmt.Println("message:", httpErr.Message)
    }
    log.Fatal(err)
}
```

### `Ptr` helper

All optional pointer fields can be set with the `Ptr` generic helper:

```go
axengine.Ptr(64)      // *int
axengine.Ptr(0.7)     // *float64
axengine.Ptr(true)    // *bool
axengine.Ptr("last")  // *string
```

## LangChain Integration

ax-engine-server speaks OpenAI-compatible HTTP, so
[langchaingo](https://github.com/tmc/langchaingo) connects to it directly
without a custom adapter — just point the OpenAI provider at the server:

```go
import (
    "github.com/tmc/langchaingo/llms"
    "github.com/tmc/langchaingo/llms/openai"
)

llm, err := openai.New(
    openai.WithBaseURL("http://127.0.0.1:8080/v1"),
    openai.WithToken("not-required"),
    openai.WithModel("qwen3_dense"),
)

// Blocking
resp, err := llms.GenerateFromSinglePrompt(ctx, llm, "Say hello in one sentence.")

// Streaming
llm.GenerateContent(ctx,
    []llms.MessageContent{
        llms.TextParts(llms.ChatMessageTypeHuman, "Count from 1 to 5."),
    },
    llms.WithMaxTokens(64),
    llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
        fmt.Print(string(chunk))
        return nil
    }),
)
```

See `examples/go/langchain/` for a runnable example. To run it:

```bash
cd examples/go/langchain
go mod tidy
go run .
```

## Running Examples

```bash
cd examples/go

# Chat completion (ax-engine SDK)
go run ./chat

# Streaming chat (ax-engine SDK)
go run ./stream

# LangChain via langchaingo (requires: go mod tidy in examples/go/langchain)
cd langchain && go mod tidy && go run .
```

All examples require `ax-engine-server` running on `http://127.0.0.1:8080`.

## Running Tests

```bash
cd sdk/go/axengine && go test ./...
```

Tests cover the SSE parser (`SSEReader`) and helper utilities; they run fully
offline without a server.
