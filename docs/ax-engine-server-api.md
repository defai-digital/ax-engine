# AX Engine Server API

This document shows the current request and response surface for
`ax-engine-server`.

Base URL examples below assume:

```text
http://127.0.0.1:3000
```

## Health

```bash
curl http://127.0.0.1:3000/healthz
```

Example response:

```json
{
  "status": "ok",
  "model": "Qwen3-8B-Q4_K_M",
  "architecture": "qwen3",
  "backend": "native",
  "context_length": 4096,
  "vocab_size": 151936,
  "support_note": null,
  "routing": null
}
```

## List Models

```bash
curl http://127.0.0.1:3000/v1/models
```

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-8B-Q4_K_M",
      "object": "model",
      "created": 1774580000,
      "owned_by": "ax-engine",
      "root": "Qwen3-8B-Q4_K_M",
      "backend": "native"
    }
  ]
}
```

## Text Completion

```bash
curl http://127.0.0.1:3000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B-Q4_K_M",
    "prompt": "Explain why local inference benefits from unified memory.",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

Example response:

```json
{
  "id": "cmpl-0000000000000001",
  "object": "text_completion",
  "created": 1774580000,
  "model": "Qwen3-8B-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "text": "Local inference benefits from unified memory because ...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 53,
    "total_tokens": 63
  }
}
```

## Chat Completion

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B-Q4_K_M",
    "messages": [
      { "role": "system", "content": "Answer concisely." },
      { "role": "user", "content": "Summarize AX Engine in two sentences." }
    ],
    "max_tokens": 96,
    "temperature": 0.4
  }'
```

Example response:

```json
{
  "id": "chatcmpl-0000000000000001",
  "object": "chat.completion",
  "created": 1774580000,
  "model": "Qwen3-8B-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "AX Engine is a local Apple Silicon inference engine ..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 31,
    "total_tokens": 53
  }
}
```

## Streaming

Set `"stream": true` to receive Server-Sent Events.

### Streaming Chat

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model": "Qwen3-8B-Q4_K_M",
    "messages": [
      { "role": "user", "content": "Write three short points about on-device AI." }
    ],
    "stream": true
  }'
```

Example event sequence:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1774580000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1774580000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{"content":"Point 1 ..."},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1774580000,"model":"Qwen3-8B-Q4_K_M","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Streaming Completion

```bash
curl http://127.0.0.1:3000/v1/completions \
  -H 'Content-Type: application/json' \
  -N \
  -d '{
    "model": "Qwen3-8B-Q4_K_M",
    "prompt": "Continue: AX Engine is useful because",
    "stream": true
  }'
```

## Request Fields

Supported fields today:

- `model`
- `max_tokens`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repeat_penalty`
- `repeat_last_n`
- `frequency_penalty`
- `presence_penalty`
- `seed`
- `stop`
- `stream`

Model metadata endpoints also expose:

- `backend`
- `routing` when the loaded model is running through `llama.cpp`

Chat requests also accept:

- `messages[].role`
- `messages[].content`

For chat content arrays, only text parts are accepted:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Hello" }
  ]
}
```

## Current Limits

- one loaded model per process
- one active generation at a time
- `n=1` only
- no embeddings endpoint
- no `responses` endpoint yet
- no tool calling or structured output protocol yet
- no persistent server-side session reuse

## Error Shape

Validation and server failures use this envelope:

```json
{
  "error": {
    "message": "human-readable error",
    "type": "invalid_request_error"
  }
}
```

Server-side failures use `type: "server_error"` instead.
