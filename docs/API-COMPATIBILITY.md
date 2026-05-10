# API Compatibility

AX Engine exposes OpenAI-shaped local HTTP endpoints where that shape fits the
current SDK contract. Treat these routes as explicit compatibility contracts,
not as a claim that the entire OpenAI API is implemented.

## Current Server Contract

| Endpoint | Status | Runtime paths | Request scope | Response scope |
|---|---|---|---|---|
| `GET /v1/models` | Preview-compatible model list | All server modes | None | OpenAI-style `object=list`, one local model card, AX runtime metadata |
| `POST /v1/completions` | Preview-compatible text completion | `llama_cpp`, `mlx_lm_delegated` | `prompt` as one string or one token array; string-array batch prompts are rejected until per-prompt result assembly is implemented; `max_tokens` optional, defaults to 256; `temperature`, `top_p`, `seed`, `stream`, `metadata` | OpenAI-style completion envelope or SSE chunks with `system_fingerprint: null`; `usage` only when backend token counts are authoritative |
| `POST /v1/chat/completions` | Preview-compatible text chat completion | `llama_cpp`, `mlx_lm_delegated` | text-only messages; roles `system`, `user`, `assistant`, `tool`, `function`; `max_tokens` optional, defaults to 256; `temperature`, `top_p`, `seed`, `stream`, `metadata` | OpenAI-style chat envelope or SSE chunks with `system_fingerprint: null` after AX renders messages with the selected model-family prompt template |
| `POST /v1/embeddings` | AX embedding route with OpenAI-shaped response | repo-owned MLX embedding-capable sessions | token-array `input`; optional `pooling`, `normalize`, and `encoding_format` placeholder | OpenAI-style embedding list with float vectors and token usage |

## Explicit Non-Goals Today

These are not in the current compatibility contract:

- tool calling / function calling request and response semantics
- JSON mode or structured output validation
- multimodal chat content beyond text parts
- assistant `tool_calls` replay semantics
- batch completion prompt arrays on `/v1/completions`
- full OpenAI parameter parity such as penalties, logprobs, `n`, `stop`, or
  response-format controls
- full tokenizer ownership or arbitrary model chat-template discovery inside
  `ax-engine-server`

Repo-owned MLX generation remains token-first on `POST /v1/generate`. The Rust
server intentionally rejects text/chat OpenAI endpoints for repo-owned MLX
generation instead of inventing tokenizer behavior at the transport layer. The
delegated text routes include a small built-in chat-template registry for common
Qwen ChatML and Llama 3 prompt rendering before forwarding to the configured
text backend.

## JavaScript/TypeScript Surface

The checked-in JavaScript package is currently `@ax-engine/preview-client`.
It is useful for Node.js integration against the local preview server, but it is
not yet a published stable SDK.

The current package has TypeScript declarations and Node tests for the preview
client methods it exposes. A versioned npm release is not part of the current
public contract.

## Tool Calling Status

Tool calling is not in the current compatibility contract. AX Engine accepts
`tool` and `function` as chat roles for text replay, but it does not yet accept
`tools`, `tool_choice`, or assistant `tool_calls` as structured request/response
semantics.

AX Engine should remain an inference runtime. Tool execution, permissions,
network effects, and workflow orchestration belong to caller applications or a
separate control plane.
