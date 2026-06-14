# API Compatibility

AX Engine exposes OpenAI-shaped local HTTP endpoints where that shape fits the
current SDK contract. Treat these routes as explicit compatibility contracts,
not as a claim that the entire OpenAI API is implemented.

## Current Server Contract

| Endpoint | Status | Runtime paths | Request scope | Response scope |
|---|---|---|---|---|
| `GET /v1/models` | Preview-compatible model list | All server modes | None | OpenAI-style `object=list`, one local model card, AX runtime metadata, conservative `capabilities`, `limit`, `context_length`, `max_output_tokens`, and `ax_engine` integration metadata |
| `POST /v1/completions` | Preview-compatible text completion | repo-owned MLX sessions with tokenizer artifacts, `llama_cpp`, `mlx_lm_delegated` | `prompt` as one string or one token array; string-array batch prompts are rejected until per-prompt result assembly is implemented; `max_tokens` optional, defaults to 256; `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stream`, `metadata`; `response_format` is accepted only as workload metadata | OpenAI-style completion envelope or SSE chunks with `system_fingerprint: null`; `usage` only when backend token counts are authoritative |
| `POST /v1/chat/completions` | Preview-compatible chat completion | repo-owned MLX sessions with tokenizer and supported chat-template artifacts, `llama_cpp`, `mlx_lm_delegated` | text messages; on Gemma 4 unified native MLX sessions also inline base64 `image_url`, `input_audio`/`audio_url` (WAV/MP3), and `video_url` (GIF, plus MP4/WebM when `ffmpeg` is on the server `PATH`) content parts (see `docs/SERVER.md` for budget/duration limits); roles `system`, `user`, `assistant`, `tool`, `function`; `max_tokens` optional, defaults to 256; `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stream`, `metadata`; `tools`, `tool_choice`, and `response_format`; native Qwen ChatML sessions render non-streaming tool schemas into the prompt | OpenAI-style chat envelope or SSE chunks with `system_fingerprint: null` after AX renders messages with the selected model-family prompt template; Gemma 4 thinking-channel framing is stripped from chat content; non-streaming native Qwen tool spans are converted into `message.tool_calls` with `finish_reason=tool_calls` |
| `POST /v1/embeddings` | AX embedding route with OpenAI-shaped response | repo-owned MLX embedding-capable sessions | token-array `input`; optional `pooling`, `normalize`, and `encoding_format` placeholder | OpenAI-style embedding list with float vectors and token usage |

## Explicit Non-Goals Today

These are not in the current compatibility contract:

- JSON mode or structured output validation
- multimodal chat for model families other than Gemma 4 unified, remote
  `http(s)` media URLs, and MP4/WebM inline video when `ffmpeg` is unavailable
- token-incremental tool-call streaming and delegated-backend prompt-side tool
  rendering
- batch completion prompt arrays on `/v1/completions`
- full OpenAI parameter parity such as penalties, logprobs, `n`, `stop`, or
  response-format controls
- full tokenizer ownership or arbitrary model chat-template discovery inside
  `ax-engine-server`

Repo-owned MLX generation remains token-first on `POST /v1/generate`, but the
OpenAI-shaped completion and chat endpoints can tokenize text when the configured
model artifacts include tokenizer files. Chat support is limited to the
server-owned template registry; unsupported model families fail closed instead
of guessing chat-template behavior. Delegated text routes continue to forward
rendered text to their configured upstream backend.

## JavaScript/TypeScript Surface

The checked-in JavaScript package is currently `@ax-engine/sdk`.
It is useful for Node.js integration against the local preview server and for
LangChain-compatible local inference clients.

The current package has TypeScript declarations and Node tests for the preview
client methods it exposes.

## Tool Calling Status

Native Qwen ChatML sessions, including Qwen3-Coder-Next, support a tool-call
contract. AX Engine renders OpenAI `tools` into the Qwen prompt, replays
assistant `tool_calls` history as explicit `<tool_call>{...}</tool_call>` spans,
and parses generated spans back into OpenAI `message.tool_calls` with
`finish_reason=tool_calls`.

AX Engine should remain an inference runtime. Tool execution, permissions,
network effects, and workflow orchestration belong to caller applications or a
separate control plane.

`GET /v1/models` reports `capabilities.toolcall=true` and
`ax_engine.openai_tool_calling_supported=true` only for the native MLX Qwen
ChatML path where prompt-side rendering and response parsing are both present.
Delegated text backends and non-Qwen families continue to report `false`.
Streaming requests with `tools` on the native Qwen path return buffered
OpenAI-shaped SSE chunks with `delta.tool_calls`; they do not stream the tool
call token-by-token yet. Integrations such as `ax-code` should treat these
fields as authoritative instead of assuming generic OpenAI-compatible providers
support tool calls.
