# API Compatibility

AX Engine exposes OpenAI-shaped and Ollama-shaped local HTTP endpoints where
those shapes fit the current SDK contract. Treat these routes as explicit
compatibility contracts, not as a claim that the entire OpenAI or Ollama API is
implemented.

The architectural baseline is still OpenAI-compatible `/v1/*`: coder engines
and agent runtimes should integrate through `/v1/chat/completions`,
`/v1/completions`, and `/v1/embeddings` when they want a provider-neutral wire
format. Ollama `/api/*` routes are runtime-specific compatibility adapters for
local clients that already speak Ollama; they should not own the agent protocol,
tool execution policy, planner state, file editing model, or task memory.

## Current Server Contract

| Endpoint | Status | Runtime paths | Request scope | Response scope |
|---|---|---|---|---|
| `GET /v1/models` | Preview-compatible model list | All server modes | None | OpenAI-style `object=list`, one card per loaded model, AX runtime metadata, conservative `capabilities`, `limit`, `context_length`, `max_output_tokens`, and `ax_engine` integration metadata |
| `POST /v1/completions` | Preview-compatible text completion | repo-owned MLX sessions with tokenizer artifacts, `llama_cpp`, `mlx_lm_delegated` | `prompt` as one string or one token array; string-array batch prompts are rejected until per-prompt result assembly is implemented; `max_tokens` optional, defaults to 256; `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stream`, `metadata`; `response_format` is accepted only as workload metadata | OpenAI-style completion envelope or SSE chunks with `system_fingerprint: null`; `usage` only when backend token counts are authoritative |
| `POST /v1/chat/completions` | Preview-compatible chat completion | repo-owned MLX sessions with tokenizer and supported chat-template artifacts, `llama_cpp`, `mlx_lm_delegated` | text messages; on Gemma 4 unified native MLX sessions also inline base64 `image_url` and `input_audio`/`audio_url` (WAV/MP3) content parts; video is rejected; roles `system`, `user`, `assistant`, `tool`, `function`; `max_tokens` optional, defaults to 256; `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stream`, `metadata`; `tools`, `tool_choice`, and `response_format` | OpenAI-style chat envelope or SSE chunks with `system_fingerprint: null` after AX renders messages with the selected model-family prompt template; Gemma 4 thinking-channel framing is stripped from chat content; non-streaming native Qwen and native Gemma 4 text tool spans are converted into `message.tool_calls` with `finish_reason=tool_calls` |
| `POST /v1/embeddings` | AX embedding route with OpenAI-shaped response | repo-owned MLX embedding-capable sessions | token-array `input` or token-array batch; optional `pooling`, `normalize`, and `encoding_format` placeholder | OpenAI-style embedding list with float vectors and token usage |
| `POST /v1/embedding_records` | AX-native structured RAG ingestion batch | repo-owned MLX embedding-capable sessions with `tokenizer.json` | `records[]` with `text` or deterministic `fields` rendering, optional chunking and metadata | Chunk-level embeddings with stable record/chunk indexes and metadata |
| `GET /api/tags` | Ollama-shaped local model list | All server modes | None | Ollama-style `models` array for the currently loaded AX model |
| `POST /api/show` | Ollama-shaped loaded-model metadata | All server modes | Optional `model` matching the loaded model; `verbose=false` accepted; `verbose=true` and unknown fields fail closed | Ollama-style metadata, `modified_at`, details, model info, and conservative capabilities for the current AX model |
| `GET /api/ps` | Ollama-shaped loaded-model list | All server modes | None | Ollama-style running-model list with the current AX model and advertised `context_length` |
| `GET /api/version` | Ollama-shaped version probe | All server modes | None | `{"version": "<ax-engine-server crate version>"}` for clients that probe Ollama readiness |
| `POST /api/chat` | Ollama-shaped chat adapter | Same runtime paths as `/v1/chat/completions` | Ollama `messages`, `tools`, `format`, `stream`, and `options` mapped onto the OpenAI chat builder; `options.num_predict`, `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`, `seed`, and `stop` are honored | Ollama chat JSON when `stream=false`; Ollama NDJSON chunks when `stream=true` or omitted; native AX-rendered Qwen/Gemma tool calls are returned as Ollama `message.tool_calls` |
| `POST /api/generate` | Ollama-shaped generate adapter | Same runtime paths as `/v1/completions` | Ollama `prompt`, `system`, `format`, `stream`, `keep_alive`, `raw`, and the same supported `options` fields as `/api/chat`; `raw=true` sends the prompt without AX's simple system-prefix wrapper; unsupported `context`, template, image, suffix, think, logprobs, and unknown fields fail closed; empty prompts return Ollama-style load/unload no-op responses | Ollama generate JSON when `stream=false`; Ollama NDJSON chunks when `stream=true` or omitted |

## Explicit Non-Goals Today

These are not in the current compatibility contract:

- JSON mode or structured output validation
- multimodal chat for model families other than Gemma 4 unified, remote
  `http(s)` media URLs, and all video input
- token-incremental tool-call streaming and delegated-backend prompt-side tool
  rendering
- batch completion prompt arrays on `/v1/completions`
- full OpenAI parameter parity such as penalties, logprobs, `n`, `stop`, or
  response-format controls; non-default `n`, `best_of`, `frequency_penalty`,
  `presence_penalty`, and `logit_bias` values fail closed with an
  `unsupported_parameter` error instead of being silently ignored
- client-supplied stop sequences (`stop`, Anthropic `stop_sequences`, Ollama
  `options.stop`) on the repo-owned native MLX backend: only the delegated
  `llama_cpp`/`mlx_lm_delegated` backends forward and honor them today; a
  non-empty stop list on native MLX fails closed with an
  `unsupported_parameter` error instead of being silently ignored. Model-
  family default stop tokens (e.g. Gemma 4's `<end_of_turn>`) are unaffected
  — those are enforced natively via the tokenizer's own EOS token id
- full tokenizer ownership or arbitrary model chat-template discovery inside
  `ax-engine-server`
- full Ollama daemon parity such as model pull/push/create/copy/delete,
  arbitrary Modelfile templates, stateful prompt context replay, thinking
  control, Ollama logprobs, or image payloads on `/api/generate`

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

Native Qwen ChatML and native Gemma 4 text sessions support model-family
tool-call contracts. AX Engine renders OpenAI `tools` into the prompt, replays
assistant `tool_calls` history, and parses generated spans back into OpenAI
`message.tool_calls` with `finish_reason=tool_calls`. Native Qwen ChatML tool
prompting follows the matching Ollama template for the selected model family:
Qwen3 dense uses the JSON
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` contract,
Qwen3.5 uses the function-XML contract, and Qwen3.6 plus Qwen3-Coder-Next use
the Qwen3-Coder XML contract. AX mirrors the selected Ollama-family template
shape: Qwen3.5 renders tool schemas as OpenAI tool JSON lines before asking for
function-XML calls, while Qwen3.6 and Qwen3-Coder render XML tool declarations.
Native
Gemma 4 text chat uses the Ollama/Gemma 4 `<|tool>`, `<|tool_call>`, and
`<|tool_response>` DSL.

AX Engine should remain an inference runtime. Tool execution, permissions,
network effects, and workflow orchestration belong to caller applications or a
separate control plane.

`GET /v1/models` reports `capabilities.toolcall=true` and
`ax_engine.openai_tool_calling_supported=true` only for native MLX families
where prompt-side rendering and response parsing are both present: Qwen ChatML
and Gemma 4 text chat. Gemma 4 tool calling fails closed for delegated,
pre-tokenized, or inline-media chat requests because AX cannot safely inject the
Ollama/Gemma 4 tool DSL into those prompt paths yet.
The `ax_engine` metadata also reports routing intent for clients: Qwen3-Coder
models set `primary_use="coding"`, `coding_only=true`, and `chat_default=false`;
Qwen3.6 models set `primary_use="general"`, `coding_supported=true`, and
`chat_default=true`, so they can remain default chat/general-agent choices while
still using the coding tool contract when tools are requested.
Streaming requests with `tools` on native AX-rendered tool-call paths return
buffered OpenAI-shaped SSE chunks with `delta.tool_calls`; they do not stream
the tool call token-by-token yet. Integrations such as `ax-code` should treat
these fields as authoritative instead of assuming generic OpenAI-compatible
providers support tool calls.

Native MLX OpenAI text/chat requests are tokenized before generation and checked
against the advertised `context_length`. If `prompt_tokens + max_tokens` would
exceed the model context, AX returns `400 context_length_exceeded` instead of
starting a request that cannot terminate.
