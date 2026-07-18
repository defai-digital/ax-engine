# Server

`ax-engine-server` is the local HTTP access layer on top of `ax-engine-sdk`.
For normal end-user serving, prefer `ax-engine serve …`; use
`ax-engine-server` when you need explicit runtime flags.

**Related:** [Getting Started](GETTING-STARTED.md) · [CLI](CLI.md) ·
[API Compatibility](API-COMPATIBILITY.md) · [SDK Docs](sdk/README.md) ·
[LAN Discovery](LAN-DISCOVERY.md)

## Current Scope

The current preview server is intentionally narrow:

- single-process local server
- built entirely on the Rust SDK contract
- fails closed on unsupported hosts (requires M2 Max or newer, macOS 26+, 32 GB RAM)
- explicit backend and support-tier reporting
- preview generation endpoint for bring-up and integration testing
- preview OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
  endpoints for native MLX sessions with tokenizer artifacts and delegated text
  integration
- OpenAI-shaped `/v1/embeddings` response envelopes for embedding-capable
  repo-owned MLX sessions
- Ollama-shaped `/api/tags`, `/api/show`, `/api/ps`, `/api/version`,
  `/api/chat`, and `/api/generate` adapters for local clients that expect
  Ollama HTTP envelopes
- stepwise request lifecycle endpoints that mirror the SDK preview contract for
  repo-owned MLX sessions plus the llama.cpp delegated path
- optional API key authentication for HTTP API routes

Optional LAN discovery:

- `--advertise-lan` publishes DNS-SD `_ax-engine._tcp` so AX Serving agents can
  resolve this server without a hard-coded IP (see [LAN-DISCOVERY.md](./LAN-DISCOVERY.md)).

It is not yet:

- a production server surface
- a remote orchestration layer
- a replacement for broader serving infrastructure

## Endpoints

Current preview endpoints:

- `GET /health`
- `GET /healthz`
- `GET /v1/discovery` (unauthenticated LAN verify document; see [LAN-DISCOVERY.md](./LAN-DISCOVERY.md))
- `GET /metrics`
- `GET /v1/runtime`
- `GET /v1/models`
- `POST /v1/model/load`
- `POST /v1/model/unload`
- `GET /api/tags`
- `POST /api/show`
- `GET /api/ps`
- `GET /api/version`
- `POST /api/chat`
- `POST /api/generate`
- `POST /v1/embeddings`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/requests`
- `GET /v1/requests/:request_id`
- `POST /v1/requests/:request_id/cancel`
- `POST /v1/step` (`?model=<loaded-model-id>` selects a non-default model)
- `POST /v1/generate/stream`
- `POST /v1/generate`

## Authentication

HTTP authentication is disabled by default for local development. To require a
Bearer token on API routes, start the server with `--api-key` or set
`AX_ENGINE_API_KEY`:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --mlx \
  --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts \
  --api-key "$AX_ENGINE_API_KEY" \
  --port 8080
```

When enabled, requests to `/v1/*` routes must include:

```text
Authorization: Bearer <key>
```

`/health` and `/healthz` remain unauthenticated readiness probes so process
supervisors and benchmark harnesses can detect startup and liveness without
holding inference credentials.

`--api-key` covers the optional gRPC adapter (`--grpc-bind-address`) as well:
every RPC must carry the same `authorization: Bearer <key>` metadata, except
the `Health` RPC, which stays unauthenticated to mirror the HTTP probe
exemption. Unauthenticated calls fail with gRPC status `UNAUTHENTICATED`.

## Resource Limits & Rate Limiting

All of the following are opt-in and disabled by default, preserving today's
behavior exactly until an operator configures them. Each flag also accepts an
env var fallback (CLI flag wins when both are set):

| Flag | Env var | Default |
|---|---|---|
| `--max-concurrent-requests <N>` | `AX_ENGINE_MAX_CONCURRENT_REQUESTS` | unlimited |
| `--max-request-body-bytes <N>` | `AX_ENGINE_MAX_REQUEST_BODY_BYTES` | 256 MiB (always enforced, even unset) |
| `--request-timeout-secs <N>` | `AX_ENGINE_REQUEST_TIMEOUT_SECS` | no timeout |
| `--grpc-request-timeout-secs <N>` | `AX_ENGINE_GRPC_REQUEST_TIMEOUT_SECS` | falls back to `--request-timeout-secs` |
| `--rate-limit-rps <N>` | `AX_ENGINE_RATE_LIMIT_RPS` | disabled |
| `--rate-limit-burst <N>` | `AX_ENGINE_RATE_LIMIT_BURST` | defaults to `--rate-limit-rps` when unset |
| `--stream-idle-timeout-secs <N>` | `AX_ENGINE_STREAM_IDLE_TIMEOUT_SECS` | no idle deadline |
| `--stream-max-duration-secs <N>` | `AX_ENGINE_STREAM_MAX_DURATION_SECS` | no hard cap |

Notes:

- The concurrency limit is shared by HTTP and gRPC engine jobs. Generation,
  streaming, embedding, and stepwise requests hold capacity until their
  blocking job is terminal, even after a transport timeout or disconnect.
  Saturated HTTP calls return 429; saturated gRPC calls return
  `RESOURCE_EXHAUSTED`. Health, metrics, and metadata reads do not consume
  engine capacity.
- The HTTP rate limit sheds transport load before handler work and returns a
  distinct 429 message, so operators can distinguish it from engine
  saturation.
- The rate limit is a single global token bucket, not per-client/per-IP —
  the server binds to `127.0.0.1` by default, so this is meant to shed a
  runaway local client rather than police a multi-tenant edge.
- `--request-timeout-secs` bounds time-to-first-byte, not a whole streaming
  response; use `--stream-idle-timeout-secs` / `--stream-max-duration-secs`
  for stream-lifetime deadlines. An idle or over-duration stream ends with an
  `error` SSE event followed by `[DONE]`, the same shape as any other
  mid-stream failure.
- `--grpc-request-timeout-secs` lets the gRPC server (its streaming RPCs can
  legitimately run far longer than typical HTTP calls) diverge from the
  shared HTTP timeout; leaving it unset keeps today's shared-timeout
  behavior.

## Observability

`GET /metrics` serves a Prometheus text exposition with HTTP request counters
(total, in-flight, 2xx/4xx/5xx), gRPC request counters (total, in-flight,
ok/error), shared engine-job admission (`ax_engine_jobs_in_flight`), persistent
generation-worker work (`ax_engine_generation_jobs_pending`), and engine-step
gauges (scheduled requests and tokens, KV block usage, accumulated prefix-cache
hits). The endpoint is read-only: engine-step values are snapshots cached when
generation endpoints drive real steps, never sampled by stepping the engine
from the scrape path. Engine-step gauges appear only after at least one step
has been observed via `POST /v1/step`. Per-model gauges are removed when their
model generation is retired, so an unloaded model cannot leave stale series.
`/metrics` requires the API key when authentication is enabled and never
exposes prompts, outputs, or credentials.

HTTP and gRPC health probes remain healthy while inference is active. They
report unavailable if any loaded model's persistent worker is no longer alive;
normal busy state is exposed through `/slots` and the metrics above. This keeps
readiness aligned with the complete model inventory advertised by the process.

The gRPC ok/error counters have one known gap: `grpc-status` for a
successful unary response and for *all* streaming RPCs lives in HTTP/2
trailers, which are invisible to the tower layer that records these
counters, so those responses are counted as "ok" by default. Request and
in-flight counts are exact regardless.

## OpenAI Surface Extensions

The OpenAI-compatible endpoints accept a few agentic-contract fields in
preview form. Everything below is non-streaming only; streaming requests that
ask for these contracts are rejected with `400 unsupported_parameter` rather
than silently dropped.

- **`logprobs`** (completions and chat): when the engine observed sampled-token
  logprobs, responses carry them in OpenAI-shaped `logprobs` blocks. The
  blocks are all-or-nothing — partially observed values are omitted entirely
  to keep token/logprob arrays aligned. `token` entries are currently token
  ids rendered as strings (not decoded text), and completion `text_offset`
  values are token indices. Field shapes follow OpenAI: chat takes a boolean
  `logprobs`, while legacy completions take an integer where `0` opts into
  sampled-token logprobs. Requests for top-N alternatives (chat
  `top_logprobs > 0`, completions `logprobs > 0`) are rejected with
  `400 unsupported_parameter` until the runner emits them.
- **`reasoning`** (chat): opt-in. When set, known model-family reasoning is
  split into `message.reasoning_content`: Qwen `<think>…</think>` text and
  Gemma 4 thinking channels (extracted token-level during native decode).
  Unknown formats fail closed — the text is left in `content` untouched.
  Without the opt-in, responses keep their existing default behavior.
  Streaming: native Qwen ChatML and Gemma 4 chat streams emit incremental
  `delta.reasoning_content` fragments (Qwen via text-level `<think>` scanning
  with marker holdback; Gemma 4 via the channel-token filter) interleaved
  before/alongside content deltas; other families and delegated backends
  keep rejecting `reasoning` + `stream` with `400 unsupported_parameter`.
- **`usage.prompt_tokens_details.cached_tokens`**: when the engine served
  part of the prompt from the prefix cache, non-streaming responses report
  the reused token count in the OpenAI prompt-caching shape; the block is
  omitted when reuse was zero or unknown.
- **`response_format: json_object`** (completions and chat): non-streaming
  responses are validated server-side; output that is not a JSON object
  returns `502 invalid_output`. This is post-hoc validation, not constrained
  decoding.
- **`response_format: json_schema`** (completions and chat): the OpenAI
  `{"type":"json_schema","json_schema":{"name","schema","strict"?}}` shape is
  accepted for non-streaming requests and the output is validated server-side
  against the schema; mismatches return `502 invalid_output` with the first
  failing path. The supported schema subset is `type` (including type
  arrays), `properties`, `required`, `additionalProperties` (boolean),
  `items`, `enum`, `const`, numeric bounds
  (`minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`), and
  string/array length bounds; schemas using any other keyword are rejected
  up front with `400 unsupported_json_schema` naming the keyword — never
  silently ignored. Like `json_object`, this is post-hoc validation, not
  constrained decoding: output is checked, not guaranteed.
- **`stop`** (completions and chat, all backends): client stop sequences are
  honored everywhere. Delegated backends forward them upstream; the native
  MLX backend enforces them server-side over decoded text with OpenAI
  semantics (output truncated at the earliest match, matched text excluded,
  `finish_reason:"stop"`). At most 4 sequences, each non-empty and at most
  256 bytes (`400 invalid_request` otherwise). Native streaming withholds at
  most `max(stop length) − 1` bytes so a match split across chunks never
  leaks, and a match ends the stream and cancels the underlying generation.
  Native non-streaming truncates after generation completes (matched-onward
  text is discarded but its compute is not saved yet). Stops match visible
  assistant content only — tool-call spans are exempt, so a stop string
  occurring inside call arguments cannot corrupt a tool call. Sampled
  logprobs are omitted from a stop-truncated response because they would
  misalign with the truncated text. The Anthropic surface reports the
  matched sequence as `stop_sequence` with `stop_reason:"stop_sequence"`.
  gRPC requests do not get server-side native stop enforcement yet.
- **`tools` / `tool_choice`** (chat): experimental. Native Qwen ChatML and
  native Gemma 4 text sessions render tool schemas into the prompt, replay
  assistant `tool_calls` history, and parse generated spans back into
  `message.tool_calls` with `finish_reason=tool_calls`. Native Qwen ChatML tool
  prompting follows the matching Ollama template for the selected model family:
  Qwen3 dense uses the JSON
  `<tool_call>{"name": ..., "arguments": ...}</tool_call>` contract,
  Qwen3.5 uses the function-XML contract, and Qwen3.6 plus Qwen3-Coder-Next
  use the Qwen3-Coder XML contract. AX mirrors the selected Ollama-family
  template shape: Qwen3.5 renders tool schemas as OpenAI tool JSON lines before
  asking for function-XML calls, while Qwen3.6 and Qwen3-Coder render XML tool
  declarations.
  Gemma 4 text chat uses the Ollama/Gemma 4 `<|tool>`, `<|tool_call>`, and
  `<|tool_response>` DSL. Gemma 4 tools still fail closed for delegated,
  pre-tokenized, and inline-media chat requests because AX cannot safely inject
  the model-specific tool DSL into those prompt paths yet.
  `/v1/models` exposes the intended routing metadata in `ax_engine`: Qwen3-Coder
  models report `primary_use="coding"`, `coding_only=true`, and
  `chat_default=false`; Qwen3.6 models report `primary_use="general"`,
  `coding_supported=true`, and `chat_default=true`.
  Streaming: native Qwen ChatML and Gemma 4 chat streams emit tool calls
  incrementally — content outside tool-call spans streams live, and each
  completed call arrives as one `delta.tool_calls` fragment carrying its
  0-based stream-wide `index`, `id`, `type`, `function.name`, and the full
  `function.arguments` string, with `finish_reason:"tool_calls"` in the final
  chunk. Marker text withheld during scanning never leaks into content, and
  unparseable tool-ish spans fall back to content deltas. Argument text is
  not streamed token-by-token — a call is emitted once complete. GLM 4.x and
  GPT-OSS tool streams keep the previous buffered single-chunk behavior
  (their tool markers do not survive the incremental stream decode).
  Bare JSON answers are never reinterpreted as tool calls.
- **Context limit preflight** (native MLX OpenAI text/chat): AX tokenizes the
  rendered prompt before generation and rejects
  `prompt_tokens + max_tokens > context_length` with
  `400 context_length_exceeded`. This catches oversized tool prompts before they
  reach the runtime terminate guard.

## Ollama Surface

AX also exposes a focused Ollama-shaped adapter for loaded local models:

- `GET /api/tags` returns an Ollama-style `models` list containing every loaded
  AX model.
- `POST /api/show`, `GET /api/ps`, and `GET /api/version` provide the
  Ollama-style metadata/readiness probes common clients use before issuing a
  chat request. `/api/show` accepts `verbose=false` as a probe shape, but
  `verbose=true` fails closed until AX can return the larger verbose Ollama
  metadata payload.
- `POST /api/chat` accepts Ollama text `messages`, `tools`, `format`, `stream`,
  and common `options` fields. It maps them onto the same chat builder used by
  `/v1/chat/completions`, so supported Qwen/Gemma templates and tool-call
  parsing stay identical across the OpenAI and Ollama surfaces.
- `POST /api/generate` accepts Ollama `prompt`, optional `system`, `format`,
  `stream`, `raw`, and common `options` fields, then maps them onto the same
  completion builder used by `/v1/completions`. When `raw=true`, AX sends the
  prompt without its simple system-prefix wrapper.

OpenAI-compatible `/v1/*` remains the recommended baseline for coder engines
and provider-neutral applications. The Ollama `/api/*` surface is a
runtime-specific adapter for existing local clients; it must not own agent
state, tool execution, file editing, approval policy, memory, or planning.

Ollama `stream` defaults to `true`, matching Ollama's API. AX returns
newline-delimited JSON with `application/x-ndjson`; the first chunk carries the
buffered text or tool-call message and the final chunk carries `done=true` plus
available token counts. This is an Ollama envelope compatibility layer, not a
full Ollama daemon: model pull/push/create/copy/delete, arbitrary Modelfile
templates, stateful prompt context replay, `/api/generate` images, Ollama
thinking/logprob controls, and other unsupported fields fail closed with
`400 unsupported_parameter` instead of being ignored. Harmless Ollama lifecycle
fields such as `keep_alive` are accepted as no-ops, and empty `/api/generate`
prompts return Ollama-style load/unload no-op responses.

## Examples

`ax-engine-server` exposes three explicit runtime paths:

- `--mlx` selects the repo-owned MLX runtime for supported local model
  artifacts
- `--support-tier mlx-lm-delegated` delegates text generation to a
  user-provided `mlx_lm.server` while preserving AX blocking, SSE, and
  OpenAI-compatible text surfaces
- `--support-tier llama-cpp` or a GGUF target delegates non-MLX inference to
  llama.cpp

Runtime metadata continues to report SDK labels such as `mlx_lm_delegated` and
`llama_cpp`; the `ax-engine-server` CLI uses hyphenated Clap values such as
`mlx-lm-delegated` and `llama-cpp`.

Retired AX native mode is not a supported user-facing server mode.

Start the repo-owned MLX path against local MLX model artifacts:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --mlx \
  --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts \
  --port 8080
```

`--speculation-profile {auto,coding,agentic,chatbot}` (short `-s`, alias
`--spec`; env `AX_MLX_SPECULATION_PROFILE`) bundles the MTP and n-gram
speculative-decode configuration into one posture (ADR-022). `auto` is the
default and is temperature-driven: it keeps the shipped gate at low temperature
and raises it for higher-temperature sampled chat to protect reply diversity.
`coding`/`agentic` defer to the shipped gate defaults (the Gemma ablation found
lowering the gate does not add code throughput); `chatbot` raises the gate and
prefers the n-gram utility gate. Explicit per-knob env vars still override the
profile, and the resolved posture is reported in route metadata as
`ax_mlx_speculation_profile`.

The experimental MLX compressed-KV runtime path and its
`--experimental-mlx-kv-compression` server flags were retired in favor of the
durable tiered prefix cache (ADR-002). Native uncompressed KV is the only
decode behavior, and the server emits no compressed-KV route metadata.

For common repo-owned MLX targets, use a preset to select the model id, MLX
runtime, support tier, and current safe defaults while keeping model artifacts
explicit:

```text
cargo run -p ax-engine-server -- \
  --preset gemma4-e2b \
  --mlx-model-artifacts-dir /absolute/path/to/gemma-4-e2b-it-4bit \
  --port 8080
```

`--list-presets` prints the built-in preset names. Presets do not download
weights and do not silently scan local caches. If the model directory is already
available through `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`, the explicit path flag can
be omitted. Current native MLX built-ins are `gemma4-e2b`, `gemma4-12b`,
`gemma4-31b`, `glm4.7-flash-4bit`, `qwen3.5-9b`, and `qwen3.6-35b`.

The `glm4.7-flash-4bit` preset (GLM-4.7 Flash, `glm4_moe_lite`) uses the
repo-owned native MLX graph with Flash MLA attention and sigmoid-routed MoE.
It can optionally be served through `mlx_lm_delegated` by passing
`--mlx-lm-server-url`; see [`SUPPORTED-MODELS.md`](SUPPORTED-MODELS.md).

Hugging Face cache discovery is opt-in:

```text
cargo run -p ax-engine-server -- \
  --preset gemma4-e2b \
  --resolve-model-artifacts hf-cache \
  --port 8080
```

By default the resolver searches `HF_HUB_CACHE`, `HF_HOME/hub`, and
`~/.cache/huggingface/hub`; pass `--hf-cache-root <path>` to pin a specific
cache root. Resolution succeeds only when exactly one matching AX-ready artifact
directory is found. The directory must contain `config.json`,
`model-manifest.json`, and safetensors, and its `model_type` must match the
preset. Plain Hugging Face snapshots without `model-manifest.json` fail closed;
generate the manifest first with:

```text
cargo run -p ax-engine-core --bin generate-manifest -- /absolute/path/to/model
```

The preview server requires a local Apple M2 Max-or-newer host running
macOS 26 (Tahoe) or later with 32 GB RAM minimum.
On M1 Macs or unsupported configurations, startup fails closed instead of
exposing an unsupported partial runtime.

Use the llama.cpp delegated path against a local GGUF model:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --llama-cli-path llama-cli \
  --llama-model-path /absolute/path/to/model.gguf \
  --port 8080
```

OpenAI-compatible delegated routes such as `vLLM` and `mistral.rs` are not part
of the shipping inference contract. Use the `llama.cpp` server route for
GGUF/non-MLX server-backed inference:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier llama-cpp \
  --llama-server-url http://127.0.0.1:8081 \
  --port 8080
```

For the default route and the explicit llama.cpp route, choose exactly one
primary target:

- `--llama-server-url` for the server-backed `llama.cpp` `/completion` adapter
- `--llama-model-path` plus `--llama-cli-path` for the local CLI fallback
  adapter

Server-backed delegated HTTP calls use explicit timeouts. The defaults are
30 seconds to connect and 300 seconds for read/write I/O; tune them with
`--delegated-http-connect-timeout-secs`,
`--delegated-http-read-timeout-secs`, and
`--delegated-http-write-timeout-secs` when the upstream backend is remote,
slow-starting, or behind a proxy. Timeout values must be greater than zero.

Local GGUF paths are treated as `llama.cpp` targets. Use `--mlx` when you want
repo-owned MLX inference.

To keep using AX server surfaces for an MLX text model that the repo-owned MLX
runtime does not yet support, run `mlx_lm.server` yourself and select the
explicit delegated backend:

```text
mlx_lm.server --model /absolute/path/to/mlx-model --host 127.0.0.1 --port 8090

cargo run -p ax-engine-server -- \
  --model-id local-mlx-model \
  --support-tier mlx-lm-delegated \
  --mlx-lm-server-url http://127.0.0.1:8090 \
  --port 8080
```

`mlx_lm_delegated` is text-only and delegates model execution to an explicitly
configured upstream `mlx_lm.server`. It is not a repo-owned MLX performance
claim, and it is not a visual/multimodal contract. Streaming responses on this
route are delegated text deltas through AX's HTTP/SSE surfaces, not AX-owned
token IDs or KV-cache evidence.

The preview server exposes OpenAI-compatible text endpoints:

- `POST /v1/completions`
- `POST /v1/chat/completions`

For repo-owned MLX sessions, these endpoints require an explicit MLX model
artifact directory with tokenizer files. Chat requests also require a supported
server-owned chat-template family. Unsupported families fail closed rather than
guessing tokenizer or template behavior. Delegated `llama_cpp` and
`mlx_lm_delegated` routes keep forwarding rendered text to their configured
upstream backend.

`GET /v1/models` advertises image and audio input only for repo-owned
native MLX sessions whose `model-manifest.json` contains the converted Gemma4
unified media tensor roles. Two input shapes are accepted on those sessions:

- **Inline media on chat.** `POST /v1/chat/completions` accepts OpenAI-style
  content parts with base64 `data:` URIs: `image_url` (PNG/JPEG) and
  `input_audio` / `audio_url` (WAV or MP3). The server decodes
  and preprocesses media into Gemma4 unified soft-token spans and tensors.
  Remote `http(s)` media URLs are rejected; callers must inline base64 data.
- **Processed tensors.** OpenAI completions and chat also accept
  `multimodal_inputs.gemma4_unified` tensors directly, but only when the
  caller supplies AX tokenized prompt IDs (`prompt` token arrays or
  `input_tokens`) so media placeholder tokens and tensors stay aligned.

Delegated `llama_cpp` and `mlx_lm_delegated` routes still fail closed on any
multimodal input.

Multimodal serving contract limits:

- **Prompt budget.** Multimodal prefill is atomic: the expanded prompt
  (media soft tokens included) must fit within `--max-batch-tokens`
  (default 2048) in one scheduler step. Over-budget requests fail with an
  actionable HTTP 400 instead of being scheduled. Under concurrent load a
  fitting request may wait for a step with enough budget; it is never split.
- **Token budgets.** Image soft-token budgets come from the model's
  `preprocessor_config.json` (Gemma 4 12B: up to 280 soft tokens per image);
  there is no per-request budget or quality override.
- **Audio.** WAV and MP3 input is downmixed to mono and resampled to the
  model rate (16 kHz). The container is sniffed from magic bytes, not the
  declared `format` field. Other formats (AAC/OGG/FLAC) are rejected; send
  pre-computed audio tensors via `/v1/generate` instead. Audio longer than
  the model's `audio_seq_length` cap (750 frames × 40 ms = 30 s by default)
  is silently truncated, and MP3 decoding stops at that cap.
- **Video.** Public server routes reject video with `unsupported_modality`.
  Lower-level Gemma4 preprocessing primitives remain internal compatibility
  code, but video is not advertised by `/v1/models`.
- **Caching.** Prefix caching is disabled for multimodal requests; every
  multimodal prefill recomputes the full prompt with that request's own
  media tensors.
- **Chat output.** Gemma 4 thinking-channel framing (`<|channel>thought…`)
  is stripped from chat content; raw `/v1/completions` output stays
  verbatim.

`scripts/qa_gemma4_multimodal.py --strict` runs an end-to-end probe set
(image color, image description, audio smoke, speech transcription) against a
live server and fails on thinking-channel leaks or content
mismatches.

You can also run the optional Python OpenAI shim with an explicit MLX model
artifact directory and tokenizer:

```text
python -m ax_engine.openai_server \
  --model-id qwen3_dense \
  --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts \
  --tokenizer /absolute/path/to/tokenizer.json \
  --port 8080
```

Install the optional server dependencies with `pip install 'ax-engine[openai]'`.

See [API Compatibility](API-COMPATIBILITY.md) for the exact OpenAI-shaped endpoint matrix,
including supported request fields, runtime paths, and non-goals such as tool
calling.

To run a repo-owned end-to-end smoke check that starts the preview binary and
exercises health, runtime, one-shot generate, cancel, and SSE streaming over
real HTTP:

```text
bash scripts/check-server-preview.sh
```

To run a direct native-MLX model compatibility smoke for the coder-facing
Qwen3-Coder-Next, Qwen3.6 35B-A3B, and Gemma 4 routes, provide local AX model
artifacts and run:

```text
AX_ENGINE_QWEN_CODER_NEXT_ARTIFACTS_DIR=/absolute/path/to/qwen3-coder-next-artifacts \
AX_ENGINE_QWEN36_35B_ARTIFACTS_DIR=/absolute/path/to/qwen3.6-35b-a3b-artifacts \
AX_ENGINE_GEMMA4_ARTIFACTS_DIR=/absolute/path/to/gemma4-artifacts \
python3 scripts/check_direct_model_compat_smoke.py
```

The check starts `ax-engine-server` with `--mlx`, verifies `/health` selected the
native MLX backend, verifies `/v1/models` advertises tool-call support, then
sends equivalent tool-enabled requests through OpenAI
`/v1/chat/completions` and Ollama `/api/chat`. It fails if either surface leaks
raw tool-call markup instead of returning a normal response envelope. Add
`--expect-tool-call` when the local model/configuration is expected to choose an
actual parsed tool call for the smoke prompt. Without configured artifacts the
script prints a JSON `skipped` result and exits zero.

To run the optional OpenWebUI integration smoke, start or provide an
OpenAI-compatible AX endpoint, then opt in explicitly:

```text
AX_OPENWEBUI_E2E=1 \
AX_OPENWEBUI_AX_BASE_URL=http://127.0.0.1:8080/v1 \
AX_OPENWEBUI_MODEL_ID=qwen3_dense \
bash scripts/check-openwebui-e2e.sh
```

The check starts OpenWebUI from
`ghcr.io/open-webui/open-webui:main` in Docker with an ephemeral data directory,
configures it to proxy to AX, verifies the model is visible through
`/openai/v1/models`, sends a chat completion through
`/openai/v1/chat/completions`, and fails on backend disconnect text or obvious
corruption patterns such as repeated punctuation-only lines. It is skipped
unless `AX_OPENWEBUI_E2E=1` is set; once opted in, missing Docker is a failure
because otherwise the integration gate would silently pass without running. Real
MLX coverage also requires local model artifacts.

For a fully self-contained local run through the Python MLX OpenAI shim:

```text
AX_OPENWEBUI_E2E=1 \
AX_OPENWEBUI_START_AX_SHIM=1 \
AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR=/absolute/path/to/mlx-model-artifacts \
AX_OPENWEBUI_TOKENIZER=/absolute/path/to/tokenizer.json \
AX_OPENWEBUI_MODEL_ID=qwen3_dense \
bash scripts/check-openwebui-e2e.sh
```

## Tracing

`ax-engine-server` supports `tracing`, but it keeps it opt-in so normal runs do
not pay steady-state formatting and subscriber overhead.

To enable server or core diagnostics, set `AX_ENGINE_SERVER_LOG` or `RUST_LOG`
before starting the server. For example:

```text
AX_ENGINE_SERVER_LOG=ax_engine_server=info,ax_engine_core=debug cargo run -p ax-engine-server -- --model-id qwen3_dense --mlx --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts --port 8080
```

For manual throughput or latency measurements, leave tracing disabled, or use
an `info` or `warn` filter instead of `debug` / `trace`. For comparable
repo-owned MLX inference numbers, use `scripts/bench_mlx_inference_stack.py`;
it starts the server, captures AX SSE `runner_time_us`, runs the required
matching `mlx_lm.benchmark` baseline, writes the canonical random-token prompt
artifacts, and records the MLX reference runtime identity explicitly. Use
`ax-engine-bench` for workload-contract artifacts rather than manual server
timing.

Inspect runtime metadata:

```text
curl http://127.0.0.1:8080/v1/runtime
```

Run a preview generation request:

```text
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4,
    "sampling": {
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0,
      "seed": 1234
    }
  }'
```

When the server is running on the repo-owned MLX path, the `.gguf` delegated
llama.cpp path, or an explicit llama.cpp server path, the same blocking endpoint
accepts `input_text`:

```text
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_text": "Hello from llama.cpp",
    "max_output_tokens": 32
  }'
```

When llama.cpp is backed by `--llama-server-url`, the blocking endpoint
also accepts pre-tokenized `input_tokens` and forwards them to
`llama.cpp /completion` as token-array prompts:

```text
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 32
  }'
```

The same `--llama-server-url` path also supports the stateless streaming
endpoint and maps `llama.cpp /completion` SSE chunks back into the SDK-owned
`request` / `step` / `response` event shape:

```text
curl -N http://127.0.0.1:8080/v1/generate/stream \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 32
  }'
```

Stream preview lifecycle events over SSE:

```text
curl -N http://127.0.0.1:8080/v1/generate/stream \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4
  }'
```

To call the same llama.cpp-backed path through the preview
OpenAI-compatible completions endpoint:

```text
curl http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "prompt": "Hello from OpenAI-compatible preview",
    "max_tokens": 32
  }'
```

If `max_tokens` is omitted on OpenAI-compatible completion or chat routes, the
server applies the preview default of 256 generated tokens. `/v1/completions`
accepts one string prompt or one token-array prompt per request; string-array
batch prompts fail closed until the server can return one independent choice per
input prompt.
For native MLX Gemma4 unified models, token-array completion prompts can also
carry processed `multimodal_inputs.gemma4_unified` tensors. Text prompts with
`multimodal_inputs` fail closed because Gemma4 media spans must point at
absolute positions in the already-expanded prompt tokens.

To stream OpenAI-compatible completion chunks instead:

```text
curl -N http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "prompt": "Hello from OpenAI-compatible preview",
    "max_tokens": 32,
    "stream": true
  }'
```

To use the preview chat-completions bridge, send text-only chat messages. AX
renders them with the configured model-family template, for example Qwen ChatML
for Qwen model ids, before forwarding the text prompt to the delegated backend:

```text
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "messages": [
      {
        "role": "user",
        "content": "Say hello from AX"
      }
    ],
    "max_tokens": 32
}'
```

For native MLX Gemma4 unified models, `/v1/chat/completions` also accepts the
AX extension `input_tokens` plus processed `multimodal_inputs.gemma4_unified`.
When `input_tokens` is present, AX validates the message roles and still rejects
raw image/audio content parts, then uses the supplied prompt tokens
directly so media spans remain aligned. Delegated chat backends reject both
`input_tokens` and `multimodal_inputs`.

For embedding-capable repo-owned MLX sessions, the server also exposes an
OpenAI-shaped embedding response over token-array input:

```text
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_embedding",
    "input": [1, 2, 3, 4],
    "pooling": "last",
    "normalize": true
  }'
```

Embedding requests use a lightweight micro-batching worker by default. Multiple
single-request `/v1/embeddings` calls that arrive within a short window are
grouped and run through one `embed_batch` call when pooling and normalization
options match. Tune it with:

- `AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS` (default `2`)
- `AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH` (default `32`)

Set `AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH=1` to disable grouping for
diagnostics.

Submit a request into the selected model's preview session:

```text
curl http://127.0.0.1:8080/v1/requests \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4
  }'
```

Advance the default model's preview session by one engine step, or select a
loaded model explicitly:

```text
curl -X POST http://127.0.0.1:8080/v1/step
curl -X POST 'http://127.0.0.1:8080/v1/step?model=gemma-4-12b-it'
```

Inspect or cancel a submitted request:

```text
curl http://127.0.0.1:8080/v1/requests/1
curl -X POST http://127.0.0.1:8080/v1/requests/1/cancel
```

The response includes:

- generated token ids
- finish status
- OpenAI-compatible text or chat response envelopes on `/v1/completions` and
  `/v1/chat/completions`
- request-state snapshots and step metrics for the stepwise lifecycle endpoints
- SSE lifecycle events for the preview streaming endpoint
- route metadata observed during execution
- MLX-mode `/v1/step` responses now also surface a compact `metal_dispatch`
  summary when the shared session is running the MLX Metal bring-up
  runner, so transport clients can inspect command-buffer completion,
  pipeline/archive state, arena sizing, checksum/validation evidence, and
  whether the step actually ran with model-conditioned / real-model tensor
  inputs, plus whether that runtime/source path supports complete
  model-forward execution and whether the step resolved decode logits / reached
  completed real-model forward execution, together with native-vs-CPU prefix
  dispatch counts for multilayer forward coverage review, model-side
  RMSNorm+QKV / o_proj+FFN token coverage, and final logits projection plus
  vocab-scan counts
  without reopening benchmark artifacts
- backend resolution metadata such as `selected_backend`, `support_tier`,
  `resolution_policy`, capability reporting, host diagnostics, and Metal
  toolchain availability

Repo-owned MLX generation, streaming, stepwise lifecycle calls, and embeddings
all use one persistent worker-owned `EngineSession`. The server loads native
weights once at startup by constructing the session inside that worker, submits
every native request into that session, and
keeps request KV/cancellation ownership on the same worker through terminal
cleanup. Unary generation is collected from the same event stream used by SSE
and gRPC; it does not construct a request-local model session. Delegated
`llama_cpp` and `mlx_lm_delegated` requests keep their stateless adapter paths.
All paths share one process-local request-id sequence.

The persistent session lets the scheduler observe overlapping native requests.
Runner-level fused batching remains separately gated: unsupported model
families still execute scheduler batch items individually and must not be
described as continuous-batching speedups without server-path evidence.
For native MLX Gemma4 unified models, `/v1/generate` and
`/v1/generate/stream` also accept preprocessed
`multimodal_inputs.gemma4_unified` image/audio tensors. AX does not
decode raw image/audio URLs or files on this path; callers must run media
loading and processor output generation before sending the native request.
The Python SDK helper can perform that client-side preprocessing for image
paths/URLs/data URIs, WAV audio paths/URLs/data URIs, and OpenAI-style
`input_audio` WAV base64 dictionaries.
Native MLX OpenAI-compatible completions and chat can also carry processed
`multimodal_inputs` when the prompt is already tokenized: completions use a
token-array `prompt`, while chat uses AX's `input_tokens` extension. Text
prompts with processed media tensors are rejected because Gemma4 unified media
spans are absolute positions in the expanded token sequence.
The server validates processed tensor span bounds, modality labels, soft-token
counts, and tensor lengths before scheduling the request; malformed inputs
return `invalid_request`.
`POST /v1/generate/stream` uses the same stateless request shape but streams
preview SSE events named `request`, `step`, `response`, and `error`.
`POST /v1/completions` and `POST /v1/chat/completions` are response-shape
adapters over the same stateless request flow. Native MLX requests tokenize text
through the configured model artifacts unless callers supply pre-tokenized
OpenAI completion prompts or chat `input_tokens`; delegated `llama_cpp` and
`mlx_lm_delegated` requests forward text to the configured upstream backend.
Streaming mode emits unnamed SSE `data:` chunks plus `[DONE]` in the familiar
OpenAI-style envelope instead of AX-specific lifecycle event names.
OpenAI-shaped completion and chat responses include `system_fingerprint: null`
because the preview server does not yet publish a stable backend fingerprint.
The `/v1/requests` endpoint routes by `model`; `/v1/step` advances the default
model unless its optional `model` query selects another loaded model. Request
snapshot and cancellation use a bounded, generation-aware ownership index
populated at submission, so lifecycle operations route directly to the owning
worker rather than scanning or waiting behind unrelated model workers. Retiring
a generation removes its ownership entries, and terminal ownership follows the
SDK's bounded terminal-report retention while preserving the lifecycle contract.
The server allocates request ids from one process-local sequence across both
paths so transport logs and client correlation do not collide when clients mix
blocking and stepwise APIs.

`POST /v1/model/load` closes native admission before changing the process-local
model registry. It rejects a change while non-terminal stepwise work exists and
waits for generation, streaming, and embedding permits to reach zero. A
disconnected load client does not cancel cleanup. Requests arriving during the
drain receive HTTP 503 or gRPC `UNAVAILABLE` and can retry after loading.

`load_mode` accepts `replace` (default, historical hot-swap semantics) or `add`.
`add` retains the existing independently-owned model sessions, makes the new
model the default for requests that omit `model` (opt out with
`"make_default": false` — `replace` swaps the default in place and rejects
`false`), and enables per-request model routing. Add mode is limited to
Qwen 3.6 35B/27B and Gemma 4 12B/26B/31B. Use `POST /v1/model/unload` with
`{"model_id":"..."}` to retire a retained model; the last loaded model cannot
be unloaded. Unloading the current default reassigns the default to another
resident model; both load and unload responses report the resulting
`default_model_id` so clients can observe the reassignment. `add` rejects a model ID that is
already loaded. Once the registry contains more than one model, `replace` also
enforces the same five-model scope because it retains the other sessions; an
unrestricted historical hot-swap remains available when only one model is
loaded. Invalid load and unload requests are rejected before admission drains,
so they do not wait for or interrupt unrelated active traffic.
The retained `model-manifest.json` is authoritative for this allowlist: AX
validates the manifest's exact family and architecture signature against the
requested model ID. Directory names, optional Hugging Face config metadata, and
substring matches cannot admit an unsupported or mismatched model.

Each retained model still owns an independent session and scheduler. A shared
process execution arbiter grants one model turn per engine step (and per
embedding execution), rotating fairly among waiting model IDs. This preserves
single-model execution on the process-wide MLX/Metal target without letting one
busy model monopolize every turn; it does not create a cross-model fused batch.
Because a turn is one engine step, a sibling model's decode waits while a long
prefill step runs (up to `--max-batch-tokens` prompt tokens, default 2048 —
roughly 1-2 s on the supported model sizes). For latency-sensitive multi-model
serving, shrink the per-step budget with `--max-batch-tokens` (for example
512), or enable `--multi-prefill-fair` with
`--max-prefill-tokens-per-request-per-step` to chunk prefills per request
without shrinking the global step budget (the per-request cap is honored only
in fair mode). Either trades some prefill throughput for decode turns
interleaving between the smaller chunks.
The optional `load_policy` request field accepts `availability_first` (the
default) or `memory_constrained`. Availability-first keeps the drained model
resident until its replacement is ready, preserving rollback at the cost of a
temporary two-model memory peak. Memory-constrained shuts down and joins the old
generation worker before loading the replacement; this lowers peak memory but
intentionally leaves the server unavailable if replacement loading fails.
Inference remains unavailable in that state, but `/v1/model/load` stays usable
so an operator can retry with valid artifacts and recover without restarting
the server process.
`load_mode=add` is incompatible with `memory_constrained`, because retaining
existing models necessarily preserves their resident memory.

The optional `--model-idle-timeout-secs` flag (env
`AX_ENGINE_MODEL_IDLE_TIMEOUT_SECS`) enables idle eviction for multi-model
serving: a background sweep retires non-default resident models that have not
admitted a request within the timeout, following the same drain/retire flow
as `POST /v1/model/unload`. The default model is never evicted, and a sweep
only runs while the server is otherwise idle (the unload flow drains global
admission, so evicting during active traffic would stall unrelated
requests). Disabled when unset.

Engine-step metrics on `/metrics` (`ax_engine_steps_total`,
`ax_engine_step_scheduled_requests`, `ax_engine_step_scheduled_tokens`,
`ax_engine_step_kv_usage_blocks`, `ax_engine_step_prefix_hits_total`) are
emitted per loaded model with a `model` label, plus an unlabeled aggregate
series (summed across models) for dashboards written against the
single-model layout. Unload and replacement remove the retired generation's
per-model sample immediately.

Load requests also run a memory admission preflight before any drain: AX
estimates each model's resident footprint from its on-disk safetensors bytes
(plus a runtime factor) and its worst-case KV pool — derived from the
manifest's attention geometry (full-attention layers at the configured
`total_blocks × block-size` pool; sliding-window layers bounded at their
ring window; hybrid linear-attention layers and KV-shared layers charge no
per-token cache) — plus a fixed runtime floor. It adds the resident models
that survive the load (including the outgoing model for availability-first
replaces, whose overlap window is the policy's point), and rejects the load
with `422 insufficient_memory` — reporting the projected peak, resident,
incoming (with its KV share), and budget numbers in GiB — when the projected
peak exceeds the Metal working-set budget (the same
`max_recommended_working_set_size` source the runner uses for its wired
limit). Models whose KV geometry is unknowable (MLA latent caches,
diffusion, unreadable manifests) fall back to the flat-floor estimate; the
check skips rather than blocks when the budget or the incoming weight layout
is unknowable, and `AX_SERVER_LOAD_MEMORY_PREFLIGHT=off` disables it
entirely if the estimate is wrong for a host.
The `mlx_lm_delegated` backend supports blocking `/v1/generate` and SSE
`/v1/generate/stream` through `mlx_lm.server` `/v1/completions`. It also
supports streamed OpenAI-compatible completion/chat endpoints by forwarding
upstream text deltas through AX response envelopes. These streams are
compatibility surfaces: AX does not tokenize the upstream text response and does
not claim AX-owned token IDs, KV state, or MLX kernel throughput for this route.
Delegated routes are text-only and return an invalid request when
`multimodal_inputs` is present.

For delegated text responses, `output_text` is authoritative. `output_tokens`
is intentionally empty because AX did not tokenize the upstream text response;
use `output_token_count` when `mlx_lm.server` reports usage. EOS/stop handling
for this path follows the upstream `finish_reason`. Repo-owned MLX token
requests stop on configured EOS token IDs in the model artifacts; raw
`/v1/completions` prompts are not chat-templated by AX, so short instruction
prompts may still finish by `max_output_tokens` unless the upstream backend
emits a stop finish reason.

For Phase 1, the llama.cpp backend supports blocking `/v1/generate`,
OpenAI-compatible `/v1/completions`, and OpenAI-compatible
`/v1/chat/completions`.
The server-backed llama.cpp adapter also supports stateless SSE
`/v1/generate/stream` plus streamed OpenAI-compatible `/v1/completions` and
`/v1/chat/completions` through the same SDK-owned llama.cpp flow. The local
`llama.cpp` CLI fallback remains a blocking-only bring-up path for streaming and
lifecycle control. For OpenAI-compatible HTTP responses, AX now omits `usage`
when a text-only backend did not report authoritative token counts rather than
inventing zero-valued token accounting.

## Design Rule

This server should remain a thin transport adapter.
It must not become the place where backend resolution, scheduler behavior, KV
ownership, or runtime semantics are redefined.
