# Server

`ax-engine-server` is the first thin access layer built on top of
`ax-engine-sdk`.

## Current Scope

The current preview server is intentionally narrow:

- single-process local server
- built entirely on the Rust SDK contract
- fails closed on unsupported hosts (requires M2 Max or newer, macOS 14+, 32 GB RAM)
- explicit backend and support-tier reporting
- preview generation endpoint for bring-up and integration testing
- preview OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
  endpoints for native MLX sessions with tokenizer artifacts and delegated text
  integration
- OpenAI-shaped `/v1/embeddings` response envelopes for embedding-capable
  repo-owned MLX sessions
- stepwise request lifecycle endpoints that mirror the SDK preview contract for
  repo-owned MLX sessions plus the llama.cpp delegated path
- optional API key authentication for HTTP API routes

It is not yet:

- a production server surface
- a remote orchestration layer
- a replacement for broader serving infrastructure

## Endpoints

Current preview endpoints:

- `GET /health`
- `GET /healthz`
- `GET /metrics`
- `GET /v1/runtime`
- `GET /v1/models`
- `POST /v1/embeddings`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/requests`
- `GET /v1/requests/:request_id`
- `POST /v1/requests/:request_id/cancel`
- `POST /v1/step`
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

`--api-key` covers HTTP routes only. The optional gRPC adapter
(`--grpc-bind-address`) has no authentication yet; bind it to loopback or keep
it disabled when the HTTP surface is key-protected.

## Observability

`GET /metrics` serves a Prometheus text exposition with HTTP request counters
(total, in-flight, 2xx/4xx/5xx) and engine-step gauges (scheduled requests and
tokens, KV block usage, accumulated prefix-cache hits). The endpoint is
read-only: engine-step values are snapshots cached when generation endpoints
drive real steps, never sampled by stepping the engine from the scrape path.
Engine-step gauges appear only after at least one step has been observed via
`POST /v1/step`. `/metrics` requires the API key when authentication is
enabled and never exposes prompts, outputs, or credentials.

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
- **`response_format: json_object`** (completions and chat): non-streaming
  responses are validated server-side; output that is not a JSON object
  returns `502 invalid_output`. This is post-hoc validation, not constrained
  decoding — JSON schema enforcement is not supported yet.
- **`tools` / `tool_choice`** (chat): experimental. When tools are present,
  explicit `<tool_call>{…}</tool_call>` spans in the model output are parsed
  into `message.tool_calls`. Bare JSON answers are never reinterpreted as tool
  calls. `/v1/models` continues to report
  `openai_tool_calling_supported: false` until prompt-side tool rendering,
  streaming deltas, and continuation handling land end-to-end.

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

MLX KV compression defaults to `turboquant-fused-experimental`. Pass
`--experimental-mlx-kv-compression disabled` to keep the full-precision KV path
unchanged, or set `AX_DISABLE_TURBOQUANT_FUSED_DECODE=1` as a runtime kill
switch that forces every layer back to the full-precision SDPA route without
restarting with a different flag. Default-on route selection does not imply
production TurboQuant support: promotion remains gated on the long-context
quality artifact. The `turboquant-shadow` mode is for benchmark evidence and
route telemetry only: it keeps generation on the existing full-precision MLX KV
path, does not change SDPA inputs, logits, sampling, or output tokens.

`turboquant-fused-experimental` (the default) is the fused route selection. It
requests compressed decode and tries the two-stage Metal cold decode plus
full-precision hot-tail merge for eligible K8/V4 single-token decode layers.
When Metal succeeds, route metadata reports `fused_compressed_decode`; when
Metal is unavailable, the runtime falls back to the full-precision MLX KV path
instead of using the CPU oracle. Use route metadata to inspect candidate,
attempt, success, fallback, and fallback-reason counters. Fallback reason label
`runner_not_integrated` means no runtime decode attempt was observed yet;
`cpu_oracle_unavailable` identifies legacy/debug artifacts where the
compressed-decode oracle path was not available. Only `fused_compressed_decode`
route evidence with successful attempts, Metal fused decode successes, and zero
fallbacks can feed the internal quality artifact gate; shadow and legacy CPU
oracle rows are diagnostic only. Public-support promotion remains blocked
unless the separate readiness report also passes the decode-throughput
performance gate.

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --mlx \
  --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts \
  --experimental-mlx-kv-compression turboquant-shadow \
  --experimental-mlx-kv-compression-hot-window-tokens 256 \
  --experimental-mlx-kv-compression-min-context-tokens 512 \
  --port 8080
```

When enabled, route metadata includes TurboQuant eligibility, estimated
compressed/saved KiB, production-readiness blockers, and runtime shadow-storage
counters, including shadow sync calls and wall time. It also reports the current
compression decode path plus fused decode candidate, attempt, success, fallback,
and fallback-reason counters. When disabled, the server emits no TurboQuant
compression metadata.

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
`gemma4-31b`, and `qwen3.6-35b`.

The `glm4.7-flash-4bit` preset (GLM-4.7 Flash, `glm4_moe_lite`) is a passby
preset: it reports the `mlx_lm_delegated` runtime tier and requires
`--mlx-lm-server-url` instead of a local artifacts dir. GLM is no longer a
direct-support model — see
[`SUPPORTED-MODELS.md`](SUPPORTED-MODELS.md).

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
macOS 14 (Sonoma) or later with 32 GB RAM minimum.
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

`GET /v1/models` advertises image, audio, and video input only for repo-owned
native MLX sessions whose `model-manifest.json` contains the converted Gemma4
unified media tensor roles. Two input shapes are accepted on those sessions:

- **Inline media on chat.** `POST /v1/chat/completions` accepts OpenAI-style
  content parts with base64 `data:` URIs: `image_url` (PNG/JPEG),
  `input_audio` / `audio_url` (WAV or MP3), and `video_url` (animated GIF, plus
  MP4/WebM when `ffmpeg` is installed on the server `PATH`). The server decodes
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
  actionable HTTP 400 instead of being scheduled. A maximum-length video
  (32 frames at ~70 soft tokens per frame plus timestamps, ~2,400+ tokens)
  needs a raised `--max-batch-tokens`. Under concurrent load a fitting
  request may wait for a step with enough budget; it is never split.
- **Token budgets.** Image and video soft-token budgets come from the model's
  `preprocessor_config.json` (Gemma 4 12B: up to 280 soft tokens per image,
  70 per video frame); there is no per-request budget or quality override.
- **Audio.** WAV and MP3 input is downmixed to mono and resampled to the
  model rate (16 kHz). The container is sniffed from magic bytes, not the
  declared `format` field. Other formats (AAC/OGG/FLAC) are rejected; send
  pre-computed audio tensors via `/v1/generate` instead. Audio longer than
  the model's `audio_seq_length` cap (750 frames × 40 ms = 30 s by default)
  is silently truncated, and MP3 decoding stops at that cap.
- **Video.** Animated GIF is decoded in-process. MP4/WebM inline input is
  decoded by `ffmpeg` if that binary is available on the server `PATH`; video
  container and codec decode is not an MLX tensor-kernel operation. Frames are
  sampled uniformly to at most 32 frames, each rendered with an `mm:ss`
  timestamp. During extraction frames are downscaled to at most 1600 px on the
  longest side and the decoded stream is capped at 512 MiB; a video whose
  decoded stream exceeds that cap is sampled from the decoded prefix only. If
  `ffmpeg` is unavailable, send pre-computed video tensors via `/v1/generate`.
- **Caching.** Prefix caching is disabled for multimodal requests; every
  multimodal prefill recomputes the full prompt with that request's own
  media tensors.
- **Chat output.** Gemma 4 thinking-channel framing (`<|channel>thought…`)
  is stripped from chat content; raw `/v1/completions` output stays
  verbatim.

`scripts/qa_gemma4_multimodal.py --strict` runs an end-to-end probe set
(image color, image description, audio smoke, speech transcription, GIF frame
count) against a live server and fails on thinking-channel leaks or content
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

See `docs/API-COMPATIBILITY.md` for the exact OpenAI-shaped endpoint matrix,
including supported request fields, runtime paths, and non-goals such as tool
calling.

To run a repo-owned end-to-end smoke check that starts the preview binary and
exercises health, runtime, one-shot generate, cancel, and SSE streaming over
real HTTP:

```text
bash scripts/check-server-preview.sh
```

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
raw image/audio/video content parts, then uses the supplied prompt tokens
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

Submit a request into the shared preview session:

```text
curl http://127.0.0.1:8080/v1/requests \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4
  }'
```

Advance the shared preview session by one engine step:

```text
curl -X POST http://127.0.0.1:8080/v1/step
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

`POST /v1/generate` remains a stateless convenience endpoint that creates a
fresh SDK session for one blocking request.
That endpoint now preserves one process-local request-id sequence even though
it creates fresh SDK sessions internally.
For native MLX Gemma4 unified models, `/v1/generate` and
`/v1/generate/stream` also accept preprocessed
`multimodal_inputs.gemma4_unified` image/audio/video tensors. AX does not
decode raw image/audio/video URLs or files on this path; callers must run media
loading and processor output generation before sending the native request.
The Python SDK helper can perform that client-side preprocessing for image
paths/URLs/data URIs, WAV audio paths/URLs/data URIs, OpenAI-style
`input_audio` WAV base64 dictionaries, and decoded video frame sequences.
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
The `/v1/requests` and `/v1/step` endpoints instead operate on one shared
preview session held by the server so they can surface the same request
lifecycle contract as the SDK.
The server allocates request ids from one process-local sequence across both
paths so transport logs and client correlation do not collide when clients mix
blocking and stepwise APIs.
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
