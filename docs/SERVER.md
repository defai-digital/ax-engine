# Server

`ax-engine-server` is the first thin access layer built on top of
`ax-engine-sdk`.

## Current Scope

The current preview server is intentionally narrow:

- single-process local server
- built entirely on the Rust SDK contract
- fails closed on pre-M4 local hosts
- explicit backend and support-tier reporting
- preview generation endpoint for bring-up and integration testing
- preview OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
  endpoints for compatibility-backed integration
- stepwise request lifecycle endpoints that mirror the SDK preview contract for
  native runtime sessions plus the server-backed compatibility preview path

It is not yet:

- a production server surface
- a remote orchestration layer
- a replacement for broader serving infrastructure

## Endpoints

Current preview endpoints:

- `GET /health`
- `GET /healthz`
- `GET /v1/runtime`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/requests`
- `GET /v1/requests/:request_id`
- `POST /v1/requests/:request_id/cancel`
- `POST /v1/step`
- `POST /v1/generate/stream`
- `POST /v1/generate`

## Example

Start the server:

```text
cargo run -p ax-engine-server -- --model-id qwen3_dense --port 8080
```

If you want the native bring-up path to use explicit validated local artifacts
instead of SDK defaults or environment auto-detect:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --native-runtime-artifacts-dir /absolute/path/to/build/metal \
  --native-model-artifacts-dir /absolute/path/to/native-model-artifacts \
  --port 8080
```

The preview server requires a local Apple M4-or-newer host.
On M3 and older Macs, startup now fails closed instead of exposing an
unsupported partial runtime.

Start the preferred compatibility preview path against a running
`llama.cpp` server:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier compatibility \
  --compat-server-url http://127.0.0.1:8081 \
  --port 8080
```

Or target a documented OpenAI-compatible server such as `vLLM` or `mistral.rs`:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier compatibility \
  --compat-backend vllm \
  --compat-server-url http://127.0.0.1:8000 \
  --port 8080
```

Or use the CLI fallback path against a local GGUF model:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier compatibility \
  --compat-cli-path llama-cli \
  --compat-model-path /absolute/path/to/model.gguf \
  --port 8080
```

Or use a direct local `mlx-lm` fallback through the official `mlx_lm.generate`
CLI:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier compatibility \
  --compat-backend mlx \
  --compat-cli-path python3 \
  --compat-model-path /absolute/path/to/mlx-model \
  --port 8080
```

For compatibility mode, choose exactly one of:

- `--compat-server-url` for the server-backed adapter
  `llama.cpp` uses `/completion`, while `vLLM` and `mistral.rs` use
  OpenAI-compatible `/v1/completions`
- `--compat-model-path` plus `--compat-cli-path` for the local CLI fallback
  adapter

Compatibility backend selection defaults to `llama_cpp`.
Use `--compat-backend vllm`, `--compat-backend mistral-rs`, or
`--compat-backend mlx` when the delegated server is not `llama.cpp`.
The CLI fallback supports `llama.cpp` and a direct `mlx-lm` path.
For `mlx`, `--compat-cli-path` should usually be `python3` so AX can invoke
`python3 -m mlx_lm.generate`.
AX passes the request text as a raw prompt and disables the tokenizer chat
template on that CLI fallback so compatibility requests preserve AX prompt
semantics. The direct MLX CLI route remains blocking and does not support
streaming.

The preview server now also exposes thin OpenAI-compatible endpoints over that
same compatibility-backed path:

- `POST /v1/completions`
- `POST /v1/chat/completions`

Those routes are intentionally compatibility-only in this repository.
AX-native preview remains token-based and therefore fails closed on those text
or chat-oriented endpoints instead of inventing tokenizer or chat-template
behavior inside the server.

To run a repo-owned end-to-end smoke check that starts the preview binary and
exercises health, runtime, one-shot generate, cancel, and SSE streaming over
real HTTP:

```text
bash scripts/check-server-preview.sh
```

## Tracing

`ax-engine-server` supports `tracing`, but it keeps it opt-in so normal runs do
not pay steady-state formatting and subscriber overhead.

To enable server or core diagnostics, set `AX_ENGINE_SERVER_LOG` or `RUST_LOG`
before starting the server. For example:

```text
AX_ENGINE_SERVER_LOG=ax_engine_server=info,ax_engine_core=debug cargo run -p ax-engine-server -- --model-id qwen3_dense --port 8080
```

For throughput or latency measurements, prefer leaving tracing disabled, or use
an `info` or `warn` filter instead of `debug` / `trace`.

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

When the server is explicitly configured for `--support-tier compatibility`,
the same blocking endpoint accepts `input_text` for both compatibility modes:

```text
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_text": "Hello from compatibility",
    "max_output_tokens": 32
  }'
```

When compatibility is backed by `--compat-server-url`, the blocking endpoint
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

The same `--compat-server-url` path also supports the stateless streaming
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

To call the same compatibility-backed path through the preview
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

To use the preview chat-completions bridge, send text-only chat messages and
let AX flatten them into the same compatibility-backed request contract:

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
- native `/v1/step` responses now also surface a compact `metal_dispatch`
  summary when the shared session is running the AX-native Metal bring-up
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
  `resolution_policy`, capability reporting, host diagnostics, Metal
  toolchain availability, and native-runner provenance through the optional
  `native_runtime` section when AX native is active

`POST /v1/generate` remains a stateless convenience endpoint that creates a
fresh SDK session for one blocking request.
That endpoint now preserves one process-local request-id sequence even though
it creates fresh SDK sessions internally.
`POST /v1/generate/stream` uses the same stateless request shape but streams
preview SSE events named `request`, `step`, `response`, and `error`.
`POST /v1/completions` and `POST /v1/chat/completions` are thin response-shape
adapters over that same stateless compatibility-backed request flow; their
streaming mode emits unnamed SSE `data:` chunks plus `[DONE]` in the familiar
OpenAI-style envelope instead of AX-specific lifecycle event names.
The `/v1/requests` and `/v1/step` endpoints instead operate on one shared
preview session held by the server so they can surface the same request
lifecycle contract as the SDK.
The server allocates request ids from one process-local sequence across both
paths so transport logs and client correlation do not collide when clients mix
blocking and stepwise APIs.
For Phase 1, compatibility backends support blocking `/v1/generate`,
OpenAI-compatible `/v1/completions`, and OpenAI-compatible
`/v1/chat/completions`.
The server-backed adapters for `llama.cpp`, `vLLM`, `mistral.rs`, and explicit
OpenAI-compatible MLX servers also
support stateless SSE `/v1/generate/stream` plus streamed OpenAI-compatible
`/v1/completions` and `/v1/chat/completions` through the same SDK-owned
compatibility flow. The same server-backed compatibility path now also
supports preview stepwise `/v1/requests`, `/v1/step`, and
`/v1/requests/:id/cancel` through the SDK-owned lifecycle contract. That
shared session can now hold multiple active compatibility requests at once,
and each `/v1/step` aggregates one delegated step across all currently active
compatibility requests. The local `llama.cpp` and direct `mlx-lm` CLI
fallbacks remain blocking-only bring-up paths for streaming and lifecycle
control. For OpenAI-compatible HTTP responses, AX now omits `usage` when a
text-only backend did not report authoritative token counts rather than
inventing zero-valued token accounting.

## Design Rule

This server should remain a thin transport adapter.
It must not become the place where backend resolution, scheduler behavior, KV
ownership, or runtime semantics are redefined.
