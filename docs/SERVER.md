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
  endpoints for llama.cpp-backed integration
- stepwise request lifecycle endpoints that mirror the SDK preview contract for
  MLX-mode sessions plus the llama.cpp bypass path

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
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --llama-server-url http://127.0.0.1:8081 \
  --port 8080
```

`ax-engine-server` now ships with two inference routes:

- `--mlx` selects the repo-owned MLX runtime
- non-MLX inference routes to `llama.cpp`

Retired AX native mode is not a supported user-facing server mode.

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --mlx \
  --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts \
  --port 8080
```

The preview server requires a local Apple M4-or-newer host.
On M3 and older Macs, startup now fails closed instead of exposing an
unsupported partial runtime.

Start the repo-owned MLX path against local MLX model artifacts:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --mlx \
  --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts \
  --port 8080
```

Or use the default bypass mode against a local GGUF model:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --llama-cli-path llama-cli \
  --llama-model-path /absolute/path/to/model.gguf \
  --port 8080
```

To run the non-MLX bypass route, configure a `llama.cpp` target:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --llama-cli-path llama-cli \
  --llama-model-path /absolute/path/to/model.gguf \
  --port 8080
```

OpenAI-compatible delegated routes such as `vLLM`, `mistral.rs`, and MLX
llama.cpp adapters are no longer part of the shipping inference contract.
Use the `llama.cpp` server route for non-MLX server-backed inference:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier llama_cpp \
  --llama-server-url http://127.0.0.1:8081 \
  --port 8080
```

For the default route and the explicit llama.cpp route, choose exactly one
primary target:

- `--llama-server-url` for the server-backed `llama.cpp` `/completion` adapter
- `--llama-model-path` plus `--llama-cli-path` for the local CLI fallback
  adapter

Local non-MLX model paths are treated as `llama.cpp` targets. Use `--mlx` when
you want AX-owned MLX inference.

The preview server now also exposes thin OpenAI-compatible endpoints over that
same llama.cpp-backed path:

- `POST /v1/completions`
- `POST /v1/chat/completions`

Those routes are intentionally llama.cpp-only in this repository.
AX-owned MLX mode remains token-based and therefore fails closed on those text
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
AX_ENGINE_SERVER_LOG=ax_engine_server=info,ax_engine_core=debug cargo run -p ax-engine-server -- --model-id qwen3_dense --mlx --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts --port 8080
```

For manual throughput or latency measurements, leave tracing disabled, or use
an `info` or `warn` filter instead of `debug` / `trace`. For comparable AX MLX
inference numbers, prefer `scripts/bench_mlx_inference_stack.py`; it starts the
server, captures AX SSE `runner_time_us`, and records the MLX reference runtime
identity explicitly.

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

When the server is running on the default MLX path, the `.gguf` llama.cpp
bypass path, or an explicit llama.cpp path, the same blocking endpoint
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
let AX flatten them into the same llama.cpp-backed request contract:

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
`POST /v1/generate/stream` uses the same stateless request shape but streams
preview SSE events named `request`, `step`, `response`, and `error`.
`POST /v1/completions` and `POST /v1/chat/completions` are thin response-shape
adapters over that same stateless llama.cpp-backed request flow; their
streaming mode emits unnamed SSE `data:` chunks plus `[DONE]` in the familiar
OpenAI-style envelope instead of AX-specific lifecycle event names.
The `/v1/requests` and `/v1/step` endpoints instead operate on one shared
preview session held by the server so they can surface the same request
lifecycle contract as the SDK.
The server allocates request ids from one process-local sequence across both
paths so transport logs and client correlation do not collide when clients mix
blocking and stepwise APIs.
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
