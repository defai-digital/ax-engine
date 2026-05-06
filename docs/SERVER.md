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
  endpoints for delegated text integration
- OpenAI-shaped `/v1/embeddings` response envelopes for embedding-capable
  repo-owned MLX sessions
- stepwise request lifecycle endpoints that mirror the SDK preview contract for
  repo-owned MLX sessions plus the llama.cpp delegated path

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
- `POST /v1/embeddings`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/requests`
- `GET /v1/requests/:request_id`
- `POST /v1/requests/:request_id/cancel`
- `POST /v1/step`
- `POST /v1/generate/stream`
- `POST /v1/generate`

## Examples

`ax-engine-server` exposes three explicit runtime paths:

- `--mlx` selects the repo-owned MLX runtime for supported local model
  artifacts
- `--support-tier mlx_lm_delegated` delegates text generation to a
  user-provided `mlx_lm.server` while preserving AX blocking, fake-SSE, and
  OpenAI-compatible text surfaces
- `--support-tier llama_cpp` or a GGUF target delegates non-MLX inference to
  llama.cpp

Retired AX native mode is not a supported user-facing server mode.

Start the repo-owned MLX path against local MLX model artifacts:

```text
cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --mlx \
  --mlx-model-artifacts-dir /absolute/path/to/mlx-model-artifacts \
  --port 8080
```

Experimental MLX KV compression is opt-in and off by default. The current
`turboquant-shadow` mode is for benchmark evidence and route telemetry only: it
keeps generation on the existing full-precision MLX KV path, does not change
SDPA inputs, logits, sampling, or output tokens, and does not imply production
TurboQuant support.

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
counters. When disabled, the server emits no TurboQuant compression metadata.

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
be omitted.

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

The preview server requires a local Apple M4-or-newer host.
On M3 and older Macs, startup now fails closed instead of exposing an
unsupported partial runtime.

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
  --support-tier llama_cpp \
  --llama-server-url http://127.0.0.1:8081 \
  --port 8080
```

For the default route and the explicit llama.cpp route, choose exactly one
primary target:

- `--llama-server-url` for the server-backed `llama.cpp` `/completion` adapter
- `--llama-model-path` plus `--llama-cli-path` for the local CLI fallback
  adapter

Local GGUF paths are treated as `llama.cpp` targets. Use `--mlx` when you want
repo-owned MLX inference.

To keep using AX server surfaces for an MLX text model that the repo-owned MLX
runtime does not yet support, run `mlx_lm.server` yourself and select the
explicit delegated backend:

```text
mlx_lm.server --model /absolute/path/to/mlx-model --host 127.0.0.1 --port 8090

cargo run -p ax-engine-server -- \
  --model-id local-mlx-model \
  --support-tier mlx_lm_delegated \
  --mlx-lm-server-url http://127.0.0.1:8090 \
  --port 8080
```

`mlx_lm_delegated` is text-only and delegates model execution to an explicitly
configured upstream `mlx_lm.server`. It is not a repo-owned MLX performance
claim, and it is not a visual/multimodal contract.

The preview server now also exposes thin OpenAI-compatible endpoints over that
same llama.cpp-backed path:

- `POST /v1/completions`
- `POST /v1/chat/completions`

Those routes are intentionally llama.cpp-only in this repository.
The repo-owned MLX runtime remains token-based and therefore fails closed on
those text or chat-oriented endpoints instead of inventing tokenizer or
chat-template behavior inside the server.

For OpenAI-compatible MLX serving, run the optional Python shim with an explicit
MLX model artifact directory and tokenizer:

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
The `mlx_lm_delegated` backend supports `/v1/generate` through
`mlx_lm.server` `/v1/completions`. AX also exposes fake SSE over
`/v1/generate/stream` and streamed OpenAI-compatible completion/chat endpoints
by chunking the blocking delegated response into the normal AX SSE envelopes.
This preserves the AX transport contract for UI clients, but it is not true
token-by-token upstream streaming.

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
