# AX Engine Server

`ax-engine-server` is the basic HTTP inference server that ships with this
repository. Its job is simple: load one GGUF model, expose a small
OpenAI-compatible API, and make AX Engine consumable by other software with no
custom transport layer.

It is built on the high-level `ax-engine-sdk` facade rather than reaching
directly into `ax-engine-core` application glue. That keeps server integration
behavior aligned with the Rust and Python SDK surfaces.

That also means server model loading now inherits AX's inference routing:
native when supported, `llama.cpp` fallback when routing is enabled.

It is not a replacement for `ax-serving`.

Use `ax-engine-server` when you want:

- a local integration surface for editors, agents, scripts, or internal tools
- one-process, one-model inference with minimal setup
- OpenAI-compatible JSON and SSE so existing clients work with little or no change
- the transport layer that powers the repo's JavaScript SDK for Node.js and Next.js

Keep using `ax-serving` when you need:

- multi-model routing
- continuous batching
- request scheduling across tenants or priorities
- production-grade auth, quotas, and operational controls

## Current Design

The current implementation is intentionally conservative:

- one loaded model per process
- one generation request at a time
- no persistent multi-request session or server-side conversation store
- no continuous batching
- no embeddings, responses API, or tool-calling surface yet

This keeps the first server aligned with AX Engine's current strengths: direct
model execution, predictable request behavior, and low integration friction.

## Build

From the repository root:

```bash
cargo build --workspace --release
```

The server binary is:

```text
./target/release/ax-engine-server
```

## Start the Server

Example:

```bash
./target/release/ax-engine-server \
  --model ./models/Qwen3-8B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 3000
```

Useful flags:

- `--backend auto|cpu|metal|hybrid|hybrid_cpu_decode`
- `--ctx-size 4096`
- `--max-tokens 256`
- `--seed -1`
- `--verbose`

You can inspect all options with:

```bash
./target/release/ax-engine-server --help
```

Useful routing env vars:

- `AX_ROUTING=auto`
- `AX_ROUTING_ARCH="mistral=llama_cpp,deepseek=llama_cpp"`
- `AX_ROUTING_MODEL="/absolute/path/to/model.gguf=llama_cpp"`
- `AX_LLAMA_SERVER_PATH=/opt/homebrew/bin/llama-server`
- `AX_LLAMA_SERVER_TIMEOUT=120`

## Endpoints

The server currently exposes:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Both completion endpoints support:

- standard JSON responses
- SSE streaming with `stream: true`

See [AX Engine Server API](./ax-engine-server-api.md) for request and response
examples.

When routing is active:

- `GET /healthz` reports `backend`
- `GET /v1/models` reports `backend`
- both endpoints include optional `routing` text when the model is running through `llama.cpp`

## Compatibility Notes

The API is OpenAI-compatible at the integration level, not byte-for-byte
feature parity.

Supported request fields today include:

- `model`
- `prompt` or `messages`
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

Current limitations:

- `n` must be `1`
- only text content parts are accepted in chat messages
- no server-side session reuse yet, so each request starts from a fresh KV state

## Recommendation

Treat `ax-engine-server` as the default adapter layer for local software that
expects an API endpoint, and treat `ax-serving` as the larger serving system
for batching, orchestration, and production concerns.
