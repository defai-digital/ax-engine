# ax-engine-server

`ax-engine-server` is the SDK-backed thin access layer for AX Engine. See
[AX Serving](../../docs/AX-SERVING.md) for the heterogeneous/NVIDIA repository
boundary and migration guide.

Current scope:

- local single-process preview server
- built entirely on `ax-engine-sdk`
- native MLX builds fail closed outside the supported M2 Max-or-newer,
  macOS 26+, 32 GB contract
- portable `delegated-server` builds omit MLX linkage and retain local
  `mlx_lm.server` and llama.cpp compatibility paths
- explicit runtime metadata reporting, including `selected_backend`,
  `support_tier`, and `resolution_policy`
- preview generation API for bring-up and integration testing
- stepwise request lifecycle endpoints over a shared preview session for
  repo-owned MLX runtime paths plus server-backed llama.cpp adapters
- thin OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
  translation over the selected delegated provider

Current preview endpoints:

- `GET /health`
- `GET /healthz`
- `GET /v1/runtime`
- `GET /v1/models`
- `POST /v1/requests`
- `GET /v1/requests/:request_id`
- `POST /v1/requests/:request_id/cancel`
- `POST /v1/step`
- `POST /v1/generate/stream`
- `POST /v1/generate`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Example:

```bash
cargo run -p ax-engine-server -- --model-id qwen3 --port 31418

bash scripts/check-server-preview.sh

curl http://127.0.0.1:31418/v1/runtime

curl http://127.0.0.1:31418/v1/requests \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4
  }'

curl -X POST http://127.0.0.1:31418/v1/step

curl http://127.0.0.1:31418/v1/requests/1

curl -N http://127.0.0.1:31418/v1/generate/stream \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4
  }'

curl http://127.0.0.1:31418/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4,
    "sampling": {
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": 0,
      "seed": 1234
    }
  }'

cargo run -p ax-engine-server -- \
  --model-id qwen3 \
  --support-tier llama-cpp \
  --llama-server-url http://127.0.0.1:8081 \
  --port 31418

curl http://127.0.0.1:31418/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 32
  }'

cargo run -p ax-engine-server -- \
  --model-id qwen3 \
  --support-tier llama-cpp \
  --llama-cli-path llama-cli \
  --llama-model-path /absolute/path/to/model.gguf \
  --port 31418

curl http://127.0.0.1:31418/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3",
    "input_text": "Hello from llama.cpp",
    "max_output_tokens": 32
  }'

```

This server is intentionally narrow.
It does not attempt multi-node routing, production transport ownership, or a
full remote orchestration surface during Phase 1 bring-up.
`/v1/generate` remains a stateless convenience path, while `/v1/requests` and
`/v1/step` expose the shared preview request lifecycle contract from the SDK.
`/v1/generate/stream` adds a minimal SSE transport over the same SDK-backed
request lifecycle rather than inventing a second streaming runtime.
For Phase 1, llama.cpp backends support blocking `/v1/generate`, plus thin
OpenAI-compatible `/v1/completions` and `/v1/chat/completions`. The
server-backed `llama.cpp` path also supports stateless SSE
`/v1/generate/stream`, streamed OpenAI-compatible responses, and preview
stepwise `/v1/requests`, `/v1/step`, and `/v1/requests/:id/cancel`.
Shared compatibility sessions can hold multiple active delegated llama.cpp
requests while `/v1/step` aggregates one delegated step across them.
`llama-cli` and direct `mlx-lm` remain blocking text-prompt fallbacks for local
bring-up.

Delegated JSON and SSE traffic use separate keep-alive pools, and the HTTP listener enables
`TCP_NODELAY` so a first small SSE event is not delayed by Nagle/delayed-ACK
interaction.

NVIDIA worker routing, compatibility identity, and release qualification live
in AX Serving. Former AX Engine CUDA flags are intentionally unavailable.
