# ax-engine-server

`ax-engine-server` is the first SDK-backed thin access layer for AX Engine v4.

Current scope:

- local single-process preview server
- built entirely on `ax-engine-sdk`
- fail-closed host validation for pre-M4 Macs
- explicit runtime metadata reporting, including `selected_backend`,
  `support_tier`, and `resolution_policy`
- preview generation API for bring-up and integration testing
- stepwise request lifecycle endpoints over a shared preview session for native
  runtime paths plus server-backed compatibility adapters
- thin OpenAI-compatible `/v1/completions` and `/v1/chat/completions`
  translation over compatibility-backed preview requests

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
cargo run -p ax-engine-server -- --model-id qwen3_dense --port 8080

bash scripts/check-server-preview.sh

curl http://127.0.0.1:8080/v1/runtime

curl http://127.0.0.1:8080/v1/requests \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4
  }'

curl -X POST http://127.0.0.1:8080/v1/step

curl http://127.0.0.1:8080/v1/requests/1

curl -N http://127.0.0.1:8080/v1/generate/stream \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 4
  }'

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

cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier compatibility \
  --compat-server-url http://127.0.0.1:8081 \
  --port 8080

curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_tokens": [1, 2, 3, 4],
    "max_output_tokens": 32
  }'

cargo run -p ax-engine-server -- \
  --model-id qwen3_dense \
  --support-tier compatibility \
  --compat-cli-path llama-cli \
  --compat-model-path /absolute/path/to/model.gguf \
  --port 8080

curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3_dense",
    "input_text": "Hello from compatibility",
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
For Phase 1, compatibility backends support blocking `/v1/generate`, plus thin
OpenAI-compatible `/v1/completions` and `/v1/chat/completions`. The
server-backed `llama.cpp`, `vLLM`, `mistral.rs`, and explicit MLX server paths
also support stateless SSE `/v1/generate/stream`, streamed OpenAI-compatible
responses, and preview stepwise `/v1/requests`, `/v1/step`, and
`/v1/requests/:id/cancel`. Shared compatibility sessions can now hold multiple
active delegated requests while `/v1/step` aggregates one delegated step across
them. `llama-cli` and direct `mlx-lm` remain blocking text-prompt fallbacks for
local bring-up.
