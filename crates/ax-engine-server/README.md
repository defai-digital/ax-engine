# ax-engine-server

`ax-engine-server` is AX Engine's lightweight single-node HTTP surface.

Current compatibility target:

- llama-server-style `GET /health`, `GET /models`, `GET /props`
- llama-server-style `GET /slots`
- llama-server-style `POST /slots/{id_slot}?action=save|restore|erase`
- llama-server-style `POST /completion`, `POST /tokenize`, `POST /detokenize`, `POST /apply-template`
- llama-server-style request fields such as `cache_prompt` and `id_slot` for single-slot flows
- OpenAI-compatible `GET /v1/models`
- OpenAI-compatible `POST /v1/completions`
- OpenAI-compatible `POST /v1/chat/completions`
- OpenAI-compatible `POST /v1/responses`
- optional `GET /metrics` when `--metrics` is enabled

Current non-goals for this crate:

- multi-node routing
- continuous batching
- tool calling
- embeddings and reranking

Run locally:

```bash
cargo run -p ax-engine-server -- \
  --model ./models/Qwen3.5-9B-Q4_K_M.gguf \
  --slot-save-path ./tmp/slots \
  --host 127.0.0.1 \
  --port 8080
```

Slot persistence uses versioned token snapshots rooted under `--slot-save-path`.
For models whose KV implementation supports snapshotting, in-process slot reuse
keeps a live snapshot in memory. When the backend cannot safely rewind a cached
slot to an arbitrary prefix, AX falls back to prompt replay instead of serving
incorrect results.

The server is intentionally thin. Production routing, fleet orchestration, and
governed deployment belong in `ax-serving`.

Compatibility plan:

- `docs/LLAMA_SERVER_COMPATIBILITY.md`

Smoke validation:

```bash
crates/ax-engine-server/scripts/compat_smoke.sh http://127.0.0.1:8080
```
