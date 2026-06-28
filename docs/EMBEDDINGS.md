# Embeddings — detailed methodology and API reference

This document is the deep version of the README's `### Embedding
throughput` section. The README keeps three tables (in-process batched,
HTTP serving, cold start) and the three-line "use the batched API"
recommendation. Everything below — apples-to-apples wording, the
anti-pattern discussion, per-API code samples, telemetry — lives here.

## TL;DR

- **Use batched APIs.** Single-sentence loops are 2–3× slower than the
  same corpus in one batched call; this is a *general* observation, not
  an ax-engine vs `mlx-lm` claim.
- **Three equivalent batched entry points:**
  - Python: `session.embed_batch_array(list_of_sentences, ...)` returns a
    NumPy `(B, H)` `float32` ndarray. Zero-copy view via
    `np.frombuffer`; suitable for `faiss.Index.add` / HNSW / sklearn
    pipelines.
  - Rust: `EngineSession::embed_batch_flat(&batch, pooling, normalize)`
    returns an `EmbeddingMatrix { data: Vec<f32>, batch_size, hidden_size }`.
    Use directly; no PyO3 boundary, fewest allocations.
  - HTTP: POST `/v1/embeddings` with `{"input": [[ids], [ids], ...]}`.
    Response `data[]` has one entry per input, in order. Goes directly
    to `embed_batch_flat` server-side.
- **Or send concurrently** and let `EmbeddingMicroBatcher` coalesce
  callers that can't see the batch boundary up front.

## Apples-to-apples methodology

Published in-process comparisons should use `scripts/bench_embedding_fair.py`.
That harness forces both `mlx-lm` and ax-engine to materialize the same
contiguous CPU `float32 [B,H]` matrix, then reports batch-size and token-length
scaling separately. The older `scripts/bench_embedding_models.py` remains useful
for smoke coverage of single-call, HTTP, and optional Swift paths, but it mixes
several API contracts and should not be the primary publication source.

The README tables only count throughput where the caller can actually
*consume* the embedding (Python `list[float]`, NumPy ndarray, raw f32
bytes — not a GPU-only `MlxArray`). Both backends pay the GPU→CPU
read-back; comparisons without it under-count the cost.

- `mlx-lm` path: `model.model(x)` → last-token slice → `astype(float32)`
  → l2-normalize → `.tolist()` (triggers eval + read-back).
- `ax-engine-py` path: `session.embed_bytes(token_ids, ...)` or
  `session.embed_batch_flat_bytes(batch, ...)` / `embed_batch_array(...)`
  (eval + one contiguous `data_f32().to_vec()` + `PyBytes` wrap; NumPy
  views the bytes without restacking).
- `ax-engine` Rust path: `EngineSession::embed_batch_flat(...)` returns
  one contiguous `Vec<f32>` directly.

The fair harness reports two workload families:

- `short_query`: the canonical 10-sentence corpus with token lengths
  `[10, 15, 13, 8, 3, 8, 10, 8, 10, 10]`, cycled to the requested batch size.
- `fixed_N`: deterministic synthetic token IDs at fixed lengths such as
  16, 64, and 256 tokens, so batch-size scaling is not hidden by mixed lengths.

Both families use `last` pooling and l2-normalized output.

## The anti-pattern: one Python call per sentence

```python
# slow
embeddings = [session.embed_bytes(ids) for ids in sentences]

# fast — same corpus, ~2-3x throughput
embeddings = session.embed_batch_array(sentences)   # NumPy (B, H)
```

Why the loop is slow:

- Each call pays a fresh Python frame, a PyO3 boundary crossing, a
  `Mutex` lock on the session state, and one GPU sync.
- Those costs do *not* amortise: for `mlx-lm` they're the same shape as
  ax-engine (the Python interpreter doesn't care which backend lives
  behind the function call).
- A batched API collapses all of that to one call. The forward pass
  itself runs over a right-padded `[B, max_seq, H]` tensor and one GPU
  sync; per-sentence wall time is divided by B.

The current Qwen fair snapshot is
`benchmarks/results/embedding-fair/2026-06-28-qwen-hf-snapshot/2026-06-27-213439/`.
It uses 2 warmup + 5 measured trials, reports medians, and keeps the complete
short-query plus 16/64/256-token matrix in `summary.md`.

## Sustained vs intermittent profiles

The README's main throughput table is sustained: back-to-back batched calls
with no cooldown. That matches workloads like vector-DB ingest, batch
evaluation, or async worker pools running at steady state.

Use a non-zero `--cooldown` in `bench_embedding_fair.py` to model intermittent
calls such as interactive search queries. Those numbers can be lower because
each trial follows a GPU idle period.

Both profiles are valid; choose the one that matches your workload's
arrival pattern.

## HTTP serving paths

The README lists three contracts. Detailed semantics:

### `{"input": [[ids], [ids], ...]}` — explicit batch

- One HTTP request, one runner call.
- Skips the per-request microbatcher (the batch is already coalesced by
  the caller).
- Response `data[]` has one `OpenAiEmbeddingObject` per input, with
  `index` matching position in the request.
- Validation: `input` must not be empty; no inner sequence may be empty.

### `{"input": [ids]}` — single sequence + microbatcher

- Concurrent single-sequence requests are collected within a short
  window (`AX_ENGINE_EMBED_MICROBATCH_WINDOW_MS`, default 2 ms) and
  coalesced into one runner call.
- Requests with different `pooling` / `normalize` options are grouped
  separately so semantics never change.
- Max coalesced batch: `AX_ENGINE_EMBED_MICROBATCH_MAX_BATCH`
  (default 32).
- Right path when the client can't see the batch boundary (per-doc
  async workers, multi-tenant queues, OpenAI-SDK-style fan-out).

### Sequential single — the worst case

Many clients (or a synchronous loop) post `{"input": [ids]}` one after
another. Each POST round-trips through the GPU on its own. The
microbatcher's window is exceeded between requests, so coalescing does
not happen. Reported in the README's HTTP table for comparison; not a
recommended pattern.

## Telemetry: compile-cache hit / miss

The MLX runner caches compiled embedding closures per `(seq_len,
target_position)` for single-call and `(batch_size, max_len,
target_positions)` for batched. Workloads with very heterogeneous
length distributions can fragment the cache. Read counters via
`MlxRunner::embed_compile_cache_stats()`:

```rust
let stats = runner.embed_compile_cache_stats();
println!(
    "single hits={} misses={} cache_len={}",
    stats.single_hits, stats.single_misses, stats.single_len,
);
println!(
    "batched hits={} misses={} cache_len={}",
    stats.batched_hits, stats.batched_misses, stats.batched_len,
);
```

Healthy ingest: `hits` dominates `misses` after a brief warmup, and
cache sizes stable. Diagnostic for fragmentation: cache size grows
unboundedly with a low hit rate. Mitigations: bucket batches by length
client-side; round `target_positions` to a small set.

## Cold start — `AX_MMAP_WEIGHTS`

The memory-mapped safetensors loader skips the userspace heap buffer +
`read()` pipeline that the default C loader does. Warm-cache wins:
-11% / -30% / -41% on 0.6B / 4B / 8B at session construction. True-cold
(post-`sudo purge`) measurement procedure and the criteria for flipping
to default-on are in [`EMBEDDING_COLDSTART.md`](EMBEDDING_COLDSTART.md).

Output is bit-exact with the C loader; opt in safely.

## Reproducing every README number

Use the fair in-process harness for README throughput claims:

```bash
.venv/bin/python scripts/bench_embedding_fair.py \
  --model qwen3-embedding-0.6b-8bit=/path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha> \
  --model qwen3-embedding-4b-4bit-dwq=/path/to/Qwen3-Embedding-4B-4bit-DWQ/snapshots/<sha> \
  --model qwen3-embedding-8b-4bit-dwq=/path/to/Qwen3-Embedding-8B-4bit-DWQ/snapshots/<sha> \
  --batch-sizes 1,8 \
  --lengths 16,64,256 \
  --warmup 2 \
  --trials 5 \
  --output-dir benchmarks/results/embedding-fair/$(date +%Y-%m-%d)-qwen
```

The legacy `scripts/bench_embedding_readme.sh` still runs HTTP serving and
cold-start paths. Use it for endpoint/cold-start evidence, not as the primary
`mlx-lm` vs ax-engine in-process publication source.
