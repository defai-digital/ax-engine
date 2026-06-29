# Embeddings — detailed methodology and API reference

This document is the deep version of the README's `### Embedding throughput`
section. The README keeps the compact fair in-process comparison table and
reproduction commands; everything below — apples-to-apples wording, the
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

Published in-process embedding numbers should use
`scripts/bench_embedding_fair.py`. For comparison rows, that harness forces both
the reference backend and ax-engine to materialize the same contiguous CPU
`float32 [B,H]` matrix. For AX-only refreshes, pass `--ax-only` to skip
loading the reference backend while preserving the same output contract and
workload matrix. The older `scripts/bench_embedding_models.py` remains useful
for smoke coverage of single-call, HTTP, and optional Swift paths, but it mixes
several API contracts and should not be the primary publication source.

Use `scripts/bench_embedding_ingest_scale.py` when the question is sustained
RAG/indexing throughput rather than one isolated batch. It reuses the fair
backend adapters and output contract, then embeds a deterministic corpus such
as 512 chunks split into repeated batches. That surfaces two effects the fair
table intentionally keeps separate: whether throughput holds across many
flushes, and how p95 batch latency rises as batch size grows. Do not use it as
a replacement for the fair table; use it as the scale profile beside it.

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

The default Qwen comparison uses `last` pooling and l2-normalized output.
EmbeddingGemma uses `--pooling mean` so the reference and AX routes match that
model family's mean-pooling contract.

## Correctness QA contract

Run `scripts/verify_embedding_models.py` before publishing embedding benchmark
claims for a refreshed model artifact. It verifies caller-consumable normalized
`float32 [B,H]` output, not just endpoint shape:

```bash
.venv/bin/python scripts/verify_embedding_models.py \
  --model-dir /path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha>

.venv/bin/python scripts/verify_embedding_models.py \
  --model-kind embeddinggemma \
  --model-dir /path/to/embeddinggemma-300m-8bit/snapshots/<sha>
```

The verifier uses family-specific oracles:

- Qwen3-Embedding: `mlx-lm` transformer body, last-token pooling, l2 norm.
- EmbeddingGemma: `mlx-embeddings`, mean pooling + Dense head + l2 norm.

For EmbeddingGemma, the correctness oracle is the `mlx-embeddings` **single-row**
path for each input. Do not use its mixed-length batch output as a correctness
oracle: the reference package is not batch-invariant for that case. AX still
checks its own batch output against AX single-row output, so padding and
batched pooling regressions remain covered.

Qwen3 8B 4-bit DWQ uses a slightly looser default cosine threshold than the
smaller Qwen rows because short text can show quantization drift while still
remaining normalized and batch-stable. Override with `--cosine-threshold` when a
specific release requires a stricter or looser gate.

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

The current README reference-comparison snapshot for Qwen is
`benchmarks/results/embedding-fair/2026-06-28-qwen-after-batch-fix/2026-06-28-051508/`.
The current README reference-comparison snapshot for EmbeddingGemma is
`benchmarks/results/embedding-fair/2026-06-28-embeddinggemma-after-batch-fix/2026-06-28-051549/`.
The current README AX-only refresh snapshots are
`benchmarks/results/embedding-fair/2026-06-28-qwen-ax-only-refresh/2026-06-28-152458/`
and
`benchmarks/results/embedding-fair/2026-06-28-embeddinggemma-ax-only-mask-refresh/2026-06-28-155600/`.
All use 2 warmup + 5 measured trials, report medians, and keep the complete
short-query plus 16/64/256-token matrix in `summary.md`. The Qwen reference
comparison uses `mlx-lm` as the baseline backend. EmbeddingGemma uses
`mlx-embeddings` with mean pooling because `mlx-lm` does not provide the
comparable EmbeddingGemma route used by this harness.

## Sustained vs intermittent profiles

The README's main throughput table is sustained: back-to-back batched calls
with no cooldown. That matches workloads like vector-DB ingest, batch
evaluation, or async worker pools running at steady state.

Use a non-zero `--cooldown` in `bench_embedding_fair.py` to model intermittent
calls such as interactive search queries. Those numbers can be lower because
each trial follows a GPU idle period.

Both profiles are valid; choose the one that matches your workload's
arrival pattern.

## Large-corpus ingest scale

The current README scale snapshot is:

`benchmarks/results/embedding-scale/2026-06-28-qwen-ingest-scale/2026-06-28-184450/`

It runs Qwen3-Embedding 0.6B 8-bit against `mlx-lm`, using last-token pooling
and l2-normalized contiguous CPU `float32 [B,H]` output for both backends.
Each trial embeds 512 deterministic chunks, with chunk lengths 256 and 512 and
batch sizes 8, 32, and 64. The harness reports median tok/s, chunks/s,
output MB/s, and per-batch p50/p95/max latency.

Read the scale table differently from the fair table. The fair table answers
"how fast is this batch shape when both engines return a caller-consumable
matrix?" The scale table answers "does that rate hold when a RAG worker keeps
feeding many batches, and what flush latency does the chosen batch size create?"
For the current 0.6B snapshot, AX is behind `mlx-lm` by 2.4-6.5% on 256-token
chunks, ahead by 3.5% at 512-token batch=8, behind by 3.9% at 512-token
batch=32, and effectively tied at 512-token batch=64.

Reproduce the scale snapshot with:

```bash
.venv/bin/python scripts/bench_embedding_ingest_scale.py \
  --model qwen3-embedding-0.6b-8bit=/path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha> \
  --batch-sizes 8,32,64 \
  --chunk-tokens 256,512 \
  --total-chunks 512 \
  --warmup 1 \
  --trials 3 \
  --output-dir benchmarks/results/embedding-scale/$(date +%Y-%m-%d)-qwen-ingest-scale
```

## HTTP serving paths

The README lists three contracts. Detailed semantics:

### `{"input": [[ids], [ids], ...]}` — explicit batch

- One HTTP request, one runner call.
- Skips the per-request microbatcher (the batch is already coalesced by
  the caller).
- Response `data[]` has one `OpenAiEmbeddingObject` per input, with
  `index` matching position in the request.
- Validation: `input` must not be empty; no inner sequence may be empty.

### `POST /v1/embedding_records` — structured RAG ingestion batch

Use this AX-native endpoint when the caller owns document records rather
than pre-tokenized prompt arrays. The request accepts `records[]` with a
stable `id`, either `text` or structured `fields`, optional `metadata`,
and optional fixed-token chunking:

```json
{
  "records": [
    {
      "id": "doc-1",
      "fields": {"title": "Release notes", "body": "AX Engine update"},
      "metadata": {"source": "notion"}
    }
  ],
  "render_template": "title: {title}\nbody: {body}",
  "chunking": {"max_tokens": 512, "overlap_tokens": 64}
}
```

The server renders fields deterministically, tokenizes with the model's
`tokenizer.json`, chunks by token range, runs the chunks through the
same contiguous batch path as `/v1/embeddings`, and returns one
embedding per chunk with `id`, `record_index`, `chunk_index`,
`token_start`, `token_end`, and `metadata`.

Do not send arbitrary JSON dumps as embedding text. Keep searchable
content in the rendered text and keep filter-only attributes in
`metadata`.

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
  --ax-only \
  --model qwen3-embedding-0.6b-8bit=/path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha> \
  --model qwen3-embedding-4b-4bit-dwq=/path/to/Qwen3-Embedding-4B-4bit-DWQ/snapshots/<sha> \
  --model qwen3-embedding-8b-4bit-dwq=/path/to/Qwen3-Embedding-8B-4bit-DWQ/snapshots/<sha> \
  --batch-sizes 1,8 \
  --lengths 16,64,256 \
  --warmup 2 \
  --trials 5 \
  --output-dir benchmarks/results/embedding-fair/$(date +%Y-%m-%d)-qwen
```

For EmbeddingGemma, use the same output contract with the embeddinggemma route:

```bash
.venv/bin/python scripts/bench_embedding_fair.py \
  --ax-only \
  --reference mlx_embeddings --pooling mean \
  --model embeddinggemma-300m-8bit=/path/to/embeddinggemma-300m-8bit/snapshots/<sha> \
  --batch-sizes 1,8 \
  --lengths 16,64,256 \
  --warmup 2 \
  --trials 5 \
  --output-dir benchmarks/results/embedding-fair/$(date +%Y-%m-%d)-embeddinggemma
```

Pass `--ax-only` when you want to refresh only the AX path without loading the
reference backend; do not use that mode for README reference-comparison claims.

The legacy `scripts/bench_embedding_readme.sh` still runs HTTP serving and
cold-start paths. Use it for endpoint/cold-start evidence, not as the primary
embedding in-process publication source.
