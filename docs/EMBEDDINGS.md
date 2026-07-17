# Embeddings — detailed methodology and API reference

Deep methodology and API reference for embedding ingest. Published ingest-scale
tables and charts live in
[Performance Results: Embeddings](PERFORMANCE-RESULTS.md#session-mode-embeddings).
This page covers apples-to-apples wording, anti-patterns, per-API samples, and
telemetry.

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

## Publication discipline and provenance

Harness artifacts now record **host**, **build** (commit + dirty flag),
**runtime_identity** (AX cdylib path/sha plus `otool` libmlx/libmlxc install
names, resolved paths, sha256, and source class), and **embed_env_flags**.
Schema versions: `ax.embedding_fair.v2` and `ax.embedding_ingest_scale.v2`.

Rules for public claims:

| Claim | Required run mode | Use for |
|---|---|---|
| `paired_delta` | same-session fair/scale **without** `--ax-only` | README AX vs mlx-lm / mlx-embeddings deltas |
| `ax_absolute_trend` | `--ax-only` | Absolute AX trend across commits; never invent a reference delta |

### How to read batching on public charts

| Surface | What is timed | Batch size |
|---|---|---|
| **README ingest scale** (yellow vs green) | Sustained multi-chunk encode with shared output contract | **B = 8, 32, 64 for both engines** |
| **Fair diagnostic** | One cooled batch per trial; short-query headlined as ms/item | Includes **B = 1** and larger B |
| **Single-sentence loop** | One Python/API call per string | Not a fair competitive mode |

**Best practices for publication and product messaging:**

1. **Say “batched encode” when the chart is green vs yellow on ingest scale.**
   Green is not AX single-call mode; both series use `embed_batch_*`-style
   matrix forwards.
2. **Keep competitive charts apples-to-apples.** Do not add an AX-only “batch”
   series next to a reference that is already batched at the same B.
3. **Teach batching as an API habit, not an exclusive engine win.** The 2–3×
   gain from dropping a one-call-per-sentence loop applies to any backend that
   can take a batch (see the anti-pattern section below).
4. **Put B = 1 latency under fair diagnostics**, not under ingest-scale
   headline tok/s. Short-query fair rows **headline latency**
   (`median_ms_per_item`, lower is better).
5. **Gate publication claims** with `check_embedding_publish_gate.py` before
   PERFORMANCE-RESULTS or chart updates.

Fixed-length and ingest-scale rows headline tok/s (and scale also reports
chunks/s + batch p95 ms). Do not publish short-query tok/s as the primary
public number: short text makes tok/s noisy and misrepresents query serving
latency.

Before wiring a new artifact into PERFORMANCE-RESULTS or charts, run:

```bash
# Paired reference delta (default)
.venv/bin/python scripts/check_embedding_publish_gate.py \
  path/to/embedding_fair.json path/to/embedding_ingest_scale.json

# AX-only absolute trend
.venv/bin/python scripts/check_embedding_publish_gate.py --claim ax_absolute_trend \
  path/to/embedding_ingest_scale.json

# Retained historical v1 rows only
.venv/bin/python scripts/check_embedding_publish_gate.py --allow-legacy \
  path/to/legacy_embedding_ingest_scale.json
```

The gate rejects `paired_delta` when AX is linked to Homebrew `libmlx` while
the reference uses a pip/venv wheel — that mismatch was the root of a false
~3× AX-vs-mlx-lm gap. Prefer the venv/pip MLX wheel and the repo rpath wiring
in `mlx-sys` / `ax-engine-py`.

The public results tables only count throughput where the caller can actually
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

The current single-batch diagnostic snapshot for Qwen is
`benchmarks/results/embedding/embedding-fair/2026-07-02-qwen-paired-cooldown15-refresh/2026-07-02-133329/`.
The current single-batch diagnostic snapshot for EmbeddingGemma is
`benchmarks/results/embedding/embedding-fair/2026-07-02-embeddinggemma-paired-cooldown15-refresh/2026-07-02-143425/`.
The current README Qwen ingest-scale snapshot is
`benchmarks/results/embedding/embedding-scale/2026-07-12-qwen-paired-v2/2026-07-12-145710/`
(schema `ax.embedding_ingest_scale.v2`, same-session paired 0.6B / 4B / 8B).
The current EmbeddingGemma ingest-scale snapshot is
`benchmarks/results/embedding/embedding-scale/2026-07-02-embeddinggemma-paired-cooldown15-refresh/2026-07-02-175206/`
with a later AX-only refresh for directional AX absolute numbers. The Qwen
paired artifacts use 2 warmups, 5 measured trials, 15-second cooldowns
before measured engine passes, alternating paired order, and median tok/s,
plus host / `runtime_identity` libmlx fingerprints. The fair artifacts keep
the complete short-query (ms/item primary) plus 16/64/256-token matrix in
`summary.md`. The Qwen reference comparison uses `mlx-lm` as the baseline
backend. EmbeddingGemma uses `mlx-embeddings` with mean pooling because
`mlx-lm` does not provide the comparable EmbeddingGemma route used by this
harness.

## Sustained vs intermittent profiles

The single-batch diagnostic harness models intermittent calls: every measured
reference or AX pass follows a 15-second cooldown, and the paired order
alternates between trials. The README intentionally does not publish those
latency-bound rows as headline throughput; it keeps public embedding rows
focused on sustained ingest and leaves the cooled single-batch artifacts for
query-latency investigation.

Use `--cooldown 0` only when the workload is truly sustained, such as a
vector-DB ingest worker or batch evaluation loop that keeps the GPU hot. Both
profiles are valid; choose the one that matches your workload's arrival
pattern, and do not mix cooled and hot-loop artifacts in one comparison.

## Large-corpus ingest scale

The current README scale snapshot is:

`benchmarks/results/embedding/embedding-scale/2026-07-12-qwen-paired-v2/2026-07-12-145710/`

It runs Qwen3-Embedding 0.6B 8-bit plus Qwen3-Embedding 4B/8B 4-bit DWQ against
`mlx-lm` in one same-session paired process, using last-token pooling and
l2-normalized contiguous CPU `float32 [B,H]` output for both backends. Each
trial embeds 512 deterministic chunks, with chunk lengths 256 and 512 and
batch sizes 8, 32, and 64. The harness reports median tok/s, chunks/s, output
MB/s, and per-batch p50/p95/max latency, and records `runtime_identity`
libmlx fingerprints so Homebrew-vs-pip linkage mismatches cannot silently
invalidate the delta.

Read the scale table differently from the single-batch diagnostic artifacts.
The diagnostic artifacts answer "how fast is this isolated batch shape when both
engines return a caller-consumable matrix?" The scale table answers "does that
rate hold when a RAG worker keeps feeding many batches, and what flush latency
does the chosen batch size create?" For the current Qwen snapshot, AX ranges
from 1.4% behind to 3.8% ahead across the 18 shapes, so read those rows as
sustained ingest parity (slightly favoring AX) rather than a stable per-shape
ranking.

Reproduce the scale snapshot with:

```bash
.venv/bin/python scripts/bench_embedding_ingest_scale.py \
  --model qwen3-embedding-0.6b-8bit=/path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha> \
  --model qwen3-embedding-4b-4bit-dwq=/path/to/Qwen3-Embedding-4B-4bit-DWQ/snapshots/<sha> \
  --model qwen3-embedding-8b-4bit-dwq=/path/to/Qwen3-Embedding-8B-4bit-DWQ/snapshots/<sha> \
  --batch-sizes 8,32,64 \
  --chunk-tokens 256,512 \
  --total-chunks 512 \
  --warmup 2 \
  --trials 5 \
  --cooldown 15 \
  --output-dir benchmarks/results/embedding/embedding-scale/$(date +%Y-%m-%d)-qwen-paired-refresh
```

Reproduce the EmbeddingGemma scale snapshot with:

```bash
.venv/bin/python scripts/bench_embedding_ingest_scale.py \
  --reference mlx_embeddings --pooling mean \
  --model embeddinggemma-300m-8bit=/path/to/embeddinggemma-300m-8bit/snapshots/<sha> \
  --batch-sizes 8,32,64 \
  --chunk-tokens 256,512 \
  --total-chunks 512 \
  --warmup 2 \
  --trials 5 \
  --cooldown 15 \
  --output-dir benchmarks/results/embedding/embedding-scale/$(date +%Y-%m-%d)-embeddinggemma-scale
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
not happen. Reported in the HTTP comparison table for comparison; not a
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

### Server-side length affinity + max_len buckets (default ON)

Multi-row `embed_batch_flat` / HTTP multi-input batches:

1. **Length-affinity split** (default ON) groups similar-length rows so
   right-pad waste stays bounded, then restores caller order.
2. **Calibrated max_len buckets** (default ON) snap pad width / compile
   keys with a conservative pad budget (`≤ max(16, 25% of max_len)`).

Kill switches (process env, restart required):

| Env | Default | Off |
| --- | --- | --- |
| `AX_EMBED_LENGTH_SPLIT` | on | `off` / `0` / `false` |
| `AX_EMBED_MAX_LEN_BUCKETS` | calibrated edges | `off` / `0` / `false` |

MTP adaptive draft gate remains **default OFF**
(`AX_MLX_MTP_ADAPTIVE_GATE=1` to experiment). See
`docs/designs/mtp-embed-perf-sprint-2026-07-16.md`.

## Cold start — `AX_MMAP_WEIGHTS`

The memory-mapped safetensors loader skips the userspace heap buffer +
`read()` pipeline that the default C loader does. Warm-cache wins:
-11% / -30% / -41% on 0.6B / 4B / 8B at session construction. True-cold
(post-`sudo purge`) measurement procedure and the criteria for flipping
to default-on are in [`EMBEDDING_COLDSTART.md`](EMBEDDING_COLDSTART.md).

Output is bit-exact with the C loader; opt in safely.

## Reproducing every README number

Use the fair in-process harness for public throughput claims:

```bash
.venv/bin/python scripts/bench_embedding_fair.py \
  --model qwen3-embedding-0.6b-8bit=/path/to/Qwen3-Embedding-0.6B-8bit/snapshots/<sha> \
  --model qwen3-embedding-4b-4bit-dwq=/path/to/Qwen3-Embedding-4B-4bit-DWQ/snapshots/<sha> \
  --model qwen3-embedding-8b-4bit-dwq=/path/to/Qwen3-Embedding-8B-4bit-DWQ/snapshots/<sha> \
  --batch-sizes 1,8 \
  --lengths 16,64,256 \
  --warmup 2 \
  --trials 5 \
  --cooldown 15 \
  --output-dir benchmarks/results/embedding/embedding-fair/$(date +%Y-%m-%d)-qwen
```

For EmbeddingGemma, use the same output contract with the embeddinggemma route:

```bash
.venv/bin/python scripts/bench_embedding_fair.py \
  --reference mlx_embeddings --pooling mean \
  --model embeddinggemma-300m-8bit=/path/to/embeddinggemma-300m-8bit/snapshots/<sha> \
  --batch-sizes 1,8 \
  --lengths 16,64,256 \
  --warmup 2 \
  --trials 5 \
  --cooldown 15 \
  --output-dir benchmarks/results/embedding/embedding-fair/$(date +%Y-%m-%d)-embeddinggemma
```

Pass `--ax-only` when you want to refresh only the AX path without loading the
reference backend; do not use that mode for public reference-comparison claims.

The legacy `scripts/bench_embedding_readme.sh` still runs HTTP serving and
cold-start paths. Use it for endpoint/cold-start evidence, not as the primary
embedding in-process publication source.
