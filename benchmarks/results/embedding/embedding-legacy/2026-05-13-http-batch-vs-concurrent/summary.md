# /v1/embeddings batched POST vs concurrent-single POSTs

Captured on 2026-05-13. The new explicit-batch `/v1/embeddings`
contract (`{"input": [[ids],[ids],...]}`) goes directly to
`EngineSession::embed_batch_flat` and bypasses the per-request
microbatcher window; the concurrent-single path
(`{"input": [ids]}` × N) goes through `EmbeddingMicroBatcher` which
collects requests for up to 20 ms before coalescing.

Both are valid serving patterns — the explicit batch is for callers
that have the batch up front (vector-DB ingest, batch evaluation);
the concurrent-single path is for callers that can't see the batch
boundary up front (per-document async workers).

## Setup

- `ax-engine-server` started fresh per model on port 8083.
- 10-sentence corpus, lengths `[10,15,13,8,3,8,10,8,10,10]` (matches
  the rest of the embedding bench artifacts).
- 10 timed trials per path, 500 ms cooldown between trials, warmup
  on both paths first.
- Concurrent path: `concurrent.futures.ThreadPoolExecutor(max_workers=10)`
  firing `post_single` for each sentence in parallel.
- Batched path: single `post_batch(sentences)`.

## Results

| Model | Concurrent x10 (microbatcher) | Batched POST (explicit batch) | Batched advantage |
|---|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 30.48 ms (3,117 tok/s) | **23.82 ms (3,989 tok/s)** | **+28.0%** |
| Qwen3-Embedding 4B 4-bit | 58.88 ms (1,614 tok/s) | **52.86 ms (1,797 tok/s)** | **+11.4%** |
| Qwen3-Embedding 8B 4-bit DWQ | 86.22 ms (1,102 tok/s) | **81.10 ms (1,171 tok/s)** | **+6.3%** |

Pattern: batched POST wins most on the smallest model and least on
the largest. Why:

- The microbatcher's collection window adds up to 20 ms before the
  GPU dispatches. On the 0.6B model the GPU pass itself is ~12 ms;
  the window is a meaningful fraction of total wall time.
- 0.6B 8-bit has 10 HTTP requests vs 1 batched POST. Per-request
  Axum / Tokio / Hyper overhead is ~0.5 ms × 9 saved ≈ 4-5 ms.
  Combined with the saved window, the gap is ~7 ms = +28%.
- 4B / 8B: GPU dispatch dominates total wall time, so saved window
  + HTTP overhead is a smaller relative win. Still meaningful for
  serving-side cost — batched POST gives ~11% / ~6% extra throughput
  with no caller-side changes other than re-shaping the request body.

## Recommendation

Callers that **have the batch up front** should send it as one
explicit-batch POST. The microbatcher remains the right path for
workloads where the batch boundary isn't visible to the client
(fan-out async ingest, per-document workers, multi-tenant request
queues).

## Reproducing

```bash
bash benchmarks/results/embedding/2026-05-13-http-batch-vs-concurrent/run.sh
```

Writes a `<model>.json` per model in this directory.
