# R1 + R2 Rust improvements — measurement summary

Captured on 2026-05-12 with the C1 compile-cache landed (commit 765272f3).

## R1: `embed_batch_flat` — one contiguous `Vec<f32>` instead of `Vec<Vec<f32>>`

Input: 10 sentences with lengths `[10,15,13,8,3,8,10,8,10,10]`, last-token
pooling, l2-normalized. 5 warmup + 15 timed reps, no cooldown (sustained).

| Model | `embed_batch` (Vec<Vec>) ms | `embed_batch_flat` (Vec<f32>) ms | Δ |
|---|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 10.94 | 10.83 | -1.0% |
| Qwen3-Embedding 4B 4-bit | 40.48 | 40.19 | -0.7% |
| Qwen3-Embedding 8B 4-bit DWQ | 66.67 | 66.62 | -0.1% |

Perf delta is small because GPU compute dominates total wall time; the
main value of `embed_batch_flat` is *API* — saves `B - 1` heap
allocations and hands the caller one contiguous buffer that downstream
code (numpy, faiss, HNSW) can mmap or wrap zero-copy.

## R2: pipeline tokenize (CPU) + embed (GPU)

Input: 64 randomly-generated short English sentences (8–25 words each,
seed=42), batched into chunks of 8 (8 chunks total). Tokenized in-thread
via `EngineTokenizer` for the serial path; tokenized in a producer
thread feeding a `mpsc::sync_channel(1)` for the pipelined path.

| Model | Serial total | Pipelined total | Speedup |
|---|---:|---:|---:|
| Qwen3-Embedding 0.6B 8-bit | 124.37 ms | 114.00 ms | **+9.1%** |
| Qwen3-Embedding 4B 4-bit | 375.06 ms | 366.18 ms | +2.4% |
| Qwen3-Embedding 8B 4-bit DWQ | 654.03 ms | 647.84 ms | +1.0% |

Pattern: pipelining helps most where the GPU pass is short relative to
CPU tokenization. 0.6B with batch=8 gets ~9% because GPU compute
(~10–14 ms / chunk) and CPU tokenization (~1–2 ms / chunk) are within
an order of magnitude, so overlap recovers most of the tokenize cost.
4B and 8B saturate the GPU per chunk, so the tokenize thread runs ahead
quickly and the channel fills — the GPU thread is the bottleneck the
whole way.

## Reproducing

```bash
# R1
cargo build -p ax-engine-bench --example embed_rust_bench --release
./target/release/examples/embed_rust_bench \
    --model-dir .internal/models/qwen3-embedding-0.6b-8bit \
    --batch 10,15,13,8,3,8,10,8,10,10 \
    --warmup 5 --trials 15

# R2
cargo build -p ax-engine-bench --example embed_pipeline_demo --release
./target/release/examples/embed_pipeline_demo \
    --model-dir .internal/models/qwen3-embedding-0.6b-8bit \
    --batch-size 8 < texts.txt
```
