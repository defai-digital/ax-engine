# Batched decode ceiling — the lever is real but the forward captures ~none of it

**Date:** 2026-07-17 · **MLX 0.32.0 (minos 26.2)** · **M3 Max** · Phase 3.0/3.1
audit for the decode-dispatch-efficiency plan.

## Why batched decode

Decode is bandwidth- and host-graph-encoding-bound
([`ax-decode-overlap-residual.md`](ax-decode-overlap-residual.md)): batch=1
reads all active weights and encodes the whole graph per token. A batched
forward reads the weights and encodes the graph once for B tokens, so aggregate
throughput should grow toward B×. This is the highest-ceiling remaining lever
(vLLM/llama.cpp show near-linear aggregate on bandwidth-bound decode).

## What the current code supports

`decode_batched_forward` / `BatchedDecodeSession` (default OFF,
`AX_MLX_BATCHED_DECODE`) is **dense full-attention only**. Structural
rejections (`architecture.rs::dense_batched_decode_structural_rejections`):
`moe`, `linear_attention`, `mla`, `sliding_window`, `layer_gating`, `diffusion`,
plus `mtp`. **Qwen3-Coder-Next (qwen3_next) is MoE + linear-attention → rejected
on both counts.** Extending to it means batched MoE and batched linear-attention
forwards — a large effort, which is why the ceiling had to be validated first.

## Ceiling measurement (dense, the supported path)

`batched-decode-ceiling-probe` (new microbench): prefill one 32-token prompt,
seed N clones into a `BatchedDecodeSession`, step the cohort, report aggregate
tok/s. Llama-3.1-8B-4bit, 0.32.0:

| batch | agg tok/s | per-req tok/s | step µs | scaling |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 53.3 | 53.3 | 18,766 | 1.00× |
| 2 | 58.7 | 29.4 | 34,065 | 1.10× |
| 4 | 64.0 | 16.0 | 62,529 | 1.20× |
| 8 | 66.3 | 8.3 | 120,698 | **1.24×** |

**Only 1.24× at batch=8** — step time grows ~6.5× for 8× batch, so the forward
barely amortizes. Far from the ×2–4 premise.

## Isolation — it is not what you'd guess

- **Not the compiled-path custom-kernel fallback.** The run spams
  `CustomKernel cannot infer output shapes` (the compiled dense-FFN path fails
  to shapeless-compile the packed-SwiGLU Metal kernel every layer and falls
  back — the same wall as MoE per-layer compile). But disabling it
  (`AX_MLX_DENSE_FFN_COMPILE=0 AX_MLX_DENSE_SWIGLU_PACKED_METAL=0`) gives
  **identical 1.24×** — the fallback is clean, not the cap.
- **The matmul ceiling is ~2.7×, not ~5× (earlier number was an artifact).**
  A per-call `mx.eval` around each matmul gives 5×, but that is amortizing the
  fixed submit overhead, not the weight read. Chaining 100 matmuls into one
  graph and evaluating once (the real forward's structure — one eval per step)
  gives the true weight-read amortization: **2.69× at batch=8** (1.62× @2,
  2.40× @4, 3.59× @16). It is not a shape issue — decode-shaped `[B,1,H]`
  matches 2D. So Apple Silicon 4-bit quantized matmul amortizes ~2.7× at
  batch=8 (compute-limited past that), and the forward nets 1.24× — still
  ~2× below the real matmul ceiling.
- **Not the attention/KV path (leading suspect REFUTED).** Sweeping the prefill
  length changes the per-request KV size 32× but barely moves scaling: seq=8 →
  1.24×, seq=256 → 1.20× at batch=8. If attention/KV were the non-amortizing
  cost it would degrade sharply with seq; it does not. The non-amortizing
  component is **seq-independent**.
- **What it is: a large seq-independent structural cost in the batched forward.**
  Solving the two-regime model (matmuls amortize 2.69×, the rest scales ~8×)
  against the measured b1/b8 step times puts ~69% of the batch=1 step in a
  non-matmul, seq-independent, batch-linear cost — far too large to be
  elementwise norms/RoPE. Pinpointing it requires stage-instrumenting the
  batched forward (`decode_batched_forward` / `layer_forward_batched`), the
  Phase 3.3 first move.

## Conclusion: viable, but the first work is fixing DENSE amortization

The lever is real — the weight-bound matmuls amortize ~5× at batch=8 in
isolation — but the AX batched forward captures almost none of it (1.24×). The
gap is **implementation**, not a hardware/MLX limit, and it is on the **dense**
path, before any MoE/linear-attention extension. The leading suspect for the
non-amortizing component is the per-request attention/KV path (each row has its
own KV cache; batched SDPA reads B caches and runs B attentions with no
cross-batch weight to amortize), plus possible structural overhead in
`layer_forward_batched`; confirming it needs stage-timing the batched step.

**Revised Phase 3 plan (updated after 3.2):**
- **Realistic ceiling: ~2.7× aggregate at batch=8** (the true matmul
  amortization on this hardware), i.e. ~53→~143 agg tok/s on Llama-8B, not the
  naive ×8. Meaningful for concurrent serving; zero for single-request.
- **3.2 DONE:** attention/KV ruled out (seq-independent); the gap is a large
  seq-independent structural cost in the batched forward, ~2× recoverable
  (1.24×→2.69×).
- **3.3 — stage-instrument `decode_batched_forward` / `layer_forward_batched`**
  (matmuls vs attention vs elementwise vs KV/mask/lm_head, batch 1 vs 8) to
  name the seq-independent non-amortizing stage, then fix it. Parity-gated.
- **3.4+ — extend to MoE / linear-attention** only after dense delivers toward
  the 2.7× ceiling.

This inverts the naive plan (extend to Coder-Next first): the measurement shows
the supported dense path itself only returns 1.24×, so that is where Phase 3
starts. The +28% auto-buffer-caps win remains the shipped single-request result;
batched decode is a real but currently-unrealized aggregate lever.

## Reproduction

```
cargo run -p ax-engine-microbench --release --bin batched-decode-ceiling-probe -- <dense_model_dir>
# vs the compile fallback:
AX_MLX_DENSE_FFN_COMPILE=0 AX_MLX_DENSE_SWIGLU_PACKED_METAL=0 cargo run ... batched-decode-ceiling-probe -- <dir>
```
