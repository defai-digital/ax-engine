# Gemma 4 E2B Direct Pipeline Op-Count Baseline (2026-05-14)

This artifact captures the first measurement from the W1.1 op-count
instrumentation introduced under `MLX-DECODE-SUBMIT-COST-PRD.md`.

## Setup

- Model: `.internal/models/gemma-4-e2b-it-4bit`
- Prompt shape: 128 prompt tokens, 128 generation tokens
- AX policy: `direct_no_ngram_acceleration`
- Repetitions: 3 measured runs
- Host: Apple M5 Max, 128 GB
- Build: includes the new `mlx-sys::op_count` thread-local counter and the
  `ax_mlx_direct_pipeline_op_count` telemetry bucket.

## Headline

| Metric | Value |
|---|---:|
| Decode tok/s (median across reps) | 185.4 |
| Direct-pipeline steps | 381 |
| Direct-pipeline wall µs/step | 5,340 |
| `async_eval` wall µs/step | 4,910 |
| `forward` wall µs/step | 423 |
| **Direct-pipeline op count per step** | **1,805** |

## Reading

The 2026-05-14 barrier A/B (`../2026-05-14-gemma-e2b-direct-pipeline-barrier-ab/`)
established that ~3,674 µs/step of the production `async_eval` bucket is
host-side MLX submit/encode work. With 1,805 op dispatches per step, the
average per-op host cost is **~2.0 µs/op**.

This is the W1 working budget. Closing the ~521 µs/step gap to mlx_lm requires
either:

- **Reducing op count** (W1.2: KV append batching; W1.3: redundant cast/reshape
  removal). Each op removed saves ~2 µs/step of submit cost. Hitting the
  ~521 µs target needs ≈260 fewer ops/step (a 14% reduction).
- **Lowering per-op submit cost** (W2: mlx_compile wrap). Same 521 µs target
  could come from cutting per-op cost from 2.0 µs to 1.7 µs (a 15% reduction)
  with op count unchanged. mlx_compile would aim for both.

## Per-layer attribution (for W1.2 sizing)

Gemma 4 E2B has 35 transformer layers. Subtracting the prelude/epilogue
(~20 ops: embed, scale, per-layer-input gating, final norm, lm_head, argmax,
async_eval), the per-layer cost is roughly:

```
(1805 - 20) / 35 ≈ 51 ops per layer
```

A typical Gemma 4 layer (decode, no compile) executes:

- QKV projection: ~3 quantized matmuls + slices/reshapes (~10 ops)
- QK norm + RoPE: ~6 ops
- KV append: 2 `slice_update` (one each for K, V) + a few helpers
- SDPA: 1 op
- Output projection: ~3 ops
- Residual + FFN norm: ~4 ops
- pli-gate (Gemma 4 specific): ~6 ops
- FFN swiglu: ~6 ops (gate, up, down, silu, multiply)
- Residual + layer gate: ~3 ops

That sums to ~41 ops/layer; the gap to 51 is per-layer overhead in
`compute_per_layer_inputs_arr` slicing or transient `astype`/`reshape` not
visible from the high-level walk.

## W1.2 sizing (KV append batching candidate)

**Initial sizing (~26% of the gap) was overstated.** The 2026-05-14
pre-check (`.internal/planning/MLX-DECODE-W12-KV-APPEND-PRECHECK.md`)
revised this:

- Per-layer KV cost is **4 ops** (2 `slice_update` + 2 `slice` for the
  SDPA-input view), not 2 — so total is **~140 ops/step from KV cache**,
  not 70.
- Gemma 4 E2B has two head_dim values (sliding=256, full=512), so the
  cross-layer single-tensor stack is unsound.
- Two narrower variants (sliding-only stack ~54 ops; slice-cache fusion
  up to ~70 ops) together cover at most ~50% of the 521 µs gap with
  medium-to-high engineering scope. PRD W1.2 has been parked; W1.3 is
  the next workstream.

## Artifacts

- `baseline.json` — full bench output (3-rep, p=128, g=128, AX direct only)
- Prompts: identical sha256 (`4ebdfdf02961…`) to the 2026-05-14 baseline
