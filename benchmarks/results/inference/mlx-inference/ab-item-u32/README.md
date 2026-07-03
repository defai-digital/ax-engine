# Single-FFI `item_u32()` attempt — REVERTED

## Hypothesis

mlx_lm's per-step decode reads the just-argmaxed token via `y.item()`,
which in C++ resolves to `array::item<uint32_t>()` and internally fuses
`eval()` + `*data<T>()` into **one** C call. ax-engine's `decode_step`
splits this into two FFI hops: `mlx_eval([&pending])` then
`mlx_array_data_uint32`. At ~1-5 µs FFI overhead per call × 128 decode
steps the saved hop could close ~128-640 µs of the per-request decode
gap to `mlx_lm.benchmark` — landing in the 1-6% gap band identified
in `docs/PERFORMANCE-DECODE-GAP.md`.

Implemented `MlxArray::item_u32()` bound to `mlx_array_item_uint32` in
`crates/mlx-sys/src/array.rs` and routed
`advance_direct_pipeline_with_timings_and_turboquant_context`
(`crates/ax-engine-mlx/src/generate.rs`) through it.

## Result (M5 Max, 30 s thermal settle + 20 s inter-row cooldown, 3 reps, vs canonical 2026-05-20 sweep)

|    | canon prefill | NEW prefill | Δpf | canon decode | NEW decode | Δdec | NEW vs mlx_lm |
|----|---------------|-------------|-----|--------------|------------|------|---------------|
| 128  | 808.3 | 815.6 | +0.90% | 33.19 | 32.97 | **−0.65%** | 97.1% |
| 512  | 951.4 | 944.9 | −0.68% | 33.21 | 32.60 | **−1.86%** | 96.1% |
| 2048 | 950.8 | 941.0 | −1.04% | 32.85 | 32.16 | **−2.11%** | 96.2% |

Decode regressed across all prompt sizes; prefill regressed at 512 and

2048. vs the `mlx_lm` baseline this fell from ~98% to ~96-97%.

## Why the fusion lost

Counter to the hypothesis, the single C++ `array::item<T>()` call is
**slower** than the split `mlx_eval([&pending]) + data<T>()` path on
the decode hot path. Likely cause: `array::item<T>()` calls
`this->eval()` (on a single array) which routes through a different
scheduler entry point than `mlx_eval(std::vector<array>)`. The single-
array eval may carry extra synchronization or queue-drain semantics
that the vector-eval path optimizes.

The exact mechanism would need MLX-side instrumentation to confirm.
What matters here: empirical wall-clock says fusing the FFI does not
win, so revert.

## Code reverted at HEAD

The `MlxArray::item_u32()` wrapper was added to
`crates/mlx-sys/src/array.rs` and the decode hot path was switched to
it in `crates/ax-engine-mlx/src/generate.rs`. Reverted on the same
investigation pass; neither change is in HEAD. The wrapper itself
was correctness-equivalent (the existing `eval_first_u32` in
`crates/mlx-sys/src/transforms.rs:32` does the same end result via
the split path) — the issue is purely throughput.

## What the next investigator should NOT do

- Re-attempt single-FFI scalar-readback fusion via `mlx_array_item_*`.
  Empirically slower than the split path on the decode hot path.
- Re-attempt single-FFI fusion of the `eval + data` pattern more
  generally without first measuring the underlying MLX scheduler
  behavior for single-array vs vector-array eval.

## What might still work

- A `mx.compile`-analog wrapping the whole per-layer decode forward
  (the build_embedding_forward_closure pattern at
  `crates/ax-engine-mlx/src/model/mod.rs:718`). This amortizes per-op
  FFI globally and would also amortize the eval-readback FFI; the
  decode-gap doc lists it as a remaining out-of-scope angle.
- Profile `array::item<T>()` vs `mlx_eval` directly in mlx-c to
  confirm the slow-path hypothesis before any further FFI fusion.

## Raw artifacts

- `qwen3_6-27b-4bit.json` — 3 reps × 3 prompts at env-on (item_u32
  routed), `--ax-direct`. mlx_lm reference rows reused from the
  canonical 2026-05-20-qwen-1024-tier-sweep.
- See sibling `../ab-direct-cpp-linear-inputs/` and `../ab-rmsnorm-add/`
  for the prior two attempts in this investigation chain.
- Permanent finding: `docs/PERFORMANCE-DECODE-GAP.md`.
