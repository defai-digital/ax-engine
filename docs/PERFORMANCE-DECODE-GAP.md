# Direct-decode gap to `mlx_lm.benchmark`

## Summary

On the 2026-05-20 sweep, ax-engine direct decode trails `mlx_lm.benchmark`
by 1–6% on every Gemma row and every Qwen 3.6 27B variant. Qwen 3.6
35B-A3B is the only model where ax wins (+3%). This document records why
the gap exists, what was tried, and what is out of scope, so future
investigators do not relitigate the same experiments.

## Measured gap

From `benchmarks/results/mlx-inference/2026-05-20-qwen-1024-tier-sweep/`:

| family | decode rate | Δ vs mlx_lm | per-step extra µs |
|---|---|---|---|
| Gemma 4 E2B 4-bit | 198 t/s | −5.7% | +288 µs |
| Gemma 4 E2B 8-bit | 142 t/s | −4.7% | +330 µs |
| Gemma 4 26B-A4B | 121 t/s | −3.3% | +271 µs |
| Gemma 4 31B | 28.0 t/s | −1.2% | +430 µs |
| Qwen 3.6 27B 4-bit | 33.2 t/s | −2.0% | +622 µs |
| Qwen 3.6 27B 8-bit | 18.2 t/s | −2.3% | +1241 µs |
| Qwen 3.6 35B-A3B | 141 t/s | +3.3% | −236 µs |

The extra cost is **roughly proportional to decode wall time per step**,
not a fixed Rust runtime overhead. Faster models (fewer ms/step) show
the gap more visibly as a percentage.

## What is *not* the gap

The investigation ruled out the following candidates with measured
evidence:

- **QKVZ projection fastpath** (`AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_INPUTS`).
  Fusing the packed QKVZ + reshape + slice + concat into one mlx-c call.
  Result: neutral within ±0.3% — confirmed at
  `benchmarks/results/mlx-inference/ab-direct-cpp-linear-inputs/`
  (commits `7ddd0ca` → `9a054da`). The flag stays opt-in.
- **Linear-attention post-input fastpath**
  (`AX_MLX_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT`). Fuses the chain
  `conv1d(+ cached state carry) -> SiLU -> last-dim split -> head-major
  reshape -> per-head RMSNorm on q/k -> scale-by-precomputed-constants`
  into one Rust-to-C++ FFI round-trip. A Qwen 3.6 35B-A3B 4-bit
  `decode-trace` smoke with packed linear-attention inputs enabled showed
  real route reach (`420/420` post-input hits, zero fallbacks) and reduced
  linear-attention op count from 42.0 to 26.0 ops/layer. The same short
  warmed trace was throughput-neutral/noisy (`157.59` vs `154.30` tok/s),
  so that smoke was **not** promotion evidence. A later real inference-stack
  A/B on the same model
  (`benchmarks/results/mlx-inference/2026-05-21-qwen35-post-input-ab/`)
  proved full route coverage (`150/150` post-input hits at each prompt length,
  zero fallbacks) and small positive decode medians at p128/p512/p2048
  (+0.42%, +0.62%, +0.51%). The flag still stays opt-in because this is a
  sub-1% gain, not a strong default-on promotion margin. The decision is now
  guarded by `scripts/check_qwen_post_input_route_promotion.py`, which requires
  paired route flags, all-hit summaries, zero fallback/profile-blocked counters,
  and a minimum decode ratio of 1.01; the current artifact is `not_promoted`.
- **Fused `add + rms_norm` Metal kernel** for the post-attention
  residual + ffn_norm boundary. The decode profile identified this stage
  as 52.7% of `AX_MLX_DECODE_PROFILE=1` wall share. The custom kernel
  was correctness-equivalent (4 round-trip tests vs split path) but
  **slower than MLX's stand-alone `rms_norm`**: decode regressed 1.5–3%,
  prefill@2048 regressed 1.2%. Reverted; raw artifacts at
  `benchmarks/results/mlx-inference/ab-rmsnorm-add/` (commit `ec5dda1`).
- **Single-FFI scalar token readback** (`MlxArray::item_u32`
  wrapping `mlx_array_item_uint32`, mirroring mlx_lm's `y.item()`).
  Collapses the `mlx_eval([&pending]) + mlx_array_data_uint32` pair
  into one C call. Theoretically saves ~1-5 µs FFI per step × 128
  steps. Empirically **regressed** decode 0.65–2.11% and prefill
  0.7–1.0% on Qwen 3.6 27B 4-bit, dropping from 98% to 96–97% of
  mlx_lm. Likely cause: `array::item<T>()` calls a single-array
  `eval()` path that has different (slower) scheduler semantics than
  the `mlx_eval(vector<array>)` used by `mlx-sys::transforms::eval`.
  Reverted; raw artifacts at
  `benchmarks/results/mlx-inference/ab-item-u32/`.
- **`WeightLayoutTelemetry::from_weights` per step** — moved to runner
  init in the same investigation. Tiny win (< 0.01%) but the loop was
  pure dead weight; kept for cleanliness.
- **Page-out / `wired_limit`** — already set at runner construction
  (`runner.rs:2595-2598`), matching mlx_lm's `wired_limit` context
  manager. Not the gap.
- **`enable_compile()`** — already on (`runner.rs:2585`). Not the gap.
- **Per-step `Instant::now()` and counter recording** — measured at
  ~600 ns / step, three orders of magnitude smaller than the gap.

## Why the gap exists

After ruling out the candidates above, the residual gap is best
explained as **per-MLX-op FFI overhead that accumulates over the
~13–20 ops per layer × 64 layers ≈ 800–1300 op dispatches per decode
step**. mlx_lm calls MLX from Python through `nn.Module.__call__`,
which dispatches a series of `mx.*` ops the same way ax-engine does
through `mlx-sys`. Both pay per-op cost; ax-engine's Rust→C FFI is
not faster than mlx_lm's pybind11.

A single fused Metal kernel saves at most a few µs per fused site (the
saved FFI hop + maybe a tensor read). 64 fusion sites × ~1 µs ≈ 64 µs
per step, which is ~0.2% of a 30 ms step. Closing 2% needs ten
independent fusions in concert, or a graph-level compile pass that
MLX's `mx.compile` provides automatically.

The exception (Qwen 3.6 35B-A3B, +3%) lines up: A3B's per-token
compute is small (3 B active params, 40 layers), so the Python
per-layer call overhead in mlx_lm becomes proportionally larger than
ax-engine's Rust per-layer call. The crossover is around 140 t/s
decode rate.

## What might still work (out of scope here)

1. **`mx.compile`-analog for the per-layer decode forward**. Wrap the
   per-layer Q/K/V → SDPA → FFN sequence in a compiled MLX graph so
   per-op dispatch is amortized once at first call. Requires
   `mlx-sys` exposure of `mlx_compile` plus careful handling of the
   KV-cache as a non-traced output. Estimated leverage: closes most
   of the 1–6% gap on Gemma; partial on Qwen 27B because the
   GatedDelta recurrent kernel is already fused.
2. **Whole-layer Metal kernel** for the linear-attention block on
   Qwen 27B. Would combine RMSNorm + QKVZ/BA projection + conv +
   gated-delta + output projection into one dispatch. Massive
   refactor; high risk of correctness drift.
3. **Profile MLX's `rms_norm` directly with Metal performance
   counters** to confirm it is bandwidth-bound. If so, no add-side
   fusion can beat it (the prior `ab-rmsnorm-add` attempt is
   consistent with this).

## Acceptance for this investigation

Closed without a measurable production-positive code change beyond
the `WeightLayoutTelemetry` cleanup. The 1–6% gap is treated as the
inherent cost of running a production runtime (engine state machine,
scheduler, request manager, telemetry, validation, request lifecycle)
on top of MLX, vs. running mlx_lm's tight benchmark loop. Future
attempts should require either an `mx.compile`-analog (item 1 above)
or whole-layer fusion (item 2) — single-op-fusion attempts have a low
ceiling and have already been tried.

## Diagnostic instrumentation retained

The follow-up Qwen linear-attention work added finer telemetry for the host
side of direct decode. The diagnostic surface is kept merged because it is
independently useful for any future MLX-host work:

- `ax_mlx_direct_pipeline_forward_layer_loop_wall_us` /
  `ax_mlx_direct_pipeline_forward_head_wall_us` route decisions
  split `forward_wall_us` so a single bench reveals whether host
  cost lives in the layer loop, the lm-head, or the embed prologue
  (`runner.rs::DecodeTelemetry`).
- `DirectPipelineTimings::linear_attention_layer_ops` /
  `full_attention_layer_ops` per-layer-kind FFI op counts via
  `mlx_sys::op_count` brackets around each layer's forward
  (`model/mod.rs` layer loop, `generate.rs::ForwardStageTimings`).
- `ax_mlx_linear_attention_qkvz_ba_packed_layers` /
  `ax_mlx_linear_attention_split_qkvba_layers` route decisions
  confirm whether load-time projection packing actually engaged
  on a given checkpoint (`runner.rs::WeightLayoutTelemetry`).
- `cargo run --release --bin pack-audit -- <model_dir>` - load
  weights and print per-layer pack state, no graph eval. Use
  before running any benchmark to confirm the artifact actually
  consumes the packed paths the code intends.
- `cargo run --release --bin decode-trace -- <model_dir> [steps]` -
  direct-pipeline decode loop with the full per-step host wall
  breakdown, including the per-layer-kind op counts. Use for
  isolated host-side experiments that don't need the full bench
  sweep harness.

## Next Qwen investigation

Do not continue optimizing the Qwen linear-attention post-input route in
isolation. It already proves route reach and dispatch reduction, but the
cooled A/B stayed below the promotion gate. The next Qwen work should first
rank the remaining bottleneck with:

- `scripts/run-gateddelta-prefill-profile.sh` on Qwen 3.6 35B A3B 4-bit, to
  split prefill time across projection, conv, Q/K norm, recurrent, and output
  stages.
- `decode-trace` with both linear-attention direct routes enabled, to identify
  the remaining decode host/op-count region before any larger layer-wide probe.

Any new direct-MLX Qwen route still needs clean paired inference-stack rows,
explicit route flags, all-hit summaries, zero fallback/profile-blocked counters,
and a repeated decode ratio of at least 1.01 before it can be considered for
default-on promotion.
