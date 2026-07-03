# Fused `add + rms_norm` Metal kernel attempt — REVERTED

## Hypothesis

The `AX_MLX_DECODE_PROFILE=1` run on Qwen 3.6 27B 4-bit (see
`../diagnose-decode-gap/`) showed the post-attention "residual + ffn_norm"
boundary at 52.7% of the profile share. Fusing `add(residual, x)` and
`rms_norm(_, weight, eps)` into one custom Metal kernel would skip one
mlx-c FFI hop per layer (~64 hops/step on a 64-layer model) and save
one intermediate tensor read.

## A/B result (M5 Max, 30 s thermal settle + 20 s inter-row cooldown, 3 reps)

|    | OFF prefill | ON prefill | Δprefill | OFF decode | ON decode | Δdecode |
|----|-------------|------------|----------|------------|-----------|---------|
| 128  | 820.3 | 817.2 | −0.38% | 33.40 | 32.88 | **−1.55%** |
| 512  | 954.2 | 951.7 | −0.27% | 33.29 | 32.79 | **−1.52%** |
| 2048 | 955.4 | 944.2 | **−1.17%** | 32.91 | 31.91 | **−3.02%** |

Decode and prefill both regressed. The fused kernel is **slower** than
MLX's split `add` + `rms_norm`.

## Why the fusion lost

MLX's standalone `rms_norm` is highly optimized (likely vectorized
threadgroup reduction with SIMD-wide loads). The naive 256-thread
two-phase reduction in `ax_rmsnorm_add_v1` is simpler but slower per
GPU dispatch, and that GPU-side loss outweighs the ~1 µs/layer FFI hop
saved.

Quantitatively: at 64 layers × 30 ms/step (decode), the FFI savings
ceiling is ~64 µs/step = ~0.2%. The custom kernel added more than 0.2%
of GPU time per dispatch, so the net was negative.

## Code reverted at commit

Implementation lived in `crates/ax-engine-mlx/src/rmsnorm_ops.rs`
behind `AX_MLX_RMSNORM_ADD_METAL` (default-on). Reverted on the same
investigation pass; not present in HEAD. The kernel itself was
correctness-verified (4 equivalence tests vs split path, all pass) —
the issue was raw throughput, not correctness.

## What the next investigator should NOT do

- Re-attempt the same fusion with a naive Metal kernel. The win is too
  small to overcome MLX's internal `rms_norm` optimization.

## What might still work

- **Fuse with the next operation downstream.** The `normed2` output
  immediately feeds an FFN gate-up matmul. A fused
  `add + rms_norm + qw(gate_up_proj)` kernel could pay for itself
  because the matmul amortizes the kernel-launch cost.
- **mlx.compile-analog** for the whole per-layer forward. Out of scope
  for a one-kernel attempt; would need a graph-level investigation.
- **Profile MLX's `rms_norm` directly** (Metal performance counters)
  to see what the floor is. If it's already at peak memory bandwidth,
  no add-side fusion can beat it.

## Raw artifacts

- `off/qwen3_6-27b-4bit.json` — env=0, baseline matches canonical sweep within 1.5%.
- `on/qwen3_6-27b-4bit.json` — env=1, fused kernel active.
- Sibling dir `../diagnose-decode-gap/qwen3_6-27b-4bit.json` — the
  `AX_MLX_DECODE_PROFILE=1` run that motivated the attempt.
