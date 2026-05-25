# Direct Mode Decode Progress — 2026-05-25

## Goal

Improve AX Engine MLX direct mode until direct decode token/s is better than
`mlx_lm.benchmark` for every published model row and prompt shape.

This report records the current progress from the 2026-05-25 continuation. It
is not a completion report: Gemma 31B and Qwen 27B remain unresolved.

## Current Source Changes

Verified source work currently in the worktree:

- Split Gemma GEGLU Metal path for already-separated gate/up tensors.
- Gemma4 per-layer input gate projection path using
  `gelu_approx(gate) * per_layer_input -> quantized_matmul`.
- Default-on fast-path flag `AX_MLX_GEGLU_MUL_METAL`.
- Existing dense packed GEGLU and per-layer gate tests still pass.

In-progress, not validated yet:

- Began a narrower dense-Gemma residual/norm shim in `mlx-sys`
  (`add_rms_norm_pair`, `quantized_matmul_rms_norm`). This was started after
  profiling Gemma 31B, but routing and validation were not completed before
  this report.

## Validation Run So Far

Passed before the latest benchmark runs:

```bash
cargo test -p ax-engine-mlx geglu_mul_metal_uses_default_on_kill_switch_contract --quiet
cargo test -p ax-engine-mlx split_geglu_metal --quiet
cargo build --release -p ax-engine-server
```

## Gemma 26B A4B Result

Artifact:

- `/private/tmp/ax-direct-gemma-26b-a4b-4bit-split-geglu-current.json`

Result: now passes all prompt shapes.

| Prompt | mlx_lm decode | AX direct decode | Ratio | Status |
|---:|---:|---:|---:|---|
| 128 | 127.9 tok/s | 131.9 tok/s | 1.031x | PASS |
| 512 | 125.0 tok/s | 129.0 tok/s | 1.031x | PASS |
| 2048 | 119.3 tok/s | 123.9 tok/s | 1.038x | PASS |

Interpretation: the split GEGLU Metal path fixes the 26B A4B direct decode gap.
This row was previously below `mlx_lm` on every prompt shape.

## Gemma 31B Result

Artifact:

- `/private/tmp/ax-direct-gemma-31b-4bit-split-geglu-current.json`

Result: still fails all prompt shapes.

| Prompt | mlx_lm decode | AX direct decode | Ratio | Status |
|---:|---:|---:|---:|---|
| 128 | 28.9 tok/s | 28.4 tok/s | 0.982x | FAIL |
| 512 | 28.3 tok/s | 27.8 tok/s | 0.981x | FAIL |
| 2048 | 27.0 tok/s | 26.1 tok/s | 0.967x | FAIL |

Interpretation: Gemma 31B is dense Gemma4, so it does not benefit from the
split MoE expert GEGLU path the same way Gemma 26B A4B does. Telemetry shows
the dense packed gate/up path is already active.

## Gemma 31B A/B Attempts

All A/B attempts below are prompt=128 only and should not be promoted.

| Experiment | Artifact | AX decode | Ratio vs mlx_lm | Result |
|---|---|---:|---:|---|
| `AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN=1` | `/private/tmp/ax-direct-gemma-31b-4bit-post-attn-ffn-p128.json` | 27.7 tok/s | 0.959x | Worse |
| `AX_MLX_DIRECT_CPP_QK_NORM_ROPE=1` | `/private/tmp/ax-direct-gemma-31b-4bit-qk-norm-rope-p128.json` | 28.2 tok/s | 0.977x | Worse |
| `AX_MLX_PACK_DENSE_FFN_GATE_UP=0` | `/private/tmp/ax-direct-gemma-31b-4bit-no-packed-ffn-p128.json` | 28.2 tok/s | 0.975x | Worse |
| `AX_MLX_ROTATING_SLIDING_DECODE=0` | `/private/tmp/ax-direct-gemma-31b-4bit-no-rotating-p128.json` | 27.4 tok/s | 0.950x | Worse |
| `AX_MLX_LAYER_SCALAR_FUSED_ADD=0` | `/private/tmp/ax-direct-gemma-31b-4bit-no-layer-scalar-p128.json` | 27.7 tok/s | 0.958x | Worse |

Notes:

- The whole post-attention FFN C++ route hit 100% of eligible layers but still
  regressed, so the larger C++ boundary is not a promotion candidate.
- Rotating sliding decode is beneficial for this row; disabling it makes the
  gap much larger.
- Dense gate/up packing is not the cause of the 31B gap.

## Gemma 31B Decode Profile

Artifact:

- `/private/tmp/ax-direct-gemma-31b-4bit-decode-profile-p128.json`

The profile inserts barriers, so its decode tok/s is not a benchmark result.
Use it only for stage ranking.

Key stage totals:

| Stage | Wall time |
|---|---:|
| pre-SDPA | 5,977,421 us |
| pre-SDPA QKV projection | 2,085,555 us |
| pre-SDPA QK norm | 1,282,320 us |
| pre-SDPA RoPE/KV | 2,580,033 us |
| SDPA | 1,349,409 us |
| post-attention | 12,557,978 us |
| post-attention FFN | 8,199,455 us |
| FFN gate/up | 3,372,429 us |
| FFN activation | 1,364,647 us |
| FFN down | 2,312,367 us |
| output projection | 1,783,998 us |
| residual norm | 1,298,847 us |
| residual gate | 1,257,494 us |
| lm head | 203,914 us |

Interpretation: the remaining 31B gap is likely in dense Gemma post-attention
or attention orchestration, not in the split GEGLU expert path.

## Current State by Model Family

Current evidence says:

- Gemma E2B/E4B rows had already moved above `mlx_lm` with the existing
  per-layer and packed GEGLU work.
- Gemma 26B A4B now passes all prompt shapes with split GEGLU Metal.
- Gemma 31B still fails all prompt shapes.
- Qwen 27B remains unresolved from the earlier stale artifact set and still
  needs current-worktree reruns plus Qwen-specific optimization.
- Qwen 35B A3B was passing in the older artifact set, but should still be
  re-audited before closing the goal.

## Next Steps

1. Finish or discard the in-progress dense-Gemma residual/norm shims based on
   compile, unit tests, and a Gemma 31B p128 A/B.
2. If the narrow dense-Gemma shim does not beat `mlx_lm`, profile a smaller
   attention/KV boundary instead of promoting the regressing whole-block route.
3. After Gemma 31B passes, rerun Qwen 27B 4/5/6/8-bit current-worktree direct
   rows and optimize the actual failing stage.
4. Only mark the goal complete after current-worktree evidence proves AX direct
   decode is above `mlx_lm` for every model row and every prompt shape.
