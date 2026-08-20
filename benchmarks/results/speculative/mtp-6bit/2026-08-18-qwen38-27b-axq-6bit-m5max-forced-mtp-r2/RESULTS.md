# Qwen3.8 27B AXQ 6-bit MTP A/B on M5 Max

Status: **diagnostic only — share with caveats; not a publication candidate**.

On an Apple M5 Max MacBook Pro with 128 GB unified memory, AX Engine direct
decode reached **30.73 tok/s** versus **25.86 tok/s** with exact depth-1 MTP
forced for every decode step. MTP delivered **0.842x** direct throughput, a
**15.83% regression**. These headline values are geometric means across the 11
prompt-case decode medians.

| Prompt suite | Cases | Direct tok/s | Forced MTP tok/s | MTP/direct | Delta |
|---|---:|---:|---:|---:|---:|
| `flappy` | 4 | 30.70 | 25.90 | 0.844x | -15.63% |
| `long_code` | 4 | 30.71 | 26.42 | 0.860x | -13.96% |
| `python_modules_long` | 3 | 30.79 | 25.09 | 0.815x | -18.53% |
| **All prompt cases** | **11** | **30.73** | **25.86** | **0.842x** | **-15.83%** |

Every one of the 11 prompt-case medians was slower with forced MTP. The median
across all case medians was 30.74 tok/s direct and 25.80 tok/s MTP (-16.06%),
which agrees with the geometric-mean result.

## What was tested

- Model package: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP`, revision
  `3e290738e96972307c6aeb9934ab170ca0eae1c1`.
- AX Engine 7.1.2, commit `4c1a37a071c68f9da1a9380ac027af1598e31cb6`,
  clean release build, MLX 0.32.0.
- Single request, batch size 1, 1,000 generated tokens, no thinking, cold prefix
  cache, 2 warmups, 5 measured repetitions, and 15-second cooldowns.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, seed 0.
- MTP used the Qwen exact verifier at depth 1. This package has one MTP head;
  it is not an MTP=3 configuration.
- N-gram stacking was disabled. `AX_MLX_MTP_BYPASS_THRESHOLD=0` and
  `AX_MLX_MTP_MIN_REMAINING_TOKENS=0` disabled the adaptive and short-tail
  bypasses; `AX_MLX_MTP_PROFITABILITY_GATE=0` disables request-local cost
  calibration in runtimes that support it. The on/off comparison therefore
  could not silently become mixed-mode.

The MTP artifacts report 89.55% aggregate verified acceptance, 100% MTP step
coverage, zero direct fallback steps, zero short-budget bypass steps, zero
n-gram activity, and zero correctness-mode conflicts.

## Validation caveats

- All 11 direct rows passed the run-stability gate; only 4 of 11 MTP rows did.
  The remaining MTP rows were classified as high variance or tail regression.
- The direct route reproduced the same 1,000 output token IDs across all five
  repetitions for every prompt. MTP produced five distinct output sequences
  for every prompt despite the fixed seed. Runtime telemetry marks every MTP
  row as distribution-exact sampled MTP, but the stricter benchmark
  seed-reproducibility oracle therefore fails.
- During the final suite, a macOS background analysis process crossed the 50%
  CPU gate and the harness waited before later samples. It may have contributed
  to the 14.01 and 20.33 tok/s outliers, but causation is not established.
- The formal summary generator intentionally rejected this run at its first
  unstable row. No publication `summary.json` or README claim was generated.

The practical recommendation for this model package on this machine is to keep
the normal profitability bypass enabled, or use direct decode. Forcing MTP on
throughout was slower on this workload.

## Raw artifacts

- `flappy`: [direct](qwen3.8-27b-axq-6bit/flappy/ax_direct.json),
  [MTP](qwen3.8-27b-axq-6bit/flappy/ax_mtp.json)
- `long_code`: [direct](qwen3.8-27b-axq-6bit/long_code/ax_direct.json),
  [MTP](qwen3.8-27b-axq-6bit/long_code/ax_mtp.json)
- `python_modules_long`:
  [direct](qwen3.8-27b-axq-6bit/python_modules_long/ax_direct.json),
  [MTP](qwen3.8-27b-axq-6bit/python_modules_long/ax_mtp.json)
