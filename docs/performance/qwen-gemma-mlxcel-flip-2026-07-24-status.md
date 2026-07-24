# Qwen/Gemma mlxcel flip status — 2026-07-24 (matvec + adaptive quantum)

**Decision: `not_yet`**

Campaign: `benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-24-full-matvec-adaptive/`

## Locked-gate medians (3 reps)

| Scenario | thr | TTFT | gap | Notes |
| --- | ---: | ---: | ---: | --- |
| S0 | **1.141×** | **0.754× PASS** | **0.866× PASS** | thr short by ~0.8% |
| S1 | 0.323× | 3.15× | **0.265× PASS** | gap OK; thr multi-process isolation |
| S2 | 1.080× | **0.772× PASS** | 2.01× | thr+gap |
| S3 | 0.732× | 15.1× | 1.87× | batch/TTFT/gap |

## S0 thr path

- Pure decode **~111.3 tok/s** with gate/up + down affine-4bit matvec Metal (vs ~107.4 OFF).
- Fresh e2e **~108.1 tok/s** vs mlxcel **~94.8** → thr **1.141 < 1.15**.
- Need pure ~112.2+ or TTFT ~35 ms for locked thr bar with current e2e overhead.
- **Rejected**: multi-row TG + threadgroup `x` cache (~39–42 tok/s regression).
- **Rejected**: heavier production warmup shapes (TTFT/thr regression).

## Code landed

1. Qwen dense FFN gate/up SwiGLU Metal matvec (default-on, preferred over split-FFN compile).
2. Matching down_proj matvec Metal on the same decode path.
3. Wall-time adaptive sibling prefill quantum (feedback from `runner_time_us`, 40 ms budget, start=4).
4. Retained: greedy OpenAI `repetition_penalty=1.0`, stream burst 64, emit batch 1, PACK_LINEAR=0.

## Next

1. Deep-review P0: C++ `mx::compile` elementwise composites (host graph shrink).
2. Deep-review R2: MLX pin/patch audit vs mlxcel.
3. S1 thr under single-process isolation (adaptive quantum keeps gap; thr still multi-process-class).
4. S3 arbiter/batch formation (+ optional tensor-batch drift product decision).
