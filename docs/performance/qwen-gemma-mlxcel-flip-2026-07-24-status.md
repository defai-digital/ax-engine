# Qwen/Gemma mlxcel flip status — 2026-07-24 (matvec path)

**Decision: `not_yet`**

Campaign: `benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-24-full-matvec/`

## Locked-gate medians (3 reps, matvec-first thr path)

| Scenario | thr | TTFT | gap | Notes |
| --- | ---: | ---: | ---: | --- |
| S0 | **1.138×** | **0.729× PASS** | **0.877× PASS** | thr short by ~1% |
| S1 | 0.349× | 2.91× | **0.273× PASS** | quantum=1 restores gap |
| S2 | 1.075× | **0.755× PASS** | 1.93× | thr+gap |
| S3 | 0.676× | 16.9× | 1.97× | batch/TTFT |

## S0 thr path

- Pure decode **~110.5–111.4 tok/s** after preferring Qwen gate/up SwiGLU **matvec Metal** over split-FFN compile (was ~107).
- Fresh e2e **~108.3–108.5 tok/s** vs mlxcel **~94.8–95.2** → thr ratio **~1.14 < 1.15**.
- Need ~+0.7% e2e thr or pure ~112+ to clear thr gate.

## Code landed

- Matvec Metal default-on; compile runs only if matvec misses
- Greedy OpenAI penalty 1.0; stream burst 64; thr-critical target env
- Sibling prefill quantum env-calibratable (S1 gap needs ≤~4 tokens fixed)

## Next

1. Fuse down_proj into matvec kernel or other BW fusion for last S0 thr %
2. Wall-time adaptive S1 quantum (fixed tokens unsafe mid-prefill)
3. S3 arbiter/batch (+ optional tensor-batch drift product decision)
