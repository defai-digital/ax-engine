# Qwen/Gemma mlxcel flip status — 2026-07-24 (matvec v1d)

**Decision: `not_yet`** (S0 **PASS**; S1–S3 still short)

Campaign: `benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-24-full-v1d/`

## Locked-gate medians (3 reps, matvec v1d 256-wide + compiled add_rms)

| Scenario | thr | TTFT | gap | Result |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.166× PASS** | **0.748× PASS** | **0.827× PASS** | **ALL GATES** |
| S1 | 0.329× | 3.09× | **0.259× PASS** | thr/TTFT (multi-process isolation) |
| S2 | 1.093× | 0.901× | 2.26× | thr+gap (lifecycle) |
| S3 | 0.752× | 14.8× | 1.84× | thr/TTFT/gap (batch) |

## S0 thr breakthrough

- Pure decode **~113.4–113.8 tok/s** with **256-thread** gate/up+down matvec (was ~111 with 32-lane).
- E2e **~110.5 tok/s** vs mlxcel **~94.8** → thr **1.166 ≥ 1.15**.
- Tiny TG partials (8 floats) for cross-simdgroup reduce; full x-cache still rejected.

## Also landed

1. `mx::compile` shapeless residual `add + rms_norm` in `activation.cpp` (P0 composite).
2. Wall-time adaptive sibling prefill quantum (µs/tok → tokens for 40 ms SLO).
3. Optional `AX_MLX_BATCHED_DECODE=1` on flip target (S3 thr/TTFT improved modestly: thr 0.75→0.79).

## Residual

| Lever | Status |
| --- | --- |
| S0 thr ≥ 1.15 | **DONE** |
| S1 thr (single-process vs multi-process) | Still ~0.25–0.33×; gap OK |
| S2 thr 1.09→1.15 + gap | Lifecycle HOL / stream stalls |
| S3 thr/TTFT/gap | Need stronger batch + arbiter |

## Next

1. S1: larger effective prefill under gap SLO without HOL-starving interactive TTFT (tick-strict decode/prefill alternate).
2. S2: shrink unload/reload interference on interactive gap.
3. S3: certified row-exact cohort engagement + optional tensor-batch product decision.
