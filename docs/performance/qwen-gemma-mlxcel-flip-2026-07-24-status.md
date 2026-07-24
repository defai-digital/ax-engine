# Qwen/Gemma mlxcel flip status — 2026-07-24 (late)

**Decision: `not_yet`** — S0 formal flip; S1 thr/TTFT still short of locked 1.15 / 0.90 after exact-S1 warm.

## Scenario ledger

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.157–1.169×** | **0.739–0.755×** | **0.830–0.850×** | **PASS** |
| **S1** | **0.983–0.991×** (exact warm) / **1.05–1.07×** (concurrent arbiter, gap FAIL) | **0.913–0.925×** | **0.25×** exclusive / **~9.8×** concurrent | thr+TTFT short exclusive; concurrent thr↑ gap blown |
| **S2** | **1.783×** | **0.819×** | **0.774×** | **PASS** (triple warm) |
| **S3** | **~0.82×** | **~7.6×** | **~1.83×** | FAIL |

Evidence: `2026-07-24-s1-exact-text-warm`, `2026-07-24-s1-exact-warm-addrms`, `2026-07-24-s0-exact-warm-reg`, `2026-07-24-s1-concurrent-arbiter`, `2026-07-24-s1-concurrent-fair`.

## S1 physics (trial-level)

Formal exact-warm trial anatomy (AX vs mlxcel):

| Side | Qwen e2e | Gemma e2e | thr wall | thr |
| --- | ---: | ---: | ---: | ---: |
| AX exclusive | ~10.5–10.8 s | ~9.4–9.6 s | **Qwen-bound ~10.6 s** | ~18.0 |
| mlxcel multi-process | ~5.6 s | ~10.3–10.5 s | **Gemma-bound ~10.4 s** | ~18.2 |

- thr = sum(output tokens=193) / max(e2e). Gate 1.15× needs wall ≤ ~9.2 s.
- Exclusive AX serializes device turns → wall ≈ pure-sum (~10.3–10.6 s) → thr ceiling ~18 tok/s ≈ 0.99× mlxcel.
- mlxcel multi-process finishes Qwen in ~5.6 s under concurrent Metal time-slicing; wall is pure-Gemma-ish.

## Experiments this session

### Exact S1 text warm (shipped)

`run_exact_s1_gemma_long_prefill_warmup` after multi-model publish closes cold-first tax (thr 0.74 → ~0.99). Gap excellent (0.25×).

### Concurrent execution arbiter (opt-in)

`AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT=2` lets distinct model workers hold turns together (each already has a dedicated `MlxStream::new_gpu` on its owner thread).

| Config | thr ratio | TTFT ratio | gap ratio | Notes |
| --- | ---: | ---: | ---: | --- |
| Concurrent, fair-prefill off | 1.054× | 0.915× | **9.8×** | gap p95 ~340 ms |
| Concurrent + fair quantum | 1.067× | 0.913× | **9.8×** | gap p50 ~9 ms, p95 ~340 ms (spike tail) |
| Exclusive + exact warm | 0.99× | 0.92× | **0.25×** | gap PASS |

**Conclusion:** concurrent holds do **not** reproduce mlxcel multi-process Qwen e2e (~5.6 s); AX Qwen stays ~10 s. Unified-memory / MLX contention plus long Gemma kernels create p95 gap spikes. Flip target stays exclusive (`max_concurrent=1` default). Concurrent arbiter remains opt-in for further research.

### Compiled elementwise activations (shipped)

`silu_mul` and `gelu_approx_mul` use mlxcel-style `mx::compile(shapeless=true)` with fail-closed fallback (matmul stays outside). Requires `#include "mlx/compile.h"`.

## Code landed

1. Exact S1 Gemma text warm after multi-model publish.
2. Concurrent multi-model execution arbiter (`AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT`, default 1).
3. Compiled shapeless `silu_mul` + `gelu_approx_mul` composites.
4. Adaptive prefill quantum (default 64, gap SLO 32 ms) under exclusive isolation.

## Next for S1 thr ≥1.15× and TTFT ≤0.90×

1. **Pure Gemma prefill −12–15%** (Metal attention/FFN, MLX pin audit R2, more elementwise composites, host graph reduction). Exclusive physics: wall must drop from ~10.5 s → ≤9.2 s without multi-process free lunch.
2. **Deeper multi-stream overlap** if concurrent is revisited: need Qwen e2e → ~5–6 s under load without gap p95 spikes (kernel-level fairness, not just arbiter dual-hold).
3. Confirm S0/S2 still green after pure-path changes.
4. S3 arbiter/batch after S1.

## Report alignment

Deep review (`.internal/reports/mlxcel-deep-review-2026-07-24.zh-TW.md`):

- P0 composites: partial (add-style residual path history + silu/gelu compile).
- P2 wall-time quantum: landed adaptive 64-token / 32 ms SLO (exclusive).
- S1 architecture note confirmed: multi-process isolation is half the mlxcel S1 story; AX single-process must solve in-process device turns.
- Do **not** abandon single-process multi-model product shape for the flip.
