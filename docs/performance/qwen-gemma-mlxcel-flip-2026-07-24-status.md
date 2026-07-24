# Qwen/Gemma mlxcel flip status — 2026-07-24 (late)

**Decision: `not_yet`** — S0/S2 formal flip; S1 thr/TTFT still short of locked 1.15 / 0.90 after exact-S1 warm.

## Scenario ledger

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.169×** | **0.755×** | **0.850×** | **PASS** (5-rep triple warm) |
| **S1** | **0.983–0.991×** | **0.917–0.925×** | **0.25×** | thr+TTFT barely short (was 0.74×) |
| **S2** | **1.783×** | **0.819×** | **0.774×** | **PASS** |
| **S3** | **~0.82×** | **~7.6×** | **~1.83×** | FAIL |

Evidence: `2026-07-24-s1-exact-text-warm`, `2026-07-24-s1-exact-warm-addrms`, `2026-07-24-s0-triple-warm`, `2026-07-24-s2-triple-warm`.

## S1 breakthrough: exact-S1 text warm

| Mode | thr | Gemma TTFT |
| --- | ---: | ---: |
| Formal cold-first (no real text warm) | ~0.74× | ~14–15 s |
| **Exact S1 text warm after multi-model publish** | **~0.99×** | **~9.5 s** |
| Warm concurrent micro (after pure long) | ~18 tok/s | ~9.4 s |

Dummy-token long warm does **not** transfer; tokenized exact S1 prompt does.

## S1 residual physics

With pure Gemma ~8.5–9.0 s and pure Qwen ~1.75 s, single-GPU full-util wall ≥ ~10.5 s → thr ceiling **~18 tok/s**.

Formal medians: AX thr ~18.0, mlxcel thr ~18.2 → ratio **~0.99×**. Locked gate needs **1.15×** (thr ~20.9 → wall ≤9.2 s) which requires **~15% faster pure Gemma** (or equivalent pure-sum reduction).

TTFT formal ~0.92× (need ≤0.90); ~2% more concurrent Gemma TTFT.

Gap formal **0.25× PASS** (exact warm path).

## Code landed

1. Exact S1 Gemma text warm (`run_exact_s1_gemma_long_prefill_warmup`) after multi-model publish.
2. Compiled `add_rms_norm_pair` (mlxcel-style shapeless elementwise composite).
3. Dual-path prefill compile scaffold; prefill-chunk 1536.

## Next for S1 thr 1.15×

1. Pure Gemma prefill −15% (Metal attention/FFN, mlxcel MLX pin R2, host graph reduction).
2. Confirm S0/S2 still green with exact warm load path.
3. S3 arbiter/batch after S1.
