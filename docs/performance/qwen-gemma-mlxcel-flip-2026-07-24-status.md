# Qwen/Gemma mlxcel flip status — 2026-07-24 (evening+)

**Decision: `not_yet`** — S0/S2 formal flip; S1/S3 open.

## Scenario ledger

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.169×** | **0.755×** | **0.850×** | **PASS** (5-rep triple warm) |
| **S1** | **~0.72–0.76×** | **~1.33–1.39×** | **~0.80–0.97×** | thr+TTFT FAIL |
| **S2** | **1.783×** | **0.819×** | **0.774×** | **PASS** (3-rep soft-park) |
| **S3** | **~0.82×** | **~7.6×** | **~1.83×** | FAIL all three |

## S1 root cause (measured)

| Mode | Gemma TTFT | thr |
| --- | ---: | ---: |
| Solo pure | ~8.8–9.3 s | n/a |
| Concurrent **after** full pure long prefill | ~9.4 s | **~18.1** (~1.03× mlxcel) |
| Concurrent as **first** long work (formal) | ~14–15 s | **~12.7–13.3** (~0.74×) |

Warm concurrent thr ~18 is still short of 1.15× (need thr ~20.5 if wall≈9.4 s with Qwen finishing first). Dummy-token load warm (even 13.8k) does not remove formal cold-first tax.

## Code landed this pass

1. Fixed-shape Gemma4 dual-path prefill compile (`AX_MLX_MOE_LAYER_COMPILE`).
2. Post multi-model publish long rewarm for Gemma (`AX_SERVER_LONG_PREFILL_WARM`).
3. Target `--prefill-chunk 1536` (best pure A/B on M5).

## Next

1. OpenAI-path warm of exact S1 text under isolation after dual load.
2. Schedule so Qwen finishes ≤ Gemma concurrent (~9.4 s).
3. Gemma pure prefill kernel −10–20% for margin.
4. S3 arbiter/batch.
