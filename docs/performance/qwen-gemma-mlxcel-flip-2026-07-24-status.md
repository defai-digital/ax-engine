# Qwen/Gemma mlxcel flip status — 2026-07-24 (matvec v1d)

**Decision: `not_yet`** — **S0 fully clears locked gates**; S1–S3 remain open.

Campaign: `benchmarks/results/profiling/qwen-gemma-mlxcel-flip/2026-07-24-full-v1d/`

## Locked-gate medians (3 reps)

| Scenario | thr | TTFT | gap | Result |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.166× PASS** | **0.748× PASS** | **0.827× PASS** | **PASS** |
| S1 | ~0.27–0.37× | ~2.8–3.7× | gap OK at q=1 | thr/TTFT fail |
| S2 | ~1.09× | ~0.76–0.90× | ~2.2× | thr+gap |
| S3 | ~0.75–0.79× | ~12–15× | ~1.4–1.8× | thr/TTFT/gap |

## S0 win

- Pure decode **~113.5 tok/s** (256-thread gate/up+down matvec v1d).
- E2e **~110.5 tok/s** vs mlxcel **~94.8** → thr **1.166 ≥ 1.15**.
- Committed: `bd447516`.

## S1 root-cause (measured)

- Qwen interactive itself is healthy under load: TTFT ~51 ms, e2e ~4–6 s for 192 tokens.
- Scenario thr is dominated by **Gemma 8k sibling prefill e2e ~28–37 s** (mlxcel ~10 s).
- Gemma is **preloaded** (~1.3 s load) — delay is interleave/prefill cost, not cold load.
- STREAM_ENGINE_STEP_BURST=64 HOL under multi-model fixed: burst→1 when sibling active.
- Larger fixed prefill quanta (8/32/64) did **not** improve scenario thr; gap can break above ~40 ms.

## Code landed after S0

1. Sibling-active engine burst cap (1).
2. µs/tok wall-time adaptive prefill quantum (start 8, max 64, 40 ms SLO).
3. Medium (512-token) production warmup shape.
4. Optional `AX_MLX_BATCHED_DECODE=1` on flip target.
5. Compiled `add+rms_norm` composite in mlx-sys.

## Next for full flip

1. **S1**: cut Gemma long-prefill wall under multi-model (tick-strict decode/prefill alternate with larger quanta that stay under 50 ms gap; profile per-turn hold).
2. **S2**: lifecycle unload/reload stream-gap spikes.
3. **S3**: stronger batch cohort + arbiter under 4-stream mixed load.
