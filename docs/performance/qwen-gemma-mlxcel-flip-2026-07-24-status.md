# Qwen/Gemma mlxcel flip status — 2026-07-24 (evening)

**Decision: `not_yet`** — S0 formal flip; S1 TTFT+gap green after exact-text warm; thr still short of locked 1.15× under exclusive single-GPU physics.

## Scenario ledger (best locked-gate evidence)

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.157–1.169×** | **0.739–0.755×** | **0.830–0.850×** | **PASS** |
| **S1** | **1.010×** | **0.900×** | **0.234×** | thr FAIL only (TTFT/gap PASS) |
| **S2** | **1.783×** | **0.819×** | **0.774×** | **PASS** (prior triple warm) |
| **S3** | **~0.82×** | **~7.6×** | **~1.83×** | FAIL |

Best S1: `2026-07-24-s1-exact-warm-v2` (exclusive arbiter, exact S1 text warm, compiled silu/gelu/add_rms).

## S1 trial anatomy (exact-warm-v2)

| Side | Qwen e2e | Gemma e2e | Wall | thr |
| --- | ---: | ---: | ---: | ---: |
| AX exclusive | ~10.7 s | ~9.5 s | Qwen-bound | ~18.0 |
| mlxcel multi-process | ~5.6 s | ~10.3 s | Gemma-bound | ~17.8 |

Gate 1.15× needs AX wall ≤ ~9.3 s (thr ≥ ~20.4). Exclusive wall ≈ pure Gemma + pure Qwen (~10.3–10.7 s). **~14–15% pure-sum reduction required.**

## Experiments (this session)

| Config | thr | TTFT | gap | Notes |
| --- | ---: | ---: | ---: | --- |
| Exclusive + exact warm (v2) | **1.010×** | **0.900×** | **0.23×** | Best locked-gate envelope |
| Exclusive + q96/SLO45 | 0.959× | 0.952× | 0.25× | No thr win; TTFT slips |
| Concurrent dual-hold (cold) | ~1.05–1.07× | ~0.91× | **~9.8×** | Gap p95 ~340 ms |
| Concurrent + exact warm + q48 | **1.033×** | 0.929× | **~10×** | Gap still spikes; Qwen e2e still ~10 s (not 5.6 s) |
| Dummy warm only | 0.72× | 1.39× | ~1.0× | Cold-first tax |

**Concurrent dual-hold does not reproduce mlxcel multi-process Qwen e2e (~5.6 s).** Flip target stays exclusive.

## Code landed (branch `codex/mlxcel-s1-concurrent-fair-composites`)

1. Exact S1 Gemma text warm after multi-model publish (`run_exact_s1_gemma_long_prefill_warmup`).
2. Compiled shapeless `silu_mul`, `gelu_approx_mul`, `add_rms_norm_pair`.
3. Standard-family post-attn residual fused via `add_rms_norm_pair`.
4. Concurrent arbiter opt-in (`AX_SERVER_EXEC_ARBITER_MAX_CONCURRENT`); fair prefill retained when sibling active.
5. Adaptive sibling prefill quantum default 64 / gap SLO 32 ms (exclusive flip baseline).

## Pure-path experiments (evening)

| Attempt | Result |
| --- | --- |
| Rotating SWA KV during multi-token prefill | **Panic** SDPA mask/KV shape mismatch (capacity ring vs ordered windowed view / hoisted masks). Wiring kept, **default OFF**. Defensive `key_len == capacity` guards landed. |
| Concurrent dual-hold + exact warm | thr ~1.03, gap ~10× FAIL |
| Exclusive exact-warm tip recheck | thr **1.001**, TTFT 0.908, gap 0.26 |

## Next for S1 thr ≥1.15×

1. Finish rotating-prefill correctly: mask hoist must use **per-layer post-append key_len**, cold layers must enter ring on first write past window, then re-A/B pure Gemma and formal S1.
2. Or other pure-sum cuts (Metal SDPA/FFN, more elementwise composites, host-graph) totaling **~14–15%** exclusive wall.
3. True multi-stream overlap remains secondary (concurrent dual-hold does not cut Qwen e2e to ~5.6 s).
4. Then full ≥3-rep S0–S3 campaign.

## Physics note

Locked thr 1.15× against mlxcel multi-process S1 is a pure-path problem under AX exclusive single-GPU serialization. Policy/quantum tuning alone cannot clear the bar once TTFT/gap are already green at thr ~1.0×.
