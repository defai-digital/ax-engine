# Qwen/Gemma mlxcel flip status — 2026-07-24 (night)

**Decision: `not_yet`**

## Full campaign (`2026-07-24-full-excl-c512`, exclusive + prefill-chunk 512)

| Scenario | thr | TTFT | gap | gap abs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| **S0** | **1.151×** | **0.956×** | **0.849×** | 8.8 ms | thr+gap PASS; **TTFT FAIL** |
| **S1** | **1.043×** | **0.866×** | **0.259×** | 8.9 ms | TTFT+gap PASS; **thr FAIL** |
| **S2** | **1.366×** | **0.718×** | **0.777×** | 8.9 ms | **PASS** |
| **S3** | **0.942×** | **0.104×** | **1.566×** | 57 ms | thr+gap FAIL (TTFT PASS) |

Compare prior full-rotating-q64 (chunk 1536): S0 thr/ttft/gap all PASS (thr 1.158, TTFT 0.839); S1 thr 1.047.

## Best S1 A/B (3-rep exclusive)

| Config | thr ratio | gap abs | Notes |
| --- | ---: | ---: | --- |
| full-rotating-q64 (chunk 1536) | **1.047×** | 9 ms | prior full campaign |
| **excl-c512-b2 (chunk 512)** | **1.089×** | **9 ms** | best S1 thr so far |
| excl-c512-q96 (SLO 40/max 96) | 1.019× | 9 ms | cold outliers |
| concurrent + fair-soft + q32 | 1.079× | **379 ms FAIL** | dual-hold gap untamed |

## Physics

Exclusive single-process: wall ≈ pure_Gemma + pure_Qwen.

| Pure measurement | Value |
| --- | ---: |
| Gemma 13826-tok prefill, chunk **512** | **~7.8–8.3 s** (best pure) |
| Gemma same, chunk 1536/2048 | ~8.5–9.0 s (slower) |
| Qwen 192-tok decode | ~1.75 s |
| Exclusive thr ceiling vs mlxcel S1 | **~1.08–1.12× < 1.15** without further pure cut |

Concurrent dual-hold cannot clear gap (Metal long-prefill monopolizes ~350–400 ms p95) even after fair multi-prefill is kept under soft KV pressure and sibling burst is capped.

## Code landed (commit `730b8f98`)

1. Fair multi-prefill **stays active under soft KV pressure**; hard exhausted still disables fair.
2. Sibling engine-step burst env (`AX_SERVER_SIBLING_ENGINE_STEP_BURST`, default 4); applies for exclusive **and** concurrent sibling-active load.
3. Flip target: exclusive, **prefill-chunk 512**, sibling burst 2.
4. Adaptive prefill remains SLO 32 ms / max 64 (40/96 regressed thr).

## Residual to flip

1. **S1 thr ≥ 1.15×** needs ~5% more pure-sum cut (or a concurrent path that keeps gap ≤50 ms and ≤0.90×).
2. **S0 TTFT** with chunk 512 slipped to 0.956× in the full campaign — verify whether chunk 512 hurts short-prompt TTFT vs noise; may need shape-sensitive chunk or keep 1536 for S0.
3. **S3** thr + gap still open (row-exact batch / arbiter).
4. Re-run full ≥3-rep until `flip-decision=flip`.
