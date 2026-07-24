# Qwen/Gemma mlxcel flip status — 2026-07-24 (late)

**Decision: `not_yet`**

## Latest full campaign (`2026-07-24-full-c512-min16-b4`)

Exclusive + prefill-chunk 512 + adaptive min 16 + sibling burst 4 + long-prompt chunk scale.

| Scenario | thr | TTFT | gap | gap abs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| **S0** | **1.153×** | **0.983×** | **0.809×** | 8.8 ms | thr+gap PASS; **TTFT FAIL** |
| **S1** | **1.041×** | **0.869×** | **0.259×** | 8.9 ms | TTFT+gap PASS; **thr FAIL** |
| **S2** | **1.354×** | **0.870×** | **0.777×** | 8.9 ms | **PASS** |
| **S3** | **0.932×** | **0.121×** | **1.597×** | 58 ms | thr+gap FAIL |

Prior good S0 TTFT sample (`full-scale-chunk`): thr **1.164×**, TTFT **0.739×** PASS — TTFT is run-to-run sensitive.

## Best S1 thr (A/B)

| Config | thr ratio | gap |
| --- | ---: | ---: |
| excl-c512-b2 | **1.089×** | 9 ms |
| full campaigns | 1.02–1.05× | 9 ms |

## Physics (exclusive)

- Pure Gemma 13826-tok prefill (chunk 512): **~7.8–8.3 s**
- Pure Qwen decode ~192 tok: **~1.75 s**
- Exclusive wall floor ≈ **9.55 s** → thr ceiling ≈ **20.2** vs mlxcel S1 ~18 → **~1.08–1.12× < 1.15**
- Concurrent dual-hold: thr ~1.08× but gap **~380 ms** FAIL

Need **~0.33 s pure-sum cut** (or concurrent with gap ≤33 ms) for locked S1 thr.

## Code on branch

1. Fair multi-prefill stays active under soft KV pressure.
2. Sibling engine-step burst (env; flip uses 4).
3. Prefill-chunk **512** + `scale_prefill_chunk_for_remaining` long clamp.
4. Adaptive prefill **min 16** (no 1-token pathology under load).

## Residual

1. S1 thr ≥ 1.15: pure Gemma/Qwen kernel/composite cuts beyond exclusive sum.
2. S0 TTFT ≤ 0.90: stabilize warm (already multi-shape); re-run when cool.
3. S3 thr+gap: arbiter/batch or batched-decode product decision.
