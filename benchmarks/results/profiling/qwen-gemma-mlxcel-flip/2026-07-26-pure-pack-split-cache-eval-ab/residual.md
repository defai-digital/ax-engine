# Pure pack vs split under thr-b8 cache_eval stack (mbp-m5)

**Decision: keep pack (default ON).** Split pure **1.055×** worse under thr-b8-like env.

Gates unchanged. Concurrent S1 smokes neither clear thr≥1.15 / gap≤0.90.

## Pure A/B (3 cold reps, thr-b8-like env)

Env: `CACHE_ONLY_CHUNK_EVAL` + `ASYNC_EVAL` + pipe `layer` + eval `block:8`.

| variant | median ms | ratio vs pack |
|---------|----------:|--------------:|
| pack ON (default) | **7819** | 1.000 |
| split (`PACK_DENSE_FFN_GATE_UP=0`) | 8246 | **1.055** |

Text parity OK (`" The"`). Keep bar ≤0.96 for thr unlock: **not met** (split regresses).

Note: earlier 2026-07-25 pack A/B without thr-b8 pipeline/async stack preferred
split; under the thr-b8 concurrent pure env, **pack (steel dual-output single
qmm) wins**. Default ON is correct for this stack.

## Concurrent dual-target S1 smokes (1-rep)

| config | thr | gap | ax thr | note |
|--------|----:|----:|-------:|------|
| thr-b8-util + pack | 1.101 | 1.345 | 20.22 | thermal/noise vs formal 1.141 |
| thr-b8-util + split | 1.127 | 1.301 | 20.64 | thr still &lt;1.15 |
| layer-eval + split | 1.050 | 1.078 | 19.20 | thr wash |

## Conclusion

Packed steel dual-gate is the better pure path under thr-b8 env; concurrent thr
still short of 1.15 and gap still ≫0.90. No cool formal from this residual.
Flip remains **not_yet**.

## Cool concurrent thr-b8 priority smokes (post thermal cool)

| config | thr | gap | ax thr | note |
|--------|----:|----:|-------:|------|
| thr-b8-util cool | **1.142** | 1.223 | 20.95 | matches formal thr 1.141 |
| thr-b8 plain (no util) | 1.064 | 1.275 | 19.44 | thr regress |
| thr-b8 + Qwen thruput_tier 0 | 1.082 | 1.262 | 19.87 | thr regress |

Elevating Qwen (t0) or dropping utility does **not** unlock thr or gap.
Best concurrent thr remains thr-b8+qwen-util ~1.141 formal.
