# Dual-stream residual: `yield:N` pipeline eval (mbp-m5, 2026-07-26)

**Decision: `not_yet` / reject_yield.** Gates unchanged. No S0–S3.

## Residual (fail-closed, default OFF)

`AX_MLX_PIPELINE_EVAL_GRANULARITY=yield:N` (`N >= 1` milliseconds):

- At multi-token prefill layer boundaries, fire a **blocking** eval only when
  ≥ `N` ms of wall time have elapsed since the previous fire.
- Caps GPU monopolization for multi-process concurrent fairness without forcing
  a barrier on every layer (layer-eval thr wash) or every 8 layers (block:8 gap
  collapse).
- Pure helpers unit-tested (`parse_pipeline_eval_granularity`,
  `pipeline_eval_yield_should_fire`). Malformed values → Off.

## Stack under test

thr-b8 concurrent base without block:8:

- Gemma: `CACHE_ONLY_CHUNK_EVAL` + `ASYNC_EVAL` + `PIPELINE_GRANULARITY=layer`
  + `PIPELINE_EVAL_GRANULARITY=yield:N`
- Qwen: optional `process_qos_clamp=utility` (same as thr-b8 formal)

## Smoke dual-target S1 (1-rep, non-theater mlxcel thr ~18.3)

| config | thr | gap | ax thr | ax gap | note |
|--------|----:|----:|-------:|-------:|------|
| thr-b8-util (same binary) | **1.141** | 1.114 | 20.91 | 41.5 | rebin baseline |
| yield:8 + util | 1.092 | 1.118 | 20.23 | 38.8 | thr regress |
| yield:12 + util | 1.065 | 1.228 | 19.51 | 42.5 | thr+gap regress |
| yield:16 + util | 1.102 | 1.285 | 20.14 | 44.0 | thr+gap regress |
| yield:24 + util | 1.068 | 1.330 | 19.56 | 46.7 | thr+gap regress |
| yield:16 (no util) | 1.082 | 1.214 | 19.85 | 41.1 | thr regress |

## Conclusion

Wall-clock yield does **not** land between layer-eval (thr 1.11 / gap 1.00) and
block:8 (thr 1.14 / gap 1.22) in a way that clears thr≥1.15 or gap≤0.90. Best
yield smoke thr is 1.102; best gap among yield smokes is 1.118 (still ≫0.90 and
worse thr than thr-b8).

**Keep default OFF.** No cool ≥3-rep formal (smoke headroom insufficient).
Gates file unchanged. Flip remains **not_yet**.

Open physics unchanged: need pure GPU (steel dual-gate ≤0.96) and/or concurrent
gap below mlxcel parity without thr wash.
