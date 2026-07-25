# Concurrent-tax residual (path B)

## Physics
Best multi-process + #672 thr **1.109×** (need ≥1.15). Concurrent wall ≈ Gemma e2e
~9.4s; thr 1.15 needs ≲9.08s (~3.4% concurrent cut). Gap ratio ~1.11 needs ~32 ms
vs ~39 ms. Pure GEMM dual-gate residuals exhausted (hybrid 1.024 reject).

## Residual vs mlxcel
- mlxcel: dual process, `set_wired_limit` → gpu max; 48GB memory caps each.
- AX multi-process: both processes default `AX_MLX_WIRED_LIMIT_SCALE=0.9` of
  recommended working set → dual over-subscribe of unified memory under 2×48GB.
- S1: Qwen interactive 192-tok stream + Gemma long prefill (wall-bound thr).

## Lever (measurement topology only)
Asymmetric wired + Qwen decode posture:
- **Gemma**: `WIRED_LIMIT_SCALE=0.55`, `CACHE_ONLY_CHUNK_EVAL=1` (keep thr stack)
- **Qwen**: `WIRED_LIMIT_SCALE=0.30`, `BATCHED_DECODE=0` (less GPU burst vs Gemma)

Cool 3-rep dual-target S1 on mbp-m5. Success → full S0–S3. Else not_yet.
