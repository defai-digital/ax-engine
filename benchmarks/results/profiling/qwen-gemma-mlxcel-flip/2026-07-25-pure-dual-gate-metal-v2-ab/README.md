# Pure-wall A/B: dual gate/up Metal v2 (X-tile BM=4) — mbp-m5

## Result

| | cold mean |
|--|--|
| OFF (MLX dual qmm) | 8631 ms |
| ON (Metal v2) | **215495 ms** |

**ratio ≈ 25×** → default remains OFF. ON also returned empty completion text.

v1 (per-row global X) was ~8.5×; v2 (shared X tile) is worse still vs MLX specialized multi-token qmm. Hand-rolled dual-gate Metal is not a pure-prefill win path on M5 Max for the flip package (bits=8 gs64).
