# Qwen/Gemma mlxcel flip status — 2026-07-24 (night)

**Decision: `not_yet`** — S1 thr still short of 1.15×; S0/S2 pass on last full campaign.

## Physics (exclusive single-process)

Under exclusive arbiter, S1 wall ≈ pure_Gemma_prefill + pure_Qwen_decode (single GPU, software serial).

| Measurement | Value |
| --- | ---: |
| Pure Gemma prefill (13826 tok, chunk 512) | **~7.8–8.3 s** (~1765 tok/s best) |
| Pure Qwen decode (~192 tok @ ~110 tok/s) | **~1.75 s** |
| Theoretical exclusive floor | **~9.6–10.0 s** → thr ceiling **~19.5–20.1** |
| mlxcel multi-process S1 thr (typical) | **~17.6–18.3** |
| Exclusive thr ratio ceiling (no pure cut) | **~1.08–1.12× < 1.15** |

Concurrent dual-hold can raise thr slightly (~1.08×) but gap p95 stays **~350–400 ms** (Metal dual-stream long prefill) — gap gate fails hard.

## Best S1 so far (3-rep exclusive)

| Config | thr ratio | gap abs | gem e2e | qw e2e |
| --- | ---: | ---: | ---: | ---: |
| full-rotating-q64 (chunk 1536) | **1.047×** | 9 ms | 8.9 s | 10.1 s |
| **excl-c512-b2 (chunk 512)** | **1.089×** | **9 ms** | **8.9 s** | **10.1 s** |
| concurrent + fair-soft + q32 | 1.079× | **379 ms FAIL** | 8.7 s | 10.0 s |

Need thr ~20.2 vs mlx ~17.6 → still **~5% pure-sum cut**.

## Prefill-chunk pure sweep (solo Gemma)

| chunk | e2e ms | prefill tok/s |
| ---: | ---: | ---: |
| 256 | 8809 | 1570 |
| 384 | 8670 | 1595 |
| **512** | **7832–8297** | **1666–1765** |
| 768 | 8494 | 1628 |
| 1536 | 8519 | 1623 |
| 2048 | 9002 | 1536 |

**Default flip target now uses `--prefill-chunk 512`.**

## Code landed this session

1. **Sibling engine-step burst** configurable (`AX_SERVER_SIBLING_ENGINE_STEP_BURST`, default 4; flip uses 2). Applies whenever a sibling is active (exclusive **and** concurrent) so concurrent prefill workers cannot flood Metal with 64×quantum steps.
2. **Fair multi-prefill stays active under soft KV pressure** (`kv_low_free_blocks:*`). Previously soft pressure disabled fair and fell back to a 256-token soft budget (S1 concurrent gap ~380 ms). Hard exhausted still disables fair.
3. Adaptive prefill gap SLO **40 ms / max 96** tokens (was 32/64) to use exclusive gap headroom.
4. Flip target: exclusive arbiter, prefill-chunk **512**, sibling burst 2.

## Next

1. Further pure Gemma/Qwen cuts (~5% wall) — compiled attention residual, NA tile, more Metal fuse.
2. Re-run formal ≥3-rep S0–S3 with chunk-512 exclusive stack.
3. S3 thr/gap still open from last full campaign.
4. Commit + push evidence when S1 thr ≥ 1.15 or stack stabilizes as new best.
