# Lever: full mlxcel #672 pair on cache-only pure prefill

## Residual

mlxcel `chunked_prefill_last_logits` does **both** between chunks:
1. `ffi::eval(&piece_logits)`
2. `ffi::clear_memory_cache()`

AX A/Bs measured them separately:
- `AX_MLX_CACHE_ONLY_CHUNK_EVAL` alone: pure median **0.968** (reject keep OFF)
- `AX_MLX_PREFILL_CLEAR_CACHE_PER_CHUNK` alone: prior reject default-on

Hypothesis: freelist reclaim without eval (or eval without freelist) misses the
#672 interaction. Re-measure **both ON** vs both OFF on pure Gemma 13.8k.

## Success

Cold median ratio ≤ 0.925 → cool S1. Else reject; leave both default OFF.

## Result (mbp-m5, 2026-07-25) — **REJECT**

| variant | cold median ms |
|---------|---------------:|
| OFF | 9109 |
| ON (eval+clear) | 8735 |

- ratio_median = **0.959** (~4.1% cut; need ≤0.925)
- decision: **reject_keep_off**
- Slightly better than eval-only (0.968) but still no thr headroom.
