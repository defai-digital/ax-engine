# Qwen/Gemma mlxcel flip status — 2026-07-24 (final cool)

**Decision: `not_yet`**

## Campaign `2026-07-24-full-final-cool` (M5 Max, exclusive, chunk 512, min quantum 16, burst 4)

| Scenario | thr | TTFT | gap | gap abs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| **S0** | **1.171×** | **0.750×** | **0.795×** | 8.8 ms | **PASS** |
| **S1** | **1.053×** | **0.860×** | **0.255×** | 8.9 ms | thr FAIL only |
| **S2** | **1.361×** | **0.772×** | **0.783×** | 9.0 ms | **PASS** |
| **S3** | **0.936×** | **0.120×** | **1.580×** | 57.8 ms | thr+gap FAIL |

## Progress vs start of day

| Gate | Before | Now |
| --- | --- | --- |
| S0 thr/TTFT/gap | thr short / mixed | **all PASS** |
| S1 thr | 0.33× historical → 1.047 rotating | **1.053–1.089** (still <1.15) |
| S1 gap | often fail under concurrent | **PASS ~9 ms** exclusive |
| S2 | mixed | **PASS** |
| S3 | fail | still thr+gap fail |

## Physics locking S1 thr

Exclusive single-process wall ≈ pure_Gemma + pure_Qwen:

- Pure Gemma 13.8k prefill @ chunk 512: **~7.8–8.3 s**
- Pure Qwen 192 decode: **~1.75 s**
- Floor wall **~9.55 s** → thr ceiling **~20.2** vs mlxcel S1 **~18.2** → **~1.08–1.12× < 1.15**

Concurrent dual-hold thr ~1.07× but gap **160–380 ms** (Metal queue). Spec/n-gram **hurts** S0/S1 thr.

## Code shipped (branch tip)

- Fair multi-prefill under soft KV pressure
- Sibling engine-step burst (flip: 4)
- Prefill-chunk 512 + long-prompt scale helper
- Adaptive prefill min floor 16

## Next to flip

1. **S1 thr ≥ 1.15**: pure GPU/composite cut ≥~4% on Gemma prefill (or concurrent Metal scheduling that keeps gap ≤33 ms).
2. **S3 thr+gap**: row-exact batch formation / optional server batched-decode product note.
