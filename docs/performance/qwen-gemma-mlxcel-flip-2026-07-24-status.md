# Qwen/Gemma mlxcel flip status — 2026-07-24 (session end)

**Decision: `not_yet`**

## Best full cool campaign (exclusive stack)

`2026-07-24-full-final-cool`:

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.171×** | **0.750×** | **0.795×** | **PASS** |
| **S1** | **1.053×** | **0.860×** | **0.255×** | thr FAIL |
| **S2** | **1.361×** | **0.772×** | **0.783×** | **PASS** |
| **S3** | **0.936×** | **0.120×** | **1.580×** / 58 ms | thr+gap FAIL |

## S1 thr physics (locked)

Exclusive single-process wall ≈ pure_Gemma + pure_Qwen:

- Pure Gemma 13.8k @ chunk 512: **~7.8–8.4 s** (best knobs; rotating/FFN/Metal already on)
- Pure Qwen ~192 tok: **~1.75 s**
- Pure sum **~9.55 s** → thr ceiling **~20.2** vs mlxcel S1 **~18.2** → **~1.08–1.12× < 1.15**
- Best S1 A/B thr: **1.089×**; cool medians **1.05×**
- Concurrent dual-hold thr ~1.07× but gap **150–380 ms** FAIL
- `gemma4_post_attn_ffn` C++ composite **slower** pure prefill (8971 vs 8409 ms) — leave OFF
- Spec/n-gram **regresses** thr

## Code landed this session

1. Long-prompt prefill chunk scale (512 clamp)
2. Fair multi-prefill under soft KV pressure  
3. Sibling engine-step burst (flip: 4)
4. Hybrid concurrent arbiter: `max_concurrent=2` with **long-prefill exclusive window** when sibling-active fair multi-prefill is on (S1 gap isolation; S3 can dual-hold after fair ends)
5. Wired (opt-in) `gemma4_post_attn_ffn` into `layer_shell_post_attention` for re-A/B

## Residual to flip

1. **S1 thr ≥ 1.15**: need ≥~4% pure Gemma prefill GPU cut (new Metal/composite beyond existing stack) or a concurrent Metal path that keeps gap ≤33 ms.
2. **S3 thr+gap**: dual-hold not enough; need batch formation / emit path cut for absolute gap ≤50 ms and thr ≥1.15.

## Adaptive quantum max 128 / SLO 40 ms (2026-07-24 night)

S1 5-rep hybrid concurrent + exclusive long-prefill window:
- gap p95 stays **~9 ms** (ratio ~0.19) — headroom confirmed
- thr ratio **inflated** when mlxcel cold (14.7 tok/s); AX absolute thr still **~18.2**
- need AX absolute thr **~21** vs stable mlxcel **~18.2** for locked 1.15×

