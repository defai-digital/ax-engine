# Qwen/Gemma mlxcel flip status — 2026-07-24 (continued)

**Decision: `not_yet`**

## Locked gates (never relaxed)

- thr ≥ 1.15×
- p95 TTFT ≤ 0.90×
- stream-gap ≤ 0.90× and absolute ≤ 50 ms
- zero errors

## Review → implementation map

From `.internal/reports/mlxcel-deep-review-2026-07-24.zh-TW.md`:

| Review lever | Status on flip path |
| --- | --- |
| P0 compiled elementwise composites | Landed (gelu/silu, add+rms, dense FFN prefill compile, GEGLU Metal) |
| P1 host-graph shrink | Same as P0; reorder rejected (RESOLVED 2026-07-17) |
| P2 wall-time prefill quantum | Landed (adaptive 64 start / max 128 / SLO 40 ms) |
| Dual-hold concurrent (max=2) | **Rejected for S1 thr**: gap p95 160–220 ms, thr regresses under Metal contention |
| Long-prefill exclusive window | **Default ON** (kill-switch `AX_SERVER_LONG_PREFILL_EXCLUSIVE=0`) |
| NA tile pad | Low leverage for 512-aligned long chunks (only last partial) |
| gemma4_post_attn_ffn composite | Leave OFF (pure prefill slower) |
| Gemma4 split prefill FFN | Default ON; kill-switch `AX_MLX_GEMMA4_SPLIT_PREFILL_FFN=0` for packed+compile A/B |

## Physics (exclusive stack)

Measured cool exclusive (best):

| Side | thr | gap p95 | Gemma wall | Qwen wall |
| --- | ---: | ---: | ---: | ---: |
| AX | **~19.2 tok/s** | **~9 ms** | **~8.9 s** | **~10.1 s** |
| mlxcel multi-proc | **~18.3 tok/s** | **~35 ms** | **~10.3 s** | **~5.6 s** |

- AX pure Gemma is already **~14% faster** than mlxcel pure Gemma.
- thr gate needs AX absolute thr **≥ ~21.0** (wall ≤ ~9.2 s).
- Exclusive interleaving already beats pure-sum; dual-hold does **not** hide Qwen under Gemma without stretching both (gap fails, thr ~flat or worse).
- Residual: **~8–10% pure Gemma prefill GPU cut** (or an unknown Metal dual-stream schedule that keeps gap ≤50 ms).

## Dual-hold experiment (2026-07-24 night)

`2026-07-24-s1-dualhold-slo28` (exclusive window off, SLO 28 ms, burst 1):

- r1 thr 19.4 gap **167 ms** wall 9.9 s
- r2 thr 17.5 gap 185 ms
- r3 thr 15.5 gap 217 ms (thermal / contention)
- **Verdict: not a thr path; isolation restored default-on**

## Best prior full cool campaign

`2026-07-24-full-final-cool` (exclusive stack):

| Scenario | thr | TTFT | gap | Status |
| --- | ---: | ---: | ---: | --- |
| **S0** | **1.171×** | **0.750×** | **0.795×** | **PASS** |
| **S1** | **1.053×** | **0.860×** | **0.255×** | thr FAIL |
| **S2** | **1.361×** | **0.772×** | **0.783×** | **PASS** |
| **S3** | **0.936×** | **0.120×** | **1.580×** / 58 ms | thr+gap FAIL |

## Code in this session

1. Env-gated long-prefill exclusive window (`AX_SERVER_LONG_PREFILL_EXCLUSIVE`, default on)
2. Restore adaptive quantum 64/128/40 ms + sibling burst 4 for exclusive thr envelope
3. Flip target exclusive (`MAX_CONCURRENT=1`) after dual-hold rejection
4. Gemma4 split-prefill FFN kill-switch for packed+compile pure A/B

## Next pure levers (ordered)

1. Cool exclusive ≥5-rep with long cooldown; require absolute thr ≥21 not inflated mlxcel cold
2. Packed FFN A/B (`AX_MLX_GEMMA4_SPLIT_PREFILL_FFN=0`) pure Gemma wall
3. Prefill stage profile (`AX_MLX_PREFILL_PROFILE`) → Metal/composite for top stages
4. S3: emit/batch path after S1 pure cut lands
5. Full ≥3-rep S0–S3 only when S1 absolute thr headroom exists

## Commit policy

Gates never relaxed. Push flip campaign artifacts only when `flip-decision=flip`. Intermediate scheduler/policy code may land with `not_yet` status.
