# Lever: smaller Gemma prefill chunk under multi-process (contention residual)

## Residual

Best S1 thr **1.109×** (multi-process + cache_eval, prefill-chunk **512**).
Chunk **1024** already regressed thr to 1.077×. Smaller chunks untested under
multi-process.

Physics: multi-process thr wall ≈ Gemma e2e; concurrent tax vs pure ~14%.
Smaller Gemma chunks (`prefill-chunk` 256/384) force more eval barriers and
may improve Metal interleaving with concurrent Qwen decode (gap + thr).
mlxcel server default remains 512; NA pad aligns to 32 so 256/384 are tile-valid
(`align_to_na_tile`).

## mlxcel

`server/batch/scheduler.rs` chunked prefill + optional NA pad; server
`prefill_chunk_size=512`. Deep review §S1: multi-process time-share is the thr
topology residual.

## Plan

1. Pure A/B under cache_eval: c512 vs c256 vs c384 (measure pure tax).
2. If pure ratio ≲1.05, cool multi-process S1 with Gemma chunk 256 or 384.
3. Full S0–S3 only if thr≥1.15 and gap/TTFT pass.

## Result (mbp-m5 pure cache_eval, 2026-07-25, 3-rep)

| chunk | median cold ms | ratio vs 512 |
|------:|---------------:|-------------:|
| 512 | 12090 | 1.000 |
| 384 | 12553 | **1.038** |
| 256 | 13171 | **1.089** |

Decision **keep_512**. Pure tax exceeds concurrent-tax budget; no multi-process S1.
