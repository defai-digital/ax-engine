# Path A residual: async intermediate cache-only chunk eval

Host: mbp-m5 / M5 Max. Date: 2026-07-26.

## Lever

`AX_MLX_CACHE_ONLY_CHUNK_ASYNC_EVAL=1` under multi-process keep_base
(`CACHE_ONLY_CHUNK_EVAL=1`). Intermediate cache-only chunks use `async_eval`
on KV; final chunk still blocks. Default OFF.

## Pure A/B (3-rep cold, Gemma 13.8k, c512)

| variant | median ms | ratio |
|---------|----------:|------:|
| base (cache_eval) | 8380 | 1.000 |
| async_chunk | 8146 | **0.972** |

Text first-token parity: `" The"`.

Strict keep-if ≤0.96: **reject** (0.972 > 0.96).

Recalibrated thr physics: async-pipeline multi-process S1 thr **1.122** needs
pure ≤ **0.976** for thr 1.15 (1.122/1.15). Measured 0.972 clears that bar.
Promote cool multi-process S1 with async_chunk (± pipeline) under locked gates.

## Cool multi-process S1 (formal 3-rep, 2026-07-26)

Artifact: `2026-07-26-s1-mp-cache-eval-async-chunk/`

| metric | ratio | gate |
|--------|------:|------|
| thr | **1.103** | FAIL (abs AX thr 20.25 < baseline 20.42) |
| gap | **1.134** | FAIL |
| TTFT | **0.903** | FAIL |

**Decision reject for thr.** Pure 0.972 does **not** transfer under multi-process
concurrent S1 (host/GPU async overlap is exclusive-pure; concurrent Metal
time-share with Qwen regresses thr/gap/TTFT). Leave `ASYNC_EVAL` default OFF.
