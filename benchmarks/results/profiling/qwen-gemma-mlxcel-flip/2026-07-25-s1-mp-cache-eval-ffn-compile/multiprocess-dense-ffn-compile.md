# Lever: multi-process Qwen decode compile (contention residual)

## Residual

Best S1: multi-process AX + Gemma cache_eval thr **1.109×**, gap ratio **1.113**
(39 ms). Pure pipeline_granularity / compose under cache_eval keep_base reject.

Physics under multi-process Metal time-share:

- thr wall ≈ Gemma e2e (~9.3–9.5 s); need ≲9.08 s for thr ≥1.15×
- concurrent tax vs pure Gemma (~8.2 s) ≈ **14%**
- Qwen interactive gap 39 ms vs mlxcel 35 ms (need ≤~31.8 for ratio 0.90)

Faster Qwen decode under concurrent load frees GPU for Gemma (thr) and
shrinks stream gap (gap gate). Dual residual.

## mlxcel source

Deep review P0: compiled composites collapse per-step host/graph cost for
decode FFN (`compiled_swiglu_*` / `compiled_gelu_*` in
`mlx_cxx_bridge.cpp`). AX already has `AX_MLX_DENSE_FFN_COMPILE` (default
**ON**) for packed dense FFN decode.

Multi-process flip target currently **forces** `AX_MLX_DENSE_FFN_COMPILE=0`
while leaving `AX_MLX_DENSE_FFN_COMPILE_PREFILL=1`. That disables the decode
compile path for Qwen under concurrent S1 — residual mismatch vs mlxcel
compiled decode.

## AX change

Target `ax-qwen-gemma-m5max-multiprocess-cache-eval-ffn-compile.json`: same as
cache-eval multi-process but `AX_MLX_DENSE_FFN_COMPILE=1`. Cool 3-rep dual-target
S1 only (not full S0–S3 unless thr≥1.15 and gap ratio looks passable).

## Success

S1 thr ≥1.15 **and** gap ratio ≤0.90 **and** TTFT ≤0.90 → full S0–S3 flip.
Else not_yet; keep prior best stack env.

## Result (mbp-m5 cool 3-rep S1)

| metric | AX | mlxcel | ratio |
|--------|---:|-------:|------:|
| thr tok/s | 19.44 | 17.63 | **1.103** |
| gap p95 ms | 40.1 | 35.8 | **1.119** |
| TTFT p95 | 9181 | 10160 | **0.904** |

vs best stack thr **1.109** (cache_eval, DENSE_FFN_COMPILE=0): **regression**.
Decision: **reject** — keep multi-process target with `DENSE_FFN_COMPILE=0`.
