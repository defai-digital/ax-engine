# Residual inventory — AX vs mlxcel flip (mbp-m5, 2026-07-25 tip)

## Locked gates (unchanged)
thr ≥1.15×, TTFT ≤0.90×, gap ≤0.90× and ≤50 ms, zero errors.

## Current exclusive S1 (tip)
| metric | AX | mlxcel | ratio | gate |
|--------|---:|-------:|------:|------|
| thr tok/s | 18.62 | 17.97 | **1.036** | FAIL (≥1.15) |
| gap p95 ms | 8.9 | 34.9 | 0.256 | PASS |
| TTFT p95 ms | 8239 | 9959 | 0.827 | PASS |

Need thr ≳20.7 (scenario wall ≲9.3s from ~10.4s, ~11% cut).

## Pure Gemma 13.8k rejects this session
| lever | ratio_median | note |
|-------|-------------:|------|
| cache-only chunk eval | 0.968 | #672 half |
| #672 eval+clear | 0.959 | full pair |
| long chunk 768 | 0.987 | keep 512 |
| long chunk 1024 | 0.981 | keep 512 |
| qmm+rms 5-rep | 0.965 | thermal noise |
| dual Metal v1/v2 | 8–25× | worse |
| dual_qmm_geglu | 1.091 | worse |
| native offset causal | 1.064 | worse |
| … prior host fuses | ~1.00–1.03 | noise/reject |

## Physics
- AX pure already ≳ mlxcel pure (~14% faster historically).
- mlxcel multi-token bits=8 MLP uses **op-at-a-time** qmm (same as AX); compile is decode/bits=4 only.
- mlxcel S1 thr advantage is **multi-process dual-stream**, not a pure FFN residual we missed.
- Dual-hold max=2 (including quantum=4) fails gap ~160–220 ms.

## Open paths (residual-backed only)
1. Pure GPU ≥11% beyond current best (needs GEMM-class dual-gate or MLX-level win).
2. Dual-stream with gap ≤50 ms (no successful measurement yet on M5 Max).
3. Do **not** relax gates; do **not** claim flip without S0–S3 decision=flip.

## Schedule A/Bs this session (S1)
| config | thr ratio | gap p95 | note |
|--------|----------:|--------:|------|
| exclusive tip (default) | 1.036 | 8.9 ms | only thr fails |
| dualhold-q4 | 1.054 | 166 ms | gap FAIL |
| thr-quanta-128 exclusive | 1.045 | 8.9 ms | thr still FAIL |

## Stacked small wins pure (2026-07-25)
#672 eval+clear + qmm_rms + chunk1024 vs portable: median ratio **1.052×** (worse).
Decision reject_stack — small wins do not compose under thermal/host interaction.

## Multi-process AX topology probe (not product flip target)
Two single-model `ax-engine-server` processes (48GB each), concurrent Qwen stream + Gemma 13.8k prefill:

| metric | median |
|--------|-------:|
| thr tok/s | **19.42** |
| gap p95 ms | **48.2** (≤50) |

vs exclusive tip thr 18.62 and mlxcel formal thr 17.97 → probe thr ratio vs mlxcel ~**1.08×** (still <1.15).
Two-process Metal time-share gap is near SLO; thr still short of gate. Product flip remains single-process.

## Metal4 / NAX attention residual
mlxcel `metal4_attention` → C++ `fused_metal4_attention` with `(void)use_metal4` and
delegates to upstream `mlx::core::fast::scaled_dot_product_attention` (same class as AX
`scaled_dot_product_attention_with_mask`). **Not a distinct GPU kernel residual** on M5
with MLX 0.32; AX pure already ≳ mlxcel pure. Leave Metal4 bridge port deferred.

## Bottom line (2026-07-25)
- Exclusive S1: thr **1.03–1.05×** FAIL, gap/TTFT PASS.
- Dual-hold single-process: gap **~166 ms** FAIL.
- Multi-process AX probe: thr **~1.08×**, gap **~48 ms** — still <1.15×.
- Pure host/graph residuals max ~4%; stacks regress.
- Flip under locked gates remains **not_yet** without a true ≥11% pure GPU cut
  (GEMM-class dual-gate) or a dual-stream product shape that still clears thr 1.15×.

## Dual-gate Metal v3 tiled GEMM (2026-07-25)
BM=8 / BN=16 / BK=128 full-TG coop loads: pure median **8.52×** slower, empty text.
Reject. Host-side + naive custom GEMM still cannot beat MLX qmm for gate_up.

## Formal multi-process AX S1 (2026-07-25)
Harness AX multi-process target vs mlxcel: thr **1.062×**, gap ratio **1.32×** (46ms abs), TTFT **0.940×**.
not_yet. Topology residual insufficient for locked thr≥1.15.

## Multi-process AX S1 ladder (formal cool 3-rep)
| config | thr ratio | gap ratio | TTFT ratio | notes |
|--------|----------:|----------:|-----------:|-------|
| multi-process baseline | 1.062 | 1.32 | 0.940 | thr lift vs exclusive |
| + Gemma #672 cache-eval | **1.109** | 1.113 | **0.899 PASS** | best thr so far |
| + chunk 1024 | 1.077 | 1.369 | 0.925 | worse than 512 |

Still short of thr≥1.15 and gap ratio ≤0.90. Absolute gap under multi-process+cache-eval is ~39 ms (≤50 abs ok).

| + Gemma full #672 (eval+clear) | 1.105 | 1.096 | 0.903 | worse than eval-only |

**Best residual stack so far:** multi-process AX + Gemma cache-only chunk eval → thr **1.109×**, TTFT PASS, gap ratio fail.
Still need ~3.7% thr and ~10% gap-ratio improvement for flip. Gates not relaxed.

## Pure compose under cache_eval (2026-07-25)
Baseline = multi-process Gemma env (`CACHE_ONLY_CHUNK_EVAL=1`). Candidates norot /
qmmrms / both: medians **1.18× / 1.026× / 1.18×** vs base **8219 ms**. Decision
**keep_base**.

## Pipeline granularity (mlxcel M5 residual, 2026-07-25)
`AX_MLX_PIPELINE_GRANULARITY` = `block:4` / `block:2` / `layer` vs off under
cache_eval pure: ratios **1.070 / 1.040 / 1.049**. Decision **reject_keep_off**.
Default remains off. No cool S1 remeasure; thr headroom still short of 1.15×.

## Multi-process + cache_eval + DENSE_FFN_COMPILE=1 (2026-07-25)
Cool S1 thr **1.103×**, gap ratio **1.119**, TTFT **0.904**.
Worse thr than cache_eval-only **1.109×**. Reject — keep DENSE_FFN_COMPILE=0 on multi-process target.

## M5 AUTO_BUFFER_CAPS kill (2026-07-25)
`AX_MLX_AUTO_BUFFER_CAPS=0` (mlxcel M5 leave-default) under cache_eval pure: median ratio **0.989**.
Insufficient for thr 1.15 physics (need ≤0.96). Keep auto-raise ON.

## Smaller prefill chunk under cache_eval pure (2026-07-25)
c384 / c256 vs c512: ratios **1.038 / 1.089**. Keep **512**; no multi-process S1.

## Compiled GeGLU activation (mlxcel parity, 2026-07-25)
`AX_MLX_COMPILED_GEGLU_ACTIVATION=1` under cache_eval pure: median **1.018×** vs Metal GEGLU base.
Imperative nometal **1.060×**. Keep Metal; compiled default OFF. Flip still not_yet.

## Chunk-stable dual_gate / #705 under cache_eval (2026-07-25)
Hypothesis: fixed chunk-512 shapes improve shape-specific compile reuse.
Results: dual_gate **1.003×**, shaped **0.996×**, both **0.993×** vs base.
Best ~0.7% pure — need ≤0.96. Keep both OFF. Flip still not_yet.

## Async dual gate/up submit (2026-07-25)
`AX_MLX_ASYNC_DUAL_GATE_UP=1` under cache_eval pure: median **1.007×**. Reject default OFF.

## Dual affine qmm one-FFI (2026-07-25)
`AX_MLX_DUAL_AFFINE_QMM=1` (dual qmm, Metal GEGLU kept) under cache_eval pure: median **1.002×**.
Reject default OFF. Gate_up thr residual still needs GEMM-class win, not host-FFI collapse.

## Dual-stream gate/up qmm (2026-07-25)
`AX_MLX_DUAL_STREAM_GATE_UP=1` under cache_eval pure: median **1.147×** (worse).
Same-stream dual_qmm **1.003×**. Reject both; defaults OFF. Gate_up still needs GEMM-class win.

## Blocked thr physics close-out (2026-07-25 tip `ed1b485d`+)
See `2026-07-25-blocked-thr-physics.md`.

- Best formal S1: multi-process + cache_eval thr **1.109×**, gap ratio **1.113**, TTFT **0.899**.
- Need thr ≥1.15 and gap ≤0.90; pure ≤0.96 under keep_base not achieved (best pure ~0.993).
- Gate_up dual-qmm residual (~3.26s / ~40% pure): Metal dual, compile, FFI, async, dual-stream all reject or noise.
- Exclusive thr ceiling ~1.03–1.05×; dual-hold gap 160–220 ms.
- **Decision `not_yet`.** Gates not relaxed. Full S0–S3 flip not claimed without thr headroom.
- Open path only: true GEMM-class dual-gate that beats MLX steel qmm (not yet achieved).

## Multi-process + cache_eval + Qwen-only DENSE_FFN_COMPILE=1 (2026-07-25)
Cool 3-rep dual-target S1 (`2026-07-25-s1-mp-cache-eval-qwen-compile`):

| metric | AX median | mlxcel median | ratio | gate |
|--------|----------:|--------------:|------:|------|
| thr tok/s | 20.013 | 18.023 | **1.110** | FAIL (≥1.15) |
| gap p95 ms | 39.18 | 35.26 | **1.111** | FAIL (≤0.90); abs ≤50 PASS |
| TTFT p95 | 8914 | 9934 | **0.897** | PASS |

Contention residual (faster Qwen free GPU for Gemma) is a **wash** vs best cache_eval thr **1.109**. Reject Qwen-only compile stack for thr; keep common `DENSE_FFN_COMPILE=0` on multi-process targets.

## Pure larger-chunk under cache_eval (2026-07-25)
Hypothesis residual: `#672` cache-only eval barrier count ∝ ceil(tokens/chunk); pure A/B c512 vs c768 vs c1024 under `CACHE_ONLY_CHUNK_EVAL=1` keep_base (bar ≤0.96).

| chunk | cold median ms | ratio vs c512 |
|------:|---------------:|--------------:|
| 512 | 8315 | 1.000 |
| 768 | 8438 | **1.015** |
| 1024 | 11202 | **1.347** |

**Decision `reject_keep_c512`.** Larger chunks do not cut pure under cache_eval (eval-barrier hypothesis fails; matmul/chunk shape tax dominates). No cool multi-process S1 remeasure. Artifact: `2026-07-25-pure-chunk768-1024-cache-eval-ab/`.

## GEMM-class dual-gate hybrid pure A/B (2026-07-25)
mlxcel review: multi-token bits=8 uses op-at-a-time dual steel qmm (#680) — no dual-output GEMM. AX load-time `pack_dense_ffn_gate_up` is the GEMM-class lever (one steel qmm, single X load). Long Gemma prefill defaults to split; prior packed A/B ~1.03×.

Hypothesis: packed qmm helps but `packed_geglu_metal` hurts. Hybrid under cache_eval keep_base:

| variant | cold median ms | ratio vs base |
|---------|---------------:|--------------:|
| base (split prefill) | 8277 | 1.000 |
| hybrid (packed qmm + split Metal GEGLU) | 8479 | **1.024** |
| packed_metal (packed qmm + packed GEGLU metal) | 8307 | **1.004** |

**Decision `reject_keep_base`.** Neither path ≤0.96. Packed single-qmm does not beat split dual steel qmm on M5 Max for bits=8 multi-token (weight bandwidth dominates X re-read). No cool S1 / no S0–S3. Artifact: `2026-07-25-pure-packed-split-geglu-ab/`.

## Concurrent-tax residual cool S1 (2026-07-25)
Path B probe: multi-process + cache_eval + asymmetric wired
(Gemma `WIRED_LIMIT_SCALE=0.55`, Qwen `0.30` + `BATCHED_DECODE=0`).

| metric | ratio | vs best cache_eval 1.109 |
|--------|------:|--------------------------|
| thr | **1.100** | worse |
| gap | **1.138** | worse |
| TTFT | **0.906** | FAIL (was PASS) |
| e2e max med | ~9460 ms | still ~9.46s (need ≲9.08) |

**Decision `reject`.** Concurrent resource asymmetry does not cut multi-process tax enough for thr 1.15 or gap 0.90. Best stack remains multi-process + Gemma cache_eval thr **1.109**. Artifact: `2026-07-25-s1-mp-cache-eval-concurrent-tax/`.

## Best-practices close-out (2026-07-25)
See `2026-07-25-best-practices-path.md`. Recommended: accept **`not_yet`**, keep
locked gates, stop residual thrash until product supplies A/B/C with a real
design or explicit gate policy.

## Terminal decision (2026-07-25 resume)
See `2026-07-25-terminal-decision.md`. Campaign closes **`not_yet`** under locked
gates; best thr **1.109×**; full S0–S3 flip not claimed.
