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
