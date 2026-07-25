# Lever: shape-stable dual-gate / #705 compile under cache_eval chunks

## Residual

Best multi-process S1 thr **1.109×**. Need pure cut ≤**0.96** under
`CACHE_ONLY_CHUNK_EVAL` keep_base (~4% of ~8.2s ≈ 330 ms).

Profile (pure 13.8k): **gate_up dual qmm ~3.26s** (largest stage).

Prior pure A/Bs **without** multi-process cache_eval baseline:

| lever | ratio | note |
|-------|------:|------|
| `AX_MLX_COMPILED_DUAL_GATE_UP` | ~1.021 | full-prompt shape mix |
| `AX_MLX_COMPILED_QGELU_PREFILL_SHAPED` (#705) | ~1.02 | full MLP shape mix |

## mlxcel / why re-try under cache_eval

- mlxcel #705: multi-token non-4bit uses **shape-specific** `mx::compile`
  (not shapeless) so prefill keeps large-matmul kernels
  (`mlx_cxx_bridge.cpp` ~1850–1917; AX `activation.cpp` dual_gate_up +
  qgelu prefill_shaped).
- With **cache_eval + prefill-chunk 512**, almost every chunk is exactly
  `[1, 512, H]` (27 steps; last may differ). Shape-specific compile
  **reuses one compiled graph** across chunks instead of recompiling per
  varying full-prompt / residual lengths.

Hypothesis: prior rejects measured under deferred full-prompt graphs where
shape_sig churn amortized poorly; under cache_eval the fixed-512 regime
matches #705's intended recovery.

## Change (env A/B only; already implemented)

Pure 13.8k cold 3-rep under cache_eval keep_base:

1. **base**: both OFF  
2. **dual_gate**: `AX_MLX_COMPILED_DUAL_GATE_UP=1`  
3. **shaped**: `AX_MLX_COMPILED_QGELU_PREFILL_SHAPED=1`  
4. **both**: dual_gate + shaped  

Keep if median ratio ≤ **0.96**. Else reject.

## Success

Cool multi-process S1 thr ≥1.15 + gap ≤0.90 + TTFT ≤0.90 → full S0–S3 flip.
Else not_yet.

## Result (mbp-m5 pure cache_eval, cool 3-rep, 2026-07-25)

| variant | median cold ms | ratio |
|---------|---------------:|------:|
| base | 8376 | 1.000 |
| dual_gate | 8406 | **1.003** |
| shaped (#705) | 8339 | **0.996** |
| both | 8319 | **0.993** |

Best **both ~0.7% pure cut** — still far from keep-if **0.96** (~4% needed for thr 1.15 physics).
Decision **keep_base** (do not enable dual_gate or PREFILL_SHAPED by default).
No cool S1 / full S0–S3.
