# Lever: dual affine qmm (one FFI, Metal GEGLU kept)

## Residual

Profile pure Gemma 13.8k: **`post_attn_ffn_gate_up` ~3.26s** (largest stage).
Prior rejects on this stage: dual Metal (8–25×), dual_qmm_geglu FFI (1.09× —
bundled imperative GEGLU), dual_gate_up compile (~1.00–1.02×), async dual
submit (1.007×), #705 full MLP shaped (~0.993 best under cache_eval).

## mlxcel

`gemma4.rs` ~917–920 multi-token bits=8:
```
gate = gate_proj.forward(x);   // one quantized_linear_forward FFI
up   = up_proj.forward(x);     // one quantized_linear_forward FFI
hidden = compiled_geglu_approx_activation(gate, up);
down = down_proj.forward(hidden);
```

AX portable path: two `qw_with_policy` → two Rust→C++ `quantized_matmul` FFIs,
then **Metal GEGLU** (proven better than compiled GeGLU on M5, pure 1.018× for
compile).

## Change (new)

`ax_mlx_dual_affine_qmm` + `dual_affine_qmm()`: one C++ call builds both affine
qmm graphs and returns `(gate, up)`. **No mx::compile, no GEGLU** — Metal
GEGLU stays on. Opt-in `AX_MLX_DUAL_AFFINE_QMM=1` (default OFF).

Hypothesis vs rejected `DUAL_QMM_GEGLU`: that path replaced Metal GEGLU with
imperative gelu (~1.06× alone). This residual only collapses dual-qmm FFI.

## Pure A/B (mbp-m5, cache_eval keep_base)

base OFF vs dual ON; keep if median cold ≤ **0.96×**.

## Success

Pure ≤0.96 → cool multi-process S1; thr≥1.15 + gap/TTFT → full S0–S3 flip.

## Result (mbp-m5 pure cache_eval, 2026-07-25, 3-rep)

| variant | median cold ms | ratio |
|---------|---------------:|------:|
| base (two qw FFI) | 8261 | 1.000 |
| dual_qmm (one C++ dual affine qmm) | 8276 | **1.002** |

Decision **keep_base** / reject. Default OFF. Text first-token parity OK (` The`).
Gate_up host-FFI collapse alone does not move pure wall toward ≤0.96.
