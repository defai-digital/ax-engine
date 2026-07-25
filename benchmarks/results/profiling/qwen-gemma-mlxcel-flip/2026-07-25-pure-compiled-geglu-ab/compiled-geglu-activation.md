# Lever: compiled GeGLU activation (mlxcel multi-token FFN residual)

## Residual

Best formal S1 thr **1.109×** (multi-process + cache_eval). Need ~**3.7%** pure /
scenario wall cut for thr ≥1.15×. Profile: FFN activation ~0.32s of pure
~8.2s; gate_up dual qmm ~3.3s already exhausted (dual Metal / dual_gate_up
compile / dual_qmm_geglu FFI rejects).

## mlxcel source

- `mlxcel_core::compiled_geglu_approx_activation` —
  `mlx_cxx_bridge.cpp` ~1351–1368: process-static
  `mx::compile(fn, /*shapeless=*/true)` over `gelu_tanh_approx(gate) * x`.
- Call site: `models/gemma4.rs` ~917–920 multi-token bits=8 dense MLP
  (op-at-a-time dual `UnifiedLinear::forward` + compiled GeGLU; full MLP
  compile disabled for multi-token 8-bit by #680).

AX production uses **custom Metal GEGLU** (`AX_MLX_GEGLU_MUL_METAL` default ON)
or imperative `gelu_approx_mul`. No process-static compiled GeGLU parity.

## AX change

1. `ax_mlx_compiled_geglu_approx_activation` + Rust wrapper (env-gated).
2. `geglu()` tries compiled path first when `AX_MLX_COMPILED_GEGLU_ACTIVATION=1`.
3. Pure 13.8k cold 3-rep under **cache_eval keep_base**:
   - `base`: metal ON, compiled OFF
   - `compiled`: metal OFF, compiled ON
   - `nometal`: metal OFF, compiled OFF

Keep if median pure ratio ≤ **0.96** vs base. Else reject, default OFF.

## Success

Cool multi-process S1 thr ≥1.15 **and** gap ≤0.90 **and** TTFT ≤0.90 → full
S0–S3 flip. Else not_yet.

## Result (mbp-m5 pure cache_eval, 2026-07-25, 3-rep)

| variant | median cold ms | ratio |
|---------|---------------:|------:|
| base (Metal GEGLU) | 8410 | 1.000 |
| compiled (mlxcel mx::compile) | 8565 | **1.018** |
| nometal (imperative) | 8911 | **1.060** |

Decision **keep_base** — Metal GEGLU stays default ON; `AX_MLX_COMPILED_GEGLU_ACTIVATION` default OFF.
No thr headroom for cool S1 / S0–S3 flip.
