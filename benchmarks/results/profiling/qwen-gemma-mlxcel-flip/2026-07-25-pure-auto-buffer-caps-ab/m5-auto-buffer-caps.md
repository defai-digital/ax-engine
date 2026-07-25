# Lever: M5 Metal command-buffer caps (mlxcel hardware residual)

## Residual

Best formal S1: multi-process AX + Gemma `CACHE_ONLY_CHUNK_EVAL` thr **1.109×**,
gap ratio **1.113**. Need ~**3.7%** scenario wall cut for thr ≥1.15×.

## mlxcel source

`mlxcel_core::hardware::metal_ops_per_buffer_default` (hardware.rs ~128–161):

- **M1–M4** (no NA): set `MLX_MAX_OPS_PER_BUFFER=1000` (+11–13% on M1 Ultra).
- **M5+** (`has_neural_accelerator`): **leave MLX default** — M5 Max sweep
  flat / larger buffers **slower** (`docs/benchmark_results/gemma3n-decode-profile-m5max.md`,
  issue #358).

`apply_metal_ops_per_buffer_default()` only sets the env when non-M5.

## AX current

`maybe_raise_metal_buffer_caps` (`weights.rs`): for eligible families (Gemma
included), always raises **`MLX_MAX_MB_PER_BUFFER=1024`** and
**`MLX_MAX_OPS_PER_BUFFER=1000`** when `AX_MLX_AUTO_BUFFER_CAPS` default ON.
Comment cites MoE gather-QMM win; dense Gemma measured ~0.998 on older A/B —
**not** re-checked under multi-process cache_eval pure, and not matched to
mlxcel's M5 "leave default" policy.

## Change (opt-in pure A/B)

Under cache_eval keep_base pure Gemma 13.8k cold 3-rep:

1. **base**: `AX_MLX_AUTO_BUFFER_CAPS` unset (raise 1024/1000)
2. **mlx_default**: `AX_MLX_AUTO_BUFFER_CAPS=0` (mlxcel M5 parity)

Keep if median pure ratio ≤ **0.96**. Else reject (keep auto-raise).

If pure wins, cool multi-process S1 with `AUTO_BUFFER_CAPS=0` on both model
processes; only full S0–S3 if thr≥1.15 and gap/TTFT pass.

## Success

Cool multi-process S1 thr ≥1.15 **and** gap ≤0.90 **and** TTFT ≤0.90 → flip.
Else not_yet; gates unchanged.

## Result (mbp-m5 pure, 2026-07-25, 3-rep)

| variant | median cold ms | ratio |
|---------|---------------:|------:|
| base (AUTO_BUFFER_CAPS=1) | 8401 | 1.000 |
| mlx_default (AUTO_BUFFER_CAPS=0) | 8305 | **0.989** |

~1.1% pure cut; need ≤0.96 for thr physics. Decision **keep_base** (auto-raise stays ON).
No multi-process S1 remeasure — insufficient thr headroom alone.
