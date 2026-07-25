# Lever: compiled proportional RoPE / Q-path (mlxcel parity)

## Profile residual (mbp-m5 pure Gemma 13.8k, pure-reprofile2)

| Stage | wall_us | ~share of profile stack |
|-------|---------|-------------------------|
| `pre_sdpa_qkv_proj` | ~1.10s | large (not this lever) |
| `pre_sdpa_qk_norm` | ~0.27s | host+kernel |
| `pre_sdpa_rope_kv` | ~0.54s | host+kernel |
| **qk_norm + rope_kv** | **~0.81s** | **~9% of ~8.8s pure wall** |

BASE pure cold mean ≈ **8845 ms**. Gate for next stage: pure median cut **≥ ~7.5%** (≤ ~0.925×).

Physics ceiling for this residual alone: even zeroing qk_norm+rope entirely is ~9% of wall — only if the stage is pure waste. Realistic compile/FFI savings are a fraction of 0.81s. **Proportional-only** further limits impact: Gemma4 has **8 full_attention** layers (proportional freqs) + **40 sliding** (default rope); mlxcel compiles only the proportional Q-path.

## mlxcel source

1. `rope_proportional.rs` — `compute_proportional_rope_freqs` (full-head exponents + `inf` tail); `apply_proportional_rope` → `compiled_proportional_rope` when `last_dim == head_dim`.
2. `mlx_cxx_bridge.cpp` ~3071–3118 — `get_compiled_proportional_rope` / `compiled_proportional_rope`: `mx::compile` around `fast::rope(x, head_dim, …, offset_arr, freqs)`.
3. `mlx_cxx_bridge.cpp` ~3120–3200 — `compiled_q_path_proportional`: single compile of  
   `reshape → fast::rms_norm → transpose → full-head rope(freqs)`  
   for Gemma4 full-attention layers only.

## AX residual

- Freqs built by `build_gemma4_proportional_rope_freqs` (`config.rs`) — same divisor convention as mlxcel.
- Portable path: multi-FFI `qk_norm` + `mlx_fast_rope` with freqs (`attention.rs` / `rope_bhsd_batch_offset_safe`).
- Existing C++ composite `qk_norm_rope_bhsd_from_proj` (as_strided → rms_norm → rope) gated by `AX_MLX_GEMMA_DIRECT_CPP_QK_NORM_ROPE` (**default OFF** after pure-gemma-qkrope-ab ~**+1.6%** worse cold wall).
- **Gap vs mlxcel:** composite was imperative; no `mx::compile` wrap of the proportional Q-path.

## Change (this iteration)

1. `mlx-sys` `activation.cpp`: opt-in `AX_MLX_COMPILED_QK_NORM_ROPE=1` wraps freqs path in process-static `mx::compile` (reshape/rms_norm/transpose/rope, offset as scalar array) — mlxcel `compiled_q_path_proportional` parity.
2. Pure A/B on mbp-m5 requires **both** `AX_MLX_GEMMA_DIRECT_CPP_QK_NORM_ROPE=1` (enter C++ entry) **and** `AX_MLX_COMPILED_QK_NORM_ROPE=1`. Default both OFF.

## Success metric

Pure 13.8k cold **median ≤ 0.925×** portable baseline → then cool exclusive S1 abs thr ≥ ~21 → ≥3-rep dual-target S0–S3. If ratio ≥ 0.925, **reject**, keep defaults OFF, decision remains **not_yet** (gates not relaxed).

## A/B results (mbp-m5, 2026-07-25-pure-compiled-qk-rope-ab)

3-rep alternating pure Gemma 13.8k cold wall:

| arm | cold samples (ms) | median | mean |
|-----|-------------------|--------|------|
| portable (both flags 0) | 8572, 9091, 9129 | **9091** | 8931 |
| gemma_direct + compiled | 9265, 8919, 9073 | **9073** | 9086 |

- **ratio_median ≈ 0.998**
- **ratio_mean ≈ 1.017**
- keep_if_ratio_lt = 0.925
- **decision: `reject_keep_portable`**

Usage OK on all reps (`prompt_tokens=13826`, `cached_tokens=13824`). No ≥7.5% pure cut → **skip cool S1 and S0–S3 flip**. Defaults remain OFF. Exclusive thr≥21 and full `flip` still blocked.
