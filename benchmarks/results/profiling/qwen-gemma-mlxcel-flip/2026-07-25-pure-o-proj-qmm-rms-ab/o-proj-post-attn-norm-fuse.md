# Lever: fuse o_proj qmm + post_attention_layernorm

## Profile residual (pure Gemma 13.8k, pure-reprofile2 / re-profile3)

| Stage | wall | vs rope |
|-------|------|---------|
| `post_attn_ffn_gate_up` | ~3.38s | larger (already exhausted dual/pack/compile) |
| `post_attn_ffn_down` | ~2.15s | larger (geglu-down / qmm-rms reject) |
| `sdpa` | ~1.24s | larger |
| `pre_sdpa_qkv_proj` | ~1.10s | larger (pack QKV already default ON) |
| **`post_attn_output_proj`** | **~0.78s** | **> rope_kv ~0.54s** |
| `pre_sdpa_rope_kv` | ~0.54s | prior reject (compiled Q-path) |

Pick: **output_proj** — next largest residual that still has an untried fuse vs mlxcel's op-at-a-time chain, without re-entering rejected gate_up/down Metal paths.

## mlxcel source

`gemma4.rs`:
- `project_output`: transpose → reshape → `o_proj.forward` (UnifiedLinear qmm only).
- Layer residual (`forward_with_profile` ~2494–2498):
  ```
  h_attn = self_attn(...);              // includes o_proj
  h_attn = post_attention_layernorm(h_attn);
  after_attn = add(x, h_attn);
  ```
Separate ops; no o_proj+post_attn_ln fuse.

## AX residual

- `attention_output_projection` = o_proj qmm (optional attn_gate).
- Then separate `rms_norm` when `attn_post_norm` (Gemma sandwich maps `post_attention_layernorm`).
- `quantized_matmul_rms_norm` already exists for **down + post_ffn** fuse (default OFF after pure confirm ~1.00×).

## Change

Opt-in `AX_MLX_O_PROJ_QMATMUL_RMS_NORM=1`: when `attn_post_norm` present and no attn_gate, route o_proj through `quantized_matmul_rms_norm` (one C++ graph-build for qmm + rms). Default OFF.

## Success metric

Pure 13.8k cold median ≤ 0.925× portable → cool S1 thr≥21 → S0–S3 flip. Else reject, keep OFF, decision not_yet.

## A/B results (mbp-m5, 2026-07-25-pure-o-proj-qmm-rms-ab)

3-rep alternating pure Gemma 13.8k cold wall:

| arm | cold samples (ms) | median | mean |
|-----|-------------------|--------|------|
| OFF (portable) | 8589, 9132, 9865 | **9132** | 9195 |
| ON (o_proj qmm+rms) | 9321, 8909, 9161 | **9161** | 9130 |

- **ratio_median ≈ 1.003**
- **ratio_mean ≈ 0.993**
- keep_if_ratio_lt = 0.925
- **decision: `reject_keep_off`**

Usage OK on all reps. No ≥7.5% pure cut → skip cool S1 and S0–S3 flip. Defaults remain OFF.

### pure-reprofile3 (same host, same day)

BASE mean cold ≈ 8982 ms; stage dominance unchanged: gate_up ~3.26s, down ~2.08s, sdpa ~1.22s, qkv ~1.07s, **o_proj ~0.78s**, rope_kv ~0.53s.
