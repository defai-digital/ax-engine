# Lever: fuse attn input RMSNorm + packed QKV

## Profile residual (pure-reprofile3, mbp-m5 Gemma 13.8k)

After exhausted gate_up (~3.26s), down (~2.08s), o_proj (~0.78s):

| Stage | wall |
|-------|------|
| **`pre_sdpa_qkv_proj`** | **~1.07s** |
| `sdpa` | ~1.22s (GPU flash; no metal4 flag residual — mlxcel voids use_metal4) |
| rope_kv / qk_norm | prior rejects |

Pick **QKV** residual: host graph for `input_layernorm` + packed QKV qmm every layer every chunk (attn_norm is currently outside the profiled qkv timer but on pure wall).

## mlxcel source

`gemma4.rs` `forward_with_profile` (~2490–2535):

```
h_attn = self.input_layernorm.forward(x);
(h_attn, stored_kv) = self.self_attn.forward(&h_attn, ...);
```

Attention path: separate `q_proj` / `k_proj` / `v_proj` (default) or opt-in
`MLXCEL_GEMMA4_ENABLE_FUSED_QKV` packed projection — still after layernorm.
No `rms_norm → qmm` C++ fuse.

## AX residual

- Default pure chunk-512: **packed QKV** (prefer_split max=511; 2026-07-25 pure-split-qkv keep packed).
- Portable: `rms_norm(hidden, attn_norm)` then `qw_with_policy(packed)`.
- Change: opt-in `AX_MLX_ATTN_NORM_QKV_FUSE=1` → `rms_norm_quantized_matmul` one C++ call.

## Success metric

Pure 13.8k cold median ≤ 0.925× OFF → cool S1 thr≥21 → S0–S3. Else reject, keep OFF.


## A/B results (mbp-m5, 2026-07-25-pure-attn-norm-qkv-fuse-ab)

3-rep alternating pure Gemma 13.8k cold wall:

| arm | cold samples (ms) | median | mean |
|-----|-------------------|--------|------|
| OFF | 8614, 9143, 8714 | **8714** | 8823 |
| ON | 9278, 9013, 8570 | **9013** | 8954 |

- **ratio_median ≈ 1.034**
- **ratio_mean ≈ 1.015**
- **decision: `reject_keep_off`**

No ≥7.5% pure cut → skip cool S1 and S0–S3 flip. Defaults remain OFF.
