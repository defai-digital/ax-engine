# Lever: native MLX causal for full-attention offset prefill (SDPA residual)

## Profile residual (pure-reprofile3, mbp-m5 Gemma 13.8k)

| Stage | wall |
|-------|------|
| **`sdpa`** | **~1.22s** |
| gate_up / down | larger, already rejected |
| qkv / o_proj / rope | host-graph fuses rejected |

Gemma4 hybrid: **8 full_attention** + **40 sliding** (window 1024). Full layers
grow KV to the full pure context (~13.8k); sliding is view-trimmed (~1.5k with
`multi_token_window_views`). Full-layer SDPA dominates the 1.22s stack after the
first prefill chunk.

Physics ceiling: zeroing all of SDPA is ~13.5% of ~9s wall — only if the stage
is pure waste. Realistic win from mask-mode change is a fraction of full-layer
SDPA + mask graph build.

## mlxcel source

1. `gemma4.rs` `Attention::attend` (~1881–1918): if no array mask →
   `causal_attention(..., window_size)`.
2. `lib.rs` `causal_attention` (~2800–2908): when `window_size == 0` and
   `softcap == 0`, always
   `ffi_fast_scaled_dot_product_attention_causal` / `metal4_causal_attention`
   — **never** materializes an offset bool array for full-window layers.
3. MLX steel kernels (`steel_attention.h` ~237–318): with `do_causal`,
   absolute query position is `local_i + qL_off` where `qL_off` is the query
   sequence start offset (`params.h`: “Offset in query sequence start”). Mask
   rule `row_pos < col_pos` ⇒ attend iff `j ≤ offset + i`, same as
   `create_causal_mask(seq, offset, None)`.

## AX residual

`attention_mask_array(seq, key_len, None)` for `offset = key_len - seq > 0`
returns `Some(create_causal_mask(...))` → `ScaledDotProductAttentionMask::Array`.

That is **correct** but pays:
- O(seq × key_len) bool array build per unique full mask (shared across 8 full layers),
- array-mask SDPA path instead of native causal / NAX-friendly causal route mlxcel uses.

Sliding layers still need array masks when the window constraint is active
(unchanged).

## Change

Opt-in `AX_MLX_NATIVE_OFFSET_CAUSAL=1`: full-attention offset multi-token returns
`None` so `full_precision_attention` uses `Causal` mode. Default **OFF** for
pure A/B; sliding path unchanged.

## Success metric

Pure 13.8k cold median ≤ 0.925× OFF → cool S1 thr≥21 → S0–S3 flip.
Else reject / document; gates not relaxed.


## A/B results (mbp-m5, 2026-07-25-pure-native-offset-causal-ab)

3-rep alternating pure Gemma 13.8k cold wall (usage OK, text `" The"` both arms):

| arm | cold samples (ms) | median | mean |
|-----|-------------------|--------|------|
| OFF (array offset mask) | 8476, 9183, 8619 | **8619** | 8759 |
| ON (native causal) | 9168, 8845, 9581 | **9168** | 9198 |

- **ratio_median ≈ 1.064**
- **ratio_mean ≈ 1.050**
- **decision: `reject_keep_off`**

Native offset-causal does not cut pure wall (slightly worse under thermal noise). Sliding-window array masks remain mandatory for 40/48 layers. SDPA residual is GPU-dominated; host mask-mode parity with mlxcel is not a flip lever.

### Impassable note (SDPA host residual)

Even a free 50% cut of the entire 1.22s SDPA stage would be ~6.8% of ~9s pure wall — under the 7.5% gate for exclusive thr headroom. Full-attention mask-mode change touches only 8/48 layers. No further pure SDPA host-graph lever is residual-backed at ≥7.5% without a new GPU kernel (out of current residual-backed scope after measured rejects). Skip cool S1 and S0–S3 flip; gates not relaxed.
