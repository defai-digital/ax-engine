use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, as_strided, astype, eval, reshape, rms_norm,
    scaled_dot_product_attention_with_mask, transpose,
};
use std::time::Instant;

use crate::attention_mask::create_causal_mask;
use crate::kv_cache::{
    MlxKVCache, MlxKvCompressionDecodeOutcome, MlxKvCompressionFusedDecodeTiming,
};

use super::super::config::ModelConfig;
use super::super::profile::saturating_profile_us;
use super::norm::{rms_norm_no_scale_bshd, use_flat_qk_norm_path};

#[allow(dead_code)]
pub(crate) fn bhsd_view_from_proj(
    qw_out: &MlxArray,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
) -> MlxArray {
    let batch = qw_out.shape()[0];
    let n_heads_i32 = n_heads as i32;
    let head_dim_i64 = head_dim as i64;
    let n_heads_head_dim = (n_heads * head_dim) as i64;
    let seq_n_heads_head_dim = (seq * n_heads * head_dim) as i64;
    let shape = [batch, n_heads_i32, seq as i32, head_dim as i32];
    let strides = [seq_n_heads_head_dim, head_dim_i64, n_heads_head_dim, 1_i64];
    as_strided(qw_out, &shape, &strides, 0, None)
}

pub(crate) fn qk_norm_bhsd_from_proj(
    qw_out: &MlxArray,
    norm: Option<&MlxArray>,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    if use_flat_qk_norm_path() {
        let batch = qw_out.shape()[0] as usize;
        let bshd = reshape(
            qw_out,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        let normed = qk_norm_bshd(bshd, norm, n_heads, head_dim, seq, eps);
        return transpose(&normed, &[0, 2, 1, 3], None);
    }

    let bhsd = bhsd_view_from_proj(qw_out, n_heads, head_dim, seq);
    let Some(n) = norm else { return bhsd };
    rms_norm(&bhsd, Some(n), eps, None)
}

pub(crate) fn qk_norm_bshd(
    x: MlxArray,
    norm: Option<&MlxArray>,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    let Some(n) = norm else { return x };
    if use_flat_qk_norm_path() {
        let batch = x.shape()[0] as usize;
        let flat = reshape(&x, &[(batch * n_heads * seq) as i32, head_dim as i32], None);
        let normed = rms_norm(&flat, Some(n), eps, None);
        return reshape(
            &normed,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
    }
    rms_norm(&x, Some(n), eps, None)
}

/// Apply optional V RMSNorm in BSHD, then convert to BHSD for attention/KV cache.
pub(crate) fn prepare_value_bhsd_from_proj(
    v_raw: &MlxArray,
    v_norm_no_scale: bool,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    if use_flat_qk_norm_path() {
        let batch = v_raw.shape()[0] as usize;
        let bshd = reshape(
            v_raw,
            &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
            None,
        );
        return prepare_value_bhsd(bshd, v_norm_no_scale, n_heads, head_dim, seq, eps);
    }

    let bhsd = bhsd_view_from_proj(v_raw, n_heads, head_dim, seq);
    if v_norm_no_scale {
        rms_norm(&bhsd, None, eps, None)
    } else {
        bhsd
    }
}

/// Apply optional V RMSNorm in BSHD, then convert to BHSD for attention/KV cache.
pub(crate) fn prepare_value_bhsd(
    v: MlxArray,
    v_norm_no_scale: bool,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    let v = if v_norm_no_scale {
        rms_norm_no_scale_bshd(v, n_heads, head_dim, seq, eps)
    } else {
        v
    };
    transpose(&v, &[0, 2, 1, 3], None)
}

/// Convert SDPA output `[B, H, S, D]` to `[B, S, H * D]` for the output projection.
///
/// Single-token decode has `S == 1`, so `[B, H, 1, D]` is already laid out in
/// the same head-major order the flattened projection expects. In that case a
/// direct reshape is equivalent to `transpose([0, 2, 1, 3]) + reshape` while
/// saving one layout graph node per transformer layer.
pub(crate) fn flatten_attention_output_bhsd(
    attn_sdpa: &MlxArray,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
) -> MlxArray {
    let batch = attn_sdpa.shape()[0];
    let hidden = (n_heads * head_dim) as i32;
    if seq == 1 {
        return reshape(attn_sdpa, &[batch, 1, hidden], None);
    }

    let attn_out = transpose(attn_sdpa, &[0, 2, 1, 3], None);
    reshape(&attn_out, &[batch, seq as i32, hidden], None)
}

/// Build the array mask only when the fast causal/none modes cannot express it.
pub(crate) fn attention_mask_array(
    seq_len: usize,
    key_len: usize,
    sliding_window: Option<usize>,
) -> Option<MlxArray> {
    if seq_len == 0 {
        return None;
    }

    let offset = key_len.saturating_sub(seq_len);
    if let Some(window) = sliding_window {
        // When there is no KV-cache offset and the prompt fits entirely within
        // the window, the sliding constraint never fires: every (i, j) pair
        // where i >= j already satisfies i - j < seq_len <= window.  A plain
        // causal mask is equivalent, so return None and let the caller use the
        // fast ScaledDotProductAttentionMask::Causal path.  This avoids adding
        // an O(seq²) boolean array (and ~5 graph nodes) per sliding-attention
        // layer to the MLX computation graph.
        if offset > 0 || seq_len > window {
            return Some(create_causal_mask(seq_len, offset, Some(window)));
        }
        // Fall through to the standard causal / None path below.
    }
    if offset > 0 && seq_len > 1 {
        return Some(create_causal_mask(seq_len, offset, None));
    }
    None
}

pub(crate) fn attention_mask_key_len(
    seq_len: usize,
    key_len: usize,
    sliding_window: Option<usize>,
) -> usize {
    if seq_len == 1
        && let Some(window) = sliding_window.filter(|window| *window > 0)
    {
        return key_len.min(window);
    }
    key_len
}

pub(crate) fn full_precision_attention(
    q_rope: &MlxArray,
    cached_k: &MlxArray,
    cached_v: &MlxArray,
    query_scale: f32,
    seq: usize,
    mask_opt: &Option<MlxArray>,
) -> MlxArray {
    let mask = match mask_opt.as_ref() {
        Some(mask) => ScaledDotProductAttentionMask::Array(mask),
        None if seq > 1 => ScaledDotProductAttentionMask::Causal,
        None => ScaledDotProductAttentionMask::None,
    };
    scaled_dot_product_attention_with_mask(q_rope, cached_k, cached_v, query_scale, mask, None)
}

pub(crate) struct TurboQuantExperimentalDecodeOutput {
    pub attention: MlxArray,
    pub outcome: MlxKvCompressionDecodeOutcome,
    pub timing: MlxKvCompressionFusedDecodeTiming,
}

pub(crate) enum TurboQuantQueryReadbackArray<'a> {
    Borrowed(&'a MlxArray),
    Owned(MlxArray),
}

impl TurboQuantQueryReadbackArray<'_> {
    pub fn as_array(&self) -> &MlxArray {
        match self {
            Self::Borrowed(array) => array,
            Self::Owned(array) => array,
        }
    }
}

pub(crate) fn turboquant_query_readback_array(
    q_rope: &MlxArray,
) -> TurboQuantQueryReadbackArray<'_> {
    if q_rope.dtype() == MlxDtype::Float32 {
        TurboQuantQueryReadbackArray::Borrowed(q_rope)
    } else {
        TurboQuantQueryReadbackArray::Owned(astype(q_rope, MlxDtype::Float32, None))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn turboquant_decode_attention_experimental(
    cache: &MlxKVCache,
    layer_idx: usize,
    q_rope: &MlxArray,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    query_scale: f32,
) -> Option<TurboQuantExperimentalDecodeOutput> {
    if seq != 1 {
        return None;
    }
    let expected_scale = (head_dim as f32).sqrt().recip();
    if !query_scale.is_finite() || query_scale <= 0.0 {
        return None;
    }

    let query_readback_started = Instant::now();
    let q_readback = turboquant_query_readback_array(q_rope);
    let q_readback_array = q_readback.as_array();
    eval(&[q_readback_array]);
    let q_data = q_readback_array.data_f32();
    let query_readback_wall_us = saturating_profile_us(query_readback_started) as u64;
    if q_data.len() != n_heads.saturating_mul(head_dim) {
        return None;
    }
    let query_multiplier = query_scale / expected_scale;
    let scaled_queries;
    let query_values = if (query_multiplier - 1.0).abs() > 1.0e-6 {
        scaled_queries = q_data
            .iter()
            .map(|value| *value * query_multiplier)
            .collect::<Vec<_>>();
        scaled_queries.as_slice()
    } else {
        q_data
    };

    let total_tokens = cache.seq_len.saturating_add(seq);
    if let Ok(decoded) = cache
        .debug_turboquant_shadow_decode_attention_metal_flat_query_timed_for_layer_with_total_tokens(
            layer_idx,
            query_values,
            n_heads,
            total_tokens,
        )
    {
        let output_started = Instant::now();
        return turboquant_attention_output_array_from_flat(
            decoded.outputs,
            n_heads,
            head_dim,
            q_rope.dtype(),
        )
        .map(|attention| {
            let mut timing = decoded.timing;
            timing.query_readback_wall_us = query_readback_wall_us;
            timing.output_staging_wall_us = saturating_profile_us(output_started) as u64;
            TurboQuantExperimentalDecodeOutput {
                attention,
                outcome: MlxKvCompressionDecodeOutcome::Metal,
                timing,
            }
        });
    }

    None
}

pub(crate) fn turboquant_attention_output_array_from_flat(
    output: Vec<f32>,
    n_heads: usize,
    head_dim: usize,
    dtype: MlxDtype,
) -> Option<MlxArray> {
    if output.len() != n_heads.saturating_mul(head_dim) {
        return None;
    }

    let out = MlxArray::from_raw_data(
        output.as_ptr().cast(),
        output.len() * std::mem::size_of::<f32>(),
        &[1, n_heads as i32, 1, head_dim as i32],
        MlxDtype::Float32,
    );
    if dtype == MlxDtype::Float32 {
        Some(out)
    } else {
        Some(astype(&out, dtype, None))
    }
}

/// Pre-compute one SDPA mask per unique sliding-window size before the layer
/// loop.  Mirrors Python mlx_lm's `_make_masks` and Swift's `maskByType`:
/// all layers of the same attention type share one mask object, avoiding
/// N redundant `create_causal_mask` calls (= N × ~7 MLX graph nodes) per
/// forward pass.
pub(crate) fn build_layer_masks(
    cfg: &ModelConfig,
    n_layers: usize,
    seq: usize,
    key_len: usize,
) -> Vec<Option<MlxArray>> {
    if cfg.layer_configs.is_empty() {
        // For uniform-SWA models (Mistral3, Mixtral) cfg.global_sliding_window is set.
        // Pass it so prefill masks correctly limit attention to the window.
        let m = attention_mask_array(seq, key_len, cfg.global_sliding_window);
        if m.is_none() {
            return vec![None; n_layers];
        }
        (0..n_layers).map(|_| m.clone()).collect()
    } else {
        // Fast path: fresh single-chunk prefill (offset==0) where every layer's
        // sliding window contains all prompt tokens → all masks are None.
        // Avoids HashMap allocation + per-layer mask computation for this common case.
        let offset = key_len.saturating_sub(seq);
        if offset == 0
            && cfg
                .layer_configs
                .iter()
                .all(|lc| lc.sliding_window.is_none_or(|w| seq <= w))
        {
            return vec![None; n_layers];
        }
        // Fast path: single-token decode — all masks are None.
        // Global layers: attention_mask_array(1, key_len, None) returns None because
        // the `offset > 0 && seq_len > 1` condition is false when seq_len == 1.
        // Sliding-window layers: KV is pre-truncated to window size; single query
        // attends all retained keys without masking.
        if seq == 1 {
            return vec![None; n_layers];
        }
        let mut memo: std::collections::HashMap<(Option<usize>, usize), Option<MlxArray>> =
            std::collections::HashMap::with_capacity(cfg.layer_configs.len());
        cfg.layer_configs
            .iter()
            .map(|lc| {
                let mask_key_len = attention_mask_key_len(seq, key_len, lc.sliding_window);
                // For decode (seq==1) with sliding window, key_len is already truncated
                // to ≤ window by attention_mask_key_len. The single query can attend to
                // all retained keys, so no mask is needed. This matches mlx_lm's behavior
                // where N==1 → None mask for all layers (base.py create_attention_mask).
                if seq == 1 && lc.sliding_window.is_some() {
                    return None;
                }
                memo.entry((lc.sliding_window, mask_key_len))
                    .or_insert_with(|| attention_mask_array(seq, mask_key_len, lc.sliding_window))
                    .clone()
            })
            .collect()
    }
}
