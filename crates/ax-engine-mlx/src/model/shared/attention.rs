use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, as_strided, astype, broadcast_to,
    concatenate, eval, matmul, multiply,
    qk_norm_rope_bhsd_from_proj as direct_qk_norm_rope_bhsd_from_proj, reshape, rms_norm, rope,
    scaled_dot_product_attention_with_mask, slice, slice_update, softmax_precise, transpose,
};
use std::time::Instant;

use crate::attention_mask::{create_causal_mask, create_ring_sliding_mask};
use crate::fastpath;
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

fn direct_qk_norm_rope_route_allowed(route_enabled: bool, norm: Option<&MlxArray>) -> bool {
    route_enabled && !use_flat_qk_norm_path() && norm.is_some()
}

pub(crate) fn direct_qk_norm_rope_route_enabled(norm: Option<&MlxArray>) -> bool {
    direct_qk_norm_rope_route_allowed(fastpath::direct_cpp_qk_norm_rope_enabled(), norm)
}

pub(crate) fn direct_qk_norm_rope_route_enabled_for_family(
    model_family: &str,
    norm: Option<&MlxArray>,
) -> bool {
    let qwen_family_default = qwen_direct_qk_norm_rope_default_family(model_family)
        && fastpath::qwen_direct_cpp_qk_norm_rope_enabled();
    direct_qk_norm_rope_route_allowed(
        fastpath::direct_cpp_qk_norm_rope_enabled() || qwen_family_default,
        norm,
    )
}

fn qwen_direct_qk_norm_rope_default_family(model_family: &str) -> bool {
    model_family.starts_with("qwen")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_bhsd_from_proj(
    qw_out: &MlxArray,
    norm: Option<&MlxArray>,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
    rope_dims: usize,
    rope_base: Option<f32>,
    token_offset: usize,
    rope_freqs: Option<&MlxArray>,
) -> MlxArray {
    qk_norm_rope_bhsd_from_proj_with_route(
        qw_out,
        norm,
        n_heads,
        head_dim,
        seq,
        eps,
        rope_dims,
        rope_base,
        token_offset,
        rope_freqs,
        direct_qk_norm_rope_route_enabled(norm),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_bhsd_from_proj_flat(
    qw_out: &MlxArray,
    norm: Option<&MlxArray>,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
    rope_dims: usize,
    rope_base: Option<f32>,
    token_offset: usize,
    rope_freqs: Option<&MlxArray>,
) -> MlxArray {
    let batch = qw_out.shape()[0] as usize;
    let bshd = reshape(
        qw_out,
        &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
        None,
    );
    let normed = if let Some(n) = norm {
        rms_norm(&bshd, Some(n), eps, None)
    } else {
        bshd
    };
    let bhsd = transpose(&normed, &[0, 2, 1, 3], None);
    rope(
        &bhsd,
        rope_dims as i32,
        false,
        rope_base,
        1.0,
        token_offset as i32,
        rope_freqs,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_bhsd_from_proj_with_route(
    qw_out: &MlxArray,
    norm: Option<&MlxArray>,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
    rope_dims: usize,
    rope_base: Option<f32>,
    token_offset: usize,
    rope_freqs: Option<&MlxArray>,
    direct_route_enabled: bool,
) -> MlxArray {
    if direct_qk_norm_rope_route_allowed(direct_route_enabled, norm) {
        return direct_qk_norm_rope_bhsd_from_proj(
            qw_out,
            norm,
            n_heads as i32,
            head_dim as i32,
            eps,
            rope_dims as i32,
            false,
            rope_base,
            token_offset as i32,
            rope_freqs,
            None,
        );
    }

    let q = qk_norm_bhsd_from_proj(qw_out, norm, n_heads, head_dim, seq, eps);
    rope(
        &q,
        rope_dims as i32,
        false,
        rope_base,
        1.0,
        token_offset as i32,
        rope_freqs,
        None,
    )
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

pub(crate) fn prepare_value_bhsd_from_proj_flat(
    v_raw: &MlxArray,
    v_norm_no_scale: bool,
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> MlxArray {
    let batch = v_raw.shape()[0] as usize;
    let bshd = reshape(
        v_raw,
        &[batch as i32, seq as i32, n_heads as i32, head_dim as i32],
        None,
    );
    prepare_value_bhsd(bshd, v_norm_no_scale, n_heads, head_dim, seq, eps)
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
        // mlx-lm's Gemma4 RotatingKVCache uses max_size == sliding_window and
        // returns no mask for single-token decode once only the retained window
        // is presented to SDPA. When key_len <= window the sliding constraint is
        // already satisfied for the lone query, so an explicit all-true mask is
        // unnecessary graph work.
        if seq_len == 1 && key_len <= window {
            return None;
        }
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
    if let Some(window) = sliding_window.filter(|window| *window > 0) {
        if seq_len == 1 {
            return key_len.min(window);
        }
        // Multi-token forwards: each query attends at most the `window` keys
        // ending at its own position, so the chunk needs only the last
        // `window + seq_len - 1` cached tokens. Must stay in lockstep with the
        // retained-view width chosen at the KV append site
        // (`families::standard::layer_forward`), hence the shared flag.
        if seq_len > 1 && crate::fastpath::multi_token_window_views_enabled() {
            return key_len.min(window + seq_len - 1);
        }
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

pub(crate) fn bidirectional_full_precision_attention(
    q_rope: &MlxArray,
    cached_k: &MlxArray,
    cached_v: &MlxArray,
    query_scale: f32,
) -> MlxArray {
    scaled_dot_product_attention_with_mask(
        q_rope,
        cached_k,
        cached_v,
        query_scale,
        ScaledDotProductAttentionMask::None,
        None,
    )
}

/// Attention with per-head learned sinks (GPT-OSS).
///
/// Computes standard scaled dot-product attention but appends a virtual "sink"
/// score per head before softmax. The sink absorbs probability mass that would
/// otherwise be distributed across real tokens, improving long-context
/// coherence. After softmax the sink column is excluded from the value
/// weighted sum.
///
/// `q`: `[B, n_q_heads, seq, head_dim]`
/// `k`: `[B, n_kv_heads, key_len, head_dim]`
/// `v`: `[B, n_kv_heads, key_len, head_dim]`
/// `sinks`: `[n_q_heads]` — per-head additive sink bias
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_with_sinks(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    sinks: &MlxArray,
    query_scale: f32,
    seq: usize,
    mask_opt: &Option<MlxArray>,
) -> MlxArray {
    // scores = Q @ K^T * scale → [B, n_q_heads, seq, key_len]
    let k_t = transpose(k, &[0, 1, 3, 2], None);
    let scores = multiply(&matmul(q, &k_t, None), &scalar_array(query_scale), None);

    // Append sink scores as an extra column → [B, n_q_heads, seq, key_len + 1]
    // Broadcast sinks from [n_q_heads] → [1, n_q_heads, 1, 1] → [B, n_q_heads, seq, 1]
    let batch = scores.shape()[0];
    let n_heads = scores.shape()[1];
    let sink_broad = broadcast_to(
        &reshape(sinks, &[1, n_heads, 1, 1], None),
        &[batch, n_heads, seq as i32, 1],
        None,
    );
    let scores_with_sink = concatenate(&[&scores, &sink_broad], 3, None);

    // Apply causal/sliding mask extended for the sink column.
    // The sink position is always visible (unmasked), so pad the mask with
    // ones along the last axis for the extra column.
    let true_val = scalar_array(1.0);
    let masked_scores = if let Some(mask) = mask_opt.as_ref() {
        let extended_mask = mlx_sys::pad(mask, &[3], &[0], &[1], &true_val, None);
        let neg_inf = scalar_array(f32::NEG_INFINITY);
        let zero = scalar_array(0.0);
        let penalty = mlx_sys::where_cond(&extended_mask, &zero, &neg_inf, None);
        mlx_sys::add(&scores_with_sink, &penalty, None)
    } else if seq > 1 {
        let key_len = k.shape()[2] as usize;
        let causal = create_causal_mask(seq, key_len.saturating_sub(seq), None);
        let extended_mask = mlx_sys::pad(&causal, &[3], &[0], &[1], &true_val, None);
        let neg_inf = scalar_array(f32::NEG_INFINITY);
        let zero = scalar_array(0.0);
        let penalty = mlx_sys::where_cond(&extended_mask, &zero, &neg_inf, None);
        mlx_sys::add(&scores_with_sink, &penalty, None)
    } else {
        scores_with_sink
    };

    // Softmax over the last axis (real tokens + sink).
    let weights = softmax_precise(&masked_scores, -1, None);

    // Exclude the sink column from value weighting.
    // weights[:, :, :, :-1] @ V
    let weights_real = slice(
        &weights,
        &[0, 0, 0, 0],
        &[
            weights.shape()[0],
            weights.shape()[1],
            weights.shape()[2],
            weights.shape()[3] - 1,
        ],
        &[1, 1, 1, 1],
        None,
    );

    matmul(&weights_real, v, None)
}

/// Create a scalar MlxArray from a single f32 value.
fn scalar_array(val: f32) -> MlxArray {
    MlxArray::from_raw_data(
        &val as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    )
}

/// Pre-allocated KV concatenation buffer for bidirectional attention.
///
/// On the first denoise step, the full `[cached_k, canvas_k]` and
/// `[cached_v, canvas_v]` concatenations are built via `concatenate`.
/// On subsequent steps, only the canvas slice is updated via
/// `slice_update`, avoiding re-copying the cached prompt prefix.
///
/// The attention mask for canvas self-attention is also cached here,
/// since it depends only on `cached_seq`, `canvas_size`, and `window`,
/// all of which are constant within a diffusion block.
pub(crate) struct KVConcatBuffer {
    /// Full K buffer: `[B, H, cached_seq + canvas_size, D]`.
    pub full_k: Option<MlxArray>,
    /// Full V buffer: `[B, H, cached_seq + canvas_size, D]`.
    pub full_v: Option<MlxArray>,
    /// Length of the cached (prompt) sequence along axis 2.
    pub cached_seq: usize,
    /// Cached attention mask for bidirectional canvas self-attention.
    /// Keyed by `cached_seq` (constant within a block); rebuilt when the
    /// buffer is first populated or when the cached sequence length changes
    /// (new block committed).
    pub cached_mask: Option<MlxArray>,
    /// The `cached_seq` for which `cached_mask` was built.
    cached_mask_seq: usize,
}

impl KVConcatBuffer {
    pub fn new() -> Self {
        Self {
            full_k: None,
            full_v: None,
            cached_seq: 0,
            cached_mask: None,
            cached_mask_seq: usize::MAX,
        }
    }
}

/// Bidirectional (non-causal) attention for DiffusionGemma denoiser.
///
/// Each canvas query attends bidirectionally to:
/// - ALL cached prompt key/value entries (cross-attention, no window constraint)
/// - ALL canvas key/value entries (self-attention), optionally limited by
///   a symmetric sliding window of width `2 * window + 1`
///
/// `cached_k/v`: `[B, n_kv_heads, cached_seq, head_dim]` — read-only prompt KV.
/// `canvas_q`: `[B, n_q_heads, canvas_size, head_dim]` — RoPE-applied queries.
/// `canvas_k/v`: `[B, n_kv_heads, canvas_size, head_dim]` — RoPE-applied keys/values.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bidirectional_attention(
    canvas_q: &MlxArray,
    cached_k: &MlxArray,
    cached_v: &MlxArray,
    canvas_k: &MlxArray,
    canvas_v: &MlxArray,
    query_scale: f32,
    sliding_window: Option<usize>,
    mut kv_buffer: Option<&mut KVConcatBuffer>,
) -> MlxArray {
    let canvas_size = canvas_q.shape()[2] as usize;

    // Build full KV: either via slice_update (buffer path) or concatenate.
    let (full_k, full_v) = if let Some(ref mut buf) = kv_buffer {
        if buf.full_k.is_none() {
            // First step: build buffer via concatenate.
            let fk = concatenate(&[cached_k, canvas_k], 2, None);
            let fv = concatenate(&[cached_v, canvas_v], 2, None);
            buf.cached_seq = cached_k.shape()[2] as usize;
            buf.full_k = Some(fk.clone());
            buf.full_v = Some(fv.clone());
            (fk, fv)
        } else {
            // Subsequent steps: update only the canvas slice.
            let total = buf.cached_seq + canvas_size;
            let start = [0, 0, buf.cached_seq as i32, 0];
            let stop = [
                canvas_k.shape()[0],
                canvas_k.shape()[1],
                total as i32,
                canvas_k.shape()[3],
            ];
            let strides = [1, 1, 1, 1];
            let fk = slice_update(
                buf.full_k.as_ref().unwrap(),
                canvas_k,
                &start,
                &stop,
                &strides,
                None,
            );
            let fv = slice_update(
                buf.full_v.as_ref().unwrap(),
                canvas_v,
                &start,
                &stop,
                &strides,
                None,
            );
            buf.full_k = Some(fk.clone());
            buf.full_v = Some(fv.clone());
            (fk, fv)
        }
    } else {
        // Standard path: concatenate every time.
        let full_k = concatenate(&[cached_k, canvas_k], 2, None);
        let full_v = concatenate(&[cached_v, canvas_v], 2, None);
        (full_k, full_v)
    };

    // Build mask only when a symmetric sliding window must be enforced.
    // Without a window, every canvas position attends to every key (no mask).
    // When a KV buffer is available, cache the mask since it depends only on
    // (canvas_size, cached_seq, window), all constant within a diffusion block.
    let mask = sliding_window.map(|window| {
        let full_key_len = full_k.shape()[2] as usize;
        let cached_seq = full_key_len.saturating_sub(canvas_size);
        if let Some(buf) = kv_buffer {
            if buf.cached_mask.is_none() {
                let m = build_bidirectional_canvas_mask(canvas_size, cached_seq, window);
                buf.cached_mask = Some(m.clone());
                buf.cached_mask_seq = cached_seq;
                m
            } else {
                buf.cached_mask.clone().unwrap()
            }
        } else {
            build_bidirectional_canvas_mask(canvas_size, cached_seq, window)
        }
    });

    let mask_arg = mask
        .as_ref()
        .map(ScaledDotProductAttentionMask::Array)
        .unwrap_or(ScaledDotProductAttentionMask::None);
    scaled_dot_product_attention_with_mask(canvas_q, &full_k, &full_v, query_scale, mask_arg, None)
}

/// Build a bidirectional mask for canvas self-attention with cross-attention
/// to a cached prompt prefix.
///
/// Layout: `[canvas_size, cached_seq + canvas_size]`
/// - Columns `0..cached_seq` (prompt): always `true` (unconstrained cross-attention).
/// - Columns `cached_seq..` (canvas): `true` when `|i - (j - cached_seq)| < window`.
fn build_bidirectional_canvas_mask(
    canvas_size: usize,
    cached_seq: usize,
    window: usize,
) -> MlxArray {
    let total_keys = cached_seq + canvas_size;
    let mut mask = vec![0_u8; canvas_size * total_keys];
    for qi in 0..canvas_size {
        // Prompt prefix: always allowed.
        for ki in 0..cached_seq {
            mask[qi * total_keys + ki] = 1;
        }
        // Canvas: symmetric window around query position.
        for ki in 0..canvas_size {
            let diff = qi.abs_diff(ki);
            if diff < window {
                mask[qi * total_keys + cached_seq + ki] = 1;
            }
        }
    }
    MlxArray::from_raw_data(
        mask.as_ptr(),
        mask.len(),
        &[canvas_size as i32, total_keys as i32],
        MlxDtype::Bool,
    )
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
        // Key length follows the retained-view trim so mask and KV view widths
        // stay in lockstep (see `attention_mask_key_len`).
        let mask_key_len = attention_mask_key_len(seq, key_len, cfg.global_sliding_window);
        let m = attention_mask_array(seq, mask_key_len, cfg.global_sliding_window);
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

/// Ring-aware variant of [`build_layer_masks`] for the bounded-rollback
/// rotating sliding KV path.
///
/// When the cache is in bounded-rollback rotating mode (`rotating_slack >
/// 0`) and this forward crosses a layer's sliding window, that layer's
/// append presents the **full ring** (`window + slack` slots, unordered) to
/// SDPA, so its mask must be a slot-validity mask
/// ([`create_ring_sliding_mask`]) instead of an ordered causal-window mask —
/// including for single-token decode, where the `slack` non-live slots would
/// otherwise receive softmax mass. Global layers and windows this forward
/// has not crossed keep the ordered logic unchanged.
///
/// Both this builder and the append site
/// (`families::standard::layer_forward`) derive their decisions from
/// [`MlxKVCache::sliding_ring_layout`], so mask and KV view cannot disagree.
pub(crate) fn build_layer_masks_for_forward(
    cfg: &ModelConfig,
    n_layers: usize,
    seq: usize,
    key_len: usize,
    cache: &MlxKVCache,
) -> Vec<Option<MlxArray>> {
    // Pure-mode rings (slack 0) stay mask-free on their single-token decode
    // path and never take multi-token ring appends; every other mode is the
    // ordered logic. Only bounded mode needs ring masks.
    if cache.rotating_sliding_slack() == 0 {
        return build_layer_masks(cfg, n_layers, seq, key_len);
    }

    if cfg.layer_configs.is_empty() {
        return match cache.sliding_ring_layout(cfg.global_sliding_window, seq) {
            Some(ring) if ring.needs_mask(seq) => {
                let m = Some(create_ring_sliding_mask(
                    seq,
                    ring.window,
                    ring.capacity,
                    ring.write_start,
                ));
                (0..n_layers).map(|_| m.clone()).collect()
            }
            Some(_) => vec![None; n_layers],
            None => build_layer_masks(cfg, n_layers, seq, key_len),
        };
    }

    let any_ring = cfg
        .layer_configs
        .iter()
        .any(|lc| cache.sliding_ring_layout(lc.sliding_window, seq).is_some());
    if !any_ring {
        return build_layer_masks(cfg, n_layers, seq, key_len);
    }

    let mut memo: std::collections::HashMap<Option<usize>, Option<MlxArray>> =
        std::collections::HashMap::with_capacity(2);
    cfg.layer_configs
        .iter()
        .map(|lc| {
            memo.entry(lc.sliding_window)
                .or_insert_with(|| {
                    match cache.sliding_ring_layout(lc.sliding_window, seq) {
                        Some(ring) if ring.needs_mask(seq) => Some(create_ring_sliding_mask(
                            seq,
                            ring.window,
                            ring.capacity,
                            ring.write_start,
                        )),
                        Some(_) => None,
                        None => {
                            // Ordered path for this window: mirror
                            // `build_layer_masks`'s per-layer logic.
                            if seq == 1 {
                                return None;
                            }
                            let mask_key_len =
                                attention_mask_key_len(seq, key_len, lc.sliding_window);
                            attention_mask_array(seq, mask_key_len, lc.sliding_window)
                        }
                    }
                })
                .clone()
        })
        .collect()
}

/// Build Gemma4 multimodal PrefixLM masks.
///
/// Mirrors the reference `masking_utils.blockwise_overlay` (transformers
/// v5.10, used by Gemma4 Unified with `use_bidirectional_attention="vision"`):
/// vision soft-token blocks attend bidirectionally to themselves, OR-composed
/// onto every layer's base mask — `causal OR block` on full-attention layers
/// and `(causal AND sliding_window) OR block` on sliding-attention layers.
/// The reference applies the overlay to both mask kinds and never filters a
/// block against the window size.
pub(crate) fn build_layer_masks_with_media_ranges(
    cfg: &ModelConfig,
    n_layers: usize,
    seq: usize,
    key_len: usize,
    media_ranges: &[(usize, usize)],
) -> Vec<Option<MlxArray>> {
    let ranges: Vec<(usize, usize)> = media_ranges
        .iter()
        .copied()
        .filter(|(start, end)| start <= end)
        .collect();
    if ranges.is_empty() {
        return build_layer_masks(cfg, n_layers, seq, key_len);
    }

    if cfg.layer_configs.is_empty() {
        let mask = media_prefix_mask_array(seq, key_len, cfg.global_sliding_window, &ranges);
        return vec![Some(mask); n_layers];
    }

    // One mask per unique window size (Gemma4 alternates sliding/global), the
    // same memoization shape as `build_layer_masks`.
    let mut memo: std::collections::HashMap<Option<usize>, MlxArray> =
        std::collections::HashMap::with_capacity(2);
    cfg.layer_configs
        .iter()
        .map(|lc| {
            Some(
                memo.entry(lc.sliding_window)
                    .or_insert_with(|| {
                        media_prefix_mask_array(seq, key_len, lc.sliding_window, &ranges)
                    })
                    .clone(),
            )
        })
        .collect()
}

fn media_prefix_mask_array(
    seq_len: usize,
    key_len: usize,
    sliding_window: Option<usize>,
    media_ranges: &[(usize, usize)],
) -> MlxArray {
    let offset = key_len.saturating_sub(seq_len);
    let mut mask = vec![0_u8; seq_len.saturating_mul(key_len)];
    for query in 0..seq_len {
        let query_abs = offset + query;
        for key in 0..key_len {
            let mut allowed = key <= query_abs;
            if allowed && let Some(window) = sliding_window {
                allowed = query_abs - key < window;
            }
            if !allowed {
                allowed = media_ranges.iter().any(|(start, end)| {
                    query_abs >= *start && query_abs <= *end && key >= *start && key <= *end
                });
            }
            mask[query * key_len + key] = u8::from(allowed);
        }
    }
    MlxArray::from_raw_data(
        mask.as_ptr(),
        mask.len(),
        &[seq_len as i32, key_len as i32],
        MlxDtype::Bool,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        build_bidirectional_canvas_mask, build_layer_masks_with_media_ranges,
        media_prefix_mask_array, qwen_direct_qk_norm_rope_default_family,
    };
    use crate::model::{LayerConfig, ModelConfig};
    use mlx_sys::{MlxArray, eval};

    fn mask_data(mask: &MlxArray) -> Vec<u8> {
        eval(&[mask]);
        let len = mask.nbytes();
        let ptr = mask.data_raw();
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    #[test]
    fn multi_token_windowed_view_matches_full_view_sliding_attention() {
        // Oracle for the multi-token retained-window views: trimming sliding
        // K/V to the last `window + seq - 1` tokens (with the matching
        // trimmed mask) must produce the same attention output as the full
        // history plus the full-width sliding mask. GQA shape (2 query heads,
        // 1 KV head) to also cover the broadcast path.
        use super::{attention_mask_array, attention_mask_key_len, full_precision_attention};
        use mlx_sys::{astype, reshape, slice};

        let (n_heads, kv_heads, head_dim) = (2usize, 1usize, 4usize);
        let seq = 3usize;
        let window = 4usize;
        let key_len = 12usize;
        let retained = attention_mask_key_len(seq, key_len, Some(window));
        assert_eq!(retained, window + seq - 1);

        let fill = |n: usize, seed: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
        };
        let q = reshape(
            &MlxArray::from_f32_slice(&fill(n_heads * seq * head_dim, 0.1)),
            &[1, n_heads as i32, seq as i32, head_dim as i32],
            None,
        );
        let k = reshape(
            &MlxArray::from_f32_slice(&fill(kv_heads * key_len * head_dim, 0.5)),
            &[1, kv_heads as i32, key_len as i32, head_dim as i32],
            None,
        );
        let v = reshape(
            &MlxArray::from_f32_slice(&fill(kv_heads * key_len * head_dim, 0.9)),
            &[1, kv_heads as i32, key_len as i32, head_dim as i32],
            None,
        );

        let full_mask = attention_mask_array(seq, key_len, Some(window));
        assert!(full_mask.is_some(), "offset sliding prefill needs a mask");
        let out_full = full_precision_attention(&q, &k, &v, 1.0, seq, &full_mask);

        let start = (key_len - retained) as i32;
        let trim = |arr: &MlxArray| {
            slice(
                arr,
                &[0, 0, start, 0],
                &[1, kv_heads as i32, key_len as i32, head_dim as i32],
                &[1, 1, 1, 1],
                None,
            )
        };
        let trim_mask = attention_mask_array(seq, retained, Some(window));
        assert!(trim_mask.is_some(), "trimmed view still needs a mask");
        let out_trim = full_precision_attention(&q, &trim(&k), &trim(&v), 1.0, seq, &trim_mask);

        let read_f32 = |arr: &MlxArray| -> Vec<f32> {
            let arr = astype(arr, mlx_sys::MlxDtype::Float32, None);
            eval(&[&arr]);
            let len = arr.nbytes() / std::mem::size_of::<f32>();
            let ptr = arr.data_raw() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
        };
        let a = read_f32(&out_full);
        let b = read_f32(&out_trim);
        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "windowed view diverged from full view: max diff {max_diff}"
        );
    }

    #[test]
    fn qwen_direct_qk_norm_rope_defaults_cover_all_qwen_families() {
        assert!(qwen_direct_qk_norm_rope_default_family("qwen3"));
        assert!(qwen_direct_qk_norm_rope_default_family("qwen3_5"));
        assert!(qwen_direct_qk_norm_rope_default_family("qwen3_next"));
        assert!(!qwen_direct_qk_norm_rope_default_family("gemma4"));
        assert!(!qwen_direct_qk_norm_rope_default_family("llama3"));
    }

    #[test]
    fn bidirectional_canvas_mask_matches_cached_prefix_plus_canvas_width() {
        let mask = build_bidirectional_canvas_mask(4, 3, 2);

        assert_eq!(mask.shape(), vec![4, 7]);
        assert_eq!(
            mask_data(&mask),
            vec![
                1, 1, 1, 1, 1, 0, 0, //
                1, 1, 1, 1, 1, 1, 0, //
                1, 1, 1, 0, 1, 1, 1, //
                1, 1, 1, 0, 0, 1, 1,
            ]
        );
    }

    #[test]
    fn media_prefix_mask_or_extends_sliding_window_inside_range() {
        let mask = media_prefix_mask_array(6, 6, Some(2), &[(1, 3)]);

        assert_eq!(mask.shape(), vec![6, 6]);
        assert_eq!(
            mask_data(&mask),
            vec![
                1, 0, 0, 0, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                0, 1, 1, 1, 0, 0, //
                0, 1, 1, 1, 0, 0, //
                0, 0, 0, 1, 1, 0, //
                0, 0, 0, 0, 1, 1,
            ]
        );
    }

    #[test]
    fn media_prefix_mask_extends_causal_mask_without_window() {
        // Full-attention layers: `causal OR block` — vision tokens at 1..=3
        // attend bidirectionally to themselves; everything else is causal.
        let mask = media_prefix_mask_array(6, 6, None, &[(1, 3)]);

        assert_eq!(
            mask_data(&mask),
            vec![
                1, 0, 0, 0, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                1, 1, 1, 1, 1, 0, //
                1, 1, 1, 1, 1, 1,
            ]
        );
    }

    #[test]
    fn media_prefix_mask_keeps_block_larger_than_sliding_window() {
        // A vision block larger than the window still attends to itself in
        // full: the reference blockwise overlay is not filtered by window size.
        let mask = media_prefix_mask_array(5, 5, Some(2), &[(0, 3)]);

        assert_eq!(
            mask_data(&mask),
            vec![
                1, 1, 1, 1, 0, //
                1, 1, 1, 1, 0, //
                1, 1, 1, 1, 0, //
                1, 1, 1, 1, 0, //
                0, 0, 0, 1, 1,
            ]
        );
    }

    fn interleaved_mask_test_config() -> ModelConfig {
        let layer = |sliding_window: Option<usize>| LayerConfig {
            head_dim: 1,
            rope_theta: 10000.0,
            rope_dims: 0,
            rope_freqs: None,
            sliding_window,
            kv_source_layer: None,
            v_norm_no_scale: false,
        };
        ModelConfig {
            model_family: "gemma4_unified".to_string(),
            layer_count: 2,
            hidden_size: 1,
            intermediate_size: 0,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            vocab_size: 1,
            rope_theta: 10000.0,
            rope_dims: 0,
            attn_output_gate: false,
            query_scale: 1.0,
            final_logit_softcapping: None,
            moe_expert_count: 0,
            moe_experts_per_token: 0,
            moe_expert_intermediate_size: 0,
            layer_configs: vec![layer(Some(2)), layer(None)],
            global_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: true,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 0.0,
            attn_temperature_scale: 0.0,
            intermediate_size_mlp: 0,
            moe_layer_freq: 0,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            gpt_oss_uses_mxfp4_experts: false,
        }
    }

    #[test]
    fn media_layer_masks_apply_block_overlay_to_full_attention_layers() {
        let cfg = interleaved_mask_test_config();

        let masks = build_layer_masks_with_media_ranges(&cfg, 2, 6, 6, &[(1, 3)]);

        assert_eq!(masks.len(), 2);
        // Sliding layer: (causal AND window 2) OR block.
        assert_eq!(
            mask_data(masks[0].as_ref().expect("sliding layer mask")),
            vec![
                1, 0, 0, 0, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                0, 1, 1, 1, 0, 0, //
                0, 1, 1, 1, 0, 0, //
                0, 0, 0, 1, 1, 0, //
                0, 0, 0, 0, 1, 1,
            ]
        );
        // Full-attention layer: causal OR block — previously plain causal.
        assert_eq!(
            mask_data(masks[1].as_ref().expect("full-attention layer mask")),
            vec![
                1, 0, 0, 0, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                1, 1, 1, 1, 0, 0, //
                1, 1, 1, 1, 1, 0, //
                1, 1, 1, 1, 1, 1,
            ]
        );
    }
}
