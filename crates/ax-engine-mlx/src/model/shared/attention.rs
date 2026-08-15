use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, arange, as_strided, astype, async_eval,
    broadcast_to, concatenate, contiguous, cos, eval, multiply, outer,
    qk_norm_rope_bhsd_from_proj as direct_qk_norm_rope_bhsd_from_proj, reshape, rms_norm, rope,
    scaled_dot_product_attention_with_mask, scaled_dot_product_attention_with_mask_and_sinks, sin,
    slice, slice_update, subtract, transpose,
};
#[cfg(test)]
use mlx_sys::{matmul, softmax_precise};
use std::cell::{Cell, RefCell};

use crate::attention_mask::{create_causal_mask, create_ring_sliding_mask};
use crate::fastpath;
use crate::kv_cache::{MlxKVCache, SlidingRingLayout};

use super::super::config::ModelConfig;
use super::norm::{rms_norm_no_scale_bshd, use_flat_qk_norm_path};

/// Materialize the Qwen full-attention activation once before QKVO qmm.
pub(crate) fn qwen_prefill_maybe_eval_attn_input(x: &MlxArray, model_family: &str, seq: i32) {
    qwen_prefill_maybe_eval_attn_input_for(
        x,
        fastpath::qwen_prefill_eval_attn_input_enabled(),
        model_family,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_eval_attn_input`].
pub(crate) fn qwen_prefill_maybe_eval_attn_input_for(
    x: &MlxArray,
    enabled: bool,
    model_family: &str,
    seq: i32,
) {
    if fastpath::should_qwen_prefill_eval_attn_input_for(enabled, model_family, seq) {
        eval(&[x]);
    }
}

/// Submit full-attn SDPA before flatten + o_proj is encoded.
pub(crate) fn qwen_prefill_maybe_async_sdpa(attn_sdpa: &MlxArray, model_family: &str, seq: i32) {
    qwen_prefill_maybe_async_sdpa_for(
        attn_sdpa,
        fastpath::qwen_prefill_async_sdpa_enabled(),
        model_family,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_async_sdpa`].
pub(crate) fn qwen_prefill_maybe_async_sdpa_for(
    attn_sdpa: &MlxArray,
    enabled: bool,
    model_family: &str,
    seq: i32,
) {
    if fastpath::should_qwen_prefill_async_sdpa_for(enabled, model_family, seq) {
        async_eval(&[attn_sdpa]);
    }
}

/// Submit the first-KV write before residual + FFN is encoded.
pub(crate) fn gemma4_prefill_maybe_async_first_kv(
    k: &MlxArray,
    v: &MlxArray,
    model_family: &str,
    seq: i32,
) {
    gemma4_prefill_maybe_async_first_kv_for(
        k,
        v,
        fastpath::should_gemma4_async_first_kv_p128(model_family, seq),
    );
}

/// Pure helper for [`gemma4_prefill_maybe_async_first_kv`].
pub(crate) fn gemma4_prefill_maybe_async_first_kv_for(k: &MlxArray, v: &MlxArray, enabled: bool) {
    if enabled {
        async_eval(&[k, v]);
    }
}

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
    let gemma_family_default = gemma_direct_qk_norm_rope_default_family(model_family)
        && fastpath::gemma_direct_cpp_qk_norm_rope_enabled();
    direct_qk_norm_rope_route_allowed(
        fastpath::direct_cpp_qk_norm_rope_enabled() || qwen_family_default || gemma_family_default,
        norm,
    )
}

fn qwen_direct_qk_norm_rope_default_family(model_family: &str) -> bool {
    model_family.starts_with("qwen")
}

fn gemma_direct_qk_norm_rope_default_family(model_family: &str) -> bool {
    // gemma4, gemma3, gemma2, gemma4-assistant, …
    model_family.starts_with("gemma")
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

thread_local! {
    static REUSE_ROPE_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static REUSE_ROPE_TABLE: RefCell<Option<ReuseRopeTable>> = const { RefCell::new(None) };
}

struct ReuseRopeTable {
    key: (i32, i32, i32, u64),
    cos: MlxArray,
    sin: MlxArray,
}

/// Arm last-token-chunk reuse of one NeoX cos/sin table for this thread.
pub(crate) fn set_qwen_prefill_reuse_rope_active(active: bool) {
    REUSE_ROPE_ACTIVE.with(|slot| slot.set(active));
    if !active {
        REUSE_ROPE_TABLE.with(|slot| *slot.borrow_mut() = None);
    }
}

fn qwen_prefill_reuse_rope_active() -> bool {
    REUSE_ROPE_ACTIVE.with(Cell::get)
}

fn reuse_rope_key(
    token_offset: i32,
    seq: i32,
    rope_dims: i32,
    rope_base: Option<f32>,
    rope_freqs: Option<&MlxArray>,
) -> (i32, i32, i32, u64) {
    let tag = rope_freqs
        .map(|freqs| freqs as *const MlxArray as u64)
        .or_else(|| rope_base.map(|base| base.to_bits() as u64))
        .unwrap_or(0);
    (token_offset, seq, rope_dims, tag)
}

fn reused_neox_cos_sin(
    token_offset: i32,
    seq: i32,
    rope_dims: i32,
    rope_base: Option<f32>,
    rope_freqs: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let key = reuse_rope_key(token_offset, seq, rope_dims, rope_base, rope_freqs);
    REUSE_ROPE_TABLE.with(|slot| {
        if let Some(cached) = slot.borrow().as_ref()
            && cached.key == key
        {
            return (cached.cos.clone(), cached.sin.clone());
        }
        let (cos_h, sin_h) =
            build_neox_rope_cos_sin(token_offset, seq, rope_dims, rope_base, rope_freqs);
        *slot.borrow_mut() = Some(ReuseRopeTable {
            key,
            cos: cos_h.clone(),
            sin: sin_h.clone(),
        });
        (cos_h, sin_h)
    })
}

fn build_neox_rope_cos_sin(
    token_offset: i32,
    seq: i32,
    rope_dims: i32,
    rope_base: Option<f32>,
    rope_freqs: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let positions = arange(
        f64::from(token_offset),
        f64::from(token_offset + seq),
        1.0,
        MlxDtype::Float32,
        None,
    );
    let half = rope_dims / 2;
    let inv_freq = if let Some(freqs) = rope_freqs {
        freqs.clone()
    } else {
        let base = rope_base.unwrap_or(10_000.0);
        let data: Vec<f32> = (0..half)
            .map(|index| 1.0 / base.powf((2 * index) as f32 / rope_dims as f32))
            .collect();
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[half],
            MlxDtype::Float32,
        )
    };
    let theta = outer(&positions, &inv_freq, None);
    let cos_h = reshape(&cos(&theta, None), &[1, 1, seq, half], None);
    let sin_h = reshape(&sin(&theta, None), &[1, 1, seq, half], None);
    (cos_h, sin_h)
}

/// Apply a cached NeoX cos/sin table to `[B, H, S, D]`.
pub(crate) fn apply_reused_neox_rope(
    bhsd: &MlxArray,
    rope_dims: i32,
    rope_base: Option<f32>,
    token_offset: i32,
    rope_freqs: Option<&MlxArray>,
) -> MlxArray {
    let shape = bhsd.shape();
    let seq = shape[2];
    let head_dim = shape[3];
    let (cos_h, sin_h) = reused_neox_cos_sin(token_offset, seq, rope_dims, rope_base, rope_freqs);
    let rotary = if head_dim > rope_dims {
        slice(
            bhsd,
            &[0, 0, 0, 0],
            &[shape[0], shape[1], seq, rope_dims],
            &[1, 1, 1, 1],
            None,
        )
    } else {
        bhsd.clone()
    };
    let half = rope_dims / 2;
    let x1 = slice(
        &rotary,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], seq, half],
        &[1, 1, 1, 1],
        None,
    );
    let x2 = slice(
        &rotary,
        &[0, 0, 0, half],
        &[shape[0], shape[1], seq, rope_dims],
        &[1, 1, 1, 1],
        None,
    );
    let rx = subtract(
        &multiply(&x1, &cos_h, None),
        &multiply(&x2, &sin_h, None),
        None,
    );
    let ry = add(
        &multiply(&x2, &cos_h, None),
        &multiply(&x1, &sin_h, None),
        None,
    );
    let embedded = concatenate(&[&rx, &ry], -1, None);
    if head_dim > rope_dims {
        let pass = slice(
            bhsd,
            &[0, 0, 0, rope_dims],
            &[shape[0], shape[1], seq, head_dim],
            &[1, 1, 1, 1],
            None,
        );
        concatenate(&[&embedded, &pass], -1, None)
    } else {
        embedded
    }
}

/// Apply RoPE to a BHSD array, working around an MLX <= 0.31.x bug: for a
/// batched single-position input ([B, H, 1, D] with B > 1) and a non-zero
/// position offset, MLX `fast::rope` rotates only batch 0 and returns the
/// remaining batches unrotated (verified against a manual reference on
/// 0.31.2; fixed upstream in 0.32.0). RoPE is per-vector, so folding the
/// batch dimension into the head dimension is exact and sidesteps the bug.
pub(crate) fn rope_bhsd_batch_offset_safe(
    bhsd: &MlxArray,
    rope_dims: i32,
    rope_base: Option<f32>,
    token_offset: i32,
    rope_freqs: Option<&MlxArray>,
) -> MlxArray {
    let shape = bhsd.shape();
    if qwen_prefill_reuse_rope_active() {
        return apply_reused_neox_rope(bhsd, rope_dims, rope_base, token_offset, rope_freqs);
    }
    if let [batch, heads, seq, head_dim] = shape[..]
        && batch > 1
        && seq == 1
        && token_offset > 0
    {
        let folded = reshape(bhsd, &[1, batch * heads, seq, head_dim], None);
        let roped = rope(
            &folded,
            rope_dims,
            false,
            rope_base,
            1.0,
            token_offset,
            rope_freqs,
            None,
        );
        return reshape(&roped, &[batch, heads, seq, head_dim], None);
    }
    rope(
        bhsd,
        rope_dims,
        false,
        rope_base,
        1.0,
        token_offset,
        rope_freqs,
        None,
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
    rope_bhsd_batch_offset_safe(
        &bhsd,
        rope_dims as i32,
        rope_base,
        token_offset as i32,
        rope_freqs,
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
    // The fused C++ route calls MLX rope directly, so it inherits the
    // MLX <= 0.31.x batched single-position offset bug; keep the buggy shape
    // on the composed path where rope_bhsd_batch_offset_safe applies.
    let batched_single_pos_offset = qw_out.shape()[0] > 1 && seq == 1 && token_offset > 0;
    // Reuse one cos/sin table across full-attn layers: skip the fused C++
    // rope so the portable apply path can share the cached trig.
    if qwen_prefill_reuse_rope_active() {
        let q = qk_norm_bhsd_from_proj(qw_out, norm, n_heads, head_dim, seq, eps);
        return apply_reused_neox_rope(
            &q,
            rope_dims as i32,
            rope_base,
            token_offset as i32,
            rope_freqs,
        );
    }
    if direct_qk_norm_rope_route_allowed(direct_route_enabled, norm) && !batched_single_pos_offset {
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
    rope_bhsd_batch_offset_safe(
        &q,
        rope_dims as i32,
        rope_base,
        token_offset as i32,
        rope_freqs,
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

/// Slice BHSD Q `[B, H, S, D]` to the last query so last-only generate
/// prefill can SDPA at S=1 after full K/V append.
pub(crate) fn qwen_prefill_maybe_last_query_q(
    q_rope: &MlxArray,
    last_query: bool,
) -> Option<MlxArray> {
    qwen_prefill_maybe_last_query_q_for(q_rope, last_query)
}

/// Pure helper for [`qwen_prefill_maybe_last_query_q`].
pub(crate) fn qwen_prefill_maybe_last_query_q_for(
    q_rope: &MlxArray,
    last_query: bool,
) -> Option<MlxArray> {
    if !last_query {
        return None;
    }
    let shape = q_rope.shape();
    if shape.len() != 4 {
        return None;
    }
    let seq = shape[2];
    if seq <= 1 {
        return None;
    }
    let last = seq - 1;
    let sliced = slice(
        q_rope,
        &[0, 0, last, 0],
        &[shape[0], shape[1], last + 1, shape[3]],
        &[1, 1, 1, 1],
        None,
    );
    Some(contiguous(&sliced, None))
}

/// BHSD query length. Last-token Q proj can shrink Q independently of the
/// last-query-SDPA flag; SDPA must use this, not the full-seq `seq`.
pub(crate) fn qwen_prefill_query_seq(q_rope: &MlxArray, fallback: usize) -> usize {
    qwen_prefill_query_seq_for(q_rope, fallback)
}

/// Pure helper for [`qwen_prefill_query_seq`].
pub(crate) fn qwen_prefill_query_seq_for(q_rope: &MlxArray, fallback: usize) -> usize {
    q_rope
        .shape()
        .get(2)
        .copied()
        .filter(|&seq| seq > 0)
        .map(|seq| seq as usize)
        .unwrap_or(fallback)
}

/// Slice `[B, S, H]` activation to the last token so last-only generate
/// can `q_proj` at S=1 after full K/V are written from the full sequence.
pub(crate) fn qwen_prefill_maybe_last_token_bsh(
    x: &MlxArray,
    last_token: bool,
) -> Option<MlxArray> {
    qwen_prefill_maybe_last_token_bsh_for(x, last_token)
}

/// Pure helper for [`qwen_prefill_maybe_last_token_bsh`].
pub(crate) fn qwen_prefill_maybe_last_token_bsh_for(
    x: &MlxArray,
    last_token: bool,
) -> Option<MlxArray> {
    if !last_token {
        return None;
    }
    let shape = x.shape();
    if shape.len() != 3 {
        return None;
    }
    let seq = shape[1];
    if seq <= 1 {
        return None;
    }
    let last = seq - 1;
    let sliced = slice(
        x,
        &[0, last, 0],
        &[shape[0], last + 1, shape[2]],
        &[1, 1, 1],
        None,
    );
    Some(contiguous(&sliced, None))
}

/// Slice flattened `[B, S, H]` attention output to the last token so
/// last-only generate prefill can o_proj at S=1. KV append has already
/// happened. Operates after flatten so the last row is contiguous.
pub(crate) fn qwen_prefill_maybe_last_token_flat(
    attn_flat: &MlxArray,
    last_token_out_proj: bool,
) -> MlxArray {
    let shape = attn_flat.shape();
    let seq = shape.get(1).copied().unwrap_or(1);
    if !last_token_out_proj || seq <= 1 {
        return attn_flat.clone();
    }
    let last = seq - 1;
    slice(
        attn_flat,
        &[0, last, 0],
        &[shape[0], last + 1, shape[2]],
        &[1, 1, 1],
        None,
    )
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
    // Full-attention multi-token with KV offset: either materialize an array
    // causal mask (legacy) or use MLX native `mask="causal"` (mlxcel parity).
    // Steel kernels apply `qL_off = key_len - seq_len`, matching
    // `create_causal_mask(seq, offset, None)` without the O(seq×key) array.
    if offset > 0 && seq_len > 1 {
        if crate::fastpath::native_offset_causal_enabled()
            || super::utils::qwen_prefill_native_offset_causal_active()
        {
            return None;
        }
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
    full_precision_attention_with_window(
        q_rope,
        cached_k,
        cached_v,
        query_scale,
        seq,
        mask_opt,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn full_precision_attention_with_window(
    q_rope: &MlxArray,
    cached_k: &MlxArray,
    cached_v: &MlxArray,
    query_scale: f32,
    seq: usize,
    mask_opt: &Option<MlxArray>,
    sliding_window: Option<usize>,
    ring_layout: Option<SlidingRingLayout>,
) -> MlxArray {
    let mask = match mask_opt.as_ref() {
        Some(mask) => ScaledDotProductAttentionMask::Array(mask),
        None if seq > 1 => ScaledDotProductAttentionMask::Causal,
        None => ScaledDotProductAttentionMask::None,
    };
    // Multi-token teacher-forced verify (seq > 1) accumulates bf16 SDPA drift
    // vs singleton pure-direct decode; near-ties (Gemma period-6 cycle break)
    // flip argmax. Upcast Q/K/V for the SDPA, then restore the input dtype.
    // Keep seq==1 on bf16: enabling f32 for pure-direct too regressed formal
    // 12B6 general exactness (smokef11).
    // MoE multi-token: per-query-position bf16 SDPA (hist-sliced K/V). Needed
    // with per-pos FFN for gen exactness (smokef51); dual-edge keep_start fixes
    // successive long-prefill drift (rows 1+ must not attend expired left keys).
    if fastpath::moe_mt_bf16_identity_enabled() {
        if seq > 1 && seq <= 8 {
            use mlx_sys::{concatenate, slice};
            let q_shape = q_rope.shape();
            let b = q_shape.first().copied().unwrap_or(1);
            let hq = q_shape.get(1).copied().unwrap_or(1);
            let d = *q_shape.last().unwrap_or(&1);
            let k_shape = cached_k.shape();
            let hk = k_shape.get(1).copied().unwrap_or(hq);
            let key_len = k_shape.get(2).copied().unwrap_or(1) as usize;
            let hist = key_len.saturating_sub(seq);
            // Fold query positions into the batch dimension so MLX dispatches
            // its singleton-query reduction for every row in one call. A
            // native q_len=seq dispatch uses a different bf16 reduction and
            // accumulates small KV/residual drift; the formal Gemma 26B long
            // loop eventually flips an EOS near-tie. Separate singleton calls
            // are exact but lose the multi-token speedup.
            let generated_mask;
            debug_assert!(ring_layout.is_none() || mask_opt.is_some());
            let position_mask = if let Some(mask) = mask_opt.as_ref() {
                mask
            } else {
                generated_mask = create_causal_mask(seq, hist, sliding_window);
                &generated_mask
            };
            if b == 1
                && k_shape.first().copied() == Some(1)
                && position_mask.shape() == vec![seq as i32, key_len as i32]
            {
                let q_singletons = reshape(
                    &transpose(q_rope, &[0, 2, 1, 3], None),
                    &[seq as i32, hq, 1, d],
                    None,
                );
                let k_singletons =
                    broadcast_to(cached_k, &[seq as i32, hk, key_len as i32, d], None);
                let v_singletons =
                    broadcast_to(cached_v, &[seq as i32, hk, key_len as i32, d], None);
                let mask_singletons =
                    reshape(position_mask, &[seq as i32, 1, 1, key_len as i32], None);
                let out_singletons = scaled_dot_product_attention_with_mask(
                    &q_singletons,
                    &k_singletons,
                    &v_singletons,
                    query_scale,
                    ScaledDotProductAttentionMask::Array(&mask_singletons),
                    None,
                );
                return transpose(
                    &reshape(&out_singletons, &[1, seq as i32, hq, d], None),
                    &[0, 2, 1, 3],
                    None,
                );
            }
            let mut rows: Vec<MlxArray> = Vec::with_capacity(seq);
            for t_idx in 0..seq {
                let q_t = slice(
                    q_rope,
                    &[0, 0, t_idx as i32, 0],
                    &[b, hq, (t_idx + 1) as i32, d],
                    &[1, 1, 1, 1],
                    None,
                );
                let allow = hist + t_idx + 1;
                // A retained multi-token view has `window + seq - 1` keys.
                // Move both edges for each query; moving only the causal
                // right edge lets later rows attend expired left-edge keys.
                let keep_start = sliding_window
                    .filter(|window| *window > 0)
                    .map(|window| allow.saturating_sub(window))
                    .unwrap_or(0) as i32;
                let allow = allow as i32;
                let k_t = slice(
                    cached_k,
                    &[0, 0, keep_start, 0],
                    &[b, hk, allow, d],
                    &[1, 1, 1, 1],
                    None,
                );
                let v_t = slice(
                    cached_v,
                    &[0, 0, keep_start, 0],
                    &[b, hk, allow, d],
                    &[1, 1, 1, 1],
                    None,
                );
                let out_t = scaled_dot_product_attention_with_mask(
                    &q_t,
                    &k_t,
                    &v_t,
                    query_scale,
                    ScaledDotProductAttentionMask::None,
                    None,
                );
                rows.push(out_t);
            }
            let refs: Vec<&MlxArray> = rows.iter().collect();
            return concatenate(&refs, 2, None);
        }
        return scaled_dot_product_attention_with_mask(
            q_rope,
            cached_k,
            cached_v,
            query_scale,
            mask,
            None,
        );
    }
    // Dense multi-token long history: bf16 singleton-query fold (mirror moe_mt).
    // Fold is required for q_len=1 reduction identity (12B6 agent first_diff@8);
    // doing it under full-history f32 K/V upcast was exact but too slow
    // (dense-sing-v4 agent weighted ~0.91×). MoE long already proves bf16 fold
    // keeps greedy exactness. Short multi-token stays on the f32 batched path
    // below (12B6 general release_ready; always-on fold regressed its weighted
    // 1.215→1.180). Kill-switch: AX_MLX_DENSE_LONG_MT_BF16_FOLD=0 restores the
    // prior f32 long fold.
    if fastpath::multi_token_f32_attention_enabled()
        && seq > 1
        && seq <= 8
        && fastpath::dense_long_mt_bf16_fold_enabled()
    {
        let q_shape = q_rope.shape();
        let b = q_shape.first().copied().unwrap_or(1);
        let hq = q_shape.get(1).copied().unwrap_or(1);
        let d = *q_shape.last().unwrap_or(&1);
        let k_shape = cached_k.shape();
        let hk = k_shape.get(1).copied().unwrap_or(hq);
        let key_len = k_shape.get(2).copied().unwrap_or(1) as usize;
        let hist = key_len.saturating_sub(seq);
        if key_len >= 512 {
            let generated_mask;
            let position_mask = if let Some(m) = mask_opt.as_ref() {
                m
            } else {
                generated_mask = create_causal_mask(seq, hist, sliding_window);
                &generated_mask
            };
            if b == 1
                && k_shape.first().copied() == Some(1)
                && position_mask.shape() == vec![seq as i32, key_len as i32]
            {
                let q_singletons = reshape(
                    &transpose(q_rope, &[0, 2, 1, 3], None),
                    &[seq as i32, hq, 1, d],
                    None,
                );
                let k_singletons =
                    broadcast_to(cached_k, &[seq as i32, hk, key_len as i32, d], None);
                let v_singletons =
                    broadcast_to(cached_v, &[seq as i32, hk, key_len as i32, d], None);
                let mask_singletons =
                    reshape(position_mask, &[seq as i32, 1, 1, key_len as i32], None);
                let out_singletons = scaled_dot_product_attention_with_mask(
                    &q_singletons,
                    &k_singletons,
                    &v_singletons,
                    query_scale,
                    ScaledDotProductAttentionMask::Array(&mask_singletons),
                    None,
                );
                return transpose(
                    &reshape(&out_singletons, &[1, seq as i32, hq, d], None),
                    &[0, 2, 1, 3],
                    None,
                );
            }
        }
    }
    // Keep seq==1 on bf16: enabling f32 for pure-direct regressed formal
    // 12B6 general exactness. Multi-token (seq > 1) uses f32 below (or the
    // dense long bf16 fold above when eligible). Qwen 27B prefill can skip
    // this Gemma-verify upcast via `AX_MLX_QWEN_PREFILL_SKIP_UNUSED_F32_SDPA`.
    if should_upcast_multi_token_sdpa_to_f32(seq) {
        let q_dtype = q_rope.dtype();
        let q = if q_dtype != MlxDtype::Float32 {
            astype(q_rope, MlxDtype::Float32, None)
        } else {
            q_rope.clone()
        };
        let k = if cached_k.dtype() != MlxDtype::Float32 {
            astype(cached_k, MlxDtype::Float32, None)
        } else {
            cached_k.clone()
        };
        let v = if cached_v.dtype() != MlxDtype::Float32 {
            astype(cached_v, MlxDtype::Float32, None)
        } else {
            cached_v.clone()
        };
        // Dense multi-token short / fallback: f32 batched SDPA (exact for short
        // gen). Long history normally takes the bf16 fold above; when that
        // path is kill-switched, fall through to f32 fold for identity.
        if seq > 1 && seq <= 8 {
            let q_shape = q.shape();
            let b = q_shape.first().copied().unwrap_or(1);
            let hq = q_shape.get(1).copied().unwrap_or(1);
            let d = *q_shape.last().unwrap_or(&1);
            let k_shape = k.shape();
            let hk = k_shape.get(1).copied().unwrap_or(hq);
            let key_len = k_shape.get(2).copied().unwrap_or(1) as usize;
            let hist = key_len.saturating_sub(seq);
            if key_len >= 512 {
                let generated_mask;
                let position_mask = if let Some(m) = mask_opt.as_ref() {
                    m
                } else {
                    generated_mask = create_causal_mask(seq, hist, sliding_window);
                    &generated_mask
                };
                if b == 1
                    && k_shape.first().copied() == Some(1)
                    && position_mask.shape() == vec![seq as i32, key_len as i32]
                {
                    let q_singletons = reshape(
                        &transpose(&q, &[0, 2, 1, 3], None),
                        &[seq as i32, hq, 1, d],
                        None,
                    );
                    let k_singletons = broadcast_to(&k, &[seq as i32, hk, key_len as i32, d], None);
                    let v_singletons = broadcast_to(&v, &[seq as i32, hk, key_len as i32, d], None);
                    let mask_singletons =
                        reshape(position_mask, &[seq as i32, 1, 1, key_len as i32], None);
                    let out_singletons = scaled_dot_product_attention_with_mask(
                        &q_singletons,
                        &k_singletons,
                        &v_singletons,
                        query_scale,
                        ScaledDotProductAttentionMask::Array(&mask_singletons),
                        None,
                    );
                    let out = transpose(
                        &reshape(&out_singletons, &[1, seq as i32, hq, d], None),
                        &[0, 2, 1, 3],
                        None,
                    );
                    if q_dtype != MlxDtype::Float32 {
                        return astype(&out, q_dtype, None);
                    }
                    return out;
                }
            }
        }
        let out = scaled_dot_product_attention_with_mask(&q, &k, &v, query_scale, mask, None);
        if q_dtype != MlxDtype::Float32 {
            return astype(&out, q_dtype, None);
        }
        return out;
    }
    scaled_dot_product_attention_with_mask(q_rope, cached_k, cached_v, query_scale, mask, None)
}

pub(crate) fn should_upcast_multi_token_sdpa_to_f32(seq: usize) -> bool {
    fastpath::multi_token_f32_attention_enabled()
        && seq > 1
        && !super::utils::qwen_prefill_skip_f32_sdpa_active()
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
    // MLX fast SDPA supports sinks natively (fused kernel, native GQA
    // broadcast). The previous hand-rolled score-matrix implementation could
    // not broadcast grouped queries against fewer KV heads (64 q-heads vs
    // 8 kv-heads is not a broadcastable batch shape for `matmul`), so it
    // aborted on every GQA sinks model; it also materialized the full
    // `[B, H, S, K+1]` score/penalty tensors in f32. Mask mapping mirrors
    // `full_precision_attention`.
    let mask = match mask_opt.as_ref() {
        Some(mask) => ScaledDotProductAttentionMask::Array(mask),
        None if seq > 1 => ScaledDotProductAttentionMask::Causal,
        None => ScaledDotProductAttentionMask::None,
    };
    // MLX's fused SDPA requires sinks to promote to the output dtype;
    // checkpoints may store sinks in f32 (e.g. DeepSeek V4) while q/k/v run
    // in bf16/f16, so cast at the call site.
    let sinks = astype(sinks, q.dtype(), None);
    scaled_dot_product_attention_with_mask_and_sinks(q, k, v, query_scale, mask, Some(&sinks), None)
}

/// Unfused reference for [`attention_with_sinks`], kept as the test oracle:
/// explicit score matrix, sink column appended before softmax and excluded
/// from the value weighted sum. K/V are expanded per query-head group so GQA
/// shapes evaluate exactly.
#[cfg(test)]
pub(crate) fn attention_with_sinks_reference(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    sinks: &MlxArray,
    query_scale: f32,
    seq: usize,
    mask_opt: &Option<MlxArray>,
) -> MlxArray {
    let n_q_heads = q.shape()[1];
    let n_kv_heads = k.shape()[1];
    let (k, v) = if n_q_heads != n_kv_heads {
        (
            repeat_kv_heads(k, n_q_heads, n_kv_heads),
            repeat_kv_heads(v, n_q_heads, n_kv_heads),
        )
    } else {
        (k.clone(), v.clone())
    };

    // scores = Q @ K^T * scale → [B, n_q_heads, seq, key_len]
    let k_t = transpose(&k, &[0, 1, 3, 2], None);
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
        // Masks arrive as [seq, key_len] (2D, SDPA-broadcast) — pad the last
        // axis, wherever it is, for the always-visible sink column.
        let last_axis = mask.shape().len() as i32 - 1;
        let extended_mask = mlx_sys::pad(mask, &[last_axis], &[0], &[1], &true_val, None);
        let neg_inf = scalar_array(f32::NEG_INFINITY);
        let zero = scalar_array(0.0);
        let penalty = mlx_sys::where_cond(&extended_mask, &zero, &neg_inf, None);
        mlx_sys::add(&scores_with_sink, &penalty, None)
    } else if seq > 1 {
        let key_len = k.shape()[2] as usize;
        let causal = create_causal_mask(seq, key_len.saturating_sub(seq), None);
        let last_axis = causal.shape().len() as i32 - 1;
        let extended_mask = mlx_sys::pad(&causal, &[last_axis], &[0], &[1], &true_val, None);
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

    matmul(&weights_real, &v, None)
}

/// Expand `[B, n_kv_heads, K, D]` K/V to `[B, n_q_heads, K, D]` by repeating
/// each KV head over its contiguous query-head group (standard GQA mapping:
/// query head `h` reads KV head `h / (n_q_heads / n_kv_heads)`).
#[cfg(test)]
fn repeat_kv_heads(kv: &MlxArray, n_q_heads: i32, n_kv_heads: i32) -> MlxArray {
    assert!(n_q_heads % n_kv_heads == 0, "GQA requires divisible heads");
    let group = n_q_heads / n_kv_heads;
    let shape = kv.shape();
    let (batch, key_len, head_dim) = (shape[0], shape[2], shape[3]);
    let expanded = broadcast_to(
        &reshape(kv, &[batch, n_kv_heads, 1, key_len, head_dim], None),
        &[batch, n_kv_heads, group, key_len, head_dim],
        None,
    );
    reshape(&expanded, &[batch, n_q_heads, key_len, head_dim], None)
}

/// Create a scalar MlxArray from a single f32 value.
#[cfg(test)]
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
    //
    // Buffer path materializes (`contiguous` + `eval`) after every write so:
    // 1. the stored buffer is dense and bit-aligned with a fresh concatenate,
    // 2. lazy slice_update chains do not deepen across denoise steps (which
    //    previously produced non-bit-equivalent results vs re-concatenate).
    let (full_k, full_v) = if let Some(ref mut buf) = kv_buffer {
        if buf.full_k.is_none() {
            // First step: build buffer via concatenate, then densify.
            let fk = contiguous(&concatenate(&[cached_k, canvas_k], 2, None), None);
            let fv = contiguous(&concatenate(&[cached_v, canvas_v], 2, None), None);
            eval(&[&fk, &fv]);
            buf.cached_seq = cached_k.shape()[2] as usize;
            buf.full_k = Some(fk.clone());
            buf.full_v = Some(fv.clone());
            (fk, fv)
        } else {
            // Subsequent steps: update only the canvas slice, then densify.
            let total = buf.cached_seq + canvas_size;
            let start = [0, 0, buf.cached_seq as i32, 0];
            let stop = [
                canvas_k.shape()[0],
                canvas_k.shape()[1],
                total as i32,
                canvas_k.shape()[3],
            ];
            let strides = [1, 1, 1, 1];
            let fk = contiguous(
                &slice_update(
                    buf.full_k.as_ref().unwrap(),
                    canvas_k,
                    &start,
                    &stop,
                    &strides,
                    None,
                ),
                None,
            );
            let fv = contiguous(
                &slice_update(
                    buf.full_v.as_ref().unwrap(),
                    canvas_v,
                    &start,
                    &stop,
                    &strides,
                    None,
                ),
                None,
            );
            eval(&[&fk, &fv]);
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
            // Reuse only when the cached-prefix length still matches: a
            // buffer carried across blocks (cached_seq grows as blocks
            // commit) would otherwise apply a stale, wrong-width mask.
            match buf.cached_mask.as_ref() {
                Some(mask) if buf.cached_mask_seq == cached_seq => mask.clone(),
                _ => {
                    let m = build_bidirectional_canvas_mask(canvas_size, cached_seq, window);
                    buf.cached_mask = Some(m.clone());
                    buf.cached_mask_seq = cached_seq;
                    m
                }
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
            .enumerate()
            .map(|(layer_idx, lc)| {
                if crate::fastpath::should_skip_linear_prefill_mask(
                    &cfg.model_family,
                    cfg.is_linear_attention_layer(layer_idx),
                ) {
                    return None;
                }
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

    // When ring layout engages, append returns capacity-wide K/V (including
    // cold ring init). Hoist capacity masks here — do not use logical
    // `token_offset + seq` as key_len for ring layers.
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
                .or_insert_with(|| match cache.sliding_ring_layout(lc.sliding_window, seq) {
                    Some(ring) if ring.needs_mask(seq) => Some(create_ring_sliding_mask(
                        seq,
                        ring.window,
                        ring.capacity,
                        ring.write_start,
                    )),
                    Some(_) => None,
                    None => {
                        if seq == 1 {
                            return None;
                        }
                        let mask_key_len = attention_mask_key_len(seq, key_len, lc.sliding_window);
                        attention_mask_array(seq, mask_key_len, lc.sliding_window)
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
        apply_reused_neox_rope, build_bidirectional_canvas_mask,
        build_layer_masks_with_media_ranges, full_precision_attention,
        gemma4_prefill_maybe_async_first_kv_for, media_prefix_mask_array,
        qwen_direct_qk_norm_rope_default_family, qwen_prefill_maybe_eval_attn_input_for,
        qwen_prefill_maybe_last_query_q_for, qwen_prefill_maybe_last_token_bsh_for,
        qwen_prefill_maybe_last_token_flat, qwen_prefill_query_seq_for,
        set_qwen_prefill_reuse_rope_active, should_upcast_multi_token_sdpa_to_f32,
    };
    use crate::model::shared::utils::{
        QwenPrefillSkipF32SdpaGuard, qwen_prefill_skip_f32_sdpa_active,
    };
    use crate::model::{LayerConfig, ModelConfig};
    use mlx_sys::{
        MlxArray, MlxDtype, ScaledDotProductAttentionMask, astype, eval,
        scaled_dot_product_attention_with_mask,
    };

    fn mask_data(mask: &MlxArray) -> Vec<u8> {
        eval(&[mask]);
        let len = mask.nbytes();
        let ptr = mask.data_raw();
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    #[test]
    fn qwen_prefill_native_offset_causal_skips_array_mask() {
        use crate::model::shared::utils::QwenPrefillNativeOffsetCausalGuard;
        assert!(
            crate::fastpath::should_qwen_prefill_native_offset_causal_for(true, "qwen3_5", 1024),
            "shipped native-offset-causal gate must accept the p2048 chunk length"
        );
        let _g = QwenPrefillNativeOffsetCausalGuard::arm(true);
        let mask = super::attention_mask_array(1024, 2048, None);
        assert!(
            mask.is_none(),
            "offset-1024 Qwen prefill must use native causal, not an O(seq×key) array"
        );
    }

    #[test]
    fn gemma4_prefill_skip_unused_f32_sdpa_disables_upcast_on_contract_p128() {
        use crate::model::shared::utils::QwenPrefillSkipF32SdpaGuard;
        assert!(
            crate::fastpath::should_gemma4_prefill_skip_unused_f32_sdpa_for(true, "gemma4", 128),
            "shipped Gemma 4 skip-f32-sdpa gate must accept contract p128"
        );
        assert!(
            super::should_upcast_multi_token_sdpa_to_f32(128),
            "without the prefill guard, contract p128 still pays the default-ON f32 upcast"
        );
        let _g = QwenPrefillSkipF32SdpaGuard::arm(true);
        assert!(
            !super::should_upcast_multi_token_sdpa_to_f32(128),
            "Gemma 4 p128 prefill must keep SDPA in the model dtype"
        );
        assert!(
            !super::should_upcast_multi_token_sdpa_to_f32(1),
            "decode seq==1 never upcasts"
        );
    }

    #[test]
    fn qwen_prefill_skip_unused_f32_sdpa_matches_model_dtype_sdpa() {
        assert!(
            crate::fastpath::should_qwen_prefill_skip_unused_f32_sdpa_for(true, "qwen3_5", 1024),
            "shipped skip-f32-sdpa gate must accept the p2048 chunk length"
        );
        let q_data = [0.0_f32, 0.0, 1.0, 0.0];
        let k_data = [0.0_f32, 0.0, 1.0, 0.0];
        let v_data = [1.0_f32, 3.0, 5.0, 7.0];
        let q_f32 = MlxArray::from_raw_data(
            q_data.as_ptr().cast(),
            std::mem::size_of_val(&q_data),
            &[1, 1, 2, 2],
            MlxDtype::Float32,
        );
        let k_f32 = MlxArray::from_raw_data(
            k_data.as_ptr().cast(),
            std::mem::size_of_val(&k_data),
            &[1, 1, 2, 2],
            MlxDtype::Float32,
        );
        let v_f32 = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            std::mem::size_of_val(&v_data),
            &[1, 1, 2, 2],
            MlxDtype::Float32,
        );
        let q = astype(&q_f32, MlxDtype::Bfloat16, None);
        let k = astype(&k_f32, MlxDtype::Bfloat16, None);
        let v = astype(&v_f32, MlxDtype::Bfloat16, None);
        let _skip = QwenPrefillSkipF32SdpaGuard::arm(true);
        assert!(qwen_prefill_skip_f32_sdpa_active());
        assert!(!should_upcast_multi_token_sdpa_to_f32(1024));
        let skipped = full_precision_attention(&q, &k, &v, 1.0, 2, &None);
        let native = scaled_dot_product_attention_with_mask(
            &q,
            &k,
            &v,
            1.0,
            ScaledDotProductAttentionMask::Causal,
            None,
        );
        eval(&[&skipped, &native]);
        assert_eq!(skipped.shape(), native.shape());
        assert_eq!(skipped.dtype(), MlxDtype::Bfloat16);
        let skipped_f32 = astype(&skipped, MlxDtype::Float32, None);
        let native_f32 = astype(&native, MlxDtype::Float32, None);
        eval(&[&skipped_f32, &native_f32]);
        let left = skipped_f32.data_f32();
        let right = native_f32.data_f32();
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right.iter()) {
            assert!(
                (a - b).abs() < 1.0e-5,
                "skip-f32-sdpa must match model-dtype causal SDPA: {a} vs {b}"
            );
        }
    }

    #[test]
    fn qwen_prefill_maybe_eval_attn_input_materializes_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let x = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 32, 1],
            mlx_sys::MlxDtype::Float32,
        );
        qwen_prefill_maybe_eval_attn_input_for(&x, true, "qwen3_5", 1024);
        eval(&[&x]);
        assert_eq!(x.shape(), vec![1, 32, 1]);
        assert!(
            x.data_f32().iter().all(|v| v.is_finite()),
            "eval-attn-input must leave a finite materialized activation"
        );
        assert!(
            crate::fastpath::should_qwen_prefill_eval_attn_input_for(true, "qwen3_5", 1024),
            "shipped attn input-eval gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_eval_attn_input_for(&x, false, "qwen3_5", 1024);
        qwen_prefill_maybe_eval_attn_input_for(&x, true, "gemma4", 1024);
        qwen_prefill_maybe_eval_attn_input_for(&x, true, "qwen3_5", 512);
    }

    #[test]
    fn qwen_prefill_maybe_async_sdpa_submits_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let sdpa = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 2, 4, 4],
            mlx_sys::MlxDtype::Float32,
        );
        super::qwen_prefill_maybe_async_sdpa_for(&sdpa, true, "qwen3_5", 1024);
        eval(&[&sdpa]);
        assert_eq!(sdpa.shape(), vec![1, 2, 4, 4]);
        assert!(
            sdpa.data_f32().iter().all(|v| v.is_finite()),
            "async SDPA must leave a finite materialized tensor"
        );
        assert!(
            crate::fastpath::should_qwen_prefill_async_sdpa_for(true, "qwen3_5", 1024),
            "shipped async-SDPA gate must accept the p2048 chunk length"
        );
        super::qwen_prefill_maybe_async_sdpa_for(&sdpa, false, "qwen3_5", 1024);
        super::qwen_prefill_maybe_async_sdpa_for(&sdpa, true, "gemma4", 1024);
        super::qwen_prefill_maybe_async_sdpa_for(&sdpa, true, "qwen3_5", 512);
    }

    #[test]
    fn gemma4_prefill_maybe_async_first_kv_submits_when_gated() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let k = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 2, 2, 4],
            mlx_sys::MlxDtype::Float32,
        );
        let v = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 2, 2, 4],
            mlx_sys::MlxDtype::Float32,
        );
        assert!(
            crate::fastpath::should_gemma4_async_first_kv_p128_for(true, "gemma4", 128),
            "shipped first-KV async submit must accept contract p128"
        );
        gemma4_prefill_maybe_async_first_kv_for(&k, &v, false);
        gemma4_prefill_maybe_async_first_kv_for(&k, &v, true);
        eval(&[&k, &v]);
        assert_eq!(k.shape(), vec![1, 2, 2, 4]);
        assert_eq!(v.shape(), vec![1, 2, 2, 4]);
    }

    #[test]
    fn qwen_prefill_query_seq_reads_bhsd_dim() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let full = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 2, 4, 2],
            mlx_sys::MlxDtype::Float32,
        );
        assert_eq!(qwen_prefill_query_seq_for(&full, 99), 4);
        let last =
            qwen_prefill_maybe_last_query_q_for(&full, true).expect("last-query slice must engage");
        eval(&[&last]);
        assert_eq!(qwen_prefill_query_seq_for(&last, 99), 1);
        assert_eq!(qwen_prefill_query_seq_for(&full, 0), 4);
    }

    #[test]
    fn qwen_prefill_maybe_last_token_bsh_slices_when_set() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let x = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 4, 2],
            mlx_sys::MlxDtype::Float32,
        );
        let sliced = qwen_prefill_maybe_last_token_bsh_for(&x, true)
            .expect("last-token BSH slice must engage at S=4");
        eval(&[&sliced]);
        assert_eq!(sliced.shape(), vec![1, 1, 2]);
        assert_eq!(sliced.data_f32(), vec![6.0, 7.0]);
        assert!(qwen_prefill_maybe_last_token_bsh_for(&x, false).is_none());
        assert!(
            crate::fastpath::should_qwen_prefill_last_query_q_proj_for(true, "qwen3_5", true, 1024),
            "shipped last-query Q proj must accept the p2048 generate last layer"
        );
        assert!(
            crate::fastpath::should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 128),
            "shipped Gemma 4 last-query must accept contract p128 last layer"
        );
        assert!(
            crate::fastpath::should_qwen_prefill_skip_unused_qk_norm_for(
                true, "qwen3_5", true, 1024
            ),
            "shipped skip-unused-QK-norm must accept the p2048 generate last layer"
        );
    }

    #[test]
    fn qwen_prefill_maybe_last_query_q_slices_bhsd_when_set() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let q = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 2, 4, 2],
            mlx_sys::MlxDtype::Float32,
        );
        let sliced = qwen_prefill_maybe_last_query_q_for(&q, true)
            .expect("last-query slice must engage at S=4");
        eval(&[&sliced]);
        assert_eq!(sliced.shape(), vec![1, 2, 1, 2]);
        assert_eq!(sliced.data_f32(), vec![6.0, 7.0, 14.0, 15.0]);
        assert!(qwen_prefill_maybe_last_query_q_for(&q, false).is_none());
        assert!(
            crate::fastpath::should_qwen_prefill_last_query_sdpa_for(true, "qwen3_5", true, 1024),
            "shipped last-query SDPA must accept the p2048 generate last layer"
        );
        assert!(
            crate::fastpath::should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 128),
            "shipped Gemma 4 last-query SDPA must accept contract p128 last layer"
        );
    }

    #[test]
    fn qwen_prefill_maybe_last_token_flat_slices_seq_when_set() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let attn = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 4, 4],
            mlx_sys::MlxDtype::Float32,
        );
        let sliced = qwen_prefill_maybe_last_token_flat(&attn, true);
        eval(&[&sliced]);
        assert_eq!(sliced.shape(), vec![1, 1, 4]);
        assert_eq!(sliced.data_f32(), vec![12.0, 13.0, 14.0, 15.0]);
        let kept = qwen_prefill_maybe_last_token_flat(&attn, false);
        eval(&[&kept]);
        assert_eq!(kept.shape(), attn.shape());
        assert!(
            crate::fastpath::should_qwen_prefill_last_token_o_proj_for(true, "qwen3_5", true, 1024),
            "shipped last-token o_proj must accept the p2048 generate last layer"
        );
        assert!(
            crate::fastpath::should_gemma4_prefill_last_query_p128_for(true, "gemma4", true, 128),
            "shipped Gemma 4 last-token o_proj must accept contract p128 last layer"
        );
    }

    #[test]
    fn apply_reused_neox_rope_matches_mlx_fast_rope() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.03125).collect();
        let x = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 2, 4, 8],
            mlx_sys::MlxDtype::Float32,
        );
        set_qwen_prefill_reuse_rope_active(true);
        let reused = apply_reused_neox_rope(&x, 8, Some(10_000.0), 0, None);
        let reference = mlx_sys::rope(&x, 8, false, Some(10_000.0), 1.0, 0, None, None);
        eval(&[&reused, &reference]);
        set_qwen_prefill_reuse_rope_active(false);
        assert_eq!(reused.shape(), reference.shape());
        for (a, b) in reused.data_f32().iter().zip(reference.data_f32().iter()) {
            assert!(
                (a - b).abs() < 2e-4 || (a - b).abs() / (b.abs().max(1e-6)) < 2e-4,
                "reused NeoX rope must match mlx_fast_rope: {a} vs {b}"
            );
        }
        assert!(
            crate::fastpath::should_qwen_prefill_reuse_rope_for(true, "qwen3_5", 1024),
            "shipped rope reuse must accept the p2048 chunk length"
        );
    }

    #[test]
    fn apply_reused_neox_rope_matches_mlx_at_qwen36_27b_shape() {
        // Qwen 3.6 27B full-attn: 24 heads, head_dim 128, partial rotary 32.
        let seq = 32;
        let heads = 24;
        let head_dim = 128;
        let rope_dims = 32;
        let n = heads * seq * head_dim;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) - 64.0) * 0.0078125).collect();
        let x = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, heads as i32, seq as i32, head_dim as i32],
            mlx_sys::MlxDtype::Float32,
        );
        set_qwen_prefill_reuse_rope_active(true);
        let reused = apply_reused_neox_rope(&x, rope_dims, Some(10_000_000.0), 0, None);
        let reference = mlx_sys::rope(&x, rope_dims, false, Some(10_000_000.0), 1.0, 0, None, None);
        eval(&[&reused, &reference]);
        set_qwen_prefill_reuse_rope_active(false);
        assert_eq!(reused.shape(), reference.shape());
        for (a, b) in reused.data_f32().iter().zip(reference.data_f32().iter()) {
            assert!(
                (a - b).abs() < 5e-4 || (a - b).abs() / (b.abs().max(1e-6)) < 5e-4,
                "27B-shaped reused rope must match mlx_fast_rope: {a} vs {b}"
            );
        }
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
            compile_cache_identity: 2,
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
            protected_prefix_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: true,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            deepseek_v4: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
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
            generation_kind: ax_engine_core::GenerationKind::Autoregressive,
            kv_cache_quant: vec![None; 2],
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

    #[test]
    fn fused_sinks_attention_matches_reference_under_gqa() {
        // GPT-OSS shape class: grouped queries (8 q-heads over 2 kv-heads).
        // The fused path must agree with the unfused reference on every mask
        // mode the gpt_oss forward can produce: single-token decode (no
        // mask), offset-causal prefill (None + seq > 1), and an explicit
        // sliding mask.
        use super::{attention_mask_array, attention_with_sinks, attention_with_sinks_reference};
        use mlx_sys::{astype, reshape};

        let (n_heads, kv_heads, head_dim) = (8usize, 2usize, 4usize);
        let fill = |n: usize, seed: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.29 + seed).sin()).collect()
        };
        let read_f32 = |arr: &MlxArray| -> Vec<f32> {
            let arr = astype(arr, mlx_sys::MlxDtype::Float32, None);
            eval(&[&arr]);
            let len = arr.nbytes() / std::mem::size_of::<f32>();
            let ptr = arr.data_raw() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
        };

        // (seq, key_len, sliding_window)
        for (seq, key_len, window) in [
            (1usize, 6usize, None),
            (1, 9, Some(4usize)),
            (5, 5, None),
            (5, 12, None),
            (5, 12, Some(4)),
        ] {
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
            let sinks = MlxArray::from_f32_slice(&fill(n_heads, 0.3));
            let mask = attention_mask_array(seq, key_len, window);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let fused = attention_with_sinks(&q, &k, &v, &sinks, scale, seq, &mask);
            let reference = attention_with_sinks_reference(&q, &k, &v, &sinks, scale, seq, &mask);
            let fused = read_f32(&fused);
            let reference = read_f32(&reference);
            assert_eq!(fused.len(), reference.len());
            let max_diff = fused
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-5,
                "fused sinks diverged from reference (seq {seq}, key_len {key_len}, \
                 window {window:?}): max diff {max_diff}"
            );
        }
    }

    /// Multi-step canvas KV updates via `KVConcatBuffer` must match re-concatenate.
    #[test]
    fn kv_concat_buffer_matches_reconcatenate_across_steps() {
        use super::{KVConcatBuffer, bidirectional_attention};
        use mlx_sys::{MlxDtype, reshape};

        let (batch, kv_heads, n_heads, head_dim) = (1usize, 1usize, 2usize, 4usize);
        let cached_seq = 8usize;
        let canvas = 4usize;
        let fill = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32 + 1.0) * 0.17 + seed).sin())
                .collect()
        };
        let shape_kv = |seq: usize| -> [i32; 4] {
            [batch as i32, kv_heads as i32, seq as i32, head_dim as i32]
        };
        let shape_q = |seq: usize| -> [i32; 4] {
            [batch as i32, n_heads as i32, seq as i32, head_dim as i32]
        };

        let cached_k = reshape(
            &MlxArray::from_f32_slice(&fill(kv_heads * cached_seq * head_dim, 0.2)),
            &shape_kv(cached_seq),
            None,
        );
        let cached_v = reshape(
            &MlxArray::from_f32_slice(&fill(kv_heads * cached_seq * head_dim, 0.4)),
            &shape_kv(cached_seq),
            None,
        );
        let q = reshape(
            &MlxArray::from_f32_slice(&fill(n_heads * canvas * head_dim, 0.1)),
            &shape_q(canvas),
            None,
        );

        let mut buf = KVConcatBuffer::new();
        let read_f32 = |arr: &MlxArray| -> Vec<f32> {
            let arr = mlx_sys::astype(arr, MlxDtype::Float32, None);
            eval(&[&arr]);
            let len = arr.nbytes() / std::mem::size_of::<f32>();
            let ptr = arr.data_raw() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
        };

        for step in 0..3 {
            let seed = 1.0 + step as f32 * 0.3;
            let canvas_k = reshape(
                &MlxArray::from_f32_slice(&fill(kv_heads * canvas * head_dim, seed)),
                &shape_kv(canvas),
                None,
            );
            let canvas_v = reshape(
                &MlxArray::from_f32_slice(&fill(kv_heads * canvas * head_dim, seed + 0.5)),
                &shape_kv(canvas),
                None,
            );

            let out_buf = bidirectional_attention(
                &q,
                &cached_k,
                &cached_v,
                &canvas_k,
                &canvas_v,
                1.0,
                None,
                Some(&mut buf),
            );
            let out_ref = bidirectional_attention(
                &q, &cached_k, &cached_v, &canvas_k, &canvas_v, 1.0, None, None,
            );

            let a = read_f32(&out_buf);
            let b = read_f32(&out_ref);
            assert_eq!(a.len(), b.len(), "step {step}: length mismatch");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-5,
                    "step {step} idx {i}: buffer={x} concat={y}"
                );
            }
        }
    }
}
