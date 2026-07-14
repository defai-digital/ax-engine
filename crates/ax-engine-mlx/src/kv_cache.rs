use std::time::Instant;

use ax_engine_core::{KvCompressionConfig, KvCompressionMode, TurboQuantPreset};
use mlx_sys::{MlxArray, MlxDtype, astype, contiguous, eval, slice, slice_update, zeros};

use crate::fastpath;
use crate::turboquant::{
    FullPrecisionKvTokenVectors, TurboQuantAttentionPartitionStats,
    TurboQuantAttentionPartitionStatsBatch, TurboQuantBlockLayout, TurboQuantBlockLayoutConfig,
    TurboQuantCodecError, TurboQuantCompressedBlockBuffer,
    TurboQuantCompressedBlockBufferMetaState, TurboQuantCompressedBlockBufferState,
    TurboQuantCompressedDecodePlan, TurboQuantKeyCentroidKind, TurboQuantKeyCodecConfig,
    append_attention_partition_stats_batch_head_with_hot_tail_from_f32_slices, packed_index_bytes,
    packed_kv_bytes_per_token, packed_value_bytes_for_preset, turboquant_query_head_to_kv_head,
};
use crate::turboquant_metal::{
    turboquant_fused_cold_decode_metal_two_stage_partition_stats_batch_with_compressed_array_flat_sparse_threshold,
    turboquant_fused_cold_decode_metal_two_stage_partition_stats_flat,
    turboquant_fused_cold_decode_metal_two_stage_sparse_partition_stats_flat,
    turboquant_fused_key_encode_metal_k8,
};

/// Pre-allocated chunk size (tokens).  The buffer grows by this amount each time
/// the logical sequence length exceeds capacity, so the number of grow operations
/// per session is at most ceil(total_tokens / CHUNK).
pub(crate) const KV_CHUNK_TOKENS: usize = 256;

fn elapsed_us_u64(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

fn chunk_ceiling(n: usize) -> usize {
    n.div_ceil(KV_CHUNK_TOKENS) * KV_CHUNK_TOKENS
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AppendShape {
    new_tokens: usize,
    n_kv_heads: i32,
    head_dim: i32,
    dtype: MlxDtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GlmMlaAppendShape {
    new_tokens: usize,
    latent_dim: i32,
    rope_dim: i32,
    dtype: MlxDtype,
}

fn validate_append_inputs(
    layer: usize,
    layer_count: usize,
    new_k: &MlxArray,
    new_v: &MlxArray,
) -> AppendShape {
    assert!(
        layer < layer_count,
        "KV cache layer {layer} out of bounds for {layer_count} layers"
    );

    let k_shape = new_k.shape();
    let v_shape = new_v.shape();
    assert_eq!(
        k_shape, v_shape,
        "KV cache append requires matching K/V shapes"
    );
    assert_eq!(
        k_shape.len(),
        4,
        "KV cache append expects [1, n_kv_heads, tokens, head_dim]"
    );
    assert_eq!(k_shape[0], 1, "KV cache append supports batch=1 only");
    assert!(
        k_shape[1] > 0 && k_shape[2] > 0 && k_shape[3] > 0,
        "KV cache append requires positive heads, tokens, and head_dim"
    );

    let k_dtype = new_k.dtype();
    assert_eq!(
        k_dtype,
        new_v.dtype(),
        "KV cache append requires matching K/V dtypes"
    );

    AppendShape {
        new_tokens: k_shape[2] as usize,
        n_kv_heads: k_shape[1],
        head_dim: k_shape[3],
        dtype: k_dtype,
    }
}

fn validate_glm_mla_append_inputs(
    layer: usize,
    layer_count: usize,
    new_kv_latent: &MlxArray,
    new_k_pe: &MlxArray,
) -> GlmMlaAppendShape {
    assert!(
        layer < layer_count,
        "GLM MLA cache layer {layer} out of bounds for {layer_count} layers"
    );

    let latent_shape = new_kv_latent.shape();
    let rope_shape = new_k_pe.shape();
    assert_eq!(
        latent_shape.len(),
        4,
        "GLM MLA latent cache append expects [1, 1, tokens, kv_lora_rank]"
    );
    assert_eq!(
        rope_shape.len(),
        4,
        "GLM MLA rope-key cache append expects [1, 1, tokens, qk_rope_head_dim]"
    );
    assert_eq!(latent_shape[0], 1, "GLM MLA cache supports batch=1 only");
    assert_eq!(rope_shape[0], 1, "GLM MLA cache supports batch=1 only");
    assert_eq!(
        latent_shape[1], 1,
        "GLM MLA latent cache stores one latent head"
    );
    assert_eq!(
        rope_shape[1], 1,
        "GLM MLA rope-key cache stores one RoPE head"
    );
    assert_eq!(
        latent_shape[2], rope_shape[2],
        "GLM MLA cache append requires matching latent/k_pe token counts"
    );
    assert!(
        latent_shape[2] > 0 && latent_shape[3] > 0 && rope_shape[3] > 0,
        "GLM MLA cache append requires positive tokens and dimensions"
    );

    let dtype = new_kv_latent.dtype();
    assert_eq!(
        dtype,
        new_k_pe.dtype(),
        "GLM MLA cache append requires matching latent/k_pe dtypes"
    );

    GlmMlaAppendShape {
        new_tokens: latent_shape[2] as usize,
        latent_dim: latent_shape[3],
        rope_dim: rope_shape[3],
        dtype,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KvHeadSliceShape {
    n_kv_heads: i32,
    head_dim: i32,
    capacity: usize,
}

#[cfg(test)]
fn kv_heads_for_token(lkv: &LayerKV, token_index: usize) -> Vec<FullPrecisionKvTokenVectors> {
    (0..lkv.n_kv_heads as usize)
        .map(|head_index| {
            (
                kv_vector_for_token_head(&lkv.k, lkv, token_index, head_index),
                kv_vector_for_token_head(&lkv.v, lkv, token_index, head_index),
            )
        })
        .collect()
}

fn kv_heads_for_token_from_f32_slices(
    k_values: &[f32],
    v_values: &[f32],
    shape: KvHeadSliceShape,
    token_index: usize,
) -> Option<Vec<FullPrecisionKvTokenVectors>> {
    (0..shape.n_kv_heads as usize)
        .map(|head_index| {
            kv_head_for_token_from_f32_slices(k_values, v_values, shape, token_index, head_index)
        })
        .collect()
}

#[cfg(test)]
fn hot_tail_partition_stats_from_f32_slices(
    query: &[f32],
    k_values: &[f32],
    v_values: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    hot_token_count: usize,
    kv_head_index: usize,
) -> Result<TurboQuantAttentionPartitionStats, TurboQuantCodecError> {
    if hot_token_count == 0 {
        return Err(TurboQuantCodecError::EmptyKvHistory);
    }
    if query.len() != head_dim {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: head_dim,
            actual: query.len(),
        });
    }
    if kv_head_index >= n_kv_heads {
        return Err(TurboQuantCodecError::MismatchedKvHeadCount {
            expected: n_kv_heads,
            actual: kv_head_index.saturating_add(1),
        });
    }
    let expected_len = n_kv_heads
        .saturating_mul(hot_token_count)
        .saturating_mul(head_dim);
    if k_values.len() < expected_len {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: expected_len,
            actual: k_values.len(),
        });
    }
    if v_values.len() < expected_len {
        return Err(TurboQuantCodecError::MismatchedVectorDimension {
            expected: expected_len,
            actual: v_values.len(),
        });
    }

    let scale = (head_dim as f32).sqrt().recip();
    let head_offset = kv_head_index
        .saturating_mul(hot_token_count)
        .saturating_mul(head_dim);
    let token_range = |token_index: usize| {
        let start = head_offset.saturating_add(token_index.saturating_mul(head_dim));
        start..start.saturating_add(head_dim)
    };

    let mut max_score = f32::NEG_INFINITY;
    for token_index in 0..hot_token_count {
        let key = &k_values[token_range(token_index)];
        let score = query
            .iter()
            .zip(key)
            .map(|(left, right)| left * right)
            .sum::<f32>()
            * scale;
        max_score = max_score.max(score);
    }

    let mut exp_sum = 0.0f32;
    let mut weighted_value_sum = vec![0.0f32; head_dim];
    for token_index in 0..hot_token_count {
        let range = token_range(token_index);
        let key = &k_values[range.clone()];
        let value = &v_values[range];
        let score = query
            .iter()
            .zip(key)
            .map(|(left, right)| left * right)
            .sum::<f32>()
            * scale;
        let weight = (score - max_score).exp();
        exp_sum += weight;
        for (acc, value) in weighted_value_sum.iter_mut().zip(value) {
            *acc += weight * value;
        }
    }

    Ok(TurboQuantAttentionPartitionStats {
        token_count: hot_token_count,
        value_dim: head_dim,
        max_score,
        exp_sum,
        weighted_value_sum,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn merge_cold_stats_with_hot_tail_from_f32_slices(
    cold_stats: &TurboQuantAttentionPartitionStats,
    query: &[f32],
    k_values: &[f32],
    v_values: &[f32],
    n_kv_heads: usize,
    head_dim: usize,
    hot_token_count: usize,
    kv_head_index: usize,
) -> Result<Vec<f32>, TurboQuantCodecError> {
    let cold_stats = TurboQuantAttentionPartitionStatsBatch {
        token_count: cold_stats.token_count,
        value_dim: cold_stats.value_dim,
        max_scores: vec![cold_stats.max_score],
        exp_sums: vec![cold_stats.exp_sum],
        weighted_value_sums: cold_stats.weighted_value_sum.clone(),
    };
    let mut output = Vec::with_capacity(head_dim);
    append_attention_partition_stats_batch_head_with_hot_tail_from_f32_slices(
        &mut output,
        cold_stats.validated()?,
        0,
        query,
        k_values,
        v_values,
        n_kv_heads,
        head_dim,
        hot_token_count,
        kv_head_index,
    )?;
    Ok(output)
}

fn kv_head_for_token_from_f32_slices(
    k_values: &[f32],
    v_values: &[f32],
    shape: KvHeadSliceShape,
    token_index: usize,
    head_index: usize,
) -> Option<FullPrecisionKvTokenVectors> {
    Some((
        kv_vector_for_token_head_from_f32_slice(k_values, shape, token_index, head_index)?,
        kv_vector_for_token_head_from_f32_slice(v_values, shape, token_index, head_index)?,
    ))
}

#[cfg(test)]
fn kv_vector_for_token_head(
    array: &MlxArray,
    lkv: &LayerKV,
    token_index: usize,
    head_index: usize,
) -> Vec<f32> {
    let capacity = lkv.capacity;
    let head_dim = lkv.head_dim as usize;
    let base_element = head_index
        .saturating_mul(capacity)
        .saturating_add(token_index)
        .saturating_mul(head_dim);
    (0..head_dim)
        .map(|dim| array_element_as_f32(array, lkv.dtype, base_element + dim))
        .collect()
}

// Fails closed: an out-of-range read means the shape/capacity bookkeeping is
// wrong, and silently compressing zero vectors would corrupt attention.
fn kv_vector_for_token_head_from_f32_slice(
    values: &[f32],
    shape: KvHeadSliceShape,
    token_index: usize,
    head_index: usize,
) -> Option<Vec<f32>> {
    let capacity = shape.capacity;
    let head_dim = shape.head_dim as usize;
    let base_element = head_index
        .saturating_mul(capacity)
        .saturating_add(token_index)
        .saturating_mul(head_dim);
    values
        .get(base_element..base_element.saturating_add(head_dim))
        .map(|slice| slice.to_vec())
}

fn copy_token_range_to_rotating(
    source: &MlxArray,
    dest: &MlxArray,
    lkv: &LayerKV,
    token_start: usize,
    token_end: usize,
    window: usize,
) -> MlxArray {
    let mut out = dest.clone();
    let mut src_start = token_start;
    while src_start < token_end {
        let dst_start = src_start % window;
        let len = (token_end - src_start).min(window - dst_start);
        let src_stop = src_start + len;
        let dst_stop = dst_start + len;
        let segment = slice(
            source,
            &[0, 0, src_start as i32, 0],
            &[1, lkv.n_kv_heads, src_stop as i32, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        out = slice_update(
            &out,
            &segment,
            &[0, 0, dst_start as i32, 0],
            &[1, lkv.n_kv_heads, dst_stop as i32, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        src_start = src_stop;
    }
    out
}

#[cfg(test)]
fn array_element_as_f32(array: &MlxArray, dtype: MlxDtype, element_index: usize) -> f32 {
    let ptr = array.data_raw();
    if ptr.is_null() {
        return 0.0;
    }

    match dtype {
        MlxDtype::Float32 => unsafe {
            std::slice::from_raw_parts(ptr.cast::<f32>(), array.nbytes() / 4)
                .get(element_index)
                .copied()
                .unwrap_or(0.0)
        },
        MlxDtype::Float16 => unsafe {
            std::slice::from_raw_parts(ptr.cast::<u16>(), array.nbytes() / 2)
                .get(element_index)
                .map(|bits| f16_bits_to_f32(*bits))
                .unwrap_or(0.0)
        },
        MlxDtype::Bfloat16 => unsafe {
            std::slice::from_raw_parts(ptr.cast::<u16>(), array.nbytes() / 2)
                .get(element_index)
                .map(|bits| f32::from_bits((*bits as u32) << 16))
                .unwrap_or(0.0)
        },
        _ => 0.0,
    }
}

#[cfg(test)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = (bits & 0x03ff) as u32;

    let f32_bits = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let leading = fraction.leading_zeros() - 22;
                let mantissa = (fraction << (leading + 1)) & 0x03ff;
                let exponent = 127 - 15 - leading;
                sign | (exponent << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => {
            let exponent = (exponent as u32) + (127 - 15);
            sign | (exponent << 23) | (fraction << 13)
        }
    };

    f32::from_bits(f32_bits)
}

fn turboquant_shadow_layout_for_layer(
    lkv: &LayerKV,
    compression: KvCompressionConfig,
) -> Option<TurboQuantBlockLayout> {
    let head_dim = usize::try_from(lkv.head_dim).ok()?;
    let n_kv_heads = usize::try_from(lkv.n_kv_heads).ok()?;
    TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
        preset: compression.preset,
        block_tokens: KV_CHUNK_TOKENS,
        n_kv_heads,
        head_dim,
        value_group_size: 32.min(head_dim).max(1),
    })
    .ok()
}

fn turboquant_shadow_metal_buffer_array(buffer: &TurboQuantCompressedBlockBuffer) -> MlxArray {
    let metal_buffer = MlxArray::from_raw_data(
        buffer.as_bytes().as_ptr(),
        buffer.as_bytes().len(),
        &[buffer.as_bytes().len() as i32],
        MlxDtype::Uint8,
    );
    eval(&[&metal_buffer]);
    metal_buffer
}

/// First byte of `token_index`'s slot run in the compressed block buffer.
/// Token slots are laid out in ascending byte order (block-major, then token
/// stride), so the bytes written for tokens `[first, last]` form one
/// contiguous range.
fn turboquant_shadow_token_byte_offset(
    layout: &TurboQuantBlockLayout,
    token_index: usize,
) -> usize {
    let block = token_index / layout.config.block_tokens;
    let token_offset = token_index % layout.config.block_tokens;
    block
        .saturating_mul(layout.block_bytes)
        .saturating_add(token_offset.saturating_mul(layout.token_stride_bytes))
}

/// Refresh the device copy of the shadow compressed buffer after a sync that
/// appended tokens `[previous_tokens, compressed_tokens)`.
///
/// The compressed buffer is append-only with a stable prefix, so when the
/// host buffer length is unchanged only the newly written byte range needs
/// uploading; `slice_update` writes it into the existing device array (the
/// prior handle is dropped before eval so MLX can donate the buffer instead
/// of copying it). Buffer growth — which happens once per 256-token block —
/// and the `AX_TURBOQUANT_INCREMENTAL_UPLOAD=0` kill switch fall back to a
/// full re-upload.
fn turboquant_shadow_metal_buffer_update(
    existing: Option<MlxArray>,
    layout: &TurboQuantBlockLayout,
    buffer: &TurboQuantCompressedBlockBuffer,
    previous_tokens: usize,
    compressed_tokens: usize,
) -> MlxArray {
    let bytes = buffer.as_bytes();
    let incremental_eligible = fastpath::turboquant_incremental_upload_enabled()
        && previous_tokens < compressed_tokens
        && bytes.len() <= i32::MAX as usize;
    let Some(existing) =
        existing.filter(|array| incremental_eligible && array.nbytes() == bytes.len())
    else {
        return turboquant_shadow_metal_buffer_array(buffer);
    };

    let start = turboquant_shadow_token_byte_offset(layout, previous_tokens);
    let end = turboquant_shadow_token_byte_offset(layout, compressed_tokens - 1)
        .saturating_add(layout.token_stride_bytes)
        .min(bytes.len());
    if start >= end {
        return turboquant_shadow_metal_buffer_array(buffer);
    }

    let update = MlxArray::from_raw_data(
        bytes[start..end].as_ptr(),
        end - start,
        &[(end - start) as i32],
        MlxDtype::Uint8,
    );
    let updated = slice_update(
        &existing,
        &update,
        &[start as i32],
        &[end as i32],
        &[1],
        None,
    );
    drop(existing);
    eval(&[&updated]);
    updated
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxKVCacheUsage {
    pub logical_tokens: usize,
    pub capacity_tokens: usize,
    pub logical_bytes: u64,
    pub capacity_bytes: u64,
    /// KV-backed attention layers, including sliding-window attention layers.
    pub full_attention_layers: usize,
    /// KV-backed layers with a configured sliding window.
    pub sliding_window_layers: usize,
    pub sliding_window_retained_tokens: usize,
    pub sliding_window_reclaimable_capacity_tokens: usize,
    pub sliding_window_reclaimable_capacity_bytes: u64,
    /// Sliding-window layers currently stored as rotating rings (backing
    /// store bounded to `window + slack` instead of O(context)).
    pub rotated_ring_layers: usize,
    /// Bounded-rollback slack configured for this cache's rings (0 = pure
    /// window-sized rings).
    pub rotating_ring_slack: usize,
    pub linear_state_layers: usize,
    pub linear_state_bytes: u64,
    pub growth_count: u64,
    pub kv_compression: MlxKvCompressionUsage,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxKvCompressionUsage {
    pub policy_enabled: bool,
    pub fused_decode_requested: bool,
    pub fused_decode_attempts: u64,
    pub fused_decode_successes: u64,
    pub fused_decode_metal_successes: u64,
    pub fused_decode_fallbacks: u64,
    pub fused_decode_ready_candidates: u64,
    pub fused_decode_blocked_prefill_only: u64,
    pub fused_decode_blocked_attention_kind: u64,
    pub fused_decode_blocked_linear_attention: u64,
    pub fused_decode_blocked_sliding_window: u64,
    pub fused_decode_blocked_kv_shared: u64,
    pub fused_decode_blocked_ineligible_layer: u64,
    pub fused_decode_blocked_unsupported_preset: u64,
    pub fused_decode_blocked_unsupported_head_dim: u64,
    pub fused_decode_blocked_gqa: u64,
    pub fused_decode_blocked_missing_storage: u64,
    pub fused_decode_blocked_short_context: u64,
    pub fused_decode_query_readback_wall_us: u64,
    pub fused_decode_cold_metal_wall_us: u64,
    pub fused_decode_hot_tail_merge_wall_us: u64,
    pub fused_decode_output_staging_wall_us: u64,
    /// 0 disabled, 1 compression storage active, 2 short context, 3 no eligible layer.
    pub status_code: u32,
    pub preset_code: u32,
    pub key_bits: u32,
    pub value_bits: u32,
    pub eligible_layers: usize,
    pub candidate_token_layers: usize,
    pub hot_token_layers: usize,
    pub full_precision_bytes: u64,
    pub estimated_compressed_bytes: u64,
    pub estimated_saved_bytes: u64,
    pub estimated_ratio_milli: u32,
    pub runtime_storage_layers: usize,
    pub runtime_storage_token_layers: usize,
    pub runtime_storage_bytes: u64,
    pub runtime_storage_written_slots: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxKvCompressionDecodeUsage {
    pub fused_decode_attempts: u64,
    pub fused_decode_successes: u64,
    pub fused_decode_metal_successes: u64,
    pub fused_decode_fallbacks: u64,
    pub fused_decode_ready_candidates: u64,
    pub fused_decode_blocked_prefill_only: u64,
    pub fused_decode_blocked_attention_kind: u64,
    pub fused_decode_blocked_linear_attention: u64,
    pub fused_decode_blocked_sliding_window: u64,
    pub fused_decode_blocked_kv_shared: u64,
    pub fused_decode_blocked_ineligible_layer: u64,
    pub fused_decode_blocked_unsupported_preset: u64,
    pub fused_decode_blocked_unsupported_head_dim: u64,
    pub fused_decode_blocked_gqa: u64,
    pub fused_decode_blocked_missing_storage: u64,
    pub fused_decode_blocked_short_context: u64,
    pub fused_decode_query_readback_wall_us: u64,
    pub fused_decode_cold_metal_wall_us: u64,
    pub fused_decode_hot_tail_merge_wall_us: u64,
    pub fused_decode_output_staging_wall_us: u64,
}

impl MlxKvCompressionUsage {
    pub fn apply_decode_usage(&mut self, usage: MlxKvCompressionDecodeUsage) {
        self.fused_decode_attempts = self
            .fused_decode_attempts
            .saturating_add(usage.fused_decode_attempts);
        self.fused_decode_successes = self
            .fused_decode_successes
            .saturating_add(usage.fused_decode_successes);
        self.fused_decode_metal_successes = self
            .fused_decode_metal_successes
            .saturating_add(usage.fused_decode_metal_successes);
        self.fused_decode_fallbacks = self
            .fused_decode_fallbacks
            .saturating_add(usage.fused_decode_fallbacks);
        self.fused_decode_ready_candidates = self
            .fused_decode_ready_candidates
            .saturating_add(usage.fused_decode_ready_candidates);
        self.fused_decode_blocked_prefill_only = self
            .fused_decode_blocked_prefill_only
            .saturating_add(usage.fused_decode_blocked_prefill_only);
        self.fused_decode_blocked_attention_kind = self
            .fused_decode_blocked_attention_kind
            .saturating_add(usage.fused_decode_blocked_attention_kind);
        self.fused_decode_blocked_linear_attention = self
            .fused_decode_blocked_linear_attention
            .saturating_add(usage.fused_decode_blocked_linear_attention);
        self.fused_decode_blocked_sliding_window = self
            .fused_decode_blocked_sliding_window
            .saturating_add(usage.fused_decode_blocked_sliding_window);
        self.fused_decode_blocked_kv_shared = self
            .fused_decode_blocked_kv_shared
            .saturating_add(usage.fused_decode_blocked_kv_shared);
        self.fused_decode_blocked_ineligible_layer = self
            .fused_decode_blocked_ineligible_layer
            .saturating_add(usage.fused_decode_blocked_ineligible_layer);
        self.fused_decode_blocked_unsupported_preset = self
            .fused_decode_blocked_unsupported_preset
            .saturating_add(usage.fused_decode_blocked_unsupported_preset);
        self.fused_decode_blocked_unsupported_head_dim = self
            .fused_decode_blocked_unsupported_head_dim
            .saturating_add(usage.fused_decode_blocked_unsupported_head_dim);
        self.fused_decode_blocked_gqa = self
            .fused_decode_blocked_gqa
            .saturating_add(usage.fused_decode_blocked_gqa);
        self.fused_decode_blocked_missing_storage = self
            .fused_decode_blocked_missing_storage
            .saturating_add(usage.fused_decode_blocked_missing_storage);
        self.fused_decode_blocked_short_context = self
            .fused_decode_blocked_short_context
            .saturating_add(usage.fused_decode_blocked_short_context);
        self.fused_decode_query_readback_wall_us = self
            .fused_decode_query_readback_wall_us
            .saturating_add(usage.fused_decode_query_readback_wall_us);
        self.fused_decode_cold_metal_wall_us = self
            .fused_decode_cold_metal_wall_us
            .saturating_add(usage.fused_decode_cold_metal_wall_us);
        self.fused_decode_hot_tail_merge_wall_us = self
            .fused_decode_hot_tail_merge_wall_us
            .saturating_add(usage.fused_decode_hot_tail_merge_wall_us);
        self.fused_decode_output_staging_wall_us = self
            .fused_decode_output_staging_wall_us
            .saturating_add(usage.fused_decode_output_staging_wall_us);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxKvCompressionFusedDecodeTiming {
    pub query_readback_wall_us: u64,
    pub cold_metal_wall_us: u64,
    pub hot_tail_merge_wall_us: u64,
    pub output_staging_wall_us: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MlxTurboQuantFusedDecodeAttention {
    pub outputs: Vec<Vec<f32>>,
    pub timing: MlxKvCompressionFusedDecodeTiming,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MlxTurboQuantFusedDecodeFlatAttention {
    pub outputs: Vec<f32>,
    pub timing: MlxKvCompressionFusedDecodeTiming,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxKvCompressionDecodeOutcome {
    Fallback,
    CpuOracle,
    Metal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxKvCompressionDecodeCandidate {
    Disabled,
    PrefillOnly,
    AttentionKind,
    LinearAttention,
    SlidingWindow,
    KvShared,
    IneligibleLayer,
    UnsupportedPreset,
    UnsupportedHeadDim,
    GroupedQueryAttention,
    MissingRuntimeStorage,
    ShortContext,
    Ready,
}

#[derive(Clone)]
struct LayerKV {
    /// Full backing buffer: `[1, n_kv_heads, capacity, head_dim]`.
    k: MlxArray,
    v: MlxArray,
    /// Cached view `[1, n_kv_heads, 0..seq_len, head_dim]` returned by the last
    /// `append` call.  KV-shared layers (Gemma4 layers 24-41) read this directly
    /// via `peek_source_kv` instead of creating a second identical `slice` node.
    /// Having two separate `slice` nodes on the same backing buffer caused MLX to
    /// dispatch the slice kernel twice, adding ~12 µs × 40 dispatches ≈ 0.5 ms per
    /// decode step for E2B.
    last_k_view: Option<MlxArray>,
    last_v_view: Option<MlxArray>,
    n_kv_heads: i32,
    head_dim: i32,
    capacity: usize,
    rotating_window: Option<usize>,
    dtype: MlxDtype,
}

#[derive(Clone)]
struct GlmMlaLayerCache {
    /// `[1, 1, capacity, kv_lora_rank]`, matching mlx-lm's latent KV cache.
    kv_latent: MlxArray,
    /// `[1, 1, capacity, qk_rope_head_dim]`, matching mlx-lm's RoPE key cache.
    k_pe: MlxArray,
    latent_dim: i32,
    rope_dim: i32,
    capacity: usize,
    dtype: MlxDtype,
}

#[derive(Clone)]
struct TurboQuantShadowLayerStorage {
    layout: TurboQuantBlockLayout,
    buffer: TurboQuantCompressedBlockBuffer,
    compressed_tokens: usize,
    metal_buffer: Option<MlxArray>,
}

struct TurboQuantShadowSyncSource {
    layer_idx: usize,
    shape: KvHeadSliceShape,
    compressed_tokens: usize,
    k: MlxArray,
    v: MlxArray,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TurboQuantShadowStorageUsage {
    pub layers: usize,
    pub token_layers: usize,
    pub bytes: u64,
    pub written_slots: usize,
}

#[derive(Clone, Default)]
struct LinearLayerState {
    /// Qwen3.5 gated-delta conv tail: `[1, conv_kernel - 1, conv_dim]`.
    conv_state: Option<MlxArray>,
    /// Qwen3.5 gated-delta recurrent state: `[1, value_heads, value_dim, key_dim]`.
    recurrent_state: Option<MlxArray>,
}

/// Destructor compatible with [`MlxArray::from_managed_data`]. Recovers
/// the `Box<Vec<u8>>` that owned the tensor's data buffer when
/// `try_deserialize_from_bytes` constructed the array, and drops it so
/// the heap allocation is freed.
///
/// # Safety
///
/// `payload` must have been produced by `Box::into_raw(Box::new(Vec<u8>))`
/// and not yet recovered or freed. MLX guarantees `payload` is non-null
/// and called exactly once per matching `from_managed_data` call.
unsafe extern "C" fn vec_payload_drop(payload: *mut std::ffi::c_void) {
    if payload.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(payload as *mut Vec<u8>);
    }
}

/// Error produced by [`MlxKVCache::try_deserialize_from_bytes`].
#[derive(Debug)]
pub enum MlxKVCacheSerializeError {
    /// Header magic did not match the F3-disk-cache wire format.
    BadMagic,
    /// Serialised payload was produced by an incompatible format version.
    UnsupportedVersion(u32),
    /// Payload ended before the structure required more bytes.
    UnexpectedEof,
    /// Encountered a dtype tag that does not map to a known MlxDtype.
    UnknownDtype(u8),
    /// Encountered an unknown layer-kind discriminator.
    UnknownLayerKind(u8),
    /// Tensor metadata declared an unsupported rank.
    BadShape(usize),
    /// TurboQuant compressed shadow state was inconsistent or used unknown tags.
    InvalidTurboQuantState(String),
}

impl std::fmt::Display for MlxKVCacheSerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "kv-cache payload has wrong magic"),
            Self::UnsupportedVersion(v) => write!(f, "kv-cache payload version {v} unsupported"),
            Self::UnexpectedEof => write!(f, "kv-cache payload truncated"),
            Self::UnknownDtype(t) => write!(f, "kv-cache payload has unknown dtype tag {t}"),
            Self::UnknownLayerKind(t) => write!(f, "kv-cache payload has unknown layer kind {t}"),
            Self::BadShape(n) => write!(f, "kv-cache payload has bad tensor rank {n}"),
            Self::InvalidTurboQuantState(message) => {
                write!(
                    f,
                    "kv-cache payload has invalid TurboQuant state: {message}"
                )
            }
        }
    }
}

impl std::error::Error for MlxKVCacheSerializeError {}

/// Minimal positional reader over a byte slice. Used by
/// `MlxKVCache::try_deserialize_from_bytes` so the per-field reads stay
/// readable; the cursor refuses to advance past the end of the slice.
struct DeserializeCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> DeserializeCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], MlxKVCacheSerializeError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(MlxKVCacheSerializeError::UnexpectedEof)?;
        if end > self.bytes.len() {
            return Err(MlxKVCacheSerializeError::UnexpectedEof);
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, MlxKVCacheSerializeError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, MlxKVCacheSerializeError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, MlxKVCacheSerializeError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, MlxKVCacheSerializeError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_usize(&mut self) -> Result<usize, MlxKVCacheSerializeError> {
        usize::try_from(self.read_u64()?).map_err(|_| MlxKVCacheSerializeError::BadShape(8))
    }
}

/// Borrowed view of a GLM-MLA layer's cached KV state. Returned by
/// [`MlxKVCache::glm_mla_layer_state`] for debug tooling that needs to
/// inspect per-layer cache contents without taking ownership.
pub struct GlmMlaLayerStateView<'a> {
    /// The full backing buffer for the latent KV cache; shape
    /// `[1, 1, capacity, latent_dim]`. Valid region is `[0..seq_len]`.
    pub kv_latent: &'a MlxArray,
    /// The full backing buffer for the RoPE key cache; shape
    /// `[1, 1, capacity, rope_dim]`. Valid region is `[0..seq_len]`.
    pub k_pe: &'a MlxArray,
    /// Inner dim of `kv_latent` — equal to the model's `kv_lora_rank`.
    pub latent_dim: i32,
    /// Inner dim of `k_pe` — equal to the model's `qk_rope_head_dim`.
    pub rope_dim: i32,
}

/// Per-request attention cache with chunked KV pre-allocation.
///
/// Full-attention KV shape convention:
/// `[1, n_kv_heads, seq_len, head_dim]` (batch=1, SDPA-native format).
///
/// ## Growth strategy
///
/// Unlike the naive approach that calls `concatenate` on every append (O(n) data
/// movement per step), this cache pre-allocates buffers in `KV_CHUNK_TOKENS`-sized
/// blocks and uses `slice_update` to write new tokens into the pre-allocated region.
/// Buffer growth (via concatenation with zeros) happens at most every `KV_CHUNK_TOKENS`
/// steps — typically 0–1 times per request for common prompt+decode lengths.
///
/// ## Draft rollback
///
/// `trim_to(prefix_len)` only updates `seq_len`.  The "trimmed" positions remain in
/// the backing buffer but are beyond the logical boundary, so SDPA never sees them.
/// The next `append` overwrites from `prefix_len`, restoring correctness.
/// TurboQuant shadow storage is truncated separately because it serializes its
/// compressed runtime buffer.
pub struct MlxKVCache {
    layers: Vec<Option<LayerKV>>,
    glm_mla_layers: Vec<Option<GlmMlaLayerCache>>,
    linear_layers: Vec<LinearLayerState>,
    turboquant_shadow_layers: Vec<Option<TurboQuantShadowLayerStorage>>,
    /// Current logical sequence length (token count cached). Private so
    /// every mutation goes through [`Self::advance`] / [`Self::set_seq_len`]
    /// — the historical footgun was call sites bumping this field out of
    /// sync with what was actually appended.
    seq_len: usize,
    /// RoPE offset added to `seq_len` for positional encoding.  Used when
    /// the KV cache has fewer physical entries than the logical sequence
    /// position (e.g. after capped MTP warmup where only the last N tokens
    /// are warmed up but RoPE needs the full prompt offset).
    pub rope_offset: usize,
    growth_count: u64,
    turboquant_decode_usage: MlxKvCompressionDecodeUsage,
    use_rotating_sliding_decode: bool,
    /// Extra ring slots beyond the sliding window that keep speculative
    /// rollback (`trim_to`) sound after rotation. `0` = pure ring: exactly
    /// window-sized, mask-free single-token SDPA, no rollback permitted
    /// (the pre-bounded-rollback behavior). `> 0` = bounded ring: capacity
    /// `window + slack`, every SDPA over the ring needs a validity mask
    /// ([`SlidingRingLayout`]), and `trim_to` accepts rollbacks up to
    /// `slack` tokens deep.
    rotating_slack: usize,
}

/// Ring geometry a forward presents to SDPA for a sliding-window layer when
/// the rotating decode path engages. `capacity == window + slack`; the ring
/// stores token `t` at slot `t % capacity`, so SDPA must mask slots whose
/// resident token falls outside a query's `(pos - window, pos]` range (or
/// was never written / rolled back). Produced by
/// [`MlxKVCache::sliding_ring_layout`]; the append site and every mask
/// builder must derive their decisions from this one predicate so view and
/// mask can never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlidingRingLayout {
    pub window: usize,
    pub capacity: usize,
    /// Logical index of the first token this forward appends (`cache.seq_len`
    /// at forward entry).
    pub write_start: usize,
}

impl SlidingRingLayout {
    /// Whether SDPA needs an explicit slot-validity mask over the ring.
    /// Pure rings (`capacity == window`) are exactly full for single-token
    /// decode, so every slot is live and mask-free SDPA is correct.
    pub fn needs_mask(&self, seq: usize) -> bool {
        self.capacity > self.window || seq > 1
    }
}

impl Clone for MlxKVCache {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            glm_mla_layers: self.glm_mla_layers.clone(),
            linear_layers: self.linear_layers.clone(),
            turboquant_shadow_layers: self.turboquant_shadow_layers.clone(),
            seq_len: self.seq_len,
            rope_offset: self.rope_offset,
            growth_count: self.growth_count,
            turboquant_decode_usage: self.turboquant_decode_usage,
            use_rotating_sliding_decode: self.use_rotating_sliding_decode,
            rotating_slack: self.rotating_slack,
        }
    }
}

impl MlxKVCache {
    fn bump_counter(counter: &mut u64) {
        *counter = counter.saturating_add(1);
    }

    fn turboquant_candidate_counter(
        usage: &mut MlxKvCompressionDecodeUsage,
        candidate: MlxKvCompressionDecodeCandidate,
    ) -> Option<&mut u64> {
        match candidate {
            MlxKvCompressionDecodeCandidate::Disabled => None,
            MlxKvCompressionDecodeCandidate::Ready => {
                Some(&mut usage.fused_decode_ready_candidates)
            }
            MlxKvCompressionDecodeCandidate::PrefillOnly => {
                Some(&mut usage.fused_decode_blocked_prefill_only)
            }
            MlxKvCompressionDecodeCandidate::AttentionKind => {
                Some(&mut usage.fused_decode_blocked_attention_kind)
            }
            MlxKvCompressionDecodeCandidate::LinearAttention => {
                Some(&mut usage.fused_decode_blocked_linear_attention)
            }
            MlxKvCompressionDecodeCandidate::SlidingWindow => {
                Some(&mut usage.fused_decode_blocked_sliding_window)
            }
            MlxKvCompressionDecodeCandidate::KvShared => {
                Some(&mut usage.fused_decode_blocked_kv_shared)
            }
            MlxKvCompressionDecodeCandidate::IneligibleLayer => {
                Some(&mut usage.fused_decode_blocked_ineligible_layer)
            }
            MlxKvCompressionDecodeCandidate::UnsupportedPreset => {
                Some(&mut usage.fused_decode_blocked_unsupported_preset)
            }
            MlxKvCompressionDecodeCandidate::UnsupportedHeadDim => {
                Some(&mut usage.fused_decode_blocked_unsupported_head_dim)
            }
            MlxKvCompressionDecodeCandidate::GroupedQueryAttention => {
                Some(&mut usage.fused_decode_blocked_gqa)
            }
            MlxKvCompressionDecodeCandidate::MissingRuntimeStorage => {
                Some(&mut usage.fused_decode_blocked_missing_storage)
            }
            MlxKvCompressionDecodeCandidate::ShortContext => {
                Some(&mut usage.fused_decode_blocked_short_context)
            }
        }
    }

    // ── Wire-format constants for `serialize_to_bytes` / `try_deserialize_from_bytes` ──
    // The format is private to this module; see the F3 disk-cache PRD
    // (`MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md`) §3.3 / §4 for the rationale.
    /// File magic for the AX disk-format payload section. Distinct from
    /// `AXKV` (used by the future disk wrapper for outer file framing) so
    /// a partial / nested payload cannot be mistaken for a complete file.
    const SERIALIZE_MAGIC: &'static [u8; 4] = b"AXKB";
    const SERIALIZE_VERSION: u32 = 3;

    /// Private KV wire-format version, exposed for the durable prefix
    /// cache's canonical key (schema v3 commits to the payload version so a
    /// format bump cleanly invalidates older disk entries).
    pub const fn serialize_version() -> u32 {
        Self::SERIALIZE_VERSION
    }
    const LAYER_KIND_EMPTY: u8 = 0;
    const LAYER_KIND_FA: u8 = 1;
    const LAYER_KIND_MLA: u8 = 2;
    const LAYER_KIND_LINEAR: u8 = 3;
    const TENSOR_PRESENT_TAG: u8 = 1;
    const TENSOR_ABSENT_TAG: u8 = 0;

    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            glm_mla_layers: (0..num_layers).map(|_| None).collect(),
            linear_layers: (0..num_layers)
                .map(|_| LinearLayerState::default())
                .collect(),
            turboquant_shadow_layers: (0..num_layers).map(|_| None).collect(),
            seq_len: 0,
            rope_offset: 0,
            growth_count: 0,
            turboquant_decode_usage: MlxKvCompressionDecodeUsage::default(),
            use_rotating_sliding_decode: false,
            rotating_slack: 0,
        }
    }

    /// Current logical sequence length (token count cached).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Advance the logical boundary after a forward pass appended `n`
    /// tokens to every KV-backed layer. Call once per forward (appends
    /// write at `seq_len` per layer, so the boundary must not move until
    /// all layers have appended).
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Set the logical boundary to an absolute position. Prefer
    /// [`Self::advance`] after forwards; this is for seeding a cache at a
    /// known position (prefill restore, warmup, tests). For rollback use
    /// [`Self::trim_to`], which also validates ring residency and truncates
    /// TurboQuant shadow storage.
    pub fn set_seq_len(&mut self, n: usize) {
        self.seq_len = n;
    }

    pub fn set_rotating_sliding_decode(&mut self, enabled: bool) {
        self.use_rotating_sliding_decode = enabled;
    }

    /// Set the bounded-rollback slack for rotating sliding-window layers.
    /// Must be latched before the first rotating append of a request and
    /// never changed afterwards — converted rings keep their capacity.
    pub fn set_rotating_sliding_slack(&mut self, slack: usize) {
        self.rotating_slack = slack;
    }

    pub fn rotating_sliding_slack(&self) -> usize {
        self.rotating_slack
    }

    /// The ring geometry an `append_with_retained_window` call on a sliding
    /// layer will use for a forward of `seq` new tokens, or `None` when the
    /// forward stays on the ordered (non-rotating) path for that window.
    ///
    /// This is the single source of truth shared by the append site and the
    /// SDPA mask builders: a multi-token forward may enter the ring only in
    /// bounded mode (`rotating_slack > 0`) and only when it fits inside the
    /// slack, which also bounds the deepest `trim_to` rollback the ring can
    /// absorb (rolled-back tokens rewrite into their own `t % capacity`
    /// slots, so rollback itself is free).
    pub fn sliding_ring_layout(
        &self,
        window: Option<usize>,
        seq: usize,
    ) -> Option<SlidingRingLayout> {
        if !self.use_rotating_sliding_decode || seq == 0 {
            return None;
        }
        let window = window.filter(|window| *window > 0)?;
        if self.seq_len + seq <= window {
            return None;
        }
        let eligible = seq == 1 || (self.rotating_slack > 0 && seq <= self.rotating_slack);
        if !eligible {
            return None;
        }
        Some(SlidingRingLayout {
            window,
            capacity: window + self.rotating_slack,
            write_start: self.seq_len,
        })
    }

    // ── F3 M1: serialization for the disk-prefix-cache disk format ──
    //
    // These two methods are the foundation for the F3 disk-prefix-cache
    // PRD (MLX-DISK-PREFIX-CACHE-PRD-2026-05-14.md): a process-restart-
    // surviving second-tier cache. M1's goal is a round-trip-correct
    // wire format for the KV state. M2 (runner wiring), M3 (eviction +
    // concurrency), M4 (cross-restart validation), and M5 (docs) are
    // separate deliverables.
    //
    // Wire format (private to this module; the outer disk-file framing
    // is the F3 disk wrapper's responsibility):
    //
    //   header:
    //     magic[4]         = b"AXKB"
    //     version: u32     = 3
    //     seq_len: u64
    //     growth_count: u64
    //     rope_offset: u64  (v3+; extra RoPE position beyond seq_len,
    //                        e.g. after capped MTP warmup)
    //     layer_count: u32
    //     reserved: u32
    //
    //   for each layer 0..layer_count:
    //     kind: u8   (0=Empty, 1=FA, 2=MLA, 3=Linear)
    //     reserved[7]
    //     layer-kind-specific payload (see below)
    //
    //   FA payload:     [rotating_window: u64 (v3+; 0 = ordered storage,
    //                    otherwise the sliding window of a rotated ring whose
    //                    K/V tensors are slot-ordered, token t at slot
    //                    t % capacity)][K tensor][V tensor]
    //   MLA payload:    [kv_latent tensor][k_pe tensor]
    //   Linear payload: [tag:u8][optional conv_state][tag:u8][optional recurrent_state]
    //   Empty payload:  (nothing)
    //
    //   TurboQuant shadow section:
    //     repeated layer_count times:
    //       present: u8 (0/1)
    //       reserved[7]
    //       if present:
    //         compressed_tokens: u64
    //         layout config, key codec, token_count, written_slot_count
    //         compressed byte payload and written-slot flags
    //
    //   tensor encoding (32 bytes header + raw bytes):
    //     dtype: u8          (MlxDtype variant index 0..=13)
    //     ndim: u8           (1..=4)
    //     reserved[6]
    //     shape: [i32; 4]    (zero-padded for ndim<4)
    //     byte_count: u64
    //     bytes: [u8; byte_count]
    //
    fn dtype_to_tag(dtype: MlxDtype) -> u8 {
        match dtype {
            MlxDtype::Bool => 0,
            MlxDtype::Uint8 => 1,
            MlxDtype::Uint16 => 2,
            MlxDtype::Uint32 => 3,
            MlxDtype::Uint64 => 4,
            MlxDtype::Int8 => 5,
            MlxDtype::Int16 => 6,
            MlxDtype::Int32 => 7,
            MlxDtype::Int64 => 8,
            MlxDtype::Float16 => 9,
            MlxDtype::Float32 => 10,
            MlxDtype::Float64 => 11,
            MlxDtype::Bfloat16 => 12,
            MlxDtype::Complex64 => 13,
        }
    }

    fn dtype_from_tag(tag: u8) -> Result<MlxDtype, MlxKVCacheSerializeError> {
        Ok(match tag {
            0 => MlxDtype::Bool,
            1 => MlxDtype::Uint8,
            2 => MlxDtype::Uint16,
            3 => MlxDtype::Uint32,
            4 => MlxDtype::Uint64,
            5 => MlxDtype::Int8,
            6 => MlxDtype::Int16,
            7 => MlxDtype::Int32,
            8 => MlxDtype::Int64,
            9 => MlxDtype::Float16,
            10 => MlxDtype::Float32,
            11 => MlxDtype::Float64,
            12 => MlxDtype::Bfloat16,
            13 => MlxDtype::Complex64,
            other => return Err(MlxKVCacheSerializeError::UnknownDtype(other)),
        })
    }

    /// Serialize only the logical token prefix of a `[B, H, tokens, D]`
    /// tensor (token axis 2). The backing buffer is capacity-sized, so
    /// serializing it whole makes every snapshot cost O(capacity) bytes —
    /// and the prefix-snapshot store serializes one snapshot per
    /// block-aligned prefix, turning that into O(prefixes × capacity)
    /// memcpy per prefill. The wire format is unchanged: the deserializer
    /// derives capacity from the stored shape and regrows on append.
    fn serialize_tensor_logical(out: &mut Vec<u8>, arr: &MlxArray, logical_tokens: Option<usize>) {
        let shape = arr.shape();
        if let Some(tokens) = logical_tokens
            && shape.len() == 4
            && (shape[2] as usize) > tokens
        {
            let stop = [shape[0], shape[1], tokens as i32, shape[3]];
            let trimmed = slice(arr, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
            // The slice is a strided view over the capacity buffer;
            // materialize it so `data_raw` reads row-contiguous bytes.
            let trimmed = contiguous(&trimmed, None);
            Self::serialize_tensor(out, &trimmed);
        } else {
            Self::serialize_tensor(out, arr);
        }
    }

    fn serialize_tensor(out: &mut Vec<u8>, arr: &MlxArray) {
        // Materialise before reading bytes; data_raw is only valid post-eval.
        eval(&[arr]);
        let dtype_tag = Self::dtype_to_tag(arr.dtype());
        let shape = arr.shape();
        let ndim = shape.len() as u8;
        debug_assert!(shape.len() <= 4, "tensor ndim must be ≤ 4");

        out.push(dtype_tag);
        out.push(ndim);
        out.extend_from_slice(&[0u8; 6]); // reserved
        let mut padded_shape = [0i32; 4];
        for (i, &s) in shape.iter().enumerate() {
            padded_shape[i] = s;
        }
        for s in padded_shape {
            out.extend_from_slice(&s.to_le_bytes());
        }

        let byte_count = arr.nbytes() as u64;
        out.extend_from_slice(&byte_count.to_le_bytes());

        // SAFETY: data_raw returns a host-visible pointer after eval. We
        // copy `byte_count` bytes; the slice is bounded by the array's
        // own reported size.
        unsafe {
            let ptr = arr.data_raw();
            let slice = std::slice::from_raw_parts(ptr, byte_count as usize);
            out.extend_from_slice(slice);
        }
    }

    fn read_tensor(
        cursor: &mut DeserializeCursor<'_>,
    ) -> Result<MlxArray, MlxKVCacheSerializeError> {
        let dtype_tag = cursor.read_u8()?;
        let ndim = cursor.read_u8()? as usize;
        if ndim == 0 || ndim > 4 {
            return Err(MlxKVCacheSerializeError::BadShape(ndim));
        }
        cursor.read_bytes(6)?; // reserved
        let mut shape = [0i32; 4];
        for s in &mut shape {
            *s = cursor.read_i32()?;
        }
        let dtype = Self::dtype_from_tag(dtype_tag)?;
        let byte_count = cursor.read_u64()? as usize;

        // Pre-validate shape × dtype against `byte_count` so a tampered or
        // corrupted payload returns a structured error instead of tripping
        // the assert inside `MlxArray::from_managed_data`. Any of:
        // negative dim, overflow in product, or undersized `byte_count`
        // is flagged as `BadShape(ndim)`.
        let mut element_count: usize = 1;
        for &dim in shape[..ndim].iter() {
            if dim < 0 {
                return Err(MlxKVCacheSerializeError::BadShape(ndim));
            }
            element_count = element_count
                .checked_mul(dim as usize)
                .ok_or(MlxKVCacheSerializeError::BadShape(ndim))?;
        }
        let required_bytes = element_count
            .checked_mul(dtype.size_bytes())
            .ok_or(MlxKVCacheSerializeError::BadShape(ndim))?;
        if byte_count < required_bytes {
            return Err(MlxKVCacheSerializeError::BadShape(ndim));
        }

        let bytes_view = cursor.read_bytes(byte_count)?;

        // `MlxArray::from_raw_data` borrows the data buffer (see
        // mlx-sys/src/array.rs:80: "MLX does **not** copy"), so passing the
        // input slice's pointer would leave the array dangling once the
        // caller's payload buffer is dropped. The deserializer must own
        // the bytes for the array's lifetime. We allocate a `Vec<u8>` on
        // the heap, hand its data pointer to MLX via `from_managed_data`,
        // and register a destructor that reclaims the boxed Vec when MLX
        // releases the array. This makes the returned `MlxArray`
        // self-sufficient and decoupled from the input slice lifetime.
        let owned: Box<Vec<u8>> = Box::new(bytes_view.to_vec());
        let data_ptr = owned.as_ptr();
        let payload = Box::into_raw(owned) as *mut std::ffi::c_void;
        // SAFETY: data_ptr points at heap memory owned by the boxed Vec.
        // The Vec's buffer outlives the MlxArray because `vec_payload_drop`
        // only fires when MLX releases the array's last reference,
        // reclaiming the Box and freeing the buffer. `byte_count` matches
        // the Vec's length, which equals shape × dtype size by
        // construction (the serialiser wrote the same value).
        Ok(unsafe {
            MlxArray::from_managed_data(
                data_ptr,
                byte_count,
                &shape[..ndim],
                dtype,
                payload,
                vec_payload_drop,
            )
        })
    }

    fn turboquant_preset_to_tag(preset: TurboQuantPreset) -> u8 {
        match preset {
            TurboQuantPreset::K8V4 => 1,
            TurboQuantPreset::K4V4 => 2,
            TurboQuantPreset::K3V4Research => 3,
            TurboQuantPreset::K16V4 => 4,
            TurboQuantPreset::K8V3_5 => 5,
            TurboQuantPreset::K7V4 => 6,
        }
    }

    fn turboquant_preset_from_tag(tag: u8) -> Result<TurboQuantPreset, MlxKVCacheSerializeError> {
        Ok(match tag {
            1 => TurboQuantPreset::K8V4,
            2 => TurboQuantPreset::K4V4,
            3 => TurboQuantPreset::K3V4Research,
            4 => TurboQuantPreset::K16V4,
            5 => TurboQuantPreset::K8V3_5,
            6 => TurboQuantPreset::K7V4,
            other => {
                return Err(MlxKVCacheSerializeError::InvalidTurboQuantState(format!(
                    "unknown preset tag {other}"
                )));
            }
        })
    }

    fn turboquant_centroid_kind_to_tag(kind: TurboQuantKeyCentroidKind) -> u8 {
        match kind {
            TurboQuantKeyCentroidKind::Uniform => 1,
            TurboQuantKeyCentroidKind::LloydMaxGaussian => 2,
        }
    }

    fn turboquant_centroid_kind_from_tag(
        tag: u8,
    ) -> Result<TurboQuantKeyCentroidKind, MlxKVCacheSerializeError> {
        Ok(match tag {
            1 => TurboQuantKeyCentroidKind::Uniform,
            2 => TurboQuantKeyCentroidKind::LloydMaxGaussian,
            other => {
                return Err(MlxKVCacheSerializeError::InvalidTurboQuantState(format!(
                    "unknown key centroid kind tag {other}"
                )));
            }
        })
    }

    fn write_turboquant_shadow_layer(out: &mut Vec<u8>, storage: &TurboQuantShadowLayerStorage) {
        out.extend_from_slice(&(storage.compressed_tokens as u64).to_le_bytes());
        let meta_state = storage.buffer.meta_state();
        let state = storage.buffer.state();
        out.push(Self::turboquant_preset_to_tag(
            meta_state.layout.config.preset,
        ));
        out.extend_from_slice(&[0u8; 7]);
        out.extend_from_slice(&(meta_state.layout.config.block_tokens as u64).to_le_bytes());
        out.extend_from_slice(&(meta_state.layout.config.n_kv_heads as u64).to_le_bytes());
        out.extend_from_slice(&(meta_state.layout.config.head_dim as u64).to_le_bytes());
        out.extend_from_slice(&(meta_state.layout.config.value_group_size as u64).to_le_bytes());
        out.extend_from_slice(&meta_state.key_codec.version.to_le_bytes());
        out.push(Self::turboquant_centroid_kind_to_tag(
            meta_state.key_codec.centroid_kind,
        ));
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(&(meta_state.token_count as u64).to_le_bytes());
        out.extend_from_slice(&(meta_state.written_slot_count as u64).to_le_bytes());
        out.extend_from_slice(&(state.bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&state.bytes);
        out.extend_from_slice(&(state.written_slots.len() as u64).to_le_bytes());
        for written in state.written_slots {
            out.push(u8::from(written));
        }
    }

    fn read_turboquant_shadow_layer(
        cursor: &mut DeserializeCursor<'_>,
    ) -> Result<TurboQuantShadowLayerStorage, MlxKVCacheSerializeError> {
        let compressed_tokens = cursor.read_usize()?;
        let preset = Self::turboquant_preset_from_tag(cursor.read_u8()?)?;
        cursor.read_bytes(7)?;
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset,
            block_tokens: cursor.read_usize()?,
            n_kv_heads: cursor.read_usize()?,
            head_dim: cursor.read_usize()?,
            value_group_size: cursor.read_usize()?,
        })
        .map_err(|error| MlxKVCacheSerializeError::InvalidTurboQuantState(error.to_string()))?;
        let key_codec = TurboQuantKeyCodecConfig {
            version: cursor.read_u32()?,
            centroid_kind: Self::turboquant_centroid_kind_from_tag(cursor.read_u8()?)?,
        };
        cursor.read_bytes(3)?;
        let token_count = cursor.read_usize()?;
        let written_slot_count = cursor.read_usize()?;
        let byte_count = cursor.read_usize()?;
        let bytes = cursor.read_bytes(byte_count)?.to_vec();
        let slot_count = cursor.read_usize()?;
        let mut written_slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let flag = cursor.read_u8()?;
            match flag {
                0 => written_slots.push(false),
                1 => written_slots.push(true),
                other => {
                    return Err(MlxKVCacheSerializeError::InvalidTurboQuantState(format!(
                        "invalid written-slot flag {other}"
                    )));
                }
            }
        }
        let buffer = TurboQuantCompressedBlockBuffer::from_state(
            TurboQuantCompressedBlockBufferMetaState {
                layout,
                key_codec,
                token_count,
                written_slot_count,
            },
            TurboQuantCompressedBlockBufferState {
                bytes,
                written_slots,
            },
        )
        .map_err(|error| MlxKVCacheSerializeError::InvalidTurboQuantState(error.to_string()))?;

        Ok(TurboQuantShadowLayerStorage {
            layout,
            buffer,
            compressed_tokens,
            metal_buffer: None,
        })
    }

    /// Serialise the cache to a self-contained byte payload that
    /// `try_deserialize_from_bytes` can reconstruct. Format is private to
    /// this module and versioned via `SERIALIZE_VERSION`; cross-version
    /// reads return an error rather than silently degrading.
    ///
    pub fn serialize_to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(Self::SERIALIZE_MAGIC);
        out.extend_from_slice(&Self::SERIALIZE_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u64).to_le_bytes());
        out.extend_from_slice(&self.growth_count.to_le_bytes());
        out.extend_from_slice(&(self.rope_offset as u64).to_le_bytes());
        let layer_count = self.layers.len() as u32;
        out.extend_from_slice(&layer_count.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved

        for idx in 0..self.layers.len() {
            // Per-index disambiguation: at most one of the three layer
            // vectors is populated. The encoded `kind` byte tells the
            // reader which payload to expect.
            if let Some(fa) = &self.layers[idx] {
                out.push(Self::LAYER_KIND_FA);
                out.extend_from_slice(&[0u8; 7]);
                // Ring geometry must survive the round trip: a rotated
                // layer's buffer is slot-ordered, and restoring it as
                // ordered storage would make the first post-restore append
                // treat ring slots as a token-ordered prefix.
                out.extend_from_slice(&(fa.rotating_window.unwrap_or(0) as u64).to_le_bytes());
                // Rotating-window layers store slots, not token order — the
                // full buffer is the logical state and must not be trimmed.
                let logical_tokens = if fa.rotating_window.is_none() {
                    Some(self.seq_len)
                } else {
                    None
                };
                Self::serialize_tensor_logical(&mut out, &fa.k, logical_tokens);
                Self::serialize_tensor_logical(&mut out, &fa.v, logical_tokens);
            } else if let Some(mla) = &self.glm_mla_layers[idx] {
                out.push(Self::LAYER_KIND_MLA);
                out.extend_from_slice(&[0u8; 7]);
                Self::serialize_tensor_logical(&mut out, &mla.kv_latent, Some(self.seq_len));
                Self::serialize_tensor_logical(&mut out, &mla.k_pe, Some(self.seq_len));
            } else if self.linear_layers[idx].conv_state.is_some()
                || self.linear_layers[idx].recurrent_state.is_some()
            {
                let linear = &self.linear_layers[idx];
                out.push(Self::LAYER_KIND_LINEAR);
                out.extend_from_slice(&[0u8; 7]);
                if let Some(arr) = &linear.conv_state {
                    out.push(Self::TENSOR_PRESENT_TAG);
                    Self::serialize_tensor(&mut out, arr);
                } else {
                    out.push(Self::TENSOR_ABSENT_TAG);
                }
                if let Some(arr) = &linear.recurrent_state {
                    out.push(Self::TENSOR_PRESENT_TAG);
                    Self::serialize_tensor(&mut out, arr);
                } else {
                    out.push(Self::TENSOR_ABSENT_TAG);
                }
            } else {
                out.push(Self::LAYER_KIND_EMPTY);
                out.extend_from_slice(&[0u8; 7]);
            }
        }

        for storage in &self.turboquant_shadow_layers {
            if let Some(storage) = storage {
                out.push(Self::TENSOR_PRESENT_TAG);
                out.extend_from_slice(&[0u8; 7]);
                Self::write_turboquant_shadow_layer(&mut out, storage);
            } else {
                out.push(Self::TENSOR_ABSENT_TAG);
                out.extend_from_slice(&[0u8; 7]);
            }
        }
        out
    }

    /// Reconstruct a cache from a byte payload produced by
    /// `serialize_to_bytes`. Returns an error on magic / version
    /// mismatch, truncated data, unknown dtype tags, or shape errors —
    /// never silently degrades.
    pub fn try_deserialize_from_bytes(bytes: &[u8]) -> Result<Self, MlxKVCacheSerializeError> {
        let mut cursor = DeserializeCursor::new(bytes);
        let magic = cursor.read_bytes(4)?;
        if magic != Self::SERIALIZE_MAGIC {
            return Err(MlxKVCacheSerializeError::BadMagic);
        }
        let version = cursor.read_u32()?;
        if version != Self::SERIALIZE_VERSION {
            return Err(MlxKVCacheSerializeError::UnsupportedVersion(version));
        }
        let seq_len = cursor.read_u64()? as usize;
        let growth_count = cursor.read_u64()?;
        let rope_offset = cursor.read_usize()?;
        let layer_count = cursor.read_u32()? as usize;
        let _reserved = cursor.read_u32()?;

        let mut cache = Self::new(layer_count);
        cache.seq_len = seq_len;
        cache.growth_count = growth_count;
        cache.rope_offset = rope_offset;

        for idx in 0..layer_count {
            let kind = cursor.read_u8()?;
            cursor.read_bytes(7)?; // reserved
            match kind {
                k if k == Self::LAYER_KIND_EMPTY => continue,
                k if k == Self::LAYER_KIND_FA => {
                    let ring_window = cursor.read_usize()?;
                    let k_arr = Self::read_tensor(&mut cursor)?;
                    let v_arr = Self::read_tensor(&mut cursor)?;
                    let shape = k_arr.shape();
                    if shape.len() < 4 {
                        return Err(MlxKVCacheSerializeError::BadShape(shape.len()));
                    }
                    let capacity = shape[2] as usize;
                    let rotating_window = (ring_window != 0).then_some(ring_window);
                    if let Some(window) = rotating_window {
                        // A ring narrower than its window (or an ordered
                        // buffer shorter than seq_len) cannot have been
                        // produced by the serializer.
                        if capacity < window {
                            return Err(MlxKVCacheSerializeError::BadShape(shape.len()));
                        }
                        // Re-latch the ring configuration so post-restore
                        // appends reproduce the same geometry instead of
                        // reconverting (which would read slot-ordered data
                        // as token order).
                        cache.use_rotating_sliding_decode = true;
                        cache.rotating_slack = capacity - window;
                    } else if seq_len > capacity {
                        return Err(MlxKVCacheSerializeError::BadShape(shape.len()));
                    }
                    cache.layers[idx] = Some(LayerKV {
                        last_k_view: None,
                        last_v_view: None,
                        n_kv_heads: shape[1],
                        head_dim: shape[3],
                        capacity,
                        rotating_window,
                        dtype: k_arr.dtype(),
                        k: k_arr,
                        v: v_arr,
                    });
                }
                k if k == Self::LAYER_KIND_MLA => {
                    let kv_latent = Self::read_tensor(&mut cursor)?;
                    let k_pe = Self::read_tensor(&mut cursor)?;
                    let kv_shape = kv_latent.shape();
                    let pe_shape = k_pe.shape();
                    if kv_shape.len() < 4 || pe_shape.len() < 4 {
                        return Err(MlxKVCacheSerializeError::BadShape(kv_shape.len()));
                    }
                    cache.glm_mla_layers[idx] = Some(GlmMlaLayerCache {
                        latent_dim: kv_shape[3],
                        rope_dim: pe_shape[3],
                        capacity: kv_shape[2] as usize,
                        dtype: kv_latent.dtype(),
                        kv_latent,
                        k_pe,
                    });
                }
                k if k == Self::LAYER_KIND_LINEAR => {
                    let conv_present = cursor.read_u8()?;
                    let conv_state = if conv_present == Self::TENSOR_PRESENT_TAG {
                        Some(Self::read_tensor(&mut cursor)?)
                    } else {
                        None
                    };
                    let rec_present = cursor.read_u8()?;
                    let recurrent_state = if rec_present == Self::TENSOR_PRESENT_TAG {
                        Some(Self::read_tensor(&mut cursor)?)
                    } else {
                        None
                    };
                    cache.linear_layers[idx] = LinearLayerState {
                        conv_state,
                        recurrent_state,
                    };
                }
                other => return Err(MlxKVCacheSerializeError::UnknownLayerKind(other)),
            }
        }

        for idx in 0..layer_count {
            let present = cursor.read_u8()?;
            cursor.read_bytes(7)?;
            match present {
                p if p == Self::TENSOR_ABSENT_TAG => {}
                p if p == Self::TENSOR_PRESENT_TAG => {
                    cache.turboquant_shadow_layers[idx] =
                        Some(Self::read_turboquant_shadow_layer(&mut cursor)?);
                }
                other => {
                    return Err(MlxKVCacheSerializeError::InvalidTurboQuantState(format!(
                        "unknown TurboQuant layer-present tag {other}"
                    )));
                }
            }
        }

        Ok(cache)
    }

    pub fn record_turboquant_fused_decode_attempt(
        &mut self,
        outcome: MlxKvCompressionDecodeOutcome,
    ) {
        Self::bump_counter(&mut self.turboquant_decode_usage.fused_decode_attempts);
        match outcome {
            MlxKvCompressionDecodeOutcome::Fallback => {
                Self::bump_counter(&mut self.turboquant_decode_usage.fused_decode_fallbacks);
            }
            MlxKvCompressionDecodeOutcome::CpuOracle | MlxKvCompressionDecodeOutcome::Metal => {
                Self::bump_counter(&mut self.turboquant_decode_usage.fused_decode_successes);
            }
        }
        if matches!(outcome, MlxKvCompressionDecodeOutcome::Metal) {
            Self::bump_counter(&mut self.turboquant_decode_usage.fused_decode_metal_successes);
        }
    }

    pub fn record_turboquant_fused_decode_timing(
        &mut self,
        timing: MlxKvCompressionFusedDecodeTiming,
    ) {
        self.turboquant_decode_usage
            .fused_decode_query_readback_wall_us = self
            .turboquant_decode_usage
            .fused_decode_query_readback_wall_us
            .saturating_add(timing.query_readback_wall_us);
        self.turboquant_decode_usage.fused_decode_cold_metal_wall_us = self
            .turboquant_decode_usage
            .fused_decode_cold_metal_wall_us
            .saturating_add(timing.cold_metal_wall_us);
        self.turboquant_decode_usage
            .fused_decode_hot_tail_merge_wall_us = self
            .turboquant_decode_usage
            .fused_decode_hot_tail_merge_wall_us
            .saturating_add(timing.hot_tail_merge_wall_us);
        self.turboquant_decode_usage
            .fused_decode_output_staging_wall_us = self
            .turboquant_decode_usage
            .fused_decode_output_staging_wall_us
            .saturating_add(timing.output_staging_wall_us);
    }

    pub fn record_turboquant_decode_candidate(
        &mut self,
        candidate: MlxKvCompressionDecodeCandidate,
    ) {
        if matches!(
            candidate,
            MlxKvCompressionDecodeCandidate::LinearAttention
                | MlxKvCompressionDecodeCandidate::SlidingWindow
                | MlxKvCompressionDecodeCandidate::KvShared
        ) {
            Self::bump_counter(
                &mut self
                    .turboquant_decode_usage
                    .fused_decode_blocked_attention_kind,
            );
        }
        if let Some(counter) =
            Self::turboquant_candidate_counter(&mut self.turboquant_decode_usage, candidate)
        {
            Self::bump_counter(counter);
        }
    }

    pub fn take_turboquant_decode_usage(&mut self) -> MlxKvCompressionDecodeUsage {
        std::mem::take(&mut self.turboquant_decode_usage)
    }

    /// Append new K/V tokens for `layer` and return the full logical K/V for SDPA.
    ///
    /// `new_k` / `new_v` shape: `[1, n_kv_heads, new_tokens, head_dim]`
    ///
    /// Returns **owned** arrays sliced to `[1, n_kv_heads, seq_len + new_tokens, head_dim]`.
    pub fn append(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
    ) -> (MlxArray, MlxArray) {
        self.append_with_retained_window(layer, new_k, new_v, None)
    }

    /// Append new K/V tokens and return a logical view retained to `window` tokens.
    ///
    /// This is used for Gemma-family sliding-window decode.  Upstream `mlx_lm`
    /// and `mlx-swift-lm` use rotating caches for sliding layers, so SDPA only
    /// sees the retained window instead of the full context.  AX uses the same
    /// bounded backing store only when the request is on the non-rollback direct
    /// decode path; rollback-capable paths keep full backing storage and return a
    /// shorter view.
    ///
    /// `window = None` preserves the full-view behavior of `append`.
    pub fn append_with_retained_window(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        window: Option<usize>,
    ) -> (MlxArray, MlxArray) {
        let append = validate_append_inputs(layer, self.layers.len(), &new_k, &new_v);
        let new_tokens = append.new_tokens;
        let write_start = self.seq_len;
        let write_end = write_start + new_tokens;
        let n_kv_heads = append.n_kv_heads;
        let head_dim = append.head_dim;
        let dtype = append.dtype;

        if let Some(ring) = self.sliding_ring_layout(window, new_tokens)
            && self.layers[layer].is_some()
        {
            return self.append_rotating_retained_window(layer, new_k, new_v, ring);
        }

        let entry = &mut self.layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                // Fresh-layer fast path: when the prompt is chunk-aligned
                // (capacity == new_tokens), skip zeros+slice_update by storing
                // new_k/new_v directly.  Capacity stays correct for usage_snapshot;
                // the first decode step grows to chunk_ceiling(new_tokens + 1) as
                // normal.  Saves ~6 MLX graph nodes per layer per chunk-aligned prefill
                // (e.g. 512-token prompt → 210 fewer nodes for a 35-layer model).
                if write_start == 0 && capacity == new_tokens {
                    let view_start = window
                        .filter(|&w| w > 0 && w < new_tokens)
                        .map(|w| new_tokens - w)
                        .unwrap_or(0);
                    let (k_view, v_view) = if view_start > 0 {
                        let s = view_start as i32;
                        let e = new_tokens as i32;
                        let kv = slice(
                            &new_k,
                            &[0, 0, s, 0],
                            &[1, n_kv_heads, e, head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        let vv = slice(
                            &new_v,
                            &[0, 0, s, 0],
                            &[1, n_kv_heads, e, head_dim],
                            &[1, 1, 1, 1],
                            None,
                        );
                        (kv, vv)
                    } else {
                        (new_k.clone(), new_v.clone())
                    };
                    self.growth_count = self.growth_count.saturating_add(1);
                    *entry = Some(LayerKV {
                        k: new_k,
                        v: new_v,
                        last_k_view: Some(k_view.clone()),
                        last_v_view: Some(v_view.clone()),
                        n_kv_heads,
                        head_dim,
                        capacity,
                        rotating_window: None,
                        dtype,
                    });
                    return (k_view, v_view);
                }
                let buf_shape = [1i32, n_kv_heads, capacity as i32, head_dim];
                let k_buf = zeros(&buf_shape, dtype, None);
                let v_buf = zeros(&buf_shape, dtype, None);
                let start = [0i32, 0, write_start as i32, 0];
                let stop = [1i32, n_kv_heads, write_end as i32, head_dim];
                let strides = [1i32, 1, 1, 1];
                let k_out = slice_update(&k_buf, &new_k, &start, &stop, &strides, None);
                let v_out = slice_update(&v_buf, &new_v, &start, &stop, &strides, None);
                self.growth_count = self.growth_count.saturating_add(1);
                *entry = Some(LayerKV {
                    k: k_out,
                    v: v_out,
                    last_k_view: None,
                    last_v_view: None,
                    n_kv_heads,
                    head_dim,
                    capacity,
                    rotating_window: None,
                    dtype,
                });
            }
            Some(lkv) => {
                // A rotated ring stores slots, not token order; writing at
                // logical positions (or growing, which copies slots as an
                // ordered prefix) would silently corrupt it. Ring-eligible
                // forwards must go through `append_rotating_retained_window`;
                // anything else on a rotated layer is a caller bug.
                assert!(
                    lkv.rotating_window.is_none(),
                    "ordered KV append on rotated ring layer {layer} (window {:?}, capacity {}): \
                     forward of {new_tokens} tokens is not ring-eligible \
                     (rotating_slack {}) and would corrupt slot-ordered state",
                    lkv.rotating_window,
                    lkv.capacity,
                    self.rotating_slack,
                );
                assert_eq!(
                    lkv.n_kv_heads, n_kv_heads,
                    "KV cache append cannot change n_kv_heads for an existing layer"
                );
                assert_eq!(
                    lkv.head_dim, head_dim,
                    "KV cache append cannot change head_dim for an existing layer"
                );
                assert_eq!(
                    lkv.dtype, dtype,
                    "KV cache append cannot change dtype for an existing layer"
                );
                if write_end > lkv.capacity {
                    // Grow: allocate a larger buffer and copy existing data.
                    let new_capacity = chunk_ceiling(write_end);
                    let buf_shape = [1i32, lkv.n_kv_heads, new_capacity as i32, lkv.head_dim];
                    let k_new = zeros(&buf_shape, lkv.dtype, None);
                    let v_new = zeros(&buf_shape, lkv.dtype, None);
                    let old_stop = [1i32, lkv.n_kv_heads, lkv.capacity as i32, lkv.head_dim];
                    let zero_start = [0i32, 0, 0, 0];
                    let ones = [1i32, 1, 1, 1];
                    lkv.k = slice_update(&k_new, &lkv.k, &zero_start, &old_stop, &ones, None);
                    lkv.v = slice_update(&v_new, &lkv.v, &zero_start, &old_stop, &ones, None);
                    lkv.capacity = new_capacity;
                    // Invalidate cached views — they point to the old (smaller) buffer.
                    lkv.last_k_view = None;
                    lkv.last_v_view = None;
                    self.growth_count = self.growth_count.saturating_add(1);
                }
                let start = [0i32, 0, write_start as i32, 0];
                let stop = [1i32, lkv.n_kv_heads, write_end as i32, lkv.head_dim];
                let strides = [1i32, 1, 1, 1];
                lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
                lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
            }
        }

        let lkv = self.layers[layer].as_mut().unwrap();
        let view_start = window
            .filter(|window| *window > 0)
            .map(|window| write_end.saturating_sub(window))
            .unwrap_or(0);
        let start = view_start as i32;
        let end = write_end as i32;
        let k_view = slice(
            &lkv.k,
            &[0, 0, start, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_view = slice(
            &lkv.v,
            &[0, 0, start, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        // Cache the views so KV-shared layers (Gemma4) can reuse them without
        // creating a second identical slice node on the same backing buffer.
        lkv.last_k_view = Some(k_view.clone());
        lkv.last_v_view = Some(v_view.clone());
        (k_view, v_view)
    }

    fn append_rotating_retained_window(
        &mut self,
        layer: usize,
        new_k: MlxArray,
        new_v: MlxArray,
        ring: SlidingRingLayout,
    ) -> (MlxArray, MlxArray) {
        let SlidingRingLayout {
            window,
            capacity,
            write_start,
        } = ring;
        let new_tokens = new_k.shape()[2] as usize;
        let lkv = self.layers[layer]
            .as_mut()
            .expect("rotating sliding decode requires an existing prefill cache");
        if lkv.rotating_window != Some(window) || lkv.capacity != capacity {
            // Conversion reads the source at logical token indices, which is
            // only meaningful for ordered storage. An already-rotated layer
            // reaching here means the ring geometry changed mid-request
            // (window or slack drift) — reconverting would read slot-ordered
            // data as token order.
            assert!(
                lkv.rotating_window.is_none(),
                "sliding ring geometry changed mid-request for layer {layer}: \
                 existing window {:?} capacity {}, requested window {window} capacity {capacity}",
                lkv.rotating_window,
                lkv.capacity,
            );
            let k_old = lkv.k.clone();
            let v_old = lkv.v.clone();
            let buf_shape = [1i32, lkv.n_kv_heads, capacity as i32, lkv.head_dim];
            let k_new = zeros(&buf_shape, lkv.dtype, None);
            let v_new = zeros(&buf_shape, lkv.dtype, None);
            let old_start = write_start.saturating_add(1).saturating_sub(window);
            let old_end = write_start;
            let k_new =
                copy_token_range_to_rotating(&k_old, &k_new, lkv, old_start, old_end, capacity);
            let v_new =
                copy_token_range_to_rotating(&v_old, &v_new, lkv, old_start, old_end, capacity);
            lkv.k = k_new;
            lkv.v = v_new;
            lkv.capacity = capacity;
            lkv.rotating_window = Some(window);
            lkv.last_k_view = None;
            lkv.last_v_view = None;
        }

        if new_tokens == 1 {
            let write_pos = (write_start % capacity) as i32;
            let start = [0i32, 0, write_pos, 0];
            let stop = [1i32, lkv.n_kv_heads, write_pos + 1, lkv.head_dim];
            let strides = [1i32, 1, 1, 1];
            lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
            lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
            lkv.last_k_view = Some(lkv.k.clone());
            lkv.last_v_view = Some(lkv.v.clone());
            return (lkv.k.clone(), lkv.v.clone());
        }

        // Write the new tokens at their `t % capacity` slots. A multi-token
        // append (bounded rings only; `new_tokens <= slack < capacity`) wraps
        // at most once, so this loop issues one or two slice_updates.
        let mut src_start = 0usize;
        while src_start < new_tokens {
            let dst_start = (write_start + src_start) % capacity;
            let len = (new_tokens - src_start).min(capacity - dst_start);
            let (seg_k, seg_v) = if src_start == 0 && len == new_tokens {
                (new_k.clone(), new_v.clone())
            } else {
                let src_stop = (src_start + len) as i32;
                let seg_start = [0i32, 0, src_start as i32, 0];
                let seg_stop = [1i32, lkv.n_kv_heads, src_stop, lkv.head_dim];
                let ones = [1i32, 1, 1, 1];
                (
                    slice(&new_k, &seg_start, &seg_stop, &ones, None),
                    slice(&new_v, &seg_start, &seg_stop, &ones, None),
                )
            };
            let start = [0i32, 0, dst_start as i32, 0];
            let stop = [1i32, lkv.n_kv_heads, (dst_start + len) as i32, lkv.head_dim];
            let strides = [1i32, 1, 1, 1];
            lkv.k = slice_update(&lkv.k, &seg_k, &start, &stop, &strides, None);
            lkv.v = slice_update(&lkv.v, &seg_v, &start, &stop, &strides, None);
            src_start += len;
        }
        lkv.last_k_view = Some(lkv.k.clone());
        lkv.last_v_view = Some(lkv.v.clone());
        (lkv.k.clone(), lkv.v.clone())
    }

    /// Append GLM4MoELite MLA cache tokens and return full logical latent/KRoPE views.
    ///
    /// Shape convention follows mlx-lm's `cache.update_and_fetch(kv_latent, k_pe)`:
    /// `new_kv_latent`: `[1, 1, new_tokens, kv_lora_rank]`
    /// `new_k_pe`: `[1, 1, new_tokens, qk_rope_head_dim]`
    ///
    /// This cache stores the compressed MLA representation, not expanded K/V.
    pub fn append_glm_mla(
        &mut self,
        layer: usize,
        new_kv_latent: MlxArray,
        new_k_pe: MlxArray,
    ) -> (MlxArray, MlxArray) {
        let append = validate_glm_mla_append_inputs(
            layer,
            self.glm_mla_layers.len(),
            &new_kv_latent,
            &new_k_pe,
        );
        let new_tokens = append.new_tokens;
        let write_start = self.seq_len;
        let write_end = write_start + new_tokens;
        let dtype = append.dtype;
        let latent_dim = append.latent_dim;
        let rope_dim = append.rope_dim;

        let entry = &mut self.glm_mla_layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
                if write_start == 0 && capacity == new_tokens {
                    self.growth_count = self.growth_count.saturating_add(1);
                    *entry = Some(GlmMlaLayerCache {
                        kv_latent: new_kv_latent.clone(),
                        k_pe: new_k_pe.clone(),
                        latent_dim,
                        rope_dim,
                        capacity,
                        dtype,
                    });
                    return (new_kv_latent, new_k_pe);
                }
                let latent_shape = [1i32, 1, capacity as i32, latent_dim];
                let rope_shape = [1i32, 1, capacity as i32, rope_dim];
                let latent_buf = zeros(&latent_shape, dtype, None);
                let rope_buf = zeros(&rope_shape, dtype, None);
                let start = [0i32, 0, write_start as i32, 0];
                let latent_stop = [1i32, 1, write_end as i32, latent_dim];
                let rope_stop = [1i32, 1, write_end as i32, rope_dim];
                let strides = [1i32, 1, 1, 1];
                let kv_latent = slice_update(
                    &latent_buf,
                    &new_kv_latent,
                    &start,
                    &latent_stop,
                    &strides,
                    None,
                );
                let k_pe = slice_update(&rope_buf, &new_k_pe, &start, &rope_stop, &strides, None);
                self.growth_count = self.growth_count.saturating_add(1);
                *entry = Some(GlmMlaLayerCache {
                    kv_latent,
                    k_pe,
                    latent_dim,
                    rope_dim,
                    capacity,
                    dtype,
                });
            }
            Some(cache) => {
                assert_eq!(
                    cache.latent_dim, latent_dim,
                    "GLM MLA cache append cannot change kv_lora_rank for an existing layer"
                );
                assert_eq!(
                    cache.rope_dim, rope_dim,
                    "GLM MLA cache append cannot change qk_rope_head_dim for an existing layer"
                );
                assert_eq!(
                    cache.dtype, dtype,
                    "GLM MLA cache append cannot change dtype for an existing layer"
                );
                if write_end > cache.capacity {
                    let new_capacity = chunk_ceiling(write_end);
                    let latent_shape = [1i32, 1, new_capacity as i32, cache.latent_dim];
                    let rope_shape = [1i32, 1, new_capacity as i32, cache.rope_dim];
                    let latent_new = zeros(&latent_shape, cache.dtype, None);
                    let rope_new = zeros(&rope_shape, cache.dtype, None);
                    let zero_start = [0i32, 0, 0, 0];
                    let latent_old_stop = [1i32, 1, cache.capacity as i32, cache.latent_dim];
                    let rope_old_stop = [1i32, 1, cache.capacity as i32, cache.rope_dim];
                    let ones = [1i32, 1, 1, 1];
                    cache.kv_latent = slice_update(
                        &latent_new,
                        &cache.kv_latent,
                        &zero_start,
                        &latent_old_stop,
                        &ones,
                        None,
                    );
                    cache.k_pe = slice_update(
                        &rope_new,
                        &cache.k_pe,
                        &zero_start,
                        &rope_old_stop,
                        &ones,
                        None,
                    );
                    cache.capacity = new_capacity;
                    self.growth_count = self.growth_count.saturating_add(1);
                }
                let start = [0i32, 0, write_start as i32, 0];
                let latent_stop = [1i32, 1, write_end as i32, cache.latent_dim];
                let rope_stop = [1i32, 1, write_end as i32, cache.rope_dim];
                let strides = [1i32, 1, 1, 1];
                cache.kv_latent = slice_update(
                    &cache.kv_latent,
                    &new_kv_latent,
                    &start,
                    &latent_stop,
                    &strides,
                    None,
                );
                cache.k_pe =
                    slice_update(&cache.k_pe, &new_k_pe, &start, &rope_stop, &strides, None);
            }
        }

        let cache = self.glm_mla_layers[layer].as_ref().unwrap();
        let end = write_end as i32;
        let kv_latent = slice(
            &cache.kv_latent,
            &[0, 0, 0, 0],
            &[1, 1, end, cache.latent_dim],
            &[1, 1, 1, 1],
            None,
        );
        let k_pe = slice(
            &cache.k_pe,
            &[0, 0, 0, 0],
            &[1, 1, end, cache.rope_dim],
            &[1, 1, 1, 1],
            None,
        );
        (kv_latent, k_pe)
    }

    /// Trim the logical boundary to `prefix_len` tokens (draft rollback).
    ///
    /// With chunked layout this is O(1) — no array data is modified.  The backing
    /// buffer retains its pre-allocated capacity.  The next `append` writes from
    /// `prefix_len`, overwriting any rejected draft positions.
    ///
    /// Returns `true` when the requested trim point was valid.  Invalid requests
    /// are clamped to the current logical length so a release build cannot extend
    /// the cache and make SDPA attend to unwritten positions.
    #[must_use]
    pub fn trim_to(&mut self, prefix_len: usize) -> bool {
        if prefix_len < self.seq_len {
            // Rotated layers can absorb a rollback only within their slack:
            // token `t` lives at slot `t % capacity`, so a token is still
            // resident iff nothing more than `capacity` positions newer has
            // overwritten it. After trimming to `L` with pre-trim end `E`,
            // the next forward reads keys back to `L - window + 1`; those
            // are intact iff `E - L <= capacity - window`. Pure rings
            // (`capacity == window`) therefore refuse every real trim, and
            // bounded rings refuse trims deeper than their slack.
            let rollback = self.seq_len - prefix_len;
            if self.layers.iter().flatten().any(|lkv| {
                lkv.rotating_window
                    .is_some_and(|window| rollback > lkv.capacity.saturating_sub(window))
            }) {
                return false;
            }
        }
        let valid = prefix_len <= self.seq_len;
        let trimmed = prefix_len < self.seq_len;
        self.seq_len = prefix_len.min(self.seq_len);
        if trimmed {
            // The retained fast-path views still span the pre-trim write end,
            // including the rejected draft positions.  Drop them so any
            // consumer between this trim and the next append re-slices from
            // the logical boundary instead of attending over trimmed tokens.
            for lkv in self.layers.iter_mut().flatten() {
                lkv.last_k_view = None;
                lkv.last_v_view = None;
            }
        }
        for storage in self.turboquant_shadow_layers.iter_mut().flatten() {
            let compressed_tokens = storage.compressed_tokens.min(self.seq_len);
            if compressed_tokens < storage.compressed_tokens {
                storage
                    .buffer
                    .truncate_token_count(compressed_tokens)
                    .expect("TurboQuant shadow trim uses valid compressed token prefix");
                storage.metal_buffer = None;
            }
            storage.compressed_tokens = compressed_tokens;
        }
        valid
    }

    /// Logical K/V view for `layer`, sliced to the current `seq_len`, or `None`
    /// if the layer has no entry yet.
    ///
    /// Used to seed a **pure** compiled MTP draft closure: the existing context
    /// is passed in as explicit closure inputs rather than read by capturing the
    /// mutable cache, which would make the compiled graph impure (the captured
    /// lazy KV would enter the trace as an un-passed constant and abort eval
    /// with "Attempting to eval an array without a primitive").
    pub fn logical_layer_kv(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let lkv = self.layers.get(layer)?.as_ref()?;
        debug_assert!(
            lkv.rotating_window.is_none(),
            "logical_layer_kv reads a [0, seq_len) prefix slice, which is meaningless \
             on a rotated ring (MTP draft seeding never coexists with rotation)"
        );
        let end = self.seq_len as i32;
        let stop = [1, lkv.n_kv_heads, end, lkv.head_dim];
        let k = slice(&lkv.k, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
        let v = slice(&lkv.v, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1], None);
        Some((k, v))
    }

    /// Commit a pure compiled MTP draft closure's threaded K/V back into the
    /// cache as a tight logical buffer (`capacity == length`) and set `seq_len`.
    ///
    /// `k`/`v` are the closure's final concatenated `[1, n_kv_heads, length,
    /// head_dim]` outputs (already evaluated).  Storing them tight is correct:
    /// the next imperative `append` sees `write_end > capacity` and grows via
    /// its normal chunk path, copying this buffer forward.  Replacing the layer
    /// entry also drops any stale views.
    pub fn set_layer_kv_logical(&mut self, layer: usize, k: MlxArray, v: MlxArray, seq_len: usize) {
        let shape = k.shape();
        debug_assert_eq!(shape.len(), 4, "set_layer_kv_logical expects a 4D K array");
        let n_kv_heads = shape[1];
        let length = shape[2] as usize;
        let head_dim = shape[3];
        let dtype = k.dtype();
        self.layers[layer] = Some(LayerKV {
            last_k_view: Some(k.clone()),
            last_v_view: Some(v.clone()),
            k,
            v,
            n_kv_heads,
            head_dim,
            capacity: length,
            rotating_window: None,
            dtype,
        });
        self.seq_len = seq_len;
    }

    pub fn sync_turboquant_shadow_storage(
        &mut self,
        layer_windows: &[Option<usize>],
        compression: KvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> TurboQuantShadowStorageUsage {
        let Some(cold_tokens) = self.turboquant_shadow_sync_cold_tokens(compression) else {
            self.clear_turboquant_shadow_storage();
            return TurboQuantShadowStorageUsage::default();
        };

        let mut sync_sources = Vec::new();
        for layer_idx in 0..self.layers.len() {
            let Some(layout) = self.turboquant_shadow_layer_sync_layout(
                layer_idx,
                layer_windows,
                compression,
                compression_eligible_layers,
            ) else {
                self.turboquant_shadow_layers[layer_idx] = None;
                continue;
            };

            let reset_storage = self.turboquant_shadow_layers[layer_idx]
                .as_ref()
                .map(|storage| storage.layout != layout || storage.compressed_tokens > cold_tokens)
                .unwrap_or(true);
            if reset_storage {
                self.turboquant_shadow_layers[layer_idx] = Some(TurboQuantShadowLayerStorage {
                    layout,
                    buffer: TurboQuantCompressedBlockBuffer::new(layout),
                    compressed_tokens: 0,
                    metal_buffer: None,
                });
            }

            let compressed_tokens = self.turboquant_shadow_layers[layer_idx]
                .as_ref()
                .expect("storage was just initialized")
                .compressed_tokens;
            if compressed_tokens >= cold_tokens {
                continue;
            }

            let lkv = self.layers[layer_idx]
                .as_ref()
                .expect("TurboQuant sync layout requires layer KV storage");
            // Materialize only the new cold range [compressed_tokens,
            // cold_tokens): the backing buffer is capacity-sized, so the
            // previous whole-buffer f32 conversion cost one full-context
            // astype allocation + kernel per sync instead of scaling with
            // the tokens actually being compressed.
            let delta_start = [0i32, 0, compressed_tokens as i32, 0];
            let delta_stop = [1i32, lkv.n_kv_heads, cold_tokens as i32, lkv.head_dim];
            let ones = [1i32, 1, 1, 1];
            let k = contiguous(&slice(&lkv.k, &delta_start, &delta_stop, &ones, None), None);
            let v = contiguous(&slice(&lkv.v, &delta_start, &delta_stop, &ones, None), None);
            let (k, v) = if lkv.dtype == MlxDtype::Float32 {
                (k, v)
            } else {
                (
                    astype(&k, MlxDtype::Float32, None),
                    astype(&v, MlxDtype::Float32, None),
                )
            };
            sync_sources.push(TurboQuantShadowSyncSource {
                layer_idx,
                // The source arrays hold just the delta, so token indices
                // into them are relative to `compressed_tokens` and the
                // slice stride ("capacity") is the delta length.
                shape: KvHeadSliceShape {
                    n_kv_heads: lkv.n_kv_heads,
                    head_dim: lkv.head_dim,
                    capacity: cold_tokens - compressed_tokens,
                },
                compressed_tokens,
                k,
                v,
            });
        }

        if sync_sources.is_empty() {
            return self.turboquant_shadow_storage_usage();
        }

        let eval_targets = sync_sources
            .iter()
            .flat_map(|source| [&source.k, &source.v])
            .collect::<Vec<_>>();
        eval(&eval_targets);

        for source in sync_sources {
            let storage = self.turboquant_shadow_layers[source.layer_idx]
                .as_mut()
                .expect("storage was just initialized");
            let preencoded_k8_keys = if storage.layout.config.preset == TurboQuantPreset::K8V4 {
                // The source array holds only the delta, so encoding
                // starts at its token 0.
                turboquant_fused_key_encode_metal_k8(
                    &source.k,
                    0,
                    cold_tokens.saturating_sub(source.compressed_tokens),
                )
                .ok()
            } else {
                None
            };
            let k_values = source.k.data_f32();
            let v_values = source.v.data_f32();
            for token_index in source.compressed_tokens..cold_tokens {
                let relative_token_index = token_index.saturating_sub(source.compressed_tokens);
                let heads = kv_heads_for_token_from_f32_slices(
                    k_values,
                    v_values,
                    source.shape,
                    relative_token_index,
                )
                .expect("TurboQuant shadow sync reads KV slices within layer capacity");
                if let Some(encoded_keys) = preencoded_k8_keys.as_ref() {
                    let key_bytes_per_token = storage
                        .layout
                        .key_payload_bytes_per_head
                        .saturating_mul(storage.layout.config.n_kv_heads);
                    let key_start = relative_token_index.saturating_mul(key_bytes_per_token);
                    let key_end = key_start.saturating_add(key_bytes_per_token);
                    let norm_start =
                        relative_token_index.saturating_mul(storage.layout.config.n_kv_heads);
                    let norm_end = norm_start.saturating_add(storage.layout.config.n_kv_heads);
                    if let (Some(packed_key_bytes), Some(key_norms)) = (
                        encoded_keys.packed_key_bytes.get(key_start..key_end),
                        encoded_keys.key_norms.get(norm_start..norm_end),
                    ) && storage
                        .buffer
                        .write_token_with_encoded_k8_keys(
                            token_index,
                            &heads,
                            packed_key_bytes,
                            key_norms,
                        )
                        .is_ok()
                    {
                        continue;
                    }
                }
                storage
                    .buffer
                    .write_token(token_index, &heads)
                    .expect("TurboQuant shadow storage writes validated KV slots");
            }
            storage.compressed_tokens = cold_tokens;
            let existing = storage.metal_buffer.take();
            storage.metal_buffer = Some(turboquant_shadow_metal_buffer_update(
                existing,
                &storage.layout,
                &storage.buffer,
                source.compressed_tokens,
                cold_tokens,
            ));
        }

        self.turboquant_shadow_storage_usage()
    }

    pub fn turboquant_shadow_storage_sync_due(
        &self,
        layer_windows: &[Option<usize>],
        compression: KvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> bool {
        if !compression.is_enabled() {
            return false;
        }

        let Some(cold_tokens) = self.turboquant_shadow_sync_cold_tokens(compression) else {
            return self.turboquant_shadow_layers.iter().any(Option::is_some);
        };

        for layer_idx in 0..self.layers.len() {
            let Some(layout) = self.turboquant_shadow_layer_sync_layout(
                layer_idx,
                layer_windows,
                compression,
                compression_eligible_layers,
            ) else {
                if self.turboquant_shadow_layers[layer_idx].is_some() {
                    return true;
                }
                continue;
            };

            let Some(storage) = self.turboquant_shadow_layers[layer_idx].as_ref() else {
                return true;
            };
            if storage.layout != layout || storage.compressed_tokens > cold_tokens {
                return true;
            }
            let cold_delta = cold_tokens.saturating_sub(storage.compressed_tokens);
            if (fastpath::turboquant_incremental_decode_enabled() && (1..=4).contains(&cold_delta))
                || cold_delta >= KV_CHUNK_TOKENS
            {
                return true;
            }
        }

        false
    }

    fn turboquant_shadow_sync_cold_tokens(
        &self,
        compression: KvCompressionConfig,
    ) -> Option<usize> {
        if !compression.is_enabled() || self.seq_len < compression.min_context_tokens {
            return None;
        }
        let cold_tokens = self.seq_len.saturating_sub(compression.hot_window_tokens);
        (cold_tokens != 0).then_some(cold_tokens)
    }

    fn turboquant_shadow_layer_sync_layout(
        &self,
        layer_idx: usize,
        layer_windows: &[Option<usize>],
        compression: KvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> Option<TurboQuantBlockLayout> {
        if layer_windows.get(layer_idx).copied().flatten().is_some() {
            return None;
        }
        if let Some(eligible_layers) = compression_eligible_layers
            && !eligible_layers.get(layer_idx).copied().unwrap_or(false)
        {
            return None;
        }
        let lkv = self.layers.get(layer_idx).and_then(Option::as_ref)?;
        turboquant_shadow_layout_for_layer(lkv, compression)
    }

    pub fn turboquant_shadow_storage_usage(&self) -> TurboQuantShadowStorageUsage {
        let mut usage = TurboQuantShadowStorageUsage::default();
        for storage in self.turboquant_shadow_layers.iter().flatten() {
            usage.layers = usage.layers.saturating_add(1);
            usage.token_layers = usage.token_layers.saturating_add(storage.compressed_tokens);
            usage.bytes = usage
                .bytes
                .saturating_add(storage.buffer.as_bytes().len() as u64);
            usage.written_slots = usage
                .written_slots
                .saturating_add(storage.buffer.written_slot_count());
        }
        usage
    }

    pub fn turboquant_shadow_storage_cold_tokens(&self, layer: usize) -> Option<usize> {
        self.turboquant_shadow_layers
            .get(layer)
            .and_then(Option::as_ref)
            .map(|storage| storage.compressed_tokens)
    }

    pub fn debug_turboquant_shadow_decode_attention_for_layer(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        hot_window_tokens: usize,
    ) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
        self.debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
            layer,
            queries,
            hot_window_tokens,
            self.seq_len,
        )
    }

    pub fn debug_turboquant_shadow_decode_attention_for_layer_with_total_tokens(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        _hot_window_tokens: usize,
        total_tokens: usize,
    ) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
        let cold_stats =
            self.debug_turboquant_shadow_decode_cold_stats_cpu(layer, queries, total_tokens)?;
        self.merge_turboquant_shadow_decode_cold_stats_with_hot_tail(
            layer,
            queries,
            total_tokens,
            &cold_stats,
        )
    }

    pub fn debug_turboquant_shadow_decode_attention_metal_for_layer_with_total_tokens(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        total_tokens: usize,
    ) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
        self.debug_turboquant_shadow_decode_attention_metal_timed_for_layer_with_total_tokens(
            layer,
            queries,
            total_tokens,
        )
        .map(|attention| attention.outputs)
    }

    pub fn debug_turboquant_shadow_decode_attention_metal_timed_for_layer_with_total_tokens(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        total_tokens: usize,
    ) -> Result<MlxTurboQuantFusedDecodeAttention, TurboQuantCodecError> {
        let attention = self
            .debug_turboquant_shadow_decode_attention_metal_flat_timed_for_layer_with_total_tokens(
                layer,
                queries,
                total_tokens,
            )?;
        let head_dim = queries.first().map(Vec::len).unwrap_or(0);
        if head_dim == 0 || attention.outputs.len() != queries.len().saturating_mul(head_dim) {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: queries.len().saturating_mul(head_dim),
                actual: attention.outputs.len(),
            });
        }
        Ok(MlxTurboQuantFusedDecodeAttention {
            outputs: attention
                .outputs
                .chunks_exact(head_dim)
                .map(|head| head.to_vec())
                .collect(),
            timing: attention.timing,
        })
    }

    pub fn debug_turboquant_shadow_decode_attention_metal_flat_timed_for_layer_with_total_tokens(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        total_tokens: usize,
    ) -> Result<MlxTurboQuantFusedDecodeFlatAttention, TurboQuantCodecError> {
        let head_dim = queries.first().map(Vec::len).unwrap_or(0);
        if head_dim == 0 || queries.iter().any(|query| query.len() != head_dim) {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: head_dim,
                actual: queries
                    .iter()
                    .find(|query| query.len() != head_dim)
                    .map(Vec::len)
                    .unwrap_or(0),
            });
        }
        let query_values = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect::<Vec<_>>();
        self.debug_turboquant_shadow_decode_attention_metal_flat_query_timed_for_layer_with_total_tokens(
            layer,
            &query_values,
            queries.len(),
            total_tokens,
        )
    }

    pub fn debug_turboquant_shadow_decode_attention_metal_flat_query_timed_for_layer_with_total_tokens(
        &self,
        layer: usize,
        query_values: &[f32],
        n_query_heads: usize,
        total_tokens: usize,
    ) -> Result<MlxTurboQuantFusedDecodeFlatAttention, TurboQuantCodecError> {
        let cold_started = Instant::now();
        let cold_stats = self.debug_turboquant_shadow_decode_cold_stats_metal_flat(
            layer,
            query_values,
            n_query_heads,
            total_tokens,
        )?;
        let cold_metal_wall_us = elapsed_us_u64(cold_started);

        let merge_started = Instant::now();
        let outputs = self.merge_turboquant_shadow_decode_cold_stats_with_hot_tail_flat(
            layer,
            query_values,
            n_query_heads,
            total_tokens,
            &cold_stats,
        )?;
        let hot_tail_merge_wall_us = elapsed_us_u64(merge_started);

        Ok(MlxTurboQuantFusedDecodeFlatAttention {
            outputs,
            timing: MlxKvCompressionFusedDecodeTiming {
                cold_metal_wall_us,
                hot_tail_merge_wall_us,
                ..MlxKvCompressionFusedDecodeTiming::default()
            },
        })
    }

    fn debug_turboquant_shadow_decode_cold_stats_cpu(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        total_tokens: usize,
    ) -> Result<Vec<TurboQuantAttentionPartitionStats>, TurboQuantCodecError> {
        let lkv = self.layers.get(layer).and_then(Option::as_ref).ok_or(
            TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots: 1,
                written_slots: 0,
            },
        )?;
        let storage = self
            .turboquant_shadow_layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or(TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots: 1,
                written_slots: 0,
            })?;
        let n_kv_heads = usize::try_from(lkv.n_kv_heads).unwrap_or(usize::MAX);
        if queries.is_empty() || !queries.len().is_multiple_of(n_kv_heads) {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: n_kv_heads,
                actual: queries.len(),
            });
        }

        let cold_tokens = storage.compressed_tokens;
        if cold_tokens == 0 {
            return Err(TurboQuantCodecError::EmptyKvHistory);
        }
        if cold_tokens > total_tokens {
            let required_slots = total_tokens.saturating_mul(n_kv_heads);
            return Err(TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots,
                written_slots: storage.compressed_tokens.saturating_mul(n_kv_heads),
            });
        }
        let cold_stats = storage
            .buffer
            .debug_decode_partition_stats_for_all_heads(queries, cold_tokens)?;
        Ok(cold_stats)
    }

    fn debug_turboquant_shadow_decode_cold_stats_metal_flat(
        &self,
        layer: usize,
        query_values: &[f32],
        n_query_heads: usize,
        total_tokens: usize,
    ) -> Result<TurboQuantAttentionPartitionStatsBatch, TurboQuantCodecError> {
        let lkv = self.layers.get(layer).and_then(Option::as_ref).ok_or(
            TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots: 1,
                written_slots: 0,
            },
        )?;
        let storage = self
            .turboquant_shadow_layers
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or(TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots: 1,
                written_slots: 0,
            })?;
        let n_kv_heads = usize::try_from(lkv.n_kv_heads).unwrap_or(usize::MAX);
        let head_dim = usize::try_from(lkv.head_dim).unwrap_or(usize::MAX);
        if n_query_heads == 0 || !n_query_heads.is_multiple_of(n_kv_heads) {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: n_kv_heads,
                actual: n_query_heads,
            });
        }
        let expected_query_values = n_query_heads.saturating_mul(head_dim);
        if query_values.len() != expected_query_values {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: expected_query_values,
                actual: query_values.len(),
            });
        }

        let cold_tokens = storage.compressed_tokens;
        if cold_tokens == 0 {
            return Err(TurboQuantCodecError::EmptyKvHistory);
        }
        if cold_tokens > total_tokens {
            let required_slots = total_tokens.saturating_mul(n_kv_heads);
            return Err(TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots,
                written_slots: storage.compressed_tokens.saturating_mul(n_kv_heads),
            });
        }
        let plan = TurboQuantCompressedDecodePlan::new(
            storage.layout,
            total_tokens,
            total_tokens.saturating_sub(cold_tokens),
        )?;
        let descriptor =
            plan.fused_decode_launch_descriptor_for_query_count(&storage.buffer, n_query_heads)?;
        let sparse_value_threshold =
            fastpath::turboquant_sparse_v_threshold_for_context(total_tokens);
        if let Some(metal_buffer) = storage.metal_buffer.as_ref() {
            turboquant_fused_cold_decode_metal_two_stage_partition_stats_batch_with_compressed_array_flat_sparse_threshold(
                descriptor,
                metal_buffer,
                query_values,
                n_query_heads,
                sparse_value_threshold,
            )
        } else {
            let stats = if sparse_value_threshold > 0.0 {
                turboquant_fused_cold_decode_metal_two_stage_sparse_partition_stats_flat(
                    descriptor,
                    &storage.buffer,
                    query_values,
                    n_query_heads,
                    sparse_value_threshold,
                )?
            } else {
                turboquant_fused_cold_decode_metal_two_stage_partition_stats_flat(
                    descriptor,
                    &storage.buffer,
                    query_values,
                    n_query_heads,
                )?
            };
            Ok(TurboQuantAttentionPartitionStatsBatch {
                token_count: cold_tokens,
                value_dim: head_dim,
                max_scores: stats.iter().map(|stats| stats.max_score).collect(),
                exp_sums: stats.iter().map(|stats| stats.exp_sum).collect(),
                weighted_value_sums: stats
                    .into_iter()
                    .flat_map(|stats| stats.weighted_value_sum)
                    .collect(),
            })
        }
    }

    fn merge_turboquant_shadow_decode_cold_stats_with_hot_tail(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        total_tokens: usize,
        cold_stats: &[TurboQuantAttentionPartitionStats],
    ) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
        let head_dim = queries.first().map(Vec::len).unwrap_or(0);
        if head_dim == 0 || queries.iter().any(|query| query.len() != head_dim) {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: head_dim,
                actual: queries
                    .iter()
                    .find(|query| query.len() != head_dim)
                    .map(Vec::len)
                    .unwrap_or(0),
            });
        }
        let query_values = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect::<Vec<_>>();
        let cold_tokens = cold_stats
            .first()
            .map(|stats| stats.token_count)
            .unwrap_or(0);
        if cold_stats
            .iter()
            .any(|stats| stats.token_count != cold_tokens)
        {
            return Err(TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens,
                required_slots: cold_tokens.saturating_mul(queries.len()),
                written_slots: cold_stats
                    .iter()
                    .map(|stats| stats.token_count)
                    .min()
                    .unwrap_or(0)
                    .saturating_mul(queries.len()),
            });
        }
        if cold_stats.iter().any(|stats| stats.value_dim != head_dim) {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: head_dim,
                actual: cold_stats
                    .iter()
                    .find(|stats| stats.value_dim != head_dim)
                    .map(|stats| stats.value_dim)
                    .unwrap_or(0),
            });
        }
        let cold_stats_batch = TurboQuantAttentionPartitionStatsBatch {
            token_count: cold_tokens,
            value_dim: head_dim,
            max_scores: cold_stats.iter().map(|stats| stats.max_score).collect(),
            exp_sums: cold_stats.iter().map(|stats| stats.exp_sum).collect(),
            weighted_value_sums: cold_stats
                .iter()
                .flat_map(|stats| stats.weighted_value_sum.iter().copied())
                .collect(),
        };
        let flat = self.merge_turboquant_shadow_decode_cold_stats_with_hot_tail_flat(
            layer,
            &query_values,
            queries.len(),
            total_tokens,
            &cold_stats_batch,
        )?;
        if head_dim == 0 || flat.len() != queries.len().saturating_mul(head_dim) {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: queries.len().saturating_mul(head_dim),
                actual: flat.len(),
            });
        }
        Ok(flat
            .chunks_exact(head_dim)
            .map(|head| head.to_vec())
            .collect())
    }

    fn merge_turboquant_shadow_decode_cold_stats_with_hot_tail_flat(
        &self,
        layer: usize,
        query_values: &[f32],
        n_query_heads: usize,
        total_tokens: usize,
        cold_stats: &TurboQuantAttentionPartitionStatsBatch,
    ) -> Result<Vec<f32>, TurboQuantCodecError> {
        let lkv = self.layers.get(layer).and_then(Option::as_ref).ok_or(
            TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots: 1,
                written_slots: 0,
            },
        )?;
        let n_kv_heads = usize::try_from(lkv.n_kv_heads).unwrap_or(usize::MAX);
        let cold_stats = cold_stats.validated()?;
        if cold_stats.query_heads() != n_query_heads
            || n_query_heads == 0
            || !n_query_heads.is_multiple_of(n_kv_heads)
        {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: n_query_heads,
                actual: cold_stats.query_heads(),
            });
        }
        let cold_tokens = cold_stats.token_count();
        let hot_token_count = total_tokens.saturating_sub(cold_tokens);
        let hot_k;
        let hot_v;
        let hot_k_compact;
        let hot_v_compact;
        let k_f32;
        let v_f32;
        let (k_values, v_values) = if hot_token_count == 0 {
            (&[][..], &[][..])
        } else {
            hot_k = slice(
                &lkv.k,
                &[0, 0, cold_tokens as i32, 0],
                &[1, lkv.n_kv_heads, total_tokens as i32, lkv.head_dim],
                &[1, 1, 1, 1],
                None,
            );
            hot_v = slice(
                &lkv.v,
                &[0, 0, cold_tokens as i32, 0],
                &[1, lkv.n_kv_heads, total_tokens as i32, lkv.head_dim],
                &[1, 1, 1, 1],
                None,
            );
            hot_k_compact = contiguous(&hot_k, None);
            hot_v_compact = contiguous(&hot_v, None);
            k_f32 = (lkv.dtype != MlxDtype::Float32)
                .then(|| astype(&hot_k_compact, MlxDtype::Float32, None));
            v_f32 = (lkv.dtype != MlxDtype::Float32)
                .then(|| astype(&hot_v_compact, MlxDtype::Float32, None));
            let k_source = k_f32.as_ref().unwrap_or(&hot_k_compact);
            let v_source = v_f32.as_ref().unwrap_or(&hot_v_compact);
            eval(&[k_source, v_source]);
            (k_source.data_f32(), v_source.data_f32())
        };
        let head_dim = usize::try_from(lkv.head_dim).unwrap_or(usize::MAX);
        let expected_query_values = n_query_heads.saturating_mul(head_dim);
        if query_values.len() != expected_query_values {
            return Err(TurboQuantCodecError::MismatchedVectorDimension {
                expected: expected_query_values,
                actual: query_values.len(),
            });
        }

        let mut outputs = Vec::with_capacity(n_query_heads.saturating_mul(head_dim));
        for (query_head_index, query) in query_values.chunks_exact(head_dim).enumerate() {
            let kv_head_index =
                turboquant_query_head_to_kv_head(query_head_index, n_query_heads, n_kv_heads)?;
            append_attention_partition_stats_batch_head_with_hot_tail_from_f32_slices(
                &mut outputs,
                cold_stats,
                query_head_index,
                query,
                k_values,
                v_values,
                n_kv_heads,
                head_dim,
                hot_token_count,
                kv_head_index,
            )?;
        }

        Ok(outputs)
    }

    fn clear_turboquant_shadow_storage(&mut self) {
        for storage in &mut self.turboquant_shadow_layers {
            *storage = None;
        }
    }

    /// Collect refs to all K and V backing buffers for bulk `eval`.
    ///
    /// Pass these alongside the output token to `mlx_sys::eval()` after each
    /// decode step.  Without this, every `slice_update` append leaves the
    /// backing buffer as a lazy graph node pointing at the previous step's
    /// buffer; after N steps each buffer is a chain of N `slice_update` nodes.
    /// Evaluating here materialises the chain into a flat buffer so the next
    /// step's `slice_update` has depth-1 ancestry instead of depth-N.
    ///
    /// Mirrors mlx_lm's `mx.eval(y, cache)` pattern.
    pub fn collect_eval_refs(&self) -> Vec<&MlxArray> {
        let mut refs = Vec::with_capacity(self.layers.len() * 4 + self.glm_mla_layers.len() * 2);
        for lkv in self.layers.iter().flatten() {
            refs.push(&lkv.k);
            refs.push(&lkv.v);
        }
        for glm_mla in self.glm_mla_layers.iter().flatten() {
            refs.push(&glm_mla.kv_latent);
            refs.push(&glm_mla.k_pe);
        }
        for linear in &self.linear_layers {
            if let Some(conv_state) = &linear.conv_state {
                refs.push(conv_state);
            }
            if let Some(recurrent_state) = &linear.recurrent_state {
                refs.push(recurrent_state);
            }
        }
        refs
    }

    /// Read-only access to a single GLM-MLA layer's cached `kv_latent` and
    /// `k_pe` arrays plus their inner dims. Used by debug bins (notably the
    /// F4 warm-extend drift probe) to compare per-layer KV state between
    /// cold and warm prefill paths.
    ///
    /// Returns `None` when the layer index is out of range or when this
    /// model has no GLM-MLA layer at that index. The arrays are over-
    /// allocated to capacity; callers that want only the valid region must
    /// slice to `[0..self.seq_len]` themselves using `self.seq_len`.
    pub fn glm_mla_layer_state(&self, layer: usize) -> Option<GlmMlaLayerStateView<'_>> {
        let entry = self.glm_mla_layers.get(layer)?.as_ref()?;
        Some(GlmMlaLayerStateView {
            kv_latent: &entry.kv_latent,
            k_pe: &entry.k_pe,
            latent_dim: entry.latent_dim,
            rope_dim: entry.rope_dim,
        })
    }

    pub fn usage_snapshot(&self) -> MlxKVCacheUsage {
        self.usage_snapshot_with_layer_windows(&[])
    }

    pub fn usage_snapshot_with_layer_windows(
        &self,
        layer_windows: &[Option<usize>],
    ) -> MlxKVCacheUsage {
        self.usage_snapshot_with_layer_windows_and_compression(
            layer_windows,
            KvCompressionConfig::disabled(),
        )
    }

    pub fn usage_snapshot_with_layer_windows_and_compression(
        &self,
        layer_windows: &[Option<usize>],
        compression: KvCompressionConfig,
    ) -> MlxKVCacheUsage {
        self.usage_snapshot_with_layer_windows_compression_and_layer_eligibility(
            layer_windows,
            compression,
            None,
        )
    }

    pub fn usage_snapshot_with_layer_windows_compression_and_layer_eligibility(
        &self,
        layer_windows: &[Option<usize>],
        compression: KvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> MlxKVCacheUsage {
        let mut usage = MlxKVCacheUsage {
            logical_tokens: self.seq_len,
            growth_count: self.growth_count,
            rotating_ring_slack: self.rotating_slack,
            ..MlxKVCacheUsage::default()
        };

        for (layer_idx, lkv) in self.layers.iter().enumerate() {
            let Some(lkv) = lkv else {
                continue;
            };
            if lkv.rotating_window.is_some() {
                usage.rotated_ring_layers = usage.rotated_ring_layers.saturating_add(1);
            }
            let elements_per_token = (lkv.n_kv_heads as u64).saturating_mul(lkv.head_dim as u64);
            let bytes_per_element = lkv.dtype.size_bytes() as u64;
            let bytes_per_token = elements_per_token
                .saturating_mul(bytes_per_element)
                .saturating_mul(2);

            usage.full_attention_layers = usage.full_attention_layers.saturating_add(1);
            usage.capacity_tokens = usage.capacity_tokens.saturating_add(lkv.capacity);
            usage.logical_bytes = usage
                .logical_bytes
                .saturating_add(bytes_per_token.saturating_mul(self.seq_len as u64));
            usage.capacity_bytes = usage
                .capacity_bytes
                .saturating_add(bytes_per_token.saturating_mul(lkv.capacity as u64));

            if let Some(window) = layer_windows.get(layer_idx).copied().flatten() {
                let retained_tokens = self.seq_len.min(window);
                let retained_capacity = chunk_ceiling(retained_tokens).min(lkv.capacity);
                let reclaimable_tokens = lkv.capacity.saturating_sub(retained_capacity);
                usage.sliding_window_layers = usage.sliding_window_layers.saturating_add(1);
                usage.sliding_window_retained_tokens = usage
                    .sliding_window_retained_tokens
                    .saturating_add(retained_tokens);
                usage.sliding_window_reclaimable_capacity_tokens = usage
                    .sliding_window_reclaimable_capacity_tokens
                    .saturating_add(reclaimable_tokens);
                usage.sliding_window_reclaimable_capacity_bytes = usage
                    .sliding_window_reclaimable_capacity_bytes
                    .saturating_add(bytes_per_token.saturating_mul(reclaimable_tokens as u64));
            }
        }

        for linear in &self.linear_layers {
            let layer_bytes = linear
                .conv_state
                .as_ref()
                .map(|array| array.nbytes() as u64)
                .unwrap_or(0)
                .saturating_add(
                    linear
                        .recurrent_state
                        .as_ref()
                        .map(|array| array.nbytes() as u64)
                        .unwrap_or(0),
                );
            if layer_bytes > 0 {
                usage.linear_state_layers = usage.linear_state_layers.saturating_add(1);
                usage.linear_state_bytes = usage.linear_state_bytes.saturating_add(layer_bytes);
            }
        }

        for glm_mla in self.glm_mla_layers.iter().flatten() {
            let elements_per_token =
                (glm_mla.latent_dim as u64).saturating_add(glm_mla.rope_dim as u64);
            let bytes_per_token =
                elements_per_token.saturating_mul(glm_mla.dtype.size_bytes() as u64);
            usage.full_attention_layers = usage.full_attention_layers.saturating_add(1);
            usage.capacity_tokens = usage.capacity_tokens.saturating_add(glm_mla.capacity);
            usage.logical_bytes = usage
                .logical_bytes
                .saturating_add(bytes_per_token.saturating_mul(self.seq_len as u64));
            usage.capacity_bytes = usage
                .capacity_bytes
                .saturating_add(bytes_per_token.saturating_mul(glm_mla.capacity as u64));
        }

        usage.kv_compression = self.estimate_kv_compression_usage(
            layer_windows,
            compression,
            compression_eligible_layers,
        );

        usage
    }

    fn estimate_kv_compression_usage(
        &self,
        layer_windows: &[Option<usize>],
        compression: KvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> MlxKvCompressionUsage {
        if !compression.is_enabled() {
            return MlxKvCompressionUsage::default();
        }

        let key_bits = compression.preset.key_bits();
        let value_bits = compression.preset.value_bits();
        let mut usage = MlxKvCompressionUsage {
            policy_enabled: true,
            fused_decode_requested: compression.requests_fused_decode(),
            status_code: 1,
            preset_code: compression.preset.route_code(),
            key_bits,
            value_bits,
            ..MlxKvCompressionUsage::default()
        };
        let storage_usage = self.turboquant_shadow_storage_usage();
        usage.runtime_storage_layers = storage_usage.layers;
        usage.runtime_storage_token_layers = storage_usage.token_layers;
        usage.runtime_storage_bytes = storage_usage.bytes;
        usage.runtime_storage_written_slots = storage_usage.written_slots;

        if self.seq_len < compression.min_context_tokens {
            usage.status_code = 2;
            return usage;
        }

        let cold_tokens = self.seq_len.saturating_sub(compression.hot_window_tokens);
        if cold_tokens == 0 {
            usage.status_code = 2;
            return usage;
        }

        for (layer_idx, lkv) in self.layers.iter().enumerate() {
            let Some(lkv) = lkv else {
                continue;
            };
            if layer_windows.get(layer_idx).copied().flatten().is_some() {
                continue;
            }
            if let Some(eligible_layers) = compression_eligible_layers
                && !eligible_layers.get(layer_idx).copied().unwrap_or(false)
            {
                continue;
            }

            usage.eligible_layers = usage.eligible_layers.saturating_add(1);
            usage.candidate_token_layers = usage.candidate_token_layers.saturating_add(cold_tokens);
            usage.hot_token_layers = usage.hot_token_layers.saturating_add(
                self.seq_len
                    .saturating_sub(cold_tokens)
                    .min(compression.hot_window_tokens),
            );

            let elements_per_token = (lkv.n_kv_heads as u64).saturating_mul(lkv.head_dim as u64);
            let bytes_per_element = lkv.dtype.size_bytes() as u64;
            let full_precision_per_token = elements_per_token
                .saturating_mul(bytes_per_element)
                .saturating_mul(2);
            let elements_per_token_usize =
                usize::try_from(elements_per_token).unwrap_or(usize::MAX);
            let compressed_bytes_per_token = if compression.preset.has_fractional_values() {
                let key_bytes =
                    packed_index_bytes(elements_per_token_usize, compression.preset.key_bits())
                        .expect("TurboQuant shadow presets use supported packed key bit widths");
                let value_bytes =
                    packed_value_bytes_for_preset(elements_per_token_usize, compression.preset)
                        .expect("TurboQuant shadow presets use supported packed value bit widths");
                key_bytes.saturating_add(value_bytes) as u64
            } else {
                packed_kv_bytes_per_token(elements_per_token_usize, key_bits, value_bits)
                    .expect("TurboQuant shadow presets use supported packed bit widths")
                    as u64
            };

            usage.full_precision_bytes = usage
                .full_precision_bytes
                .saturating_add(full_precision_per_token.saturating_mul(cold_tokens as u64));
            usage.estimated_compressed_bytes = usage
                .estimated_compressed_bytes
                .saturating_add(compressed_bytes_per_token.saturating_mul(cold_tokens as u64));
        }

        if usage.eligible_layers == 0 {
            usage.status_code = 3;
            return usage;
        }

        usage.estimated_saved_bytes = usage
            .full_precision_bytes
            .saturating_sub(usage.estimated_compressed_bytes);
        usage.estimated_ratio_milli = if usage.full_precision_bytes == 0 {
            0
        } else {
            usage
                .estimated_compressed_bytes
                .saturating_mul(1000)
                .saturating_div(usage.full_precision_bytes)
                .min(u32::MAX as u64) as u32
        };

        if matches!(compression.mode, KvCompressionMode::TurboQuantShadow) {
            usage.status_code = 1;
        }

        usage
    }

    /// Read the cached gated-delta states for a Qwen3.5 linear-attention layer.
    pub fn linear_state(&self, layer: usize) -> (Option<&MlxArray>, Option<&MlxArray>) {
        let state = &self.linear_layers[layer];
        (state.conv_state.as_ref(), state.recurrent_state.as_ref())
    }

    /// Store the gated-delta states for a Qwen3.5 linear-attention layer.
    pub fn set_linear_state(
        &mut self,
        layer: usize,
        conv_state: MlxArray,
        recurrent_state: MlxArray,
    ) {
        let state = &mut self.linear_layers[layer];
        state.conv_state = Some(conv_state);
        state.recurrent_state = Some(recurrent_state);
    }

    /// Read K/V already written by `source_layer` during the current forward pass.
    ///
    /// Used by KV-shared layers (e.g. Gemma4 layers 24-41) that attend against
    /// a prior layer's cache instead of computing their own K/V projections.
    ///
    /// Returns the views cached by the last `append` call, which are identical to
    /// what a fresh `slice(lkv.k, 0..seq_len+new_tokens)` would produce.  Reusing
    /// the same MLX graph node avoids a duplicate GPU kernel dispatch per KV-shared
    /// layer: for E2B (20 shared layers) this eliminates 40 extra slice kernels
    /// (~12 µs each), saving ~0.5 ms per decode step.
    ///
    /// `new_tokens` is retained for the panic check that validates the source layer
    /// was updated in the current forward pass.
    pub fn peek_source_kv(&self, source_layer: usize, new_tokens: usize) -> (MlxArray, MlxArray) {
        let lkv = self.layers[source_layer]
            .as_ref()
            .expect("KV-shared source layer has no cached KV — source layer must appear earlier");
        if lkv.rotating_window.is_some() {
            // Rotated ring: the backing store IS the full ring view (the
            // storing layer's append returned exactly this), and the ordered
            // `[0, seq_len)` fallback below would slice past the ring's
            // capacity. Consumers mask via the hoisted ring validity mask.
            return (lkv.k.clone(), lkv.v.clone());
        }
        let (k_view, v_view) = match (&lkv.last_k_view, &lkv.last_v_view) {
            (Some(k), Some(v)) => (k.clone(), v.clone()),
            _ => {
                // Fallback: create fresh views (e.g., first append in a grow-then-slice sequence).
                let end = (self.seq_len + new_tokens) as i32;
                let k = slice(
                    &lkv.k,
                    &[0, 0, 0, 0],
                    &[1, lkv.n_kv_heads, end, lkv.head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                let v = slice(
                    &lkv.v,
                    &[0, 0, 0, 0],
                    &[1, lkv.n_kv_heads, end, lkv.head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                (k, v)
            }
        };
        (k_view, v_view)
    }

    /// Read K/V already stored for `layer` without mutating cache state.
    ///
    /// Gemma4 Assistant uses this to attend against target full/sliding K/V states
    /// from a separate assistant forward pass. Unlike `peek_source_kv`, this does
    /// not assert that the layer was written during the current forward call.
    pub fn peek_layer_kv(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let lkv = self.layers.get(layer)?.as_ref()?;
        if lkv.rotating_window.is_some() {
            // Rotated ring: return the full ring — valid at any time,
            // including right after a `trim_to` rollback cleared the cached
            // views (when the ordered `[0, seq_len)` fallback below would
            // slice past the ring's capacity). Consumers must apply the
            // slot-validity mask derived from [`Self::layer_sliding_ring`].
            return Some((lkv.k.clone(), lkv.v.clone()));
        }
        let (k_view, v_view) = match (&lkv.last_k_view, &lkv.last_v_view) {
            (Some(k), Some(v)) => (k.clone(), v.clone()),
            _ => {
                let end = self.seq_len as i32;
                let k = slice(
                    &lkv.k,
                    &[0, 0, 0, 0],
                    &[1, lkv.n_kv_heads, end, lkv.head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                let v = slice(
                    &lkv.v,
                    &[0, 0, 0, 0],
                    &[1, lkv.n_kv_heads, end, lkv.head_dim],
                    &[1, 1, 1, 1],
                    None,
                );
                (k, v)
            }
        };
        Some((k_view, v_view))
    }

    /// The current ring geometry of `layer` if it has converted to a
    /// rotating ring, for consumers that read ring K/V outside a forward
    /// (e.g. the Gemma4 assistant drafter attending target KV between
    /// appends). `write_start` is set to the cache's current `seq_len`:
    /// a reader whose query logically sits at the *end* of the context
    /// builds its mask as `create_ring_sliding_mask(1, window, capacity,
    /// seq_len - 1)`, which keeps exactly the last `window` live tokens
    /// and automatically excludes slots holding rolled-back drafts (their
    /// resident-token index decodes below `seq_len - window` under the
    /// post-trim end).
    pub fn layer_sliding_ring(&self, layer: usize) -> Option<SlidingRingLayout> {
        let lkv = self.layers.get(layer)?.as_ref()?;
        let window = lkv.rotating_window?;
        Some(SlidingRingLayout {
            window,
            capacity: lkv.capacity,
            write_start: self.seq_len,
        })
    }

    /// Read a fresh full-prefix K/V view for `layer`.
    ///
    /// Unlike `peek_layer_kv`, this intentionally ignores cached retained views
    /// from the most recent append. Diffusion denoising attends against the
    /// committed prompt prefix, so its bidirectional mask must match exactly
    /// `self.seq_len` cached keys.
    pub fn peek_layer_full_kv(&self, layer: usize) -> Option<(MlxArray, MlxArray)> {
        let lkv = self.layers.get(layer)?.as_ref()?;
        let end = self.seq_len as i32;
        let k = slice(
            &lkv.k,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v = slice(
            &lkv.v,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        Some((k, v))
    }

    /// Replace the backing K/V arrays for a layer.
    ///
    /// Used by the whole-layer compiled decode path to update the cache
    /// with new arrays returned from the compiled closure. Unlike `append`,
    /// this replaces the full backing buffer rather than writing new tokens
    /// to the existing one. The `seq_len` is not incremented; callers must
    /// manage `seq_len` separately.
    pub fn replace_layer_kv(&mut self, layer: usize, new_k: MlxArray, new_v: MlxArray) {
        let Some(Some(lkv)) = self.layers.get_mut(layer) else {
            return;
        };
        lkv.k = new_k;
        lkv.v = new_v;
        lkv.last_k_view = None;
        lkv.last_v_view = None;
    }

    /// Reset cache entirely (e.g., between requests).
    pub fn reset(&mut self) {
        for entry in &mut self.layers {
            *entry = None;
        }
        for entry in &mut self.glm_mla_layers {
            *entry = None;
        }
        for state in &mut self.linear_layers {
            *state = LinearLayerState::default();
        }
        self.clear_turboquant_shadow_storage();
        self.seq_len = 0;
        self.rope_offset = 0;
        self.growth_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::reference_decode_attention;

    #[test]
    fn turboquant_shadow_metal_buffer_incremental_update_matches_host_bytes() {
        let layout = TurboQuantBlockLayout::new(TurboQuantBlockLayoutConfig {
            preset: TurboQuantPreset::K8V4,
            block_tokens: 4,
            n_kv_heads: 2,
            head_dim: 8,
            value_group_size: 4,
        })
        .expect("layout");
        let mut buffer = TurboQuantCompressedBlockBuffer::new(layout);
        let mut metal: Option<MlxArray> = None;
        let mut uploaded_tokens = 0usize;

        // Sync widths chosen to cover: initial full upload, 1-token
        // incremental updates, a multi-token update, and block-boundary
        // growth (block_tokens = 4) forcing the full-upload fallback.
        for sync_width in [1usize, 1, 2, 1, 3, 1] {
            for token_index in uploaded_tokens..uploaded_tokens + sync_width {
                let heads = (0..2)
                    .map(|head| {
                        (
                            (0..8)
                                .map(|dim| {
                                    ((token_index * 7 + head * 3 + dim) % 11) as f32 / 5.0 - 1.0
                                })
                                .collect::<Vec<_>>(),
                            (0..8)
                                .map(|dim| {
                                    ((token_index * 5 + head * 2 + dim) % 13) as f32 / 6.0 - 1.0
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                buffer
                    .write_token(token_index, &heads)
                    .expect("write token");
            }
            let synced_tokens = uploaded_tokens + sync_width;

            let updated = turboquant_shadow_metal_buffer_update(
                metal.take(),
                &layout,
                &buffer,
                uploaded_tokens,
                synced_tokens,
            );
            assert_eq!(updated.nbytes(), buffer.as_bytes().len());
            let device =
                unsafe { std::slice::from_raw_parts(updated.data_raw(), updated.nbytes()) };
            assert_eq!(
                device,
                buffer.as_bytes(),
                "device bytes diverged after syncing to {synced_tokens} tokens"
            );
            metal = Some(updated);
            uploaded_tokens = synced_tokens;
        }
    }

    #[test]
    fn compression_usage_accumulates_decode_usage() {
        let mut usage = MlxKvCompressionUsage {
            fused_decode_attempts: 3,
            fused_decode_successes: 2,
            fused_decode_fallbacks: u64::MAX,
            ..MlxKvCompressionUsage::default()
        };

        usage.apply_decode_usage(MlxKvCompressionDecodeUsage {
            fused_decode_attempts: 4,
            fused_decode_successes: 5,
            fused_decode_metal_successes: 1,
            fused_decode_fallbacks: 1,
            fused_decode_ready_candidates: 6,
            fused_decode_blocked_prefill_only: 7,
            fused_decode_blocked_attention_kind: 8,
            fused_decode_blocked_linear_attention: 9,
            fused_decode_blocked_sliding_window: 11,
            fused_decode_blocked_kv_shared: 12,
            fused_decode_blocked_ineligible_layer: 13,
            fused_decode_blocked_unsupported_preset: 14,
            fused_decode_blocked_unsupported_head_dim: 15,
            fused_decode_blocked_gqa: 16,
            fused_decode_blocked_missing_storage: 17,
            fused_decode_blocked_short_context: 22,
            fused_decode_query_readback_wall_us: 18,
            fused_decode_cold_metal_wall_us: 19,
            fused_decode_hot_tail_merge_wall_us: 20,
            fused_decode_output_staging_wall_us: 21,
        });

        assert_eq!(usage.fused_decode_attempts, 7);
        assert_eq!(usage.fused_decode_successes, 7);
        assert_eq!(usage.fused_decode_metal_successes, 1);
        assert_eq!(usage.fused_decode_fallbacks, u64::MAX);
        assert_eq!(usage.fused_decode_ready_candidates, 6);
        assert_eq!(usage.fused_decode_blocked_prefill_only, 7);
        assert_eq!(usage.fused_decode_blocked_attention_kind, 8);
        assert_eq!(usage.fused_decode_blocked_linear_attention, 9);
        assert_eq!(usage.fused_decode_blocked_sliding_window, 11);
        assert_eq!(usage.fused_decode_blocked_kv_shared, 12);
        assert_eq!(usage.fused_decode_blocked_ineligible_layer, 13);
        assert_eq!(usage.fused_decode_blocked_unsupported_preset, 14);
        assert_eq!(usage.fused_decode_blocked_unsupported_head_dim, 15);
        assert_eq!(usage.fused_decode_blocked_gqa, 16);
        assert_eq!(usage.fused_decode_blocked_missing_storage, 17);
        assert_eq!(usage.fused_decode_blocked_short_context, 22);
        assert_eq!(usage.fused_decode_query_readback_wall_us, 18);
        assert_eq!(usage.fused_decode_cold_metal_wall_us, 19);
        assert_eq!(usage.fused_decode_hot_tail_merge_wall_us, 20);
        assert_eq!(usage.fused_decode_output_staging_wall_us, 21);
    }

    #[test]
    fn turboquant_attention_kind_subreason_candidates_keep_aggregate_counter() {
        let mut cache = MlxKVCache::new(1);

        cache.record_turboquant_decode_candidate(MlxKvCompressionDecodeCandidate::LinearAttention);
        cache.record_turboquant_decode_candidate(MlxKvCompressionDecodeCandidate::SlidingWindow);
        cache.record_turboquant_decode_candidate(MlxKvCompressionDecodeCandidate::KvShared);

        let usage = cache.take_turboquant_decode_usage();
        assert_eq!(usage.fused_decode_blocked_attention_kind, 3);
        assert_eq!(usage.fused_decode_blocked_linear_attention, 1);
        assert_eq!(usage.fused_decode_blocked_sliding_window, 1);
        assert_eq!(usage.fused_decode_blocked_kv_shared, 1);
    }

    #[test]
    fn linear_state_is_eval_tracked_and_reset() {
        let mut cache = MlxKVCache::new(2);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.set_linear_state(1, conv, recurrent);

        let (conv_state, recurrent_state) = cache.linear_state(1);
        assert_eq!(conv_state.expect("conv state").shape(), vec![1, 3, 14]);
        assert_eq!(
            recurrent_state.expect("recurrent state").shape(),
            vec![1, 4, 8, 6]
        );
        assert_eq!(cache.collect_eval_refs().len(), 2);

        cache.reset();

        let (conv_state, recurrent_state) = cache.linear_state(1);
        assert!(conv_state.is_none());
        assert!(recurrent_state.is_none());
        assert!(cache.collect_eval_refs().is_empty());
    }

    #[test]
    fn linear_state_survives_trim_to() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.seq_len = 8;
        cache.set_linear_state(0, conv, recurrent);
        assert!(cache.trim_to(4));

        assert_eq!(cache.seq_len, 4);
        let (conv_state, recurrent_state) = cache.linear_state(0);
        assert!(
            conv_state.is_some() && recurrent_state.is_some(),
            "linear recurrent state is not rolled back by seq_len trim"
        );
    }

    #[test]
    fn trim_to_does_not_extend_logical_sequence() {
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 8;

        assert!(!cache.trim_to(12));

        assert_eq!(
            cache.seq_len, 8,
            "invalid rollback points must not expose unwritten KV slots"
        );
    }

    #[test]
    fn clone_preserves_linear_state_for_draft_branch() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.seq_len = 12;
        cache.set_linear_state(0, conv, recurrent);
        let branch = cache.clone();
        cache.reset();

        assert_eq!(branch.seq_len, 12);
        let (conv_state, recurrent_state) = branch.linear_state(0);
        assert!(conv_state.is_some());
        assert!(recurrent_state.is_some());
    }

    #[test]
    fn glm_mla_cache_appends_latent_and_rope_key_history() {
        let mut cache = MlxKVCache::new(1);
        let kv_latent = zeros(&[1, 1, 2, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 2, 64], MlxDtype::Bfloat16, None);

        let (latent_history, rope_history) = cache.append_glm_mla(0, kv_latent, k_pe);

        assert_eq!(latent_history.shape(), vec![1, 1, 2, 512]);
        assert_eq!(rope_history.shape(), vec![1, 1, 2, 64]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
        cache.seq_len = 2;

        let kv_latent = zeros(&[1, 1, 1, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 1, 64], MlxDtype::Bfloat16, None);
        let (latent_history, rope_history) = cache.append_glm_mla(0, kv_latent, k_pe);

        assert_eq!(latent_history.shape(), vec![1, 1, 3, 512]);
        assert_eq!(rope_history.shape(), vec![1, 1, 3, 64]);
        assert_eq!(cache.collect_eval_refs().len(), 2);
    }

    #[test]
    fn usage_snapshot_tracks_glm_mla_compressed_cache_bytes() {
        let mut cache = MlxKVCache::new(1);
        let kv_latent = zeros(&[1, 1, 2, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 2, 64], MlxDtype::Bfloat16, None);

        cache.append_glm_mla(0, kv_latent, k_pe);
        cache.seq_len = 2;

        let usage = cache.usage_snapshot();
        assert_eq!(usage.logical_tokens, 2);
        assert_eq!(usage.capacity_tokens, 256);
        assert_eq!(usage.full_attention_layers, 1);
        assert_eq!(usage.logical_bytes, 2304);
        assert_eq!(usage.capacity_bytes, 294_912);
        assert_eq!(usage.growth_count, 1);
    }

    #[test]
    fn reset_clears_glm_mla_cache_eval_refs() {
        let mut cache = MlxKVCache::new(1);
        let kv_latent = zeros(&[1, 1, 1, 512], MlxDtype::Bfloat16, None);
        let k_pe = zeros(&[1, 1, 1, 64], MlxDtype::Bfloat16, None);

        cache.append_glm_mla(0, kv_latent, k_pe);
        cache.seq_len = 1;
        assert_eq!(cache.collect_eval_refs().len(), 2);

        cache.reset();

        assert_eq!(cache.seq_len, 0);
        assert!(cache.collect_eval_refs().is_empty());
        assert_eq!(cache.usage_snapshot(), MlxKVCacheUsage::default());
    }

    #[test]
    fn usage_snapshot_tracks_full_attention_capacity_and_growth() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 3;

        let usage = cache.usage_snapshot();
        assert_eq!(usage.logical_tokens, 3);
        assert_eq!(usage.capacity_tokens, 256);
        assert_eq!(usage.full_attention_layers, 1);
        assert_eq!(usage.logical_bytes, 96);
        assert_eq!(usage.capacity_bytes, 8192);
        assert_eq!(usage.growth_count, 1);
    }

    #[test]
    fn peek_layer_full_kv_ignores_retained_last_view() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 5, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 5, 4], MlxDtype::Bfloat16, None);

        cache.append_with_retained_window(0, k, v, Some(3));
        cache.seq_len = 5;

        let (retained_k, _) = cache.peek_layer_kv(0).expect("retained view");
        let (full_k, _) = cache.peek_layer_full_kv(0).expect("full view");
        assert_eq!(retained_k.shape(), vec![1, 2, 3, 4]);
        assert_eq!(full_k.shape(), vec![1, 2, 5, 4]);
    }

    #[test]
    fn multi_token_append_retains_window_plus_seq_view() {
        // Prefill 8 tokens (full view), then append a 4-token chunk with the
        // multi-token retained bound window + seq - 1 = 3 + 4 - 1 = 6: the
        // returned view must be the last 6 tokens of the 12-token history,
        // contents intact, while full storage stays available for rollback
        // and prefix-cache snapshots.
        let head_dim = 2usize;
        let fill = |start: usize, tokens: usize| -> MlxArray {
            let data: Vec<f32> = (start..start + tokens * head_dim)
                .map(|i| i as f32)
                .collect();
            let flat = MlxArray::from_f32_slice(&data);
            mlx_sys::reshape(&flat, &[1, 1, tokens as i32, head_dim as i32], None)
        };
        let read_f32 = |arr: &MlxArray| -> Vec<f32> {
            let arr = astype(arr, MlxDtype::Float32, None);
            eval(&[&arr]);
            let len = arr.nbytes() / std::mem::size_of::<f32>();
            let ptr = arr.data_raw() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
        };

        let mut cache = MlxKVCache::new(1);
        cache.append(0, fill(0, 8), fill(100, 8));
        cache.seq_len = 8;

        let (k_view, v_view) =
            cache.append_with_retained_window(0, fill(16, 4), fill(116, 4), Some(6));
        cache.seq_len = 12;

        assert_eq!(k_view.shape(), vec![1, 1, 6, head_dim as i32]);
        assert_eq!(v_view.shape(), vec![1, 1, 6, head_dim as i32]);
        // Last 6 tokens = prompt tokens 6..8 (values 12..16) + the 4 new
        // tokens (values 16..24).
        let expected_k: Vec<f32> = (12..24).map(|i| i as f32).collect();
        let expected_v: Vec<f32> = (112..124).map(|i| i as f32).collect();
        assert_eq!(read_f32(&k_view), expected_k);
        assert_eq!(read_f32(&v_view), expected_v);

        let (full_k, full_v) = cache.peek_layer_full_kv(0).expect("full view");
        assert_eq!(full_k.shape(), vec![1, 1, 12, head_dim as i32]);
        let full_k_data = read_f32(&full_k);
        assert_eq!(
            &full_k_data[..16],
            (0..16).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            &full_k_data[16..],
            (16..24).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert_eq!(read_f32(&full_v).len(), 24);
    }

    #[test]
    fn usage_snapshot_tracks_sliding_window_trim_opportunity() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 300;

        let usage = cache.usage_snapshot_with_layer_windows(&[Some(128)]);
        assert_eq!(usage.capacity_tokens, 512);
        assert_eq!(usage.sliding_window_layers, 1);
        assert_eq!(usage.sliding_window_retained_tokens, 128);
        assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 256);
        assert_eq!(usage.sliding_window_reclaimable_capacity_bytes, 8192);
    }

    #[test]
    fn usage_snapshot_does_not_emit_compression_by_default() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 600;

        let usage = cache.usage_snapshot_with_layer_windows(&[None]);
        assert_eq!(usage.kv_compression, MlxKvCompressionUsage::default());
    }

    #[test]
    fn turboquant_shadow_estimates_full_attention_cold_tokens_only() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 600;

        let usage = cache.usage_snapshot_with_layer_windows_and_compression(
            &[None],
            KvCompressionConfig::turboquant_shadow(),
        );

        assert!(usage.kv_compression.policy_enabled);
        assert_eq!(usage.kv_compression.status_code, 1);
        assert_eq!(usage.kv_compression.preset_code, 1);
        assert_eq!(usage.kv_compression.key_bits, 8);
        assert_eq!(usage.kv_compression.value_bits, 4);
        assert_eq!(usage.kv_compression.eligible_layers, 1);
        assert_eq!(usage.kv_compression.candidate_token_layers, 344);
        assert_eq!(usage.kv_compression.hot_token_layers, 256);
        assert_eq!(usage.kv_compression.full_precision_bytes, 11_008);
        assert_eq!(usage.kv_compression.estimated_compressed_bytes, 4_128);
        assert_eq!(usage.kv_compression.estimated_saved_bytes, 6_880);
        assert_eq!(usage.kv_compression.estimated_ratio_milli, 375);
    }

    #[test]
    fn turboquant_shadow_estimates_fractional_value_bits_without_rounding_up() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 1, 600, 128], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 600, 128], MlxDtype::Bfloat16, None);
        let compression = KvCompressionConfig {
            preset: TurboQuantPreset::K8V3_5,
            ..KvCompressionConfig::turboquant_shadow()
        };

        cache.append(0, k, v);
        cache.seq_len = 600;

        let usage = cache.usage_snapshot_with_layer_windows_and_compression(&[None], compression);

        assert_eq!(usage.kv_compression.candidate_token_layers, 344);
        assert_eq!(
            usage.kv_compression.estimated_compressed_bytes,
            (128 + 56) * 344
        );
    }

    #[test]
    fn turboquant_shadow_sync_writes_runtime_compressed_storage() {
        let mut cache = MlxKVCache::new(1);
        let k_data: Vec<f32> = (0..48).map(|idx| idx as f32 / 16.0).collect();
        let v_data: Vec<f32> = (0..48).map(|idx| 1.0 - (idx as f32 / 64.0)).collect();
        let k = MlxArray::from_raw_data(
            k_data.as_ptr().cast(),
            k_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let compression = KvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..KvCompressionConfig::turboquant_shadow()
        };

        cache.append(0, k, v);
        cache.seq_len = 6;
        let storage = cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));

        assert_eq!(storage.layers, 1);
        assert_eq!(storage.token_layers, 4);
        assert_eq!(storage.written_slots, 8);
        assert!(storage.bytes > 0);

        let usage = cache.usage_snapshot_with_layer_windows_compression_and_layer_eligibility(
            &[None],
            compression,
            Some(&[true]),
        );
        assert_eq!(usage.kv_compression.runtime_storage_layers, 1);
        assert_eq!(usage.kv_compression.runtime_storage_token_layers, 4);
        assert_eq!(usage.kv_compression.runtime_storage_written_slots, 8);
        assert_eq!(usage.kv_compression.runtime_storage_bytes, storage.bytes);

        cache.reset();
        assert_eq!(
            cache.turboquant_shadow_storage_usage(),
            TurboQuantShadowStorageUsage::default()
        );
    }

    #[test]
    fn turboquant_shadow_trim_truncates_runtime_storage() {
        let mut cache = MlxKVCache::new(1);
        let k_data: Vec<f32> = (0..48).map(|idx| idx as f32 / 16.0).collect();
        let v_data: Vec<f32> = (0..48).map(|idx| 1.0 - (idx as f32 / 64.0)).collect();
        let k = MlxArray::from_raw_data(
            k_data.as_ptr().cast(),
            k_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let compression = KvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..KvCompressionConfig::turboquant_shadow()
        };

        cache.append(0, k, v);
        cache.seq_len = 6;
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));
        assert_eq!(cache.turboquant_shadow_storage_cold_tokens(0), Some(4));
        assert!(
            cache.turboquant_shadow_layers[0]
                .as_ref()
                .expect("shadow storage")
                .metal_buffer
                .is_some()
        );

        assert!(cache.trim_to(3));

        let storage = cache.turboquant_shadow_layers[0]
            .as_ref()
            .expect("shadow storage");
        assert_eq!(storage.compressed_tokens, 3);
        assert_eq!(storage.buffer.token_count(), 3);
        assert_eq!(storage.buffer.written_slot_count(), 6);
        assert!(storage.metal_buffer.is_none());
        assert_eq!(cache.turboquant_shadow_storage_usage().token_layers, 3);
    }

    #[test]
    fn turboquant_shadow_decode_merges_runtime_cold_storage_with_hot_tail() {
        let mut cache = MlxKVCache::new(1);
        let k_data: Vec<f32> = (0..48).map(|idx| ((idx % 13) as f32 - 6.0) / 8.0).collect();
        let v_data: Vec<f32> = (0..48).map(|idx| ((idx % 11) as f32 - 5.0) / 7.0).collect();
        let k = MlxArray::from_raw_data(
            k_data.as_ptr().cast(),
            k_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let compression = KvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..KvCompressionConfig::turboquant_shadow()
        };
        let queries = vec![vec![0.15, -0.25, 0.35, -0.45], vec![-0.2, 0.1, 0.4, -0.3]];

        cache.append(0, k, v);
        cache.seq_len = 6;
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));

        let actual = cache
            .debug_turboquant_shadow_decode_attention_for_layer(0, &queries, 2)
            .expect("shadow runtime storage should decode");
        let lkv = cache.layers[0].as_ref().expect("layer cache");
        let expected = queries
            .iter()
            .enumerate()
            .map(|(head_index, query)| {
                let history = (0..cache.seq_len)
                    .map(|token_index| kv_heads_for_token(lkv, token_index)[head_index].clone())
                    .collect::<Vec<_>>();
                reference_decode_attention(query, &history).expect("full precision attention")
            })
            .collect::<Vec<_>>();

        for (expected, actual) in expected.iter().zip(&actual) {
            for (expected, actual) in expected.iter().zip(actual) {
                assert!((actual - expected).abs() < 0.05);
            }
        }
    }

    #[test]
    fn turboquant_hot_tail_partition_stats_read_compact_slices_without_token_materialization() {
        let n_kv_heads = 2;
        let head_dim = 4;
        let hot_token_count = 3;
        let k_values = (0..n_kv_heads * hot_token_count * head_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) / 9.0)
            .collect::<Vec<_>>();
        let v_values = (0..n_kv_heads * hot_token_count * head_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) / 7.0)
            .collect::<Vec<_>>();
        let queries = [
            vec![0.1, -0.2, 0.3, -0.4],
            vec![-0.3, 0.2, 0.1, 0.5],
            vec![0.4, 0.3, -0.2, 0.1],
            vec![0.0, -0.5, 0.2, 0.3],
        ];

        for (query_head_index, query) in queries.iter().enumerate() {
            let kv_head_index =
                turboquant_query_head_to_kv_head(query_head_index, queries.len(), n_kv_heads)
                    .expect("GQA head mapping");
            let shape = KvHeadSliceShape {
                n_kv_heads: n_kv_heads as i32,
                head_dim: head_dim as i32,
                capacity: hot_token_count,
            };
            let hot_tokens = (0..hot_token_count)
                .map(|token_index| {
                    kv_head_for_token_from_f32_slices(
                        &k_values,
                        &v_values,
                        shape,
                        token_index,
                        kv_head_index,
                    )
                    .expect("test KV slices are in bounds")
                })
                .collect::<Vec<_>>();
            let expected =
                crate::turboquant::reference_decode_attention_partition_stats(query, &hot_tokens)
                    .expect("reference hot stats");
            let actual = hot_tail_partition_stats_from_f32_slices(
                query,
                &k_values,
                &v_values,
                n_kv_heads,
                head_dim,
                hot_token_count,
                kv_head_index,
            )
            .expect("slice hot stats");

            assert_eq!(actual.token_count, expected.token_count);
            assert_eq!(actual.value_dim, expected.value_dim);
            assert!((actual.max_score - expected.max_score).abs() < 1e-6);
            assert!((actual.exp_sum - expected.exp_sum).abs() < 1e-6);
            for (actual, expected) in actual
                .weighted_value_sum
                .iter()
                .zip(&expected.weighted_value_sum)
            {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn turboquant_hot_tail_merge_combines_cold_stats_without_cloning_partition_payloads() {
        let n_kv_heads = 2;
        let head_dim = 4;
        let cold_token_count = 2;
        let hot_token_count = 3;
        let hot_k_values = (0..n_kv_heads * hot_token_count * head_dim)
            .map(|idx| ((idx % 19) as f32 - 9.0) / 10.0)
            .collect::<Vec<_>>();
        let hot_v_values = (0..n_kv_heads * hot_token_count * head_dim)
            .map(|idx| ((idx % 23) as f32 - 11.0) / 12.0)
            .collect::<Vec<_>>();
        let cold_k_values = (0..n_kv_heads * cold_token_count * head_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let cold_v_values = (0..n_kv_heads * cold_token_count * head_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) / 9.0)
            .collect::<Vec<_>>();
        let queries = [
            vec![0.15, -0.2, 0.35, -0.45],
            vec![-0.25, 0.3, 0.2, -0.1],
            vec![0.4, 0.1, -0.3, 0.25],
            vec![0.05, -0.35, 0.45, 0.15],
        ];
        let hot_shape = KvHeadSliceShape {
            n_kv_heads: n_kv_heads as i32,
            head_dim: head_dim as i32,
            capacity: hot_token_count,
        };
        let cold_shape = KvHeadSliceShape {
            n_kv_heads: n_kv_heads as i32,
            head_dim: head_dim as i32,
            capacity: cold_token_count,
        };

        for (query_head_index, query) in queries.iter().enumerate() {
            let kv_head_index =
                turboquant_query_head_to_kv_head(query_head_index, queries.len(), n_kv_heads)
                    .expect("GQA head mapping");
            let cold_tokens = (0..cold_token_count)
                .map(|token_index| {
                    kv_head_for_token_from_f32_slices(
                        &cold_k_values,
                        &cold_v_values,
                        cold_shape,
                        token_index,
                        kv_head_index,
                    )
                    .expect("test KV slices are in bounds")
                })
                .collect::<Vec<_>>();
            let hot_tokens = (0..hot_token_count)
                .map(|token_index| {
                    kv_head_for_token_from_f32_slices(
                        &hot_k_values,
                        &hot_v_values,
                        hot_shape,
                        token_index,
                        kv_head_index,
                    )
                    .expect("test KV slices are in bounds")
                })
                .collect::<Vec<_>>();
            let cold_stats =
                crate::turboquant::reference_decode_attention_partition_stats(query, &cold_tokens)
                    .expect("reference cold stats");
            let hot_stats =
                crate::turboquant::reference_decode_attention_partition_stats(query, &hot_tokens)
                    .expect("reference hot stats");
            let expected = crate::turboquant::merge_attention_partition_stats(&[
                cold_stats.clone(),
                hot_stats,
            ])
            .expect("reference merged output");
            let actual = merge_cold_stats_with_hot_tail_from_f32_slices(
                &cold_stats,
                query,
                &hot_k_values,
                &hot_v_values,
                n_kv_heads,
                head_dim,
                hot_token_count,
                kv_head_index,
            )
            .expect("direct merged output");

            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "actual {actual} expected {expected}"
                );
            }
        }
    }

    #[test]
    fn turboquant_shadow_decode_fails_closed_without_runtime_storage() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 6, 4], MlxDtype::Float32, None);
        let v = zeros(&[1, 2, 6, 4], MlxDtype::Float32, None);
        cache.append(0, k, v);
        cache.seq_len = 6;

        let error = cache
            .debug_turboquant_shadow_decode_attention_for_layer(0, &[vec![0.0; 4], vec![0.0; 4]], 2)
            .expect_err("missing compressed storage should fail closed");
        assert!(matches!(
            error,
            TurboQuantCodecError::CompressedDecodePlanIncomplete { .. }
        ));
    }

    #[test]
    fn turboquant_shadow_sync_due_waits_for_block_sized_cold_advances() {
        let mut cache = MlxKVCache::new(1);
        let k_data: Vec<f32> = (0..48).map(|idx| idx as f32 / 16.0).collect();
        let v_data: Vec<f32> = (0..48).map(|idx| 1.0 - (idx as f32 / 64.0)).collect();
        let k = MlxArray::from_raw_data(
            k_data.as_ptr().cast(),
            k_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            v_data.len() * std::mem::size_of::<f32>(),
            &[1, 2, 6, 4],
            MlxDtype::Float32,
        );
        let compression = KvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..KvCompressionConfig::turboquant_shadow()
        };

        cache.append(0, k, v);
        cache.seq_len = 6;
        assert!(cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])));
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));
        assert!(!cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])));

        cache.seq_len = 10;
        assert!(
            cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])),
            "short cold advances should use the incremental sync path"
        );

        cache.seq_len = 6 + KV_CHUNK_TOKENS - 1;
        assert!(!cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])));

        cache.seq_len = 6 + KV_CHUNK_TOKENS;
        assert!(cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])));

        cache.seq_len = 2;
        assert!(cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])));
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));
        assert_eq!(
            cache.turboquant_shadow_storage_usage(),
            TurboQuantShadowStorageUsage::default()
        );
    }

    #[test]
    fn turboquant_shadow_sync_due_matches_layer_selection_changes() {
        let mut cache = MlxKVCache::new(2);
        let k0 = zeros(&[1, 2, 600, 4], MlxDtype::Float32, None);
        let v0 = zeros(&[1, 2, 600, 4], MlxDtype::Float32, None);
        let k1 = zeros(&[1, 2, 600, 4], MlxDtype::Float32, None);
        let v1 = zeros(&[1, 2, 600, 4], MlxDtype::Float32, None);
        let compression = KvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..KvCompressionConfig::turboquant_shadow()
        };

        cache.append(0, k0, v0);
        cache.append(1, k1, v1);
        cache.seq_len = 600;
        cache.sync_turboquant_shadow_storage(&[None, None], compression, Some(&[true, true]));
        assert_eq!(cache.turboquant_shadow_storage_usage().layers, 2);

        assert!(
            cache.turboquant_shadow_storage_sync_due(
                &[None, None],
                compression,
                Some(&[false, true])
            ),
            "eligibility changes should request stale storage cleanup"
        );
        cache.sync_turboquant_shadow_storage(&[None, None], compression, Some(&[false, true]));
        assert!(cache.turboquant_shadow_layers[0].is_none());
        assert!(cache.turboquant_shadow_layers[1].is_some());

        assert!(
            cache.turboquant_shadow_storage_sync_due(
                &[None, Some(512)],
                compression,
                Some(&[false, true])
            ),
            "window changes should request stale storage cleanup"
        );
        cache.sync_turboquant_shadow_storage(&[None, Some(512)], compression, Some(&[false, true]));
        assert_eq!(
            cache.turboquant_shadow_storage_usage(),
            TurboQuantShadowStorageUsage::default()
        );
    }

    #[test]
    fn turboquant_shadow_respects_layer_eligibility_mask() {
        let mut cache = MlxKVCache::new(2);
        let k0 = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);
        let v0 = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);
        let k1 = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);
        let v1 = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k0, v0);
        cache.append(1, k1, v1);
        cache.seq_len = 600;

        let usage = cache.usage_snapshot_with_layer_windows_compression_and_layer_eligibility(
            &[None, None],
            KvCompressionConfig::turboquant_shadow(),
            Some(&[false, true]),
        );

        assert_eq!(usage.kv_compression.status_code, 1);
        assert_eq!(usage.kv_compression.eligible_layers, 1);
        assert_eq!(usage.kv_compression.candidate_token_layers, 344);

        let usage = cache.usage_snapshot_with_layer_windows_compression_and_layer_eligibility(
            &[None, None],
            KvCompressionConfig::turboquant_shadow(),
            Some(&[false]),
        );

        assert_eq!(usage.kv_compression.status_code, 3);
        assert_eq!(usage.kv_compression.eligible_layers, 0);
        assert_eq!(usage.kv_compression.candidate_token_layers, 0);
    }

    #[test]
    fn turboquant_shadow_does_not_target_short_context_or_sliding_window_layers() {
        let mut cache = MlxKVCache::new(2);
        let k = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
        cache.append(0, k, v);
        cache.seq_len = 300;

        let usage = cache.usage_snapshot_with_layer_windows_and_compression(
            &[None, None],
            KvCompressionConfig::turboquant_shadow(),
        );
        assert_eq!(usage.kv_compression.status_code, 2);
        assert_eq!(usage.kv_compression.candidate_token_layers, 0);

        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 600, 4], MlxDtype::Bfloat16, None);
        cache.append(0, k, v);
        cache.seq_len = 600;

        let usage = cache.usage_snapshot_with_layer_windows_and_compression(
            &[Some(512)],
            KvCompressionConfig::turboquant_shadow(),
        );
        assert_eq!(usage.kv_compression.status_code, 3);
        assert_eq!(usage.kv_compression.eligible_layers, 0);
        assert_eq!(usage.kv_compression.candidate_token_layers, 0);
    }

    #[test]
    fn usage_snapshot_ignores_unwritten_sliding_window_layers() {
        let mut cache = MlxKVCache::new(2);
        let k = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 300, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 300;

        let usage = cache.usage_snapshot_with_layer_windows(&[Some(128), Some(128)]);
        assert_eq!(usage.full_attention_layers, 1);
        assert_eq!(usage.sliding_window_layers, 1);
        assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 256);
    }

    #[test]
    fn usage_snapshot_does_not_report_reclaimable_capacity_inside_window() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 120, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 120, 4], MlxDtype::Bfloat16, None);

        cache.append(0, k, v);
        cache.seq_len = 120;

        let usage = cache.usage_snapshot_with_layer_windows(&[Some(512)]);
        assert_eq!(usage.capacity_tokens, 256);
        assert_eq!(usage.sliding_window_layers, 1);
        assert_eq!(usage.sliding_window_retained_tokens, 120);
        assert_eq!(usage.sliding_window_reclaimable_capacity_tokens, 0);
        assert_eq!(usage.sliding_window_reclaimable_capacity_bytes, 0);
    }

    #[test]
    #[should_panic(expected = "matching K/V shapes")]
    fn append_rejects_mismatched_kv_shapes() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 4, 4], MlxDtype::Bfloat16, None);

        let _ = cache.append(0, k, v);
    }

    #[test]
    #[should_panic(expected = "cannot change head_dim")]
    fn append_rejects_existing_layer_shape_drift() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let _ = cache.append(0, k, v);
        cache.seq_len = 3;

        let k = zeros(&[1, 2, 1, 5], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 1, 5], MlxDtype::Bfloat16, None);
        let _ = cache.append(0, k, v);
    }

    #[test]
    #[should_panic(expected = "requires matching K/V dtypes")]
    fn append_rejects_mismatched_kv_dtypes() {
        let mut cache = MlxKVCache::new(1);
        let k = zeros(&[1, 2, 3, 4], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 2, 3, 4], MlxDtype::Float32, None);

        let _ = cache.append(0, k, v);
    }

    #[test]
    fn usage_snapshot_tracks_linear_state_bytes() {
        let mut cache = MlxKVCache::new(1);
        let conv = zeros(&[1, 3, 14], MlxDtype::Float32, None);
        let recurrent = zeros(&[1, 4, 8, 6], MlxDtype::Float32, None);

        cache.set_linear_state(0, conv, recurrent);

        let usage = cache.usage_snapshot();
        assert_eq!(usage.linear_state_layers, 1);
        assert_eq!(usage.linear_state_bytes, 936);
    }

    #[test]
    fn peek_source_kv_reuses_cached_views_from_append() {
        use mlx_sys::eval;
        // Two-layer cache: layer 0 is the source, layer 1 is a KV-shared consumer.
        let mut cache = MlxKVCache::new(2);

        let k = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        let (k_from_append, _) = cache.append(0, k, v);
        cache.seq_len = 4;

        let (k_from_peek, _) = cache.peek_source_kv(0, 0);

        // Materialise both arrays. If peek returned the same lazy node as append,
        // the results must be numerically identical (same shape and dtype).
        eval(&[&k_from_append, &k_from_peek]);
        assert_eq!(k_from_append.shape(), k_from_peek.shape());
        assert_eq!(k_from_append.dtype(), k_from_peek.dtype());

        // After a buffer grow, last_k_view is cleared and peek falls back to a
        // fresh slice — verify the fallback also produces the correct shape.
        let k2 = zeros(&[1, 1, 300, 8], MlxDtype::Bfloat16, None);
        let v2 = zeros(&[1, 1, 300, 8], MlxDtype::Bfloat16, None);
        cache.append(0, k2, v2);
        cache.seq_len = 304;

        let (k_grow, _) = cache.peek_source_kv(0, 0);
        eval(&[&k_grow]);
        assert_eq!(k_grow.shape(), vec![1, 1, 304, 8]);
    }

    #[test]
    fn append_with_retained_window_returns_windowed_cached_views() {
        use mlx_sys::eval;

        let mut cache = MlxKVCache::new(2);
        let k = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);

        let (k_from_append, v_from_append) = cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 6;
        let (k_from_peek, v_from_peek) = cache.peek_source_kv(0, 0);

        eval(&[&k_from_append, &v_from_append, &k_from_peek, &v_from_peek]);
        assert_eq!(k_from_append.shape(), vec![1, 1, 4, 8]);
        assert_eq!(v_from_append.shape(), vec![1, 1, 4, 8]);
        assert_eq!(k_from_peek.shape(), vec![1, 1, 4, 8]);
        assert_eq!(v_from_peek.shape(), vec![1, 1, 4, 8]);
    }

    #[test]
    fn rotating_sliding_decode_uses_bounded_backing_store() {
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);

        let k = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 6, 8], MlxDtype::Bfloat16, None);
        let (prefill_k, _) = cache.append(0, k, v);
        cache.seq_len = 6;
        assert_eq!(prefill_k.shape(), vec![1, 1, 6, 8]);

        let next_k = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        let next_v = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        let (decode_k, decode_v) = cache.append_with_retained_window(0, next_k, next_v, Some(4));

        let lkv = cache.layers[0].as_ref().expect("layer cache");
        assert_eq!(lkv.capacity, 4);
        assert_eq!(lkv.rotating_window, Some(4));
        assert_eq!(decode_k.shape(), vec![1, 1, 4, 8]);
        assert_eq!(decode_v.shape(), vec![1, 1, 4, 8]);
    }

    #[test]
    fn trim_to_rejects_rollback_after_rotating_sliding_decode() {
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        let k = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        let v = zeros(&[1, 1, 4, 8], MlxDtype::Bfloat16, None);
        cache.append(0, k, v);
        cache.seq_len = 4;

        let next_k = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        let next_v = zeros(&[1, 1, 1, 8], MlxDtype::Bfloat16, None);
        cache.append_with_retained_window(0, next_k, next_v, Some(4));
        cache.seq_len = 5;

        assert!(!cache.trim_to(4));
        assert_eq!(cache.seq_len, 5);
    }

    // ── Bounded-rollback rotating ring tests ──

    /// `[1, 1, len, head_dim]` f32 array where token row `i` is filled with
    /// `values[i]`, so slot contents are identifiable after ring writes.
    fn tokens_f32(values: &[f32], head_dim: usize) -> MlxArray {
        let data: Vec<f32> = values
            .iter()
            .flat_map(|&value| std::iter::repeat_n(value, head_dim))
            .collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, 1, values.len() as i32, head_dim as i32],
            MlxDtype::Float32,
        )
    }

    /// First element of each token row of a `[1, 1, len, head_dim]` array.
    fn token_row_values(arr: &MlxArray, head_dim: usize) -> Vec<f32> {
        eval(&[arr]);
        arr.data_f32().chunks(head_dim).map(|row| row[0]).collect()
    }

    #[test]
    fn sliding_ring_layout_gates_by_mode_seq_and_crossing() {
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 10;
        // Rotation disabled: never a ring.
        assert_eq!(cache.sliding_ring_layout(Some(4), 1), None);
        cache.set_rotating_sliding_decode(true);
        // Pure mode: single-token only.
        let pure = cache.sliding_ring_layout(Some(4), 1).expect("pure ring");
        assert_eq!((pure.window, pure.capacity, pure.write_start), (4, 4, 10));
        assert!(!pure.needs_mask(1));
        assert_eq!(cache.sliding_ring_layout(Some(4), 2), None);
        // Bounded mode: multi-token up to the slack, always masked.
        cache.set_rotating_sliding_slack(3);
        let ring = cache.sliding_ring_layout(Some(4), 3).expect("bounded ring");
        assert_eq!((ring.window, ring.capacity, ring.write_start), (4, 7, 10));
        assert!(ring.needs_mask(1));
        assert_eq!(cache.sliding_ring_layout(Some(4), 4), None);
        // Not yet crossing the window, and no window at all: ordered path.
        cache.seq_len = 2;
        assert_eq!(cache.sliding_ring_layout(Some(4), 2), None);
        cache.seq_len = 10;
        assert_eq!(cache.sliding_ring_layout(None, 1), None);
    }

    #[test]
    fn bounded_ring_multi_token_append_places_tokens_by_slot_and_wraps() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3); // window 4 → capacity 7
        // Prefill 4 tokens (values 1..=4 for tokens 0..=3).
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;

        // 3-token verify-style append crosses the window: tokens 4, 5, 6.
        let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let (ck, _) = cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;

        let lkv = cache.layers[0].as_ref().expect("layer exists");
        assert_eq!(lkv.rotating_window, Some(4));
        assert_eq!(lkv.capacity, 7);
        assert_eq!(ck.shape(), vec![1, 1, 7, HD as i32]);
        // Conversion copies tokens 1..=3 (window - 1 back from write_start 4)
        // to slots 1..=3; new tokens 4..=6 land at slots 4..=6; slot 0 (token
        // 0's slot) was outside the copy range and stays zero.
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );

        // Two single-token appends wrap: token 7 → slot 0, token 8 → slot 1.
        for (t, value) in [(7usize, 8.0f32), (8, 9.0)] {
            let k = tokens_f32(&[value], HD);
            let v = tokens_f32(&[value], HD);
            cache.append_with_retained_window(0, k, v, Some(4));
            cache.seq_len = t + 1;
        }
        let lkv = cache.layers[0].as_ref().expect("layer exists");
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn bounded_ring_trim_within_slack_rewrites_same_slots() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3);
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        // Verify forward: draft tokens 4, 5, 6 (values 5, 6, 7).
        let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;

        // Reject the last two draft tokens (rollback depth 2 <= slack 3).
        assert!(cache.trim_to(5));
        assert_eq!(cache.seq_len, 5);

        // The corrected continuation rewrites tokens 5 and 6 with new values;
        // they land in the same slots the rejected tokens occupied.
        let k = tokens_f32(&[60.0, 70.0], HD);
        let v = tokens_f32(&[60.0, 70.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;
        let lkv = cache.layers[0].as_ref().expect("layer exists");
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![0.0, 2.0, 3.0, 4.0, 5.0, 60.0, 70.0]
        );

        // Rollback deeper than the slack is refused (fail-closed).
        assert!(!cache.trim_to(3));
        assert_eq!(cache.seq_len, 7);
        // Pure rings (slack 0 on another cache) still refuse any real trim —
        // covered by trim_to_rejects_rollback_after_rotating_sliding_decode.
    }

    /// The Gemma4 assistant drafter reads target KV between appends via
    /// `peek_layer_kv` — including right after a verify rollback, when the
    /// cached views are cleared and the ordered `[0, seq_len)` fallback
    /// would slice past a ring's capacity. Rotated layers must return the
    /// full ring plus geometry for the slot-validity mask.
    #[test]
    fn peek_layer_kv_returns_full_ring_after_rollback() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3); // window 4 → capacity 7
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        let k = tokens_f32(&[5.0, 6.0, 7.0], HD);
        let v = tokens_f32(&[5.0, 6.0, 7.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 7;
        // Partial reject: views are cleared, seq_len (5) < ring capacity (7).
        assert!(cache.trim_to(5));

        let (k, _v) = cache.peek_layer_kv(0).expect("layer peek");
        assert_eq!(k.shape(), vec![1, 1, 7, HD as i32]);
        let ring = cache.layer_sliding_ring(0).expect("ring geometry");
        assert_eq!((ring.window, ring.capacity, ring.write_start), (4, 7, 5));

        // The end-anchored drafter mask keeps exactly the last `window` live
        // tokens (1..=4) and excludes the rolled-back slots (tokens 5, 6 at
        // slots 5, 6) and token 0's never-copied slot 0.
        let mask = crate::attention_mask::create_ring_sliding_mask(
            1,
            ring.window,
            ring.capacity,
            ring.write_start - 1,
        );
        eval(&[&mask]);
        let bits: Vec<u8> =
            unsafe { std::slice::from_raw_parts(mask.data_raw(), mask.nbytes()).to_vec() };
        assert_eq!(bits, vec![0, 1, 1, 1, 1, 0, 0]);

        // Un-rotated layers keep the ordered peek contract.
        let plain = MlxKVCache::new(1);
        assert!(plain.layer_sliding_ring(0).is_none());
    }

    /// End-to-end oracle: masked SDPA over the ring (unordered slots + slot-
    /// validity mask) must equal unmasked SDPA over the ordered sliding
    /// window from a plain cache, through a conversion → wrap → rollback →
    /// rewrite trajectory. This is the property that makes bounded rings a
    /// drop-in for ordered window views.
    #[test]
    fn bounded_ring_sdpa_matches_ordered_window_reference() {
        use crate::attention_mask::create_ring_sliding_mask;
        use mlx_sys::{ScaledDotProductAttentionMask, scaled_dot_product_attention_with_mask};
        const HD: usize = 4;
        const WINDOW: usize = 4;
        const SLACK: usize = 3;
        let scale = 1.0 / (HD as f32).sqrt();

        let mut ring_cache = MlxKVCache::new(1);
        ring_cache.set_rotating_sliding_decode(true);
        ring_cache.set_rotating_sliding_slack(SLACK);
        let mut plain_cache = MlxKVCache::new(1);

        // Distinct K/V rows per token so misplaced slots change the output.
        let tok = |t: usize| ((t + 1) as f32) * 0.25;
        let prefill: Vec<f32> = (0..WINDOW).map(tok).collect();
        for cache in [&mut ring_cache, &mut plain_cache] {
            let k = tokens_f32(&prefill, HD);
            let v = tokens_f32(&prefill, HD);
            cache.append(0, k, v);
            cache.seq_len = WINDOW;
        }

        // Trajectory: 3-token verify (tokens 4-6), reject 2 (trim to 5),
        // 2-token re-verify (tokens 5-6 with new values), then a single-token
        // step (token 7) that wraps the ring.
        struct Step {
            values: Vec<f32>,
            trim_to: Option<usize>,
        }
        let steps = [
            Step {
                values: vec![tok(4), tok(5), tok(6)],
                trim_to: Some(5),
            },
            Step {
                values: vec![9.5, 10.5],
                trim_to: None,
            },
            Step {
                values: vec![11.5],
                trim_to: None,
            },
        ];

        for step in steps {
            let seq = step.values.len();
            let write_start = ring_cache.seq_len;
            assert_eq!(write_start, plain_cache.seq_len, "caches stay in sync");
            let q = tokens_f32(&step.values, HD); // queries; values arbitrary

            // Ring side: append with the raw window, mask over the ring.
            let ring = ring_cache
                .sliding_ring_layout(Some(WINDOW), seq)
                .expect("trajectory stays on the ring past the window");
            let k = tokens_f32(&step.values, HD);
            let v = tokens_f32(&step.values, HD);
            let (ring_k, ring_v) = ring_cache.append_with_retained_window(0, k, v, Some(WINDOW));
            let ring_mask =
                create_ring_sliding_mask(seq, ring.window, ring.capacity, ring.write_start);
            let ring_out = scaled_dot_product_attention_with_mask(
                &q,
                &ring_k,
                &ring_v,
                scale,
                ScaledDotProductAttentionMask::Array(&ring_mask),
                None,
            );

            // Plain side: ordered append; reference is the ordered full view
            // with the standard causal sliding-window mask, so both sides run
            // the same masked-SDPA kernel and the comparison isolates ring
            // slot/mask placement (kernel-level masked-vs-unmasked numeric
            // drift is ~1e-3 in f32 and not what this test is about).
            let k = tokens_f32(&step.values, HD);
            let v = tokens_f32(&step.values, HD);
            plain_cache.append_with_retained_window(0, k, v, None);
            let plain = plain_cache.layers[0].as_ref().expect("layer exists");
            let write_end = (write_start + seq) as i32;
            let ones = [1i32, 1, 1, 1];
            let ordered_k = slice(
                &plain.k,
                &[0, 0, 0, 0],
                &[1, 1, write_end, HD as i32],
                &ones,
                None,
            );
            let ordered_v = slice(
                &plain.v,
                &[0, 0, 0, 0],
                &[1, 1, write_end, HD as i32],
                &ones,
                None,
            );
            let ordered_mask =
                crate::attention_mask::create_causal_mask(seq, write_start, Some(WINDOW));
            let want = scaled_dot_product_attention_with_mask(
                &q,
                &ordered_k,
                &ordered_v,
                scale,
                ScaledDotProductAttentionMask::Array(&ordered_mask),
                None,
            );
            eval(&[&ring_out, &want]);
            let got = ring_out.data_f32().to_vec();
            let want = want.data_f32().to_vec();
            for i in 0..seq {
                for d in 0..HD {
                    let g = got[i * HD + d];
                    let w = want[i * HD + d];
                    assert!(
                        (g - w).abs() < 1e-5,
                        "step query {i} dim {d}: ring {g} vs ordered {w}"
                    );
                }
            }

            let new_len = write_start + seq;
            ring_cache.seq_len = new_len;
            plain_cache.seq_len = new_len;
            if let Some(target) = step.trim_to {
                assert!(ring_cache.trim_to(target), "trim within slack");
                assert!(plain_cache.trim_to(target));
            }
        }
    }

    // ── F3 M1 serialize / deserialize round-trip tests ──

    fn build_fa_array_f32(seq_len: usize, n_kv_heads: i32, head_dim: i32) -> MlxArray {
        let total = (n_kv_heads as usize) * seq_len * (head_dim as usize);
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, n_kv_heads, seq_len as i32, head_dim],
            MlxDtype::Float32,
        )
    }

    fn build_mla_latent_f32(seq_len: usize, inner: i32) -> MlxArray {
        let total = seq_len * (inner as usize);
        let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.0007).collect();
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &[1, 1, seq_len as i32, inner],
            MlxDtype::Float32,
        )
    }

    fn host_f32(arr: &MlxArray) -> Vec<f32> {
        eval(&[arr]);
        arr.data_f32().to_vec()
    }

    #[test]
    fn serialize_empty_cache_roundtrips() {
        let cache = MlxKVCache::new(3);
        let bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");
        assert_eq!(restored.seq_len, 0);
        assert_eq!(restored.layers.len(), 3);
        assert!(restored.layers.iter().all(Option::is_none));
        assert!(restored.glm_mla_layers.iter().all(Option::is_none));
        assert!(
            restored
                .linear_layers
                .iter()
                .all(|l| l.conv_state.is_none() && l.recurrent_state.is_none())
        );
    }

    #[test]
    fn serialize_fa_cache_roundtrips_values() {
        // Two FA layers, varying head counts to catch shape mistakes.
        let mut cache = MlxKVCache::new(2);
        let seq_len = 4;
        let k0 = build_fa_array_f32(seq_len, 2, 8);
        let v0 = build_fa_array_f32(seq_len, 2, 8);
        let k1 = build_fa_array_f32(seq_len, 4, 16);
        let v1 = build_fa_array_f32(seq_len, 4, 16);
        cache.layers[0] = Some(LayerKV {
            last_k_view: None,
            last_v_view: None,
            n_kv_heads: 2,
            head_dim: 8,
            capacity: seq_len,
            rotating_window: None,
            dtype: MlxDtype::Float32,
            k: k0,
            v: v0,
        });
        cache.layers[1] = Some(LayerKV {
            last_k_view: None,
            last_v_view: None,
            n_kv_heads: 4,
            head_dim: 16,
            capacity: seq_len,
            rotating_window: None,
            dtype: MlxDtype::Float32,
            k: k1,
            v: v1,
        });
        cache.seq_len = seq_len;
        cache.growth_count = 7;

        let bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");

        assert_eq!(restored.seq_len, seq_len);
        assert_eq!(restored.growth_count, 7);
        for layer in 0..2 {
            let orig = cache.layers[layer].as_ref().unwrap();
            let back = restored.layers[layer].as_ref().expect("layer present");
            assert_eq!(back.n_kv_heads, orig.n_kv_heads);
            assert_eq!(back.head_dim, orig.head_dim);
            assert_eq!(back.capacity, orig.capacity);
            assert_eq!(back.dtype, orig.dtype);
            assert_eq!(host_f32(&back.k), host_f32(&orig.k));
            assert_eq!(host_f32(&back.v), host_f32(&orig.v));
        }
    }

    #[test]
    fn serialize_turboquant_shadow_storage_roundtrips_decode_state() {
        let mut cache = MlxKVCache::new(1);
        let k_data: Vec<f32> = (0..96).map(|idx| ((idx % 17) as f32 - 8.0) / 9.0).collect();
        let v_data: Vec<f32> = (0..96)
            .map(|idx| ((idx % 19) as f32 - 9.0) / 11.0)
            .collect();
        let k = MlxArray::from_raw_data(
            k_data.as_ptr().cast(),
            std::mem::size_of_val(k_data.as_slice()),
            &[1, 2, 6, 8],
            MlxDtype::Float32,
        );
        let v = MlxArray::from_raw_data(
            v_data.as_ptr().cast(),
            std::mem::size_of_val(v_data.as_slice()),
            &[1, 2, 6, 8],
            MlxDtype::Float32,
        );
        let compression = KvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..KvCompressionConfig::turboquant_shadow()
        };
        let queries = vec![
            vec![0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 1.4, -1.6],
            vec![-0.3, 0.5, -0.7, 0.9, -1.1, 1.3, -1.5, 1.7],
        ];

        cache.append(0, k, v);
        cache.seq_len = 6;
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));
        let expected = cache
            .debug_turboquant_shadow_decode_attention_for_layer(0, &queries, 2)
            .expect("decode before serialization");

        let bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");
        let usage = restored.turboquant_shadow_storage_usage();
        assert_eq!(usage.layers, 1);
        assert_eq!(usage.token_layers, 4);
        assert_eq!(usage.written_slots, 8);
        assert_eq!(restored.turboquant_shadow_storage_cold_tokens(0), Some(4));
        assert_eq!(
            restored.turboquant_shadow_layers[0]
                .as_ref()
                .expect("shadow storage")
                .buffer
                .key_codec(),
            TurboQuantKeyCodecConfig::rht_lloyd_max()
        );

        let actual = restored
            .debug_turboquant_shadow_decode_attention_for_layer(0, &queries, 2)
            .expect("decode after serialization");
        for (before_head, after_head) in expected.iter().zip(actual) {
            for (before, after) in before_head.iter().zip(after_head) {
                assert!(
                    (before - after).abs() < 1e-6,
                    "expected {before}, got {after}"
                );
            }
        }
    }

    #[test]
    fn serialize_mla_cache_roundtrips_values() {
        let mut cache = MlxKVCache::new(1);
        let seq_len = 6;
        let latent_dim = 4;
        let rope_dim = 2;
        let kv_latent = build_mla_latent_f32(seq_len, latent_dim);
        let k_pe = build_mla_latent_f32(seq_len, rope_dim);
        cache.glm_mla_layers[0] = Some(GlmMlaLayerCache {
            latent_dim,
            rope_dim,
            capacity: seq_len,
            dtype: MlxDtype::Float32,
            kv_latent,
            k_pe,
        });
        cache.seq_len = seq_len;

        let bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip");

        let orig = cache.glm_mla_layers[0].as_ref().unwrap();
        let back = restored.glm_mla_layers[0]
            .as_ref()
            .expect("mla layer present");
        assert_eq!(back.latent_dim, orig.latent_dim);
        assert_eq!(back.rope_dim, orig.rope_dim);
        assert_eq!(back.capacity, orig.capacity);
        assert_eq!(back.dtype, orig.dtype);
        assert_eq!(host_f32(&back.kv_latent), host_f32(&orig.kv_latent));
        assert_eq!(host_f32(&back.k_pe), host_f32(&orig.k_pe));
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let bytes = b"NOPE\x00\x00\x00\x00".to_vec();
        let result = MlxKVCache::try_deserialize_from_bytes(&bytes);
        assert!(matches!(
            result,
            Err(MlxKVCacheSerializeError::BadMagic) | Err(MlxKVCacheSerializeError::UnexpectedEof)
        ));
    }

    #[test]
    fn deserialize_rejects_unsupported_version() {
        let mut payload = MlxKVCache::SERIALIZE_MAGIC.to_vec();
        payload.extend_from_slice(&99u32.to_le_bytes()); // wrong version
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&0u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved
        let result = MlxKVCache::try_deserialize_from_bytes(&payload);
        assert!(matches!(
            result,
            Err(MlxKVCacheSerializeError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn deserialize_rejects_truncated_payload() {
        let cache = MlxKVCache::new(1);
        let bytes = cache.serialize_to_bytes();
        // Cut off the last byte to simulate a torn write.
        let truncated = &bytes[..bytes.len() - 1];
        let result = MlxKVCache::try_deserialize_from_bytes(truncated);
        assert!(matches!(
            result,
            Err(MlxKVCacheSerializeError::UnexpectedEof)
        ));
    }

    #[test]
    fn deserialize_rejects_undersized_byte_count() {
        // Hand-craft a payload whose tensor header declares a shape
        // requiring more bytes than `byte_count` advertises. Without
        // the pre-validation guard, `MlxArray::from_managed_data` would
        // panic; with it, we surface a structured `BadShape` error.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Single FA layer
        payload.push(MlxKVCache::LAYER_KIND_FA);
        payload.extend_from_slice(&[0u8; 7]);
        payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
        // K tensor header: f32, 4-dim shape [1, 2, 4, 8] = 64 elements × 4 bytes
        payload.push(MlxKVCache::dtype_to_tag(MlxDtype::Float32));
        payload.push(4);
        payload.extend_from_slice(&[0u8; 6]);
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&2i32.to_le_bytes());
        payload.extend_from_slice(&4i32.to_le_bytes());
        payload.extend_from_slice(&8i32.to_le_bytes());
        // Declared byte_count = 1 (too small for the declared shape)
        payload.extend_from_slice(&1u64.to_le_bytes());
        payload.push(0u8);

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("undersized byte_count must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::BadShape(4)),
            "expected BadShape(4), got {err:?}"
        );
    }

    #[test]
    fn deserialize_rejects_unknown_dtype_tag() {
        // Hand-craft a payload whose tensor header carries a dtype tag that
        // is not in `dtype_from_tag`'s match table. The deserializer must
        // fail-close (PRD §7.1 "per-layer cache type, tensor shape, dtype")
        // rather than silently accept an unknown dtype and trip MLX later.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Single FA layer with one tensor carrying an invalid dtype.
        payload.push(MlxKVCache::LAYER_KIND_FA);
        payload.extend_from_slice(&[0u8; 7]);
        payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
        // 0xEE is not a valid dtype tag in dtype_from_tag's table.
        payload.push(0xEE);
        payload.push(4); // ndim
        payload.extend_from_slice(&[0u8; 6]); // reserved
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&1i32.to_le_bytes());
        payload.extend_from_slice(&4u64.to_le_bytes()); // byte_count

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("unknown dtype tag must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::UnknownDtype(0xEE)),
            "expected UnknownDtype(0xEE), got {err:?}"
        );
    }

    #[test]
    fn deserialize_rejects_unknown_layer_kind() {
        // PRD §7.1 requires per-layer cache type validation. An unknown
        // discriminator byte at the layer header position must fail-close.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // seq_len
        payload.extend_from_slice(&0u64.to_le_bytes()); // growth_count
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes()); // layer_count
        payload.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // 0x7F is intentionally outside the four documented layer kinds
        // (EMPTY/FA/MLA/LINEAR) but inside `u8`'s range.
        payload.push(0x7F);
        payload.extend_from_slice(&[0u8; 7]);

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("unknown layer kind must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::UnknownLayerKind(0x7F)),
            "expected UnknownLayerKind(0x7F), got {err:?}"
        );
    }

    #[test]
    fn deserialize_rejects_zero_rank_tensor() {
        // Tensor rank 0 is never valid for an FA layer; the deserializer's
        // shape guard (`ndim == 0 || ndim > 4`) must reject before any
        // MlxArray construction is attempted. This is the dtype-aware
        // complement to `deserialize_rejects_undersized_byte_count`.
        let mut payload = Vec::new();
        payload.extend_from_slice(MlxKVCache::SERIALIZE_MAGIC);
        payload.extend_from_slice(&MlxKVCache::SERIALIZE_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes()); // rope_offset
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());

        payload.push(MlxKVCache::LAYER_KIND_FA);
        payload.extend_from_slice(&[0u8; 7]);
        payload.extend_from_slice(&0u64.to_le_bytes()); // rotating_window: none
        payload.push(MlxKVCache::dtype_to_tag(MlxDtype::Float32));
        payload.push(0); // ndim = 0 (invalid)
        payload.extend_from_slice(&[0u8; 6]);
        payload.extend_from_slice(&[0u8; 16]); // 4 i32 shape entries
        payload.extend_from_slice(&0u64.to_le_bytes());

        let err = MlxKVCache::try_deserialize_from_bytes(&payload)
            .err()
            .expect("rank 0 tensor must be rejected");
        assert!(
            matches!(err, MlxKVCacheSerializeError::BadShape(0)),
            "expected BadShape(0), got {err:?}"
        );
    }

    #[test]
    fn serialize_trims_fa_capacity_to_logical_seq_len() {
        // The backing buffer holds `capacity` tokens but only `seq_len`
        // are logical; the payload must carry the logical prefix only.
        let capacity = 8usize;
        let seq_len = 3usize;
        let head_dim = 4;
        let n_kv_heads = 2;
        let mut cache = MlxKVCache::new(1);
        let k = build_fa_array_f32(capacity, n_kv_heads, head_dim);
        let v = build_fa_array_f32(capacity, n_kv_heads, head_dim);
        // The slice is a strided view; materialize it before reading host
        // bytes (data_f32 on a strided view walks the backing buffer
        // linearly — the same hazard serialize_tensor_logical guards).
        let expected_k = host_f32(&contiguous(
            &slice(
                &k,
                &[0, 0, 0, 0],
                &[1, n_kv_heads, seq_len as i32, head_dim],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        ));
        cache.layers[0] = Some(LayerKV {
            last_k_view: None,
            last_v_view: None,
            n_kv_heads,
            head_dim,
            capacity,
            rotating_window: None,
            dtype: MlxDtype::Float32,
            k,
            v,
        });
        cache.seq_len = seq_len;

        let trimmed_bytes = cache.serialize_to_bytes();
        let restored = MlxKVCache::try_deserialize_from_bytes(&trimmed_bytes).expect("round-trip");
        assert_eq!(restored.seq_len, seq_len);
        let restored_layer = restored.layers[0].as_ref().expect("layer restored");
        assert_eq!(
            restored_layer.capacity, seq_len,
            "restored capacity must equal the logical length, not the source capacity"
        );
        assert_eq!(
            host_f32(&restored_layer.k),
            expected_k,
            "restored K must match the logical prefix of the source buffer"
        );

        // Payload must scale with seq_len, not capacity: the same cache
        // reporting full capacity as logical length serializes strictly more.
        cache.seq_len = capacity;
        let full_bytes = cache.serialize_to_bytes();
        assert!(
            trimmed_bytes.len() < full_bytes.len(),
            "trimmed payload ({}) must be smaller than full-capacity payload ({})",
            trimmed_bytes.len(),
            full_bytes.len()
        );
    }

    #[test]
    fn serialize_roundtrips_rope_offset() {
        let mut cache = MlxKVCache::new(1);
        cache.seq_len = 4;
        cache.rope_offset = 9;

        let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes())
            .expect("round-trip");

        assert_eq!(restored.seq_len, 4);
        assert_eq!(
            restored.rope_offset, 9,
            "rope_offset is positional state and must survive the round trip"
        );
    }

    #[test]
    fn serialize_roundtrips_rotated_ring_geometry_and_appends_continue() {
        const HD: usize = 4;
        // Build a bounded ring exactly like the rotating-decode tests:
        // window 4, slack 3 → capacity 7, tokens 0..=8 with 7 and 8 wrapped.
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        cache.set_rotating_sliding_slack(3);
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        for (t, value) in [(4usize, 5.0f32), (5, 6.0), (6, 7.0), (7, 8.0), (8, 9.0)] {
            let k = tokens_f32(&[value], HD);
            let v = tokens_f32(&[value], HD);
            cache.append_with_retained_window(0, k, v, Some(4));
            cache.seq_len = t + 1;
        }
        let expected_slots = token_row_values(&cache.layers[0].as_ref().unwrap().k, HD);
        assert_eq!(expected_slots, vec![8.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let restored = MlxKVCache::try_deserialize_from_bytes(&cache.serialize_to_bytes())
            .expect("round-trip");
        let lkv = restored.layers[0].as_ref().expect("layer restored");
        assert_eq!(lkv.rotating_window, Some(4), "ring window must survive");
        assert_eq!(lkv.capacity, 7, "ring capacity must survive");
        assert_eq!(restored.rotating_sliding_slack(), 3, "slack re-latched");
        assert_eq!(
            token_row_values(&lkv.k, HD),
            expected_slots,
            "slot-ordered ring contents must survive byte-identical"
        );

        // Post-restore decode must keep rotating: token 9 lands at slot
        // 9 % 7 = 2, not at logical position 9 of an ordered buffer.
        let mut restored = restored;
        let k = tokens_f32(&[10.0], HD);
        let v = tokens_f32(&[10.0], HD);
        restored.append_with_retained_window(0, k, v, Some(4));
        restored.seq_len = 10;
        let lkv = restored.layers[0].as_ref().unwrap();
        assert_eq!(lkv.capacity, 7, "restored ring must not regrow");
        assert_eq!(
            token_row_values(&lkv.k, HD),
            vec![8.0, 9.0, 10.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    #[should_panic(expected = "ordered KV append on rotated ring layer")]
    fn ordered_append_on_rotated_ring_fails_closed() {
        const HD: usize = 4;
        let mut cache = MlxKVCache::new(1);
        cache.set_rotating_sliding_decode(true);
        let k = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        let v = tokens_f32(&[1.0, 2.0, 3.0, 4.0], HD);
        cache.append(0, k, v);
        cache.seq_len = 4;
        // Convert to a pure ring (window 4, slack 0).
        let k = tokens_f32(&[5.0], HD);
        let v = tokens_f32(&[5.0], HD);
        cache.append_with_retained_window(0, k, v, Some(4));
        cache.seq_len = 5;

        // A 2-token forward is not ring-eligible in pure mode; before the
        // fail-closed assert this fell through to the ordered path and
        // silently grew the ring, copying slots as a token-ordered prefix.
        let k = tokens_f32(&[6.0, 7.0], HD);
        let v = tokens_f32(&[6.0, 7.0], HD);
        let _ = cache.append_with_retained_window(0, k, v, Some(4));
    }

    #[test]
    fn deserialized_cache_outlives_input_buffer() {
        // Regression test for the lifetime bug fixed alongside this
        // commit: `from_raw_data` borrows its data pointer (per the
        // mlx-sys array.rs:80 doc, "MLX does **not** copy"), so handing
        // it a slice of the caller's input buffer would leave the
        // deserialised array dangling once that buffer is freed.
        // `try_deserialize_from_bytes` must construct arrays that own
        // their data via `from_managed_data` + a heap-Box deleter.
        //
        // This test arranges the scenario explicitly: build an input
        // buffer, deserialise from it, drop the buffer, then read the
        // cache's tensors. With the fix, every read returns the
        // original byte pattern; without the fix this exhibits
        // undefined behaviour (typically reads bogus values or
        // SIGBUS / SIGSEGV under MLX's evaluator).
        let seq_len = 4;
        let head_dim = 8;
        let n_kv_heads = 2;
        let original = {
            let mut cache = MlxKVCache::new(1);
            let k = build_fa_array_f32(seq_len, n_kv_heads, head_dim);
            let v = build_fa_array_f32(seq_len, n_kv_heads, head_dim);
            cache.layers[0] = Some(LayerKV {
                last_k_view: None,
                last_v_view: None,
                n_kv_heads,
                head_dim,
                capacity: seq_len,
                rotating_window: None,
                dtype: MlxDtype::Float32,
                k,
                v,
            });
            cache.seq_len = seq_len;
            cache
        };
        let expected_k = host_f32(&original.layers[0].as_ref().unwrap().k);
        let expected_v = host_f32(&original.layers[0].as_ref().unwrap().v);

        let restored = {
            let bytes = original.serialize_to_bytes();
            MlxKVCache::try_deserialize_from_bytes(&bytes).expect("round-trip")
            // `bytes` drops here. The restored cache must remain valid.
        };

        // Read the restored tensors AFTER the input buffer has been
        // dropped. If `read_tensor` had borrowed the slice, this would
        // be UB; the managed-data + heap-owned pattern keeps it sound.
        let restored_k = host_f32(&restored.layers[0].as_ref().unwrap().k);
        let restored_v = host_f32(&restored.layers[0].as_ref().unwrap().v);
        assert_eq!(restored_k, expected_k);
        assert_eq!(restored_v, expected_v);
    }
}
