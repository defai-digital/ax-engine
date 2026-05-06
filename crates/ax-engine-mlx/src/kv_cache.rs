use ax_engine_core::{MlxKvCompressionConfig, MlxKvCompressionMode};
use mlx_sys::{MlxArray, MlxDtype, eval, slice, slice_update, zeros};

use crate::turboquant::{
    FullPrecisionKvTokenVectors, TurboQuantAttentionPartitionStats, TurboQuantBlockLayout,
    TurboQuantBlockLayoutConfig, TurboQuantCodecError, TurboQuantCompressedBlockBuffer,
    TurboQuantCompressedDecodePlan, merge_attention_partition_stats, packed_kv_bytes_per_token,
    reference_decode_attention_partition_stats,
};
use crate::turboquant_metal::turboquant_fused_cold_decode_metal_two_stage_partition_stats;

/// Pre-allocated chunk size (tokens).  The buffer grows by this amount each time
/// the logical sequence length exceeds capacity, so the number of grow operations
/// per session is at most ceil(total_tokens / CHUNK).
const KV_CHUNK_TOKENS: usize = 256;

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
    compression: MlxKvCompressionConfig,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxKvCompressionDecodeOutcome {
    Fallback,
    CpuOracle,
    Metal,
}

#[derive(Clone)]
struct LayerKV {
    /// Full backing buffer: `[1, n_kv_heads, capacity, head_dim]`.
    k: MlxArray,
    v: MlxArray,
    n_kv_heads: i32,
    head_dim: i32,
    capacity: usize,
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
pub struct MlxKVCache {
    layers: Vec<Option<LayerKV>>,
    glm_mla_layers: Vec<Option<GlmMlaLayerCache>>,
    linear_layers: Vec<LinearLayerState>,
    turboquant_shadow_layers: Vec<Option<TurboQuantShadowLayerStorage>>,
    /// Current logical sequence length (token count cached).
    pub seq_len: usize,
    growth_count: u64,
    turboquant_decode_usage: MlxKvCompressionDecodeUsage,
}

impl Clone for MlxKVCache {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            glm_mla_layers: self.glm_mla_layers.clone(),
            linear_layers: self.linear_layers.clone(),
            turboquant_shadow_layers: self.turboquant_shadow_layers.clone(),
            seq_len: self.seq_len,
            growth_count: self.growth_count,
            turboquant_decode_usage: self.turboquant_decode_usage,
        }
    }
}

impl MlxKVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            glm_mla_layers: (0..num_layers).map(|_| None).collect(),
            linear_layers: (0..num_layers)
                .map(|_| LinearLayerState::default())
                .collect(),
            turboquant_shadow_layers: (0..num_layers).map(|_| None).collect(),
            seq_len: 0,
            growth_count: 0,
            turboquant_decode_usage: MlxKvCompressionDecodeUsage::default(),
        }
    }

    pub fn record_turboquant_fused_decode_attempt(
        &mut self,
        outcome: MlxKvCompressionDecodeOutcome,
    ) {
        self.turboquant_decode_usage.fused_decode_attempts = self
            .turboquant_decode_usage
            .fused_decode_attempts
            .saturating_add(1);
        match outcome {
            MlxKvCompressionDecodeOutcome::Fallback => {
                self.turboquant_decode_usage.fused_decode_fallbacks = self
                    .turboquant_decode_usage
                    .fused_decode_fallbacks
                    .saturating_add(1);
            }
            MlxKvCompressionDecodeOutcome::CpuOracle => {
                self.turboquant_decode_usage.fused_decode_successes = self
                    .turboquant_decode_usage
                    .fused_decode_successes
                    .saturating_add(1);
            }
            MlxKvCompressionDecodeOutcome::Metal => {
                self.turboquant_decode_usage.fused_decode_successes = self
                    .turboquant_decode_usage
                    .fused_decode_successes
                    .saturating_add(1);
                self.turboquant_decode_usage.fused_decode_metal_successes = self
                    .turboquant_decode_usage
                    .fused_decode_metal_successes
                    .saturating_add(1);
            }
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
        let append = validate_append_inputs(layer, self.layers.len(), &new_k, &new_v);
        let new_tokens = append.new_tokens;
        let write_start = self.seq_len;
        let write_end = write_start + new_tokens;
        let n_kv_heads = append.n_kv_heads;
        let head_dim = append.head_dim;
        let dtype = append.dtype;

        let entry = &mut self.layers[layer];
        match entry {
            None => {
                let capacity = chunk_ceiling(write_end);
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
                    n_kv_heads,
                    head_dim,
                    capacity,
                    dtype,
                });
            }
            Some(lkv) => {
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
                    self.growth_count = self.growth_count.saturating_add(1);
                }
                let start = [0i32, 0, write_start as i32, 0];
                let stop = [1i32, lkv.n_kv_heads, write_end as i32, lkv.head_dim];
                let strides = [1i32, 1, 1, 1];
                lkv.k = slice_update(&lkv.k, &new_k, &start, &stop, &strides, None);
                lkv.v = slice_update(&lkv.v, &new_v, &start, &stop, &strides, None);
            }
        }

        let lkv = self.layers[layer].as_ref().unwrap();
        let end = write_end as i32;
        let k_view = slice(
            &lkv.k,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_view = slice(
            &lkv.v,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        (k_view, v_view)
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
        let valid = prefix_len <= self.seq_len;
        self.seq_len = prefix_len.min(self.seq_len);
        for storage in self.turboquant_shadow_layers.iter_mut().flatten() {
            storage.compressed_tokens = storage.compressed_tokens.min(self.seq_len);
        }
        valid
    }

    pub fn sync_turboquant_shadow_storage(
        &mut self,
        layer_windows: &[Option<usize>],
        compression: MlxKvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> TurboQuantShadowStorageUsage {
        if !compression.is_enabled() || self.seq_len < compression.min_context_tokens {
            self.clear_turboquant_shadow_storage();
            return TurboQuantShadowStorageUsage::default();
        }

        let cold_tokens = self.seq_len.saturating_sub(compression.hot_window_tokens);
        if cold_tokens == 0 {
            self.clear_turboquant_shadow_storage();
            return TurboQuantShadowStorageUsage::default();
        }

        for layer_idx in 0..self.layers.len() {
            if layer_windows.get(layer_idx).copied().flatten().is_some() {
                self.turboquant_shadow_layers[layer_idx] = None;
                continue;
            }
            if let Some(eligible_layers) = compression_eligible_layers
                && !eligible_layers.get(layer_idx).copied().unwrap_or(false)
            {
                self.turboquant_shadow_layers[layer_idx] = None;
                continue;
            }

            let Some(lkv) = self.layers[layer_idx].as_ref() else {
                self.turboquant_shadow_layers[layer_idx] = None;
                continue;
            };

            let Some(layout) = turboquant_shadow_layout_for_layer(lkv, compression) else {
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
                });
            }

            let storage = self.turboquant_shadow_layers[layer_idx]
                .as_mut()
                .expect("storage was just initialized");
            if storage.compressed_tokens >= cold_tokens {
                continue;
            }

            eval(&[&lkv.k, &lkv.v]);
            for token_index in storage.compressed_tokens..cold_tokens {
                let heads = kv_heads_for_token(lkv, token_index);
                storage
                    .buffer
                    .write_token(token_index, &heads)
                    .expect("TurboQuant shadow storage writes validated KV slots");
            }
            storage.compressed_tokens = cold_tokens;
        }

        self.turboquant_shadow_storage_usage()
    }

    pub fn turboquant_shadow_storage_sync_due(
        &self,
        layer_windows: &[Option<usize>],
        compression: MlxKvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> bool {
        if !compression.is_enabled() {
            return false;
        }

        if self.seq_len < compression.min_context_tokens {
            return self.turboquant_shadow_layers.iter().any(Option::is_some);
        }

        let cold_tokens = self.seq_len.saturating_sub(compression.hot_window_tokens);
        if cold_tokens == 0 {
            return self.turboquant_shadow_layers.iter().any(Option::is_some);
        }

        for layer_idx in 0..self.layers.len() {
            if layer_windows.get(layer_idx).copied().flatten().is_some() {
                if self.turboquant_shadow_layers[layer_idx].is_some() {
                    return true;
                }
                continue;
            }
            if let Some(eligible_layers) = compression_eligible_layers
                && !eligible_layers.get(layer_idx).copied().unwrap_or(false)
            {
                if self.turboquant_shadow_layers[layer_idx].is_some() {
                    return true;
                }
                continue;
            }

            let Some(lkv) = self.layers[layer_idx].as_ref() else {
                if self.turboquant_shadow_layers[layer_idx].is_some() {
                    return true;
                }
                continue;
            };
            let Some(layout) = turboquant_shadow_layout_for_layer(lkv, compression) else {
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
            if cold_tokens.saturating_sub(storage.compressed_tokens) >= KV_CHUNK_TOKENS {
                return true;
            }
        }

        false
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
        let cold_stats =
            self.debug_turboquant_shadow_decode_cold_stats_metal(layer, queries, total_tokens)?;
        self.merge_turboquant_shadow_decode_cold_stats_with_hot_tail(
            layer,
            queries,
            total_tokens,
            &cold_stats,
        )
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
        if queries.len() != n_kv_heads {
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

    fn debug_turboquant_shadow_decode_cold_stats_metal(
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
        if queries.len() != n_kv_heads {
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
        let plan = TurboQuantCompressedDecodePlan::new(
            storage.layout,
            total_tokens,
            total_tokens.saturating_sub(cold_tokens),
        )?;
        let descriptor = plan.fused_decode_launch_descriptor(&storage.buffer, queries)?;
        turboquant_fused_cold_decode_metal_two_stage_partition_stats(
            descriptor,
            &storage.buffer,
            queries,
        )
    }

    fn merge_turboquant_shadow_decode_cold_stats_with_hot_tail(
        &self,
        layer: usize,
        queries: &[Vec<f32>],
        total_tokens: usize,
        cold_stats: &[TurboQuantAttentionPartitionStats],
    ) -> Result<Vec<Vec<f32>>, TurboQuantCodecError> {
        let lkv = self.layers.get(layer).and_then(Option::as_ref).ok_or(
            TurboQuantCodecError::CompressedDecodePlanIncomplete {
                cold_tokens: total_tokens,
                required_slots: 1,
                written_slots: 0,
            },
        )?;
        let n_kv_heads = usize::try_from(lkv.n_kv_heads).unwrap_or(usize::MAX);
        if cold_stats.len() != n_kv_heads {
            return Err(TurboQuantCodecError::MismatchedKvHeadCount {
                expected: n_kv_heads,
                actual: cold_stats.len(),
            });
        }
        eval(&[&lkv.k, &lkv.v]);

        let mut outputs = Vec::with_capacity(n_kv_heads);
        for (head_index, query) in queries.iter().enumerate() {
            let cold_tokens = cold_stats[head_index].token_count;
            let hot_tokens = (cold_tokens..total_tokens)
                .map(|token_index| kv_heads_for_token(lkv, token_index)[head_index].clone())
                .collect::<Vec<_>>();
            if hot_tokens.is_empty() {
                outputs.push(merge_attention_partition_stats(&[
                    cold_stats[head_index].clone()
                ])?);
            } else {
                let hot_stats = reference_decode_attention_partition_stats(query, &hot_tokens)?;
                outputs.push(merge_attention_partition_stats(&[
                    cold_stats[head_index].clone(),
                    hot_stats,
                ])?);
            }
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

    pub fn usage_snapshot(&self) -> MlxKVCacheUsage {
        self.usage_snapshot_with_layer_windows(&[])
    }

    pub fn usage_snapshot_with_layer_windows(
        &self,
        layer_windows: &[Option<usize>],
    ) -> MlxKVCacheUsage {
        self.usage_snapshot_with_layer_windows_and_compression(
            layer_windows,
            MlxKvCompressionConfig::disabled(),
        )
    }

    pub fn usage_snapshot_with_layer_windows_and_compression(
        &self,
        layer_windows: &[Option<usize>],
        compression: MlxKvCompressionConfig,
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
        compression: MlxKvCompressionConfig,
        compression_eligible_layers: Option<&[bool]>,
    ) -> MlxKVCacheUsage {
        let mut usage = MlxKVCacheUsage {
            logical_tokens: self.seq_len,
            growth_count: self.growth_count,
            ..MlxKVCacheUsage::default()
        };

        for (layer_idx, lkv) in self.layers.iter().enumerate() {
            let Some(lkv) = lkv else {
                continue;
            };
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
        compression: MlxKvCompressionConfig,
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
            let compressed_bytes_per_token =
                packed_kv_bytes_per_token(elements_per_token_usize, key_bits, value_bits)
                    .expect("TurboQuant shadow presets use supported packed bit widths")
                    as u64;

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

        if matches!(compression.mode, MlxKvCompressionMode::TurboQuantShadow) {
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
    /// `new_tokens` must equal the number of new tokens being processed this
    /// step (matches the `write_end = seq_len + new_tokens` slice written by the
    /// source layer's `append` call earlier in the same forward pass).
    pub fn peek_source_kv(&self, source_layer: usize, new_tokens: usize) -> (MlxArray, MlxArray) {
        let lkv = self.layers[source_layer]
            .as_ref()
            .expect("KV-shared source layer has no cached KV — source layer must appear earlier");
        let end = (self.seq_len + new_tokens) as i32;
        let k_view = slice(
            &lkv.k,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_view = slice(
            &lkv.v,
            &[0, 0, 0, 0],
            &[1, lkv.n_kv_heads, end, lkv.head_dim],
            &[1, 1, 1, 1],
            None,
        );
        (k_view, v_view)
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
        self.growth_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::reference_decode_attention;

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
            MlxKvCompressionConfig::turboquant_shadow(),
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
        let compression = MlxKvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..MlxKvCompressionConfig::turboquant_shadow()
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
        let compression = MlxKvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..MlxKvCompressionConfig::turboquant_shadow()
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
        let compression = MlxKvCompressionConfig {
            hot_window_tokens: 2,
            min_context_tokens: 4,
            ..MlxKvCompressionConfig::turboquant_shadow()
        };

        cache.append(0, k, v);
        cache.seq_len = 6;
        assert!(cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])));
        cache.sync_turboquant_shadow_storage(&[None], compression, Some(&[true]));
        assert!(!cache.turboquant_shadow_storage_sync_due(&[None], compression, Some(&[true])));

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
            MlxKvCompressionConfig::turboquant_shadow(),
            Some(&[false, true]),
        );

        assert_eq!(usage.kv_compression.status_code, 1);
        assert_eq!(usage.kv_compression.eligible_layers, 1);
        assert_eq!(usage.kv_compression.candidate_token_layers, 344);

        let usage = cache.usage_snapshot_with_layer_windows_compression_and_layer_eligibility(
            &[None, None],
            MlxKvCompressionConfig::turboquant_shadow(),
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
            MlxKvCompressionConfig::turboquant_shadow(),
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
            MlxKvCompressionConfig::turboquant_shadow(),
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
}
