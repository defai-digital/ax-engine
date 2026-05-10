use std::mem::size_of;

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use crate::model::NativeTensorDataType;

use super::tensor::{native_dtype_size_bytes, q4km_row_byte_offset};
use super::{MetalPipelineHandle, ResolvedLinearAttentionDims, saturating_usize_to_u32};

/// Each output row is handled by one simd-group (32 threads) that cooperates
/// via simd_sum. Callers multiply output_rows by this constant for the total
/// thread count and use 32 as the threadgroup size.
pub(super) const PROJECTION_SIMD_WIDTH: u64 = 32;
/// Threadgroup size for parallel rms_norm kernels (1 TG dispatched per call or per head).
pub(super) const NORM_TG_SIZE: u64 = 256;
/// Threadgroup size for parallel argmax/sample_argmax kernels (1 TG dispatched per vocab scan).
pub(super) const ARGMAX_TG_SIZE: u64 = 1024;
/// Rows per threadgroup for Q4Km GEMV: 1 row/simdgroup x 2 simdgroups/TG.
const Q4KM_ROWS_PER_TG: u64 = 2;

#[derive(Clone, Copy)]
#[repr(C)]
struct CacheDispatchParams {
    element_count: u32,
    head_size: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct AttentionDispatchParams {
    element_count: u32,
    num_seqs: u32,
    head_count: u32,
    head_dim: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct GatherDispatchParams {
    element_count: u32,
    num_seqs: u32,
    block_size_tokens: u32,
    block_table_stride: u32,
    head_size: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CopyBlockDispatchParams {
    num_pairs: u32,
    numel_per_block_key: u32,
    numel_per_block_value: u32,
    head_size: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct LogitsProjectionDispatchParams {
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedLogitsProjectionDispatchParams {
    token_count: u32,
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
    hidden_stride: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct LogitsArgmaxDispatchParams {
    element_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedLogitsArgmaxDispatchParams {
    token_count: u32,
    vocab_rows: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RmsNormDispatchParams {
    element_count: u32,
    epsilon: f32,
    weight_offset: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedRmsNormDispatchParams {
    head_count: u32,
    head_dim: u32,
    epsilon: f32,
    weight_offset: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct FfnGateProductDispatchParams {
    element_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct VectorAddDispatchParams {
    element_count: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedRowScaleDispatchParams {
    row_count: u32,
    row_width: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedRowVectorScaleDispatchParams {
    row_count: u32,
    row_width: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct LinearAttentionConvDispatchParams {
    batch_size: u32,
    conv_dim: u32,
    conv_kernel_dim: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct LinearGatedDeltaDispatchParams {
    batch_size: u32,
    num_key_heads: u32,
    num_value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    repeat_factor: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct ModelStageRopeDispatchParams {
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    rotary_dim: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedModelStageRopeDispatchParams {
    token_count: u32,
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    freq_base: f32,
    rotary_dim: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct GroupedKvExpandDispatchParams {
    output_element_count: u32,
    kv_head_count: u32,
    heads_per_kv: u32,
    head_dim: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct EmbeddingGatherDispatchParams {
    token_count: u32,
    embedding_rows: u32,
    hidden_dim: u32,
    scale: f32,
}

#[repr(C)]
struct Q4KMProjectionDispatchParams {
    n_rows: u32,
    input_width: u32,
}

fn set_dispatch_params<T>(encoder: &ComputeCommandEncoderRef, buffer_index: u64, params: &T) {
    encoder.set_bytes(
        buffer_index,
        size_of::<T>() as u64,
        (params as *const T).cast(),
    );
}

pub(super) fn projection_dispatch_threads(output_rows: usize) -> u64 {
    (output_rows as u64)
        .saturating_mul(PROJECTION_SIMD_WIDTH)
        .max(PROJECTION_SIMD_WIDTH)
}

/// Returns (threadgroup_count, threadgroup_size) for `decode_projection_q4km`.
/// Threadgroup is (32, 2, 1) = 64 threads, covering 2 output rows per TG.
pub(super) fn q4km_dispatch(n_rows: usize) -> (MTLSize, MTLSize) {
    let tg_count = (n_rows as u64).div_ceil(Q4KM_ROWS_PER_TG).max(1);
    (MTLSize::new(tg_count, 1, 1), MTLSize::new(32, 2, 1))
}

/// Byte offset for a weight row, handling the non-linear Q4Km block layout.
pub(super) fn fused_weight_byte_offset(
    row_offset: usize,
    cols: usize,
    dtype: NativeTensorDataType,
) -> Option<usize> {
    if dtype == NativeTensorDataType::Q4Km {
        q4km_row_byte_offset(row_offset, cols)
    } else {
        row_offset
            .checked_mul(cols)?
            .checked_mul(native_dtype_size_bytes(dtype))
    }
}

/// Encode a single GEMV projection dispatch into an already-open encoder.
/// Handles both float (F16/BF16/F32) and Q4Km weight dtypes.
pub(super) fn encode_fused_projection(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &MetalPipelineHandle,
    input_buf: &Buffer,
    weight_buf: &Buffer,
    weight_byte_offset: u64,
    output_buf: &Buffer,
    weight_dtype: NativeTensorDataType,
    n_rows: u32,
    cols: u32,
    input_width: u32,
) {
    encoder.set_compute_pipeline_state(&pipeline.pipeline);
    encoder.set_buffer(0, Some(input_buf), 0);
    encoder.set_buffer(1, Some(weight_buf), weight_byte_offset);
    encoder.set_buffer(2, Some(output_buf), 0);
    if weight_dtype == NativeTensorDataType::Q4Km {
        set_q4km_projection_dispatch_params(encoder, 3, n_rows, input_width);
        let (tg_count, tg_size) = q4km_dispatch(n_rows as usize);
        encoder.dispatch_thread_groups(tg_count, tg_size);
    } else {
        set_logits_projection_dispatch_params(encoder, 3, n_rows, cols, input_width);
        encoder.dispatch_threads(
            MTLSize::new(projection_dispatch_threads(n_rows as usize), 1, 1),
            MTLSize::new(PROJECTION_SIMD_WIDTH, 1, 1),
        );
    }
}

pub(super) fn set_cache_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    head_size: u32,
) {
    let params = CacheDispatchParams {
        element_count,
        head_size,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_attention_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    num_seqs: u32,
    head_count: u32,
    head_dim: u32,
) {
    let params = AttentionDispatchParams {
        element_count,
        num_seqs,
        head_count,
        head_dim,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_gather_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    block_size_tokens: u32,
    num_seqs: u32,
    block_table_stride: u32,
    head_size: u32,
) {
    let params = GatherDispatchParams {
        element_count,
        num_seqs,
        block_size_tokens,
        block_table_stride,
        head_size,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_copy_block_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    numel_per_block_key: u32,
    numel_per_block_value: u32,
    num_pairs: u32,
    head_size: u32,
) {
    let params = CopyBlockDispatchParams {
        num_pairs,
        numel_per_block_key,
        numel_per_block_value,
        head_size,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_q4km_projection_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    n_rows: u32,
    input_width: u32,
) {
    let params = Q4KMProjectionDispatchParams {
        n_rows,
        input_width,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_logits_projection_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
) {
    let params = LogitsProjectionDispatchParams {
        vocab_rows,
        projection_cols,
        input_width,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_batched_logits_projection_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
    hidden_stride: u32,
) {
    let params = BatchedLogitsProjectionDispatchParams {
        token_count,
        vocab_rows,
        projection_cols,
        input_width,
        hidden_stride,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_logits_argmax_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
) {
    let params = LogitsArgmaxDispatchParams { element_count };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_batched_logits_argmax_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    vocab_rows: u32,
) {
    let params = BatchedLogitsArgmaxDispatchParams {
        token_count,
        vocab_rows,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_rms_norm_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    epsilon: f32,
    weight_offset: f32,
) {
    let params = RmsNormDispatchParams {
        element_count,
        epsilon,
        weight_offset,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_batched_rms_norm_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    head_count: u32,
    head_dim: u32,
    epsilon: f32,
    weight_offset: f32,
) {
    let params = BatchedRmsNormDispatchParams {
        head_count,
        head_dim,
        epsilon,
        weight_offset,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_ffn_gate_product_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
) {
    let params = FfnGateProductDispatchParams { element_count };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_vector_add_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
) {
    let params = VectorAddDispatchParams { element_count };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_batched_row_scale_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    row_count: u32,
    row_width: u32,
) {
    let params = BatchedRowScaleDispatchParams {
        row_count,
        row_width,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_batched_row_vector_scale_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    row_count: u32,
    row_width: u32,
) {
    let params = BatchedRowVectorScaleDispatchParams {
        row_count,
        row_width,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_linear_attention_conv_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    batch_size: u32,
    dims: ResolvedLinearAttentionDims,
) {
    let params = LinearAttentionConvDispatchParams {
        batch_size,
        conv_dim: saturating_usize_to_u32(dims.conv_dim),
        conv_kernel_dim: saturating_usize_to_u32(dims.conv_kernel_dim),
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_linear_gated_delta_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    batch_size: u32,
    dims: ResolvedLinearAttentionDims,
) {
    let params = LinearGatedDeltaDispatchParams {
        batch_size,
        num_key_heads: saturating_usize_to_u32(dims.num_key_heads),
        num_value_heads: saturating_usize_to_u32(dims.num_value_heads),
        key_head_dim: saturating_usize_to_u32(dims.key_head_dim),
        value_head_dim: saturating_usize_to_u32(dims.value_head_dim),
        repeat_factor: saturating_usize_to_u32(dims.repeat_factor),
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_model_stage_rope_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    rotary_dim: u32,
) {
    let params = ModelStageRopeDispatchParams {
        query_head_count,
        key_head_count,
        head_dim,
        rope_style,
        rotary_dim,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn set_batched_model_stage_rope_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    freq_base: f32,
    rotary_dim: u32,
) {
    let params = BatchedModelStageRopeDispatchParams {
        token_count,
        query_head_count,
        key_head_count,
        head_dim,
        rope_style,
        freq_base,
        rotary_dim,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_grouped_kv_expand_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    output_element_count: u32,
    kv_head_count: u32,
    heads_per_kv: u32,
    head_dim: u32,
) {
    let params = GroupedKvExpandDispatchParams {
        output_element_count,
        kv_head_count,
        heads_per_kv,
        head_dim,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}

pub(super) fn set_embedding_gather_dispatch_params(
    encoder: &ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    embedding_rows: u32,
    hidden_dim: u32,
    scale: f32,
) {
    let params = EmbeddingGatherDispatchParams {
        token_count,
        embedding_rows,
        hidden_dim,
        scale,
    };
    set_dispatch_params(encoder, buffer_index, &params);
}
