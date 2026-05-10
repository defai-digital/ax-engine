use std::collections::{BTreeMap, BTreeSet};

use crate::model::NativeTensorDataType;

use super::{
    MetalDispatchWorkload, MetalNativeTensorBufferBinding, MetalRuntimeBringup,
    ModelBoundDecodeDims, ModelStageRopeStyle, PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD,
    ResolvedLinearAttentionDims,
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct MetalOptionalKernelFeedbackState {
    pub(super) consecutive_failures_by_kernel: BTreeMap<MetalOptionalKernelFeedbackKey, u32>,
    pub(super) disabled_kernels: BTreeSet<MetalOptionalKernelFeedbackKey>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum MetalOptionalKernelFeedbackKey {
    Kernel(&'static str),
    SamplerBatchedGroup {
        group_size: usize,
        logits_width: usize,
    },
    BatchedProjection {
        kernel_name: &'static str,
        row_count: usize,
        output_dim: usize,
        input_width: usize,
        hidden_stride: usize,
        matrix_cols: usize,
    },
    Projection {
        kernel_name: &'static str,
        output_dim: usize,
        input_width: usize,
        matrix_cols: usize,
    },
    Sampler {
        kernel_name: &'static str,
        logits_width: usize,
    },
    BatchedLogitsArgmax {
        kernel_name: &'static str,
        row_count: usize,
        vocab_rows: usize,
    },
    BatchedSampler {
        kernel_name: &'static str,
        row_count: usize,
        logits_width: usize,
    },
    LogitsArgmax {
        kernel_name: &'static str,
        vocab_rows: usize,
    },
    BatchedFfnGateProduct {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    FfnGateProduct {
        kernel_name: &'static str,
        value_count: usize,
    },
    Rope {
        kernel_name: &'static str,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_style: ModelStageRopeStyle,
    },
    EmbeddingGather {
        kernel_name: &'static str,
        token_count: usize,
        embedding_rows: usize,
        hidden_dim: usize,
    },
    BatchedGroupedKvExpand {
        kernel_name: &'static str,
        token_count: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    GroupedKvExpand {
        kernel_name: &'static str,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    VectorAdd {
        kernel_name: &'static str,
        element_count: usize,
    },
    BatchedRowScale {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    BatchedRowVectorScale {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    BatchedRope {
        kernel_name: &'static str,
        token_count: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_style: ModelStageRopeStyle,
    },
    RmsNorm {
        kernel_name: &'static str,
        value_count: usize,
    },
    BatchedRmsNorm {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    DirectDecodeBatchedGroup {
        group_size: usize,
        dims: ModelBoundDecodeDims,
    },
    PrefixAttentionBatchedGroup {
        scheduled_requests: u32,
        prefill_requests: u32,
        decode_requests: u32,
        scheduled_tokens: u32,
        gather_tokens: u32,
        block_size_tokens: u32,
        head_count: u32,
        head_dim: u32,
    },
    LinearAttentionConv1d {
        batch_size: usize,
        dtype: NativeTensorDataType,
        dims: ResolvedLinearAttentionDims,
    },
    LinearGatedDelta {
        batch_size: usize,
        dims: ResolvedLinearAttentionDims,
    },
}

#[allow(dead_code)]
pub(super) fn optional_kernel_name_feedback_key(
    kernel_name: &'static str,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Kernel(kernel_name)
}

pub(super) fn sampler_batched_group_feedback_key(
    group_size: usize,
    logits_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::SamplerBatchedGroup {
        group_size,
        logits_width,
    }
}

pub(super) fn batched_projection_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    output_dim: usize,
    input_width: usize,
    hidden_stride: usize,
    matrix_cols: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedProjection {
        kernel_name,
        row_count,
        output_dim,
        input_width,
        hidden_stride,
        matrix_cols,
    }
}

pub(super) fn projection_feedback_key(
    kernel_name: &'static str,
    output_dim: usize,
    input_width: usize,
    matrix_cols: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Projection {
        kernel_name,
        output_dim,
        input_width,
        matrix_cols,
    }
}

pub(super) fn sampler_feedback_key(
    kernel_name: &'static str,
    logits_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Sampler {
        kernel_name,
        logits_width,
    }
}

pub(super) fn batched_logits_argmax_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    vocab_rows: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedLogitsArgmax {
        kernel_name,
        row_count,
        vocab_rows,
    }
}

pub(super) fn batched_sampler_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    logits_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedSampler {
        kernel_name,
        row_count,
        logits_width,
    }
}

pub(super) fn logits_argmax_feedback_key(
    kernel_name: &'static str,
    vocab_rows: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::LogitsArgmax {
        kernel_name,
        vocab_rows,
    }
}

pub(super) fn batched_ffn_gate_product_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedFfnGateProduct {
        kernel_name,
        row_count,
        row_width,
    }
}

pub(super) fn ffn_gate_product_feedback_key(
    kernel_name: &'static str,
    value_count: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::FfnGateProduct {
        kernel_name,
        value_count,
    }
}

pub(super) fn rope_feedback_key(
    kernel_name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_style: ModelStageRopeStyle,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Rope {
        kernel_name,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    }
}

pub(super) fn embedding_gather_feedback_key(
    kernel_name: &'static str,
    token_count: usize,
    embedding_rows: usize,
    hidden_dim: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::EmbeddingGather {
        kernel_name,
        token_count,
        embedding_rows,
        hidden_dim,
    }
}

pub(super) fn batched_grouped_kv_expand_feedback_key(
    kernel_name: &'static str,
    token_count: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedGroupedKvExpand {
        kernel_name,
        token_count,
        q_heads,
        kv_heads,
        head_dim,
    }
}

pub(super) fn grouped_kv_expand_feedback_key(
    kernel_name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::GroupedKvExpand {
        kernel_name,
        q_heads,
        kv_heads,
        head_dim,
    }
}

pub(super) fn vector_add_feedback_key(
    kernel_name: &'static str,
    element_count: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::VectorAdd {
        kernel_name,
        element_count,
    }
}

pub(super) fn batched_row_scale_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRowScale {
        kernel_name,
        row_count,
        row_width,
    }
}

pub(super) fn batched_row_vector_scale_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRowVectorScale {
        kernel_name,
        row_count,
        row_width,
    }
}

pub(super) fn batched_rope_feedback_key(
    kernel_name: &'static str,
    token_count: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_style: ModelStageRopeStyle,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRope {
        kernel_name,
        token_count,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    }
}

pub(super) fn rms_norm_feedback_key(
    kernel_name: &'static str,
    value_count: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::RmsNorm {
        kernel_name,
        value_count,
    }
}

pub(super) fn batched_rms_norm_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRmsNorm {
        kernel_name,
        row_count,
        row_width,
    }
}

pub(super) fn batched_rms_norm_feedback_binding(
    bringup: &MetalRuntimeBringup,
    weight_binding: &MetalNativeTensorBufferBinding,
    row_count: usize,
    row_width: usize,
) -> Option<(&'static str, usize, MetalOptionalKernelFeedbackKey)> {
    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_rms_norm_kernel(weight_binding.native_dtype)?;
    Some((
        kernel_name,
        pipeline_index,
        batched_rms_norm_feedback_key(kernel_name, row_count, row_width),
    ))
}

pub(super) fn direct_decode_batched_group_feedback_key(
    group_size: usize,
    dims: ModelBoundDecodeDims,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::DirectDecodeBatchedGroup { group_size, dims }
}

pub(super) fn direct_decode_group_feedback_key_for_group(
    group_size: usize,
    dims: ModelBoundDecodeDims,
) -> Option<MetalOptionalKernelFeedbackKey> {
    (group_size > 1).then(|| direct_decode_batched_group_feedback_key(group_size, dims))
}

pub(super) fn prefix_attention_group_feedback_key(
    workload: &MetalDispatchWorkload,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::PrefixAttentionBatchedGroup {
        scheduled_requests: workload.scheduled_requests,
        prefill_requests: workload.prefill_requests,
        decode_requests: workload.decode_requests,
        scheduled_tokens: workload.scheduled_tokens,
        gather_tokens: workload.kv_metadata.gather_token_count(),
        block_size_tokens: workload.kv_metadata.block_size_tokens,
        head_count: workload.numeric_layout.head_count,
        head_dim: workload.numeric_layout.head_dim,
    }
}

pub(super) fn prefix_attention_group_feedback_key_for_item_count(
    item_count: usize,
    workload: &MetalDispatchWorkload,
) -> Option<MetalOptionalKernelFeedbackKey> {
    (item_count > 1).then(|| prefix_attention_group_feedback_key(workload))
}

pub(super) fn linear_attention_conv_feedback_key(
    batch_size: usize,
    dtype: NativeTensorDataType,
    dims: ResolvedLinearAttentionDims,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::LinearAttentionConv1d {
        batch_size,
        dtype,
        dims,
    }
}

pub(super) fn linear_gated_delta_feedback_key(
    batch_size: usize,
    dims: ResolvedLinearAttentionDims,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::LinearGatedDelta { batch_size, dims }
}

pub(super) fn optional_kernel_allowed(
    bringup: &MetalRuntimeBringup,
    kernel_key: &MetalOptionalKernelFeedbackKey,
) -> bool {
    let Ok(feedback) = bringup.state.optional_kernel_feedback.lock() else {
        return true;
    };
    optional_kernel_allowed_in_feedback_state(&feedback, kernel_key)
}

pub(super) fn optional_kernel_allowed_in_feedback_state(
    feedback: &MetalOptionalKernelFeedbackState,
    kernel_key: &MetalOptionalKernelFeedbackKey,
) -> bool {
    !feedback.disabled_kernels.contains(kernel_key)
}

pub(super) fn record_optional_kernel_result(
    bringup: &MetalRuntimeBringup,
    kernel_key: &MetalOptionalKernelFeedbackKey,
    success: bool,
) {
    let Ok(mut feedback) = bringup.state.optional_kernel_feedback.lock() else {
        return;
    };
    record_optional_kernel_feedback_state(&mut feedback, kernel_key, success);
}

pub(super) fn record_optional_kernel_feedback_state(
    feedback: &mut MetalOptionalKernelFeedbackState,
    kernel_key: &MetalOptionalKernelFeedbackKey,
    success: bool,
) {
    if success {
        feedback.consecutive_failures_by_kernel.remove(kernel_key);
        feedback.disabled_kernels.remove(kernel_key);
        return;
    }
    let consecutive_failures = feedback
        .consecutive_failures_by_kernel
        .entry(*kernel_key)
        .or_insert(0);
    *consecutive_failures = consecutive_failures.saturating_add(1);
    if *consecutive_failures >= PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        feedback.disabled_kernels.insert(*kernel_key);
    }
}
