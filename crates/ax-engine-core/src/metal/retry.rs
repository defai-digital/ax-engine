use crate::model::{NativeModelArtifacts, NativeTensorDataType};

use super::feedback::{
    batched_ffn_gate_product_feedback_key, ffn_gate_product_feedback_key,
    logits_argmax_feedback_key, optional_kernel_allowed, projection_feedback_key,
    rms_norm_feedback_key, rope_feedback_key, vector_add_feedback_key,
};
use super::tensor::tensor_matrix_dimensions;
use super::{
    MetalFfnGateUpBindings, MetalNativeModelBufferBindings, MetalNativeTensorBufferBinding,
    MetalRuntimeBringup, ModelBoundDecodeDims, ModelFfnActivation, ModelStageDims,
    ModelStageRopeStyle,
};

pub(super) fn batched_projection_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_projection_kernel(binding.native_dtype)
            .is_some()
    })
}

pub(super) fn batched_rms_norm_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    weight_binding: &MetalNativeTensorBufferBinding,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_rms_norm_kernel(weight_binding.native_dtype)
            .is_some()
    })
}

pub(super) fn batched_rms_norm_without_weights_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_rms_norm_kernel(NativeTensorDataType::F32)
            .is_some()
    })
}

pub(super) fn batched_vector_add_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .vector_add_kernel()
            .is_some()
    })
}

pub(super) fn batched_ffn_gate_product_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    activation: ModelFfnActivation,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .ffn_gate_product_kernel(activation)
            .is_some()
    })
}

pub(super) fn batched_attention_output_gate_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .attention_output_gate_kernel()
            .is_some()
    })
}

pub(super) fn batched_linear_attention_gate_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_gate_kernel()
            .is_some()
    })
}

pub(super) fn batched_linear_attention_beta_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_beta_kernel()
            .is_some()
    })
}

pub(super) fn batched_linear_attention_decay_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_decay_kernel()
            .is_some()
    })
}

pub(super) fn batched_linear_attention_conv_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    dtype: NativeTensorDataType,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_conv1d_kernel(dtype)
            .is_some()
    })
}

pub(super) fn batched_linear_attention_recurrent_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_gated_delta_step_kernel()
            .is_some()
    })
}

pub(super) fn batched_rope_split_retry_worthwhile(bringup: Option<&MetalRuntimeBringup>) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .apply_rope_batched_f32
            .is_some()
    })
}

pub(super) fn batched_grouped_kv_expand_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .expand_grouped_kv_heads_f32
            .is_some()
    })
}

pub(super) fn single_projection_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
    output_dim: usize,
    input_width: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        tensor_matrix_dimensions(&binding.meta.spec).is_some_and(|(_, cols)| {
            bringup
                .state
                .optional_kernel_dispatch_plan
                .projection_kernel(binding.native_dtype)
                .is_some_and(|(kernel_name, _)| {
                    optional_kernel_allowed(
                        bringup,
                        &projection_feedback_key(kernel_name, output_dim, input_width, cols),
                    )
                })
        })
    })
}

pub(super) fn single_rms_norm_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    weight_binding: &MetalNativeTensorBufferBinding,
    value_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .rms_norm_kernel(weight_binding.native_dtype)
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(bringup, &rms_norm_feedback_key(kernel_name, value_count))
            })
    })
}

pub(super) fn single_rms_norm_without_weights_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    value_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .rms_norm_kernel(NativeTensorDataType::F32)
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(bringup, &rms_norm_feedback_key(kernel_name, value_count))
            })
    })
}

pub(super) fn single_vector_add_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    element_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .vector_add_kernel()
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(
                    bringup,
                    &vector_add_feedback_key(kernel_name, element_count),
                )
            })
    })
}

pub(super) fn single_ffn_gate_product_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    activation: ModelFfnActivation,
    value_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .ffn_gate_product_kernel(activation)
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(
                    bringup,
                    &ffn_gate_product_feedback_key(kernel_name, value_count),
                )
            })
    })
}

pub(super) fn single_rope_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    stage_dims: ModelStageDims,
    rotary_dim: usize,
    rope_style: ModelStageRopeStyle,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .apply_rope_f32
            .is_some_and(|_| {
                optional_kernel_allowed(
                    bringup,
                    &rope_feedback_key(
                        "apply_rope_f32",
                        stage_dims.q_heads,
                        stage_dims.kv_heads,
                        stage_dims.head_dim,
                        rotary_dim,
                        rope_style,
                    ),
                )
            })
    })
}

pub(super) fn single_decode_logits_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden_width: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        tensor_matrix_dimensions(&decode_projection.meta.spec).is_some_and(
            |(vocab_rows, projection_cols)| {
                let input_width = hidden_width.min(projection_cols);
                if vocab_rows == 0 || input_width == 0 {
                    return false;
                }
                bringup
                    .state
                    .optional_kernel_dispatch_plan
                    .projection_kernel(decode_projection.native_dtype)
                    .is_some_and(|(projection_kernel_name, _)| {
                        bringup
                            .state
                            .optional_kernel_dispatch_plan
                            .logits_argmax_f32
                            .is_some_and(|_| {
                                optional_kernel_allowed(
                                    bringup,
                                    &projection_feedback_key(
                                        projection_kernel_name,
                                        vocab_rows,
                                        input_width,
                                        projection_cols,
                                    ),
                                ) && optional_kernel_allowed(
                                    bringup,
                                    &logits_argmax_feedback_key("logits_argmax_f32", vocab_rows),
                                )
                            })
                    })
            },
        )
    })
}

pub(super) fn single_attention_output_gate_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    row_width: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .attention_output_gate_kernel()
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(
                    bringup,
                    &batched_ffn_gate_product_feedback_key(kernel_name, 1, row_width),
                )
            })
    })
}

#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
pub(super) struct SingleFfnGateUpProjectionNativeRetryPolicy<'a> {
    pub(super) gate_projection: Option<&'a MetalRuntimeBringup>,
    pub(super) up_projection: Option<&'a MetalRuntimeBringup>,
}

#[allow(dead_code)]
pub(super) fn single_ffn_gate_up_projection_retry_policy<'a>(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input_width: usize,
    bringup: Option<&'a MetalRuntimeBringup>,
) -> Option<SingleFfnGateUpProjectionNativeRetryPolicy<'a>> {
    let policy = match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let native_bringup =
                single_projection_retry_worthwhile(bringup, packed, intermediate_dim, input_width)
                    .then_some(bringup)
                    .flatten();
            SingleFfnGateUpProjectionNativeRetryPolicy {
                gate_projection: native_bringup,
                up_projection: native_bringup,
            }
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            SingleFfnGateUpProjectionNativeRetryPolicy {
                gate_projection: single_projection_retry_worthwhile(
                    bringup,
                    gate_binding,
                    intermediate_dim,
                    input_width,
                )
                .then_some(bringup)
                .flatten(),
                up_projection: single_projection_retry_worthwhile(
                    bringup,
                    up_binding,
                    intermediate_dim,
                    input_width,
                )
                .then_some(bringup)
                .flatten(),
            }
        }
    };
    Some(policy)
}

#[derive(Clone, Copy, Default)]
pub(super) struct DirectDecodeSingleNativeRetryPolicy<'a> {
    pub(super) attention_output_gate: Option<&'a MetalRuntimeBringup>,
    pub(super) attention_o_projection: Option<&'a MetalRuntimeBringup>,
    pub(super) attention_residual_add: Option<&'a MetalRuntimeBringup>,
    pub(super) final_norm: Option<&'a MetalRuntimeBringup>,
    pub(super) logits_projection: Option<&'a MetalRuntimeBringup>,
}

pub(super) fn direct_decode_single_native_retry_policy<'a>(
    artifacts: &NativeModelArtifacts,
    attention_o: &MetalNativeTensorBufferBinding,
    final_norm: &MetalNativeTensorBufferBinding,
    decode_projection: &MetalNativeTensorBufferBinding,
    dims: ModelBoundDecodeDims,
    bringup: Option<&'a MetalRuntimeBringup>,
) -> Option<DirectDecodeSingleNativeRetryPolicy<'a>> {
    let hidden_vector_add = single_vector_add_retry_worthwhile(bringup, dims.hidden_dim)
        .then_some(bringup)
        .flatten();
    Some(DirectDecodeSingleNativeRetryPolicy {
        attention_output_gate: artifacts
            .manifest()
            .attn_output_gate
            .then(|| {
                single_attention_output_gate_retry_worthwhile(bringup, dims.input_width)
                    .then_some(bringup)
                    .flatten()
            })
            .flatten(),
        attention_o_projection: single_projection_retry_worthwhile(
            bringup,
            attention_o,
            dims.hidden_dim,
            dims.input_width,
        )
        .then_some(bringup)
        .flatten(),
        attention_residual_add: hidden_vector_add,
        final_norm: single_rms_norm_retry_worthwhile(bringup, final_norm, dims.hidden_dim)
            .then_some(bringup)
            .flatten(),
        logits_projection: single_decode_logits_retry_worthwhile(
            bringup,
            decode_projection,
            dims.hidden_dim,
        )
        .then_some(bringup)
        .flatten(),
    })
}
