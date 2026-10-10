use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalCommandBufferStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchKvMetadata {
    pub block_size_tokens: u32,
    pub slot_mapping: Vec<u32>,
    pub attention_block_table: Vec<u32>,
    pub gather_block_table: Vec<u32>,
    pub gather_block_table_stride: u32,
    pub copy_block_mapping: Vec<[u32; 2]>,
    pub seq_lens: Vec<u32>,
    pub cu_seq_lens: Vec<u32>,
    pub scheduled_cu_seq_lens: Vec<u32>,
}

impl MetalDispatchKvMetadata {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchWorkload {
    pub scheduled_requests: u32,
    pub prefill_requests: u32,
    pub decode_requests: u32,
    pub scheduled_tokens: u32,
    pub scheduled_token_ids: Vec<u32>,
    pub scheduled_positions: Vec<u32>,
    pub resolved_blocks: u32,
    pub token_elements: u32,
    pub block_elements: u32,
    pub scratch_elements: u32,
    pub kv_slot_capacity: u32,
    pub kv_block_capacity: u32,
    #[serde(default)]
    pub numeric_layout: MetalDispatchNumericLayout,
    pub kv_metadata: MetalDispatchKvMetadata,
}

impl MetalDispatchWorkload {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchKernelTrace {
    pub function_name: String,
    pub element_count: u32,
    pub threads_per_grid: MetalThreadgroupSize,
    pub threads_per_threadgroup: MetalThreadgroupSize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchArenaInfo {
    pub token_capacity: u32,
    pub slot_capacity: u32,
    pub attention_ref_capacity: u32,
    pub gather_ref_capacity: u32,
    pub gather_output_capacity: u32,
    pub copy_pair_capacity: u32,
    pub sequence_capacity: u32,
    pub reused_existing: bool,
    pub grew_existing: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchTrace {
    pub command_queue_label: String,
    pub command_buffer_label: String,
    pub command_buffer_status: MetalCommandBufferStatus,
    pub runtime: MetalDispatchRuntimeInfo,
    pub workload: MetalDispatchWorkload,
    pub arena: MetalDispatchArenaInfo,
    #[serde(default)]
    pub execution: MetalDispatchExecutionInfo,
    pub kernels: Vec<MetalDispatchKernelTrace>,
    pub numeric: MetalDispatchNumericTrace,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct MetalDispatchExecutionInfo {
    #[serde(default)]
    pub direct_decode_token_count: u32,
    #[serde(default)]
    pub direct_decode_checksum_lo: u32,
    #[serde(default)]
    pub logits_output_count: u32,
    #[serde(default)]
    pub remaining_logits_handle_count: u32,
    #[serde(default)]
    pub model_bound_ffn_decode: bool,
    #[serde(default)]
    pub real_model_forward_completed: bool,
    #[serde(default)]
    pub prefix_native_dispatch_count: u32,
    #[serde(default)]
    pub prefix_cpu_reference_dispatch_count: u32,
    #[serde(default)]
    pub qkv_projection_token_count: u32,
    #[serde(default)]
    pub layer_continuation_token_count: u32,
    #[serde(default)]
    pub logits_projection_token_count: u32,
    #[serde(default)]
    pub logits_vocab_scan_row_count: u32,
    #[serde(default)]
    pub prefix_native_projection_row_count: u32,
    #[serde(default)]
    pub prefix_cpu_projection_row_count: u32,
    #[serde(default)]
    pub prefix_native_rms_norm_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_rms_norm_element_count: u32,
    #[serde(default)]
    pub prefix_native_ffn_activation_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_ffn_activation_element_count: u32,
    #[serde(default)]
    pub prefix_native_residual_add_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_residual_add_element_count: u32,
    #[serde(default)]
    pub prefix_native_scale_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_scale_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_projection_row_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_projection_row_count: u32,
    #[serde(default)]
    pub direct_decode_native_rms_norm_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_rms_norm_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_ffn_activation_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_ffn_activation_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_residual_add_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_residual_add_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_scale_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_scale_element_count: u32,
    #[serde(default)]
    pub direct_decode_batched_logits_group_count: u32,
    #[serde(default)]
    pub direct_decode_batched_logits_token_count: u32,
    #[serde(default)]
    pub direct_decode_batched_group_fallback_count: u32,
    #[serde(default)]
    pub direct_decode_batched_group_fallback_token_count: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct MetalNativeDenseKernelCoverage {
    #[serde(default)]
    pub projection_f32_binding_count: u32,
    #[serde(default)]
    pub projection_f16_binding_count: u32,
    #[serde(default)]
    pub projection_bf16_binding_count: u32,
    #[serde(default)]
    pub projection_unsupported_binding_count: u32,
    #[serde(default)]
    pub projection_source_quantized_binding_count: u32,
    #[serde(default)]
    pub rms_norm_f32_binding_count: u32,
    #[serde(default)]
    pub rms_norm_f16_binding_count: u32,
    #[serde(default)]
    pub rms_norm_bf16_binding_count: u32,
    #[serde(default)]
    pub rms_norm_unsupported_binding_count: u32,
    #[serde(default)]
    pub rms_norm_source_quantized_binding_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchRuntimeInfo {
    pub device_name: String,
    pub required_pipeline_count: u32,
    pub max_thread_execution_width: u64,
    pub binary_archive: MetalBinaryArchiveInfo,
    pub command_queue_ready: bool,
    #[serde(default)]
    pub model_conditioned_inputs: bool,
    #[serde(default)]
    pub real_model_tensor_inputs: bool,
    #[serde(default)]
    pub complete_model_forward_supported: bool,
    #[serde(default)]
    pub model_bindings_prepared: bool,
    #[serde(default)]
    pub model_buffers_bound: bool,
    #[serde(default)]
    pub model_buffer_count: u32,
    #[serde(default)]
    pub model_buffer_bytes: u64,
    #[serde(default)]
    pub native_dense_kernel_coverage: MetalNativeDenseKernelCoverage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<NativeModelArtifactsSummary>,
}
