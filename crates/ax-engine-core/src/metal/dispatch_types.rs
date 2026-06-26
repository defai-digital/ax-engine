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

impl MetalDispatchKvMetadata {
    pub(super) fn gather_token_count(&self) -> u32 {
        self.cu_seq_lens.last().copied().unwrap_or(0).max(1)
    }

    pub(super) fn slot_capacity(&self) -> u32 {
        let max_direct_slot = self
            .slot_mapping
            .iter()
            .chain(self.attention_block_table.iter())
            .copied()
            .max()
            .unwrap_or(0);

        max_direct_slot
            .max(self.block_capacity().saturating_sub(1))
            .saturating_add(1)
            .max(1)
    }

    pub(super) fn block_capacity(&self) -> u32 {
        let block_span = self.block_size_tokens.saturating_sub(1);
        let max_gather_slot = self
            .gather_block_table
            .iter()
            .copied()
            .map(|base| base.saturating_add(block_span))
            .max()
            .unwrap_or(0);
        let max_copy_slot = self
            .copy_block_mapping
            .iter()
            .flat_map(|pair| {
                [
                    pair[0].saturating_add(block_span),
                    pair[1].saturating_add(block_span),
                ]
            })
            .max()
            .unwrap_or(0);

        max_gather_slot.max(max_copy_slot).saturating_add(1).max(1)
    }
}

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

impl MetalDispatchWorkload {
    pub fn from_runner_input(input: &RunnerInput) -> Result<Self, MetalRuntimeError> {
        let kv_metadata = build_dispatch_kv_metadata(input)?;
        let scheduled_requests = input.execution_batch.items.len() as u32;
        let decode_requests = input
            .execution_batch
            .items
            .iter()
            .filter(|item| item.mode == ExecutionMode::Decode)
            .count() as u32;
        let prefill_requests = scheduled_requests.saturating_sub(decode_requests);
        let scheduled_tokens = input.execution_batch.total_scheduled_tokens;
        let mut scheduled_token_ids = Vec::with_capacity(scheduled_tokens as usize);
        let mut scheduled_positions = Vec::with_capacity(scheduled_tokens as usize);
        for item in &input.execution_batch.items {
            if item.input_token_slice.len() != item.scheduled_token_count as usize {
                return Err(MetalRuntimeError::InvalidDispatchInput {
                    message: format!(
                        "request {} scheduled_token_count={} does not match input_token_slice length={}",
                        item.request_id.0,
                        item.scheduled_token_count,
                        item.input_token_slice.len()
                    ),
                });
            }
            scheduled_token_ids.extend(item.input_token_slice.iter().copied());
            for offset in 0..item.scheduled_token_count {
                scheduled_positions.push(item.position_range.start.saturating_add(offset));
            }
        }
        if scheduled_token_ids.len() as u32 != scheduled_tokens {
            return Err(MetalRuntimeError::InvalidDispatchInput {
                message: format!(
                    "execution_batch total_scheduled_tokens={} does not match flattened scheduled token count={}",
                    scheduled_tokens,
                    scheduled_token_ids.len()
                ),
            });
        }
        let resolved_blocks = input
            .block_tables
            .iter()
            .map(|resolved| resolved.block_table.block_ids.len() as u32)
            .sum::<u32>();
        let token_elements = scheduled_tokens.max(1);
        let block_elements = resolved_blocks.max(1);
        let scratch_elements = token_elements.max(block_elements);
        let kv_slot_capacity = kv_metadata.slot_capacity();
        let kv_block_capacity = kv_metadata.block_capacity();
        if scheduled_token_ids.is_empty() {
            scheduled_token_ids.push(0);
            scheduled_positions.push(0);
        }

        Ok(Self {
            scheduled_requests,
            prefill_requests,
            decode_requests,
            scheduled_tokens,
            scheduled_token_ids,
            scheduled_positions,
            resolved_blocks,
            token_elements,
            block_elements,
            scratch_elements,
            kv_slot_capacity,
            kv_block_capacity,
            numeric_layout: MetalDispatchNumericLayout::default(),
            kv_metadata,
        })
    }

    pub(super) fn with_numeric_layout(&self, numeric_layout: MetalDispatchNumericLayout) -> Self {
        let mut workload = self.clone();
        workload.numeric_layout = if numeric_layout.is_valid() {
            numeric_layout
        } else {
            MetalDispatchNumericLayout::default()
        };
        workload
    }

    pub(super) fn scheduled_numeric_elements(&self) -> u32 {
        self.token_elements
            .saturating_mul(self.numeric_layout.head_size())
    }

    pub(super) fn slot_numeric_capacity(&self) -> u32 {
        self.kv_slot_capacity
            .saturating_mul(self.numeric_layout.head_size())
    }

    pub(super) fn gather_numeric_elements(&self) -> u32 {
        self.kv_metadata
            .gather_token_count()
            .saturating_mul(self.numeric_layout.head_size())
    }

    pub(super) fn attention_numeric_elements(&self) -> u32 {
        self.token_elements
            .saturating_mul(self.numeric_layout.head_size())
    }

    pub(super) fn block_numeric_elements(&self) -> u32 {
        self.kv_metadata
            .block_size_tokens
            .max(1)
            .saturating_mul(self.numeric_layout.head_size())
    }

    pub(super) fn copy_numeric_elements(&self) -> u32 {
        (self.kv_metadata.copy_block_mapping.len() as u32)
            .max(1)
            .saturating_mul(self.block_numeric_elements())
    }
}

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

impl MetalDispatchRuntimeInfo {
    pub(super) fn from_bringup_report(report: &MetalRuntimeBringupReport) -> Self {
        Self {
            device_name: report.device.name.clone(),
            required_pipeline_count: report.required_pipelines.len() as u32,
            max_thread_execution_width: report
                .required_pipelines
                .iter()
                .map(|pipeline| pipeline.thread_execution_width)
                .max()
                .unwrap_or(0),
            binary_archive: report.binary_archive.clone(),
            command_queue_ready: report.command_queue_ready,
            model_conditioned_inputs: false,
            real_model_tensor_inputs: false,
            complete_model_forward_supported: false,
            model_bindings_prepared: false,
            model_buffers_bound: false,
            model_buffer_count: 0,
            model_buffer_bytes: 0,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: report.model.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MetalStagedInputSource {
    SyntheticTokenIds,
    ModelConditionedMiniProjection,
    ModelConditionedCpuPrefixAttention,
    ModelConditionedNativePrefixAttention,
    ModelConditionedMixedPrefixAttention,
}

#[derive(Clone, Debug)]
pub(super) struct MetalDispatchStagedInputs {
    pub(super) key: Vec<f32>,
    pub(super) value: Vec<f32>,
    pub(super) query: Vec<f32>,
    pub(super) layout: MetalDispatchNumericLayout,
    pub(super) source: MetalStagedInputSource,
    pub(super) prefix_attention_tally: PrefixAttentionExecutionTally,
    #[cfg(target_os = "macos")]
    pub(super) final_layer_hidden_state_cache: Option<ModelFinalLayerHiddenStateCache>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ModelFinalLayerHiddenStateCache {
    pub(super) hidden_states: Vec<Vec<f32>>,
    pub(super) final_layer_index: usize,
}
