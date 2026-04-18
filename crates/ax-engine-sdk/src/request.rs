use ax_engine_core::{
    EngineStepOutcome, MetalBinaryArchiveState, MetalCommandBufferStatus, MetalDispatchTrace,
    RequestSnapshot, RequestState, StepMetrics, StopReason,
};
use serde::{Deserialize, Serialize};

use crate::generate::{GenerateFinishReason, GenerateRouteReport};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRequestState {
    Waiting,
    Runnable,
    Running,
    BlockedOnMemory,
    Finished,
    Cancelled,
    Failed,
}

impl From<RequestState> for SessionRequestState {
    fn from(value: RequestState) -> Self {
        match value {
            RequestState::Waiting => Self::Waiting,
            RequestState::Runnable => Self::Runnable,
            RequestState::Running => Self::Running,
            RequestState::BlockedOnMemory => Self::BlockedOnMemory,
            RequestState::Finished => Self::Finished,
            RequestState::Cancelled => Self::Cancelled,
            RequestState::Failed => Self::Failed,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SessionRequestReport {
    pub request_id: u64,
    pub model_id: String,
    pub state: SessionRequestState,
    pub prompt_tokens: Vec<u32>,
    pub processed_prompt_tokens: u32,
    pub output_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_token_logprobs: Vec<Option<f32>>,
    pub prompt_len: u32,
    pub output_len: u32,
    pub max_output_tokens: u32,
    pub cancel_requested: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_plan_ref: Option<String>,
    pub route: GenerateRouteReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<GenerateFinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal_stop_reason: Option<StopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

impl From<RequestSnapshot> for SessionRequestReport {
    fn from(snapshot: RequestSnapshot) -> Self {
        Self {
            request_id: snapshot.request_id.0,
            model_id: snapshot.model_id.0,
            state: snapshot.state.into(),
            prompt_tokens: snapshot.prompt_tokens,
            processed_prompt_tokens: snapshot.processed_prompt_tokens,
            output_tokens: snapshot.generated_tokens,
            output_token_logprobs: snapshot.generated_token_logprobs,
            prompt_len: snapshot.prompt_len,
            output_len: snapshot.generated_len,
            max_output_tokens: snapshot.max_output_tokens,
            cancel_requested: snapshot.cancel_requested,
            execution_plan_ref: snapshot.execution_plan_ref,
            route: GenerateRouteReport::from_route(&snapshot.route_metadata_hint),
            finish_reason: GenerateFinishReason::from_request_state(
                snapshot.state,
                snapshot.terminal_stop_reason,
            ),
            terminal_stop_reason: snapshot.terminal_stop_reason,
            last_error: snapshot.last_error,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EngineStepReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_id: Option<u64>,
    pub scheduled_requests: u32,
    pub scheduled_tokens: u32,
    pub ttft_events: u32,
    pub prefix_hits: u32,
    pub kv_usage_blocks: u32,
    pub evictions: u32,
    pub cpu_time_us: u64,
    pub runner_time_us: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<GenerateRouteReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal_dispatch: Option<MetalDispatchStepReport>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalDispatchStepReport {
    pub command_queue_label: String,
    pub command_buffer_label: String,
    pub command_buffer_status: MetalCommandBufferStatus,
    pub runtime_device_name: String,
    pub runtime_required_pipeline_count: u32,
    pub runtime_max_thread_execution_width: u64,
    #[serde(default)]
    pub runtime_model_conditioned_inputs: bool,
    #[serde(default)]
    pub runtime_real_model_tensor_inputs: bool,
    #[serde(default)]
    pub runtime_complete_model_forward_supported: bool,
    #[serde(default)]
    pub runtime_model_bindings_prepared: bool,
    #[serde(default)]
    pub runtime_model_buffers_bound: bool,
    #[serde(default)]
    pub runtime_model_buffer_count: u32,
    #[serde(default)]
    pub runtime_model_buffer_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_model_family: Option<String>,
    #[serde(default)]
    pub execution_direct_decode_token_count: u32,
    #[serde(default)]
    pub execution_direct_decode_checksum_lo: u32,
    #[serde(default)]
    pub execution_logits_output_count: u32,
    #[serde(default)]
    pub execution_remaining_logits_handle_count: u32,
    #[serde(default)]
    pub execution_model_bound_ffn_decode: bool,
    #[serde(default)]
    pub execution_real_model_forward_completed: bool,
    #[serde(default)]
    pub execution_prefix_native_dispatch_count: u32,
    #[serde(default)]
    pub execution_prefix_cpu_reference_dispatch_count: u32,
    #[serde(default)]
    pub execution_qkv_projection_token_count: u32,
    #[serde(default)]
    pub execution_layer_continuation_token_count: u32,
    #[serde(default)]
    pub execution_logits_projection_token_count: u32,
    #[serde(default)]
    pub execution_logits_vocab_scan_row_count: u32,
    pub binary_archive_state: MetalBinaryArchiveState,
    pub binary_archive_attached_pipeline_count: u32,
    pub binary_archive_serialized: bool,
    pub arena_token_capacity: u32,
    pub arena_slot_capacity: u32,
    pub arena_attention_ref_capacity: u32,
    pub arena_gather_ref_capacity: u32,
    pub arena_gather_output_capacity: u32,
    pub arena_copy_pair_capacity: u32,
    pub arena_sequence_capacity: u32,
    pub arena_reused_existing: bool,
    pub arena_grew_existing: bool,
    pub kernels: Vec<MetalDispatchKernelStepReport>,
    pub numeric: MetalDispatchNumericStepReport,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalDispatchKernelStepReport {
    pub function_name: String,
    pub element_count: u32,
    pub threads_per_grid_width: u64,
    pub threads_per_threadgroup_width: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalDispatchNumericStepReport {
    pub key_cache_checksum: u64,
    pub attention_output_checksum: u64,
    pub gather_output_checksum: u64,
    pub copy_output_checksum: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<MetalDispatchValidationStepReport>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalDispatchValidationStepReport {
    pub expected_key_cache_checksum: u64,
    pub expected_attention_output_checksum: u64,
    pub expected_gather_output_checksum: u64,
    pub expected_copy_output_checksum: u64,
    pub attention_max_abs_diff_microunits: u32,
}

impl EngineStepReport {
    pub fn accumulate(&mut self, other: Self) {
        if self.step_id.is_none() {
            self.step_id = other.step_id;
        }
        self.scheduled_requests += other.scheduled_requests;
        self.scheduled_tokens += other.scheduled_tokens;
        self.ttft_events += other.ttft_events;
        self.prefix_hits += other.prefix_hits;
        self.kv_usage_blocks = self.kv_usage_blocks.max(other.kv_usage_blocks);
        self.evictions += other.evictions;
        self.cpu_time_us += other.cpu_time_us;
        self.runner_time_us += other.runner_time_us;
    }

    pub fn from_native_outcome(
        outcome: &EngineStepOutcome,
        metal_dispatch: Option<MetalDispatchStepReport>,
    ) -> Self {
        let route = outcome
            .schedule_plan
            .execution_batch
            .as_ref()
            .map(|batch| &batch.route_metadata)
            .filter(|route| route_has_data(route))
            .map(GenerateRouteReport::from_route);

        Self {
            step_id: outcome.metrics.step_id.map(|step_id| step_id.0),
            scheduled_requests: outcome.metrics.scheduled_requests,
            scheduled_tokens: outcome.metrics.scheduled_tokens,
            ttft_events: outcome.metrics.ttft_events,
            prefix_hits: outcome.metrics.prefix_hits,
            kv_usage_blocks: outcome.metrics.kv_usage_blocks,
            evictions: outcome.metrics.evictions,
            cpu_time_us: outcome.metrics.cpu_time_us,
            runner_time_us: outcome.metrics.runner_time_us,
            route,
            metal_dispatch,
        }
    }
}

impl From<StepMetrics> for EngineStepReport {
    fn from(metrics: StepMetrics) -> Self {
        Self {
            step_id: metrics.step_id.map(|step_id| step_id.0),
            scheduled_requests: metrics.scheduled_requests,
            scheduled_tokens: metrics.scheduled_tokens,
            ttft_events: metrics.ttft_events,
            prefix_hits: metrics.prefix_hits,
            kv_usage_blocks: metrics.kv_usage_blocks,
            evictions: metrics.evictions,
            cpu_time_us: metrics.cpu_time_us,
            runner_time_us: metrics.runner_time_us,
            route: None,
            metal_dispatch: None,
        }
    }
}

impl MetalDispatchStepReport {
    pub fn from_trace(trace: &MetalDispatchTrace) -> Self {
        Self {
            command_queue_label: trace.command_queue_label.clone(),
            command_buffer_label: trace.command_buffer_label.clone(),
            command_buffer_status: trace.command_buffer_status,
            runtime_device_name: trace.runtime.device_name.clone(),
            runtime_required_pipeline_count: trace.runtime.required_pipeline_count,
            runtime_max_thread_execution_width: trace.runtime.max_thread_execution_width,
            runtime_model_conditioned_inputs: trace.runtime.model_conditioned_inputs,
            runtime_real_model_tensor_inputs: trace.runtime.real_model_tensor_inputs,
            runtime_complete_model_forward_supported: trace
                .runtime
                .complete_model_forward_supported,
            runtime_model_bindings_prepared: trace.runtime.model_bindings_prepared,
            runtime_model_buffers_bound: trace.runtime.model_buffers_bound,
            runtime_model_buffer_count: trace.runtime.model_buffer_count,
            runtime_model_buffer_bytes: trace.runtime.model_buffer_bytes,
            runtime_model_family: trace
                .runtime
                .model
                .as_ref()
                .map(|model| model.model_family.clone()),
            execution_direct_decode_token_count: trace.execution.direct_decode_token_count,
            execution_direct_decode_checksum_lo: trace.execution.direct_decode_checksum_lo,
            execution_logits_output_count: trace.execution.logits_output_count,
            execution_remaining_logits_handle_count: trace.execution.remaining_logits_handle_count,
            execution_model_bound_ffn_decode: trace.execution.model_bound_ffn_decode,
            execution_real_model_forward_completed: trace.execution.real_model_forward_completed,
            execution_prefix_native_dispatch_count: trace.execution.prefix_native_dispatch_count,
            execution_prefix_cpu_reference_dispatch_count: trace
                .execution
                .prefix_cpu_reference_dispatch_count,
            execution_qkv_projection_token_count: trace.execution.qkv_projection_token_count,
            execution_layer_continuation_token_count: trace
                .execution
                .layer_continuation_token_count,
            execution_logits_projection_token_count: trace.execution.logits_projection_token_count,
            execution_logits_vocab_scan_row_count: trace.execution.logits_vocab_scan_row_count,
            binary_archive_state: trace.runtime.binary_archive.state,
            binary_archive_attached_pipeline_count: trace
                .runtime
                .binary_archive
                .attached_pipeline_count,
            binary_archive_serialized: trace.runtime.binary_archive.serialized,
            arena_token_capacity: trace.arena.token_capacity,
            arena_slot_capacity: trace.arena.slot_capacity,
            arena_attention_ref_capacity: trace.arena.attention_ref_capacity,
            arena_gather_ref_capacity: trace.arena.gather_ref_capacity,
            arena_gather_output_capacity: trace.arena.gather_output_capacity,
            arena_copy_pair_capacity: trace.arena.copy_pair_capacity,
            arena_sequence_capacity: trace.arena.sequence_capacity,
            arena_reused_existing: trace.arena.reused_existing,
            arena_grew_existing: trace.arena.grew_existing,
            kernels: trace
                .kernels
                .iter()
                .map(|kernel| MetalDispatchKernelStepReport {
                    function_name: kernel.function_name.clone(),
                    element_count: kernel.element_count,
                    threads_per_grid_width: kernel.threads_per_grid.width,
                    threads_per_threadgroup_width: kernel.threads_per_threadgroup.width,
                })
                .collect(),
            numeric: MetalDispatchNumericStepReport {
                key_cache_checksum: trace.numeric.key_cache_checksum,
                attention_output_checksum: trace.numeric.attention_output_checksum,
                gather_output_checksum: trace.numeric.gather_output_checksum,
                copy_output_checksum: trace.numeric.copy_output_checksum,
                validation: trace.numeric.validation.as_ref().map(|validation| {
                    MetalDispatchValidationStepReport {
                        expected_key_cache_checksum: validation.expected_key_cache_checksum,
                        expected_attention_output_checksum: validation
                            .expected_attention_output_checksum,
                        expected_gather_output_checksum: validation.expected_gather_output_checksum,
                        expected_copy_output_checksum: validation.expected_copy_output_checksum,
                        attention_max_abs_diff_microunits: validation
                            .attention_max_abs_diff_microunits,
                    }
                }),
            },
        }
    }
}

fn route_has_data(route: &ax_engine_core::RouteMetadata) -> bool {
    route.execution_plan.is_some()
        || route.attention_route.is_some()
        || route.kv_mode.is_some()
        || route.prefix_cache_path.is_some()
        || route.barrier_mode.is_some()
        || !route.crossover_decisions.is_empty()
}
