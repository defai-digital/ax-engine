use std::collections::BTreeSet;
use std::fmt;

use crate::ids::{RequestId, StepId};
use crate::kv::BlockTableView;
use crate::metal::MetalDispatchTrace;
use crate::model::NativeModelArtifactsSummary;
use crate::sampling::StopReason;
use crate::scheduler::{ExecutionBatch, ExecutionMode, RouteMetadata};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedBlockTable {
    pub request_id: RequestId,
    pub block_table: BlockTableView,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunnerInput {
    pub block_size_tokens: u32,
    pub execution_batch: ExecutionBatch,
    pub block_tables: Vec<ResolvedBlockTable>,
    pub request_contexts: Vec<RunnerRequestContext>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunnerRequestContext {
    pub request_id: RequestId,
    pub prompt_len: u32,
    pub processed_prompt_tokens: u32,
    pub generated_len: u32,
    pub max_output_tokens: u32,
    pub deterministic_argmax_sampling: bool,
    /// Temperature for token sampling. 0.0 means greedy argmax (same as
    /// `deterministic_argmax_sampling = true`). Runners that support
    /// probabilistic sampling must use this when > 0.0; runners that don't
    /// must return an error rather than silently falling back to greedy.
    pub temperature: f32,
}

impl RunnerInput {
    pub fn request_context(&self, request_id: RequestId) -> Option<&RunnerRequestContext> {
        self.request_contexts
            .iter()
            .find(|context| context.request_id == request_id)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionStatus {
    Success,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestExecutionUpdate {
    pub request_id: RequestId,
    pub tokens_executed: u32,
    pub output_token: Option<u32>,
    pub stop_reason: Option<StopReason>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvWriteSummary {
    pub tokens_written: u32,
    pub blocks_touched: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestLogitsOutput {
    pub request_id: RequestId,
    pub logits: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunnerOutput {
    pub step_id: StepId,
    pub request_updates: Vec<RequestExecutionUpdate>,
    pub logits_handles: Vec<RequestId>,
    pub logits_outputs: Vec<RequestLogitsOutput>,
    pub kv_write_summary: KvWriteSummary,
    pub route_metadata: RouteMetadata,
    pub execution_status: ExecutionStatus,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeModelBindingSummary {
    pub bindings_prepared: bool,
    pub buffers_bound: bool,
    pub buffer_count: u32,
    pub buffer_bytes: u64,
    pub source_quantized_binding_count: u32,
    pub source_q4_k_binding_count: u32,
    pub source_q5_k_binding_count: u32,
    pub source_q6_k_binding_count: u32,
    pub source_q8_0_binding_count: u32,
}

pub trait ExecutionRunner: fmt::Debug + Send + Sync {
    fn run(&self, input: RunnerInput) -> RunnerOutput;

    fn release_request_state(&self, _request_id: RequestId) {}

    fn metal_dispatch_trace(&self) -> Option<MetalDispatchTrace> {
        None
    }

    fn native_model_artifacts_summary(&self) -> Option<NativeModelArtifactsSummary> {
        None
    }

    fn native_model_binding_summary(&self) -> Option<NativeModelBindingSummary> {
        None
    }
}

pub(crate) fn successful_runner_output_from_input(input: &RunnerInput) -> RunnerOutput {
    let mut request_updates = Vec::with_capacity(input.execution_batch.items.len());
    let mut logits_handles = Vec::new();

    for item in &input.execution_batch.items {
        if item.mode == ExecutionMode::Decode {
            logits_handles.push(item.request_id);
        }

        request_updates.push(RequestExecutionUpdate {
            request_id: item.request_id,
            tokens_executed: item.scheduled_token_count,
            output_token: None,
            stop_reason: None,
            error: None,
        });
    }

    let blocks_touched = input
        .block_tables
        .iter()
        .flat_map(|resolved| resolved.block_table.block_ids.iter().copied())
        .collect::<BTreeSet<_>>()
        .len() as u32;

    RunnerOutput {
        step_id: input.execution_batch.step_id,
        request_updates,
        logits_handles,
        logits_outputs: Vec::new(),
        kv_write_summary: KvWriteSummary {
            tokens_written: input.execution_batch.total_scheduled_tokens,
            blocks_touched,
        },
        route_metadata: forward_route_metadata(&input.execution_batch),
        execution_status: ExecutionStatus::Success,
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeterministicRunner;

impl ExecutionRunner for DeterministicRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        successful_runner_output_from_input(&input)
    }
}

fn forward_route_metadata(execution_batch: &ExecutionBatch) -> RouteMetadata {
    execution_batch.route_metadata.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RequestId;
    use crate::kv::BlockTableView;
    use crate::scheduler::{
        ExecutionBatch, ExecutionItem, ExecutionMode, PositionRange, RouteMetadata,
    };

    #[test]
    fn deterministic_runner_emits_prefill_and_decode_updates() {
        let runner = DeterministicRunner;
        let route_metadata = RouteMetadata {
            execution_plan: Some("dense.plan".into()),
            attention_route: Some("qwen3_prefill".into()),
            kv_mode: Some("paged_metadata".into()),
            prefix_cache_path: Some("metadata_lookup".into()),
            barrier_mode: Some("serial".into()),
            crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
        };
        let output = runner.run(RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(9),
                model_id: "qwen3".into(),
                execution_plan_ref: Some("dense.plan".into()),
                items: vec![
                    ExecutionItem {
                        request_id: RequestId(1),
                        mode: ExecutionMode::Prefill,
                        input_token_slice: vec![1, 2, 3],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 0,
                            end_exclusive: 3,
                        },
                        scheduled_token_count: 3,
                        block_table_ref: RequestId(1),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                    ExecutionItem {
                        request_id: RequestId(2),
                        mode: ExecutionMode::Decode,
                        input_token_slice: vec![10],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 4,
                            end_exclusive: 5,
                        },
                        scheduled_token_count: 1,
                        block_table_ref: RequestId(2),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                ],
                total_scheduled_tokens: 4,
                route_metadata: route_metadata.clone(),
            },
            block_tables: vec![
                ResolvedBlockTable {
                    request_id: RequestId(1),
                    block_table: BlockTableView {
                        cache_group_id: crate::ids::CacheGroupId(1),
                        block_ids: vec![crate::ids::BlockId(0)],
                    },
                },
                ResolvedBlockTable {
                    request_id: RequestId(2),
                    block_table: BlockTableView {
                        cache_group_id: crate::ids::CacheGroupId(1),
                        block_ids: vec![crate::ids::BlockId(1)],
                    },
                },
            ],
            request_contexts: vec![
                RunnerRequestContext {
                    request_id: RequestId(1),
                    prompt_len: 3,
                    processed_prompt_tokens: 0,
                    generated_len: 0,
                    max_output_tokens: 4,
                    deterministic_argmax_sampling: true,
                    temperature: 0.0,
                },
                RunnerRequestContext {
                    request_id: RequestId(2),
                    prompt_len: 4,
                    processed_prompt_tokens: 4,
                    generated_len: 0,
                    max_output_tokens: 4,
                    deterministic_argmax_sampling: true,
                    temperature: 0.0,
                },
            ],
        });

        assert_eq!(output.execution_status, ExecutionStatus::Success);
        assert_eq!(output.request_updates.len(), 2);
        assert_eq!(output.request_updates[0].output_token, None);
        assert_eq!(output.request_updates[1].output_token, None);
        assert_eq!(output.logits_handles, vec![RequestId(2)]);
        assert!(output.logits_outputs.is_empty());
        assert_eq!(output.kv_write_summary.tokens_written, 4);
        assert_eq!(output.kv_write_summary.blocks_touched, 2);
        assert_eq!(output.route_metadata, route_metadata);
    }

    #[test]
    fn deterministic_runner_counts_shared_blocks_once() {
        let runner = DeterministicRunner;
        let output = runner.run(RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(9),
                model_id: "qwen3".into(),
                execution_plan_ref: Some("dense.plan".into()),
                items: vec![
                    ExecutionItem {
                        request_id: RequestId(1),
                        mode: ExecutionMode::Prefill,
                        input_token_slice: vec![1],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 0,
                            end_exclusive: 1,
                        },
                        scheduled_token_count: 1,
                        block_table_ref: RequestId(1),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                    ExecutionItem {
                        request_id: RequestId(2),
                        mode: ExecutionMode::Prefill,
                        input_token_slice: vec![2],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 0,
                            end_exclusive: 1,
                        },
                        scheduled_token_count: 1,
                        block_table_ref: RequestId(2),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                ],
                total_scheduled_tokens: 2,
                route_metadata: RouteMetadata::empty(),
            },
            block_tables: vec![
                ResolvedBlockTable {
                    request_id: RequestId(1),
                    block_table: BlockTableView {
                        cache_group_id: crate::ids::CacheGroupId(1),
                        block_ids: vec![crate::ids::BlockId(0), crate::ids::BlockId(1)],
                    },
                },
                ResolvedBlockTable {
                    request_id: RequestId(2),
                    block_table: BlockTableView {
                        cache_group_id: crate::ids::CacheGroupId(1),
                        block_ids: vec![
                            crate::ids::BlockId(0),
                            crate::ids::BlockId(1),
                            crate::ids::BlockId(2),
                        ],
                    },
                },
            ],
            request_contexts: Vec::new(),
        });

        assert_eq!(output.kv_write_summary.blocks_touched, 3);
    }

    #[test]
    fn deterministic_runner_does_not_synthesize_stub_route_labels() {
        let runner = DeterministicRunner;
        let output = runner.run(RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(10),
                model_id: "qwen3".into(),
                execution_plan_ref: Some("dense.plan".into()),
                items: vec![ExecutionItem {
                    request_id: RequestId(1),
                    mode: ExecutionMode::Prefill,
                    input_token_slice: vec![1, 2],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 0,
                        end_exclusive: 2,
                    },
                    scheduled_token_count: 2,
                    block_table_ref: RequestId(1),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                }],
                total_scheduled_tokens: 2,
                route_metadata: RouteMetadata::empty(),
            },
            block_tables: vec![ResolvedBlockTable {
                request_id: RequestId(1),
                block_table: BlockTableView {
                    cache_group_id: crate::ids::CacheGroupId(1),
                    block_ids: vec![crate::ids::BlockId(0)],
                },
            }],
            request_contexts: vec![RunnerRequestContext {
                request_id: RequestId(1),
                prompt_len: 2,
                processed_prompt_tokens: 0,
                generated_len: 0,
                max_output_tokens: 2,
                deterministic_argmax_sampling: true,
                temperature: 0.0,
            }],
        });

        assert_eq!(output.route_metadata, RouteMetadata::empty());
    }
}
