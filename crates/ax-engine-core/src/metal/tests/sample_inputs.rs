use crate::ids::{BlockId, CacheGroupId, RequestId, StepId};
use crate::kv::BlockTableView;
use crate::runner::{ResolvedBlockTable, RunnerInput, RunnerRequestContext};
use crate::scheduler::{
    ExecutionBatch, ExecutionItem, ExecutionMode, PositionRange, RouteMetadata,
};

pub(super) fn sample_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(3),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.dense_prefill".into()),
            items: vec![
                ExecutionItem {
                    request_id: RequestId(7),
                    mode: ExecutionMode::Prefill,
                    input_token_slice: vec![1, 2, 3],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 0,
                        end_exclusive: 3,
                    },
                    scheduled_token_count: 3,
                    block_table_ref: RequestId(7),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
                ExecutionItem {
                    request_id: RequestId(9),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![4],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 3,
                        end_exclusive: 4,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(9),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
            ],
            total_scheduled_tokens: 4,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.dense_prefill".into()),
                attention_route: Some("qwen3_dense_prefill".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![
            ResolvedBlockTable {
                request_id: RequestId(7),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(0), BlockId(1)],
                },
            },
            ResolvedBlockTable {
                request_id: RequestId(9),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(2)],
                },
            },
        ],
        request_contexts: vec![
            RunnerRequestContext {
                request_id: RequestId(7),
                prompt_len: 3,
                processed_prompt_tokens: 0,
                generated_len: 0,
                max_output_tokens: 32,
                deterministic_argmax_sampling: false,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                repetition_context_size: None,
            },
            RunnerRequestContext {
                request_id: RequestId(9),
                prompt_len: 3,
                processed_prompt_tokens: 3,
                generated_len: 0,
                max_output_tokens: 32,
                deterministic_argmax_sampling: false,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                repetition_context_size: None,
            },
        ],
    }
}

pub(super) fn sample_decode_only_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(4),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.decode_only".into()),
            items: vec![
                ExecutionItem {
                    request_id: RequestId(9),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![4],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 3,
                        end_exclusive: 4,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(9),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
                ExecutionItem {
                    request_id: RequestId(11),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![8],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 5,
                        end_exclusive: 6,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(11),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
            ],
            total_scheduled_tokens: 2,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.decode_only".into()),
                attention_route: Some("qwen3_dense_decode".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![
            ResolvedBlockTable {
                request_id: RequestId(9),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(2)],
                },
            },
            ResolvedBlockTable {
                request_id: RequestId(11),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(3)],
                },
            },
        ],
        request_contexts: vec![
            RunnerRequestContext {
                request_id: RequestId(9),
                prompt_len: 3,
                processed_prompt_tokens: 3,
                generated_len: 0,
                max_output_tokens: 32,
                deterministic_argmax_sampling: false,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                repetition_context_size: None,
            },
            RunnerRequestContext {
                request_id: RequestId(11),
                prompt_len: 5,
                processed_prompt_tokens: 5,
                generated_len: 0,
                max_output_tokens: 32,
                deterministic_argmax_sampling: false,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                repetition_context_size: None,
            },
        ],
    }
}

pub(super) fn sample_prefill_only_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(5),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.prefill_only".into()),
            items: vec![ExecutionItem {
                request_id: RequestId(17),
                mode: ExecutionMode::Prefill,
                input_token_slice: vec![1, 2, 3, 4],
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: 0,
                    end_exclusive: 4,
                },
                scheduled_token_count: 4,
                block_table_ref: RequestId(17),
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
            total_scheduled_tokens: 4,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.prefill_only".into()),
                attention_route: Some("qwen3_dense_prefill".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![ResolvedBlockTable {
            request_id: RequestId(17),
            block_table: BlockTableView {
                cache_group_id: CacheGroupId(1),
                block_ids: vec![BlockId(0)],
            },
        }],
        request_contexts: vec![RunnerRequestContext {
            request_id: RequestId(17),
            prompt_len: 4,
            processed_prompt_tokens: 0,
            generated_len: 0,
            max_output_tokens: 32,
            deterministic_argmax_sampling: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context_size: None,
        }],
    }
}

pub(super) fn sample_decode_continuation_runner_input() -> RunnerInput {
    RunnerInput {
        block_size_tokens: 16,
        execution_batch: ExecutionBatch {
            step_id: StepId(6),
            model_id: "qwen3_dense".into(),
            execution_plan_ref: Some("phase1.qwen3_dense.decode_continuation".into()),
            items: vec![ExecutionItem {
                request_id: RequestId(17),
                mode: ExecutionMode::Decode,
                input_token_slice: vec![4],
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: 4,
                    end_exclusive: 5,
                },
                scheduled_token_count: 1,
                block_table_ref: RequestId(17),
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
            total_scheduled_tokens: 1,
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3_dense.decode_continuation".into()),
                attention_route: Some("qwen3_dense_decode".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: Some("metadata_lookup".into()),
                barrier_mode: Some("serial".into()),
                crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
            },
        },
        block_tables: vec![ResolvedBlockTable {
            request_id: RequestId(17),
            block_table: BlockTableView {
                cache_group_id: CacheGroupId(1),
                block_ids: vec![BlockId(0)],
            },
        }],
        request_contexts: vec![RunnerRequestContext {
            request_id: RequestId(17),
            prompt_len: 4,
            processed_prompt_tokens: 4,
            generated_len: 0,
            max_output_tokens: 32,
            deterministic_argmax_sampling: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context_size: None,
        }],
    }
}
