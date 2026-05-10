use crate::ids::{RequestId, StepId};
use crate::request::{RequestSnapshot, RequestState};

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerInput {
    pub step_id: StepId,
    pub request_snapshots: Vec<RequestSnapshot>,
    pub memory_pressure: Option<String>,
    pub global_token_budget: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionMode {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionRange {
    pub start: u32,
    pub end_exclusive: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteMetadata {
    pub execution_plan: Option<String>,
    pub attention_route: Option<String>,
    pub kv_mode: Option<String>,
    pub prefix_cache_path: Option<String>,
    pub barrier_mode: Option<String>,
    pub crossover_decisions: Vec<(String, u32)>,
}

pub const ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS: &str = "ax_mlx_kv_request_snapshots";
pub const ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS: &str = "ax_mlx_kv_logical_tokens";
pub const ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS: &str = "ax_mlx_kv_capacity_tokens";
pub const ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB: &str = "ax_mlx_kv_logical_kib";
pub const ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB: &str = "ax_mlx_kv_capacity_kib";
pub const ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS: &str = "ax_mlx_kv_full_attention_layers";
pub const ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS: &str = "ax_mlx_kv_sliding_window_layers";
pub const ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS: &str =
    "ax_mlx_kv_sliding_retained_tokens";
pub const ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS: &str =
    "ax_mlx_kv_sliding_reclaimable_capacity_tokens";
pub const ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB: &str =
    "ax_mlx_kv_sliding_reclaimable_capacity_kib";
pub const ROUTE_KV_MODE_PAGED_METADATA: &str = "paged_metadata";
pub const ROUTE_BARRIER_MODE_SERIAL: &str = "serial";
pub const ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS: &str = "ax_mlx_kv_linear_state_layers";
pub const ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB: &str = "ax_mlx_kv_linear_state_kib";
pub const ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT: &str = "ax_mlx_kv_growth_count";
pub const ROUTE_DECISION_AX_MLX_MODEL_MLA_KV_LATENT_DIM: &str = "ax_mlx_model_mla_kv_latent_dim";
pub const ROUTE_DECISION_AX_MLX_MODEL_MOE_ACTIVE_EXPERTS: &str = "ax_mlx_model_moe_active_experts";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_REQUEST_SNAPSHOTS: &str =
    "ax_mlx_kv_compression_request_snapshots";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_STATUS: &str = "ax_mlx_kv_compression_status";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRESET: &str = "ax_mlx_kv_compression_preset";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEY_BITS: &str = "ax_mlx_kv_compression_key_bits";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_VALUE_BITS: &str =
    "ax_mlx_kv_compression_value_bits";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ELIGIBLE_LAYERS: &str =
    "ax_mlx_kv_compression_eligible_layers";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_CANDIDATE_TOKEN_LAYERS: &str =
    "ax_mlx_kv_compression_candidate_token_layers";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_HOT_TOKEN_LAYERS: &str =
    "ax_mlx_kv_compression_hot_token_layers";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FULL_PRECISION_KIB: &str =
    "ax_mlx_kv_compression_full_precision_kib";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_COMPRESSED_KIB: &str =
    "ax_mlx_kv_compression_estimated_compressed_kib";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_SAVED_KIB: &str =
    "ax_mlx_kv_compression_estimated_saved_kib";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RATIO_MILLI: &str =
    "ax_mlx_kv_compression_ratio_milli";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ROUTE_METADATA_SCHEMA: &str =
    "ax_mlx_kv_compression_route_metadata_schema";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_READY: &str =
    "ax_mlx_kv_compression_production_ready";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_BLOCKERS: &str =
    "ax_mlx_kv_compression_production_blockers";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_LAYERS: &str =
    "ax_mlx_kv_compression_runtime_storage_layers";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_TOKEN_LAYERS: &str =
    "ax_mlx_kv_compression_runtime_storage_token_layers";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_KIB: &str =
    "ax_mlx_kv_compression_runtime_storage_kib";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_WRITTEN_SLOTS: &str =
    "ax_mlx_kv_compression_runtime_storage_written_slots";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_CALLS: &str =
    "ax_mlx_kv_compression_shadow_sync_calls";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_WALL_US: &str =
    "ax_mlx_kv_compression_shadow_sync_wall_us";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH: &str =
    "ax_mlx_kv_compression_decode_path";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES: &str =
    "ax_mlx_kv_compression_fused_decode_candidates";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS: &str =
    "ax_mlx_kv_compression_fused_decode_attempts";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES: &str =
    "ax_mlx_kv_compression_fused_decode_successes";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS: &str =
    "ax_mlx_kv_compression_fused_decode_fallbacks";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON: &str =
    "ax_mlx_kv_compression_fused_decode_fallback_reason";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_READY_CANDIDATES: &str =
    "ax_mlx_kv_compression_fused_decode_ready_candidates";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_PREFILL_ONLY: &str =
    "ax_mlx_kv_compression_fused_decode_blocked_prefill_only";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_ATTENTION_KIND: &str =
    "ax_mlx_kv_compression_fused_decode_blocked_attention_kind";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_INELIGIBLE_LAYER: &str =
    "ax_mlx_kv_compression_fused_decode_blocked_ineligible_layer";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_PRESET: &str =
    "ax_mlx_kv_compression_fused_decode_blocked_unsupported_preset";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_HEAD_DIM: &str =
    "ax_mlx_kv_compression_fused_decode_blocked_unsupported_head_dim";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_GQA: &str =
    "ax_mlx_kv_compression_fused_decode_blocked_gqa";
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_MISSING_STORAGE: &str =
    "ax_mlx_kv_compression_fused_decode_blocked_missing_storage";
pub const ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_PREFILL_TOKENS: &str =
    "ax_scheduler_scheduled_prefill_tokens";
pub const ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS: &str =
    "ax_scheduler_scheduled_decode_tokens";
pub const ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS: &str =
    "ax_scheduler_skipped_prefill_tokens";
pub const ROUTE_DECISION_AX_SCHEDULER_SKIPPED_DECODE_TOKENS: &str =
    "ax_scheduler_skipped_decode_tokens";
pub const ROUTE_DECISION_AX_SCHEDULER_MIXED_PREFILL_DECODE_BATCHES: &str =
    "ax_scheduler_mixed_prefill_decode_batches";
pub const ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS: [&str; 5] = [
    ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_PREFILL_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_SKIPPED_DECODE_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_MIXED_PREFILL_DECODE_BATCHES,
];
pub const ROUTE_DECISION_AX_MLX_KV_KEYS: [&str; 13] = [
    ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB,
    ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB,
    ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB,
    ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT,
];
pub const ROUTE_DECISION_AX_MLX_MODEL_KEYS: [&str; 2] = [
    ROUTE_DECISION_AX_MLX_MODEL_MLA_KV_LATENT_DIM,
    ROUTE_DECISION_AX_MLX_MODEL_MOE_ACTIVE_EXPERTS,
];
pub const ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS: [&str; 35] = [
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_STATUS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRESET,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEY_BITS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_VALUE_BITS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ELIGIBLE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_CANDIDATE_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_HOT_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FULL_PRECISION_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_COMPRESSED_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_SAVED_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RATIO_MILLI,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ROUTE_METADATA_SCHEMA,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_READY,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_BLOCKERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_WRITTEN_SLOTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_CALLS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_WALL_US,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_READY_CANDIDATES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_PREFILL_ONLY,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_ATTENTION_KIND,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_INELIGIBLE_LAYER,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_PRESET,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_HEAD_DIM,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_GQA,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_MISSING_STORAGE,
];

impl RouteMetadata {
    pub fn empty() -> Self {
        Self {
            execution_plan: None,
            attention_route: None,
            kv_mode: None,
            prefix_cache_path: None,
            barrier_mode: None,
            crossover_decisions: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionItem {
    pub request_id: RequestId,
    pub mode: ExecutionMode,
    pub input_token_slice: Vec<u32>,
    pub reused_prefix_token_slice: Vec<u32>,
    pub position_range: PositionRange,
    pub scheduled_token_count: u32,
    pub block_table_ref: RequestId,
    pub prefix_tokens_reused: u32,
    pub prefix_blocks_reused: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionBatch {
    pub step_id: StepId,
    pub model_id: String,
    pub execution_plan_ref: Option<String>,
    pub items: Vec<ExecutionItem>,
    pub total_scheduled_tokens: u32,
    pub route_metadata: RouteMetadata,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulePlan {
    pub step_id: StepId,
    pub selected_requests: Vec<RequestId>,
    pub deferred_requests: Vec<RequestId>,
    pub memory_blocked_requests: Vec<RequestId>,
    pub execution_batch: Option<ExecutionBatch>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BatchRouteSeed {
    mode: ExecutionMode,
    execution_plan_ref: Option<String>,
    route_metadata: RouteMetadata,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct TokenBudgetTelemetry {
    scheduled_prefill_tokens: u32,
    scheduled_decode_tokens: u32,
    skipped_prefill_tokens: u32,
    skipped_decode_tokens: u32,
    mixed_prefill_decode_batches: u32,
}

impl TokenBudgetTelemetry {
    fn record_scheduled(&mut self, mode: ExecutionMode, tokens: u32) {
        match mode {
            ExecutionMode::Prefill => {
                self.scheduled_prefill_tokens =
                    self.scheduled_prefill_tokens.saturating_add(tokens);
            }
            ExecutionMode::Decode => {
                self.scheduled_decode_tokens = self.scheduled_decode_tokens.saturating_add(tokens);
            }
        }
    }

    fn record_skipped(&mut self, mode: ExecutionMode, tokens: u32) {
        match mode {
            ExecutionMode::Prefill => {
                self.skipped_prefill_tokens = self.skipped_prefill_tokens.saturating_add(tokens);
            }
            ExecutionMode::Decode => {
                self.skipped_decode_tokens = self.skipped_decode_tokens.saturating_add(tokens);
            }
        }
    }

    fn append_route_decisions(&self, route_metadata: &mut RouteMetadata) {
        route_metadata.crossover_decisions.extend([
            (
                ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_PREFILL_TOKENS.into(),
                self.scheduled_prefill_tokens,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS.into(),
                self.scheduled_decode_tokens,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS.into(),
                self.skipped_prefill_tokens,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_SKIPPED_DECODE_TOKENS.into(),
                self.skipped_decode_tokens,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_MIXED_PREFILL_DECODE_BATCHES.into(),
                self.mixed_prefill_decode_batches,
            ),
        ]);
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scheduler;

const MEMORY_PRESSURE_MAX_PREFILL_TOKENS_PER_STEP: u32 = 1;
const MEMORY_PRESSURE_KV_EXHAUSTED: &str = "kv_exhausted";
const MEMORY_PRESSURE_KV_EXHAUSTED_RECLAIMABLE_CACHE: &str = "kv_exhausted_reclaimable_cache";

impl Scheduler {
    pub fn new() -> Self {
        Self
    }

    pub fn plan(&self, input: &SchedulerInput) -> SchedulePlan {
        let mut runnable = input
            .request_snapshots
            .iter()
            .filter(|snapshot| snapshot.state == RequestState::Runnable)
            .cloned()
            .collect::<Vec<_>>();
        runnable.sort_by_key(|snapshot| (snapshot.arrival_sequence, snapshot.request_id));

        if runnable.is_empty() {
            return SchedulePlan {
                step_id: input.step_id,
                selected_requests: Vec::new(),
                deferred_requests: Vec::new(),
                memory_blocked_requests: Vec::new(),
                execution_batch: None,
            };
        }

        if input.global_token_budget == 0 {
            return SchedulePlan {
                step_id: input.step_id,
                selected_requests: Vec::new(),
                deferred_requests: runnable
                    .into_iter()
                    .map(|snapshot| snapshot.request_id)
                    .collect(),
                memory_blocked_requests: Vec::new(),
                execution_batch: None,
            };
        }

        let batch_model_id = runnable[0].model_id.clone();
        let mut remaining_budget = input.global_token_budget;
        let mut selected_requests = Vec::new();
        let mut deferred_requests = Vec::new();
        let mut items = Vec::new();
        let mut candidates = Vec::new();
        let mut route_seed_anchor: Option<BatchRouteSeed> = None;
        let mut batch_mixes_prefill_decode = false;
        let mut token_budget = TokenBudgetTelemetry::default();
        let mut pressure_prefill_budget =
            prefill_budget_for_memory_pressure(input.memory_pressure.as_deref());

        for snapshot in runnable {
            if snapshot.model_id != batch_model_id {
                deferred_requests.push(snapshot.request_id);
                continue;
            }

            let Some(mode) = self.request_mode(&snapshot) else {
                deferred_requests.push(snapshot.request_id);
                continue;
            };
            candidates.push((snapshot, mode));
        }

        candidates.sort_by_key(|(snapshot, mode)| {
            (
                execution_mode_priority(*mode),
                snapshot.arrival_sequence,
                snapshot.request_id,
            )
        });

        for (snapshot, mode) in candidates {
            let requested_tokens = schedulable_token_count(&snapshot, mode);
            if remaining_budget == 0 {
                token_budget.record_skipped(mode, requested_tokens);
                deferred_requests.push(snapshot.request_id);
                continue;
            }

            let candidate_budget = match (mode, pressure_prefill_budget) {
                (ExecutionMode::Prefill, Some(0)) => {
                    token_budget.record_skipped(mode, requested_tokens);
                    deferred_requests.push(snapshot.request_id);
                    continue;
                }
                (ExecutionMode::Prefill, Some(prefill_budget)) => {
                    remaining_budget.min(prefill_budget)
                }
                _ => remaining_budget,
            };

            let Some(item) = self.build_execution_item(&snapshot, candidate_budget) else {
                deferred_requests.push(snapshot.request_id);
                continue;
            };

            let candidate_route = BatchRouteSeed {
                mode,
                execution_plan_ref: snapshot.execution_plan_ref.clone(),
                route_metadata: route_seed(&snapshot),
            };
            let can_join = route_seed_anchor
                .as_ref()
                .is_none_or(|anchor| route_seed_can_join_batch(anchor, &candidate_route));
            if !can_join {
                deferred_requests.push(snapshot.request_id);
                continue;
            }

            if let Some(anchor) = &route_seed_anchor {
                if anchor.mode != mode {
                    batch_mixes_prefill_decode = true;
                }
            } else {
                route_seed_anchor = Some(candidate_route);
            }

            remaining_budget -= item.scheduled_token_count;
            if let (ExecutionMode::Prefill, Some(prefill_budget)) =
                (item.mode, pressure_prefill_budget)
            {
                pressure_prefill_budget =
                    Some(prefill_budget.saturating_sub(item.scheduled_token_count));
            }
            token_budget.record_scheduled(item.mode, item.scheduled_token_count);
            if requested_tokens > item.scheduled_token_count {
                token_budget
                    .record_skipped(item.mode, requested_tokens - item.scheduled_token_count);
            }

            selected_requests.push(snapshot.request_id);
            items.push(item);
        }

        let execution_batch = if items.is_empty() {
            None
        } else {
            let total_scheduled_tokens = items.iter().map(|item| item.scheduled_token_count).sum();
            if batch_mixes_prefill_decode {
                token_budget.mixed_prefill_decode_batches = 1;
            }
            let mut route_metadata =
                route_metadata_for_batch(route_seed_anchor.as_ref(), batch_mixes_prefill_decode);
            token_budget.append_route_decisions(&mut route_metadata);
            Some(ExecutionBatch {
                step_id: input.step_id,
                model_id: batch_model_id.0.clone(),
                execution_plan_ref: route_metadata.execution_plan.clone(),
                total_scheduled_tokens,
                items,
                route_metadata,
            })
        };

        SchedulePlan {
            step_id: input.step_id,
            selected_requests,
            deferred_requests,
            memory_blocked_requests: Vec::new(),
            execution_batch,
        }
    }

    fn build_execution_item(
        &self,
        snapshot: &RequestSnapshot,
        remaining_budget: u32,
    ) -> Option<ExecutionItem> {
        if remaining_budget == 0 {
            return None;
        }

        if snapshot.processed_prompt_tokens < snapshot.prompt_len {
            let scheduled_token_count =
                remaining_budget.min(snapshot.prompt_len - snapshot.processed_prompt_tokens);
            let start = snapshot.processed_prompt_tokens as usize;
            let end = start + scheduled_token_count as usize;

            return Some(ExecutionItem {
                request_id: snapshot.request_id,
                mode: ExecutionMode::Prefill,
                input_token_slice: snapshot.prompt_tokens[start..end].to_vec(),
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: snapshot.processed_prompt_tokens,
                    end_exclusive: snapshot.processed_prompt_tokens + scheduled_token_count,
                },
                scheduled_token_count,
                block_table_ref: snapshot.request_id,
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            });
        }

        if snapshot.generated_len >= snapshot.max_output_tokens {
            return None;
        }

        let decode_token = snapshot
            .generated_tokens
            .last()
            .copied()
            .or_else(|| snapshot.prompt_tokens.last().copied())?;
        let position_start = snapshot.prompt_len.saturating_add(snapshot.generated_len);

        Some(ExecutionItem {
            request_id: snapshot.request_id,
            mode: ExecutionMode::Decode,
            input_token_slice: vec![decode_token],
            reused_prefix_token_slice: Vec::new(),
            position_range: PositionRange {
                start: position_start,
                end_exclusive: position_start + 1,
            },
            scheduled_token_count: 1,
            block_table_ref: snapshot.request_id,
            prefix_tokens_reused: 0,
            prefix_blocks_reused: 0,
        })
    }

    fn request_mode(&self, snapshot: &RequestSnapshot) -> Option<ExecutionMode> {
        if snapshot.processed_prompt_tokens < snapshot.prompt_len {
            return Some(ExecutionMode::Prefill);
        }

        if snapshot.generated_len >= snapshot.max_output_tokens {
            return None;
        }

        snapshot
            .generated_tokens
            .last()
            .copied()
            .or_else(|| snapshot.prompt_tokens.last().copied())
            .map(|_| ExecutionMode::Decode)
    }
}

fn execution_mode_priority(mode: ExecutionMode) -> u8 {
    match mode {
        ExecutionMode::Decode => 0,
        ExecutionMode::Prefill => 1,
    }
}

fn schedulable_token_count(snapshot: &RequestSnapshot, mode: ExecutionMode) -> u32 {
    match mode {
        ExecutionMode::Prefill => snapshot
            .prompt_len
            .saturating_sub(snapshot.processed_prompt_tokens),
        ExecutionMode::Decode => 1,
    }
}

fn prefill_budget_for_memory_pressure(memory_pressure: Option<&str>) -> Option<u32> {
    match memory_pressure {
        None => None,
        Some(MEMORY_PRESSURE_KV_EXHAUSTED) => Some(0),
        Some(MEMORY_PRESSURE_KV_EXHAUSTED_RECLAIMABLE_CACHE) => {
            Some(MEMORY_PRESSURE_MAX_PREFILL_TOKENS_PER_STEP)
        }
        Some(_) => Some(MEMORY_PRESSURE_MAX_PREFILL_TOKENS_PER_STEP),
    }
}

fn route_seed_can_join_batch(anchor: &BatchRouteSeed, candidate: &BatchRouteSeed) -> bool {
    if anchor.mode == candidate.mode {
        return anchor.execution_plan_ref == candidate.execution_plan_ref
            && anchor.route_metadata == candidate.route_metadata;
    }

    mixed_route_metadata_compatible(&anchor.route_metadata, &candidate.route_metadata)
}

fn mixed_route_metadata_compatible(anchor: &RouteMetadata, candidate: &RouteMetadata) -> bool {
    if *anchor == RouteMetadata::empty() && *candidate == RouteMetadata::empty() {
        return true;
    }

    anchor.kv_mode == candidate.kv_mode
        && anchor.barrier_mode == candidate.barrier_mode
        && anchor.prefix_cache_path == candidate.prefix_cache_path
}

fn route_metadata_for_batch(
    anchor: Option<&BatchRouteSeed>,
    mixed_prefill_decode: bool,
) -> RouteMetadata {
    let Some(anchor) = anchor else {
        return RouteMetadata::empty();
    };
    let mut route_metadata = anchor.route_metadata.clone();
    if mixed_prefill_decode {
        route_metadata.execution_plan = Some("phase2.token_budget".into());
        route_metadata.attention_route = Some("mixed_prefill_decode".into());
    }
    route_metadata
}

fn route_seed(snapshot: &RequestSnapshot) -> RouteMetadata {
    if snapshot.route_metadata_hint != RouteMetadata::empty() {
        return snapshot.route_metadata_hint.clone();
    }

    RouteMetadata {
        execution_plan: snapshot.execution_plan_ref.clone(),
        attention_route: None,
        kv_mode: None,
        prefix_cache_path: None,
        barrier_mode: None,
        crossover_decisions: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{ModelId, SequenceNo};
    use std::collections::BTreeSet;

    fn make_snapshot(
        request_id: u64,
        arrival_sequence: u64,
        model_id: &str,
        prompt_tokens: &[u32],
        processed_prompt_tokens: u32,
        generated_tokens: &[u32],
        max_output_tokens: u32,
    ) -> RequestSnapshot {
        RequestSnapshot {
            request_id: RequestId(request_id),
            model_id: ModelId(model_id.into()),
            arrival_sequence: SequenceNo(arrival_sequence),
            state: RequestState::Runnable,
            prompt_tokens: prompt_tokens.to_vec(),
            processed_prompt_tokens,
            generated_tokens: generated_tokens.to_vec(),
            generated_token_logprobs: vec![Some(0.0); generated_tokens.len()],
            prompt_len: prompt_tokens.len() as u32,
            generated_len: generated_tokens.len() as u32,
            max_output_tokens,
            cancel_requested: false,
            execution_plan_ref: None,
            route_metadata_hint: RouteMetadata::empty(),
            terminal_stop_reason: None,
            last_error: None,
        }
    }

    #[test]
    fn mlx_kv_route_decision_keys_are_stable_and_unique() {
        assert_eq!(
            ROUTE_DECISION_AX_MLX_KV_KEYS.len(),
            ROUTE_DECISION_AX_MLX_KV_KEYS
                .into_iter()
                .collect::<BTreeSet<_>>()
                .len()
        );
        assert!(
            ROUTE_DECISION_AX_MLX_KV_KEYS
                .iter()
                .all(|key| key.starts_with("ax_mlx_kv_"))
        );
        assert!(
            ROUTE_DECISION_AX_MLX_KV_KEYS
                .iter()
                .all(|key| key.is_ascii())
        );
        assert!(
            ROUTE_DECISION_AX_MLX_KV_KEYS
                .iter()
                .all(|key| !key.contains('-'))
        );
    }

    #[test]
    fn mlx_model_route_decision_keys_are_stable_and_unique() {
        assert_eq!(
            ROUTE_DECISION_AX_MLX_MODEL_KEYS.len(),
            ROUTE_DECISION_AX_MLX_MODEL_KEYS
                .into_iter()
                .collect::<BTreeSet<_>>()
                .len()
        );
        assert!(
            ROUTE_DECISION_AX_MLX_MODEL_KEYS
                .iter()
                .all(|key| key.starts_with("ax_mlx_model_"))
        );
        assert!(
            ROUTE_DECISION_AX_MLX_MODEL_KEYS
                .iter()
                .all(|key| key.is_ascii())
        );
        assert!(
            ROUTE_DECISION_AX_MLX_MODEL_KEYS
                .iter()
                .all(|key| !key.contains('-'))
        );
    }

    #[test]
    fn mlx_kv_compression_route_decision_keys_are_stable_and_unique() {
        assert_eq!(
            ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS.len(),
            ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS
                .into_iter()
                .collect::<BTreeSet<_>>()
                .len()
        );
        assert!(
            ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS
                .iter()
                .all(|key| key.starts_with("ax_mlx_kv_compression_"))
        );
        assert!(
            ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS
                .iter()
                .all(|key| key.is_ascii())
        );
        assert!(
            ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS
                .iter()
                .all(|key| !key.contains('-'))
        );
    }

    #[test]
    fn scheduler_token_budget_route_decision_keys_are_stable_and_unique() {
        assert_eq!(
            ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS.len(),
            ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS
                .into_iter()
                .collect::<BTreeSet<_>>()
                .len()
        );
        assert!(
            ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS
                .iter()
                .all(|key| key.starts_with("ax_scheduler_"))
        );
        assert!(
            ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS
                .iter()
                .all(|key| key.is_ascii())
        );
        assert!(
            ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS
                .iter()
                .all(|key| !key.contains('-'))
        );
    }

    #[test]
    fn batches_oldest_model_family_first_with_chunked_prefill() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(4),
            request_snapshots: vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22], 0, &[], 16),
                make_snapshot(3, 3, "llama", &[30, 31], 0, &[], 16),
            ],
            memory_pressure: None,
            global_token_budget: 5,
        });

        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(1), RequestId(2)]
        );
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(3)]);

        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.model_id, "qwen3");
        assert_eq!(execution_batch.total_scheduled_tokens, 5);
        assert_eq!(execution_batch.items.len(), 2);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 3);
        assert_eq!(execution_batch.items[1].scheduled_token_count, 2);
        assert_eq!(execution_batch.items[1].input_token_slice, vec![20, 21]);
    }

    #[test]
    fn builds_decode_item_from_last_known_token() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(8),
            request_snapshots: vec![make_snapshot(9, 1, "qwen3", &[1, 2, 3], 3, &[7], 16)],
            memory_pressure: None,
            global_token_budget: 1,
        });

        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.items.len(), 1);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Decode);
        assert_eq!(execution_batch.items[0].input_token_slice, vec![7]);
        assert_eq!(
            execution_batch.items[0].position_range,
            PositionRange {
                start: 4,
                end_exclusive: 5,
            }
        );
    }

    #[test]
    fn builds_decode_item_from_last_prompt_token_without_generated_history() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(9),
            request_snapshots: vec![make_snapshot(10, 1, "qwen3", &[1, 2, 3], 3, &[], 16)],
            memory_pressure: None,
            global_token_budget: 1,
        });

        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.items.len(), 1);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Decode);
        assert_eq!(execution_batch.items[0].input_token_slice, vec![3]);
        assert_eq!(
            execution_batch.items[0].position_range,
            PositionRange {
                start: 3,
                end_exclusive: 4,
            }
        );
    }

    #[test]
    fn max_output_runnable_request_remains_visible_as_deferred() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(10),
            request_snapshots: vec![make_snapshot(12, 1, "qwen3", &[1, 2, 3], 3, &[7, 8], 2)],
            memory_pressure: None,
            global_token_budget: 1,
        });

        assert!(schedule_plan.selected_requests.is_empty());
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(12)]);
        assert!(schedule_plan.execution_batch.is_none());
    }

    #[test]
    fn mixes_decode_with_bounded_prefill_when_routes_are_compatible() {
        let scheduler = Scheduler::new();
        let mut prefill = make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16);
        prefill.execution_plan_ref = Some("phase1.qwen3.dense_prefill".into());
        prefill.route_metadata_hint = RouteMetadata {
            execution_plan: Some("phase1.qwen3.dense_prefill".into()),
            attention_route: Some("qwen3_prefill".into()),
            kv_mode: Some("paged_metadata".into()),
            prefix_cache_path: None,
            barrier_mode: Some("serial".into()),
            crossover_decisions: Vec::new(),
        };
        let mut decode = make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16);
        decode.execution_plan_ref = Some("phase1.qwen3.paged_decode".into());
        decode.route_metadata_hint = RouteMetadata {
            execution_plan: Some("phase1.qwen3.paged_decode".into()),
            attention_route: Some("qwen3_paged_decode".into()),
            kv_mode: Some("paged_metadata".into()),
            prefix_cache_path: None,
            barrier_mode: Some("serial".into()),
            crossover_decisions: Vec::new(),
        };

        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(11),
            request_snapshots: vec![prefill, decode],
            memory_pressure: None,
            global_token_budget: 3,
        });

        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(2), RequestId(1)]
        );
        assert!(schedule_plan.deferred_requests.is_empty());
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(
            execution_batch.execution_plan_ref.as_deref(),
            Some("phase2.token_budget")
        );
        assert_eq!(
            execution_batch.route_metadata.attention_route.as_deref(),
            Some("mixed_prefill_decode")
        );
        assert_eq!(execution_batch.total_scheduled_tokens, 3);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Decode);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 1);
        assert_eq!(execution_batch.items[1].mode, ExecutionMode::Prefill);
        assert_eq!(execution_batch.items[1].input_token_slice, vec![10, 11]);

        let decisions = execution_batch
            .route_metadata
            .crossover_decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_PREFILL_TOKENS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_MIXED_PREFILL_DECODE_BATCHES),
            Some(&1)
        );
    }

    #[test]
    fn defers_prefill_when_decode_exhausts_token_budget() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(14),
            request_snapshots: vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            memory_pressure: None,
            global_token_budget: 1,
        });

        assert_eq!(schedule_plan.selected_requests, vec![RequestId(2)]);
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
        let decisions = schedule_plan
            .execution_batch
            .unwrap()
            .route_metadata
            .crossover_decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS),
            Some(&4)
        );
    }

    #[test]
    fn memory_pressure_caps_prefill_tokens_without_starving_decode() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(15),
            request_snapshots: vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            memory_pressure: Some("kv_low_free_blocks:1/8".into()),
            global_token_budget: 8,
        });

        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(2), RequestId(1)]
        );
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.total_scheduled_tokens, 2);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Decode);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 1);
        assert_eq!(execution_batch.items[1].mode, ExecutionMode::Prefill);
        assert_eq!(execution_batch.items[1].scheduled_token_count, 1);

        let decisions = execution_batch
            .route_metadata
            .crossover_decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_PREFILL_TOKENS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS),
            Some(&3)
        );
    }

    #[test]
    fn exhausted_memory_pressure_defers_prefill_without_starving_decode() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(16),
            request_snapshots: vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            memory_pressure: Some("kv_exhausted".into()),
            global_token_budget: 8,
        });

        assert_eq!(schedule_plan.selected_requests, vec![RequestId(2)]);
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.total_scheduled_tokens, 1);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Decode);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 1);

        let decisions = execution_batch
            .route_metadata
            .crossover_decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_PREFILL_TOKENS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS),
            Some(&4)
        );
    }

    #[test]
    fn exhausted_reclaimable_cache_pressure_caps_prefill_without_starving_decode() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(17),
            request_snapshots: vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            memory_pressure: Some("kv_exhausted_reclaimable_cache".into()),
            global_token_budget: 8,
        });

        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(2), RequestId(1)]
        );
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.total_scheduled_tokens, 2);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Decode);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 1);
        assert_eq!(execution_batch.items[1].mode, ExecutionMode::Prefill);
        assert_eq!(execution_batch.items[1].scheduled_token_count, 1);
    }

    #[test]
    fn does_not_mix_different_execution_plan_refs_for_same_mode() {
        let scheduler = Scheduler::new();
        let mut first = make_snapshot(1, 1, "qwen3", &[10, 11, 12], 0, &[], 16);
        first.execution_plan_ref = Some("phase1.qwen3.dense_prefill".into());
        first.route_metadata_hint = RouteMetadata {
            execution_plan: Some("phase1.qwen3.dense_prefill".into()),
            attention_route: Some("qwen3_prefill".into()),
            kv_mode: Some("paged_metadata".into()),
            prefix_cache_path: None,
            barrier_mode: Some("serial".into()),
            crossover_decisions: Vec::new(),
        };
        let mut second = make_snapshot(2, 2, "qwen3", &[20, 21, 22], 0, &[], 16);
        second.execution_plan_ref = Some("phase1.qwen3.special_prefill".into());
        second.route_metadata_hint = RouteMetadata {
            execution_plan: Some("phase1.qwen3.special_prefill".into()),
            attention_route: Some("qwen3_prefill_alt".into()),
            kv_mode: Some("paged_metadata".into()),
            prefix_cache_path: None,
            barrier_mode: Some("serial".into()),
            crossover_decisions: Vec::new(),
        };

        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(12),
            request_snapshots: vec![first, second],
            memory_pressure: None,
            global_token_budget: 8,
        });

        assert_eq!(schedule_plan.selected_requests, vec![RequestId(1)]);
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(2)]);
        assert_eq!(
            schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.execution_plan_ref.as_deref()),
            Some("phase1.qwen3.dense_prefill")
        );
        assert_eq!(
            schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.route_metadata.attention_route.as_deref()),
            Some("qwen3_prefill")
        );
    }

    #[test]
    fn does_not_synthesize_generic_route_labels_without_binding_hint() {
        let scheduler = Scheduler::new();
        let mut snapshot = make_snapshot(1, 1, "qwen3", &[10, 11, 12], 0, &[], 16);
        snapshot.execution_plan_ref = Some("phase1.qwen3.dense_prefill".into());

        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(13),
            request_snapshots: vec![snapshot],
            memory_pressure: None,
            global_token_budget: 8,
        });

        let execution_batch = schedule_plan
            .execution_batch
            .expect("scheduler should still build the batch");
        assert_eq!(
            execution_batch.route_metadata.execution_plan.as_deref(),
            Some("phase1.qwen3.dense_prefill")
        );
        assert_eq!(execution_batch.route_metadata.attention_route, None);
        assert_eq!(execution_batch.route_metadata.kv_mode, None);
        assert_eq!(execution_batch.route_metadata.barrier_mode, None);
    }

    #[test]
    fn defers_all_requests_when_global_token_budget_is_zero() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(20),
            request_snapshots: vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21], 0, &[], 16),
            ],
            memory_pressure: None,
            global_token_budget: 0,
        });

        assert!(schedule_plan.selected_requests.is_empty());
        assert_eq!(
            schedule_plan.deferred_requests,
            vec![RequestId(1), RequestId(2)]
        );
        assert!(schedule_plan.execution_batch.is_none());
    }

    #[test]
    fn defers_requests_with_no_remaining_work_so_they_stay_visible() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(21),
            request_snapshots: vec![make_snapshot(1, 1, "qwen3", &[10, 11, 12], 3, &[7, 8], 2)],
            memory_pressure: None,
            global_token_budget: 8,
        });

        assert!(schedule_plan.selected_requests.is_empty());
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
        assert!(schedule_plan.execution_batch.is_none());
    }

    #[test]
    fn defers_different_model_requests_to_next_step() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(22),
            request_snapshots: vec![
                make_snapshot(1, 1, "qwen3", &[10, 11], 0, &[], 16),
                make_snapshot(2, 2, "gemma", &[20, 21], 0, &[], 16),
                make_snapshot(3, 3, "qwen3", &[30, 31], 0, &[], 16),
            ],
            memory_pressure: None,
            global_token_budget: 16,
        });

        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(1), RequestId(3)]
        );
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(2)]);

        let batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(batch.model_id, "qwen3");
        assert_eq!(batch.items.len(), 2);
    }

    #[test]
    fn returns_empty_plan_when_no_runnable_requests() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput {
            step_id: StepId(23),
            request_snapshots: vec![],
            memory_pressure: None,
            global_token_budget: 16,
        });

        assert!(schedule_plan.selected_requests.is_empty());
        assert!(schedule_plan.deferred_requests.is_empty());
        assert!(schedule_plan.execution_batch.is_none());
    }
}
