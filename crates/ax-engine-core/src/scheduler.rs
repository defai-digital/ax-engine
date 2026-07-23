use crate::generation::{
    GenerationKind, GenerationProgress, GenerationStrategyDescriptor, WorkUnitKind,
};
use crate::ids::{RequestId, StepId};
use crate::request::{RequestSnapshot, RequestState};

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerInput {
    pub step_id: StepId,
    pub request_snapshots: Vec<RequestSnapshot>,
    pub memory_pressure: Option<String>,
    pub global_token_budget: u32,
    /// When true and KV pressure is absent, cap each *text* prefill item and
    /// admit at most `max_inflight_prefill_requests` prefills so multiple
    /// prompts make progress. Multimodal prefills stay atomic (full-or-defer).
    /// Fair mode **raises** concurrent live KV footprint; headroom admission
    /// uses `available_kv_blocks` / `block_size_tokens`. Default: false.
    pub multi_prefill_fair: bool,
    /// Per-request prefill token cap when fair mode is active. `0` means
    /// "use `block_size_tokens` as the fair chunk floor".
    pub max_prefill_tokens_per_request_per_step: u32,
    /// Max concurrent prefill requests admitted under fair mode. `0` means
    /// unlimited (still subject to residual budget and headroom).
    pub max_inflight_prefill_requests: u32,
    /// Tokens per logical KV block (from `KvManagerConfig`).
    pub block_size_tokens: u32,
    /// Free physical/logical blocks available for new allocation this step.
    pub available_kv_blocks: u32,
    /// Total blocks in the pool (telemetry / ratio only).
    pub total_kv_blocks: u32,
}

impl SchedulerInput {
    /// Legacy-compatible constructor: fair multi-prefill off, unlimited KV headroom.
    pub fn new(
        step_id: StepId,
        request_snapshots: Vec<RequestSnapshot>,
        memory_pressure: Option<String>,
        global_token_budget: u32,
    ) -> Self {
        Self {
            step_id,
            request_snapshots,
            memory_pressure,
            global_token_budget,
            multi_prefill_fair: false,
            max_prefill_tokens_per_request_per_step: 0,
            max_inflight_prefill_requests: 0,
            block_size_tokens: 16,
            available_kv_blocks: u32::MAX,
            total_kv_blocks: u32::MAX,
        }
    }
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

pub fn upsert_route_decision(decisions: &mut Vec<(String, u32)>, key: &str, value: u32) {
    let mut updated = false;
    decisions.retain_mut(|(existing_key, existing_value)| {
        if existing_key == key {
            if updated {
                false
            } else {
                *existing_value = value;
                updated = true;
                true
            }
        } else {
            true
        }
    });

    if !updated {
        decisions.push((key.to_string(), value));
    }
}

/// Telemetry key for [`crate::GenerationKind::telemetry_code`].
pub const ROUTE_DECISION_AX_MLX_GENERATION_KIND: &str = "ax_mlx_generation_kind";
/// Telemetry key for the planned [`crate::WorkUnitKind::telemetry_code`] of the step.
pub const ROUTE_DECISION_AX_MLX_GENERATION_WORK_UNIT: &str = "ax_mlx_generation_work_unit";
/// Telemetry key for [`crate::LayerForwardRoute::telemetry_code`].
pub const ROUTE_DECISION_AX_MLX_LAYER_FORWARD_ROUTE: &str = "ax_mlx_layer_forward_route";
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
pub const ROUTE_DECISION_AX_MLX_KV_ROTATED_RING_LAYERS: &str = "ax_mlx_kv_rotated_ring_layers";
pub const ROUTE_DECISION_AX_MLX_KV_ROTATING_RING_SLACK: &str = "ax_mlx_kv_rotating_ring_slack";
pub const ROUTE_KV_MODE_PAGED_METADATA: &str = "paged_metadata";
pub const ROUTE_BARRIER_MODE_SERIAL: &str = "serial";
pub const ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS: &str = "ax_mlx_kv_linear_state_layers";
pub const ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB: &str = "ax_mlx_kv_linear_state_kib";
pub const ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT: &str = "ax_mlx_kv_growth_count";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_MATERIALIZE_US: &str = "ax_mlx_kv_paged_materialize_us";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_EXHAUSTION_FALLBACKS: &str =
    "ax_mlx_kv_paged_pool_exhaustion_fallbacks";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_COW_COPIES: &str = "ax_mlx_kv_paged_cow_copies";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_BLOCKS_USED: &str =
    "ax_mlx_kv_paged_pool_blocks_used";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SHARED_BLOCKS: &str =
    "ax_mlx_kv_paged_pool_shared_blocks";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLABS: &str = "ax_mlx_kv_paged_pool_slabs";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_KIB: &str = "ax_mlx_kv_paged_pool_slab_kib";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_GROW_EVENTS: &str =
    "ax_mlx_kv_paged_pool_slab_grow_events";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_CALLS: &str = "ax_mlx_kv_paged_attention_calls";
pub const ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_FALLBACKS: &str =
    "ax_mlx_kv_paged_attention_fallbacks";
pub const ROUTE_DECISION_AX_MLX_MODEL_MLA_KV_LATENT_DIM: &str = "ax_mlx_model_mla_kv_latent_dim";
pub const ROUTE_DECISION_AX_MLX_MODEL_MOE_ACTIVE_EXPERTS: &str = "ax_mlx_model_moe_active_experts";
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
pub const ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ENABLED: &str =
    "ax_scheduler_fair_multi_prefill_enabled";
pub const ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_CHUNK_TOKENS: &str =
    "ax_scheduler_fair_multi_prefill_chunk_tokens";
pub const ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMISSION_CAP: &str =
    "ax_scheduler_fair_multi_prefill_admission_cap";
pub const ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMITTED: &str =
    "ax_scheduler_fair_multi_prefill_admitted";
pub const ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_DEFERRED_BY_ADMISSION: &str =
    "ax_scheduler_fair_multi_prefill_deferred_by_admission";
pub const ROUTE_DECISION_AX_SCHEDULER_TOKEN_BUDGET_KEYS: [&str; 5] = [
    ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_PREFILL_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_SCHEDULED_DECODE_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_SKIPPED_PREFILL_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_SKIPPED_DECODE_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_MIXED_PREFILL_DECODE_BATCHES,
];
pub const ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_KEYS: [&str; 5] = [
    ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ENABLED,
    ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_CHUNK_TOKENS,
    ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMISSION_CAP,
    ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMITTED,
    ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_DEFERRED_BY_ADMISSION,
];
pub const ROUTE_DECISION_AX_MLX_KV_KEYS: [&str; 25] = [
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
    ROUTE_DECISION_AX_MLX_KV_ROTATED_RING_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_ROTATING_RING_SLACK,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB,
    ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT,
    ROUTE_DECISION_AX_MLX_KV_PAGED_MATERIALIZE_US,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_EXHAUSTION_FALLBACKS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_COW_COPIES,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_BLOCKS_USED,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SHARED_BLOCKS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLABS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_KIB,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_GROW_EVENTS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_CALLS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_FALLBACKS,
];
pub const ROUTE_DECISION_AX_MLX_MODEL_KEYS: [&str; 2] = [
    ROUTE_DECISION_AX_MLX_MODEL_MLA_KV_LATENT_DIM,
    ROUTE_DECISION_AX_MLX_MODEL_MOE_ACTIVE_EXPERTS,
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
    /// Strategy-planned work unit for this item (ADR-038).
    ///
    /// Prefill remains [`WorkUnitKind::PrefillChunk`]. Decode maps to
    /// [`WorkUnitKind::TokenDecode`] for AR, [`WorkUnitKind::DenoiseStep`] for
    /// block diffusion (denoise dominates a diffusion "decode" step today), or
    /// [`WorkUnitKind::EmbedForward`] for encoder-embed models.
    pub planned_work_unit: WorkUnitKind,
    pub input_token_slice: Vec<u32>,
    pub reused_prefix_token_slice: Vec<u32>,
    pub position_range: PositionRange,
    pub scheduled_token_count: u32,
    pub block_table_ref: RequestId,
    pub prefix_tokens_reused: u32,
    pub prefix_blocks_reused: u32,
}

/// Plan the work unit for a request snapshot via generation strategy metadata.
pub fn plan_work_unit_for_snapshot(snapshot: &RequestSnapshot) -> WorkUnitKind {
    let progress = GenerationProgress {
        processed_prompt_tokens: snapshot.processed_prompt_tokens,
        prompt_len: snapshot.prompt_len,
        generated_visible_tokens: snapshot.generated_len,
        denoise_steps_in_block: snapshot.diffusion_denoise_steps_in_block,
        commit_ready: snapshot.diffusion_commit_ready,
        block_committed: snapshot.diffusion_block_committed,
    };
    GenerationStrategyDescriptor::for_kind(snapshot.generation_kind).plan_next_work_unit(progress)
}

/// Map legacy execution mode to an AR work unit (tests / fallbacks).
pub fn work_unit_for_execution_mode(mode: ExecutionMode) -> WorkUnitKind {
    match mode {
        ExecutionMode::Prefill => WorkUnitKind::PrefillChunk,
        ExecutionMode::Decode => WorkUnitKind::TokenDecode,
    }
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

        // Multi-model batches remain invalid, but rotate the model anchor by
        // step so a long-running request from the oldest model cannot starve
        // every other model indefinitely. Model order is first-arrival order,
        // making the choice deterministic for a given SchedulerInput.
        let mut runnable_models = Vec::new();
        for snapshot in &runnable {
            if !runnable_models.contains(&snapshot.model_id) {
                runnable_models.push(snapshot.model_id.clone());
            }
        }
        let model_index = input.step_id.0 as usize % runnable_models.len();
        let batch_model_id = runnable_models[model_index].clone();
        let mut remaining_budget = input.global_token_budget;
        let mut selected_requests = Vec::new();
        let mut deferred_requests = Vec::new();
        let mut items = Vec::new();
        // Generation kind of the first selected item (for batch telemetry).
        let mut selected_generation_kind: Option<GenerationKind> = None;
        let mut candidates = Vec::new();
        let mut route_seed_anchor: Option<BatchRouteSeed> = None;
        // First-seen seed per execution mode. Checking candidates only against
        // the batch anchor lets two same-mode requests with different execution
        // plans share a mixed batch (each passing the looser cross-mode check);
        // every candidate must also pass the strict same-mode check against its
        // own mode's seed.
        let mut mode_route_seeds: Vec<BatchRouteSeed> = Vec::new();
        let mut batch_mixes_prefill_decode = false;
        let mut token_budget = TokenBudgetTelemetry::default();
        let mut pressure_prefill_budget =
            prefill_budget_for_memory_pressure(input.memory_pressure.as_deref());
        // Fair multi-prefill is decode-progress-preserving and only engages
        // when there is no KV memory pressure. Under pressure the existing
        // one-token / defer policy remains the sole prefill throttle.
        let fair_active = input.multi_prefill_fair && pressure_prefill_budget.is_none();
        let mut admitted_prefill_requests: u32 = 0;
        // fair_chunk / prefill_request_cap are sized lazily, right before the
        // first prefill candidate is considered, from whatever budget is left
        // once decode-priority candidates have already been served (see the
        // loop below) — not the full step budget — so a single competing
        // prefill isn't throttled to the block-size floor just because
        // decode hasn't run yet.
        let mut fair_chunk: u32 = 0;
        let mut prefill_request_cap: u32 = u32::MAX;
        let mut fair_admission_initialized = false;
        let mut fair_telemetry = FairMultiPrefillTelemetry::default();
        if fair_active {
            fair_telemetry.enabled = 1;
        }

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

        // Splittable-text prefill candidates share the fair chunk; multimodal
        // prefills stay atomic and are excluded from the fair-N denominator.
        let candidate_prefill_count = candidates
            .iter()
            .filter(|(snapshot, mode)| {
                *mode == ExecutionMode::Prefill && !snapshot.has_multimodal_inputs
            })
            .count() as u32;

        for (snapshot, mode) in candidates {
            let requested_tokens = schedulable_token_count(&snapshot, mode);
            if remaining_budget == 0 {
                token_budget.record_skipped(mode, requested_tokens);
                deferred_requests.push(snapshot.request_id);
                continue;
            }

            if fair_active && !fair_admission_initialized && mode == ExecutionMode::Prefill {
                fair_admission_initialized = true;
                fair_chunk = fair_prefill_chunk_tokens(
                    input.max_prefill_tokens_per_request_per_step,
                    input.block_size_tokens,
                    remaining_budget,
                    candidate_prefill_count,
                );
                prefill_request_cap = fair_prefill_admission_cap(
                    input.max_inflight_prefill_requests,
                    fair_chunk,
                    input.block_size_tokens,
                    input.available_kv_blocks,
                );
                fair_telemetry.fair_chunk_tokens = fair_chunk;
                fair_telemetry.admission_cap = if prefill_request_cap == u32::MAX {
                    0
                } else {
                    prefill_request_cap
                };
            }

            if mode == ExecutionMode::Prefill && admitted_prefill_requests >= prefill_request_cap {
                token_budget.record_skipped(mode, requested_tokens);
                deferred_requests.push(snapshot.request_id);
                fair_telemetry.deferred_by_admission =
                    fair_telemetry.deferred_by_admission.saturating_add(1);
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
                (ExecutionMode::Prefill, None)
                    if fair_active && !snapshot.has_multimodal_inputs =>
                {
                    // Text-only fair cap. Multimodal remains full-or-defer via
                    // build_execution_item atomicity (no partial multimodal).
                    remaining_budget.min(fair_chunk)
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
            let can_join = mode_route_seeds
                .iter()
                .all(|seed| route_seed_can_join_batch(seed, &candidate_route));
            if !can_join {
                deferred_requests.push(snapshot.request_id);
                continue;
            }

            if let Some(anchor) = &route_seed_anchor {
                if anchor.mode != mode {
                    batch_mixes_prefill_decode = true;
                }
            } else {
                route_seed_anchor = Some(candidate_route.clone());
            }
            if !mode_route_seeds.iter().any(|seed| seed.mode == mode) {
                mode_route_seeds.push(candidate_route);
            }

            remaining_budget -= item.scheduled_token_count;
            if let (ExecutionMode::Prefill, Some(prefill_budget)) =
                (item.mode, pressure_prefill_budget)
            {
                pressure_prefill_budget =
                    Some(prefill_budget.saturating_sub(item.scheduled_token_count));
            }
            if item.mode == ExecutionMode::Prefill {
                admitted_prefill_requests = admitted_prefill_requests.saturating_add(1);
                fair_telemetry.admitted_prefills =
                    fair_telemetry.admitted_prefills.saturating_add(1);
            }
            token_budget.record_scheduled(item.mode, item.scheduled_token_count);
            if requested_tokens > item.scheduled_token_count {
                token_budget
                    .record_skipped(item.mode, requested_tokens - item.scheduled_token_count);
            }

            if selected_generation_kind.is_none() {
                selected_generation_kind = Some(snapshot.generation_kind);
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
            fair_telemetry.append_route_decisions(&mut route_metadata);
            // ADR-038: surface planned work units and generation kind on the batch.
            if let Some(first) = items.first() {
                upsert_route_decision(
                    &mut route_metadata.crossover_decisions,
                    ROUTE_DECISION_AX_MLX_GENERATION_WORK_UNIT,
                    first.planned_work_unit.telemetry_code(),
                );
            }
            // Always emit the request's generation kind (not work-unit inference)
            // so PrefillChunk / BlockCommit for diffusion still label correctly.
            if let Some(kind) = selected_generation_kind {
                upsert_route_decision(
                    &mut route_metadata.crossover_decisions,
                    ROUTE_DECISION_AX_MLX_GENERATION_KIND,
                    kind.telemetry_code(),
                );
            }
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
            let remaining_prompt = snapshot.prompt_len - snapshot.processed_prompt_tokens;
            // Multimodal prefill is atomic: the runner rejects any prefill item
            // that does not complete the prompt, and an item error permanently
            // fails the request. Defer the whole prefill until a step has
            // enough budget for the full remainder instead of splitting it.
            if snapshot.has_multimodal_inputs && remaining_budget < remaining_prompt {
                return None;
            }
            let scheduled_token_count = remaining_budget.min(remaining_prompt);
            let start = snapshot.processed_prompt_tokens as usize;
            let end = start + scheduled_token_count as usize;

            return Some(ExecutionItem {
                request_id: snapshot.request_id,
                mode: ExecutionMode::Prefill,
                planned_work_unit: plan_work_unit_for_snapshot(snapshot),
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
            planned_work_unit: plan_work_unit_for_snapshot(snapshot),
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

/// Fair prefill chunk: explicit per-request cap; else
/// `max(block_size_tokens, residual_budget / candidate_prefill_count)` so a
/// single competing prefill gets close to the full residual budget and many
/// competitors divide it evenly, instead of always collapsing to the
/// block-size floor.
fn fair_prefill_chunk_tokens(
    max_tokens_per_request: u32,
    block_size_tokens: u32,
    residual_budget: u32,
    candidate_prefill_count: u32,
) -> u32 {
    if max_tokens_per_request != 0 {
        return max_tokens_per_request;
    }
    let floor = block_size_tokens.max(1);
    let share = residual_budget / candidate_prefill_count.max(1);
    floor.max(share)
}

/// Admit min(configured max, free-block headroom). `0` configured max → unlimited
/// by count (still headroom-limited). Headroom uses ceil(fair_chunk / block_size).
/// Nonzero headroom too small for one full fair chunk still admits a single
/// greedy prefill so fair mode cannot stall all prefill progress while free
/// blocks exist (design doc §B.2 item 8); only `available_kv_blocks == 0`
/// defers entirely to the existing allocate / blocked-on-memory path.
fn fair_prefill_admission_cap(
    max_inflight_prefill_requests: u32,
    fair_chunk: u32,
    block_size_tokens: u32,
    available_kv_blocks: u32,
) -> u32 {
    let block_size = block_size_tokens.max(1);
    let blocks_per = fair_chunk.div_ceil(block_size).max(1);
    let headroom_cap = available_kv_blocks / blocks_per;
    let headroom_cap = if headroom_cap == 0 && available_kv_blocks > 0 {
        1
    } else {
        headroom_cap
    };
    if max_inflight_prefill_requests == 0 {
        headroom_cap
    } else {
        max_inflight_prefill_requests.min(headroom_cap)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct FairMultiPrefillTelemetry {
    enabled: u32,
    fair_chunk_tokens: u32,
    admission_cap: u32,
    admitted_prefills: u32,
    deferred_by_admission: u32,
}

impl FairMultiPrefillTelemetry {
    fn append_route_decisions(&self, route_metadata: &mut RouteMetadata) {
        if self.enabled == 0 {
            return;
        }
        route_metadata.crossover_decisions.extend([
            (
                ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ENABLED.into(),
                self.enabled,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_CHUNK_TOKENS.into(),
                self.fair_chunk_tokens,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMISSION_CAP.into(),
                self.admission_cap,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMITTED.into(),
                self.admitted_prefills,
            ),
            (
                ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_DEFERRED_BY_ADMISSION.into(),
                self.deferred_by_admission,
            ),
        ]);
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
            has_multimodal_inputs: false,
            execution_plan_ref: None,
            route_metadata_hint: RouteMetadata::empty(),
            terminal_stop_reason: None,
            last_error: None,
            generation_kind: GenerationKind::Autoregressive,
            diffusion_denoise_steps_in_block: 0,
            diffusion_commit_ready: false,
            diffusion_block_committed: false,
        }
    }

    #[test]
    fn plans_diffusion_commit_when_runner_marks_ready() {
        let mut snapshot = make_snapshot(1, 1, "diffusion_gemma", &[1, 2, 3, 4], 4, &[], 256);
        snapshot.generation_kind = GenerationKind::BlockDiffusion;
        snapshot.diffusion_denoise_steps_in_block = 12;
        snapshot.diffusion_commit_ready = true;
        snapshot.diffusion_block_committed = false;
        assert_eq!(
            plan_work_unit_for_snapshot(&snapshot),
            WorkUnitKind::BlockCommit
        );

        // After commit, next unit is denoise for the following block.
        snapshot.diffusion_commit_ready = false;
        snapshot.diffusion_block_committed = true;
        snapshot.diffusion_denoise_steps_in_block = 0;
        assert_eq!(
            plan_work_unit_for_snapshot(&snapshot),
            WorkUnitKind::DenoiseStep
        );
    }

    #[test]
    fn plans_diffusion_decode_as_denoise_work_unit() {
        let scheduler = Scheduler::new();
        let mut snapshot = make_snapshot(1, 1, "diffusion_gemma", &[1, 2, 3, 4], 4, &[], 256);
        snapshot.generation_kind = GenerationKind::BlockDiffusion;
        let plan = scheduler.plan(&SchedulerInput::new(StepId(1), vec![snapshot], None, 32));
        let batch = plan.execution_batch.expect("batch");
        assert_eq!(batch.items.len(), 1);
        assert_eq!(batch.items[0].mode, ExecutionMode::Decode);
        assert_eq!(batch.items[0].planned_work_unit, WorkUnitKind::DenoiseStep);
        assert!(
            batch
                .route_metadata
                .crossover_decisions
                .iter()
                .any(|(k, v)| k == ROUTE_DECISION_AX_MLX_GENERATION_WORK_UNIT
                    && *v == WorkUnitKind::DenoiseStep.telemetry_code())
        );
        assert!(
            batch
                .route_metadata
                .crossover_decisions
                .iter()
                .any(|(k, v)| k == ROUTE_DECISION_AX_MLX_GENERATION_KIND
                    && *v == GenerationKind::BlockDiffusion.telemetry_code())
        );
    }

    #[test]
    fn diffusion_prefill_still_emits_block_diffusion_generation_kind() {
        let scheduler = Scheduler::new();
        let mut snapshot = make_snapshot(1, 1, "diffusion_gemma", &[1, 2, 3, 4], 0, &[], 256);
        snapshot.generation_kind = GenerationKind::BlockDiffusion;
        let plan = scheduler.plan(&SchedulerInput::new(StepId(3), vec![snapshot], None, 32));
        let batch = plan.execution_batch.expect("batch");
        assert_eq!(batch.items[0].planned_work_unit, WorkUnitKind::PrefillChunk);
        assert!(
            batch
                .route_metadata
                .crossover_decisions
                .iter()
                .any(|(k, v)| k == ROUTE_DECISION_AX_MLX_GENERATION_KIND
                    && *v == GenerationKind::BlockDiffusion.telemetry_code()),
            "diffusion prefill must still label generation_kind (not only DenoiseStep)"
        );
    }

    #[test]
    fn plans_ar_prefill_and_decode_work_units() {
        let scheduler = Scheduler::new();
        let prefill = make_snapshot(1, 1, "qwen3", &[1, 2, 3, 4], 0, &[], 16);
        let decode = make_snapshot(2, 2, "qwen3", &[1, 2, 3, 4], 4, &[5], 16);
        assert_eq!(
            plan_work_unit_for_snapshot(&prefill),
            WorkUnitKind::PrefillChunk
        );
        assert_eq!(
            plan_work_unit_for_snapshot(&decode),
            WorkUnitKind::TokenDecode
        );
        let plan = scheduler.plan(&SchedulerInput::new(
            StepId(2),
            vec![prefill, decode],
            None,
            32,
        ));
        let batch = plan.execution_batch.expect("batch");
        // Decode-first ordering: decode item first.
        assert_eq!(batch.items[0].planned_work_unit, WorkUnitKind::TokenDecode);
        assert_eq!(batch.items[1].planned_work_unit, WorkUnitKind::PrefillChunk);
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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(4),
            vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22], 0, &[], 16),
                make_snapshot(3, 3, "llama", &[30, 31], 0, &[], 16),
            ],
            None,
            5,
        ));

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

    fn make_multimodal_snapshot(
        request_id: u64,
        arrival_sequence: u64,
        model_id: &str,
        prompt_tokens: &[u32],
        processed_prompt_tokens: u32,
    ) -> RequestSnapshot {
        let mut snapshot = make_snapshot(
            request_id,
            arrival_sequence,
            model_id,
            prompt_tokens,
            processed_prompt_tokens,
            &[],
            16,
        );
        snapshot.has_multimodal_inputs = true;
        snapshot
    }

    #[test]
    fn defers_multimodal_prefill_instead_of_splitting() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(4),
            vec![make_multimodal_snapshot(
                1,
                1,
                "gemma4",
                &[10, 11, 12, 13],
                0,
            )],
            None,
            3,
        ));

        assert!(schedule_plan.selected_requests.is_empty());
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
        assert!(schedule_plan.execution_batch.is_none());
    }

    #[test]
    fn schedules_multimodal_prefill_atomically_when_budget_fits() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(4),
            vec![make_multimodal_snapshot(
                1,
                1,
                "gemma4",
                &[10, 11, 12, 13],
                0,
            )],
            None,
            4,
        ));

        assert_eq!(schedule_plan.selected_requests, vec![RequestId(1)]);
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.items.len(), 1);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 4);
        assert_eq!(
            execution_batch.items[0].input_token_slice,
            vec![10, 11, 12, 13]
        );
    }

    #[test]
    fn multimodal_prefill_defers_while_text_prefill_still_chunks() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(4),
            vec![
                make_multimodal_snapshot(1, 1, "gemma4", &[10, 11, 12, 13], 0),
                make_snapshot(2, 2, "gemma4", &[20, 21, 22, 23], 0, &[], 16),
            ],
            None,
            3,
        ));

        // The older multimodal prefill cannot complete in this step's budget,
        // so it defers whole; the younger text prefill still chunks.
        assert_eq!(schedule_plan.selected_requests, vec![RequestId(2)]);
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.items.len(), 1);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Prefill);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 3);
    }

    #[test]
    fn defers_multimodal_prefill_when_decode_consumes_budget() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(4),
            vec![
                make_multimodal_snapshot(1, 1, "gemma4", &[10, 11, 12, 13], 0),
                make_snapshot(2, 2, "gemma4", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            None,
            4,
        ));

        // Decode is scheduled first and leaves only 3 budget tokens — not
        // enough for the 4-token multimodal prompt, which must not split.
        assert_eq!(schedule_plan.selected_requests, vec![RequestId(2)]);
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.items.len(), 1);
        assert_eq!(execution_batch.items[0].mode, ExecutionMode::Decode);
    }

    #[test]
    fn schedules_multimodal_prefill_remainder_after_prefix_reuse() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(4),
            vec![make_multimodal_snapshot(
                1,
                1,
                "gemma4",
                &[10, 11, 12, 13, 14, 15],
                4,
            )],
            None,
            2,
        ));

        // Prefix reuse advanced processed_prompt_tokens; the remainder fits
        // the budget, so the prefill completes the prompt in one item.
        assert_eq!(schedule_plan.selected_requests, vec![RequestId(1)]);
        let execution_batch = schedule_plan.execution_batch.unwrap();
        assert_eq!(execution_batch.items.len(), 1);
        assert_eq!(execution_batch.items[0].scheduled_token_count, 2);
        assert_eq!(execution_batch.items[0].input_token_slice, vec![14, 15]);
    }

    #[test]
    fn builds_decode_item_from_last_known_token() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(8),
            vec![make_snapshot(9, 1, "qwen3", &[1, 2, 3], 3, &[7], 16)],
            None,
            1,
        ));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(9),
            vec![make_snapshot(10, 1, "qwen3", &[1, 2, 3], 3, &[], 16)],
            None,
            1,
        ));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(10),
            vec![make_snapshot(12, 1, "qwen3", &[1, 2, 3], 3, &[7, 8], 2)],
            None,
            1,
        ));

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

        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(11),
            vec![prefill, decode],
            None,
            3,
        ));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(14),
            vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            None,
            1,
        ));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(15),
            vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            Some("kv_low_free_blocks:1/8".into()),
            8,
        ));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(16),
            vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            Some("kv_exhausted".into()),
            8,
        ));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(17),
            vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12, 13], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 4, &[99], 16),
            ],
            Some("kv_exhausted_reclaimable_cache".into()),
            8,
        ));

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

        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(12),
            vec![first, second],
            None,
            8,
        ));

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
    fn does_not_mix_different_prefill_plans_behind_a_decode_anchor() {
        let scheduler = Scheduler::new();
        let mixed_compatible_kv = |execution_plan: &str, attention_route: &str| RouteMetadata {
            execution_plan: Some(execution_plan.into()),
            attention_route: Some(attention_route.into()),
            kv_mode: Some("paged_metadata".into()),
            prefix_cache_path: None,
            barrier_mode: Some("serial".into()),
            crossover_decisions: Vec::new(),
        };

        let mut decode = make_snapshot(1, 1, "qwen3", &[20, 21, 22, 23], 4, &[99], 16);
        decode.execution_plan_ref = Some("phase1.qwen3.paged_decode".into());
        decode.route_metadata_hint =
            mixed_compatible_kv("phase1.qwen3.paged_decode", "qwen3_paged_decode");
        let mut first_prefill = make_snapshot(2, 2, "qwen3", &[10, 11], 0, &[], 16);
        first_prefill.execution_plan_ref = Some("phase1.qwen3.dense_prefill".into());
        first_prefill.route_metadata_hint =
            mixed_compatible_kv("phase1.qwen3.dense_prefill", "qwen3_prefill");
        let mut second_prefill = make_snapshot(3, 3, "qwen3", &[30, 31], 0, &[], 16);
        second_prefill.execution_plan_ref = Some("phase1.qwen3.special_prefill".into());
        second_prefill.route_metadata_hint =
            mixed_compatible_kv("phase1.qwen3.special_prefill", "qwen3_prefill_alt");

        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(14),
            vec![decode, first_prefill, second_prefill],
            None,
            8,
        ));

        // Both prefills are mixed-compatible with the decode anchor, but they
        // carry different execution plans, so only the first may join.
        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(1), RequestId(2)]
        );
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(3)]);
    }

    #[test]
    fn does_not_synthesize_generic_route_labels_without_binding_hint() {
        let scheduler = Scheduler::new();
        let mut snapshot = make_snapshot(1, 1, "qwen3", &[10, 11, 12], 0, &[], 16);
        snapshot.execution_plan_ref = Some("phase1.qwen3.dense_prefill".into());

        let schedule_plan =
            scheduler.plan(&SchedulerInput::new(StepId(13), vec![snapshot], None, 8));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(20),
            vec![
                make_snapshot(1, 1, "qwen3", &[10, 11, 12], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20, 21], 0, &[], 16),
            ],
            None,
            0,
        ));

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
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(21),
            vec![make_snapshot(1, 1, "qwen3", &[10, 11, 12], 3, &[7, 8], 2)],
            None,
            8,
        ));

        assert!(schedule_plan.selected_requests.is_empty());
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
        assert!(schedule_plan.execution_batch.is_none());
    }

    #[test]
    fn defers_different_model_requests_to_next_step() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(
            StepId(22),
            vec![
                make_snapshot(1, 1, "qwen3", &[10, 11], 0, &[], 16),
                make_snapshot(2, 2, "gemma", &[20, 21], 0, &[], 16),
                make_snapshot(3, 3, "qwen3", &[30, 31], 0, &[], 16),
            ],
            None,
            16,
        ));

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
    fn rotates_model_anchor_across_steps_to_prevent_starvation() {
        let scheduler = Scheduler::new();
        let snapshots = vec![
            make_snapshot(1, 1, "qwen3", &[10, 11], 0, &[], 16),
            make_snapshot(2, 2, "gemma", &[20, 21], 0, &[], 16),
            make_snapshot(3, 3, "qwen3", &[30, 31], 0, &[], 16),
        ];

        let first = scheduler.plan(&SchedulerInput::new(
            StepId(22),
            snapshots.clone(),
            None,
            16,
        ));
        let second = scheduler.plan(&SchedulerInput::new(StepId(23), snapshots, None, 16));

        assert_eq!(first.execution_batch.unwrap().model_id, "qwen3");
        assert_eq!(second.execution_batch.unwrap().model_id, "gemma");
        assert_eq!(second.selected_requests, vec![RequestId(2)]);
    }

    #[test]
    fn returns_empty_plan_when_no_runnable_requests() {
        let scheduler = Scheduler::new();
        let schedule_plan = scheduler.plan(&SchedulerInput::new(StepId(23), vec![], None, 16));

        assert!(schedule_plan.selected_requests.is_empty());
        assert!(schedule_plan.deferred_requests.is_empty());
        assert!(schedule_plan.execution_batch.is_none());
    }

    #[test]
    fn fair_multi_prefill_splits_budget_across_text_prefills() {
        let scheduler = Scheduler::new();
        let mut input = SchedulerInput::new(
            StepId(30),
            vec![
                make_snapshot(1, 1, "qwen3", &[10; 64], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20; 64], 0, &[], 16),
                make_snapshot(3, 3, "qwen3", &[30; 64], 0, &[], 16),
            ],
            None,
            48,
        );
        input.multi_prefill_fair = true;
        input.max_prefill_tokens_per_request_per_step = 16;
        input.max_inflight_prefill_requests = 3;
        input.block_size_tokens = 16;
        input.available_kv_blocks = 1024;
        input.total_kv_blocks = 1024;

        let schedule_plan = scheduler.plan(&input);
        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(1), RequestId(2), RequestId(3)]
        );
        let batch = schedule_plan.execution_batch.expect("batch");
        assert_eq!(batch.items.len(), 3);
        for item in &batch.items {
            assert_eq!(item.mode, ExecutionMode::Prefill);
            assert_eq!(item.scheduled_token_count, 16);
        }
        assert_eq!(batch.total_scheduled_tokens, 48);
        let decisions: std::collections::BTreeMap<_, _> = batch
            .route_metadata
            .crossover_decisions
            .into_iter()
            .collect();
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ENABLED),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMITTED),
            Some(&3)
        );
    }

    #[test]
    fn fair_prefill_interleave_bounds_a_lone_prefill_sharing_the_decode_cohort_step() {
        // The "fair chunked-prefill interleave" property (Phase 3): when a single
        // new prefill shares a step with an active decode cohort, the per-request
        // cap chunks that prefill so it cannot inflate the whole cohort's step
        // latency. Decode is priority 0 (served first, in full); the lone prefill
        // is bounded to the cap instead of grabbing the (ample) residual budget.
        let scheduler = Scheduler::new();
        let cohort_and_prefill = || {
            SchedulerInput::new(
                StepId(50),
                vec![
                    make_snapshot(1, 1, "qwen3", &[7; 8], 8, &[42], 128), // decode
                    make_snapshot(2, 2, "qwen3", &[7; 8], 8, &[43], 128), // decode
                    make_snapshot(3, 3, "qwen3", &[9; 256], 0, &[], 128), // long new prefill
                ],
                None,
                4096, // ample: an uncapped lone prefill would take all 256 tokens
            )
        };
        let find = |batch: &ExecutionBatch, id: u64| {
            batch
                .items
                .iter()
                .find(|it| it.request_id.0 == id)
                .expect("item present")
                .clone()
        };

        // Cap ON: prefill chunked to 32; decode cohort untouched (1 token each).
        let mut capped = cohort_and_prefill();
        capped.multi_prefill_fair = true;
        capped.max_prefill_tokens_per_request_per_step = 32;
        capped.block_size_tokens = 16;
        capped.available_kv_blocks = 1024;
        capped.total_kv_blocks = 1024;
        let batch = scheduler.plan(&capped).execution_batch.expect("batch");
        assert_eq!(find(&batch, 1).mode, ExecutionMode::Decode);
        assert_eq!(find(&batch, 1).scheduled_token_count, 1);
        assert_eq!(find(&batch, 2).scheduled_token_count, 1);
        assert_eq!(find(&batch, 3).mode, ExecutionMode::Prefill);
        assert_eq!(
            find(&batch, 3).scheduled_token_count,
            32,
            "lone prefill must be chunked to the interleave cap, not the residual budget"
        );

        // Cap OFF (0): the same lone prefill grabs its whole 256-token prompt in
        // one step — the latency spike the interleave cap exists to prevent. This
        // asserts the cap is load-bearing, not incidental.
        let mut uncapped = cohort_and_prefill();
        uncapped.multi_prefill_fair = true;
        uncapped.max_prefill_tokens_per_request_per_step = 0;
        uncapped.block_size_tokens = 16;
        uncapped.available_kv_blocks = 1024;
        uncapped.total_kv_blocks = 1024;
        let batch = scheduler.plan(&uncapped).execution_batch.expect("batch");
        assert_eq!(find(&batch, 3).scheduled_token_count, 256);
    }

    #[test]
    fn fair_prefill_interleave_advances_decode_every_step_while_chunking_a_long_prefill() {
        // The *temporal* interleave (what "chunked-prefill interleave" means):
        // drive `plan` across steps, advancing each request's state by what it
        // was scheduled. A 1024-token prefill (cap 256) trickles in over 4 steps,
        // and the 2-request decode cohort advances one token on EVERY one of
        // those steps — decode never stalls waiting for the long prefill to
        // finish. Deterministic scheduler simulation; no serving stack needed.
        let scheduler = Scheduler::new();
        const CAP: u32 = 256;
        const PROMPT: usize = 1024;
        let mut prefill_processed: u32 = 0;
        let mut decode_generated: [usize; 2] = [1, 1]; // already decoding
        let mut prefill_steps = 0u32;
        let mut decode_steps_advanced = 0u32;

        for _ in 0..16 {
            // guard against a non-terminating loop
            if prefill_processed >= PROMPT as u32 {
                break;
            }
            let snaps = vec![
                make_snapshot(
                    1,
                    1,
                    "qwen3",
                    &[7; 8],
                    8,
                    &vec![9u32; decode_generated[0]],
                    512,
                ),
                make_snapshot(
                    2,
                    2,
                    "qwen3",
                    &[7; 8],
                    8,
                    &vec![9u32; decode_generated[1]],
                    512,
                ),
                make_snapshot(
                    3,
                    3,
                    "qwen3",
                    &vec![5u32; PROMPT],
                    prefill_processed,
                    &[],
                    512,
                ),
            ];
            let mut input = SchedulerInput::new(StepId(100), snaps, None, 4096);
            input.multi_prefill_fair = true;
            input.max_prefill_tokens_per_request_per_step = CAP;
            input.block_size_tokens = 16;
            input.available_kv_blocks = 4096;
            input.total_kv_blocks = 4096;
            let batch = scheduler.plan(&input).execution_batch.expect("batch");
            let find = |id: u64| batch.items.iter().find(|it| it.request_id.0 == id).cloned();

            let prefill = find(3).expect("prefill scheduled");
            assert_eq!(prefill.mode, ExecutionMode::Prefill);
            assert!(
                prefill.scheduled_token_count <= CAP,
                "prefill chunk must respect the interleave cap"
            );
            prefill_processed += prefill.scheduled_token_count;
            prefill_steps += 1;

            let decode_advanced = find(1).is_some_and(|it| it.mode == ExecutionMode::Decode)
                && find(2).is_some_and(|it| it.mode == ExecutionMode::Decode);
            assert!(
                decode_advanced,
                "decode cohort must advance on every prefill step (no stall)"
            );
            decode_generated[0] += 1;
            decode_generated[1] += 1;
            decode_steps_advanced += 1;
        }

        assert_eq!(
            prefill_steps,
            (PROMPT as u32).div_ceil(CAP),
            "1024-token prefill chunked across ceil(1024/256)=4 steps"
        );
        assert_eq!(
            decode_steps_advanced, prefill_steps,
            "decode advanced on every one of those steps — the interleave"
        );
    }

    #[test]
    fn fair_multi_prefill_headroom_limits_admission() {
        let scheduler = Scheduler::new();
        // fair_chunk=16, block_size=16 → 1 block per prefill; available=1 → admit 1.
        let mut input = SchedulerInput::new(
            StepId(31),
            vec![
                make_snapshot(1, 1, "qwen3", &[10; 32], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20; 32], 0, &[], 16),
            ],
            None,
            64,
        );
        input.multi_prefill_fair = true;
        input.max_prefill_tokens_per_request_per_step = 16;
        input.max_inflight_prefill_requests = 4;
        input.block_size_tokens = 16;
        input.available_kv_blocks = 1;
        input.total_kv_blocks = 64;

        let schedule_plan = scheduler.plan(&input);
        assert_eq!(schedule_plan.selected_requests, vec![RequestId(1)]);
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(2)]);
        let batch = schedule_plan.execution_batch.expect("batch");
        let decisions: std::collections::BTreeMap<_, _> = batch
            .route_metadata
            .crossover_decisions
            .into_iter()
            .collect();
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ADMISSION_CAP),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_DEFERRED_BY_ADMISSION),
            Some(&1)
        );
    }

    #[test]
    fn fair_multi_prefill_disabled_under_memory_pressure() {
        let scheduler = Scheduler::new();
        let mut input = SchedulerInput::new(
            StepId(32),
            vec![
                make_snapshot(1, 1, "qwen3", &[10; 32], 0, &[], 16),
                make_snapshot(2, 2, "qwen3", &[20; 32], 0, &[], 16),
            ],
            Some("kv_low_free_blocks:4/64".into()),
            64,
        );
        input.multi_prefill_fair = true;
        input.max_prefill_tokens_per_request_per_step = 16;
        input.max_inflight_prefill_requests = 4;
        input.block_size_tokens = 16;
        input.available_kv_blocks = 4;
        input.total_kv_blocks = 64;

        let schedule_plan = scheduler.plan(&input);
        // Pressure path: one token total for prefill, greedy oldest-only.
        let batch = schedule_plan.execution_batch.expect("batch");
        assert_eq!(batch.total_scheduled_tokens, 1);
        assert_eq!(batch.items.len(), 1);
        assert_eq!(batch.items[0].request_id, RequestId(1));
        let decisions: std::collections::BTreeMap<_, _> = batch
            .route_metadata
            .crossover_decisions
            .into_iter()
            .collect();
        assert!(
            !decisions.contains_key(ROUTE_DECISION_AX_SCHEDULER_FAIR_MULTI_PREFILL_ENABLED),
            "fair telemetry must not engage under pressure"
        );
    }

    #[test]
    fn fair_multi_prefill_keeps_multimodal_atomic() {
        let scheduler = Scheduler::new();
        let mut input = SchedulerInput::new(
            StepId(33),
            vec![
                make_multimodal_snapshot(1, 1, "qwen3", &[10, 11, 12, 13, 14, 15, 16, 17], 0),
                make_snapshot(2, 2, "qwen3", &[20; 32], 0, &[], 16),
            ],
            None,
            24,
        );
        input.multi_prefill_fair = true;
        input.max_prefill_tokens_per_request_per_step = 4;
        input.max_inflight_prefill_requests = 4;
        input.block_size_tokens = 16;
        input.available_kv_blocks = 1024;
        input.total_kv_blocks = 1024;

        let schedule_plan = scheduler.plan(&input);
        // Multimodal needs 8 tokens but fair would only offer residual after
        // or full remainder; atomic path defers multimodal when residual < 8.
        // Oldest multimodal with budget 24: build_execution_item sees full
        // remaining_budget for multimodal (not fair-capped), so it takes 8.
        // Then text gets fair-capped 4.
        assert!(
            schedule_plan.selected_requests.contains(&RequestId(1)),
            "multimodal with enough residual budget must schedule atomically"
        );
        let batch = schedule_plan.execution_batch.expect("batch");
        let mm = batch
            .items
            .iter()
            .find(|item| item.request_id == RequestId(1))
            .expect("multimodal item");
        assert_eq!(mm.scheduled_token_count, 8);
    }

    #[test]
    fn fair_multi_prefill_admission_cap_can_defer_multimodal_via_shared_counter() {
        // Multimodal prefills are excluded from the fair-N *chunk-sizing*
        // denominator (candidate_prefill_count) and are exempt from the
        // per-request *token* cap (always full-or-defer, never split — see
        // `fair_multi_prefill_keeps_multimodal_atomic`). They are NOT
        // exempt from the admission *count* gate
        // (`admitted_prefill_requests >= prefill_request_cap`): that check
        // fires for any Prefill-mode candidate regardless of modality. A
        // multimodal item arriving after the cap is already exhausted by
        // text prefills gets deferred purely by the shared counter, even
        // though its own token/block footprint would trivially fit. The
        // design doc's own wording here is soft ("may use a block
        // estimate... when deciding whether residual can host it") rather
        // than mandating a separate admission path, so this test pins the
        // current behavior as a known interaction rather than asserting it
        // is the only correct design.
        let scheduler = Scheduler::new();
        let mut input = SchedulerInput::new(
            StepId(37),
            vec![
                make_snapshot(1, 1, "qwen3", &[10; 32], 0, &[], 16),
                make_multimodal_snapshot(2, 2, "qwen3", &[20, 21, 22, 23], 0),
            ],
            None,
            64,
        );
        input.multi_prefill_fair = true;
        input.max_prefill_tokens_per_request_per_step = 16;
        input.max_inflight_prefill_requests = 1;
        input.block_size_tokens = 16;
        input.available_kv_blocks = 1024;
        input.total_kv_blocks = 1024;

        let schedule_plan = scheduler.plan(&input);
        assert!(
            schedule_plan.selected_requests.contains(&RequestId(1)),
            "the earlier text prefill fills the admission cap"
        );
        assert!(
            schedule_plan.deferred_requests.contains(&RequestId(2)),
            "multimodal item deferred by the shared admission-count gate \
             even though its own footprint would trivially fit"
        );
    }

    #[test]
    fn fair_multi_prefill_auto_chunk_scales_with_residual_budget_and_candidates() {
        let scheduler = Scheduler::new();
        let mut input = SchedulerInput::new(
            StepId(34),
            vec![make_snapshot(1, 1, "qwen3", &[10; 100], 0, &[], 16)],
            None,
            2048,
        );
        input.multi_prefill_fair = true;
        // 0 = auto: fair_chunk must scale with residual_budget / candidate
        // count, not collapse to the block_size_tokens floor when there is
        // no real contention (regression for a bug where a lone prefill was
        // throttled to 16 tokens/step even under a 2048-token budget with no
        // competing prefills).
        input.max_prefill_tokens_per_request_per_step = 0;
        input.max_inflight_prefill_requests = 0;
        input.block_size_tokens = 16;
        input.available_kv_blocks = 1024;
        input.total_kv_blocks = 1024;

        let schedule_plan = scheduler.plan(&input);
        let batch = schedule_plan.execution_batch.expect("batch");
        assert_eq!(batch.items.len(), 1);
        assert_eq!(
            batch.items[0].scheduled_token_count, 100,
            "lone prefill under auto fair-chunk should get the full prompt in \
             one step, not be throttled to the block-size floor"
        );
    }

    #[test]
    fn fair_multi_prefill_admits_one_greedy_prefill_when_headroom_below_one_chunk() {
        let scheduler = Scheduler::new();
        let mut input = SchedulerInput::new(
            StepId(35),
            vec![make_snapshot(1, 1, "qwen3", &[10; 300], 0, &[], 16)],
            None,
            1024,
        );
        input.multi_prefill_fair = true;
        input.max_prefill_tokens_per_request_per_step = 256;
        input.max_inflight_prefill_requests = 0;
        input.block_size_tokens = 16;
        // blocks_per = ceil(256/16) = 16; available=5 < 16 rounds headroom
        // down to 0 under naive division. Nonzero headroom must still admit
        // one greedy prefill instead of stalling all prefill progress
        // (regression for a liveness bug where fair mode could defer every
        // prefill candidate forever despite free blocks and residual budget
        // existing).
        input.available_kv_blocks = 5;
        input.total_kv_blocks = 64;

        let schedule_plan = scheduler.plan(&input);
        assert_eq!(
            schedule_plan.selected_requests,
            vec![RequestId(1)],
            "one free-but-tight-headroom prefill must still make progress"
        );
        let batch = schedule_plan.execution_batch.expect("batch");
        assert_eq!(batch.items.len(), 1);
        assert_eq!(batch.items[0].scheduled_token_count, 256);
    }

    #[test]
    fn fair_multi_prefill_defers_when_available_blocks_are_exactly_zero() {
        let scheduler = Scheduler::new();
        let mut input = SchedulerInput::new(
            StepId(36),
            vec![make_snapshot(1, 1, "qwen3", &[10; 300], 0, &[], 16)],
            None,
            1024,
        );
        input.multi_prefill_fair = true;
        input.max_prefill_tokens_per_request_per_step = 256;
        input.max_inflight_prefill_requests = 0;
        input.block_size_tokens = 16;
        // True exhaustion (0 free blocks) must still defer — the greedy
        // fallback only rescues nonzero-but-too-small headroom, not genuine
        // exhaustion (design doc §B.2: "if available_kv_blocks == 0 ...
        // still defer").
        input.available_kv_blocks = 0;
        input.total_kv_blocks = 64;

        let schedule_plan = scheduler.plan(&input);
        assert!(schedule_plan.selected_requests.is_empty());
        assert_eq!(schedule_plan.deferred_requests, vec![RequestId(1)]);
    }
}
