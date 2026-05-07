use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, MlxStream, add, astype, clear_cache, divide, enable_compile,
    max_recommended_working_set_size, multiply, power, set_wired_limit, sum_axis, take,
};

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::scheduler::ExecutionMode;
use ax_engine_core::{
    EmbeddingPooling, ExecutionRunner, ExecutionStatus, KvWriteSummary, MlxKvCompressionConfig,
    NativeModelArtifacts, NativeModelBindingSummary, NativeModelManifest, NativeTensorRole,
    ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB, ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_CANDIDATE_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ELIGIBLE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_COMPRESSED_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_SAVED_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FULL_PRECISION_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_ATTENTION_KIND,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_GQA,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_INELIGIBLE_LAYER,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_MISSING_STORAGE,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_PREFILL_ONLY,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_HEAD_DIM,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_PRESET,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_READY_CANDIDATES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_HOT_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEY_BITS, ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRESET,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_BLOCKERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_READY,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RATIO_MILLI,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ROUTE_METADATA_SCHEMA,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_WRITTEN_SLOTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_CALLS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_WALL_US,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_STATUS, ROUTE_DECISION_AX_MLX_KV_COMPRESSION_VALUE_BITS,
    ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS, ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB, ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB, ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS, RequestExecutionUpdate, RequestId, RunnerInput,
    RunnerOutput, StopReason,
};

use crate::generate::{
    advance_direct_pipeline_with_turboquant_context, chunked_prefill, decode_step,
    start_direct_pipeline_with_turboquant_context,
};
use crate::kv_cache::{MlxKVCache, MlxKVCacheUsage};
use crate::model::{
    Gemma4MoeProfileSnapshot, LinearAttentionProfileSnapshot, ModelConfig,
    TurboQuantModelDecodeContext, take_gemma4_moe_profile_snapshot,
    take_linear_attention_profile_snapshot,
};
use crate::ngram_accel::{
    DEFAULT_DRAFT_LEN, DRAFT_CONFIDENCE_THRESHOLD, LINEAR_MIN_NGRAM_SUPPORT, MAX_DRAFT_LEN,
    NgramTable, ngram_accel_decode_step, single_decode_with_turboquant_context,
};
use crate::sampling::Xorshift64;
use crate::turboquant::{
    TURBOQUANT_ROUTE_METADATA_SCHEMA_VERSION, TurboQuantProductionRequirements,
    turboquant_support_report,
};
use crate::weights::{ModelWeights, load_weights};

/// Beta prior counts for the n-gram acceleration accept-rate gate.
///
/// Beta(3, 1) → initial posterior mean = 0.75, above the accept threshold,
/// so n-gram acceleration is enabled optimistically from the first step and is only
/// suppressed once the posterior accumulates evidence of a low accept rate.
const NGRAM_BETA_PRIOR_ALPHA: f32 = 3.0;
const NGRAM_BETA_PRIOR_BETA: f32 = 1.0;

/// Cap total Beta observations to ~100 to bound the "memory" of the gate
/// and allow the posterior to adapt if token statistics change mid-sequence.
/// Equivalent to an EMA span of roughly 100 n-gram acceleration steps.
const NGRAM_BETA_MAX_TOTAL: f32 = 100.0;

const NGRAM_ACCEPT_THRESHOLD: f32 = 0.5;
const NGRAM_RETRY_INTERVAL: u32 = 8;
/// Steps to suppress n-gram acceleration after a complete miss (0 draft tokens accepted)
/// on a linear-attention model.  Recompute cost is O(1) token regardless of context
/// length, so 128 was far too conservative; 16 gives the n-gram table time to
/// recover without sacrificing the whole generation window.
const LINEAR_NGRAM_RETRY_INTERVAL: u32 = 16;
/// Steps to suppress after a *partial* accept (≥1 draft token accepted but not all).
/// Partial accept means the n-gram is close — retry quickly.
const LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL: u32 = 4;
/// If a linear-attention request cannot produce any n-gram draft after one full
/// retry window, stop probing for the rest of the request and use the direct
/// pipeline.  Qwen3.5 short prompts can start finding drafts after 16 generated
/// tokens, so the threshold intentionally gives them that full window first.
const LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD: u32 = LINEAR_NGRAM_RETRY_INTERVAL + 1;
/// Maximum number of prompt tail tokens fed into the n-gram table.
/// Long prompts (especially random-token benchmarks) would otherwise fill the
/// table with useless bigrams that trigger false-positive n-gram acceleration and force
/// expensive recompute on the very first n-gram acceleration attempt.
const NGRAM_PROMPT_FEED_MAX: usize = 64;
/// Minimum max_output_tokens budget required to enable n-gram acceleration.
/// Below this, failed speculation attempts + cooldown intervals (8-16 steps)
/// consume a disproportionate share of the total generation window.
const NGRAM_MIN_OUTPUT_FOR_ACCELERATION: u32 = 64;
const KV_COMPRESSION_DECODE_PATH_FULL_PRECISION_SHADOW: u32 = 1;
const KV_COMPRESSION_DECODE_PATH_FUSED_COMPRESSED_DECODE: u32 = 2;
const KV_COMPRESSION_DECODE_PATH_CPU_ORACLE_COMPRESSED_DECODE: u32 = 3;
const KV_COMPRESSION_FUSED_DECODE_FALLBACK_NONE: u32 = 0;
const KV_COMPRESSION_FUSED_DECODE_FALLBACK_SHADOW_ONLY: u32 = 1;
const KV_COMPRESSION_FUSED_DECODE_FALLBACK_MISSING_RUNTIME_STORAGE: u32 = 2;
const KV_COMPRESSION_FUSED_DECODE_FALLBACK_UNSUPPORTED_PRESET: u32 = 3;
const KV_COMPRESSION_FUSED_DECODE_FALLBACK_RUNNER_NOT_INTEGRATED: u32 = 4;
const KV_COMPRESSION_FUSED_DECODE_FALLBACK_CPU_ORACLE_UNAVAILABLE: u32 = 5;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NgramAccelerationTelemetry {
    draft_attempts: u32,
    draft_tokens: u32,
    accepted_tokens: u32,
    rejected_tokens: u32,
    full_accepts: u32,
    partial_rejects: u32,
    complete_misses: u32,
    no_draft_steps: u32,
    cooldown_steps: u32,
    cooldown_events: u32,
    cooldown_steps_scheduled: u32,
    request_disable_events: u32,
    request_disabled_steps: u32,
}

impl NgramAccelerationTelemetry {
    fn record_draft(&mut self, draft_len: usize, accept_count: usize) {
        self.draft_attempts = self.draft_attempts.saturating_add(1);
        self.draft_tokens = self.draft_tokens.saturating_add(saturating_u32(draft_len));
        self.accepted_tokens = self
            .accepted_tokens
            .saturating_add(saturating_u32(accept_count));
        self.rejected_tokens = self
            .rejected_tokens
            .saturating_add(saturating_u32(draft_len.saturating_sub(accept_count)));

        if accept_count == draft_len {
            self.full_accepts = self.full_accepts.saturating_add(1);
        } else if accept_count == 0 {
            self.complete_misses = self.complete_misses.saturating_add(1);
        } else {
            self.partial_rejects = self.partial_rejects.saturating_add(1);
        }
    }

    fn record_no_draft(&mut self) {
        self.no_draft_steps = self.no_draft_steps.saturating_add(1);
    }

    fn record_cooldown_step(&mut self) {
        self.cooldown_steps = self.cooldown_steps.saturating_add(1);
    }

    fn record_cooldown_event(&mut self, disabled_steps: u32) {
        self.cooldown_events = self.cooldown_events.saturating_add(1);
        self.cooldown_steps_scheduled =
            self.cooldown_steps_scheduled.saturating_add(disabled_steps);
    }

    fn record_request_disable_event(&mut self) {
        self.request_disable_events = self.request_disable_events.saturating_add(1);
    }

    fn record_request_disabled_step(&mut self) {
        self.request_disabled_steps = self.request_disabled_steps.saturating_add(1);
    }

    fn merge_from(&mut self, other: Self) {
        self.draft_attempts = self.draft_attempts.saturating_add(other.draft_attempts);
        self.draft_tokens = self.draft_tokens.saturating_add(other.draft_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(other.accepted_tokens);
        self.rejected_tokens = self.rejected_tokens.saturating_add(other.rejected_tokens);
        self.full_accepts = self.full_accepts.saturating_add(other.full_accepts);
        self.partial_rejects = self.partial_rejects.saturating_add(other.partial_rejects);
        self.complete_misses = self.complete_misses.saturating_add(other.complete_misses);
        self.no_draft_steps = self.no_draft_steps.saturating_add(other.no_draft_steps);
        self.cooldown_steps = self.cooldown_steps.saturating_add(other.cooldown_steps);
        self.cooldown_events = self.cooldown_events.saturating_add(other.cooldown_events);
        self.cooldown_steps_scheduled = self
            .cooldown_steps_scheduled
            .saturating_add(other.cooldown_steps_scheduled);
        self.request_disable_events = self
            .request_disable_events
            .saturating_add(other.request_disable_events);
        self.request_disabled_steps = self
            .request_disabled_steps
            .saturating_add(other.request_disabled_steps);
    }

    fn append_route_decisions(&self, decisions: &mut Vec<(String, u32)>) {
        let entries = [
            ("ax_ngram_draft_attempts", self.draft_attempts),
            ("ax_ngram_draft_tokens", self.draft_tokens),
            ("ax_ngram_accepted_tokens", self.accepted_tokens),
            ("ax_ngram_rejected_tokens", self.rejected_tokens),
            ("ax_ngram_full_accepts", self.full_accepts),
            ("ax_ngram_partial_rejects", self.partial_rejects),
            ("ax_ngram_complete_misses", self.complete_misses),
            ("ax_ngram_no_draft_steps", self.no_draft_steps),
            ("ax_ngram_cooldown_steps", self.cooldown_steps),
            ("ax_ngram_cooldown_events", self.cooldown_events),
            (
                "ax_ngram_cooldown_steps_scheduled",
                self.cooldown_steps_scheduled,
            ),
            (
                "ax_ngram_request_disable_events",
                self.request_disable_events,
            ),
            (
                "ax_ngram_request_disabled_steps",
                self.request_disabled_steps,
            ),
        ];

        for (key, value) in entries {
            upsert_route_decision(decisions, key, value);
        }
    }
}

fn saturating_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

fn saturating_u32_from_u64(value: u64) -> u32 {
    value.min(u32::MAX as u64) as u32
}

fn saturating_u32_from_u128(value: u128) -> u32 {
    value.min(u32::MAX as u128) as u32
}

fn elapsed_us(started: Instant) -> u32 {
    saturating_u32_from_u128(started.elapsed().as_micros())
}

fn direct_pipeline_clear_cache_due(emitted_tokens: u32, cadence: u32) -> bool {
    cadence != 0
        && emitted_tokens != 0
        && (emitted_tokens == 1 || emitted_tokens.saturating_sub(1).is_multiple_of(cadence))
}

fn kib_ceil(bytes: u64) -> u32 {
    if bytes == 0 {
        0
    } else {
        saturating_u32_from_u64(bytes.saturating_add(1023) / 1024)
    }
}

fn upsert_route_decision(decisions: &mut Vec<(String, u32)>, key: &str, value: u32) {
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

fn kv_layer_windows_from_config(cfg: &ModelConfig) -> Vec<Option<usize>> {
    let mut windows = vec![None; cfg.layer_count];
    for (idx, layer) in cfg.layer_configs.iter().enumerate().take(cfg.layer_count) {
        windows[idx] = layer.sliding_window;
    }
    windows
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DecodeTelemetry {
    prefill_steps: u32,
    prefill_wall_us: u32,
    decode_steps: u32,
    decode_wall_us: u32,
    direct_bootstrap_steps: u32,
    direct_bootstrap_wall_us: u32,
    direct_pipeline_steps: u32,
    direct_pipeline_wall_us: u32,
    single_decode_steps: u32,
    single_decode_wall_us: u32,
    ngram_decode_steps: u32,
    ngram_decode_wall_us: u32,
    bonus_tokens: u32,
}

impl DecodeTelemetry {
    fn record_prefill(&mut self, wall_us: u32) {
        self.prefill_steps = self.prefill_steps.saturating_add(1);
        self.prefill_wall_us = self.prefill_wall_us.saturating_add(wall_us);
    }

    fn record_decode(&mut self, wall_us: u32) {
        self.decode_steps = self.decode_steps.saturating_add(1);
        self.decode_wall_us = self.decode_wall_us.saturating_add(wall_us);
    }

    fn record_direct_bootstrap(&mut self, wall_us: u32) {
        self.direct_bootstrap_steps = self.direct_bootstrap_steps.saturating_add(1);
        self.direct_bootstrap_wall_us = self.direct_bootstrap_wall_us.saturating_add(wall_us);
    }

    fn record_direct_pipeline(&mut self, wall_us: u32) {
        self.direct_pipeline_steps = self.direct_pipeline_steps.saturating_add(1);
        self.direct_pipeline_wall_us = self.direct_pipeline_wall_us.saturating_add(wall_us);
    }

    fn record_single_decode(&mut self, wall_us: u32) {
        self.single_decode_steps = self.single_decode_steps.saturating_add(1);
        self.single_decode_wall_us = self.single_decode_wall_us.saturating_add(wall_us);
    }

    fn record_ngram_decode(&mut self, wall_us: u32) {
        self.ngram_decode_steps = self.ngram_decode_steps.saturating_add(1);
        self.ngram_decode_wall_us = self.ngram_decode_wall_us.saturating_add(wall_us);
    }

    fn record_bonus_token(&mut self) {
        self.bonus_tokens = self.bonus_tokens.saturating_add(1);
    }

    fn merge_from(&mut self, other: Self) {
        self.prefill_steps = self.prefill_steps.saturating_add(other.prefill_steps);
        self.prefill_wall_us = self.prefill_wall_us.saturating_add(other.prefill_wall_us);
        self.decode_steps = self.decode_steps.saturating_add(other.decode_steps);
        self.decode_wall_us = self.decode_wall_us.saturating_add(other.decode_wall_us);
        self.direct_bootstrap_steps = self
            .direct_bootstrap_steps
            .saturating_add(other.direct_bootstrap_steps);
        self.direct_bootstrap_wall_us = self
            .direct_bootstrap_wall_us
            .saturating_add(other.direct_bootstrap_wall_us);
        self.direct_pipeline_steps = self
            .direct_pipeline_steps
            .saturating_add(other.direct_pipeline_steps);
        self.direct_pipeline_wall_us = self
            .direct_pipeline_wall_us
            .saturating_add(other.direct_pipeline_wall_us);
        self.single_decode_steps = self
            .single_decode_steps
            .saturating_add(other.single_decode_steps);
        self.single_decode_wall_us = self
            .single_decode_wall_us
            .saturating_add(other.single_decode_wall_us);
        self.ngram_decode_steps = self
            .ngram_decode_steps
            .saturating_add(other.ngram_decode_steps);
        self.ngram_decode_wall_us = self
            .ngram_decode_wall_us
            .saturating_add(other.ngram_decode_wall_us);
        self.bonus_tokens = self.bonus_tokens.saturating_add(other.bonus_tokens);
    }

    fn append_route_decisions(&self, decisions: &mut Vec<(String, u32)>) {
        let entries = [
            ("ax_mlx_prefill_steps", self.prefill_steps),
            ("ax_mlx_prefill_wall_us", self.prefill_wall_us),
            ("ax_mlx_decode_steps", self.decode_steps),
            ("ax_mlx_decode_wall_us", self.decode_wall_us),
            ("ax_mlx_direct_bootstrap_steps", self.direct_bootstrap_steps),
            (
                "ax_mlx_direct_bootstrap_wall_us",
                self.direct_bootstrap_wall_us,
            ),
            ("ax_mlx_direct_pipeline_steps", self.direct_pipeline_steps),
            (
                "ax_mlx_direct_pipeline_wall_us",
                self.direct_pipeline_wall_us,
            ),
            ("ax_mlx_single_decode_steps", self.single_decode_steps),
            ("ax_mlx_single_decode_wall_us", self.single_decode_wall_us),
            ("ax_mlx_ngram_decode_steps", self.ngram_decode_steps),
            ("ax_mlx_ngram_decode_wall_us", self.ngram_decode_wall_us),
            ("ax_mlx_bonus_tokens", self.bonus_tokens),
        ];

        for (key, value) in entries {
            upsert_route_decision(decisions, key, value);
        }
    }
}

impl Gemma4MoeProfileSnapshot {
    fn merge_from(&mut self, other: Self) {
        self.enabled = self.enabled.max(other.enabled);
        self.decode_layers = self.decode_layers.saturating_add(other.decode_layers);
        self.topk_selections = self.topk_selections.saturating_add(other.topk_selections);
        self.sorted_gather_layers = self
            .sorted_gather_layers
            .saturating_add(other.sorted_gather_layers);
        self.unsorted_gather_layers = self
            .unsorted_gather_layers
            .saturating_add(other.unsorted_gather_layers);
        self.attention_wall_us = self
            .attention_wall_us
            .saturating_add(other.attention_wall_us);
        self.dense_wall_us = self.dense_wall_us.saturating_add(other.dense_wall_us);
        self.router_wall_us = self.router_wall_us.saturating_add(other.router_wall_us);
        self.expert_wall_us = self.expert_wall_us.saturating_add(other.expert_wall_us);
        self.post_wall_us = self.post_wall_us.saturating_add(other.post_wall_us);
    }

    fn append_route_decisions(&self, decisions: &mut Vec<(String, u32)>) {
        if self.enabled == 0 {
            return;
        }

        let entries = [
            ("ax_mlx_gemma4_moe_profile_enabled", self.enabled),
            (
                "ax_mlx_gemma4_moe_profile_decode_layers",
                self.decode_layers,
            ),
            (
                "ax_mlx_gemma4_moe_profile_topk_selections",
                self.topk_selections,
            ),
            (
                "ax_mlx_gemma4_moe_profile_sorted_gather_layers",
                self.sorted_gather_layers,
            ),
            (
                "ax_mlx_gemma4_moe_profile_unsorted_gather_layers",
                self.unsorted_gather_layers,
            ),
            (
                "ax_mlx_gemma4_moe_profile_attention_wall_us",
                self.attention_wall_us,
            ),
            (
                "ax_mlx_gemma4_moe_profile_dense_wall_us",
                self.dense_wall_us,
            ),
            (
                "ax_mlx_gemma4_moe_profile_router_wall_us",
                self.router_wall_us,
            ),
            (
                "ax_mlx_gemma4_moe_profile_expert_wall_us",
                self.expert_wall_us,
            ),
            ("ax_mlx_gemma4_moe_profile_post_wall_us", self.post_wall_us),
        ];

        for (key, value) in entries {
            upsert_route_decision(decisions, key, value);
        }
    }
}

impl LinearAttentionProfileSnapshot {
    fn merge_from(&mut self, other: Self) {
        self.enabled = self.enabled.max(other.enabled);
        self.layers = self.layers.saturating_add(other.layers);
        self.tokens = self.tokens.saturating_add(other.tokens);
        self.projection_wall_us = self
            .projection_wall_us
            .saturating_add(other.projection_wall_us);
        self.conv_wall_us = self.conv_wall_us.saturating_add(other.conv_wall_us);
        self.qk_norm_wall_us = self.qk_norm_wall_us.saturating_add(other.qk_norm_wall_us);
        self.recurrent_wall_us = self
            .recurrent_wall_us
            .saturating_add(other.recurrent_wall_us);
        self.output_wall_us = self.output_wall_us.saturating_add(other.output_wall_us);
    }

    fn append_route_decisions(&self, decisions: &mut Vec<(String, u32)>) {
        if self.enabled == 0 {
            return;
        }

        let entries = [
            ("ax_mlx_linear_attention_profile_enabled", self.enabled),
            ("ax_mlx_linear_attention_profile_layers", self.layers),
            ("ax_mlx_linear_attention_profile_tokens", self.tokens),
            (
                "ax_mlx_linear_attention_profile_projection_wall_us",
                self.projection_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_conv_wall_us",
                self.conv_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_qk_norm_wall_us",
                self.qk_norm_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_recurrent_wall_us",
                self.recurrent_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_output_wall_us",
                self.output_wall_us,
            ),
        ];

        for (key, value) in entries {
            upsert_route_decision(decisions, key, value);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct KvCacheTelemetry {
    request_snapshots: u32,
    logical_tokens: u64,
    capacity_tokens: u64,
    logical_bytes: u64,
    capacity_bytes: u64,
    full_attention_layers: u64,
    sliding_window_layers: u64,
    sliding_window_retained_tokens: u64,
    sliding_window_reclaimable_capacity_tokens: u64,
    sliding_window_reclaimable_capacity_bytes: u64,
    linear_state_layers: u64,
    linear_state_bytes: u64,
    growth_count: u64,
    compression_request_snapshots: u32,
    compression_status: u32,
    compression_preset: u32,
    compression_key_bits: u32,
    compression_value_bits: u32,
    compression_eligible_layers: u64,
    compression_candidate_token_layers: u64,
    compression_hot_token_layers: u64,
    compression_full_precision_bytes: u64,
    compression_estimated_compressed_bytes: u64,
    compression_estimated_saved_bytes: u64,
    compression_ratio_milli: u32,
    compression_route_metadata_schema: u32,
    compression_production_ready: u32,
    compression_production_blockers: u32,
    compression_runtime_storage_layers: u64,
    compression_runtime_storage_token_layers: u64,
    compression_runtime_storage_bytes: u64,
    compression_runtime_storage_written_slots: u64,
    compression_shadow_sync_calls: u64,
    compression_shadow_sync_wall_us: u64,
    compression_decode_path: u32,
    compression_fused_decode_candidates: u64,
    compression_fused_decode_attempts: u64,
    compression_fused_decode_successes: u64,
    compression_fused_decode_fallbacks: u64,
    compression_fused_decode_fallback_reason: u32,
    compression_fused_decode_ready_candidates: u64,
    compression_fused_decode_blocked_prefill_only: u64,
    compression_fused_decode_blocked_attention_kind: u64,
    compression_fused_decode_blocked_ineligible_layer: u64,
    compression_fused_decode_blocked_unsupported_preset: u64,
    compression_fused_decode_blocked_unsupported_head_dim: u64,
    compression_fused_decode_blocked_gqa: u64,
    compression_fused_decode_blocked_missing_storage: u64,
}

impl KvCacheTelemetry {
    fn record_compression_shadow_sync(&mut self, wall_us: u32) {
        self.compression_shadow_sync_calls = self.compression_shadow_sync_calls.saturating_add(1);
        self.compression_shadow_sync_wall_us = self
            .compression_shadow_sync_wall_us
            .saturating_add(wall_us as u64);
    }

    fn merge_from(&mut self, usage: MlxKVCacheUsage) {
        self.request_snapshots = self.request_snapshots.saturating_add(1);
        self.logical_tokens = self
            .logical_tokens
            .saturating_add(usage.logical_tokens as u64);
        self.capacity_tokens = self
            .capacity_tokens
            .saturating_add(usage.capacity_tokens as u64);
        self.logical_bytes = self.logical_bytes.saturating_add(usage.logical_bytes);
        self.capacity_bytes = self.capacity_bytes.saturating_add(usage.capacity_bytes);
        self.full_attention_layers = self
            .full_attention_layers
            .saturating_add(usage.full_attention_layers as u64);
        self.sliding_window_layers = self
            .sliding_window_layers
            .saturating_add(usage.sliding_window_layers as u64);
        self.sliding_window_retained_tokens = self
            .sliding_window_retained_tokens
            .saturating_add(usage.sliding_window_retained_tokens as u64);
        self.sliding_window_reclaimable_capacity_tokens = self
            .sliding_window_reclaimable_capacity_tokens
            .saturating_add(usage.sliding_window_reclaimable_capacity_tokens as u64);
        self.sliding_window_reclaimable_capacity_bytes = self
            .sliding_window_reclaimable_capacity_bytes
            .saturating_add(usage.sliding_window_reclaimable_capacity_bytes);
        self.linear_state_layers = self
            .linear_state_layers
            .saturating_add(usage.linear_state_layers as u64);
        self.linear_state_bytes = self
            .linear_state_bytes
            .saturating_add(usage.linear_state_bytes);
        self.growth_count = self.growth_count.saturating_add(usage.growth_count);

        let compression = usage.kv_compression;
        if compression.policy_enabled {
            let production_readiness =
                TurboQuantProductionRequirements::mlx_shadow_fused_kernel().evaluate();
            self.compression_request_snapshots =
                self.compression_request_snapshots.saturating_add(1);
            self.compression_status = compression.status_code;
            self.compression_preset = compression.preset_code;
            self.compression_key_bits = compression.key_bits;
            self.compression_value_bits = compression.value_bits;
            self.compression_decode_path = KV_COMPRESSION_DECODE_PATH_FULL_PRECISION_SHADOW;
            self.compression_route_metadata_schema = TURBOQUANT_ROUTE_METADATA_SCHEMA_VERSION;
            self.compression_production_ready = u32::from(production_readiness.is_ready());
            self.compression_production_blockers =
                saturating_u32_from_u64(production_readiness.blockers.len() as u64);
            self.compression_runtime_storage_layers = self
                .compression_runtime_storage_layers
                .saturating_add(compression.runtime_storage_layers as u64);
            self.compression_runtime_storage_token_layers = self
                .compression_runtime_storage_token_layers
                .saturating_add(compression.runtime_storage_token_layers as u64);
            self.compression_runtime_storage_bytes = self
                .compression_runtime_storage_bytes
                .saturating_add(compression.runtime_storage_bytes);
            self.compression_runtime_storage_written_slots = self
                .compression_runtime_storage_written_slots
                .saturating_add(compression.runtime_storage_written_slots as u64);
            self.compression_eligible_layers = self
                .compression_eligible_layers
                .saturating_add(compression.eligible_layers as u64);
            self.compression_candidate_token_layers = self
                .compression_candidate_token_layers
                .saturating_add(compression.candidate_token_layers as u64);
            self.compression_hot_token_layers = self
                .compression_hot_token_layers
                .saturating_add(compression.hot_token_layers as u64);
            self.compression_full_precision_bytes = self
                .compression_full_precision_bytes
                .saturating_add(compression.full_precision_bytes);
            self.compression_estimated_compressed_bytes = self
                .compression_estimated_compressed_bytes
                .saturating_add(compression.estimated_compressed_bytes);
            self.compression_estimated_saved_bytes = self
                .compression_estimated_saved_bytes
                .saturating_add(compression.estimated_saved_bytes);
            self.compression_ratio_milli = if self.compression_full_precision_bytes == 0 {
                0
            } else {
                self.compression_estimated_compressed_bytes
                    .saturating_mul(1000)
                    .saturating_div(self.compression_full_precision_bytes)
                    .min(u32::MAX as u64) as u32
            };

            let fused_decode_candidate = compression.preset_code == 1
                && compression.key_bits == 8
                && compression.value_bits == 4
                && compression.candidate_token_layers > 0
                && compression.runtime_storage_written_slots > 0;
            if fused_decode_candidate {
                self.compression_fused_decode_candidates =
                    self.compression_fused_decode_candidates.saturating_add(1);
                self.compression_fused_decode_ready_candidates = self
                    .compression_fused_decode_ready_candidates
                    .saturating_add(compression.fused_decode_ready_candidates);
                self.compression_fused_decode_blocked_prefill_only = self
                    .compression_fused_decode_blocked_prefill_only
                    .saturating_add(compression.fused_decode_blocked_prefill_only);
                self.compression_fused_decode_blocked_attention_kind = self
                    .compression_fused_decode_blocked_attention_kind
                    .saturating_add(compression.fused_decode_blocked_attention_kind);
                self.compression_fused_decode_blocked_ineligible_layer = self
                    .compression_fused_decode_blocked_ineligible_layer
                    .saturating_add(compression.fused_decode_blocked_ineligible_layer);
                self.compression_fused_decode_blocked_unsupported_preset = self
                    .compression_fused_decode_blocked_unsupported_preset
                    .saturating_add(compression.fused_decode_blocked_unsupported_preset);
                self.compression_fused_decode_blocked_unsupported_head_dim = self
                    .compression_fused_decode_blocked_unsupported_head_dim
                    .saturating_add(compression.fused_decode_blocked_unsupported_head_dim);
                self.compression_fused_decode_blocked_gqa = self
                    .compression_fused_decode_blocked_gqa
                    .saturating_add(compression.fused_decode_blocked_gqa);
                self.compression_fused_decode_blocked_missing_storage = self
                    .compression_fused_decode_blocked_missing_storage
                    .saturating_add(compression.fused_decode_blocked_missing_storage);
                if compression.fused_decode_attempts > 0 {
                    self.compression_fused_decode_attempts = self
                        .compression_fused_decode_attempts
                        .saturating_add(compression.fused_decode_attempts);
                    self.compression_fused_decode_successes = self
                        .compression_fused_decode_successes
                        .saturating_add(compression.fused_decode_successes);
                    self.compression_fused_decode_fallbacks = self
                        .compression_fused_decode_fallbacks
                        .saturating_add(compression.fused_decode_fallbacks);
                    if compression.fused_decode_metal_successes > 0 {
                        self.compression_decode_path =
                            KV_COMPRESSION_DECODE_PATH_FUSED_COMPRESSED_DECODE;
                    } else if compression.fused_decode_successes > 0 {
                        self.compression_decode_path =
                            KV_COMPRESSION_DECODE_PATH_CPU_ORACLE_COMPRESSED_DECODE;
                    }
                    self.compression_fused_decode_fallback_reason =
                        if compression.fused_decode_fallbacks > 0 {
                            KV_COMPRESSION_FUSED_DECODE_FALLBACK_CPU_ORACLE_UNAVAILABLE
                        } else {
                            KV_COMPRESSION_FUSED_DECODE_FALLBACK_NONE
                        };
                } else if compression.fused_decode_requested {
                    self.compression_fused_decode_fallback_reason =
                        KV_COMPRESSION_FUSED_DECODE_FALLBACK_RUNNER_NOT_INTEGRATED;
                } else {
                    self.compression_fused_decode_fallback_reason =
                        KV_COMPRESSION_FUSED_DECODE_FALLBACK_SHADOW_ONLY;
                }
            } else if compression.preset_code != 1
                || compression.key_bits != 8
                || compression.value_bits != 4
            {
                self.compression_fused_decode_fallback_reason =
                    KV_COMPRESSION_FUSED_DECODE_FALLBACK_UNSUPPORTED_PRESET;
            } else if compression.candidate_token_layers > 0 {
                self.compression_fused_decode_fallback_reason =
                    KV_COMPRESSION_FUSED_DECODE_FALLBACK_MISSING_RUNTIME_STORAGE;
            } else {
                self.compression_fused_decode_fallback_reason =
                    KV_COMPRESSION_FUSED_DECODE_FALLBACK_NONE;
            }

            if compression.fused_decode_requested
                && !fused_decode_candidate
                && self.compression_fused_decode_fallback_reason
                    != KV_COMPRESSION_FUSED_DECODE_FALLBACK_NONE
            {
                self.compression_fused_decode_fallbacks =
                    self.compression_fused_decode_fallbacks.saturating_add(1);
            }
        }
    }

    fn append_route_decisions(&self, decisions: &mut Vec<(String, u32)>) {
        if self.request_snapshots == 0 {
            return;
        }

        let entries = [
            (
                ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS,
                self.request_snapshots,
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS,
                saturating_u32_from_u64(self.logical_tokens),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS,
                saturating_u32_from_u64(self.capacity_tokens),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB,
                kib_ceil(self.logical_bytes),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB,
                kib_ceil(self.capacity_bytes),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS,
                saturating_u32_from_u64(self.full_attention_layers),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS,
                saturating_u32_from_u64(self.sliding_window_layers),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS,
                saturating_u32_from_u64(self.sliding_window_retained_tokens),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS,
                saturating_u32_from_u64(self.sliding_window_reclaimable_capacity_tokens),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB,
                kib_ceil(self.sliding_window_reclaimable_capacity_bytes),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS,
                saturating_u32_from_u64(self.linear_state_layers),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB,
                kib_ceil(self.linear_state_bytes),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT,
                saturating_u32_from_u64(self.growth_count),
            ),
        ];

        for (key, value) in entries {
            upsert_route_decision(decisions, key, value);
        }

        if self.compression_request_snapshots > 0 {
            let compression_entries = [
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_REQUEST_SNAPSHOTS,
                    self.compression_request_snapshots,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_STATUS,
                    self.compression_status,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRESET,
                    self.compression_preset,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEY_BITS,
                    self.compression_key_bits,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_VALUE_BITS,
                    self.compression_value_bits,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ELIGIBLE_LAYERS,
                    saturating_u32_from_u64(self.compression_eligible_layers),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_CANDIDATE_TOKEN_LAYERS,
                    saturating_u32_from_u64(self.compression_candidate_token_layers),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_HOT_TOKEN_LAYERS,
                    saturating_u32_from_u64(self.compression_hot_token_layers),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FULL_PRECISION_KIB,
                    kib_ceil(self.compression_full_precision_bytes),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_COMPRESSED_KIB,
                    kib_ceil(self.compression_estimated_compressed_bytes),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_SAVED_KIB,
                    kib_ceil(self.compression_estimated_saved_bytes),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RATIO_MILLI,
                    self.compression_ratio_milli,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ROUTE_METADATA_SCHEMA,
                    self.compression_route_metadata_schema,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_READY,
                    self.compression_production_ready,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_BLOCKERS,
                    self.compression_production_blockers,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_LAYERS,
                    saturating_u32_from_u64(self.compression_runtime_storage_layers),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_TOKEN_LAYERS,
                    saturating_u32_from_u64(self.compression_runtime_storage_token_layers),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_KIB,
                    kib_ceil(self.compression_runtime_storage_bytes),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_WRITTEN_SLOTS,
                    saturating_u32_from_u64(self.compression_runtime_storage_written_slots),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_CALLS,
                    saturating_u32_from_u64(self.compression_shadow_sync_calls),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_WALL_US,
                    saturating_u32_from_u64(self.compression_shadow_sync_wall_us),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH,
                    self.compression_decode_path,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES,
                    saturating_u32_from_u64(self.compression_fused_decode_candidates),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS,
                    saturating_u32_from_u64(self.compression_fused_decode_attempts),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES,
                    saturating_u32_from_u64(self.compression_fused_decode_successes),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS,
                    saturating_u32_from_u64(self.compression_fused_decode_fallbacks),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON,
                    self.compression_fused_decode_fallback_reason,
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_READY_CANDIDATES,
                    saturating_u32_from_u64(self.compression_fused_decode_ready_candidates),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_PREFILL_ONLY,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_prefill_only),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_ATTENTION_KIND,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_attention_kind),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_INELIGIBLE_LAYER,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_ineligible_layer),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_PRESET,
                    saturating_u32_from_u64(
                        self.compression_fused_decode_blocked_unsupported_preset,
                    ),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_HEAD_DIM,
                    saturating_u32_from_u64(
                        self.compression_fused_decode_blocked_unsupported_head_dim,
                    ),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_GQA,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_gqa),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_MISSING_STORAGE,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_missing_storage),
                ),
            ];

            for (key, value) in compression_entries {
                upsert_route_decision(decisions, key, value);
            }
        }
    }
}

/// Per-request mutable state persisted across prefill → decode steps.
struct RequestState {
    cache: MlxKVCache,
    ngram: NgramTable,
    /// Per-request PRNG for temperature sampling.  Seeded from request_id so
    /// deterministic seeds produce reproducible outputs.
    rng: Xorshift64,
    /// Beta-Bernoulli posterior α for the n-gram acceleration accept-rate gate.
    /// Incremented by accepted draft tokens each n-gram acceleration step.
    ngram_beta_alpha: f32,
    /// Beta-Bernoulli posterior β for the n-gram acceleration accept-rate gate.
    /// Incremented by rejected draft tokens each n-gram acceleration step.
    ngram_beta_beta: f32,
    /// Steps remaining before re-enabling ngram_acceleration (0 = n-gram acceleration allowed).
    ngram_disabled_steps: u32,
    /// Consecutive decode steps where a linear-attention request had no viable
    /// n-gram draft.  Used to avoid spending an entire request in single-decode
    /// fallback when acceleration has no evidence to act on.
    linear_ngram_no_draft_streak: u32,
    /// Request-local fallback: once a linear-attention request proves it has no
    /// useful n-gram support, finish it on the direct pipeline.
    ngram_acceleration_disabled_for_request: bool,
    /// Pre-verified bonus tokens ready to serve without a model run.
    bonus_queue: VecDeque<u32>,
    /// The token to use as `last_token` for the next model run.
    /// None on the very first decode step (use framework-supplied input instead).
    next_model_last_token: Option<u32>,
    /// Lazy token from the previous direct decode step (double-buffer pipeline).
    ///
    /// When `Some`, the next call to `decode_one` uses `advance_direct_pipeline`
    /// to materialise this token while simultaneously submitting the next step
    /// to the GPU — eliminating the GPU idle gap between steps.
    ///
    /// Only set when `disable_ngram_acceleration = true` and `temperature == 0.0`.
    pending_direct: Option<MlxArray>,
    /// Direct-pipeline tokens emitted since the current generation started.
    direct_pipeline_emitted_tokens: u32,
    /// Cumulative per-request counters surfaced through route metadata for
    /// benchmark auditability.
    ngram_acceleration: NgramAccelerationTelemetry,
    decode_telemetry: DecodeTelemetry,
}

impl RequestState {
    fn new(num_layers: usize, request_id: RequestId) -> Self {
        Self {
            cache: MlxKVCache::new(num_layers),
            ngram: NgramTable::new(),
            rng: Xorshift64::new(request_id.0),
            ngram_beta_alpha: NGRAM_BETA_PRIOR_ALPHA,
            ngram_beta_beta: NGRAM_BETA_PRIOR_BETA,
            ngram_disabled_steps: 0,
            linear_ngram_no_draft_streak: 0,
            ngram_acceleration_disabled_for_request: false,
            bonus_queue: VecDeque::new(),
            next_model_last_token: None,
            pending_direct: None,
            direct_pipeline_emitted_tokens: 0,
            ngram_acceleration: NgramAccelerationTelemetry::default(),
            decode_telemetry: DecodeTelemetry::default(),
        }
    }

    fn ngram_posterior_mean(&self) -> f32 {
        self.ngram_beta_alpha / (self.ngram_beta_alpha + self.ngram_beta_beta)
    }
}

/// ExecutionRunner backed by the MLX inference path.
pub struct MlxRunner {
    cfg: ModelConfig,
    weights: Arc<ModelWeights>,
    prefill_chunk: usize,
    kv_layer_windows: Vec<Option<usize>>,
    binding_summary: NativeModelBindingSummary,
    terminal_token_ids: Vec<u32>,
    states: Mutex<HashMap<RequestId, RequestState>>,
    /// Dedicated GPU stream kept alive for the runner's lifetime.
    _stream: MlxStream,
    /// When true, disable n-gram acceleration and use the direct decode path.
    disable_ngram_acceleration: bool,
    /// Optional KV compression policy. Disabled by default and never changes logits in shadow mode.
    kv_compression: MlxKvCompressionConfig,
    /// Per-layer compression eligibility. Empty when compression is disabled.
    kv_compression_layer_eligible: Vec<bool>,
    /// Experimental Gemma sliding-window rotating backing store for direct decode.
    rotating_sliding_decode: bool,
    /// Optional mlx_lm-style `clear_cache` cadence for the direct decode pipeline.
    direct_clear_cache_cadence: u32,
}

impl fmt::Debug for MlxRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxRunner")
            .field("layers", &self.cfg.layer_count)
            .field("vocab", &self.cfg.vocab_size)
            .finish()
    }
}

impl MlxRunner {
    pub fn from_artifacts(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        kv_compression: MlxKvCompressionConfig,
    ) -> Result<Self, MlxRunnerError> {
        // Enable MLX compute-graph compilation globally.
        // This caches and reuses compiled Metal shaders across calls with the same
        // graph structure — the equivalent of mlx_lm's per-step mx.compile() JIT.
        // Without this, MLX rebuilds the dispatch graph on every decode step,
        // causing measurable CPU overhead (~10-15% throughput gap vs mlx_lm).
        enable_compile();

        // Dedicated GPU stream — mirrors mlx_lm's `mx.new_stream(mx.default_device())`.
        // Setting it as default avoids implicit cross-stream synchronization on the
        // shared default stream.
        let stream = MlxStream::new_gpu();
        stream.set_as_default();

        // Wire weights into GPU memory to prevent paging between requests.
        // Use Metal's recommendedMaxWorkingSetSize — values above this are rejected.
        let wired_cap = max_recommended_working_set_size();
        if wired_cap > 0 {
            set_wired_limit(wired_cap);
        }

        validate_mlx_supported_manifest(artifacts)?;

        let cfg = ModelConfig::from_manifest(artifacts.manifest());
        let terminal_token_ids = resolve_terminal_token_ids(artifacts);
        let kv_layer_windows = kv_layer_windows_from_config(&cfg);
        let rotating_sliding_decode = disable_ngram_acceleration
            && std::env::var("AX_MLX_ROTATING_SLIDING_DECODE").as_deref() == Ok("1");
        let direct_clear_cache_cadence = std::env::var("AX_MLX_DIRECT_CLEAR_CACHE_CADENCE")
            .ok()
            .and_then(|raw| raw.parse::<u32>().ok())
            .unwrap_or(0);
        let kv_compression_layer_eligible = if kv_compression.is_enabled() {
            turboquant_support_report(&cfg, kv_compression.preset)
                .map_err(|error| {
                    MlxRunnerError::UnsupportedFeature(format!(
                        "invalid TurboQuant compression policy: {error}"
                    ))
                })?
                .eligible_layer_mask()
        } else {
            Vec::new()
        };
        let weights = load_weights(artifacts).map_err(MlxRunnerError::Weights)?;

        let binding_summary = binding_summary_from_specs(artifacts.tensor_specs());

        let weights = Arc::new(weights);

        // JIT warm-up: trigger Metal shader compilation for both decode and prefill paths.
        {
            let mut dummy_cache = MlxKVCache::new(cfg.layer_count);
            let mut dummy_rng = Xorshift64::new(0);
            decode_step(&cfg, &weights, 0, &mut dummy_cache, 0.0, &mut dummy_rng);
            dummy_cache.reset();
            let dummy_tokens: Vec<u32> = vec![0u32; 8];
            chunked_prefill(
                &cfg,
                &weights,
                &dummy_tokens,
                &mut dummy_cache,
                prefill_chunk,
                0.0,
                &mut dummy_rng,
            );
        }
        let _ = take_gemma4_moe_profile_snapshot();
        let _ = take_linear_attention_profile_snapshot();

        // Qwen3.5 linear-attention uses `ngram_accel_decode_step_linear_safe` which
        // clones the cache for verification and recomputes the committed prefix on
        // partial accept, so n-gram acceleration is safe to enable for these models.
        Ok(Self {
            cfg,
            weights,
            prefill_chunk: prefill_chunk.max(1),
            kv_layer_windows,
            binding_summary,
            terminal_token_ids,
            states: Mutex::new(HashMap::new()),
            _stream: stream,
            disable_ngram_acceleration,
            kv_compression,
            kv_compression_layer_eligible,
            rotating_sliding_decode,
            direct_clear_cache_cadence,
        })
    }

    fn turboquant_model_decode_context(&self) -> Option<TurboQuantModelDecodeContext<'_>> {
        self.kv_compression
            .requests_fused_decode()
            .then_some(TurboQuantModelDecodeContext {
                config: self.kv_compression,
                layer_eligible: &self.kv_compression_layer_eligible,
            })
    }
}

impl ExecutionRunner for MlxRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let step_id = input.execution_batch.step_id;
        let mut request_updates = Vec::new();
        let logits_handles = Vec::new();
        let logits_outputs = Vec::new();

        let mut route_metadata = input.execution_batch.route_metadata.clone();
        let mut ngram_acceleration = NgramAccelerationTelemetry::default();
        let mut decode_telemetry = DecodeTelemetry::default();
        let mut gemma4_moe_profile = Gemma4MoeProfileSnapshot::default();
        let mut linear_attention_profile = LinearAttentionProfileSnapshot::default();
        let mut kv_cache = KvCacheTelemetry::default();

        for item in &input.execution_batch.items {
            let ctx = input
                .request_contexts
                .iter()
                .find(|c| c.request_id == item.request_id);

            let result = self.run_item(item, ctx);
            ngram_acceleration.merge_from(result.ngram_acceleration);
            decode_telemetry.merge_from(result.decode_telemetry);
            gemma4_moe_profile.merge_from(result.gemma4_moe_profile);
            linear_attention_profile.merge_from(result.linear_attention_profile);
            kv_cache.merge_from(result.kv_usage);
            if let Some(wall_us) = result.kv_compression_shadow_sync_wall_us {
                kv_cache.record_compression_shadow_sync(wall_us);
            }
            request_updates.push(result.update);
        }
        ngram_acceleration.append_route_decisions(&mut route_metadata.crossover_decisions);
        decode_telemetry.append_route_decisions(&mut route_metadata.crossover_decisions);
        gemma4_moe_profile.append_route_decisions(&mut route_metadata.crossover_decisions);
        linear_attention_profile.append_route_decisions(&mut route_metadata.crossover_decisions);
        kv_cache.append_route_decisions(&mut route_metadata.crossover_decisions);

        let tokens_written: u32 = input
            .execution_batch
            .items
            .iter()
            .map(|i| i.scheduled_token_count)
            .sum();

        RunnerOutput {
            step_id,
            request_updates,
            logits_handles,
            logits_outputs,
            kv_write_summary: KvWriteSummary {
                tokens_written,
                blocks_touched: 0,
            },
            route_metadata,
            execution_status: ExecutionStatus::Success,
        }
    }

    fn native_model_binding_summary(&self) -> Option<NativeModelBindingSummary> {
        Some(self.binding_summary)
    }

    fn embed(
        &self,
        token_ids: &[u32],
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<Vec<f32>, &'static str> {
        if token_ids.is_empty() {
            return Err("token_ids must not be empty");
        }
        let hidden = crate::model::forward_for_embedding(&self.cfg, &self.weights, token_ids);
        // hidden: [1, seq, hidden_size] bfloat16
        let seq = token_ids.len() as i32;
        let pooled = match pooling {
            EmbeddingPooling::Mean => {
                let summed = sum_axis(&hidden, 1, false, None);
                // summed: [1, hidden_size] bfloat16
                let scale = 1.0_f32 / seq as f32;
                let scale_arr = MlxArray::from_raw_data(
                    &scale as *const f32 as *const u8,
                    std::mem::size_of::<f32>(),
                    &[],
                    MlxDtype::Float32,
                );
                multiply(&summed, &scale_arr, None)
            }
            EmbeddingPooling::Last => {
                let last_idx = (seq - 1) as u32;
                let idx_arr = MlxArray::from_raw_data(
                    &last_idx as *const u32 as *const u8,
                    std::mem::size_of::<u32>(),
                    &[1_i32],
                    MlxDtype::Uint32,
                );
                // take on axis 1: [1, seq, hidden] → [1, 1, hidden]
                take(&hidden, &idx_arr, 1, None)
            }
            EmbeddingPooling::Cls => {
                let first_idx: u32 = 0;
                let idx_arr = MlxArray::from_raw_data(
                    &first_idx as *const u32 as *const u8,
                    std::mem::size_of::<u32>(),
                    &[1_i32],
                    MlxDtype::Uint32,
                );
                // take on axis 1: [1, seq, hidden] → [1, 1, hidden]
                take(&hidden, &idx_arr, 1, None)
            }
        };
        // Convert to float32 before normalization for numerical precision.
        // pooled: [1, hidden_size] (Mean) or [1, 1, hidden_size] (Last/Cls)
        let pooled_f32 = astype(&pooled, MlxDtype::Float32, None);
        let result = if normalize {
            l2_normalize_last_dim(&pooled_f32)
        } else {
            pooled_f32
        };
        mlx_sys::eval(&[&result]);
        Ok(result.data_f32().to_vec())
    }
}

/// L2-normalize `x` along its last dimension.
///
/// Works on any shape `[..., d]` — broadcasts the norm back for division.
/// Adds a small epsilon (1e-12) for numerical stability on near-zero vectors.
fn l2_normalize_last_dim(x: &MlxArray) -> MlxArray {
    let ndim = x.shape().len() as i32;
    let last_axis = ndim - 1;
    let x_sq = multiply(x, x, None);
    let sum_sq = sum_axis(&x_sq, last_axis, true, None);
    let half = MlxArray::from_raw_data(
        &0.5_f32 as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[],
        MlxDtype::Float32,
    );
    let norm = power(&sum_sq, &half, None);
    let eps_val = 1e-12_f32;
    let eps = MlxArray::from_raw_data(
        &eps_val as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[],
        MlxDtype::Float32,
    );
    let norm_stable = add(&norm, &eps, None);
    divide(x, &norm_stable, None)
}

impl MlxRunner {
    fn run_item(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
    ) -> MlxItemRun {
        let token_ids = &item.input_token_slice;
        if token_ids.is_empty() {
            return MlxItemRun {
                update: RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: 0,
                    output_token: None,
                    stop_reason: None,
                    error: Some("empty token slice".into()),
                },
                ngram_acceleration: NgramAccelerationTelemetry::default(),
                decode_telemetry: DecodeTelemetry::default(),
                gemma4_moe_profile: Gemma4MoeProfileSnapshot::default(),
                linear_attention_profile: LinearAttentionProfileSnapshot::default(),
                kv_usage: MlxKVCacheUsage::default(),
                kv_compression_shadow_sync_wall_us: None,
            };
        }

        let max_output = ctx.map(|c| c.max_output_tokens).unwrap_or(1);
        let generated_len = ctx.map(|c| c.generated_len).unwrap_or(0);
        let prefill_completes_prompt = prefill_item_completes_prompt(item, ctx);
        let is_prefill = matches!(item.mode, ExecutionMode::Prefill);

        // Extract per-request state from the map and release the lock before GPU
        // work.  This ensures a long prefill for one request does not block state
        // access for any other request: the mutex is held only for the O(1)
        // HashMap remove and subsequent insert, never across a GPU forward pass.
        //
        // Concurrency contract: the scheduler must not route the same request_id
        // to two concurrent run() calls — otherwise one call would create a fresh
        // empty state from None while the other holds the extracted state.
        let mut state = {
            let mut states = self.states.lock().unwrap();
            states
                .remove(&item.request_id)
                .unwrap_or_else(|| RequestState::new(self.cfg.layer_count, item.request_id))
        };

        let temperature = ctx.map(|c| c.temperature).unwrap_or(0.0);
        let is_greedy = temperature == 0.0;
        let kv_compression_layer_eligible = if self.kv_compression.is_enabled() {
            Some(self.kv_compression_layer_eligible.as_slice())
        } else {
            None
        };
        state
            .cache
            .set_rotating_sliding_decode(self.rotating_sliding_decode && is_greedy);
        let mut kv_compression_shadow_sync_wall_us = None;

        // GPU work — mutex is NOT held during prefill, decode, or n-gram acceleration steps.
        let sampled_token = match item.mode {
            ExecutionMode::Prefill => {
                let prefill_started = Instant::now();
                let tok = chunked_prefill(
                    &self.cfg,
                    &self.weights,
                    token_ids,
                    &mut state.cache,
                    self.prefill_chunk,
                    temperature,
                    &mut state.rng,
                );
                if prefill_completes_prompt {
                    kv_compression_shadow_sync_wall_us = self.initialize_generation_state(
                        &mut state,
                        token_ids,
                        tok,
                        max_output,
                        is_greedy,
                        kv_compression_layer_eligible,
                    );
                }

                state
                    .decode_telemetry
                    .record_prefill(elapsed_us(prefill_started));
                prefill_completes_prompt.then_some(tok)
            }
            ExecutionMode::Decode => {
                let decode_started = Instant::now();
                let tok = self.decode_one(&mut state, token_ids, temperature, is_greedy);
                state
                    .decode_telemetry
                    .record_decode(elapsed_us(decode_started));
                Some(tok)
            }
        };

        let stop_reason = sampled_token.and_then(|sampled_token| {
            stop_reason_for_sampled_token(
                sampled_token,
                generated_len,
                max_output,
                &self.terminal_token_ids,
            )
        });

        // Re-insert state only if the request continues — lock held briefly.
        let ngram_acceleration = state.ngram_acceleration;
        let decode_telemetry = state.decode_telemetry;
        let gemma4_moe_profile = take_gemma4_moe_profile_snapshot();
        let linear_attention_profile = take_linear_attention_profile_snapshot();
        {
            let force_shadow_sync = is_prefill
                && prefill_completes_prompt
                && kv_compression_shadow_sync_wall_us.is_none();
            let sync_wall_us = self.sync_turboquant_shadow_storage_if_needed(
                &mut state,
                force_shadow_sync,
                kv_compression_layer_eligible,
            );
            if sync_wall_us.is_some() {
                kv_compression_shadow_sync_wall_us = sync_wall_us;
            }
        }
        let mut kv_usage = state
            .cache
            .usage_snapshot_with_layer_windows_compression_and_layer_eligibility(
                &self.kv_layer_windows,
                self.kv_compression,
                kv_compression_layer_eligible,
            );
        let turboquant_decode_usage = state.cache.take_turboquant_decode_usage();
        kv_usage.kv_compression.fused_decode_attempts =
            turboquant_decode_usage.fused_decode_attempts;
        kv_usage.kv_compression.fused_decode_successes =
            turboquant_decode_usage.fused_decode_successes;
        kv_usage.kv_compression.fused_decode_metal_successes =
            turboquant_decode_usage.fused_decode_metal_successes;
        kv_usage.kv_compression.fused_decode_fallbacks =
            turboquant_decode_usage.fused_decode_fallbacks;
        kv_usage.kv_compression.fused_decode_ready_candidates =
            turboquant_decode_usage.fused_decode_ready_candidates;
        kv_usage.kv_compression.fused_decode_blocked_prefill_only =
            turboquant_decode_usage.fused_decode_blocked_prefill_only;
        kv_usage.kv_compression.fused_decode_blocked_attention_kind =
            turboquant_decode_usage.fused_decode_blocked_attention_kind;
        kv_usage
            .kv_compression
            .fused_decode_blocked_ineligible_layer =
            turboquant_decode_usage.fused_decode_blocked_ineligible_layer;
        kv_usage
            .kv_compression
            .fused_decode_blocked_unsupported_preset =
            turboquant_decode_usage.fused_decode_blocked_unsupported_preset;
        kv_usage
            .kv_compression
            .fused_decode_blocked_unsupported_head_dim =
            turboquant_decode_usage.fused_decode_blocked_unsupported_head_dim;
        kv_usage.kv_compression.fused_decode_blocked_gqa =
            turboquant_decode_usage.fused_decode_blocked_gqa;
        kv_usage.kv_compression.fused_decode_blocked_missing_storage =
            turboquant_decode_usage.fused_decode_blocked_missing_storage;
        if stop_reason.is_none() {
            let mut states = self.states.lock().unwrap();
            states.insert(item.request_id, state);
        } else {
            // Free MLX's intermediate graph and compute cache after each completed
            // request.  Mirrors mlx_lm's mx.metal.clear_cache() at end of generation;
            // reclaims GPU memory that would otherwise persist until the next request.
            clear_cache();
        }

        MlxItemRun {
            update: RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: item.scheduled_token_count,
                output_token: sampled_token,
                stop_reason,
                error: None,
            },
            ngram_acceleration,
            decode_telemetry,
            gemma4_moe_profile,
            linear_attention_profile,
            kv_usage,
            kv_compression_shadow_sync_wall_us,
        }
    }

    /// Produce one output token for a decode step.
    ///
    /// Pops from the bonus queue when pre-verified tokens are available.
    /// Uses the double-buffer direct pipeline when `disable_ngram_acceleration = true` and
    /// `temperature == 0.0` (bootstrapped during prefill).
    /// Otherwise runs an n-gram accelerated or single-token decode pass.
    fn decode_one(
        &self,
        state: &mut RequestState,
        input_tokens: &[u32],
        temperature: f32,
        is_greedy: bool,
    ) -> u32 {
        // Serve pre-verified bonus tokens without re-running the model.
        // (Bonus tokens only exist on the n-gram acceleration path; the direct pipeline
        // never populates the bonus queue.)
        if let Some(tok) = state.bonus_queue.pop_front() {
            state.decode_telemetry.record_bonus_token();
            return tok;
        }

        // Double-buffer direct pipeline: materialise the pending lazy token while
        // simultaneously submitting the next step to the GPU.  This mirrors
        // mlx_lm's `_step(y)` → `async_eval(next_y)` → `eval(y)` loop and
        // eliminates the GPU idle gap between consecutive direct decode steps.
        if self.disable_ngram_acceleration && is_greedy && state.pending_direct.is_some() {
            return self.run_direct_pipeline_continue(state);
        }

        let last_token = state
            .next_model_last_token
            .or_else(|| input_tokens.last().copied())
            .unwrap_or(0);

        let result = self.run_model_decode(state, last_token, temperature, is_greedy);
        apply_decode_result(state, &result, &self.terminal_token_ids)
    }

    /// Decode one deterministic token on the direct double-buffer pipeline.
    ///
    /// Used both by explicit direct mode and by request-local n-gram fallback after
    /// a linear-attention request proves it has no useful draft support.  The
    /// pipeline may keep the cache one lazy token ahead, so callers must continue
    /// using this path until the request finishes.
    fn run_direct_pipeline_decode(&self, state: &mut RequestState, last_token: u32) -> u32 {
        let tok = self.run_direct_pipeline_bootstrap(state, last_token);
        state.ngram.feed(&[tok]);
        tok
    }

    fn run_direct_pipeline_continue(&self, state: &mut RequestState) -> u32 {
        let bootstrap_token = state
            .pending_direct
            .take()
            .expect("direct pipeline continue requires pending_direct to be initialized");
        self.run_direct_pipeline_once(state, bootstrap_token)
    }

    fn run_direct_pipeline_bootstrap(&self, state: &mut RequestState, last_token: u32) -> u32 {
        let turboquant_context = self.turboquant_model_decode_context();
        let bootstrap_token = start_direct_pipeline_with_turboquant_context(
            &self.cfg,
            &self.weights,
            last_token,
            &mut state.cache,
            turboquant_context.as_ref(),
        );
        self.run_direct_pipeline_once(state, bootstrap_token)
    }

    fn run_direct_pipeline_once(&self, state: &mut RequestState, bootstrap_token: MlxArray) -> u32 {
        let branch_started = Instant::now();
        let turboquant_context = self.turboquant_model_decode_context();
        let (tok, next_pending) = advance_direct_pipeline_with_turboquant_context(
            &self.cfg,
            &self.weights,
            &bootstrap_token,
            &mut state.cache,
            turboquant_context.as_ref(),
        );
        state
            .decode_telemetry
            .record_direct_pipeline(elapsed_us(branch_started));
        state.pending_direct = Some(next_pending);
        self.maybe_clear_direct_pipeline_cache(state);
        tok
    }

    fn run_request_disabled_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        temperature: f32,
        is_greedy: bool,
    ) -> Vec<u32> {
        state.ngram_acceleration.record_request_disabled_step();
        if is_greedy {
            vec![self.run_direct_pipeline_decode(state, last_token)]
        } else {
            self.run_single_decode(state, last_token, temperature)
        }
    }

    fn run_no_draft_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        temperature: f32,
        has_linear_attention: bool,
        is_greedy: bool,
    ) -> Option<Vec<u32>> {
        state.ngram_acceleration.record_no_draft();
        if has_linear_attention {
            state.linear_ngram_no_draft_streak =
                state.linear_ngram_no_draft_streak.saturating_add(1);
            if is_greedy && linear_ngram_no_draft_should_disable(state.linear_ngram_no_draft_streak)
            {
                state.ngram_acceleration_disabled_for_request = true;
                state.ngram_acceleration.record_request_disable_event();
                return Some(self.run_request_disabled_decode(
                    state,
                    last_token,
                    temperature,
                    is_greedy,
                ));
            }
        }
        None
    }

    fn run_non_ngram_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        temperature: f32,
        is_greedy: bool,
    ) -> Option<Vec<u32>> {
        if self.disable_ngram_acceleration {
            return Some(self.run_single_decode(state, last_token, temperature));
        }

        if state.ngram_acceleration_disabled_for_request {
            return Some(self.run_request_disabled_decode(
                state,
                last_token,
                temperature,
                is_greedy,
            ));
        }

        // N-gram acceleration disabled: count down and use single decode.
        if state.ngram_disabled_steps > 0 {
            state.ngram_disabled_steps -= 1;
            state.ngram_acceleration.record_cooldown_step();
            return Some(self.run_single_decode(state, last_token, temperature));
        }

        None
    }

    fn maybe_clear_direct_pipeline_cache(&self, state: &mut RequestState) {
        state.direct_pipeline_emitted_tokens =
            state.direct_pipeline_emitted_tokens.saturating_add(1);
        if direct_pipeline_clear_cache_due(
            state.direct_pipeline_emitted_tokens,
            self.direct_clear_cache_cadence,
        ) {
            clear_cache();
        }
    }

    fn sync_turboquant_shadow_storage_if_needed(
        &self,
        state: &mut RequestState,
        force: bool,
        layer_eligible: Option<&[bool]>,
    ) -> Option<u32> {
        if !self.kv_compression.is_enabled() {
            return None;
        }

        let should_sync = force
            || state.cache.turboquant_shadow_storage_sync_due(
                &self.kv_layer_windows,
                self.kv_compression,
                layer_eligible,
            );

        if should_sync {
            let sync_started = Instant::now();
            state.cache.sync_turboquant_shadow_storage(
                &self.kv_layer_windows,
                self.kv_compression,
                layer_eligible,
            );
            Some(elapsed_us(sync_started))
        } else {
            None
        }
    }

    fn run_single_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let branch_started = Instant::now();
        let turboquant_context = self.turboquant_model_decode_context();
        let result = single_decode_with_turboquant_context(
            &self.cfg,
            &self.weights,
            &mut state.cache,
            &mut state.ngram,
            last_token,
            temperature,
            &mut state.rng,
            turboquant_context.as_ref(),
        );
        state
            .decode_telemetry
            .record_single_decode(elapsed_us(branch_started));
        result
    }

    fn initialize_generation_state(
        &self,
        state: &mut RequestState,
        token_ids: &[u32],
        bootstrap_token: u32,
        max_output: u32,
        is_greedy: bool,
        layer_eligible: Option<&[bool]>,
    ) -> Option<u32> {
        // Seed the n-gram table with the tail of the prompt.
        // Only the last NGRAM_PROMPT_FEED_MAX tokens are fed: long prompts
        // (e.g. random-token benchmarks with 512+ tokens) would otherwise inject
        // hundreds of useless bigrams, causing a false-positive spec attempt on
        // the first decode step and disabling n-gram acceleration for
        // LINEAR_NGRAM_RETRY_INTERVAL steps — wiping out most of the generation.
        let feed_start = token_ids.len().saturating_sub(NGRAM_PROMPT_FEED_MAX);
        state.ngram.feed(&token_ids[feed_start..]);

        // Reset per-generation state.
        state.bonus_queue.clear();
        state.next_model_last_token = None;
        state.pending_direct = None;
        state.direct_pipeline_emitted_tokens = 0;
        state.ngram_disabled_steps = 0;
        state.linear_ngram_no_draft_streak = 0;
        // Skip n-gram entirely for short output budgets: failed speculation
        // attempts and cooldown intervals (8-16 steps) are a net loss when
        // max_output_tokens is smaller than two full retry windows.
        state.ngram_acceleration_disabled_for_request =
            max_output < NGRAM_MIN_OUTPUT_FOR_ACCELERATION;

        let kv_compression_shadow_sync_wall_us =
            self.sync_turboquant_shadow_storage_if_needed(state, true, layer_eligible);

        // Bootstrap the double-buffer direct pipeline: submit the second token's
        // forward pass to the GPU asynchronously so the first decode step can
        // materialise it while the GPU is already computing the third token.
        // Only for direct same-policy runs with temperature = 0.
        if self.disable_ngram_acceleration && is_greedy {
            let bootstrap_started = Instant::now();
            let turboquant_context = self.turboquant_model_decode_context();
            state.pending_direct = Some(start_direct_pipeline_with_turboquant_context(
                &self.cfg,
                &self.weights,
                bootstrap_token,
                &mut state.cache,
                turboquant_context.as_ref(),
            ));
            state
                .decode_telemetry
                .record_direct_bootstrap(elapsed_us(bootstrap_started));
        }

        kv_compression_shadow_sync_wall_us
    }

    /// Run one model decode step, updating the n-gram accept-rate gate.
    fn run_model_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        temperature: f32,
        is_greedy: bool,
    ) -> Vec<u32> {
        let has_linear_attention = self.cfg.linear_attention.is_some();
        if let Some(result) = self.run_non_ngram_decode(state, last_token, temperature, is_greedy) {
            return result;
        }

        let draft = ngram_acceleration_draft(&state.ngram, has_linear_attention);
        if draft.is_empty() {
            if let Some(result) = self.run_no_draft_decode(
                state,
                last_token,
                temperature,
                has_linear_attention,
                is_greedy,
            ) {
                return result;
            }
            return self.run_single_decode(state, last_token, temperature);
        }

        state.linear_ngram_no_draft_streak = 0;

        let draft_len = draft.len();
        let branch_started = Instant::now();
        let result = ngram_accel_decode_step(
            &self.cfg,
            &self.weights,
            &mut state.cache,
            &mut state.ngram,
            last_token,
            &draft,
            temperature,
            &mut state.rng,
        );
        state
            .decode_telemetry
            .record_ngram_decode(elapsed_us(branch_started));

        // Beta-Bernoulli posterior update.
        // accept_count = result.len() - 1 (last element is next model input, not bonus).
        let accept_count = result.len().saturating_sub(1);
        state
            .ngram_acceleration
            .record_draft(draft_len, accept_count);
        state.ngram_beta_alpha += accept_count as f32;
        state.ngram_beta_beta += (draft_len - accept_count) as f32;

        // Normalise to keep total observations bounded — prevents the posterior
        // from becoming overconfident and unable to adapt to changing statistics.
        let total = state.ngram_beta_alpha + state.ngram_beta_beta;
        if total > NGRAM_BETA_MAX_TOTAL {
            let scale = NGRAM_BETA_MAX_TOTAL / total;
            state.ngram_beta_alpha *= scale;
            state.ngram_beta_beta *= scale;
        }

        if let Some(disabled_steps) = ngram_acceleration_disabled_steps(
            has_linear_attention,
            accept_count,
            draft_len,
            state.ngram_posterior_mean(),
        ) {
            state.ngram_disabled_steps = disabled_steps;
            state
                .ngram_acceleration
                .record_cooldown_event(disabled_steps);
        }

        result
    }
}

fn apply_decode_result(
    state: &mut RequestState,
    result: &[u32],
    terminal_token_ids: &[u32],
) -> u32 {
    debug_assert!(
        !result.is_empty(),
        "MLX decode path must return at least one token"
    );

    // result[0] is the output for this step.
    // result[1..] are already verified output tokens to emit before the next
    // model run.  The last token also drives that next model run, but it still
    // belongs in the user-visible output stream; dropping it corrupts generation.
    let output = result[0];
    for &token in result.get(1..).unwrap_or(&[]) {
        state.bonus_queue.push_back(token);
        if token_is_terminal(token, terminal_token_ids) {
            break;
        }
    }
    state.next_model_last_token = state
        .bonus_queue
        .back()
        .copied()
        .or_else(|| result.last().copied());
    output
}

fn stop_reason_for_sampled_token(
    sampled_token: u32,
    generated_len: u32,
    max_output: u32,
    terminal_token_ids: &[u32],
) -> Option<StopReason> {
    if token_is_terminal(sampled_token, terminal_token_ids) {
        Some(StopReason::EosToken)
    } else if generated_len.saturating_add(1) >= max_output {
        Some(StopReason::MaxOutputTokens)
    } else {
        None
    }
}

fn token_is_terminal(token: u32, terminal_token_ids: &[u32]) -> bool {
    terminal_token_ids.contains(&token)
}

fn prefill_item_completes_prompt(
    item: &ax_engine_core::ExecutionItem,
    ctx: Option<&RunnerRequestContext>,
) -> bool {
    if item.mode != ExecutionMode::Prefill {
        return false;
    }
    ctx.map(|c| {
        c.processed_prompt_tokens
            .saturating_add(item.scheduled_token_count)
            >= c.prompt_len
    })
    .unwrap_or(true)
}

struct MlxItemRun {
    update: RequestExecutionUpdate,
    ngram_acceleration: NgramAccelerationTelemetry,
    decode_telemetry: DecodeTelemetry,
    gemma4_moe_profile: Gemma4MoeProfileSnapshot,
    linear_attention_profile: LinearAttentionProfileSnapshot,
    kv_usage: MlxKVCacheUsage,
    kv_compression_shadow_sync_wall_us: Option<u32>,
}

fn ngram_acceleration_disabled_steps(
    has_linear_attention: bool,
    accept_count: usize,
    draft_len: usize,
    posterior_mean: f32,
) -> Option<u32> {
    if draft_len == 0 {
        return None;
    }

    if has_linear_attention {
        // Linear-attention recurrent state cannot be rolled back with trim_to; any
        // partial reject pays branch verification + committed-prefix recompute.
        // Recompute cost is O(accepted+1) tokens — bounded at DEFAULT_DRAFT_LEN+1,
        // not O(context length) — so a large retry interval is unwarranted.
        //
        // Differentiate complete miss from partial accept: a partial accept means
        // the n-gram was directionally correct; retry quickly.  A complete miss
        // means the table prediction is off; back off longer.
        if accept_count == 0 {
            return Some(LINEAR_NGRAM_RETRY_INTERVAL); // complete miss: 16 steps
        }
        return (accept_count < draft_len).then_some(LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL); // partial: 4 steps
    }

    (posterior_mean < NGRAM_ACCEPT_THRESHOLD).then_some(NGRAM_RETRY_INTERVAL)
}

fn linear_ngram_no_draft_should_disable(streak: u32) -> bool {
    streak >= LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD
}

fn ngram_acceleration_draft(ngram: &NgramTable, has_linear_attention: bool) -> Vec<u32> {
    if has_linear_attention {
        // Dense rollback is O(1); linear-attention partial-reject pays
        // branch/recompute, so cap at DEFAULT_DRAFT_LEN to bound recompute cost.
        ngram.predict_with_confidence(
            DEFAULT_DRAFT_LEN,
            LINEAR_MIN_NGRAM_SUPPORT,
            DRAFT_CONFIDENCE_THRESHOLD,
        )
    } else {
        // Dense models extend up to MAX_DRAFT_LEN when the n-gram chain is
        // high-confidence; the confidence gate stops the chain early otherwise.
        ngram.predict_with_confidence(MAX_DRAFT_LEN, 1, DRAFT_CONFIDENCE_THRESHOLD)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MlxRunnerError {
    #[error("MLX model feature is not supported: {0}")]
    UnsupportedFeature(String),
    #[error("weight loading failed: {0}")]
    Weights(#[from] crate::weights::WeightLoadError),
}

fn validate_mlx_supported_manifest(artifacts: &NativeModelArtifacts) -> Result<(), MlxRunnerError> {
    let manifest = artifacts.manifest();
    if manifest.model_family == "glm4_moe_lite" || has_glm_mla_tensors(artifacts) {
        validate_glm4_moe_lite_manifest(manifest)?;
    }
    if manifest.linear_attention.is_enabled() || has_linear_attention_tensors(artifacts) {
        validate_qwen_gated_delta_linear_attention(manifest)?;
    }
    if manifest.sliding_window_size.is_some()
        || !manifest.layer_types.is_empty()
        || !manifest.kv_shared_source_layers.is_empty()
        || manifest.global_head_dim.is_some()
        || manifest.rope_theta_swa.is_some()
    {
        validate_gemma4_interleaved_attention(manifest)?;
    }
    Ok(())
}

fn validate_glm4_moe_lite_manifest(manifest: &NativeModelManifest) -> Result<(), MlxRunnerError> {
    if manifest.model_family != "glm4_moe_lite" {
        return Err(MlxRunnerError::UnsupportedFeature(
            "GLM MLA tensor roles are supported only for glm4_moe_lite manifests".to_string(),
        ));
    }
    if !manifest.mla_attention.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires mla_attention metadata".to_string(),
        ));
    }
    if !manifest.glm_router.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router metadata".to_string(),
        ));
    }
    if !manifest.moe.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires moe metadata".to_string(),
        ));
    }

    let first_dense_layers = manifest.glm_router.first_dense_layer_count.ok_or_else(|| {
        MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router.first_dense_layer_count".to_string(),
        )
    })?;
    let has_shared_experts = manifest.glm_router.has_shared_experts;

    for layer_index in 0..manifest.layer_count {
        for role in [
            NativeTensorRole::AttentionNorm,
            NativeTensorRole::AttentionQa,
            NativeTensorRole::AttentionQaNorm,
            NativeTensorRole::AttentionQb,
            NativeTensorRole::AttentionKvA,
            NativeTensorRole::AttentionKvANorm,
            NativeTensorRole::AttentionEmbedQ,
            NativeTensorRole::AttentionUnembedOut,
            NativeTensorRole::AttentionO,
            NativeTensorRole::AttentionPostNorm,
        ] {
            require_manifest_role(manifest, layer_index, role)?;
        }

        if layer_index < first_dense_layers {
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnGate)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnUp)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnDown)?;
        } else {
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnGateInp)?;
            require_manifest_role(
                manifest,
                layer_index,
                NativeTensorRole::FfnGateInpCorrectionBias,
            )?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnGateExps)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnUpExps)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnDownExps)?;
            if has_shared_experts {
                require_manifest_role(
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnSharedExpertGate,
                )?;
                require_manifest_role(manifest, layer_index, NativeTensorRole::FfnSharedExpertUp)?;
                require_manifest_role(
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnSharedExpertDown,
                )?;
            }
        }
    }

    Ok(())
}

fn require_manifest_role(
    manifest: &NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
) -> Result<(), MlxRunnerError> {
    if manifest
        .tensors
        .iter()
        .any(|tensor| tensor.layer_index == Some(layer_index) && tensor.role == role)
    {
        return Ok(());
    }

    Err(MlxRunnerError::UnsupportedFeature(format!(
        "glm4_moe_lite layer {layer_index} is missing required tensor role {role:?}"
    )))
}

fn has_glm_mla_tensors(artifacts: &NativeModelArtifacts) -> bool {
    artifacts.tensor_specs().iter().any(|tensor| {
        matches!(
            tensor.role,
            NativeTensorRole::AttentionQa
                | NativeTensorRole::AttentionQaNorm
                | NativeTensorRole::AttentionQb
                | NativeTensorRole::AttentionKvA
                | NativeTensorRole::AttentionKvANorm
                | NativeTensorRole::AttentionEmbedQ
                | NativeTensorRole::AttentionUnembedOut
                | NativeTensorRole::FfnGateInpCorrectionBias
        )
    })
}

fn has_linear_attention_tensors(artifacts: &NativeModelArtifacts) -> bool {
    artifacts.tensor_specs().iter().any(|tensor| {
        matches!(
            tensor.role,
            NativeTensorRole::LinearAttentionInProjQkv
                | NativeTensorRole::LinearAttentionInProjZ
                | NativeTensorRole::LinearAttentionInProjA
                | NativeTensorRole::LinearAttentionInProjB
                | NativeTensorRole::LinearAttentionConv1d
                | NativeTensorRole::LinearAttentionDtBias
                | NativeTensorRole::LinearAttentionALog
                | NativeTensorRole::LinearAttentionNorm
                | NativeTensorRole::LinearAttentionOutProj
        )
    })
}

fn binding_summary_from_specs(
    specs: &[ax_engine_core::NativeTensorSpec],
) -> NativeModelBindingSummary {
    let mut summary = NativeModelBindingSummary {
        bindings_prepared: true,
        buffers_bound: true,
        buffer_count: specs.len().min(u32::MAX as usize) as u32,
        buffer_bytes: 0,
        source_quantized_binding_count: 0,
        source_q4_k_binding_count: 0,
        source_q5_k_binding_count: 0,
        source_q6_k_binding_count: 0,
        source_q8_0_binding_count: 0,
    };

    for spec in specs {
        summary.buffer_bytes = summary.buffer_bytes.saturating_add(spec.length_bytes);
        if !spec.source_quantized {
            continue;
        }
        summary.source_quantized_binding_count =
            summary.source_quantized_binding_count.saturating_add(1);
        match spec.source_tensor_type.as_deref() {
            Some("q4_k") => {
                summary.source_q4_k_binding_count =
                    summary.source_q4_k_binding_count.saturating_add(1);
            }
            Some("q5_k") => {
                summary.source_q5_k_binding_count =
                    summary.source_q5_k_binding_count.saturating_add(1);
            }
            Some("q6_k") => {
                summary.source_q6_k_binding_count =
                    summary.source_q6_k_binding_count.saturating_add(1);
            }
            Some("q8_0") => {
                summary.source_q8_0_binding_count =
                    summary.source_q8_0_binding_count.saturating_add(1);
            }
            _ => {}
        }
    }

    summary
}

fn resolve_terminal_token_ids(artifacts: &NativeModelArtifacts) -> Vec<u32> {
    let mut token_ids = BTreeSet::new();
    let mut token_strings = BTreeSet::new();

    for file_name in ["config.json", "tokenizer_config.json"] {
        let Some(value) = read_json_file(&artifacts.root_dir().join(file_name)) else {
            continue;
        };
        collect_token_ids(value.get("eos_token_id"), &mut token_ids);
        collect_token_ids(value.get("eos_token_ids"), &mut token_ids);
        collect_token_ids(value.get("pad_token_id"), &mut token_ids);
        collect_token_strings(value.get("eos_token"), &mut token_strings);
        collect_token_strings(value.get("pad_token"), &mut token_strings);
    }

    if !token_strings.is_empty()
        && let Some(tokenizer) = read_json_file(&artifacts.root_dir().join("tokenizer.json"))
    {
        collect_added_token_ids_for_strings(&tokenizer, &token_strings, &mut token_ids);
    }

    let vocab_size = artifacts.manifest().vocab_size;
    token_ids.retain(|token_id| *token_id < vocab_size);
    token_ids.into_iter().collect()
}

fn read_json_file(path: &std::path::Path) -> Option<serde_json::Value> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn collect_token_ids(value: Option<&serde_json::Value>, token_ids: &mut BTreeSet<u32>) {
    match value {
        Some(serde_json::Value::Number(number)) => {
            if let Some(token_id) = number.as_u64().and_then(|id| u32::try_from(id).ok()) {
                token_ids.insert(token_id);
            }
        }
        Some(serde_json::Value::Array(values)) => {
            for value in values {
                collect_token_ids(Some(value), token_ids);
            }
        }
        Some(serde_json::Value::Object(object)) => {
            collect_token_ids(object.get("id"), token_ids);
        }
        _ => {}
    }
}

fn collect_token_strings(value: Option<&serde_json::Value>, token_strings: &mut BTreeSet<String>) {
    match value {
        Some(serde_json::Value::String(token)) => {
            token_strings.insert(token.clone());
        }
        Some(serde_json::Value::Array(values)) => {
            for value in values {
                collect_token_strings(Some(value), token_strings);
            }
        }
        Some(serde_json::Value::Object(object)) => {
            if let Some(content) = object.get("content") {
                collect_token_strings(Some(content), token_strings);
            }
        }
        _ => {}
    }
}

fn collect_added_token_ids_for_strings(
    tokenizer: &serde_json::Value,
    token_strings: &BTreeSet<String>,
    token_ids: &mut BTreeSet<u32>,
) {
    let Some(added_tokens) = tokenizer
        .get("added_tokens")
        .and_then(|value| value.as_array())
    else {
        return;
    };

    for token in added_tokens {
        let Some(content) = token.get("content").and_then(|value| value.as_str()) else {
            continue;
        };
        if !token_strings.contains(content) {
            continue;
        }
        collect_token_ids(token.get("id"), token_ids);
    }
}

fn validate_qwen_gated_delta_linear_attention(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if !matches!(manifest.model_family.as_str(), "qwen3_5" | "qwen3_next") {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention is currently supported only for qwen3_5/qwen3_next MLX manifests"
                .to_string(),
        ));
    }
    let Some(key_head_dim) = manifest.linear_attention.key_head_dim else {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.key_head_dim must be configured".to_string(),
        ));
    };
    if key_head_dim % 32 != 0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "linear_attention.key_head_dim {key_head_dim} must be divisible by 32 for the MLX gated-delta kernel"
        )));
    }
    Ok(())
}

fn validate_gemma4_interleaved_attention(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if manifest.model_family != "gemma4" {
        return Err(MlxRunnerError::UnsupportedFeature(
            "interleaved sliding/full attention is only implemented for Gemma4 manifests"
                .to_string(),
        ));
    }
    if manifest.layer_types.len() != manifest.layer_count as usize {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "Gemma4 interleaved attention requires one layer_type per layer, got {} for {} layers",
            manifest.layer_types.len(),
            manifest.layer_count
        )));
    }

    for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
        if layer_type != "sliding_attention" && layer_type != "full_attention" {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "Gemma4 layer {idx} uses unsupported layer_type {layer_type:?}"
            )));
        }
    }

    let has_sliding = manifest
        .layer_types
        .iter()
        .any(|layer_type| layer_type == "sliding_attention");
    if has_sliding && manifest.sliding_window_size.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "Gemma4 sliding_attention layers require sliding_window_size".to_string(),
        ));
    }

    for (&layer, &source) in &manifest.kv_shared_source_layers {
        if layer >= manifest.layer_count || source >= manifest.layer_count || source >= layer {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "Gemma4 KV-shared layer {layer} has invalid source layer {source}"
            )));
        }
        let layer_type = &manifest.layer_types[layer as usize];
        let source_type = &manifest.layer_types[source as usize];
        if layer_type != source_type {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "Gemma4 KV-shared layer {layer} type {layer_type:?} cannot reuse source {source} type {source_type:?}"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::model::{NativeGlmRouterConfig, NativeMlaAttentionConfig};
    use ax_engine_core::scheduler::PositionRange;
    use ax_engine_core::{
        AX_NATIVE_MODEL_MANIFEST_FILE, NativeLinearAttentionConfig, NativeModelManifest,
        NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType, NativeTensorFormat,
        NativeTensorSpec,
    };
    use std::fs;
    use std::path::{Path, PathBuf};

    // Verify that the extract-work-reinsert mutex pattern correctly isolates
    // per-request state without GPU execution required.
    #[test]
    fn state_extraction_isolates_concurrent_requests() {
        let mut states: HashMap<RequestId, RequestState> = HashMap::new();
        let a = RequestId(1);
        let b = RequestId(2);

        // Extract A from the map (simulates the lock-brief-remove step).
        // While A is extracted, B's slot is accessible without contention.
        let state_a = states.remove(&a).unwrap_or_else(|| RequestState::new(2, a));
        let state_b = states.remove(&b).unwrap_or_else(|| RequestState::new(2, b));

        // GPU work would run here with state_a / state_b outside the map.
        // Verify B can be reinserted independently of A.
        states.insert(b, state_b);
        states.insert(a, state_a);

        assert_eq!(states.len(), 2);
        assert!(states.contains_key(&a));
        assert!(states.contains_key(&b));
    }

    #[test]
    fn completed_request_state_is_not_reinserted() {
        let mut states: HashMap<RequestId, RequestState> = HashMap::new();
        let id = RequestId(42);
        states.insert(id, RequestState::new(2, id));

        // Extract and simulate a completed request (stop_reason.is_some()).
        // The state should not be reinserted, mirroring the run_item control flow.
        let _state = states.remove(&id).unwrap();
        // No states.insert here — dropped at end of scope.

        assert!(
            !states.contains_key(&id),
            "completed request must not leave orphaned state"
        );
    }

    #[test]
    fn prefill_clears_bonus_and_last_token() {
        let mut state = RequestState::new(2, RequestId(0));
        state.bonus_queue.push_back(99);
        state.bonus_queue.push_back(100);
        state.next_model_last_token = Some(5);
        state.ngram_disabled_steps = 3;
        state.linear_ngram_no_draft_streak = 7;
        state.ngram_acceleration_disabled_for_request = true;

        // Simulate the prefill reset branch of run_item.
        state.bonus_queue.clear();
        state.next_model_last_token = None;
        state.ngram_disabled_steps = 0;
        state.linear_ngram_no_draft_streak = 0;
        state.ngram_acceleration_disabled_for_request = false;

        assert!(
            state.bonus_queue.is_empty(),
            "bonus queue must be cleared on prefill"
        );
        assert!(
            state.next_model_last_token.is_none(),
            "last_token pointer must be reset on prefill"
        );
        assert_eq!(state.ngram_disabled_steps, 0);
        assert_eq!(state.linear_ngram_no_draft_streak, 0);
        assert!(!state.ngram_acceleration_disabled_for_request);
    }

    #[test]
    fn ngram_decode_result_keeps_correction_token_in_output_queue() {
        let mut state = RequestState::new(2, RequestId(7));

        let output = apply_decode_result(&mut state, &[11, 12], &[]);

        assert_eq!(output, 11);
        assert_eq!(
            state.bonus_queue.iter().copied().collect::<Vec<_>>(),
            vec![12],
            "correction token must be emitted before the next model run"
        );
        assert_eq!(state.next_model_last_token, Some(12));
    }

    #[test]
    fn ngram_decode_result_queues_full_accept_tail_and_bonus() {
        let mut state = RequestState::new(2, RequestId(8));

        let output = apply_decode_result(&mut state, &[21, 22, 23, 24], &[]);

        assert_eq!(output, 21);
        assert_eq!(
            state.bonus_queue.iter().copied().collect::<Vec<_>>(),
            vec![22, 23, 24],
            "accepted drafts and final bonus token are all user-visible output"
        );
        assert_eq!(state.next_model_last_token, Some(24));
    }

    #[test]
    fn stop_reason_prefers_eos_before_max_output() {
        assert_eq!(
            stop_reason_for_sampled_token(151645, 1, 32, &[151645]),
            Some(StopReason::EosToken)
        );
        assert_eq!(
            stop_reason_for_sampled_token(7, 31, 32, &[151645]),
            Some(StopReason::MaxOutputTokens)
        );
        assert_eq!(stop_reason_for_sampled_token(7, 1, 32, &[151645]), None);
    }

    #[test]
    fn ngram_decode_result_truncates_bonus_queue_at_eos() {
        let mut state = RequestState::new(2, RequestId(9));

        let output = apply_decode_result(&mut state, &[31, 32, 151645, 33], &[151645]);

        assert_eq!(output, 31);
        assert_eq!(
            state.bonus_queue.iter().copied().collect::<Vec<_>>(),
            vec![32, 151645],
            "verified bonus tokens after EOS must not be emitted"
        );
        assert_eq!(state.next_model_last_token, Some(151645));
    }

    #[test]
    fn split_prefill_only_completes_on_final_prompt_chunk() {
        let item = ax_engine_core::ExecutionItem {
            request_id: RequestId(10),
            mode: ExecutionMode::Prefill,
            input_token_slice: vec![0; 2048],
            reused_prefix_token_slice: Vec::new(),
            position_range: PositionRange {
                start: 0,
                end_exclusive: 2048,
            },
            scheduled_token_count: 2048,
            block_table_ref: RequestId(10),
            prefix_tokens_reused: 0,
            prefix_blocks_reused: 0,
        };
        let first_context = RunnerRequestContext {
            request_id: RequestId(10),
            prompt_len: 2722,
            processed_prompt_tokens: 0,
            generated_len: 0,
            max_output_tokens: 24,
            deterministic_argmax_sampling: true,
            temperature: 0.0,
        };
        assert!(!prefill_item_completes_prompt(&item, Some(&first_context)));

        let final_item = ax_engine_core::ExecutionItem {
            input_token_slice: vec![0; 674],
            position_range: PositionRange {
                start: 2048,
                end_exclusive: 2722,
            },
            scheduled_token_count: 674,
            ..item
        };
        let final_context = RunnerRequestContext {
            processed_prompt_tokens: 2048,
            ..first_context
        };
        assert!(prefill_item_completes_prompt(
            &final_item,
            Some(&final_context)
        ));
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        static NEXT_TEST_DIR_ID: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = NEXT_TEST_DIR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ax-mlx-runner-{label}-{}-{id}-{nanos}",
            std::process::id()
        ))
    }

    fn tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    #[test]
    fn binding_summary_reports_manifest_bytes_and_quantized_sources() {
        let dense = tensor("dense", NativeTensorRole::AttentionNorm, Some(0), vec![4]);
        let mut q4 = tensor("q4", NativeTensorRole::AttentionQ, Some(0), vec![8, 4]);
        q4.source_quantized = true;
        q4.source_tensor_type = Some("q4_k".to_string());
        q4.length_bytes = 64;
        let mut u32_affine = tensor("u32", NativeTensorRole::AttentionO, Some(0), vec![4, 1]);
        u32_affine.source_quantized = true;
        u32_affine.length_bytes = 16;

        let summary = binding_summary_from_specs(&[dense, q4, u32_affine]);

        assert!(summary.bindings_prepared);
        assert!(summary.buffers_bound);
        assert_eq!(summary.buffer_count, 3);
        assert_eq!(summary.buffer_bytes, 112);
        assert_eq!(summary.source_quantized_binding_count, 2);
        assert_eq!(summary.source_q4_k_binding_count, 1);
        assert_eq!(summary.source_q5_k_binding_count, 0);
        assert_eq!(summary.source_q6_k_binding_count, 0);
        assert_eq!(summary.source_q8_0_binding_count, 0);
    }

    fn dense_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "test_dense".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 1,
            hidden_size: 4,
            intermediate_size: 8,
            attention_head_count: 1,
            attention_head_dim: 4,
            kv_head_count: 1,
            vocab_size: 16,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            tensors: vec![
                tensor(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    vec![16, 4],
                ),
                tensor(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    vec![4],
                ),
                tensor(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    vec![16, 4],
                ),
                tensor(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![4],
                ),
                tensor(
                    "model.layers.0.self_attn.q_proj.weight",
                    NativeTensorRole::AttentionQ,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.self_attn.k_proj.weight",
                    NativeTensorRole::AttentionK,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.self_attn.v_proj.weight",
                    NativeTensorRole::AttentionV,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.mlp.norm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![4],
                ),
                tensor(
                    "model.layers.0.mlp.gate_proj.weight",
                    NativeTensorRole::FfnGate,
                    Some(0),
                    vec![8, 4],
                ),
                tensor(
                    "model.layers.0.mlp.up_proj.weight",
                    NativeTensorRole::FfnUp,
                    Some(0),
                    vec![8, 4],
                ),
                tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    vec![4, 8],
                ),
            ],
        }
    }

    fn set_vocab_size(manifest: &mut NativeModelManifest, vocab_size: u32) {
        manifest.vocab_size = vocab_size;
        for tensor in &mut manifest.tensors {
            if matches!(
                tensor.role,
                NativeTensorRole::TokenEmbedding | NativeTensorRole::LmHead
            ) {
                tensor.shape[0] = vocab_size as u64;
            }
        }
    }

    fn write_artifacts(manifest: NativeModelManifest) -> NativeModelArtifacts {
        let dir = unique_test_dir("manifest");
        fs::create_dir_all(&dir).expect("fixture directory should create");
        fs::write(dir.join("model.safetensors"), vec![0_u8; 4096]).expect("weights should write");
        fs::write(
            dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
        NativeModelArtifacts::from_dir(&dir).expect("fixture manifest should validate")
    }

    #[test]
    fn terminal_token_ids_resolve_from_config_json_array() {
        let mut manifest = dense_manifest();
        set_vocab_size(&mut manifest, 128);
        let artifacts = write_artifacts(manifest);
        fs::write(
            artifacts.root_dir().join("config.json"),
            r#"{"eos_token_id":[1,106,999]}"#,
        )
        .expect("config should write");

        assert_eq!(resolve_terminal_token_ids(&artifacts), vec![1, 106]);
    }

    #[test]
    fn terminal_token_ids_resolve_tokenizer_config_string_from_tokenizer_json() {
        let mut manifest = dense_manifest();
        set_vocab_size(&mut manifest, 200_000);
        let artifacts = write_artifacts(manifest);
        fs::write(
            artifacts.root_dir().join("tokenizer_config.json"),
            r#"{"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"}"#,
        )
        .expect("tokenizer config should write");
        fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":151643,"content":"<|endoftext|>"},{"id":151645,"content":"<|im_end|>"}]}"#,
        )
        .expect("tokenizer should write");

        assert_eq!(resolve_terminal_token_ids(&artifacts), vec![151643, 151645]);
    }

    fn qwen35_linear_manifest() -> NativeModelManifest {
        let mut manifest = dense_manifest();
        manifest.model_family = "qwen3_5".to_string();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: None,
            num_value_heads: Some(1),
            num_key_heads: Some(1),
            key_head_dim: Some(32),
            value_head_dim: Some(4),
            conv_kernel_dim: Some(4),
        };
        manifest.tensors.retain(|tensor| {
            !matches!(
                tensor.role,
                NativeTensorRole::AttentionQ
                    | NativeTensorRole::AttentionK
                    | NativeTensorRole::AttentionV
                    | NativeTensorRole::AttentionO
            )
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                NativeTensorRole::LinearAttentionInProjQkv,
                Some(0),
                vec![68, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.in_proj_z.weight",
                NativeTensorRole::LinearAttentionInProjZ,
                Some(0),
                vec![4, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.in_proj_a.weight",
                NativeTensorRole::LinearAttentionInProjA,
                Some(0),
                vec![1, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.in_proj_b.weight",
                NativeTensorRole::LinearAttentionInProjB,
                Some(0),
                vec![1, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.conv1d.weight",
                NativeTensorRole::LinearAttentionConv1d,
                Some(0),
                vec![68, 4, 1],
            ),
            tensor(
                "model.layers.0.linear_attn.dt_bias",
                NativeTensorRole::LinearAttentionDtBias,
                Some(0),
                vec![1],
            ),
            tensor(
                "model.layers.0.linear_attn.A_log",
                NativeTensorRole::LinearAttentionALog,
                Some(0),
                vec![1],
            ),
            tensor(
                "model.layers.0.linear_attn.norm.weight",
                NativeTensorRole::LinearAttentionNorm,
                Some(0),
                vec![4],
            ),
            tensor(
                "model.layers.0.linear_attn.out_proj.weight",
                NativeTensorRole::LinearAttentionOutProj,
                Some(0),
                vec![4, 4],
            ),
        ]);
        manifest
    }

    fn glm4_moe_lite_manifest() -> NativeModelManifest {
        let mut manifest = dense_manifest();
        manifest.model_family = "glm4_moe_lite".to_string();
        manifest.layer_count = 2;
        manifest.attention_head_dim = 4;
        manifest.mla_attention = NativeMlaAttentionConfig {
            q_lora_rank: Some(2),
            kv_lora_rank: Some(2),
            qk_nope_head_dim: Some(2),
            qk_rope_head_dim: Some(2),
            value_head_dim: Some(2),
        };
        manifest.moe = NativeMoeConfig {
            expert_count: Some(4),
            experts_per_token: Some(2),
            expert_intermediate_size: Some(8),
        };
        manifest.glm_router = NativeGlmRouterConfig {
            first_dense_layer_count: Some(1),
            routed_scaling_factor: Some(1.8),
            n_group: Some(1),
            topk_group: Some(1),
            has_shared_experts: true,
        };
        manifest.tensors.retain(|tensor| {
            !matches!(
                tensor.role,
                NativeTensorRole::AttentionQ
                    | NativeTensorRole::AttentionK
                    | NativeTensorRole::AttentionV
                    | NativeTensorRole::AttentionQkvPacked
            )
        });
        for tensor in &mut manifest.tensors {
            if tensor.role == NativeTensorRole::AttentionO {
                tensor.shape = vec![4, 2];
            }
        }

        for layer in 0..2 {
            for (role, shape) in [
                (NativeTensorRole::AttentionPostNorm, vec![4]),
                (NativeTensorRole::AttentionQa, vec![2, 4]),
                (NativeTensorRole::AttentionQaNorm, vec![2]),
                (NativeTensorRole::AttentionQb, vec![4, 2]),
                (NativeTensorRole::AttentionKvA, vec![4, 4]),
                (NativeTensorRole::AttentionKvANorm, vec![2]),
                (NativeTensorRole::AttentionEmbedQ, vec![1, 2, 2]),
                (NativeTensorRole::AttentionUnembedOut, vec![1, 2, 2]),
            ] {
                manifest.tensors.push(tensor(
                    &format!("model.layers.{layer}.{role:?}.weight"),
                    role,
                    Some(layer),
                    shape,
                ));
            }

            if layer == 1 {
                for (role, shape) in [
                    (NativeTensorRole::AttentionNorm, vec![4]),
                    (NativeTensorRole::AttentionO, vec![4, 2]),
                    (NativeTensorRole::FfnGateInp, vec![4, 4]),
                    (NativeTensorRole::FfnGateInpCorrectionBias, vec![4]),
                    (NativeTensorRole::FfnGateExps, vec![4, 8, 4]),
                    (NativeTensorRole::FfnUpExps, vec![4, 8, 4]),
                    (NativeTensorRole::FfnDownExps, vec![4, 4, 8]),
                    (NativeTensorRole::FfnSharedExpertGate, vec![8, 4]),
                    (NativeTensorRole::FfnSharedExpertUp, vec![8, 4]),
                    (NativeTensorRole::FfnSharedExpertDown, vec![4, 8]),
                ] {
                    manifest.tensors.push(tensor(
                        &format!("model.layers.{layer}.{role:?}.weight"),
                        role,
                        Some(layer),
                        shape,
                    ));
                }
            }
        }

        manifest
    }

    #[test]
    fn mlx_manifest_validation_rejects_linear_attention_for_non_qwen35() {
        let mut manifest = dense_manifest();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: Some(4),
            num_value_heads: Some(1),
            num_key_heads: Some(1),
            key_head_dim: Some(4),
            value_head_dim: Some(4),
            conv_kernel_dim: Some(4),
        };
        let artifacts = write_artifacts(manifest);

        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("linear attention should fail closed");

        assert!(error.to_string().contains("qwen3_5/qwen3_next"));
    }

    #[test]
    fn mlx_manifest_validation_rejects_incomplete_glm_contract() {
        let mut manifest = dense_manifest();
        manifest.model_family = "glm4_moe_lite".to_string();
        manifest.tensors.push(tensor(
            "model.layers.0.self_attn.q_a_proj.weight",
            NativeTensorRole::AttentionQa,
            Some(0),
            vec![4, 4],
        ));

        let error = validate_glm4_moe_lite_manifest(&manifest)
            .expect_err("incomplete GLM runtime contract should fail closed");

        assert!(
            error
                .to_string()
                .contains("glm4_moe_lite requires mla_attention metadata")
        );
    }

    #[test]
    fn mlx_manifest_validation_allows_glm4_moe_lite_contract() {
        let artifacts = write_artifacts(glm4_moe_lite_manifest());

        validate_mlx_supported_manifest(&artifacts)
            .expect("GLM4MoELite runtime contract is wired for the MLX path");
    }

    #[test]
    fn mlx_manifest_validation_allows_qwen35_linear_attention() {
        let artifacts = write_artifacts(qwen35_linear_manifest());

        validate_mlx_supported_manifest(&artifacts)
            .expect("Qwen3.5 linear attention is wired for the MLX path");
    }

    #[test]
    fn real_mlx_manifest_resolves_qwen35_linear_interval_when_configured() {
        let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
            return;
        };
        let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
            .expect("real MLX manifest should load");

        validate_mlx_supported_manifest(&artifacts).expect("real MLX manifest should be supported");
        let binding = binding_summary_from_specs(artifacts.tensor_specs());
        assert!(binding.buffer_bytes > 0);
        assert!(binding.source_quantized_binding_count > 0);
        let cfg = ModelConfig::from_manifest(artifacts.manifest());

        assert_eq!(
            cfg.linear_attention
                .as_ref()
                .expect("real manifest should configure linear attention")
                .full_attention_interval,
            4
        );
        assert!(cfg.is_linear_attention_layer(0));
        assert!(!cfg.is_linear_attention_layer(3));
    }

    #[test]
    fn real_mlx_runner_warms_up_qwen35_when_configured() {
        if std::env::var("AX_ENGINE_MLX_RUN_REAL_FORWARD").as_deref() != Ok("1") {
            return;
        }
        let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
            return;
        };
        let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
            .expect("real MLX manifest should load");

        MlxRunner::from_artifacts(&artifacts, 8, true, MlxKvCompressionConfig::disabled())
            .expect("real Qwen3.5 MLX runner should warm up");
    }

    #[test]
    fn linear_attention_ngram_acceleration_cools_down_after_reject() {
        // Complete miss (0 accepted): long cooldown.
        assert_eq!(
            ngram_acceleration_disabled_steps(true, 0, DEFAULT_DRAFT_LEN, 0.95),
            Some(LINEAR_NGRAM_RETRY_INTERVAL)
        );
        // Partial accept (some but not all): short cooldown to retry quickly.
        assert_eq!(
            ngram_acceleration_disabled_steps(true, 3, DEFAULT_DRAFT_LEN, 0.95),
            Some(LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL)
        );
        // Full accept: no cooldown.
        assert_eq!(
            ngram_acceleration_disabled_steps(true, DEFAULT_DRAFT_LEN, DEFAULT_DRAFT_LEN, 0.25),
            None
        );
    }

    #[test]
    fn ngram_acceleration_telemetry_records_acceptance_and_cooldown_counters() {
        let mut telemetry = NgramAccelerationTelemetry::default();

        telemetry.record_no_draft();
        telemetry.record_draft(DEFAULT_DRAFT_LEN, DEFAULT_DRAFT_LEN);
        telemetry.record_draft(DEFAULT_DRAFT_LEN, 0);
        telemetry.record_draft(DEFAULT_DRAFT_LEN, 2);
        telemetry.record_cooldown_step();
        telemetry.record_cooldown_event(4);
        telemetry.record_request_disable_event();
        telemetry.record_request_disabled_step();

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_ngram_no_draft_steps"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_draft_attempts"), Some(&3));
        assert_eq!(
            decisions.get("ax_ngram_draft_tokens"),
            Some(&(DEFAULT_DRAFT_LEN as u32 * 3))
        );
        assert_eq!(
            decisions.get("ax_ngram_accepted_tokens"),
            Some(&(DEFAULT_DRAFT_LEN as u32 + 2))
        );
        assert_eq!(
            decisions.get("ax_ngram_rejected_tokens"),
            Some(&(DEFAULT_DRAFT_LEN as u32 * 2 - 2))
        );
        assert_eq!(decisions.get("ax_ngram_full_accepts"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_complete_misses"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_partial_rejects"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_cooldown_steps"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_cooldown_events"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_cooldown_steps_scheduled"), Some(&4));
        assert_eq!(decisions.get("ax_ngram_request_disable_events"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_request_disabled_steps"), Some(&1));

        let mut zero_decisions = Vec::new();
        NgramAccelerationTelemetry::default().append_route_decisions(&mut zero_decisions);
        let zero_decisions = zero_decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(zero_decisions.get("ax_ngram_draft_attempts"), Some(&0));
        assert_eq!(zero_decisions.get("ax_ngram_complete_misses"), Some(&0));
        assert_eq!(zero_decisions.get("ax_ngram_cooldown_events"), Some(&0));
        assert_eq!(
            zero_decisions.get("ax_ngram_request_disable_events"),
            Some(&0)
        );
    }

    #[test]
    fn gemma4_moe_profile_route_decisions_emit_only_when_enabled() {
        let mut profile = Gemma4MoeProfileSnapshot {
            enabled: 1,
            decode_layers: 2,
            topk_selections: 16,
            sorted_gather_layers: 0,
            unsorted_gather_layers: 2,
            attention_wall_us: 100,
            dense_wall_us: 80,
            router_wall_us: 30,
            expert_wall_us: 90,
            post_wall_us: 20,
        };
        profile.merge_from(Gemma4MoeProfileSnapshot {
            enabled: 1,
            decode_layers: 3,
            topk_selections: 24,
            sorted_gather_layers: 0,
            unsorted_gather_layers: 3,
            attention_wall_us: 150,
            dense_wall_us: 120,
            router_wall_us: 45,
            expert_wall_us: 135,
            post_wall_us: 30,
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_gemma4_moe_profile_enabled"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_gemma4_moe_profile_decode_layers"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_moe_profile_topk_selections"),
            Some(&40)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_moe_profile_unsorted_gather_layers"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_moe_profile_attention_wall_us"),
            Some(&250)
        );

        let mut disabled_decisions = Vec::new();
        Gemma4MoeProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
        assert!(disabled_decisions.is_empty());
    }

    #[test]
    fn linear_attention_profile_route_decisions_emit_only_when_enabled() {
        let mut profile = LinearAttentionProfileSnapshot {
            enabled: 1,
            layers: 2,
            tokens: 1024,
            projection_wall_us: 100,
            conv_wall_us: 80,
            qk_norm_wall_us: 30,
            recurrent_wall_us: 90,
            output_wall_us: 20,
        };
        profile.merge_from(LinearAttentionProfileSnapshot {
            enabled: 1,
            layers: 3,
            tokens: 2048,
            projection_wall_us: 150,
            conv_wall_us: 120,
            qk_norm_wall_us: 45,
            recurrent_wall_us: 135,
            output_wall_us: 30,
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_linear_attention_profile_enabled"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_profile_layers"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_profile_tokens"),
            Some(&3072)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_profile_projection_wall_us"),
            Some(&250)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_profile_recurrent_wall_us"),
            Some(&225)
        );

        let mut disabled_decisions = Vec::new();
        LinearAttentionProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
        assert!(disabled_decisions.is_empty());
    }

    #[test]
    fn linear_attention_no_draft_threshold_disables_request_acceleration() {
        assert!(!linear_ngram_no_draft_should_disable(
            LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD - 1
        ));
        assert!(linear_ngram_no_draft_should_disable(
            LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD
        ));
    }

    #[test]
    fn route_decision_upsert_replaces_existing_value_and_removes_duplicates() {
        let mut decisions = vec![
            ("other".to_string(), 7),
            ("ax_ngram_draft_tokens".to_string(), 1),
            ("ax_ngram_draft_tokens".to_string(), 2),
        ];

        upsert_route_decision(&mut decisions, "ax_ngram_draft_tokens", 9);

        assert_eq!(decisions[0], ("other".to_string(), 7));
        assert_eq!(decisions[1], ("ax_ngram_draft_tokens".to_string(), 9));
        assert_eq!(
            decisions
                .iter()
                .filter(|(key, _)| key == "ax_ngram_draft_tokens")
                .count(),
            1
        );
    }

    #[test]
    fn decode_telemetry_records_route_counters() {
        let mut telemetry = DecodeTelemetry::default();

        telemetry.record_prefill(100);
        telemetry.record_decode(40);
        telemetry.record_direct_bootstrap(7);
        telemetry.record_direct_pipeline(11);
        telemetry.record_single_decode(13);
        telemetry.record_ngram_decode(17);
        telemetry.record_bonus_token();
        telemetry.record_bonus_token();

        let mut decisions = vec![
            ("ax_mlx_decode_steps".to_string(), 999),
            ("other_counter".to_string(), 3),
            ("ax_mlx_decode_steps".to_string(), 111),
        ];
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_prefill_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_prefill_wall_us"), Some(&100));
        assert_eq!(decisions.get("ax_mlx_decode_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_decode_wall_us"), Some(&40));
        assert_eq!(decisions.get("ax_mlx_direct_bootstrap_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_direct_bootstrap_wall_us"), Some(&7));
        assert_eq!(decisions.get("ax_mlx_direct_pipeline_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_direct_pipeline_wall_us"), Some(&11));
        assert_eq!(decisions.get("ax_mlx_single_decode_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_single_decode_wall_us"), Some(&13));
        assert_eq!(decisions.get("ax_mlx_ngram_decode_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_ngram_decode_wall_us"), Some(&17));
        assert_eq!(decisions.get("ax_mlx_bonus_tokens"), Some(&2));
        assert_eq!(decisions.get("other_counter"), Some(&3));
    }

    #[test]
    fn direct_pipeline_clear_cache_cadence_matches_mlx_lm_loop() {
        let due_tokens = (0..=260)
            .filter(|emitted| direct_pipeline_clear_cache_due(*emitted, 256))
            .collect::<Vec<_>>();

        assert_eq!(due_tokens, vec![1, 257]);
        assert!(!direct_pipeline_clear_cache_due(1, 0));
        assert!(direct_pipeline_clear_cache_due(1, 1));
        assert!(direct_pipeline_clear_cache_due(2, 1));
    }

    #[test]
    fn kv_cache_telemetry_records_route_counters() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            logical_tokens: 3,
            capacity_tokens: 256,
            logical_bytes: 96,
            capacity_bytes: 8192,
            full_attention_layers: 1,
            linear_state_layers: 0,
            linear_state_bytes: 0,
            growth_count: 1,
            ..MlxKVCacheUsage::default()
        });
        telemetry.merge_from(MlxKVCacheUsage {
            logical_tokens: 5,
            capacity_tokens: 512,
            logical_bytes: 160,
            capacity_bytes: 16384,
            full_attention_layers: 2,
            sliding_window_layers: 1,
            sliding_window_retained_tokens: 4,
            sliding_window_reclaimable_capacity_tokens: 256,
            sliding_window_reclaimable_capacity_bytes: 8192,
            linear_state_layers: 1,
            linear_state_bytes: 936,
            growth_count: 2,
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS),
            Some(&8)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS),
            Some(&768)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB),
            Some(&24)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS),
            Some(&3)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS),
            Some(&4)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS),
            Some(&256)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB),
            Some(&8)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
            Some(&3)
        );
        assert!(
            !decisions
                .keys()
                .any(|key| key.starts_with("ax_mlx_kv_compression_")),
            "compression counters must stay absent when the policy is disabled"
        );
        assert!(
            ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS
                .iter()
                .all(|key| decisions.contains_key(*key)),
            "KV telemetry must emit the full canonical counter set after any snapshot"
        );
    }

    #[test]
    fn kv_cache_telemetry_emits_zero_counters_after_empty_snapshot() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage::default());

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.len(),
            ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS.len(),
            "KV telemetry should distinguish zero counters from unsupported telemetry"
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS),
            Some(&1)
        );
        for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
            assert!(
                decisions.contains_key(key),
                "missing canonical KV counter {key}"
            );
        }
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
            Some(&0)
        );
        assert!(
            !decisions
                .keys()
                .any(|key| key.starts_with("ax_mlx_kv_compression_")),
            "empty disabled snapshots must not add compression metadata"
        );
    }

    #[test]
    fn kv_cache_telemetry_emits_compression_counters_only_when_enabled() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            kv_compression: crate::kv_cache::MlxKvCompressionUsage {
                policy_enabled: true,
                status_code: 1,
                preset_code: 1,
                key_bits: 8,
                value_bits: 4,
                eligible_layers: 2,
                candidate_token_layers: 688,
                hot_token_layers: 512,
                full_precision_bytes: 22_016,
                estimated_compressed_bytes: 8_256,
                estimated_saved_bytes: 13_760,
                estimated_ratio_milli: 375,
                ..crate::kv_cache::MlxKvCompressionUsage::default()
            },
            ..MlxKVCacheUsage::default()
        });
        telemetry.record_compression_shadow_sync(1_234);
        telemetry.record_compression_shadow_sync(4_321);

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_REQUEST_SNAPSHOTS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_STATUS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_CANDIDATE_TOKEN_LAYERS),
            Some(&688)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FULL_PRECISION_KIB),
            Some(&22)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_COMPRESSED_KIB),
            Some(&9)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_SAVED_KIB),
            Some(&14)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RATIO_MILLI),
            Some(&375)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ROUTE_METADATA_SCHEMA),
            Some(&TURBOQUANT_ROUTE_METADATA_SCHEMA_VERSION)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_READY),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_BLOCKERS),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_LAYERS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_TOKEN_LAYERS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_KIB),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_WRITTEN_SLOTS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_CALLS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_WALL_US),
            Some(&5_555)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH),
            Some(&KV_COMPRESSION_DECODE_PATH_FULL_PRECISION_SHADOW)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON),
            Some(&KV_COMPRESSION_FUSED_DECODE_FALLBACK_MISSING_RUNTIME_STORAGE)
        );
        for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS {
            assert!(
                decisions.contains_key(key),
                "missing canonical compression counter {key}"
            );
        }
    }

    #[test]
    fn kv_cache_telemetry_marks_unsupported_value_bits_as_unsupported_preset() {
        // K8V8 satisfies preset_code==1 and key_bits==8 but value_bits!=4.
        // Before the fix the condition `preset_code != 1 || key_bits != 8` was
        // false, so the code fell through to MISSING_RUNTIME_STORAGE even though
        // the real reason is an unsupported value quantisation depth.
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            kv_compression: crate::kv_cache::MlxKvCompressionUsage {
                policy_enabled: true,
                status_code: 1,
                preset_code: 1,
                key_bits: 8,
                value_bits: 8, // NOT 4-bit — should be UNSUPPORTED_PRESET
                eligible_layers: 1,
                candidate_token_layers: 512,
                runtime_storage_written_slots: 512,
                ..crate::kv_cache::MlxKvCompressionUsage::default()
            },
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON),
            Some(&KV_COMPRESSION_FUSED_DECODE_FALLBACK_UNSUPPORTED_PRESET)
        );
    }

    #[test]
    fn kv_cache_telemetry_marks_fused_decode_candidates_as_shadow_only() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            kv_compression: crate::kv_cache::MlxKvCompressionUsage {
                policy_enabled: true,
                status_code: 1,
                preset_code: 1,
                key_bits: 8,
                value_bits: 4,
                eligible_layers: 1,
                candidate_token_layers: 512,
                runtime_storage_written_slots: 512,
                ..crate::kv_cache::MlxKvCompressionUsage::default()
            },
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH),
            Some(&KV_COMPRESSION_DECODE_PATH_FULL_PRECISION_SHADOW)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON),
            Some(&KV_COMPRESSION_FUSED_DECODE_FALLBACK_SHADOW_ONLY)
        );
    }

    #[test]
    fn kv_cache_telemetry_does_not_invent_fused_experimental_attempts() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            kv_compression: crate::kv_cache::MlxKvCompressionUsage {
                policy_enabled: true,
                fused_decode_requested: true,
                status_code: 1,
                preset_code: 1,
                key_bits: 8,
                value_bits: 4,
                eligible_layers: 1,
                candidate_token_layers: 512,
                hot_token_layers: 256,
                runtime_storage_written_slots: 512,
                ..crate::kv_cache::MlxKvCompressionUsage::default()
            },
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH),
            Some(&KV_COMPRESSION_DECODE_PATH_FULL_PRECISION_SHADOW)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES),
            Some(&1)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON),
            Some(&KV_COMPRESSION_FUSED_DECODE_FALLBACK_RUNNER_NOT_INTEGRATED)
        );
    }

    #[test]
    fn kv_cache_telemetry_records_cpu_oracle_successes_as_experimental_path() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            kv_compression: crate::kv_cache::MlxKvCompressionUsage {
                policy_enabled: true,
                fused_decode_requested: true,
                fused_decode_attempts: 2,
                fused_decode_successes: 2,
                status_code: 1,
                preset_code: 1,
                key_bits: 8,
                value_bits: 4,
                eligible_layers: 1,
                candidate_token_layers: 512,
                hot_token_layers: 256,
                runtime_storage_written_slots: 512,
                ..crate::kv_cache::MlxKvCompressionUsage::default()
            },
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH),
            Some(&KV_COMPRESSION_DECODE_PATH_CPU_ORACLE_COMPRESSED_DECODE)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON),
            Some(&KV_COMPRESSION_FUSED_DECODE_FALLBACK_NONE)
        );
    }

    #[test]
    fn kv_cache_telemetry_records_metal_successes_as_fused_decode_path() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            kv_compression: crate::kv_cache::MlxKvCompressionUsage {
                policy_enabled: true,
                fused_decode_requested: true,
                fused_decode_attempts: 2,
                fused_decode_successes: 2,
                fused_decode_metal_successes: 2,
                status_code: 1,
                preset_code: 1,
                key_bits: 8,
                value_bits: 4,
                eligible_layers: 1,
                candidate_token_layers: 512,
                hot_token_layers: 256,
                runtime_storage_written_slots: 512,
                ..crate::kv_cache::MlxKvCompressionUsage::default()
            },
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH),
            Some(&KV_COMPRESSION_DECODE_PATH_FUSED_COMPRESSED_DECODE)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON),
            Some(&KV_COMPRESSION_FUSED_DECODE_FALLBACK_NONE)
        );
    }

    #[test]
    fn kv_cache_telemetry_records_cpu_oracle_fallbacks_without_path_promotion() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            kv_compression: crate::kv_cache::MlxKvCompressionUsage {
                policy_enabled: true,
                fused_decode_requested: true,
                fused_decode_attempts: 2,
                fused_decode_successes: 0,
                fused_decode_fallbacks: 2,
                status_code: 1,
                preset_code: 1,
                key_bits: 8,
                value_bits: 4,
                eligible_layers: 1,
                candidate_token_layers: 512,
                hot_token_layers: 256,
                runtime_storage_written_slots: 512,
                ..crate::kv_cache::MlxKvCompressionUsage::default()
            },
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH),
            Some(&KV_COMPRESSION_DECODE_PATH_FULL_PRECISION_SHADOW)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES),
            Some(&0)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS),
            Some(&2)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON),
            Some(&KV_COMPRESSION_FUSED_DECODE_FALLBACK_CPU_ORACLE_UNAVAILABLE)
        );
    }

    #[test]
    fn kv_cache_telemetry_upserts_existing_canonical_counters() {
        let mut telemetry = KvCacheTelemetry::default();
        telemetry.merge_from(MlxKVCacheUsage {
            capacity_tokens: 256,
            capacity_bytes: 8192,
            growth_count: 1,
            ..MlxKVCacheUsage::default()
        });

        let mut decisions = vec![
            (ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS.to_string(), 999),
            ("other_counter".to_string(), 7),
            (ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS.to_string(), 111),
        ];
        telemetry.append_route_decisions(&mut decisions);

        assert_eq!(
            decisions
                .iter()
                .filter(|(key, _)| key == ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS)
                .count(),
            1
        );
        assert_eq!(
            decisions
                .iter()
                .find(|(key, _)| key == ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS)
                .map(|(_, value)| *value),
            Some(256)
        );
        assert!(decisions.contains(&("other_counter".to_string(), 7)));
    }

    #[test]
    fn dense_ngram_acceleration_uses_beta_posterior_gate() {
        // Posterior mean above threshold → no cooldown.
        assert_eq!(
            ngram_acceleration_disabled_steps(false, 3, DEFAULT_DRAFT_LEN, 0.95),
            None
        );
        // Posterior mean below threshold → cooldown period.
        assert_eq!(
            ngram_acceleration_disabled_steps(false, 0, DEFAULT_DRAFT_LEN, 0.49),
            Some(NGRAM_RETRY_INTERVAL)
        );
    }

    #[test]
    fn linear_attention_draft_requires_repeated_ngram_evidence() {
        let mut ngram = NgramTable::new();
        ngram.feed(&[1, 2, 3, 1, 2, 3]);

        // Dense: 3-token cycle builds high-confidence bigrams → draft up to MAX_DRAFT_LEN.
        let dense_draft = ngram_acceleration_draft(&ngram, false);
        assert!(!dense_draft.is_empty(), "dense draft should be non-empty");
        assert!(
            dense_draft.len() <= MAX_DRAFT_LEN,
            "dense draft must not exceed MAX_DRAFT_LEN"
        );

        // Linear-attention: min_support=2 filters one-off n-grams.
        assert!(
            ngram_acceleration_draft(&ngram, true).is_empty(),
            "linear attention should not probe one-off prompt n-grams"
        );

        ngram.feed(&[1, 2, 3]);
        let lin_draft = ngram_acceleration_draft(&ngram, true);
        assert!(
            !lin_draft.is_empty(),
            "linear attention draft should be non-empty after second repeat"
        );
        assert!(
            lin_draft.len() <= DEFAULT_DRAFT_LEN,
            "linear attention draft must not exceed DEFAULT_DRAFT_LEN"
        );
    }

    #[test]
    fn mlx_manifest_validation_rejects_unsupported_linear_key_dim() {
        let mut manifest = qwen35_linear_manifest();
        manifest.linear_attention.key_head_dim = Some(4);
        for tensor in &mut manifest.tensors {
            match tensor.role {
                NativeTensorRole::LinearAttentionInProjQkv => tensor.shape = vec![12, 4],
                NativeTensorRole::LinearAttentionConv1d => tensor.shape = vec![12, 4, 1],
                _ => {}
            }
        }
        let artifacts = write_artifacts(manifest);

        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("Dk must match the gated-delta kernel contract");

        assert!(error.to_string().contains("divisible by 32"));
    }

    #[test]
    fn mlx_manifest_validation_allows_attn_output_gate() {
        let mut manifest = dense_manifest();
        manifest.attn_output_gate = true;
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::AttentionQ)
            .expect("q tensor should exist")
            .shape = vec![8, 4];
        let artifacts = write_artifacts(manifest);

        validate_mlx_supported_manifest(&artifacts)
            .expect("attention output gate is implemented in the MLX model graph");
    }

    #[test]
    fn mlx_manifest_validation_allows_gemma4_interleaved_attention() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec!["sliding_attention".to_string()];
        manifest.global_head_dim = Some(8);
        let artifacts = write_artifacts(manifest);

        validate_mlx_supported_manifest(&artifacts)
            .expect("Gemma4 interleaved attention is implemented in the MLX model graph");
    }

    #[test]
    fn mlx_manifest_validation_rejects_unknown_interleaved_attention() {
        let mut manifest = dense_manifest();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec!["sliding_attention".to_string()];
        manifest.global_head_dim = Some(8);
        let artifacts = write_artifacts(manifest);

        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("non-Gemma4 interleaved attention should fail closed");

        assert!(error.to_string().contains("Gemma4"));
    }

    #[test]
    fn mlx_manifest_validation_allows_valid_gemma4_kv_shared_layers() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.layer_count = 2;
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);

        validate_gemma4_interleaved_attention(&manifest)
            .expect("same-type Gemma4 KV sharing should be supported");
    }

    #[test]
    fn mlx_manifest_validation_rejects_cross_type_gemma4_kv_shared_layers() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.layer_count = 2;
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);

        let error = validate_gemma4_interleaved_attention(&manifest)
            .expect_err("cross-type KV sharing should fail closed");

        assert!(error.to_string().contains("cannot reuse"));
    }
}
