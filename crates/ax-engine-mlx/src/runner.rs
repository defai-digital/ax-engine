use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, MlxStream, add, astype, clear_cache, divide, enable_compile,
    max_recommended_working_set_size, multiply, power, set_wired_limit, sum_axis,
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
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_METAL_SUCCESSES,
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
    DirectPipelineTimings, advance_direct_pipeline_with_timings_and_turboquant_context,
    chunked_prefill, decode_step, start_direct_pipeline_with_turboquant_context,
};
use crate::kv_cache::{MlxKVCache, MlxKVCacheUsage};
use crate::model::{
    DecodeProfileSnapshot, Gemma4MoeProfileSnapshot, LinearAttentionProfileSnapshot, ModelConfig,
    TurboQuantModelDecodeContext, take_decode_profile_snapshot, take_gemma4_moe_profile_snapshot,
    take_linear_attention_profile_snapshot,
};
use crate::ngram_accel::{
    DEFAULT_DRAFT_LEN, LINEAR_MIN_NGRAM_SUPPORT, MAX_DRAFT_LEN, NgramDraftOutcome,
    NgramDraftPolicy, NgramDraftRejection, NgramPolicyVariant, NgramTable, classify_prompt_class,
    effective_draft_confidence_threshold, ngram_accel_decode_step,
    single_decode_with_turboquant_context,
};
use crate::sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64};
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
const NGRAM_DRAFT_LEN_LOW_CONFIDENCE: usize = 2;
const NGRAM_DRAFT_LEN_SHRINK_THRESHOLD: f32 = 0.60;
const NGRAM_RETRY_INTERVAL: u32 = 8;
/// Steps to suppress n-gram acceleration after a complete miss (0 draft tokens accepted)
/// on a linear-attention model.  Recompute cost is O(1) token regardless of context
/// length, so 128 was far too conservative; 16 gives the n-gram table time to
/// recover without sacrificing the whole generation window.
const LINEAR_NGRAM_RETRY_INTERVAL: u32 = 16;
/// Steps to suppress after a *partial* accept (≥1 draft token accepted but not all).
/// Partial accept means the n-gram is close — retry quickly.
const LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL: u32 = 4;
/// If a linear-attention request repeatedly cannot produce any n-gram draft
/// after several short probe windows, stop probing for the rest of the request
/// and use the direct pipeline. Empty drafts have no verifier feedback, but
/// Qwen3-Next coding-style output can develop repeated continuations after the
/// first few generated tokens, so do not permanently disable on the first empty
/// probe window.
const LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD: u32 = 8;
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
const DEFAULT_PREFIX_CACHE_MAX_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_PREFIX_CACHE_MAX_ENTRIES: usize = 64;
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_HITS: &str = "ax_mlx_prefix_cache_hits";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_MISSES: &str = "ax_mlx_prefix_cache_misses";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED: &str = "ax_mlx_prefix_cache_blocked";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_POLICY_DISABLED: &str =
    "ax_mlx_prefix_cache_blocked_policy_disabled";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_UNSUPPORTED_LAYOUT: &str =
    "ax_mlx_prefix_cache_blocked_unsupported_layout";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_TRIM_FAILURE: &str =
    "ax_mlx_prefix_cache_blocked_trim_failure";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_STORES: &str = "ax_mlx_prefix_cache_stores";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_EVICTIONS: &str = "ax_mlx_prefix_cache_evictions";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_REUSED_TOKENS: &str = "ax_mlx_prefix_cache_reused_tokens";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_WARMUP_TOKENS: &str = "ax_mlx_prefix_cache_warmup_tokens";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_ENTRIES: &str = "ax_mlx_prefix_cache_entries";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BYTES_KIB: &str = "ax_mlx_prefix_cache_bytes_kib";
const COMMON_EOT_TOKEN_STRINGS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<turn|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "<EOT>",
    "_<EOT>",
    "<｜end▁of▁sentence｜>",
];

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MlxPrefixCacheKey {
    model_id: String,
    route_policy: String,
    layer_layout: String,
    block_size_tokens: u32,
    token_count: u32,
    token_hash: u64,
}

#[derive(Clone)]
struct MlxPrefixSnapshot {
    cache: MlxKVCache,
    tokens: Vec<u32>,
    token_count: usize,
    bytes: u64,
    greedy_prefill_output_token: Option<u32>,
}

struct MlxPrefixCacheEntry {
    snapshot: Arc<MlxPrefixSnapshot>,
    touch_tick: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MlxPrefixCachePolicy {
    max_bytes: u64,
    max_entries: usize,
}

impl MlxPrefixCachePolicy {
    fn from_env() -> Self {
        Self {
            max_bytes: std::env::var("AX_MLX_PREFIX_CACHE_MAX_BYTES")
                .ok()
                .and_then(|raw| raw.parse::<u64>().ok())
                .unwrap_or(DEFAULT_PREFIX_CACHE_MAX_BYTES),
            max_entries: std::env::var("AX_MLX_PREFIX_CACHE_MAX_ENTRIES")
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .unwrap_or(DEFAULT_PREFIX_CACHE_MAX_ENTRIES),
        }
    }

    fn enabled(self) -> bool {
        self.max_bytes > 0 && self.max_entries > 0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MlxPrefixCacheStats {
    entries: u32,
    bytes: u64,
}

#[derive(Default)]
struct MlxPrefixCache {
    policy: MlxPrefixCachePolicy,
    entries: HashMap<MlxPrefixCacheKey, MlxPrefixCacheEntry>,
    lru: VecDeque<(MlxPrefixCacheKey, u64)>,
    bytes: u64,
    next_touch_tick: u64,
}

impl MlxPrefixCache {
    fn new(policy: MlxPrefixCachePolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    fn get(
        &mut self,
        key: &MlxPrefixCacheKey,
        requested_tokens: &[u32],
    ) -> Option<Arc<MlxPrefixSnapshot>> {
        if !self.policy.enabled() {
            return None;
        }
        let touch_tick = self.allocate_touch_tick();
        let snapshot = {
            let entry = self.entries.get_mut(key)?;
            if entry.snapshot.tokens.as_slice() != requested_tokens {
                return None;
            }
            entry.touch_tick = touch_tick;
            Arc::clone(&entry.snapshot)
        };
        self.lru.push_back((key.clone(), touch_tick));
        self.compact_stale_lru_if_needed();
        Some(snapshot)
    }

    /// Non-mutating exact membership check. Used by the iterative-chat probe
    /// in `restore_reused_prefix_state` to ask "do we already have a snapshot
    /// for these tokens?" without changing LRU touch ordering. The actual hit
    /// (which does touch the entry) goes through `get`.
    fn contains_exact_tokens(&self, key: &MlxPrefixCacheKey, tokens: &[u32]) -> bool {
        self.policy.enabled()
            && self
                .entries
                .get(key)
                .is_some_and(|entry| entry.snapshot.tokens == tokens)
    }

    fn insert(
        &mut self,
        key: MlxPrefixCacheKey,
        snapshot: MlxPrefixSnapshot,
    ) -> MlxPrefixCacheInsertOutcome {
        if !self.policy.enabled() || snapshot.token_count == 0 {
            return MlxPrefixCacheInsertOutcome::default();
        }

        if let Some(previous) = self.entries.remove(&key) {
            self.bytes = self.bytes.saturating_sub(previous.snapshot.bytes);
        }

        let touch_tick = self.allocate_touch_tick();
        self.bytes = self.bytes.saturating_add(snapshot.bytes);
        self.entries.insert(
            key.clone(),
            MlxPrefixCacheEntry {
                snapshot: Arc::new(snapshot),
                touch_tick,
            },
        );
        self.lru.push_back((key.clone(), touch_tick));
        self.compact_stale_lru_if_needed();

        let evictions = self.evict_until_within_policy();
        MlxPrefixCacheInsertOutcome {
            stored: self.entries.contains_key(&key),
            evictions,
        }
    }

    fn stats(&self) -> MlxPrefixCacheStats {
        MlxPrefixCacheStats {
            entries: saturating_u32(self.entries.len()),
            bytes: self.bytes,
        }
    }

    fn enabled(&self) -> bool {
        self.policy.enabled()
    }

    fn allocate_touch_tick(&mut self) -> u64 {
        let tick = self.next_touch_tick;
        self.next_touch_tick = self.next_touch_tick.wrapping_add(1);
        tick
    }

    fn stale_lru_compaction_limit(&self) -> usize {
        self.entries
            .len()
            .max(self.policy.max_entries)
            .max(1)
            .saturating_mul(4)
    }

    fn compact_stale_lru_if_needed(&mut self) {
        if self.lru.len() <= self.stale_lru_compaction_limit() {
            return;
        }

        let entries = &self.entries;
        self.lru.retain(|(key, touch_tick)| {
            entries
                .get(key)
                .is_some_and(|entry| entry.touch_tick == *touch_tick)
        });
    }

    fn evict_until_within_policy(&mut self) -> u32 {
        let mut evictions = 0u32;
        while self.bytes > self.policy.max_bytes || self.entries.len() > self.policy.max_entries {
            let Some((key, touch_tick)) = self.lru.pop_front() else {
                break;
            };
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if entry.touch_tick != touch_tick {
                continue;
            }
            if let Some(removed) = self.entries.remove(&key) {
                self.bytes = self.bytes.saturating_sub(removed.snapshot.bytes);
                evictions = evictions.saturating_add(1);
            }
        }
        evictions
    }
}

impl Default for MlxPrefixCachePolicy {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_PREFIX_CACHE_MAX_BYTES,
            max_entries: DEFAULT_PREFIX_CACHE_MAX_ENTRIES,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MlxPrefixCacheInsertOutcome {
    stored: bool,
    evictions: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MlxPrefixCacheTelemetry {
    hits: u32,
    misses: u32,
    blocked: u32,
    blocked_policy_disabled: u32,
    blocked_unsupported_layout: u32,
    blocked_trim_failure: u32,
    stores: u32,
    evictions: u32,
    reused_tokens: u32,
    warmup_tokens: u32,
    entries: u32,
    bytes: u64,
}

impl MlxPrefixCacheTelemetry {
    fn record_stats(&mut self, stats: MlxPrefixCacheStats) {
        self.entries = stats.entries;
        self.bytes = stats.bytes;
    }

    fn merge_from(&mut self, other: Self) {
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.blocked = self.blocked.saturating_add(other.blocked);
        self.blocked_policy_disabled = self
            .blocked_policy_disabled
            .saturating_add(other.blocked_policy_disabled);
        self.blocked_unsupported_layout = self
            .blocked_unsupported_layout
            .saturating_add(other.blocked_unsupported_layout);
        self.blocked_trim_failure = self
            .blocked_trim_failure
            .saturating_add(other.blocked_trim_failure);
        self.stores = self.stores.saturating_add(other.stores);
        self.evictions = self.evictions.saturating_add(other.evictions);
        self.reused_tokens = self.reused_tokens.saturating_add(other.reused_tokens);
        self.warmup_tokens = self.warmup_tokens.saturating_add(other.warmup_tokens);
        self.entries = self.entries.max(other.entries);
        self.bytes = self.bytes.max(other.bytes);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if *self == Self::default() {
            return;
        }

        let entries = [
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_HITS, self.hits),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_MISSES, self.misses),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED, self.blocked),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_POLICY_DISABLED,
                self.blocked_policy_disabled,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_UNSUPPORTED_LAYOUT,
                self.blocked_unsupported_layout,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_TRIM_FAILURE,
                self.blocked_trim_failure,
            ),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_STORES, self.stores),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_EVICTIONS, self.evictions),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_REUSED_TOKENS,
                self.reused_tokens,
            ),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_WARMUP_TOKENS,
                self.warmup_tokens,
            ),
            (ROUTE_DECISION_AX_MLX_PREFIX_CACHE_ENTRIES, self.entries),
            (
                ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BYTES_KIB,
                kib_ceil(self.bytes),
            ),
        ];

        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
        }
    }

    fn record_blocked_policy_disabled(&mut self) {
        self.blocked = self.blocked.saturating_add(1);
        self.blocked_policy_disabled = self.blocked_policy_disabled.saturating_add(1);
    }

    fn record_blocked_unsupported_layout(&mut self) {
        self.blocked = self.blocked.saturating_add(1);
        self.blocked_unsupported_layout = self.blocked_unsupported_layout.saturating_add(1);
    }

    fn record_blocked_trim_failure(&mut self) {
        self.blocked = self.blocked.saturating_add(1);
        self.blocked_trim_failure = self.blocked_trim_failure.saturating_add(1);
    }
}

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
    fallback_no_candidate_steps: u32,
    fallback_confidence_filtered_steps: u32,
    fallback_short_output_steps: u32,
    fallback_linear_no_draft_steps: u32,
    policy_variant_code: u32,
    adaptive_draft_len_steps: u32,
    adaptive_draft_len_total: u32,
    prompt_class_code: u32,
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

    fn record_no_draft_reason(&mut self, reason: Option<NgramDraftRejection>) {
        match reason {
            Some(NgramDraftRejection::ConfidenceFiltered) => {
                self.fallback_confidence_filtered_steps =
                    self.fallback_confidence_filtered_steps.saturating_add(1);
            }
            Some(NgramDraftRejection::NoCandidate) | None => {
                self.fallback_no_candidate_steps =
                    self.fallback_no_candidate_steps.saturating_add(1);
            }
        }
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

    fn record_request_disabled_reason(&mut self, reason: NgramRequestDisableReason) {
        match reason {
            NgramRequestDisableReason::None => {}
            NgramRequestDisableReason::ShortOutputBudget => {
                self.fallback_short_output_steps =
                    self.fallback_short_output_steps.saturating_add(1);
            }
            NgramRequestDisableReason::LinearNoDraft => {
                self.fallback_linear_no_draft_steps =
                    self.fallback_linear_no_draft_steps.saturating_add(1);
            }
        }
    }

    fn record_policy(&mut self, variant: NgramPolicyVariant, requested_draft_len: usize) {
        self.policy_variant_code = variant.route_code();
        self.adaptive_draft_len_steps = self.adaptive_draft_len_steps.saturating_add(1);
        self.adaptive_draft_len_total = self
            .adaptive_draft_len_total
            .saturating_add(saturating_u32(requested_draft_len));
    }

    fn record_prompt_class(&mut self, class_code: u32) {
        self.prompt_class_code = self.prompt_class_code.max(class_code);
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
        self.fallback_no_candidate_steps = self
            .fallback_no_candidate_steps
            .saturating_add(other.fallback_no_candidate_steps);
        self.fallback_confidence_filtered_steps = self
            .fallback_confidence_filtered_steps
            .saturating_add(other.fallback_confidence_filtered_steps);
        self.fallback_short_output_steps = self
            .fallback_short_output_steps
            .saturating_add(other.fallback_short_output_steps);
        self.fallback_linear_no_draft_steps = self
            .fallback_linear_no_draft_steps
            .saturating_add(other.fallback_linear_no_draft_steps);
        self.policy_variant_code = self.policy_variant_code.max(other.policy_variant_code);
        self.adaptive_draft_len_steps = self
            .adaptive_draft_len_steps
            .saturating_add(other.adaptive_draft_len_steps);
        self.adaptive_draft_len_total = self
            .adaptive_draft_len_total
            .saturating_add(other.adaptive_draft_len_total);
        self.prompt_class_code = self.prompt_class_code.max(other.prompt_class_code);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
            (
                "ax_ngram_fallback_no_candidate_steps",
                self.fallback_no_candidate_steps,
            ),
            (
                "ax_ngram_fallback_confidence_filtered_steps",
                self.fallback_confidence_filtered_steps,
            ),
            (
                "ax_ngram_fallback_short_output_steps",
                self.fallback_short_output_steps,
            ),
            (
                "ax_ngram_fallback_linear_no_draft_steps",
                self.fallback_linear_no_draft_steps,
            ),
            ("ax_ngram_policy_variant", self.policy_variant_code),
            (
                "ax_ngram_adaptive_draft_len_steps",
                self.adaptive_draft_len_steps,
            ),
            (
                "ax_ngram_adaptive_draft_len_total",
                self.adaptive_draft_len_total,
            ),
            ("ax_prompt_class_code", self.prompt_class_code),
        ];

        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectPipelineAction {
    ContinuePending,
    Bootstrap,
}

fn direct_pipeline_action(has_pending_direct: bool) -> DirectPipelineAction {
    if has_pending_direct {
        DirectPipelineAction::ContinuePending
    } else {
        DirectPipelineAction::Bootstrap
    }
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

trait RouteDecisionSink {
    fn upsert_route_decision(&mut self, key: &str, value: u32);
}

impl RouteDecisionSink for Vec<(String, u32)> {
    fn upsert_route_decision(&mut self, key: &str, value: u32) {
        upsert_route_decision(self, key, value);
    }
}

struct IndexedRouteDecisions<'a> {
    decisions: &'a mut Vec<(String, u32)>,
    index: HashMap<String, usize>,
}

impl<'a> IndexedRouteDecisions<'a> {
    fn new(decisions: &'a mut Vec<(String, u32)>) -> Self {
        let mut compacted = Vec::with_capacity(decisions.len());
        let mut index = HashMap::with_capacity(decisions.len());
        for (key, value) in decisions.drain(..) {
            if index.contains_key(&key) {
                continue;
            }
            let position = compacted.len();
            index.insert(key.clone(), position);
            compacted.push((key, value));
        }
        *decisions = compacted;

        Self { decisions, index }
    }
}

impl RouteDecisionSink for IndexedRouteDecisions<'_> {
    fn upsert_route_decision(&mut self, key: &str, value: u32) {
        if let Some(position) = self.index.get(key).copied() {
            self.decisions[position].1 = value;
            return;
        }

        let position = self.decisions.len();
        self.decisions.push((key.to_string(), value));
        self.index.insert(key.to_string(), position);
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
    prefill_forward_wall_us: u32,
    prefill_prefix_cache_wall_us: u32,
    prefill_generation_state_wall_us: u32,
    decode_steps: u32,
    decode_wall_us: u32,
    direct_bootstrap_steps: u32,
    direct_bootstrap_wall_us: u32,
    direct_pipeline_steps: u32,
    direct_pipeline_wall_us: u32,
    direct_pipeline_forward_wall_us: u32,
    direct_pipeline_argmax_wall_us: u32,
    direct_pipeline_async_eval_wall_us: u32,
    direct_pipeline_next_complete_wall_us: u32,
    direct_pipeline_pending_eval_wall_us: u32,
    direct_pipeline_pending_read_wall_us: u32,
    direct_pipeline_op_count: u64,
    single_decode_steps: u32,
    single_decode_wall_us: u32,
    ngram_decode_steps: u32,
    ngram_decode_wall_us: u32,
    bonus_tokens: u32,
    // W1 sync-count instrumentation (ADR 0017 §policy 1).
    // Counts blocking eval() calls by category so profiles can separate
    // production-path GPU barriers from prefill drains.
    production_decode_evals: u32,
    prefill_eval_barriers: u32,
    prefill_drain_async_evals: u32,
}

impl DecodeTelemetry {
    fn record_prefill(&mut self, wall_us: u32) {
        self.prefill_steps = self.prefill_steps.saturating_add(1);
        self.prefill_wall_us = self.prefill_wall_us.saturating_add(wall_us);
    }

    fn record_prefill_breakdown(
        &mut self,
        forward_wall_us: u32,
        prefix_cache_wall_us: u32,
        generation_state_wall_us: u32,
    ) {
        self.prefill_forward_wall_us = self.prefill_forward_wall_us.saturating_add(forward_wall_us);
        self.prefill_prefix_cache_wall_us = self
            .prefill_prefix_cache_wall_us
            .saturating_add(prefix_cache_wall_us);
        self.prefill_generation_state_wall_us = self
            .prefill_generation_state_wall_us
            .saturating_add(generation_state_wall_us);
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

    fn record_direct_pipeline_op_count(&mut self, ops: u64) {
        self.direct_pipeline_op_count = self.direct_pipeline_op_count.saturating_add(ops);
    }

    fn record_direct_pipeline_timings(&mut self, timings: DirectPipelineTimings) {
        self.direct_pipeline_forward_wall_us = self
            .direct_pipeline_forward_wall_us
            .saturating_add(timings.forward_wall_us);
        self.direct_pipeline_argmax_wall_us = self
            .direct_pipeline_argmax_wall_us
            .saturating_add(timings.argmax_wall_us);
        self.direct_pipeline_async_eval_wall_us = self
            .direct_pipeline_async_eval_wall_us
            .saturating_add(timings.async_eval_wall_us);
        self.direct_pipeline_next_complete_wall_us = self
            .direct_pipeline_next_complete_wall_us
            .saturating_add(timings.next_complete_wall_us);
        self.direct_pipeline_pending_eval_wall_us = self
            .direct_pipeline_pending_eval_wall_us
            .saturating_add(timings.pending_eval_wall_us);
        self.direct_pipeline_pending_read_wall_us = self
            .direct_pipeline_pending_read_wall_us
            .saturating_add(timings.pending_read_wall_us);
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

    fn record_production_decode_eval(&mut self) {
        self.production_decode_evals = self.production_decode_evals.saturating_add(1);
    }

    fn record_prefill_eval_barrier(&mut self) {
        self.prefill_eval_barriers = self.prefill_eval_barriers.saturating_add(1);
    }

    fn record_prefill_drain_async_evals(&mut self, count: u32) {
        self.prefill_drain_async_evals = self.prefill_drain_async_evals.saturating_add(count);
    }

    fn merge_from(&mut self, other: Self) {
        self.prefill_steps = self.prefill_steps.saturating_add(other.prefill_steps);
        self.prefill_wall_us = self.prefill_wall_us.saturating_add(other.prefill_wall_us);
        self.prefill_forward_wall_us = self
            .prefill_forward_wall_us
            .saturating_add(other.prefill_forward_wall_us);
        self.prefill_prefix_cache_wall_us = self
            .prefill_prefix_cache_wall_us
            .saturating_add(other.prefill_prefix_cache_wall_us);
        self.prefill_generation_state_wall_us = self
            .prefill_generation_state_wall_us
            .saturating_add(other.prefill_generation_state_wall_us);
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
        self.direct_pipeline_forward_wall_us = self
            .direct_pipeline_forward_wall_us
            .saturating_add(other.direct_pipeline_forward_wall_us);
        self.direct_pipeline_argmax_wall_us = self
            .direct_pipeline_argmax_wall_us
            .saturating_add(other.direct_pipeline_argmax_wall_us);
        self.direct_pipeline_async_eval_wall_us = self
            .direct_pipeline_async_eval_wall_us
            .saturating_add(other.direct_pipeline_async_eval_wall_us);
        self.direct_pipeline_next_complete_wall_us = self
            .direct_pipeline_next_complete_wall_us
            .saturating_add(other.direct_pipeline_next_complete_wall_us);
        self.direct_pipeline_pending_eval_wall_us = self
            .direct_pipeline_pending_eval_wall_us
            .saturating_add(other.direct_pipeline_pending_eval_wall_us);
        self.direct_pipeline_pending_read_wall_us = self
            .direct_pipeline_pending_read_wall_us
            .saturating_add(other.direct_pipeline_pending_read_wall_us);
        self.direct_pipeline_op_count = self
            .direct_pipeline_op_count
            .saturating_add(other.direct_pipeline_op_count);
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
        self.production_decode_evals = self
            .production_decode_evals
            .saturating_add(other.production_decode_evals);
        self.prefill_eval_barriers = self
            .prefill_eval_barriers
            .saturating_add(other.prefill_eval_barriers);
        self.prefill_drain_async_evals = self
            .prefill_drain_async_evals
            .saturating_add(other.prefill_drain_async_evals);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        let entries = [
            ("ax_mlx_prefill_steps", self.prefill_steps),
            ("ax_mlx_prefill_wall_us", self.prefill_wall_us),
            (
                "ax_mlx_prefill_forward_wall_us",
                self.prefill_forward_wall_us,
            ),
            (
                "ax_mlx_prefill_prefix_cache_wall_us",
                self.prefill_prefix_cache_wall_us,
            ),
            (
                "ax_mlx_prefill_generation_state_wall_us",
                self.prefill_generation_state_wall_us,
            ),
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
            (
                "ax_mlx_direct_pipeline_forward_wall_us",
                self.direct_pipeline_forward_wall_us,
            ),
            (
                "ax_mlx_direct_pipeline_argmax_wall_us",
                self.direct_pipeline_argmax_wall_us,
            ),
            (
                "ax_mlx_direct_pipeline_async_eval_wall_us",
                self.direct_pipeline_async_eval_wall_us,
            ),
            (
                "ax_mlx_direct_pipeline_next_complete_wall_us",
                self.direct_pipeline_next_complete_wall_us,
            ),
            (
                "ax_mlx_direct_pipeline_pending_eval_wall_us",
                self.direct_pipeline_pending_eval_wall_us,
            ),
            (
                "ax_mlx_direct_pipeline_pending_read_wall_us",
                self.direct_pipeline_pending_read_wall_us,
            ),
            (
                "ax_mlx_direct_pipeline_op_count",
                u32::try_from(self.direct_pipeline_op_count).unwrap_or(u32::MAX),
            ),
            ("ax_mlx_single_decode_steps", self.single_decode_steps),
            ("ax_mlx_single_decode_wall_us", self.single_decode_wall_us),
            ("ax_mlx_ngram_decode_steps", self.ngram_decode_steps),
            ("ax_mlx_ngram_decode_wall_us", self.ngram_decode_wall_us),
            ("ax_mlx_bonus_tokens", self.bonus_tokens),
            (
                "ax_mlx_production_decode_evals",
                self.production_decode_evals,
            ),
            ("ax_mlx_prefill_eval_barriers", self.prefill_eval_barriers),
            (
                "ax_mlx_prefill_drain_async_evals",
                self.prefill_drain_async_evals,
            ),
        ];

        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
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

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
            decisions.upsert_route_decision(key, value);
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
        self.projection_qkvz_wall_us = self
            .projection_qkvz_wall_us
            .saturating_add(other.projection_qkvz_wall_us);
        self.projection_ba_wall_us = self
            .projection_ba_wall_us
            .saturating_add(other.projection_ba_wall_us);
        self.projection_qkv_wall_us = self
            .projection_qkv_wall_us
            .saturating_add(other.projection_qkv_wall_us);
        self.projection_z_wall_us = self
            .projection_z_wall_us
            .saturating_add(other.projection_z_wall_us);
        self.projection_a_wall_us = self
            .projection_a_wall_us
            .saturating_add(other.projection_a_wall_us);
        self.projection_b_wall_us = self
            .projection_b_wall_us
            .saturating_add(other.projection_b_wall_us);
        self.conv_wall_us = self.conv_wall_us.saturating_add(other.conv_wall_us);
        self.qk_norm_wall_us = self.qk_norm_wall_us.saturating_add(other.qk_norm_wall_us);
        self.recurrent_wall_us = self
            .recurrent_wall_us
            .saturating_add(other.recurrent_wall_us);
        self.output_wall_us = self.output_wall_us.saturating_add(other.output_wall_us);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
                "ax_mlx_linear_attention_profile_projection_qkvz_wall_us",
                self.projection_qkvz_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_projection_ba_wall_us",
                self.projection_ba_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_projection_qkv_wall_us",
                self.projection_qkv_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_projection_z_wall_us",
                self.projection_z_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_projection_a_wall_us",
                self.projection_a_wall_us,
            ),
            (
                "ax_mlx_linear_attention_profile_projection_b_wall_us",
                self.projection_b_wall_us,
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
            decisions.upsert_route_decision(key, value);
        }
    }
}

impl DecodeProfileSnapshot {
    fn merge_from(&mut self, other: Self) {
        self.enabled = self.enabled.max(other.enabled);
        self.decode_steps = self.decode_steps.saturating_add(other.decode_steps);
        self.layers = self.layers.saturating_add(other.layers);
        self.per_layer_input_wall_us = self
            .per_layer_input_wall_us
            .saturating_add(other.per_layer_input_wall_us);
        self.pre_sdpa_wall_us = self.pre_sdpa_wall_us.saturating_add(other.pre_sdpa_wall_us);
        self.pre_sdpa_qkv_proj_wall_us = self
            .pre_sdpa_qkv_proj_wall_us
            .saturating_add(other.pre_sdpa_qkv_proj_wall_us);
        self.pre_sdpa_qk_norm_wall_us = self
            .pre_sdpa_qk_norm_wall_us
            .saturating_add(other.pre_sdpa_qk_norm_wall_us);
        self.pre_sdpa_rope_kv_wall_us = self
            .pre_sdpa_rope_kv_wall_us
            .saturating_add(other.pre_sdpa_rope_kv_wall_us);
        self.sdpa_wall_us = self.sdpa_wall_us.saturating_add(other.sdpa_wall_us);
        self.post_attn_wall_us = self
            .post_attn_wall_us
            .saturating_add(other.post_attn_wall_us);
        self.post_attn_ffn_wall_us = self
            .post_attn_ffn_wall_us
            .saturating_add(other.post_attn_ffn_wall_us);
        self.post_attn_output_proj_wall_us = self
            .post_attn_output_proj_wall_us
            .saturating_add(other.post_attn_output_proj_wall_us);
        self.post_attn_residual_norm_wall_us = self
            .post_attn_residual_norm_wall_us
            .saturating_add(other.post_attn_residual_norm_wall_us);
        self.post_attn_residual_gate_wall_us = self
            .post_attn_residual_gate_wall_us
            .saturating_add(other.post_attn_residual_gate_wall_us);
        self.lm_head_wall_us = self.lm_head_wall_us.saturating_add(other.lm_head_wall_us);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if self.enabled == 0 {
            return;
        }

        let entries = [
            ("ax_mlx_decode_profile_enabled", self.enabled),
            ("ax_mlx_decode_profile_decode_steps", self.decode_steps),
            ("ax_mlx_decode_profile_layers", self.layers),
            (
                "ax_mlx_decode_profile_per_layer_input_wall_us",
                self.per_layer_input_wall_us,
            ),
            (
                "ax_mlx_decode_profile_pre_sdpa_wall_us",
                self.pre_sdpa_wall_us,
            ),
            (
                "ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us",
                self.pre_sdpa_qkv_proj_wall_us,
            ),
            (
                "ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us",
                self.pre_sdpa_qk_norm_wall_us,
            ),
            (
                "ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us",
                self.pre_sdpa_rope_kv_wall_us,
            ),
            ("ax_mlx_decode_profile_sdpa_wall_us", self.sdpa_wall_us),
            (
                "ax_mlx_decode_profile_post_attn_wall_us",
                self.post_attn_wall_us,
            ),
            (
                "ax_mlx_decode_profile_post_attn_ffn_wall_us",
                self.post_attn_ffn_wall_us,
            ),
            (
                "ax_mlx_decode_profile_post_attn_output_proj_wall_us",
                self.post_attn_output_proj_wall_us,
            ),
            (
                "ax_mlx_decode_profile_post_attn_residual_norm_wall_us",
                self.post_attn_residual_norm_wall_us,
            ),
            (
                "ax_mlx_decode_profile_post_attn_residual_gate_wall_us",
                self.post_attn_residual_gate_wall_us,
            ),
            (
                "ax_mlx_decode_profile_lm_head_wall_us",
                self.lm_head_wall_us,
            ),
        ];

        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
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
    compression_fused_decode_metal_successes: u64,
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

    fn merge_fused_decode_blocked_counters(
        &mut self,
        compression: &crate::kv_cache::MlxKvCompressionUsage,
    ) {
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
                self.merge_fused_decode_blocked_counters(&compression);
                if compression.fused_decode_attempts > 0 {
                    self.compression_fused_decode_attempts = self
                        .compression_fused_decode_attempts
                        .saturating_add(compression.fused_decode_attempts);
                    self.compression_fused_decode_successes = self
                        .compression_fused_decode_successes
                        .saturating_add(compression.fused_decode_successes);
                    self.compression_fused_decode_metal_successes = self
                        .compression_fused_decode_metal_successes
                        .saturating_add(compression.fused_decode_metal_successes);
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

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
            decisions.upsert_route_decision(key, value);
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
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_METAL_SUCCESSES,
                    saturating_u32_from_u64(self.compression_fused_decode_metal_successes),
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
                decisions.upsert_route_decision(key, value);
            }
        }
    }
}

/// Per-request mutable state persisted across prefill → decode steps.
struct RequestState {
    cache: MlxKVCache,
    prompt_prefix_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    cached_prefill_output_token: Option<u32>,
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
    ngram_request_disable_reason: NgramRequestDisableReason,
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
    /// Set for explicit direct mode and for request-local linear-attention
    /// n-gram fallback when greedy decoding can continue on the direct pipeline.
    pending_direct: Option<MlxArray>,
    /// Direct-pipeline tokens emitted since the current generation started.
    direct_pipeline_emitted_tokens: u32,
    /// Cumulative per-request counters surfaced through route metadata for
    /// benchmark auditability.
    ngram_acceleration: NgramAccelerationTelemetry,
    decode_telemetry: DecodeTelemetry,
    /// Per-request cumulative decode profile.  The MLX-global
    /// `take_decode_profile_snapshot` returns the delta since the last call;
    /// we merge each batch's delta into this field so the surfaced totals
    /// reflect the full request, not just the latest step.
    decode_profile: DecodeProfileSnapshot,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum NgramRequestDisableReason {
    #[default]
    None,
    ShortOutputBudget,
    LinearNoDraft,
}

impl RequestState {
    fn new(num_layers: usize, request_id: RequestId) -> Self {
        Self {
            cache: MlxKVCache::new(num_layers),
            prompt_prefix_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            cached_prefill_output_token: None,
            ngram: NgramTable::new(),
            rng: Xorshift64::new(request_id.0),
            ngram_beta_alpha: NGRAM_BETA_PRIOR_ALPHA,
            ngram_beta_beta: NGRAM_BETA_PRIOR_BETA,
            ngram_disabled_steps: 0,
            linear_ngram_no_draft_streak: 0,
            ngram_acceleration_disabled_for_request: false,
            ngram_request_disable_reason: NgramRequestDisableReason::None,
            bonus_queue: VecDeque::new(),
            next_model_last_token: None,
            pending_direct: None,
            direct_pipeline_emitted_tokens: 0,
            ngram_acceleration: NgramAccelerationTelemetry::default(),
            decode_telemetry: DecodeTelemetry::default(),
            decode_profile: DecodeProfileSnapshot::default(),
        }
    }

    fn ngram_posterior_mean(&self) -> f32 {
        self.ngram_beta_alpha / (self.ngram_beta_alpha + self.ngram_beta_beta)
    }

    fn repetition_history(
        &self,
        additional_prompt_tokens: &[u32],
        sampling: MlxSamplingParams,
    ) -> Vec<u32> {
        if !sampling.uses_repetition_penalty() {
            return Vec::new();
        }

        let total_len = self
            .prompt_prefix_tokens
            .len()
            .saturating_add(additional_prompt_tokens.len())
            .saturating_add(self.generated_tokens.len());
        let keep_len = sampling
            .repetition_context_size
            .map(|size| size as usize)
            .unwrap_or(total_len)
            .min(total_len);
        if keep_len == 0 {
            return Vec::new();
        }

        let start = total_len - keep_len;
        let mut history = Vec::with_capacity(keep_len);
        let mut remaining_skip = start;
        append_tail(
            &mut history,
            &self.prompt_prefix_tokens,
            &mut remaining_skip,
        );
        append_tail(&mut history, additional_prompt_tokens, &mut remaining_skip);
        append_tail(&mut history, &self.generated_tokens, &mut remaining_skip);
        history
    }
}

fn append_tail(target: &mut Vec<u32>, source: &[u32], skip: &mut usize) {
    if *skip >= source.len() {
        *skip -= source.len();
        return;
    }
    target.extend_from_slice(&source[*skip..]);
    *skip = 0;
}

fn seed_generation_ngram_from_prompt(state: &mut RequestState) {
    let feed_start = state
        .prompt_prefix_tokens
        .len()
        .saturating_sub(NGRAM_PROMPT_FEED_MAX);
    state.ngram.feed(&state.prompt_prefix_tokens[feed_start..]);
}

/// Cache key for the embedding-forward compiled closure: shape-specific.
/// `target_position` is baked into the trace, so we key on (seq_len, pos).
type EmbedCompileKey = (usize, Option<usize>);

/// Cache key for the batched embedding-forward compiled closure.
/// `target_positions` is baked into the trace, so two batches with the same
/// `(batch_size, max_len)` but different per-sequence target positions hit
/// distinct keys.
type EmbedBatchCompileKey = (usize, usize, Option<Vec<usize>>);

/// ExecutionRunner backed by the MLX inference path.
pub struct MlxRunner {
    cfg: ModelConfig,
    cfg_arc: Arc<ModelConfig>,
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
    ngram_policy_variant: NgramPolicyVariant,
    /// Optional KV compression policy. Disabled by default and never changes logits in shadow mode.
    kv_compression: MlxKvCompressionConfig,
    /// Per-layer compression eligibility. Empty when compression is disabled.
    kv_compression_layer_eligible: Vec<bool>,
    /// Immutable MLX KV snapshots for block-aligned exact prompt prefixes.
    prefix_cache: Mutex<MlxPrefixCache>,
    /// Experimental Gemma sliding-window rotating backing store for direct decode.
    rotating_sliding_decode: bool,
    /// Optional mlx_lm-style `clear_cache` cadence for the direct decode pipeline.
    direct_clear_cache_cadence: u32,
    /// Per-shape compiled embedding-forward closures. Each entry is built on
    /// the first `embed()` call at a new `(seq_len, target_position)` shape
    /// and reused for subsequent calls. Set `AX_EMBED_NO_COMPILE=1` to skip
    /// the compiled path and fall back to imperative `forward_for_embedding`.
    embed_compile_cache: Mutex<HashMap<EmbedCompileKey, mlx_sys::MlxClosure>>,
    /// Per-shape compiled batched-embedding-forward closures. Keyed on
    /// `(batch_size, max_len, target_positions)`; same kill switch as the
    /// single-call cache (`AX_EMBED_NO_COMPILE`).
    embed_batch_compile_cache: Mutex<HashMap<EmbedBatchCompileKey, mlx_sys::MlxClosure>>,
    /// Cumulative hit / miss counters for the two embedding compile
    /// caches. Useful to confirm a workload is reusing compiled
    /// closures vs trashing the cache with shape variation. Exported
    /// via `MlxRunner::embed_compile_cache_stats()`.
    embed_compile_stats: Mutex<EmbedCompileStats>,
}

/// Snapshot of the embedding compile-cache telemetry. `len()` is the
/// current cache size (number of distinct compiled closures retained);
/// `hits` / `misses` are cumulative since session creation.
#[derive(Clone, Copy, Debug, Default)]
pub struct EmbedCompileCacheStats {
    pub single_hits: u64,
    pub single_misses: u64,
    pub single_len: usize,
    pub batched_hits: u64,
    pub batched_misses: u64,
    pub batched_len: usize,
}

#[derive(Default)]
struct EmbedCompileStats {
    single_hits: u64,
    single_misses: u64,
    batched_hits: u64,
    batched_misses: u64,
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
        // AX_NO_SPEC is the CLAUDE.md-documented kill switch. Honor it at
        // the runner boundary so server and SDK paths behave the same as
        // the bench CLI, which already reads the env before constructing
        // the runner.
        let disable_ngram_acceleration =
            disable_ngram_acceleration || crate::fastpath::ngram_acceleration_disabled();
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
        let ngram_policy_variant = ngram_policy_variant_from_env();
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
            decode_step(
                &cfg,
                &weights,
                0,
                &mut dummy_cache,
                MlxSamplingParams::greedy(),
                &mut dummy_rng,
            );
            dummy_cache.reset();
            let dummy_tokens: Vec<u32> = vec![0u32; 8];
            chunked_prefill(
                &cfg,
                &weights,
                &dummy_tokens,
                &mut dummy_cache,
                prefill_chunk,
                MlxSamplingRequest::new(MlxSamplingParams::greedy(), &dummy_tokens),
                &mut dummy_rng,
            );
        }
        let _ = take_gemma4_moe_profile_snapshot();
        let _ = take_linear_attention_profile_snapshot();
        let _ = take_decode_profile_snapshot();

        // Qwen3.5 linear-attention uses `ngram_accel_decode_step_linear_safe` which
        // clones the cache for verification and recomputes the committed prefix on
        // partial accept, so n-gram acceleration is safe to enable for these models.
        let cfg_arc = Arc::new(cfg.clone());
        Ok(Self {
            cfg,
            cfg_arc,
            weights,
            prefill_chunk: prefill_chunk.max(1),
            kv_layer_windows,
            binding_summary,
            terminal_token_ids,
            states: Mutex::new(HashMap::new()),
            _stream: stream,
            disable_ngram_acceleration,
            ngram_policy_variant,
            kv_compression,
            kv_compression_layer_eligible,
            prefix_cache: Mutex::new(MlxPrefixCache::new(MlxPrefixCachePolicy::from_env())),
            rotating_sliding_decode,
            direct_clear_cache_cadence,
            embed_compile_cache: Mutex::new(HashMap::new()),
            embed_batch_compile_cache: Mutex::new(HashMap::new()),
            embed_compile_stats: Mutex::new(EmbedCompileStats::default()),
        })
    }

    /// Snapshot the embedding compile-cache hit / miss / size counters.
    /// Use this to diagnose fragmentation: a healthy ingest workload
    /// has `single_hits + batched_hits` dominating the misses, and the
    /// cache sizes stable. A growing cache with a low hit rate signals
    /// the workload's shape distribution is too wide; consider length-
    /// bucketing batches before submitting them.
    pub fn embed_compile_cache_stats(&self) -> EmbedCompileCacheStats {
        let stats = self
            .embed_compile_stats
            .lock()
            .expect("embed_compile_stats mutex poisoned");
        let single_len = self
            .embed_compile_cache
            .lock()
            .map(|c| c.len())
            .unwrap_or(0);
        let batched_len = self
            .embed_batch_compile_cache
            .lock()
            .map(|c| c.len())
            .unwrap_or(0);
        EmbedCompileCacheStats {
            single_hits: stats.single_hits,
            single_misses: stats.single_misses,
            single_len,
            batched_hits: stats.batched_hits,
            batched_misses: stats.batched_misses,
            batched_len,
        }
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
        let mut decode_profile = DecodeProfileSnapshot::default();
        let mut kv_cache = KvCacheTelemetry::default();
        let mut prefix_cache = MlxPrefixCacheTelemetry::default();

        for item in &input.execution_batch.items {
            let ctx = input
                .request_contexts
                .iter()
                .find(|c| c.request_id == item.request_id);

            let result = self.run_item(
                item,
                ctx,
                &input.execution_batch.model_id,
                input.block_size_tokens,
            );
            ngram_acceleration.merge_from(result.ngram_acceleration);
            decode_telemetry.merge_from(result.decode_telemetry);
            gemma4_moe_profile.merge_from(result.gemma4_moe_profile);
            linear_attention_profile.merge_from(result.linear_attention_profile);
            decode_profile.merge_from(result.decode_profile);
            kv_cache.merge_from(result.kv_usage);
            prefix_cache.merge_from(result.prefix_cache);
            if let Some(wall_us) = result.kv_compression_shadow_sync_wall_us {
                kv_cache.record_compression_shadow_sync(wall_us);
            }
            request_updates.push(result.update);
        }
        {
            let mut route_decisions =
                IndexedRouteDecisions::new(&mut route_metadata.crossover_decisions);
            ngram_acceleration.append_route_decisions(&mut route_decisions);
            decode_telemetry.append_route_decisions(&mut route_decisions);
            gemma4_moe_profile.append_route_decisions(&mut route_decisions);
            linear_attention_profile.append_route_decisions(&mut route_decisions);
            decode_profile.append_route_decisions(&mut route_decisions);
            kv_cache.append_route_decisions(&mut route_decisions);
            prefix_cache.append_route_decisions(&mut route_decisions);
        }

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
        // For Last/Cls: tell the forward pass which position to extract before
        // the final norm, so we norm [1, 1, H] instead of [1, seq, H].
        let target_position = match pooling {
            EmbeddingPooling::Last => Some(token_ids.len() - 1),
            EmbeddingPooling::Cls => Some(0),
            EmbeddingPooling::Mean => None,
        };
        let encode_started = Instant::now();
        let hidden = self.embedding_forward(token_ids, target_position);
        let encode_us = elapsed_us(encode_started);

        // Last/Cls: hidden is [1, 1, H] (already at the target position).
        // Mean:     hidden is [1, seq, H]; pool across the sequence here.
        let pool_started = Instant::now();
        let seq = token_ids.len() as i32;
        let pooled = match pooling {
            EmbeddingPooling::Mean => {
                let summed = sum_axis(&hidden, 1, false, None);
                let scale_arr = mlx_scalar_f32(1.0_f32 / seq as f32);
                multiply(&summed, &scale_arr, None)
            }
            EmbeddingPooling::Last | EmbeddingPooling::Cls => hidden,
        };
        let pool_us = elapsed_us(pool_started);

        let normalize_started = Instant::now();
        let pooled_f32 = astype(&pooled, MlxDtype::Float32, None);
        let cpu_normalize = normalize
            && std::env::var("AX_EMBED_GPU_NORMALIZE")
                .map(|v| v == "0" || v.is_empty())
                .unwrap_or(true);
        let result = if normalize && !cpu_normalize {
            l2_normalize_last_dim(&pooled_f32)
        } else {
            pooled_f32
        };
        let normalize_us = elapsed_us(normalize_started);

        let eval_started = Instant::now();
        mlx_sys::eval(&[&result]);
        let eval_us = elapsed_us(eval_started);

        tracing::debug!(
            seq_len = token_ids.len(),
            encode_us,
            pool_us,
            normalize_us,
            eval_us,
            "embed_single stage timing"
        );
        let mut data = result.data_f32().to_vec();
        if cpu_normalize {
            // Single sentence → one row of length hidden_size.
            let hs = data.len();
            l2_normalize_rows_in_place(&mut data, hs);
        }
        Ok(data)
    }

    fn embed_batch(
        &self,
        batch: &[Vec<u32>],
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>, &'static str> {
        if batch.is_empty() {
            return Ok(vec![]);
        }
        for ids in batch {
            if ids.is_empty() {
                return Err("token_ids must not be empty");
            }
        }
        // For Last/Cls: compute per-sequence extraction positions before the
        // forward pass so the model can extract them before the final norm,
        // avoiding norming the full [B, max_seq, H] padded tensor.
        let target_positions: Option<Vec<usize>> = match pooling {
            EmbeddingPooling::Last => Some(batch.iter().map(|ids| ids.len() - 1).collect()),
            EmbeddingPooling::Cls => Some(vec![0; batch.len()]),
            EmbeddingPooling::Mean => None,
        };
        let encode_started = Instant::now();
        let (hidden, actual_lens) =
            self.embedding_batch_forward(batch, target_positions.as_deref());
        let encode_us = elapsed_us(encode_started);

        // Last/Cls: hidden is [B, H] (already extracted).
        // Mean:     hidden is [B, max_seq, H]; pool across sequence here.
        let pool_started = Instant::now();
        let batch_size = batch.len() as i32;
        let hidden_size = hidden.shape()[hidden.shape().len() - 1] as usize;

        let pooled = match pooling {
            EmbeddingPooling::Mean => {
                let max_seq = hidden.shape()[1] as usize;
                // Build a [B, max_seq, 1] mask (1.0 for real tokens, 0.0 for padding)
                // so that right-padded positions are excluded from the mean.
                let mut mask_data = vec![0.0f32; batch.len() * max_seq];
                for (i, &l) in actual_lens.iter().enumerate() {
                    for j in 0..l {
                        mask_data[i * max_seq + j] = 1.0;
                    }
                }
                let mask_arr = MlxArray::from_raw_data(
                    mask_data.as_ptr() as *const u8,
                    mask_data.len() * std::mem::size_of::<f32>(),
                    &[batch_size, max_seq as i32, 1_i32],
                    MlxDtype::Float32,
                );
                let mask_bf16 = astype(&mask_arr, MlxDtype::Bfloat16, None);
                let masked = multiply(&hidden, &mask_bf16, None); // zero padding positions
                let sums = sum_axis(&masked, 1, false, None); // [B, H] bf16
                let scales: Vec<f32> = actual_lens.iter().map(|&l| 1.0 / l as f32).collect();
                let scale_arr = MlxArray::from_raw_data(
                    scales.as_ptr() as *const u8,
                    scales.len() * std::mem::size_of::<f32>(),
                    &[batch_size, 1_i32],
                    MlxDtype::Float32,
                );
                let scale_bf16 = astype(&scale_arr, MlxDtype::Bfloat16, None);
                multiply(&sums, &scale_bf16, None)
            }
            EmbeddingPooling::Last | EmbeddingPooling::Cls => hidden,
        };
        let pool_us = elapsed_us(pool_started);

        let normalize_started = Instant::now();
        let pooled_f32 = astype(&pooled, MlxDtype::Float32, None);
        let cpu_normalize = normalize
            && std::env::var("AX_EMBED_GPU_NORMALIZE")
                .map(|v| v == "0" || v.is_empty())
                .unwrap_or(true);
        let result = if normalize && !cpu_normalize {
            l2_normalize_last_dim(&pooled_f32)
        } else {
            pooled_f32
        };
        let normalize_us = elapsed_us(normalize_started);

        let eval_started = Instant::now();
        mlx_sys::eval(&[&result]);
        let eval_us = elapsed_us(eval_started);

        tracing::debug!(
            batch_size = batch.len(),
            encode_us,
            pool_us,
            normalize_us,
            eval_us,
            "embed_batch stage timing"
        );

        let data_slice_ref = result.data_f32();
        // Read once into a flat buffer; legacy API wants Vec<Vec<f32>> so we
        // split-collect after optional CPU normalize. The B*H read itself is
        // unavoidable in this legacy path; new callers should prefer
        // embed_batch_flat which keeps the buffer contiguous.
        let mut flat: Vec<f32> = data_slice_ref.to_vec();
        if cpu_normalize {
            l2_normalize_rows_in_place(&mut flat, hidden_size);
        }
        // Re-establish original Vec<Vec<f32>> output shape.
        let data: &[f32] = &flat;
        let vecs = (0..batch.len())
            .map(|i| data[i * hidden_size..(i + 1) * hidden_size].to_vec())
            .collect();
        Ok(vecs)
    }

    fn embed_batch_flat(
        &self,
        batch: &[Vec<u32>],
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<ax_engine_core::EmbeddingMatrix, &'static str> {
        if batch.is_empty() {
            return Ok(ax_engine_core::EmbeddingMatrix {
                data: Vec::new(),
                batch_size: 0,
                hidden_size: 0,
            });
        }
        for ids in batch {
            if ids.is_empty() {
                return Err("token_ids must not be empty");
            }
        }
        let target_positions: Option<Vec<usize>> = match pooling {
            EmbeddingPooling::Last => Some(batch.iter().map(|ids| ids.len() - 1).collect()),
            EmbeddingPooling::Cls => Some(vec![0; batch.len()]),
            EmbeddingPooling::Mean => None,
        };
        let (hidden, actual_lens) =
            self.embedding_batch_forward(batch, target_positions.as_deref());
        let batch_size = batch.len() as i32;
        let hidden_size = hidden.shape()[hidden.shape().len() - 1] as usize;
        let pooled = match pooling {
            EmbeddingPooling::Mean => {
                let max_seq = hidden.shape()[1] as usize;
                let mut mask_data = vec![0.0f32; batch.len() * max_seq];
                for (i, &l) in actual_lens.iter().enumerate() {
                    for j in 0..l {
                        mask_data[i * max_seq + j] = 1.0;
                    }
                }
                let mask_arr = MlxArray::from_raw_data(
                    mask_data.as_ptr() as *const u8,
                    mask_data.len() * std::mem::size_of::<f32>(),
                    &[batch_size, max_seq as i32, 1_i32],
                    MlxDtype::Float32,
                );
                let mask_bf16 = astype(&mask_arr, MlxDtype::Bfloat16, None);
                let masked = multiply(&hidden, &mask_bf16, None);
                let sums = sum_axis(&masked, 1, false, None);
                let mut len_data = vec![0.0f32; batch.len()];
                for (i, &l) in actual_lens.iter().enumerate() {
                    len_data[i] = l as f32;
                }
                let len_arr = MlxArray::from_raw_data(
                    len_data.as_ptr() as *const u8,
                    len_data.len() * std::mem::size_of::<f32>(),
                    &[batch_size, 1_i32],
                    MlxDtype::Float32,
                );
                let len_bf16 = astype(&len_arr, MlxDtype::Bfloat16, None);
                divide(&sums, &len_bf16, None)
            }
            EmbeddingPooling::Last | EmbeddingPooling::Cls => hidden,
        };
        let pooled_f32 = astype(&pooled, MlxDtype::Float32, None);
        // R3: when CPU normalize is enabled (default), skip the GPU
        // sqrt + sum + divide ops and l2-normalize the result on the host
        // after read-back. The data is being read back to a CPU `Vec<f32>`
        // anyway, so doing the normalize on the same CPU side amortises a
        // few MLX op dispatches. AX_EMBED_GPU_NORMALIZE=1 reverts.
        let cpu_normalize = normalize
            && std::env::var("AX_EMBED_GPU_NORMALIZE")
                .map(|v| v == "0" || v.is_empty())
                .unwrap_or(true);
        let result = if normalize && !cpu_normalize {
            l2_normalize_last_dim(&pooled_f32)
        } else {
            pooled_f32
        };
        mlx_sys::eval(&[&result]);
        // Single contiguous read-back: B * H floats in one allocation.
        let mut data = result.data_f32().to_vec();
        if cpu_normalize {
            l2_normalize_rows_in_place(&mut data, hidden_size);
        }
        Ok(ax_engine_core::EmbeddingMatrix {
            data,
            batch_size: batch.len(),
            hidden_size,
        })
    }
}

/// L2-normalize `data` in place, viewing it as `[B, hidden_size]` row-major.
/// Auto-vectorises on Apple Silicon (Neon) and x86 (AVX2/SSE) for free —
/// this is hot enough on the embedding read-back path that hand-rolling
/// SIMD adds little vs the compiler's vectoriser, and keeps the code free
/// of platform-specific intrinsics. Adds 1e-12 to the denominator for
/// numerical stability on near-zero vectors (matches the MLX path's eps).
#[inline]
fn l2_normalize_rows_in_place(data: &mut [f32], hidden_size: usize) {
    if hidden_size == 0 || data.is_empty() {
        return;
    }
    debug_assert_eq!(data.len() % hidden_size, 0);
    for row in data.chunks_exact_mut(hidden_size) {
        let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
        let inv_norm = 1.0_f32 / (sum_sq.sqrt() + 1e-12);
        for x in row {
            *x *= inv_norm;
        }
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
    let half = mlx_scalar_f32(0.5);
    let norm = power(&sum_sq, &half, None);
    let eps = mlx_scalar_f32(1e-12);
    let norm_stable = add(&norm, &eps, None);
    divide(x, &norm_stable, None)
}

fn mlx_scalar_f32(value: f32) -> MlxArray {
    MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[],
        MlxDtype::Float32,
    )
}

impl MlxRunner {
    /// Run the embedding forward pass, preferring the compiled-closure path
    /// when caching is permitted. Falls back to imperative
    /// `forward_for_embedding` when:
    ///   * `AX_EMBED_NO_COMPILE` is set (kill switch / A-B benchmarking),
    ///   * the compile step itself returns an error.
    ///
    /// The compiled closure is shape-specific (per `seq_len`,
    /// `target_position`) — first call at a new shape pays the trace cost
    /// once, subsequent calls hit the cache.
    fn embedding_forward(&self, token_ids: &[u32], target_position: Option<usize>) -> MlxArray {
        let disable_compile = std::env::var("AX_EMBED_NO_COMPILE").is_ok();
        if disable_compile {
            return crate::model::forward_for_embedding(
                &self.cfg,
                &self.weights,
                token_ids,
                target_position,
            );
        }
        // The closure body operates on the pre-embedded bf16 hidden state.
        // `embed_tokens` itself is fast (one gather + reshape) and produces
        // an array whose shape encodes seq_len — so we keep it outside the
        // closure and use seq_len as the cache key dimension.
        let mut hidden = crate::model::embed_tokens(
            token_ids,
            &self.weights.token_embedding,
            self.cfg.hidden_size,
        );
        if hidden.dtype() != MlxDtype::Bfloat16 {
            hidden = astype(&hidden, MlxDtype::Bfloat16, None);
        }
        if let Some(scale) = self.cfg.hidden_states_scale {
            hidden = crate::model::scale_hidden_pub(&hidden, scale);
        }

        let key: EmbedCompileKey = (token_ids.len(), target_position);
        let mut cache = self
            .embed_compile_cache
            .lock()
            .expect("embed_compile_cache mutex poisoned");
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                target_position,
            ) {
                Ok(cls) => {
                    cache.insert(key, cls);
                }
                Err(_) => {
                    drop(cache);
                    return crate::model::forward_for_embedding(
                        &self.cfg,
                        &self.weights,
                        token_ids,
                        target_position,
                    );
                }
            }
        }
        {
            let mut stats = self
                .embed_compile_stats
                .lock()
                .expect("embed_compile_stats mutex poisoned");
            if was_present {
                stats.single_hits += 1;
            } else {
                stats.single_misses += 1;
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        let outputs = cls.apply(&[&hidden]);
        outputs
            .into_iter()
            .next()
            .expect("compiled embedding closure must return one output")
    }

    /// Batched version of `embedding_forward`. Same compile-cache strategy,
    /// keyed on `(batch_size, max_len, target_positions)`. Mean pooling
    /// (`target_positions = None`) currently falls back to the imperative
    /// path because Mean pools after the closure result is materialized.
    fn embedding_batch_forward(
        &self,
        batch_token_ids: &[Vec<u32>],
        target_positions: Option<&[usize]>,
    ) -> (MlxArray, Vec<usize>) {
        let disable_compile = std::env::var("AX_EMBED_NO_COMPILE").is_ok();
        if disable_compile || target_positions.is_none() {
            // Mean pooling needs the full [B, max_seq, H] hidden; the closure
            // body is only built for the Last/Cls (extract-then-norm) shape.
            return crate::model::forward_for_embedding_batch(
                &self.cfg,
                &self.weights,
                batch_token_ids,
                target_positions,
            );
        }
        let (hidden, batch, max_len, actual_lens) = crate::model::build_embedding_batch_hidden_pub(
            &self.cfg,
            &self.weights,
            batch_token_ids,
        );
        let target_positions_vec: Vec<usize> = target_positions.expect("checked above").to_vec();
        let key: EmbedBatchCompileKey = (batch, max_len, Some(target_positions_vec.clone()));
        let mut cache = self
            .embed_batch_compile_cache
            .lock()
            .expect("embed_batch_compile_cache mutex poisoned");
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_batch_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                Some(target_positions_vec.clone()),
            ) {
                Ok(cls) => {
                    cache.insert(key.clone(), cls);
                }
                Err(_) => {
                    drop(cache);
                    return crate::model::forward_for_embedding_batch(
                        &self.cfg,
                        &self.weights,
                        batch_token_ids,
                        target_positions,
                    );
                }
            }
        }
        {
            let mut stats = self
                .embed_compile_stats
                .lock()
                .expect("embed_compile_stats mutex poisoned");
            if was_present {
                stats.batched_hits += 1;
            } else {
                stats.batched_misses += 1;
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        let outputs = cls.apply(&[&hidden]);
        let out = outputs
            .into_iter()
            .next()
            .expect("compiled batched embedding closure must return one output");
        (out, actual_lens)
    }

    fn run_item(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
        model_id: &str,
        block_size_tokens: u32,
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
                decode_profile: DecodeProfileSnapshot::default(),
                kv_usage: MlxKVCacheUsage::default(),
                prefix_cache: MlxPrefixCacheTelemetry::default(),
                kv_compression_shadow_sync_wall_us: None,
            };
        }

        let max_output = ctx.map(|c| c.max_output_tokens).unwrap_or(1);
        let generated_len = ctx.map(|c| c.generated_len).unwrap_or(0);
        let prefill_completes_prompt = prefill_item_completes_prompt(item, ctx);
        let is_prefill = matches!(item.mode, ExecutionMode::Prefill);
        let sampling = ctx
            .map(|c| {
                MlxSamplingParams::new(c.temperature, c.top_p, c.top_k)
                    .with_repetition_penalty(c.repetition_penalty, c.repetition_context_size)
            })
            .unwrap_or_default();
        let is_greedy = ctx
            .map(|c| c.deterministic_argmax_sampling)
            .unwrap_or(sampling == MlxSamplingParams::greedy());

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
        let mut prefix_cache = self.restore_reused_prefix_state(
            &mut state,
            item,
            ctx,
            model_id,
            block_size_tokens,
            sampling,
        );

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
                let full_recompute_tokens = full_prefill_recompute_tokens_for_warmup_fallback(
                    item,
                    token_ids,
                    &prefix_cache,
                    &state,
                );
                let prefill_tokens = full_recompute_tokens.as_deref().unwrap_or(token_ids);
                if full_recompute_tokens.is_some() {
                    state.cache.reset();
                    state.prompt_prefix_tokens.clear();
                    state.cached_prefill_output_token = None;
                }
                let repetition_history = state.repetition_history(prefill_tokens, sampling);
                let prefill_forward_started = Instant::now();
                let tok = chunked_prefill(
                    &self.cfg,
                    &self.weights,
                    prefill_tokens,
                    &mut state.cache,
                    self.prefill_chunk,
                    MlxSamplingRequest::new(sampling, &repetition_history),
                    &mut state.rng,
                );
                let prefill_forward_wall_us = elapsed_us(prefill_forward_started);
                let prefill_token_count = prefill_tokens.len();
                if let Some(tokens) = full_recompute_tokens {
                    state.prompt_prefix_tokens = tokens;
                } else {
                    extend_prompt_prefix_tokens(&mut state, item, token_ids);
                }
                let prefix_cache_started = Instant::now();
                prefix_cache.merge_from(
                    self.store_prompt_prefix_snapshots(
                        model_id,
                        block_size_tokens,
                        &state,
                        prefill_completes_prompt
                            .then_some(tok)
                            .filter(|_| is_greedy),
                    ),
                );
                let prefill_prefix_cache_wall_us = elapsed_us(prefix_cache_started);
                let mut prefill_generation_state_wall_us = 0;
                if prefill_completes_prompt {
                    let generation_state_started = Instant::now();
                    kv_compression_shadow_sync_wall_us = self.initialize_generation_state(
                        &mut state,
                        max_output,
                        kv_compression_layer_eligible,
                    );
                    prefill_generation_state_wall_us = elapsed_us(generation_state_started);
                }

                // Each non-final chunk in chunked_prefill calls async_eval; only the
                // last chunk calls a blocking eval.  Compute counts from prompt length.
                let drain_count =
                    prefill_token_count.saturating_sub(1) as u32 / self.prefill_chunk as u32;
                state
                    .decode_telemetry
                    .record_prefill_drain_async_evals(drain_count);
                state.decode_telemetry.record_prefill_eval_barrier();
                state
                    .decode_telemetry
                    .record_prefill(elapsed_us(prefill_started));
                state.decode_telemetry.record_prefill_breakdown(
                    prefill_forward_wall_us,
                    prefill_prefix_cache_wall_us,
                    prefill_generation_state_wall_us,
                );
                prefill_completes_prompt.then_some(tok)
            }
            ExecutionMode::Decode => {
                let decode_started = Instant::now();
                let tok = if generated_len == 0 {
                    if let Some(tok) = state.cached_prefill_output_token.take() {
                        kv_compression_shadow_sync_wall_us = self.initialize_generation_state(
                            &mut state,
                            max_output,
                            kv_compression_layer_eligible,
                        );
                        tok
                    } else {
                        self.decode_one(&mut state, token_ids, sampling, is_greedy)
                    }
                } else {
                    self.decode_one(&mut state, token_ids, sampling, is_greedy)
                };
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
        if let Some(sampled_token) = sampled_token {
            state.generated_tokens.push(sampled_token);
        }

        // Re-insert state only if the request continues — lock held briefly.
        let ngram_acceleration = state.ngram_acceleration;
        let decode_telemetry = state.decode_telemetry;
        let gemma4_moe_profile = take_gemma4_moe_profile_snapshot();
        let linear_attention_profile = take_linear_attention_profile_snapshot();
        state
            .decode_profile
            .merge_from(take_decode_profile_snapshot());
        let decode_profile = state.decode_profile;
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
        kv_usage
            .kv_compression
            .apply_decode_usage(turboquant_decode_usage);
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
            decode_profile,
            kv_usage,
            prefix_cache,
            kv_compression_shadow_sync_wall_us,
        }
    }

    fn prefix_cache_supported(&self) -> bool {
        // Every native-tier architecture (standard FA, linear, sliding-window,
        // MLA) is now supported via the unified store-side restriction in
        // `store_prompt_prefix_snapshots`: non-FA architectures store only
        // the full-prompt snapshot when the prompt is exactly block-aligned.
        // Lookups remain exact-match-safe via `MlxPrefixCache::get`'s
        // token-equality check.
        let _ = self; // architecture gating now happens inside the store path.
        true
    }

    fn prefix_cache_route_policy(&self) -> String {
        let decode_policy = if self.disable_ngram_acceleration {
            "direct"
        } else {
            "ngram"
        };
        format!(
            "{decode_policy};kv_mode={:?};kv_preset={:?};hot={};min={}",
            self.kv_compression.mode,
            self.kv_compression.preset,
            self.kv_compression.hot_window_tokens,
            self.kv_compression.min_context_tokens
        )
    }

    fn prefix_cache_layer_layout(&self) -> String {
        format!("layers={};full_attention_only", self.cfg.layer_count)
    }

    fn prefix_cache_key(
        &self,
        model_id: &str,
        block_size_tokens: u32,
        tokens: &[u32],
    ) -> MlxPrefixCacheKey {
        MlxPrefixCacheKey {
            model_id: model_id.to_string(),
            route_policy: self.prefix_cache_route_policy(),
            layer_layout: self.prefix_cache_layer_layout(),
            block_size_tokens,
            token_count: saturating_u32(tokens.len()),
            token_hash: hash_prefix_tokens(tokens),
        }
    }

    fn longest_block_aligned_prefix_by_probe<F>(
        block_size_tokens: u32,
        input: &[u32],
        mut has_snapshot: F,
    ) -> Option<Vec<u32>>
    where
        F: FnMut(&[u32]) -> bool,
    {
        let block_size = block_size_tokens as usize;
        if block_size == 0 || input.len() < block_size {
            return None;
        }
        let mut prefix_len = (input.len() / block_size) * block_size;
        while prefix_len >= block_size {
            let prefix = &input[..prefix_len];
            if has_snapshot(prefix) {
                return Some(prefix.to_vec());
            }
            prefix_len -= block_size;
        }
        None
    }

    /// Probe the runner-side snapshot cache for the longest block-aligned
    /// prefix of `input` that has a stored entry, returning that prefix as
    /// a fresh `Vec<u32>`. Used when the scheduler did not annotate
    /// `reused_prefix_token_slice` (e.g. iterative-chat turn 3+ where the
    /// scheduler's per-request block table no longer tracks the original
    /// prompt, but the runner-side snapshot from turn 1 is still resident).
    ///
    /// Returns `None` when no aligned prefix hits, when `input` is shorter
    /// than one block, or when the cache is disabled. The probe is O(input.len() /
    /// block_size) hash-map lookups, all read-only, and verifies exact token
    /// equality so a longer hash collision cannot hide a shorter valid prefix.
    fn probe_runner_snapshot_for_prefix(
        &self,
        model_id: &str,
        block_size_tokens: u32,
        input: &[u32],
    ) -> Option<Vec<u32>> {
        let cache = self.prefix_cache.lock().unwrap();
        Self::longest_block_aligned_prefix_by_probe(block_size_tokens, input, |prefix| {
            let key = self.prefix_cache_key(model_id, block_size_tokens, prefix);
            cache.contains_exact_tokens(&key, prefix)
        })
    }

    fn restore_reused_prefix_state(
        &self,
        state: &mut RequestState,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
        model_id: &str,
        block_size_tokens: u32,
        sampling: MlxSamplingParams,
    ) -> MlxPrefixCacheTelemetry {
        let mut telemetry = MlxPrefixCacheTelemetry::default();
        // Scheduler annotation comes from `ax-engine-core`'s prefix-lookup
        // table, which is keyed on the scheduler-side block table. That
        // table can disagree with the runner-side `MlxPrefixCache` in two
        // ways: (a) it can be empty for a fresh request even when the
        // runner's cache still holds a valid snapshot from an earlier
        // request, and (b) it can over-report — claim `reused_tokens.len()`
        // larger than any snapshot the runner actually stored, because the
        // scheduler tracks logical block reuse cumulatively across turns
        // while the runner stored only the original prompt.
        //
        // Probe the runner-side cache to find the longest block-aligned
        // prefix that is *actually* in the snapshot map, capped at the
        // scheduler's annotation when one exists (so the probe never
        // claims more tokens than core does). Cache `get` below still
        // bit-equality-checks the tokens, so a wrong probe cannot produce
        // a stale restore.
        let probe_upper_bound = if !item.reused_prefix_token_slice.is_empty() {
            &item.reused_prefix_token_slice[..]
        } else {
            &item.input_token_slice[..]
        };
        let probed_tokens: Vec<u32> = if state.cache.seq_len == 0
            && !probe_upper_bound.is_empty()
            && matches!(item.mode, ExecutionMode::Prefill)
        {
            self.probe_runner_snapshot_for_prefix(model_id, block_size_tokens, probe_upper_bound)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let reused_tokens: &[u32] = if !probed_tokens.is_empty() {
            &probed_tokens
        } else {
            &item.reused_prefix_token_slice
        };
        if reused_tokens.is_empty() || state.cache.seq_len != 0 {
            return telemetry;
        }
        let capture_prefill_output =
            item.mode == ExecutionMode::Decode && ctx.is_some_and(|ctx| ctx.generated_len == 0);
        let defer_prefill_warmup = item.mode == ExecutionMode::Prefill;

        if !self.prefix_cache_supported() {
            telemetry.record_blocked_unsupported_layout();
            if !defer_prefill_warmup {
                self.warm_reused_prefix_without_cache(
                    state,
                    item.request_id,
                    reused_tokens,
                    sampling,
                    capture_prefill_output,
                );
            }
            telemetry.warmup_tokens = telemetry
                .warmup_tokens
                .saturating_add(saturating_u32(reused_tokens.len()));
            return telemetry;
        }

        if !self.prefix_cache.lock().unwrap().enabled() {
            telemetry.record_blocked_policy_disabled();
            if !defer_prefill_warmup {
                self.warm_reused_prefix_without_cache(
                    state,
                    item.request_id,
                    reused_tokens,
                    sampling,
                    capture_prefill_output,
                );
            }
            telemetry.warmup_tokens = telemetry
                .warmup_tokens
                .saturating_add(saturating_u32(reused_tokens.len()));
            return telemetry;
        }

        let key = self.prefix_cache_key(model_id, block_size_tokens, reused_tokens);
        let hit = {
            let mut cache = self.prefix_cache.lock().unwrap();
            let hit = cache.get(&key, reused_tokens);
            telemetry.record_stats(cache.stats());
            hit
        };

        // MLA's compressed-latent KV survives a snapshot+restore cleanly for the
        // same-prompt (warm_repeat / Decode-mode) case — the entire forward pass
        // runs against a restored cache and never crosses a non-zero-seq_len
        // chunk boundary. But for warm_extend (Prefill-mode with a suffix), the
        // post-restore `chunked_prefill` must process the suffix starting from
        // `seq_len == base_len`, and the resulting forward+attention math drifts
        // fp-wise against a cold `chunked_prefill` over `base + suffix` from
        // seq_len=0. This was reproduced on GLM-4.7-Flash via the equivalence
        // harness (`verify_prefix_reuse_equivalence.py --mode warm_extend
        // --pad-to-block-size 16`): p2_medium_explain diverges at decode token
        // idx=13 with `ax_mlx_prefix_cache_hits=1` and `warmup_tokens=0`. The
        // safe behavior is to treat the snapshot as unusable for MLA + Prefill,
        // surfacing it as `blocked_unsupported_layout` so cee4227e's full-prompt
        // recompute path (`full_prefill_recompute_tokens_for_warmup_fallback`)
        // takes over. warm_repeat hits remain unaffected.
        let mla_extend_unsafe =
            self.cfg.glm_mla_attention.is_some() && item.mode == ExecutionMode::Prefill;

        if let Some(snapshot) = hit {
            if !mla_extend_unsafe {
                state.cache = snapshot.cache.clone();
                state.prompt_prefix_tokens = reused_tokens.to_vec();
                state.cached_prefill_output_token = snapshot.greedy_prefill_output_token;
                telemetry.hits = telemetry.hits.saturating_add(1);
                telemetry.reused_tokens = telemetry
                    .reused_tokens
                    .saturating_add(saturating_u32(snapshot.token_count));
                return telemetry;
            }
            telemetry.record_blocked_unsupported_layout();
            telemetry.warmup_tokens = telemetry
                .warmup_tokens
                .saturating_add(saturating_u32(reused_tokens.len()));
            return telemetry;
        }

        telemetry.misses = telemetry.misses.saturating_add(1);
        if !defer_prefill_warmup {
            self.warm_reused_prefix_without_cache(
                state,
                item.request_id,
                reused_tokens,
                sampling,
                capture_prefill_output,
            );
        }
        telemetry.warmup_tokens = telemetry
            .warmup_tokens
            .saturating_add(saturating_u32(reused_tokens.len()));
        telemetry
    }

    fn warm_reused_prefix_without_cache(
        &self,
        state: &mut RequestState,
        request_id: RequestId,
        tokens: &[u32],
        sampling: MlxSamplingParams,
        capture_prefill_output: bool,
    ) {
        let mut warmup_rng = if capture_prefill_output {
            state.rng
        } else {
            Xorshift64::new(request_id.0 ^ 0xA5A5_5A5A_F00D_CAFE)
        };
        let repetition_history = if capture_prefill_output {
            state.repetition_history(tokens, sampling)
        } else {
            Vec::new()
        };
        let prefill_output_token = chunked_prefill(
            &self.cfg,
            &self.weights,
            tokens,
            &mut state.cache,
            self.prefill_chunk,
            MlxSamplingRequest::new(
                if capture_prefill_output {
                    sampling
                } else {
                    MlxSamplingParams::greedy()
                },
                &repetition_history,
            ),
            &mut warmup_rng,
        );
        if capture_prefill_output {
            state.rng = warmup_rng;
            state.cached_prefill_output_token = Some(prefill_output_token);
        }
        state.prompt_prefix_tokens = tokens.to_vec();
    }

    fn store_prompt_prefix_snapshots(
        &self,
        model_id: &str,
        block_size_tokens: u32,
        state: &RequestState,
        greedy_prefill_output_token: Option<u32>,
    ) -> MlxPrefixCacheTelemetry {
        let mut telemetry = MlxPrefixCacheTelemetry::default();
        if block_size_tokens == 0 || state.prompt_prefix_tokens.is_empty() {
            return telemetry;
        }
        if !self.prefix_cache_supported() {
            telemetry.record_blocked_unsupported_layout();
            return telemetry;
        }
        if !self.prefix_cache.lock().unwrap().enabled() {
            telemetry.record_blocked_policy_disabled();
            return telemetry;
        }

        let block_size = block_size_tokens as usize;
        let available_tokens = state.prompt_prefix_tokens.len().min(state.cache.seq_len);
        let full_block_tokens = available_tokens - (available_tokens % block_size);
        if full_block_tokens == 0 {
            return telemetry;
        }

        // Non-standard-FA architectures (linear attention, sliding window, MLA)
        // share the same store constraint: only the full-prompt snapshot is
        // sound, and only when the prompt is exactly block-aligned (so
        // `trim_to(full_block_tokens) == seq_len` is a no-op).
        //   - Linear: `trim_to` does not roll back recurrent state.
        //   - Sliding-window: `trim_to` returns false once any rotating-window
        //     layer has rotated past `prefix_len`.
        //   - MLA: trim_to itself is sound for MLA buffers, but the warmup
        //     re-prefill path has observed fp-drift on this architecture
        //     (slice 6 baseline harness: GLM-4.7 warm_repeat 3/5 PASS through
        //     warmup), so routing through the bit-exact snapshot path for
        //     aligned prompts both delivers TTFT speedup AND sidesteps that
        //     pre-existing warmup correctness issue for the same-prompt case.
        // The `verify_prefix_reuse_equivalence.py` harness fails-closed on
        // any resulting token drift; any future change here must keep that
        // harness green on every model in the supported tier.
        let linear_attention = self.cfg.linear_attention.is_some();
        let sliding_window = self.kv_layer_windows.iter().any(Option::is_some);
        let glm_mla_attention = self.cfg.glm_mla_attention.is_some();
        let alignment_restricted = linear_attention || sliding_window || glm_mla_attention;
        if alignment_restricted && full_block_tokens != available_tokens {
            telemetry.record_blocked_trim_failure();
            return telemetry;
        }
        let snapshot_start_tokens = if alignment_restricted {
            full_block_tokens
        } else {
            block_size
        };

        let mut cache = self.prefix_cache.lock().unwrap();
        for prefix_len in (snapshot_start_tokens..=full_block_tokens).step_by(block_size) {
            let tokens = &state.prompt_prefix_tokens[..prefix_len];
            let key = self.prefix_cache_key(model_id, block_size_tokens, tokens);
            let mut snapshot_cache = state.cache.clone();
            if !snapshot_cache.trim_to(prefix_len) {
                telemetry.record_blocked_trim_failure();
                continue;
            }
            let usage = snapshot_cache.usage_snapshot_with_layer_windows(&self.kv_layer_windows);
            let outcome = cache.insert(
                key,
                MlxPrefixSnapshot {
                    cache: snapshot_cache,
                    tokens: tokens.to_vec(),
                    token_count: prefix_len,
                    bytes: usage.logical_bytes,
                    greedy_prefill_output_token: (prefix_len == available_tokens)
                        .then_some(greedy_prefill_output_token)
                        .flatten(),
                },
            );
            if outcome.stored {
                telemetry.stores = telemetry.stores.saturating_add(1);
            }
            telemetry.evictions = telemetry.evictions.saturating_add(outcome.evictions);
        }
        telemetry.record_stats(cache.stats());
        telemetry
    }

    /// Produce one output token for a decode step.
    ///
    /// Pops from the bonus queue when pre-verified tokens are available.
    /// Uses the double-buffer direct pipeline when `disable_ngram_acceleration = true` and
    /// greedy argmax sampling (bootstrapped during prefill).
    /// Otherwise runs an n-gram accelerated or single-token decode pass.
    fn decode_one(
        &self,
        state: &mut RequestState,
        input_tokens: &[u32],
        sampling: MlxSamplingParams,
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
        //
        // The pipeline is bootstrapped lazily on the first decode call rather
        // than inside initialize_generation_state, so that the prefill runner
        // step returns (and the first SSE event fires) without waiting for the
        // decode graph construction.
        if self.disable_ngram_acceleration && is_greedy {
            if state.pending_direct.is_some() {
                return self.run_direct_pipeline_continue(state);
            }
            let last_token = state
                .next_model_last_token
                .or_else(|| input_tokens.last().copied())
                .unwrap_or(0);
            return self.run_direct_pipeline_bootstrap(state, last_token);
        }

        let last_token = state
            .next_model_last_token
            .or_else(|| input_tokens.last().copied())
            .unwrap_or(0);

        let result = self.run_model_decode(state, last_token, sampling, is_greedy);
        apply_decode_result(state, &result, &self.terminal_token_ids)
    }

    /// Decode one deterministic token on the direct double-buffer pipeline.
    ///
    /// Used both by explicit direct mode and by request-local n-gram fallback after
    /// a linear-attention request proves it has no useful draft support.  The
    /// pipeline may keep the cache one lazy token ahead, so callers must continue
    /// using this path until the request finishes.
    fn run_direct_pipeline_decode(&self, state: &mut RequestState, last_token: u32) -> u32 {
        let tok = match direct_pipeline_action(state.pending_direct.is_some()) {
            DirectPipelineAction::ContinuePending => self.run_direct_pipeline_continue(state),
            DirectPipelineAction::Bootstrap => {
                self.run_direct_pipeline_bootstrap(state, last_token)
            }
        };
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
        let bootstrap_started = Instant::now();
        let turboquant_context = self.turboquant_model_decode_context();
        let bootstrap_token = start_direct_pipeline_with_turboquant_context(
            &self.cfg,
            &self.weights,
            last_token,
            &mut state.cache,
            turboquant_context.as_ref(),
        );
        state
            .decode_telemetry
            .record_direct_bootstrap(elapsed_us(bootstrap_started));
        self.run_direct_pipeline_once(state, bootstrap_token)
    }

    fn run_direct_pipeline_once(&self, state: &mut RequestState, bootstrap_token: MlxArray) -> u32 {
        let branch_started = Instant::now();
        let op_count_before = mlx_sys::op_count_snapshot();
        let turboquant_context = self.turboquant_model_decode_context();
        let advanced = advance_direct_pipeline_with_timings_and_turboquant_context(
            &self.cfg,
            &self.weights,
            &bootstrap_token,
            &mut state.cache,
            turboquant_context.as_ref(),
        );
        let op_count_delta = mlx_sys::op_count_take(op_count_before);
        state
            .decode_telemetry
            .record_direct_pipeline(elapsed_us(branch_started));
        state
            .decode_telemetry
            .record_direct_pipeline_op_count(op_count_delta);
        state
            .decode_telemetry
            .record_direct_pipeline_timings(advanced.timings);
        state.decode_telemetry.record_production_decode_eval();
        state.pending_direct = Some(advanced.next_pending);
        self.maybe_clear_direct_pipeline_cache(state);
        advanced.token
    }

    fn run_request_disabled_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        is_greedy: bool,
    ) -> Vec<u32> {
        state.ngram_acceleration.record_request_disabled_step();
        state
            .ngram_acceleration
            .record_request_disabled_reason(state.ngram_request_disable_reason);
        if is_greedy {
            vec![self.run_direct_pipeline_decode(state, last_token)]
        } else {
            self.run_single_decode(state, last_token, sampling)
        }
    }

    fn run_no_draft_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        has_linear_attention: bool,
        is_greedy: bool,
        rejection: Option<NgramDraftRejection>,
    ) -> Option<Vec<u32>> {
        state.ngram_acceleration.record_no_draft();
        state.ngram_acceleration.record_no_draft_reason(rejection);
        if has_linear_attention {
            state.linear_ngram_no_draft_streak =
                state.linear_ngram_no_draft_streak.saturating_add(1);
            if is_greedy && linear_ngram_no_draft_should_disable(state.linear_ngram_no_draft_streak)
            {
                state.ngram_acceleration_disabled_for_request = true;
                state.ngram_request_disable_reason = NgramRequestDisableReason::LinearNoDraft;
                state.ngram_acceleration.record_request_disable_event();
                return Some(
                    self.run_request_disabled_decode(state, last_token, sampling, is_greedy),
                );
            }
            if is_greedy {
                state.ngram_disabled_steps = LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL;
                state
                    .ngram_acceleration
                    .record_cooldown_event(LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL);
            }
        }
        None
    }

    fn run_non_ngram_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        is_greedy: bool,
    ) -> Option<Vec<u32>> {
        if self.disable_ngram_acceleration {
            return Some(self.run_single_decode(state, last_token, sampling));
        }

        if sampling.uses_repetition_penalty() {
            return Some(self.run_single_decode(state, last_token, sampling));
        }

        if state.ngram_acceleration_disabled_for_request {
            return Some(self.run_request_disabled_decode(state, last_token, sampling, is_greedy));
        }

        // N-gram acceleration disabled: count down and use single decode.
        if state.ngram_disabled_steps > 0 {
            state.ngram_disabled_steps -= 1;
            state.ngram_acceleration.record_cooldown_step();
            return Some(self.run_single_decode(state, last_token, sampling));
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
        sampling: MlxSamplingParams,
    ) -> Vec<u32> {
        let branch_started = Instant::now();
        let turboquant_context = self.turboquant_model_decode_context();
        let repetition_history = state.repetition_history(&[], sampling);
        let result = single_decode_with_turboquant_context(
            &self.cfg,
            &self.weights,
            &mut state.cache,
            &mut state.ngram,
            last_token,
            sampling,
            &repetition_history,
            &mut state.rng,
            turboquant_context.as_ref(),
        );
        state
            .decode_telemetry
            .record_single_decode(elapsed_us(branch_started));
        state.decode_telemetry.record_production_decode_eval();
        result
    }

    fn initialize_generation_state(
        &self,
        state: &mut RequestState,
        max_output: u32,
        layer_eligible: Option<&[bool]>,
    ) -> Option<u32> {
        // Seed the n-gram table with the tail of the prompt.
        // Only the last NGRAM_PROMPT_FEED_MAX tokens are fed: long prompts
        // (e.g. random-token benchmarks with 512+ tokens) would otherwise inject
        // hundreds of useless bigrams, causing a false-positive spec attempt on
        // the first decode step and disabling n-gram acceleration for
        // LINEAR_NGRAM_RETRY_INTERVAL steps — wiping out most of the generation.
        seed_generation_ngram_from_prompt(state);

        // Classify the full prompt once per generation. record_prompt_class is
        // max-merge friendly so re-entry from an unusual code path cannot
        // downgrade an already-set class.
        let prompt_class = classify_prompt_class(&state.prompt_prefix_tokens);
        state.ngram_acceleration.record_prompt_class(prompt_class);

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
        state.ngram_request_disable_reason = if state.ngram_acceleration_disabled_for_request {
            NgramRequestDisableReason::ShortOutputBudget
        } else {
            NgramRequestDisableReason::None
        };

        let kv_compression_shadow_sync_wall_us =
            self.sync_turboquant_shadow_storage_if_needed(state, true, layer_eligible);

        // The direct pipeline bootstrap (start_direct_pipeline_with_turboquant_context)
        // is deferred to the first decode call in decode_one, so that prefill runner
        // steps return before the decode graph is constructed.  This keeps the
        // bootstrap's CPU graph-build time out of the TTFT measurement.

        kv_compression_shadow_sync_wall_us
    }

    /// Run one model decode step, updating the n-gram accept-rate gate.
    fn run_model_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        is_greedy: bool,
    ) -> Vec<u32> {
        let has_linear_attention = self.cfg.linear_attention.is_some();
        if let Some(result) = self.run_non_ngram_decode(state, last_token, sampling, is_greedy) {
            return result;
        }

        let draft_outcome = ngram_acceleration_draft(
            &state.ngram,
            has_linear_attention,
            state.ngram_posterior_mean(),
            self.ngram_policy_variant,
        );
        state
            .ngram_acceleration
            .record_policy(self.ngram_policy_variant, draft_outcome.requested_max_len);
        let NgramDraftOutcome {
            draft, rejection, ..
        } = draft_outcome;
        if draft.is_empty() {
            if let Some(result) = self.run_no_draft_decode(
                state,
                last_token,
                sampling,
                has_linear_attention,
                is_greedy,
                rejection,
            ) {
                return result;
            }
            return self.run_single_decode(state, last_token, sampling);
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
            sampling,
            &mut state.rng,
        );
        state
            .decode_telemetry
            .record_ngram_decode(elapsed_us(branch_started));
        state.decode_telemetry.record_production_decode_eval();

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

fn hash_prefix_tokens(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in tokens {
        hash ^= u64::from(*token);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash ^ (tokens.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn extend_prompt_prefix_tokens(
    state: &mut RequestState,
    item: &ax_engine_core::ExecutionItem,
    token_ids: &[u32],
) {
    let expected_start = item.position_range.start as usize;
    if state.prompt_prefix_tokens.len() > expected_start {
        state.prompt_prefix_tokens.truncate(expected_start);
    } else if state.prompt_prefix_tokens.len() < expected_start {
        state.prompt_prefix_tokens = item.reused_prefix_token_slice.clone();
    }
    state.prompt_prefix_tokens.extend_from_slice(token_ids);
}

fn full_prefill_recompute_tokens_for_warmup_fallback(
    item: &ax_engine_core::ExecutionItem,
    token_ids: &[u32],
    prefix_cache: &MlxPrefixCacheTelemetry,
    state: &RequestState,
) -> Option<Vec<u32>> {
    if item.mode != ExecutionMode::Prefill
        || item.reused_prefix_token_slice.is_empty()
        || prefix_cache.warmup_tokens == 0
        || state.cache.seq_len != 0
    {
        return None;
    }

    let mut tokens = Vec::with_capacity(
        item.reused_prefix_token_slice
            .len()
            .saturating_add(token_ids.len()),
    );
    tokens.extend_from_slice(&item.reused_prefix_token_slice);
    tokens.extend_from_slice(token_ids);
    Some(tokens)
}

struct MlxItemRun {
    update: RequestExecutionUpdate,
    ngram_acceleration: NgramAccelerationTelemetry,
    decode_telemetry: DecodeTelemetry,
    gemma4_moe_profile: Gemma4MoeProfileSnapshot,
    linear_attention_profile: LinearAttentionProfileSnapshot,
    decode_profile: DecodeProfileSnapshot,
    kv_usage: MlxKVCacheUsage,
    prefix_cache: MlxPrefixCacheTelemetry,
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

fn ngram_policy_variant_from_env() -> NgramPolicyVariant {
    match std::env::var("AX_MLX_NGRAM_POLICY")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .replace('_', "-")
        .as_str()
    {
        "llama-map" | "llama" | "latest" => NgramPolicyVariant::LlamaMapLatest,
        "shared-pool" | "shared" => NgramPolicyVariant::SharedPoolMajority,
        _ => NgramPolicyVariant::MajorityRecency,
    }
}

fn ngram_acceleration_draft(
    ngram: &NgramTable,
    has_linear_attention: bool,
    posterior_mean: f32,
    variant: NgramPolicyVariant,
) -> NgramDraftOutcome {
    let max_len = adaptive_ngram_draft_len(has_linear_attention, posterior_mean);
    let confidence_threshold = effective_draft_confidence_threshold();
    let policy = if has_linear_attention {
        // Dense rollback is O(1); linear-attention partial-reject pays
        // branch/recompute, so cap at DEFAULT_DRAFT_LEN to bound recompute cost.
        NgramDraftPolicy {
            variant,
            max_len,
            min_support: LINEAR_MIN_NGRAM_SUPPORT,
            confidence_threshold,
        }
    } else {
        // Dense models extend up to MAX_DRAFT_LEN when the posterior and n-gram
        // chain are high-confidence; otherwise probe shorter drafts first.
        NgramDraftPolicy {
            variant,
            max_len,
            min_support: 1,
            confidence_threshold,
        }
    };
    ngram.predict_with_policy(policy)
}

fn adaptive_ngram_draft_len(has_linear_attention: bool, posterior_mean: f32) -> usize {
    if has_linear_attention {
        if posterior_mean < NGRAM_DRAFT_LEN_SHRINK_THRESHOLD {
            NGRAM_DRAFT_LEN_LOW_CONFIDENCE
        } else {
            DEFAULT_DRAFT_LEN
        }
    } else {
        // Dense models: always allow MAX_DRAFT_LEN and let the ngram confidence
        // gate in predict_with_policy prune the chain naturally. Pre-capping at
        // DEFAULT_DRAFT_LEN for mid-range posterior hurts throughput because it
        // discards valid chain extensions that the confidence gate would keep.
        MAX_DRAFT_LEN
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
    // `GlmRouterConfig::from_manifest` `.expect()`s these three fields once the router
    // is considered enabled (`is_enabled()` returns true if *any* field is set), and
    // `glm_router_apply_group_selection` follows up with runtime `assert!`s on the
    // group invariants. Surface every panic-source as a typed manifest error here.
    if manifest.glm_router.routed_scaling_factor.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router.routed_scaling_factor".to_string(),
        ));
    }
    let n_group = manifest.glm_router.n_group.ok_or_else(|| {
        MlxRunnerError::UnsupportedFeature("glm4_moe_lite requires glm_router.n_group".to_string())
    })?;
    let topk_group = manifest.glm_router.topk_group.ok_or_else(|| {
        MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router.topk_group".to_string(),
        )
    })?;
    if n_group == 0 {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite glm_router.n_group must be greater than zero".to_string(),
        ));
    }
    if topk_group == 0 || topk_group > n_group {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "glm4_moe_lite glm_router.topk_group {topk_group} must satisfy 0 < topk_group <= n_group ({n_group})"
        )));
    }
    // `NativeMoeConfig::is_enabled` (checked above) only requires that *some*
    // MoE field is present, but `ModelConfig::from_manifest` then
    // `unwrap_or(0)`s the missing ones. With `n_group > 1`,
    // `glm_router_apply_group_selection` asserts both divisibility and
    // `experts_per_group >= 2`, so a missing `expert_count` (decoded as 0)
    // would silently slip past the divisibility check and then crash on the
    // group-size assert. Require the fields explicitly here.
    let expert_count = manifest.moe.expert_count.ok_or_else(|| {
        MlxRunnerError::UnsupportedFeature("glm4_moe_lite requires moe.expert_count".to_string())
    })?;
    if manifest.moe.experts_per_token.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires moe.experts_per_token".to_string(),
        ));
    }
    if n_group > 1 {
        if expert_count % n_group != 0 {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "glm4_moe_lite moe.expert_count {expert_count} must be divisible by glm_router.n_group {n_group}"
            )));
        }
        if expert_count / n_group < 2 {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "glm4_moe_lite moe.expert_count {expert_count} divided by glm_router.n_group {n_group} must yield at least two experts per group"
            )));
        }
    }
    if first_dense_layers > manifest.layer_count {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "glm4_moe_lite glm_router.first_dense_layer_count {first_dense_layers} cannot exceed layer_count {}",
            manifest.layer_count
        )));
    }
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

    for token in COMMON_EOT_TOKEN_STRINGS {
        token_strings.insert((*token).to_string());
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
    let cfg = &manifest.linear_attention;
    let Some(key_head_dim) = cfg.key_head_dim else {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.key_head_dim must be configured".to_string(),
        ));
    };
    if key_head_dim % 32 != 0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "linear_attention.key_head_dim {key_head_dim} must be divisible by 32 for the MLX gated-delta kernel"
        )));
    }
    if cfg.num_value_heads.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.num_value_heads must be configured".to_string(),
        ));
    }
    if cfg.num_key_heads.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.num_key_heads must be configured".to_string(),
        ));
    }
    if cfg.value_head_dim.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.value_head_dim must be configured".to_string(),
        ));
    }
    if cfg.conv_kernel_dim.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.conv_kernel_dim must be configured".to_string(),
        ));
    }
    // `resolved_full_attention_interval` falls back to QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL
    // when the manifest omits the field, so None here means an explicit zero (or an
    // unsupported family that slipped past the model_family gate above). Reject zero
    // explicitly: `is_linear_layer` uses `is_multiple_of(interval)`, which would silently
    // treat every layer as linear when interval == 0.
    match cfg.resolved_full_attention_interval(&manifest.model_family) {
        Some(0) => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "linear_attention.full_attention_interval must be greater than zero".to_string(),
            ));
        }
        Some(_) => {}
        None => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "linear_attention.full_attention_interval must be configured".to_string(),
            ));
        }
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
    if has_sliding {
        match manifest.sliding_window_size {
            None => {
                return Err(MlxRunnerError::UnsupportedFeature(
                    "Gemma4 sliding_attention layers require sliding_window_size".to_string(),
                ));
            }
            Some(0) => {
                // build_layer_configs maps Some(0) to Some(0), and the cache path then
                // filters it back to None — sliding layers would silently degrade to a
                // grow-forever window. Reject up front instead of running with a layout
                // the user did not ask for.
                return Err(MlxRunnerError::UnsupportedFeature(
                    "Gemma4 sliding_window_size must be greater than zero".to_string(),
                ));
            }
            Some(_) => {}
        }
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
        // Chained KV sharing would panic at runtime in `MlxKVCache::peek_source_kv`
        // (the source layer never writes its own K/V, so the cached entry is None
        // and the `.expect("…source layer must appear earlier")` fires). Reject it
        // here so the manifest fails closed instead of producing a midstream panic.
        if manifest.kv_shared_source_layers.contains_key(&source) {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "Gemma4 KV-shared layer {layer} cannot use shared layer {source} as its source"
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

    fn test_prefix_key(token: u32) -> MlxPrefixCacheKey {
        MlxPrefixCacheKey {
            model_id: "model".into(),
            route_policy: "direct".into(),
            layer_layout: "layers=2;full_attention_only".into(),
            block_size_tokens: 4,
            token_count: 4,
            token_hash: hash_prefix_tokens(&[token; 4]),
        }
    }

    fn test_prefix_snapshot(token: u32, token_count: usize, bytes: u64) -> MlxPrefixSnapshot {
        MlxPrefixSnapshot {
            cache: MlxKVCache::new(2),
            tokens: vec![token; token_count],
            token_count,
            bytes,
            greedy_prefill_output_token: Some(7),
        }
    }

    #[test]
    fn prefix_cache_returns_exact_snapshot_and_updates_stats() {
        let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 1024,
            max_entries: 4,
        });
        let key = test_prefix_key(1);

        let outcome = cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));
        assert!(outcome.stored);
        assert_eq!(outcome.evictions, 0);

        let hit = cache
            .get(&key, &[1; 4])
            .expect("prefix snapshot should hit");
        assert_eq!(hit.token_count, 4);
        assert_eq!(hit.greedy_prefill_output_token, Some(7));
        assert_eq!(
            cache.stats(),
            MlxPrefixCacheStats {
                entries: 1,
                bytes: 128,
            }
        );
    }

    #[test]
    fn prefix_cache_hits_do_not_grow_lru_without_bound() {
        let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 1024,
            max_entries: 4,
        });
        let key = test_prefix_key(1);
        cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));

        for _ in 0..10_000 {
            assert!(cache.get(&key, &[1; 4]).is_some());
        }

        assert_eq!(cache.stats().entries, 1);
        assert!(
            cache.lru.len() <= cache.stale_lru_compaction_limit(),
            "stale LRU ticks should be compacted under repeated hits"
        );
    }

    #[test]
    fn prefix_cache_eviction_is_lru_and_visible() {
        let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 256,
            max_entries: 2,
        });
        let key1 = test_prefix_key(1);
        let key2 = test_prefix_key(2);
        let key3 = test_prefix_key(3);

        assert!(
            cache
                .insert(key1.clone(), test_prefix_snapshot(1, 4, 100))
                .stored
        );
        assert!(
            cache
                .insert(key2.clone(), test_prefix_snapshot(2, 4, 100))
                .stored
        );
        assert!(
            cache.get(&key1, &[1; 4]).is_some(),
            "key1 should become most recent"
        );
        let outcome = cache.insert(key3.clone(), test_prefix_snapshot(3, 4, 100));

        assert!(outcome.stored);
        assert_eq!(outcome.evictions, 1);
        assert!(cache.get(&key1, &[1; 4]).is_some());
        assert!(cache.get(&key2, &[2; 4]).is_none());
        assert!(cache.get(&key3, &[3; 4]).is_some());
    }

    #[test]
    fn prefix_cache_disabled_policy_does_not_store_snapshots() {
        let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 0,
            max_entries: 4,
        });
        let key = test_prefix_key(1);

        let outcome = cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));

        assert!(!cache.enabled());
        assert!(!outcome.stored);
        assert_eq!(outcome.evictions, 0);
        assert!(cache.get(&key, &[1; 4]).is_none());
        assert_eq!(
            cache.stats(),
            MlxPrefixCacheStats {
                entries: 0,
                bytes: 0,
            }
        );
    }

    #[test]
    fn prefix_cache_hash_collision_misses_without_reusing_snapshot() {
        let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 1024,
            max_entries: 4,
        });
        let key = test_prefix_key(1);

        let outcome = cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));
        assert!(outcome.stored);

        assert!(
            cache.get(&key, &[9; 4]).is_none(),
            "same hash key must still require exact prefix tokens"
        );
        assert_eq!(cache.stats().entries, 1);
        assert!(cache.get(&key, &[1; 4]).is_some());
    }

    #[test]
    fn prefix_cache_exact_membership_rejects_collision_tokens_without_touching_lru() {
        let mut cache = MlxPrefixCache::new(MlxPrefixCachePolicy {
            max_bytes: 1024,
            max_entries: 4,
        });
        let key = test_prefix_key(1);
        cache.insert(key.clone(), test_prefix_snapshot(1, 4, 128));

        assert!(cache.contains_exact_tokens(&key, &[1; 4]));
        assert!(!cache.contains_exact_tokens(&key, &[9; 4]));
        assert_eq!(
            cache.lru.len(),
            1,
            "read-only membership probe must not touch LRU state"
        );
    }

    #[test]
    fn prefix_probe_continues_past_longer_miss_to_shorter_match() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut probed_lengths = Vec::new();

        let prefix = MlxRunner::longest_block_aligned_prefix_by_probe(4, &input, |tokens| {
            probed_lengths.push(tokens.len());
            tokens == [1, 2, 3, 4]
        });

        assert_eq!(prefix, Some(vec![1, 2, 3, 4]));
        assert_eq!(
            probed_lengths,
            vec![8, 4],
            "probe must keep searching after a longer non-exact entry"
        );
    }

    #[test]
    fn prefix_cache_telemetry_merges_blocked_reason_counters() {
        let mut telemetry = MlxPrefixCacheTelemetry::default();
        telemetry.record_blocked_policy_disabled();
        telemetry.record_blocked_unsupported_layout();

        let mut other = MlxPrefixCacheTelemetry::default();
        other.record_blocked_trim_failure();
        other.record_blocked_unsupported_layout();
        telemetry.merge_from(other);

        assert_eq!(telemetry.blocked, 4);
        assert_eq!(telemetry.blocked_policy_disabled, 1);
        assert_eq!(telemetry.blocked_unsupported_layout, 2);
        assert_eq!(telemetry.blocked_trim_failure, 1);
    }

    #[test]
    fn prefix_cache_telemetry_writes_route_counters() {
        let telemetry = MlxPrefixCacheTelemetry {
            hits: 1,
            misses: 2,
            blocked: 3,
            blocked_policy_disabled: 1,
            blocked_unsupported_layout: 1,
            blocked_trim_failure: 1,
            stores: 4,
            evictions: 5,
            reused_tokens: 16,
            warmup_tokens: 8,
            entries: 2,
            bytes: 4096,
        };
        let mut decisions = Vec::new();

        telemetry.append_route_decisions(&mut decisions);

        assert!(decisions.contains(&("ax_mlx_prefix_cache_hits".into(), 1)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_misses".into(), 2)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked".into(), 3)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_policy_disabled".into(), 1)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_unsupported_layout".into(), 1)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_blocked_trim_failure".into(), 1)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_evictions".into(), 5)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_bytes_kib".into(), 4)));
    }

    #[test]
    fn prefill_warmup_fallback_recomputes_full_prompt_once() {
        let item = ax_engine_core::ExecutionItem {
            request_id: RequestId(21),
            mode: ExecutionMode::Prefill,
            input_token_slice: vec![5, 6],
            reused_prefix_token_slice: vec![1, 2, 3, 4],
            position_range: PositionRange {
                start: 4,
                end_exclusive: 6,
            },
            scheduled_token_count: 2,
            block_table_ref: RequestId(21),
            prefix_tokens_reused: 4,
            prefix_blocks_reused: 1,
        };
        let state = RequestState::new(2, RequestId(21));
        let telemetry = MlxPrefixCacheTelemetry {
            misses: 1,
            warmup_tokens: 4,
            ..MlxPrefixCacheTelemetry::default()
        };

        let tokens = full_prefill_recompute_tokens_for_warmup_fallback(
            &item,
            &item.input_token_slice,
            &telemetry,
            &state,
        )
        .expect("warmup fallback prefill should run prefix+suffix together");

        assert_eq!(tokens, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn prefill_warmup_fallback_does_not_recompute_cache_hits_or_decode() {
        let mut item = ax_engine_core::ExecutionItem {
            request_id: RequestId(22),
            mode: ExecutionMode::Prefill,
            input_token_slice: vec![5, 6],
            reused_prefix_token_slice: vec![1, 2, 3, 4],
            position_range: PositionRange {
                start: 4,
                end_exclusive: 6,
            },
            scheduled_token_count: 2,
            block_table_ref: RequestId(22),
            prefix_tokens_reused: 4,
            prefix_blocks_reused: 1,
        };
        let state = RequestState::new(2, RequestId(22));
        let hit_telemetry = MlxPrefixCacheTelemetry {
            hits: 1,
            reused_tokens: 4,
            ..MlxPrefixCacheTelemetry::default()
        };

        assert!(
            full_prefill_recompute_tokens_for_warmup_fallback(
                &item,
                &item.input_token_slice,
                &hit_telemetry,
                &state,
            )
            .is_none(),
            "snapshot hits already restored prefix KV and must prefill only the suffix",
        );

        item.mode = ExecutionMode::Decode;
        let warmup_telemetry = MlxPrefixCacheTelemetry {
            misses: 1,
            warmup_tokens: 4,
            ..MlxPrefixCacheTelemetry::default()
        };
        assert!(
            full_prefill_recompute_tokens_for_warmup_fallback(
                &item,
                &item.input_token_slice,
                &warmup_telemetry,
                &state,
            )
            .is_none(),
            "decode still needs a warmed prefix KV and optional prefill output token",
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
    fn generation_ngram_seed_uses_reconstructed_prompt_after_prefix_warmup() {
        let mut warm = RequestState::new(2, RequestId(11));
        warm.prompt_prefix_tokens = vec![10, 11, 12, 13, 10, 11, 12];

        seed_generation_ngram_from_prompt(&mut warm);

        assert_eq!(
            warm.ngram.predict(1),
            vec![13],
            "warm prefix+suffix prefill must seed n-grams from the reconstructed full prompt",
        );

        let mut suffix_only = RequestState::new(2, RequestId(12));
        suffix_only.prompt_prefix_tokens = vec![10, 11, 12];
        seed_generation_ngram_from_prompt(&mut suffix_only);

        assert!(
            suffix_only.ngram.predict(1).is_empty(),
            "feeding only the final prefill item loses the prompt context needed for deterministic warm_extend",
        );
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
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context_size: None,
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
            rms_norm_eps: None,
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
            weight_sanitize: ax_engine_core::WeightSanitize::None,
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

    #[test]
    fn terminal_token_ids_resolve_common_chatml_eot_from_tokenizer_json() {
        let mut manifest = dense_manifest();
        manifest.model_family = "qwen3_dense".to_string();
        set_vocab_size(&mut manifest, 200_000);
        let artifacts = write_artifacts(manifest);
        fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":151643,"content":"<|endoftext|>"},{"id":151645,"content":"<|im_end|>"}]}"#,
        )
        .expect("tokenizer should write");

        assert_eq!(resolve_terminal_token_ids(&artifacts), vec![151643, 151645]);
    }

    #[test]
    fn terminal_token_ids_resolve_common_gemma_eot_for_other_families() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma3".to_string();
        set_vocab_size(&mut manifest, 200_000);
        let artifacts = write_artifacts(manifest);
        fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":106,"content":"<end_of_turn>"},{"id":151645,"content":"<|im_end|>"}]}"#,
        )
        .expect("tokenizer should write");

        assert_eq!(resolve_terminal_token_ids(&artifacts), vec![106, 151645]);
    }

    #[test]
    fn terminal_token_ids_resolve_common_gemma4_turn_end_from_tokenizer_json() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        set_vocab_size(&mut manifest, 200_000);
        let artifacts = write_artifacts(manifest);
        fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":105,"content":"<|turn>"},{"id":106,"content":"<turn|>"}]}"#,
        )
        .expect("tokenizer should write");

        assert_eq!(resolve_terminal_token_ids(&artifacts), vec![106]);
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
    fn mlx_manifest_validation_rejects_partial_glm_router_fields() {
        // Each `.expect()` in `GlmRouterConfig::from_manifest` and each runtime
        // `assert!` in `glm_router_apply_group_selection` corresponds to one of
        // these checks. A partial manifest passed the prior validator (only
        // `first_dense_layer_count` was required) and then panicked downstream.
        type Mutator = Box<dyn Fn(&mut NativeModelManifest)>;
        for (label, mutate) in [
            (
                "missing routed_scaling_factor",
                Box::new(|m: &mut NativeModelManifest| m.glm_router.routed_scaling_factor = None)
                    as Mutator,
            ),
            (
                "missing n_group",
                Box::new(|m: &mut NativeModelManifest| m.glm_router.n_group = None),
            ),
            (
                "missing topk_group",
                Box::new(|m: &mut NativeModelManifest| m.glm_router.topk_group = None),
            ),
            (
                "zero n_group",
                Box::new(|m: &mut NativeModelManifest| m.glm_router.n_group = Some(0)),
            ),
            (
                "zero topk_group",
                Box::new(|m: &mut NativeModelManifest| m.glm_router.topk_group = Some(0)),
            ),
            (
                "topk_group exceeds n_group",
                Box::new(|m: &mut NativeModelManifest| {
                    m.glm_router.n_group = Some(2);
                    m.glm_router.topk_group = Some(3);
                    m.moe.expert_count = Some(4);
                }),
            ),
            (
                "expert_count not divisible by n_group",
                Box::new(|m: &mut NativeModelManifest| {
                    m.glm_router.n_group = Some(3);
                    m.glm_router.topk_group = Some(1);
                    m.moe.expert_count = Some(4);
                }),
            ),
            (
                "expert_count per group below two",
                Box::new(|m: &mut NativeModelManifest| {
                    // 4 experts / 4 groups = 1 per group; the runtime assert
                    // `experts_per_group >= 2` would fire mid-forward.
                    m.glm_router.n_group = Some(4);
                    m.glm_router.topk_group = Some(1);
                    m.moe.expert_count = Some(4);
                }),
            ),
            (
                "missing moe.expert_count",
                Box::new(|m: &mut NativeModelManifest| {
                    // `is_enabled()` stays true via experts_per_token, but
                    // `unwrap_or(0)` downstream then crashes the group-size assert
                    // when n_group > 1.
                    m.moe.expert_count = None;
                    m.glm_router.n_group = Some(2);
                    m.glm_router.topk_group = Some(1);
                }),
            ),
            (
                "missing moe.experts_per_token",
                Box::new(|m: &mut NativeModelManifest| m.moe.experts_per_token = None),
            ),
            (
                "first_dense_layer_count exceeds layer_count",
                Box::new(|m: &mut NativeModelManifest| {
                    m.glm_router.first_dense_layer_count = Some(m.layer_count + 1);
                }),
            ),
        ] {
            let mut manifest = glm4_moe_lite_manifest();
            mutate(&mut manifest);
            let error = validate_glm4_moe_lite_manifest(&manifest)
                .expect_err(&format!("{label} should fail closed"));
            let message = error.to_string();
            assert!(
                message.contains("glm_router") || message.contains("glm4_moe_lite"),
                "{label}: unexpected error message: {message}"
            );
        }
    }

    #[test]
    fn mlx_manifest_validation_allows_qwen35_linear_attention() {
        let artifacts = write_artifacts(qwen35_linear_manifest());

        validate_mlx_supported_manifest(&artifacts)
            .expect("Qwen3.5 linear attention is wired for the MLX path");
    }

    #[test]
    fn mlx_manifest_validation_rejects_partial_linear_attention_fields() {
        // Each `.expect()` in `LinearAttentionConfig::from_manifest` corresponds
        // to one of these required fields; the validator must surface a typed
        // error before the runner panics on a partially-configured manifest.
        for (label, mutate) in [
            (
                "missing num_value_heads",
                Box::new(|m: &mut NativeModelManifest| m.linear_attention.num_value_heads = None)
                    as Box<dyn Fn(&mut NativeModelManifest)>,
            ),
            (
                "missing num_key_heads",
                Box::new(|m: &mut NativeModelManifest| m.linear_attention.num_key_heads = None),
            ),
            (
                "missing value_head_dim",
                Box::new(|m: &mut NativeModelManifest| m.linear_attention.value_head_dim = None),
            ),
            (
                "missing conv_kernel_dim",
                Box::new(|m: &mut NativeModelManifest| m.linear_attention.conv_kernel_dim = None),
            ),
            (
                "zero full_attention_interval",
                Box::new(|m: &mut NativeModelManifest| {
                    m.linear_attention.full_attention_interval = Some(0)
                }),
            ),
        ] {
            let mut manifest = qwen35_linear_manifest();
            mutate(&mut manifest);
            let error = validate_qwen_gated_delta_linear_attention(&manifest)
                .expect_err(&format!("{label} should fail closed"));
            assert!(
                error.to_string().contains("linear_attention"),
                "{label}: unexpected error message: {error}"
            );
        }
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
        telemetry.record_request_disabled_reason(NgramRequestDisableReason::LinearNoDraft);
        telemetry.record_no_draft_reason(Some(NgramDraftRejection::NoCandidate));
        telemetry.record_no_draft_reason(Some(NgramDraftRejection::ConfidenceFiltered));
        telemetry.record_policy(NgramPolicyVariant::SharedPoolMajority, MAX_DRAFT_LEN);
        telemetry.record_prompt_class(crate::ngram_accel::PROMPT_CLASS_REPEATING);

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
        assert_eq!(
            decisions.get("ax_ngram_fallback_no_candidate_steps"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_ngram_fallback_confidence_filtered_steps"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_ngram_fallback_linear_no_draft_steps"),
            Some(&1)
        );
        assert_eq!(decisions.get("ax_ngram_policy_variant"), Some(&3));
        assert_eq!(decisions.get("ax_ngram_adaptive_draft_len_steps"), Some(&1));
        assert_eq!(
            decisions.get("ax_ngram_adaptive_draft_len_total"),
            Some(&(MAX_DRAFT_LEN as u32))
        );
        assert_eq!(
            decisions.get("ax_prompt_class_code"),
            Some(&crate::ngram_accel::PROMPT_CLASS_REPEATING)
        );

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
        assert_eq!(
            zero_decisions.get("ax_prompt_class_code"),
            Some(&crate::ngram_accel::PROMPT_CLASS_UNSET)
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
            projection_qkvz_wall_us: 70,
            projection_ba_wall_us: 30,
            projection_qkv_wall_us: 0,
            projection_z_wall_us: 0,
            projection_a_wall_us: 0,
            projection_b_wall_us: 0,
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
            projection_qkvz_wall_us: 105,
            projection_ba_wall_us: 45,
            projection_qkv_wall_us: 0,
            projection_z_wall_us: 0,
            projection_a_wall_us: 0,
            projection_b_wall_us: 0,
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
            decisions.get("ax_mlx_linear_attention_profile_projection_qkvz_wall_us"),
            Some(&175)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_profile_projection_ba_wall_us"),
            Some(&75)
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
    fn decode_profile_route_decisions_emit_only_when_enabled() {
        let mut profile = DecodeProfileSnapshot {
            enabled: 1,
            decode_steps: 64,
            layers: 1536,
            per_layer_input_wall_us: 800,
            pre_sdpa_wall_us: 1200,
            pre_sdpa_qkv_proj_wall_us: 700,
            pre_sdpa_qk_norm_wall_us: 200,
            pre_sdpa_rope_kv_wall_us: 300,
            sdpa_wall_us: 500,
            post_attn_wall_us: 2400,
            post_attn_ffn_wall_us: 1800,
            post_attn_output_proj_wall_us: 300,
            post_attn_residual_norm_wall_us: 100,
            post_attn_residual_gate_wall_us: 200,
            lm_head_wall_us: 150,
        };
        profile.merge_from(DecodeProfileSnapshot {
            enabled: 1,
            decode_steps: 64,
            layers: 1536,
            per_layer_input_wall_us: 200,
            pre_sdpa_wall_us: 300,
            pre_sdpa_qkv_proj_wall_us: 200,
            pre_sdpa_qk_norm_wall_us: 50,
            pre_sdpa_rope_kv_wall_us: 75,
            sdpa_wall_us: 100,
            post_attn_wall_us: 600,
            post_attn_ffn_wall_us: 400,
            post_attn_output_proj_wall_us: 75,
            post_attn_residual_norm_wall_us: 25,
            post_attn_residual_gate_wall_us: 50,
            lm_head_wall_us: 50,
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_decode_profile_enabled"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_decode_steps"),
            Some(&128)
        );
        assert_eq!(decisions.get("ax_mlx_decode_profile_layers"), Some(&3072));
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_per_layer_input_wall_us"),
            Some(&1000)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_pre_sdpa_wall_us"),
            Some(&1500)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_pre_sdpa_qkv_proj_wall_us"),
            Some(&900)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_pre_sdpa_qk_norm_wall_us"),
            Some(&250)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_pre_sdpa_rope_kv_wall_us"),
            Some(&375)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_sdpa_wall_us"),
            Some(&600)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_post_attn_wall_us"),
            Some(&3000)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_post_attn_ffn_wall_us"),
            Some(&2200)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_post_attn_output_proj_wall_us"),
            Some(&375)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_post_attn_residual_norm_wall_us"),
            Some(&125)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_post_attn_residual_gate_wall_us"),
            Some(&250)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_lm_head_wall_us"),
            Some(&200)
        );

        let mut disabled_decisions = Vec::new();
        DecodeProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
        assert!(disabled_decisions.is_empty());
    }

    #[test]
    fn linear_attention_no_draft_threshold_disables_request_acceleration() {
        assert_eq!(
            LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD, 8,
            "empty linear-attention drafts should allow several generated-output probe windows"
        );
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
    fn indexed_route_decisions_update_in_place_and_remove_initial_duplicates() {
        let mut decisions = vec![
            ("other".to_string(), 7),
            ("ax_ngram_draft_tokens".to_string(), 1),
            ("ax_ngram_draft_tokens".to_string(), 2),
        ];

        {
            let mut indexed = IndexedRouteDecisions::new(&mut decisions);
            indexed.upsert_route_decision("ax_ngram_draft_tokens", 9);
            indexed.upsert_route_decision("new_counter", 3);
        }

        assert_eq!(
            decisions,
            vec![
                ("other".to_string(), 7),
                ("ax_ngram_draft_tokens".to_string(), 9),
                ("new_counter".to_string(), 3),
            ]
        );
    }

    #[test]
    fn decode_telemetry_records_route_counters() {
        let mut telemetry = DecodeTelemetry::default();

        telemetry.record_prefill(100);
        telemetry.record_prefill_breakdown(70, 20, 5);
        telemetry.record_prefill_eval_barrier();
        telemetry.record_prefill_drain_async_evals(2);
        telemetry.record_decode(40);
        telemetry.record_direct_bootstrap(7);
        telemetry.record_direct_pipeline(11);
        telemetry.record_direct_pipeline_timings(DirectPipelineTimings {
            forward_wall_us: 3,
            argmax_wall_us: 4,
            async_eval_wall_us: 2,
            next_complete_wall_us: 6,
            pending_eval_wall_us: 5,
            pending_read_wall_us: 1,
        });
        telemetry.record_direct_pipeline_op_count(42);
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
        assert_eq!(decisions.get("ax_mlx_prefill_forward_wall_us"), Some(&70));
        assert_eq!(
            decisions.get("ax_mlx_prefill_prefix_cache_wall_us"),
            Some(&20)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_generation_state_wall_us"),
            Some(&5)
        );
        assert_eq!(decisions.get("ax_mlx_prefill_eval_barriers"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_prefill_drain_async_evals"), Some(&2));
        assert_eq!(decisions.get("ax_mlx_decode_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_decode_wall_us"), Some(&40));
        assert_eq!(decisions.get("ax_mlx_direct_bootstrap_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_direct_bootstrap_wall_us"), Some(&7));
        assert_eq!(decisions.get("ax_mlx_direct_pipeline_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_direct_pipeline_wall_us"), Some(&11));
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_forward_wall_us"),
            Some(&3)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_argmax_wall_us"),
            Some(&4)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_async_eval_wall_us"),
            Some(&2)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_next_complete_wall_us"),
            Some(&6)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_pending_eval_wall_us"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_pending_read_wall_us"),
            Some(&1)
        );
        assert_eq!(decisions.get("ax_mlx_direct_pipeline_op_count"), Some(&42));
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
    fn request_fallback_direct_pipeline_reuses_pending_step() {
        assert_eq!(
            direct_pipeline_action(false),
            DirectPipelineAction::Bootstrap,
            "first fallback direct step must bootstrap the pipeline"
        );
        assert_eq!(
            direct_pipeline_action(true),
            DirectPipelineAction::ContinuePending,
            "later fallback direct steps must continue the pending lazy token"
        );
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
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_METAL_SUCCESSES),
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
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_METAL_SUCCESSES),
            Some(&0)
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
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_METAL_SUCCESSES),
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
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_METAL_SUCCESSES),
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
        let dense_draft =
            ngram_acceleration_draft(&ngram, false, 0.95, NgramPolicyVariant::MajorityRecency)
                .draft;
        assert!(!dense_draft.is_empty(), "dense draft should be non-empty");
        assert!(
            dense_draft.len() <= MAX_DRAFT_LEN,
            "dense draft must not exceed MAX_DRAFT_LEN"
        );

        // Linear-attention: min_support=2 filters one-off n-grams.
        assert!(
            ngram_acceleration_draft(&ngram, true, 0.95, NgramPolicyVariant::MajorityRecency,)
                .draft
                .is_empty(),
            "linear attention should not probe one-off prompt n-grams"
        );

        ngram.feed(&[1, 2, 3]);
        let lin_draft =
            ngram_acceleration_draft(&ngram, true, 0.95, NgramPolicyVariant::MajorityRecency).draft;
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
    fn ngram_adaptive_draft_len_shrinks_and_extends_from_acceptance() {
        // Dense models always use MAX_DRAFT_LEN — confidence gate prunes naturally.
        assert_eq!(adaptive_ngram_draft_len(false, 0.95), MAX_DRAFT_LEN);
        assert_eq!(adaptive_ngram_draft_len(false, 0.70), MAX_DRAFT_LEN);
        assert_eq!(adaptive_ngram_draft_len(false, 0.40), MAX_DRAFT_LEN);
        assert_eq!(
            adaptive_ngram_draft_len(true, 0.95),
            DEFAULT_DRAFT_LEN,
            "linear attention stays capped even at high confidence"
        );
        assert_eq!(
            adaptive_ngram_draft_len(true, 0.40),
            NGRAM_DRAFT_LEN_LOW_CONFIDENCE
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

    #[test]
    fn mlx_manifest_validation_rejects_chained_gemma4_kv_shared_layers() {
        // Layer 2 tries to share KV from layer 1, but layer 1 is itself a shared
        // layer (no own K/V cache). `MlxKVCache::peek_source_kv` would `.expect()`
        // on the missing source and panic mid-decode; reject the manifest up front.
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.layer_count = 3;
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);
        manifest.kv_shared_source_layers.insert(2, 1);

        let error = validate_gemma4_interleaved_attention(&manifest)
            .expect_err("chained KV sharing should fail closed");

        assert!(error.to_string().contains("shared layer"));
    }

    #[test]
    fn mlx_manifest_validation_rejects_zero_gemma4_sliding_window() {
        // `Some(0)` survives build_layer_configs as Some(0) and is then filtered
        // back to None by the rotating-window cache path, silently turning sliding
        // layers into grow-forever ones. Force the manifest to fail closed instead.
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.sliding_window_size = Some(0);
        manifest.layer_types = vec!["sliding_attention".to_string()];

        let error = validate_gemma4_interleaved_attention(&manifest)
            .expect_err("zero sliding_window_size should fail closed");

        assert!(error.to_string().contains("sliding_window_size"));
    }

    #[test]
    fn embed_batch_mean_mask_excludes_padding_positions() {
        // Verify the mask layout used by embed_batch Mean pooling.
        // batch = [[a,b,c], [x,y]] padded to max_len=3: positions 0..len are 1.0, rest 0.0.
        let actual_lens: Vec<usize> = vec![3, 2];
        let max_seq = 3usize;
        let mut mask_data = vec![0.0f32; actual_lens.len() * max_seq];
        for (i, &l) in actual_lens.iter().enumerate() {
            for j in 0..l {
                mask_data[i * max_seq + j] = 1.0;
            }
        }
        // seq 0 (len 3): all positions active
        assert_eq!(&mask_data[0..3], &[1.0, 1.0, 1.0]);
        // seq 1 (len 2): position 2 is padding
        assert_eq!(&mask_data[3..6], &[1.0, 1.0, 0.0]);
    }
}
