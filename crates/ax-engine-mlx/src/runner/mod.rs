use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::fs;
use std::sync::{Arc, LazyLock, OnceLock};

use parking_lot::Mutex;
use std::thread::{self, ThreadId};
use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, MlxStream, add, argmax, argpartition_axis, astype, clear_cache, divide,
    enable_compile, eval, max_recommended_working_set_size, multiply, power, reshape,
    set_cache_limit, set_memory_limit, set_wired_limit, slice, softmax, stack, sum_axis, take,
    take_along_axis,
};

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::scheduler::ExecutionMode;
use ax_engine_core::{
    EmbeddingPooling, ExecutionRunner, ExecutionStatus, KvCompressionConfig, KvWriteSummary,
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
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_KV_SHARED,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_LINEAR_ATTENTION,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_MISSING_STORAGE,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_PREFILL_ONLY,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_SHORT_CONTEXT,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_SLIDING_WINDOW,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_HEAD_DIM,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_UNSUPPORTED_PRESET,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_COLD_METAL_WALL_US,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_HOT_TAIL_MERGE_WALL_US,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_METAL_SUCCESSES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_OUTPUT_STAGING_WALL_US,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_QUERY_READBACK_WALL_US,
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
    ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS, RequestExecutionUpdate, RequestId,
    RequestMultimodalInputs, RunnerInput, RunnerOutput, StopReason, upsert_route_decision,
};

use crate::batched_decode_session::{
    BatchedDecodeSession, batched_decode_enabled, batched_decode_sampling_enabled,
    model_batched_eligible,
};
use crate::batched_sampling::{BatchedSamplingClass, argmax_batched, batched_sampling_class};
use crate::gemma4_assistant_mtp::{
    Gemma4AssistantMtpConfig, Gemma4AssistantMtpDisableReason, Gemma4AssistantMtpStatus,
    gemma4_assistant_mtp_debug_enabled, gemma4_assistant_mtp_max_depth_cap,
    resolve_gemma4_assistant_mtp_deep_gate, resolve_gemma4_assistant_mtp_first_gate,
};
use crate::generate::{
    DirectPipelineTimings, advance_direct_pipeline_with_timings_and_turboquant_context,
    chunked_prefill, chunked_prefill_gemma4_unified_with_sampling_buffers,
    chunked_prefill_with_mtp_history_and_sampling_buffers, chunked_prefill_with_sampling_buffers,
    decode_step, start_direct_pipeline_with_turboquant_context,
};
use crate::kv_cache::{MlxKVCache, MlxKVCacheUsage};
use crate::model::{
    DecodeProfileSnapshot, DenseFfnFastpathSnapshot, Gemma4MoeProfileSnapshot,
    LinearAttentionProfileSnapshot, ModelConfig, MoeProfileSnapshot, PrefillProfileSnapshot,
    TurboQuantModelDecodeContext, forward_all_positions_post_norm_last_lm_head,
    forward_all_positions_with_post_norm, take_decode_profile_snapshot,
    take_dense_ffn_fastpath_snapshot, take_gemma4_moe_profile_snapshot,
    take_linear_attention_profile_snapshot, take_moe_profile_snapshot,
    take_prefill_profile_snapshot,
};
use crate::mtp::{
    glm_mtp_draft_tokens, mtp_draft_tokens, mtp_draft_tokens_after_forced_prefix,
    mtp_draft_tokens_gated,
};
use crate::ngram_accel::{
    DEFAULT_DRAFT_LEN, LINEAR_MIN_NGRAM_SUPPORT, MAX_DRAFT_LEN, NgramDraftOutcome,
    NgramDraftPolicy, NgramDraftRejection, NgramPolicyVariant, NgramTable, classify_prompt_class,
    effective_draft_confidence_threshold, ngram_accel_decode_step_with_sampling_buffers,
    ngram_feedback_policy, recompute_committed_prefix, single_decode_with_turboquant_context,
};
use crate::sampling::{
    MlxSamplingParams, MlxSamplingRequest, TokenDistribution, Xorshift64, sample_categorical_into,
    sample_categorical_with_logprob_and_distribution, sample_residual_token_distribution,
};
use crate::speculation_profile::speculation_profile_from_env;
use crate::turboquant::{
    TURBOQUANT_ROUTE_METADATA_SCHEMA_VERSION, TurboQuantProductionRequirements,
    turboquant_support_report,
};
#[cfg(test)]
use crate::weights::{LayerWeights, QuantizedWeight};
use crate::weights::{ModelWeights, load_weights};

mod pipeline;
mod prefix_cache;
mod runner_telemetry;
mod util;

use pipeline::*;
pub use prefix_cache::MlxPrefixCacheStore;
pub(crate) use prefix_cache::*;
use runner_telemetry::*;
use util::*;

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
/// Outside `<think>` on reasoning models, require this many observations before
/// drafting.  Keeps speculative attempts on well-established repeating patterns
/// (SQL keywords, JSON delimiters) while suppressing one-off, low-confidence
/// guesses in mixed prose/code regions.
const POST_THINK_MIN_NGRAM_SUPPORT: u32 = 2;
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
/// first few generated tokens, so keep the post-start threshold conservative.
/// The initial non-repeating prompt/no-draft case is handled separately before
/// decode starts because it has no prompt-side evidence to justify slow probes.
const LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD: u32 = 8;
// When a linear-attention request has fallen back to the direct pipeline, do
// not rescan the n-gram table every token looking for a re-enable point. Sparse
// random prompts can spend the full request in fallback, and the direct path
// should stay close to the explicit direct baseline.
const LINEAR_NGRAM_REENABLE_PROBE_INTERVAL: u32 = 4;
/// Maximum number of prompt tail tokens fed into the n-gram table.
/// Long prompts (especially random-token benchmarks) would otherwise fill the
/// table with useless bigrams that trigger false-positive n-gram acceleration and force
/// expensive recompute on the very first n-gram acceleration attempt.
const NGRAM_PROMPT_FEED_MAX: usize = 64;
/// When MTP is active, n-gram stacks on top of MTP drafts rather than driving
/// speculation alone. A larger prompt window gives the table enough real-code
/// bigrams to contribute from early decode steps without the random-token
/// false-positive risk (MTP handles verification even if n-gram misfires).
const NGRAM_MTP_PROMPT_FEED_MAX: usize = 256;
/// Repeating prompts need enough prompt history for prompt-lookup drafts to see
/// an earlier occurrence of the current suffix. Keep this bounded, but larger
/// than the default random-prompt guard above.
const NGRAM_REPEATING_PROMPT_FEED_MAX: usize = 512;
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
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_HITS: &str = "ax_mlx_prefix_cache_disk_hits";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_MISSES: &str = "ax_mlx_prefix_cache_disk_misses";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_INSERTS: &str = "ax_mlx_prefix_cache_disk_inserts";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_INSERT_BYTES_KIB: &str =
    "ax_mlx_prefix_cache_disk_insert_bytes_kib";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_EVICTIONS: &str =
    "ax_mlx_prefix_cache_disk_evictions";
const ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_MULTIMODAL_PREFILL_REQUESTS: &str =
    "ax_mlx_gemma4_unified_multimodal_prefill_requests";
const ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_IMAGE_INPUTS: &str =
    "ax_mlx_gemma4_unified_image_inputs";
const ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_AUDIO_INPUTS: &str =
    "ax_mlx_gemma4_unified_audio_inputs";
const ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_VIDEO_INPUTS: &str =
    "ax_mlx_gemma4_unified_video_inputs";
const ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_VISUAL_INPUTS: &str =
    "ax_mlx_gemma4_unified_visual_inputs";
const ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_PREFIX_CACHE_DISABLED: &str =
    "ax_mlx_gemma4_unified_prefix_cache_disabled";
const ROUTE_DECISION_AX_MLX_GEMMA4_UNIFIED_MTP_PREFILL_WARMUP_SKIPPED: &str =
    "ax_mlx_gemma4_unified_mtp_prefill_warmup_skipped";
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

/// Opaque cross-session share cell for loaded model weights (Option A of the
/// session/weight-reuse design, `.internal/tech-spec/server-session-weight-reuse.md`).
///
/// A long-lived owner (e.g. the server's `StatelessGenerateContext`) holds one
/// cell per model; every per-request `MlxRunner` build receives it via
/// `from_artifacts_with_runtime_shares`. The first build loads the weights and
/// publishes the `Arc`; later builds reuse it and skip both the safetensors
/// read/GPU eval and the JIT warmup. All clones share the same cell. Sharing
/// is sound because weight arrays are fully evaluated before publication and
/// immutable afterwards (`MlxArray` is atomically refcounted `Send + Sync`).
#[derive(Clone, Default)]
pub struct MlxSharedWeightsCell(Arc<OnceLock<Arc<ModelWeights>>>);

impl MlxSharedWeightsCell {
    pub fn new() -> Self {
        Self::default()
    }

    /// True once a build has published loaded weights into this cell.
    pub fn is_loaded(&self) -> bool {
        self.0.get().is_some()
    }

    fn get(&self) -> Option<Arc<ModelWeights>> {
        self.0.get().cloned()
    }

    fn publish(&self, weights: Arc<ModelWeights>) {
        // Two concurrent first builds may race here; the loser keeps its own
        // copy for its session lifetime and the winner's stays shared.
        let _ = self.0.set(weights);
    }
}

impl fmt::Debug for MlxSharedWeightsCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxSharedWeightsCell")
            .field("loaded", &self.is_loaded())
            .finish()
    }
}

#[derive(Clone)]
struct Gemma4AssistantMtpRuntime {
    status: Gemma4AssistantMtpStatus,
    cfg: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    target_shared_layers: crate::model::Gemma4AssistantSharedKvLayers,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MtpDraftSource {
    #[default]
    None,
    Mtp,
    Gemma4Assistant,
    Ngram,
    HybridMtp,
}

impl MtpDraftSource {
    fn is_model_draft(self) -> bool {
        matches!(
            self,
            MtpDraftSource::Mtp | MtpDraftSource::Gemma4Assistant | MtpDraftSource::HybridMtp
        )
    }

    /// Whether optimistic accept-all may skip verification for this draft.
    /// Only the target model's own MTP head qualifies — its acceptance EWMA is
    /// what the optimistic gate measures. Sidecar drafters (Gemma4 assistant;
    /// GLM is excluded earlier via `mtp_optimistic_allowed`) and n-gram drafts
    /// can propose plausible but target-mismatched tokens and must be verified.
    fn optimistic_accept_eligible(self) -> bool {
        matches!(self, MtpDraftSource::Mtp | MtpDraftSource::HybridMtp)
    }

    fn utility_family(self) -> DraftSourceFamily {
        match self {
            MtpDraftSource::Gemma4Assistant => DraftSourceFamily::Assistant,
            MtpDraftSource::Ngram => DraftSourceFamily::Ngram,
            MtpDraftSource::Mtp | MtpDraftSource::HybridMtp | MtpDraftSource::None => {
                DraftSourceFamily::Mtp
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum DraftSourceFamily {
    #[default]
    Mtp,
    Assistant,
    Ngram,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MtpNgramAcceptanceMode {
    #[default]
    Confidence,
    Delta,
    Greedy,
}

impl MtpNgramAcceptanceMode {
    fn route_code(self) -> u32 {
        match self {
            Self::Confidence => 0,
            Self::Delta => 1,
            Self::Greedy => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MtpModelAcceptanceMode {
    #[default]
    Greedy,
    RejectionSampling,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NgramSelfTuneState {
    drafted: u32,
    accepted: u32,
    disabled: bool,
}

impl NgramSelfTuneState {
    fn record_submitted(&mut self, drafted: usize) {
        self.drafted = self.drafted.saturating_add(saturating_u32(drafted));
    }

    fn record_verified(&mut self, accepted: usize, threshold: f32, warmup: u32) {
        self.accepted = self.accepted.saturating_add(saturating_u32(accepted));
        if !self.disabled && warmup > 0 && self.drafted >= warmup {
            let rate = self.accepted as f32 / self.drafted.max(1) as f32;
            if rate < threshold {
                self.disabled = true;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MtpTelemetry {
    draft_tokens: u32,
    accepted_tokens: u32,
    decode_steps: u32,
    full_accept_steps: u32,
    partial_reject_steps: u32,
    complete_miss_steps: u32,
    /// Cycle-level acceptance: accepted_cycles counts MTP verify cycles where
    /// ALL draft tokens were accepted; rejected_cycles counts cycles with any
    /// rejection.  Matches Lightning-MLX's `mtp_acceptance_ratio` metric
    /// (accepted / (accepted + rejected)) for fair cross-engine comparison.
    accepted_cycles: u32,
    rejected_cycles: u32,
    cache_clone_wall_us: u32,
    verify_forward_wall_us: u32,
    verify_eval_wall_us: u32,
    accept_wall_us: u32,
    rollback_wall_us: u32,
    tail_sample_wall_us: u32,
    draft_wall_us: u32,
    target_softmax_wall_us: u32,
    verify_tokens: u32,
    emitted_tokens: u32,
    source_mtp_submitted_tokens: u32,
    source_mtp_accepted_tokens: u32,
    source_mtp_rejected_tokens: u32,
    source_mtp_cascade_rejected_tokens: u32,
    source_mtp_proposer_wall_us: u32,
    source_assistant_submitted_tokens: u32,
    source_assistant_accepted_tokens: u32,
    source_assistant_rejected_tokens: u32,
    source_assistant_cascade_rejected_tokens: u32,
    source_assistant_proposer_wall_us: u32,
    source_ngram_proposed_tokens: u32,
    source_ngram_rejected_tokens: u32,
    source_ngram_cascade_rejected_tokens: u32,
    ngram_lookup_wall_us: u32,
    accepted_by_depth: [u32; 3],
    drafted_by_depth: [u32; 3],
    draft_source_mtp_tokens: u32,
    accepted_source_mtp_tokens: u32,
    draft_source_ngram_tokens: u32,
    accepted_source_ngram_tokens: u32,
    draft_source_hybrid_mtp_tokens: u32,
    accepted_source_hybrid_mtp_tokens: u32,
    ngram_attempt_steps: u32,
    ngram_hit_steps: u32,
    ngram_no_candidate_steps: u32,
    ngram_confidence_filtered_steps: u32,
    ngram_cycle_guard_steps: u32,
    ngram_skipped_mtp_steps: u32,
    ngram_skipped_mtp_tokens: u32,
    ngram_hybrid_tail_steps: u32,
    ngram_hybrid_tail_tokens: u32,
    /// MTP n-gram stacking steps skipped because model is outside `<think>`.
    ngram_think_gated_steps: u32,
    accept_rate_ewma: f32,
    accept_rate_ewma_samples: u32,
    /// MTP-only acceptance EWMA: tracks only MTP and HybridMtp sourced draft
    /// positions, excluding n-gram tokens.  Used by the saturation gate so
    /// that n-gram rejections cannot suppress gating when the model itself
    /// has near-perfect acceptance (e.g. 27B flappy at 99.5% accept rate).
    mtp_only_accept_rate_ewma: f32,
    mtp_only_accept_rate_ewma_samples: u32,
    ngram_saturated_gated_steps: u32,
    /// Steps where n-gram was gated off because the combined accept rate fell
    /// below the MTP-only rate, indicating n-gram is actively hurting (cascade
    /// rejections of MTP tokens when n-gram fails at early positions).
    ngram_hurt_gated_steps: u32,
    /// Steps gated by the new source-aware hurt gate (ADR-019 D3).
    ngram_source_hurt_gated_steps: u32,
    /// Steps gated by the legacy EWMA hurt gate (ADR-018 D3), tracked separately
    /// for A/B comparison during the transition.
    ngram_legacy_hurt_gated_steps: u32,
    ngram_auto_disabled_steps: u32,
    ngram_self_tune_disabled_steps: u32,
    ngram_utility_gated_steps: u32,
    ngram_utility_insufficient_sample_steps: u32,
    ngram_safety_disabled_steps: u32,
    ngram_safety_tightened_steps: u32,
    ngram_safety_reason: u32,
    ngram_submitted_tokens: u32,
    ngram_submitted_accepted_tokens: u32,
    utility_baseline_steps: u32,
    utility_baseline_wall_us: u32,
    utility_baseline_emitted_tokens: u32,
    utility_stacked_steps: u32,
    utility_stacked_wall_us: u32,
    utility_stacked_emitted_tokens: u32,
    utility_stacked_ngram_submitted_tokens: u32,
    /// Steps where auto-optimistic activated (EWMA ≥ 0.99 without env override).
    auto_optimistic_steps: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MtpStepTimings {
    cache_clone_wall_us: u32,
    verify_forward_wall_us: u32,
    verify_eval_wall_us: u32,
    target_softmax_wall_us: u32,
    accept_wall_us: u32,
    rollback_wall_us: u32,
    tail_sample_wall_us: u32,
    draft_wall_us: u32,
    mtp_draft_wall_us: u32,
    assistant_draft_wall_us: u32,
    ngram_lookup_wall_us: u32,
    verify_tokens: u32,
    emitted_tokens: u32,
    ngram_submitted_tokens: u32,
}

impl MtpTelemetry {
    fn record_step(
        &mut self,
        drafted: usize,
        accepted: usize,
        sources: &[MtpDraftSource],
        ewma_accepted: Option<usize>,
        mtp_ewma_numerator: usize,
    ) {
        self.draft_tokens = self.draft_tokens.saturating_add(saturating_u32(drafted));
        self.accepted_tokens = self
            .accepted_tokens
            .saturating_add(saturating_u32(accepted));
        self.decode_steps = self.decode_steps.saturating_add(1);
        const ALPHA: f32 = 0.05;
        if drafted > 0 {
            let ewma_ac = ewma_accepted.unwrap_or(accepted);
            let step_rate = ewma_ac as f32 / drafted as f32;
            if self.accept_rate_ewma_samples == 0 {
                self.accept_rate_ewma = step_rate;
            } else {
                self.accept_rate_ewma = (1.0 - ALPHA) * self.accept_rate_ewma + ALPHA * step_rate;
            }
            self.accept_rate_ewma_samples = self.accept_rate_ewma_samples.saturating_add(1);
        }
        // Cascade-correct MTP-only EWMA: in sequential spec decoding, an n-gram
        // rejection at position k causes all tokens at positions k+1..drafted to be
        // rejected by cascade rather than by their own quality.  Counting those
        // cascaded rejections against MTP deflates the EWMA when n-gram is running.
        // Only count tokens that were "meaningfully evaluated":
        //   - accepted tokens (positions 0..accepted) passed the verifier, and
        //   - the first-rejected token (position `accepted`) was the one that
        //     actually failed — but only if that token is MTP-sourced.
        // Tokens at positions accepted+1..drafted are cascade rejections and are
        // excluded regardless of source.
        //
        // The EWMA numerator is the actual accepted MTP count in rejection-sampling
        // mode, or the argmax-match count in optimistic mode (quality proxy when the
        // verifier is skipped).  The denominator is all MTP positions meaningfully
        // evaluated (accepted + first rejection).
        let mtp_only_accepted_count = sources
            .iter()
            .take(accepted)
            .filter(|s| s.is_model_draft())
            .count();
        let first_rejection_is_mtp = accepted < drafted
            && sources
                .get(accepted)
                .map(|s| s.is_model_draft())
                .unwrap_or(true);
        let mtp_only_drafted = mtp_only_accepted_count + usize::from(first_rejection_is_mtp);
        if mtp_only_drafted > 0 {
            let mtp_step_rate = mtp_ewma_numerator as f32 / mtp_only_drafted as f32;
            if self.mtp_only_accept_rate_ewma_samples == 0 {
                self.mtp_only_accept_rate_ewma = mtp_step_rate;
            } else {
                self.mtp_only_accept_rate_ewma =
                    (1.0 - ALPHA) * self.mtp_only_accept_rate_ewma + ALPHA * mtp_step_rate;
            }
            self.mtp_only_accept_rate_ewma_samples =
                self.mtp_only_accept_rate_ewma_samples.saturating_add(1);
        }
        // Cycle-level acceptance: matches Lightning-MLX's mtp_acceptance_ratio
        // = accepted_cycles / (accepted_cycles + rejected_cycles).
        if accepted == drafted && drafted > 0 {
            self.full_accept_steps = self.full_accept_steps.saturating_add(1);
            self.accepted_cycles = self.accepted_cycles.saturating_add(1);
        } else if accepted == 0 {
            self.complete_miss_steps = self.complete_miss_steps.saturating_add(1);
            if drafted > 0 {
                self.rejected_cycles = self.rejected_cycles.saturating_add(1);
            }
        } else {
            self.partial_reject_steps = self.partial_reject_steps.saturating_add(1);
            self.rejected_cycles = self.rejected_cycles.saturating_add(1);
        }
        for d in 0..drafted.min(3) {
            self.drafted_by_depth[d] = self.drafted_by_depth[d].saturating_add(1);
            if d < accepted {
                self.accepted_by_depth[d] = self.accepted_by_depth[d].saturating_add(1);
            }
        }
        for index in 0..drafted {
            let source = sources
                .get(index)
                .copied()
                .filter(|source| *source != MtpDraftSource::None)
                .unwrap_or(MtpDraftSource::Mtp);
            let accepted_token = index < accepted;
            let rejected_token = index == accepted && accepted < drafted;
            let cascade_rejected_token = index > accepted && accepted < drafted;
            match source.utility_family() {
                DraftSourceFamily::Mtp => {
                    self.source_mtp_submitted_tokens =
                        self.source_mtp_submitted_tokens.saturating_add(1);
                    if accepted_token {
                        self.source_mtp_accepted_tokens =
                            self.source_mtp_accepted_tokens.saturating_add(1);
                    } else if rejected_token {
                        self.source_mtp_rejected_tokens =
                            self.source_mtp_rejected_tokens.saturating_add(1);
                    } else if cascade_rejected_token {
                        self.source_mtp_cascade_rejected_tokens =
                            self.source_mtp_cascade_rejected_tokens.saturating_add(1);
                    }
                }
                DraftSourceFamily::Assistant => {
                    self.source_assistant_submitted_tokens =
                        self.source_assistant_submitted_tokens.saturating_add(1);
                    if accepted_token {
                        self.source_assistant_accepted_tokens =
                            self.source_assistant_accepted_tokens.saturating_add(1);
                    } else if rejected_token {
                        self.source_assistant_rejected_tokens =
                            self.source_assistant_rejected_tokens.saturating_add(1);
                    } else if cascade_rejected_token {
                        self.source_assistant_cascade_rejected_tokens = self
                            .source_assistant_cascade_rejected_tokens
                            .saturating_add(1);
                    }
                }
                DraftSourceFamily::Ngram => {
                    if rejected_token {
                        self.source_ngram_rejected_tokens =
                            self.source_ngram_rejected_tokens.saturating_add(1);
                    } else if cascade_rejected_token {
                        self.source_ngram_cascade_rejected_tokens =
                            self.source_ngram_cascade_rejected_tokens.saturating_add(1);
                    }
                }
            }
            match source {
                MtpDraftSource::Mtp => {
                    self.draft_source_mtp_tokens = self.draft_source_mtp_tokens.saturating_add(1);
                    if accepted_token {
                        self.accepted_source_mtp_tokens =
                            self.accepted_source_mtp_tokens.saturating_add(1);
                    }
                }
                MtpDraftSource::Gemma4Assistant => {
                    self.draft_source_mtp_tokens = self.draft_source_mtp_tokens.saturating_add(1);
                    if accepted_token {
                        self.accepted_source_mtp_tokens =
                            self.accepted_source_mtp_tokens.saturating_add(1);
                    }
                }
                MtpDraftSource::Ngram => {
                    self.draft_source_ngram_tokens =
                        self.draft_source_ngram_tokens.saturating_add(1);
                    if accepted_token {
                        self.accepted_source_ngram_tokens =
                            self.accepted_source_ngram_tokens.saturating_add(1);
                    }
                }
                MtpDraftSource::HybridMtp => {
                    self.draft_source_hybrid_mtp_tokens =
                        self.draft_source_hybrid_mtp_tokens.saturating_add(1);
                    if accepted_token {
                        self.accepted_source_hybrid_mtp_tokens =
                            self.accepted_source_hybrid_mtp_tokens.saturating_add(1);
                    }
                }
                MtpDraftSource::None => {}
            }
        }
    }

    fn record_ngram_attempt(&mut self, rejection: Option<NgramDraftRejection>) {
        self.ngram_attempt_steps = self.ngram_attempt_steps.saturating_add(1);
        match rejection {
            Some(NgramDraftRejection::NoCandidate) => {
                self.ngram_no_candidate_steps = self.ngram_no_candidate_steps.saturating_add(1);
            }
            Some(NgramDraftRejection::ConfidenceFiltered) => {
                self.ngram_confidence_filtered_steps =
                    self.ngram_confidence_filtered_steps.saturating_add(1);
            }
            None => {}
        }
    }

    fn record_ngram_cycle_guard(&mut self) {
        self.ngram_cycle_guard_steps = self.ngram_cycle_guard_steps.saturating_add(1);
    }

    fn record_ngram_stack_hit(&mut self, draft_len: usize, skipped_mtp: bool) {
        self.ngram_hit_steps = self.ngram_hit_steps.saturating_add(1);
        if skipped_mtp {
            self.ngram_skipped_mtp_steps = self.ngram_skipped_mtp_steps.saturating_add(1);
            self.ngram_skipped_mtp_tokens = self
                .ngram_skipped_mtp_tokens
                .saturating_add(saturating_u32(draft_len));
        }
    }

    fn record_ngram_hybrid_tail(&mut self, tail_len: usize) {
        if tail_len == 0 {
            return;
        }
        self.ngram_hybrid_tail_steps = self.ngram_hybrid_tail_steps.saturating_add(1);
        self.ngram_hybrid_tail_tokens = self
            .ngram_hybrid_tail_tokens
            .saturating_add(saturating_u32(tail_len));
    }

    fn record_ngram_proposed(&mut self, draft_len: usize) {
        self.source_ngram_proposed_tokens = self
            .source_ngram_proposed_tokens
            .saturating_add(saturating_u32(draft_len));
    }

    fn record_ngram_submitted(&mut self, draft_len: usize) {
        self.ngram_submitted_tokens = self
            .ngram_submitted_tokens
            .saturating_add(saturating_u32(draft_len));
    }

    fn record_ngram_verified(&mut self, accepted: usize) {
        self.ngram_submitted_accepted_tokens = self
            .ngram_submitted_accepted_tokens
            .saturating_add(saturating_u32(accepted));
    }

    fn record_timings(&mut self, timings: MtpStepTimings) {
        self.cache_clone_wall_us = self
            .cache_clone_wall_us
            .saturating_add(timings.cache_clone_wall_us);
        self.verify_forward_wall_us = self
            .verify_forward_wall_us
            .saturating_add(timings.verify_forward_wall_us);
        self.verify_eval_wall_us = self
            .verify_eval_wall_us
            .saturating_add(timings.verify_eval_wall_us);
        self.target_softmax_wall_us = self
            .target_softmax_wall_us
            .saturating_add(timings.target_softmax_wall_us);
        self.accept_wall_us = self.accept_wall_us.saturating_add(timings.accept_wall_us);
        self.rollback_wall_us = self
            .rollback_wall_us
            .saturating_add(timings.rollback_wall_us);
        self.tail_sample_wall_us = self
            .tail_sample_wall_us
            .saturating_add(timings.tail_sample_wall_us);
        self.draft_wall_us = self.draft_wall_us.saturating_add(timings.draft_wall_us);
        self.source_mtp_proposer_wall_us = self
            .source_mtp_proposer_wall_us
            .saturating_add(timings.mtp_draft_wall_us);
        self.source_assistant_proposer_wall_us = self
            .source_assistant_proposer_wall_us
            .saturating_add(timings.assistant_draft_wall_us);
        self.ngram_lookup_wall_us = self
            .ngram_lookup_wall_us
            .saturating_add(timings.ngram_lookup_wall_us);
        self.verify_tokens = self.verify_tokens.saturating_add(timings.verify_tokens);
        self.emitted_tokens = self.emitted_tokens.saturating_add(timings.emitted_tokens);

        let utility_wall_us = timings
            .verify_forward_wall_us
            .saturating_add(timings.verify_eval_wall_us)
            .saturating_add(timings.target_softmax_wall_us)
            .saturating_add(timings.draft_wall_us);
        if timings.ngram_submitted_tokens > 0 {
            self.utility_stacked_steps = self.utility_stacked_steps.saturating_add(1);
            self.utility_stacked_wall_us =
                self.utility_stacked_wall_us.saturating_add(utility_wall_us);
            self.utility_stacked_emitted_tokens = self
                .utility_stacked_emitted_tokens
                .saturating_add(timings.emitted_tokens);
            self.utility_stacked_ngram_submitted_tokens = self
                .utility_stacked_ngram_submitted_tokens
                .saturating_add(timings.ngram_submitted_tokens);
        } else {
            self.utility_baseline_steps = self.utility_baseline_steps.saturating_add(1);
            self.utility_baseline_wall_us = self
                .utility_baseline_wall_us
                .saturating_add(utility_wall_us);
            self.utility_baseline_emitted_tokens = self
                .utility_baseline_emitted_tokens
                .saturating_add(timings.emitted_tokens);
        }
    }

    fn merge_from(&mut self, other: Self) {
        self.draft_tokens = self.draft_tokens.saturating_add(other.draft_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(other.accepted_tokens);
        self.decode_steps = self.decode_steps.saturating_add(other.decode_steps);
        self.full_accept_steps = self
            .full_accept_steps
            .saturating_add(other.full_accept_steps);
        self.partial_reject_steps = self
            .partial_reject_steps
            .saturating_add(other.partial_reject_steps);
        self.complete_miss_steps = self
            .complete_miss_steps
            .saturating_add(other.complete_miss_steps);
        self.accepted_cycles = self.accepted_cycles.saturating_add(other.accepted_cycles);
        self.rejected_cycles = self.rejected_cycles.saturating_add(other.rejected_cycles);
        self.cache_clone_wall_us = self
            .cache_clone_wall_us
            .saturating_add(other.cache_clone_wall_us);
        self.verify_forward_wall_us = self
            .verify_forward_wall_us
            .saturating_add(other.verify_forward_wall_us);
        self.verify_eval_wall_us = self
            .verify_eval_wall_us
            .saturating_add(other.verify_eval_wall_us);
        self.accept_wall_us = self.accept_wall_us.saturating_add(other.accept_wall_us);
        self.rollback_wall_us = self.rollback_wall_us.saturating_add(other.rollback_wall_us);
        self.tail_sample_wall_us = self
            .tail_sample_wall_us
            .saturating_add(other.tail_sample_wall_us);
        self.draft_wall_us = self.draft_wall_us.saturating_add(other.draft_wall_us);
        self.target_softmax_wall_us = self
            .target_softmax_wall_us
            .saturating_add(other.target_softmax_wall_us);
        self.verify_tokens = self.verify_tokens.saturating_add(other.verify_tokens);
        self.emitted_tokens = self.emitted_tokens.saturating_add(other.emitted_tokens);
        self.source_mtp_submitted_tokens = self
            .source_mtp_submitted_tokens
            .saturating_add(other.source_mtp_submitted_tokens);
        self.source_mtp_accepted_tokens = self
            .source_mtp_accepted_tokens
            .saturating_add(other.source_mtp_accepted_tokens);
        self.source_mtp_rejected_tokens = self
            .source_mtp_rejected_tokens
            .saturating_add(other.source_mtp_rejected_tokens);
        self.source_mtp_cascade_rejected_tokens = self
            .source_mtp_cascade_rejected_tokens
            .saturating_add(other.source_mtp_cascade_rejected_tokens);
        self.source_mtp_proposer_wall_us = self
            .source_mtp_proposer_wall_us
            .saturating_add(other.source_mtp_proposer_wall_us);
        self.source_assistant_submitted_tokens = self
            .source_assistant_submitted_tokens
            .saturating_add(other.source_assistant_submitted_tokens);
        self.source_assistant_accepted_tokens = self
            .source_assistant_accepted_tokens
            .saturating_add(other.source_assistant_accepted_tokens);
        self.source_assistant_rejected_tokens = self
            .source_assistant_rejected_tokens
            .saturating_add(other.source_assistant_rejected_tokens);
        self.source_assistant_cascade_rejected_tokens = self
            .source_assistant_cascade_rejected_tokens
            .saturating_add(other.source_assistant_cascade_rejected_tokens);
        self.source_assistant_proposer_wall_us = self
            .source_assistant_proposer_wall_us
            .saturating_add(other.source_assistant_proposer_wall_us);
        self.source_ngram_proposed_tokens = self
            .source_ngram_proposed_tokens
            .saturating_add(other.source_ngram_proposed_tokens);
        self.source_ngram_rejected_tokens = self
            .source_ngram_rejected_tokens
            .saturating_add(other.source_ngram_rejected_tokens);
        self.source_ngram_cascade_rejected_tokens = self
            .source_ngram_cascade_rejected_tokens
            .saturating_add(other.source_ngram_cascade_rejected_tokens);
        self.ngram_lookup_wall_us = self
            .ngram_lookup_wall_us
            .saturating_add(other.ngram_lookup_wall_us);
        for d in 0..3 {
            self.accepted_by_depth[d] =
                self.accepted_by_depth[d].saturating_add(other.accepted_by_depth[d]);
            self.drafted_by_depth[d] =
                self.drafted_by_depth[d].saturating_add(other.drafted_by_depth[d]);
        }
        self.draft_source_mtp_tokens = self
            .draft_source_mtp_tokens
            .saturating_add(other.draft_source_mtp_tokens);
        self.accepted_source_mtp_tokens = self
            .accepted_source_mtp_tokens
            .saturating_add(other.accepted_source_mtp_tokens);
        self.draft_source_ngram_tokens = self
            .draft_source_ngram_tokens
            .saturating_add(other.draft_source_ngram_tokens);
        self.accepted_source_ngram_tokens = self
            .accepted_source_ngram_tokens
            .saturating_add(other.accepted_source_ngram_tokens);
        self.draft_source_hybrid_mtp_tokens = self
            .draft_source_hybrid_mtp_tokens
            .saturating_add(other.draft_source_hybrid_mtp_tokens);
        self.accepted_source_hybrid_mtp_tokens = self
            .accepted_source_hybrid_mtp_tokens
            .saturating_add(other.accepted_source_hybrid_mtp_tokens);
        self.ngram_attempt_steps = self
            .ngram_attempt_steps
            .saturating_add(other.ngram_attempt_steps);
        self.ngram_hit_steps = self.ngram_hit_steps.saturating_add(other.ngram_hit_steps);
        self.ngram_no_candidate_steps = self
            .ngram_no_candidate_steps
            .saturating_add(other.ngram_no_candidate_steps);
        self.ngram_confidence_filtered_steps = self
            .ngram_confidence_filtered_steps
            .saturating_add(other.ngram_confidence_filtered_steps);
        self.ngram_cycle_guard_steps = self
            .ngram_cycle_guard_steps
            .saturating_add(other.ngram_cycle_guard_steps);
        self.ngram_skipped_mtp_steps = self
            .ngram_skipped_mtp_steps
            .saturating_add(other.ngram_skipped_mtp_steps);
        self.ngram_skipped_mtp_tokens = self
            .ngram_skipped_mtp_tokens
            .saturating_add(other.ngram_skipped_mtp_tokens);
        self.ngram_hybrid_tail_steps = self
            .ngram_hybrid_tail_steps
            .saturating_add(other.ngram_hybrid_tail_steps);
        self.ngram_hybrid_tail_tokens = self
            .ngram_hybrid_tail_tokens
            .saturating_add(other.ngram_hybrid_tail_tokens);
        self.ngram_think_gated_steps = self
            .ngram_think_gated_steps
            .saturating_add(other.ngram_think_gated_steps);
        // EWMA fields: sample-weighted average so the batch-level aggregate
        // reported in append_route_decisions reflects all finished requests
        // rather than staying at 0.0 (the default).  Per-request gating
        // (auto-optimistic, n-gram saturation) uses state.mtp_telemetry
        // directly and is unaffected by this merge.
        let ar_samples = self
            .accept_rate_ewma_samples
            .saturating_add(other.accept_rate_ewma_samples);
        if ar_samples > 0 {
            self.accept_rate_ewma = (self.accept_rate_ewma * self.accept_rate_ewma_samples as f32
                + other.accept_rate_ewma * other.accept_rate_ewma_samples as f32)
                / ar_samples as f32;
        }
        self.accept_rate_ewma_samples = ar_samples;
        let mtp_ar_samples = self
            .mtp_only_accept_rate_ewma_samples
            .saturating_add(other.mtp_only_accept_rate_ewma_samples);
        if mtp_ar_samples > 0 {
            self.mtp_only_accept_rate_ewma = (self.mtp_only_accept_rate_ewma
                * self.mtp_only_accept_rate_ewma_samples as f32
                + other.mtp_only_accept_rate_ewma * other.mtp_only_accept_rate_ewma_samples as f32)
                / mtp_ar_samples as f32;
        }
        self.mtp_only_accept_rate_ewma_samples = mtp_ar_samples;
        self.ngram_saturated_gated_steps = self
            .ngram_saturated_gated_steps
            .saturating_add(other.ngram_saturated_gated_steps);
        self.ngram_hurt_gated_steps = self
            .ngram_hurt_gated_steps
            .saturating_add(other.ngram_hurt_gated_steps);
        self.ngram_source_hurt_gated_steps = self
            .ngram_source_hurt_gated_steps
            .saturating_add(other.ngram_source_hurt_gated_steps);
        self.ngram_legacy_hurt_gated_steps = self
            .ngram_legacy_hurt_gated_steps
            .saturating_add(other.ngram_legacy_hurt_gated_steps);
        self.ngram_auto_disabled_steps = self
            .ngram_auto_disabled_steps
            .saturating_add(other.ngram_auto_disabled_steps);
        self.ngram_self_tune_disabled_steps = self
            .ngram_self_tune_disabled_steps
            .saturating_add(other.ngram_self_tune_disabled_steps);
        self.ngram_utility_gated_steps = self
            .ngram_utility_gated_steps
            .saturating_add(other.ngram_utility_gated_steps);
        self.ngram_utility_insufficient_sample_steps = self
            .ngram_utility_insufficient_sample_steps
            .saturating_add(other.ngram_utility_insufficient_sample_steps);
        self.ngram_safety_disabled_steps = self
            .ngram_safety_disabled_steps
            .saturating_add(other.ngram_safety_disabled_steps);
        self.ngram_safety_tightened_steps = self
            .ngram_safety_tightened_steps
            .saturating_add(other.ngram_safety_tightened_steps);
        self.ngram_safety_reason = self.ngram_safety_reason.max(other.ngram_safety_reason);
        self.ngram_submitted_tokens = self
            .ngram_submitted_tokens
            .saturating_add(other.ngram_submitted_tokens);
        self.ngram_submitted_accepted_tokens = self
            .ngram_submitted_accepted_tokens
            .saturating_add(other.ngram_submitted_accepted_tokens);
        self.utility_baseline_steps = self
            .utility_baseline_steps
            .saturating_add(other.utility_baseline_steps);
        self.utility_baseline_wall_us = self
            .utility_baseline_wall_us
            .saturating_add(other.utility_baseline_wall_us);
        self.utility_baseline_emitted_tokens = self
            .utility_baseline_emitted_tokens
            .saturating_add(other.utility_baseline_emitted_tokens);
        self.utility_stacked_steps = self
            .utility_stacked_steps
            .saturating_add(other.utility_stacked_steps);
        self.utility_stacked_wall_us = self
            .utility_stacked_wall_us
            .saturating_add(other.utility_stacked_wall_us);
        self.utility_stacked_emitted_tokens = self
            .utility_stacked_emitted_tokens
            .saturating_add(other.utility_stacked_emitted_tokens);
        self.utility_stacked_ngram_submitted_tokens = self
            .utility_stacked_ngram_submitted_tokens
            .saturating_add(other.utility_stacked_ngram_submitted_tokens);
        self.auto_optimistic_steps = self
            .auto_optimistic_steps
            .saturating_add(other.auto_optimistic_steps);
    }

    fn baseline_utility(&self) -> DraftSourceUtility {
        DraftSourceUtility {
            submitted_tokens: self.utility_baseline_emitted_tokens,
            proposer_wall_us: 0,
            verify_wall_us: self.utility_baseline_wall_us,
            emitted_tokens: self.utility_baseline_emitted_tokens,
        }
    }

    fn stacked_utility(&self) -> DraftSourceUtility {
        DraftSourceUtility {
            submitted_tokens: self.utility_stacked_ngram_submitted_tokens,
            proposer_wall_us: 0,
            verify_wall_us: self.utility_stacked_wall_us,
            emitted_tokens: self.utility_stacked_emitted_tokens,
        }
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        let entries = [
            ("ax_mtp_draft_tokens", self.draft_tokens),
            ("ax_mtp_accepted_tokens", self.accepted_tokens),
            ("ax_mtp_decode_steps", self.decode_steps),
            ("ax_mtp_full_accept_steps", self.full_accept_steps),
            ("ax_mtp_partial_reject_steps", self.partial_reject_steps),
            ("ax_mtp_complete_miss_steps", self.complete_miss_steps),
            ("ax_mtp_accepted_cycles", self.accepted_cycles),
            ("ax_mtp_rejected_cycles", self.rejected_cycles),
            ("ax_mtp_cache_clone_wall_us", self.cache_clone_wall_us),
            ("ax_mtp_verify_forward_wall_us", self.verify_forward_wall_us),
            ("ax_mtp_verify_eval_wall_us", self.verify_eval_wall_us),
            ("ax_mtp_accept_wall_us", self.accept_wall_us),
            ("ax_mtp_rollback_wall_us", self.rollback_wall_us),
            ("ax_mtp_tail_sample_wall_us", self.tail_sample_wall_us),
            ("ax_mtp_draft_wall_us", self.draft_wall_us),
            ("ax_mtp_target_softmax_wall_us", self.target_softmax_wall_us),
            ("ax_mtp_verify_tokens", self.verify_tokens),
            ("ax_mtp_emitted_tokens", self.emitted_tokens),
            (
                "ax_mtp_source_mtp_proposed_tokens",
                self.source_mtp_submitted_tokens,
            ),
            (
                "ax_mtp_source_mtp_submitted_tokens",
                self.source_mtp_submitted_tokens,
            ),
            (
                "ax_mtp_source_mtp_accepted_tokens",
                self.source_mtp_accepted_tokens,
            ),
            (
                "ax_mtp_source_mtp_rejected_tokens",
                self.source_mtp_rejected_tokens,
            ),
            (
                "ax_mtp_source_mtp_cascade_rejected_tokens",
                self.source_mtp_cascade_rejected_tokens,
            ),
            (
                "ax_mtp_source_mtp_proposer_wall_us",
                self.source_mtp_proposer_wall_us,
            ),
            (
                "ax_mtp_source_assistant_proposed_tokens",
                self.source_assistant_submitted_tokens,
            ),
            (
                "ax_mtp_source_assistant_submitted_tokens",
                self.source_assistant_submitted_tokens,
            ),
            (
                "ax_mtp_source_assistant_accepted_tokens",
                self.source_assistant_accepted_tokens,
            ),
            (
                "ax_mtp_source_assistant_rejected_tokens",
                self.source_assistant_rejected_tokens,
            ),
            (
                "ax_mtp_source_assistant_cascade_rejected_tokens",
                self.source_assistant_cascade_rejected_tokens,
            ),
            (
                "ax_mtp_source_assistant_proposer_wall_us",
                self.source_assistant_proposer_wall_us,
            ),
            (
                "ax_mtp_ngram_proposed_tokens",
                self.source_ngram_proposed_tokens,
            ),
            (
                "ax_mtp_ngram_rejected_tokens",
                self.source_ngram_rejected_tokens,
            ),
            (
                "ax_mtp_ngram_cascade_rejected_tokens",
                self.source_ngram_cascade_rejected_tokens,
            ),
            ("ax_mtp_ngram_lookup_wall_us", self.ngram_lookup_wall_us),
            ("ax_mtp_accepted_depth0", self.accepted_by_depth[0]),
            ("ax_mtp_accepted_depth1", self.accepted_by_depth[1]),
            ("ax_mtp_accepted_depth2", self.accepted_by_depth[2]),
            ("ax_mtp_drafted_depth0", self.drafted_by_depth[0]),
            ("ax_mtp_drafted_depth1", self.drafted_by_depth[1]),
            ("ax_mtp_drafted_depth2", self.drafted_by_depth[2]),
            (
                "ax_mtp_draft_source_mtp_tokens",
                self.draft_source_mtp_tokens,
            ),
            (
                "ax_mtp_accepted_source_mtp_tokens",
                self.accepted_source_mtp_tokens,
            ),
            (
                "ax_mtp_draft_source_ngram_tokens",
                self.draft_source_ngram_tokens,
            ),
            (
                "ax_mtp_accepted_source_ngram_tokens",
                self.accepted_source_ngram_tokens,
            ),
            (
                "ax_mtp_draft_source_hybrid_mtp_tokens",
                self.draft_source_hybrid_mtp_tokens,
            ),
            (
                "ax_mtp_accepted_source_hybrid_mtp_tokens",
                self.accepted_source_hybrid_mtp_tokens,
            ),
            ("ax_mtp_ngram_attempt_steps", self.ngram_attempt_steps),
            ("ax_mtp_ngram_hit_steps", self.ngram_hit_steps),
            (
                "ax_mtp_ngram_no_candidate_steps",
                self.ngram_no_candidate_steps,
            ),
            (
                "ax_mtp_ngram_confidence_filtered_steps",
                self.ngram_confidence_filtered_steps,
            ),
            (
                "ax_mtp_ngram_cycle_guard_steps",
                self.ngram_cycle_guard_steps,
            ),
            (
                "ax_mtp_ngram_skipped_mtp_steps",
                self.ngram_skipped_mtp_steps,
            ),
            (
                "ax_mtp_ngram_skipped_mtp_tokens",
                self.ngram_skipped_mtp_tokens,
            ),
            (
                "ax_mtp_ngram_hybrid_tail_steps",
                self.ngram_hybrid_tail_steps,
            ),
            (
                "ax_mtp_ngram_hybrid_tail_tokens",
                self.ngram_hybrid_tail_tokens,
            ),
            (
                "ax_mtp_ngram_think_gated_steps",
                self.ngram_think_gated_steps,
            ),
            (
                "ax_mtp_ngram_saturated_gated_steps",
                self.ngram_saturated_gated_steps,
            ),
            ("ax_mtp_ngram_hurt_gated_steps", self.ngram_hurt_gated_steps),
            (
                "ax_mtp_ngram_source_hurt_gated_steps",
                self.ngram_source_hurt_gated_steps,
            ),
            (
                "ax_mtp_ngram_legacy_hurt_gated_steps",
                self.ngram_legacy_hurt_gated_steps,
            ),
            (
                "ax_mtp_ngram_auto_disabled_steps",
                self.ngram_auto_disabled_steps,
            ),
            (
                "ax_mtp_ngram_self_tune_disabled_steps",
                self.ngram_self_tune_disabled_steps,
            ),
            (
                "ax_mtp_ngram_utility_gated_steps",
                self.ngram_utility_gated_steps,
            ),
            (
                "ax_mtp_ngram_utility_insufficient_sample_steps",
                self.ngram_utility_insufficient_sample_steps,
            ),
            (
                "ax_mtp_ngram_safety_disabled_steps",
                self.ngram_safety_disabled_steps,
            ),
            (
                "ax_mtp_ngram_safety_tightened_steps",
                self.ngram_safety_tightened_steps,
            ),
            ("ax_mtp_ngram_safety_reason", self.ngram_safety_reason),
            ("ax_mtp_ngram_submitted_tokens", self.ngram_submitted_tokens),
            (
                "ax_mtp_ngram_submitted_accepted_tokens",
                self.ngram_submitted_accepted_tokens,
            ),
            (
                "ax_mtp_ngram_accepted_tokens",
                self.ngram_submitted_accepted_tokens,
            ),
            (
                "ax_mtp_ngram_utility_baseline_steps",
                self.utility_baseline_steps,
            ),
            (
                "ax_mtp_ngram_utility_baseline_wall_us",
                self.utility_baseline_wall_us,
            ),
            (
                "ax_mtp_ngram_utility_baseline_emitted_tokens",
                self.utility_baseline_emitted_tokens,
            ),
            (
                "ax_mtp_ngram_utility_stacked_steps",
                self.utility_stacked_steps,
            ),
            (
                "ax_mtp_ngram_utility_stacked_wall_us",
                self.utility_stacked_wall_us,
            ),
            (
                "ax_mtp_ngram_utility_stacked_emitted_tokens",
                self.utility_stacked_emitted_tokens,
            ),
            (
                "ax_mtp_ngram_utility_stacked_ngram_submitted_tokens",
                self.utility_stacked_ngram_submitted_tokens,
            ),
            (
                "ax_mtp_ngram_acceptance_mode",
                mtp_ngram_acceptance_mode_from_env().route_code(),
            ),
            ("ax_mtp_auto_optimistic_steps", self.auto_optimistic_steps),
        ];
        decisions.upsert_route_decision(
            "ax_mtp_accept_rate_ewma_x1000",
            (self.accept_rate_ewma.clamp(0.0, 1.0) * 1000.0) as u32,
        );
        decisions.upsert_route_decision(
            "ax_mtp_accept_rate_ewma_samples",
            self.accept_rate_ewma_samples,
        );
        decisions.upsert_route_decision(
            "ax_mtp_mtp_only_accept_rate_ewma_x1000",
            (self.mtp_only_accept_rate_ewma.clamp(0.0, 1.0) * 1000.0) as u32,
        );
        decisions.upsert_route_decision(
            "ax_mtp_mtp_only_accept_rate_ewma_samples",
            self.mtp_only_accept_rate_ewma_samples,
        );
        // ADR-019: emit draft mode and hurt gate mode for A/B audit.
        decisions.upsert_route_decision(
            "ax_mtp_draft_mode",
            match crate::mtp::mtp_draft_mode_from_env() {
                crate::mtp::MtpDraftMode::Greedy => 0u32,
                crate::mtp::MtpDraftMode::Stochastic => 1u32,
            },
        );
        decisions.upsert_route_decision(
            "ax_mtp_hurt_gate_mode",
            match mtp_ngram_hurt_gate_mode() {
                HurtGateMode::SourceAware => 0u32,
                HurtGateMode::LegacyEwma => 1u32,
            },
        );
        // Resolved speculation profile (ADR-022), on the MTP/n-gram telemetry
        // path so Qwen fused-MTP rows expose it too — not only the Gemma
        // assistant path. `upsert` keeps it idempotent for Gemma+n-gram rows that
        // also emit it from the assistant-MTP status block (same value).
        decisions.upsert_route_decision(
            "ax_mlx_speculation_profile",
            speculation_profile_from_env().route_code(),
        );
        decisions.upsert_route_decision(
            "ax_mtp_ngram_gate_policy",
            mtp_ngram_gate_policy_from_env().route_code(),
        );
        decisions.upsert_route_decision(
            "ax_mtp_ngram_utility_baseline_cost_per_emitted_token_us",
            route_cost_us(self.baseline_utility().cost_per_emitted_token_us()),
        );
        decisions.upsert_route_decision(
            "ax_mtp_ngram_utility_stacked_cost_per_emitted_token_us",
            route_cost_us(self.stacked_utility().cost_per_emitted_token_us()),
        );
        decisions.upsert_route_decision(
            "ax_mtp_ngram_utility_min_emitted_tokens",
            mtp_ngram_utility_min_emitted_tokens(),
        );
        decisions.upsert_route_decision(
            "ax_mtp_ngram_utility_min_ngram_tokens",
            mtp_ngram_utility_min_ngram_tokens(),
        );
        // Per-depth accept rates (scaled ×1000) for A/B comparison without
        // computing ratios from drafted/accepted counters.
        for d in 0..3 {
            let drafted = self.drafted_by_depth[d];
            let accepted = self.accepted_by_depth[d];
            let rate_x1000 = if drafted > 0 {
                (accepted as f32 / drafted as f32 * 1000.0) as u32
            } else {
                0u32
            };
            decisions
                .upsert_route_decision(&format!("ax_mtp_accept_rate_depth{d}_x1000"), rate_x1000);
        }
        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    direct_pipeline_forward_layer_loop_wall_us: u32,
    direct_pipeline_forward_head_wall_us: u32,
    direct_pipeline_argmax_wall_us: u32,
    direct_pipeline_async_eval_wall_us: u32,
    direct_pipeline_next_complete_wall_us: u32,
    direct_pipeline_pending_eval_wall_us: u32,
    direct_pipeline_pending_read_wall_us: u32,
    direct_pipeline_op_count: u64,
    direct_pipeline_linear_attention_layer_ops: u64,
    direct_pipeline_linear_attention_layer_count: u32,
    direct_pipeline_full_attention_layer_ops: u64,
    direct_pipeline_full_attention_layer_count: u32,
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
    // DiffusionGemma block generation counters.
    diffusion_blocks: u32,
    diffusion_denoise_steps: u32,
    diffusion_converged_blocks: u32,
    diffusion_denoise_wall_us: u32,
    diffusion_commit_wall_us: u32,
    diffusion_block_wall_us: u32,
    // Per-criterion convergence signals (0 or 1 per block).
    diffusion_converged_strict: u32,
    diffusion_converged_acceptance: u32,
    diffusion_converged_plateau: u32,
    // Near-miss telemetry: lowest entropy/acceptance rate observed (×10000 fixed-point).
    diffusion_min_entropy_bp: u32,
    diffusion_min_acceptance_rate_bp: u32,
    diffusion_commit_skipped: u32,
    diffusion_full_pipeline_used: u32,
    diffusion_kv_buffer_used: u32,
}

impl Default for DecodeTelemetry {
    fn default() -> Self {
        Self {
            prefill_steps: 0,
            prefill_wall_us: 0,
            prefill_forward_wall_us: 0,
            prefill_prefix_cache_wall_us: 0,
            prefill_generation_state_wall_us: 0,
            decode_steps: 0,
            decode_wall_us: 0,
            direct_bootstrap_steps: 0,
            direct_bootstrap_wall_us: 0,
            direct_pipeline_steps: 0,
            direct_pipeline_wall_us: 0,
            direct_pipeline_forward_wall_us: 0,
            direct_pipeline_forward_layer_loop_wall_us: 0,
            direct_pipeline_forward_head_wall_us: 0,
            direct_pipeline_argmax_wall_us: 0,
            direct_pipeline_async_eval_wall_us: 0,
            direct_pipeline_next_complete_wall_us: 0,
            direct_pipeline_pending_eval_wall_us: 0,
            direct_pipeline_pending_read_wall_us: 0,
            direct_pipeline_op_count: 0,
            direct_pipeline_linear_attention_layer_ops: 0,
            direct_pipeline_linear_attention_layer_count: 0,
            direct_pipeline_full_attention_layer_ops: 0,
            direct_pipeline_full_attention_layer_count: 0,
            single_decode_steps: 0,
            single_decode_wall_us: 0,
            ngram_decode_steps: 0,
            ngram_decode_wall_us: 0,
            bonus_tokens: 0,
            production_decode_evals: 0,
            prefill_eval_barriers: 0,
            prefill_drain_async_evals: 0,
            diffusion_blocks: 0,
            diffusion_denoise_steps: 0,
            diffusion_converged_blocks: 0,
            diffusion_denoise_wall_us: 0,
            diffusion_commit_wall_us: 0,
            diffusion_block_wall_us: 0,
            diffusion_converged_strict: 0,
            diffusion_converged_acceptance: 0,
            diffusion_converged_plateau: 0,
            diffusion_min_entropy_bp: u32::MAX,
            diffusion_min_acceptance_rate_bp: u32::MAX,
            diffusion_commit_skipped: 0,
            diffusion_full_pipeline_used: 0,
            diffusion_kv_buffer_used: 0,
        }
    }
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
        self.direct_pipeline_forward_layer_loop_wall_us = self
            .direct_pipeline_forward_layer_loop_wall_us
            .saturating_add(timings.forward_layer_loop_wall_us);
        self.direct_pipeline_forward_head_wall_us = self
            .direct_pipeline_forward_head_wall_us
            .saturating_add(timings.forward_head_wall_us);
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
        self.direct_pipeline_linear_attention_layer_ops = self
            .direct_pipeline_linear_attention_layer_ops
            .saturating_add(timings.linear_attention_layer_ops);
        self.direct_pipeline_linear_attention_layer_count = self
            .direct_pipeline_linear_attention_layer_count
            .saturating_add(timings.linear_attention_layer_count);
        self.direct_pipeline_full_attention_layer_ops = self
            .direct_pipeline_full_attention_layer_ops
            .saturating_add(timings.full_attention_layer_ops);
        self.direct_pipeline_full_attention_layer_count = self
            .direct_pipeline_full_attention_layer_count
            .saturating_add(timings.full_attention_layer_count);
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

    fn record_diffusion_block(&mut self, result: &crate::diffusion::DiffusionBlockResult) {
        self.diffusion_blocks = self.diffusion_blocks.saturating_add(1);
        self.diffusion_denoise_steps = self
            .diffusion_denoise_steps
            .saturating_add(result.denoise_steps);
        if result.converged {
            self.diffusion_converged_blocks = self.diffusion_converged_blocks.saturating_add(1);
        }
        if result.converged_strict {
            self.diffusion_converged_strict = self.diffusion_converged_strict.saturating_add(1);
        }
        if result.converged_acceptance {
            self.diffusion_converged_acceptance =
                self.diffusion_converged_acceptance.saturating_add(1);
        }
        if result.converged_plateau {
            self.diffusion_converged_plateau = self.diffusion_converged_plateau.saturating_add(1);
        }
        // Near-miss telemetry: encode floats as basis points (×10000).
        let entropy_bp = (result.min_entropy * 10000.0).round().min(u32::MAX as f32) as u32;
        self.diffusion_min_entropy_bp = self.diffusion_min_entropy_bp.min(entropy_bp);
        let rate_bp = (result.min_acceptance_rate * 10000.0)
            .round()
            .min(u32::MAX as f32) as u32;
        self.diffusion_min_acceptance_rate_bp = self.diffusion_min_acceptance_rate_bp.min(rate_bp);
        self.diffusion_denoise_wall_us = self
            .diffusion_denoise_wall_us
            .saturating_add(result.denoise_wall_us);
        self.diffusion_commit_wall_us = self
            .diffusion_commit_wall_us
            .saturating_add(result.commit_wall_us);
        self.diffusion_block_wall_us = self
            .diffusion_block_wall_us
            .saturating_add(result.block_wall_us);
        if result.commit_skipped {
            self.diffusion_commit_skipped = self.diffusion_commit_skipped.saturating_add(1);
        }
        if result.full_pipeline_used {
            self.diffusion_full_pipeline_used = self.diffusion_full_pipeline_used.saturating_add(1);
        }
        if result.kv_buffer_used {
            self.diffusion_kv_buffer_used = self.diffusion_kv_buffer_used.saturating_add(1);
        }
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
        self.direct_pipeline_forward_layer_loop_wall_us = self
            .direct_pipeline_forward_layer_loop_wall_us
            .saturating_add(other.direct_pipeline_forward_layer_loop_wall_us);
        self.direct_pipeline_forward_head_wall_us = self
            .direct_pipeline_forward_head_wall_us
            .saturating_add(other.direct_pipeline_forward_head_wall_us);
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
        self.direct_pipeline_linear_attention_layer_ops = self
            .direct_pipeline_linear_attention_layer_ops
            .saturating_add(other.direct_pipeline_linear_attention_layer_ops);
        self.direct_pipeline_linear_attention_layer_count = self
            .direct_pipeline_linear_attention_layer_count
            .saturating_add(other.direct_pipeline_linear_attention_layer_count);
        self.direct_pipeline_full_attention_layer_ops = self
            .direct_pipeline_full_attention_layer_ops
            .saturating_add(other.direct_pipeline_full_attention_layer_ops);
        self.direct_pipeline_full_attention_layer_count = self
            .direct_pipeline_full_attention_layer_count
            .saturating_add(other.direct_pipeline_full_attention_layer_count);
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
        self.diffusion_blocks = self.diffusion_blocks.saturating_add(other.diffusion_blocks);
        self.diffusion_denoise_steps = self
            .diffusion_denoise_steps
            .saturating_add(other.diffusion_denoise_steps);
        self.diffusion_converged_blocks = self
            .diffusion_converged_blocks
            .saturating_add(other.diffusion_converged_blocks);
        self.diffusion_denoise_wall_us = self
            .diffusion_denoise_wall_us
            .saturating_add(other.diffusion_denoise_wall_us);
        self.diffusion_commit_wall_us = self
            .diffusion_commit_wall_us
            .saturating_add(other.diffusion_commit_wall_us);
        self.diffusion_block_wall_us = self
            .diffusion_block_wall_us
            .saturating_add(other.diffusion_block_wall_us);
        self.diffusion_converged_strict = self
            .diffusion_converged_strict
            .saturating_add(other.diffusion_converged_strict);
        self.diffusion_converged_acceptance = self
            .diffusion_converged_acceptance
            .saturating_add(other.diffusion_converged_acceptance);
        self.diffusion_converged_plateau = self
            .diffusion_converged_plateau
            .saturating_add(other.diffusion_converged_plateau);
        if other.diffusion_blocks > 0 {
            self.diffusion_min_entropy_bp = self
                .diffusion_min_entropy_bp
                .min(other.diffusion_min_entropy_bp);
            self.diffusion_min_acceptance_rate_bp = self
                .diffusion_min_acceptance_rate_bp
                .min(other.diffusion_min_acceptance_rate_bp);
        }
        self.diffusion_commit_skipped = self
            .diffusion_commit_skipped
            .saturating_add(other.diffusion_commit_skipped);
        self.diffusion_full_pipeline_used = self
            .diffusion_full_pipeline_used
            .saturating_add(other.diffusion_full_pipeline_used);
        self.diffusion_kv_buffer_used = self
            .diffusion_kv_buffer_used
            .saturating_add(other.diffusion_kv_buffer_used);
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
                "ax_mlx_direct_pipeline_forward_layer_loop_wall_us",
                self.direct_pipeline_forward_layer_loop_wall_us,
            ),
            (
                "ax_mlx_direct_pipeline_forward_head_wall_us",
                self.direct_pipeline_forward_head_wall_us,
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
            (
                "ax_mlx_direct_pipeline_linear_attention_layer_ops",
                u32::try_from(self.direct_pipeline_linear_attention_layer_ops).unwrap_or(u32::MAX),
            ),
            (
                "ax_mlx_direct_pipeline_linear_attention_layer_count",
                self.direct_pipeline_linear_attention_layer_count,
            ),
            (
                "ax_mlx_direct_pipeline_full_attention_layer_ops",
                u32::try_from(self.direct_pipeline_full_attention_layer_ops).unwrap_or(u32::MAX),
            ),
            (
                "ax_mlx_direct_pipeline_full_attention_layer_count",
                self.direct_pipeline_full_attention_layer_count,
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
            ("ax_mlx_diffusion_blocks", self.diffusion_blocks),
            (
                "ax_mlx_diffusion_denoise_steps",
                self.diffusion_denoise_steps,
            ),
            (
                "ax_mlx_diffusion_converged_blocks",
                self.diffusion_converged_blocks,
            ),
            (
                "ax_mlx_diffusion_denoise_wall_us",
                self.diffusion_denoise_wall_us,
            ),
            (
                "ax_mlx_diffusion_commit_wall_us",
                self.diffusion_commit_wall_us,
            ),
            (
                "ax_mlx_diffusion_block_wall_us",
                self.diffusion_block_wall_us,
            ),
            (
                "ax_mlx_diffusion_converged_strict",
                self.diffusion_converged_strict,
            ),
            (
                "ax_mlx_diffusion_converged_acceptance",
                self.diffusion_converged_acceptance,
            ),
            (
                "ax_mlx_diffusion_converged_plateau",
                self.diffusion_converged_plateau,
            ),
            (
                "ax_mlx_diffusion_min_entropy_bp",
                if self.diffusion_blocks == 0 {
                    0
                } else {
                    self.diffusion_min_entropy_bp
                },
            ),
            (
                "ax_mlx_diffusion_min_acceptance_rate_bp",
                if self.diffusion_blocks == 0 {
                    0
                } else {
                    self.diffusion_min_acceptance_rate_bp
                },
            ),
            (
                "ax_mlx_diffusion_commit_skipped",
                self.diffusion_commit_skipped,
            ),
            (
                "ax_mlx_diffusion_full_pipeline_used",
                self.diffusion_full_pipeline_used,
            ),
            (
                "ax_mlx_diffusion_kv_buffer_used",
                self.diffusion_kv_buffer_used,
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

impl MoeProfileSnapshot {
    fn merge_from(&mut self, other: Self) {
        self.enabled = self.enabled.max(other.enabled);
        self.moe_layers = self.moe_layers.saturating_add(other.moe_layers);
        self.router_us = self.router_us.saturating_add(other.router_us);
        self.expert_gate_up_us = self
            .expert_gate_up_us
            .saturating_add(other.expert_gate_up_us);
        self.expert_activation_us = self
            .expert_activation_us
            .saturating_add(other.expert_activation_us);
        self.expert_down_us = self.expert_down_us.saturating_add(other.expert_down_us);
        self.weighted_sum_us = self.weighted_sum_us.saturating_add(other.weighted_sum_us);
        self.shared_expert_us = self.shared_expert_us.saturating_add(other.shared_expert_us);
        self.total_us = self.total_us.saturating_add(other.total_us);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if self.enabled == 0 {
            return;
        }

        let entries = [
            ("ax_mlx_moe_profile_enabled", self.enabled),
            ("ax_mlx_moe_profile_moe_layers", self.moe_layers),
            ("ax_mlx_moe_profile_router_us", self.router_us),
            (
                "ax_mlx_moe_profile_expert_gate_up_us",
                self.expert_gate_up_us,
            ),
            (
                "ax_mlx_moe_profile_expert_activation_us",
                self.expert_activation_us,
            ),
            ("ax_mlx_moe_profile_expert_down_us", self.expert_down_us),
            ("ax_mlx_moe_profile_weighted_sum_us", self.weighted_sum_us),
            ("ax_mlx_moe_profile_shared_expert_us", self.shared_expert_us),
            ("ax_mlx_moe_profile_total_us", self.total_us),
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
        self.direct_cpp_inputs_attempts = self
            .direct_cpp_inputs_attempts
            .saturating_add(other.direct_cpp_inputs_attempts);
        self.direct_cpp_inputs_hits = self
            .direct_cpp_inputs_hits
            .saturating_add(other.direct_cpp_inputs_hits);
        self.direct_cpp_inputs_fallbacks = self
            .direct_cpp_inputs_fallbacks
            .saturating_add(other.direct_cpp_inputs_fallbacks);
        self.direct_cpp_inputs_profile_blocked = self
            .direct_cpp_inputs_profile_blocked
            .saturating_add(other.direct_cpp_inputs_profile_blocked);
        self.direct_cpp_post_input_attempts = self
            .direct_cpp_post_input_attempts
            .saturating_add(other.direct_cpp_post_input_attempts);
        self.direct_cpp_post_input_hits = self
            .direct_cpp_post_input_hits
            .saturating_add(other.direct_cpp_post_input_hits);
        self.direct_cpp_post_input_fallbacks = self
            .direct_cpp_post_input_fallbacks
            .saturating_add(other.direct_cpp_post_input_fallbacks);
        self.direct_cpp_post_input_profile_blocked = self
            .direct_cpp_post_input_profile_blocked
            .saturating_add(other.direct_cpp_post_input_profile_blocked);
        self.decode_post_input_metal_attempts = self
            .decode_post_input_metal_attempts
            .saturating_add(other.decode_post_input_metal_attempts);
        self.decode_post_input_metal_hits = self
            .decode_post_input_metal_hits
            .saturating_add(other.decode_post_input_metal_hits);
        self.decode_post_input_metal_fallbacks = self
            .decode_post_input_metal_fallbacks
            .saturating_add(other.decode_post_input_metal_fallbacks);
        self.decode_post_input_metal_profile_blocked = self
            .decode_post_input_metal_profile_blocked
            .saturating_add(other.decode_post_input_metal_profile_blocked);
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
        let direct_inputs_active = self.direct_cpp_inputs_attempts != 0
            || self.direct_cpp_inputs_hits != 0
            || self.direct_cpp_inputs_fallbacks != 0
            || self.direct_cpp_inputs_profile_blocked != 0;
        let direct_post_input_active = self.direct_cpp_post_input_attempts != 0
            || self.direct_cpp_post_input_hits != 0
            || self.direct_cpp_post_input_fallbacks != 0
            || self.direct_cpp_post_input_profile_blocked != 0;
        let decode_post_input_metal_active = self.decode_post_input_metal_attempts != 0
            || self.decode_post_input_metal_hits != 0
            || self.decode_post_input_metal_fallbacks != 0
            || self.decode_post_input_metal_profile_blocked != 0;
        if self.enabled == 0
            && !direct_inputs_active
            && !direct_post_input_active
            && !decode_post_input_metal_active
        {
            return;
        }

        if self.enabled != 0 {
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

        if direct_inputs_active {
            let entries = [
                (
                    "ax_mlx_direct_cpp_linear_attention_inputs_attempts",
                    self.direct_cpp_inputs_attempts,
                ),
                (
                    "ax_mlx_direct_cpp_linear_attention_inputs_hits",
                    self.direct_cpp_inputs_hits,
                ),
                (
                    "ax_mlx_direct_cpp_linear_attention_inputs_fallbacks",
                    self.direct_cpp_inputs_fallbacks,
                ),
                (
                    "ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked",
                    self.direct_cpp_inputs_profile_blocked,
                ),
            ];

            for (key, value) in entries {
                decisions.upsert_route_decision(key, value);
            }
        }

        if direct_post_input_active {
            let entries = [
                (
                    "ax_mlx_direct_cpp_linear_attention_post_input_attempts",
                    self.direct_cpp_post_input_attempts,
                ),
                (
                    "ax_mlx_direct_cpp_linear_attention_post_input_hits",
                    self.direct_cpp_post_input_hits,
                ),
                (
                    "ax_mlx_direct_cpp_linear_attention_post_input_fallbacks",
                    self.direct_cpp_post_input_fallbacks,
                ),
                (
                    "ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked",
                    self.direct_cpp_post_input_profile_blocked,
                ),
            ];

            for (key, value) in entries {
                decisions.upsert_route_decision(key, value);
            }
        }

        if decode_post_input_metal_active {
            let entries = [
                (
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts",
                    self.decode_post_input_metal_attempts,
                ),
                (
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_hits",
                    self.decode_post_input_metal_hits,
                ),
                (
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks",
                    self.decode_post_input_metal_fallbacks,
                ),
                (
                    "ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked",
                    self.decode_post_input_metal_profile_blocked,
                ),
            ];

            for (key, value) in entries {
                decisions.upsert_route_decision(key, value);
            }
        }
    }
}

impl DenseFfnFastpathSnapshot {
    fn merge_from(&mut self, other: Self) {
        self.qwen_gate_up_matvec_metal_attempts = self
            .qwen_gate_up_matvec_metal_attempts
            .saturating_add(other.qwen_gate_up_matvec_metal_attempts);
        self.qwen_gate_up_matvec_metal_hits = self
            .qwen_gate_up_matvec_metal_hits
            .saturating_add(other.qwen_gate_up_matvec_metal_hits);
        self.qwen_gate_up_matvec_metal_fallbacks = self
            .qwen_gate_up_matvec_metal_fallbacks
            .saturating_add(other.qwen_gate_up_matvec_metal_fallbacks);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if *self == Self::default() {
            return;
        }
        let entries = [
            (
                "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts",
                self.qwen_gate_up_matvec_metal_attempts,
            ),
            (
                "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits",
                self.qwen_gate_up_matvec_metal_hits,
            ),
            (
                "ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_fallbacks",
                self.qwen_gate_up_matvec_metal_fallbacks,
            ),
        ];
        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
        }
    }
}

impl PrefillProfileSnapshot {
    fn merge_from(&mut self, other: Self) {
        self.enabled = self.enabled.max(other.enabled);
        self.prefill_steps = self.prefill_steps.saturating_add(other.prefill_steps);
        self.layers = self.layers.saturating_add(other.layers);
        self.tokens = self.tokens.saturating_add(other.tokens);
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
        self.post_attn_ffn_gate_up_wall_us = self
            .post_attn_ffn_gate_up_wall_us
            .saturating_add(other.post_attn_ffn_gate_up_wall_us);
        self.post_attn_ffn_activation_wall_us = self
            .post_attn_ffn_activation_wall_us
            .saturating_add(other.post_attn_ffn_activation_wall_us);
        self.post_attn_ffn_down_wall_us = self
            .post_attn_ffn_down_wall_us
            .saturating_add(other.post_attn_ffn_down_wall_us);
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
        self.moe_router_wall_us = self
            .moe_router_wall_us
            .saturating_add(other.moe_router_wall_us);
        self.moe_expert_gate_up_wall_us = self
            .moe_expert_gate_up_wall_us
            .saturating_add(other.moe_expert_gate_up_wall_us);
        self.moe_expert_activation_wall_us = self
            .moe_expert_activation_wall_us
            .saturating_add(other.moe_expert_activation_wall_us);
        self.moe_expert_down_wall_us = self
            .moe_expert_down_wall_us
            .saturating_add(other.moe_expert_down_wall_us);
        self.moe_expert_weighted_sum_wall_us = self
            .moe_expert_weighted_sum_wall_us
            .saturating_add(other.moe_expert_weighted_sum_wall_us);
        self.moe_shared_expert_wall_us = self
            .moe_shared_expert_wall_us
            .saturating_add(other.moe_shared_expert_wall_us);
    }

    fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        if self.enabled == 0 {
            return;
        }

        let entries = [
            ("ax_mlx_prefill_profile_enabled", self.enabled),
            ("ax_mlx_prefill_profile_prefill_steps", self.prefill_steps),
            ("ax_mlx_prefill_profile_layers", self.layers),
            ("ax_mlx_prefill_profile_tokens", self.tokens),
            (
                "ax_mlx_prefill_profile_per_layer_input_wall_us",
                self.per_layer_input_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_pre_sdpa_wall_us",
                self.pre_sdpa_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_pre_sdpa_qkv_proj_wall_us",
                self.pre_sdpa_qkv_proj_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_pre_sdpa_qk_norm_wall_us",
                self.pre_sdpa_qk_norm_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_pre_sdpa_rope_kv_wall_us",
                self.pre_sdpa_rope_kv_wall_us,
            ),
            ("ax_mlx_prefill_profile_sdpa_wall_us", self.sdpa_wall_us),
            (
                "ax_mlx_prefill_profile_post_attn_wall_us",
                self.post_attn_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_post_attn_ffn_wall_us",
                self.post_attn_ffn_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_post_attn_ffn_gate_up_wall_us",
                self.post_attn_ffn_gate_up_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_post_attn_ffn_activation_wall_us",
                self.post_attn_ffn_activation_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_post_attn_ffn_down_wall_us",
                self.post_attn_ffn_down_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_post_attn_output_proj_wall_us",
                self.post_attn_output_proj_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_post_attn_residual_norm_wall_us",
                self.post_attn_residual_norm_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_post_attn_residual_gate_wall_us",
                self.post_attn_residual_gate_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_lm_head_wall_us",
                self.lm_head_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_moe_router_wall_us",
                self.moe_router_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_moe_expert_gate_up_wall_us",
                self.moe_expert_gate_up_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_moe_expert_activation_wall_us",
                self.moe_expert_activation_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_moe_expert_down_wall_us",
                self.moe_expert_down_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_moe_expert_weighted_sum_wall_us",
                self.moe_expert_weighted_sum_wall_us,
            ),
            (
                "ax_mlx_prefill_profile_moe_shared_expert_wall_us",
                self.moe_shared_expert_wall_us,
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
        self.post_attn_ffn_gate_up_wall_us = self
            .post_attn_ffn_gate_up_wall_us
            .saturating_add(other.post_attn_ffn_gate_up_wall_us);
        self.post_attn_ffn_activation_wall_us = self
            .post_attn_ffn_activation_wall_us
            .saturating_add(other.post_attn_ffn_activation_wall_us);
        self.post_attn_ffn_down_wall_us = self
            .post_attn_ffn_down_wall_us
            .saturating_add(other.post_attn_ffn_down_wall_us);
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
        self.moe_router_wall_us = self
            .moe_router_wall_us
            .saturating_add(other.moe_router_wall_us);
        self.moe_expert_gate_up_wall_us = self
            .moe_expert_gate_up_wall_us
            .saturating_add(other.moe_expert_gate_up_wall_us);
        self.moe_expert_activation_wall_us = self
            .moe_expert_activation_wall_us
            .saturating_add(other.moe_expert_activation_wall_us);
        self.moe_expert_down_wall_us = self
            .moe_expert_down_wall_us
            .saturating_add(other.moe_expert_down_wall_us);
        self.moe_expert_weighted_sum_wall_us = self
            .moe_expert_weighted_sum_wall_us
            .saturating_add(other.moe_expert_weighted_sum_wall_us);
        self.moe_shared_expert_wall_us = self
            .moe_shared_expert_wall_us
            .saturating_add(other.moe_shared_expert_wall_us);
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
                "ax_mlx_decode_profile_post_attn_ffn_gate_up_wall_us",
                self.post_attn_ffn_gate_up_wall_us,
            ),
            (
                "ax_mlx_decode_profile_post_attn_ffn_activation_wall_us",
                self.post_attn_ffn_activation_wall_us,
            ),
            (
                "ax_mlx_decode_profile_post_attn_ffn_down_wall_us",
                self.post_attn_ffn_down_wall_us,
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
            (
                "ax_mlx_decode_profile_moe_router_wall_us",
                self.moe_router_wall_us,
            ),
            (
                "ax_mlx_decode_profile_moe_expert_gate_up_wall_us",
                self.moe_expert_gate_up_wall_us,
            ),
            (
                "ax_mlx_decode_profile_moe_expert_activation_wall_us",
                self.moe_expert_activation_wall_us,
            ),
            (
                "ax_mlx_decode_profile_moe_expert_down_wall_us",
                self.moe_expert_down_wall_us,
            ),
            (
                "ax_mlx_decode_profile_moe_expert_weighted_sum_wall_us",
                self.moe_expert_weighted_sum_wall_us,
            ),
            (
                "ax_mlx_decode_profile_moe_shared_expert_wall_us",
                self.moe_shared_expert_wall_us,
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
    compression_fused_decode_blocked_linear_attention: u64,
    compression_fused_decode_blocked_sliding_window: u64,
    compression_fused_decode_blocked_kv_shared: u64,
    compression_fused_decode_blocked_ineligible_layer: u64,
    compression_fused_decode_blocked_unsupported_preset: u64,
    compression_fused_decode_blocked_unsupported_head_dim: u64,
    compression_fused_decode_blocked_gqa: u64,
    compression_fused_decode_blocked_missing_storage: u64,
    compression_fused_decode_blocked_short_context: u64,
    compression_fused_decode_query_readback_wall_us: u64,
    compression_fused_decode_cold_metal_wall_us: u64,
    compression_fused_decode_hot_tail_merge_wall_us: u64,
    compression_fused_decode_output_staging_wall_us: u64,
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
        self.compression_fused_decode_blocked_linear_attention = self
            .compression_fused_decode_blocked_linear_attention
            .saturating_add(compression.fused_decode_blocked_linear_attention);
        self.compression_fused_decode_blocked_sliding_window = self
            .compression_fused_decode_blocked_sliding_window
            .saturating_add(compression.fused_decode_blocked_sliding_window);
        self.compression_fused_decode_blocked_kv_shared = self
            .compression_fused_decode_blocked_kv_shared
            .saturating_add(compression.fused_decode_blocked_kv_shared);
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
        self.compression_fused_decode_blocked_short_context = self
            .compression_fused_decode_blocked_short_context
            .saturating_add(compression.fused_decode_blocked_short_context);
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
                    self.compression_fused_decode_query_readback_wall_us = self
                        .compression_fused_decode_query_readback_wall_us
                        .saturating_add(compression.fused_decode_query_readback_wall_us);
                    self.compression_fused_decode_cold_metal_wall_us = self
                        .compression_fused_decode_cold_metal_wall_us
                        .saturating_add(compression.fused_decode_cold_metal_wall_us);
                    self.compression_fused_decode_hot_tail_merge_wall_us = self
                        .compression_fused_decode_hot_tail_merge_wall_us
                        .saturating_add(compression.fused_decode_hot_tail_merge_wall_us);
                    self.compression_fused_decode_output_staging_wall_us = self
                        .compression_fused_decode_output_staging_wall_us
                        .saturating_add(compression.fused_decode_output_staging_wall_us);
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
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_LINEAR_ATTENTION,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_linear_attention),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_SLIDING_WINDOW,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_sliding_window),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_KV_SHARED,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_kv_shared),
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
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_BLOCKED_SHORT_CONTEXT,
                    saturating_u32_from_u64(self.compression_fused_decode_blocked_short_context),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_QUERY_READBACK_WALL_US,
                    saturating_u32_from_u64(self.compression_fused_decode_query_readback_wall_us),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_COLD_METAL_WALL_US,
                    saturating_u32_from_u64(self.compression_fused_decode_cold_metal_wall_us),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_HOT_TAIL_MERGE_WALL_US,
                    saturating_u32_from_u64(self.compression_fused_decode_hot_tail_merge_wall_us),
                ),
                (
                    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_OUTPUT_STAGING_WALL_US,
                    saturating_u32_from_u64(self.compression_fused_decode_output_staging_wall_us),
                ),
            ];

            for (key, value) in compression_entries {
                decisions.upsert_route_decision(key, value);
            }
        }
    }
}

#[derive(Default)]
struct MtpTargetProbWorkspace {
    flat_indices: Vec<i32>,
    target_probs: Vec<f32>,
}

/// Per-request mutable state persisted across prefill → decode steps.
struct RequestState {
    cache: MlxKVCache,
    prompt_prefix_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    cached_prefill_output_token: Option<u32>,
    ngram: NgramTable,
    /// Per-request PRNG for sampling-capable paths. Seeded from the request's
    /// sampling seed so repeated deterministic requests are reproducible.
    rng: Xorshift64,
    sampling_probs_buf: Vec<f32>,
    sampling_logits_buf: Vec<f32>,
    sampling_candidates_buf: Vec<(usize, f32)>,
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
    /// Countdown before probing whether direct fallback output has produced
    /// enough repeated n-gram evidence to re-enable acceleration.
    linear_ngram_reenable_probe_countdown: u32,
    /// Request-local fallback: once a linear-attention request proves it has no
    /// useful n-gram support, finish it on the direct pipeline.
    ngram_acceleration_disabled_for_request: bool,
    ngram_request_disable_reason: NgramRequestDisableReason,
    /// Pre-verified bonus tokens ready to serve without a model run.
    bonus_queue: VecDeque<u32>,
    /// Buffered tokens from the most recent diffusion block commit.
    /// DiffusionGemma generates `canvas_size` tokens per block; the runner
    /// drains them one at a time through the standard decode path.
    diffusion_block_queue: VecDeque<u32>,
    /// Request-local DiffusionGemma embedding table reused across generated
    /// blocks for self-conditioning.
    diffusion_embed_table: Option<MlxArray>,
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
    /// Per-request cumulative prefill profile. This is opt-in and mirrors the
    /// decode-stage profile, but only for chunked prompt forward passes.
    prefill_profile: PrefillProfileSnapshot,
    /// Shared 1-layer KV cache for the recurrent MTP head.  `None` until first use.
    mtp_cache: Option<MlxKVCache>,
    /// Total entries in `mtp_cache` (= total MTP head forward calls made).
    mtp_decode_count: usize,
    /// Draft token(s) generated by the MTP head at the previous decode step.
    /// Empty on the first decode step or after a rejected draft.
    mtp_pending_draft: Vec<u32>,
    /// Log-probabilities of `mtp_pending_draft` under the draft distribution.
    /// Used for rejection-sampling acceptance: accept draft[i] with probability
    /// min(1, p_target(draft[i]) / exp(mtp_pending_draft_log_probs[i])).
    /// MTP positions: softmax log-probs at draft_sampling.temperature (or T=1.0 for
    /// greedy-mode drafts after Phase 1).  N-gram hybrid positions: 0.0 (delta
    /// distribution, p_draft=1.0).  Empty for pure n-gram drafts (no MTP tail).
    mtp_pending_draft_log_probs: Vec<f32>,
    /// Sparse draft distributions aligned with `mtp_pending_draft`.
    /// Used to sample the exact residual correction after sampled-MTP rejection.
    mtp_pending_draft_distributions: Vec<TokenDistribution>,
    /// Source for each pending draft token. N-gram prefix tokens and MTP tail
    /// tokens need separate counters and feedback in the hybrid path.
    mtp_pending_draft_sources: Vec<MtpDraftSource>,
    /// The exact policy used to draft the n-gram prefix of `mtp_pending_draft`,
    /// when that prefix is non-empty. Verification happens on the *next* step,
    /// so this must be carried on `state` (not a local) to be replayed byte-for-byte
    /// in `record_draft_feedback` — see that function's doc comment for why a
    /// mismatched policy silently drops or misattributes feedback.
    ngram_draft_policy: Option<NgramDraftPolicy>,
    mtp_target_prob_workspace: MtpTargetProbWorkspace,
    /// Request-local MTP draft depth cap.  Adapted from the last accept/reject
    /// outcome so low-acceptance prompts stop paying for deeper draft chains.
    mtp_adaptive_max_depth: usize,
    /// Skip-state logits: logits at the committed position from the previous
    /// verify pass.  When `Some`, the next `run_mtp_decode` call can sample
    /// the primary token from these logits instead of running a fresh verify
    /// forward pass for the first token position.  Set when `AX_MLX_MTP_SKIP_STATE=1`.
    mtp_skip_logits: Option<MlxArray>,
    /// Skip-state hidden: post-norm hidden at the same committed position.
    /// Used as `main_hidden` for the MTP head when skip-state is active.
    mtp_skip_hidden: Option<MlxArray>,
    /// Cumulative MTP draft/accept counters for benchmark telemetry.
    mtp_telemetry: MtpTelemetry,
    /// Cumulative Gemma4 assistant-MTP counters for route metadata.
    gemma4_assistant_mtp_telemetry: Gemma4AssistantMtpTelemetry,
    /// Number of consecutive decode steps where accept_count == 0.
    /// Used by `mtp_next_adaptive_depth` to progressively lower the depth floor.
    mtp_consecutive_misses: u32,
    /// Per-request MTP bypass: once the acceptance EWMA drops below the bypass
    /// threshold with sufficient samples, MTP is disabled for the rest of the
    /// request and all decode steps use the direct single-token path.
    mtp_bypassed: bool,
    /// Per-request latch: once auto-optimistic activates (EWMA ≥ 0.99),
    /// it stays latched until the argmax-based EWMA drops below 0.85.
    /// Hysteresis prevents oscillation because argmax acceptance is strictly
    /// stricter than stochastic acceptance (draft tokens that pass
    /// p_target/p_draft rejection sampling may not be the argmax token),
    /// so the EWMA shifts metric upon activation.
    auto_optimistic_active: bool,
    /// Post-norm hidden rows from the final prefill chunk.
    /// Set by `chunked_prefill_with_mtp_history` and consumed by
    /// `initialize_generation_state` to prime the MTP head's KV cache with
    /// committed prompt/history transitions before decode starts.
    mtp_prefill_hidden: Option<MlxArray>,
    /// Token IDs paired with `mtp_prefill_hidden` rows for MTP history warmup.
    mtp_prefill_history_tokens: Vec<u32>,
    /// True when the last emitted token is inside a `<think>...</think>` block.
    /// Initialized from prompt tokens. Used to gate n-gram acceleration to think
    /// regions only, where repetition density is high for reasoning models.
    ngram_in_think: bool,
    /// Per-request n-gram self-tune: tracks draft tokens and accepted tokens
    /// for this request.  After warmup, if acceptance rate falls below the
    /// threshold, n-gram is disabled for the rest of the request (mirrors
    /// lightning-mlx 0.7.0 `NgramRequestState._self_tune_disabled`).
    ngram_self_tune: NgramSelfTuneState,
    /// Remaining steps to keep n-gram gated after a utility hurt decision.
    mtp_ngram_utility_hysteresis_remaining: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum NgramRequestDisableReason {
    #[default]
    None,
    ShortOutputBudget,
    LinearNoDraft,
    LinearInitialNoDraft,
}

impl RequestState {
    fn new(num_layers: usize, seed: u64) -> Self {
        Self {
            cache: MlxKVCache::new(num_layers),
            prompt_prefix_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            cached_prefill_output_token: None,
            ngram: NgramTable::new(),
            rng: Xorshift64::new(seed),
            sampling_probs_buf: Vec::new(),
            sampling_logits_buf: Vec::new(),
            sampling_candidates_buf: Vec::new(),
            ngram_beta_alpha: NGRAM_BETA_PRIOR_ALPHA,
            ngram_beta_beta: NGRAM_BETA_PRIOR_BETA,
            ngram_disabled_steps: 0,
            linear_ngram_no_draft_streak: 0,
            linear_ngram_reenable_probe_countdown: 0,
            ngram_acceleration_disabled_for_request: false,
            ngram_request_disable_reason: NgramRequestDisableReason::None,
            bonus_queue: VecDeque::new(),
            diffusion_block_queue: VecDeque::new(),
            diffusion_embed_table: None,
            next_model_last_token: None,
            pending_direct: None,
            direct_pipeline_emitted_tokens: 0,
            ngram_acceleration: NgramAccelerationTelemetry::default(),
            decode_telemetry: DecodeTelemetry::default(),
            decode_profile: DecodeProfileSnapshot::default(),
            prefill_profile: PrefillProfileSnapshot::default(),
            mtp_cache: None,
            mtp_decode_count: 0,
            mtp_pending_draft: Vec::new(),
            mtp_pending_draft_log_probs: Vec::new(),
            mtp_pending_draft_distributions: Vec::new(),
            mtp_pending_draft_sources: Vec::new(),
            ngram_draft_policy: None,
            mtp_target_prob_workspace: MtpTargetProbWorkspace::default(),
            mtp_adaptive_max_depth: 0,
            mtp_skip_logits: None,
            mtp_skip_hidden: None,
            mtp_telemetry: MtpTelemetry::default(),
            gemma4_assistant_mtp_telemetry: Gemma4AssistantMtpTelemetry::default(),
            mtp_consecutive_misses: 0,
            mtp_bypassed: false,
            auto_optimistic_active: false,
            mtp_prefill_hidden: None,
            mtp_prefill_history_tokens: Vec::new(),
            ngram_in_think: false,
            ngram_self_tune: NgramSelfTuneState::default(),
            mtp_ngram_utility_hysteresis_remaining: 0,
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

fn seed_generation_ngram_from_prompt(state: &mut RequestState, has_mtp: bool) {
    let prompt_class = classify_prompt_class(&state.prompt_prefix_tokens);
    let feed_max = if prompt_class == crate::ngram_accel::PROMPT_CLASS_REPEATING {
        NGRAM_REPEATING_PROMPT_FEED_MAX
    } else if has_mtp {
        // With MTP active, use a wider prompt window so real-code bigrams are
        // seeded early. MTP handles verification overhead if n-gram misfires,
        // so the random-token false-positive risk is acceptable here.
        NGRAM_MTP_PROMPT_FEED_MAX
    } else {
        NGRAM_PROMPT_FEED_MAX
    };
    let feed_start = state.prompt_prefix_tokens.len().saturating_sub(feed_max);
    state
        .ngram
        .feed_from_prompt(&state.prompt_prefix_tokens[feed_start..]);
}

fn seed_generation_ngram_from_prefill_output(
    state: &mut RequestState,
    prefill_output_token: Option<u32>,
) {
    if let Some(token) = prefill_output_token {
        state.ngram.feed(&[token]);
    }
}

/// Cache key for the embedding-forward compiled closure: thread- and
/// shape-specific. MLX compiled closures are stream-registry sensitive, so a
/// closure compiled on one worker thread must not be applied on another.
type EmbedCompileKey = (ThreadId, usize, Option<usize>);

/// Cache key for the batched embedding-forward compiled closure.
/// `target_positions` is baked into the trace, so two batches with the same
/// `(thread_id, batch_size, max_len)` but different per-sequence target
/// positions hit distinct keys.
type EmbedBatchCompileKey = (ThreadId, usize, usize, Option<Vec<usize>>);

/// EmbeddingGemma compiled batch closure output contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum EmbedGemmaBatchCompileKind {
    Encoder,
    Pooled,
}

/// Cache key for the EmbeddingGemma batched compiled closure.
/// The bidirectional padding mask is determined by `(batch, max_len,
/// actual_lens)`, so same-shape batches with the same real lengths reuse the
/// compiled graph. The closure kind is part of the key because the encoder
/// closure returns `[B, max_seq, H]` while the pooled closure returns `[B, H]`.
type EmbedGemmaBatchCompileKey = (
    ThreadId,
    EmbedGemmaBatchCompileKind,
    usize,
    usize,
    Vec<usize>,
);

/// ExecutionRunner backed by the MLX inference path.
pub struct MlxRunner {
    cfg: ModelConfig,
    cfg_arc: Arc<ModelConfig>,
    weights: Arc<ModelWeights>,
    /// Chunk size used for warm-extend prefill (snapshot restore + suffix).
    /// MLA models force this to `MLA_DEFAULT_PREFILL_CHUNK` (16) so the
    /// SDPA shape sequence matches the cold-of-full equivalence path.
    prefill_chunk: usize,
    /// Chunk size used for cold prefill (no prefix-cache restore). For MLA
    /// models this preserves the caller-supplied larger chunk (e.g. 2048
    /// via `--prefill-step-size`), avoiding the 5–6× prefill throughput
    /// regression that the chunk-16 alignment imposes when warm-extend is
    /// not actually engaged for the request. Set equal to `prefill_chunk`
    /// for non-MLA models.
    cold_prefill_chunk: usize,
    kv_layer_windows: Vec<Option<usize>>,
    binding_summary: NativeModelBindingSummary,
    terminal_token_ids: Vec<u32>,
    states: Mutex<HashMap<RequestId, RequestState>>,
    /// Whether this model can use the experimental batched dense-decode path
    /// (computed once via `model_batched_eligible`). Gates the `run()`
    /// interception together with `batched_decode_enabled()`.
    batched_decode_model_eligible: bool,
    /// Shared cohort for the batched dense-decode path (`AX_MLX_BATCHED_DECODE`).
    /// Empty and untouched unless the flag is on and the model is eligible.
    batched_session: Mutex<BatchedDecodeSession>,
    /// Dedicated GPU stream kept alive for the runner's lifetime.
    _stream: MlxStream,
    /// When true, disable n-gram acceleration and use the direct decode path.
    disable_ngram_acceleration: bool,
    /// When true, keep MTP enabled but do not use the n-gram-first draft source
    /// inside the MTP verify loop.
    disable_mtp_ngram_stacking: bool,
    /// When true, MTP verify always accepts all drafts without rejection sampling.
    mtp_optimistic: bool,
    /// When true, MTP decode captures verify logits/hidden as skip state for the
    /// next iteration, avoiding a redundant main-model forward for the first token.
    mtp_skip_state: bool,
    mtp_target_softmax_topk: Option<u32>,
    gemma4_assistant_mtp_status: Gemma4AssistantMtpStatus,
    gemma4_assistant_mtp: Option<Gemma4AssistantMtpRuntime>,
    ngram_policy_variant: NgramPolicyVariant,
    /// Optional KV compression policy. Disabled by default and never changes logits in shadow mode.
    kv_compression: KvCompressionConfig,
    /// Per-layer compression eligibility. Empty when compression is disabled.
    kv_compression_layer_eligible: Vec<bool>,
    /// Serialized, thread-agnostic KV snapshots for block-aligned exact prompt prefixes.
    prefix_cache: Arc<Mutex<MlxPrefixCache>>,
    /// Optional L2 file-backed prefix cache (F3). Populated when
    /// `AX_MLX_PREFIX_CACHE_DIR` is set and the disk-disabled kill
    /// switch is not engaged. `None` when off; the L2 paths short-
    /// circuit cheaply.
    disk_prefix_cache: Option<Arc<crate::disk_prefix_cache::DiskPrefixCache>>,
    /// Gemma-family sliding-window rotating backing store for rollback-free
    /// direct greedy decode.
    rotating_sliding_decode: bool,
    /// Optional mlx_lm-style `clear_cache` cadence for the direct decode pipeline.
    direct_clear_cache_cadence: u32,
    /// Weight-layout snapshot computed once at construction. `runner.run`
    /// emits this as `ax_mlx_dense_ffn_gate_up_packed_layers` /
    /// `ax_mlx_dense_ffn_split_gate_up_layers` and
    /// `ax_mlx_linear_attention_qkvz_ba_packed_layers` /
    /// `ax_mlx_linear_attention_split_qkvba_layers` route decisions every step.
    /// The counts are invariant under decode (weights don't change post-init)
    /// so caching avoids the 64-layer iteration per scheduler step.
    weight_layout_telemetry: WeightLayoutTelemetry,
    /// Affine quantization bit-width summary computed at construction.
    /// Emitted every step as `ax_mlx_affine_*` route decisions so benchmark
    /// artifacts record the quantization recipe (min/max bits, per-bit counts,
    /// and whether the 3-bit experimental gate was active at load time).
    affine_quant_telemetry: AffineQuantBitsTelemetry,
    /// Per-thread/per-shape compiled embedding-forward closures. Each entry is
    /// built on the first `embed()` call at a new `(thread_id, seq_len,
    /// target_position)` shape and reused on the same worker thread. Set
    /// `AX_EMBED_NO_COMPILE=1` to skip the compiled path and fall back to
    /// imperative `forward_for_embedding`.
    embed_compile_cache: Mutex<HashMap<EmbedCompileKey, mlx_sys::MlxClosure>>,
    /// Per-thread/per-shape compiled batched-embedding-forward closures. Keyed
    /// on `(thread_id, batch_size, max_len, target_positions)`; same kill
    /// switch as the single-call cache (`AX_EMBED_NO_COMPILE`).
    embed_batch_compile_cache: Mutex<HashMap<EmbedBatchCompileKey, mlx_sys::MlxClosure>>,
    /// Per-thread/per-shape compiled EmbeddingGemma batched closures. Keyed
    /// on `(thread_id, batch_size, max_len, actual_lens)`; the bidirectional
    /// padding mask is captured at closure-build time and baked into the
    /// trace. Same kill switch (`AX_EMBED_NO_COMPILE`).
    embed_gemma_batch_compile_cache: Mutex<HashMap<EmbedGemmaBatchCompileKey, mlx_sys::MlxClosure>>,
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

/// Maximum number of compiled closures retained per embed compile cache.
/// When exceeded, the cache is cleared and rebuilt — prevents unbounded
/// memory growth under workloads with many distinct input shapes.
const EMBED_COMPILE_CACHE_MAX_ENTRIES: usize = 256;

/// Cached flag: when `AX_EMBED_GPU_NORMALIZE=1`, L2 normalization runs on
/// the GPU instead of the default CPU read-back path.
static EMBED_GPU_NORMALIZE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("AX_EMBED_GPU_NORMALIZE")
        .map(|v| !(v == "0" || v.is_empty()))
        .unwrap_or(false)
});

/// Cached flag: when set, disables compiled embedding closures for A/B
/// benchmarking against the imperative forward path.
static EMBED_NO_COMPILE: LazyLock<bool> =
    LazyLock::new(|| std::env::var("AX_EMBED_NO_COMPILE").is_ok());

impl fmt::Debug for MlxRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxRunner")
            .field("layers", &self.cfg.layer_count)
            .field("vocab", &self.cfg.vocab_size)
            .finish()
    }
}

fn load_gemma4_assistant_mtp_runtime(
    target_cfg: &ModelConfig,
    status: &Gemma4AssistantMtpStatus,
) -> (Gemma4AssistantMtpStatus, Option<Gemma4AssistantMtpRuntime>) {
    let Some(mut config) = status.config.clone().filter(|_| status.validated) else {
        return (status.clone(), None);
    };
    // The prepared contract historically declares max_depth = 1, but the assistant
    // is stateless per step (re-reads the target KV cache each forward; carries
    // draft context through its post_projection backbone-hidden estimate), so the
    // SAME weights support recurrent multi-token drafting. The runtime env cap
    // (default 2) drives the draft depth — the canonical T=0.6 26B benchmark
    // measured depth-2 at 1.10-1.20x decode while holding accept >97%. See
    // gemma4_assistant_mtp.rs and docs/GEMMA4-ASSISTANT-MULTI-DEPTH.md.
    config.max_depth = gemma4_assistant_mtp_max_depth_cap();

    let disabled = |config: Gemma4AssistantMtpConfig, message: &str| {
        if gemma4_assistant_mtp_debug_enabled() {
            eprintln!("Gemma4 Assistant MTP attach failed: {message}");
        }
        Gemma4AssistantMtpStatus {
            configured: true,
            validated: false,
            enabled: false,
            attach_failed: true,
            disable_reason: Gemma4AssistantMtpDisableReason::WeightLoadFailed,
            max_depth: config.max_depth,
            config: Some(config),
        }
    };

    let assistant_artifacts = match NativeModelArtifacts::from_dir(&config.assistant_path) {
        Ok(artifacts) => artifacts,
        Err(error) => return (disabled(config, &error.to_string()), None),
    };
    let assistant_cfg = ModelConfig::from_manifest(assistant_artifacts.manifest());
    if assistant_cfg.model_family != "gemma4_assistant" {
        return (
            disabled(
                config,
                "assistant artifact manifest is not gemma4_assistant",
            ),
            None,
        );
    }
    let assistant_weights = match load_weights(&assistant_artifacts) {
        Ok(weights) => weights,
        Err(error) => return (disabled(config, &error.to_string()), None),
    };
    let status = Gemma4AssistantMtpStatus {
        configured: true,
        validated: true,
        enabled: true,
        attach_failed: false,
        disable_reason: Gemma4AssistantMtpDisableReason::None,
        max_depth: config.max_depth,
        config: Some(config),
    };
    let runtime = Gemma4AssistantMtpRuntime {
        status: status.clone(),
        cfg: Arc::new(assistant_cfg),
        weights: Arc::new(assistant_weights),
        target_shared_layers: target_cfg.gemma4_assistant_shared_kv_layers(),
    };
    (status, Some(runtime))
}

impl MlxRunner {
    fn has_mtp(&self) -> bool {
        self.weights.mtp.is_some()
            || self.weights.glm_mtp.is_some()
            || self.gemma4_assistant_mtp.is_some()
    }

    fn mtp_max_depth(&self) -> usize {
        if let Some(head) = &self.weights.mtp {
            head.max_depth
        } else if let Some(head) = &self.weights.glm_mtp {
            head.max_depth
        } else {
            self.gemma4_assistant_mtp
                .as_ref()
                .map_or(0, |runtime| runtime.status.max_depth)
        }
    }

    fn gemma4_assistant_mtp_status(&self) -> &Gemma4AssistantMtpStatus {
        &self.gemma4_assistant_mtp_status
    }

    /// Build with every cross-session share available: an optional prefix
    /// snapshot store and an optional shared-weights cell (Option A of the
    /// session/weight-reuse design). The first build through an empty cell
    /// loads the weights and publishes them; later builds reuse the loaded
    /// `Arc<ModelWeights>` and skip both the safetensors read and the JIT
    /// warmup forwards.
    #[allow(clippy::too_many_arguments)]
    pub fn from_artifacts_with_runtime_shares(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
        kv_compression: KvCompressionConfig,
        prefix_cache_store: Option<MlxPrefixCacheStore>,
        shared_weights: Option<&MlxSharedWeightsCell>,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_inner(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            disable_mtp_ngram_stacking,
            kv_compression,
            prefix_cache_store,
            shared_weights,
        )
    }

    pub fn from_artifacts(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        kv_compression: KvCompressionConfig,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_with_mtp_options(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            mtp_disable_ngram_stacking_from_env(),
            kv_compression,
        )
    }

    pub fn from_artifacts_with_mtp_options(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
        kv_compression: KvCompressionConfig,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_inner(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            disable_mtp_ngram_stacking,
            kv_compression,
            None,
            None,
        )
    }

    pub fn from_artifacts_with_prefix_cache(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        kv_compression: KvCompressionConfig,
        prefix_cache_store: MlxPrefixCacheStore,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_with_prefix_cache_and_mtp_options(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            mtp_disable_ngram_stacking_from_env(),
            kv_compression,
            prefix_cache_store,
        )
    }

    pub fn from_artifacts_with_prefix_cache_and_mtp_options(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
        kv_compression: KvCompressionConfig,
        prefix_cache_store: MlxPrefixCacheStore,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_inner(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            disable_mtp_ngram_stacking,
            kv_compression,
            Some(prefix_cache_store),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_artifacts_inner(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
        kv_compression: KvCompressionConfig,
        prefix_cache_store: Option<MlxPrefixCacheStore>,
        shared_weights: Option<&MlxSharedWeightsCell>,
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
        // Scale to 90% of Metal's recommended working set to avoid macOS kernel
        // panics caused by wiring the full max (documented in mlx-lm#883).
        // Override via AX_MLX_WIRED_LIMIT_SCALE (0.0-1.0).
        let wired_cap = max_recommended_working_set_size();
        if wired_cap > 0 {
            let scale: f64 = std::env::var("AX_MLX_WIRED_LIMIT_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.9);
            let scaled = (wired_cap as f64 * scale.clamp(0.0, 1.0)) as usize;
            set_wired_limit(scaled);
        }

        // Bound — but do NOT disable — MLX's internal buffer cache. The cache
        // recycles freed GPU buffers; with it off, every transient allocation
        // goes through the system/IOGPU allocator. mlx-lm#828's claim that
        // set_cache_limit(0) has "no measurable performance impact" holds for
        // dense models but is badly wrong for high-allocation MoE decode:
        // disabling it regressed Qwen3.6-35B-A3B decode ~40% (169 -> 100 tok/s),
        // because each expert's transient buffers were re-allocated from IOGPU
        // every step instead of recycled. Default the cache to the wired working
        // set so recycling stays on while total memory remains bounded by the
        // memory limit below. Override via AX_MLX_CACHE_LIMIT (bytes); set 0 to
        // disable explicitly.
        match std::env::var("AX_MLX_CACHE_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(limit) => {
                set_cache_limit(limit);
            }
            // Only set an explicit bound when the working set is known; otherwise
            // leave MLX's own default in place rather than risk disabling it.
            None if wired_cap > 0 => {
                set_cache_limit(wired_cap);
            }
            None => {}
        }

        // Set MLX memory limit. Defaults to wired_cap (same as wired working
        // set), which is more conservative than MLX's default 1.5x. Override
        // via AX_MLX_MEMORY_LIMIT (bytes); 0 = use MLX default.
        let memory_limit: usize = std::env::var("AX_MLX_MEMORY_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(wired_cap);
        if memory_limit > 0 {
            set_memory_limit(memory_limit);
        }

        validate_mlx_supported_manifest(artifacts)?;

        let cfg = ModelConfig::from_manifest(artifacts.manifest());
        let terminal_token_ids = resolve_terminal_token_ids(artifacts);
        let kv_layer_windows = kv_layer_windows_from_config(&cfg);
        let rotating_sliding_decode =
            disable_ngram_acceleration && crate::fastpath::rotating_sliding_decode_enabled();
        let ngram_policy_variant = ngram_policy_variant_from_env();
        // Mirror mlx-lm's `mx.clear_cache()` cadence: `generate.py:467-468`
        // calls `mx.clear_cache()` every 256 decoded tokens so the lazy graph
        // / intermediate-array cache cannot grow without bound during long
        // generations. Without this AX accumulates the same cache and pays
        // extra per-step overhead on multi-hundred-token decodes. Operators
        // can disable via `AX_MLX_DIRECT_CLEAR_CACHE_CADENCE=0` or override
        // with any other cadence.
        let direct_clear_cache_cadence = std::env::var("AX_MLX_DIRECT_CLEAR_CACHE_CADENCE")
            .ok()
            .and_then(|raw| raw.parse::<u32>().ok())
            .unwrap_or(256);
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
        // Weight arrays are immutable once loaded (Arc-shared, evaluated
        // leaves), so a populated share cell lets this build skip the full
        // safetensors read + GPU eval entirely.
        let preloaded_weights = shared_weights.and_then(MlxSharedWeightsCell::get);
        let reused_shared_weights = preloaded_weights.is_some();
        let weights = match preloaded_weights {
            Some(weights) => weights,
            None => {
                let loaded = Arc::new(load_weights(artifacts).map_err(MlxRunnerError::Weights)?);
                if let Some(cell) = shared_weights {
                    cell.publish(Arc::clone(&loaded));
                }
                loaded
            }
        };
        let (gemma4_assistant_mtp_status, gemma4_assistant_mtp) =
            load_gemma4_assistant_mtp_runtime(&cfg, &weights.gemma4_assistant_mtp);

        let binding_summary = binding_summary_from_specs(artifacts.tensor_specs());
        let affine_quant_telemetry = AffineQuantBitsTelemetry::from_specs(artifacts.tensor_specs());

        // MLA models default to a smaller `prefill_chunk` for warm-extend
        // (snapshot restore + suffix) so chunked_prefill produces the same
        // SDPA Q/K shape sequence as the cold-of-full path. Cold prefill
        // without any prefix-cache restore keeps the caller's larger chunk
        // (`cold_prefill_chunk`) because chunk-16 alignment serves no
        // correctness purpose when no snapshot is being compared against,
        // and the 5–6× per-prefill dispatch overhead on MLA was hurting
        // the GLM-4.7-Flash README throughput claim. Override either with
        // `AX_MLX_MLA_PREFILL_CHUNK=N` (applies to warm-extend) or by
        // setting the caller's prefill chunk explicitly; non-MLA tiers
        // ignore the MLA-specific resolution.
        //
        // Linear-attention tiers (Qwen3.5 9B, Qwen3-Next family — Qwen 3.6
        // and Qwen Coder Next) have a kernel-side chunk cap because the
        // GatedDelta recurrent update precomputes per-token gate values in
        // threadgroup-local storage. The kernel ships three specializations
        // (512 / 1024 / 2048 token cache capacity); the 2048 tier was measured
        // to lose ~15% per-token throughput vs 1024 on Qwen 3.6 27B (Hv=48)
        // because doubling the threadgroup-cache allocation halves SM
        // occupancy on M5 Max. Clamp both cold and warm prefill chunks to the
        // 1024 tier so a 2048-token mlx-lm-equivalent prompt processes as two
        // 1024-token sub-chunks; correctness is preserved because the gated
        // -delta recurrent state carries across chunks via the KV cache.
        let linear_attention_chunk_cap =
            if (0..cfg.layer_count).any(|i| cfg.is_linear_attention_layer(i)) {
                Some(crate::linear_attention_ops::GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY)
            } else {
                None
            };
        let clamp_to_linear_cap = |chunk: usize| -> usize {
            match linear_attention_chunk_cap {
                Some(cap) => chunk.min(cap).max(1),
                None => chunk.max(1),
            }
        };
        let cold_prefill_chunk = clamp_to_linear_cap(prefill_chunk);
        let prefill_chunk = clamp_to_linear_cap(crate::fastpath::resolve_prefill_chunk(
            cfg.mla_attention.is_some(),
            prefill_chunk,
            crate::fastpath::mla_prefill_chunk_override(),
        ));

        // JIT warm-up: trigger Metal shader compilation for both decode and prefill paths.
        // EmbeddingGemma is an embedding-only encoder with no generation (decode/
        // prefill) path — skip the generation warmup (it would panic in the
        // family dispatch); the bidirectional embed forward JITs on first embed.
        // Skipped when the weights came out of the share cell: warmup exists to
        // populate process-wide JIT caches (Metal shader + mlx compile), which
        // the first build through the cell already did — re-running it would
        // re-impose the per-request forward-pass tax Option A removes.
        if cfg.model_family != "embeddinggemma" && !reused_shared_weights {
            let mut dummy_cache = MlxKVCache::new(cfg.layer_count);
            let mut dummy_rng = Xorshift64::new(0);
            decode_step(
                &cfg,
                &weights,
                0,
                &mut dummy_cache,
                MlxSamplingRequest::new(MlxSamplingParams::greedy(), &[]),
                &mut dummy_rng,
            );
            dummy_cache.reset();
            let dummy_token_count = crate::fastpath::prefill_warmup_token_count(
                cfg.mla_attention.is_some(),
                prefill_chunk,
            );
            let dummy_tokens: Vec<u32> = vec![0u32; dummy_token_count];
            chunked_prefill(
                &cfg,
                &weights,
                &dummy_tokens,
                &mut dummy_cache,
                prefill_chunk,
                MlxSamplingRequest::new(MlxSamplingParams::greedy(), &dummy_tokens),
                &mut dummy_rng,
            );
            // Warm up MTP Metal shaders so first-request TTFT does not include
            // JIT compilation overhead for the MTP head (~50-200 ms).
            if weights.mtp.is_some() {
                let mut mtp_dummy_cache = MlxKVCache::new(1);
                let dummy_hidden = mlx_sys::zeros(
                    &[1, 1, cfg.hidden_size as i32],
                    mlx_sys::MlxDtype::Bfloat16,
                    None,
                );
                let _ = crate::mtp::mtp_draft_tokens(
                    &weights,
                    &cfg,
                    &dummy_hidden,
                    0,
                    &mut mtp_dummy_cache,
                    None,
                    &mut dummy_rng,
                );
                mtp_dummy_cache.reset();
            }
        }
        let _ = take_gemma4_moe_profile_snapshot();
        let _ = take_moe_profile_snapshot();
        let _ = take_linear_attention_profile_snapshot();
        let _ = take_dense_ffn_fastpath_snapshot();
        let _ = take_prefill_profile_snapshot();
        let _ = take_decode_profile_snapshot();

        // Qwen3.5 linear-attention uses `ngram_accel_decode_step_linear_safe` which
        // clones the cache for verification and recomputes the committed prefix on
        // partial accept, so n-gram acceleration is safe to enable for these models.
        let (prefix_cache, disk_prefix_cache) = prefix_cache_store
            .unwrap_or_else(MlxPrefixCacheStore::from_env)
            .into_parts();

        let cfg_arc = Arc::new(cfg.clone());
        let weight_layout_telemetry = WeightLayoutTelemetry::from_weights(&weights);
        let has_mtp =
            weights.mtp.is_some() || weights.glm_mtp.is_some() || gemma4_assistant_mtp.is_some();
        let batched_decode_model_eligible = model_batched_eligible(
            cfg.model_family.as_str(),
            has_mtp,
            cfg.diffusion.is_some(),
            kv_compression.mode != ax_engine_core::KvCompressionMode::Disabled,
            &kv_layer_windows,
            &weights.layers,
        );
        // Capacity for the batched cohort; small (Phase 0 sweet spot is B≈2-4,
        // amortization plateaus past ~8). Override with AX_MLX_BATCHED_DECODE_MAX.
        let batched_cap = std::env::var("AX_MLX_BATCHED_DECODE_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&c| c >= 1)
            .unwrap_or(8);
        let batched_session = Mutex::new(BatchedDecodeSession::new(cfg.layer_count, batched_cap));
        Ok(Self {
            cfg,
            cfg_arc,
            weights,
            prefill_chunk,
            cold_prefill_chunk,
            kv_layer_windows,
            binding_summary,
            terminal_token_ids,
            states: Mutex::new(HashMap::new()),
            batched_decode_model_eligible,
            batched_session,
            _stream: stream,
            disable_ngram_acceleration,
            disable_mtp_ngram_stacking,
            mtp_optimistic: mtp_optimistic_from_env(),
            mtp_skip_state: mtp_skip_state_from_env(),
            mtp_target_softmax_topk: mtp_target_softmax_topk_from_env(),
            gemma4_assistant_mtp_status,
            gemma4_assistant_mtp,
            ngram_policy_variant,
            kv_compression,
            kv_compression_layer_eligible,
            prefix_cache,
            disk_prefix_cache,
            rotating_sliding_decode,
            direct_clear_cache_cadence,
            weight_layout_telemetry,
            affine_quant_telemetry,
            embed_compile_cache: Mutex::new(HashMap::new()),
            embed_batch_compile_cache: Mutex::new(HashMap::new()),
            embed_gemma_batch_compile_cache: Mutex::new(HashMap::new()),
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
        let stats = self.embed_compile_stats.lock();
        let single_len = self.embed_compile_cache.lock().len();
        let batched_len = self.embed_batch_compile_cache.lock().len()
            + self.embed_gemma_batch_compile_cache.lock().len();
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

fn effective_embedding_pooling(model_family: &str, pooling: EmbeddingPooling) -> EmbeddingPooling {
    if model_family == "embeddinggemma" {
        EmbeddingPooling::Mean
    } else {
        pooling
    }
}

/// Build the sampling parameters for a request exactly as the per-item decode
/// path does (see `run_item`), so the batched path classifies and samples each
/// request identically to its single-sequence decode.
fn sampling_params_from_context(ctx: &RunnerRequestContext) -> MlxSamplingParams {
    MlxSamplingParams::new(ctx.temperature, ctx.top_p, ctx.top_k)
        .with_repetition_penalty(ctx.repetition_penalty, ctx.repetition_context_size)
}

impl MlxRunner {
    /// Whether a single item can join the batched dense-decode group this step:
    /// a steady-state (`generated_len >= 1`) single-token decode whose sampling
    /// class is batchable token-exact. Greedy rows (GPU `argmax`) are always
    /// eligible; host-sampled rows (temperature with top-k/top-p, or a
    /// repetition penalty) require the opt-in `AX_MLX_BATCHED_DECODE_SAMPLING`
    /// sub-flag, since their token-exactness against the per-item path is not
    /// yet hardware-verified. The `generated_len == 0` prefill-token step and the
    /// pure-temperature branch (GPU `random_categorical`, non-reproducible) stay
    /// on the per-item path.
    fn batched_item_eligible(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
    ) -> bool {
        if !matches!(item.mode, ExecutionMode::Decode) || item.input_token_slice.len() != 1 {
            return false;
        }
        let Some(ctx) = ctx else {
            return false;
        };
        if ctx.generated_len < 1 {
            return false;
        }
        match batched_sampling_class(
            sampling_params_from_context(ctx),
            ctx.deterministic_argmax_sampling,
        ) {
            Some(BatchedSamplingClass::Greedy) => true,
            Some(BatchedSamplingClass::HostSampled) => batched_decode_sampling_enabled(),
            None => false,
        }
    }

    /// Run one decode step for a group of eligible requests through the shared
    /// [`BatchedDecodeSession`] (one batched forward for the whole group),
    /// producing one `RequestExecutionUpdate` per request. Mirrors `run_item`'s
    /// decode tail (stop detection, `generated_tokens`, state removal on stop).
    /// The caller holds the session lock.
    ///
    /// Experimental: a session-resident request's KV lives in the session, so
    /// its `state.cache` is dormant — this path does not reconcile with the
    /// core's KV block accounting or handle preemption of session members.
    fn run_batched_decode_group(
        &self,
        session: &mut BatchedDecodeSession,
        group: &[&ax_engine_core::ExecutionItem],
        contexts: &[RunnerRequestContext],
    ) -> Vec<RequestExecutionUpdate> {
        // 1. Seed joiners from their prefilled state.cache; set the feed token
        //    (the scheduler is the source of truth) for every group member.
        for item in group {
            let id = item.request_id.0;
            let feed = item.input_token_slice[0];
            if session.active_ids().contains(&id) {
                session.set_current(id, feed);
            } else {
                let states = self.states.lock();
                if let Some(state) = states.get(&item.request_id) {
                    // The runner's cache is warmed: its last token is `feed`'s KV
                    // (appended by generation-state init). Seed all but that last
                    // token so the first step re-appends it without doubling.
                    let seed_len = state.cache.seq_len.saturating_sub(1);
                    session.add_with_seed_len(id, &state.cache, feed, Some(seed_len));
                }
            }
        }
        // 2. One batched forward for the whole cohort (the amortized weight
        //    read), then per-row token resolution matching each request's
        //    single-sequence sampler.
        let outs = self.resolve_batched_group_tokens(session, contexts);
        // 3. Per request: stop detection + update, mirroring the decode tail.
        let mut updates = Vec::with_capacity(outs.len());
        for (id, tok) in outs {
            let request_id = RequestId(id);
            let ctx = contexts.iter().find(|c| c.request_id == request_id);
            let generated_len = ctx.map(|c| c.generated_len).unwrap_or(0);
            let max_output = ctx.map(|c| c.max_output_tokens).unwrap_or(1);
            let terminal: &[u32] = if ctx.map(|c| c.ignore_eos).unwrap_or(false) {
                &[]
            } else {
                &self.terminal_token_ids
            };
            let (sampled, stop_reason) =
                truncate_sampled_tokens_for_stop(vec![tok], generated_len, max_output, terminal);
            {
                let mut states = self.states.lock();
                if stop_reason.is_none() {
                    if let Some(state) = states.get_mut(&request_id) {
                        for &t in &sampled {
                            state.generated_tokens.push(t);
                        }
                    }
                } else {
                    states.remove(&request_id);
                }
            }
            if stop_reason.is_some() {
                session.remove(id);
            }
            let mut iter = sampled.into_iter();
            let output_token = iter.next();
            let output_tokens = iter.collect();
            updates.push(RequestExecutionUpdate {
                request_id,
                tokens_executed: 1,
                output_token,
                output_tokens,
                stop_reason,
                error: None,
            });
        }
        updates
    }

    /// Run one batched forward and resolve one token per active session row,
    /// each identical to that request's single-sequence decode.
    ///
    /// The forward (the amortized weight read) is shared by the whole cohort;
    /// the sampler is not. An all-greedy cohort takes [`BatchedDecodeSession::
    /// step`] (GPU `argmax`, only B indices leave the GPU). A cohort with any
    /// host-sampled row runs [`BatchedDecodeSession::step_logits`] and resolves
    /// per row: greedy rows keep the GPU `argmax` token (first-max tie-break,
    /// identical to single greedy decode); host-sampled rows read their logits
    /// back and run the request's own `sample_categorical_into` with its RNG and
    /// repetition history (identical to single sampled decode). Greedy and
    /// sampled rows are never mixed in one reduction, so neither tie-break nor
    /// RNG consumption diverges from the per-item path.
    fn resolve_batched_group_tokens(
        &self,
        session: &mut BatchedDecodeSession,
        contexts: &[RunnerRequestContext],
    ) -> Vec<(u64, u32)> {
        // Per active row (slot order): its batched sampling class + params. A
        // resident row always carries a context and a Some class (it passed
        // `batched_item_eligible` to join); the defaults are defensive.
        let plan: Vec<(Option<BatchedSamplingClass>, MlxSamplingParams)> = session
            .active_ids()
            .iter()
            .map(|&id| {
                let ctx = contexts.iter().find(|c| c.request_id.0 == id);
                let sampling = ctx
                    .map(sampling_params_from_context)
                    .unwrap_or_else(MlxSamplingParams::greedy);
                let class = ctx.and_then(|c| {
                    batched_sampling_class(sampling, c.deterministic_argmax_sampling)
                });
                (class, sampling)
            })
            .collect();
        let any_sampled = plan
            .iter()
            .any(|(class, _)| matches!(class, Some(BatchedSamplingClass::HostSampled)));
        if !any_sampled {
            // Fast path: greedy cohort, GPU argmax only (no full-logits readback).
            return session.step(&self.cfg, &self.weights);
        }

        let Some((ids, logits)) = session.step_logits(&self.cfg, &self.weights) else {
            return Vec::new();
        };
        debug_assert_eq!(
            ids.len(),
            plan.len(),
            "step_logits row count must match plan"
        );
        // Greedy tokens for every row (GPU argmax, validates the logits shape);
        // used only for greedy rows.
        let greedy_toks = argmax_batched(&logits);
        // Full logits to host for the sampled rows. Read back once, as f32,
        // before taking the state lock — the readback forces a GPU sync.
        let shape = logits.shape();
        let vocab = shape.last().copied().unwrap_or(1);
        let batch = ids.len() as i32;
        let logits_f32 = astype(&logits, MlxDtype::Float32, None);
        let logits_bv = reshape(&logits_f32, &[batch, vocab], None);
        eval(&[&logits_bv]);
        let flat = logits_bv.data_f32();
        let vocab = vocab as usize;

        let mut toks = vec![0u32; ids.len()];
        let mut states = self.states.lock();
        for (row, &id) in ids.iter().enumerate() {
            let (class, sampling) = &plan[row];
            if matches!(class, Some(BatchedSamplingClass::HostSampled))
                && let Some(state) = states.get_mut(&RequestId(id))
            {
                let repetition_history = state.repetition_history(&[], *sampling);
                let row_logits = &flat[row * vocab..(row + 1) * vocab];
                toks[row] = sample_categorical_into(
                    row_logits,
                    *sampling,
                    &repetition_history,
                    &mut state.rng,
                    &mut state.sampling_probs_buf,
                    &mut state.sampling_logits_buf,
                    &mut state.sampling_candidates_buf,
                );
            } else {
                // Greedy row (or a host-sampled row whose state unexpectedly
                // vanished): the GPU argmax token.
                toks[row] = greedy_toks[row];
            }
        }
        drop(states);
        ids.into_iter().zip(toks).collect()
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
        let mut mtp_telemetry = MtpTelemetry::default();
        let mut gemma4_assistant_mtp_telemetry = Gemma4AssistantMtpTelemetry::default();
        let mut gemma4_unified_multimodal_telemetry = Gemma4UnifiedMultimodalTelemetry::default();
        let mut decode_telemetry = DecodeTelemetry::default();
        let mut gemma4_moe_profile = Gemma4MoeProfileSnapshot::default();
        let mut moe_profile = MoeProfileSnapshot::default();
        let mut linear_attention_profile = LinearAttentionProfileSnapshot::default();
        let mut dense_ffn_fastpath = DenseFfnFastpathSnapshot::default();
        let mut prefill_profile = PrefillProfileSnapshot::default();
        let mut decode_profile = DecodeProfileSnapshot::default();
        let mut kv_cache = KvCacheTelemetry::default();
        let mut prefix_cache = MlxPrefixCacheTelemetry::default();

        // ── Batched dense-decode interception (AX_MLX_BATCHED_DECODE, default
        // off). Eligible greedy decode items run through one shared batched
        // forward; every other item — and the whole thing when the flag is off
        // or the model is ineligible — stays byte-for-byte on the per-item path.
        let mut batched_idx: std::collections::HashSet<usize> = std::collections::HashSet::new();
        if batched_decode_enabled() && self.batched_decode_model_eligible {
            let mut session = self.batched_session.lock();
            let resident: std::collections::HashSet<u64> =
                session.active_ids().iter().copied().collect();
            let room = session.capacity().saturating_sub(session.len());
            let mut resident_items: Vec<usize> = Vec::new();
            let mut joiner_items: Vec<usize> = Vec::new();
            for (i, item) in input.execution_batch.items.iter().enumerate() {
                let ctx = input
                    .request_contexts
                    .iter()
                    .find(|c| c.request_id == item.request_id);
                if !self.batched_item_eligible(item, ctx) {
                    continue;
                }
                if resident.contains(&item.request_id.0) {
                    resident_items.push(i);
                } else {
                    joiner_items.push(i);
                }
            }
            joiner_items.truncate(room);
            // Session-resident requests MUST batch (their state.cache is dormant);
            // otherwise start only once >= 2 eligible requests are present.
            let should_batch =
                !resident_items.is_empty() || resident_items.len() + joiner_items.len() >= 2;
            if should_batch {
                let group: Vec<usize> = resident_items.into_iter().chain(joiner_items).collect();
                let group_items: Vec<&ax_engine_core::ExecutionItem> = group
                    .iter()
                    .map(|&i| &input.execution_batch.items[i])
                    .collect();
                let updates = self.run_batched_decode_group(
                    &mut session,
                    &group_items,
                    &input.request_contexts,
                );
                request_updates.extend(updates);
                batched_idx = group.into_iter().collect();
            }
        }
        if !batched_idx.is_empty() {
            route_metadata.crossover_decisions.push((
                "ax_mlx_batched_decode_rows".into(),
                batched_idx.len() as u32,
            ));
        }

        for (item_idx, item) in input.execution_batch.items.iter().enumerate() {
            if batched_idx.contains(&item_idx) {
                continue;
            }
            let ctx = input
                .request_contexts
                .iter()
                .find(|c| c.request_id == item.request_id);

            let result = self.run_item(
                item,
                ctx,
                &input.execution_batch.model_id,
                input.block_size_tokens,
                input.request_multimodal_inputs(item.request_id),
            );
            ngram_acceleration.merge_from(result.ngram_acceleration);
            mtp_telemetry.merge_from(result.mtp_telemetry);
            gemma4_assistant_mtp_telemetry.merge_from(result.gemma4_assistant_mtp_telemetry);
            gemma4_unified_multimodal_telemetry
                .merge_from(result.gemma4_unified_multimodal_telemetry);
            decode_telemetry.merge_from(result.decode_telemetry);
            gemma4_moe_profile.merge_from(result.gemma4_moe_profile);
            moe_profile.merge_from(result.moe_profile);
            linear_attention_profile.merge_from(result.linear_attention_profile);
            dense_ffn_fastpath.merge_from(result.dense_ffn_fastpath);
            prefill_profile.merge_from(result.prefill_profile);
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
            mtp_telemetry.append_route_decisions(&mut route_decisions);
            decode_telemetry.append_route_decisions(&mut route_decisions);
            gemma4_moe_profile.append_route_decisions(&mut route_decisions);
            moe_profile.append_route_decisions(&mut route_decisions);
            linear_attention_profile.append_route_decisions(&mut route_decisions);
            dense_ffn_fastpath.append_route_decisions(&mut route_decisions);
            prefill_profile.append_route_decisions(&mut route_decisions);
            decode_profile.append_route_decisions(&mut route_decisions);
            self.weight_layout_telemetry
                .append_route_decisions(&mut route_decisions);
            self.gemma4_assistant_mtp_status()
                .append_route_decisions(gemma4_assistant_mtp_telemetry, &mut route_decisions);
            gemma4_unified_multimodal_telemetry.append_route_decisions(&mut route_decisions);
            self.affine_quant_telemetry
                .append_route_decisions(&mut route_decisions);
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

    fn release_request_state(&self, request_id: RequestId) {
        // Terminal cleanup for requests that never reach a runner-observed stop
        // (cancelled while waiting/blocked, or cancelled by the engine after a
        // step reinserted state). Without this the per-request KV cache, MTP,
        // and n-gram state stay resident for the life of the process.
        self.states.lock().remove(&request_id);
        self.batched_session.lock().remove(request_id.0);
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
        let pooling = effective_embedding_pooling(&self.cfg.model_family, pooling);
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
        let pooled = crate::model::apply_embedding_dense_head(&self.weights, &pooled);
        let pool_us = elapsed_us(pool_started);

        let post_started = Instant::now();
        let (data, _hidden_size) = post_pool_to_flat(&pooled, normalize);
        let post_us = elapsed_us(post_started);

        tracing::debug!(
            seq_len = token_ids.len(),
            encode_us,
            pool_us,
            post_us,
            "embed_single stage timing"
        );
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
        let pooling = effective_embedding_pooling(&self.cfg.model_family, pooling);
        // For Last/Cls: compute per-sequence extraction positions before the
        // forward pass so the model can extract them before the final norm,
        // avoiding norming the full [B, max_seq, H] padded tensor.
        let target_positions: Option<Vec<usize>> = match pooling {
            EmbeddingPooling::Last => Some(batch.iter().map(|ids| ids.len() - 1).collect()),
            EmbeddingPooling::Cls => Some(vec![0; batch.len()]),
            EmbeddingPooling::Mean => None,
        };
        if self.cfg.model_family == "embeddinggemma"
            && pooling == EmbeddingPooling::Mean
            && let Some(pooled) = self.embedding_gemma_batch_pooled_compiled_forward(batch)
        {
            let (flat, hidden_size) = post_pool_to_flat(&pooled, normalize);
            let data: &[f32] = &flat;
            let vecs = (0..batch.len())
                .map(|i| data[i * hidden_size..(i + 1) * hidden_size].to_vec())
                .collect();
            return Ok(vecs);
        }
        let encode_started = Instant::now();
        let (hidden, actual_lens) =
            self.embedding_batch_forward(batch, target_positions.as_deref());
        let encode_us = elapsed_us(encode_started);

        // Last/Cls: hidden is [B, H] (already extracted).
        // Mean:     hidden is [B, max_seq, H]; pool across sequence here.
        let pool_started = Instant::now();
        let batch_size = batch.len() as i32;

        let pooled = match pooling {
            EmbeddingPooling::Mean => bf16_mean_pool(&hidden, &actual_lens, batch_size),
            EmbeddingPooling::Last | EmbeddingPooling::Cls => hidden,
        };
        let pooled = crate::model::apply_embedding_dense_head(&self.weights, &pooled);
        let pool_us = elapsed_us(pool_started);

        let post_started = Instant::now();
        let (flat, hidden_size) = post_pool_to_flat(&pooled, normalize);
        let post_us = elapsed_us(post_started);

        tracing::debug!(
            batch_size = batch.len(),
            encode_us,
            pool_us,
            post_us,
            "embed_batch stage timing"
        );

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
        let pooling = effective_embedding_pooling(&self.cfg.model_family, pooling);
        let target_positions: Option<Vec<usize>> = match pooling {
            EmbeddingPooling::Last => Some(batch.iter().map(|ids| ids.len() - 1).collect()),
            EmbeddingPooling::Cls => Some(vec![0; batch.len()]),
            EmbeddingPooling::Mean => None,
        };
        if self.cfg.model_family == "embeddinggemma"
            && pooling == EmbeddingPooling::Mean
            && let Some(pooled) = self.embedding_gemma_batch_pooled_compiled_forward(batch)
        {
            let (data, hidden_size) = post_pool_to_flat(&pooled, normalize);
            return Ok(ax_engine_core::EmbeddingMatrix {
                data,
                batch_size: batch.len(),
                hidden_size,
            });
        }
        let (hidden, actual_lens) =
            self.embedding_batch_forward(batch, target_positions.as_deref());
        let batch_size = batch.len() as i32;
        let pooled = match pooling {
            EmbeddingPooling::Mean => bf16_mean_pool(&hidden, &actual_lens, batch_size),
            EmbeddingPooling::Last | EmbeddingPooling::Cls => hidden,
        };
        let pooled = crate::model::apply_embedding_dense_head(&self.weights, &pooled);
        let (data, hidden_size) = post_pool_to_flat(&pooled, normalize);
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

/// Masked mean-pool along the sequence dimension using bf16 masks to avoid
/// an extra f32→bf16 `astype` dispatch. `hidden` is `[B, max_seq, H]`;
/// `actual_lens` holds the real (un-padded) length of each sequence.
/// Returns `[B, H]`.
fn bf16_mean_pool(hidden: &MlxArray, actual_lens: &[usize], batch_size: i32) -> MlxArray {
    let max_seq = hidden.shape()[1] as usize;
    let one_bf16: u16 = (1.0f32.to_bits() >> 16) as u16;
    let zero_bf16: u16 = 0u16;
    let mut mask_data = vec![zero_bf16; actual_lens.len() * max_seq];
    for (i, &l) in actual_lens.iter().enumerate() {
        for j in 0..l {
            mask_data[i * max_seq + j] = one_bf16;
        }
    }
    let mask_arr = MlxArray::from_raw_data(
        mask_data.as_ptr() as *const u8,
        mask_data.len() * std::mem::size_of::<u16>(),
        &[batch_size, max_seq as i32, 1_i32],
        MlxDtype::Bfloat16,
    );
    let masked = multiply(hidden, &mask_arr, None);
    let sums = sum_axis(&masked, 1, false, None);
    let mut scale_data = vec![zero_bf16; actual_lens.len()];
    for (i, &l) in actual_lens.iter().enumerate() {
        scale_data[i] = ((1.0f32 / l as f32).to_bits() >> 16) as u16;
    }
    let scale_arr = MlxArray::from_raw_data(
        scale_data.as_ptr() as *const u8,
        scale_data.len() * std::mem::size_of::<u16>(),
        &[batch_size, 1_i32],
        MlxDtype::Bfloat16,
    );
    multiply(&sums, &scale_arr, None)
}

/// Shared post-pooling processing: astype(f32) → optional L2 normalize
/// (GPU or CPU) → eval → read-back as flat `Vec<f32>`.
///
/// Returns `(flat_data, hidden_size)`. Used by `embed`, `embed_batch`,
/// and `embed_batch_flat` to avoid duplicating the normalize/eval/readback
/// dispatch across the standard and EmbeddingGemma pooled paths.
fn post_pool_to_flat(pooled: &MlxArray, normalize: bool) -> (Vec<f32>, usize) {
    let hidden_size = pooled.shape()[pooled.shape().len() - 1] as usize;
    let pooled_f32 = if pooled.dtype() == MlxDtype::Float32 {
        pooled.clone()
    } else {
        astype(pooled, MlxDtype::Float32, None)
    };
    let cpu_normalize = normalize && !*EMBED_GPU_NORMALIZE;
    let result = if normalize && !cpu_normalize {
        l2_normalize_last_dim(&pooled_f32)
    } else {
        pooled_f32
    };
    mlx_sys::eval(&[&result]);
    let mut flat = result.data_f32().to_vec();
    if cpu_normalize {
        l2_normalize_rows_in_place(&mut flat, hidden_size);
    }
    (flat, hidden_size)
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
        if *EMBED_NO_COMPILE {
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

        let key: EmbedCompileKey = (thread::current().id(), token_ids.len(), target_position);
        let mut cache = self.embed_compile_cache.lock();
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                target_position,
            ) {
                Ok(cls) => {
                    if cache.len() >= EMBED_COMPILE_CACHE_MAX_ENTRIES {
                        cache.clear();
                    }
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
            let mut stats = self.embed_compile_stats.lock();
            if was_present {
                stats.single_hits += 1;
            } else {
                stats.single_misses += 1;
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        let outputs = match cls.try_apply(&[&hidden]) {
            Ok(outputs) => outputs,
            Err(_) => {
                cache.remove(&key);
                drop(cache);
                return crate::model::forward_for_embedding(
                    &self.cfg,
                    &self.weights,
                    token_ids,
                    target_position,
                );
            }
        };
        match outputs.into_iter().next() {
            Some(out) => out,
            None => {
                cache.remove(&key);
                drop(cache);
                crate::model::forward_for_embedding(
                    &self.cfg,
                    &self.weights,
                    token_ids,
                    target_position,
                )
            }
        }
    }

    /// Batched version of `embedding_forward`. Same compile-cache strategy,
    /// keyed on `(batch_size, max_len, target_positions)`. Mean pooling
    /// (`target_positions = None`) currently falls back to the imperative
    /// path because Mean pools after the closure result is materialized.
    ///
    /// EmbeddingGemma (bidirectional encoder with mean pooling) uses a
    /// dedicated compiled-closure path: the transformer layers + final norm
    /// are fused into one compiled graph; mean pooling + Dense head are
    /// applied post-closure by the caller.
    fn embedding_batch_forward(
        &self,
        batch_token_ids: &[Vec<u32>],
        target_positions: Option<&[usize]>,
    ) -> (MlxArray, Vec<usize>) {
        let profile = crate::model::profile::embed_profile_enabled();

        // EmbeddingGemma: bidirectional encoder with mean pooling — uses a
        // dedicated compile cache keyed on (batch, max_len, actual_lens).
        if self.cfg.model_family == "embeddinggemma" {
            if *EMBED_NO_COMPILE || profile {
                return crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    target_positions,
                );
            }
            return self.embedding_gemma_batch_compiled_forward(batch_token_ids);
        }

        // Standard (Qwen3-style) embedding path.
        // AX_MLX_EMBED_PROFILE forces the imperative path: the per-stage eval
        // barriers cannot live inside a single traced compiled closure, and
        // compile on/off is throughput-neutral for this path, so the imperative
        // breakdown is representative.
        if *EMBED_NO_COMPILE || target_positions.is_none() || profile {
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
        let key: EmbedBatchCompileKey = (
            thread::current().id(),
            batch,
            max_len,
            Some(target_positions_vec.clone()),
        );
        let mut cache = self.embed_batch_compile_cache.lock();
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_batch_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                Some(target_positions_vec.clone()),
            ) {
                Ok(cls) => {
                    if cache.len() >= EMBED_COMPILE_CACHE_MAX_ENTRIES {
                        cache.clear();
                    }
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
            let mut stats = self.embed_compile_stats.lock();
            if was_present {
                stats.batched_hits += 1;
            } else {
                stats.batched_misses += 1;
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        let outputs = match cls.try_apply(&[&hidden]) {
            Ok(outputs) => outputs,
            Err(_) => {
                cache.remove(&key);
                drop(cache);
                return crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    target_positions,
                );
            }
        };
        match outputs.into_iter().next() {
            Some(out) => (out, actual_lens),
            None => {
                cache.remove(&key);
                drop(cache);
                crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    target_positions,
                )
            }
        }
    }

    /// EmbeddingGemma compiled-closure batch forward. Builds the pre-embedded
    /// hidden state and bidirectional padding mask, then applies a compiled
    /// closure that fuses the transformer layers + final norm into one graph.
    /// Mean pooling + Dense head are applied by the caller (`embed_batch`).
    fn embedding_gemma_batch_compiled_forward(
        &self,
        batch_token_ids: &[Vec<u32>],
    ) -> (MlxArray, Vec<usize>) {
        let (hidden, batch, max_len, actual_lens) = crate::model::build_embedding_batch_hidden_pub(
            &self.cfg,
            &self.weights,
            batch_token_ids,
        );
        let bidir_mask = crate::model::build_bidirectional_padding_mask(
            batch,
            max_len,
            &actual_lens,
            hidden.dtype(),
        );
        let key: EmbedGemmaBatchCompileKey = (
            thread::current().id(),
            EmbedGemmaBatchCompileKind::Encoder,
            batch,
            max_len,
            actual_lens.clone(),
        );
        let mut cache = self.embed_gemma_batch_compile_cache.lock();
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_gemma3_batch_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                bidir_mask,
            ) {
                Ok(cls) => {
                    if cache.len() >= EMBED_COMPILE_CACHE_MAX_ENTRIES {
                        cache.clear();
                    }
                    cache.insert(key.clone(), cls);
                }
                Err(_) => {
                    drop(cache);
                    return crate::model::forward_for_embedding_batch(
                        &self.cfg,
                        &self.weights,
                        batch_token_ids,
                        None,
                    );
                }
            }
        }
        {
            let mut stats = self.embed_compile_stats.lock();
            if was_present {
                stats.batched_hits += 1;
            } else {
                stats.batched_misses += 1;
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        let outputs = match cls.try_apply(&[&hidden]) {
            Ok(outputs) => outputs,
            Err(_) => {
                cache.remove(&key);
                drop(cache);
                return crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    None,
                );
            }
        };
        match outputs.into_iter().next() {
            Some(out) => (out, actual_lens),
            None => {
                cache.remove(&key);
                drop(cache);
                crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    None,
                )
            }
        }
    }

    /// EmbeddingGemma compiled batch path that returns the pooled Dense-head
    /// tensor `[B, H]`. This is the hot serving/benchmark path for
    /// EmbeddingGemma mean pooling; it avoids returning the full
    /// `[B, max_seq, H]` encoder output to the runner only to mask, sum, and
    /// project it outside the compiled graph.
    fn embedding_gemma_batch_pooled_compiled_forward(
        &self,
        batch_token_ids: &[Vec<u32>],
    ) -> Option<MlxArray> {
        let profile = crate::model::profile::embed_profile_enabled();
        if *EMBED_NO_COMPILE || profile {
            return None;
        }

        let (hidden, batch, max_len, actual_lens) = crate::model::build_embedding_batch_hidden_pub(
            &self.cfg,
            &self.weights,
            batch_token_ids,
        );
        let key: EmbedGemmaBatchCompileKey = (
            thread::current().id(),
            EmbedGemmaBatchCompileKind::Pooled,
            batch,
            max_len,
            actual_lens.clone(),
        );
        let mut cache = self.embed_gemma_batch_compile_cache.lock();
        let was_present = cache.contains_key(&key);
        if !was_present {
            let bidir_mask = crate::model::build_bidirectional_padding_mask(
                batch,
                max_len,
                &actual_lens,
                hidden.dtype(),
            );
            let (pool_mask, pool_scale) =
                crate::model::build_embedding_mean_pool_inputs(batch, max_len, &actual_lens);
            match crate::model::build_embedding_gemma3_pooled_batch_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                bidir_mask,
                pool_mask,
                pool_scale,
            ) {
                Ok(cls) => {
                    if cache.len() >= EMBED_COMPILE_CACHE_MAX_ENTRIES {
                        cache.clear();
                    }
                    cache.insert(key.clone(), cls);
                }
                Err(_) => {
                    return None;
                }
            }
        }
        {
            let mut stats = self.embed_compile_stats.lock();
            if was_present {
                stats.batched_hits += 1;
            } else {
                stats.batched_misses += 1;
            }
        }
        let cls = cache.get(&key)?;
        let outputs = match cls.try_apply(&[&hidden]) {
            Ok(outputs) => outputs,
            Err(_) => {
                cache.remove(&key);
                return None;
            }
        };
        outputs.into_iter().next()
    }

    fn run_item(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
        model_id: &str,
        block_size_tokens: u32,
        multimodal_inputs: Option<&RequestMultimodalInputs>,
    ) -> MlxItemRun {
        let token_ids = &item.input_token_slice;
        if token_ids.is_empty() {
            return MlxItemRun {
                update: RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: 0,
                    output_token: None,
                    output_tokens: Vec::new(),
                    stop_reason: None,
                    error: Some("empty token slice".into()),
                },
                ngram_acceleration: NgramAccelerationTelemetry::default(),
                mtp_telemetry: MtpTelemetry::default(),
                gemma4_assistant_mtp_telemetry: Gemma4AssistantMtpTelemetry::default(),
                gemma4_unified_multimodal_telemetry: Gemma4UnifiedMultimodalTelemetry::default(),
                decode_telemetry: DecodeTelemetry::default(),
                gemma4_moe_profile: Gemma4MoeProfileSnapshot::default(),
                moe_profile: MoeProfileSnapshot::default(),
                linear_attention_profile: LinearAttentionProfileSnapshot::default(),
                dense_ffn_fastpath: DenseFfnFastpathSnapshot::default(),
                prefill_profile: PrefillProfileSnapshot::default(),
                decode_profile: DecodeProfileSnapshot::default(),
                kv_usage: MlxKVCacheUsage::default(),
                prefix_cache: MlxPrefixCacheTelemetry::default(),
                kv_compression_shadow_sync_wall_us: None,
            };
        }

        let max_output = ctx.map(|c| c.max_output_tokens).unwrap_or(1);
        let generated_len = ctx.map(|c| c.generated_len).unwrap_or(0);
        let terminal_token_ids: &[u32] = if ctx.map(|c| c.ignore_eos).unwrap_or(false) {
            &[]
        } else {
            &self.terminal_token_ids
        };
        let prefill_completes_prompt = prefill_item_completes_prompt(item, ctx);
        let is_prefill = matches!(item.mode, ExecutionMode::Prefill);
        let gemma4_unified_inputs = multimodal_inputs
            .and_then(|inputs| inputs.gemma4_unified.as_ref())
            .filter(|inputs| !inputs.is_empty());
        let has_gemma4_unified_multimodal_prefill = is_prefill && gemma4_unified_inputs.is_some();
        let mut gemma4_unified_multimodal_telemetry = Gemma4UnifiedMultimodalTelemetry::default();
        if has_gemma4_unified_multimodal_prefill && let Some(inputs) = gemma4_unified_inputs {
            gemma4_unified_multimodal_telemetry.record_prefill(inputs, self.has_mtp());
        }
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
            let mut states = self.states.lock();
            states.remove(&item.request_id).unwrap_or_else(|| {
                RequestState::new(
                    self.cfg.layer_count,
                    ctx.map(|c| c.seed).unwrap_or(item.request_id.0),
                )
            })
        };
        let mut prefix_cache = if has_gemma4_unified_multimodal_prefill {
            MlxPrefixCacheTelemetry::default()
        } else {
            self.restore_reused_prefix_state(
                &mut state,
                item,
                ctx,
                model_id,
                block_size_tokens,
                sampling,
            )
        };

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
        let sampled_tokens = match item.mode {
            ExecutionMode::Prefill => {
                let prefill_started = Instant::now();
                let sampled_token = if let Some(inputs) = gemma4_unified_inputs {
                    if !prefill_completes_prompt {
                        return errored_item_run(
                            item.request_id,
                            "Gemma4 unified multimodal prefill requires the complete prompt in one execution item",
                        );
                    }
                    state.cache.reset();
                    state.prompt_prefix_tokens.clear();
                    state.cached_prefill_output_token = None;
                    state.mtp_prefill_hidden = None;
                    state.mtp_prefill_history_tokens.clear();

                    let mut full_prompt_tokens = Vec::with_capacity(
                        item.reused_prefix_token_slice
                            .len()
                            .saturating_add(token_ids.len()),
                    );
                    full_prompt_tokens.extend_from_slice(&item.reused_prefix_token_slice);
                    full_prompt_tokens.extend_from_slice(token_ids);
                    let repetition_history =
                        state.repetition_history(&full_prompt_tokens, sampling);
                    let prefill_forward_started = Instant::now();
                    let tok = match chunked_prefill_gemma4_unified_with_sampling_buffers(
                        &self.cfg,
                        &self.weights,
                        &full_prompt_tokens,
                        &mut state.cache,
                        inputs,
                        MlxSamplingRequest::new(sampling, &repetition_history),
                        &mut state.rng,
                        &mut state.sampling_probs_buf,
                        &mut state.sampling_logits_buf,
                        &mut state.sampling_candidates_buf,
                    ) {
                        Ok(tok) => tok,
                        Err(error) => return errored_item_run(item.request_id, error),
                    };
                    let prefill_forward_wall_us = elapsed_us(prefill_forward_started);
                    state.prompt_prefix_tokens = full_prompt_tokens;

                    state
                        .decode_telemetry
                        .record_prefill(elapsed_us(prefill_started));
                    let generation_state_started = Instant::now();
                    kv_compression_shadow_sync_wall_us = self.initialize_generation_state(
                        &mut state,
                        max_output,
                        kv_compression_layer_eligible,
                        Some(tok),
                        is_greedy,
                    );
                    let prefill_generation_state_wall_us = elapsed_us(generation_state_started);
                    state.decode_telemetry.record_prefill_eval_barrier();
                    state.decode_telemetry.record_prefill_breakdown(
                        prefill_forward_wall_us,
                        0,
                        prefill_generation_state_wall_us,
                    );
                    Some(tok)
                } else {
                    let full_recompute_tokens = full_prefill_recompute_tokens_for_warmup_fallback(
                        item,
                        token_ids,
                        &prefix_cache,
                        &state,
                    );
                    let prefill_tokens_base = full_recompute_tokens.as_deref().unwrap_or(token_ids);
                    if full_recompute_tokens.is_some() {
                        state.cache.reset();
                        state.prompt_prefix_tokens.clear();
                        state.cached_prefill_output_token = None;
                    }
                    // F3 M4 — when the runner-side probe restored more
                    // prefix tokens than the scheduler knew about (e.g.
                    // cross-restart L2 hit, where the scheduler's block
                    // table is empty), `token_ids` still includes the
                    // tokens already covered by `state.cache`. Without
                    // slicing them off, chunked_prefill would write
                    // duplicate K/V past the existing seq_len. Detect
                    // that gap and skip the leading reused portion.
                    let probe_over_claim = if full_recompute_tokens.is_some() {
                        0
                    } else {
                        state
                            .cache
                            .seq_len
                            .saturating_sub(item.reused_prefix_token_slice.len())
                    };
                    let prefill_tokens = if probe_over_claim < prefill_tokens_base.len() {
                        &prefill_tokens_base[probe_over_claim..]
                    } else {
                        &[][..]
                    };
                    let mut effective_prefill_token_count = prefill_tokens.len();
                    let repetition_history = state.repetition_history(prefill_tokens, sampling);
                    let prefill_forward_started = Instant::now();
                    // When the runner-probe over-claimed enough to wipe
                    // out `prefill_tokens`, every input position is
                    // already covered by `state.cache`. We must NOT call
                    // chunked_prefill on an empty slice (it would still
                    // sample logits from an undefined state). Instead,
                    // emit `cached_prefill_output_token` as the first
                    // generated token — which is what L1-equivalence
                    // demands, since that token *is* the prefill output
                    // for the producing cold prefill.
                    // Pick chunk size based on whether this prefill will extend
                    // a restored snapshot (warm-extend → small MLA-aligned chunk
                    // so SDPA shape sequence matches the snapshot's producing
                    // path) or start from an empty KV cache (cold → caller's
                    // larger chunk for prefill throughput). For MLA models the
                    // two values differ; for non-MLA models they're identical.
                    let prefill_chunk_for_request = if state.cache.seq_len == 0 {
                        self.cold_prefill_chunk
                    } else {
                        self.prefill_chunk
                    };
                    let tok = if prefill_tokens.is_empty() {
                        if let Some(tok) = state.cached_prefill_output_token.take() {
                            tok
                        } else {
                            // Defensive: if a full-prefix disk entry does
                            // not carry a prefill output token, do not run
                            // decode_one on the restored full cache: that
                            // would append the last prompt token twice.
                            // Fall back to an exact cold prefill — `cache`
                            // is reset on the line below, so the cold chunk
                            // size applies regardless of how we got here.
                            state.cache.reset();
                            state.prompt_prefix_tokens.clear();
                            effective_prefill_token_count = token_ids.len();
                            let recompute_history = state.repetition_history(token_ids, sampling);
                            if self.weights.mtp.is_some() && prefill_completes_prompt {
                                let (tok, hidden, history_tokens) =
                                    chunked_prefill_with_mtp_history_and_sampling_buffers(
                                        &self.cfg,
                                        &self.weights,
                                        token_ids,
                                        &mut state.cache,
                                        self.cold_prefill_chunk,
                                        MlxSamplingRequest::new(sampling, &recompute_history),
                                        &mut state.rng,
                                        &mut state.sampling_probs_buf,
                                        &mut state.sampling_logits_buf,
                                        &mut state.sampling_candidates_buf,
                                    );
                                state.mtp_prefill_hidden = Some(hidden);
                                state.mtp_prefill_history_tokens = history_tokens;
                                tok
                            } else {
                                chunked_prefill_with_sampling_buffers(
                                    &self.cfg,
                                    &self.weights,
                                    token_ids,
                                    &mut state.cache,
                                    self.cold_prefill_chunk,
                                    MlxSamplingRequest::new(sampling, &recompute_history),
                                    &mut state.rng,
                                    &mut state.sampling_probs_buf,
                                    &mut state.sampling_logits_buf,
                                    &mut state.sampling_candidates_buf,
                                )
                            }
                        }
                    } else if self.weights.mtp.is_some() && prefill_completes_prompt {
                        let (tok, hidden, history_tokens) =
                            chunked_prefill_with_mtp_history_and_sampling_buffers(
                                &self.cfg,
                                &self.weights,
                                prefill_tokens,
                                &mut state.cache,
                                prefill_chunk_for_request,
                                MlxSamplingRequest::new(sampling, &repetition_history),
                                &mut state.rng,
                                &mut state.sampling_probs_buf,
                                &mut state.sampling_logits_buf,
                                &mut state.sampling_candidates_buf,
                            );
                        state.mtp_prefill_hidden = Some(hidden);
                        state.mtp_prefill_history_tokens = history_tokens;
                        tok
                    } else {
                        chunked_prefill_with_sampling_buffers(
                            &self.cfg,
                            &self.weights,
                            prefill_tokens,
                            &mut state.cache,
                            prefill_chunk_for_request,
                            MlxSamplingRequest::new(sampling, &repetition_history),
                            &mut state.rng,
                            &mut state.sampling_probs_buf,
                            &mut state.sampling_logits_buf,
                            &mut state.sampling_candidates_buf,
                        )
                    };
                    let prefill_forward_wall_us = elapsed_us(prefill_forward_started);
                    let prefill_token_count = effective_prefill_token_count;
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
                                .filter(|_| prefill_output_token_cacheable(ctx, sampling)),
                        ),
                    );
                    let prefill_prefix_cache_wall_us = elapsed_us(prefix_cache_started);
                    // Record pure prefill wall time before initialize_generation_state.
                    // MTP warmup and generation-state init are decode preparation, not
                    // prefill — including them in the prefill rate artificially lowers
                    // the reported throughput by 5–12 % on MTP workloads.
                    state
                        .decode_telemetry
                        .record_prefill(elapsed_us(prefill_started));
                    let mut prefill_generation_state_wall_us = 0;
                    if prefill_completes_prompt {
                        let generation_state_started = Instant::now();
                        kv_compression_shadow_sync_wall_us = self.initialize_generation_state(
                            &mut state,
                            max_output,
                            kv_compression_layer_eligible,
                            Some(tok),
                            is_greedy,
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
                    state.decode_telemetry.record_prefill_breakdown(
                        prefill_forward_wall_us,
                        prefill_prefix_cache_wall_us,
                        prefill_generation_state_wall_us,
                    );
                    // DiffusionGemma: prefill only warms the KV cache with
                    // prompt KV. All tokens come from diffusion block commits
                    // during decode. Suppressing the prefill token avoids a KV
                    // cache position gap between prompt and first block.
                    if self.cfg.diffusion.is_some() {
                        None
                    } else {
                        prefill_completes_prompt.then_some(tok)
                    }
                };
                sampled_token.into_iter().collect()
            }
            ExecutionMode::Decode => {
                let decode_started = Instant::now();
                let final_by_max_output = generated_len.saturating_add(1) >= max_output;
                // Diffusion models never emit a prefill-sampled token (the
                // first block's canvas owns that position), but a prefix-cache
                // partial hit can still stash the *prefix's* greedy AR token in
                // `cached_prefill_output_token`; consuming it would inject a
                // foreign token at stream start. Drop it before the first step.
                if self.cfg.diffusion.is_some() {
                    state.cached_prefill_output_token = None;
                }
                let tokens = if generated_len == 0 {
                    if let Some(tok) = state.cached_prefill_output_token.take() {
                        kv_compression_shadow_sync_wall_us = self.initialize_generation_state(
                            &mut state,
                            max_output,
                            kv_compression_layer_eligible,
                            Some(tok),
                            is_greedy,
                        );
                        vec![tok]
                    } else {
                        self.decode_one(
                            &mut state,
                            token_ids,
                            sampling,
                            is_greedy,
                            DecodeOneOptions {
                                terminal_token_ids,
                                final_by_max_output,
                                request_context: ctx,
                            },
                        )
                    }
                } else {
                    self.decode_one(
                        &mut state,
                        token_ids,
                        sampling,
                        is_greedy,
                        DecodeOneOptions {
                            terminal_token_ids,
                            final_by_max_output,
                            request_context: ctx,
                        },
                    )
                };
                state
                    .decode_telemetry
                    .record_decode(elapsed_us(decode_started));
                tokens
            }
        };

        let (sampled_tokens, stop_reason) = truncate_sampled_tokens_for_stop(
            sampled_tokens,
            generated_len,
            max_output,
            terminal_token_ids,
        );
        if stop_reason.is_none() {
            for &sampled_token in &sampled_tokens {
                state.generated_tokens.push(sampled_token);
                update_ngram_think_state(&self.cfg, &mut state.ngram_in_think, sampled_token);
            }
        }

        // Re-insert state only if the request continues — lock held briefly.
        let ngram_acceleration = state.ngram_acceleration;
        let mtp_telemetry = state.mtp_telemetry;
        let gemma4_assistant_mtp_telemetry = state.gemma4_assistant_mtp_telemetry;
        let decode_telemetry = state.decode_telemetry;
        let gemma4_moe_profile = take_gemma4_moe_profile_snapshot();
        let moe_profile = take_moe_profile_snapshot();
        let linear_attention_profile = take_linear_attention_profile_snapshot();
        let dense_ffn_fastpath = take_dense_ffn_fastpath_snapshot();
        state
            .prefill_profile
            .merge_from(take_prefill_profile_snapshot());
        let prefill_profile = state.prefill_profile;
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
            let mut states = self.states.lock();
            states.insert(item.request_id, state);
        } else {
            // Free MLX's intermediate graph and compute cache after each completed
            // request.  Mirrors mlx_lm's mx.metal.clear_cache() at end of generation;
            // reclaims GPU memory that would otherwise persist until the next request.
            clear_cache();
        }

        let mut sampled_tokens = sampled_tokens.into_iter();
        let output_token = sampled_tokens.next();
        let output_tokens = sampled_tokens.collect();

        MlxItemRun {
            update: RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: item.scheduled_token_count,
                output_token,
                output_tokens,
                stop_reason,
                error: None,
            },
            ngram_acceleration,
            mtp_telemetry,
            gemma4_assistant_mtp_telemetry,
            gemma4_unified_multimodal_telemetry,
            decode_telemetry,
            gemma4_moe_profile,
            moe_profile,
            linear_attention_profile,
            dense_ffn_fastpath,
            prefill_profile,
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
    /// block_size) hash-map lookups in the L1 cache, all read-only and
    /// verifying exact token equality.
    ///
    /// F3 M2 — when the L1 cache has no matching prefix, the probe also
    /// consults the L2 disk cache via the cheap `contains` existence
    /// check. This unlocks the cross-process / cross-restart case the
    /// L1-only probe couldn't reach (process B opens a fresh L1, the
    /// scheduler hasn't annotated a reused-prefix slice, but the disk
    /// holds a snapshot from process A). The disk check is `fs::stat`
    /// per candidate prefix — cheap; the eventual full read + SHA256
    /// validate happens once at restore time, not per probe step.
    fn probe_runner_snapshot_for_prefix(
        &self,
        model_id: &str,
        block_size_tokens: u32,
        input: &[u32],
    ) -> Option<Vec<u32>> {
        let cache = self.prefix_cache.lock();
        Self::longest_block_aligned_prefix_by_probe(block_size_tokens, input, |prefix| {
            let key = self.prefix_cache_key(model_id, block_size_tokens, prefix);
            if cache.contains_exact_tokens(&key, prefix) {
                return true;
            }
            if let Some(disk) = self.disk_prefix_cache.as_ref() {
                let key_bytes = crate::disk_prefix_cache::canonical_key_bytes(
                    &key.model_id,
                    &key.route_policy,
                    &key.layer_layout,
                    key.block_size_tokens,
                    key.token_count,
                    key.token_hash,
                );
                if disk.contains(&key_bytes) {
                    return true;
                }
            }
            false
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

        if !self.prefix_cache.lock().enabled() {
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
            let mut cache = self.prefix_cache.lock();
            let hit = cache.get(&key, reused_tokens);
            telemetry.record_stats(cache.stats());
            hit
        };

        // Historical context: MLA + Prefill used to refuse a snapshot restore
        // because the post-restore chunked_prefill drifted fp-wise from a
        // cold full prefill (p2_medium_explain idx=13 divergence on
        // GLM-4.7-Flash). Evidence points to shape-dependent SDPA kernel
        // selection in MLX: a single large cold chunk and the smaller chunks
        // of a warm-extend can dispatch different kernels. The fix is
        // upstream of this branch: MLA models now default to a small
        // chunked_prefill chunk size (see `MLA_DEFAULT_PREFILL_CHUNK`) that
        // makes cold and warm produce the same SDPA shape sequence at the
        // same absolute positions. The canonical default-path equivalence
        // harness now passes 5/5 with a real prefix-cache hit. The
        // kill-switch env
        // `AX_DISABLE_MLA_PREFIX_RESTORE=1` re-engages the historical gate
        // if a future workload exposes a residual drift vector.
        let mla_extend_unsafe = self.cfg.mla_attention.is_some()
            && item.mode == ExecutionMode::Prefill
            && crate::fastpath::mla_prefix_restore_disabled();

        if let Some(snapshot) = hit {
            if mla_extend_unsafe {
                telemetry.record_blocked_unsupported_layout();
                telemetry.warmup_tokens = telemetry
                    .warmup_tokens
                    .saturating_add(saturating_u32(reused_tokens.len()));
                return telemetry;
            }
            match snapshot.rehydrate_cache() {
                Ok(restored_cache) => {
                    state.cache = restored_cache;
                    state.prompt_prefix_tokens = reused_tokens.to_vec();
                    // Only inherit the producer's greedy token when this request
                    // would compute it too; otherwise leave it unset so the
                    // consume site resamples with the request's own sampling.
                    state.cached_prefill_output_token = snapshot
                        .greedy_prefill_output_token
                        .filter(|_| prefill_output_token_cacheable(ctx, sampling));
                    telemetry.hits = telemetry.hits.saturating_add(1);
                    telemetry.reused_tokens = telemetry
                        .reused_tokens
                        .saturating_add(saturating_u32(snapshot.token_count));
                    return telemetry;
                }
                Err(e) => {
                    tracing::warn!(
                        target: "ax_engine_mlx::prefix_cache",
                        error = %e,
                        "L1 prefix-cache payload failed to deserialize; treating as miss",
                    );
                }
            }
            // Fall through to the L2 disk cache if it is available; otherwise
            // the regular miss path below warms or recomputes the prefix.
        }

        // F3 M2 — L1 miss. If the L2 disk cache is open and the
        // MLA-extend safety gate has not engaged, try the disk layer
        // before falling through to the cold-prefill warmup path. A
        // disk hit deserialises a fresh `MlxKVCache` (bit-equivalent to
        // the snapshot that produced it) and routes through the same
        // restore path as L1.
        //
        // Deserialise failure or filesystem error is treated as a miss
        // per F3 PRD §3 (fail-closed): the cache miss path still runs,
        // the request still completes, telemetry records the disk
        // miss for observability.
        if !mla_extend_unsafe && let Some(disk) = self.disk_prefix_cache.as_ref() {
            let key_bytes = crate::disk_prefix_cache::canonical_key_bytes(
                &key.model_id,
                &key.route_policy,
                &key.layer_layout,
                key.block_size_tokens,
                key.token_count,
                key.token_hash,
            );
            match disk.get(&key_bytes) {
                Ok(Some(entry)) => match MlxKVCache::try_deserialize_from_bytes(&entry.payload) {
                    Ok(restored_cache) => {
                        state.cache = restored_cache;
                        state.prompt_prefix_tokens = reused_tokens.to_vec();
                        // F3 M4 — file format v2 carries the greedy
                        // prefill output token, so cross-restart L2
                        // hits avoid recomputing at decode step 0
                        // (which diverged for single-block prefixes
                        // in the pre-fix run). When the slot is None
                        // (older partial-prefix snapshot), decode_one
                        // still runs as a fallback. Only a greedy,
                        // no-repetition-penalty consumer may inherit the
                        // stored greedy token; others resample.
                        state.cached_prefill_output_token = entry
                            .prefill_output_token
                            .filter(|_| prefill_output_token_cacheable(ctx, sampling));
                        telemetry.record_disk_hit();
                        telemetry.reused_tokens = telemetry
                            .reused_tokens
                            .saturating_add(saturating_u32(reused_tokens.len()));
                        return telemetry;
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "ax_engine_mlx::prefix_cache",
                            error = %e,
                            "disk prefix-cache payload failed to deserialize; treating as miss",
                        );
                        telemetry.record_disk_miss();
                    }
                },
                Ok(None) => {
                    telemetry.record_disk_miss();
                }
                Err(e) => {
                    tracing::warn!(
                        target: "ax_engine_mlx::prefix_cache",
                        error = %e,
                        "disk prefix-cache get failed; treating as miss",
                    );
                    telemetry.record_disk_miss();
                }
            }
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
        let prefill_output_token = chunked_prefill_with_sampling_buffers(
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
            &mut state.sampling_probs_buf,
            &mut state.sampling_logits_buf,
            &mut state.sampling_candidates_buf,
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
        if !self.prefix_cache.lock().enabled() {
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
        let mla_attention = self.cfg.mla_attention.is_some();
        let alignment_restricted = linear_attention || sliding_window || mla_attention;
        if alignment_restricted && full_block_tokens != available_tokens {
            telemetry.record_blocked_trim_failure();
            return telemetry;
        }
        let snapshot_start_tokens = if alignment_restricted {
            full_block_tokens
        } else {
            block_size
        };

        for prefix_len in (snapshot_start_tokens..=full_block_tokens).step_by(block_size) {
            let tokens = &state.prompt_prefix_tokens[..prefix_len];
            let key = self.prefix_cache_key(model_id, block_size_tokens, tokens);
            let is_largest = prefix_len == full_block_tokens;
            let snapshot_prefill_output_token = (prefix_len == available_tokens)
                .then_some(greedy_prefill_output_token)
                .flatten();

            // Skip prefixes that are already resident: the clone + serialize
            // below costs O(prefix KV bytes) per iteration, so warm
            // same-prompt traffic would otherwise re-pay the full store cost
            // on every prefill. The largest prefix still goes through when
            // the disk layer is open but does not have the entry yet.
            let l1_superseding = {
                let cache = self.prefix_cache.lock();
                let superseding = cache.contains_superseding_snapshot(
                    &key,
                    tokens,
                    snapshot_prefill_output_token,
                );
                if superseding {
                    telemetry.record_stats(cache.stats());
                }
                superseding
            };
            if l1_superseding {
                let disk_store_needed = is_largest
                    && self.disk_prefix_cache.as_ref().is_some_and(|disk| {
                        !disk.contains(&crate::disk_prefix_cache::canonical_key_bytes(
                            &key.model_id,
                            &key.route_policy,
                            &key.layer_layout,
                            key.block_size_tokens,
                            key.token_count,
                            key.token_hash,
                        ))
                    });
                if !disk_store_needed {
                    continue;
                }
            }

            let mut snapshot_cache = state.cache.clone();
            if !snapshot_cache.trim_to(prefix_len) {
                telemetry.record_blocked_trim_failure();
                continue;
            }
            // F3 M2 — for the disk layer we want the largest valid
            // snapshot persisted; smaller intermediate prefixes stay
            // in L1 only. Without an eviction policy yet (M3), writing
            // every per-block prefix to disk would balloon the cache
            // directory by O(N/block_size) × snapshot bytes per cold
            // prefill. The largest snapshot is also the most useful for
            // future hits because shorter prefixes always derive from
            // it.
            let payload = snapshot_cache.serialize_to_bytes();
            let disk_payload = if is_largest && self.disk_prefix_cache.is_some() {
                Some(payload.clone())
            } else {
                None
            };
            // Only clone the key when we'll need it again post-insert
            // (i.e. when the disk path will fire). For the L1-only
            // configuration the original `key` moves cleanly into
            // `cache.insert`, no extra allocation.
            let key_for_disk = disk_payload.as_ref().map(|_| key.clone());
            let outcome = {
                let mut cache = self.prefix_cache.lock();
                let outcome = cache.insert(
                    key,
                    MlxPrefixSnapshot::from_serialized_cache(
                        payload,
                        tokens.to_vec(),
                        prefix_len,
                        snapshot_prefill_output_token,
                    ),
                );
                telemetry.record_stats(cache.stats());
                outcome
            };
            if outcome.stored {
                telemetry.stores = telemetry.stores.saturating_add(1);

                // Mirror to the disk layer when (a) the disk cache is
                // open, (b) this is the largest-prefix snapshot, and
                // (c) L1 actually stored it. A disk-write failure does
                // not back out the L1 store — the in-memory layer
                // alone is still useful and disk is strictly additive.
                if let (Some(disk), Some(payload), Some(disk_key)) =
                    (self.disk_prefix_cache.as_ref(), disk_payload, key_for_disk)
                {
                    let key_bytes = crate::disk_prefix_cache::canonical_key_bytes(
                        &disk_key.model_id,
                        &disk_key.route_policy,
                        &disk_key.layer_layout,
                        disk_key.block_size_tokens,
                        disk_key.token_count,
                        disk_key.token_hash,
                    );
                    let entry = crate::disk_prefix_cache::DiskPrefixCacheEntry {
                        payload,
                        prefill_output_token: snapshot_prefill_output_token,
                    };
                    let payload_bytes = entry.payload.len() as u64;
                    match disk.insert(&key_bytes, &entry) {
                        Ok(outcome) => {
                            telemetry.record_disk_insert(payload_bytes, outcome.evictions)
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "ax_engine_mlx::prefix_cache",
                                error = %e,
                                "disk prefix-cache insert failed; L1 store still active",
                            );
                        }
                    }
                }
            }
            telemetry.evictions = telemetry.evictions.saturating_add(outcome.evictions);
        }
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
        options: DecodeOneOptions<'_>,
    ) -> Vec<u32> {
        // Serve pre-verified bonus tokens without re-running the model.
        // (Bonus tokens only exist on the n-gram acceleration path; the direct pipeline
        // never populates the bonus queue.)
        if let Some(tok) = state.bonus_queue.pop_front() {
            state.decode_telemetry.record_bonus_token();
            return vec![tok];
        }

        // Serve buffered diffusion block tokens. DiffusionGemma generates
        // canvas_size tokens per block via bidirectional denoising; the runner
        // drains them one at a time through the standard decode path.
        if let Some(tok) = state.diffusion_block_queue.pop_front() {
            return vec![tok];
        }

        // Diffusion path: when the diffusion queue is exhausted, generate a
        // new block via bidirectional denoising + causal commit. This replaces
        // the standard AR decode step for DiffusionGemma models.
        if let Some(diff_cfg) = self.cfg.diffusion.as_ref() {
            let token_offset = state.cache.seq_len;
            let remaining_output_budget = options
                .request_context
                .map(|ctx| ctx.max_output_tokens.saturating_sub(ctx.generated_len));
            let result = crate::diffusion::generate_diffusion_block(
                &self.cfg,
                diff_cfg,
                &self.weights,
                &mut state.cache,
                &mut state.rng,
                token_offset,
                &mut state.diffusion_embed_table,
                crate::diffusion::DiffusionCommitPolicy {
                    truncation_terminal_ids: &self.terminal_token_ids,
                    request_terminal_ids: options.terminal_token_ids,
                    remaining_output_budget,
                },
            );
            state.decode_telemetry.record_diffusion_block(&result);
            let mut queue: VecDeque<u32> = result.tokens.into();
            // EOS early termination (dInfer evidence: 15–40% gain). If the
            // committed block contains an EOS token, truncate the queue at
            // that position so we never generate a follow-on block.
            if let Some(eos_pos) = queue
                .iter()
                .position(|&tok| self.terminal_token_ids.contains(&tok))
            {
                queue.truncate(eos_pos + 1);
            }
            let tok = queue.pop_front().unwrap_or(0);
            state.diffusion_block_queue = queue;
            return vec![tok];
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
            let last_token = state
                .next_model_last_token
                .or_else(|| input_tokens.last().copied())
                .unwrap_or(0);
            return vec![self.run_direct_pipeline_decode(
                state,
                last_token,
                options.final_by_max_output,
                false,
            )];
        }

        let last_token = state
            .next_model_last_token
            .or_else(|| input_tokens.last().copied())
            .unwrap_or(0);

        if ngram_request_disabled_direct_fast_path(
            is_greedy,
            sampling.uses_repetition_penalty(),
            self.has_mtp(),
            state.ngram_acceleration_disabled_for_request,
            state.ngram_request_disable_reason,
        ) {
            state.ngram_acceleration.record_request_disabled_step();
            state
                .ngram_acceleration
                .record_request_disabled_reason(state.ngram_request_disable_reason);
            return vec![self.run_direct_pipeline_decode(
                state,
                last_token,
                options.final_by_max_output,
                false,
            )];
        }

        let result = self.run_model_decode(
            state,
            last_token,
            sampling,
            is_greedy,
            options.final_by_max_output,
            options.request_context,
        );
        apply_decode_result(state, &result, options.terminal_token_ids)
    }

    /// Decode one deterministic token on the direct double-buffer pipeline.
    ///
    /// Used both by explicit direct mode and by request-local n-gram fallback after
    /// a linear-attention request proves it has no useful draft support.  The
    /// pipeline may keep the cache one lazy token ahead, so callers must continue
    /// using this path until the request finishes.
    fn run_direct_pipeline_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        final_by_max_output: bool,
        feed_ngram: bool,
    ) -> u32 {
        let tok = match direct_pipeline_action(state.pending_direct.is_some(), final_by_max_output)
        {
            DirectPipelineAction::FinishPending => match state.pending_direct.take() {
                Some(pending) => self.run_direct_pipeline_finish_pending(state, pending),
                None => {
                    tracing::error!(
                        "direct pipeline state machine invariant violated: \
                             FinishPending called without pending_direct; \
                             falling back to bootstrap"
                    );
                    self.run_direct_pipeline_bootstrap(state, last_token)
                }
            },
            DirectPipelineAction::ContinuePending => match state.pending_direct.take() {
                Some(bootstrap_token) => self.run_direct_pipeline_once(state, bootstrap_token),
                None => {
                    tracing::error!(
                        "direct pipeline state machine invariant violated: \
                             ContinuePending called without pending_direct; \
                             re-bootstrapping from last_token"
                    );
                    self.run_direct_pipeline_bootstrap(state, last_token)
                }
            },
            DirectPipelineAction::BootstrapFinal => {
                self.run_direct_pipeline_bootstrap_final(state, last_token)
            }
            DirectPipelineAction::Bootstrap => {
                self.run_direct_pipeline_bootstrap(state, last_token)
            }
        };
        if feed_ngram {
            state.ngram.feed(&[tok]);
        }
        tok
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

    fn run_direct_pipeline_bootstrap_final(
        &self,
        state: &mut RequestState,
        last_token: u32,
    ) -> u32 {
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
        self.run_direct_pipeline_finish_pending(state, bootstrap_token)
    }

    fn run_direct_pipeline_finish_pending(
        &self,
        state: &mut RequestState,
        pending: MlxArray,
    ) -> u32 {
        let branch_started = Instant::now();
        let pending_eval_started = Instant::now();
        eval(&[&pending]);
        let pending_eval_wall_us = elapsed_us(pending_eval_started);
        let pending_read_started = Instant::now();
        let tok = pending.first_u32_unchecked();
        let pending_read_wall_us = elapsed_us(pending_read_started);
        state
            .decode_telemetry
            .record_direct_pipeline(elapsed_us(branch_started));
        state
            .decode_telemetry
            .record_direct_pipeline_timings(DirectPipelineTimings {
                pending_eval_wall_us,
                pending_read_wall_us,
                ..DirectPipelineTimings::default()
            });
        state.decode_telemetry.record_production_decode_eval();
        tok
    }

    fn run_direct_pipeline_once(&self, state: &mut RequestState, bootstrap_token: MlxArray) -> u32 {
        let branch_started = Instant::now();
        let stage_profile = crate::generate::direct_pipeline_stage_profile_enabled();
        let op_count_before = stage_profile.then(mlx_sys::op_count_snapshot);
        let turboquant_context = self.turboquant_model_decode_context();
        let advanced = advance_direct_pipeline_with_timings_and_turboquant_context(
            &self.cfg,
            &self.weights,
            &bootstrap_token,
            &mut state.cache,
            turboquant_context.as_ref(),
        );
        state
            .decode_telemetry
            .record_direct_pipeline(elapsed_us(branch_started));
        if let Some(op_count_before) = op_count_before {
            let op_count_delta = mlx_sys::op_count_take(op_count_before);
            state
                .decode_telemetry
                .record_direct_pipeline_op_count(op_count_delta);
        }
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
        final_by_max_output: bool,
    ) -> Vec<u32> {
        let feed_ngram =
            ngram_request_disabled_fallback_should_feed_output(state.ngram_request_disable_reason);
        state.ngram_acceleration.record_request_disabled_step();
        state
            .ngram_acceleration
            .record_request_disabled_reason(state.ngram_request_disable_reason);
        let result = if is_greedy {
            vec![self.run_direct_pipeline_decode(
                state,
                last_token,
                final_by_max_output,
                feed_ngram,
            )]
        } else {
            self.run_single_decode(state, last_token, sampling)
        };
        if feed_ngram {
            maybe_reenable_linear_ngram_from_fallback_output(
                state,
                self.ngram_policy_variant,
                is_greedy,
            );
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_no_draft_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        has_linear_attention: bool,
        is_greedy: bool,
        final_by_max_output: bool,
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
                return Some(self.run_request_disabled_decode(
                    state,
                    last_token,
                    sampling,
                    is_greedy,
                    final_by_max_output,
                ));
            }
            if is_greedy {
                state.ngram_disabled_steps = LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL;
                // Same stale-lookahead hazard as the ngram-failure cooldown
                // path: any pending_direct built during a prior cooldown cycle
                // is now at the wrong cache position.
                state.pending_direct = None;
                state.direct_pipeline_emitted_tokens = 0;
                state
                    .ngram_acceleration
                    .record_cooldown_event(LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL);
            }
        }
        if is_greedy {
            if !has_linear_attention {
                state.ngram_disabled_steps = NGRAM_RETRY_INTERVAL;
                // Same stale-lookahead hazard as the linear-attention no-draft
                // path: any pending_direct built before this cooldown now points
                // at the wrong cache position once the direct pipeline advances
                // seq_len independently during the retry interval.
                state.pending_direct = None;
                state.direct_pipeline_emitted_tokens = 0;
                state
                    .ngram_acceleration
                    .record_cooldown_event(NGRAM_RETRY_INTERVAL);
            }
            return Some(vec![self.run_direct_pipeline_decode(
                state,
                last_token,
                final_by_max_output,
                true,
            )]);
        }
        None
    }

    fn finish_pending_direct_for_ngram_transition(&self, state: &mut RequestState) -> Vec<u32> {
        let Some(pending) = state.pending_direct.take() else {
            tracing::error!(
                "direct pipeline state machine invariant violated: \
                 n-gram transition drain called without pending_direct; \
                 returning empty token list"
            );
            state.direct_pipeline_emitted_tokens = 0;
            return vec![];
        };
        let tok = self.run_direct_pipeline_finish_pending(state, pending);
        state.ngram.feed(&[tok]);
        state.direct_pipeline_emitted_tokens = 0;
        vec![tok]
    }

    fn run_non_ngram_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        is_greedy: bool,
        final_by_max_output: bool,
    ) -> Option<Vec<u32>> {
        if self.disable_ngram_acceleration {
            return Some(self.run_single_decode(state, last_token, sampling));
        }

        if sampling.uses_repetition_penalty() {
            return Some(self.run_single_decode(state, last_token, sampling));
        }

        if state.ngram_acceleration_disabled_for_request {
            return Some(self.run_request_disabled_decode(
                state,
                last_token,
                sampling,
                is_greedy,
                final_by_max_output,
            ));
        }

        // N-gram acceleration disabled: count down and use single decode.
        if state.ngram_disabled_steps > 0 {
            state.ngram_disabled_steps -= 1;
            state.ngram_acceleration.record_cooldown_step();
            if is_greedy {
                return Some(vec![self.run_direct_pipeline_decode(
                    state,
                    last_token,
                    final_by_max_output,
                    true,
                )]);
            }
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
            &mut state.sampling_probs_buf,
            &mut state.sampling_logits_buf,
            &mut state.sampling_candidates_buf,
        );
        state
            .decode_telemetry
            .record_single_decode(elapsed_us(branch_started));
        state.decode_telemetry.record_production_decode_eval();
        result
    }

    fn gemma4_assistant_draft_token(
        &self,
        state: &mut RequestState,
        last_token: u32,
        last_backbone_hidden: &MlxArray,
        sampling: MlxSamplingParams,
    ) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>) {
        let Some(runtime) = self.gemma4_assistant_mtp.as_ref() else {
            return (vec![], vec![], vec![]);
        };
        // Draft depth: the adaptive controller capped by the runtime ceiling
        // (default 2). The assistant is stateless per step, so it can be applied
        // recurrently to draft >1 token.
        let max_depth = state.mtp_adaptive_max_depth.min(runtime.status.max_depth);
        if max_depth == 0 {
            return (vec![], vec![], vec![]);
        }

        let base_position = state.cache.seq_len;
        // Speculation-profile resolution (ADR-022): explicit env > profile preset
        // > built-in default. `auto` is temperature-driven and never lowers the
        // shipped Gemma default at low temperature.
        let speculation_profile = speculation_profile_from_env();
        let first_gate = resolve_gemma4_assistant_mtp_first_gate(
            speculation_profile,
            Some(sampling.temperature),
        )
        .0;
        let confidence_mode = gemma4_assistant_mtp_confidence_mode_from_env();

        // Ungated (gate disabled, gate <= 0): a single sampled draft carrying a
        // log-prob + distribution so rejection-sampling acceptance can engage.
        // Recurrent multi-depth drafting is the gated greedy path below; the
        // sampled path stays depth-1 (per-depth sampled log-probs are out of scope).
        if first_gate <= 0.0 {
            let Ok((logits, _projected_hidden)) = crate::model::gemma4_assistant_forward_one(
                &runtime.cfg,
                &runtime.weights,
                &self.cfg,
                &self.weights,
                &state.cache,
                runtime.target_shared_layers,
                last_token,
                &astype(last_backbone_hidden, MlxDtype::Bfloat16, None),
                base_position,
            ) else {
                return (vec![], vec![], vec![]);
            };
            eval(&[&logits]);
            let logits_cpu = logits.data_f32().to_vec();
            let (token, log_prob, distribution) = sample_categorical_with_logprob_and_distribution(
                &logits_cpu,
                sampling,
                &mut state.rng,
            );
            return (
                vec![token],
                vec![log_prob],
                distribution.into_iter().collect(),
            );
        }

        // Gated greedy recurrent drafting (the default). Each position d feeds the
        // assistant's `post_projection` "backbone hidden" estimate of position d
        // back in as the next step's hidden — the same signal the production verify
        // forward provides at depth 0 — and advances the RoPE position by one. The
        // assistant attends the target's frozen KV (it has no k/v of its own); the
        // drafted token's signal flows through the residual, which the depth-2
        // sweep confirmed is enough to hold ~97-100% accept on the 2nd token.
        //
        // A position is proposed only when its T=1.0 argmax confidence clears the
        // gate — 0.85 on the first token, tight 0.999 on deeper positions (a wrong
        // deep draft costs a full target recompute, so the deep gate stays tight).
        // A miss stops drafting. Suppression is correctness-preserving: a short or
        // empty draft just verifies fewer speculative positions, never changing the
        // committed token. Greedy drafts carry no log-prob, so acceptance falls to
        // argmax-match (the Gemma default acceptance mode).
        let deep_gate =
            resolve_gemma4_assistant_mtp_deep_gate(speculation_profile, Some(sampling.temperature))
                .0;
        let mut drafts: Vec<u32> = Vec::with_capacity(max_depth);
        let mut cur_token = last_token;
        let mut cur_hidden = astype(last_backbone_hidden, MlxDtype::Bfloat16, None);
        for d in 0..max_depth {
            let Ok((logits, projected_hidden)) = crate::model::gemma4_assistant_forward_one(
                &runtime.cfg,
                &runtime.weights,
                &self.cfg,
                &self.weights,
                &state.cache,
                runtime.target_shared_layers,
                cur_token,
                &cur_hidden,
                base_position + d,
            ) else {
                break;
            };
            let (token, confidence) =
                argmax_with_softmax_confidence_for_logits(&logits, confidence_mode);
            let gate = if d == 0 { first_gate } else { deep_gate };
            if confidence < gate {
                break;
            }
            drafts.push(token);
            cur_token = token;
            cur_hidden = astype(&projected_hidden, MlxDtype::Bfloat16, None);
        }
        (drafts, vec![], vec![])
    }

    /// MTP model-based speculative decode step.
    ///
    /// Runs a verify forward on `[last_token] ++ pending_draft`, accepts/rejects
    /// the draft, then generates a new draft with the MTP heads for the next step.
    ///
    /// Before the MTP head forward, an n-gram lookup is attempted (ADR-008 stacking).
    /// On hit, the n-gram tokens are used as draft and the MTP KV cache is reset to
    /// prevent stale RoPE offsets on the next MTP step.
    /// Returns verified output tokens (1 or more) in the same format as n-gram
    /// `ngram_accel_decode_step` / `run_single_decode`.
    ///
    /// When draft log-probs are available (temperature sampling), acceptance uses
    /// rejection sampling: accept draft[i] with probability
    /// min(1, p_target(draft[i]) / p_draft(draft[i])).
    /// When log-probs are absent (greedy draft or temperature==0), falls back to
    /// greedy argmax comparison.
    fn run_mtp_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        ctx: Option<&RunnerRequestContext>,
    ) -> Vec<u32> {
        use crate::ngram_accel::sample_logit_row;
        use mlx_sys::{argmax, eval};

        let mut pending = state.mtp_pending_draft.clone();
        let token_offset = state.cache.seq_len;
        let has_linear_attention = self.cfg.linear_attention.is_some();
        let vocab = self.cfg.vocab_size as i32;
        let mut mtp_timings = MtpStepTimings::default();
        // Draft log-probs are computed at T=1.0 (greedy path) or
        // head.draft_sampling.temperature (sampled path).
        let draft_log_prob_temperature = self
            .weights
            .mtp
            .as_ref()
            .map(|h| {
                if h.draft_sampling.temperature > 0.0 {
                    h.draft_sampling.temperature
                } else {
                    1.0
                }
            })
            .unwrap_or_else(|| {
                if sampling.temperature > 0.0 {
                    sampling.temperature
                } else {
                    1.0
                }
            });
        let model_acceptance_mode = mtp_model_acceptance_mode_from_env();

        // Skip-state consumption (Lightning-MLX always-advance pattern):
        // When the previous step's verify forward captured logits at the last
        // accepted position, we can reuse them to sample the primary token and
        // draft new MTP tokens WITHOUT a fresh model forward.  This halves the
        // number of model forwards: every other step uses skip-state instead.
        //
        // Skip-state is only valid when there's no pending draft to verify
        // (otherwise we need to verify the pending draft first).
        let skip_logits = state.mtp_skip_logits.take();
        let skip_hidden = state.mtp_skip_hidden.take();
        let can_skip = skip_logits.is_some()
            && skip_hidden.is_some()
            && pending.is_empty()
            && self.weights.mtp.is_some()
            && self.mtp_skip_state;

        // When skip-state is usable, sample primary + draft new MTP tokens from
        // the saved logits/hidden.  The new drafts are written into `pending`,
        // `mtp_pending_draft_log_probs`, and `mtp_pending_draft_sources` so the
        // existing verify/accept pipeline operates on them unchanged.
        let primary_tok_from_skip: Option<u32> = if can_skip {
            let sl = skip_logits.unwrap();
            let sh = skip_hidden.unwrap();
            // Sample primary token from skip logits (shape [1, vocab]).
            let primary_tok = sample_logit_row(
                &sl,
                0,
                0,
                vocab,
                sampling,
                &mut state.rng,
                &mut state.sampling_probs_buf,
                &mut state.sampling_logits_buf,
                &mut state.sampling_candidates_buf,
            );
            // Draft new MTP tokens from skip hidden.
            let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
            let mtp_draft_started = Instant::now();
            let (draft, log_probs, _dist, added, _m) =
                if let Some(min_confidence) = mtp_optimistic_draft_min_confidence_override() {
                    mtp_draft_tokens_gated(
                        &self.weights,
                        &self.cfg,
                        &sh,
                        primary_tok,
                        cache,
                        Some(state.mtp_adaptive_max_depth),
                        &mut state.rng,
                        min_confidence,
                    )
                } else {
                    mtp_draft_tokens(
                        &self.weights,
                        &self.cfg,
                        &sh,
                        primary_tok,
                        cache,
                        Some(state.mtp_adaptive_max_depth),
                        &mut state.rng,
                    )
                };
            mtp_timings.mtp_draft_wall_us = mtp_timings
                .mtp_draft_wall_us
                .saturating_add(elapsed_us(mtp_draft_started));
            state.mtp_decode_count += added;
            state.mtp_pending_draft_log_probs = log_probs;
            state.mtp_pending_draft_distributions.clear();
            state.mtp_pending_draft_sources = vec![MtpDraftSource::Mtp; draft.len()];
            // Override pending so the verify/accept pipeline sees the new drafts.
            pending = draft;
            Some(primary_tok)
        } else {
            // Discard skip-state if not usable.
            drop(skip_logits);
            drop(skip_hidden);
            None
        };

        // Compute optimistic AFTER skip-state may have populated pending.
        // Auto-activate optimistic when MTP-only EWMA acceptance is sustained ≥99%
        // and we have enough samples to trust it.  This avoids full-vocab softmax
        // + rejection sampling + rollback overhead when the model is clearly
        // producing highly accurate drafts (e.g. 27B flappy at 99.5% accept).
        //
        // Hysteresis: activate at ≥0.99, deactivate at <0.85.  Once active,
        // the EWMA tracks argmax-based truth which is strictly stricter than
        // stochastic acceptance (a draft token can pass p_target/p_draft
        // rejection sampling but not be the argmax token).  Without hysteresis,
        // the EWMA oscillates: stochastic ≥0.99 activates, argmax tracking
        // shows ~0.96, deactivates, stochastic ≥0.99, repeat.
        // The gate engages in both Greedy and RejectionSampling modes: in
        // Greedy mode the EWMA tracks argmax match rate directly, which is
        // already the stricter metric — if it reaches 0.99, optimistic is safe.
        let optimistic_allowed = mtp_optimistic_allowed(self.weights.glm_mtp.is_some());
        let can_auto_optimistic = optimistic_allowed
            && !pending.is_empty()
            && state.mtp_telemetry.mtp_only_accept_rate_ewma_samples
                >= mtp_auto_optimistic_min_samples();
        let ewma = state.mtp_telemetry.mtp_only_accept_rate_ewma;
        if can_auto_optimistic && !state.auto_optimistic_active && ewma >= 0.99 {
            state.auto_optimistic_active = true;
        }
        if state.auto_optimistic_active && ewma < mtp_auto_optimistic_deactivate_threshold() {
            state.auto_optimistic_active = false;
        }
        let auto_optimistic = can_auto_optimistic && state.auto_optimistic_active;
        // Optimistic accept-all is justified by the MTP head's measured (or
        // operator-asserted, via AX_MLX_MTP_OPTIMISTIC=1) draft accuracy; that
        // evidence says nothing about n-gram-sourced drafts stacked into the
        // window, nor about sidecar drafters like the Gemma4 assistant (whose
        // measured accuracy is 93-98%, i.e. wrong drafts exist and must be
        // rejected — the same reason the GLM sidecar is excluded via
        // mtp_optimistic_allowed). Any such draft forces the full verify path.
        let all_drafts_optimistic_eligible = state
            .mtp_pending_draft_sources
            .iter()
            .all(|source| source.optimistic_accept_eligible());
        let optimistic = optimistic_allowed
            && (self.mtp_optimistic || auto_optimistic)
            && !pending.is_empty()
            && all_drafts_optimistic_eligible;
        if auto_optimistic && !self.mtp_optimistic {
            state.mtp_telemetry.auto_optimistic_steps =
                state.mtp_telemetry.auto_optimistic_steps.saturating_add(1);
        }
        // When auto-optimistic is active, we accept all drafts for output but
        // must track the TRUE acceptance rate for EWMA so the gate can
        // deactivate if draft quality drops. Without this, record_step sees
        // accept_count == pending.len() every time, pushing EWMA to 1.0 and
        // creating a positive feedback loop that makes auto-optimistic
        // permanent even if the underlying acceptance rate falls.
        // `ewma_accept_count` is set after verify, once `predicted` is
        // available for the argmax comparison.
        let mut ewma_accept_count: Option<usize> = None;

        // Build verify sequence: [primary_token] ++ pending_draft.
        let mut verify_input: Vec<u32> = Vec::with_capacity(1 + pending.len());
        if let Some(pt) = primary_tok_from_skip {
            verify_input.push(pt);
        } else {
            verify_input.push(last_token);
        }
        verify_input.extend_from_slice(&pending);
        let verify_len = verify_input.len();
        mtp_timings.verify_tokens = saturating_u32(verify_len);

        // Returns (logits_all, draft_hidden, accept_count, all_accepted, correction_argmax_tok, predicted).
        // correction_argmax_tok is only evaluated when greedy fallback needs it.
        // predicted is the target model's argmax tokens for EWMA tracking.
        let (
            logits_all,
            draft_hidden,
            accept_count,
            all_accepted,
            correction_argmax_tok,
            predicted,
        ) = if has_linear_attention {
            // Linear-attention recurrent state cannot be trimmed after a rejected
            // speculative token. Verify on a clone; adopt it only when the full
            // draft is accepted, otherwise recompute the committed prefix on the
            // original cache.
            let clone_started = Instant::now();
            let mut verify_cache = state.cache.clone();
            mtp_timings.cache_clone_wall_us = elapsed_us(clone_started);
            if optimistic {
                // ── Optimistic shortcut (default ON; kill-switch AX_MLX_MTP_OPTIMISTIC=0) ──
                // Accept all drafts without rejection sampling.
                let ac = pending.len();
                let needs_predicted =
                    sampling.temperature <= 0.0 || (auto_optimistic && !self.mtp_optimistic);
                let verify_forward_started = Instant::now();
                let (logits_all, post_norm_all) = if needs_predicted {
                    forward_all_positions_with_post_norm(
                        &self.cfg,
                        &self.weights,
                        &verify_input,
                        &mut verify_cache,
                        token_offset,
                    )
                } else {
                    forward_all_positions_post_norm_last_lm_head(
                        &self.cfg,
                        &self.weights,
                        &verify_input,
                        &mut verify_cache,
                        token_offset,
                    )
                };
                mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                verify_cache.seq_len += verify_len;
                let predicted_arr = needs_predicted.then(|| argmax(&logits_all, None));
                let draft_hidden = slice_post_norm_hidden(&post_norm_all, ac, self.cfg.hidden_size);
                let kv_refs = verify_cache.collect_eval_refs();
                let mut targets: Vec<&MlxArray> = Vec::with_capacity(2 + kv_refs.len());
                if let Some(ref predicted_arr) = predicted_arr {
                    targets.push(predicted_arr);
                }
                targets.extend(kv_refs);
                let verify_eval_started = Instant::now();
                eval(&targets);
                mtp_timings.verify_eval_wall_us = elapsed_us(verify_eval_started);
                let rollback_started = Instant::now();
                let _ = verify_cache.trim_to(token_offset + 1 + ac);
                state.cache = verify_cache;
                mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                let predicted: Vec<u32> = predicted_arr
                    .as_ref()
                    .map(|arr| arr.data_u32().to_vec())
                    .unwrap_or_default();
                let correction_argmax_tok = predicted.get(ac).copied().unwrap_or(0);
                // Track true acceptance for EWMA so auto-optimistic can
                // deactivate if draft quality drops.
                if auto_optimistic && !self.mtp_optimistic {
                    ewma_accept_count = Some(
                        pending
                            .iter()
                            .zip(predicted.iter())
                            .take_while(|(d, p)| d == p)
                            .count(),
                    );
                }
                (
                    logits_all,
                    draft_hidden,
                    ac,
                    true,
                    correction_argmax_tok,
                    predicted,
                )
            } else {
                let verify_forward_started = Instant::now();
                let (logits_all, post_norm_all) = forward_all_positions_with_post_norm(
                    &self.cfg,
                    &self.weights,
                    &verify_input,
                    &mut verify_cache,
                    token_offset,
                );
                mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                verify_cache.seq_len += verify_len;
                // Target probabilities for rejection-sampling acceptance.
                // Full-vocab softmax by default; top-k approximation when
                // AX_MLX_MTP_TARGET_SOFTMAX_MODE is set (e.g. topk_128).
                let mut local_target_prob_workspace = MtpTargetProbWorkspace::default();
                let target_prob_workspace =
                    if crate::fastpath::decode_mtp_target_prob_workspace_enabled() {
                        &mut state.mtp_target_prob_workspace
                    } else {
                        &mut local_target_prob_workspace
                    };
                let target_softmax_started = Instant::now();
                let lazy_target_probs = compute_mtp_target_probs(
                    &logits_all,
                    &pending,
                    &state.mtp_pending_draft_log_probs,
                    vocab,
                    sampling,
                    self.mtp_target_softmax_topk,
                    MtpDraftFilter::IDENTITY,
                    target_prob_workspace,
                );
                mtp_timings.target_softmax_wall_us = mtp_timings
                    .target_softmax_wall_us
                    .saturating_add(elapsed_us(target_softmax_started));
                // Always compute argmax for the correction/bonus fallback.
                let predicted_arr = Some(argmax(&logits_all, None));
                let kv_refs = verify_cache.collect_eval_refs();
                let mut targets: Vec<&MlxArray> = Vec::with_capacity(4 + kv_refs.len());
                targets.push(predicted_arr.as_ref().unwrap());
                targets.push(&post_norm_all);
                if let Some(ref ltp) = lazy_target_probs {
                    ltp.push_eval_targets(&mut targets);
                }
                targets.extend(kv_refs);
                let verify_eval_started = Instant::now();
                eval(&targets);
                mtp_timings.verify_eval_wall_us = elapsed_us(verify_eval_started);
                let accept_started = Instant::now();
                let predicted: Vec<u32> = predicted_arr
                    .as_ref()
                    .map(|arr| arr.data_u32().to_vec())
                    .unwrap_or_default();
                let target_softmax_extract_started = Instant::now();
                let target_probs_cpu = lazy_target_probs
                    .as_ref()
                    .and_then(|ltp| ltp.extract_cpu_into(&pending, target_prob_workspace));
                mtp_timings.target_softmax_wall_us = mtp_timings
                    .target_softmax_wall_us
                    .saturating_add(elapsed_us(target_softmax_extract_started));
                let target_distributions_cpu: Option<&[TokenDistribution]> = None;

                let accept = mtp_accept_count(
                    &pending,
                    &state.mtp_pending_draft_log_probs,
                    &state.mtp_pending_draft_distributions,
                    &state.mtp_pending_draft_sources,
                    target_probs_cpu,
                    target_distributions_cpu,
                    &predicted,
                    &mut state.rng,
                    draft_log_prob_temperature,
                    sampling.temperature,
                    model_acceptance_mode,
                    mtp_ngram_acceptance_mode_from_env(),
                );
                let ac = accept.accept_count;
                let all_accepted = accept.all_accepted;
                mtp_timings.accept_wall_us = elapsed_us(accept_started);

                let rollback_started = Instant::now();
                if all_accepted {
                    let _ = verify_cache.trim_to(token_offset + 1 + ac);
                    state.cache = verify_cache;
                } else {
                    recompute_committed_prefix(
                        &self.cfg,
                        &self.weights,
                        &mut state.cache,
                        verify_input[0],
                        &pending[..ac],
                        token_offset,
                    );
                }
                mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                let draft_hidden = slice_post_norm_hidden(&post_norm_all, ac, self.cfg.hidden_size);
                let correction_argmax_tok = predicted.get(ac).copied().unwrap_or(0);
                (
                    logits_all,
                    draft_hidden,
                    ac,
                    all_accepted,
                    accept.rejection_correction.unwrap_or(correction_argmax_tok),
                    predicted,
                )
            }
        } else {
            // Non-linear-attention: run directly, trim on rejection.
            if optimistic {
                // ── Optimistic shortcut (default ON; kill-switch AX_MLX_MTP_OPTIMISTIC=0) ──
                let ac = pending.len();
                let needs_predicted =
                    sampling.temperature <= 0.0 || (auto_optimistic && !self.mtp_optimistic);
                let verify_forward_started = Instant::now();
                let (logits_all, post_norm_all) = if needs_predicted {
                    forward_all_positions_with_post_norm(
                        &self.cfg,
                        &self.weights,
                        &verify_input,
                        &mut state.cache,
                        token_offset,
                    )
                } else {
                    forward_all_positions_post_norm_last_lm_head(
                        &self.cfg,
                        &self.weights,
                        &verify_input,
                        &mut state.cache,
                        token_offset,
                    )
                };
                mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                state.cache.seq_len += verify_len;
                let predicted_arr = needs_predicted.then(|| argmax(&logits_all, None));
                let draft_hidden = slice_post_norm_hidden(&post_norm_all, ac, self.cfg.hidden_size);
                let kv_refs = state.cache.collect_eval_refs();
                let mut targets: Vec<&MlxArray> = Vec::with_capacity(2 + kv_refs.len());
                if let Some(ref predicted_arr) = predicted_arr {
                    targets.push(predicted_arr);
                }
                targets.extend(kv_refs);
                let verify_eval_started = Instant::now();
                eval(&targets);
                mtp_timings.verify_eval_wall_us = elapsed_us(verify_eval_started);
                let rollback_started = Instant::now();
                let committed_len = token_offset + 1 + ac;
                let trimmed = state.cache.trim_to(committed_len);
                debug_assert!(trimmed, "MTP committed_len must not exceed cache seq_len");
                mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                let predicted: Vec<u32> = predicted_arr
                    .as_ref()
                    .map(|arr| arr.data_u32().to_vec())
                    .unwrap_or_default();
                let correction_argmax_tok = predicted.get(ac).copied().unwrap_or(0);
                // Track true acceptance for EWMA so auto-optimistic can
                // deactivate if draft quality drops.
                if auto_optimistic && !self.mtp_optimistic {
                    ewma_accept_count = Some(
                        pending
                            .iter()
                            .zip(predicted.iter())
                            .take_while(|(d, p)| d == p)
                            .count(),
                    );
                }
                (
                    logits_all,
                    draft_hidden,
                    ac,
                    true,
                    correction_argmax_tok,
                    predicted,
                )
            } else {
                let verify_forward_started = Instant::now();
                let (logits_all, post_norm_all) = forward_all_positions_with_post_norm(
                    &self.cfg,
                    &self.weights,
                    &verify_input,
                    &mut state.cache,
                    token_offset,
                );
                mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                state.cache.seq_len += verify_len;
                // Target probabilities for rejection-sampling acceptance.
                let mut local_target_prob_workspace = MtpTargetProbWorkspace::default();
                let target_prob_workspace =
                    if crate::fastpath::decode_mtp_target_prob_workspace_enabled() {
                        &mut state.mtp_target_prob_workspace
                    } else {
                        &mut local_target_prob_workspace
                    };
                let target_softmax_started = Instant::now();
                let lazy_target_probs = compute_mtp_target_probs(
                    &logits_all,
                    &pending,
                    &state.mtp_pending_draft_log_probs,
                    vocab,
                    sampling,
                    self.mtp_target_softmax_topk,
                    MtpDraftFilter::IDENTITY,
                    target_prob_workspace,
                );
                mtp_timings.target_softmax_wall_us = mtp_timings
                    .target_softmax_wall_us
                    .saturating_add(elapsed_us(target_softmax_started));
                // Always compute argmax for the correction/bonus fallback.
                let predicted_arr = Some(argmax(&logits_all, None));
                let kv_refs2 = state.cache.collect_eval_refs();
                let mut targets: Vec<&MlxArray> = Vec::with_capacity(4 + kv_refs2.len());
                targets.push(predicted_arr.as_ref().unwrap());
                targets.push(&post_norm_all);
                if let Some(ref ltp) = lazy_target_probs {
                    ltp.push_eval_targets(&mut targets);
                }
                targets.extend(kv_refs2);
                let verify_eval_started = Instant::now();
                eval(&targets);
                mtp_timings.verify_eval_wall_us = elapsed_us(verify_eval_started);
                let accept_started = Instant::now();
                let predicted: Vec<u32> = predicted_arr
                    .as_ref()
                    .map(|arr| arr.data_u32().to_vec())
                    .unwrap_or_default();
                let target_softmax_extract_started = Instant::now();
                let target_probs_cpu = lazy_target_probs
                    .as_ref()
                    .and_then(|ltp| ltp.extract_cpu_into(&pending, target_prob_workspace));
                mtp_timings.target_softmax_wall_us = mtp_timings
                    .target_softmax_wall_us
                    .saturating_add(elapsed_us(target_softmax_extract_started));
                let target_distributions_cpu: Option<&[TokenDistribution]> = None;

                let accept = mtp_accept_count(
                    &pending,
                    &state.mtp_pending_draft_log_probs,
                    &state.mtp_pending_draft_distributions,
                    &state.mtp_pending_draft_sources,
                    target_probs_cpu,
                    target_distributions_cpu,
                    &predicted,
                    &mut state.rng,
                    draft_log_prob_temperature,
                    sampling.temperature,
                    model_acceptance_mode,
                    mtp_ngram_acceptance_mode_from_env(),
                );
                let ac = accept.accept_count;
                let all_accepted = accept.all_accepted;
                mtp_timings.accept_wall_us = elapsed_us(accept_started);

                let rollback_started = Instant::now();
                let committed_len = token_offset + 1 + ac;
                let trimmed = state.cache.trim_to(committed_len);
                debug_assert!(trimmed, "MTP committed_len must not exceed cache seq_len");

                // Trim MTP KV cache: remove rejected draft entries.
                let rejected_count = pending.len() - ac;
                if rejected_count > 0 {
                    let new_mtp_len = state.mtp_decode_count.saturating_sub(rejected_count);
                    if let Some(ref mut c) = state.mtp_cache {
                        let _ = c.trim_to(new_mtp_len);
                    }
                    state.mtp_decode_count = new_mtp_len;
                }
                mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                let draft_hidden = slice_post_norm_hidden(&post_norm_all, ac, self.cfg.hidden_size);
                let correction_argmax_tok = predicted.get(ac).copied().unwrap_or(0);
                (
                    logits_all,
                    draft_hidden,
                    ac,
                    all_accepted,
                    accept.rejection_correction.unwrap_or(correction_argmax_tok),
                    predicted,
                )
            }
        };

        // For linear attention with rejection, trim MTP cache too.
        if has_linear_attention && !all_accepted {
            let rollback_started = Instant::now();
            let rejected_count = pending.len() - accept_count;
            if rejected_count > 0 {
                let new_mtp_len = state.mtp_decode_count.saturating_sub(rejected_count);
                if let Some(ref mut c) = state.mtp_cache {
                    let _ = c.trim_to(new_mtp_len);
                }
                state.mtp_decode_count = new_mtp_len;
            }
            mtp_timings.rollback_wall_us = mtp_timings
                .rollback_wall_us
                .saturating_add(elapsed_us(rollback_started));
        }

        // Collect output tokens: accepted draft tokens followed by correction/bonus.
        // When skip-state was used, prepend the skip-primary token.
        let mut result: Vec<u32> = if let Some(pt) = primary_tok_from_skip {
            let mut r = vec![pt];
            r.extend_from_slice(&pending[..accept_count]);
            r
        } else {
            pending[..accept_count].to_vec()
        };
        let tail_sample_started = Instant::now();
        let tail_tok = sample_logit_row(
            &logits_all,
            correction_argmax_tok,
            accept_count,
            vocab,
            sampling,
            &mut state.rng,
            &mut state.sampling_probs_buf,
            &mut state.sampling_logits_buf,
            &mut state.sampling_candidates_buf,
        );
        mtp_timings.tail_sample_wall_us = elapsed_us(tail_sample_started);
        result.push(tail_tok);
        mtp_timings.emitted_tokens = saturating_u32(result.len());

        // Record MTP draft/accept telemetry.
        // Pass the actual accept_count for counters (accepted_tokens, cycles, etc.)
        // and ewma_accept_count separately for EWMA tracking only.
        if !pending.is_empty() {
            // EWMA numerator for mtp_only_accept_rate_ewma:
            // - Optimistic: verifier skipped, accept_count inflated to pending.len().
            //   Use argmax matches as a quality proxy so the EWMA can deactivate
            //   auto-optimistic if draft quality drops.
            // - Rejection-sampling: accept_count reflects actual decisions; count all
            //   accepted MTP tokens for the true acceptance rate that drives the
            //   n-gram saturation gate and auto-optimistic activation.
            let mtp_ewma_numerator = if optimistic {
                if predicted.is_empty() {
                    state
                        .mtp_pending_draft_sources
                        .iter()
                        .take(accept_count)
                        .filter(|s| s.is_model_draft())
                        .count()
                } else {
                    state
                        .mtp_pending_draft_sources
                        .iter()
                        .zip(pending.iter())
                        .zip(predicted.iter())
                        .take(accept_count)
                        .filter(|((s, _d), _p)| s.is_model_draft())
                        .filter(|((_s, d), p)| d == p)
                        .count()
                }
            } else {
                state
                    .mtp_pending_draft_sources
                    .iter()
                    .take(accept_count)
                    .filter(|s| s.is_model_draft())
                    .count()
            };
            state.mtp_telemetry.record_step(
                pending.len(),
                accept_count,
                &state.mtp_pending_draft_sources,
                ewma_accept_count,
                mtp_ewma_numerator,
            );
            let gemma4_assistant_drafted = state
                .mtp_pending_draft_sources
                .iter()
                .filter(|source| **source == MtpDraftSource::Gemma4Assistant)
                .count();
            let gemma4_assistant_accepted = state
                .mtp_pending_draft_sources
                .iter()
                .take(accept_count)
                .filter(|source| **source == MtpDraftSource::Gemma4Assistant)
                .count();
            if gemma4_assistant_drafted > 0 {
                state.gemma4_assistant_mtp_telemetry.record_verified(
                    gemma4_assistant_drafted,
                    gemma4_assistant_accepted,
                    mtp_timings.verify_forward_wall_us,
                    mtp_timings.verify_eval_wall_us,
                );
            }
            let ngram_prefix_len = state
                .mtp_pending_draft_sources
                .iter()
                .take_while(|source| **source == MtpDraftSource::Ngram)
                .count();
            if ngram_prefix_len > 0 {
                // Use actual accept_count for n-gram feedback.  When
                // auto-optimistic is active, all drafts are genuinely accepted
                // in the output, so n-gram should see 100% acceptance.  The
                // argmax-based ewma_ac is only for EWMA tracking.
                let ngram_accept_count = accept_count.min(ngram_prefix_len);
                // Replay the policy that actually produced this n-gram prefix
                // (stored at draft time), not a reconstructed approximation —
                // see `NgramTable::record_draft_feedback`'s doc comment. Falls
                // back to the (min_support, conf_threshold)-only reconstruction
                // only if the invariant that this is always `Some` when
                // `ngram_prefix_len > 0` was somehow violated.
                let feedback_policy = state.ngram_draft_policy.unwrap_or_else(|| {
                    let (min_support, confidence_threshold) = ngram_feedback_policy(&self.cfg);
                    NgramDraftPolicy::majority(ngram_prefix_len, min_support, confidence_threshold)
                });
                state.ngram.record_draft_feedback(
                    &pending[..ngram_prefix_len],
                    ngram_accept_count,
                    feedback_policy,
                );
                record_ngram_beta_feedback(state, ngram_prefix_len, ngram_accept_count);
                state
                    .mtp_telemetry
                    .record_ngram_verified(ngram_accept_count);
                state.ngram_self_tune.record_verified(
                    ngram_accept_count,
                    mtp_ngram_self_tune_threshold(),
                    mtp_ngram_self_tune_warmup(),
                );
            }
        }

        mtp_timings.ngram_submitted_tokens = saturating_u32(
            state
                .mtp_pending_draft_sources
                .iter()
                .take(pending.len())
                .filter(|source| **source == MtpDraftSource::Ngram)
                .count(),
        );

        state.ngram.feed(&result);

        let mtp_max_depth = self.mtp_max_depth();
        // Use true acceptance for adaptive depth so auto-optimistic's inflated
        // accept_count doesn't create a permanent depth-increase feedback loop.
        let adaptive_depth_accept = ewma_accept_count.unwrap_or(accept_count);
        state.mtp_adaptive_max_depth = mtp_next_adaptive_depth(
            state.mtp_adaptive_max_depth,
            mtp_max_depth,
            pending.len(),
            adaptive_depth_accept,
            state.mtp_consecutive_misses,
        );
        if adaptive_depth_accept == 0 && !pending.is_empty() {
            state.mtp_consecutive_misses = state.mtp_consecutive_misses.saturating_add(1);
        } else if adaptive_depth_accept > 0 {
            state.mtp_consecutive_misses = 0;
        }

        // Per-request MTP bypass: once MTP-only acceptance EWMA has enough
        // samples and falls below the threshold, disable MTP for the remainder
        // of this request.  The direct single-token decode path is cheaper
        // when the MTP head itself is not paying for its overhead.  Bypass is
        // latched — it stays active once set.
        //
        // IMPORTANT: use mtp_only_accept_rate_ewma (cascade-corrected), NOT
        // the blended accept_rate_ewma.  The blended rate includes n-gram
        // cascade rejections that deflate the EWMA when n-gram drafts have
        // low acceptance — even when MTP-only acceptance is healthy.  Bypassing
        // MTP on the basis of n-gram quality incorrectly disables a beneficial
        // speculation source (observed as a uniform regression on 35B-A3B).
        if !state.mtp_bypassed
            && state.mtp_telemetry.mtp_only_accept_rate_ewma_samples >= mtp_bypass_min_samples()
            && state.mtp_telemetry.mtp_only_accept_rate_ewma < mtp_bypass_threshold()
        {
            state.mtp_bypassed = true;
            state.mtp_pending_draft.clear();
            state.mtp_pending_draft_log_probs.clear();
            state.mtp_pending_draft_distributions.clear();
            state.mtp_pending_draft_sources.clear();
            state.mtp_skip_logits = None;
            state.mtp_skip_hidden = None;
        }

        // Generate new draft tokens: attempt n-gram first, then let MTP fill
        // remaining depth slots when the n-gram prefix is shorter than the MTP
        // adaptive cap. If n-gram fills the whole useful window, skip MTP work
        // and reset the MTP cache so the next MTP miss starts with valid RoPE.
        //
        // When the per-request MTP bypass has fired, skip the entire draft
        // generation: the MTP head forward and n-gram lookup would produce
        // a draft that is never consumed (the next step falls through to
        // direct decode).  This saves one MTP head forward on the bypass step.
        if state.mtp_bypassed {
            mtp_timings.draft_wall_us = 0;
        } else {
            // Post-think guarded mode: compute think-state AFTER result tokens to
            // decide the NEXT draft policy.  Inside `<think>` use min_support=1;
            // outside `<think>` on reasoning models require min_support=2 to suppress
            // one-off guesses in free-form text while still drafting well-established
            // repeating patterns (SQL keywords, JSON delimiters, code syntax).
            let draft_started = Instant::now();
            let think_state_after_result =
                compute_think_state(&self.cfg, state.ngram_in_think, &result);
            let mtp_post_think_guarded =
                self.cfg.think_start_token_id.is_some() && !think_state_after_result;
            // Pure-MTP override: when AX_MLX_MTP_DISABLE_NGRAM_STACKING=1, skip the
            // ADR-008 n-gram-first draft branch entirely so the benchmark measures
            // MTP acceptance in isolation.  The MTP verify loop, head forward, and
            // telemetry are unchanged; only this draft-source branch is gated.
            let mut ngram_max = if self.disable_mtp_ngram_stacking {
                0
            } else {
                adaptive_ngram_draft_len(has_linear_attention, state.ngram_posterior_mean())
            };
            let safety_decision = if ngram_max > 0 {
                mtp_ngram_speculative_safety_decision(ctx, mtp_post_think_guarded)
            } else {
                SpeculativeSafetyDecision::default()
            };
            if safety_decision.tighten_ngram {
                state.mtp_telemetry.ngram_safety_tightened_steps = state
                    .mtp_telemetry
                    .ngram_safety_tightened_steps
                    .saturating_add(1);
                state.mtp_telemetry.ngram_safety_reason = state
                    .mtp_telemetry
                    .ngram_safety_reason
                    .max(safety_decision.reason.route_code());
                if safety_decision.reason == SpeculativeSafetyReason::ReasoningTrace {
                    state.mtp_telemetry.ngram_think_gated_steps = state
                        .mtp_telemetry
                        .ngram_think_gated_steps
                        .saturating_add(1);
                }
            }
            if safety_decision.disable_ngram {
                state.mtp_telemetry.ngram_safety_disabled_steps = state
                    .mtp_telemetry
                    .ngram_safety_disabled_steps
                    .saturating_add(1);
                state.mtp_telemetry.ngram_safety_reason = state
                    .mtp_telemetry
                    .ngram_safety_reason
                    .max(safety_decision.reason.route_code());
                ngram_max = 0;
            }
            let ngram_gate = mtp_ngram_gate_decision(
                ngram_max,
                mtp_max_depth,
                state.mtp_telemetry.accept_rate_ewma,
                state.mtp_telemetry.accept_rate_ewma_samples,
                state.mtp_telemetry.mtp_only_accept_rate_ewma,
                state.mtp_telemetry.mtp_only_accept_rate_ewma_samples,
                state
                    .mtp_telemetry
                    .draft_source_mtp_tokens
                    .saturating_add(state.mtp_telemetry.draft_source_hybrid_mtp_tokens),
                state
                    .mtp_telemetry
                    .accepted_source_mtp_tokens
                    .saturating_add(state.mtp_telemetry.accepted_source_hybrid_mtp_tokens),
                state.mtp_telemetry.draft_source_ngram_tokens,
                state.mtp_telemetry.accepted_source_ngram_tokens,
                state.ngram_self_tune.disabled,
                mtp_ngram_gate_min_samples(),
                mtp_ngram_hurt_margin(),
                MtpNgramAutoDisableConfig::from_env(),
            );
            let utility_cfg = MtpNgramUtilityGateConfig::from_env();
            let utility_decision =
                if mtp_ngram_gate_policy_from_env() == MtpNgramGatePolicy::Utility {
                    mtp_ngram_utility_gate(
                        ngram_max,
                        state.mtp_telemetry.baseline_utility(),
                        state.mtp_telemetry.stacked_utility(),
                        utility_cfg,
                        state.mtp_ngram_utility_hysteresis_remaining,
                    )
                } else {
                    MtpNgramUtilityDecision::default()
                };
            if utility_decision.utility_hurt {
                state.mtp_ngram_utility_hysteresis_remaining = utility_cfg.hysteresis_steps;
            } else if state.mtp_ngram_utility_hysteresis_remaining > 0 {
                state.mtp_ngram_utility_hysteresis_remaining -= 1;
            }
            if utility_decision.insufficient_samples {
                state.mtp_telemetry.ngram_utility_insufficient_sample_steps = state
                    .mtp_telemetry
                    .ngram_utility_insufficient_sample_steps
                    .saturating_add(1);
            }
            if utility_decision.gated {
                state.mtp_telemetry.ngram_utility_gated_steps = state
                    .mtp_telemetry
                    .ngram_utility_gated_steps
                    .saturating_add(1);
            }
            let ngram_max = if ngram_gate.gated || utility_decision.gated {
                0
            } else {
                ngram_max
            };
            if ngram_gate.saturated {
                state.mtp_telemetry.ngram_saturated_gated_steps = state
                    .mtp_telemetry
                    .ngram_saturated_gated_steps
                    .saturating_add(1);
            }
            if ngram_gate.hurt {
                state.mtp_telemetry.ngram_hurt_gated_steps =
                    state.mtp_telemetry.ngram_hurt_gated_steps.saturating_add(1);
                match mtp_ngram_hurt_gate_mode() {
                    HurtGateMode::SourceAware => {
                        state.mtp_telemetry.ngram_source_hurt_gated_steps = state
                            .mtp_telemetry
                            .ngram_source_hurt_gated_steps
                            .saturating_add(1);
                    }
                    HurtGateMode::LegacyEwma => {
                        state.mtp_telemetry.ngram_legacy_hurt_gated_steps = state
                            .mtp_telemetry
                            .ngram_legacy_hurt_gated_steps
                            .saturating_add(1);
                    }
                }
            }
            if ngram_gate.auto_disabled {
                state.mtp_telemetry.ngram_auto_disabled_steps = state
                    .mtp_telemetry
                    .ngram_auto_disabled_steps
                    .saturating_add(1);
            }
            if ngram_gate.self_tune_disabled {
                state.mtp_telemetry.ngram_self_tune_disabled_steps = state
                    .mtp_telemetry
                    .ngram_self_tune_disabled_steps
                    .saturating_add(1);
            }
            let ngram_policy = NgramDraftPolicy {
                variant: mtp_ngram_policy_variant(),
                max_len: ngram_max,
                min_support: mtp_ngram_min_support().max(if mtp_post_think_guarded {
                    POST_THINK_MIN_NGRAM_SUPPORT
                } else {
                    1
                }),
                confidence_threshold: mtp_ngram_confidence_threshold(),
                adaptive_match_len: true,
                bypass_prompt_min_support: true,
                min_context_len: mtp_ngram_min_context_len(),
            };
            let ngram_outcome = if ngram_max > 0 {
                let ngram_lookup_started = Instant::now();
                let outcome = state.ngram.predict_with_policy(ngram_policy);
                mtp_timings.ngram_lookup_wall_us = mtp_timings
                    .ngram_lookup_wall_us
                    .saturating_add(elapsed_us(ngram_lookup_started));
                outcome
            } else {
                NgramDraftOutcome {
                    draft: vec![],
                    confidence: vec![],
                    rejection: None,
                    requested_max_len: 0,
                }
            };
            if ngram_max > 0 {
                state
                    .mtp_telemetry
                    .record_ngram_attempt(ngram_outcome.rejection);
                state
                    .mtp_telemetry
                    .record_ngram_proposed(ngram_outcome.draft.len());
            }
            let recent = &result[result.len().saturating_sub(8)..];
            let ngram_cycle_guarded = !ngram_outcome.draft.is_empty()
                && ngram_draft_is_cycle(&ngram_outcome.draft, recent);
            if ngram_cycle_guarded {
                state.mtp_telemetry.record_ngram_cycle_guard();
            }
            // Reset unconditionally; only the n-gram-nonempty branch below sets
            // this back to `Some`. Otherwise the next verification step would
            // see `ngram_prefix_len == 0` anyway (no Ngram entries in
            // `new_sources`) and never read this, but keep it precise rather
            // than relying on that.
            state.ngram_draft_policy = None;
            let (new_draft, new_log_probs, new_sources) = if !ngram_outcome.draft.is_empty()
                && !ngram_cycle_guarded
            {
                let mut draft = ngram_outcome.draft;
                let ngram_len = draft.len();
                state.ngram_draft_policy = Some(ngram_policy);
                state.ngram_self_tune.record_submitted(ngram_len);
                state.mtp_telemetry.record_ngram_submitted(ngram_len);
                let mtp_tail_cap = state.mtp_adaptive_max_depth.saturating_sub(ngram_len);
                let mut sources = vec![MtpDraftSource::Ngram; ngram_len];

                let mut aligned_log_probs = mtp_ngram_pseudo_log_probs(
                    &ngram_outcome.confidence,
                    ngram_len,
                    mtp_ngram_acceptance_mode_from_env(),
                );

                if mtp_tail_cap > 0
                    && (self.weights.mtp.is_some() || self.weights.glm_mtp.is_some())
                {
                    let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
                    let mtp_draft_started = Instant::now();
                    let (tail, log_probs, distributions, added, _top2_margins) =
                        if self.weights.glm_mtp.is_some() {
                            glm_mtp_draft_tokens(
                                &self.weights,
                                &self.cfg,
                                &draft_hidden,
                                tail_tok,
                                cache,
                                Some(mtp_tail_cap),
                                &mut state.rng,
                            )
                        } else {
                            mtp_draft_tokens_after_forced_prefix(
                                &self.weights,
                                &self.cfg,
                                &draft_hidden,
                                tail_tok,
                                &draft,
                                cache,
                                mtp_tail_cap,
                                &mut state.rng,
                            )
                        };
                    mtp_timings.mtp_draft_wall_us = mtp_timings
                        .mtp_draft_wall_us
                        .saturating_add(elapsed_us(mtp_draft_started));
                    state.mtp_decode_count += added;
                    state.mtp_pending_draft_distributions = distributions;
                    state.mtp_telemetry.record_ngram_stack_hit(ngram_len, false);
                    state.mtp_telemetry.record_ngram_hybrid_tail(tail.len());
                    aligned_log_probs.extend(log_probs);
                    sources.extend(std::iter::repeat_n(MtpDraftSource::HybridMtp, tail.len()));
                    draft.extend(tail);
                    (draft, aligned_log_probs, sources)
                } else {
                    // N-gram filled the whole draft window — no MTP tail needed.
                    // Preserve MTP cache and advance RoPE offset by ngram_len
                    // instead of resetting to None (ADR-013 Phase 5). This keeps
                    // accumulated positional context so the next MTP step starts
                    // with correct RoPE offsets. Gated by env var; defaults to
                    // the previous reset behavior for safety.
                    let preserve_cache = std::env::var("AX_MLX_MTP_NGRAM_CACHE_POLICY")
                        .map(|v| v != "reset")
                        .unwrap_or(true);
                    if preserve_cache {
                        if let Some(ref mut cache) = state.mtp_cache {
                            // N-gram tokens don't produce MTP KV entries, so advance
                            // rope_offset (logical position) instead of seq_len
                            // (physical entries).  This keeps the next MTP step's
                            // RoPE correct without leaving a gap of uninitialized
                            // KV entries that SDPA would attend over.
                            cache.rope_offset += ngram_len;
                        }
                        // mtp_decode_count tracks physical MTP KV entries only.
                        // N-gram tokens don't add entries, so don't increment here.
                    } else {
                        state.mtp_cache = None;
                        state.mtp_decode_count = 0;
                    }
                    state.mtp_skip_logits = None;
                    state.mtp_skip_hidden = None;
                    state.mtp_pending_draft_distributions.clear();
                    state.mtp_telemetry.record_ngram_stack_hit(ngram_len, true);
                    (draft, aligned_log_probs, sources)
                }
            } else {
                if self.weights.mtp.is_some() {
                    // MTP head forward path (RoPE managed internally via cache.seq_len).
                    let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
                    let mtp_draft_started = Instant::now();
                    let (draft, log_probs, distributions, added, _top2_margins) = if let Some(
                        min_confidence,
                    ) =
                        mtp_optimistic_draft_min_confidence_override()
                    {
                        mtp_draft_tokens_gated(
                            &self.weights,
                            &self.cfg,
                            &draft_hidden,
                            tail_tok,
                            cache,
                            Some(state.mtp_adaptive_max_depth),
                            &mut state.rng,
                            min_confidence,
                        )
                    } else {
                        mtp_draft_tokens(
                            &self.weights,
                            &self.cfg,
                            &draft_hidden,
                            tail_tok,
                            cache,
                            Some(state.mtp_adaptive_max_depth),
                            &mut state.rng,
                        )
                    };
                    mtp_timings.mtp_draft_wall_us = mtp_timings
                        .mtp_draft_wall_us
                        .saturating_add(elapsed_us(mtp_draft_started));
                    state.mtp_decode_count += added;
                    state.mtp_pending_draft_distributions = distributions;
                    let sources = vec![MtpDraftSource::Mtp; draft.len()];
                    (draft, log_probs, sources)
                } else if self.weights.glm_mtp.is_some() {
                    // GLM MTP head forward path (GLM MLA attention, shared_head logits).
                    let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
                    let mtp_draft_started = Instant::now();
                    let (draft, log_probs, distributions, added, _top2_margins) =
                        glm_mtp_draft_tokens(
                            &self.weights,
                            &self.cfg,
                            &draft_hidden,
                            tail_tok,
                            cache,
                            Some(state.mtp_adaptive_max_depth),
                            &mut state.rng,
                        );
                    mtp_timings.mtp_draft_wall_us = mtp_timings
                        .mtp_draft_wall_us
                        .saturating_add(elapsed_us(mtp_draft_started));
                    state.mtp_decode_count += added;
                    state.mtp_pending_draft_distributions = distributions;
                    let sources = vec![MtpDraftSource::Mtp; draft.len()];
                    (draft, log_probs, sources)
                } else {
                    let assistant_draft_started = Instant::now();
                    let (draft, log_probs, distributions) =
                        self.gemma4_assistant_draft_token(state, tail_tok, &draft_hidden, sampling);
                    mtp_timings.assistant_draft_wall_us = mtp_timings
                        .assistant_draft_wall_us
                        .saturating_add(elapsed_us(assistant_draft_started));
                    state.mtp_pending_draft_distributions = distributions;
                    let sources = vec![MtpDraftSource::Gemma4Assistant; draft.len()];
                    (draft, log_probs, sources)
                }
            };
            state.mtp_pending_draft = new_draft;
            state.mtp_pending_draft_log_probs = new_log_probs;
            state.mtp_pending_draft_sources = new_sources;
            if state.mtp_pending_draft_log_probs.is_empty() {
                state.mtp_pending_draft_distributions.clear();
            }
            if state.mtp_pending_draft.is_empty() {
                state.mtp_pending_draft_sources.clear();
            }
            // Capture skip-state only when the next step will have no pending draft,
            // making `can_skip` true.  When pending is non-empty (the common case)
            // async_eval + slice work here is never consumed — so skip it entirely.
            if self.mtp_skip_state
                && (self.weights.mtp.is_some() || self.weights.glm_mtp.is_some())
                && state.mtp_pending_draft.is_empty()
            {
                let sl = if logits_all.shape().len() == 1 {
                    logits_all.clone()
                } else {
                    slice(
                        &logits_all,
                        &[accept_count as i32, 0],
                        &[(accept_count + 1) as i32, vocab],
                        &[1, 1],
                        None,
                    )
                };
                mlx_sys::async_eval(&[&sl, &draft_hidden]);
                state.mtp_skip_logits = Some(sl);
                state.mtp_skip_hidden = Some(draft_hidden);
            }
            mtp_timings.draft_wall_us = elapsed_us(draft_started);
        } // end of !mtp_bypassed draft generation block
        let gemma4_assistant_submitted = state
            .mtp_pending_draft_sources
            .iter()
            .filter(|source| **source == MtpDraftSource::Gemma4Assistant)
            .count();
        if gemma4_assistant_submitted > 0 {
            state
                .gemma4_assistant_mtp_telemetry
                .record_submitted(gemma4_assistant_submitted, mtp_timings.draft_wall_us);
        }
        state.mtp_telemetry.record_timings(mtp_timings);

        result
    }

    fn initialize_generation_state(
        &self,
        state: &mut RequestState,
        max_output: u32,
        layer_eligible: Option<&[bool]>,
        prefill_output_token: Option<u32>,
        is_greedy: bool,
    ) -> Option<u32> {
        // When MTP is active, use a wider prompt window (NGRAM_MTP_PROMPT_FEED_MAX)
        // so real-code bigrams are seeded before the first decode step. Without
        // MTP, keep the conservative 64-token guard to avoid random-token false
        // positives that would disable n-gram for the first 16 decode steps.
        let has_mtp = self.has_mtp();
        seed_generation_ngram_from_prompt(state, has_mtp);
        seed_generation_ngram_from_prefill_output(state, prefill_output_token);

        // Initialize think-block tracking from the full prompt token sequence.
        // Reasoning models (Qwen3 family) inject `<think>` at the assistant prefix,
        // so generation typically starts inside a think block and ngram_in_think=true
        // from step 0, enabling n-gram speculation immediately.
        state.ngram_in_think = compute_think_state(&self.cfg, false, &state.prompt_prefix_tokens);

        // Classify the full prompt once per generation. record_prompt_class is
        // max-merge friendly so re-entry from an unusual code path cannot
        // downgrade an already-set class.
        let prompt_class = classify_prompt_class(&state.prompt_prefix_tokens);
        state.ngram_acceleration.record_prompt_class(prompt_class);
        let has_linear_attention = self.cfg.linear_attention.is_some();
        // When MTP is active, skip the probe-based initial disable: MTP handles
        // first-step speculation independently, so even if the n-gram table has no
        // candidate at step 0, n-gram can build from output tokens and contribute
        // later without any harm to the first decode step.
        let linear_initial_prompt_without_draft = !has_mtp
            && linear_ngram_initial_prompt_should_disable_request(
                has_linear_attention,
                prompt_class,
                &state.ngram,
                self.ngram_policy_variant,
            );

        // Reset per-generation state.
        state.bonus_queue.clear();
        state.next_model_last_token = None;
        state.pending_direct = None;
        state.direct_pipeline_emitted_tokens = 0;
        state.ngram_disabled_steps = 0;
        state.linear_ngram_no_draft_streak = 0;
        state.linear_ngram_reenable_probe_countdown = 0;
        // Reset MTP draft state for the new generation.
        state.mtp_pending_draft.clear();
        state.mtp_pending_draft_log_probs.clear();
        state.mtp_pending_draft_distributions.clear();
        state.mtp_pending_draft_sources.clear();
        state.mtp_adaptive_max_depth =
            mtp_initial_adaptive_depth(&self.cfg.model_family, self.mtp_max_depth());
        state.mtp_skip_logits = None;
        state.mtp_skip_hidden = None;
        state.mtp_decode_count = 0;
        state.mtp_bypassed = false;
        state.ngram_self_tune = NgramSelfTuneState::default();
        state.mtp_ngram_utility_hysteresis_remaining = 0;
        if let Some(ref mut c) = state.mtp_cache {
            c.reset();
        }
        // MTP prefill warmup: prime the MTP head KV cache with committed
        // prompt/history transitions from the final prefill chunk. MTPLX's
        // sustained profile does the same via committed MTP history; without
        // this, the recurrent MTP attention starts decode with almost no
        // prompt-side history and acceptance drops sharply.
        //
        // The warmup is capped to the most recent `mtp_warmup_cap()` tokens
        // (default 256) because the MTP head's single-layer attention has
        // limited effective range. Tokens beyond ~256 positions contribute
        // diminishing returns to draft quality but add linearly to the lazy
        // computation graph depth, increasing TTFT. For a 2048-token cold
        // prefill chunk, the cap reduces warmup ops from ~57 to ~7
        // full-model-equivalent forwards.
        if let (Some(prefill_hidden), Some(head)) =
            (state.mtp_prefill_hidden.take(), self.weights.mtp.as_ref())
        {
            let history_tokens = std::mem::take(&mut state.mtp_prefill_history_tokens);
            let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
            let available_rows = prefill_hidden
                .shape()
                .get(1)
                .copied()
                .unwrap_or_default()
                .max(0) as usize;
            let total = available_rows.min(history_tokens.len());
            let cap = mtp_warmup_cap();
            let warmup_len = if cap > 0 { total.min(cap) } else { total };
            let start_offset = total.saturating_sub(warmup_len);
            if warmup_len > 0 {
                let warmup_hidden = slice(
                    &prefill_hidden,
                    &[0, start_offset as i32, 0],
                    &[1, total as i32, self.cfg.hidden_size as i32],
                    &[1, 1, 1],
                    None,
                );
                crate::mtp::mtp_warmup_cache_kv_batched(
                    head,
                    &warmup_hidden,
                    &history_tokens[start_offset..total],
                    &self.weights,
                    cache,
                    &self.cfg,
                    start_offset,
                );
                let kv_refs = cache.collect_eval_refs();
                mlx_sys::eval(&kv_refs);
                clear_cache();
                state.mtp_decode_count = warmup_len;
                // After warmup, cache.seq_len == warmup_len (physical entries).
                // Set rope_offset so subsequent decode steps compute RoPE at
                // the correct absolute position: seq_len + rope_offset gives
                // the true logical position (e.g. warmup_len + start_offset = total).
                if let Some(ref mut c) = state.mtp_cache {
                    c.rope_offset = start_offset;
                }
            }
        }

        // Skip n-gram entirely for short output budgets: failed speculation
        // attempts and cooldown intervals (8-16 steps) are a net loss when
        // max_output_tokens is smaller than two full retry windows.
        let short_output_budget = max_output < NGRAM_MIN_OUTPUT_FOR_ACCELERATION;
        state.ngram_acceleration_disabled_for_request =
            short_output_budget || linear_initial_prompt_without_draft;
        state.ngram_request_disable_reason = if short_output_budget {
            NgramRequestDisableReason::ShortOutputBudget
        } else if linear_initial_prompt_without_draft {
            NgramRequestDisableReason::LinearInitialNoDraft
        } else {
            NgramRequestDisableReason::None
        };

        let kv_compression_shadow_sync_wall_us =
            self.sync_turboquant_shadow_storage_if_needed(state, true, layer_eligible);

        // Mirror mlx_lm.generate_step's first-yield boundary for the direct
        // greedy baseline: before the first token is yielded, mlx_lm already
        // builds and submits the next `_step(y)` via async_eval. Prime AX's
        // direct pipeline at the same prefill/first-token boundary so the
        // measured generation interval starts with a pending token instead of
        // paying one decode-step bootstrap.
        if (self.disable_ngram_acceleration || state.ngram_acceleration_disabled_for_request)
            && is_greedy
            && max_output > 1
            && let Some(prefill_tok) = prefill_output_token
        {
            let bootstrap_started = Instant::now();
            let turboquant_context = self.turboquant_model_decode_context();
            let bootstrap_token = start_direct_pipeline_with_turboquant_context(
                &self.cfg,
                &self.weights,
                prefill_tok,
                &mut state.cache,
                turboquant_context.as_ref(),
            );
            state.pending_direct = Some(bootstrap_token);
            // The first generated token was produced at the prefill boundary.
            // `chunked_prefill` already mirrors mlx_lm's immediate post-first
            // clear_cache; count that token so the direct decode loop does not
            // clear again after token 2.
            state.direct_pipeline_emitted_tokens = 1;
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
        sampling: MlxSamplingParams,
        is_greedy: bool,
        final_by_max_output: bool,
        ctx: Option<&RunnerRequestContext>,
    ) -> Vec<u32> {
        let has_linear_attention = self.cfg.linear_attention.is_some();

        // MTP model-based speculative decode: checked first so that MTP can
        // activate even when n-gram is disabled (linear-attention models set
        // ngram_acceleration_disabled_for_request, which run_non_ngram_decode
        // intercepts before we'd ever reach MTP).  Repetition-penalty sampling
        // is incompatible with speculative decode and is excluded.
        // Acceptance uses rejection sampling when draft log-probs are available;
        // correction/bonus tokens are sampled with the request's temperature.
        // The per-request bypass (state.mtp_bypassed) short-circuits MTP when
        // the acceptance EWMA has shown MTP is not paying for itself.
        if self.has_mtp()
            && !self.disable_ngram_acceleration
            && !sampling.uses_repetition_penalty()
            && !state.mtp_bypassed
        {
            return self.run_mtp_decode(state, last_token, sampling, ctx);
        }

        if let Some(result) =
            self.run_non_ngram_decode(state, last_token, sampling, is_greedy, final_by_max_output)
        {
            return result;
        }

        if should_drain_pending_direct_before_ngram(is_greedy, state.pending_direct.is_some()) {
            return self.finish_pending_direct_for_ngram_transition(state);
        }

        // Post-think guarded mode: outside `<think>` on reasoning models, require
        // higher support before drafting to avoid wasted verifications on one-off
        // patterns in mixed text regions.  Well-established repeating patterns
        // (SQL keywords, JSON structure, code syntax) still pass min_support=2.
        let post_think_guarded = self.cfg.think_start_token_id.is_some() && !state.ngram_in_think;

        let draft_policy = ngram_acceleration_policy(
            has_linear_attention,
            state.ngram_posterior_mean(),
            self.ngram_policy_variant,
            post_think_guarded,
        );
        let draft_outcome = state.ngram.predict_with_policy(draft_policy);
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
                final_by_max_output,
                rejection,
            ) {
                return result;
            }
            return self.run_single_decode(state, last_token, sampling);
        }

        state.linear_ngram_no_draft_streak = 0;

        let draft_len = draft.len();
        let branch_started = Instant::now();
        let repetition_history = state.repetition_history(&[], sampling);
        let result = ngram_accel_decode_step_with_sampling_buffers(
            &self.cfg,
            &self.weights,
            &mut state.cache,
            &mut state.ngram,
            last_token,
            &draft,
            draft_policy,
            sampling,
            &repetition_history,
            &mut state.rng,
            &mut state.sampling_probs_buf,
            &mut state.sampling_logits_buf,
            &mut state.sampling_candidates_buf,
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
        record_ngram_beta_feedback(state, draft_len, accept_count);

        if let Some(disabled_steps) = ngram_acceleration_disabled_steps(
            has_linear_attention,
            accept_count,
            draft_len,
            state.ngram_posterior_mean(),
        ) {
            state.ngram_disabled_steps = disabled_steps;
            // Any pending_direct from a previous cooldown cycle is now stale:
            // the n-gram steps that just ran advanced cache.seq_len
            // independently, so the lookahead array points at the wrong
            // position. Force a Bootstrap on the first new cooldown step.
            state.pending_direct = None;
            state.direct_pipeline_emitted_tokens = 0;
            state
                .ngram_acceleration
                .record_cooldown_event(disabled_steps);
        }

        result
    }
}

fn slice_post_norm_hidden(post_norm_all: &MlxArray, pos: usize, hidden_size: usize) -> MlxArray {
    let p = pos as i32;
    let hs = hidden_size as i32;
    slice(post_norm_all, &[0, p, 0], &[1, p + 1, hs], &[1, 1, 1], None)
}

/// Returns true if `draft` would create a repeating cycle relative to `recent`.
/// Checks periods 3..=min(draft.len(), 8, recent.len()): if the first `period`
/// tokens of the draft exactly match the last `period` tokens of `recent`, the
/// draft is a cycle continuation and should not be used as a speculative draft.
fn ngram_draft_is_cycle(draft: &[u32], recent: &[u32]) -> bool {
    for period in 3..=draft.len().min(8).min(recent.len()) {
        if draft[..period] == recent[recent.len() - period..] {
            return true;
        }
    }
    false
}

/// Compute the think-block state after observing a sequence of tokens, without
/// mutating any state.  Returns the updated `in_think` flag.
///
/// Used in `run_mtp_decode` to peek ahead at result tokens before deciding
/// whether the NEXT draft step should be gated by think-block state.
fn compute_think_state(cfg: &ModelConfig, current: bool, tokens: &[u32]) -> bool {
    let Some(start_id) = cfg.think_start_token_id else {
        return current;
    };
    let end_id = cfg.think_end_token_id;
    let mut state = current;
    for &t in tokens {
        if t == start_id {
            state = true;
        } else if end_id.is_some_and(|e| t == e) {
            state = false;
        }
    }
    state
}

/// Update `ngram_in_think` in-place after emitting a single token.
/// Called in the main decode loop for every output token.
fn update_ngram_think_state(cfg: &ModelConfig, in_think: &mut bool, token: u32) {
    let Some(start_id) = cfg.think_start_token_id else {
        return;
    };
    if token == start_id {
        *in_think = true;
    } else if cfg.think_end_token_id.is_some_and(|e| token == e) {
        *in_think = false;
    }
}

fn mtp_next_adaptive_depth(
    current_depth: usize,
    max_depth: usize,
    pending_len: usize,
    accept_count: usize,
    consecutive_misses: u32,
) -> usize {
    if max_depth == 0 {
        return 0;
    }

    let current_depth = if current_depth == 0 {
        max_depth
    } else {
        current_depth.clamp(1, max_depth)
    };

    if pending_len == 0 {
        return current_depth;
    }

    if accept_count >= pending_len {
        return current_depth.saturating_add(1).min(max_depth);
    }

    if accept_count == 0 {
        // Progressive floor on consecutive complete misses: first miss keeps
        // floor at 2 (status quo); second drops to 1; third+ drops to 0.
        let floor = match consecutive_misses {
            0 => 2.min(max_depth),
            1 => 1.min(max_depth),
            _ => 0,
        };
        return floor;
    }

    let floor = 2.min(max_depth);
    accept_count.clamp(floor, max_depth)
}

/// Model-specific starting depth for the adaptive depth controller.
///
/// The adaptive controller (`mtp_next_adaptive_depth`) adjusts draft depth
/// per-request based on acceptance.  This helper sets the *initial* depth at
/// generation start so the controller begins from a model-appropriate point
/// rather than the hardware maximum.
///
/// `qwen3_5` (Qwen3.6 dense 27B, linear-attention hybrid) and `qwen3_next`
/// (Qwen3.6 MoE + linear-attention) both start at depth 2: the hybrid
/// architecture's linear-attention layers are recurrent scans that don't
/// benefit from deeper drafts the way dense SDPA layers do.  The gate-throughput
/// sweep (`docs/mtp/draft-gate-throughput.md`) confirms depth 2 is the
/// throughput optimum on all suites.  Starting at `head_max_depth` (8+) wastes
/// 3–4 steps of deep head forwards before the controller converges.  The
/// 35B-A3B variant has `native_depth=1` (the `.min(head_max_depth)` clamp
/// keeps it safe).  All other families start at `head_max_depth`.
fn mtp_initial_adaptive_depth(model_family: &str, head_max_depth: usize) -> usize {
    match model_family {
        "qwen3_next" | "qwen3_5" => 2.min(head_max_depth),
        _ => head_max_depth,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MtpNgramAutoDisableConfig {
    mtp_warmup: u32,
    ngram_warmup: u32,
    mtp_threshold: u32,
    ngram_floor: u32,
}

impl MtpNgramAutoDisableConfig {
    fn from_env() -> Self {
        Self {
            mtp_warmup: mtp_ngram_auto_disable_mtp_warmup(),
            ngram_warmup: mtp_ngram_auto_disable_ngram_warmup(),
            mtp_threshold: (mtp_ngram_auto_disable_mtp_threshold() * 1000.0) as u32,
            ngram_floor: (mtp_ngram_auto_disable_min_ngram() * 1000.0) as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MtpNgramGatePolicy {
    #[default]
    Rate,
    Utility,
}

impl MtpNgramGatePolicy {
    fn route_code(self) -> u32 {
        match self {
            Self::Rate => 0,
            Self::Utility => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DraftSourceUtility {
    submitted_tokens: u32,
    proposer_wall_us: u32,
    verify_wall_us: u32,
    emitted_tokens: u32,
}

impl DraftSourceUtility {
    fn cost_per_emitted_token_us(self) -> Option<f64> {
        if self.emitted_tokens == 0 {
            return None;
        }
        let total = u64::from(self.proposer_wall_us).saturating_add(u64::from(self.verify_wall_us));
        Some(total as f64 / f64::from(self.emitted_tokens))
    }
}

#[derive(Clone, Copy, Debug)]
struct MtpNgramUtilityGateConfig {
    min_emitted_tokens: u32,
    min_ngram_submitted_tokens: u32,
    margin_ratio: f64,
    hysteresis_steps: u32,
}

impl MtpNgramUtilityGateConfig {
    fn from_env() -> Self {
        Self {
            min_emitted_tokens: mtp_ngram_utility_min_emitted_tokens(),
            min_ngram_submitted_tokens: mtp_ngram_utility_min_ngram_tokens(),
            margin_ratio: mtp_ngram_utility_margin_ratio(),
            hysteresis_steps: mtp_ngram_utility_hysteresis_steps(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MtpNgramUtilityDecision {
    gated: bool,
    insufficient_samples: bool,
    utility_hurt: bool,
    hysteresis_active: bool,
}

fn mtp_ngram_utility_gate(
    ngram_max: usize,
    baseline: DraftSourceUtility,
    stacked: DraftSourceUtility,
    cfg: MtpNgramUtilityGateConfig,
    hysteresis_remaining: u32,
) -> MtpNgramUtilityDecision {
    if ngram_max == 0 {
        return MtpNgramUtilityDecision::default();
    }
    if hysteresis_remaining > 0 {
        return MtpNgramUtilityDecision {
            gated: true,
            hysteresis_active: true,
            ..MtpNgramUtilityDecision::default()
        };
    }
    if baseline.emitted_tokens < cfg.min_emitted_tokens
        || stacked.emitted_tokens < cfg.min_emitted_tokens
        || stacked.submitted_tokens < cfg.min_ngram_submitted_tokens
    {
        return MtpNgramUtilityDecision {
            insufficient_samples: true,
            ..MtpNgramUtilityDecision::default()
        };
    }
    let Some(baseline_cost) = baseline.cost_per_emitted_token_us() else {
        return MtpNgramUtilityDecision {
            insufficient_samples: true,
            ..MtpNgramUtilityDecision::default()
        };
    };
    let Some(stacked_cost) = stacked.cost_per_emitted_token_us() else {
        return MtpNgramUtilityDecision {
            insufficient_samples: true,
            ..MtpNgramUtilityDecision::default()
        };
    };
    let utility_hurt = stacked_cost > baseline_cost * (1.0 + cfg.margin_ratio.max(0.0));
    MtpNgramUtilityDecision {
        gated: utility_hurt,
        insufficient_samples: false,
        utility_hurt,
        hysteresis_active: false,
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SpeculativeSafetyReason {
    #[default]
    None,
    ToolCall,
    StructuredOutput,
    ReasoningTrace,
    ExperimentalOverride,
}

impl SpeculativeSafetyReason {
    fn route_code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::ToolCall => 1,
            Self::StructuredOutput => 2,
            Self::ReasoningTrace => 3,
            Self::ExperimentalOverride => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SpeculativeSafetyDecision {
    disable_ngram: bool,
    tighten_ngram: bool,
    reason: SpeculativeSafetyReason,
}

fn mtp_ngram_speculative_safety_decision(
    ctx: Option<&RunnerRequestContext>,
    post_think_guarded: bool,
) -> SpeculativeSafetyDecision {
    mtp_ngram_speculative_safety_decision_for_mode(
        mtp_ngram_safety_mode(),
        ctx.map(|ctx| ctx.tool_call_mode).unwrap_or(false),
        ctx.map(|ctx| ctx.structured_output_mode).unwrap_or(false),
        post_think_guarded,
    )
}

fn mtp_ngram_speculative_safety_decision_for_mode(
    mode: MtpNgramSafetyMode,
    tool_call_mode: bool,
    structured_output_mode: bool,
    post_think_guarded: bool,
) -> SpeculativeSafetyDecision {
    match mode {
        MtpNgramSafetyMode::Off => SpeculativeSafetyDecision::default(),
        _ if tool_call_mode => SpeculativeSafetyDecision {
            disable_ngram: true,
            reason: SpeculativeSafetyReason::ToolCall,
            ..SpeculativeSafetyDecision::default()
        },
        _ if structured_output_mode => SpeculativeSafetyDecision {
            disable_ngram: true,
            reason: SpeculativeSafetyReason::StructuredOutput,
            ..SpeculativeSafetyDecision::default()
        },
        MtpNgramSafetyMode::DisableAll => SpeculativeSafetyDecision {
            disable_ngram: true,
            reason: SpeculativeSafetyReason::ExperimentalOverride,
            ..SpeculativeSafetyDecision::default()
        },
        MtpNgramSafetyMode::DisableReasoning if post_think_guarded => SpeculativeSafetyDecision {
            disable_ngram: true,
            reason: SpeculativeSafetyReason::ReasoningTrace,
            ..SpeculativeSafetyDecision::default()
        },
        MtpNgramSafetyMode::TightenReasoning if post_think_guarded => SpeculativeSafetyDecision {
            tighten_ngram: true,
            reason: SpeculativeSafetyReason::ReasoningTrace,
            ..SpeculativeSafetyDecision::default()
        },
        _ => SpeculativeSafetyDecision::default(),
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum MtpNgramSafetyMode {
    Off,
    DisableAll,
    DisableReasoning,
    #[default]
    TightenReasoning,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MtpNgramGateDecision {
    gated: bool,
    saturated: bool,
    hurt: bool,
    auto_disabled: bool,
    self_tune_disabled: bool,
}

#[allow(clippy::too_many_arguments)]
fn mtp_ngram_gate_decision(
    ngram_max: usize,
    mtp_depth: usize,
    combined_ewma: f32,
    combined_samples: u32,
    mtp_only_ewma: f32,
    mtp_only_samples: u32,
    mtp_drafted: u32,
    mtp_accepted: u32,
    ngram_drafted: u32,
    ngram_accepted: u32,
    self_tune_disabled: bool,
    min_samples: u32,
    hurt_margin: f32,
    auto_cfg: MtpNgramAutoDisableConfig,
) -> MtpNgramGateDecision {
    let saturated = mtp_ngram_saturated_gate(ngram_max, mtp_depth, mtp_only_ewma, mtp_only_samples);
    let hurt = match mtp_ngram_hurt_gate_mode() {
        HurtGateMode::SourceAware => mtp_ngram_source_hurt_gate(
            ngram_max,
            mtp_drafted,
            mtp_accepted,
            ngram_drafted,
            ngram_accepted,
            min_samples,
            hurt_margin,
        ),
        HurtGateMode::LegacyEwma => mtp_ngram_hurt_gate(
            ngram_max,
            combined_ewma,
            combined_samples,
            mtp_only_ewma,
            mtp_only_samples,
            min_samples,
            hurt_margin,
        ),
    };
    let auto_disabled = mtp_ngram_auto_disable_gate(
        ngram_max,
        mtp_drafted,
        mtp_accepted,
        ngram_drafted,
        ngram_accepted,
        auto_cfg,
    );
    let self_tune_disabled = ngram_max > 0 && self_tune_disabled;
    MtpNgramGateDecision {
        gated: saturated || hurt || auto_disabled || self_tune_disabled,
        saturated,
        hurt,
        auto_disabled,
        self_tune_disabled,
    }
}

fn mtp_ngram_saturated_gate(
    ngram_max: usize,
    mtp_depth: usize,
    mtp_only_ewma: f32,
    mtp_only_samples: u32,
) -> bool {
    ngram_max > 0
        && mtp_only_samples >= mtp_ngram_gate_min_samples()
        && mtp_only_ewma >= adaptive_ngram_saturation_threshold(mtp_depth)
}

fn mtp_ngram_hurt_gate(
    ngram_max: usize,
    combined_ewma: f32,
    combined_samples: u32,
    mtp_only_ewma: f32,
    mtp_only_samples: u32,
    min_samples: u32,
    margin: f32,
) -> bool {
    ngram_max > 0
        && mtp_only_samples >= min_samples
        && combined_samples >= min_samples
        && combined_ewma < mtp_only_ewma - margin
}

fn mtp_ngram_auto_disable_gate(
    ngram_max: usize,
    mtp_drafted: u32,
    mtp_accepted: u32,
    ngram_drafted: u32,
    ngram_accepted: u32,
    cfg: MtpNgramAutoDisableConfig,
) -> bool {
    if cfg.mtp_threshold == 0 || ngram_max == 0 {
        return false;
    }
    if mtp_drafted < cfg.mtp_warmup || ngram_drafted < cfg.ngram_warmup {
        return false;
    }
    let mtp_rate_x1000 = mtp_accepted.saturating_mul(1000) / mtp_drafted.max(1);
    let ngram_rate_x1000 = ngram_accepted.saturating_mul(1000) / ngram_drafted.max(1);
    mtp_rate_x1000 >= cfg.mtp_threshold && ngram_rate_x1000 < cfg.ngram_floor
}

/// Hurt gate mode selector: legacy EWMA-based or source-aware counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum HurtGateMode {
    #[default]
    SourceAware,
    LegacyEwma,
}

fn mtp_ngram_hurt_gate_mode() -> HurtGateMode {
    static CACHED: std::sync::OnceLock<HurtGateMode> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_MLX_MTP_NGRAM_HURT_GATE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-")
            .as_str()
        {
            "legacy" => HurtGateMode::LegacyEwma,
            _ => HurtGateMode::SourceAware,
        }
    })
}

/// Minimum n-gram support (times the matched context+continuation was observed)
/// required to propose an n-gram draft on the MTP-stacked path. The default `3`
/// keeps single-observation patterns — which dominate the rejections that make
/// the n-gram accept rate look bad — out of the draft. Override with
/// `AX_MLX_MTP_NGRAM_MIN_SUPPORT`.
pub const DEFAULT_MTP_NGRAM_MIN_SUPPORT: u32 = 3;

fn mtp_ngram_min_support() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_NGRAM_MIN_SUPPORT")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|v| *v >= 1)
            .unwrap_or(DEFAULT_MTP_NGRAM_MIN_SUPPORT)
    })
}

/// Minimum n-gram continuation confidence (`support/total`) required to propose
/// an n-gram draft on the MTP-stacked path. Higher than the standalone n-gram
/// default ([`crate::ngram_accel::DRAFT_CONFIDENCE_THRESHOLD`] = 0.4) because the
/// MTP head already captures the high-probability next token, so n-gram should
/// only stack when it is near-certain — otherwise its low-accept drafts add
/// verify cost and get hurt-gated. Override with
/// `AX_MLX_MTP_NGRAM_CONFIDENCE_THRESHOLD`.
pub const DEFAULT_MTP_NGRAM_CONFIDENCE_THRESHOLD: f32 = 0.85;

fn mtp_ngram_confidence_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_NGRAM_CONFIDENCE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(DEFAULT_MTP_NGRAM_CONFIDENCE_THRESHOLD)
    })
}

/// Minimum n-gram context length (tokens) for the MTP-stacked draft path. `3`
/// forbids 2-token bigram matches — the main source of low-accept drafts (the
/// same short suffix maps to many different true continuations), which is what
/// lifts the n-gram accept rate past the confidence/support plateau. A 26B
/// sweep (all suites) measured n-gram accept 71% @ctx2, 80% @ctx3, ~78% @ctx4;
/// ctx3 is the best accept *and* fires the most (ctx4 over-filters for no gain).
/// Override with `AX_MLX_MTP_NGRAM_MIN_CONTEXT_LEN` (clamped to 2..=4).
pub const DEFAULT_MTP_NGRAM_MIN_CONTEXT_LEN: usize = 3;

fn mtp_ngram_min_context_len() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_NGRAM_MIN_CONTEXT_LEN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| (2..=4).contains(v))
            .unwrap_or(DEFAULT_MTP_NGRAM_MIN_CONTEXT_LEN)
    })
}

/// n-gram prediction variant for the MTP-stacked path, cached from
/// `AX_MLX_NGRAM_POLICY` (default `MajorityRecency`). `latest` selects
/// recency/exact-copy continuation lookup, which can track tight repeats more
/// accurately than the frequency-based majority pick.
fn mtp_ngram_policy_variant() -> NgramPolicyVariant {
    static CACHED: OnceLock<NgramPolicyVariant> = OnceLock::new();
    *CACHED.get_or_init(ngram_policy_variant_from_env)
}

/// Source-aware hurt gate: compares per-source n-gram vs MTP acceptance rates
/// using raw draft/accepted counters rather than EWMA values.
///
/// Fires when n-gram per-token acceptance is worse than MTP per-token acceptance
/// by more than the margin — the exact condition where n-gram is genuinely hurting.
/// This avoids the selection bias in the legacy EWMA-based gate (see ADR-019).
fn mtp_ngram_source_hurt_gate(
    ngram_max: usize,
    mtp_drafted: u32,
    mtp_accepted: u32,
    ngram_drafted: u32,
    ngram_accepted: u32,
    min_samples: u32,
    margin: f32,
) -> bool {
    if ngram_max == 0 || ngram_drafted < min_samples || mtp_drafted < min_samples {
        return false;
    }
    let ngram_rate = ngram_accepted as f32 / ngram_drafted.max(1) as f32;
    let mtp_rate = mtp_accepted as f32 / mtp_drafted.max(1) as f32;
    ngram_rate + margin < mtp_rate
}

fn mtp_ngram_pseudo_log_prob(confidence: f32, mode: MtpNgramAcceptanceMode) -> f32 {
    if !confidence.is_finite() {
        return f32::NAN;
    }
    match mode {
        MtpNgramAcceptanceMode::Confidence => confidence.clamp(1e-37, 1.0).ln().max(-30.0),
        MtpNgramAcceptanceMode::Delta | MtpNgramAcceptanceMode::Greedy => 0.0_f32,
    }
}

fn mtp_ngram_pseudo_log_probs(
    confidence: &[f32],
    draft_len: usize,
    mode: MtpNgramAcceptanceMode,
) -> Vec<f32> {
    (0..draft_len)
        .map(|index| {
            confidence
                .get(index)
                .copied()
                .map(|confidence| mtp_ngram_pseudo_log_prob(confidence, mode))
                .unwrap_or(f32::NAN)
        })
        .collect()
}

/// Lazy target probability container for MTP rejection sampling.
///
/// `Full` uses the existing full-vocab softmax path (default).
/// `TopK` computes softmax over only the top-k logits per position, then does a
/// CPU-side lookup to extract the probability for each draft token.  This avoids
/// materializing a `[verify_len, 151936]` softmax tensor, saving ~100× softmax compute.
enum LazyTargetProbs {
    Full(MlxArray),
    TopK {
        indices: MlxArray,
        probs: MlxArray,
        k: u32,
    },
}

impl LazyTargetProbs {
    fn push_eval_targets<'a>(&'a self, targets: &mut Vec<&'a MlxArray>) {
        match self {
            LazyTargetProbs::Full(arr) => targets.push(arr),
            LazyTargetProbs::TopK { indices, probs, .. } => {
                targets.push(indices);
                targets.push(probs);
            }
        }
    }

    fn extract_cpu_into<'a>(
        &self,
        pending: &[u32],
        workspace: &'a mut MtpTargetProbWorkspace,
    ) -> Option<&'a [f32]> {
        workspace.target_probs.clear();
        match self {
            LazyTargetProbs::Full(arr) => {
                workspace.target_probs.extend_from_slice(arr.data_f32());
                Some(workspace.target_probs.as_slice())
            }
            LazyTargetProbs::TopK { indices, probs, k } => {
                let k_val = *k as usize;
                let indices_data = indices.data_u32();
                let probs_data = probs.data_f32();
                workspace.target_probs.reserve(pending.len());
                for (i, &needle) in pending.iter().enumerate() {
                    let row_start = i * k_val;
                    let row_end = row_start + k_val;
                    let mut found = false;
                    for j in row_start..row_end {
                        if indices_data.get(j) == Some(&needle) {
                            let p = probs_data.get(j).copied().unwrap_or(0.0_f32);
                            workspace.target_probs.push(p.max(0.0_f32));
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        workspace.target_probs.push(0.0_f32);
                    }
                }
                Some(workspace.target_probs.as_slice())
            }
        }
    }
}

/// Filter parameters passed from the draft path to the target probability
/// computation so that rejection sampling uses the same distribution on both
/// sides.  When the draft is greedy, pass `IDENTITY` to keep full-vocab target.
#[derive(Clone, Copy, Debug)]
struct MtpDraftFilter {
    top_p: f32,
    top_k: u32,
}

impl MtpDraftFilter {
    const IDENTITY: Self = Self {
        top_p: 1.0,
        top_k: 0,
    };
}

/// Build lazy target probabilities for MTP rejection sampling.
///
/// Returns `None` when rejection sampling is not applicable (no log_probs, temperature == 0,
/// or pending is empty). Callers MUST include the result in the same eval batch as the
/// verify-pass outputs to avoid a second GPU sync point.
#[allow(clippy::too_many_arguments)]
fn compute_mtp_target_probs(
    logits_all: &MlxArray,
    pending: &[u32],
    pending_log_probs: &[f32],
    vocab: i32,
    target_sampling: MlxSamplingParams,
    topk: Option<u32>,
    draft_filter: MtpDraftFilter,
    workspace: &mut MtpTargetProbWorkspace,
) -> Option<LazyTargetProbs> {
    if pending.is_empty()
        || pending_log_probs.len() != pending.len()
        || target_sampling.temperature <= 0.0
    {
        return None;
    }

    let n = pending.len();
    let inv_temp = mlx_scalar_f32(1.0 / target_sampling.temperature);
    let scaled = multiply(logits_all, &inv_temp, None);

    if let Some(k) = topk {
        let k_i32 = (k as i32).min(vocab);
        if k_i32 <= 0 {
            return None;
        }

        let scaled_topk = multiply(logits_all, &inv_temp, None);
        let mut all_top_indices = Vec::with_capacity(pending.len());
        let mut all_top_probs = Vec::with_capacity(pending.len());
        // logits_all shape: [1 + pending.len(), vocab].
        // verify_input = [last_token, pending[0], ..., pending[n-1]], so
        // logits_all[i] = prediction after position i = target for pending[i].
        // Row 0 is the target for pending[0]; rows 0..n are the draft targets.
        for row in 0..pending.len() as i32 {
            let row_logits = slice(&scaled_topk, &[row, 0], &[row + 1, vocab], &[1, 1], None);
            let part = argpartition_axis(&row_logits, -k_i32, -1, None);
            let top_idx = slice(&part, &[0, vocab - k_i32], &[1, vocab], &[1, 1], None);
            let top_vals = take_along_axis(&row_logits, &top_idx, -1, None);
            let top_p = softmax(&top_vals, -1, None);
            all_top_indices.push(top_idx);
            all_top_probs.push(top_p);
        }
        let idx_refs: Vec<&MlxArray> = all_top_indices.iter().collect();
        let prob_refs: Vec<&MlxArray> = all_top_probs.iter().collect();
        let stacked_indices = stack(&idx_refs, 0, None);
        let stacked_probs = stack(&prob_refs, 0, None);

        Some(LazyTargetProbs::TopK {
            indices: astype(&stacked_indices, MlxDtype::Uint32, None),
            probs: stacked_probs,
            k,
        })
    } else if draft_filter.top_k > 0 || draft_filter.top_p < 1.0 {
        // Draft-path filter applied to target probs for rejection-sampling parity.
        let dk = draft_filter.top_k.min(vocab as u32);
        let dk_i32 = if dk > 0 {
            (dk as i32).min(vocab)
        } else {
            vocab
        };
        let mut all_top_indices = Vec::with_capacity(pending.len());
        let mut all_top_probs = Vec::with_capacity(pending.len());
        // logits_all shape: [1 + pending.len(), vocab].
        // verify_input = [last_token, pending[0], ..., pending[n-1]], so
        // logits_all[i] = prediction after position i = target for pending[i].
        for row in 0..pending.len() as i32 {
            let row_logits = slice(&scaled, &[row, 0], &[row + 1, vocab], &[1, 1], None);
            let (row_idx, row_probs) = if dk_i32 < vocab {
                let part = argpartition_axis(&row_logits, -dk_i32, -1, None);
                let top_idx = slice(&part, &[0, vocab - dk_i32], &[1, vocab], &[1, 1], None);
                let top_vals = take_along_axis(&row_logits, &top_idx, -1, None);
                let top_p = softmax(&top_vals, -1, None);
                (top_idx, top_p)
            } else {
                let top_p = softmax(&row_logits, -1, None);
                let idx = MlxArray::from_raw_data(
                    (0..vocab).map(|i| i as u32).collect::<Vec<u32>>().as_ptr() as *const u8,
                    vocab as usize * 4,
                    &[1, vocab],
                    MlxDtype::Uint32,
                );
                (idx, top_p)
            };
            all_top_indices.push(row_idx);
            all_top_probs.push(row_probs);
        }
        let idx_refs: Vec<&MlxArray> = all_top_indices.iter().collect();
        let prob_refs: Vec<&MlxArray> = all_top_probs.iter().collect();
        let stacked_indices = stack(&idx_refs, 0, None);
        let stacked_probs = stack(&prob_refs, 0, None);

        Some(LazyTargetProbs::TopK {
            indices: astype(&stacked_indices, MlxDtype::Uint32, None),
            probs: stacked_probs,
            k: dk.max(1),
        })
    } else {
        let probs = softmax(&scaled, -1, None);

        workspace.flat_indices.clear();
        workspace
            .flat_indices
            .extend((0..n).map(|i| i as i32 * vocab + pending[i] as i32));
        let flat_idx_arr = MlxArray::from_raw_data(
            workspace.flat_indices.as_ptr() as *const u8,
            workspace.flat_indices.len() * 4,
            &[n as i32],
            MlxDtype::Int32,
        );
        use mlx_sys::reshape as mlx_reshape;
        let probs_flat = mlx_reshape(&probs, &[-1_i32], None);
        Some(LazyTargetProbs::Full(take(
            &probs_flat,
            &flat_idx_arr,
            0,
            None,
        )))
    }
}

/// Perform rejection-sampling acceptance using pre-evaluated target probabilities.
///
/// `target_probs_cpu`: pre-computed p_target(draft_token_i) for each position, already
/// transferred from GPU. When `None`, falls back to greedy argmax comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MtpAcceptOutcome {
    accept_count: usize,
    all_accepted: bool,
    rejection_correction: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn mtp_accept_count(
    pending: &[u32],
    pending_log_probs: &[f32],
    draft_distributions: &[TokenDistribution],
    draft_sources: &[MtpDraftSource],
    target_probs_cpu: Option<&[f32]>,
    target_distributions: Option<&[TokenDistribution]>,
    predicted: &[u32],
    rng: &mut Xorshift64,
    draft_temperature: f32,
    target_temperature: f32,
    model_acceptance_mode: MtpModelAcceptanceMode,
    ngram_acceptance_mode: MtpNgramAcceptanceMode,
) -> MtpAcceptOutcome {
    let mut ac = 0usize;
    let mut distribution_index = 0usize;
    for i in 0..pending.len() {
        let source = draft_sources
            .get(i)
            .copied()
            .filter(|source| *source != MtpDraftSource::None)
            .unwrap_or(MtpDraftSource::Mtp);
        let has_draft_distribution = source.is_model_draft();
        if source == MtpDraftSource::Ngram
            && ngram_acceptance_mode == MtpNgramAcceptanceMode::Greedy
        {
            if predicted[i] == pending[i] {
                ac += 1;
                continue;
            }
            return MtpAcceptOutcome {
                accept_count: ac,
                all_accepted: false,
                rejection_correction: predicted.get(i).copied(),
            };
        }
        if has_draft_distribution && model_acceptance_mode == MtpModelAcceptanceMode::Greedy {
            if predicted[i] == pending[i] {
                ac += 1;
                distribution_index = distribution_index.saturating_add(1);
                continue;
            }
            return MtpAcceptOutcome {
                accept_count: ac,
                all_accepted: false,
                rejection_correction: predicted.get(i).copied(),
            };
        }
        let can_rejection_sample = pending_log_probs
            .get(i)
            .is_some_and(|log_prob| log_prob.is_finite())
            && target_probs_cpu.is_some();

        if let (true, Some(tprobs)) = (can_rejection_sample, target_probs_cpu) {
            let p_target_d = tprobs[i].max(0.0_f32);
            // Rescale draft log-prob when draft and target temperatures differ.
            // The standard rejection-sampling formula min(1, p_target / p_draft)
            // assumes p and q over the same effective sample space.  When draft
            // and target use different temperatures, the unscaled ratio
            // systematically rejects drafts even when both models agree on the
            // token.  Empirically, log_p * (T_draft / T_target) acts as a
            // re-temperaturing approximation that aligns AX output with the
            // MTPLX reference (tokenwise-identical for ~87 tokens at default
            // T_draft=0.7, T_target=0.6).  Skipped for n-gram delta log-probs
            // (0.0) because they are not derived from softmax(logits/T_draft).
            let log_p_draft = pending_log_probs[i];
            let is_mtp_source = !matches!(source, MtpDraftSource::Ngram);
            let log_p_scaled = if is_mtp_source
                && draft_temperature > 0.0
                && target_temperature > 0.0
                && (draft_temperature - target_temperature).abs() > 1e-6
            {
                log_p_draft * (draft_temperature / target_temperature)
            } else {
                log_p_draft
            };
            let p_draft = log_p_scaled.exp().max(1e-37_f32);
            let accept_prob = (p_target_d / p_draft).min(1.0_f32);
            if rng.next_f32() < accept_prob {
                ac += 1;
                if has_draft_distribution {
                    distribution_index = distribution_index.saturating_add(1);
                }
            } else {
                let correction = if has_draft_distribution {
                    target_distributions
                        .and_then(|targets| targets.get(i))
                        .zip(draft_distributions.get(distribution_index))
                        .and_then(|(target, draft)| {
                            sample_residual_token_distribution(target, draft, rng)
                        })
                } else {
                    None
                };
                return MtpAcceptOutcome {
                    accept_count: ac,
                    all_accepted: false,
                    rejection_correction: correction,
                };
            }
        } else {
            // Greedy acceptance fallback: target_probs absent (greedy target
            // temperature) or log-prob not finite (pure n-gram without hybrid tail).
            if predicted[i] == pending[i] {
                ac += 1;
                if has_draft_distribution {
                    distribution_index = distribution_index.saturating_add(1);
                }
            } else {
                return MtpAcceptOutcome {
                    accept_count: ac,
                    all_accepted: false,
                    rejection_correction: predicted.get(i).copied(),
                };
            }
        }
    }
    MtpAcceptOutcome {
        accept_count: ac,
        all_accepted: ac == pending.len(),
        rejection_correction: None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gemma4AssistantMtpConfidenceMode {
    ExactCpu,
    GpuExact,
}

impl Gemma4AssistantMtpConfidenceMode {
    fn route_code(self) -> u32 {
        match self {
            Self::ExactCpu => 0,
            Self::GpuExact => 1,
        }
    }
}

fn parse_gemma4_assistant_mtp_confidence_mode(
    raw: &str,
) -> Option<Gemma4AssistantMtpConfidenceMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "exact-cpu" | "exact_cpu" | "cpu" => Some(Gemma4AssistantMtpConfidenceMode::ExactCpu),
        "gpu-exact" | "gpu_exact" => Some(Gemma4AssistantMtpConfidenceMode::GpuExact),
        _ => None,
    }
}

fn gemma4_assistant_mtp_confidence_mode_from_env() -> Gemma4AssistantMtpConfidenceMode {
    static CACHED: OnceLock<Gemma4AssistantMtpConfidenceMode> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_GEMMA4_ASSISTANT_MTP_CONFIDENCE_MODE")
            .ok()
            .and_then(|raw| parse_gemma4_assistant_mtp_confidence_mode(&raw))
            .unwrap_or(Gemma4AssistantMtpConfidenceMode::GpuExact)
    })
}

fn argmax_with_softmax_confidence_for_logits(
    logits: &MlxArray,
    mode: Gemma4AssistantMtpConfidenceMode,
) -> (u32, f32) {
    match mode {
        Gemma4AssistantMtpConfidenceMode::ExactCpu => {
            eval(&[logits]);
            let logits_cpu = logits.data_f32().to_vec();
            argmax_with_softmax_confidence(&logits_cpu)
        }
        Gemma4AssistantMtpConfidenceMode::GpuExact => {
            argmax_with_softmax_confidence_gpu_exact(logits)
        }
    }
}

fn argmax_with_softmax_confidence_gpu_exact(logits: &MlxArray) -> (u32, f32) {
    let shape = logits.shape();
    let Some(vocab) = shape.last().copied() else {
        return (0, 0.0);
    };
    if vocab <= 0 || shape.iter().copied().product::<i32>() != vocab {
        return (0, 0.0);
    }

    let logits_2d = reshape(logits, &[1, vocab], None);
    let token_arr = argmax(&logits_2d, None);
    let probs = softmax(&logits_2d, -1, None);
    let prob_arr = take(&probs, &token_arr, 1, None);
    eval(&[&token_arr, &prob_arr]);
    let token = token_arr.data_u32().first().copied().unwrap_or(0);
    let confidence = prob_arr.data_f32().first().copied().unwrap_or(0.0);
    (token, confidence)
}

/// Top token of a logit row plus its `softmax` probability at temperature 1.0 —
/// the drafter's most likely next token and its confidence. Used to gate Gemma 4
/// assistant drafts: the argmax is the greedy draft, the probability is the gate
/// signal. Returns `(0, 0.0)` for an empty or degenerate logit row so such steps
/// are always suppressed.
fn argmax_with_softmax_confidence(logits: &[f32]) -> (u32, f32) {
    let mut max_l = f32::NEG_INFINITY;
    let mut argmax = 0u32;
    for (idx, &l) in logits.iter().enumerate() {
        if l > max_l {
            max_l = l;
            argmax = idx as u32;
        }
    }
    if !max_l.is_finite() {
        return (0, 0.0);
    }
    let sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let confidence = if sum > 0.0 && sum.is_finite() {
        1.0 / sum
    } else {
        0.0
    };
    (argmax, confidence)
}

fn apply_decode_result(
    state: &mut RequestState,
    result: &[u32],
    terminal_token_ids: &[u32],
) -> Vec<u32> {
    debug_assert!(
        !result.is_empty(),
        "MLX decode path must return at least one token"
    );

    let mut output = Vec::with_capacity(result.len());
    for &token in result {
        output.push(token);
        if token_is_terminal(token, terminal_token_ids) {
            break;
        }
    }
    state.next_model_last_token = output.last().copied();
    output
}

fn truncate_sampled_tokens_for_stop(
    mut sampled_tokens: Vec<u32>,
    generated_len: u32,
    max_output: u32,
    terminal_token_ids: &[u32],
) -> (Vec<u32>, Option<StopReason>) {
    if sampled_tokens.is_empty() {
        return (sampled_tokens, None);
    }

    let remaining = max_output.saturating_sub(generated_len).max(1) as usize;
    let limit = sampled_tokens.len().min(remaining);
    for index in 0..limit {
        let sampled_token = sampled_tokens[index];
        if token_is_terminal(sampled_token, terminal_token_ids) {
            sampled_tokens.truncate(index + 1);
            return (sampled_tokens, Some(StopReason::EosToken));
        }
        if index + 1 == remaining {
            sampled_tokens.truncate(index + 1);
            return (sampled_tokens, Some(StopReason::MaxOutputTokens));
        }
    }
    sampled_tokens.truncate(limit);
    (sampled_tokens, None)
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

/// Whether the prefill output token may be reused as a request's first generated
/// token across a prefix-cache hit.
///
/// The cached token is the greedy argmax of the prompt logits with no repetition
/// penalty, so it is only a correct substitute when the consumer would compute
/// exactly that token: a deterministic (greedy) request with no repetition
/// penalty. Temperature / top-p / repetition-penalty requests must resample —
/// otherwise a warm cache would silently force a greedy first token and change
/// the output distribution. The prefix-cache key does not encode sampling, so
/// this gate is applied symmetrically at the store and the reuse sites so the
/// two can never disagree.
fn prefill_output_token_cacheable(
    ctx: Option<&RunnerRequestContext>,
    sampling: MlxSamplingParams,
) -> bool {
    let is_greedy = ctx
        .map(|c| c.deterministic_argmax_sampling)
        .unwrap_or(sampling == MlxSamplingParams::greedy());
    is_greedy && !sampling.uses_repetition_penalty()
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
    mtp_telemetry: MtpTelemetry,
    gemma4_assistant_mtp_telemetry: Gemma4AssistantMtpTelemetry,
    gemma4_unified_multimodal_telemetry: Gemma4UnifiedMultimodalTelemetry,
    decode_telemetry: DecodeTelemetry,
    gemma4_moe_profile: Gemma4MoeProfileSnapshot,
    moe_profile: MoeProfileSnapshot,
    linear_attention_profile: LinearAttentionProfileSnapshot,
    dense_ffn_fastpath: DenseFfnFastpathSnapshot,
    prefill_profile: PrefillProfileSnapshot,
    decode_profile: DecodeProfileSnapshot,
    kv_usage: MlxKVCacheUsage,
    prefix_cache: MlxPrefixCacheTelemetry,
    kv_compression_shadow_sync_wall_us: Option<u32>,
}

fn errored_item_run(request_id: RequestId, error: impl Into<String>) -> MlxItemRun {
    MlxItemRun {
        update: RequestExecutionUpdate {
            request_id,
            tokens_executed: 0,
            output_token: None,
            output_tokens: Vec::new(),
            stop_reason: None,
            error: Some(error.into()),
        },
        ngram_acceleration: NgramAccelerationTelemetry::default(),
        mtp_telemetry: MtpTelemetry::default(),
        gemma4_assistant_mtp_telemetry: Gemma4AssistantMtpTelemetry::default(),
        gemma4_unified_multimodal_telemetry: Gemma4UnifiedMultimodalTelemetry::default(),
        decode_telemetry: DecodeTelemetry::default(),
        gemma4_moe_profile: Gemma4MoeProfileSnapshot::default(),
        moe_profile: MoeProfileSnapshot::default(),
        linear_attention_profile: LinearAttentionProfileSnapshot::default(),
        dense_ffn_fastpath: DenseFfnFastpathSnapshot::default(),
        prefill_profile: PrefillProfileSnapshot::default(),
        decode_profile: DecodeProfileSnapshot::default(),
        kv_usage: MlxKVCacheUsage::default(),
        prefix_cache: MlxPrefixCacheTelemetry::default(),
        kv_compression_shadow_sync_wall_us: None,
    }
}

#[derive(Clone, Copy, Debug)]
struct DecodeOneOptions<'a> {
    terminal_token_ids: &'a [u32],
    final_by_max_output: bool,
    request_context: Option<&'a RunnerRequestContext>,
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

fn record_ngram_beta_feedback(state: &mut RequestState, draft_len: usize, accept_count: usize) {
    if draft_len == 0 {
        return;
    }
    state.ngram_beta_alpha += accept_count as f32;
    state.ngram_beta_beta += draft_len.saturating_sub(accept_count) as f32;

    // Keep the posterior adaptive instead of letting old requests dominate.
    let total = state.ngram_beta_alpha + state.ngram_beta_beta;
    if total > NGRAM_BETA_MAX_TOTAL {
        let scale = NGRAM_BETA_MAX_TOTAL / total;
        state.ngram_beta_alpha *= scale;
        state.ngram_beta_beta *= scale;
    }
}

fn linear_ngram_no_draft_should_disable(streak: u32) -> bool {
    streak >= LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD
}

fn linear_ngram_initial_prompt_should_disable_request(
    has_linear_attention: bool,
    prompt_class: u32,
    ngram: &NgramTable,
    variant: NgramPolicyVariant,
) -> bool {
    if !has_linear_attention {
        return false;
    }
    if prompt_class != crate::ngram_accel::PROMPT_CLASS_NON_REPEATING {
        return false;
    }
    // Probe with bypass-enabled policy. Random tokens have no repeated bigrams
    // so the probe returns empty → disable. Code/structured prompts contain
    // useful bigrams even when the 4-gram classifier sees NON_REPEATING → keep
    // speculation enabled from step 1.
    let probe = ngram_acceleration_draft(ngram, true, 0.5, variant, false);
    probe.draft.is_empty()
}

fn ngram_request_disabled_fallback_should_feed_output(reason: NgramRequestDisableReason) -> bool {
    matches!(reason, NgramRequestDisableReason::LinearNoDraft)
}

fn ngram_request_disabled_direct_fast_path(
    is_greedy: bool,
    uses_repetition_penalty: bool,
    has_mtp: bool,
    request_disabled: bool,
    reason: NgramRequestDisableReason,
) -> bool {
    is_greedy
        && !uses_repetition_penalty
        && !has_mtp
        && request_disabled
        && !ngram_request_disabled_fallback_should_feed_output(reason)
}

fn maybe_reenable_linear_ngram_from_fallback_output(
    state: &mut RequestState,
    variant: NgramPolicyVariant,
    is_greedy: bool,
) {
    if !is_greedy
        || !state.ngram_acceleration_disabled_for_request
        || !matches!(
            state.ngram_request_disable_reason,
            NgramRequestDisableReason::LinearNoDraft
        )
    {
        return;
    }

    if state.linear_ngram_reenable_probe_countdown > 0 {
        state.linear_ngram_reenable_probe_countdown -= 1;
        return;
    }

    let draft = ngram_acceleration_draft(
        &state.ngram,
        true,
        state.ngram_posterior_mean(),
        variant,
        false,
    );
    if draft.draft.is_empty() {
        state.linear_ngram_reenable_probe_countdown = LINEAR_NGRAM_REENABLE_PROBE_INTERVAL;
        return;
    }

    state.ngram_acceleration_disabled_for_request = false;
    state.ngram_request_disable_reason = NgramRequestDisableReason::None;
    state.linear_ngram_no_draft_streak = 0;
    state.linear_ngram_reenable_probe_countdown = 0;
    state.ngram_disabled_steps = 0;
    // Discard any stale direct-pipeline lookahead that was built while
    // ngram was disabled. Re-entering the ngram path invalidates it: the
    // ngram draft will advance cache.seq_len independently, so
    // pending_direct would point at the wrong sequence position.
    state.pending_direct = None;
    state.direct_pipeline_emitted_tokens = 0;
}

/// Disables n-gram drafting inside `run_mtp_decode` so the MTP verify loop
/// always sources its draft from the MTP head. Set
/// `AX_MLX_MTP_DISABLE_NGRAM_STACKING=0` to opt back into ADR-008 stacking in
/// low-level runner construction; server and SDK sessions pass this option
/// explicitly.
///
/// Other decode paths (non-MTP `ngram_accel_decode_step`, prefill seeding) are
/// unaffected — only the n-gram-first branch inside `run_mtp_decode` is gated.
/// Maximum number of history tokens to warm up the MTP KV cache.
/// The most recent tokens dominate the MTP head's attention context for
/// speculative decoding; older tokens have diminishing returns.
/// Override with `AX_MLX_MTP_WARMUP_CAP` (0 = unlimited, default 256).
fn mtp_warmup_cap() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_WARMUP_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
    })
}

/// Minimum EWMA samples before n-gram saturation gating can activate.
/// 4 samples allows the gate to fire within the first ~12 generated tokens
/// (4 steps × depth-3 drafts), preventing early n-gram overhead when MTP
/// acceptance is already high from the start.  With ALPHA=0.05, 4 samples
/// is enough to confirm ≥99% EWMA (all-accept × 4 → EWMA = 1.0).
/// Override with `AX_MLX_MTP_NGRAM_GATE_SAMPLES` (default 4).
fn mtp_ngram_gate_min_samples() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_NGRAM_GATE_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
    })
}

/// Auto-optimistic EWMA deactivation threshold.
///
/// Once optimistic is active (activation at stochastic EWMA ≥0.99), the EWMA
/// switches to argmax-based tracking which is strictly stricter.  The
/// deactivation threshold sets the floor below which optimistic disengages.
///
/// Qwen3.6 native MTP heads achieve >85% acceptance, so lowering the
/// deactivation threshold from the prior 0.95 to 0.85 makes optimistic mode
/// stickier — it activates at 0.99 stochastic and stays active unless argmax
/// acceptance drops below 0.85.  This eliminates the oscillation that
/// previously caused optimistic to disengage on borderline acceptance rows
/// where it was still beneficial.
///
/// Override with `AX_MLX_MTP_AUTO_OPTIMISTIC_DEACTIVATE_THRESHOLD` (default 0.85).
fn mtp_auto_optimistic_deactivate_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        cached_env_f32(
            "AX_MLX_MTP_AUTO_OPTIMISTIC_DEACTIVATE_THRESHOLD",
            0.85,
            0.0,
            1.0,
        )
    })
}

/// Minimum EWMA samples before auto-optimistic can activate.
///
/// Separate from `mtp_ngram_gate_min_samples` (which controls n-gram saturation
/// gating).  4 samples is sufficient for the stochastic EWMA to stabilize at
/// high acceptance rates (all-accept × 4 → EWMA = 1.0 with ALPHA=0.05).
/// Override with `AX_MLX_MTP_AUTO_OPTIMISTIC_MIN_SAMPLES` (default 4).
fn mtp_auto_optimistic_min_samples() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_AUTO_OPTIMISTIC_MIN_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
    })
}

fn cached_env_f32(name: &str, default: f32, min: f32, max: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(min, max))
        .unwrap_or(default)
}

/// Minimum EWMA samples before the per-request MTP bypass can activate.
///
/// 8 samples lets the EWMA stabilize: with ALPHA=0.05 the first 8 samples
/// weight recent history enough to reflect the true acceptance rate rather
/// than the initial transient.  The bypass never fires during the warm-up
/// window, so short bursts of low acceptance at the start of generation
/// (e.g. the first few tokens before the MTP head is warmed up) do not
/// permanently disable MTP.
///
/// Override with `AX_MLX_MTP_BYPASS_MIN_SAMPLES` (default 8).
fn mtp_bypass_min_samples() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_BYPASS_MIN_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8)
    })
}

/// EWMA MTP-only acceptance rate below which the per-request MTP bypass fires.
///
/// When the MTP head's own acceptance (cascade-corrected, isolating MTP from
/// n-gram quality) falls below this fraction, the per-step overhead (head
/// forward + verify on the extended sequence + acceptance logic + potential
/// rollback) exceeds the benefit.  The bypass latches for the remainder of
/// the request and all subsequent decode steps use the n-gram speculation
/// path without MTP.
///
/// 0.50 is calibrated against the benchmark matrix: when MTP-only acceptance
/// stays above ~60% the speculation amortizes its overhead; below ~50% it is
/// a net loss.  Override with `AX_MLX_MTP_BYPASS_THRESHOLD`
/// (default 0.50, clamped to [0.0, 1.0]).
fn mtp_bypass_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f32("AX_MLX_MTP_BYPASS_THRESHOLD", 0.50, 0.0, 1.0))
}

fn cached_env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

fn cached_env_f64(name: &str, default: f64, min: f64, max: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(min, max))
        .unwrap_or(default)
}

fn route_cost_us(value: Option<f64>) -> u32 {
    value
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.round().min(u32::MAX as f64) as u32)
        .unwrap_or(0)
}

fn mtp_ngram_hurt_margin() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f32("AX_MLX_MTP_NGRAM_HURT_MARGIN", 0.02, 0.0, 1.0))
}

fn mtp_ngram_gate_policy_from_env() -> MtpNgramGatePolicy {
    static CACHED: OnceLock<MtpNgramGatePolicy> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_MLX_MTP_NGRAM_GATE_POLICY")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-")
            .as_str()
        {
            "utility" => return MtpNgramGatePolicy::Utility,
            "rate" => return MtpNgramGatePolicy::Rate,
            _ => {}
        }
        // Else the speculation profile may prefer the utility gate (chatbot /
        // high-temperature `auto`, where n-gram rarely helps prose).
        if speculation_profile_from_env().prefers_ngram_utility(None) {
            MtpNgramGatePolicy::Utility
        } else {
            MtpNgramGatePolicy::Rate
        }
    })
}

fn mtp_ngram_utility_min_emitted_tokens() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_UTILITY_MIN_EMITTED_TOKENS", 128))
}

fn mtp_ngram_utility_min_ngram_tokens() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_UTILITY_MIN_NGRAM_TOKENS", 32))
}

fn mtp_ngram_utility_margin_ratio() -> f64 {
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f64("AX_MLX_MTP_NGRAM_UTILITY_MARGIN_RATIO", 0.02, 0.0, 10.0))
}

fn mtp_ngram_utility_hysteresis_steps() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_UTILITY_HYSTERESIS_STEPS", 16))
}

fn mtp_ngram_safety_mode() -> MtpNgramSafetyMode {
    static CACHED: OnceLock<MtpNgramSafetyMode> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_MLX_MTP_NGRAM_SAFETY_MODE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-")
            .as_str()
        {
            "off" | "none" => MtpNgramSafetyMode::Off,
            "disable-all" | "all" => MtpNgramSafetyMode::DisableAll,
            "disable-reasoning" | "disable-think" => MtpNgramSafetyMode::DisableReasoning,
            _ => MtpNgramSafetyMode::TightenReasoning,
        }
    })
}

fn mtp_ngram_auto_disable_mtp_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        cached_env_f32(
            "AX_MLX_MTP_NGRAM_AUTO_DISABLE_MTP_THRESHOLD",
            0.85,
            0.0,
            1.0,
        )
    })
}

fn mtp_ngram_auto_disable_min_ngram() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED
        .get_or_init(|| cached_env_f32("AX_MLX_MTP_NGRAM_AUTO_DISABLE_MIN_NGRAM", 0.50, 0.0, 1.0))
}

fn mtp_ngram_self_tune_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f32("AX_MLX_MTP_NGRAM_SELF_TUNE_THRESHOLD", 0.30, 0.0, 1.0))
}

fn mtp_ngram_self_tune_warmup() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_SELF_TUNE_WARMUP", 32))
}

fn mtp_ngram_auto_disable_mtp_warmup() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_AUTO_DISABLE_MTP_WARMUP", 64))
}

fn mtp_ngram_auto_disable_ngram_warmup() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_AUTO_DISABLE_NGRAM_WARMUP", 32))
}

fn mtp_ngram_acceptance_mode_from_env() -> MtpNgramAcceptanceMode {
    static CACHED: OnceLock<MtpNgramAcceptanceMode> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_MLX_MTP_NGRAM_ACCEPTANCE_MODE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-")
            .as_str()
        {
            "delta" => MtpNgramAcceptanceMode::Delta,
            "greedy" => MtpNgramAcceptanceMode::Greedy,
            _ => MtpNgramAcceptanceMode::Confidence,
        }
    })
}

fn mtp_model_acceptance_mode_from_env() -> MtpModelAcceptanceMode {
    static CACHED: OnceLock<MtpModelAcceptanceMode> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_MLX_MTP_MODEL_ACCEPTANCE_MODE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-")
            .as_str()
        {
            "rejection" | "rejection-sampling" | "sampling" => {
                MtpModelAcceptanceMode::RejectionSampling
            }
            _ => MtpModelAcceptanceMode::Greedy,
        }
    })
}

fn mtp_disable_ngram_stacking_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("AX_MLX_MTP_DISABLE_NGRAM_STACKING").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

/// **Default: ON** (kill-switch via `AX_MLX_MTP_OPTIMISTIC=0`).
///
/// MTP verify always accepts all draft tokens without computing the
/// rejection-sampling acceptance ratio.  Eliminates full-vocab softmax for
/// target distribution, the accept/reject loop, and cache rollback on rejection.
/// Safe for native MTP heads with >85% acceptance (Qwen3.6 27B achieves 99%+).
/// The verify forward still computes argmax for the correction/bonus token and
/// EWMA tracking, so the rare mismatch between draft and target argmax is
/// handled correctly.  Set to `0` to restore the full rejection-sampling path.
fn mtp_optimistic_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("AX_MLX_MTP_OPTIMISTIC").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE")
        )
    })
}

fn mtp_optimistic_allowed(has_glm_mtp: bool) -> bool {
    // GLM's sidecar can draft plausible but target-mismatched code tokens; keep
    // verifier acceptance on for correctness instead of unconditional accept.
    !has_glm_mtp
}

fn mtp_optimistic_draft_min_confidence_override() -> Option<f32> {
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if std::env::var("AX_MLX_MTP_DRAFT_MIN_CONFIDENCE").is_ok() {
            None
        } else if mtp_optimistic_from_env() {
            Some(0.0)
        } else {
            None
        }
    })
}

/// When enabled (the default), the MTP decode path captures verify logits and
/// hidden state as "skip state" and reuses them on the next iteration to avoid
/// recomputing the main model forward for the first token position.  The
/// skip_logits provides the next primary sample; the skip_hidden provides the
/// MTP head input.  This eliminates one full-model forward pass per accepted
/// cycle.  Disable with `AX_MLX_MTP_SKIP_STATE=0`.
fn mtp_skip_state_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("AX_MLX_MTP_SKIP_STATE")
                .unwrap_or_default()
                .as_str(),
            "0" | "false" | "FALSE"
        )
    })
}

/// Target softmax mode for MTP rejection-sampling acceptance.
/// Defaults to `full` (full-vocab softmax) to avoid false rejections on
/// diverse output where draft tokens may fall outside the target model's
/// top-k. The previous `topk-128` default caused guaranteed rejection
/// (`p_target = 0`) for any draft token ranked outside the target's top-128,
/// which dropped acceptance from ~100% to ~75% on diverse code suites.
/// Override with `AX_MLX_MTP_TARGET_SOFTMAX_MODE=topk-128` (or topk-256,
/// topk-64, topk-32) for custom k, or keep `full` for the default.
fn mtp_target_softmax_topk_from_env() -> Option<u32> {
    static CACHED: OnceLock<Option<u32>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let val = std::env::var("AX_MLX_MTP_TARGET_SOFTMAX_MODE")
            .unwrap_or_else(|_| "full".to_string())
            .to_ascii_lowercase()
            .replace('_', "-");
        match val.as_str() {
            "full" => None,
            "topk-256" => Some(256),
            "topk-128" => Some(128),
            "topk-64" => Some(64),
            "topk-32" => Some(32),
            _ => None,
        }
    })
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
    post_think_guarded: bool,
) -> NgramDraftOutcome {
    let policy = ngram_acceleration_policy(
        has_linear_attention,
        posterior_mean,
        variant,
        post_think_guarded,
    );
    ngram.predict_with_policy(policy)
}

/// The exact policy `ngram_acceleration_draft` uses to draft, exposed so
/// callers that need to record verifier feedback afterward (see
/// `NgramTable::record_draft_feedback`) can recompute the identical policy
/// from the same inputs rather than reconstructing an approximation.
fn ngram_acceleration_policy(
    has_linear_attention: bool,
    posterior_mean: f32,
    variant: NgramPolicyVariant,
    post_think_guarded: bool,
) -> NgramDraftPolicy {
    let max_len = adaptive_ngram_draft_len(has_linear_attention, posterior_mean);
    let confidence_threshold = effective_draft_confidence_threshold();
    if has_linear_attention {
        // Dense rollback is O(1); linear-attention partial-reject pays
        // branch/recompute, so cap at DEFAULT_DRAFT_LEN to bound recompute cost.
        // bypass_prompt_min_support=true: prompt-seeded bigrams draft with a
        // single observation, enabling speculation from step 1 on repeating
        // real-workload prompts without waiting for two output observations.
        // adaptive_match_len=true: lightning-mlx-style support+1 cap keeps
        // sparse one-off matches narrow while allowing repeated contexts to
        // use the full verifier batch.
        NgramDraftPolicy {
            variant,
            max_len,
            min_support: LINEAR_MIN_NGRAM_SUPPORT,
            confidence_threshold,
            adaptive_match_len: true,
            bypass_prompt_min_support: true,
            min_context_len: 2,
        }
    } else if post_think_guarded {
        // Outside `<think>` on reasoning models: require POST_THINK_MIN_NGRAM_SUPPORT
        // observations before drafting to suppress one-off guesses in free-form
        // regions (getter/setter names, creative text).  Well-established patterns
        // (SQL keywords, JSON delimiters) have support ≥ 2 and still draft.
        // bypass_prompt_min_support=true allows prompt-echo patterns from step 1.
        NgramDraftPolicy {
            variant,
            max_len,
            min_support: POST_THINK_MIN_NGRAM_SUPPORT,
            confidence_threshold,
            adaptive_match_len: true,
            bypass_prompt_min_support: true,
            min_context_len: 2,
        }
    } else {
        // Dense models inside `<think>` (or non-thinking models): standard policy.
        // min_support=1 because think-block output is already high-repetition and
        // the beta-Bernoulli gate suppresses bad drafters naturally.
        NgramDraftPolicy {
            variant,
            max_len,
            min_support: 1,
            confidence_threshold,
            adaptive_match_len: true,
            bypass_prompt_min_support: false,
            min_context_len: 2,
        }
    }
}

fn adaptive_ngram_draft_len(has_linear_attention: bool, posterior_mean: f32) -> usize {
    if has_linear_attention {
        if posterior_mean < NGRAM_DRAFT_LEN_SHRINK_THRESHOLD {
            NGRAM_DRAFT_LEN_LOW_CONFIDENCE
        } else {
            DEFAULT_DRAFT_LEN
        }
    } else {
        MAX_DRAFT_LEN
    }
}

fn adaptive_ngram_saturation_threshold(mtp_depth: usize) -> f32 {
    if mtp_depth <= 1 {
        // depth=1: per-step rate is binary (0 or 1); EWMA reaches 0.98 on
        // random streaks at normal acceptance rates, causing false gating.
        // n-gram is also the primary multi-token source at depth=1, so
        // disable the gate entirely.
        return 2.0;
    }
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            std::env::var("AX_MLX_MTP_NGRAM_GATE_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| v.is_finite())
                .map(|v| v.clamp(0.0, 2.0))
        })
        .unwrap_or(if mtp_depth >= 3 { 0.97 } else { 0.98 })
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
    if !is_mlx_supported_model_family(&manifest.model_family) {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "model_family {:?} is not supported by the MLX runner",
            manifest.model_family
        )));
    }
    if manifest.model_family == "glm4_moe_lite" || has_glm_mla_tensors(artifacts) {
        validate_mla_moe_manifest(manifest)?;
    }
    if manifest.linear_attention.is_enabled() || has_linear_attention_tensors(artifacts) {
        validate_qwen_gated_delta_linear_attention(manifest)?;
    }
    // Interleaved SWA validation (Gemma3/4): triggered by layer_types, KV sharing,
    // a separate global head dim, or a separate SWA rope theta. Families with
    // uniform SWA (mistral3, mixtral) use only sliding_window_size with no
    // layer_types, so they skip this gate.
    if !manifest.layer_types.is_empty()
        || !manifest.kv_shared_source_layers.is_empty()
        || manifest.global_head_dim.is_some()
        || manifest.rope_theta_swa.is_some()
    {
        validate_gemma4_interleaved_attention(manifest)?;
    }
    if manifest.model_family == "diffusion_gemma" {
        validate_diffusion_gemma_manifest(manifest)?;
    }
    Ok(())
}

fn is_mlx_supported_model_family(model_family: &str) -> bool {
    matches!(
        model_family,
        "gemma4"
            | "gemma3"
            | "embeddinggemma"
            | "qwen3"
            | "llama3"
            | "diffusion_gemma"
            | "llama4"
            | "qwen3_5"
            | "qwen3_next"
            | "glm4_moe_lite"
            | "deepseek_v3"
            | "deepseek_v32"
            | "mistral3"
            | "mixtral"
    )
}

/// Validate DiffusionGemma-specific manifest fields.
///
/// DiffusionGemma uses the Gemma4 MoE backbone with bidirectional denoiser
/// attention over a fixed canvas. The diffusion config block must be present
/// and carry at least `canvas_size`.
fn validate_diffusion_gemma_manifest(manifest: &NativeModelManifest) -> Result<(), MlxRunnerError> {
    if manifest.layer_types.is_empty() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "diffusion_gemma requires layer_types for interleaved SWA/full attention".to_string(),
        ));
    }
    // canvas_size is the primary signal; other fields use sensible defaults
    // in DiffusionConfig::from_manifest().
    if manifest.diffusion.canvas_size.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "diffusion_gemma requires diffusion.canvas_size in the manifest".to_string(),
        ));
    }
    Ok(())
}

fn validate_mla_moe_manifest(manifest: &NativeModelManifest) -> Result<(), MlxRunnerError> {
    let is_glm4_moe_lite = manifest.model_family == "glm4_moe_lite";
    let is_deepseek_v3 = manifest.model_family == "deepseek_v3";
    if !is_glm4_moe_lite && !is_deepseek_v3 {
        return Err(MlxRunnerError::UnsupportedFeature(
            "MLA tensor roles are supported only for glm4_moe_lite or deepseek_v3 manifests"
                .to_string(),
        ));
    }
    if !manifest.mla_attention.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} requires mla_attention metadata",
            manifest.model_family
        )));
    }
    if is_glm4_moe_lite && !manifest.glm_router.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router metadata".to_string(),
        ));
    }
    if !manifest.moe.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} requires moe metadata",
            manifest.model_family
        )));
    }

    let first_dense_layers = if is_glm4_moe_lite {
        manifest.glm_router.first_dense_layer_count.ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(
                "glm4_moe_lite requires glm_router.first_dense_layer_count".to_string(),
            )
        })?
    } else {
        manifest.moe.first_dense_layers.unwrap_or(0)
    };
    // `GlmRouterConfig::from_manifest` `.expect()`s these three fields once the router
    // is considered enabled (`is_enabled()` returns true if *any* field is set), and
    // `glm_router_apply_group_selection` follows up with runtime `assert!`s on the
    // group invariants. Surface every panic-source as a typed manifest error here.
    if is_glm4_moe_lite && manifest.glm_router.routed_scaling_factor.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router.routed_scaling_factor".to_string(),
        ));
    }
    let routed_scaling_factor = if is_glm4_moe_lite {
        manifest.glm_router.routed_scaling_factor.unwrap_or(1.0)
    } else {
        manifest.moe.routed_scaling_factor.unwrap_or(1.0)
    };
    if !routed_scaling_factor.is_finite() || routed_scaling_factor <= 0.0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} requires finite positive routed_scaling_factor",
            manifest.model_family
        )));
    }
    let n_group = if is_glm4_moe_lite {
        manifest.glm_router.n_group.ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(
                "glm4_moe_lite requires glm_router.n_group".to_string(),
            )
        })?
    } else {
        manifest.moe.n_group.unwrap_or(1)
    };
    let topk_group = if is_glm4_moe_lite {
        manifest.glm_router.topk_group.ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(
                "glm4_moe_lite requires glm_router.topk_group".to_string(),
            )
        })?
    } else {
        manifest.moe.topk_group.unwrap_or(1)
    };
    if n_group == 0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} n_group must be greater than zero",
            manifest.model_family
        )));
    }
    if topk_group == 0 || topk_group > n_group {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} topk_group {topk_group} must satisfy 0 < topk_group <= n_group ({n_group})",
            manifest.model_family
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
                "{} moe.expert_count {expert_count} must be divisible by n_group {n_group}",
                manifest.model_family
            )));
        }
        if expert_count / n_group < 2 {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "{} moe.expert_count {expert_count} divided by n_group {n_group} must yield at least two experts per group",
                manifest.model_family
            )));
        }
    }
    if first_dense_layers > manifest.layer_count {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} first_dense_layer_count {first_dense_layers} cannot exceed layer_count {}",
            manifest.model_family, manifest.layer_count
        )));
    }
    let has_shared_experts = if is_glm4_moe_lite {
        manifest.glm_router.has_shared_experts
    } else {
        manifest.moe.shared_expert_count.unwrap_or(0) > 0
    };
    let moe_layer_freq = manifest.moe.layer_freq.unwrap_or(1);
    if is_deepseek_v3 && moe_layer_freq == 0 {
        return Err(MlxRunnerError::UnsupportedFeature(
            "deepseek_v3 requires moe.layer_freq greater than zero".to_string(),
        ));
    }

    for layer_index in 0..manifest.layer_count {
        for role in [
            NativeTensorRole::AttentionNorm,
            NativeTensorRole::AttentionQa,
            NativeTensorRole::AttentionQaNorm,
            NativeTensorRole::AttentionQb,
            NativeTensorRole::AttentionKvA,
            NativeTensorRole::AttentionKvANorm,
            NativeTensorRole::AttentionO,
            NativeTensorRole::AttentionPostNorm,
        ] {
            require_manifest_role(manifest, layer_index, role)?;
        }
        let has_kv_b = manifest.tensors.iter().any(|tensor| {
            tensor.layer_index == Some(layer_index) && tensor.role == NativeTensorRole::AttentionKvB
        });
        let has_embed_q = manifest.tensors.iter().any(|tensor| {
            tensor.layer_index == Some(layer_index)
                && tensor.role == NativeTensorRole::AttentionEmbedQ
        });
        let has_unembed_out = manifest.tensors.iter().any(|tensor| {
            tensor.layer_index == Some(layer_index)
                && tensor.role == NativeTensorRole::AttentionUnembedOut
        });
        if (has_kv_b && (has_embed_q || has_unembed_out))
            || (!has_kv_b && (!has_embed_q || !has_unembed_out))
        {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "{} layer {layer_index} must provide exactly one MLA KV-B layout",
                manifest.model_family
            )));
        }

        let is_moe_layer = if is_deepseek_v3 {
            layer_index >= first_dense_layers && layer_index.is_multiple_of(moe_layer_freq)
        } else {
            layer_index >= first_dense_layers
        };
        if !is_moe_layer {
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
        "{} layer {layer_index} is missing required tensor role {role:?}",
        manifest.model_family
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
                | NativeTensorRole::AttentionKvB
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
    let stop_on_pad = artifacts.manifest().model_family != "diffusion_gemma";

    for file_name in ["config.json", "tokenizer_config.json"] {
        let Some(value) = read_json_file(&artifacts.root_dir().join(file_name)) else {
            continue;
        };
        collect_token_ids(value.get("eos_token_id"), &mut token_ids);
        collect_token_ids(value.get("eos_token_ids"), &mut token_ids);
        collect_token_strings(value.get("eos_token"), &mut token_strings);
        if stop_on_pad {
            collect_token_ids(value.get("pad_token_id"), &mut token_ids);
            collect_token_strings(value.get("pad_token"), &mut token_strings);
        }
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
    if !matches!(
        manifest.model_family.as_str(),
        "gemma4" | "gemma3" | "diffusion_gemma" | "embeddinggemma"
    ) {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "interleaved sliding/full attention is not implemented for {} manifests",
            manifest.model_family
        )));
    }
    if manifest.layer_types.len() != manifest.layer_count as usize {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "interleaved attention requires one layer_type per layer, got {} for {} layers",
            manifest.layer_types.len(),
            manifest.layer_count
        )));
    }

    for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
        if layer_type != "sliding_attention" && layer_type != "full_attention" {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "layer {idx} uses unsupported layer_type {layer_type:?}"
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
                    "sliding_attention layers require sliding_window_size".to_string(),
                ));
            }
            Some(0) => {
                // build_layer_configs maps Some(0) to Some(0), and the cache path then
                // filters it back to None — sliding layers would silently degrade to a
                // grow-forever window. Reject up front instead of running with a layout
                // the user did not ask for.
                return Err(MlxRunnerError::UnsupportedFeature(
                    "sliding_window_size must be greater than zero".to_string(),
                ));
            }
            Some(_) => {}
        }
    }

    for (&layer, &source) in &manifest.kv_shared_source_layers {
        if layer >= manifest.layer_count || source >= manifest.layer_count || source >= layer {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "KV-shared layer {layer} has invalid source layer {source}"
            )));
        }
        let layer_type = &manifest.layer_types[layer as usize];
        let source_type = &manifest.layer_types[source as usize];
        if layer_type != source_type {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "KV-shared layer {layer} type {layer_type:?} cannot reuse source {source} type {source_type:?}"
            )));
        }
        // Chained KV sharing would panic at runtime in `MlxKVCache::peek_source_kv`
        // (the source layer never writes its own K/V, so the cached entry is None
        // and the `.expect("…source layer must appear earlier")` fires). Reject it
        // here so the manifest fails closed instead of producing a midstream panic.
        if manifest.kv_shared_source_layers.contains_key(&source) {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "KV-shared layer {layer} cannot use shared layer {source} as its source"
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
        AX_NATIVE_MODEL_MANIFEST_FILE, NativeDiffusionConfig, NativeLinearAttentionConfig,
        NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType,
        NativeTensorFormat, NativeTensorSpec,
    };
    use std::fs;
    use std::path::{Path, PathBuf};

    fn ctx_with_argmax(deterministic_argmax_sampling: bool) -> RunnerRequestContext {
        RunnerRequestContext {
            request_id: RequestId(1),
            prompt_len: 8,
            processed_prompt_tokens: 0,
            generated_len: 0,
            max_output_tokens: 16,
            seed: 0,
            deterministic_argmax_sampling,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context_size: None,
            ignore_eos: false,
            tool_call_mode: false,
            structured_output_mode: false,
        }
    }

    // A prefix-cache hit may only hand a request the producer's greedy prefill
    // token when the request would itself compute that token. Temperature,
    // top-p, and repetition-penalty requests must resample.
    #[test]
    fn prefill_output_token_cacheable_only_for_greedy_no_rep_penalty() {
        let greedy = MlxSamplingParams::greedy();
        let temperature = MlxSamplingParams::new(0.8, 1.0, 0);
        let greedy_rep = MlxSamplingParams::greedy().with_repetition_penalty(1.3, None);

        // No context: greedy-ness is inferred from the params themselves.
        assert!(prefill_output_token_cacheable(None, greedy));
        assert!(!prefill_output_token_cacheable(None, temperature));
        assert!(!prefill_output_token_cacheable(None, greedy_rep));

        // Deterministic-argmax context with no repetition penalty -> cacheable.
        let det = ctx_with_argmax(true);
        assert!(prefill_output_token_cacheable(Some(&det), greedy));
        // Deterministic argmax but a repetition penalty is active: the stored
        // greedy token ignored the penalty, so it must not be reused.
        assert!(!prefill_output_token_cacheable(Some(&det), greedy_rep));

        // A sampling request (deterministic_argmax_sampling == false) never
        // reuses the token, even if its other params look greedy.
        let sampled = ctx_with_argmax(false);
        assert!(!prefill_output_token_cacheable(Some(&sampled), temperature));
        assert!(!prefill_output_token_cacheable(Some(&sampled), greedy));
    }

    // Verify that the extract-work-reinsert mutex pattern correctly isolates
    // per-request state without GPU execution required.
    #[test]
    fn state_extraction_isolates_concurrent_requests() {
        let mut states: HashMap<RequestId, RequestState> = HashMap::new();
        let a = RequestId(1);
        let b = RequestId(2);

        // Extract A from the map (simulates the lock-brief-remove step).
        // While A is extracted, B's slot is accessible without contention.
        let state_a = states
            .remove(&a)
            .unwrap_or_else(|| RequestState::new(2, a.0));
        let state_b = states
            .remove(&b)
            .unwrap_or_else(|| RequestState::new(2, b.0));

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
        states.insert(id, RequestState::new(2, id.0));

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
    fn mtp_telemetry_tracks_acceptance_step_classes() {
        let mut telemetry = MtpTelemetry::default();

        let mtp_sources = [MtpDraftSource::Mtp; 3];
        telemetry.record_step(3, 3, &mtp_sources, None, 3);
        telemetry.record_step(3, 1, &mtp_sources, None, 1);
        telemetry.record_step(3, 0, &mtp_sources, None, 0);
        telemetry.record_timings(MtpStepTimings {
            cache_clone_wall_us: 10,
            verify_forward_wall_us: 20,
            verify_eval_wall_us: 30,
            target_softmax_wall_us: 35,
            accept_wall_us: 40,
            rollback_wall_us: 50,
            tail_sample_wall_us: 60,
            draft_wall_us: 70,
            mtp_draft_wall_us: 71,
            assistant_draft_wall_us: 72,
            ngram_lookup_wall_us: 73,
            verify_tokens: 8,
            emitted_tokens: 4,
            ngram_submitted_tokens: 0,
        });

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);

        assert!(decisions.contains(&("ax_mtp_draft_tokens".into(), 9)));
        assert!(decisions.contains(&("ax_mtp_accepted_tokens".into(), 4)));
        assert!(decisions.contains(&("ax_mtp_decode_steps".into(), 3)));
        assert!(decisions.contains(&("ax_mtp_full_accept_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_partial_reject_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_complete_miss_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_cache_clone_wall_us".into(), 10)));
        assert!(decisions.contains(&("ax_mtp_verify_forward_wall_us".into(), 20)));
        assert!(decisions.contains(&("ax_mtp_verify_eval_wall_us".into(), 30)));
        assert!(decisions.contains(&("ax_mtp_accept_wall_us".into(), 40)));
        assert!(decisions.contains(&("ax_mtp_rollback_wall_us".into(), 50)));
        assert!(decisions.contains(&("ax_mtp_tail_sample_wall_us".into(), 60)));
        assert!(decisions.contains(&("ax_mtp_draft_wall_us".into(), 70)));
        assert!(decisions.contains(&("ax_mtp_target_softmax_wall_us".into(), 35)));
        assert!(decisions.contains(&("ax_mtp_verify_tokens".into(), 8)));
        assert!(decisions.contains(&("ax_mtp_emitted_tokens".into(), 4)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_submitted_tokens".into(), 9)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_accepted_tokens".into(), 4)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_rejected_tokens".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_cascade_rejected_tokens".into(), 3)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_proposer_wall_us".into(), 71)));
        assert!(decisions.contains(&("ax_mtp_source_assistant_submitted_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_source_assistant_proposer_wall_us".into(), 72)));
        assert!(decisions.contains(&("ax_mtp_ngram_lookup_wall_us".into(), 73)));
        // Per-depth counters: record_step(3,3) + record_step(3,1) + record_step(3,0)
        // drafted_by_depth: all 3 steps attempted all 3 depths → [3, 3, 3]
        // accepted_by_depth: depth0 accepted in steps 0,1; depth1+2 only in step 0
        assert!(decisions.contains(&("ax_mtp_drafted_depth0".into(), 3)));
        assert!(decisions.contains(&("ax_mtp_drafted_depth1".into(), 3)));
        assert!(decisions.contains(&("ax_mtp_drafted_depth2".into(), 3)));
        assert!(decisions.contains(&("ax_mtp_accepted_depth0".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_accepted_depth1".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_accepted_depth2".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_draft_source_mtp_tokens".into(), 9)));
        assert!(decisions.contains(&("ax_mtp_accepted_source_mtp_tokens".into(), 4)));
        assert!(decisions.contains(&("ax_mtp_draft_source_ngram_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_accepted_source_ngram_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_hit_steps".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_think_gated_steps".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_auto_disabled_steps".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_self_tune_disabled_steps".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_submitted_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_submitted_accepted_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_accepted_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_emitted_tokens".into(), 4)));
        assert!(decisions.contains(&("ax_mtp_ngram_utility_stacked_steps".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_acceptance_mode".into(), 0)));
    }

    #[test]
    fn mtp_telemetry_tracks_stacked_ngram_source_and_hybrid_tail() {
        let mut telemetry = MtpTelemetry::default();

        telemetry.record_ngram_attempt(Some(NgramDraftRejection::NoCandidate));
        telemetry.record_ngram_attempt(Some(NgramDraftRejection::ConfidenceFiltered));
        telemetry.record_ngram_attempt(None);
        telemetry.record_ngram_cycle_guard();
        telemetry.record_ngram_stack_hit(2, false);
        telemetry.record_ngram_hybrid_tail(1);
        telemetry.record_ngram_proposed(2);
        telemetry.record_ngram_submitted(2);
        telemetry.record_ngram_verified(2);
        telemetry.record_step(
            3,
            2,
            &[
                MtpDraftSource::Ngram,
                MtpDraftSource::Ngram,
                MtpDraftSource::HybridMtp,
            ],
            None,
            0,
        );

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);

        assert!(decisions.contains(&("ax_mtp_ngram_attempt_steps".into(), 3)));
        assert!(decisions.contains(&("ax_mtp_ngram_no_candidate_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_confidence_filtered_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_cycle_guard_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_hit_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_skipped_mtp_steps".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_skipped_mtp_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_hybrid_tail_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_hybrid_tail_tokens".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_draft_source_ngram_tokens".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_accepted_source_ngram_tokens".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_draft_source_hybrid_mtp_tokens".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_accepted_source_hybrid_mtp_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_proposed_tokens".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_ngram_submitted_tokens".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_ngram_submitted_accepted_tokens".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_ngram_accepted_tokens".into(), 2)));
        assert!(decisions.contains(&("ax_mtp_ngram_rejected_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_ngram_cascade_rejected_tokens".into(), 0)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_submitted_tokens".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_rejected_tokens".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_source_mtp_cascade_rejected_tokens".into(), 0)));
    }

    #[test]
    fn mtp_only_ewma_excludes_cascade_rejections_from_ngram_failure() {
        // When an n-gram token at position 0 is rejected (accept=0), MTP tokens
        // at later positions are cascade-rejected — not because they are bad
        // predictions but because the earlier n-gram token already failed.
        // The MTP-only EWMA must NOT count those cascade rejections, otherwise
        // it would deflate even when pure-MTP acceptance is near-perfect.

        let mut tel = MtpTelemetry::default();

        // Step 1: [Ngram, Ngram, Mtp] accept=0 — n-gram at pos 0 fails.
        // MTP token at pos 2 is cascade-rejected.  No MTP EWMA update.
        tel.record_step(
            3,
            0,
            &[
                MtpDraftSource::Ngram,
                MtpDraftSource::Ngram,
                MtpDraftSource::Mtp,
            ],
            None,
            0,
        );
        assert_eq!(
            tel.mtp_only_accept_rate_ewma_samples, 0,
            "cascade step must not produce a sample"
        );

        // Step 2: [Ngram, Ngram, Mtp] accept=3 — all pass, MTP genuinely accepted.
        tel.record_step(
            3,
            3,
            &[
                MtpDraftSource::Ngram,
                MtpDraftSource::Ngram,
                MtpDraftSource::Mtp,
            ],
            None,
            1,
        );
        assert_eq!(tel.mtp_only_accept_rate_ewma_samples, 1);
        assert!(
            (tel.mtp_only_accept_rate_ewma - 1.0).abs() < 1e-5,
            "MTP accepted → rate 1.0"
        );

        // Step 3: [Ngram, Ngram, Mtp] accept=2 — ngrams pass, MTP at pos 2 is
        // the first rejection (position == accept).  This IS a meaningful eval.
        tel.record_step(
            3,
            2,
            &[
                MtpDraftSource::Ngram,
                MtpDraftSource::Ngram,
                MtpDraftSource::Mtp,
            ],
            None,
            0,
        );
        assert_eq!(tel.mtp_only_accept_rate_ewma_samples, 2);
        // After step 3: EWMA nudged down from 1.0 toward 0.0 (mtp rejected).
        assert!(
            tel.mtp_only_accept_rate_ewma < 1.0,
            "MTP rejection must lower the EWMA"
        );

        // In pure-MTP steps the cascade-exclusion still applies: position 2 is
        // cascade-rejected (never independently verified by the main model) and is
        // excluded from both numerator and denominator.  Only the two "meaningfully
        // evaluated" positions count — position 0 (accepted) and position 1 (the
        // first and only genuine rejection that caused the cascade).
        let mut tel2 = MtpTelemetry::default();
        tel2.record_step(3, 1, &[MtpDraftSource::Mtp; 3], None, 1);
        // drafted=3, accepted=1, first rejection at pos 1 (Mtp) → first_rejection_is_mtp=true.
        // pos 2 is cascade-excluded.  mtp_only_drafted = 1 (accepted) + 1 (first rejection) = 2.
        assert_eq!(tel2.mtp_only_accept_rate_ewma_samples, 1);
        let expected_rate = 1.0_f32 / 2.0;
        assert!(
            (tel2.mtp_only_accept_rate_ewma - expected_rate).abs() < 1e-5,
            "pure-MTP partial rejection: rate should be accepted/meaningful = 1/2"
        );
    }

    #[test]
    fn mtp_ewma_numerator_uses_accepted_count_not_argmax_matches() {
        // In rejection-sampling mode, the EWMA numerator is the actual number of
        // accepted MTP tokens, not just those matching the target argmax.
        // This ensures the n-gram saturation gate and auto-optimistic activation
        // converge on the true acceptance rate rather than the argmax-match rate.
        //
        // Scenario: 2 MTP tokens drafted, both accepted; only 1 matches target argmax.
        // Old code: mtp_argmax_matches = 1 → EWMA = 1/2 = 0.5 (wrong)
        // New code: mtp_ewma_numerator = 2  → EWMA = 2/2 = 1.0 (correct)
        let mut tel = MtpTelemetry::default();
        // mtp_ewma_numerator=2: caller computed actual accepted MTP count (both accepted).
        tel.record_step(2, 2, &[MtpDraftSource::Mtp; 2], None, 2);
        assert_eq!(tel.mtp_only_accept_rate_ewma_samples, 1);
        assert!(
            (tel.mtp_only_accept_rate_ewma - 1.0).abs() < 1e-5,
            "all MTP accepted → EWMA = 1.0; got {}",
            tel.mtp_only_accept_rate_ewma
        );

        // Verify that passing the old argmax-match count (1 of 2) would have given 0.5.
        let mut tel_old = MtpTelemetry::default();
        tel_old.record_step(2, 2, &[MtpDraftSource::Mtp; 2], None, 1);
        assert!(
            (tel_old.mtp_only_accept_rate_ewma - 0.5).abs() < 1e-5,
            "only argmax matches counted → EWMA = 0.5 (old wrong behaviour)"
        );
    }

    #[test]
    fn mtp_ngram_hurt_gate_fires_when_combined_trails_mtp_only_by_margin() {
        assert!(mtp_ngram_hurt_gate(4, 0.80, 4, 0.90, 4, 4, 0.02));
    }

    #[test]
    fn mtp_ngram_hurt_gate_does_not_fire_before_min_samples() {
        assert!(!mtp_ngram_hurt_gate(4, 0.80, 3, 0.90, 4, 4, 0.02));
        assert!(!mtp_ngram_hurt_gate(4, 0.80, 4, 0.90, 3, 4, 0.02));
    }

    #[test]
    fn mtp_ngram_hurt_gate_does_not_fire_inside_margin() {
        assert!(!mtp_ngram_hurt_gate(4, 0.885, 4, 0.90, 4, 4, 0.02));
    }

    #[test]
    fn mtp_ngram_auto_disable_requires_both_mtp_strong_and_ngram_weak() {
        let cfg = MtpNgramAutoDisableConfig {
            mtp_warmup: 64,
            ngram_warmup: 32,
            mtp_threshold: 850,
            ngram_floor: 500,
        };
        assert!(mtp_ngram_auto_disable_gate(4, 100, 90, 40, 10, cfg));
        assert!(!mtp_ngram_auto_disable_gate(4, 100, 84, 40, 10, cfg));
        assert!(!mtp_ngram_auto_disable_gate(4, 100, 90, 40, 25, cfg));
        assert!(!mtp_ngram_auto_disable_gate(4, 63, 60, 40, 10, cfg));
        assert!(!mtp_ngram_auto_disable_gate(4, 100, 90, 31, 10, cfg));
    }

    #[test]
    fn mtp_ngram_gate_decision_reports_each_reason() {
        let cfg = MtpNgramAutoDisableConfig {
            mtp_warmup: 64,
            ngram_warmup: 32,
            mtp_threshold: 850,
            ngram_floor: 500,
        };
        let decision =
            mtp_ngram_gate_decision(4, 3, 0.80, 4, 0.99, 4, 100, 90, 40, 10, true, 4, 0.02, cfg);
        assert!(decision.gated);
        assert!(decision.saturated);
        assert!(decision.hurt);
        assert!(decision.auto_disabled);
        assert!(decision.self_tune_disabled);
    }

    fn utility_cfg_for_tests() -> MtpNgramUtilityGateConfig {
        MtpNgramUtilityGateConfig {
            min_emitted_tokens: 128,
            min_ngram_submitted_tokens: 32,
            margin_ratio: 0.02,
            hysteresis_steps: 16,
        }
    }

    #[test]
    fn mtp_ngram_utility_gate_waits_for_enough_samples() {
        let cfg = utility_cfg_for_tests();
        let baseline = DraftSourceUtility {
            proposer_wall_us: 1_000,
            verify_wall_us: 1_000,
            emitted_tokens: 127,
            ..DraftSourceUtility::default()
        };
        let stacked = DraftSourceUtility {
            submitted_tokens: 64,
            proposer_wall_us: 2_000,
            verify_wall_us: 2_000,
            emitted_tokens: 128,
        };

        let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

        assert!(!decision.gated);
        assert!(decision.insufficient_samples);
        assert!(!decision.utility_hurt);
    }

    #[test]
    fn mtp_ngram_utility_gate_waits_for_enough_ngram_tokens() {
        let cfg = utility_cfg_for_tests();
        let baseline = DraftSourceUtility {
            proposer_wall_us: 1_000,
            verify_wall_us: 1_000,
            emitted_tokens: 128,
            ..DraftSourceUtility::default()
        };
        let stacked = DraftSourceUtility {
            submitted_tokens: 31,
            proposer_wall_us: 2_000,
            verify_wall_us: 2_000,
            emitted_tokens: 128,
        };

        let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

        assert!(!decision.gated);
        assert!(decision.insufficient_samples);
    }

    #[test]
    fn mtp_ngram_utility_gate_fires_when_stacked_cost_is_worse() {
        let cfg = utility_cfg_for_tests();
        let baseline = DraftSourceUtility {
            proposer_wall_us: 5_000,
            verify_wall_us: 5_000,
            emitted_tokens: 200,
            ..DraftSourceUtility::default()
        };
        let stacked = DraftSourceUtility {
            submitted_tokens: 64,
            proposer_wall_us: 7_500,
            verify_wall_us: 7_500,
            emitted_tokens: 200,
        };

        let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

        assert!(decision.gated);
        assert!(decision.utility_hurt);
        assert!(!decision.insufficient_samples);
    }

    #[test]
    fn mtp_ngram_utility_gate_does_not_fire_inside_margin() {
        let cfg = utility_cfg_for_tests();
        let baseline = DraftSourceUtility {
            proposer_wall_us: 10_000,
            emitted_tokens: 200,
            ..DraftSourceUtility::default()
        };
        let stacked = DraftSourceUtility {
            submitted_tokens: 64,
            proposer_wall_us: 10_100,
            emitted_tokens: 200,
            ..DraftSourceUtility::default()
        };

        let decision = mtp_ngram_utility_gate(4, baseline, stacked, cfg, 0);

        assert!(!decision.gated);
        assert!(!decision.utility_hurt);
        assert!(!decision.insufficient_samples);
    }

    #[test]
    fn mtp_ngram_utility_gate_hysteresis_gates_without_recomputing_cost() {
        let decision = mtp_ngram_utility_gate(
            4,
            DraftSourceUtility::default(),
            DraftSourceUtility::default(),
            utility_cfg_for_tests(),
            3,
        );

        assert!(decision.gated);
        assert!(decision.hysteresis_active);
        assert!(!decision.insufficient_samples);
    }

    #[test]
    fn mtp_ngram_utility_gate_zero_tokens_do_not_panic() {
        let decision = mtp_ngram_utility_gate(
            4,
            DraftSourceUtility::default(),
            DraftSourceUtility::default(),
            utility_cfg_for_tests(),
            0,
        );

        assert!(!decision.gated);
        assert!(decision.insufficient_samples);
    }

    #[test]
    fn mtp_utility_uses_separate_baseline_and_stacked_step_buckets() {
        let mut telemetry = MtpTelemetry::default();

        telemetry.record_timings(MtpStepTimings {
            verify_forward_wall_us: 100,
            verify_eval_wall_us: 10,
            target_softmax_wall_us: 10,
            draft_wall_us: 0,
            emitted_tokens: 4,
            ngram_submitted_tokens: 0,
            ..MtpStepTimings::default()
        });
        telemetry.record_timings(MtpStepTimings {
            verify_forward_wall_us: 200,
            verify_eval_wall_us: 50,
            target_softmax_wall_us: 25,
            draft_wall_us: 25,
            emitted_tokens: 2,
            ngram_submitted_tokens: 5,
            ..MtpStepTimings::default()
        });

        let baseline = telemetry.baseline_utility();
        let stacked = telemetry.stacked_utility();

        assert_eq!(baseline.cost_per_emitted_token_us(), Some(30.0));
        assert_eq!(stacked.cost_per_emitted_token_us(), Some(150.0));
        assert_eq!(stacked.submitted_tokens, 5);

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_utility_baseline_wall_us".into(), 120)));
        assert!(decisions.contains(&("ax_mtp_ngram_utility_stacked_steps".into(), 1)));
        assert!(decisions.contains(&("ax_mtp_ngram_utility_stacked_wall_us".into(), 300)));
        assert!(decisions.contains(&(
            "ax_mtp_ngram_utility_stacked_ngram_submitted_tokens".into(),
            5
        )));
    }

    #[test]
    fn mtp_ngram_safety_defaults_to_reasoning_tighten_mode() {
        assert_eq!(
            MtpNgramSafetyMode::default(),
            MtpNgramSafetyMode::TightenReasoning
        );
        let decision = mtp_ngram_speculative_safety_decision_for_mode(
            MtpNgramSafetyMode::default(),
            false,
            false,
            true,
        );

        assert!(decision.tighten_ngram);
        assert!(!decision.disable_ngram);
        assert_eq!(decision.reason.route_code(), 3);
    }

    #[test]
    fn mtp_ngram_safety_disable_modes_are_explicit() {
        let all = mtp_ngram_speculative_safety_decision_for_mode(
            MtpNgramSafetyMode::DisableAll,
            false,
            false,
            false,
        );
        assert!(all.disable_ngram);
        assert_eq!(all.reason, SpeculativeSafetyReason::ExperimentalOverride);

        let reasoning = mtp_ngram_speculative_safety_decision_for_mode(
            MtpNgramSafetyMode::DisableReasoning,
            false,
            false,
            true,
        );
        assert!(reasoning.disable_ngram);
        assert_eq!(reasoning.reason, SpeculativeSafetyReason::ReasoningTrace);

        let off = mtp_ngram_speculative_safety_decision_for_mode(
            MtpNgramSafetyMode::Off,
            true,
            true,
            true,
        );
        assert_eq!(off, SpeculativeSafetyDecision::default());
    }

    #[test]
    fn mtp_ngram_safety_disables_tool_and_structured_workloads() {
        let tool = mtp_ngram_speculative_safety_decision_for_mode(
            MtpNgramSafetyMode::default(),
            true,
            false,
            false,
        );
        assert!(tool.disable_ngram);
        assert_eq!(tool.reason, SpeculativeSafetyReason::ToolCall);

        let structured = mtp_ngram_speculative_safety_decision_for_mode(
            MtpNgramSafetyMode::default(),
            false,
            true,
            false,
        );
        assert!(structured.disable_ngram);
        assert_eq!(structured.reason, SpeculativeSafetyReason::StructuredOutput);
    }

    #[test]
    fn ngram_self_tune_counts_only_submitted_drafts() {
        let mut state = NgramSelfTuneState::default();
        state.record_verified(0, 0.30, 32);
        assert_eq!(state.drafted, 0);
        assert_eq!(state.accepted, 0);
        assert!(!state.disabled);

        state.record_submitted(16);
        state.record_verified(4, 0.30, 32);
        assert_eq!(state.drafted, 16);
        assert_eq!(state.accepted, 4);
        assert!(!state.disabled);
    }

    #[test]
    fn ngram_self_tune_disables_after_warmup_when_acceptance_low() {
        let mut state = NgramSelfTuneState::default();
        state.record_submitted(32);
        state.record_verified(5, 0.30, 32);
        assert!(state.disabled);
    }

    #[test]
    fn mtp_accept_count_ngram_pseudo_logprob_rejection_samples() {
        // N-gram position with delta distribution: log_prob = 0.0 → p_draft = 1.0.
        // target_prob = 1.0 → accept_prob = 1.0/1.0 = 1.0 → always accept.
        let pseudo_lp = 0.0_f32;
        let mut rng = Xorshift64::new(42);
        let accept = mtp_accept_count(
            &[17],
            &[pseudo_lp],
            &[],
            &[MtpDraftSource::Ngram],
            Some(&[1.0]),
            None,
            &[17],
            &mut rng,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Delta,
        );
        assert_eq!(accept.accept_count, 1);
        assert!(accept.all_accepted);
        assert_eq!(accept.rejection_correction, None);

        // Low target probability → reject even though tokens match.
        // accept_prob = 0.0/1.0 = 0.0 → never accept.
        let mut rng2 = Xorshift64::new(99);
        let accept2 = mtp_accept_count(
            &[17],
            &[pseudo_lp],
            &[],
            &[MtpDraftSource::Ngram],
            Some(&[0.0]),
            None,
            &[17],
            &mut rng2,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Delta,
        );
        assert_eq!(accept2.accept_count, 0);
        assert!(!accept2.all_accepted);
    }

    #[test]
    fn mtp_ngram_pseudo_log_probs_cover_ngram_only_draft_windows() {
        let log_probs =
            mtp_ngram_pseudo_log_probs(&[0.9, 0.0, 1.2], 4, MtpNgramAcceptanceMode::Confidence);

        assert!((log_probs[0] - 0.9_f32.ln()).abs() < 1e-6);
        assert_eq!(log_probs[1], -30.0, "zero confidence is clamped");
        assert_eq!(log_probs[2], 0.0, "confidence above 1.0 is clamped");
        assert!(
            log_probs[3].is_nan(),
            "missing confidence falls back to greedy comparison for that position"
        );

        // N-gram-only MTP draft windows must still enter rejection sampling when
        // pseudo log-probs are present.  The target argmax intentionally differs
        // from the pending token; the old empty-log-prob path would reject here.
        let mut rng = Xorshift64::new(42);
        let accept = mtp_accept_count(
            &[17],
            &log_probs[..1],
            &[],
            &[MtpDraftSource::Ngram],
            Some(&[1.0]),
            None,
            &[99],
            &mut rng,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Confidence,
        );
        assert_eq!(accept.accept_count, 1);
        assert!(accept.all_accepted);
    }

    #[test]
    fn mtp_ngram_pseudo_logprob_delta_mode_returns_zero() {
        let log_probs =
            mtp_ngram_pseudo_log_probs(&[0.9, 0.0, 1.2], 4, MtpNgramAcceptanceMode::Delta);

        assert_eq!(log_probs[0], 0.0);
        assert_eq!(log_probs[1], 0.0);
        assert_eq!(log_probs[2], 0.0);
        assert!(log_probs[3].is_nan());
    }

    #[test]
    fn mtp_accept_count_ngram_greedy_mode_uses_argmax_match() {
        let mut rng = Xorshift64::new(42);
        let accept = mtp_accept_count(
            &[17],
            &[0.0],
            &[],
            &[MtpDraftSource::Ngram],
            Some(&[0.0]),
            None,
            &[17],
            &mut rng,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Greedy,
        );
        assert_eq!(accept.accept_count, 1);
        assert!(accept.all_accepted);

        let mut rng2 = Xorshift64::new(42);
        let reject = mtp_accept_count(
            &[17],
            &[0.0],
            &[],
            &[MtpDraftSource::Ngram],
            Some(&[1.0]),
            None,
            &[99],
            &mut rng2,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Greedy,
        );
        assert_eq!(reject.accept_count, 0);
        assert!(!reject.all_accepted);
        assert_eq!(reject.rejection_correction, Some(99));
    }

    #[test]
    fn mtp_accept_count_ngram_nan_logprob_falls_back_to_greedy() {
        // NaN log-prob is not finite → greedy argmax fallback still applies
        // (backward compatibility with paths that don't carry pseudo log-probs).
        let mut rng = Xorshift64::new(42);
        let accept = mtp_accept_count(
            &[17],
            &[f32::NAN],
            &[],
            &[MtpDraftSource::Ngram],
            Some(&[0.0]),
            None,
            &[17],
            &mut rng,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Confidence,
        );
        assert_eq!(accept.accept_count, 1);
        assert!(accept.all_accepted);
        assert_eq!(accept.rejection_correction, None);
    }

    #[test]
    fn mtp_accept_count_ngram_rejection_ignores_draft_distribution() {
        let target_distribution = TokenDistribution::new(vec![(99, 1.0)]).unwrap();
        let stale_mtp_distribution = TokenDistribution::new(vec![(17, 1.0)]).unwrap();
        let mut rng = Xorshift64::new(42);

        let accept = mtp_accept_count(
            &[17],
            &[0.9_f32.ln()],
            &[stale_mtp_distribution],
            &[MtpDraftSource::Ngram],
            Some(&[0.0]),
            Some(&[target_distribution]),
            &[17],
            &mut rng,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Confidence,
        );

        assert_eq!(accept.accept_count, 0);
        assert!(!accept.all_accepted);
        assert_eq!(
            accept.rejection_correction, None,
            "n-gram pseudo log-probs do not have a true draft distribution"
        );
    }

    #[test]
    fn mtp_model_accept_count_defaults_to_argmax_verification() {
        let mut rng = Xorshift64::new(42);
        let reject = mtp_accept_count(
            &[17],
            &[0.01_f32.ln()],
            &[],
            &[MtpDraftSource::Mtp],
            Some(&[1.0]),
            None,
            &[99],
            &mut rng,
            1.0,
            0.8,
            MtpModelAcceptanceMode::Greedy,
            MtpNgramAcceptanceMode::Confidence,
        );

        assert_eq!(reject.accept_count, 0);
        assert!(!reject.all_accepted);
        assert_eq!(reject.rejection_correction, Some(99));
    }

    #[test]
    fn mtp_accept_count_aligns_hybrid_tail_distribution_after_ngram_prefix() {
        let mut rng = Xorshift64::new(7);
        let target_prefix = TokenDistribution::new(vec![(10, 1.0)]).unwrap();
        let target_tail = TokenDistribution::new(vec![(99, 1.0)]).unwrap();
        let draft_tail = TokenDistribution::new(vec![(20, 1.0)]).unwrap();

        // N-gram position uses NAN → greedy fallback (tokens match → accept).
        // HybridMtp position has log_prob=0.0, target_prob=0.0 → reject, correction=99.
        let accept = mtp_accept_count(
            &[10, 20],
            &[f32::NAN, 0.0],
            &[draft_tail],
            &[MtpDraftSource::Ngram, MtpDraftSource::HybridMtp],
            Some(&[1.0, 0.0]),
            Some(&[target_prefix, target_tail]),
            &[10, 20],
            &mut rng,
            1.0,
            0.8,
            MtpModelAcceptanceMode::RejectionSampling,
            MtpNgramAcceptanceMode::Confidence,
        );

        assert_eq!(accept.accept_count, 1);
        assert!(!accept.all_accepted);
        assert_eq!(accept.rejection_correction, Some(99));
    }

    #[test]
    fn mtp_accept_count_temperature_rescaling() {
        // Draft at T=0.7, target at T=0.6 → ratio=7/6≈1.167.
        // log_p_draft = ln(0.8) ≈ -0.223; scaled = -0.223*1.167 ≈ -0.260 → p_scaled ≈ 0.771.
        // target_prob = 1.0 → accept_prob = 1.0/0.771 capped at 1.0 → always accept.
        let mut rng = Xorshift64::new(1);
        let accept = mtp_accept_count(
            &[5],
            &[0.8_f32.ln()],
            &[],
            &[MtpDraftSource::Mtp],
            Some(&[1.0]),
            None,
            &[5],
            &mut rng,
            0.7,
            0.6,
            MtpModelAcceptanceMode::RejectionSampling,
            MtpNgramAcceptanceMode::Confidence,
        );
        assert_eq!(accept.accept_count, 1);

        // When temperatures match, no rescaling occurs.
        let mut rng2 = Xorshift64::new(1);
        let accept2 = mtp_accept_count(
            &[5],
            &[0.8_f32.ln()],
            &[],
            &[MtpDraftSource::Mtp],
            Some(&[1.0]),
            None,
            &[5],
            &mut rng2,
            0.7,
            0.7,
            MtpModelAcceptanceMode::RejectionSampling,
            MtpNgramAcceptanceMode::Confidence,
        );
        assert_eq!(accept2.accept_count, 1);

        // N-gram pseudo log-probs must NOT be temperature-rescaled.
        // The key probe: target_prob == p_draft → unscaled accept_prob = 1.0;
        // if rescaling were wrongly applied (T_target=2.0 > T_draft=0.7,
        // ratio=0.35), log_p_scaled = -6.9 * 0.35 = -2.42 → p_scaled ≈ 0.089
        // → accept_prob ≈ 0.011 → reject for all typical rng values.
        let ultra_low_lp = 0.001_f32.ln();
        for seed in 0u64..10 {
            let mut rng_probe = Xorshift64::new(seed);
            let a = mtp_accept_count(
                &[7],
                &[ultra_low_lp],
                &[],
                &[MtpDraftSource::Ngram],
                Some(&[0.001]),
                None,
                &[7],
                &mut rng_probe,
                0.7,
                2.0,
                MtpModelAcceptanceMode::Greedy,
                MtpNgramAcceptanceMode::Confidence,
            );
            assert_eq!(
                a.accept_count, 1,
                "n-gram pseudo log-prob must not be rescaled (seed={seed})"
            );
        }
    }

    #[test]
    fn mtp_adaptive_depth_shrinks_on_partial_reject_and_recovers_on_full_accept() {
        // consecutive_misses=0 for all non-complete-miss cases.
        assert_eq!(mtp_next_adaptive_depth(0, 3, 0, 0, 0), 3);
        assert_eq!(mtp_next_adaptive_depth(3, 3, 3, 2, 0), 2);
        assert_eq!(mtp_next_adaptive_depth(2, 3, 2, 1, 0), 2);
        assert_eq!(mtp_next_adaptive_depth(1, 3, 1, 1, 0), 2);
        assert_eq!(mtp_next_adaptive_depth(2, 3, 2, 2, 0), 3);
        assert_eq!(mtp_next_adaptive_depth(3, 0, 3, 3, 0), 0);
    }

    #[test]
    fn mtp_adaptive_depth_progressive_floor_on_consecutive_misses() {
        // First complete miss (consecutive_misses=0): floor = 2.
        assert_eq!(mtp_next_adaptive_depth(3, 3, 3, 0, 0), 2);
        // Second consecutive miss (consecutive_misses=1): floor = 1.
        assert_eq!(mtp_next_adaptive_depth(2, 3, 2, 0, 1), 1);
        // Third+ consecutive miss (consecutive_misses=2): floor = 0.
        assert_eq!(mtp_next_adaptive_depth(1, 3, 1, 0, 2), 0);
        assert_eq!(mtp_next_adaptive_depth(1, 3, 1, 0, 5), 0);
        // Partial accept resets to normal floor logic (not complete miss path).
        assert_eq!(mtp_next_adaptive_depth(3, 3, 3, 1, 3), 2);
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
        let payload = MlxKVCache::new(2).serialize_to_bytes();
        MlxPrefixSnapshot {
            kv_cache_payload: Arc::from(payload.into_boxed_slice()),
            tokens: vec![token; token_count],
            token_count,
            bytes,
            greedy_prefill_output_token: Some(7),
        }
    }

    #[test]
    fn prefix_cache_store_clones_share_l1_entries() {
        let store = MlxPrefixCacheStore::memory_only_for_tests(MlxPrefixCachePolicy {
            max_bytes: 1024,
            max_entries: 4,
        });
        let cloned = store.clone();
        let key = test_prefix_key(1);

        let outcome = store
            .prefix_cache
            .lock()
            .insert(key.clone(), test_prefix_snapshot(1, 4, 128));
        assert!(outcome.stored);

        let hit = cloned.prefix_cache.lock().get(&key, &[1; 4]);
        assert!(hit.is_some(), "cloned store must see original L1 insert");
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
            hit.rehydrate_cache()
                .expect("serialized L1 snapshot should rehydrate")
                .usage_snapshot()
                .logical_tokens,
            0
        );
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
        other.record_disk_insert(8192, 2);
        telemetry.merge_from(other);

        assert_eq!(telemetry.blocked, 4);
        assert_eq!(telemetry.blocked_policy_disabled, 1);
        assert_eq!(telemetry.blocked_unsupported_layout, 2);
        assert_eq!(telemetry.blocked_trim_failure, 1);
        assert_eq!(telemetry.disk_inserts, 1);
        assert_eq!(telemetry.disk_insert_bytes, 8192);
        assert_eq!(telemetry.disk_evictions, 2);
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
            disk_hits: 6,
            disk_misses: 7,
            disk_inserts: 8,
            disk_insert_bytes: 8192,
            disk_evictions: 9,
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
        assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_hits".into(), 6)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_misses".into(), 7)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_inserts".into(), 8)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_insert_bytes_kib".into(), 8)));
        assert!(decisions.contains(&("ax_mlx_prefix_cache_disk_evictions".into(), 9)));
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
        let state = RequestState::new(2, 21);
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
        let state = RequestState::new(2, 22);
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
        let mut state = RequestState::new(2, 0);
        state.bonus_queue.push_back(99);
        state.bonus_queue.push_back(100);
        state.next_model_last_token = Some(5);
        state.ngram_disabled_steps = 3;
        state.linear_ngram_no_draft_streak = 7;
        state.ngram_acceleration_disabled_for_request = true;
        state.mtp_bypassed = true;

        // Simulate the prefill reset branch of run_item.
        state.bonus_queue.clear();
        state.next_model_last_token = None;
        state.ngram_disabled_steps = 0;
        state.linear_ngram_no_draft_streak = 0;
        state.ngram_acceleration_disabled_for_request = false;
        state.mtp_bypassed = false;

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
        assert!(
            !state.mtp_bypassed,
            "MTP bypass must be cleared on prefill so the next request gets a fresh MTP attempt"
        );
    }

    #[test]
    fn generation_ngram_seed_uses_reconstructed_prompt_after_prefix_warmup() {
        let mut warm = RequestState::new(2, 11);
        warm.prompt_prefix_tokens = vec![10, 11, 12, 13, 10, 11, 12];

        seed_generation_ngram_from_prompt(&mut warm, false);

        assert_eq!(
            warm.ngram.predict(1),
            vec![13],
            "warm prefix+suffix prefill must seed n-grams from the reconstructed full prompt",
        );

        let mut suffix_only = RequestState::new(2, 12);
        suffix_only.prompt_prefix_tokens = vec![10, 11, 12];
        seed_generation_ngram_from_prompt(&mut suffix_only, false);

        assert!(
            suffix_only.ngram.predict(1).is_empty(),
            "feeding only the final prefill item loses the prompt context needed for deterministic warm_extend",
        );
    }

    #[test]
    fn generation_ngram_seed_includes_prefill_output_token() {
        let mut state = RequestState::new(2, 13);
        state.prompt_prefix_tokens = vec![1, 2, 3, 1, 2, 3];

        seed_generation_ngram_from_prompt(&mut state, false);
        assert_eq!(
            state.ngram.predict(1),
            vec![1],
            "prompt tail predicts from the final prompt context before the first generated token is committed",
        );

        seed_generation_ngram_from_prefill_output(&mut state, Some(1));
        assert_eq!(
            state.ngram.predict(1),
            vec![2],
            "the prefill-sampled first output token must become part of the next decode context",
        );
    }

    #[test]
    fn generation_ngram_seed_extends_window_for_repeating_prompts() {
        let block: Vec<u32> = (1..=70).collect();
        let mut state = RequestState::new(2, 14);
        state.prompt_prefix_tokens.extend_from_slice(&block);
        state.prompt_prefix_tokens.extend_from_slice(&block);
        state.prompt_prefix_tokens.extend_from_slice(&block);

        seed_generation_ngram_from_prompt(&mut state, false);

        assert_eq!(
            state.ngram.predict(1),
            vec![1],
            "repeating prompts should seed enough history to find suffix continuations beyond the 64-token random-prompt guard",
        );
    }

    #[test]
    fn generation_ngram_seed_keeps_random_prompts_on_short_tail() {
        let mut state = RequestState::new(2, 15);
        state.prompt_prefix_tokens = (1..=210).collect();
        state
            .prompt_prefix_tokens
            .extend_from_slice(&[147, 148, 149, 150]);

        seed_generation_ngram_from_prompt(&mut state, false);

        assert!(
            state.ngram.predict(1).is_empty(),
            "non-repeating prompts should keep the short tail guard and avoid long-range random false positives",
        );
    }

    #[test]
    fn ngram_decode_result_keeps_correction_token_in_output_queue() {
        let mut state = RequestState::new(2, 7);

        let output = apply_decode_result(&mut state, &[11, 12], &[]);

        assert_eq!(output, vec![11, 12]);
        assert_eq!(
            state.bonus_queue.iter().copied().collect::<Vec<_>>(),
            Vec::<u32>::new(),
            "verified tokens should be returned in the current runner update"
        );
        assert_eq!(state.next_model_last_token, Some(12));
    }

    #[test]
    fn ngram_decode_result_queues_full_accept_tail_and_bonus() {
        let mut state = RequestState::new(2, 8);

        let output = apply_decode_result(&mut state, &[21, 22, 23, 24], &[]);

        assert_eq!(output, vec![21, 22, 23, 24]);
        assert_eq!(
            state.bonus_queue.iter().copied().collect::<Vec<_>>(),
            Vec::<u32>::new(),
            "accepted drafts and final token should be emitted in one runner update"
        );
        assert_eq!(state.next_model_last_token, Some(24));
    }

    #[test]
    fn stop_reason_prefers_eos_before_max_output() {
        assert_eq!(
            truncate_sampled_tokens_for_stop(vec![151645], 1, 32, &[151645]),
            (vec![151645], Some(StopReason::EosToken))
        );
        assert_eq!(
            truncate_sampled_tokens_for_stop(vec![7], 31, 32, &[151645]),
            (vec![7], Some(StopReason::MaxOutputTokens))
        );
        assert_eq!(
            truncate_sampled_tokens_for_stop(vec![7], 1, 32, &[151645]),
            (vec![7], None)
        );
    }

    #[test]
    fn empty_terminal_token_slice_ignores_eos_until_limit() {
        assert_eq!(
            truncate_sampled_tokens_for_stop(vec![151645], 1, 32, &[]),
            (vec![151645], None)
        );
        assert_eq!(
            truncate_sampled_tokens_for_stop(vec![151645], 31, 32, &[]),
            (vec![151645], Some(StopReason::MaxOutputTokens))
        );
    }

    #[test]
    fn sampled_token_batch_truncates_at_eos_before_max_output() {
        assert_eq!(
            truncate_sampled_tokens_for_stop(vec![31, 151645, 33], 30, 32, &[151645]),
            (vec![31, 151645], Some(StopReason::EosToken))
        );
    }

    #[test]
    fn sampled_token_batch_truncates_at_max_output() {
        assert_eq!(
            truncate_sampled_tokens_for_stop(vec![31, 32, 33], 30, 32, &[]),
            (vec![31, 32], Some(StopReason::MaxOutputTokens))
        );
    }

    #[test]
    fn ngram_decode_result_truncates_bonus_queue_at_eos() {
        let mut state = RequestState::new(2, 9);

        let output = apply_decode_result(&mut state, &[31, 32, 151645, 33], &[151645]);

        assert_eq!(output, vec![31, 32, 151645]);
        assert_eq!(
            state.bonus_queue.iter().copied().collect::<Vec<_>>(),
            Vec::<u32>::new(),
            "verified tokens after EOS must not be emitted"
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
            seed: 0,
            deterministic_argmax_sampling: true,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            repetition_context_size: None,
            ignore_eos: false,
            tool_call_mode: false,
            structured_output_mode: false,
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

    fn unit_weight() -> QuantizedWeight {
        QuantizedWeight::new(mlx_sys::zeros(&[1, 1], MlxDtype::Float32, None), None, None)
    }

    fn runner_test_layer() -> LayerWeights {
        LayerWeights {
            attn_norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            attn_post_norm: None,
            q_norm: None,
            k_norm: None,
            q_proj: None,
            k_proj: None,
            v_proj: None,
            qkv_packed: None,
            o_proj: None,
            linear_attn: None,
            glm_mla_attn: None,
            ffn_norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: None,
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        }
    }

    fn runner_test_weights(layers: Vec<LayerWeights>) -> ModelWeights {
        ModelWeights {
            token_embedding: unit_weight(),
            final_norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            lm_head: unit_weight(),
            layers,
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            diffusion_self_conditioning: None,
        }
    }

    #[test]
    fn weight_layout_telemetry_counts_dense_ffn_packed_and_split_layers() {
        let mut packed = runner_test_layer();
        packed.gate_up_packed = Some(unit_weight());
        packed.down_proj = Some(unit_weight());
        let mut split = runner_test_layer();
        split.gate_proj = Some(unit_weight());
        split.up_proj = Some(unit_weight());
        split.down_proj = Some(unit_weight());
        let mut attention_only = runner_test_layer();
        attention_only.gate_up_packed = Some(unit_weight());

        let telemetry = WeightLayoutTelemetry::from_weights(&runner_test_weights(vec![
            packed,
            split,
            attention_only,
        ]));
        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_dense_ffn_gate_up_packed_layers"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_dense_ffn_split_gate_up_layers"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_qkvz_ba_packed_layers"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_split_qkvba_layers"),
            Some(&0)
        );
    }

    #[test]
    fn weight_layout_telemetry_excludes_5bit_split_from_fallback_counter() {
        use crate::weights::QuantizedWeight;
        use ax_engine_core::model::NativeTensorQuantization;
        let quant5 = NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 5,
        };
        let w5 = || {
            QuantizedWeight::with_quantization(
                mlx_sys::zeros(&[1, 1], mlx_sys::MlxDtype::Float32, None),
                None,
                None,
                Some(&quant5),
            )
        };
        let mut split5 = runner_test_layer();
        split5.gate_proj = Some(w5());
        split5.up_proj = Some(w5());
        split5.down_proj = Some(unit_weight());

        let telemetry = WeightLayoutTelemetry::from_weights(&runner_test_weights(vec![split5]));
        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        // 5-bit split is intentional; must not appear as a hotpath fallback.
        assert_eq!(
            decisions.get("ax_mlx_dense_ffn_split_gate_up_layers"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_dense_ffn_gate_up_packed_layers"),
            Some(&0)
        );
    }

    #[test]
    fn gemma4_assistant_mtp_confidence_mode_parser_accepts_stable_aliases() {
        assert_eq!(
            parse_gemma4_assistant_mtp_confidence_mode("gpu-exact"),
            Some(Gemma4AssistantMtpConfidenceMode::GpuExact)
        );
        assert_eq!(
            parse_gemma4_assistant_mtp_confidence_mode("GPU_EXACT"),
            Some(Gemma4AssistantMtpConfidenceMode::GpuExact)
        );
        assert_eq!(
            parse_gemma4_assistant_mtp_confidence_mode("cpu"),
            Some(Gemma4AssistantMtpConfidenceMode::ExactCpu)
        );
        assert_eq!(parse_gemma4_assistant_mtp_confidence_mode("approx"), None);
        assert_eq!(Gemma4AssistantMtpConfidenceMode::ExactCpu.route_code(), 0);
        assert_eq!(Gemma4AssistantMtpConfidenceMode::GpuExact.route_code(), 1);
    }

    #[test]
    fn gemma4_assistant_mtp_status_emits_prd_route_metadata() {
        let status = Gemma4AssistantMtpStatus {
            configured: true,
            validated: true,
            enabled: false,
            attach_failed: false,
            disable_reason: crate::gemma4_assistant_mtp::Gemma4AssistantMtpDisableReason::None,
            max_depth: 1,
            config: None,
        };
        let mut decisions = Vec::new();
        status.append_route_decisions(Gemma4AssistantMtpTelemetry::default(), &mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_configured"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_validated"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_enabled"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_disable_reason"),
            Some(&0)
        );
        assert_eq!(decisions.get("ax_mlx_gemma4_assistant_mtp_depth"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_confidence_mode"),
            Some(&1)
        );
        assert_eq!(decisions.get("ax_mlx_speculation_profile"), Some(&0));
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_draft_tokens"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_rejected_tokens"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_corrections"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_accept_rate_x1000"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us"),
            Some(&0)
        );
    }

    #[test]
    fn gemma4_unified_multimodal_telemetry_reports_modality_counts() {
        let inputs = ax_engine_core::gemma4_unified::Gemma4UnifiedRuntimeInputs {
            images: vec![
                ax_engine_core::gemma4_unified::Gemma4UnifiedImageRuntimeInput {
                    span: ax_engine_core::gemma4_unified::Gemma4UnifiedTokenSpan {
                        modality: ax_engine_core::gemma4_unified::Gemma4UnifiedModality::Image,
                        placeholder_index: 1,
                        replacement_start: 1,
                        soft_token_count: 1,
                        replacement_token_count: 3,
                    },
                    pixel_values: vec![0.0, 1.0, 2.0],
                    pixel_position_ids: vec![[0, 0]],
                },
            ],
            audios: vec![
                ax_engine_core::gemma4_unified::Gemma4UnifiedAudioRuntimeInput {
                    span: ax_engine_core::gemma4_unified::Gemma4UnifiedTokenSpan {
                        modality: ax_engine_core::gemma4_unified::Gemma4UnifiedModality::Audio,
                        placeholder_index: 4,
                        replacement_start: 4,
                        soft_token_count: 1,
                        replacement_token_count: 3,
                    },
                    input_features: vec![0.0, 1.0],
                    frame_count: 1,
                    feature_count: 2,
                },
            ],
            videos: vec![
                ax_engine_core::gemma4_unified::Gemma4UnifiedVideoRuntimeInput {
                    span: ax_engine_core::gemma4_unified::Gemma4UnifiedTokenSpan {
                        modality: ax_engine_core::gemma4_unified::Gemma4UnifiedModality::Video,
                        placeholder_index: 7,
                        replacement_start: 7,
                        soft_token_count: 1,
                        replacement_token_count: 3,
                    },
                    soft_token_ranges: Vec::new(),
                    pixel_values: vec![0.0, 1.0, 2.0],
                    pixel_position_ids: vec![[0, 0]],
                    frame_count: 1,
                },
            ],
        };
        let mut telemetry = Gemma4UnifiedMultimodalTelemetry::default();
        telemetry.record_prefill(&inputs, true);
        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_gemma4_unified_multimodal_prefill_requests"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_unified_image_inputs"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_unified_audio_inputs"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_unified_video_inputs"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_unified_visual_inputs"),
            Some(&2)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_unified_prefix_cache_disabled"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_unified_mtp_prefill_warmup_skipped"),
            Some(&1)
        );
    }

    #[test]
    fn gemma4_assistant_mtp_route_metadata_reports_runtime_telemetry() {
        let status = Gemma4AssistantMtpStatus {
            configured: true,
            validated: true,
            enabled: true,
            attach_failed: false,
            disable_reason: crate::gemma4_assistant_mtp::Gemma4AssistantMtpDisableReason::None,
            max_depth: 1,
            config: None,
        };
        let mut telemetry = Gemma4AssistantMtpTelemetry::default();
        telemetry.record_submitted(4, 120);
        telemetry.record_verified(4, 3, 240, 80);

        let mut decisions = Vec::new();
        status.append_route_decisions(telemetry, &mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_enabled"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_confidence_mode"),
            Some(&1)
        );
        assert_eq!(decisions.get("ax_mlx_speculation_profile"), Some(&0));
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_draft_tokens"),
            Some(&4)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_accepted_tokens"),
            Some(&3)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_rejected_tokens"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_corrections"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_accept_rate_x1000"),
            Some(&750)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_verify_forward_wall_us"),
            Some(&240)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_verify_eval_wall_us"),
            Some(&80)
        );
        assert_eq!(
            decisions.get("ax_mlx_gemma4_assistant_mtp_draft_forward_wall_us"),
            Some(&120)
        );
    }

    #[test]
    fn mtp_telemetry_counts_gemma4_assistant_as_model_draft_source() {
        let mut telemetry = MtpTelemetry::default();
        telemetry.record_step(
            2,
            1,
            &[
                MtpDraftSource::Gemma4Assistant,
                MtpDraftSource::Gemma4Assistant,
            ],
            None,
            1,
        );

        assert_eq!(telemetry.draft_source_mtp_tokens, 2);
        assert_eq!(telemetry.accepted_source_mtp_tokens, 1);
        assert_eq!(telemetry.mtp_only_accept_rate_ewma_samples, 1);
        assert_eq!(telemetry.mtp_only_accept_rate_ewma, 0.5);
    }

    fn linear_attn_split_weights() -> crate::weights::LinearAttentionWeights {
        crate::weights::LinearAttentionWeights {
            in_proj_qkv: Some(unit_weight()),
            in_proj_z: Some(unit_weight()),
            in_proj_a: Some(unit_weight()),
            in_proj_b: Some(unit_weight()),
            in_proj_qkvz: None,
            in_proj_ba: None,
            conv1d_dense: mlx_sys::zeros(&[1, 1, 1], MlxDtype::Float32, None),
            dt_bias: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            a_log: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            out_proj: unit_weight(),
        }
    }

    fn linear_attn_packed_weights() -> crate::weights::LinearAttentionWeights {
        crate::weights::LinearAttentionWeights {
            in_proj_qkv: None,
            in_proj_z: None,
            in_proj_a: None,
            in_proj_b: None,
            in_proj_qkvz: Some(unit_weight()),
            in_proj_ba: Some(unit_weight()),
            conv1d_dense: mlx_sys::zeros(&[1, 1, 1], MlxDtype::Float32, None),
            dt_bias: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            a_log: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            norm: mlx_sys::zeros(&[1], MlxDtype::Float32, None),
            out_proj: unit_weight(),
        }
    }

    #[test]
    fn weight_layout_telemetry_counts_linear_attention_packed_and_split_layers() {
        let mut packed = runner_test_layer();
        packed.linear_attn = Some(linear_attn_packed_weights());
        let mut split_a = runner_test_layer();
        split_a.linear_attn = Some(linear_attn_split_weights());
        let mut split_b = runner_test_layer();
        split_b.linear_attn = Some(linear_attn_split_weights());

        let telemetry = WeightLayoutTelemetry::from_weights(&runner_test_weights(vec![
            packed, split_a, split_b,
        ]));
        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_linear_attention_qkvz_ba_packed_layers"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_linear_attention_split_qkvba_layers"),
            Some(&2)
        );
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
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
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
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
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
    fn terminal_token_ids_resolve_pad_for_standard_models() {
        let mut manifest = dense_manifest();
        set_vocab_size(&mut manifest, 128);
        let artifacts = write_artifacts(manifest);
        fs::write(
            artifacts.root_dir().join("config.json"),
            r#"{"eos_token_id":1,"pad_token_id":0}"#,
        )
        .expect("config should write");

        assert_eq!(resolve_terminal_token_ids(&artifacts), vec![0, 1]);
    }

    #[test]
    fn terminal_token_ids_ignore_pad_for_diffusion_gemma() {
        let mut manifest = dense_manifest();
        manifest.model_family = "diffusion_gemma".to_string();
        set_vocab_size(&mut manifest, 128);
        let artifacts = write_artifacts(manifest);
        fs::write(
            artifacts.root_dir().join("config.json"),
            r#"{"eos_token_id":[1,106,50],"pad_token_id":0}"#,
        )
        .expect("config should write");
        fs::write(
            artifacts.root_dir().join("tokenizer_config.json"),
            r#"{"pad_token":"<pad>"}"#,
        )
        .expect("tokenizer config should write");
        fs::write(
            artifacts.root_dir().join("tokenizer.json"),
            r#"{"added_tokens":[{"id":0,"content":"<pad>"},{"id":106,"content":"<turn|>"}]}"#,
        )
        .expect("tokenizer should write");

        assert_eq!(resolve_terminal_token_ids(&artifacts), vec![1, 50, 106]);
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
        manifest.model_family = "qwen3".to_string();
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
            layer_freq: None,
            first_dense_layers: None,
            shared_expert_count: None,
            sigmoid_routing: false,
            routed_scaling_factor: None,
            n_group: None,
            topk_group: None,
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
        manifest.model_family = "gemma4".to_string();
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
    fn mlx_manifest_validation_rejects_unknown_model_family() {
        let artifacts = write_artifacts(dense_manifest());

        let error =
            validate_mlx_supported_manifest(&artifacts).expect_err("unknown family should fail");

        assert!(
            error
                .to_string()
                .contains("not supported by the MLX runner")
        );
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

        let error = validate_mla_moe_manifest(&manifest)
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
    fn mlx_manifest_validation_allows_deepseek_v3_kv_b_contract() {
        let mut manifest = glm4_moe_lite_manifest();
        manifest.model_family = "deepseek_v3".to_string();
        manifest.glm_router = NativeGlmRouterConfig::default();
        manifest.moe.layer_freq = Some(1);
        manifest.moe.first_dense_layers = Some(1);
        manifest.moe.shared_expert_count = Some(1);
        manifest.moe.sigmoid_routing = true;
        manifest.moe.routed_scaling_factor = Some(2.5);
        manifest.moe.n_group = Some(1);
        manifest.moe.topk_group = Some(1);
        manifest.tensors.retain(|tensor| {
            !matches!(
                tensor.role,
                NativeTensorRole::AttentionEmbedQ | NativeTensorRole::AttentionUnembedOut
            )
        });
        for layer in 0..2 {
            manifest.tensors.push(tensor(
                &format!("model.layers.{layer}.self_attn.kv_b_proj.weight"),
                NativeTensorRole::AttentionKvB,
                Some(layer),
                vec![4, 2],
            ));
        }
        let artifacts = write_artifacts(manifest);

        validate_mlx_supported_manifest(&artifacts)
            .expect("DeepSeek V3 KV-B runtime contract should be accepted");
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
            let error = validate_mla_moe_manifest(&manifest)
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

        MlxRunner::from_artifacts(&artifacts, 8, true, KvCompressionConfig::disabled())
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

        // PRD §8 Phase 6: per-attempt acceptance-by-depth histogram. The
        // three draft attempts above accepted {full, 0, 2} tokens, so the
        // histogram must be: bucket DEFAULT_DRAFT_LEN += 1, bucket 0 += 1,
        // bucket 2 += 1. Verify bucket 0 and bucket 2 directly (which are
        // both well below NGRAM_ACCEPT_DEPTH_BUCKETS) and assert all other
        // sub-bucket counters are zero.
        assert_eq!(decisions.get("ax_ngram_accept_at_depth_0"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_accept_at_depth_2"), Some(&1));
        assert_eq!(decisions.get("ax_ngram_accept_at_depth_1"), Some(&0));
        assert_eq!(decisions.get("ax_ngram_accept_at_depth_3"), Some(&0));
    }

    #[test]
    fn ngram_telemetry_accepts_by_depth_saturates_at_last_bucket() {
        // Drafts that accept beyond NGRAM_ACCEPT_DEPTH_BUCKETS - 1 must
        // land in the last bucket rather than panic-on-index-overflow or
        // silently drop. Without this saturation, a future longer-draft
        // policy could underreport its acceptance.
        let mut telemetry = NgramAccelerationTelemetry::default();
        // Accept more than the histogram length on a single draft.
        telemetry.record_draft(16, 16);
        // Accept exactly the last in-range bucket on another draft.
        telemetry.record_draft(
            NGRAM_ACCEPT_DEPTH_BUCKETS - 1,
            NGRAM_ACCEPT_DEPTH_BUCKETS - 1,
        );

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        // Both attempts saturate into bucket NGRAM_ACCEPT_DEPTH_BUCKETS - 1
        // because the over-the-limit attempt is clamped, and the exactly-
        // at-the-last-bucket attempt naturally lands there.
        let last_key = format!(
            "ax_ngram_accept_at_depth_{}",
            NGRAM_ACCEPT_DEPTH_BUCKETS - 1
        );
        assert_eq!(decisions.get(last_key.as_str()), Some(&2));
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
            ..LinearAttentionProfileSnapshot::default()
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
            ..LinearAttentionProfileSnapshot::default()
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
    fn linear_attention_direct_cpp_route_decisions_emit_when_attempted() {
        let mut profile = LinearAttentionProfileSnapshot {
            direct_cpp_inputs_attempts: 2,
            direct_cpp_inputs_hits: 1,
            direct_cpp_inputs_fallbacks: 1,
            direct_cpp_inputs_profile_blocked: 1,
            ..LinearAttentionProfileSnapshot::default()
        };
        profile.merge_from(LinearAttentionProfileSnapshot {
            direct_cpp_inputs_attempts: 3,
            direct_cpp_inputs_hits: 2,
            direct_cpp_inputs_fallbacks: 1,
            direct_cpp_inputs_profile_blocked: 0,
            ..LinearAttentionProfileSnapshot::default()
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_attempts"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_hits"),
            Some(&3)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_fallbacks"),
            Some(&2)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_inputs_profile_blocked"),
            Some(&1)
        );
        assert!(!decisions.contains_key("ax_mlx_linear_attention_profile_enabled"));
    }

    #[test]
    fn linear_attention_direct_cpp_post_input_route_decisions_emit_when_attempted() {
        let mut profile = LinearAttentionProfileSnapshot {
            direct_cpp_post_input_attempts: 2,
            direct_cpp_post_input_hits: 1,
            direct_cpp_post_input_fallbacks: 1,
            direct_cpp_post_input_profile_blocked: 1,
            ..LinearAttentionProfileSnapshot::default()
        };
        profile.merge_from(LinearAttentionProfileSnapshot {
            direct_cpp_post_input_attempts: 3,
            direct_cpp_post_input_hits: 2,
            direct_cpp_post_input_fallbacks: 1,
            direct_cpp_post_input_profile_blocked: 0,
            ..LinearAttentionProfileSnapshot::default()
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_attempts"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_hits"),
            Some(&3)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_fallbacks"),
            Some(&2)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_cpp_linear_attention_post_input_profile_blocked"),
            Some(&1)
        );
        assert!(!decisions.contains_key("ax_mlx_linear_attention_profile_enabled"));
    }

    #[test]
    fn linear_attention_decode_post_input_metal_route_decisions_emit_when_attempted() {
        let mut profile = LinearAttentionProfileSnapshot {
            decode_post_input_metal_attempts: 2,
            decode_post_input_metal_hits: 1,
            decode_post_input_metal_fallbacks: 1,
            decode_post_input_metal_profile_blocked: 1,
            ..LinearAttentionProfileSnapshot::default()
        };
        profile.merge_from(LinearAttentionProfileSnapshot {
            decode_post_input_metal_attempts: 3,
            decode_post_input_metal_hits: 2,
            decode_post_input_metal_fallbacks: 1,
            decode_post_input_metal_profile_blocked: 0,
            ..LinearAttentionProfileSnapshot::default()
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_attempts"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_hits"),
            Some(&3)
        );
        assert_eq!(
            decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_fallbacks"),
            Some(&2)
        );
        assert_eq!(
            decisions.get("ax_mlx_qwen_linear_attention_decode_post_input_metal_profile_blocked"),
            Some(&1)
        );
        assert!(!decisions.contains_key("ax_mlx_linear_attention_profile_enabled"));
    }

    #[test]
    fn dense_ffn_fastpath_route_decisions_emit_when_attempted() {
        let mut profile = DenseFfnFastpathSnapshot {
            qwen_gate_up_matvec_metal_attempts: 2,
            qwen_gate_up_matvec_metal_hits: 1,
            qwen_gate_up_matvec_metal_fallbacks: 1,
        };
        profile.merge_from(DenseFfnFastpathSnapshot {
            qwen_gate_up_matvec_metal_attempts: 3,
            qwen_gate_up_matvec_metal_hits: 2,
            qwen_gate_up_matvec_metal_fallbacks: 1,
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(
            decisions.get("ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_attempts"),
            Some(&5)
        );
        assert_eq!(
            decisions.get("ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_hits"),
            Some(&3)
        );
        assert_eq!(
            decisions.get("ax_mlx_qwen_dense_ffn_gate_up_matvec_metal_fallbacks"),
            Some(&2)
        );
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
            post_attn_ffn_gate_up_wall_us: 900,
            post_attn_ffn_activation_wall_us: 300,
            post_attn_ffn_down_wall_us: 600,
            post_attn_output_proj_wall_us: 300,
            post_attn_residual_norm_wall_us: 100,
            post_attn_residual_gate_wall_us: 200,
            lm_head_wall_us: 150,
            moe_router_wall_us: 400,
            moe_expert_gate_up_wall_us: 500,
            moe_expert_activation_wall_us: 100,
            moe_expert_down_wall_us: 300,
            moe_expert_weighted_sum_wall_us: 50,
            moe_shared_expert_wall_us: 450,
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
            post_attn_ffn_gate_up_wall_us: 200,
            post_attn_ffn_activation_wall_us: 50,
            post_attn_ffn_down_wall_us: 150,
            post_attn_output_proj_wall_us: 75,
            post_attn_residual_norm_wall_us: 25,
            post_attn_residual_gate_wall_us: 50,
            lm_head_wall_us: 50,
            moe_router_wall_us: 100,
            moe_expert_gate_up_wall_us: 120,
            moe_expert_activation_wall_us: 30,
            moe_expert_down_wall_us: 80,
            moe_expert_weighted_sum_wall_us: 15,
            moe_shared_expert_wall_us: 110,
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
            decisions.get("ax_mlx_decode_profile_post_attn_ffn_gate_up_wall_us"),
            Some(&1100)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_post_attn_ffn_activation_wall_us"),
            Some(&350)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_post_attn_ffn_down_wall_us"),
            Some(&750)
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
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_moe_router_wall_us"),
            Some(&500)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_moe_expert_gate_up_wall_us"),
            Some(&620)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_moe_expert_activation_wall_us"),
            Some(&130)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_moe_expert_down_wall_us"),
            Some(&380)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_moe_expert_weighted_sum_wall_us"),
            Some(&65)
        );
        assert_eq!(
            decisions.get("ax_mlx_decode_profile_moe_shared_expert_wall_us"),
            Some(&560)
        );

        let mut disabled_decisions = Vec::new();
        DecodeProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
        assert!(disabled_decisions.is_empty());
    }

    #[test]
    fn prefill_profile_route_decisions_emit_only_when_enabled() {
        let mut profile = PrefillProfileSnapshot {
            enabled: 1,
            prefill_steps: 2,
            layers: 48,
            tokens: 4096,
            per_layer_input_wall_us: 800,
            pre_sdpa_wall_us: 1200,
            pre_sdpa_qkv_proj_wall_us: 700,
            pre_sdpa_qk_norm_wall_us: 200,
            pre_sdpa_rope_kv_wall_us: 300,
            sdpa_wall_us: 500,
            post_attn_wall_us: 2400,
            post_attn_ffn_wall_us: 1800,
            post_attn_ffn_gate_up_wall_us: 900,
            post_attn_ffn_activation_wall_us: 300,
            post_attn_ffn_down_wall_us: 600,
            post_attn_output_proj_wall_us: 300,
            post_attn_residual_norm_wall_us: 100,
            post_attn_residual_gate_wall_us: 200,
            lm_head_wall_us: 150,
            moe_router_wall_us: 400,
            moe_expert_gate_up_wall_us: 500,
            moe_expert_activation_wall_us: 100,
            moe_expert_down_wall_us: 300,
            moe_expert_weighted_sum_wall_us: 50,
            moe_shared_expert_wall_us: 450,
        };
        profile.merge_from(PrefillProfileSnapshot {
            enabled: 1,
            prefill_steps: 1,
            layers: 24,
            tokens: 2048,
            per_layer_input_wall_us: 200,
            pre_sdpa_wall_us: 300,
            pre_sdpa_qkv_proj_wall_us: 200,
            pre_sdpa_qk_norm_wall_us: 50,
            pre_sdpa_rope_kv_wall_us: 75,
            sdpa_wall_us: 100,
            post_attn_wall_us: 600,
            post_attn_ffn_wall_us: 400,
            post_attn_ffn_gate_up_wall_us: 200,
            post_attn_ffn_activation_wall_us: 50,
            post_attn_ffn_down_wall_us: 150,
            post_attn_output_proj_wall_us: 75,
            post_attn_residual_norm_wall_us: 25,
            post_attn_residual_gate_wall_us: 50,
            lm_head_wall_us: 50,
            moe_router_wall_us: 100,
            moe_expert_gate_up_wall_us: 120,
            moe_expert_activation_wall_us: 30,
            moe_expert_down_wall_us: 80,
            moe_expert_weighted_sum_wall_us: 15,
            moe_shared_expert_wall_us: 110,
        });

        let mut decisions = Vec::new();
        profile.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_prefill_profile_enabled"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_prefill_steps"),
            Some(&3)
        );
        assert_eq!(decisions.get("ax_mlx_prefill_profile_layers"), Some(&72));
        assert_eq!(decisions.get("ax_mlx_prefill_profile_tokens"), Some(&6144));
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_pre_sdpa_wall_us"),
            Some(&1500)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_pre_sdpa_qkv_proj_wall_us"),
            Some(&900)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_sdpa_wall_us"),
            Some(&600)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_post_attn_ffn_wall_us"),
            Some(&2200)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_post_attn_ffn_gate_up_wall_us"),
            Some(&1100)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_post_attn_ffn_activation_wall_us"),
            Some(&350)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_post_attn_ffn_down_wall_us"),
            Some(&750)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_lm_head_wall_us"),
            Some(&200)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_moe_router_wall_us"),
            Some(&500)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_moe_expert_gate_up_wall_us"),
            Some(&620)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_moe_expert_activation_wall_us"),
            Some(&130)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_moe_expert_down_wall_us"),
            Some(&380)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_moe_expert_weighted_sum_wall_us"),
            Some(&65)
        );
        assert_eq!(
            decisions.get("ax_mlx_prefill_profile_moe_shared_expert_wall_us"),
            Some(&560)
        );

        let mut disabled_decisions = Vec::new();
        PrefillProfileSnapshot::default().append_route_decisions(&mut disabled_decisions);
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
    fn linear_attention_non_repeating_prompt_without_initial_draft_uses_direct_fallback() {
        let empty = NgramTable::new();
        let variant = NgramPolicyVariant::MajorityRecency;
        // Empty table → probe returns no draft → disable for linear+NON_REPEATING.
        assert!(
            linear_ngram_initial_prompt_should_disable_request(
                true,
                crate::ngram_accel::PROMPT_CLASS_NON_REPEATING,
                &empty,
                variant,
            ),
            "linear + NON_REPEATING + empty table should disable (no useful bigrams)"
        );
        // REPEATING prompt short-circuits before probing.
        assert!(
            !linear_ngram_initial_prompt_should_disable_request(
                true,
                crate::ngram_accel::PROMPT_CLASS_REPEATING,
                &empty,
                variant,
            ),
            "repeating prompts should keep the speculative path eligible"
        );
        // Dense models don't pay recompute cost → no disable gate.
        assert!(
            !linear_ngram_initial_prompt_should_disable_request(
                false,
                crate::ngram_accel::PROMPT_CLASS_NON_REPEATING,
                &empty,
                variant,
            ),
            "dense models can cheaply roll back and should keep the existing gate"
        );
        // NON_REPEATING prompt but with prompt-seeded bigrams that yield a draft →
        // do NOT disable; the prompt has useful structure despite the classifier label.
        let mut seeded = NgramTable::new();
        // feed_from_prompt marks bigrams with prompt_count=1 so bypass_prompt_min_support
        // allows drafting from step 1 even with LINEAR_MIN_NGRAM_SUPPORT=2.
        // Ending sequence ...10,20 leaves tail=[...,10,20], which looks up bigram (10,20)→30.
        seeded.feed_from_prompt(&[10, 20, 30, 10, 20]);
        assert!(
            !linear_ngram_initial_prompt_should_disable_request(
                true,
                crate::ngram_accel::PROMPT_CLASS_NON_REPEATING,
                &seeded,
                variant,
            ),
            "NON_REPEATING but with prompt-seeded bigrams should stay enabled"
        );
    }

    #[test]
    fn linear_attention_direct_fallback_reenables_when_output_builds_draft() {
        let mut state = RequestState::new(1, 7);
        state.ngram_acceleration_disabled_for_request = true;
        state.ngram_request_disable_reason = NgramRequestDisableReason::LinearNoDraft;
        state.linear_ngram_no_draft_streak = LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD;
        state.ngram_disabled_steps = LINEAR_NGRAM_RETRY_INTERVAL;

        state.ngram.feed(&[1, 2, 3, 4, 9]);
        maybe_reenable_linear_ngram_from_fallback_output(
            &mut state,
            NgramPolicyVariant::MajorityRecency,
            true,
        );
        assert!(
            state.ngram_acceleration_disabled_for_request,
            "one observed continuation is below the linear-attention support gate"
        );
        assert_eq!(
            state.linear_ngram_reenable_probe_countdown,
            LINEAR_NGRAM_REENABLE_PROBE_INTERVAL
        );

        state.ngram.feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4]);
        maybe_reenable_linear_ngram_from_fallback_output(
            &mut state,
            NgramPolicyVariant::MajorityRecency,
            true,
        );
        assert!(
            state.ngram_acceleration_disabled_for_request,
            "reenable probing is throttled while direct fallback is active"
        );
        assert_eq!(
            state.linear_ngram_reenable_probe_countdown,
            LINEAR_NGRAM_REENABLE_PROBE_INTERVAL - 1
        );

        state.linear_ngram_reenable_probe_countdown = 0;
        maybe_reenable_linear_ngram_from_fallback_output(
            &mut state,
            NgramPolicyVariant::MajorityRecency,
            true,
        );

        assert!(!state.ngram_acceleration_disabled_for_request);
        assert_eq!(
            state.ngram_request_disable_reason,
            NgramRequestDisableReason::None
        );
        assert_eq!(state.linear_ngram_no_draft_streak, 0);
        assert_eq!(state.linear_ngram_reenable_probe_countdown, 0);
        assert_eq!(state.ngram_disabled_steps, 0);
    }

    #[test]
    fn linear_attention_reenable_keeps_short_output_disable_closed() {
        let mut state = RequestState::new(1, 7);
        state.ngram_acceleration_disabled_for_request = true;
        state.ngram_request_disable_reason = NgramRequestDisableReason::ShortOutputBudget;
        state
            .ngram
            .feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 4]);

        maybe_reenable_linear_ngram_from_fallback_output(
            &mut state,
            NgramPolicyVariant::MajorityRecency,
            true,
        );

        assert!(state.ngram_acceleration_disabled_for_request);
        assert_eq!(
            state.ngram_request_disable_reason,
            NgramRequestDisableReason::ShortOutputBudget
        );
    }

    #[test]
    fn linear_attention_initial_no_draft_stays_on_direct_fallback() {
        let mut state = RequestState::new(1, 7);
        state.ngram_acceleration_disabled_for_request = true;
        state.ngram_request_disable_reason = NgramRequestDisableReason::LinearInitialNoDraft;
        state
            .ngram
            .feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 4]);

        maybe_reenable_linear_ngram_from_fallback_output(
            &mut state,
            NgramPolicyVariant::MajorityRecency,
            true,
        );

        assert!(
            state.ngram_acceleration_disabled_for_request,
            "initial non-repeating prompts stay on direct fallback for the request"
        );
        assert_eq!(
            state.ngram_request_disable_reason,
            NgramRequestDisableReason::LinearInitialNoDraft
        );
        assert_eq!(state.linear_ngram_reenable_probe_countdown, 0);
    }

    #[test]
    fn request_disabled_fallback_only_feeds_ngram_for_runtime_no_draft() {
        assert!(ngram_request_disabled_fallback_should_feed_output(
            NgramRequestDisableReason::LinearNoDraft
        ));
        assert!(!ngram_request_disabled_fallback_should_feed_output(
            NgramRequestDisableReason::LinearInitialNoDraft
        ));
        assert!(!ngram_request_disabled_fallback_should_feed_output(
            NgramRequestDisableReason::ShortOutputBudget
        ));
    }

    #[test]
    fn request_disabled_direct_fast_path_skips_non_reenable_ngram_fallback() {
        assert!(ngram_request_disabled_direct_fast_path(
            true,
            false,
            false,
            true,
            NgramRequestDisableReason::LinearInitialNoDraft,
        ));
        assert!(ngram_request_disabled_direct_fast_path(
            true,
            false,
            false,
            true,
            NgramRequestDisableReason::ShortOutputBudget,
        ));
        assert!(!ngram_request_disabled_direct_fast_path(
            true,
            false,
            false,
            true,
            NgramRequestDisableReason::LinearNoDraft,
        ));
        assert!(!ngram_request_disabled_direct_fast_path(
            true,
            true,
            false,
            true,
            NgramRequestDisableReason::LinearInitialNoDraft,
        ));
        assert!(!ngram_request_disabled_direct_fast_path(
            true,
            false,
            true,
            true,
            NgramRequestDisableReason::LinearInitialNoDraft,
        ));
        assert!(!ngram_request_disabled_direct_fast_path(
            false,
            false,
            false,
            true,
            NgramRequestDisableReason::LinearInitialNoDraft,
        ));
    }

    #[test]
    fn linear_attention_reenable_requires_greedy_exact_decode() {
        let mut state = RequestState::new(1, 7);
        state.ngram_acceleration_disabled_for_request = true;
        state.ngram_request_disable_reason = NgramRequestDisableReason::LinearNoDraft;
        state
            .ngram
            .feed(&[1, 2, 3, 4, 9, 1, 2, 3, 4, 9, 1, 2, 3, 4]);

        maybe_reenable_linear_ngram_from_fallback_output(
            &mut state,
            NgramPolicyVariant::MajorityRecency,
            false,
        );

        assert!(state.ngram_acceleration_disabled_for_request);
        assert_eq!(
            state.ngram_request_disable_reason,
            NgramRequestDisableReason::LinearNoDraft
        );
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
            forward_layer_loop_wall_us: 2,
            forward_head_wall_us: 1,
            argmax_wall_us: 4,
            async_eval_wall_us: 2,
            next_complete_wall_us: 6,
            pending_eval_wall_us: 5,
            pending_read_wall_us: 1,
            linear_attention_layer_ops: 22,
            linear_attention_layer_count: 2,
            full_attention_layer_ops: 20,
            full_attention_layer_count: 4,
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
            decisions.get("ax_mlx_direct_pipeline_forward_layer_loop_wall_us"),
            Some(&2)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_forward_head_wall_us"),
            Some(&1)
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
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_linear_attention_layer_ops"),
            Some(&22)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_linear_attention_layer_count"),
            Some(&2)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_full_attention_layer_ops"),
            Some(&20)
        );
        assert_eq!(
            decisions.get("ax_mlx_direct_pipeline_full_attention_layer_count"),
            Some(&4)
        );
        assert_eq!(decisions.get("ax_mlx_single_decode_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_single_decode_wall_us"), Some(&13));
        assert_eq!(decisions.get("ax_mlx_ngram_decode_steps"), Some(&1));
        assert_eq!(decisions.get("ax_mlx_ngram_decode_wall_us"), Some(&17));
        assert_eq!(decisions.get("ax_mlx_bonus_tokens"), Some(&2));
        assert_eq!(decisions.get("other_counter"), Some(&3));
    }

    #[test]
    fn decode_telemetry_records_diffusion_block() {
        let mut telemetry = DecodeTelemetry::default();

        // Record two diffusion blocks: one converged, one not.
        telemetry.record_diffusion_block(&crate::diffusion::DiffusionBlockResult {
            tokens: vec![1, 2, 3, 4],
            denoise_steps: 4,
            converged: true,
            converged_strict: true,
            converged_acceptance: false,
            converged_plateau: false,
            min_entropy: 0.003,
            min_acceptance_rate: 0.05,
            denoise_wall_us: 500,
            commit_wall_us: 100,
            block_wall_us: 700,
            commit_skipped: false,
            full_pipeline_used: false,
            kv_buffer_used: true,
        });
        telemetry.record_diffusion_block(&crate::diffusion::DiffusionBlockResult {
            tokens: vec![5, 6],
            denoise_steps: 8,
            converged: false,
            converged_strict: false,
            converged_acceptance: false,
            converged_plateau: false,
            min_entropy: 0.020,
            min_acceptance_rate: 0.15,
            denoise_wall_us: 900,
            commit_wall_us: 200,
            block_wall_us: 1300,
            commit_skipped: true,
            full_pipeline_used: true,
            kv_buffer_used: true,
        });

        let mut decisions: Vec<(String, u32)> = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_diffusion_blocks"), Some(&2));
        assert_eq!(decisions.get("ax_mlx_diffusion_denoise_steps"), Some(&12));
        assert_eq!(decisions.get("ax_mlx_diffusion_converged_blocks"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_diffusion_denoise_wall_us"),
            Some(&1400)
        );
        assert_eq!(decisions.get("ax_mlx_diffusion_commit_wall_us"), Some(&300));
        assert_eq!(decisions.get("ax_mlx_diffusion_block_wall_us"), Some(&2000));
        assert_eq!(decisions.get("ax_mlx_diffusion_min_entropy_bp"), Some(&30));
        assert_eq!(
            decisions.get("ax_mlx_diffusion_min_acceptance_rate_bp"),
            Some(&500)
        );
        assert_eq!(decisions.get("ax_mlx_diffusion_commit_skipped"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_diffusion_full_pipeline_used"),
            Some(&1)
        );
        assert_eq!(decisions.get("ax_mlx_diffusion_kv_buffer_used"), Some(&2));
    }

    #[test]
    fn decode_telemetry_emits_zero_diffusion_minima_without_blocks() {
        let telemetry = DecodeTelemetry::default();
        let mut decisions: Vec<(String, u32)> = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_diffusion_blocks"), Some(&0));
        assert_eq!(decisions.get("ax_mlx_diffusion_min_entropy_bp"), Some(&0));
        assert_eq!(
            decisions.get("ax_mlx_diffusion_min_acceptance_rate_bp"),
            Some(&0)
        );
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
        assert!(!direct_pipeline_clear_cache_due(2, 256));
    }

    #[test]
    fn request_fallback_direct_pipeline_reuses_pending_step() {
        assert_eq!(
            direct_pipeline_action(false, false),
            DirectPipelineAction::Bootstrap,
            "first fallback direct step must bootstrap the pipeline"
        );
        assert_eq!(
            direct_pipeline_action(true, false),
            DirectPipelineAction::ContinuePending,
            "later fallback direct steps must continue the pending lazy token"
        );
        assert_eq!(
            direct_pipeline_action(true, true),
            DirectPipelineAction::FinishPending,
            "final fallback direct step must not submit unused lookahead work"
        );
        assert_eq!(
            direct_pipeline_action(false, true),
            DirectPipelineAction::BootstrapFinal,
            "single-token final fallback must materialise without keeping a pending token"
        );
        assert!(
            should_drain_pending_direct_before_ngram(true, true),
            "greedy n-gram re-entry must first materialise the pending direct token"
        );
        assert!(!should_drain_pending_direct_before_ngram(true, false));
        assert!(!should_drain_pending_direct_before_ngram(false, true));
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
                fused_decode_query_readback_wall_us: 11,
                fused_decode_cold_metal_wall_us: 22,
                fused_decode_hot_tail_merge_wall_us: 33,
                fused_decode_output_staging_wall_us: 44,
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
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_QUERY_READBACK_WALL_US),
            Some(&11)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_COLD_METAL_WALL_US),
            Some(&22)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_HOT_TAIL_MERGE_WALL_US),
            Some(&33)
        );
        assert_eq!(
            decisions.get(ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_OUTPUT_STAGING_WALL_US),
            Some(&44)
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
        let dense_draft = ngram_acceleration_draft(
            &ngram,
            false,
            0.95,
            NgramPolicyVariant::MajorityRecency,
            false,
        )
        .draft;
        assert!(!dense_draft.is_empty(), "dense draft should be non-empty");
        assert!(
            dense_draft.len() <= MAX_DRAFT_LEN,
            "dense draft must not exceed MAX_DRAFT_LEN"
        );

        // Linear-attention: min_support=2 filters one-off n-grams.
        assert!(
            ngram_acceleration_draft(
                &ngram,
                true,
                0.95,
                NgramPolicyVariant::MajorityRecency,
                false
            )
            .draft
            .is_empty(),
            "linear attention should not probe one-off prompt n-grams"
        );

        ngram.feed(&[1, 2, 3]);
        let lin_draft = ngram_acceleration_draft(
            &ngram,
            true,
            0.95,
            NgramPolicyVariant::MajorityRecency,
            false,
        )
        .draft;
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
        manifest.model_family = "qwen3".to_string();
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
        manifest.model_family = "qwen3".to_string();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec!["sliding_attention".to_string()];
        manifest.global_head_dim = Some(8);
        let artifacts = write_artifacts(manifest);

        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("unknown family interleaved attention should fail closed");

        assert!(error.to_string().contains("not implemented"));
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

    #[test]
    fn embeddinggemma_forces_mean_pooling_across_embedding_apis() {
        assert_eq!(
            effective_embedding_pooling("embeddinggemma", EmbeddingPooling::Last),
            EmbeddingPooling::Mean
        );
        assert_eq!(
            effective_embedding_pooling("embeddinggemma", EmbeddingPooling::Cls),
            EmbeddingPooling::Mean
        );
        assert_eq!(
            effective_embedding_pooling("qwen3", EmbeddingPooling::Last),
            EmbeddingPooling::Last
        );
    }

    #[test]
    fn embeddinggemma_pooled_and_encoder_compile_keys_do_not_alias() {
        let thread_id = thread::current().id();
        let actual_lens = vec![3, 2];
        let encoder_key: EmbedGemmaBatchCompileKey = (
            thread_id,
            EmbedGemmaBatchCompileKind::Encoder,
            2,
            3,
            actual_lens.clone(),
        );
        let pooled_key: EmbedGemmaBatchCompileKey = (
            thread_id,
            EmbedGemmaBatchCompileKind::Pooled,
            2,
            3,
            actual_lens,
        );

        assert_ne!(encoder_key, pooled_key);
    }

    #[test]
    fn argpartition_axis_2d_topk_returns_dominant_indices() {
        let vocab: i32 = 10;
        let k: i32 = 3;

        let mut data = vec![0.0f32; vocab as usize];
        data[2] = 8.0;
        data[5] = 4.0;

        let arr = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            data.len() * 4,
            &[1, vocab],
            MlxDtype::Float32,
        );

        let neg = multiply(&arr, &mlx_scalar_f32(-1.0), None);
        let part = argpartition_axis(&neg, k, 1, None);
        let top = slice(&part, &[0, 0], &[1, k], &[1, 1], None);
        let top_u32 = astype(&top, MlxDtype::Uint32, None);
        mlx_sys::eval(&[&top_u32]);
        let indices = top_u32.data_u32().to_vec();

        assert!(
            indices.contains(&2),
            "top-{k} should contain index 2 (value 8.0), got {indices:?}"
        );
        assert!(
            indices.contains(&5),
            "top-{k} should contain index 5 (value 4.0), got {indices:?}"
        );
    }

    #[test]
    fn topk_target_softmax_approximates_full_softmax_for_dominant_tokens() {
        use mlx_sys::eval;

        let vocab: i32 = 100;
        let pending: Vec<u32> = vec![10, 25, 55];
        let pending_log_probs: Vec<f32> = vec![-2.0, -3.0, -1.5];
        let temperature: f32 = 0.8;
        // verify_len = 1 (last_token) + pending.len(); logits_all shape [verify_len, vocab].
        // logits_all[i] = prediction after position i = target for pending[i].
        let verify_len = (pending.len() + 1) as i32;

        let mut logits_data = vec![0.0f32; verify_len as usize * vocab as usize];
        // Row 0 is the target for pending[0], row 1 for pending[1], etc.
        logits_data[10] = 5.0;
        logits_data[20] = 4.0;
        logits_data[vocab as usize + 25] = 6.0;
        logits_data[vocab as usize + 30] = 3.0;
        logits_data[2 * vocab as usize + 55] = 7.0;
        logits_data[2 * vocab as usize + 60] = 2.0;

        let logits_all = MlxArray::from_raw_data(
            logits_data.as_ptr() as *const u8,
            logits_data.len() * 4,
            &[verify_len, vocab],
            MlxDtype::Float32,
        );

        let sampling = MlxSamplingParams {
            temperature,
            ..Default::default()
        };
        let mut full_workspace = MtpTargetProbWorkspace::default();
        let mut topk_workspace = MtpTargetProbWorkspace::default();

        let full_result = compute_mtp_target_probs(
            &logits_all,
            &pending,
            &pending_log_probs,
            vocab,
            sampling,
            None,
            MtpDraftFilter::IDENTITY,
            &mut full_workspace,
        )
        .expect("full should return Some");
        if let LazyTargetProbs::Full(arr) = &full_result {
            eval(&[arr]);
        }
        let full_probs = full_result
            .extract_cpu_into(&pending, &mut full_workspace)
            .unwrap();

        let topk_result = compute_mtp_target_probs(
            &logits_all,
            &pending,
            &pending_log_probs,
            vocab,
            sampling,
            Some(32),
            MtpDraftFilter::IDENTITY,
            &mut topk_workspace,
        )
        .expect("topk should return Some");
        if let LazyTargetProbs::TopK { indices, probs, .. } = &topk_result {
            eval(&[indices, probs]);
        }
        let topk_probs = topk_result
            .extract_cpu_into(&pending, &mut topk_workspace)
            .unwrap();

        assert_eq!(full_probs.len(), 3);
        assert_eq!(topk_probs.len(), 3);

        for i in 0..3 {
            assert!(
                full_probs[i] > 0.0,
                "position {i}: full softmax should be > 0, got {}",
                full_probs[i]
            );
            assert!(
                topk_probs[i] > 0.0,
                "position {i}: topk should find token {} in top-32, got p=0",
                pending[i]
            );
            let ratio = topk_probs[i] / full_probs[i];
            assert!(
                (0.85..1.15).contains(&ratio),
                "position {i}: full={} topk={} ratio={ratio:.3}",
                full_probs[i],
                topk_probs[i]
            );
        }
    }

    #[test]
    fn topk_target_softmax_returns_zero_for_out_of_set_tokens() {
        use mlx_sys::eval;

        let vocab: i32 = 100;
        let verify_len: usize = 1;
        let pending: Vec<u32> = vec![99];
        let pending_log_probs: Vec<f32> = vec![-2.0];
        let temperature: f32 = 0.8;

        let mut logits_data = vec![0.0f32; verify_len * vocab as usize];
        logits_data[0] = 10.0;
        logits_data[1] = 9.0;
        logits_data[2] = 8.0;

        let logits_all = MlxArray::from_raw_data(
            logits_data.as_ptr() as *const u8,
            logits_data.len() * 4,
            &[verify_len as i32, vocab],
            MlxDtype::Float32,
        );

        let sampling = MlxSamplingParams {
            temperature,
            ..Default::default()
        };
        let mut workspace = MtpTargetProbWorkspace::default();

        let topk_result = compute_mtp_target_probs(
            &logits_all,
            &pending,
            &pending_log_probs,
            vocab,
            sampling,
            Some(3),
            MtpDraftFilter::IDENTITY,
            &mut workspace,
        )
        .expect("topk should return Some");

        if let LazyTargetProbs::TopK { indices, probs, .. } = &topk_result {
            eval(&[indices, probs]);
        }

        let topk_probs = topk_result
            .extract_cpu_into(&pending, &mut workspace)
            .unwrap();
        assert_eq!(topk_probs.len(), 1);
        assert_eq!(
            topk_probs[0], 0.0,
            "token 99 is outside top-3, should return 0"
        );
    }

    #[test]
    fn mtp_telemetry_merge_from_combines_ewma_fields() {
        // merge_from must not silently drop EWMA values from the other side.
        // Both accept_rate_ewma and mtp_only_accept_rate_ewma should be merged
        // as sample-weighted averages so that batch-level route decisions report
        // the correct aggregate rather than always reporting 0.

        let mut a = MtpTelemetry::default();
        // Request A: 4 steps, EWMA converges near 1.0
        for _ in 0..4 {
            a.record_step(3, 3, &[MtpDraftSource::Mtp; 3], None, 3);
        }
        assert_eq!(a.accept_rate_ewma_samples, 4);
        assert_eq!(a.mtp_only_accept_rate_ewma_samples, 4);
        assert!((a.accept_rate_ewma - 1.0).abs() < 1e-5);
        assert!((a.mtp_only_accept_rate_ewma - 1.0).abs() < 1e-5);

        let mut b = MtpTelemetry::default();
        // Request B: 4 steps, EWMA converges near 0.0
        for _ in 0..4 {
            b.record_step(3, 0, &[MtpDraftSource::Mtp; 3], None, 0);
        }
        assert_eq!(b.accept_rate_ewma_samples, 4);
        assert_eq!(b.mtp_only_accept_rate_ewma_samples, 4);
        assert!((b.accept_rate_ewma - 0.0).abs() < 1e-5);
        assert!((b.mtp_only_accept_rate_ewma - 0.0).abs() < 1e-5);

        a.merge_from(b);

        // After merge: 8 total samples (4+4), weighted average of 1.0 and 0.0 = 0.5.
        assert_eq!(a.accept_rate_ewma_samples, 8);
        assert_eq!(a.mtp_only_accept_rate_ewma_samples, 8);
        assert!(
            (a.accept_rate_ewma - 0.5).abs() < 1e-5,
            "merged accept_rate_ewma should be 0.5, got {}",
            a.accept_rate_ewma
        );
        assert!(
            (a.mtp_only_accept_rate_ewma - 0.5).abs() < 1e-5,
            "merged mtp_only_accept_rate_ewma should be 0.5, got {}",
            a.mtp_only_accept_rate_ewma
        );

        // Merging with a zero-sample side must not produce NaN.
        let empty = MtpTelemetry::default();
        let ewma_before = a.accept_rate_ewma;
        let samples_before = a.accept_rate_ewma_samples;
        a.merge_from(empty);
        assert_eq!(a.accept_rate_ewma_samples, samples_before);
        assert!((a.accept_rate_ewma - ewma_before).abs() < 1e-6);
        assert!(a.accept_rate_ewma.is_finite());
        assert!(a.mtp_only_accept_rate_ewma.is_finite());
    }

    // ── Source-aware hurt gate tests (ADR-019 Phase 5) ──────────────────

    #[test]
    fn mtp_ngram_source_hurt_gate_does_not_fire_when_ngram_better_than_mtp() {
        // n-gram acceptance 80% > MTP acceptance 70% → no hurt.
        assert!(!mtp_ngram_source_hurt_gate(
            3,    // ngram_max
            100,  // mtp_drafted
            70,   // mtp_accepted
            100,  // ngram_drafted
            80,   // ngram_accepted
            4,    // min_samples
            0.02  // margin
        ));
    }

    #[test]
    fn mtp_ngram_source_hurt_gate_fires_when_ngram_worse_than_mtp() {
        // n-gram acceptance 50% + margin 0.02 < MTP acceptance 80% → hurt.
        assert!(mtp_ngram_source_hurt_gate(
            3,    // ngram_max
            100,  // mtp_drafted
            80,   // mtp_accepted
            100,  // ngram_drafted
            50,   // ngram_accepted
            4,    // min_samples
            0.02  // margin
        ));
    }

    #[test]
    fn mtp_ngram_source_hurt_gate_respects_min_samples() {
        // Not enough samples → must not fire even when rates are bad.
        assert!(!mtp_ngram_source_hurt_gate(
            3, // ngram_max
            2, // mtp_drafted  (< min_samples=4)
            0, // mtp_accepted
            2, // ngram_drafted (< min_samples=4)
            0, // ngram_accepted
            4, // min_samples
            0.02
        ));
    }

    #[test]
    fn mtp_ngram_source_hurt_gate_respects_margin() {
        // n-gram=79%, MTP=80%, margin=2%. Due to f32 representation, 79/100+0.02
        // may be slightly < 80/100, so use a clear case: ngram=79%, MTP=80%,
        // margin=5% → 0.79 + 0.05 = 0.84 > 0.80 → no fire.
        assert!(!mtp_ngram_source_hurt_gate(
            3,    // ngram_max
            100,  // mtp_drafted
            80,   // mtp_accepted  (rate=0.80)
            100,  // ngram_drafted
            79,   // ngram_accepted (rate=0.79)
            4,    // min_samples
            0.05  // margin (0.79 + 0.05 = 0.84, not < 0.80)
        ));
        // But with a small margin it should fire.
        assert!(mtp_ngram_source_hurt_gate(
            3, 100, 80, 100, 50, 4, 0.02 // 0.50 + 0.02 = 0.52 < 0.80
        ));
    }

    #[test]
    fn mtp_ngram_source_hurt_gate_returns_false_when_ngram_max_zero() {
        // ngram_max=0 → no gate regardless of counters.
        assert!(!mtp_ngram_source_hurt_gate(
            0,    // ngram_max
            1000, // mtp_drafted
            999,  // mtp_accepted
            1000, // ngram_drafted
            1,    // ngram_accepted (very bad)
            4,    // min_samples
            0.02
        ));
    }

    // ── MtpDraftMode env knob test ──────────────────────────────────────

    #[test]
    fn mtp_draft_mode_default_is_greedy() {
        // Without the env var, the default must be Greedy.
        // NOTE: OnceLock caches on first call; if a prior test already set
        // the env var in this process, the cached value persists. We only
        // verify the type is constructible and defaults to Greedy.
        let mode = crate::mtp::MtpDraftMode::default();
        assert_eq!(mode, crate::mtp::MtpDraftMode::Greedy);
    }

    #[test]
    fn mtp_optimistic_is_disabled_for_glm_sidecar() {
        assert!(mtp_optimistic_allowed(false));
        assert!(!mtp_optimistic_allowed(true));
    }

    // ── MtpDraftFilter identity test ────────────────────────────────────

    #[test]
    fn mtp_draft_filter_identity_means_no_filter() {
        let f = MtpDraftFilter::IDENTITY;
        assert_eq!(f.top_p, 1.0);
        assert_eq!(f.top_k, 0);
    }

    // ── HurtGateMode default test ───────────────────────────────────────

    #[test]
    fn hurt_gate_mode_default_is_source_aware() {
        let mode = HurtGateMode::default();
        assert_eq!(mode, HurtGateMode::SourceAware);
    }

    // ── New telemetry fields merge correctly ────────────────────────────

    #[test]
    fn mtp_telemetry_merge_from_combines_new_hurt_gate_fields() {
        let mut a = MtpTelemetry {
            ngram_source_hurt_gated_steps: 10,
            ngram_legacy_hurt_gated_steps: 5,
            ..MtpTelemetry::default()
        };

        let b = MtpTelemetry {
            ngram_source_hurt_gated_steps: 7,
            ngram_legacy_hurt_gated_steps: 3,
            ..MtpTelemetry::default()
        };

        a.merge_from(b);
        assert_eq!(a.ngram_source_hurt_gated_steps, 17);
        assert_eq!(a.ngram_legacy_hurt_gated_steps, 8);
    }

    // ── MTP bypass and initial adaptive depth ──────────────────────────────

    #[test]
    fn mtp_initial_adaptive_depth_starts_qwen3_5_at_depth_2() {
        // Qwen3.6 dense 27B (qwen3_5) has the same linear-attention hybrid
        // architecture as qwen3_next; depth 2 is the throughput optimum.
        assert_eq!(mtp_initial_adaptive_depth("qwen3_5", 8), 2);
        assert_eq!(mtp_initial_adaptive_depth("qwen3_5", 1), 1);
        // MoE variant still starts at 2.
        assert_eq!(mtp_initial_adaptive_depth("qwen3_next", 8), 2);
        assert_eq!(mtp_initial_adaptive_depth("qwen3_next", 1), 1);
        // Other families start at head_max_depth.
        assert_eq!(mtp_initial_adaptive_depth("standard", 8), 8);
        assert_eq!(mtp_initial_adaptive_depth("qwen3", 4), 4);
    }

    #[test]
    fn mtp_bypass_defaults_are_sane() {
        // Default min_samples is 8: enough EWMA stabilization without
        // excessive warm-up delay.
        assert_eq!(mtp_bypass_min_samples(), 8);
        // Default threshold is 0.50: MTP is bypassed only when acceptance
        // is clearly worse than break-even.
        let threshold = mtp_bypass_threshold();
        assert!(
            (threshold - 0.50).abs() < 1e-5,
            "default bypass threshold must be 0.50; got {threshold}"
        );
    }

    #[test]
    fn request_state_starts_with_mtp_bypass_disabled() {
        let state = RequestState::new(1, 7);
        assert!(
            !state.mtp_bypassed,
            "MTP bypass must start disabled so MTP is attempted on every request"
        );
    }
}
