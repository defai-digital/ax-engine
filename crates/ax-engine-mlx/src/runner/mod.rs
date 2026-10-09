use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, LazyLock, OnceLock};

use parking_lot::Mutex;
use std::thread::{self, ThreadId};
use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, MlxStream, add, argmax, argpartition_axis, astype, async_eval, clear_cache,
    concatenate, divide, enable_compile, eval, max_recommended_working_set_size, multiply, power,
    reshape, set_cache_limit, set_memory_limit, set_wired_limit, slice, softmax, stack, sum_axis,
    take, take_along_axis,
};

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::scheduler::ExecutionMode;
use ax_engine_core::{
    DiffusionScheduleUpdate, EmbeddingPooling, ExecutionRunner, ExecutionStatus, KvWriteSummary,
    MultimodalPrefillAdapter, NativeModelArtifacts, NativeModelBindingSummary,
    ROUTE_DECISION_AX_MLX_GENERATION_KIND, ROUTE_DECISION_AX_MLX_GENERATION_WORK_UNIT,
    ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB, ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS, ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB, ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB, ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_CALLS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_ATTENTION_FALLBACKS, ROUTE_DECISION_AX_MLX_KV_PAGED_COW_COPIES,
    ROUTE_DECISION_AX_MLX_KV_PAGED_MATERIALIZE_US, ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_BLOCKS_USED,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_EXHAUSTION_FALLBACKS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SHARED_BLOCKS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_GROW_EVENTS,
    ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLAB_KIB, ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_SLABS,
    ROUTE_DECISION_AX_MLX_KV_QUANTIZED_LAYERS, ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_ROTATED_RING_LAYERS, ROUTE_DECISION_AX_MLX_KV_ROTATING_RING_SLACK,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS, ROUTE_DECISION_AX_MLX_LAYER_FORWARD_ROUTE,
    RequestExecutionUpdate, RequestId, RequestMultimodalInputs, RunnerInput, RunnerOutput,
    StopReason, upsert_route_decision,
};

use crate::batched_decode_certification::load_batched_decode_certification;
use crate::batched_decode_session::{
    BatchedDecodeCapabilities, BatchedDecodeSession, batched_decode_allow_uncertified,
    batched_decode_enabled, batched_decode_sampling_enabled, mtp_multirow_batch_enabled,
};
use crate::batched_sampling::{BatchedSamplingClass, argmax_batched, batched_sampling_class};
use crate::gemma4_assistant_mtp::{
    Gemma4AssistantMtpConfig, Gemma4AssistantMtpDisableReason, Gemma4AssistantMtpStatus,
    gemma4_assistant_mtp_debug_enabled, gemma4_assistant_mtp_max_depth_cap,
    resolve_gemma4_assistant_mtp_deep_gate, resolve_gemma4_assistant_mtp_first_gate,
};
use crate::generate::{
    CacheOnlyPrefillLayout, DirectPipelineTimings, advance_direct_pipeline_capturing_with_timings,
    advance_direct_pipeline_with_timings, chunked_prefill_cache_only,
    chunked_prefill_gemma4_unified_with_mtp_history_and_sampling_buffers,
    chunked_prefill_minicpm_v46_with_sampling_buffers,
    chunked_prefill_nemotron_omni_with_sampling_buffers,
    chunked_prefill_qwen3_vl_with_sampling_buffers,
    chunked_prefill_unlimited_ocr_with_sampling_buffers,
    chunked_prefill_with_deepseek_v4_mtp_history_and_sampling_buffers,
    chunked_prefill_with_mtp_history_and_sampling_buffers, chunked_prefill_with_sampling_buffers,
    decode_step, direct_pipeline_barrier_enabled, prepare_direct_pipeline_advance,
    sample_token_from_prefill_logits, start_direct_pipeline, start_direct_pipeline_capturing,
};
use crate::kv_block_pool::{
    FaBlockPoolConfig, FaBlockPoolError, SharedFaBlockPool, default_fa_block_pool_config,
    fa_block_pool_max_blocks_override, fa_kv_block_pool_enabled, fa_kv_block_sharing_enabled,
    fa_native_paged_attention_enabled,
};
use crate::kv_cache::{MlxKVCache, MlxKVCacheSerializeError, MlxKVCacheUsage};
use crate::model::{
    DecodeProfileSnapshot, DenseFfnFastpathSnapshot, Gemma4MoeProfileSnapshot,
    LinearAttentionProfileSnapshot, ModelConfig, MoeProfileSnapshot, PrefillProfileSnapshot,
    forward_all_positions_post_norm_last_lm_head, forward_all_positions_with_post_norm,
    forward_all_positions_with_post_norm_greedy, forward_all_positions_with_post_norm_ids,
    gemma4_assistant_draft_rope_position, replay_linear_attention_mtp_prefix,
    take_decode_profile_snapshot, take_dense_ffn_fastpath_snapshot,
    take_gemma4_moe_profile_snapshot, take_linear_attention_profile_snapshot,
    take_moe_profile_snapshot, take_prefill_profile_snapshot, try_whole_compiled_qwen_verify,
};
use crate::model::{prefill_batched_forward, supports_batched_prefill};
use crate::mtp::{
    deepseek_v4_mtp_draft_tokens_after_forced_prefix, deepseek_v4_mtp_draft_tokens_gated,
    deepseek_v4_mtp_warmup_cache, glm_mtp_draft_tokens_after_forced_prefix,
    glm_mtp_draft_tokens_gated, mtp_draft_tokens_after_forced_prefix, mtp_draft_tokens_gated,
    sequential_greedy_deepseek_v4_mtp_verify,
};
use crate::mtp_adaptive_gate::{
    AdaptiveStepSignals, MtpAdaptiveGateState, adaptive_gate_enabled_from_env,
    maybe_init_state as mtp_adaptive_maybe_init, next_gate_config_from_env, observe_step,
    resolve_mtp_gate_from_env,
};
use crate::ngram_accel::{
    NgramDraftOutcome, NgramDraftPolicy, NgramDraftRejection, NgramPolicyVariant, NgramTable,
    THINK_SOFT_CLOSE_PROBE_RANK, ThinkSoftCloseProbe, classify_prompt_class,
    ngram_accel_decode_step_with_sampling_buffers, ngram_feedback_policy,
    recompute_committed_prefix_with_argmax, revalidate_greedy_prefix_with_argmax,
    sequential_greedy_mtp_verify, single_decode_with_sampling_buffers_capturing,
};
use crate::sampling::{
    MlxSamplingParams, MlxSamplingRequest, TokenDistribution, Xorshift64, sample_categorical_into,
    sample_categorical_with_logprob_and_distribution, sample_residual_token_distribution,
    token_distribution,
};
use crate::speculation_profile::speculation_profile_from_env;
use crate::unlimited_ocr::preprocess_document_rgb_u8;
#[cfg(test)]
use crate::weights::{LayerWeights, QuantizedWeight};
use crate::weights::{ModelWeights, load_weights};

mod manifest_validation;
mod mtp_depth_controller;
mod mtp_model_policy;
mod mtp_ngram_gates;
mod mtp_profitability;
mod mtp_routing;
mod mtp_tuning;
mod pipeline;
mod prefix_cache;
mod runner_telemetry;
mod step_telemetry;
mod util;

use manifest_validation::*;
use mtp_depth_controller::*;
use mtp_model_policy::*;
use mtp_ngram_gates::*;
use mtp_profitability::*;
use mtp_routing::*;
use mtp_tuning::*;
use step_telemetry::*;

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
/// Historical permanent-disable threshold for linear-attention models after
/// repeated no-draft steps. Permanently killing n-gram on Qwen3.5 hybrid
/// models left S0 coding streams on pure AR for the whole request after the
/// first ~8 cold tokens — right when ring-buffer/code patterns would start
/// paying off multi-token verify thr.
///
/// Use a high threshold so the path stays on short cooldowns
/// (`LINEAR_NGRAM_PARTIAL_RETRY_INTERVAL`) instead of request-long disable.
/// Reenable probing still works if a future path sets permanent disable.
const LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD: u32 = u32::MAX;
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
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_SNAPSHOT_INCOMPLETE: &str =
    "ax_mlx_prefix_cache_blocked_snapshot_incomplete";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_MEDIA_IDENTITY: &str =
    "ax_mlx_prefix_cache_blocked_media_identity";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_RESTORE_ERROR: &str =
    "ax_mlx_prefix_cache_blocked_restore_error";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BLOCKED_ENTRY_TOO_LARGE: &str =
    "ax_mlx_prefix_cache_blocked_entry_too_large";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_STORES: &str = "ax_mlx_prefix_cache_stores";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_EVICTIONS: &str = "ax_mlx_prefix_cache_evictions";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_REUSED_TOKENS: &str = "ax_mlx_prefix_cache_reused_tokens";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_WARMUP_TOKENS: &str = "ax_mlx_prefix_cache_warmup_tokens";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_ENTRIES: &str = "ax_mlx_prefix_cache_entries";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_BYTES_KIB: &str = "ax_mlx_prefix_cache_bytes_kib";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_NATIVE_HITS: &str = "ax_mlx_prefix_cache_native_hits";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_NATIVE_STORES: &str = "ax_mlx_prefix_cache_native_stores";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_NATIVE_EVICTIONS: &str =
    "ax_mlx_prefix_cache_native_evictions";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DEMOTIONS: &str = "ax_mlx_prefix_cache_demotions";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DEMOTED_BYTES_KIB: &str =
    "ax_mlx_prefix_cache_demoted_bytes_kib";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DEMOTION_SKIPS: &str =
    "ax_mlx_prefix_cache_demotion_skips";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_HITS: &str = "ax_mlx_prefix_cache_disk_hits";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_MISSES: &str = "ax_mlx_prefix_cache_disk_misses";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_INSERTS: &str = "ax_mlx_prefix_cache_disk_inserts";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_INSERT_BYTES_KIB: &str =
    "ax_mlx_prefix_cache_disk_insert_bytes_kib";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_RESTORE_SOURCE: &str =
    "ax_mlx_prefix_cache_restore_source";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_READ_WALL_US: &str =
    "ax_mlx_prefix_cache_disk_read_wall_us";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_CHECKSUM_WALL_US: &str =
    "ax_mlx_prefix_cache_disk_checksum_wall_us";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_DESERIALIZE_WALL_US: &str =
    "ax_mlx_prefix_cache_disk_deserialize_wall_us";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_RESTORE_TOTAL_WALL_US: &str =
    "ax_mlx_prefix_cache_disk_restore_total_wall_us";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_BYTES_READ_KIB: &str =
    "ax_mlx_prefix_cache_disk_bytes_read_kib";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_ADMITTED: &str = "ax_mlx_prefix_cache_disk_admitted";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_ADMISSION_REJECTED: &str =
    "ax_mlx_prefix_cache_disk_admission_rejected";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_ADMISSION_REASON_CODE: &str =
    "ax_mlx_prefix_cache_disk_admission_reason_code";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_STORE_ENQUEUED: &str =
    "ax_mlx_prefix_cache_disk_store_enqueued";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_STORE_COMMITTED: &str =
    "ax_mlx_prefix_cache_disk_store_committed";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_STORE_COMMIT_FAILED: &str =
    "ax_mlx_prefix_cache_disk_store_commit_failed";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_STORE_DROPPED: &str =
    "ax_mlx_prefix_cache_disk_store_dropped";
const ROUTE_DECISION_AX_MLX_PREFIX_CACHE_DISK_FALLBACK_RECOMPUTE: &str =
    "ax_mlx_prefix_cache_disk_fallback_recompute";
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
    pub(crate) fn is_loaded(&self) -> bool {
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

/// Per-request mutable state persisted across prefill → decode steps.
struct RequestState {
    cache: MlxKVCache,
    prompt_prefix_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    cached_prefill_output_token: Option<u32>,
    /// Lazy cache clone captured mid-prefill at the largest block-aligned
    /// boundary of an unaligned prompt on a linear-attention model, where
    /// store-time `trim_to` cannot produce a prefix snapshot (conv/recurrent
    /// state does not roll back). `(boundary_token_len, cache_at_boundary)`;
    /// consumed by `store_prompt_prefix_snapshots` after the prefill item.
    prefill_boundary_snapshot: Option<(usize, MlxKVCache)>,
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
    /// Per-request rotating sliding-KV decision `(rotate, slack)`, latched
    /// once by `initialize_generation_state` at the prefill→decode boundary.
    /// `run()` re-applies it to the cache at the top of **every** subsequent
    /// step: rotation is irreversible for a request (a converted ring cannot
    /// grow back through the ordered path without scrambling token order),
    /// so the per-run session default must never clobber a latched decision.
    /// `None` until the first latch (prefill runs use the session default).
    rotating_sliding_latch: Option<(bool, usize)>,
    /// Pre-verified bonus tokens ready to serve without a model run.
    bonus_queue: VecDeque<u32>,
    /// Buffered tokens from the most recent diffusion block commit.
    /// DiffusionGemma generates `canvas_size` tokens per block; the runner
    /// drains them one at a time through the standard decode path.
    diffusion_block_queue: VecDeque<u32>,
    /// Schedule feedback for the engine after the latest monoblock generate.
    pending_diffusion_schedule: Option<DiffusionScheduleUpdate>,
    /// In-progress multi-step denoise workspace (Phase B). Present only when
    /// `AX_DIFFUSION_STEPS_PER_ENGINE_STEP` is set and a block is mid-denoise.
    diffusion_workspace: Option<crate::diffusion::DiffusionBlockWorkspace>,
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
    /// Async-scheduled greedy draft awaiting host extraction
    /// (`AX_MLX_MTP_ASYNC_DRAFT`). When `Some`, `mtp_pending_draft` is empty
    /// and is filled from these arrays at the start of the next decode cycle,
    /// overlapping the draft head's GPU forward with per-token host work.
    mtp_pending_draft_lazy: Option<crate::mtp::MtpLazyDraft>,
    /// Log-probabilities of `mtp_pending_draft` under the draft distribution.
    /// Used for rejection-sampling acceptance: accept draft[i] with probability
    /// min(1, p_target(draft[i]) / exp(mtp_pending_draft_log_probs[i])).
    /// MTP positions: softmax log-probs at draft_sampling.temperature (or T=1.0 for
    /// greedy-mode drafts after Phase 1).  N-gram hybrid positions: 0.0 (delta
    /// distribution, p_draft=1.0).  Empty for pure n-gram drafts (no MTP tail).
    mtp_pending_draft_log_probs: Vec<f32>,
    /// Temperature used when recording `mtp_pending_draft_log_probs` (DeepSeek V4
    /// think-aware draft T, Qwen/GLM draft_sampling T, or 1.0 for greedy). Accept
    /// on the *next* step must reuse this value for rejection rescale — do not
    /// recompute from a possibly-changed think state (DI-DS-MTP).
    mtp_pending_draft_log_prob_temperature: Option<f32>,
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
    /// Argmax of `mtp_skip_logits`, computed lazily at capture inside the
    /// same async_eval batch. Greedy skip cycles read this 1-element array
    /// for the primary token — `sample_logit_row`'s greedy shortcut trusts
    /// its `argmax_tok` argument verbatim, so passing a placeholder emits
    /// that placeholder as a real token (the token-0 corruption class).
    mtp_skip_argmax: Option<MlxArray>,
    /// Skip-state hidden: post-norm hidden at the same committed position.
    /// Used as `main_hidden` for the MTP head when skip-state is active.
    mtp_skip_hidden: Option<MlxArray>,
    /// Cumulative MTP draft/accept counters for benchmark telemetry.
    mtp_telemetry: MtpTelemetry,
    /// Cumulative Gemma4 assistant-MTP counters for route metadata.
    gemma4_assistant_mtp_telemetry: Gemma4AssistantMtpTelemetry,
    /// Once the formal cycle-continuation guard fires on a request, latch pure-
    /// direct sequential verify for the rest of the generation. Residual
    /// multi-token identity failures often appear after the first loop is
    /// detected; latching is exactness-preserving (never routes away from
    /// sequential) under `SEQUENTIAL_ORACLE=0`.
    gemma_mtp_cycle_latched: bool,
    /// Number of consecutive decode steps where accept_count == 0.
    /// Used by `mtp_next_adaptive_depth` to progressively lower the depth floor.
    mtp_consecutive_misses: u32,
    /// Per-request MTP bypass: once the acceptance EWMA drops below the bypass
    /// threshold with sufficient samples, MTP is disabled for the rest of the
    /// request and all decode steps use the direct single-token path.
    mtp_bypassed: bool,
    /// The request transferred its committed target state into the shared
    /// direct-decode session because a compatible multirow cohort formed.
    /// MTP state is discarded and cannot be restored until a new generation
    /// starts. Session KV is normally authoritative; scheduler deferral may
    /// write it back to the private cache without clearing this direct-only
    /// latch.
    mtp_suspended_for_batched_decode: bool,
    /// Measured direct-vs-MTP cost policy for exact greedy Qwen depth one.
    mtp_profitability: MtpProfitabilityState,
    /// Measured cost-model draft-depth controller (opt-in via
    /// `AX_MLX_MTP_COST_MODEL_DEPTH`); supersedes the streak controller when
    /// enabled and parks speculation for the request when direct decode wins.
    mtp_cost_depth: MtpCostDepthController,
    /// Decode steps taken since the cost controller's last decision. The
    /// controller's cycle wall is only a clean depth-cost sample when exactly
    /// one step ran in it (any fallback / think-window / n-gram step in
    /// between spans foreign work).
    mtp_cost_steps_since_decision: u32,
    /// Per-request latch: once auto-optimistic activates (EWMA ≥ 0.99),
    /// it stays latched until the argmax-based EWMA drops below 0.85.
    /// Hysteresis prevents oscillation because argmax acceptance is strictly
    /// stricter than stochastic acceptance (draft tokens that pass
    /// p_target/p_draft rejection sampling may not be the argmax token),
    /// so the EWMA shifts metric upon activation.
    auto_optimistic_active: bool,
    /// Online adaptive MTP draft gate state (default OFF / low-T auto only).
    /// See `docs/designs/mtp-embed-perf-sprint-2026-07-16.md`.
    mtp_adaptive_gate: Option<MtpAdaptiveGateState>,
    /// Last resolved draft gate ×1000 for route telemetry.
    mtp_draft_gate_x1000: u32,
    /// ResolutionSource route code for last draft gate.
    mtp_draft_gate_source: u32,
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
    /// Tokens emitted while inside an open think block (budget controller).
    think_emitted_tokens: u32,
    /// Recomputed each decode step: inside an open think block within the
    /// soft-close window ahead of the answer reserve / think cap. While
    /// armed, decode routes through logits-materializing single decode and
    /// rank-probes the think-close token before sampling.
    think_soft_close_armed: bool,
    /// Per-request n-gram self-tune: tracks draft tokens and accepted tokens
    /// for this request.  After warmup, if acceptance rate falls below the
    /// threshold, n-gram is disabled for the rest of the request (mirrors
    /// lightning-mlx 0.7.0 `NgramRequestState._self_tune_disabled`).
    ngram_self_tune: NgramSelfTuneState,
    /// Remaining steps to keep n-gram gated after a utility hurt decision.
    mtp_ngram_utility_hysteresis_remaining: u32,
    /// Request-local Flash Next MTP candidate state. The live cursor object
    /// is never shared with the generic MTP head cache and never aliased
    /// across requests; only the cursor's serialized form rides along in
    /// portable prefix snapshots (via
    /// `Qwen4ExpDraftCursor::prefix_snapshot_parts`).
    flash_next_mtp: FlashNextMtpRequestState,
}

/// Draft history for the Flash Next candidate plus its route counters.
///
/// The cursor is created by a cold prefill (empty cache) or reconstructed
/// from a restored prefix snapshot's serialized sidecar payload, and then
/// advanced together with the authoritative trunk. Any trunk advance that
/// bypasses the cursor without a catch-up buffer (a restore without a usable
/// payload, sampled prefill, an exhausted budget) drops it, and the request
/// decodes direct for the rest of its generation.
#[derive(Default)]
struct FlashNextMtpRequestState {
    cursor: Option<crate::model::qwen4_exp_mtp::Qwen4ExpDraftCursor>,
    /// Draft cursor eagerly decoded from a restored prefix snapshot's
    /// sidecar payload at L1-restore time, pending installation by
    /// `prepare_flash_next_prefill_cursor` later in the same request. A
    /// take-once slot: whichever decision arm runs consumes it, so a stale
    /// pending cursor can never leak into a later prefill quantum.
    pending_restored_cursor: Option<crate::model::qwen4_exp_mtp::Qwen4ExpDraftCursor>,
    /// `Some` while the cursor is paused: a blocked step kept it, the trunk is
    /// advancing direct, and every token the trunk consumes has its stream row
    /// buffered for the catch-up that ends the pause.
    paused: Option<FlashNextCursorPauseBuffer>,
    /// Sticky per-request latch: the Flash Next route has blocked at least one
    /// decode step, so steps it does not serve decode on the greedy direct
    /// double buffer instead of the generic n-gram route.
    blocked_direct: bool,
    /// Emitted tokens since the last MLX buffer-cache clear.
    emitted_since_clear: u32,
    telemetry: FlashNextMtpTelemetry,
}

/// Catch-up rows buffered while a Flash Next cursor is paused.
///
/// Entry `(token, row)` pairs a token the trunk consumed with the stream row
/// that consuming it produced. `Qwen4ExpDraftCursor::absorb` pairs `rows[i]`
/// with `tokens[i + 1]` and adopts the last row as the head's next preceding
/// hidden state, so a buffer holding exactly the tokens the trunk consumed
/// since the pause began — in order — advances the head to the same boundary.
///
/// Rows arrive already materialized by the forward that produced them; the
/// buffer never holds a lazy view of the decode graph.
#[derive(Default)]
struct FlashNextCursorPauseBuffer {
    rows: Vec<(u32, MlxArray)>,
}

/// Paused-catch-up row cap. One row is ~16 KiB on Flash Next; 256 of them
/// bound the buffer at a few MiB, which covers a full think window while
/// keeping a pathological window from growing without limit.
const FLASH_NEXT_CURSOR_PAUSE_ROW_CAP: usize = 256;

impl FlashNextCursorPauseBuffer {
    /// Buffer one consumed token's stream row. `false` means the cap is
    /// reached and the caller must fail closed instead of keeping a gap.
    fn push(&mut self, token: u32, row: MlxArray) -> bool {
        if self.rows.len() >= FLASH_NEXT_CURSOR_PAUSE_ROW_CAP {
            return false;
        }
        self.rows.push((token, row));
        true
    }

    fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The consumed tokens, in trunk order, and their rows stacked into the
    /// `[1, count, width]` tensor `absorb` takes. Fails closed when any row is
    /// not the `[1, 1, width]` stream row the others share: a foreign row
    /// would panic inside the concatenate instead of dropping the cursor.
    fn try_absorb_parts(&self) -> Result<(Vec<u32>, MlxArray), String> {
        let tokens: Vec<u32> = self.rows.iter().map(|(token, _)| *token).collect();
        let Some((_, first)) = self.rows.first() else {
            return Err("Flash Next pause absorb with no rows".into());
        };
        let expected = first.shape();
        for (_, row) in &self.rows {
            let shape = row.shape();
            if shape.len() != 3 || shape[0] != 1 || shape[1] != 1 || shape != expected {
                return Err(format!(
                    "Flash Next pause absorb row shape {shape:?} does not match {expected:?}"
                ));
            }
        }
        let rows = self
            .rows
            .iter()
            .map(|(_, row)| row)
            .collect::<Vec<&MlxArray>>();
        Ok((tokens, concatenate(&rows, 1, None)))
    }
}

impl FlashNextMtpRequestState {
    /// Drop the draft history. Counts only a live cursor so repeated fallback
    /// steps after the first drop do not inflate the counter. The catch-up
    /// buffer goes with it: buffered rows are only meaningful for the cursor
    /// that consumed them.
    fn drop_cursor(&mut self) {
        self.paused = None;
        if self.cursor.take().is_some() {
            self.telemetry.cursor_dropped = self.telemetry.cursor_dropped.saturating_add(1);
        }
    }

    /// Buffer one consumed token's stream row while the cursor is paused.
    ///
    /// Overflow drops the cursor: a gap in the row history would misalign the
    /// head, so no step may extend a buffer that can never be flushed. Callers
    /// hold the invariant that a pause only exists while a cursor is live.
    fn extend_pause(&mut self, token: u32, row: MlxArray) -> bool {
        let buffer = self.paused.get_or_insert_with(Default::default);
        if buffer.push(token, row) {
            return true;
        }
        self.telemetry.cursor_pause_overflows =
            self.telemetry.cursor_pause_overflows.saturating_add(1);
        self.drop_cursor();
        false
    }

    /// Whether the live cursor may ride along in a prefix snapshot's sidecar
    /// payload. A paused cursor is mid-catch-up: its draft state is behind the
    /// snapshot trunk and its buffered rows have no payload representation, so
    /// a restore must replay instead.
    fn prefix_snapshot_eligible(&self) -> bool {
        self.cursor.is_some() && self.paused.is_none()
    }

    /// Tokens and rows for one catch-up `absorb` call, when the buffer has rows.
    fn take_pause_absorb_parts(&mut self) -> Option<Result<(Vec<u32>, MlxArray), String>> {
        let buffer = self.paused.take()?;
        if buffer.is_empty() {
            return None;
        }
        Some(buffer.try_absorb_parts())
    }
}

/// Cumulative per-request Flash Next MTP route counters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FlashNextMtpTelemetry {
    /// Cold prefills that created a draft cursor.
    cursor_initialized: u32,
    /// Prefill quanta that resumed a cached prefix without a usable draft
    /// cursor: no cursor existed, or a stashed payload failed to decode,
    /// rebind, or verify aligned. A restored payload that decoded cleanly
    /// counts as `cursor_restored` instead.
    resumed_without_cursor: u32,
    /// Prefill quanta that restored a working draft cursor from a prefix
    /// snapshot's sidecar payload.
    cursor_restored: u32,
    /// Prefill quanta whose cursor was discarded by a draft absorb failure.
    prefill_absorb_failures: u32,
    /// Live cursors discarded by fallback, misalignment, or step errors.
    cursor_dropped: u32,
    /// Decode steps served direct while a live cursor was paused for a later
    /// catch-up.
    cursor_paused_steps: u32,
    /// Paused cursors that caught up through the buffered rows and resumed
    /// proposing.
    cursor_resumed: u32,
    /// Paused cursors discarded because the catch-up buffer filled up.
    cursor_pause_overflows: u32,
    /// Verified cursor steps.
    verified_steps: u32,
    /// Verified steps whose draft token was accepted.
    accepted_steps: u32,
    /// Decode steps served direct while the candidate was requested.
    direct_fallback_steps: u32,
    /// `direct_fallback_steps` split by the block or error reason that forced
    /// direct decode, indexed by `FlashNextMtpFallbackReason::index`. Summing
    /// the slots always equals `direct_fallback_steps`.
    direct_fallback_by_reason: [u32; FLASH_NEXT_MTP_FALLBACK_REASON_COUNT],
    /// Cursor steps that returned an error before publishing any state.
    step_errors: u32,
    /// Batched length-2 correction forward time across verified steps.
    correction_wall_us: u32,
    /// Bonus-token forward time. Zero on the batched acceptance path.
    bonus_wall_us: u32,
    /// Rejection singleton forward time. Zero on acceptance.
    rejection_wall_us: u32,
    /// Tokens returned by verified cursor steps, including the next primary.
    emitted_tokens: u32,
    /// Minimum correction-row top-two margin over verified steps, milli-logits.
    /// `u32::MAX` until the first verified step.
    min_correction_margin_milli: u32,
}

impl Default for FlashNextMtpTelemetry {
    fn default() -> Self {
        Self {
            cursor_initialized: 0,
            resumed_without_cursor: 0,
            cursor_restored: 0,
            prefill_absorb_failures: 0,
            cursor_dropped: 0,
            cursor_paused_steps: 0,
            cursor_resumed: 0,
            cursor_pause_overflows: 0,
            verified_steps: 0,
            accepted_steps: 0,
            direct_fallback_steps: 0,
            direct_fallback_by_reason: [0; FLASH_NEXT_MTP_FALLBACK_REASON_COUNT],
            step_errors: 0,
            correction_wall_us: 0,
            bonus_wall_us: 0,
            rejection_wall_us: 0,
            emitted_tokens: 0,
            min_correction_margin_milli: u32::MAX,
        }
    }
}

impl FlashNextMtpTelemetry {
    fn merge_from(&mut self, other: Self) {
        self.cursor_initialized = self
            .cursor_initialized
            .saturating_add(other.cursor_initialized);
        self.resumed_without_cursor = self
            .resumed_without_cursor
            .saturating_add(other.resumed_without_cursor);
        self.cursor_restored = self.cursor_restored.saturating_add(other.cursor_restored);
        self.prefill_absorb_failures = self
            .prefill_absorb_failures
            .saturating_add(other.prefill_absorb_failures);
        self.cursor_dropped = self.cursor_dropped.saturating_add(other.cursor_dropped);
        self.cursor_paused_steps = self
            .cursor_paused_steps
            .saturating_add(other.cursor_paused_steps);
        self.cursor_resumed = self.cursor_resumed.saturating_add(other.cursor_resumed);
        self.cursor_pause_overflows = self
            .cursor_pause_overflows
            .saturating_add(other.cursor_pause_overflows);
        self.verified_steps = self.verified_steps.saturating_add(other.verified_steps);
        self.accepted_steps = self.accepted_steps.saturating_add(other.accepted_steps);
        self.direct_fallback_steps = self
            .direct_fallback_steps
            .saturating_add(other.direct_fallback_steps);
        for (slot, other_slot) in self
            .direct_fallback_by_reason
            .iter_mut()
            .zip(other.direct_fallback_by_reason)
        {
            *slot = slot.saturating_add(other_slot);
        }
        self.step_errors = self.step_errors.saturating_add(other.step_errors);
        self.correction_wall_us = self
            .correction_wall_us
            .saturating_add(other.correction_wall_us);
        self.bonus_wall_us = self.bonus_wall_us.saturating_add(other.bonus_wall_us);
        self.rejection_wall_us = self
            .rejection_wall_us
            .saturating_add(other.rejection_wall_us);
        self.emitted_tokens = self.emitted_tokens.saturating_add(other.emitted_tokens);
        self.min_correction_margin_milli = self
            .min_correction_margin_milli
            .min(other.min_correction_margin_milli);
    }

    fn append_route_decisions(self, decisions: &mut impl RouteDecisionSink) {
        for (key, value) in [
            (
                "ax_mlx_flash_next_mtp_cursor_initialized",
                self.cursor_initialized,
            ),
            (
                "ax_mlx_flash_next_mtp_resumed_without_cursor",
                self.resumed_without_cursor,
            ),
            (
                "ax_mlx_flash_next_mtp_cursor_restored",
                self.cursor_restored,
            ),
            (
                "ax_mlx_flash_next_mtp_prefill_absorb_failures",
                self.prefill_absorb_failures,
            ),
            ("ax_mlx_flash_next_mtp_cursor_dropped", self.cursor_dropped),
            (
                "ax_mlx_flash_next_mtp_cursor_paused_steps",
                self.cursor_paused_steps,
            ),
            ("ax_mlx_flash_next_mtp_cursor_resumed", self.cursor_resumed),
            (
                "ax_mlx_flash_next_mtp_cursor_pause_overflows",
                self.cursor_pause_overflows,
            ),
            ("ax_mlx_flash_next_mtp_verified_steps", self.verified_steps),
            ("ax_mlx_flash_next_mtp_accepted_steps", self.accepted_steps),
            (
                "ax_mlx_flash_next_mtp_direct_fallback_steps",
                self.direct_fallback_steps,
            ),
            ("ax_mlx_flash_next_mtp_step_errors", self.step_errors),
            (
                "ax_mlx_flash_next_mtp_correction_wall_us",
                self.correction_wall_us,
            ),
            ("ax_mlx_flash_next_mtp_bonus_wall_us", self.bonus_wall_us),
            (
                "ax_mlx_flash_next_mtp_rejection_wall_us",
                self.rejection_wall_us,
            ),
            ("ax_mlx_flash_next_mtp_emitted_tokens", self.emitted_tokens),
            (
                "ax_mlx_flash_next_mtp_min_correction_margin_milli",
                self.min_correction_margin_milli,
            ),
        ] {
            decisions.upsert_route_decision(key, value);
        }
        for reason in FlashNextMtpFallbackReason::ALL {
            decisions.upsert_route_decision(
                reason.route_key(),
                self.direct_fallback_by_reason[reason.index()],
            );
        }
    }
}

/// Which block or error condition forced a Flash Next decode step to run
/// direct while the cursor candidate was requested. One counter per reason so
/// `/metrics` attributes the fallback instead of reporting a single total.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashNextMtpFallbackReason {
    /// Sampling, repetition processors, or a missing request context.
    NotStrictGreedy,
    /// Think soft-close probing or a due budget close.
    ThinkControl,
    /// A lazy direct-pipeline token already occupies the next position.
    PendingDirect,
    /// No output budget remains for this step.
    NoBudget,
    /// No cursor, or its history does not end at the trunk boundary.
    CursorUnavailable,
    /// Admission passed but a trunk, head or cursor component was absent.
    ComponentsUnavailable,
    /// The cursor step returned an error before publishing state.
    StepError,
}

const FLASH_NEXT_MTP_FALLBACK_REASON_COUNT: usize = 7;

impl FlashNextMtpFallbackReason {
    const ALL: [Self; FLASH_NEXT_MTP_FALLBACK_REASON_COUNT] = [
        Self::NotStrictGreedy,
        Self::ThinkControl,
        Self::PendingDirect,
        Self::NoBudget,
        Self::CursorUnavailable,
        Self::ComponentsUnavailable,
        Self::StepError,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    /// Route key consumed by the server's per-step route-decision accumulation.
    const fn route_key(self) -> &'static str {
        match self {
            Self::NotStrictGreedy => "ax_mlx_flash_next_mtp_direct_fallback_not_strict_greedy",
            Self::ThinkControl => "ax_mlx_flash_next_mtp_direct_fallback_think_control",
            Self::PendingDirect => "ax_mlx_flash_next_mtp_direct_fallback_pending_direct",
            Self::NoBudget => "ax_mlx_flash_next_mtp_direct_fallback_no_budget",
            Self::CursorUnavailable => "ax_mlx_flash_next_mtp_direct_fallback_cursor_unavailable",
            Self::ComponentsUnavailable => {
                "ax_mlx_flash_next_mtp_direct_fallback_components_unavailable"
            }
            Self::StepError => "ax_mlx_flash_next_mtp_direct_fallback_step_error",
        }
    }
}

impl FlashNextMtpDecodeBlock {
    /// Reason bucket used for route accounting. A paused step is not a distinct
    /// reason: it serves direct because the cursor cannot verify this step — it
    /// is still catching up to the trunk. The pause counters carry that state.
    const fn fallback_reason(self) -> FlashNextMtpFallbackReason {
        match self {
            Self::NotStrictGreedy => FlashNextMtpFallbackReason::NotStrictGreedy,
            Self::ThinkControl => FlashNextMtpFallbackReason::ThinkControl,
            Self::PendingDirect => FlashNextMtpFallbackReason::PendingDirect,
            Self::NoBudget => FlashNextMtpFallbackReason::NoBudget,
            Self::CursorUnavailable | Self::CursorPaused => {
                FlashNextMtpFallbackReason::CursorUnavailable
            }
        }
    }
}

/// How a blocked Flash Next decode step treats the live draft cursor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashNextMtpFallbackDisposition {
    /// Discard the cursor: the request decodes direct for the rest of its
    /// generation.
    Drop,
    /// Keep the cursor without proposing: the trunk advances direct while the
    /// rows of every token it consumes are buffered for a later catch-up.
    RetainPaused,
}

/// Disposition of one blocked Flash Next decode step. Pure so the reason ×
/// kill-switch truth table is testable without model weights.
///
/// `sticky_fallback` is the `AX_MLX_FLASH_NEXT_STICKY_FALLBACK=1` legacy
/// policy: every block drops the cursor, exactly as before the paused state
/// existed.
///
/// Think control and a pending direct token are transient: the next step can
/// propose again once the window closes or the lazy token drains, so the
/// cursor is worth retaining. A step error drops on the first failure — the
/// transaction contract publishes nothing on error, but the graph may already
/// be half applied, so retry is not worth the risk. No budget, non-greedy
/// sampling, a missing cursor and a failed capture are all terminal for the
/// candidate: dropping keeps the request on a route that cannot misalign the
/// head.
const fn flash_next_mtp_fallback_disposition(
    reason: FlashNextMtpFallbackReason,
    sticky_fallback: bool,
) -> FlashNextMtpFallbackDisposition {
    if sticky_fallback {
        return FlashNextMtpFallbackDisposition::Drop;
    }
    match reason {
        FlashNextMtpFallbackReason::ThinkControl | FlashNextMtpFallbackReason::PendingDirect => {
            FlashNextMtpFallbackDisposition::RetainPaused
        }
        FlashNextMtpFallbackReason::NotStrictGreedy
        | FlashNextMtpFallbackReason::NoBudget
        | FlashNextMtpFallbackReason::CursorUnavailable
        | FlashNextMtpFallbackReason::ComponentsUnavailable
        | FlashNextMtpFallbackReason::StepError => FlashNextMtpFallbackDisposition::Drop,
    }
}

/// Why a Flash Next decode step cannot use the draft cursor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashNextMtpDecodeBlock {
    /// Sampling, repetition processors, or a missing request context.
    NotStrictGreedy,
    /// Think soft-close probing or a due budget close needs single-token
    /// emission so a forced close cannot strand a committed draft token.
    ThinkControl,
    /// A lazy direct-pipeline token already occupies the next position.
    PendingDirect,
    /// No output budget remains for this step.
    NoBudget,
    /// No cursor, or its history does not end at the trunk boundary.
    CursorUnavailable,
    /// The cursor is retained but deliberately behind the trunk: this step
    /// continues the paused catch-up instead of proposing.
    CursorPaused,
}

/// Decode-time admission for the Flash Next cursor. Pure so the gate order is
/// testable without model weights.
///
/// `cursor_paused` is the state *before* this step. A paused cursor is behind
/// the trunk by construction, so the stale-gap reason means "keep pausing"
/// rather than "drop": the pause ends only when every other gate clears.
#[allow(clippy::too_many_arguments)]
fn flash_next_mtp_decode_block(
    strict_greedy: bool,
    think_soft_close_armed: bool,
    think_budget_close_due: bool,
    pending_direct: bool,
    remaining_output: usize,
    cursor_paused: bool,
    cursor_aligned: bool,
) -> Option<FlashNextMtpDecodeBlock> {
    if !strict_greedy {
        Some(FlashNextMtpDecodeBlock::NotStrictGreedy)
    } else if think_soft_close_armed || think_budget_close_due {
        Some(FlashNextMtpDecodeBlock::ThinkControl)
    } else if pending_direct {
        Some(FlashNextMtpDecodeBlock::PendingDirect)
    } else if remaining_output == 0 {
        Some(FlashNextMtpDecodeBlock::NoBudget)
    } else if cursor_paused {
        Some(FlashNextMtpDecodeBlock::CursorPaused)
    } else if !cursor_aligned {
        Some(FlashNextMtpDecodeBlock::CursorUnavailable)
    } else {
        None
    }
}

/// What a prefill quantum must do with the request's draft cursor before the
/// trunk advances.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlashNextPrefillCursorAction {
    /// Cold prefill: start a fresh cursor (replacing any stale one).
    Initialize,
    /// Continuation whose cursor ends exactly at the trunk boundary.
    Keep,
    /// Continuation whose cursor is stale for this cache.
    Drop,
    /// Continuation of a restored prefix whose stashed cursor payload
    /// decoded, rebound, and verified aligned against the just-restored
    /// trunk.
    ResumeWithRestoredCursor,
    /// Continuation of a restored prefix: no recapture, decode falls back.
    ResumeWithoutCursor,
}

const fn flash_next_prefill_cursor_action(
    cache_seq_len: usize,
    has_cursor: bool,
    cursor_aligned: bool,
    restored_cursor_aligned: bool,
) -> FlashNextPrefillCursorAction {
    if cache_seq_len == 0 {
        FlashNextPrefillCursorAction::Initialize
    } else if has_cursor && cursor_aligned {
        FlashNextPrefillCursorAction::Keep
    } else if has_cursor {
        FlashNextPrefillCursorAction::Drop
    } else if restored_cursor_aligned {
        FlashNextPrefillCursorAction::ResumeWithRestoredCursor
    } else {
        FlashNextPrefillCursorAction::ResumeWithoutCursor
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum NgramRequestDisableReason {
    #[default]
    None,
    ShortOutputBudget,
    LinearNoDraft,
    LinearInitialNoDraft,
}

struct PromptPrefixSnapshotStoreOptions<'a> {
    linear_boundary_snapshot: Option<&'a (usize, MlxKVCache)>,
    prefill_completes_prompt: bool,
    greedy_prefill_output_token: Option<u32>,
    cold_prefill_us: u64,
    /// WS-M3 media digests folded into prefix keys (empty for text-only).
    /// `None` = media present but no digest available; the store fails
    /// closed instead of writing a text-only key for a media prompt.
    media_key: Option<&'a str>,
}

impl RequestState {
    #[cfg(test)]
    fn new(num_layers: usize, seed: u64, fa_block_pool_config: Option<FaBlockPoolConfig>) -> Self {
        Self::new_with_shared_fa_pool(num_layers, Some(seed), seed, fa_block_pool_config, None)
    }

    fn new_with_shared_fa_pool(
        num_layers: usize,
        explicit_seed: Option<u64>,
        fallback_seed: u64,
        fa_block_pool_config: Option<FaBlockPoolConfig>,
        shared_fa_block_pool: Option<SharedFaBlockPool>,
    ) -> Self {
        let cache = if let Some(pool) = shared_fa_block_pool {
            MlxKVCache::new_with_shared_fa_block_pool(num_layers, pool)
        } else if let Some(config) = fa_block_pool_config {
            MlxKVCache::new_with_fa_block_pool(num_layers, config)
        } else {
            MlxKVCache::new_contiguous(num_layers)
        };
        Self {
            cache,
            prompt_prefix_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            cached_prefill_output_token: None,
            prefill_boundary_snapshot: None,
            ngram: NgramTable::new(),
            rng: crate::sampling::request_rng(explicit_seed, fallback_seed),
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
            rotating_sliding_latch: None,
            bonus_queue: VecDeque::new(),
            diffusion_block_queue: VecDeque::new(),
            pending_diffusion_schedule: None,
            diffusion_workspace: None,
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
            mtp_pending_draft_lazy: None,
            mtp_pending_draft_log_probs: Vec::new(),
            mtp_pending_draft_log_prob_temperature: None,
            mtp_pending_draft_distributions: Vec::new(),
            mtp_pending_draft_sources: Vec::new(),
            ngram_draft_policy: None,
            mtp_target_prob_workspace: MtpTargetProbWorkspace::default(),
            mtp_adaptive_max_depth: 0,
            mtp_skip_logits: None,
            mtp_skip_argmax: None,
            mtp_skip_hidden: None,
            mtp_telemetry: MtpTelemetry::default(),
            gemma4_assistant_mtp_telemetry: Gemma4AssistantMtpTelemetry::default(),
            gemma_mtp_cycle_latched: false,
            mtp_consecutive_misses: 0,
            mtp_bypassed: false,
            mtp_suspended_for_batched_decode: false,
            mtp_profitability: MtpProfitabilityState::default(),
            mtp_cost_depth: MtpCostDepthController::default(),
            mtp_cost_steps_since_decision: 0,
            auto_optimistic_active: false,
            mtp_adaptive_gate: None,
            mtp_draft_gate_x1000: 0,
            mtp_draft_gate_source: 0,
            mtp_prefill_hidden: None,
            mtp_prefill_history_tokens: Vec::new(),
            ngram_in_think: false,
            think_emitted_tokens: 0,
            think_soft_close_armed: false,
            ngram_self_tune: NgramSelfTuneState::default(),
            mtp_ngram_utility_hysteresis_remaining: 0,
            flash_next_mtp: FlashNextMtpRequestState::default(),
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
        if !sampling.uses_logits_processors() {
            return Vec::new();
        }

        let total_len = self
            .prompt_prefix_tokens
            .len()
            .saturating_add(additional_prompt_tokens.len())
            .saturating_add(self.generated_tokens.len());
        let repetition_keep = if sampling.uses_repetition_penalty() {
            sampling
                .repetition_context_size
                .map(|size| size as usize)
                .unwrap_or(total_len)
        } else {
            0
        };
        let ngram_keep = if sampling.uses_no_repeat_ngram() {
            (sampling.ngram_window as usize).max(sampling.no_repeat_ngram_size as usize)
        } else {
            0
        };
        let keep_len = repetition_keep.max(ngram_keep).min(total_len);
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

/// Drop every unverified proposal while preserving committed target and MTP
/// head cache state. Used when the measured-cost policy latches a bypass.
fn clear_pending_mtp_proposal(state: &mut RequestState) {
    state.mtp_pending_draft.clear();
    state.mtp_pending_draft_lazy = None;
    state.mtp_pending_draft_log_probs.clear();
    state.mtp_pending_draft_log_prob_temperature = None;
    state.mtp_pending_draft_distributions.clear();
    state.mtp_pending_draft_sources.clear();
    state.ngram_draft_policy = None;
    state.mtp_skip_logits = None;
    state.mtp_skip_argmax = None;
    state.mtp_skip_hidden = None;
}

/// A shared batched session takes its feed token from the core scheduler and
/// owns all model lookahead while the row is resident. Private pipeline/feed
/// latches must not survive that handoff: `next_model_last_token` otherwise
/// takes precedence over the scheduler token after a deferred row is written
/// back, and a non-zero direct-pipeline count would resume at the wrong stage.
fn clear_private_decode_latches_for_batched_session(state: &mut RequestState) {
    state.pending_direct = None;
    state.direct_pipeline_emitted_tokens = 0;
    state.next_model_last_token = None;
}

fn restore_private_cache_after_batched_writeback(state: &mut RequestState, cache: MlxKVCache) {
    state.cache = cache;
    clear_private_decode_latches_for_batched_session(state);
}

/// Transfer an MTP request to direct batched decode. Only committed target
/// state survives: drafts are unverified, and the MTP head cache is not used by
/// the target-only cohort. The caller seeds the full target cache into the
/// batched session immediately after this transition.
fn suspend_mtp_for_batched_decode(state: &mut RequestState) {
    state.flash_next_mtp.drop_cursor();
    clear_pending_mtp_proposal(state);
    state.mtp_cache = None;
    state.mtp_decode_count = 0;
    state.mtp_prefill_hidden = None;
    state.mtp_prefill_history_tokens.clear();
    clear_private_decode_latches_for_batched_session(state);
    state.mtp_suspended_for_batched_decode = true;
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
/// The final `bool` is `has_dense_head`: when true the Dense head projection
/// is fused into the compiled graph, so the closure output shape differs
/// from the raw hidden-state path and the two must not share a cache slot.
type EmbedCompileKey = (ThreadId, usize, Option<usize>, bool);

/// Cache key for the batched embedding-forward compiled closure.
/// `target_positions` is baked into the trace, so two batches with the same
/// `(thread_id, batch_size, max_len)` but different per-sequence target
/// positions hit distinct keys. The trailing `bool` is `has_dense_head`
/// (same rationale as `EmbedCompileKey`).
type EmbedBatchCompileKey = (ThreadId, usize, usize, Option<Vec<usize>>, bool);

/// Cache key for mean-pool compiled embedding closures.
/// (thread_id, batch_size, max_seq_len)
type EmbedMeanPoolCompileKey = (ThreadId, usize, usize);

fn should_run_load_time_generation_warmup(
    is_encoder_embed_family: bool,
    reused_shared_weights: bool,
    expert_streaming_active: bool,
) -> bool {
    !is_encoder_embed_family && !reused_shared_weights && !expert_streaming_active
}

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
    /// Chunk size used for cold prefill (no prefix-cache restore).
    ///
    /// For MLA, defaults to the same value as `prefill_chunk` (R2: one shape
    /// trail for store producers and cold baselines). Opt into a larger cold
    /// chunk with `AX_MLX_MLA_COLD_PREFILL_CHUNK` only for throughput
    /// experiments — that dual path can re-open warm_extend token drift.
    /// Non-MLA models keep the caller-supplied larger chunk here.
    cold_prefill_chunk: usize,
    /// When set, new requests build an FA block-pool `MlxKVCache`.
    /// Geometry is aligned to session `KvManager` via
    /// [`Self::align_fa_block_pool_to_kv`] after construction.
    fa_block_pool_config: Option<FaBlockPoolConfig>,
    /// One runner-wide allocator when the second sharing gate is enabled.
    /// `None` preserves per-request private pools (or the contiguous path).
    shared_fa_block_pool: Option<SharedFaBlockPool>,
    kv_layer_windows: Vec<Option<usize>>,
    binding_summary: NativeModelBindingSummary,
    terminal_token_ids: Vec<u32>,
    /// Canonicalized model artifacts directory, folded into the disk
    /// prefix-cache key (see `prefix_cache_layer_layout`). A hot-swap via
    /// `POST /v1/model/load` can reuse the same caller-supplied `model_id`
    /// label for a genuinely different checkpoint (e.g. a routine
    /// re-quantization rollout); `model_id` alone is not a reliable cache
    /// key because it carries no relationship to the actual model content,
    /// and same-architecture checkpoints can share every shape field in
    /// `ModelConfig` while their weights (and thus KV values) differ. This
    /// closes that gap without a wire-format change, since `layer_layout`
    /// is already a free-form bucket in the on-disk key.
    model_artifacts_root: String,
    /// Content-derived artifact identity (manifest + content-addressed
    /// tensor hashes), computed once at load. `None` when the artifact
    /// source cannot supply a stable content fingerprint — the durable L2
    /// prefix tier is then ineligible (a path-only identity could restore
    /// stale KV after an in-place checkpoint replacement).
    artifact_fingerprint: Option<String>,
    states: Mutex<HashMap<RequestId, RequestState>>,
    /// Whether this model can use the experimental batched dense-decode path
    /// (computed once from the loaded model capabilities). Gates the `run()`
    /// interception together with `batched_decode_enabled()`.
    batched_decode_model_eligible: bool,
    batched_decode_model_rejections: Vec<&'static str>,
    /// Shared cohort for the batched dense-decode path (`AX_MLX_BATCHED_DECODE`).
    /// Empty and untouched unless the flag is on and the model is eligible.
    batched_session: Mutex<BatchedDecodeSession>,
    /// Dedicated GPU stream kept alive for the runner's lifetime.
    _stream: MlxStream,
    /// Releases this runner's process-wide wired-residency hold on drop so a
    /// Tiel/Cyber export loading later cannot silently unwire a sibling model.
    _wired_residency: crate::tiel_memory_policy::WiredResidencyGuard,
    /// When true, disable n-gram acceleration. Model-based MTP is controlled
    /// independently by `mtp_requested`.
    disable_ngram_acceleration: bool,
    /// Whether model-based MTP may be used when compatible artifacts and
    /// request sampling are available. Studio deliberately disables n-gram
    /// while leaving this enabled for validated MTP packages.
    mtp_requested: bool,
    /// Immutable, model-scoped MTP family/default/safety policy. The setter
    /// cannot bypass its hard route gate.
    mtp_model_policy: MtpModelPolicy,
    /// Per-artifact Qwen linear-MTP exact capability and resolved selection.
    qwen_linear_mtp_exact_eligible: bool,
    qwen_linear_mtp_exact_enabled: bool,
    qwen_linear_mtp_exact_selection: u32,
    /// Force singleton replay instead of the exact lazy-checkpoint verify
    /// path. Currently initialized `false` for every pack: MXFP4 checkpoint
    /// adopt became sequence-equivalent once the exact profile skipped the
    /// fused RMS+SiLU Metal gate (see the initializer), and replay costs
    /// ~2× S=1 plus identity drift vs MTP-off. Kept as a kill switch.
    qwen_linear_mtp_force_replay: bool,
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
    /// Serialized, thread-agnostic KV snapshots for block-aligned exact prompt prefixes.
    prefix_cache: Arc<Mutex<MlxPrefixCache>>,
    /// Runner-local native FA snapshots. Never shared through
    /// `MlxPrefixCacheStore`; live MLX arrays remain bound to this runner.
    native_prefix_cache: Mutex<MlxNativePrefixCache>,
    /// Optional L2 file-backed prefix cache (F3). Populated when
    /// `AX_MLX_PREFIX_CACHE_DIR` is set and the disk-disabled kill
    /// switch is not engaged. `None` when off; the L2 paths short-
    /// circuit cheaply.
    disk_prefix_cache: Option<Arc<crate::disk_prefix_cache::DiskPrefixCache>>,
    /// Background writer for L2 disk stores. `None` when the disk cache is
    /// off or the worker failed to spawn (stores then run inline).
    disk_prefix_writer: Option<Arc<DiskPrefixCacheWriter>>,
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
    /// Per-thread/per-shape compiled mean-pool embedding closures. Keyed
    /// on `(thread_id, batch_size, max_seq_len)`; the layer loop + final
    /// norm are fused, mean-pool masking is applied post-closure by the
    /// caller. Same kill switch (`AX_EMBED_NO_COMPILE`).
    embed_mean_pool_compile_cache: Mutex<HashMap<EmbedMeanPoolCompileKey, mlx_sys::MlxClosure>>,
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
#[non_exhaustive]
pub struct EmbedCompileCacheStats {
    pub single_hits: u64,
    pub single_misses: u64,
    pub single_len: usize,
    pub batched_hits: u64,
    pub batched_misses: u64,
    pub batched_len: usize,
    pub mean_pool_hits: u64,
    pub mean_pool_misses: u64,
    pub mean_pool_len: usize,
}

#[derive(Clone, Copy, Default)]
struct EmbedCompileStats {
    single_hits: u64,
    single_misses: u64,
    batched_hits: u64,
    batched_misses: u64,
    mean_pool_hits: u64,
    mean_pool_misses: u64,
}

/// Maximum number of compiled closures retained per embed compile cache.
/// When exceeded, the oldest ~25% of entries are evicted (insertion-order
/// sweep) instead of dropping the entire working set — prevents unbounded
/// memory growth and avoids forced recompilation of every live shape.
const EMBED_COMPILE_CACHE_MAX_ENTRIES: usize = 256;

/// Evict approximately 25% of entries when an embed compile cache is full.
/// `HashMap` does not track true LRU access order; we drop the first keys
/// yielded by iteration, which is enough to keep the working set warm under
/// high-cardinality batch-shape workloads.
fn embed_compile_cache_evict_if_full<K, V>(cache: &mut std::collections::HashMap<K, V>)
where
    K: Eq + std::hash::Hash + Clone,
{
    if cache.len() < EMBED_COMPILE_CACHE_MAX_ENTRIES {
        return;
    }
    let evict = (cache.len() / 4).max(1);
    let victims: Vec<K> = cache.keys().take(evict).cloned().collect();
    for key in victims {
        cache.remove(&key);
    }
}

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

/// Global compiled-closure cache counters.
static COMPILE_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COMPILE_CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COMPILE_FALLBACK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn default_mlx_cache_limit(wired_cap: usize) -> usize {
    wired_cap.saturating_add(wired_cap / 2)
}

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
    // gemma4_assistant_mtp.rs and docs/mtp/gemma4-assistant-multi-depth.md.
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

fn aligned_fa_pool_max_blocks(
    total_blocks: u32,
    layer_count: usize,
    sharing: bool,
    explicit_override: Option<u32>,
) -> u32 {
    explicit_override.unwrap_or_else(|| {
        if sharing {
            let layer_count = u32::try_from(layer_count).unwrap_or(u32::MAX);
            total_blocks.saturating_mul(layer_count).max(1)
        } else {
            total_blocks.max(1)
        }
    })
}

fn prefix_snapshot_start_tokens(
    block_size: usize,
    full_block_tokens: usize,
    alignment_restricted: bool,
    native_page_sharing: bool,
) -> usize {
    if alignment_restricted || native_page_sharing {
        full_block_tokens
    } else {
        block_size
    }
}

/// ADR-016 demote-on-evict kill switch. Default ON;
/// `AX_MLX_PREFIX_DEMOTE_ON_EVICT=0` restores the pre-change behavior of
/// dropping evicted native prefix snapshots instead of demoting them into
/// the portable host-RAM store. Read live (like `AX_KV_QUANT`) so tests and
/// A/B harnesses can toggle it without a process restart.
pub(crate) const AX_MLX_PREFIX_DEMOTE_ON_EVICT_ENV: &str = "AX_MLX_PREFIX_DEMOTE_ON_EVICT";

fn prefix_demote_on_evict_enabled() -> bool {
    !std::env::var(AX_MLX_PREFIX_DEMOTE_ON_EVICT_ENV).is_ok_and(|value| value == "0")
}

/// ADR-016 demote-on-evict. Serialize an evicted native prefix snapshot into
/// the portable host-RAM L1 store instead of dropping it, subject to that
/// store's existing budgets (`insert` runs `evict_until_within_policy`, so
/// overflow evicts per policy). Restore is unchanged: the portable tier is
/// already consulted between native L1 and disk L2. Runs on the runner owner
/// thread; the caller still drops the native snapshot after releasing the
/// native-cache lock (ADR-006 lock-order contract).
fn demote_native_prefix_snapshot(
    portable_cache: &Mutex<MlxPrefixCache>,
    key: &MlxPrefixCacheKey,
    snapshot: &MlxNativePrefixSnapshot,
    telemetry: &mut MlxPrefixCacheTelemetry,
) {
    if !prefix_demote_on_evict_enabled() {
        telemetry.record_demotion_skip();
        return;
    }
    // The wire (v4) does serialize rotating_window; it is the runner's
    // restore path that refuses slot-ordered rotating payloads as misses.
    // Never publish a snapshot restore would reject. Native snapshots are
    // all-Paged, so this is dead defense rather than an expected skip.
    if snapshot.token_count == 0 || snapshot.cache.has_rotated_sliding_layers() {
        telemetry.record_demotion_skip();
        return;
    }
    {
        let cache = portable_cache.lock();
        // Cheap admission probes before the O(KV) serialize: a disabled tier,
        // or a resident entry that already supersedes this snapshot, makes the
        // serialize pure waste. `insert_unless_superseded` below re-checks
        // under the insert lock and stays authoritative.
        let superseded = cache.enabled()
            && cache.contains_superseding_snapshot(
                key,
                &snapshot.tokens,
                snapshot.greedy_prefill_output_token,
                false,
            );
        if !cache.enabled() || superseded || snapshot.cache.has_unserializable_layers() {
            telemetry.record_demotion_skip();
            return;
        }
    }
    let payload: Arc<[u8]> = snapshot.cache.serialize_to_bytes().into();
    let demoted = MlxPrefixSnapshot::from_shared_payload(
        payload,
        snapshot.tokens.clone(),
        snapshot.token_count,
        snapshot.greedy_prefill_output_token,
        // The native snapshot this is demoted from has no draft-cursor
        // sidecar (it is a runner-local live snapshot), so the demoted
        // portable entry is trunk-only.
        None,
    );
    let demoted_bytes = demoted.bytes;
    let mut cache = portable_cache.lock();
    // An entry the byte budget could never hold would flush every healthy
    // entry before self-evicting; refuse it up front so one huge demote
    // cannot wipe a healthy portable cache.
    if demoted_bytes > cache.policy.max_bytes {
        telemetry.record_demotion_skip();
        return;
    }
    // The superseding check and the insert share this one lock acquisition:
    // a boundary store that already covers the prefix makes the demote a
    // pure re-serialize, and a full store racing the O(prefix) serialize
    // above must not be overwritten by this staler snapshot.
    let Some(outcome) = cache.insert_unless_superseded(key.clone(), demoted) else {
        telemetry.record_demotion_skip();
        return;
    };
    telemetry.record_stats(cache.stats());
    drop(cache);
    if outcome.stored {
        telemetry.record_demotion(demoted_bytes, outcome.evictions);
    } else {
        // The portable budget refused the entry outright; the drop stands.
        telemetry.evictions = telemetry.evictions.saturating_add(outcome.evictions);
        telemetry.record_demotion_skip();
    }
}

/// Demote entries evicted by a native-tier insert (ADR-016). Suppressed
/// when the portable store itself would not publish for this boundary
/// (`portable_store_phase == false` during scheduler-split prefills):
/// portable serialization is deferred until prompt completion there because
/// per-boundary publishes turn an O(N) prefill into O(N²) memory traffic —
/// a partial-boundary demote would reintroduce exactly that hazard.
fn demote_native_insert_evictions(
    portable_cache: &Mutex<MlxPrefixCache>,
    evicted: &[(Arc<MlxPrefixCacheKey>, Arc<MlxNativePrefixSnapshot>)],
    portable_store_phase: bool,
    telemetry: &mut MlxPrefixCacheTelemetry,
) {
    if !portable_store_phase {
        return;
    }
    for (key, snapshot) in evicted {
        demote_native_prefix_snapshot(portable_cache, key, snapshot, telemetry);
    }
}

/// Evict native snapshots until the pool can satisfy the caller's current
/// demand. Recompute demand after every eviction: dropping a snapshot can
/// make an active adopter the unique owner of its tail block, eliminating a
/// pending COW allocation even when `available_blocks` itself did not grow.
///
/// Each evicted snapshot is offered to `on_evicted` (ADR-016 demotion into
/// the portable store) before it is dropped.
fn reclaim_native_prefix_entries(
    native_cache: &Mutex<MlxNativePrefixCache>,
    pool: &SharedFaBlockPool,
    mut required_blocks: impl FnMut() -> Option<u32>,
    mut on_evicted: impl FnMut(&MlxPrefixCacheKey, &MlxNativePrefixSnapshot),
) -> u32 {
    let mut evictions = 0u32;
    while let Some(required) = required_blocks() {
        if pool.snapshot().available_blocks >= required {
            break;
        }
        let retired = { native_cache.lock().take_lru() };
        let Some((key, retired)) = retired else {
            break;
        };
        on_evicted(&key, &retired);
        drop(retired);
        evictions = evictions.saturating_add(1);
    }
    evictions
}

impl MlxRunner {
    /// Whether the loaded target has a validated Qwen/GLM/DeepSeek-V4 MTP head
    /// or Gemma assistant drafter attached.
    pub(crate) fn has_mtp(&self) -> bool {
        self.mtp_model_policy.has_attached_drafter()
    }

    /// Whether the attached drafter may actually run: an uncertified or
    /// conflicting drafter is attached but never route-safe, so a `Required`
    /// policy must fail closed on it instead of silently decoding direct.
    pub fn mtp_usable(&self) -> bool {
        self.mtp_model_policy.usable()
    }

    /// Drafter served by the generic MTP decode/prefill paths. The Flash Next
    /// candidate owns a dedicated cursor route and is invisible here, so an
    /// unrequested candidate routes exactly like a model without a drafter.
    fn has_generic_mtp(&self) -> bool {
        self.mtp_model_policy.has_generic_drafter()
    }

    /// The session requested the attached Flash Next candidate.
    fn flash_next_mtp_session(&self) -> bool {
        self.mtp_requested
            && self.mtp_model_policy.is_flash_next_candidate()
            && self.weights.qwen4_exp.is_some()
            && self.weights.qwen4_exp_mtp.is_some()
    }

    /// Prefill may carry draft history only for requests the cursor can
    /// decode: argmax sampling without logits processors.
    fn flash_next_mtp_prefill_eligible(
        &self,
        is_greedy: bool,
        sampling: MlxSamplingParams,
    ) -> bool {
        self.flash_next_mtp_session() && is_greedy && !sampling.uses_logits_processors()
    }

    /// Reconcile the request cursor with the cache immediately before a
    /// Flash Next prefill helper advances the trunk.
    fn prepare_flash_next_prefill_cursor(&self, state: &mut RequestState) {
        // Consume the stash unconditionally: a pending cursor is valid only
        // for this one prefill quantum and must never leak forward. `Some`
        // means the payload already decoded, rebound, and verified aligned at
        // L1-restore time, so presence is the only check needed here.
        let pending_restored_cursor = state.flash_next_mtp.pending_restored_cursor.take();
        let restored_cursor_aligned = pending_restored_cursor.is_some();
        let Some(head) = self.weights.qwen4_exp_mtp.as_deref() else {
            drop(pending_restored_cursor);
            state.flash_next_mtp.drop_cursor();
            return;
        };
        let cursor_aligned = match (&state.flash_next_mtp.cursor, &state.cache.qwen4_exp) {
            (Some(cursor), Some(trunk)) => cursor.aligned(trunk),
            _ => false,
        };
        match flash_next_prefill_cursor_action(
            state.cache.seq_len(),
            state.flash_next_mtp.cursor.is_some(),
            cursor_aligned,
            restored_cursor_aligned,
        ) {
            FlashNextPrefillCursorAction::Initialize => {
                drop(pending_restored_cursor);
                state.flash_next_mtp.cursor =
                    Some(crate::model::qwen4_exp_mtp::Qwen4ExpDraftCursor::new(
                        head,
                        self.cfg.compile_cache_identity,
                    ));
                state.flash_next_mtp.emitted_since_clear = 0;
                let telemetry = &mut state.flash_next_mtp.telemetry;
                telemetry.cursor_initialized = telemetry.cursor_initialized.saturating_add(1);
            }
            FlashNextPrefillCursorAction::Keep => drop(pending_restored_cursor),
            FlashNextPrefillCursorAction::Drop => {
                drop(pending_restored_cursor);
                state.flash_next_mtp.drop_cursor();
            }
            FlashNextPrefillCursorAction::ResumeWithRestoredCursor => {
                let cursor = pending_restored_cursor
                    .expect("restored_cursor_aligned implies a pending cursor");
                state.flash_next_mtp.cursor = Some(cursor);
                state.flash_next_mtp.emitted_since_clear = 0;
                let telemetry = &mut state.flash_next_mtp.telemetry;
                telemetry.cursor_restored = telemetry.cursor_restored.saturating_add(1);
            }
            FlashNextPrefillCursorAction::ResumeWithoutCursor => {
                drop(pending_restored_cursor);
                let telemetry = &mut state.flash_next_mtp.telemetry;
                telemetry.resumed_without_cursor =
                    telemetry.resumed_without_cursor.saturating_add(1);
            }
        }
    }

    /// Run one Flash Next prefill quantum through the draft-aware helper. The
    /// trunk result is authoritative whether or not the cursor survives.
    fn run_flash_next_mtp_prefill(
        &self,
        state: &mut RequestState,
        tokens: &[u32],
        chunk_size: usize,
        completes_prompt: bool,
    ) -> Option<u32> {
        self.prepare_flash_next_prefill_cursor(state);
        let had_cursor = state.flash_next_mtp.cursor.is_some();
        let token = crate::generate::chunked_prefill_flash_next_mtp(
            &self.cfg,
            &self.weights,
            tokens,
            &mut state.cache,
            chunk_size,
            completes_prompt,
            &mut state.flash_next_mtp.cursor,
        );
        if had_cursor && state.flash_next_mtp.cursor.is_none() {
            let telemetry = &mut state.flash_next_mtp.telemetry;
            telemetry.prefill_absorb_failures = telemetry.prefill_absorb_failures.saturating_add(1);
        }
        token
    }

    /// Request or suppress model-based MTP independently from the n-gram
    /// speculation switch. The process-wide `AX_NO_SPEC` kill switch remains
    /// authoritative and cannot be re-enabled through this setter.
    pub fn set_mtp_requested(&mut self, requested: bool) {
        self.mtp_requested = requested
            && self.mtp_model_policy.route_safe()
            && !crate::fastpath::ngram_acceleration_disabled();
    }

    pub fn mtp_requested(&self) -> bool {
        self.mtp_requested
    }

    /// Whether this pack is certified for *default-on* model MTP
    /// (`MlxMtpPolicy::Auto`). Dense QwenCalibrated still requires publisher
    /// certification. Linear Qwen candidates require an explicit opt-in until
    /// AX default promotion; publisher speed metadata is insufficient. `MlxMtpPolicy::Required` and
    /// `AX_MLX_MTP_FORCE_REQUESTED` bypass this gate; route safety and
    /// speculation kill switches do not.
    pub fn mtp_certified_default_on(&self) -> bool {
        self.mtp_model_policy.certified_default_on() || crate::fastpath::mtp_force_requested()
    }

    fn mtp_max_depth(&self) -> usize {
        self.mtp_model_policy.max_depth()
    }

    /// Width of the hidden row consumed by the MTP draft head: DeepSeek V4's
    /// nextn block reads the packed pre-collapse residual (`hc*hidden`);
    /// every other head reads the post-norm E-wide hidden.
    fn mtp_draft_hidden_width(&self) -> usize {
        if self.weights.deepseek_v4_nextn.is_some()
            && let Some(v4) = self.cfg.deepseek_v4.as_ref()
        {
            return v4.hc_mult * self.cfg.hidden_size;
        }
        self.cfg.hidden_size
    }

    /// Fresh MTP draft-head KV cache. Qwen/GLM heads own a single layer; the
    /// DeepSeek V4 nextn block appends at slot `num_hidden_layers` (llama.cpp
    /// `il = n_layer + nextn_layer_offset`), so its cache needs one slot past
    /// the main stack.
    fn new_mtp_draft_cache(&self) -> MlxKVCache {
        if self.weights.deepseek_v4_nextn.is_some() {
            MlxKVCache::new(crate::mtp::deepseek_v4_mtp_cache_layer_count(&self.cfg))
        } else {
            MlxKVCache::new(1)
        }
    }

    fn gemma4_assistant_mtp_status(&self) -> &Gemma4AssistantMtpStatus {
        &self.gemma4_assistant_mtp_status
    }

    /// Align FA block-pool geometry to the session `KvManager`.
    ///
    /// No-op when `AX_MLX_FA_KV_BLOCK_POOL` is off. When the flag is on, sets
    /// `block_size_tokens` / `max_blocks` so private FA capacity matches logical
    /// block accounting. Env `AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS` still wins
    /// when set (explicit operator override).
    pub fn align_fa_block_pool_to_kv(&mut self, block_size_tokens: u32, total_blocks: u32) {
        if self.shared_fa_block_pool.is_some() {
            let request_state_exists = !self.states.lock().is_empty();
            let native_state_exists = !self.native_prefix_cache.lock().is_empty();
            if request_state_exists || native_state_exists {
                tracing::error!(
                    target: "ax_engine_mlx::kv_pool",
                    request_state_exists,
                    native_state_exists,
                    "refusing to realign shared FA pool after ownership became live",
                );
                return;
            }
        }
        let Some(config) = self.fa_block_pool_config.as_mut() else {
            return;
        };
        config.block_size_tokens = block_size_tokens.max(1);
        // The PR4 compatibility representation allocates one ID per
        // layer-token block. A runner-wide pool therefore needs one
        // layer-block slot for every logical KvManager block on every
        // standard-FA layer. Fixed per-layer slabs preserve that mapping
        // without allocating the configured ceiling up front.
        config.max_blocks = aligned_fa_pool_max_blocks(
            total_blocks,
            self.cfg.layer_count,
            self.shared_fa_block_pool.is_some(),
            fa_block_pool_max_blocks_override(),
        );
        // max_blocks is always a real memory budget here — either the
        // operator's explicit override, or KvManager.total_blocks (itself an
        // operator-configured session capacity, not an arbitrary scaffold
        // default). Either way exhaustion must fail the request instead of
        // demoting to unbounded contiguous growth (see kv_cache.rs
        // append_paged_fa / MlxRunner::run_item).
        config.hard_cap = true;
        if self.shared_fa_block_pool.is_some() {
            let native_attention = self
                .shared_fa_block_pool
                .as_ref()
                .is_some_and(SharedFaBlockPool::native_attention_enabled);
            self.shared_fa_block_pool = Some(if native_attention {
                SharedFaBlockPool::new_with_native_slab_storage(*config)
                    .expect("aligned native FA slab pool config must remain valid")
            } else {
                SharedFaBlockPool::new_with_slab_storage(*config)
                    .expect("aligned FA slab pool config must remain valid")
            });
        }
    }

    #[cfg(test)]
    /// Build with every cross-session share available: an optional prefix
    /// snapshot store and an optional shared-weights cell (Option A of the
    /// session/weight-reuse design). The first build through an empty cell
    /// loads the weights and publishes them; later builds reuse the loaded
    /// `Arc<ModelWeights>` and skip both the safetensors read and the JIT
    /// warmup forwards.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_artifacts_with_runtime_shares(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
        prefix_cache_store: Option<MlxPrefixCacheStore>,
        shared_weights: Option<&MlxSharedWeightsCell>,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_inner(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            disable_mtp_ngram_stacking,
            prefix_cache_store,
            shared_weights,
            None,
        )
    }

    /// Build a session with its admitted KV pool available to load-time Auto
    /// residency. Existing raw constructors retain conservative admission.
    pub fn from_artifacts_with_session_budget(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
        prefix_cache_store: Option<MlxPrefixCacheStore>,
        shared_weights: Option<&MlxSharedWeightsCell>,
        kv_pool_tokens: u64,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_inner(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            disable_mtp_ngram_stacking,
            prefix_cache_store,
            shared_weights,
            Some(crate::expert_stream::SessionResidencyBudget {
                kv_pool_tokens,
                prefill_chunk,
            }),
        )
    }

    pub fn from_artifacts(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_with_mtp_options(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            mtp_disable_ngram_stacking_from_env(),
        )
    }

    pub(crate) fn from_artifacts_with_mtp_options(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
    ) -> Result<Self, MlxRunnerError> {
        Self::from_artifacts_inner(
            artifacts,
            prefill_chunk,
            disable_ngram_acceleration,
            disable_mtp_ngram_stacking,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_artifacts_inner(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        disable_ngram_acceleration: bool,
        disable_mtp_ngram_stacking: bool,
        prefix_cache_store: Option<MlxPrefixCacheStore>,
        shared_weights: Option<&MlxSharedWeightsCell>,
        session_budget: Option<crate::expert_stream::SessionResidencyBudget>,
    ) -> Result<Self, MlxRunnerError> {
        // Admission and all CPU-only manifest contracts must pass before any
        // process-global MLX setup. Otherwise a rejected first artifact could
        // permanently decide Metal buffer caps or mutate the default stream
        // and memory limits for later valid model loads.
        validate_mlx_supported_manifest(artifacts)?;

        // AX_NO_SPEC is the CLAUDE.md-documented kill switch. Honor it at
        // the runner boundary so server and SDK paths behave the same as
        // the bench CLI, which already reads the env before constructing
        // the runner.
        let speculation_disabled = crate::fastpath::ngram_acceleration_disabled();
        let disable_ngram_acceleration = disable_ngram_acceleration || speculation_disabled;
        // Decide the MLX Metal buffer caps BEFORE the first MLX call below:
        // `set_wired_limit` (and the device-info query feeding it) constructs
        // the Metal device, which is when MLX reads MLX_MAX_*_PER_BUFFER once.
        // Deciding inside `load_weights` alone was too late on this path and
        // silently disabled the MoE auto-caps for every runner-based consumer
        // (server, SDK, Python) while decode-trace — which calls
        // `load_weights` directly — still got them.
        crate::weights::maybe_raise_metal_buffer_caps(artifacts);
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
        let wired_residency = if wired_cap > 0 {
            let scale: f64 = std::env::var("AX_MLX_WIRED_LIMIT_SCALE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.9);
            let scaled = (wired_cap as f64 * scale.clamp(0.0, 1.0)) as usize;
            set_wired_limit(scaled);
            crate::tiel_memory_policy::WiredResidencyGuard::acquire()
        } else {
            crate::tiel_memory_policy::WiredResidencyGuard::unheld()
        };

        // Bound — but do NOT disable — MLX's internal buffer cache. The cache
        // recycles freed GPU buffers; with it off, every transient allocation
        // goes through the system/IOGPU allocator. mlx-lm#828's claim that
        // set_cache_limit(0) has "no measurable performance impact" holds for
        // dense models but is badly wrong for high-allocation MoE decode:
        // disabling it regressed Qwen3.6-35B-A3B decode ~40% (169 -> 100 tok/s),
        // because each expert's transient buffers were re-allocated from IOGPU
        // every step instead of recycled. Default the cache to MLX's own 1.5x
        // working-set policy while keeping the active evaluation memory limit
        // conservative below. Override via AX_MLX_CACHE_LIMIT (bytes); set 0 to
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
                set_cache_limit(default_mlx_cache_limit(wired_cap));
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
        // Weight arrays are immutable once loaded (Arc-shared, evaluated
        // leaves), so a populated share cell lets this build skip the full
        // safetensors read + GPU eval entirely.
        let preloaded_weights = shared_weights.and_then(MlxSharedWeightsCell::get);
        let reused_shared_weights = preloaded_weights.is_some();
        let weights = match preloaded_weights {
            Some(weights) => weights,
            None => {
                let loaded = Arc::new(
                    crate::weights::load_weights_with_session_budget(artifacts, session_budget)
                        .map_err(MlxRunnerError::Weights)?,
                );
                if let Some(cell) = shared_weights {
                    cell.publish(Arc::clone(&loaded));
                }
                loaded
            }
        };
        // Apply the guarded wired-residency policy: after weights are
        // loaded and `weights.expert_stream` is known, clear wired residency
        // for the exactly tested Tiel/Cyber export configuration on M5 Max
        // 128 GiB. Memory/cache limits and the default MTP route/certification
        // are unchanged; see `crate::tiel_memory_policy`.
        crate::tiel_memory_policy::maybe_clear_wired_residency(
            artifacts.root_dir(),
            weights.expert_stream.is_some(),
            wired_residency.is_held(),
        );
        let has_mxfp4_linears = artifacts.tensor_specs().iter().any(|tensor| {
            tensor
                .quantization
                .as_ref()
                .is_some_and(|quantization| quantization.mode == "mxfp4")
        });
        let qwen_linear_mtp_exact_eligible = qwen_linear_mtp_exact_model_eligible(
            &cfg.model_family,
            cfg.linear_attention.is_some(),
            weights.mtp.as_ref().map_or(0, |head| head.max_depth),
            artifacts.tensor_specs(),
        );
        let qwen_linear_throughput_default = cfg.model_family == "qwen3_5"
            && cfg.linear_attention.is_some()
            && weights.mtp.as_ref().is_some_and(|head| head.max_depth > 0)
            && crate::fastpath::qwen_linear_throughput_mtp_enabled();
        let (qwen_linear_mtp_exact_enabled, qwen_linear_mtp_exact_selection) =
            if qwen_linear_throughput_default
                && crate::fastpath::qwen_linear_mtp_exact_env_override() != Some(true)
            {
                crate::fastpath::resolve_qwen_linear_mtp_exact_with_override(
                    qwen_linear_mtp_exact_eligible,
                    Some(false),
                )
            } else {
                crate::fastpath::resolve_qwen_linear_mtp_exact(qwen_linear_mtp_exact_eligible)
            };
        // Cover load-time JIT warm-up with the same arithmetic contract used
        // by production decode. The scope is runner-local and restores on
        // every early return.
        let _qwen_linear_mtp_exact_scope =
            crate::fastpath::scoped_qwen_linear_mtp_exact(qwen_linear_mtp_exact_enabled);
        let (gemma4_assistant_mtp_status, gemma4_assistant_mtp) =
            load_gemma4_assistant_mtp_runtime(&cfg, &weights.gemma4_assistant_mtp);
        let qwen_linear_certification_env_opt_in =
            qwen_linear_mtp_certification_candidate_from_env();
        let qwen_linear_projected_replay_enabled =
            crate::fastpath::mtp_linear_projected_replay_enabled();
        // The sidecar is only usable next to its dedicated trunk. Shared
        // weights may have been loaded by an earlier build, so derive the
        // failure from the live opt-in plus the attached handles.
        let flash_next_mtp_candidate_attached =
            weights.qwen4_exp.is_some() && weights.qwen4_exp_mtp.is_some();
        let flash_next_mtp_candidate_attach_failed = cfg.model_family == "qwen4_exp"
            && crate::weights::flash_next_mtp_sidecar_present(artifacts.root_dir())
            && !flash_next_mtp_candidate_attached;
        let mtp_model_policy = MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            qwen_depth: weights.mtp.as_ref().map(|head| head.max_depth),
            glm_depth: weights.glm_mtp.as_ref().map(|head| head.max_depth),
            gemma4_assistant_depth: gemma4_assistant_mtp
                .as_ref()
                .map(|runtime| runtime.status.max_depth),
            // The nextn head attaches only with its block layer present and
            // exactly one predictor block configured (llama.cpp asserts
            // `n_layer_nextn == 1`); anything else drafts nothing.
            deepseek_v4_depth: weights
                .deepseek_v4_nextn
                .as_ref()
                .filter(|nextn| nextn.layer.is_some())
                .and(cfg.deepseek_v4.as_ref())
                .filter(|v4| v4.num_nextn_predict_layers == 1)
                .map(|_| 1),
            qwen_linear_attention: cfg.linear_attention.is_some(),
            qwen_linear_exact_enabled: qwen_linear_mtp_exact_enabled,
            qwen_linear_projected_replay_enabled,
            // Exact-eligible linear Qwen takes the certified MTP route when
            // speculation is on. `--ax-direct` / disable-ngram stays
            // fail-closed. Env opt-in still wins for formal harnesses.
            // MXFP4 is exact-capable but not auto-promoted. Formal candidate
            // env still opts in; checkpoint adopt is the measured exact path.
            qwen_linear_certification_candidate: qwen_linear_throughput_default
                || resolve_qwen_linear_certification_candidate(
                    qwen_linear_certification_env_opt_in,
                    qwen_linear_mtp_exact_eligible
                        && (qwen_linear_mtp_exact_enabled || qwen_linear_projected_replay_enabled)
                        && !has_mxfp4_linears,
                ),
            qwen_linear_certification_env_opt_in,
            // DeepSeek V4 nextn: same fail-closed product default until Tier 2.
            deepseek_v4_certification_candidate: deepseek_v4_mtp_certification_candidate_from_env(),
            // Publisher-declared default-on certification from the pack's
            // axquant_runtime.json "mtp" block; fail-closed when absent.
            runtime_certification: mtp_runtime_certification(artifacts.root_dir()),
            qwen_linear_throughput_default,
            flash_next_candidate_attached: flash_next_mtp_candidate_attached,
            flash_next_candidate_attach_failed: flash_next_mtp_candidate_attach_failed,
        });
        if flash_next_mtp_candidate_attach_failed {
            tracing::error!(
                target: "ax_engine_mlx::runner",
                model_family = %cfg.model_family,
                "Flash Next MTP sidecar is present but not attached; \
                 no MTP drafter is advertised and MlxMtpPolicy::Required sessions will fail",
            );
        }

        let expert_streaming_active = weights.expert_stream.is_some();
        let binding_summary = binding_summary_from_specs(artifacts.tensor_specs());
        let affine_quant_telemetry = AffineQuantBitsTelemetry::from_specs(artifacts.tensor_specs());

        // MLA models default to a small `prefill_chunk` for warm-extend
        // (snapshot restore + suffix). Evidence (GLM-4.7-Flash warm_extend
        // p2_medium_explain idx=13) shows dual-path cold-large / warm-16
        // re-opens token drift when a real snapshot hit occurs, so cold
        // prefill defaults to the **same** chunk (R2). Operators may opt
        // into large cold via `AX_MLX_MLA_COLD_PREFILL_CHUNK=N` for
        // throughput experiments only. Override warm-extend with
        // `AX_MLX_MLA_PREFILL_CHUNK=N`. Non-MLA tiers ignore MLA resolution
        // and keep the caller-supplied cold chunk.
        //
        // Linear-attention tiers keep a 2048 runner chunk. GatedDelta tiles
        // at 1024 TG (the winning specialization). Streaming remains opt-in.
        // Whole-stack 1024 clamp forced FFN to M=1024 (891 vs mlx_lm 951).
        let linear_attention_chunk_cap =
            if (0..cfg.layer_count).any(|i| cfg.is_linear_attention_layer(i)) {
                Some(
                    crate::linear_attention_ops::linear_attention_prefill_chunk_cap(
                        crate::fastpath::qwen_gated_delta_prefill_streaming_enabled(),
                        cfg.moe_expert_count > 0,
                    ),
                )
            } else {
                None
            };
        // Load-time Metal kernel warm-up (P1-B): compile the decode-shape
        // gated-delta / conv1d specializations now so the lazy MSL →
        // pipeline compile does not land inside the first request. Only
        // interval-classified linear families run these kernels; nemotron
        // routes through its own mamba path. Best-effort by design.
        if let Some(linear) = cfg.linear_attention.as_ref()
            && !cfg.model_family.starts_with("nemotron")
            && linear_attention_chunk_cap.is_some()
            && crate::fastpath::load_kernel_warmup_enabled()
        {
            let warm_started = std::time::Instant::now();
            match crate::linear_attention_ops::warm_gated_delta_decode_kernels(linear) {
                Ok(()) => tracing::info!(
                    target: "ax_engine_mlx::runner",
                    elapsed_ms = warm_started.elapsed().as_millis() as u64,
                    "warmed linear-attention Metal kernels at model load"
                ),
                Err(error) => tracing::warn!(
                    target: "ax_engine_mlx::runner",
                    %error,
                    "linear-attention kernel warm-up failed; first request pays the lazy compile"
                ),
            }
        }
        let clamp_to_linear_cap = |chunk: usize| -> usize {
            match linear_attention_chunk_cap {
                Some(cap) => chunk.min(cap).max(1),
                None => chunk.max(1),
            }
        };
        let has_mla = cfg.mla_attention.is_some();
        let requested_prefill_chunk = prefill_chunk;
        let prefill_chunk = clamp_to_linear_cap(crate::fastpath::resolve_prefill_chunk(
            has_mla,
            requested_prefill_chunk,
            crate::fastpath::mla_prefill_chunk_override(),
        ));
        let cold_prefill_chunk = if has_mla {
            clamp_to_linear_cap(crate::fastpath::resolve_mla_cold_prefill_chunk(
                prefill_chunk,
                crate::fastpath::mla_cold_prefill_chunk_override(),
            ))
        } else {
            clamp_to_linear_cap(requested_prefill_chunk)
        };

        // JIT warm-up: trigger Metal shader compilation for both decode and prefill paths.
        // Encoder-only families (EmbeddingGemma, Nemotron Embed) have no generation
        // (decode/prefill) path — skip the generation warmup (it would panic in
        // the family dispatch); the bidirectional embed forward JITs on first
        // embed. Skipped when the weights came out of the share cell: warmup
        // exists to populate process-wide JIT caches (Metal shader + mlx compile),
        // which the first build through the cell already did — re-running it would
        // re-impose the per-request forward-pass tax Option A removes.
        //
        // Short-prompt serving (e.g. the Qwen/Gemma flip S0 contract of ~34
        // prompt tokens) historically paid 200+ ms of first-request TTFT after
        // an 8-token-only warm-up because SDPA/linear-attention graphs and the
        // direct-pipeline bootstrap are shape-sensitive. Warm the production
        // prefill sizes and one start_direct_pipeline+advance pair so cold
        // first-request TTFT matches the warmed path under the fresh-process
        // contract.
        let is_encoder_embed_family =
            cfg.model_family == "embeddinggemma" || cfg.model_family == "nemotron_embed";
        // A streamed model must not run the resident-model JIT warmup suite.
        // Every forward below walks all MoE layers, so even a tiny dummy
        // request performs a complete SSD expert sweep. The server also skips
        // its production-path warmups using the resolved worker state. The
        // first real request pays the one necessary cold sweep instead of
        // startup performing dozens of hidden sweeps.
        if should_run_load_time_generation_warmup(
            is_encoder_embed_family,
            reused_shared_weights,
            expert_streaming_active,
        ) {
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
            let warmup_lengths = crate::fastpath::prefill_warmup_token_lengths(
                cfg.mla_attention.is_some(),
                prefill_chunk,
            );
            let mut sampling_probs_buf = Vec::new();
            let mut sampling_logits_buf = Vec::new();
            let mut sampling_candidates_buf = Vec::new();
            for &token_count in &warmup_lengths {
                let token_count = token_count.max(1);
                let dummy_tokens: Vec<u32> = vec![0u32; token_count];
                let prefill_tok = chunked_prefill_with_sampling_buffers(
                    &cfg,
                    &weights,
                    &dummy_tokens,
                    &mut dummy_cache,
                    prefill_chunk,
                    MlxSamplingRequest::new(MlxSamplingParams::greedy(), &dummy_tokens),
                    &mut dummy_rng,
                    &mut sampling_probs_buf,
                    &mut sampling_logits_buf,
                    &mut sampling_candidates_buf,
                );
                // Prime the direct double-buffer path at the prefill/decode
                // boundary for short-prompt and largest warm-up lengths so the
                // Qwen gate/up SwiGLU matvec Metal kernel is JIT-hot before the
                // first client request under the fresh-process flip contract.
                if token_count == 34
                    || token_count == *warmup_lengths.last().unwrap_or(&token_count)
                {
                    let pending =
                        start_direct_pipeline(&cfg, &weights, prefill_tok, &mut dummy_cache);
                    let _ = advance_direct_pipeline_with_timings(
                        &cfg,
                        &weights,
                        &pending,
                        &mut dummy_cache,
                    );
                }
                dummy_cache.reset();
            }
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
        let (prefix_cache, disk_prefix_cache, disk_prefix_writer) = prefix_cache_store
            .unwrap_or_else(MlxPrefixCacheStore::from_env)
            .into_parts();
        let native_prefix_cache_policy = prefix_cache.lock().policy;

        let cfg_arc = Arc::new(cfg.clone());
        let weight_layout_telemetry = WeightLayoutTelemetry::from_weights(&weights);
        let has_mtp = mtp_model_policy.has_attached_drafter();
        let mtp_model_route_safe = mtp_model_policy.route_safe();
        if mtp_model_policy.is_qwen_linear_direct_fallback() && !speculation_disabled {
            tracing::warn!(
                target: "ax_engine_mlx::runner",
                model_family = %cfg.model_family,
                mtp_depth = weights.mtp.as_ref().map_or(0, |head| head.max_depth),
                exact_profile_eligible = qwen_linear_mtp_exact_eligible,
                exact_profile_selection = qwen_linear_mtp_exact_selection.route_code(),
                certification_candidate = mtp_model_policy.is_qwen_linear_certification_candidate(),
                "Qwen linear-attention MTP uncertified or ineligible; using direct decode (ADR-020)",
            );
        } else if mtp_model_policy.is_deepseek_v4_direct_fallback() {
            tracing::warn!(
                target: "ax_engine_mlx::runner",
                model_family = %cfg.model_family,
                mtp_depth = mtp_model_policy.max_depth(),
                certification_candidate = mtp_model_policy.is_deepseek_v4_certification_candidate(),
                "DeepSeek V4 nextn MTP uncertified; using direct decode until Tier 2 evidence",
            );
        } else if mtp_model_policy.has_conflicting_drafters() {
            tracing::error!(
                target: "ax_engine_mlx::runner",
                model_family = %cfg.model_family,
                "multiple model-based MTP drafters are attached; using direct decode",
            );
        }
        let batched_decode_certification = load_batched_decode_certification(artifacts);
        let batched_decode_capabilities = BatchedDecodeCapabilities::from_loaded_model(
            // This gates the model's actual decode dispatch, which is driven
            // by the resolved diffusion config. Manifest validation above
            // guarantees every block-diffusion manifest resolves one.
            cfg.diffusion.is_some(),
            &kv_layer_windows,
            &weights.layers,
            batched_decode_certification,
        );
        let allow_uncertified_batched_decode = batched_decode_allow_uncertified();
        let mut batched_decode_model_rejections =
            batched_decode_capabilities.rejection_reasons(allow_uncertified_batched_decode);
        if weights.qwen4_exp.is_some() {
            batched_decode_model_rejections.push("flash_next_request_state");
        }
        let batched_decode_model_eligible = batched_decode_capabilities
            .eligible(allow_uncertified_batched_decode)
            && weights.qwen4_exp.is_none();
        // Capacity for the batched cohort; small (Phase 0 sweet spot is B≈2-4,
        // amortization plateaus past ~8). Override with AX_MLX_BATCHED_DECODE_MAX.
        let batched_cap = std::env::var("AX_MLX_BATCHED_DECODE_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&c| c >= 1)
            .unwrap_or(8);
        let batched_session = Mutex::new(BatchedDecodeSession::new(cfg.layer_count, batched_cap));
        // The CLI n-gram switch controls only the independent n-gram drafter.
        // AX_NO_SPEC remains the process-wide kill switch for every speculative
        // route. Packaged MTP is otherwise admitted independently and still
        // requires route safety plus publisher certification (fail closed).
        let mtp_requested = default_mtp_requested(
            speculation_disabled,
            mtp_model_route_safe,
            mtp_model_policy.certified_default_on() || crate::fastpath::mtp_force_requested(),
        );
        tracing::info!(
            target: "ax_engine_mlx::runner",
            model_family = %cfg.model_family,
            mtp_depth = mtp_model_policy.max_depth(),
            throughput_default = qwen_linear_throughput_default,
            certified_default_on = mtp_model_policy.certified_default_on(),
            route_safe = mtp_model_route_safe,
            disable_ngram = disable_ngram_acceleration,
            mtp_requested,
            linear_candidate = mtp_model_policy.is_qwen_linear_certification_candidate(),
            linear_direct_fallback = mtp_model_policy.is_qwen_linear_direct_fallback(),
            "MTP admission at runner construction"
        );
        let fa_block_pool_config = if fa_kv_block_pool_enabled() {
            Some(default_fa_block_pool_config())
        } else {
            None
        };
        let shared_fa_model_eligible = !has_mtp
            && cfg.mla_attention.is_none()
            && cfg.linear_attention.is_none()
            && cfg.diffusion.is_none()
            && kv_layer_windows.iter().all(Option::is_none)
            && weights.gemma4_unified_vision.is_none()
            && weights.gemma4_unified_audio.is_none()
            && weights.unlimited_ocr_vision.is_none();
        // The diagnostic native slab kernel is consumed only by the standard causal
        // decode graph. Keep the broader P0 page-sharing path available for
        // other eligible models, but fail this experimental compute route
        // closed for sinks, KV-source layers, mixed head geometry, and
        // non-autoregressive generation.
        let native_paged_model_eligible = shared_fa_model_eligible
            && cfg.generation_kind == ax_engine_core::GenerationKind::Autoregressive
            && ax_engine_core::resolve_layer_forward_route(&cfg.model_family)
                == Some(ax_engine_core::LayerForwardRoute::Standard)
            && cfg.layer_configs.iter().all(|layer| {
                layer.kv_source_layer.is_none()
                    && layer.sliding_window.is_none()
                    && layer.head_dim == cfg.head_dim
            })
            && weights.layers.iter().all(|layer| {
                layer.attn_sink.is_none()
                    && layer.linear_attn.is_none()
                    && layer.glm_mla_attn.is_none()
            });
        let native_paged_requested = fa_native_paged_attention_enabled();
        if native_paged_requested && shared_fa_model_eligible && !native_paged_model_eligible {
            tracing::warn!(
                target: "ax_engine_mlx::kv_pool",
                model_family = %cfg.model_family,
                "native paged attention is structurally ineligible; retaining page sharing with dense attention",
            );
        }
        let shared_fa_block_pool = fa_block_pool_config
            .filter(|_| fa_kv_block_sharing_enabled() && shared_fa_model_eligible)
            .map(|config| {
                if native_paged_requested && native_paged_model_eligible {
                    SharedFaBlockPool::new_with_native_slab_storage(config)
                        .expect("default native FA slab pool config must be valid")
                } else {
                    SharedFaBlockPool::new_with_slab_storage(config)
                        .expect("default FA slab pool config must be valid")
                }
            });
        Ok(Self {
            cfg,
            cfg_arc,
            weights,
            prefill_chunk,
            cold_prefill_chunk,
            fa_block_pool_config,
            shared_fa_block_pool,
            kv_layer_windows,
            binding_summary,
            terminal_token_ids,
            model_artifacts_root: artifacts.root_dir().to_string_lossy().into_owned(),
            artifact_fingerprint: crate::artifact_identity::artifact_fingerprint_sha256_with_domain(
                artifacts,
                crate::artifact_identity::PREFIX_CACHE_ARTIFACT_DOMAIN,
            ),
            states: Mutex::new(HashMap::new()),
            batched_decode_model_eligible,
            batched_decode_model_rejections,
            batched_session,
            _stream: stream,
            _wired_residency: wired_residency,
            disable_ngram_acceleration,
            mtp_requested,
            mtp_model_policy,
            qwen_linear_mtp_exact_eligible,
            qwen_linear_mtp_exact_enabled,
            qwen_linear_mtp_exact_selection: qwen_linear_mtp_exact_selection.route_code(),
            // Checkpoint adopt is sequence-equivalent on this pack when the
            // exact profile skips fused RMS+SiLU Metal. Do not force replay
            // for MXFP4 — that is ~2× S=1 plus identity drift vs MTP-off.
            qwen_linear_mtp_force_replay: false,
            disable_mtp_ngram_stacking,
            mtp_optimistic: mtp_optimistic_from_env(),
            mtp_skip_state: mtp_skip_state_from_env(),
            mtp_target_softmax_topk: mtp_target_softmax_topk_from_env(),
            gemma4_assistant_mtp_status,
            gemma4_assistant_mtp,
            ngram_policy_variant,
            prefix_cache,
            native_prefix_cache: Mutex::new(MlxNativePrefixCache::new(native_prefix_cache_policy)),
            disk_prefix_cache,
            disk_prefix_writer,
            rotating_sliding_decode,
            direct_clear_cache_cadence,
            weight_layout_telemetry,
            affine_quant_telemetry,
            embed_compile_cache: Mutex::new(HashMap::new()),
            embed_batch_compile_cache: Mutex::new(HashMap::new()),
            embed_gemma_batch_compile_cache: Mutex::new(HashMap::new()),
            embed_mean_pool_compile_cache: Mutex::new(HashMap::new()),
            embed_compile_stats: Mutex::new(EmbedCompileStats::default()),
        })
    }
}

/// Inputs to [`MlxRunner::complete_multimodal_prefill`]: the branch-specific
/// prefill result plus the per-request constants the shared tail needs.
/// `publish_prefix_snapshot` is false for branches that never store a
/// multimodal prefix (Unlimited-OCR).
struct MultimodalPrefillTail<'a> {
    full_prompt_tokens: Vec<u32>,
    first_token: u32,
    prefill_started: Instant,
    prefill_forward_wall_us: u32,
    publish_prefix_snapshot: bool,
    model_id: &'a str,
    block_size_tokens: u32,
    media_key: Option<&'a str>,
    ctx: Option<&'a RunnerRequestContext>,
    sampling: MlxSamplingParams,
    max_output: u32,
    is_greedy: bool,
}

/// Reset per-request prefill state and assemble the full prompt (reused
/// prefix plus this item's tokens) for a native multimodal prefill branch.
fn begin_multimodal_prefill(
    state: &mut RequestState,
    item: &ax_engine_core::ExecutionItem,
    token_ids: &[u32],
) -> Vec<u32> {
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
    full_prompt_tokens
}

fn effective_embedding_pooling(model_family: &str, pooling: EmbeddingPooling) -> EmbeddingPooling {
    if model_family == "embeddinggemma" || model_family == "nemotron_embed" {
        EmbeddingPooling::Mean
    } else {
        pooling
    }
}

/// DI-W2-002: single-item EmbeddingGemma embed must use the bidirectional
/// Gemma3 sandwich path (same as `embedding_batch_forward` batch-of-one), not
/// the causal dense compiled body (`build_embedding_forward_closure`).
pub(crate) fn embedding_single_item_uses_gemma3_path(model_family: &str) -> bool {
    model_family == "embeddinggemma"
}

/// Build the sampling parameters for a request exactly as the per-item decode
/// path does (see `run_item`), so the batched path classifies and samples each
/// request identically to its single-sequence decode.
fn sampling_params_from_context(ctx: &RunnerRequestContext) -> MlxSamplingParams {
    MlxSamplingParams::new(ctx.temperature, ctx.top_p, ctx.top_k)
        .with_min_p(ctx.min_p)
        .with_repetition_penalty(ctx.repetition_penalty, ctx.repetition_context_size)
        .with_no_repeat_ngram(ctx.no_repeat_ngram_size, ctx.ngram_window)
}

fn seed_batched_session_and_reclaim_private_cache(
    session: &mut BatchedDecodeSession,
    request_id: u64,
    cache: &mut MlxKVCache,
    feed_token: u32,
    seed_len: Option<usize>,
) {
    session.add_with_seed_len(request_id, cache, feed_token, seed_len);
    cache.reset();
}

fn batched_admission_blocked_by_memory_pressure(memory_pressure: Option<&str>) -> bool {
    matches!(
        memory_pressure,
        Some("kv_exhausted" | "kv_exhausted_reclaimable_cache")
    )
}

fn should_suspend_qwen_linear_mtp_for_multirow_batch(
    feature_enabled: bool,
    mtp_requested: bool,
    qwen_linear_candidate: bool,
    mtp_max_depth: usize,
    decode_candidate_count: usize,
) -> bool {
    feature_enabled
        && mtp_requested
        && qwen_linear_candidate
        && mtp_max_depth == 1
        && decode_candidate_count >= 2
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CoalescedDirectDisposition {
    /// Remain on the ordinary direct pipeline.
    Direct,
    /// Request-local n-gram disable: preserve its telemetry and optional
    /// recurrent n-gram feed/re-enable probe.
    RequestDisabled { feed_ngram: bool },
    /// N-gram cooldown consumes one counter and feeds the emitted token.
    NgramCooldown,
    /// The pending direct token is a transition barrier back to n-gram. Finish
    /// it without building another lookahead graph.
    DrainToNgram,
}

fn row_exact_coalescing_family(model_family: &str) -> bool {
    matches!(
        model_family,
        "qwen3" | "qwen3_5" | "qwen3_next" | "gemma3" | "gemma4"
    )
}

fn shared_wall_time_share(total_us: u32, ordinal: usize, count: usize) -> u32 {
    if count == 0 {
        return 0;
    }
    let count_u32 = u32::try_from(count).unwrap_or(u32::MAX);
    let base = total_us / count_u32;
    let remainder = total_us % count_u32;
    base.saturating_add(u32::from(
        u32::try_from(ordinal).unwrap_or(u32::MAX) < remainder,
    ))
}

impl MlxRunner {
    /// Whether a single item can join the batched dense-decode group this step:
    /// a steady-state (`generated_len >= 1`) single-token decode whose sampler
    /// can be applied independently per row. Model-level numerical certification
    /// is enforced before this item-level gate. Host-sampled rows (temperature
    /// with top-k/top-p, or a repetition penalty) require the additional
    /// `AX_MLX_BATCHED_DECODE_SAMPLING` opt-in. The `generated_len == 0`
    /// prefill-token step and pure-temperature branch (GPU
    /// `random_categorical`, non-reproducible) stay on the per-item path.
    fn batched_item_eligible(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
        mtp_suspended: bool,
    ) -> bool {
        // MTP is a request/session route, not a property of the target
        // weights. Keep strict MTP requests on their speculative path while
        // allowing the same package to use batched direct decode when MTP is
        // disabled for the session.
        let strict_mtp = self.has_mtp() && self.mtp_requested;
        if strict_mtp && !mtp_suspended {
            return false;
        }
        if !matches!(item.mode, ExecutionMode::Decode) || item.input_token_slice.len() != 1 {
            return false;
        }
        let Some(ctx) = ctx else {
            return false;
        };
        if think_budget_controller_active(ctx.max_think_tokens, ctx.answer_reserve_tokens) {
            return false;
        }
        if ctx.generated_len < 1 {
            return false;
        }
        match batched_sampling_class(
            sampling_params_from_context(ctx),
            ctx.deterministic_argmax_sampling,
        ) {
            Some(BatchedSamplingClass::Greedy) => true,
            // The initial oMLX-backed promotion is intentionally greedy-only.
            // Sampled MTP-to-direct equivalence needs a separate certificate.
            Some(BatchedSamplingClass::HostSampled) => {
                !strict_mtp && batched_decode_sampling_enabled()
            }
            None => false,
        }
    }

    fn coalesced_direct_disposition(&self, state: &RequestState) -> CoalescedDirectDisposition {
        if self.disable_ngram_acceleration {
            return CoalescedDirectDisposition::Direct;
        }
        if state.ngram_acceleration_disabled_for_request {
            return CoalescedDirectDisposition::RequestDisabled {
                feed_ngram: ngram_request_disabled_fallback_should_feed_output(
                    state.ngram_request_disable_reason,
                ),
            };
        }
        if state.ngram_disabled_steps > 0 {
            return CoalescedDirectDisposition::NgramCooldown;
        }
        CoalescedDirectDisposition::DrainToNgram
    }

    /// Coalesce independent batch=1 direct pipelines into one MLX submission.
    ///
    /// This is the production fail-closed fallback for Qwen/Gemma weights that
    /// have not earned a tensor-batch numerical certificate. Every row retains
    /// its single-request forward graph, cache, argmax, and sampler semantics;
    /// only `async_eval(next_rows)` and `eval(pending_rows)` are shared. That
    /// preserves the reduction order that the sequential oracle exercises
    /// while removing one host submission/barrier pair per sibling request.
    fn run_row_exact_coalesced_group(
        &self,
        group: &[(usize, &ax_engine_core::ExecutionItem)],
        contexts: &[RunnerRequestContext],
    ) -> (Vec<(usize, MlxItemRun)>, usize) {
        struct Row {
            item_index: usize,
            request_id: RequestId,
            scheduled_token_count: u32,
            generated_len: u32,
            max_output: u32,
            ignore_eos: bool,
            final_by_max_output: bool,
            disposition: CoalescedDirectDisposition,
            state: RequestState,
            pending: MlxArray,
            prepared: Option<crate::generate::PreparedDirectPipelineAdvance>,
            token: u32,
            read_wall_us: u32,
        }

        let mut rows = Vec::with_capacity(group.len());
        {
            let mut states = self.states.lock();
            for &(item_index, item) in group {
                let Some(mut state) = states.remove(&item.request_id) else {
                    continue;
                };
                let Some(pending) = state.pending_direct.take() else {
                    states.insert(item.request_id, state);
                    continue;
                };
                let ctx = contexts
                    .iter()
                    .find(|ctx| ctx.request_id == item.request_id);
                let generated_len = ctx.map(|ctx| ctx.generated_len).unwrap_or(0);
                let max_output = ctx.map(|ctx| ctx.max_output_tokens).unwrap_or(1);
                let disposition = self.coalesced_direct_disposition(&state);
                rows.push(Row {
                    item_index,
                    request_id: item.request_id,
                    scheduled_token_count: item.scheduled_token_count,
                    generated_len,
                    max_output,
                    ignore_eos: ctx.map(|ctx| ctx.ignore_eos).unwrap_or(false),
                    final_by_max_output: generated_len.saturating_add(1) >= max_output,
                    disposition,
                    state,
                    pending,
                    prepared: None,
                    token: 0,
                    read_wall_us: 0,
                });
            }
        }

        // A singleton cannot save a submission or barrier. Restore it exactly
        // as found so the ordinary per-item path handles it below.
        if rows.len() < 2 {
            let mut states = self.states.lock();
            for mut row in rows {
                row.state.pending_direct = Some(row.pending);
                states.insert(row.request_id, row.state);
            }
            return (Vec::new(), 0);
        }

        let group_started = Instant::now();
        for row in &mut rows {
            match row.disposition {
                CoalescedDirectDisposition::RequestDisabled { .. } => {
                    row.state.ngram_acceleration.record_request_disabled_step();
                    row.state
                        .ngram_acceleration
                        .record_request_disabled_reason(row.state.ngram_request_disable_reason);
                }
                CoalescedDirectDisposition::NgramCooldown => {
                    row.state.ngram_disabled_steps =
                        row.state.ngram_disabled_steps.saturating_sub(1);
                    row.state.ngram_acceleration.record_cooldown_step();
                }
                CoalescedDirectDisposition::Direct | CoalescedDirectDisposition::DrainToNgram => {}
            }

            let should_prepare = !row.final_by_max_output
                && !matches!(row.disposition, CoalescedDirectDisposition::DrainToNgram);
            if should_prepare {
                row.prepared = Some(prepare_direct_pipeline_advance(
                    &self.cfg,
                    &self.weights,
                    &row.pending,
                    &mut row.state.cache,
                ));
            }
        }

        let prepared_count = rows.iter().filter(|row| row.prepared.is_some()).count();
        let async_eval_wall_us = if prepared_count > 0 {
            let started = Instant::now();
            let refs = rows
                .iter()
                .filter_map(|row| row.prepared.as_ref().map(|prepared| &prepared.next_pending))
                .collect::<Vec<_>>();
            async_eval(&refs);
            elapsed_us(started)
        } else {
            0
        };
        let next_complete_wall_us = if prepared_count > 0 && direct_pipeline_barrier_enabled() {
            let started = Instant::now();
            let refs = rows
                .iter()
                .filter_map(|row| row.prepared.as_ref().map(|prepared| &prepared.next_pending))
                .collect::<Vec<_>>();
            eval(&refs);
            elapsed_us(started)
        } else {
            0
        };

        let pending_eval_started = Instant::now();
        {
            let refs = rows.iter().map(|row| &row.pending).collect::<Vec<_>>();
            eval(&refs);
        }
        let pending_eval_wall_us = elapsed_us(pending_eval_started);
        for row in &mut rows {
            let started = Instant::now();
            row.token = row.pending.first_u32_unchecked();
            row.read_wall_us = elapsed_us(started);
        }
        let group_wall_us = elapsed_us(group_started);

        let row_count = rows.len();
        let mut prepared_ordinal = 0usize;
        let mut runs = Vec::with_capacity(row_count);
        let mut continuing = Vec::with_capacity(row_count);
        let mut should_clear_cache = false;
        for (row_ordinal, mut row) in rows.into_iter().enumerate() {
            let mut timings = DirectPipelineTimings::default();
            if let Some(prepared) = row.prepared.take() {
                timings = prepared.timings;
                timings.async_eval_wall_us =
                    shared_wall_time_share(async_eval_wall_us, prepared_ordinal, prepared_count);
                timings.next_complete_wall_us =
                    shared_wall_time_share(next_complete_wall_us, prepared_ordinal, prepared_count);
                row.state.pending_direct = Some(prepared.next_pending);
                row.state.direct_pipeline_emitted_tokens =
                    row.state.direct_pipeline_emitted_tokens.saturating_add(1);
                should_clear_cache |= direct_pipeline_clear_cache_due(
                    row.state.direct_pipeline_emitted_tokens,
                    self.direct_clear_cache_cadence,
                );
                prepared_ordinal = prepared_ordinal.saturating_add(1);
            }
            timings.pending_eval_wall_us =
                shared_wall_time_share(pending_eval_wall_us, row_ordinal, row_count);
            timings.pending_read_wall_us = row.read_wall_us;
            row.state
                .decode_telemetry
                .record_direct_pipeline(group_wall_us);
            row.state
                .decode_telemetry
                .record_direct_pipeline_timings(timings);
            row.state.decode_telemetry.record_production_decode_eval();
            row.state.decode_telemetry.record_decode(group_wall_us);

            match row.disposition {
                CoalescedDirectDisposition::RequestDisabled { feed_ngram } => {
                    if feed_ngram {
                        row.state.ngram.feed(&[row.token]);
                        maybe_reenable_linear_ngram_from_fallback_output(
                            &mut row.state,
                            self.ngram_policy_variant,
                            true,
                        );
                    }
                }
                CoalescedDirectDisposition::NgramCooldown => {
                    row.state.ngram.feed(&[row.token]);
                }
                CoalescedDirectDisposition::DrainToNgram => {
                    row.state.ngram.feed(&[row.token]);
                    row.state.direct_pipeline_emitted_tokens = 0;
                }
                CoalescedDirectDisposition::Direct => {}
            }

            // The latch wins over the scheduler feed on the next per-item
            // step, so it must track the token this step emitted — a
            // DrainToNgram row leaves `pending_direct` empty and would
            // otherwise re-feed a stale latch from before the group.
            row.state.next_model_last_token = Some(row.token);

            if row.state.cache.hard_cap_exhausted() {
                should_clear_cache = true;
                runs.push((
                    row.item_index,
                    errored_item_run(
                        row.request_id,
                        "FA KV block pool exhausted under explicit \
                         AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS cap",
                    ),
                ));
                continue;
            }

            let terminal = if row.ignore_eos {
                &[][..]
            } else {
                self.terminal_token_ids.as_slice()
            };
            let (sampled, stop_reason) = truncate_sampled_tokens_for_stop(
                vec![row.token],
                row.generated_len,
                row.max_output,
                terminal,
            );
            // Same loop-detection tail as the per-item decode, so a grouped
            // row stops exactly where its solo decode would.
            let (sampled, stop_reason) = apply_loop_detection_stop(
                sampled,
                stop_reason,
                &row.state.generated_tokens,
                loop_detection_for_request(row.ignore_eos, &self.cfg.model_family),
            );
            if stop_reason.is_none() || matches!(stop_reason, Some(StopReason::LoopDetected)) {
                for &token in &sampled {
                    row.state.generated_tokens.push(token);
                    update_ngram_think_state(&self.cfg, &mut row.state.ngram_in_think, token);
                }
            } else {
                should_clear_cache = true;
            }
            let kv_usage = row
                .state
                .cache
                .usage_snapshot_with_layer_windows(&self.kv_layer_windows);
            let mut sampled_iter = sampled.into_iter();
            let run = MlxItemRun {
                update: RequestExecutionUpdate {
                    request_id: row.request_id,
                    tokens_executed: row.scheduled_token_count,
                    output_token: sampled_iter.next(),
                    output_tokens: sampled_iter.collect(),
                    stop_reason,
                    error: None,
                    diffusion_schedule: None,
                },
                ngram_acceleration: row.state.ngram_acceleration,
                mtp_telemetry: row.state.mtp_telemetry,
                flash_next_mtp_telemetry: row.state.flash_next_mtp.telemetry,
                gemma4_assistant_mtp_telemetry: row.state.gemma4_assistant_mtp_telemetry,
                gemma4_unified_multimodal_telemetry: Gemma4UnifiedMultimodalTelemetry::default(),
                decode_telemetry: row.state.decode_telemetry,
                gemma4_moe_profile: Gemma4MoeProfileSnapshot::default(),
                moe_profile: MoeProfileSnapshot::default(),
                linear_attention_profile: LinearAttentionProfileSnapshot::default(),
                dense_ffn_fastpath: DenseFfnFastpathSnapshot::default(),
                prefill_profile: row.state.prefill_profile,
                decode_profile: row.state.decode_profile,
                kv_usage,
                prefix_cache: MlxPrefixCacheTelemetry::default(),
            };
            if stop_reason.is_none() {
                continuing.push((row.request_id, row.state));
            }
            runs.push((row.item_index, run));
        }
        if !continuing.is_empty() {
            let mut states = self.states.lock();
            for (request_id, state) in continuing {
                states.insert(request_id, state);
            }
        }
        if should_clear_cache {
            clear_cache();
        }
        if let Some((_, first)) = runs.first_mut() {
            first.gemma4_moe_profile = take_gemma4_moe_profile_snapshot();
            first.moe_profile = take_moe_profile_snapshot();
            first.linear_attention_profile = take_linear_attention_profile_snapshot();
            first.dense_ffn_fastpath = take_dense_ffn_fastpath_snapshot();
            first
                .decode_profile
                .merge_from(take_decode_profile_snapshot());
        }
        (runs, prepared_count)
    }

    fn gemma4_assistant_mtp_coalesced_item_eligible(
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
        if think_budget_controller_active(ctx.max_think_tokens, ctx.answer_reserve_tokens) {
            return false;
        }
        let sampling = sampling_params_from_context(ctx);
        let states = self.states.lock();
        let Some(state) = states.get(&item.request_id) else {
            return false;
        };
        let assistant_only_draft = !state.mtp_pending_draft.is_empty()
            && state.mtp_pending_draft.len() == state.mtp_pending_draft_sources.len()
            && state
                .mtp_pending_draft_sources
                .iter()
                .all(|source| *source == MtpDraftSource::Gemma4Assistant);
        gemma4_assistant_mtp_coalesced_verify_route(
            crate::fastpath::gemma4_assistant_mtp_coalesced_verify_enabled(),
            self.gemma4_assistant_mtp_status.enabled,
            self.weights.mtp.is_some()
                || self.weights.glm_mtp.is_some()
                || self.weights.deepseek_v4_nextn.is_some(),
            self.mtp_requested,
            self.disable_mtp_ngram_stacking,
            self.mtp_skip_state,
            ctx.deterministic_argmax_sampling,
            sampling.uses_logits_processors(),
            assistant_only_draft,
            state.mtp_adaptive_gate.is_some(),
        ) && !state.mtp_bypassed
            && state.pending_direct.is_none()
            && state.bonus_queue.is_empty()
            && state.diffusion_block_queue.is_empty()
    }

    /// Coalesce independent, exact-greedy Gemma assistant-MTP target verifies.
    ///
    /// Each row keeps a private batch=1 target graph and KV topology. Only the
    /// MLX completion barrier is shared, so reduction order, acceptance,
    /// rollback, and assistant drafting stay identical to the sequential route.
    /// This is deliberately narrower than tensor batching and fails closed for
    /// sampled targets, n-gram stacking, skip-state, or adaptive draft gates.
    fn run_gemma4_assistant_mtp_coalesced_group(
        &self,
        group: &[(usize, &ax_engine_core::ExecutionItem)],
        contexts: &[RunnerRequestContext],
    ) -> Vec<(usize, MlxItemRun)> {
        struct Row {
            item_index: usize,
            request_id: RequestId,
            scheduled_token_count: u32,
            generated_len: u32,
            max_output: u32,
            ignore_eos: bool,
            sampling: MlxSamplingParams,
            state: RequestState,
            last_token: u32,
            pending: Vec<u32>,
            pending_sources: Vec<MtpDraftSource>,
            token_offset: usize,
            verify_len: usize,
            post_norm_all: MlxArray,
            predicted: MlxArray,
            verify_forward_wall_us: u32,
        }

        let mut rows = Vec::with_capacity(group.len());
        {
            let mut states = self.states.lock();
            for &(item_index, item) in group {
                let Some(ctx) = contexts
                    .iter()
                    .find(|ctx| ctx.request_id == item.request_id)
                else {
                    continue;
                };
                let Some(mut state) = states.remove(&item.request_id) else {
                    continue;
                };
                let pending = state.mtp_pending_draft.clone();
                let pending_sources = state.mtp_pending_draft_sources.clone();
                let valid = !pending.is_empty()
                    && pending.len() == pending_sources.len()
                    && pending_sources
                        .iter()
                        .all(|source| *source == MtpDraftSource::Gemma4Assistant)
                    && !state.mtp_bypassed
                    && state.pending_direct.is_none()
                    && state.bonus_queue.is_empty()
                    && state.diffusion_block_queue.is_empty();
                if !valid {
                    states.insert(item.request_id, state);
                    continue;
                }

                let sampling = sampling_params_from_context(ctx);
                let (rotate_sliding, rotate_slack) = cache_rotation_for_execution(
                    item.mode,
                    state.rotating_sliding_latch,
                    self.rotating_sliding_decode,
                    true,
                    self.prefill_chunk,
                );
                state.cache.set_rotating_sliding_decode(rotate_sliding);
                state.cache.set_rotating_sliding_slack(rotate_slack);
                if item.mode == ExecutionMode::Prefill && state.rotating_sliding_latch.is_none() {
                    state.rotating_sliding_latch = Some((rotate_sliding, rotate_slack));
                }

                let last_token = state
                    .next_model_last_token
                    .or_else(|| item.input_token_slice.last().copied())
                    .unwrap_or(0);
                let token_offset = state.cache.seq_len();
                let mut verify_input = Vec::with_capacity(1 + pending.len());
                verify_input.push(last_token);
                verify_input.extend_from_slice(&pending);
                let verify_len = verify_input.len();
                let verify_forward_started = Instant::now();
                let (logits_all, post_norm_all) = forward_all_positions_with_post_norm(
                    &self.cfg,
                    &self.weights,
                    &verify_input,
                    &mut state.cache,
                    token_offset,
                );
                let verify_forward_wall_us = elapsed_us(verify_forward_started);
                state.cache.advance(verify_len);
                let predicted = argmax(&logits_all, None);
                rows.push(Row {
                    item_index,
                    request_id: item.request_id,
                    scheduled_token_count: item.scheduled_token_count,
                    generated_len: ctx.generated_len,
                    max_output: ctx.max_output_tokens,
                    ignore_eos: ctx.ignore_eos,
                    sampling,
                    state,
                    last_token,
                    pending,
                    pending_sources,
                    token_offset,
                    verify_len,
                    post_norm_all,
                    predicted,
                    verify_forward_wall_us,
                });
            }
        }

        if rows.len() < 2 {
            let mut states = self.states.lock();
            for row in rows {
                states.insert(row.request_id, row.state);
            }
            return Vec::new();
        }

        let group_started = Instant::now();
        let mut owned_targets = Vec::new();
        for row in &rows {
            owned_targets.push(row.predicted.clone());
            owned_targets.push(row.post_norm_all.clone());
            owned_targets.extend(row.state.cache.collect_eval_refs().into_iter().cloned());
        }
        let verify_eval_started = Instant::now();
        let target_refs = owned_targets.iter().collect::<Vec<_>>();
        eval(&target_refs);
        let verify_eval_wall_us = elapsed_us(verify_eval_started);

        let row_count = rows.len();
        let mut should_clear_cache = false;
        let mut continuing = Vec::with_capacity(row_count);
        let mut runs = Vec::with_capacity(row_count);
        for (row_ordinal, mut row) in rows.into_iter().enumerate() {
            // Greedy coalesced path: optionally roll back multi-token
            // teacher-forced KV and re-verify with the sequential production
            // oracle so accepted drafts match MTP-off greedy (same contract as
            // single-item run_mtp_decode). Kill-switch
            // AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE=0 keeps multi-token
            // accept decisions (faster; re-check exactness before Tier 2),
            // except when the cycle-continuation guard forces sequential.
            let oracle_on = crate::fastpath::gemma4_assistant_mtp_sequential_oracle_enabled();
            let cycle_guard_on = crate::fastpath::gemma4_assistant_mtp_cycle_guard_enabled();
            let early_gen_on =
                crate::fastpath::gemma4_assistant_mtp_early_gen_pure_direct_enabled();
            let early_gen_force = gemma_early_gen_pure_direct_force(
                early_gen_on,
                row.state.generated_tokens.len(),
                GEMMA_MT_EARLY_GEN_PURE_DIRECT_TOKENS,
            );
            let mut cycle_history_buf = [0u32;
                GEMMA_CYCLE_GUARD_MAX_PERIOD
                    .saturating_mul(2)
                    .saturating_add(1)];
            let cycle_history_len =
                if row.sampling.temperature <= 0.0 && !oracle_on && cycle_guard_on {
                    fill_gemma_cycle_history(
                        &row.state.generated_tokens,
                        row.last_token,
                        &mut cycle_history_buf,
                    )
                } else {
                    0
                };
            let cycle_hit = cycle_history_len > 0
                && draft_continues_committed_cycle(
                    &cycle_history_buf[..cycle_history_len],
                    &row.pending,
                );
            let cycle_force = cycle_hit || row.state.gemma_mtp_cycle_latched;
            let use_sequential_oracle = row.sampling.temperature <= 0.0
                && matches!(
                    gemma_greedy_verify_route(
                        oracle_on,
                        cycle_guard_on,
                        cycle_force,
                        early_gen_force,
                    ),
                    GemmaGreedyVerifyRoute::SequentialOracle
                );
            if cycle_force && !oracle_on {
                if cycle_hit {
                    row.state.gemma_mtp_cycle_latched = true;
                }
                row.state.mtp_telemetry.record_gemma_cycle_guard();
            }
            let (accept_count, draft_hidden, tail_token, accept_wall_us, rollback_wall_us) =
                if use_sequential_oracle {
                    let accept_started = Instant::now();
                    let trimmed_back = row.state.cache.trim_to(row.token_offset);
                    if !trimmed_back {
                        should_clear_cache = true;
                        runs.push((
                            row.item_index,
                            errored_item_run(
                                row.request_id,
                                "coalesced Gemma assistant-MTP sequential rollback was refused",
                            ),
                        ));
                        continue;
                    }
                    let seq = sequential_greedy_mtp_verify(
                        &self.cfg,
                        &self.weights,
                        &mut row.state.cache,
                        row.last_token,
                        &row.pending,
                        row.token_offset,
                    );
                    let accept_wall_us = elapsed_us(accept_started);
                    let rejected_count = row.pending.len().saturating_sub(seq.accept_count);
                    if rejected_count > 0 {
                        let new_mtp_len = row.state.mtp_decode_count.saturating_sub(rejected_count);
                        if let Some(cache) = row.state.mtp_cache.as_mut() {
                            let _ = cache.trim_to(new_mtp_len);
                        }
                        row.state.mtp_decode_count = new_mtp_len;
                    }
                    (
                        seq.accept_count,
                        seq.draft_hidden,
                        seq.correction_token,
                        accept_wall_us,
                        0,
                    )
                } else {
                    let predicted = row.predicted.data_u32().to_vec();
                    let accept_started = Instant::now();
                    let accept_count = row
                        .pending
                        .iter()
                        .zip(predicted.iter())
                        .take_while(|(draft, target)| draft == target)
                        .count();
                    let accept_wall_us = elapsed_us(accept_started);

                    let rollback_started = Instant::now();
                    let committed_len = row.token_offset + 1 + accept_count;
                    let trimmed = row.state.cache.trim_to(committed_len);
                    let rejected_count = row.pending.len().saturating_sub(accept_count);
                    if rejected_count > 0 {
                        let new_mtp_len = row.state.mtp_decode_count.saturating_sub(rejected_count);
                        if let Some(cache) = row.state.mtp_cache.as_mut() {
                            let _ = cache.trim_to(new_mtp_len);
                        }
                        row.state.mtp_decode_count = new_mtp_len;
                    }
                    let rollback_wall_us = elapsed_us(rollback_started);
                    if !trimmed {
                        should_clear_cache = true;
                        runs.push((
                            row.item_index,
                            errored_item_run(
                                row.request_id,
                                "coalesced Gemma assistant-MTP rollback trim was refused",
                            ),
                        ));
                        continue;
                    }

                    let draft_hidden = slice_post_norm_hidden(
                        &row.post_norm_all,
                        accept_count,
                        self.cfg.hidden_size,
                    );
                    let tail_token = predicted.get(accept_count).copied().unwrap_or(0);
                    (
                        accept_count,
                        draft_hidden,
                        tail_token,
                        accept_wall_us,
                        rollback_wall_us,
                    )
                };
            let tail_sample_started = Instant::now();
            let tail_sample_wall_us = elapsed_us(tail_sample_started);
            let mut result = row.pending[..accept_count].to_vec();
            result.push(tail_token);

            let proposal_law = if row.state.mtp_pending_draft_distributions.is_empty()
                && crate::mtp::mtp_draft_mode_from_env() == crate::mtp::MtpDraftMode::Greedy
            {
                MtpProposalLaw::DeterministicDelta
            } else {
                MtpProposalLaw::Stochastic
            };
            row.state
                .mtp_telemetry
                .record_correctness_mode(MtpCorrectnessMode::GreedyExact, proposal_law);
            row.state.mtp_telemetry.record_step(
                row.pending.len(),
                accept_count,
                &row.pending_sources,
                None,
                accept_count,
            );
            let row_eval_wall_us =
                shared_wall_time_share(verify_eval_wall_us, row_ordinal, row_count);
            row.state.gemma4_assistant_mtp_telemetry.record_verified(
                row.pending.len(),
                accept_count,
                row.verify_forward_wall_us,
                row_eval_wall_us,
            );
            row.state.ngram.feed(&result);

            let (next_depth, _) = mtp_next_adaptive_depth(
                row.state.mtp_adaptive_max_depth,
                self.mtp_max_depth(),
                row.pending.len(),
                accept_count,
                row.state.mtp_consecutive_misses,
                // Early short-context only: stop-loss reject storms on tiny
                // gens (gen med killers) without aborting multi-token mid
                // long-ish short-ctx gens (smokef139 trial3 1.20→1.04).
                // Window 24 covers formal general short tails that still
                // stack low-accept cost after the first dozen tokens.
                self.gemma4_assistant_mtp.is_some()
                    && row.state.cache.seq_len() < 512
                    && row.state.generated_tokens.len() < 24,
                crate::fastpath::mtp_depth3_miss_backoff_enabled(),
                crate::fastpath::mtp_depth3_hysteresis_enabled(),
                None, // Coalesced Gemma assistant MTP is outside this policy.
            );
            row.state.mtp_adaptive_max_depth = next_depth;
            if accept_count == 0 {
                row.state.mtp_consecutive_misses =
                    row.state.mtp_consecutive_misses.saturating_add(1);
            } else {
                row.state.mtp_consecutive_misses = 0;
            }
            if row.state.mtp_telemetry.mtp_only_accept_rate_ewma_samples >= mtp_bypass_min_samples()
                && row.state.mtp_telemetry.mtp_only_accept_rate_ewma < mtp_bypass_threshold()
            {
                row.state.mtp_bypassed = true;
            }

            let draft_started = Instant::now();
            if row.state.mtp_bypassed {
                row.state.mtp_pending_draft.clear();
                row.state.mtp_pending_draft_log_probs.clear();
                row.state.mtp_pending_draft_log_prob_temperature = None;
                row.state.mtp_pending_draft_distributions.clear();
                row.state.mtp_pending_draft_sources.clear();
            } else {
                let (draft, log_probs, distributions) = self.gemma4_assistant_draft_token(
                    &mut row.state,
                    tail_token,
                    &draft_hidden,
                    row.sampling,
                );
                row.state.mtp_pending_draft = draft;
                row.state.mtp_pending_draft_log_probs = log_probs;
                row.state.mtp_pending_draft_log_prob_temperature = Some(1.0);
                row.state.mtp_pending_draft_distributions = distributions;
                row.state.mtp_pending_draft_sources =
                    vec![MtpDraftSource::Gemma4Assistant; row.state.mtp_pending_draft.len()];
            }
            let draft_wall_us = elapsed_us(draft_started);
            if !row.state.mtp_pending_draft.is_empty() {
                row.state
                    .gemma4_assistant_mtp_telemetry
                    .record_submitted(row.state.mtp_pending_draft.len(), draft_wall_us);
            }
            let mtp_timings = MtpStepTimings {
                verify_forward_wall_us: row.verify_forward_wall_us,
                verify_eval_wall_us: row_eval_wall_us,
                accept_wall_us,
                rollback_wall_us,
                tail_sample_wall_us,
                draft_wall_us,
                assistant_draft_wall_us: draft_wall_us,
                verify_tokens: saturating_u32(row.verify_len),
                emitted_tokens: saturating_u32(result.len()),
                ..MtpStepTimings::default()
            };
            row.state.mtp_telemetry.record_timings(mtp_timings);
            row.state
                .decode_telemetry
                .record_decode(shared_wall_time_share(
                    elapsed_us(group_started),
                    row_ordinal,
                    row_count,
                ));

            let terminal = if row.ignore_eos {
                &[][..]
            } else {
                self.terminal_token_ids.as_slice()
            };
            let sampled = apply_decode_result(&mut row.state, &result, terminal);
            let (sampled, stop_reason) = truncate_sampled_tokens_for_stop(
                sampled,
                row.generated_len,
                row.max_output,
                terminal,
            );
            if row.state.cache.hard_cap_exhausted() {
                should_clear_cache = true;
                runs.push((
                    row.item_index,
                    errored_item_run(
                        row.request_id,
                        "FA KV block pool exhausted under explicit \
                         AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS cap",
                    ),
                ));
                continue;
            }
            if stop_reason.is_none() {
                for &token in &sampled {
                    row.state.generated_tokens.push(token);
                    update_ngram_think_state(&self.cfg, &mut row.state.ngram_in_think, token);
                }
            } else {
                should_clear_cache = true;
            }

            let kv_usage = row
                .state
                .cache
                .usage_snapshot_with_layer_windows(&self.kv_layer_windows);
            let mut sampled_iter = sampled.into_iter();
            let run = MlxItemRun {
                update: RequestExecutionUpdate {
                    request_id: row.request_id,
                    tokens_executed: row.scheduled_token_count,
                    output_token: sampled_iter.next(),
                    output_tokens: sampled_iter.collect(),
                    stop_reason,
                    error: None,
                    diffusion_schedule: None,
                },
                ngram_acceleration: row.state.ngram_acceleration,
                mtp_telemetry: row.state.mtp_telemetry,
                flash_next_mtp_telemetry: row.state.flash_next_mtp.telemetry,
                gemma4_assistant_mtp_telemetry: row.state.gemma4_assistant_mtp_telemetry,
                gemma4_unified_multimodal_telemetry: Gemma4UnifiedMultimodalTelemetry::default(),
                decode_telemetry: row.state.decode_telemetry,
                gemma4_moe_profile: Gemma4MoeProfileSnapshot::default(),
                moe_profile: MoeProfileSnapshot::default(),
                linear_attention_profile: LinearAttentionProfileSnapshot::default(),
                dense_ffn_fastpath: DenseFfnFastpathSnapshot::default(),
                prefill_profile: row.state.prefill_profile,
                decode_profile: row.state.decode_profile,
                kv_usage,
                prefix_cache: MlxPrefixCacheTelemetry::default(),
            };
            if stop_reason.is_none() {
                continuing.push((row.request_id, row.state));
            }
            runs.push((row.item_index, run));
        }

        if !continuing.is_empty() {
            let mut states = self.states.lock();
            for (request_id, state) in continuing {
                states.insert(request_id, state);
            }
        }
        if let Some((_, first)) = runs.first_mut() {
            first.gemma4_moe_profile = take_gemma4_moe_profile_snapshot();
            first.moe_profile = take_moe_profile_snapshot();
            first.linear_attention_profile = take_linear_attention_profile_snapshot();
            first.dense_ffn_fastpath = take_dense_ffn_fastpath_snapshot();
            first
                .decode_profile
                .merge_from(take_decode_profile_snapshot());
        }
        if should_clear_cache {
            clear_cache();
        }
        runs
    }

    /// Run one decode step for a group of eligible requests through the shared
    /// [`BatchedDecodeSession`] (one batched forward for the whole group),
    /// producing one `RequestExecutionUpdate` per request. Mirrors `run_item`'s
    /// decode tail (stop detection, `generated_tokens`, state removal on stop).
    /// The caller holds the session lock.
    ///
    /// A session-resident request's KV lives in the session; its private cache
    /// is reset immediately after the seed copy so the two representations do
    /// not retain duplicate GPU storage.
    fn run_batched_decode_group(
        &self,
        session: &mut BatchedDecodeSession,
        group: &[&ax_engine_core::ExecutionItem],
        contexts: &[RunnerRequestContext],
        mtp_transition_ids: &std::collections::HashSet<u64>,
    ) -> Vec<RequestExecutionUpdate> {
        // 1. Seed joiners from their prefilled state.cache; set the feed token
        //    (the scheduler is the source of truth) for every group member.
        for item in group {
            let id = item.request_id.0;
            let feed = item.input_token_slice[0];
            if session.active_ids().contains(&id) {
                session.set_current(id, feed);
            } else {
                let mut states = self.states.lock();
                if let Some(state) = states.get_mut(&item.request_id) {
                    // Cold-joiner invariant: a joiner's private cache must not
                    // already hold `feed`'s KV, or the `None` seed below would
                    // double it. Warm direct pipelines are intercepted by the
                    // `direct_pipeline_pending` filter / bootstrap route, so a
                    // joiner here never carries a pending direct token.
                    debug_assert!(
                        state.pending_direct.is_none(),
                        "batched decode joiner must be cold (no pending direct token)"
                    );
                    let mtp_transition = mtp_transition_ids.contains(&id);
                    if mtp_transition {
                        // Target KV contains only committed inputs; `feed` is
                        // the next unprocessed token. Discard unverified draft
                        // state and transfer the complete target cache.
                        suspend_mtp_for_batched_decode(state);
                    }
                    clear_private_decode_latches_for_batched_session(state);
                    seed_batched_session_and_reclaim_private_cache(
                        session,
                        id,
                        &mut state.cache,
                        feed,
                        // Every joiner reaching this path is cold: warm direct
                        // pipelines are intercepted by join_direct_pipeline_group.
                        // The scheduler feed token has not entered this cache,
                        // so transfer every committed KV position.
                        None,
                    );
                }
            }
        }
        // 2. One batched forward for the whole cohort (the amortized weight
        //    read), then per-row token resolution matching each request's
        //    single-sequence sampler.
        //
        // A mixed scheduler turn can still hold an exact-MTP sibling outside
        // this group, leaving the outer model scope enabled. The tensor cohort
        // is target-only, so restore the ordinary direct kernel contract for
        // its forward just as the per-row direct fallback does.
        let _direct_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
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
                diffusion_schedule: None,
            });
        }
        updates
    }

    fn join_direct_pipeline_group(
        &self,
        session: &mut BatchedDecodeSession,
        group: &[&ax_engine_core::ExecutionItem],
        contexts: &[RunnerRequestContext],
    ) -> Vec<RequestExecutionUpdate> {
        let mut updates = Vec::with_capacity(group.len());
        for item in group {
            let request_id = item.request_id;
            let ctx = contexts.iter().find(|ctx| ctx.request_id == request_id);
            let generated_len = ctx.map(|ctx| ctx.generated_len).unwrap_or(0);
            let max_output = ctx.map(|ctx| ctx.max_output_tokens).unwrap_or(1);
            let terminal = if ctx.map(|ctx| ctx.ignore_eos).unwrap_or(false) {
                &[][..]
            } else {
                self.terminal_token_ids.as_slice()
            };
            let mut states = self.states.lock();
            let Some(state) = states.get_mut(&request_id) else {
                continue;
            };
            let Some(pending) = state.pending_direct.take() else {
                continue;
            };
            let (token, _, _) = finish_pending_token(&pending);
            let (sampled, stop_reason) =
                truncate_sampled_tokens_for_stop(vec![token], generated_len, max_output, terminal);
            if stop_reason.is_none() {
                clear_private_decode_latches_for_batched_session(state);
                seed_batched_session_and_reclaim_private_cache(
                    session,
                    request_id.0,
                    &mut state.cache,
                    token,
                    None,
                );
                state.generated_tokens.extend_from_slice(&sampled);
            } else {
                states.remove(&request_id);
            }
            let mut sampled = sampled.into_iter();
            updates.push(RequestExecutionUpdate {
                request_id,
                tokens_executed: 1,
                output_token: sampled.next(),
                output_tokens: sampled.collect(),
                stop_reason,
                error: None,
                diffusion_schedule: None,
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

/// One request's padded-batched-prefill result, staged by
/// `MlxRunner::stage_batched_prefill` and consumed inside `run_item`'s plain
/// prefill branch when its preconditions still hold there (cold cache, same
/// prompt tokens); otherwise dropped in favor of sequential prefill.
struct StagedBatchedPrefill {
    prompt_tokens: Vec<u32>,
    /// `[vocab]` f32 last-prompt-position logits, already evaluated.
    logits: MlxArray,
    /// Per-layer `[1, kv_heads, len, head_dim]` K/V, already evaluated.
    layer_kv: Vec<(MlxArray, MlxArray)>,
    seq_len: usize,
}

impl ExecutionRunner for MlxRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        // Keep exact arithmetic model-scoped. This prevents one resident Qwen
        // MTP model from changing the projection path of another model in the
        // same server process. Direct sessions stay off the verifier
        // contract so an AXQ sidecar does not disable fused decode.
        //
        // Direct escape: when every item is a decode step whose MTP
        // profitability bypass or multirow-batch suspension has latched, no
        // verify can happen in this run. Drop the verifier exactness scope so
        // target-only decode uses its certified direct kernels. Mixed and
        // prefill batches keep the scope.
        let all_decode_items_mtp_inactive = {
            let items = &input.execution_batch.items;
            !items.is_empty() && items.iter().all(|item| item.mode == ExecutionMode::Decode) && {
                let states = self.states.lock();
                items.iter().all(|item| {
                    states.get(&item.request_id).is_some_and(|state| {
                        state.mtp_bypassed || state.mtp_suspended_for_batched_decode
                    })
                })
            }
        };
        // The relaxed-target profile already accepts stock-MLX target
        // arithmetic during verification. Keep the cold prefill on the same
        // arithmetic contract so the first sampled token and recurrent target
        // state do not come from the exact-profile projection order while all
        // later target rows come from stock arithmetic. Draft construction is
        // decode-only and therefore remains under the outer exact scope.
        let all_items_prefill = !input.execution_batch.items.is_empty()
            && input
                .execution_batch
                .items
                .iter()
                .all(|item| item.mode == ExecutionMode::Prefill);
        let relaxed_target_prefill = all_items_prefill
            && crate::fastpath::mtp_relaxed_target_verify_enabled()
            && crate::fastpath::mtp_linear_projected_replay_enabled();
        let relaxed_mtp_session = self.mtp_requested
            && !all_decode_items_mtp_inactive
            && crate::fastpath::mtp_relaxed_target_verify_enabled()
            && crate::fastpath::mtp_linear_projected_replay_enabled();
        let exact_arithmetic_enabled = qwen_linear_mtp_exact_scope_for_request(
            self.qwen_linear_mtp_exact_enabled,
            self.mtp_requested && !all_decode_items_mtp_inactive && !relaxed_target_prefill,
        );
        let _qwen_linear_mtp_relaxed_session_scope =
            crate::fastpath::scoped_qwen_linear_mtp_relaxed_session(relaxed_mtp_session);
        let _qwen_linear_mtp_exact_scope =
            crate::fastpath::scoped_qwen_linear_mtp_exact(exact_arithmetic_enabled);
        // Selected reads execute synchronously on this worker. Drain warmup
        // work before attributing successful gathers to the current step.
        let _ = crate::expert_stream::take_selected_expert_read_stats();
        let step_id = input.execution_batch.step_id;
        let mut request_updates = Vec::new();
        let logits_handles = Vec::new();
        let logits_outputs = Vec::new();

        let mut route_metadata = input.execution_batch.route_metadata.clone();
        let mut ngram_acceleration = NgramAccelerationTelemetry::default();
        let mut mtp_telemetry = MtpTelemetry::default();
        let mut flash_next_mtp_telemetry = FlashNextMtpTelemetry::default();
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

        // ADR-038: emit generation kind + layer-forward route on every step so
        // benchmarks and serving telemetry do not infer paradigm from family
        // strings alone.
        upsert_route_decision(
            &mut route_metadata.crossover_decisions,
            ROUTE_DECISION_AX_MLX_GENERATION_KIND,
            self.cfg.generation_kind.telemetry_code(),
        );
        if let Some(route) = ax_engine_core::resolve_layer_forward_route(&self.cfg.model_family) {
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                ROUTE_DECISION_AX_MLX_LAYER_FORWARD_ROUTE,
                route.telemetry_code(),
            );
        }
        // Prefer scheduler-planned work unit (ADR-038); fall back to local plan.
        if let Some(first) = input.execution_batch.items.first() {
            let work_unit = first.planned_work_unit;
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                ROUTE_DECISION_AX_MLX_GENERATION_WORK_UNIT,
                work_unit.telemetry_code(),
            );
        }
        append_qwen_linear_mtp_exact_route_decisions(
            &mut route_metadata.crossover_decisions,
            self.qwen_linear_mtp_exact_eligible,
            self.qwen_linear_mtp_exact_enabled,
            exact_arithmetic_enabled,
            self.qwen_linear_mtp_exact_selection,
        );
        {
            let mut route_decisions =
                IndexedRouteDecisions::new(&mut route_metadata.crossover_decisions);
            self.mtp_model_policy
                .append_route_decisions(self.mtp_requested, &mut route_decisions);
        }
        upsert_route_decision(
            &mut route_metadata.crossover_decisions,
            "ax_mlx_qwen_linear_mtp_depth_one_gate_zero_model_default",
            u32::from(self.mtp_requested && self.mtp_model_policy.qwen_gate_default() == Some(0.0)),
        );
        if self.mtp_model_policy.is_qwen_linear_direct_fallback() {
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                "ax_mlx_qwen_linear_mtp_direct_fallback",
                1,
            );
        }

        // ── Batched decode interception (AX_MLX_BATCHED_DECODE, default on
        // with fail-closed certification). Certified eligible decode items run
        // through one shared tensor-batched forward. Qwen/Gemma direct rows
        // without a numerical certificate use the row-exact coalesced fallback:
        // independent batch=1 graphs with one grouped submission/barrier.
        // Everything else stays byte-for-byte on the per-item path.
        let mut batched_idx: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut batched_forward_rows = 0usize;
        let mut row_exact_coalesced_rows = 0usize;
        let mut row_exact_coalesced_forward_rows = 0usize;
        let mut gemma4_assistant_mtp_coalesced_rows = 0usize;
        let mut mtp_suspended_batched_rows = 0usize;
        let batched_enabled = batched_decode_enabled();
        let decode_candidate_count = input
            .execution_batch
            .items
            .iter()
            .filter(|item| {
                matches!(item.mode, ExecutionMode::Decode) && item.input_token_slice.len() == 1
            })
            .count();
        let mtp_multirow_candidate_count = input
            .execution_batch
            .items
            .iter()
            .filter(|item| {
                if !matches!(item.mode, ExecutionMode::Decode) || item.input_token_slice.len() != 1
                {
                    return false;
                }
                let Some(ctx) = input
                    .request_contexts
                    .iter()
                    .find(|ctx| ctx.request_id == item.request_id)
                else {
                    return false;
                };
                ctx.generated_len >= 1
                    && matches!(
                        batched_sampling_class(
                            sampling_params_from_context(ctx),
                            ctx.deterministic_argmax_sampling,
                        ),
                        Some(BatchedSamplingClass::Greedy)
                    )
            })
            .count();
        let mtp_multirow_batch = should_suspend_qwen_linear_mtp_for_multirow_batch(
            mtp_multirow_batch_enabled(),
            self.mtp_requested,
            self.mtp_model_policy
                .is_qwen_linear_certification_candidate(),
            self.mtp_model_policy.max_depth(),
            mtp_multirow_candidate_count,
        );
        if batched_enabled {
            let group = input
                .execution_batch
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| {
                    self.gemma4_assistant_mtp_coalesced_item_eligible(
                        item,
                        input
                            .request_contexts
                            .iter()
                            .find(|ctx| ctx.request_id == item.request_id),
                    )
                })
                .collect::<Vec<_>>();
            if group.len() >= 2 {
                let runs =
                    self.run_gemma4_assistant_mtp_coalesced_group(&group, &input.request_contexts);
                gemma4_assistant_mtp_coalesced_rows = runs.len();
                for (item_index, result) in runs {
                    batched_idx.insert(item_index);
                    ngram_acceleration.merge_from(result.ngram_acceleration);
                    mtp_telemetry.merge_from(result.mtp_telemetry);
                    flash_next_mtp_telemetry.merge_from(result.flash_next_mtp_telemetry);
                    gemma4_assistant_mtp_telemetry
                        .merge_from(result.gemma4_assistant_mtp_telemetry);
                    decode_telemetry.merge_from(result.decode_telemetry);
                    gemma4_moe_profile.merge_from(result.gemma4_moe_profile);
                    moe_profile.merge_from(result.moe_profile);
                    linear_attention_profile.merge_from(result.linear_attention_profile);
                    dense_ffn_fastpath.merge_from(result.dense_ffn_fastpath);
                    prefill_profile.merge_from(result.prefill_profile);
                    decode_profile.merge_from(result.decode_profile);
                    kv_cache.merge_from(result.kv_usage);
                    prefix_cache.merge_from(result.prefix_cache);
                    request_updates.push(result.update);
                }
            }
        }
        if decode_candidate_count >= 2 {
            if !batched_enabled {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_batched_decode_rejected_flag_disabled".to_string(),
                    decode_candidate_count as u32,
                ));
            } else if self.has_mtp() && self.mtp_requested && !mtp_multirow_batch {
                let rejected =
                    decode_candidate_count.saturating_sub(gemma4_assistant_mtp_coalesced_rows);
                if rejected > 0 {
                    route_metadata.crossover_decisions.push((
                        "ax_mlx_batched_decode_rejected_mtp_requested".to_string(),
                        rejected as u32,
                    ));
                }
            } else {
                for reason in &self.batched_decode_model_rejections {
                    route_metadata.crossover_decisions.push((
                        format!("ax_mlx_batched_decode_rejected_{reason}"),
                        decode_candidate_count as u32,
                    ));
                }
            }
        }
        if batched_enabled && self.batched_decode_model_eligible {
            let (direct_pipeline_pending, suspended_request_ids, bypassed_request_ids) = {
                let states = self.states.lock();
                let direct_pipeline_pending = states
                    .iter()
                    .filter_map(|(request_id, state)| {
                        state.pending_direct.is_some().then_some(request_id.0)
                    })
                    .collect::<std::collections::HashSet<_>>();
                let suspended_request_ids = states
                    .iter()
                    .filter_map(|(request_id, state)| {
                        state
                            .mtp_suspended_for_batched_decode
                            .then_some(request_id.0)
                    })
                    .collect::<std::collections::HashSet<_>>();
                let bypassed_request_ids = states
                    .iter()
                    .filter_map(|(request_id, state)| state.mtp_bypassed.then_some(request_id.0))
                    .collect::<std::collections::HashSet<_>>();
                (
                    direct_pipeline_pending,
                    suspended_request_ids,
                    bypassed_request_ids,
                )
            };
            let mut session = self.batched_session.lock();
            // A session forward advances every resident row. Evict any row the
            // scheduler did not select for this turn (or that is no longer
            // eligible) before stepping, restoring its complete cache so it
            // can safely resume later without a phantom token/update.
            let eligible_request_ids = input
                .execution_batch
                .items
                .iter()
                .filter_map(|item| {
                    let ctx = input
                        .request_contexts
                        .iter()
                        .find(|ctx| ctx.request_id == item.request_id);
                    let mtp_suspended =
                        mtp_multirow_batch || suspended_request_ids.contains(&item.request_id.0);
                    self.batched_item_eligible(item, ctx, mtp_suspended)
                        .then_some(item.request_id.0)
                })
                .collect::<std::collections::HashSet<_>>();
            let residents_to_write_back = session
                .active_ids()
                .iter()
                .copied()
                .filter(|id| !eligible_request_ids.contains(id))
                .collect::<Vec<_>>();
            if !residents_to_write_back.is_empty() {
                let mut states = self.states.lock();
                for id in &residents_to_write_back {
                    if let Some(cache) = session.writeback_remove(*id)
                        && let Some(state) = states.get_mut(&RequestId(*id))
                    {
                        restore_private_cache_after_batched_writeback(state, cache);
                    }
                }
                route_metadata.crossover_decisions.push((
                    "ax_mlx_batched_decode_unscheduled_writebacks".to_string(),
                    residents_to_write_back.len() as u32,
                ));
            }
            let resident: std::collections::HashSet<u64> =
                session.active_ids().iter().copied().collect();
            let room = session.capacity().saturating_sub(session.len());
            let mut resident_items: Vec<usize> = Vec::new();
            let mut joiner_items: Vec<usize> = Vec::new();
            let mut unseedable_joiners = 0usize;
            for (i, item) in input.execution_batch.items.iter().enumerate() {
                let ctx = input
                    .request_contexts
                    .iter()
                    .find(|c| c.request_id == item.request_id);
                let mtp_suspended =
                    mtp_multirow_batch || suspended_request_ids.contains(&item.request_id.0);
                if !self.batched_item_eligible(item, ctx, mtp_suspended) {
                    continue;
                }
                if resident.contains(&item.request_id.0) {
                    resident_items.push(i);
                } else {
                    let seedable = self
                        .states
                        .lock()
                        .get(&item.request_id)
                        .is_some_and(|state| session.can_seed(&state.cache));
                    if seedable {
                        joiner_items.push(i);
                    } else {
                        unseedable_joiners = unseedable_joiners.saturating_add(1);
                    }
                }
            }
            if unseedable_joiners > 0 {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_batched_decode_rejected_compacted_cache".to_string(),
                    unseedable_joiners as u32,
                ));
            }
            let admission_blocked =
                batched_admission_blocked_by_memory_pressure(input.memory_pressure.as_deref());
            if admission_blocked && !joiner_items.is_empty() {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_batched_decode_admission_blocked_pressure".to_string(),
                    joiner_items.len() as u32,
                ));
                joiner_items.clear();
            } else if input
                .memory_pressure
                .as_deref()
                .is_some_and(|pressure| pressure.starts_with("kv_low_free_blocks:"))
            {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_batched_decode_admission_low_pressure".to_string(),
                    joiner_items.len() as u32,
                ));
            }
            joiner_items.truncate(room);
            let bootstrap_joiners = joiner_items
                .iter()
                .copied()
                .filter(|index| {
                    direct_pipeline_pending
                        .contains(&input.execution_batch.items[*index].request_id.0)
                })
                .collect::<Vec<_>>();
            let bootstrap_transition = resident_items.is_empty() && bootstrap_joiners.len() >= 2;
            if bootstrap_transition {
                let group_items = bootstrap_joiners
                    .iter()
                    .map(|&index| &input.execution_batch.items[index])
                    .collect::<Vec<_>>();
                let updates = self.join_direct_pipeline_group(
                    &mut session,
                    &group_items,
                    &input.request_contexts,
                );
                request_updates.extend(updates);
                batched_idx.extend(bootstrap_joiners);
            } else {
                joiner_items.retain(|index| {
                    !direct_pipeline_pending
                        .contains(&input.execution_batch.items[*index].request_id.0)
                });
            }
            // Session-resident requests MUST batch (their state.cache is dormant);
            // otherwise start only once >= 2 eligible requests are present.
            let should_batch = !bootstrap_transition
                && (!resident_items.is_empty() || resident_items.len() + joiner_items.len() >= 2);
            if should_batch {
                let group: Vec<usize> = resident_items.into_iter().chain(joiner_items).collect();
                let mtp_transition_ids = if mtp_multirow_batch {
                    group
                        .iter()
                        .filter_map(|index| {
                            let id = input.execution_batch.items[*index].request_id.0;
                            (!suspended_request_ids.contains(&id)
                                && !bypassed_request_ids.contains(&id))
                            .then_some(id)
                        })
                        .collect::<std::collections::HashSet<_>>()
                } else {
                    std::collections::HashSet::new()
                };
                mtp_suspended_batched_rows = group
                    .iter()
                    .filter(|index| {
                        let id = input.execution_batch.items[**index].request_id.0;
                        suspended_request_ids.contains(&id) || mtp_transition_ids.contains(&id)
                    })
                    .count();
                let group_items: Vec<&ax_engine_core::ExecutionItem> = group
                    .iter()
                    .map(|&i| &input.execution_batch.items[i])
                    .collect();
                let updates = self.run_batched_decode_group(
                    &mut session,
                    &group_items,
                    &input.request_contexts,
                    &mtp_transition_ids,
                );
                request_updates.extend(updates);
                batched_forward_rows = group.len();
                // Extend, not overwrite: `batched_idx` may already hold rows
                // suppressed by the assistant-MTP coalesced group above (3541);
                // replacing it here would re-execute those rows this step.
                batched_idx.extend(group);
            }
        }

        // `batched_idx` is the execution-suppression set for every grouped
        // route, including assistant-MTP verifies handled above. Keep the
        // historical tensor-batch counter scoped to the dense tensor/session
        // route so benchmark contracts cannot mistake an MTP barrier
        // coalescing win for a tensor-batched decode promotion.
        let tensor_batched_rows = batched_idx
            .len()
            .saturating_sub(gemma4_assistant_mtp_coalesced_rows);

        // A tensor batch is promoted only by a real-weight numerical
        // certificate. The fallback below preserves each row's batch=1 graph
        // and therefore its sequential reduction order, but still coalesces
        // MLX submission/completion work across sibling Qwen/Gemma requests.
        if batched_enabled && row_exact_coalescing_family(&self.cfg.model_family) {
            let pending_ids = self
                .states
                .lock()
                .iter()
                .filter_map(|(request_id, state)| {
                    state.pending_direct.is_some().then_some(request_id.0)
                })
                .collect::<std::collections::HashSet<_>>();
            let group = input
                .execution_batch
                .items
                .iter()
                .enumerate()
                .filter(|(index, item)| {
                    !batched_idx.contains(index)
                        && pending_ids.contains(&item.request_id.0)
                        && self.batched_item_eligible(
                            item,
                            input
                                .request_contexts
                                .iter()
                                .find(|ctx| ctx.request_id == item.request_id),
                            false,
                        )
                })
                .collect::<Vec<_>>();
            if group.len() >= 2 {
                let (runs, forward_rows) =
                    self.run_row_exact_coalesced_group(&group, &input.request_contexts);
                row_exact_coalesced_rows = runs.len();
                row_exact_coalesced_forward_rows = forward_rows;
                for (item_index, result) in runs {
                    batched_idx.insert(item_index);
                    ngram_acceleration.merge_from(result.ngram_acceleration);
                    mtp_telemetry.merge_from(result.mtp_telemetry);
                    flash_next_mtp_telemetry.merge_from(result.flash_next_mtp_telemetry);
                    gemma4_assistant_mtp_telemetry
                        .merge_from(result.gemma4_assistant_mtp_telemetry);
                    decode_telemetry.merge_from(result.decode_telemetry);
                    gemma4_moe_profile.merge_from(result.gemma4_moe_profile);
                    moe_profile.merge_from(result.moe_profile);
                    linear_attention_profile.merge_from(result.linear_attention_profile);
                    dense_ffn_fastpath.merge_from(result.dense_ffn_fastpath);
                    prefill_profile.merge_from(result.prefill_profile);
                    decode_profile.merge_from(result.decode_profile);
                    kv_cache.merge_from(result.kv_usage);
                    prefix_cache.merge_from(result.prefix_cache);
                    request_updates.push(result.update);
                }
            }
        }
        let pure_single_token_decode = input.execution_batch.items.iter().all(|item| {
            matches!(item.mode, ExecutionMode::Decode) && item.input_token_slice.len() == 1
        });
        let skip_route_telemetry =
            pure_single_token_decode && crate::fastpath::skip_decode_route_telemetry();

        if !skip_route_telemetry {
            if tensor_batched_rows > 0 {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_batched_decode_rows".into(),
                    tensor_batched_rows as u32,
                ));
            }
            if batched_forward_rows > 0 {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_batched_decode_forward_rows".into(),
                    batched_forward_rows as u32,
                ));
            }
            if row_exact_coalesced_rows > 0 {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_row_exact_coalesced_decode_rows".into(),
                    row_exact_coalesced_rows as u32,
                ));
                route_metadata.crossover_decisions.push((
                    "ax_mlx_row_exact_coalesced_decode_forward_rows".into(),
                    row_exact_coalesced_forward_rows as u32,
                ));
                route_metadata
                    .crossover_decisions
                    .push(("ax_mlx_row_exact_coalesced_decode_eval_barriers".into(), 1));
            }
            if gemma4_assistant_mtp_coalesced_rows > 0 {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_gemma4_assistant_mtp_coalesced_verify_rows".into(),
                    gemma4_assistant_mtp_coalesced_rows as u32,
                ));
                route_metadata.crossover_decisions.push((
                    "ax_mlx_gemma4_assistant_mtp_coalesced_verify_forward_rows".into(),
                    gemma4_assistant_mtp_coalesced_rows as u32,
                ));
                route_metadata.crossover_decisions.push((
                    "ax_mlx_gemma4_assistant_mtp_coalesced_verify_eval_barriers".into(),
                    1,
                ));
            }
            if mtp_multirow_batch {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_mtp_multirow_batch_candidate_rows".into(),
                    mtp_multirow_candidate_count as u32,
                ));
            }
            if mtp_suspended_batched_rows > 0 {
                route_metadata.crossover_decisions.push((
                    "ax_mlx_mtp_multirow_batch_suspended_rows".into(),
                    mtp_suspended_batched_rows as u32,
                ));
            }
            route_metadata.crossover_decisions.extend([
                ("ax_mtp_available".into(), u32::from(self.has_mtp())),
                ("ax_mtp_requested".into(), u32::from(self.mtp_requested)),
            ]);
        }

        // ── Padded batched prefill (AX_MLX_BATCHED_PREFILL, default off).
        // Eligible cold text prefill cohorts run one shared padded forward
        // here; each row's result is consumed inside its own run_item call
        // below, so per-request bookkeeping and ordering are unchanged.
        let mut staged_prefill = self.stage_batched_prefill(&input, &mut route_metadata);
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
                staged_prefill.remove(&item.request_id.0),
            );
            ngram_acceleration.merge_from(result.ngram_acceleration);
            mtp_telemetry.merge_from(result.mtp_telemetry);
            flash_next_mtp_telemetry.merge_from(result.flash_next_mtp_telemetry);
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
            request_updates.push(result.update);
        }
        if std::env::var_os("AX_MLX_MTP_DEBUG_SUMMARY").is_some() && mtp_telemetry.decode_steps > 0
        {
            eprintln!(
                "AX_MTP_DEBUG steps={} drafted={} accepted={} emitted={} verify_tokens={} full={} partial={} miss={} clone_us={} verify_forward_us={} verify_eval_us={} accept_us={} rollback_us={} tail_us={} draft_us={} softmax_us={} depth_accepted={:?} depth_drafted={:?}",
                mtp_telemetry.decode_steps,
                mtp_telemetry.draft_tokens,
                mtp_telemetry.accepted_tokens,
                mtp_telemetry.emitted_tokens,
                mtp_telemetry.verify_tokens,
                mtp_telemetry.full_accept_steps,
                mtp_telemetry.partial_reject_steps,
                mtp_telemetry.complete_miss_steps,
                mtp_telemetry.cache_clone_wall_us,
                mtp_telemetry.verify_forward_wall_us,
                mtp_telemetry.verify_eval_wall_us,
                mtp_telemetry.accept_wall_us,
                mtp_telemetry.rollback_wall_us,
                mtp_telemetry.tail_sample_wall_us,
                mtp_telemetry.draft_wall_us,
                mtp_telemetry.target_softmax_wall_us,
                mtp_telemetry.accepted_by_depth,
                mtp_telemetry.drafted_by_depth,
            );
        }
        if std::env::var_os("AX_MLX_MTP_DEBUG_SUMMARY").is_some() && prefill_profile.enabled != 0 {
            eprintln!(
                "AX_PREFILL_PROFILE steps={} layers={} tokens={} pre_sdpa_us={} qkv_us={} qknorm_us={} rope_kv_us={} sdpa_us={} post_attn_us={} ffn_us={} gate_up_us={} activation_us={} down_us={} oproj_us={} residual_norm_us={} residual_gate_us={} lm_head_us={}",
                prefill_profile.prefill_steps,
                prefill_profile.layers,
                prefill_profile.tokens,
                prefill_profile.pre_sdpa_wall_us,
                prefill_profile.pre_sdpa_qkv_proj_wall_us,
                prefill_profile.pre_sdpa_qk_norm_wall_us,
                prefill_profile.pre_sdpa_rope_kv_wall_us,
                prefill_profile.sdpa_wall_us,
                prefill_profile.post_attn_wall_us,
                prefill_profile.post_attn_ffn_wall_us,
                prefill_profile.post_attn_ffn_gate_up_wall_us,
                prefill_profile.post_attn_ffn_activation_wall_us,
                prefill_profile.post_attn_ffn_down_wall_us,
                prefill_profile.post_attn_output_proj_wall_us,
                prefill_profile.post_attn_residual_norm_wall_us,
                prefill_profile.post_attn_residual_gate_wall_us,
                prefill_profile.lm_head_wall_us,
            );
        }
        if std::env::var_os("AX_MLX_MTP_DEBUG_SUMMARY").is_some()
            && linear_attention_profile.enabled != 0
        {
            eprintln!(
                "AX_LINEAR_PROFILE layers={} tokens={} projection_us={} qkvz_us={} ba_us={} qkv_us={} z_us={} a_us={} b_us={} conv_us={} qknorm_us={} recurrent_us={} output_us={} direct_inputs={}/{}/{} blocked={} post_input={}/{}/{} blocked={} metal={}/{}/{} blocked={}",
                linear_attention_profile.layers,
                linear_attention_profile.tokens,
                linear_attention_profile.projection_wall_us,
                linear_attention_profile.projection_qkvz_wall_us,
                linear_attention_profile.projection_ba_wall_us,
                linear_attention_profile.projection_qkv_wall_us,
                linear_attention_profile.projection_z_wall_us,
                linear_attention_profile.projection_a_wall_us,
                linear_attention_profile.projection_b_wall_us,
                linear_attention_profile.conv_wall_us,
                linear_attention_profile.qk_norm_wall_us,
                linear_attention_profile.recurrent_wall_us,
                linear_attention_profile.output_wall_us,
                linear_attention_profile.direct_cpp_inputs_attempts,
                linear_attention_profile.direct_cpp_inputs_hits,
                linear_attention_profile.direct_cpp_inputs_fallbacks,
                linear_attention_profile.direct_cpp_inputs_profile_blocked,
                linear_attention_profile.direct_cpp_post_input_attempts,
                linear_attention_profile.direct_cpp_post_input_hits,
                linear_attention_profile.direct_cpp_post_input_fallbacks,
                linear_attention_profile.direct_cpp_post_input_profile_blocked,
                linear_attention_profile.decode_post_input_metal_attempts,
                linear_attention_profile.decode_post_input_metal_hits,
                linear_attention_profile.decode_post_input_metal_fallbacks,
                linear_attention_profile.decode_post_input_metal_profile_blocked,
            );
        }
        // Always export decode path counters (pipeline / single / bootstrap).
        // `AX_MLX_SKIP_DECODE_ROUTE_TELEMETRY` only omits the heavy profile /
        // layout maps used for residual campaigns — not the effective-route
        // identity counters required by README publication gates.
        {
            let mut route_decisions =
                IndexedRouteDecisions::new(&mut route_metadata.crossover_decisions);
            decode_telemetry.append_route_decisions(&mut route_decisions);
            let selected_reads = crate::expert_stream::take_selected_expert_read_stats();
            route_decisions.upsert_route_decision(
                "ax_mlx_flash_next_selected_expert_gathers",
                u32::try_from(selected_reads.gathers).unwrap_or(u32::MAX),
            );
            route_decisions.upsert_route_decision(
                "ax_mlx_flash_next_selected_expert_payload_kib",
                u32::try_from(selected_reads.payload_bytes / 1024).unwrap_or(u32::MAX),
            );
            if let Some(trunk) = self.weights.qwen4_exp.as_ref() {
                route_decisions.upsert_route_decision(
                    "ax_mlx_flash_next_mtp_verifier_schedule",
                    crate::model::qwen4_exp_mtp::target_schedule_route_code(trunk),
                );
            }
            if !skip_route_telemetry {
                ngram_acceleration.append_route_decisions(&mut route_decisions);
                mtp_telemetry.append_route_decisions(&mut route_decisions);
                flash_next_mtp_telemetry.append_route_decisions(&mut route_decisions);
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
        }

        let tokens_written: u32 = input
            .execution_batch
            .items
            .iter()
            .map(|i| i.scheduled_token_count)
            .sum();

        // A recoverable expert paging failure during any forward of this
        // step cannot return through the generic MoE path; it is recorded
        // per thread and turned into a per-request error here so the engine
        // fails those requests and the process (and sibling models) go on.
        if let Some(message) = crate::expert_stream::take_paging_failure() {
            tracing::error!(
                target: "ax_engine_mlx::runner",
                %message,
                "expert paging failed during the step; failing its requests"
            );
            for update in &mut request_updates {
                update.error = Some(message.clone());
                update.output_token = None;
                update.output_tokens.clear();
                update.tokens_executed = 0;
            }
        }

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

    fn native_expert_streaming_active(&self) -> bool {
        self.weights.expert_stream.is_some()
    }

    fn release_request_state(&self, request_id: RequestId) {
        // Terminal cleanup for requests that never reach a runner-observed stop
        // (cancelled while waiting/blocked, or cancelled by the engine after a
        // step reinserted state). Without this the per-request KV cache, MTP,
        // and n-gram state stay resident for the life of the process.
        // Drain a leftover double-buffer token so its lazy Metal graph is not
        // left in the process residency set for the next request's prefill.
        let pending = self
            .states
            .lock()
            .remove(&request_id)
            .and_then(|mut state| state.pending_direct.take());
        self.batched_session.lock().remove(request_id.0);
        if let Some(pending) = pending {
            eval(&[&pending]);
        }
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
        let (hidden, dense_head_fused) = self.embedding_forward(token_ids, target_position);
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
        // Compiled path fuses Dense head; imperative fallback needs it here.
        let pooled = if dense_head_fused {
            pooled
        } else {
            crate::model::apply_embedding_dense_head(&self.weights, &pooled)
        };
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
        // Prefer flat path (includes length-affinity split + buckets).
        let matrix = self.embed_batch_flat(batch, pooling, normalize)?;
        Ok((0..matrix.batch_size)
            .map(|i| matrix.row(i).to_vec())
            .collect())
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
        // Length-affinity split (default ON): process similar-length rows together
        // so right-pad waste stays bounded, then reassemble original order.
        if crate::model::embed_length_split_enabled() && batch.len() > 1 {
            let lens: Vec<usize> = batch.iter().map(Vec::len).collect();
            let groups = crate::model::embed_length_affinity_groups(&lens, 1.5, 32);
            if groups.len() > 1 {
                let mut row_data: Vec<Option<Vec<f32>>> = (0..batch.len()).map(|_| None).collect();
                let mut hidden_size = 0usize;
                for group in groups {
                    let sub: Vec<Vec<u32>> = group.iter().map(|&i| batch[i].clone()).collect();
                    let mat = self.embed_batch_flat_contiguous(&sub, pooling, normalize)?;
                    hidden_size = mat.hidden_size;
                    for (j, &orig) in group.iter().enumerate() {
                        row_data[orig] = Some(mat.row(j).to_vec());
                    }
                }
                let mut data = Vec::with_capacity(batch.len() * hidden_size);
                for row in row_data {
                    let row = row.ok_or("embedding length-split output missing")?;
                    data.extend_from_slice(&row);
                }
                return Ok(ax_engine_core::EmbeddingMatrix {
                    data,
                    batch_size: batch.len(),
                    hidden_size,
                });
            }
        }
        self.embed_batch_flat_contiguous(batch, pooling, normalize)
    }
}

impl MlxRunner {
    /// Single pad/compile-key embed for one length-homogeneous (or unsplit) batch.
    /// Kept off the `ExecutionRunner` trait so length-split can call it without recursion.
    fn embed_batch_flat_contiguous(
        &self,
        batch: &[Vec<u32>],
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<ax_engine_core::EmbeddingMatrix, &'static str> {
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
        let (hidden, actual_lens, dense_head_fused) =
            self.embedding_batch_forward(batch, target_positions.as_deref());
        let batch_size = batch.len() as i32;
        let pooled = match pooling {
            EmbeddingPooling::Mean => bf16_mean_pool(&hidden, &actual_lens, batch_size),
            EmbeddingPooling::Last | EmbeddingPooling::Cls => hidden,
        };
        let pooled = if dense_head_fused {
            pooled
        } else {
            crate::model::apply_embedding_dense_head(&self.weights, &pooled)
        };
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
    ///
    /// Returns `(output, dense_head_fused)`: when the compiled path was used
    /// and the model has a Dense head, the head is fused into the closure and
    /// the caller must NOT apply it again.
    fn embedding_forward(
        &self,
        token_ids: &[u32],
        target_position: Option<usize>,
    ) -> (MlxArray, bool) {
        // Fuse Dense head only for Last/Cls pooling (target_position.is_some());
        // mean pooling applies Dense head after pooling (outside the closure).
        let has_dense_head = self.weights.embedding_dense_0.is_some() && target_position.is_some();
        // DI-W2-002: production single-item EmbeddingGemma must use the same
        // bidirectional Gemma3 sandwich path as batch-of-one. The default
        // compiled dense body (`build_embedding_forward_closure` →
        // `forward_for_embedding_body`) is causal and wrong for this family.
        if embedding_single_item_uses_gemma3_path(&self.cfg.model_family) {
            if *EMBED_NO_COMPILE {
                let target_positions = target_position.map(|p| vec![p]);
                let (out, _lens) = crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    &[token_ids.to_vec()],
                    target_positions.as_deref(),
                );
                return (out, false);
            }
            let (out, _lens) = self.embedding_gemma_batch_compiled_forward(&[token_ids.to_vec()]);
            return (out, false);
        }
        if *EMBED_NO_COMPILE {
            return (
                crate::model::forward_for_embedding(
                    &self.cfg,
                    &self.weights,
                    token_ids,
                    target_position,
                ),
                false,
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

        let key: EmbedCompileKey = (
            thread::current().id(),
            token_ids.len(),
            target_position,
            has_dense_head,
        );
        let mut cache = self.embed_compile_cache.lock();
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                target_position,
                has_dense_head,
            ) {
                Ok(cls) => {
                    embed_compile_cache_evict_if_full(&mut cache);
                    cache.insert(key, cls);
                }
                Err(err) => {
                    tracing::warn!(
                        target = "ax_engine_mlx",
                        path = "embed_single",
                        reason = %err,
                        "compiled_closure_build_failed"
                    );
                    COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    drop(cache);
                    return (
                        crate::model::forward_for_embedding(
                            &self.cfg,
                            &self.weights,
                            token_ids,
                            target_position,
                        ),
                        false,
                    );
                }
            }
        }
        {
            let mut stats = self.embed_compile_stats.lock();
            if was_present {
                stats.single_hits += 1;
                COMPILE_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                stats.single_misses += 1;
                COMPILE_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        let outputs = match cls.try_apply(&[&hidden]) {
            Ok(outputs) => outputs,
            Err(err) => {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "embed_single",
                    reason = %err,
                    "compiled_closure_apply_failed"
                );
                COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cache.remove(&key);
                drop(cache);
                return (
                    crate::model::forward_for_embedding(
                        &self.cfg,
                        &self.weights,
                        token_ids,
                        target_position,
                    ),
                    false,
                );
            }
        };
        match outputs.into_iter().next() {
            Some(out) => (out, has_dense_head),
            None => {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "embed_single",
                    reason = "empty_output",
                    "compiled_closure_apply_failed"
                );
                COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cache.remove(&key);
                drop(cache);
                (
                    crate::model::forward_for_embedding(
                        &self.cfg,
                        &self.weights,
                        token_ids,
                        target_position,
                    ),
                    false,
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
    ///
    /// Returns `(output, actual_lens, dense_head_fused)`.
    fn embedding_batch_forward(
        &self,
        batch_token_ids: &[Vec<u32>],
        target_positions: Option<&[usize]>,
    ) -> (MlxArray, Vec<usize>, bool) {
        let profile = crate::model::profile::embed_profile_enabled();
        // Fuse Dense head only for Last/Cls pooling (target_positions.is_some());
        // mean pooling applies Dense head after pooling (outside the closure).
        let has_dense_head = self.weights.embedding_dense_0.is_some() && target_positions.is_some();

        // EmbeddingGemma: bidirectional encoder with mean pooling — uses a
        // dedicated compile cache keyed on (batch, max_len, actual_lens).
        if self.cfg.model_family == "embeddinggemma" {
            if *EMBED_NO_COMPILE || profile {
                let (out, lens) = crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    target_positions,
                );
                return (out, lens, false);
            }
            let (out, lens) = self.embedding_gemma_batch_compiled_forward(batch_token_ids);
            return (out, lens, false);
        }

        // Nemotron Embed: always use the imperative path so pad-aware
        // bidirectional masks see real per-row lengths (compiled mean-pool
        // traces bake full-seq masks).
        if self.cfg.model_family == "nemotron_embed" || profile {
            let (out, lens) = crate::model::forward_for_embedding_batch(
                &self.cfg,
                &self.weights,
                batch_token_ids,
                target_positions,
            );
            return (out, lens, false);
        }

        // Standard (Qwen3-style) embedding path.
        // AX_MLX_EMBED_PROFILE forces the imperative path: the per-stage eval
        // barriers cannot live inside a single traced compiled closure, and
        // compile on/off is throughput-neutral for this path, so the imperative
        // breakdown is representative.

        // Mean-pool compiled closure path: the layer loop + final norm are
        // fused; mean-pool masking is applied post-closure by the caller.
        if target_positions.is_none() {
            if let Some((out, lens)) = self.embedding_mean_pool_compiled_forward(batch_token_ids) {
                return (out, lens, false);
            }
            // Fall through to imperative mean-pool path.
            let (out, lens) = crate::model::forward_for_embedding_batch(
                &self.cfg,
                &self.weights,
                batch_token_ids,
                target_positions,
            );
            return (out, lens, false);
        }

        if *EMBED_NO_COMPILE {
            let (out, lens) = crate::model::forward_for_embedding_batch(
                &self.cfg,
                &self.weights,
                batch_token_ids,
                target_positions,
            );
            return (out, lens, false);
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
            has_dense_head,
        );
        let mut cache = self.embed_batch_compile_cache.lock();
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_batch_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
                Some(target_positions_vec.clone()),
                has_dense_head,
            ) {
                Ok(cls) => {
                    embed_compile_cache_evict_if_full(&mut cache);
                    cache.insert(key.clone(), cls);
                }
                Err(err) => {
                    tracing::warn!(
                        target = "ax_engine_mlx",
                        path = "embed_batch",
                        reason = %err,
                        "compiled_closure_build_failed"
                    );
                    COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    drop(cache);
                    let (out, lens) = crate::model::forward_for_embedding_batch(
                        &self.cfg,
                        &self.weights,
                        batch_token_ids,
                        target_positions,
                    );
                    return (out, lens, false);
                }
            }
        }
        {
            let mut stats = self.embed_compile_stats.lock();
            if was_present {
                stats.batched_hits += 1;
                COMPILE_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                stats.batched_misses += 1;
                COMPILE_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        let outputs = match cls.try_apply(&[&hidden]) {
            Ok(outputs) => outputs,
            Err(err) => {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "embed_batch",
                    reason = %err,
                    "compiled_closure_apply_failed"
                );
                COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cache.remove(&key);
                drop(cache);
                let (out, lens) = crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    target_positions,
                );
                return (out, lens, false);
            }
        };
        match outputs.into_iter().next() {
            Some(out) => (out, actual_lens, has_dense_head),
            None => {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "embed_batch",
                    reason = "empty_output",
                    "compiled_closure_apply_failed"
                );
                COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cache.remove(&key);
                drop(cache);
                let (out, lens) = crate::model::forward_for_embedding_batch(
                    &self.cfg,
                    &self.weights,
                    batch_token_ids,
                    target_positions,
                );
                (out, lens, false)
            }
        }
    }

    /// Compiled-closure forward for mean-pooled embedding batches. Builds the
    /// pre-embedded hidden state, then applies a compiled closure fusing the
    /// transformer layers + final norm. Returns the full `[B, max_seq, H]`
    /// tensor along with `actual_lens` so the caller can masked-mean-pool.
    /// Returns `None` when the compiled path is disabled or below threshold.
    fn embedding_mean_pool_compiled_forward(
        &self,
        batch_token_ids: &[Vec<u32>],
    ) -> Option<(MlxArray, Vec<usize>)> {
        if *EMBED_NO_COMPILE {
            return None;
        }
        let (hidden, batch, max_len, actual_lens) = crate::model::build_embedding_batch_hidden_pub(
            &self.cfg,
            &self.weights,
            batch_token_ids,
        );

        // Size-gating: skip compilation for small batches.
        let threshold: usize = crate::fastpath::embed_mean_compile_threshold();
        if batch * max_len < threshold {
            return None;
        }

        let thread_id = thread::current().id();
        let key: EmbedMeanPoolCompileKey = (thread_id, batch, max_len);

        let mut cache = self.embed_mean_pool_compile_cache.lock();
        let was_present = cache.contains_key(&key);
        if !was_present {
            match crate::model::build_embedding_mean_pool_forward_closure(
                Arc::clone(&self.cfg_arc),
                Arc::clone(&self.weights),
            ) {
                Ok(cls) => {
                    embed_compile_cache_evict_if_full(&mut cache);
                    cache.insert(key, cls);
                }
                Err(err) => {
                    tracing::warn!(
                        target = "ax_engine_mlx",
                        path = "embed_mean_pool",
                        reason = %err,
                        "compiled_closure_build_failed"
                    );
                    COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
            }
        }
        {
            let mut stats = self.embed_compile_stats.lock();
            if was_present {
                stats.mean_pool_hits += 1;
                COMPILE_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                stats.mean_pool_misses += 1;
                COMPILE_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let cls = cache.get(&key).expect("just inserted");
        match cls.try_apply(&[&hidden]) {
            Ok(outputs) => outputs.into_iter().next().map(|out| (out, actual_lens)),
            Err(err) => {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "embed_mean_pool",
                    reason = %err,
                    "compiled_closure_apply_failed"
                );
                COMPILE_FALLBACK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cache.remove(&key);
                None
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
                    embed_compile_cache_evict_if_full(&mut cache);
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
                    embed_compile_cache_evict_if_full(&mut cache);
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

    /// Plan and execute padded batched prefill for this step's eligible cold
    /// text prefill items (`AX_MLX_BATCHED_PREFILL`, default off). Returns
    /// per-request staged results that `run_item` consumes at its plain
    /// prefill call site — bookkeeping (prefix-cache probe/store, telemetry,
    /// generation-state init) stays entirely on the per-item path, and a
    /// staged row whose preconditions no longer hold there (e.g. a runner
    /// prefix probe restored KV first) is simply dropped in favor of the
    /// sequential path.
    fn stage_batched_prefill(
        &self,
        input: &RunnerInput,
        route_metadata: &mut ax_engine_core::RouteMetadata,
    ) -> std::collections::HashMap<u64, StagedBatchedPrefill> {
        use ax_engine_core::prefill_cohort::{
            BatchedPrefillCapability, PaddedCostLimits, PrefillCohortKind, PrefillRowClass,
            default_padded_token_budget, plan_prefill_cohorts,
        };

        let mut staged = std::collections::HashMap::new();
        // The prefill window, in item (scheduler plan) order.
        let window: Vec<&ax_engine_core::ExecutionItem> = input
            .execution_batch
            .items
            .iter()
            .filter(|item| matches!(item.mode, ExecutionMode::Prefill))
            .collect();
        if window.len() < 2 {
            return staged;
        }
        if !crate::fastpath::batched_prefill_enabled() {
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                "ax_mlx_batched_prefill_rejected_flag_disabled",
                window.len() as u32,
            );
            return staged;
        }
        if !supports_batched_prefill(&self.cfg, &self.weights) {
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                "ax_mlx_batched_prefill_rejected_model_capability",
                window.len() as u32,
            );
            return staged;
        }

        let eligible = |item: &ax_engine_core::ExecutionItem| -> bool {
            let ctx = input
                .request_contexts
                .iter()
                .find(|ctx| ctx.request_id == item.request_id);
            item.position_range.start == 0
                && item.prefix_tokens_reused == 0
                && item.reused_prefix_token_slice.is_empty()
                && !item.input_token_slice.is_empty()
                && prefill_item_completes_prompt(item, ctx)
                && input.request_multimodal_inputs(item.request_id).is_none()
        };
        let rows: Vec<PrefillRowClass> = window
            .iter()
            .map(|item| {
                if eligible(item) {
                    PrefillRowClass {
                        kv_offset_tokens: 0,
                        multimodal: false,
                        prompt_len: item.input_token_slice.len() as u32,
                    }
                } else {
                    // Runner-level ineligibility (adopted prefix, partial
                    // prompt item, multimodal inputs, …) is expressed as a
                    // non-zero KV offset: the planner only needs to know the
                    // row cannot take a padded slot.
                    PrefillRowClass {
                        kv_offset_tokens: 1,
                        multimodal: false,
                        prompt_len: item.input_token_slice.len() as u32,
                    }
                }
            })
            .collect();
        let max_rows = crate::fastpath::batched_prefill_max_rows();
        let limits = PaddedCostLimits {
            max_padded_tokens: crate::fastpath::batched_prefill_token_budget_override()
                .unwrap_or_else(|| {
                    default_padded_token_budget(self.prefill_chunk as u32, max_rows.max(1))
                }),
            max_rows,
        };
        let cohorts = plan_prefill_cohorts(
            &rows,
            BatchedPrefillCapability {
                supports_batched: true,
                supports_padding: true,
            },
            limits,
        );

        let mut staged_rows = 0u32;
        let mut staged_cohorts = 0u32;
        let mut forward_errors = 0u32;
        for cohort in &cohorts {
            if cohort.kind != PrefillCohortKind::BatchedCold {
                continue;
            }
            let members: Vec<&ax_engine_core::ExecutionItem> =
                cohort.members.iter().map(|&index| window[index]).collect();
            let prompts: Vec<&[u32]> = members
                .iter()
                .map(|item| item.input_token_slice.as_slice())
                .collect();
            match prefill_batched_forward(&self.cfg, &self.weights, &prompts) {
                Ok(batch) => {
                    staged_cohorts += 1;
                    for ((item, logits), layer_kv) in
                        members.iter().zip(batch.row_logits).zip(batch.row_layer_kv)
                    {
                        staged_rows += 1;
                        staged.insert(
                            item.request_id.0,
                            StagedBatchedPrefill {
                                prompt_tokens: item.input_token_slice.clone(),
                                logits,
                                layer_kv,
                                seq_len: item.input_token_slice.len(),
                            },
                        );
                    }
                }
                Err(error) => {
                    forward_errors += 1;
                    tracing::warn!(
                        target: "ax_engine_mlx::batched_prefill",
                        %error,
                        rows = members.len(),
                        "padded batched prefill forward failed; cohort falls back to sequential prefill",
                    );
                }
            }
        }
        if staged_rows > 0 {
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                "ax_mlx_batched_prefill_rows",
                staged_rows,
            );
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                "ax_mlx_batched_prefill_cohorts",
                staged_cohorts,
            );
        }
        if forward_errors > 0 {
            upsert_route_decision(
                &mut route_metadata.crossover_decisions,
                "ax_mlx_batched_prefill_forward_errors",
                forward_errors,
            );
        }
        staged
    }

    /// Shared tail of every native multimodal prefill branch: adopt the full
    /// prompt as the request's prefix identity, publish the media-keyed prefix
    /// snapshot when multimodal prefix reuse is on, then seed the generation
    /// state from the sampled first token.
    fn complete_multimodal_prefill(
        &self,
        state: &mut RequestState,
        prefix_cache: &mut MlxPrefixCacheTelemetry,
        tail: MultimodalPrefillTail<'_>,
    ) {
        let MultimodalPrefillTail {
            full_prompt_tokens,
            first_token: tok,
            prefill_started,
            prefill_forward_wall_us,
            publish_prefix_snapshot,
            model_id,
            block_size_tokens,
            media_key,
            ctx,
            sampling,
            max_output,
            is_greedy,
        } = tail;
        state.prompt_prefix_tokens = full_prompt_tokens;

        let mut prefill_prefix_cache_wall_us = 0u32;
        if publish_prefix_snapshot && multimodal_prefix_reuse_enabled() {
            let prefix_cache_started = Instant::now();
            let cold_prefill_us = elapsed_us(prefill_started);
            prefix_cache.merge_from(self.store_prompt_prefix_snapshots(
                model_id,
                block_size_tokens,
                state,
                PromptPrefixSnapshotStoreOptions {
                    linear_boundary_snapshot: None,
                    prefill_completes_prompt: true,
                    greedy_prefill_output_token:
                        prefill_output_token_cacheable(ctx, sampling).then_some(tok),
                    cold_prefill_us: u64::from(cold_prefill_us),
                    media_key,
                },
            ));
            prefill_prefix_cache_wall_us = elapsed_us(prefix_cache_started);
        }

        state
            .decode_telemetry
            .record_prefill(elapsed_us(prefill_started));
        let generation_state_started = Instant::now();
        self.initialize_generation_state(
            state,
            max_output,
            Some(tok),
            is_greedy,
            sampling,
            ctx.map(|c| (c.max_think_tokens, c.answer_reserve_tokens))
                .unwrap_or((None, None)),
        );
        let prefill_generation_state_wall_us = elapsed_us(generation_state_started);
        state.decode_telemetry.record_prefill_eval_barrier();
        state.decode_telemetry.record_prefill_breakdown(
            prefill_forward_wall_us,
            prefill_prefix_cache_wall_us,
            prefill_generation_state_wall_us,
        );
    }

    fn run_item(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
        model_id: &str,
        block_size_tokens: u32,
        multimodal_inputs: Option<&RequestMultimodalInputs>,
        mut staged_batched_prefill: Option<StagedBatchedPrefill>,
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
                    diffusion_schedule: None,
                },
                ngram_acceleration: NgramAccelerationTelemetry::default(),
                mtp_telemetry: MtpTelemetry::default(),
                flash_next_mtp_telemetry: FlashNextMtpTelemetry::default(),
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
        let unlimited_ocr_inputs = multimodal_inputs
            .and_then(|inputs| inputs.unlimited_ocr.as_ref())
            .filter(|inputs| !inputs.is_empty());
        let qwen3_vl_inputs = multimodal_inputs
            .and_then(|inputs| inputs.qwen3_vl.as_ref())
            .filter(|inputs| !inputs.is_empty());
        let minicpm_v46_inputs = multimodal_inputs
            .and_then(|inputs| inputs.minicpm_v46.as_ref())
            .filter(|inputs| !inputs.is_empty());
        let nemotron_omni_inputs = multimodal_inputs
            .and_then(|inputs| inputs.nemotron_omni.as_ref())
            .filter(|inputs| !inputs.is_empty());
        let multimodal_provider_count = usize::from(gemma4_unified_inputs.is_some())
            + usize::from(unlimited_ocr_inputs.is_some())
            + usize::from(qwen3_vl_inputs.is_some())
            + usize::from(minicpm_v46_inputs.is_some())
            + usize::from(nemotron_omni_inputs.is_some());
        if multimodal_provider_count > 1 {
            return errored_item_run(
                item.request_id,
                "multimodal request may select only one provider schema",
            );
        }
        let has_gemma4_unified_multimodal_prefill = is_prefill && gemma4_unified_inputs.is_some();
        let has_unlimited_ocr_multimodal_prefill = is_prefill && unlimited_ocr_inputs.is_some();
        let has_qwen3_vl_multimodal_prefill = is_prefill && qwen3_vl_inputs.is_some();
        let has_minicpm_v46_multimodal_prefill = is_prefill && minicpm_v46_inputs.is_some();
        let has_nemotron_omni_multimodal_prefill = is_prefill && nemotron_omni_inputs.is_some();
        let has_native_multimodal_prefill = has_gemma4_unified_multimodal_prefill
            || has_unlimited_ocr_multimodal_prefill
            || has_qwen3_vl_multimodal_prefill
            || has_minicpm_v46_multimodal_prefill
            || has_nemotron_omni_multimodal_prefill;
        // ADR-038 Phase 4: multimodal is a prefill adapter into the same
        // generation strategy — never a parallel generation engine.
        if let Some(inputs) = multimodal_inputs {
            let adapter =
                MultimodalPrefillAdapter::from_request_inputs(inputs, self.cfg.generation_kind);
            debug_assert!(
                !adapter.is_separate_generation_engine(),
                "multimodal must not invent a separate generation engine"
            );
            debug_assert_eq!(
                adapter.feeds_generation, self.cfg.generation_kind,
                "multimodal adapter must feed the model generation kind"
            );
            let _ = adapter.requires_prefill_projection;
        }
        let mut gemma4_unified_multimodal_telemetry = Gemma4UnifiedMultimodalTelemetry::default();
        if has_gemma4_unified_multimodal_prefill && let Some(inputs) = gemma4_unified_inputs {
            gemma4_unified_multimodal_telemetry.record_prefill(inputs, self.has_mtp());
        }
        let sampling = ctx
            .map(|c| {
                MlxSamplingParams::new(c.temperature, c.top_p, c.top_k)
                    .with_min_p(c.min_p)
                    .with_repetition_penalty(c.repetition_penalty, c.repetition_context_size)
                    .with_no_repeat_ngram(c.no_repeat_ngram_size, c.ngram_window)
                    // `seed == 0` is the native no-seed default; only a non-zero
                    // seed marks an explicitly reproducible request.
                    .with_seed((c.seed != 0).then_some(c.seed))
            })
            .unwrap_or_default();
        // Prefer the engine's deterministic-argmax bit, but also treat
        // temperature-0 / default-greedy shaped contexts as greedy so pure
        // direct sessions never miss the double-buffer pipeline when a session
        // flag omitted `deterministic` for temperature-0 requests.
        let is_greedy = ctx
            .map(|c| {
                c.deterministic_argmax_sampling
                    || (c.temperature <= 0.0
                        && c.top_k == 0
                        && c.top_p >= 1.0
                        && (c.repetition_penalty - 1.0).abs() < f32::EPSILON
                        && c.no_repeat_ngram_size == 0)
            })
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
                RequestState::new_with_shared_fa_pool(
                    self.cfg.layer_count,
                    sampling.seed,
                    item.request_id.0,
                    self.fa_block_pool_config,
                    self.shared_fa_block_pool.clone(),
                )
            })
        };
        // WS-M3: when multimodal prefix reuse is enabled, restore like text
        // (media identity is folded into the prefix key via media_key).
        // `None` = media present without a digest; restore and store both
        // fail closed on it rather than using a text-only key.
        let media_key = self.media_key_from_inputs(multimodal_inputs);
        let mut prefix_cache =
            if has_native_multimodal_prefill && !multimodal_prefix_reuse_enabled() {
                MlxPrefixCacheTelemetry::default()
            } else {
                self.restore_reused_prefix_state(
                    &mut state,
                    item,
                    ctx,
                    model_id,
                    block_size_tokens,
                    sampling,
                    !has_native_multimodal_prefill
                        && self.flash_next_mtp_prefill_eligible(is_greedy, sampling),
                    media_key.as_deref(),
                )
            };
        self.reclaim_native_prefix_capacity(
            &state.cache,
            item.input_token_slice.len(),
            &mut prefix_cache,
        );

        // Phase 3b: per-layer KV-cache quantization from the model manifest.
        // Injected after prefix restore because restore can adopt a serialized
        // snapshot that replaces `state.cache` wholesale (the wire format is
        // dense; adopted snapshots re-quantize on the first append). The call
        // is idempotent and honors the `AX_KV_QUANT=0` kill-switch.
        state
            .cache
            .set_kv_quant_table(self.cfg.kv_cache_quant.clone());

        // Apply the request's rotating sliding-KV decision. Prefill may rotate
        // when `AX_MLX_ROTATING_SLIDING_PREFILL=1` (default OFF): SWA layers
        // then keep O(window+slack) physical storage instead of O(context)
        // contiguous buffers. Default stays ordered so pure window decode can
        // convert cleanly without carrying oversized prefill ring capacity
        // through long-context tok/s. Decode still latches in
        // `initialize_generation_state`; the latch wins on later runs.
        let (rotate_sliding, rotate_slack) = cache_rotation_for_execution(
            item.mode,
            state.rotating_sliding_latch,
            self.rotating_sliding_decode,
            is_greedy,
            self.prefill_chunk,
        );
        state.cache.set_rotating_sliding_decode(rotate_sliding);
        state.cache.set_rotating_sliding_slack(rotate_slack);
        if item.mode == ExecutionMode::Prefill && state.rotating_sliding_latch.is_none() {
            state.rotating_sliding_latch = Some((rotate_sliding, rotate_slack));
        }

        // GPU work — mutex is NOT held during prefill, decode, or n-gram acceleration steps.
        let sampled_tokens = match item.mode {
            ExecutionMode::Prefill => {
                let prefill_started = Instant::now();
                // Only the text prefill below carries Flash Next draft history.
                // Any other prefill advances the trunk without the cursor.
                let flash_next_mtp_prefill = !has_native_multimodal_prefill
                    && self.flash_next_mtp_prefill_eligible(is_greedy, sampling);
                if !flash_next_mtp_prefill {
                    state.flash_next_mtp.drop_cursor();
                }
                let sampled_token = if let Some(inputs) = gemma4_unified_inputs {
                    if !prefill_completes_prompt {
                        return errored_item_run(
                            item.request_id,
                            "Gemma4 unified multimodal prefill requires the complete prompt in one execution item",
                        );
                    }
                    let full_prompt_tokens = begin_multimodal_prefill(&mut state, item, token_ids);
                    let repetition_history =
                        state.repetition_history(&full_prompt_tokens, sampling);
                    let prefill_forward_started = Instant::now();
                    // WS-M5: capture MTP post-norm when the model has an MTP head.
                    let capture_mtp = self.weights.mtp.is_some();
                    let tok =
                        match chunked_prefill_gemma4_unified_with_mtp_history_and_sampling_buffers(
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
                            capture_mtp,
                        ) {
                            Ok((tok, mtp_hidden, history_tokens)) => {
                                if let Some(hidden) = mtp_hidden {
                                    state.mtp_prefill_hidden = Some(hidden);
                                    state.mtp_prefill_history_tokens = history_tokens;
                                }
                                tok
                            }
                            Err(error) => return errored_item_run(item.request_id, error),
                        };
                    let prefill_forward_wall_us = elapsed_us(prefill_forward_started);
                    self.complete_multimodal_prefill(
                        &mut state,
                        &mut prefix_cache,
                        MultimodalPrefillTail {
                            full_prompt_tokens,
                            first_token: tok,
                            prefill_started,
                            prefill_forward_wall_us,
                            publish_prefix_snapshot: true,
                            model_id,
                            block_size_tokens,
                            media_key: media_key.as_deref(),
                            ctx,
                            sampling,
                            max_output,
                            is_greedy,
                        },
                    );
                    Some(tok)
                } else if let Some(inputs) = qwen3_vl_inputs {
                    if !prefill_completes_prompt {
                        return errored_item_run(
                            item.request_id,
                            "Qwen3-VL multimodal prefill requires the complete prompt in one execution item",
                        );
                    }
                    let full_prompt_tokens = begin_multimodal_prefill(&mut state, item, token_ids);
                    if let Err(error) = inputs.validate_for_prompt_len(full_prompt_tokens.len()) {
                        return errored_item_run(item.request_id, error.to_string());
                    }
                    let repetition_history =
                        state.repetition_history(&full_prompt_tokens, sampling);
                    let prefill_forward_started = Instant::now();
                    let tok = match chunked_prefill_qwen3_vl_with_sampling_buffers(
                        &self.cfg,
                        &self.weights,
                        &full_prompt_tokens,
                        inputs,
                        &mut state.cache,
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
                    self.complete_multimodal_prefill(
                        &mut state,
                        &mut prefix_cache,
                        MultimodalPrefillTail {
                            full_prompt_tokens,
                            first_token: tok,
                            prefill_started,
                            prefill_forward_wall_us,
                            publish_prefix_snapshot: true,
                            model_id,
                            block_size_tokens,
                            media_key: media_key.as_deref(),
                            ctx,
                            sampling,
                            max_output,
                            is_greedy,
                        },
                    );
                    Some(tok)
                } else if let Some(inputs) = minicpm_v46_inputs {
                    if !prefill_completes_prompt {
                        return errored_item_run(
                            item.request_id,
                            "MiniCPM-V 4.6 multimodal prefill requires the complete prompt in one execution item",
                        );
                    }
                    let full_prompt_tokens = begin_multimodal_prefill(&mut state, item, token_ids);
                    if let Err(error) = inputs.validate_for_prompt_len(full_prompt_tokens.len()) {
                        return errored_item_run(item.request_id, error.to_string());
                    }
                    let repetition_history =
                        state.repetition_history(&full_prompt_tokens, sampling);
                    let prefill_forward_started = Instant::now();
                    let tok = match chunked_prefill_minicpm_v46_with_sampling_buffers(
                        &self.cfg,
                        &self.weights,
                        &full_prompt_tokens,
                        inputs,
                        &mut state.cache,
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
                    self.complete_multimodal_prefill(
                        &mut state,
                        &mut prefix_cache,
                        MultimodalPrefillTail {
                            full_prompt_tokens,
                            first_token: tok,
                            prefill_started,
                            prefill_forward_wall_us,
                            publish_prefix_snapshot: true,
                            model_id,
                            block_size_tokens,
                            media_key: media_key.as_deref(),
                            ctx,
                            sampling,
                            max_output,
                            is_greedy,
                        },
                    );
                    Some(tok)
                } else if let Some(inputs) = nemotron_omni_inputs {
                    if !prefill_completes_prompt {
                        return errored_item_run(
                            item.request_id,
                            "Nemotron H Nano Omni multimodal prefill requires the complete prompt in one execution item",
                        );
                    }
                    let full_prompt_tokens = begin_multimodal_prefill(&mut state, item, token_ids);
                    if let Err(error) = inputs.validate_for_prompt_len(full_prompt_tokens.len()) {
                        return errored_item_run(item.request_id, error.to_string());
                    }
                    let repetition_history =
                        state.repetition_history(&full_prompt_tokens, sampling);
                    let prefill_forward_started = Instant::now();
                    let tok = match chunked_prefill_nemotron_omni_with_sampling_buffers(
                        &self.cfg,
                        &self.weights,
                        &full_prompt_tokens,
                        inputs,
                        &mut state.cache,
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
                    self.complete_multimodal_prefill(
                        &mut state,
                        &mut prefix_cache,
                        MultimodalPrefillTail {
                            full_prompt_tokens,
                            first_token: tok,
                            prefill_started,
                            prefill_forward_wall_us,
                            publish_prefix_snapshot: true,
                            model_id,
                            block_size_tokens,
                            media_key: media_key.as_deref(),
                            ctx,
                            sampling,
                            max_output,
                            is_greedy,
                        },
                    );
                    Some(tok)
                } else if let Some(inputs) = unlimited_ocr_inputs {
                    if !prefill_completes_prompt {
                        return errored_item_run(
                            item.request_id,
                            "Unlimited-OCR multimodal prefill requires the complete prompt in one execution item",
                        );
                    }
                    let full_prompt_tokens = begin_multimodal_prefill(&mut state, item, token_ids);
                    if let Err(error) = inputs.validate_for_prompt_tokens(&full_prompt_tokens) {
                        return errored_item_run(item.request_id, error.to_string());
                    }
                    let Some(image) = inputs.images.first() else {
                        return errored_item_run(
                            item.request_id,
                            "Unlimited-OCR request has no image",
                        );
                    };
                    let image_views = match preprocess_document_rgb_u8(
                        &image.rgb_bytes,
                        image.width,
                        image.height,
                        inputs.cropping,
                    ) {
                        Ok(image) => image,
                        Err(error) => {
                            return errored_item_run(item.request_id, error.to_string());
                        }
                    };
                    let repetition_history =
                        state.repetition_history(&full_prompt_tokens, sampling);
                    let prefill_forward_started = Instant::now();
                    let tok = match chunked_prefill_unlimited_ocr_with_sampling_buffers(
                        &self.cfg,
                        &self.weights,
                        &full_prompt_tokens,
                        &image_views,
                        inputs.image_token_id,
                        &mut state.cache,
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
                    self.complete_multimodal_prefill(
                        &mut state,
                        &mut prefix_cache,
                        MultimodalPrefillTail {
                            full_prompt_tokens,
                            first_token: tok,
                            prefill_started,
                            prefill_forward_wall_us,
                            publish_prefix_snapshot: false,
                            model_id,
                            block_size_tokens,
                            media_key: media_key.as_deref(),
                            ctx,
                            sampling,
                            max_output,
                            is_greedy,
                        },
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
                        runner_probe_over_claim(state.cache.seq_len(), item)
                    };
                    let prefill_tokens = if probe_over_claim < prefill_tokens_base.len() {
                        &prefill_tokens_base[probe_over_claim..]
                    } else {
                        &[][..]
                    };
                    let mut effective_prefill_token_count = prefill_tokens.len();
                    let repetition_history = state.repetition_history(prefill_tokens, sampling);
                    if crate::fastpath::should_clear_mlx_cache_before_cold_prefill(
                        state.cache.seq_len(),
                    ) {
                        clear_cache();
                    }
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
                    let (base_prefill_chunk, prefill_chunk_mode) =
                        crate::fastpath::select_prefill_chunk_for_request(
                            state.cache.seq_len(),
                            self.cold_prefill_chunk,
                            self.prefill_chunk,
                        );
                    // Long remaining prompts clamp to the pure-thr chunk (512);
                    // short prompts keep the session base (e.g. 1536 for S0 TTFT).
                    let prefill_chunk_for_request =
                        crate::fastpath::scale_prefill_chunk_for_remaining_in_family(
                            base_prefill_chunk,
                            prefill_tokens.len(),
                            &self.cfg.model_family,
                        );
                    state.decode_telemetry.record_prefill_chunk_selection(
                        prefill_chunk_for_request,
                        matches!(
                            prefill_chunk_mode,
                            crate::fastpath::PrefillChunkMode::WarmExtend
                        ),
                    );
                    // Linear-attention conv/recurrent state cannot be trimmed
                    // at snapshot-store time, so an unaligned prompt can only
                    // get a prefix snapshot if the cache is captured
                    // mid-prefill exactly at the largest block-aligned
                    // boundary. Split the prefill there: run the aligned head
                    // through the no-logits/no-sampling continuation API,
                    // stash a lazy cache clone for
                    // `store_prompt_prefix_snapshots`, and let the normal path
                    // below prefill the remainder. The clone is handle-only;
                    // its one real cost is a single non-donated full-attention
                    // K/V append on the following chunk.
                    let prefill_tokens = if self.cfg.linear_attention.is_some()
                        && self.prefix_cache.lock().enabled()
                        && let Some(head) = Self::linear_boundary_capture_head_len(
                            block_size_tokens as usize,
                            state.cache.seq_len(),
                            prefill_tokens.len(),
                        ) {
                        let head_tokens = &prefill_tokens[..head];
                        if flash_next_mtp_prefill {
                            // Same n-1 plus singleton schedule as the
                            // cache-only head, with draft history absorbed.
                            let _ = self.run_flash_next_mtp_prefill(
                                &mut state,
                                head_tokens,
                                prefill_chunk_for_request,
                                false,
                            );
                        } else {
                            chunked_prefill_cache_only(
                                &self.cfg,
                                &self.weights,
                                head_tokens,
                                &mut state.cache,
                                prefill_chunk_for_request,
                                CacheOnlyPrefillLayout::PreserveFinalTokenStep,
                                crate::generate::CacheOnlyBarrier::Blocking,
                            );
                        }
                        state.prefill_boundary_snapshot =
                            Some((state.cache.seq_len(), state.cache.clone()));
                        &prefill_tokens[head..]
                    } else {
                        prefill_tokens
                    };
                    let prefill_output_token = if prefill_tokens.is_empty() {
                        if !prefill_completes_prompt {
                            None
                        } else if let Some(tok) = state.cached_prefill_output_token.take() {
                            Some(tok)
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
                            // After reset, entry-point matrix requires the cold trail.
                            let (base_recompute_chunk, recompute_mode) =
                                crate::fastpath::select_prefill_chunk_for_request(
                                    state.cache.seq_len(),
                                    self.cold_prefill_chunk,
                                    self.prefill_chunk,
                                );
                            let recompute_chunk =
                                crate::fastpath::scale_prefill_chunk_for_remaining_in_family(
                                    base_recompute_chunk,
                                    token_ids.len(),
                                    &self.cfg.model_family,
                                );
                            state.decode_telemetry.record_prefill_chunk_selection(
                                recompute_chunk,
                                matches!(
                                    recompute_mode,
                                    crate::fastpath::PrefillChunkMode::WarmExtend
                                ),
                            );
                            let recompute_history = state.repetition_history(token_ids, sampling);
                            let tok = if flash_next_mtp_prefill {
                                // The cache was reset above, so this cold
                                // recompute starts a fresh cursor.
                                let Some(tok) = self.run_flash_next_mtp_prefill(
                                    &mut state,
                                    token_ids,
                                    recompute_chunk,
                                    true,
                                ) else {
                                    return errored_item_run(
                                        item.request_id,
                                        "Flash Next MTP recompute prefill produced no token",
                                    );
                                };
                                tok
                            } else if should_capture_qwen_mtp_prefill_history(
                                self.mtp_requested,
                                self.weights.mtp.is_some(),
                            ) {
                                let (tok, hidden, history_tokens) =
                                    chunked_prefill_with_mtp_history_and_sampling_buffers(
                                        &self.cfg,
                                        &self.weights,
                                        token_ids,
                                        &mut state.cache,
                                        recompute_chunk,
                                        MlxSamplingRequest::new(sampling, &recompute_history),
                                        &mut state.rng,
                                        &mut state.sampling_probs_buf,
                                        &mut state.sampling_logits_buf,
                                        &mut state.sampling_candidates_buf,
                                    );
                                state.mtp_prefill_hidden = Some(hidden);
                                state.mtp_prefill_history_tokens = history_tokens;
                                tok
                            } else if self.weights.deepseek_v4_nextn.is_some() {
                                let (tok, packed, history_tokens) =
                                    chunked_prefill_with_deepseek_v4_mtp_history_and_sampling_buffers(
                                        &self.cfg,
                                        &self.weights,
                                        token_ids,
                                        &mut state.cache,
                                        recompute_chunk,
                                        MlxSamplingRequest::new(sampling, &recompute_history),
                                        &mut state.rng,
                                        &mut state.sampling_probs_buf,
                                        &mut state.sampling_logits_buf,
                                        &mut state.sampling_candidates_buf,
                                    );
                                state.mtp_prefill_hidden = Some(packed);
                                state.mtp_prefill_history_tokens = history_tokens;
                                tok
                            } else {
                                chunked_prefill_with_sampling_buffers(
                                    &self.cfg,
                                    &self.weights,
                                    token_ids,
                                    &mut state.cache,
                                    recompute_chunk,
                                    MlxSamplingRequest::new(sampling, &recompute_history),
                                    &mut state.rng,
                                    &mut state.sampling_probs_buf,
                                    &mut state.sampling_logits_buf,
                                    &mut state.sampling_candidates_buf,
                                )
                            };
                            Some(tok)
                        }
                    } else if !prefill_completes_prompt && flash_next_mtp_prefill {
                        // Intermediate scheduler quantum: the cursor persists
                        // in request state until the final quantum.
                        let _ = self.run_flash_next_mtp_prefill(
                            &mut state,
                            prefill_tokens,
                            prefill_chunk_for_request,
                            false,
                        );
                        state
                            .decode_telemetry
                            .record_prefill_cache_only_continuation();
                        None
                    } else if flash_next_mtp_prefill {
                        let Some(tok) = self.run_flash_next_mtp_prefill(
                            &mut state,
                            prefill_tokens,
                            prefill_chunk_for_request,
                            true,
                        ) else {
                            return errored_item_run(
                                item.request_id,
                                "Flash Next MTP prefill produced no token",
                            );
                        };
                        Some(tok)
                    } else if !prefill_completes_prompt {
                        chunked_prefill_cache_only(
                            &self.cfg,
                            &self.weights,
                            prefill_tokens,
                            &mut state.cache,
                            prefill_chunk_for_request,
                            if is_greedy {
                                CacheOnlyPrefillLayout::PreserveFinalTokenStep
                            } else {
                                CacheOnlyPrefillLayout::Batched
                            },
                            // This quantum does not complete the prompt: more
                            // prefill items follow, so submit instead of
                            // blocking on a growing full-cache barrier.
                            crate::generate::CacheOnlyBarrier::AsyncSubmit,
                        );
                        state
                            .decode_telemetry
                            .record_prefill_cache_only_continuation();
                        None
                    } else if should_capture_qwen_mtp_prefill_history(
                        self.mtp_requested,
                        self.weights.mtp.is_some(),
                    ) {
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
                        Some(tok)
                    } else if self.weights.deepseek_v4_nextn.is_some() {
                        let (tok, packed, history_tokens) =
                            chunked_prefill_with_deepseek_v4_mtp_history_and_sampling_buffers(
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
                        state.mtp_prefill_hidden = Some(packed);
                        state.mtp_prefill_history_tokens = history_tokens;
                        Some(tok)
                    } else if let Some(staged) = staged_batched_prefill.take().filter(|staged| {
                        state.cache.seq_len() == 0
                            && staged.prompt_tokens.as_slice() == prefill_tokens
                    }) {
                        // Consume the padded-batched-prefill result computed
                        // for this exact item: install the row's per-layer
                        // K/V (trimmed to the real prompt length) and sample
                        // from the staged last-token logits with this
                        // request's own RNG — the same single draw the
                        // sequential path below would have made.
                        for (layer, (k, v)) in staged.layer_kv.into_iter().enumerate() {
                            state
                                .cache
                                .set_layer_kv_logical(layer, k, v, staged.seq_len);
                        }
                        Some(sample_token_from_prefill_logits(
                            &staged.logits,
                            MlxSamplingRequest::new(sampling, &repetition_history),
                            &mut state.rng,
                            &mut state.sampling_probs_buf,
                            &mut state.sampling_logits_buf,
                            &mut state.sampling_candidates_buf,
                        ))
                    } else {
                        Some(chunked_prefill_with_sampling_buffers(
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
                        ))
                    };
                    let prefill_forward_wall_us = elapsed_us(prefill_forward_started);
                    let prefill_token_count = effective_prefill_token_count;
                    if let Some(tokens) = full_recompute_tokens {
                        state.prompt_prefix_tokens = tokens;
                    } else {
                        extend_prompt_prefix_tokens(&mut state, item, token_ids);
                    }
                    let prefix_cache_started = Instant::now();
                    let linear_boundary_snapshot = state.prefill_boundary_snapshot.take();
                    // Admission's cold_prefill_us must reflect the cost of
                    // cold-prefilling the *whole* persisted prefix, not just
                    // this scheduler step's own increment: a prompt longer
                    // than one prefill chunk runs this block across several
                    // steps, and `elapsed_us(prefill_started)` alone only
                    // covers the latest chunk. `decode_telemetry.prefill_wall_us`
                    // already accumulates prior steps' prefill time for this
                    // request (its own `record_prefill` call for *this* step
                    // runs after this point, so it must be added in here).
                    // This still understates the true cost when part of the
                    // prefix was hydrated from an L1/L2 restore rather than
                    // cold-computed by this request — that portion's original
                    // compute cost isn't tracked forward through the restore
                    // path, so warm-extend requests conservatively bias
                    // toward under-admission rather than over-admission.
                    let cold_prefill_us = state
                        .decode_telemetry
                        .prefill_wall_us
                        .saturating_add(elapsed_us(prefill_started));
                    prefix_cache.merge_from(
                        self.store_prompt_prefix_snapshots(
                            model_id,
                            block_size_tokens,
                            &state,
                            PromptPrefixSnapshotStoreOptions {
                                linear_boundary_snapshot: linear_boundary_snapshot.as_ref(),
                                prefill_completes_prompt,
                                greedy_prefill_output_token: prefill_output_token
                                    .filter(|_| prefill_output_token_cacheable(ctx, sampling)),
                                cold_prefill_us: u64::from(cold_prefill_us),
                                media_key: media_key.as_deref(),
                            },
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
                    if let Some(tok) = prefill_output_token {
                        debug_assert!(prefill_completes_prompt);
                        let generation_state_started = Instant::now();
                        self.initialize_generation_state(
                            &mut state,
                            max_output,
                            Some(tok),
                            is_greedy,
                            sampling,
                            ctx.map(|c| (c.max_think_tokens, c.answer_reserve_tokens))
                                .unwrap_or((None, None)),
                        );
                        prefill_generation_state_wall_us = elapsed_us(generation_state_started);
                    }

                    // Each non-final chunk in chunked_prefill calls async_eval; only the
                    // last chunk calls a blocking eval.  Compute counts from prompt length.
                    let drain_count =
                        prefill_drain_async_eval_count(prefill_token_count, self.prefill_chunk);
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
                        prefill_output_token
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
                        self.initialize_generation_state(
                            &mut state,
                            max_output,
                            Some(tok),
                            is_greedy,
                            sampling,
                            ctx.map(|c| (c.max_think_tokens, c.answer_reserve_tokens))
                                .unwrap_or((None, None)),
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
        // Fixed-token / `ignore_eos` requests (README AX-direct benches) must
        // run through `max_output_tokens`. Gemma 4 loop detection is default-on
        // for production chat quality, but on random-prompt greedy decode it
        // fires within a handful of tokens and collapses the measured decode
        // window from 128 → ~4 tokens — the 2026-07-26 Gemma@2048 "slow"
        // readout compared a micro-burst against the 7/14 full-length run.
        let (sampled_tokens, stop_reason) = apply_loop_detection_stop(
            sampled_tokens,
            stop_reason,
            &state.generated_tokens,
            loop_detection_for_request(
                ctx.map(|c| c.ignore_eos).unwrap_or(false),
                &self.cfg.model_family,
            ),
        );
        // Thinking-budget controller: force the think-close token when the
        // answer reserve / think cap is exhausted or the loop detector fired
        // inside an open think block (ds4-style soft+hard close). A budget
        // edge carries no stop reason: generation continues so the remaining
        // output budget funds the answer (ds4 hard-limit semantics). Budget
        // edges are routed onto the single-token path by `decode_one` (the
        // swap is KV-exact there); this post-decode hook still handles the
        // loop-detected close and re-applies the budget close on that route.
        let (sampled_tokens, stop_reason) = if let Some(ctx) = ctx
            && let Some((forced_tokens, forced_stop)) = think_budget_close_override(
                stop_reason,
                !sampled_tokens.is_empty(),
                state.ngram_in_think,
                self.cfg.think_end_token_id,
                ctx.max_output_tokens.saturating_sub(ctx.generated_len),
                state.think_emitted_tokens,
                ctx.max_think_tokens,
                ctx.answer_reserve_tokens,
            ) {
            state.pending_direct = None;
            // Queued n-gram bonus tokens were verified against a think
            // trace the forced close token just interrupted; they point at
            // the wrong continuation and must not be served.
            state.bonus_queue.clear();
            // The forced close token is what `generated_tokens` will record,
            // so it is also what the next step must feed.
            state.next_model_last_token = forced_tokens.last().copied();
            (forced_tokens, forced_stop)
        } else {
            (sampled_tokens, stop_reason)
        };
        // Keep tokens for LoopDetected (and continuing decode).
        if stop_reason.is_none() || matches!(stop_reason, Some(StopReason::LoopDetected)) {
            let was_in_think = state.ngram_in_think;
            for &sampled_token in &sampled_tokens {
                state.generated_tokens.push(sampled_token);
                update_ngram_think_state(&self.cfg, &mut state.ngram_in_think, sampled_token);
            }
            if was_in_think {
                state.think_emitted_tokens = state
                    .think_emitted_tokens
                    .saturating_add(sampled_tokens.len() as u32);
            }
        }

        // Re-insert state only if the request continues — lock held briefly.
        let ngram_acceleration = state.ngram_acceleration;
        let mtp_telemetry = state.mtp_telemetry;
        let flash_next_mtp_telemetry = state.flash_next_mtp.telemetry;
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
        let kv_usage = state
            .cache
            .usage_snapshot_with_layer_windows(&self.kv_layer_windows);
        // Capture schedule feedback only after a monoblock generate.
        // Drain-only steps must not overwrite denoise_steps with 0.
        let diffusion_schedule = state.pending_diffusion_schedule.take();

        if state.cache.hard_cap_exhausted() {
            // Operator set AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS as a hard
            // memory cap. append_paged_fa already demoted the offending
            // layer to contiguous storage (correct data — proven token-exact
            // by fa_paged_pool_exhaustion_demotion_matches_contiguous_oracle)
            // so the forward above completed safely, but the cap's whole
            // point is to bound memory rather than silently fall through to
            // unbounded contiguous growth. Fail just this request instead of
            // returning a token computed past the cap; do not re-insert
            // state (mirrors every other terminal path below).
            clear_cache();
            return errored_item_run(
                item.request_id,
                "FA KV block pool exhausted under explicit AX_MLX_FA_KV_BLOCK_POOL_MAX_BLOCKS cap",
            );
        }

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
        let output_tokens: Vec<u32> = sampled_tokens.collect();
        // Mid-denoise schedule-only progress executes zero visible tokens.
        let tokens_executed = if output_token.is_none()
            && output_tokens.is_empty()
            && matches!(
                diffusion_schedule,
                Some(DiffusionScheduleUpdate {
                    commit_ready: false,
                    block_committed: false,
                    ..
                })
            ) {
            0
        } else {
            item.scheduled_token_count
        };

        MlxItemRun {
            update: RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed,
                output_token,
                output_tokens,
                stop_reason,
                error: None,
                diffusion_schedule,
            },
            ngram_acceleration,
            mtp_telemetry,
            flash_next_mtp_telemetry,
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

    fn native_fa_prefix_sharing_enabled(&self) -> bool {
        self.shared_fa_block_pool.is_some()
            && !self.has_mtp()
            && self.cfg.mla_attention.is_none()
            && self.cfg.linear_attention.is_none()
            && self.cfg.diffusion.is_none()
            && self.kv_layer_windows.iter().all(Option::is_none)
            && self.weights.gemma4_unified_vision.is_none()
            && self.weights.gemma4_unified_audio.is_none()
            && self.weights.unlimited_ocr_vision.is_none()
            && self.weights.minicpm_v46_vision.is_none()
            && self.weights.nemotron_omni.is_none()
    }

    fn reclaim_native_prefix_capacity(
        &self,
        cache: &MlxKVCache,
        new_tokens: usize,
        telemetry: &mut MlxPrefixCacheTelemetry,
    ) {
        if !self.native_fa_prefix_sharing_enabled() {
            return;
        }
        let Some(pool) = self.shared_fa_block_pool.as_ref() else {
            return;
        };
        if !cache.uses_fa_block_pool(pool) {
            return;
        }
        let evictions = reclaim_native_prefix_entries(
            &self.native_prefix_cache,
            pool,
            || cache.additional_fa_blocks_for_append(new_tokens),
            |key, snapshot| {
                demote_native_prefix_snapshot(&self.prefix_cache, key, snapshot, telemetry);
            },
        );
        telemetry.native_evictions = telemetry.native_evictions.saturating_add(evictions);
    }

    fn reclaim_native_prefix_blocks(
        &self,
        required: u32,
        telemetry: &mut MlxPrefixCacheTelemetry,
    ) -> u32 {
        let Some(pool) = self.shared_fa_block_pool.as_ref() else {
            return 0;
        };
        reclaim_native_prefix_entries(
            &self.native_prefix_cache,
            pool,
            || Some(required),
            |key, snapshot| {
                demote_native_prefix_snapshot(&self.prefix_cache, key, snapshot, telemetry);
            },
        )
    }

    /// Restore portable dense snapshots back under the runner-wide physical
    /// budget. The wire representation remains dense, but an opted-in pure-FA
    /// runner must not escape its pool merely because native L1 missed.
    fn prepare_portable_prefix_restore(
        &self,
        restored: MlxKVCache,
        telemetry: &mut MlxPrefixCacheTelemetry,
    ) -> Result<MlxKVCache, FaBlockPoolError> {
        if !self.native_fa_prefix_sharing_enabled() {
            return Ok(restored);
        }
        let Some(pool) = self.shared_fa_block_pool.as_ref() else {
            return Ok(restored);
        };
        let required = restored.fa_blocks_required_for_repage(pool)?;
        let evictions = self.reclaim_native_prefix_blocks(required, telemetry);
        telemetry.native_evictions = telemetry.native_evictions.saturating_add(evictions);
        restored.clone_repage_into_shared_fa_pool(pool.clone())
    }

    fn prefix_cache_route_policy(&self) -> String {
        let decode_policy = if self.disable_ngram_acceleration {
            "direct"
        } else {
            "ngram"
        };
        decode_policy.to_string()
    }

    fn prefix_cache_layer_layout(&self) -> String {
        // When a content-derived artifact fingerprint is available it is
        // already in the durable L2 key (schema v3). Keep the layout class
        // path-free so the same weights at a new path still share L1/L2.
        // Without a fingerprint, fold in `model_artifacts_root` so a
        // hot-swap that reuses `model_id` cannot hit wrong-checkpoint KV.
        if let Some(fingerprint) = self.artifact_fingerprint.as_deref() {
            // Path-free, but checkpoint-bound: the in-memory L1 tier is shared
            // across sessions, so a hot swap that keeps `model_id` and the
            // layer count must not hit KV captured under different weights.
            format!(
                "layers={};ordered-prefix-v2;art={fingerprint}",
                self.cfg.layer_count
            )
        } else {
            format!(
                "layers={};ordered-prefix-v2;root={}",
                self.cfg.layer_count, self.model_artifacts_root
            )
        }
    }

    /// Canonical disk key (schema v3) for `key` plus the exact tokens, or
    /// `None` when the loaded model has no content-derived artifact
    /// identity — the durable L2 tier is then ineligible, because a
    /// path-only identity could restore stale KV after an in-place
    /// checkpoint replacement (spec §6).
    fn disk_prefix_key_bytes(&self, key: &MlxPrefixCacheKey, tokens: &[u32]) -> Option<Vec<u8>> {
        let fingerprint = self.artifact_fingerprint.as_deref()?;
        Some(crate::disk_prefix_cache::canonical_key_bytes(
            &crate::disk_prefix_cache::DiskPrefixKeyFields {
                model_id: &key.model_id,
                artifact_fingerprint_sha256: fingerprint,
                route_policy: &key.route_policy,
                layer_layout: &key.layer_layout,
                kv_payload_version: MlxKVCache::serialize_version(),
                block_size_tokens: key.block_size_tokens,
                token_count: key.token_count,
                tokens,
            },
        ))
    }

    /// Model fingerprint used for media digests (artifact identity when known).
    fn media_model_fingerprint(&self) -> String {
        self.artifact_fingerprint
            .clone()
            .unwrap_or_else(|| self.model_artifacts_root.clone())
    }

    /// `Some(key)` when the request is text-only (empty key) or a
    /// recognized multimodal family produced a content digest; `None`
    /// when media is present but no digest covers it. `None` must fail
    /// closed at every prefix-cache touch point: a text-only key for a
    /// media-bearing prompt would let two prompts with identical tokens
    /// but different media share KV.
    fn media_key_from_inputs(
        &self,
        multimodal_inputs: Option<&RequestMultimodalInputs>,
    ) -> Option<String> {
        media_key_for_fingerprint(multimodal_inputs, &self.media_model_fingerprint())
    }

    fn prefix_cache_key_with_media(
        &self,
        model_id: &str,
        block_size_tokens: u32,
        tokens: &[u32],
        media_key: &str,
    ) -> MlxPrefixCacheKey {
        // Fold media digests into layer_layout so disk L2 v3 keys also
        // domain-separate without a schema bump (WS-M3).
        let layer_layout = format_prefix_layer_layout(&self.prefix_cache_layer_layout(), media_key);
        MlxPrefixCacheKey {
            model_id: model_id.to_string(),
            route_policy: self.prefix_cache_route_policy(),
            layer_layout,
            block_size_tokens,
            token_count: saturating_u32(tokens.len()),
            token_hash: hash_prefix_tokens(tokens),
            media_key: media_key.to_string(),
        }
    }

    /// Relative length of the aligned "head" of a prefill item that must be
    /// prefilled separately so the cache can be captured at the largest
    /// block-aligned boundary — the only sound snapshot point for
    /// linear-attention models, whose conv/recurrent state cannot be trimmed
    /// back at store time. `None` when the prompt end is already aligned
    /// (the exact-alignment store path handles it), when the boundary falls
    /// inside already-cached context, or when there is nothing to prefill.
    /// Return the aligned cache-only head used by linear prefix capture.
    /// Diagnostics use this policy to reproduce the runner's prefill layout.
    pub fn linear_boundary_capture_head_len(
        block_size: usize,
        prior_seq_len: usize,
        item_len: usize,
    ) -> Option<usize> {
        if block_size == 0 || item_len == 0 {
            return None;
        }
        let total = prior_seq_len + item_len;
        if total.is_multiple_of(block_size) {
            return None;
        }
        let boundary = total - (total % block_size);
        (boundary > prior_seq_len).then(|| boundary - prior_seq_len)
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
        media_key: &str,
    ) -> Option<Vec<u32>> {
        let requested_key =
            self.prefix_cache_key_with_media(model_id, block_size_tokens, input, media_key);
        let native_len = self
            .native_fa_prefix_sharing_enabled()
            .then(|| {
                self.native_prefix_cache
                    .lock()
                    .longest_prefix_len(&requested_key, input)
            })
            .flatten();
        // Re-acquire the L1 lock per candidate rather than holding it for
        // the whole walk: a long prompt with no L1 hit at any block-aligned
        // length can probe many candidates, each with an `fs::stat` disk
        // check — holding the mutex across all of them would serialize
        // every other concurrent request's L1 access behind that I/O for
        // no correctness reason.
        let portable =
            Self::longest_block_aligned_prefix_by_probe(block_size_tokens, input, |prefix| {
                let key = self.prefix_cache_key_with_media(
                    model_id,
                    block_size_tokens,
                    prefix,
                    media_key,
                );
                if self.prefix_cache.lock().contains_exact_tokens(&key, prefix) {
                    return true;
                }
                if let Some(disk) = self.disk_prefix_cache.as_ref()
                    && let Some(key_bytes) = self.disk_prefix_key_bytes(&key, prefix)
                    && disk.contains(&key_bytes)
                {
                    return true;
                }
                false
            });
        let portable_len = portable.as_ref().map(Vec::len);
        match (native_len, portable_len) {
            (Some(native), Some(portable_len)) if native <= portable_len => portable,
            (Some(native), _) => Some(input[..native].to_vec()),
            (None, _) => portable,
        }
    }

    /// Discard a probed prefix that is shorter than a *non-empty* scheduler
    /// claim (`item.reused_prefix_token_slice`).
    ///
    /// The scheduler sizes `token_ids` (and every downstream RoPE position
    /// for this item) assuming exactly `reused_prefix_token_slice.len()`
    /// tokens precede them. Restoring a probed prefix shorter than that claim
    /// would silently drop the shortfall tokens from the model's context and
    /// compute `token_ids` at positions shifted earlier by the gap — a real,
    /// silent correctness bug, reachable whenever the scheduler's cumulative
    /// multi-turn block tracking claims more reuse than the runner's own
    /// snapshot map ever stored (see the divergence documented on
    /// `restore_reused_prefix_state`).
    ///
    /// Discarding here routes the caller through the exact-match lookup on
    /// the *full* claimed slice instead, which correctly misses (the probe
    /// already proved that length isn't resident) and falls to the miss
    /// path's `full_prefill_recompute_tokens_for_warmup_fallback`, which
    /// performs a full, correctly-positioned recompute of the entire prefix.
    ///
    /// When `reused_prefix_token_slice` is empty, `probe_upper_bound` fell
    /// back to `input_token_slice`, so any probed length is a pure bonus
    /// beyond the scheduler's expectation of zero reuse — never discarded
    /// here; the existing `probe_over_claim` slicing in the caller already
    /// handles trimming that case safely.
    fn discard_probe_shorter_than_scheduler_claim(
        probed_tokens: Vec<u32>,
        item: &ax_engine_core::ExecutionItem,
    ) -> Vec<u32> {
        if !item.reused_prefix_token_slice.is_empty()
            && probed_tokens.len() < item.reused_prefix_token_slice.len()
        {
            Vec::new()
        } else {
            probed_tokens
        }
    }

    /// Per-layer mask of positions that must carry complete linear
    /// (conv + recurrent) state in a restored prefix snapshot, or `None`
    /// for families whose layer kinds are classified from weights rather
    /// than the `full_attention_interval` pattern (nemotron-style): the
    /// pattern cannot be trusted for them, and their MLP/MoE layers
    /// legitimately serialize as `EMPTY`.
    fn restored_linear_layer_requirements(&self) -> Option<Vec<bool>> {
        self.cfg.linear_attention.as_ref()?;
        if self.cfg.model_family.starts_with("nemotron") {
            return None;
        }
        Some(
            (0..self.cfg.layer_count)
                .map(|idx| self.cfg.is_linear_attention_layer(idx))
                .collect(),
        )
    }

    /// Fail-closed structural check for a snapshot about to be adopted
    /// from the portable L1 or durable L2 prefix cache. Key equality
    /// proves token identity, not payload completeness; this rejects
    /// snapshots whose layer structure cannot represent the claimed
    /// prefix (all-`EMPTY` layers, truncated layer count, missing
    /// linear conv/recurrent state).
    fn verify_restored_prefix_snapshot(
        &self,
        cache: &mut MlxKVCache,
        expected_tokens: usize,
    ) -> Result<(), MlxKVCacheSerializeError> {
        let required = self.restored_linear_layer_requirements();
        cache.verify_restored_snapshot(
            self.cfg.layer_count,
            expected_tokens,
            required.as_deref(),
        )?;
        match (&self.weights.qwen4_exp, &mut cache.qwen4_exp) {
            (Some(weights), Some(state)) => state
                .rebind_for_model(weights, self.cfg.compile_cache_identity)
                .map_err(MlxKVCacheSerializeError::FlashNextState),
            (None, None) => Ok(()),
            _ => Err(MlxKVCacheSerializeError::FlashNextState(
                "snapshot family does not match the model".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_reused_prefix_state(
        &self,
        state: &mut RequestState,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
        model_id: &str,
        block_size_tokens: u32,
        sampling: MlxSamplingParams,
        needs_flash_next_cursor: bool,
        media_identity: Option<&str>,
    ) -> MlxPrefixCacheTelemetry {
        let mut telemetry = MlxPrefixCacheTelemetry::default();
        // Media present but no digest: never touch the prefix-cache
        // keyspace (a text-only key would alias prompts with different
        // media). Runs before the probe so an unidentifiable request
        // cannot match snapshots either. Warm semantics mirror the other
        // blocked_* branches below.
        let Some(media_key) = media_identity else {
            telemetry.record_blocked_media_identity();
            let reused_tokens: &[u32] = &item.reused_prefix_token_slice;
            Self::pfx_dbg(
                "restore",
                &format!(
                    "reused_tokens={} cache_seq={}",
                    reused_tokens.len(),
                    state.cache.seq_len()
                ),
            );
            if !reused_tokens.is_empty() && state.cache.seq_len() == 0 {
                let capture_prefill_output = item.mode == ExecutionMode::Decode
                    && ctx.is_some_and(|ctx| ctx.generated_len == 0);
                if item.mode != ExecutionMode::Prefill {
                    self.warm_reused_prefix_without_cache(
                        state,
                        item.request_id,
                        reused_tokens,
                        sampling,
                        capture_prefill_output,
                        needs_flash_next_cursor,
                    );
                }
                telemetry.warmup_tokens = telemetry
                    .warmup_tokens
                    .saturating_add(saturating_u32(reused_tokens.len()));
            }
            return telemetry;
        };
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
        let probed_tokens: Vec<u32> = if state.cache.seq_len() == 0
            && !probe_upper_bound.is_empty()
            && matches!(item.mode, ExecutionMode::Prefill)
        {
            self.probe_runner_snapshot_for_prefix(
                model_id,
                block_size_tokens,
                probe_upper_bound,
                media_key,
            )
            .unwrap_or_default()
        } else {
            Vec::new()
        };
        let probed_tokens = Self::discard_probe_shorter_than_scheduler_claim(probed_tokens, item);
        let reused_tokens: &[u32] = if !probed_tokens.is_empty() {
            &probed_tokens
        } else {
            &item.reused_prefix_token_slice
        };
        Self::pfx_dbg(
            "restore-entry",
            &format!(
                "core_claim={} probed={} cache_seq={}",
                item.reused_prefix_token_slice.len(),
                probed_tokens.len(),
                state.cache.seq_len()
            ),
        );
        if state.cache.seq_len() != 0 {
            return telemetry;
        }
        // Shape-exact warm extension: a multi-token extension chunk whose
        // (offset, len) is off the cold chunk grid dispatches different
        // SDPA kernels than the cold baseline at the same absolute
        // positions and can drift fp-wise (default-corpus warm_extend
        // measured p2 idx=14 / p3 idx=18 / p5 idx=0 on Gemma 4 12B —
        // same class as the MLA R2 note below). Snap the restore point
        // down to the cold prefill-chunk grid so every extension chunk
        // replays the cold trail shape-for-shape; a claim that sits
        // entirely inside the first cold chunk skips reuse — recomputing
        // under one chunk is cheaper than any drift risk. Restores that
        // need no multi-token extension (warm_repeat tails of <=1 token)
        // keep their full length.
        let Some(restore_len) =
            grid_aligned_prefix_restore_len(item, reused_tokens.len(), self.prefill_chunk.max(1))
        else {
            // The scheduler already removed this prefix from the input slice.
            // Recompute prefix+suffix together instead of leaving a context gap.
            telemetry.warmup_tokens = saturating_u32(item.reused_prefix_token_slice.len());
            return telemetry;
        };
        let reused_tokens = &reused_tokens[..restore_len];
        if reused_tokens.is_empty() {
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
                    needs_flash_next_cursor,
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
                    needs_flash_next_cursor,
                );
            }
            telemetry.warmup_tokens = telemetry
                .warmup_tokens
                .saturating_add(saturating_u32(reused_tokens.len()));
            return telemetry;
        }

        let key =
            self.prefix_cache_key_with_media(model_id, block_size_tokens, reused_tokens, media_key);
        let native_hit = if self.native_fa_prefix_sharing_enabled() {
            self.native_prefix_cache
                .lock()
                .get_longest_prefix(&key, reused_tokens)
        } else {
            None
        };
        let hit = {
            let mut cache = self.prefix_cache.lock();
            let hit = cache.get(&key, reused_tokens);
            telemetry.record_stats(cache.stats());
            hit
        };
        Self::pfx_dbg(
            "restore-get",
            &format!(
                "reused={} l1_hit={} native_hit={}",
                reused_tokens.len(),
                hit.is_some(),
                native_hit.is_some()
            ),
        );

        // Historical context: MLA + Prefill used to refuse a snapshot restore
        // because the post-restore chunked_prefill drifted fp-wise from a
        // cold full prefill (p2_medium_explain idx=13 divergence on
        // GLM-4.7-Flash). Evidence points to shape-dependent SDPA kernel
        // selection in MLX: a large cold chunk and smaller warm-extend
        // chunks can dispatch different kernels at the same absolute
        // positions. The fix is upstream of this branch:
        //   - warm-extend chunk = `MLA_DEFAULT_PREFILL_CHUNK` (16)
        //   - cold production defaults to the same trail (R2; see
        //     `resolve_mla_cold_prefill_chunk`) so store producers match
        //     the cold baseline under warm_extend.
        // The kill-switch env `AX_DISABLE_MLA_PREFIX_RESTORE=1` re-engages
        // the historical gate if a future workload exposes a residual
        // drift vector.
        let mla_extend_unsafe = self.cfg.mla_attention.is_some()
            && item.mode == ExecutionMode::Prefill
            && crate::fastpath::mla_prefix_restore_disabled();

        if let Some((snapshot, matched_len)) = native_hit
            && matched_len == reused_tokens.len()
            && let Some(pool) = self.shared_fa_block_pool.as_ref()
            && snapshot.cache.uses_fa_block_pool(pool)
            && snapshot.cache.is_native_fa_shareable()
        {
            let mut adopted = snapshot.cache.clone();
            if adopted.trim_to(matched_len) {
                state.cache = adopted;
                state.prompt_prefix_tokens = reused_tokens.to_vec();
                state.cached_prefill_output_token = (matched_len == snapshot.token_count)
                    .then_some(snapshot.greedy_prefill_output_token)
                    .flatten()
                    .filter(|_| prefill_output_token_cacheable(ctx, sampling));
                telemetry.hits = telemetry.hits.saturating_add(1);
                telemetry.native_hits = telemetry.native_hits.saturating_add(1);
                telemetry.record_restore_source(RESTORE_SOURCE_MEMORY_L1);
                telemetry.reused_tokens = telemetry
                    .reused_tokens
                    .saturating_add(saturating_u32(matched_len));
                return telemetry;
            }
            // Symmetric with the store path's blocked_trim_failure: the
            // native hit was unusable; the portable tiers below may
            // still serve the prefix.
            telemetry.record_blocked_trim_failure();
        }

        if let Some(snapshot) = hit {
            if mla_extend_unsafe {
                telemetry.record_blocked_unsupported_layout();
                telemetry.warmup_tokens = telemetry
                    .warmup_tokens
                    .saturating_add(saturating_u32(reused_tokens.len()));
                return telemetry;
            }
            match snapshot.rehydrate_cache() {
                Ok(restored_cache) if restored_cache.has_rotated_sliding_layers() => {
                    telemetry.record_blocked_unsupported_layout();
                    tracing::warn!(
                        target: "ax_engine_mlx::prefix_cache",
                        "L1 prefix-cache payload contains slot-ordered rotating KV; \
                         treating as miss and recomputing ordered prefix state",
                    );
                    // Fall through to L2/miss. The ordered-prefix-v2 key
                    // prevents newly written entries from reaching here;
                    // this guard also contains in-process legacy entries.
                }
                Ok(mut restored_cache) => {
                    if let Err(e) = self
                        .verify_restored_prefix_snapshot(&mut restored_cache, reused_tokens.len())
                    {
                        telemetry.record_blocked_snapshot_incomplete();
                        tracing::warn!(
                            target: "ax_engine_mlx::prefix_cache",
                            error = %e,
                            "L1 prefix-cache payload is structurally incomplete; treating as miss",
                        );
                    } else {
                        let prepared =
                            self.prepare_portable_prefix_restore(restored_cache, &mut telemetry);
                        match prepared {
                            Ok(restored_cache) => {
                                // Flash Next session with a stashed draft-cursor
                                // sidecar: decode it before adopting the trunk.
                                // Eligible MTP requests replay a prefix whose
                                // draft history is absent or corrupt. The trunk for
                                // alignment is this snapshot's own qwen4_exp
                                // state, taken before the cache moves into the
                                // request.
                                let pending_restored_cursor = if self.flash_next_mtp_session()
                                    && let Some(head) = self.weights.qwen4_exp_mtp.as_deref()
                                    && let Some(bytes) = snapshot.mtp_cursor_payload.as_deref()
                                    && let Some(trunk) = restored_cache.qwen4_exp.as_ref()
                                {
                                    match crate::model::qwen4_exp_mtp::Qwen4ExpDraftCursor::from_prefix_snapshot(
                                        head,
                                        self.cfg.compile_cache_identity,
                                        trunk,
                                        bytes,
                                    ) {
                                        Ok(cursor) => Some(cursor),
                                        Err(error) => {
                                            tracing::warn!(
                                                target: "ax_engine_mlx::prefix_cache",
                                                error = %error,
                                                "Flash Next draft-cursor payload failed to decode",
                                            );
                                            None
                                        }
                                    }
                                } else {
                                    None
                                };
                                if needs_flash_next_cursor && pending_restored_cursor.is_none() {
                                    telemetry.record_blocked_snapshot_incomplete();
                                } else {
                                    state.flash_next_mtp.pending_restored_cursor =
                                        pending_restored_cursor;
                                    state.cache = restored_cache;
                                    state.prompt_prefix_tokens = reused_tokens.to_vec();
                                    // Only inherit the producer's greedy token when this request
                                    // would compute it too; otherwise leave it unset so the
                                    // consume site resamples with the request's own sampling.
                                    state.cached_prefill_output_token = snapshot
                                        .greedy_prefill_output_token
                                        .filter(|_| prefill_output_token_cacheable(ctx, sampling));
                                    telemetry.hits = telemetry.hits.saturating_add(1);
                                    telemetry.record_restore_source(RESTORE_SOURCE_MEMORY_L1);
                                    telemetry.reused_tokens = telemetry
                                        .reused_tokens
                                        .saturating_add(saturating_u32(snapshot.token_count));
                                    return telemetry;
                                }
                            }
                            Err(e) => {
                                telemetry.record_blocked_restore_error();
                                tracing::warn!(
                                    target: "ax_engine_mlx::prefix_cache",
                                    error = %e,
                                    "L1 prefix-cache payload could not enter the shared FA pool; treating as miss",
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    telemetry.record_blocked_restore_error();
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
        // Durable entries currently contain trunk state only. Replaying an
        // eligible Flash Next prefix rebuilds both histories at the same
        // token boundary; inventing an empty draft cache here is incorrect.
        let disk_restore_unsupported = mla_extend_unsafe || needs_flash_next_cursor;
        if disk_restore_unsupported && self.disk_prefix_cache.is_some() {
            // Name the incompatible restore layout instead of silently
            // skipping the durable tier for this request.
            telemetry.record_blocked_unsupported_layout();
            telemetry.record_disk_admission(
                crate::disk_prefix_cache::DiskAdmissionReason::UnsupportedLayout,
            );
        }
        if !disk_restore_unsupported && self.disk_prefix_cache.is_some() {
            self.record_disk_artifact_identity_if_unavailable(&mut telemetry);
            if let Some(writer) = self.disk_prefix_writer.as_ref() {
                telemetry.absorb_writer_commits(writer.drain_commits());
            }
        }
        if !disk_restore_unsupported
            && let Some(disk) = self.disk_prefix_cache.as_ref()
            && let Some(key_bytes) = self.disk_prefix_key_bytes(&key, reused_tokens)
        {
            match disk.get_restored_timed(&key_bytes) {
                Ok(Some((mut restored, read_timings))) => {
                    // Streaming restore materializes tensors while reading;
                    // stage timings already include payload IO + checksum.
                    let deserialize_us = 0u64;
                    telemetry.record_disk_hit();
                    telemetry.record_disk_restore_stages(read_timings, deserialize_us);
                    // Feed the observed restore throughput back into
                    // the admission cost model.
                    if let Some(writer) = self.disk_prefix_writer.as_ref() {
                        writer.cost_model().record_restore(
                            read_timings.bytes_read,
                            read_timings
                                .read_wall_us
                                .saturating_add(read_timings.checksum_wall_us)
                                .saturating_add(deserialize_us),
                        );
                    }
                    if restored.cache.has_rotated_sliding_layers() {
                        telemetry.record_blocked_unsupported_layout();
                        telemetry.record_disk_admission(
                            crate::disk_prefix_cache::DiskAdmissionReason::UnsupportedLayout,
                        );
                        telemetry.record_disk_fallback_recompute();
                        tracing::warn!(
                            target: "ax_engine_mlx::prefix_cache",
                            "disk prefix-cache payload contains slot-ordered rotating KV; \
                             treating as miss and recomputing ordered prefix state",
                        );
                    } else if let Err(e) = self
                        .verify_restored_prefix_snapshot(&mut restored.cache, reused_tokens.len())
                    {
                        telemetry.record_blocked_snapshot_incomplete();
                        telemetry.record_disk_fallback_recompute();
                        tracing::warn!(
                            target: "ax_engine_mlx::prefix_cache",
                            error = %e,
                            "disk prefix-cache payload is structurally incomplete; \
                             treating as miss and recomputing",
                        );
                    } else {
                        let prepared =
                            self.prepare_portable_prefix_restore(restored.cache, &mut telemetry);
                        match prepared {
                            Ok(restored_cache) => {
                                state.cache = restored_cache;
                                state.prompt_prefix_tokens = reused_tokens.to_vec();
                                // F3 M4 — the entry carries the greedy prefill
                                // output token, so cross-restart L2 hits avoid
                                // recomputing at decode step 0. Sampling policy
                                // still controls whether the token is reusable.
                                state.cached_prefill_output_token = restored
                                    .prefill_output_token
                                    .filter(|_| prefill_output_token_cacheable(ctx, sampling));
                                telemetry.record_restore_source(RESTORE_SOURCE_DISK_L2);
                                telemetry.reused_tokens = telemetry
                                    .reused_tokens
                                    .saturating_add(saturating_u32(reused_tokens.len()));
                                return telemetry;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "ax_engine_mlx::prefix_cache",
                                    error = %e,
                                    "disk prefix-cache payload could not enter the shared FA pool; recomputing",
                                );
                                telemetry.record_blocked_restore_error();
                                telemetry.record_disk_fallback_recompute();
                            }
                        }
                    }
                }
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
                    telemetry.record_disk_fallback_recompute();
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
                needs_flash_next_cursor,
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
        needs_flash_next_cursor: bool,
    ) {
        let mut warmup_rng = if capture_prefill_output {
            state.rng
        } else {
            // Request-id fallback: no explicit client seed, so keep the stream
            // unseeded (greedy warmup never draws anyway, but stay consistent).
            crate::sampling::request_rng(None, request_id.0 ^ 0xA5A5_5A5A_F00D_CAFE)
        };
        let repetition_history = if capture_prefill_output {
            state.repetition_history(tokens, sampling)
        } else {
            Vec::new()
        };
        // Entry-point matrix (docs/designs/kv-weak-surfaces-2026-07-14.md
        // Track A): callers only reach this function with an empty cache
        // (`state.cache.seq_len() == 0`), so it must resolve through the
        // same shared cold/warm selection as every other prefill entry
        // point instead of hardcoding the warm chunk — that divergence is
        // exactly the class of bug that caused the historical MLA
        // warm-extend drift.
        let (prefill_chunk_for_request, prefill_chunk_mode) =
            crate::fastpath::select_prefill_chunk_for_request(
                state.cache.seq_len(),
                self.cold_prefill_chunk,
                self.prefill_chunk,
            );
        state.decode_telemetry.record_prefill_chunk_selection(
            prefill_chunk_for_request,
            matches!(
                prefill_chunk_mode,
                crate::fastpath::PrefillChunkMode::WarmExtend
            ),
        );
        if needs_flash_next_cursor {
            state.cached_prefill_output_token = self.run_flash_next_mtp_prefill(
                state,
                tokens,
                prefill_chunk_for_request,
                capture_prefill_output,
            );
            state.prompt_prefix_tokens = tokens.to_vec();
            return;
        }
        let prefill_output_token = chunked_prefill_with_sampling_buffers(
            &self.cfg,
            &self.weights,
            tokens,
            &mut state.cache,
            prefill_chunk_for_request,
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

    /// Store a snapshot from a mid-prefill boundary capture on a
    /// linear-attention model. The capture's cache is already exactly at
    /// `prefix_len`, so unlike the trim-based store loop no `trim_to` runs.
    /// The snapshot goes to L1 and — since the boundary is by construction
    /// the largest aligned prefix — mirrors to the L2 disk layer when one is
    /// open, matching `store_prompt_prefix_snapshots`' largest-snapshot disk
    /// policy (restore is architecture-agnostic: the serialized payload
    /// carries the linear conv/recurrent state).
    /// Store a snapshot payload in the L2 disk tier. Queued to the
    /// background writer when one is running — `insert` pays a full file
    /// write + `F_FULLFSYNC` + eviction walk under the cross-process lock,
    /// which does not belong inline on the prefill path — and run inline
    /// only as the no-writer fallback. A queue-full drop or write failure
    /// is a skipped store (disk is strictly additive); worker-side
    /// evictions drain into the next recording request's telemetry.
    /// Run the L2 admission policy for one candidate snapshot and record
    /// the decision in request telemetry. Returns whether the entry may be
    /// mirrored to disk. Adaptive value inputs come from the producing
    /// request's measured cold prefill and the background writer's
    /// calibrated throughput model; without either, adaptive fails closed
    /// with `NoCostModel`.
    fn evaluate_disk_admission(
        &self,
        prefix_tokens: usize,
        entry_bytes: u64,
        cold_prefill_us: u64,
        telemetry: &mut MlxPrefixCacheTelemetry,
    ) -> bool {
        let Some(disk) = self.disk_prefix_cache.as_ref() else {
            return false;
        };
        let throughput = self
            .disk_prefix_writer
            .as_ref()
            .and_then(|writer| writer.cost_model().snapshot());
        let (reason, _estimate) = disk.policy().evaluate_admission(
            u32::try_from(prefix_tokens).unwrap_or(u32::MAX),
            entry_bytes,
            (cold_prefill_us > 0).then_some(cold_prefill_us),
            throughput,
        );
        telemetry.record_disk_admission(reason);
        reason.admitted()
    }

    #[allow(clippy::too_many_arguments)]
    fn store_disk_snapshot(
        &self,
        disk: &crate::disk_prefix_cache::DiskPrefixCache,
        key_bytes: Vec<u8>,
        payload: Arc<[u8]>,
        prefill_output_token: Option<u32>,
        producer_cold_prefill_us: u64,
        producer_serialize_us: u64,
        telemetry: &mut MlxPrefixCacheTelemetry,
    ) {
        let payload_bytes = payload.len() as u64;
        if let Some(writer) = self.disk_prefix_writer.as_ref() {
            // Absorb earlier background commits before enqueue so durable
            // counters stay fresh even when this request only queues work.
            telemetry.absorb_writer_commits(writer.drain_commits());
            if writer.enqueue(DiskPrefixWriteJob {
                key_bytes,
                payload,
                prefill_output_token,
                producer_cold_prefill_us,
                producer_serialize_us,
            }) {
                telemetry.record_disk_store_enqueued(payload_bytes);
            } else {
                telemetry.record_disk_store_dropped();
            }
            return;
        }
        match disk.insert_parts(
            &key_bytes,
            &payload,
            prefill_output_token,
            producer_cold_prefill_us,
            producer_serialize_us,
        ) {
            Ok(outcome) => telemetry.record_disk_store_committed(payload_bytes, outcome.evictions),
            Err(e) => {
                telemetry.record_disk_store_commit_failed();
                tracing::warn!(
                    target: "ax_engine_mlx::prefix_cache",
                    error = %e,
                    "disk prefix-cache insert failed; L1 store still active",
                );
            }
        }
    }

    /// When L2 is open but the loaded model has no content fingerprint,
    /// durable keys are ineligible (spec §6). Emit a closed admission
    /// reason so ops can see the bypass instead of a silent skip.
    fn record_disk_artifact_identity_if_unavailable(
        &self,
        telemetry: &mut MlxPrefixCacheTelemetry,
    ) {
        if self.disk_prefix_cache.is_some() && self.artifact_fingerprint.is_none() {
            telemetry.record_disk_admission(
                crate::disk_prefix_cache::DiskAdmissionReason::ArtifactIdentityUnavailable,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn store_linear_boundary_snapshot(
        &self,
        model_id: &str,
        block_size_tokens: u32,
        state: &RequestState,
        prefix_len: usize,
        snapshot_cache: &MlxKVCache,
        cold_prefill_us: u64,
        media_key: &str,
    ) -> MlxPrefixCacheTelemetry {
        let mut telemetry = MlxPrefixCacheTelemetry::default();
        let tokens = &state.prompt_prefix_tokens[..prefix_len];
        let key = self.prefix_cache_key_with_media(model_id, block_size_tokens, tokens, media_key);
        // The boundary snapshot is by construction the largest aligned
        // prefix, so it follows the main store path's disk policy: mirror it
        // to the L2 disk layer when one is open. Compute the disk key up
        // front — it also decides whether an L1-superseding snapshot may
        // early-return (an L1-resident entry still needs a disk write when
        // the disk layer is open but does not have it yet, e.g. after disk
        // eviction).
        let disk_key_bytes = self
            .disk_prefix_cache
            .as_ref()
            .and_then(|_| self.disk_prefix_key_bytes(&key, tokens));
        // Drop the L1 lock before the disk `contains()` stat: holding a
        // mutex across filesystem I/O serializes every other concurrent
        // request's L1 cache access behind the syscall for no correctness
        // reason (matches the pattern in `store_prompt_prefix_snapshots`).
        let l1_superseding = {
            let cache = self.prefix_cache.lock();
            let superseding = cache.contains_superseding_snapshot(&key, tokens, None, false);
            if superseding {
                telemetry.record_stats(cache.stats());
            }
            superseding
        };
        if l1_superseding {
            let disk_store_needed = match (self.disk_prefix_cache.as_ref(), &disk_key_bytes) {
                (Some(disk), Some(key_bytes)) => !disk.contains(key_bytes),
                _ => false,
            };
            if !disk_store_needed {
                return telemetry;
            }
        }
        if !l1_superseding {
            let lower_bound = snapshot_cache
                .usage_snapshot()
                .logical_bytes
                .saturating_add((tokens.len() as u64).saturating_mul(size_of::<u32>() as u64));
            if !self
                .prefix_cache
                .lock()
                .policy
                .admits_lower_bound(lower_bound)
            {
                telemetry.record_blocked_entry_too_large();
                Self::pfx_dbg(
                    "store-skip",
                    &format!("entry_too_large lower_bound={lower_bound}"),
                );
                return telemetry;
            }
        }
        if snapshot_cache.has_unserializable_layers() {
            // DeepSeek V4 compressor state has no wire encoding: the
            // snapshot would claim tokens and restore as empty.
            Self::pfx_dbg("store-skip", "unserializable_layers");
            return telemetry;
        }
        let serialize_started = Instant::now();
        let payload: Arc<[u8]> = snapshot_cache.serialize_to_bytes().into();
        let serialize_us = u64::from(elapsed_us(serialize_started));
        if self.disk_prefix_cache.is_some() {
            self.record_disk_artifact_identity_if_unavailable(&mut telemetry);
        }
        let disk_payload = disk_key_bytes
            .as_ref()
            .filter(|_| {
                self.evaluate_disk_admission(
                    prefix_len,
                    payload.len() as u64,
                    cold_prefill_us,
                    &mut telemetry,
                )
            })
            .map(|_| Arc::clone(&payload));
        let outcome = {
            let mut cache = self.prefix_cache.lock();
            let outcome = cache.insert(
                key,
                MlxPrefixSnapshot::from_shared_payload(
                    payload,
                    tokens.to_vec(),
                    prefix_len,
                    None,
                    None,
                ),
            );
            telemetry.record_stats(cache.stats());
            outcome
        };
        if outcome.stored {
            telemetry.stores = telemetry.stores.saturating_add(1);

            // Boundary snapshots carry no greedy prefill output token: the
            // aligned boundary is mid-prompt, so decode step 0 always
            // recomputes. A disk-write failure does not back out the L1
            // store (disk is strictly additive, matching
            // `store_prompt_prefix_snapshots`).
            if let (Some(disk), Some(key_bytes), Some(payload)) = (
                self.disk_prefix_cache.as_ref(),
                disk_key_bytes,
                disk_payload,
            ) {
                self.store_disk_snapshot(
                    disk,
                    key_bytes,
                    payload,
                    None,
                    cold_prefill_us,
                    serialize_us,
                    &mut telemetry,
                );
            }
        }
        telemetry
    }

    fn pfx_dbg(stage: &str, detail: &str) {
        if crate::fastpath::prefill_time_debug_env() {
            eprintln!("AX_PREFIX_DEBUG {stage} {detail}");
        }
    }

    fn store_prompt_prefix_snapshots(
        &self,
        model_id: &str,
        block_size_tokens: u32,
        state: &RequestState,
        options: PromptPrefixSnapshotStoreOptions<'_>,
    ) -> MlxPrefixCacheTelemetry {
        let PromptPrefixSnapshotStoreOptions {
            linear_boundary_snapshot,
            prefill_completes_prompt,
            greedy_prefill_output_token,
            cold_prefill_us,
            media_key,
        } = options;
        let mut telemetry = MlxPrefixCacheTelemetry::default();
        if block_size_tokens == 0 || state.prompt_prefix_tokens.is_empty() {
            Self::pfx_dbg("store-skip", "empty_prompt_or_block");
            return telemetry;
        }
        // Media present without a digest: storing under a text-only key
        // would let a later prompt with identical tokens but different
        // media adopt this KV. Fail closed.
        let Some(media_key) = media_key else {
            telemetry.record_blocked_media_identity();
            Self::pfx_dbg("store-skip", "media_identity");
            return telemetry;
        };
        if !self.prefix_cache_supported() {
            telemetry.record_blocked_unsupported_layout();
            Self::pfx_dbg("store-skip", "unsupported_layout");
            return telemetry;
        }
        if !self.prefix_cache.lock().enabled() {
            telemetry.record_blocked_policy_disabled();
            Self::pfx_dbg("store-skip", "policy_disabled");
            return telemetry;
        }
        let native_store_enabled = self.native_fa_prefix_sharing_enabled();
        if !portable_prefix_store_allowed(prefill_completes_prompt, native_store_enabled) {
            Self::pfx_dbg("store-skip", "portable_store_not_allowed_partial");
            // A scheduler-split prefill can yield hundreds of execution
            // items. Portable snapshots serialize the complete live KV
            // state, so doing that after every partial item turns an O(N)
            // prefill into O(N²) memory traffic and holds the process-wide
            // Metal turn while sibling decodes wait. Only the native
            // reference-counted FA pool has a cheap partial-boundary store;
            // every other topology publishes its portable snapshot once the
            // prompt is complete.
            return telemetry;
        }
        if state.cache.has_rotated_sliding_layers() {
            Self::pfx_dbg("store-skip", "rotated_sliding_layers");
            // Slot order is a decode-local physical representation, not a
            // prompt-prefix representation. A no-op trim would otherwise
            // serialize it and a later warm extension could issue a
            // multi-token ordered append into ring slots.
            telemetry.record_blocked_unsupported_layout();
            return telemetry;
        }

        let block_size = block_size_tokens as usize;
        let available_tokens = state.prompt_prefix_tokens.len().min(state.cache.seq_len());
        let full_block_tokens = available_tokens - (available_tokens % block_size);
        if full_block_tokens == 0 {
            return telemetry;
        }

        // Non-standard-FA architectures (linear attention, sliding window, MLA)
        // store at most the largest block-aligned prefix snapshot (standard FA
        // stores every block-aligned prefix).
        //   - Linear: `trim_to` does not roll back conv/recurrent state, so a
        //     snapshot is only sound when the trim is a no-op — the prompt must
        //     be exactly block-aligned and only the full-prompt snapshot may be
        //     stored (or a mid-prefill linear boundary capture).
        //   - MLA: `trim_to` is sound for latent/k_pe buffers. Store the
        //     largest block-aligned prefix even when the live prompt is
        //     unaligned so multi-turn sessions can keep hitting after the
        //     first turn (historical exact-alignment-only blocked every
        //     store with `blocked_trim_failure` once chat length left a
        //     block boundary). Warm-extend after that partial store relies
        //     on the R2 same-shape-trail cold/warm chunks.
        //   - Sliding-window: KV storage is append-only until a rotating
        //     backing store engages, so trimming to any block-aligned prefix
        //     is as sound as for standard FA on the default (non-rotating)
        //     serving path; unaligned prompts therefore still store their
        //     largest aligned prefix — the one a follow-up turn extends. On
        //     rollback-free direct sessions where a layer HAS rotated are
        //     rejected above even for a no-op full-prompt trim: slot order is
        //     a request-local decode representation and cannot be safely
        //     warm-extended by an unrelated request.
        // The `verify_prefix_reuse_equivalence.py` harness fails-closed on
        // any resulting token drift; any future change here must keep that
        // harness green on every model in the supported tier.
        let linear_attention = self.cfg.linear_attention.is_some();
        let sliding_window = self.kv_layer_windows.iter().any(Option::is_some);
        let mla_attention = self.cfg.mla_attention.is_some();
        let alignment_restricted = linear_attention || sliding_window || mla_attention;
        let exact_alignment_required = linear_attention;
        if exact_alignment_required && full_block_tokens != available_tokens {
            // Unaligned linear prompts can still store the snapshot captured
            // mid-prefill at the largest block-aligned boundary (see
            // `linear_boundary_capture_head_len`); the capture's cache state
            // is exact at that boundary, so no trim is needed.
            if let Some((boundary_len, boundary_cache)) = linear_boundary_snapshot
                && linear_attention
                && *boundary_len == full_block_tokens
                && boundary_cache.seq_len() == *boundary_len
                && *boundary_len <= state.prompt_prefix_tokens.len()
            {
                telemetry.merge_from(self.store_linear_boundary_snapshot(
                    model_id,
                    block_size_tokens,
                    state,
                    *boundary_len,
                    boundary_cache,
                    cold_prefill_us,
                    media_key,
                ));
                return telemetry;
            }
            telemetry.record_blocked_trim_failure();
            Self::pfx_dbg("store-skip", "trim_failure_unaligned");
            return telemetry;
        }
        let snapshot_start_tokens = prefix_snapshot_start_tokens(
            block_size,
            full_block_tokens,
            alignment_restricted,
            native_store_enabled,
        );

        Self::pfx_dbg(
            "store-attempt",
            &format!(
                "prompt={} full_block={} start={}",
                state.prompt_prefix_tokens.len(),
                full_block_tokens,
                snapshot_start_tokens
            ),
        );
        for prefix_len in (snapshot_start_tokens..=full_block_tokens).step_by(block_size) {
            let tokens = &state.prompt_prefix_tokens[..prefix_len];
            let key =
                self.prefix_cache_key_with_media(model_id, block_size_tokens, tokens, media_key);
            let is_largest = prefix_len == full_block_tokens;
            let snapshot_prefill_output_token = (prefix_len == available_tokens)
                .then_some(greedy_prefill_output_token)
                .flatten();
            // Optional Flash Next MTP draft-cursor sidecar. Keeping it a
            // named local (rather than a literal at each use) keeps the
            // byte-budget pre-check and the supersede check correct.
            //
            // A payload is attached only when this snapshot's trunk is the
            // live cache untrimmed (`prefix_len == state.cache.seq_len()`,
            // the no-op-trim case the exact-alignment trunk store already
            // requires): the cursor's draft history was absorbed through the
            // full live sequence, so a payload bound to a shorter trimmed
            // snapshot could never satisfy `aligned()` on restore.
            // `prefix_snapshot_parts` returns `None` for an unaligned cursor,
            // and no live cursor simply stores a trunk-only snapshot; either
            // way a sidecar problem never fails the store itself.
            //
            // A paused cursor is unaligned by construction (the trunk advanced
            // past it while it buffers rows) and has no payload representation
            // for its buffered rows, so `prefix_snapshot_eligible` fences it
            // out and a restore replays instead.
            let mtp_cursor_payload: Option<Arc<[u8]>> = if prefix_len == state.cache.seq_len()
                && state.flash_next_mtp.prefix_snapshot_eligible()
            {
                let parts = state
                    .flash_next_mtp
                    .cursor
                    .as_ref()
                    .zip(state.cache.qwen4_exp.as_ref())
                    .and_then(|(cursor, trunk)| cursor.prefix_snapshot_parts(trunk));
                parts.map(|(draft_state, stream_hidden)| {
                    Arc::from(MlxKVCache::serialize_qwen4_exp_draft_cursor(
                        draft_state,
                        stream_hidden,
                    ))
                })
            } else {
                None
            };

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
                    mtp_cursor_payload.is_some(),
                );
                if superseding {
                    telemetry.record_stats(cache.stats());
                }
                superseding
            };
            let native_store_needed = native_store_enabled
                && !self
                    .native_prefix_cache
                    .lock()
                    .contains_exact_tokens(&key, tokens);
            let disk_store_needed = is_largest
                && self.disk_prefix_cache.as_ref().is_some_and(|disk| {
                    self.disk_prefix_key_bytes(&key, tokens)
                        .is_some_and(|key_bytes| !disk.contains(&key_bytes))
                });
            // A live standard-FA prefill boundary needs only a runner-local
            // clone-and-pin. Portable serialization is deferred until prompt
            // completion, matching mlxcel's separation between live paged
            // ownership and detached/durable cache payloads. Unsupported or
            // runtime-demoted layouts still fall back to the portable store.
            let portable_store_phase = !native_store_enabled || prefill_completes_prompt;
            if !portable_store_phase && !native_store_needed {
                continue;
            }
            if portable_store_phase && l1_superseding && !disk_store_needed && !native_store_needed
            {
                continue;
            }

            let mut snapshot_cache = state.cache.clone();
            if !snapshot_cache.trim_to(prefix_len) {
                telemetry.record_blocked_trim_failure();
                continue;
            }
            let native_snapshot_eligible = native_store_enabled
                && self
                    .shared_fa_block_pool
                    .as_ref()
                    .is_some_and(|pool| snapshot_cache.uses_fa_block_pool(pool))
                && snapshot_cache.is_native_fa_shareable();
            let mut native_snapshot_ready = native_store_enabled && !native_store_needed;
            if native_store_needed && native_snapshot_eligible {
                let logical_kv_bytes = snapshot_cache.usage_snapshot().logical_bytes;
                let native_outcome = {
                    let mut native = self.native_prefix_cache.lock();
                    if native.rejects_oversized(logical_kv_bytes, tokens.len()) {
                        telemetry.record_blocked_entry_too_large();
                    }
                    native.insert(
                        key.clone(),
                        MlxNativePrefixSnapshot::new(
                            snapshot_cache.clone(),
                            tokens.to_vec(),
                            logical_kv_bytes,
                            snapshot_prefill_output_token,
                        ),
                    )
                };
                native_snapshot_ready = native_outcome.stored;
                if native_outcome.stored {
                    telemetry.native_stores = telemetry.native_stores.saturating_add(1);
                }
                telemetry.native_evictions = telemetry
                    .native_evictions
                    .saturating_add(native_outcome.evictions);
                // Demote live evictions into the portable host-RAM store
                // (ADR-016) before releasing their pool references, and only
                // after the native-cache lock is gone (ADR-006 lock-order
                // contract): both serialization and `MlxKVCache::drop` may
                // touch the shared pool. The demote honors the same
                // portable-store phase gate as the portable publish below.
                demote_native_insert_evictions(
                    &self.prefix_cache,
                    &native_outcome.evicted,
                    portable_store_phase,
                    &mut telemetry,
                );
                drop(native_outcome.retired);
                drop(native_outcome.evicted);
            }
            if !portable_store_phase && native_snapshot_ready {
                Self::pfx_dbg(
                    "store-loop-skip",
                    &format!("native_ready prefix={prefix_len}"),
                );
                continue;
            }
            if l1_superseding && !disk_store_needed {
                Self::pfx_dbg(
                    "store-loop-skip",
                    &format!("l1_superseding prefix={prefix_len}"),
                );
                continue;
            }
            if !l1_superseding {
                let lower_bound = snapshot_cache
                    .usage_snapshot()
                    .logical_bytes
                    .saturating_add((tokens.len() as u64).saturating_mul(size_of::<u32>() as u64))
                    // Account for the optional MTP draft-cursor sidecar up
                    // front: an entry that only fits without the cursor must
                    // not pass the pre-check and then exceed the budget once
                    // the cursor is included.
                    .saturating_add(
                        mtp_cursor_payload
                            .as_ref()
                            .map_or(0, |payload| payload.len() as u64),
                    );
                if !self
                    .prefix_cache
                    .lock()
                    .policy
                    .admits_lower_bound(lower_bound)
                {
                    telemetry.record_blocked_entry_too_large();
                    Self::pfx_dbg(
                        "store-loop-skip",
                        &format!("entry_too_large prefix={prefix_len} lower_bound={lower_bound}"),
                    );
                    continue;
                }
            }
            // F3 M2 — for the disk layer we want the largest valid
            // snapshot persisted; smaller intermediate prefixes stay
            // in L1 only. Without an eviction policy yet (M3), writing
            // every per-block prefix to disk would balloon the cache
            // directory by O(N/block_size) × snapshot bytes per cold
            // prefill. The largest snapshot is also the most useful for
            // future hits because shorter prefixes always derive from
            // it.
            if snapshot_cache.has_unserializable_layers() {
                Self::pfx_dbg("store-skip", "unserializable_layers");
                continue;
            }
            let serialize_started = Instant::now();
            let payload: Arc<[u8]> = snapshot_cache.serialize_to_bytes().into();
            let serialize_us = u64::from(elapsed_us(serialize_started));
            let disk_open = self.disk_prefix_cache.is_some();
            if is_largest && disk_open {
                self.record_disk_artifact_identity_if_unavailable(&mut telemetry);
            }
            let disk_payload = if is_largest
                && disk_open
                && self.artifact_fingerprint.is_some()
                && self.evaluate_disk_admission(
                    prefix_len,
                    payload.len() as u64,
                    cold_prefill_us,
                    &mut telemetry,
                ) {
                Some(Arc::clone(&payload))
            } else {
                None
            };
            // Only clone the key when we'll need it again post-insert
            // (i.e. when the disk path will fire). For the L1-only
            // configuration the original `key` moves cleanly into
            // `cache.insert`, no extra allocation.
            let key_for_disk = disk_payload.as_ref().map(|_| key.clone());
            let payload_len = payload.len();
            let outcome = {
                let mut cache = self.prefix_cache.lock();
                let outcome = cache.insert(
                    key,
                    MlxPrefixSnapshot::from_shared_payload(
                        payload,
                        tokens.to_vec(),
                        prefix_len,
                        snapshot_prefill_output_token,
                        mtp_cursor_payload,
                    ),
                );
                Self::pfx_dbg(
                    "store-insert",
                    &format!(
                        "prefix={prefix_len} payload_bytes={payload_len} serialize_us={serialize_us}"
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
                if let (Some(disk), Some(payload), Some(key_bytes)) = (
                    self.disk_prefix_cache.as_ref(),
                    disk_payload,
                    key_for_disk
                        .as_ref()
                        .and_then(|disk_key| self.disk_prefix_key_bytes(disk_key, tokens)),
                ) {
                    self.store_disk_snapshot(
                        disk,
                        key_bytes,
                        payload,
                        snapshot_prefill_output_token,
                        cold_prefill_us,
                        serialize_us,
                        &mut telemetry,
                    );
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
        // Every decode step counts toward the cost controller's cycle window,
        // including the fallback / think-window / direct routes below that
        // never reach its decision site: the window between two decisions is
        // only a clean depth-cost sample when exactly one step ran in it.
        state.mtp_cost_steps_since_decision = state.mtp_cost_steps_since_decision.saturating_add(1);
        // Think soft-close arming (ds4-style rank probe): refreshed every
        // step from the budget controller inputs. While armed, this step
        // bypasses speculative/direct routing and rank-probes the think
        // close token against materialized logits before sampling.
        state.think_soft_close_armed = options.request_context.is_some_and(|ctx| {
            think_soft_close_armed(
                state.ngram_in_think,
                self.cfg.think_end_token_id,
                ctx.max_output_tokens.saturating_sub(ctx.generated_len),
                state.think_emitted_tokens,
                ctx.max_think_tokens,
                ctx.answer_reserve_tokens,
            )
        });

        // Hard think-budget close due this step: force the single-token path
        // (like the soft-close probe) instead of the post-decode override
        // swapping the sampled token after the fact. The swap is KV-exact
        // only when the step commits nothing beyond its feed token — the
        // double-buffer direct pipeline and the speculative routes commit
        // their emitted tokens in-step, and linear/recurrent state cannot
        // roll back. Routing through run_single_decode keeps the close exact
        // on every path. Loop-detected closes stay post-decode: they keep
        // their stop reason and end the request, so the stale cache is
        // discarded.
        let think_hard_close_due = options.request_context.is_some_and(|ctx| {
            think_budget_close_decision(
                state.ngram_in_think,
                self.cfg.think_end_token_id,
                ctx.max_output_tokens.saturating_sub(ctx.generated_len),
                state.think_emitted_tokens,
                ctx.max_think_tokens,
                ctx.answer_reserve_tokens,
                false,
            )
        });

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

        // Diffusion path: when the diffusion queue is exhausted, denoise a
        // new block (monoblock by default, or budgeted multi-step when
        // AX_DIFFUSION_STEPS_PER_ENGINE_STEP is set).
        if let Some(diff_cfg) = self.cfg.diffusion.as_ref() {
            let token_offset = state.cache.seq_len();
            let remaining_output_budget = options
                .request_context
                .map(|ctx| ctx.max_output_tokens.saturating_sub(ctx.generated_len));
            let commit_policy = crate::diffusion::DiffusionCommitPolicy {
                truncation_terminal_ids: &self.terminal_token_ids,
                request_terminal_ids: options.terminal_token_ids,
                remaining_output_budget,
            };
            // Unset budget → monoblock (historical default).
            let step_budget =
                crate::fastpath::diffusion_steps_per_engine_step().unwrap_or(usize::MAX);

            if step_budget == usize::MAX && state.diffusion_workspace.is_none() {
                let result = crate::diffusion::generate_diffusion_block(
                    &self.cfg,
                    diff_cfg,
                    &self.weights,
                    &mut state.cache,
                    &mut state.rng,
                    token_offset,
                    &mut state.diffusion_embed_table,
                    commit_policy,
                );
                state.decode_telemetry.record_diffusion_block(&result);
                state.pending_diffusion_schedule = Some(DiffusionScheduleUpdate {
                    denoise_steps_in_block: result.denoise_steps,
                    commit_ready: false,
                    block_committed: true,
                });
                let mut queue: VecDeque<u32> = result.tokens.into();
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

            // Budgeted multi-step denoise (Phase B).
            if state.diffusion_workspace.is_none() {
                state.diffusion_workspace = Some(crate::diffusion::open_diffusion_block(
                    &self.cfg,
                    diff_cfg,
                    &self.weights,
                    &mut state.rng,
                    token_offset,
                    &mut state.diffusion_embed_table,
                ));
            }
            let progress = match state.diffusion_workspace.as_mut() {
                Some(ws) => crate::diffusion::advance_diffusion_workspace(
                    ws,
                    &self.cfg,
                    &self.weights,
                    &state.cache,
                    &mut state.rng,
                    &state.diffusion_embed_table,
                    step_budget,
                ),
                None => {
                    // Defensive: fall back to monoblock if open failed unexpectedly.
                    let result = crate::diffusion::generate_diffusion_block(
                        &self.cfg,
                        diff_cfg,
                        &self.weights,
                        &mut state.cache,
                        &mut state.rng,
                        token_offset,
                        &mut state.diffusion_embed_table,
                        commit_policy,
                    );
                    state.decode_telemetry.record_diffusion_block(&result);
                    state.pending_diffusion_schedule = Some(DiffusionScheduleUpdate {
                        denoise_steps_in_block: result.denoise_steps,
                        commit_ready: false,
                        block_committed: true,
                    });
                    let mut queue: VecDeque<u32> = result.tokens.into();
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
            };
            if !progress.commit_ready {
                // Mid-denoise: no visible token yet; schedule DenoiseStep again.
                state.pending_diffusion_schedule = Some(progress.schedule_update());
                return Vec::new();
            }
            let Some(ws) = state.diffusion_workspace.take() else {
                return Vec::new();
            };
            let result = crate::diffusion::commit_diffusion_workspace(
                ws,
                &self.cfg,
                &self.weights,
                &mut state.cache,
                commit_policy,
            );
            state.decode_telemetry.record_diffusion_block(&result);
            state.pending_diffusion_schedule = Some(DiffusionScheduleUpdate {
                denoise_steps_in_block: result.denoise_steps,
                commit_ready: false,
                block_committed: true,
            });
            let mut queue: VecDeque<u32> = result.tokens.into();
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
        // Pure session-direct (README `--ax-direct` / `AX_NO_SPEC`) must take
        // this path for temperature-0 shaped decode *before* MTP / n-gram
        // branches so a stale context bit or attached MTP package cannot
        // regress long-context Gemma to single-step decode (the 2026-07-26
        // decode@2048 failure mode). Speculative sessions still use the
        // stricter `should_use_session_direct_pipeline` predicate.
        // Uncertified V4 nextn is attached (`has_mtp`) but `route_safe` is
        // false, so MTP is not requested. Without this, greedy Flash decode
        // skipped the mlx-lm-style async_eval double-buffer and sat ~1.8×
        // behind mlx-lm (15 vs 28 tok/s on Flash-0731 AXQ 2-bit).
        // Requested Flash Next candidate: must run before the session-direct
        // double buffer so a direct session cannot bypass it. Ineligible
        // steps fall through to the generic routes, which see no drafter.
        if self.flash_next_mtp_session()
            && let Some(tokens) =
                self.try_flash_next_mtp_decode(state, input_tokens, sampling, is_greedy, &options)
        {
            return tokens;
        }

        let pure_direct_pipeline = !think_hard_close_due
            && v4_uncertified_uses_pure_direct_pipeline(
                self.mtp_model_policy.is_deepseek_v4_direct_fallback(),
                self.disable_ngram_acceleration,
                state.think_soft_close_armed,
                sampling.uses_logits_processors(),
                is_greedy || sampling.temperature <= 0.0,
                self.mtp_requested,
            );
        // Greedy Flash Next fallback: a step the cursor cannot serve (it was
        // dropped, or it is paused) decodes on the same double buffer
        // pure-direct sessions use instead of spending the request on the
        // n-gram route. The latch is set by every recorded fallback, so the
        // step that blocked already takes this route; see
        // `FlashNextMtpRequestState::blocked_direct`.
        let flash_next_blocked_greedy = !state.think_soft_close_armed
            && !think_hard_close_due
            && (is_greedy || sampling.temperature <= 0.0)
            && !sampling.uses_logits_processors()
            && state.flash_next_mtp.blocked_direct
            && self.flash_next_mtp_session();
        let direct_pipeline = pure_direct_pipeline
            || flash_next_blocked_greedy
            || (!state.think_soft_close_armed
                && !think_hard_close_due
                && should_use_session_direct_pipeline(
                    self.disable_ngram_acceleration,
                    is_greedy || sampling.temperature <= 0.0,
                    self.has_generic_mtp(),
                    self.mtp_requested,
                ));
        if direct_pipeline {
            let last_token = state
                .next_model_last_token
                .or_else(|| input_tokens.last().copied())
                .unwrap_or(0);
            // LONG_MT uses the materialized singleton arithmetic that the
            // batched-singleton verifier matches. Keep this request-stable:
            // short requests that started on the double buffer must not switch
            // merely because generation later crosses position 512.
            let gemma_moe_long_mt_singleton = self.gemma_moe_aligned_greedy_decode()
                && gemma4_moe_long_mt_enabled()
                && state.cache.seq_len() >= 512
                && state.pending_direct.is_none()
                && state.direct_pipeline_emitted_tokens == 0;
            if gemma_moe_long_mt_singleton {
                return vec![self.run_gemma_sequential_pure_direct(state, last_token)];
            }
            // Default MoE baseline stays on the double buffer. Earlier short
            // S=1 alignment drifted at token 1; an S=1 long path that retained
            // the direct bootstrap double-wrote 74418 and self-looped.
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

        if !state.think_soft_close_armed
            && !think_hard_close_due
            && ngram_request_disabled_direct_fast_path(
                is_greedy,
                sampling.uses_logits_processors(),
                self.has_generic_mtp(),
                state.ngram_acceleration_disabled_for_request,
                state.ngram_request_disable_reason,
            )
        {
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
            think_hard_close_due,
        );
        apply_decode_result(state, &result, options.terminal_token_ids)
    }

    /// One verified Flash Next cursor step, or `None` when this step must
    /// decode direct. Every `None` path leaves the trunk untouched; because
    /// the direct step then advances the trunk without the cursor, the cursor
    /// is dropped and the request stays direct for the rest of generation.
    fn try_flash_next_mtp_decode(
        &self,
        state: &mut RequestState,
        input_tokens: &[u32],
        sampling: MlxSamplingParams,
        is_greedy: bool,
        options: &DecodeOneOptions<'_>,
    ) -> Option<Vec<u32>> {
        // A full-prompt prefix hit restores the whole trunk and schedules the
        // request straight to Decode, so `prepare_flash_next_prefill_cursor`
        // never runs and the cursor stashed at restore time would otherwise
        // sit unused while every step fell back to direct decode. Install it
        // here on the first decode step instead. The stash was already
        // decoded, rebound and verified aligned at restore time; re-check
        // alignment against the live trunk anyway, because a prefill quantum
        // may have advanced it since (in which case `prepare_` already
        // consumed the stash and this take() yields `None`).
        if let Some(restored) = state.flash_next_mtp.pending_restored_cursor.take()
            && state.flash_next_mtp.cursor.is_none()
        {
            let aligned = state
                .cache
                .qwen4_exp
                .as_ref()
                .is_some_and(|trunk| restored.aligned(trunk));
            if aligned {
                state.flash_next_mtp.cursor = Some(restored);
                state.flash_next_mtp.emitted_since_clear = 0;
                let telemetry = &mut state.flash_next_mtp.telemetry;
                telemetry.cursor_restored = telemetry.cursor_restored.saturating_add(1);
            } else {
                // The trunk moved between restore and install (for example a
                // lazy direct prime consumed the first position), so the
                // stash can never install. The telemetry contract counts a
                // stashed payload that fails to verify aligned here; dropping
                // it silently would under-report resumed-without-cursor
                // resumes and hide the misattributed PendingDirect fallback.
                let telemetry = &mut state.flash_next_mtp.telemetry;
                telemetry.resumed_without_cursor =
                    telemetry.resumed_without_cursor.saturating_add(1);
                tracing::warn!(
                    target: "ax_engine_mlx::runner",
                    "Flash Next restored draft cursor lost alignment before install; resuming without a cursor"
                );
            }
        }
        let ctx = options.request_context;
        let remaining_budget = ctx.map_or(0, |ctx| {
            ctx.max_output_tokens.saturating_sub(ctx.generated_len)
        });
        let think_budget_close_due = ctx.is_some_and(|ctx| {
            think_budget_close_decision(
                state.ngram_in_think,
                self.cfg.think_end_token_id,
                remaining_budget,
                state.think_emitted_tokens,
                ctx.max_think_tokens,
                ctx.answer_reserve_tokens,
                false,
            )
        });
        let primary = state
            .next_model_last_token
            .or_else(|| input_tokens.last().copied());
        let strict_greedy = ctx.is_some() && is_greedy && !sampling.uses_logits_processors();
        let mut cursor_aligned = self.flash_next_cursor_aligned(state, primary);
        let mut block = flash_next_mtp_decode_block(
            strict_greedy,
            state.think_soft_close_armed,
            think_budget_close_due,
            state.pending_direct.is_some(),
            remaining_budget as usize,
            state.flash_next_mtp.paused.is_some(),
            cursor_aligned,
        );
        if matches!(block, Some(FlashNextMtpDecodeBlock::CursorPaused)) {
            // `CursorPaused` means every cursor-independent gate has cleared and
            // the cursor is behind only because this request is paused. Catch the
            // head up now so this same step can propose again; when the flush
            // fails the cursor is already gone (fail closed) and the pause is
            // over without a catch-up.
            if !self.flush_flash_next_mtp_pause(state) {
                return self.run_flash_next_mtp_paused_step(
                    state,
                    input_tokens,
                    sampling,
                    FlashNextMtpFallbackReason::CursorUnavailable,
                    options,
                );
            }
            cursor_aligned = self.flash_next_cursor_aligned(state, primary);
            block = flash_next_mtp_decode_block(
                strict_greedy,
                state.think_soft_close_armed,
                think_budget_close_due,
                state.pending_direct.is_some(),
                remaining_budget as usize,
                false,
                cursor_aligned,
            );
        }
        let (None, Some(primary), Some(trunk_weights), Some(head), Some(cursor), Some(trunk_state)) = (
            block,
            primary,
            self.weights.qwen4_exp.as_deref(),
            self.weights.qwen4_exp_mtp.as_deref(),
            state.flash_next_mtp.cursor.as_mut(),
            state.cache.qwen4_exp.as_ref(),
        ) else {
            let reason = block
                .map(FlashNextMtpDecodeBlock::fallback_reason)
                // `ComponentsUnavailable` is defensive: reaching the else with
                // `block == None` requires the tuple destructure above to fail
                // while `cursor_aligned` (which implies primary, cursor and
                // trunk state are all present) held, and
                // `flash_next_mtp_session()` already guarantees both weight
                // sets. The bucket is currently unreachable; keep it so a
                // future session-gate change fails closed into a named reason.
                .unwrap_or(FlashNextMtpFallbackReason::ComponentsUnavailable);
            tracing::debug!(
                target: "ax_engine_mlx::runner",
                ?block,
                cursor_present = state.flash_next_mtp.cursor.is_some(),
                trunk_state_present = state.cache.qwen4_exp.is_some(),
                "Flash Next MTP decode step blocked; falling back to direct decode"
            );
            let disposition = flash_next_mtp_fallback_disposition(
                reason,
                crate::fastpath::flash_next_sticky_fallback_enabled(),
            );
            // A transient block with a usable cursor enters (or continues) the
            // paused state: the cursor keeps its line, this step is served
            // direct, and the stream rows of every token the trunk consumes are
            // buffered for the catch-up.
            if disposition == FlashNextMtpFallbackDisposition::RetainPaused
                && (state.flash_next_mtp.paused.is_some() || cursor_aligned)
            {
                if state.flash_next_mtp.paused.is_none() {
                    state.flash_next_mtp.paused = Some(FlashNextCursorPauseBuffer::default());
                }
                return self.run_flash_next_mtp_paused_step(
                    state,
                    input_tokens,
                    sampling,
                    reason,
                    options,
                );
            }
            // Terminal reason, missing or unusable cursor, or the legacy
            // kill-switch policy: account the step and drop the draft history.
            // `drop_cursor` is a counter no-op when nothing is live.
            self.record_flash_next_mtp_direct_fallback(state, reason);
            state.flash_next_mtp.drop_cursor();
            return None;
        };

        let remaining_output = remaining_budget as usize;
        let step = match cursor.step(
            trunk_weights,
            head,
            trunk_state,
            self.cfg.compile_cache_identity,
            primary,
            remaining_output,
            options.terminal_token_ids,
        ) {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!(
                    target: "ax_engine_mlx::runner",
                    %error,
                    "Flash Next MTP step failed before publishing state; decoding direct"
                );
                let telemetry = &mut state.flash_next_mtp.telemetry;
                telemetry.step_errors = telemetry.step_errors.saturating_add(1);
                self.record_flash_next_mtp_direct_fallback(
                    state,
                    FlashNextMtpFallbackReason::StepError,
                );
                // The transaction contract publishes nothing on error, but a
                // half-applied graph cannot be rolled back from here: drop on
                // the first error instead of retrying through the pause.
                state.flash_next_mtp.drop_cursor();
                return None;
            }
        };

        // Publish the verified trunk, then move the shared cache boundary.
        state.cache.qwen4_exp = Some(step.trunk_state);
        state.cache.advance(step.committed_len);

        // A one-token budget cannot accept a draft; it is neither a miss nor
        // a partial reject, so it contributes no accept/reject sample.
        state.mtp_telemetry.record_correctness_mode(
            MtpCorrectnessMode::GreedyExact,
            MtpProposalLaw::DeterministicDelta,
        );
        if remaining_output > 1 && step.drafted > 0 {
            let sources = [MtpDraftSource::Mtp; 3];
            let drafted = step.drafted.min(sources.len());
            state.mtp_telemetry.record_step(
                drafted,
                step.accepted_drafts.min(drafted),
                &sources[..drafted],
                None,
                step.accepted_drafts.min(drafted),
            );
        }
        state.mtp_telemetry.record_timings(MtpStepTimings {
            verify_forward_wall_us: step.verify_wall_us,
            draft_wall_us: step.draft_wall_us,
            mtp_draft_wall_us: step.draft_wall_us,
            verify_tokens: saturating_u32(step.committed_len),
            emitted_tokens: saturating_u32(step.emitted.len()),
            ..MtpStepTimings::default()
        });
        state.decode_telemetry.record_production_decode_eval();
        let flash_next = &mut state.flash_next_mtp;
        flash_next.telemetry.verified_steps = flash_next.telemetry.verified_steps.saturating_add(1);
        flash_next.telemetry.accepted_steps = flash_next
            .telemetry
            .accepted_steps
            .saturating_add(u32::from(step.accepted));
        flash_next.telemetry.correction_wall_us = flash_next
            .telemetry
            .correction_wall_us
            .saturating_add(step.correction_wall_us);
        flash_next.telemetry.bonus_wall_us = flash_next
            .telemetry
            .bonus_wall_us
            .saturating_add(step.bonus_wall_us);
        flash_next.telemetry.rejection_wall_us = flash_next
            .telemetry
            .rejection_wall_us
            .saturating_add(step.rejection_wall_us);
        flash_next.telemetry.emitted_tokens = flash_next
            .telemetry
            .emitted_tokens
            .saturating_add(saturating_u32(step.emitted.len()));
        let correction_milli =
            crate::model::qwen4_exp_mtp::correction_margin_milli(step.correction_margin);
        flash_next.telemetry.min_correction_margin_milli = flash_next
            .telemetry
            .min_correction_margin_milli
            .min(correction_milli);
        // Mirror the direct pipeline's buffer-cache cadence without relying
        // on an exact modulo hit, since accepted steps emit two tokens.
        flash_next.emitted_since_clear = flash_next
            .emitted_since_clear
            .saturating_add(saturating_u32(step.emitted.len()));
        if self.direct_clear_cache_cadence > 0
            && flash_next.emitted_since_clear >= self.direct_clear_cache_cadence
        {
            flash_next.emitted_since_clear = 0;
            clear_cache();
        }
        Some(apply_decode_result(
            state,
            &step.emitted,
            options.terminal_token_ids,
        ))
    }

    /// Route accounting for one Flash Next step served direct. Pure metrics:
    /// whether the cursor is dropped or kept pending is decided by the caller
    /// through `flash_next_mtp_fallback_disposition`.
    fn record_flash_next_mtp_direct_fallback(
        &self,
        state: &mut RequestState,
        reason: FlashNextMtpFallbackReason,
    ) {
        state.mtp_telemetry.record_direct_fallback();
        state.flash_next_mtp.blocked_direct = true;
        let telemetry = &mut state.flash_next_mtp.telemetry;
        telemetry.direct_fallback_steps = telemetry.direct_fallback_steps.saturating_add(1);
        let slot = &mut telemetry.direct_fallback_by_reason[reason.index()];
        *slot = slot.saturating_add(1);
    }

    /// Whether the live cursor ends exactly at the trunk boundary, so this step
    /// could verify against it.
    fn flash_next_cursor_aligned(&self, state: &RequestState, primary: Option<u32>) -> bool {
        primary.is_some()
            && !state.mtp_bypassed
            && !state.mtp_suspended_for_batched_decode
            && state.cache.mrope_position_delta() == 0
            && state.cache.rope_offset == 0
            && match (&state.flash_next_mtp.cursor, &state.cache.qwen4_exp) {
                (Some(cursor), Some(trunk)) => {
                    trunk.position() == state.cache.seq_len() && cursor.aligned(trunk)
                }
                _ => false,
            }
    }

    /// End a paused Flash Next catch-up.
    ///
    /// The buffered rows carry the head over every token the trunk consumed
    /// while the cursor was paused, in order, so one `absorb` restores the
    /// strict head/trunk alignment the cursor needs to propose again. An
    /// `absorb` failure drops the cursor: a half-advanced head must never
    /// propose. Returns `false` only when the cursor is gone afterwards.
    fn flush_flash_next_mtp_pause(&self, state: &mut RequestState) -> bool {
        let Some(parts) = state.flash_next_mtp.take_pause_absorb_parts() else {
            // No buffered rows: the cursor never fell behind, or the pause ends
            // empty.
            state.flash_next_mtp.paused = None;
            return state.flash_next_mtp.cursor.is_some();
        };
        let (tokens, rows) = match parts {
            Ok(parts) => parts,
            Err(error) => {
                tracing::warn!(
                    target: "ax_engine_mlx::runner",
                    %error,
                    "Flash Next pause buffer is corrupt; dropping the cursor"
                );
                state.flash_next_mtp.drop_cursor();
                return false;
            }
        };
        let Some(head) = self.weights.qwen4_exp_mtp.as_deref() else {
            state.flash_next_mtp.drop_cursor();
            return false;
        };
        let Some(cursor) = state.flash_next_mtp.cursor.as_mut() else {
            return false;
        };
        match cursor.absorb(head, &tokens, &rows) {
            Ok(()) => {
                let telemetry = &mut state.flash_next_mtp.telemetry;
                telemetry.cursor_resumed = telemetry.cursor_resumed.saturating_add(1);
                true
            }
            Err(error) => {
                tracing::warn!(
                    target: "ax_engine_mlx::runner",
                    %error,
                    rows = tokens.len(),
                    "Flash Next paused draft cursor failed to catch up; dropping it"
                );
                state.flash_next_mtp.drop_cursor();
                false
            }
        }
    }

    /// One decode step served direct while the Flash Next cursor is paused.
    ///
    /// The cursor keeps its line and does not propose. The trunk advances on
    /// the route the block reason dictates — a lazy direct token drains first, a
    /// think-control step stays on single decode, anything else runs the greedy
    /// direct double buffer — and the stream row of every token the trunk
    /// consumes joins the catch-up buffer. A step whose capture comes back
    /// empty drops the cursor: a gap in the row history would misalign the head.
    fn run_flash_next_mtp_paused_step(
        &self,
        state: &mut RequestState,
        input_tokens: &[u32],
        sampling: MlxSamplingParams,
        reason: FlashNextMtpFallbackReason,
        options: &DecodeOneOptions<'_>,
    ) -> Option<Vec<u32>> {
        if !matches!(
            reason,
            FlashNextMtpFallbackReason::CursorUnavailable
                | FlashNextMtpFallbackReason::ThinkControl
                | FlashNextMtpFallbackReason::PendingDirect
        ) {
            // Only the pause-preserving reasons reach here; anything else drops
            // in the caller.
            self.record_flash_next_mtp_direct_fallback(state, reason);
            state.flash_next_mtp.drop_cursor();
            return None;
        }
        if state.flash_next_mtp.cursor.is_none() {
            // The cursor went away (a failed catch-up, a step error, or an
            // external drop): there is nothing to keep paused, so end the pause
            // and serve the step on the generic route.
            state.flash_next_mtp.paused = None;
            self.record_flash_next_mtp_direct_fallback(state, reason);
            return None;
        }
        self.record_flash_next_mtp_direct_fallback(state, reason);
        let telemetry = &mut state.flash_next_mtp.telemetry;
        telemetry.cursor_paused_steps = telemetry.cursor_paused_steps.saturating_add(1);

        let last_token = state
            .next_model_last_token
            .or_else(|| input_tokens.last().copied())
            .unwrap_or(0);
        if reason == FlashNextMtpFallbackReason::PendingDirect {
            // The lazy token already occupies the next position and its row was
            // captured by the step that consumed it: materialize it and stop
            // extending the double buffer so the next step can propose again.
            if let DirectPipelineStep::FinishPending(pending) =
                next_direct_pipeline_step(&mut state.pending_direct, true)
            {
                let tok = self.run_direct_pipeline_finish_pending(state, pending);
                state.next_model_last_token = Some(tok);
                return Some(vec![tok]);
            }
        }
        let (tokens, consumed) = if reason == FlashNextMtpFallbackReason::ThinkControl {
            // Think control needs materialized logits for its rank probe and
            // commits nothing beyond the feed token, so it stays on single
            // decode even while paused.
            let (tokens, row) = self.run_single_decode_capturing(state, last_token, sampling);
            (tokens, vec![(last_token, row)])
        } else {
            // Defensive arm: a paused greedy step normally ends through the
            // catch-up flush, so only a cursor that cannot flush lands here.
            let (tok, consumed) = self.run_direct_pipeline_decode_capturing(
                state,
                last_token,
                options.final_by_max_output,
            );
            (vec![tok], consumed)
        };
        for (token, row) in consumed {
            let Some(row) = row else {
                // No stream row for a token the trunk just consumed: the head
                // could never catch up across it, so the cursor goes now.
                tracing::warn!(
                    target: "ax_engine_mlx::runner",
                    "Flash Next paused step has no trunk stream row; dropping the cursor"
                );
                self.record_flash_next_mtp_direct_fallback(
                    state,
                    FlashNextMtpFallbackReason::ComponentsUnavailable,
                );
                state.flash_next_mtp.drop_cursor();
                return None;
            };
            if !state.flash_next_mtp.extend_pause(token, row) {
                tracing::warn!(
                    target: "ax_engine_mlx::runner",
                    "Flash Next paused catch-up buffer overflowed; dropping the cursor"
                );
                return None;
            }
        }
        Some(tokens)
    }

    /// Decode one deterministic token on the direct double-buffer pipeline.
    ///
    /// Used both by explicit direct mode and by request-local n-gram fallback after
    /// a linear-attention request proves it has no useful draft support.  The
    /// pipeline may keep the cache one lazy token ahead, so callers must continue
    /// using this path until the request finishes.
    /// MoE Gemma assistant greedy: pure-direct must be sequential (not
    /// double-buffer) under invariant projections. Dense Gemma keeps
    /// double-buffer.
    fn gemma_moe_aligned_greedy_decode(&self) -> bool {
        self.gemma4_assistant_mtp.is_some()
            && self.cfg.moe_expert_count > 0
            && self.weights.deepseek_v4_nextn.is_none()
    }

    /// MoE pure-direct (smokef95): multi-token greedy S=1 — same function as
    /// MTP-on multi-token under Shared+invariant (formal A/B alignment).
    fn run_gemma_sequential_pure_direct(&self, state: &mut RequestState, last_token: u32) -> u32 {
        use crate::model::forward_all_positions_with_post_norm_greedy;
        use mlx_sys::{argmax, eval};
        state.pending_direct = None;
        let token_offset = state.cache.seq_len();
        let (logits, _post_norm) = forward_all_positions_with_post_norm_greedy(
            &self.cfg,
            &self.weights,
            &[last_token],
            &mut state.cache,
            token_offset,
        );
        state.cache.advance(1);
        let predicted = argmax(&logits, None);
        {
            let kv_refs = state.cache.collect_eval_refs();
            let mut targets: Vec<&mlx_sys::MlxArray> = Vec::with_capacity(1 + kv_refs.len());
            targets.push(&predicted);
            targets.extend(kv_refs);
            eval(&targets);
        }
        let tok = predicted.data_u32()[0];
        state.next_model_last_token = Some(tok);
        state.decode_telemetry.record_production_decode_eval();
        tok
    }

    fn run_direct_pipeline_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        final_by_max_output: bool,
        feed_ngram: bool,
    ) -> u32 {
        let (tok, _consumed) =
            self.run_direct_pipeline_decode_capturing(state, last_token, final_by_max_output);
        // The latch always wins over the scheduler feed, so it must track the
        // token this step emitted: a stale value from an earlier n-gram/MTP
        // step would be fed a second time by the next bootstrap.
        if feed_ngram {
            state.ngram.feed(&[tok]);
        }
        tok
    }

    /// One direct-pipeline step that also reports the stream row of every token
    /// the trunk consumed, in trunk order.
    ///
    /// The row is `None` when the model family has no packed residual stream;
    /// only the Flash Next paused catch-up reads these, and it fails closed on a
    /// missing row. Capture costs one array handle per consumed token: the row
    /// is already part of the forward's output.
    fn run_direct_pipeline_decode_capturing(
        &self,
        state: &mut RequestState,
        last_token: u32,
        final_by_max_output: bool,
    ) -> (u32, Vec<(u32, Option<MlxArray>)>) {
        let mut consumed: Vec<(u32, Option<MlxArray>)> = Vec::with_capacity(2);
        let tok = match next_direct_pipeline_step(&mut state.pending_direct, final_by_max_output) {
            DirectPipelineStep::FinishPending(pending) => {
                self.run_direct_pipeline_finish_pending(state, pending)
            }
            DirectPipelineStep::ContinuePending(bootstrap_token) => {
                self.run_direct_pipeline_once(state, bootstrap_token, &mut consumed)
            }
            DirectPipelineStep::BootstrapFinal => {
                self.run_direct_pipeline_bootstrap_final(state, last_token, &mut consumed)
            }
            DirectPipelineStep::Bootstrap => {
                self.run_direct_pipeline_bootstrap(state, last_token, &mut consumed)
            }
        };
        state.next_model_last_token = Some(tok);
        (tok, consumed)
    }

    fn run_direct_pipeline_bootstrap(
        &self,
        state: &mut RequestState,
        last_token: u32,
        consumed: &mut Vec<(u32, Option<MlxArray>)>,
    ) -> u32 {
        // First generated token: single forward + eval only (TTFT path), then
        // re-enter Bootstrap once to establish the double-buffer
        // (start + advance) for the rest of the stream.
        //
        // The 60e46624 variant established the double-buffer immediately and
        // held the pending slot live across engine turns. Interleaved 3x3 A/B
        // on the S1 dual-model contract (M5 Max, Qwen3.5-9B stream + Gemma 4
        // 12B long prefill) convicted that behavior of a 10% Qwen stream
        // throughput loss (17.08 vs 18.98 tok/s median) and a 13% Gemma
        // prefill-leg slowdown (9552 vs 8427 ms): the held async forward from
        // one model's stream contends with the sibling model's prefill GPU
        // work, and every re-entry after preemption pays two full forwards.
        // Exclusive single-model decode measured neutral either way (~146
        // tok/s p128), so first-token-single-forward is strictly better.
        let bootstrap_started = Instant::now();
        let (first_lazy, stream_row) =
            start_direct_pipeline_capturing(&self.cfg, &self.weights, last_token, &mut state.cache);
        consumed.push((last_token, stream_row));
        state
            .decode_telemetry
            .record_direct_bootstrap(elapsed_us(bootstrap_started));
        if state.direct_pipeline_emitted_tokens == 0 {
            let branch_started = Instant::now();
            let (tok, pending_eval_wall_us, pending_read_wall_us) =
                finish_pending_token(&first_lazy);
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
            state.pending_direct = None;
            state.direct_pipeline_emitted_tokens = 1;
            return tok;
        }
        // Second entry: catch up to double-buffer for the rest of the stream.
        self.run_direct_pipeline_once(state, first_lazy, consumed)
    }

    fn run_direct_pipeline_bootstrap_final(
        &self,
        state: &mut RequestState,
        last_token: u32,
        consumed: &mut Vec<(u32, Option<MlxArray>)>,
    ) -> u32 {
        let bootstrap_started = Instant::now();
        let (bootstrap_token, stream_row) =
            start_direct_pipeline_capturing(&self.cfg, &self.weights, last_token, &mut state.cache);
        consumed.push((last_token, stream_row));
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
        let (tok, pending_eval_wall_us, pending_read_wall_us) = finish_pending_token(&pending);
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

    fn run_direct_pipeline_once(
        &self,
        state: &mut RequestState,
        bootstrap_token: MlxArray,
        consumed: &mut Vec<(u32, Option<MlxArray>)>,
    ) -> u32 {
        let branch_started = Instant::now();
        let stage_profile = crate::generate::direct_pipeline_stage_profile_enabled();
        let op_count_before = stage_profile.then(mlx_sys::op_count_snapshot);
        let (advanced, stream_row) = advance_direct_pipeline_capturing_with_timings(
            &self.cfg,
            &self.weights,
            &bootstrap_token,
            &mut state.cache,
        );
        // This step's forward consumed the pending token and emitted it, so the
        // captured row belongs to the token this step returns.
        consumed.push((advanced.token, stream_row));
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
        // Same consuming step type as the main direct path (I1): with
        // final_by_max_output=true the only happy arm is FinishPending.
        // Callers only invoke this when should_drain_pending_direct_before_ngram
        // is true (greedy + pending present); BootstrapFinal is fail-closed.
        match next_direct_pipeline_step(&mut state.pending_direct, true) {
            DirectPipelineStep::FinishPending(pending) => {
                let tok = self.run_direct_pipeline_finish_pending(state, pending);
                state.ngram.feed(&[tok]);
                state.direct_pipeline_emitted_tokens = 0;
                vec![tok]
            }
            DirectPipelineStep::BootstrapFinal
            | DirectPipelineStep::ContinuePending(_)
            | DirectPipelineStep::Bootstrap => {
                tracing::error!(
                    "direct pipeline state machine invariant violated: \
                     n-gram transition drain without pending_direct; \
                     returning empty token list"
                );
                state.direct_pipeline_emitted_tokens = 0;
                vec![]
            }
        }
    }

    fn run_non_ngram_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        is_greedy: bool,
        final_by_max_output: bool,
    ) -> Option<Vec<u32>> {
        // Fail-closed safety net: session-direct temperature-0 must never
        // regress to single-decode. Non-greedy (temp>0) direct still uses
        // single-decode because the pipeline is argmax-only.
        if self.disable_ngram_acceleration {
            if !sampling.uses_logits_processors() && (is_greedy || sampling.temperature <= 0.0) {
                return Some(vec![self.run_direct_pipeline_decode(
                    state,
                    last_token,
                    final_by_max_output,
                    false,
                )]);
            }
            return Some(self.run_single_decode(state, last_token, sampling));
        }

        if sampling.uses_logits_processors() {
            return Some(self.run_single_decode(state, last_token, sampling));
        }

        if ngram_sampled_verify_unsupported(self.cfg.protected_prefix_sliding_window, sampling) {
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

    fn run_single_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
    ) -> Vec<u32> {
        self.run_single_decode_capturing(state, last_token, sampling)
            .0
    }

    /// One single-decode step, reporting the stream row of the token it consumed
    /// (`None` when the family has no packed residual stream). Only the Flash
    /// Next paused catch-up reads the row; it is empty work for every other
    /// family because the forward already materializes it.
    fn run_single_decode_capturing(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
    ) -> (Vec<u32>, Option<MlxArray>) {
        let branch_started = Instant::now();
        let repetition_history = state.repetition_history(&[], sampling);
        let soft_close = if state.think_soft_close_armed {
            self.cfg
                .think_end_token_id
                .map(|think_end_token_id| ThinkSoftCloseProbe {
                    think_end_token_id,
                    rank: THINK_SOFT_CLOSE_PROBE_RANK,
                })
        } else {
            None
        };
        let mut stream_row = None;
        let result = single_decode_with_sampling_buffers_capturing(
            &self.cfg,
            &self.weights,
            &mut state.cache,
            &mut state.ngram,
            last_token,
            sampling,
            &repetition_history,
            &mut state.rng,
            &mut state.sampling_probs_buf,
            &mut state.sampling_logits_buf,
            &mut state.sampling_candidates_buf,
            soft_close,
            Some(&mut stream_row),
        );
        state
            .decode_telemetry
            .record_single_decode(elapsed_us(branch_started));
        state.decode_telemetry.record_production_decode_eval();
        // See `run_direct_pipeline_decode`: keep the latch on the last token
        // this step produced so the next step feeds it, not an older one.
        if let Some(last) = result.last().copied() {
            state.next_model_last_token = Some(last);
        }
        (result, stream_row)
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
        let mut max_depth = state.mtp_adaptive_max_depth.min(runtime.status.max_depth);
        let base_position = state.cache.seq_len();
        // Dual-edge window fix: successive teacher-forced S=2/S=3 exact in probe;
        // formal LONG_MT reaches first_diff@89 (EOS near-tie) at ~1.16× long.
        // Residual recheck / path alignment attempts desynced earlier. Fail-closed
        // long pure-direct for release exact (smokef132). Opt-in LONG_MT=1.
        let allow_long_mt = gemma4_moe_long_mt_enabled();
        if self.cfg.moe_expert_count > 0 && base_position >= 512 && !allow_long_mt {
            max_depth = 0;
        }
        if max_depth == 0 {
            return (vec![], vec![], vec![]);
        }
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

        // Open once-validated draft session and freeze shared target K/V once
        // for the multi-depth loop (family/projection checks + peek amortize).
        let Ok(mut session) = crate::model::Gemma4AssistantDraftSession::open(
            &runtime.cfg,
            &runtime.weights,
            &self.cfg,
            &self.weights,
            runtime.target_shared_layers,
        ) else {
            return (vec![], vec![], vec![]);
        };
        if session.bind_target_cache(&state.cache).is_err() {
            return (vec![], vec![], vec![]);
        }

        // Ungated (gate disabled, gate <= 0): a single sampled draft carrying a
        // log-prob + distribution so rejection-sampling acceptance can engage.
        // Recurrent multi-depth drafting is the gated greedy path below; the
        // sampled path stays depth-1 (per-depth sampled log-probs are out of scope).
        if first_gate <= 0.0 {
            let bf16_hidden = astype(last_backbone_hidden, MlxDtype::Bfloat16, None);
            let Ok((logits, _projected_hidden)) =
                session.forward_one(last_token, &bf16_hidden, base_position)
            else {
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
        // assistant's `post_projection` "backbone hidden" estimate back in as the
        // next step's hidden. Gemma assistant KV is frozen for the entire block, so
        // every recurrent query uses the bonus token's constant absolute position.
        // Confidence gates stop drafting early (correctness-preserving).
        let deep_gate =
            resolve_gemma4_assistant_mtp_deep_gate(speculation_profile, Some(sampling.temperature))
                .0;
        let cur_hidden = astype(last_backbone_hidden, MlxDtype::Bfloat16, None);

        // Single-materialize multi-depth (opt-in via
        // `AX_MLX_GEMMA4_ASSISTANT_LAZY_MULTI_DEPTH=1`; default OFF — the
        // 12B same-artifact A/B was accept-neutral but not a decode win, see
        // `fastpath.rs`): chain lazy argmax tokens through the depth loop and
        // eval all tokens + confidences once. Gates still apply on the host
        // after materialisation.
        if max_depth > 1
            && matches!(confidence_mode, Gemma4AssistantMtpConfidenceMode::GpuExact)
            && crate::fastpath::gemma4_assistant_lazy_multi_depth_enabled()
        {
            return gemma4_assistant_draft_token_lazy_multi_depth(
                &session,
                last_token,
                &cur_hidden,
                base_position,
                max_depth,
                first_gate,
                deep_gate,
            );
        }

        let mut drafts: Vec<u32> = Vec::with_capacity(max_depth);
        let mut cur_token = last_token;
        let mut cur_hidden = cur_hidden;
        let deep_needs_first_conf =
            crate::fastpath::gemma4_assistant_deep_needs_first_conf_enabled();
        for d in 0..max_depth {
            let draft_position = gemma4_assistant_draft_rope_position(base_position, d);
            let Ok((logits, projected_hidden)) =
                session.forward_one(cur_token, &cur_hidden, draft_position)
            else {
                break;
            };
            // Fuse post_projection into the conf materialize so depth-d's
            // backbone estimate is ready for depth d+1 without a second wave.
            let (token, confidence) = match confidence_mode {
                Gemma4AssistantMtpConfidenceMode::ExactCpu => {
                    eval(&[&logits, &projected_hidden]);
                    argmax_with_softmax_confidence_for_logits(&logits, confidence_mode)
                }
                Gemma4AssistantMtpConfidenceMode::GpuExact => {
                    argmax_with_softmax_confidence_gpu_exact_and_arrays(
                        &logits,
                        &[&projected_hidden],
                    )
                }
            };
            if !crate::model::gemma4_assistant_draft_position_accepted(
                d, confidence, first_gate, deep_gate,
            ) {
                break;
            }
            drafts.push(token);
            // vLLM-aligned dynamic depth: do not pay for a deep forward unless
            // conf0 already clears the deep gate (same bar deep drafts need).
            if d == 0 && deep_needs_first_conf && confidence < deep_gate {
                break;
            }
            cur_token = token;
            cur_hidden = astype(&projected_hidden, MlxDtype::Bfloat16, None);
        }
        (drafts, vec![], vec![])
    }

    /// Measure one true direct singleton step without mutating request state.
    ///
    /// The outer runner enables the exact S=2 verifier arithmetic while MTP is
    /// active. Timing an empty verifier step would therefore overestimate the
    /// direct fallback cost because it suppresses singleton-only kernels. Run
    /// the canonical direct pipeline bootstrap under a nested direct scope on
    /// a cloned target cache, then fully materialize its token before return.
    fn measure_mtp_direct_probe(&self, cache: &MlxKVCache, last_token: u32) -> u32 {
        let mut probe_cache = cache.clone();
        // Lazy-adopt throughput leaves the previous verify graph unevaluated.
        // Materialize it before the probe timer so the reference is a true
        // singleton step, not the leftover verifier eval.
        let pending_eval = probe_cache.collect_eval_refs();
        if !pending_eval.is_empty() {
            eval(&pending_eval);
        }
        let _direct_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        let started = Instant::now();
        let pending = start_direct_pipeline(&self.cfg, &self.weights, last_token, &mut probe_cache);
        let (_token, _eval_wall_us, _read_wall_us) = finish_pending_token(&pending);
        elapsed_us(started)
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
    /// Sampled targets use rejection sampling: accept draft[i] with probability
    /// min(1, p_target(draft[i]) / p_draft(draft[i])). Deterministic greedy
    /// proposals are delta distributions with p_draft=1 and use exact residual
    /// correction on rejection. Greedy targets use argmax verification.
    fn run_mtp_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        ctx: Option<&RunnerRequestContext>,
    ) -> Vec<u32> {
        use crate::ngram_accel::sample_logit_row;
        use mlx_sys::{argmax, eval};

        // Gemma assistant-MTP greedy exact path with no pending draft: pure
        // direct (matches MTP-off / AX_NO_SPEC). Empty-draft pure-direct only
        // under SEQUENTIAL_ORACLE=1. With ORACLE=0, MoE empty draft uses
        // greedy multi-token S=1 (smokef89) so ArgmaxOnly matches pure-direct
        // and post_norm is real for drafting (zeros-hidden drafts contaminate
        // Shared QKV pos0 — smokef88 first_diff@1 after token-0 match).
        // MoE (smokef95): empty-draft always pure-direct S=1 multi-token-greedy.
        let gemma_greedy_exact = sampling.temperature <= 0.0
            && self.gemma4_assistant_mtp.is_some()
            && self.weights.deepseek_v4_nextn.is_none()
            && !sampling.uses_logits_processors()
            && (crate::fastpath::gemma4_assistant_mtp_sequential_oracle_enabled()
                || self.cfg.moe_expert_count > 0);
        if gemma_greedy_exact
            && state.mtp_pending_draft.is_empty()
            && state.mtp_pending_draft_lazy.is_none()
        {
            let final_by_max_output = ctx
                .map(|c| c.generated_len.saturating_add(1) >= c.max_output_tokens)
                .unwrap_or(false);
            // MoE empty-draft:
            // - short: S=1 multi-token for real post_norm (seed drafts) and a
            //   single KV geometry even after adaptive depth drops to 0
            //   (do not mix double-buffer mid-stream — smokef137 agent div).
            // - long fail-closed: double-buffer pure-direct = MTP-off twin.
            let token_offset = state.cache.seq_len();
            let moe_long_fail_closed = self.cfg.moe_expert_count > 0
                && token_offset >= 512
                && !gemma4_moe_long_mt_enabled();
            let moe_force_direct = self.cfg.moe_expert_count > 0
                && crate::fastpath::gemma4_assistant_mtp_sequential_oracle_enabled();
            let (tok, draft_hidden) = if self.cfg.moe_expert_count > 0
                && !moe_force_direct
                && !moe_long_fail_closed
            {
                use crate::model::forward_all_positions_with_post_norm_greedy;
                use mlx_sys::{argmax, eval};
                state.pending_direct = None;
                let (logits, post_norm) = forward_all_positions_with_post_norm_greedy(
                    &self.cfg,
                    &self.weights,
                    &[last_token],
                    &mut state.cache,
                    token_offset,
                );
                state.cache.advance(1);
                let predicted = argmax(&logits, None);
                {
                    let kv_refs = state.cache.collect_eval_refs();
                    let mut targets: Vec<&mlx_sys::MlxArray> =
                        Vec::with_capacity(2 + kv_refs.len());
                    targets.push(&predicted);
                    targets.push(&post_norm);
                    targets.extend(kv_refs);
                    eval(&targets);
                }
                let tok = predicted.data_u32()[0];
                state.next_model_last_token = Some(tok);
                let hidden = slice_post_norm_hidden(&post_norm, 0, self.cfg.hidden_size);
                (tok, hidden)
            } else {
                // Dense empty, sequential-oracle, or MoE long pure-direct.
                let tok =
                    self.run_direct_pipeline_decode(state, last_token, final_by_max_output, false);
                let hidden_size = self.cfg.hidden_size;
                let draft_hidden = mlx_sys::zeros(
                    &[1_i32, 1, hidden_size as i32],
                    mlx_sys::MlxDtype::Bfloat16,
                    None,
                );
                (tok, draft_hidden)
            };
            let draft_started = Instant::now();
            let (draft, log_probs, distributions) =
                self.gemma4_assistant_draft_token(state, tok, &draft_hidden, sampling);
            let mtp_timings = MtpStepTimings {
                draft_wall_us: elapsed_us(draft_started),
                emitted_tokens: 1,
                ..MtpStepTimings::default()
            };
            let drafted = draft.len();
            state.mtp_pending_draft = draft;
            state.mtp_pending_draft_log_probs = log_probs;
            state.mtp_pending_draft_log_prob_temperature = Some(1.0);
            state.mtp_pending_draft_distributions = distributions;
            state.mtp_pending_draft_sources = vec![MtpDraftSource::Gemma4Assistant; drafted];
            // The seed draft has not been verified yet: the next decode step
            // owns its accept/reject `record_step`. Recording it here as a
            // zero-accept step would count a fake complete miss and a 0.0
            // acceptance sample, and double-count the first window.
            if drafted > 0 {
                state
                    .gemma4_assistant_mtp_telemetry
                    .record_submitted(drafted, mtp_timings.draft_wall_us);
            }
            state.mtp_telemetry.record_timings(mtp_timings);
            return vec![tok];
        }

        // Direct single-token reference for the cost-model depth controller.
        // Deliberately not gated on a dense lm_head: it measures the same
        // target pipeline the post-park direct route runs (quantized heads
        // included). Mutually exclusive with the profitability probe by
        // construction: the profitability policy is left ineligible whenever
        // the controller owns the decision.
        if state.mtp_cost_depth.wants_direct_probe() {
            let probe_wall_us = self.measure_mtp_direct_probe(&state.cache, last_token);
            state.mtp_cost_depth.record_direct_probe(probe_wall_us);
        } else if state.mtp_profitability.should_probe_now() {
            let probe_wall_us = self.measure_mtp_direct_probe(&state.cache, last_token);
            state.mtp_profitability.record_direct_probe(probe_wall_us);
        }

        // Complete-round wall time for the measured profitability policy.
        // Start after any calibration probe and before lazy proposal
        // materialization so host synchronization and extraction costs are
        // included, not only named kernel buckets.
        let profitability_round_started = Instant::now();

        // Async-scheduled draft (`AX_MLX_MTP_ASYNC_DRAFT`): in the greedy
        // exact or explicit projected-replay regime the verify graph chains
        // directly on the lazy token arrays, so extraction defers past the verify graph build
        // (overlapping the draft head's GPU forward with that CPU work) and
        // completes inside the verify eval batch. Any other flow
        // materialises the draft here, before the first consumer of
        // `mtp_pending_draft`. Placeholder zeros keep every length-based
        // branch decision identical; no value-based consumer runs before the
        // post-eval overwrite (target-prob construction returns None at
        // temperature zero, and acceptance runs after the verify eval).
        let mut deferred_lazy_draft: Option<crate::mtp::MtpLazyDraft> = None;
        if let Some(lazy) = state.mtp_pending_draft_lazy.take() {
            // Exactness-preserving async draft: Qwen linear-MTP (historical)
            // plus Gemma4 assistant-MTP (same lazy-draft contract; no
            // value-based consumer before verify eval).
            let gemma_assistant_async =
                self.gemma4_assistant_mtp_status.enabled || self.gemma4_assistant_mtp.is_some();
            let defer_qualified = crate::fastpath::mtp_async_draft_enabled()
                && (qwen_linear_mtp_async_draft_allowed(
                    self.cfg.linear_attention.is_some(),
                    crate::fastpath::qwen_linear_mtp_exact_enabled(),
                    crate::fastpath::mtp_linear_projected_replay_enabled(),
                ) || gemma_assistant_async)
                && sampling.temperature <= 0.0
                && !self.mtp_optimistic
                && !state.mtp_bypassed;
            let count = lazy.tokens.len();
            state.mtp_pending_draft_sources = vec![MtpDraftSource::Mtp; count];
            state.mtp_pending_draft_log_probs = Vec::new();
            state.mtp_pending_draft_log_prob_temperature = None;
            state.mtp_pending_draft_distributions = Vec::new();
            if defer_qualified {
                state.mtp_pending_draft = vec![0; count];
                deferred_lazy_draft = Some(lazy);
            } else {
                state.mtp_pending_draft = crate::mtp::mtp_lazy_draft_extract(&lazy);
            }
        }
        let mut pending = state.mtp_pending_draft.clone();
        let token_offset = state.cache.seq_len();
        let has_linear_attention = self.cfg.linear_attention.is_some();
        let replay_kill_switch =
            has_linear_attention && crate::fastpath::mtp_linear_exact_replay_enabled();
        let forced_greedy_replay = has_linear_attention
            && forced_linear_mtp_greedy_revalidation(
                replay_kill_switch || self.qwen_linear_mtp_force_replay,
                sampling,
            );
        let vocab = self.cfg.vocab_size as i32;
        let mut mtp_timings = MtpStepTimings::default();
        // Draft log-probs are computed at T=1.0 (greedy path) or the draft
        // head's sampling temperature (stochastic path). Resolve Qwen, GLM,
        // then DeepSeek V4 nextn — nextn has no draft_sampling struct.
        // DI-DS-MTP: DeepSeek accept rescale must use the *same* temperature
        // as draft sampling (think-aware stochastic T, greedy always 1.0),
        // not mode-only 0.7 while think drafts sampled at 1.0.
        let deepseek_draft_temperature = if self.weights.deepseek_v4_nextn.is_some() {
            Some(
                crate::mtp::deepseek_v4_mtp_sample_and_log_temperature_from_env(
                    state.ngram_in_think,
                    sampling.temperature,
                ),
            )
        } else {
            None
        };
        let draft_sampling_temperature = self
            .weights
            .mtp
            .as_ref()
            .map(|h| h.draft_sampling.temperature)
            .or_else(|| {
                // GLM drafts follow the Qwen gate rule for the recorded T
                // (gate-forced greedy writes 1.0 log-probs), but GLM always
                // computes head-T log-probs, so it records the computed
                // branch — the Qwen consumer rule that can skip them does not
                // apply to this path.
                self.weights.glm_mtp.as_ref().map(|h| {
                    let (glm_gate, _) = resolve_mtp_gate_from_env(
                        Some(sampling.temperature),
                        state.mtp_adaptive_gate.as_ref(),
                        mtp_optimistic_draft_min_confidence_override(),
                        self.mtp_model_policy.glm_gate_default(),
                    );
                    crate::mtp::glm_mtp_draft_log_prob_temperature_from_env(
                        h.draft_sampling.temperature,
                        glm_gate,
                    )
                })
            })
            .or(deepseek_draft_temperature);
        let draft_log_prob_temperature_for_new_drafts = draft_sampling_temperature
            .map(|t| if t > 0.0 { t } else { 1.0 })
            .unwrap_or_else(|| {
                if sampling.temperature > 0.0 {
                    sampling.temperature
                } else {
                    1.0
                }
            });
        // Accept pending drafts with the T they were *drafted* at (carried on state).
        let draft_log_prob_temperature = state
            .mtp_pending_draft_log_prob_temperature
            .unwrap_or(draft_log_prob_temperature_for_new_drafts);
        let model_acceptance_mode = if sampling.temperature > 0.0 {
            MtpModelAcceptanceMode::RejectionSampling
        } else {
            mtp_model_acceptance_mode_from_env()
        };

        // Skip-state consumption (Lightning-MLX always-advance pattern):
        // When the previous step's verify forward captured logits at the last
        // accepted position, we can reuse them to sample the primary token and
        // draft new MTP tokens WITHOUT a fresh model forward.  This halves the
        // number of model forwards: every other step uses skip-state instead.
        //
        // Skip-state is only valid when there's no pending draft to verify
        // (otherwise we need to verify the pending draft first).
        let (skip_logits, skip_argmax, skip_hidden) =
            take_mtp_skip_state(state, forced_greedy_replay);
        let can_skip_qwen = skip_logits.is_some()
            && skip_hidden.is_some()
            && pending.is_empty()
            && self.weights.mtp.is_some()
            && self.mtp_skip_state;
        let can_skip_gemma = skip_logits.is_some()
            && skip_hidden.is_some()
            && pending.is_empty()
            && self.gemma4_assistant_mtp.is_some()
            && self.mtp_skip_state;

        // When skip-state is usable, sample primary + draft new MTP tokens from
        // the saved logits/hidden.  The new drafts are written into `pending`,
        // `mtp_pending_draft_log_probs`, and `mtp_pending_draft_sources` so the
        // existing verify/accept pipeline operates on them unchanged.
        let primary_tok_from_skip: Option<u32> = if can_skip_qwen {
            let sl = skip_logits.unwrap();
            let sh = skip_hidden.unwrap();
            // Sample primary token from skip logits (shape [1, vocab]).
            let primary_tok = skip_state_primary_token(
                &sl,
                skip_argmax,
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
            let (gate, src) = resolve_mtp_gate_from_env(
                Some(sampling.temperature),
                state.mtp_adaptive_gate.as_ref(),
                mtp_optimistic_draft_min_confidence_override(),
                self.mtp_model_policy.qwen_gate_default(),
            );
            state.mtp_draft_gate_x1000 = (gate.clamp(0.0, 1.0) * 1000.0) as u32;
            state.mtp_draft_gate_source = src.route_code();
            let (draft, log_probs, _dist, added, _m) = mtp_draft_tokens_gated(
                &self.weights,
                &self.cfg,
                &sh,
                primary_tok,
                cache,
                Some(state.mtp_adaptive_max_depth),
                &mut state.rng,
                gate,
            );
            mtp_timings.mtp_draft_wall_us = mtp_timings
                .mtp_draft_wall_us
                .saturating_add(elapsed_us(mtp_draft_started));
            state.mtp_decode_count += added;
            state.mtp_pending_draft_log_probs = log_probs;
            // DI-QW-MTP: skip-state drafts use the same gated path as pure MTP.
            // Match the post-verify draft write: a greedy window that computed
            // no log-probs must not carry a temperature either.
            let skip_log_prob_t = self
                .weights
                .mtp
                .as_ref()
                .map(|head| {
                    crate::mtp::qwen_mtp_draft_log_prob_temperature_from_env(
                        head.draft_sampling.temperature,
                        gate,
                    )
                })
                .unwrap_or(draft_log_prob_temperature_for_new_drafts);
            state.mtp_pending_draft_log_prob_temperature = pending_draft_log_prob_temperature_for(
                &state.mtp_pending_draft_log_probs,
                skip_log_prob_t,
            );
            state.mtp_pending_draft_distributions.clear();
            state.mtp_pending_draft_sources = vec![MtpDraftSource::Mtp; draft.len()];
            // Override pending so the verify/accept pipeline sees the new drafts.
            pending = draft;
            Some(primary_tok)
        } else if can_skip_gemma {
            let sl = skip_logits.unwrap();
            let sh = skip_hidden.unwrap();
            let primary_tok = skip_state_primary_token(
                &sl,
                skip_argmax,
                vocab,
                sampling,
                &mut state.rng,
                &mut state.sampling_probs_buf,
                &mut state.sampling_logits_buf,
                &mut state.sampling_candidates_buf,
            );
            let assistant_draft_started = Instant::now();
            let (draft, log_probs, distributions) =
                self.gemma4_assistant_draft_token(state, primary_tok, &sh, sampling);
            mtp_timings.assistant_draft_wall_us = mtp_timings
                .assistant_draft_wall_us
                .saturating_add(elapsed_us(assistant_draft_started));
            state.mtp_pending_draft_log_probs = log_probs;
            state.mtp_pending_draft_log_prob_temperature =
                Some(draft_log_prob_temperature_for_new_drafts);
            state.mtp_pending_draft_distributions = distributions;
            state.mtp_pending_draft_sources = vec![MtpDraftSource::Gemma4Assistant; draft.len()];
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
        let optimistic_allowed = mtp_optimistic_allowed(
            self.weights.glm_mtp.is_some() || self.weights.deepseek_v4_nextn.is_some(),
        );
        let auto_optimistic_enabled = mtp_auto_optimistic_enabled_from_env();
        let can_auto_optimistic = auto_optimistic_enabled
            && optimistic_allowed
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
        let proposal_law = if state.mtp_pending_draft_distributions.is_empty()
            && crate::mtp::mtp_draft_mode_from_env() == crate::mtp::MtpDraftMode::Greedy
        {
            MtpProposalLaw::DeterministicDelta
        } else {
            MtpProposalLaw::Stochastic
        };
        let deterministic_delta_log_probs = (proposal_law == MtpProposalLaw::DeterministicDelta)
            .then(|| vec![0.0_f32; pending.len()]);
        let acceptance_log_probs = deterministic_delta_log_probs
            .as_deref()
            .unwrap_or(&state.mtp_pending_draft_log_probs);
        let target_filter = MtpDraftFilter {
            top_p: sampling.top_p,
            top_k: sampling.top_k,
        };
        let approximate_profile =
            optimistic_allowed && (self.mtp_optimistic || auto_optimistic_enabled);
        state.mtp_telemetry.record_correctness_mode(
            if approximate_profile {
                MtpCorrectnessMode::ApproximateOptimistic
            } else if sampling.temperature > 0.0 {
                MtpCorrectnessMode::SampledExact
            } else {
                MtpCorrectnessMode::GreedyExact
            },
            proposal_law,
        );
        if optimistic {
            state.mtp_telemetry.record_optimistic_step();
        }
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

        // Safety net for the deferred lazy draft: only the exact
        // linear-attention rejection branch chains on the lazy arrays. If the
        // step resolved to any other verify flow (auto-optimistic activated
        // mid-request, kill switches), materialise the real token values now
        // so no placeholder can reach a verifier input.
        if (optimistic || !has_linear_attention)
            && let Some(lazy) = deferred_lazy_draft.take()
        {
            let tokens = crate::mtp::mtp_lazy_draft_extract(&lazy);
            state.mtp_pending_draft = tokens.clone();
            pending = tokens;
        }

        let profitability_mtp_round = is_profitability_mtp_round(
            pending.len(),
            &state.mtp_pending_draft_sources,
            self.mtp_max_depth(),
        );

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
        // Returns logits, draft hidden, acceptance outcome, correction token,
        // whether exact residual correction succeeded, and target argmax tokens.
        // predicted is the target model's argmax tokens for EWMA tracking.
        let mut mtp_refold_hidden: Option<MlxArray> = None;
        let (
            logits_all,
            draft_hidden,
            accept_count,
            all_accepted,
            correction_argmax_tok,
            exact_residual_correction_applied,
            predicted,
        ) = if has_linear_attention {
            // Optional hybrid: retain the exact-profile MTP drafter across
            // cycles, but build this target verifier with stock MLX arithmetic.
            // Projected replay is mandatory so the adopted recurrent state is
            // derived from that same verifier graph. The outer exact scope is
            // restored before the next draft head is built.
            let exact_profile_enabled = crate::fastpath::qwen_linear_mtp_exact_enabled();
            let relaxed_target_verify = crate::fastpath::mtp_relaxed_target_verify_enabled()
                && crate::fastpath::mtp_linear_projected_replay_enabled();
            let _target_verify_fast_scope =
                crate::fastpath::scoped_qwen_linear_mtp_target_verify(relaxed_target_verify);
            let _target_verify_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(
                exact_profile_enabled && !relaxed_target_verify,
            );
            let _verify_qmm_scope =
                crate::model::shared::verify_qmm::QwenMtpVerifyQmmGuard::arm(relaxed_target_verify);
            let native_greedy_logits = sampling.temperature <= 0.0
                && !sampling.uses_logits_processors()
                && crate::fastpath::mtp_native_greedy_verify_logits_enabled();
            if optimistic {
                // ── Explicit approximate optimistic shortcut ──
                // Accept all drafts without rejection sampling. The full draft
                // commits, so no rollback can occur — verify directly on the
                // request cache. Cloning here (as the rejection path below must)
                // would defeat MLX buffer donation and copy every full-attention
                // layer's capacity-sized K/V per MTP cycle, an O(context) tax on
                // the steady-state greedy path.
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
                state.cache.advance(verify_len);
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
                    false,
                    predicted,
                )
            } else {
                // Linear-attention recurrent state cannot be trimmed after a
                // rejected speculative token. Verify on a clone; adopt it only
                // when the full draft is accepted, otherwise recompute the
                // committed prefix on the original cache.
                // Exact profile + drafts within qwen_linear_max_verify_drafts()
                // use the lazy committed-prefix checkpoint (full accept adopts
                // the verify cache; complete miss restores; partial recompute).
                // AX_MLX_MTP_LINEAR_EXACT_REPLAY!=0 forces singleton replay.
                // Unprocessed greedy requests in this non-optimistic branch
                // also revalidate accepted drafts before consuming them.
                let projected_replay = linear_mtp_projected_replay_allowed(
                    pending.len(),
                    crate::fastpath::mtp_linear_projected_replay_enabled(),
                    replay_kill_switch,
                    self.qwen_linear_mtp_force_replay,
                );
                let exact_linear_replay = !projected_replay
                    && linear_mtp_requires_singleton_replay(
                        pending.len(),
                        crate::fastpath::qwen_linear_mtp_exact_enabled(),
                        replay_kill_switch || self.qwen_linear_mtp_force_replay,
                    );
                let clone_started = Instant::now();
                let mut verify_cache = state.cache.clone();
                if !exact_linear_replay && !pending.is_empty() {
                    verify_cache.begin_linear_prefix_capture(1);
                }
                mtp_timings.cache_clone_wall_us = elapsed_us(clone_started);
                let verify_forward_started = Instant::now();
                let (logits_all, post_norm_all) = if let Some(lazy) = deferred_lazy_draft.as_ref() {
                    // Chain the verify graph on the lazy draft-token
                    // arrays: the whole build overlaps the draft head's
                    // still-running GPU forward, and one eval batch
                    // materialises both.
                    let first_tok_data = [verify_input[0]];
                    let first_arr = MlxArray::from_raw_data(
                        first_tok_data.as_ptr() as *const u8,
                        4,
                        &[1_i32],
                        MlxDtype::Uint32,
                    );
                    let mut parts: Vec<&MlxArray> = Vec::with_capacity(1 + lazy.tokens.len());
                    parts.push(&first_arr);
                    for tok in &lazy.tokens {
                        parts.push(tok);
                    }
                    let ids_1d = mlx_sys::concatenate(&parts, 0, None);
                    let compiled = try_whole_compiled_qwen_verify(
                        &self.cfg,
                        &self.weights,
                        &ids_1d,
                        verify_len,
                        &mut verify_cache,
                        token_offset,
                    );
                    if compiled.is_some() {
                        mtp_timings.whole_verify_compile_hits = 1;
                    } else if crate::fastpath::mtp_whole_verify_compile_enabled() {
                        mtp_timings.whole_verify_compile_fallbacks = 1;
                    }
                    compiled.unwrap_or_else(|| {
                        forward_all_positions_with_post_norm_ids(
                            &self.cfg,
                            &self.weights,
                            &ids_1d,
                            verify_len,
                            &mut verify_cache,
                            token_offset,
                            native_greedy_logits,
                        )
                    })
                } else {
                    let ids_1d = MlxArray::from_raw_data(
                        verify_input.as_ptr() as *const u8,
                        verify_input.len().saturating_mul(4),
                        &[i32::try_from(verify_input.len()).unwrap_or(0)],
                        MlxDtype::Uint32,
                    );
                    let compiled = try_whole_compiled_qwen_verify(
                        &self.cfg,
                        &self.weights,
                        &ids_1d,
                        verify_len,
                        &mut verify_cache,
                        token_offset,
                    );
                    if compiled.is_some() {
                        mtp_timings.whole_verify_compile_hits = 1;
                    } else if crate::fastpath::mtp_whole_verify_compile_enabled() {
                        mtp_timings.whole_verify_compile_fallbacks = 1;
                    }
                    compiled.unwrap_or_else(|| {
                        if native_greedy_logits {
                            forward_all_positions_with_post_norm_greedy(
                                &self.cfg,
                                &self.weights,
                                &verify_input,
                                &mut verify_cache,
                                token_offset,
                            )
                        } else {
                            forward_all_positions_with_post_norm(
                                &self.cfg,
                                &self.weights,
                                &verify_input,
                                &mut verify_cache,
                                token_offset,
                            )
                        }
                    })
                };
                mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                verify_cache.advance(verify_len);
                if relaxed_target_verify
                    && projected_replay
                    && crate::fastpath::mtp_rebind_verify_fa_enabled()
                {
                    // The verifier's K/V output is a valid backing store for
                    // every acceptance outcome; only the logical boundary and
                    // gated-delta state differ. Rebind the rollback source now
                    // so its obsolete K/V handles cannot block MLX donation at
                    // the evaluation fence below.
                    let _ = state
                        .cache
                        .rebind_contiguous_attention_storage_from(&verify_cache);
                }
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
                    acceptance_log_probs,
                    vocab,
                    sampling,
                    self.mtp_target_softmax_topk,
                    target_filter,
                    target_prob_workspace,
                );
                mtp_timings.target_softmax_wall_us = mtp_timings
                    .target_softmax_wall_us
                    .saturating_add(elapsed_us(target_softmax_started));
                // Always compute argmax for the correction/bonus fallback.
                let predicted_arr = Some(argmax(&logits_all, None));
                // Opt-in device-side greedy acceptance: reduce the accept
                // decision to a packed `[accept_count, correction]` pair
                // evaluated with the verify batch, skipping the full
                // `predicted` download. The async-draft window compares
                // against the lazy device tokens because host `pending` still
                // holds placeholders here.
                let device_greedy_accept = mtp_device_greedy_accept_allowed(
                    crate::fastpath::mtp_device_greedy_accept_enabled(),
                    sampling.temperature,
                    exact_linear_replay,
                    pending.len(),
                    &state.mtp_pending_draft_sources,
                );
                let device_accept_packed = if device_greedy_accept {
                    let drafts_dev = if let Some(lazy) = deferred_lazy_draft.as_ref() {
                        if lazy.tokens.len() == 1 {
                            lazy.tokens[0].clone()
                        } else {
                            let parts: Vec<&MlxArray> = lazy.tokens.iter().collect();
                            mlx_sys::concatenate(&parts, 0, None)
                        }
                    } else {
                        MlxArray::from_raw_data(
                            pending.as_ptr() as *const u8,
                            pending.len().saturating_mul(4),
                            &[i32::try_from(pending.len()).unwrap_or(0)],
                            MlxDtype::Uint32,
                        )
                    };
                    Some(crate::mtp::mtp_greedy_accept_lazy(
                        predicted_arr.as_ref().unwrap(),
                        &drafts_dev,
                        pending.len(),
                    ))
                } else {
                    None
                };
                let kv_refs = verify_cache.collect_eval_refs();
                // Accept only needs predicted tokens (and optional target
                // probs / lazy drafts). Materialising the full verify cache
                // on a complete miss wastes the reject path; adopt/partial
                // eval it after the decision.
                let split_verify_hidden_eval =
                    crate::fastpath::mtp_split_verify_hidden_eval_enabled();
                let mut accept_targets: Vec<&MlxArray> = Vec::with_capacity(5);
                // With device greedy acceptance armed, the packed pair is a
                // `predicted` consumer and its own top-level target forces the
                // whole window's computation, so `predicted` is not pushed
                // again: the host never downloads it in that arm (and a failed
                // readback below finds it materialised by this same eval).
                if device_accept_packed.is_none() {
                    accept_targets.push(predicted_arr.as_ref().unwrap());
                }
                if !split_verify_hidden_eval {
                    accept_targets.push(&post_norm_all);
                }
                if let Some(ref ltp) = lazy_target_probs {
                    ltp.push_eval_targets(&mut accept_targets);
                }
                if let Some(lazy) = deferred_lazy_draft.as_ref() {
                    for tok in &lazy.tokens {
                        accept_targets.push(tok);
                    }
                }
                if let Some(ref packed) = device_accept_packed {
                    accept_targets.push(packed);
                }
                let verify_eval_started = Instant::now();
                if split_verify_hidden_eval {
                    // MTPLX's lazy-logits verifier deliberately fences the
                    // target trunk before the vocabulary projection. Besides
                    // keeping capture/state side outputs lazy until the
                    // acceptance decision, this prevents the large LM-head
                    // graph from competing with attention/GDN work in one
                    // scheduler batch.
                    eval(&[&post_norm_all]);
                }
                eval(&accept_targets);
                mtp_timings.verify_eval_wall_us = elapsed_us(verify_eval_started);
                if let Some(lazy) = deferred_lazy_draft.take() {
                    // The eval batch above materialised the draft tokens;
                    // replace the placeholder pending values with the real
                    // ones before acceptance reads them.
                    let tokens: Vec<u32> = lazy.tokens.iter().map(|a| a.data_u32()[0]).collect();
                    state.mtp_pending_draft = tokens.clone();
                    pending = tokens;
                }
                let accept_started = Instant::now();
                // Device greedy accept reads back only the packed pair and
                // fails closed: a malformed or truncated readback (or a count
                // outside the window) re-downloads `predicted` and takes the
                // host acceptance path for this step. Nothing else in this arm
                // consumes the host window: rollback, hidden slicing, refold,
                // emission, and telemetry all key off `accept` plus the
                // correction token.
                let device_accept = device_accept_packed
                    .as_ref()
                    .and_then(|packed| device_greedy_accept_pair(packed.data_u32(), pending.len()));
                let mut predicted: Vec<u32> = if device_accept.is_some() {
                    Vec::new()
                } else {
                    predicted_arr
                        .as_ref()
                        .map(|arr| arr.data_u32().to_vec())
                        .unwrap_or_default()
                };
                let target_softmax_extract_started = Instant::now();
                let target_probs_cpu = lazy_target_probs
                    .as_ref()
                    .and_then(|ltp| ltp.extract_cpu_into(&pending, target_prob_workspace));
                mtp_timings.target_softmax_wall_us = mtp_timings
                    .target_softmax_wall_us
                    .saturating_add(elapsed_us(target_softmax_extract_started));
                let target_distributions_cpu: Option<&[TokenDistribution]> = None;

                let (mut accept, device_correction_tok) =
                    if let Some((accept_count, correction_tok)) = device_accept {
                        let all_accepted = accept_count == pending.len();
                        (
                            MtpAcceptOutcome {
                                accept_count,
                                all_accepted,
                                rejection_correction: (!all_accepted).then_some(correction_tok),
                            },
                            Some(correction_tok),
                        )
                    } else {
                        (
                            mtp_accept_count(
                                &pending,
                                acceptance_log_probs,
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
                            ),
                            None,
                        )
                    };
                let ac = accept.accept_count;
                let all_accepted = accept.all_accepted;
                let exact_rejection_correction = (!all_accepted
                    && proposal_law == MtpProposalLaw::DeterministicDelta)
                    .then(|| {
                        sample_exact_mtp_delta_rejection_correction(
                            &logits_all,
                            ac,
                            vocab,
                            sampling,
                            pending[ac],
                            &mut state.rng,
                        )
                    })
                    .flatten();
                let exact_residual_correction_applied = exact_rejection_correction.is_some();
                mtp_timings.accept_wall_us = elapsed_us(accept_started);

                let rollback_started = Instant::now();
                // Batched and singleton target arithmetic can disagree.
                // Explicit forced greedy replay revalidates acceptance;
                // other replay paths retain their existing state/correction
                // contract, and projected replay avoids a backbone replay.
                let recomputed_correction_argmax = if exact_linear_replay {
                    Some(replay_linear_mtp_accepted_prefix(
                        &self.cfg,
                        &self.weights,
                        &mut state.cache,
                        verify_input[0],
                        &pending,
                        token_offset,
                        &mut accept,
                        &mut predicted,
                        forced_greedy_replay,
                    ))
                } else if all_accepted {
                    // Keep the accepted cache lazy in the opt-in scheduling
                    // trial. The next target forward consumes these arrays,
                    // and its normal acceptance barrier materialises the
                    // dependency chain without an extra host-side fence.
                    if !crate::fastpath::mtp_lazy_adopt_state_enabled() {
                        let adopt_eval_started = Instant::now();
                        eval(&kv_refs);
                        mtp_timings.verify_eval_wall_us = mtp_timings
                            .verify_eval_wall_us
                            .saturating_add(elapsed_us(adopt_eval_started));
                    }
                    drop(kv_refs);
                    if verify_cache.trim_to(token_offset + 1 + ac) {
                        verify_cache.clear_linear_prefix_checkpoint();
                        state.cache = verify_cache;
                        None
                    } else {
                        Some(recompute_committed_prefix_with_argmax(
                            &self.cfg,
                            &self.weights,
                            &mut state.cache,
                            verify_input[0],
                            &pending[..ac],
                            token_offset,
                        ))
                    }
                } else if ac == 0 {
                    // The capture point is the committed prefix (after the
                    // last committed token), so a complete miss restores it
                    // at any draft depth inside the exact contract. A whole
                    // verifier carries a compact delta tape instead of full
                    // prefix checkpoints; replay one row from the unchanged
                    // source state in that case.
                    drop(kv_refs);
                    let restored = verify_cache.restore_linear_prefix_checkpoint();
                    let tape_replayed = if restored {
                        false
                    } else if crate::fastpath::mtp_linear_projected_replay_enabled() {
                        self.weights
                            .layers
                            .iter()
                            .enumerate()
                            .filter(|(_, layer)| layer.linear_attn.is_some())
                            .all(|(layer_idx, layer)| {
                                replay_linear_attention_mtp_prefix(
                                    &self.cfg,
                                    layer,
                                    &state.cache,
                                    &mut verify_cache,
                                    layer_idx,
                                    1,
                                )
                            })
                    } else {
                        false
                    };
                    if (restored || tape_replayed) && verify_cache.trim_to(token_offset + 1) {
                        // Keep the restored checkpoint lazy. The next target
                        // forward consumes these recurrent states and its
                        // normal eval barrier materialises them in dependency
                        // order, avoiding a rejection-only GPU/CPU round trip.
                        // Skip-checkpoint complete-misses take the replay arm
                        // and would otherwise retain a stale capture/stash.
                        verify_cache.clear_linear_prefix_checkpoint();
                        state.cache = verify_cache;
                        None
                    } else {
                        Some(recompute_committed_prefix_with_argmax(
                            &self.cfg,
                            &self.weights,
                            &mut state.cache,
                            verify_input[0],
                            &pending[..ac],
                            token_offset,
                        ))
                    }
                } else if crate::fastpath::mtp_linear_projected_replay_enabled() {
                    // The verifier already produced every linear layer's
                    // QKV/A/B inputs. Reuse their accepted prefix to rebuild
                    // only gated-delta state, then trim full-attention KV.
                    // This is the replay-free-backbone rollback used by oMLX
                    // 0.6.2; the unchanged source cache remains the fallback.
                    drop(kv_refs);
                    let keep = 1 + ac;
                    let replayed = self
                        .weights
                        .layers
                        .iter()
                        .enumerate()
                        .filter(|(_, layer)| layer.linear_attn.is_some())
                        .all(|(layer_idx, layer)| {
                            replay_linear_attention_mtp_prefix(
                                &self.cfg,
                                layer,
                                &state.cache,
                                &mut verify_cache,
                                layer_idx,
                                keep,
                            )
                        });
                    if replayed && verify_cache.trim_to(token_offset + keep) {
                        verify_cache.clear_linear_prefix_checkpoint();
                        let replay_refs = verify_cache.collect_eval_refs();
                        if !crate::fastpath::mtp_lazy_adopt_state_enabled() {
                            let replay_eval_started = Instant::now();
                            eval(&replay_refs);
                            mtp_timings.verify_eval_wall_us = mtp_timings
                                .verify_eval_wall_us
                                .saturating_add(elapsed_us(replay_eval_started));
                        }
                        drop(replay_refs);
                        state.cache = verify_cache;
                        None
                    } else {
                        Some(recompute_committed_prefix_with_argmax(
                            &self.cfg,
                            &self.weights,
                            &mut state.cache,
                            verify_input[0],
                            &pending[..ac],
                            token_offset,
                        ))
                    }
                } else {
                    Some(recompute_committed_prefix_with_argmax(
                        &self.cfg,
                        &self.weights,
                        &mut state.cache,
                        verify_input[0],
                        &pending[..ac],
                        token_offset,
                    ))
                };
                mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                let ac = accept.accept_count;
                let all_accepted = accept.all_accepted;
                let draft_hidden = slice_post_norm_hidden(&post_norm_all, ac, self.cfg.hidden_size);
                mtp_refold_hidden = Some(post_norm_all.clone());
                // Device accept packs `predicted[ac]` as the correction token;
                // `exact_linear_replay` (the only mutator of `accept` /
                // `predicted`) is excluded by `mtp_device_greedy_accept_allowed`.
                let verifier_argmax_tok = device_correction_tok
                    .unwrap_or_else(|| predicted.get(ac).copied().unwrap_or(0));
                let correction_token = select_linear_mtp_correction_token(
                    sampling.temperature,
                    recomputed_correction_argmax,
                    exact_rejection_correction,
                    accept.rejection_correction,
                    verifier_argmax_tok,
                );
                (
                    logits_all,
                    draft_hidden,
                    ac,
                    all_accepted,
                    correction_token,
                    exact_residual_correction_applied,
                    predicted,
                )
            }
        } else {
            // Non-linear-attention: run directly, trim on rejection.
            if optimistic {
                // ── Explicit approximate optimistic shortcut ──
                let ac = pending.len();
                let needs_predicted =
                    sampling.temperature <= 0.0 || (auto_optimistic && !self.mtp_optimistic);
                let verify_forward_started = Instant::now();
                let (logits_all, post_norm_all) = if self.weights.deepseek_v4_nextn.is_some() {
                    // V4 nextn consumes the packed pre-collapse residual; the
                    // second return is `[1, seq, hc*hidden]`, not post-norm.
                    crate::model::deepseek_v4_forward_all_positions_with_packed(
                        &self.cfg,
                        &self.weights,
                        &verify_input,
                        &mut state.cache,
                        token_offset,
                    )
                } else if needs_predicted {
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
                state.cache.advance(verify_len);
                let predicted_arr = needs_predicted.then(|| argmax(&logits_all, None));
                let draft_hidden =
                    slice_post_norm_hidden(&post_norm_all, ac, self.mtp_draft_hidden_width());
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
                    false,
                    predicted,
                )
            } else {
                // Greedy Gemma4 assistant-MTP: multi-token teacher-forced
                // argmax can disagree with singleton production decode (shared
                // KV / sliding window / softcap path). Formal pilots accepted
                // drafts that sequential greedy would not emit, breaking
                // MTP-off/on exactness. Fail closed via sequential oracle
                // (kill-switch: AX_MLX_GEMMA4_ASSISTANT_MTP_SEQUENTIAL_ORACLE=0
                // restores multi-token for experiments only).
                // Empty pending is handled at run_mtp_decode entry (direct
                // pipeline) so this branch always has drafts when the oracle
                // is engaged.
                // Sequential pure-direct oracle intentionally disabled for the
                // aligned multi-token path: pure-direct re-verify is exact but
                // never multi-token-fast. Kill-switch still exists for experiments
                // that restore pure-direct MTP-off + oracle, but formal Tier 2
                // uses multi-token on both arms via gemma_assistant_exact_align_path.
                // Greedy Gemma4 assistant-MTP hybrid:
                // - Default / SEQUENTIAL_ORACLE=1 on looping history: pure-direct
                //   sequential (exact vs MTP-off; formal pilot divergences were
                //   almost always cycle-continuation false accepts).
                // - SEQUENTIAL_ORACLE=0 and non-looping: multi-token on a clone,
                //   full-accept adopt for Tier 2 speed.
                let gemma_assistant_draft = sampling.temperature <= 0.0
                    && !pending.is_empty()
                    && state
                        .mtp_pending_draft_sources
                        .contains(&MtpDraftSource::Gemma4Assistant)
                    && self.weights.deepseek_v4_nextn.is_none();
                // SEQUENTIAL_ORACLE=1 → always pure-direct.
                // SEQUENTIAL_ORACLE=0 → multi-token full-accept only; pure-direct
                // on reject (avoids multi-token pos0 contamination, smoke18 t0)
                // and when history is looping (smoke18 t4 cycle false-accept).
                // SEQUENTIAL_ORACLE=1 → always pure-direct (exact, ~0.91×).
                // SEQUENTIAL_ORACLE=0 → pure multi-token always-adopt + f32 SDPA
                // + full-context sliding views (identity for dense 12B6/31B;
                // MoE/4-bit rechecked under formal A/B).
                // SEQUENTIAL_ORACLE=1 → pure-direct. ORACLE=0 → multi-token
                // always-adopt + f32 SDPA (formal4: 12B6 general + all 31B).
                // MoE multi-token still diverges; pure-direct keeps exactness
                // (speed deferred until MoE multi-token identity lands).
                // ORACLE=1 → pure-direct. ORACLE=0 → multi-token always-adopt
                // for dense. MoE multi-token still diverges early; pure-direct
                // keeps exactness (speed deferred).
                // ORACLE=1 → pure-direct. ORACLE=0 → multi-token for dense.
                // MoE multi-token still diverges (f32 router reduced div 3→2 only
                // with per-pos FFN that killed speed); pure-direct keeps exact.
                // ORACLE=1 → pure-direct. ORACLE=0 → multi-token (dense + MoE).
                // MoE uses tail per-pos dual-path FFN for short multi-token verify.
                // Pending-draft verify: multi-token when ORACLE=0 (dense + MoE).
                // Empty-draft pure-direct for MoE is separate (gemma_greedy_exact).
                // ORACLE=1 → pure-direct. ORACLE=0 → multi-token (dense + MoE)
                // unless the cycle-continuation guard forces pure-direct (formal
                // pilot divergence mode: teacher-forced multi-token false-accepts
                // cycle drafts while sequential greedy would break the loop).
                let force_pure_direct =
                    crate::fastpath::gemma4_assistant_mtp_sequential_oracle_enabled();
                let cycle_guard_on = crate::fastpath::gemma4_assistant_mtp_cycle_guard_enabled();
                let early_gen_on =
                    crate::fastpath::gemma4_assistant_mtp_early_gen_pure_direct_enabled();
                let early_gen_force = gemma_assistant_draft
                    && gemma_early_gen_pure_direct_force(
                        early_gen_on,
                        state.generated_tokens.len(),
                        GEMMA_MT_EARLY_GEN_PURE_DIRECT_TOKENS,
                    );
                let mut cycle_history_buf = [0u32;
                    GEMMA_CYCLE_GUARD_MAX_PERIOD
                        .saturating_mul(2)
                        .saturating_add(1)];
                let cycle_history_len =
                    if gemma_assistant_draft && !force_pure_direct && cycle_guard_on {
                        fill_gemma_cycle_history(
                            &state.generated_tokens,
                            verify_input[0],
                            &mut cycle_history_buf,
                        )
                    } else {
                        0
                    };
                let cycle_hit = cycle_history_len > 0
                    && draft_continues_committed_cycle(
                        &cycle_history_buf[..cycle_history_len],
                        &pending,
                    );
                let cycle_force = cycle_hit || state.gemma_mtp_cycle_latched;
                let verify_route = gemma_greedy_verify_route(
                    force_pure_direct,
                    cycle_guard_on && gemma_assistant_draft,
                    cycle_force,
                    early_gen_force,
                );
                let gemma_sequential_oracle = gemma_assistant_draft
                    && matches!(verify_route, GemmaGreedyVerifyRoute::SequentialOracle);
                let gemma_multitoken_adopt = gemma_assistant_draft
                    && matches!(verify_route, GemmaGreedyVerifyRoute::MultiTokenAdopt);
                if gemma_assistant_draft && cycle_force && !force_pure_direct {
                    if cycle_hit {
                        state.gemma_mtp_cycle_latched = true;
                    }
                    state.mtp_telemetry.record_gemma_cycle_guard();
                }
                if gemma_sequential_oracle {
                    let verify_forward_started = Instant::now();
                    let final_by_max = false;
                    let mut ac = 0usize;
                    let mut predicted: Vec<u32> = Vec::with_capacity(pending.len() + 1);
                    let mut correction = 0u32;
                    for &draft in &pending {
                        let tok = self.run_direct_pipeline_decode(
                            state,
                            verify_input[0],
                            final_by_max,
                            false,
                        );
                        predicted.push(tok);
                        if tok != draft {
                            correction = tok;
                            break;
                        }
                        ac += 1;
                        correction = tok;
                    }
                    let all_accepted = ac == pending.len();
                    if all_accepted {
                        correction = self.run_direct_pipeline_decode(
                            state,
                            verify_input[0],
                            final_by_max,
                            false,
                        );
                        predicted.push(correction);
                    }
                    mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                    mtp_timings.verify_eval_wall_us = 0;
                    let accept_started = Instant::now();
                    mtp_timings.accept_wall_us = elapsed_us(accept_started);
                    let rejected_count = pending.len() - ac;
                    if rejected_count > 0 {
                        let new_mtp_len = state.mtp_decode_count.saturating_sub(rejected_count);
                        if let Some(ref mut c) = state.mtp_cache
                            && !c.trim_to(new_mtp_len)
                        {
                            tracing::warn!(
                                new_mtp_len,
                                "MTP head cache trim refused; draft quality may degrade"
                            );
                        }
                        state.mtp_decode_count = new_mtp_len;
                    }
                    mtp_timings.rollback_wall_us = 0;
                    let hidden_size = self.cfg.hidden_size;
                    let draft_hidden = mlx_sys::zeros(
                        &[1_i32, 1, hidden_size as i32],
                        mlx_sys::MlxDtype::Bfloat16,
                        None,
                    );
                    let logits_all =
                        mlx_sys::zeros(&[1_i32, vocab], mlx_sys::MlxDtype::Float32, None);
                    (
                        logits_all,
                        draft_hidden,
                        ac,
                        all_accepted,
                        correction,
                        false,
                        predicted,
                    )
                } else if gemma_multitoken_adopt {
                    // In-place multi-token always-adopt + dual-edge window.
                    // LONG_MT aligns attention reductions and quantized
                    // projections with its materialized S=1 baseline; no
                    // accepted KV is replayed or rewritten mid-stream.
                    let verify_forward_started = Instant::now();
                    let sensitive_f32 = gemma_sensitive_f32_active(
                        self.cfg.moe_expert_count == 0,
                        token_offset.saturating_add(verify_len),
                        crate::fastpath::dense_long_mt_bf16_fold_enabled(),
                        crate::fastpath::gemma4_assistant_mtp_sensitive_f32_enabled(),
                    );
                    let sensitive_range = sensitive_f32
                        .then(|| gemma_sensitive_f32_layer_range(self.cfg.layer_count))
                        .flatten();
                    if sensitive_range.is_some() {
                        state.mtp_telemetry.record_gemma_sensitive_f32();
                    }
                    let _sensitive_fold_scope = sensitive_range.map(|(start, count)| {
                        crate::fastpath::scoped_dense_long_mt_f32_range(start, count)
                    });
                    state.pending_direct = None;
                    let (logits_mt, post_norm_all) = forward_all_positions_with_post_norm_greedy(
                        &self.cfg,
                        &self.weights,
                        &verify_input,
                        &mut state.cache,
                        token_offset,
                    );
                    state.cache.advance(verify_len);
                    let predicted_arr = argmax(&logits_mt, None);
                    {
                        let kv_refs = state.cache.collect_eval_refs();
                        let mut targets: Vec<&MlxArray> = Vec::with_capacity(2 + kv_refs.len());
                        targets.push(&predicted_arr);
                        targets.push(&post_norm_all);
                        targets.extend(kv_refs);
                        eval(&targets);
                    }
                    mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                    let predicted: Vec<u32> = predicted_arr.data_u32().to_vec();
                    let accept_started = Instant::now();
                    let ac =
                        crate::ngram_accel::greedy_draft_target_accept_count(&pending, &predicted);
                    let all_accepted = ac == pending.len();
                    let correction = predicted.get(ac).copied().unwrap_or(0);
                    let committed_len = token_offset + 1 + ac;
                    if !state.cache.trim_to(committed_len) {
                        tracing::warn!(committed_len, "Gemma multi-token production trim refused");
                    }
                    mtp_timings.accept_wall_us = elapsed_us(accept_started);
                    mtp_timings.verify_eval_wall_us = 0;
                    let rejected_count = pending.len() - ac;
                    if rejected_count > 0 {
                        let new_mtp_len = state.mtp_decode_count.saturating_sub(rejected_count);
                        if let Some(ref mut c) = state.mtp_cache
                            && !c.trim_to(new_mtp_len)
                        {
                            tracing::warn!(
                                new_mtp_len,
                                "MTP head cache trim refused; draft quality may degrade"
                            );
                        }
                        state.mtp_decode_count = new_mtp_len;
                    }
                    mtp_timings.rollback_wall_us = 0;
                    let draft_hidden =
                        slice_post_norm_hidden(&post_norm_all, ac, self.cfg.hidden_size);
                    (
                        logits_mt,
                        draft_hidden,
                        ac,
                        all_accepted,
                        correction,
                        false,
                        predicted,
                    )
                } else if self.weights.deepseek_v4_nextn.is_some() {
                    // DeepSeek V4 greedy exact path: singleton sequential verify
                    // on production (same route as direct decode). Multi-token
                    // teacher-forced verify + clone adopt can disagree with
                    // sequential greedy on compressor / latent-K state after a
                    // full draft accept — that was the observed Tier-2 greedy
                    // mismatch. Sampled targets keep the clone + multi-token
                    // path below for rejection sampling.
                    let greedy_exact = sampling.temperature <= 0.0
                        && !sampling.uses_logits_processors()
                        && model_acceptance_mode == MtpModelAcceptanceMode::Greedy;
                    if greedy_exact {
                        let verify_forward_started = Instant::now();
                        let seq = sequential_greedy_deepseek_v4_mtp_verify(
                            &self.cfg,
                            &self.weights,
                            &mut state.cache,
                            verify_input[0],
                            &pending,
                            token_offset,
                            self.mtp_draft_hidden_width(),
                        );
                        mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                        mtp_timings.verify_eval_wall_us = 0;
                        let accept_started = Instant::now();
                        let ac = seq.accept_count;
                        let all_accepted = ac == pending.len();
                        mtp_timings.accept_wall_us = elapsed_us(accept_started);

                        let rollback_started = Instant::now();
                        let committed_len =
                            token_offset + deepseek_v4_mtp_committed_verify_len(ac, pending.len());
                        debug_assert_eq!(
                            state.cache.seq_len(),
                            committed_len,
                            "DeepSeek V4 sequential MTP commit length must match accept count"
                        );
                        let rejected_count = pending.len().saturating_sub(ac);
                        if rejected_count > 0 {
                            let new_mtp_len = state.mtp_decode_count.saturating_sub(rejected_count);
                            if let Some(ref mut c) = state.mtp_cache
                                && !c.trim_to(new_mtp_len)
                            {
                                tracing::warn!(
                                    new_mtp_len,
                                    "MTP head cache trim refused; draft quality may degrade"
                                );
                            }
                            state.mtp_decode_count = new_mtp_len;
                        }
                        mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                        state.pending_direct = None;
                        (
                            seq.last_logits,
                            seq.draft_hidden,
                            ac,
                            all_accepted,
                            seq.correction_token,
                            false,
                            seq.predicted,
                        )
                    } else {
                        // Sampled / non-greedy: verify on a clone so rejected
                        // drafts never corrupt production compressor state.
                        // Live-cache trim_to only best-effort rewinds the
                        // compressor buffer and can zero-fill missing history.
                        let mut verify_cache = state.cache.clone();
                        let verify_forward_started = Instant::now();
                        let (logits_all, post_norm_all) =
                            crate::model::deepseek_v4_forward_all_positions_with_packed(
                                &self.cfg,
                                &self.weights,
                                &verify_input,
                                &mut verify_cache,
                                token_offset,
                            );
                        mtp_timings.verify_forward_wall_us = elapsed_us(verify_forward_started);
                        verify_cache.advance(verify_len);
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
                            acceptance_log_probs,
                            vocab,
                            sampling,
                            self.mtp_target_softmax_topk,
                            target_filter,
                            target_prob_workspace,
                        );
                        mtp_timings.target_softmax_wall_us = mtp_timings
                            .target_softmax_wall_us
                            .saturating_add(elapsed_us(target_softmax_started));
                        let predicted_arr = Some(argmax(&logits_all, None));
                        let kv_refs2 = verify_cache.collect_eval_refs();
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
                            acceptance_log_probs,
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
                        let exact_rejection_correction = (!all_accepted
                            && proposal_law == MtpProposalLaw::DeterministicDelta)
                            .then(|| {
                                sample_exact_mtp_delta_rejection_correction(
                                    &logits_all,
                                    ac,
                                    vocab,
                                    sampling,
                                    pending[ac],
                                    &mut state.rng,
                                )
                            })
                            .flatten();
                        let exact_residual_correction_applied =
                            exact_rejection_correction.is_some();
                        mtp_timings.accept_wall_us = elapsed_us(accept_started);

                        let rollback_started = Instant::now();
                        // Primary (verify_input[0] == last_token) is not yet in
                        // production at token_offset; even ac==0 commits length 1.
                        let committed_len =
                            token_offset + deepseek_v4_mtp_committed_verify_len(ac, pending.len());
                        if all_accepted {
                            state.cache = verify_cache;
                        } else {
                            // Reject: never adopt the clone (compressor trim is
                            // incomplete). Replay committed prefix on production.
                            drop(verify_cache);
                            let accepted_input = &verify_input[..=ac];
                            let _ = crate::model::deepseek_v4_forward_all_positions_with_packed(
                                &self.cfg,
                                &self.weights,
                                accepted_input,
                                &mut state.cache,
                                token_offset,
                            );
                            state.cache.advance(accepted_input.len());
                            let kv_refs = state.cache.collect_eval_refs();
                            eval(&kv_refs);
                        }
                        debug_assert_eq!(
                            state.cache.seq_len(),
                            committed_len,
                            "DeepSeek V4 MTP commit length must match accept count"
                        );

                        let rejected_count = pending.len() - ac;
                        if rejected_count > 0 {
                            let new_mtp_len = state.mtp_decode_count.saturating_sub(rejected_count);
                            if let Some(ref mut c) = state.mtp_cache
                                && !c.trim_to(new_mtp_len)
                            {
                                tracing::warn!(
                                    new_mtp_len,
                                    "MTP head cache trim refused; draft quality may degrade"
                                );
                            }
                            state.mtp_decode_count = new_mtp_len;
                        }
                        mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                        state.pending_direct = None;
                        let draft_hidden = slice_post_norm_hidden(
                            &post_norm_all,
                            ac,
                            self.mtp_draft_hidden_width(),
                        );
                        let correction_argmax_tok = predicted.get(ac).copied().unwrap_or(0);
                        (
                            logits_all,
                            draft_hidden,
                            ac,
                            all_accepted,
                            exact_rejection_correction
                                .or(accept.rejection_correction)
                                .unwrap_or(correction_argmax_tok),
                            exact_residual_correction_applied,
                            predicted,
                        )
                    }
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
                    state.cache.advance(verify_len);
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
                        acceptance_log_probs,
                        vocab,
                        sampling,
                        self.mtp_target_softmax_topk,
                        target_filter,
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
                        acceptance_log_probs,
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
                    let exact_rejection_correction = (!all_accepted
                        && proposal_law == MtpProposalLaw::DeterministicDelta)
                        .then(|| {
                            sample_exact_mtp_delta_rejection_correction(
                                &logits_all,
                                ac,
                                vocab,
                                sampling,
                                pending[ac],
                                &mut state.rng,
                            )
                        })
                        .flatten();
                    let exact_residual_correction_applied = exact_rejection_correction.is_some();
                    mtp_timings.accept_wall_us = elapsed_us(accept_started);

                    let rollback_started = Instant::now();
                    let committed_len = token_offset + 1 + ac;
                    let trimmed = state.cache.trim_to(committed_len);
                    debug_assert!(trimmed, "MTP committed_len must not exceed cache seq_len");
                    if !trimmed {
                        // Ring slack is sized to absorb a full draft-depth
                        // rollback; a refusal means rejected-draft KV stays
                        // attendable. There is no cheap recovery at this point —
                        // surface it instead of silently degrading output.
                        tracing::error!(
                            committed_len,
                            "speculative rollback trim refused by rotated KV ring"
                        );
                    }

                    // Trim MTP KV cache: remove rejected draft entries.
                    let rejected_count = pending.len() - ac;
                    if rejected_count > 0 {
                        let new_mtp_len = state.mtp_decode_count.saturating_sub(rejected_count);
                        if let Some(ref mut c) = state.mtp_cache
                            && !c.trim_to(new_mtp_len)
                        {
                            tracing::warn!(
                                new_mtp_len,
                                "MTP head cache trim refused; draft quality may degrade"
                            );
                        }
                        state.mtp_decode_count = new_mtp_len;
                    }
                    mtp_timings.rollback_wall_us = elapsed_us(rollback_started);
                    // Multi-token verify advanced production KV outside the
                    // double-buffer direct pipeline; drop any pending direct
                    // token so empty-draft fallthrough cannot desync.
                    state.pending_direct = None;
                    let draft_hidden =
                        slice_post_norm_hidden(&post_norm_all, ac, self.mtp_draft_hidden_width());
                    let correction_argmax_tok = predicted.get(ac).copied().unwrap_or(0);
                    (
                        logits_all,
                        draft_hidden,
                        ac,
                        all_accepted,
                        exact_rejection_correction
                            .or(accept.rejection_correction)
                            .unwrap_or(correction_argmax_tok),
                        exact_residual_correction_applied,
                        predicted,
                    )
                }
            }
        };

        // A depth-D head chain has D new cache entries, but they represent the
        // committed main token followed by drafts 1..D-1. Keep the main entry,
        // remove every head-chained draft entry, then rebuild accepted drafts
        // from the verifier's target hidden rows. This matches the head's
        // training pairs and fixes the legacy one-position rollback shift.
        let pure_qwen_mtp_pending = !pending.is_empty()
            && state.mtp_pending_draft_sources.len() == pending.len()
            && state
                .mtp_pending_draft_sources
                .iter()
                .all(|source| *source == MtpDraftSource::Mtp);
        // Same "meaningfully evaluated" contract as the profitability policy:
        // an n-gram-stacked window never feeds the cost controller's
        // per-position acceptance EMAs (cascade rejects there are n-gram's).
        let pure_mtp_round = is_profitability_mtp_round(
            pending.len(),
            &state.mtp_pending_draft_sources,
            self.mtp_max_depth(),
        );
        // Exactly one decode step ran since the controller's previous decision:
        // a budget/EWMA bypass, think window, or n-gram fallback step in
        // between would make the closed window span foreign work.
        let cost_wall_valid = state.mtp_cost_steps_since_decision == 1;
        // The MTP cache owns the committed/speculative timeline. Do not infer
        // this rollback boundary from `mtp_decode_count`: confidence gating
        // and hybrid proposal paths can trim the cache independently.
        let refold_base_len = state
            .mtp_cache
            .as_ref()
            .and_then(|cache| cache.seq_len().checked_sub(pending.len().saturating_sub(1)));
        let mut committed_fold_lazy: Option<crate::mtp::MtpLazyDraft> = None;
        let mut committed_fold_depth = None;
        let refolded_history = if has_linear_attention
            && pure_qwen_mtp_pending
            && crate::fastpath::mtp_refold_accepted_history_enabled()
        {
            if let (Some(hidden), Some(head), Some(cache), Some(retain_len)) = (
                mtp_refold_hidden.as_ref(),
                self.weights.mtp.as_ref(),
                state.mtp_cache.as_mut(),
                refold_base_len,
            ) {
                let combined_fold = crate::fastpath::mtp_batched_committed_fold_enabled()
                    && sampling.temperature <= 0.0
                    && crate::fastpath::mtp_async_draft_enabled()
                    && qwen_linear_mtp_async_draft_allowed(
                        true,
                        crate::fastpath::qwen_linear_mtp_exact_enabled(),
                        crate::fastpath::mtp_linear_projected_replay_enabled(),
                    )
                    && !self.mtp_skip_state
                    && crate::mtp::mtp_draft_mode_from_env() == crate::mtp::MtpDraftMode::Greedy
                    && qwen_batched_committed_fold_gate_ok(
                        crate::mtp::mtp_draft_min_confidence_from_env(),
                        state.mtp_draft_gate_x1000,
                    );
                if combined_fold {
                    let (legacy_depth, _) = mtp_next_adaptive_depth(
                        state.mtp_adaptive_max_depth,
                        self.mtp_max_depth(),
                        pending.len(),
                        ewma_accept_count.unwrap_or(accept_count),
                        state.mtp_consecutive_misses,
                        false,
                        crate::fastpath::mtp_depth3_miss_backoff_enabled(),
                        crate::fastpath::mtp_depth3_hysteresis_enabled(),
                        Some(mtp_depth_observations_after_step(
                            &state.mtp_telemetry,
                            pending.len(),
                            accept_count,
                        )),
                    );
                    // The async fold drafts before `record_step`, so preview
                    // the cost controller's decision on a copy. Its decision
                    // reads only measurements recorded by earlier cycles, so
                    // the copy and the live controller below return the same
                    // depth and the debug_assert holds.
                    let mut preview = state.mtp_cost_depth;
                    let next_depth = mtp_cost_depth_cycle_depth(
                        &mut preview,
                        legacy_depth,
                        pending.len(),
                        ewma_accept_count.unwrap_or(accept_count),
                        pure_mtp_round,
                        cost_wall_valid,
                        crate::fastpath::mtp_fixed_draft_depth(),
                    );
                    committed_fold_depth = Some(next_depth);
                    let committed_hidden = slice(
                        hidden,
                        &[0, 0, 0],
                        &[1, (accept_count + 1) as i32, self.cfg.hidden_size as i32],
                        &[1, 1, 1],
                        None,
                    );
                    let mut committed = Vec::with_capacity(accept_count + 1);
                    committed.extend_from_slice(&pending[..accept_count]);
                    committed.push(correction_argmax_tok);
                    committed_fold_lazy = crate::mtp::mtp_refold_committed_draft_greedy_async(
                        &self.weights,
                        &self.cfg,
                        &committed_hidden,
                        &committed,
                        cache,
                        retain_len,
                        Some(next_depth),
                    );
                    committed_fold_lazy.is_some()
                } else {
                    let accepted_hidden = slice(
                        hidden,
                        &[0, 0, 0],
                        &[1, accept_count as i32, self.cfg.hidden_size as i32],
                        &[1, 1, 1],
                        None,
                    );
                    crate::mtp::mtp_refold_accepted_cache_kv_batched(
                        head,
                        &accepted_hidden,
                        &pending[..accept_count],
                        &self.weights,
                        cache,
                        &self.cfg,
                        retain_len,
                    )
                }
            } else {
                false
            }
        } else {
            false
        };
        if refolded_history {
            state.mtp_decode_count = state
                .mtp_cache
                .as_ref()
                .map(MlxKVCache::seq_len)
                .unwrap_or(0);
        } else if has_linear_attention && !all_accepted {
            // Legacy path: remove rejected cache entries using the historical
            // count contract. Retained as the default-off fallback while the
            // shifted committed-history route is being admitted.
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
        let exact_residual_required = !optimistic
            && !all_accepted
            && sampling.temperature > 0.0
            && proposal_law == MtpProposalLaw::DeterministicDelta;
        if exact_residual_required && !exact_residual_correction_applied {
            state.mtp_telemetry.record_direct_fallback();
        }
        let uses_exact_residual = exact_residual_required && exact_residual_correction_applied;
        let tail_tok = if uses_exact_residual {
            state.mtp_telemetry.residual_correction_tokens = state
                .mtp_telemetry
                .residual_correction_tokens
                .saturating_add(1);
            correction_argmax_tok
        } else {
            sample_logit_row(
                &logits_all,
                correction_argmax_tok,
                accept_count,
                vocab,
                sampling,
                &mut state.rng,
                &mut state.sampling_probs_buf,
                &mut state.sampling_logits_buf,
                &mut state.sampling_candidates_buf,
            )
        };
        mtp_timings.tail_sample_wall_us = elapsed_us(tail_sample_started);
        result.push(tail_tok);
        // Emission-accounting identity: every emitted token must be the skip
        // primary, an accepted draft, or the tail — the skip-state token-0
        // corruption entered the stream exactly through this seam, and the
        // telemetry mismatch (emitted vs accepted/drafted) was the tell.
        debug_assert_eq!(
            result.len(),
            usize::from(primary_tok_from_skip.is_some()) + accept_count + 1,
            "MTP emitted {} tokens but primary={:?}, accepted={}, +1 tail",
            result.len(),
            primary_tok_from_skip,
            accept_count
        );
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
        let (legacy_depth, conservative_depth) = mtp_next_adaptive_depth(
            state.mtp_adaptive_max_depth,
            mtp_max_depth,
            pending.len(),
            adaptive_depth_accept,
            state.mtp_consecutive_misses,
            // Early short-context only: stop-loss reject storms on tiny
            // gens (gen med killers) without aborting multi-token mid
            // long-ish short-ctx gens (smokef139 trial3 1.20→1.04).
            // Window 24 covers formal general short tails that still
            // stack low-accept cost after the first dozen tokens.
            self.gemma4_assistant_mtp.is_some()
                && state.cache.seq_len() < 512
                && state.generated_tokens.len() < 24,
            crate::fastpath::mtp_depth3_miss_backoff_enabled(),
            crate::fastpath::mtp_depth3_hysteresis_enabled(),
            (has_linear_attention && self.weights.mtp.is_some()).then_some((
                state.mtp_telemetry.drafted_by_depth[0],
                state.mtp_telemetry.accepted_by_depth[0],
                state.mtp_telemetry.drafted_by_depth[1],
                state.mtp_telemetry.accepted_by_depth[1],
            )),
        );
        let next_depth = mtp_cost_depth_cycle_depth(
            &mut state.mtp_cost_depth,
            legacy_depth,
            pending.len(),
            adaptive_depth_accept,
            pure_mtp_round,
            cost_wall_valid,
            crate::fastpath::mtp_fixed_draft_depth(),
        );
        state.mtp_cost_steps_since_decision = 0;
        if !state.mtp_cost_depth.parked() {
            state.mtp_adaptive_max_depth = next_depth;
        }
        if committed_fold_lazy.is_some() {
            debug_assert_eq!(committed_fold_depth, Some(next_depth));
        }
        // Park application: a parked controller latches the request onto direct
        // decode, mirroring the profitability latch. Park (like the other
        // bypasses) also suspends n-gram drafting for the remaining steps —
        // `MtpRequestRoute::DirectFallback` owns them, not the n-gram route.
        if state.mtp_cost_depth.parked() {
            state.mtp_bypassed = true;
            clear_pending_mtp_proposal(state);
        }
        // The controller's own depth decisions supersede the conservative
        // policy; a legacy classification that was never applied is not a
        // conservative-depth decision.
        if conservative_depth && !state.mtp_cost_depth.enabled() {
            state.mtp_telemetry.conservative_depth_decisions = state
                .mtp_telemetry
                .conservative_depth_decisions
                .saturating_add(1);
        }
        if adaptive_depth_accept == 0 && !pending.is_empty() {
            state.mtp_consecutive_misses = state.mtp_consecutive_misses.saturating_add(1);
        } else if adaptive_depth_accept > 0 {
            state.mtp_consecutive_misses = 0;
        }

        // Adaptive draft gate observe (default OFF; low-T auto only when allocated).
        if let Some(ref mut adaptive) = state.mtp_adaptive_gate {
            let mean_conf = if pending.is_empty() {
                0.0
            } else {
                // Approximate pre-gate confidence from pending log-probs when present.
                let confs: Vec<f32> = state
                    .mtp_pending_draft_log_probs
                    .iter()
                    .take(pending.len())
                    .map(|lp| lp.exp().clamp(0.0, 1.0))
                    .collect();
                if confs.is_empty() {
                    0.9
                } else {
                    confs.iter().sum::<f32>() / confs.len() as f32
                }
            };
            let recomputed = false; // detailed recompute flag is path-local; residual OFF by default
            let cfg = next_gate_config_from_env();
            let _ = observe_step(
                adaptive,
                AdaptiveStepSignals {
                    pre_gate_mean_conf: mean_conf,
                    gated_draft_len: pending.len(),
                    recomputed,
                    mtp_only_accept_rate_ewma: state.mtp_telemetry.mtp_only_accept_rate_ewma,
                    mtp_only_accept_rate_ewma_samples: state
                        .mtp_telemetry
                        .mtp_only_accept_rate_ewma_samples,
                    mtp_bypassed: state.mtp_bypassed,
                    adaptive_depth: state.mtp_adaptive_max_depth,
                    auto_optimistic_active: state.auto_optimistic_active,
                },
                &cfg,
            );
        }

        // ADR-020 short remaining-budget bypass: draft/verify fixed cost cannot
        // amortize when few tokens remain. Formal harnesses that must force MTP
        // set AX_MLX_MTP_MIN_REMAINING_TOKENS=0. Latched for the rest of the
        // request because remaining budget only decreases.
        if !state.mtp_bypassed {
            let min_remaining = mtp_min_remaining_tokens();
            if min_remaining > 0
                && let Some(request_ctx) = ctx
            {
                let remaining = request_ctx
                    .max_output_tokens
                    .saturating_sub(request_ctx.generated_len);
                if remaining < min_remaining {
                    state.mtp_bypassed = true;
                    state.mtp_telemetry.record_short_budget_bypass();
                    state.mtp_pending_draft.clear();
                    state.mtp_pending_draft_lazy = None;
                    state.mtp_pending_draft_log_probs.clear();
                    state.mtp_pending_draft_log_prob_temperature = None;
                    state.mtp_pending_draft_distributions.clear();
                    state.mtp_pending_draft_sources.clear();
                    state.mtp_skip_logits = None;
                    state.mtp_skip_argmax = None;
                    state.mtp_skip_hidden = None;
                }
            }
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
        // The cost controller owns the bypass decision while it is enabled:
        // its measured park latch replaces this acceptance-only stop-loss.
        if !state.mtp_bypassed
            && !state.mtp_cost_depth.enabled()
            && mtp_ewma_bypass_enabled()
            && state.mtp_telemetry.mtp_only_accept_rate_ewma_samples >= mtp_bypass_min_samples()
            && state.mtp_telemetry.mtp_only_accept_rate_ewma < mtp_bypass_threshold()
        {
            state.mtp_bypassed = true;
            state.mtp_pending_draft.clear();
            state.mtp_pending_draft_lazy = None;
            state.mtp_pending_draft_log_probs.clear();
            state.mtp_pending_draft_log_prob_temperature = None;
            state.mtp_pending_draft_distributions.clear();
            state.mtp_pending_draft_sources.clear();
            state.mtp_skip_logits = None;
            state.mtp_skip_argmax = None;
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
            // DI-DS-MTP residual: next draft (and the T locked into pending
            // log-probs) must use post-result think state. Accept rescale for
            // *this* step still uses the T carried from when pending was
            // drafted (`mtp_pending_draft_log_prob_temperature`); only the
            // new draft generation below must advance past `</think>` /
            // `<think>` boundaries crossed by `result`.
            let deepseek_next_draft_temperature = if self.weights.deepseek_v4_nextn.is_some() {
                Some(
                    crate::mtp::deepseek_v4_mtp_sample_and_log_temperature_from_env(
                        think_state_after_result,
                        sampling.temperature,
                    ),
                )
            } else {
                None
            };
            // Qwen draft branches below may overwrite this with the T actually
            // used for log-probs (often 1.0 on exact/gated greedy — not head 0.7).
            let mut next_draft_log_prob_temperature = deepseek_next_draft_temperature
                .unwrap_or(draft_log_prob_temperature_for_new_drafts);
            let mtp_post_think_guarded =
                self.cfg.think_start_token_id.is_some() && !think_state_after_result;
            // Pure-MTP override: explicit AX_MLX_MTP_DISABLE_NGRAM_STACKING=1
            // or the session-level --mlx-mtp-disable-ngram-stacking flag still
            // skips n-gram. Unset + exact Qwen linear MTP allows
            // stacking — official Qwen38 --full leaves the var unset and
            // general-long ignore_eos is a special-token loop n-gram can hit.
            let mut ngram_max = if (self.mtp_requested
                && crate::fastpath::qwen_linear_throughput_mtp_enabled())
                || !mtp_ngram_stacking_allowed(
                    mtp_ngram_stacking_env(),
                    crate::fastpath::qwen_linear_mtp_exact_enabled(),
                    self.disable_mtp_ngram_stacking,
                ) {
                0
            } else {
                mtp_ngram_stack_len(
                    adaptive_ngram_draft_len(has_linear_attention, state.ngram_posterior_mean()),
                    state.mtp_adaptive_max_depth,
                )
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
                min_support: mtp_ngram_min_support_for_exact(
                    crate::fastpath::qwen_linear_mtp_exact_enabled(),
                    mtp_ngram_min_support(),
                )
                .max(if mtp_post_think_guarded {
                    POST_THINK_MIN_NGRAM_SUPPORT
                } else {
                    1
                }),
                confidence_threshold: mtp_ngram_confidence_threshold(),
                adaptive_match_len: true,
                bypass_prompt_min_support: true,
                min_context_len: mtp_ngram_min_context_len_for_exact(
                    crate::fastpath::qwen_linear_mtp_exact_enabled(),
                    mtp_ngram_min_context_len(),
                ),
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
            let cycle_tok = if mtp_ngram_stacking_allowed(
                mtp_ngram_stacking_env(),
                crate::fastpath::qwen_linear_mtp_exact_enabled(),
                self.disable_mtp_ngram_stacking,
            ) {
                // `result` is only this step; the loop lives in generated_tokens.
                short_cycle_next_token_from_parts(&state.generated_tokens, &result)
            } else {
                None
            };
            let ngram_outcome = if let Some(tok) = cycle_tok {
                NgramDraftOutcome {
                    draft: vec![tok],
                    confidence: vec![1.0],
                    rejection: None,
                    requested_max_len: 1,
                }
            } else {
                ngram_outcome
            };
            if ngram_max > 0 || cycle_tok.is_some() {
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
                    && (self.weights.mtp.is_some()
                        || self.weights.glm_mtp.is_some()
                        || self.weights.deepseek_v4_nextn.is_some())
                {
                    let cache = state
                        .mtp_cache
                        .get_or_insert_with(|| self.new_mtp_draft_cache());
                    let mtp_draft_started = Instant::now();
                    // Share the pure-MTP gate (adaptive/optimistic/model default)
                    // so hybrid tails do not re-resolve a different confidence.
                    let (hybrid_gate, _src) = resolve_mtp_gate_from_env(
                        Some(sampling.temperature),
                        state.mtp_adaptive_gate.as_ref(),
                        mtp_optimistic_draft_min_confidence_override(),
                        if self.weights.mtp.is_some() {
                            self.mtp_model_policy.qwen_gate_default()
                        } else if self.weights.glm_mtp.is_some() {
                            self.mtp_model_policy.glm_gate_default()
                        } else {
                            None
                        },
                    );
                    let (tail, log_probs, distributions, added, _top2_margins) =
                        if self.weights.glm_mtp.is_some() {
                            glm_mtp_draft_tokens_after_forced_prefix(
                                &self.weights,
                                &self.cfg,
                                &draft_hidden,
                                tail_tok,
                                &draft,
                                cache,
                                mtp_tail_cap,
                                &mut state.rng,
                                Some(hybrid_gate),
                            )
                        } else if self.weights.deepseek_v4_nextn.is_some() {
                            // Same think-aware draft T as pure-MTP / accept rescale
                            // (post-result think state; see deepseek_next_draft_temperature).
                            let hybrid_draft_t = deepseek_next_draft_temperature
                                .expect("deepseek nextn implies next-draft temperature");
                            deepseek_v4_mtp_draft_tokens_after_forced_prefix(
                                &self.weights,
                                &self.cfg,
                                &draft_hidden,
                                tail_tok,
                                &draft,
                                cache,
                                mtp_tail_cap,
                                &mut state.rng,
                                Some(hybrid_gate),
                                hybrid_draft_t,
                            )
                        } else {
                            // DI-QW-MTP: lock accept T to the T used for Qwen
                            // hybrid-tail log-probs (gated greedy → 1.0).
                            if let Some(head) = self.weights.mtp.as_ref() {
                                next_draft_log_prob_temperature =
                                    crate::mtp::qwen_mtp_draft_log_prob_temperature_from_env(
                                        head.draft_sampling.temperature,
                                        hybrid_gate,
                                    );
                            }
                            mtp_draft_tokens_after_forced_prefix(
                                &self.weights,
                                &self.cfg,
                                &draft_hidden,
                                tail_tok,
                                &draft,
                                cache,
                                mtp_tail_cap,
                                &mut state.rng,
                                Some(hybrid_gate),
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
                    let preserve_cache = crate::fastpath::mtp_ngram_cache_preserved();
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
                    state.mtp_skip_argmax = None;
                    state.mtp_skip_hidden = None;
                    state.mtp_pending_draft_distributions.clear();
                    state.mtp_telemetry.record_ngram_stack_hit(ngram_len, true);
                    (draft, aligned_log_probs, sources)
                }
            } else {
                if self.weights.mtp.is_some() {
                    // MTP head forward path (RoPE managed internally via cache.seq_len()).
                    let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
                    let mtp_draft_started = Instant::now();
                    let (gate, src) = resolve_mtp_gate_from_env(
                        Some(sampling.temperature),
                        state.mtp_adaptive_gate.as_ref(),
                        mtp_optimistic_draft_min_confidence_override(),
                        self.mtp_model_policy.qwen_gate_default(),
                    );
                    state.mtp_draft_gate_x1000 = (gate.clamp(0.0, 1.0) * 1000.0) as u32;
                    state.mtp_draft_gate_source = src.route_code();
                    // Async draft (`AX_MLX_MTP_ASYNC_DRAFT`): only in a
                    // linear verifier regime where the synchronous greedy path computes no
                    // log-probs or distributions, so the deferred extraction
                    // reconstructs the identical pending state. Skip-state
                    // capture below keys off an empty pending draft, so the
                    // two features are mutually exclusive.
                    let async_qualified = crate::fastpath::mtp_async_draft_enabled()
                        && qwen_linear_mtp_async_draft_allowed(
                            self.cfg.linear_attention.is_some(),
                            crate::fastpath::qwen_linear_mtp_exact_enabled(),
                            crate::fastpath::mtp_linear_projected_replay_enabled(),
                        )
                        && !self.mtp_skip_state
                        && gate == 0.0
                        && crate::mtp::mtp_draft_mode_from_env()
                            != crate::mtp::MtpDraftMode::Stochastic;
                    if let Some(lazy) = committed_fold_lazy.take() {
                        state.mtp_pending_draft_distributions = Vec::new();
                        state.mtp_pending_draft_lazy = Some(lazy);
                        (Vec::new(), Vec::new(), Vec::new())
                    } else if async_qualified
                        && let Some(lazy) = crate::mtp::mtp_draft_tokens_greedy_async(
                            &self.weights,
                            &self.cfg,
                            &draft_hidden,
                            tail_tok,
                            cache,
                            Some(state.mtp_adaptive_max_depth),
                        )
                    {
                        mtp_timings.mtp_draft_wall_us = mtp_timings
                            .mtp_draft_wall_us
                            .saturating_add(elapsed_us(mtp_draft_started));
                        state.mtp_decode_count += lazy.tokens.len();
                        state.mtp_pending_draft_distributions = Vec::new();
                        state.mtp_pending_draft_lazy = Some(lazy);
                        (Vec::new(), Vec::new(), Vec::new())
                    } else {
                        // DI-QW-MTP: accept must use the T at which gated drafts
                        // recorded log-probs (gated greedy → 1.0, not head 0.7).
                        if let Some(head) = self.weights.mtp.as_ref() {
                            next_draft_log_prob_temperature =
                                crate::mtp::qwen_mtp_draft_log_prob_temperature_from_env(
                                    head.draft_sampling.temperature,
                                    gate,
                                );
                        }
                        let (draft, log_probs, distributions, added, _top2_margins) =
                            mtp_draft_tokens_gated(
                                &self.weights,
                                &self.cfg,
                                &draft_hidden,
                                tail_tok,
                                cache,
                                Some(state.mtp_adaptive_max_depth),
                                &mut state.rng,
                                gate,
                            );
                        mtp_timings.mtp_draft_wall_us = mtp_timings
                            .mtp_draft_wall_us
                            .saturating_add(elapsed_us(mtp_draft_started));
                        state.mtp_decode_count += added;
                        state.mtp_pending_draft_distributions = distributions;
                        let sources = vec![MtpDraftSource::Mtp; draft.len()];
                        (draft, log_probs, sources)
                    }
                } else if self.weights.glm_mtp.is_some() {
                    // GLM MTP head forward path (GLM MLA attention, shared_head logits).
                    let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
                    let mtp_draft_started = Instant::now();
                    let (gate, src) = resolve_mtp_gate_from_env(
                        Some(sampling.temperature),
                        state.mtp_adaptive_gate.as_ref(),
                        mtp_optimistic_draft_min_confidence_override(),
                        self.mtp_model_policy.glm_gate_default(),
                    );
                    state.mtp_draft_gate_x1000 = (gate.clamp(0.0, 1.0) * 1000.0) as u32;
                    state.mtp_draft_gate_source = src.route_code();
                    let (draft, log_probs, distributions, added, _top2_margins) =
                        glm_mtp_draft_tokens_gated(
                            &self.weights,
                            &self.cfg,
                            &draft_hidden,
                            tail_tok,
                            cache,
                            Some(state.mtp_adaptive_max_depth),
                            &mut state.rng,
                            gate,
                        );
                    mtp_timings.mtp_draft_wall_us = mtp_timings
                        .mtp_draft_wall_us
                        .saturating_add(elapsed_us(mtp_draft_started));
                    state.mtp_decode_count += added;
                    state.mtp_pending_draft_distributions = distributions;
                    let sources = vec![MtpDraftSource::Mtp; draft.len()];
                    (draft, log_probs, sources)
                } else if self.weights.deepseek_v4_nextn.is_some() {
                    // DeepSeek V4 nextn head (packed-hidden raw-path block).
                    let cache = state
                        .mtp_cache
                        .get_or_insert_with(|| self.new_mtp_draft_cache());
                    let mtp_draft_started = Instant::now();
                    let (gate, src) = resolve_mtp_gate_from_env(
                        Some(sampling.temperature),
                        state.mtp_adaptive_gate.as_ref(),
                        mtp_optimistic_draft_min_confidence_override(),
                        None,
                    );
                    state.mtp_draft_gate_x1000 = (gate.clamp(0.0, 1.0) * 1000.0) as u32;
                    state.mtp_draft_gate_source = src.route_code();
                    // Must match next_draft_log_prob_temperature / accept rescale
                    // on the following step (post-result think state).
                    let draft_temperature = deepseek_next_draft_temperature
                        .expect("deepseek nextn implies next-draft temperature");
                    let (draft, log_probs, distributions, added, _top2_margins) =
                        deepseek_v4_mtp_draft_tokens_gated(
                            &self.weights,
                            &self.cfg,
                            &draft_hidden,
                            tail_tok,
                            cache,
                            Some(state.mtp_adaptive_max_depth),
                            &mut state.rng,
                            gate,
                            draft_temperature,
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
            // Lock accept rescale to the T used when these log-probs were
            // written (DeepSeek post-result think T, or the Qwen gated draft
            // log-prob T — see next_draft_log_prob_temperature); a draft that
            // wrote none clears the temperature with the vector.
            state.mtp_pending_draft_log_prob_temperature = pending_draft_log_prob_temperature_for(
                &state.mtp_pending_draft_log_probs,
                next_draft_log_prob_temperature,
            );
            if state.mtp_pending_draft_log_probs.is_empty() {
                state.mtp_pending_draft_distributions.clear();
            }
            if state.mtp_pending_draft.is_empty() {
                state.mtp_pending_draft_sources.clear();
            }
            debug_assert_eq!(
                state.mtp_pending_draft.len(),
                state.mtp_pending_draft_sources.len(),
                "pending draft tokens and their sources must stay aligned \
                 (a phantom token without a source bypasses accept accounting)"
            );
            // Capture skip-state only when the next step will have no pending draft,
            // making `can_skip` true.  When pending is non-empty (the common case)
            // async_eval + slice work here is never consumed — so skip it entirely.
            // Includes Gemma 4 assistant MTP (vLLM/Lightning always-advance pattern).
            let can_capture_skip = self.mtp_skip_state
                && !forced_greedy_replay
                && state.mtp_pending_draft.is_empty()
                && (self.weights.mtp.is_some()
                    || self.weights.glm_mtp.is_some()
                    || self.gemma4_assistant_mtp.is_some());
            if can_capture_skip {
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
                // Capture the row argmax alongside the logits: greedy skip
                // cycles commit it as the primary token, and the shared
                // async_eval batch makes it effectively free here.
                let sl_argmax = argmax(&sl, None);
                mlx_sys::async_eval(&[&sl, &sl_argmax, &draft_hidden]);
                state.mtp_skip_logits = Some(sl);
                state.mtp_skip_argmax = Some(sl_argmax);
                state.mtp_skip_hidden = Some(draft_hidden);
            }
            mtp_timings.draft_wall_us = elapsed_us(draft_started);
        } // end of !mtp_bypassed draft generation block

        // Compare complete MTP-round wall time with the real target probes.
        // A losing decision latches only for this request; committed cache
        // state is intentionally retained.
        if profitability_mtp_round
            && !state.mtp_bypassed
            && state
                .mtp_profitability
                .record_mtp_round(elapsed_us(profitability_round_started), result.len())
        {
            state.mtp_bypassed = true;
            clear_pending_mtp_proposal(state);
        }
        state
            .mtp_telemetry
            .record_profitability_snapshot(state.mtp_profitability.snapshot());
        state
            .mtp_telemetry
            .record_cost_depth_snapshot(state.mtp_cost_depth.snapshot());
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
        if std::env::var_os("AX_MLX_MTP_DEBUG_FINAL_SUMMARY").is_some()
            && ctx.is_some_and(|ctx| {
                ctx.generated_len
                    .saturating_add(saturating_u32(result.len()))
                    >= ctx.max_output_tokens
            })
        {
            let telemetry = &state.mtp_telemetry;
            eprintln!(
                "AX_MTP_FINAL steps={} drafted={} accepted={} emitted={} verify_tokens={} full={} partial={} miss={} clone_us={} verify_forward_us={} verify_eval_us={} accept_us={} rollback_us={} tail_us={} draft_us={} softmax_us={} depth_accepted={:?} depth_drafted={:?}",
                telemetry.decode_steps,
                telemetry.draft_tokens,
                telemetry.accepted_tokens,
                telemetry.emitted_tokens,
                telemetry.verify_tokens,
                telemetry.full_accept_steps,
                telemetry.partial_reject_steps,
                telemetry.complete_miss_steps,
                telemetry.cache_clone_wall_us,
                telemetry.verify_forward_wall_us,
                telemetry.verify_eval_wall_us,
                telemetry.accept_wall_us,
                telemetry.rollback_wall_us,
                telemetry.tail_sample_wall_us,
                telemetry.draft_wall_us,
                telemetry.target_softmax_wall_us,
                telemetry.accepted_by_depth,
                telemetry.drafted_by_depth,
            );
        }

        result
    }

    fn initialize_generation_state(
        &self,
        state: &mut RequestState,
        max_output: u32,
        prefill_output_token: Option<u32>,
        is_greedy: bool,
        sampling: MlxSamplingParams,
        // (max_think_tokens, answer_reserve_tokens) from the request context.
        think_budget_knobs: (Option<u32>, Option<u32>),
    ) {
        // When MTP is active, use a wider prompt window (NGRAM_MTP_PROMPT_FEED_MAX)
        // so real-code bigrams are seeded before the first decode step. Without
        // MTP, keep the conservative 64-token guard to avoid random-token false
        // positives that would disable n-gram for the first 16 decode steps.
        let has_mtp = self.has_generic_mtp();
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
        state.mtp_pending_draft_lazy = None;
        state.mtp_pending_draft_log_probs.clear();
        state.mtp_pending_draft_log_prob_temperature = None;
        state.mtp_pending_draft_distributions.clear();
        state.mtp_pending_draft_sources.clear();
        state.mtp_adaptive_max_depth =
            mtp_initial_adaptive_depth(&self.cfg.model_family, self.mtp_max_depth());
        state.mtp_skip_logits = None;
        state.mtp_skip_argmax = None;
        state.mtp_skip_hidden = None;
        state.mtp_decode_count = 0;
        state.mtp_bypassed = false;
        state.mtp_suspended_for_batched_decode = false;
        // Cost-model depth controller: opt-in, Qwen linear MTP head only. The
        // fixed-depth harness contract preempts it outright, and every other
        // head keeps the legacy streak controller.
        let cost_depth_config = mtp_cost_depth_config_from_env();
        let cost_depth_enabled = cost_depth_config.enabled
            && has_linear_attention
            && self.weights.mtp.is_some()
            && crate::fastpath::mtp_fixed_draft_depth().is_none();
        state.mtp_cost_depth.reset(
            cost_depth_enabled,
            cost_depth_config,
            mtp_cost_depth_width(qwen_linear_max_verify_drafts(), self.mtp_max_depth()),
        );
        state.mtp_cost_steps_since_decision = 0;
        let profitability_config = mtp_profitability_config_from_env();
        let profitability_max_tokens_per_round =
            if crate::fastpath::mtp_profitability_throughput_enabled()
                && crate::fastpath::qwen_linear_throughput_mtp_enabled()
            {
                // Drafts plus the committed token: 4 at the default depth 3.
                // Family-scoped so a widened env cannot enlarge a non-Qwen model.
                crate::fastpath::qwen_linear_mtp_max_verify_seq_for_family(
                    crate::fastpath::qwen_linear_throughput_mtp_enabled(),
                    crate::fastpath::qwen_linear_throughput_mtp_depth(),
                    crate::fastpath::qwen_linear_throughput_family(&self.cfg.model_family),
                ) as u32
            } else {
                2
            };
        let profitability_eligible = MtpProfitabilityEligibility {
            mtp_requested: self.mtp_requested,
            exact_qwen_linear: self.qwen_linear_mtp_exact_enabled,
            relaxed_projected: crate::fastpath::mtp_relaxed_target_verify_enabled()
                && crate::fastpath::mtp_linear_projected_replay_enabled(),
            has_linear_attention,
            has_qwen_mtp: self.weights.mtp.is_some(),
            depth_one: self.mtp_max_depth() == 1,
            throughput_depth3: crate::fastpath::mtp_profitability_throughput_enabled()
                && crate::fastpath::qwen_linear_throughput_mtp_enabled()
                && self.mtp_max_depth() > 0
                && self.mtp_max_depth()
                    <= crate::fastpath::qwen_linear_mtp_max_verify_drafts_for_family(
                        crate::fastpath::qwen_linear_throughput_mtp_enabled(),
                        crate::fastpath::qwen_linear_throughput_mtp_depth(),
                        crate::fastpath::qwen_linear_throughput_family(&self.cfg.model_family),
                    ),
            dense_lm_head: !self.weights.lm_head.is_quantized(),
            greedy: is_greedy,
            skip_state_disabled: !self.mtp_skip_state,
            optimistic_disabled: !self.mtp_optimistic && !mtp_auto_optimistic_enabled_from_env(),
            automatic_bypass_allowed: mtp_bypass_threshold() > 0.0,
            output_budget_sufficient: profitability_config.has_observation_budget(
                max_output,
                mtp_min_remaining_tokens(),
                profitability_max_tokens_per_round,
            ),
        }
        .eligible()
            // While the cost controller owns depth decisions it also owns the
            // direct probe and the bypass decision; suppress the profitability
            // policy at the call site so the two never probe twice.
            && !cost_depth_enabled;
        state
            .mtp_profitability
            .reset(profitability_eligible, profitability_config);
        state
            .mtp_telemetry
            .record_profitability_snapshot(state.mtp_profitability.snapshot());
        state.gemma_mtp_cycle_latched = false;
        // Adaptive gate: allocate only for low-T auto when flag is on (A.0b).
        state.mtp_adaptive_gate = mtp_adaptive_maybe_init(
            adaptive_gate_enabled_from_env(),
            speculation_profile_from_env(),
            Some(sampling.temperature),
        );
        state.mtp_draft_gate_x1000 = 0;
        state.mtp_draft_gate_source = 0;
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
        // Do not `take()` the prefill hidden until the matching drafter is
        // confirmed present: a naive `(take(), mtp.as_ref())` pair would
        // discard DeepSeek packed history when only `deepseek_v4_nextn` is set.
        if self.weights.mtp.is_some()
            && let Some(prefill_hidden) = state.mtp_prefill_hidden.take()
        {
            let head = self
                .weights
                .mtp
                .as_ref()
                .expect("mtp head present after is_some guard");
            let history_tokens = std::mem::take(&mut state.mtp_prefill_history_tokens);
            let cache = state.mtp_cache.get_or_insert_with(|| MlxKVCache::new(1));
            let available_rows = prefill_hidden
                .shape()
                .get(1)
                .copied()
                .unwrap_or_default()
                .max(0) as usize;
            let total = available_rows.min(history_tokens.len());
            let cap = crate::fastpath::mtp_warmup_cap();
            let warmup_len = if cap > 0 { total.min(cap) } else { total };
            let start_offset = total.saturating_sub(warmup_len);
            // Absolute RoPE base: final-chunk history is only the last
            // `available_rows` of the main prompt. Multi-chunk prefill must
            // not warm at relative indices 0..C or long-prompt MTP RoPE is wrong.
            let rope_start =
                mtp_warmup_absolute_rope_start(state.cache.seq_len(), available_rows, start_offset);
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
                    rope_start,
                );
                let kv_refs = cache.collect_eval_refs();
                mlx_sys::eval(&kv_refs);
                clear_cache();
                state.mtp_decode_count = warmup_len;
                // Physical entries = warmup_len; next absolute pos = abs_base + total.
                if let Some(ref mut c) = state.mtp_cache {
                    c.rope_offset = rope_start;
                }
            }
        } else if self.weights.deepseek_v4_nextn.is_some()
            && let Some(prefill_packed) = state.mtp_prefill_hidden.take()
        {
            // DeepSeek V4 nextn: packed residual rows + nextn-layer warmup.
            // Without this the predictor starts with a cold KV and acceptance
            // collapses (Qwen path above already warms its head).
            let nextn = self
                .weights
                .deepseek_v4_nextn
                .as_ref()
                .expect("deepseek nextn present after is_some guard");
            let history_tokens = std::mem::take(&mut state.mtp_prefill_history_tokens);
            let cache = state
                .mtp_cache
                .get_or_insert_with(|| self.new_mtp_draft_cache());
            let available_rows = prefill_packed
                .shape()
                .get(1)
                .copied()
                .unwrap_or_default()
                .max(0) as usize;
            let total = available_rows.min(history_tokens.len());
            let cap = crate::fastpath::mtp_warmup_cap();
            let warmup_len = if cap > 0 { total.min(cap) } else { total };
            let start_offset = total.saturating_sub(warmup_len);
            let rope_start =
                mtp_warmup_absolute_rope_start(state.cache.seq_len(), available_rows, start_offset);
            if warmup_len > 0 {
                let width = prefill_packed.shape().get(2).copied().unwrap_or(0);
                let warmup_packed = slice(
                    &prefill_packed,
                    &[0, start_offset as i32, 0],
                    &[1, total as i32, width],
                    &[1, 1, 1],
                    None,
                );
                deepseek_v4_mtp_warmup_cache(
                    nextn,
                    &warmup_packed,
                    &history_tokens[start_offset..total],
                    &self.weights,
                    cache,
                    &self.cfg,
                    rope_start,
                );
                let kv_refs = cache.collect_eval_refs();
                mlx_sys::eval(&kv_refs);
                clear_cache();
                state.mtp_decode_count = warmup_len;
                if let Some(ref mut c) = state.mtp_cache {
                    c.rope_offset = rope_start;
                }
            }
        } else {
            // Unconsumed history for non-MTP families.
            state.mtp_prefill_hidden = None;
            state.mtp_prefill_history_tokens.clear();
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

        // Latch the rotating-sliding-KV decision now that the per-request
        // n-gram gate is known; `run()` re-applies the latch to the cache at
        // the top of every subsequent step, so the per-run session default
        // can never clobber it after a ring has converted. Three classes:
        //
        //  - Greedy direct sessions (session-level flag): pure window-sized
        //    rings, mask-free single-token SDPA — nothing can roll the cache
        //    back.
        //  - Greedy requests with n-gram disabled under a sticky reason and
        //    no MTP: same pure rings (the 38d8c70c class).
        //  - Every other non-MTP request on standard-family sliding models
        //    (n-gram active, and sampled requests regardless of their n-gram
        //    gate): bounded-rollback rings (`window + slack` slots plus a
        //    slot-validity mask). Every rollback source for this class is an
        //    n-gram verify `trim_to`, bounded by `MAX_DRAFT_LEN` under both
        //    greedy accept and rejection sampling, which the slack absorbs —
        //    rolled-back tokens rewrite into their own `t % capacity` slots.
        //
        // Gemma4 assistant-MTP requests join the bounded class with a
        // widened slack: every assistant rollback is a `state.cache.trim_to`
        // bounded by the pending draft (assistant depth + stacked n-gram),
        // and the drafter reads rotated target KV through the ring-aware
        // `peek_layer_kv` + slot-validity mask. qwen/GLM MTP heads stay
        // ring-excluded (fail-closed; their models have no sliding windows).
        // Rotation bounds every sliding layer's backing store — the dominant
        // KV memory for long-context requests on ~5:1 sliding:global models.
        // This init runs only once the full prompt is prefilled
        // (`prefill_completes_prompt` gates every call site), so all
        // post-latch appends are decode-sized (1 for direct steps, ≤ slack
        // for speculative verifies) and the predicate is deterministic per
        // request, making re-runs idempotent.
        // Gemma assistant: always use bounded ring slack so MTP-off (AX_NO_SPEC)
        // and MTP-on share KV geometry. Pure rings panic on multi-token ordered
        // append; mismatched pure-vs-bounded rings also drift greedy A/B.
        let mtp_ring_slack = if !has_mtp {
            Some(0)
        } else if self.gemma4_assistant_mtp.is_some()
            && self.weights.mtp.is_none()
            && self.weights.glm_mtp.is_none()
            && self.weights.deepseek_v4_nextn.is_none()
        {
            // Verify width = 1 (primary) + pending; pending is capped by the
            // assistant depth plus any stacked n-gram draft tokens.
            Some(self.mtp_max_depth() + crate::ngram_accel::MAX_DRAFT_LEN + 1)
        } else {
            None
        };
        let rotating_latch = request_rotating_sliding_slack(
            self.rotating_sliding_decode,
            crate::fastpath::rotating_sliding_decode_enabled(),
            crate::fastpath::rotating_bounded_rollback_enabled(),
            rotating_bounded_family_eligible(&self.cfg),
            state.ngram_acceleration_disabled_for_request,
            state.ngram_request_disable_reason,
            mtp_ring_slack,
            is_greedy,
        );
        state.rotating_sliding_latch =
            Some((rotating_latch.is_some(), rotating_latch.unwrap_or(0)));
        state
            .cache
            .set_rotating_sliding_decode(rotating_latch.is_some());
        state
            .cache
            .set_rotating_sliding_slack(rotating_latch.unwrap_or(0));

        let approximate_profile = mtp_optimistic_allowed(
            self.weights.glm_mtp.is_some() || self.weights.deepseek_v4_nextn.is_some(),
        ) && (self.mtp_optimistic
            || mtp_auto_optimistic_enabled_from_env());
        let exact_supported = mtp_exact_sampling_supported(sampling, self.mtp_target_softmax_topk);
        let mtp_uses_direct_pipeline = mtp_fallback_primes_direct_pipeline(
            mtp_request_route(
                self.has_generic_mtp(),
                self.mtp_requested,
                exact_supported,
                approximate_profile,
                false,
                sampling.uses_logits_processors(),
            ),
            is_greedy,
            sampling.uses_logits_processors(),
        );

        // Mirror mlx_lm.generate_step's first-yield boundary for the direct
        // greedy baseline: before the first token is yielded, mlx_lm already
        // builds and submits the next `_step(y)` via async_eval. Prime AX's
        // direct pipeline at the same prefill/first-token boundary so the
        // measured generation interval starts with a pending token instead of
        // paying one decode-step bootstrap.
        //
        // Pure session-direct also primes when temperature is already 0 even
        // if the engine's deterministic-argmax bit was not latched (bench
        // requests often omit `deterministic: true` and inherit a non-det
        // session default while still sending temperature=0).
        //
        // Sequential-oracle mode retains the direct-pipeline prime. Normal
        // assistant-MTP starts with an S=1 target step so it can retain the
        // post-norm hidden that seeds the first real draft.
        // MoE long-prompt: prime double-buffer so MTP-off matches empty pure-
        // direct. Never prime short MTP-on (smokef121 first_diff@1).
        let gemma_exact_direct_bootstrap = self.gemma4_assistant_mtp.is_some()
            && self.weights.deepseek_v4_nextn.is_none()
            && is_greedy
            && (crate::fastpath::gemma4_assistant_mtp_sequential_oracle_enabled()
                || (self.cfg.moe_expert_count > 0
                    && state.cache.seq_len() >= 512
                    && !gemma4_moe_long_mt_enabled()));
        // The LONG_MT identity route deliberately uses materialized S=1 for
        // MTP-off and an S=1 target step to seed MTP-on. Priming the lazy
        // double buffer here writes that bootstrap token a second time and
        // produces the proven 74418 self-loop.
        let gemma_moe_long_mt_singleton = self.gemma4_assistant_mtp.is_some()
            && self.cfg.moe_expert_count > 0
            && self.weights.deepseek_v4_nextn.is_none()
            && is_greedy
            && state.cache.seq_len() >= 512
            && gemma4_moe_long_mt_enabled();
        // A live Flash Next cursor — or a cursor restored from a prefix-cache
        // sidecar and still pending install — must see the first decode step:
        // a primed lazy direct token would occupy that position and strand the
        // cursor. Direct greedy prefill therefore snapshots at prompt_len+1;
        // MTP-with-cursor stays at prompt_len.
        let flash_next_cursor_owns_decode = self.flash_next_mtp_session()
            && (state.flash_next_mtp.cursor.is_some()
                || state.flash_next_mtp.pending_restored_cursor.is_some())
            && is_greedy
            && !sampling.uses_logits_processors();
        // A hard think-budget close due at the first token must not prime the
        // double buffer: priming commits the prefill token that the
        // post-decode override is about to replace with the close token, and
        // the commit cannot be rolled back.
        let think_close_due_at_first_token = think_budget_close_decision(
            state.ngram_in_think,
            self.cfg.think_end_token_id,
            max_output,
            state.think_emitted_tokens,
            think_budget_knobs.0,
            think_budget_knobs.1,
            false,
        );
        if (should_bootstrap_direct_pipeline(
            self.disable_ngram_acceleration,
            state.ngram_acceleration_disabled_for_request,
            self.has_generic_mtp(),
            mtp_uses_direct_pipeline,
            self.mtp_requested,
        ) || gemma_exact_direct_bootstrap)
            && !think_close_due_at_first_token
            && !gemma_moe_long_mt_singleton
            && !flash_next_cursor_owns_decode
            && (is_greedy || (self.disable_ngram_acceleration && sampling.temperature <= 0.0))
            && max_output > 1
            && let Some(prefill_tok) = prefill_output_token
        {
            let bootstrap_started = Instant::now();
            let bootstrap_token =
                start_direct_pipeline(&self.cfg, &self.weights, prefill_tok, &mut state.cache);
            state.pending_direct = Some(bootstrap_token);
            // The first generated token was produced at the prefill boundary.
            // Count it so the periodic cleanup cadence remains aligned with
            // mlx_lm even when E-series models defer the immediate cleanup.
            state.direct_pipeline_emitted_tokens = 1;
            state
                .decode_telemetry
                .record_direct_bootstrap(elapsed_us(bootstrap_started));
        }
    }

    /// Run one model decode step, updating the n-gram accept-rate gate.
    #[allow(clippy::too_many_arguments)]
    fn run_model_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        sampling: MlxSamplingParams,
        is_greedy: bool,
        final_by_max_output: bool,
        ctx: Option<&RunnerRequestContext>,
        think_hard_close_due: bool,
    ) -> Vec<u32> {
        // Think soft-close window: bypass MTP / n-gram speculation so every
        // step decides one token against materialized logits where the rank
        // probe can fire. A due hard close takes the same route: the
        // post-decode override swaps the sampled token for the close token,
        // which is KV-exact only on this single-token path (the direct
        // pipeline and speculative routes commit emitted tokens in-step).
        // MTP session state resets exactly like the
        // DirectFallback arm; a pending direct-pipeline token drains first
        // (already committed — the probe applies from the next step).
        if state.think_soft_close_armed || think_hard_close_due {
            state.mtp_pending_draft.clear();
            state.mtp_pending_draft_lazy = None;
            state.mtp_pending_draft_log_probs.clear();
            state.mtp_pending_draft_log_prob_temperature = None;
            state.mtp_pending_draft_distributions.clear();
            state.mtp_pending_draft_sources.clear();
            // Skip-state logits/hidden were captured against the pre-close
            // context; a later unarmed step would consume them with an empty
            // pending draft and emit the pre-close prediction a position late.
            state.mtp_skip_logits = None;
            state.mtp_skip_argmax = None;
            state.mtp_skip_hidden = None;
            state.mtp_decode_count = 0;
            if let Some(cache) = state.mtp_cache.as_mut() {
                cache.reset();
            }
            if state.pending_direct.is_some() {
                return self.finish_pending_direct_for_ngram_transition(state);
            }
            return self.run_single_decode(state, last_token, sampling);
        }

        let has_linear_attention = self.cfg.linear_attention.is_some();

        // MTP model-based speculative decode: checked first so that MTP can
        // activate even when n-gram is disabled (linear-attention models set
        // ngram_acceleration_disabled_for_request, which run_non_ngram_decode
        // intercepts before we'd ever reach MTP).  Repetition-penalty sampling
        // is incompatible with speculative decode and is excluded.
        // Exact MTP supports deterministic-delta drafts with greedy or filtered
        // target sampling. Unsupported proposal/filter combinations fail closed
        // to direct; optimistic verification remains an explicit approximation.
        // The per-request bypass short-circuits MTP when acceptance shows it is
        // not paying for itself. A request transferred into direct tensor
        // batching is likewise direct-only: if scheduler deferral writes it
        // back and it resumes alone, it must not rebuild speculative state.
        let approximate_profile = mtp_optimistic_allowed(
            self.weights.glm_mtp.is_some() || self.weights.deepseek_v4_nextn.is_some(),
        ) && (self.mtp_optimistic
            || mtp_auto_optimistic_enabled_from_env());
        let exact_supported = state.mtp_pending_draft_distributions.is_empty()
            && mtp_exact_sampling_supported(sampling, self.mtp_target_softmax_topk);
        let mtp_direct_only = state.mtp_bypassed || state.mtp_suspended_for_batched_decode;
        match mtp_request_route(
            self.has_generic_mtp(),
            self.mtp_requested,
            exact_supported,
            approximate_profile,
            mtp_direct_only,
            sampling.uses_logits_processors(),
        ) {
            MtpRequestRoute::DirectFallback => {
                // Mixed batches keep the outer exact-verifier scope enabled
                // for sibling MTP requests. This row has selected direct
                // fallback, so restore the singleton kernel contract locally.
                let _direct_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
                state.mtp_telemetry.record_direct_fallback();
                state.mtp_pending_draft.clear();
                state.mtp_pending_draft_lazy = None;
                state.mtp_pending_draft_log_probs.clear();
                state.mtp_pending_draft_log_prob_temperature = None;
                state.mtp_pending_draft_distributions.clear();
                state.mtp_pending_draft_sources.clear();
                state.mtp_decode_count = 0;
                if let Some(cache) = state.mtp_cache.as_mut() {
                    cache.reset();
                }
                if is_greedy && !sampling.uses_logits_processors() {
                    return vec![self.run_direct_pipeline_decode(
                        state,
                        last_token,
                        final_by_max_output,
                        false,
                    )];
                }
                return self.run_single_decode(state, last_token, sampling);
            }
            MtpRequestRoute::StrictMtp => {
                return self.run_mtp_decode(state, last_token, sampling, ctx);
            }
            MtpRequestRoute::Other => {}
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
            // the n-gram steps that just ran advanced cache.seq_len()
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

fn gemma4_moe_long_mt_enabled() -> bool {
    crate::fastpath::gemma4_moe_long_mt_enabled()
}

/// Compute the think-block state after observing a sequence of tokens, without
/// mutating any state.  Returns the updated `in_think` flag.
///
/// Used in `run_mtp_decode` to peek ahead at result tokens before deciding
/// whether the NEXT draft step should be gated by think-block state.
/// Number of verify tokens (primary + accepted drafts) DeepSeek V4 MTP must
/// commit into production KV after acceptance.
///
/// `verify_input = [last_token] ++ pending_drafts`. The primary is not yet in
/// the production cache at `token_offset`, so even a full draft reject
/// (`accept_count == 0`) still commits length 1. Returning 0 here was a real
/// bug: leaving production unadvanced dropped the primary token's KV write and
/// desynced the next decode step.
pub(super) fn deepseek_v4_mtp_committed_verify_len(
    accept_count: usize,
    pending_len: usize,
) -> usize {
    1 + accept_count.min(pending_len)
}

/// Absolute RoPE start for MTP prefill warmup when history is only the last
/// `available_rows` of a multi-chunk prompt of length `main_seq_len`.
///
/// First warmed row sits at absolute position
/// `main_seq_len - available_rows + start_offset` (not the relative
/// `start_offset` alone — that mis-positions long multi-chunk prompts).
pub(super) fn mtp_warmup_absolute_rope_start(
    main_seq_len: usize,
    available_rows: usize,
    start_offset: usize,
) -> usize {
    let abs_base = main_seq_len.saturating_sub(available_rows);
    abs_base.saturating_add(start_offset)
}

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

/// Think-budget controllers (the forced think-close override, the soft-close
/// rank probe, and `think_emitted_tokens` accounting) are enforced only on
/// the per-item decode path. Grouped routes — tensor batching and row-exact
/// coalescing — mirror the stop-detection tail but never run that controller,
/// so a request carrying a think budget must stay on the per-item path.
fn think_budget_controller_active(
    max_think_tokens: Option<u32>,
    answer_reserve_tokens: Option<u32>,
) -> bool {
    max_think_tokens.is_some() || answer_reserve_tokens.is_some()
}

/// Decide whether the thinking-budget controller must force the think-close
/// token at this step. Hard close: the answer reserve (or the per-request
/// think budget) is exhausted. Soft close: the loop detector fired while
/// still inside the think block (a stuck reasoning trace). Budget edges are
/// decidable from pre-step state, so `decode_one` routes a due step through
/// the single-token path (as with the armed soft-close probe): the
/// post-decode override then swaps the sampled token for the close token,
/// which is KV-exact only when the step commits nothing beyond its feed
/// token — the direct pipeline and speculative routes commit their emitted
/// tokens in-step, and linear/recurrent state cannot roll back. The
/// post-decode override remains for loop-detected closes; those keep their
/// stop reason and end the request, so the stale cache is discarded.
pub(crate) fn think_budget_close_decision(
    in_think: bool,
    think_end_token_id: Option<u32>,
    remaining_budget: u32,
    think_emitted_tokens: u32,
    max_think_tokens: Option<u32>,
    answer_reserve_tokens: Option<u32>,
    loop_detected: bool,
) -> bool {
    if !in_think || think_end_token_id.is_none() {
        return false;
    }
    if max_think_tokens.is_some_and(|cap| think_emitted_tokens >= cap) {
        return true;
    }
    if answer_reserve_tokens.is_some_and(|reserve| remaining_budget <= reserve) {
        return true;
    }
    loop_detected
}

/// Apply the thinking-budget controller override to a decoded step (ds4-style
/// soft+hard close). When [`think_budget_close_decision`] is due on a step
/// that would otherwise continue — or on a loop-detected step — the step
/// emits only the think-close token. Budget-edge closes carry `None` as the
/// stop reason: generation continues so the remaining output budget funds the
/// answer (ds4 hard-limit semantics), and the regular max-output path
/// terminates the request. A loop-detected close keeps its stop reason so a
/// stuck reasoning trace still ends the request after the forced close.
/// Returns `None` when no override applies.
#[allow(clippy::too_many_arguments)]
pub(crate) fn think_budget_close_override(
    stop_reason: Option<StopReason>,
    sampled_nonempty: bool,
    in_think: bool,
    think_end_token_id: Option<u32>,
    remaining_budget: u32,
    think_emitted_tokens: u32,
    max_think_tokens: Option<u32>,
    answer_reserve_tokens: Option<u32>,
) -> Option<(Vec<u32>, Option<StopReason>)> {
    if !sampled_nonempty {
        return None;
    }
    let loop_detected = matches!(stop_reason, Some(StopReason::LoopDetected));
    if !(stop_reason.is_none() || loop_detected) {
        return None;
    }
    if !think_budget_close_decision(
        in_think,
        think_end_token_id,
        remaining_budget,
        think_emitted_tokens,
        max_think_tokens,
        answer_reserve_tokens,
        loop_detected,
    ) {
        return None;
    }
    Some((
        vec![think_end_token_id.expect("checked in decision")],
        stop_reason,
    ))
}

/// Think soft-close probe window ahead of the hard budget edge (ds4:
/// `soft_limit_reply_budget = 1024` ahead of `hard_limit_reply_budget`).
pub(crate) const THINK_SOFT_CLOSE_WINDOW_TOKENS: u32 = 1024;

/// Decide whether the think soft-close rank probe is armed for this step.
/// Armed only inside an open think block with a budget controller active,
/// before the hard close is due, once generation nears the hard edge:
/// within `THINK_SOFT_CLOSE_WINDOW_TOKENS` of the answer reserve, or of the
/// per-request think cap. While armed, decode routes through a
/// logits-materializing single decode that emits the think-close token
/// early when the model itself ranks it in its top-3.
pub(crate) fn think_soft_close_armed(
    in_think: bool,
    think_end_token_id: Option<u32>,
    remaining_budget: u32,
    think_emitted_tokens: u32,
    max_think_tokens: Option<u32>,
    answer_reserve_tokens: Option<u32>,
) -> bool {
    if !crate::fastpath::think_soft_close_enabled() || !in_think || think_end_token_id.is_none() {
        return false;
    }
    if max_think_tokens.is_none() && answer_reserve_tokens.is_none() {
        return false;
    }
    // Hard close already due: the post-decode budget hook force-closes this
    // step; probing would be wasted work.
    if think_budget_close_decision(
        in_think,
        think_end_token_id,
        remaining_budget,
        think_emitted_tokens,
        max_think_tokens,
        answer_reserve_tokens,
        false,
    ) {
        return false;
    }
    if answer_reserve_tokens.is_some_and(|reserve| {
        remaining_budget <= reserve.saturating_add(THINK_SOFT_CLOSE_WINDOW_TOKENS)
    }) {
        return true;
    }
    max_think_tokens.is_some_and(|cap| {
        think_emitted_tokens.saturating_add(THINK_SOFT_CLOSE_WINDOW_TOKENS) >= cap
    })
}

#[cfg(test)]
mod think_budget_controller_active_tests {
    use super::think_budget_controller_active;

    #[test]
    fn active_with_either_budget_knob() {
        assert!(!think_budget_controller_active(None, None));
        assert!(think_budget_controller_active(Some(4096), None));
        assert!(think_budget_controller_active(None, Some(512)));
        assert!(think_budget_controller_active(Some(4096), Some(512)));
        // An explicit zero cap is still an active controller.
        assert!(think_budget_controller_active(Some(0), None));
    }
}

#[cfg(test)]
mod think_soft_close_armed_tests {
    use super::{THINK_SOFT_CLOSE_WINDOW_TOKENS, think_soft_close_armed};

    #[test]
    fn inactive_without_controller_or_think_state() {
        // No controller knobs set.
        assert!(!think_soft_close_armed(true, Some(5), 100, 10, None, None));
        // Outside think.
        assert!(!think_soft_close_armed(
            false,
            Some(5),
            100,
            10,
            None,
            Some(512)
        ));
        // No close token.
        assert!(!think_soft_close_armed(
            true,
            None,
            100,
            10,
            None,
            Some(512)
        ));
    }

    #[test]
    fn arms_within_window_of_answer_reserve() {
        let reserve = 512;
        // Just outside the window.
        assert!(!think_soft_close_armed(
            true,
            Some(5),
            reserve + THINK_SOFT_CLOSE_WINDOW_TOKENS + 1,
            10,
            None,
            Some(reserve),
        ));
        // Window edge and inside.
        assert!(think_soft_close_armed(
            true,
            Some(5),
            reserve + THINK_SOFT_CLOSE_WINDOW_TOKENS,
            10,
            None,
            Some(reserve),
        ));
        assert!(think_soft_close_armed(
            true,
            Some(5),
            reserve + 1,
            10,
            None,
            Some(reserve)
        ));
    }

    #[test]
    fn hard_close_due_disarms_probe() {
        // remaining == reserve: hard close takes this step.
        assert!(!think_soft_close_armed(
            true,
            Some(5),
            512,
            10,
            None,
            Some(512)
        ));
        // Think cap exhausted: hard close takes this step.
        assert!(!think_soft_close_armed(
            true,
            Some(5),
            8192,
            1000,
            Some(1000),
            None
        ));
    }

    #[test]
    fn arms_within_window_of_think_cap() {
        // Far from the cap.
        assert!(!think_soft_close_armed(
            true,
            Some(5),
            8192,
            10,
            Some(5000),
            None
        ));
        // Within 1024 of the cap.
        assert!(think_soft_close_armed(
            true,
            Some(5),
            8192,
            4000,
            Some(5000),
            None,
        ));
        // Cap larger than remaining+window is irrelevant: cap window only.
        assert!(think_soft_close_armed(
            true,
            Some(5),
            8192,
            4999,
            Some(5000),
            None,
        ));
    }
}

#[cfg(test)]
mod think_budget_close_decision_tests {
    use super::think_budget_close_decision;

    #[test]
    fn inactive_without_think_state_or_close_token() {
        assert!(!think_budget_close_decision(
            false,
            Some(5),
            0,
            0,
            Some(1),
            Some(1),
            true
        ));
        assert!(!think_budget_close_decision(
            true,
            None,
            0,
            0,
            Some(1),
            Some(1),
            true
        ));
    }

    #[test]
    fn hard_close_on_answer_reserve() {
        assert!(think_budget_close_decision(
            true,
            Some(5),
            512,
            100,
            None,
            Some(512),
            false
        ));
        assert!(!think_budget_close_decision(
            true,
            Some(5),
            513,
            100,
            None,
            Some(512),
            false
        ));
    }

    #[test]
    fn hard_close_on_think_cap() {
        assert!(think_budget_close_decision(
            true,
            Some(5),
            9999,
            4096,
            Some(4096),
            None,
            false
        ));
        assert!(!think_budget_close_decision(
            true,
            Some(5),
            9999,
            4095,
            Some(4096),
            None,
            false
        ));
    }

    #[test]
    fn soft_close_only_on_loop_inside_think() {
        assert!(think_budget_close_decision(
            true,
            Some(5),
            9999,
            10,
            None,
            None,
            true
        ));
        assert!(!think_budget_close_decision(
            true,
            Some(5),
            9999,
            10,
            None,
            None,
            false
        ));
    }
}

#[cfg(test)]
mod think_budget_close_override_tests {
    use super::{StopReason, think_budget_close_override};

    #[test]
    fn budget_edge_forces_close_but_keeps_generating() {
        // ds4 hard-limit semantics: the forced close token carries no stop
        // reason, so the remaining output budget funds the answer instead of
        // truncating the request at the reserve edge.
        let override_result =
            think_budget_close_override(None, true, true, Some(5), 512, 100, None, Some(512));
        assert_eq!(override_result, Some((vec![5], None)));
    }

    #[test]
    fn think_cap_edge_forces_close_but_keeps_generating() {
        let override_result =
            think_budget_close_override(None, true, true, Some(5), 9999, 4096, Some(4096), None);
        assert_eq!(override_result, Some((vec![5], None)));
    }

    #[test]
    fn loop_detected_close_keeps_stop_reason() {
        let override_result = think_budget_close_override(
            Some(StopReason::LoopDetected),
            true,
            true,
            Some(5),
            9999,
            10,
            None,
            None,
        );
        assert_eq!(
            override_result,
            Some((vec![5], Some(StopReason::LoopDetected)))
        );
    }

    #[test]
    fn no_override_outside_eligible_steps() {
        // Terminal stops (EOS / max output) are never overridden.
        assert_eq!(
            think_budget_close_override(
                Some(StopReason::EosToken),
                true,
                true,
                Some(5),
                512,
                100,
                None,
                Some(512)
            ),
            None
        );
        assert_eq!(
            think_budget_close_override(
                Some(StopReason::MaxOutputTokens),
                true,
                true,
                Some(5),
                512,
                100,
                None,
                Some(512)
            ),
            None
        );
        // Empty step, outside think, or no controller due: untouched.
        assert_eq!(
            think_budget_close_override(None, false, true, Some(5), 512, 100, None, Some(512)),
            None
        );
        assert_eq!(
            think_budget_close_override(None, true, false, Some(5), 512, 100, None, Some(512)),
            None
        );
        assert_eq!(
            think_budget_close_override(None, true, true, Some(5), 513, 100, None, Some(512)),
            None
        );
    }
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

/// Temperature to carry with a freshly drafted log-prob vector.
///
/// A draft path that wrote no temperature-scaled log-probs must not record a
/// temperature: the accept path would then rescale a cached T against an empty
/// vector. Both fresh-draft writes (skip-state and post-verify) share this
/// rule.
fn pending_draft_log_prob_temperature_for(
    draft_log_probs: &[f32],
    computed_temperature: f32,
) -> Option<f32> {
    (!draft_log_probs.is_empty()).then_some(computed_temperature)
}

#[allow(clippy::too_many_arguments)]
fn mtp_next_adaptive_depth(
    current_depth: usize,
    max_depth: usize,
    pending_len: usize,
    accept_count: usize,
    consecutive_misses: u32,
    // Gemma assistant formal gen: short low-accept trials stack reject-cost
    // steps (median 0.63–0.79×). One complete miss (or half-or-worse accept
    // under the short-ctx early-gen gate) stops further drafting so remaining
    // tokens use empty pure path (~1.0×). Agent coding keeps high accept so
    // this rarely trips on long winning trials.
    aggressive_miss_to_zero: bool,
    depth3_miss_backoff_enabled: bool,
    depth3_hysteresis_enabled: bool,
    draft_observations: Option<(u32, u32, u32, u32)>,
) -> (usize, bool) {
    let fixed_depth = crate::fastpath::mtp_fixed_draft_depth();
    let w = qwen_linear_max_verify_drafts().min(max_depth);
    let conservative_depth = crate::fastpath::mtp_conservative_depth_enabled()
        && draft_observations.is_some_and(mtp_conservative_depth_admitted)
        && depth3_miss_backoff_enabled
        && w >= 3
        && max_depth == w
        && fixed_depth.is_none();
    let depth = mtp_next_adaptive_depth_with_policy(
        current_depth,
        max_depth,
        pending_len,
        accept_count,
        consecutive_misses,
        aggressive_miss_to_zero,
        MtpAdaptiveDepthPolicy {
            fixed_depth,
            conservative_depth,
            depth3_miss_backoff: depth3_miss_backoff_enabled,
            depth3_hysteresis: depth3_hysteresis_enabled,
        },
    );
    (depth, conservative_depth)
}

/// The async committed fold schedules the next draft before record_step;
/// predict its first two position counters so both depth decisions consume
/// the same observations. An empty window contributes no evidence.
fn mtp_depth_observations_after_step(
    telemetry: &MtpTelemetry,
    pending_len: usize,
    accept_count: usize,
) -> (u32, u32, u32, u32) {
    (
        telemetry.drafted_by_depth[0].saturating_add(u32::from(pending_len > 0)),
        telemetry.accepted_by_depth[0]
            .saturating_add(u32::from(pending_len > 0 && accept_count > 0)),
        telemetry.drafted_by_depth[1].saturating_add(u32::from(pending_len > 1)),
        telemetry.accepted_by_depth[1]
            .saturating_add(u32::from(pending_len > 1 && accept_count > 1)),
    )
}

fn mtp_conservative_depth_admitted(
    (first_drafted, first_accepted, second_drafted, second_accepted): (u32, u32, u32, u32),
) -> bool {
    // A moderately reliable first proposal can still justify deeper windows
    // when the second proposal usually survives (as in Cyber-Tiel). Count
    // only actually attempted positions; depth-one windows do not dilute
    // the second-position estimate or permanently lock the policy shallow.
    // Wait for a meaningful sample: eight early misses can misclassify a
    // request whose later deep proposals are consistently useful.
    first_drafted >= 32
        && second_drafted >= 32
        && u64::from(first_accepted) * 4 < u64::from(first_drafted) * 3
        && u64::from(second_accepted) * 2 < u64::from(second_drafted)
}

#[derive(Clone, Copy)]
struct MtpAdaptiveDepthPolicy {
    fixed_depth: Option<usize>,
    conservative_depth: bool,
    depth3_miss_backoff: bool,
    depth3_hysteresis: bool,
}

fn mtp_next_adaptive_depth_with_policy(
    current_depth: usize,
    max_depth: usize,
    pending_len: usize,
    accept_count: usize,
    consecutive_misses: u32,
    aggressive_miss_to_zero: bool,
    policy: MtpAdaptiveDepthPolicy,
) -> usize {
    mtp_next_adaptive_depth_with_policy_for_window(
        current_depth,
        max_depth,
        pending_len,
        accept_count,
        consecutive_misses,
        aggressive_miss_to_zero,
        policy,
        qwen_linear_max_verify_drafts(),
    )
}

/// Width-parameterized body of [`mtp_next_adaptive_depth_with_policy`].
///
/// `window` is the configured throughput window
/// (`qwen_linear_max_verify_drafts()`); the effective controller width is
/// `w = window.min(max_depth)`. A head whose loaded depth differs from the
/// configured window still runs the miss-backoff / hysteresis controllers at
/// the narrower width, while a head wider than the window stays on the generic
/// progressive path exactly as before.
#[allow(clippy::too_many_arguments)]
fn mtp_next_adaptive_depth_with_policy_for_window(
    current_depth: usize,
    max_depth: usize,
    pending_len: usize,
    accept_count: usize,
    consecutive_misses: u32,
    aggressive_miss_to_zero: bool,
    policy: MtpAdaptiveDepthPolicy,
    window: usize,
) -> usize {
    if max_depth == 0 {
        return 0;
    }

    if let Some(fixed_depth) = policy.fixed_depth {
        return fixed_depth.min(max_depth);
    }

    let w = window.min(max_depth);

    // The miss-backoff / hysteresis controllers were written for the
    // three-draft throughput window; they apply to the effective width w
    // (3 by default, 4 under `AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP_DEPTH=4`)
    // with the same shape: start at w, back off to w - 1 only after a
    // complete miss, any accepted draft restores w.
    if policy.depth3_miss_backoff && w >= 3 && max_depth == w {
        if policy.conservative_depth {
            let depth = current_depth.clamp(1, max_depth);
            if pending_len == 0 {
                return depth;
            }
            // Never let a clipped final window increase the requested depth.
            // Acceptance still comes exclusively from the existing verifier.
            return if accept_count >= pending_len {
                if pending_len >= depth {
                    depth.saturating_add(1).min(max_depth)
                } else {
                    depth
                }
            } else {
                accept_count.clamp(1, max_depth)
            };
        }
        if pending_len == 0 {
            return if current_depth == 0 {
                w
            } else {
                current_depth.clamp(1, max_depth)
            };
        }
        return if accept_count == 0 { w - 1 } else { w };
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

    if policy.depth3_hysteresis
        && w >= 3
        && current_depth == w
        && pending_len == w
        && accept_count + 1 == w
    {
        return w;
    }

    // Short-gen stop-loss: complete miss, or half-or-worse accept rate on the
    // aggressive short-ctx early-gen path. Gemma assistant max depth is 2, so
    // "majority reject" as strict <50% only ever matched complete miss; half
    // accepts (1/2) still stack draft+verify cost that kills formal median.
    if aggressive_miss_to_zero
        && (accept_count == 0 || accept_count.saturating_mul(2) <= pending_len)
    {
        return 0;
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
    if let Some(fixed_depth) = crate::fastpath::mtp_fixed_draft_depth() {
        return fixed_depth.min(head_max_depth);
    }
    let w = qwen_linear_max_verify_drafts().min(head_max_depth);
    if crate::fastpath::mtp_depth3_miss_backoff_enabled() && w >= 3 && head_max_depth == w {
        return w;
    }
    if crate::fastpath::qwen_linear_throughput_mtp_enabled()
        && matches!(model_family, "qwen3_next" | "qwen3_5")
    {
        return crate::fastpath::qwen_linear_throughput_mtp_depth().min(head_max_depth);
    }
    match model_family {
        "qwen3_next" | "qwen3_5" => 2.min(head_max_depth),
        // vLLM Gemma 4 MTP docs recommend starting with
        // `num_speculative_tokens: 1`; adaptive depth can still grow to the
        // runtime ceiling (default 2) after full accepts.
        "gemma4" => 1.min(head_max_depth),
        _ => head_max_depth,
    }
}

/// Lazy target probability container for MTP rejection sampling.
///
/// `Full` uses the existing full-vocab softmax path (default): only the draft
/// token probabilities are gathered on GPU.
/// `FullRows` materializes the draft-target softmax rows so CPU-side filters
/// (min_p, matching the primary sampler) can renorm before accept/reject.
/// `TopK` gathers full-vocabulary softmax probabilities for only the top-k tokens
/// per position, then does a CPU-side lookup for each draft token. This avoids
/// transferring a `[verify_len, vocab]` softmax tensor to the CPU.
enum LazyTargetProbs {
    Full(MlxArray),
    /// Softmax rows for draft targets, shape `[pending_len, vocab]`.
    FullRows {
        probs: MlxArray,
        vocab: i32,
        min_p: Option<f32>,
    },
    TopK {
        indices: MlxArray,
        probs: MlxArray,
        k: u32,
        top_p: f32,
        min_p: Option<f32>,
    },
}

impl LazyTargetProbs {
    fn push_eval_targets<'a>(&'a self, targets: &mut Vec<&'a MlxArray>) {
        match self {
            LazyTargetProbs::Full(arr) => targets.push(arr),
            LazyTargetProbs::FullRows { probs, .. } => targets.push(probs),
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
            LazyTargetProbs::FullRows {
                probs,
                vocab,
                min_p,
            } => {
                let vocab = *vocab as usize;
                let data = probs.data_f32();
                workspace.target_probs.reserve(pending.len());
                for (i, &needle) in pending.iter().enumerate() {
                    let row = &data[i * vocab..(i + 1) * vocab];
                    let probability =
                        filtered_target_token_probability(row, needle, *min_p, 1.0, 0);
                    workspace.target_probs.push(probability);
                }
                Some(workspace.target_probs.as_slice())
            }
            LazyTargetProbs::TopK {
                indices,
                probs,
                k,
                top_p,
                min_p,
            } => {
                let k_val = *k as usize;
                let indices_data = indices.data_u32();
                let probs_data = probs.data_f32();
                workspace.target_probs.reserve(pending.len());
                for (i, &needle) in pending.iter().enumerate() {
                    let row_start = i * k_val;
                    let row_end = row_start + k_val;
                    workspace.target_candidates.clear();
                    for j in row_start..row_end {
                        if let (Some(&token), Some(&prob)) =
                            (indices_data.get(j), probs_data.get(j))
                            && prob > 0.0
                            && prob.is_finite()
                        {
                            workspace.target_candidates.push((token, prob));
                        }
                    }
                    // min_p relative to the unfiltered max mass present in the
                    // gathered candidates (top-k already truncated; when k=vocab
                    // this matches full-row min_p).
                    if let Some(min_p) = *min_p
                        && min_p.is_finite()
                        && min_p > 0.0
                    {
                        let max_prob = workspace
                            .target_candidates
                            .iter()
                            .map(|(_, p)| *p)
                            .fold(0.0f32, f32::max);
                        let cutoff = min_p * max_prob;
                        workspace
                            .target_candidates
                            .retain(|(_, prob)| *prob >= cutoff);
                    }
                    workspace.target_candidates.sort_by(
                        |(left_token, left_prob), (right_token, right_prob)| {
                            right_prob
                                .total_cmp(left_prob)
                                .then_with(|| left_token.cmp(right_token))
                        },
                    );
                    if top_p.is_finite() && *top_p > 0.0 && *top_p < 1.0 {
                        let mut cumulative = 0.0_f32;
                        let mut keep = 0_usize;
                        for (_, prob) in &workspace.target_candidates {
                            cumulative += *prob;
                            keep += 1;
                            if cumulative >= *top_p {
                                break;
                            }
                        }
                        workspace.target_candidates.truncate(keep.max(1));
                    }
                    let filtered_sum: f32 = workspace
                        .target_candidates
                        .iter()
                        .map(|(_, prob)| *prob)
                        .sum();
                    let probability = workspace
                        .target_candidates
                        .iter()
                        .find(|(token, _)| *token == needle)
                        .map_or(0.0_f32, |(_, prob)| *prob / filtered_sum.max(1e-37_f32));
                    workspace.target_probs.push(probability.max(0.0_f32));
                }
                Some(workspace.target_probs.as_slice())
            }
        }
    }
}

/// Probability of `token` under the target distribution after min_p / top-p /
/// top-k filtering, renormalized over the kept mass. Matches the primary
/// sampler's law so rejection sampling sees the same `p(token)`.
fn filtered_target_token_probability(
    row_probs: &[f32],
    token: u32,
    min_p: Option<f32>,
    top_p: f32,
    top_k: u32,
) -> f32 {
    let mut candidates: Vec<(u32, f32)> = row_probs
        .iter()
        .enumerate()
        .filter_map(|(idx, &prob)| {
            if prob > 0.0 && prob.is_finite() {
                Some((idx as u32, prob))
            } else {
                None
            }
        })
        .collect();
    if candidates.is_empty() {
        return 0.0;
    }
    if let Some(min_p) = min_p
        && min_p.is_finite()
        && min_p > 0.0
    {
        let max_prob = candidates.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
        let cutoff = min_p * max_prob;
        candidates.retain(|(_, p)| *p >= cutoff);
        if candidates.is_empty() {
            return 0.0;
        }
    }
    candidates.sort_by(|(lt, lp), (rt, rp)| rp.total_cmp(lp).then_with(|| lt.cmp(rt)));
    if top_k > 0 && (top_k as usize) < candidates.len() {
        candidates.truncate(top_k as usize);
    }
    if top_p.is_finite() && top_p > 0.0 && top_p < 1.0 {
        let mut cumulative = 0.0_f32;
        let mut keep = 0_usize;
        for (_, prob) in &candidates {
            cumulative += *prob;
            keep += 1;
            if cumulative >= top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
    }
    let filtered_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if filtered_sum <= 0.0 || !filtered_sum.is_finite() {
        return 0.0;
    }
    candidates
        .iter()
        .find(|(t, _)| *t == token)
        .map_or(0.0, |(_, p)| (*p / filtered_sum).max(0.0))
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
    #[cfg(test)]
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

        let mut all_top_indices = Vec::with_capacity(pending.len());
        let mut all_top_probs = Vec::with_capacity(pending.len());
        // logits_all shape: [1 + pending.len(), vocab].
        // verify_input = [last_token, pending[0], ..., pending[n-1]], so
        // logits_all[i] = prediction after position i = target for pending[i].
        // Row 0 is the target for pending[0]; rows 0..n are the draft targets.
        for row in 0..pending.len() as i32 {
            let row_logits = slice(&scaled, &[row, 0], &[row + 1, vocab], &[1, 1], None);
            let part = argpartition_axis(&row_logits, -k_i32, -1, None);
            let top_idx = slice(&part, &[0, vocab - k_i32], &[1, vocab], &[1, 1], None);
            let full_probs = softmax(&row_logits, -1, None);
            let top_p = take_along_axis(&full_probs, &top_idx, -1, None);
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
            top_p: target_sampling.top_p,
            min_p: target_sampling.min_p.filter(|&m| m.is_finite() && m > 0.0),
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
                let full_probs = softmax(&row_logits, -1, None);
                let top_p = take_along_axis(&full_probs, &top_idx, -1, None);
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
            k: dk_i32 as u32,
            top_p: draft_filter.top_p,
            min_p: target_sampling.min_p.filter(|&m| m.is_finite() && m > 0.0),
        })
    } else if target_sampling.uses_min_p() {
        // DeepSeek thinking defaults min_p=0.05. Materialize draft-target
        // softmax rows so extract can renorm under the same min_p law as the
        // primary sampler (raw single-token take would inflate p_target).
        let probs = softmax(&scaled, -1, None);
        // logits_all rows 0..n are the draft targets (see comment above).
        let draft_rows = slice(&probs, &[0, 0], &[n as i32, vocab], &[1, 1], None);
        Some(LazyTargetProbs::FullRows {
            probs: draft_rows,
            vocab,
            min_p: target_sampling.min_p,
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

fn sample_exact_mtp_delta_rejection_correction(
    logits_all: &MlxArray,
    position: usize,
    vocab: i32,
    sampling: MlxSamplingParams,
    draft_token: u32,
    rng: &mut Xorshift64,
) -> Option<u32> {
    if sampling.temperature <= 0.0 {
        return None;
    }
    let position = position as i32;
    let row = slice(
        logits_all,
        &[position, 0],
        &[position + 1, vocab],
        &[1, 1],
        None,
    );
    eval(&[&row]);
    let target = token_distribution(row.data_f32(), sampling)?;
    let draft = TokenDistribution::new(vec![(draft_token, 1.0)])?;
    sample_residual_token_distribution(&target, &draft, rng)
}

fn select_linear_mtp_correction_token(
    target_temperature: f32,
    recomputed_argmax: Option<u32>,
    exact_residual_correction: Option<u32>,
    verifier_rejection_correction: Option<u32>,
    verifier_argmax: u32,
) -> u32 {
    if target_temperature <= 0.0 {
        // Greedy linear-attention decode must follow the replayed singleton
        // production graph. The verifier correction was derived from the
        // rejected multi-token graph and is precisely the value being replaced.
        recomputed_argmax
            .or(exact_residual_correction)
            .or(verifier_rejection_correction)
            .unwrap_or(verifier_argmax)
    } else {
        // Preserve distribution-correct residual sampling when available.
        exact_residual_correction
            .or(verifier_rejection_correction)
            .or(recomputed_argmax)
            .unwrap_or(verifier_argmax)
    }
}

fn forced_linear_mtp_greedy_revalidation(forced_replay: bool, sampling: MlxSamplingParams) -> bool {
    forced_replay && sampling.temperature <= 0.0 && !sampling.uses_logits_processors()
}

/// Forced greedy replay must not reuse a batched row as the next primary.
fn take_mtp_skip_state(
    state: &mut RequestState,
    forced_greedy_replay: bool,
) -> (Option<MlxArray>, Option<MlxArray>, Option<MlxArray>) {
    let captured = (
        state.mtp_skip_logits.take(),
        state.mtp_skip_argmax.take(),
        state.mtp_skip_hidden.take(),
    );
    if forced_greedy_replay {
        (None, None, None)
    } else {
        captured
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_linear_mtp_accepted_prefix(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    primary: u32,
    pending: &[u32],
    token_offset: usize,
    accept: &mut MtpAcceptOutcome,
    predicted: &mut Vec<u32>,
    revalidate_greedy: bool,
) -> u32 {
    if revalidate_greedy {
        let replay = revalidate_greedy_prefix_with_argmax(
            cfg,
            weights,
            cache,
            primary,
            &pending[..accept.accept_count],
            token_offset,
        );
        *accept = MtpAcceptOutcome {
            accept_count: replay.accept_count,
            all_accepted: replay.accept_count == pending.len(),
            rejection_correction: (replay.accept_count < pending.len())
                .then_some(replay.correction_token),
        };
        *predicted = replay.predicted;
        return replay.correction_token;
    }
    recompute_committed_prefix_with_argmax(
        cfg,
        weights,
        cache,
        primary,
        &pending[..accept.accept_count],
        token_offset,
    )
}

/// Maximum draft length served by the exact lazy-checkpoint path.
///
/// The invariant-projection arithmetic contract is validated for a 1-4 token
/// verifier (up to three drafts plus the committed token). Longer drafts retain
/// singleton state replay. `AX_MLX_MTP_LINEAR_EXACT_REPLAY!=0` also revalidates
/// greedy acceptance without logits processors in the non-optimistic path;
/// unforced replay alone does not establish singleton acceptance identity.
/// Drafts the lazy-checkpoint / projected-replay path serves in this
/// process: the certified three, or the configured experimental throughput
/// width when it is wider (`AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP_DEPTH`).
fn qwen_linear_max_verify_drafts() -> usize {
    crate::fastpath::qwen_linear_mtp_max_verify_drafts()
}

/// Exact arithmetic is the speculative-verifier contract.
///
/// An exact-eligible pack (AXQ sidecar, affine 4/6/8, or MXFP4 gs32) auto-selects
/// the profile at load time. Applying that profile on `--ax-direct` skips the
/// packed linear-attention inputs route and other fused S=1 kernels. Direct
/// decode therefore keeps the community-4-bit fast path; MTP still installs
/// the verifier contract when `mtp_requested` is on.
const fn qwen_linear_mtp_exact_scope_for_request(
    resolved_profile_enabled: bool,
    mtp_requested: bool,
) -> bool {
    resolved_profile_enabled && mtp_requested
}

/// Resolves default packaged-MTP admission independently from the CLI n-gram
/// switch. `AX_NO_SPEC` supplies `speculation_disabled` and remains the only
/// construction-time kill switch shared by both speculative mechanisms.
const fn default_mtp_requested(
    speculation_disabled: bool,
    route_safe: bool,
    certified_or_forced: bool,
) -> bool {
    !speculation_disabled && route_safe && certified_or_forced
}

/// A deferred Qwen draft is valid whenever the linear-attention verifier is
/// allowed to consume the same lazy token arrays. The exact profile supplies
/// that contract automatically; projected replay supplies the equivalent
/// explicit opt-in for the stock-arithmetic verifier. Other architectures stay
/// excluded because their verify/rollback paths do not consume this lazy chain.
const fn qwen_linear_mtp_async_draft_allowed(
    has_linear_attention: bool,
    exact_profile_enabled: bool,
    projected_replay_enabled: bool,
) -> bool {
    has_linear_attention && (exact_profile_enabled || projected_replay_enabled)
}

/// Batched committed-history fold is the ungated greedy regime. Throughput MTP
/// resolves the draft gate to 0 even when the process env still holds 0.90.
const fn qwen_batched_committed_fold_gate_ok(env_gate: f32, resolved_gate_x1000: u32) -> bool {
    env_gate == 0.0 || resolved_gate_x1000 == 0
}

/// Emit the stable runner contract separately from the per-step arithmetic
/// scope. A terminal short-budget or latched-bypass step can legitimately drop
/// the scope after earlier MTP verification work; it must not rewrite the
/// resolved model profile to disabled in the final response.
fn append_qwen_linear_mtp_exact_route_decisions(
    decisions: &mut Vec<(String, u32)>,
    eligible: bool,
    resolved_profile_enabled: bool,
    active_for_step: bool,
    selection: u32,
) {
    upsert_route_decision(
        decisions,
        "ax_mlx_qwen_linear_mtp_exact_eligible",
        u32::from(eligible),
    );
    upsert_route_decision(
        decisions,
        "ax_mlx_qwen_linear_mtp_exact_enabled",
        u32::from(resolved_profile_enabled),
    );
    upsert_route_decision(
        decisions,
        "ax_mlx_qwen_linear_mtp_exact_active",
        u32::from(active_for_step),
    );
    upsert_route_decision(
        decisions,
        "ax_mlx_qwen_linear_mtp_exact_selection",
        selection,
    );
}

fn qwen_linear_mtp_exact_tensor_supported(
    source_quantized: bool,
    quantization: Option<(&str, u32, u32)>,
) -> bool {
    match quantization {
        Some((mode, bits, group_size)) => {
            (mode == "affine" && matches!(bits, 4 | 6 | 8) && matches!(group_size, 32 | 64))
                || (mode == "mxfp4" && bits == 4 && group_size == 32)
        }
        // Dense BF16/F16/F32 projections are covered by the invariant dense
        // kernel. A tensor marked source-quantized without normalized affine
        // metadata is not a safe implicit capability.
        None => !source_quantized,
    }
}

/// Runtime-owned capability gate for the exact Qwen linear-MTP verifier.
///
/// This intentionally derives from the loaded model contract rather than an
/// artifact's marketing/runtime recommendation. Qwen3.5 and Qwen3.6 share the
/// `qwen3_5` runtime family. The invariant kernels and recurrent checkpoint are
/// certified for draft depths 1-3 and dense, affine 4/6/8-bit, or MXFP4
/// (bits=4, group_size=32) tensors with the production group sizes used by
/// uniform, OptiQ, and AXQ artifacts. MXFP4 uses the mode-aware MLX qmm
/// path (no affine group-bias channel); the invariant affine Metal kernel
/// does not host it.
fn qwen_linear_mtp_exact_model_eligible(
    model_family: &str,
    has_linear_attention: bool,
    mtp_depth: usize,
    tensor_specs: &[ax_engine_core::NativeTensorSpec],
) -> bool {
    model_family == "qwen3_5"
        && has_linear_attention
        && (1..=qwen_linear_max_verify_drafts()).contains(&mtp_depth)
        && tensor_specs.iter().all(|tensor| {
            qwen_linear_mtp_exact_tensor_supported(
                tensor.source_quantized,
                tensor.quantization.as_ref().map(|quantization| {
                    (
                        quantization.mode.as_str(),
                        quantization.bits,
                        quantization.group_size,
                    )
                }),
            )
        })
}

fn linear_mtp_requires_singleton_replay(
    pending_len: usize,
    exact_profile_enabled: bool,
    replay_kill_switch: bool,
) -> bool {
    // Exact profile drafts within the validated verifier width ride the lazy
    // checkpoint path (fast accept/restore). Kill switch, empty drafts, or
    // longer drafts keep singleton state recompute.
    pending_len == 0
        || pending_len > qwen_linear_max_verify_drafts()
        || !exact_profile_enabled
        || replay_kill_switch
}

fn linear_mtp_projected_replay_allowed(
    pending_len: usize,
    enabled: bool,
    replay_kill_switch: bool,
    model_force_replay: bool,
) -> bool {
    enabled
        && !replay_kill_switch
        && !model_force_replay
        && (1..=qwen_linear_max_verify_drafts()).contains(&pending_len)
}

/// Device-side greedy acceptance (`AX_MLX_MTP_DEVICE_GREEDY_ACCEPT`) applies
/// only when the accept decision is a pure argmax prefix match: greedy
/// request, non-optimistic linear arm without singleton replay, and an
/// all-MTP draft window (n-gram / hybrid / assistant sources keep the host
/// path, as does any sources/pending misalignment).
fn mtp_device_greedy_accept_allowed(
    flag_enabled: bool,
    target_temperature: f32,
    exact_linear_replay: bool,
    pending_len: usize,
    sources: &[MtpDraftSource],
) -> bool {
    flag_enabled
        && target_temperature <= 0.0
        && !exact_linear_replay
        && pending_len > 0
        && sources.len() == pending_len
        && sources.iter().all(|source| *source == MtpDraftSource::Mtp)
}

/// Validate a device greedy-accept readback. The packed `[accept_count,
/// correction_token]` pair is only trusted when it has exactly the two expected
/// elements and a count that fits the verified window; anything else (a
/// truncated or malformed array, or an out-of-range count) fails closed to the
/// host acceptance path for that step instead of committing a bogus accept
/// prefix.
fn device_greedy_accept_pair(pair: &[u32], pending_len: usize) -> Option<(usize, u32)> {
    let [accept_count, correction_token] = pair else {
        return None;
    };
    let accept_count = usize::try_from(*accept_count).ok()?;
    (accept_count <= pending_len).then_some((accept_count, *correction_token))
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
    argmax_with_softmax_confidence_gpu_exact_and_arrays(logits, &[])
}

/// GPU-exact argmax + softmax confidence, materialising `extra` arrays in the
/// same `eval` (e.g. assistant `post_projection` backbone for the next depth).
fn argmax_with_softmax_confidence_gpu_exact_and_arrays(
    logits: &MlxArray,
    extra: &[&MlxArray],
) -> (u32, f32) {
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
    let mut refs: Vec<&MlxArray> = Vec::with_capacity(2 + extra.len());
    refs.push(&token_arr);
    refs.push(&prob_arr);
    refs.extend_from_slice(extra);
    eval(&refs);
    let token = token_arr.data_u32().first().copied().unwrap_or(0);
    let confidence = prob_arr.data_f32().first().copied().unwrap_or(0.0);
    (token, confidence)
}

/// Lazy multi-depth Gemma assistant draft: build the full depth chain without
/// per-depth GPU sync, materialise once, then apply host confidence gates.
///
/// Mirrors the Qwen MTP fused lazy draft pattern. Always computes up to
/// `max_depth` forwards (no early host abort mid-chain); the returned prefix
/// still stops at the first gate miss, so committed drafts match the
/// per-depth path's correctness contract.
fn gemma4_assistant_draft_token_lazy_multi_depth(
    session: &crate::model::Gemma4AssistantDraftSession<'_>,
    last_token: u32,
    last_backbone_hidden: &MlxArray,
    base_position: usize,
    max_depth: usize,
    first_gate: f32,
    deep_gate: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>) {
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_confs: Vec<MlxArray> = Vec::with_capacity(max_depth);

    let first_token_data = [last_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        std::mem::size_of_val(&first_token_data),
        &[1_i32],
        MlxDtype::Uint32,
    );
    let mut cur_hidden = last_backbone_hidden.clone();

    for d in 0..max_depth {
        let draft_position = gemma4_assistant_draft_rope_position(base_position, d);
        let Ok((logits, projected_hidden)) =
            session.forward_one_from_token_arr(&prev_token_arr, &cur_hidden, draft_position)
        else {
            break;
        };
        let shape = logits.shape();
        let Some(vocab) = shape.last().copied() else {
            break;
        };
        if vocab <= 0 || shape.iter().copied().product::<i32>() != vocab {
            break;
        }
        let logits_2d = reshape(&logits, &[1, vocab], None);
        let lazy_tok = argmax(&logits_2d, None);
        let probs = softmax(&logits_2d, -1, None);
        let lazy_conf = take(&probs, &lazy_tok, 1, None);
        lazy_tokens.push(lazy_tok.clone());
        lazy_confs.push(lazy_conf);
        prev_token_arr = lazy_tok;
        cur_hidden = astype(&projected_hidden, MlxDtype::Bfloat16, None);
    }

    if lazy_tokens.is_empty() {
        return (vec![], vec![], vec![]);
    }

    // Single batch eval: all depth tokens + confidences (and chained
    // post_projection work already in the token/conf graph).
    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(lazy_tokens.len() * 2);
    for t in &lazy_tokens {
        all_refs.push(t);
    }
    for c in &lazy_confs {
        all_refs.push(c);
    }
    eval(&all_refs);

    let mut drafts: Vec<u32> = Vec::with_capacity(lazy_tokens.len());
    for d in 0..lazy_tokens.len() {
        let token = lazy_tokens[d].data_u32().first().copied().unwrap_or(0);
        let confidence = lazy_confs[d].data_f32().first().copied().unwrap_or(0.0);
        if !crate::model::gemma4_assistant_draft_position_accepted(
            d, confidence, first_gate, deep_gate,
        ) {
            break;
        }
        drafts.push(token);
    }
    (drafts, vec![], vec![])
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

/// Apply n-gram loop detection after EOS/max-output truncation (WS-C2).
///
/// Keeps emitted tokens and sets [`StopReason::LoopDetected`] (maps to OpenAI
/// `finish_reason=stop`). Does not invent an EOS token id. Distinct from
/// `no_repeat_ngram_size` logit bans.
fn apply_loop_detection_stop(
    mut sampled_tokens: Vec<u32>,
    stop_reason: Option<StopReason>,
    generated_history: &[u32],
    loop_cfg: Option<ax_engine_core::LoopDetectionConfig>,
) -> (Vec<u32>, Option<StopReason>) {
    if stop_reason.is_some() || sampled_tokens.is_empty() {
        return (sampled_tokens, stop_reason);
    }
    let Some(cfg) = loop_cfg.filter(|c| c.is_enabled()) else {
        return (sampled_tokens, stop_reason);
    };
    // Multi-token MTP can cross the loop threshold mid-batch. Truncate to the
    // shortest prefix that triggers detection so MTP-on matches single-token
    // direct decode (which stops on the token that first completes the loop).
    let mut probe = Vec::with_capacity(generated_history.len().saturating_add(1));
    probe.extend_from_slice(generated_history);
    for index in 0..sampled_tokens.len() {
        probe.push(sampled_tokens[index]);
        if ax_engine_core::detects_loop(&probe, cfg) {
            sampled_tokens.truncate(index + 1);
            return (sampled_tokens, Some(StopReason::LoopDetected));
        }
    }
    (sampled_tokens, stop_reason)
}

/// Resolve loop-detection config for this model family (default-on Gemma 4 only).
fn loop_detection_config_for_family(
    model_family: &str,
) -> Option<ax_engine_core::LoopDetectionConfig> {
    match crate::fastpath::gemma4_loop_detection_mode() {
        crate::fastpath::Gemma4LoopDetectionMode::Off => None,
        crate::fastpath::Gemma4LoopDetectionMode::Force => {
            Some(ax_engine_core::LoopDetectionConfig::GEMMA4_DEFAULT)
        }
        crate::fastpath::Gemma4LoopDetectionMode::Default => model_family
            .starts_with("gemma4")
            .then_some(ax_engine_core::LoopDetectionConfig::GEMMA4_DEFAULT),
    }
}

/// Loop detection for one request. Fixed-token `ignore_eos` benches never arm
/// it so generation always reaches `max_output_tokens`.
fn loop_detection_for_request(
    ignore_eos: bool,
    model_family: &str,
) -> Option<ax_engine_core::LoopDetectionConfig> {
    if ignore_eos {
        None
    } else {
        loop_detection_config_for_family(model_family)
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

fn portable_prefix_store_allowed(
    prefill_completes_prompt: bool,
    native_store_enabled: bool,
) -> bool {
    prefill_completes_prompt || native_store_enabled
}

/// Resolve the physical sliding-KV mode for one runner item.
///
/// Scheduler-level fair prefill can split one prompt across many execution
/// items. Non-terminal items are cache-only, while the completing item ends in
/// a one-token logits forward. Allowing the session's decode-default rotation
/// during either kind would convert the cache to slot order before a later
/// multi-token prefill item arrives. Keep every prefill item ordered and let
/// `initialize_generation_state` latch rotation only after the complete prompt
/// has finished.
fn cache_rotation_for_execution(
    mode: ExecutionMode,
    request_latch: Option<(bool, usize)>,
    session_rotating_decode: bool,
    is_greedy: bool,
    prefill_chunk: usize,
) -> (bool, usize) {
    cache_rotation_for_execution_with_prefill_flag(
        mode,
        request_latch,
        session_rotating_decode,
        is_greedy,
        prefill_chunk,
        crate::fastpath::rotating_sliding_prefill_enabled()
            || crate::fastpath::sibling_prefill_rotation(),
    )
}

fn cache_rotation_for_execution_with_prefill_flag(
    mode: ExecutionMode,
    request_latch: Option<(bool, usize)>,
    session_rotating_decode: bool,
    is_greedy: bool,
    prefill_chunk: usize,
    prefill_rotation_enabled: bool,
) -> (bool, usize) {
    if mode == ExecutionMode::Prefill {
        // A request that already latched its rotation decision keeps it for
        // every later prefill item. The sibling-rotation hint is a process
        // atomic that can flip mid-prompt (the sibling going idle clears
        // it); re-evaluating per item then hands (false, 0) to a cache whose
        // rings already rotated, and the next multi-token append panics on
        // the slot-order invariant (measured: dummy 13.8k warm dying at
        // seq_len 11776 with rotating_slack 0 once the qwen stream ended).
        if let Some(latched) = request_latch {
            return latched;
        }
        // Ordered prefill is the production default (portable prefix snapshots
        // + pure window decode). When prefill rotation is opted in, size slack
        // so multi-token chunks fit the ring eligibility gate in
        // `sliding_ring_layout` (seq <= rotating_slack).
        if session_rotating_decode && is_greedy && prefill_rotation_enabled {
            let slack = prefill_chunk.max(64);
            return (true, slack);
        }
        return (false, 0);
    }
    // Decode: pure-direct (seq=1) can rotate with slack 0. Multi-token verify
    // (Gemma assistant-MTP depth-2 → seq=3) only enters the ring when
    // seq <= rotating_slack; otherwise pure-direct uses the ring while
    // multi-token uses ordered storage + window views — long-context agent
    // identity break. Size slack for formal draft depth + a little headroom.
    request_latch.unwrap_or((
        session_rotating_decode && is_greedy,
        if session_rotating_decode && is_greedy {
            8
        } else {
            0
        },
    ))
}

fn prefill_drain_async_eval_count(token_count: usize, prefill_chunk: usize) -> u32 {
    let count = token_count.saturating_sub(1) / prefill_chunk.max(1);
    count.min(u32::MAX as usize) as u32
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
    is_greedy && !sampling.uses_logits_processors()
}

/// Derive the media component of a prefix-cache key. `Some("")` for
/// text-only requests, `Some(digest)` when a recognized family covers the
/// media, and `None` when media is present but no family digest maps it —
/// today `unlimited_ocr`, and any future `RequestMultimodalInputs` field
/// added without a `media_prefix_key` mapping here. Callers must treat
/// `None` as "prefix cache unusable for this request" (fail closed), never
/// as a text-only key.
fn media_key_for_fingerprint(
    multimodal_inputs: Option<&RequestMultimodalInputs>,
    fingerprint: &str,
) -> Option<String> {
    let Some(inputs) = multimodal_inputs else {
        return Some(String::new());
    };
    if let Some(g) = inputs.gemma4_unified.as_ref().filter(|g| !g.is_empty()) {
        return Some(g.media_prefix_key(fingerprint));
    }
    if let Some(q) = inputs.qwen3_vl.as_ref().filter(|q| !q.is_empty()) {
        return Some(q.media_prefix_key(fingerprint));
    }
    if let Some(minicpm) = inputs.minicpm_v46.as_ref().filter(|v| !v.is_empty()) {
        return Some(minicpm.media_prefix_key(fingerprint));
    }
    if let Some(omni) = inputs.nemotron_omni.as_ref().filter(|v| !v.is_empty()) {
        return Some(omni.media_prefix_key(fingerprint));
    }
    if inputs.is_empty() {
        Some(String::new())
    } else {
        None
    }
}

fn hash_prefix_tokens(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in tokens {
        hash ^= u64::from(*token);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash ^ (tokens.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Fold media digests into the prefix-cache layer layout string (WS-M3).
///
/// Empty `media_key` keeps the base layout unchanged (text-only path).
pub(crate) fn format_prefix_layer_layout(base_layout: &str, media_key: &str) -> String {
    if media_key.is_empty() {
        base_layout.to_string()
    } else {
        format!("{base_layout};media={media_key}")
    }
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

/// Select a cold-grid restore boundary, or require a full scheduler-prefix replay.
fn grid_aligned_prefix_restore_len(
    item: &ax_engine_core::ExecutionItem,
    candidate_len: usize,
    chunk: usize,
) -> Option<usize> {
    let scheduler_claim = item.reused_prefix_token_slice.len();
    let extension_tokens = if item.mode == ExecutionMode::Prefill && scheduler_claim > 0 {
        // The scheduler input already contains only the uncached suffix.
        item.input_token_slice.len()
    } else {
        item.input_token_slice.len().saturating_sub(candidate_len)
    };
    let restore_len = if extension_tokens > 1 {
        candidate_len / chunk.max(1) * chunk.max(1)
    } else {
        candidate_len
    };
    if item.mode == ExecutionMode::Prefill && restore_len < scheduler_claim {
        None
    } else {
        Some(restore_len)
    }
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
        || state.cache.seq_len() != 0
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

/// Number of leading item tokens already covered by a runner-side snapshot
/// beyond the scheduler's absolute prefill position.
///
/// `position_range.start`, not `reused_prefix_token_slice.len()`, is the
/// scheduler's complete expectation for how much context precedes this item.
/// On an ordinary multi-step prefill both cache length and position start grow
/// together (2K, 4K, ...), so the result must stay zero. Using only the reused
/// slice length treated every later cold chunk as a probe bonus, discarded the
/// whole input, and repeatedly rebuilt an isolated 2K cache.
fn runner_probe_over_claim(cache_seq_len: usize, item: &ax_engine_core::ExecutionItem) -> usize {
    cache_seq_len.saturating_sub(item.position_range.start as usize)
}

struct MlxItemRun {
    update: RequestExecutionUpdate,
    ngram_acceleration: NgramAccelerationTelemetry,
    mtp_telemetry: MtpTelemetry,
    flash_next_mtp_telemetry: FlashNextMtpTelemetry,
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
            diffusion_schedule: None,
        },
        ngram_acceleration: NgramAccelerationTelemetry::default(),
        mtp_telemetry: MtpTelemetry::default(),
        flash_next_mtp_telemetry: FlashNextMtpTelemetry::default(),
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
    streak == LINEAR_NGRAM_NO_DRAFT_DISABLE_THRESHOLD
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

/// Extra ring slots for bounded-rollback rotating sliding KV. Must absorb
/// the widest speculative forward a covered request can roll back: an
/// n-gram verify of `1 + MAX_DRAFT_LEN` tokens (the verify input is
/// `[last_token, draft...]` and at least one token always commits, so the
/// deepest `trim_to` is `MAX_DRAFT_LEN`; the ring additionally needs the
/// whole verify batch resident, hence `+ 1`).
const ROTATING_BOUNDED_ROLLBACK_SLACK: usize = 8;
const _: () = assert!(
    crate::ngram_accel::MAX_DRAFT_LEN < ROTATING_BOUNDED_ROLLBACK_SLACK,
    "bounded-rollback ring slack must cover the widest n-gram verify forward (MAX_DRAFT_LEN + 1)"
);

/// Whether this model routes sliding-window attention through
/// `families::standard::layer_forward` — the only append/mask seam that
/// understands bounded-rollback rings. `llama4` and `gpt_oss` pass raw
/// windows to `append_with_retained_window` but build family-local masks
/// with no ring awareness, and DiffusionGemma's block decode reads KV
/// through its own machinery, so bounded rings must never engage for them.
/// Families without sliding windows are harmlessly included: the ring
/// predicate never fires without a window.
fn rotating_bounded_family_eligible(cfg: &crate::model::ModelConfig) -> bool {
    cfg.diffusion.is_none()
        && matches!(
            cfg.model_family.as_str(),
            // gemma4_vl text SWA uses families::standard::layer_forward (same as gemma4).
            "gemma4" | "gemma4_vl" | "gemma3" | "qwen3" | "llama3" | "qwen3_5" | "qwen3_next"
        )
}

/// The rotating sliding-KV mode for a request: `None` = no rotation,
/// `Some(0)` = pure window-sized ring (mask-free single-token SDPA, no
/// rollback ever), `Some(slack)` = bounded-rollback ring (`window + slack`
/// slots, slot-validity masks, `trim_to` up to `slack` deep).
///
/// Pure rings require that no speculative source can roll the cache back
/// for the request's entire lifetime: the whole session on the rollback-free
/// direct policy, or n-gram disabled under a sticky reason with no MTP
/// drafting. Bounded rings extend rotation to n-gram-ACTIVE requests — every
/// rollback source there is an n-gram verify `trim_to` bounded by
/// `MAX_DRAFT_LEN` — but only for families whose sliding attention flows
/// through the ring-aware `families::standard` seam. `LinearNoDraft` lands
/// in the bounded arm because `maybe_reenable_linear_ngram_from_fallback_
/// output` can re-enable n-gram mid-request for that reason (moot in
/// practice: it only arises on linear-attention models, which have no
/// sliding windows to rotate). MTP stays excluded from both arms.
#[allow(clippy::too_many_arguments)]
fn request_rotating_sliding_slack(
    session_rotating: bool,
    rotating_flag_enabled: bool,
    bounded_rollback_enabled: bool,
    bounded_family_eligible: bool,
    request_ngram_disabled: bool,
    disable_reason: NgramRequestDisableReason,
    mtp_ring_slack: Option<usize>,
    is_greedy: bool,
) -> Option<usize> {
    // Pure rings (slack 0) tolerate zero rollback ever, so both pure
    // classes keep the historical greedy requirement and exclude any MTP
    // drafting. Bounded rings only require every rollback to fit inside
    // the slack:
    //  - a sampled non-MTP request has exactly the same rollback sources
    //    as a greedy one (n-gram verify trims, bounded by MAX_DRAFT_LEN
    //    under rejection sampling too: the correction token always
    //    commits), so sampled requests are bounded-eligible and always
    //    take the slack ring, never a pure one;
    //  - `mtp_ring_slack` classifies the model's MTP drafting: `Some(0)`
    //    = none, `Some(n)` = every MTP rollback is a `state.cache.trim_to`
    //    bounded by a pending draft of at most `n - 1` tokens (the Gemma4
    //    assistant), `None` = an MTP source rings cannot cover (qwen/GLM
    //    MTP heads — moot in practice, those models have no sliding
    //    windows, but fail closed).
    let has_mtp = mtp_ring_slack != Some(0);
    // Direct sessions keep pure rings on non-MTP models. Gemma assistant packs
    // set mtp_ring_slack > 0 even under AX_NO_SPEC so MTP-off/on share bounded
    // geometry (required for multi-token append/trim and greedy A/B identity).
    if session_rotating && is_greedy {
        if let Some(slack) = mtp_ring_slack.filter(|s| *s > 0) {
            return Some(ROTATING_BOUNDED_ROLLBACK_SLACK.max(slack));
        }
        return Some(0);
    }
    if !rotating_flag_enabled {
        return None;
    }
    if is_greedy
        && !has_mtp
        && request_ngram_disabled
        && !matches!(disable_reason, NgramRequestDisableReason::LinearNoDraft)
    {
        return Some(0);
    }
    if bounded_rollback_enabled && bounded_family_eligible {
        let mtp_slack = mtp_ring_slack?;
        return Some(ROTATING_BOUNDED_ROLLBACK_SLACK.max(mtp_slack));
    }
    None
}

/// Primary token for a skip-state MTP cycle.
///
/// The greedy path must read a real argmax for the committed primary:
/// `sample_logit_row`'s greedy shortcut returns its `argmax_tok` argument
/// verbatim, so passing a placeholder there emits the placeholder as a real
/// token (historically literal token id 0 — rendered as "!" — whenever the
/// draft gate left `pending` empty and the skip path carried the cycle).
/// The argmax array is captured lazily with the skip logits; if a stale
/// state misses it, compute it on demand rather than guess.
#[allow(clippy::too_many_arguments)]
fn skip_state_primary_token(
    skip_logits: &MlxArray,
    skip_argmax: Option<MlxArray>,
    vocab: i32,
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> u32 {
    use crate::ngram_accel::sample_logit_row;
    use mlx_sys::{argmax, eval};

    if sampling.temperature <= 0.0 {
        let argmax_arr = skip_argmax.unwrap_or_else(|| argmax(skip_logits, None));
        eval(&[&argmax_arr]);
        // argmax over a [1, vocab] (or [vocab]) row yields exactly one
        // element; `unwrap_or_default` is unreachable and exists only to
        // honor the no-panic rule.
        return argmax_arr.data_u32().first().copied().unwrap_or_default();
    }
    sample_logit_row(
        skip_logits,
        0,
        0,
        vocab,
        sampling,
        rng,
        sampling_probs_buf,
        sampling_logits_buf,
        sampling_candidates_buf,
    )
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
    // ngram draft will advance cache.seq_len() independently, so
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

#[derive(Debug, thiserror::Error)]
pub enum MlxRunnerError {
    #[error("MLX model feature is not supported: {0}")]
    UnsupportedFeature(String),
    #[error("weight loading failed: {0}")]
    Weights(#[from] crate::weights::WeightLoadError),
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod flash_next_tests;

#[cfg(test)]
mod flash_next_pause_tests;
