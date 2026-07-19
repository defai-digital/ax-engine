//! MTP / n-gram speculation tuning knobs for the MLX runner.
//!
//! Split out of `runner/mod.rs` (Phase 2 slice 2 of the decode-dispatch
//! efficiency plan): the cached env-threshold readers, mode parsers, and the
//! n-gram draft-length policy helpers. Everything here is host-side policy —
//! no MLX graph state. The gate *decision* machinery (utility/safety/hurt
//! gates) still lives in `runner/mod.rs` pending the next slice.

use std::sync::OnceLock;

use crate::ngram_accel::{
    DEFAULT_DRAFT_LEN, LINEAR_MIN_NGRAM_SUPPORT, MAX_DRAFT_LEN, NgramDraftOutcome,
    NgramDraftPolicy, NgramPolicyVariant, NgramTable, effective_draft_confidence_threshold,
};
use crate::speculation_profile::speculation_profile_from_env;

use super::{
    MtpModelAcceptanceMode, MtpNgramAcceptanceMode, MtpNgramGatePolicy, MtpNgramSafetyMode,
    NGRAM_DRAFT_LEN_LOW_CONFIDENCE, NGRAM_DRAFT_LEN_SHRINK_THRESHOLD, POST_THINK_MIN_NGRAM_SUPPORT,
};

/// Maximum number of history tokens to warm up the MTP KV cache.
/// The most recent tokens dominate the MTP head's attention context for
/// speculative decoding; older tokens have diminishing returns.
/// Override with `AX_MLX_MTP_WARMUP_CAP` (0 = unlimited, default 256).
pub(super) fn mtp_warmup_cap() -> usize {
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
pub(super) fn mtp_ngram_gate_min_samples() -> u32 {
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
pub(super) fn mtp_auto_optimistic_deactivate_threshold() -> f32 {
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
pub(super) fn mtp_auto_optimistic_min_samples() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_AUTO_OPTIMISTIC_MIN_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
    })
}

pub(super) fn cached_env_f32(name: &str, default: f32, min: f32, max: f32) -> f32 {
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
pub(super) fn mtp_bypass_min_samples() -> u32 {
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
pub(super) fn mtp_bypass_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f32("AX_MLX_MTP_BYPASS_THRESHOLD", 0.50, 0.0, 1.0))
}

pub(super) fn cached_env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

pub(super) fn cached_env_f64(name: &str, default: f64, min: f64, max: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(min, max))
        .unwrap_or(default)
}

pub(super) fn route_cost_us(value: Option<f64>) -> u32 {
    value
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.round().min(u32::MAX as f64) as u32)
        .unwrap_or(0)
}

pub(super) fn mtp_ngram_hurt_margin() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f32("AX_MLX_MTP_NGRAM_HURT_MARGIN", 0.02, 0.0, 1.0))
}

pub(super) fn mtp_ngram_gate_policy_from_env() -> MtpNgramGatePolicy {
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

pub(super) fn mtp_ngram_utility_min_emitted_tokens() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_UTILITY_MIN_EMITTED_TOKENS", 128))
}

pub(super) fn mtp_ngram_utility_min_ngram_tokens() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_UTILITY_MIN_NGRAM_TOKENS", 32))
}

pub(super) fn mtp_ngram_utility_margin_ratio() -> f64 {
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f64("AX_MLX_MTP_NGRAM_UTILITY_MARGIN_RATIO", 0.02, 0.0, 10.0))
}

pub(super) fn mtp_ngram_utility_hysteresis_steps() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_UTILITY_HYSTERESIS_STEPS", 16))
}

pub(super) fn mtp_ngram_safety_mode() -> MtpNgramSafetyMode {
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

pub(super) fn mtp_ngram_auto_disable_mtp_threshold() -> f32 {
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

pub(super) fn mtp_ngram_auto_disable_min_ngram() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED
        .get_or_init(|| cached_env_f32("AX_MLX_MTP_NGRAM_AUTO_DISABLE_MIN_NGRAM", 0.50, 0.0, 1.0))
}

pub(super) fn mtp_ngram_self_tune_threshold() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_f32("AX_MLX_MTP_NGRAM_SELF_TUNE_THRESHOLD", 0.30, 0.0, 1.0))
}

pub(super) fn mtp_ngram_self_tune_warmup() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_SELF_TUNE_WARMUP", 32))
}

pub(super) fn mtp_ngram_auto_disable_mtp_warmup() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_AUTO_DISABLE_MTP_WARMUP", 64))
}

pub(super) fn mtp_ngram_auto_disable_ngram_warmup() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| cached_env_u32("AX_MLX_MTP_NGRAM_AUTO_DISABLE_NGRAM_WARMUP", 32))
}

pub(super) fn mtp_ngram_acceptance_mode_from_env() -> MtpNgramAcceptanceMode {
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

pub(super) fn mtp_model_acceptance_mode_from_env() -> MtpModelAcceptanceMode {
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

pub(super) fn mtp_disable_ngram_stacking_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        !matches!(
            std::env::var("AX_MLX_MTP_DISABLE_NGRAM_STACKING").as_deref(),
            Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
        )
    })
}

/// **Default: OFF** (explicit opt-in via `AX_MLX_MTP_OPTIMISTIC=1`).
///
/// MTP verify always accepts all draft tokens without computing the
/// rejection-sampling acceptance ratio.  Eliminates full-vocab softmax for
/// target distribution, the accept/reject loop, and cache rollback on rejection.
/// This is an approximate speed-ceiling profile: draft/target mismatches are
/// committed, so it is not eligible for an exact correctness claim.
pub(super) fn mtp_optimistic_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_MTP_OPTIMISTIC").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

pub(super) fn mtp_auto_optimistic_enabled_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_MTP_AUTO_OPTIMISTIC").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
    })
}

pub(super) fn mtp_optimistic_allowed(has_glm_mtp: bool) -> bool {
    // GLM's sidecar can draft plausible but target-mismatched code tokens; keep
    // verifier acceptance on for correctness instead of unconditional accept.
    !has_glm_mtp
}

pub(super) fn mtp_optimistic_draft_min_confidence_override() -> Option<f32> {
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

/// Experimental — **default OFF** since 2026-07-18. When enabled with
/// `AX_MLX_MTP_SKIP_STATE=1`, the MTP decode path captures verify logits and
/// hidden state as "skip state" and reuses them on the next cycle instead of
/// running the main model forward for the first token position.
///
/// The implementation as designed is not output-correct, which is why the
/// default flipped: a capture cycle emits its tail token but never forwards
/// it, so every skip cycle leaves the previous tail out of the KV history —
/// the "one forward saved per cycle" was exactly one token missing from the
/// model's context (`.internal/bugs/2026-07-18-mtp-repetition-penalty-`
/// `corruption.md`). On top of that, the greedy primary was committed
/// through `sample_logit_row`'s argmax shortcut with a placeholder `0`,
/// emitting literal token id 0 (fixed — the capture now carries the row
/// argmax — but greedy skip cycles still duplicate the previous tail by
/// construction). The path only engages when the draft gate leaves a cycle
/// with no pending draft, so benchmark workloads (which draft nearly every
/// cycle) never exercised it. Keep it off unless studying a corrected
/// always-advance design.
pub(super) fn mtp_skip_state_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let enabled = matches!(
            std::env::var("AX_MLX_MTP_SKIP_STATE")
                .unwrap_or_default()
                .as_str(),
            "1" | "true" | "TRUE"
        );
        if enabled {
            tracing::warn!(
                "AX_MLX_MTP_SKIP_STATE=1: experimental MTP skip-state is \
                 enabled; skip cycles omit the previous tail token from the \
                 KV history and duplicate it in greedy output (see \
                 .internal/bugs/2026-07-18-mtp-repetition-penalty-corruption.md)"
            );
        }
        enabled
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
pub(super) fn mtp_target_softmax_topk_from_env() -> Option<u32> {
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

pub(super) fn ngram_policy_variant_from_env() -> NgramPolicyVariant {
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

pub(super) fn ngram_acceleration_draft(
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
pub(super) fn ngram_acceleration_policy(
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

pub(super) fn adaptive_ngram_draft_len(has_linear_attention: bool, posterior_mean: f32) -> usize {
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

pub(super) fn adaptive_ngram_saturation_threshold(mtp_depth: usize) -> f32 {
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
