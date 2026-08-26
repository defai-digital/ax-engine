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

/// Minimum remaining generation budget (tokens) required to schedule MTP draft.
///
/// When fewer than this many tokens remain on the request budget, draft/verify
/// fixed cost cannot amortize (ADR-020 short-output policy). Default **16**.
/// Formal harnesses that must force MTP set `AX_MLX_MTP_MIN_REMAINING_TOKENS=0`.
pub(super) fn mtp_min_remaining_tokens() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_MIN_REMAINING_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(16)
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
    !matches!(
        mtp_ngram_stacking_env(),
        MtpNgramStackingEnv::ExplicitlyEnabled
    )
}

/// Parsed `AX_MLX_MTP_DISABLE_NGRAM_STACKING` so unset can be distinguished
/// from an explicit `=1` (isolated MTP benches).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MtpNgramStackingEnv {
    Unset,
    ExplicitlyDisabled,
    ExplicitlyEnabled,
}

pub(super) fn mtp_ngram_stacking_env() -> MtpNgramStackingEnv {
    static CACHED: OnceLock<MtpNgramStackingEnv> = OnceLock::new();
    *CACHED.get_or_init(
        || match std::env::var("AX_MLX_MTP_DISABLE_NGRAM_STACKING") {
            Err(_) => MtpNgramStackingEnv::Unset,
            Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "0" | "false" | "no" => MtpNgramStackingEnv::ExplicitlyEnabled,
                _ => MtpNgramStackingEnv::ExplicitlyDisabled,
            },
        },
    )
}

/// Whether `run_mtp_decode` may put n-gram tokens in front of the MTP head.
///
/// Unset keeps historical pure-MTP benches. Official Qwen38 `--full` leaves
/// the var unset and sets `AX_MLX_QWEN_LINEAR_MTP_EXACT=1`; general-long
/// ignore_eos then loops special tokens that n-gram can accept and MTP
/// cannot (~51%). Explicit `=1` still forces isolated MTP.
///
/// `runner_disabled` is the session/CLI contract (`--mlx-mtp-disable-ngram-
/// stacking` → `MlxRunner::disable_mtp_ngram_stacking`). It must gate here
/// and not only in the coalesced-verify route: the harness pure-MTP row
/// (CLI flag, env unset) otherwise still fires n-gram drafts and trips its
/// zero-n-gram contract (observed 105 hit steps on qwen3.8-27b-axq-4bit,
/// 2026-08-16).
pub(super) fn mtp_ngram_stacking_allowed(
    env: MtpNgramStackingEnv,
    qwen_linear_mtp_exact: bool,
    runner_disabled: bool,
) -> bool {
    if runner_disabled {
        return false;
    }
    match env {
        MtpNgramStackingEnv::ExplicitlyEnabled => true,
        MtpNgramStackingEnv::ExplicitlyDisabled => false,
        MtpNgramStackingEnv::Unset => qwen_linear_mtp_exact,
    }
}

/// Cap stacked n-gram length to the MTP adaptive depth.
///
/// Factory `944fa8a7` allowed `DEFAULT_DRAFT_LEN` (4) on exact S=2 and
/// general-long fell to 1.00× (S=5 verify). Official Qwen38 depth is 1.
pub(super) fn mtp_ngram_stack_len(requested: usize, adaptive_mtp_depth: usize) -> usize {
    requested.min(adaptive_mtp_depth.max(1))
}

/// Exact Qwen38 general-long ignore_eos loops after ~one 3-token cycle.
/// Default stacked `min_support=3` only fired near the tail (`fa6e1e79`
/// 23/39). Cap at 2 when the operator did not raise the env.
pub(super) fn mtp_ngram_min_support_for_exact(exact: bool, configured: u32) -> u32 {
    if exact { configured.min(2) } else { configured }
}

/// Exact ignore_eos loops are a 3-token cycle; allowing bigrams (context 2)
/// lets n-gram fire as soon as `198 → 248045` has been seen twice.
pub(super) fn mtp_ngram_min_context_len_for_exact(exact: bool, configured: usize) -> usize {
    if exact { configured.min(2) } else { configured }
}

/// Next token of a period-2 or period-3 suffix that already repeated twice.
///
/// Factory general-long ignore_eos emits `248045,248046,198` after the short
/// answer. Table n-gram at min_support=2 / conf=0.85 only raised accept
/// 21/41 → 24/38. Two visible periods are enough to draft the continuation
/// at depth 1 without replacing MTP on non-loop text.
pub(super) fn short_cycle_next_token(recent: &[u32]) -> Option<u32> {
    // Period-1: four identical tokens (EOS / pad loops). Two or three
    // repeats is too common in natural text to draft from.
    if recent.len() >= 4 {
        let last = recent[recent.len() - 1];
        if recent[recent.len() - 4..].iter().all(|&tok| tok == last) {
            return Some(last);
        }
    }
    for period in [3_usize, 2] {
        let need = period.saturating_mul(2);
        if recent.len() >= need {
            let tail = &recent[recent.len() - need..];
            if tail[..period] == tail[period..] {
                return Some(tail[0]);
            }
        }
        // One visible period plus the first token of the next (`[P][P[0]]`)
        // is enough to draft `P[1]` at depth 1. Two full periods still win
        // when present so the existing factory loop keeps drafting `P[0]`.
        let prefix_need = period.saturating_add(1);
        if recent.len() >= prefix_need {
            let window = &recent[recent.len() - prefix_need..];
            let pattern = &window[..period];
            // All-equal `[a,a]` / `[a,a,a]` is period-1; the four-token
            // rule above owns that so a triple repeat does not draft.
            if window[period] == pattern[0] && pattern.iter().any(|&tok| tok != pattern[0]) {
                return Some(pattern[1]);
            }
        }
    }
    None
}

/// `history` is prior committed output; `extra` is this step's newly
/// accepted tokens (not yet pushed onto `generated_tokens`).
pub(super) fn short_cycle_next_token_from_parts(history: &[u32], extra: &[u32]) -> Option<u32> {
    let take_h = history.len().min(8);
    let mut buf = [0u32; 16];
    let mut n = 0usize;
    for &tok in &history[history.len() - take_h..] {
        buf[n] = tok;
        n += 1;
    }
    for &tok in extra {
        if n >= buf.len() {
            break;
        }
        buf[n] = tok;
        n += 1;
    }
    short_cycle_next_token(&buf[..n])
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

/// Publisher-declared MTP runtime certification, parsed from the optional
/// `"mtp"` block of `axquant_runtime.json` in the pack root.
///
/// This is the fail-closed release gate for default-on MTP: a pack only gets
/// MTP enabled by default when its publisher has *measured* the pack and
/// stamped it as optimized (or recorded a >= 1.0x speedup). Absent, missing,
/// or malformed metadata all resolve to `default_on == false`; explicit
/// requests (`MlxMtpPolicy::Required`, env overrides) are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct MtpRuntimeCertification {
    /// `mtp.enabled_by_default` as published (false when absent/malformed).
    pub enabled_by_default: bool,
    /// `mtp.optimized` as published (false when absent/malformed).
    pub optimized: bool,
    /// `mtp.measured_speedup` scaled by 1000 (e.g. 1.20x -> 1200), kept as an
    /// integer so the struct stays `Copy + Eq`. `None` when absent/invalid.
    pub measured_speedup_x1000: Option<u32>,
    /// The certification verdict: publisher opted in *and* backed it with
    /// either an `optimized` stamp or a measured speedup >= 1.0x.
    pub default_on: bool,
}

impl MtpRuntimeCertification {
    pub(crate) fn from_runtime_json(value: &serde_json::Value) -> Self {
        let Some(mtp) = value.get("mtp").and_then(|v| v.as_object()) else {
            return Self::default();
        };
        let enabled_by_default = mtp
            .get("enabled_by_default")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let optimized = mtp
            .get("optimized")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let measured_speedup_x1000 = mtp
            .get("measured_speedup")
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(|v| (v * 1000.0).round().min(f64::from(u32::MAX)) as u32);
        let default_on =
            enabled_by_default && (optimized || measured_speedup_x1000.is_some_and(|x| x >= 1000));
        Self {
            enabled_by_default,
            optimized,
            measured_speedup_x1000,
            default_on,
        }
    }
}

/// Read the MTP runtime certification from `<root>/axquant_runtime.json`.
/// Fail-closed: a missing or unparseable file yields the all-false default.
pub(crate) fn mtp_runtime_certification(root: &std::path::Path) -> MtpRuntimeCertification {
    let path = root.join("axquant_runtime.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return MtpRuntimeCertification::default();
    };
    match serde_json::from_str::<serde_json::Value>(&raw) {
        Ok(value) => MtpRuntimeCertification::from_runtime_json(&value),
        Err(_) => MtpRuntimeCertification::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MtpNgramStackingEnv, MtpRuntimeCertification, mtp_ngram_stacking_allowed,
        mtp_runtime_certification,
    };

    #[test]
    fn certification_defaults_closed_without_mtp_block() {
        let v: serde_json::Value = serde_json::json!({"quantization": {"bits": 6}});
        assert_eq!(
            MtpRuntimeCertification::from_runtime_json(&v),
            MtpRuntimeCertification::default()
        );
        assert!(!MtpRuntimeCertification::default().default_on);
    }

    #[test]
    fn certification_enabled_but_unoptimized_stays_off() {
        // Today's fleet state: every published MTP pack carries exactly this.
        let v = serde_json::json!({"mtp": {
            "enabled_by_default": true, "optimized": false, "measured_speedup": null
        }});
        let cert = MtpRuntimeCertification::from_runtime_json(&v);
        assert!(cert.enabled_by_default);
        assert!(!cert.optimized);
        assert_eq!(cert.measured_speedup_x1000, None);
        assert!(!cert.default_on);
    }

    #[test]
    fn certification_optimized_stamp_turns_default_on() {
        let v = serde_json::json!({"mtp": {"enabled_by_default": true, "optimized": true}});
        assert!(MtpRuntimeCertification::from_runtime_json(&v).default_on);
    }

    #[test]
    fn certification_measured_speedup_gates_at_unity() {
        let win = serde_json::json!({"mtp": {
            "enabled_by_default": true, "optimized": false, "measured_speedup": 1.20
        }});
        let cert = MtpRuntimeCertification::from_runtime_json(&win);
        assert_eq!(cert.measured_speedup_x1000, Some(1200));
        assert!(cert.default_on);

        let lose = serde_json::json!({"mtp": {
            "enabled_by_default": true, "measured_speedup": 0.96
        }});
        let cert = MtpRuntimeCertification::from_runtime_json(&lose);
        assert_eq!(cert.measured_speedup_x1000, Some(960));
        assert!(!cert.default_on, "0.96x flagship row must not default on");
    }

    #[test]
    fn certification_requires_enabled_by_default_opt_in() {
        let v = serde_json::json!({"mtp": {
            "enabled_by_default": false, "optimized": true, "measured_speedup": 1.55
        }});
        assert!(!MtpRuntimeCertification::from_runtime_json(&v).default_on);
    }

    #[test]
    fn certification_rejects_malformed_fields() {
        let v = serde_json::json!({"mtp": {
            "enabled_by_default": "yes", "optimized": 1, "measured_speedup": "fast"
        }});
        let cert = MtpRuntimeCertification::from_runtime_json(&v);
        assert_eq!(cert, MtpRuntimeCertification::default());

        let nan = serde_json::json!({"mtp": {
            "enabled_by_default": true, "measured_speedup": -1.0
        }});
        assert_eq!(
            MtpRuntimeCertification::from_runtime_json(&nan).measured_speedup_x1000,
            None
        );
    }

    #[test]
    fn certification_missing_file_fails_closed() {
        let dir = std::env::temp_dir().join("ax-mtp-cert-missing-test");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::remove_file(dir.join("axquant_runtime.json"));
        assert_eq!(
            mtp_runtime_certification(&dir),
            MtpRuntimeCertification::default()
        );
        // Unparseable file also fails closed.
        std::fs::write(dir.join("axquant_runtime.json"), b"{not json").unwrap();
        assert_eq!(
            mtp_runtime_certification(&dir),
            MtpRuntimeCertification::default()
        );
        let _ = std::fs::remove_file(dir.join("axquant_runtime.json"));
    }

    #[test]
    fn exact_qwen_linear_mtp_stacks_ngram_when_env_unset() {
        assert!(
            mtp_ngram_stacking_allowed(MtpNgramStackingEnv::Unset, true, false),
            "official Qwen38 exact --full leaves DISABLE_NGRAM_STACKING unset"
        );
        assert!(
            !mtp_ngram_stacking_allowed(MtpNgramStackingEnv::Unset, false, false),
            "non-exact unset keeps historical pure-MTP benches"
        );
        assert!(!mtp_ngram_stacking_allowed(
            MtpNgramStackingEnv::ExplicitlyDisabled,
            true,
            false
        ));
        assert!(mtp_ngram_stacking_allowed(
            MtpNgramStackingEnv::ExplicitlyEnabled,
            false,
            false
        ));
    }

    #[test]
    fn runner_cli_flag_disables_stacking_regardless_of_env() {
        // --mlx-mtp-disable-ngram-stacking reaches the runner as a session
        // field, not the env var; it must still suppress n-gram stacking
        // (harness pure-MTP rows rely on the CLI flag alone).
        assert!(!mtp_ngram_stacking_allowed(
            MtpNgramStackingEnv::Unset,
            true,
            true
        ));
        assert!(!mtp_ngram_stacking_allowed(
            MtpNgramStackingEnv::ExplicitlyEnabled,
            true,
            true
        ));
    }

    #[test]
    fn exact_ngram_stack_len_never_exceeds_mtp_depth() {
        assert_eq!(super::mtp_ngram_stack_len(4, 1), 1);
        assert_eq!(super::mtp_ngram_stack_len(4, 3), 3);
        assert_eq!(super::mtp_ngram_stack_len(1, 1), 1);
        assert_eq!(super::mtp_ngram_stack_len(0, 1), 0);
    }

    #[test]
    fn exact_ngram_min_support_caps_default_at_two() {
        assert_eq!(super::mtp_ngram_min_support_for_exact(true, 3), 2);
        assert_eq!(super::mtp_ngram_min_support_for_exact(true, 1), 1);
        assert_eq!(super::mtp_ngram_min_support_for_exact(false, 3), 3);
    }

    #[test]
    fn exact_ngram_min_context_caps_default_at_two() {
        assert_eq!(super::mtp_ngram_min_context_len_for_exact(true, 3), 2);
        assert_eq!(super::mtp_ngram_min_context_len_for_exact(false, 3), 3);
    }

    #[test]
    fn short_cycle_next_token_drafts_factory_general_long_loop() {
        // Measured trial-1/2 prefixes from gateffi general-long MTP-on.
        assert_eq!(
            super::short_cycle_next_token(&[
                271, 84068, 248044, 248045, 248046, 198, 248045, 248046, 198
            ]),
            Some(248045)
        );
        assert_eq!(
            super::short_cycle_next_token(&[
                271, 39, 30763, 46, 248044, 248045, 248045, 248046, 198, 248045, 248046, 198
            ]),
            Some(248045)
        );
        assert_eq!(super::short_cycle_next_token(&[1, 2, 1, 2]), Some(1));
        assert_eq!(super::short_cycle_next_token(&[1, 2, 3, 1, 2, 3]), Some(1));
        assert_eq!(super::short_cycle_next_token(&[1, 2, 3, 4, 5]), None);
        assert_eq!(
            super::short_cycle_next_token(&[248045, 248046, 198]),
            None,
            "one period without a confirming prefix is not enough"
        );
        assert_eq!(
            super::short_cycle_next_token(&[248045, 248046, 198, 248045]),
            Some(248046),
            "one period plus the first token of the next drafts P[1]"
        );
        assert_eq!(
            super::short_cycle_next_token(&[248045, 248046, 198, 248045, 248046]),
            Some(198)
        );
        assert_eq!(super::short_cycle_next_token(&[1, 2, 1]), Some(2));
        assert_eq!(
            super::short_cycle_next_token(&[7, 7, 7, 7]),
            Some(7),
            "four identical tokens is a period-1 loop"
        );
        assert_eq!(
            super::short_cycle_next_token(&[7, 7, 7]),
            None,
            "three identical tokens is too weak"
        );
        assert_eq!(
            super::short_cycle_next_token_from_parts(
                &[271, 84068, 248044, 248045, 248046, 198, 248045],
                &[248046, 198]
            ),
            Some(248045),
            "must see the loop across generated_tokens + this-step result"
        );
        assert_eq!(
            super::short_cycle_next_token(&[248046, 198]),
            None,
            "this-step result alone is too short (the factory bug)"
        );
    }
}
