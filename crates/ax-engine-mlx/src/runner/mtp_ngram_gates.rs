//! MTP + n-gram speculation gate decisions for the MLX runner.
//!
//! Split out of `runner/mod.rs` (Phase 2 slice 3 of the decode-dispatch
//! efficiency plan): the utility / speculative-safety / hurt / saturation /
//! auto-disable gate machinery that decides per step whether n-gram drafts
//! may stack onto (or replace) MTP drafts, plus the pseudo-log-prob mapping
//! for n-gram-sourced draft positions. Pure host-side policy over telemetry
//! counters — no MLX graph state. The cached env readers feeding these gates
//! live in `runner/mtp_tuning.rs`.

use std::sync::OnceLock;

use ax_engine_core::runner::RunnerRequestContext;

use super::MtpNgramAcceptanceMode;
use super::mtp_tuning::{
    adaptive_ngram_saturation_threshold, mtp_ngram_auto_disable_min_ngram,
    mtp_ngram_auto_disable_mtp_threshold, mtp_ngram_auto_disable_mtp_warmup,
    mtp_ngram_auto_disable_ngram_warmup, mtp_ngram_gate_min_samples, mtp_ngram_safety_mode,
    mtp_ngram_utility_hysteresis_steps, mtp_ngram_utility_margin_ratio,
    mtp_ngram_utility_min_emitted_tokens, mtp_ngram_utility_min_ngram_tokens,
    ngram_policy_variant_from_env,
};
use crate::ngram_accel::NgramPolicyVariant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MtpNgramAutoDisableConfig {
    pub(super) mtp_warmup: u32,
    pub(super) ngram_warmup: u32,
    pub(super) mtp_threshold: u32,
    pub(super) ngram_floor: u32,
}

impl MtpNgramAutoDisableConfig {
    pub(super) fn from_env() -> Self {
        Self {
            mtp_warmup: mtp_ngram_auto_disable_mtp_warmup(),
            ngram_warmup: mtp_ngram_auto_disable_ngram_warmup(),
            mtp_threshold: (mtp_ngram_auto_disable_mtp_threshold() * 1000.0) as u32,
            ngram_floor: (mtp_ngram_auto_disable_min_ngram() * 1000.0) as u32,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum MtpNgramGatePolicy {
    #[default]
    Rate,
    Utility,
}

impl MtpNgramGatePolicy {
    pub(super) fn route_code(self) -> u32 {
        match self {
            Self::Rate => 0,
            Self::Utility => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DraftSourceUtility {
    pub(super) submitted_tokens: u32,
    pub(super) proposer_wall_us: u32,
    pub(super) verify_wall_us: u32,
    pub(super) emitted_tokens: u32,
}

impl DraftSourceUtility {
    pub(super) fn cost_per_emitted_token_us(self) -> Option<f64> {
        if self.emitted_tokens == 0 {
            return None;
        }
        let total = u64::from(self.proposer_wall_us).saturating_add(u64::from(self.verify_wall_us));
        Some(total as f64 / f64::from(self.emitted_tokens))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MtpNgramUtilityGateConfig {
    pub(super) min_emitted_tokens: u32,
    pub(super) min_ngram_submitted_tokens: u32,
    pub(super) margin_ratio: f64,
    pub(super) hysteresis_steps: u32,
}

impl MtpNgramUtilityGateConfig {
    pub(super) fn from_env() -> Self {
        Self {
            min_emitted_tokens: mtp_ngram_utility_min_emitted_tokens(),
            min_ngram_submitted_tokens: mtp_ngram_utility_min_ngram_tokens(),
            margin_ratio: mtp_ngram_utility_margin_ratio(),
            hysteresis_steps: mtp_ngram_utility_hysteresis_steps(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MtpNgramUtilityDecision {
    pub(super) gated: bool,
    pub(super) insufficient_samples: bool,
    pub(super) utility_hurt: bool,
    pub(super) hysteresis_active: bool,
}

pub(super) fn mtp_ngram_utility_gate(
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
pub(super) enum SpeculativeSafetyReason {
    #[default]
    None,
    ToolCall,
    StructuredOutput,
    ReasoningTrace,
    ExperimentalOverride,
}

impl SpeculativeSafetyReason {
    pub(super) fn route_code(self) -> u32 {
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
pub(super) struct SpeculativeSafetyDecision {
    pub(super) disable_ngram: bool,
    pub(super) tighten_ngram: bool,
    pub(super) reason: SpeculativeSafetyReason,
}

pub(super) fn mtp_ngram_speculative_safety_decision(
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

pub(super) fn mtp_ngram_speculative_safety_decision_for_mode(
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
pub(super) enum MtpNgramSafetyMode {
    Off,
    DisableAll,
    DisableReasoning,
    #[default]
    TightenReasoning,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MtpNgramGateDecision {
    pub(super) gated: bool,
    pub(super) saturated: bool,
    pub(super) hurt: bool,
    pub(super) auto_disabled: bool,
    pub(super) self_tune_disabled: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mtp_ngram_gate_decision(
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

pub(super) fn mtp_ngram_saturated_gate(
    ngram_max: usize,
    mtp_depth: usize,
    mtp_only_ewma: f32,
    mtp_only_samples: u32,
) -> bool {
    ngram_max > 0
        && mtp_only_samples >= mtp_ngram_gate_min_samples()
        && mtp_only_ewma >= adaptive_ngram_saturation_threshold(mtp_depth)
}

pub(super) fn mtp_ngram_hurt_gate(
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

pub(super) fn mtp_ngram_auto_disable_gate(
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
pub(super) enum HurtGateMode {
    #[default]
    SourceAware,
    LegacyEwma,
}

pub(super) fn mtp_ngram_hurt_gate_mode() -> HurtGateMode {
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

pub(super) fn mtp_ngram_min_support() -> u32 {
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

pub(super) fn mtp_ngram_confidence_threshold() -> f32 {
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

pub(super) fn mtp_ngram_min_context_len() -> usize {
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
pub(super) fn mtp_ngram_policy_variant() -> NgramPolicyVariant {
    static CACHED: OnceLock<NgramPolicyVariant> = OnceLock::new();
    *CACHED.get_or_init(ngram_policy_variant_from_env)
}

/// Source-aware hurt gate: compares per-source n-gram vs MTP acceptance rates
/// using raw draft/accepted counters rather than EWMA values.
///
/// Fires when n-gram per-token acceptance is worse than MTP per-token acceptance
/// by more than the margin — the exact condition where n-gram is genuinely hurting.
/// This avoids the selection bias in the legacy EWMA-based gate (see ADR-019).
pub(super) fn mtp_ngram_source_hurt_gate(
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

pub(super) fn mtp_ngram_pseudo_log_prob(confidence: f32, mode: MtpNgramAcceptanceMode) -> f32 {
    if !confidence.is_finite() {
        return f32::NAN;
    }
    match mode {
        MtpNgramAcceptanceMode::Confidence => confidence.clamp(1e-37, 1.0).ln().max(-30.0),
        MtpNgramAcceptanceMode::Delta | MtpNgramAcceptanceMode::Greedy => 0.0_f32,
    }
}

pub(super) fn mtp_ngram_pseudo_log_probs(
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
