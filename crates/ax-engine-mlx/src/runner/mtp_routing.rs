//! MTP draft-source routing enums and per-request speculation route policy.
//!
//! Split out of `runner/mod.rs` (Phase 2 slice 4 of the decode-dispatch
//! efficiency plan): the draft-source taxonomy (MTP head vs n-gram vs
//! assistant), acceptance/correctness/proposal-law modes, the per-request
//! route resolution, and the n-gram self-tune accumulator. These are the
//! shared vocabulary between the telemetry block, the gate machinery, and
//! the decode loops — extracting them first unblocks the telemetry move.

use crate::sampling::MlxSamplingParams;

use super::saturating_u32;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum MtpDraftSource {
    #[default]
    None,
    Mtp,
    Gemma4Assistant,
    Ngram,
    HybridMtp,
}

impl MtpDraftSource {
    pub(super) fn is_model_draft(self) -> bool {
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
    pub(super) fn optimistic_accept_eligible(self) -> bool {
        matches!(self, MtpDraftSource::Mtp | MtpDraftSource::HybridMtp)
    }

    pub(super) fn utility_family(self) -> DraftSourceFamily {
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
pub(super) enum DraftSourceFamily {
    #[default]
    Mtp,
    Assistant,
    Ngram,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum MtpNgramAcceptanceMode {
    #[default]
    Confidence,
    Delta,
    Greedy,
}

impl MtpNgramAcceptanceMode {
    pub(super) fn route_code(self) -> u32 {
        match self {
            Self::Confidence => 0,
            Self::Delta => 1,
            Self::Greedy => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum MtpModelAcceptanceMode {
    #[default]
    Greedy,
    RejectionSampling,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum MtpCorrectnessMode {
    #[default]
    Unknown,
    GreedyExact,
    SampledExact,
    ApproximateOptimistic,
    DirectFallback,
}

impl MtpCorrectnessMode {
    pub(super) const fn route_code(self) -> u32 {
        match self {
            Self::Unknown => 0,
            Self::GreedyExact => 1,
            Self::SampledExact => 2,
            Self::ApproximateOptimistic => 3,
            Self::DirectFallback => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum MtpProposalLaw {
    #[default]
    Unknown,
    DeterministicDelta,
    Stochastic,
}

impl MtpProposalLaw {
    pub(super) const fn route_code(self) -> u32 {
        match self {
            Self::Unknown => 0,
            Self::DeterministicDelta => 1,
            Self::Stochastic => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MtpRequestRoute {
    DirectFallback,
    StrictMtp,
    Other,
}

pub(super) const fn mtp_request_route(
    has_mtp: bool,
    mtp_requested: bool,
    exact_supported: bool,
    approximate_profile: bool,
    mtp_bypassed: bool,
    uses_repetition_penalty: bool,
) -> MtpRequestRoute {
    if has_mtp && mtp_requested {
        if (exact_supported || approximate_profile) && !mtp_bypassed && !uses_repetition_penalty {
            MtpRequestRoute::StrictMtp
        } else {
            MtpRequestRoute::DirectFallback
        }
    } else {
        MtpRequestRoute::Other
    }
}

pub(super) fn mtp_exact_sampling_supported(
    sampling: MlxSamplingParams,
    target_softmax_topk: Option<u32>,
) -> bool {
    if sampling.uses_logits_processors()
        || sampling.uses_min_p()
        || crate::mtp::mtp_draft_mode_from_env() != crate::mtp::MtpDraftMode::Greedy
    {
        // min_p requires the full-row / residual-filtered accept path (DeepSeek
        // thinking defaults); the Qwen linear exact profile only covers plain
        // temperature + optional top-k/top-p.
        return false;
    }
    sampling.temperature <= 0.0
        || (target_softmax_topk.is_none() && (sampling.top_k > 0 || sampling.top_p >= 1.0))
}

pub(super) const fn should_bootstrap_direct_pipeline(
    session_direct: bool,
    request_ngram_disabled: bool,
    has_mtp: bool,
    mtp_uses_direct_pipeline: bool,
) -> bool {
    // Pure session-direct (n-gram disabled at the runner boundary) is the
    // README/direct-mode contract: always prime the double-buffer pipeline.
    // MTP weights may still be attached to the package, but pure direct
    // sessions clear `mtp_requested` so they must not skip bootstrap.
    //
    // When the session still allows speculation, only bootstrap when MTP is
    // explicitly on the direct-fallback route, or when no MTP is attached and
    // the request itself disabled n-gram.
    session_direct || mtp_uses_direct_pipeline || (request_ngram_disabled && !has_mtp)
}

/// Greedy Flash-0731 with uncertified nextn must use the mlx-lm-style
/// async_eval double-buffer even when n-gram is still session-on. The sidecar
/// makes `has_mtp` true, but `route_safe` is false so MTP is not requested.
pub(super) const fn v4_uncertified_uses_pure_direct_pipeline(
    v4_direct_fallback: bool,
    disable_ngram: bool,
    think_soft_close_armed: bool,
    uses_logits_processors: bool,
    greedy: bool,
) -> bool {
    !think_soft_close_armed
        && (disable_ngram || v4_direct_fallback)
        && !uses_logits_processors
        && greedy
}

pub(super) const fn should_use_session_direct_pipeline(
    session_direct: bool,
    is_greedy: bool,
    has_mtp: bool,
    mtp_requested: bool,
) -> bool {
    // Pure direct sessions (n-gram disabled at the session boundary) must use
    // the double-buffer pipeline for greedy decode. MTP is only an alternative
    // when it is both attached and still requested; callers that want direct
    // mode must clear `mtp_requested` when constructing the runner (see
    // `MlxRunner::from_artifacts_inner`).
    session_direct && is_greedy && !(has_mtp && mtp_requested)
}

#[allow(clippy::too_many_arguments)]
pub(super) const fn gemma4_assistant_mtp_coalesced_verify_route(
    enabled: bool,
    assistant_attached: bool,
    target_mtp_attached: bool,
    mtp_requested: bool,
    ngram_stacking_disabled: bool,
    skip_state_enabled: bool,
    deterministic_greedy: bool,
    uses_logits_processors: bool,
    has_pending_assistant_draft: bool,
    adaptive_gate_active: bool,
) -> bool {
    enabled
        && assistant_attached
        && !target_mtp_attached
        && mtp_requested
        && ngram_stacking_disabled
        && !skip_state_enabled
        && deterministic_greedy
        && !uses_logits_processors
        && has_pending_assistant_draft
        && !adaptive_gate_active
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct NgramSelfTuneState {
    pub(super) drafted: u32,
    pub(super) accepted: u32,
    pub(super) disabled: bool,
}

impl NgramSelfTuneState {
    pub(super) fn record_submitted(&mut self, drafted: usize) {
        self.drafted = self.drafted.saturating_add(saturating_u32(drafted));
    }

    pub(super) fn record_verified(&mut self, accepted: usize, threshold: f32, warmup: u32) {
        self.accepted = self.accepted.saturating_add(saturating_u32(accepted));
        if !self.disabled && warmup > 0 && self.drafted >= warmup {
            let rate = self.accepted as f32 / self.drafted.max(1) as f32;
            if rate < threshold {
                self.disabled = true;
            }
        }
    }
}

/// Longest period considered by the Gemma assistant-MTP cycle guard.
///
/// Formal pilot divergences were short-period cycle continuations (period 1–4
/// stuck tokens / short loops). Caps keep the predicate O(1) and conservative.
pub(super) const GEMMA_CYCLE_GUARD_MAX_PERIOD: usize = 16;

/// Minimum full periods that must already appear at the history tail before a
/// cycle is treated as "established" (avoids killing legitimate first repeats).
pub(super) const GEMMA_CYCLE_GUARD_MIN_ESTABLISHED_PERIODS: usize = 2;

/// Generated-token count below which formal multi-token adopt is forced onto
/// pure-direct sequential. Covers measured general-long `first_diff@13` without
/// permanently disabling LONG_MT multi-token later in a request.
pub(super) const GEMMA_MT_EARLY_GEN_PURE_DIRECT_TOKENS: usize = 32;

/// Greedy Gemma assistant-MTP verify route under the formal multi-token profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GemmaGreedyVerifyRoute {
    /// Singleton pure-direct sequential re-verify (exact vs MTP-off by construction).
    SequentialOracle,
    /// In-place multi-token teacher-forced adopt (speed path; formal profile default).
    MultiTokenAdopt,
}

/// Resolve greedy Gemma assistant-MTP verify route.
///
/// - `oracle_on`: product default sequential oracle (`SEQUENTIAL_ORACLE=1`).
/// - `guard_on`: cycle-continuation guard (default ON; only forces more oracle).
/// - `cycle_hit`: draft starts by continuing an established committed-tail cycle.
/// - `early_gen_force`: force pure-direct for early generated tokens under formal
///   multi-token (non-cycle residual identity failures).
///
/// Fail-closed: every force only adds sequential verification, never removes it.
pub(super) const fn gemma_greedy_verify_route(
    oracle_on: bool,
    guard_on: bool,
    cycle_hit: bool,
    early_gen_force: bool,
) -> GemmaGreedyVerifyRoute {
    if oracle_on || early_gen_force || (guard_on && cycle_hit) {
        GemmaGreedyVerifyRoute::SequentialOracle
    } else {
        GemmaGreedyVerifyRoute::MultiTokenAdopt
    }
}

/// Whether early-generation pure-direct should force sequential under formal
/// multi-token verify.
pub(super) const fn gemma_early_gen_pure_direct_force(
    early_gen_enabled: bool,
    generated_tokens: usize,
    threshold: usize,
) -> bool {
    early_gen_enabled && generated_tokens < threshold
}

/// True when `draft` begins by continuing a repetition cycle already
/// established (≥ [`GEMMA_CYCLE_GUARD_MIN_ESTABLISHED_PERIODS`] full periods)
/// at the tail of committed `history`.
///
/// Pure decision rule used before multi-token always-adopt under formal
/// `SEQUENTIAL_ORACLE=0`. Cycle-continuation false accepts were the dominant
/// formal-pilot divergence mode (teacher-forced multi-token matching a looping
/// draft while sequential greedy would break the cycle).
pub(super) fn draft_continues_committed_cycle(history: &[u32], draft: &[u32]) -> bool {
    if draft.is_empty() || history.is_empty() {
        return false;
    }
    let min_periods = GEMMA_CYCLE_GUARD_MIN_ESTABLISHED_PERIODS;
    let max_period = GEMMA_CYCLE_GUARD_MAX_PERIOD;
    for period in 1..=max_period {
        let need = period.saturating_mul(min_periods);
        if history.len() < need {
            continue;
        }
        let tail = &history[history.len() - need..];
        if !tail_is_exact_period(tail, period) {
            continue;
        }
        // Route the whole speculative window to sequential as soon as its first
        // token continues the cycle. Requiring every draft token to continue it
        // is fail-open: a later proposed break can hide an earlier looping token
        // that the multi-token verifier may accept incorrectly.
        if draft_begins_period_continuation(history, period, draft) {
            return true;
        }
    }
    false
}

/// Layer band whose long-context attention stays in f32 for dense Gemma MTP.
pub(super) fn gemma_sensitive_f32_layer_range(layer_count: usize) -> Option<(usize, usize)> {
    if layer_count == 0 {
        return None;
    }
    let start = layer_count.saturating_mul(7) / 12;
    let count = layer_count.div_ceil(6).min(layer_count - start).max(1);
    Some((start, count))
}

/// Build a short cycle-history view ending with `last_token` (the verify root).
///
/// `generated_tokens` may already include `last_token` (after emission) or may
/// lag by one token on the empty-draft → pending handoff. Returns the number of
/// tokens written into `buf` (≤ `buf.len()`).
pub(super) fn fill_gemma_cycle_history(
    generated: &[u32],
    last_token: u32,
    buf: &mut [u32],
) -> usize {
    if buf.is_empty() {
        return 0;
    }
    let include_last = generated.last() != Some(&last_token);
    let take_from_generated = if include_last {
        buf.len().saturating_sub(1).min(generated.len())
    } else {
        buf.len().min(generated.len())
    };
    let start = generated.len().saturating_sub(take_from_generated);
    let src = &generated[start..];
    buf[..src.len()].copy_from_slice(src);
    let mut n = src.len();
    if include_last && n < buf.len() {
        buf[n] = last_token;
        n += 1;
    }
    n
}

fn tail_is_exact_period(tail: &[u32], period: usize) -> bool {
    if period == 0 || tail.len() < period.saturating_mul(2) || !tail.len().is_multiple_of(period) {
        return false;
    }
    for i in period..tail.len() {
        if tail[i] != tail[i - period] {
            return false;
        }
    }
    true
}

fn draft_begins_period_continuation(history: &[u32], period: usize, draft: &[u32]) -> bool {
    if period == 0 || history.len() < period {
        return false;
    }
    let Some(&first_draft_token) = draft.first() else {
        return false;
    };
    first_draft_token == history[history.len() - period]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_greedy_verify_route_truth_table() {
        use GemmaGreedyVerifyRoute::{MultiTokenAdopt, SequentialOracle};
        // oracle_on wins regardless of other forces.
        assert_eq!(
            gemma_greedy_verify_route(true, true, true, false),
            SequentialOracle
        );
        assert_eq!(
            gemma_greedy_verify_route(true, false, false, false),
            SequentialOracle
        );
        // Cycle force under formal multi-token.
        assert_eq!(
            gemma_greedy_verify_route(false, true, true, false),
            SequentialOracle
        );
        // Early-gen force under formal multi-token.
        assert_eq!(
            gemma_greedy_verify_route(false, false, false, true),
            SequentialOracle
        );
        assert_eq!(
            gemma_greedy_verify_route(false, true, false, true),
            SequentialOracle
        );
        // No forces → multi-token adopt.
        assert_eq!(
            gemma_greedy_verify_route(false, true, false, false),
            MultiTokenAdopt
        );
        assert_eq!(
            gemma_greedy_verify_route(false, false, true, false),
            MultiTokenAdopt
        );
        assert_eq!(
            gemma_greedy_verify_route(false, false, false, false),
            MultiTokenAdopt
        );
    }

    #[test]
    fn gemma_early_gen_pure_direct_force_threshold() {
        assert!(gemma_early_gen_pure_direct_force(true, 0, 32));
        assert!(gemma_early_gen_pure_direct_force(true, 13, 32));
        assert!(gemma_early_gen_pure_direct_force(true, 31, 32));
        assert!(!gemma_early_gen_pure_direct_force(true, 32, 32));
        assert!(!gemma_early_gen_pure_direct_force(true, 100, 32));
        assert!(!gemma_early_gen_pure_direct_force(false, 0, 32));
    }

    #[test]
    fn gemma_sensitive_f32_layer_range_scales_with_depth() {
        assert_eq!(gemma_sensitive_f32_layer_range(0), None);
        assert_eq!(gemma_sensitive_f32_layer_range(1), Some((0, 1)));
        assert_eq!(gemma_sensitive_f32_layer_range(48), Some((28, 8)));
        assert_eq!(gemma_sensitive_f32_layer_range(62), Some((36, 11)));
    }

    #[test]
    fn draft_continues_period4_cycle_formal_pilot_shape() {
        // Established period-4 cycle (3 periods) + draft continuing it.
        // Multi-token would accept a cycle-continuation false accept at the
        // 5th draft token while sequential greedy would break the cycle.
        let history = [
            3574, 711, 1161, 496, // period 1
            3574, 711, 1161, 496, // period 2
            3574, 711, 1161, 496, // period 3 (established)
        ];
        let draft = [3574, 711, 1161, 496, 2633];
        // 2633 eventually breaks the period, but the speculative window starts
        // by continuing it. The later break must not hide the risky prefix.
        assert!(
            draft_continues_committed_cycle(&history, &draft),
            "a later proposed break must not hide an initial cycle continuation"
        );
        let continuing = [3574, 711, 1161, 496, 3574];
        assert!(draft_continues_committed_cycle(&history, &continuing));
    }

    #[test]
    fn draft_continues_period1_stuck_token_loop() {
        let history = [7, 7, 7, 7];
        assert!(draft_continues_committed_cycle(&history, &[7, 7]));
        assert!(draft_continues_committed_cycle(&history, &[7, 8]));
        assert!(!draft_continues_committed_cycle(&history, &[8, 7]));
    }

    #[test]
    fn non_repeating_history_does_not_trip_guard() {
        let history: Vec<u32> = (1..40).collect();
        let draft = [40, 41];
        assert!(!draft_continues_committed_cycle(&history, &draft));
    }

    #[test]
    fn single_unestablished_period_does_not_trip_guard() {
        // Only one period present — not "established".
        let history = [1, 2, 3, 4];
        let draft = [1, 2];
        assert!(!draft_continues_committed_cycle(&history, &draft));
    }

    #[test]
    fn cycle_orthogonal_draft_on_looping_history_preserves_multitoken() {
        let history = [10, 20, 10, 20, 10, 20];
        // Established period-2, but draft is orthogonal (speed path kept).
        assert!(!draft_continues_committed_cycle(&history, &[99, 100]));
        assert!(draft_continues_committed_cycle(&history, &[10, 20, 10]));
    }

    #[test]
    fn wrong_phase_draft_does_not_trip_guard() {
        let history = [1, 2, 3, 1, 2, 3];
        // Phase should continue at 1, not 2.
        assert!(!draft_continues_committed_cycle(&history, &[2, 3, 1]));
        assert!(draft_continues_committed_cycle(&history, &[1, 2, 3]));
    }

    #[test]
    fn period_above_max_is_ignored() {
        let period = GEMMA_CYCLE_GUARD_MAX_PERIOD + 1;
        let mut history = Vec::new();
        for _ in 0..2 {
            for i in 0..period {
                history.push(i as u32 + 1);
            }
        }
        let draft: Vec<u32> = (1..=period as u32).collect();
        assert!(!draft_continues_committed_cycle(&history, &draft));
    }

    #[test]
    fn empty_draft_or_short_history_is_safe() {
        assert!(!draft_continues_committed_cycle(&[1, 1, 1, 1], &[]));
        assert!(!draft_continues_committed_cycle(&[], &[1]));
        assert!(!draft_continues_committed_cycle(&[1], &[1]));
    }

    #[test]
    fn fill_gemma_cycle_history_appends_last_token_when_missing() {
        let mut buf = [0u32; 8];
        let n = fill_gemma_cycle_history(&[1, 2, 3], 4, &mut buf);
        assert_eq!(&buf[..n], &[1, 2, 3, 4]);
        let n = fill_gemma_cycle_history(&[1, 2, 4], 4, &mut buf);
        assert_eq!(&buf[..n], &[1, 2, 4]);
    }
}
