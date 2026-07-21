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
        || crate::mtp::mtp_draft_mode_from_env() != crate::mtp::MtpDraftMode::Greedy
    {
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
    session_direct || (request_ngram_disabled && !has_mtp) || mtp_uses_direct_pipeline
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
