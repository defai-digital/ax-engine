//! Per-step decode / MTP / KV-cache telemetry for the MLX runner.
//!
//! Split out of `runner/mod.rs` (Phase 2 slice 5 of the decode-dispatch
//! efficiency plan): the step-level counter blocks (`MtpTelemetry`,
//! `DecodeTelemetry`, `KvCacheTelemetry`), their route-decision exports, the
//! route-decision exports for the model profile snapshots, and the reusable
//! MTP target-prob workspace. Follows the `runner_telemetry.rs` convention
//! of inheriting the parent scope wholesale.

use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MtpTelemetry {
    pub(super) correctness_mode: MtpCorrectnessMode,
    pub(super) proposal_law: MtpProposalLaw,
    pub(super) correctness_mode_conflicts: u32,
    pub(super) proposal_law_conflicts: u32,
    pub(super) optimistic_steps: u32,
    pub(super) direct_fallback_steps: u32,
    pub(super) residual_correction_tokens: u32,
    pub(super) draft_tokens: u32,
    pub(super) accepted_tokens: u32,
    pub(super) decode_steps: u32,
    pub(super) full_accept_steps: u32,
    pub(super) partial_reject_steps: u32,
    pub(super) complete_miss_steps: u32,
    /// Cycle-level acceptance: accepted_cycles counts MTP verify cycles where
    /// ALL draft tokens were accepted; rejected_cycles counts cycles with any
    /// rejection.  Matches Lightning-MLX's `mtp_acceptance_ratio` metric
    /// (accepted / (accepted + rejected)) for fair cross-engine comparison.
    pub(super) accepted_cycles: u32,
    pub(super) rejected_cycles: u32,
    pub(super) cache_clone_wall_us: u32,
    pub(super) verify_forward_wall_us: u32,
    pub(super) verify_eval_wall_us: u32,
    pub(super) accept_wall_us: u32,
    pub(super) rollback_wall_us: u32,
    pub(super) tail_sample_wall_us: u32,
    pub(super) draft_wall_us: u32,
    pub(super) target_softmax_wall_us: u32,
    pub(super) verify_tokens: u32,
    pub(super) emitted_tokens: u32,
    pub(super) source_mtp_submitted_tokens: u32,
    pub(super) source_mtp_accepted_tokens: u32,
    pub(super) source_mtp_rejected_tokens: u32,
    pub(super) source_mtp_cascade_rejected_tokens: u32,
    pub(super) source_mtp_proposer_wall_us: u32,
    pub(super) source_assistant_submitted_tokens: u32,
    pub(super) source_assistant_accepted_tokens: u32,
    pub(super) source_assistant_rejected_tokens: u32,
    pub(super) source_assistant_cascade_rejected_tokens: u32,
    pub(super) source_assistant_proposer_wall_us: u32,
    pub(super) source_ngram_proposed_tokens: u32,
    pub(super) source_ngram_rejected_tokens: u32,
    pub(super) source_ngram_cascade_rejected_tokens: u32,
    pub(super) ngram_lookup_wall_us: u32,
    pub(super) accepted_by_depth: [u32; 3],
    pub(super) drafted_by_depth: [u32; 3],
    pub(super) draft_source_mtp_tokens: u32,
    pub(super) accepted_source_mtp_tokens: u32,
    pub(super) draft_source_ngram_tokens: u32,
    pub(super) accepted_source_ngram_tokens: u32,
    pub(super) draft_source_hybrid_mtp_tokens: u32,
    pub(super) accepted_source_hybrid_mtp_tokens: u32,
    pub(super) ngram_attempt_steps: u32,
    pub(super) ngram_hit_steps: u32,
    pub(super) ngram_no_candidate_steps: u32,
    pub(super) ngram_confidence_filtered_steps: u32,
    pub(super) ngram_cycle_guard_steps: u32,
    pub(super) ngram_skipped_mtp_steps: u32,
    pub(super) ngram_skipped_mtp_tokens: u32,
    pub(super) ngram_hybrid_tail_steps: u32,
    pub(super) ngram_hybrid_tail_tokens: u32,
    /// MTP n-gram stacking steps skipped because model is outside `<think>`.
    pub(super) ngram_think_gated_steps: u32,
    pub(super) accept_rate_ewma: f32,
    pub(super) accept_rate_ewma_samples: u32,
    /// MTP-only acceptance EWMA: tracks only MTP and HybridMtp sourced draft
    /// positions, excluding n-gram tokens.  Used by the saturation gate so
    /// that n-gram rejections cannot suppress gating when the model itself
    /// has near-perfect acceptance (e.g. 27B flappy at 99.5% accept rate).
    pub(super) mtp_only_accept_rate_ewma: f32,
    pub(super) mtp_only_accept_rate_ewma_samples: u32,
    pub(super) ngram_saturated_gated_steps: u32,
    /// Steps where n-gram was gated off because the combined accept rate fell
    /// below the MTP-only rate, indicating n-gram is actively hurting (cascade
    /// rejections of MTP tokens when n-gram fails at early positions).
    pub(super) ngram_hurt_gated_steps: u32,
    /// Steps gated by the new source-aware hurt gate (ADR-019 D3).
    pub(super) ngram_source_hurt_gated_steps: u32,
    /// Steps gated by the legacy EWMA hurt gate (ADR-018 D3), tracked separately
    /// for A/B comparison during the transition.
    pub(super) ngram_legacy_hurt_gated_steps: u32,
    pub(super) ngram_auto_disabled_steps: u32,
    pub(super) ngram_self_tune_disabled_steps: u32,
    pub(super) ngram_utility_gated_steps: u32,
    pub(super) ngram_utility_insufficient_sample_steps: u32,
    pub(super) ngram_safety_disabled_steps: u32,
    pub(super) ngram_safety_tightened_steps: u32,
    pub(super) ngram_safety_reason: u32,
    pub(super) ngram_submitted_tokens: u32,
    pub(super) ngram_submitted_accepted_tokens: u32,
    pub(super) utility_baseline_steps: u32,
    pub(super) utility_baseline_wall_us: u32,
    pub(super) utility_baseline_emitted_tokens: u32,
    pub(super) utility_stacked_steps: u32,
    pub(super) utility_stacked_wall_us: u32,
    pub(super) utility_stacked_emitted_tokens: u32,
    pub(super) utility_stacked_ngram_submitted_tokens: u32,
    /// Steps where auto-optimistic activated (EWMA ≥ 0.99 without env override).
    pub(super) auto_optimistic_steps: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MtpStepTimings {
    pub(super) cache_clone_wall_us: u32,
    pub(super) verify_forward_wall_us: u32,
    pub(super) verify_eval_wall_us: u32,
    pub(super) target_softmax_wall_us: u32,
    pub(super) accept_wall_us: u32,
    pub(super) rollback_wall_us: u32,
    pub(super) tail_sample_wall_us: u32,
    pub(super) draft_wall_us: u32,
    pub(super) mtp_draft_wall_us: u32,
    pub(super) assistant_draft_wall_us: u32,
    pub(super) ngram_lookup_wall_us: u32,
    pub(super) verify_tokens: u32,
    pub(super) emitted_tokens: u32,
    pub(super) ngram_submitted_tokens: u32,
}

impl MtpTelemetry {
    pub(super) fn record_correctness_mode(
        &mut self,
        correctness_mode: MtpCorrectnessMode,
        proposal_law: MtpProposalLaw,
    ) {
        if self.correctness_mode == MtpCorrectnessMode::Unknown {
            self.correctness_mode = correctness_mode;
        } else if self.correctness_mode != correctness_mode {
            self.correctness_mode_conflicts = self.correctness_mode_conflicts.saturating_add(1);
        }
        if proposal_law != MtpProposalLaw::Unknown {
            if self.proposal_law == MtpProposalLaw::Unknown {
                self.proposal_law = proposal_law;
            } else if self.proposal_law != proposal_law {
                self.proposal_law_conflicts = self.proposal_law_conflicts.saturating_add(1);
            }
        }
    }

    pub(super) fn record_direct_fallback(&mut self) {
        self.record_correctness_mode(MtpCorrectnessMode::DirectFallback, MtpProposalLaw::Unknown);
        self.direct_fallback_steps = self.direct_fallback_steps.saturating_add(1);
    }

    pub(super) fn record_optimistic_step(&mut self) {
        self.optimistic_steps = self.optimistic_steps.saturating_add(1);
    }

    pub(super) fn record_step(
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

    pub(super) fn record_ngram_attempt(&mut self, rejection: Option<NgramDraftRejection>) {
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

    pub(super) fn record_ngram_cycle_guard(&mut self) {
        self.ngram_cycle_guard_steps = self.ngram_cycle_guard_steps.saturating_add(1);
    }

    pub(super) fn record_ngram_stack_hit(&mut self, draft_len: usize, skipped_mtp: bool) {
        self.ngram_hit_steps = self.ngram_hit_steps.saturating_add(1);
        if skipped_mtp {
            self.ngram_skipped_mtp_steps = self.ngram_skipped_mtp_steps.saturating_add(1);
            self.ngram_skipped_mtp_tokens = self
                .ngram_skipped_mtp_tokens
                .saturating_add(saturating_u32(draft_len));
        }
    }

    pub(super) fn record_ngram_hybrid_tail(&mut self, tail_len: usize) {
        if tail_len == 0 {
            return;
        }
        self.ngram_hybrid_tail_steps = self.ngram_hybrid_tail_steps.saturating_add(1);
        self.ngram_hybrid_tail_tokens = self
            .ngram_hybrid_tail_tokens
            .saturating_add(saturating_u32(tail_len));
    }

    pub(super) fn record_ngram_proposed(&mut self, draft_len: usize) {
        self.source_ngram_proposed_tokens = self
            .source_ngram_proposed_tokens
            .saturating_add(saturating_u32(draft_len));
    }

    pub(super) fn record_ngram_submitted(&mut self, draft_len: usize) {
        self.ngram_submitted_tokens = self
            .ngram_submitted_tokens
            .saturating_add(saturating_u32(draft_len));
    }

    pub(super) fn record_ngram_verified(&mut self, accepted: usize) {
        self.ngram_submitted_accepted_tokens = self
            .ngram_submitted_accepted_tokens
            .saturating_add(saturating_u32(accepted));
    }

    pub(super) fn record_timings(&mut self, timings: MtpStepTimings) {
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

    pub(super) fn merge_from(&mut self, other: Self) {
        if self.correctness_mode == MtpCorrectnessMode::Unknown {
            self.correctness_mode = other.correctness_mode;
        } else if other.correctness_mode != MtpCorrectnessMode::Unknown
            && self.correctness_mode != other.correctness_mode
        {
            self.correctness_mode_conflicts = self.correctness_mode_conflicts.saturating_add(1);
        }
        if self.proposal_law == MtpProposalLaw::Unknown {
            self.proposal_law = other.proposal_law;
        } else if other.proposal_law != MtpProposalLaw::Unknown
            && self.proposal_law != other.proposal_law
        {
            self.proposal_law_conflicts = self.proposal_law_conflicts.saturating_add(1);
        }
        self.correctness_mode_conflicts = self
            .correctness_mode_conflicts
            .saturating_add(other.correctness_mode_conflicts);
        self.proposal_law_conflicts = self
            .proposal_law_conflicts
            .saturating_add(other.proposal_law_conflicts);
        self.optimistic_steps = self.optimistic_steps.saturating_add(other.optimistic_steps);
        self.direct_fallback_steps = self
            .direct_fallback_steps
            .saturating_add(other.direct_fallback_steps);
        self.residual_correction_tokens = self
            .residual_correction_tokens
            .saturating_add(other.residual_correction_tokens);
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

    pub(super) fn baseline_utility(&self) -> DraftSourceUtility {
        DraftSourceUtility {
            submitted_tokens: self.utility_baseline_emitted_tokens,
            proposer_wall_us: 0,
            verify_wall_us: self.utility_baseline_wall_us,
            emitted_tokens: self.utility_baseline_emitted_tokens,
        }
    }

    pub(super) fn stacked_utility(&self) -> DraftSourceUtility {
        DraftSourceUtility {
            submitted_tokens: self.utility_stacked_ngram_submitted_tokens,
            proposer_wall_us: 0,
            verify_wall_us: self.utility_stacked_wall_us,
            emitted_tokens: self.utility_stacked_emitted_tokens,
        }
    }

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
        let entries = [
            (
                "ax_mtp_correctness_mode",
                self.correctness_mode.route_code(),
            ),
            ("ax_mtp_proposal_law", self.proposal_law.route_code()),
            (
                "ax_mtp_correctness_mode_conflicts",
                self.correctness_mode_conflicts,
            ),
            ("ax_mtp_proposal_law_conflicts", self.proposal_law_conflicts),
            ("ax_mtp_optimistic_steps", self.optimistic_steps),
            ("ax_mtp_direct_fallback_steps", self.direct_fallback_steps),
            (
                "ax_mtp_residual_correction_tokens",
                self.residual_correction_tokens,
            ),
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
        decisions.upsert_route_decision(
            "ax_mtp_adaptive_gate_enabled",
            u32::from(adaptive_gate_enabled_from_env()),
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
pub(super) struct DecodeTelemetry {
    pub(super) prefill_steps: u32,
    pub(super) prefill_wall_us: u32,
    pub(super) prefill_forward_wall_us: u32,
    pub(super) prefill_prefix_cache_wall_us: u32,
    pub(super) prefill_generation_state_wall_us: u32,
    pub(super) decode_steps: u32,
    pub(super) decode_wall_us: u32,
    pub(super) direct_bootstrap_steps: u32,
    pub(super) direct_bootstrap_wall_us: u32,
    pub(super) direct_pipeline_steps: u32,
    pub(super) direct_pipeline_wall_us: u32,
    pub(super) direct_pipeline_forward_wall_us: u32,
    pub(super) direct_pipeline_forward_layer_loop_wall_us: u32,
    pub(super) direct_pipeline_forward_head_wall_us: u32,
    pub(super) direct_pipeline_argmax_wall_us: u32,
    pub(super) direct_pipeline_async_eval_wall_us: u32,
    pub(super) direct_pipeline_next_complete_wall_us: u32,
    pub(super) direct_pipeline_pending_eval_wall_us: u32,
    pub(super) direct_pipeline_pending_read_wall_us: u32,
    pub(super) direct_pipeline_op_count: u64,
    pub(super) direct_pipeline_linear_attention_layer_ops: u64,
    pub(super) direct_pipeline_linear_attention_layer_count: u32,
    pub(super) direct_pipeline_full_attention_layer_ops: u64,
    pub(super) direct_pipeline_full_attention_layer_count: u32,
    pub(super) single_decode_steps: u32,
    pub(super) single_decode_wall_us: u32,
    pub(super) ngram_decode_steps: u32,
    pub(super) ngram_decode_wall_us: u32,
    pub(super) bonus_tokens: u32,
    // W1 sync-count instrumentation (ADR 0017 §policy 1).
    // Counts blocking eval() calls by category so profiles can separate
    // production-path GPU barriers from prefill drains.
    pub(super) production_decode_evals: u32,
    pub(super) prefill_eval_barriers: u32,
    pub(super) prefill_drain_async_evals: u32,
    // DiffusionGemma block generation counters.
    pub(super) diffusion_blocks: u32,
    pub(super) diffusion_denoise_steps: u32,
    pub(super) diffusion_converged_blocks: u32,
    pub(super) diffusion_denoise_wall_us: u32,
    pub(super) diffusion_commit_wall_us: u32,
    pub(super) diffusion_block_wall_us: u32,
    // Per-criterion convergence signals (0 or 1 per block).
    pub(super) diffusion_converged_strict: u32,
    pub(super) diffusion_converged_acceptance: u32,
    pub(super) diffusion_converged_plateau: u32,
    // Near-miss telemetry: lowest entropy/acceptance rate observed (×10000 fixed-point).
    pub(super) diffusion_min_entropy_bp: u32,
    pub(super) diffusion_min_acceptance_rate_bp: u32,
    pub(super) diffusion_commit_skipped: u32,
    pub(super) diffusion_full_pipeline_used: u32,
    pub(super) diffusion_kv_buffer_used: u32,
    /// Last prefill chunk size selected for this request (tokens).
    pub(super) prefill_chunk_selected: u32,
    /// 0 = cold (`seq_len == 0`), 1 = warm-extend (`seq_len > 0`).
    pub(super) prefill_chunk_mode: u32,
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
            prefill_chunk_selected: 0,
            prefill_chunk_mode: 0,
        }
    }
}

impl DecodeTelemetry {
    pub(super) fn record_prefill(&mut self, wall_us: u32) {
        self.prefill_steps = self.prefill_steps.saturating_add(1);
        self.prefill_wall_us = self.prefill_wall_us.saturating_add(wall_us);
    }

    pub(super) fn record_prefill_chunk_selection(&mut self, chunk: usize, warm_extend: bool) {
        self.prefill_chunk_selected = saturating_u32(chunk);
        self.prefill_chunk_mode = u32::from(warm_extend);
    }

    pub(super) fn record_prefill_breakdown(
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

    pub(super) fn record_decode(&mut self, wall_us: u32) {
        self.decode_steps = self.decode_steps.saturating_add(1);
        self.decode_wall_us = self.decode_wall_us.saturating_add(wall_us);
    }

    pub(super) fn record_direct_bootstrap(&mut self, wall_us: u32) {
        self.direct_bootstrap_steps = self.direct_bootstrap_steps.saturating_add(1);
        self.direct_bootstrap_wall_us = self.direct_bootstrap_wall_us.saturating_add(wall_us);
    }

    pub(super) fn record_direct_pipeline(&mut self, wall_us: u32) {
        self.direct_pipeline_steps = self.direct_pipeline_steps.saturating_add(1);
        self.direct_pipeline_wall_us = self.direct_pipeline_wall_us.saturating_add(wall_us);
    }

    pub(super) fn record_direct_pipeline_op_count(&mut self, ops: u64) {
        self.direct_pipeline_op_count = self.direct_pipeline_op_count.saturating_add(ops);
    }

    pub(super) fn record_direct_pipeline_timings(&mut self, timings: DirectPipelineTimings) {
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

    pub(super) fn record_single_decode(&mut self, wall_us: u32) {
        self.single_decode_steps = self.single_decode_steps.saturating_add(1);
        self.single_decode_wall_us = self.single_decode_wall_us.saturating_add(wall_us);
    }

    pub(super) fn record_ngram_decode(&mut self, wall_us: u32) {
        self.ngram_decode_steps = self.ngram_decode_steps.saturating_add(1);
        self.ngram_decode_wall_us = self.ngram_decode_wall_us.saturating_add(wall_us);
    }

    pub(super) fn record_bonus_token(&mut self) {
        self.bonus_tokens = self.bonus_tokens.saturating_add(1);
    }

    pub(super) fn record_production_decode_eval(&mut self) {
        self.production_decode_evals = self.production_decode_evals.saturating_add(1);
    }

    pub(super) fn record_prefill_eval_barrier(&mut self) {
        self.prefill_eval_barriers = self.prefill_eval_barriers.saturating_add(1);
    }

    pub(super) fn record_prefill_drain_async_evals(&mut self, count: u32) {
        self.prefill_drain_async_evals = self.prefill_drain_async_evals.saturating_add(count);
    }

    pub(super) fn record_diffusion_block(
        &mut self,
        result: &crate::diffusion::DiffusionBlockResult,
    ) {
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

    pub(super) fn merge_from(&mut self, other: Self) {
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
        // Last-writer wins for chunk selection (most recent prefill on the request).
        if other.prefill_steps > 0 {
            self.prefill_chunk_selected = other.prefill_chunk_selected;
            self.prefill_chunk_mode = other.prefill_chunk_mode;
        }
    }

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
            ("ax_mlx_prefill_chunk_selected", self.prefill_chunk_selected),
            ("ax_mlx_prefill_chunk_mode", self.prefill_chunk_mode),
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
    pub(super) fn merge_from(&mut self, other: Self) {
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

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
    pub(super) fn merge_from(&mut self, other: Self) {
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

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
    pub(super) fn merge_from(&mut self, other: Self) {
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

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
    pub(super) fn merge_from(&mut self, other: Self) {
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

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
    pub(super) fn merge_from(&mut self, other: Self) {
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

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
    pub(super) fn merge_from(&mut self, other: Self) {
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

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
pub(super) struct KvCacheTelemetry {
    pub(super) request_snapshots: u32,
    pub(super) logical_tokens: u64,
    pub(super) capacity_tokens: u64,
    pub(super) logical_bytes: u64,
    pub(super) capacity_bytes: u64,
    pub(super) full_attention_layers: u64,
    pub(super) sliding_window_layers: u64,
    pub(super) sliding_window_retained_tokens: u64,
    pub(super) sliding_window_reclaimable_capacity_tokens: u64,
    pub(super) sliding_window_reclaimable_capacity_bytes: u64,
    /// Peak rotated-ring layer count across this step's request snapshots
    /// (gauge, max-merged — layer counts are per-request, not additive).
    pub(super) rotated_ring_layers_max: u64,
    /// Bounded-rollback ring slack observed on rotating requests (gauge).
    pub(super) rotating_ring_slack: u64,
    pub(super) linear_state_layers: u64,
    pub(super) linear_state_bytes: u64,
    pub(super) growth_count: u64,
    pub(super) paged_materialize_us: u64,
    pub(super) paged_pool_exhaustion_fallbacks: u64,
}

impl KvCacheTelemetry {
    pub(super) fn merge_from(&mut self, usage: MlxKVCacheUsage) {
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
        self.rotated_ring_layers_max = self
            .rotated_ring_layers_max
            .max(usage.rotated_ring_layers as u64);
        if usage.rotated_ring_layers > 0 {
            self.rotating_ring_slack = usage.rotating_ring_slack as u64;
        }
        self.linear_state_layers = self
            .linear_state_layers
            .saturating_add(usage.linear_state_layers as u64);
        self.linear_state_bytes = self
            .linear_state_bytes
            .saturating_add(usage.linear_state_bytes);
        self.growth_count = self.growth_count.saturating_add(usage.growth_count);
        self.paged_materialize_us = self
            .paged_materialize_us
            .saturating_add(usage.paged_materialize_us);
        self.paged_pool_exhaustion_fallbacks = self
            .paged_pool_exhaustion_fallbacks
            .saturating_add(usage.paged_pool_exhaustion_fallbacks);
    }

    pub(super) fn append_route_decisions(&self, decisions: &mut impl RouteDecisionSink) {
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
                ROUTE_DECISION_AX_MLX_KV_ROTATED_RING_LAYERS,
                saturating_u32_from_u64(self.rotated_ring_layers_max),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_ROTATING_RING_SLACK,
                saturating_u32_from_u64(self.rotating_ring_slack),
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
            (
                ROUTE_DECISION_AX_MLX_KV_PAGED_MATERIALIZE_US,
                saturating_u32_from_u64(self.paged_materialize_us),
            ),
            (
                ROUTE_DECISION_AX_MLX_KV_PAGED_POOL_EXHAUSTION_FALLBACKS,
                saturating_u32_from_u64(self.paged_pool_exhaustion_fallbacks),
            ),
        ];

        for (key, value) in entries {
            decisions.upsert_route_decision(key, value);
        }
    }
}

#[derive(Default)]
pub(super) struct MtpTargetProbWorkspace {
    pub(super) flat_indices: Vec<i32>,
    pub(super) target_probs: Vec<f32>,
    pub(super) target_candidates: Vec<(u32, f32)>,
}
