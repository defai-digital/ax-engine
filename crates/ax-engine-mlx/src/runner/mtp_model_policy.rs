//! Model-scoped MTP policy selected once when a runner is constructed.
//!
//! Keep family classification and capability-specific defaults here instead of
//! passing bare gate values around the runner. Qwen linear-attention MTP stays
//! on direct decode until its *end-to-end* acceleration contract is certified
//! (ADR-020): tensor eligibility alone is not evidence that a batched verifier
//! preserves the canonical singleton greedy stream. An explicit
//! certification-candidate opt-in exposes that route to the formal harness
//! without promoting it for normal users. GLM and Gemma retain their
//! independently calibrated policies, and incompatible drafters fail closed.

use super::mtp_tuning::MtpRuntimeCertification;
use super::pipeline::RouteDecisionSink;

const CERTIFICATION_DEPTH_ONE_GATE: f32 = 0.0;
const QWEN_LINEAR_CERTIFICATION_CANDIDATE_ENV: &str =
    "AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE";
const DEEPSEEK_V4_MTP_CERTIFICATION_CANDIDATE_ENV: &str =
    "AX_MLX_DEEPSEEK_V4_MTP_CERTIFICATION_CANDIDATE";

fn truthy_opt_in(raw: &str) -> bool {
    let value = raw.trim();
    value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
}

/// Explicitly expose the uncertified Qwen linear-MTP route to a formal test run.
///
/// This switch cannot bypass the loaded-model exact-capability check and is
/// intentionally separate from the exact-arithmetic selector: arithmetic
/// eligibility is necessary, but it is not an end-to-end acceleration claim.
pub(super) fn qwen_linear_mtp_certification_candidate_from_env() -> bool {
    std::env::var(QWEN_LINEAR_CERTIFICATION_CANDIDATE_ENV)
        .ok()
        .is_some_and(|raw| truthy_opt_in(&raw))
}

/// Decide whether the loaded Qwen linear model may take the certified MTP
/// route.
///
/// Capability is independent of n-gram/direct session flags: the SDK
/// constructs with n-gram disabled and later enables MTP via
/// `set_mtp_requested`. Route safety must already be true then, or the
/// setter is a no-op and "MTP" silently stays direct. Activation remains
/// `mtp_requested` / `AX_NO_SPEC`. Env opt-in still wins when the exact
/// profile is off. The explicit projected-replay profile is also a verifier
/// candidate, but remains default-off and must still be requested by the
/// session or formal harness.
pub(super) fn resolve_qwen_linear_certification_candidate(
    env_opt_in: bool,
    exact_profile_enabled: bool,
) -> bool {
    env_opt_in || exact_profile_enabled
}

/// Prefill should materialize MTP hidden history only when decode will consume
/// it. Sidecar presence alone is not enough: AXQ packs ship `mtp.safetensors`
/// and `--ax-direct` must keep the regular last-token prefill path.
pub(super) fn should_capture_qwen_mtp_prefill_history(
    mtp_requested: bool,
    has_mtp_weights: bool,
) -> bool {
    mtp_requested && has_mtp_weights
}

/// Explicitly expose the experimental DeepSeek V4 nextn MTP route to a formal
/// harness. Product default stays direct-fallback until Tier 2 evidence exists
/// (docs: no MTP acceptance-rate claim).
pub(super) fn deepseek_v4_mtp_certification_candidate_from_env() -> bool {
    std::env::var(DEEPSEEK_V4_MTP_CERTIFICATION_CANDIDATE_ENV)
        .ok()
        .is_some_and(|raw| truthy_opt_in(&raw))
}

/// Stable route code describing the loaded model's MTP policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MtpModelPolicyKind {
    None,
    QwenCalibrated,
    /// Formal-harness-only depth-1 exact route (route code 2; stable telemetry).
    QwenLinearCertificationCandidateDepthOne,
    /// Formal-harness-only multi-depth exact route (route code 3).
    QwenLinearCertificationCandidateMultiDepth,
    /// Product default for linear Qwen until Tier 2 promotion (route code 4).
    QwenLinearUncertifiedDirectFallback,
    GlmCalibrated,
    Gemma4AssistantCalibrated,
    /// Formal-harness-only DeepSeek V4 nextn route (opt-in).
    DeepseekV4CertificationCandidate,
    /// Product default until Tier 2 evidence: MTP attached but not requested.
    DeepseekV4UncertifiedDirectFallback,
    ConflictingDrafters,
}

impl MtpModelPolicyKind {
    const fn route_code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::QwenCalibrated => 1,
            Self::QwenLinearCertificationCandidateDepthOne => 2,
            Self::QwenLinearCertificationCandidateMultiDepth => 3,
            Self::QwenLinearUncertifiedDirectFallback => 4,
            Self::GlmCalibrated => 5,
            Self::Gemma4AssistantCalibrated => 6,
            Self::ConflictingDrafters => 7,
            // Stable code 8: previously "DeepseekV4Calibrated"; now the
            // certification-candidate path only (fail-closed by default).
            Self::DeepseekV4CertificationCandidate => 8,
            Self::DeepseekV4UncertifiedDirectFallback => 9,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GateResolverFamily {
    Qwen,
    Glm,
}

/// Inputs derived exclusively from the validated, loaded runtime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MtpModelPolicyInputs {
    pub(super) qwen_depth: Option<usize>,
    pub(super) glm_depth: Option<usize>,
    pub(super) gemma4_assistant_depth: Option<usize>,
    pub(super) deepseek_v4_depth: Option<usize>,
    pub(super) deepseek_v4_certification_candidate: bool,
    pub(super) qwen_linear_attention: bool,
    pub(super) qwen_linear_exact_enabled: bool,
    pub(super) qwen_linear_projected_replay_enabled: bool,
    pub(super) qwen_linear_certification_candidate: bool,
    /// The explicit `AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE` opt-in,
    /// kept separate from the resolved candidate bit above (which the
    /// exact-profile auto-default also satisfies): only the env form keeps
    /// default-on MTP for an uncertified pack (formal-harness recipe).
    pub(super) qwen_linear_certification_env_opt_in: bool,
    /// Publisher-declared MTP certification from `axquant_runtime.json`.
    pub(super) runtime_certification: MtpRuntimeCertification,
}

/// Immutable policy snapshot owned by one [`super::MlxRunner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MtpModelPolicy {
    kind: MtpModelPolicyKind,
    max_depth: usize,
    qwen_linear_certification_env_opt_in: bool,
    runtime_certification: MtpRuntimeCertification,
}

impl MtpModelPolicy {
    pub(super) fn from_loaded(inputs: MtpModelPolicyInputs) -> Self {
        let (kind, max_depth) = Self::classify(inputs);
        Self {
            kind,
            max_depth,
            qwen_linear_certification_env_opt_in: inputs.qwen_linear_certification_env_opt_in,
            runtime_certification: inputs.runtime_certification,
        }
    }

    fn classify(inputs: MtpModelPolicyInputs) -> (MtpModelPolicyKind, usize) {
        let attachment_count = [
            inputs.qwen_depth.is_some(),
            inputs.glm_depth.is_some(),
            inputs.gemma4_assistant_depth.is_some(),
            inputs.deepseek_v4_depth.is_some(),
        ]
        .into_iter()
        .filter(|attached| *attached)
        .count();

        if attachment_count > 1 {
            return (
                MtpModelPolicyKind::ConflictingDrafters,
                inputs
                    .qwen_depth
                    .into_iter()
                    .chain(inputs.glm_depth)
                    .chain(inputs.gemma4_assistant_depth)
                    .chain(inputs.deepseek_v4_depth)
                    .max()
                    .unwrap_or(0),
            );
        }

        if let Some(max_depth) = inputs.qwen_depth {
            let kind = if !inputs.qwen_linear_attention {
                MtpModelPolicyKind::QwenCalibrated
            } else if (inputs.qwen_linear_exact_enabled
                || inputs.qwen_linear_projected_replay_enabled)
                && inputs.qwen_linear_certification_candidate
                && max_depth == 1
            {
                MtpModelPolicyKind::QwenLinearCertificationCandidateDepthOne
            } else if (inputs.qwen_linear_exact_enabled
                || inputs.qwen_linear_projected_replay_enabled)
                && inputs.qwen_linear_certification_candidate
            {
                MtpModelPolicyKind::QwenLinearCertificationCandidateMultiDepth
            } else {
                // Tensor eligibility does not prove that the batched verifier
                // is sequence-equivalent to the production singleton graph.
                // Promotion requires shipped runner-level golden evidence and
                // M5 performance gates (ADR-020), not an environment override
                // alone.
                MtpModelPolicyKind::QwenLinearUncertifiedDirectFallback
            };
            return (kind, max_depth);
        }

        if let Some(max_depth) = inputs.glm_depth {
            return (MtpModelPolicyKind::GlmCalibrated, max_depth);
        }

        if let Some(max_depth) = inputs.gemma4_assistant_depth {
            return (MtpModelPolicyKind::Gemma4AssistantCalibrated, max_depth);
        }

        if let Some(max_depth) = inputs.deepseek_v4_depth {
            let kind = if inputs.deepseek_v4_certification_candidate {
                MtpModelPolicyKind::DeepseekV4CertificationCandidate
            } else {
                // Tensor eligibility is not end-to-end acceleration evidence.
                MtpModelPolicyKind::DeepseekV4UncertifiedDirectFallback
            };
            return (kind, max_depth);
        }

        (MtpModelPolicyKind::None, 0)
    }

    /// Hard safety gate for model-based speculation.
    pub(super) const fn route_safe(self) -> bool {
        !matches!(
            self.kind,
            MtpModelPolicyKind::QwenLinearUncertifiedDirectFallback
                | MtpModelPolicyKind::DeepseekV4UncertifiedDirectFallback
                | MtpModelPolicyKind::ConflictingDrafters
        )
    }

    pub(super) const fn has_attached_drafter(self) -> bool {
        !matches!(self.kind, MtpModelPolicyKind::None)
    }

    pub(super) const fn max_depth(self) -> usize {
        self.max_depth
    }

    pub(super) const fn is_qwen_calibrated(self) -> bool {
        matches!(self.kind, MtpModelPolicyKind::QwenCalibrated)
    }

    /// Release gate for *default-on* model MTP. Every Qwen route that can be
    /// reached without an explicit env opt-in additionally requires publisher
    /// runtime certification (the `axquant_runtime.json` `"mtp"` block:
    /// `enabled_by_default` plus either an `optimized` stamp or a measured
    /// speedup >= 1.0x):
    ///
    /// - `QwenCalibrated` (dense Qwen, route 1) is default-on today.
    /// - `QwenLinearCertificationCandidate*` (routes 2/3) are ALSO reached by
    ///   default on every exact-eligible linear pack, because the exact
    ///   profile auto-enables and `resolve_qwen_linear_certification_candidate`
    ///   accepts the exact profile alone — the Qwen3.8-27B AXQ family
    ///   (including the 0.96x 6-bit flagship) lives here, not on route 1.
    ///   The explicit `AX_MLX_QWEN_LINEAR_MTP_CERTIFICATION_CANDIDATE` env
    ///   opt-in keeps default-on for the formal harness.
    ///
    /// Every other kind keeps its existing default: GLM/Gemma4 policies were
    /// calibrated independently and their packs ship no AXQuant `mtp` block,
    /// the DeepSeek V4 candidate kind is reachable only via env, fallback and
    /// conflicting kinds are already route-blocked, and `None` has nothing to
    /// gate. Explicit requests (`set_mtp_requested`, `MlxMtpPolicy::Required`)
    /// are unaffected.
    pub(super) const fn certified_default_on(self) -> bool {
        if self.is_qwen_calibrated() {
            return self.runtime_certification.default_on;
        }
        match self.kind {
            MtpModelPolicyKind::QwenLinearCertificationCandidateDepthOne
            | MtpModelPolicyKind::QwenLinearCertificationCandidateMultiDepth => {
                self.runtime_certification.default_on || self.qwen_linear_certification_env_opt_in
            }
            _ => true,
        }
    }

    pub(super) const fn is_qwen_linear_direct_fallback(self) -> bool {
        matches!(
            self.kind,
            MtpModelPolicyKind::QwenLinearUncertifiedDirectFallback
        )
    }

    pub(super) const fn is_deepseek_v4_direct_fallback(self) -> bool {
        matches!(
            self.kind,
            MtpModelPolicyKind::DeepseekV4UncertifiedDirectFallback
        )
    }

    pub(super) const fn is_deepseek_v4_certification_candidate(self) -> bool {
        matches!(
            self.kind,
            MtpModelPolicyKind::DeepseekV4CertificationCandidate
        )
    }

    pub(super) const fn is_qwen_linear_certification_candidate(self) -> bool {
        matches!(
            self.kind,
            MtpModelPolicyKind::QwenLinearCertificationCandidateDepthOne
                | MtpModelPolicyKind::QwenLinearCertificationCandidateMultiDepth
        )
    }

    pub(super) const fn has_conflicting_drafters(self) -> bool {
        matches!(self.kind, MtpModelPolicyKind::ConflictingDrafters)
    }

    /// Certified model default for Qwen's shared confidence resolver.
    pub(super) const fn qwen_gate_default(self) -> Option<f32> {
        self.gate_default_for(GateResolverFamily::Qwen)
    }

    /// Certified model default for GLM's shared confidence resolver.
    ///
    /// GLM currently has no model-specific override, so it retains the global
    /// calibrated default. Keeping this family-specific accessor prevents a
    /// future Qwen default from leaking into the GLM branch.
    pub(super) const fn glm_gate_default(self) -> Option<f32> {
        self.gate_default_for(GateResolverFamily::Glm)
    }

    const fn gate_default_for(self, family: GateResolverFamily) -> Option<f32> {
        match (self.kind, family) {
            (
                MtpModelPolicyKind::QwenLinearCertificationCandidateDepthOne,
                GateResolverFamily::Qwen,
            ) => Some(CERTIFICATION_DEPTH_ONE_GATE),
            _ => None,
        }
    }

    const fn model_gate_default(self) -> Option<f32> {
        match self.kind {
            MtpModelPolicyKind::QwenCalibrated
            | MtpModelPolicyKind::QwenLinearCertificationCandidateDepthOne
            | MtpModelPolicyKind::QwenLinearCertificationCandidateMultiDepth
            | MtpModelPolicyKind::QwenLinearUncertifiedDirectFallback => self.qwen_gate_default(),
            MtpModelPolicyKind::GlmCalibrated => self.glm_gate_default(),
            MtpModelPolicyKind::None
            | MtpModelPolicyKind::Gemma4AssistantCalibrated
            | MtpModelPolicyKind::DeepseekV4CertificationCandidate
            | MtpModelPolicyKind::DeepseekV4UncertifiedDirectFallback
            | MtpModelPolicyKind::ConflictingDrafters => None,
        }
    }

    pub(super) fn append_route_decisions(
        self,
        mtp_requested: bool,
        decisions: &mut impl RouteDecisionSink,
    ) {
        let model_default = self.model_gate_default();
        decisions.upsert_route_decision("ax_mlx_mtp_model_policy", self.kind.route_code());
        decisions.upsert_route_decision(
            "ax_mlx_mtp_model_policy_depth",
            u32::try_from(self.max_depth).unwrap_or(u32::MAX),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_model_policy_route_safe",
            u32::from(self.route_safe()),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_model_policy_active",
            u32::from(
                mtp_requested
                    && self.route_safe()
                    && !matches!(self.kind, MtpModelPolicyKind::None),
            ),
        );
        decisions.upsert_route_decision(
            "ax_mlx_qwen_linear_mtp_certification_candidate",
            u32::from(self.is_qwen_linear_certification_candidate()),
        );
        decisions.upsert_route_decision(
            "ax_mlx_deepseek_v4_mtp_certification_candidate",
            u32::from(self.is_deepseek_v4_certification_candidate()),
        );
        decisions.upsert_route_decision(
            "ax_mlx_deepseek_v4_mtp_direct_fallback",
            u32::from(self.is_deepseek_v4_direct_fallback()),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_model_gate_default_present",
            u32::from(model_default.is_some()),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_model_gate_default_x1000",
            model_default.map_or(0, |gate| (gate.clamp(0.0, 1.0) * 1000.0) as u32),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_certified_default_on",
            u32::from(self.certified_default_on()),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_runtime_enabled_by_default",
            u32::from(self.runtime_certification.enabled_by_default),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_runtime_optimized",
            u32::from(self.runtime_certification.optimized),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_runtime_measured_speedup_x1000",
            self.runtime_certification
                .measured_speedup_x1000
                .unwrap_or(0),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn exact_profile_admits_certified_route_independently_of_session_flags() {
        assert!(resolve_qwen_linear_certification_candidate(false, true));
        assert!(resolve_qwen_linear_certification_candidate(true, false));
        assert!(resolve_qwen_linear_certification_candidate(true, true));
        assert!(!resolve_qwen_linear_certification_candidate(false, false));
    }

    #[test]
    fn direct_session_does_not_capture_mtp_prefill_history() {
        assert!(!should_capture_qwen_mtp_prefill_history(false, true));
        assert!(!should_capture_qwen_mtp_prefill_history(true, false));
        assert!(!should_capture_qwen_mtp_prefill_history(false, false));
        assert!(should_capture_qwen_mtp_prefill_history(true, true));
    }

    #[test]
    fn certification_candidate_opt_in_is_strictly_truthy() {
        for enabled in ["1", "true", "TRUE", "yes", " Yes "] {
            assert!(truthy_opt_in(enabled));
        }
        for disabled in ["", "0", "false", "no", "candidate", "2"] {
            assert!(!truthy_opt_in(disabled));
        }
    }

    fn policy(
        qwen_depth: Option<usize>,
        glm_depth: Option<usize>,
        gemma_depth: Option<usize>,
        qwen_linear: bool,
        qwen_exact: bool,
        certification_candidate: bool,
    ) -> MtpModelPolicy {
        MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            qwen_depth,
            glm_depth,
            gemma4_assistant_depth: gemma_depth,
            qwen_linear_attention: qwen_linear,
            qwen_linear_exact_enabled: qwen_exact,
            qwen_linear_certification_candidate: certification_candidate,
            ..Default::default()
        })
    }

    fn certification(default_on: bool) -> MtpRuntimeCertification {
        MtpRuntimeCertification {
            enabled_by_default: true,
            optimized: default_on,
            measured_speedup_x1000: None,
            default_on,
        }
    }

    #[test]
    fn certified_default_on_gates_default_reachable_qwen_routes() {
        // QwenCalibrated (dense Qwen, route 1) follows the pack certification.
        for (default_on, expect) in [(false, false), (true, true)] {
            let dense_qwen = MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
                qwen_depth: Some(1),
                runtime_certification: certification(default_on),
                ..Default::default()
            });
            assert!(dense_qwen.is_qwen_calibrated());
            assert!(dense_qwen.route_safe());
            assert_eq!(dense_qwen.certified_default_on(), expect);
        }

        // The linear certification-candidate routes 2/3 are reached WITHOUT
        // env on every exact-eligible pack (the Qwen3.8 AXQ family — incl.
        // the 0.96x 6-bit flagship — lives here), so they are gated too.
        for max_depth in [1, 2] {
            let auto_exact = MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
                qwen_depth: Some(max_depth),
                qwen_linear_attention: true,
                qwen_linear_exact_enabled: true,
                qwen_linear_certification_candidate: true,
                qwen_linear_certification_env_opt_in: false,
                ..Default::default()
            });
            assert!(auto_exact.route_safe());
            assert!(
                !auto_exact.certified_default_on(),
                "exact-auto candidate route must be gated without env opt-in"
            );

            // Pack certification turns default-on back on for the same route.
            let certified = MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
                qwen_depth: Some(max_depth),
                qwen_linear_attention: true,
                qwen_linear_exact_enabled: true,
                qwen_linear_certification_candidate: true,
                runtime_certification: certification(true),
                ..Default::default()
            });
            assert!(certified.certified_default_on());

            // The explicit env opt-in preserves the formal-harness recipe on
            // an uncertified pack.
            let env_opt_in = MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
                qwen_depth: Some(max_depth),
                qwen_linear_attention: true,
                qwen_linear_exact_enabled: true,
                qwen_linear_certification_candidate: true,
                qwen_linear_certification_env_opt_in: true,
                ..Default::default()
            });
            assert!(env_opt_in.certified_default_on());
        }

        // Every other kind keeps its existing default (gate returns true).
        let others = [
            policy(None, None, None, false, false, false), // None
            policy(None, Some(1), None, false, false, false), // GLM
            policy(None, None, Some(1), false, false, false), // Gemma4
            policy(Some(1), None, None, true, false, false), // linear fallback (route-blocked)
            policy_v4(Some(1)),                            // V4 fallback (route-blocked)
        ];
        for other in others {
            assert!(!other.is_qwen_calibrated());
            assert!(
                other.certified_default_on(),
                "ungated kinds must keep their existing default: {other:?}"
            );
        }
    }

    #[test]
    fn route_telemetry_exposes_runtime_certification() {
        let mut decisions = Vec::new();
        MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            qwen_depth: Some(1),
            runtime_certification: MtpRuntimeCertification {
                enabled_by_default: true,
                optimized: false,
                measured_speedup_x1000: Some(960),
                default_on: false,
            },
            ..Default::default()
        })
        .append_route_decisions(false, &mut decisions);
        let map = decisions.into_iter().collect::<BTreeMap<_, _>>();
        assert_eq!(map.get("ax_mlx_mtp_certified_default_on"), Some(&0));
        assert_eq!(map.get("ax_mlx_mtp_runtime_enabled_by_default"), Some(&1));
        assert_eq!(map.get("ax_mlx_mtp_runtime_optimized"), Some(&0));
        assert_eq!(
            map.get("ax_mlx_mtp_runtime_measured_speedup_x1000"),
            Some(&960)
        );
    }

    fn policy_v4(v4_depth: Option<usize>) -> MtpModelPolicy {
        MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            deepseek_v4_depth: v4_depth,
            ..Default::default()
        })
    }

    #[test]
    fn linear_exact_without_candidate_fails_closed() {
        let fallback = policy(Some(1), None, None, true, true, false);
        assert_eq!(
            fallback.kind,
            MtpModelPolicyKind::QwenLinearUncertifiedDirectFallback
        );
        assert!(!fallback.route_safe());
        assert!(fallback.is_qwen_linear_direct_fallback());
        assert!(!fallback.is_qwen_linear_certification_candidate());
        assert_eq!(fallback.qwen_gate_default(), None);
    }

    #[test]
    fn certification_candidate_depth_one_is_route_safe() {
        let candidate = policy(Some(1), None, None, true, true, true);
        assert_eq!(
            candidate.kind,
            MtpModelPolicyKind::QwenLinearCertificationCandidateDepthOne
        );
        assert_eq!(candidate.qwen_gate_default(), Some(0.0));
        assert_eq!(candidate.glm_gate_default(), None);
        assert!(candidate.route_safe());
        assert!(candidate.is_qwen_linear_certification_candidate());
        assert!(candidate.has_attached_drafter());
        assert_eq!(candidate.max_depth(), 1);
    }

    #[test]
    fn projected_replay_candidate_is_route_safe_without_exact_profile() {
        let candidate = MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            qwen_depth: Some(3),
            qwen_linear_attention: true,
            qwen_linear_exact_enabled: false,
            qwen_linear_projected_replay_enabled: true,
            qwen_linear_certification_candidate: true,
            ..Default::default()
        });

        assert_eq!(
            candidate.kind,
            MtpModelPolicyKind::QwenLinearCertificationCandidateMultiDepth
        );
        assert!(candidate.route_safe());
        assert!(!candidate.certified_default_on());
    }

    #[test]
    fn every_supported_mtp_family_has_an_explicit_policy() {
        assert_eq!(
            policy(Some(2), None, None, false, false, false).kind,
            MtpModelPolicyKind::QwenCalibrated
        );
        assert_eq!(
            policy(Some(2), None, None, true, true, true).kind,
            MtpModelPolicyKind::QwenLinearCertificationCandidateMultiDepth
        );
        assert_eq!(
            policy(Some(1), None, None, true, false, true).kind,
            MtpModelPolicyKind::QwenLinearUncertifiedDirectFallback
        );
        assert_eq!(
            policy(None, Some(2), None, false, false, false).kind,
            MtpModelPolicyKind::GlmCalibrated
        );
        assert_eq!(
            policy(None, None, Some(2), false, false, false).kind,
            MtpModelPolicyKind::Gemma4AssistantCalibrated
        );
        // Product default: attached but fail-closed (not route-safe).
        assert_eq!(
            policy_v4(Some(1)).kind,
            MtpModelPolicyKind::DeepseekV4UncertifiedDirectFallback
        );
        let v4 = policy_v4(Some(1));
        assert!(!v4.route_safe());
        assert!(v4.is_deepseek_v4_direct_fallback());
        assert!(!v4.is_deepseek_v4_certification_candidate());
        assert!(v4.has_attached_drafter());
        assert_eq!(v4.max_depth(), 1);
        assert_eq!(v4.qwen_gate_default(), None);
        assert_eq!(v4.glm_gate_default(), None);
        assert!(!policy_v4(None).has_attached_drafter());

        let candidate = MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            deepseek_v4_depth: Some(1),
            deepseek_v4_certification_candidate: true,
            ..Default::default()
        });
        assert_eq!(
            candidate.kind,
            MtpModelPolicyKind::DeepseekV4CertificationCandidate
        );
        assert!(candidate.route_safe());
        assert!(candidate.is_deepseek_v4_certification_candidate());
    }

    #[test]
    fn conflicting_drafters_fail_closed() {
        for conflict in [
            policy(Some(1), Some(1), None, true, true, true),
            policy(Some(1), None, Some(1), true, true, true),
            policy(None, Some(1), Some(1), false, false, false),
            MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
                deepseek_v4_depth: Some(1),
                glm_depth: Some(1),
                ..Default::default()
            }),
        ] {
            assert_eq!(conflict.kind, MtpModelPolicyKind::ConflictingDrafters);
            assert!(!conflict.route_safe());
            assert!(conflict.has_conflicting_drafters());
            assert_eq!(conflict.qwen_gate_default(), None);
            assert_eq!(conflict.glm_gate_default(), None);
        }
    }

    #[test]
    fn route_telemetry_exposes_policy_depth_safety_and_default() {
        let mut decisions = Vec::new();
        policy(Some(1), None, None, true, true, true).append_route_decisions(true, &mut decisions);
        let decisions = decisions.into_iter().collect::<BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_mtp_model_policy"), Some(&2));
        assert_eq!(decisions.get("ax_mlx_mtp_model_policy_depth"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_mtp_model_policy_route_safe"),
            Some(&1)
        );
        assert_eq!(decisions.get("ax_mlx_mtp_model_policy_active"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_qwen_linear_mtp_certification_candidate"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_deepseek_v4_mtp_certification_candidate"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_deepseek_v4_mtp_direct_fallback"),
            Some(&0)
        );
        assert_eq!(
            decisions.get("ax_mlx_mtp_model_gate_default_present"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_mtp_model_gate_default_x1000"),
            Some(&0)
        );
    }

    #[test]
    fn deepseek_v4_route_telemetry_exposes_fallback_and_candidate() {
        let mut fallback_decisions = Vec::new();
        policy_v4(Some(1)).append_route_decisions(true, &mut fallback_decisions);
        let fallback = fallback_decisions.into_iter().collect::<BTreeMap<_, _>>();
        assert_eq!(fallback.get("ax_mlx_mtp_model_policy"), Some(&9));
        assert_eq!(fallback.get("ax_mlx_mtp_model_policy_route_safe"), Some(&0));
        assert_eq!(fallback.get("ax_mlx_mtp_model_policy_active"), Some(&0));
        assert_eq!(
            fallback.get("ax_mlx_deepseek_v4_mtp_direct_fallback"),
            Some(&1)
        );
        assert_eq!(
            fallback.get("ax_mlx_deepseek_v4_mtp_certification_candidate"),
            Some(&0)
        );

        let mut candidate_decisions = Vec::new();
        MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            deepseek_v4_depth: Some(1),
            deepseek_v4_certification_candidate: true,
            ..Default::default()
        })
        .append_route_decisions(true, &mut candidate_decisions);
        let candidate = candidate_decisions.into_iter().collect::<BTreeMap<_, _>>();
        assert_eq!(candidate.get("ax_mlx_mtp_model_policy"), Some(&8));
        assert_eq!(
            candidate.get("ax_mlx_mtp_model_policy_route_safe"),
            Some(&1)
        );
        assert_eq!(candidate.get("ax_mlx_mtp_model_policy_active"), Some(&1));
        assert_eq!(
            candidate.get("ax_mlx_deepseek_v4_mtp_direct_fallback"),
            Some(&0)
        );
        assert_eq!(
            candidate.get("ax_mlx_deepseek_v4_mtp_certification_candidate"),
            Some(&1)
        );
    }

    #[test]
    fn default_product_linear_route_is_not_active_without_candidate() {
        let mut decisions = Vec::new();
        // mtp_requested may be true from the caller, but policy must not be
        // route-safe without certification-candidate.
        policy(Some(1), None, None, true, true, false).append_route_decisions(true, &mut decisions);
        let decisions = decisions.into_iter().collect::<BTreeMap<_, _>>();
        assert_eq!(decisions.get("ax_mlx_mtp_model_policy"), Some(&4));
        assert_eq!(
            decisions.get("ax_mlx_mtp_model_policy_route_safe"),
            Some(&0)
        );
        assert_eq!(decisions.get("ax_mlx_mtp_model_policy_active"), Some(&0));
        assert_eq!(
            decisions.get("ax_mlx_qwen_linear_mtp_certification_candidate"),
            Some(&0)
        );
    }
}
