//! Model-scoped MTP policy selected once when a runner is constructed.
//!
//! Keep family classification and capability-specific defaults here instead of
//! passing bare gate values around the runner. This makes the Qwen exact
//! depth-one optimization explicit, keeps GLM and Gemma on their independently
//! calibrated policies, and fails closed if incompatible drafters are ever
//! attached to the same artifact.

use super::pipeline::RouteDecisionSink;

const EXACT_DEPTH_ONE_GATE: f32 = 0.0;

/// Stable route code describing the loaded model's MTP policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MtpModelPolicyKind {
    None,
    QwenCalibrated,
    QwenLinearExactDepthOne,
    QwenLinearExactMultiDepth,
    QwenLinearDirectFallback,
    GlmCalibrated,
    Gemma4AssistantCalibrated,
    DeepseekV4Calibrated,
    ConflictingDrafters,
}

impl MtpModelPolicyKind {
    const fn route_code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::QwenCalibrated => 1,
            Self::QwenLinearExactDepthOne => 2,
            Self::QwenLinearExactMultiDepth => 3,
            Self::QwenLinearDirectFallback => 4,
            Self::GlmCalibrated => 5,
            Self::Gemma4AssistantCalibrated => 6,
            Self::ConflictingDrafters => 7,
            Self::DeepseekV4Calibrated => 8,
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
    pub(super) qwen_linear_attention: bool,
    pub(super) qwen_linear_exact_enabled: bool,
}

/// Immutable policy snapshot owned by one [`super::MlxRunner`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MtpModelPolicy {
    kind: MtpModelPolicyKind,
    max_depth: usize,
}

impl MtpModelPolicy {
    pub(super) fn from_loaded(inputs: MtpModelPolicyInputs) -> Self {
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
            return Self {
                kind: MtpModelPolicyKind::ConflictingDrafters,
                max_depth: inputs
                    .qwen_depth
                    .into_iter()
                    .chain(inputs.glm_depth)
                    .chain(inputs.gemma4_assistant_depth)
                    .chain(inputs.deepseek_v4_depth)
                    .max()
                    .unwrap_or(0),
            };
        }

        if let Some(max_depth) = inputs.qwen_depth {
            let kind = if !inputs.qwen_linear_attention {
                MtpModelPolicyKind::QwenCalibrated
            } else if !inputs.qwen_linear_exact_enabled {
                MtpModelPolicyKind::QwenLinearDirectFallback
            } else if max_depth == 1 {
                MtpModelPolicyKind::QwenLinearExactDepthOne
            } else {
                MtpModelPolicyKind::QwenLinearExactMultiDepth
            };
            return Self { kind, max_depth };
        }

        if let Some(max_depth) = inputs.glm_depth {
            return Self {
                kind: MtpModelPolicyKind::GlmCalibrated,
                max_depth,
            };
        }

        if let Some(max_depth) = inputs.gemma4_assistant_depth {
            return Self {
                kind: MtpModelPolicyKind::Gemma4AssistantCalibrated,
                max_depth,
            };
        }

        if let Some(max_depth) = inputs.deepseek_v4_depth {
            return Self {
                kind: MtpModelPolicyKind::DeepseekV4Calibrated,
                max_depth,
            };
        }

        Self {
            kind: MtpModelPolicyKind::None,
            max_depth: 0,
        }
    }

    /// Hard safety gate for model-based speculation.
    pub(super) const fn route_safe(self) -> bool {
        !matches!(
            self.kind,
            MtpModelPolicyKind::QwenLinearDirectFallback | MtpModelPolicyKind::ConflictingDrafters
        )
    }

    pub(super) const fn has_attached_drafter(self) -> bool {
        !matches!(self.kind, MtpModelPolicyKind::None)
    }

    pub(super) const fn max_depth(self) -> usize {
        self.max_depth
    }

    pub(super) const fn is_qwen_linear_direct_fallback(self) -> bool {
        matches!(self.kind, MtpModelPolicyKind::QwenLinearDirectFallback)
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
            (MtpModelPolicyKind::QwenLinearExactDepthOne, GateResolverFamily::Qwen) => {
                Some(EXACT_DEPTH_ONE_GATE)
            }
            _ => None,
        }
    }

    const fn model_gate_default(self) -> Option<f32> {
        match self.kind {
            MtpModelPolicyKind::QwenCalibrated
            | MtpModelPolicyKind::QwenLinearExactDepthOne
            | MtpModelPolicyKind::QwenLinearExactMultiDepth
            | MtpModelPolicyKind::QwenLinearDirectFallback => self.qwen_gate_default(),
            MtpModelPolicyKind::GlmCalibrated => self.glm_gate_default(),
            MtpModelPolicyKind::None
            | MtpModelPolicyKind::Gemma4AssistantCalibrated
            | MtpModelPolicyKind::DeepseekV4Calibrated
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
            "ax_mlx_mtp_model_gate_default_present",
            u32::from(model_default.is_some()),
        );
        decisions.upsert_route_decision(
            "ax_mlx_mtp_model_gate_default_x1000",
            model_default.map_or(0, |gate| (gate.clamp(0.0, 1.0) * 1000.0) as u32),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn policy(
        qwen_depth: Option<usize>,
        glm_depth: Option<usize>,
        gemma_depth: Option<usize>,
        qwen_linear: bool,
        qwen_exact: bool,
    ) -> MtpModelPolicy {
        MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            qwen_depth,
            glm_depth,
            gemma4_assistant_depth: gemma_depth,
            deepseek_v4_depth: None,
            qwen_linear_attention: qwen_linear,
            qwen_linear_exact_enabled: qwen_exact,
        })
    }

    fn policy_v4(v4_depth: Option<usize>) -> MtpModelPolicy {
        MtpModelPolicy::from_loaded(MtpModelPolicyInputs {
            deepseek_v4_depth: v4_depth,
            ..Default::default()
        })
    }

    #[test]
    fn exact_depth_one_default_is_scoped_to_qwen_linear() {
        let exact_depth_one = policy(Some(1), None, None, true, true);
        assert_eq!(
            exact_depth_one.kind,
            MtpModelPolicyKind::QwenLinearExactDepthOne
        );
        assert_eq!(exact_depth_one.qwen_gate_default(), Some(0.0));
        assert_eq!(exact_depth_one.glm_gate_default(), None);
        assert!(exact_depth_one.route_safe());
        assert!(exact_depth_one.has_attached_drafter());
        assert_eq!(exact_depth_one.max_depth(), 1);

        for other in [
            policy(Some(1), None, None, false, false),
            policy(Some(2), None, None, true, true),
            policy(Some(3), None, None, true, true),
            policy(None, Some(1), None, false, false),
            policy(None, None, Some(2), false, false),
            policy(None, None, None, false, false),
        ] {
            assert_eq!(other.qwen_gate_default(), None);
            assert_eq!(other.glm_gate_default(), None);
        }
        assert!(!policy(None, None, None, false, false).has_attached_drafter());
    }

    #[test]
    fn every_supported_mtp_family_has_an_explicit_policy() {
        assert_eq!(
            policy(Some(2), None, None, false, false).kind,
            MtpModelPolicyKind::QwenCalibrated
        );
        assert_eq!(
            policy(Some(2), None, None, true, true).kind,
            MtpModelPolicyKind::QwenLinearExactMultiDepth
        );
        assert_eq!(
            policy(Some(1), None, None, true, false).kind,
            MtpModelPolicyKind::QwenLinearDirectFallback
        );
        assert_eq!(
            policy(None, Some(2), None, false, false).kind,
            MtpModelPolicyKind::GlmCalibrated
        );
        assert_eq!(
            policy(None, None, Some(2), false, false).kind,
            MtpModelPolicyKind::Gemma4AssistantCalibrated
        );
        assert_eq!(
            policy_v4(Some(1)).kind,
            MtpModelPolicyKind::DeepseekV4Calibrated
        );
        let v4 = policy_v4(Some(1));
        assert!(v4.route_safe());
        assert!(v4.has_attached_drafter());
        assert_eq!(v4.max_depth(), 1);
        assert_eq!(v4.qwen_gate_default(), None);
        assert_eq!(v4.glm_gate_default(), None);
        assert!(!policy_v4(None).has_attached_drafter());
    }

    #[test]
    fn conflicting_drafters_fail_closed() {
        for conflict in [
            policy(Some(1), Some(1), None, true, true),
            policy(Some(1), None, Some(1), true, true),
            policy(None, Some(1), Some(1), false, false),
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
        policy(Some(1), None, None, true, true).append_route_decisions(true, &mut decisions);
        let decisions = decisions.into_iter().collect::<BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_mlx_mtp_model_policy"), Some(&2));
        assert_eq!(decisions.get("ax_mlx_mtp_model_policy_depth"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_mtp_model_policy_route_safe"),
            Some(&1)
        );
        assert_eq!(decisions.get("ax_mlx_mtp_model_policy_active"), Some(&1));
        assert_eq!(
            decisions.get("ax_mlx_mtp_model_gate_default_present"),
            Some(&1)
        );
        assert_eq!(
            decisions.get("ax_mlx_mtp_model_gate_default_x1000"),
            Some(&0)
        );
    }
}
