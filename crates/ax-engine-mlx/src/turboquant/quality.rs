use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantDecodeQualityProfile {
    StrictDebug,
    ReferenceK8V4,
    ResearchLoose,
}

impl TurboQuantDecodeQualityProfile {
    pub const fn code(self) -> u32 {
        match self {
            Self::StrictDebug => 1,
            Self::ReferenceK8V4 => 2,
            Self::ResearchLoose => 3,
        }
    }

    pub const fn gate(self) -> TurboQuantDecodeQualityGate {
        match self {
            Self::StrictDebug => TurboQuantDecodeQualityGate::STRICT_DEBUG,
            Self::ReferenceK8V4 => TurboQuantDecodeQualityGate::REFERENCE_K8V4,
            Self::ResearchLoose => TurboQuantDecodeQualityGate::RESEARCH_LOOSE,
        }
    }

    pub const fn for_quantization_preset(preset: TurboQuantPreset) -> Self {
        match preset {
            TurboQuantPreset::K8V4 | TurboQuantPreset::K16V4 => Self::ReferenceK8V4,
            TurboQuantPreset::K4V4
            | TurboQuantPreset::K3V4Research
            | TurboQuantPreset::K8V3_5
            | TurboQuantPreset::K7V4 => Self::ResearchLoose,
        }
    }

    pub fn evaluate(
        self,
        report: &TurboQuantDecodeComparisonReport,
    ) -> TurboQuantDecodeQualityDecision {
        self.gate().evaluate(report)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityGate {
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub min_cosine_similarity: f32,
}

impl TurboQuantDecodeQualityGate {
    pub const STRICT_DEBUG: Self = Self::new(0.02, 0.01, 0.999);
    pub const REFERENCE_K8V4: Self = Self::new(0.04, 0.02, 0.998);
    pub const RESEARCH_LOOSE: Self = Self::new(0.08, 0.04, 0.995);

    pub const fn new(max_abs_diff: f32, mean_abs_diff: f32, min_cosine_similarity: f32) -> Self {
        Self {
            max_abs_diff,
            mean_abs_diff,
            min_cosine_similarity,
        }
    }

    pub fn evaluate(
        self,
        report: &TurboQuantDecodeComparisonReport,
    ) -> TurboQuantDecodeQualityDecision {
        let max_abs_diff_passed = report.max_abs_diff <= self.max_abs_diff;
        let mean_abs_diff_passed = report.mean_abs_diff <= self.mean_abs_diff;
        let min_cosine_similarity_passed =
            report.min_cosine_similarity >= self.min_cosine_similarity;

        TurboQuantDecodeQualityDecision {
            passed: max_abs_diff_passed && mean_abs_diff_passed && min_cosine_similarity_passed,
            max_abs_diff_passed,
            mean_abs_diff_passed,
            min_cosine_similarity_passed,
            max_abs_diff: report.max_abs_diff,
            max_abs_diff_limit: self.max_abs_diff,
            mean_abs_diff: report.mean_abs_diff,
            mean_abs_diff_limit: self.mean_abs_diff,
            min_cosine_similarity: report.min_cosine_similarity,
            min_cosine_similarity_limit: self.min_cosine_similarity,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityDecision {
    pub passed: bool,
    pub max_abs_diff_passed: bool,
    pub mean_abs_diff_passed: bool,
    pub min_cosine_similarity_passed: bool,
    pub max_abs_diff: f32,
    pub max_abs_diff_limit: f32,
    pub mean_abs_diff: f32,
    pub mean_abs_diff_limit: f32,
    pub min_cosine_similarity: f32,
    pub min_cosine_similarity_limit: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityEvaluation {
    pub preset: TurboQuantPreset,
    pub profile: TurboQuantDecodeQualityProfile,
    pub gate: TurboQuantDecodeQualityGate,
    pub decision: TurboQuantDecodeQualityDecision,
}

pub fn evaluate_decode_quality_for_preset(
    preset: TurboQuantPreset,
    report: &TurboQuantDecodeComparisonReport,
) -> TurboQuantDecodeQualityEvaluation {
    let profile = TurboQuantDecodeQualityProfile::for_quantization_preset(preset);
    let gate = profile.gate();
    let decision = gate.evaluate(report);
    TurboQuantDecodeQualityEvaluation {
        preset,
        profile,
        gate,
        decision,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantDecodeQualityCheck {
    pub comparison: TurboQuantDecodeComparisonReport,
    pub evaluation: TurboQuantDecodeQualityEvaluation,
}

impl TurboQuantDecodeQualityCheck {
    pub fn for_preset(
        preset: TurboQuantPreset,
        comparison: TurboQuantDecodeComparisonReport,
    ) -> Self {
        let evaluation = evaluate_decode_quality_for_preset(preset, &comparison);
        Self {
            comparison,
            evaluation,
        }
    }
}
