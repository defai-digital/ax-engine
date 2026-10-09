//! Three-tier model quality grading (P1 support-tier system).
//!
//! Every listed model family carries an explicit support tier so coverage
//! claims stay honest instead of inflating a raw architecture count:
//!
//! - [`ModelSupportTier::Certified`]: repo-owned `ax-engine-mlx` graph plus
//!   current certification / benchmark evidence. Performance and correctness
//!   claims are allowed only for these families.
//! - [`ModelSupportTier::Compatible`]: loadable through the generic
//!   `standard` family path (or another registered route) with manifest
//!   capability probing. No certification or performance guarantee.
//! - [`ModelSupportTier::Experimental`]: feature-gated paths (diffusion,
//!   pipeline-parallel, batched SWA pilots). Shape and behavior may change
//!   without notice.
//!
//! This is a per-family quality grade and is intentionally distinct from the
//! SDK `SupportTier` backend-selection enum (`mlx_certified` /
//! `mlx_lm_delegated` / …), which records *who runs the model* for a
//! resolved session rather than how well a family is supported.
//!
//! The tier for registered families is declared data-driven on each
//! [`ArchitectureRegistration`](crate::architecture_registry::ArchitectureRegistration)
//! row; unregistered labels resolve to [`ModelSupportTier::Compatible`] with
//! the manifest-probing caveat (loadability is decided by
//! `ArchitectureSpec::from_manifest` at load time, not by name allowlists).

use serde::{Deserialize, Serialize};

use crate::architecture_registry::lookup_architecture;
use crate::generation::GenerationKind;
use crate::model::NativeModelManifest;

/// Per-family model quality grade.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSupportTier {
    /// Repo-owned graph + certification / benchmark evidence.
    Certified,
    /// Loadable via manifest capability probing; no cert/perf guarantee.
    Compatible,
    /// Feature-gated path (diffusion, pipeline-parallel, SWA pilots).
    Experimental,
}

impl ModelSupportTier {
    /// Stable lowercase label for JSON artifacts and CLI/TUI display.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::Compatible => "compatible",
            Self::Experimental => "experimental",
        }
    }
}

impl std::fmt::Display for ModelSupportTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Resolve the support tier for a manifest `model_family` label.
///
/// Registered families return the tier declared on their registry row.
/// Unknown labels resolve to [`ModelSupportTier::Compatible`]: they are only
/// loadable when manifest capability probing accepts the structure, and they
/// carry no certification or performance guarantee.
pub fn support_tier_for_family(family_label: &str) -> ModelSupportTier {
    lookup_architecture(family_label)
        .map(|entry| entry.support_tier)
        .unwrap_or(ModelSupportTier::Compatible)
}

/// Resolve the support tier from a manifest.
///
/// A manifest whose structural signals force a feature-gated generation kind
/// (block diffusion) is [`ModelSupportTier::Experimental`] even when its
/// family label is otherwise Certified or Compatible — the tier reflects the
/// path that will actually run.
pub fn support_tier_for_manifest(manifest: &NativeModelManifest) -> ModelSupportTier {
    if matches!(
        GenerationKind::from_manifest(manifest),
        GenerationKind::BlockDiffusion
    ) {
        return ModelSupportTier::Experimental;
    }
    support_tier_for_family(&manifest.model_family)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_registry::ARCHITECTURE_REGISTRY;
    use crate::model::{NativeDiffusionConfig, base_manifest};

    /// Expected tier for every statically registered family. Keep in sync
    /// with the `support_tier` field on each registry row; the test below
    /// fails loudly when a row is added without a deliberate tier decision.
    fn expected_tier(family_label: &str) -> ModelSupportTier {
        match family_label {
            "qwen3" | "qwen3_5" | "qwen3_next" | "qwen3_vl" | "gemma4" | "gemma4_vl"
            | "glm4_moe_lite" | "gpt_oss" | "deepseek_v3" | "deepseek_v32" => {
                ModelSupportTier::Certified
            }
            "diffusion_gemma" | "deepseek_v4" | "muse_glimmer" | "minimax_m3" | "qwen4_exp" => {
                ModelSupportTier::Experimental
            }
            _ => ModelSupportTier::Compatible,
        }
    }

    #[test]
    fn every_registered_family_declares_expected_tier() {
        for entry in ARCHITECTURE_REGISTRY {
            assert_eq!(
                entry.support_tier,
                expected_tier(entry.family_label),
                "unexpected support tier for registered family {}",
                entry.family_label
            );
        }
    }

    #[test]
    fn unknown_family_is_compatible_with_probing_caveat() {
        assert_eq!(
            support_tier_for_family("brand_new_hf_layout"),
            ModelSupportTier::Compatible
        );
        assert_eq!(support_tier_for_family(""), ModelSupportTier::Compatible);
    }

    #[test]
    fn certified_families_match_explicit_list() {
        for label in [
            "qwen3",
            "qwen3_5",
            "qwen3_next",
            "qwen3_vl",
            "gemma4",
            "gemma4_vl",
            "glm4_moe_lite",
            "gpt_oss",
            "deepseek_v3",
            "deepseek_v32",
        ] {
            assert_eq!(
                support_tier_for_family(label),
                ModelSupportTier::Certified,
                "{label} must stay Certified per the explicit certified list"
            );
        }
    }

    #[test]
    fn flash_next_execution_does_not_imply_checkpoint_certification() {
        assert_eq!(
            support_tier_for_family("qwen4_exp"),
            ModelSupportTier::Experimental
        );
    }

    #[test]
    fn experimental_gate_covers_diffusion_family() {
        assert_eq!(
            support_tier_for_family("diffusion_gemma"),
            ModelSupportTier::Experimental
        );
        assert_eq!(
            support_tier_for_family("muse_glimmer"),
            ModelSupportTier::Experimental
        );
    }

    #[test]
    fn deepseek_v4_is_experimental_until_graph_certification() {
        // Converter + registry plumbing only; the repo-owned runtime graph is
        // still in progress, so no certification evidence exists.
        assert_eq!(
            support_tier_for_family("deepseek_v4"),
            ModelSupportTier::Experimental
        );
    }

    #[test]
    fn compatible_families_have_no_cert_promise() {
        for label in [
            "llama3",
            "gemma3",
            "gemma4_assistant",
            "gemma4_unified",
            "qwen3_vl_moe",
            "minicpmv4_6",
            "embeddinggemma",
            "mistral3",
            "mixtral",
            "llama4",
            "nemotron_h",
            "nemotron_embed",
            "unlimited_ocr",
            "whisper",
        ] {
            assert_eq!(
                support_tier_for_family(label),
                ModelSupportTier::Compatible,
                "{label} should remain Compatible until certification evidence lands"
            );
        }
    }

    #[test]
    fn manifest_forcing_diffusion_is_experimental_regardless_of_family() {
        let mut manifest = base_manifest("gemma4", 4);
        assert_eq!(
            support_tier_for_manifest(&manifest),
            ModelSupportTier::Certified
        );
        manifest.diffusion = NativeDiffusionConfig {
            canvas_size: Some(256),
            ..Default::default()
        };
        assert_eq!(
            support_tier_for_manifest(&manifest),
            ModelSupportTier::Experimental,
            "diffusion structural signals must force the experimental path"
        );
        // The feature-gated family label itself resolves to Experimental too.
        let diffusion = base_manifest("diffusion_gemma", 4);
        assert_eq!(
            support_tier_for_manifest(&diffusion),
            ModelSupportTier::Experimental
        );
    }

    #[test]
    fn tier_labels_are_stable() {
        assert_eq!(ModelSupportTier::Certified.as_str(), "certified");
        assert_eq!(ModelSupportTier::Compatible.as_str(), "compatible");
        assert_eq!(ModelSupportTier::Experimental.as_str(), "experimental");
        assert_eq!(
            serde_json::to_string(&ModelSupportTier::Certified).unwrap_or_default(),
            "\"certified\""
        );
    }
}
