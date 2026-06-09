use clap::ValueEnum;

use super::PreviewSupportTier;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ServerPreset {
    #[value(name = "gemma4-e2b")]
    Gemma4E2b,
    #[value(name = "gemma4-12b")]
    Gemma4_12b,
    #[value(name = "gemma4-26b")]
    Gemma4_26b,
    #[value(name = "gemma4-31b")]
    Gemma4_31b,
    #[value(
        name = "glm4.7-flash-4bit",
        alias = "glm47-flash-4bit",
        alias = "glm4-moe-lite"
    )]
    Glm47Flash4bit,
    #[value(name = "qwen3.6-35b", alias = "qwen36-35b")]
    Qwen36_35b,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PresetDefinition {
    pub(super) preset: ServerPreset,
    pub(super) label: &'static str,
    pub(super) model_id: &'static str,
    pub(super) aliases: &'static [&'static str],
    pub(super) model_types: &'static [&'static str],
    pub(super) support_tier: PreviewSupportTier,
    pub(super) max_batch_tokens: u32,
}

impl ServerPreset {
    pub(super) fn definition(self) -> PresetDefinition {
        match self {
            Self::Gemma4E2b => PresetDefinition {
                preset: self,
                label: "gemma4-e2b",
                model_id: "gemma4-e2b",
                aliases: &["gemma4-e2b", "gemma-4-e2b", "gemma-4-e2b-it"],
                model_types: &["gemma4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Gemma4_12b => PresetDefinition {
                preset: self,
                label: "gemma4-12b",
                model_id: "gemma4-12b",
                aliases: &[
                    "gemma4-12b",
                    "gemma4-12b-4bit",
                    "gemma-4-12b",
                    "gemma-4-12b-it",
                    "gemma-4-12b-it-4bit",
                ],
                model_types: &["gemma4_unified", "gemma4_unified_text", "gemma4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Gemma4_26b => PresetDefinition {
                preset: self,
                label: "gemma4-26b",
                model_id: "gemma4-26b",
                aliases: &[
                    "gemma4-26b",
                    "gemma-4-26b",
                    "gemma-4-26b-a4b-it",
                    "gemma4-26b-4bit",
                ],
                model_types: &["gemma4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Gemma4_31b => PresetDefinition {
                preset: self,
                label: "gemma4-31b",
                model_id: "gemma4-31b",
                aliases: &["gemma4-31b", "gemma-4-31b", "gemma-4-31b-it"],
                model_types: &["gemma4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            // GLM 4.7 Flash is served through mlx-lm passby rather than the
            // repo-owned native graph: the native decode path is only at parity
            // with `mlx_lm` (see docs/PERFORMANCE-DECODE-GAP.md) and the 4-bit
            // export ships no MTP head, so AX speculation cannot accelerate it.
            // The preset therefore selects the delegated tier and requires
            // `--mlx-lm-server-url`. See docs/SUPPORTED-MODELS.md.
            Self::Glm47Flash4bit => PresetDefinition {
                preset: self,
                label: "glm4.7-flash-4bit",
                model_id: "glm4_moe_lite",
                aliases: &[
                    "glm4.7-flash-4bit",
                    "glm47-flash-4bit",
                    "glm4-moe-lite",
                    "glm4_moe_lite",
                    "glm-4.7-flash-4bit",
                    "glm-4-7-flash-4bit",
                ],
                model_types: &["glm4_moe_lite"],
                support_tier: PreviewSupportTier::MlxLmDelegated,
                max_batch_tokens: 2048,
            },
            Self::Qwen36_35b => PresetDefinition {
                preset: self,
                label: "qwen3.6-35b",
                model_id: "qwen3.6-35b",
                aliases: &[
                    "qwen3.6-35b",
                    "qwen36-35b",
                    "qwen3-6-35b",
                    "qwen3.6-35b-a3b",
                    "qwen36-35b-a3b",
                ],
                model_types: &["qwen3_next", "qwen3_6", "qwen3.6"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
        }
    }
}

pub fn render_presets() -> String {
    [
        ServerPreset::Gemma4E2b,
        ServerPreset::Gemma4_12b,
        ServerPreset::Gemma4_26b,
        ServerPreset::Gemma4_31b,
        ServerPreset::Glm47Flash4bit,
        ServerPreset::Qwen36_35b,
    ]
    .into_iter()
    .map(|preset| {
        let definition = preset.definition();
        let requirement = match definition.support_tier {
            PreviewSupportTier::MlxLmDelegated => "requires --mlx-lm-server-url (mlx-lm passby)",
            _ => "requires --mlx-model-artifacts-dir or explicit resolver",
        };
        format!(
            "{}\tmodel_id={}\tsupport_tier={:?}\t{}",
            definition.label, definition.model_id, definition.support_tier, requirement
        )
    })
    .collect::<Vec<_>>()
    .join("\n")
}
