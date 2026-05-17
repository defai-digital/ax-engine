use clap::ValueEnum;

use super::PreviewSupportTier;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ServerPreset {
    #[value(name = "gemma4-e2b")]
    Gemma4E2b,
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
            Self::Gemma4_31b => PresetDefinition {
                preset: self,
                label: "gemma4-31b",
                model_id: "gemma4-31b",
                aliases: &["gemma4-31b", "gemma-4-31b", "gemma-4-31b-it"],
                model_types: &["gemma4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
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
                support_tier: PreviewSupportTier::MlxPreview,
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
        ServerPreset::Gemma4_31b,
        ServerPreset::Glm47Flash4bit,
        ServerPreset::Qwen36_35b,
    ]
    .into_iter()
    .map(|preset| {
        let definition = preset.definition();
        format!(
            "{}\tmodel_id={}\tsupport_tier={:?}\trequires --mlx-model-artifacts-dir or explicit resolver",
            definition.label, definition.model_id, definition.support_tier
        )
    })
    .collect::<Vec<_>>()
    .join("\n")
}
