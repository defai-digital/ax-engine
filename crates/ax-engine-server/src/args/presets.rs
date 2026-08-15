use clap::ValueEnum;

use super::PreviewSupportTier;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ServerPreset {
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
    #[value(name = "qwen3.5-9b", alias = "qwen35-9b")]
    Qwen35_9b,
    #[value(name = "qwen3.6-27b", alias = "qwen36-27b")]
    Qwen36_27b,
    #[value(name = "qwen3.6-35b", alias = "qwen36-35b")]
    Qwen36_35b,
    /// H Company Holo3-35B-A3B GUI-agent VLM (Qwen3.5-class 35B-A3B MoE text path).
    /// Same native `qwen3_5` graph as Qwen 3.6 35B; vision tower BF16-sidecar.
    #[value(name = "holo3-35b", alias = "holo3-35b-a3b", alias = "holo3")]
    Holo3_35b,
    /// DeepReinforce Ornith-1.0-35B coding agent (Qwen3.5-class 35B-A3B MoE).
    /// Same native `qwen3_5` graph as Holo3 / Qwen 3.6 35B; vision BF16-sidecar.
    #[value(name = "ornith-35b", alias = "ornith-1.0-35b", alias = "ornith")]
    Ornith35b,
    /// Meta Muse-Glimmer-30B image-text agent (dense `muse_glimmer` VLM).
    /// AXQ-only family; product default is the compact 4-bit pack. Native
    /// decode is incubating (gated attention + centered RMSNorm + ATEM).
    #[value(
        name = "muse-glimmer-30b",
        alias = "muse-glimmer",
        alias = "glimmer-30b"
    )]
    MuseGlimmer30b,
    #[value(name = "qwen3-coder-next", alias = "qwen3-coder")]
    Qwen3CoderNext,
    #[value(name = "qwen3.8-27b", alias = "qwen38-27b", alias = "qwen3.8")]
    Qwen38_27b,
    #[value(name = "qwen3-vl-30b", alias = "qwen3-vl-30b-a3b")]
    Qwen3Vl30b,
    #[value(name = "qwen3-vl-8b", alias = "qwen3-vl-8b-instruct")]
    Qwen3Vl8b,
    /// NVIDIA Nemotron 3 Nano 30B-A3B (`nemotron_h` text path).
    #[value(
        name = "nemotron-3-nano",
        alias = "nemotron-3-nano-30b",
        alias = "nemotron"
    )]
    Nemotron3Nano,
    /// Ministral 3 8B Instruct 2512 (mistral3). Distinct from historical 2410.
    #[value(name = "ministral-3-8b", alias = "ministral-3")]
    Ministral3_8b,
    #[value(name = "ministral-3-14b")]
    Ministral3_14b,
    // Secondary: research / enterprise Llama
    #[value(name = "llama3.1-8b", alias = "llama31-8b")]
    Llama31_8b,
    #[value(name = "llama3.3-70b", alias = "llama33-70b")]
    Llama33_70b,
    #[value(name = "llama4-scout", alias = "llama-4-scout")]
    Llama4Scout,
    // Secondary: European market Mistral
    #[value(name = "mistral-small", alias = "mistral-small-24b")]
    MistralSmall,
    #[value(name = "ministral-8b", alias = "ministral")]
    Ministral8b,
    #[value(name = "devstral-small", alias = "devstral")]
    DevstralSmall,
    // Secondary: open reasoner GPT-OSS (MXFP4)
    #[value(name = "gpt-oss-20b", alias = "gptoss-20b")]
    GptOss20b,
    #[value(name = "gpt-oss-120b", alias = "gptoss-120b")]
    GptOss120b,
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
            // GLM 4.7 Flash direct-support preset: repo-owned MLA graph
            // with sigmoid-routed MoE, dense+MoE split, and shared expert.
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
            // Qwen3.5 9B direct-support preset: dense GatedDeltaNet linear-
            // attention backbone (qwen3_5 family) served natively by the MLX
            // runner.
            Self::Qwen35_9b => PresetDefinition {
                preset: self,
                label: "qwen3.5-9b",
                model_id: "qwen3.5-9b",
                aliases: &[
                    "qwen3.5-9b",
                    "qwen35-9b",
                    "qwen3-5-9b",
                    "qwen3.5-9b-4bit",
                    "qwen3-5-9b-mlx-4bit",
                ],
                model_types: &["qwen3_5", "qwen3.5"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Qwen36_27b => PresetDefinition {
                preset: self,
                label: "qwen3.6-27b",
                model_id: "qwen36-27b",
                aliases: &[
                    "qwen3.6-27b",
                    "qwen36-27b",
                    "qwen3-6-27b",
                    "qwen3.6-27b-4bit",
                    "qwen36-27b-4bit",
                ],
                model_types: &["qwen3_next", "qwen3_6", "qwen3.6", "qwen3_5"],
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
                model_types: &["qwen3_next", "qwen3_6", "qwen3.6", "qwen3_5_moe"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            // Holo3-35B-A3B: Qwen3.5-class 35B-A3B MoE fine-tune (GUI agent VLM).
            // Language path uses the same native qwen3_5 MoE runner as Qwen 3.6
            // 35B; product id stays holo3-35b so Hub/catalog rows are not
            // mislabeled as the official Qwen 3.6 certificate family.
            Self::Holo3_35b => PresetDefinition {
                preset: self,
                label: "holo3-35b",
                model_id: "holo3-35b",
                aliases: &[
                    "holo3-35b",
                    "holo3-35b-a3b",
                    "holo3",
                    "holo-3-35b",
                    "ax-holo3-35b",
                    "ax-holo3-35b-a3b",
                ],
                model_types: &["qwen3_5_moe", "qwen3_5", "qwen3_next", "qwen3_6", "qwen3.6"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            // Ornith-1.0-35B: Qwen3.5-class 35B-A3B MoE coding fine-tune.
            // Language path uses the same native qwen3_5 MoE runner as Holo3;
            // product id stays ornith-35b so Hub/catalog rows are not
            // mislabeled as the official Qwen 3.6 certificate family.
            Self::Ornith35b => PresetDefinition {
                preset: self,
                label: "ornith-35b",
                model_id: "ornith-35b",
                aliases: &[
                    "ornith-35b",
                    "ornith",
                    "ornith-1.0-35b",
                    "ornith-1.0",
                    "ax-ornith-35b",
                    "ax-ornith-1.0-35b",
                ],
                model_types: &["qwen3_5_moe", "qwen3_5", "qwen3_next", "qwen3_6", "qwen3.6"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            // Muse-Glimmer-30B: Meta dense image-text agent. Product id stays
            // muse-glimmer-30b so Hub/catalog rows are not mislabeled as Gemma.
            // Convert recognizes model_type=muse_glimmer; native decode is not
            // yet on the standard Gemma SWA path (gated attention + ATEM).
            Self::MuseGlimmer30b => PresetDefinition {
                preset: self,
                label: "muse-glimmer-30b",
                model_id: "muse-glimmer-30b",
                aliases: &[
                    "muse-glimmer-30b",
                    "muse-glimmer",
                    "glimmer-30b",
                    "ax-muse-glimmer-30b",
                    "ax-muse-glimmer",
                ],
                model_types: &["muse_glimmer", "muse_glimmer_text"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            // Qwen3-Coder-Next direct-support preset: hybrid GatedDeltaNet
            // linear attention + sparse MoE on the qwen3_next family (same
            // repo-owned graph as Qwen 3.6), coder chat template.
            Self::Qwen3CoderNext => PresetDefinition {
                preset: self,
                label: "qwen3-coder-next",
                model_id: "qwen3-coder-next",
                aliases: &[
                    "qwen3-coder-next",
                    "qwen3-coder",
                    "qwen3-coder-next-4bit",
                    "qwen3-coder-next-6bit",
                    "ax-qwen3-coder-next",
                ],
                model_types: &["qwen3_next"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Qwen38_27b => PresetDefinition {
                preset: self,
                label: "qwen3.8-27b",
                model_id: "qwen3.8-27b",
                aliases: &[
                    "qwen3.8-27b",
                    "qwen38-27b",
                    "qwen3.8",
                    "ax-qwen3.8-27b",
                    "ax-qwen38-27b",
                ],
                model_types: &["qwen3_5", "qwen3.5", "qwen3_5_moe", "qwen3_5_moe_text"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Qwen3Vl30b => PresetDefinition {
                preset: self,
                label: "qwen3-vl-30b",
                model_id: "qwen3-vl-30b-a3b",
                aliases: &[
                    "qwen3-vl-30b",
                    "qwen3-vl-30b-a3b",
                    "ax-qwen3-vl-30b",
                    "ax-qwen3-vl-30b-a3b",
                ],
                model_types: &["qwen3_vl_moe", "qwen3-vl-moe", "qwen3_vl"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Qwen3Vl8b => PresetDefinition {
                preset: self,
                label: "qwen3-vl-8b",
                model_id: "qwen3-vl-8b",
                aliases: &["qwen3-vl-8b", "qwen3-vl-8b-instruct", "ax-qwen3-vl-8b"],
                model_types: &["qwen3_vl", "qwen3-vl"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Nemotron3Nano => PresetDefinition {
                preset: self,
                label: "nemotron-3-nano",
                model_id: "nemotron-3-nano",
                aliases: &[
                    "nemotron-3-nano",
                    "nemotron-3-nano-30b",
                    "nemotron",
                    "ax-nemotron-3-nano",
                ],
                model_types: &["nemotron_h", "nemotron_h_nano_omni"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Ministral3_8b => PresetDefinition {
                preset: self,
                label: "ministral-3-8b",
                model_id: "ministral-3-8b",
                aliases: &["ministral-3-8b", "ministral-3", "ax-ministral-3-8b"],
                model_types: &["mistral3", "mistral"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Ministral3_14b => PresetDefinition {
                preset: self,
                label: "ministral-3-14b",
                model_id: "ministral-3-14b",
                aliases: &["ministral-3-14b", "ax-ministral-3-14b"],
                model_types: &["mistral3", "mistral"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Llama31_8b => PresetDefinition {
                preset: self,
                label: "llama3.1-8b",
                model_id: "llama3.1-8b",
                aliases: &[
                    "llama3.1-8b",
                    "llama31-8b",
                    "llama-3.1-8b",
                    "llama3.1-8b-4bit",
                    "llama-3.1-8b-instruct-4bit",
                ],
                model_types: &["llama"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Llama33_70b => PresetDefinition {
                preset: self,
                label: "llama3.3-70b",
                model_id: "llama3.3-70b",
                aliases: &[
                    "llama3.3-70b",
                    "llama33-70b",
                    "llama-3.3-70b",
                    "llama3.3-70b-4bit",
                    "llama-3.3-70b-instruct-4bit",
                ],
                model_types: &["llama"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Llama4Scout => PresetDefinition {
                preset: self,
                label: "llama4-scout",
                model_id: "llama4-scout",
                aliases: &[
                    "llama4-scout",
                    "llama-4-scout",
                    "llama4-scout-4bit",
                    "llama-4-scout-17b-16e-4bit",
                ],
                model_types: &["llama4"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::MistralSmall => PresetDefinition {
                preset: self,
                label: "mistral-small",
                model_id: "mistral-small",
                aliases: &[
                    "mistral-small",
                    "mistral-small-24b",
                    "mistral-small-4bit",
                    "mistral-small-24b-4bit",
                    "mistral-small-3.1",
                ],
                model_types: &["mistral3", "mistral"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::Ministral8b => PresetDefinition {
                preset: self,
                label: "ministral-8b",
                model_id: "ministral-8b",
                aliases: &[
                    "ministral-8b",
                    "ministral",
                    "ministral-8b-4bit",
                    "ministral-8b-instruct-4bit",
                ],
                model_types: &["mistral", "mistral3"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::DevstralSmall => PresetDefinition {
                preset: self,
                label: "devstral-small",
                model_id: "devstral-small",
                aliases: &[
                    "devstral-small",
                    "devstral",
                    "devstral-small-4bit",
                    "devstral-small-2505-4bit",
                ],
                model_types: &["mistral", "mistral3"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::GptOss20b => PresetDefinition {
                preset: self,
                label: "gpt-oss-20b",
                model_id: "gpt-oss-20b",
                aliases: &[
                    "gpt-oss-20b",
                    "gptoss-20b",
                    "gpt-oss-20b-4bit",
                    "gpt-oss-20b-mxfp4",
                    "gpt-oss-20b-mxfp4-q4",
                ],
                model_types: &["gpt_oss"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
            Self::GptOss120b => PresetDefinition {
                preset: self,
                label: "gpt-oss-120b",
                model_id: "gpt-oss-120b",
                aliases: &[
                    "gpt-oss-120b",
                    "gptoss-120b",
                    "gpt-oss-120b-4bit",
                    "gpt-oss-120b-mxfp4",
                    "gpt-oss-120b-mxfp4-q4",
                ],
                model_types: &["gpt_oss"],
                support_tier: PreviewSupportTier::MlxPreview,
                max_batch_tokens: 2048,
            },
        }
    }
}

pub fn render_presets() -> String {
    [
        ServerPreset::Gemma4_12b,
        ServerPreset::Gemma4_26b,
        ServerPreset::Gemma4_31b,
        ServerPreset::Glm47Flash4bit,
        ServerPreset::Qwen35_9b,
        ServerPreset::Qwen36_27b,
        ServerPreset::Qwen36_35b,
        ServerPreset::Holo3_35b,
        ServerPreset::Ornith35b,
        ServerPreset::MuseGlimmer30b,
        ServerPreset::Qwen3CoderNext,
        ServerPreset::Qwen38_27b,
        ServerPreset::Qwen3Vl30b,
        ServerPreset::Qwen3Vl8b,
        ServerPreset::Nemotron3Nano,
        ServerPreset::Ministral3_8b,
        ServerPreset::Ministral3_14b,
        ServerPreset::Llama31_8b,
        ServerPreset::Llama33_70b,
        ServerPreset::Llama4Scout,
        ServerPreset::MistralSmall,
        ServerPreset::Ministral8b,
        ServerPreset::DevstralSmall,
        ServerPreset::GptOss20b,
        ServerPreset::GptOss120b,
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
