//! Typed chat-serving contracts carried by architecture registrations
//! (ADR-025, v8.0 foundation hardening Phase 1).
//!
//! The server chat layer historically selected prompt framing, stop
//! sequences, thinking defaults, and output sanitizers by substring-matching
//! the user-facing model id (`ChatPromptTemplate::for_model_id`). New
//! families (DeepSeek V4 Pro, MiniMax M3) must not extend that chain: every
//! registered family carries its chat contract here, and the server resolves
//! it from the manifest `model_family` before falling back to model-id
//! heuristics for unconverted artifacts and delegated backends.
//!
//! Data describes semantics; compiled server code executes them (ADR-023 D3).
//! No template text lives in this module.

/// Prompt framing family selected for chat rendering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatTemplateKind {
    /// Qwen ChatML (`<|im_start|>` / `<|im_end|>`), shared by Qwen3-class,
    /// MiniCPM-V, and Nemotron text backbones.
    QwenChatMl,
    /// Llama 3.x Instruct (`<|start_header_id|>` / `<|eot_id|>`).
    Llama3,
    /// Llama 4 Instruct (`<|header_start|>` / `<|eot|>`).
    Llama4,
    /// Gemma 4 IT turn framing (`<turn|>` terminator, thought channels).
    Gemma4,
    /// GLM 4.x (`[gMASK]<sop>` + structural tool markers).
    Glm47,
    /// Devstral / Mistral Small-style `[SYSTEM_PROMPT]…[INST]…[/INST]`.
    MistralInstruct,
    /// Ministral classic Instruct (system folded into the first `[INST]`).
    MinistralInstruct,
    /// OpenAI Harmony framing used by GPT-OSS.
    GptOssHarmony,
    /// Meta Muse-Glimmer ATEM framing.
    MuseGlimmerAtem,
    /// DeepSeek V3/R1 framing (`<｜User｜>` / `<｜Assistant｜>` + think blocks).
    DeepSeekChat,
    /// DeepSeek V4 canonical encoding_dsv4 framing (Jinja-equivalent path).
    DeepSeekV4Chat,
    /// MiniMax M3 hub jinja (`]~b]user` / `]~b]ai` / `[e~[` / `</mm:think>`).
    MiniMaxM3,
    /// Families without a verified AX chat fixture use plain role prefixes.
    PlainRolePrefix,
    /// Registered but chat-unsupported today; the renderer rejects with
    /// guidance. The unsupported label derives from the registration's
    /// `family_label` (only `gemma3` / `mixtral` may carry this kind).
    Unsupported,
    /// No family-level chat contract: embed/ASR/auxiliary rows, or rows
    /// whose framing depends on sub-family model-id distinctions (e.g.
    /// `mistral3` covers both Mistral Instruct and Ministral artifacts).
    /// Resolution falls back to model-id heuristics.
    NotApplicable,
}

/// Output rewrite contract applied when decoding chat completions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatOutputPolicy {
    /// No family-specific rewriting; verbatim decode plus stop sequences.
    Plain,
    /// Strip Gemma 4 thinking-channel spans (`<|channel>…<channel|>`).
    Gemma4Channels,
    /// Surface only the last GPT-OSS Harmony `final` channel body.
    GptOssFinalChannel,
    /// Split DeepSeek reasoning (before the think-close tag) from visible
    /// content.
    DeepSeekThinkSplit,
    /// Preserve GLM structural tool-call markers during decode.
    GlmToolMarkers,
}

/// Serving-visible chat behavior for a registered architecture family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatContract {
    /// Prompt framing family for chat rendering.
    pub template: ChatTemplateKind,
    /// Output rewrite contract for decoded chat completions.
    pub output_policy: ChatOutputPolicy,
    /// Whether OpenAI-compatible requests inject `enable_thinking=false` by
    /// default (short answers; callers opt into thinking explicitly).
    pub default_thinking_off: bool,
    /// Whether chat requires an instruction-tuned artifact providing
    /// `chat_template.jinja` (Gemma 4 artifact validation gate).
    pub requires_instruct_artifact: bool,
}

impl ChatContract {
    /// Convenience constructor for rows without a chat surface.
    pub const fn not_applicable() -> Self {
        Self {
            template: ChatTemplateKind::NotApplicable,
            output_policy: ChatOutputPolicy::Plain,
            default_thinking_off: false,
            requires_instruct_artifact: false,
        }
    }
}

/// Resolve the chat contract for a manifest `model_family` label.
pub fn chat_contract_for_family(family_label: &str) -> Option<&'static ChatContract> {
    crate::architecture_registry::lookup_architecture(family_label)
        .map(|registration| &registration.chat_contract)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture_registry::{
        ARCHITECTURE_REGISTRY, LayerForwardRoute, lookup_architecture,
    };

    #[test]
    fn every_registered_family_resolves_a_contract() {
        for entry in ARCHITECTURE_REGISTRY {
            let contract = chat_contract_for_family(entry.family_label)
                .unwrap_or_else(|| panic!("{} missing chat contract", entry.family_label));
            assert_eq!(contract, &entry.chat_contract);
        }
    }

    #[test]
    fn qwen_families_use_chatml_with_thinking_off() {
        for label in [
            "qwen3",
            "qwen3_5",
            "qwen3_next",
            "minicpmv4_6",
            "qwen3_vl",
            "qwen3_vl_moe",
            "nemotron_h",
        ] {
            let contract = chat_contract_for_family(label).unwrap_or_else(|| panic!("{label}"));
            assert_eq!(contract.template, ChatTemplateKind::QwenChatMl, "{label}");
            assert_eq!(contract.output_policy, ChatOutputPolicy::Plain, "{label}");
            assert!(contract.default_thinking_off, "{label}");
            assert!(!contract.requires_instruct_artifact, "{label}");
        }
    }

    #[test]
    fn gemma4_families_use_channel_output_and_instruct_gate() {
        for label in ["gemma4", "gemma4_unified", "gemma4_vl", "diffusion_gemma"] {
            let contract = chat_contract_for_family(label).unwrap_or_else(|| panic!("{label}"));
            assert_eq!(contract.template, ChatTemplateKind::Gemma4, "{label}");
            assert_eq!(
                contract.output_policy,
                ChatOutputPolicy::Gemma4Channels,
                "{label}"
            );
            assert!(contract.default_thinking_off, "{label}");
            assert!(contract.requires_instruct_artifact, "{label}");
        }
    }

    #[test]
    fn deepseek_rows_split_v3_and_v4_framing() {
        for label in ["deepseek_v3", "deepseek_v32"] {
            let contract = chat_contract_for_family(label).unwrap_or_else(|| panic!("{label}"));
            assert_eq!(contract.template, ChatTemplateKind::DeepSeekChat, "{label}");
            assert_eq!(
                contract.output_policy,
                ChatOutputPolicy::DeepSeekThinkSplit,
                "{label}"
            );
            assert!(!contract.default_thinking_off, "{label}");
        }
        let v4 = chat_contract_for_family("deepseek_v4").expect("deepseek_v4 registered");
        assert_eq!(v4.template, ChatTemplateKind::DeepSeekV4Chat);
        assert_eq!(v4.output_policy, ChatOutputPolicy::DeepSeekThinkSplit);
    }

    #[test]
    fn deepseek_v4_chat_kind_is_unique_to_deepseek_v4() {
        let v4_rows: Vec<&str> = ARCHITECTURE_REGISTRY
            .iter()
            .filter(|entry| entry.chat_contract.template == ChatTemplateKind::DeepSeekV4Chat)
            .map(|entry| entry.family_label)
            .collect();
        assert_eq!(v4_rows, vec!["deepseek_v4"]);
    }

    #[test]
    fn only_gemma3_and_mixtral_are_chat_unsupported() {
        let unsupported: Vec<&str> = ARCHITECTURE_REGISTRY
            .iter()
            .filter(|entry| entry.chat_contract.template == ChatTemplateKind::Unsupported)
            .map(|entry| entry.family_label)
            .collect();
        assert_eq!(unsupported, vec!["gemma3", "mixtral"]);
    }

    #[test]
    fn not_applicable_rows_have_no_chat_surface_or_are_ambiguous() {
        let not_applicable: Vec<&str> = ARCHITECTURE_REGISTRY
            .iter()
            .filter(|entry| entry.chat_contract.template == ChatTemplateKind::NotApplicable)
            .map(|entry| entry.family_label)
            .collect();
        // Embed/ASR/auxiliary rows plus mistral3, whose Instruct-vs-Ministral
        // split is decided by model-id heuristics (see tech-spec §2.1).
        // Order follows ARCHITECTURE_REGISTRY row order.
        assert_eq!(
            not_applicable,
            vec![
                "gemma4_assistant",
                "embeddinggemma",
                "nemotron_embed",
                "mistral3",
                "whisper"
            ]
        );
    }

    #[test]
    fn specialized_output_policies_match_their_families() {
        assert_eq!(
            chat_contract_for_family("glm4_moe_lite")
                .unwrap()
                .output_policy,
            ChatOutputPolicy::GlmToolMarkers
        );
        assert_eq!(
            chat_contract_for_family("gpt_oss").unwrap().output_policy,
            ChatOutputPolicy::GptOssFinalChannel
        );
        assert_eq!(
            chat_contract_for_family("muse_glimmer").unwrap().template,
            ChatTemplateKind::MuseGlimmerAtem
        );
        assert_eq!(
            chat_contract_for_family("unlimited_ocr").unwrap().template,
            ChatTemplateKind::PlainRolePrefix
        );
        assert_eq!(
            chat_contract_for_family("minimax_m3").unwrap().template,
            ChatTemplateKind::MiniMaxM3
        );
    }

    #[test]
    fn trunk_style_marks_only_dedicated_forward_routes() {
        use crate::architecture_registry::TrunkStyle;
        for entry in ARCHITECTURE_REGISTRY {
            let style = entry.layer_forward_route.trunk_style();
            if entry.layer_forward_route == LayerForwardRoute::DeepseekV4 {
                assert_eq!(
                    style,
                    TrunkStyle::DedicatedTrunk,
                    "{} must stay a dedicated trunk",
                    entry.family_label
                );
            } else {
                assert_eq!(
                    style,
                    TrunkStyle::PerLayer,
                    "{} unexpectedly claims a dedicated trunk",
                    entry.family_label
                );
            }
        }
        // The registry row for the dedicated trunk keeps its experimental
        // certification note so admission diagnostics stay honest.
        let v4 = lookup_architecture("deepseek_v4").expect("deepseek_v4 registered");
        assert!(!v4.cert_gate_note.is_empty());
    }
}
