//! Static architecture registration (ADR-038 Phase 3+).
//!
//! Maps known family labels to default generation kind, layer-forward route,
//! and certification notes. Convert and runtime gates should prefer this
//! registry + structural caps over ad-hoc string allowlists when adding
//! hybrid variants.

use crate::generation::GenerationKind;

/// Which MLX family forward implementation owns the layer graph.
///
/// Linear-attention layers still short-circuit before this route (per-layer
/// capability); this selects the non-linear / default family implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerForwardRoute {
    /// Shared standard transformer path (dense/SWA/MoE Gemma/Qwen/Llama3/…).
    Standard,
    Llama4,
    GlmMoeLite,
    DeepseekV3,
    Mistral3,
    Mixtral,
    GptOss,
    /// Nemotron-H hybrid: per-layer Mamba-2 / attention / ReLU² MoE mixers.
    NemotronH,
}

impl LayerForwardRoute {
    /// Stable telemetry code for route decisions.
    pub const fn telemetry_code(self) -> u32 {
        match self {
            Self::Standard => 0,
            Self::Llama4 => 1,
            Self::GlmMoeLite => 2,
            Self::DeepseekV3 => 3,
            Self::Mistral3 => 4,
            Self::Mixtral => 5,
            Self::GptOss => 6,
            Self::NemotronH => 7,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Llama4 => "llama4",
            Self::GlmMoeLite => "glm4_moe_lite",
            Self::DeepseekV3 => "deepseek_v3",
            Self::Mistral3 => "mistral3",
            Self::Mixtral => "mixtral",
            Self::GptOss => "gpt_oss",
            Self::NemotronH => "nemotron_h",
        }
    }
}

/// Static registration entry for a supported (or incubating) architecture label.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchitectureRegistration {
    /// Canonical `model_family` label stored on the manifest.
    pub family_label: &'static str,
    /// Default generation paradigm when the manifest does not force another.
    pub default_generation: GenerationKind,
    /// Layer-forward implementation route (ADR-038 composition boundary).
    pub layer_forward_route: LayerForwardRoute,
    /// Whether continuous dense batched decode is *structurally* in scope
    /// (still requires numerical certification).
    pub dense_batched_decode_candidate: bool,
    /// Human-readable cert / support note for docs and diagnostics.
    pub cert_gate_note: &'static str,
}

/// All statically registered architecture labels.
///
/// Adding a hybrid that reuses existing primitives should primarily add a row
/// here plus convert mapping — not a new eligibility allowlist of family names.
pub static ARCHITECTURE_REGISTRY: &[ArchitectureRegistration] = &[
    ArchitectureRegistration {
        family_label: "qwen3",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: true,
        cert_gate_note: "dense full-attention AR; batched decode when certified",
    },
    ArchitectureRegistration {
        family_label: "qwen3_5",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "hybrid linear+full; structural rejections include linear_attention",
    },
    ArchitectureRegistration {
        family_label: "qwen3_next",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "hybrid gated-delta / MoE; capability-gated, not name-allowlisted",
    },
    ArchitectureRegistration {
        family_label: "llama3",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: true,
        cert_gate_note: "dense full-attention AR when structurally dense",
    },
    ArchitectureRegistration {
        family_label: "gemma3",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Gemma3 SWA text backbone; standard path",
    },
    ArchitectureRegistration {
        family_label: "gemma4",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "interleaved SWA / optional MoE; dense pilot rejects SWA+MoE; SWA text may use gemma_swa structural helper + multi_token_window_views",
    },
    ArchitectureRegistration {
        family_label: "gemma4_assistant",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "assistant MTP drafter; not dense-batch candidate",
    },
    ArchitectureRegistration {
        family_label: "gemma4_unified",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "multimodal prefill adapters feed AR generation",
    },
    ArchitectureRegistration {
        family_label: "gemma4_vl",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Gemma 4 E2B/E4B ViT+Conformer towers into gemma4 AR backbone (WS-V1)",
    },
    ArchitectureRegistration {
        family_label: "qwen3_vl",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: true,
        cert_gate_note: "Qwen3-VL dense: text path rides certified qwen3 batched decode when text-only",
    },
    ArchitectureRegistration {
        family_label: "qwen3_vl_moe",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Qwen3-VL-MoE; text decode shares qwen3-MoE graphs; batch cert separate",
    },
    ArchitectureRegistration {
        family_label: "diffusion_gemma",
        default_generation: GenerationKind::BlockDiffusion,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "block diffusion; generation kind BlockDiffusion",
    },
    ArchitectureRegistration {
        family_label: "embeddinggemma",
        default_generation: GenerationKind::EncoderEmbed,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "encoder embed strategy; not a decode path",
    },
    ArchitectureRegistration {
        family_label: "glm4_moe_lite",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::GlmMoeLite,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MLA + MoE; structural rejections",
    },
    ArchitectureRegistration {
        family_label: "deepseek_v3",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::DeepseekV3,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MLA + MoE",
    },
    ArchitectureRegistration {
        family_label: "deepseek_v32",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::DeepseekV3,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MLA + MoE",
    },
    ArchitectureRegistration {
        family_label: "mistral3",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Mistral3,
        dense_batched_decode_candidate: false,
        cert_gate_note: "uniform SWA; sliding_window rejection",
    },
    ArchitectureRegistration {
        family_label: "mixtral",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Mixtral,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MoE",
    },
    ArchitectureRegistration {
        family_label: "llama4",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Llama4,
        dense_batched_decode_candidate: false,
        cert_gate_note: "iRoPE / MoE hybrid",
    },
    ArchitectureRegistration {
        family_label: "gpt_oss",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::GptOss,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MXFP4 MoE",
    },
    ArchitectureRegistration {
        family_label: "nemotron_h",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::NemotronH,
        dense_batched_decode_candidate: false,
        cert_gate_note: "hybrid Mamba-2 + GQA + ReLU2 MoE; pattern-driven mixers",
    },
    ArchitectureRegistration {
        family_label: "unlimited_ocr",
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Unlimited-OCR multimodal: dual vision + SWA MoE language tower",
    },
];

/// Look up a static registration by manifest `model_family` label.
pub fn lookup_architecture(family_label: &str) -> Option<&'static ArchitectureRegistration> {
    ARCHITECTURE_REGISTRY
        .iter()
        .find(|entry| entry.family_label == family_label)
}

/// Resolve the layer-forward route for a family label.
///
/// Prefer this over open-coding family string matches at dispatch sites.
pub fn resolve_layer_forward_route(family_label: &str) -> Option<LayerForwardRoute> {
    lookup_architecture(family_label).map(|r| r.layer_forward_route)
}

/// Default generation from the registry when present; falls back to
/// [`GenerationKind::from_manifest`] for unregistered labels.
pub fn default_generation_for_family(
    family_label: &str,
    manifest_generation: GenerationKind,
) -> GenerationKind {
    // Manifest-derived kind wins when it already encodes diffusion/embed
    // structural signals; registry only supplies defaults for AR labels.
    if !matches!(manifest_generation, GenerationKind::Autoregressive) {
        return manifest_generation;
    }
    lookup_architecture(family_label)
        .map(|r| r.default_generation)
        .unwrap_or(manifest_generation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::{ArchitectureSpec, StructuralCapabilities};
    use crate::generation::GenerationKind;
    use crate::model::{
        AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeDiffusionConfig,
        NativeLinearAttentionConfig, NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus,
        NativeTensorFormat, WeightSanitize,
    };

    fn base_manifest(family: &str, layer_count: u32) -> NativeModelManifest {
        NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: family.to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count,
            hidden_size: 128,
            intermediate_size: 256,
            attention_head_count: 4,
            attention_head_dim: 32,
            kv_head_count: 2,
            vocab_size: 1000,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: NativeMoeConfig::default(),
            glm_router: Default::default(),
            weight_sanitize: WeightSanitize::default(),
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            tensors: Vec::new(),
        }
    }

    #[test]
    fn registry_has_qwen3_and_diffusion() {
        let qwen = lookup_architecture("qwen3").expect("qwen3 registered");
        assert!(qwen.dense_batched_decode_candidate);
        assert_eq!(qwen.default_generation, GenerationKind::Autoregressive);
        assert_eq!(qwen.layer_forward_route, LayerForwardRoute::Standard);

        let diff = lookup_architecture("diffusion_gemma").expect("diffusion registered");
        assert!(!diff.dense_batched_decode_candidate);
        assert_eq!(diff.default_generation, GenerationKind::BlockDiffusion);
        assert_eq!(diff.layer_forward_route, LayerForwardRoute::Standard);
    }

    #[test]
    fn resolve_layer_forward_route_covers_specialized_families() {
        assert_eq!(
            resolve_layer_forward_route("qwen3"),
            Some(LayerForwardRoute::Standard)
        );
        assert_eq!(
            resolve_layer_forward_route("qwen3_5"),
            Some(LayerForwardRoute::Standard)
        );
        assert_eq!(
            resolve_layer_forward_route("llama4"),
            Some(LayerForwardRoute::Llama4)
        );
        assert_eq!(
            resolve_layer_forward_route("glm4_moe_lite"),
            Some(LayerForwardRoute::GlmMoeLite)
        );
        assert_eq!(
            resolve_layer_forward_route("deepseek_v32"),
            Some(LayerForwardRoute::DeepseekV3)
        );
        assert_eq!(
            resolve_layer_forward_route("nemotron_h"),
            Some(LayerForwardRoute::NemotronH)
        );
        assert_eq!(
            resolve_layer_forward_route("unlimited_ocr"),
            Some(LayerForwardRoute::Standard)
        );
        assert_eq!(
            resolve_layer_forward_route("gpt_oss"),
            Some(LayerForwardRoute::GptOss)
        );
        assert_eq!(resolve_layer_forward_route("not_a_family"), None);
    }

    #[test]
    fn structural_caps_accept_qwen3_next_linear_and_hybrid_moe() {
        // Gated-delta linear attention is now handled by the batched linear path
        // (Phase 3.7), so a qwen3_5 linear + dense-FFN model is structurally
        // eligible (no sliding/mla/gating/moe here).
        let mut m = base_manifest("qwen3_5", 4);
        m.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: Some(4),
            num_value_heads: Some(4),
            num_key_heads: Some(4),
            key_head_dim: Some(32),
            value_head_dim: Some(32),
            conv_kernel_dim: Some(4),
        };
        let caps = ArchitectureSpec::from_manifest(&m).capabilities;
        let reasons = caps.dense_batched_decode_structural_rejections();
        assert!(
            !reasons.contains(&"linear_attention"),
            "linear attention is no longer a structural rejection, got {reasons:?}"
        );
        assert!(
            reasons.is_empty(),
            "linear + dense qwen3_5 should be structurally eligible, got {reasons:?}"
        );
        // Still not the *dense full-attention* pilot shape.
        assert!(!caps.is_structurally_dense_full_attention_only());

        // Add MoE → qwen3_5 family sets batched_qwen3_moe_router explicitly.
        m.model_family = "qwen3_next".into();
        m.moe.expert_count = Some(8);
        m.moe.experts_per_token = Some(2);
        m.moe.expert_intermediate_size = Some(64);
        let caps_moe = ArchitectureSpec::from_manifest(&m).capabilities;
        assert!(
            caps_moe.batched_qwen3_moe_router,
            "qwen3_next MoE must set explicit qwen3 router capability"
        );
        let reasons_moe = caps_moe.batched_decode_structural_rejections();
        assert!(
            !reasons_moe.contains(&"moe"),
            "qwen3 MoE hybrid should not be moe-rejected, got {reasons_moe:?}"
        );
    }

    #[test]
    fn structural_caps_admit_qwen3_moe_without_linear_via_router_bit() {
        // Router kind is explicit from family — pure dense MoE on qwen3 is
        // structurally eligible (still needs numerical certification).
        let mut m = base_manifest("qwen3", 4);
        m.moe.expert_count = Some(8);
        m.moe.experts_per_token = Some(2);
        m.moe.expert_intermediate_size = Some(64);
        let caps = ArchitectureSpec::from_manifest(&m).capabilities;
        assert!(caps.batched_qwen3_moe_router);
        assert!(
            !caps.batched_decode_structural_rejections().contains(&"moe"),
            "qwen3 MoE router bit must admit MoE without linear proxy"
        );
    }

    #[test]
    fn structural_caps_reject_unsupported_moe_router_families() {
        // Gemma4 / GPT-OSS use different routers; family bit stays false.
        for family in ["gemma4", "gpt_oss", "glm4_moe_lite", "deepseek_v3"] {
            let mut m = base_manifest(family, 4);
            m.moe.expert_count = Some(8);
            m.moe.experts_per_token = Some(2);
            m.moe.expert_intermediate_size = Some(64);
            let caps = ArchitectureSpec::from_manifest(&m).capabilities;
            assert!(
                !caps.batched_qwen3_moe_router,
                "{family} must not claim qwen3 batched MoE router"
            );
            assert!(
                caps.batched_decode_structural_rejections().contains(&"moe")
                    || caps.batched_decode_structural_rejections().contains(&"mla")
                    || caps
                        .batched_decode_structural_rejections()
                        .contains(&"layer_gating"),
                "{family} MoE must be structurally rejected for batched decode: {:?}",
                caps.batched_decode_structural_rejections()
            );
        }
    }

    #[test]
    fn structural_caps_accept_dense_qwen_shape() {
        let m = base_manifest("qwen3", 4);
        let spec = ArchitectureSpec::from_manifest(&m);
        assert!(
            spec.capabilities
                .batched_decode_structural_rejections()
                .is_empty()
        );
        assert!(
            spec.capabilities
                .is_structurally_dense_full_attention_only()
        );
        let reg = lookup_architecture("qwen3").expect("registered");
        assert!(reg.dense_batched_decode_candidate);
    }

    /// WS-T3: Gemma 4 must not flip dense batched-decode candidacy without
    /// certification artifacts. Structural pilot still supports SWA helpers.
    #[test]
    fn gemma4_families_remain_non_candidates_until_cert() {
        for label in ["gemma4", "gemma4_unified", "gemma4_vl", "gemma4_assistant"] {
            let reg = lookup_architecture(label).expect(label);
            assert!(
                !reg.dense_batched_decode_candidate,
                "{label} must stay non-candidate until SWA/SWA+MoE cert lands"
            );
            assert!(
                !reg.cert_gate_note.is_empty(),
                "{label} must document the cert gate"
            );
        }
        // Structural readiness: SWA note present on text gemma4.
        let g4 = lookup_architecture("gemma4").unwrap();
        assert!(
            g4.cert_gate_note.contains("SWA") || g4.cert_gate_note.contains("swa"),
            "gemma4 cert note should mention SWA structural path"
        );
    }

    #[test]
    fn qwen3_vl_text_only_rides_certified_qwen3_batch_candidate() {
        let q = lookup_architecture("qwen3_vl").unwrap();
        assert!(q.dense_batched_decode_candidate);
    }

    #[test]
    fn registry_default_generation_defers_to_manifest_diffusion() {
        let mut m = base_manifest("gemma4", 2);
        m.diffusion.canvas_size = Some(256);
        let derived = GenerationKind::from_manifest(&m);
        let resolved = default_generation_for_family("gemma4", derived);
        assert_eq!(resolved, GenerationKind::BlockDiffusion);
    }

    #[test]
    fn empty_caps_report_no_attention() {
        let caps = StructuralCapabilities::default();
        let reasons = caps.batched_decode_structural_rejections();
        assert!(reasons.contains(&"no_attention"));
    }
}
