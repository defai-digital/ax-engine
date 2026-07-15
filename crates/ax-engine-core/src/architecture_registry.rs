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
        assert_eq!(resolve_layer_forward_route("not_a_family"), None);
    }

    #[test]
    fn structural_caps_reject_hybrid_without_family_allowlist() {
        let mut m = base_manifest("qwen3_5", 4);
        m.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: Some(4),
            num_value_heads: Some(4),
            num_key_heads: Some(4),
            key_head_dim: Some(32),
            value_head_dim: Some(32),
            conv_kernel_dim: Some(4),
        };
        let spec = ArchitectureSpec::from_manifest(&m);
        let reasons = spec
            .capabilities
            .dense_batched_decode_structural_rejections();
        assert!(
            reasons.contains(&"linear_attention"),
            "expected linear_attention rejection, got {reasons:?}"
        );
        assert!(
            !spec
                .capabilities
                .is_structurally_dense_full_attention_only()
        );
    }

    #[test]
    fn structural_caps_accept_dense_qwen_shape() {
        let m = base_manifest("qwen3", 4);
        let spec = ArchitectureSpec::from_manifest(&m);
        assert!(
            spec.capabilities
                .dense_batched_decode_structural_rejections()
                .is_empty()
        );
        assert!(
            spec.capabilities
                .is_structurally_dense_full_attention_only()
        );
        let reg = lookup_architecture("qwen3").expect("registered");
        assert!(reg.dense_batched_decode_candidate);
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
        let reasons = caps.dense_batched_decode_structural_rejections();
        assert!(reasons.contains(&"no_attention"));
    }
}
