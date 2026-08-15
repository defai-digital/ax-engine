//! Static architecture registration (ADR-038 Phase 3+).
//!
//! Maps known family labels to default generation kind, layer-forward route,
//! and certification notes. Convert and runtime gates should prefer this
//! registry + structural caps over ad-hoc string allowlists when adding
//! hybrid variants.

use crate::generation::GenerationKind;
use crate::support_tier::ModelSupportTier;

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
    /// DeepSeek V4 (Flash): dedicated repo-owned graph with experimental
    /// certification status.
    DeepseekV4,
    Mistral3,
    Mixtral,
    GptOss,
    /// Nemotron-H hybrid: per-layer Mamba-2 / attention / ReLU² MoE mixers.
    NemotronH,
}

/// Whether an architecture artifact may be loaded as the primary MLX runner.
///
/// Registration alone is not admission: auxiliary artifacts such as an MTP
/// assistant are known to the converter and runtime but must be attached to a
/// primary model instead of being loaded as a standalone generation runner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxRunnerAdmission {
    /// The manifest may enter the primary MLX runner validation pipeline.
    Primary,
    /// The manifest is a known auxiliary artifact and cannot run standalone.
    AuxiliaryOnly,
}

impl MlxRunnerAdmission {
    /// Whether this registration is eligible for primary-runner validation.
    pub const fn allows_primary(self) -> bool {
        matches!(self, Self::Primary)
    }
}

impl LayerForwardRoute {
    /// Stable telemetry code for route decisions.
    pub const fn telemetry_code(self) -> u32 {
        match self {
            Self::Standard => 0,
            Self::Llama4 => 1,
            Self::GlmMoeLite => 2,
            Self::DeepseekV3 => 3,
            Self::DeepseekV4 => 8,
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
            Self::DeepseekV4 => "deepseek_v4",
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
    /// Primary-versus-auxiliary admission policy for the MLX runner.
    pub mlx_runner_admission: MlxRunnerAdmission,
    /// Default generation paradigm when the manifest does not force another.
    pub default_generation: GenerationKind,
    /// Layer-forward implementation route (ADR-038 composition boundary).
    pub layer_forward_route: LayerForwardRoute,
    /// Whether continuous dense batched decode is *structurally* in scope
    /// (still requires numerical certification).
    pub dense_batched_decode_candidate: bool,
    /// Human-readable cert / support note for docs and diagnostics.
    pub cert_gate_note: &'static str,
    /// Three-tier model quality grade (see [`crate::support_tier`]).
    pub support_tier: ModelSupportTier,
}

/// Forward-compatible name for a complete model-family descriptor.
///
/// The existing type name and struct literal spelling remain stable because
/// repository smoke tooling source-parses registry rows today.
pub type FamilyDescriptor = ArchitectureRegistration;

/// All statically registered architecture labels.
///
/// Adding a hybrid that reuses existing primitives should primarily add a row
/// here plus convert mapping — not a new eligibility allowlist of family names.
pub static ARCHITECTURE_REGISTRY: &[ArchitectureRegistration] = &[
    ArchitectureRegistration {
        family_label: "qwen3",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: true,
        cert_gate_note: "dense full-attention AR; batched decode when certified",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "qwen3_5",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "hybrid linear+full; structural rejections include linear_attention",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "qwen3_next",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "hybrid gated-delta / MoE; capability-gated, not name-allowlisted",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "minicpmv4_6",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Qwen3.5 hybrid text backbone with MiniCPM-V vision prefill",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "llama3",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: true,
        cert_gate_note: "dense full-attention AR when structurally dense",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "gemma3",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Gemma3 SWA text backbone; standard path",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "gemma4",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "interleaved SWA / optional MoE; dense pilot rejects SWA+MoE; SWA text may use gemma_swa structural helper + multi_token_window_views",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "gemma4_assistant",
        mlx_runner_admission: MlxRunnerAdmission::AuxiliaryOnly,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "assistant MTP drafter; not dense-batch candidate",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "gemma4_unified",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "multimodal prefill adapters feed AR generation",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "gemma4_vl",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Gemma 4 E2B/E4B ViT+Conformer towers into gemma4 AR backbone (WS-V1)",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "qwen3_vl",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: true,
        cert_gate_note: "Qwen3-VL dense: text path rides certified qwen3 batched decode when text-only",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "qwen3_vl_moe",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Qwen3-VL-MoE; text decode shares qwen3-MoE graphs; batch cert separate",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "muse_glimmer",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Muse-Glimmer dense SWA + gated attention; convert recognized; native decode incubating (not on Gemma SWA allowlist)",
        support_tier: ModelSupportTier::Experimental,
    },
    ArchitectureRegistration {
        family_label: "diffusion_gemma",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::BlockDiffusion,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "block diffusion; generation kind BlockDiffusion",
        support_tier: ModelSupportTier::Experimental,
    },
    ArchitectureRegistration {
        family_label: "embeddinggemma",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::EncoderEmbed,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "encoder embed strategy; not a decode path",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "nemotron_embed",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::EncoderEmbed,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Nemotron 3 Embed: bidirectional Ministral encoder + mean pool",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "glm4_moe_lite",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::GlmMoeLite,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MLA + MoE; structural rejections",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "deepseek_v3",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::DeepseekV3,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MLA + MoE",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "deepseek_v32",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::DeepseekV3,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MLA + MoE",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "deepseek_v4",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::DeepseekV4,
        dense_batched_decode_candidate: false,
        cert_gate_note: "sparse attention + hash-routed MoE; repo-owned graph with limited smoke evidence; no certification evidence",
        support_tier: ModelSupportTier::Experimental,
    },
    ArchitectureRegistration {
        family_label: "mistral3",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Mistral3,
        dense_batched_decode_candidate: false,
        cert_gate_note: "uniform SWA; sliding_window rejection",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "mixtral",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Mixtral,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MoE",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "llama4",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Llama4,
        dense_batched_decode_candidate: false,
        cert_gate_note: "iRoPE / MoE hybrid",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "gpt_oss",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::GptOss,
        dense_batched_decode_candidate: false,
        cert_gate_note: "MXFP4 MoE",
        support_tier: ModelSupportTier::Certified,
    },
    ArchitectureRegistration {
        family_label: "nemotron_h",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::NemotronH,
        dense_batched_decode_candidate: false,
        cert_gate_note: "hybrid Mamba-2 + GQA + ReLU2 MoE; pattern-driven mixers",
        support_tier: ModelSupportTier::Compatible,
    },
    ArchitectureRegistration {
        family_label: "unlimited_ocr",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Unlimited-OCR multimodal: dual vision + SWA MoE language tower",
        support_tier: ModelSupportTier::Compatible,
    },
    // Whisper uses a dedicated encoder-decoder ASR runtime (`ax-engine-mlx`
    // whisper module + audio endpoints). It is still a convert-supported
    // direct family and must appear here so support_tier / lookup stay honest
    // (Wave 0 DI-W0: convert `family_name=whisper` was previously registry-orphan).
    ArchitectureRegistration {
        family_label: "whisper",
        mlx_runner_admission: MlxRunnerAdmission::Primary,
        default_generation: GenerationKind::Autoregressive,
        layer_forward_route: LayerForwardRoute::Standard,
        dense_batched_decode_candidate: false,
        cert_gate_note: "Whisper large-v3-turbo ASR: dedicated encoder-decoder; audio endpoints only",
        support_tier: ModelSupportTier::Compatible,
    },
];

/// Look up a static registration by manifest `model_family` label.
pub fn lookup_architecture(family_label: &str) -> Option<&'static ArchitectureRegistration> {
    ARCHITECTURE_REGISTRY
        .iter()
        .find(|entry| entry.family_label == family_label)
}

/// Resolve the MLX runner admission policy for a registered family.
///
/// `None` means the family is unknown, which is distinct from a known
/// [`MlxRunnerAdmission::AuxiliaryOnly`] artifact.
pub fn mlx_runner_admission_for_family(family_label: &str) -> Option<MlxRunnerAdmission> {
    lookup_architecture(family_label).map(|entry| entry.mlx_runner_admission)
}

/// Return whether a registered family may enter primary MLX runner validation.
///
/// Unknown labels and registered auxiliary-only artifacts both fail closed.
pub fn is_primary_mlx_runner_family(family_label: &str) -> bool {
    mlx_runner_admission_for_family(family_label).is_some_and(MlxRunnerAdmission::allows_primary)
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
    use std::collections::BTreeSet;

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
            rope_beta_fast: None,
            rope_beta_slow: None,
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
            global_kv_head_count: None,
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
            deepseek_v4: Default::default(),
            weight_sanitize: WeightSanitize::default(),
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
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

        let muse = lookup_architecture("muse_glimmer").expect("muse_glimmer registered");
        assert!(!muse.dense_batched_decode_candidate);
        assert_eq!(muse.support_tier, ModelSupportTier::Experimental);
        assert_eq!(muse.layer_forward_route, LayerForwardRoute::Standard);
    }

    #[test]
    fn registry_family_labels_are_unique() {
        let mut seen = BTreeSet::new();
        for entry in ARCHITECTURE_REGISTRY {
            assert!(
                seen.insert(entry.family_label),
                "duplicate architecture registration for {}",
                entry.family_label
            );
        }
    }

    #[test]
    fn primary_mlx_runner_admission_excludes_auxiliary_and_unknown_artifacts() {
        let auxiliary_families = ARCHITECTURE_REGISTRY
            .iter()
            .filter(|entry| !entry.mlx_runner_admission.allows_primary())
            .map(|entry| entry.family_label)
            .collect::<Vec<_>>();

        assert_eq!(auxiliary_families, vec!["gemma4_assistant"]);
        assert_eq!(
            mlx_runner_admission_for_family("gemma4_assistant"),
            Some(MlxRunnerAdmission::AuxiliaryOnly)
        );
        assert_eq!(mlx_runner_admission_for_family("not_a_family"), None);
        assert!(is_primary_mlx_runner_family("qwen3"));
        assert!(is_primary_mlx_runner_family("deepseek_v4"));
        assert!(!is_primary_mlx_runner_family("gemma4_assistant"));
        assert!(!is_primary_mlx_runner_family("not_a_family"));
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
            resolve_layer_forward_route("deepseek_v4"),
            Some(LayerForwardRoute::DeepseekV4)
        );
        assert_eq!(
            resolve_layer_forward_route("nemotron_h"),
            Some(LayerForwardRoute::NemotronH)
        );
        assert_eq!(
            resolve_layer_forward_route("nemotron_embed"),
            Some(LayerForwardRoute::Standard)
        );
        assert_eq!(
            lookup_architecture("nemotron_embed").map(|e| e.default_generation),
            Some(GenerationKind::EncoderEmbed)
        );
        assert_eq!(
            resolve_layer_forward_route("unlimited_ocr"),
            Some(LayerForwardRoute::Standard)
        );
        assert_eq!(
            resolve_layer_forward_route("gpt_oss"),
            Some(LayerForwardRoute::GptOss)
        );
        assert_eq!(
            resolve_layer_forward_route("whisper"),
            Some(LayerForwardRoute::Standard)
        );
        assert_eq!(resolve_layer_forward_route("not_a_family"), None);
    }

    /// Convert emits these `family_name` values; each must have a registry row
    /// so support_tier and layer-forward lookup do not silently fall through.
    #[test]
    fn convert_family_names_are_registered() {
        // Keep in lockstep with `convert/model_family.rs` `family_name:` arms.
        const CONVERT_FAMILY_NAMES: &[&str] = &[
            "deepseek_v3",
            "deepseek_v4",
            "diffusion_gemma",
            "embeddinggemma",
            "gemma4",
            "gemma4_assistant",
            "gemma4_unified",
            "gemma4_vl",
            "glm4_moe_lite",
            "gpt_oss",
            "llama3",
            "llama4",
            "minicpmv4_6",
            "mistral3",
            "mixtral",
            "nemotron_embed",
            "nemotron_h",
            "qwen3",
            "qwen3_5",
            "qwen3_next",
            "qwen3_vl",
            "qwen3_vl_moe",
            "unlimited_ocr",
            "whisper",
        ];
        for label in CONVERT_FAMILY_NAMES {
            assert!(
                lookup_architecture(label).is_some(),
                "convert family_name {label:?} missing from ARCHITECTURE_REGISTRY"
            );
        }
    }

    #[test]
    fn architecture_registry_labels_are_unique() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for entry in ARCHITECTURE_REGISTRY {
            assert!(
                seen.insert(entry.family_label),
                "duplicate ARCHITECTURE_REGISTRY family_label {}",
                entry.family_label
            );
        }
    }

    #[test]
    fn gemma4_unified_model_type_emits_unified_family_not_certified_gemma4() {
        // Validate tier honesty without depending on private convert modules:
        // registry rows must disagree so convert's family_name choice matters.
        assert_eq!(
            crate::support_tier::support_tier_for_family("gemma4"),
            crate::support_tier::ModelSupportTier::Certified
        );
        assert_eq!(
            crate::support_tier::support_tier_for_family("gemma4_unified"),
            crate::support_tier::ModelSupportTier::Compatible
        );
        // Convert contract is covered in convert/tests.rs
        // (`converts_gemma4_unified_text_without_tower_tensors`).
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
        for family in [
            "gemma4",
            "gpt_oss",
            "glm4_moe_lite",
            "deepseek_v3",
            "deepseek_v4",
        ] {
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
