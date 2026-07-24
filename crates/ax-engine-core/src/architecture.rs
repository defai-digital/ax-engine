//! Structural architecture view derived from [`NativeModelManifest`] (ADR-038).
//!
//! This is a *portable* description of layer composition and generation kind.
//! It does not replace the on-disk manifest schema and is not a second wire
//! format in Phase 1.

use crate::generation::{
    GenerationKind, GenerationStrategyDescriptor, is_block_diffusion_manifest,
    is_encoder_embed_manifest,
};
use crate::model::{NativeModelManifest, NativeTensorRole};

/// Attention mechanism for one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    Full,
    Sliding {
        window: u32,
    },
    LinearGatedDelta,
    Mla,
    /// Non-causal / bidirectional encoder-style attention.
    Bidirectional,
}

/// Feed-forward block for one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FfnKind {
    DenseSwiglu,
    DenseGeglu,
    MoE { experts: u32, active: u32 },
    Mxfp4MoE { experts: u32, active: u32 },
}

/// Physical cache / state layout for one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheKind {
    StandardKv,
    SlidingRing {
        window: u32,
    },
    MlaLatent,
    LinearRecurrent,
    /// Encoder-only or cache-free passes.
    None,
}

/// Per-layer structural descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerSpec {
    pub index: u32,
    pub attention: AttentionKind,
    pub ffn: FfnKind,
    pub cache: CacheKind,
    pub kv_source_layer: Option<u32>,
}

/// Aggregated structural gates for route eligibility (prefer over family name).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StructuralCapabilities {
    pub has_full_attention: bool,
    pub has_sliding_window: bool,
    pub has_linear_attention: bool,
    pub has_mla: bool,
    pub has_moe: bool,
    /// True when MoE layers use the qwen3 softmax router that the batched FFN
    /// implements (`moe_router_qwen3` + non-MXFP4 expert stacks). Explicit
    /// family/capability bit — **not** inferred from linear attention.
    /// Gemma4, GPT-OSS (MXFP4), GLM, and DeepSeek routers stay false.
    pub batched_qwen3_moe_router: bool,
    pub has_layer_gating: bool,
    pub is_diffusion: bool,
    pub is_encoder_embed: bool,
    pub is_multimodal_capable: bool,
}

impl StructuralCapabilities {
    pub fn from_layers(
        layers: &[LayerSpec],
        generation: GenerationKind,
        has_layer_gating: bool,
        is_multimodal_capable: bool,
    ) -> Self {
        let mut caps = Self {
            has_layer_gating,
            is_diffusion: matches!(generation, GenerationKind::BlockDiffusion),
            is_encoder_embed: matches!(generation, GenerationKind::EncoderEmbed),
            is_multimodal_capable,
            ..Self::default()
        };
        for layer in layers {
            match layer.attention {
                AttentionKind::Full => caps.has_full_attention = true,
                AttentionKind::Sliding { .. } => {
                    caps.has_sliding_window = true;
                    // Sliding layers still participate in hybrid full/sliding stacks.
                }
                AttentionKind::LinearGatedDelta => caps.has_linear_attention = true,
                AttentionKind::Mla => caps.has_mla = true,
                AttentionKind::Bidirectional => caps.has_full_attention = true,
            }
            if matches!(layer.ffn, FfnKind::MoE { .. } | FfnKind::Mxfp4MoE { .. }) {
                caps.has_moe = true;
            }
        }
        caps
    }

    /// Structural rejections for continuous batched decode (dense full-attention
    /// pilot **and** hybrid linear / qwen3-MoE extensions).
    ///
    /// Prefer this over family-string allowlists (ADR-038 Phase 3). Numerical
    /// certification remains a separate gate at the runner (ADR-003 D5).
    ///
    /// Formerly `dense_batched_decode_structural_rejections` — renamed after
    /// Phase 3.7 admitted linear + qwen3-MoE hybrids; the dense-only pilot
    /// predicate is [`Self::is_structurally_dense_full_attention_only`].
    pub fn batched_decode_structural_rejections(self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if self.is_diffusion {
            reasons.push("diffusion");
        }
        if self.is_encoder_embed {
            reasons.push("encoder_embed");
        }
        // Full/sliding stacks use a per-layer batched mask. Requests whose
        // private sliding cache has already compacted are rejected later at
        // cohort admission because their absolute prefix cannot be recovered.
        // MoE is supported only via the explicit qwen3 router capability bit
        // (set from model family at ArchitectureSpec construction). Linear
        // attention is an independent attention path — it no longer proxies
        // router eligibility. Gemma4 / GPT-OSS / GLM / DeepSeek MoE stay out.
        if self.has_moe && !self.batched_qwen3_moe_router {
            reasons.push("moe");
        }
        // Linear attention (gated-delta) is handled by the batched linear path
        // (`BatchedLinearState` + the batch-native gated_delta kernel), so it is
        // no longer a structural rejection. Numerical certification remains a
        // separate per-model gate at the runner.
        if self.has_mla {
            reasons.push("mla");
        }
        if self.has_layer_gating {
            reasons.push("layer_gating");
        }
        if !self.has_full_attention && !self.has_sliding_window && !self.has_linear_attention {
            // Degenerate empty graph — treat as not structurally eligible.
            reasons.push("no_attention");
        }
        reasons
    }

    /// Backward-compatible alias for [`Self::batched_decode_structural_rejections`].
    #[inline]
    pub fn dense_batched_decode_structural_rejections(self) -> Vec<&'static str> {
        self.batched_decode_structural_rejections()
    }

    /// True when structural caps match the dense full-attention batched pilot
    /// shape (still requires numerical certification separately).
    pub fn is_structurally_dense_full_attention_only(self) -> bool {
        self.batched_decode_structural_rejections().is_empty()
            && self.has_full_attention
            && !self.has_sliding_window
            && !self.has_linear_attention
            && !self.has_mla
            && !self.has_moe
    }

    /// Structural blockers for a **Gemma-style** continuous decode pilot
    /// (interleaved SWA is allowed; MoE / MLA / linear / diffusion are not).
    ///
    /// This is not the dense full-attention Qwen pilot. A future SWA-aware
    /// multi-request path would still need numerical certification and
    /// windowed KV views (see `AX_MLX_MULTI_TOKEN_WINDOW_VIEWS`).
    pub fn gemma_swa_decode_structural_rejections(self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if self.is_diffusion {
            reasons.push("diffusion");
        }
        if self.is_encoder_embed {
            reasons.push("encoder_embed");
        }
        if self.has_moe {
            reasons.push("moe");
        }
        if self.has_linear_attention {
            reasons.push("linear_attention");
        }
        if self.has_mla {
            reasons.push("mla");
        }
        // Layer gating (per-layer embeds) is Gemma-E2B/E4B-specific complexity.
        if self.has_layer_gating {
            reasons.push("layer_gating");
        }
        if !self.has_full_attention && !self.has_sliding_window {
            reasons.push("no_attention");
        }
        // Pure dense full-attn without SWA can use the dense pilot instead.
        if self.has_full_attention && !self.has_sliding_window {
            reasons.push("not_interleaved_swa");
        }
        reasons
    }

    /// True when caps look like interleaved SWA Gemma text (not MoE/gating).
    pub fn is_structurally_gemma_swa_text_candidate(self) -> bool {
        self.gemma_swa_decode_structural_rejections().is_empty()
            && self.has_sliding_window
            && self.has_full_attention
    }
}

/// Full structural view of a model architecture for scheduling and routing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchitectureSpec {
    /// Human/debug label copied from `model_family` — not an eligibility key.
    pub family_label: String,
    pub layer_count: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub layers: Vec<LayerSpec>,
    pub generation: GenerationKind,
    pub strategy: GenerationStrategyDescriptor,
    pub capabilities: StructuralCapabilities,
}

impl ArchitectureSpec {
    /// Derive a structural architecture from a validated (or test) manifest.
    pub fn from_manifest(manifest: &NativeModelManifest) -> Self {
        let generation = GenerationKind::from_manifest(manifest);
        let strategy = GenerationStrategyDescriptor::for_kind(generation);
        let layers = build_layer_specs(manifest, generation);
        let has_layer_gating = manifest.hidden_size_per_layer_input > 0
            || manifest.tensors.iter().any(|t| {
                matches!(
                    t.role,
                    NativeTensorRole::PerLayerInputGate | NativeTensorRole::PerLayerInputProjection
                )
            });
        let is_multimodal_capable = manifest.tensors.iter().any(|t| {
            matches!(
                t.role,
                NativeTensorRole::Gemma4UnifiedVisionPatchDense
                    | NativeTensorRole::Gemma4UnifiedVisionProjection
                    | NativeTensorRole::Gemma4UnifiedAudioProjection
                    | NativeTensorRole::Qwen3VlVisionPatchEmbed
                    | NativeTensorRole::Qwen3VlVisionMerger
            )
        });
        let mut capabilities = StructuralCapabilities::from_layers(
            &layers,
            generation,
            has_layer_gating,
            is_multimodal_capable,
        );
        // Explicit MoE router kind for the batched path (not inferred from linear).
        capabilities.batched_qwen3_moe_router =
            capabilities.has_moe && family_uses_batched_qwen3_moe_router(&manifest.model_family);

        Self {
            family_label: manifest.model_family.clone(),
            layer_count: manifest.layer_count,
            hidden_size: manifest.hidden_size,
            vocab_size: manifest.vocab_size,
            layers,
            generation,
            strategy,
            capabilities,
        }
    }

    pub fn is_block_diffusion(&self) -> bool {
        matches!(self.generation, GenerationKind::BlockDiffusion)
    }

    pub fn is_encoder_embed(&self) -> bool {
        matches!(self.generation, GenerationKind::EncoderEmbed)
    }
}

fn uses_geglu(family: &str) -> bool {
    matches!(
        family,
        "gemma4" | "gemma4_assistant" | "diffusion_gemma" | "gemma3" | "embeddinggemma"
    )
}

fn uses_mxfp4_moe(family: &str) -> bool {
    family == "gpt_oss"
}

/// Families whose decode MoE path uses `moe_router_qwen3` (the only router
/// the continuous batched FFN implements). Mixtral shares that router layout
/// but is still rejected for sliding-window structure.
fn family_uses_batched_qwen3_moe_router(family: &str) -> bool {
    matches!(
        family,
        "qwen3" | "qwen3_moe" | "qwen3_5" | "qwen3_next" | "mixtral"
    )
}

fn build_layer_specs(manifest: &NativeModelManifest, generation: GenerationKind) -> Vec<LayerSpec> {
    let layer_count = manifest.layer_count;
    let window = manifest.sliding_window_size;
    let full_attn_interval = manifest
        .linear_attention
        .resolved_full_attention_interval(&manifest.model_family);
    let linear_enabled = manifest.linear_attention.is_enabled();
    let mla_enabled = manifest.mla_attention.is_enabled()
        || matches!(
            manifest.model_family.as_str(),
            "glm4_moe_lite" | "deepseek_v3" | "deepseek_v32"
        );
    let moe_enabled = manifest.moe.is_enabled();
    let experts = manifest.moe.expert_count.unwrap_or(0);
    let active = manifest.moe.experts_per_token.unwrap_or(0);
    let first_dense = manifest.moe.first_dense_layers.unwrap_or(0);
    let layer_freq = manifest.moe.layer_freq.unwrap_or(1).max(1);
    let geglu = uses_geglu(&manifest.model_family);
    let mxfp4 = uses_mxfp4_moe(&manifest.model_family);
    let encoder =
        matches!(generation, GenerationKind::EncoderEmbed) || is_encoder_embed_manifest(manifest);

    let mut layers = Vec::with_capacity(layer_count as usize);
    for index in 0..layer_count {
        let is_linear_layer = linear_enabled
            && full_attn_interval.is_some_and(|interval| !(index + 1).is_multiple_of(interval));

        let attention = if encoder {
            AttentionKind::Bidirectional
        } else if mla_enabled {
            AttentionKind::Mla
        } else if is_linear_layer {
            AttentionKind::LinearGatedDelta
        } else if !manifest.layer_types.is_empty() {
            match manifest.layer_types.get(index as usize).map(String::as_str) {
                Some("sliding_attention") => AttentionKind::Sliding {
                    window: window.unwrap_or(0),
                },
                _ => AttentionKind::Full,
            }
        } else if let Some(w) = window {
            // Uniform SWA families (e.g. mistral3): every layer sliding.
            AttentionKind::Sliding { window: w }
        } else {
            AttentionKind::Full
        };

        let is_moe_layer = moe_enabled
            && index >= first_dense
            && (layer_freq == 1 || (index + 1).is_multiple_of(layer_freq));

        let ffn = if is_moe_layer {
            if mxfp4 {
                FfnKind::Mxfp4MoE { experts, active }
            } else {
                FfnKind::MoE { experts, active }
            }
        } else if geglu {
            FfnKind::DenseGeglu
        } else {
            FfnKind::DenseSwiglu
        };

        let cache = if encoder {
            CacheKind::None
        } else {
            match attention {
                AttentionKind::LinearGatedDelta => CacheKind::LinearRecurrent,
                AttentionKind::Mla => CacheKind::MlaLatent,
                AttentionKind::Sliding { window } => CacheKind::SlidingRing { window },
                AttentionKind::Full | AttentionKind::Bidirectional => CacheKind::StandardKv,
            }
        };

        let kv_source_layer = manifest.kv_shared_source_layers.get(&index).copied();

        layers.push(LayerSpec {
            index,
            attention,
            ffn,
            cache,
            kv_source_layer,
        });
    }

    // Diffusion uses the same causal backbone layer specs for the encoder
    // path; bidirectionality is carried by GenerationKind, not per-layer
    // attention kinds, so denoise can still share Full/Sliding structure.
    let _ = is_block_diffusion_manifest(manifest);

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeDiffusionConfig,
        NativeLinearAttentionConfig, NativeMlaAttentionConfig, NativeModelManifest,
        NativeMoeConfig, NativeRuntimeStatus, NativeTensorFormat, WeightSanitize,
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
    fn gemma_swa_text_candidate_allows_interleaved_swa_not_moe() {
        let mut m = base_manifest("gemma4", 4);
        m.layer_types = vec![
            "sliding_attention".into(),
            "full_attention".into(),
            "sliding_attention".into(),
            "full_attention".into(),
        ];
        m.sliding_window_size = Some(512);
        let caps = ArchitectureSpec::from_manifest(&m).capabilities;
        assert!(caps.has_sliding_window);
        assert!(caps.has_full_attention);
        assert!(
            caps.is_structurally_gemma_swa_text_candidate(),
            "interleaved SWA dense Gemma text should be a SWA pilot candidate: {:?}",
            caps.gemma_swa_decode_structural_rejections()
        );
        // The production batched route now carries a per-layer SWA mask.
        assert!(
            !caps
                .batched_decode_structural_rejections()
                .contains(&"sliding_window")
        );

        // MoE blocks the Gemma SWA pilot.
        m.moe.expert_count = Some(8);
        m.moe.experts_per_token = Some(2);
        m.moe.expert_intermediate_size = Some(64);
        let caps_moe = ArchitectureSpec::from_manifest(&m).capabilities;
        assert!(
            caps_moe
                .gemma_swa_decode_structural_rejections()
                .contains(&"moe")
        );
        assert!(!caps_moe.is_structurally_gemma_swa_text_candidate());
    }

    #[test]
    fn qwen3_dense_is_ar_full_swiglu() {
        let m = base_manifest("qwen3", 4);
        let spec = ArchitectureSpec::from_manifest(&m);
        assert_eq!(spec.generation, GenerationKind::Autoregressive);
        assert_eq!(spec.layers.len(), 4);
        assert!(spec.layers.iter().all(|l| {
            l.attention == AttentionKind::Full
                && l.ffn == FfnKind::DenseSwiglu
                && l.cache == CacheKind::StandardKv
        }));
        assert!(spec.capabilities.has_full_attention);
        assert!(!spec.capabilities.is_diffusion);
        assert!(!spec.capabilities.has_linear_attention);
    }

    #[test]
    fn qwen35_hybrid_linear_layers() {
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
        // layers 0,1,2 linear; layer 3 (index 3, 1-based 4) full
        assert_eq!(spec.layers[0].attention, AttentionKind::LinearGatedDelta);
        assert_eq!(spec.layers[0].cache, CacheKind::LinearRecurrent);
        assert_eq!(spec.layers[3].attention, AttentionKind::Full);
        assert!(spec.capabilities.has_linear_attention);
        assert!(spec.capabilities.has_full_attention);
    }

    #[test]
    fn gemma4_interleaved_swa() {
        let mut m = base_manifest("gemma4", 4);
        m.sliding_window_size = Some(512);
        m.layer_types = vec![
            "sliding_attention".into(),
            "sliding_attention".into(),
            "sliding_attention".into(),
            "full_attention".into(),
        ];
        let spec = ArchitectureSpec::from_manifest(&m);
        assert_eq!(
            spec.layers[0].attention,
            AttentionKind::Sliding { window: 512 }
        );
        assert_eq!(spec.layers[0].cache, CacheKind::SlidingRing { window: 512 });
        assert_eq!(spec.layers[3].attention, AttentionKind::Full);
        assert!(spec.capabilities.has_sliding_window);
        assert!(spec.capabilities.has_full_attention);
        assert!(matches!(spec.layers[0].ffn, FfnKind::DenseGeglu));
    }

    #[test]
    fn diffusion_gemma_generation_and_moe() {
        let mut m = base_manifest("diffusion_gemma", 2);
        m.diffusion = NativeDiffusionConfig {
            canvas_size: Some(256),
            ..Default::default()
        };
        m.moe = NativeMoeConfig {
            expert_count: Some(16),
            experts_per_token: Some(4),
            expert_intermediate_size: Some(128),
            ..Default::default()
        };
        let spec = ArchitectureSpec::from_manifest(&m);
        assert!(spec.is_block_diffusion());
        assert!(spec.capabilities.is_diffusion);
        assert!(spec.capabilities.has_moe);
        assert!(matches!(
            spec.layers[0].ffn,
            FfnKind::MoE {
                experts: 16,
                active: 4
            }
        ));
        assert_eq!(
            spec.strategy.first_visible,
            crate::generation::FirstVisibleEventKind::FirstBlock
        );
    }

    #[test]
    fn embeddinggemma_encoder() {
        let m = base_manifest("embeddinggemma", 2);
        let spec = ArchitectureSpec::from_manifest(&m);
        assert!(spec.is_encoder_embed());
        assert!(spec.capabilities.is_encoder_embed);
        assert!(spec.layers.iter().all(|l| {
            l.attention == AttentionKind::Bidirectional && l.cache == CacheKind::None
        }));
    }

    #[test]
    fn mla_family_marks_mla_cache() {
        let mut m = base_manifest("glm4_moe_lite", 2);
        m.mla_attention = NativeMlaAttentionConfig {
            q_lora_rank: Some(64),
            kv_lora_rank: Some(64),
            qk_nope_head_dim: Some(64),
            qk_rope_head_dim: Some(32),
            value_head_dim: Some(64),
        };
        let spec = ArchitectureSpec::from_manifest(&m);
        assert!(spec.capabilities.has_mla);
        assert!(
            spec.layers
                .iter()
                .all(|l| l.attention == AttentionKind::Mla && l.cache == CacheKind::MlaLatent)
        );
    }
}
