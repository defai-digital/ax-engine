//! Multimodal prefill adapters (ADR-038 Phase 4).
//!
//! Vision/audio inputs are **prefill-side input producers** that inject soft
//! tokens / embeddings into the same generation strategy. They are not a
//! separate generation engine or `GenerationKind`.

use crate::generation::GenerationKind;
use crate::request::RequestMultimodalInputs;

/// Modalities that can contribute prefill-side material.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PrefillModality {
    Vision,
    Audio,
    Text,
}

/// Description of multimodal material attached to a request for prefill.
///
/// The adapter always **feeds** an existing [`GenerationKind`]; it never
/// replaces it. Multimodal Gemma4 unified remains `Autoregressive` (or
/// whatever the backbone strategy is).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultimodalPrefillAdapter {
    /// Modalities present on this request.
    pub modalities: Vec<PrefillModality>,
    /// Generation strategy the adapter injects into (never a new kind).
    pub feeds_generation: GenerationKind,
    /// True when soft-token / projector work must run during prefill.
    pub requires_prefill_projection: bool,
}

impl MultimodalPrefillAdapter {
    /// Build an adapter from request multimodal inputs + the model's generation kind.
    ///
    /// Empty multimodal inputs yield a text-only adapter that still reports the
    /// same `feeds_generation`.
    pub fn from_request_inputs(
        inputs: &RequestMultimodalInputs,
        generation: GenerationKind,
    ) -> Self {
        if inputs.is_empty() {
            return Self {
                modalities: vec![PrefillModality::Text],
                feeds_generation: generation,
                requires_prefill_projection: false,
            };
        }

        let mut modalities = vec![PrefillModality::Text];
        if let Some(unified) = inputs.gemma4_unified.as_ref() {
            if !unified.images.is_empty() {
                modalities.push(PrefillModality::Vision);
            }
            if !unified.audios.is_empty() {
                modalities.push(PrefillModality::Audio);
            }
        }
        if inputs
            .unlimited_ocr
            .as_ref()
            .is_some_and(|ocr| !ocr.images.is_empty())
            && !modalities.contains(&PrefillModality::Vision)
        {
            modalities.push(PrefillModality::Vision);
        }

        Self {
            modalities,
            feeds_generation: generation,
            requires_prefill_projection: true,
        }
    }

    /// Multimodal never invents a parallel generation engine.
    pub fn is_separate_generation_engine(&self) -> bool {
        false
    }

    pub fn has_vision(&self) -> bool {
        self.modalities.contains(&PrefillModality::Vision)
    }

    pub fn has_audio(&self) -> bool {
        self.modalities.contains(&PrefillModality::Audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::ArchitectureSpec;
    use crate::gemma4_unified::{
        Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedModality, Gemma4UnifiedRuntimeInputs,
        Gemma4UnifiedTokenSpan,
    };
    use crate::generation::{FirstVisibleEventKind, GenerationKind, GenerationStrategyDescriptor};
    use crate::model::{
        AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeDiffusionConfig,
        NativeLinearAttentionConfig, NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus,
        NativeTensorDataType, NativeTensorFormat, NativeTensorRole, NativeTensorSpec,
        WeightSanitize,
    };

    fn base_manifest(family: &str) -> NativeModelManifest {
        NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: family.to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 2,
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
    fn text_only_adapter_feeds_same_generation() {
        let generation = GenerationKind::Autoregressive;
        let adapter = MultimodalPrefillAdapter::from_request_inputs(
            &RequestMultimodalInputs::default(),
            generation,
        );
        assert_eq!(adapter.feeds_generation, GenerationKind::Autoregressive);
        assert!(!adapter.requires_prefill_projection);
        assert!(!adapter.is_separate_generation_engine());
        assert_eq!(adapter.modalities, vec![PrefillModality::Text]);
    }

    #[test]
    fn vision_adapter_does_not_change_generation_kind() {
        let inputs = RequestMultimodalInputs {
            gemma4_unified: Some(Gemma4UnifiedRuntimeInputs {
                images: vec![Gemma4UnifiedImageRuntimeInput {
                    span: Gemma4UnifiedTokenSpan {
                        modality: Gemma4UnifiedModality::Image,
                        placeholder_index: 0,
                        replacement_start: 0,
                        soft_token_count: 4,
                        replacement_token_count: 4,
                    },
                    pixel_values: vec![0.0; 16],
                    pixel_position_ids: vec![[0, 0]; 4],
                }],
                audios: Vec::new(),
                videos: Vec::new(),
            }),
            unlimited_ocr: None,
        };

        let backbone = GenerationKind::Autoregressive;
        let adapter = MultimodalPrefillAdapter::from_request_inputs(&inputs, backbone);
        assert!(adapter.has_vision());
        assert!(!adapter.has_audio());
        assert!(adapter.requires_prefill_projection);
        // Critical Phase 4 invariant: multimodal is not a new generation engine.
        assert_eq!(adapter.feeds_generation, backbone);
        assert!(!adapter.is_separate_generation_engine());
        assert_eq!(
            GenerationStrategyDescriptor::for_kind(adapter.feeds_generation).first_visible,
            FirstVisibleEventKind::FirstToken
        );
    }

    #[test]
    fn multimodal_capable_architecture_stays_autoregressive() {
        use std::path::PathBuf;

        let mut m = base_manifest("gemma4");
        m.tensors.push(NativeTensorSpec {
            name: "vision.projection.weight".into(),
            role: NativeTensorRole::Gemma4UnifiedVisionProjection,
            layer_index: None,
            dtype: NativeTensorDataType::Bf16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![128, 128],
            file: PathBuf::from("weights.safetensors"),
            offset_bytes: 0,
            length_bytes: 128 * 128 * 2,
        });
        let spec = ArchitectureSpec::from_manifest(&m);
        assert!(spec.capabilities.is_multimodal_capable);
        assert_eq!(spec.generation, GenerationKind::Autoregressive);
        assert!(!spec.capabilities.is_diffusion);
    }
}
