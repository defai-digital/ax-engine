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
            // Videos are per-frame image-processor output through the same
            // vision tower, so they are a vision modality too.
            if !unified.images.is_empty() || !unified.videos.is_empty() {
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
        if inputs
            .qwen3_vl
            .as_ref()
            .is_some_and(|vl| !vl.images.is_empty())
            && !modalities.contains(&PrefillModality::Vision)
        {
            modalities.push(PrefillModality::Vision);
        }
        if inputs
            .minicpm_v46
            .as_ref()
            .is_some_and(|vl| !vl.images.is_empty())
            && !modalities.contains(&PrefillModality::Vision)
        {
            modalities.push(PrefillModality::Vision);
        }
        if let Some(omni) = inputs.nemotron_omni.as_ref() {
            if !omni.images.is_empty() && !modalities.contains(&PrefillModality::Vision) {
                modalities.push(PrefillModality::Vision);
            }
            if !omni.audios.is_empty() && !modalities.contains(&PrefillModality::Audio) {
                modalities.push(PrefillModality::Audio);
            }
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
    use crate::model::{NativeTensorDataType, NativeTensorRole, NativeTensorSpec, base_manifest};

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
    fn video_only_gemma4_inputs_are_a_vision_modality() {
        let inputs = RequestMultimodalInputs {
            gemma4_unified: Some(Gemma4UnifiedRuntimeInputs {
                images: Vec::new(),
                audios: Vec::new(),
                videos: vec![crate::gemma4_unified::Gemma4UnifiedVideoRuntimeInput {
                    span: Gemma4UnifiedTokenSpan {
                        modality: Gemma4UnifiedModality::Video,
                        placeholder_index: 0,
                        replacement_start: 0,
                        soft_token_count: 4,
                        replacement_token_count: 4,
                    },
                    soft_token_ranges: Vec::new(),
                    pixel_values: vec![0.0; 16],
                    pixel_position_ids: vec![[0, 0]; 4],
                    frame_count: 2,
                }],
            }),
            unlimited_ocr: None,
            qwen3_vl: None,
            minicpm_v46: None,
            nemotron_omni: None,
        };

        let adapter =
            MultimodalPrefillAdapter::from_request_inputs(&inputs, GenerationKind::Autoregressive);
        assert!(adapter.has_vision());
        assert!(!adapter.has_audio());
        assert!(adapter.requires_prefill_projection);
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
            qwen3_vl: None,
            minicpm_v46: None,
            nemotron_omni: None,
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

        let mut m = base_manifest("gemma4", 2);
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
