//! Gemma 4 E2B/E4B VL path (WS-V1 / R-V1).
//!
//! ViT vision tower (+ optional Conformer audio) soft tokens scatter into the
//! gemma4 text backbone via the ADR-038 multimodal adapter. Implementation
//! reuses the encoder-free connector embedding path from
//! [`crate::gemma4_unified`] when vision weights are present on the checkpoint
//! (convert maps E2B/E4B tower tensors onto the same roles as gemma4_unified).
//!
//! Text-only prompts on a VL checkpoint use the standard gemma4 AR graph.
//! Image/audio without tower weights fail closed with
//! [`Gemma4VlError::MissingVisionWeights`] / [`MissingAudioWeights`].

use ax_engine_core::gemma4_unified::{Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedRuntimeInputs};
use ax_engine_core::vl_geometry::{scatter_merge_indices, vit_soft_token_count};
use thiserror::Error;

use crate::gemma4_unified::{
    Gemma4UnifiedChunkEmbeddings, Gemma4UnifiedError as UnifiedEmbedError, build_chunk_embeddings,
};
use crate::model::ModelConfig;
use crate::weights::ModelWeights;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Gemma4VlError {
    #[error("gemma4_vl requires vision tower weights for image input")]
    MissingVisionWeights,
    #[error("gemma4_vl requires Conformer audio weights for audio input")]
    MissingAudioWeights,
    #[error("gemma4_vl image geometry invalid: {0}")]
    InvalidGeometry(String),
    #[error("gemma4_vl scatter merge failed: {0}")]
    Scatter(String),
    #[error("gemma4_vl embed failed: {0}")]
    Embed(String),
}

impl From<UnifiedEmbedError> for Gemma4VlError {
    fn from(value: UnifiedEmbedError) -> Self {
        match value {
            UnifiedEmbedError::MissingVisionWeights
            | UnifiedEmbedError::MissingVideoVisionWeights => Self::MissingVisionWeights,
            UnifiedEmbedError::MissingAudioWeights => Self::MissingAudioWeights,
            other => Self::Embed(other.to_string()),
        }
    }
}

/// Image request geometry for the E2B/E4B ViT path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4VlImageGeometry {
    pub height: u32,
    pub width: u32,
    pub patch_size: u32,
    pub merge_size: u32,
    pub max_soft_tokens: u32,
}

impl Gemma4VlImageGeometry {
    pub fn soft_token_count(self) -> Result<u32, Gemma4VlError> {
        vit_soft_token_count(
            self.height,
            self.width,
            self.patch_size,
            self.merge_size,
            self.max_soft_tokens,
        )
        .ok_or_else(|| {
            Gemma4VlError::InvalidGeometry(format!(
                "h={} w={} patch={} merge={} max={}",
                self.height, self.width, self.patch_size, self.merge_size, self.max_soft_tokens
            ))
        })
    }
}

/// Plan soft-token scatter positions for one or more images in a prompt.
pub fn plan_image_scatter(
    placeholder_positions: &[usize],
    geometries: &[Gemma4VlImageGeometry],
) -> Result<Vec<usize>, Gemma4VlError> {
    if placeholder_positions.len() != geometries.len() {
        return Err(Gemma4VlError::Scatter(format!(
            "placeholders {} != images {}",
            placeholder_positions.len(),
            geometries.len()
        )));
    }
    let mut counts = Vec::with_capacity(geometries.len());
    for g in geometries {
        counts.push(g.soft_token_count()?);
    }
    scatter_merge_indices(placeholder_positions, &counts).map_err(Gemma4VlError::Scatter)
}

pub fn is_gemma4_vl_family(model_family: &str) -> bool {
    model_family == "gemma4_vl"
}

/// Text-only decode on a VL checkpoint reuses the certified gemma4 standard path.
pub fn text_only_uses_standard_gemma4_path(model_family: &str, has_media: bool) -> bool {
    is_gemma4_vl_family(model_family) && !has_media
}

/// True when the loaded weights include a vision tower (required for images).
pub fn has_vision_tower(weights: &ModelWeights) -> bool {
    weights.gemma4_unified_vision.is_some()
}

/// True when the loaded weights include an audio tower.
pub fn has_audio_tower(weights: &ModelWeights) -> bool {
    weights.gemma4_unified_audio.is_some()
}

/// Prefill-side soft-token injection for gemma4_vl (ADR-038 adapter).
///
/// Fails closed if the request carries image/audio/video but the corresponding
/// tower weights are absent — never silent text-only degrade.
pub(crate) fn build_vl_prefill_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    inputs: &Gemma4UnifiedRuntimeInputs,
) -> Result<Gemma4UnifiedChunkEmbeddings, Gemma4VlError> {
    if !inputs.images.is_empty() && !has_vision_tower(weights) {
        return Err(Gemma4VlError::MissingVisionWeights);
    }
    if !inputs.videos.is_empty() && !has_vision_tower(weights) {
        return Err(Gemma4VlError::MissingVisionWeights);
    }
    if !inputs.audios.is_empty() && !has_audio_tower(weights) {
        return Err(Gemma4VlError::MissingAudioWeights);
    }
    // Reuse the gemma4_unified connector embedder (same role map for E2B/E4B
    // convert extras). Soft tokens scatter into the text residual stream.
    Ok(build_chunk_embeddings(cfg, weights, token_ids, 0, inputs)?)
}

/// Validate a single image tensor against soft-token geometry before serve.
pub fn validate_image_soft_tokens(
    image: &Gemma4UnifiedImageRuntimeInput,
    geometry: Gemma4VlImageGeometry,
) -> Result<(), Gemma4VlError> {
    let expected = geometry.soft_token_count()?;
    if image.span.soft_token_count != expected {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "soft_token_count {} != geometry {}",
            image.span.soft_token_count, expected
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::gemma4_unified::{Gemma4UnifiedModality, Gemma4UnifiedTokenSpan};

    #[test]
    fn soft_token_and_scatter_plan() {
        let g = Gemma4VlImageGeometry {
            height: 224,
            width: 224,
            patch_size: 14,
            merge_size: 2,
            max_soft_tokens: 256,
        };
        assert_eq!(g.soft_token_count().unwrap(), 64);
        let idx = plan_image_scatter(&[3], &[g]).unwrap();
        assert_eq!(idx.len(), 64);
        assert_eq!(idx[0], 3);
        assert_eq!(idx[63], 3 + 63);
    }

    #[test]
    fn text_only_route() {
        assert!(text_only_uses_standard_gemma4_path("gemma4_vl", false));
        assert!(!text_only_uses_standard_gemma4_path("gemma4_vl", true));
        assert!(!text_only_uses_standard_gemma4_path("gemma4", false));
    }

    #[test]
    fn validate_image_soft_tokens_enforces_geometry() {
        let g = Gemma4VlImageGeometry {
            height: 224,
            width: 224,
            patch_size: 14,
            merge_size: 2,
            max_soft_tokens: 256,
        };
        let image = Gemma4UnifiedImageRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Image,
                placeholder_index: 0,
                replacement_start: 0,
                soft_token_count: 64,
                replacement_token_count: 66,
            },
            pixel_values: vec![0.0; 64 * 3],
            pixel_position_ids: vec![[0, 0]; 64],
        };
        assert!(validate_image_soft_tokens(&image, g).is_ok());
        let bad = Gemma4UnifiedImageRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                soft_token_count: 8,
                ..image.span
            },
            ..image
        };
        assert!(validate_image_soft_tokens(&bad, g).is_err());
    }
}
