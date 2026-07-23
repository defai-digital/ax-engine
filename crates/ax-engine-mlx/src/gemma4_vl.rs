//! Gemma 4 E2B/E4B VL path (WS-V1 / R-V1).
//!
//! Encoder towers (ViT vision + Conformer audio) produce soft-token embeddings
//! that scatter into the gemma4 text backbone via the ADR-038 adapter.
//! Text-only prompts on a VL checkpoint use the standard gemma4 graph.
//!
//! Weight roles for the towers land through convert maps; until a checkpoint
//! provides them, [`Gemma4VlError::MissingVisionWeights`] fails closed.

use ax_engine_core::vl_geometry::{scatter_merge_indices, vit_soft_token_count};
use thiserror::Error;

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

/// Capability probe: family label is gemma4_vl (or convert alias).
pub fn is_gemma4_vl_family(model_family: &str) -> bool {
    model_family == "gemma4_vl"
}

/// Text-only decode on a VL checkpoint reuses the certified gemma4 standard path.
pub fn text_only_uses_standard_gemma4_path(model_family: &str, has_media: bool) -> bool {
    is_gemma4_vl_family(model_family) && !has_media
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
