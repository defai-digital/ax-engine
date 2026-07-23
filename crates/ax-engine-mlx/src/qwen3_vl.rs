//! Qwen3-VL / Qwen3-VL-MoE path (WS-V2 / R-V2).
//!
//! ViT encoder with 2-D RoPE, DeepStack injection, MRoPE text positions, and
//! LLaVA-style scatter merge into certified qwen3 / qwen3-MoE graphs.
//! Text-only prompts on a VL checkpoint must route identically to qwen3.

use ax_engine_core::vl_geometry::{
    MropeSections, deepstack_injection_layers, mrope_position_ids, scatter_merge_indices,
    vit_soft_token_count,
};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Qwen3VlError {
    #[error("qwen3_vl requires vision tower weights for image input")]
    MissingVisionWeights,
    #[error("qwen3_vl image geometry invalid: {0}")]
    InvalidGeometry(String),
    #[error("qwen3_vl scatter merge failed: {0}")]
    Scatter(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3VlImageGeometry {
    pub height: u32,
    pub width: u32,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub max_soft_tokens: u32,
}

impl Qwen3VlImageGeometry {
    pub fn grid_hw(self) -> Result<(u32, u32), Qwen3VlError> {
        if self.patch_size == 0 || self.spatial_merge_size == 0 {
            return Err(Qwen3VlError::InvalidGeometry(
                "patch_size and spatial_merge_size must be > 0".into(),
            ));
        }
        let gh = self.height / self.patch_size / self.spatial_merge_size;
        let gw = self.width / self.patch_size / self.spatial_merge_size;
        if gh == 0 || gw == 0 {
            return Err(Qwen3VlError::InvalidGeometry(format!(
                "grid collapsed for {}x{} patch={} merge={}",
                self.height, self.width, self.patch_size, self.spatial_merge_size
            )));
        }
        Ok((gh, gw))
    }

    pub fn soft_token_count(self) -> Result<u32, Qwen3VlError> {
        vit_soft_token_count(
            self.height,
            self.width,
            self.patch_size,
            self.spatial_merge_size,
            self.max_soft_tokens,
        )
        .ok_or_else(|| {
            Qwen3VlError::InvalidGeometry(format!(
                "h={} w={} patch={} merge={} max={}",
                self.height,
                self.width,
                self.patch_size,
                self.spatial_merge_size,
                self.max_soft_tokens
            ))
        })
    }

    pub fn mrope_sections(self) -> Result<MropeSections, Qwen3VlError> {
        let (h, w) = self.grid_hw()?;
        Ok(MropeSections::for_image(h, w))
    }
}

pub fn plan_image_scatter(
    placeholder_positions: &[usize],
    geometries: &[Qwen3VlImageGeometry],
) -> Result<Vec<usize>, Qwen3VlError> {
    if placeholder_positions.len() != geometries.len() {
        return Err(Qwen3VlError::Scatter(format!(
            "placeholders {} != images {}",
            placeholder_positions.len(),
            geometries.len()
        )));
    }
    let mut counts = Vec::with_capacity(geometries.len());
    for g in geometries {
        counts.push(g.soft_token_count()?);
    }
    scatter_merge_indices(placeholder_positions, &counts).map_err(Qwen3VlError::Scatter)
}

pub fn plan_mrope_for_images(
    geometries: &[Qwen3VlImageGeometry],
) -> Result<Vec<u32>, Qwen3VlError> {
    let mut all = Vec::new();
    for g in geometries {
        all.extend(mrope_position_ids(g.mrope_sections()?));
    }
    Ok(all)
}

pub fn deepstack_layers(num_feature_maps: usize, language_layers: u32) -> Vec<u32> {
    deepstack_injection_layers(num_feature_maps, language_layers)
}

pub fn is_qwen3_vl_family(model_family: &str) -> bool {
    matches!(model_family, "qwen3_vl" | "qwen3_vl_moe")
}

/// Text-only prompts on VL checkpoints must use the certified qwen3 decode path.
pub fn text_only_decode_family(model_family: &str) -> Option<&'static str> {
    match model_family {
        "qwen3_vl" => Some("qwen3"),
        "qwen3_vl_moe" => Some("qwen3"), // MoE graph shared with qwen3_moe maps
        _ => None,
    }
}

/// Prefill gate: image inputs require a vision tower. Until ViT weights are
/// mapped for this family, fail closed (never silent text-only degrade).
pub fn require_vision_for_images(
    has_image_inputs: bool,
    has_vision_weights: bool,
) -> Result<(), Qwen3VlError> {
    if has_image_inputs && !has_vision_weights {
        return Err(Qwen3VlError::MissingVisionWeights);
    }
    Ok(())
}

/// Decode-route selection for a loaded VL checkpoint.
pub fn select_decode_route(
    model_family: &str,
    has_media: bool,
) -> Result<&'static str, Qwen3VlError> {
    if !is_qwen3_vl_family(model_family) {
        // Non-VL families: caller keeps its own label (not static).
        return Ok(if model_family.is_empty() {
            "unknown"
        } else {
            // Map known families to static labels when possible.
            match model_family {
                "qwen3" => "qwen3",
                "qwen3_5" | "qwen3.5" => "qwen3_5",
                "qwen3_next" | "qwen3.6" | "qwen3_6" => "qwen3_next",
                "gemma4" => "gemma4",
                "gemma4_vl" => "gemma4_vl",
                _ => "other",
            }
        });
    }
    if has_media {
        // Multimodal path: still text-graph after scatter; vision tower required.
        return Ok(if model_family == "qwen3_vl_moe" {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        });
    }
    Ok(text_only_decode_family(model_family).unwrap_or("qwen3"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_and_mrope() {
        let g = Qwen3VlImageGeometry {
            height: 448,
            width: 448,
            patch_size: 14,
            spatial_merge_size: 2,
            max_soft_tokens: 1024,
        };
        // 448/14=32, /2=16 → 16×16=256 soft tokens
        assert_eq!(g.soft_token_count().unwrap(), 256);
        let sections = g.mrope_sections().unwrap();
        assert_eq!(sections.height, 16);
        assert_eq!(sections.width, 16);
        let ids = plan_mrope_for_images(&[g]).unwrap();
        assert_eq!(ids.len(), 256 * 3);
    }

    #[test]
    fn deepstack_and_text_route() {
        assert_eq!(deepstack_layers(3, 36), vec![0, 1, 2]);
        assert_eq!(text_only_decode_family("qwen3_vl"), Some("qwen3"));
        assert!(is_qwen3_vl_family("qwen3_vl_moe"));
    }

    #[test]
    fn scatter_plan() {
        let g = Qwen3VlImageGeometry {
            height: 28,
            width: 28,
            patch_size: 14,
            spatial_merge_size: 1,
            max_soft_tokens: 16,
        };
        // 2×2 patches, merge 1 → 4 soft tokens
        assert_eq!(g.soft_token_count().unwrap(), 4);
        let idx = plan_image_scatter(&[1], &[g]).unwrap();
        assert_eq!(idx, vec![1, 2, 3, 4]);
    }

    #[test]
    fn vision_required_and_text_route() {
        assert!(require_vision_for_images(true, false).is_err());
        assert!(require_vision_for_images(true, true).is_ok());
        assert!(require_vision_for_images(false, false).is_ok());
        assert_eq!(select_decode_route("qwen3_vl", false).unwrap(), "qwen3");
        assert_eq!(select_decode_route("qwen3_vl", true).unwrap(), "qwen3_vl");
    }
}
