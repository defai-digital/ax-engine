//! Qwen3-VL runtime request contract (WS-V2 / R-V2).
//!
//! Prefill-side image patches + placeholder positions for LLaVA-style scatter
//! into the certified qwen3 text graph. HF vision-tower weight mapping remains
//! a load-path concern; this module is the request schema and media identity.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::media_digest::{media_digest_f32, ordered_media_digests_key};

#[derive(Debug, Error, Eq, PartialEq)]
pub enum Qwen3VlRuntimeInputError {
    #[error("qwen3_vl image placeholder_index {0} out of prompt range [0, {1})")]
    PlaceholderOutOfRange(usize, usize),
    #[error("qwen3_vl image patches empty")]
    EmptyPatches,
    #[error("qwen3_vl soft_token_count must be > 0")]
    ZeroSoftTokens,
    #[error("qwen3_vl geometry invalid: {0}")]
    InvalidGeometry(String),
}

/// One image for Qwen3-VL prefill (already patch-projected or raw patches).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Qwen3VlImageRuntimeInput {
    /// Absolute index of the `<image>` placeholder token in the prompt.
    pub placeholder_index: usize,
    /// Soft tokens this image expands to after spatial merge.
    pub soft_token_count: u32,
    /// Patch tensor values (row-major, typically `[num_patches * patch_dim]`).
    pub patches: Vec<f32>,
    /// Patch count along sequence (S in `[1, S, patch_dim]`).
    pub num_patches: u32,
    /// Patch feature dim (last dim of patch embed input).
    pub patch_dim: u32,
    /// Temporal grid entries consumed by the vision tower. Still images use
    /// one; videos use one entry per temporal patch group.
    #[serde(default = "default_grid_t")]
    pub grid_t: u32,
    pub height: u32,
    pub width: u32,
    pub patch_size: u32,
    /// Frames folded into each Conv3D patch (two for Qwen3-VL/Qwen3.5).
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    /// Selects the checkpoint's video token when constructing MRoPE axes.
    #[serde(default)]
    pub is_video: bool,
}

const fn default_grid_t() -> u32 {
    1
}

const fn default_temporal_patch_size() -> u32 {
    2
}

impl Qwen3VlImageRuntimeInput {
    pub fn validate(&self, prompt_len: usize) -> Result<(), Qwen3VlRuntimeInputError> {
        if self.soft_token_count == 0 {
            return Err(Qwen3VlRuntimeInputError::ZeroSoftTokens);
        }
        if self.patches.is_empty() {
            return Err(Qwen3VlRuntimeInputError::EmptyPatches);
        }
        if self.placeholder_index >= prompt_len {
            return Err(Qwen3VlRuntimeInputError::PlaceholderOutOfRange(
                self.placeholder_index,
                prompt_len,
            ));
        }
        if self.num_patches == 0 || self.patch_dim == 0 {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(
                "num_patches and patch_dim must be > 0".into(),
            ));
        }
        if self.grid_t == 0
            || self.patch_size == 0
            || self.temporal_patch_size == 0
            || self.spatial_merge_size == 0
        {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(
                "grid_t, patch_size, temporal_patch_size, and spatial_merge_size must be > 0"
                    .into(),
            ));
        }
        if !self.height.is_multiple_of(self.patch_size)
            || !self.width.is_multiple_of(self.patch_size)
        {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                "{}x{} is not divisible by patch_size {}",
                self.height, self.width, self.patch_size
            )));
        }
        let grid_h = self.height / self.patch_size;
        let grid_w = self.width / self.patch_size;
        if !grid_h.is_multiple_of(self.spatial_merge_size)
            || !grid_w.is_multiple_of(self.spatial_merge_size)
        {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                "patch grid {grid_h}x{grid_w} is not divisible by spatial_merge_size {}",
                self.spatial_merge_size
            )));
        }
        let expected_patches = self
            .grid_t
            .checked_mul(grid_h)
            .and_then(|value| value.checked_mul(grid_w))
            .ok_or_else(|| {
                Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                    "grid {}x{grid_h}x{grid_w} overflows the patch count",
                    self.grid_t
                ))
            })?;
        if self.num_patches != expected_patches {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                "num_patches {} != grid_t*grid_h*grid_w {}",
                self.num_patches, expected_patches
            )));
        }
        // One soft token per merged grid cell per temporal entry; the vision
        // tower produces exactly this many rows and the scatter reserves
        // `soft_token_count` prompt slots, so the two must agree up front.
        let merge = self.spatial_merge_size;
        let expected_soft_tokens = self
            .grid_t
            .checked_mul(grid_h / merge)
            .and_then(|value| value.checked_mul(grid_w / merge))
            .ok_or_else(|| {
                Qwen3VlRuntimeInputError::InvalidGeometry("soft token count overflow".into())
            })?;
        if self.soft_token_count != expected_soft_tokens {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                "soft_token_count {} != grid_t*(grid_h/merge)*(grid_w/merge) {expected_soft_tokens}",
                self.soft_token_count
            )));
        }
        let expected = (self.num_patches as usize)
            .checked_mul(self.patch_dim as usize)
            .ok_or_else(|| {
                Qwen3VlRuntimeInputError::InvalidGeometry("patch tensor size overflow".into())
            })?;
        if self.patches.len() != expected {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                "patches len {} != num_patches*patch_dim {}",
                self.patches.len(),
                expected
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Qwen3VlRuntimeInputs {
    #[serde(default)]
    pub images: Vec<Qwen3VlImageRuntimeInput>,
}

impl Qwen3VlRuntimeInputs {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn validate_for_prompt_len(
        &self,
        prompt_len: usize,
    ) -> Result<(), Qwen3VlRuntimeInputError> {
        let mut placeholders = std::collections::HashSet::with_capacity(self.images.len());
        for image in &self.images {
            image.validate(prompt_len)?;
            if !placeholders.insert(image.placeholder_index) {
                return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                    "placeholder_index {} is used by more than one image",
                    image.placeholder_index
                )));
            }
        }
        Ok(())
    }

    /// Ordered media digests for prefix-cache identity (WS-M3).
    pub fn media_prefix_key(&self, model_fingerprint: &str) -> String {
        let mut digests = Vec::with_capacity(self.images.len());
        for image in &self.images {
            digests.push(media_digest_f32(
                &image.patches,
                image.soft_token_count,
                model_fingerprint,
            ));
        }
        if digests.is_empty() {
            String::new()
        } else {
            ordered_media_digests_key(&digests)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_image() -> Qwen3VlImageRuntimeInput {
        Qwen3VlImageRuntimeInput {
            placeholder_index: 1,
            soft_token_count: 4,
            patches: vec![0.1; 4 * 6],
            num_patches: 4,
            patch_dim: 6,
            grid_t: 1,
            height: 28,
            width: 28,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 1,
            is_video: false,
        }
    }

    #[test]
    fn soft_token_count_must_match_merged_grid() {
        // 28x28 with patch 14 is a 2x2 patch grid; merge 2 collapses it to one
        // soft token, so claiming four is rejected.
        let image = Qwen3VlImageRuntimeInput {
            spatial_merge_size: 2,
            ..sample_image()
        };
        let error = image.validate(8).expect_err("merged grid mismatch");
        assert!(error.to_string().contains("soft_token_count"), "{error}");
        let image = Qwen3VlImageRuntimeInput {
            spatial_merge_size: 2,
            soft_token_count: 1,
            ..sample_image()
        };
        assert!(image.validate(8).is_ok());
    }

    #[test]
    fn duplicate_placeholders_and_overflowing_grids_are_rejected() {
        let inputs = Qwen3VlRuntimeInputs {
            images: vec![sample_image(), sample_image()],
        };
        let error = inputs
            .validate_for_prompt_len(8)
            .expect_err("two images cannot share a placeholder");
        assert!(error.to_string().contains("placeholder_index"), "{error}");

        let image = Qwen3VlImageRuntimeInput {
            grid_t: 2,
            height: 65_536,
            width: 32_768,
            patch_size: 1,
            spatial_merge_size: 1,
            num_patches: u32::MAX,
            patch_dim: 1,
            patches: vec![0.0; 8],
            ..sample_image()
        };
        let error = image.validate(8).expect_err("grid product overflow");
        assert!(error.to_string().contains("overflow"), "{error}");
    }

    #[test]
    fn validate_and_media_key() {
        let inputs = Qwen3VlRuntimeInputs {
            images: vec![sample_image()],
        };
        assert!(inputs.validate_for_prompt_len(8).is_ok());
        assert!(inputs.validate_for_prompt_len(1).is_err());
        let mut other = inputs.clone();
        other.images[0].patches[0] = 0.9;
        assert_ne!(inputs.media_prefix_key("fp"), other.media_prefix_key("fp"));
        assert!(!inputs.media_prefix_key("fp").is_empty());
    }
}
