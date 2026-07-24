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
        let expected_patches = self.grid_t.saturating_mul(grid_h).saturating_mul(grid_w);
        if self.num_patches != expected_patches {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                "num_patches {} != grid_t*grid_h*grid_w {}",
                self.num_patches, expected_patches
            )));
        }
        let expected = (self.num_patches as usize).saturating_mul(self.patch_dim as usize);
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
        for image in &self.images {
            image.validate(prompt_len)?;
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
