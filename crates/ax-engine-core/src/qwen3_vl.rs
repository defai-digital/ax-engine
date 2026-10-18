//! Qwen3-VL runtime request contract (WS-V2 / R-V2).
//!
//! Prefill-side image patches + placeholder positions for LLaVA-style scatter
//! into the certified qwen3 text graph. HF vision-tower weight mapping remains
//! a load-path concern; this module is the request schema and media identity.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::media_digest::{RuntimeMediaDigest, ordered_media_digests_key_or_empty};

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum Qwen3VlRuntimeInputError {
    #[error("qwen3_vl image placeholder_index {0} out of prompt range [0, {1})")]
    PlaceholderOutOfRange(usize, usize),
    #[error("qwen3_vl image patches empty")]
    EmptyPatches,
    #[error("qwen3_vl soft_token_count must be > 0")]
    ZeroSoftTokens,
    #[error("qwen3_vl request carries {0} images, exceeding the {1} image limit")]
    TooManyImages(usize, usize),
    #[error("qwen3_vl geometry invalid: {0}")]
    InvalidGeometry(String),
}

/// Matches the serving edge's image limit and bounds validation bookkeeping
/// for native API and SDK-direct callers as well.
const MAX_IMAGES_PER_REQUEST: usize = 40;

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
        let span_end = self
            .placeholder_index
            .checked_add(self.soft_token_count as usize)
            .ok_or_else(|| {
                Qwen3VlRuntimeInputError::InvalidGeometry("visual token range overflow".to_string())
            })?;
        if span_end > prompt_len {
            return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                "placeholder_index {} + soft_token_count {} exceeds prompt length {prompt_len}",
                self.placeholder_index, self.soft_token_count
            )));
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
        if self.images.len() > MAX_IMAGES_PER_REQUEST {
            return Err(Qwen3VlRuntimeInputError::TooManyImages(
                self.images.len(),
                MAX_IMAGES_PER_REQUEST,
            ));
        }
        // The count guard bounds growth before any validation bookkeeping.
        let mut spans = Vec::new();
        for image in &self.images {
            image.validate(prompt_len)?;
            spans.push((image.placeholder_index, image.soft_token_count as usize));
        }
        // Slices must not claim the same prompt tokens: later feature writes
        // would clobber earlier ones. Sorting also catches a repeated start.
        spans.sort_unstable();
        for pair in spans.windows(2) {
            let (previous_start, previous_count) = pair[0];
            let (next_start, _) = pair[1];
            if previous_start + previous_count > next_start {
                return Err(Qwen3VlRuntimeInputError::InvalidGeometry(format!(
                    "placeholder_index span starting at {previous_start} overlaps the span starting at {next_start}"
                )));
            }
        }
        Ok(())
    }

    /// Ordered media digests for prefix-cache identity (WS-M3).
    pub fn media_prefix_key(&self, model_fingerprint: &str) -> String {
        let mut digests = Vec::with_capacity(self.images.len());
        for image in &self.images {
            let mut digest = RuntimeMediaDigest::new("qwen3_vl.patches.f32", model_fingerprint);
            digest.usize(image.placeholder_index);
            digest.u32(image.soft_token_count);
            digest.u32(image.num_patches);
            digest.u32(image.patch_dim);
            digest.u32(image.grid_t);
            digest.u32(image.height);
            digest.u32(image.width);
            digest.u32(image.patch_size);
            digest.u32(image.temporal_patch_size);
            digest.u32(image.spatial_merge_size);
            digest.u32(u32::from(image.is_video));
            digest.f32s(&image.patches);
            digests.push(digest.finish());
        }
        ordered_media_digests_key_or_empty(&digests)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
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
    fn spans_must_fit_and_must_not_overlap() {
        // Placeholder start is in range, but the four-token span runs past the
        // three-token prompt; the runner would otherwise encode the image
        // before failing the scatter.
        let error = sample_image()
            .validate(3)
            .expect_err("span past the prompt");
        assert!(
            error.to_string().contains("exceeds prompt length"),
            "{error}"
        );

        let mut second = sample_image();
        second.placeholder_index = 3;
        let inputs = Qwen3VlRuntimeInputs {
            images: vec![sample_image(), second],
        };
        let error = inputs
            .validate_for_prompt_len(8)
            .expect_err("overlapping spans");
        assert!(error.to_string().contains("placeholder_index"), "{error}");
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

    #[test]
    fn rejects_excess_images_before_validating_geometry() {
        let inputs = Qwen3VlRuntimeInputs {
            images: vec![sample_image(); MAX_IMAGES_PER_REQUEST + 1],
        };
        // Even an invalid prompt must hit the count guard before image checks.
        assert_eq!(
            inputs.validate_for_prompt_len(0),
            Err(Qwen3VlRuntimeInputError::TooManyImages(
                MAX_IMAGES_PER_REQUEST + 1,
                MAX_IMAGES_PER_REQUEST,
            ))
        );
    }

    #[test]
    fn accepts_empty_inputs_and_the_image_count_boundary() {
        assert!(
            Qwen3VlRuntimeInputs::default()
                .validate_for_prompt_len(0)
                .is_ok()
        );
        let inputs = Qwen3VlRuntimeInputs {
            images: (0..MAX_IMAGES_PER_REQUEST)
                .map(|index| Qwen3VlImageRuntimeInput {
                    placeholder_index: index * 4,
                    ..sample_image()
                })
                .collect(),
        };
        assert!(
            inputs
                .validate_for_prompt_len(MAX_IMAGES_PER_REQUEST * 4)
                .is_ok()
        );
    }
}
