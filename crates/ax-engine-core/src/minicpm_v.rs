//! MiniCPM-V 4.6 request-time image contract.
//!
//! Pixels are resized and normalized by the serving front-end. The MLX
//! runtime owns the SigLIP tower, VitMerger, and final pixel-shuffle merger.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::media_digest::{media_digest_f32, ordered_media_digests_key};

#[derive(Debug, Error, Eq, PartialEq)]
pub enum MiniCpmV46RuntimeInputError {
    #[error("MiniCPM-V 4.6 image placeholder_index {0} out of prompt range [0, {1})")]
    PlaceholderOutOfRange(usize, usize),
    #[error("MiniCPM-V 4.6 image pixel_values are empty")]
    EmptyPixels,
    #[error("MiniCPM-V 4.6 soft_token_count must be > 0")]
    ZeroSoftTokens,
    #[error("MiniCPM-V 4.6 geometry invalid: {0}")]
    InvalidGeometry(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MiniCpmV46ImageRuntimeInput {
    /// First `<unk>` token between `<image>` and `</image>`.
    pub placeholder_index: usize,
    /// Number of `<unk>` tokens replaced by vision features.
    pub soft_token_count: u32,
    /// Normalized NHWC pixels for one image, without the singleton batch axis.
    pub pixel_values: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub patch_size: u32,
    /// Combined spatial reduction from VitMerger and final Merger (4 by default).
    #[serde(default = "default_spatial_downsample_factor")]
    pub spatial_downsample_factor: u32,
}

const fn default_spatial_downsample_factor() -> u32 {
    4
}

impl MiniCpmV46ImageRuntimeInput {
    pub fn validate(&self, prompt_len: usize) -> Result<(), MiniCpmV46RuntimeInputError> {
        if self.soft_token_count == 0 {
            return Err(MiniCpmV46RuntimeInputError::ZeroSoftTokens);
        }
        if self.pixel_values.is_empty() {
            return Err(MiniCpmV46RuntimeInputError::EmptyPixels);
        }
        if self.placeholder_index >= prompt_len {
            return Err(MiniCpmV46RuntimeInputError::PlaceholderOutOfRange(
                self.placeholder_index,
                prompt_len,
            ));
        }
        if self.height == 0
            || self.width == 0
            || self.patch_size == 0
            || self.spatial_downsample_factor == 0
        {
            return Err(MiniCpmV46RuntimeInputError::InvalidGeometry(
                "height, width, patch_size, and spatial_downsample_factor must be positive"
                    .to_string(),
            ));
        }
        let unit = self
            .patch_size
            .saturating_mul(self.spatial_downsample_factor);
        if !self.height.is_multiple_of(unit) || !self.width.is_multiple_of(unit) {
            return Err(MiniCpmV46RuntimeInputError::InvalidGeometry(format!(
                "{}x{} is not divisible by patch_size*downsample_factor {unit}",
                self.height, self.width
            )));
        }
        let expected_pixels = (self.height as usize)
            .saturating_mul(self.width as usize)
            .saturating_mul(3);
        if self.pixel_values.len() != expected_pixels {
            return Err(MiniCpmV46RuntimeInputError::InvalidGeometry(format!(
                "pixel_values len {} != height*width*3 {expected_pixels}",
                self.pixel_values.len()
            )));
        }
        let expected_soft = (self.height / unit).saturating_mul(self.width / unit);
        if self.soft_token_count != expected_soft {
            return Err(MiniCpmV46RuntimeInputError::InvalidGeometry(format!(
                "soft_token_count {} != downsampled grid token count {expected_soft}",
                self.soft_token_count
            )));
        }
        let end = self
            .placeholder_index
            .saturating_add(self.soft_token_count as usize);
        if end > prompt_len {
            return Err(MiniCpmV46RuntimeInputError::InvalidGeometry(format!(
                "placeholder span {}..{end} exceeds prompt length {prompt_len}",
                self.placeholder_index
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct MiniCpmV46RuntimeInputs {
    #[serde(default)]
    pub images: Vec<MiniCpmV46ImageRuntimeInput>,
}

impl MiniCpmV46RuntimeInputs {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn validate_for_prompt_len(
        &self,
        prompt_len: usize,
    ) -> Result<(), MiniCpmV46RuntimeInputError> {
        for image in &self.images {
            image.validate(prompt_len)?;
        }
        Ok(())
    }

    pub fn media_prefix_key(&self, model_fingerprint: &str) -> String {
        let digests: Vec<String> = self
            .images
            .iter()
            .map(|image| {
                media_digest_f32(
                    &image.pixel_values,
                    image.soft_token_count,
                    model_fingerprint,
                )
            })
            .collect();
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

    #[test]
    fn validates_dynamic_image_grid() {
        let image = MiniCpmV46ImageRuntimeInput {
            placeholder_index: 2,
            soft_token_count: 4,
            pixel_values: vec![0.0; 112 * 112 * 3],
            height: 112,
            width: 112,
            patch_size: 14,
            spatial_downsample_factor: 4,
        };
        assert!(image.validate(8).is_ok());
        let mut invalid = image;
        invalid.soft_token_count = 3;
        assert!(invalid.validate(8).is_err());
    }
}
