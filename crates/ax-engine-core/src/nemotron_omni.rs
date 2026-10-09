//! Nemotron H Nano Omni request-time media contract.
//!
//! The serving front-end owns bounded image/audio decoding. The native MLX
//! runtime owns the RADIO vision tower, pixel-shuffle projector, and Parakeet
//! audio encoder.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::media_digest::{media_digest_f32, ordered_media_digests_key};

#[derive(Debug, Error, Eq, PartialEq)]
pub enum NemotronOmniRuntimeInputError {
    #[error("Nemotron Omni placeholder_index {0} out of prompt range [0, {1})")]
    PlaceholderOutOfRange(usize, usize),
    #[error("Nemotron Omni media tensor is empty")]
    EmptyMedia,
    #[error("Nemotron Omni soft_token_count must be > 0")]
    ZeroSoftTokens,
    #[error("Nemotron Omni request carries {0} images, exceeding the {1} item limit")]
    TooManyImages(usize, usize),
    #[error("Nemotron Omni request carries {0} audios, exceeding the {1} item limit")]
    TooManyAudios(usize, usize),
    #[error("Nemotron Omni media geometry invalid: {0}")]
    InvalidGeometry(String),
}

/// Per-modality upper bound on media items accepted in one request. Mirrors the
/// serving edge's `MAX_INLINE_IMAGES_PER_REQUEST` so SDK-direct callers get the
/// same bound, and keeps the span bookkeeping bounded.
pub(crate) const MAX_MEDIA_ITEMS_PER_MODALITY: usize = 40;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NemotronOmniImageRuntimeInput {
    /// First `<image>` context token replaced by projected RADIO features.
    pub placeholder_index: usize,
    pub soft_token_count: u32,
    /// Normalized channel-first pixels, `[3, height, width]`.
    pub pixel_values: Vec<f32>,
    pub height: u32,
    pub width: u32,
    pub patch_size: u32,
    #[serde(default = "default_image_downsample_factor")]
    pub spatial_downsample_factor: u32,
}

const fn default_image_downsample_factor() -> u32 {
    2
}

impl NemotronOmniImageRuntimeInput {
    pub(crate) fn validate(&self, prompt_len: usize) -> Result<(), NemotronOmniRuntimeInputError> {
        validate_span(self.placeholder_index, self.soft_token_count, prompt_len)?;
        if self.pixel_values.is_empty() {
            return Err(NemotronOmniRuntimeInputError::EmptyMedia);
        }
        if self.height == 0
            || self.width == 0
            || self.patch_size == 0
            || self.spatial_downsample_factor == 0
        {
            return Err(NemotronOmniRuntimeInputError::InvalidGeometry(
                "height, width, patch_size, and spatial_downsample_factor must be positive"
                    .to_string(),
            ));
        }
        let unit = self
            .patch_size
            .saturating_mul(self.spatial_downsample_factor);
        if !self.height.is_multiple_of(unit) || !self.width.is_multiple_of(unit) {
            return Err(NemotronOmniRuntimeInputError::InvalidGeometry(format!(
                "{}x{} is not divisible by patch_size*downsample_factor {unit}",
                self.height, self.width
            )));
        }
        let expected_pixels = 3usize
            .saturating_mul(self.height as usize)
            .saturating_mul(self.width as usize);
        if self.pixel_values.len() != expected_pixels {
            return Err(NemotronOmniRuntimeInputError::InvalidGeometry(format!(
                "pixel_values len {} != 3*height*width {expected_pixels}",
                self.pixel_values.len()
            )));
        }
        let expected_soft = u64::from(self.height / unit) * u64::from(self.width / unit);
        if u64::from(self.soft_token_count) != expected_soft {
            return Err(NemotronOmniRuntimeInputError::InvalidGeometry(format!(
                "soft_token_count {} != downsampled patch count {expected_soft}",
                self.soft_token_count
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NemotronOmniAudioRuntimeInput {
    /// First `<so_embedding>` token replaced by projected Parakeet features.
    pub placeholder_index: usize,
    pub soft_token_count: u32,
    /// Mono waveform at `sample_rate`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl NemotronOmniAudioRuntimeInput {
    pub(crate) fn validate(&self, prompt_len: usize) -> Result<(), NemotronOmniRuntimeInputError> {
        validate_span(self.placeholder_index, self.soft_token_count, prompt_len)?;
        if self.samples.is_empty() {
            return Err(NemotronOmniRuntimeInputError::EmptyMedia);
        }
        if self.sample_rate == 0 || self.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(NemotronOmniRuntimeInputError::InvalidGeometry(
                "sample_rate must be positive and waveform samples must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_span(
    placeholder_index: usize,
    soft_token_count: u32,
    prompt_len: usize,
) -> Result<(), NemotronOmniRuntimeInputError> {
    if soft_token_count == 0 {
        return Err(NemotronOmniRuntimeInputError::ZeroSoftTokens);
    }
    if placeholder_index >= prompt_len {
        return Err(NemotronOmniRuntimeInputError::PlaceholderOutOfRange(
            placeholder_index,
            prompt_len,
        ));
    }
    let end = placeholder_index.saturating_add(soft_token_count as usize);
    if end > prompt_len {
        return Err(NemotronOmniRuntimeInputError::InvalidGeometry(format!(
            "placeholder span {placeholder_index}..{end} exceeds prompt length {prompt_len}"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NemotronOmniRuntimeInputs {
    #[serde(default)]
    pub images: Vec<NemotronOmniImageRuntimeInput>,
    #[serde(default)]
    pub audios: Vec<NemotronOmniAudioRuntimeInput>,
}

impl NemotronOmniRuntimeInputs {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty()
    }

    pub fn validate_for_prompt_len(
        &self,
        prompt_len: usize,
    ) -> Result<(), NemotronOmniRuntimeInputError> {
        if self.images.len() > MAX_MEDIA_ITEMS_PER_MODALITY {
            return Err(NemotronOmniRuntimeInputError::TooManyImages(
                self.images.len(),
                MAX_MEDIA_ITEMS_PER_MODALITY,
            ));
        }
        if self.audios.len() > MAX_MEDIA_ITEMS_PER_MODALITY {
            return Err(NemotronOmniRuntimeInputError::TooManyAudios(
                self.audios.len(),
                MAX_MEDIA_ITEMS_PER_MODALITY,
            ));
        }
        // The guards above bound the loop. No capacity hint is derived from the
        // request, so the allocation size stays independent of caller input.
        let mut spans = Vec::new();
        for image in &self.images {
            image.validate(prompt_len)?;
            spans.push((image.placeholder_index, image.soft_token_count as usize));
        }
        for audio in &self.audios {
            audio.validate(prompt_len)?;
            spans.push((audio.placeholder_index, audio.soft_token_count as usize));
        }
        // Each prompt token can be replaced by one media tensor at most.
        spans.sort_unstable();
        for pair in spans.windows(2) {
            let (previous_start, previous_count) = pair[0];
            let (next_start, _) = pair[1];
            if previous_start + previous_count > next_start {
                return Err(NemotronOmniRuntimeInputError::InvalidGeometry(format!(
                    "placeholder span starting at {previous_start} overlaps the span starting at {next_start}"
                )));
            }
        }
        Ok(())
    }

    pub fn media_prefix_key(&self, model_fingerprint: &str) -> String {
        let mut digests = Vec::with_capacity(self.images.len() + self.audios.len());
        digests.extend(self.images.iter().map(|image| {
            media_digest_f32(
                &image.pixel_values,
                image.soft_token_count,
                model_fingerprint,
            )
        }));
        digests.extend(self.audios.iter().map(|audio| {
            media_digest_f32(&audio.samples, audio.soft_token_count, model_fingerprint)
        }));
        if digests.is_empty() {
            String::new()
        } else {
            ordered_media_digests_key(&digests)
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn unit_image(placeholder_index: usize) -> NemotronOmniImageRuntimeInput {
        NemotronOmniImageRuntimeInput {
            placeholder_index,
            soft_token_count: 1,
            pixel_values: vec![0.0; 3 * 32 * 32],
            height: 32,
            width: 32,
            patch_size: 16,
            spatial_downsample_factor: 2,
        }
    }

    fn unit_audio(placeholder_index: usize) -> NemotronOmniAudioRuntimeInput {
        NemotronOmniAudioRuntimeInput {
            placeholder_index,
            soft_token_count: 1,
            samples: vec![0.0; 1600],
            sample_rate: 16_000,
        }
    }

    #[test]
    fn validates_image_and_audio_spans() {
        let inputs = NemotronOmniRuntimeInputs {
            images: vec![NemotronOmniImageRuntimeInput {
                placeholder_index: 1,
                soft_token_count: 4,
                pixel_values: vec![0.0; 3 * 64 * 64],
                height: 64,
                width: 64,
                patch_size: 16,
                spatial_downsample_factor: 2,
            }],
            audios: vec![NemotronOmniAudioRuntimeInput {
                placeholder_index: 6,
                soft_token_count: 2,
                samples: vec![0.0; 1600],
                sample_rate: 16_000,
            }],
        };
        assert!(inputs.validate_for_prompt_len(8).is_ok());

        // Audio span [4, 6) overlaps the image span [1, 5).
        let mut overlapping = inputs.clone();
        overlapping.audios[0].placeholder_index = 4;
        let error = overlapping
            .validate_for_prompt_len(8)
            .expect_err("overlapping spans must be rejected");
        assert!(error.to_string().contains("overlaps"), "{error}");

        // Two images at the same placeholder overlap too.
        let mut duplicate = inputs.clone();
        duplicate.images.push(duplicate.images[0].clone());
        assert!(duplicate.validate_for_prompt_len(8).is_err());
    }

    #[test]
    fn rejects_media_over_the_per_modality_limit() {
        let images_over = NemotronOmniRuntimeInputs {
            images: (0..=MAX_MEDIA_ITEMS_PER_MODALITY).map(unit_image).collect(),
            audios: Vec::new(),
        };
        let error = images_over
            .validate_for_prompt_len(MAX_MEDIA_ITEMS_PER_MODALITY + 1)
            .expect_err("image count over the limit must be rejected");
        assert_eq!(
            error,
            NemotronOmniRuntimeInputError::TooManyImages(
                MAX_MEDIA_ITEMS_PER_MODALITY + 1,
                MAX_MEDIA_ITEMS_PER_MODALITY
            )
        );

        let audios_over = NemotronOmniRuntimeInputs {
            images: Vec::new(),
            audios: (0..=MAX_MEDIA_ITEMS_PER_MODALITY).map(unit_audio).collect(),
        };
        let error = audios_over
            .validate_for_prompt_len(MAX_MEDIA_ITEMS_PER_MODALITY + 1)
            .expect_err("audio count over the limit must be rejected");
        assert_eq!(
            error,
            NemotronOmniRuntimeInputError::TooManyAudios(
                MAX_MEDIA_ITEMS_PER_MODALITY + 1,
                MAX_MEDIA_ITEMS_PER_MODALITY
            )
        );

        // The cap is per modality: a full batch of each still passes.
        let at_limit = NemotronOmniRuntimeInputs {
            images: (0..MAX_MEDIA_ITEMS_PER_MODALITY).map(unit_image).collect(),
            audios: (0..MAX_MEDIA_ITEMS_PER_MODALITY)
                .map(|index| unit_audio(index + MAX_MEDIA_ITEMS_PER_MODALITY))
                .collect(),
        };
        assert!(
            at_limit
                .validate_for_prompt_len(MAX_MEDIA_ITEMS_PER_MODALITY * 2)
                .is_ok()
        );
    }
}
