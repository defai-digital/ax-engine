//! Runtime request contract for native Unlimited-OCR image prefill.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Base-resolution Unlimited-OCR emits a 16x16 feature grid, one newline
/// embedding per row, and one view separator: `256 + 16 + 1 = 273`.
pub const UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT: u32 = 273;
/// Released processor tile edge for high-resolution document crops.
pub const UNLIMITED_OCR_LOCAL_TILE_SIZE: u32 = 640;
/// Maximum local tiles selected by the released Unlimited-OCR processor.
pub const UNLIMITED_OCR_MAX_LOCAL_TILES: u32 = 32;
/// Each 640px tile becomes a 10x10 projected feature grid.
pub const UNLIMITED_OCR_LOCAL_QUERY_GRID: u32 = 10;

/// Bound decoded RGB payloads before any native allocation or MLX work.
pub const UNLIMITED_OCR_MAX_RGB_BYTES: usize = 256 * 1024 * 1024;
pub const UNLIMITED_OCR_MAX_IMAGE_DIMENSION: u32 = 32_768;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct UnlimitedOcrRuntimeInputs {
    /// Token ID used for the expanded `<image>` soft-token run.
    pub image_token_id: u32,
    /// OCR-2 page marker; absent for the single-image v1 contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_token_id: Option<u32>,
    /// Number of image tokens inserted into the prompt by the public helper.
    pub soft_token_count: u32,
    /// Preserve the released processor's local-tile path for dense documents.
    #[serde(default = "default_cropping")]
    pub cropping: bool,
    /// V1 accepts one image; OCR-2 accepts ordered page images.
    pub images: Vec<UnlimitedOcrImageRuntimeInput>,
}

const fn default_cropping() -> bool {
    true
}

/// Select the same bounded crop grid as the released Unlimited-OCR processor.
///
/// `(1, 1)` means global-only.  Larger images select between two and 32
/// 640px tiles using the closest source aspect ratio.
pub fn unlimited_ocr_crop_grid(width: u32, height: u32, cropping: bool) -> (u32, u32) {
    unlimited_ocr_crop_grid_with_limit(width, height, cropping, UNLIMITED_OCR_MAX_LOCAL_TILES)
}

pub fn unlimited_ocr_crop_grid_with_limit(
    width: u32,
    height: u32,
    cropping: bool,
    max_tiles: u32,
) -> (u32, u32) {
    if !cropping
        || (width <= UNLIMITED_OCR_LOCAL_TILE_SIZE && height <= UNLIMITED_OCR_LOCAL_TILE_SIZE)
        || width == 0
        || height == 0
    {
        return (1, 1);
    }

    let aspect = f64::from(width) / f64::from(height);
    let area = f64::from(width) * f64::from(height);
    let tile_area = f64::from(UNLIMITED_OCR_LOCAL_TILE_SIZE).powi(2);
    let mut ratios = Vec::new();
    for n in 2..=max_tiles {
        for columns in 1..=n {
            for rows in 1..=n {
                let count = columns * rows;
                if (2..=max_tiles).contains(&count) && !ratios.contains(&(columns, rows)) {
                    ratios.push((columns, rows));
                }
            }
        }
    }
    ratios.sort_by_key(|(columns, rows)| columns * rows);

    let mut best = (1, 1);
    let mut best_diff = f64::INFINITY;
    for ratio @ (columns, rows) in ratios {
        let diff = (aspect - f64::from(columns) / f64::from(rows)).abs();
        if diff < best_diff
            || (diff == best_diff && area > 0.5 * tile_area * f64::from(columns * rows))
        {
            best_diff = diff;
            best = ratio;
        }
    }
    best
}

/// Image-token count for the global view plus an optional reassembled tile grid.
pub fn unlimited_ocr_soft_token_count(width: u32, height: u32, cropping: bool) -> u32 {
    let (columns, rows) = unlimited_ocr_crop_grid(width, height, cropping);
    if columns == 1 && rows == 1 {
        return UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT;
    }
    let local_rows = rows * UNLIMITED_OCR_LOCAL_QUERY_GRID;
    let local_columns = columns * UNLIMITED_OCR_LOCAL_QUERY_GRID;
    UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT + local_rows * (local_columns + 1)
}

impl UnlimitedOcrRuntimeInputs {
    fn validate_pages(&self, prompt: &[u32]) -> Result<(), UnlimitedOcrRuntimeInputError> {
        let page_id = self
            .page_token_id
            .ok_or_else(|| invalid_input("page_token_id", "required for OCR-2"))?;
        if self.images.is_empty() || self.images.len() > 32 || page_id == self.image_token_id {
            return Err(invalid_input(
                "images",
                "OCR-2 requires 1..32 ordered pages and distinct marker ids",
            ));
        }
        if prompt.iter().filter(|id| **id == page_id).count() != self.images.len() {
            return Err(invalid_input(
                "page_token_id",
                "OCR-2 requires one <PAGE> marker per input page",
            ));
        }
        let mut runs = Vec::new();
        let mut i = 0;
        while i < prompt.len() {
            if prompt[i] != self.image_token_id {
                i += 1;
                continue;
            }
            if i == 0 || prompt[i - 1] != page_id {
                return Err(invalid_input(
                    "image_token_id",
                    "OCR-2 image runs require a preceding <PAGE>",
                ));
            }
            let start = i;
            while i < prompt.len() && prompt[i] == self.image_token_id {
                i += 1;
            }
            runs.push(i - start);
        }
        if runs.len() != self.images.len() {
            return Err(invalid_input(
                "images",
                "OCR-2 page/image run count mismatch",
            ));
        }
        let mut total = 0_u32;
        let mut rgb = 0_usize;
        for (image, run) in self.images.iter().zip(runs) {
            validate_image(image)?;
            rgb = rgb
                .checked_add(image.rgb_bytes.len())
                .ok_or_else(|| invalid_input("images", "RGB size overflow"))?;
            let (cols, rows) =
                unlimited_ocr_crop_grid_with_limit(image.width, image.height, self.cropping, 16);
            let soft = if (cols, rows) == (1, 1) {
                273
            } else {
                273 + rows * 10 * (cols * 10 + 1)
            };
            if run != soft as usize {
                return Err(invalid_input(
                    "image_token_id",
                    "OCR-2 page soft-token count mismatch",
                ));
            }
            total += soft;
        }
        if total != self.soft_token_count || rgb > UNLIMITED_OCR_MAX_RGB_BYTES {
            return Err(invalid_input(
                "images",
                "OCR-2 total soft-token/RGB payload limit mismatch",
            ));
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn validate_for_prompt_tokens(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<(), UnlimitedOcrRuntimeInputError> {
        if self.page_token_id.is_some() {
            return self.validate_pages(prompt_tokens);
        }
        if self.images.len() != 1 {
            return Err(invalid_input(
                "images",
                format!(
                    "native Unlimited-OCR requires exactly one source image, found {}",
                    self.images.len()
                ),
            ));
        }
        let expected_soft_tokens = unlimited_ocr_soft_token_count(
            self.images[0].width,
            self.images[0].height,
            self.cropping,
        );
        if self.soft_token_count != expected_soft_tokens {
            return Err(invalid_input(
                "soft_token_count",
                format!(
                    "Unlimited-OCR preprocessing requires {expected_soft_tokens}, found {}",
                    self.soft_token_count
                ),
            ));
        }

        let positions = prompt_tokens
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == self.image_token_id).then_some(index))
            .collect::<Vec<_>>();
        if positions.len() != self.soft_token_count as usize {
            return Err(invalid_input(
                "image_token_id",
                format!(
                    "prompt contains {} image tokens, expected {}",
                    positions.len(),
                    self.soft_token_count
                ),
            ));
        }
        if !positions.windows(2).all(|pair| pair[1] == pair[0] + 1) {
            return Err(invalid_input(
                "image_token_id",
                "expanded image tokens must form one contiguous run",
            ));
        }

        validate_image(&self.images[0])
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct UnlimitedOcrImageRuntimeInput {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB8 bytes with exactly `width * height * 3` entries.
    pub rgb_bytes: Vec<u8>,
}

fn validate_image(
    image: &UnlimitedOcrImageRuntimeInput,
) -> Result<(), UnlimitedOcrRuntimeInputError> {
    if image.width == 0 || image.height == 0 {
        return Err(invalid_input("images[0].dimensions", "must be non-zero"));
    }
    if image.width > UNLIMITED_OCR_MAX_IMAGE_DIMENSION
        || image.height > UNLIMITED_OCR_MAX_IMAGE_DIMENSION
    {
        return Err(invalid_input(
            "images[0].dimensions",
            format!("must not exceed {UNLIMITED_OCR_MAX_IMAGE_DIMENSION} pixels per side"),
        ));
    }
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| invalid_input("images[0].dimensions", "RGB byte length overflow"))?;
    if expected > UNLIMITED_OCR_MAX_RGB_BYTES {
        return Err(invalid_input(
            "images[0].rgb_bytes",
            format!("decoded RGB payload exceeds {UNLIMITED_OCR_MAX_RGB_BYTES} bytes"),
        ));
    }
    if image.rgb_bytes.len() != expected {
        return Err(invalid_input(
            "images[0].rgb_bytes",
            format!(
                "length {} does not match {}x{}x3={expected}",
                image.rgb_bytes.len(),
                image.width,
                image.height
            ),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum UnlimitedOcrRuntimeInputError {
    #[error("invalid Unlimited-OCR runtime input {field}: {message}")]
    InvalidField { field: String, message: String },
}

fn invalid_input(
    field: impl Into<String>,
    message: impl Into<String>,
) -> UnlimitedOcrRuntimeInputError {
    UnlimitedOcrRuntimeInputError::InvalidField {
        field: field.into(),
        message: message.into(),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn valid_inputs() -> (Vec<u32>, UnlimitedOcrRuntimeInputs) {
        let image_token_id = 128_815;
        let mut prompt = vec![0];
        prompt.extend(std::iter::repeat_n(
            image_token_id,
            UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT as usize,
        ));
        prompt.extend([21431, 126041, 16]);
        let inputs = UnlimitedOcrRuntimeInputs {
            page_token_id: None,
            image_token_id,
            soft_token_count: UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT,
            cropping: false,
            images: vec![UnlimitedOcrImageRuntimeInput {
                width: 2,
                height: 1,
                rgb_bytes: vec![0, 1, 2, 3, 4, 5],
            }],
        };
        (prompt, inputs)
    }

    #[test]
    fn accepts_one_bounded_image_and_contiguous_soft_tokens() {
        let (prompt, inputs) = valid_inputs();
        assert_eq!(inputs.validate_for_prompt_tokens(&prompt), Ok(()));
    }

    #[test]
    fn ocr2_requires_ordered_marked_image_runs_and_exact_counts() {
        let (_, mut inputs) = valid_inputs();
        inputs.page_token_id = Some(19);
        inputs.images.push(inputs.images[0].clone());
        inputs.soft_token_count = 546;
        let mut prompt = vec![0, 19];
        prompt.extend(vec![inputs.image_token_id; 273]);
        prompt.push(19);
        prompt.extend(vec![inputs.image_token_id; 273]);
        prompt.push(8);
        assert!(inputs.validate_for_prompt_tokens(&prompt).is_ok());
        let mut wrong_marker = prompt.clone();
        wrong_marker[275] = 8;
        assert!(inputs.validate_for_prompt_tokens(&wrong_marker).is_err());
        let mut wrong_count = prompt.clone();
        wrong_count.remove(3);
        assert!(inputs.validate_for_prompt_tokens(&wrong_count).is_err());
        inputs.page_token_id = None;
        assert!(inputs.validate_for_prompt_tokens(&prompt).is_err());
    }

    #[test]
    fn rejects_wrong_rgb_length() {
        let (prompt, mut inputs) = valid_inputs();
        inputs.images[0].rgb_bytes.pop();
        let error = inputs
            .validate_for_prompt_tokens(&prompt)
            .expect_err("short RGB input must fail");
        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn rejects_non_contiguous_image_tokens() {
        let (mut prompt, inputs) = valid_inputs();
        prompt[10] = 7;
        prompt.push(inputs.image_token_id);
        let error = inputs
            .validate_for_prompt_tokens(&prompt)
            .expect_err("split image token run must fail");
        assert!(error.to_string().contains("contiguous"));
    }

    #[test]
    fn released_portrait_page_uses_two_by_three_tiles() {
        assert_eq!(unlimited_ocr_crop_grid(911, 1287, true), (2, 3));
        assert_eq!(unlimited_ocr_soft_token_count(911, 1287, true), 903);
    }

    #[test]
    fn small_or_explicit_global_only_image_uses_base_tokens() {
        assert_eq!(unlimited_ocr_crop_grid(640, 640, true), (1, 1));
        assert_eq!(
            unlimited_ocr_soft_token_count(911, 1287, false),
            UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT
        );
    }
}
