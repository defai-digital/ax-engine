use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Default audio soft-token cap when `preprocessor_config.json` omits
/// `audio_seq_length`, matching the reference Gemma4UnifiedProcessor.
const DEFAULT_AUDIO_SEQ_LENGTH: u32 = 750;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Gemma4UnifiedModality {
    Image,
    Audio,
    Video,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4UnifiedSpecialTokens {
    pub image_token_id: u32,
    pub audio_token_id: u32,
    pub video_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4UnifiedVisionProcessor {
    pub patch_size: u32,
    pub model_patch_size: u32,
    pub pooling_kernel_size: u32,
    pub max_soft_tokens: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4UnifiedAudioProcessor {
    pub sampling_rate: u32,
    /// Raw waveform samples consumed per audio soft token. The encoder-free
    /// connector chunks the (zero-padded) waveform into frames of this size and
    /// projects one embedding per frame, so one frame == one soft token.
    pub audio_samples_per_token: u32,
    pub audio_seq_length: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4UnifiedProcessorConfig {
    pub tokens: Gemma4UnifiedSpecialTokens,
    pub vision: Gemma4UnifiedVisionProcessor,
    pub audio: Option<Gemma4UnifiedAudioProcessor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4UnifiedImageInput {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4UnifiedAudioInput {
    pub sample_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4UnifiedVideoInput {
    pub frame_count: u32,
    pub soft_tokens_per_frame: u32,
    /// Per-frame timestamp token IDs, already tokenized without special tokens.
    ///
    /// vLLM/HF format videos as:
    /// `mm:ss <boi><|video|>*N<eoi> ...`.
    /// AX core stays tokenizer-agnostic, so callers that need vLLM parity pass
    /// the timestamp token IDs here. An empty outer vector keeps the legacy
    /// tensor-only fallback with no timestamp tokens.
    pub timestamp_token_ids_per_frame: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Gemma4UnifiedTokenSpan {
    pub modality: Gemma4UnifiedModality,
    pub placeholder_index: usize,
    pub replacement_start: usize,
    pub soft_token_count: u32,
    pub replacement_token_count: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Gemma4UnifiedSoftTokenRange {
    pub start: usize,
    pub soft_token_count: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Gemma4UnifiedExpandedVideoPlaceholder {
    pub span: Gemma4UnifiedTokenSpan,
    pub soft_token_ranges: Vec<Gemma4UnifiedSoftTokenRange>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Gemma4UnifiedRuntimeInputs {
    pub images: Vec<Gemma4UnifiedImageRuntimeInput>,
    pub audios: Vec<Gemma4UnifiedAudioRuntimeInput>,
    pub videos: Vec<Gemma4UnifiedVideoRuntimeInput>,
}

impl Gemma4UnifiedRuntimeInputs {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty() && self.videos.is_empty()
    }

    pub fn validate_for_prompt_len(
        &self,
        prompt_token_len: usize,
    ) -> Result<(), Gemma4UnifiedRuntimeInputError> {
        for (idx, image) in self.images.iter().enumerate() {
            validate_contiguous_span(
                &format!("images[{idx}].span"),
                &image.span,
                Gemma4UnifiedModality::Image,
                prompt_token_len,
            )?;
            validate_vision_tensors(
                &format!("images[{idx}]"),
                &image.pixel_values,
                &image.pixel_position_ids,
                image.span.soft_token_count as usize,
            )?;
        }

        for (idx, audio) in self.audios.iter().enumerate() {
            validate_contiguous_span(
                &format!("audios[{idx}].span"),
                &audio.span,
                Gemma4UnifiedModality::Audio,
                prompt_token_len,
            )?;
            validate_audio_tensors(idx, audio)?;
        }

        for (idx, video) in self.videos.iter().enumerate() {
            validate_video_span(idx, video, prompt_token_len)?;
            validate_vision_tensors(
                &format!("videos[{idx}]"),
                &video.pixel_values,
                &video.pixel_position_ids,
                video.span.soft_token_count as usize,
            )?;
            validate_video_frame_shape(idx, video)?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Gemma4UnifiedImageRuntimeInput {
    pub span: Gemma4UnifiedTokenSpan,
    /// vLLM/HF processor output: `[patch_count, model_patch_size^2 * 3]`.
    pub pixel_values: Vec<f32>,
    /// vLLM/HF processor output: `[patch_count, 2]`, with `[-1, -1]` for padding.
    pub pixel_position_ids: Vec<[i32; 2]>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Gemma4UnifiedAudioRuntimeInput {
    pub span: Gemma4UnifiedTokenSpan,
    /// vLLM/HF processor output after audio feature extraction: `[frames, features]`.
    pub input_features: Vec<f32>,
    pub frame_count: u32,
    pub feature_count: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Gemma4UnifiedVideoRuntimeInput {
    pub span: Gemma4UnifiedTokenSpan,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub soft_token_ranges: Vec<Gemma4UnifiedSoftTokenRange>,
    /// Flattened per-frame image processor output, same patch shape as images.
    pub pixel_values: Vec<f32>,
    pub pixel_position_ids: Vec<[i32; 2]>,
    pub frame_count: u32,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum Gemma4UnifiedError {
    #[error("missing Gemma4 unified config field {0}")]
    MissingField(&'static str),
    #[error("invalid Gemma4 unified config field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("expected {expected} {modality:?} placeholder tokens, found {actual}")]
    PlaceholderCountMismatch {
        modality: Gemma4UnifiedModality,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum Gemma4UnifiedRuntimeInputError {
    #[error("invalid Gemma4 unified runtime input {field}: {message}")]
    InvalidField { field: String, message: String },
}

fn invalid_runtime_input(
    field: impl Into<String>,
    message: impl Into<String>,
) -> Gemma4UnifiedRuntimeInputError {
    Gemma4UnifiedRuntimeInputError::InvalidField {
        field: field.into(),
        message: message.into(),
    }
}

fn validate_span_bounds(
    field: &str,
    span: &Gemma4UnifiedTokenSpan,
    expected_modality: Gemma4UnifiedModality,
    prompt_token_len: usize,
) -> Result<usize, Gemma4UnifiedRuntimeInputError> {
    if span.modality != expected_modality {
        return Err(invalid_runtime_input(
            format!("{field}.modality"),
            format!("expected {expected_modality:?}, found {:?}", span.modality),
        ));
    }
    if span.soft_token_count == 0 {
        return Err(invalid_runtime_input(
            format!("{field}.soft_token_count"),
            "must be greater than zero",
        ));
    }
    if span.replacement_token_count == 0 {
        return Err(invalid_runtime_input(
            format!("{field}.replacement_token_count"),
            "must be greater than zero",
        ));
    }
    let replacement_end = span
        .replacement_start
        .checked_add(span.replacement_token_count as usize)
        .ok_or_else(|| {
            invalid_runtime_input(format!("{field}.replacement_start"), "span end overflow")
        })?;
    if replacement_end > prompt_token_len {
        return Err(invalid_runtime_input(
            field,
            format!(
                "replacement span [{}..{}) exceeds prompt length {prompt_token_len}",
                span.replacement_start, replacement_end
            ),
        ));
    }
    Ok(replacement_end)
}

fn validate_contiguous_span(
    field: &str,
    span: &Gemma4UnifiedTokenSpan,
    expected_modality: Gemma4UnifiedModality,
    prompt_token_len: usize,
) -> Result<(), Gemma4UnifiedRuntimeInputError> {
    validate_span_bounds(field, span, expected_modality, prompt_token_len)?;
    let expected_replacement = span.soft_token_count.checked_add(2).ok_or_else(|| {
        invalid_runtime_input(format!("{field}.soft_token_count"), "span length overflow")
    })?;
    if span.replacement_token_count != expected_replacement {
        return Err(invalid_runtime_input(
            format!("{field}.replacement_token_count"),
            format!(
                "expected soft_token_count + 2 boundary tokens ({expected_replacement}), found {}",
                span.replacement_token_count
            ),
        ));
    }
    Ok(())
}

fn validate_vision_tensors(
    field: &str,
    pixel_values: &[f32],
    pixel_position_ids: &[[i32; 2]],
    expected_soft_tokens: usize,
) -> Result<(), Gemma4UnifiedRuntimeInputError> {
    let patch_count = pixel_position_ids.len();
    if patch_count == 0 {
        return Err(invalid_runtime_input(
            format!("{field}.pixel_position_ids"),
            "must contain at least one patch position",
        ));
    }
    let Some(patch_dim) = pixel_values
        .len()
        .checked_div(patch_count)
        .filter(|dim| *dim > 0 && *dim * patch_count == pixel_values.len())
    else {
        return Err(invalid_runtime_input(
            format!("{field}.pixel_values"),
            format!(
                "length {} must divide evenly by patch count {patch_count}",
                pixel_values.len()
            ),
        ));
    };
    if patch_dim == 0 {
        return Err(invalid_runtime_input(
            format!("{field}.pixel_values"),
            "patch dimension must be greater than zero",
        ));
    }
    let mut valid_patch_count = 0usize;
    let mut saw_padding = false;
    for (idx, [x, y]) in pixel_position_ids.iter().enumerate() {
        let is_valid = *x >= 0 && *y >= 0;
        let is_padding = *x == -1 && *y == -1;
        if is_valid {
            if saw_padding {
                return Err(invalid_runtime_input(
                    format!("{field}.pixel_position_ids[{idx}]"),
                    "valid patch positions must be a prefix before [-1, -1] padding rows",
                ));
            }
            valid_patch_count += 1;
        } else if is_padding {
            saw_padding = true;
        } else {
            return Err(invalid_runtime_input(
                format!("{field}.pixel_position_ids[{idx}]"),
                "must be a non-negative patch position or [-1, -1] padding",
            ));
        }
    }
    if valid_patch_count != expected_soft_tokens {
        return Err(invalid_runtime_input(
            format!("{field}.pixel_position_ids"),
            format!(
                "contains {valid_patch_count} valid patch positions, but span expects {expected_soft_tokens} soft tokens"
            ),
        ));
    }
    Ok(())
}

fn validate_audio_tensors(
    idx: usize,
    audio: &Gemma4UnifiedAudioRuntimeInput,
) -> Result<(), Gemma4UnifiedRuntimeInputError> {
    let frame_count = audio.frame_count as usize;
    let feature_count = audio.feature_count as usize;
    if frame_count == 0 || feature_count == 0 {
        return Err(invalid_runtime_input(
            format!("audios[{idx}]"),
            "frame_count and feature_count must be greater than zero",
        ));
    }
    if audio.input_features.len() != frame_count * feature_count {
        return Err(invalid_runtime_input(
            format!("audios[{idx}].input_features"),
            format!(
                "length {} must equal frame_count * feature_count ({})",
                audio.input_features.len(),
                frame_count * feature_count
            ),
        ));
    }
    if audio.span.soft_token_count as usize != frame_count {
        return Err(invalid_runtime_input(
            format!("audios[{idx}].frame_count"),
            format!(
                "must match span.soft_token_count {}; found {frame_count}",
                audio.span.soft_token_count
            ),
        ));
    }
    Ok(())
}

fn validate_video_span(
    idx: usize,
    video: &Gemma4UnifiedVideoRuntimeInput,
    prompt_token_len: usize,
) -> Result<(), Gemma4UnifiedRuntimeInputError> {
    let field = format!("videos[{idx}].span");
    let replacement_end = validate_span_bounds(
        &field,
        &video.span,
        Gemma4UnifiedModality::Video,
        prompt_token_len,
    )?;
    if video.frame_count == 0 {
        return Err(invalid_runtime_input(
            format!("videos[{idx}].frame_count"),
            "must be greater than zero",
        ));
    }
    if video.soft_token_ranges.is_empty() {
        validate_contiguous_span(
            &field,
            &video.span,
            Gemma4UnifiedModality::Video,
            prompt_token_len,
        )?;
        return Ok(());
    }
    if video.soft_token_ranges.len() != video.frame_count as usize {
        return Err(invalid_runtime_input(
            format!("videos[{idx}].soft_token_ranges"),
            format!(
                "expected one range per frame ({}), found {}",
                video.frame_count,
                video.soft_token_ranges.len()
            ),
        ));
    }
    let mut previous_end = video.span.replacement_start;
    let mut summed_soft_tokens = 0usize;
    for (range_idx, range) in video.soft_token_ranges.iter().enumerate() {
        if range.soft_token_count == 0 {
            return Err(invalid_runtime_input(
                format!("videos[{idx}].soft_token_ranges[{range_idx}].soft_token_count"),
                "must be greater than zero",
            ));
        }
        if range.start < video.span.replacement_start || range.start >= replacement_end {
            return Err(invalid_runtime_input(
                format!("videos[{idx}].soft_token_ranges[{range_idx}].start"),
                format!(
                    "must be inside replacement span [{}..{})",
                    video.span.replacement_start, replacement_end
                ),
            ));
        }
        let range_end = range
            .start
            .checked_add(range.soft_token_count as usize)
            .ok_or_else(|| {
                invalid_runtime_input(
                    format!("videos[{idx}].soft_token_ranges[{range_idx}]"),
                    "range end overflow",
                )
            })?;
        if range_end > replacement_end {
            return Err(invalid_runtime_input(
                format!("videos[{idx}].soft_token_ranges[{range_idx}]"),
                format!("range end {range_end} exceeds replacement end {replacement_end}"),
            ));
        }
        if range.start < previous_end {
            return Err(invalid_runtime_input(
                format!("videos[{idx}].soft_token_ranges[{range_idx}].start"),
                "ranges must be non-overlapping and sorted",
            ));
        }
        previous_end = range_end;
        summed_soft_tokens += range.soft_token_count as usize;
    }
    if summed_soft_tokens != video.span.soft_token_count as usize {
        return Err(invalid_runtime_input(
            format!("videos[{idx}].soft_token_ranges"),
            format!(
                "sum to {summed_soft_tokens} soft tokens, but span expects {}",
                video.span.soft_token_count
            ),
        ));
    }
    Ok(())
}

fn validate_video_frame_shape(
    idx: usize,
    video: &Gemma4UnifiedVideoRuntimeInput,
) -> Result<(), Gemma4UnifiedRuntimeInputError> {
    let frame_count = video.frame_count as usize;
    if !video.pixel_position_ids.len().is_multiple_of(frame_count) {
        return Err(invalid_runtime_input(
            format!("videos[{idx}].pixel_position_ids"),
            format!(
                "patch count {} must divide evenly by frame_count {frame_count}",
                video.pixel_position_ids.len()
            ),
        ));
    }
    Ok(())
}

impl Gemma4UnifiedProcessorConfig {
    pub fn from_model_and_processor_config(
        model_config: &Value,
        processor_config: &Value,
    ) -> Result<Self, Gemma4UnifiedError> {
        let vision_config = model_config
            .get("vision_config")
            .ok_or(Gemma4UnifiedError::MissingField("vision_config"))?;
        let image_processor = processor_config.get("image_processor");
        let feature_extractor = processor_config
            .get("feature_extractor")
            .or_else(|| processor_config.get("audio_feature_extractor"));

        Ok(Self {
            tokens: Gemma4UnifiedSpecialTokens {
                image_token_id: required_u32(model_config, "image_token_id")?,
                audio_token_id: required_u32(model_config, "audio_token_id")?,
                video_token_id: optional_u32(model_config, "video_token_id").unwrap_or(0),
                boi_token_id: required_u32(model_config, "boi_token_id")?,
                eoi_token_id: required_u32(model_config, "eoi_token_id")?,
                boa_token_id: required_u32(model_config, "boa_token_id")?,
                eoa_token_id: required_u32(model_config, "eoa_token_index")
                    .or_else(|_| required_u32(model_config, "eoa_token_id"))?,
            },
            vision: Gemma4UnifiedVisionProcessor {
                patch_size: optional_nested_u32(image_processor, "patch_size")
                    .or_else(|| optional_u32(vision_config, "patch_size"))
                    .ok_or(Gemma4UnifiedError::MissingField("vision.patch_size"))?,
                model_patch_size: optional_nested_u32(image_processor, "model_patch_size")
                    .or_else(|| optional_u32(vision_config, "model_patch_size"))
                    .ok_or(Gemma4UnifiedError::MissingField("vision.model_patch_size"))?,
                pooling_kernel_size: optional_nested_u32(image_processor, "pooling_kernel_size")
                    .or_else(|| optional_u32(vision_config, "pooling_kernel_size"))
                    .ok_or(Gemma4UnifiedError::MissingField(
                        "vision.pooling_kernel_size",
                    ))?,
                max_soft_tokens: optional_nested_u32(image_processor, "max_soft_tokens")
                    .or_else(|| optional_u32(vision_config, "num_soft_tokens"))
                    .or_else(|| optional_u32(vision_config, "default_output_length"))
                    .ok_or(Gemma4UnifiedError::MissingField("vision.max_soft_tokens"))?,
            },
            audio: feature_extractor.map(|feature_extractor| {
                let audio_config = model_config.get("audio_config");
                Gemma4UnifiedAudioProcessor {
                    sampling_rate: optional_u32(feature_extractor, "sampling_rate")
                        .unwrap_or(16000),
                    audio_samples_per_token: optional_nested_u32(
                        audio_config,
                        "audio_samples_per_token",
                    )
                    .or_else(|| optional_u32(feature_extractor, "audio_samples_per_token"))
                    .or_else(|| optional_u32(feature_extractor, "feature_size"))
                    .or_else(|| optional_nested_u32(audio_config, "audio_embed_dim"))
                    .unwrap_or(640),
                    // The reference Gemma4UnifiedProcessor caps audio at
                    // `audio_seq_length` and defaults it to 750 when the config
                    // omits it (processing_gemma4_unified.py / processing_gemma4.py
                    // `_compute_audio_num_tokens` -> min(..., audio_seq_length)).
                    // Mirror that default so long audio doesn't expand to an
                    // unbounded soft-token count that diverges from the reference.
                    audio_seq_length: Some(
                        optional_u32(processor_config, "audio_seq_length")
                            .unwrap_or(DEFAULT_AUDIO_SEQ_LENGTH),
                    ),
                }
            }),
        })
    }

    pub fn image_soft_tokens(
        &self,
        image: Gemma4UnifiedImageInput,
    ) -> Result<u32, Gemma4UnifiedError> {
        self.vision.compute_soft_tokens(image.width, image.height)
    }

    pub fn image_replacement_tokens(
        &self,
        image: Gemma4UnifiedImageInput,
    ) -> Result<Vec<u32>, Gemma4UnifiedError> {
        let soft_tokens = self.image_soft_tokens(image)?;
        let mut tokens = Vec::with_capacity(soft_tokens as usize + 2);
        tokens.push(self.tokens.boi_token_id);
        tokens.extend(std::iter::repeat_n(
            self.tokens.image_token_id,
            soft_tokens as usize,
        ));
        tokens.push(self.tokens.eoi_token_id);
        Ok(tokens)
    }

    pub fn audio_soft_tokens(
        &self,
        audio: Gemma4UnifiedAudioInput,
    ) -> Result<u32, Gemma4UnifiedError> {
        let processor = self
            .audio
            .as_ref()
            .ok_or(Gemma4UnifiedError::MissingField("feature_extractor"))?;
        Ok(processor.compute_soft_tokens(audio.sample_count))
    }

    pub fn audio_replacement_tokens(
        &self,
        audio: Gemma4UnifiedAudioInput,
    ) -> Result<Vec<u32>, Gemma4UnifiedError> {
        let soft_tokens = self.audio_soft_tokens(audio)?;
        let mut tokens = Vec::with_capacity(soft_tokens as usize + 2);
        tokens.push(self.tokens.boa_token_id);
        tokens.extend(std::iter::repeat_n(
            self.tokens.audio_token_id,
            soft_tokens as usize,
        ));
        tokens.push(self.tokens.eoa_token_id);
        Ok(tokens)
    }

    pub fn video_replacement_tokens(
        &self,
        video: Gemma4UnifiedVideoInput,
    ) -> Result<Vec<u32>, Gemma4UnifiedError> {
        self.video_replacement_tokens_with_ranges(video)
            .map(|(tokens, _)| tokens)
    }

    pub fn video_replacement_tokens_with_ranges(
        &self,
        video: Gemma4UnifiedVideoInput,
    ) -> Result<(Vec<u32>, Vec<Gemma4UnifiedSoftTokenRange>), Gemma4UnifiedError> {
        if video.frame_count == 0 || video.soft_tokens_per_frame == 0 {
            return Err(Gemma4UnifiedError::InvalidField {
                field: "video",
                message: "frame_count and soft_tokens_per_frame must be greater than zero"
                    .to_string(),
            });
        }
        if !video.timestamp_token_ids_per_frame.is_empty()
            && video.timestamp_token_ids_per_frame.len() != video.frame_count as usize
        {
            return Err(Gemma4UnifiedError::InvalidField {
                field: "video.timestamp_token_ids_per_frame",
                message: format!(
                    "expected {} timestamp entries, found {}",
                    video.frame_count,
                    video.timestamp_token_ids_per_frame.len()
                ),
            });
        }

        let timestamp_token_ids_per_frame = if video.timestamp_token_ids_per_frame.is_empty() {
            vec![Vec::new(); video.frame_count as usize]
        } else {
            video.timestamp_token_ids_per_frame
        };
        let timestamp_tokens = timestamp_token_ids_per_frame
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        let soft_tokens = video
            .frame_count
            .checked_mul(video.soft_tokens_per_frame)
            .ok_or_else(|| Gemma4UnifiedError::InvalidField {
                field: "video",
                message: "soft token count overflow".to_string(),
            })?;
        let mut tokens = Vec::with_capacity(
            timestamp_tokens
                .saturating_add(soft_tokens as usize)
                .saturating_add(video.frame_count as usize * 2),
        );
        let mut soft_token_ranges = Vec::with_capacity(video.frame_count as usize);

        for timestamp_token_ids in timestamp_token_ids_per_frame {
            tokens.extend_from_slice(&timestamp_token_ids);
            tokens.push(self.tokens.boi_token_id);
            let start = tokens.len();
            tokens.extend(std::iter::repeat_n(
                self.tokens.video_token_id,
                video.soft_tokens_per_frame as usize,
            ));
            soft_token_ranges.push(Gemma4UnifiedSoftTokenRange {
                start,
                soft_token_count: video.soft_tokens_per_frame,
            });
            tokens.push(self.tokens.eoi_token_id);
        }
        Ok((tokens, soft_token_ranges))
    }

    pub fn expand_image_placeholders(
        &self,
        input_tokens: &[u32],
        images: &[Gemma4UnifiedImageInput],
    ) -> Result<(Vec<u32>, Vec<Gemma4UnifiedTokenSpan>), Gemma4UnifiedError> {
        expand_placeholders(
            Gemma4UnifiedModality::Image,
            input_tokens,
            self.tokens.image_token_id,
            images,
            |image| {
                self.image_replacement_tokens(*image)
                    .map(|tokens| (tokens, self.image_soft_tokens(*image).unwrap_or(0)))
            },
        )
    }

    pub fn expand_audio_placeholders(
        &self,
        input_tokens: &[u32],
        audios: &[Gemma4UnifiedAudioInput],
    ) -> Result<(Vec<u32>, Vec<Gemma4UnifiedTokenSpan>), Gemma4UnifiedError> {
        expand_placeholders(
            Gemma4UnifiedModality::Audio,
            input_tokens,
            self.tokens.audio_token_id,
            audios,
            |audio| {
                let soft_tokens = self.audio_soft_tokens(audio.clone()).unwrap_or(0);
                self.audio_replacement_tokens(audio.clone())
                    .map(|tokens| (tokens, soft_tokens))
            },
        )
    }

    pub fn expand_video_placeholders(
        &self,
        input_tokens: &[u32],
        videos: &[Gemma4UnifiedVideoInput],
    ) -> Result<(Vec<u32>, Vec<Gemma4UnifiedExpandedVideoPlaceholder>), Gemma4UnifiedError> {
        let actual = input_tokens
            .iter()
            .filter(|&&token| token == self.tokens.video_token_id)
            .count();
        if actual != videos.len() {
            return Err(Gemma4UnifiedError::PlaceholderCountMismatch {
                modality: Gemma4UnifiedModality::Video,
                expected: videos.len(),
                actual,
            });
        }

        let mut video_index = 0usize;
        let mut output = Vec::new();
        let mut spans = Vec::new();
        for (idx, token) in input_tokens.iter().copied().enumerate() {
            if token == self.tokens.video_token_id {
                let replacement_start = output.len();
                let (replacement_tokens, relative_ranges) =
                    self.video_replacement_tokens_with_ranges(videos[video_index].clone())?;
                output.extend_from_slice(&replacement_tokens);
                let soft_token_count = relative_ranges.iter().fold(0_u32, |acc, range| {
                    acc.saturating_add(range.soft_token_count)
                });
                let soft_token_ranges = relative_ranges
                    .into_iter()
                    .map(|range| Gemma4UnifiedSoftTokenRange {
                        start: replacement_start + range.start,
                        soft_token_count: range.soft_token_count,
                    })
                    .collect();
                spans.push(Gemma4UnifiedExpandedVideoPlaceholder {
                    span: Gemma4UnifiedTokenSpan {
                        modality: Gemma4UnifiedModality::Video,
                        placeholder_index: idx,
                        replacement_start,
                        soft_token_count,
                        replacement_token_count: replacement_tokens.len() as u32,
                    },
                    soft_token_ranges,
                });
                video_index += 1;
            } else {
                output.push(token);
            }
        }
        Ok((output, spans))
    }
}

impl Gemma4UnifiedVisionProcessor {
    pub fn compute_soft_tokens(
        &self,
        image_width: u32,
        image_height: u32,
    ) -> Result<u32, Gemma4UnifiedError> {
        if image_width == 0 || image_height == 0 {
            return Err(Gemma4UnifiedError::InvalidField {
                field: "image_size",
                message: "width and height must be greater than zero".to_string(),
            });
        }
        if self.patch_size == 0 || self.pooling_kernel_size == 0 || self.max_soft_tokens == 0 {
            return Err(Gemma4UnifiedError::InvalidField {
                field: "vision",
                message:
                    "patch_size, pooling_kernel_size, and max_soft_tokens must be greater than zero"
                        .to_string(),
            });
        }

        let (target_w, target_h) = self.resize_target(image_width, image_height);
        let num_patches = (target_h / self.patch_size) * (target_w / self.patch_size);
        Ok((num_patches / self.pooling_kernel_size.pow(2)).min(self.max_soft_tokens))
    }

    /// Aspect-ratio-preserving resize target `(width, height)`, mirroring the
    /// reference `aspect_ratio_preserving_resize`: the largest dimensions that
    /// produce at most `max_soft_tokens` model patches and are divisible by
    /// `patch_size * pooling_kernel_size`. Includes the reference's single-axis
    /// fallback and `max_side_length` clamp so extreme aspect ratios resize the
    /// same way the model expects (a plain per-axis `max(unit, …)` would crop a
    /// wide/tall image to one half instead of downscaling it whole).
    ///
    /// Callers that patchify pixels (e.g. the server's media preprocessing) must
    /// resize to exactly these dimensions so their patch grid matches
    /// [`Self::compute_soft_tokens`].
    pub fn resize_target(&self, width: u32, height: u32) -> (u32, u32) {
        let patch = self.patch_size as f64;
        let pooling = self.pooling_kernel_size as f64;
        let unit = patch * pooling;
        if width == 0 || height == 0 || unit == 0.0 {
            return (unit as u32, unit as u32);
        }
        let max_patches = self.max_soft_tokens as f64 * pooling * pooling;
        let target_px = max_patches * patch * patch;
        let factor = (target_px / (height as f64 * width as f64)).sqrt();
        let mut target_h = (factor * height as f64 / unit).floor() * unit;
        let mut target_w = (factor * width as f64 / unit).floor() * unit;
        // Reference: max_side_length = (max_patches / pooling^2) * side_mult.
        let max_side = self.max_soft_tokens as f64 * unit;
        if target_h == 0.0 && target_w == 0.0 {
            // Reference raises here; clamp to a single patch instead of crashing
            // (width/height are already validated as non-zero by the caller).
            target_h = unit;
            target_w = unit;
        } else if target_h == 0.0 {
            target_h = unit;
            target_w = ((width as f64 / height as f64).floor() * unit).min(max_side);
        } else if target_w == 0.0 {
            target_w = unit;
            target_h = ((height as f64 / width as f64).floor() * unit).min(max_side);
        }
        (target_w as u32, target_h as u32)
    }
}

impl Gemma4UnifiedAudioProcessor {
    pub fn compute_soft_tokens(&self, sample_count: u32) -> u32 {
        if sample_count == 0 || self.audio_samples_per_token == 0 {
            return 0;
        }
        // Encoder-free connector: the waveform is zero-padded up to a multiple of
        // `audio_samples_per_token` and reshaped into that many fixed-size frames;
        // each frame is projected into exactly one soft token. This mirrors the
        // reference processor (waveform.reshape(-1, audio_samples_per_token)) and
        // the AX Python/MLX runtime, which truncate to `audio_seq_length`.
        let frames = sample_count.div_ceil(self.audio_samples_per_token);
        self.audio_seq_length
            .map(|limit| frames.min(limit))
            .unwrap_or(frames)
    }
}

fn expand_placeholders<T>(
    modality: Gemma4UnifiedModality,
    input_tokens: &[u32],
    placeholder_token: u32,
    items: &[T],
    replacement: impl Fn(&T) -> Result<(Vec<u32>, u32), Gemma4UnifiedError>,
) -> Result<(Vec<u32>, Vec<Gemma4UnifiedTokenSpan>), Gemma4UnifiedError> {
    let actual = input_tokens
        .iter()
        .filter(|&&token| token == placeholder_token)
        .count();
    if actual != items.len() {
        return Err(Gemma4UnifiedError::PlaceholderCountMismatch {
            modality,
            expected: items.len(),
            actual,
        });
    }

    let mut item_index = 0usize;
    let mut output = Vec::new();
    let mut spans = Vec::new();
    for (idx, token) in input_tokens.iter().copied().enumerate() {
        if token == placeholder_token {
            let (replacement_tokens, soft_token_count) = replacement(&items[item_index])?;
            let replacement_start = output.len();
            output.extend_from_slice(&replacement_tokens);
            spans.push(Gemma4UnifiedTokenSpan {
                modality,
                placeholder_index: idx,
                replacement_start,
                soft_token_count,
                replacement_token_count: replacement_tokens.len() as u32,
            });
            item_index += 1;
        } else {
            output.push(token);
        }
    }
    Ok((output, spans))
}

fn required_u32(value: &Value, field: &'static str) -> Result<u32, Gemma4UnifiedError> {
    optional_u32(value, field).ok_or(Gemma4UnifiedError::MissingField(field))
}

fn optional_nested_u32(value: Option<&Value>, field: &'static str) -> Option<u32> {
    value.and_then(|value| optional_u32(value, field))
}

fn optional_u32(value: &Value, field: &'static str) -> Option<u32> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn local_like_config() -> Gemma4UnifiedProcessorConfig {
        let model = json!({
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "video_token_id": 258884,
            "boi_token_id": 255999,
            "eoi_token_id": 258882,
            "boa_token_id": 256000,
            "eoa_token_index": 258883,
            "vision_config": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "num_soft_tokens": 280
            }
        });
        let processor = json!({
            "image_processor": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "max_soft_tokens": 280
            },
            "feature_extractor": {
                "feature_extractor_type": "Gemma4UnifiedAudioFeatureExtractor",
                "sampling_rate": 16000
            },
            "audio_seq_length": 1500
        });
        Gemma4UnifiedProcessorConfig::from_model_and_processor_config(&model, &processor)
            .expect("config should parse")
    }

    #[test]
    fn computes_vllm_matching_image_soft_tokens() {
        let cfg = local_like_config();

        assert_eq!(
            cfg.image_soft_tokens(Gemma4UnifiedImageInput {
                width: 224,
                height: 224
            })
            .unwrap(),
            256
        );
        assert_eq!(
            cfg.image_soft_tokens(Gemma4UnifiedImageInput {
                width: 1024,
                height: 256
            })
            .unwrap(),
            264
        );
        assert_eq!(
            cfg.image_soft_tokens(Gemma4UnifiedImageInput {
                width: 3,
                height: 900
            })
            .unwrap(),
            280
        );
    }

    #[test]
    fn resize_target_handles_extreme_aspect_ratios_like_reference() {
        // patch=4, pooling=2 -> unit=8, max_soft_tokens=4, max_side=32.
        let vision = Gemma4UnifiedVisionProcessor {
            patch_size: 4,
            model_patch_size: 8,
            pooling_kernel_size: 2,
            max_soft_tokens: 4,
        };
        // Very wide image: reference downscales the whole width (32x8), it must
        // NOT collapse height to 0 or crop to a half-width grid.
        assert_eq!(vision.resize_target(100, 5), (32, 8));
        // Very tall image is the symmetric case.
        assert_eq!(vision.resize_target(5, 100), (8, 32));
        // Degenerate aspect ratios stay non-zero: the single-axis fallback
        // branch only runs when the floored axis is the *smaller* one, so the
        // width/height ratio it floors is always >= 1 (regression pin for the
        // resize-extreme-aspect-zero report, which assumed the opposite).
        assert_eq!(vision.resize_target(1, 10_000), (8, 32));
        assert_eq!(vision.resize_target(10_000, 1), (32, 8));
        // Both axes stay non-zero and divisible by unit, and the soft-token count
        // stays within budget.
        let (w, h) = vision.resize_target(100, 5);
        assert_eq!(w % 8, 0);
        assert_eq!(h % 8, 0);
        assert!(vision.compute_soft_tokens(100, 5).unwrap() <= 4);
    }

    #[test]
    fn expands_image_placeholder_to_boundary_and_soft_tokens() {
        let cfg = local_like_config();
        let (tokens, spans) = cfg
            .expand_image_placeholders(
                &[10, 258880, 11],
                &[Gemma4UnifiedImageInput {
                    width: 224,
                    height: 224,
                }],
            )
            .expect("placeholder should expand");

        assert_eq!(tokens[0], 10);
        assert_eq!(tokens[1], 255999);
        assert_eq!(tokens[2], 258880);
        assert_eq!(tokens[tokens.len() - 2], 258882);
        assert_eq!(tokens[tokens.len() - 1], 11);
        assert_eq!(tokens.len(), 260);
        assert_eq!(
            spans,
            vec![Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Image,
                placeholder_index: 1,
                replacement_start: 1,
                soft_token_count: 256,
                replacement_token_count: 258,
            }]
        );
    }

    #[test]
    fn rejects_placeholder_count_mismatch() {
        let cfg = local_like_config();
        let error = cfg
            .expand_image_placeholders(
                &[10, 11],
                &[Gemma4UnifiedImageInput {
                    width: 1,
                    height: 1,
                }],
            )
            .expect_err("missing placeholder should fail");

        assert_eq!(
            error,
            Gemma4UnifiedError::PlaceholderCountMismatch {
                modality: Gemma4UnifiedModality::Image,
                expected: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn computes_audio_soft_tokens_from_sample_count() {
        let cfg = local_like_config();

        // Encoder-free chunking: one soft token per `audio_samples_per_token`
        // (640) waveform samples, padding the final partial frame. 16000 / 640
        // divides evenly into 25 frames.
        assert_eq!(
            cfg.audio_soft_tokens(Gemma4UnifiedAudioInput {
                sample_count: 16000
            })
            .unwrap(),
            25
        );

        // A non-multiple sample count rounds up (ceil), matching the reference
        // processor's zero-padding of the trailing frame.
        assert_eq!(
            cfg.audio_soft_tokens(Gemma4UnifiedAudioInput {
                sample_count: 16001
            })
            .unwrap(),
            26
        );
        assert_eq!(
            cfg.audio_soft_tokens(Gemma4UnifiedAudioInput { sample_count: 1 })
                .unwrap(),
            1
        );
    }

    #[test]
    fn audio_soft_tokens_respect_seq_length_cap_and_config_override() {
        // audio_seq_length caps the frame count; audio_samples_per_token is read
        // from the model audio_config when present.
        let model = json!({
            "image_token_id": 256,
            "audio_token_id": 257,
            "video_token_id": 258,
            "boi_token_id": 255,
            "eoi_token_id": 262,
            "boa_token_id": 259,
            "eoa_token_id": 260,
            "vision_config": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "num_soft_tokens": 280
            },
            "audio_config": { "audio_samples_per_token": 320 }
        });
        let processor = json!({
            "feature_extractor": { "sampling_rate": 16000 },
            "audio_seq_length": 4
        });
        let cfg = Gemma4UnifiedProcessorConfig::from_model_and_processor_config(&model, &processor)
            .expect("config should parse");

        // 1600 / 320 = 5 frames, capped to audio_seq_length = 4.
        assert_eq!(
            cfg.audio_soft_tokens(Gemma4UnifiedAudioInput { sample_count: 1600 })
                .unwrap(),
            4
        );
        // 960 / 320 = 3 frames, under the cap.
        assert_eq!(
            cfg.audio_soft_tokens(Gemma4UnifiedAudioInput { sample_count: 960 })
                .unwrap(),
            3
        );
    }

    #[test]
    fn audio_seq_length_defaults_to_reference_cap_when_config_omits_it() {
        // The reference caps audio at 750 soft tokens by default; a config that
        // omits `audio_seq_length` must not produce an unbounded count.
        let model = json!({
            "image_token_id": 256,
            "audio_token_id": 257,
            "video_token_id": 258,
            "boi_token_id": 255,
            "eoi_token_id": 262,
            "boa_token_id": 259,
            "eoa_token_id": 260,
            "vision_config": {
                "patch_size": 16,
                "model_patch_size": 48,
                "pooling_kernel_size": 3,
                "num_soft_tokens": 280
            }
        });
        let processor = json!({
            "feature_extractor": { "sampling_rate": 16000 }
        });
        let cfg = Gemma4UnifiedProcessorConfig::from_model_and_processor_config(&model, &processor)
            .expect("config should parse");
        assert_eq!(cfg.audio.as_ref().unwrap().audio_seq_length, Some(750));
        // 800 frames worth of samples (800 * 640) caps to the default 750.
        assert_eq!(
            cfg.audio_soft_tokens(Gemma4UnifiedAudioInput {
                sample_count: 800 * 640
            })
            .unwrap(),
            750
        );
    }

    #[test]
    fn expands_audio_placeholder_to_boundary_and_soft_tokens() {
        let cfg = local_like_config();
        let (tokens, spans) = cfg
            .expand_audio_placeholders(
                &[10, 258881, 11],
                &[Gemma4UnifiedAudioInput {
                    sample_count: 16000,
                }],
            )
            .expect("placeholder should expand");

        assert_eq!(tokens[0], 10);
        assert_eq!(tokens[1], 256000);
        assert_eq!(tokens[2], 258881);
        assert_eq!(tokens[tokens.len() - 2], 258883);
        assert_eq!(tokens[tokens.len() - 1], 11);
        assert_eq!(tokens.len(), 29);
        assert_eq!(
            spans,
            vec![Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Audio,
                placeholder_index: 1,
                replacement_start: 1,
                soft_token_count: 25,
                replacement_token_count: 27,
            }]
        );
    }

    #[test]
    fn rejects_audio_placeholder_count_mismatch_with_audio_modality() {
        let cfg = local_like_config();
        let error = cfg
            .expand_audio_placeholders(
                &[10, 11],
                &[Gemma4UnifiedAudioInput {
                    sample_count: 16000,
                }],
            )
            .expect_err("missing placeholder should fail");

        assert_eq!(
            error,
            Gemma4UnifiedError::PlaceholderCountMismatch {
                modality: Gemma4UnifiedModality::Audio,
                expected: 1,
                actual: 0
            }
        );
    }

    #[test]
    fn builds_video_replacement_tokens() {
        let cfg = local_like_config();
        let (tokens, ranges) = cfg
            .video_replacement_tokens_with_ranges(Gemma4UnifiedVideoInput {
                frame_count: 2,
                soft_tokens_per_frame: 70,
                timestamp_token_ids_per_frame: Vec::new(),
            })
            .expect("video replacement should build");

        assert_eq!(tokens.len(), 144);
        assert_eq!(tokens[0], 255999);
        assert_eq!(tokens[1], 258884);
        assert_eq!(tokens[70], 258884);
        assert_eq!(tokens[71], 258882);
        assert_eq!(tokens[72], 255999);
        assert_eq!(tokens[tokens.len() - 1], 258882);
        assert_eq!(
            ranges,
            vec![
                Gemma4UnifiedSoftTokenRange {
                    start: 1,
                    soft_token_count: 70,
                },
                Gemma4UnifiedSoftTokenRange {
                    start: 73,
                    soft_token_count: 70,
                },
            ]
        );
    }

    #[test]
    fn expands_video_placeholder_with_timestamped_frame_ranges() {
        let cfg = local_like_config();
        let (tokens, videos) = cfg
            .expand_video_placeholders(
                &[10, 258884, 11],
                &[Gemma4UnifiedVideoInput {
                    frame_count: 2,
                    soft_tokens_per_frame: 3,
                    timestamp_token_ids_per_frame: vec![vec![901, 902], vec![903]],
                }],
            )
            .expect("placeholder should expand");

        assert_eq!(
            tokens,
            vec![
                10, 901, 902, 255999, 258884, 258884, 258884, 258882, 903, 255999, 258884, 258884,
                258884, 258882, 11
            ]
        );
        assert_eq!(
            videos,
            vec![Gemma4UnifiedExpandedVideoPlaceholder {
                span: Gemma4UnifiedTokenSpan {
                    modality: Gemma4UnifiedModality::Video,
                    placeholder_index: 1,
                    replacement_start: 1,
                    soft_token_count: 6,
                    replacement_token_count: 13,
                },
                soft_token_ranges: vec![
                    Gemma4UnifiedSoftTokenRange {
                        start: 4,
                        soft_token_count: 3,
                    },
                    Gemma4UnifiedSoftTokenRange {
                        start: 10,
                        soft_token_count: 3,
                    },
                ],
            }]
        );
    }

    #[test]
    fn rejects_video_timestamp_count_mismatch() {
        let cfg = local_like_config();
        let error = cfg
            .video_replacement_tokens_with_ranges(Gemma4UnifiedVideoInput {
                frame_count: 2,
                soft_tokens_per_frame: 70,
                timestamp_token_ids_per_frame: vec![vec![901]],
            })
            .expect_err("timestamp count mismatch should fail");

        assert_eq!(
            error,
            Gemma4UnifiedError::InvalidField {
                field: "video.timestamp_token_ids_per_frame",
                message: "expected 2 timestamp entries, found 1".to_string(),
            }
        );
    }

    #[test]
    fn validates_processed_runtime_inputs_for_all_modalities() {
        let cfg = local_like_config();
        let (tokens, image_spans) = cfg
            .expand_image_placeholders(
                &[10, 258880, 258881, 258884, 11],
                &[Gemma4UnifiedImageInput {
                    width: 1,
                    height: 1,
                }],
            )
            .expect("image placeholder should expand");
        let (tokens, audio_spans) = cfg
            .expand_audio_placeholders(
                &tokens,
                &[Gemma4UnifiedAudioInput {
                    sample_count: 16000,
                }],
            )
            .expect("audio placeholder should expand");
        let (tokens, video_spans) = cfg
            .expand_video_placeholders(
                &tokens,
                &[Gemma4UnifiedVideoInput {
                    frame_count: 2,
                    soft_tokens_per_frame: 2,
                    timestamp_token_ids_per_frame: vec![vec![901], vec![902]],
                }],
            )
            .expect("video placeholder should expand");
        let image_soft = image_spans[0].soft_token_count as usize;
        let audio_soft = audio_spans[0].soft_token_count as usize;
        let video = &video_spans[0];
        let video_soft = video.span.soft_token_count as usize;

        let inputs = Gemma4UnifiedRuntimeInputs {
            images: vec![Gemma4UnifiedImageRuntimeInput {
                span: image_spans[0].clone(),
                pixel_values: vec![0.25; image_soft * 3],
                pixel_position_ids: (0..image_soft).map(|idx| [idx as i32, 0]).collect(),
            }],
            audios: vec![Gemma4UnifiedAudioRuntimeInput {
                span: audio_spans[0].clone(),
                input_features: vec![0.5; audio_soft * 4],
                frame_count: audio_soft as u32,
                feature_count: 4,
            }],
            videos: vec![Gemma4UnifiedVideoRuntimeInput {
                span: video.span.clone(),
                soft_token_ranges: video.soft_token_ranges.clone(),
                pixel_values: vec![0.75; video_soft * 3],
                pixel_position_ids: (0..video_soft).map(|idx| [idx as i32, 0]).collect(),
                frame_count: 2,
            }],
        };

        inputs
            .validate_for_prompt_len(tokens.len())
            .expect("processed tensors should validate against expanded prompt");
    }

    #[test]
    fn rejects_processed_runtime_inputs_with_bad_video_ranges() {
        let inputs = Gemma4UnifiedRuntimeInputs {
            images: Vec::new(),
            audios: Vec::new(),
            videos: vec![Gemma4UnifiedVideoRuntimeInput {
                span: Gemma4UnifiedTokenSpan {
                    modality: Gemma4UnifiedModality::Video,
                    placeholder_index: 1,
                    replacement_start: 1,
                    soft_token_count: 4,
                    replacement_token_count: 8,
                },
                soft_token_ranges: vec![
                    Gemma4UnifiedSoftTokenRange {
                        start: 3,
                        soft_token_count: 2,
                    },
                    Gemma4UnifiedSoftTokenRange {
                        start: 4,
                        soft_token_count: 2,
                    },
                ],
                pixel_values: vec![0.0; 12],
                pixel_position_ids: vec![[0, 0], [1, 0], [2, 0], [3, 0]],
                frame_count: 2,
            }],
        };
        let error = inputs
            .validate_for_prompt_len(12)
            .expect_err("overlapping ranges must fail");

        assert_eq!(
            error,
            Gemma4UnifiedRuntimeInputError::InvalidField {
                field: "videos[0].soft_token_ranges[1].start".to_string(),
                message: "ranges must be non-overlapping and sorted".to_string(),
            }
        );
    }

    #[test]
    fn rejects_processed_runtime_inputs_with_image_tensor_mismatch() {
        let inputs = Gemma4UnifiedRuntimeInputs {
            images: vec![Gemma4UnifiedImageRuntimeInput {
                span: Gemma4UnifiedTokenSpan {
                    modality: Gemma4UnifiedModality::Image,
                    placeholder_index: 1,
                    replacement_start: 1,
                    soft_token_count: 2,
                    replacement_token_count: 4,
                },
                pixel_values: vec![0.0, 1.0, 2.0],
                pixel_position_ids: vec![[0, 0]],
            }],
            audios: Vec::new(),
            videos: Vec::new(),
        };
        let error = inputs
            .validate_for_prompt_len(6)
            .expect_err("valid patch count mismatch must fail");

        assert_eq!(
            error,
            Gemma4UnifiedRuntimeInputError::InvalidField {
                field: "images[0].pixel_position_ids".to_string(),
                message: "contains 1 valid patch positions, but span expects 2 soft tokens"
                    .to_string(),
            }
        );
    }

    #[test]
    fn rejects_processed_runtime_inputs_with_non_prefix_image_padding() {
        let inputs = Gemma4UnifiedRuntimeInputs {
            images: vec![Gemma4UnifiedImageRuntimeInput {
                span: Gemma4UnifiedTokenSpan {
                    modality: Gemma4UnifiedModality::Image,
                    placeholder_index: 1,
                    replacement_start: 1,
                    soft_token_count: 2,
                    replacement_token_count: 4,
                },
                pixel_values: vec![0.0; 9],
                pixel_position_ids: vec![[0, 0], [-1, -1], [1, 0]],
            }],
            audios: Vec::new(),
            videos: Vec::new(),
        };
        let error = inputs
            .validate_for_prompt_len(6)
            .expect_err("valid patch positions after padding must fail");

        assert_eq!(
            error,
            Gemma4UnifiedRuntimeInputError::InvalidField {
                field: "images[0].pixel_position_ids[2]".to_string(),
                message: "valid patch positions must be a prefix before [-1, -1] padding rows"
                    .to_string(),
            }
        );
    }
}
