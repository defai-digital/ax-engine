use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

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
            audio: feature_extractor.map(|feature_extractor| Gemma4UnifiedAudioProcessor {
                sampling_rate: optional_u32(feature_extractor, "sampling_rate").unwrap_or(16000),
                audio_seq_length: optional_u32(processor_config, "audio_seq_length"),
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

        let patch_size = self.patch_size as f64;
        let pooling = self.pooling_kernel_size as f64;
        let unit = patch_size * pooling;
        let max_patches = self.max_soft_tokens as f64 * pooling * pooling;
        let original_patches =
            (image_height as f64 / patch_size) * (image_width as f64 / patch_size);
        let scale = (max_patches / original_patches).sqrt();
        let target_h = unit.max(((image_height as f64 * scale / unit).floor()) * unit);
        let target_w = unit.max(((image_width as f64 * scale / unit).floor()) * unit);
        let num_patches = ((target_h / patch_size) as u32) * ((target_w / patch_size) as u32);
        Ok((num_patches / self.pooling_kernel_size.pow(2)).min(self.max_soft_tokens))
    }
}

impl Gemma4UnifiedAudioProcessor {
    pub fn compute_soft_tokens(&self, sample_count: u32) -> u32 {
        if sample_count == 0 || self.sampling_rate == 0 {
            return 0;
        }
        let sampling_rate = self.sampling_rate as f64;
        let frame_length = (sampling_rate * 20.0 / 1000.0).round() as i64;
        let hop_length = (sampling_rate * 10.0 / 1000.0).round() as i64;
        let frame_size_for_unfold = frame_length + 1;
        let pad_left = frame_length / 2;
        let padded_samples = i64::from(sample_count) + pad_left;
        let num_mel_frames = (padded_samples - frame_size_for_unfold) / hop_length + 1;
        if num_mel_frames <= 0 {
            return 0;
        }
        let mut t = num_mel_frames;
        for _ in 0..2 {
            t = (t + 2 - 3) / 2 + 1;
        }
        let tokens = t.max(0) as u32;
        self.audio_seq_length
            .map(|limit| tokens.min(limit))
            .unwrap_or(tokens)
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

        assert_eq!(
            cfg.audio_soft_tokens(Gemma4UnifiedAudioInput {
                sample_count: 16000
            })
            .unwrap(),
            25
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
}
