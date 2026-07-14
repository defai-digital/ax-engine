use ax_engine_core::gemma4_unified::{
    Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedRuntimeInputs,
    Gemma4UnifiedSoftTokenRange, Gemma4UnifiedTokenSpan, Gemma4UnifiedVideoRuntimeInput,
};
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, layer_norm, multiply, reshape, rms_norm, slice, slice_update,
    take,
};

use thiserror::Error;

use crate::model::{ModelConfig, embed_tokens};
use crate::model::{scale_hidden_pub, shared::qw};
use crate::weights::{Gemma4UnifiedAudioWeights, Gemma4UnifiedVisionWeights, ModelWeights};

#[derive(Debug, Error)]
pub enum Gemma4UnifiedError {
    #[error("Gemma4 unified image input requires vision weights")]
    MissingVisionWeights,
    #[error("Gemma4 unified video input requires vision weights")]
    MissingVideoVisionWeights,
    #[error("Gemma4 unified audio input requires audio weights")]
    MissingAudioWeights,
    #[error("Gemma4 unified video input frame_count must be greater than zero")]
    ZeroVideoFrameCount,
    #[error("Gemma4 unified image/video input has no patches")]
    NoPatches,
    #[error("Gemma4 unified image/video pixel_values length must divide by patch count")]
    PixelValuesNotDivisible,
    #[error(
        "Gemma4 unified image/video soft-token count mismatch: span expects {expected}, processor tensors contain {actual} valid patches"
    )]
    SoftTokenMismatch { expected: usize, actual: usize },
    #[error("Gemma4 unified audio input frame_count and feature_count must be greater than zero")]
    ZeroAudioFrameOrFeatureCount,
    #[error("Gemma4 unified audio feature length mismatch: got {got}, expected {expected}")]
    AudioFeatureLengthMismatch { got: usize, expected: usize },
    #[error(
        "Gemma4 unified audio soft-token count mismatch: span expects {expected}, features contain {actual} frames"
    )]
    AudioSoftTokenMismatch { expected: u32, actual: usize },
    #[error("Gemma4 unified video soft-token range overflow")]
    VideoSoftTokenRangeOverflow,
    #[error(
        "Gemma4 unified video soft-token range mismatch: ranges expect {expected}, embeddings contain {actual}"
    )]
    VideoSoftTokenRangeCountMismatch { expected: usize, actual: usize },
}

const GEMMA4_UNIFIED_LAYER_NORM_EPS: f32 = 1.0e-5;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Gemma4UnifiedMediaRange {
    pub start: usize,
    pub end_inclusive: usize,
}

#[derive(Debug)]
pub(crate) struct Gemma4UnifiedChunkEmbeddings {
    pub hidden: MlxArray,
    pub media_ranges: Vec<Gemma4UnifiedMediaRange>,
}

pub(crate) fn build_chunk_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    chunk_start: usize,
    inputs: &Gemma4UnifiedRuntimeInputs,
) -> Result<Gemma4UnifiedChunkEmbeddings, Gemma4UnifiedError> {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden_pub(&hidden, scale);
    }

    let mut media_ranges = Vec::new();
    for image in &inputs.images {
        let embeddings = image_embeddings(cfg, weights, image)?;
        hidden = overwrite_span(
            hidden,
            &embeddings,
            &image.span,
            chunk_start,
            cfg.hidden_size,
        )?;
        push_media_range(&mut media_ranges, &image.span);
    }
    for audio in &inputs.audios {
        let embeddings = audio_embeddings(cfg, weights, audio)?;
        hidden = overwrite_span(
            hidden,
            &embeddings,
            &audio.span,
            chunk_start,
            cfg.hidden_size,
        )?;
    }
    for video in &inputs.videos {
        let embeddings = video_embeddings(cfg, weights, video)?;
        hidden = overwrite_video_spans(hidden, &embeddings, video, chunk_start, cfg.hidden_size)?;
        push_video_media_ranges(&mut media_ranges, video);
    }

    Ok(Gemma4UnifiedChunkEmbeddings {
        hidden,
        media_ranges,
    })
}

fn image_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    image: &Gemma4UnifiedImageRuntimeInput,
) -> Result<MlxArray, Gemma4UnifiedError> {
    let vision = weights
        .gemma4_unified_vision
        .as_ref()
        .ok_or(Gemma4UnifiedError::MissingVisionWeights)?;
    embed_vision_patch_values(
        cfg,
        vision,
        &image.pixel_values,
        &image.pixel_position_ids,
        image.span.soft_token_count as usize,
    )
}

fn video_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    video: &Gemma4UnifiedVideoRuntimeInput,
) -> Result<MlxArray, Gemma4UnifiedError> {
    let vision = weights
        .gemma4_unified_vision
        .as_ref()
        .ok_or(Gemma4UnifiedError::MissingVideoVisionWeights)?;
    if video.frame_count == 0 {
        return Err(Gemma4UnifiedError::ZeroVideoFrameCount);
    }
    embed_vision_patch_values(
        cfg,
        vision,
        &video.pixel_values,
        &video.pixel_position_ids,
        video.span.soft_token_count as usize,
    )
}

fn audio_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    audio: &Gemma4UnifiedAudioRuntimeInput,
) -> Result<MlxArray, Gemma4UnifiedError> {
    let audio_weights = weights
        .gemma4_unified_audio
        .as_ref()
        .ok_or(Gemma4UnifiedError::MissingAudioWeights)?;
    embed_audio_features(cfg, audio_weights, audio)
}

fn embed_vision_patch_values(
    cfg: &ModelConfig,
    weights: &Gemma4UnifiedVisionWeights,
    pixel_values: &[f32],
    pixel_position_ids: &[[i32; 2]],
    expected_soft_tokens: usize,
) -> Result<MlxArray, Gemma4UnifiedError> {
    let patch_count = pixel_position_ids.len();
    if patch_count == 0 {
        return Err(Gemma4UnifiedError::NoPatches);
    }
    let patch_dim = pixel_values
        .len()
        .checked_div(patch_count)
        .filter(|dim| *dim > 0 && *dim * patch_count == pixel_values.len())
        .ok_or(Gemma4UnifiedError::PixelValuesNotDivisible)?;

    let pixel = MlxArray::from_raw_data(
        pixel_values.as_ptr().cast(),
        std::mem::size_of_val(pixel_values),
        &[1, patch_count as i32, patch_dim as i32],
        MlxDtype::Float32,
    );
    let pixel = astype(&pixel, weights.pos_embedding.dtype(), None);
    let hidden = layer_norm(
        &pixel,
        &weights.patch_ln1_weight,
        &weights.patch_ln1_bias,
        GEMMA4_UNIFIED_LAYER_NORM_EPS,
        None,
    );
    let hidden = add(
        &qw(&hidden, &weights.patch_dense),
        &weights.patch_dense_bias,
        None,
    );
    let hidden = layer_norm(
        &hidden,
        &weights.patch_ln2_weight,
        &weights.patch_ln2_bias,
        GEMMA4_UNIFIED_LAYER_NORM_EPS,
        None,
    );
    let pos = factorized_position_embedding(weights, pixel_position_ids, cfg.hidden_size);
    let hidden = add(&hidden, &pos, None);
    let hidden = layer_norm(
        &hidden,
        &weights.pos_norm_weight,
        &weights.pos_norm_bias,
        GEMMA4_UNIFIED_LAYER_NORM_EPS,
        None,
    );
    let hidden = rms_norm(&hidden, None, cfg.rms_norm_eps, None);
    let projected = qw(&hidden, &weights.projection);
    let projected = reshape(
        &projected,
        &[patch_count as i32, cfg.hidden_size as i32],
        None,
    );
    let valid_indices = valid_patch_indices(pixel_position_ids);
    if valid_indices.len() != expected_soft_tokens {
        return Err(Gemma4UnifiedError::SoftTokenMismatch {
            expected: expected_soft_tokens,
            actual: valid_indices.len(),
        });
    }
    if valid_indices.len() == patch_count {
        Ok(projected)
    } else {
        let indices = u32_array(&valid_indices);
        Ok(take(&projected, &indices, 0, None))
    }
}

fn factorized_position_embedding(
    weights: &Gemma4UnifiedVisionWeights,
    pixel_position_ids: &[[i32; 2]],
    hidden_size: usize,
) -> MlxArray {
    let patch_count = pixel_position_ids.len();
    let pos_shape = weights.pos_embedding.shape();
    let posemb_size = pos_shape.first().copied().unwrap_or(0);
    let hidden = hidden_size as i32;
    let axis_x = slice(
        &weights.pos_embedding,
        &[0, 0, 0],
        &[posemb_size, 1, hidden],
        &[1, 1, 1],
        None,
    );
    let axis_x = reshape(&axis_x, &[posemb_size, hidden], None);
    let axis_y = slice(
        &weights.pos_embedding,
        &[0, 1, 0],
        &[posemb_size, 2, hidden],
        &[1, 1, 1],
        None,
    );
    let axis_y = reshape(&axis_y, &[posemb_size, hidden], None);

    let mut x_indices = Vec::with_capacity(patch_count);
    let mut y_indices = Vec::with_capacity(patch_count);
    let mut x_mask = Vec::with_capacity(patch_count);
    let mut y_mask = Vec::with_capacity(patch_count);
    for [x, y] in pixel_position_ids {
        x_indices.push((*x).max(0) as u32);
        y_indices.push((*y).max(0) as u32);
        x_mask.push(if *x >= 0 { 1.0_f32 } else { 0.0 });
        y_mask.push(if *y >= 0 { 1.0_f32 } else { 0.0 });
    }

    let x = take(&axis_x, &u32_array(&x_indices), 0, None);
    let y = take(&axis_y, &u32_array(&y_indices), 0, None);
    let x_mask = mask_array(&x_mask, patch_count, weights.pos_embedding.dtype());
    let y_mask = mask_array(&y_mask, patch_count, weights.pos_embedding.dtype());
    let pos = add(
        &multiply(&x, &x_mask, None),
        &multiply(&y, &y_mask, None),
        None,
    );
    reshape(&pos, &[1, patch_count as i32, hidden], None)
}

fn embed_audio_features(
    cfg: &ModelConfig,
    weights: &Gemma4UnifiedAudioWeights,
    audio: &Gemma4UnifiedAudioRuntimeInput,
) -> Result<MlxArray, Gemma4UnifiedError> {
    let frame_count = audio.frame_count as usize;
    let feature_count = audio.feature_count as usize;
    if frame_count == 0 || feature_count == 0 {
        return Err(Gemma4UnifiedError::ZeroAudioFrameOrFeatureCount);
    }
    if audio.input_features.len() != frame_count * feature_count {
        return Err(Gemma4UnifiedError::AudioFeatureLengthMismatch {
            got: audio.input_features.len(),
            expected: frame_count * feature_count,
        });
    }
    if audio.span.soft_token_count as usize != frame_count {
        return Err(Gemma4UnifiedError::AudioSoftTokenMismatch {
            expected: audio.span.soft_token_count,
            actual: frame_count,
        });
    }
    let features = MlxArray::from_raw_data(
        audio.input_features.as_ptr().cast(),
        std::mem::size_of_val(audio.input_features.as_slice()),
        &[1, frame_count as i32, feature_count as i32],
        MlxDtype::Float32,
    );
    let features = astype(&features, MlxDtype::Bfloat16, None);
    let normed = rms_norm(&features, None, cfg.rms_norm_eps, None);
    let projected = qw(&normed, &weights.projection);
    Ok(reshape(
        &projected,
        &[frame_count as i32, cfg.hidden_size as i32],
        None,
    ))
}

fn overwrite_span(
    hidden: MlxArray,
    embeddings: &MlxArray,
    span: &Gemma4UnifiedTokenSpan,
    chunk_start: usize,
    hidden_size: usize,
) -> Result<MlxArray, Gemma4UnifiedError> {
    let soft_start = span.replacement_start.saturating_add(1);
    let soft_end = soft_start.saturating_add(span.soft_token_count as usize);
    let chunk_end =
        chunk_start.saturating_add(hidden.shape().get(1).copied().unwrap_or(0) as usize);
    let start = soft_start.max(chunk_start);
    let end = soft_end.min(chunk_end);
    if start >= end {
        return Ok(hidden);
    }

    let source_start = start - soft_start;
    let source_end = end - soft_start;
    let local_start = start - chunk_start;
    let local_end = end - chunk_start;
    let update = slice(
        embeddings,
        &[source_start as i32, 0],
        &[source_end as i32, hidden_size as i32],
        &[1, 1],
        None,
    );
    let update = reshape(
        &update,
        &[1, (source_end - source_start) as i32, hidden_size as i32],
        None,
    );
    Ok(slice_update(
        &hidden,
        &update,
        &[0, local_start as i32, 0],
        &[1, local_end as i32, hidden_size as i32],
        &[1, 1, 1],
        None,
    ))
}

fn overwrite_video_spans(
    hidden: MlxArray,
    embeddings: &MlxArray,
    video: &Gemma4UnifiedVideoRuntimeInput,
    chunk_start: usize,
    hidden_size: usize,
) -> Result<MlxArray, Gemma4UnifiedError> {
    if video.soft_token_ranges.is_empty() {
        return overwrite_span(hidden, embeddings, &video.span, chunk_start, hidden_size);
    }
    overwrite_soft_token_ranges(
        hidden,
        embeddings,
        &video.soft_token_ranges,
        chunk_start,
        hidden_size,
    )
}

fn overwrite_soft_token_ranges(
    mut hidden: MlxArray,
    embeddings: &MlxArray,
    ranges: &[Gemma4UnifiedSoftTokenRange],
    chunk_start: usize,
    hidden_size: usize,
) -> Result<MlxArray, Gemma4UnifiedError> {
    let expected_tokens = ranges.iter().try_fold(0usize, |acc, range| {
        acc.checked_add(range.soft_token_count as usize)
            .ok_or(Gemma4UnifiedError::VideoSoftTokenRangeOverflow)
    })?;
    let actual_tokens = embeddings.shape().first().copied().unwrap_or(0) as usize;
    if expected_tokens != actual_tokens {
        return Err(Gemma4UnifiedError::VideoSoftTokenRangeCountMismatch {
            expected: expected_tokens,
            actual: actual_tokens,
        });
    }

    let chunk_end =
        chunk_start.saturating_add(hidden.shape().get(1).copied().unwrap_or(0) as usize);
    let mut source_cursor = 0usize;
    for range in ranges {
        let soft_start = range.start;
        let soft_end = soft_start.saturating_add(range.soft_token_count as usize);
        let start = soft_start.max(chunk_start);
        let end = soft_end.min(chunk_end);
        if start < end {
            let source_start = source_cursor + start - soft_start;
            let source_end = source_cursor + end - soft_start;
            let local_start = start - chunk_start;
            let local_end = end - chunk_start;
            let update = slice(
                embeddings,
                &[source_start as i32, 0],
                &[source_end as i32, hidden_size as i32],
                &[1, 1],
                None,
            );
            let update = reshape(
                &update,
                &[1, (source_end - source_start) as i32, hidden_size as i32],
                None,
            );
            hidden = slice_update(
                &hidden,
                &update,
                &[0, local_start as i32, 0],
                &[1, local_end as i32, hidden_size as i32],
                &[1, 1, 1],
                None,
            );
        }
        source_cursor += range.soft_token_count as usize;
    }
    Ok(hidden)
}

fn push_media_range(ranges: &mut Vec<Gemma4UnifiedMediaRange>, span: &Gemma4UnifiedTokenSpan) {
    if span.soft_token_count == 0 {
        return;
    }
    let start = span.replacement_start.saturating_add(1);
    ranges.push(Gemma4UnifiedMediaRange {
        start,
        end_inclusive: start + span.soft_token_count as usize - 1,
    });
}

fn push_video_media_ranges(
    ranges: &mut Vec<Gemma4UnifiedMediaRange>,
    video: &Gemma4UnifiedVideoRuntimeInput,
) {
    if video.soft_token_ranges.is_empty() {
        push_media_range(ranges, &video.span);
        return;
    }
    for range in &video.soft_token_ranges {
        if range.soft_token_count == 0 {
            continue;
        }
        let Some(end_inclusive) = range
            .start
            .checked_add(range.soft_token_count as usize)
            .and_then(|end| end.checked_sub(1))
        else {
            continue;
        };
        ranges.push(Gemma4UnifiedMediaRange {
            start: range.start,
            end_inclusive,
        });
    }
}

fn valid_patch_indices(pixel_position_ids: &[[i32; 2]]) -> Vec<u32> {
    pixel_position_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, [x, y])| (*x >= 0 && *y >= 0).then_some(idx as u32))
        .collect()
}

fn u32_array(values: &[u32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        &[values.len() as i32],
        MlxDtype::Uint32,
    )
}

fn mask_array(values: &[f32], len: usize, dtype: MlxDtype) -> MlxArray {
    let arr = MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        &[len as i32, 1],
        MlxDtype::Float32,
    );
    astype(&arr, dtype, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4_assistant_mtp::Gemma4AssistantMtpStatus;
    use crate::model::ModelConfig;
    use crate::weights::{ModelWeights, QuantizedWeight};
    use ax_engine_core::gemma4_unified::{
        Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedModality, Gemma4UnifiedSoftTokenRange,
    };
    use mlx_sys::{MlxDtype, zeros};

    fn text_only_weights() -> ModelWeights {
        let dummy = || QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None);
        ModelWeights {
            token_embedding: dummy(),
            final_norm: zeros(&[1], MlxDtype::Float32, None),
            lm_head: dummy(),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            gemma4_assistant_mtp: Gemma4AssistantMtpStatus::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            diffusion_self_conditioning: None,
        }
    }

    fn text_only_config() -> ModelConfig {
        ModelConfig {
            compile_cache_identity: 4,
            model_family: "gemma4_unified".to_string(),
            layer_count: 0,
            hidden_size: 1,
            intermediate_size: 0,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            vocab_size: 1,
            rope_theta: 10000.0,
            rope_dims: 0,
            attn_output_gate: false,
            query_scale: 1.0,
            final_logit_softcapping: None,
            moe_expert_count: 0,
            moe_experts_per_token: 0,
            moe_expert_intermediate_size: 0,
            layer_configs: Vec::new(),
            global_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: true,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 0.0,
            attn_temperature_scale: 0.0,
            intermediate_size_mlp: 0,
            moe_layer_freq: 0,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            gpt_oss_uses_mxfp4_experts: false,
        }
    }

    fn stub_image() -> Gemma4UnifiedImageRuntimeInput {
        Gemma4UnifiedImageRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Image,
                placeholder_index: 0,
                replacement_start: 0,
                soft_token_count: 1,
                replacement_token_count: 1,
            },
            pixel_values: vec![0.0],
            pixel_position_ids: vec![[0, 0]],
        }
    }

    fn stub_audio() -> Gemma4UnifiedAudioRuntimeInput {
        Gemma4UnifiedAudioRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Audio,
                placeholder_index: 0,
                replacement_start: 0,
                soft_token_count: 1,
                replacement_token_count: 1,
            },
            input_features: vec![0.0],
            frame_count: 1,
            feature_count: 1,
        }
    }

    #[test]
    fn image_input_rejected_when_vision_weights_absent() {
        let err =
            image_embeddings(&text_only_config(), &text_only_weights(), &stub_image()).unwrap_err();
        assert!(matches!(err, Gemma4UnifiedError::MissingVisionWeights));
    }

    #[test]
    fn audio_input_rejected_when_audio_weights_absent() {
        let err =
            audio_embeddings(&text_only_config(), &text_only_weights(), &stub_audio()).unwrap_err();
        assert!(matches!(err, Gemma4UnifiedError::MissingAudioWeights));
    }

    #[test]
    fn video_input_rejected_when_vision_weights_absent() {
        let video = video_with_ranges(vec![Gemma4UnifiedSoftTokenRange {
            start: 0,
            soft_token_count: 1,
        }]);
        let err = video_embeddings(&text_only_config(), &text_only_weights(), &video).unwrap_err();
        assert!(matches!(err, Gemma4UnifiedError::MissingVideoVisionWeights));
    }

    fn video_with_ranges(
        ranges: Vec<Gemma4UnifiedSoftTokenRange>,
    ) -> Gemma4UnifiedVideoRuntimeInput {
        Gemma4UnifiedVideoRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Video,
                placeholder_index: 1,
                replacement_start: 1,
                soft_token_count: 6,
                replacement_token_count: 13,
            },
            soft_token_ranges: ranges,
            pixel_values: Vec::new(),
            pixel_position_ids: Vec::new(),
            frame_count: 2,
        }
    }

    #[test]
    fn video_media_ranges_follow_non_contiguous_frame_ranges() {
        let video = video_with_ranges(vec![
            Gemma4UnifiedSoftTokenRange {
                start: 4,
                soft_token_count: 3,
            },
            Gemma4UnifiedSoftTokenRange {
                start: 10,
                soft_token_count: 3,
            },
        ]);
        let mut ranges = Vec::new();

        push_video_media_ranges(&mut ranges, &video);

        assert_eq!(
            ranges,
            vec![
                Gemma4UnifiedMediaRange {
                    start: 4,
                    end_inclusive: 6,
                },
                Gemma4UnifiedMediaRange {
                    start: 10,
                    end_inclusive: 12,
                },
            ]
        );
    }

    #[test]
    fn video_media_ranges_fall_back_to_legacy_contiguous_span() {
        let video = video_with_ranges(Vec::new());
        let mut ranges = Vec::new();

        push_video_media_ranges(&mut ranges, &video);

        assert_eq!(
            ranges,
            vec![Gemma4UnifiedMediaRange {
                start: 2,
                end_inclusive: 7,
            }]
        );
    }
}
