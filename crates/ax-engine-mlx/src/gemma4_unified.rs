use ax_engine_core::gemma4_unified::{
    Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedRuntimeInputs,
    Gemma4UnifiedTokenSpan, Gemma4UnifiedVideoRuntimeInput,
};
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, layer_norm, multiply, reshape, rms_norm, slice, slice_update,
    take,
};

use crate::model::{ModelConfig, embed_tokens};
use crate::model::{scale_hidden_pub, shared::qw};
use crate::weights::{Gemma4UnifiedAudioWeights, Gemma4UnifiedVisionWeights, ModelWeights};

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
) -> Result<Gemma4UnifiedChunkEmbeddings, String> {
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
        hidden = overwrite_span(
            hidden,
            &embeddings,
            &video.span,
            chunk_start,
            cfg.hidden_size,
        )?;
        push_media_range(&mut media_ranges, &video.span);
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
) -> Result<MlxArray, String> {
    let vision = weights
        .gemma4_unified_vision
        .as_ref()
        .ok_or_else(|| "Gemma4 unified image input requires vision weights".to_string())?;
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
) -> Result<MlxArray, String> {
    let vision = weights
        .gemma4_unified_vision
        .as_ref()
        .ok_or_else(|| "Gemma4 unified video input requires vision weights".to_string())?;
    if video.frame_count == 0 {
        return Err("Gemma4 unified video input frame_count must be greater than zero".to_string());
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
) -> Result<MlxArray, String> {
    let audio_weights = weights
        .gemma4_unified_audio
        .as_ref()
        .ok_or_else(|| "Gemma4 unified audio input requires audio weights".to_string())?;
    embed_audio_features(cfg, audio_weights, audio)
}

fn embed_vision_patch_values(
    cfg: &ModelConfig,
    weights: &Gemma4UnifiedVisionWeights,
    pixel_values: &[f32],
    pixel_position_ids: &[[i32; 2]],
    expected_soft_tokens: usize,
) -> Result<MlxArray, String> {
    let patch_count = pixel_position_ids.len();
    if patch_count == 0 {
        return Err("Gemma4 unified image/video input has no patches".to_string());
    }
    let patch_dim = pixel_values
        .len()
        .checked_div(patch_count)
        .filter(|dim| *dim > 0 && *dim * patch_count == pixel_values.len())
        .ok_or_else(|| {
            "Gemma4 unified image/video pixel_values length must divide by patch count".to_string()
        })?;

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
        return Err(format!(
            "Gemma4 unified image/video soft-token count mismatch: span expects {expected_soft_tokens}, processor tensors contain {} valid patches",
            valid_indices.len()
        ));
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
) -> Result<MlxArray, String> {
    let frame_count = audio.frame_count as usize;
    let feature_count = audio.feature_count as usize;
    if frame_count == 0 || feature_count == 0 {
        return Err(
            "Gemma4 unified audio input frame_count and feature_count must be greater than zero"
                .to_string(),
        );
    }
    if audio.input_features.len() != frame_count * feature_count {
        return Err(format!(
            "Gemma4 unified audio feature length mismatch: got {}, expected {}",
            audio.input_features.len(),
            frame_count * feature_count
        ));
    }
    if audio.span.soft_token_count as usize != frame_count {
        return Err(format!(
            "Gemma4 unified audio soft-token count mismatch: span expects {}, features contain {frame_count} frames",
            audio.span.soft_token_count
        ));
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
) -> Result<MlxArray, String> {
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
