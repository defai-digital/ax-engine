//! Native Unlimited-OCR dual vision tower (SAM-ViT-B + CLIP-L) + projector.
//!
//! Mirrors mlx-vlm `deepseekocr/sam.py`, `deepseekocr/vision.py`, and the
//! base-resolution (global-only) path in `unlimited_ocr/unlimitedocr.py`.
//! Vision weights stay BF16 and are pulled from leftover safetensors keys
//! after language load; the mxfp8 projector / image_newline / view_separator
//! come from manifest roles.

use std::collections::HashMap;

use ax_engine_core::{UNLIMITED_OCR_LOCAL_QUERY_GRID, UNLIMITED_OCR_LOCAL_TILE_SIZE};
use image::{Rgb, RgbImage, imageops};
use mlx_sys::{
    MlxArray, MlxDtype, ScaledDotProductAttentionMask, add, arange, astype, broadcast_to,
    concatenate, conv2d, expand_dims, flatten, gelu, layer_norm, matmul, multiply, pad, reshape,
    scaled_dot_product_attention, scaled_dot_product_attention_with_mask, sigmoid, slice,
    slice_update, split, subtract, take, transpose, zeros,
};
use thiserror::Error;

use crate::model::shared::qw;
use crate::model::{ModelConfig, embed_tokens, scale_hidden_pub};
use crate::weights::{ModelWeights, QuantizedWeight};

/// Default `<image>` soft-token id in the Unlimited-OCR tokenizer.
pub const DEFAULT_IMAGE_TOKEN_ID: u32 = 128_815;

const IMAGE_SIZE: i32 = 1024;
const LOCAL_IMAGE_SIZE: i32 = UNLIMITED_OCR_LOCAL_TILE_SIZE as i32;
const SAM_PATCH: i32 = 16;
const SAM_WIDTH: i32 = 768;
const SAM_DEPTH: usize = 12;
const SAM_HEADS: usize = 12;
const SAM_WINDOW: i32 = 14;
const SAM_GLOBAL_ATTN: [usize; 4] = [2, 5, 8, 11];
const CLIP_WIDTH: i32 = 1024;
const CLIP_LAYERS: usize = 24;
const CLIP_HEADS: usize = 16;
const CLIP_INTERMEDIATE: i32 = 4096;
/// SAM ViT LayerNorm / LayerNorm2d epsilon (`eps=1e-6` in deepencoder.py).
const SAM_LN_EPS: f32 = 1.0e-6;
/// The CLIP embedding pre-normalization keeps MLX LayerNorm's 1e-5 default.
const CLIP_PRE_LN_EPS: f32 = 1.0e-5;
/// CLIP transformer LayerNorm epsilon: `layernorm_epsilon = 1e-5` in the
/// released `deepencoder.py` (shipped with the v1 and v2 checkpoints).
const CLIP_LAYER_LN_EPS: f32 = 1.0e-5;
/// OpenCLIP QuickGELU coefficient: `deepencoder.py` `quick_gelu(x) =
/// x * sigmoid(1.702 * x)` in the CLIP MLP.
const QUICK_GELU_SCALE: f32 = 1.702;
const DOWNSAMPLE_RATIO: i32 = 4;

#[derive(Debug, Error)]
pub enum UnlimitedOcrError {
    #[error("Unlimited-OCR vision weights are not loaded")]
    MissingVisionWeights,
    #[error("Unlimited-OCR weight missing: {0}")]
    WeightMissing(String),
    #[error("Unlimited-OCR pixel_values empty or not divisible by 3×{IMAGE_SIZE}×{IMAGE_SIZE}")]
    BadPixelValues,
    #[error(
        "Unlimited-OCR soft-token count mismatch: prompt has {prompt_tokens} image tokens, vision produced {feature_tokens}"
    )]
    SoftTokenMismatch {
        prompt_tokens: usize,
        feature_tokens: usize,
    },
    #[error("Unlimited-OCR image feature assembly failed: {0}")]
    Assembly(String),
}

/// Dense Linear: weight [out, in] + optional bias [out].
#[derive(Clone)]
struct DenseLinear {
    weight: MlxArray,
    bias: Option<MlxArray>,
}

impl DenseLinear {
    fn forward(&self, x: &MlxArray) -> MlxArray {
        let wt = transpose(&self.weight, &[1, 0], None);
        let y = matmul(x, &wt, None);
        match &self.bias {
            Some(b) => add(&y, b, None),
            None => y,
        }
    }
}

#[derive(Clone)]
struct LayerNormW {
    weight: MlxArray,
    bias: MlxArray,
    eps: f32,
}

impl LayerNormW {
    fn forward(&self, x: &MlxArray) -> MlxArray {
        layer_norm(x, &self.weight, &self.bias, self.eps, None)
    }
}

#[derive(Clone)]
struct SamAttention {
    qkv: DenseLinear,
    proj: DenseLinear,
    rel_pos_h: Option<MlxArray>,
    rel_pos_w: Option<MlxArray>,
    rel_pos_h_local: Option<MlxArray>,
    rel_pos_w_local: Option<MlxArray>,
    num_heads: usize,
    scale: f32,
}

#[derive(Clone)]
struct SamBlock {
    norm1: LayerNormW,
    attn: SamAttention,
    norm2: LayerNormW,
    mlp_lin1: DenseLinear,
    mlp_lin2: DenseLinear,
    window_size: i32,
}

#[derive(Clone)]
struct SamEncoder {
    patch_embed: DenseLinearLikeConv,
    pos_embed: MlxArray,
    pos_embed_local: MlxArray,
    blocks: Vec<SamBlock>,
    neck0: DenseLinearLikeConv,
    neck1: LayerNormW,
    neck2: DenseLinearLikeConv,
    neck3: LayerNormW,
    net_2: DenseLinearLikeConv,
    net_3: DenseLinearLikeConv,
}

/// Conv2d stored as OHWI weight (+ optional bias broadcast over NHWC).
#[derive(Clone)]
struct DenseLinearLikeConv {
    weight: MlxArray,
    bias: Option<MlxArray>,
    stride: i32,
    padding: i32,
}

impl DenseLinearLikeConv {
    fn forward(&self, x: &MlxArray) -> MlxArray {
        let y = conv2d(x, &self.weight, self.stride, self.padding, 1, 1, None);
        match &self.bias {
            Some(b) => {
                // bias [C] → broadcast on NHWC
                let shape = y.shape();
                let b = reshape(b, &[1, 1, 1, shape[3]], None);
                add(&y, &b, None)
            }
            None => y,
        }
    }
}

#[derive(Clone)]
struct ClipAttention {
    qkv: DenseLinear,
    out_proj: DenseLinear,
    num_heads: usize,
    scale: f32,
}

#[derive(Clone)]
struct ClipEncoderLayer {
    ln1: LayerNormW,
    attn: ClipAttention,
    ln2: LayerNormW,
    fc1: DenseLinear,
    fc2: DenseLinear,
}

#[derive(Clone)]
struct ClipVision {
    class_embedding: MlxArray,
    /// Unused when SAM patch embeds are supplied (base Unlimited-OCR path).
    #[allow(dead_code)]
    patch_embedding: DenseLinearLikeConv,
    position_embedding: MlxArray,
    position_embedding_local: MlxArray,
    pre_layernorm: LayerNormW,
    layers: Vec<ClipEncoderLayer>,
}

/// Loaded Unlimited-OCR dual vision + projector bundle.
pub struct UnlimitedOcrVisionWeights {
    sam: SamEncoder,
    clip: ClipVision,
    projector: QuantizedWeight,
    image_newline: MlxArray,
    view_separator: MlxArray,
    pub image_size: i32,
    pub image_token_id: u32,
}

/// Preprocessed global view plus the released processor's optional local tiles.
pub struct UnlimitedOcrImageViews {
    pub global_nhwc: MlxArray,
    pub local_nhwc: Option<MlxArray>,
    pub crop_columns: u32,
    pub crop_rows: u32,
}

/// Soft-token count for base-resolution global-only mode.
///
/// `num_queries = ceil((image_size / patch_size) / downsample_ratio)`
/// tokens = `(num_queries + 1) * num_queries + 1`  (grid + per-row newline + separator)
pub(crate) fn base_soft_token_count(
    image_size: i32,
    patch_size: i32,
    downsample_ratio: i32,
) -> usize {
    let grid = (image_size / patch_size) as f32 / downsample_ratio as f32;
    let num_queries = grid.ceil() as usize;
    (num_queries + 1) * num_queries + 1
}

/// Default base-mode soft token count for 1024 / SAM patch 16 / downsample 4 → 273.
pub fn default_base_soft_token_count() -> usize {
    base_soft_token_count(IMAGE_SIZE, SAM_PATCH, DOWNSAMPLE_RATIO)
}

fn cubic_weight(distance: f32, coefficient: f32) -> f32 {
    let value = distance.abs();
    if value <= 1.0 {
        (coefficient + 2.0) * value.powi(3) - (coefficient + 3.0) * value.powi(2) + 1.0
    } else if value < 2.0 {
        coefficient * value.powi(3) - 5.0 * coefficient * value.powi(2) + 8.0 * coefficient * value
            - 4.0 * coefficient
    } else {
        0.0
    }
}

fn cubic_axis_weights(input: usize, output: usize, antialias: bool) -> Vec<Vec<(usize, f32)>> {
    let scale = output as f32 / input as f32;
    let filter_scale = if antialias && scale < 1.0 {
        1.0 / scale
    } else {
        1.0
    };
    let support = 2.0 * filter_scale;
    let taps = (2.0 * support + 1.0) as usize;
    let coefficient = if antialias { -0.5 } else { -0.75 };

    (0..output)
        .map(|index| {
            let coordinate = (index as f32 + 0.5) / output as f32 * input as f32 - 0.5;
            let start = (coordinate - support).floor() as isize + 1;
            let mut values = Vec::with_capacity(taps);
            let mut total = 0.0_f32;
            for offset in 0..taps {
                let source = start + offset as isize;
                if source < 0 || source >= input as isize {
                    continue;
                }
                let weight = cubic_weight((coordinate - source as f32) / filter_scale, coefficient);
                if weight != 0.0 {
                    values.push((source as usize, weight));
                    total += weight;
                }
            }
            if total != 0.0 {
                for (_, weight) in &mut values {
                    *weight /= total;
                }
            }
            values
        })
        .collect()
}

fn resize_hwc_cubic(
    input: &[f32],
    input_height: usize,
    input_width: usize,
    channels: usize,
    output_height: usize,
    output_width: usize,
    antialias: bool,
) -> Vec<f32> {
    let y_weights = cubic_axis_weights(input_height, output_height, antialias);
    let x_weights = cubic_axis_weights(input_width, output_width, antialias);
    let mut vertical = vec![0.0_f32; output_height * input_width * channels];
    for (output_y, entries) in y_weights.iter().enumerate() {
        for x in 0..input_width {
            for channel in 0..channels {
                let mut value = 0.0_f32;
                for &(input_y, weight) in entries {
                    value += input[(input_y * input_width + x) * channels + channel] * weight;
                }
                vertical[(output_y * input_width + x) * channels + channel] = value;
            }
        }
    }

    let mut output = vec![0.0_f32; output_height * output_width * channels];
    for y in 0..output_height {
        for (output_x, entries) in x_weights.iter().enumerate() {
            for channel in 0..channels {
                let mut value = 0.0_f32;
                for &(input_x, weight) in entries {
                    value += vertical[(y * input_width + input_x) * channels + channel] * weight;
                }
                output[(y * output_width + output_x) * channels + channel] = value;
            }
        }
    }
    output
}

fn array_f32_values(array: &MlxArray) -> Vec<f32> {
    let values = astype(array, MlxDtype::Float32, None);
    mlx_sys::eval(&[&values]);
    values.data_f32().to_vec()
}

fn array_from_f32(values: &[f32], shape: &[i32], dtype: MlxDtype) -> MlxArray {
    let value = MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Float32,
    );
    let value = astype(&value, dtype, None);
    mlx_sys::eval(&[&value]);
    value
}

fn resize_sam_position_embedding(position: &MlxArray) -> Result<MlxArray, UnlimitedOcrError> {
    let shape = position.shape();
    // The resize reads a fixed 64x64 source grid; any other shape would index
    // past the tensor or be reinterpreted as the wrong image.
    if shape.len() != 4 || shape[..3] != [1, 64, 64] || shape[3] <= 0 {
        return Err(UnlimitedOcrError::Assembly(format!(
            "sam_model.pos_embed must be [1, 64, 64, C], got {shape:?}"
        )));
    }
    let channels = shape[3] as usize;
    let values = array_f32_values(position);
    let resized = resize_hwc_cubic(&values, 64, 64, channels, 40, 40, true);
    Ok(array_from_f32(
        &resized,
        &[1, 40, 40, shape[3]],
        position.dtype(),
    ))
}

fn resize_clip_position_embedding(position: &MlxArray) -> Result<MlxArray, UnlimitedOcrError> {
    let shape = position.shape();
    // One class row followed by a fixed 16x16 patch grid.
    if shape.len() != 2 || shape[0] != 257 || shape[1] <= 0 {
        return Err(UnlimitedOcrError::Assembly(format!(
            "vision position_embedding must be [257, C], got {shape:?}"
        )));
    }
    let channels = shape[1] as usize;
    let values = array_f32_values(position);
    // `get_abs_pos`: bicubic, `antialias=True`, `align_corners=False`.
    let resized = resize_hwc_cubic(&values[channels..], 16, 16, channels, 10, 10, true);
    let mut combined = Vec::with_capacity((1 + 100) * channels);
    combined.extend_from_slice(&values[..channels]);
    combined.extend_from_slice(&resized);
    Ok(array_from_f32(
        &combined,
        &[101, shape[1]],
        position.dtype(),
    ))
}

fn resize_relative_position(position: &MlxArray, output_length: usize) -> MlxArray {
    let shape = position.shape();
    let input_length = shape[0] as usize;
    let channels = shape[1] as usize;
    let values = array_f32_values(position);
    let scale = input_length as f32 / output_length as f32;
    let mut resized = vec![0.0_f32; output_length * channels];
    // `get_rel_pos`: `F.interpolate(mode="linear")` with the default
    // `align_corners=False`, i.e. half-pixel centres clamped at the start.
    for output in 0..output_length {
        let coordinate = ((output as f32 + 0.5) * scale - 0.5).max(0.0);
        let lower = (coordinate.floor() as usize).min(input_length - 1);
        let upper = (lower + 1).min(input_length - 1);
        let weight = coordinate - lower as f32;
        for channel in 0..channels {
            resized[output * channels + channel] = values[lower * channels + channel]
                * (1.0 - weight)
                + values[upper * channels + channel] * weight;
        }
    }
    array_from_f32(
        &resized,
        &[output_length as i32, shape[1]],
        position.dtype(),
    )
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

fn take_arr(map: &mut HashMap<String, MlxArray>, key: &str) -> Result<MlxArray, UnlimitedOcrError> {
    map.remove(key)
        .ok_or_else(|| UnlimitedOcrError::WeightMissing(key.to_string()))
}

fn take_linear(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
) -> Result<DenseLinear, UnlimitedOcrError> {
    let weight = take_arr(map, &format!("{base}.weight"))?;
    let bias = map.remove(&format!("{base}.bias"));
    Ok(DenseLinear { weight, bias })
}

fn take_ln(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
    eps: f32,
) -> Result<LayerNormW, UnlimitedOcrError> {
    Ok(LayerNormW {
        weight: take_arr(map, &format!("{base}.weight"))?,
        bias: take_arr(map, &format!("{base}.bias"))?,
        eps,
    })
}

fn take_conv(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
    stride: i32,
    padding: i32,
    with_bias: bool,
) -> Result<DenseLinearLikeConv, UnlimitedOcrError> {
    let weight = take_arr(map, &format!("{base}.weight"))?;
    let bias = if with_bias {
        Some(take_arr(map, &format!("{base}.bias"))?)
    } else {
        map.remove(&format!("{base}.bias"))
    };
    Ok(DenseLinearLikeConv {
        weight,
        bias,
        stride,
        padding,
    })
}

fn load_sam(map: &mut HashMap<String, MlxArray>) -> Result<SamEncoder, UnlimitedOcrError> {
    let patch_embed = take_conv(map, "sam_model.patch_embed.proj", SAM_PATCH, 0, true)?;
    let pos_embed = take_arr(map, "sam_model.pos_embed")?;
    let pos_embed_local = resize_sam_position_embedding(&pos_embed)?;
    let mut blocks = Vec::with_capacity(SAM_DEPTH);
    for i in 0..SAM_DEPTH {
        let p = format!("sam_model.blocks.{i}");
        let window_size = if SAM_GLOBAL_ATTN.contains(&i) {
            0
        } else {
            SAM_WINDOW
        };
        let qkv = take_linear(map, &format!("{p}.attn.qkv"))?;
        let proj = take_linear(map, &format!("{p}.attn.proj"))?;
        let rel_pos_h = map.remove(&format!("{p}.attn.rel_pos_h"));
        let rel_pos_w = map.remove(&format!("{p}.attn.rel_pos_w"));
        let rel_pos_h_local = (window_size == 0)
            .then(|| {
                rel_pos_h
                    .as_ref()
                    .map(|value| resize_relative_position(value, 79))
            })
            .flatten();
        let rel_pos_w_local = (window_size == 0)
            .then(|| {
                rel_pos_w
                    .as_ref()
                    .map(|value| resize_relative_position(value, 79))
            })
            .flatten();
        let head_dim = (SAM_WIDTH as usize) / SAM_HEADS;
        let scale = (head_dim as f32).powf(-0.5);
        blocks.push(SamBlock {
            norm1: take_ln(map, &format!("{p}.norm1"), SAM_LN_EPS)?,
            attn: SamAttention {
                qkv,
                proj,
                rel_pos_h,
                rel_pos_w,
                rel_pos_h_local,
                rel_pos_w_local,
                num_heads: SAM_HEADS,
                scale,
            },
            norm2: take_ln(map, &format!("{p}.norm2"), SAM_LN_EPS)?,
            mlp_lin1: take_linear(map, &format!("{p}.mlp.lin1"))?,
            mlp_lin2: take_linear(map, &format!("{p}.mlp.lin2"))?,
            window_size,
        });
    }
    Ok(SamEncoder {
        patch_embed,
        pos_embed,
        pos_embed_local,
        blocks,
        neck0: take_conv(map, "sam_model.neck.0", 1, 0, false)?,
        neck1: take_ln(map, "sam_model.neck.1", SAM_LN_EPS)?,
        neck2: take_conv(map, "sam_model.neck.2", 1, 1, false)?,
        neck3: take_ln(map, "sam_model.neck.3", SAM_LN_EPS)?,
        net_2: take_conv(map, "sam_model.net_2", 2, 1, false)?,
        net_3: take_conv(map, "sam_model.net_3", 2, 1, false)?,
    })
}

fn load_clip(map: &mut HashMap<String, MlxArray>) -> Result<ClipVision, UnlimitedOcrError> {
    let class_embedding = take_arr(map, "vision_model.embeddings.class_embedding")?;
    let patch_embedding = take_conv(map, "vision_model.embeddings.patch_embedding", 14, 0, false)?;
    let position_embedding = take_arr(map, "vision_model.embeddings.position_embedding.weight")?;
    let position_embedding_local = resize_clip_position_embedding(&position_embedding)?;
    // typo preserved from HF checkpoint: pre_layrnorm
    let pre_layernorm = take_ln(map, "vision_model.pre_layrnorm", CLIP_PRE_LN_EPS)?;
    let mut layers = Vec::with_capacity(CLIP_LAYERS);
    let head_dim = (CLIP_WIDTH as usize) / CLIP_HEADS;
    let scale = (head_dim as f32).powf(-0.5);
    for i in 0..CLIP_LAYERS {
        let p = format!("vision_model.transformer.layers.{i}");
        layers.push(ClipEncoderLayer {
            ln1: take_ln(map, &format!("{p}.layer_norm1"), CLIP_LAYER_LN_EPS)?,
            attn: ClipAttention {
                qkv: take_linear(map, &format!("{p}.self_attn.qkv_proj"))?,
                out_proj: take_linear(map, &format!("{p}.self_attn.out_proj"))?,
                num_heads: CLIP_HEADS,
                scale,
            },
            ln2: take_ln(map, &format!("{p}.layer_norm2"), CLIP_LAYER_LN_EPS)?,
            fc1: take_linear(map, &format!("{p}.mlp.fc1"))?,
            fc2: take_linear(map, &format!("{p}.mlp.fc2"))?,
        });
    }
    let _ = CLIP_INTERMEDIATE;
    Ok(ClipVision {
        class_embedding,
        patch_embedding,
        position_embedding,
        position_embedding_local,
        pre_layernorm,
        layers,
    })
}

/// Load dual vision from leftover name_map keys + manifest projector roles.
pub(crate) fn load_unlimited_ocr_vision_weights(
    specs: &[ax_engine_core::NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<Option<UnlimitedOcrVisionWeights>, crate::weights::WeightLoadError> {
    use crate::weights::WeightLoadError;
    use ax_engine_core::NativeTensorRole;

    let has_projector = specs
        .iter()
        .any(|s| s.role == NativeTensorRole::UnlimitedOcrProjector && s.layer_index.is_none());
    let has_sam = name_map.contains_key("sam_model.pos_embed")
        || name_map.contains_key("sam_model.patch_embed.proj.weight");
    if !has_projector && !has_sam {
        return Ok(None);
    }
    if !has_sam {
        return Err(WeightLoadError::RoleMissing(
            "unlimited_ocr vision (sam_model.* leftover tensors)".to_string(),
        ));
    }

    let projector = {
        let role = NativeTensorRole::UnlimitedOcrProjector;
        let spec = specs
            .iter()
            .find(|s| s.role == role && s.layer_index.is_none())
            .ok_or_else(|| WeightLoadError::RoleMissing("unlimited_ocr_projector".into()))?;
        let name = spec.name.clone();
        let weight = name_map
            .remove(&name)
            .ok_or_else(|| WeightLoadError::TensorMissing(name.clone()))?;
        let base = name.strip_suffix(".weight").unwrap_or(&name);
        let scales = name_map.remove(&format!("{base}.scales"));
        let quant_biases = name_map.remove(&format!("{base}.biases"));
        let linear_bias = name_map.remove(&format!("{base}.bias"));
        if spec.source_quantized && scales.is_none() {
            return Err(WeightLoadError::QuantizationMissing(format!(
                "{base}.scales"
            )));
        }
        QuantizedWeight::with_quantization(weight, scales, quant_biases, spec.quantization.as_ref())
            .with_linear_bias(linear_bias)
    };

    let image_newline = {
        let spec = specs
            .iter()
            .find(|s| {
                s.role == NativeTensorRole::UnlimitedOcrImageNewline && s.layer_index.is_none()
            })
            .ok_or_else(|| WeightLoadError::RoleMissing("unlimited_ocr_image_newline".into()))?;
        name_map
            .remove(&spec.name)
            .ok_or_else(|| WeightLoadError::TensorMissing(spec.name.clone()))?
    };
    let view_separator = {
        let spec = specs
            .iter()
            .find(|s| {
                s.role == NativeTensorRole::UnlimitedOcrViewSeparator && s.layer_index.is_none()
            })
            .ok_or_else(|| WeightLoadError::RoleMissing("unlimited_ocr_view_separator".into()))?;
        name_map
            .remove(&spec.name)
            .ok_or_else(|| WeightLoadError::TensorMissing(spec.name.clone()))?
    };

    let sam = load_sam(name_map).map_err(|e| WeightLoadError::InvalidLayer(e.to_string()))?;
    let clip = load_clip(name_map).map_err(|e| WeightLoadError::InvalidLayer(e.to_string()))?;

    // Drop any residual vision keys (e.g. unused CLIP patch path sidecars).
    name_map.retain(|k, _| !k.starts_with("sam_model.") && !k.starts_with("vision_model."));

    Ok(Some(UnlimitedOcrVisionWeights {
        sam,
        clip,
        projector,
        image_newline,
        view_separator,
        image_size: IMAGE_SIZE,
        image_token_id: DEFAULT_IMAGE_TOKEN_ID,
    }))
}

// ---------------------------------------------------------------------------
// Preprocess
// ---------------------------------------------------------------------------

fn normalized_image_batch(images: &[RgbImage]) -> Result<MlxArray, UnlimitedOcrError> {
    let Some(first) = images.first() else {
        return Err(UnlimitedOcrError::BadPixelValues);
    };
    let width = first.width();
    let height = first.height();
    if images
        .iter()
        .any(|candidate| candidate.dimensions() != (width, height))
    {
        return Err(UnlimitedOcrError::BadPixelValues);
    }
    let image_bytes = width as usize * height as usize * 3;
    let mut pixels = Vec::with_capacity(images.len() * image_bytes);
    for source in images {
        pixels.extend(source.as_raw().iter().map(|value| {
            let unit = f32::from(*value) / 255.0;
            2.0 * unit - 1.0
        }));
    }
    let array = MlxArray::from_raw_data(
        pixels.as_ptr().cast(),
        std::mem::size_of_val(pixels.as_slice()),
        &[images.len() as i32, height as i32, width as i32, 3],
        MlxDtype::Float32,
    );
    let output = astype(&array, MlxDtype::Bfloat16, None);
    mlx_sys::eval(&[&output]);
    Ok(output)
}

fn global_letterbox(source: &RgbImage) -> RgbImage {
    let target = IMAGE_SIZE as u32;
    let (width, height) = source.dimensions();
    // Compare the raw dimensions: scaling both sides by `target` overflowed
    // u32 for widths above 2^22 and could flip the branch.
    let (new_width, new_height) = if width > height {
        (
            target,
            (f64::from(height) / f64::from(width) * f64::from(target)).round() as u32,
        )
    } else {
        (
            (f64::from(width) / f64::from(height) * f64::from(target)).round() as u32,
            target,
        )
    };
    let resized = imageops::resize(
        source,
        new_width.max(1),
        new_height.max(1),
        imageops::FilterType::CatmullRom,
    );
    if resized.dimensions() == (target, target) {
        return resized;
    }
    let mut canvas = RgbImage::from_pixel(target, target, Rgb([127, 127, 127]));
    let x = i64::from((target - resized.width()) / 2);
    let y = i64::from((target - resized.height()) / 2);
    imageops::replace(&mut canvas, &resized, x, y);
    canvas
}

/// Prepare the released global view and optional 640px high-resolution tiles.
pub fn preprocess_document_rgb_u8(
    rgb: &[u8],
    width: u32,
    height: u32,
    cropping: bool,
) -> Result<UnlimitedOcrImageViews, UnlimitedOcrError> {
    preprocess_document_rgb_u8_with_tile_limit(rgb, width, height, cropping, 32)
}

pub fn preprocess_document_rgb_u8_with_tile_limit(
    rgb: &[u8],
    width: u32,
    height: u32,
    cropping: bool,
    max_tiles: u32,
) -> Result<UnlimitedOcrImageViews, UnlimitedOcrError> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(UnlimitedOcrError::BadPixelValues)?;
    if width == 0 || height == 0 || rgb.len() != expected {
        return Err(UnlimitedOcrError::BadPixelValues);
    }
    let source =
        RgbImage::from_raw(width, height, rgb.to_vec()).ok_or(UnlimitedOcrError::BadPixelValues)?;
    let global_nhwc = normalized_image_batch(&[global_letterbox(&source)])?;
    let (crop_columns, crop_rows) =
        ax_engine_core::unlimited_ocr::unlimited_ocr_crop_grid_with_limit(
            width, height, cropping, max_tiles,
        );
    let local_nhwc = if crop_columns > 1 || crop_rows > 1 {
        let target_width = UNLIMITED_OCR_LOCAL_TILE_SIZE * crop_columns;
        let target_height = UNLIMITED_OCR_LOCAL_TILE_SIZE * crop_rows;
        let resized = imageops::resize(
            &source,
            target_width,
            target_height,
            imageops::FilterType::CatmullRom,
        );
        let mut tiles = Vec::with_capacity((crop_columns * crop_rows) as usize);
        for row in 0..crop_rows {
            for column in 0..crop_columns {
                tiles.push(
                    imageops::crop_imm(
                        &resized,
                        column * UNLIMITED_OCR_LOCAL_TILE_SIZE,
                        row * UNLIMITED_OCR_LOCAL_TILE_SIZE,
                        UNLIMITED_OCR_LOCAL_TILE_SIZE,
                        UNLIMITED_OCR_LOCAL_TILE_SIZE,
                    )
                    .to_image(),
                );
            }
        }
        Some(normalized_image_batch(&tiles)?)
    } else {
        None
    };
    Ok(UnlimitedOcrImageViews {
        global_nhwc,
        local_nhwc,
        crop_columns,
        crop_rows,
    })
}

// ---------------------------------------------------------------------------
// SAM forward
// ---------------------------------------------------------------------------

fn window_partition(x: &MlxArray, window_size: i32) -> (MlxArray, (i32, i32), (i32, i32)) {
    let shape = x.shape();
    let b = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];
    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let mut x = x.clone();
    if pad_h > 0 || pad_w > 0 {
        let zero = zeros(&[1], x.dtype(), None);
        x = pad(&x, &[1, 2], &[0, 0], &[pad_h, pad_w], &zero, None);
    }
    let hp = h + pad_h;
    let wp = w + pad_w;
    let x = reshape(
        &x,
        &[
            b,
            hp / window_size,
            window_size,
            wp / window_size,
            window_size,
            c,
        ],
        None,
    );
    let x = transpose(&x, &[0, 1, 3, 2, 4, 5], None);
    let windows = reshape(&x, &[-1, window_size, window_size, c], None);
    (windows, (hp, wp), (h, w))
}

fn window_unpartition(
    windows: &MlxArray,
    window_size: i32,
    pad_hw: (i32, i32),
    hw: (i32, i32),
) -> MlxArray {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let c = windows.shape()[3];
    let b = windows.shape()[0] / ((hp / window_size) * (wp / window_size));
    let x = reshape(
        windows,
        &[
            b,
            hp / window_size,
            wp / window_size,
            window_size,
            window_size,
            c,
        ],
        None,
    );
    let x = transpose(&x, &[0, 1, 3, 2, 4, 5], None);
    let x = reshape(&x, &[b, hp, wp, c], None);
    if hp > h || wp > w {
        slice(&x, &[0, 0, 0, 0], &[b, h, w, c], &[1, 1, 1, 1], None)
    } else {
        x
    }
}

fn get_rel_pos(q_size: i32, k_size: i32, rel_pos: &MlxArray) -> MlxArray {
    // Fixed 1024 path keeps rel_pos length == 2*max(q,k)-1; no interpolation.
    let c = rel_pos.shape()[1];
    // relative_coords[i,j] = i * max(k/q,1) - j * max(q/k,1) + (k-1)*max(q/k,1)
    let q_coords = arange(0.0, f64::from(q_size), 1.0, MlxDtype::Float32, None);
    let k_coords = arange(0.0, f64::from(k_size), 1.0, MlxDtype::Float32, None);
    let q_coords = expand_dims(&q_coords, 1, None); // [q,1]
    let k_coords = expand_dims(&k_coords, 0, None); // [1,k]
    let scale_q = (k_size as f32 / q_size as f32).max(1.0);
    let scale_k = (q_size as f32 / k_size as f32).max(1.0);
    let scale_q_arr = mlx_sys::ops::cached_scalar(scale_q, MlxDtype::Float32);
    let scale_k_arr = mlx_sys::ops::cached_scalar(scale_k, MlxDtype::Float32);
    let q_scaled = multiply(&q_coords, &scale_q_arr, None);
    let k_scaled = multiply(&k_coords, &scale_k_arr, None);
    let offset = mlx_sys::ops::cached_scalar((k_size as f32 - 1.0) * scale_k, MlxDtype::Float32);
    let relative = add(&subtract(&q_scaled, &k_scaled, None), &offset, None);
    let indices = astype(&relative, MlxDtype::Int32, None);
    let flat = reshape(&indices, &[-1], None);
    let gathered = take(rel_pos, &flat, 0, None);
    reshape(&gathered, &[q_size, k_size, c], None)
}

fn add_decomposed_rel_pos(
    q: &MlxArray, // [B*heads, H*W, C]
    rel_pos_h: &MlxArray,
    rel_pos_w: &MlxArray,
    q_h: i32,
    q_w: i32,
    k_h: i32,
    k_w: i32,
) -> MlxArray {
    let shape = q.shape();
    let b = shape[0];
    let dim = shape[2];
    let r_q = reshape(q, &[b, q_h, q_w, dim], None);
    let rh = get_rel_pos(q_h, k_h, rel_pos_h); // [q_h, k_h, C]
    let rw = get_rel_pos(q_w, k_w, rel_pos_w); // [q_w, k_w, C]

    // rel_h: einsum bhwc,hkc -> bhwk via batched matmul
    // r_q [B,q_h,q_w,C] @ Rh^T [q_h,C,k_h] (broadcast B)
    let rh_t = transpose(&rh, &[0, 2, 1], None); // [q_h, C, k_h]
    let rh_t = reshape(&rh_t, &[1, q_h, dim, k_h], None);
    let rel_h = matmul(&r_q, &rh_t, None); // [B, q_h, q_w, k_h]

    // rel_w: einsum bhwc,wkc -> bhwk
    // For each (h,w): sum_c r_q[b,h,w,c] * Rw[w,k,c]
    // reshape r_q to [B, q_h, q_w, C], Rw [q_w, k_w, C] → [1,1,q_w,C,k_w]
    let rw_t = transpose(&rw, &[0, 2, 1], None); // [q_w, C, k_w]
    let rw_t = reshape(&rw_t, &[1, 1, q_w, dim, k_w], None);
    // matmul on last two of r_q as [..., 1, C] @ [..., C, k_w]
    let r_q_m = reshape(&r_q, &[b, q_h, q_w, 1, dim], None);
    let rel_w = matmul(&r_q_m, &rw_t, None); // [B, q_h, q_w, 1, k_w]
    let rel_w = reshape(&rel_w, &[b, q_h, q_w, k_w], None);

    // Build additive bias [B, H*W, H*W] then we'll reshape with heads outside.
    // rel_h[..., None] + rel_w[..., None, :]
    let rel_h = reshape(&rel_h, &[b, q_h, q_w, k_h, 1], None);
    let rel_w = reshape(&rel_w, &[b, q_h, q_w, 1, k_w], None);
    let bias = add(&rel_h, &rel_w, None); // [B, q_h, q_w, k_h, k_w]
    reshape(&bias, &[b, q_h * q_w, k_h * k_w], None)
}

fn sam_attention(attn: &SamAttention, x: &MlxArray) -> MlxArray {
    let shape = x.shape();
    let b = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];
    let heads = attn.num_heads as i32;
    let head_dim = c / heads;

    let qkv = attn.qkv.forward(x);
    // [B, H, W, 3*C] → [B, H*W, 3, heads, head_dim] → [3, B, heads, H*W, head_dim]
    let qkv = reshape(&qkv, &[b, h * w, 3, heads, head_dim], None);
    let qkv = transpose(&qkv, &[2, 0, 3, 1, 4], None);
    // [3, B*heads, H*W, head_dim]
    let qkv = reshape(&qkv, &[3, b * heads, h * w, head_dim], None);
    let parts = split(&qkv, 3, 0, None);
    let q = reshape(&parts[0], &[b * heads, h * w, head_dim], None);
    let k = reshape(&parts[1], &[b * heads, h * w, head_dim], None);
    let v = reshape(&parts[2], &[b * heads, h * w, head_dim], None);

    let q4 = reshape(&q, &[b, heads, h * w, head_dim], None);
    let k4 = reshape(&k, &[b, heads, h * w, head_dim], None);
    let v4 = reshape(&v, &[b, heads, h * w, head_dim], None);

    let (relative_h, relative_w) = if h == LOCAL_IMAGE_SIZE / SAM_PATCH {
        (
            attn.rel_pos_h_local.as_ref().or(attn.rel_pos_h.as_ref()),
            attn.rel_pos_w_local.as_ref().or(attn.rel_pos_w.as_ref()),
        )
    } else {
        (attn.rel_pos_h.as_ref(), attn.rel_pos_w.as_ref())
    };
    let out = if let (Some(rh), Some(rw)) = (relative_h, relative_w) {
        let bias = add_decomposed_rel_pos(&q, rh, rw, h, w, h, w); // [B*heads, HW, HW]
        let bias = reshape(&bias, &[b, heads, h * w, h * w], None);
        scaled_dot_product_attention_with_mask(
            &q4,
            &k4,
            &v4,
            attn.scale,
            ScaledDotProductAttentionMask::Array(&bias),
            None,
        )
    } else {
        scaled_dot_product_attention(&q4, &k4, &v4, attn.scale, false, None)
    };

    let out = reshape(&out, &[b, heads, h, w, head_dim], None);
    let out = transpose(&out, &[0, 2, 3, 1, 4], None);
    let out = reshape(&out, &[b, h, w, c], None);
    attn.proj.forward(&out)
}

fn sam_block_forward(block: &SamBlock, x: &MlxArray) -> MlxArray {
    let shortcut = x.clone();
    let mut x = block.norm1.forward(x);
    let (mut pad_hw, mut hw) = ((0, 0), (0, 0));
    if block.window_size > 0 {
        let shape = x.shape();
        hw = (shape[1], shape[2]);
        let (windows, phw, _) = window_partition(&x, block.window_size);
        pad_hw = phw;
        x = windows;
    }
    x = sam_attention(&block.attn, &x);
    if block.window_size > 0 {
        x = window_unpartition(&x, block.window_size, pad_hw, hw);
    }
    x = add(&shortcut, &x, None);
    let mlp = block.mlp_lin2.forward(&gelu(
        &block.mlp_lin1.forward(&block.norm2.forward(&x)),
        None,
    ));
    add(&x, &mlp, None)
}

fn sam_forward(sam: &SamEncoder, x: &MlxArray) -> MlxArray {
    // x: NHWC
    let mut x = sam.patch_embed.forward(x);
    let position = if x.shape()[1] == LOCAL_IMAGE_SIZE / SAM_PATCH {
        &sam.pos_embed_local
    } else {
        &sam.pos_embed
    };
    x = add(&x, position, None);
    for block in &sam.blocks {
        x = sam_block_forward(block, &x);
    }
    x = sam.neck0.forward(&x);
    x = sam.neck1.forward(&x);
    x = sam.neck2.forward(&x);
    x = sam.neck3.forward(&x);
    x = sam.net_2.forward(&x);
    sam.net_3.forward(&x)
}

// ---------------------------------------------------------------------------
// CLIP forward
// ---------------------------------------------------------------------------

fn clip_attention(attn: &ClipAttention, x: &MlxArray) -> MlxArray {
    let shape = x.shape();
    let b = shape[0];
    let l = shape[1];
    let d = shape[2];
    let heads = attn.num_heads as i32;
    let head_dim = d / heads;
    let qkv = attn.qkv.forward(x);
    let parts = split(&qkv, 3, -1, None);
    let q = reshape(&parts[0], &[b, l, heads, head_dim], None);
    let k = reshape(&parts[1], &[b, l, heads, head_dim], None);
    let v = reshape(&parts[2], &[b, l, heads, head_dim], None);
    let q = transpose(&q, &[0, 2, 1, 3], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = transpose(&v, &[0, 2, 1, 3], None);
    let out = scaled_dot_product_attention(&q, &k, &v, attn.scale, false, None);
    let out = transpose(&out, &[0, 2, 1, 3], None);
    let out = reshape(&out, &[b, l, d], None);
    attn.out_proj.forward(&out)
}

/// The CLIP MLP activation (`deepencoder.py` `quick_gelu`); SAM keeps the
/// erf GELU of its `nn.GELU` MLP blocks.
fn quick_gelu(x: &MlxArray) -> MlxArray {
    let scale = mlx_sys::ops::cached_scalar(QUICK_GELU_SCALE, x.dtype());
    multiply(x, &sigmoid(&multiply(x, &scale, None), None), None)
}

fn clip_layer_forward(layer: &ClipEncoderLayer, x: &MlxArray) -> MlxArray {
    let y = clip_attention(&layer.attn, &layer.ln1.forward(x));
    let x = add(x, &y, None);
    let y = layer
        .fc2
        .forward(&quick_gelu(&layer.fc1.forward(&layer.ln2.forward(&x))));
    add(&x, &y, None)
}

fn clip_forward(clip: &ClipVision, patch_embeds: &MlxArray) -> MlxArray {
    // patch_embeds: [B, H, W, C] from SAM
    let shape = patch_embeds.shape();
    let b = shape[0];
    let patch_flat = flatten(patch_embeds, 1, 2, None); // [B, HW, C]
    let hw = patch_flat.shape()[1];
    let class_embeds = reshape(&clip.class_embedding, &[1, 1, CLIP_WIDTH], None);
    let class_embeds = broadcast_to(&class_embeds, &[b, 1, CLIP_WIDTH], None);
    let class_embeds = astype(&class_embeds, patch_flat.dtype(), None);
    let mut embeddings = concatenate(&[&class_embeds, &patch_flat], 1, None);
    // position embedding [num_pos, C] — for 16×16+1 == 257 matches CLIP-L/14@224
    let num_pos = embeddings.shape()[1];
    let position = if num_pos == clip.position_embedding_local.shape()[0] {
        &clip.position_embedding_local
    } else {
        &clip.position_embedding
    };
    let pos = reshape(position, &[1, num_pos, CLIP_WIDTH], None);
    let pos = broadcast_to(&pos, &[b, num_pos, CLIP_WIDTH], None);
    let pos = astype(&pos, embeddings.dtype(), None);
    embeddings = add(&embeddings, &pos, None);
    let _ = hw;
    let mut x = clip.pre_layernorm.forward(&embeddings);
    for layer in &clip.layers {
        x = clip_layer_forward(layer, &x);
    }
    x
}

// ---------------------------------------------------------------------------
// Feature assembly (base / global-only)
// ---------------------------------------------------------------------------

/// Run SAM+CLIP+projector for a single base-resolution image (NHWC BF16).
///
/// Returns `[n_soft, hidden]` embeddings ready to inject into image-token slots.
pub(crate) fn encode_base_image(
    vision: &UnlimitedOcrVisionWeights,
    image_nhwc: &MlxArray,
) -> Result<MlxArray, UnlimitedOcrError> {
    let sam_out = sam_forward(&vision.sam, image_nhwc); // [B, 16, 16, 1024]
    let clip_out = clip_forward(&vision.clip, &sam_out); // [B, 257, 1024]
    // CLIP[:, 1:] + SAM.flatten(1,2) along last dim
    let clip_patches = slice(
        &clip_out,
        &[0, 1, 0],
        &[
            clip_out.shape()[0],
            clip_out.shape()[1],
            clip_out.shape()[2],
        ],
        &[1, 1, 1],
        None,
    );
    let sam_flat = flatten(&sam_out, 1, 2, None); // [B, 256, 1024]
    let fused = concatenate(&[&clip_patches, &sam_flat], -1, None); // [B, 256, 2048]
    let projected = qw(&fused, &vision.projector); // [B, 256, 1280]
    let b = projected.shape()[0];
    if b != 1 {
        return Err(UnlimitedOcrError::Assembly(
            "batch size > 1 not supported in base path".into(),
        ));
    }
    let features = reshape(
        &projected,
        &[projected.shape()[1], projected.shape()[2]],
        None,
    ); // [256, 1280]
    let hw = features.shape()[0];
    let n_dim = features.shape()[1];
    let h = (hw as f32).sqrt() as i32;
    let w = h;
    if h * w != hw {
        return Err(UnlimitedOcrError::Assembly(format!(
            "projected token count {hw} is not a square grid"
        )));
    }
    let grid = reshape(&features, &[h, w, n_dim], None);
    let newline = reshape(&vision.image_newline, &[1, 1, n_dim], None);
    let newline = broadcast_to(&newline, &[h, 1, n_dim], None);
    let with_nl = concatenate(&[&grid, &newline], 1, None); // [h, w+1, dim]
    let flat = reshape(&with_nl, &[h * (w + 1), n_dim], None);
    let sep = reshape(&vision.view_separator, &[1, n_dim], None);
    Ok(concatenate(&[&flat, &sep], 0, None))
}

/// Encode the global view and, when present, reassemble local tiles into the
/// same two-dimensional feature canvas used by the released processor.
pub(crate) fn encode_document_image(
    vision: &UnlimitedOcrVisionWeights,
    views: &UnlimitedOcrImageViews,
) -> Result<MlxArray, UnlimitedOcrError> {
    let global = encode_base_image(vision, &views.global_nhwc)?;
    let Some(local_images) = views.local_nhwc.as_ref() else {
        return Ok(global);
    };
    let expected_tiles = (views.crop_columns * views.crop_rows) as i32;
    if expected_tiles <= 1 || local_images.shape()[0] != expected_tiles {
        return Err(UnlimitedOcrError::Assembly(format!(
            "local tile batch {} does not match {}x{} crop grid",
            local_images.shape()[0],
            views.crop_columns,
            views.crop_rows
        )));
    }

    let sam = sam_forward(&vision.sam, local_images);
    let clip = clip_forward(&vision.clip, &sam);
    let clip_patches = slice(
        &clip,
        &[0, 1, 0],
        &[clip.shape()[0], clip.shape()[1], clip.shape()[2]],
        &[1, 1, 1],
        None,
    );
    let sam_flat = flatten(&sam, 1, 2, None);
    let fused = concatenate(&[&clip_patches, &sam_flat], -1, None);
    let projected = qw(&fused, &vision.projector);
    let feature_count = projected.shape()[1];
    let dimensions = projected.shape()[2];
    let side = (feature_count as f32).sqrt() as i32;
    if side * side != feature_count || side != UNLIMITED_OCR_LOCAL_QUERY_GRID as i32 {
        return Err(UnlimitedOcrError::Assembly(format!(
            "local projected feature grid is {feature_count}, expected {}x{}",
            UNLIMITED_OCR_LOCAL_QUERY_GRID, UNLIMITED_OCR_LOCAL_QUERY_GRID
        )));
    }

    let rows = views.crop_rows as i32;
    let columns = views.crop_columns as i32;
    let local = reshape(&projected, &[rows, columns, side, side, dimensions], None);
    let local = transpose(&local, &[0, 2, 1, 3, 4], None);
    let canvas_rows = rows * side;
    let canvas_columns = columns * side;
    let local = reshape(&local, &[canvas_rows, canvas_columns, dimensions], None);
    let newline = reshape(&vision.image_newline, &[1, 1, dimensions], None);
    let newline = broadcast_to(&newline, &[canvas_rows, 1, dimensions], None);
    let local = concatenate(&[&local, &newline], 1, None);
    let local = reshape(
        &local,
        &[canvas_rows * (canvas_columns + 1), dimensions],
        None,
    );
    Ok(concatenate(&[&local, &global], 0, None))
}

// ---------------------------------------------------------------------------
// Embed injection + prefill helper
// ---------------------------------------------------------------------------

fn overwrite_image_token_positions(
    hidden: MlxArray,
    features: &MlxArray,
    token_ids: &[u32],
    image_token_id: u32,
    hidden_size: usize,
) -> Result<MlxArray, UnlimitedOcrError> {
    let positions: Vec<usize> = token_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == image_token_id).then_some(i))
        .collect();
    let n_feat = features.shape()[0] as usize;
    if positions.len() != n_feat {
        return Err(UnlimitedOcrError::SoftTokenMismatch {
            prompt_tokens: positions.len(),
            feature_tokens: n_feat,
        });
    }
    if positions.is_empty() {
        return Ok(hidden);
    }
    // Contiguous span is the common case (processor expands one block of <image>).
    let contiguous = positions.windows(2).all(|w| w[1] == w[0] + 1);
    if contiguous {
        let start = positions[0];
        let end = start + positions.len();
        let update = reshape(features, &[1, n_feat as i32, hidden_size as i32], None);
        return Ok(slice_update(
            &hidden,
            &update,
            &[0, start as i32, 0],
            &[1, end as i32, hidden_size as i32],
            &[1, 1, 1],
            None,
        ));
    }
    // Scatter one token at a time for non-contiguous masks.
    let mut hidden = hidden;
    for (src, &pos) in positions.iter().enumerate() {
        let row = slice(
            features,
            &[src as i32, 0],
            &[(src + 1) as i32, hidden_size as i32],
            &[1, 1],
            None,
        );
        let update = reshape(&row, &[1, 1, hidden_size as i32], None);
        hidden = slice_update(
            &hidden,
            &update,
            &[0, pos as i32, 0],
            &[1, (pos + 1) as i32, hidden_size as i32],
            &[1, 1, 1],
            None,
        );
    }
    Ok(hidden)
}

/// Embed prompt tokens and overwrite image-token slots with vision features.
pub(crate) fn build_embeddings_with_images(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    views: &[UnlimitedOcrImageViews],
    image_token_id: u32,
) -> Result<MlxArray, UnlimitedOcrError> {
    let vision = weights
        .unlimited_ocr_vision
        .as_ref()
        .ok_or(UnlimitedOcrError::MissingVisionWeights)?;
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden_pub(&hidden, scale);
    }
    let mut pages = Vec::with_capacity(views.len());
    for view in views {
        let features = encode_document_image(vision, view)?;
        pages.push(features);
    }
    let refs: Vec<_> = pages.iter().collect();
    let features = concatenate(&refs, 0, None);
    let features = astype(&features, hidden.dtype(), None);
    // The request carries its own image token id (the client resolves it from
    // the checkpoint's config or tokenizer); the weight struct's constant is
    // only a default for callers without a request.
    overwrite_image_token_positions(
        hidden,
        &features,
        token_ids,
        image_token_id,
        cfg.hidden_size,
    )
}

/// Build prompt token ids for Free OCR with the correct soft-token count.
///
/// Layout: `[bos?] + image_token * N + encode("\nFree OCR.")` — caller supplies
/// text token ids (without image placeholders).
pub fn build_free_ocr_token_ids(
    text_prefix_ids: &[u32],
    text_suffix_ids: &[u32],
    image_token_id: u32,
    soft_token_count: usize,
    prepend_bos: Option<u32>,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(
        prepend_bos.is_some() as usize
            + text_prefix_ids.len()
            + soft_token_count
            + text_suffix_ids.len(),
    );
    if let Some(bos) = prepend_bos {
        out.push(bos);
    }
    out.extend_from_slice(text_prefix_ids);
    out.extend(std::iter::repeat_n(image_token_id, soft_token_count));
    out.extend_from_slice(text_suffix_ids);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_embedding_resizers_reject_unexpected_grids() {
        let sam = mlx_sys::zeros(&[1, 32, 32, 4], MlxDtype::Float32, None);
        assert!(resize_sam_position_embedding(&sam).is_err());
        let clip = mlx_sys::zeros(&[101, 4], MlxDtype::Float32, None);
        assert!(resize_clip_position_embedding(&clip).is_err());
        let sam = mlx_sys::zeros(&[1, 64, 64, 4], MlxDtype::Float32, None);
        assert_eq!(
            resize_sam_position_embedding(&sam).unwrap().shape(),
            vec![1, 40, 40, 4]
        );
        let clip = mlx_sys::zeros(&[257, 4], MlxDtype::Float32, None);
        assert_eq!(
            resize_clip_position_embedding(&clip).unwrap().shape(),
            vec![101, 4]
        );
    }

    #[test]
    fn base_soft_token_count_1024() {
        assert_eq!(default_base_soft_token_count(), 273);
        assert_eq!(base_soft_token_count(1024, 16, 4), 273);
    }

    #[test]
    fn layer_norm_eps_constants_match_reference_towers() {
        // `deepencoder.py`: SAM blocks and LayerNorm2d use 1e-6; the CLIP
        // tower uses `layernorm_epsilon = pre_layernorm_epsilon = 1e-5`.
        assert!((SAM_LN_EPS - 1.0e-6).abs() < f32::EPSILON);
        assert!((CLIP_PRE_LN_EPS - 1.0e-5).abs() < f32::EPSILON);
        assert!((CLIP_LAYER_LN_EPS - 1.0e-5).abs() < f32::EPSILON);
    }

    // Expected values below come from torch 2.x `F.interpolate` / `sigmoid`
    // in float32, called exactly as `deepencoder.py` calls them, on the
    // deterministic inputs each test builds.
    const TORCH_REL_POS_127_TO_79: [f32; 79] = [
        0.1098579,
        0.6465828,
        0.947692,
        0.9404259,
        0.5964866,
        0.05486142,
        -0.5124924,
        -0.8911532,
        -0.9783576,
        -0.7201429,
        -0.2177591,
        0.3633894,
        0.8098358,
        0.9889616,
        0.824126,
        0.3743901,
        -0.2026329,
        -0.7059373,
        -0.9720635,
        -0.905504,
        -0.5204864,
        0.03645635,
        0.5822644,
        0.9282284,
        0.9619405,
        0.6520658,
        0.1299254,
        -0.4421607,
        -0.858742,
        -0.9917601,
        -0.7655405,
        -0.2919471,
        0.2894211,
        0.765573,
        0.9939975,
        0.8578072,
        0.4451947,
        -0.1281931,
        -0.651319,
        -0.9684294,
        -0.9263296,
        -0.5855127,
        -0.03713155,
        0.5191366,
        0.9109223,
        0.9692057,
        0.7091194,
        0.2020251,
        -0.3726559,
        -0.8283497,
        -0.9852178,
        -0.8126929,
        -0.3619572,
        0.2158909,
        0.7231225,
        0.9738674,
        0.8934631,
        0.5124952,
        -0.05312027,
        -0.5982479,
        -0.9353912,
        -0.9492854,
        -0.6494423,
        -0.1112054,
        0.4572375,
        0.8707539,
        0.9786844,
        0.7676132,
        0.2726015,
        -0.3040251,
        -0.7816319,
        -0.9808996,
        -0.8615681,
        -0.426645,
        0.1428434,
        0.6703718,
        0.9558984,
        0.9310724,
        0.569112,
    ];

    const TORCH_CLIP_POS_16_TO_10: [f32; 100] = [
        0.5729402,
        0.2175829,
        -0.2837713,
        -0.3436106,
        0.08729513,
        0.538159,
        0.4875561,
        0.002273378,
        -0.3669515,
        -0.2314915,
        1.005344,
        0.6499867,
        0.1486325,
        0.08879324,
        0.5196989,
        0.9705626,
        0.9199598,
        0.4346772,
        0.06545231,
        0.2009123,
        1.343032,
        0.9876751,
        0.486321,
        0.4264817,
        0.8573872,
        1.308251,
        1.257648,
        0.7723657,
        0.4031408,
        0.5386007,
        1.47027,
        1.114913,
        0.6135585,
        0.5537193,
        0.9846249,
        1.435489,
        1.384886,
        0.8996032,
        0.5303783,
        0.6658382,
        1.379007,
        1.023649,
        0.5222952,
        0.4624559,
        0.8933615,
        1.344225,
        1.293622,
        0.8083399,
        0.439115,
        0.5745749,
        1.080522,
        0.7251652,
        0.2238111,
        0.1639718,
        0.5948774,
        1.045741,
        0.9951384,
        0.5098558,
        0.1406308,
        0.2760908,
        0.64342,
        0.2880628,
        -0.2132914,
        -0.2731307,
        0.157775,
        0.6086388,
        0.5580359,
        0.07275323,
        -0.2964716,
        -0.1610117,
        0.1697001,
        -0.1856571,
        -0.6870114,
        -0.7468506,
        -0.3159449,
        0.134919,
        0.08431608,
        -0.4009667,
        -0.7701915,
        -0.6347316,
        -0.235455,
        -0.5908122,
        -1.092166,
        -1.152006,
        -0.7210999,
        -0.2702362,
        -0.3208391,
        -0.8061217,
        -1.175346,
        -1.039887,
        -0.4753974,
        -0.8307546,
        -1.332109,
        -1.391948,
        -0.9610423,
        -0.5101786,
        -0.5607815,
        -1.046064,
        -1.415289,
        -1.279829,
    ];

    const TORCH_QUICK_GELU: [f32; 6] = [
        -0.01807131,
        -0.1542042,
        -0.09880035,
        0.0,
        0.3503884,
        1.935659,
    ];

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tolerance, "{label}[{index}]: {a} vs {e}");
        }
    }

    #[test]
    fn relative_position_resize_matches_torch_linear_half_pixel() {
        // The tile path resizes SAM's 127-entry rel_pos to 79 (`get_rel_pos`).
        let input: Vec<f32> = (0..127).map(|i| (0.37 * i as f32).sin()).collect();
        let position = array_from_f32(&input, &[127, 1], MlxDtype::Float32);
        let resized = array_f32_values(&resize_relative_position(&position, 79));
        assert_close(&resized, &TORCH_REL_POS_127_TO_79, 1e-5, "rel_pos");
    }

    #[test]
    fn clip_position_resize_matches_torch_antialiased_bicubic() {
        // The tile path resizes CLIP's 16x16 grid to 10x10 (`get_abs_pos`);
        // the class row passes through unchanged.
        let mut input = vec![7.0_f32];
        for y in 0..16 {
            for x in 0..16 {
                input.push((0.3 * y as f32).sin() + 0.5 * (0.7 * x as f32).cos());
            }
        }
        let position = array_from_f32(&input, &[257, 1], MlxDtype::Float32);
        let resized = array_f32_values(&resize_clip_position_embedding(&position));
        assert_eq!(resized[0], 7.0);
        assert_close(&resized[1..], &TORCH_CLIP_POS_16_TO_10, 1e-5, "clip_pos");
    }

    #[test]
    fn clip_mlp_activation_is_quick_gelu() {
        let x = [-3.0_f32, -1.0, -0.25, 0.0, 0.5, 2.0];
        let input = array_from_f32(&x, &[6], MlxDtype::Float32);
        let output = array_f32_values(&quick_gelu(&input));
        assert_close(&output, &TORCH_QUICK_GELU, 1e-6, "quick_gelu");
    }
}
