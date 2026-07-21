//! Native Unlimited-OCR dual vision tower (SAM-ViT-B + CLIP-L) + projector.
//!
//! Mirrors mlx-vlm `deepseekocr/sam.py`, `deepseekocr/vision.py`, and the
//! base-resolution (global-only) path in `unlimited_ocr/unlimitedocr.py`.
//! Vision weights stay BF16 and are pulled from leftover safetensors keys
//! after language load; the mxfp8 projector / image_newline / view_separator
//! come from manifest roles.

use std::collections::HashMap;

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
/// OpenCLIP / Unlimited-OCR CLIP tower LayerNorm epsilon (`layernorm_epsilon=1e-5`).
const CLIP_LN_EPS: f32 = 1.0e-5;
/// OpenCLIP QuickGELU coefficient: `x * sigmoid(1.702 * x)`.
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

/// OpenCLIP QuickGELU used by Unlimited-OCR CLIP MLP (not standard GELU).
///
/// `quick_gelu(x) = x * sigmoid(1.702 * x)` — see baidu/Unlimited-OCR
/// `deepencoder.NoTPFeedForward`.
fn quick_gelu(x: &MlxArray) -> MlxArray {
    let scale = mlx_sys::ops::cached_scalar(QUICK_GELU_SCALE, x.dtype());
    let scaled = multiply(x, &scale, None);
    multiply(x, &sigmoid(&scaled, None), None)
}

#[derive(Clone)]
struct SamAttention {
    qkv: DenseLinear,
    proj: DenseLinear,
    rel_pos_h: Option<MlxArray>,
    rel_pos_w: Option<MlxArray>,
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

/// Soft-token count for base-resolution global-only mode.
///
/// `num_queries = ceil((image_size / patch_size) / downsample_ratio)`
/// tokens = `(num_queries + 1) * num_queries + 1`  (grid + per-row newline + separator)
pub fn base_soft_token_count(image_size: i32, patch_size: i32, downsample_ratio: i32) -> usize {
    let grid = (image_size / patch_size) as f32 / downsample_ratio as f32;
    let num_queries = grid.ceil() as usize;
    (num_queries + 1) * num_queries + 1
}

/// Default base-mode soft token count for 1024 / SAM patch 16 / downsample 4 → 273.
pub fn default_base_soft_token_count() -> usize {
    base_soft_token_count(IMAGE_SIZE, SAM_PATCH, DOWNSAMPLE_RATIO)
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
        let head_dim = (SAM_WIDTH as usize) / SAM_HEADS;
        let scale = (head_dim as f32).powf(-0.5);
        blocks.push(SamBlock {
            norm1: take_ln(map, &format!("{p}.norm1"), SAM_LN_EPS)?,
            attn: SamAttention {
                qkv,
                proj,
                rel_pos_h,
                rel_pos_w,
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
    // typo preserved from HF checkpoint: pre_layrnorm
    let pre_layernorm = take_ln(map, "vision_model.pre_layrnorm", CLIP_LN_EPS)?;
    let mut layers = Vec::with_capacity(CLIP_LAYERS);
    let head_dim = (CLIP_WIDTH as usize) / CLIP_HEADS;
    let scale = (head_dim as f32).powf(-0.5);
    for i in 0..CLIP_LAYERS {
        let p = format!("vision_model.transformer.layers.{i}");
        layers.push(ClipEncoderLayer {
            ln1: take_ln(map, &format!("{p}.layer_norm1"), CLIP_LN_EPS)?,
            attn: ClipAttention {
                qkv: take_linear(map, &format!("{p}.self_attn.qkv_proj"))?,
                out_proj: take_linear(map, &format!("{p}.self_attn.out_proj"))?,
                num_heads: CLIP_HEADS,
                scale,
            },
            ln2: take_ln(map, &format!("{p}.layer_norm2"), CLIP_LN_EPS)?,
            fc1: take_linear(map, &format!("{p}.mlp.fc1"))?,
            fc2: take_linear(map, &format!("{p}.mlp.fc2"))?,
        });
    }
    let _ = CLIP_INTERMEDIATE;
    Ok(ClipVision {
        class_embedding,
        patch_embedding,
        position_embedding,
        pre_layernorm,
        layers,
    })
}

/// Load dual vision from leftover name_map keys + manifest projector roles.
pub fn load_unlimited_ocr_vision_weights(
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

/// Letterbox RGB u8 image into 1024×1024, normalize mean/std 0.5, return NHWC f32.
///
/// `rgb` is row-major H×W×3.
pub fn preprocess_rgb_u8(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<MlxArray, UnlimitedOcrError> {
    if width == 0 || height == 0 || rgb.len() < (width as usize) * (height as usize) * 3 {
        return Err(UnlimitedOcrError::BadPixelValues);
    }
    let target = IMAGE_SIZE as u32;
    // Letterbox (ImageOps.pad): scale to fit, center on mean-colored canvas.
    let scale = (target as f32 / width as f32).min(target as f32 / height as f32);
    let new_w = ((width as f32) * scale).round().max(1.0) as u32;
    let new_h = ((height as f32) * scale).round().max(1.0) as u32;
    let pad_x = ((target - new_w) / 2) as i32;
    let pad_y = ((target - new_h) / 2) as i32;
    // mean 0.5 → pad value 127 (matches int(0.5*255))
    let mut canvas = vec![127u8; (target as usize) * (target as usize) * 3];
    // nearest-neighbor resize into the centered box
    for y in 0..new_h {
        let src_y = ((y as f32 + 0.5) / scale - 0.5)
            .floor()
            .clamp(0.0, (height - 1) as f32) as u32;
        for x in 0..new_w {
            let src_x = ((x as f32 + 0.5) / scale - 0.5)
                .floor()
                .clamp(0.0, (width - 1) as f32) as u32;
            let si = ((src_y * width + src_x) * 3) as usize;
            let di = (((pad_y as u32 + y) * target + (pad_x as u32 + x)) * 3) as usize;
            canvas[di] = rgb[si];
            canvas[di + 1] = rgb[si + 1];
            canvas[di + 2] = rgb[si + 2];
        }
    }
    // u8 → f32 /255, normalize (x-0.5)/0.5 = 2x-1, keep NHWC
    let mut pixels = Vec::with_capacity(canvas.len());
    for &v in &canvas {
        let x = (v as f32) / 255.0;
        pixels.push(2.0 * x - 1.0);
    }
    let arr = MlxArray::from_raw_data(
        pixels.as_ptr().cast(),
        std::mem::size_of_val(pixels.as_slice()),
        &[1, IMAGE_SIZE, IMAGE_SIZE, 3],
        MlxDtype::Float32,
    );
    // mlx_array_new_data copies host bytes; cast + eval materializes BF16.
    let out = astype(&arr, MlxDtype::Bfloat16, None);
    mlx_sys::eval(&[&out]);
    drop(pixels);
    Ok(out)
}

/// Build NHWC BF16 pixels from planar CHW f32 already normalized (mean/std 0.5).
pub fn pixels_from_chw_f32(chw: &[f32]) -> Result<MlxArray, UnlimitedOcrError> {
    let expected = (3 * IMAGE_SIZE * IMAGE_SIZE) as usize;
    if chw.len() != expected {
        return Err(UnlimitedOcrError::BadPixelValues);
    }
    // CHW → NHWC
    let mut nhwc = vec![0f32; expected];
    let hw = (IMAGE_SIZE * IMAGE_SIZE) as usize;
    for y in 0..IMAGE_SIZE as usize {
        for x in 0..IMAGE_SIZE as usize {
            let spatial = y * IMAGE_SIZE as usize + x;
            for c in 0..3 {
                nhwc[spatial * 3 + c] = chw[c * hw + spatial];
            }
        }
    }
    let arr = MlxArray::from_raw_data(
        nhwc.as_ptr().cast(),
        std::mem::size_of_val(nhwc.as_slice()),
        &[1, IMAGE_SIZE, IMAGE_SIZE, 3],
        MlxDtype::Float32,
    );
    let out = astype(&arr, MlxDtype::Bfloat16, None);
    mlx_sys::eval(&[&out]);
    drop(nhwc);
    Ok(out)
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

    let out = if let (Some(rh), Some(rw)) = (&attn.rel_pos_h, &attn.rel_pos_w) {
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
    // pos embed same spatial size for 1024 base
    x = add(&x, &sam.pos_embed, None);
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

fn clip_layer_forward(layer: &ClipEncoderLayer, x: &MlxArray) -> MlxArray {
    let y = clip_attention(&layer.attn, &layer.ln1.forward(x));
    let x = add(x, &y, None);
    // OpenCLIP MLP uses QuickGELU, not the erf-based GELU used by SAM.
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
    let pos = if clip.position_embedding.shape()[0] == num_pos {
        let pos = reshape(&clip.position_embedding, &[1, num_pos, CLIP_WIDTH], None);
        broadcast_to(&pos, &[b, num_pos, CLIP_WIDTH], None)
    } else {
        // Should not happen for base mode; fall back to first num_pos rows.
        let pos = slice(
            &clip.position_embedding,
            &[0, 0],
            &[num_pos, CLIP_WIDTH],
            &[1, 1],
            None,
        );
        let pos = reshape(&pos, &[1, num_pos, CLIP_WIDTH], None);
        broadcast_to(&pos, &[b, num_pos, CLIP_WIDTH], None)
    };
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
pub fn encode_base_image(
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
pub fn build_embeddings_with_image(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    image_nhwc: &MlxArray,
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
    let features = encode_base_image(vision, image_nhwc)?;
    let features = astype(&features, hidden.dtype(), None);
    overwrite_image_token_positions(
        hidden,
        &features,
        token_ids,
        vision.image_token_id,
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
    fn base_soft_token_count_1024() {
        assert_eq!(default_base_soft_token_count(), 273);
        assert_eq!(base_soft_token_count(1024, 16, 4), 273);
    }

    /// Regression: CLIP MLP must use OpenCLIP QuickGELU, not erf-GELU.
    ///
    /// Reference: `quick_gelu(x) = x * sigmoid(1.702 * x)` from
    /// baidu/Unlimited-OCR `deepencoder.NoTPFeedForward`. Standard GELU
    /// diverges enough at moderate magnitudes to corrupt vision features.
    #[test]
    fn quick_gelu_matches_openclip_formula() {
        let xs = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5];
        let arr = MlxArray::from_raw_data(
            xs.as_ptr().cast(),
            std::mem::size_of_val(xs.as_slice()),
            &[xs.len() as i32],
            MlxDtype::Float32,
        );
        let out = quick_gelu(&arr);
        mlx_sys::eval(&[&out]);
        let got = out.data_f32();
        assert_eq!(got.len(), xs.len());
        for (i, &x) in xs.iter().enumerate() {
            let expected = x * (1.0 / (1.0 + (-1.702 * x).exp()));
            let g = got[i];
            let err = (g - expected).abs();
            assert!(
                err < 1e-5,
                "quick_gelu({x}) = {g}, expected {expected}, err={err}"
            );
            // Approximate GELU (tanh form) must diverge for |x| >= 1 so this
            // test would not pass if the activation were standard GELU.
            let gelu_ref = {
                let t = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + t.tanh())
            };
            if x.abs() >= 1.0 {
                assert!(
                    (expected - gelu_ref).abs() > 1e-3,
                    "test point x={x} does not separate quick_gelu from approx gelu"
                );
            }
        }
    }

    #[test]
    fn layer_norm_eps_constants_match_reference_towers() {
        // SAM uses 1e-6; CLIP OpenCLIP tower uses 1e-5.
        assert!((SAM_LN_EPS - 1.0e-6).abs() < f32::EPSILON);
        assert!((CLIP_LN_EPS - 1.0e-5).abs() < f32::EPSILON);
        assert!(CLIP_LN_EPS > SAM_LN_EPS);
    }
}
