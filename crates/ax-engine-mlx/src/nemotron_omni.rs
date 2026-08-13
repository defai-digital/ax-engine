//! Native Nemotron H Nano Omni media prefill.
//!
//! Vision uses the checkpoint's RADIO v2.5-H tower followed by the v2 pixel
//! shuffle and quantized `mlp1` projector. The Nemotron-H language backbone
//! remains on AX's existing hybrid Mamba/attention/MoE implementation.

use std::collections::HashMap;

use ax_engine_core::nemotron_omni::NemotronOmniRuntimeInputs;
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, broadcast_to, concatenate, gelu, layer_norm, matmul, maximum,
    multiply, reshape, rms_norm, scaled_dot_product_attention, slice, take, transpose, zeros,
};
use serde_json::Value;
use thiserror::Error;

use crate::model::shared::qw;
use crate::model::{ModelConfig, embed_tokens};
use crate::nemotron_omni_audio::NemotronOmniAudioWeights;
use crate::qwen3_vl::scatter_vision_into_text;
use crate::weights::{ModelWeights, QuantizedWeight, WeightLoadError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NemotronOmniError {
    #[error("Nemotron H Nano Omni media input requires loaded media weights")]
    MissingMediaWeights,
    #[error("Nemotron H Nano Omni geometry invalid: {0}")]
    InvalidGeometry(String),
    #[error("Nemotron H Nano Omni media scatter failed: {0}")]
    Scatter(String),
    #[error("Nemotron H Nano Omni audio path is unavailable: {0}")]
    AudioUnavailable(String),
}

#[derive(Clone, Debug)]
struct VisionConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    patch_size: usize,
}

#[derive(Clone)]
struct DenseLinear {
    weight: MlxArray,
    bias: Option<MlxArray>,
}

impl DenseLinear {
    fn forward(&self, input: &MlxArray) -> MlxArray {
        let input = if input.dtype() == self.weight.dtype() {
            input.clone()
        } else {
            astype(input, self.weight.dtype(), None)
        };
        let output = matmul(&input, &transpose(&self.weight, &[1, 0], None), None);
        self.bias
            .as_ref()
            .map_or(output.clone(), |bias| add(&output, bias, None))
    }
}

#[derive(Clone)]
struct LayerNormWeights {
    weight: MlxArray,
    bias: Option<MlxArray>,
    eps: f32,
}

impl LayerNormWeights {
    fn forward(&self, input: &MlxArray) -> MlxArray {
        let zero;
        let bias = match self.bias.as_ref() {
            Some(bias) => bias,
            None => {
                zero = zeros(&self.weight.shape(), self.weight.dtype(), None);
                &zero
            }
        };
        layer_norm(input, &self.weight, bias, self.eps, None)
    }
}

#[derive(Clone)]
struct RadioAttention {
    qkv: DenseLinear,
    proj: DenseLinear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl RadioAttention {
    fn forward(&self, input: &MlxArray) -> MlxArray {
        let shape = input.shape();
        let batch = shape[0];
        let seq = shape[1];
        let hidden = shape[2];
        let qkv = self.qkv.forward(input);
        let qkv = reshape(
            &qkv,
            &[batch, seq, 3, self.num_heads as i32, self.head_dim as i32],
            None,
        );
        let qkv = transpose(&qkv, &[2, 0, 3, 1, 4], None);
        let q = slice(
            &qkv,
            &[0, 0, 0, 0, 0],
            &[1, batch, self.num_heads as i32, seq, self.head_dim as i32],
            &[1, 1, 1, 1, 1],
            None,
        );
        let k = slice(
            &qkv,
            &[1, 0, 0, 0, 0],
            &[2, batch, self.num_heads as i32, seq, self.head_dim as i32],
            &[1, 1, 1, 1, 1],
            None,
        );
        let v = slice(
            &qkv,
            &[2, 0, 0, 0, 0],
            &[3, batch, self.num_heads as i32, seq, self.head_dim as i32],
            &[1, 1, 1, 1, 1],
            None,
        );
        let q = reshape(
            &q,
            &[batch, self.num_heads as i32, seq, self.head_dim as i32],
            None,
        );
        let k = reshape(
            &k,
            &[batch, self.num_heads as i32, seq, self.head_dim as i32],
            None,
        );
        let v = reshape(
            &v,
            &[batch, self.num_heads as i32, seq, self.head_dim as i32],
            None,
        );
        let output = scaled_dot_product_attention(&q, &k, &v, self.scale, false, None);
        let output = transpose(&output, &[0, 2, 1, 3], None);
        self.proj
            .forward(&reshape(&output, &[batch, seq, hidden], None))
    }
}

#[derive(Clone)]
struct RadioBlock {
    norm1: LayerNormWeights,
    attn: RadioAttention,
    norm2: LayerNormWeights,
    fc1: DenseLinear,
    fc2: DenseLinear,
}

impl RadioBlock {
    fn forward(&self, input: &MlxArray) -> MlxArray {
        let attn = self.attn.forward(&self.norm1.forward(input));
        let hidden = add(input, &attn, None);
        let mlp = self
            .fc2
            .forward(&gelu(&self.fc1.forward(&self.norm2.forward(&hidden)), None));
        add(&hidden, &mlp, None)
    }
}

#[derive(Clone)]
struct RadioVision {
    config: VisionConfig,
    cls_token: MlxArray,
    patch_embedder: DenseLinear,
    pos_embed: MlxArray,
    pos_side: usize,
    blocks: Vec<RadioBlock>,
}

impl RadioVision {
    fn forward(
        &self,
        pixels: &MlxArray,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<MlxArray, NemotronOmniError> {
        let expected = [
            1,
            3,
            (grid_h * self.config.patch_size) as i32,
            (grid_w * self.config.patch_size) as i32,
        ];
        if pixels.shape() != expected {
            return Err(NemotronOmniError::InvalidGeometry(format!(
                "pixel tensor {:?}, expected {expected:?}",
                pixels.shape()
            )));
        }
        // The released outer model calls RADIO's
        // `make_preprocessor_external()`: channel normalization belongs to
        // the image processor and must not be applied again in the tower.
        let pixels = astype(pixels, self.patch_embedder.weight.dtype(), None);
        let patches = reshape(
            &pixels,
            &[
                1,
                3,
                grid_h as i32,
                self.config.patch_size as i32,
                grid_w as i32,
                self.config.patch_size as i32,
            ],
            None,
        );
        let patches = transpose(&patches, &[0, 2, 4, 1, 3, 5], None);
        let patches = reshape(
            &patches,
            &[
                1,
                (grid_h * grid_w) as i32,
                (3 * self.config.patch_size * self.config.patch_size) as i32,
            ],
            None,
        );
        let mut hidden = self.patch_embedder.forward(&patches);
        let positions = radio_position_embeddings(
            &self.pos_embed,
            self.pos_side,
            grid_h,
            grid_w,
            self.config.hidden_size,
        )?;
        hidden = add(&hidden, &astype(&positions, hidden.dtype(), None), None);

        let cls = reshape(
            &self.cls_token,
            &[1, self.cls_token.shape()[0], self.config.hidden_size as i32],
            None,
        );
        let cls = broadcast_to(
            &astype(&cls, hidden.dtype(), None),
            &[1, self.cls_token.shape()[0], self.config.hidden_size as i32],
            None,
        );
        hidden = concatenate(&[&cls, &hidden], 1, None);
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        let skip = self.cls_token.shape()[0];
        let total = hidden.shape()[1];
        Ok(slice(
            &hidden,
            &[0, skip, 0],
            &[1, total, self.config.hidden_size as i32],
            &[1, 1, 1],
            None,
        ))
    }
}

#[derive(Clone)]
struct VisionProjector {
    norm: MlxArray,
    fc1: QuantizedWeight,
    fc2: QuantizedWeight,
    downsample_factor: usize,
    ps_version_v1: bool,
}

impl VisionProjector {
    fn forward(
        &self,
        features: &MlxArray,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<MlxArray, NemotronOmniError> {
        let factor = self.downsample_factor;
        if !grid_h.is_multiple_of(factor) || !grid_w.is_multiple_of(factor) {
            return Err(NemotronOmniError::InvalidGeometry(format!(
                "RADIO patch grid {grid_h}x{grid_w} is not divisible by {factor}"
            )));
        }
        let channels = features.shape()[2];
        let grid = reshape(features, &[1, grid_h as i32, grid_w as i32, channels], None);
        let shuffled = if factor == 1 {
            grid
        } else {
            let stage1 = reshape(
                &grid,
                &[
                    1,
                    grid_h as i32,
                    (grid_w / factor) as i32,
                    channels * factor as i32,
                ],
                None,
            );
            let stage2 = transpose(&stage1, &[0, 2, 1, 3], None);
            let stage3 = reshape(
                &stage2,
                &[
                    1,
                    (grid_w / factor) as i32,
                    (grid_h / factor) as i32,
                    channels * (factor * factor) as i32,
                ],
                None,
            );
            if self.ps_version_v1 {
                stage3
            } else {
                transpose(&stage3, &[0, 2, 1, 3], None)
            }
        };
        let flattened = reshape(
            &shuffled,
            &[
                1,
                ((grid_h / factor) * (grid_w / factor)) as i32,
                shuffled.shape()[3],
            ],
            None,
        );
        let normed = rms_norm(&flattened, Some(&self.norm), 1.0e-5, None);
        let hidden = qw(&normed, &self.fc1);
        let zero = zeros(&[], hidden.dtype(), None);
        let relu = maximum(&hidden, &zero, None);
        Ok(qw(&multiply(&relu, &relu, None), &self.fc2))
    }
}

/// Loaded RADIO tower and multimodal projectors.
#[derive(Clone)]
pub struct NemotronOmniWeights {
    vision: RadioVision,
    vision_projector: VisionProjector,
    audio: Option<NemotronOmniAudioWeights>,
}

impl NemotronOmniWeights {
    fn image_features(
        &self,
        pixels: &MlxArray,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<MlxArray, NemotronOmniError> {
        let radio = self.vision.forward(pixels, grid_h, grid_w)?;
        self.vision_projector.forward(&radio, grid_h, grid_w)
    }
}

pub fn load_nemotron_omni_weights(
    name_map: &mut HashMap<String, MlxArray>,
    config_json: Option<&Value>,
) -> Result<Option<NemotronOmniWeights>, WeightLoadError> {
    let is_omni = config_json
        .and_then(|config| config.get("model_type"))
        .and_then(Value::as_str)
        .is_some_and(|model_type| {
            model_type == "NemotronH_Nano_Omni_Reasoning_V3"
                || model_type.eq_ignore_ascii_case("nemotron_h_nano_omni")
        });
    if !is_omni {
        return Ok(None);
    }
    let config_json = config_json.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "Nemotron H Nano Omni checkpoint has no readable config.json".to_string(),
        )
    })?;
    let vision_config = parse_vision_config(config_json)?;
    let prefix = "vision_model.radio_model";
    // These tensors remain in the RADIO checkpoint, but the outer Nemotron
    // model explicitly disables the internal input conditioner.
    let _ = take_optional(name_map, &format!("{prefix}.input_conditioner.norm_mean"));
    let _ = take_optional(name_map, &format!("{prefix}.input_conditioner.norm_std"));
    let cls_token = take_required(
        name_map,
        &format!("{prefix}.model.patch_generator.cls_token.token"),
    )?;
    let patch_embedder = load_dense_linear(
        name_map,
        &format!("{prefix}.model.patch_generator.embedder"),
    )?;
    let pos_embed = take_required(
        name_map,
        &format!("{prefix}.model.patch_generator.pos_embed"),
    )?;
    let pos_tokens = match pos_embed.shape().as_slice() {
        [1, tokens, hidden] if *hidden == vision_config.hidden_size as i32 => *tokens as usize,
        shape => {
            return Err(WeightLoadError::InvalidLayer(format!(
                "Nemotron RADIO pos_embed has shape {shape:?}"
            )));
        }
    };
    let pos_side = integer_square_side(pos_tokens).ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!(
            "Nemotron RADIO pos_embed token count {pos_tokens} is not square"
        ))
    })?;

    let mut blocks = Vec::with_capacity(vision_config.num_hidden_layers);
    for layer_index in 0..vision_config.num_hidden_layers {
        let base = format!("{prefix}.model.blocks.{layer_index}");
        let head_dim = vision_config.hidden_size / vision_config.num_attention_heads;
        blocks.push(RadioBlock {
            norm1: load_layer_norm(name_map, &format!("{base}.norm1"), 1.0e-6)?,
            attn: RadioAttention {
                qkv: load_dense_linear(name_map, &format!("{base}.attn.qkv"))?,
                proj: load_dense_linear(name_map, &format!("{base}.attn.proj"))?,
                num_heads: vision_config.num_attention_heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
            },
            norm2: load_layer_norm(name_map, &format!("{base}.norm2"), 1.0e-6)?,
            fc1: load_dense_linear(name_map, &format!("{base}.mlp.fc1"))?,
            fc2: load_dense_linear(name_map, &format!("{base}.mlp.fc2"))?,
        });
    }

    let downsample_ratio = config_json
        .get("downsample_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.5) as f32;
    if !downsample_ratio.is_finite() || downsample_ratio <= 0.0 || downsample_ratio > 1.0 {
        return Err(WeightLoadError::InvalidLayer(format!(
            "Nemotron Omni downsample_ratio {downsample_ratio} is invalid"
        )));
    }
    let downsample_factor = (1.0 / downsample_ratio).round() as usize;
    // PyTorch/HF serializes nn.Sequential as `mlp1.0`, while mlx-vlm's
    // sanitized tree uses `mlp1.layers.0`. Accept both reviewed layouts.
    let mlp_prefix = if name_map.contains_key("mlp1.0.weight") {
        "mlp1"
    } else {
        "mlp1.layers"
    };
    let vision_projector = VisionProjector {
        norm: take_required(name_map, &format!("{mlp_prefix}.0.weight"))?,
        fc1: load_quantized_linear(name_map, &format!("{mlp_prefix}.1"), 64, 4)?,
        fc2: load_quantized_linear(name_map, &format!("{mlp_prefix}.3"), 64, 4)?,
        downsample_factor: downsample_factor.max(1),
        ps_version_v1: config_json.get("ps_version").and_then(Value::as_str) == Some("v1"),
    };
    let has_audio_weights = name_map
        .keys()
        .any(|name| name.starts_with("sound_encoder."))
        && name_map
            .keys()
            .any(|name| name.starts_with("sound_projection."));
    let audio = if config_json.get("sound_config").is_some() {
        if !has_audio_weights {
            return Err(WeightLoadError::InvalidLayer(
                "Nemotron Omni sound_config is present but sound encoder/projector weights are missing"
                    .to_string(),
            ));
        }
        Some(NemotronOmniAudioWeights::load(name_map, config_json)?)
    } else {
        None
    };

    Ok(Some(NemotronOmniWeights {
        vision: RadioVision {
            config: vision_config,
            cls_token,
            patch_embedder,
            pos_embed,
            pos_side,
            blocks,
        },
        vision_projector,
        audio,
    }))
}

pub(crate) fn build_omni_prefill_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    inputs: &NemotronOmniRuntimeInputs,
) -> Result<MlxArray, NemotronOmniError> {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    let media = weights
        .nemotron_omni
        .as_ref()
        .ok_or(NemotronOmniError::MissingMediaWeights)?;
    for image in &inputs.images {
        let grid_h = image.height as usize / image.patch_size as usize;
        let grid_w = image.width as usize / image.patch_size as usize;
        let pixels = MlxArray::from_raw_data(
            image.pixel_values.as_ptr().cast(),
            std::mem::size_of_val(image.pixel_values.as_slice()),
            &[1, 3, image.height as i32, image.width as i32],
            MlxDtype::Float32,
        );
        let features = media.image_features(&pixels, grid_h, grid_w)?;
        let produced = features.shape()[1] as usize;
        if produced != image.soft_token_count as usize
            || features.shape()[2] != cfg.hidden_size as i32
        {
            return Err(NemotronOmniError::InvalidGeometry(format!(
                "vision projector produced [{produced}, {}], expected [{}, {}]",
                features.shape()[2],
                image.soft_token_count,
                cfg.hidden_size
            )));
        }
        let features = astype(&features, hidden.dtype(), None);
        let end = image.placeholder_index.saturating_add(produced);
        let positions: Vec<usize> = (image.placeholder_index..end).collect();
        hidden = scatter_vision_into_text(&hidden, &features, &positions)
            .map_err(|error| NemotronOmniError::Scatter(error.to_string()))?;
    }
    for audio in &inputs.audios {
        let audio_weights = media.audio.as_ref().ok_or_else(|| {
            NemotronOmniError::AudioUnavailable(
                "checkpoint has no sound_encoder/sound_projection graph".to_string(),
            )
        })?;
        let features = audio_weights
            .forward(&audio.samples, audio.sample_rate)
            .map_err(NemotronOmniError::AudioUnavailable)?;
        let produced = features.shape()[1] as usize;
        if produced != audio.soft_token_count as usize
            || features.shape()[2] != cfg.hidden_size as i32
        {
            return Err(NemotronOmniError::InvalidGeometry(format!(
                "audio projector produced [{produced}, {}], expected [{}, {}]",
                features.shape()[2],
                audio.soft_token_count,
                cfg.hidden_size
            )));
        }
        let features = astype(&features, hidden.dtype(), None);
        let end = audio.placeholder_index.saturating_add(produced);
        let positions: Vec<usize> = (audio.placeholder_index..end).collect();
        hidden = scatter_vision_into_text(&hidden, &features, &positions)
            .map_err(|error| NemotronOmniError::Scatter(error.to_string()))?;
    }
    Ok(hidden)
}

fn parse_vision_config(config: &Value) -> Result<VisionConfig, WeightLoadError> {
    let vision = config.get("vision_config").unwrap_or(&Value::Null);
    let parsed = VisionConfig {
        hidden_size: optional_usize(vision, "hidden_size", 1280),
        num_hidden_layers: optional_usize(vision, "num_hidden_layers", 32),
        num_attention_heads: optional_usize(vision, "num_attention_heads", 16),
        patch_size: optional_usize(vision, "patch_size", 16),
    };
    if parsed.hidden_size == 0
        || parsed.num_hidden_layers == 0
        || parsed.num_attention_heads == 0
        || !parsed
            .hidden_size
            .is_multiple_of(parsed.num_attention_heads)
        || parsed.patch_size == 0
    {
        return Err(WeightLoadError::InvalidLayer(format!(
            "invalid Nemotron RADIO config: {parsed:?}"
        )));
    }
    Ok(parsed)
}

fn optional_usize(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
}

fn load_dense_linear(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
) -> Result<DenseLinear, WeightLoadError> {
    let weight = take_required(map, &format!("{base}.weight"))?;
    if weight.dtype() == MlxDtype::Uint32 {
        return Err(WeightLoadError::InvalidLayer(format!(
            "Nemotron RADIO tensor {base} is quantized; reviewed checkpoints keep RADIO dense"
        )));
    }
    Ok(DenseLinear {
        weight,
        bias: take_optional(map, &format!("{base}.bias")),
    })
}

fn load_layer_norm(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
    eps: f32,
) -> Result<LayerNormWeights, WeightLoadError> {
    Ok(LayerNormWeights {
        weight: take_required(map, &format!("{base}.weight"))?,
        bias: take_optional(map, &format!("{base}.bias")),
        eps,
    })
}

pub(super) fn load_quantized_linear(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
    group_size: i32,
    bits: i32,
) -> Result<QuantizedWeight, WeightLoadError> {
    let weight = take_required(map, &format!("{base}.weight"))?;
    let scales = take_optional(map, &format!("{base}.scales"));
    let biases = take_optional(map, &format!("{base}.biases"));
    if weight.dtype() == MlxDtype::Uint32 && scales.is_none() {
        return Err(WeightLoadError::QuantizationMissing(base.to_string()));
    }
    Ok(QuantizedWeight {
        weight,
        scales,
        biases,
        group_size,
        bits,
        mode: "affine".to_string(),
        linear_bias: take_optional(map, &format!("{base}.bias")),
        decode_weight_t: None,
    })
}

pub(super) fn take_required(
    map: &mut HashMap<String, MlxArray>,
    name: &str,
) -> Result<MlxArray, WeightLoadError> {
    map.remove(name)
        .ok_or_else(|| WeightLoadError::TensorMissing(name.to_string()))
}

pub(super) fn take_optional(map: &mut HashMap<String, MlxArray>, name: &str) -> Option<MlxArray> {
    map.remove(name)
}

fn integer_square_side(tokens: usize) -> Option<usize> {
    let side = (tokens as f64).sqrt() as usize;
    (side.saturating_mul(side) == tokens).then_some(side)
}

fn radio_position_embeddings(
    table: &MlxArray,
    source_side: usize,
    grid_h: usize,
    grid_w: usize,
    hidden_size: usize,
) -> Result<MlxArray, NemotronOmniError> {
    if grid_h == 0 || grid_w == 0 || source_side == 0 {
        return Err(NemotronOmniError::InvalidGeometry(
            "position grid dimensions must be positive".to_string(),
        ));
    }
    let table = reshape(
        table,
        &[source_side as i32, source_side as i32, hidden_size as i32],
        None,
    );
    // RADIO CPE mode: resize the square table to the larger runtime axis,
    // then take the top-left window for the shorter axis.
    let max_side = grid_h.max(grid_w);
    let resized = bilinear_resize_positions(
        &table,
        source_side,
        source_side,
        max_side,
        max_side,
        hidden_size,
    );
    let cropped = if grid_h == max_side && grid_w == max_side {
        resized
    } else {
        slice(
            &resized,
            &[0, 0, 0],
            &[grid_h as i32, grid_w as i32, hidden_size as i32],
            &[1, 1, 1],
            None,
        )
    };
    Ok(reshape(
        &cropped,
        &[1, (grid_h * grid_w) as i32, hidden_size as i32],
        None,
    ))
}

fn bilinear_resize_positions(
    table: &MlxArray,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    hidden: usize,
) -> MlxArray {
    if src_h == dst_h && src_w == dst_w {
        return table.clone();
    }
    let mut indices = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut weights = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let scale_h = src_h as f32 / dst_h as f32;
    let scale_w = src_w as f32 / dst_w as f32;
    for row in 0..dst_h {
        let source_row = (row as f32 + 0.5) * scale_h - 0.5;
        let row_floor_raw = source_row.floor() as i32;
        let row_floor = row_floor_raw.clamp(0, src_h as i32 - 1);
        let row_ceil = (row_floor_raw + 1).clamp(0, src_h as i32 - 1);
        let dy = source_row - row_floor as f32;
        for column in 0..dst_w {
            let source_column = (column as f32 + 0.5) * scale_w - 0.5;
            let column_floor_raw = source_column.floor() as i32;
            let column_floor = column_floor_raw.clamp(0, src_w as i32 - 1);
            let column_ceil = (column_floor_raw + 1).clamp(0, src_w as i32 - 1);
            let dx = source_column - column_floor as f32;
            for (slot, (r, c, weight)) in [
                (row_floor, column_floor, (1.0 - dy) * (1.0 - dx)),
                (row_floor, column_ceil, (1.0 - dy) * dx),
                (row_ceil, column_floor, dy * (1.0 - dx)),
                (row_ceil, column_ceil, dy * dx),
            ]
            .into_iter()
            .enumerate()
            {
                indices[slot].push((r as usize * src_w + c as usize) as u32);
                weights[slot].push(weight);
            }
        }
    }
    let flat = reshape(table, &[(src_h * src_w) as i32, hidden as i32], None);
    let mut sum: Option<MlxArray> = None;
    for slot in 0..4 {
        let index = u32_array(&indices[slot], &[(dst_h * dst_w) as i32]);
        let corner = take(&flat, &index, 0, None);
        let weight = f32_array(&weights[slot], &[(dst_h * dst_w) as i32, 1]);
        let weighted = multiply(&corner, &astype(&weight, corner.dtype(), None), None);
        sum = Some(match sum {
            Some(accumulator) => add(&accumulator, &weighted, None),
            None => weighted,
        });
    }
    reshape(
        &sum.expect("four bilinear corners"),
        &[dst_h as i32, dst_w as i32, hidden as i32],
        None,
    )
}

fn u32_array(values: &[u32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Uint32,
    )
}

fn f32_array(values: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Float32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_side_requires_exact_square() {
        assert_eq!(integer_square_side(16_384), Some(128));
        assert_eq!(integer_square_side(15), None);
    }

    #[test]
    fn released_vision_config_uses_radio_h_defaults() {
        let config = serde_json::json!({
            "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
            "vision_config": {"patch_size": 16}
        });
        let parsed = parse_vision_config(&config).expect("config");
        assert_eq!(parsed.hidden_size, 1280);
        assert_eq!(parsed.num_hidden_layers, 32);
        assert_eq!(parsed.num_attention_heads, 16);
    }
}
