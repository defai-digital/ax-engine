//! MiniCPM-V 4.6 native vision prefill.
//!
//! The released model combines a Qwen3.5 text backbone with a SigLIP-style
//! vision tower. The visual graph is:
//!
//! `patch conv + learned positions -> 27 transformer blocks`
//! `-> VitMerger after block 6 -> post LayerNorm -> pixel-shuffle MLP merger`.
//!
//! Text decode remains on AX's normal Qwen3.5 hybrid path. This module owns
//! only the image tower and replacement of the `<unk>` placeholder span.

use std::collections::HashMap;

use ax_engine_core::minicpm_v::MiniCpmV46RuntimeInputs;
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, conv2d, gelu_approx, layer_norm, matmul, multiply, reshape,
    scaled_dot_product_attention, sum_axis, take, transpose, zeros,
};
use serde_json::Value;
use thiserror::Error;

use crate::model::{ModelConfig, embed_tokens};
use crate::qwen3_vl::scatter_vision_into_text;
use crate::weights::{ModelWeights, WeightLoadError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MiniCpmV46Error {
    #[error("MiniCPM-V 4.6 image input requires a loaded vision tower")]
    MissingVisionWeights,
    #[error("MiniCPM-V 4.6 geometry invalid: {0}")]
    InvalidGeometry(String),
    #[error("MiniCPM-V 4.6 image scatter failed: {0}")]
    Scatter(String),
}

#[derive(Clone, Debug)]
pub struct MiniCpmV46VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f32,
    pub window_kernel_size: [usize; 2],
    pub insert_layer_id: usize,
    pub merge_kernel_size: [usize; 2],
    pub merger_times: usize,
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
struct Attention {
    q_proj: DenseLinear,
    k_proj: DenseLinear,
    v_proj: DenseLinear,
    out_proj: DenseLinear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Attention {
    fn forward(&self, queries: &MlxArray, keys: &MlxArray, values: &MlxArray) -> MlxArray {
        let q_shape = queries.shape();
        let kv_shape = keys.shape();
        let batch = q_shape[0];
        let q_len = q_shape[1];
        let kv_len = kv_shape[1];
        let hidden = q_shape[2];

        let q = self.q_proj.forward(queries);
        let k = self.k_proj.forward(keys);
        let v = self.v_proj.forward(values);
        let q = reshape(
            &q,
            &[batch, q_len, self.num_heads as i32, self.head_dim as i32],
            None,
        );
        let k = reshape(
            &k,
            &[batch, kv_len, self.num_heads as i32, self.head_dim as i32],
            None,
        );
        let v = reshape(
            &v,
            &[batch, kv_len, self.num_heads as i32, self.head_dim as i32],
            None,
        );
        let q = transpose(&q, &[0, 2, 1, 3], None);
        let k = transpose(&k, &[0, 2, 1, 3], None);
        let v = transpose(&v, &[0, 2, 1, 3], None);
        let context = scaled_dot_product_attention(&q, &k, &v, self.scale, false, None);
        let context = transpose(&context, &[0, 2, 1, 3], None);
        let context = reshape(&context, &[batch, q_len, hidden], None);
        self.out_proj.forward(&context)
    }
}

#[derive(Clone)]
struct VisionLayer {
    self_attn: Attention,
    layer_norm1: LayerNormWeights,
    layer_norm2: LayerNormWeights,
    fc1: DenseLinear,
    fc2: DenseLinear,
}

impl VisionLayer {
    fn forward(&self, input: &MlxArray) -> MlxArray {
        let normed = self.layer_norm1.forward(input);
        let attention = self.self_attn.forward(&normed, &normed, &normed);
        let hidden = add(input, &attention, None);
        let normed = self.layer_norm2.forward(&hidden);
        let mlp = self
            .fc2
            .forward(&gelu_approx(&self.fc1.forward(&normed), None));
        add(&hidden, &mlp, None)
    }
}

#[derive(Clone)]
struct VitMerger {
    pre_norm: LayerNormWeights,
    self_attn: Attention,
    layer_norm1: LayerNormWeights,
    linear_1: DenseLinear,
    linear_2: DenseLinear,
    group_h: usize,
    group_w: usize,
    hidden_size: usize,
}

impl VitMerger {
    fn forward(
        &self,
        input: &MlxArray,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<(MlxArray, usize, usize), MiniCpmV46Error> {
        if !grid_h.is_multiple_of(self.group_h) || !grid_w.is_multiple_of(self.group_w) {
            return Err(MiniCpmV46Error::InvalidGeometry(format!(
                "patch grid {grid_h}x{grid_w} is not divisible by VitMerger {}x{}",
                self.group_h, self.group_w
            )));
        }
        let merged_h = grid_h / self.group_h;
        let merged_w = grid_w / self.group_w;
        let windows = merged_h.saturating_mul(merged_w);
        let group_tokens = self.group_h.saturating_mul(self.group_w);

        let grouped = reshape(
            input,
            &[
                merged_h as i32,
                self.group_h as i32,
                merged_w as i32,
                self.group_w as i32,
                self.hidden_size as i32,
            ],
            None,
        );
        let grouped = transpose(&grouped, &[0, 2, 1, 3, 4], None);
        let grouped = reshape(
            &grouped,
            &[windows as i32, group_tokens as i32, self.hidden_size as i32],
            None,
        );

        let normed = self.layer_norm1.forward(&grouped);
        let attention = self.self_attn.forward(&normed, &normed, &normed);
        let grouped = add(&grouped, &attention, None);
        let residual = sum_axis(&grouped, 1, false, None);
        let residual = multiply_scalar(&residual, 1.0 / group_tokens as f32);

        let flattened = reshape(
            &grouped,
            &[windows as i32, (group_tokens * self.hidden_size) as i32],
            None,
        );
        let merged = self.pre_norm.forward(&flattened);
        let merged = self.linear_1.forward(&merged);
        let merged = gelu_approx(&merged, None);
        let merged = self.linear_2.forward(&merged);
        Ok((add(&merged, &residual, None), merged_h, merged_w))
    }
}

#[derive(Clone)]
struct MergerBlock {
    pre_norm: LayerNormWeights,
    linear_1: DenseLinear,
    linear_2: DenseLinear,
}

impl MergerBlock {
    fn forward(&self, input: &MlxArray) -> MlxArray {
        let hidden = self.pre_norm.forward(input);
        let hidden = self.linear_1.forward(&hidden);
        self.linear_2.forward(&gelu_approx(&hidden, None))
    }
}

#[derive(Clone)]
struct FinalMerger {
    blocks: Vec<MergerBlock>,
    merge_h: usize,
    merge_w: usize,
}

impl FinalMerger {
    fn forward(
        &self,
        input: &MlxArray,
        mut grid_h: usize,
        mut grid_w: usize,
    ) -> Result<MlxArray, MiniCpmV46Error> {
        let mut hidden = input.clone();
        for block in &self.blocks {
            if !grid_h.is_multiple_of(self.merge_h) || !grid_w.is_multiple_of(self.merge_w) {
                return Err(MiniCpmV46Error::InvalidGeometry(format!(
                    "post-ViT grid {grid_h}x{grid_w} is not divisible by merger {}x{}",
                    self.merge_h, self.merge_w
                )));
            }
            let merged_h = grid_h / self.merge_h;
            let merged_w = grid_w / self.merge_w;
            let inner = hidden.shape()[1];
            let grouped = reshape(&hidden, &[grid_h as i32, grid_w as i32, inner], None);
            let grouped = reshape(
                &grouped,
                &[
                    merged_h as i32,
                    self.merge_h as i32,
                    merged_w as i32,
                    self.merge_w as i32,
                    inner,
                ],
                None,
            );
            let grouped = transpose(&grouped, &[0, 2, 1, 3, 4], None);
            let grouped = reshape(
                &grouped,
                &[
                    (merged_h * merged_w) as i32,
                    (self.merge_h * self.merge_w) as i32 * inner,
                ],
                None,
            );
            hidden = block.forward(&grouped);
            grid_h = merged_h;
            grid_w = merged_w;
        }
        Ok(hidden)
    }
}

/// Loaded MiniCPM-V 4.6 SigLIP + merger weights.
#[derive(Clone)]
pub struct MiniCpmV46VisionWeights {
    pub config: MiniCpmV46VisionConfig,
    patch_embedding: MlxArray,
    patch_embedding_bias: Option<MlxArray>,
    position_embedding: MlxArray,
    layers: Vec<VisionLayer>,
    post_layernorm: LayerNormWeights,
    vit_merger: VitMerger,
    merger: FinalMerger,
}

impl MiniCpmV46VisionWeights {
    fn forward(
        &self,
        pixels: &MlxArray,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<MlxArray, MiniCpmV46Error> {
        let shape = pixels.shape();
        let expected = [
            1,
            (grid_h * self.config.patch_size) as i32,
            (grid_w * self.config.patch_size) as i32,
            self.config.num_channels as i32,
        ];
        if shape != expected {
            return Err(MiniCpmV46Error::InvalidGeometry(format!(
                "pixel tensor {shape:?}, expected {expected:?}"
            )));
        }

        let pixels = astype(pixels, self.patch_embedding.dtype(), None);
        let mut hidden = conv2d(
            &pixels,
            &self.patch_embedding,
            self.config.patch_size as i32,
            0,
            1,
            1,
            None,
        );
        if let Some(bias) = &self.patch_embedding_bias {
            hidden = add(&hidden, bias, None);
        }
        hidden = reshape(
            &hidden,
            &[1, (grid_h * grid_w) as i32, self.config.hidden_size as i32],
            None,
        );
        let position_ids = dynamic_position_ids(
            self.config.image_size / self.config.patch_size,
            grid_h,
            grid_w,
        );
        let position_ids = u32_array(&position_ids, &[position_ids.len() as i32]);
        let positions = take(&self.position_embedding, &position_ids, 0, None);
        let positions = reshape(
            &positions,
            &[1, (grid_h * grid_w) as i32, self.config.hidden_size as i32],
            None,
        );
        hidden = add(&hidden, &positions, None);

        let mut current_h = grid_h;
        let mut current_w = grid_w;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden);
            if layer_index == self.config.insert_layer_id {
                let flat = reshape(
                    &hidden,
                    &[
                        (current_h * current_w) as i32,
                        self.config.hidden_size as i32,
                    ],
                    None,
                );
                let (merged, merged_h, merged_w) =
                    self.vit_merger.forward(&flat, current_h, current_w)?;
                current_h = merged_h;
                current_w = merged_w;
                hidden = reshape(
                    &merged,
                    &[
                        1,
                        (current_h * current_w) as i32,
                        self.config.hidden_size as i32,
                    ],
                    None,
                );
            }
        }
        hidden = self.post_layernorm.forward(&hidden);
        let hidden = reshape(
            &hidden,
            &[
                (current_h * current_w) as i32,
                self.config.hidden_size as i32,
            ],
            None,
        );
        self.merger.forward(&hidden, current_h, current_w)
    }
}

pub fn load_minicpm_v46_vision_weights(
    name_map: &mut HashMap<String, MlxArray>,
    config_json: Option<&Value>,
) -> Result<Option<MiniCpmV46VisionWeights>, WeightLoadError> {
    let is_minicpm = config_json
        .and_then(|config| config.get("model_type"))
        .and_then(Value::as_str)
        .is_some_and(|model_type| {
            matches!(
                model_type,
                "minicpmv4_6" | "minicpm_v4_6" | "minicpm-v4_6" | "minicpm-v-4.6"
            )
        });
    if !is_minicpm {
        return Ok(None);
    }
    if !has_any(
        name_map,
        &["vision_tower.embeddings.patch_embedding.weight"],
    ) {
        return Ok(None);
    }
    let config_json = config_json.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "MiniCPM-V 4.6 checkpoint is missing a readable config.json".to_string(),
        )
    })?;
    let config = parse_config(config_json)?;
    if config.insert_layer_id >= config.num_hidden_layers {
        return Err(WeightLoadError::InvalidLayer(format!(
            "MiniCPM-V 4.6 insert_layer_id {} exceeds {} vision layers",
            config.insert_layer_id, config.num_hidden_layers
        )));
    }

    let patch_embedding_raw = take_any(
        name_map,
        &["vision_tower.embeddings.patch_embedding.weight"],
    )?;
    reject_quantized(&patch_embedding_raw, "vision_tower patch embedding")?;
    let patch_embedding = normalize_patch_weight(&patch_embedding_raw, &config)?;
    let patch_embedding_bias =
        take_any_optional(name_map, &["vision_tower.embeddings.patch_embedding.bias"]);
    let position_embedding = take_any(
        name_map,
        &["vision_tower.embeddings.position_embedding.weight"],
    )?;
    reject_quantized(&position_embedding, "vision_tower position embedding")?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_index in 0..config.num_hidden_layers {
        let base = format!("vision_tower.encoder.layers.{layer_index}");
        layers.push(VisionLayer {
            self_attn: load_attention(
                name_map,
                &format!("{base}.self_attn"),
                config.hidden_size,
                config.num_attention_heads,
            )?,
            layer_norm1: load_layer_norm(
                name_map,
                &format!("{base}.layer_norm1"),
                config.layer_norm_eps,
            )?,
            layer_norm2: load_layer_norm(
                name_map,
                &format!("{base}.layer_norm2"),
                config.layer_norm_eps,
            )?,
            fc1: load_linear(name_map, &format!("{base}.mlp.fc1"))?,
            fc2: load_linear(name_map, &format!("{base}.mlp.fc2"))?,
        });
    }

    let vit_merger = VitMerger {
        pre_norm: load_layer_norm(name_map, "vit_merger.pre_norm", config.layer_norm_eps)?,
        self_attn: load_attention(
            name_map,
            "vit_merger.self_attn",
            config.hidden_size,
            config.num_attention_heads,
        )?,
        layer_norm1: load_layer_norm(name_map, "vit_merger.layer_norm1", config.layer_norm_eps)?,
        linear_1: load_linear(name_map, "vit_merger.linear_1")?,
        linear_2: load_linear(name_map, "vit_merger.linear_2")?,
        group_h: config.window_kernel_size[0],
        group_w: config.window_kernel_size[1],
        hidden_size: config.hidden_size,
    };

    let mut merger_blocks = Vec::with_capacity(config.merger_times);
    for index in 0..config.merger_times {
        let base = format!("merger.mlp.{index}");
        merger_blocks.push(MergerBlock {
            pre_norm: load_layer_norm(
                name_map,
                &format!("{base}.pre_norm"),
                config.layer_norm_eps,
            )?,
            linear_1: load_linear(name_map, &format!("{base}.linear_1"))?,
            linear_2: load_linear(name_map, &format!("{base}.linear_2"))?,
        });
    }

    Ok(Some(MiniCpmV46VisionWeights {
        config,
        patch_embedding,
        patch_embedding_bias,
        position_embedding,
        layers,
        post_layernorm: load_layer_norm(
            name_map,
            "vision_tower.post_layernorm",
            parse_vision_eps(config_json),
        )?,
        vit_merger,
        merger: FinalMerger {
            blocks: merger_blocks,
            merge_h: parse_pair(config_json.get("merge_kernel_size"), [2, 2])?[0],
            merge_w: parse_pair(config_json.get("merge_kernel_size"), [2, 2])?[1],
        },
    }))
}

pub(crate) fn build_vl_prefill_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    inputs: &MiniCpmV46RuntimeInputs,
) -> Result<MlxArray, MiniCpmV46Error> {
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    if inputs.images.is_empty() {
        return Ok(hidden);
    }
    let vision = weights
        .minicpm_v46_vision
        .as_ref()
        .ok_or(MiniCpmV46Error::MissingVisionWeights)?;

    for image in &inputs.images {
        let grid_h = image.height as usize / vision.config.patch_size;
        let grid_w = image.width as usize / vision.config.patch_size;
        // DI-W2-F1b: guard buffer length vs claimed NHWC geometry before MLX view.
        validate_minicpm_pixel_buffer_len(image.pixel_values.len(), image.height, image.width)?;
        let pixels = MlxArray::from_raw_data(
            image.pixel_values.as_ptr().cast(),
            std::mem::size_of_val(image.pixel_values.as_slice()),
            &[1, image.height as i32, image.width as i32, 3],
            MlxDtype::Float32,
        );
        let visual = vision.forward(&pixels, grid_h, grid_w)?;
        let produced = visual.shape().first().copied().unwrap_or(0) as usize;
        if produced != image.soft_token_count as usize {
            return Err(MiniCpmV46Error::InvalidGeometry(format!(
                "vision tower produced {produced} tokens, request reserved {}",
                image.soft_token_count
            )));
        }
        if visual.shape().get(1).copied().unwrap_or(0) != cfg.hidden_size as i32 {
            return Err(MiniCpmV46Error::InvalidGeometry(format!(
                "vision merger output width {}, language hidden width {}",
                visual.shape().get(1).copied().unwrap_or(0),
                cfg.hidden_size
            )));
        }
        let visual = astype(&visual, hidden.dtype(), None);
        let visual = reshape(&visual, &[1, produced as i32, cfg.hidden_size as i32], None);
        let end = image.placeholder_index.saturating_add(produced);
        if end > token_ids.len() {
            return Err(MiniCpmV46Error::Scatter(format!(
                "placeholder span {}..{end} exceeds prompt length {}",
                image.placeholder_index,
                token_ids.len()
            )));
        }
        let positions: Vec<usize> = (image.placeholder_index..end).collect();
        hidden = scatter_vision_into_text(&hidden, &visual, &positions)
            .map_err(|error| MiniCpmV46Error::Scatter(error.to_string()))?;
    }
    Ok(hidden)
}

fn parse_config(config: &Value) -> Result<MiniCpmV46VisionConfig, WeightLoadError> {
    let vision = config.get("vision_config").ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "MiniCPM-V 4.6 config has no vision_config object".to_string(),
        )
    })?;
    let parsed = MiniCpmV46VisionConfig {
        hidden_size: required_usize(vision, "hidden_size")?,
        intermediate_size: required_usize(vision, "intermediate_size")?,
        num_hidden_layers: required_usize(vision, "num_hidden_layers")?,
        num_attention_heads: required_usize(vision, "num_attention_heads")?,
        num_channels: optional_usize(vision, "num_channels", 3),
        image_size: optional_usize(vision, "image_size", 980),
        patch_size: optional_usize(vision, "patch_size", 14),
        layer_norm_eps: parse_vision_eps(config),
        window_kernel_size: parse_pair(vision.get("window_kernel_size"), [2, 2])?,
        insert_layer_id: optional_usize(config, "insert_layer_id", 6),
        merge_kernel_size: parse_pair(config.get("merge_kernel_size"), [2, 2])?,
        merger_times: optional_usize(config, "merger_times", 1),
    };
    if parsed.hidden_size == 0
        || parsed.num_hidden_layers == 0
        || parsed.num_attention_heads == 0
        || !parsed
            .hidden_size
            .is_multiple_of(parsed.num_attention_heads)
        || parsed.patch_size == 0
        || parsed.window_kernel_size.contains(&0)
        || parsed.merge_kernel_size.contains(&0)
        || parsed.merger_times == 0
    {
        return Err(WeightLoadError::InvalidLayer(format!(
            "invalid MiniCPM-V 4.6 vision config: {parsed:?}"
        )));
    }
    Ok(parsed)
}

fn parse_vision_eps(config: &Value) -> f32 {
    config
        .get("vision_config")
        .and_then(|vision| vision.get("layer_norm_eps"))
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(1.0e-6)
}

fn required_usize(config: &Value, key: &str) -> Result<usize, WeightLoadError> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            WeightLoadError::InvalidLayer(format!("MiniCPM-V 4.6 config is missing integer {key}"))
        })
}

fn optional_usize(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
}

fn parse_pair(value: Option<&Value>, default: [usize; 2]) -> Result<[usize; 2], WeightLoadError> {
    let Some(value) = value else {
        return Ok(default);
    };
    let Some(array) = value.as_array() else {
        return Err(WeightLoadError::InvalidLayer(
            "MiniCPM-V 4.6 merge size must be a two-entry array".to_string(),
        ));
    };
    if array.len() != 2 {
        return Err(WeightLoadError::InvalidLayer(
            "MiniCPM-V 4.6 merge size must contain exactly two entries".to_string(),
        ));
    }
    let mut pair = [0usize; 2];
    for (index, entry) in array.iter().enumerate() {
        pair[index] = entry
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                WeightLoadError::InvalidLayer(
                    "MiniCPM-V 4.6 merge size entries must be positive integers".to_string(),
                )
            })?;
    }
    Ok(pair)
}

fn normalize_patch_weight(
    weight: &MlxArray,
    config: &MiniCpmV46VisionConfig,
) -> Result<MlxArray, WeightLoadError> {
    let shape = weight.shape();
    let mlx_shape = [
        config.hidden_size as i32,
        config.patch_size as i32,
        config.patch_size as i32,
        config.num_channels as i32,
    ];
    let torch_shape = [
        config.hidden_size as i32,
        config.num_channels as i32,
        config.patch_size as i32,
        config.patch_size as i32,
    ];
    if shape == mlx_shape {
        Ok(weight.clone())
    } else if shape == torch_shape {
        Ok(transpose(weight, &[0, 2, 3, 1], None))
    } else {
        Err(WeightLoadError::InvalidLayer(format!(
            "MiniCPM-V patch embedding has shape {shape:?}, expected {mlx_shape:?} or {torch_shape:?}"
        )))
    }
}

fn load_attention(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
    hidden_size: usize,
    num_heads: usize,
) -> Result<Attention, WeightLoadError> {
    let head_dim = hidden_size / num_heads;
    Ok(Attention {
        q_proj: load_linear(map, &format!("{base}.q_proj"))?,
        k_proj: load_linear(map, &format!("{base}.k_proj"))?,
        v_proj: load_linear(map, &format!("{base}.v_proj"))?,
        out_proj: load_linear(map, &format!("{base}.out_proj"))?,
        num_heads,
        head_dim,
        scale: (head_dim as f32).powf(-0.5),
    })
}

fn load_linear(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
) -> Result<DenseLinear, WeightLoadError> {
    let weight = take_any(map, &[&format!("{base}.weight")])?;
    reject_quantized(&weight, base)?;
    let bias = take_any_optional(map, &[&format!("{base}.bias")]);
    Ok(DenseLinear { weight, bias })
}

fn load_layer_norm(
    map: &mut HashMap<String, MlxArray>,
    base: &str,
    eps: f32,
) -> Result<LayerNormWeights, WeightLoadError> {
    let weight = take_any(map, &[&format!("{base}.weight")])?;
    let bias = take_any_optional(map, &[&format!("{base}.bias")]);
    Ok(LayerNormWeights { weight, bias, eps })
}

fn reject_quantized(array: &MlxArray, name: &str) -> Result<(), WeightLoadError> {
    if array.dtype() == MlxDtype::Uint32 {
        Err(WeightLoadError::InvalidLayer(format!(
            "MiniCPM-V 4.6 vision tensor {name} is quantized; use the reviewed BF16 checkpoint"
        )))
    } else {
        Ok(())
    }
}

fn prefixes_for(name: &str) -> [String; 4] {
    [
        name.to_string(),
        format!("model.{name}"),
        if let Some(rest) = name.strip_prefix("vit_merger.") {
            format!("vision_tower.vit_merger.{rest}")
        } else {
            String::new()
        },
        if let Some(rest) = name.strip_prefix("vision_tower.") {
            format!("model.vpm.{rest}")
        } else {
            String::new()
        },
    ]
}

fn has_any(map: &HashMap<String, MlxArray>, names: &[&str]) -> bool {
    names
        .iter()
        .flat_map(|name| prefixes_for(name))
        .filter(|name| !name.is_empty())
        .any(|name| map.contains_key(&name))
}

fn take_any(
    map: &mut HashMap<String, MlxArray>,
    names: &[&str],
) -> Result<MlxArray, WeightLoadError> {
    for name in names {
        for candidate in prefixes_for(name) {
            if !candidate.is_empty()
                && let Some(value) = map.remove(&candidate)
            {
                return Ok(value);
            }
        }
    }
    Err(WeightLoadError::TensorMissing(
        names
            .first()
            .copied()
            .unwrap_or("MiniCPM-V tensor")
            .to_string(),
    ))
}

fn take_any_optional(map: &mut HashMap<String, MlxArray>, names: &[&str]) -> Option<MlxArray> {
    for name in names {
        for candidate in prefixes_for(name) {
            if !candidate.is_empty()
                && let Some(value) = map.remove(&candidate)
            {
                return Some(value);
            }
        }
    }
    None
}

fn dynamic_position_ids(base_side: usize, grid_h: usize, grid_w: usize) -> Vec<u32> {
    let mut positions = Vec::with_capacity(grid_h.saturating_mul(grid_w));
    for row in 0..grid_h {
        let bucket_row = row.saturating_mul(base_side) / grid_h.max(1);
        for column in 0..grid_w {
            let bucket_column = column.saturating_mul(base_side) / grid_w.max(1);
            positions.push(
                bucket_row
                    .saturating_mul(base_side)
                    .saturating_add(bucket_column) as u32,
            );
        }
    }
    positions
}

fn multiply_scalar(input: &MlxArray, value: f32) -> MlxArray {
    let scalar = MlxArray::from_raw_data(
        (&value as *const f32).cast(),
        std::mem::size_of::<f32>(),
        &[1],
        MlxDtype::Float32,
    );
    let scalar = astype(&scalar, input.dtype(), None);
    multiply(input, &scalar, None)
}

fn u32_array(values: &[u32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Uint32,
    )
}

/// Reject NHWC pixel buffers whose length does not match H*W*3 (DI-W2-F1b).
fn validate_minicpm_pixel_buffer_len(
    buffer_len: usize,
    height: u32,
    width: u32,
) -> Result<(), MiniCpmV46Error> {
    let expected_elems = (height as usize)
        .checked_mul(width as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| MiniCpmV46Error::InvalidGeometry("pixel tensor size overflow".into()))?;
    if buffer_len != expected_elems {
        return Err(MiniCpmV46Error::InvalidGeometry(format!(
            "pixel buffer length {buffer_len} != H*W*3 ({height}x{width}x3)"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_buffer_length_mismatch_is_rejected() {
        assert!(validate_minicpm_pixel_buffer_len(12, 2, 2).is_ok());
        let err = validate_minicpm_pixel_buffer_len(4, 2, 2).unwrap_err();
        assert!(
            matches!(err, MiniCpmV46Error::InvalidGeometry(msg) if msg.contains("pixel buffer length")),
            "short NHWC buffer must fail closed before from_raw_data"
        );
    }

    #[test]
    fn dynamic_position_ids_use_square_reference_buckets() {
        assert_eq!(dynamic_position_ids(4, 2, 2), vec![0, 2, 8, 10]);
    }

    #[test]
    fn released_config_defaults_to_two_fourfold_merges() {
        let config = serde_json::json!({
            "model_type": "minicpmv4_6",
            "insert_layer_id": 6,
            "vision_config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "image_size": 980,
                "patch_size": 14,
                "layer_norm_eps": 0.000001
            }
        });
        let parsed = parse_config(&config).unwrap();
        assert_eq!(parsed.window_kernel_size, [2, 2]);
        assert_eq!(parsed.merge_kernel_size, [2, 2]);
        assert_eq!(parsed.merger_times, 1);
    }
}
