//! Native Qwen3-VL, Qwen3.5, and Qwen 3.6 visual path.
//!
//! The vision tower follows the unified Qwen implementation used by current
//! Transformers and mlx-vlm checkpoints: merge-grouped Conv3D patches,
//! interpolated learned positions, 2-D rotary attention, the two-layer patch
//! merger, optional DeepStack branches, and interleaved multimodal RoPE in the
//! language model. Text-only requests continue through AX's normal language
//! graph.

use std::collections::HashMap;

use ax_engine_core::qwen3_vl::Qwen3VlRuntimeInputs;
use ax_engine_core::vl_geometry::{
    MropeSections, deepstack_injection_layers, mrope_position_ids, scatter_merge_indices,
};
use ax_engine_core::{NativeTensorRole, NativeTensorSpec};
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, gelu, gelu_approx, layer_norm, matmul, multiply,
    negative, reshape, scaled_dot_product_attention, slice, take, transpose, zeros,
};
use serde_json::Value;
use thiserror::Error;

use crate::model::{ModelConfig, embed_tokens};
use crate::weights::{ModelWeights, WeightLoadError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Qwen3VlError {
    #[error("qwen visual input requires a loaded vision tower")]
    MissingVisionWeights,
    #[error("qwen visual geometry invalid: {0}")]
    InvalidGeometry(String),
    #[error("qwen visual scatter merge failed: {0}")]
    Scatter(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3VlImageGeometry {
    pub height: u32,
    pub width: u32,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub max_soft_tokens: u32,
}

impl Qwen3VlImageGeometry {
    pub fn grid_hw(self) -> Result<(u32, u32), Qwen3VlError> {
        if self.patch_size == 0 || self.spatial_merge_size == 0 {
            return Err(Qwen3VlError::InvalidGeometry(
                "patch_size and spatial_merge_size must be > 0".into(),
            ));
        }
        let gh = self.height / self.patch_size / self.spatial_merge_size;
        let gw = self.width / self.patch_size / self.spatial_merge_size;
        if gh == 0 || gw == 0 {
            return Err(Qwen3VlError::InvalidGeometry(format!(
                "grid collapsed for {}x{} patch={} merge={}",
                self.height, self.width, self.patch_size, self.spatial_merge_size
            )));
        }
        Ok((gh, gw))
    }

    pub fn soft_token_count(self) -> Result<u32, Qwen3VlError> {
        // Qwen3-VL spatial merge emits one soft token per merged grid cell
        // `(h/p/merge)×(w/p/merge)`, matching MRoPE `grid_hw` and the runtime
        // check in `qwen_mrope_position_axes`. Do not use the Gemma pooling
        // product helper `vit_soft_token_count` here: for a 3×3 patch grid with
        // merge 2 it overcounts (product 2 vs grid 1×1 = 1). Fail closed when
        // the grid product exceeds `max_soft_tokens` so scatter/MRoPE lengths
        // never silently diverge from a capped count.
        let (gh, gw) = self.grid_hw()?;
        let count = gh.saturating_mul(gw);
        if count > self.max_soft_tokens {
            return Err(Qwen3VlError::InvalidGeometry(format!(
                "soft tokens {count} exceed max {} for {}x{} patch={} merge={}",
                self.max_soft_tokens,
                self.height,
                self.width,
                self.patch_size,
                self.spatial_merge_size
            )));
        }
        Ok(count)
    }

    pub fn mrope_sections(self) -> Result<MropeSections, Qwen3VlError> {
        let (height, width) = self.grid_hw()?;
        Ok(MropeSections::for_image(height, width))
    }
}

pub fn plan_image_scatter(
    placeholder_positions: &[usize],
    geometries: &[Qwen3VlImageGeometry],
) -> Result<Vec<usize>, Qwen3VlError> {
    if placeholder_positions.len() != geometries.len() {
        return Err(Qwen3VlError::Scatter(format!(
            "placeholders {} != images {}",
            placeholder_positions.len(),
            geometries.len()
        )));
    }
    let counts = geometries
        .iter()
        .map(|geometry| geometry.soft_token_count())
        .collect::<Result<Vec<_>, _>>()?;
    scatter_merge_indices(placeholder_positions, &counts).map_err(Qwen3VlError::Scatter)
}

pub fn plan_mrope_for_images(
    geometries: &[Qwen3VlImageGeometry],
) -> Result<Vec<u32>, Qwen3VlError> {
    let mut result = Vec::new();
    for geometry in geometries {
        // Gate on soft_token_count so MRoPE length stays consistent with
        // plan_image_scatter when the grid product exceeds max_soft_tokens.
        let expected = geometry.soft_token_count()?;
        let ids = mrope_position_ids(geometry.mrope_sections()?);
        debug_assert_eq!(ids.len(), expected as usize * 3);
        result.extend(ids);
    }
    Ok(result)
}

pub fn deepstack_layers(num_feature_maps: usize, language_layers: u32) -> Vec<u32> {
    deepstack_injection_layers(num_feature_maps, language_layers)
}

pub fn is_qwen3_vl_family(model_family: &str) -> bool {
    // Dense/MoE VL packs, plus hybrid text families that may carry a vision
    // tower (Qwen3.5 / Qwen3.6 packs sharing the portable ViT path).
    matches!(
        model_family,
        "qwen3_vl"
            | "qwen3_vl_moe"
            | "qwen3_5"
            | "qwen3.5"
            | "qwen3_5_moe"
            | "qwen3_next"
            | "qwen3.6"
            | "qwen3_6"
    )
}

pub fn text_only_decode_family(model_family: &str) -> Option<&'static str> {
    match model_family {
        "qwen3_vl" | "qwen3_vl_moe" => Some("qwen3"),
        "qwen3_5" | "qwen3.5" | "qwen3_5_moe" => Some("qwen3_5"),
        "qwen3_next" | "qwen3.6" | "qwen3_6" => Some("qwen3_next"),
        _ => None,
    }
}

pub fn require_vision_for_images(
    has_image_inputs: bool,
    has_vision_weights: bool,
) -> Result<(), Qwen3VlError> {
    if has_image_inputs && !has_vision_weights {
        return Err(Qwen3VlError::MissingVisionWeights);
    }
    Ok(())
}

pub fn has_vision_tower(weights: &ModelWeights) -> bool {
    weights.qwen3_vl_vision.is_some()
}

pub fn select_decode_route(
    model_family: &str,
    has_media: bool,
) -> Result<&'static str, Qwen3VlError> {
    if !is_qwen3_vl_family(model_family) {
        return Ok(match model_family {
            "" => "unknown",
            "qwen3" => "qwen3",
            "qwen3_5" | "qwen3.5" => "qwen3_5",
            "qwen3_next" | "qwen3.6" | "qwen3_6" => "qwen3_next",
            "gemma4" => "gemma4",
            "gemma4_vl" => "gemma4_vl",
            _ => "other",
        });
    }
    if has_media {
        Ok(if model_family == "qwen3_vl_moe" {
            "qwen3_vl_moe"
        } else if matches!(model_family, "qwen3_5" | "qwen3.5" | "qwen3_5_moe") {
            "qwen3_5"
        } else if matches!(model_family, "qwen3_next" | "qwen3.6" | "qwen3_6") {
            // Hybrid Qwen3.6 packs that load a vision tower still route media
            // through the VL prefill path; text-only decode stays qwen3_next.
            "qwen3_vl"
        } else {
            "qwen3_vl"
        })
    } else {
        Ok(text_only_decode_family(model_family).unwrap_or("qwen3"))
    }
}

/// Prepared language-side state for one visual prefill.
pub(crate) struct Qwen3VlPrefillEmbeddings {
    pub hidden: MlxArray,
    pub mrope: Option<QwenMropeCosSin>,
    pub rope_delta: i32,
    pub deepstack: Option<Qwen3VlDeepstackPrefill>,
}

/// Interleaved multimodal RoPE factors shared by full-attention layers.
pub(crate) struct QwenMropeCosSin {
    pub cos: MlxArray,
    pub sin: MlxArray,
}

/// Vision side branches injected after language layers 0, 1, ... .
pub(crate) struct Qwen3VlDeepstackPrefill {
    pub positions: Vec<usize>,
    pub features: Vec<MlxArray>,
}

pub(crate) fn build_vl_prefill_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    inputs: &Qwen3VlRuntimeInputs,
) -> Result<Qwen3VlPrefillEmbeddings, Qwen3VlError> {
    if !inputs.images.is_empty() && !has_vision_tower(weights) {
        return Err(Qwen3VlError::MissingVisionWeights);
    }
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    if inputs.images.is_empty() {
        return Ok(Qwen3VlPrefillEmbeddings {
            hidden,
            mrope: None,
            rope_delta: 0,
            deepstack: None,
        });
    }

    let vision = weights
        .qwen3_vl_vision
        .as_ref()
        .ok_or(Qwen3VlError::MissingVisionWeights)?;
    let mut deepstack_parts: Vec<Vec<MlxArray>> =
        vec![Vec::new(); vision.config.deepstack_visual_indexes.len()];
    let mut visual_positions = Vec::new();

    for media in &inputs.images {
        // DI-W2-F1c: MLX indexes by claimed shape; reject buffer/shape drift
        // (mirror gemma4_vl pixel_values.len() == expected).
        validate_qwen3_vl_patch_buffer_len(
            media.patches.len(),
            media.num_patches,
            media.patch_dim,
        )?;
        let patches = MlxArray::from_raw_data(
            media.patches.as_ptr().cast(),
            std::mem::size_of_val(media.patches.as_slice()),
            &[media.num_patches as i32, media.patch_dim as i32],
            MlxDtype::Float32,
        );
        let grid_h = media.height / media.patch_size;
        let grid_w = media.width / media.patch_size;
        let (soft, deepstack) =
            vision_encoder_forward(vision, &patches, (media.grid_t, grid_h, grid_w))?;
        let produced = soft.shape().get(1).copied().unwrap_or(0);
        if produced != media.soft_token_count as i32 {
            return Err(Qwen3VlError::InvalidGeometry(format!(
                "vision tower produced {produced} soft tokens, request declared {}",
                media.soft_token_count
            )));
        }
        let end = media
            .placeholder_index
            .checked_add(media.soft_token_count as usize)
            .ok_or_else(|| Qwen3VlError::Scatter("visual token range overflow".into()))?;
        if end > token_ids.len() {
            return Err(Qwen3VlError::Scatter(format!(
                "visual token range {}..{end} exceeds prompt length {}",
                media.placeholder_index,
                token_ids.len()
            )));
        }
        let positions: Vec<usize> = (media.placeholder_index..end).collect();
        hidden = scatter_vision_into_text(&hidden, &soft, &positions)?;
        visual_positions.extend_from_slice(&positions);

        if deepstack.len() != deepstack_parts.len() {
            return Err(Qwen3VlError::InvalidGeometry(format!(
                "vision tower produced {} DeepStack maps, expected {}",
                deepstack.len(),
                deepstack_parts.len()
            )));
        }
        for (layer, feature) in deepstack.into_iter().enumerate() {
            deepstack_parts[layer].push(feature);
        }
    }

    let axes = qwen_mrope_position_axes(token_ids.len(), inputs)?;
    let max_position = axes
        .iter()
        .flat_map(|position| position.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let prompt_len = i32::try_from(token_ids.len()).unwrap_or(i32::MAX);
    let rope_delta = max_position.saturating_add(1).saturating_sub(prompt_len);
    let mrope = Some(build_interleaved_mrope(
        &axes,
        cfg.rope_dims,
        cfg.rope_theta,
        &vision.mrope_section,
    )?);

    let deepstack = if deepstack_parts.is_empty() {
        None
    } else {
        let mut features = Vec::with_capacity(deepstack_parts.len());
        for parts in deepstack_parts {
            let refs: Vec<&MlxArray> = parts.iter().collect();
            features.push(match refs.as_slice() {
                [] => {
                    return Err(Qwen3VlError::InvalidGeometry(
                        "DeepStack feature list is empty".into(),
                    ));
                }
                [single] => (*single).clone(),
                _ => concatenate(&refs, 1, None),
            });
        }
        Some(Qwen3VlDeepstackPrefill {
            positions: visual_positions,
            features,
        })
    };

    Ok(Qwen3VlPrefillEmbeddings {
        hidden,
        mrope,
        rope_delta,
        deepstack,
    })
}

fn qwen_mrope_position_axes(
    prompt_len: usize,
    inputs: &Qwen3VlRuntimeInputs,
) -> Result<Vec<[i32; 3]>, Qwen3VlError> {
    let mut media: Vec<_> = inputs.images.iter().collect();
    media.sort_by_key(|item| item.placeholder_index);
    let mut result = Vec::with_capacity(prompt_len);
    let mut cursor = 0usize;
    let mut current_position = 0i32;

    for item in media {
        if item.placeholder_index < cursor {
            return Err(Qwen3VlError::Scatter(
                "visual token ranges overlap or are out of order".into(),
            ));
        }
        for _ in cursor..item.placeholder_index {
            result.push([current_position; 3]);
            current_position = current_position.saturating_add(1);
        }

        let merge = item.spatial_merge_size;
        let llm_t = item.grid_t;
        let llm_h = item.height / item.patch_size / merge;
        let llm_w = item.width / item.patch_size / merge;
        let expected = llm_t.saturating_mul(llm_h).saturating_mul(llm_w);
        if expected != item.soft_token_count {
            return Err(Qwen3VlError::InvalidGeometry(format!(
                "soft_token_count {} != merged grid {}x{}x{} ({expected})",
                item.soft_token_count, llm_t, llm_h, llm_w
            )));
        }
        for t in 0..llm_t {
            for h in 0..llm_h {
                for w in 0..llm_w {
                    result.push([
                        current_position.saturating_add(t as i32),
                        current_position.saturating_add(h as i32),
                        current_position.saturating_add(w as i32),
                    ]);
                }
            }
        }
        current_position = current_position
            .saturating_add(i32::try_from(llm_t.max(llm_h).max(llm_w)).unwrap_or(i32::MAX));
        cursor = item
            .placeholder_index
            .saturating_add(item.soft_token_count as usize);
    }

    for _ in cursor..prompt_len {
        result.push([current_position; 3]);
        current_position = current_position.saturating_add(1);
    }
    if result.len() != prompt_len {
        return Err(Qwen3VlError::Scatter(format!(
            "MRoPE positions {} != prompt length {prompt_len}",
            result.len()
        )));
    }
    Ok(result)
}

fn build_interleaved_mrope(
    positions: &[[i32; 3]],
    rotary_dim: usize,
    theta: f32,
    sections: &[usize],
) -> Result<QwenMropeCosSin, Qwen3VlError> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(Qwen3VlError::InvalidGeometry(format!(
            "rotary_dim {rotary_dim} must be a positive even number"
        )));
    }
    if sections.len() != 3 || sections.iter().sum::<usize>() * 2 != rotary_dim {
        return Err(Qwen3VlError::InvalidGeometry(format!(
            "MRoPE sections {sections:?} do not cover rotary_dim {rotary_dim}"
        )));
    }
    let half = rotary_dim / 2;
    let mut source_axis = vec![0usize; half];
    for (axis, section) in sections.iter().copied().enumerate().skip(1) {
        let mut index = axis;
        while index < section.saturating_mul(3) && index < half {
            source_axis[index] = axis;
            index += 3;
        }
    }
    let inv_freq: Vec<f32> = (0..half)
        .map(|index| 1.0 / theta.powf((2 * index) as f32 / rotary_dim as f32))
        .collect();
    let mut cos_values = Vec::with_capacity(positions.len() * rotary_dim);
    let mut sin_values = Vec::with_capacity(positions.len() * rotary_dim);
    for position in positions {
        let frequencies: Vec<f32> = (0..half)
            .map(|index| position[source_axis[index]] as f32 * inv_freq[index])
            .collect();
        for _ in 0..2 {
            cos_values.extend(frequencies.iter().map(|frequency| frequency.cos()));
            sin_values.extend(frequencies.iter().map(|frequency| frequency.sin()));
        }
    }
    let shape = [1, positions.len() as i32, rotary_dim as i32];
    Ok(QwenMropeCosSin {
        cos: f32_array(&cos_values, &shape),
        sin: f32_array(&sin_values, &shape),
    })
}

/// Apply precomputed multimodal rotary factors to `[B, H, S, D]` Q/K.
pub(crate) fn apply_interleaved_mrope(
    tensor: &MlxArray,
    factors: &QwenMropeCosSin,
    rotary_dim: usize,
) -> MlxArray {
    let shape = tensor.shape();
    let batch = shape[0];
    let heads = shape[1];
    let seq = shape[2];
    let head_dim = shape[3];
    let rotary_dim = rotary_dim as i32;
    let rotated_input = slice(
        tensor,
        &[0, 0, 0, 0],
        &[batch, heads, seq, rotary_dim],
        &[1, 1, 1, 1],
        None,
    );
    let pass = (rotary_dim < head_dim).then(|| {
        slice(
            tensor,
            &[0, 0, 0, rotary_dim],
            &[batch, heads, seq, head_dim],
            &[1, 1, 1, 1],
            None,
        )
    });
    let cos = astype(&factors.cos, tensor.dtype(), None);
    let sin = astype(&factors.sin, tensor.dtype(), None);
    let cos = reshape(&cos, &[batch, 1, seq, rotary_dim], None);
    let sin = reshape(&sin, &[batch, 1, seq, rotary_dim], None);
    let first = multiply(&rotated_input, &cos, None);
    let second = multiply(&rotate_half(&rotated_input), &sin, None);
    let embedded = add(&first, &second, None);
    pass.map_or(embedded.clone(), |pass| {
        concatenate(&[&embedded, &pass], -1, None)
    })
}

#[derive(Clone, Debug)]
pub struct Qwen3VlVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Qwen3VlVisionLayerWeights {
    pub qkv: MlxArray,
    pub qkv_bias: Option<MlxArray>,
    pub proj: MlxArray,
    pub proj_bias: Option<MlxArray>,
    pub norm1_weight: MlxArray,
    pub norm1_bias: Option<MlxArray>,
    pub fc1: MlxArray,
    pub fc1_bias: Option<MlxArray>,
    pub fc2: MlxArray,
    pub fc2_bias: Option<MlxArray>,
    pub norm2_weight: MlxArray,
    pub norm2_bias: Option<MlxArray>,
}

#[derive(Clone, Debug)]
pub struct Qwen3VlMergerWeights {
    pub norm_weight: MlxArray,
    pub norm_bias: Option<MlxArray>,
    pub linear_fc1: MlxArray,
    pub linear_fc1_bias: Option<MlxArray>,
    pub linear_fc2: MlxArray,
    pub linear_fc2_bias: Option<MlxArray>,
    pub postshuffle_norm: bool,
}

#[derive(Clone, Debug)]
pub struct Qwen3VlVisionWeights {
    pub config: Qwen3VlVisionConfig,
    pub patch_embed: MlxArray,
    pub patch_embed_bias: Option<MlxArray>,
    pub pos_embed: MlxArray,
    pub layers: Vec<Qwen3VlVisionLayerWeights>,
    pub merger: Qwen3VlMergerWeights,
    pub deepstack_mergers: Vec<Qwen3VlMergerWeights>,
    pub mrope_section: Vec<usize>,
}

pub fn load_qwen3_vl_vision_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    config_json: Option<&Value>,
) -> Result<Option<Qwen3VlVisionWeights>, WeightLoadError> {
    let is_qwen_visual_config = config_json
        .and_then(|config| config.get("model_type"))
        .and_then(Value::as_str)
        .is_some_and(|model_type| {
            matches!(
                model_type,
                "qwen3_vl"
                    | "qwen3-vl"
                    | "qwen3_vl_moe"
                    | "qwen3-vl-moe"
                    | "qwen3_5"
                    | "qwen3.5"
                    | "qwen3_5_moe"
                    // Qwen3.6 HF aliases (convert → qwen3_next) may still ship a
                    // vision tower; silent skip left packs without vision weights.
                    | "qwen3_next"
                    | "qwen3.6"
                    | "qwen3_6"
                    | "qwen3_5_moe_text"
            )
        });
    if !is_qwen_visual_config {
        return Ok(None);
    }
    let has_tower = name_map.keys().any(|name| {
        name.starts_with("vision_tower.")
            || name.starts_with("visual.")
            || name.starts_with("model.visual.")
    }) || specs.iter().any(|spec| {
        matches!(
            spec.role,
            NativeTensorRole::Qwen3VlVisionPatchEmbed
                | NativeTensorRole::Qwen3VlVisionMerger
                | NativeTensorRole::Qwen3VlVisionLayerQkv
        )
    });
    if !has_tower {
        return Ok(None);
    }
    let config_json = config_json.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "Qwen visual checkpoint is missing a readable config.json".into(),
        )
    })?;
    let (config, mrope_section) = parse_qwen_vision_config(config_json)?;

    let patch_embed_raw = take_visual(name_map, "patch_embed.proj.weight")?;
    let patch_embed = normalize_patch_embed_weight(&patch_embed_raw, &config)?;
    let patch_embed_bias = take_visual_optional(name_map, "patch_embed.proj.bias");
    let pos_embed = take_visual(name_map, "pos_embed.weight")?;

    let mut layers = Vec::with_capacity(config.depth);
    for layer in 0..config.depth {
        let prefix = format!("blocks.{layer}");
        layers.push(Qwen3VlVisionLayerWeights {
            qkv: take_visual(name_map, &format!("{prefix}.attn.qkv.weight"))?,
            qkv_bias: take_visual_optional(name_map, &format!("{prefix}.attn.qkv.bias")),
            proj: take_visual(name_map, &format!("{prefix}.attn.proj.weight"))?,
            proj_bias: take_visual_optional(name_map, &format!("{prefix}.attn.proj.bias")),
            norm1_weight: take_visual(name_map, &format!("{prefix}.norm1.weight"))?,
            norm1_bias: take_visual_optional(name_map, &format!("{prefix}.norm1.bias")),
            fc1: take_visual(name_map, &format!("{prefix}.mlp.linear_fc1.weight"))?,
            fc1_bias: take_visual_optional(name_map, &format!("{prefix}.mlp.linear_fc1.bias")),
            fc2: take_visual(name_map, &format!("{prefix}.mlp.linear_fc2.weight"))?,
            fc2_bias: take_visual_optional(name_map, &format!("{prefix}.mlp.linear_fc2.bias")),
            norm2_weight: take_visual(name_map, &format!("{prefix}.norm2.weight"))?,
            norm2_bias: take_visual_optional(name_map, &format!("{prefix}.norm2.bias")),
        });
    }

    let merger = load_merger(name_map, "merger", false)?;
    let mut deepstack_mergers = Vec::with_capacity(config.deepstack_visual_indexes.len());
    for index in 0..config.deepstack_visual_indexes.len() {
        deepstack_mergers.push(load_merger(
            name_map,
            &format!("deepstack_merger_list.{index}"),
            true,
        )?);
    }
    name_map.retain(|name, _| {
        !name.starts_with("vision_tower.")
            && !name.starts_with("visual.")
            && !name.starts_with("model.visual.")
    });

    Ok(Some(Qwen3VlVisionWeights {
        config,
        patch_embed,
        patch_embed_bias,
        pos_embed,
        layers,
        merger,
        deepstack_mergers,
        mrope_section,
    }))
}

fn parse_qwen_vision_config(
    config: &Value,
) -> Result<(Qwen3VlVisionConfig, Vec<usize>), WeightLoadError> {
    let vision = config.get("vision_config").ok_or_else(|| {
        WeightLoadError::InvalidLayer("Qwen checkpoint has no vision_config".into())
    })?;
    let required = |key: &str| -> Result<usize, WeightLoadError> {
        vision
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                WeightLoadError::InvalidLayer(format!(
                    "Qwen vision_config.{key} is missing or invalid"
                ))
            })
    };
    let deepstack_visual_indexes = vision
        .get("deepstack_visual_indexes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    item.as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            WeightLoadError::InvalidLayer(
                                "invalid deepstack_visual_indexes entry".into(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let text = config.get("text_config").unwrap_or(config);
    let rope = text
        .get("rope_scaling")
        .or_else(|| text.get("rope_parameters"));
    let mrope_section = rope
        .and_then(|value| value.get("mrope_section"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|value| usize::try_from(value).ok())
                .collect::<Vec<_>>()
        })
        .filter(|items| items.len() == 3)
        .unwrap_or_else(|| {
            // Holo3 / Ornith / official Qwen 3.5 MoE packs use model_type
            // `qwen3_5_moe` (not bare `qwen3_5`). Their default MRoPE split is
            // the Qwen 3.5 [11, 11, 10] layout; the VL [24, 20, 20] fallback
            // is only for Qwen3-VL-class checkpoints.
            if matches!(
                config.get("model_type").and_then(Value::as_str),
                Some("qwen3_5" | "qwen3.5" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_5_text")
            ) {
                vec![11, 11, 10]
            } else {
                vec![24, 20, 20]
            }
        });
    Ok((
        Qwen3VlVisionConfig {
            depth: required("depth")?,
            hidden_size: required("hidden_size")?,
            intermediate_size: required("intermediate_size")?,
            out_hidden_size: required("out_hidden_size")?,
            num_heads: required("num_heads")?,
            in_channels: required("in_channels")?,
            patch_size: required("patch_size")?,
            temporal_patch_size: required("temporal_patch_size")?,
            spatial_merge_size: required("spatial_merge_size")?,
            num_position_embeddings: required("num_position_embeddings")?,
            deepstack_visual_indexes,
        },
        mrope_section,
    ))
}

fn visual_names(suffix: &str) -> [String; 3] {
    [
        format!("vision_tower.{suffix}"),
        format!("visual.{suffix}"),
        format!("model.visual.{suffix}"),
    ]
}

fn take_visual(
    name_map: &mut HashMap<String, MlxArray>,
    suffix: &str,
) -> Result<MlxArray, WeightLoadError> {
    for name in visual_names(suffix) {
        if let Some(value) = name_map.remove(&name) {
            return Ok(value);
        }
    }
    Err(WeightLoadError::TensorMissing(format!(
        "vision_tower.{suffix}, visual.{suffix}, or model.visual.{suffix}"
    )))
}

fn take_visual_optional(
    name_map: &mut HashMap<String, MlxArray>,
    suffix: &str,
) -> Option<MlxArray> {
    visual_names(suffix)
        .into_iter()
        .find_map(|name| name_map.remove(&name))
}

fn load_merger(
    name_map: &mut HashMap<String, MlxArray>,
    prefix: &str,
    postshuffle_norm: bool,
) -> Result<Qwen3VlMergerWeights, WeightLoadError> {
    Ok(Qwen3VlMergerWeights {
        norm_weight: take_visual(name_map, &format!("{prefix}.norm.weight"))?,
        norm_bias: take_visual_optional(name_map, &format!("{prefix}.norm.bias")),
        linear_fc1: take_visual(name_map, &format!("{prefix}.linear_fc1.weight"))?,
        linear_fc1_bias: take_visual_optional(name_map, &format!("{prefix}.linear_fc1.bias")),
        linear_fc2: take_visual(name_map, &format!("{prefix}.linear_fc2.weight"))?,
        linear_fc2_bias: take_visual_optional(name_map, &format!("{prefix}.linear_fc2.bias")),
        postshuffle_norm,
    })
}

fn normalize_patch_embed_weight(
    weight: &MlxArray,
    config: &Qwen3VlVisionConfig,
) -> Result<MlxArray, WeightLoadError> {
    let shape = weight.shape();
    let out = config.hidden_size as i32;
    let input =
        (config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size)
            as i32;
    if shape.len() == 2 {
        if shape != [out, input] {
            return Err(WeightLoadError::InvalidLayer(format!(
                "Qwen patch_embed 2-D shape {shape:?}, expected [{out}, {input}]"
            )));
        }
        return Ok(weight.clone());
    }
    if shape.len() != 5 || shape[0] != out {
        return Err(WeightLoadError::InvalidLayer(format!(
            "Qwen patch_embed shape {shape:?} is not a supported Conv3D layout"
        )));
    }
    let channels = config.in_channels as i32;
    let temporal = config.temporal_patch_size as i32;
    let reordered = if shape[1] == temporal && shape[4] == channels {
        // Sanitized MLX Conv3D: [out, T, H, W, C] -> [out, C, T, H, W].
        transpose(weight, &[0, 4, 1, 2, 3], None)
    } else if shape[1] == channels && shape[2] == temporal {
        // Raw PyTorch Conv3D already matches request rows [C, T, H, W].
        weight.clone()
    } else {
        return Err(WeightLoadError::InvalidLayer(format!(
            "cannot identify Qwen patch_embed Conv3D layout {shape:?}"
        )));
    };
    Ok(reshape(&reordered, &[out, input], None))
}

pub fn vision_encoder_forward(
    weights: &Qwen3VlVisionWeights,
    patches: &MlxArray,
    grid_thw: (u32, u32, u32),
) -> Result<(MlxArray, Vec<MlxArray>), Qwen3VlError> {
    let (grid_t, grid_h, grid_w) = grid_thw;
    validate_vision_geometry(weights, patches, grid_t, grid_h, grid_w)?;
    let patch_input = astype(patches, weights.patch_embed.dtype(), None);
    let mut hidden = linear(
        &patch_input,
        &weights.patch_embed,
        weights.patch_embed_bias.as_ref(),
    );
    let positions = interpolated_position_embeddings(
        &weights.pos_embed,
        weights.config.num_position_embeddings,
        grid_t,
        grid_h,
        grid_w,
        weights.config.spatial_merge_size as u32,
    )?;
    hidden = add(&hidden, &astype(&positions, hidden.dtype(), None), None);
    let rotary = vision_rotary_factors(
        grid_t,
        grid_h,
        grid_w,
        weights.config.spatial_merge_size as u32,
        weights.config.hidden_size / weights.config.num_heads,
    );

    let frame_tokens = (grid_h * grid_w) as usize;
    let mut deepstack = Vec::with_capacity(weights.deepstack_mergers.len());
    for (layer_index, layer) in weights.layers.iter().enumerate() {
        let normed = layer_norm_optional(
            &hidden,
            &layer.norm1_weight,
            layer.norm1_bias.as_ref(),
            1e-6,
        );
        let attention = vision_attention(
            &normed,
            layer,
            weights.config.num_heads,
            &rotary,
            grid_t as usize,
            frame_tokens,
        );
        hidden = add(&hidden, &attention, None);
        let normed = layer_norm_optional(
            &hidden,
            &layer.norm2_weight,
            layer.norm2_bias.as_ref(),
            1e-6,
        );
        let mlp = linear(
            &gelu_approx(&linear(&normed, &layer.fc1, layer.fc1_bias.as_ref()), None),
            &layer.fc2,
            layer.fc2_bias.as_ref(),
        );
        hidden = add(&hidden, &mlp, None);
        if let Some(index) = weights
            .config
            .deepstack_visual_indexes
            .iter()
            .position(|candidate| *candidate == layer_index)
        {
            let feature = merger_forward(
                &hidden,
                &weights.deepstack_mergers[index],
                weights.config.hidden_size,
                weights.config.spatial_merge_size,
            );
            deepstack.push(reshape(
                &feature,
                &[1, feature.shape()[0], feature.shape()[1]],
                None,
            ));
        }
    }
    let merged = merger_forward(
        &hidden,
        &weights.merger,
        weights.config.hidden_size,
        weights.config.spatial_merge_size,
    );
    let merged = reshape(&merged, &[1, merged.shape()[0], merged.shape()[1]], None);
    Ok((merged, deepstack))
}

fn validate_vision_geometry(
    weights: &Qwen3VlVisionWeights,
    patches: &MlxArray,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
) -> Result<(), Qwen3VlError> {
    let config = &weights.config;
    if grid_t == 0 || grid_h == 0 || grid_w == 0 {
        return Err(Qwen3VlError::InvalidGeometry(
            "grid_t, grid_h, and grid_w must be > 0".into(),
        ));
    }
    let merge = config.spatial_merge_size as u32;
    if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
        return Err(Qwen3VlError::InvalidGeometry(format!(
            "grid {grid_h}x{grid_w} is not divisible by merge {merge}"
        )));
    }
    let shape = patches.shape();
    let expected_rows = grid_t.saturating_mul(grid_h).saturating_mul(grid_w) as i32;
    let expected_dim =
        (config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size)
            as i32;
    if shape != [expected_rows, expected_dim] {
        return Err(Qwen3VlError::InvalidGeometry(format!(
            "patch tensor {shape:?}, expected [{expected_rows}, {expected_dim}]"
        )));
    }
    if config.hidden_size == 0
        || config.num_heads == 0
        || !config.hidden_size.is_multiple_of(config.num_heads)
    {
        return Err(Qwen3VlError::InvalidGeometry(
            "vision hidden_size must divide num_heads".into(),
        ));
    }
    Ok(())
}

fn interpolated_position_embeddings(
    table: &MlxArray,
    num_positions: usize,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    merge: u32,
) -> Result<MlxArray, Qwen3VlError> {
    let base = (num_positions as f64).sqrt() as u32;
    if base == 0 || (base as usize).saturating_mul(base as usize) != num_positions {
        return Err(Qwen3VlError::InvalidGeometry(format!(
            "num_position_embeddings {num_positions} is not a square"
        )));
    }
    let mut indices = [Vec::<u32>::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut weights = [Vec::<f32>::new(), Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..grid_t {
        for block_h in 0..grid_h / merge {
            for block_w in 0..grid_w / merge {
                for inner_h in 0..merge {
                    for inner_w in 0..merge {
                        let h = block_h * merge + inner_h;
                        let w = block_w * merge + inner_w;
                        let h_pos = if grid_h > 1 {
                            h as f32 * (base - 1) as f32 / (grid_h - 1) as f32
                        } else {
                            0.0
                        };
                        let w_pos = if grid_w > 1 {
                            w as f32 * (base - 1) as f32 / (grid_w - 1) as f32
                        } else {
                            0.0
                        };
                        let h0 = h_pos.floor() as u32;
                        let w0 = w_pos.floor() as u32;
                        let h1 = (h0 + 1).min(base - 1);
                        let w1 = (w0 + 1).min(base - 1);
                        let dh = h_pos - h0 as f32;
                        let dw = w_pos - w0 as f32;
                        indices[0].push(h0 * base + w0);
                        indices[1].push(h0 * base + w1);
                        indices[2].push(h1 * base + w0);
                        indices[3].push(h1 * base + w1);
                        weights[0].push((1.0 - dh) * (1.0 - dw));
                        weights[1].push((1.0 - dh) * dw);
                        weights[2].push(dh * (1.0 - dw));
                        weights[3].push(dh * dw);
                    }
                }
            }
        }
    }
    let count = indices[0].len() as i32;
    let mut result: Option<MlxArray> = None;
    for corner in 0..4 {
        let index = u32_array(&indices[corner], &[count]);
        let selected = take(table, &index, 0, None);
        let coefficient = astype(
            &f32_array(&weights[corner], &[count, 1]),
            selected.dtype(),
            None,
        );
        let weighted = multiply(&selected, &coefficient, None);
        result = Some(match result {
            Some(accumulator) => add(&accumulator, &weighted, None),
            None => weighted,
        });
    }
    result.ok_or_else(|| {
        Qwen3VlError::InvalidGeometry("position interpolation produced no values".into())
    })
}

struct VisionRotaryFactors {
    cos: MlxArray,
    sin: MlxArray,
}

fn vision_rotary_factors(
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    merge: u32,
    head_dim: usize,
) -> VisionRotaryFactors {
    let quarter = head_dim / 4;
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|index| 1.0 / 10_000f32.powf((2 * index) as f32 / (head_dim / 2) as f32))
        .collect();
    let count = grid_t.saturating_mul(grid_h).saturating_mul(grid_w) as usize;
    let mut cos_values = Vec::with_capacity(count * head_dim);
    let mut sin_values = Vec::with_capacity(count * head_dim);
    for _ in 0..grid_t {
        for block_h in 0..grid_h / merge {
            for block_w in 0..grid_w / merge {
                for inner_h in 0..merge {
                    for inner_w in 0..merge {
                        let h = (block_h * merge + inner_h) as f32;
                        let w = (block_w * merge + inner_w) as f32;
                        let mut half = Vec::with_capacity(head_dim / 2);
                        half.extend(inv_freq.iter().map(|frequency| h * frequency));
                        half.extend(inv_freq.iter().map(|frequency| w * frequency));
                        for _ in 0..2 {
                            cos_values.extend(half.iter().map(|frequency| frequency.cos()));
                            sin_values.extend(half.iter().map(|frequency| frequency.sin()));
                        }
                    }
                }
            }
        }
    }
    VisionRotaryFactors {
        cos: f32_array(&cos_values, &[count as i32, 1, head_dim as i32]),
        sin: f32_array(&sin_values, &[count as i32, 1, head_dim as i32]),
    }
}

fn vision_attention(
    hidden: &MlxArray,
    layer: &Qwen3VlVisionLayerWeights,
    num_heads: usize,
    rotary: &VisionRotaryFactors,
    frames: usize,
    frame_tokens: usize,
) -> MlxArray {
    let seq = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];
    let head_dim = hidden_size / num_heads as i32;
    let qkv = linear(hidden, &layer.qkv, layer.qkv_bias.as_ref());
    let qkv = reshape(&qkv, &[seq, 3, num_heads as i32, head_dim], None);
    let qkv = transpose(&qkv, &[1, 0, 2, 3], None);
    let extract = |index: i32| {
        let value = slice(
            &qkv,
            &[index, 0, 0, 0],
            &[index + 1, seq, num_heads as i32, head_dim],
            &[1, 1, 1, 1],
            None,
        );
        reshape(&value, &[seq, num_heads as i32, head_dim], None)
    };
    let q = apply_vision_rope(&extract(0), rotary);
    let k = apply_vision_rope(&extract(1), rotary);
    let v = extract(2);
    let q = reshape(
        &transpose(&q, &[1, 0, 2], None),
        &[1, num_heads as i32, seq, head_dim],
        None,
    );
    let k = reshape(
        &transpose(&k, &[1, 0, 2], None),
        &[1, num_heads as i32, seq, head_dim],
        None,
    );
    let v = reshape(
        &transpose(&v, &[1, 0, 2], None),
        &[1, num_heads as i32, seq, head_dim],
        None,
    );
    let mut parts = Vec::with_capacity(frames);
    for frame in 0..frames {
        let start = (frame * frame_tokens) as i32;
        let end = start + frame_tokens as i32;
        let segment = |value: &MlxArray| {
            slice(
                value,
                &[0, 0, start, 0],
                &[1, num_heads as i32, end, head_dim],
                &[1, 1, 1, 1],
                None,
            )
        };
        parts.push(scaled_dot_product_attention(
            &segment(&q),
            &segment(&k),
            &segment(&v),
            (head_dim as f32).powf(-0.5),
            false,
            None,
        ));
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    let context = if refs.len() == 1 {
        refs[0].clone()
    } else {
        concatenate(&refs, 2, None)
    };
    let context = transpose(&context, &[0, 2, 1, 3], None);
    let context = reshape(&context, &[seq, hidden_size], None);
    linear(&context, &layer.proj, layer.proj_bias.as_ref())
}

fn apply_vision_rope(tensor: &MlxArray, factors: &VisionRotaryFactors) -> MlxArray {
    let cos = astype(&factors.cos, tensor.dtype(), None);
    let sin = astype(&factors.sin, tensor.dtype(), None);
    add(
        &multiply(tensor, &cos, None),
        &multiply(&rotate_half(tensor), &sin, None),
        None,
    )
}

fn merger_forward(
    hidden: &MlxArray,
    merger: &Qwen3VlMergerWeights,
    hidden_size: usize,
    spatial_merge_size: usize,
) -> MlxArray {
    let merged_hidden = (hidden_size * spatial_merge_size * spatial_merge_size) as i32;
    let normalized = if merger.postshuffle_norm {
        let reshaped = reshape(hidden, &[-1, merged_hidden], None);
        layer_norm_optional(
            &reshaped,
            &merger.norm_weight,
            merger.norm_bias.as_ref(),
            1e-6,
        )
    } else {
        let normalized =
            layer_norm_optional(hidden, &merger.norm_weight, merger.norm_bias.as_ref(), 1e-6);
        reshape(&normalized, &[-1, merged_hidden], None)
    };
    let first = linear(
        &normalized,
        &merger.linear_fc1,
        merger.linear_fc1_bias.as_ref(),
    );
    linear(
        &gelu(&first, None),
        &merger.linear_fc2,
        merger.linear_fc2_bias.as_ref(),
    )
}

fn linear(input: &MlxArray, weight: &MlxArray, bias: Option<&MlxArray>) -> MlxArray {
    let input = if input.dtype() == weight.dtype() {
        input.clone()
    } else {
        astype(input, weight.dtype(), None)
    };
    let output = matmul(&input, &transpose(weight, &[1, 0], None), None);
    bias.map_or(output.clone(), |bias| add(&output, bias, None))
}

fn layer_norm_optional(
    input: &MlxArray,
    weight: &MlxArray,
    bias: Option<&MlxArray>,
    eps: f32,
) -> MlxArray {
    let zero;
    let bias = match bias {
        Some(bias) => bias,
        None => {
            zero = zeros(&weight.shape(), weight.dtype(), None);
            &zero
        }
    };
    layer_norm(input, weight, bias, eps, None)
}

fn rotate_half(input: &MlxArray) -> MlxArray {
    let shape = input.shape();
    let ndim = shape.len();
    let half = shape[ndim - 1] / 2;
    let mut first_start = vec![0; ndim];
    let mut first_stop = shape.clone();
    first_stop[ndim - 1] = half;
    let first = slice(input, &first_start, &first_stop, &vec![1; ndim], None);
    first_start[ndim - 1] = half;
    first_stop[ndim - 1] = shape[ndim - 1];
    let second = slice(input, &first_start, &first_stop, &vec![1; ndim], None);
    concatenate(&[&negative(&second, None), &first], -1, None)
}

pub fn scatter_vision_into_text(
    text_hidden: &MlxArray,
    vision: &MlxArray,
    positions: &[usize],
) -> Result<MlxArray, Qwen3VlError> {
    scatter_or_add_visual(text_hidden, vision, positions, false)
}

pub(crate) fn add_deepstack_into_text(
    text_hidden: &MlxArray,
    vision: &MlxArray,
    positions: &[usize],
) -> Result<MlxArray, Qwen3VlError> {
    scatter_or_add_visual(text_hidden, vision, positions, true)
}

fn scatter_or_add_visual(
    text_hidden: &MlxArray,
    vision: &MlxArray,
    positions: &[usize],
    add_to_existing: bool,
) -> Result<MlxArray, Qwen3VlError> {
    if positions.is_empty() {
        return Ok(text_hidden.clone());
    }
    let shape = text_hidden.shape();
    if shape.len() != 3 || vision.shape().len() != 3 {
        return Err(Qwen3VlError::Scatter(
            "text and vision tensors must both be rank 3".into(),
        ));
    }
    let tokens = shape[1] as usize;
    let hidden = shape[2];
    if positions.len() != vision.shape()[1] as usize {
        return Err(Qwen3VlError::Scatter(format!(
            "positions {} != vision tokens {}",
            positions.len(),
            vision.shape()[1]
        )));
    }
    if positions.iter().any(|position| *position >= tokens) {
        return Err(Qwen3VlError::Scatter(
            "visual position exceeds text sequence".into(),
        ));
    }
    let mut visual_by_position = HashMap::with_capacity(positions.len());
    for (index, position) in positions.iter().copied().enumerate() {
        visual_by_position.insert(position, index);
    }
    let mut rows = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let text_row = slice(
            text_hidden,
            &[0, token as i32, 0],
            &[1, token as i32 + 1, hidden],
            &[1, 1, 1],
            None,
        );
        if let Some(index) = visual_by_position.get(&token).copied() {
            let visual_row = slice(
                vision,
                &[0, index as i32, 0],
                &[1, index as i32 + 1, hidden],
                &[1, 1, 1],
                None,
            );
            rows.push(if add_to_existing {
                add(&text_row, &visual_row, None)
            } else {
                visual_row
            });
        } else {
            rows.push(text_row);
        }
    }
    let refs: Vec<&MlxArray> = rows.iter().collect();
    Ok(concatenate(&refs, 1, None))
}

fn f32_array(values: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Float32,
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

/// Reject patch buffers whose length does not match the claimed geometry
/// (DI-W2-F1c / gemma4_vl defensive pattern).
fn validate_qwen3_vl_patch_buffer_len(
    buffer_len: usize,
    num_patches: u32,
    patch_dim: u32,
) -> Result<(), Qwen3VlError> {
    let expected_elems = (num_patches as usize)
        .checked_mul(patch_dim as usize)
        .ok_or_else(|| Qwen3VlError::InvalidGeometry("patch tensor size overflow".into()))?;
    if buffer_len != expected_elems {
        return Err(Qwen3VlError::InvalidGeometry(format!(
            "patch buffer length {buffer_len} != num_patches {num_patches} * patch_dim {patch_dim}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::qwen3_vl::Qwen3VlImageRuntimeInput;
    use mlx_sys::{eval, zeros};
    use serde_json::json;

    #[test]
    fn patch_buffer_length_mismatch_is_rejected() {
        // 2 patches * 4 dims = 8 buffer elements
        assert!(validate_qwen3_vl_patch_buffer_len(8, 2, 4).is_ok());
        let err = validate_qwen3_vl_patch_buffer_len(3, 2, 4).unwrap_err();
        assert!(
            matches!(err, Qwen3VlError::InvalidGeometry(msg) if msg.contains("patch buffer length")),
            "short patch buffer must fail closed before from_raw_data"
        );
    }

    #[test]
    fn qwen35_moe_aliases_default_to_qwen35_mrope_section() {
        // Holo3 / Ornith ship model_type=qwen3_5_moe and may omit mrope_section
        // on older snapshots. They must not inherit the Qwen3-VL [24, 20, 20]
        // fallback used for qwen3_vl*.
        for model_type in [
            "qwen3_5",
            "qwen3.5",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5_text",
        ] {
            let config = json!({
                "model_type": model_type,
                "text_config": {},
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 2,
                    "intermediate_size": 4,
                    "out_hidden_size": 2,
                    "num_heads": 1,
                    "in_channels": 3,
                    "patch_size": 2,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 1,
                    "num_position_embeddings": 4
                }
            });
            let (_, section) = parse_qwen_vision_config(&config).expect(model_type);
            assert_eq!(section, vec![11, 11, 10], "{model_type}");
        }

        let vl = json!({
            "model_type": "qwen3_vl",
            "text_config": {},
            "vision_config": {
                "depth": 1,
                "hidden_size": 2,
                "intermediate_size": 4,
                "out_hidden_size": 2,
                "num_heads": 1,
                "in_channels": 3,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 1,
                "num_position_embeddings": 4
            }
        });
        let (_, section) = parse_qwen_vision_config(&vl).expect("qwen3_vl");
        assert_eq!(section, vec![24, 20, 20]);
    }

    #[test]
    fn qwen36_moe_model_type_enters_visual_loader() {
        let config = json!({
            "model_type": "qwen3_5_moe",
            "text_config": {
                "rope_parameters": {"mrope_section": [1, 1, 1]}
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 2,
                "intermediate_size": 4,
                "out_hidden_size": 2,
                "num_heads": 1,
                "in_channels": 3,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 1,
                "num_position_embeddings": 4,
                "deepstack_visual_indexes": []
            }
        });
        let mut tensors = HashMap::from([(
            "vision_tower.patch_embed.proj.weight".to_string(),
            zeros(&[2, 3, 2, 2, 2], MlxDtype::Float32, None),
        )]);
        let result = load_qwen3_vl_vision_weights(&[], &mut tensors, Some(&config));
        assert!(
            matches!(result, Err(WeightLoadError::TensorMissing(_))),
            "a Qwen 3.6 config must enter, not bypass, the visual loader"
        );
    }

    #[test]
    fn qwen36_and_next_aliases_enter_visual_loader() {
        // DI-VL-A001: model_type qwen3.6 / qwen3_next / qwen3_6 must not silent-
        // skip the vision tower when patch weights are present.
        for model_type in ["qwen3.6", "qwen3_6", "qwen3_next"] {
            let config = json!({
                "model_type": model_type,
                "text_config": {
                    "rope_parameters": {"mrope_section": [1, 1, 1]}
                },
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 2,
                    "intermediate_size": 4,
                    "out_hidden_size": 2,
                    "num_heads": 1,
                    "in_channels": 3,
                    "patch_size": 2,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 1,
                    "num_position_embeddings": 4,
                    "deepstack_visual_indexes": []
                }
            });
            let mut tensors = HashMap::from([(
                "vision_tower.patch_embed.proj.weight".to_string(),
                zeros(&[2, 3, 2, 2, 2], MlxDtype::Float32, None),
            )]);
            let result = load_qwen3_vl_vision_weights(&[], &mut tensors, Some(&config));
            assert!(
                matches!(result, Err(WeightLoadError::TensorMissing(_))),
                "{model_type} must enter the visual loader, not return Ok(None)"
            );
        }
    }

    #[test]
    fn select_decode_route_media_covers_qwen_next_aliases() {
        assert_eq!(
            select_decode_route("qwen3_next", true).expect("route"),
            "qwen3_vl"
        );
        assert_eq!(
            select_decode_route("qwen3_next", false).expect("route"),
            "qwen3_next"
        );
        assert_eq!(
            select_decode_route("qwen3_vl_moe", true).expect("route"),
            "qwen3_vl_moe"
        );
        assert!(is_qwen3_vl_family("qwen3_next"));
        assert!(is_qwen3_vl_family("qwen3.6"));
    }

    #[test]
    fn geometry_and_mrope() {
        let geometry = Qwen3VlImageGeometry {
            height: 448,
            width: 448,
            patch_size: 16,
            spatial_merge_size: 2,
            max_soft_tokens: 1024,
        };
        assert_eq!(geometry.soft_token_count().unwrap(), 196);
        assert_eq!(plan_mrope_for_images(&[geometry]).unwrap().len(), 196 * 3);
        assert_eq!(deepstack_layers(3, 36), vec![0, 1, 2]);
    }

    #[test]
    fn soft_token_count_matches_grid_hw_on_non_merge_aligned_patch_grid() {
        // 3×3 patch grid with merge 2: soft tokens must be grid_h×grid_w (1),
        // not the product-overcount (9/4=2). MRoPE and scatter share this count.
        let geometry = Qwen3VlImageGeometry {
            height: 48,
            width: 48,
            patch_size: 16,
            spatial_merge_size: 2,
            max_soft_tokens: 1024,
        };
        let (gh, gw) = geometry.grid_hw().expect("grid");
        assert_eq!((gh, gw), (1, 1));
        assert_eq!(geometry.soft_token_count().unwrap(), gh * gw);
        assert_eq!(plan_mrope_for_images(&[geometry]).unwrap().len(), 3);
        // Collapsed merge axis must fail closed for both APIs.
        let collapsed = Qwen3VlImageGeometry {
            height: 16,
            width: 64,
            patch_size: 16,
            spatial_merge_size: 2,
            max_soft_tokens: 1024,
        };
        assert!(collapsed.grid_hw().is_err());
        assert!(collapsed.soft_token_count().is_err());
    }

    #[test]
    fn soft_token_count_fails_closed_when_grid_exceeds_max() {
        // 448² / 16 / 2 → 14×14 = 196 soft tokens. Capping max must not return
        // a partial count that disagrees with grid_hw / MRoPE length.
        let geometry = Qwen3VlImageGeometry {
            height: 448,
            width: 448,
            patch_size: 16,
            spatial_merge_size: 2,
            max_soft_tokens: 10,
        };
        let (gh, gw) = geometry.grid_hw().expect("grid");
        assert_eq!(gh * gw, 196);
        assert!(geometry.soft_token_count().is_err());
        assert!(plan_mrope_for_images(&[geometry]).is_err());
    }

    #[test]
    fn mrope_compresses_visual_run_and_preserves_three_axes() {
        let inputs = Qwen3VlRuntimeInputs {
            images: vec![Qwen3VlImageRuntimeInput {
                placeholder_index: 2,
                soft_token_count: 4,
                patches: vec![0.0; 16],
                num_patches: 16,
                patch_dim: 1,
                grid_t: 1,
                height: 4,
                width: 4,
                patch_size: 1,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                is_video: false,
            }],
        };
        let axes = qwen_mrope_position_axes(8, &inputs).unwrap();
        assert_eq!(axes[0], [0, 0, 0]);
        assert_eq!(axes[1], [1, 1, 1]);
        assert_eq!(axes[2], [2, 2, 2]);
        assert_eq!(axes[3], [2, 2, 3]);
        assert_eq!(axes[4], [2, 3, 2]);
        assert_eq!(axes[5], [2, 3, 3]);
        assert_eq!(axes[6], [4, 4, 4]);
    }

    #[test]
    fn interleaved_mrope_has_expected_shape() {
        let positions = vec![[0, 0, 0], [1, 2, 3]];
        let factors = build_interleaved_mrope(&positions, 8, 10_000.0, &[2, 1, 1]).unwrap();
        assert_eq!(factors.cos.shape(), vec![1, 2, 8]);
        assert_eq!(factors.sin.shape(), vec![1, 2, 8]);
    }

    #[test]
    fn scatter_and_deepstack_add_use_visual_positions() {
        let text = zeros(&[1, 4, 2], MlxDtype::Float32, None);
        let vision = f32_array(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let scattered = scatter_vision_into_text(&text, &vision, &[1, 2]).unwrap();
        let added = add_deepstack_into_text(&scattered, &vision, &[1, 2]).unwrap();
        eval(&[&added]);
        assert_eq!(added.shape(), vec![1, 4, 2]);
    }

    #[test]
    fn patch_embed_layouts_normalize_to_tchw() {
        let config = Qwen3VlVisionConfig {
            depth: 1,
            hidden_size: 2,
            intermediate_size: 4,
            out_hidden_size: 2,
            num_heads: 1,
            in_channels: 3,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 1,
            num_position_embeddings: 4,
            deepstack_visual_indexes: Vec::new(),
        };
        let raw = zeros(&[2, 3, 2, 2, 2], MlxDtype::Float32, None);
        let mlx = zeros(&[2, 2, 2, 2, 3], MlxDtype::Float32, None);
        assert_eq!(
            normalize_patch_embed_weight(&raw, &config).unwrap().shape(),
            vec![2, 24]
        );
        assert_eq!(
            normalize_patch_embed_weight(&mlx, &config).unwrap().shape(),
            vec![2, 24]
        );
    }
}
