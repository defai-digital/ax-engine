//! Standard Gemma 4 vision-language path.
//!
//! Gemma 4 checkpoints use a full bidirectional ViT followed by a learned
//! projection into the language embedding space. This is distinct from the
//! encoder-free `gemma4_unified` connector. The HTTP processor deliberately
//! keeps one common runtime tensor contract: each soft-token pixel block is
//! `patch_size * pooling_kernel_size` square. This module expands those blocks
//! back into the base ViT patches, runs the tower, pools each spatial group,
//! and scatters the projected features into the Gemma text sequence.

use std::collections::HashMap;

use ax_engine_core::NativeTensorSpec;
use ax_engine_core::gemma4_unified::{
    Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedRuntimeInputs, Gemma4UnifiedVideoRuntimeInput,
};
use ax_engine_core::vl_geometry::{scatter_merge_indices, vit_soft_token_count};
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, clip, concatenate, divide, gelu_approx, multiply, negative,
    repeat_axis, reshape, rms_norm, scaled_dot_product_attention, slice, subtract, sum_axis, take,
    transpose,
};
use serde_json::Value;
use thiserror::Error;

use crate::gemma4_unified::{
    Gemma4UnifiedChunkEmbeddings, Gemma4UnifiedError as UnifiedEmbedError, build_chunk_embeddings,
    overwrite_span, overwrite_video_spans, push_media_range, push_video_media_ranges,
};
use crate::model::shared::qw;
use crate::model::{ModelConfig, embed_tokens, scale_hidden_pub};
use crate::weights::{ModelWeights, QuantizedWeight, WeightLoadError, take_named_weight};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Gemma4VlError {
    #[error("standard Gemma 4 requires vision tower weights for image/video input")]
    MissingVisionWeights,
    #[error("standard Gemma 4 Conformer audio is not supported")]
    MissingAudioWeights,
    #[error("standard Gemma 4 image geometry invalid: {0}")]
    InvalidGeometry(String),
    #[error("standard Gemma 4 scatter merge failed: {0}")]
    Scatter(String),
    #[error("standard Gemma 4 embed failed: {0}")]
    Embed(String),
}

impl From<UnifiedEmbedError> for Gemma4VlError {
    fn from(value: UnifiedEmbedError) -> Self {
        match value {
            UnifiedEmbedError::MissingVisionWeights
            | UnifiedEmbedError::MissingVideoVisionWeights => Self::MissingVisionWeights,
            UnifiedEmbedError::MissingAudioWeights => Self::MissingAudioWeights,
            other => Self::Embed(other.to_string()),
        }
    }
}

/// Geometry used by the standard Gemma 4 image processor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4VlImageGeometry {
    pub height: u32,
    pub width: u32,
    pub patch_size: u32,
    pub merge_size: u32,
    pub max_soft_tokens: u32,
}

impl Gemma4VlImageGeometry {
    pub fn soft_token_count(self) -> Result<u32, Gemma4VlError> {
        vit_soft_token_count(
            self.height,
            self.width,
            self.patch_size,
            self.merge_size,
            self.max_soft_tokens,
        )
        .ok_or_else(|| {
            Gemma4VlError::InvalidGeometry(format!(
                "h={} w={} patch={} merge={} max={}",
                self.height, self.width, self.patch_size, self.merge_size, self.max_soft_tokens
            ))
        })
    }
}

/// Plan soft-token scatter positions for one or more images in a prompt.
pub fn plan_image_scatter(
    placeholder_positions: &[usize],
    geometries: &[Gemma4VlImageGeometry],
) -> Result<Vec<usize>, Gemma4VlError> {
    if placeholder_positions.len() != geometries.len() {
        return Err(Gemma4VlError::Scatter(format!(
            "placeholders {} != images {}",
            placeholder_positions.len(),
            geometries.len()
        )));
    }
    let counts = geometries
        .iter()
        .map(|geometry| geometry.soft_token_count())
        .collect::<Result<Vec<_>, _>>()?;
    scatter_merge_indices(placeholder_positions, &counts).map_err(Gemma4VlError::Scatter)
}

pub fn is_gemma4_vl_family(model_family: &str) -> bool {
    model_family == "gemma4_vl"
}

/// Text-only decode on a VL checkpoint reuses the standard Gemma 4 graph.
pub fn text_only_uses_standard_gemma4_path(model_family: &str, has_media: bool) -> bool {
    is_gemma4_vl_family(model_family) && !has_media
}

pub fn has_vision_tower(weights: &ModelWeights) -> bool {
    weights.gemma4_vl_vision.is_some() || weights.gemma4_unified_vision.is_some()
}

pub fn has_audio_tower(weights: &ModelWeights) -> bool {
    weights.gemma4_unified_audio.is_some()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4VlVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub use_clipped_linears: bool,
    pub standardize: bool,
    pub language_hidden_size: usize,
}

struct Gemma4VlClippableLinear {
    linear: QuantizedWeight,
    input_min: Option<MlxArray>,
    input_max: Option<MlxArray>,
    output_min: Option<MlxArray>,
    output_max: Option<MlxArray>,
}

impl Gemma4VlClippableLinear {
    fn load(
        specs: &[NativeTensorSpec],
        name_map: &mut HashMap<String, MlxArray>,
        prefix: &str,
        use_clipping: bool,
    ) -> Result<Self, WeightLoadError> {
        let (input_min, input_max) = take_clip_pair(name_map, prefix, "input", use_clipping)?;
        let (output_min, output_max) = take_clip_pair(name_map, prefix, "output", use_clipping)?;
        Ok(Self {
            linear: take_named_weight(specs, name_map, &format!("{prefix}.linear.weight"))?,
            input_min,
            input_max,
            output_min,
            output_max,
        })
    }

    fn forward(&self, input: &MlxArray) -> MlxArray {
        let clipped = clip_optional(input, self.input_min.as_ref(), self.input_max.as_ref());
        let output = qw(&clipped, &self.linear);
        clip_optional(&output, self.output_min.as_ref(), self.output_max.as_ref())
    }
}

struct Gemma4VlVisionLayerWeights {
    q_proj: Gemma4VlClippableLinear,
    k_proj: Gemma4VlClippableLinear,
    v_proj: Gemma4VlClippableLinear,
    o_proj: Gemma4VlClippableLinear,
    q_norm: MlxArray,
    k_norm: MlxArray,
    input_layernorm: MlxArray,
    post_attention_layernorm: MlxArray,
    pre_feedforward_layernorm: MlxArray,
    post_feedforward_layernorm: MlxArray,
    gate_proj: Gemma4VlClippableLinear,
    up_proj: Gemma4VlClippableLinear,
    down_proj: Gemma4VlClippableLinear,
}

/// Loaded standard Gemma 4 ViT and vision-to-language projection.
pub struct Gemma4VlVisionWeights {
    pub config: Gemma4VlVisionConfig,
    patch_projection: QuantizedWeight,
    position_embedding_table: MlxArray,
    layers: Vec<Gemma4VlVisionLayerWeights>,
    std_bias: Option<MlxArray>,
    std_scale: Option<MlxArray>,
    embedding_projection: QuantizedWeight,
}

pub fn load_gemma4_vl_vision_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    config_json: Option<&Value>,
) -> Result<Option<Gemma4VlVisionWeights>, WeightLoadError> {
    let Some(vision_prefix) = find_prefix(
        specs,
        name_map,
        &[
            "vision_tower.patch_embedder.input_proj.weight",
            "model.vision_tower.patch_embedder.input_proj.weight",
        ],
        ".patch_embedder.input_proj.weight",
    ) else {
        return Ok(None);
    };
    let config_json = config_json.ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "standard Gemma 4 vision checkpoint is missing config.json".into(),
        )
    })?;
    // convert packages may label model_type as gemma4_vl / gemma4-vl while
    // keeping the same encoder-ViT layout as model_type=gemma4.
    let model_type = config_json.get("model_type").and_then(Value::as_str);
    if !matches!(model_type, Some("gemma4" | "gemma4_vl" | "gemma4-vl")) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "vision_tower.* weights require model_type gemma4, gemma4_vl, or gemma4-vl (got {model_type:?})"
        )));
    }
    let config = parse_gemma4_vision_config(config_json)?;
    let projection_prefix = if name_map.contains_key("embed_vision.embedding_projection.weight") {
        "embed_vision"
    } else if name_map.contains_key("model.embed_vision.embedding_projection.weight") {
        "model.embed_vision"
    } else {
        return Err(WeightLoadError::TensorMissing(
            "embed_vision.embedding_projection.weight".into(),
        ));
    };

    let patch_projection = take_named_weight(
        specs,
        name_map,
        &format!("{vision_prefix}.patch_embedder.input_proj.weight"),
    )?;
    let position_embedding_table = take_plain_required(
        name_map,
        &format!("{vision_prefix}.patch_embedder.position_embedding_table"),
    )?;
    let mut layers = Vec::with_capacity(config.depth);
    for layer_index in 0..config.depth {
        let prefix = format!("{vision_prefix}.encoder.layers.{layer_index}");
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        layers.push(Gemma4VlVisionLayerWeights {
            q_proj: Gemma4VlClippableLinear::load(
                specs,
                name_map,
                &format!("{attention}.q_proj"),
                config.use_clipped_linears,
            )?,
            k_proj: Gemma4VlClippableLinear::load(
                specs,
                name_map,
                &format!("{attention}.k_proj"),
                config.use_clipped_linears,
            )?,
            v_proj: Gemma4VlClippableLinear::load(
                specs,
                name_map,
                &format!("{attention}.v_proj"),
                config.use_clipped_linears,
            )?,
            o_proj: Gemma4VlClippableLinear::load(
                specs,
                name_map,
                &format!("{attention}.o_proj"),
                config.use_clipped_linears,
            )?,
            q_norm: take_plain_required(name_map, &format!("{attention}.q_norm.weight"))?,
            k_norm: take_plain_required(name_map, &format!("{attention}.k_norm.weight"))?,
            input_layernorm: take_plain_required(
                name_map,
                &format!("{prefix}.input_layernorm.weight"),
            )?,
            post_attention_layernorm: take_plain_required(
                name_map,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            pre_feedforward_layernorm: take_plain_required(
                name_map,
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
            )?,
            post_feedforward_layernorm: take_plain_required(
                name_map,
                &format!("{prefix}.post_feedforward_layernorm.weight"),
            )?,
            gate_proj: Gemma4VlClippableLinear::load(
                specs,
                name_map,
                &format!("{mlp}.gate_proj"),
                config.use_clipped_linears,
            )?,
            up_proj: Gemma4VlClippableLinear::load(
                specs,
                name_map,
                &format!("{mlp}.up_proj"),
                config.use_clipped_linears,
            )?,
            down_proj: Gemma4VlClippableLinear::load(
                specs,
                name_map,
                &format!("{mlp}.down_proj"),
                config.use_clipped_linears,
            )?,
        });
    }

    let std_bias_name = format!("{vision_prefix}.std_bias");
    let std_scale_name = format!("{vision_prefix}.std_scale");
    let (std_bias, std_scale) = if config.standardize {
        (
            Some(take_plain_required(name_map, &std_bias_name)?),
            Some(take_plain_required(name_map, &std_scale_name)?),
        )
    } else {
        (
            name_map.remove(&std_bias_name),
            name_map.remove(&std_scale_name),
        )
    };
    let embedding_projection = take_named_weight(
        specs,
        name_map,
        &format!("{projection_prefix}.embedding_projection.weight"),
    )?;

    Ok(Some(Gemma4VlVisionWeights {
        config,
        patch_projection,
        position_embedding_table,
        layers,
        std_bias,
        std_scale,
        embedding_projection,
    }))
}

fn find_prefix<'a>(
    specs: &[NativeTensorSpec],
    name_map: &HashMap<String, MlxArray>,
    candidates: &[&'a str],
    suffix: &str,
) -> Option<&'a str> {
    candidates
        .iter()
        .find(|candidate| {
            name_map.contains_key(**candidate) && specs.iter().any(|spec| spec.name == **candidate)
        })
        .and_then(|candidate| candidate.strip_suffix(suffix))
}

fn parse_gemma4_vision_config(config: &Value) -> Result<Gemma4VlVisionConfig, WeightLoadError> {
    let vision = config.get("vision_config").ok_or_else(|| {
        WeightLoadError::InvalidLayer("Gemma 4 checkpoint has no vision_config".into())
    })?;
    let text = config.get("text_config").unwrap_or(config);
    let required = |value: &Value, key: &str| -> Result<usize, WeightLoadError> {
        value
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| {
                WeightLoadError::InvalidLayer(format!(
                    "Gemma 4 vision config field {key} is missing or invalid"
                ))
            })
    };
    let head_dim = required(vision, "head_dim")?;
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "Gemma 4 vision head_dim must be positive and divisible by 4, got {head_dim}"
        )));
    }
    let num_heads = required(vision, "num_attention_heads")?;
    let num_kv_heads = required(vision, "num_key_value_heads")?;
    if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "Gemma 4 vision heads {num_heads} must divide evenly by kv heads {num_kv_heads}"
        )));
    }
    let rope_theta = vision
        .get("rope_parameters")
        .and_then(|rope| rope.get("rope_theta"))
        .and_then(Value::as_f64)
        .unwrap_or(100.0) as f32;
    let rms_norm_eps = vision
        .get("rms_norm_eps")
        .and_then(Value::as_f64)
        .unwrap_or(1.0e-6) as f32;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(WeightLoadError::InvalidLayer(
            "Gemma 4 vision rope_theta must be positive".into(),
        ));
    }
    if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
        return Err(WeightLoadError::InvalidLayer(
            "Gemma 4 vision rms_norm_eps must be positive".into(),
        ));
    }
    Ok(Gemma4VlVisionConfig {
        depth: required(vision, "num_hidden_layers")?,
        hidden_size: required(vision, "hidden_size")?,
        intermediate_size: required(vision, "intermediate_size")?,
        num_heads,
        num_kv_heads,
        head_dim,
        patch_size: required(vision, "patch_size")?,
        pooling_kernel_size: required(vision, "pooling_kernel_size")?,
        position_embedding_size: required(vision, "position_embedding_size")?,
        rms_norm_eps,
        rope_theta,
        use_clipped_linears: vision
            .get("use_clipped_linears")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        standardize: vision
            .get("standardize")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        language_hidden_size: required(text, "hidden_size")?,
    })
}

fn take_plain_required(
    name_map: &mut HashMap<String, MlxArray>,
    name: &str,
) -> Result<MlxArray, WeightLoadError> {
    name_map
        .remove(name)
        .ok_or_else(|| WeightLoadError::TensorMissing(name.to_string()))
}

fn take_clip_pair(
    name_map: &mut HashMap<String, MlxArray>,
    prefix: &str,
    kind: &str,
    enabled: bool,
) -> Result<(Option<MlxArray>, Option<MlxArray>), WeightLoadError> {
    if !enabled {
        return Ok((None, None));
    }
    let min_name = format!("{prefix}.{kind}_min");
    let max_name = format!("{prefix}.{kind}_max");
    let min = name_map.remove(&min_name);
    let max = name_map.remove(&max_name);
    if min.is_some() != max.is_some() {
        return Err(WeightLoadError::InvalidLayer(format!(
            "Gemma 4 clipping requires both {min_name} and {max_name}"
        )));
    }
    Ok((min, max))
}

fn clip_optional(input: &MlxArray, min: Option<&MlxArray>, max: Option<&MlxArray>) -> MlxArray {
    match (min, max) {
        (Some(min), Some(max)) => {
            let min = astype(min, input.dtype(), None);
            let max = astype(max, input.dtype(), None);
            clip(input, &min, &max, None)
        }
        _ => input.clone(),
    }
}

/// Build multimodal prefill embeddings for either the standard ViT checkpoint
/// or the older encoder-free compatibility route.
pub(crate) fn build_vl_prefill_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    inputs: &Gemma4UnifiedRuntimeInputs,
) -> Result<Gemma4UnifiedChunkEmbeddings, Gemma4VlError> {
    let Some(vision) = weights.gemma4_vl_vision.as_ref() else {
        if (!inputs.images.is_empty() || !inputs.videos.is_empty())
            && weights.gemma4_unified_vision.is_none()
        {
            return Err(Gemma4VlError::MissingVisionWeights);
        }
        if !inputs.audios.is_empty() && weights.gemma4_unified_audio.is_none() {
            return Err(Gemma4VlError::MissingAudioWeights);
        }
        return Ok(build_chunk_embeddings(cfg, weights, token_ids, 0, inputs)?);
    };
    if !inputs.audios.is_empty() {
        return Err(Gemma4VlError::MissingAudioWeights);
    }
    if vision.config.language_hidden_size != cfg.hidden_size {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "vision projection targets hidden size {}, language model uses {}",
            vision.config.language_hidden_size, cfg.hidden_size
        )));
    }

    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = scale_hidden_pub(&hidden, scale);
    }
    let mut media_ranges = Vec::new();
    for image in &inputs.images {
        let features = embed_standard_image(vision, image, cfg.hidden_size)?;
        let features = astype(&features, hidden.dtype(), None);
        hidden = overwrite_span(hidden, &features, &image.span, 0, cfg.hidden_size)
            .map_err(|error| Gemma4VlError::Embed(error.to_string()))?;
        push_media_range(&mut media_ranges, &image.span);
    }
    for video in &inputs.videos {
        let features = embed_standard_video(vision, video, cfg.hidden_size)?;
        let features = astype(&features, hidden.dtype(), None);
        hidden = overwrite_video_spans(hidden, &features, video, 0, cfg.hidden_size)
            .map_err(|error| Gemma4VlError::Embed(error.to_string()))?;
        push_video_media_ranges(&mut media_ranges, video);
    }
    Ok(Gemma4UnifiedChunkEmbeddings {
        hidden,
        media_ranges,
    })
}

fn embed_standard_image(
    weights: &Gemma4VlVisionWeights,
    image: &Gemma4UnifiedImageRuntimeInput,
    language_hidden_size: usize,
) -> Result<MlxArray, Gemma4VlError> {
    if image.span.soft_token_count as usize != image.pixel_position_ids.len() {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "image span expects {} soft tokens but processor supplied {}",
            image.span.soft_token_count,
            image.pixel_position_ids.len()
        )));
    }
    embed_standard_pixels(
        weights,
        &image.pixel_values,
        &image.pixel_position_ids,
        language_hidden_size,
    )
}

fn embed_standard_video(
    weights: &Gemma4VlVisionWeights,
    video: &Gemma4UnifiedVideoRuntimeInput,
    language_hidden_size: usize,
) -> Result<MlxArray, Gemma4VlError> {
    let frame_count = video.frame_count as usize;
    if frame_count == 0 {
        return Err(Gemma4VlError::InvalidGeometry(
            "video frame_count must be greater than zero".into(),
        ));
    }
    if !video.pixel_position_ids.len().is_multiple_of(frame_count) {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "video patch count {} does not divide by {frame_count} frames",
            video.pixel_position_ids.len()
        )));
    }
    let soft_tokens_per_frame = video.pixel_position_ids.len() / frame_count;
    if soft_tokens_per_frame == 0 {
        return Err(Gemma4VlError::InvalidGeometry(
            "video contains no visual soft tokens".into(),
        ));
    }
    let effective_patch = weights
        .config
        .patch_size
        .checked_mul(weights.config.pooling_kernel_size)
        .ok_or_else(|| Gemma4VlError::InvalidGeometry("effective patch overflow".into()))?;
    let values_per_frame = soft_tokens_per_frame
        .checked_mul(effective_patch)
        .and_then(|value| value.checked_mul(effective_patch))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| Gemma4VlError::InvalidGeometry("video tensor size overflow".into()))?;
    if video.pixel_values.len() != values_per_frame * frame_count {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "video pixel value count {} != expected {}",
            video.pixel_values.len(),
            values_per_frame * frame_count
        )));
    }
    let expected_soft_tokens = if video.soft_token_ranges.is_empty() {
        video.span.soft_token_count as usize
    } else {
        video
            .soft_token_ranges
            .iter()
            .map(|range| range.soft_token_count as usize)
            .sum()
    };
    if expected_soft_tokens != video.pixel_position_ids.len() {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "video prompt expects {expected_soft_tokens} soft tokens but processor supplied {}",
            video.pixel_position_ids.len()
        )));
    }

    let mut frames = Vec::with_capacity(frame_count);
    for frame in 0..frame_count {
        let value_start = frame * values_per_frame;
        let position_start = frame * soft_tokens_per_frame;
        frames.push(embed_standard_pixels(
            weights,
            &video.pixel_values[value_start..value_start + values_per_frame],
            &video.pixel_position_ids[position_start..position_start + soft_tokens_per_frame],
            language_hidden_size,
        )?);
    }
    let refs = frames.iter().collect::<Vec<_>>();
    Ok(concatenate(&refs, 0, None))
}

fn embed_standard_pixels(
    weights: &Gemma4VlVisionWeights,
    pixel_values: &[f32],
    soft_positions: &[[i32; 2]],
    language_hidden_size: usize,
) -> Result<MlxArray, Gemma4VlError> {
    let unpacked = unpack_soft_token_blocks(&weights.config, pixel_values, soft_positions)?;
    let base_patch_count = unpacked.positions.len();
    let patch_dim = weights
        .config
        .patch_size
        .checked_mul(weights.config.patch_size)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| Gemma4VlError::InvalidGeometry("base patch size overflow".into()))?;
    let pixel = f32_array(
        &unpacked.values,
        &[1, base_patch_count as i32, patch_dim as i32],
    );
    let pixel = astype(&pixel, weights.position_embedding_table.dtype(), None);
    let mut hidden = qw(&pixel, &weights.patch_projection);
    hidden = add(
        &hidden,
        &position_embeddings(
            &weights.position_embedding_table,
            &unpacked.positions,
            weights.config.hidden_size,
            weights.config.position_embedding_size,
        )?,
        None,
    );
    let rope = Gemma4VisionRope::new(
        &unpacked.positions,
        weights.config.head_dim,
        weights.config.rope_theta,
    );
    for layer in &weights.layers {
        hidden = vision_layer(&hidden, layer, &weights.config, &rope);
    }

    let pool = weights.config.pooling_kernel_size;
    let pool_area = pool * pool;
    let soft_count = soft_positions.len();
    let hidden = reshape(
        &hidden,
        &[
            1,
            soft_count as i32,
            pool_area as i32,
            weights.config.hidden_size as i32,
        ],
        None,
    );
    let hidden = sum_axis(&hidden, 2, false, None);
    let hidden = divide(
        &hidden,
        &mlx_sys::ops::cached_scalar(pool_area as f32, hidden.dtype()),
        None,
    );
    let mut hidden = multiply(
        &hidden,
        &mlx_sys::ops::cached_scalar((weights.config.hidden_size as f32).sqrt(), hidden.dtype()),
        None,
    );
    if let (Some(std_bias), Some(std_scale)) = (&weights.std_bias, &weights.std_scale) {
        let std_bias = astype(std_bias, hidden.dtype(), None);
        let std_scale = astype(std_scale, hidden.dtype(), None);
        hidden = multiply(&subtract(&hidden, &std_bias, None), &std_scale, None);
    }
    hidden = rms_norm(&hidden, None, weights.config.rms_norm_eps, None);
    let projected = qw(&hidden, &weights.embedding_projection);
    Ok(reshape(
        &projected,
        &[soft_count as i32, language_hidden_size as i32],
        None,
    ))
}

struct UnpackedPatches {
    values: Vec<f32>,
    positions: Vec<[u32; 2]>,
}

fn unpack_soft_token_blocks(
    config: &Gemma4VlVisionConfig,
    pixel_values: &[f32],
    soft_positions: &[[i32; 2]],
) -> Result<UnpackedPatches, Gemma4VlError> {
    if soft_positions.is_empty() {
        return Err(Gemma4VlError::InvalidGeometry(
            "image contains no soft-token patches".into(),
        ));
    }
    let patch = config.patch_size;
    let pool = config.pooling_kernel_size;
    if patch == 0 || pool == 0 {
        return Err(Gemma4VlError::InvalidGeometry(
            "patch_size and pooling_kernel_size must be positive".into(),
        ));
    }
    let effective = patch
        .checked_mul(pool)
        .ok_or_else(|| Gemma4VlError::InvalidGeometry("effective patch overflow".into()))?;
    let block_dim = effective
        .checked_mul(effective)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| Gemma4VlError::InvalidGeometry("soft-token block overflow".into()))?;
    let expected = soft_positions
        .len()
        .checked_mul(block_dim)
        .ok_or_else(|| Gemma4VlError::InvalidGeometry("pixel tensor size overflow".into()))?;
    if pixel_values.len() != expected {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "pixel value count {} != {} soft blocks * {block_dim}",
            pixel_values.len(),
            soft_positions.len()
        )));
    }

    let base_patch_count = soft_positions.len() * pool * pool;
    let base_patch_dim = patch * patch * 3;
    let mut values = Vec::with_capacity(base_patch_count * base_patch_dim);
    let mut positions = Vec::with_capacity(base_patch_count);
    for (block_index, [soft_x, soft_y]) in soft_positions.iter().copied().enumerate() {
        if soft_x < 0 || soft_y < 0 {
            return Err(Gemma4VlError::InvalidGeometry(format!(
                "soft position {block_index} is negative: [{soft_x}, {soft_y}]"
            )));
        }
        let soft_x = soft_x as usize;
        let soft_y = soft_y as usize;
        let block_start = block_index * block_dim;
        for inner_y in 0..pool {
            for inner_x in 0..pool {
                let base_x = soft_x * pool + inner_x;
                let base_y = soft_y * pool + inner_y;
                if base_x >= config.position_embedding_size
                    || base_y >= config.position_embedding_size
                {
                    return Err(Gemma4VlError::InvalidGeometry(format!(
                        "base patch position [{base_x}, {base_y}] exceeds position table {}",
                        config.position_embedding_size
                    )));
                }
                positions.push([base_x as u32, base_y as u32]);
                for row in 0..patch {
                    let source_row = inner_y * patch + row;
                    for col in 0..patch {
                        let source_col = inner_x * patch + col;
                        let source = block_start + (source_row * effective + source_col) * 3;
                        values.extend(
                            pixel_values[source..source + 3]
                                .iter()
                                .map(|value| 2.0 * *value - 1.0),
                        );
                    }
                }
            }
        }
    }
    Ok(UnpackedPatches { values, positions })
}

fn position_embeddings(
    table: &MlxArray,
    positions: &[[u32; 2]],
    hidden_size: usize,
    position_size: usize,
) -> Result<MlxArray, Gemma4VlError> {
    let shape = table.shape();
    if shape != [2, position_size as i32, hidden_size as i32] {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "position table shape {shape:?} != [2, {position_size}, {hidden_size}]"
        )));
    }
    let x_table = reshape(
        &slice(
            table,
            &[0, 0, 0],
            &[1, position_size as i32, hidden_size as i32],
            &[1, 1, 1],
            None,
        ),
        &[position_size as i32, hidden_size as i32],
        None,
    );
    let y_table = reshape(
        &slice(
            table,
            &[1, 0, 0],
            &[2, position_size as i32, hidden_size as i32],
            &[1, 1, 1],
            None,
        ),
        &[position_size as i32, hidden_size as i32],
        None,
    );
    let x = positions
        .iter()
        .map(|position| position[0])
        .collect::<Vec<_>>();
    let y = positions
        .iter()
        .map(|position| position[1])
        .collect::<Vec<_>>();
    let position = add(
        &take(&x_table, &u32_array(&x), 0, None),
        &take(&y_table, &u32_array(&y), 0, None),
        None,
    );
    Ok(reshape(
        &position,
        &[1, positions.len() as i32, hidden_size as i32],
        None,
    ))
}

struct Gemma4VisionRope {
    cos_x: MlxArray,
    sin_x: MlxArray,
    cos_y: MlxArray,
    sin_y: MlxArray,
    channels_per_dimension: usize,
}

impl Gemma4VisionRope {
    fn new(positions: &[[u32; 2]], head_dim: usize, theta: f32) -> Self {
        let channels = 2 * (head_dim / 4);
        let half = channels / 2;
        let timescales = (0..half)
            .map(|index| theta.powf((2 * index) as f32 / channels as f32))
            .collect::<Vec<_>>();
        let build = |axis: usize| {
            let mut cos = Vec::with_capacity(positions.len() * channels);
            let mut sin = Vec::with_capacity(positions.len() * channels);
            for position in positions {
                for _ in 0..2 {
                    for timescale in &timescales {
                        let angle = position[axis] as f32 / *timescale;
                        cos.push(angle.cos());
                        sin.push(angle.sin());
                    }
                }
            }
            (
                f32_array(&cos, &[1, positions.len() as i32, 1, channels as i32]),
                f32_array(&sin, &[1, positions.len() as i32, 1, channels as i32]),
            )
        };
        let (cos_x, sin_x) = build(0);
        let (cos_y, sin_y) = build(1);
        Self {
            cos_x,
            sin_x,
            cos_y,
            sin_y,
            channels_per_dimension: channels,
        }
    }

    fn apply(&self, input: &MlxArray) -> MlxArray {
        let shape = input.shape();
        let channels = self.channels_per_dimension as i32;
        let x = slice(
            input,
            &[0, 0, 0, 0],
            &[shape[0], shape[1], shape[2], channels],
            &[1, 1, 1, 1],
            None,
        );
        let y = slice(
            input,
            &[0, 0, 0, channels],
            &[shape[0], shape[1], shape[2], channels * 2],
            &[1, 1, 1, 1],
            None,
        );
        let apply_axis = |value: &MlxArray, cos: &MlxArray, sin: &MlxArray| {
            let cos = astype(cos, value.dtype(), None);
            let sin = astype(sin, value.dtype(), None);
            add(
                &multiply(value, &cos, None),
                &multiply(&rotate_half(value), &sin, None),
                None,
            )
        };
        concatenate(
            &[
                &apply_axis(&x, &self.cos_x, &self.sin_x),
                &apply_axis(&y, &self.cos_y, &self.sin_y),
            ],
            -1,
            None,
        )
    }
}

fn vision_layer(
    input: &MlxArray,
    layer: &Gemma4VlVisionLayerWeights,
    config: &Gemma4VlVisionConfig,
    rope: &Gemma4VisionRope,
) -> MlxArray {
    let normalized = rms_norm(
        input,
        Some(&layer.input_layernorm),
        config.rms_norm_eps,
        None,
    );
    let attention = vision_attention(&normalized, layer, config, rope);
    let attention = rms_norm(
        &attention,
        Some(&layer.post_attention_layernorm),
        config.rms_norm_eps,
        None,
    );
    let hidden = add(input, &attention, None);
    let normalized = rms_norm(
        &hidden,
        Some(&layer.pre_feedforward_layernorm),
        config.rms_norm_eps,
        None,
    );
    let gate = layer.gate_proj.forward(&normalized);
    let up = layer.up_proj.forward(&normalized);
    let feed_forward = layer
        .down_proj
        .forward(&multiply(&gelu_approx(&gate, None), &up, None));
    let feed_forward = rms_norm(
        &feed_forward,
        Some(&layer.post_feedforward_layernorm),
        config.rms_norm_eps,
        None,
    );
    add(&hidden, &feed_forward, None)
}

fn vision_attention(
    input: &MlxArray,
    layer: &Gemma4VlVisionLayerWeights,
    config: &Gemma4VlVisionConfig,
    rope: &Gemma4VisionRope,
) -> MlxArray {
    let shape = input.shape();
    let batch = shape[0];
    let sequence = shape[1];
    let q = reshape(
        &layer.q_proj.forward(input),
        &[
            batch,
            sequence,
            config.num_heads as i32,
            config.head_dim as i32,
        ],
        None,
    );
    let k = reshape(
        &layer.k_proj.forward(input),
        &[
            batch,
            sequence,
            config.num_kv_heads as i32,
            config.head_dim as i32,
        ],
        None,
    );
    let v = reshape(
        &layer.v_proj.forward(input),
        &[
            batch,
            sequence,
            config.num_kv_heads as i32,
            config.head_dim as i32,
        ],
        None,
    );
    let q = rope.apply(&rms_norm(
        &q,
        Some(&layer.q_norm),
        config.rms_norm_eps,
        None,
    ));
    let k = rope.apply(&rms_norm(
        &k,
        Some(&layer.k_norm),
        config.rms_norm_eps,
        None,
    ));
    let v = rms_norm(&v, None, config.rms_norm_eps, None);
    let q = transpose(&q, &[0, 2, 1, 3], None);
    let mut k = transpose(&k, &[0, 2, 1, 3], None);
    let mut v = transpose(&v, &[0, 2, 1, 3], None);
    if config.num_heads != config.num_kv_heads {
        let repeats = (config.num_heads / config.num_kv_heads) as i32;
        k = repeat_axis(&k, repeats, 1, None);
        v = repeat_axis(&v, repeats, 1, None);
    }
    let attention = scaled_dot_product_attention(&q, &k, &v, 1.0, false, None);
    let attention = reshape(
        &transpose(&attention, &[0, 2, 1, 3], None),
        &[batch, sequence, (config.num_heads * config.head_dim) as i32],
        None,
    );
    layer.o_proj.forward(&attention)
}

fn rotate_half(input: &MlxArray) -> MlxArray {
    let shape = input.shape();
    let ndim = shape.len();
    let half = shape[ndim - 1] / 2;
    let mut starts = vec![0; ndim];
    let mut stops = shape.clone();
    stops[ndim - 1] = half;
    let first = slice(input, &starts, &stops, &vec![1; ndim], None);
    starts[ndim - 1] = half;
    stops[ndim - 1] = shape[ndim - 1];
    let second = slice(input, &starts, &stops, &vec![1; ndim], None);
    concatenate(&[&negative(&second, None), &first], -1, None)
}

fn f32_array(values: &[f32], shape: &[i32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        shape,
        MlxDtype::Float32,
    )
}

fn u32_array(values: &[u32]) -> MlxArray {
    MlxArray::from_raw_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
        &[values.len() as i32],
        MlxDtype::Uint32,
    )
}

/// Validate a single image tensor against its declared soft-token geometry.
pub fn validate_image_soft_tokens(
    image: &Gemma4UnifiedImageRuntimeInput,
    geometry: Gemma4VlImageGeometry,
) -> Result<(), Gemma4VlError> {
    let expected = geometry.soft_token_count()?;
    if image.span.soft_token_count != expected {
        return Err(Gemma4VlError::InvalidGeometry(format!(
            "soft_token_count {} != geometry {}",
            image.span.soft_token_count, expected
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::gemma4_unified::{Gemma4UnifiedModality, Gemma4UnifiedTokenSpan};
    use serde_json::json;

    #[test]
    fn soft_token_and_scatter_plan() {
        let geometry = Gemma4VlImageGeometry {
            height: 224,
            width: 224,
            patch_size: 16,
            merge_size: 3,
            max_soft_tokens: 280,
        };
        assert_eq!(geometry.soft_token_count().unwrap(), 21);
        let indices = plan_image_scatter(&[3], &[geometry]).unwrap();
        assert_eq!(indices, (3..24).collect::<Vec<_>>());
    }

    #[test]
    fn text_only_route() {
        assert!(text_only_uses_standard_gemma4_path("gemma4_vl", false));
        assert!(!text_only_uses_standard_gemma4_path("gemma4_vl", true));
        assert!(!text_only_uses_standard_gemma4_path("gemma4", false));
    }

    #[test]
    fn parses_standard_gemma4_vision_config() {
        let config = json!({
            "model_type": "gemma4",
            "text_config": {"hidden_size": 1536},
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 16,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "head_dim": 64,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "position_embedding_size": 10240,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 100.0},
                "use_clipped_linears": true,
                "standardize": false
            }
        });
        let parsed = parse_gemma4_vision_config(&config).unwrap();
        assert_eq!(parsed.depth, 16);
        assert_eq!(parsed.language_hidden_size, 1536);
        assert!(parsed.use_clipped_linears);
        assert!(!parsed.standardize);
    }

    #[test]
    fn stale_text_manifest_does_not_trigger_vision_loading() {
        let mut name_map = HashMap::from([(
            "vision_tower.patch_embedder.input_proj.weight".to_string(),
            MlxArray::from_f32_slice(&[1.0]),
        )]);

        let loaded = load_gemma4_vl_vision_weights(&[], &mut name_map, None)
            .expect("undeclared tower tensors should be ignored for text-only compatibility");

        assert!(loaded.is_none());
        assert!(
            name_map.contains_key("vision_tower.patch_embedder.input_proj.weight"),
            "skipped tower data must remain untouched"
        );
    }

    #[test]
    fn load_accepts_gemma4_vl_model_type_for_vision_tower() {
        // convert leaves config.json model_type as gemma4_vl while mapping
        // family gemma4_vl. Vision load must not hard-require model_type=gemma4.
        use ax_engine_core::{NativeTensorDataType, NativeTensorRole, NativeTensorSpec};
        use std::path::PathBuf;

        let tower_name = "vision_tower.patch_embedder.input_proj.weight";
        let specs = [NativeTensorSpec {
            name: tower_name.to_string(),
            role: NativeTensorRole::Other,
            layer_index: None,
            dtype: NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![1],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 4,
        }];
        let mut name_map =
            HashMap::from([(tower_name.to_string(), MlxArray::from_f32_slice(&[1.0]))]);
        let config = json!({
            "model_type": "gemma4_vl",
            "text_config": {"hidden_size": 1536},
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 1,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "head_dim": 64,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "position_embedding_size": 10240,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_theta": 100.0},
                "use_clipped_linears": true,
                "standardize": false
            }
        });
        match load_gemma4_vl_vision_weights(&specs, &mut name_map, Some(&config)) {
            Ok(Some(_)) => panic!("incomplete tower must not fully load"),
            Ok(None) => panic!("vision prefix present must attempt load"),
            Err(err) => {
                let message = err.to_string();
                assert!(
                    !message.contains("require model_type")
                        && !message.contains("got Some(\"qwen3\")"),
                    "gemma4_vl must pass model_type gate, got: {message}"
                );
                // Next failure is missing embed_vision projection (real load path).
                assert!(
                    message.contains("embed_vision"),
                    "expected to progress past model_type to embed_vision lookup, got: {message}"
                );
            }
        }

        // Non-Gemma model_type still rejected.
        let mut name_map =
            HashMap::from([(tower_name.to_string(), MlxArray::from_f32_slice(&[1.0]))]);
        let bad = json!({
            "model_type": "qwen3",
            "text_config": {"hidden_size": 1536},
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 1,
                "num_attention_heads": 12,
                "num_key_value_heads": 12,
                "head_dim": 64,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "position_embedding_size": 10240
            }
        });
        match load_gemma4_vl_vision_weights(&specs, &mut name_map, Some(&bad)) {
            Ok(_) => panic!("qwen3 must not load via Gemma VL vision path"),
            Err(err) => {
                assert!(
                    err.to_string().contains("model_type"),
                    "expected model_type rejection, got: {err}"
                );
            }
        }
    }

    #[test]
    fn expands_soft_blocks_into_spatial_base_patches() {
        let config = Gemma4VlVisionConfig {
            depth: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            patch_size: 2,
            pooling_kernel_size: 2,
            position_embedding_size: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 100.0,
            use_clipped_linears: false,
            standardize: false,
            language_hidden_size: 4,
        };
        let pixels = (0..4 * 4 * 3)
            .map(|index| index as f32 / 100.0)
            .collect::<Vec<_>>();
        let unpacked = unpack_soft_token_blocks(&config, &pixels, &[[2, 3]]).unwrap();
        assert_eq!(unpacked.positions, vec![[4, 6], [5, 6], [4, 7], [5, 7]]);
        assert_eq!(unpacked.values.len(), pixels.len());
        assert_eq!(unpacked.values[0], -1.0);
        assert_eq!(unpacked.values[6], 2.0 * pixels[12] - 1.0);
    }

    #[test]
    fn validate_image_soft_tokens_enforces_geometry() {
        let geometry = Gemma4VlImageGeometry {
            height: 224,
            width: 224,
            patch_size: 16,
            merge_size: 3,
            max_soft_tokens: 280,
        };
        let image = Gemma4UnifiedImageRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                modality: Gemma4UnifiedModality::Image,
                placeholder_index: 0,
                replacement_start: 0,
                soft_token_count: 21,
                replacement_token_count: 23,
            },
            pixel_values: Vec::new(),
            pixel_position_ids: Vec::new(),
        };
        assert!(validate_image_soft_tokens(&image, geometry).is_ok());
        let bad = Gemma4UnifiedImageRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                soft_token_count: 8,
                ..image.span
            },
            ..image
        };
        assert!(validate_image_soft_tokens(&bad, geometry).is_err());
    }
}
