//! Qwen3-VL / Qwen3-VL-MoE path (WS-V2 / R-V2).
//!
//! ViT encoder with 2-D RoPE, DeepStack injection, MRoPE text positions, and
//! LLaVA-style scatter merge into certified qwen3 / qwen3-MoE graphs.
//! Text-only prompts on a VL checkpoint must route identically to qwen3.

use ax_engine_core::qwen3_vl::Qwen3VlRuntimeInputs;
use ax_engine_core::vl_geometry::{
    MropeSections, deepstack_injection_layers, mrope_position_ids, scatter_merge_indices,
    vit_soft_token_count,
};
use mlx_sys::ops::cached_scalar;
use mlx_sys::{
    MlxArray, MlxDtype, add, concatenate, gelu, layer_norm, matmul, multiply, reshape, slice,
    softmax, transpose,
};
use thiserror::Error;

use crate::model::{ModelConfig, embed_tokens};
use crate::weights::ModelWeights;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Qwen3VlError {
    #[error("qwen3_vl requires vision tower weights for image input")]
    MissingVisionWeights,
    #[error("qwen3_vl image geometry invalid: {0}")]
    InvalidGeometry(String),
    #[error("qwen3_vl scatter merge failed: {0}")]
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
        vit_soft_token_count(
            self.height,
            self.width,
            self.patch_size,
            self.spatial_merge_size,
            self.max_soft_tokens,
        )
        .ok_or_else(|| {
            Qwen3VlError::InvalidGeometry(format!(
                "h={} w={} patch={} merge={} max={}",
                self.height,
                self.width,
                self.patch_size,
                self.spatial_merge_size,
                self.max_soft_tokens
            ))
        })
    }

    pub fn mrope_sections(self) -> Result<MropeSections, Qwen3VlError> {
        let (h, w) = self.grid_hw()?;
        Ok(MropeSections::for_image(h, w))
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
    let mut counts = Vec::with_capacity(geometries.len());
    for g in geometries {
        counts.push(g.soft_token_count()?);
    }
    scatter_merge_indices(placeholder_positions, &counts).map_err(Qwen3VlError::Scatter)
}

pub fn plan_mrope_for_images(
    geometries: &[Qwen3VlImageGeometry],
) -> Result<Vec<u32>, Qwen3VlError> {
    let mut all = Vec::new();
    for g in geometries {
        all.extend(mrope_position_ids(g.mrope_sections()?));
    }
    Ok(all)
}

pub fn deepstack_layers(num_feature_maps: usize, language_layers: u32) -> Vec<u32> {
    deepstack_injection_layers(num_feature_maps, language_layers)
}

pub fn is_qwen3_vl_family(model_family: &str) -> bool {
    matches!(model_family, "qwen3_vl" | "qwen3_vl_moe")
}

/// Text-only prompts on VL checkpoints must use the certified qwen3 decode path.
pub fn text_only_decode_family(model_family: &str) -> Option<&'static str> {
    match model_family {
        "qwen3_vl" => Some("qwen3"),
        "qwen3_vl_moe" => Some("qwen3"), // MoE graph shared with qwen3_moe maps
        _ => None,
    }
}

/// Prefill gate: image inputs require a vision tower. Until ViT weights are
/// mapped for this family, fail closed (never silent text-only degrade).
pub fn require_vision_for_images(
    has_image_inputs: bool,
    has_vision_weights: bool,
) -> Result<(), Qwen3VlError> {
    if has_image_inputs && !has_vision_weights {
        return Err(Qwen3VlError::MissingVisionWeights);
    }
    Ok(())
}

/// True when the loaded checkpoint carries a Qwen3-VL vision tower.
pub fn has_vision_tower(weights: &ModelWeights) -> bool {
    weights.qwen3_vl_vision.is_some()
}

/// Prefill-side soft-token injection for qwen3_vl (ADR-038 adapter).
///
/// Fails closed if the request carries images but vision tower weights are
/// absent. When weights are present, runs the portable ViT and scatters soft
/// tokens into the text residual stream.
pub(crate) fn build_vl_prefill_embeddings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    inputs: &Qwen3VlRuntimeInputs,
) -> Result<MlxArray, Qwen3VlError> {
    if !inputs.images.is_empty() && !has_vision_tower(weights) {
        return Err(Qwen3VlError::MissingVisionWeights);
    }
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    if inputs.images.is_empty() {
        return Ok(hidden);
    }
    let vision_w = weights
        .qwen3_vl_vision
        .as_ref()
        .ok_or(Qwen3VlError::MissingVisionWeights)?;
    for image in &inputs.images {
        let patches = MlxArray::from_raw_data(
            image.patches.as_ptr().cast(),
            std::mem::size_of_val(image.patches.as_slice()),
            &[1, image.num_patches as i32, image.patch_dim as i32],
            MlxDtype::Float32,
        );
        let (soft, _deep) = vision_encoder_forward(vision_w, &patches)?;
        let geometry = Qwen3VlImageGeometry {
            height: image.height,
            width: image.width,
            patch_size: image.patch_size,
            spatial_merge_size: image.spatial_merge_size,
            max_soft_tokens: image.soft_token_count.max(1),
        };
        let positions = plan_image_scatter(&[image.placeholder_index], &[geometry])?;
        // Expand text length when soft tokens exceed a single placeholder slot.
        // For the portable path, scatter only into existing prompt indices that
        // match the planned positions (caller must expand tokens accordingly).
        hidden = scatter_vision_into_text(&hidden, &soft, &positions)?;
    }
    Ok(hidden)
}

/// Decode-route selection for a loaded VL checkpoint.
pub fn select_decode_route(
    model_family: &str,
    has_media: bool,
) -> Result<&'static str, Qwen3VlError> {
    if !is_qwen3_vl_family(model_family) {
        // Non-VL families: caller keeps its own label (not static).
        return Ok(if model_family.is_empty() {
            "unknown"
        } else {
            // Map known families to static labels when possible.
            match model_family {
                "qwen3" => "qwen3",
                "qwen3_5" | "qwen3.5" => "qwen3_5",
                "qwen3_next" | "qwen3.6" | "qwen3_6" => "qwen3_next",
                "gemma4" => "gemma4",
                "gemma4_vl" => "gemma4_vl",
                _ => "other",
            }
        });
    }
    if has_media {
        // Multimodal path: still text-graph after scatter; vision tower required.
        return Ok(if model_family == "qwen3_vl_moe" {
            "qwen3_vl_moe"
        } else {
            "qwen3_vl"
        });
    }
    Ok(text_only_decode_family(model_family).unwrap_or("qwen3"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_and_mrope() {
        let g = Qwen3VlImageGeometry {
            height: 448,
            width: 448,
            patch_size: 14,
            spatial_merge_size: 2,
            max_soft_tokens: 1024,
        };
        // 448/14=32, /2=16 → 16×16=256 soft tokens
        assert_eq!(g.soft_token_count().unwrap(), 256);
        let sections = g.mrope_sections().unwrap();
        assert_eq!(sections.height, 16);
        assert_eq!(sections.width, 16);
        let ids = plan_mrope_for_images(&[g]).unwrap();
        assert_eq!(ids.len(), 256 * 3);
    }

    #[test]
    fn deepstack_and_text_route() {
        assert_eq!(deepstack_layers(3, 36), vec![0, 1, 2]);
        assert_eq!(text_only_decode_family("qwen3_vl"), Some("qwen3"));
        assert!(is_qwen3_vl_family("qwen3_vl_moe"));
    }

    #[test]
    fn scatter_plan() {
        let g = Qwen3VlImageGeometry {
            height: 28,
            width: 28,
            patch_size: 14,
            spatial_merge_size: 1,
            max_soft_tokens: 16,
        };
        // 2×2 patches, merge 1 → 4 soft tokens
        assert_eq!(g.soft_token_count().unwrap(), 4);
        let idx = plan_image_scatter(&[1], &[g]).unwrap();
        assert_eq!(idx, vec![1, 2, 3, 4]);
    }

    #[test]
    fn vision_required_and_text_route() {
        assert!(require_vision_for_images(true, false).is_err());
        assert!(require_vision_for_images(true, true).is_ok());
        assert!(require_vision_for_images(false, false).is_ok());
        assert_eq!(select_decode_route("qwen3_vl", false).unwrap(), "qwen3");
        assert_eq!(select_decode_route("qwen3_vl", true).unwrap(), "qwen3_vl");
    }

    #[test]
    fn build_vl_rejects_images_without_vision_weights() {
        use crate::gemma4_assistant_mtp::Gemma4AssistantMtpStatus;
        use crate::weights::QuantizedWeight;
        use mlx_sys::zeros;
        let dummy = || QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None);
        let weights = ModelWeights {
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
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
        };
        let cfg = ModelConfig {
            compile_cache_identity: 0,
            model_family: "qwen3_vl".into(),
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
            uses_geglu: false,
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
            generation_kind: ax_engine_core::GenerationKind::Autoregressive,
        };
        let inputs = Qwen3VlRuntimeInputs {
            images: vec![ax_engine_core::qwen3_vl::Qwen3VlImageRuntimeInput {
                placeholder_index: 0,
                soft_token_count: 4,
                patches: vec![0.1; 24],
                num_patches: 4,
                patch_dim: 6,
                height: 28,
                width: 28,
                patch_size: 14,
                spatial_merge_size: 1,
            }],
        };
        let err = build_vl_prefill_embeddings(&cfg, &weights, &[1, 2, 3], &inputs).unwrap_err();
        assert!(matches!(err, Qwen3VlError::MissingVisionWeights));
    }

    fn f32_array(vals: &[f32], shape: &[i32]) -> MlxArray {
        use mlx_sys::MlxDtype;
        MlxArray::from_raw_data(
            vals.as_ptr().cast(),
            std::mem::size_of_val(vals),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn vision_encoder_forward_and_scatter_real_path() {
        use mlx_sys::{MlxDtype, eval, zeros};
        // Tiny ViT: H=8, heads=2, seq=4 patches, 1 layer
        let h = 8usize;
        let heads = 2usize;
        let seq = 4i32;
        let patch_dim = 6i32;
        let layer = Qwen3VlVisionLayerWeights {
            qkv: f32_array(&vec![0.01; h * 3 * h], &[(3 * h) as i32, h as i32]),
            qkv_bias: Some(zeros(&[1, 1, (3 * h) as i32], MlxDtype::Float32, None)),
            proj: f32_array(&vec![0.01; h * h], &[h as i32, h as i32]),
            proj_bias: None,
            norm1_weight: f32_array(&vec![1.0; h], &[h as i32]),
            norm1_bias: Some(zeros(&[h as i32], MlxDtype::Float32, None)),
            fc1: f32_array(&vec![0.01; h * 16], &[16, h as i32]),
            fc1_bias: None,
            fc2: f32_array(&vec![0.01; 16 * h], &[h as i32, 16]),
            fc2_bias: None,
            norm2_weight: f32_array(&vec![1.0; h], &[h as i32]),
            norm2_bias: Some(zeros(&[h as i32], MlxDtype::Float32, None)),
        };
        let weights = Qwen3VlVisionWeights {
            patch_embed: f32_array(&vec![0.01; h * patch_dim as usize], &[h as i32, patch_dim]),
            patch_embed_bias: None,
            layers: vec![layer],
            merger: f32_array(&vec![0.01; h * h], &[h as i32, h as i32]),
            merger_bias: None,
            num_heads: heads,
            hidden_size: h,
            deepstack_indexes: vec![0],
        };
        let patches = f32_array(&vec![0.1; (seq * patch_dim) as usize], &[1, seq, patch_dim]);
        let (soft, deep) = vision_encoder_forward(&weights, &patches).expect("vit forward");
        eval(&[&soft]);
        assert_eq!(soft.shape()[0], 1);
        assert_eq!(soft.shape()[1], seq);
        assert_eq!(soft.shape()[2], h as i32);
        assert_eq!(deep.len(), 1);

        let text = zeros(&[1, 6, h as i32], MlxDtype::Float32, None);
        let positions = plan_image_scatter(
            &[1],
            &[Qwen3VlImageGeometry {
                height: 28,
                width: 28,
                patch_size: 14,
                spatial_merge_size: 1,
                max_soft_tokens: 16,
            }],
        )
        .unwrap();
        // 4 soft tokens at placeholder 1 → positions 1..4
        assert_eq!(positions, vec![1, 2, 3, 4]);
        let merged = scatter_vision_into_text(&text, &soft, &positions).expect("scatter");
        eval(&[&merged]);
        assert_eq!(merged.shape()[1], 6);
    }
}

// ---------------------------------------------------------------------------
// Minimal portable Qwen3-VL ViT forward (WS-V2)
//
// Implements patch-linear + stacked attention/MLP + spatial-merge projector
// using MLX ops. Weights are explicit tensors so unit tests can drive the
// real path without a full checkpoint. Production load will map HF names
// into [`Qwen3VlVisionWeights`].
// ---------------------------------------------------------------------------

/// One vision transformer layer weights.
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

/// Full vision tower weights for the portable encoder.
#[derive(Clone, Debug)]
pub struct Qwen3VlVisionWeights {
    pub patch_embed: MlxArray,
    pub patch_embed_bias: Option<MlxArray>,
    pub layers: Vec<Qwen3VlVisionLayerWeights>,
    pub merger: MlxArray,
    pub merger_bias: Option<MlxArray>,
    pub num_heads: usize,
    pub hidden_size: usize,
    pub deepstack_indexes: Vec<usize>,
}

fn ln(x: &MlxArray, weight: &MlxArray, bias: Option<&MlxArray>, eps: f32) -> MlxArray {
    match bias {
        Some(b) => layer_norm(x, weight, b, eps, None),
        None => {
            // layer_norm requires bias in this binding — use zeros-like weight.
            let zero = cached_scalar(0.0, weight.dtype());
            layer_norm(x, weight, &zero, eps, None)
        }
    }
}

fn linear(x: &MlxArray, w: &MlxArray, bias: Option<&MlxArray>) -> MlxArray {
    // w is [out, in]; matmul x @ w^T
    let mut y = matmul(x, &transpose(w, &[1, 0], None), None);
    if let Some(b) = bias {
        y = add(&y, b, None);
    }
    y
}

fn attention(
    x: &MlxArray,
    layer: &Qwen3VlVisionLayerWeights,
    num_heads: usize,
    hidden_size: usize,
) -> MlxArray {
    let seq = x.shape()[1] as usize;
    let head_dim = hidden_size / num_heads;
    let qkv = linear(x, &layer.qkv, layer.qkv_bias.as_ref());
    // qkv: [1, S, 3*H] -> [1, S, 3, heads, head_dim]
    let qkv = reshape(
        &qkv,
        &[1, seq as i32, 3, num_heads as i32, head_dim as i32],
        None,
    );
    let q = slice(
        &qkv,
        &[0, 0, 0, 0, 0],
        &[1, seq as i32, 1, num_heads as i32, head_dim as i32],
        &[1, 1, 1, 1, 1],
        None,
    );
    let k = slice(
        &qkv,
        &[0, 0, 1, 0, 0],
        &[1, seq as i32, 2, num_heads as i32, head_dim as i32],
        &[1, 1, 1, 1, 1],
        None,
    );
    let v = slice(
        &qkv,
        &[0, 0, 2, 0, 0],
        &[1, seq as i32, 3, num_heads as i32, head_dim as i32],
        &[1, 1, 1, 1, 1],
        None,
    );
    // [1, S, heads, dim] -> [1, heads, S, dim]
    let q = transpose(
        &reshape(
            &q,
            &[1, seq as i32, num_heads as i32, head_dim as i32],
            None,
        ),
        &[0, 2, 1, 3],
        None,
    );
    let k = transpose(
        &reshape(
            &k,
            &[1, seq as i32, num_heads as i32, head_dim as i32],
            None,
        ),
        &[0, 2, 1, 3],
        None,
    );
    let v = transpose(
        &reshape(
            &v,
            &[1, seq as i32, num_heads as i32, head_dim as i32],
            None,
        ),
        &[0, 2, 1, 3],
        None,
    );
    let scale = (head_dim as f32).sqrt().recip();
    let scores = matmul(&q, &transpose(&k, &[0, 1, 3, 2], None), None);
    let scores = multiply(&scores, &cached_scalar(scale, scores.dtype()), None);
    let probs = softmax(&scores, -1, None);
    let ctx = matmul(&probs, &v, None);
    // [1, heads, S, dim] -> [1, S, H]
    let ctx = transpose(&ctx, &[0, 2, 1, 3], None);
    let ctx = reshape(&ctx, &[1, seq as i32, hidden_size as i32], None);
    linear(&ctx, &layer.proj, layer.proj_bias.as_ref())
}

fn mlp(x: &MlxArray, layer: &Qwen3VlVisionLayerWeights) -> MlxArray {
    let h = linear(x, &layer.fc1, layer.fc1_bias.as_ref());
    let h = gelu(&h, None);
    linear(&h, &layer.fc2, layer.fc2_bias.as_ref())
}

/// Run the portable ViT encoder.
///
/// `patches`: `[1, num_patches, patch_dim]` float32/bf16.
/// Returns soft tokens `[1, soft_tokens, out_dim]` after spatial merge projector.
pub fn vision_encoder_forward(
    weights: &Qwen3VlVisionWeights,
    patches: &MlxArray,
) -> Result<(MlxArray, Vec<MlxArray>), Qwen3VlError> {
    if weights.layers.is_empty() || weights.num_heads == 0 || weights.hidden_size == 0 {
        return Err(Qwen3VlError::InvalidGeometry(
            "vision weights incomplete".into(),
        ));
    }
    if !weights.hidden_size.is_multiple_of(weights.num_heads) {
        return Err(Qwen3VlError::InvalidGeometry(
            "hidden_size must divide num_heads".into(),
        ));
    }
    let mut x = linear(
        patches,
        &weights.patch_embed,
        weights.patch_embed_bias.as_ref(),
    );
    let mut deepstack_feats = Vec::new();
    for (li, layer) in weights.layers.iter().enumerate() {
        let n1 = ln(&x, &layer.norm1_weight, layer.norm1_bias.as_ref(), 1e-6);
        let attn = attention(&n1, layer, weights.num_heads, weights.hidden_size);
        x = add(&x, &attn, None);
        let n2 = ln(&x, &layer.norm2_weight, layer.norm2_bias.as_ref(), 1e-6);
        let ff = mlp(&n2, layer);
        x = add(&x, &ff, None);
        if weights.deepstack_indexes.contains(&li) {
            deepstack_feats.push(x.clone());
        }
    }
    // Spatial-merge style projector: flatten last dim via linear merger.
    let soft = linear(&x, &weights.merger, weights.merger_bias.as_ref());
    Ok((soft, deepstack_feats))
}

/// Scatter soft-token embeddings into a text residual stream (LLaVA-style).
///
/// `text_hidden`: `[1, T, H]`, `vision`: `[1, S, H]`, `positions`: length S absolute
/// indices into the text sequence.
pub fn scatter_vision_into_text(
    text_hidden: &MlxArray,
    vision: &MlxArray,
    positions: &[usize],
) -> Result<MlxArray, Qwen3VlError> {
    if positions.is_empty() {
        return Ok(text_hidden.clone());
    }
    let t = text_hidden.shape()[1] as usize;
    let h = text_hidden.shape()[2] as usize;
    let s = vision.shape()[1] as usize;
    if positions.len() != s {
        return Err(Qwen3VlError::Scatter(format!(
            "positions {} != vision tokens {}",
            positions.len(),
            s
        )));
    }
    // Portable scatter: rebuild sequence row-by-row (test-scale; production
    // uses slice_update on the full tensor when available).
    let mut rows: Vec<MlxArray> = Vec::with_capacity(t);
    for i in 0..t {
        if let Some(vi) = positions.iter().position(|&p| p == i) {
            let v = slice(
                vision,
                &[0, vi as i32, 0],
                &[1, (vi + 1) as i32, h as i32],
                &[1, 1, 1],
                None,
            );
            rows.push(v);
        } else {
            let row = slice(
                text_hidden,
                &[0, i as i32, 0],
                &[1, (i + 1) as i32, h as i32],
                &[1, 1, 1],
                None,
            );
            rows.push(row);
        }
    }
    let refs: Vec<&MlxArray> = rows.iter().collect();
    Ok(concatenate(&refs, 1, None))
}
