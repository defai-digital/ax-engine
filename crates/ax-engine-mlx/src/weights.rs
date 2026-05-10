use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;

use mlx_sys::{MlxArray, MlxDtype, add, astype, concatenate, eval, load_safetensors, multiply};

use ax_engine_core::{
    NativeModelArtifacts, NativeTensorQuantization, NativeTensorRole, NativeTensorSpec,
};

/// All weight arrays for one model.
pub struct ModelWeights {
    pub token_embedding: QuantizedWeight,
    pub final_norm: MlxArray,
    pub lm_head: QuantizedWeight,
    pub layers: Vec<LayerWeights>,
    /// Per-layer token embedding table (Gemma4 2B/4B, shape [vocab_per_layer, num_layers*per_layer_dim]).
    pub per_layer_embed: Option<QuantizedWeight>,
    /// Global projection hidden → num_layers*per_layer_dim (Gemma4 2B/4B).
    pub per_layer_model_proj: Option<QuantizedWeight>,
    /// RMSNorm weight over per_layer_dim applied after model projection (Gemma4 2B/4B).
    pub per_layer_proj_norm: Option<MlxArray>,
}

/// Weights (and optional quantization data) for one transformer layer.
pub struct LayerWeights {
    pub attn_norm: MlxArray,
    /// post_attention_layernorm (Gemma4): applied to attn output BEFORE residual add.
    /// For Qwen3 (no separate pre-FFN norm), this field is None and the same norm
    /// is used as ffn_norm instead.
    pub attn_post_norm: Option<MlxArray>,
    pub q_norm: Option<MlxArray>,
    pub k_norm: Option<MlxArray>,
    // Split Q/K/V projections (None for KV-shared layers that reuse a source layer's KV).
    pub q_proj: Option<QuantizedWeight>,
    pub k_proj: Option<QuantizedWeight>,
    pub v_proj: Option<QuantizedWeight>,
    // Packed QKV projection (some architectures).
    pub qkv_packed: Option<QuantizedWeight>,
    pub o_proj: Option<QuantizedWeight>,
    // Linear attention (Qwen3.5 hybrid layers). Present instead of full-attention QKV/O.
    pub linear_attn: Option<LinearAttentionWeights>,
    // GLM4MoELite MLA attention. Present instead of standard full-attention Q/K/V.
    pub glm_mla_attn: Option<GlmMlaAttentionWeights>,
    // Dense FFN norms and weights.
    pub ffn_norm: MlxArray,
    pub ffn_post_norm: Option<MlxArray>,
    pub gate_proj: Option<QuantizedWeight>,
    pub up_proj: Option<QuantizedWeight>,
    pub gate_up_packed: Option<QuantizedWeight>,
    pub down_proj: Option<QuantizedWeight>,
    // MoE: extra norms (present when this layer has a MoE block).
    pub ffn_norm2: Option<MlxArray>,
    pub ffn_post_norm1: Option<MlxArray>,
    pub ffn_post_norm2: Option<MlxArray>,
    // MoE: router weights.
    pub router_proj: Option<QuantizedWeight>,
    pub router_correction_bias: Option<MlxArray>,
    pub router_scale: Option<MlxArray>,
    /// Precomputed `router_scale * hidden_size^-0.5` for Gemma4 MoE router RMSNorm.
    pub router_combined_scale: Option<MlxArray>,
    /// Per-expert output scale (Gemma4 MoE): multiply top-k weights by this after softmax.
    pub router_expert_scale: Option<MlxArray>,
    /// Per-layer scalar applied to hidden states after the FFN residual (Gemma4).
    pub layer_scalar: Option<MlxArray>,
    /// Per-layer input gate projection: hidden → per_layer_dim (Gemma4 2B/4B).
    pub per_layer_gate: Option<QuantizedWeight>,
    /// Per-layer output projection: per_layer_dim → hidden (Gemma4 2B/4B).
    pub per_layer_proj_w: Option<QuantizedWeight>,
    /// Post-gating RMSNorm weight (Gemma4 2B/4B).
    pub per_layer_post_norm: Option<MlxArray>,
    // MoE: expert weights (shape [num_experts, expert_size, hidden] / packed).
    pub shared_expert_gate: Option<QuantizedWeight>,
    pub shared_gate_proj: Option<QuantizedWeight>,
    pub shared_up_proj: Option<QuantizedWeight>,
    pub shared_down_proj: Option<QuantizedWeight>,
    pub gate_up_exps_packed: Option<QuantizedWeight>,
    pub gate_exps: Option<QuantizedWeight>,
    pub up_exps: Option<QuantizedWeight>,
    pub down_exps: Option<QuantizedWeight>,
}

/// Weights for a GLM4MoELite MLA attention layer.
pub struct GlmMlaAttentionWeights {
    /// q_a_proj and kv_a_proj fused into one `[q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden]`
    /// weight. Eliminates one matmul kernel launch per layer during prefill.
    pub qa_kva_fused: QuantizedWeight,
    pub q_a_norm: MlxArray,
    pub q_b_proj: QuantizedWeight,
    pub kv_a_norm: MlxArray,
    pub embed_q: QuantizedWeight,
    pub unembed_out: QuantizedWeight,
}

/// Weights for a Qwen3.5 GatedDelta linear-attention layer.
pub struct LinearAttentionWeights {
    pub in_proj_qkv: Option<QuantizedWeight>,
    pub in_proj_z: Option<QuantizedWeight>,
    pub in_proj_a: Option<QuantizedWeight>,
    pub in_proj_b: Option<QuantizedWeight>,
    pub in_proj_qkvz: Option<QuantizedWeight>,
    pub in_proj_ba: Option<QuantizedWeight>,
    pub conv1d: QuantizedWeight,
    pub dt_bias: MlxArray,
    pub a_log: MlxArray,
    pub norm: MlxArray,
    pub out_proj: QuantizedWeight,
}

/// A weight matrix plus optional MLX affine quantization metadata.
///
/// When `scales` is `Some`, the weight tensor contains packed affine-quantized
/// integers and must be multiplied via `mlx_quantized_matmul` rather than
/// regular matmul.
pub struct QuantizedWeight {
    pub weight: MlxArray,
    pub scales: Option<MlxArray>,
    pub biases: Option<MlxArray>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedWeight {
    pub fn new(weight: MlxArray, scales: Option<MlxArray>, biases: Option<MlxArray>) -> Self {
        Self::with_quantization(weight, scales, biases, None)
    }

    pub fn with_quantization(
        weight: MlxArray,
        scales: Option<MlxArray>,
        biases: Option<MlxArray>,
        quantization: Option<&NativeTensorQuantization>,
    ) -> Self {
        let quantization = quantization.cloned().unwrap_or_default();
        Self {
            weight,
            scales,
            biases,
            group_size: quantization.group_size as i32,
            bits: quantization.bits as i32,
        }
    }
    pub fn is_quantized(&self) -> bool {
        self.scales.is_some()
    }
}

pub fn load_weights(artifacts: &NativeModelArtifacts) -> Result<ModelWeights, WeightLoadError> {
    let root = artifacts.root_dir().to_path_buf();
    let mut file_cache: HashMap<PathBuf, HashMap<String, MlxArray>> = HashMap::new();
    for spec in artifacts.tensor_specs() {
        let full = root.join(&spec.file);
        if let Entry::Vacant(entry) = file_cache.entry(full) {
            let path = entry.key().clone();
            let tensors = load_safetensors(&path, None).map_err(WeightLoadError::FileMissing)?;
            if tensors.is_empty() {
                return Err(WeightLoadError::FileMissing(path.display().to_string()));
            }
            let refs: Vec<&MlxArray> = tensors.values().collect();
            mlx_sys::eval(&refs);
            entry.insert(tensors);
        }
    }

    // Merge all tensors from all files into one flat map.
    let mut name_map: HashMap<String, MlxArray> = HashMap::new();
    for tensors in file_cache.into_values() {
        name_map.extend(tensors);
    }

    let specs = artifacts.tensor_specs();
    let layer_count = artifacts.manifest().layer_count as usize;

    let token_embedding = take_weight(
        specs,
        &mut name_map,
        NativeTensorRole::TokenEmbedding,
        None,
        "token_embedding",
    )?;
    let final_norm = take_weight(
        specs,
        &mut name_map,
        NativeTensorRole::FinalNorm,
        None,
        "final_norm",
    )?
    .weight;
    let lm_head = if artifacts.manifest().tie_word_embeddings {
        let mut tied = QuantizedWeight::new(
            token_embedding.weight.clone(),
            token_embedding.scales.clone(),
            token_embedding.biases.clone(),
        );
        tied.group_size = token_embedding.group_size;
        tied.bits = token_embedding.bits;
        tied
    } else {
        take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::LmHead,
            None,
            "lm_head",
        )?
    };

    // Global per-layer input gating weights (Gemma4 2B/4B, optional).
    let per_layer_embed = if has_role(specs, NativeTensorRole::PerLayerEmbedding, None) {
        Some(take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerEmbedding,
            None,
            "per_layer_embed",
        )?)
    } else {
        None
    };
    let per_layer_model_proj = if has_role(specs, NativeTensorRole::PerLayerModelProjection, None) {
        Some(take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerModelProjection,
            None,
            "per_layer_model_proj",
        )?)
    } else {
        None
    };
    let per_layer_proj_norm = if has_role(specs, NativeTensorRole::PerLayerProjectionNorm, None) {
        let w = take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerProjectionNorm,
            None,
            "per_layer_proj_norm",
        )?;
        Some(w.weight)
    } else {
        None
    };

    let mut layers = Vec::with_capacity(layer_count);
    for li in 0..layer_count {
        let idx = Some(li as u32);
        let uses_shared_kv = artifacts
            .manifest()
            .kv_shared_source_layers
            .contains_key(&(li as u32));
        let uses_value_from_key = artifacts
            .manifest()
            .attention_value_from_key_layers
            .contains(&(li as u32));
        let attention_layout = attention_layout_for_layer(specs, idx)?;

        let attn_norm = take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::AttentionNorm,
            idx,
            "attn_norm",
        )?
        .weight;
        let o_proj = match attention_layout {
            AttentionLayout::Full => Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::AttentionO,
                idx,
                "o_proj",
            )?),
            AttentionLayout::Linear => None,
        };
        let linear_attn = match attention_layout {
            AttentionLayout::Full => None,
            AttentionLayout::Linear => {
                Some(load_linear_attention_weights(specs, &mut name_map, idx)?)
            }
        };
        // When FfnNorm (pre_feedforward_layernorm) is present (Gemma4), AttentionPostNorm
        // is a genuine post-attention norm applied before the residual add.  When FfnNorm
        // is absent (Qwen3), AttentionPostNorm doubles as the pre-FFN norm instead.
        let (attn_post_norm, ffn_norm) = if has_role(specs, NativeTensorRole::FfnNorm, idx) {
            let apn = try_take_plain(
                specs,
                &mut name_map,
                NativeTensorRole::AttentionPostNorm,
                idx,
            )?;
            let fn_w = take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnNorm,
                idx,
                "ffn_norm",
            )?
            .weight;
            (apn, fn_w)
        } else {
            let fn_w = take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::AttentionPostNorm,
                idx,
                "attention_post_norm",
            )?
            .weight;
            (None, fn_w)
        };
        let down_proj = if has_role(specs, NativeTensorRole::FfnDown, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnDown,
                idx,
                "down_proj",
            )?)
        } else {
            None
        };

        let ffn_post_norm =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnPostNorm, idx)?;
        let ffn_norm2 = try_take_plain(specs, &mut name_map, NativeTensorRole::FfnNorm2, idx)?;
        let ffn_post_norm1 =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnPostNorm1, idx)?;
        let ffn_post_norm2 =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnPostNorm2, idx)?;

        let router_proj = if has_role(specs, NativeTensorRole::FfnGateInp, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateInp,
                idx,
                "router_proj",
            )?)
        } else {
            None
        };
        let router_scale =
            try_take_plain(specs, &mut name_map, NativeTensorRole::FfnGateInpScale, idx)?;
        let router_combined_scale = if artifacts.manifest().model_family == "gemma4" {
            router_scale
                .as_ref()
                .map(|scale| gemma4_router_combined_scale(artifacts.manifest().hidden_size, scale))
        } else {
            None
        };
        let router_expert_scale = try_take_plain(
            specs,
            &mut name_map,
            NativeTensorRole::FfnGateInpExpertScale,
            idx,
        )?;
        let layer_scalar =
            try_take_plain(specs, &mut name_map, NativeTensorRole::LayerScalar, idx)?;
        let per_layer_gate = if has_role(specs, NativeTensorRole::PerLayerInputGate, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::PerLayerInputGate,
                idx,
                "per_layer_gate",
            )?)
        } else {
            None
        };
        let per_layer_proj_w = if has_role(specs, NativeTensorRole::PerLayerInputProjection, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::PerLayerInputProjection,
                idx,
                "per_layer_proj_w",
            )?)
        } else {
            None
        };
        let per_layer_post_norm = try_take_plain(
            specs,
            &mut name_map,
            NativeTensorRole::PerLayerInputPostNorm,
            idx,
        )?;

        let shared_expert_gate = if has_role(specs, NativeTensorRole::FfnSharedExpertGateInp, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertGateInp,
                idx,
                "shared_expert_gate",
            )?)
        } else {
            None
        };
        let shared_gate_proj = if has_role(specs, NativeTensorRole::FfnSharedExpertGate, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertGate,
                idx,
                "shared_gate_proj",
            )?)
        } else {
            None
        };
        let shared_up_proj = if has_role(specs, NativeTensorRole::FfnSharedExpertUp, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertUp,
                idx,
                "shared_up_proj",
            )?)
        } else {
            None
        };
        let shared_down_proj = if has_role(specs, NativeTensorRole::FfnSharedExpertDown, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnSharedExpertDown,
                idx,
                "shared_down_proj",
            )?)
        } else {
            None
        };

        let gate_up_exps_packed = if has_role(specs, NativeTensorRole::FfnGateUpExpsPacked, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateUpExpsPacked,
                idx,
                "gate_up_exps",
            )?)
        } else {
            None
        };
        let gate_exps = if has_role(specs, NativeTensorRole::FfnGateExps, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateExps,
                idx,
                "gate_exps",
            )?)
        } else {
            None
        };
        let up_exps = if has_role(specs, NativeTensorRole::FfnUpExps, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnUpExps,
                idx,
                "up_exps",
            )?)
        } else {
            None
        };
        let down_exps = if has_role(specs, NativeTensorRole::FfnDownExps, idx) {
            Some(take_weight(
                specs,
                &mut name_map,
                NativeTensorRole::FfnDownExps,
                idx,
                "down_exps",
            )?)
        } else {
            None
        };

        let q_norm = try_take_plain(specs, &mut name_map, NativeTensorRole::AttentionQNorm, idx)?;
        let k_norm = try_take_plain(specs, &mut name_map, NativeTensorRole::AttentionKNorm, idx)?;

        let (qkv_packed, q_proj, k_proj, v_proj, glm_mla_attn) = if attention_layout
            == AttentionLayout::Linear
        {
            (None, None, None, None, None)
        } else {
            match full_attention_projection_layout(specs, idx, uses_shared_kv, uses_value_from_key)?
            {
                FullAttentionProjectionLayout::GlmMla => {
                    let glm = load_glm_mla_attention_weights(specs, &mut name_map, idx)?;
                    (None, None, None, None, Some(glm))
                }
                FullAttentionProjectionLayout::QOnly => {
                    let q = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQ,
                        idx,
                        "q_proj",
                    )?;
                    (None, Some(q), None, None, None)
                }
                FullAttentionProjectionLayout::PackedQkv => {
                    let p = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQkvPacked,
                        idx,
                        "qkv",
                    )?;
                    (Some(p), None, None, None, None)
                }
                FullAttentionProjectionLayout::SplitQkValueFromKey => {
                    let q = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQ,
                        idx,
                        "q_proj",
                    )?;
                    let k = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionK,
                        idx,
                        "k_proj",
                    )?;
                    (None, Some(q), Some(k), None, None)
                }
                FullAttentionProjectionLayout::SplitQkv => {
                    let q = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQ,
                        idx,
                        "q_proj",
                    )?;
                    let k = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionK,
                        idx,
                        "k_proj",
                    )?;
                    let v = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionV,
                        idx,
                        "v_proj",
                    )?;
                    (None, Some(q), Some(k), Some(v), None)
                }
            }
        };

        let (gate_up_packed, gate_proj, up_proj) =
            if has_role(specs, NativeTensorRole::FfnGateUpPacked, idx) {
                let p = take_weight(
                    specs,
                    &mut name_map,
                    NativeTensorRole::FfnGateUpPacked,
                    idx,
                    "gate_up",
                )?;
                (Some(p), None, None)
            } else if has_role(specs, NativeTensorRole::FfnGate, idx) {
                let g = take_weight(
                    specs,
                    &mut name_map,
                    NativeTensorRole::FfnGate,
                    idx,
                    "gate_proj",
                )?;
                let u = take_weight(
                    specs,
                    &mut name_map,
                    NativeTensorRole::FfnUp,
                    idx,
                    "up_proj",
                )?;
                (None, Some(g), Some(u))
            } else {
                (None, None, None)
            };

        layers.push(LayerWeights {
            attn_norm,
            attn_post_norm,
            q_norm,
            k_norm,
            q_proj,
            k_proj,
            v_proj,
            qkv_packed,
            o_proj,
            linear_attn,
            glm_mla_attn,
            ffn_norm,
            ffn_post_norm,
            gate_proj,
            up_proj,
            gate_up_packed,
            down_proj,
            ffn_norm2,
            ffn_post_norm1,
            ffn_post_norm2,
            router_proj,
            router_correction_bias: try_take_plain(
                specs,
                &mut name_map,
                NativeTensorRole::FfnGateInpCorrectionBias,
                idx,
            )?,
            router_scale,
            router_combined_scale,
            router_expert_scale,
            layer_scalar,
            per_layer_gate,
            per_layer_proj_w,
            per_layer_post_norm,
            shared_expert_gate,
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            gate_up_exps_packed,
            gate_exps,
            up_exps,
            down_exps,
        });
    }

    // Raw HF checkpoints store linear-attention norm weights as zero-centred deltas
    // (mean ≈ 0); sanitized mlx-community weights have +1.0 applied (mean ≈ 1.0).
    // Apply the offset on load so that unsanitized artifacts (e.g. models cloned
    // directly from HuggingFace without running mlx_lm.convert) work correctly.
    for layer in &mut layers {
        if let Some(la) = layer.linear_attn.as_mut() {
            la.norm = sanitize_norm_if_needed(la.norm.clone());
        }
    }

    Ok(ModelWeights {
        token_embedding,
        final_norm,
        lm_head,
        layers,
        per_layer_embed,
        per_layer_model_proj,
        per_layer_proj_norm,
    })
}

/// Auto-sanitize a linear-attention norm weight if needed.
///
/// Raw HuggingFace checkpoints store these weights as zero-centred deltas (mean ≈ 0);
/// the correct inference value requires adding +1.0, which mlx-lm's sanitize() step
/// applies when converting. If the weight is already sanitized (mean abs ≈ 1.0) it is
/// returned unchanged.
fn sanitize_norm_if_needed(norm: MlxArray) -> MlxArray {
    let f32_norm = astype(&norm, MlxDtype::Float32, None);
    eval(&[&f32_norm]);
    let data = f32_norm.data_f32();
    if data.len() < 8 {
        return norm;
    }
    let mean_abs: f32 = data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32;
    if mean_abs < 0.15 {
        let orig_dtype = norm.dtype();
        let norm_f32 = astype(&norm, MlxDtype::Float32, None);
        let one = MlxArray::from_f32_slice(&[1.0_f32]);
        let sanitized = add(&norm_f32, &one, None);
        astype(&sanitized, orig_dtype, None)
    } else {
        norm
    }
}

fn gemma4_router_combined_scale(hidden_size: u32, router_scale: &MlxArray) -> MlxArray {
    let root_factor = 1.0_f32 / (hidden_size as f32).sqrt();
    let scale_arr = MlxArray::from_raw_data(
        &root_factor as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    let scale_arr = astype(&scale_arr, MlxDtype::Bfloat16, None);
    multiply(router_scale, &scale_arr, None)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionLayout {
    Full,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FullAttentionProjectionLayout {
    /// GLM4MoELite MLA attention uses q_a/q_b + latent KV projections.
    GlmMla,
    /// KV-shared layers compute only Q and reuse a source layer's K/V.
    QOnly,
    PackedQkv,
    /// K=V full-attention layers compute Q/K and reuse K as V (Gemma4 full attention).
    SplitQkValueFromKey,
    SplitQkv,
}

fn attention_layout_for_layer(
    specs: &[NativeTensorSpec],
    layer_index: Option<u32>,
) -> Result<AttentionLayout, WeightLoadError> {
    let has_full = has_full_attention_role(specs, layer_index);
    let has_linear = has_linear_attention_role(specs, layer_index);

    if has_full && has_linear {
        return Err(WeightLoadError::InvalidLayer(format!(
            "layer {layer_index:?} mixes full-attention and linear-attention tensor roles"
        )));
    }
    if has_linear {
        Ok(AttentionLayout::Linear)
    } else {
        Ok(AttentionLayout::Full)
    }
}

fn full_attention_projection_layout(
    specs: &[NativeTensorSpec],
    layer_index: Option<u32>,
    uses_shared_kv: bool,
    uses_value_from_key: bool,
) -> Result<FullAttentionProjectionLayout, WeightLoadError> {
    let has_glm_mla = has_glm_mla_attention_role(specs, layer_index);
    let has_standard_full = has_standard_full_attention_projection_role(specs, layer_index);
    if has_glm_mla {
        if uses_shared_kv || uses_value_from_key || has_standard_full {
            return Err(WeightLoadError::InvalidLayer(format!(
                "layer {layer_index:?} mixes GLM MLA attention with standard full-attention layout"
            )));
        }
        return Ok(FullAttentionProjectionLayout::GlmMla);
    }

    let has_packed = has_role(specs, NativeTensorRole::AttentionQkvPacked, layer_index);
    if uses_shared_kv {
        if has_packed {
            return Err(WeightLoadError::InvalidLayer(format!(
                "layer {layer_index:?} is KV-shared but provides packed QKV weights"
            )));
        }
        return Ok(FullAttentionProjectionLayout::QOnly);
    }
    if has_packed {
        if uses_value_from_key {
            return Err(WeightLoadError::InvalidLayer(format!(
                "layer {layer_index:?} is marked value-from-key but provides packed QKV weights"
            )));
        }
        Ok(FullAttentionProjectionLayout::PackedQkv)
    } else if uses_value_from_key {
        Ok(FullAttentionProjectionLayout::SplitQkValueFromKey)
    } else {
        Ok(FullAttentionProjectionLayout::SplitQkv)
    }
}

fn has_full_attention_role(specs: &[NativeTensorSpec], layer_index: Option<u32>) -> bool {
    [
        NativeTensorRole::AttentionO,
        NativeTensorRole::AttentionQ,
        NativeTensorRole::AttentionK,
        NativeTensorRole::AttentionV,
        NativeTensorRole::AttentionQkvPacked,
        NativeTensorRole::AttentionQa,
        NativeTensorRole::AttentionQaNorm,
        NativeTensorRole::AttentionQb,
        NativeTensorRole::AttentionKvA,
        NativeTensorRole::AttentionKvANorm,
        NativeTensorRole::AttentionEmbedQ,
        NativeTensorRole::AttentionUnembedOut,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn has_standard_full_attention_projection_role(
    specs: &[NativeTensorSpec],
    layer_index: Option<u32>,
) -> bool {
    [
        NativeTensorRole::AttentionQ,
        NativeTensorRole::AttentionK,
        NativeTensorRole::AttentionV,
        NativeTensorRole::AttentionQkvPacked,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn has_glm_mla_attention_role(specs: &[NativeTensorSpec], layer_index: Option<u32>) -> bool {
    [
        NativeTensorRole::AttentionQa,
        NativeTensorRole::AttentionQaNorm,
        NativeTensorRole::AttentionQb,
        NativeTensorRole::AttentionKvA,
        NativeTensorRole::AttentionKvANorm,
        NativeTensorRole::AttentionEmbedQ,
        NativeTensorRole::AttentionUnembedOut,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn has_linear_attention_role(specs: &[NativeTensorSpec], layer_index: Option<u32>) -> bool {
    [
        NativeTensorRole::LinearAttentionInProjQkv,
        NativeTensorRole::LinearAttentionInProjQkvz,
        NativeTensorRole::LinearAttentionInProjZ,
        NativeTensorRole::LinearAttentionInProjA,
        NativeTensorRole::LinearAttentionInProjB,
        NativeTensorRole::LinearAttentionInProjBa,
        NativeTensorRole::LinearAttentionConv1d,
        NativeTensorRole::LinearAttentionDtBias,
        NativeTensorRole::LinearAttentionALog,
        NativeTensorRole::LinearAttentionNorm,
        NativeTensorRole::LinearAttentionOutProj,
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn load_linear_attention_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
) -> Result<LinearAttentionWeights, WeightLoadError> {
    Ok(LinearAttentionWeights {
        in_proj_qkv: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjQkv,
            layer_index,
            "linear_attention_in_proj_qkv",
        )?,
        in_proj_z: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjZ,
            layer_index,
            "linear_attention_in_proj_z",
        )?,
        in_proj_a: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjA,
            layer_index,
            "linear_attention_in_proj_a",
        )?,
        in_proj_b: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjB,
            layer_index,
            "linear_attention_in_proj_b",
        )?,
        in_proj_qkvz: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjQkvz,
            layer_index,
            "linear_attention_in_proj_qkvz",
        )?,
        in_proj_ba: try_take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjBa,
            layer_index,
            "linear_attention_in_proj_ba",
        )?,
        conv1d: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionConv1d,
            layer_index,
            "linear_attention_conv1d",
        )?,
        dt_bias: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionDtBias,
            layer_index,
            "linear_attention_dt_bias",
        )?
        .weight,
        a_log: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionALog,
            layer_index,
            "linear_attention_a_log",
        )?
        .weight,
        norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionNorm,
            layer_index,
            "linear_attention_norm",
        )?
        .weight,
        out_proj: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionOutProj,
            layer_index,
            "linear_attention_out_proj",
        )?,
    })
}

fn load_glm_mla_attention_weights(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer_index: Option<u32>,
) -> Result<GlmMlaAttentionWeights, WeightLoadError> {
    let q_a_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionQa,
        layer_index,
        "glm_q_a_proj",
    )?;
    let kv_a_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionKvA,
        layer_index,
        "glm_kv_a_proj",
    )?;
    Ok(GlmMlaAttentionWeights {
        qa_kva_fused: concat_quantized_weight_rows(&q_a_proj, &kv_a_proj)?,
        q_a_norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionQaNorm,
            layer_index,
            "glm_q_a_norm",
        )?
        .weight,
        q_b_proj: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionQb,
            layer_index,
            "glm_q_b_proj",
        )?,
        kv_a_norm: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionKvANorm,
            layer_index,
            "glm_kv_a_norm",
        )?
        .weight,
        embed_q: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionEmbedQ,
            layer_index,
            "glm_embed_q",
        )?,
        unembed_out: take_weight(
            specs,
            name_map,
            NativeTensorRole::AttentionUnembedOut,
            layer_index,
            "glm_unembed_out",
        )?,
    })
}

/// Concatenate two weight matrices along the output (row) dimension.
///
/// Used to fuse parallel projections that read the same input (e.g. q_a_proj
/// and kv_a_proj in GLM MLA), replacing two matmul kernel launches with one.
fn concat_quantized_weight_rows(
    a: &QuantizedWeight,
    b: &QuantizedWeight,
) -> Result<QuantizedWeight, WeightLoadError> {
    let (scales, biases) = match (&a.scales, &b.scales) {
        (Some(sa), Some(sb)) => {
            if a.group_size != b.group_size {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "cannot fuse quantized GLM MLA qa/kv projections with different group sizes: {} vs {}",
                    a.group_size, b.group_size
                )));
            }
            if a.bits != b.bits {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "cannot fuse quantized GLM MLA qa/kv projections with different bit widths: {} vs {}",
                    a.bits, b.bits
                )));
            }
            let biases = match (a.biases.as_ref(), b.biases.as_ref()) {
                (Some(ba), Some(bb)) => Some(concatenate(&[ba, bb], 0, None)),
                (None, None) => None,
                _ => {
                    return Err(WeightLoadError::InvalidLayer(
                        "cannot fuse GLM MLA qa/kv projections where only one has quantization biases"
                            .to_string(),
                    ));
                }
            };
            (Some(concatenate(&[sa, sb], 0, None)), biases)
        }
        (None, None) => (None, None),
        _ => {
            return Err(WeightLoadError::InvalidLayer(
                "cannot fuse GLM MLA qa/kv projections where only one has quantization scales"
                    .to_string(),
            ));
        }
    };
    Ok(QuantizedWeight {
        weight: concatenate(&[&a.weight, &b.weight], 0, None),
        scales,
        biases,
        group_size: a.group_size,
        bits: a.bits,
    })
}

/// Load a weight tensor together with its `.scales` and `.biases` siblings
/// if they exist in the safetensors map (MLX affine quantization format).
fn take_weight(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    label: &str,
) -> Result<QuantizedWeight, WeightLoadError> {
    let spec = specs
        .iter()
        .find(|s| s.role == role && s.layer_index == layer_index)
        .ok_or_else(|| WeightLoadError::RoleMissing(format!("{label}[{layer_index:?}]")))?;
    let name = spec.name.clone();

    let weight = name_map
        .remove(&name)
        .ok_or_else(|| WeightLoadError::TensorMissing(name.clone()))?;

    // Look for co-located `.scales` and `.biases` (MLX quantized format).
    let base = name.strip_suffix(".weight").unwrap_or(&name);
    let scales = name_map.remove(&format!("{base}.scales"));
    let biases = name_map.remove(&format!("{base}.biases"));
    let has_quantization_sidecars = scales.is_some() || biases.is_some();

    if !spec.source_quantized && has_quantization_sidecars {
        return Err(WeightLoadError::InvalidLayer(format!(
            "tensor {name} has MLX quantization sidecar tensors but source_quantized is false"
        )));
    }

    if spec.source_quantized && scales.is_none() {
        return Err(WeightLoadError::QuantizationMissing(format!(
            "{base}.scales"
        )));
    }

    Ok(QuantizedWeight::with_quantization(
        weight,
        scales,
        biases,
        spec.quantization.as_ref(),
    ))
}

fn try_take_weight(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    label: &str,
) -> Result<Option<QuantizedWeight>, WeightLoadError> {
    if has_role(specs, role, layer_index) {
        take_weight(specs, name_map, role, layer_index, label).map(Some)
    } else {
        Ok(None)
    }
}

/// Load a plain (non-quantized) weight tensor or return None if not present.
fn try_take_plain(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer_index: Option<u32>,
) -> Result<Option<MlxArray>, WeightLoadError> {
    let Some(name) = specs
        .iter()
        .find(|s| s.role == role && s.layer_index == layer_index)
        .map(|s| s.name.clone())
    else {
        return Ok(None);
    };
    Ok(name_map.remove(&name))
}

fn has_role(specs: &[NativeTensorSpec], role: NativeTensorRole, layer_index: Option<u32>) -> bool {
    specs
        .iter()
        .any(|s| s.role == role && s.layer_index == layer_index)
}

#[derive(Debug, thiserror::Error)]
pub enum WeightLoadError {
    #[error("weight file not found or empty: {0}")]
    FileMissing(String),
    #[error("tensor not found: {0}")]
    TensorMissing(String),
    #[error("required tensor role missing: {0}")]
    RoleMissing(String),
    #[error("quantized tensor metadata missing: {0}")]
    QuantizationMissing(String),
    #[error("invalid layer tensor layout: {0}")]
    InvalidLayer(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::NativeTensorDataType;
    use mlx_sys::{MlxDtype, zeros};
    use std::path::Path;

    fn spec(role: NativeTensorRole) -> NativeTensorSpec {
        NativeTensorSpec {
            name: format!("{role:?}"),
            role,
            layer_index: Some(0),
            dtype: NativeTensorDataType::Bf16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![1],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 2,
        }
    }

    #[test]
    fn attention_layout_detects_linear_attention_without_full_attention_roles() {
        let specs = vec![spec(NativeTensorRole::LinearAttentionInProjQkv)];

        let layout = attention_layout_for_layer(&specs, Some(0)).expect("layout should resolve");

        assert_eq!(layout, AttentionLayout::Linear);
    }

    #[test]
    fn attention_layout_defaults_to_full_attention() {
        let specs = vec![spec(NativeTensorRole::AttentionO)];

        let layout = attention_layout_for_layer(&specs, Some(0)).expect("layout should resolve");

        assert_eq!(layout, AttentionLayout::Full);
    }

    #[test]
    fn attention_layout_rejects_mixed_attention_families() {
        let specs = vec![
            spec(NativeTensorRole::AttentionO),
            spec(NativeTensorRole::LinearAttentionInProjQkv),
        ];

        let error = attention_layout_for_layer(&specs, Some(0))
            .expect_err("mixed attention families should fail");

        assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
    }

    #[test]
    fn full_attention_projection_layout_uses_q_only_for_kv_shared_layers() {
        let specs = vec![
            spec(NativeTensorRole::AttentionQ),
            spec(NativeTensorRole::AttentionO),
        ];

        let layout = full_attention_projection_layout(&specs, Some(0), true, false)
            .expect("KV-shared layout should resolve");

        assert_eq!(layout, FullAttentionProjectionLayout::QOnly);
    }

    #[test]
    fn full_attention_projection_layout_uses_qk_for_value_from_key_layers() {
        let specs = vec![
            spec(NativeTensorRole::AttentionQ),
            spec(NativeTensorRole::AttentionK),
            spec(NativeTensorRole::AttentionO),
        ];

        let layout = full_attention_projection_layout(&specs, Some(0), false, true)
            .expect("K=V layout should resolve");

        assert_eq!(layout, FullAttentionProjectionLayout::SplitQkValueFromKey);
    }

    #[test]
    fn full_attention_projection_layout_uses_glm_mla_roles() {
        let specs = vec![
            spec(NativeTensorRole::AttentionQa),
            spec(NativeTensorRole::AttentionQaNorm),
            spec(NativeTensorRole::AttentionQb),
            spec(NativeTensorRole::AttentionKvA),
            spec(NativeTensorRole::AttentionKvANorm),
            spec(NativeTensorRole::AttentionEmbedQ),
            spec(NativeTensorRole::AttentionUnembedOut),
            spec(NativeTensorRole::AttentionO),
        ];

        let layout = full_attention_projection_layout(&specs, Some(0), false, false)
            .expect("GLM MLA layout should resolve");

        assert_eq!(layout, FullAttentionProjectionLayout::GlmMla);
    }

    #[test]
    fn full_attention_projection_layout_rejects_glm_mla_mixed_with_standard_qkv() {
        let specs = vec![
            spec(NativeTensorRole::AttentionQa),
            spec(NativeTensorRole::AttentionQ),
            spec(NativeTensorRole::AttentionO),
        ];

        let error = full_attention_projection_layout(&specs, Some(0), false, false)
            .expect_err("GLM MLA cannot mix with standard QKV projections");

        assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
    }

    #[test]
    fn full_attention_projection_layout_rejects_packed_qkv_for_kv_shared_layers() {
        let specs = vec![spec(NativeTensorRole::AttentionQkvPacked)];

        let error = full_attention_projection_layout(&specs, Some(0), true, false)
            .expect_err("packed QKV cannot represent Q-only KV sharing");

        assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
    }

    #[test]
    fn load_glm_mla_attention_weights_takes_all_reference_roles() {
        let roles = [
            NativeTensorRole::AttentionQa,
            NativeTensorRole::AttentionQaNorm,
            NativeTensorRole::AttentionQb,
            NativeTensorRole::AttentionKvA,
            NativeTensorRole::AttentionKvANorm,
            NativeTensorRole::AttentionEmbedQ,
            NativeTensorRole::AttentionUnembedOut,
        ];
        let specs = roles.iter().copied().map(spec).collect::<Vec<_>>();
        let mut name_map = roles
            .iter()
            .map(|role| (format!("{role:?}"), zeros(&[1, 1], MlxDtype::Float32, None)))
            .collect::<HashMap<_, _>>();

        let weights = load_glm_mla_attention_weights(&specs, &mut name_map, Some(0))
            .expect("GLM MLA weights should load");

        assert_eq!(weights.q_a_norm.shape(), vec![1, 1]);
        assert_eq!(weights.kv_a_norm.shape(), vec![1, 1]);
        assert!(weights.qa_kva_fused.scales.is_none());
        assert!(weights.q_b_proj.scales.is_none());
        assert!(weights.embed_q.scales.is_none());
        assert!(weights.unembed_out.scales.is_none());
        assert!(name_map.is_empty());
    }

    fn glm_quantized_weight(group_size: i32, bits: i32, with_biases: bool) -> QuantizedWeight {
        QuantizedWeight {
            weight: zeros(&[2, 2], MlxDtype::Uint32, None),
            scales: Some(zeros(&[2, 1], MlxDtype::Bfloat16, None)),
            biases: with_biases.then(|| zeros(&[2, 1], MlxDtype::Bfloat16, None)),
            group_size,
            bits,
        }
    }

    fn invalid_layer_message(result: Result<QuantizedWeight, WeightLoadError>) -> String {
        match result {
            Err(WeightLoadError::InvalidLayer(message)) => message,
            Err(error) => panic!("expected invalid layer error, got {error}"),
            Ok(_) => panic!("expected fused GLM MLA weights to be rejected"),
        }
    }

    #[test]
    fn concat_quantized_weight_rows_accepts_matching_quantized_metadata() {
        let a = glm_quantized_weight(64, 4, true);
        let b = glm_quantized_weight(64, 4, true);

        let fused = concat_quantized_weight_rows(&a, &b).expect("matching quantization can fuse");

        assert_eq!(fused.group_size, 64);
        assert_eq!(fused.bits, 4);
        assert!(fused.scales.is_some());
        assert!(fused.biases.is_some());
    }

    #[test]
    fn concat_quantized_weight_rows_rejects_mismatched_group_size() {
        let a = glm_quantized_weight(64, 4, true);
        let b = glm_quantized_weight(32, 4, true);

        let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

        assert!(message.contains("different group sizes"));
    }

    #[test]
    fn concat_quantized_weight_rows_rejects_mismatched_bits() {
        let a = glm_quantized_weight(64, 4, true);
        let b = glm_quantized_weight(64, 8, true);

        let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

        assert!(message.contains("different bit widths"));
    }

    #[test]
    fn concat_quantized_weight_rows_rejects_mismatched_bias_presence() {
        let a = glm_quantized_weight(64, 4, true);
        let b = glm_quantized_weight(64, 4, false);

        let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

        assert!(message.contains("only one has quantization biases"));
    }

    #[test]
    fn concat_quantized_weight_rows_rejects_mixed_dense_and_quantized_weights() {
        let a = QuantizedWeight::new(zeros(&[2, 2], MlxDtype::Float32, None), None, None);
        let b = glm_quantized_weight(64, 4, false);

        let message = invalid_layer_message(concat_quantized_weight_rows(&a, &b));

        assert!(message.contains("only one has quantization scales"));
    }

    #[test]
    fn quantized_weight_uses_tensor_specific_quantization_metadata() {
        let quantization = NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 32,
            bits: 8,
        };
        let weight = zeros(&[1, 1], MlxDtype::Uint32, None);
        let scales = Some(zeros(&[1, 1], MlxDtype::Bfloat16, None));

        let quantized =
            QuantizedWeight::with_quantization(weight, scales, None, Some(&quantization));

        assert_eq!(quantized.group_size, 32);
        assert_eq!(quantized.bits, 8);
    }

    #[test]
    fn take_weight_preserves_tensor_specific_quantization_metadata() {
        let mut router = spec(NativeTensorRole::FfnGateInp);
        router.name = "model.layers.0.router.proj.weight".to_string();
        router.dtype = NativeTensorDataType::U32;
        router.source_quantized = true;
        router.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 8,
        });
        let specs = vec![router];
        let mut name_map = HashMap::from([
            (
                "model.layers.0.router.proj.weight".to_string(),
                zeros(&[128, 704], MlxDtype::Uint32, None),
            ),
            (
                "model.layers.0.router.proj.scales".to_string(),
                zeros(&[128, 44], MlxDtype::Bfloat16, None),
            ),
        ]);

        let weight = take_weight(
            &specs,
            &mut name_map,
            NativeTensorRole::FfnGateInp,
            Some(0),
            "router_proj",
        )
        .expect("quantized router should load");

        assert_eq!(weight.group_size, 64);
        assert_eq!(weight.bits, 8);
        assert!(weight.scales.is_some());
    }

    #[test]
    fn take_weight_rejects_quantized_tensor_without_scales() {
        let mut router = spec(NativeTensorRole::FfnGateInp);
        router.name = "model.layers.0.router.proj.weight".to_string();
        router.dtype = NativeTensorDataType::U32;
        router.source_quantized = true;
        router.quantization = Some(NativeTensorQuantization {
            mode: "affine".to_string(),
            group_size: 64,
            bits: 8,
        });
        let specs = vec![router];
        let mut name_map = HashMap::from([(
            "model.layers.0.router.proj.weight".to_string(),
            zeros(&[128, 704], MlxDtype::Uint32, None),
        )]);

        let error = match take_weight(
            &specs,
            &mut name_map,
            NativeTensorRole::FfnGateInp,
            Some(0),
            "router_proj",
        ) {
            Ok(_) => panic!("quantized MLX tensors require co-located scales"),
            Err(error) => error,
        };

        assert!(matches!(error, WeightLoadError::QuantizationMissing(_)));
    }

    #[test]
    fn take_weight_rejects_quantization_sidecars_when_manifest_is_dense() {
        let mut router = spec(NativeTensorRole::FfnGateInp);
        router.name = "model.layers.0.router.proj.weight".to_string();
        router.dtype = NativeTensorDataType::Bf16;
        router.source_quantized = false;
        let specs = vec![router];
        let mut name_map = HashMap::from([
            (
                "model.layers.0.router.proj.weight".to_string(),
                zeros(&[128, 2816], MlxDtype::Bfloat16, None),
            ),
            (
                "model.layers.0.router.proj.scales".to_string(),
                zeros(&[128, 44], MlxDtype::Bfloat16, None),
            ),
        ]);

        let error = match take_weight(
            &specs,
            &mut name_map,
            NativeTensorRole::FfnGateInp,
            Some(0),
            "router_proj",
        ) {
            Ok(_) => panic!("dense manifest tensors must not consume quantization sidecars"),
            Err(error) => error,
        };

        assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
    }

    #[test]
    fn real_mlx_weights_load_qwen35_linear_attention_when_configured() {
        if std::env::var("AX_ENGINE_MLX_LOAD_REAL_WEIGHTS").as_deref() != Ok("1") {
            return;
        }
        let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
            return;
        };
        let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
            .expect("real MLX manifest should load");

        let weights = load_weights(&artifacts).expect("real MLX weights should load");

        assert_eq!(
            weights.layers.len(),
            artifacts.manifest().layer_count as usize
        );
        assert!(
            weights
                .layers
                .first()
                .and_then(|layer| layer.linear_attn.as_ref())
                .is_some(),
            "Qwen3.5 layer 0 should load linear-attention weights"
        );
    }
}
