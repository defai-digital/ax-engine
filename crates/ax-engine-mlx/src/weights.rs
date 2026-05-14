use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;

use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, contiguous, dequantize, eval, load_safetensors,
    multiply, transpose,
};

use ax_engine_core::{
    NativeModelArtifacts, NativeTensorQuantization, NativeTensorRole, NativeTensorSpec,
    WeightSanitize,
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
    /// Per-layer AWQ-lite smoothing reciprocal `1/s` of shape `[hidden_size]`.
    /// Populated by `apply_rotated_checkpoint` when the rotated checkpoint was
    /// generated with `--smoothing weight_mag`. The forward path multiplies
    /// the rotated activation by this vector before the gate/up matmul so
    /// the per-channel scaling baked into the rotated weights cancels.
    pub rotation_smoothing_inverse: Option<MlxArray>,
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
    /// Conv1d kernel dequantized at load time so `linear_attention_forward` never
    /// re-dequantizes per step. Shape: `[conv_dim, conv_kernel_dim, 1]`.
    pub conv1d_dense: MlxArray,
    pub dt_bias: MlxArray,
    pub a_log: MlxArray,
    pub norm: MlxArray,
    pub out_proj: QuantizedWeight,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinearAttentionProjectionRowSource {
    Qkv(usize),
    Z(usize),
    B(usize),
    A(usize),
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
    // AX_MMAP_WEIGHTS=1 uses the memory-mapped safetensors path. No bytes
    // are read into a heap buffer up front; pages are pulled in by the
    // OS on first access (CPU touch or GPU dispatch). On warm page cache
    // this is roughly equivalent to the C loader; on cold disk it lets
    // the kernel decide when to read what, which can roughly halve cold
    // startup time for large models. The default remains the C loader
    // until the mmap path has wider integration test coverage.
    let use_mmap = std::env::var("AX_MMAP_WEIGHTS")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);
    let mut file_cache: HashMap<PathBuf, HashMap<String, MlxArray>> = HashMap::new();
    for spec in artifacts.tensor_specs() {
        let full = root.join(&spec.file);
        if let Entry::Vacant(entry) = file_cache.entry(full) {
            let path = entry.key().clone();
            let tensors = if use_mmap {
                mlx_sys::load_safetensors_mmap(&path).map_err(WeightLoadError::FileMissing)?
            } else {
                load_safetensors(&path, None).map_err(WeightLoadError::FileMissing)?
            };
            if tensors.is_empty() {
                return Err(WeightLoadError::FileMissing(path.display().to_string()));
            }
            // Both loaders need an explicit eval here. The C loader path
            // builds in-memory MLX arrays that haven't been routed
            // through the lazy graph yet; the mmap path needs the eval
            // to wire the page-backed buffers into MLX's working set so
            // GPU dispatches see initialised data. (Without it, GPU
            // reads return uninitialised memory → NaN outputs.)
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

    // Raw HuggingFace checkpoints store RMSNorm weights as zero-centered deltas
    // and conv1d weights in a different axis order than MLX expects. The MLX
    // community fork pre-applies these transforms; raw HF checkpoints need them
    // applied here. The manifest's `weight_sanitize` field selects the path.
    //
    // When the manifest leaves `weight_sanitize=None` (the default emitted by
    // `convert_hf_model_dir`), some mlx-community quantized hybrid models
    // (e.g. `Qwen3-Coder-Next-4bit`) still ship unsanitized norm weights
    // because mlx_lm runs `sanitize()` at load time rather than persisting
    // the +1.0 baseline on disk. Re-running `mlx_lm.convert` on an already
    // quantized checkpoint dequantizes → normalizes → re-quantizes and
    // produces garbage weights, so auto-detection here is the only viable
    // recovery path.
    let effective_sanitize = match artifacts.manifest().weight_sanitize {
        WeightSanitize::None => auto_detect_linear_attention_sanitize(specs, &name_map),
        explicit => explicit,
    };
    match effective_sanitize {
        WeightSanitize::HfToMlx => apply_hf_sanitize_transforms(specs, &mut name_map, true),
        WeightSanitize::HfNormOnly => apply_hf_sanitize_transforms(specs, &mut name_map, false),
        WeightSanitize::None => {}
    }

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
            rotation_smoothing_inverse: None,
        });
    }

    // Raw HF Qwen3.5/Next checkpoints store several RMSNorm weights as zero-centred
    // deltas that require `mlx_lm.convert` sanitization.  Fail closed instead of
    // partially fixing one norm and silently leaving the others wrong.
    for (idx, layer) in layers.iter().enumerate() {
        if let Some(la) = layer.linear_attn.as_ref() {
            ensure_sanitized_linear_attention_norm(idx, &la.norm)?;
            ensure_conv1d_mlx_layout(idx, &la.conv1d_dense)?;
        }
    }

    crate::weight_rotation::shadow_log_rotation_candidates(specs);

    let mut model = ModelWeights {
        token_embedding,
        final_norm,
        lm_head,
        layers,
        per_layer_embed,
        per_layer_model_proj,
        per_layer_proj_norm,
    };

    apply_rotated_checkpoint(&mut model, artifacts)?;

    Ok(model)
}

/// Detect a sibling `model.rotated.safetensors` and, when the runtime is in
/// `WeightRotationMode::Apply`, replace each layer's `gate_proj` / `up_proj`
/// with the rotated f32 version produced offline by
/// `scripts/quantize_rotated_weights.py --apply`. Fail-closed: if Apply mode
/// is selected but the rotated checkpoint is missing / incomplete / not
/// applicable, return an error rather than silently running with broken math.
fn apply_rotated_checkpoint(
    model: &mut ModelWeights,
    artifacts: &NativeModelArtifacts,
) -> Result<(), WeightLoadError> {
    use crate::weight_rotation::{WeightRotationMode, weight_rotation_mode};
    if weight_rotation_mode() != WeightRotationMode::Apply {
        return Ok(());
    }
    let rotated_path = artifacts.root_dir().join("model.rotated.safetensors");
    if !rotated_path.is_file() {
        return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
            "AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION=apply requires {} (run scripts/quantize_rotated_weights.py --apply first)",
            rotated_path.display()
        )));
    }
    let rotated = load_safetensors(&rotated_path, None).map_err(|e| {
        WeightLoadError::RotatedCheckpointInvalid(format!("{}: {}", rotated_path.display(), e))
    })?;
    if rotated.is_empty() {
        return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
            "{} is empty",
            rotated_path.display()
        )));
    }

    let mut replaced = 0usize;
    let mut smoothing_loaded = 0usize;
    let mut missing_layers: Vec<usize> = Vec::new();
    for (layer_idx, layer) in model.layers.iter_mut().enumerate() {
        // Per-layer AWQ-lite smoothing vector, if the offline tool produced
        // one. Stored as `ax_smoothing.layers.{idx}` of shape [hidden_size]
        // holding 1/s already (so the forward path multiplies, not divides).
        let smoothing_key = format!("ax_smoothing.layers.{}", layer_idx);
        if let Some(smoothing) = rotated.get(&smoothing_key) {
            layer.rotation_smoothing_inverse = Some(smoothing.clone());
            smoothing_loaded += 1;
        }
        for (suffix, slot) in [
            ("gate_proj", &mut layer.gate_proj),
            ("up_proj", &mut layer.up_proj),
        ] {
            let Some(target) = slot.as_mut() else {
                continue;
            };
            // Standard MLX-community Qwen naming. Other families would need
            // alternative key probes added here as P2a expands beyond Qwen.
            let key = format!(
                "language_model.model.layers.{}.mlp.{}.weight",
                layer_idx, suffix
            );
            let Some(rotated_w) = rotated.get(&key) else {
                if layer.gate_up_packed.is_none() {
                    missing_layers.push(layer_idx);
                }
                continue;
            };
            // Rotated tensors are EITHER stored as f32 (P2a baseline) or
            // re-quantized as u32-packed plus sibling .scales / .biases (P2b).
            // Detect by weight dtype: u32 => quantized path; anything else
            // => plain f32 path with cast to bf16.
            let stem = format!("language_model.model.layers.{}.mlp.{}", layer_idx, suffix);
            let scales_key = format!("{stem}.scales");
            let biases_key = format!("{stem}.biases");
            if rotated_w.dtype() == MlxDtype::Uint32 {
                let scales = rotated.get(&scales_key).ok_or_else(|| {
                    WeightLoadError::RotatedCheckpointInvalid(format!(
                        "missing {scales_key} for u32-packed rotated weight"
                    ))
                })?;
                let biases = rotated.get(&biases_key).ok_or_else(|| {
                    WeightLoadError::RotatedCheckpointInvalid(format!(
                        "missing {biases_key} for u32-packed rotated weight"
                    ))
                })?;
                // Infer bits from packed shape vs logical dim. For u32 packing
                // with `bits` bits per element: packed_inner = logical_inner * bits / 32.
                // logical_inner is the dim the activation rotation acts on; we
                // know the original weight's logical inner was rotation_dim,
                // which equals `original.weight` last-axis * 8 (4-bit packed)
                // or equivalent. Compute from this checkpoint's shape directly.
                let packed_shape = rotated_w.shape();
                let scales_shape = scales.shape();
                // logical_inner = group_size * groups_per_row, with groups_per_row
                // = scales last axis. Hardcoded group_size 64 (matches script).
                let groups_per_row = *scales_shape.last().unwrap_or(&0) as i64;
                let logical_inner = groups_per_row * 64;
                let packed_inner = *packed_shape.last().unwrap_or(&0) as i64;
                if logical_inner <= 0 || packed_inner <= 0 {
                    return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
                        "{key}: cannot infer bits from shapes packed={packed_shape:?} scales={scales_shape:?}"
                    )));
                }
                let bits_calc = packed_inner * 32 / logical_inner;
                if !(2..=8).contains(&bits_calc) {
                    return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
                        "{key}: inferred bits={bits_calc} outside 2..=8"
                    )));
                }
                target.weight = rotated_w.clone();
                target.scales = Some(scales.clone());
                target.biases = Some(biases.clone());
                target.bits = bits_calc as i32;
                target.group_size = 64;
            } else {
                // f32 path: cast to bf16, drop scales/biases, forward picks
                // the plain matmul branch.
                let cast = astype(rotated_w, MlxDtype::Bfloat16, None);
                target.weight = cast;
                target.scales = None;
                target.biases = None;
            }
            replaced += 1;
        }
    }

    if replaced == 0 {
        return Err(WeightLoadError::RotatedCheckpointInvalid(format!(
            "{} contained 0 matching tensors for this model (key format mismatch?)",
            rotated_path.display()
        )));
    }

    tracing::info!(
        target: "ax_mlx::weight_rotation",
        replaced_tensor_count = replaced,
        smoothing_loaded = smoothing_loaded,
        rotated_checkpoint = %rotated_path.display(),
        "loaded rotated checkpoint"
    );
    eprintln!(
        "[ax_mlx::weight_rotation apply] loaded {}: {} tensors replaced, {} layers with smoothing",
        rotated_path.display(),
        replaced,
        smoothing_loaded,
    );
    Ok(())
}

/// Apply the `hf_to_mlx` weight transforms in place on a freshly loaded
/// safetensors `name_map`.
///
/// Raw HuggingFace checkpoints store RMSNorm weights as zero-centered deltas
/// (the "+1.0" is folded into the model's runtime forward pass). MLX expects
/// the weight to already be the multiplier. We add 1.0 to every norm-role
/// tensor here so downstream `rms_norm()` calls produce identical results.
///
/// HF also stores conv1d projection weights with axes (out, in, kernel)
/// while MLX expects (out, kernel, in). The `transpose(_, [0, 2, 1])` swap
/// brings them into MLX layout.
///
/// The companion `ensure_sanitized_linear_attention_norm` check downstream
/// validates these transforms succeeded; it remains as the safety net for
/// the `WeightSanitize::None` path where a manifest mis-declares its layout.
fn apply_hf_sanitize_transforms(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    swap_conv1d: bool,
) {
    let one = MlxArray::from_f32_slice(&[1.0_f32]);
    for spec in specs {
        if !name_map.contains_key(&spec.name) {
            continue;
        }
        let transformed = match spec.role {
            NativeTensorRole::AttentionNorm
            | NativeTensorRole::AttentionPostNorm
            | NativeTensorRole::AttentionQNorm
            | NativeTensorRole::AttentionKNorm
            | NativeTensorRole::AttentionQaNorm
            | NativeTensorRole::AttentionKvANorm
            | NativeTensorRole::LinearAttentionNorm
            | NativeTensorRole::FfnNorm
            | NativeTensorRole::FfnNorm2
            | NativeTensorRole::FfnPostNorm
            | NativeTensorRole::FfnPostNorm1
            | NativeTensorRole::FfnPostNorm2
            | NativeTensorRole::PerLayerProjectionNorm
            | NativeTensorRole::PerLayerInputPostNorm
            | NativeTensorRole::FinalNorm => {
                let tensor = name_map.get(&spec.name).expect("checked via contains_key");
                // MLX promotes (bf16/f16 + f32) to f32, which would silently
                // change the stored dtype of the norm weight. Cast back to the
                // original dtype so downstream consumers see the same shape
                // and dtype they would have for an already-sanitized
                // mlx-community weight.
                let original_dtype = tensor.dtype();
                Some(astype(&add(tensor, &one, None), original_dtype, None))
            }
            NativeTensorRole::LinearAttentionConv1d if swap_conv1d => {
                let tensor = name_map.get(&spec.name).expect("checked via contains_key");
                // `transpose` returns a stride-only view; consumers that read
                // the raw buffer (e.g. the conv1d kernel) need a contiguous
                // layout, so materialize here.
                Some(contiguous(&transpose(tensor, &[0, 2, 1], None), None))
            }
            _ => None,
        };
        if let Some(new_tensor) = transformed {
            name_map.insert(spec.name.clone(), new_tensor);
        }
    }
    // Force evaluation so subsequent inspection (e.g.
    // ensure_sanitized_linear_attention_norm) sees materialised values rather
    // than the lazy MLX op graph.
    let refs: Vec<&MlxArray> = name_map.values().collect();
    eval(&refs);
}

/// Lower bound on `mean_abs` of a sanitized RMSNorm weight (post `+1.0` lift).
///
/// mlx_lm-sanitized norms cluster tightly around `1.0` (typically `1.0 ± 0.1`);
/// the raw HF zero-centred-delta form clusters around `0.0` (typically `< 0.05`).
/// `0.15` is comfortably between both modes — well above the deltas yet far
/// below any plausible sanitized weight, so it doubles as the auto-detect
/// trigger and the post-load fail-closed assertion.
const SANITIZED_NORM_MIN_MEAN_ABS: f32 = 0.15;

fn ensure_sanitized_linear_attention_norm(
    layer_index: usize,
    norm: &MlxArray,
) -> Result<(), WeightLoadError> {
    if let Some(mean_abs) = linear_attention_norm_mean_abs(norm)
        && mean_abs < SANITIZED_NORM_MIN_MEAN_ABS
    {
        return Err(WeightLoadError::UnsanitizedWeights(format!(
            "linear attention layer {layer_index} norm mean_abs={mean_abs:.6}; \
             expected ~1.0 after the HF→MLX +1.0 transform. \
             Auto-detection should have caught this — if you see this error, \
             the layer-0 linear-attention norm tensor was not sampled at load \
             time (perhaps missing role tag in the manifest). Set the manifest \
             `weight_sanitize` to `\"hf_norm_only\"` (conv1d already in MLX \
             layout) or `\"hf_to_mlx\"` (raw HF conv1d) and reload."
        )));
    }
    Ok(())
}

/// Compute mean_abs of a norm tensor as f32, or `None` if the sample is too
/// small to draw a conclusion. Shared by load-time auto-detection and the
/// post-sanitize verification.
fn linear_attention_norm_mean_abs(norm: &MlxArray) -> Option<f32> {
    let f32_norm = astype(norm, MlxDtype::Float32, None);
    eval(&[&f32_norm]);
    let data = f32_norm.data_f32();
    if data.len() < 8 {
        return None;
    }
    Some(data.iter().map(|v| v.abs()).sum::<f32>() / data.len() as f32)
}

/// When the manifest sets `weight_sanitize=None`, peek at the lowest-indexed
/// linear-attention layer's norm and conv1d tensors to decide whether the
/// weights on disk are actually pre-sanitized.
///
/// Returns the sanitize mode to apply:
/// - `None`: weights look sanitized (norm baseline near 1.0, conv1d in MLX layout)
/// - `HfNormOnly`: norm needs +1.0 but conv1d is already MLX layout (mlx-community
///   quantized hybrid models: Qwen3-Coder-Next-4bit, Qwen3.5-9B-4bit, …)
/// - `HfToMlx`: both norm and conv1d are raw HF (rare for distributed mlx checkpoints)
///
/// For non-hybrid models (no linear-attention layers) this returns `None`
/// because there is no signal — and historically mlx-community pre-sanitizes
/// the non-hybrid families before publishing, so leaving sanitize off is safe.
fn auto_detect_linear_attention_sanitize(
    specs: &[NativeTensorSpec],
    name_map: &HashMap<String, MlxArray>,
) -> WeightSanitize {
    // Sample the lowest-indexed linear-attention norm rather than hardcoding
    // layer 0 — hybrid architectures may interleave a full-attention layer at
    // index 0, in which case there is no LinearAttentionNorm at that index.
    let norm_spec = specs
        .iter()
        .filter(|s| matches!(s.role, NativeTensorRole::LinearAttentionNorm))
        .min_by_key(|s| s.layer_index.unwrap_or(u32::MAX));
    let Some(norm_spec) = norm_spec else {
        return WeightSanitize::None;
    };
    let Some(norm) = name_map.get(&norm_spec.name) else {
        return WeightSanitize::None;
    };
    let Some(mean_abs) = linear_attention_norm_mean_abs(norm) else {
        return WeightSanitize::None;
    };
    let norm_needs_sanitize = mean_abs < SANITIZED_NORM_MIN_MEAN_ABS;

    let sample_layer = norm_spec.layer_index;
    let conv1d_spec = specs
        .iter()
        .find(|s| {
            matches!(s.role, NativeTensorRole::LinearAttentionConv1d)
                && s.layer_index == sample_layer
        })
        .or_else(|| {
            specs
                .iter()
                .filter(|s| matches!(s.role, NativeTensorRole::LinearAttentionConv1d))
                .min_by_key(|s| s.layer_index.unwrap_or(u32::MAX))
        });
    let conv1d_needs_swap = conv1d_spec
        .and_then(|spec| name_map.get(&spec.name))
        .map(|conv1d| {
            let shape = conv1d.shape();
            // HF layout: [conv_dim, in=1, kernel]. MLX layout: [conv_dim, kernel, in=1].
            // Detect HF by `shape[1] == 1 && shape[2] != 1` to avoid the
            // ambiguous all-ones edge case (a 1x1 conv has shape [_, 1, 1]).
            shape.len() == 3 && shape[1] == 1 && shape[2] != 1
        })
        .unwrap_or(false);

    let chosen = match (norm_needs_sanitize, conv1d_needs_swap) {
        (false, false) => WeightSanitize::None,
        (true, false) => WeightSanitize::HfNormOnly,
        (true, true) => WeightSanitize::HfToMlx,
        // Norm looks sanitized but conv1d is still HF layout. A partially
        // transformed checkpoint should not be silently patched — let the
        // downstream `ensure_conv1d_mlx_layout` check fire with its
        // diagnostic.
        (false, true) => WeightSanitize::None,
    };

    if !matches!(chosen, WeightSanitize::None) {
        tracing::warn!(
            target: "ax_mlx::weights",
            mean_abs = mean_abs,
            conv1d_needs_swap = conv1d_needs_swap,
            sanitize = ?chosen,
            "manifest weight_sanitize=None but on-disk linear-attention weights look unsanitized; \
             applying {chosen:?}. Set weight_sanitize explicitly in model-manifest.json to silence \
             this warning."
        );
        eprintln!(
            "[ax_mlx::weights] auto-detected unsanitized linear-attention weights \
             (norm mean_abs={mean_abs:.6}, conv1d_hf_layout={conv1d_needs_swap}); \
             applying {chosen:?} sanitize transform. Set weight_sanitize in \
             model-manifest.json to silence this warning."
        );
    }

    chosen
}

/// Verify conv1d is in MLX layout `[conv_dim, kernel, 1]` (last dim = 1).
///
/// Catches manifests that mis-declare `weight_sanitize` and leave conv1d in
/// HuggingFace layout `[conv_dim, 1, kernel]`, which would produce silently
/// wrong conv outputs without any other runtime error.
fn ensure_conv1d_mlx_layout(layer_index: usize, conv1d: &MlxArray) -> Result<(), WeightLoadError> {
    let shape = conv1d.shape();
    if shape.len() != 3 || shape[2] != 1 {
        return Err(WeightLoadError::UnsanitizedWeights(format!(
            "linear attention layer {layer_index} conv1d shape {shape:?}: expected \
             [conv_dim, kernel, 1] (MLX layout). Raw HuggingFace checkpoints store conv1d as \
             [conv_dim, 1, kernel]; set weight_sanitize to \"hf_to_mlx\" or run \
             mlx_lm.convert on the unquantized source weights first \
             (re-running mlx_lm.convert on an already-quantized checkpoint corrupts the weights)."
        )));
    }
    Ok(())
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
        conv1d_dense: {
            let raw = take_weight(
                specs,
                name_map,
                NativeTensorRole::LinearAttentionConv1d,
                layer_index,
                "linear_attention_conv1d",
            )?;
            if let Some(scales) = &raw.scales {
                dequantize(
                    &raw.weight,
                    scales,
                    raw.biases.as_ref(),
                    Some(raw.group_size),
                    Some(raw.bits),
                    None,
                )
            } else {
                raw.weight
            }
        },
        // Cast at load time so the per-step linear_attention_forward does not
        // pay an astype dispatch for each layer. `gated_delta_kernel` expects
        // both as f32, matching mlx_lm's reference behaviour. For a 12-layer
        // hybrid model this removes ~24 small astype ops per decode step.
        dt_bias: astype(
            &take_weight(
                specs,
                name_map,
                NativeTensorRole::LinearAttentionDtBias,
                layer_index,
                "linear_attention_dt_bias",
            )?
            .weight,
            MlxDtype::Float32,
            None,
        ),
        a_log: astype(
            &take_weight(
                specs,
                name_map,
                NativeTensorRole::LinearAttentionALog,
                layer_index,
                "linear_attention_a_log",
            )?
            .weight,
            MlxDtype::Float32,
            None,
        ),
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
/// and kv_a_proj in GLM MLA, or Q/K/V in standard full-attention), replacing
/// multiple matmul kernel launches with one.
fn concat_quantized_weight_rows(
    a: &QuantizedWeight,
    b: &QuantizedWeight,
) -> Result<QuantizedWeight, WeightLoadError> {
    let (scales, biases) = match (&a.scales, &b.scales) {
        (Some(sa), Some(sb)) => {
            if a.group_size != b.group_size {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "cannot fuse quantized projections with different group sizes: {} vs {}",
                    a.group_size, b.group_size
                )));
            }
            if a.bits != b.bits {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "cannot fuse quantized projections with different bit widths: {} vs {}",
                    a.bits, b.bits
                )));
            }
            let biases = match (a.biases.as_ref(), b.biases.as_ref()) {
                (Some(ba), Some(bb)) => Some(concatenate(&[ba, bb], 0, None)),
                (None, None) => None,
                _ => {
                    return Err(WeightLoadError::InvalidLayer(
                        "cannot fuse projections where only one has quantization biases"
                            .to_string(),
                    ));
                }
            };
            (Some(concatenate(&[sa, sb], 0, None)), biases)
        }
        (None, None) => (None, None),
        _ => {
            return Err(WeightLoadError::InvalidLayer(
                "cannot fuse projections where only one has quantization scales".to_string(),
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

#[cfg(test)]
fn linear_attention_qkvz_row_sources(
    num_key_heads: usize,
    key_head_dim: usize,
    num_value_heads: usize,
    value_head_dim: usize,
) -> Result<Vec<LinearAttentionProjectionRowSource>, WeightLoadError> {
    if num_key_heads == 0 || key_head_dim == 0 || num_value_heads == 0 || value_head_dim == 0 {
        return Err(WeightLoadError::InvalidLayer(
            "linear attention projection pack dimensions must be non-zero".to_string(),
        ));
    }
    if !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "linear attention projection pack requires value heads divisible by key heads: {num_value_heads} vs {num_key_heads}"
        )));
    }

    let key_dim = num_key_heads * key_head_dim;
    let value_heads_per_key = num_value_heads / num_key_heads;
    let value_dim_per_key = value_heads_per_key * value_head_dim;
    let q_base = 0;
    let k_base = key_dim;
    let v_base = key_dim * 2;
    let mut rows = Vec::with_capacity(key_dim * 2 + num_value_heads * value_head_dim * 2);

    for key_head in 0..num_key_heads {
        let q_start = q_base + key_head * key_head_dim;
        let k_start = k_base + key_head * key_head_dim;
        let value_start = key_head * value_dim_per_key;
        rows.extend((q_start..q_start + key_head_dim).map(LinearAttentionProjectionRowSource::Qkv));
        rows.extend((k_start..k_start + key_head_dim).map(LinearAttentionProjectionRowSource::Qkv));
        rows.extend(
            (v_base + value_start..v_base + value_start + value_dim_per_key)
                .map(LinearAttentionProjectionRowSource::Qkv),
        );
        rows.extend(
            (value_start..value_start + value_dim_per_key)
                .map(LinearAttentionProjectionRowSource::Z),
        );
    }
    Ok(rows)
}

#[cfg(test)]
fn linear_attention_ba_row_sources(
    num_key_heads: usize,
    num_value_heads: usize,
) -> Result<Vec<LinearAttentionProjectionRowSource>, WeightLoadError> {
    if num_key_heads == 0 || num_value_heads == 0 {
        return Err(WeightLoadError::InvalidLayer(
            "linear attention BA pack dimensions must be non-zero".to_string(),
        ));
    }
    if !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "linear attention BA pack requires value heads divisible by key heads: {num_value_heads} vs {num_key_heads}"
        )));
    }

    let value_heads_per_key = num_value_heads / num_key_heads;
    let mut rows = Vec::with_capacity(num_value_heads * 2);
    for key_head in 0..num_key_heads {
        let value_start = key_head * value_heads_per_key;
        rows.extend(
            (value_start..value_start + value_heads_per_key)
                .map(LinearAttentionProjectionRowSource::B),
        );
        rows.extend(
            (value_start..value_start + value_heads_per_key)
                .map(LinearAttentionProjectionRowSource::A),
        );
    }
    Ok(rows)
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
    #[error("unsanitized weights: {0}")]
    UnsanitizedWeights(String),
    #[error("rotated checkpoint required but invalid: {0}")]
    RotatedCheckpointInvalid(String),
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
    fn unsanitized_linear_attention_norm_is_rejected() {
        let norm = zeros(&[8], MlxDtype::Float32, None);

        let error = ensure_sanitized_linear_attention_norm(2, &norm)
            .expect_err("zero-centred raw HF norm should fail closed");

        let WeightLoadError::UnsanitizedWeights(message) = error else {
            panic!("expected unsanitized weights error");
        };
        assert!(message.contains("layer 2"));
        assert!(message.contains("mean_abs=0.000000"));
        assert!(
            message.contains("weight_sanitize"),
            "error should mention manifest weight_sanitize knob"
        );
    }

    #[test]
    fn sanitized_linear_attention_norm_is_allowed() {
        let data = [1.0_f32; 8];
        let norm = MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(&data),
            &[8],
            MlxDtype::Float32,
        );

        ensure_sanitized_linear_attention_norm(0, &norm)
            .expect("mlx_lm-sanitized norm should load");
    }

    #[test]
    fn hf_layout_conv1d_is_rejected() {
        // HuggingFace stores conv1d as [conv_dim, in=1, kernel]. A manifest
        // that mis-declares weight_sanitize would skip the axis swap, producing
        // silently wrong conv outputs. The check must catch this at load time.
        let conv1d = zeros(&[64, 1, 4], MlxDtype::Float32, None);

        let error = ensure_conv1d_mlx_layout(3, &conv1d)
            .expect_err("HF-layout conv1d [conv_dim, 1, kernel] must be rejected");

        let WeightLoadError::UnsanitizedWeights(message) = error else {
            panic!("expected unsanitized weights error");
        };
        assert!(message.contains("layer 3"));
        assert!(message.contains("[64, 1, 4]"));
        assert!(message.contains("mlx_lm.convert"));
    }

    #[test]
    fn mlx_layout_conv1d_is_accepted() {
        // MLX layout: [conv_dim, kernel, in=1]. Both the HfToMlx and HfNormOnly
        // sanitization paths produce this shape; the check must allow it.
        let conv1d = zeros(&[64, 4, 1], MlxDtype::Float32, None);

        ensure_conv1d_mlx_layout(0, &conv1d)
            .expect("MLX-layout conv1d [conv_dim, kernel, 1] should load");
    }

    #[test]
    fn apply_hf_sanitize_transforms_lifts_norm_deltas_and_swaps_conv1d_axes() {
        // A raw HuggingFace checkpoint stores norm weights as zero-centered
        // deltas (so the "weight = 1.0 + delta" multiplier is materialised
        // by the runtime forward path). The sanitizer must restore the +1.0
        // baseline before loading.
        let delta = [-0.1_f32, 0.2, 0.05, 0.0];
        let norm_delta = MlxArray::from_raw_data(
            delta.as_ptr().cast(),
            std::mem::size_of_val(&delta),
            &[delta.len() as i32],
            MlxDtype::Float32,
        );

        // Conv1d weight in HF axis order (out, in, kernel) = (2, 3, 4).
        // Encode coordinates into values: data[o, i, k] = 100*o + 10*i + k.
        // After moveaxis(2, 1) MLX expects (out, kernel, in) = (2, 4, 3),
        // and the value at new[o, k, i] must equal 100*o + 10*i + k.
        const OUT_DIM: usize = 2;
        const IN_DIM: usize = 3;
        const KERNEL_DIM: usize = 4;
        let mut conv = [0.0_f32; OUT_DIM * IN_DIM * KERNEL_DIM];
        for o in 0..OUT_DIM {
            for i in 0..IN_DIM {
                for k in 0..KERNEL_DIM {
                    conv[o * IN_DIM * KERNEL_DIM + i * KERNEL_DIM + k] =
                        (100 * o + 10 * i + k) as f32;
                }
            }
        }
        let conv1d_hf = MlxArray::from_raw_data(
            conv.as_ptr().cast(),
            std::mem::size_of_val(&conv),
            &[OUT_DIM as i32, IN_DIM as i32, KERNEL_DIM as i32],
            MlxDtype::Float32,
        );

        let mut name_map: HashMap<String, MlxArray> = HashMap::new();
        name_map.insert("layers.0.attn_norm".to_string(), norm_delta);
        name_map.insert("layers.0.conv1d".to_string(), conv1d_hf);

        fn make_spec(name: &str, role: NativeTensorRole) -> NativeTensorSpec {
            NativeTensorSpec {
                name: name.to_string(),
                role,
                layer_index: Some(0),
                dtype: NativeTensorDataType::F32,
                source_tensor_type: None,
                source_quantized: false,
                quantization: None,
                quantized_source: None,
                shape: vec![1],
                file: PathBuf::from("model.safetensors"),
                offset_bytes: 0,
                length_bytes: 4,
            }
        }
        let specs = vec![
            make_spec("layers.0.attn_norm", NativeTensorRole::AttentionNorm),
            make_spec("layers.0.conv1d", NativeTensorRole::LinearAttentionConv1d),
        ];

        apply_hf_sanitize_transforms(&specs, &mut name_map, true);

        let sanitized_norm = name_map
            .get("layers.0.attn_norm")
            .expect("norm tensor must still be present");
        let norm_values = sanitized_norm.data_f32();
        for (got, want) in norm_values.iter().zip([0.9_f32, 1.2, 1.05, 1.0].iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "norm sanitize: got {got}, want {want}"
            );
        }

        let sanitized_conv = name_map
            .get("layers.0.conv1d")
            .expect("conv1d tensor must still be present");
        assert_eq!(
            sanitized_conv.shape(),
            vec![OUT_DIM as i32, KERNEL_DIM as i32, IN_DIM as i32],
            "conv1d axes should swap from (out, in, kernel) to (out, kernel, in)"
        );
        // Verify every coordinate: new[o, k, i] must equal the encoded
        // coordinate 100*o + 10*i + k (note: encoding uses original axis
        // assignments, so the value identifies the source element).
        let conv_values = sanitized_conv.data_f32();
        for o in 0..OUT_DIM {
            for k in 0..KERNEL_DIM {
                for i in 0..IN_DIM {
                    let flat = o * KERNEL_DIM * IN_DIM + k * IN_DIM + i;
                    let want = (100 * o + 10 * i + k) as f32;
                    let got = conv_values[flat];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "transposed conv[o={o}, k={k}, i={i}] at flat[{flat}]: got {got}, want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn apply_hf_sanitize_transforms_hf_norm_only_lifts_norm_but_preserves_conv1d_axes() {
        // Qwen3-Coder-Next ships with conv1d already in MLX layout (out, kernel, in)
        // but RMSNorm weights are still HF-style zero-centred deltas. The
        // HfNormOnly path must add +1.0 to norms without swapping conv1d axes.
        const OUT: usize = 2;
        const KERNEL: usize = 4;
        const IN: usize = 1;

        let norm_data = [-0.1_f32, 0.2, 0.05, 0.0];
        let norm = MlxArray::from_raw_data(
            norm_data.as_ptr().cast(),
            std::mem::size_of_val(&norm_data),
            &[norm_data.len() as i32],
            MlxDtype::Float32,
        );

        // Conv1d already in MLX layout (out, kernel, in) = (2, 4, 1)
        let conv_mlx_data: Vec<f32> = (0..OUT * KERNEL * IN).map(|i| i as f32).collect();
        let conv_mlx = MlxArray::from_raw_data(
            conv_mlx_data.as_ptr().cast(),
            std::mem::size_of_val(conv_mlx_data.as_slice()),
            &[OUT as i32, KERNEL as i32, IN as i32],
            MlxDtype::Float32,
        );

        let mut name_map: HashMap<String, MlxArray> = HashMap::new();
        name_map.insert("attn_norm".to_string(), norm);
        name_map.insert("conv1d".to_string(), conv_mlx);

        fn make_spec(name: &str, role: NativeTensorRole) -> NativeTensorSpec {
            NativeTensorSpec {
                name: name.to_string(),
                role,
                layer_index: Some(0),
                dtype: NativeTensorDataType::F32,
                source_tensor_type: None,
                source_quantized: false,
                quantization: None,
                quantized_source: None,
                shape: vec![1],
                file: PathBuf::from("model.safetensors"),
                offset_bytes: 0,
                length_bytes: 4,
            }
        }
        let specs = vec![
            make_spec("attn_norm", NativeTensorRole::AttentionNorm),
            make_spec("conv1d", NativeTensorRole::LinearAttentionConv1d),
        ];

        apply_hf_sanitize_transforms(&specs, &mut name_map, false);

        let norm_out = name_map.get("attn_norm").expect("attn_norm present");
        for (got, want) in norm_out
            .data_f32()
            .iter()
            .zip([0.9_f32, 1.2, 1.05, 1.0].iter())
        {
            assert!((got - want).abs() < 1e-5, "norm: got {got}, want {want}");
        }

        let conv_out = name_map.get("conv1d").expect("conv1d present");
        assert_eq!(
            conv_out.shape(),
            vec![OUT as i32, KERNEL as i32, IN as i32],
            "HfNormOnly must NOT swap conv1d axes — they are already in MLX layout"
        );
        for (i, (got, want)) in conv_out
            .data_f32()
            .iter()
            .zip(conv_mlx_data.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-5,
                "conv1d[{i}]: got {got}, want {want}"
            );
        }
    }

    /// Build a minimal `name_map` + spec list mimicking the layer-0 slice of
    /// a hybrid (linear-attention) checkpoint, for auto-detection tests.
    fn fixture_layer0_linear_attention(
        norm_data: &[f32],
        conv1d_shape: &[i32],
    ) -> (Vec<NativeTensorSpec>, HashMap<String, MlxArray>) {
        let norm = MlxArray::from_raw_data(
            norm_data.as_ptr().cast(),
            std::mem::size_of_val(norm_data),
            &[norm_data.len() as i32],
            MlxDtype::Float32,
        );
        let conv_elements: i32 = conv1d_shape.iter().product();
        let conv_data = vec![0.0_f32; conv_elements as usize];
        let conv1d = MlxArray::from_raw_data(
            conv_data.as_ptr().cast(),
            std::mem::size_of_val(conv_data.as_slice()),
            conv1d_shape,
            MlxDtype::Float32,
        );
        let mut name_map = HashMap::new();
        name_map.insert("layers.0.linear_attn.norm".to_string(), norm);
        name_map.insert("layers.0.linear_attn.conv1d".to_string(), conv1d);

        let make_spec = |name: &str, role: NativeTensorRole| NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index: Some(0),
            dtype: NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![1],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 4,
        };
        let specs = vec![
            make_spec(
                "layers.0.linear_attn.norm",
                NativeTensorRole::LinearAttentionNorm,
            ),
            make_spec(
                "layers.0.linear_attn.conv1d",
                NativeTensorRole::LinearAttentionConv1d,
            ),
        ];
        (specs, name_map)
    }

    #[test]
    fn auto_detect_picks_hf_norm_only_for_unsanitized_norm_with_mlx_conv1d() {
        // Layer-0 fingerprint of mlx-community/Qwen3-Coder-Next-4bit: norm
        // weights stored as zero-centred deltas (mean_abs ≈ 0.011), conv1d
        // already in MLX layout (last dim = 1).
        let norm_data: Vec<f32> = (0..256).map(|i| 0.01 * ((i as f32).sin())).collect();
        let (specs, name_map) = fixture_layer0_linear_attention(&norm_data, &[64, 4, 1]);

        let chosen = auto_detect_linear_attention_sanitize(&specs, &name_map);

        assert_eq!(
            chosen,
            WeightSanitize::HfNormOnly,
            "unsanitized norm + MLX-layout conv1d ⇒ HfNormOnly"
        );
    }

    #[test]
    fn auto_detect_picks_hf_to_mlx_when_both_norm_and_conv1d_are_raw_hf() {
        // Raw HF safetensors path: norm is zero-centred deltas AND conv1d
        // is in HF layout `[conv_dim, in=1, kernel]`.
        let norm_data: Vec<f32> = (0..256).map(|i| 0.01 * ((i as f32).cos())).collect();
        let (specs, name_map) = fixture_layer0_linear_attention(&norm_data, &[64, 1, 4]);

        let chosen = auto_detect_linear_attention_sanitize(&specs, &name_map);

        assert_eq!(
            chosen,
            WeightSanitize::HfToMlx,
            "raw HF norm + HF conv1d ⇒ HfToMlx"
        );
    }

    #[test]
    fn auto_detect_returns_none_when_weights_already_sanitized() {
        // Pre-sanitized norm clusters near 1.0; conv1d in MLX layout.
        let norm_data = vec![1.0_f32; 256];
        let (specs, name_map) = fixture_layer0_linear_attention(&norm_data, &[64, 4, 1]);

        let chosen = auto_detect_linear_attention_sanitize(&specs, &name_map);

        assert_eq!(chosen, WeightSanitize::None);
    }

    #[test]
    fn auto_detect_returns_none_for_non_hybrid_models() {
        // No LinearAttentionNorm in the spec list: nothing to sample, fall
        // through to manifest setting unchanged.
        let specs: Vec<NativeTensorSpec> = Vec::new();
        let name_map: HashMap<String, MlxArray> = HashMap::new();

        let chosen = auto_detect_linear_attention_sanitize(&specs, &name_map);

        assert_eq!(chosen, WeightSanitize::None);
    }

    #[test]
    fn auto_detect_returns_none_for_sanitized_norm_with_hf_conv1d() {
        // Partially-transformed checkpoint (norm OK, conv1d not). Don't
        // silently re-sanitize — let `ensure_conv1d_mlx_layout` fire with
        // its specific diagnostic so the user sees the actual inconsistency.
        let norm_data = vec![1.0_f32; 256];
        let (specs, name_map) = fixture_layer0_linear_attention(&norm_data, &[64, 1, 4]);

        let chosen = auto_detect_linear_attention_sanitize(&specs, &name_map);

        assert_eq!(chosen, WeightSanitize::None);
    }

    #[test]
    fn apply_hf_sanitize_transforms_skips_non_norm_non_conv1d_roles() {
        // The sanitizer must leave projection weights, embeddings, and
        // other non-norm tensors untouched. Otherwise it would corrupt
        // the layout of every weight matrix in the model.
        let data = [3.0_f32, 4.0, 5.0, 6.0];
        let proj = MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(&data),
            &[data.len() as i32],
            MlxDtype::Float32,
        );
        let mut name_map: HashMap<String, MlxArray> = HashMap::new();
        name_map.insert("q_proj".to_string(), proj);

        let specs = vec![NativeTensorSpec {
            name: "q_proj".to_string(),
            role: NativeTensorRole::AttentionQ,
            layer_index: Some(0),
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

        apply_hf_sanitize_transforms(&specs, &mut name_map, true);

        let preserved = name_map.get("q_proj").expect("q_proj tensor still present");
        let values = preserved.data_f32();
        for (got, want) in values.iter().zip([3.0_f32, 4.0, 5.0, 6.0].iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "q_proj must be untouched: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn apply_hf_sanitize_transforms_preserves_norm_dtype() {
        // Raw HF norm weights are typically bf16. MLX's `add(bf16, f32)` would
        // promote the result to f32 without preservation, silently doubling
        // the stored norm-weight footprint. The sanitizer must cast back to
        // the original dtype so callers see a bf16 weight, matching what
        // mlx-community pre-sanitized weights look like.
        let delta_f32 = [-0.1_f32, 0.2, 0.05, 0.0];
        let mut delta_bf16_bytes = Vec::with_capacity(delta_f32.len() * 2);
        for v in &delta_f32 {
            // Round-to-nearest cast f32 -> bf16 by chopping the low 16 bits
            // of the f32 representation (sufficient for this small test).
            let bits = v.to_bits();
            delta_bf16_bytes.extend_from_slice(&(bits >> 16).to_le_bytes()[..2]);
        }
        let norm_bf16 = MlxArray::from_raw_data(
            delta_bf16_bytes.as_ptr(),
            delta_bf16_bytes.len(),
            &[delta_f32.len() as i32],
            MlxDtype::Bfloat16,
        );
        assert_eq!(norm_bf16.dtype(), MlxDtype::Bfloat16);

        let mut name_map: HashMap<String, MlxArray> = HashMap::new();
        name_map.insert("layers.0.attn_norm".to_string(), norm_bf16);

        let specs = vec![NativeTensorSpec {
            name: "layers.0.attn_norm".to_string(),
            role: NativeTensorRole::AttentionNorm,
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
        }];

        apply_hf_sanitize_transforms(&specs, &mut name_map, true);

        let sanitized = name_map
            .get("layers.0.attn_norm")
            .expect("norm tensor present");
        assert_eq!(
            sanitized.dtype(),
            MlxDtype::Bfloat16,
            "sanitize must preserve bf16 dtype, not silently upcast to f32"
        );
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
    fn linear_attention_qkvz_pack_order_interleaves_by_key_head() {
        let rows =
            linear_attention_qkvz_row_sources(2, 2, 4, 3).expect("valid linear attention dims");

        assert_eq!(
            rows,
            vec![
                LinearAttentionProjectionRowSource::Qkv(0),
                LinearAttentionProjectionRowSource::Qkv(1),
                LinearAttentionProjectionRowSource::Qkv(4),
                LinearAttentionProjectionRowSource::Qkv(5),
                LinearAttentionProjectionRowSource::Qkv(8),
                LinearAttentionProjectionRowSource::Qkv(9),
                LinearAttentionProjectionRowSource::Qkv(10),
                LinearAttentionProjectionRowSource::Qkv(11),
                LinearAttentionProjectionRowSource::Qkv(12),
                LinearAttentionProjectionRowSource::Qkv(13),
                LinearAttentionProjectionRowSource::Z(0),
                LinearAttentionProjectionRowSource::Z(1),
                LinearAttentionProjectionRowSource::Z(2),
                LinearAttentionProjectionRowSource::Z(3),
                LinearAttentionProjectionRowSource::Z(4),
                LinearAttentionProjectionRowSource::Z(5),
                LinearAttentionProjectionRowSource::Qkv(2),
                LinearAttentionProjectionRowSource::Qkv(3),
                LinearAttentionProjectionRowSource::Qkv(6),
                LinearAttentionProjectionRowSource::Qkv(7),
                LinearAttentionProjectionRowSource::Qkv(14),
                LinearAttentionProjectionRowSource::Qkv(15),
                LinearAttentionProjectionRowSource::Qkv(16),
                LinearAttentionProjectionRowSource::Qkv(17),
                LinearAttentionProjectionRowSource::Qkv(18),
                LinearAttentionProjectionRowSource::Qkv(19),
                LinearAttentionProjectionRowSource::Z(6),
                LinearAttentionProjectionRowSource::Z(7),
                LinearAttentionProjectionRowSource::Z(8),
                LinearAttentionProjectionRowSource::Z(9),
                LinearAttentionProjectionRowSource::Z(10),
                LinearAttentionProjectionRowSource::Z(11),
            ]
        );
    }

    #[test]
    fn linear_attention_ba_pack_order_is_b_then_a_per_key_head() {
        let rows = linear_attention_ba_row_sources(2, 4).expect("valid linear attention dims");

        assert_eq!(
            rows,
            vec![
                LinearAttentionProjectionRowSource::B(0),
                LinearAttentionProjectionRowSource::B(1),
                LinearAttentionProjectionRowSource::A(0),
                LinearAttentionProjectionRowSource::A(1),
                LinearAttentionProjectionRowSource::B(2),
                LinearAttentionProjectionRowSource::B(3),
                LinearAttentionProjectionRowSource::A(2),
                LinearAttentionProjectionRowSource::A(3),
            ]
        );
    }

    #[test]
    fn linear_attention_pack_order_rejects_uneven_value_heads() {
        let qkvz_message =
            invalid_layer_message(linear_attention_qkvz_row_sources(3, 2, 4, 3).map(|_| {
                QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None)
            }));
        let ba_message =
            invalid_layer_message(linear_attention_ba_row_sources(3, 4).map(|_| {
                QuantizedWeight::new(zeros(&[1, 1], MlxDtype::Float32, None), None, None)
            }));

        assert!(qkvz_message.contains("value heads divisible by key heads"));
        assert!(ba_message.contains("value heads divisible by key heads"));
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
