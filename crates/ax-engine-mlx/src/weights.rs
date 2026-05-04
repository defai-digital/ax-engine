use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;

use mlx_sys::{MlxArray, load_safetensors};

use ax_engine_core::{
    NativeModelArtifacts, NativeTensorQuantization, NativeTensorRole, NativeTensorSpec,
};

/// All weight arrays for one model.
pub struct ModelWeights {
    pub token_embedding: QuantizedWeight,
    pub final_norm: MlxArray,
    pub lm_head: QuantizedWeight,
    pub layers: Vec<LayerWeights>,
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
    // Dense FFN norms and weights.
    pub ffn_norm: MlxArray,
    pub ffn_post_norm: Option<MlxArray>,
    pub gate_proj: Option<QuantizedWeight>,
    pub up_proj: Option<QuantizedWeight>,
    pub gate_up_packed: Option<QuantizedWeight>,
    pub down_proj: QuantizedWeight,
    // MoE: extra norms (present when this layer has a MoE block).
    pub ffn_norm2: Option<MlxArray>,
    pub ffn_post_norm1: Option<MlxArray>,
    pub ffn_post_norm2: Option<MlxArray>,
    // MoE: router weights.
    pub router_proj: Option<QuantizedWeight>,
    pub router_scale: Option<MlxArray>,
    // MoE: expert weights (shape [num_experts, expert_size, hidden] / packed).
    pub gate_up_exps_packed: Option<QuantizedWeight>,
    pub gate_exps: Option<QuantizedWeight>,
    pub up_exps: Option<QuantizedWeight>,
    pub down_exps: Option<QuantizedWeight>,
}

/// Weights for a Qwen3.5 GatedDelta linear-attention layer.
pub struct LinearAttentionWeights {
    pub in_proj_qkv: QuantizedWeight,
    pub in_proj_z: QuantizedWeight,
    pub in_proj_a: QuantizedWeight,
    pub in_proj_b: QuantizedWeight,
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
        let down_proj = take_weight(
            specs,
            &mut name_map,
            NativeTensorRole::FfnDown,
            idx,
            "down_proj",
        )?;

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

        let (qkv_packed, q_proj, k_proj, v_proj) = if attention_layout == AttentionLayout::Linear {
            (None, None, None, None)
        } else {
            match full_attention_projection_layout(specs, idx, uses_shared_kv, uses_value_from_key)?
            {
                FullAttentionProjectionLayout::QOnly => {
                    let q = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQ,
                        idx,
                        "q_proj",
                    )?;
                    (None, Some(q), None, None)
                }
                FullAttentionProjectionLayout::PackedQkv => {
                    let p = take_weight(
                        specs,
                        &mut name_map,
                        NativeTensorRole::AttentionQkvPacked,
                        idx,
                        "qkv",
                    )?;
                    (Some(p), None, None, None)
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
                    (None, Some(q), Some(k), None)
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
                    (None, Some(q), Some(k), Some(v))
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
            } else {
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
            router_scale,
            gate_up_exps_packed,
            gate_exps,
            up_exps,
            down_exps,
        });
    }

    Ok(ModelWeights {
        token_embedding,
        final_norm,
        lm_head,
        layers,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionLayout {
    Full,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FullAttentionProjectionLayout {
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
    ]
    .into_iter()
    .any(|role| has_role(specs, role, layer_index))
}

fn has_linear_attention_role(specs: &[NativeTensorSpec], layer_index: Option<u32>) -> bool {
    [
        NativeTensorRole::LinearAttentionInProjQkv,
        NativeTensorRole::LinearAttentionInProjZ,
        NativeTensorRole::LinearAttentionInProjA,
        NativeTensorRole::LinearAttentionInProjB,
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
        in_proj_qkv: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjQkv,
            layer_index,
            "linear_attention_in_proj_qkv",
        )?,
        in_proj_z: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjZ,
            layer_index,
            "linear_attention_in_proj_z",
        )?,
        in_proj_a: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjA,
            layer_index,
            "linear_attention_in_proj_a",
        )?,
        in_proj_b: take_weight(
            specs,
            name_map,
            NativeTensorRole::LinearAttentionInProjB,
            layer_index,
            "linear_attention_in_proj_b",
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
    fn full_attention_projection_layout_rejects_packed_qkv_for_kv_shared_layers() {
        let specs = vec![spec(NativeTensorRole::AttentionQkvPacked)];

        let error = full_attention_projection_layout(&specs, Some(0), true, false)
            .expect_err("packed QKV cannot represent Q-only KV sharing");

        assert!(matches!(error, WeightLoadError::InvalidLayer(_)));
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
