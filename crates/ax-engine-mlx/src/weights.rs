use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;

use mlx_sys::{MlxArray, load_safetensors};

use ax_engine_core::{NativeModelArtifacts, NativeTensorRole, NativeTensorSpec};

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
    pub q_norm: Option<MlxArray>,
    pub k_norm: Option<MlxArray>,
    // Split Q/K/V projections.
    pub q_proj: Option<QuantizedWeight>,
    pub k_proj: Option<QuantizedWeight>,
    pub v_proj: Option<QuantizedWeight>,
    // Packed QKV projection (some architectures).
    pub qkv_packed: Option<QuantizedWeight>,
    pub o_proj: QuantizedWeight,
    pub ffn_norm: MlxArray,
    pub gate_proj: Option<QuantizedWeight>,
    pub up_proj: Option<QuantizedWeight>,
    pub gate_up_packed: Option<QuantizedWeight>,
    pub down_proj: QuantizedWeight,
}

/// A weight matrix plus optional MLX affine quantization metadata.
///
/// When `scales` is `Some`, the weight tensor contains packed 4-bit integers
/// and must be multiplied via `mlx_quantized_matmul` rather than regular
/// matmul.
pub struct QuantizedWeight {
    pub weight: MlxArray,
    pub scales: Option<MlxArray>,
    pub biases: Option<MlxArray>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedWeight {
    pub fn new(weight: MlxArray, scales: Option<MlxArray>, biases: Option<MlxArray>) -> Self {
        Self { weight, scales, biases, group_size: 64, bits: 4 }
    }
    pub fn is_quantized(&self) -> bool { self.scales.is_some() }
}

pub fn load_weights(artifacts: &NativeModelArtifacts) -> Result<ModelWeights, WeightLoadError> {
    let root = artifacts.root_dir().to_path_buf();
    let mut file_cache: HashMap<PathBuf, HashMap<String, MlxArray>> = HashMap::new();
    for spec in artifacts.tensor_specs() {
        let full = root.join(&spec.file);
        if let Entry::Vacant(entry) = file_cache.entry(full) {
            let path = entry.key().clone();
            let tensors = load_safetensors(&path, None)
                .map_err(WeightLoadError::FileMissing)?;
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

    let token_embedding = take_weight(specs, &mut name_map, NativeTensorRole::TokenEmbedding, None, "token_embedding")?;
    let final_norm = take_weight(specs, &mut name_map, NativeTensorRole::FinalNorm, None, "final_norm")?
        .weight;
    let lm_head = if artifacts.manifest().tie_word_embeddings {
        QuantizedWeight::new(
            token_embedding.weight.clone(),
            token_embedding.scales.clone(),
            token_embedding.biases.clone(),
        )
    } else {
        take_weight(specs, &mut name_map, NativeTensorRole::LmHead, None, "lm_head")?
    };

    let mut layers = Vec::with_capacity(layer_count);
    for li in 0..layer_count {
        let idx = Some(li as u32);

        let attn_norm = take_weight(specs, &mut name_map, NativeTensorRole::AttentionNorm, idx, "attn_norm")?.weight;
        let o_proj = take_weight(specs, &mut name_map, NativeTensorRole::AttentionO, idx, "o_proj")?;
        let ffn_norm = take_weight(specs, &mut name_map, NativeTensorRole::FfnNorm, idx, "ffn_norm")?.weight;
        let down_proj = take_weight(specs, &mut name_map, NativeTensorRole::FfnDown, idx, "down_proj")?;

        let q_norm = try_take_plain(specs, &mut name_map, NativeTensorRole::AttentionQNorm, idx)?;
        let k_norm = try_take_plain(specs, &mut name_map, NativeTensorRole::AttentionKNorm, idx)?;

        let (qkv_packed, q_proj, k_proj, v_proj) =
            if has_role(specs, NativeTensorRole::AttentionQkvPacked, idx) {
                let p = take_weight(specs, &mut name_map, NativeTensorRole::AttentionQkvPacked, idx, "qkv")?;
                (Some(p), None, None, None)
            } else {
                let q = take_weight(specs, &mut name_map, NativeTensorRole::AttentionQ, idx, "q_proj")?;
                let k = take_weight(specs, &mut name_map, NativeTensorRole::AttentionK, idx, "k_proj")?;
                let v = take_weight(specs, &mut name_map, NativeTensorRole::AttentionV, idx, "v_proj")?;
                (None, Some(q), Some(k), Some(v))
            };

        let (gate_up_packed, gate_proj, up_proj) =
            if has_role(specs, NativeTensorRole::FfnGateUpPacked, idx) {
                let p = take_weight(specs, &mut name_map, NativeTensorRole::FfnGateUpPacked, idx, "gate_up")?;
                (Some(p), None, None)
            } else {
                let g = take_weight(specs, &mut name_map, NativeTensorRole::FfnGate, idx, "gate_proj")?;
                let u = take_weight(specs, &mut name_map, NativeTensorRole::FfnUp, idx, "up_proj")?;
                (None, Some(g), Some(u))
            };

        layers.push(LayerWeights {
            attn_norm, q_norm, k_norm,
            q_proj, k_proj, v_proj, qkv_packed,
            o_proj, ffn_norm,
            gate_proj, up_proj, gate_up_packed, down_proj,
        });
    }

    Ok(ModelWeights { token_embedding, final_norm, lm_head, layers })
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
    let name = specs
        .iter()
        .find(|s| s.role == role && s.layer_index == layer_index)
        .map(|s| s.name.clone())
        .ok_or_else(|| WeightLoadError::RoleMissing(format!("{label}[{layer_index:?}]")))?;

    let weight = name_map
        .remove(&name)
        .ok_or_else(|| WeightLoadError::TensorMissing(name.clone()))?;

    // Look for co-located `.scales` and `.biases` (MLX quantized format).
    let base = name.strip_suffix(".weight").unwrap_or(&name);
    let scales = name_map.remove(&format!("{base}.scales"));
    let biases = name_map.remove(&format!("{base}.biases"));

    Ok(QuantizedWeight::new(weight, scales, biases))
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
    else { return Ok(None); };
    Ok(name_map.remove(&name))
}

fn has_role(specs: &[NativeTensorSpec], role: NativeTensorRole, layer_index: Option<u32>) -> bool {
    specs.iter().any(|s| s.role == role && s.layer_index == layer_index)
}

#[derive(Debug, thiserror::Error)]
pub enum WeightLoadError {
    #[error("weight file not found or empty: {0}")]
    FileMissing(String),
    #[error("tensor not found: {0}")]
    TensorMissing(String),
    #[error("required tensor role missing: {0}")]
    RoleMissing(String),
}
