//! Convert HuggingFace / MLX model directories to ax-engine native model manifests.
//!
//! Reads `config.json` and safetensors headers from a model directory and produces
//! a `NativeModelManifest` that can be written as `model-manifest.json`. No tensor
//! data is copied or converted — the manifest points directly at the original
//! safetensors files.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::model::{
    NativeModelManifest, NativeTensorDataType, NativeTensorFormat, NativeTensorRole,
    NativeTensorSpec, AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Errors returned by the conversion process.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("failed to read {path}: {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    ParseJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported model type {model_type}; supported: qwen3, qwen3_5, qwen3_next, gemma4")]
    UnsupportedModelType { model_type: String },
    #[error("missing config field: {field}")]
    MissingConfigField { field: &'static str },
    #[error("no safetensors files found in {dir}")]
    NoSafetensors { dir: PathBuf },
    #[error("failed to parse safetensors header in {path}: {message}")]
    InvalidSafetensorsHeader { path: PathBuf, message: String },
    #[error("unsupported tensor dtype {dtype} for tensor {name}")]
    UnsupportedDtype { name: String, dtype: String },
}

/// Convert a HuggingFace / MLX model directory into a `NativeModelManifest`.
///
/// The directory must contain `config.json` and one or more `model*.safetensors`
/// files. The returned manifest references the safetensors files by relative path,
/// so it can be written to the same directory as `model-manifest.json`.
pub fn convert_hf_model_dir(model_dir: &Path) -> Result<NativeModelManifest, ConvertError> {
    let config = load_hf_config(model_dir)?;
    let model_type = resolve_model_type(&config)?;
    let family = model_family_for_type(&model_type)?;
    let arch = resolve_architecture(&config, &model_type)?;
    let safetensors_files = find_safetensors_files(model_dir)?;
    let all_tensors = parse_all_safetensors_headers(model_dir, &safetensors_files)?;
    let mapped_tensors = map_tensors(&all_tensors, &family)?;

    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let rope_theta = arch_f64(&config, &model_type, "rope_theta")
        .or_else(|| {
            // Qwen3.5+ nests rope_theta inside rope_parameters
            let text_config = config.get("text_config")?;
            text_config
                .get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
                .and_then(|v| v.as_f64())
        })
        .map(|v| v as u32);

    let query_pre_attn_scalar =
        arch_f64(&config, &model_type, "query_pre_attn_scalar").map(|v| v as u32);

    let attention_logit_softcap =
        arch_f64(&config, &model_type, "attn_logit_softcapping").map(|v| v as u32);

    Ok(NativeModelManifest {
        schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: family.family_name.to_string(),
        tensor_format: NativeTensorFormat::Safetensors,
        layer_count: arch.layer_count,
        hidden_size: arch.hidden_size,
        attention_head_count: arch.attention_head_count,
        attention_head_dim: arch.attention_head_dim,
        kv_head_count: arch.kv_head_count,
        vocab_size: arch.vocab_size,
        tie_word_embeddings,
        rope_theta,
        query_pre_attn_scalar,
        attention_logit_softcap,
        attention_value_from_key_layers: Vec::new(),
        attention_v_norm_no_scale_layers: Vec::new(),
        tensors: mapped_tensors,
    })
}

/// Write a `model-manifest.json` file in the given directory.
pub fn write_manifest(
    model_dir: &Path,
    manifest: &NativeModelManifest,
) -> Result<(), ConvertError> {
    let manifest_path = model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    let json = serde_json::to_vec_pretty(manifest).map_err(|source| ConvertError::ParseJson {
        path: manifest_path.clone(),
        source,
    })?;
    fs::write(&manifest_path, json).map_err(|source| ConvertError::ReadFile {
        path: manifest_path,
        source,
    })
}

// ---------------------------------------------------------------------------
// Model family definitions
// ---------------------------------------------------------------------------

struct ModelFamily {
    family_name: &'static str,
    tensor_map: &'static [(&'static str, TensorMapping)],
    uses_language_model_prefix: bool,
}

#[derive(Clone, Copy)]
enum TensorMapping {
    Global(NativeTensorRole),
    PerLayer(NativeTensorRole),
}

/// HuggingFace tensor name patterns shared by Qwen3/Gemma4.
///
/// The HuggingFace convention is:
///   model.embed_tokens.weight
///   model.layers.{i}.self_attn.q_proj.weight
///   model.layers.{i}.self_attn.k_proj.weight
///   model.layers.{i}.self_attn.v_proj.weight
///   model.layers.{i}.self_attn.o_proj.weight
///   model.layers.{i}.self_attn.q_norm.weight        (Qwen3, Gemma4)
///   model.layers.{i}.self_attn.k_norm.weight        (Qwen3, Gemma4)
///   model.layers.{i}.input_layernorm.weight
///   model.layers.{i}.post_attention_layernorm.weight
///   model.layers.{i}.mlp.gate_proj.weight
///   model.layers.{i}.mlp.up_proj.weight
///   model.layers.{i}.mlp.down_proj.weight
///   model.norm.weight
///   lm_head.weight
///
/// MLX sanitises the `model.` prefix differently per family, but the
/// safetensors on disk use the HuggingFace names above.
const HF_STANDARD_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "model.embed_tokens.weight",
        TensorMapping::Global(NativeTensorRole::TokenEmbedding),
    ),
    (
        "model.norm.weight",
        TensorMapping::Global(NativeTensorRole::FinalNorm),
    ),
    (
        "lm_head.weight",
        TensorMapping::Global(NativeTensorRole::LmHead),
    ),
    // per-layer attention
    (
        "self_attn.q_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQ),
    ),
    (
        "self_attn.k_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionK),
    ),
    (
        "self_attn.v_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionV),
    ),
    (
        "self_attn.o_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionO),
    ),
    (
        "self_attn.q_norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQNorm),
    ),
    (
        "self_attn.k_norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKNorm),
    ),
    // per-layer attention (packed QKV variant)
    (
        "self_attn.qkv_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQkvPacked),
    ),
    // per-layer norms
    (
        "input_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionNorm),
    ),
    (
        "post_attention_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnNorm),
    ),
    // per-layer FFN
    (
        "mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGate),
    ),
    (
        "mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUp),
    ),
    (
        "mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDown),
    ),
    // packed gate+up
    (
        "mlp.gate_up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateUpPacked),
    ),
];

/// Gemma4 and Qwen3.5+ wrap the text model under `language_model.model.`, so
/// tensor names in safetensors appear as
/// `language_model.model.layers.0.self_attn.q_proj.weight`.
/// We also accept `model.layers.…` for already-sanitised weights.
const LANGUAGE_MODEL_PREFIX_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "language_model.model.embed_tokens.weight",
        TensorMapping::Global(NativeTensorRole::TokenEmbedding),
    ),
    (
        "language_model.model.norm.weight",
        TensorMapping::Global(NativeTensorRole::FinalNorm),
    ),
    (
        "language_model.lm_head.weight",
        TensorMapping::Global(NativeTensorRole::LmHead),
    ),
];

fn model_family_for_type(model_type: &str) -> Result<ModelFamily, ConvertError> {
    match model_type {
        "qwen3" => Ok(ModelFamily {
            family_name: "qwen3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            uses_language_model_prefix: false,
        }),
        "qwen3_5" | "qwen3.5" | "qwen3_5_moe" | "qwen3_5_text" => Ok(ModelFamily {
            family_name: "qwen3_5",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            uses_language_model_prefix: true,
        }),
        "qwen3_next" | "qwen3.6" | "qwen3_6" => Ok(ModelFamily {
            family_name: "qwen3_next",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            uses_language_model_prefix: true,
        }),
        "gemma4" => Ok(ModelFamily {
            family_name: "gemma4",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            uses_language_model_prefix: true,
        }),
        other => Err(ConvertError::UnsupportedModelType {
            model_type: other.to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

fn load_hf_config(model_dir: &Path) -> Result<serde_json::Value, ConvertError> {
    let config_path = model_dir.join("config.json");
    let bytes = fs::read(&config_path).map_err(|source| ConvertError::ReadFile {
        path: config_path.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| ConvertError::ParseJson {
        path: config_path,
        source,
    })
}

fn resolve_model_type(config: &serde_json::Value) -> Result<String, ConvertError> {
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or(ConvertError::MissingConfigField {
            field: "model_type",
        })
}

struct ArchitectureParams {
    layer_count: u32,
    hidden_size: u32,
    attention_head_count: u32,
    attention_head_dim: u32,
    kv_head_count: u32,
    vocab_size: u32,
}

/// Whether this model type nests architecture params under `text_config`.
fn uses_text_config(model_type: &str) -> bool {
    matches!(
        model_type,
        "gemma4" | "qwen3_5" | "qwen3_next" | "qwen3_6" | "qwen3.5" | "qwen3.6"
    )
}

/// Get a u64 field, checking both the top-level config and a nested `text_config`
/// (Gemma4 and Qwen3.5+ nest architecture params under `text_config`).
fn arch_u64(config: &serde_json::Value, model_type: &str, field: &str) -> Option<u64> {
    config.get(field).and_then(|v| v.as_u64()).or_else(|| {
        if uses_text_config(model_type) {
            config
                .get("text_config")
                .and_then(|tc| tc.get(field))
                .and_then(|v| v.as_u64())
        } else {
            None
        }
    })
}

fn arch_f64(config: &serde_json::Value, model_type: &str, field: &str) -> Option<f64> {
    config.get(field).and_then(|v| v.as_f64()).or_else(|| {
        if uses_text_config(model_type) {
            config
                .get("text_config")
                .and_then(|tc| tc.get(field))
                .and_then(|v| v.as_f64())
        } else {
            None
        }
    })
}

fn require_arch_u64(
    config: &serde_json::Value,
    model_type: &str,
    field: &'static str,
) -> Result<u64, ConvertError> {
    arch_u64(config, model_type, field).ok_or(ConvertError::MissingConfigField { field })
}

fn resolve_architecture(
    config: &serde_json::Value,
    model_type: &str,
) -> Result<ArchitectureParams, ConvertError> {
    let hidden_size = require_arch_u64(config, model_type, "hidden_size")? as u32;
    let attention_head_count = require_arch_u64(config, model_type, "num_attention_heads")? as u32;
    let kv_head_count = arch_u64(config, model_type, "num_key_value_heads")
        .map(|v| v as u32)
        .unwrap_or(attention_head_count);
    let attention_head_dim = arch_u64(config, model_type, "head_dim")
        .map(|v| v as u32)
        .unwrap_or_else(|| {
            if attention_head_count > 0 {
                hidden_size / attention_head_count
            } else {
                0
            }
        });
    let layer_count = require_arch_u64(config, model_type, "num_hidden_layers")? as u32;
    let vocab_size = require_arch_u64(config, model_type, "vocab_size")? as u32;

    Ok(ArchitectureParams {
        layer_count,
        hidden_size,
        attention_head_count,
        attention_head_dim,
        kv_head_count,
        vocab_size,
    })
}

// ---------------------------------------------------------------------------
// Safetensors header parsing
// ---------------------------------------------------------------------------

/// Parsed tensor info from a safetensors file header.
struct SafetensorEntry {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    file: PathBuf,
    offset_bytes: u64,
    length_bytes: u64,
}

#[derive(Deserialize)]
struct SafetensorHeaderEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn find_safetensors_files(model_dir: &Path) -> Result<Vec<PathBuf>, ConvertError> {
    let mut files: Vec<PathBuf> = fs::read_dir(model_dir)
        .map_err(|source| ConvertError::ReadFile {
            path: model_dir.to_path_buf(),
            source,
        })?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if files.is_empty() {
        return Err(ConvertError::NoSafetensors {
            dir: model_dir.to_path_buf(),
        });
    }
    files.sort();
    Ok(files)
}

fn parse_safetensors_header(path: &Path) -> Result<Vec<SafetensorEntry>, ConvertError> {
    let mut file = fs::File::open(path).map_err(|source| ConvertError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;

    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)
        .map_err(|source| ConvertError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes)
        .map_err(|source| ConvertError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?;

    let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
        .map_err(|source| ConvertError::ParseJson {
            path: path.to_path_buf(),
            source,
        })?;

    let data_base_offset = 8 + header_size as u64;
    let file_name = path
        .file_name()
        .map(|n| PathBuf::from(n))
        .unwrap_or_else(|| path.to_path_buf());

    let mut entries = Vec::new();
    for (name, value) in &header {
        if name == "__metadata__" {
            continue;
        }
        let entry: SafetensorHeaderEntry = serde_json::from_value(value.clone()).map_err(|_| {
            ConvertError::InvalidSafetensorsHeader {
                path: path.to_path_buf(),
                message: format!("invalid tensor entry for {name}"),
            }
        })?;

        entries.push(SafetensorEntry {
            name: name.clone(),
            dtype: entry.dtype,
            shape: entry.shape,
            file: file_name.clone(),
            offset_bytes: data_base_offset + entry.data_offsets[0],
            length_bytes: entry.data_offsets[1] - entry.data_offsets[0],
        });
    }

    Ok(entries)
}

fn parse_all_safetensors_headers(
    _model_dir: &Path,
    files: &[PathBuf],
) -> Result<Vec<SafetensorEntry>, ConvertError> {
    let mut all = Vec::new();
    for file in files {
        all.extend(parse_safetensors_header(file)?);
    }
    Ok(all)
}

fn convert_dtype(dtype: &str, name: &str) -> Result<NativeTensorDataType, ConvertError> {
    match dtype {
        "F16" => Ok(NativeTensorDataType::F16),
        "BF16" => Ok(NativeTensorDataType::Bf16),
        "F32" => Ok(NativeTensorDataType::F32),
        "I8" => Ok(NativeTensorDataType::I8),
        "U8" => Ok(NativeTensorDataType::U8),
        other => Err(ConvertError::UnsupportedDtype {
            name: name.to_string(),
            dtype: other.to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tensor name → role mapping
// ---------------------------------------------------------------------------

/// Try to match a tensor name against the family's mapping table.
fn match_tensor(name: &str, family: &ModelFamily) -> Option<(NativeTensorRole, Option<u32>)> {
    // Try standard map (model.embed_tokens, model.layers.N.…)
    if let Some(result) = match_tensor_in_map(name, family.tensor_map) {
        return Some(result);
    }

    // Try language_model.model.… prefix (Gemma4, Qwen3.5, Qwen3.6)
    if family.uses_language_model_prefix {
        if let Some(result) = match_tensor_in_map(name, LANGUAGE_MODEL_PREFIX_TENSOR_MAP) {
            return Some(result);
        }
        if let Some(result) =
            match_prefixed_per_layer(name, "language_model.model.layers.", family.tensor_map)
        {
            return Some(result);
        }
    }

    None
}

fn match_tensor_in_map(
    name: &str,
    tensor_map: &[(&str, TensorMapping)],
) -> Option<(NativeTensorRole, Option<u32>)> {
    for (pattern, mapping) in tensor_map {
        match mapping {
            TensorMapping::Global(role) => {
                if name == *pattern {
                    return Some((*role, None));
                }
            }
            TensorMapping::PerLayer(role) => {
                // Match "model.layers.{N}.{pattern}"
                if let Some(layer_index) = match_per_layer_pattern(name, "model.layers.", pattern) {
                    return Some((*role, Some(layer_index)));
                }
            }
        }
    }
    None
}

fn match_prefixed_per_layer(
    name: &str,
    prefix: &str,
    tensor_map: &[(&str, TensorMapping)],
) -> Option<(NativeTensorRole, Option<u32>)> {
    for (pattern, mapping) in tensor_map {
        if let TensorMapping::PerLayer(role) = mapping {
            if let Some(layer_index) = match_per_layer_pattern(name, prefix, pattern) {
                return Some((*role, Some(layer_index)));
            }
        }
    }
    None
}

fn match_per_layer_pattern(name: &str, prefix: &str, suffix: &str) -> Option<u32> {
    let rest = name.strip_prefix(prefix)?;
    let dot = rest.find('.')?;
    let layer_index: u32 = rest[..dot].parse().ok()?;
    let after_layer = &rest[dot + 1..];
    if after_layer == suffix {
        Some(layer_index)
    } else {
        None
    }
}

fn map_tensors(
    all_tensors: &[SafetensorEntry],
    family: &ModelFamily,
) -> Result<Vec<NativeTensorSpec>, ConvertError> {
    let mut mapped = Vec::new();

    for entry in all_tensors {
        let Some((role, layer_index)) = match_tensor(&entry.name, family) else {
            // Skip unrecognised tensors (e.g. bias tensors, MoE experts, vision tower).
            continue;
        };

        let dtype = convert_dtype(&entry.dtype, &entry.name)?;

        mapped.push(NativeTensorSpec {
            name: entry.name.clone(),
            role,
            layer_index,
            dtype,
            shape: entry.shape.clone(),
            file: entry.file.clone(),
            offset_bytes: entry.offset_bytes,
            length_bytes: entry.length_bytes,
        });
    }

    // Sort by (layer_index, role ordinal) for deterministic output.
    mapped.sort_by_key(|spec| (spec.layer_index, format!("{:?}", spec.role)));
    Ok(mapped)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fake_safetensors(dir: &Path, filename: &str, tensors: &[(&str, &str, &[u64])]) {
        let mut header = BTreeMap::new();
        let mut offset = 0u64;
        for (name, dtype, shape) in tensors {
            let elem_size: u64 = match *dtype {
                "F16" | "BF16" => 2,
                "F32" => 4,
                _ => 1,
            };
            let num_elements: u64 = shape.iter().product();
            let byte_len = num_elements * elem_size;
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + byte_len],
                }),
            );
            offset += byte_len;
        }

        let header_json = serde_json::to_vec(&header).unwrap();
        let header_size = header_json.len() as u64;
        let data = vec![0u8; offset as usize];

        let path = dir.join(filename);
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&header_size.to_le_bytes()).unwrap();
        file.write_all(&header_json).unwrap();
        file.write_all(&data).unwrap();
    }

    fn write_config(dir: &Path, config: serde_json::Value) {
        let path = dir.join("config.json");
        fs::write(path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ax-convert-{label}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn converts_qwen3_f16_model_directory() {
        let dir = unique_test_dir("qwen3");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "qwen3",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 2,
                "vocab_size": 151936,
                "tie_word_embeddings": true,
                "rope_theta": 1000000,
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "F16", &[151936, 4096]),
                ("model.norm.weight", "F16", &[4096]),
                ("model.layers.0.input_layernorm.weight", "F16", &[4096]),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "F16",
                    &[4096, 4096],
                ),
                (
                    "model.layers.0.self_attn.k_proj.weight",
                    "F16",
                    &[1024, 4096],
                ),
                (
                    "model.layers.0.self_attn.v_proj.weight",
                    "F16",
                    &[1024, 4096],
                ),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    "F16",
                    &[4096, 4096],
                ),
                ("model.layers.0.self_attn.q_norm.weight", "F16", &[128]),
                ("model.layers.0.self_attn.k_norm.weight", "F16", &[128]),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "F16",
                    &[4096],
                ),
                ("model.layers.0.mlp.gate_proj.weight", "F16", &[12288, 4096]),
                ("model.layers.0.mlp.up_proj.weight", "F16", &[12288, 4096]),
                ("model.layers.0.mlp.down_proj.weight", "F16", &[4096, 12288]),
                ("model.layers.1.input_layernorm.weight", "F16", &[4096]),
                (
                    "model.layers.1.self_attn.q_proj.weight",
                    "F16",
                    &[4096, 4096],
                ),
                (
                    "model.layers.1.self_attn.k_proj.weight",
                    "F16",
                    &[1024, 4096],
                ),
                (
                    "model.layers.1.self_attn.v_proj.weight",
                    "F16",
                    &[1024, 4096],
                ),
                (
                    "model.layers.1.self_attn.o_proj.weight",
                    "F16",
                    &[4096, 4096],
                ),
                ("model.layers.1.self_attn.q_norm.weight", "F16", &[128]),
                ("model.layers.1.self_attn.k_norm.weight", "F16", &[128]),
                (
                    "model.layers.1.post_attention_layernorm.weight",
                    "F16",
                    &[4096],
                ),
                ("model.layers.1.mlp.gate_proj.weight", "F16", &[12288, 4096]),
                ("model.layers.1.mlp.up_proj.weight", "F16", &[12288, 4096]),
                ("model.layers.1.mlp.down_proj.weight", "F16", &[4096, 12288]),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("qwen3 conversion should succeed");

        assert_eq!(manifest.model_family, "qwen3");
        assert_eq!(manifest.layer_count, 2);
        assert_eq!(manifest.hidden_size, 4096);
        assert_eq!(manifest.attention_head_count, 32);
        assert_eq!(manifest.attention_head_dim, 128);
        assert_eq!(manifest.kv_head_count, 8);
        assert_eq!(manifest.vocab_size, 151936);
        assert!(manifest.tie_word_embeddings);
        assert_eq!(manifest.rope_theta, Some(1000000));

        let roles: Vec<_> = manifest.tensors.iter().map(|t| t.role).collect();
        assert!(roles.contains(&NativeTensorRole::TokenEmbedding));
        assert!(roles.contains(&NativeTensorRole::FinalNorm));
        assert!(roles.contains(&NativeTensorRole::AttentionQ));
        assert!(roles.contains(&NativeTensorRole::AttentionQNorm));
        assert!(roles.contains(&NativeTensorRole::AttentionKNorm));
        assert!(roles.contains(&NativeTensorRole::FfnGate));
        assert!(roles.contains(&NativeTensorRole::FfnUp));
        assert!(roles.contains(&NativeTensorRole::FfnDown));

        let layer0_q = manifest
            .tensors
            .iter()
            .find(|t| t.role == NativeTensorRole::AttentionQ && t.layer_index == Some(0))
            .expect("layer 0 q_proj should exist");
        assert_eq!(layer0_q.dtype, NativeTensorDataType::F16);
        assert_eq!(layer0_q.shape, vec![4096, 4096]);

        // Verify no lm_head when tie_word_embeddings is true
        // (the model doesn't have lm_head.weight in the safetensors)
        assert!(!roles.contains(&NativeTensorRole::LmHead));

        // Write and re-read
        write_manifest(&dir, &manifest).expect("write should succeed");
        let reloaded = crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("reloaded manifest should validate");
        assert_eq!(reloaded.manifest().layer_count, 2);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn converts_gemma4_with_text_config_nesting() {
        let dir = unique_test_dir("gemma4");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "gemma4",
                "vocab_size": 262144,
                "text_config": {
                    "hidden_size": 3072,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "num_hidden_layers": 2,
                    "vocab_size": 262144,
                    "rope_theta": 1000000,
                }
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "BF16", &[262144, 3072]),
                ("model.norm.weight", "BF16", &[3072]),
                ("lm_head.weight", "BF16", &[262144, 3072]),
                ("model.layers.0.input_layernorm.weight", "BF16", &[3072]),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "BF16",
                    &[4096, 3072],
                ),
                (
                    "model.layers.0.self_attn.k_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "model.layers.0.self_attn.v_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    "BF16",
                    &[3072, 4096],
                ),
                ("model.layers.0.self_attn.q_norm.weight", "BF16", &[128]),
                ("model.layers.0.self_attn.k_norm.weight", "BF16", &[128]),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "model.layers.0.mlp.gate_proj.weight",
                    "BF16",
                    &[12288, 3072],
                ),
                ("model.layers.0.mlp.up_proj.weight", "BF16", &[12288, 3072]),
                (
                    "model.layers.0.mlp.down_proj.weight",
                    "BF16",
                    &[3072, 12288],
                ),
                ("model.layers.1.input_layernorm.weight", "BF16", &[3072]),
                (
                    "model.layers.1.self_attn.q_proj.weight",
                    "BF16",
                    &[4096, 3072],
                ),
                (
                    "model.layers.1.self_attn.k_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "model.layers.1.self_attn.v_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "model.layers.1.self_attn.o_proj.weight",
                    "BF16",
                    &[3072, 4096],
                ),
                ("model.layers.1.self_attn.q_norm.weight", "BF16", &[128]),
                ("model.layers.1.self_attn.k_norm.weight", "BF16", &[128]),
                (
                    "model.layers.1.post_attention_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "model.layers.1.mlp.gate_proj.weight",
                    "BF16",
                    &[12288, 3072],
                ),
                ("model.layers.1.mlp.up_proj.weight", "BF16", &[12288, 3072]),
                (
                    "model.layers.1.mlp.down_proj.weight",
                    "BF16",
                    &[3072, 12288],
                ),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

        assert_eq!(manifest.model_family, "gemma4");
        assert_eq!(manifest.hidden_size, 3072);
        assert_eq!(manifest.attention_head_count, 32);
        assert_eq!(manifest.attention_head_dim, 128);
        assert_eq!(manifest.kv_head_count, 8);
        assert_eq!(manifest.vocab_size, 262144);
        assert_eq!(manifest.rope_theta, Some(1000000));

        let has_lm_head = manifest
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::LmHead);
        assert!(has_lm_head);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_unsupported_model_type() {
        let dir = unique_test_dir("unsupported");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "llama",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 2,
                "vocab_size": 32000,
            }),
        );
        write_fake_safetensors(&dir, "model.safetensors", &[]);

        let error = convert_hf_model_dir(&dir).expect_err("llama should be unsupported");
        assert!(matches!(error, ConvertError::UnsupportedModelType { .. }));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_quantized_dtypes() {
        let dir = unique_test_dir("quantized");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "qwen3",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 1,
                "vocab_size": 151936,
            }),
        );
        // "I32" is not a supported safetensors dtype in our converter
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[("model.embed_tokens.weight", "I32", &[151936, 4096])],
        );

        let error = convert_hf_model_dir(&dir).expect_err("I32 dtype should fail");
        assert!(matches!(error, ConvertError::UnsupportedDtype { .. }));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn skips_unrecognised_tensors() {
        let dir = unique_test_dir("extra-tensors");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "qwen3",
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 32,
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "F32", &[32, 8]),
                ("model.norm.weight", "F32", &[8]),
                ("model.layers.0.input_layernorm.weight", "F32", &[8]),
                ("model.layers.0.self_attn.q_proj.weight", "F32", &[8, 8]),
                ("model.layers.0.self_attn.k_proj.weight", "F32", &[8, 8]),
                ("model.layers.0.self_attn.v_proj.weight", "F32", &[8, 8]),
                ("model.layers.0.self_attn.o_proj.weight", "F32", &[8, 8]),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "F32",
                    &[8],
                ),
                ("model.layers.0.mlp.gate_proj.weight", "F32", &[16, 8]),
                ("model.layers.0.mlp.up_proj.weight", "F32", &[16, 8]),
                ("model.layers.0.mlp.down_proj.weight", "F32", &[8, 16]),
                // These should be silently skipped:
                ("model.layers.0.self_attn.rotary_emb.inv_freq", "F32", &[64]),
                ("some.unknown.tensor", "F32", &[100]),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("conversion should succeed");
        assert_eq!(manifest.tensors.len(), 11);

        let names: Vec<_> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(!names.contains(&"model.layers.0.self_attn.rotary_emb.inv_freq"));
        assert!(!names.contains(&"some.unknown.tensor"));

        let _ = fs::remove_dir_all(dir);
    }

    /// Real model integration test — requires `.internal/models/Qwen3.5-2B-bf16`.
    /// Run with: cargo test -p ax-engine-core -- convert::tests::converts_real_qwen3_5 --ignored
    #[test]
    #[ignore]
    fn converts_real_qwen3_5_bf16_model() {
        let model_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.internal/models/Qwen3.5-2B-bf16");
        if !model_dir.join("config.json").exists() {
            eprintln!("skipping: model not downloaded at {}", model_dir.display());
            return;
        }

        let manifest = convert_hf_model_dir(&model_dir)
            .expect("real Qwen3.5-2B-bf16 conversion should succeed");

        assert_eq!(manifest.model_family, "qwen3_5");
        assert_eq!(manifest.layer_count, 24);
        assert_eq!(manifest.hidden_size, 2048);
        assert_eq!(manifest.attention_head_count, 8);
        assert_eq!(manifest.attention_head_dim, 256);
        assert_eq!(manifest.kv_head_count, 2);
        assert_eq!(manifest.vocab_size, 248320);
        assert!(manifest.tie_word_embeddings);
        assert_eq!(manifest.rope_theta, Some(10000000));

        // Qwen3.5 has mixed layers: only full_attention layers (3,7,11,15,19,23)
        // have self_attn tensors. All 24 layers have FFN + norms.
        let attn_q_layers: Vec<u32> = manifest
            .tensors
            .iter()
            .filter(|t| t.role == NativeTensorRole::AttentionQ)
            .filter_map(|t| t.layer_index)
            .collect();
        assert_eq!(attn_q_layers, vec![3, 7, 11, 15, 19, 23]);

        let ffn_gate_count = manifest
            .tensors
            .iter()
            .filter(|t| t.role == NativeTensorRole::FfnGate)
            .count();
        assert_eq!(ffn_gate_count, 24, "all 24 layers should have FFN gate");

        let norm_count = manifest
            .tensors
            .iter()
            .filter(|t| t.role == NativeTensorRole::AttentionNorm)
            .count();
        assert_eq!(norm_count, 24, "all 24 layers should have attention norm");

        // All tensors should be BF16
        assert!(
            manifest
                .tensors
                .iter()
                .all(|t| t.dtype == NativeTensorDataType::Bf16
                    || t.dtype == NativeTensorDataType::F32)
        );

        // Write manifest, then validate the full NativeModelArtifacts pipeline
        write_manifest(&model_dir, &manifest).expect("write manifest should succeed");
        let artifacts = crate::model::NativeModelArtifacts::from_dir(&model_dir)
            .expect("NativeModelArtifacts should validate the real Qwen3.5 model");
        assert_eq!(artifacts.manifest().layer_count, 24);
        assert_eq!(
            artifacts.summary().tensor_count,
            manifest.tensors.len() as u32
        );

        eprintln!(
            "✓ converted {} tensors, {} layers, family={}",
            manifest.tensors.len(),
            manifest.layer_count,
            manifest.model_family
        );

        // Clean up the generated manifest
        let _ = fs::remove_file(model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE));
    }
}
