//! Convert HuggingFace / MLX model directories to ax-engine native model manifests.
//!
//! Reads `config.json` and safetensors headers from a model directory and produces
//! a `NativeModelManifest` that can be written as `model-manifest.json`. No tensor
//! data is copied or converted — the manifest points directly at the original
//! safetensors files.
//!
//! # OptiQ / mlx-lm mixed-precision quantization
//!
//! mlx-optiq (and stock mlx-lm) mixed-precision quants store a global default under
//! `quantization` or `quantization_config` (`bits`, `group_size`, `mode`) plus
//! nested **per-tensor overrides** keyed by module path without the `.weight`
//! suffix (for example `language_model.model.layers.0.mlp.gate_proj`). mlx-lm
//! passes each override dictionary directly to MLX, so omitted fields use the
//! override mode's defaults rather than inheriting the global settings. This
//! matters for OptiQ checkpoints that mix global MXFP4 with affine 8-bit
//! sensitive layers. Convert applies those overrides onto U32 weight tensors;
//! packing skips mixed-precision fusions when sibling projections disagree.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::model::{
    AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeDiffusionConfig, NativeGlmRouterConfig,
    NativeLinearAttentionConfig, NativeMlaAttentionConfig, NativeModelManifest, NativeMoeConfig,
    NativeRuntimeStatus, NativeTensorDataType, NativeTensorFormat, NativeTensorQuantization,
    NativeTensorRole, NativeTensorSpec, WeightSanitize,
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
    #[error(
        "unsupported model type {model_type}; supported: qwen3, qwen3_5, qwen3_next, gemma4, gemma4_unified, gemma4_assistant, diffusion_gemma, glm4_moe_lite, llama, llama3, mistral, mistral3, mixtral, deepseek_v3, llama4, gpt_oss, unlimited_ocr"
    )]
    UnsupportedModelType { model_type: String },
    #[error("missing config field: {field}")]
    MissingConfigField { field: &'static str },
    #[error("no safetensors files found in {dir}")]
    NoSafetensors { dir: PathBuf },
    #[error("failed to parse safetensors header in {path}: {message}")]
    InvalidSafetensorsHeader { path: PathBuf, message: String },
    #[error("unsupported tensor dtype {dtype} for tensor {name}")]
    UnsupportedDtype { name: String, dtype: String },
    #[error("invalid {model_type} conversion contract: {message}")]
    InvalidModelContract { model_type: String, message: String },
}

/// Convert a HuggingFace / MLX model directory into a `NativeModelManifest`.
///
/// The directory must contain `config.json` and one or more `model*.safetensors`
/// files. The returned manifest references the safetensors files by relative path,
/// so it can be written to the same directory as `model-manifest.json`.
pub fn convert_hf_model_dir(model_dir: &Path) -> Result<NativeModelManifest, ConvertError> {
    let config = load_hf_config(model_dir)?;
    let model_type = resolve_model_type(&config)?;
    let family = model_family_for_type(&model_type, &config)?;
    let arch = resolve_architecture(&config, &model_type)?;
    let safetensors_files = find_safetensors_files(model_dir)?;
    let all_tensors = parse_all_safetensors_headers(model_dir, &safetensors_files)?;
    let mut mapped_tensors = map_tensors(&config, &all_tensors, &family)?;

    // KV-shared layers have K/V weights in the checkpoint (mlx-lm ignores them), but
    // our manifest must not include them — the runtime reuses K/V from the source layer.
    // Build the shared-layer set early so we can filter before the manifest is constructed.
    let kv_shared_layers_early: std::collections::HashSet<u32> = {
        let layer_types_early = parse_layer_types(&config, &model_type, arch.layer_count);
        compute_kv_shared_sources(&config, &model_type, &layer_types_early, arch.layer_count)
            .into_keys()
            .collect()
    };
    if !kv_shared_layers_early.is_empty() {
        mapped_tensors.retain(|spec| {
            let is_kv_role = matches!(
                spec.role,
                NativeTensorRole::AttentionK | NativeTensorRole::AttentionV
            );
            !(is_kv_role
                && spec
                    .layer_index
                    .is_some_and(|li| kv_shared_layers_early.contains(&li)))
        });
    }

    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Record the tokenizer's <think>/</think> special-token ids in the
    // manifest so the runtime never has to guess them from the model family:
    // Qwen's two tokenizer generations place them at different ids
    // (151668/151669 for the ~151k vocab, 248068/248069 for the 248k one).
    let think_token_ids = parse_think_token_ids(model_dir);

    let (rope_theta, rope_theta_swa, partial_rotary_factor) =
        parse_rope_params(&config, &model_type);

    let (
        rope_scaling_type,
        rope_scaling_factor,
        rope_low_freq_factor,
        rope_high_freq_factor,
        rope_original_context_len,
    ) = parse_rope_scaling(&config, &model_type);

    let query_pre_attn_scalar =
        arch_f64(&config, &model_type, "query_pre_attn_scalar").and_then(f64_to_u32);

    let attention_logit_softcap =
        arch_f64(&config, &model_type, "attn_logit_softcapping").and_then(f64_to_u32);
    let rms_norm_eps = parse_rms_norm_eps(&config, &model_type);
    let linear_attention = linear_attention_config(&config, &model_type);
    let mla_attention = mla_attention_config(&config, &model_type);
    let glm_router = glm_router_config(&config, &model_type);

    let layer_types = parse_layer_types(&config, &model_type, arch.layer_count);
    let global_head_dim = arch_u64(&config, &model_type, "global_head_dim").and_then(u64_to_u32);
    // Unlimited-OCR uses ring-SWA (R-SWA): full attention during prefill, then a
    // decode-only ring of `sliding_window` slots. AX's standard SWA applies the
    // window during prefill as well and destroys OCR quality on ~273 soft tokens.
    // Keep full attention for this family until a dedicated R-SWA cache lands.
    let sliding_window_size = if is_unlimited_ocr(&model_type) {
        None
    } else {
        arch_u64(&config, &model_type, "sliding_window").and_then(u64_to_u32)
    };
    let final_logit_softcapping =
        arch_f64(&config, &model_type, "final_logit_softcapping").map(|v| v as f32);
    let hidden_size_per_layer_input = arch_u64(&config, &model_type, "hidden_size_per_layer_input")
        .and_then(u64_to_u32)
        .unwrap_or(0);
    let vocab_size_per_layer_input = if hidden_size_per_layer_input > 0 {
        arch_u64(&config, &model_type, "vocab_size_per_layer_input")
            .and_then(u64_to_u32)
            .filter(|v| *v > 0)
    } else {
        None
    };
    let kv_shared_source_layers =
        compute_kv_shared_sources(&config, &model_type, &layer_types, arch.layer_count);
    let attention_value_from_key_layers = compute_attention_value_from_key_layers(
        &config,
        &model_type,
        &layer_types,
        &kv_shared_source_layers,
        arch.layer_count,
    );

    let manifest = NativeModelManifest {
        schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: family.family_name.to_string(),
        tensor_format: NativeTensorFormat::Safetensors,
        source_quantization: None,
        runtime_status: runtime_status_for_model_type(&model_type),
        layer_count: arch.layer_count,
        hidden_size: arch.hidden_size,
        intermediate_size: arch.intermediate_size,
        attention_head_count: arch.attention_head_count,
        attention_head_dim: arch.attention_head_dim,
        kv_head_count: arch.kv_head_count,
        vocab_size: arch.vocab_size,
        tie_word_embeddings,
        rope_theta,
        rope_theta_swa,
        rope_scaling_type,
        rope_scaling_factor,
        rope_low_freq_factor,
        rope_high_freq_factor,
        rope_original_context_len,
        // Llama4 iRoPE period: mlx-lm hardcodes `(layer_idx + 1) % 4 != 0` for
        // use_rope. Do **not** reuse `interleave_moe_layer_step` (that selects MoE
        // layers). Prefer deriving the period from the `no_rope_layers` mask when
        // present (1 = RoPE, 0 = no-RoPE); fall back to 4.
        no_rope_layer_interval: if model_type == "llama4" {
            llama4_no_rope_layer_interval(&config, &model_type)
        } else {
            0
        },
        attn_temperature_floor: arch_f64(&config, &model_type, "floor_scale").and_then(f64_to_u32),
        attn_temperature_scale: arch_f64(&config, &model_type, "attn_scale").map(|v| v as f32),
        intermediate_size_mlp: arch_u64(&config, &model_type, "intermediate_size_mlp")
            .and_then(u64_to_u32)
            .unwrap_or(0),
        query_pre_attn_scalar,
        attention_logit_softcap,
        // Qwen3.5/Qwen3Next full-attention layers split q_proj into queries and a sigmoid
        // output gate. The MLX references instantiate that gate unconditionally, so absent
        // config metadata must default to the reference architecture rather than false.
        attn_output_gate: arch_bool(&config, &model_type, "attn_output_gate")
            .unwrap_or(defaults_attn_output_gate(&model_type)),
        partial_rotary_factor,
        rms_norm_eps,
        attention_value_from_key_layers,
        attention_v_norm_no_scale_layers: if is_gemma4_target_model_type(&model_type) {
            (0..arch.layer_count)
                .filter(|&i| !kv_shared_source_layers.contains_key(&i))
                .collect()
        } else {
            Vec::new()
        },
        global_head_dim,
        sliding_window_size,
        layer_types,
        kv_shared_source_layers,
        final_logit_softcapping,
        hidden_states_scale: if is_gemma4_target_model_type(&model_type)
            || is_embeddinggemma_model_type(&model_type)
        {
            Some((arch.hidden_size as f32).sqrt())
        } else {
            None
        },
        moe_norm_topk_prob: arch_bool(&config, &model_type, "norm_topk_prob")
            .unwrap_or(default_moe_norm_topk_prob(&model_type)),
        hidden_size_per_layer_input,
        vocab_size_per_layer_input,
        linear_attention,
        mla_attention,
        moe: moe_config(&config, &model_type),
        glm_router,
        // Converter assumes the on-disk weights are mlx-community pre-sanitized;
        // raw HuggingFace checkpoints need this set to `HfToMlx` by hand (or via
        // the doctor command when REQ-L4 lands). EmbeddingGemma's mlx-community
        // weights store raw Gemma `gamma` norms (mlx-lm applies `1 + weight` at
        // runtime), so lift the `+1` into the norm weights at load.
        weight_sanitize: if is_embeddinggemma_model_type(&model_type) {
            WeightSanitize::HfNormOnly
        } else {
            WeightSanitize::None
        },
        think_start_token_id: think_token_ids.0,
        think_end_token_id: think_token_ids.1,
        diffusion: parse_diffusion_config(&config, &model_type),
        tensors: mapped_tensors,
    };

    validate_converted_model_contract(&config, &model_type, &manifest)?;

    Ok(manifest)
}

/// Read `<think>` / `</think>` special-token ids from the model directory's
/// `tokenizer.json` `added_tokens` list.
///
/// Returns `(None, None)` when the file is absent, unparsable, or carries no
/// think tokens — families without think blocks simply never define them, and
/// the runtime falls back to family defaults for manifests converted before
/// this field was recorded.
fn parse_think_token_ids(model_dir: &Path) -> (Option<u32>, Option<u32>) {
    let path = model_dir.join("tokenizer.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return (None, None);
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return (None, None);
    };
    let Some(added) = value.get("added_tokens").and_then(|v| v.as_array()) else {
        return (None, None);
    };
    let mut start = None;
    let mut end = None;
    for token in added {
        let id = token.get("id").and_then(|i| i.as_u64()).map(|i| i as u32);
        match token.get("content").and_then(|c| c.as_str()) {
            Some("<think>") => start = id,
            Some("</think>") => end = id,
            _ => {}
        }
    }
    (start, end)
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

#[cfg(test)]
pub(crate) fn with_real_model_manifest_lock<T>(body: impl FnOnce() -> T) -> T {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _guard = LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("real-model manifest lock should not be poisoned");
    body()
}

#[cfg(test)]
pub(crate) fn ensure_manifest_for_hf_model_dir(model_dir: &Path) -> Result<(), ConvertError> {
    let manifest_path = model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    if manifest_path.exists() {
        return Ok(());
    }

    let manifest = convert_hf_model_dir(model_dir)?;
    write_manifest(model_dir, &manifest)
}

mod hf_config;
mod model_family;
mod tensor_mapping;
#[cfg(test)]
mod tests;

pub(crate) use hf_config::*;
pub(crate) use model_family::*;
pub(crate) use tensor_mapping::*;

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
    const MAX_SAFETENSORS_HEADER_SIZE: usize = 64 * 1024 * 1024;

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
    if header_size == 0 || header_size > MAX_SAFETENSORS_HEADER_SIZE {
        return Err(ConvertError::InvalidSafetensorsHeader {
            path: path.to_path_buf(),
            message: format!("header_size {header_size} is out of valid range"),
        });
    }

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
        .map(PathBuf::from)
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
            length_bytes: entry.data_offsets[1]
                .checked_sub(entry.data_offsets[0])
                .ok_or_else(|| ConvertError::InvalidSafetensorsHeader {
                    path: path.to_path_buf(),
                    message: format!(
                        "invalid data_offsets for tensor {name}: end ({}) < start ({})",
                        entry.data_offsets[1], entry.data_offsets[0]
                    ),
                })?,
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
        "U32" => Ok(NativeTensorDataType::U32),
        other => Err(ConvertError::UnsupportedDtype {
            name: name.to_string(),
            dtype: other.to_string(),
        }),
    }
}

fn tensor_quantization(
    config: &serde_json::Value,
    family: &ModelFamily,
    tensor_name: &str,
) -> Option<NativeTensorQuantization> {
    let global = config_quantization(config).unwrap_or_default();
    let mut quantization = global.clone();
    if let Some(override_quantization) = tensor_quantization_override(config, tensor_name) {
        quantization = override_quantization;
    }
    // mlx-lm's Gemma4 quantization predicate keeps router.proj at 8-bit while
    // the rest of the affine-quantized model uses the global 4-bit setting.
    if family.family_name == "gemma4" && tensor_name.ends_with(".router.proj.weight") {
        quantization.bits = 8;
    }
    Some(quantization)
}

fn config_quantization(config: &serde_json::Value) -> Option<NativeTensorQuantization> {
    let obj = quantization_root(config)?;
    parse_quantization_value(obj, false)
}

/// Top-level mlx-lm / OptiQ quant block. Prefer `quantization`, then
/// `quantization_config` (both present and usually identical on OptiQ cards).
fn quantization_root(config: &serde_json::Value) -> Option<&serde_json::Value> {
    config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
}

fn tensor_quantization_override(
    config: &serde_json::Value,
    tensor_name: &str,
) -> Option<NativeTensorQuantization> {
    let obj = quantization_root(config)?;
    let module_name = tensor_name.strip_suffix(".weight").unwrap_or(tensor_name);
    let unprefixed_name = tensor_name
        .strip_prefix("language_model.")
        .unwrap_or(tensor_name);
    let unprefixed_module_name = unprefixed_name
        .strip_suffix(".weight")
        .unwrap_or(unprefixed_name);
    let candidates = [
        tensor_name,
        module_name,
        unprefixed_name,
        unprefixed_module_name,
    ];
    candidates
        .iter()
        .find_map(|key| obj.get(*key))
        .and_then(|value| parse_quantization_value(value, true))
}

fn parse_quantization_value(
    value: &serde_json::Value,
    require_quantization_field: bool,
) -> Option<NativeTensorQuantization> {
    let object = value.as_object()?;
    // A class-predicate dictionary is forwarded straight to MLX's
    // `to_quantized`, rather than merged with nn.quantize's global arguments.
    // Match MLX's per-mode defaults for any fields the dictionary omits.
    let mode = object
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("affine")
        .to_string();
    let (default_group, default_bits) = match mode.as_str() {
        "mxfp4" => (32, 4),
        "nvfp4" => (16, 4),
        "mxfp8" => (32, 8),
        _ => (64, 4),
    };
    let group_size = object
        .get("group_size")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32)
        .unwrap_or(default_group);
    let bits = object
        .get("bits")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32)
        .unwrap_or(default_bits);
    // Reject a plain nested object that happened to match a tensor path but
    // carries no quantization settings. Top-level blocks may rely on defaults.
    if require_quantization_field
        && object.get("bits").is_none()
        && object.get("group_size").is_none()
        && object.get("mode").is_none()
    {
        return None;
    }
    Some(NativeTensorQuantization {
        mode,
        group_size,
        bits,
    })
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

    // Try extra per-family map (e.g. Qwen3 MoE switch-expert tensors).
    if let Some(extra) = family.extra_tensor_map {
        if let Some(result) = match_tensor_in_map(name, extra) {
            return Some(result);
        }
        if let Some(result) = match_prefixed_per_layer(name, "model.layers.", extra) {
            return Some(result);
        }
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
        if let Some(extra) = family.extra_tensor_map {
            if let Some(result) =
                match_prefixed_per_layer(name, "language_model.model.layers.", extra)
            {
                return Some(result);
            }
        }
        if is_qwen_gated_delta_family(family.family_name) {
            if let Some(result) =
                match_prefixed_per_layer(name, "model.layers.", QWEN35_LINEAR_TENSOR_MAP)
            {
                return Some(result);
            }
            if let Some(result) = match_prefixed_per_layer(
                name,
                "language_model.model.layers.",
                QWEN35_LINEAR_TENSOR_MAP,
            ) {
                return Some(result);
            }
        }
    } else if is_qwen_gated_delta_family(family.family_name) {
        if let Some(result) =
            match_prefixed_per_layer(name, "model.layers.", QWEN35_LINEAR_TENSOR_MAP)
        {
            return Some(result);
        }
    }

    // Try model.decoder.* prefix (DiffusionGemma)
    if family.uses_decoder_prefix {
        if let Some(result) = match_tensor_in_map(name, DECODER_PREFIX_TENSOR_MAP) {
            return Some(result);
        }
        if let Some(result) =
            match_prefixed_per_layer(name, "model.decoder.layers.", family.tensor_map)
        {
            return Some(result);
        }
        if let Some(extra) = family.extra_tensor_map {
            if let Some(result) = match_prefixed_per_layer(name, "model.decoder.layers.", extra) {
                return Some(result);
            }
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
    config: &serde_json::Value,
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
        let source_quantized = dtype == NativeTensorDataType::U32;
        let quantization = source_quantized
            .then(|| tensor_quantization(config, family, &entry.name))
            .flatten();

        mapped.push(NativeTensorSpec {
            name: entry.name.clone(),
            role,
            layer_index,
            dtype,
            source_tensor_type: None,
            source_quantized,
            quantization,
            quantized_source: None,
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
// Conversion contract validation
// ---------------------------------------------------------------------------

fn validate_converted_model_contract(
    config: &serde_json::Value,
    model_type: &str,
    manifest: &NativeModelManifest,
) -> Result<(), ConvertError> {
    if is_gemma4_target_model_type(model_type) {
        return validate_gemma4_contract(manifest);
    }
    if model_type == "gemma4_assistant" {
        return validate_gemma4_assistant_contract(manifest);
    }
    if is_glm4_moe_lite(model_type) {
        return validate_glm4_moe_lite_contract(config, manifest);
    }
    if matches!(model_type, "deepseek_v3" | "deepseek_v32") {
        return validate_deepseek_v3_contract(config, model_type, manifest);
    }
    if is_qwen_family_model_type(model_type) {
        validate_qwen_rope_scaling(config, model_type)?;
    }
    if is_qwen_gated_delta_family(model_type) {
        validate_qwen_gated_delta_contract(model_type, manifest)?;
    }

    Ok(())
}

fn validate_qwen_gated_delta_contract(
    model_type: &str,
    manifest: &NativeModelManifest,
) -> Result<(), ConvertError> {
    let Some(interval) = manifest
        .linear_attention
        .resolved_full_attention_interval(&manifest.model_family)
    else {
        return Ok(());
    };
    if interval == 0 {
        return invalid_model_contract(
            model_type,
            "linear_attention.full_attention_interval must be > 0",
        );
    }
    // Only enforce divisibility when the model has enough layers to include at
    // least one full-attention layer. Models with fewer layers than the interval
    // have no full-attention layers (all-linear), which is valid for testing but
    // unusual in production.
    if manifest.layer_count >= interval && !manifest.layer_count.is_multiple_of(interval) {
        return invalid_model_contract(
            model_type,
            format!(
                "layer_count ({}) must be divisible by full_attention_interval ({})",
                manifest.layer_count, interval
            ),
        );
    }
    Ok(())
}

fn validate_gemma4_contract(manifest: &NativeModelManifest) -> Result<(), ConvertError> {
    if manifest.layer_types.len() != manifest.layer_count as usize {
        return invalid_model_contract(
            "gemma4",
            format!(
                "layer_types must contain one entry per layer, got {} for layer_count {}",
                manifest.layer_types.len(),
                manifest.layer_count
            ),
        );
    }
    for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
        if layer_type != "sliding_attention" && layer_type != "full_attention" {
            return invalid_model_contract(
                "gemma4",
                format!(
                    "layer_types[{idx}] must be sliding_attention or full_attention, got {layer_type:?}"
                ),
            );
        }
    }

    if manifest.hidden_size_per_layer_input > 0 {
        require_gemma4_global_role(manifest, NativeTensorRole::PerLayerEmbedding)?;
        require_gemma4_global_role(manifest, NativeTensorRole::PerLayerModelProjection)?;
        require_gemma4_global_role(manifest, NativeTensorRole::PerLayerProjectionNorm)?;
        for layer_index in 0..manifest.layer_count {
            require_gemma4_layer_role(manifest, layer_index, NativeTensorRole::PerLayerInputGate)?;
            require_gemma4_layer_role(
                manifest,
                layer_index,
                NativeTensorRole::PerLayerInputProjection,
            )?;
            require_gemma4_layer_role(
                manifest,
                layer_index,
                NativeTensorRole::PerLayerInputPostNorm,
            )?;
        }
    }

    if manifest.moe.expert_count.is_some() {
        if manifest.moe.expert_intermediate_size.unwrap_or(0) == 0 {
            return invalid_model_contract(
                "gemma4",
                "moe.expert_intermediate_size must be > 0 for MoE models",
            );
        }
        for layer_index in 0..manifest.layer_count {
            require_gemma4_layer_role(manifest, layer_index, NativeTensorRole::FfnNorm2)?;
            require_gemma4_layer_role(manifest, layer_index, NativeTensorRole::FfnPostNorm1)?;
            require_gemma4_layer_role(manifest, layer_index, NativeTensorRole::FfnPostNorm2)?;
        }
    }

    Ok(())
}

fn validate_gemma4_assistant_contract(manifest: &NativeModelManifest) -> Result<(), ConvertError> {
    if manifest.layer_types.len() != manifest.layer_count as usize {
        return invalid_model_contract(
            "gemma4_assistant",
            format!(
                "layer_types must contain one entry per layer, got {} for layer_count {}",
                manifest.layer_types.len(),
                manifest.layer_count
            ),
        );
    }
    for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
        if layer_type != "sliding_attention" && layer_type != "full_attention" {
            return invalid_model_contract(
                "gemma4_assistant",
                format!(
                    "layer_types[{idx}] must be sliding_attention or full_attention, got {layer_type:?}"
                ),
            );
        }
    }
    if manifest.hidden_size_per_layer_input != 0 || manifest.vocab_size_per_layer_input.is_some() {
        return invalid_model_contract(
            "gemma4_assistant",
            "per-layer input embeddings are target-only and must be disabled",
        );
    }
    if manifest.moe.expert_count.is_some() {
        return invalid_model_contract(
            "gemma4_assistant",
            "Gemma4 assistant dense backend does not support target MoE tensors",
        );
    }
    require_model_global_role(
        "gemma4_assistant",
        manifest,
        NativeTensorRole::AssistantPreProjection,
    )?;
    require_model_global_role(
        "gemma4_assistant",
        manifest,
        NativeTensorRole::AssistantPostProjection,
    )?;
    for layer_index in 0..manifest.layer_count {
        require_model_role(
            "gemma4_assistant",
            manifest,
            layer_index,
            NativeTensorRole::AttentionQ,
        )?;
        require_model_role(
            "gemma4_assistant",
            manifest,
            layer_index,
            NativeTensorRole::AttentionO,
        )?;
    }
    Ok(())
}

fn require_gemma4_global_role(
    manifest: &NativeModelManifest,
    role: NativeTensorRole,
) -> Result<(), ConvertError> {
    if manifest
        .tensors
        .iter()
        .any(|tensor| tensor.layer_index.is_none() && tensor.role == role)
    {
        return Ok(());
    }

    invalid_model_contract(
        "gemma4",
        format!("manifest is missing required per-layer input tensor role {role:?}"),
    )
}

fn require_gemma4_layer_role(
    manifest: &NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
) -> Result<(), ConvertError> {
    if manifest
        .tensors
        .iter()
        .any(|tensor| tensor.layer_index == Some(layer_index) && tensor.role == role)
    {
        return Ok(());
    }

    invalid_model_contract(
        "gemma4",
        format!("layer {layer_index} is missing required per-layer input tensor role {role:?}"),
    )
}

fn validate_glm4_moe_lite_contract(
    config: &serde_json::Value,
    manifest: &NativeModelManifest,
) -> Result<(), ConvertError> {
    let model_type = "glm4_moe_lite";

    validate_glm4_moe_lite_rope_scaling(config)?;

    let first_dense_layers = arch_u64(config, "glm4_moe_lite", "first_k_dense_replace")
        .and_then(u64_to_u32)
        .unwrap_or(1)
        .min(manifest.layer_count);
    let has_shared_experts = arch_u64(config, "glm4_moe_lite", "n_shared_experts").unwrap_or(0) > 0;

    require_model_config(
        model_type,
        manifest.mla_attention.q_lora_rank,
        "mla_attention.q_lora_rank",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.kv_lora_rank,
        "mla_attention.kv_lora_rank",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.qk_nope_head_dim,
        "mla_attention.qk_nope_head_dim",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.qk_rope_head_dim,
        "mla_attention.qk_rope_head_dim",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.value_head_dim,
        "mla_attention.value_head_dim",
    )?;
    if manifest.glm_router.first_dense_layer_count.is_none() {
        return invalid_model_contract(
            "glm4_moe_lite",
            "glm_router.first_dense_layer_count must be configured",
        );
    }
    if manifest
        .glm_router
        .routed_scaling_factor
        .is_none_or(|value| !value.is_finite() || value <= 0.0)
    {
        return invalid_model_contract(
            "glm4_moe_lite",
            "glm_router.routed_scaling_factor must be finite and > 0",
        );
    }
    require_glm_config(manifest.glm_router.n_group, "glm_router.n_group")?;
    require_glm_config(manifest.glm_router.topk_group, "glm_router.topk_group")?;

    if let (Some(nope_dim), Some(rope_dim)) = (
        manifest.mla_attention.qk_nope_head_dim,
        manifest.mla_attention.qk_rope_head_dim,
    ) {
        if nope_dim + rope_dim != manifest.attention_head_dim {
            return invalid_model_contract(
                "glm4_moe_lite",
                format!(
                    "mla_attention qk_nope_head_dim + qk_rope_head_dim must equal attention_head_dim {}, got {} + {}",
                    manifest.attention_head_dim, nope_dim, rope_dim
                ),
            );
        }
    }

    for layer_index in 0..manifest.layer_count {
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionNorm)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionQa)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionQaNorm)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionQb)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionKvA)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionKvANorm)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionEmbedQ)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionUnembedOut)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionO)?;
        require_glm_role(manifest, layer_index, NativeTensorRole::AttentionPostNorm)?;

        if layer_index < first_dense_layers {
            require_glm_role(manifest, layer_index, NativeTensorRole::FfnGate)?;
            require_glm_role(manifest, layer_index, NativeTensorRole::FfnUp)?;
            require_glm_role(manifest, layer_index, NativeTensorRole::FfnDown)?;
        } else {
            require_glm_role(manifest, layer_index, NativeTensorRole::FfnGateInp)?;
            require_glm_role(
                manifest,
                layer_index,
                NativeTensorRole::FfnGateInpCorrectionBias,
            )?;
            require_glm_role(manifest, layer_index, NativeTensorRole::FfnGateExps)?;
            require_glm_role(manifest, layer_index, NativeTensorRole::FfnUpExps)?;
            require_glm_role(manifest, layer_index, NativeTensorRole::FfnDownExps)?;
            if has_shared_experts {
                require_glm_role(manifest, layer_index, NativeTensorRole::FfnSharedExpertGate)?;
                require_glm_role(manifest, layer_index, NativeTensorRole::FfnSharedExpertUp)?;
                require_glm_role(manifest, layer_index, NativeTensorRole::FfnSharedExpertDown)?;
            }
        }
    }

    Ok(())
}

fn validate_deepseek_v3_contract(
    config: &serde_json::Value,
    model_type: &str,
    manifest: &NativeModelManifest,
) -> Result<(), ConvertError> {
    require_model_config(
        model_type,
        manifest.mla_attention.q_lora_rank,
        "mla_attention.q_lora_rank",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.kv_lora_rank,
        "mla_attention.kv_lora_rank",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.qk_nope_head_dim,
        "mla_attention.qk_nope_head_dim",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.qk_rope_head_dim,
        "mla_attention.qk_rope_head_dim",
    )?;
    require_model_config(
        model_type,
        manifest.mla_attention.value_head_dim,
        "mla_attention.value_head_dim",
    )?;

    if let (Some(nope_dim), Some(rope_dim)) = (
        manifest.mla_attention.qk_nope_head_dim,
        manifest.mla_attention.qk_rope_head_dim,
    ) {
        if nope_dim + rope_dim != manifest.attention_head_dim {
            return invalid_model_contract(
                model_type,
                format!(
                    "mla_attention qk_nope_head_dim + qk_rope_head_dim must equal attention_head_dim {}, got {} + {}",
                    manifest.attention_head_dim, nope_dim, rope_dim
                ),
            );
        }
    }

    let first_dense_layers = arch_u64(config, model_type, "first_k_dense_replace")
        .and_then(u64_to_u32)
        .unwrap_or(0)
        .min(manifest.layer_count);
    let layer_freq = arch_u64(config, model_type, "moe_layer_freq")
        .and_then(u64_to_u32)
        .unwrap_or(1);
    if layer_freq == 0 {
        return invalid_model_contract(model_type, "moe_layer_freq must be greater than zero");
    }
    let has_shared_experts = arch_u64(config, model_type, "n_shared_experts").unwrap_or(0) > 0;

    for layer_index in 0..manifest.layer_count {
        for role in [
            NativeTensorRole::AttentionNorm,
            NativeTensorRole::AttentionQa,
            NativeTensorRole::AttentionQaNorm,
            NativeTensorRole::AttentionQb,
            NativeTensorRole::AttentionKvA,
            NativeTensorRole::AttentionKvANorm,
            NativeTensorRole::AttentionO,
            NativeTensorRole::AttentionPostNorm,
        ] {
            require_model_role(model_type, manifest, layer_index, role)?;
        }

        let has_kv_b = has_model_role(manifest, layer_index, NativeTensorRole::AttentionKvB);
        let has_embed_q = has_model_role(manifest, layer_index, NativeTensorRole::AttentionEmbedQ);
        let has_unembed_out =
            has_model_role(manifest, layer_index, NativeTensorRole::AttentionUnembedOut);
        if (has_kv_b && (has_embed_q || has_unembed_out))
            || (!has_kv_b && (!has_embed_q || !has_unembed_out))
        {
            return invalid_model_contract(
                model_type,
                format!(
                    "layer {layer_index} must provide exactly one MLA KV-B layout: AttentionKvB or AttentionEmbedQ plus AttentionUnembedOut"
                ),
            );
        }

        let is_moe_layer = layer_index >= first_dense_layers
            && layer_freq > 0
            && layer_index.is_multiple_of(layer_freq);
        if !is_moe_layer {
            require_model_role(model_type, manifest, layer_index, NativeTensorRole::FfnGate)?;
            require_model_role(model_type, manifest, layer_index, NativeTensorRole::FfnUp)?;
            require_model_role(model_type, manifest, layer_index, NativeTensorRole::FfnDown)?;
        } else {
            require_model_role(
                model_type,
                manifest,
                layer_index,
                NativeTensorRole::FfnGateInp,
            )?;
            require_model_role(
                model_type,
                manifest,
                layer_index,
                NativeTensorRole::FfnGateInpCorrectionBias,
            )?;
            require_model_role(
                model_type,
                manifest,
                layer_index,
                NativeTensorRole::FfnGateExps,
            )?;
            require_model_role(
                model_type,
                manifest,
                layer_index,
                NativeTensorRole::FfnUpExps,
            )?;
            require_model_role(
                model_type,
                manifest,
                layer_index,
                NativeTensorRole::FfnDownExps,
            )?;
            if has_shared_experts {
                require_model_role(
                    model_type,
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnSharedExpertGate,
                )?;
                require_model_role(
                    model_type,
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnSharedExpertUp,
                )?;
                require_model_role(
                    model_type,
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnSharedExpertDown,
                )?;
            }
        }
    }

    Ok(())
}

fn validate_glm4_moe_lite_rope_scaling(config: &serde_json::Value) -> Result<(), ConvertError> {
    if config
        .get("rope_scaling")
        .is_some_and(|rope_scaling| !rope_scaling.is_null())
    {
        return invalid_model_contract(
            "glm4_moe_lite",
            "rope_scaling is not yet supported for GLM MLA; mscale_all_dim changes attention scale and scaling_config changes RoPE frequencies",
        );
    }

    Ok(())
}

fn validate_qwen_rope_scaling(
    config: &serde_json::Value,
    model_type: &str,
) -> Result<(), ConvertError> {
    let has_unsupported_rope_scaling = config
        .get("rope_scaling")
        .is_some_and(|value| !value.is_null())
        || (uses_text_config(model_type)
            && config
                .get("text_config")
                .and_then(|tc| tc.get("rope_scaling"))
                .is_some_and(|value| !value.is_null()));
    if has_unsupported_rope_scaling {
        return invalid_model_contract(
            model_type,
            "rope_scaling is not yet supported for Qwen MLX runtime; current manifest/runtime only support absent or null rope_scaling",
        );
    }

    Ok(())
}

fn require_glm_role(
    manifest: &NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
) -> Result<(), ConvertError> {
    if has_model_role(manifest, layer_index, role) {
        return Ok(());
    }

    invalid_model_contract(
        "glm4_moe_lite",
        format!("layer {layer_index} is missing required draft tensor role {role:?}"),
    )
}

fn require_model_role(
    model_type: &str,
    manifest: &NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
) -> Result<(), ConvertError> {
    if has_model_role(manifest, layer_index, role) {
        return Ok(());
    }

    invalid_model_contract(
        model_type,
        format!("layer {layer_index} is missing required draft tensor role {role:?}"),
    )
}

fn require_model_global_role(
    model_type: &str,
    manifest: &NativeModelManifest,
    role: NativeTensorRole,
) -> Result<(), ConvertError> {
    if manifest
        .tensors
        .iter()
        .any(|tensor| tensor.layer_index.is_none() && tensor.role == role)
    {
        return Ok(());
    }

    invalid_model_contract(
        model_type,
        format!("missing required global tensor role {role:?}"),
    )
}

fn has_model_role(
    manifest: &NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
) -> bool {
    manifest
        .tensors
        .iter()
        .any(|tensor| tensor.layer_index == Some(layer_index) && tensor.role == role)
}

fn require_glm_config(value: Option<u32>, field: &str) -> Result<(), ConvertError> {
    if value.is_some_and(|value| value > 0) {
        return Ok(());
    }

    invalid_model_contract(
        "glm4_moe_lite",
        format!("{field} must be configured and > 0"),
    )
}

fn require_model_config(
    model_type: &str,
    value: Option<u32>,
    field: &str,
) -> Result<(), ConvertError> {
    if value.is_some_and(|value| value > 0) {
        return Ok(());
    }

    invalid_model_contract(model_type, format!("{field} must be configured and > 0"))
}

fn invalid_model_contract(
    model_type: &str,
    message: impl Into<String>,
) -> Result<(), ConvertError> {
    Err(ConvertError::InvalidModelContract {
        model_type: model_type.to_string(),
        message: message.into(),
    })
}

/// Llama4 iRoPE period for `no_rope_layer_interval`.
///
/// Runtime: no RoPE when `(layer_idx + 1) % interval == 0` (matches mlx-lm
/// `use_rope = (layer_idx + 1) % 4 != 0`).
///
/// Prefer deriving the period from HF `no_rope_layers` (1 = RoPE, 0 = no-RoPE).
/// Fall back to 4 — mlx-lm hardcodes that period and must not be confused with
/// `interleave_moe_layer_step` (MoE interleaving).
pub(super) fn llama4_no_rope_layer_interval(config: &serde_json::Value, model_type: &str) -> u32 {
    let mask = config
        .get("no_rope_layers")
        .or_else(|| {
            if uses_text_config(model_type) {
                config
                    .get("text_config")
                    .and_then(|tc| tc.get("no_rope_layers"))
            } else {
                None
            }
        })
        .and_then(|v| v.as_array());

    if let Some(mask) = mask {
        let flags: Vec<bool> = mask
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n != 0).or_else(|| v.as_bool()))
            .collect();
        if let Some(interval) = no_rope_period_from_rope_mask(&flags) {
            return interval;
        }
    }

    4
}

/// Infer iRoPE period from a repeating RoPE mask (`true` = use RoPE).
///
/// For Scout's `[1,1,1,0]` pattern returns 4 (no-RoPE every 4th layer, 0-based
/// indices 3, 7, 11, …).
fn no_rope_period_from_rope_mask(flags: &[bool]) -> Option<u32> {
    if flags.is_empty() {
        return None;
    }
    let no_rope: Vec<usize> = flags
        .iter()
        .enumerate()
        .filter_map(|(idx, use_rope)| (!use_rope).then_some(idx))
        .collect();
    if no_rope.is_empty() {
        // All layers use RoPE → interval 0 disables iRoPE branching.
        return Some(0);
    }
    // Period is the first no-rope index + 1 when that matches the full set.
    let period = no_rope[0] + 1;
    if period == 0 {
        return None;
    }
    let matches = no_rope.iter().all(|&idx| (idx + 1).is_multiple_of(period))
        && (0..flags.len()).all(|idx| {
            let expect_no_rope = (idx + 1).is_multiple_of(period);
            flags[idx] != expect_no_rope
        });
    if matches {
        u32::try_from(period).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
