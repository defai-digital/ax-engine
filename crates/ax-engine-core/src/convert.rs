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
    AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeLinearAttentionConfig, NativeModelManifest,
    NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType, NativeTensorFormat,
    NativeTensorQuantization, NativeTensorRole, NativeTensorSpec,
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

    let (rope_theta, rope_theta_swa, partial_rotary_factor) =
        parse_rope_params(&config, &model_type);

    let query_pre_attn_scalar =
        arch_f64(&config, &model_type, "query_pre_attn_scalar").and_then(f64_to_u32);

    let attention_logit_softcap =
        arch_f64(&config, &model_type, "attn_logit_softcapping").and_then(f64_to_u32);
    let linear_attention = linear_attention_config(&config, &model_type);

    let layer_types = parse_layer_types(&config, &model_type, arch.layer_count);
    let global_head_dim = arch_u64(&config, &model_type, "global_head_dim").and_then(u64_to_u32);
    let sliding_window_size = arch_u64(&config, &model_type, "sliding_window").and_then(u64_to_u32);
    let final_logit_softcapping =
        arch_f64(&config, &model_type, "final_logit_softcapping").map(|v| v as f32);
    let kv_shared_source_layers =
        compute_kv_shared_sources(&config, &model_type, &layer_types, arch.layer_count);
    let attention_value_from_key_layers = compute_attention_value_from_key_layers(
        &config,
        &model_type,
        &layer_types,
        &kv_shared_source_layers,
        arch.layer_count,
    );

    Ok(NativeModelManifest {
        schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
        model_family: family.family_name.to_string(),
        tensor_format: NativeTensorFormat::Safetensors,
        source_quantization: None,
        runtime_status: NativeRuntimeStatus::default(),
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
        query_pre_attn_scalar,
        attention_logit_softcap,
        // qwen3_next always uses a sigmoid output gate on the attention result (the gate is
        // split from q_proj output, not a separate configurable flag).  vllm's reference
        // implementation defaults this to true when absent: getattr(config, "attn_output_gate", True).
        // Absent the field in config.json, ax-engine must apply the same default.
        attn_output_gate: arch_bool(&config, &model_type, "attn_output_gate").unwrap_or(
            matches!(model_type.as_str(), "qwen3_next" | "qwen3_6" | "qwen3.6"),
        ),
        partial_rotary_factor,
        attention_value_from_key_layers,
        attention_v_norm_no_scale_layers: if model_type == "gemma4" {
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
        hidden_states_scale: if model_type == "gemma4" {
            Some((arch.hidden_size as f32).sqrt())
        } else {
            None
        },
        moe_norm_topk_prob: arch_bool(&config, &model_type, "norm_topk_prob").unwrap_or(false),
        hidden_size_per_layer_input: arch_u64(&config, &model_type, "hidden_size_per_layer_input")
            .and_then(u64_to_u32)
            .unwrap_or(0),
        vocab_size_per_layer_input: arch_u64(&config, &model_type, "vocab_size_per_layer_input")
            .and_then(u64_to_u32),
        linear_attention,
        moe: moe_config(&config, &model_type),
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

// ---------------------------------------------------------------------------
// Model family definitions
// ---------------------------------------------------------------------------

struct ModelFamily {
    family_name: &'static str,
    tensor_map: &'static [(&'static str, TensorMapping)],
    extra_tensor_map: Option<&'static [(&'static str, TensorMapping)]>,
    uses_language_model_prefix: bool,
}

#[derive(Clone, Copy)]
enum TensorMapping {
    Global(NativeTensorRole),
    PerLayer(NativeTensorRole),
}

/// Extra per-layer tensor patterns for Qwen3 MoE (mlp.gate → router; switch_mlp → experts).
const QWEN3_MOE_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "mlp.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mlp.switch_mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "mlp.switch_mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "mlp.switch_mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
];

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
///   model.layers.{i}.pre_feedforward_layernorm.weight
///   model.layers.{i}.post_feedforward_layernorm.weight
///   model.layers.{i}.pre_feedforward_layernorm_2.weight   (Gemma4 MoE)
///   model.layers.{i}.post_feedforward_layernorm_1.weight  (Gemma4 MoE)
///   model.layers.{i}.post_feedforward_layernorm_2.weight  (Gemma4 MoE)
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
    // Per-layer input gating global weights (Gemma4 2B/4B, sanitised model. prefix)
    (
        "model.embed_tokens_per_layer.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerEmbedding),
    ),
    (
        "model.per_layer_model_projection.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerModelProjection),
    ),
    (
        "model.per_layer_projection_norm.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerProjectionNorm),
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
        TensorMapping::PerLayer(NativeTensorRole::AttentionPostNorm),
    ),
    (
        "pre_feedforward_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnNorm),
    ),
    (
        "pre_feedforward_layernorm_2.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnNorm2),
    ),
    (
        "post_feedforward_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnPostNorm),
    ),
    (
        "post_feedforward_layernorm_1.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnPostNorm1),
    ),
    (
        "post_feedforward_layernorm_2.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnPostNorm2),
    ),
    // per-layer FFN
    (
        "mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGate),
    ),
    (
        "router.proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "router.scale",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpScale),
    ),
    (
        "router.per_expert_scale",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpExpertScale),
    ),
    (
        "layer_scalar",
        TensorMapping::PerLayer(NativeTensorRole::LayerScalar),
    ),
    // Per-layer input gating (Gemma4 2B/4B)
    (
        "per_layer_input_gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::PerLayerInputGate),
    ),
    (
        "per_layer_projection.weight",
        TensorMapping::PerLayer(NativeTensorRole::PerLayerInputProjection),
    ),
    (
        "post_per_layer_input_norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::PerLayerInputPostNorm),
    ),
    (
        "mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUp),
    ),
    (
        "mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDown),
    ),
    (
        "mlp.shared_expert_gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGateInp),
    ),
    (
        "mlp.shared_expert.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "mlp.shared_expert.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mlp.shared_expert.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
    // packed gate+up
    (
        "mlp.gate_up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateUpPacked),
    ),
    (
        "experts.gate_up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateUpExpsPacked),
    ),
    (
        "experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    (
        "experts.switch_glu.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "experts.switch_glu.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "experts.switch_glu.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
];

const QWEN35_LINEAR_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "linear_attn.in_proj_qkv.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjQkv),
    ),
    (
        "linear_attn.in_proj_qkvz.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjQkvz),
    ),
    (
        "linear_attn.in_proj_z.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjZ),
    ),
    (
        "linear_attn.in_proj_a.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjA),
    ),
    (
        "linear_attn.in_proj_b.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjB),
    ),
    (
        "linear_attn.in_proj_ba.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionInProjBa),
    ),
    (
        "linear_attn.conv1d.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionConv1d),
    ),
    (
        "linear_attn.dt_bias",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionDtBias),
    ),
    (
        "linear_attn.A_log",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionALog),
    ),
    (
        "linear_attn.norm.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionNorm),
    ),
    (
        "linear_attn.out_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::LinearAttentionOutProj),
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
    // Per-layer input gating global weights (Gemma4 2B/4B, language_model. prefix)
    (
        "language_model.model.embed_tokens_per_layer.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerEmbedding),
    ),
    (
        "language_model.model.per_layer_model_projection.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerModelProjection),
    ),
    (
        "language_model.model.per_layer_projection_norm.weight",
        TensorMapping::Global(NativeTensorRole::PerLayerProjectionNorm),
    ),
];

fn model_family_for_type(model_type: &str) -> Result<ModelFamily, ConvertError> {
    match model_type {
        "qwen3" => Ok(ModelFamily {
            family_name: "qwen3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: false,
        }),
        "qwen3_5" | "qwen3.5" | "qwen3_5_moe" | "qwen3_5_text" => Ok(ModelFamily {
            family_name: "qwen3_5",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: if matches!(model_type, "qwen3_5_moe" | "qwen3_5_text") {
                Some(QWEN3_MOE_EXTRA_TENSOR_MAP)
            } else {
                None
            },
            uses_language_model_prefix: true,
        }),
        "qwen3_next" | "qwen3.6" | "qwen3_6" => Ok(ModelFamily {
            family_name: "qwen3_next",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(QWEN3_MOE_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: true,
        }),
        "qwen3_moe" => Ok(ModelFamily {
            family_name: "qwen3_moe",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(QWEN3_MOE_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
        }),
        "gemma4" => Ok(ModelFamily {
            family_name: "gemma4",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
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
    intermediate_size: u32,
    attention_head_count: u32,
    attention_head_dim: u32,
    kv_head_count: u32,
    vocab_size: u32,
}

/// Whether this model type nests architecture params under `text_config`.
fn uses_text_config(model_type: &str) -> bool {
    matches!(
        model_type,
        "gemma4"
            | "qwen3_5"
            | "qwen3_5_moe"
            | "qwen3_5_text"
            | "qwen3_next"
            | "qwen3_6"
            | "qwen3.5"
            | "qwen3.6"
    )
}

fn is_qwen3_5_family(model_type: &str) -> bool {
    matches!(
        model_type,
        "qwen3_5" | "qwen3.5" | "qwen3_5_moe" | "qwen3_5_text"
    )
}

fn is_qwen_gated_delta_family(model_type: &str) -> bool {
    is_qwen3_5_family(model_type) || matches!(model_type, "qwen3_next" | "qwen3_6" | "qwen3.6")
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

fn arch_bool(config: &serde_json::Value, model_type: &str, field: &str) -> Option<bool> {
    config.get(field).and_then(|v| v.as_bool()).or_else(|| {
        if uses_text_config(model_type) {
            config
                .get("text_config")
                .and_then(|tc| tc.get(field))
                .and_then(|v| v.as_bool())
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

fn linear_attention_config(
    config: &serde_json::Value,
    model_type: &str,
) -> NativeLinearAttentionConfig {
    if !is_qwen_gated_delta_family(model_type) {
        return NativeLinearAttentionConfig::default();
    }

    NativeLinearAttentionConfig {
        full_attention_interval: arch_u64(config, model_type, "full_attention_interval")
            .and_then(u64_to_u32),
        num_value_heads: arch_u64(config, model_type, "linear_num_value_heads")
            .and_then(u64_to_u32),
        num_key_heads: arch_u64(config, model_type, "linear_num_key_heads").and_then(u64_to_u32),
        key_head_dim: arch_u64(config, model_type, "linear_key_head_dim").and_then(u64_to_u32),
        value_head_dim: arch_u64(config, model_type, "linear_value_head_dim").and_then(u64_to_u32),
        conv_kernel_dim: arch_u64(config, model_type, "linear_conv_kernel_dim")
            .and_then(u64_to_u32),
    }
}

fn moe_config(config: &serde_json::Value, model_type: &str) -> NativeMoeConfig {
    let is_gemma4_moe = arch_bool(config, model_type, "enable_moe_block").unwrap_or(false);
    let is_qwen3_moe = matches!(model_type, "qwen3_moe" | "qwen3_5_moe" | "qwen3_5_text");
    let is_qwen3_next_moe = model_type == "qwen3_next";
    if !is_gemma4_moe && !is_qwen3_moe && !is_qwen3_next_moe {
        return NativeMoeConfig::default();
    }

    let expert_count = arch_u64(config, model_type, "num_experts")
        .or_else(|| arch_u64(config, model_type, "num_local_experts"))
        .and_then(u64_to_u32);
    let experts_per_token = arch_u64(config, model_type, "top_k_experts")
        .or_else(|| arch_u64(config, model_type, "num_experts_per_tok"))
        .and_then(u64_to_u32);
    NativeMoeConfig {
        expert_count,
        experts_per_token,
        expert_intermediate_size: arch_u64(config, model_type, "moe_intermediate_size")
            .and_then(u64_to_u32),
    }
}

/// Parse per-model rope theta (main + SWA) and partial_rotary_factor.
///
/// Returns `(rope_theta, rope_theta_swa, partial_rotary_factor)`.
/// Gemma4 stores these nested under `text_config.rope_parameters.{full,sliding}_attention`.
/// Qwen3.5/Next stores them flat inside `text_config.rope_parameters`.
fn parse_rope_params(
    config: &serde_json::Value,
    model_type: &str,
) -> (Option<u32>, Option<u32>, Option<f32>) {
    if model_type == "gemma4" {
        let rp = config
            .get("text_config")
            .and_then(|tc| tc.get("rope_parameters"));
        let full_theta = rp
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| arch_f64(config, model_type, "rope_theta"))
            .and_then(f64_to_u32);
        let sliding_theta = rp
            .and_then(|rp| rp.get("sliding_attention"))
            .and_then(|sa| sa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .and_then(f64_to_u32);
        let partial_rotary = rp
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .or_else(|| arch_f64(config, model_type, "partial_rotary_factor"))
            .map(|v| v as f32)
            .filter(|&v| v <= 1.0);
        return (full_theta, sliding_theta, partial_rotary);
    }

    let theta = arch_f64(config, model_type, "rope_theta")
        .or_else(|| {
            let text_config = config.get("text_config")?;
            text_config
                .get("rope_parameters")
                .and_then(|rp| rp.get("rope_theta"))
                .and_then(|v| v.as_f64())
        })
        .and_then(f64_to_u32);

    let partial_rotary = arch_f64(config, model_type, "partial_rotary_factor")
        .or_else(|| {
            let text_config = config.get("text_config")?;
            text_config
                .get("rope_parameters")
                .and_then(|rp| rp.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
        })
        .map(|v| v as f32)
        .filter(|&v| v <= 1.0);

    (theta, None, partial_rotary)
}

/// Parse per-layer type list from config (e.g. Gemma4's `layer_types` field).
/// Gemma4 reference implementations derive it from `sliding_window_pattern`
/// when the field is omitted.
fn parse_layer_types(
    config: &serde_json::Value,
    model_type: &str,
    layer_count: u32,
) -> Vec<String> {
    if model_type != "gemma4" {
        return Vec::new();
    }
    if let Some(layer_types) = config
        .get("layer_types")
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|tc| tc.get("layer_types"))
        })
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
    {
        return layer_types;
    }

    let pattern = arch_u64(config, model_type, "sliding_window_pattern")
        .filter(|&pattern| pattern > 0)
        .unwrap_or(5);
    (0..layer_count)
        .map(|i| {
            if (u64::from(i) + 1).is_multiple_of(pattern) {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect()
}

/// Compute the KV-source mapping for KV-shared layers (Gemma4).
///
/// The last `num_kv_shared_layers` layers reuse K/V from the last non-shared
/// layer of the same type (sliding vs. full).
fn compute_kv_shared_sources(
    config: &serde_json::Value,
    model_type: &str,
    layer_types: &[String],
    layer_count: u32,
) -> BTreeMap<u32, u32> {
    if model_type != "gemma4" || layer_types.is_empty() {
        return BTreeMap::new();
    }
    let num_shared = arch_u64(config, model_type, "num_kv_shared_layers")
        .unwrap_or(20)
        .min(u64::from(layer_count)) as usize;
    let non_shared_count = layer_count as usize - num_shared;

    let mut last_full: Option<u32> = None;
    let mut last_sliding: Option<u32> = None;
    let mut sources = BTreeMap::new();

    for (i, lt) in layer_types.iter().enumerate() {
        if i < non_shared_count {
            if lt == "full_attention" {
                last_full = Some(i as u32);
            } else {
                last_sliding = Some(i as u32);
            }
        } else if let Some(src) = if lt == "full_attention" {
            last_full
        } else {
            last_sliding
        } {
            sources.insert(i as u32, src);
        }
    }
    sources
}

fn compute_attention_value_from_key_layers(
    config: &serde_json::Value,
    model_type: &str,
    layer_types: &[String],
    kv_shared_source_layers: &BTreeMap<u32, u32>,
    layer_count: u32,
) -> Vec<u32> {
    if model_type != "gemma4" || !arch_bool(config, model_type, "attention_k_eq_v").unwrap_or(false)
    {
        return Vec::new();
    }

    (0..layer_count)
        .filter(|&i| {
            layer_types
                .get(i as usize)
                .is_some_and(|layer_type| layer_type == "full_attention")
                && !kv_shared_source_layers.contains_key(&i)
        })
        .collect()
}

fn f64_to_u32(value: f64) -> Option<u32> {
    if value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX) {
        Some(value as u32)
    } else {
        None
    }
}

fn u64_to_u32(value: u64) -> Option<u32> {
    u32::try_from(value).ok()
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
        .unwrap_or_else(|| hidden_size.checked_div(attention_head_count).unwrap_or(0));
    let layer_count = require_arch_u64(config, model_type, "num_hidden_layers")? as u32;
    let vocab_size = require_arch_u64(config, model_type, "vocab_size")? as u32;
    let intermediate_size = arch_u64(config, model_type, "intermediate_size")
        .map(|v| v as u32)
        .unwrap_or(0);

    Ok(ArchitectureParams {
        layer_count,
        hidden_size,
        intermediate_size,
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
    let mut quantization = config_quantization(config).unwrap_or_default();
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
    let obj = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))?;
    let mode = obj
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("affine")
        .to_string();
    let group_size = obj
        .get("group_size")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32)
        .unwrap_or(64);
    let bits = obj
        .get("bits")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32)
        .unwrap_or(4);
    Some(NativeTensorQuantization {
        mode,
        group_size,
        bits,
    })
}

fn tensor_quantization_override(
    config: &serde_json::Value,
    tensor_name: &str,
) -> Option<NativeTensorQuantization> {
    let obj = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))?;
    let candidates = [
        tensor_name,
        tensor_name.strip_suffix(".weight").unwrap_or(tensor_name),
        tensor_name
            .strip_prefix("language_model.")
            .unwrap_or(tensor_name),
        tensor_name
            .strip_prefix("language_model.")
            .unwrap_or(tensor_name)
            .strip_suffix(".weight")
            .unwrap_or(tensor_name),
    ];
    candidates
        .iter()
        .find_map(|key| obj.get(*key))
        .and_then(parse_quantization_value)
}

fn parse_quantization_value(value: &serde_json::Value) -> Option<NativeTensorQuantization> {
    let object = value.as_object()?;
    let mode = object
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("affine")
        .to_string();
    let group_size = object
        .get("group_size")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32)
        .unwrap_or(64);
    let bits = object
        .get("bits")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32)
        .unwrap_or(4);
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
                "F32" | "U32" => 4,
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
    fn gemma4_moe_expert_names_map_to_unambiguous_roles() {
        let family = model_family_for_type("gemma4").expect("gemma4 should be supported");

        assert_eq!(
            match_tensor(
                "language_model.model.layers.0.experts.gate_up_proj.weight",
                &family,
            ),
            Some((NativeTensorRole::FfnGateUpExpsPacked, Some(0)))
        );
        assert_eq!(
            match_tensor(
                "language_model.model.layers.0.experts.switch_glu.gate_proj.weight",
                &family,
            ),
            Some((NativeTensorRole::FfnGateExps, Some(0)))
        );
        assert_eq!(
            match_tensor(
                "language_model.model.layers.0.experts.switch_glu.up_proj.weight",
                &family,
            ),
            Some((NativeTensorRole::FfnUpExps, Some(0)))
        );
        assert_eq!(
            match_tensor(
                "language_model.model.layers.0.experts.switch_glu.down_proj.weight",
                &family,
            ),
            Some((NativeTensorRole::FfnDownExps, Some(0)))
        );
    }

    #[test]
    fn converts_gemma4_default_layer_type_pattern_for_k_eq_v() {
        let dir = unique_test_dir("gemma4_default_layer_types");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "gemma4",
                "vocab_size": 262144,
                "text_config": {
                    "hidden_size": 3072,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 4,
                    "head_dim": 128,
                    "global_head_dim": 256,
                    "attention_k_eq_v": true,
                    "sliding_window_pattern": 5,
                    "num_hidden_layers": 5,
                    "vocab_size": 262144
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
                ("model.layers.4.input_layernorm.weight", "BF16", &[3072]),
                (
                    "model.layers.4.self_attn.q_proj.weight",
                    "BF16",
                    &[8192, 3072],
                ),
                (
                    "model.layers.4.self_attn.k_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "model.layers.4.self_attn.o_proj.weight",
                    "BF16",
                    &[3072, 8192],
                ),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

        assert_eq!(
            manifest.layer_types,
            vec![
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "sliding_attention".to_string(),
                "full_attention".to_string(),
            ]
        );
        assert_eq!(manifest.attention_value_from_key_layers, vec![4]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn parses_root_gemma4_layer_types_before_text_config() {
        let config = serde_json::json!({
            "model_type": "gemma4",
            "layer_types": ["full_attention"],
            "text_config": {
                "layer_types": ["sliding_attention"]
            }
        });

        let layer_types = parse_layer_types(&config, "gemma4", 1);

        assert_eq!(layer_types, vec!["full_attention".to_string()]);
    }

    #[test]
    fn converts_gemma4_default_kv_shared_layers() {
        let dir = unique_test_dir("gemma4_default_kv_shared");
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
                    "global_head_dim": 256,
                    "sliding_window_pattern": 5,
                    "num_hidden_layers": 35,
                    "vocab_size": 262144
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
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

        assert_eq!(manifest.layer_types.len(), 35);
        assert_eq!(manifest.layer_types[4], "full_attention");
        assert_eq!(manifest.layer_types[34], "full_attention");
        assert_eq!(manifest.kv_shared_source_layers.len(), 20);
        assert_eq!(manifest.kv_shared_source_layers.get(&15), Some(&13));
        assert_eq!(manifest.kv_shared_source_layers.get(&19), Some(&14));
        assert!(!manifest.kv_shared_source_layers.contains_key(&14));
        assert!(manifest.attention_v_norm_no_scale_layers.contains(&14));
        assert!(!manifest.attention_v_norm_no_scale_layers.contains(&15));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn converts_gemma4_k_eq_v_full_attention_layers() {
        let dir = unique_test_dir("gemma4_k_eq_v");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "gemma4",
                "vocab_size": 262144,
                "text_config": {
                    "hidden_size": 3072,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 4,
                    "head_dim": 128,
                    "global_head_dim": 256,
                    "attention_k_eq_v": true,
                    "num_hidden_layers": 1,
                    "vocab_size": 262144,
                    "layer_types": ["full_attention"]
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
                    &[8192, 3072],
                ),
                (
                    "model.layers.0.self_attn.k_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    "BF16",
                    &[3072, 8192],
                ),
                ("model.layers.0.self_attn.q_norm.weight", "BF16", &[256]),
                ("model.layers.0.self_attn.k_norm.weight", "BF16", &[256]),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "model.layers.0.pre_feedforward_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "model.layers.0.post_feedforward_layernorm.weight",
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
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("gemma4 conversion should succeed");

        assert_eq!(manifest.attention_value_from_key_layers, vec![0]);
        assert!(
            !manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::AttentionV),
            "K=V Gemma4 full-attention layer should not require v_proj"
        );
        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("K=V Gemma4 manifest should validate");

        let _ = fs::remove_dir_all(dir);
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
                    "global_head_dim": 256,
                    "sliding_window": 1024,
                    "num_hidden_layers": 2,
                    "vocab_size": 262144,
                    "rope_theta": 1000000,
                    "final_logit_softcapping": 30.0,
                    "layer_types": ["sliding_attention", "sliding_attention"],
                    "num_kv_shared_layers": 1,
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
                    "model.layers.0.pre_feedforward_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "model.layers.0.post_feedforward_layernorm.weight",
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
                    "model.layers.1.self_attn.o_proj.weight",
                    "BF16",
                    &[3072, 4096],
                ),
                ("model.layers.1.self_attn.q_norm.weight", "BF16", &[128]),
                (
                    "model.layers.1.post_attention_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "model.layers.1.pre_feedforward_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "model.layers.1.post_feedforward_layernorm.weight",
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
        assert_eq!(manifest.global_head_dim, Some(256));
        assert_eq!(manifest.sliding_window_size, Some(1024));
        assert_eq!(manifest.final_logit_softcapping, Some(30.0));
        assert_eq!(manifest.hidden_states_scale, Some((3072_f32).sqrt()));
        assert!(!manifest.moe_norm_topk_prob);
        assert_eq!(manifest.attention_v_norm_no_scale_layers, vec![0]);
        assert_eq!(
            manifest.layer_types,
            vec![
                "sliding_attention".to_string(),
                "sliding_attention".to_string()
            ]
        );
        assert_eq!(manifest.kv_shared_source_layers.get(&1), Some(&0));
        assert!(
            !manifest.tensors.iter().any(|tensor| {
                tensor.layer_index == Some(1)
                    && matches!(
                        tensor.role,
                        NativeTensorRole::AttentionK | NativeTensorRole::AttentionV
                    )
            }),
            "KV-shared Gemma4 layers should reuse source K/V instead of mapping their own"
        );

        let has_lm_head = manifest
            .tensors
            .iter()
            .any(|t| t.role == NativeTensorRole::LmHead);
        assert!(has_lm_head);

        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("Gemma4 KV-shared manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn converts_qwen3_5_linear_attention_model_directory() {
        let dir = unique_test_dir("qwen3_5_linear");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "qwen3_5",
                "vocab_size": 32,
                "text_config": {
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "num_hidden_layers": 1,
                    "vocab_size": 32,
                    "linear_num_value_heads": 2,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 2,
                    "linear_conv_kernel_dim": 4,
                    "full_attention_interval": 4
                }
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("language_model.model.embed_tokens.weight", "BF16", &[32, 8]),
                ("language_model.model.norm.weight", "BF16", &[8]),
                ("language_model.lm_head.weight", "BF16", &[32, 8]),
                (
                    "language_model.model.layers.0.input_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                    "BF16",
                    &[12, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_z.weight",
                    "BF16",
                    &[4, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_a.weight",
                    "BF16",
                    &[2, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_b.weight",
                    "BF16",
                    &[2, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.conv1d.weight",
                    "BF16",
                    &[12, 1, 4],
                ),
                (
                    "language_model.model.layers.0.linear_attn.dt_bias",
                    "F32",
                    &[2],
                ),
                (
                    "language_model.model.layers.0.linear_attn.A_log",
                    "F32",
                    &[2],
                ),
                (
                    "language_model.model.layers.0.linear_attn.norm.weight",
                    "BF16",
                    &[2],
                ),
                (
                    "language_model.model.layers.0.linear_attn.out_proj.weight",
                    "BF16",
                    &[8, 4],
                ),
                (
                    "language_model.model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                (
                    "language_model.model.layers.0.pre_feedforward_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                (
                    "language_model.model.layers.0.mlp.gate_proj.weight",
                    "BF16",
                    &[16, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.up_proj.weight",
                    "BF16",
                    &[16, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.down_proj.weight",
                    "BF16",
                    &[8, 16],
                ),
            ],
        );

        let manifest =
            convert_hf_model_dir(&dir).expect("qwen3.5 linear conversion should succeed");

        assert_eq!(manifest.model_family, "qwen3_5");
        assert_eq!(manifest.linear_attention.num_value_heads, Some(2));
        assert_eq!(manifest.linear_attention.num_key_heads, Some(1));
        assert_eq!(manifest.linear_attention.key_head_dim, Some(4));
        assert_eq!(manifest.linear_attention.value_head_dim, Some(2));
        assert_eq!(manifest.linear_attention.conv_kernel_dim, Some(4));
        assert_eq!(manifest.linear_attention.full_attention_interval, Some(4));
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::LinearAttentionInProjQkv)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::LinearAttentionConv1d)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::LinearAttentionOutProj)
        );

        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("linear-attention manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn converts_qwen3_5_moe_language_model_switch_mlp_directory() {
        let dir = unique_test_dir("qwen3_5_moe_language_model");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "qwen3_5_moe",
                "vocab_size": 32,
                "text_config": {
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "num_hidden_layers": 1,
                    "vocab_size": 32,
                    "linear_num_value_heads": 2,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 2,
                    "linear_conv_kernel_dim": 4,
                    "full_attention_interval": 4,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 8
                },
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "language_model.model.layers.0.mlp.gate": {
                        "group_size": 64,
                        "bits": 8
                    }
                }
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("language_model.model.embed_tokens.weight", "BF16", &[32, 8]),
                ("language_model.model.norm.weight", "BF16", &[8]),
                ("language_model.lm_head.weight", "BF16", &[32, 8]),
                (
                    "language_model.model.layers.0.input_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
                    "BF16",
                    &[12, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_z.weight",
                    "BF16",
                    &[4, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_a.weight",
                    "BF16",
                    &[2, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.in_proj_b.weight",
                    "BF16",
                    &[2, 8],
                ),
                (
                    "language_model.model.layers.0.linear_attn.conv1d.weight",
                    "BF16",
                    &[12, 1, 4],
                ),
                (
                    "language_model.model.layers.0.linear_attn.dt_bias",
                    "F32",
                    &[2],
                ),
                (
                    "language_model.model.layers.0.linear_attn.A_log",
                    "F32",
                    &[2],
                ),
                (
                    "language_model.model.layers.0.linear_attn.norm.weight",
                    "BF16",
                    &[2],
                ),
                (
                    "language_model.model.layers.0.linear_attn.out_proj.weight",
                    "BF16",
                    &[8, 4],
                ),
                (
                    "language_model.model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                (
                    "language_model.model.layers.0.mlp.gate.weight",
                    "U32",
                    &[4, 2],
                ),
                (
                    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    "BF16",
                    &[4, 8, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
                    "BF16",
                    &[4, 8, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
                    "BF16",
                    &[4, 8, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.shared_expert_gate.weight",
                    "BF16",
                    &[1, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
                    "BF16",
                    &[8, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
                    "BF16",
                    &[8, 8],
                ),
                (
                    "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
                    "BF16",
                    &[8, 8],
                ),
            ],
        );

        let manifest =
            convert_hf_model_dir(&dir).expect("qwen3.5 MoE conversion should succeed");

        assert_eq!(manifest.model_family, "qwen3_5");
        assert_eq!(manifest.moe.expert_count, Some(4));
        assert_eq!(manifest.moe.experts_per_token, Some(2));
        assert_eq!(manifest.moe.expert_intermediate_size, Some(8));
        for role in [
            NativeTensorRole::FfnGateInp,
            NativeTensorRole::FfnGateExps,
            NativeTensorRole::FfnUpExps,
            NativeTensorRole::FfnDownExps,
            NativeTensorRole::FfnSharedExpertDown,
        ] {
            assert!(
                manifest.tensors.iter().any(|tensor| tensor.role == role),
                "missing role {role:?}"
            );
        }
        let gate = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
            .expect("router should map");
        assert_eq!(gate.quantization.as_ref().map(|q| q.bits), Some(8));

        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("qwen3.5 MoE manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn maps_qwen3_moe_switch_mlp_tensors() {
        let family = model_family_for_type("qwen3_moe").expect("qwen3_moe should be supported");

        assert_eq!(
            match_tensor("model.layers.2.mlp.gate.weight", &family),
            Some((NativeTensorRole::FfnGateInp, Some(2)))
        );
        assert_eq!(
            match_tensor("model.layers.2.mlp.switch_mlp.gate_proj.weight", &family),
            Some((NativeTensorRole::FfnGateExps, Some(2)))
        );
        assert_eq!(
            match_tensor("model.layers.2.mlp.switch_mlp.up_proj.weight", &family),
            Some((NativeTensorRole::FfnUpExps, Some(2)))
        );
        assert_eq!(
            match_tensor("model.layers.2.mlp.switch_mlp.down_proj.weight", &family),
            Some((NativeTensorRole::FfnDownExps, Some(2)))
        );
    }

    #[test]
    fn converts_qwen3_next_linear_moe_shared_expert_model_directory() {
        let dir = unique_test_dir("qwen3_next_linear_moe");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "qwen3_next",
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "num_hidden_layers": 1,
                "vocab_size": 128,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "norm_topk_prob": true,
                "quantization_config": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                    "model.layers.0.mlp.gate": {
                        "group_size": 64,
                        "bits": 8
                    },
                    "model.layers.0.mlp.shared_expert_gate": {
                        "group_size": 64,
                        "bits": 8
                    }
                }
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "BF16", &[128, 64]),
                ("model.norm.weight", "BF16", &[64]),
                ("lm_head.weight", "BF16", &[128, 64]),
                ("model.layers.0.input_layernorm.weight", "BF16", &[64]),
                (
                    "model.layers.0.linear_attn.in_proj_qkvz.weight",
                    "BF16",
                    &[128, 64],
                ),
                (
                    "model.layers.0.linear_attn.in_proj_ba.weight",
                    "BF16",
                    &[4, 64],
                ),
                (
                    "model.layers.0.linear_attn.conv1d.weight",
                    "BF16",
                    &[96, 1, 4],
                ),
                ("model.layers.0.linear_attn.dt_bias", "F32", &[2]),
                ("model.layers.0.linear_attn.A_log", "F32", &[2]),
                ("model.layers.0.linear_attn.norm.weight", "BF16", &[16]),
                (
                    "model.layers.0.linear_attn.out_proj.weight",
                    "BF16",
                    &[64, 32],
                ),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[64],
                ),
                ("model.layers.0.mlp.gate.weight", "U32", &[4, 16]),
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    "BF16",
                    &[4, 8, 64],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.up_proj.weight",
                    "BF16",
                    &[4, 8, 64],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.down_proj.weight",
                    "BF16",
                    &[4, 64, 8],
                ),
                (
                    "model.layers.0.mlp.shared_expert_gate.weight",
                    "U32",
                    &[1, 16],
                ),
                (
                    "model.layers.0.mlp.shared_expert.gate_proj.weight",
                    "BF16",
                    &[8, 64],
                ),
                (
                    "model.layers.0.mlp.shared_expert.up_proj.weight",
                    "BF16",
                    &[8, 64],
                ),
                (
                    "model.layers.0.mlp.shared_expert.down_proj.weight",
                    "BF16",
                    &[64, 8],
                ),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("qwen3_next conversion should succeed");

        assert_eq!(manifest.model_family, "qwen3_next");
        assert_eq!(manifest.linear_attention.full_attention_interval, Some(4));
        assert_eq!(manifest.moe.expert_count, Some(4));
        assert_eq!(manifest.moe.experts_per_token, Some(2));
        assert_eq!(manifest.moe.expert_intermediate_size, Some(8));
        assert!(manifest.moe_norm_topk_prob);
        // attn_output_gate must default to true for qwen3_next even when absent from config.json.
        // All full-attention layers in the qwen3_next architecture use the sigmoid output gate.
        assert!(manifest.attn_output_gate, "qwen3_next attn_output_gate must default to true");
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnSharedExpertDown)
        );
        let gate = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
            .expect("Qwen3Next router should map");
        assert_eq!(gate.quantization.as_ref().map(|q| q.bits), Some(8));
        let shared_gate = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == NativeTensorRole::FfnSharedExpertGateInp)
            .expect("Qwen3Next shared expert gate should map");
        assert_eq!(shared_gate.quantization.as_ref().map(|q| q.bits), Some(8));

        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("qwen3_next manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn converts_gemma4_moe_model_directory() {
        let dir = unique_test_dir("gemma4_moe");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "gemma4",
                "vocab_size": 262144,
                "tie_word_embeddings": true,
                "text_config": {
                    "hidden_size": 2816,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "num_hidden_layers": 1,
                    "vocab_size": 262144,
                    "enable_moe_block": true,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 704
                },
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine"
                }
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                (
                    "language_model.model.embed_tokens.weight",
                    "BF16",
                    &[262144, 2816],
                ),
                ("language_model.model.norm.weight", "BF16", &[2816]),
                (
                    "language_model.model.layers.0.input_layernorm.weight",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.self_attn.q_proj.weight",
                    "BF16",
                    &[2048, 2816],
                ),
                (
                    "language_model.model.layers.0.self_attn.k_proj.weight",
                    "BF16",
                    &[512, 2816],
                ),
                (
                    "language_model.model.layers.0.self_attn.v_proj.weight",
                    "BF16",
                    &[512, 2816],
                ),
                (
                    "language_model.model.layers.0.self_attn.o_proj.weight",
                    "BF16",
                    &[2816, 2048],
                ),
                (
                    "language_model.model.layers.0.self_attn.q_norm.weight",
                    "BF16",
                    &[256],
                ),
                (
                    "language_model.model.layers.0.self_attn.k_norm.weight",
                    "BF16",
                    &[256],
                ),
                (
                    "language_model.model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.pre_feedforward_layernorm.weight",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.post_feedforward_layernorm.weight",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.pre_feedforward_layernorm_2.weight",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.post_feedforward_layernorm_1.weight",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.post_feedforward_layernorm_2.weight",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.mlp.gate_proj.weight",
                    "BF16",
                    &[2112, 2816],
                ),
                (
                    "language_model.model.layers.0.mlp.up_proj.weight",
                    "BF16",
                    &[2112, 2816],
                ),
                (
                    "language_model.model.layers.0.mlp.down_proj.weight",
                    "BF16",
                    &[2816, 2112],
                ),
                (
                    "language_model.model.layers.0.router.proj.weight",
                    "U32",
                    &[128, 704],
                ),
                (
                    "language_model.model.layers.0.router.scale",
                    "BF16",
                    &[2816],
                ),
                (
                    "language_model.model.layers.0.experts.switch_glu.gate_proj.weight",
                    "BF16",
                    &[128, 704, 2816],
                ),
                (
                    "language_model.model.layers.0.experts.switch_glu.up_proj.weight",
                    "BF16",
                    &[128, 704, 2816],
                ),
                (
                    "language_model.model.layers.0.experts.switch_glu.down_proj.weight",
                    "BF16",
                    &[128, 2816, 704],
                ),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("gemma4 moe conversion should succeed");

        assert_eq!(manifest.model_family, "gemma4");
        assert_eq!(manifest.moe.expert_count, Some(128));
        assert_eq!(manifest.moe.experts_per_token, Some(8));
        assert_eq!(manifest.moe.expert_intermediate_size, Some(704));
        assert_eq!(manifest.hidden_states_scale, Some((2816_f32).sqrt()));
        assert!(!manifest.moe_norm_topk_prob);
        assert_eq!(manifest.attention_v_norm_no_scale_layers, vec![0]);
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
        );
        let router = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.role == NativeTensorRole::FfnGateInp)
            .expect("router should map");
        assert!(router.source_quantized);
        assert_eq!(
            router.quantization.as_ref().map(|q| q.bits),
            Some(8),
            "Gemma4 router.proj should keep mlx-lm's 8-bit quantization contract"
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::AttentionPostNorm)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnNorm2)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnPostNorm)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnPostNorm1)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnPostNorm2)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnGateExps)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::FfnDownExps)
        );

        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir).expect("moe manifest should validate");

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
                (
                    "model.layers.0.pre_feedforward_layernorm.weight",
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
        assert_eq!(manifest.tensors.len(), 12);

        let names: Vec<_> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(!names.contains(&"model.layers.0.self_attn.rotary_emb.inv_freq"));
        assert!(!names.contains(&"some.unknown.tensor"));

        let _ = fs::remove_dir_all(dir);
    }

    /// Real model integration test — uses `.internal/models/Qwen3.5-2B-bf16` when available.
    /// If the model is absent locally, the test exits early without failing.
    #[test]
    fn converts_real_qwen3_5_bf16_model() {
        let model_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.internal/models/Qwen3.5-2B-bf16");
        if !model_dir.join("config.json").exists() {
            eprintln!("skipping: model not downloaded at {}", model_dir.display());
            return;
        }

        with_real_model_manifest_lock(|| {
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

            // Write manifest, then validate the full NativeModelArtifacts pipeline.
            // Hold the shared lock for the whole duration so parallel metal tests
            // never observe the temporary cleanup window.
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

            // Clean up the generated manifest before releasing the shared lock.
            let _ = fs::remove_file(model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE));
        });
    }
}
