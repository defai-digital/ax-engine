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
        "unsupported model type {model_type}; supported: qwen3, qwen3_5, qwen3_next, gemma4, gemma4_unified, gemma4_assistant, diffusion_gemma, glm4_moe_lite, llama3, mistral3, mixtral, deepseek_v3, llama4 draft manifests"
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
    let sliding_window_size = arch_u64(&config, &model_type, "sliding_window").and_then(u64_to_u32);
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
        no_rope_layer_interval: if model_type == "llama4" {
            arch_u64(&config, &model_type, "interleave_moe_layer_step")
                .and_then(u64_to_u32)
                .unwrap_or(0)
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
        hidden_states_scale: if is_gemma4_target_model_type(&model_type) {
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
        // the doctor command when REQ-L4 lands).
        weight_sanitize: WeightSanitize::None,
        think_start_token_id: None,
        think_end_token_id: None,
        diffusion: parse_diffusion_config(&config, &model_type),
        tensors: mapped_tensors,
    };

    validate_converted_model_contract(&config, &model_type, &manifest)?;

    Ok(manifest)
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

/// Extra per-layer tensor patterns for GLM4MoELite.
///
/// These roles intentionally make the manifest graph-specific instead of
/// pretending GLM's MLA projections are ordinary split Q/K/V attention.
const GLM4_MOE_LITE_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "self_attn.q_a_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQa),
    ),
    (
        "self_attn.q_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQaNorm),
    ),
    (
        "self_attn.q_b_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQb),
    ),
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvA),
    ),
    (
        "self_attn.kv_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvANorm),
    ),
    (
        "self_attn.embed_q.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionEmbedQ),
    ),
    (
        "self_attn.unembed_out.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionUnembedOut),
    ),
    (
        "mlp.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mlp.gate.e_score_correction_bias",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpCorrectionBias),
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
    (
        "mlp.shared_experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "mlp.shared_experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mlp.shared_experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
];

/// Extra per-layer tensor patterns for DeepSeek V3/V3.2.
///
/// Raw HuggingFace checkpoints store the MLA KV-B projection as
/// `kv_b_proj.weight`; the MLX runtime splits it into the same `embed_q` and
/// `unembed_out` layout used by mlx-lm at load time.
const DEEPSEEK_V3_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "self_attn.q_a_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQa),
    ),
    (
        "self_attn.q_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQaNorm),
    ),
    (
        "self_attn.q_b_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionQb),
    ),
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvA),
    ),
    (
        "self_attn.kv_a_layernorm.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvANorm),
    ),
    (
        "self_attn.kv_b_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionKvB),
    ),
    (
        "self_attn.embed_q.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionEmbedQ),
    ),
    (
        "self_attn.unembed_out.weight",
        TensorMapping::PerLayer(NativeTensorRole::AttentionUnembedOut),
    ),
    (
        "mlp.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "mlp.gate.e_score_correction_bias",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInpCorrectionBias),
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
    (
        "mlp.shared_experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "mlp.shared_experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "mlp.shared_experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
    ),
];

/// Per-layer tensor patterns for Mixtral sparse MoE layers.
const MIXTRAL_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "block_sparse_moe.gate.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    (
        "block_sparse_moe.switch_mlp.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "block_sparse_moe.switch_mlp.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "block_sparse_moe.switch_mlp.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
];

/// Per-layer tensor patterns for LLaMA 4 (uses feed_forward.* instead of mlp.*,
/// plus MoE experts and shared expert paths).
const LLAMA4_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    // Dense FFN layers (non-MoE)
    (
        "feed_forward.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGate),
    ),
    (
        "feed_forward.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUp),
    ),
    (
        "feed_forward.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDown),
    ),
    // MoE router
    (
        "feed_forward.router.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateInp),
    ),
    // Packed expert weights (SwitchGLU)
    (
        "feed_forward.experts.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnGateExps),
    ),
    (
        "feed_forward.experts.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnUpExps),
    ),
    (
        "feed_forward.experts.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnDownExps),
    ),
    // Shared expert
    (
        "feed_forward.shared_expert.gate_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertGate),
    ),
    (
        "feed_forward.shared_expert.up_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertUp),
    ),
    (
        "feed_forward.shared_expert.down_proj.weight",
        TensorMapping::PerLayer(NativeTensorRole::FfnSharedExpertDown),
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

const GEMMA4_ASSISTANT_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "pre_projection.weight",
        TensorMapping::Global(NativeTensorRole::AssistantPreProjection),
    ),
    (
        "post_projection.weight",
        TensorMapping::Global(NativeTensorRole::AssistantPostProjection),
    ),
];

/// Extra global tensors for Gemma4 Unified's encoder-free multimodal path.
///
/// This mirrors vLLM's `Gemma4UnifiedVisionEmbedder` plus
/// `Gemma4MultimodalEmbedder` modules:
/// raw patches -> LayerNorm -> Dense -> LayerNorm -> factorized pos emb ->
/// LayerNorm -> multimodal projection.
const GEMMA4_UNIFIED_EXTRA_TENSOR_MAP: &[(&str, TensorMapping)] = &[
    (
        "vision_embedder.patch_dense.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchDense),
    ),
    (
        "vision_embedder.patch_dense.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchDenseBias),
    ),
    (
        "vision_embedder.patch_ln1.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm1),
    ),
    (
        "vision_embedder.patch_ln1.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm1Bias),
    ),
    (
        "vision_embedder.patch_ln2.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm2),
    ),
    (
        "vision_embedder.patch_ln2.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPatchNorm2Bias),
    ),
    (
        "vision_embedder.pos_embedding",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPositionEmbedding),
    ),
    (
        "vision_embedder.pos_norm.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPositionNorm),
    ),
    (
        "vision_embedder.pos_norm.bias",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionPositionNormBias),
    ),
    (
        "embed_vision.embedding_projection.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedVisionProjection),
    ),
    (
        "embed_audio.embedding_projection.weight",
        TensorMapping::Global(NativeTensorRole::Gemma4UnifiedAudioProjection),
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

fn config_has_moe_experts(config: &serde_json::Value, model_type: &str) -> bool {
    arch_u64(config, model_type, "num_experts")
        .or_else(|| arch_u64(config, model_type, "num_local_experts"))
        .or_else(|| arch_u64(config, model_type, "n_routed_experts"))
        .is_some_and(|n| n > 0)
}

fn model_family_for_type(
    model_type: &str,
    config: &serde_json::Value,
) -> Result<ModelFamily, ConvertError> {
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
            extra_tensor_map: if matches!(model_type, "qwen3_5_moe" | "qwen3_5_text")
                || config_has_moe_experts(config, model_type)
            {
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
            family_name: "qwen3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(QWEN3_MOE_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
        }),
        "gemma4" | "gemma4_unified" | "gemma4_unified_text" => Ok(ModelFamily {
            family_name: "gemma4",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: if matches!(model_type, "gemma4_unified" | "gemma4_unified_text") {
                Some(GEMMA4_UNIFIED_EXTRA_TENSOR_MAP)
            } else {
                None
            },
            uses_language_model_prefix: true,
        }),
        "gemma4_assistant" => Ok(ModelFamily {
            family_name: "gemma4_assistant",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(GEMMA4_ASSISTANT_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
        }),
        "diffusion_gemma" => Ok(ModelFamily {
            family_name: "diffusion_gemma",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: true,
        }),
        "glm4_moe_lite" => Ok(ModelFamily {
            family_name: "glm4_moe_lite",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(GLM4_MOE_LITE_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
        }),
        "llama" => Ok(ModelFamily {
            family_name: "llama3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: false,
        }),
        "mistral3" | "ministral3" => Ok(ModelFamily {
            family_name: "mistral3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: None,
            uses_language_model_prefix: false,
        }),
        "mixtral" => Ok(ModelFamily {
            family_name: "mixtral",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(MIXTRAL_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
        }),
        "deepseek_v3" | "deepseek_v32" => Ok(ModelFamily {
            family_name: "deepseek_v3",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(DEEPSEEK_V3_EXTRA_TENSOR_MAP),
            uses_language_model_prefix: false,
        }),
        "llama4" => Ok(ModelFamily {
            family_name: "llama4",
            tensor_map: HF_STANDARD_TENSOR_MAP,
            extra_tensor_map: Some(LLAMA4_EXTRA_TENSOR_MAP),
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
            | "gemma4_unified"
            | "gemma4_unified_text"
            | "gemma4_assistant"
            | "diffusion_gemma"
            | "llama4"
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

fn is_gemma4_target_model_type(model_type: &str) -> bool {
    matches!(
        model_type,
        "gemma4" | "gemma4_unified" | "gemma4_unified_text" | "diffusion_gemma"
    )
}

fn is_gemma4_text_model_type(model_type: &str) -> bool {
    is_gemma4_target_model_type(model_type) || model_type == "gemma4_assistant"
}

fn is_gemma4_unified_model_type(model_type: &str) -> bool {
    matches!(model_type, "gemma4_unified" | "gemma4_unified_text")
}

fn is_qwen_family_model_type(model_type: &str) -> bool {
    model_type.starts_with("qwen3")
}

fn is_glm4_moe_lite(model_type: &str) -> bool {
    model_type == "glm4_moe_lite"
}

/// Parse diffusion-specific config fields from config.json.
///
/// DiffusionGemma may expose these at the top level or nested under a
/// `diffusion_config` key. This helper checks both locations.
fn parse_diffusion_config(config: &serde_json::Value, model_type: &str) -> NativeDiffusionConfig {
    if model_type != "diffusion_gemma" {
        return NativeDiffusionConfig::default();
    }

    let diffusion = config
        .get("diffusion_config")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    let get_u32 = |key: &str| -> Option<u32> {
        diffusion
            .get(key)
            .and_then(|v| v.as_u64())
            .and_then(u64_to_u32)
            .or_else(|| arch_u64(config, model_type, key).and_then(u64_to_u32))
    };

    let get_f32 = |key: &str| -> Option<f32> {
        diffusion
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .or_else(|| arch_f64(config, model_type, key).map(|v| v as f32))
    };

    let get_bool = |key: &str| -> Option<bool> {
        diffusion
            .get(key)
            .and_then(|v| v.as_bool())
            .or_else(|| arch_bool(config, model_type, key))
    };

    NativeDiffusionConfig {
        canvas_size: get_u32("canvas_size"),
        max_denoise_steps: get_u32("max_denoise_steps"),
        self_conditioning: get_bool("self_conditioning"),
        entropy_bound: get_f32("entropy_bound"),
        entropy_threshold: get_f32("entropy_threshold"),
        convergence_steps: get_u32("convergence_steps"),
        temperature_start: get_f32("temperature_start"),
        temperature_end: get_f32("temperature_end"),
    }
}

fn is_mla_family(model_type: &str) -> bool {
    is_glm4_moe_lite(model_type) || matches!(model_type, "deepseek_v3" | "deepseek_v32")
}

fn defaults_attn_output_gate(model_type: &str) -> bool {
    is_qwen3_5_family(model_type) || matches!(model_type, "qwen3_next" | "qwen3_6" | "qwen3.6")
}

fn default_moe_norm_topk_prob(model_type: &str) -> bool {
    is_qwen3_5_family(model_type)
}

fn runtime_status_for_model_type(_model_type: &str) -> NativeRuntimeStatus {
    NativeRuntimeStatus::default()
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

fn parse_rms_norm_eps(config: &serde_json::Value, model_type: &str) -> Option<f32> {
    ["rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon"]
        .into_iter()
        .find_map(|field| arch_f64(config, model_type, field))
        .map(|value| value as f32)
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

fn mla_attention_config(config: &serde_json::Value, model_type: &str) -> NativeMlaAttentionConfig {
    if !is_mla_family(model_type) {
        return NativeMlaAttentionConfig::default();
    }

    NativeMlaAttentionConfig {
        q_lora_rank: arch_u64(config, model_type, "q_lora_rank").and_then(u64_to_u32),
        kv_lora_rank: arch_u64(config, model_type, "kv_lora_rank").and_then(u64_to_u32),
        qk_nope_head_dim: arch_u64(config, model_type, "qk_nope_head_dim").and_then(u64_to_u32),
        qk_rope_head_dim: arch_u64(config, model_type, "qk_rope_head_dim").and_then(u64_to_u32),
        value_head_dim: arch_u64(config, model_type, "v_head_dim").and_then(u64_to_u32),
    }
}

fn glm_router_config(config: &serde_json::Value, model_type: &str) -> NativeGlmRouterConfig {
    if !is_glm4_moe_lite(model_type) {
        return NativeGlmRouterConfig::default();
    }

    NativeGlmRouterConfig {
        first_dense_layer_count: arch_u64(config, model_type, "first_k_dense_replace")
            .and_then(u64_to_u32),
        routed_scaling_factor: arch_f64(config, model_type, "routed_scaling_factor")
            .map(|value| value as f32),
        n_group: arch_u64(config, model_type, "n_group")
            .and_then(u64_to_u32)
            .or(Some(1)),
        topk_group: arch_u64(config, model_type, "topk_group")
            .and_then(u64_to_u32)
            .or(Some(1)),
        has_shared_experts: arch_u64(config, model_type, "n_shared_experts").unwrap_or(0) > 0,
    }
}

fn moe_config(config: &serde_json::Value, model_type: &str) -> NativeMoeConfig {
    let is_gemma4_moe = arch_bool(config, model_type, "enable_moe_block").unwrap_or(false);
    let is_qwen3_moe = matches!(model_type, "qwen3_moe" | "qwen3_5_moe" | "qwen3_5_text")
        || (is_qwen3_5_family(model_type) && config_has_moe_experts(config, model_type));
    let is_qwen3_next_moe = matches!(model_type, "qwen3_next" | "qwen3_6" | "qwen3.6");
    let is_glm_moe = is_glm4_moe_lite(model_type);
    let is_mixtral = model_type == "mixtral";
    let is_deepseek_v3 = matches!(model_type, "deepseek_v3" | "deepseek_v32");
    let is_llama4 = model_type == "llama4";
    if !is_gemma4_moe
        && !is_qwen3_moe
        && !is_qwen3_next_moe
        && !is_glm_moe
        && !is_mixtral
        && !is_deepseek_v3
        && !is_llama4
    {
        return NativeMoeConfig::default();
    }

    let expert_count = arch_u64(config, model_type, "num_experts")
        .or_else(|| arch_u64(config, model_type, "num_local_experts"))
        .or_else(|| arch_u64(config, model_type, "n_routed_experts"))
        .and_then(u64_to_u32);
    let experts_per_token = arch_u64(config, model_type, "top_k_experts")
        .or_else(|| arch_u64(config, model_type, "num_experts_per_tok"))
        .and_then(u64_to_u32);

    let layer_freq = if is_deepseek_v3 {
        arch_u64(config, model_type, "moe_layer_freq").and_then(u64_to_u32)
    } else if is_llama4 {
        arch_u64(config, model_type, "interleave_moe_layer_step").and_then(u64_to_u32)
    } else {
        None
    };

    let first_dense_layers = if is_deepseek_v3 {
        arch_u64(config, model_type, "first_k_dense_replace").and_then(u64_to_u32)
    } else {
        None
    };

    let shared_expert_count = if is_deepseek_v3 {
        arch_u64(config, model_type, "n_shared_experts").and_then(u64_to_u32)
    } else if is_llama4 {
        // LLaMA 4 always has 1 shared expert when MoE is active
        Some(1)
    } else {
        None
    };

    NativeMoeConfig {
        expert_count,
        experts_per_token,
        expert_intermediate_size: arch_u64(config, model_type, "moe_intermediate_size")
            .and_then(u64_to_u32),
        layer_freq,
        first_dense_layers,
        shared_expert_count,
        sigmoid_routing: is_deepseek_v3,
        routed_scaling_factor: if is_deepseek_v3 {
            arch_f64(config, model_type, "routed_scaling_factor").map(|v| v as f32)
        } else {
            None
        },
        n_group: if is_deepseek_v3 {
            arch_u64(config, model_type, "n_group").and_then(u64_to_u32)
        } else {
            None
        },
        topk_group: if is_deepseek_v3 {
            arch_u64(config, model_type, "topk_group").and_then(u64_to_u32)
        } else {
            None
        },
    }
}

/// Parse per-model rope theta (main + SWA) and partial_rotary_factor.
///
/// Returns `(rope_theta, rope_theta_swa, partial_rotary_factor)`.
/// Gemma4 stores these nested under `text_config.rope_parameters.{full,sliding}_attention`.
/// Qwen3.5/Next stores them flat inside `text_config.rope_parameters`.
///
/// The gemma4 assistant drafter (`gemma4_assistant`) carries the identical
/// nested `rope_parameters` layout and must share the target's RoPE geometry,
/// so it takes the same parsing branch — otherwise `rope_theta` / `rope_theta_swa`
/// / `partial_rotary_factor` fall through to None and the drafter's Q rotation
/// stops matching the target's cached K.
fn parse_rope_params(
    config: &serde_json::Value,
    model_type: &str,
) -> (Option<u32>, Option<u32>, Option<f32>) {
    if is_gemma4_text_model_type(model_type) {
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
            .or_else(|| arch_f64(config, model_type, "rope_theta"))
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

/// Parse LLaMA 3 / LLaMA 4 rope_scaling dict from config.json.
#[allow(clippy::type_complexity)]
fn parse_rope_scaling(
    config: &serde_json::Value,
    model_type: &str,
) -> (
    Option<String>,
    Option<f32>,
    Option<f32>,
    Option<f32>,
    Option<u32>,
) {
    let rs = if uses_text_config(model_type) {
        config
            .get("text_config")
            .and_then(|tc| tc.get("rope_scaling"))
    } else {
        config.get("rope_scaling")
    };
    let Some(rs) = rs else {
        return (None, None, None, None, None);
    };
    let rope_type = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let factor = rs.get("factor").and_then(|v| v.as_f64()).map(|f| f as f32);
    let low_freq_factor = rs
        .get("low_freq_factor")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let high_freq_factor = rs
        .get("high_freq_factor")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let original_context_len = rs
        .get("original_max_position_embeddings")
        .and_then(|v| v.as_u64())
        .and_then(u64_to_u32);
    (
        rope_type,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_context_len,
    )
}

/// Parse per-layer type list from config (e.g. Gemma4's `layer_types` field).
/// Gemma4 reference implementations derive it from `sliding_window_pattern`
/// when the field is omitted.
fn parse_layer_types(
    config: &serde_json::Value,
    model_type: &str,
    layer_count: u32,
) -> Vec<String> {
    if !is_gemma4_text_model_type(model_type) {
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
                .map(|v| {
                    v.as_str()
                        .map(String::from)
                        .unwrap_or_else(|| v.to_string())
                })
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
    if !is_gemma4_target_model_type(model_type) || layer_types.is_empty() {
        return BTreeMap::new();
    }
    let default_shared_layers = if model_type == "gemma4" { 20 } else { 0 };
    let num_shared = arch_u64(config, model_type, "num_kv_shared_layers")
        .unwrap_or(default_shared_layers)
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
    // The reference config dataclasses default this field differently per model
    // type: standard gemma4 (27B) defaults False, while gemma4_unified (12B)
    // defaults True. AX reads config.json directly without a dataclass to supply
    // the default, so we mirror the per-type default when the field is absent.
    let default_k_eq_v = is_gemma4_unified_model_type(model_type);
    if !is_gemma4_target_model_type(model_type)
        || !arch_bool(config, model_type, "attention_k_eq_v").unwrap_or(default_k_eq_v)
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
    let attention_head_dim = if is_mla_family(model_type) {
        let qk_nope = require_arch_u64(config, model_type, "qk_nope_head_dim")?;
        let qk_rope = require_arch_u64(config, model_type, "qk_rope_head_dim")?;
        (qk_nope + qk_rope) as u32
    } else {
        arch_u64(config, model_type, "head_dim")
            .map(|v| v as u32)
            .unwrap_or_else(|| hidden_size.checked_div(attention_head_count).unwrap_or(0))
    };
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
    fn converts_gemma4_assistant_q_only_external_kv_contract() {
        let dir = unique_test_dir("gemma4_assistant");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "gemma4_assistant",
                "backbone_hidden_size": 16,
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine"
                },
                "text_config": {
                    "model_type": "gemma4_assistant",
                    "hidden_size": 8,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "num_hidden_layers": 2,
                    "num_kv_shared_layers": 2,
                    "vocab_size": 64,
                    "intermediate_size": 32,
                    "hidden_size_per_layer_input": 0,
                    "vocab_size_per_layer_input": 0,
                    "sliding_window_pattern": 2
                }
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "BF16", &[64, 8]),
                ("model.norm.weight", "BF16", &[8]),
                ("lm_head.weight", "BF16", &[64, 8]),
                ("pre_projection.weight", "U32", &[8, 4]),
                ("post_projection.weight", "U32", &[16, 1]),
                ("model.layers.0.input_layernorm.weight", "BF16", &[8]),
                ("model.layers.0.self_attn.q_proj.weight", "BF16", &[8, 8]),
                ("model.layers.0.self_attn.o_proj.weight", "BF16", &[8, 8]),
                ("model.layers.0.self_attn.q_norm.weight", "BF16", &[4]),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                ("model.layers.0.mlp.gate_proj.weight", "BF16", &[32, 8]),
                ("model.layers.0.mlp.up_proj.weight", "BF16", &[32, 8]),
                ("model.layers.0.mlp.down_proj.weight", "BF16", &[8, 32]),
                ("model.layers.1.input_layernorm.weight", "BF16", &[8]),
                ("model.layers.1.self_attn.q_proj.weight", "BF16", &[8, 8]),
                ("model.layers.1.self_attn.o_proj.weight", "BF16", &[8, 8]),
                ("model.layers.1.self_attn.q_norm.weight", "BF16", &[4]),
                (
                    "model.layers.1.post_attention_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                ("model.layers.1.mlp.gate_proj.weight", "BF16", &[32, 8]),
                ("model.layers.1.mlp.up_proj.weight", "BF16", &[32, 8]),
                ("model.layers.1.mlp.down_proj.weight", "BF16", &[8, 32]),
            ],
        );

        let manifest =
            convert_hf_model_dir(&dir).expect("Gemma4 assistant conversion should succeed");
        assert_eq!(manifest.model_family, "gemma4_assistant");
        assert_eq!(manifest.hidden_size_per_layer_input, 0);
        assert_eq!(manifest.vocab_size_per_layer_input, None);
        assert_eq!(
            manifest.layer_types,
            vec![
                "sliding_attention".to_string(),
                "full_attention".to_string()
            ]
        );
        assert!(manifest.kv_shared_source_layers.is_empty());
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::AssistantPreProjection)
        );
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::AssistantPostProjection)
        );
        assert!(!manifest.tensors.iter().any(|tensor| matches!(
            tensor.role,
            NativeTensorRole::AttentionK | NativeTensorRole::AttentionV
        )));
        crate::model::NativeModelArtifacts::from_manifest_and_root(dir, manifest)
            .expect("Gemma4 assistant manifest should validate");
    }

    #[test]
    fn gemma4_moe_expert_names_map_to_unambiguous_roles() {
        let empty_config = serde_json::json!({});
        let family =
            model_family_for_type("gemma4", &empty_config).expect("gemma4 should be supported");

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
    fn parses_gemma4_sliding_rope_theta_from_flat_fallback() {
        let config = serde_json::json!({
            "model_type": "gemma4",
            "text_config": {
                "rope_theta": 1000000
            }
        });

        let (full_theta, sliding_theta, partial_rotary) = parse_rope_params(&config, "gemma4");

        assert_eq!(full_theta, Some(1000000));
        assert_eq!(sliding_theta, Some(1000000));
        assert_eq!(partial_rotary, None);
    }

    #[test]
    fn parses_gemma4_assistant_nested_rope_like_gemma4() {
        // The assistant drafter carries the identical nested rope_parameters
        // layout and must take the gemma4 branch — otherwise its Q rotation
        // stops matching the target's cached K and the draft accept rate
        // collapses (~20%). Mirrors the real assistant config.json shape.
        let config = serde_json::json!({
            "model_type": "gemma4_assistant",
            "text_config": {
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1000000,
                        "partial_rotary_factor": 0.25,
                    },
                    "sliding_attention": {
                        "rope_theta": 10000,
                    },
                }
            }
        });

        let (full_theta, sliding_theta, partial_rotary) =
            parse_rope_params(&config, "gemma4_assistant");

        assert_eq!(full_theta, Some(1000000));
        assert_eq!(sliding_theta, Some(10000));
        assert_eq!(partial_rotary, Some(0.25));
    }

    #[test]
    fn rejects_gemma4_layer_types_length_mismatch_at_conversion() {
        let dir = unique_test_dir("gemma4_bad_layer_types");
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
                    "layer_types": ["sliding_attention"]
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

        let error = convert_hf_model_dir(&dir).expect_err("bad layer_types should fail closed");
        let ConvertError::InvalidModelContract {
            model_type,
            message,
        } = error
        else {
            panic!("expected invalid model contract");
        };
        assert_eq!(model_type, "gemma4");
        assert!(message.contains("layer_types"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_gemma4_per_layer_input_missing_weights_at_conversion() {
        let dir = unique_test_dir("gemma4_missing_per_layer_input");
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
                    "hidden_size_per_layer_input": 64,
                    "layer_types": ["sliding_attention", "sliding_attention"]
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

        let error =
            convert_hf_model_dir(&dir).expect_err("missing per-layer inputs should fail closed");
        let ConvertError::InvalidModelContract {
            model_type,
            message,
        } = error
        else {
            panic!("expected invalid model contract");
        };
        assert_eq!(model_type, "gemma4");
        assert!(message.contains("per-layer input"));

        let _ = fs::remove_dir_all(dir);
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
    fn converts_gemma4_unified_text_without_tower_tensors() {
        let dir = unique_test_dir("gemma4_unified_text");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "gemma4_unified",
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "vocab_size": 262144,
                "tie_word_embeddings": false,
                "text_config": {
                    "model_type": "gemma4_unified_text",
                    "hidden_size": 3072,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 4,
                    "head_dim": 128,
                    "global_head_dim": 256,
                    "sliding_window": 1024,
                    "attention_k_eq_v": true,
                    "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 0,
                    "vocab_size_per_layer_input": 262144,
                    "final_logit_softcapping": 30.0,
                    "num_hidden_layers": 2,
                    "vocab_size": 262144,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "rope_parameters": {
                        "full_attention": {
                            "rope_theta": 1000000,
                            "partial_rotary_factor": 0.25
                        },
                        "sliding_attention": {
                            "rope_theta": 10000
                        }
                    }
                },
                "vision_config": {
                    "model_type": "gemma4_unified_vision"
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
                    &[262144, 3072],
                ),
                ("language_model.model.norm.weight", "BF16", &[3072]),
                ("language_model.lm_head.weight", "BF16", &[262144, 3072]),
                (
                    "language_model.model.layers.0.input_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.0.self_attn.q_proj.weight",
                    "BF16",
                    &[4096, 3072],
                ),
                (
                    "language_model.model.layers.0.self_attn.k_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "language_model.model.layers.0.self_attn.v_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "language_model.model.layers.0.self_attn.o_proj.weight",
                    "BF16",
                    &[3072, 4096],
                ),
                (
                    "language_model.model.layers.0.self_attn.q_norm.weight",
                    "BF16",
                    &[128],
                ),
                (
                    "language_model.model.layers.0.self_attn.k_norm.weight",
                    "BF16",
                    &[128],
                ),
                (
                    "language_model.model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.0.pre_feedforward_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.0.post_feedforward_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.0.mlp.gate_proj.weight",
                    "BF16",
                    &[12288, 3072],
                ),
                (
                    "language_model.model.layers.0.mlp.up_proj.weight",
                    "BF16",
                    &[12288, 3072],
                ),
                (
                    "language_model.model.layers.0.mlp.down_proj.weight",
                    "BF16",
                    &[3072, 12288],
                ),
                (
                    "language_model.model.layers.1.input_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.1.self_attn.q_proj.weight",
                    "BF16",
                    &[8192, 3072],
                ),
                (
                    "language_model.model.layers.1.self_attn.k_proj.weight",
                    "BF16",
                    &[1024, 3072],
                ),
                (
                    "language_model.model.layers.1.self_attn.o_proj.weight",
                    "BF16",
                    &[3072, 8192],
                ),
                (
                    "language_model.model.layers.1.self_attn.q_norm.weight",
                    "BF16",
                    &[256],
                ),
                (
                    "language_model.model.layers.1.self_attn.k_norm.weight",
                    "BF16",
                    &[256],
                ),
                (
                    "language_model.model.layers.1.post_attention_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.1.pre_feedforward_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.1.post_feedforward_layernorm.weight",
                    "BF16",
                    &[3072],
                ),
                (
                    "language_model.model.layers.1.mlp.gate_proj.weight",
                    "BF16",
                    &[12288, 3072],
                ),
                (
                    "language_model.model.layers.1.mlp.up_proj.weight",
                    "BF16",
                    &[12288, 3072],
                ),
                (
                    "language_model.model.layers.1.mlp.down_proj.weight",
                    "BF16",
                    &[3072, 12288],
                ),
                // Unified multimodal tensors are global projector roles, not towers.
                ("vision_embedder.pos_embedding", "BF16", &[1120, 2, 3072]),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("unified text conversion should succeed");

        assert_eq!(manifest.model_family, "gemma4");
        assert_eq!(manifest.hidden_size, 3072);
        assert_eq!(manifest.rope_theta, Some(1000000));
        assert_eq!(manifest.rope_theta_swa, Some(10000));
        assert_eq!(manifest.partial_rotary_factor, Some(0.25));
        assert_eq!(manifest.global_head_dim, Some(256));
        assert_eq!(manifest.sliding_window_size, Some(1024));
        assert_eq!(manifest.final_logit_softcapping, Some(30.0));
        assert_eq!(manifest.hidden_states_scale, Some((3072_f32).sqrt()));
        assert_eq!(manifest.hidden_size_per_layer_input, 0);
        assert_eq!(manifest.vocab_size_per_layer_input, None);
        assert_eq!(
            manifest.layer_types,
            vec![
                "sliding_attention".to_string(),
                "full_attention".to_string()
            ]
        );
        assert!(manifest.kv_shared_source_layers.is_empty());
        assert_eq!(manifest.attention_value_from_key_layers, vec![1]);
        assert_eq!(manifest.attention_v_norm_no_scale_layers, vec![0, 1]);
        assert!(
            manifest.tensors.iter().any(|tensor| {
                tensor.role == NativeTensorRole::Gemma4UnifiedVisionPositionEmbedding
                    && tensor.name == "vision_embedder.pos_embedding"
            }),
            "unified projector tensors should be mapped for multimodal runtime support"
        );
        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("Gemma4 unified text manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn attention_k_eq_v_default_is_model_type_aware_when_field_absent() {
        use std::collections::BTreeMap;

        let layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        let no_shared = BTreeMap::new();
        // Field omitted from config: the dataclass defaults differ by type, so
        // gemma4_unified must default True (full-attention layer 1 uses V-from-K)
        // while standard gemma4 must default False (no V-from-K layers).
        let empty = serde_json::json!({});
        assert_eq!(
            compute_attention_value_from_key_layers(
                &empty,
                "gemma4_unified_text",
                &layer_types,
                &no_shared,
                2,
            ),
            vec![1],
        );
        assert_eq!(
            compute_attention_value_from_key_layers(
                &empty,
                "gemma4_unified",
                &layer_types,
                &no_shared,
                2,
            ),
            vec![1],
        );
        assert!(
            compute_attention_value_from_key_layers(&empty, "gemma4", &layer_types, &no_shared, 2,)
                .is_empty(),
        );
        // An explicit value still overrides the per-type default.
        let disabled = serde_json::json!({ "attention_k_eq_v": false });
        assert!(
            compute_attention_value_from_key_layers(
                &disabled,
                "gemma4_unified_text",
                &layer_types,
                &no_shared,
                2,
            )
            .is_empty(),
        );
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

        let manifest = convert_hf_model_dir(&dir).expect("qwen3.5 MoE conversion should succeed");

        assert_eq!(manifest.model_family, "qwen3_5");
        assert_eq!(manifest.moe.expert_count, Some(4));
        assert_eq!(manifest.moe.experts_per_token, Some(2));
        assert_eq!(manifest.moe.expert_intermediate_size, Some(8));
        assert!(
            manifest.moe_norm_topk_prob,
            "Qwen3.5 MoE defaults norm_topk_prob=true in mlx_lm"
        );
        assert!(
            manifest.attn_output_gate,
            "Qwen3.5 MoE full-attention layers must default to the reference output gate"
        );
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
        let empty_config = serde_json::json!({});
        let family = model_family_for_type("qwen3_moe", &empty_config)
            .expect("qwen3_moe should be supported");

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
    fn qwen3_5_model_type_activates_moe_tensor_map_when_config_has_experts() {
        let moe_config = serde_json::json!({
            "text_config": {
                "num_experts": 128,
                "num_experts_per_tok": 8
            }
        });
        let family =
            model_family_for_type("qwen3_5", &moe_config).expect("qwen3_5 should be supported");
        assert_eq!(family.family_name, "qwen3_5");
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
    fn qwen3_5_model_type_without_moe_config_has_no_moe_tensors() {
        let dense_config = serde_json::json!({
            "text_config": {
                "hidden_size": 64
            }
        });
        let family =
            model_family_for_type("qwen3_5", &dense_config).expect("qwen3_5 should be supported");
        assert_eq!(family.family_name, "qwen3_5");
        assert_eq!(
            match_tensor("model.layers.2.mlp.gate.weight", &family),
            None
        );
        assert_eq!(
            match_tensor("model.layers.2.mlp.switch_mlp.gate_proj.weight", &family),
            None
        );
    }

    #[test]
    fn qwen3_5_moe_config_detected_from_num_experts_in_text_config() {
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 256
            }
        });
        let moe = moe_config(&config, "qwen3_5");
        assert_eq!(moe.expert_count, Some(128));
        assert_eq!(moe.experts_per_token, Some(8));
        assert_eq!(moe.expert_intermediate_size, Some(256));
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
        assert!(
            manifest.attn_output_gate,
            "qwen3_next attn_output_gate must default to true"
        );
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
    fn qwen3_6_alias_produces_moe_config() {
        // Regression test for B1: moe_config() previously checked only
        // `model_type == "qwen3_next"` and missed the "qwen3.6" / "qwen3_6"
        // aliases. A checkpoint using either alias must still get MoE config.
        for alias in ["qwen3.6", "qwen3_6"] {
            let dir = unique_test_dir(&format!("qwen3_6_alias_{}", alias.replace('.', "_")));
            write_config(
                &dir,
                serde_json::json!({
                    "model_type": alias,
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
                    "num_experts": 8,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 8,
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
                    ("model.layers.0.mlp.gate.weight", "BF16", &[8, 64]),
                    (
                        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                        "BF16",
                        &[8, 8, 64],
                    ),
                    (
                        "model.layers.0.mlp.switch_mlp.up_proj.weight",
                        "BF16",
                        &[8, 8, 64],
                    ),
                    (
                        "model.layers.0.mlp.switch_mlp.down_proj.weight",
                        "BF16",
                        &[8, 64, 8],
                    ),
                ],
            );
            let manifest = convert_hf_model_dir(&dir)
                .unwrap_or_else(|e| panic!("convert with alias '{alias}' failed: {e}"));
            assert_eq!(
                manifest.moe.expert_count,
                Some(8),
                "alias '{alias}' must produce MoE config with expert_count=8"
            );
            assert_eq!(manifest.model_family, "qwen3_next");
            let _ = fs::remove_dir_all(dir);
        }
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
    fn converts_glm4_moe_lite_to_draft_manifest_with_mla_roles() {
        let dir = unique_test_dir("glm4_moe_lite");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "glm4_moe_lite",
                "hidden_size": 2048,
                "intermediate_size": 10240,
                "num_attention_heads": 20,
                "num_key_value_heads": 20,
                "num_hidden_layers": 2,
                "vocab_size": 154880,
                "qk_nope_head_dim": 192,
                "qk_rope_head_dim": 64,
                "v_head_dim": 256,
                "q_lora_rank": 768,
                "kv_lora_rank": 512,
                "first_k_dense_replace": 1,
                "n_routed_experts": 64,
                "n_shared_experts": 1,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 1536,
                "routed_scaling_factor": 1.8,
                "norm_topk_prob": true,
                "rope_theta": 1000000,
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
                ("model.embed_tokens.weight", "BF16", &[154880, 2048]),
                ("model.norm.weight", "BF16", &[2048]),
                ("lm_head.weight", "BF16", &[154880, 2048]),
                ("model.layers.0.input_layernorm.weight", "BF16", &[2048]),
                (
                    "model.layers.0.self_attn.q_a_proj.weight",
                    "U32",
                    &[768, 256],
                ),
                (
                    "model.layers.0.self_attn.q_a_layernorm.weight",
                    "BF16",
                    &[768],
                ),
                (
                    "model.layers.0.self_attn.q_b_proj.weight",
                    "U32",
                    &[5120, 96],
                ),
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    "U32",
                    &[576, 256],
                ),
                (
                    "model.layers.0.self_attn.kv_a_layernorm.weight",
                    "BF16",
                    &[512],
                ),
                (
                    "model.layers.0.self_attn.embed_q.weight",
                    "U32",
                    &[20, 512, 24],
                ),
                (
                    "model.layers.0.self_attn.unembed_out.weight",
                    "U32",
                    &[20, 256, 64],
                ),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    "U32",
                    &[2048, 640],
                ),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[2048],
                ),
                ("model.layers.0.mlp.gate_proj.weight", "U32", &[10240, 256]),
                ("model.layers.0.mlp.up_proj.weight", "U32", &[10240, 256]),
                ("model.layers.0.mlp.down_proj.weight", "U32", &[2048, 1280]),
                ("model.layers.1.input_layernorm.weight", "BF16", &[2048]),
                (
                    "model.layers.1.self_attn.q_a_proj.weight",
                    "U32",
                    &[768, 256],
                ),
                (
                    "model.layers.1.self_attn.q_a_layernorm.weight",
                    "BF16",
                    &[768],
                ),
                (
                    "model.layers.1.self_attn.q_b_proj.weight",
                    "U32",
                    &[5120, 96],
                ),
                (
                    "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                    "U32",
                    &[576, 256],
                ),
                (
                    "model.layers.1.self_attn.kv_a_layernorm.weight",
                    "BF16",
                    &[512],
                ),
                (
                    "model.layers.1.self_attn.embed_q.weight",
                    "U32",
                    &[20, 512, 24],
                ),
                (
                    "model.layers.1.self_attn.unembed_out.weight",
                    "U32",
                    &[20, 256, 64],
                ),
                (
                    "model.layers.1.self_attn.o_proj.weight",
                    "U32",
                    &[2048, 640],
                ),
                (
                    "model.layers.1.post_attention_layernorm.weight",
                    "BF16",
                    &[2048],
                ),
                ("model.layers.1.mlp.gate.weight", "BF16", &[64, 2048]),
                (
                    "model.layers.1.mlp.gate.e_score_correction_bias",
                    "BF16",
                    &[64],
                ),
                (
                    "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                    "U32",
                    &[64, 1536, 256],
                ),
                (
                    "model.layers.1.mlp.switch_mlp.up_proj.weight",
                    "U32",
                    &[64, 1536, 256],
                ),
                (
                    "model.layers.1.mlp.switch_mlp.down_proj.weight",
                    "U32",
                    &[64, 2048, 192],
                ),
                (
                    "model.layers.1.mlp.shared_experts.gate_proj.weight",
                    "U32",
                    &[1536, 256],
                ),
                (
                    "model.layers.1.mlp.shared_experts.up_proj.weight",
                    "U32",
                    &[1536, 256],
                ),
                (
                    "model.layers.1.mlp.shared_experts.down_proj.weight",
                    "U32",
                    &[2048, 192],
                ),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("GLM conversion should succeed");

        assert_eq!(manifest.model_family, "glm4_moe_lite");
        assert_eq!(manifest.attention_head_dim, 256);
        assert_eq!(manifest.mla_attention.q_lora_rank, Some(768));
        assert_eq!(manifest.mla_attention.kv_lora_rank, Some(512));
        assert_eq!(manifest.mla_attention.qk_nope_head_dim, Some(192));
        assert_eq!(manifest.mla_attention.qk_rope_head_dim, Some(64));
        assert_eq!(manifest.mla_attention.value_head_dim, Some(256));
        assert_eq!(manifest.moe.expert_count, Some(64));
        assert_eq!(manifest.moe.experts_per_token, Some(4));
        assert_eq!(manifest.moe.expert_intermediate_size, Some(1536));
        assert_eq!(manifest.glm_router.first_dense_layer_count, Some(1));
        assert_eq!(manifest.glm_router.routed_scaling_factor, Some(1.8));
        assert_eq!(manifest.glm_router.n_group, Some(1));
        assert_eq!(manifest.glm_router.topk_group, Some(1));
        assert!(manifest.glm_router.has_shared_experts);
        assert!(manifest.moe_norm_topk_prob);
        assert!(manifest.runtime_status.ready);
        assert!(manifest.runtime_status.blockers.is_empty());

        for role in [
            NativeTensorRole::AttentionQa,
            NativeTensorRole::AttentionQaNorm,
            NativeTensorRole::AttentionQb,
            NativeTensorRole::AttentionKvA,
            NativeTensorRole::AttentionKvANorm,
            NativeTensorRole::AttentionEmbedQ,
            NativeTensorRole::AttentionUnembedOut,
            NativeTensorRole::FfnGateInpCorrectionBias,
            NativeTensorRole::FfnSharedExpertGate,
        ] {
            assert!(
                manifest.tensors.iter().any(|tensor| tensor.role == role),
                "GLM manifest should map {role:?}"
            );
        }

        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("runtime-ready GLM manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn converts_deepseek_v3_raw_kv_b_projection_to_mla_manifest() {
        let dir = unique_test_dir("deepseek_v3");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "deepseek_v3",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 64,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 2,
                "v_head_dim": 3,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "n_routed_experts": 3,
                "n_shared_experts": 1,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 5,
                "routed_scaling_factor": 2.5,
                "n_group": 1,
                "topk_group": 1,
                "norm_topk_prob": true,
                "rope_theta": 1000000
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "BF16", &[64, 16]),
                ("model.norm.weight", "BF16", &[16]),
                ("lm_head.weight", "BF16", &[64, 16]),
                ("model.layers.0.input_layernorm.weight", "BF16", &[16]),
                ("model.layers.0.self_attn.q_a_proj.weight", "BF16", &[4, 16]),
                (
                    "model.layers.0.self_attn.q_a_layernorm.weight",
                    "BF16",
                    &[4],
                ),
                ("model.layers.0.self_attn.q_b_proj.weight", "BF16", &[12, 4]),
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    "BF16",
                    &[6, 16],
                ),
                (
                    "model.layers.0.self_attn.kv_a_layernorm.weight",
                    "BF16",
                    &[4],
                ),
                (
                    "model.layers.0.self_attn.kv_b_proj.weight",
                    "BF16",
                    &[14, 4],
                ),
                ("model.layers.0.self_attn.o_proj.weight", "BF16", &[16, 6]),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[16],
                ),
                ("model.layers.0.mlp.gate_proj.weight", "BF16", &[32, 16]),
                ("model.layers.0.mlp.up_proj.weight", "BF16", &[32, 16]),
                ("model.layers.0.mlp.down_proj.weight", "BF16", &[16, 32]),
                ("model.layers.1.input_layernorm.weight", "BF16", &[16]),
                ("model.layers.1.self_attn.q_a_proj.weight", "BF16", &[4, 16]),
                (
                    "model.layers.1.self_attn.q_a_layernorm.weight",
                    "BF16",
                    &[4],
                ),
                ("model.layers.1.self_attn.q_b_proj.weight", "BF16", &[12, 4]),
                (
                    "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                    "BF16",
                    &[6, 16],
                ),
                (
                    "model.layers.1.self_attn.kv_a_layernorm.weight",
                    "BF16",
                    &[4],
                ),
                (
                    "model.layers.1.self_attn.kv_b_proj.weight",
                    "BF16",
                    &[14, 4],
                ),
                ("model.layers.1.self_attn.o_proj.weight", "BF16", &[16, 6]),
                (
                    "model.layers.1.post_attention_layernorm.weight",
                    "BF16",
                    &[16],
                ),
                ("model.layers.1.mlp.gate.weight", "BF16", &[3, 16]),
                (
                    "model.layers.1.mlp.gate.e_score_correction_bias",
                    "BF16",
                    &[3],
                ),
                (
                    "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                    "BF16",
                    &[3, 5, 16],
                ),
                (
                    "model.layers.1.mlp.switch_mlp.up_proj.weight",
                    "BF16",
                    &[3, 5, 16],
                ),
                (
                    "model.layers.1.mlp.switch_mlp.down_proj.weight",
                    "BF16",
                    &[3, 16, 5],
                ),
                (
                    "model.layers.1.mlp.shared_experts.gate_proj.weight",
                    "BF16",
                    &[5, 16],
                ),
                (
                    "model.layers.1.mlp.shared_experts.up_proj.weight",
                    "BF16",
                    &[5, 16],
                ),
                (
                    "model.layers.1.mlp.shared_experts.down_proj.weight",
                    "BF16",
                    &[16, 5],
                ),
            ],
        );

        let manifest = convert_hf_model_dir(&dir).expect("DeepSeek V3 conversion should succeed");

        assert_eq!(manifest.model_family, "deepseek_v3");
        assert_eq!(manifest.attention_head_dim, 6);
        assert!(
            manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::AttentionKvB),
            "raw DeepSeek kv_b_proj should be preserved in the manifest"
        );
        assert!(
            !manifest
                .tensors
                .iter()
                .any(|tensor| tensor.role == NativeTensorRole::AttentionEmbedQ),
            "raw DeepSeek kv_b_proj should not be misreported as embed_q"
        );

        write_manifest(&dir, &manifest).expect("write should succeed");
        crate::model::NativeModelArtifacts::from_dir(&dir)
            .expect("runtime-ready DeepSeek manifest should validate");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_glm4_moe_lite_rope_scaling_until_scale_contract_is_manifested() {
        validate_glm4_moe_lite_rope_scaling(&serde_json::json!({}))
            .expect("missing rope_scaling should use current GLM scale contract");
        validate_glm4_moe_lite_rope_scaling(&serde_json::json!({ "rope_scaling": null }))
            .expect("null rope_scaling should use current GLM scale contract");

        let error = validate_glm4_moe_lite_rope_scaling(&serde_json::json!({
            "rope_scaling": {
                "factor": 2.0,
                "mscale_all_dim": 1.0
            }
        }))
        .expect_err("GLM rope scaling should fail closed until represented in the manifest");
        let ConvertError::InvalidModelContract {
            model_type,
            message,
        } = error
        else {
            panic!("expected invalid model contract");
        };
        assert_eq!(model_type, "glm4_moe_lite");
        assert!(message.contains("rope_scaling"));
        assert!(message.contains("mscale_all_dim"));
    }

    #[test]
    fn rejects_qwen_rope_scaling_until_runtime_contract_is_manifested() {
        validate_qwen_rope_scaling(&serde_json::json!({}), "qwen3_next")
            .expect("missing rope_scaling should use current Qwen RoPE contract");
        validate_qwen_rope_scaling(
            &serde_json::json!({ "text_config": { "rope_scaling": null } }),
            "qwen3_next",
        )
        .expect("null rope_scaling should use current Qwen RoPE contract");

        let error = validate_qwen_rope_scaling(
            &serde_json::json!({
                "text_config": {
                    "rope_scaling": {
                        "type": "longrope",
                        "factor": 4.0
                    }
                }
            }),
            "qwen3_next",
        )
        .expect_err("Qwen rope scaling should fail closed until represented in the manifest");
        let ConvertError::InvalidModelContract {
            model_type,
            message,
        } = error
        else {
            panic!("expected invalid model contract");
        };
        assert_eq!(model_type, "qwen3_next");
        assert!(message.contains("rope_scaling"));
        assert!(message.contains("Qwen MLX runtime"));
    }

    #[test]
    fn rejects_glm4_moe_lite_draft_manifest_when_router_bias_is_missing() {
        let dir = unique_test_dir("glm4_missing_router_bias");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "glm4_moe_lite",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 1,
                "vocab_size": 64,
                "qk_nope_head_dim": 6,
                "qk_rope_head_dim": 2,
                "v_head_dim": 8,
                "q_lora_rank": 8,
                "kv_lora_rank": 8,
                "first_k_dense_replace": 0,
                "n_routed_experts": 4,
                "n_shared_experts": 0,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "routed_scaling_factor": 1.8
            }),
        );
        write_fake_safetensors(
            &dir,
            "model.safetensors",
            &[
                ("model.embed_tokens.weight", "BF16", &[64, 16]),
                ("model.norm.weight", "BF16", &[16]),
                ("lm_head.weight", "BF16", &[64, 16]),
                ("model.layers.0.input_layernorm.weight", "BF16", &[16]),
                ("model.layers.0.self_attn.q_a_proj.weight", "BF16", &[8, 16]),
                (
                    "model.layers.0.self_attn.q_a_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                ("model.layers.0.self_attn.q_b_proj.weight", "BF16", &[16, 8]),
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    "BF16",
                    &[10, 16],
                ),
                (
                    "model.layers.0.self_attn.kv_a_layernorm.weight",
                    "BF16",
                    &[8],
                ),
                ("model.layers.0.self_attn.embed_q.weight", "BF16", &[12, 8]),
                (
                    "model.layers.0.self_attn.unembed_out.weight",
                    "BF16",
                    &[16, 8],
                ),
                ("model.layers.0.self_attn.o_proj.weight", "BF16", &[16, 16]),
                (
                    "model.layers.0.post_attention_layernorm.weight",
                    "BF16",
                    &[16],
                ),
                ("model.layers.0.mlp.gate.weight", "BF16", &[4, 16]),
                (
                    "model.layers.0.mlp.switch_mlp.gate_proj.weight",
                    "BF16",
                    &[4, 8, 16],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.up_proj.weight",
                    "BF16",
                    &[4, 8, 16],
                ),
                (
                    "model.layers.0.mlp.switch_mlp.down_proj.weight",
                    "BF16",
                    &[4, 16, 8],
                ),
            ],
        );

        let error = convert_hf_model_dir(&dir)
            .expect_err("GLM MoE layer without correction bias should fail conversion");

        assert!(
            error.to_string().contains("FfnGateInpCorrectionBias"),
            "{error}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_unsupported_model_type() {
        let dir = unique_test_dir("unsupported");
        write_config(
            &dir,
            serde_json::json!({
                "model_type": "gpt2",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_hidden_layers": 2,
                "vocab_size": 32000,
            }),
        );
        write_fake_safetensors(&dir, "model.safetensors", &[]);

        let error = convert_hf_model_dir(&dir).expect_err("gpt2 should be unsupported");
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
