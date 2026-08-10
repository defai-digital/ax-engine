//! Convert HuggingFace / MLX model directories to ax-engine native model manifests.
//!
//! Reads `config.json` and safetensors headers from a model directory and produces
//! a `NativeModelManifest` that can be written as `model-manifest.json`. Most
//! families are metadata-only — the manifest points directly at the original
//! safetensors files. The one exception is DeepSeek V4: quantized (FP8 dense +
//! FP4 expert) checkpoints are dequantized/repacked into a generated
//! safetensors file by `deepseek_v4_quantized` before mapping.
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

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;

use crate::model::{
    AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, DroppedTensorsProvenance, KvCacheQuantizationManifest,
    NativeDeepseekV4AttentionConfig, NativeDeepseekV4Config, NativeDiffusionConfig,
    NativeGlmRouterConfig, NativeLinearAttentionConfig, NativeMlaAttentionConfig,
    NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType,
    NativeTensorFormat, NativeTensorQuantization, NativeTensorRole, NativeTensorSpec,
    WeightSanitize,
};

/// Env: when set to `1`/`true`/`on`, convert hard-errors if any tensors are dropped.
pub const AX_CONVERT_STRICT_TENSORS: &str = "AX_CONVERT_STRICT_TENSORS";
/// Env: when not explicitly off, emit dropped-tensor warnings (default on).
pub const AX_CONVERT_DROPPED_TENSOR_REPORT: &str = "AX_CONVERT_DROPPED_TENSOR_REPORT";

const MANIFEST_TEMP_FILE_PREFIX: &str = ".ax-engine-manifest.tmp-";
const MANIFEST_TEMP_CREATE_ATTEMPTS: usize = 128;
static MANIFEST_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Case-insensitive substrings that indicate media-role weights (WS-C1).
const MEDIA_ROLE_NAME_MARKERS: &[&str] = &[
    "vision_tower",
    "vision",
    "visual",
    "vit",
    "image_newline",
    "projector",
    "multi_modal",
    "audio_tower",
    "conformer",
    "sam.",
    "clip.",
];

/// Ledger for tensors skipped during convert mapping (WS-C1 / R-C1).
#[derive(Clone, Debug, Default)]
pub struct DroppedTensorLedger {
    pub count: u64,
    pub bytes: u64,
    pub media_role_hits: u64,
    pub names_sample: Vec<String>,
}

impl DroppedTensorLedger {
    const SAMPLE_CAP: usize = 16;

    pub fn record(&mut self, name: &str, length_bytes: u64) {
        self.count = self.count.saturating_add(1);
        self.bytes = self.bytes.saturating_add(length_bytes);
        if tensor_name_looks_like_media_role(name) {
            self.media_role_hits = self.media_role_hits.saturating_add(1);
        }
        if self.names_sample.len() < Self::SAMPLE_CAP {
            self.names_sample.push(name.to_string());
        }
    }

    pub fn to_provenance(&self) -> DroppedTensorsProvenance {
        DroppedTensorsProvenance {
            count: self.count,
            media_role_hits: self.media_role_hits,
            names_sample: self.names_sample.clone(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Returns true when a tensor name matches media-role heuristics.
pub fn tensor_name_looks_like_media_role(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    MEDIA_ROLE_NAME_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

fn convert_report_enabled() -> bool {
    match std::env::var(AX_CONVERT_DROPPED_TENSOR_REPORT) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        }
        Err(_) => true,
    }
}

fn convert_strict_tensors() -> bool {
    match std::env::var(AX_CONVERT_STRICT_TENSORS) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on" || v == "yes"
        }
        Err(_) => false,
    }
}

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
        "unsupported model type {model_type}; supported: qwen3, qwen3_5, qwen3_next, qwen3_vl, qwen3_vl_moe, minicpmv4_6, gemma4, gemma4_unified, gemma4_vl, gemma4_assistant, diffusion_gemma, embeddinggemma, glm4_moe_lite, llama, llama3, mistral, mistral3, mixtral, deepseek_v3, deepseek_v32, deepseek_v4, llama4, gpt_oss, nemotron_h, nemotron_h_nano_omni, nemotron_embed, unlimited_ocr, whisper"
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
    #[error("generated manifest for {dir} failed loader validation and was not written: {message}")]
    GeneratedManifestInvalid { dir: PathBuf, message: String },
    #[error(
        "convert dropped {count} unrecognised tensor(s) ({media_role_hits} media-role); \
set {strict_env}=0 to allow, or map the tensors (sample: {sample})"
    )]
    DroppedTensorsStrict {
        count: u64,
        media_role_hits: u64,
        sample: String,
        strict_env: &'static str,
    },
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
    // DeepSeek V4 quantized checkpoints (FP8 dense weights + FP4 routed
    // experts) need a real data pass: dequantize/repack into a generated
    // safetensors file and consume the quantized sources + scale sidecars so
    // they never reach the manifest.
    let converted_v4 = if is_deepseek_v4(&model_type) {
        convert_deepseek_v4_quantized_tensors(
            model_dir,
            &model_type,
            &config,
            &family,
            &all_tensors,
        )?
    } else {
        None
    };
    let (mut mapped_tensors, dropped_ledger) = map_tensors(
        &config,
        &all_tensors,
        &family,
        converted_v4.as_ref().map(|converted| &converted.consumed),
    )?;
    if is_deepseek_v4(&model_type) {
        // AXQ/mlx-lm affine weights are `X.weight` + `X.scales` + `X.biases`
        // triplets; only `X.weight` enters the manifest and the runtime
        // resolves the sidecars by name, so a missing sidecar must fail here.
        // Runs before the converted-spec merge: FP8/FP4-repacked tensors
        // carry their scales inside the generated safetensors file instead.
        validate_deepseek_v4_quantized_triplets(&mapped_tensors, &all_tensors)?;
    }
    if let Some(converted) = converted_v4 {
        mapped_tensors.extend(converted.specs);
        mapped_tensors.sort_by_key(|spec| (spec.layer_index, format!("{:?}", spec.role)));
    }
    if !dropped_ledger.is_empty() && convert_report_enabled() {
        tracing::warn!(
            target = "ax_engine_core::convert",
            count = dropped_ledger.count,
            bytes = dropped_ledger.bytes,
            media_role_hits = dropped_ledger.media_role_hits,
            names_sample = ?dropped_ledger.names_sample,
            "convert_dropped_unrecognised_tensors"
        );
    }
    if !dropped_ledger.is_empty() && convert_strict_tensors() {
        return Err(ConvertError::DroppedTensorsStrict {
            count: dropped_ledger.count,
            media_role_hits: dropped_ledger.media_role_hits,
            sample: dropped_ledger.names_sample.join(", "),
            strict_env: AX_CONVERT_STRICT_TENSORS,
        });
    }

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
        rope_beta_fast,
        rope_beta_slow,
    ) = parse_rope_scaling(&config, &model_type);

    let query_pre_attn_scalar =
        arch_f64(&config, &model_type, "query_pre_attn_scalar").and_then(f64_to_u32);

    let attention_logit_softcap =
        arch_f64(&config, &model_type, "attn_logit_softcapping").and_then(f64_to_u32);
    // Unlimited-OCR language tower is DeepSeek-V2 with default rms_norm_eps=1e-6.
    // HF checkpoints often omit the field; without a convert-side default the
    // runtime falls through to the generic 1e-5 family default and drifts.
    let rms_norm_eps = parse_rms_norm_eps(&config, &model_type)
        .or_else(|| is_unlimited_ocr(&model_type).then_some(1e-6));
    let linear_attention = linear_attention_config(&config, &model_type);
    let mla_attention = mla_attention_config(&config, &model_type);
    let glm_router = glm_router_config(&config, &model_type);
    let deepseek_v4 = deepseek_v4_config(&config, &model_type);

    let layer_types = parse_layer_types(&config, &model_type, arch.layer_count);
    let global_head_dim = arch_u64(&config, &model_type, "global_head_dim").and_then(u64_to_u32);
    let global_kv_head_count = arch_u64(&config, &model_type, "num_global_key_value_heads")
        .and_then(u64_to_u32)
        .or_else(|| infer_global_kv_head_count(&mapped_tensors, &layer_types, global_head_dim));
    // Unlimited-OCR's value is interpreted by the MLX runtime as protected-prefix
    // R-SWA: the complete image/text prefill remains resident and only generated
    // tokens rotate through this many decode slots. Other uniform-SWA families
    // apply the same manifest field to both prefill and decode.
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

    let mut manifest = NativeModelManifest {
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
        rope_beta_fast,
        rope_beta_slow,
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
        global_kv_head_count,
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
        deepseek_v4,
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
        dropped_tensors: dropped_ledger.to_provenance(),
        kv_cache_quantization: None,
        tensors: mapped_tensors,
    };

    // Best-effort bridge: lift AXQuant's per-layer KV-cache quantization table
    // from `axquant_runtime.json` into the manifest so runtimes that only read
    // `model-manifest.json` can see it. A missing/malformed file or an
    // inconsistent table must never fail conversion.
    manifest.kv_cache_quantization = lift_axquant_kv_cache_quantization(model_dir, &manifest);

    validate_converted_model_contract(&config, &model_type, &manifest)?;

    Ok(manifest)
}

/// Infer a uniform full-attention KV head count from split K projections.
///
/// Early Gemma 4 configs (notably E2B/E4B) omit
/// `num_global_key_value_heads`, and the global layers do not necessarily
/// preserve the sliding layers' total KV width. Safetensors row counts are
/// authoritative and remain logical output dimensions even for MLX affine
/// weights, so record the inferred count in newly generated manifests.
fn infer_global_kv_head_count(
    tensors: &[NativeTensorSpec],
    layer_types: &[String],
    global_head_dim: Option<u32>,
) -> Option<u32> {
    let head_dim = u64::from(global_head_dim?);
    if head_dim == 0 {
        return None;
    }

    let counts: BTreeSet<u32> = tensors
        .iter()
        .filter(|tensor| tensor.role == NativeTensorRole::AttentionK)
        .filter_map(|tensor| {
            let layer_index = tensor.layer_index? as usize;
            (layer_types.get(layer_index).map(String::as_str) == Some("full_attention"))
                .then_some(())?;
            let rows = *tensor.shape.first()?;
            (tensor.shape.len() == 2 && rows > 0 && rows.is_multiple_of(head_dim))
                .then(|| u32::try_from(rows / head_dim).ok())
                .flatten()
        })
        .collect();

    (counts.len() == 1)
        .then(|| counts.first().copied())
        .flatten()
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

/// Best-effort lift of the per-layer KV-cache quantization table from a
/// converted checkpoint's `axquant_runtime.json` (`kv_cache` block, schema
/// `axquant.runtime.v1`) into the manifest, so runtimes that only read
/// `model-manifest.json` can see it.
///
/// Returns `None` — leaving the manifest field unset — when the file is
/// absent, malformed, or carries no `kv_cache` block (debug-logged), and when
/// the table is present but inconsistent with the manifest's `layer_count` or
/// the allowed value sets (warn-logged). Conversion never fails on this
/// bridge.
fn lift_axquant_kv_cache_quantization(
    model_dir: &Path,
    manifest: &NativeModelManifest,
) -> Option<KvCacheQuantizationManifest> {
    let path = model_dir.join("axquant_runtime.json");
    let Ok(bytes) = fs::read(&path) else {
        // Absent sidecar is the common case — nothing to lift.
        return None;
    };
    let value = match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(value) => value,
        Err(error) => {
            tracing::debug!(
                target: "ax_engine_core::convert",
                path = %path.display(),
                %error,
                "axquant_runtime_json_unparseable_skipping_kv_cache_lift"
            );
            return None;
        }
    };
    let kv_cache = value.get("kv_cache")?;
    let layer_bits = json_u32_array(kv_cache.get("layer_bits"));
    let layer_group_sizes = json_u32_array(kv_cache.get("layer_group_sizes"));
    let (Some(layer_bits), Some(layer_group_sizes)) = (layer_bits, layer_group_sizes) else {
        tracing::warn!(
            target: "ax_engine_core::convert",
            path = %path.display(),
            "axquant_runtime_kv_cache_missing_or_typed_arrays_skipping_lift"
        );
        return None;
    };
    let basis = kv_cache
        .get("allocation_basis")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let layer_count = manifest.layer_count as usize;
    if layer_bits.len() != layer_count || layer_group_sizes.len() != layer_count {
        tracing::warn!(
            target: "ax_engine_core::convert",
            path = %path.display(),
            layer_bits = layer_bits.len(),
            layer_group_sizes = layer_group_sizes.len(),
            layer_count,
            "axquant_runtime_kv_cache_length_mismatch_skipping_lift"
        );
        return None;
    }
    let values_valid =
        layer_bits
            .iter()
            .zip(layer_group_sizes.iter())
            .all(|(&bits, &group_size)| {
                matches!(bits, 4 | 6 | 8 | 16)
                    && (bits == 16 || matches!(group_size, 32 | 64 | 128))
            });
    if !values_valid {
        tracing::warn!(
            target: "ax_engine_core::convert",
            path = %path.display(),
            "axquant_runtime_kv_cache_invalid_values_skipping_lift"
        );
        return None;
    }

    Some(KvCacheQuantizationManifest {
        layer_bits,
        layer_group_sizes,
        basis,
    })
}

fn json_u32_array(value: Option<&serde_json::Value>) -> Option<Vec<u32>> {
    value?
        .as_array()?
        .iter()
        .map(|entry| entry.as_u64().and_then(|n| u32::try_from(n).ok()))
        .collect()
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
    let (temp_path, mut temp_file) =
        create_manifest_temp_file(model_dir).map_err(|source| ConvertError::ReadFile {
            path: manifest_path.clone(),
            source,
        })?;
    let mut temp_guard = ManifestTempGuard::new(temp_path);
    temp_file
        .write_all(&json)
        .and_then(|()| temp_file.flush())
        .and_then(|()| temp_file.sync_all())
        .map_err(|source| ConvertError::ReadFile {
            path: manifest_path.clone(),
            source,
        })?;
    drop(temp_file);

    // On Unix, renaming a file over a symlink replaces the symlink directory
    // entry itself. It never opens or writes through the symlink target.
    fs::rename(temp_guard.path(), &manifest_path).map_err(|source| ConvertError::ReadFile {
        path: manifest_path,
        source,
    })?;
    temp_guard.disarm();
    Ok(())
}

fn create_manifest_temp_file(model_dir: &Path) -> std::io::Result<(PathBuf, fs::File)> {
    for _ in 0..MANIFEST_TEMP_CREATE_ATTEMPTS {
        let sequence = MANIFEST_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp_path = model_dir.join(format!(
            "{MANIFEST_TEMP_FILE_PREFIX}{}-{sequence}",
            std::process::id()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!(
            "could not create a unique manifest temp file after {MANIFEST_TEMP_CREATE_ATTEMPTS} attempts"
        ),
    ))
}

struct ManifestTempGuard {
    path: PathBuf,
    armed: bool,
}

impl ManifestTempGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ManifestTempGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
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

/// Provenance note recorded in manifests auto-generated at load time from a
/// raw HuggingFace / MLX snapshot (rather than by an explicit convert step).
pub const AUTO_GENERATED_MANIFEST_NOTE: &str =
    "auto-generated by ax-engine from config.json + safetensors headers (no tensor data converted)";

/// Ensure a raw HuggingFace / MLX model directory carries a
/// `model-manifest.json`, generating one from `config.json` and the
/// safetensors headers when it is absent.
///
/// Returns `Ok(false)` when the manifest already exists (left untouched) and
/// `Ok(true)` after a successful generate + validate + atomic write.
/// Directories without `config.json` or without any `*.safetensors` fail with
/// the descriptive `convert_hf_model_dir` error (`ReadFile` /
/// `NoSafetensors`); conversion contract violations (dropped tensors under
/// strict mode, invalid model contract) propagate unchanged so they stay
/// fail-loud. A generated manifest that fails loader validation is never
/// written (`GeneratedManifestInvalid`), so the directory stays convertible
/// instead of being poisoned by a permanently-invalid manifest file.
pub fn ensure_manifest_for_hf_model_dir(model_dir: &Path) -> Result<bool, ConvertError> {
    let manifest_path = model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
    if manifest_path.exists() {
        return Ok(false);
    }

    let mut manifest = convert_hf_model_dir(model_dir)?;
    manifest
        .runtime_status
        .notes
        .push(AUTO_GENERATED_MANIFEST_NOTE.to_string());
    // Validate before writing: a manifest that fails loader validation must
    // not reach disk, because its mere presence stops every later
    // `from_dir_or_convert` from retrying conversion (only the NotFound arm
    // converts) and flips AX-ready detection in other tools.
    crate::model::validate_native_model_manifest(model_dir, &manifest).map_err(|error| {
        ConvertError::GeneratedManifestInvalid {
            dir: model_dir.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    write_manifest(model_dir, &manifest)?;
    Ok(true)
}

mod hf_config;
mod model_family;
mod tensor_mapping;
#[cfg(test)]
mod tests;

mod deepseek_v4_quantized;

pub(crate) use deepseek_v4_quantized::*;
pub(crate) use hf_config::*;
pub(crate) use model_family::*;
pub(crate) use tensor_mapping::*;

// ---------------------------------------------------------------------------
// Safetensors header parsing
// ---------------------------------------------------------------------------

/// Parsed tensor info from a safetensors file header.
pub(crate) struct SafetensorEntry {
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
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
                // Converter-generated artifacts (DeepSeek V4 FP8/FP4 repack)
                // are outputs, never source inputs — a re-run regenerates them.
                && path.file_name().and_then(|n| n.to_str())
                    != Some(DEEPSEEK_V4_CONVERTED_SAFETENSORS_FILE)
            {
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
            offset_bytes: data_base_offset
                .checked_add(entry.data_offsets[0])
                .ok_or_else(|| ConvertError::InvalidSafetensorsHeader {
                    path: path.to_path_buf(),
                    message: format!(
                        "data offset overflow for tensor {name}: base {data_base_offset} + start {}",
                        entry.data_offsets[0]
                    ),
                })?,
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
        // Signed 32-bit integers (DeepSeek V4 `ffn.gate.tid2eid` hash-routing
        // table). Stored with the same 4-byte U32 container dtype the GGUF
        // loader already uses for GGML_TYPE_I32; not a quantized weight.
        "I32" => Ok(NativeTensorDataType::U32),
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
        .or_else(|| tensor_name.strip_prefix("backbone."))
        .unwrap_or(tensor_name);
    let unprefixed_module_name = unprefixed_name
        .strip_suffix(".weight")
        .unwrap_or(unprefixed_name);
    // OptiQ keys are full module paths (e.g. backbone.layers.0.mixer.in_proj).
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
    // Whisper is an encoder-decoder model with a dedicated runtime rather than
    // the decoder-only role schema used by `ModelWeights`. Preserve every
    // inference tensor by its exact checkpoint name and let that runtime
    // validate/load the native Whisper layout.
    if family.family_name == "whisper" {
        return (name != "alignment_heads").then_some((NativeTensorRole::Other, None));
    }

    // Nemotron-H: backbone.embeddings / backbone.norm_f / backbone.layers.N.* / lm_head.
    if family.family_name == "nemotron_h" {
        if let Some(result) = match_nemotron_h_tensor(name, family.tensor_map) {
            return Some(result);
        }
        if name.starts_with("vision_model.")
            || name.starts_with("mlp1.")
            || name.starts_with("sound_encoder.")
            || name.starts_with("sound_projection.")
        {
            return Some((NativeTensorRole::Other, None));
        }
    }

    // DeepSeek V4 raw HuggingFace checkpoints use a bare `layers.N.…` layout
    // (no `model.` prefix) plus `mtp.N.…` sidecar tensors. Handle those before
    // the generic maps; `model.`-prefixed / sanitized names fall through to
    // the standard + extra-map matching below.
    if family.family_name == "deepseek_v4" {
        if let Some(extra) = family.extra_tensor_map {
            if let Some(result) = match_deepseek_v4_tensor(name, extra) {
                return Some(result);
            }
        }
    }

    // Nemotron 3 Embed HF packs use bare `embed_tokens` / `layers.N.*` / `norm`
    // (no `model.` prefix). Also accept the standard `model.*` layout.
    if family.family_name == "nemotron_embed" {
        if let Some(result) = match_nemotron_embed_tensor(name, family.tensor_map) {
            return Some(result);
        }
    }

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

    // Unified Qwen checkpoints keep the full vision tower under
    // `vision_tower.*` / `visual.*` (sanitized MLX variants) or
    // `model.visual.*` (raw HF). Preserve every tower tensor by name. The
    // runtime loader is config-driven and consumes exact names, while the
    // generic `Other` role keeps converter validation from pretending these
    // tensors are language layers.
    if matches!(family.family_name, "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5")
        && (name.starts_with("vision_tower.")
            || name.starts_with("visual.")
            || name.starts_with("model.visual."))
    {
        return Some((NativeTensorRole::Other, None));
    }

    // Standard Gemma 4 checkpoints (E2B/E4B/26B/31B) carry an encoder-based
    // ViT and projection, distinct from gemma4_unified's encoder-free roles.
    // Preserve the exact names so the config-driven native vision loader can
    // consume both MLX and raw-HF prefix layouts.
    if matches!(family.family_name, "gemma4" | "gemma4_vl")
        && (name.starts_with("vision_tower.")
            || name.starts_with("model.vision_tower.")
            || name.starts_with("embed_vision.")
            || name.starts_with("model.embed_vision."))
    {
        return Some((NativeTensorRole::Other, None));
    }

    if family.family_name == "minicpmv4_6"
        && (name.starts_with("vision_tower.")
            || name.starts_with("model.vision_tower.")
            || name.starts_with("model.vpm.")
            || name.starts_with("vit_merger.")
            || name.starts_with("model.vit_merger.")
            || name.starts_with("merger.")
            || name.starts_with("model.merger."))
    {
        return Some((NativeTensorRole::Other, None));
    }

    // Legacy Qwen3-VL manifests used dedicated roles for a subset of the
    // tower. Continue accepting those sanitized names.
    if matches!(family.family_name, "qwen3_vl" | "qwen3_vl_moe") {
        if let Some((idx, role)) = match_qwen3_vl_vision_layer(name) {
            return Some((role, Some(idx)));
        }
        // VL globals for MoE family (dense VL uses family.extra_tensor_map above).
        if family.family_name == "qwen3_vl_moe" {
            if let Some(result) = match_tensor_in_map(name, QWEN3_VL_EXTRA_TENSOR_MAP) {
                return Some(result);
            }
        }
        // Also accept language_model-prefixed visual tower names if present.
        if let Some(stripped) = name.strip_prefix("model.") {
            if let Some((idx, role)) = match_qwen3_vl_vision_layer(stripped) {
                return Some((role, Some(idx)));
            }
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
        // Current unified Qwen3-VL and Qwen3.5 Hugging Face checkpoints use
        // `model.language_model.*` (without the older intermediate `.model`).
        if let Some(result) = match_unified_qwen_language_model_tensor(
            name,
            family.tensor_map,
            family.extra_tensor_map,
        ) {
            return Some(result);
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
            if let Some(result) = match_prefixed_per_layer(
                name,
                "model.language_model.layers.",
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

/// Match the unified Qwen layout emitted by current Transformers checkpoints:
/// `model.language_model.{embed_tokens,norm,layers.*}`.
fn match_unified_qwen_language_model_tensor(
    name: &str,
    tensor_map: &[(&str, TensorMapping)],
    extra_tensor_map: Option<&[(&str, TensorMapping)]>,
) -> Option<(NativeTensorRole, Option<u32>)> {
    match name {
        "model.language_model.embed_tokens.weight" => {
            return Some((NativeTensorRole::TokenEmbedding, None));
        }
        "model.language_model.norm.weight" => {
            return Some((NativeTensorRole::FinalNorm, None));
        }
        "model.lm_head.weight" => {
            return Some((NativeTensorRole::LmHead, None));
        }
        _ => {}
    }
    if let Some(result) = match_prefixed_per_layer(name, "model.language_model.layers.", tensor_map)
    {
        return Some(result);
    }
    extra_tensor_map
        .and_then(|extra| match_prefixed_per_layer(name, "model.language_model.layers.", extra))
}

/// Match Nemotron-H backbone-prefixed tensors.
fn match_nemotron_h_tensor(
    name: &str,
    tensor_map: &[(&str, TensorMapping)],
) -> Option<(NativeTensorRole, Option<u32>)> {
    let name = name.strip_prefix("language_model.").unwrap_or(name);
    match name {
        "backbone.embeddings.weight" | "backbone.embed_tokens.weight" => {
            return Some((NativeTensorRole::TokenEmbedding, None));
        }
        "backbone.norm_f.weight" | "backbone.norm.weight" => {
            return Some((NativeTensorRole::FinalNorm, None));
        }
        "lm_head.weight" => {
            return Some((NativeTensorRole::LmHead, None));
        }
        _ => {}
    }
    if let Some(result) = match_prefixed_per_layer(name, "backbone.layers.", tensor_map) {
        return Some(result);
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

/// Match Nemotron 3 Embed bare HF tensors (`embed_tokens`, `layers.N.*`, `norm`).
fn match_nemotron_embed_tensor(
    name: &str,
    tensor_map: &[(&str, TensorMapping)],
) -> Option<(NativeTensorRole, Option<u32>)> {
    match name {
        "embed_tokens.weight" => return Some((NativeTensorRole::TokenEmbedding, None)),
        "norm.weight" => return Some((NativeTensorRole::FinalNorm, None)),
        "lm_head.weight" => return Some((NativeTensorRole::LmHead, None)),
        _ => {}
    }
    if let Some(result) = match_prefixed_per_layer(name, "layers.", tensor_map) {
        return Some(result);
    }
    // Sanitized / mlx-community style `model.*` names.
    if let Some(result) = match_tensor_in_map(name, tensor_map) {
        return Some(result);
    }
    None
}

/// Match DeepSeek V4 raw-HF tensors that the generic maps cannot reach: the
/// bare `layers.N.…` prefix (no `model.`), `layers.N.nextn.…` / `mtp.N.…` MTP
/// sidecar tensors, and raw per-expert `ffn.experts.{eid}.w{1,2,3}.weight`
/// stacks. `model.`-prefixed names are handled by the generic maps.
fn match_deepseek_v4_tensor(
    name: &str,
    extra_tensor_map: &[(&str, TensorMapping)],
) -> Option<(NativeTensorRole, Option<u32>)> {
    if let Some(rest) = name.strip_prefix("layers.") {
        let dot = rest.find('.')?;
        let layer_index: u32 = rest[..dot].parse().ok()?;
        let suffix = &rest[dot + 1..];
        // MTP layers live at indices >= num_hidden_layers, which the
        // layer-indexed role schema cannot represent; map them root-level.
        if let Some(nextn) = suffix.strip_prefix("nextn.") {
            return deepseek_v4_nextn_role(nextn).map(|role| (role, None));
        }
        for (pattern, mapping) in extra_tensor_map {
            if let TensorMapping::PerLayer(role) = mapping {
                if suffix == *pattern {
                    return Some((*role, Some(layer_index)));
                }
            }
        }
        // Raw HF per-expert `ffn.experts.{eid}.w{1,2,3}.weight` tensors are
        // intentionally not mapped: conversion is metadata-only and cannot
        // stack them, so they land in the dropped ledger (fail-loud in strict
        // mode) exactly like raw per-expert DeepSeek V3 checkpoints.
        return None;
    }
    if let Some(rest) = name.strip_prefix("mtp.") {
        let dot = rest.find('.')?;
        // The MTP index only orders the predictor layers; roles stay root-level.
        if rest[..dot].parse::<u32>().is_err() {
            return None;
        }
        return deepseek_v4_mtp_role(&rest[dot + 1..]).map(|role| (role, None));
    }
    None
}

/// MTP sidecar roles for `layers.N.nextn.<suffix>` (llama.cpp/GGUF layout)
/// and `mtp.N.<suffix>` (raw HF layout, minus the hc_head trio).
fn deepseek_v4_nextn_role(suffix: &str) -> Option<NativeTensorRole> {
    match suffix {
        "eh_proj.weight" => Some(NativeTensorRole::NextnEhProj),
        "e_proj.weight" => Some(NativeTensorRole::NextnEproj),
        "h_proj.weight" => Some(NativeTensorRole::NextnHproj),
        "enorm.weight" => Some(NativeTensorRole::NextnEnorm),
        "hnorm.weight" => Some(NativeTensorRole::NextnHnorm),
        "shared_head_norm.weight" | "norm.weight" => Some(NativeTensorRole::NextnSharedHeadNorm),
        "embed_tokens.weight" => Some(NativeTensorRole::NextnEmbedTokens),
        "shared_head_head.weight" => Some(NativeTensorRole::NextnSharedHeadHead),
        _ => None,
    }
}

/// Raw HF `mtp.N.<suffix>` roles: the hc_head trio maps to the same
/// root-level roles as the main checkpoint's `hc_head_*` tensors.
fn deepseek_v4_mtp_role(suffix: &str) -> Option<NativeTensorRole> {
    match suffix {
        "hc_head_fn" => Some(NativeTensorRole::HcHeadFn),
        "hc_head_base" => Some(NativeTensorRole::HcHeadBase),
        "hc_head_scale" => Some(NativeTensorRole::HcHeadScale),
        other => deepseek_v4_nextn_role(other),
    }
}

fn map_tensors(
    config: &serde_json::Value,
    all_tensors: &[SafetensorEntry],
    family: &ModelFamily,
    consumed_source_names: Option<&BTreeSet<String>>,
) -> Result<(Vec<NativeTensorSpec>, DroppedTensorLedger), ConvertError> {
    let mut mapped = Vec::new();
    let mut dropped = DroppedTensorLedger::default();

    for entry in all_tensors {
        // Quantized sources consumed by a family data pass (DeepSeek V4 FP8
        // weights, FP4 experts, and their scale sidecars) are replaced by
        // converted tensors; they must not map or drop as raw entries.
        if consumed_source_names.is_some_and(|names| names.contains(&entry.name)) {
            continue;
        }
        // PyTorch BatchNorm persists this integer training counter beside the
        // inference parameters. It has no role in eval and MLX does not load
        // it; skip it before dtype conversion because safetensors stores it as
        // I64, which is intentionally not a native runtime tensor dtype.
        if is_training_only_tensor(&entry.name) {
            continue;
        }
        let Some((role, layer_index)) = match_tensor(&entry.name, family) else {
            // Fail-loud: count every unrecognised tensor (WS-C1).
            dropped.record(&entry.name, entry.length_bytes);
            continue;
        };

        let dtype = convert_dtype(&entry.dtype, &entry.name)?;
        // Only genuine MLX affine source tensors (safetensors U32) count as
        // source-quantized; I32 integer tensors also decode to the U32
        // container dtype but carry no quantization metadata.
        let source_quantized = entry.dtype == "U32";
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
    Ok((mapped, dropped))
}

fn is_training_only_tensor(name: &str) -> bool {
    name.ends_with(".num_batches_tracked") || name == "alignment_heads"
}

/// Verify that every source-quantized (U32) DeepSeek V4 tensor in the
/// manifest has its MLX affine sidecars (`{base}.scales` and
/// `{base}.biases`) among the parsed source tensors. The manifest maps only
/// the `.weight` member; `take_weight_spec` in the runtime resolves the
/// sidecars by name from the merged safetensors map and hard-errors when
/// `.scales` is absent, so an incomplete triplet must fail conversion
/// instead of poisoning the generated manifest.
fn validate_deepseek_v4_quantized_triplets(
    mapped_tensors: &[NativeTensorSpec],
    all_tensors: &[SafetensorEntry],
) -> Result<(), ConvertError> {
    let model_type = "deepseek_v4";
    let mut source_names: Option<BTreeSet<&str>> = None;
    for spec in mapped_tensors {
        if !spec.source_quantized {
            continue;
        }
        let names = source_names.get_or_insert_with(|| {
            all_tensors
                .iter()
                .map(|entry| entry.name.as_str())
                .collect()
        });
        let base = spec.name.strip_suffix(".weight").unwrap_or(&spec.name);
        for sidecar in ["scales", "biases"] {
            let sidecar_name = format!("{base}.{sidecar}");
            if !names.contains(sidecar_name.as_str()) {
                return invalid_model_contract(
                    model_type,
                    format!(
                        "quantized tensor {} is missing its MLX affine sidecar {sidecar_name}",
                        spec.name
                    ),
                );
            }
        }
    }
    Ok(())
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
    if is_deepseek_v4(model_type) {
        return validate_deepseek_v4_contract(config, model_type, manifest);
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

fn validate_deepseek_v4_contract(
    config: &serde_json::Value,
    model_type: &str,
    manifest: &NativeModelManifest,
) -> Result<(), ConvertError> {
    let attention = &manifest.deepseek_v4.attention;
    require_model_config(
        model_type,
        attention.head_dim,
        "deepseek_v4.attention.head_dim",
    )?;
    require_model_config(
        model_type,
        attention.qk_rope_head_dim,
        "deepseek_v4.attention.qk_rope_head_dim",
    )?;
    require_model_config(
        model_type,
        attention.q_lora_rank,
        "deepseek_v4.attention.q_lora_rank",
    )?;
    require_model_config(
        model_type,
        attention.o_lora_rank,
        "deepseek_v4.attention.o_lora_rank",
    )?;
    require_model_config(
        model_type,
        attention.o_groups,
        "deepseek_v4.attention.o_groups",
    )?;
    require_model_config(
        model_type,
        attention.index_topk,
        "deepseek_v4.attention.index_topk",
    )?;
    require_model_config(
        model_type,
        attention.index_n_heads,
        "deepseek_v4.attention.index_n_heads",
    )?;
    require_model_config(
        model_type,
        attention.index_head_dim,
        "deepseek_v4.attention.index_head_dim",
    )?;
    require_model_config(
        model_type,
        attention.compress_rope_theta,
        "deepseek_v4.attention.compress_rope_theta",
    )?;

    // V4 gives each attention head a full head_dim; the manifest-level head
    // dim must come from the same config value (no hardcoded constant).
    if attention.head_dim != Some(manifest.attention_head_dim) {
        return invalid_model_contract(
            model_type,
            format!(
                "deepseek_v4.attention.head_dim {:?} must equal attention_head_dim {}",
                attention.head_dim, manifest.attention_head_dim
            ),
        );
    }
    // V4 attention uses a single fused KV projection (num_key_value_heads = 1).
    if manifest.kv_head_count != 1 {
        return invalid_model_contract(
            model_type,
            format!(
                "deepseek_v4 requires num_key_value_heads == 1, got {}",
                manifest.kv_head_count
            ),
        );
    }

    require_model_config(
        model_type,
        manifest.deepseek_v4.hc_mult,
        "deepseek_v4.hc_mult",
    )?;
    require_model_config(
        model_type,
        manifest.deepseek_v4.hc_sinkhorn_iters,
        "deepseek_v4.hc_sinkhorn_iters",
    )?;
    if manifest
        .deepseek_v4
        .hc_eps
        .is_none_or(|value| !value.is_finite() || value <= 0.0)
    {
        return invalid_model_contract(
            model_type,
            "deepseek_v4.hc_eps must be configured, finite, and > 0",
        );
    }
    let num_hash_layers =
        manifest
            .deepseek_v4
            .num_hash_layers
            .ok_or_else(|| ConvertError::InvalidModelContract {
                model_type: model_type.to_string(),
                message: "deepseek_v4.num_hash_layers must be configured".to_string(),
            })?;
    if num_hash_layers > manifest.layer_count {
        return invalid_model_contract(
            model_type,
            format!(
                "deepseek_v4.num_hash_layers {num_hash_layers} must be <= layer_count {}",
                manifest.layer_count
            ),
        );
    }
    if manifest
        .deepseek_v4
        .swiglu_limit
        .is_none_or(|value| !value.is_finite() || value <= 0.0)
    {
        return invalid_model_contract(
            model_type,
            "deepseek_v4.swiglu_limit must be configured, finite, and > 0",
        );
    }
    // Only the sqrtsoftplus routing scorer is understood; reject unknown
    // scorers instead of silently routing with the wrong function.
    if manifest.deepseek_v4.scoring_func.as_deref() != Some("sqrtsoftplus") {
        return invalid_model_contract(
            model_type,
            format!(
                "deepseek_v4.scoring_func must be \"sqrtsoftplus\", got {:?}",
                manifest.deepseek_v4.scoring_func
            ),
        );
    }
    // V4 routing is scoring_func-based; the V3 sigmoid-routing flag must
    // never leak into a V4 manifest.
    if manifest.moe.sigmoid_routing {
        return invalid_model_contract(
            model_type,
            "deepseek_v4 must not enable moe.sigmoid_routing (routing is scoring_func-based)",
        );
    }

    if manifest.deepseek_v4.compress_ratios.len() != manifest.layer_count as usize {
        return invalid_model_contract(
            model_type,
            format!(
                "deepseek_v4.compress_ratios must contain one entry per layer, got {} for layer_count {}",
                manifest.deepseek_v4.compress_ratios.len(),
                manifest.layer_count
            ),
        );
    }
    for (layer_index, ratio) in manifest.deepseek_v4.compress_ratios.iter().enumerate() {
        if !matches!(ratio, 0 | 4 | 128) {
            return invalid_model_contract(
                model_type,
                format!(
                    "deepseek_v4.compress_ratios[{layer_index}] must be 0, 4, or 128, got {ratio}"
                ),
            );
        }
    }
    let has_shared_experts = arch_u64(config, model_type, "n_shared_experts").unwrap_or(0) > 0;

    for layer_index in 0..manifest.layer_count {
        for role in [
            NativeTensorRole::AttentionNorm,
            NativeTensorRole::AttentionQa,
            NativeTensorRole::AttentionQaNorm,
            NativeTensorRole::AttentionQb,
            NativeTensorRole::AttentionKv,
            NativeTensorRole::AttentionKvNorm,
            NativeTensorRole::AttentionOutA,
            NativeTensorRole::AttentionOutB,
            NativeTensorRole::AttnSink,
            NativeTensorRole::HcAttnFn,
            NativeTensorRole::HcAttnBase,
            NativeTensorRole::HcAttnScale,
            NativeTensorRole::HcFfnFn,
            NativeTensorRole::HcFfnBase,
            NativeTensorRole::HcFfnScale,
            NativeTensorRole::FfnNorm,
            NativeTensorRole::FfnGateInp,
            NativeTensorRole::FfnDownExps,
        ] {
            require_model_role(model_type, manifest, layer_index, role)?;
        }
        // Routed experts ship exactly one layout: split gate/up stacks (raw
        // HF / sanitized `ffn.experts.{gate,up}`) or one fused gate+up tensor
        // (AXQ/mlx-lm `ffn.switch_mlp.gate_proj` → ffn_gate_up_exps_packed).
        let has_packed_experts =
            has_model_role(manifest, layer_index, NativeTensorRole::FfnGateUpExpsPacked);
        let has_gate_exps = has_model_role(manifest, layer_index, NativeTensorRole::FfnGateExps);
        let has_up_exps = has_model_role(manifest, layer_index, NativeTensorRole::FfnUpExps);
        if has_packed_experts == (has_gate_exps || has_up_exps) || has_gate_exps != has_up_exps {
            return invalid_model_contract(
                model_type,
                format!(
                    "layer {layer_index} must provide exactly one routed-expert layout: ffn_gate_up_exps_packed or ffn_gate_exps plus ffn_up_exps"
                ),
            );
        }
        if has_shared_experts {
            for role in [
                NativeTensorRole::FfnSharedExpertGate,
                NativeTensorRole::FfnSharedExpertUp,
                NativeTensorRole::FfnSharedExpertDown,
            ] {
                require_model_role(model_type, manifest, layer_index, role)?;
            }
        }

        let compress_ratio = manifest.deepseek_v4.compress_ratios[layer_index as usize];
        if matches!(compress_ratio, 4 | 128) {
            for role in [
                NativeTensorRole::CompressorKv,
                NativeTensorRole::CompressorGate,
                NativeTensorRole::CompressorApe,
                NativeTensorRole::CompressorNorm,
            ] {
                require_model_role(model_type, manifest, layer_index, role)?;
            }
        } else {
            // The compressor exists iff the layer compresses (ratio 4/128):
            // reject stray compressor tensors on raw sliding-window layers.
            for role in [
                NativeTensorRole::CompressorKv,
                NativeTensorRole::CompressorGate,
                NativeTensorRole::CompressorApe,
                NativeTensorRole::CompressorNorm,
            ] {
                if has_model_role(manifest, layer_index, role) {
                    return invalid_model_contract(
                        model_type,
                        format!(
                            "layer {layer_index} must not provide compressor role {role:?} with compress_ratio {compress_ratio}"
                        ),
                    );
                }
            }
        }
        if compress_ratio == 4 {
            for role in [
                NativeTensorRole::IndexerProj,
                NativeTensorRole::IndexerQb,
                NativeTensorRole::IndexerCompressorKv,
                NativeTensorRole::IndexerCompressorGate,
                NativeTensorRole::IndexerCompressorApe,
                NativeTensorRole::IndexerCompressorNorm,
            ] {
                require_model_role(model_type, manifest, layer_index, role)?;
            }
        } else {
            // The sparse indexer exists iff compress_ratio == 4.
            for role in [
                NativeTensorRole::IndexerProj,
                NativeTensorRole::IndexerQb,
                NativeTensorRole::IndexerCompressorKv,
                NativeTensorRole::IndexerCompressorGate,
                NativeTensorRole::IndexerCompressorApe,
                NativeTensorRole::IndexerCompressorNorm,
            ] {
                if has_model_role(manifest, layer_index, role) {
                    return invalid_model_contract(
                        model_type,
                        format!(
                            "layer {layer_index} must not provide indexer role {role:?} with compress_ratio {compress_ratio}"
                        ),
                    );
                }
            }
        }

        let has_tid2eid = has_model_role(manifest, layer_index, NativeTensorRole::FfnGateTid2Eid);
        let has_correction_bias = has_model_role(
            manifest,
            layer_index,
            NativeTensorRole::FfnGateInpCorrectionBias,
        );
        let is_hash_layer = layer_index < num_hash_layers;
        if is_hash_layer != has_tid2eid || is_hash_layer == has_correction_bias {
            return invalid_model_contract(
                model_type,
                format!(
                    "layer {layer_index} must provide ffn_gate_tid2eid on hash layers (index < num_hash_layers {num_hash_layers}) or ffn_gate_inp_correction_bias otherwise, exactly one"
                ),
            );
        }
    }

    for role in [
        NativeTensorRole::HcHeadFn,
        NativeTensorRole::HcHeadBase,
        NativeTensorRole::HcHeadScale,
    ] {
        require_model_global_role(model_type, manifest, role)?;
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
    let rope_scaling = if uses_text_config(model_type) {
        config
            .get("text_config")
            .and_then(|text_config| text_config.get("rope_scaling"))
            .or_else(|| config.get("rope_scaling"))
    } else {
        config.get("rope_scaling")
    };
    // Unified Qwen VL checkpoints describe their ordinary, unscaled RoPE plus
    // multimodal axis split in this object. `rope_type=default` does not scale
    // frequencies; the MRoPE fields are consumed by the native VL prefill path.
    // Accept all HF model_type aliases that map to the qwen3_5 runtime family
    // (plus dedicated VL types). Omitting `qwen3_5_moe` / `qwen3.5` used to
    // reject valid default-MRoPE configs as unsupported rope_scaling.
    let is_qwen_mrope_family = matches!(
        model_type,
        "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3.5" | "qwen3_5_moe" | "qwen3_5_text"
    );
    let is_default_mrope = is_qwen_mrope_family
        && rope_scaling.is_some_and(|value| {
            value
                .get("rope_type")
                .or_else(|| value.get("type"))
                .and_then(serde_json::Value::as_str)
                == Some("default")
                && value
                    .get("mrope_section")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|sections| sections.len() == 3)
                && value.get("factor").is_none_or(serde_json::Value::is_null)
        });
    let has_unsupported_rope_scaling =
        rope_scaling.is_some_and(|value| !value.is_null()) && !is_default_mrope;
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
