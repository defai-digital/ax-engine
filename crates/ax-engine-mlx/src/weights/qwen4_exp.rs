//! Bounded dedicated weight loader for Qwen 3.8 Flash Next (`qwen4_exp`).
//!
//! Loads every resident tensor (everything except the n-gram lexical table)
//! through [`mlx_sys::load_safetensors_filtered`], so the 51B n-gram table
//! never enters this crate's resident working set. The table itself is
//! opened separately through [`crate::ngram_table::NgramTable`], which serves
//! bounded row gathers instead of a whole-tensor load. Construction never
//! reads a full n-gram row block; it only reads the small per-head hash
//! metadata buffers (`multipliers`, `head_offsets`, `head_vocab_sizes`).
//!
//! This module owns tensor resolution, path containment, checkpoint sanitize
//! transforms, and per-component construction. It does not own trunk
//! composition, request state, or top-level `ModelWeights` dispatch.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ax_engine_core::{NativeModelManifest, NativeTensorRole, NativeTensorSpec, WeightSanitize};
#[cfg(test)]
use mlx_sys::eval;
use mlx_sys::{
    DEFAULT_MAX_GATHER_BYTES, MlxArray, MlxDtype, SafetensorsNameFilter, add, astype, contiguous,
    load_safetensors_filtered, slice, transpose,
};

use crate::model::LinearAttentionConfig;
use crate::model::shared::qwen4_exp_attention::{
    Qwen4ExpAttention, Qwen4ExpAttentionConfig, Qwen4ExpAttentionWeights,
};
use crate::model::shared::qwen4_exp_gdn::{Qwen4ExpGdn, Qwen4ExpGdnWeights};
use crate::model::shared::qwen4_exp_moe::{
    Qwen4ExpExpertWeights, Qwen4ExpMoe, Qwen4ExpMoeWeights, Qwen4ExpResidentExperts,
};
use crate::model::shared::qwen4_exp_ple::{Qwen4ExpPle, Qwen4ExpPleWeights};
use crate::model::shared::qwen4_exp_residual::{
    Qwen4ExpGatedResidual, Qwen4ExpGatedResidualWeights, Qwen4ExpStreamLayout,
};
use crate::ngram_table::NgramTable;
use crate::qwen4_exp_ngram::NgramLayout;
use crate::qwen4_exp_qsa::{QsaConfig, QsaIndexer, QsaIndexerWeights};

use super::{QuantizedWeight, WeightLoadError, take_weight, try_take_plain};

/// All Flash Next weights: shared trunk pieces plus one entry per layer.
///
/// Bounded by construction: the n-gram table inside each [`Qwen4ExpPleBundle`]
/// serves rows on demand and is never materialized whole here.
pub(crate) struct Qwen4ExpWeights {
    pub(crate) token_embedding: QuantizedWeight,
    pub(crate) lm_head: QuantizedWeight,
    pub(crate) layout: Qwen4ExpStreamLayout,
    pub(crate) mixer: Qwen4ExpGatedResidual,
    pub(crate) layers: Vec<Qwen4ExpLayerWeights>,
    pub(crate) expert_stream: Option<Arc<crate::expert_stream::ExpertStackPager>>,
}

pub(crate) struct Qwen4ExpLayerWeights {
    pub(crate) attention_hc: Qwen4ExpGatedResidual,
    pub(crate) mlp_hc: Qwen4ExpGatedResidual,
    pub(crate) attention: Qwen4ExpAttentionBranch,
    pub(crate) moe: Qwen4ExpMoe,
    pub(crate) ple: Option<Qwen4ExpPleBundle>,
}

pub(crate) enum Qwen4ExpAttentionBranch {
    Gdn(Qwen4ExpGdn),
    Qsa(Qwen4ExpAttention),
}

/// One PLE layer's injection operator plus its bounded n-gram row source.
pub(crate) struct Qwen4ExpPleBundle {
    pub(crate) operator: Qwen4ExpPle,
    pub(crate) table: NgramTable,
    pub(crate) layout: NgramLayout,
    /// Full PLE embedding width (`ple_embed_dim`), not the table's per-head
    /// row width used by [`NgramTable::open`].
    pub(crate) embedding_width: usize,
}

pub(crate) fn load(
    root: &Path,
    manifest: &NativeModelManifest,
) -> Result<Qwen4ExpWeights, WeightLoadError> {
    load_with_paging_policy(
        root,
        manifest,
        // Fail closed on an invalid AX_STREAM_EXPERTS rather than paging under
        // a silently substituted Auto.
        crate::expert_stream::stream_experts_mode_checked()?,
        crate::expert_stream::expert_layer_budget(),
    )
}

pub(crate) fn load_with_paging_policy(
    root: &Path,
    manifest: &NativeModelManifest,
    mode: crate::expert_stream::StreamExpertsMode,
    budget_layers: usize,
) -> Result<Qwen4ExpWeights, WeightLoadError> {
    if manifest.model_family != "qwen4_exp" {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp loader requires model_family qwen4_exp, got {:?}",
            manifest.model_family
        )));
    }
    if matches!(manifest.weight_sanitize, WeightSanitize::HfLayerNormsOnly) {
        return Err(WeightLoadError::UnsanitizedWeights(
            "qwen4_exp does not support weight_sanitize=HfLayerNormsOnly".to_string(),
        ));
    }
    let stream_manifest = crate::expert_stream::resolve_expert_stream(
        mode,
        crate::expert_stream::ExpertStreamManifest::read_from_dir(root)?,
        || {
            crate::expert_stream::infer_layer_stack_manifest(
                &manifest.tensors,
                manifest.moe.experts_per_token.unwrap_or(1),
            )
        },
        crate::expert_stream::unified_memory_bytes(),
    )?;
    let canonical_root = root.canonicalize().map_err(|e| {
        WeightLoadError::FileMissing(format!("cannot resolve root {}: {e}", root.display()))
    })?;

    let streamed_layers = stream_manifest
        .as_ref()
        .map(|stream| {
            validate_expert_paging_contract(
                root,
                &canonical_root,
                &manifest.tensors,
                manifest.moe.experts_per_token.unwrap_or(1),
                stream,
            )
        })
        .transpose()?
        .unwrap_or_default();
    let stream_skip = stream_manifest
        .as_ref()
        .map(crate::expert_stream::streamed_skip_names)
        .unwrap_or_default();
    let expert_stream = stream_manifest.map(|stream| {
        Arc::new(crate::expert_stream::ExpertStackPager::new(
            Arc::new(stream),
            root.to_path_buf(),
            budget_layers,
        ))
    });
    let specs = manifest.tensors.as_slice();
    let mut name_map = load_resident_tensors(root, &canonical_root, manifest, &stream_skip)?;
    sanitize_norms_and_convs(manifest.weight_sanitize, &manifest.tensors, &mut name_map)?;

    let hidden = as_usize(manifest.hidden_size, "hidden_size")?;
    let query_heads = as_usize(manifest.attention_head_count, "attention_head_count")?;
    let kv_heads = as_usize(manifest.kv_head_count, "kv_head_count")?;
    let head_dim = as_usize(manifest.attention_head_dim, "attention_head_dim")?;
    let rope_base = manifest.rope_theta.map(|v| v as f32).unwrap_or(10_000.0);
    let rms_eps = manifest.rms_norm_eps.unwrap_or(1e-6);
    let factor = require_f32(manifest.partial_rotary_factor, "partial_rotary_factor")?;
    let rotary_dim = as_usize(
        (manifest.attention_head_dim as f32 * factor) as u32,
        "rotary_dim",
    )?;

    let cfg = &manifest.qwen4_exp;
    let hc_count = as_usize(require_u32(cfg.hc_count, "hc_count")?, "hc_count")?;
    let hc_lowrank = as_usize(require_u32(cfg.hc_lowrank, "hc_lowrank")?, "hc_lowrank")?;
    let layout = Qwen4ExpStreamLayout::new(hc_count, hidden)
        .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp stream layout: {e}")))?;

    let indexer_budget = as_usize(
        require_u32(cfg.indexer_budget, "indexer_budget")?,
        "indexer_budget",
    )?;
    let indexer_ratio = as_usize(
        require_u32(cfg.indexer_compress_ratio, "indexer_compress_ratio")?,
        "indexer_compress_ratio",
    )?;
    let indexer_head_dim = as_usize(
        require_u32(cfg.indexer_head_dim, "indexer_head_dim")?,
        "indexer_head_dim",
    )?;
    let indexer_n_heads = as_usize(
        require_u32(cfg.indexer_n_heads, "indexer_n_heads")?,
        "indexer_n_heads",
    )?;
    let indexer_kv_heads = as_usize(
        require_u32(cfg.indexer_kv_heads, "indexer_kv_heads")?,
        "indexer_kv_heads",
    )?;

    let qsa_config = QsaConfig::new(
        indexer_n_heads,
        indexer_kv_heads,
        indexer_head_dim,
        rotary_dim,
        indexer_ratio,
        indexer_budget,
        hidden,
        rms_eps,
        rope_base,
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp qsa config: {e}")))?;

    let main_attn_config = Qwen4ExpAttentionConfig::new(
        hidden,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_base,
        rms_eps,
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp attention config: {e}")))?;

    let la = &manifest.linear_attention;
    let key_head_dim = as_usize(
        require_u32(la.key_head_dim, "linear_attention.key_head_dim")?,
        "linear_attention.key_head_dim",
    )?;
    let (q_scale, k_scale) = crate::linear_attention_ops::linear_attention_qk_scale(key_head_dim);
    let la_config = LinearAttentionConfig {
        full_attention_interval: as_usize(
            require_u32(
                la.resolved_full_attention_interval(&manifest.model_family),
                "linear_attention.full_attention_interval",
            )?,
            "linear_attention.full_attention_interval",
        )?,
        num_value_heads: as_usize(
            require_u32(la.num_value_heads, "linear_attention.num_value_heads")?,
            "linear_attention.num_value_heads",
        )?,
        num_key_heads: as_usize(
            require_u32(la.num_key_heads, "linear_attention.num_key_heads")?,
            "linear_attention.num_key_heads",
        )?,
        key_head_dim,
        value_head_dim: as_usize(
            require_u32(la.value_head_dim, "linear_attention.value_head_dim")?,
            "linear_attention.value_head_dim",
        )?,
        conv_kernel_dim: as_usize(
            require_u32(la.conv_kernel_dim, "linear_attention.conv_kernel_dim")?,
            "linear_attention.conv_kernel_dim",
        )?,
        q_scale,
        k_scale,
    };

    let expert_count = as_usize(
        require_u32(manifest.moe.expert_count, "moe.expert_count")?,
        "moe.expert_count",
    )?;
    let top_k = as_usize(
        require_u32(manifest.moe.experts_per_token, "moe.experts_per_token")?,
        "moe.experts_per_token",
    )?;
    let expert_intermediate = as_usize(
        require_u32(
            manifest.moe.expert_intermediate_size,
            "moe.expert_intermediate_size",
        )?,
        "moe.expert_intermediate_size",
    )?;
    let normalize_top_k = manifest.moe_norm_topk_prob;

    let ple_layer_ids = &cfg.ple_layer_ids;
    let eos = if ple_layer_ids.is_empty() {
        None
    } else {
        Some(resolve_eos_token(root, manifest.vocab_size)?)
    };
    let ple_embed_dim = require_u32(cfg.ple_embed_dim, "ple_embed_dim")?;
    let ple_kernel = as_usize(
        require_u32(cfg.ple_conv_kernel_size, "ple_conv_kernel_size")?,
        "ple_conv_kernel_size",
    )?;
    let ngram_size = require_u32(cfg.ngram_size, "ngram_size")?;
    let heads_per_ngram = require_u32(cfg.heads_per_ngram, "heads_per_ngram")?;

    let token_embedding = take_weight(
        specs,
        &mut name_map,
        NativeTensorRole::TokenEmbedding,
        None,
        "token_embedding",
    )?;
    let has_explicit_lm_head = spec_present(specs, NativeTensorRole::LmHead, None);
    let lm_head = if manifest.tie_word_embeddings && !has_explicit_lm_head {
        let mut tied = QuantizedWeight::new(
            token_embedding.weight.clone(),
            token_embedding.scales.clone(),
            token_embedding.biases.clone(),
        );
        tied.group_size = token_embedding.group_size;
        tied.bits = token_embedding.bits;
        tied.mode = token_embedding.mode.clone();
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

    let mixer = take_gated_residual(
        specs,
        &mut name_map,
        None,
        layout,
        hc_lowrank,
        rms_eps,
        NativeTensorRole::Qwen4ExpHcMixerNorm,
        NativeTensorRole::Qwen4ExpHcMixerMixDown,
        NativeTensorRole::Qwen4ExpHcMixerMixUp,
        None,
        "hc_mixer",
    )?;

    let mut layers = Vec::with_capacity(as_usize(manifest.layer_count, "layer_count")?);
    for layer in 0..manifest.layer_count {
        let attention_hc = take_gated_residual(
            specs,
            &mut name_map,
            Some(layer),
            layout,
            hc_lowrank,
            rms_eps,
            NativeTensorRole::Qwen4ExpAttnHcNorm,
            NativeTensorRole::Qwen4ExpAttnHcMixDown,
            NativeTensorRole::Qwen4ExpAttnHcMixUp,
            Some(NativeTensorRole::Qwen4ExpAttnHcInject),
            "attn_hc",
        )?;
        let mlp_hc = take_gated_residual(
            specs,
            &mut name_map,
            Some(layer),
            layout,
            hc_lowrank,
            rms_eps,
            NativeTensorRole::Qwen4ExpMlpHcNorm,
            NativeTensorRole::Qwen4ExpMlpHcMixDown,
            NativeTensorRole::Qwen4ExpMlpHcMixUp,
            Some(NativeTensorRole::Qwen4ExpMlpHcInject),
            "mlp_hc",
        )?;

        let attention = match layer_kind(manifest, layer)? {
            Qwen4ExpLayerKind::LinearAttention => Qwen4ExpAttentionBranch::Gdn(build_gdn(
                specs,
                &mut name_map,
                layer,
                hidden,
                la_config.clone(),
                rms_eps,
            )?),
            Qwen4ExpLayerKind::Attention => Qwen4ExpAttentionBranch::Qsa(build_qsa_attention(
                specs,
                &mut name_map,
                layer,
                main_attn_config,
                qsa_config,
            )?),
        };

        let moe = build_moe(
            specs,
            &mut name_map,
            layer,
            hidden,
            expert_count,
            expert_intermediate,
            top_k,
            normalize_top_k,
            expert_stream
                .as_ref()
                .filter(|_| streamed_layers.contains(&layer)),
        )?;

        let ple = if ple_layer_ids.contains(&(layer + 1)) {
            let eos = eos.ok_or_else(|| {
                WeightLoadError::InvalidLayer(format!(
                    "qwen4_exp layer {layer}: ple layer requires an eos token but none was resolved"
                ))
            })?;
            Some(build_ple_bundle(
                root,
                specs,
                &mut name_map,
                layer,
                manifest.vocab_size,
                eos,
                hc_count,
                hidden,
                ple_embed_dim,
                ple_kernel,
                ngram_size,
                heads_per_ngram,
                rms_eps,
            )?)
        } else {
            None
        };

        layers.push(Qwen4ExpLayerWeights {
            attention_hc,
            mlp_hc,
            attention,
            moe,
            ple,
        });
    }

    Ok(Qwen4ExpWeights {
        token_embedding,
        lm_head,
        layout,
        mixer,
        layers,
        expert_stream,
    })
}

pub(super) enum Qwen4ExpLayerKind {
    LinearAttention,
    Attention,
}

pub(super) fn layer_kind(
    manifest: &NativeModelManifest,
    layer: u32,
) -> Result<Qwen4ExpLayerKind, WeightLoadError> {
    let kind = manifest.layer_types.get(layer as usize).ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!(
            "qwen4_exp layer {layer}: missing layer_types entry"
        ))
    })?;
    match kind.as_str() {
        "linear_attention" => Ok(Qwen4ExpLayerKind::LinearAttention),
        "full_attention" | "qwen_sparse_attention" => Ok(Qwen4ExpLayerKind::Attention),
        other => Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp layer {layer}: unsupported layer_types entry {other:?}"
        ))),
    }
}

fn spec_present(specs: &[NativeTensorSpec], role: NativeTensorRole, layer: Option<u32>) -> bool {
    specs
        .iter()
        .any(|s| s.role == role && s.layer_index == layer)
}

pub(super) fn as_usize(value: u32, field: &str) -> Result<usize, WeightLoadError> {
    usize::try_from(value)
        .map_err(|_| WeightLoadError::InvalidLayer(format!("qwen4_exp {field} does not fit usize")))
}

pub(super) fn require_u32(value: Option<u32>, field: &str) -> Result<u32, WeightLoadError> {
    value.ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp manifest is missing {field}"))
    })
}

pub(super) fn require_f32(value: Option<f32>, field: &str) -> Result<f32, WeightLoadError> {
    value.ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp manifest is missing {field}"))
    })
}

/// Read `config.json` and resolve `text_config.eos_token_id` (a plain integer
/// or a single-element list). The manifest carries no tokenizer/EOS field for
/// `qwen4_exp`, so this is the only source; a missing or out-of-vocab value
/// fails closed rather than defaulting to token 0.
fn resolve_eos_token(root: &Path, vocab: u32) -> Result<u32, WeightLoadError> {
    let config_path = root.join("config.json");
    let bytes = std::fs::read(&config_path).map_err(|e| {
        WeightLoadError::FileMissing(format!("cannot read {}: {e}", config_path.display()))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        WeightLoadError::InvalidLayer(format!("cannot parse {}: {e}", config_path.display()))
    })?;
    let text_config = value.get("text_config").ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!("{} is missing text_config", config_path.display()))
    })?;
    let eos_field = text_config.get("eos_token_id").ok_or_else(|| {
        WeightLoadError::InvalidLayer(
            "qwen4_exp config.json text_config.eos_token_id is required".to_string(),
        )
    })?;
    let invalid_value = || {
        WeightLoadError::InvalidLayer(
            "qwen4_exp eos_token_id must be a nonnegative integer or a single-element list"
                .to_string(),
        )
    };
    let eos_u64 = match eos_field {
        serde_json::Value::Number(n) => n.as_u64().ok_or_else(invalid_value)?,
        serde_json::Value::Array(items) => {
            if items.len() != 1 {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "qwen4_exp eos_token_id list must have exactly one entry, got {}",
                    items.len()
                )));
            }
            items[0].as_u64().ok_or_else(invalid_value)?
        }
        _ => return Err(invalid_value()),
    };
    let eos = u32::try_from(eos_u64).map_err(|_| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp eos_token_id {eos_u64} exceeds u32"))
    })?;
    if eos >= vocab {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp eos_token_id {eos} exceeds vocab_size {vocab}"
        )));
    }
    Ok(eos)
}

pub(super) fn resolve_in_root(
    root: &Path,
    canonical_root: &Path,
    relative: &Path,
) -> Result<PathBuf, WeightLoadError> {
    crate::artifact_path::resolve_file(root, canonical_root, relative).map_err(|error| {
        WeightLoadError::FileMissing(format!(
            "cannot resolve {} inside {}: {error}",
            relative.display(),
            root.display()
        ))
    })
}

/// `model.safetensors.index.json` `weight_map`, resolved to in-root absolute
/// paths. `None` for single-file checkpoints (no index present).
pub(super) fn read_weight_index(
    root: &Path,
    canonical_root: &Path,
) -> Result<Option<HashMap<String, PathBuf>>, WeightLoadError> {
    let index_path = root.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&index_path).map_err(|e| {
        WeightLoadError::FileMissing(format!("cannot read {}: {e}", index_path.display()))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        WeightLoadError::InvalidLayer(format!("cannot parse {}: {e}", index_path.display()))
    })?;
    let map = value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            WeightLoadError::InvalidLayer(format!(
                "{} is missing a weight_map object",
                index_path.display()
            ))
        })?;
    let mut out = HashMap::with_capacity(map.len());
    for (name, file) in map {
        let file_str = file.as_str().ok_or_else(|| {
            WeightLoadError::InvalidLayer(format!(
                "{}: weight_map entry {name:?} is not a string",
                index_path.display()
            ))
        })?;
        out.insert(
            name.clone(),
            resolve_in_root(root, canonical_root, Path::new(file_str))?,
        );
    }
    Ok(Some(out))
}

/// Bind the paging sidecar to the already validated native tensor contract.
/// The pager currently interprets quantized projections as affine only.
fn validate_expert_paging_contract(
    root: &Path,
    canonical_root: &Path,
    specs: &[NativeTensorSpec],
    expected_top_k: u32,
    stream: &crate::expert_stream::ExpertStreamManifest,
) -> Result<HashSet<u32>, WeightLoadError> {
    use crate::expert_stream::ExpertProj;
    let invalid = |message: String| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp expert paging: {message}"))
    };
    if stream.tensors.is_empty() || stream.experts_per_tok != expected_top_k {
        return Err(invalid(
            "empty plan or experts_per_tok disagrees with model".into(),
        ));
    }
    let projection = |role| match role {
        NativeTensorRole::FfnGateUpExpsPacked => Some(ExpertProj::GateUp),
        NativeTensorRole::FfnGateExps => Some(ExpertProj::Gate),
        NativeTensorRole::FfnUpExps => Some(ExpertProj::Up),
        NativeTensorRole::FfnDownExps => Some(ExpertProj::Down),
        _ => None,
    };
    let by_name: HashMap<&str, &NativeTensorSpec> = specs
        .iter()
        .filter(|spec| projection(spec.role).is_some())
        .map(|spec| (spec.name.as_str(), spec))
        .collect();
    let index = read_weight_index(root, canonical_root)?;
    let mut declared = HashSet::new();
    let mut layers = HashSet::new();
    for entry in &stream.tensors {
        if !declared.insert(entry.name.clone()) {
            return Err(invalid(format!("duplicate tensor {}", entry.name)));
        }
        let sidecar_base = entry
            .name
            .strip_suffix(".scales")
            .or_else(|| entry.name.strip_suffix(".biases"));
        let candidate = sidecar_base.map(|name| format!("{name}.weight"));
        let spec = by_name
            .get(entry.name.as_str())
            .copied()
            .or_else(|| {
                candidate
                    .as_ref()
                    .and_then(|name| by_name.get(name.as_str()).copied())
            })
            .or_else(|| sidecar_base.and_then(|name| by_name.get(name).copied()))
            .ok_or_else(|| {
                invalid(format!(
                    "{} is not a declared expert projection or affine sidecar",
                    entry.name
                ))
            })?;
        if spec.layer_index != Some(entry.layer)
            || entry.expert_axis != 0
            || spec.shape.first().copied() != Some(u64::from(entry.num_experts))
            || entry.num_experts != stream.num_experts
            || entry.parsed_proj != projection(spec.role)
        {
            return Err(invalid(format!(
                "{} disagrees with native layer/projection/expert geometry",
                entry.name
            )));
        }
        let (bits, group_size) = match &spec.quantization {
            Some(quant) if quant.mode == "affine" => (quant.bits, quant.group_size),
            Some(quant) => {
                return Err(invalid(format!(
                    "{} uses unsupported paging quantization {}",
                    spec.name, quant.mode
                )));
            }
            None if matches!(
                spec.dtype,
                ax_engine_core::NativeTensorDataType::F16
                    | ax_engine_core::NativeTensorDataType::Bf16
                    | ax_engine_core::NativeTensorDataType::F32
            ) && !spec.source_quantized =>
            {
                (4, 64)
            }
            None => {
                return Err(invalid(format!(
                    "{} lacks a supported dense or affine contract",
                    spec.name
                )));
            }
        };
        if (entry.bits, entry.group_size) != (bits, group_size) {
            return Err(invalid(format!(
                "{} disagrees with native bits/group_size",
                entry.name
            )));
        }
        let base_file = resolve_in_root(root, canonical_root, &spec.file)?;
        let expected_file = index
            .as_ref()
            .and_then(|map| map.get(&entry.name))
            .unwrap_or(&base_file);
        let actual_file = resolve_in_root(root, canonical_root, &entry.file)?;
        if &actual_file != expected_file || (sidecar_base.is_none() && actual_file != base_file) {
            return Err(invalid(format!(
                "{} disagrees with native file/index",
                entry.name
            )));
        }
        layers.insert(entry.layer);
    }
    for spec in by_name.values().filter(|spec| {
        spec.layer_index
            .is_some_and(|layer| layers.contains(&layer))
    }) {
        if !declared.contains(&spec.name) {
            return Err(invalid(format!(
                "streamed layer is missing expert projection {}",
                spec.name
            )));
        }
        let base = spec.name.strip_suffix(".weight").unwrap_or(&spec.name);
        let linear_bias = format!("{base}.bias");
        if index
            .as_ref()
            .is_some_and(|map| map.contains_key(&linear_bias))
            || specs.iter().any(|tensor| tensor.name == linear_bias)
        {
            return Err(invalid(format!(
                "unsupported dense expert bias {linear_bias}"
            )));
        }
        // The pager automatically reads colocated sidecars. Cross-file sidecars
        // must also appear in its plan so their shard is opened on a cache miss.
        if spec.quantization.is_some() {
            let base_file = resolve_in_root(root, canonical_root, &spec.file)?;
            for suffix in ["scales", "biases"] {
                let name = format!("{base}.{suffix}");
                if let Some(file) = index.as_ref().and_then(|map| map.get(&name))
                    && file != &base_file
                    && !declared.contains(&name)
                {
                    return Err(invalid(format!(
                        "cross-file sidecar {name} is missing from the paging plan"
                    )));
                }
            }
        }
    }
    Ok(layers)
}

/// Select the exact trunk tensors and their named sidecars before opening
/// payloads. Unknown tensors and table-only files never enter MLX storage.
fn load_resident_tensors(
    root: &Path,
    canonical_root: &Path,
    manifest: &NativeModelManifest,
    stream_skip: &HashSet<String>,
) -> Result<HashMap<String, MlxArray>, WeightLoadError> {
    let weight_index = read_weight_index(root, canonical_root)?;
    let mut files = std::collections::BTreeMap::<PathBuf, HashSet<String>>::new();
    let mut declared = HashSet::new();
    for spec in &manifest.tensors {
        if !is_trunk_resident_role(spec.role) || stream_skip.contains(&spec.name) {
            continue;
        }
        if !declared.insert(spec.name.clone()) {
            return Err(WeightLoadError::InvalidLayer(format!(
                "duplicate qwen4_exp tensor {}",
                spec.name
            )));
        }
        let file = resolve_in_root(root, canonical_root, &spec.file)?;
        if weight_index
            .as_ref()
            .and_then(|index| index.get(&spec.name))
            .is_some_and(|indexed| indexed != &file)
        {
            return Err(WeightLoadError::InvalidLayer(format!(
                "qwen4_exp tensor {} conflicts with its file index",
                spec.name
            )));
        }
        files
            .entry(file.clone())
            .or_default()
            .insert(spec.name.clone());
        if let Some(base) = spec.name.strip_suffix(".weight") {
            for suffix in [".scales", ".biases", ".bias"] {
                let name = format!("{base}{suffix}");
                let sidecar_file = weight_index
                    .as_ref()
                    .and_then(|index| index.get(&name))
                    .unwrap_or(&file);
                files.entry(sidecar_file.clone()).or_default().insert(name);
            }
        }
    }
    let mut result = HashMap::new();
    for (path, names) in files {
        let tensors = load_safetensors_filtered(&path, SafetensorsNameFilter::Keep(&names))
            .map_err(WeightLoadError::FileMissing)?;
        mlx_sys::try_eval(&tensors.values().collect::<Vec<_>>()).map_err(|e| {
            WeightLoadError::InvalidLayer(format!("qwen4_exp weight evaluation: {e}"))
        })?;
        for (name, array) in tensors {
            if result.insert(name.clone(), array).is_some() {
                return Err(WeightLoadError::InvalidLayer(format!(
                    "qwen4_exp tensor {name} appears in multiple files"
                )));
            }
        }
    }
    Ok(result)
}

fn is_trunk_resident_role(role: NativeTensorRole) -> bool {
    matches!(
        role,
        NativeTensorRole::AttentionK
            | NativeTensorRole::AttentionKNorm
            | NativeTensorRole::AttentionO
            | NativeTensorRole::AttentionQ
            | NativeTensorRole::AttentionQNorm
            | NativeTensorRole::AttentionV
            | NativeTensorRole::FfnDownExps
            | NativeTensorRole::FfnGateExps
            | NativeTensorRole::FfnGateInp
            | NativeTensorRole::FfnGateUpExpsPacked
            | NativeTensorRole::FfnSharedExpertDown
            | NativeTensorRole::FfnSharedExpertGate
            | NativeTensorRole::FfnSharedExpertGateInp
            | NativeTensorRole::FfnSharedExpertUp
            | NativeTensorRole::FfnUpExps
            | NativeTensorRole::LinearAttentionALog
            | NativeTensorRole::LinearAttentionConv1d
            | NativeTensorRole::LinearAttentionDtBias
            | NativeTensorRole::LinearAttentionInProjA
            | NativeTensorRole::LinearAttentionInProjB
            | NativeTensorRole::LinearAttentionInProjQkv
            | NativeTensorRole::LinearAttentionInProjZ
            | NativeTensorRole::LinearAttentionNorm
            | NativeTensorRole::LinearAttentionOutProj
            | NativeTensorRole::LmHead
            | NativeTensorRole::Qwen4ExpAttnHcInject
            | NativeTensorRole::Qwen4ExpAttnHcMixDown
            | NativeTensorRole::Qwen4ExpAttnHcMixUp
            | NativeTensorRole::Qwen4ExpAttnHcNorm
            | NativeTensorRole::Qwen4ExpHcMixerMixDown
            | NativeTensorRole::Qwen4ExpHcMixerMixUp
            | NativeTensorRole::Qwen4ExpHcMixerNorm
            | NativeTensorRole::Qwen4ExpIndexerKNorm
            | NativeTensorRole::Qwen4ExpIndexerQNorm
            | NativeTensorRole::Qwen4ExpIndexerQkProj
            | NativeTensorRole::Qwen4ExpMlpHcInject
            | NativeTensorRole::Qwen4ExpMlpHcMixDown
            | NativeTensorRole::Qwen4ExpMlpHcMixUp
            | NativeTensorRole::Qwen4ExpMlpHcNorm
            | NativeTensorRole::Qwen4ExpPleConv1d
            | NativeTensorRole::Qwen4ExpPleHeadOffsets
            | NativeTensorRole::Qwen4ExpPleHeadVocabSizes
            | NativeTensorRole::Qwen4ExpPleKeyProj
            | NativeTensorRole::Qwen4ExpPleMultipliers
            | NativeTensorRole::Qwen4ExpPleNormConv
            | NativeTensorRole::Qwen4ExpPleNormKey
            | NativeTensorRole::Qwen4ExpPleNormQuery
            | NativeTensorRole::Qwen4ExpPleValueProj
            | NativeTensorRole::TokenEmbedding
    )
}

/// Apply the manifest's explicit `weight_sanitize` mode. `HfToMlx` and
/// `HfNormOnly` lift every Flash Next HC/PLE/QSA/main-QK norm gain once
/// (`gain = 1 + stored_delta`); the GDN norm stays plain either way. Only
/// `HfToMlx` also transposes raw `[C, 1, K]` convolution weights (GDN and
/// PLE) to the MLX `[C, K, 1]` layout. `None` assumes already-converted
/// gains and layout and applies no transform.
fn sanitize_norms_and_convs(
    mode: WeightSanitize,
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<(), WeightLoadError> {
    let lift_norms = matches!(mode, WeightSanitize::HfToMlx | WeightSanitize::HfNormOnly);
    let swap_conv = matches!(mode, WeightSanitize::HfToMlx);
    if !lift_norms && !swap_conv {
        return Ok(());
    }
    let one = MlxArray::from_f32_slice(&[1.0_f32]);
    for spec in specs {
        let Some(tensor) = name_map.get(&spec.name).cloned() else {
            continue;
        };
        if lift_norms && is_qwen4_exp_norm_lift_role(spec.role) {
            // Raw HF norms add the delta in FP32 before the activation cast.
            // Keeping this gain in FP32 avoids rounding small BF16 deltas away.
            let lifted = add(&astype(&tensor, MlxDtype::Float32, None), &one, None);
            name_map.insert(spec.name.clone(), lifted);
        } else if swap_conv
            && matches!(
                spec.role,
                NativeTensorRole::LinearAttentionConv1d | NativeTensorRole::Qwen4ExpPleConv1d
            )
        {
            let swapped = contiguous(&transpose(&tensor, &[0, 2, 1], None), None);
            name_map.insert(spec.name.clone(), swapped);
        }
    }
    let refs: Vec<&MlxArray> = name_map.values().collect();
    mlx_sys::try_eval(&refs)
        .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp sanitation evaluation: {e}")))
}

fn is_qwen4_exp_norm_lift_role(role: NativeTensorRole) -> bool {
    matches!(
        role,
        NativeTensorRole::AttentionQNorm
            | NativeTensorRole::AttentionKNorm
            | NativeTensorRole::Qwen4ExpHcMixerNorm
            | NativeTensorRole::Qwen4ExpAttnHcNorm
            | NativeTensorRole::Qwen4ExpMlpHcNorm
            | NativeTensorRole::Qwen4ExpPleNormQuery
            | NativeTensorRole::Qwen4ExpPleNormKey
            | NativeTensorRole::Qwen4ExpPleNormConv
            | NativeTensorRole::Qwen4ExpIndexerQNorm
            | NativeTensorRole::Qwen4ExpIndexerKNorm
    )
}

fn take_plain_required(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    role: NativeTensorRole,
    layer: Option<u32>,
    label: &str,
) -> Result<MlxArray, WeightLoadError> {
    try_take_plain(specs, name_map, role, layer)?
        .ok_or_else(|| WeightLoadError::RoleMissing(format!("{label}[{layer:?}]")))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn take_gated_residual(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer: Option<u32>,
    layout: Qwen4ExpStreamLayout,
    low_rank: usize,
    eps: f32,
    norm_role: NativeTensorRole,
    down_role: NativeTensorRole,
    up_role: NativeTensorRole,
    inject_role: Option<NativeTensorRole>,
    label: &str,
) -> Result<Qwen4ExpGatedResidual, WeightLoadError> {
    let norm_gain = take_plain_required(specs, name_map, norm_role, layer, label)?;
    let read_down = take_weight(specs, name_map, down_role, layer, label)?;
    let read_up = take_weight(specs, name_map, up_role, layer, label)?;
    let write_inject = match inject_role {
        Some(role) => Some(take_weight(specs, name_map, role, layer, label)?),
        None => None,
    };
    Qwen4ExpGatedResidual::new(
        layout,
        low_rank,
        eps,
        Qwen4ExpGatedResidualWeights {
            norm_gain,
            read_down,
            read_up,
            write_inject,
        },
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp {label}[{layer:?}]: {e}")))
}

fn build_gdn(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer: u32,
    hidden: usize,
    config: LinearAttentionConfig,
    eps: f32,
) -> Result<Qwen4ExpGdn, WeightLoadError> {
    let qkv = take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjQkv,
        Some(layer),
        "linear_attention_in_proj_qkv",
    )?;
    let gate = take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjZ,
        Some(layer),
        "linear_attention_in_proj_z",
    )?;
    let decay = take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjA,
        Some(layer),
        "linear_attention_in_proj_a",
    )?;
    let beta = take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionInProjB,
        Some(layer),
        "linear_attention_in_proj_b",
    )?;
    let output = take_weight(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionOutProj,
        Some(layer),
        "linear_attention_out_proj",
    )?;
    let conv = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionConv1d,
        Some(layer),
        "linear_attention_conv1d",
    )?;
    let a_log = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionALog,
        Some(layer),
        "linear_attention_a_log",
    )?;
    let dt_bias = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionDtBias,
        Some(layer),
        "linear_attention_dt_bias",
    )?;
    let norm_gain = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::LinearAttentionNorm,
        Some(layer),
        "linear_attention_norm",
    )?;
    Qwen4ExpGdn::new(
        config,
        hidden,
        eps,
        Qwen4ExpGdnWeights {
            qkv,
            gate,
            decay,
            beta,
            output,
            conv,
            a_log,
            dt_bias,
            norm_gain,
        },
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp layer {layer} gdn: {e}")))
}

pub(super) fn build_qsa_attention(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer: u32,
    main_config: Qwen4ExpAttentionConfig,
    qsa_config: QsaConfig,
) -> Result<Qwen4ExpAttention, WeightLoadError> {
    let qk_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpIndexerQkProj,
        Some(layer),
        "indexer_qk_proj",
    )?;
    let indexer_q_norm = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpIndexerQNorm,
        Some(layer),
        "indexer_q_norm",
    )?;
    let indexer_k_norm = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpIndexerKNorm,
        Some(layer),
        "indexer_k_norm",
    )?;
    let indexer = QsaIndexer::new(
        qsa_config,
        QsaIndexerWeights {
            qk_proj,
            q_norm: indexer_q_norm,
            k_norm: indexer_k_norm,
        },
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp layer {layer} indexer: {e}")))?;

    let q_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionQ,
        Some(layer),
        "attention_q",
    )?;
    let k_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionK,
        Some(layer),
        "attention_k",
    )?;
    let v_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionV,
        Some(layer),
        "attention_v",
    )?;
    let o_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::AttentionO,
        Some(layer),
        "attention_o",
    )?;
    let q_norm = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::AttentionQNorm,
        Some(layer),
        "attention_q_norm",
    )?;
    let k_norm = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::AttentionKNorm,
        Some(layer),
        "attention_k_norm",
    )?;

    Qwen4ExpAttention::new(
        main_config,
        Qwen4ExpAttentionWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        },
        indexer,
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp layer {layer} attention: {e}")))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_moe(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer: u32,
    hidden: usize,
    expert_count: usize,
    expert_intermediate: usize,
    top_k: usize,
    normalize_top_k: bool,
    pager: Option<&Arc<crate::expert_stream::ExpertStackPager>>,
) -> Result<Qwen4ExpMoe, WeightLoadError> {
    let router = take_weight(
        specs,
        name_map,
        NativeTensorRole::FfnGateInp,
        Some(layer),
        "ffn_gate_inp",
    )?;

    let experts = if let Some(pager) = pager {
        Qwen4ExpExpertWeights::Streamed(Arc::new(crate::expert_stream::ExpertLayerSource::new(
            pager.clone(),
            layer,
        )))
    } else {
        let expert_intermediate_i32 = i32::try_from(expert_intermediate).map_err(|_| {
            WeightLoadError::InvalidLayer(format!(
                "qwen4_exp layer {layer}: expert_intermediate_size exceeds i32"
            ))
        })?;

        let has_packed = spec_present(specs, NativeTensorRole::FfnGateUpExpsPacked, Some(layer));
        let (expert_gate, expert_up) = if has_packed {
            let packed = take_weight(
                specs,
                name_map,
                NativeTensorRole::FfnGateUpExpsPacked,
                Some(layer),
                "ffn_gate_up_exps_packed",
            )?;
            reject_expert_linear_bias(&packed, layer, "ffn_gate_up_exps_packed")?;
            split_packed_expert_gate_up(
                &packed,
                expert_intermediate_i32,
                "ffn_gate_up_exps_packed",
            )?
        } else {
            let gate = take_weight(
                specs,
                name_map,
                NativeTensorRole::FfnGateExps,
                Some(layer),
                "ffn_gate_exps",
            )?;
            let up = take_weight(
                specs,
                name_map,
                NativeTensorRole::FfnUpExps,
                Some(layer),
                "ffn_up_exps",
            )?;
            reject_expert_linear_bias(&gate, layer, "ffn_gate_exps")?;
            reject_expert_linear_bias(&up, layer, "ffn_up_exps")?;
            (gate, up)
        };

        let expert_down = take_weight(
            specs,
            name_map,
            NativeTensorRole::FfnDownExps,
            Some(layer),
            "ffn_down_exps",
        )?;
        reject_expert_linear_bias(&expert_down, layer, "ffn_down_exps")?;

        Qwen4ExpExpertWeights::Resident(Box::new(Qwen4ExpResidentExperts {
            gate: expert_gate,
            up: expert_up,
            down: expert_down,
        }))
    };

    let shared_gate = take_weight(
        specs,
        name_map,
        NativeTensorRole::FfnSharedExpertGate,
        Some(layer),
        "ffn_shared_expert_gate",
    )?;
    let shared_up = take_weight(
        specs,
        name_map,
        NativeTensorRole::FfnSharedExpertUp,
        Some(layer),
        "ffn_shared_expert_up",
    )?;
    let shared_down = take_weight(
        specs,
        name_map,
        NativeTensorRole::FfnSharedExpertDown,
        Some(layer),
        "ffn_shared_expert_down",
    )?;
    let shared_router = take_weight(
        specs,
        name_map,
        NativeTensorRole::FfnSharedExpertGateInp,
        Some(layer),
        "ffn_shared_expert_gate_inp",
    )?;

    // Shared-expert width is not assumed equal to the routed width: it is
    // read back from the loaded matrix's own output-row count.
    let shared_rows = shared_gate.weight.shape().first().copied().ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!(
            "qwen4_exp layer {layer}: ffn_shared_expert_gate has no rows"
        ))
    })?;
    let shared_intermediate = usize::try_from(shared_rows).map_err(|_| {
        WeightLoadError::InvalidLayer(format!(
            "qwen4_exp layer {layer}: ffn_shared_expert_gate row count {shared_rows} is invalid"
        ))
    })?;

    Qwen4ExpMoe::new(
        hidden,
        expert_intermediate,
        shared_intermediate,
        expert_count,
        top_k,
        normalize_top_k,
        Qwen4ExpMoeWeights {
            router,
            experts,
            shared_gate,
            shared_up,
            shared_down,
            shared_router,
        },
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp layer {layer} moe: {e}")))
}

fn reject_expert_linear_bias(
    weight: &QuantizedWeight,
    layer: u32,
    label: &str,
) -> Result<(), WeightLoadError> {
    if weight.linear_bias.is_some() {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp layer {layer}: {label} must not carry a dense linear bias"
        )));
    }
    Ok(())
}

/// Split a packed `[experts, 2*intermediate, hidden]` gate/up projection into
/// two `[experts, intermediate, hidden]` views on the output axis (axis 1),
/// including quantized scales/biases. Any other orientation is rejected
/// rather than guessed.
pub(crate) fn split_packed_expert_gate_up(
    packed: &QuantizedWeight,
    expert_intermediate: i32,
    label: &str,
) -> Result<(QuantizedWeight, QuantizedWeight), WeightLoadError> {
    let width = expert_intermediate.checked_mul(2).ok_or_else(|| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp {label}: width overflow"))
    })?;
    let weight_shape = packed.weight.shape();
    if weight_shape.len() != 3 || weight_shape[1] != width {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp {label}: packed weight shape {weight_shape:?} does not match [experts, {width}, hidden]"
        )));
    }
    let build = |start: i32| -> Result<QuantizedWeight, WeightLoadError> {
        let weight =
            slice_expert_output_axis(&packed.weight, width, start, expert_intermediate, label)?;
        let scales = packed
            .scales
            .as_ref()
            .map(|s| slice_expert_output_axis(s, width, start, expert_intermediate, label))
            .transpose()?;
        let biases = packed
            .biases
            .as_ref()
            .map(|b| slice_expert_output_axis(b, width, start, expert_intermediate, label))
            .transpose()?;
        Ok(QuantizedWeight {
            weight,
            scales,
            biases,
            group_size: packed.group_size,
            bits: packed.bits,
            mode: packed.mode.clone(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        })
    };
    Ok((build(0)?, build(expert_intermediate)?))
}

fn slice_expert_output_axis(
    array: &MlxArray,
    expected_width: i32,
    start: i32,
    len: i32,
    label: &str,
) -> Result<MlxArray, WeightLoadError> {
    let shape = array.shape();
    if shape.len() != 3 || shape[1] != expected_width {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp {label}: sidecar shape {shape:?} does not match packed orientation width {expected_width}"
        )));
    }
    let stop = start + len;
    Ok(contiguous(
        &slice(
            array,
            &[0, start, 0],
            &[shape[0], stop, shape[2]],
            &[1, 1, 1],
            None,
        ),
        None,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_ple_operator(
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer: u32,
    stream_count: usize,
    hidden: usize,
    embed_dim: usize,
    kernel: usize,
    dilation: usize,
    eps: f32,
) -> Result<Qwen4ExpPle, WeightLoadError> {
    let norm_key_gain = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpPleNormKey,
        Some(layer),
        "ple_norm_key",
    )?;
    let norm_query_gain = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpPleNormQuery,
        Some(layer),
        "ple_norm_query",
    )?;
    let norm_conv_gain = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpPleNormConv,
        Some(layer),
        "ple_norm_conv",
    )?;
    let key_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpPleKeyProj,
        Some(layer),
        "ple_key_proj",
    )?;
    let value_proj = take_weight(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpPleValueProj,
        Some(layer),
        "ple_value_proj",
    )?;
    let conv_weight = take_plain_required(
        specs,
        name_map,
        NativeTensorRole::Qwen4ExpPleConv1d,
        Some(layer),
        "ple_conv1d",
    )?;
    Qwen4ExpPle::new(
        stream_count,
        hidden,
        embed_dim,
        kernel,
        dilation,
        eps,
        Qwen4ExpPleWeights {
            norm_key_gain,
            norm_query_gain,
            norm_conv_gain,
            key_proj,
            value_proj,
            conv_weight,
        },
    )
    .map_err(|e| WeightLoadError::InvalidLayer(format!("qwen4_exp layer {layer} ple: {e}")))
}

/// Sum the checkpoint's exact I64 metadata buffer to a nonnegative `Vec<u64>`.
/// Evaluates and materializes the array first, per [`MlxArray::data_i64`]'s
/// contract, and rejects a non-Int64 dtype instead of panicking.
fn read_nonneg_i64_vec(array: &MlxArray, label: &str) -> Result<Vec<u64>, WeightLoadError> {
    let array = contiguous(array, None);
    mlx_sys::try_eval(&[&array]).map_err(|e| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp metadata evaluation: {e}"))
    })?;
    if array.dtype() != MlxDtype::Int64 {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp {label}: expected an int64 metadata tensor, got {:?}",
            array.dtype()
        )));
    }
    let data = array.data_i64();
    let mut out = Vec::new();
    out.try_reserve_exact(data.len()).map_err(|_| {
        WeightLoadError::InvalidLayer(format!(
            "qwen4_exp {label}: cannot allocate {} entries",
            data.len()
        ))
    })?;
    for &value in data {
        let v = u64::try_from(value).map_err(|_| {
            WeightLoadError::InvalidLayer(format!(
                "qwen4_exp {label}: value {value} is negative or unrepresentable"
            ))
        })?;
        out.push(v);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn build_ple_bundle(
    root: &Path,
    specs: &[NativeTensorSpec],
    name_map: &mut HashMap<String, MlxArray>,
    layer: u32,
    vocab: u32,
    eos: u32,
    hc_count: usize,
    hidden: usize,
    ple_embed_dim: u32,
    ple_kernel: usize,
    ngram_size: u32,
    heads_per_ngram: u32,
    eps: f32,
) -> Result<Qwen4ExpPleBundle, WeightLoadError> {
    if ngram_size < 2 {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp layer {layer}: ngram_size must be >= 2"
        )));
    }
    let orders = u64::from(ngram_size - 1);
    let ple_heads = orders
        .checked_mul(u64::from(heads_per_ngram))
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| {
            WeightLoadError::InvalidLayer(format!(
                "qwen4_exp layer {layer}: ple head count overflow"
            ))
        })?;
    if ple_heads == 0 || !u64::from(ple_embed_dim).is_multiple_of(u64::from(ple_heads)) {
        return Err(WeightLoadError::InvalidLayer(format!(
            "qwen4_exp layer {layer}: ple_embed_dim {ple_embed_dim} is not divisible by head count {ple_heads}"
        )));
    }
    let row_width = as_usize(ple_embed_dim / ple_heads, "ple ngram row width")?;

    let expected_shards = specs
        .iter()
        .filter(|s| s.role == NativeTensorRole::NgramEmbedding && s.layer_index == Some(layer))
        .count();
    let table = NgramTable::open(
        root,
        specs,
        layer,
        expected_shards,
        row_width,
        DEFAULT_MAX_GATHER_BYTES,
    )
    .map_err(|e| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp layer {layer} ngram table: {e}"))
    })?;

    let multipliers = read_nonneg_i64_vec(
        &take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Qwen4ExpPleMultipliers,
            Some(layer),
            "ple_multipliers",
        )?,
        "ple_multipliers",
    )?;
    let head_offsets = read_nonneg_i64_vec(
        &take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Qwen4ExpPleHeadOffsets,
            Some(layer),
            "ple_head_offsets",
        )?,
        "ple_head_offsets",
    )?;
    let head_sizes = read_nonneg_i64_vec(
        &take_plain_required(
            specs,
            name_map,
            NativeTensorRole::Qwen4ExpPleHeadVocabSizes,
            Some(layer),
            "ple_head_vocab_sizes",
        )?,
        "ple_head_vocab_sizes",
    )?;

    let layout = NgramLayout::new(
        vocab,
        eos,
        multipliers,
        heads_per_ngram as usize,
        head_sizes,
        head_offsets,
        table.rows(),
    )
    .map_err(|e| {
        WeightLoadError::InvalidLayer(format!("qwen4_exp layer {layer} ngram layout: {e}"))
    })?;

    let operator = build_ple_operator(
        specs,
        name_map,
        layer,
        hc_count,
        hidden,
        ple_embed_dim as usize,
        ple_kernel,
        ngram_size as usize,
        eps,
    )?;

    Ok(Qwen4ExpPleBundle {
        operator,
        table,
        layout,
        embedding_width: ple_embed_dim as usize,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    #[ignore = "requires a real Flash Next pack and captured QSA inputs"]
    fn qsa_real_pack_same_input_replay() {
        use crate::model::qwen4_exp::profiling;
        use crate::model::shared::ProjectionBatchPolicy;
        use crate::model::shared::qwen4_exp_attention::Qwen4ExpAttentionCache;
        use sha2::{Digest, Sha256};

        let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR").unwrap());
        let replay = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_QSA_REPLAY_DIR").unwrap());
        let output = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_SMOKE_OUTPUT").unwrap());
        let mut manifest: NativeModelManifest =
            serde_json::from_slice(&std::fs::read(root.join("model-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.model_family, "qwen4_exp");
        let roles = [
            NativeTensorRole::AttentionQ,
            NativeTensorRole::AttentionK,
            NativeTensorRole::AttentionV,
            NativeTensorRole::AttentionO,
            NativeTensorRole::AttentionQNorm,
            NativeTensorRole::AttentionKNorm,
            NativeTensorRole::Qwen4ExpIndexerQkProj,
            NativeTensorRole::Qwen4ExpIndexerQNorm,
            NativeTensorRole::Qwen4ExpIndexerKNorm,
        ];
        manifest
            .tensors
            .retain(|spec| spec.layer_index == Some(3) && roles.contains(&spec.role));
        assert_eq!(manifest.tensors.len(), roles.len());
        for role in roles {
            assert_eq!(
                manifest.tensors.iter().filter(|s| s.role == role).count(),
                1
            );
        }
        let mut name_map = load_resident_tensors(
            &root,
            &root.canonicalize().unwrap(),
            &manifest,
            &HashSet::new(),
        )
        .unwrap();
        let loaded_names: Vec<_> = {
            let mut names: Vec<_> = name_map.keys().cloned().collect();
            names.sort();
            names
        };
        sanitize_norms_and_convs(manifest.weight_sanitize, &manifest.tensors, &mut name_map)
            .unwrap();
        let hidden = manifest.hidden_size as usize;
        let rotary =
            (manifest.attention_head_dim as f32 * manifest.partial_rotary_factor.unwrap()) as usize;
        let rope = manifest.rope_theta.unwrap() as f32;
        let eps = manifest.rms_norm_eps.unwrap();
        let cfg = &manifest.qwen4_exp;
        let main = Qwen4ExpAttentionConfig::new(
            hidden,
            manifest.attention_head_count as usize,
            manifest.kv_head_count as usize,
            manifest.attention_head_dim as usize,
            rotary,
            rope,
            eps,
        )
        .unwrap();
        let indexer = QsaConfig::new(
            cfg.indexer_n_heads.unwrap() as usize,
            cfg.indexer_kv_heads.unwrap() as usize,
            cfg.indexer_head_dim.unwrap() as usize,
            rotary,
            cfg.indexer_compress_ratio.unwrap() as usize,
            cfg.indexer_budget.unwrap() as usize,
            hidden,
            eps,
            rope,
        )
        .unwrap();
        let attention =
            build_qsa_attention(&manifest.tensors, &mut name_map, 3, main, indexer).unwrap();
        assert!(name_map.is_empty());
        let inputs: serde_json::Value =
            serde_json::from_slice(&std::fs::read(replay.join("manifest.json")).unwrap()).unwrap();
        let tokens = inputs["input_ids"].as_array().unwrap().len();
        let chunk = inputs["prefill_chunk_size"].as_u64().unwrap() as usize;
        let singles = 1 + inputs["teacher_force_ids"].as_array().unwrap().len();
        assert!((2..=4096).contains(&tokens) && (1..=128).contains(&chunk));
        assert!(tokens + singles - 1 <= 4096);
        let prefix_steps = (tokens - 1).div_ceil(chunk);
        let stages = replay.join("native-observed/stages");
        let bytes = |array: &MlxArray| {
            let array = astype(array, MlxDtype::Float32, None);
            mlx_sys::try_eval(&[&array]).unwrap();
            array
                .data_f32()
                .iter()
                .flat_map(|value| {
                    assert!(value.is_finite());
                    value.to_le_bytes()
                })
                .collect::<Vec<_>>()
        };
        let fingerprint = |array: &MlxArray| {
            serde_json::json!({"shape": array.shape(), "dtype": format!("{:?}", array.dtype()),
                "sha256": format!("{:x}", Sha256::digest(bytes(array)))})
        };
        let mut cache = Qwen4ExpAttentionCache::empty();
        let mut position = 0;
        let mut records = Vec::new();
        for step in 0..prefix_steps + singles {
            let name = format!("forward-{}", step + 1);
            let meta: serde_json::Value = serde_json::from_slice(
                &std::fs::read(stages.join(format!("{name}-attention_hc_read.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(meta["layer"], 3);
            assert_eq!(meta["dtype"], "Bfloat16");
            let seq = if step < prefix_steps {
                chunk.min(tokens - 1 - position)
            } else {
                1
            };
            assert_eq!(meta["shape"], serde_json::json!([1, seq, hidden]));
            let payload =
                std::fs::read(stages.join(format!("{name}-attention_hc_read.f32le"))).unwrap();
            assert_eq!(payload.len(), seq * hidden * 4);
            let values: Vec<f32> = payload
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            let input = mlx_sys::reshape(
                &MlxArray::from_f32_slice(&values),
                &[1, seq as i32, hidden as i32],
                None,
            );
            let input = astype(&input, MlxDtype::Bfloat16, None);
            assert_eq!(bytes(&input), payload);
            profiling::begin_forward_dump();
            profiling::layer(3);
            let result = attention
                .forward(&input, &cache, position, ProjectionBatchPolicy::Shared)
                .unwrap();
            let delta = bytes(result.delta());
            assert_eq!(
                delta,
                std::fs::read(stages.join(format!("{name}-qsa_output.f32le"))).unwrap(),
                "full-model output at step {step}"
            );
            position += seq;
            cache = result.into_next_state();
            assert_eq!(cache.token_count().unwrap(), position);
            assert_eq!(cache.index().token_count().unwrap(), position);
            records.push(serde_json::json!({"step": step, "position": position,
                "output_sha256": format!("{:x}", Sha256::digest(&delta)),
                "input_sha256": format!("{:x}", Sha256::digest(&payload)),
                "cache": [fingerprint(cache.keys().unwrap()), fingerprint(cache.values().unwrap()),
                    fingerprint(cache.index().keys().unwrap())]}));
            eprintln!("qsa replay position={position}");
        }
        assert_eq!(position, tokens + singles - 1);
        std::fs::write(
            output,
            serde_json::to_vec_pretty(&serde_json::json!({"completed": true,
                "qualification": false, "loaded_names": loaded_names, "records": records,
                "full_model_outputs_exact": true, "peak_mlx_bytes": mlx_sys::get_peak_memory()}))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn paging_plan_binds_native_metadata_without_reading_payloads() {
        use ax_engine_core::{NativeTensorDataType, NativeTensorQuantization};
        let root = std::env::temp_dir().join(format!(
            "ax-flash-paging-contract-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        for name in ["experts.safetensors", "other.safetensors"] {
            // Admission must not attempt to parse or evaluate these payloads.
            std::fs::write(root.join(name), b"metadata-only fixture").unwrap();
        }
        let canonical = root.canonicalize().unwrap();
        let specs: Vec<_> = [
            ("gate", NativeTensorRole::FfnGateExps),
            ("up", NativeTensorRole::FfnUpExps),
            ("down", NativeTensorRole::FfnDownExps),
        ]
        .into_iter()
        .map(|(name, role)| NativeTensorSpec {
            name: format!("layers.0.mlp.{name}.weight"),
            role,
            layer_index: Some(0),
            dtype: NativeTensorDataType::U32,
            source_tensor_type: None,
            source_quantized: true,
            quantization: Some(NativeTensorQuantization {
                bits: 4,
                group_size: 32,
                mode: "affine".into(),
            }),
            quantized_source: None,
            shape: vec![4, 64, 8],
            file: "experts.safetensors".into(),
            offset_bytes: 0,
            length_bytes: 8192,
        })
        .collect();
        let plan = crate::expert_stream::infer_layer_stack_manifest(&specs, 2).unwrap();
        assert_eq!(
            validate_expert_paging_contract(&root, &canonical, &specs, 2, &plan).unwrap(),
            HashSet::from([0])
        );
        for violation in [
            "bits",
            "group",
            "layer",
            "count",
            "projection",
            "partial",
            "duplicate",
            "file",
            "unknown",
            "mode",
            "topk",
        ] {
            let mut changed = plan.clone();
            let mut changed_specs = specs.clone();
            match violation {
                "bits" => changed.tensors[0].bits = 8,
                "group" => changed.tensors[0].group_size = 64,
                "layer" => changed.tensors[0].layer = 1,
                "count" => changed.tensors[0].num_experts = 8,
                "projection" => {
                    changed.tensors[0].parsed_proj = Some(crate::expert_stream::ExpertProj::Up)
                }
                "partial" => {
                    changed.tensors.pop();
                }
                "duplicate" => changed.tensors.push(changed.tensors[0].clone()),
                "file" => changed.tensors[0].file = "other.safetensors".into(),
                "unknown" => changed.tensors[0].name = "unrecognized.weight".into(),
                "mode" => changed_specs[0].quantization.as_mut().unwrap().mode = "mxfp4".into(),
                "topk" => changed.experts_per_tok = 1,
                _ => unreachable!(),
            }
            assert!(
                validate_expert_paging_contract(&root, &canonical, &changed_specs, 2, &changed)
                    .is_err(),
                "accepted {violation}"
            );
        }
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"layers.0.mlp.gate.scales": "other.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(
            validate_expert_paging_contract(&root, &canonical, &specs, 2, &plan).is_err(),
            "cross-file scale needs an explicit paging entry"
        );
        let mut explicit = plan;
        let mut sidecar = explicit.tensors[0].clone();
        sidecar.name = "layers.0.mlp.gate.scales".into();
        sidecar.file = "other.safetensors".into();
        explicit.tensors.push(sidecar);
        assert!(validate_expert_paging_contract(&root, &canonical, &specs, 2, &explicit).is_ok());
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"layers.0.mlp.gate.bias": "other.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        explicit.tensors.pop();
        assert!(
            validate_expert_paging_contract(&root, &canonical, &specs, 2, &explicit).is_err(),
            "an unsupported indexed dense bias must not be silently skipped"
        );
    }

    #[test]
    fn hf_norm_only_preserves_small_bf16_deltas_and_convolution_layout() {
        use ax_engine_core::NativeTensorDataType;
        let specs: Vec<_> = [
            ("hc", NativeTensorRole::Qwen4ExpAttnHcNorm, vec![2]),
            ("gdn", NativeTensorRole::LinearAttentionNorm, vec![2]),
            (
                "conv",
                NativeTensorRole::LinearAttentionConv1d,
                vec![2, 3, 1],
            ),
        ]
        .into_iter()
        .map(|(name, role, shape)| NativeTensorSpec {
            name: name.into(),
            role,
            layer_index: Some(0),
            dtype: NativeTensorDataType::Bf16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: "unused.safetensors".into(),
            offset_bytes: 0,
            length_bytes: 0,
        })
        .collect();
        let raw = astype(
            &MlxArray::from_f32_slice(&[0.003, -0.003]),
            MlxDtype::Bfloat16,
            None,
        );
        let gdn = astype(
            &MlxArray::from_f32_slice(&[0.75, 1.25]),
            MlxDtype::Bfloat16,
            None,
        );
        let conv = mlx_sys::reshape(
            &MlxArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            &[2, 3, 1],
            None,
        );
        let mut tensors = HashMap::from([
            ("hc".into(), raw.clone()),
            ("gdn".into(), gdn.clone()),
            ("conv".into(), conv),
        ]);
        sanitize_norms_and_convs(WeightSanitize::HfNormOnly, &specs, &mut tensors).unwrap();
        assert_eq!(tensors["hc"].dtype(), MlxDtype::Float32);
        // These exact BF16 deltas would disappear or change if +1 were
        // rounded back into BF16 during loading.
        assert_eq!(
            tensors["hc"].data_f32(),
            &[f32::from_bits(0x3f806280), f32::from_bits(0x3f7f3b00)]
        );
        assert_eq!(tensors["conv"].shape(), [2, 3, 1]);
        let actual_gdn = astype(&tensors["gdn"], MlxDtype::Float32, None);
        eval(&[&actual_gdn]);
        assert_eq!(actual_gdn.data_f32(), &[0.75, 1.25]);
        let mut untouched = HashMap::from([("hc".into(), raw)]);
        sanitize_norms_and_convs(WeightSanitize::None, &specs, &mut untouched).unwrap();
        assert_eq!(untouched["hc"].dtype(), MlxDtype::Bfloat16);
    }

    #[test]
    fn ple_eos_uses_text_contract_instead_of_generation_stop_list() {
        let root = std::env::temp_dir().join(format!(
            "ax-qwen4-eos-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            br#"{"eos_token_id":[248046,248044],"text_config":{"eos_token_id":248044}}"#,
        )
        .unwrap();
        assert_eq!(resolve_eos_token(&root, 248320).unwrap(), 248044);
        assert!(resolve_eos_token(&root, 248044).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
