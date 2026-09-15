//! Bounded loader for the Flash Next (`qwen4_exp`) MTP sidecar.
//!
//! The sidecar is one dedicated QSA/MoE/dual-HC layer plus its own final HC
//! mixer and the pre-FC norm/projection pairs. Admission is metadata-first:
//! the exact supported tensor names, shapes, dtypes, and file bindings are
//! checked against the validated manifest geometry before any payload is
//! opened, and only the sidecar's own tensors are ever read. Trunk and n-gram
//! table tensors are never loaded again; the graph shares the trunk embedding
//! and `lm_head` handles.
//!
//! This first contract accepts resident BF16/F16/F32 tensors only. Quantized
//! sidecars are rejected rather than guessed. The MTP forward/combiner is not
//! owned here, and nothing attaches this graph to a runner automatically.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use ax_engine_core::{
    NativeModelManifest, NativeTensorDataType, NativeTensorRole, NativeTensorSpec,
};
use mlx_sys::{MlxArray, MlxDtype, SafetensorsNameFilter, add, astype, load_safetensors_filtered};

use crate::model::shared::qwen4_exp_attention::Qwen4ExpAttentionConfig;
use crate::qwen4_exp_qsa::QsaConfig;

use super::qwen4_exp::{
    Qwen4ExpAttentionBranch, Qwen4ExpLayerKind, Qwen4ExpLayerWeights, Qwen4ExpWeights, as_usize,
    build_moe, build_qsa_attention, layer_kind, read_weight_index, require_f32, require_u32,
    resolve_in_root, take_gated_residual,
};
use super::{QuantizedWeight, WeightLoadError, take_named_weight};

const MTP_SIDECAR_FILE: &str = "mtp.safetensors";
const MTP_RUNTIME_FILE: &str = "mtplx_runtime.json";
const MTP_HEADER_MAX_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) struct Qwen4ExpMtpWeights {
    pub pre_fc_norm_embedding: MlxArray,
    pub pre_fc_norm_hidden: MlxArray,
    pub fc_embedding: QuantizedWeight,
    pub fc_hidden: QuantizedWeight,
    pub graph: Qwen4ExpWeights,
    pub rms_eps: f32,
}

/// Explicit `mtp_norm_layout` declaration from `mtplx_runtime.json`. There is
/// no auto-detection for this contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qwen4ExpMtpNormLayout {
    /// Stored norms are zero-centred HF deltas; gains become `1 + delta` in FP32.
    RawHfDelta,
    /// Stored norms are already multiplicative gains.
    MlxMultiplier,
}

#[derive(Clone, Copy)]
enum ShapeRule {
    Hidden,
    PackedHidden,
    HiddenSquare,
    /// Logical shape of the same role in the validated trunk manifest.
    TrunkAnalog,
    Router,
    PackedGateUp,
    ExpertDown,
}

/// `(name, role, is_layer_local, shape rule, is_rms_norm_gain)`.
const MTP_TENSORS: &[(&str, NativeTensorRole, bool, ShapeRule, bool)] = {
    use NativeTensorRole as R;
    use ShapeRule as S;
    &[
        (
            "mtp.pre_fc_norm_embedding.weight",
            R::Other,
            false,
            S::Hidden,
            true,
        ),
        (
            "mtp.pre_fc_norm_hidden.weight",
            R::Other,
            false,
            S::PackedHidden,
            true,
        ),
        (
            "mtp.fc_embedding.weight",
            R::Other,
            false,
            S::HiddenSquare,
            false,
        ),
        (
            "mtp.fc_hidden.weight",
            R::Other,
            false,
            S::HiddenSquare,
            false,
        ),
        (
            "mtp.hyper_connection_mixer.hc_norm.weight",
            R::Qwen4ExpHcMixerNorm,
            false,
            S::TrunkAnalog,
            true,
        ),
        (
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            R::Qwen4ExpHcMixerMixDown,
            false,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            R::Qwen4ExpHcMixerMixUp,
            false,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
            R::Qwen4ExpAttnHcNorm,
            true,
            S::TrunkAnalog,
            true,
        ),
        (
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
            R::Qwen4ExpAttnHcMixDown,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
            R::Qwen4ExpAttnHcMixUp,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
            R::Qwen4ExpAttnHcInject,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
            R::Qwen4ExpMlpHcNorm,
            true,
            S::TrunkAnalog,
            true,
        ),
        (
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
            R::Qwen4ExpMlpHcMixDown,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
            R::Qwen4ExpMlpHcMixUp,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
            R::Qwen4ExpMlpHcInject,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.self_attn.q_proj.weight",
            R::AttentionQ,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.self_attn.k_proj.weight",
            R::AttentionK,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.self_attn.v_proj.weight",
            R::AttentionV,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.self_attn.o_proj.weight",
            R::AttentionO,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.self_attn.q_norm.weight",
            R::AttentionQNorm,
            true,
            S::TrunkAnalog,
            true,
        ),
        (
            "mtp.layers.0.self_attn.k_norm.weight",
            R::AttentionKNorm,
            true,
            S::TrunkAnalog,
            true,
        ),
        (
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
            R::Qwen4ExpIndexerQkProj,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
            R::Qwen4ExpIndexerQNorm,
            true,
            S::TrunkAnalog,
            true,
        ),
        (
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
            R::Qwen4ExpIndexerKNorm,
            true,
            S::TrunkAnalog,
            true,
        ),
        (
            "mtp.layers.0.mlp.gate.weight",
            R::FfnGateInp,
            true,
            S::Router,
            false,
        ),
        (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            R::FfnGateUpExpsPacked,
            true,
            S::PackedGateUp,
            false,
        ),
        (
            "mtp.layers.0.mlp.experts.down_proj",
            R::FfnDownExps,
            true,
            S::ExpertDown,
            false,
        ),
        (
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            R::FfnSharedExpertGate,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            R::FfnSharedExpertUp,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
            R::FfnSharedExpertDown,
            true,
            S::TrunkAnalog,
            false,
        ),
        (
            "mtp.layers.0.mlp.shared_expert_gate.weight",
            R::FfnSharedExpertGateInp,
            true,
            S::TrunkAnalog,
            false,
        ),
    ]
};

const PRE_FC_NORM_EMBEDDING: &str = "mtp.pre_fc_norm_embedding.weight";
const PRE_FC_NORM_HIDDEN: &str = "mtp.pre_fc_norm_hidden.weight";
const FC_EMBEDDING: &str = "mtp.fc_embedding.weight";
const FC_HIDDEN: &str = "mtp.fc_hidden.weight";

struct ExpectedTensor {
    name: &'static str,
    role: NativeTensorRole,
    layer_index: Option<u32>,
    shape: Vec<u64>,
    norm: bool,
}

/// Manifest-derived dimensions. Trunk analogs are logical (unpacked) shapes
/// so a quantized trunk still yields the dense sidecar geometry.
struct MtpGeometry {
    hidden: u64,
    expert_count: u64,
    expert_intermediate: u64,
    analogs: Vec<(NativeTensorRole, Vec<u64>)>,
}

impl MtpGeometry {
    fn from_manifest(manifest: &NativeModelManifest) -> Result<Self, WeightLoadError> {
        let attention_layer = (0..manifest.layer_count)
            .find(|&layer| {
                matches!(
                    layer_kind(manifest, layer),
                    Ok(Qwen4ExpLayerKind::Attention)
                )
            })
            .ok_or_else(|| {
                invalid("trunk manifest has no attention layer to derive geometry from".into())
            })?;
        let mut analogs = Vec::new();
        for &(name, role, layer_local, rule, _) in MTP_TENSORS {
            if !matches!(rule, ShapeRule::TrunkAnalog) {
                continue;
            }
            let layer = layer_local.then_some(attention_layer);
            let spec = manifest
                .tensors
                .iter()
                .find(|spec| spec.role == role && spec.layer_index == layer)
                .ok_or_else(|| {
                    invalid(format!(
                        "trunk manifest lacks the geometry analog for {name}"
                    ))
                })?;
            analogs.push((role, logical_shape(spec)?));
        }
        Ok(Self {
            hidden: u64::from(manifest.hidden_size),
            expert_count: u64::from(require_u32(manifest.moe.expert_count, "moe.expert_count")?),
            expert_intermediate: u64::from(require_u32(
                manifest.moe.expert_intermediate_size,
                "moe.expert_intermediate_size",
            )?),
            analogs,
        })
    }

    fn expected(&self) -> Result<Vec<ExpectedTensor>, WeightLoadError> {
        let (h, e, i) = (self.hidden, self.expert_count, self.expert_intermediate);
        MTP_TENSORS
            .iter()
            .map(|&(name, role, layer_local, rule, norm)| {
                let shape = match rule {
                    ShapeRule::Hidden => vec![h],
                    ShapeRule::PackedHidden => self
                        .analogs
                        .iter()
                        .find(|(role, _)| *role == NativeTensorRole::Qwen4ExpHcMixerNorm)
                        .map(|(_, shape)| shape.clone())
                        .ok_or_else(|| invalid("missing packed hidden geometry".into()))?,
                    ShapeRule::HiddenSquare => vec![h, h],
                    ShapeRule::Router => vec![e, h],
                    ShapeRule::PackedGateUp => vec![e, 2 * i, h],
                    ShapeRule::ExpertDown => vec![e, h, i],
                    ShapeRule::TrunkAnalog => self
                        .analogs
                        .iter()
                        .find(|(analog, _)| *analog == role)
                        .map(|(_, shape)| shape.clone())
                        .ok_or_else(|| invalid(format!("missing geometry analog for {name}")))?,
                };
                Ok(ExpectedTensor {
                    name,
                    role,
                    layer_index: layer_local.then_some(0),
                    shape,
                    norm,
                })
            })
            .collect()
    }
}

pub(crate) fn load(
    root: &Path,
    manifest: &NativeModelManifest,
    trunk: &Qwen4ExpWeights,
) -> Result<Qwen4ExpMtpWeights, WeightLoadError> {
    if manifest.model_family != "qwen4_exp" {
        return Err(invalid(format!(
            "requires model_family qwen4_exp, got {:?}",
            manifest.model_family
        )));
    }
    if trunk.layout.hidden_size() != manifest.hidden_size as usize
        || Some(trunk.layout.stream_count() as u32) != manifest.qwen4_exp.hc_count
    {
        return Err(invalid(
            "trunk stream layout disagrees with manifest".into(),
        ));
    }
    let canonical_root = root.canonicalize().map_err(|e| {
        WeightLoadError::FileMissing(format!("cannot resolve root {}: {e}", root.display()))
    })?;
    let norm_layout = read_norm_layout(root, &canonical_root)?;
    let expected = MtpGeometry::from_manifest(manifest)?.expected()?;
    let sidecar = resolve_in_root(root, &canonical_root, Path::new(MTP_SIDECAR_FILE))?;
    let manifest_specs: Vec<NativeTensorSpec> = manifest
        .tensors
        .iter()
        .filter(|spec| spec.name.starts_with("mtp."))
        .cloned()
        .collect();
    let declared = read_sidecar_header_specs(&sidecar)?;
    let index = read_weight_index(root, &canonical_root)?;
    if !manifest_specs.is_empty() {
        let bound_manifest = bind_declared_specs(
            &expected,
            &manifest_specs,
            root,
            &canonical_root,
            &sidecar,
            index.as_ref(),
        )?;
        for spec in &bound_manifest {
            let actual = declared
                .iter()
                .find(|actual| actual.name == spec.name)
                .ok_or_else(|| invalid(format!("sidecar header is missing {}", spec.name)))?;
            if (
                actual.dtype,
                &actual.shape,
                actual.offset_bytes,
                actual.length_bytes,
            ) != (
                spec.dtype,
                &spec.shape,
                spec.offset_bytes,
                spec.length_bytes,
            ) {
                return Err(invalid(format!(
                    "sidecar header disagrees with manifest for {}",
                    spec.name
                )));
            }
        }
    }
    let specs = bind_declared_specs(
        &expected,
        &declared,
        root,
        &canonical_root,
        &sidecar,
        index.as_ref(),
    )?;

    // Payload IO starts only after the full metadata contract is admitted.
    let names: HashSet<String> = specs.iter().map(|spec| spec.name.clone()).collect();
    let mut name_map = load_safetensors_filtered(&sidecar, SafetensorsNameFilter::Keep(&names))
        .map_err(WeightLoadError::FileMissing)?;
    mlx_sys::try_eval(&name_map.values().collect::<Vec<_>>())
        .map_err(|e| WeightLoadError::InvalidLayer(format!("Flash Next MTP evaluation: {e}")))?;
    validate_loaded(&specs, &name_map)?;
    apply_norm_layout(norm_layout, &expected, &mut name_map)?;

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
    let hc_lowrank = as_usize(require_u32(cfg.hc_lowrank, "hc_lowrank")?, "hc_lowrank")?;
    let config_usize =
        |value: Option<u32>, field: &str| as_usize(require_u32(value, field)?, field);
    let qsa_config = QsaConfig::new(
        config_usize(cfg.indexer_n_heads, "indexer_n_heads")?,
        config_usize(cfg.indexer_kv_heads, "indexer_kv_heads")?,
        config_usize(cfg.indexer_head_dim, "indexer_head_dim")?,
        rotary_dim,
        config_usize(cfg.indexer_compress_ratio, "indexer_compress_ratio")?,
        config_usize(cfg.indexer_budget, "indexer_budget")?,
        hidden,
        rms_eps,
        rope_base,
    )
    .map_err(|e| invalid(format!("qsa config: {e}")))?;
    let attention_config = Qwen4ExpAttentionConfig::new(
        hidden,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_base,
        rms_eps,
    )
    .map_err(|e| invalid(format!("attention config: {e}")))?;

    let layout = trunk.layout;
    let specs = specs.as_slice();
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
        "mtp_hc_mixer",
    )?;
    let attention_hc = take_gated_residual(
        specs,
        &mut name_map,
        Some(0),
        layout,
        hc_lowrank,
        rms_eps,
        NativeTensorRole::Qwen4ExpAttnHcNorm,
        NativeTensorRole::Qwen4ExpAttnHcMixDown,
        NativeTensorRole::Qwen4ExpAttnHcMixUp,
        Some(NativeTensorRole::Qwen4ExpAttnHcInject),
        "mtp_attn_hc",
    )?;
    let mlp_hc = take_gated_residual(
        specs,
        &mut name_map,
        Some(0),
        layout,
        hc_lowrank,
        rms_eps,
        NativeTensorRole::Qwen4ExpMlpHcNorm,
        NativeTensorRole::Qwen4ExpMlpHcMixDown,
        NativeTensorRole::Qwen4ExpMlpHcMixUp,
        Some(NativeTensorRole::Qwen4ExpMlpHcInject),
        "mtp_mlp_hc",
    )?;
    let attention = Qwen4ExpAttentionBranch::Qsa(build_qsa_attention(
        specs,
        &mut name_map,
        0,
        attention_config,
        qsa_config,
    )?);
    let moe = build_moe(
        specs,
        &mut name_map,
        0,
        hidden,
        config_usize(manifest.moe.expert_count, "moe.expert_count")?,
        config_usize(
            manifest.moe.expert_intermediate_size,
            "moe.expert_intermediate_size",
        )?,
        config_usize(manifest.moe.experts_per_token, "moe.experts_per_token")?,
        manifest.moe_norm_topk_prob,
        None,
    )?;
    let pre_fc_norm_embedding = take_required_plain(&mut name_map, PRE_FC_NORM_EMBEDDING)?;
    let pre_fc_norm_hidden = take_required_plain(&mut name_map, PRE_FC_NORM_HIDDEN)?;
    let fc_embedding = take_named_weight(specs, &mut name_map, FC_EMBEDDING)?;
    let fc_hidden = take_named_weight(specs, &mut name_map, FC_HIDDEN)?;
    if let Some(name) = name_map.keys().next() {
        return Err(invalid(format!(
            "tensor {name} was loaded but not consumed"
        )));
    }

    Ok(Qwen4ExpMtpWeights {
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        fc_embedding,
        fc_hidden,
        graph: Qwen4ExpWeights {
            token_embedding: trunk.token_embedding.clone(),
            lm_head: trunk.lm_head.clone(),
            layout,
            mixer,
            layers: vec![Qwen4ExpLayerWeights {
                attention_hc,
                mlp_hc,
                attention,
                moe,
                ple: None,
            }],
            expert_stream: None,
        },
        rms_eps,
    })
}

fn invalid(message: String) -> WeightLoadError {
    WeightLoadError::InvalidLayer(format!("qwen4_exp mtp: {message}"))
}

fn take_required_plain(
    name_map: &mut HashMap<String, MlxArray>,
    name: &str,
) -> Result<MlxArray, WeightLoadError> {
    name_map
        .remove(name)
        .ok_or_else(|| WeightLoadError::TensorMissing(name.to_string()))
}

/// Unpacked shape of a trunk tensor. MLX affine packing requires the input
/// width to be a multiple of the group size, so `packed * 32 / bits` is exact.
fn logical_shape(spec: &NativeTensorSpec) -> Result<Vec<u64>, WeightLoadError> {
    let mut shape = spec.shape.clone();
    match &spec.quantization {
        Some(quant) => {
            let bits = u64::from(quant.bits);
            let last = shape
                .last_mut()
                .ok_or_else(|| invalid(format!("trunk analog {} has an empty shape", spec.name)))?;
            let widened = last.checked_mul(32).filter(|w| bits != 0 && w % bits == 0);
            *last = widened.map(|w| w / bits).ok_or_else(|| {
                invalid(format!(
                    "trunk analog {} packed width is not exact for {bits} bits",
                    spec.name
                ))
            })?;
        }
        None if spec.source_quantized => {
            return Err(invalid(format!(
                "trunk analog {} is source-quantized without metadata",
                spec.name
            )));
        }
        None => {}
    }
    Ok(shape)
}

fn parse_norm_layout(value: &serde_json::Value) -> Result<Qwen4ExpMtpNormLayout, WeightLoadError> {
    match value.get("mtp_norm_layout") {
        Some(serde_json::Value::String(s)) if s == "raw_hf_delta" => {
            Ok(Qwen4ExpMtpNormLayout::RawHfDelta)
        }
        Some(serde_json::Value::String(s)) if s == "mlx_multiplier" => {
            Ok(Qwen4ExpMtpNormLayout::MlxMultiplier)
        }
        Some(other) => Err(invalid(format!(
            "{MTP_RUNTIME_FILE} mtp_norm_layout must be raw_hf_delta or mlx_multiplier, got {other}"
        ))),
        None => Err(invalid(format!(
            "{MTP_RUNTIME_FILE} must explicitly declare mtp_norm_layout"
        ))),
    }
}

fn read_norm_layout(
    root: &Path,
    canonical_root: &Path,
) -> Result<Qwen4ExpMtpNormLayout, WeightLoadError> {
    let path = resolve_in_root(root, canonical_root, Path::new(MTP_RUNTIME_FILE))?;
    let bytes = std::fs::read(&path).map_err(|e| {
        WeightLoadError::FileMissing(format!("cannot read {}: {e}", path.display()))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("cannot parse {}: {e}", path.display())))?;
    parse_norm_layout(&value)
}

/// Read only the safetensors JSON header (names, dtypes, shapes, offsets).
fn read_sidecar_header_specs(path: &Path) -> Result<Vec<NativeTensorSpec>, WeightLoadError> {
    let io_error =
        |e: std::io::Error| WeightLoadError::FileMissing(format!("{}: {e}", path.display()));
    let mut file = std::fs::File::open(path).map_err(io_error)?;
    let file_bytes = file.metadata().map_err(io_error)?.len();
    let mut len = [0_u8; 8];
    file.read_exact(&mut len).map_err(io_error)?;
    let header_len = u64::from_le_bytes(len);
    if header_len == 0 || header_len > MTP_HEADER_MAX_BYTES {
        return Err(invalid(format!(
            "sidecar header length {header_len} is outside 1..={MTP_HEADER_MAX_BYTES}"
        )));
    }
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header).map_err(io_error)?;
    let value: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| invalid(format!("cannot parse sidecar header: {e}")))?;
    let entries = value
        .as_object()
        .ok_or_else(|| invalid("sidecar header is not an object".into()))?;
    let mut specs = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let malformed = || invalid(format!("sidecar header entry {name} is malformed"));
        let dtype = match entry.get("dtype").and_then(|v| v.as_str()) {
            Some("BF16") => NativeTensorDataType::Bf16,
            Some("F16") => NativeTensorDataType::F16,
            Some("F32") => NativeTensorDataType::F32,
            Some("U32") => NativeTensorDataType::U32,
            Some("U8") => NativeTensorDataType::U8,
            Some("I8") => NativeTensorDataType::I8,
            Some("I64") => NativeTensorDataType::I64,
            Some(other) => {
                return Err(invalid(format!(
                    "sidecar tensor {name} has unsupported dtype {other}"
                )));
            }
            None => return Err(malformed()),
        };
        let shape = entry
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(malformed)?
            .iter()
            .map(|dim| dim.as_u64().ok_or_else(malformed))
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(malformed)?;
        let start = offsets[0].as_u64().ok_or_else(malformed)?;
        let end = offsets[1]
            .as_u64()
            .filter(|&end| end >= start)
            .ok_or_else(malformed)?;
        let offset_bytes = (8 + header_len).checked_add(start).ok_or_else(malformed)?;
        if offset_bytes
            .checked_add(end - start)
            .is_none_or(|end| end > file_bytes)
        {
            return Err(invalid(format!(
                "sidecar tensor {name} extends past its file"
            )));
        }
        specs.push(NativeTensorSpec {
            name: name.clone(),
            role: NativeTensorRole::Other,
            layer_index: None,
            dtype,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from(MTP_SIDECAR_FILE),
            offset_bytes,
            length_bytes: end - start,
        });
    }
    Ok(specs)
}

/// Admit the exact MTP tensor set and rebind each declared spec to its graph
/// role. Never opens payloads.
fn bind_declared_specs(
    expected: &[ExpectedTensor],
    declared: &[NativeTensorSpec],
    root: &Path,
    canonical_root: &Path,
    sidecar: &Path,
    index: Option<&HashMap<String, PathBuf>>,
) -> Result<Vec<NativeTensorSpec>, WeightLoadError> {
    let mut by_name = HashMap::with_capacity(declared.len());
    for spec in declared {
        if by_name.insert(spec.name.as_str(), spec).is_some() {
            return Err(invalid(format!("duplicate tensor {}", spec.name)));
        }
    }
    let supported: HashSet<&str> = expected.iter().map(|e| e.name).collect();
    let mut extras: Vec<&str> = by_name
        .keys()
        .copied()
        .filter(|name| !supported.contains(name))
        .collect();
    if !extras.is_empty() {
        extras.sort_unstable();
        return Err(invalid(format!("unsupported sidecar tensors {extras:?}")));
    }
    let missing: Vec<&str> = expected
        .iter()
        .map(|e| e.name)
        .filter(|name| !by_name.contains_key(name))
        .collect();
    if !missing.is_empty() {
        return Err(invalid(format!("missing sidecar tensors {missing:?}")));
    }
    let mut bound = Vec::with_capacity(expected.len());
    for tensor in expected {
        let spec = by_name[tensor.name];
        if spec.source_quantized || spec.quantization.is_some() || spec.quantized_source.is_some() {
            return Err(invalid(format!(
                "{} is quantized; this contract requires resident BF16/F16/F32",
                tensor.name
            )));
        }
        if !matches!(
            spec.dtype,
            NativeTensorDataType::Bf16 | NativeTensorDataType::F16 | NativeTensorDataType::F32
        ) {
            return Err(invalid(format!(
                "{} must be BF16/F16/F32, got {:?}",
                tensor.name, spec.dtype
            )));
        }
        if spec.shape != tensor.shape {
            return Err(invalid(format!(
                "{} shape {:?} disagrees with manifest geometry {:?}",
                tensor.name, spec.shape, tensor.shape
            )));
        }
        if resolve_in_root(root, canonical_root, &spec.file)? != sidecar {
            return Err(invalid(format!(
                "{} is bound to {} instead of {MTP_SIDECAR_FILE}",
                tensor.name,
                spec.file.display()
            )));
        }
        if let Some(index) = index {
            let base = tensor.name.strip_suffix(".weight").unwrap_or(tensor.name);
            if index.get(tensor.name).is_some_and(|file| file != sidecar) {
                return Err(invalid(format!(
                    "{} conflicts with its file index",
                    tensor.name
                )));
            }
            if [".scales", ".biases"]
                .iter()
                .any(|suffix| index.contains_key(&format!("{base}{suffix}")))
            {
                return Err(invalid(format!(
                    "{} has quantization sidecars in the file index",
                    tensor.name
                )));
            }
        }
        let mut spec = spec.clone();
        spec.role = tensor.role;
        spec.layer_index = tensor.layer_index;
        bound.push(spec);
    }
    Ok(bound)
}

fn validate_loaded(
    specs: &[NativeTensorSpec],
    name_map: &HashMap<String, MlxArray>,
) -> Result<(), WeightLoadError> {
    if name_map.len() != specs.len() {
        return Err(invalid(format!(
            "loaded {} tensors, contract declares {}",
            name_map.len(),
            specs.len()
        )));
    }
    for spec in specs {
        let array = name_map
            .get(&spec.name)
            .ok_or_else(|| WeightLoadError::TensorMissing(spec.name.clone()))?;
        let dtype = match spec.dtype {
            NativeTensorDataType::Bf16 => MlxDtype::Bfloat16,
            NativeTensorDataType::F16 => MlxDtype::Float16,
            NativeTensorDataType::F32 => MlxDtype::Float32,
            _ => return Err(invalid(format!("{} has an unsupported dtype", spec.name))),
        };
        let shape: Vec<u64> = array.shape().iter().map(|&d| d as u64).collect();
        if array.dtype() != dtype || shape != spec.shape {
            return Err(invalid(format!(
                "{} loaded as {:?} {shape:?}, declared {dtype:?} {:?}",
                spec.name,
                array.dtype(),
                spec.shape
            )));
        }
    }
    Ok(())
}

/// Every RMS norm gain in this contract follows the one runtime declaration.
/// Raw deltas are shifted in FP32 and kept FP32 so small BF16 deltas survive.
fn apply_norm_layout(
    layout: Qwen4ExpMtpNormLayout,
    expected: &[ExpectedTensor],
    name_map: &mut HashMap<String, MlxArray>,
) -> Result<(), WeightLoadError> {
    if layout != Qwen4ExpMtpNormLayout::RawHfDelta {
        return Ok(());
    }
    let one = MlxArray::from_f32_slice(&[1.0_f32]);
    for tensor in expected.iter().filter(|tensor| tensor.norm) {
        if let Some(raw) = name_map.get(tensor.name) {
            let gain = add(&astype(raw, MlxDtype::Float32, None), &one, None);
            name_map.insert(tensor.name.to_string(), gain);
        }
    }
    mlx_sys::try_eval(&name_map.values().collect::<Vec<_>>())
        .map_err(|e| WeightLoadError::InvalidLayer(format!("Flash Next MTP norm evaluation: {e}")))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use ax_engine_core::NativeTensorQuantization;

    struct TempRoot(PathBuf);
    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_root(tag: &str) -> TempRoot {
        let root = std::env::temp_dir().join(format!(
            "ax-flash-mtp-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        for name in [MTP_SIDECAR_FILE, "other.safetensors"] {
            // Admission must never parse these payload placeholders.
            std::fs::write(root.join(name), b"metadata-only fixture").unwrap();
        }
        TempRoot(root)
    }

    fn tiny_expected() -> Vec<ExpectedTensor> {
        let analogs = MTP_TENSORS
            .iter()
            .filter(|t| matches!(t.3, ShapeRule::TrunkAnalog))
            .map(|t| (t.1, if t.4 { vec![8] } else { vec![5, 8] }))
            .collect();
        MtpGeometry {
            hidden: 8,
            expert_count: 4,
            expert_intermediate: 6,
            analogs,
        }
        .expected()
        .unwrap()
    }

    fn dense_specs(expected: &[ExpectedTensor]) -> Vec<NativeTensorSpec> {
        expected
            .iter()
            .map(|tensor| NativeTensorSpec {
                name: tensor.name.to_string(),
                role: NativeTensorRole::Other,
                layer_index: None,
                dtype: NativeTensorDataType::Bf16,
                source_tensor_type: None,
                source_quantized: false,
                quantization: None,
                quantized_source: None,
                shape: tensor.shape.clone(),
                file: MTP_SIDECAR_FILE.into(),
                offset_bytes: 0,
                length_bytes: 0,
            })
            .collect()
    }

    fn bind(
        root: &TempRoot,
        declared: &[NativeTensorSpec],
        index: Option<&HashMap<String, PathBuf>>,
    ) -> Result<Vec<NativeTensorSpec>, WeightLoadError> {
        let canonical = root.0.canonicalize().unwrap();
        let sidecar = canonical.join(MTP_SIDECAR_FILE);
        bind_declared_specs(
            &tiny_expected(),
            declared,
            &root.0,
            &canonical,
            &sidecar,
            index,
        )
    }

    fn rejection<T>(result: Result<T, WeightLoadError>) -> String {
        match result {
            Ok(_) => panic!("malformed MTP contract was admitted"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn contract_is_exactly_thirty_one_unique_tensors() {
        let expected = tiny_expected();
        assert_eq!(expected.len(), 31);
        let names: HashSet<_> = expected.iter().map(|t| t.name).collect();
        assert_eq!(names.len(), 31);
        assert_eq!(expected.iter().filter(|t| t.norm).count(), 9);
        let packed = expected
            .iter()
            .find(|t| t.role == NativeTensorRole::FfnGateUpExpsPacked)
            .unwrap();
        assert_eq!(packed.shape, vec![4, 12, 8]);
    }

    #[test]
    fn exact_dense_contract_binds_roles_without_reading_payloads() {
        let root = temp_root("exact");
        let expected = tiny_expected();
        let bound = bind(&root, &dense_specs(&expected), None).unwrap();
        assert_eq!(bound.len(), 31);
        let down = bound
            .iter()
            .find(|s| s.name == "mtp.layers.0.mlp.experts.down_proj")
            .unwrap();
        assert_eq!(down.role, NativeTensorRole::FfnDownExps);
        assert_eq!(down.layer_index, Some(0));
        let mixer = bound
            .iter()
            .find(|s| s.name == "mtp.hyper_connection_mixer.hc_norm.weight")
            .unwrap();
        assert_eq!(mixer.layer_index, None);
    }

    #[test]
    fn rejects_missing_extra_and_duplicate_tensors() {
        let root = temp_root("set");
        let expected = tiny_expected();

        let mut missing = dense_specs(&expected);
        missing.pop();
        assert!(rejection(bind(&root, &missing, None)).contains("missing"));

        let mut extra = dense_specs(&expected);
        let mut norm = extra[0].clone();
        norm.name = "mtp.norm.weight".into();
        extra.push(norm);
        assert!(rejection(bind(&root, &extra, None)).contains("unsupported"));

        let mut duplicate = dense_specs(&expected);
        duplicate.push(duplicate[3].clone());
        assert!(rejection(bind(&root, &duplicate, None)).contains("duplicate"));
    }

    #[test]
    fn rejects_quantized_wrong_dtype_and_wrong_shape() {
        let root = temp_root("dtype");
        let expected = tiny_expected();

        let mut quantized = dense_specs(&expected);
        quantized[2].dtype = NativeTensorDataType::U32;
        quantized[2].source_quantized = true;
        quantized[2].quantization = Some(NativeTensorQuantization {
            mode: "affine".into(),
            group_size: 64,
            bits: 4,
        });
        assert!(rejection(bind(&root, &quantized, None)).contains("quantized"));

        let mut int_dtype = dense_specs(&expected);
        int_dtype[5].dtype = NativeTensorDataType::I8;
        assert!(rejection(bind(&root, &int_dtype, None)).contains("BF16/F16/F32"));

        let mut shape = dense_specs(&expected);
        shape[0].shape = vec![16];
        assert!(rejection(bind(&root, &shape, None)).contains("geometry"));
    }

    #[test]
    fn rejects_wrong_file_binding_escape_and_index_conflicts() {
        let root = temp_root("file");
        let expected = tiny_expected();

        let mut wrong_file = dense_specs(&expected);
        wrong_file[7].file = "other.safetensors".into();
        assert!(rejection(bind(&root, &wrong_file, None)).contains("instead of"));

        let mut escape = dense_specs(&expected);
        escape[7].file = format!("../{}", MTP_SIDECAR_FILE).into();
        assert!(matches!(
            bind(&root, &escape, None),
            Err(WeightLoadError::FileMissing(_))
        ));

        let specs = dense_specs(&expected);
        let other = root.0.canonicalize().unwrap().join("other.safetensors");
        let conflict = HashMap::from([(expected[1].name.to_string(), other.clone())]);
        assert!(rejection(bind(&root, &specs, Some(&conflict))).contains("index"));
        let sidecars = HashMap::from([("mtp.fc_hidden.scales".to_string(), other)]);
        assert!(rejection(bind(&root, &specs, Some(&sidecars))).contains("quantization"));
    }

    #[test]
    fn norm_layout_requires_explicit_supported_declaration() {
        assert_eq!(
            parse_norm_layout(&serde_json::json!({"mtp_norm_layout": "raw_hf_delta"})).unwrap(),
            Qwen4ExpMtpNormLayout::RawHfDelta
        );
        assert_eq!(
            parse_norm_layout(&serde_json::json!({"mtp_norm_layout": "mlx_multiplier"})).unwrap(),
            Qwen4ExpMtpNormLayout::MlxMultiplier
        );
        for malformed in [
            serde_json::json!({"mtp_depth_max": 1}),
            serde_json::json!({"mtp_norm_layout": "auto"}),
            serde_json::json!({"mtp_norm_layout": null}),
            serde_json::json!({"mtp_norm_layout": 1}),
        ] {
            assert!(parse_norm_layout(&malformed).is_err(), "{malformed}");
        }
    }

    #[test]
    fn missing_runtime_declaration_file_is_rejected() {
        let root = temp_root("runtime");
        let canonical = root.0.canonicalize().unwrap();
        assert!(matches!(
            read_norm_layout(&root.0, &canonical),
            Err(WeightLoadError::FileMissing(_))
        ));
        std::fs::write(root.0.join(MTP_RUNTIME_FILE), br#"{"mtp_depth_max": 1}"#).unwrap();
        assert!(rejection(read_norm_layout(&root.0, &canonical)).contains("explicitly"));
    }

    #[test]
    fn header_only_sidecar_with_quantized_extras_is_rejected_before_payloads() {
        let root = temp_root("header");
        let expected = tiny_expected();
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        for tensor in &expected {
            header.insert(
                tensor.name.into(),
                serde_json::json!({"dtype": "BF16", "shape": tensor.shape, "data_offsets": [0, 0]}),
            );
        }
        header.insert(
            "mtp.fc_hidden.scales".into(),
            serde_json::json!({"dtype": "F16", "shape": [8, 1], "data_offsets": [0, 0]}),
        );
        let json = serde_json::to_vec(&header).unwrap();
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&json);
        let path = root.0.join(MTP_SIDECAR_FILE);
        std::fs::write(&path, &bytes).unwrap();

        let declared = read_sidecar_header_specs(&path).unwrap();
        assert_eq!(declared.len(), 32);
        assert!(rejection(bind(&root, &declared, None)).contains("mtp.fc_hidden.scales"));
        let admitted: Vec<_> = declared
            .into_iter()
            .filter(|s| !s.name.ends_with(".scales"))
            .collect();
        assert_eq!(bind(&root, &admitted, None).unwrap().len(), 31);

        std::fs::write(&path, (MTP_HEADER_MAX_BYTES + 1).to_le_bytes()).unwrap();
        assert!(rejection(read_sidecar_header_specs(&path)).contains("header length"));
    }

    #[test]
    fn quantized_trunk_analog_unpacks_to_logical_width() {
        let mut spec = dense_specs(&tiny_expected()).remove(15);
        spec.shape = vec![5, 2];
        spec.source_quantized = true;
        spec.quantization = Some(NativeTensorQuantization {
            mode: "affine".into(),
            group_size: 16,
            bits: 4,
        });
        assert_eq!(logical_shape(&spec).unwrap(), vec![5, 16]);
        spec.quantization = None;
        assert!(logical_shape(&spec).is_err());
    }
}
