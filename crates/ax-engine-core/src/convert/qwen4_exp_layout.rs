//! Legacy AXQuant 1.9.0 / MLX-VLM Qwen 3.8 Flash Next weight-layout interpretation.
//!
//! The audited legacy exporter stores raw norm deltas with MLX conv1d axes.
//! Its identity tuple selects `HfNormOnly`; unknown exports remain blocked.

use std::path::Path;

use crate::model::{
    NativeModelManifest, NativeTensorDataType, NativeTensorRole, NativeTensorSpec, WeightSanitize,
};

use super::ConvertError;

const AXQUANT_MANIFEST_FILE: &str = "axquant_manifest.json";

const EXPECTED_SCHEMA_VERSION: &str = "axquant.artifact.v2";
const EXPECTED_AXQUANT_VERSION: &str = "1.9.0";
const EXPECTED_QUANTIZER: &str = "axquant";
const EXPECTED_FORMAT: &str = "mlx";
const EXPECTED_ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";
const EXPECTED_MODEL_ID: &str = "Qwen/Qwen3.8-Flash-Next";
const EXPECTED_REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";
const EXPECTED_RUNTIME_NAME: &str = "mlx-vlm";
const EXPECTED_MLX_VLM_COMMIT: &str = "6102cb4ad1a5b3cc38d8dc7e6cbe2aca395596cb";
const HF_NORM_ONLY: &str = "hf_norm_only";

/// Runtime blocker recorded when the on-disk weight layout could not be
/// interpreted from `axquant_manifest.json` (missing file or unrecognized
/// exporter tuple).
pub const QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER: &str = "qwen4_exp_weight_layout_unknown";

fn is_expert_role(role: NativeTensorRole) -> bool {
    matches!(
        role,
        NativeTensorRole::FfnGateExps
            | NativeTensorRole::FfnUpExps
            | NativeTensorRole::FfnDownExps
            | NativeTensorRole::FfnGateUpExpsPacked
    )
}

fn has_unknown_layout_blocker(manifest: &NativeModelManifest) -> bool {
    manifest
        .runtime_status
        .blockers
        .iter()
        .any(|blocker| blocker == QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER)
}

fn protected_projections_match(manifest: &NativeModelManifest, expert_layout: (u32, u32)) -> bool {
    manifest
        .tensors
        .iter()
        .filter_map(|tensor| tensor.quantization.as_ref())
        .all(|quant| match expert_layout {
            (2, 32) => matches!((quant.bits, quant.group_size), (2 | 4 | 8, 32)),
            (4, 64) => matches!((quant.bits, quant.group_size), (4 | 8, 32 | 64)),
            (6, 64) => matches!((quant.bits, quant.group_size), (6, 64) | (8, 32 | 64)),
            _ => false,
        })
}

/// Inspect expert quantization. Mixed layouts, MXFP4, and unquantized experts
/// are hard errors; a uniform (2,32) / (4,64) / (6,64) affine layout is `Ok`.
fn inspect_expert_layout(manifest: &NativeModelManifest) -> Result<Option<(u32, u32)>, String> {
    let mut expert_layout = None;
    for tensor in &manifest.tensors {
        if let Some(quantization) = &tensor.quantization {
            if quantization.mode != "affine" {
                return Err(format!(
                    "qwen4_exp rejects quantization mode {} on tensor {} (MXFP4 and other \
non-affine formats are not product)",
                    quantization.mode, tensor.name
                ));
            }
            if is_expert_role(tensor.role) {
                let layout = (quantization.bits, quantization.group_size);
                if !matches!(layout, (2, 32) | (4 | 6, 64)) {
                    return Err(format!(
                        "qwen4_exp expert tensor {} has unsupported affine layout bits={} \
group_size={} (product is 4-bit/group64 or 6-bit/group64; 2-bit/group32 is experimental)",
                        tensor.name, layout.0, layout.1
                    ));
                }
                if expert_layout.is_some_and(|expected| expected != layout) {
                    return Err(format!(
                        "qwen4_exp mixed expert layouts are rejected (saw {expert_layout:?} and \
{layout:?})"
                    ));
                }
                expert_layout = Some(layout);
            }
        } else if is_expert_role(tensor.role) {
            return Err(format!(
                "qwen4_exp expert tensor {} must be affine-quantized",
                tensor.name
            ));
        }
    }
    Ok(expert_layout)
}

fn layout_apply_matches(model_dir: &Path, manifest: &NativeModelManifest) -> bool {
    let mut checked = manifest.clone();
    apply(model_dir, &mut checked).is_ok()
        && checked.runtime_status.blockers == manifest.runtime_status.blockers
        && checked.weight_sanitize == manifest.weight_sanitize
}

/// Product 4-bit/group64 and 6-bit/group64 packs admit without an environment
/// variable. 2-bit/group32 still requires `AX_ENGINE_FLASH_NEXT_EXPERIMENTAL`.
/// Mixed layouts, MXFP4, and unknown exporters stay rejected.
pub(crate) fn experimental_runtime_admission(
    model_dir: &Path,
    manifest: &NativeModelManifest,
    enabled: bool,
) -> bool {
    if manifest.model_family != "qwen4_exp"
        || has_unknown_layout_blocker(manifest)
        || manifest
            .runtime_status
            .blockers
            .iter()
            .any(|blocker| blocker != "qwen4_exp_native_trunk_not_implemented")
        || manifest.weight_sanitize != WeightSanitize::HfNormOnly
    {
        return false;
    }
    let Ok(Some(expert_layout)) = inspect_expert_layout(manifest) else {
        return false;
    };
    if !protected_projections_match(manifest, expert_layout) {
        return false;
    }
    if !layout_apply_matches(model_dir, manifest) {
        return false;
    }
    match expert_layout {
        (4 | 6, 64) => true,
        (2, 32) => enabled,
        _ => false,
    }
}

/// Hard-error format gate for Flash Next. Product 4/6-bit packs pass without
/// env. 2-bit still needs the family opt-in. Mixed layouts, MXFP4, and
/// protected-projection mismatches are errors, not silent admission failures.
pub(crate) fn validate_qwen4_exp_runtime_formats(
    model_dir: &Path,
    manifest: &NativeModelManifest,
    experimental: bool,
) -> Result<(), String> {
    if manifest.model_family != "qwen4_exp" || has_unknown_layout_blocker(manifest) {
        return Ok(());
    }
    if manifest.weight_sanitize != WeightSanitize::HfNormOnly {
        return Ok(());
    }
    let expert_layout = inspect_expert_layout(manifest)?;
    let Some(expert_layout) = expert_layout else {
        return Ok(());
    };
    if !protected_projections_match(manifest, expert_layout) {
        return Err(format!(
            "qwen4_exp protected-projection layout does not match expert format {expert_layout:?}"
        ));
    }
    if expert_layout == (2, 32) && !experimental {
        return Err(
            "qwen4_exp 2-bit/group32 is not a product format (set AX_ENGINE_FLASH_NEXT_EXPERIMENTAL=1; \
AX_ENGINE_2BIT_EXPERIMENTAL=1 is also required)"
                .to_string(),
        );
    }
    if !layout_apply_matches(model_dir, manifest) {
        return Err(
            "qwen4_exp audited exporter identity or convolution layout check failed".to_string(),
        );
    }
    Ok(())
}

/// Interpret a freshly-converted `qwen4_exp` manifest's weight layout from
/// `axquant_manifest.json` in `model_dir`, when present.
///
/// The audited legacy tuple sets `weight_sanitize=hf_norm_only` and leaves
/// `ready=true`. Unrecognized or missing exporters append
/// `qwen4_exp_weight_layout_unknown` and clear readiness.
pub(super) fn apply(
    model_dir: &Path,
    manifest: &mut NativeModelManifest,
) -> Result<(), ConvertError> {
    if manifest.model_family != "qwen4_exp" {
        return Ok(());
    }
    let manifest_path = model_dir.join(AXQUANT_MANIFEST_FILE);
    let bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            push_layout_unknown(manifest);
            return Ok(());
        }
        Err(source) => {
            return Err(ConvertError::ReadFile {
                path: manifest_path,
                source,
            });
        }
    };

    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|source| ConvertError::ParseJson {
            path: manifest_path.clone(),
            source,
        })?;

    if !matches_audited_legacy_tuple(&value) {
        push_layout_unknown(manifest);
        return Ok(());
    }

    validate_optional_mlx_vlm_commit(&value)?;
    validate_optional_sanitize_field(value.get("weight_sanitize"), "weight_sanitize")?;
    validate_optional_sanitize_field(
        value
            .get("runtime")
            .and_then(|runtime| runtime.get("ax_engine"))
            .and_then(|ax_engine| ax_engine.get("weight_sanitize")),
        "runtime.ax_engine.weight_sanitize",
    )?;

    validate_conv_tensors(manifest)?;

    manifest.weight_sanitize = WeightSanitize::HfNormOnly;
    manifest.runtime_status.notes.push(format!(
        "AX interpreted axquant_manifest.json as the audited AXQuant {EXPECTED_AXQUANT_VERSION} \
legacy export from {EXPECTED_RUNTIME_NAME} (source {EXPECTED_MODEL_ID}@{EXPECTED_REVISION}): \
weights are raw norm deltas (weight_sanitize=hf_norm_only) with conv1d already in MLX axis \
order. This is a legacy interpretation exception for one audited exporter, not proof the \
layout is correct."
    ));

    Ok(())
}

fn push_layout_unknown(manifest: &mut NativeModelManifest) {
    manifest.runtime_status.ready = false;
    manifest
        .runtime_status
        .blockers
        .push(QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER.to_string());
    manifest.runtime_status.notes.push(format!(
        "qwen4_exp weight layout could not be determined from {AXQUANT_MANIFEST_FILE} (missing \
or an unrecognized exporter); set an explicit weight_sanitize in the native model manifest \
(e.g. hf_norm_only or hf_to_mlx) before this checkpoint can load."
    ));
}

fn contract_error(message: String) -> ConvertError {
    ConvertError::InvalidModelContract {
        model_type: "qwen4_exp".to_string(),
        message,
    }
}

fn json_str<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for key in path {
        current = current.get(key)?;
    }
    current.as_str()
}

fn json_bool(value: &serde_json::Value, path: &[&str]) -> Option<bool> {
    let mut current = value;
    for key in path {
        current = current.get(key)?;
    }
    current.as_bool()
}

/// Matches the audited legacy `mlx-vlm` AXQuant export tuple exactly. All
/// listed fields are required; a missing or differing field means this is
/// not the audited exporter, not an error.
fn matches_audited_legacy_tuple(value: &serde_json::Value) -> bool {
    json_str(value, &["schema_version"]) == Some(EXPECTED_SCHEMA_VERSION)
        && json_str(value, &["axquant_version"]) == Some(EXPECTED_AXQUANT_VERSION)
        && json_str(value, &["quantizer"]) == Some(EXPECTED_QUANTIZER)
        && json_str(value, &["format"]) == Some(EXPECTED_FORMAT)
        && json_str(value, &["source_model", "architecture"]) == Some(EXPECTED_ARCHITECTURE)
        && json_str(value, &["source_model", "model_id"]) == Some(EXPECTED_MODEL_ID)
        && json_str(value, &["source_model", "revision"]) == Some(EXPECTED_REVISION)
        && json_str(value, &["runtime", "primary_runtime", "name"]) == Some(EXPECTED_RUNTIME_NAME)
        && json_bool(
            value,
            &["runtime", "primary_runtime", "standard_mlx_weights"],
        ) == Some(true)
}

fn validate_optional_mlx_vlm_commit(value: &serde_json::Value) -> Result<(), ConvertError> {
    let Some(commit) = value
        .get("software_versions")
        .and_then(|v| v.get("mlx_vlm_commit"))
    else {
        return Ok(());
    };
    match commit.as_str() {
        Some(EXPECTED_MLX_VLM_COMMIT) => Ok(()),
        _ => Err(contract_error(format!(
            "axquant_manifest.json software_versions.mlx_vlm_commit must be \
\"{EXPECTED_MLX_VLM_COMMIT}\" when present, got {commit}"
        ))),
    }
}

fn validate_optional_sanitize_field(
    value: Option<&serde_json::Value>,
    field_label: &str,
) -> Result<(), ConvertError> {
    let Some(value) = value else {
        return Ok(());
    };
    match value.as_str() {
        Some(HF_NORM_ONLY) => Ok(()),
        _ => Err(contract_error(format!(
            "axquant_manifest.json {field_label} must be the string \"{HF_NORM_ONLY}\" when \
present, got {value}"
        ))),
    }
}

/// Prevent the exporter/manifest admitting contradictory conv1d axes: every
/// mapped GDN / PLE conv1d tensor must be an unquantized, dense floating
/// rank-3 tensor shaped `[channels, kernel, 1]` (MLX axis order), and a conv
/// tensor must be present whenever the manifest's own config says the
/// feature is enabled. Full per-layer shape/count checks remain the job of
/// the dedicated `qwen4_exp` model validator; this only rejects manifests
/// that are already self-contradictory before that validator ever runs.
fn validate_conv_tensors(manifest: &NativeModelManifest) -> Result<(), ConvertError> {
    let mut found_gdn_conv = false;
    let mut found_ple_conv = false;

    for tensor in &manifest.tensors {
        match tensor.role {
            NativeTensorRole::LinearAttentionConv1d => {
                found_gdn_conv = true;
                let kernel = manifest.linear_attention.conv_kernel_dim.ok_or_else(|| {
                    contract_error(format!(
                        "tensor {} has role linear_attention_conv1d but \
linear_attention.conv_kernel_dim is not configured",
                        tensor.name
                    ))
                })?;
                validate_mlx_conv_tensor(tensor, u64::from(kernel), "linear_attention_conv1d")?;
            }
            NativeTensorRole::Qwen4ExpPleConv1d => {
                found_ple_conv = true;
                let kernel = manifest.qwen4_exp.ple_conv_kernel_size.ok_or_else(|| {
                    contract_error(format!(
                        "tensor {} has role qwen4_exp_ple_conv1d but \
qwen4_exp.ple_conv_kernel_size is not configured",
                        tensor.name
                    ))
                })?;
                validate_mlx_conv_tensor(tensor, u64::from(kernel), "qwen4_exp_ple_conv1d")?;
            }
            _ => {}
        }
    }

    if manifest
        .layer_types
        .iter()
        .any(|kind| kind == "linear_attention")
        && !found_gdn_conv
    {
        return Err(contract_error(
            "layer_types includes linear_attention but no linear_attention_conv1d tensor is \
mapped"
                .to_string(),
        ));
    }
    if !manifest.qwen4_exp.ple_layer_ids.is_empty() && !found_ple_conv {
        return Err(contract_error(
            "qwen4_exp.ple_layer_ids is nonempty but no qwen4_exp_ple_conv1d tensor is mapped"
                .to_string(),
        ));
    }

    Ok(())
}

fn validate_mlx_conv_tensor(
    tensor: &NativeTensorSpec,
    kernel: u64,
    label: &str,
) -> Result<(), ConvertError> {
    let is_quantized = tensor.source_quantized
        || tensor.quantization.is_some()
        || tensor.quantized_source.is_some();
    if is_quantized {
        return Err(contract_error(format!(
            "{label} tensor {} must be unquantized, but the manifest marks it quantized",
            tensor.name
        )));
    }
    if !matches!(
        tensor.dtype,
        NativeTensorDataType::F16 | NativeTensorDataType::Bf16 | NativeTensorDataType::F32
    ) {
        return Err(contract_error(format!(
            "{label} tensor {} must be a dense floating tensor, got dtype {:?}",
            tensor.name, tensor.dtype
        )));
    }
    if tensor.shape.len() != 3 {
        return Err(contract_error(format!(
            "{label} tensor {} must be rank 3, got shape {:?}",
            tensor.name, tensor.shape
        )));
    }
    let channels = tensor.shape[0];
    if channels == 0 {
        return Err(contract_error(format!(
            "{label} tensor {} must have a positive channel count, got shape {:?}",
            tensor.name, tensor.shape
        )));
    }
    let expected = [channels, kernel, 1];
    if tensor.shape != expected {
        return Err(contract_error(format!(
            "{label} tensor {} must have MLX conv1d axes [channels, kernel, 1] = {:?}, got {:?}",
            tensor.name, expected, tensor.shape
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::path::PathBuf;

    use crate::model::{
        NativeLinearAttentionConfig, NativeQwen4ExpConfig, NativeRuntimeStatus, NativeTensorFormat,
        NativeTensorQuantization,
    };

    use super::*;

    fn conv_tensor(
        name: &str,
        role: NativeTensorRole,
        shape: Vec<u64>,
        dtype: NativeTensorDataType,
        quantized: bool,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index: Some(0),
            dtype,
            source_tensor_type: None,
            source_quantized: false,
            quantization: quantized.then(NativeTensorQuantization::default),
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 64,
        }
    }

    fn gdn_conv() -> NativeTensorSpec {
        conv_tensor(
            "layers.0.linear_attn.conv1d.weight",
            NativeTensorRole::LinearAttentionConv1d,
            vec![8, 4, 1],
            NativeTensorDataType::Bf16,
            false,
        )
    }

    fn ple_conv() -> NativeTensorSpec {
        conv_tensor(
            "layers.0.ple.conv1d.weight",
            NativeTensorRole::Qwen4ExpPleConv1d,
            vec![8, 3, 1],
            NativeTensorDataType::Bf16,
            false,
        )
    }

    /// One canonical `qwen4_exp` manifest matching the audited legacy tuple's
    /// GDN + PLE conv contract; tests clone and mutate from here rather than
    /// repeating the full initializer.
    fn base_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: "ax.native_model.v1".to_string(),
            model_family: "qwen4_exp".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus {
                ready: true,
                blockers: Vec::new(),
                notes: Vec::new(),
            },
            layer_count: 1,
            hidden_size: 8,
            intermediate_size: 0,
            attention_head_count: 2,
            attention_head_dim: 4,
            kv_head_count: 1,
            vocab_size: 32,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_low_freq_factor: None,
            rope_high_freq_factor: None,
            rope_original_context_len: None,
            rope_beta_fast: None,
            rope_beta_slow: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: None,
            attn_temperature_scale: None,
            intermediate_size_mlp: 0,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: true,
            partial_rotary_factor: None,
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            global_kv_head_count: None,
            sliding_window_size: None,
            layer_types: vec!["linear_attention".to_string()],
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            final_logits_scale: None,
            attention_scale_multiplier: None,
            post_norm_eps: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig {
                conv_kernel_dim: Some(4),
                ..Default::default()
            },
            mla_attention: Default::default(),
            moe: Default::default(),
            glm_router: Default::default(),
            deepseek_v4: Default::default(),
            qwen4_exp: NativeQwen4ExpConfig {
                ple_layer_ids: vec![0],
                ple_conv_kernel_size: Some(3),
                ..Default::default()
            },
            weight_sanitize: WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: Default::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
            tensors: vec![gdn_conv(), ple_conv()],
        }
    }

    fn audited_legacy_json() -> serde_json::Value {
        serde_json::json!({
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "axquant_version": EXPECTED_AXQUANT_VERSION,
            "quantizer": EXPECTED_QUANTIZER,
            "format": EXPECTED_FORMAT,
            "source_model": {
                "architecture": EXPECTED_ARCHITECTURE,
                "model_id": EXPECTED_MODEL_ID,
                "revision": EXPECTED_REVISION,
            },
            "runtime": {
                "primary_runtime": {
                    "name": EXPECTED_RUNTIME_NAME,
                    "standard_mlx_weights": true,
                },
            },
        })
    }

    fn write_axquant_manifest(model_dir: &std::path::Path, value: &serde_json::Value) {
        std::fs::write(
            model_dir.join(AXQUANT_MANIFEST_FILE),
            serde_json::to_vec_pretty(value).expect("fixture should serialize"),
        )
        .expect("fixture should write");
    }

    fn temp_model_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ax-qwen4-exp-layout-test-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp model dir should create");
        dir
    }

    #[test]
    fn experimental_admission_is_narrow_and_preserves_readiness() {
        let dir = temp_model_dir("experimental");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        apply(&dir, &mut manifest).unwrap();
        manifest.tensors.push(conv_tensor(
            "expert",
            NativeTensorRole::FfnGateExps,
            vec![4, 8, 1],
            NativeTensorDataType::U32,
            true,
        ));
        let before = serde_json::to_vec(&manifest).unwrap();
        assert!(
            experimental_runtime_admission(&dir, &manifest, false),
            "product 4-bit/group64 must admit without env"
        );
        assert!(experimental_runtime_admission(&dir, &manifest, true));
        assert_eq!(serde_json::to_vec(&manifest).unwrap(), before);
        validate_qwen4_exp_runtime_formats(&dir, &manifest, false).unwrap();
        for bits in [2, 3, 5, 8] {
            let mut changed = manifest.clone();
            changed
                .tensors
                .last_mut()
                .unwrap()
                .quantization
                .as_mut()
                .unwrap()
                .bits = bits;
            assert!(!experimental_runtime_admission(&dir, &changed, true));
        }
        for (mode, group) in [("mxfp4", 64), ("affine", 128)] {
            let mut changed = manifest.clone();
            let quant = changed
                .tensors
                .last_mut()
                .unwrap()
                .quantization
                .as_mut()
                .unwrap();
            quant.mode = mode.into();
            quant.group_size = group;
            assert!(!experimental_runtime_admission(&dir, &changed, true));
            assert!(validate_qwen4_exp_runtime_formats(&dir, &changed, true).is_err());
        }
        let mut missing = manifest.clone();
        missing.tensors.pop();
        assert!(!experimental_runtime_admission(&dir, &missing, true));
        let mut changed = manifest.clone();
        changed
            .runtime_status
            .blockers
            .push("unresolved_contract".into());
        assert!(!experimental_runtime_admission(&dir, &changed, true));
        changed = manifest.clone();
        changed.runtime_status.ready = false;
        changed
            .runtime_status
            .blockers
            .push("qwen4_exp_native_trunk_not_implemented".into());
        assert!(
            experimental_runtime_admission(&dir, &changed, false),
            "legacy not-ready 4-bit manifests still admit without env"
        );
        changed = manifest.clone();
        changed.model_family = "qwen3_5".into();
        assert!(!experimental_runtime_admission(&dir, &changed, true));
        changed = manifest.clone();
        changed.weight_sanitize = WeightSanitize::None;
        assert!(!experimental_runtime_admission(&dir, &changed, true));
        let mut metadata = audited_legacy_json();
        metadata["source_model"]["revision"] = "unknown".into();
        write_axquant_manifest(&dir, &metadata);
        assert!(!experimental_runtime_admission(&dir, &manifest, true));
        std::fs::remove_file(dir.join(AXQUANT_MANIFEST_FILE)).unwrap();
        assert!(!experimental_runtime_admission(&dir, &manifest, true));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn experimental_affine_formats_require_observed_and_uniform_expert_layouts() {
        let dir = temp_model_dir("experimental-affine-formats");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        apply(&dir, &mut manifest).unwrap();
        let mut expert = conv_tensor(
            "expert",
            NativeTensorRole::FfnGateExps,
            vec![4, 8, 1],
            NativeTensorDataType::U32,
            true,
        );
        for (bits, group_size) in [(2, 32), (4, 64), (6, 64)] {
            let quant = expert.quantization.as_mut().unwrap();
            quant.bits = bits;
            quant.group_size = group_size;
            let mut candidate = manifest.clone();
            candidate.tensors.push(expert.clone());
            let before = serde_json::to_vec(&candidate).unwrap();
            if bits == 2 {
                assert!(!experimental_runtime_admission(&dir, &candidate, false));
                assert!(experimental_runtime_admission(&dir, &candidate, true));
                assert!(validate_qwen4_exp_runtime_formats(&dir, &candidate, false).is_err());
                validate_qwen4_exp_runtime_formats(&dir, &candidate, true).unwrap();
            } else {
                assert!(experimental_runtime_admission(&dir, &candidate, false));
                assert!(experimental_runtime_admission(&dir, &candidate, true));
                validate_qwen4_exp_runtime_formats(&dir, &candidate, false).unwrap();
            }
            assert_eq!(serde_json::to_vec(&candidate).unwrap(), before);

            let mut mixed = expert.clone();
            mixed.name = "other_expert".into();
            let other = mixed.quantization.as_mut().unwrap();
            (other.bits, other.group_size) = if bits == 2 { (4, 64) } else { (2, 32) };
            candidate.tensors.push(mixed);
            assert!(!experimental_runtime_admission(&dir, &candidate, true));
            assert!(
                validate_qwen4_exp_runtime_formats(&dir, &candidate, true)
                    .unwrap_err()
                    .contains("mixed expert layouts")
            );
            candidate.tensors.pop();

            let mut projection = expert.clone();
            projection.name = "projection".into();
            projection.role = NativeTensorRole::AttentionQ;
            for (bad_bits, bad_group) in [(2, 64), (6, 32), (3, 32), (8, 128)] {
                let quant = projection.quantization.as_mut().unwrap();
                quant.bits = bad_bits;
                quant.group_size = bad_group;
                candidate.tensors.push(projection.clone());
                assert!(!experimental_runtime_admission(&dir, &candidate, true));
                assert!(
                    validate_qwen4_exp_runtime_formats(&dir, &candidate, true)
                        .unwrap_err()
                        .contains("protected-projection")
                );
                candidate.tensors.pop();
            }
            let outside_pack = projection.quantization.as_mut().unwrap();
            (outside_pack.bits, outside_pack.group_size) = match bits {
                2 => (8, 64),
                4 => (6, 64),
                _ => (4, 64),
            };
            candidate.tensors.push(projection);
            assert!(!experimental_runtime_admission(&dir, &candidate, true));
            candidate.tensors.pop();
            candidate.tensors.last_mut().unwrap().quantization = None;
            assert!(!experimental_runtime_admission(&dir, &candidate, true));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn other_family_ignores_even_malformed_exporter_metadata() {
        let dir = temp_model_dir("unrelated-family");
        std::fs::write(dir.join(AXQUANT_MANIFEST_FILE), b"{invalid").unwrap();
        let mut manifest = base_manifest();
        manifest.model_family = "qwen3_5".into();
        let before = serde_json::to_vec(&manifest).unwrap();
        apply(&dir, &mut manifest).unwrap();
        assert_eq!(serde_json::to_vec(&manifest).unwrap(), before);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_manifest_blocks_layout_and_leaves_sanitize_unchanged() {
        let dir = temp_model_dir("missing");
        let mut manifest = base_manifest();

        apply(&dir, &mut manifest).expect("missing axquant manifest is not an error");

        assert_eq!(manifest.weight_sanitize, WeightSanitize::None);
        assert!(
            manifest
                .runtime_status
                .blockers
                .contains(&QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER.to_string())
        );
        assert!(
            !manifest.runtime_status.ready,
            "unknown layout must clear readiness"
        );
        assert!(
            manifest
                .runtime_status
                .notes
                .iter()
                .any(|note| note.contains("weight_sanitize")),
            "note must actionably point at an explicit weight_sanitize"
        );
    }

    #[test]
    fn malformed_json_returns_parse_error() {
        let dir = temp_model_dir("malformed");
        std::fs::write(dir.join(AXQUANT_MANIFEST_FILE), b"{not json")
            .expect("fixture should write");
        let mut manifest = base_manifest();

        let error = apply(&dir, &mut manifest).expect_err("malformed JSON must error");
        assert!(matches!(error, ConvertError::ParseJson { .. }));
    }

    #[test]
    fn unknown_tuple_blocks_layout_like_missing_file() {
        for mutation in ["schema_version", "axquant_version", "quantizer", "format"] {
            let dir = temp_model_dir(&format!("unknown-{mutation}"));
            let mut value = audited_legacy_json();
            value[mutation] = serde_json::json!("something-else");
            write_axquant_manifest(&dir, &value);
            let mut manifest = base_manifest();

            apply(&dir, &mut manifest).expect("unrecognized tuple is not an error");

            assert_eq!(manifest.weight_sanitize, WeightSanitize::None);
            assert!(
                manifest
                    .runtime_status
                    .blockers
                    .contains(&QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER.to_string()),
                "mutating {mutation} must fall back to the layout-unknown blocker"
            );
        }
    }

    #[test]
    fn missing_required_field_is_unknown_tuple_not_error() {
        let dir = temp_model_dir("missing-field");
        let mut value = audited_legacy_json();
        value
            .as_object_mut()
            .expect("fixture root is an object")
            .remove("axquant_version");
        write_axquant_manifest(&dir, &value);
        let mut manifest = base_manifest();

        apply(&dir, &mut manifest).expect("missing required field is not an error");

        assert_eq!(manifest.weight_sanitize, WeightSanitize::None);
        assert!(
            manifest
                .runtime_status
                .blockers
                .contains(&QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER.to_string())
        );
    }

    #[test]
    fn other_axquant_version_is_unknown_tuple() {
        let dir = temp_model_dir("other-version");
        let mut value = audited_legacy_json();
        value["axquant_version"] = serde_json::json!("2.0.0");
        write_axquant_manifest(&dir, &value);
        let mut manifest = base_manifest();

        apply(&dir, &mut manifest).expect("other version is not an error");

        assert_eq!(manifest.weight_sanitize, WeightSanitize::None);
        assert!(
            manifest
                .runtime_status
                .blockers
                .contains(&QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER.to_string())
        );
    }

    #[test]
    fn matching_tuple_with_valid_convs_sets_hf_norm_only_and_stays_ready() {
        let dir = temp_model_dir("matching");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();

        apply(&dir, &mut manifest).expect("audited tuple with valid convs should apply");

        assert_eq!(manifest.weight_sanitize, WeightSanitize::HfNormOnly);
        assert!(manifest.runtime_status.ready);
        assert!(manifest.runtime_status.blockers.is_empty());
        assert!(
            !manifest
                .runtime_status
                .blockers
                .contains(&QWEN4_EXP_WEIGHT_LAYOUT_UNKNOWN_BLOCKER.to_string()),
            "recognized tuple must not also carry the unknown-layout blocker"
        );
        assert!(
            manifest
                .runtime_status
                .notes
                .iter()
                .any(|note| note.contains("legacy interpretation exception")),
        );
    }

    #[test]
    fn matching_tuple_with_matching_optional_commit_and_sanitize_hints_applies() {
        let dir = temp_model_dir("matching-optional");
        let mut value = audited_legacy_json();
        value["software_versions"] =
            serde_json::json!({ "mlx_vlm_commit": EXPECTED_MLX_VLM_COMMIT });
        value["weight_sanitize"] = serde_json::json!(HF_NORM_ONLY);
        value["runtime"]["ax_engine"] = serde_json::json!({ "weight_sanitize": HF_NORM_ONLY });
        write_axquant_manifest(&dir, &value);
        let mut manifest = base_manifest();

        apply(&dir, &mut manifest).expect("matching optional hints should apply");

        assert_eq!(manifest.weight_sanitize, WeightSanitize::HfNormOnly);
    }

    #[test]
    fn wrong_optional_commit_rejects() {
        let dir = temp_model_dir("wrong-commit");
        let mut value = audited_legacy_json();
        value["software_versions"] = serde_json::json!({ "mlx_vlm_commit": "deadbeef" });
        write_axquant_manifest(&dir, &value);
        let mut manifest = base_manifest();

        let error = apply(&dir, &mut manifest).expect_err("wrong commit must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
        assert_eq!(manifest.weight_sanitize, WeightSanitize::None);
    }

    #[test]
    fn conflicting_root_sanitize_field_rejects() {
        let dir = temp_model_dir("conflicting-sanitize");
        let mut value = audited_legacy_json();
        value["weight_sanitize"] = serde_json::json!("hf_to_mlx");
        write_axquant_manifest(&dir, &value);
        let mut manifest = base_manifest();

        let error = apply(&dir, &mut manifest).expect_err("conflicting sanitize must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn ill_typed_nested_sanitize_field_rejects() {
        let dir = temp_model_dir("ill-typed-sanitize");
        let mut value = audited_legacy_json();
        value["runtime"]["ax_engine"] = serde_json::json!({ "weight_sanitize": 1 });
        write_axquant_manifest(&dir, &value);
        let mut manifest = base_manifest();

        let error = apply(&dir, &mut manifest).expect_err("ill-typed sanitize must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn raw_hf_conv_axes_reject() {
        let dir = temp_model_dir("raw-axes");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        // HF axis order is (channels, 1, kernel), not MLX's (channels, kernel, 1).
        manifest.tensors = vec![
            conv_tensor(
                "layers.0.linear_attn.conv1d.weight",
                NativeTensorRole::LinearAttentionConv1d,
                vec![8, 1, 4],
                NativeTensorDataType::Bf16,
                false,
            ),
            ple_conv(),
        ];

        let error = apply(&dir, &mut manifest).expect_err("raw HF axes must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
        assert_eq!(manifest.weight_sanitize, WeightSanitize::None);
    }

    #[test]
    fn mixed_valid_gdn_invalid_ple_axes_rejects() {
        let dir = temp_model_dir("mixed-axes");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        manifest.tensors = vec![
            gdn_conv(),
            conv_tensor(
                "layers.0.ple.conv1d.weight",
                NativeTensorRole::Qwen4ExpPleConv1d,
                vec![8, 1, 3],
                NativeTensorDataType::Bf16,
                false,
            ),
        ];

        let error = apply(&dir, &mut manifest).expect_err("mixed axes must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn quantized_conv_tensor_rejects() {
        let dir = temp_model_dir("quantized-conv");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        manifest.tensors = vec![
            conv_tensor(
                "layers.0.linear_attn.conv1d.weight",
                NativeTensorRole::LinearAttentionConv1d,
                vec![8, 4, 1],
                NativeTensorDataType::Bf16,
                true,
            ),
            ple_conv(),
        ];

        let error = apply(&dir, &mut manifest).expect_err("quantized conv must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn nonfloating_conv_tensor_rejects() {
        let dir = temp_model_dir("nonfloating-conv");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        manifest.tensors = vec![
            conv_tensor(
                "layers.0.linear_attn.conv1d.weight",
                NativeTensorRole::LinearAttentionConv1d,
                vec![8, 4, 1],
                NativeTensorDataType::U32,
                false,
            ),
            ple_conv(),
        ];

        let error = apply(&dir, &mut manifest).expect_err("non-floating conv must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn zero_channel_conv_tensor_rejects() {
        let dir = temp_model_dir("zero-channel");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        manifest.tensors = vec![
            conv_tensor(
                "layers.0.linear_attn.conv1d.weight",
                NativeTensorRole::LinearAttentionConv1d,
                vec![0, 4, 1],
                NativeTensorDataType::Bf16,
                false,
            ),
            ple_conv(),
        ];

        let error = apply(&dir, &mut manifest).expect_err("zero channels must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn missing_gdn_conv_when_linear_attention_layer_type_present_rejects() {
        let dir = temp_model_dir("missing-gdn");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        manifest.tensors = vec![ple_conv()];

        let error = apply(&dir, &mut manifest).expect_err("missing GDN conv must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn missing_ple_conv_when_ple_layer_ids_present_rejects() {
        let dir = temp_model_dir("missing-ple");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        manifest.tensors = vec![gdn_conv()];

        let error = apply(&dir, &mut manifest).expect_err("missing PLE conv must reject");
        assert!(matches!(error, ConvertError::InvalidModelContract { .. }));
    }

    #[test]
    fn conv_features_disabled_do_not_require_conv_tensors() {
        let dir = temp_model_dir("other-family");
        write_axquant_manifest(&dir, &audited_legacy_json());
        let mut manifest = base_manifest();
        manifest.layer_types = vec!["full_attention".to_string()];
        manifest.linear_attention = Default::default();
        manifest.qwen4_exp.ple_layer_ids = Vec::new();
        manifest.qwen4_exp.ple_conv_kernel_size = None;
        manifest.tensors = vec![NativeTensorSpec {
            name: "layers.0.self_attn.q_proj.weight".to_string(),
            role: NativeTensorRole::AttentionQ,
            layer_index: Some(0),
            dtype: NativeTensorDataType::Bf16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: vec![8, 8],
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 128,
        }];

        apply(&dir, &mut manifest)
            .expect("families without GDN/PLE conv features must not require them");

        assert_eq!(manifest.weight_sanitize, WeightSanitize::HfNormOnly);
    }
}
