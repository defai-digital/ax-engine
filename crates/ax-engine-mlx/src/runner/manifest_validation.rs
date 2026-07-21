//! Manifest validation and binding-summary helpers for the MLX runner.
//!
//! Split out of `runner/mod.rs` (Phase 2 slice 1 of the decode-dispatch
//! efficiency plan): pure functions over `NativeModelArtifacts` /
//! `NativeModelManifest` that gate which checkpoints the MLX runner accepts,
//! plus the tokenizer-derived terminal-token resolution. No decode-path
//! state; everything here runs at runner construction time.

use std::collections::BTreeSet;
use std::fs;

use ax_engine_core::{
    NativeModelArtifacts, NativeModelManifest, NativeTensorRole, runner::NativeModelBindingSummary,
};

use super::{COMMON_EOT_TOKEN_STRINGS, MlxRunnerError};

pub(super) fn validate_mlx_supported_manifest(
    artifacts: &NativeModelArtifacts,
) -> Result<(), MlxRunnerError> {
    let manifest = artifacts.manifest();
    if !is_mlx_supported_model_family(&manifest.model_family) {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "model_family {:?} is not supported by the MLX runner",
            manifest.model_family
        )));
    }
    if manifest.model_family == "glm4_moe_lite" || has_glm_mla_tensors(artifacts) {
        validate_mla_moe_manifest(manifest)?;
    }
    if manifest.linear_attention.is_enabled() || has_linear_attention_tensors(artifacts) {
        validate_qwen_gated_delta_linear_attention(manifest)?;
    }
    if manifest.model_family == "llama4" {
        validate_llama4_manifest(manifest)?;
    }
    // Interleaved SWA validation (Gemma3/4): triggered by layer_types, KV sharing,
    // a separate global head dim, or a separate SWA rope theta. Families with
    // uniform SWA (mistral3, mixtral) use only sliding_window_size with no
    // layer_types, so they skip this gate.
    if !manifest.layer_types.is_empty()
        || !manifest.kv_shared_source_layers.is_empty()
        || manifest.global_head_dim.is_some()
        || manifest.rope_theta_swa.is_some()
    {
        validate_gemma4_interleaved_attention(manifest)?;
    }
    // Prefer generation kind (ADR-038) over family-string-only gating; keep the
    // family label as a belt-and-suspenders for older manifests without a
    // filled diffusion config block.
    if matches!(
        manifest.generation_kind(),
        ax_engine_core::GenerationKind::BlockDiffusion
    ) || manifest.model_family == "diffusion_gemma"
    {
        validate_diffusion_gemma_manifest(manifest)?;
    }
    Ok(())
}

pub(super) fn is_mlx_supported_model_family(model_family: &str) -> bool {
    matches!(
        model_family,
        "gemma4"
            | "gemma3"
            | "embeddinggemma"
            | "qwen3"
            | "llama3"
            | "diffusion_gemma"
            | "llama4"
            | "qwen3_5"
            | "qwen3_next"
            | "glm4_moe_lite"
            | "deepseek_v3"
            | "deepseek_v32"
            | "mistral3"
            | "mixtral"
            // Secondary open reasoner (catalog + family implementation + registry).
            | "gpt_oss"
            // Multimodal text backbone (same Standard route as gemma4).
            | "gemma4_unified"
            // Unlimited-OCR / DeepSeek-OCR multimodal MoE + dual vision.
            | "unlimited_ocr"
    )
}

/// Validate DiffusionGemma-specific manifest fields.
///
/// DiffusionGemma uses the Gemma4 MoE backbone with bidirectional denoiser
/// attention over a fixed canvas. The diffusion config block must be present
/// and carry at least `canvas_size`.
pub(super) fn validate_diffusion_gemma_manifest(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if manifest.layer_types.is_empty() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "diffusion_gemma requires layer_types for interleaved SWA/full attention".to_string(),
        ));
    }
    match manifest.diffusion.canvas_size {
        Some(value) if value > 0 => {}
        Some(_) => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "diffusion.canvas_size must be greater than zero".to_string(),
            ));
        }
        None => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "diffusion_gemma requires diffusion.canvas_size in the manifest".to_string(),
            ));
        }
    }
    for (name, value) in [
        ("max_denoise_steps", manifest.diffusion.max_denoise_steps),
        ("convergence_steps", manifest.diffusion.convergence_steps),
        (
            "convergence_check_interval",
            manifest.diffusion.convergence_check_interval,
        ),
    ] {
        if value == Some(0) {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "diffusion.{name} must be greater than zero"
            )));
        }
    }
    for (name, value) in [
        ("entropy_bound", manifest.diffusion.entropy_bound),
        ("entropy_threshold", manifest.diffusion.entropy_threshold),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "diffusion.{name} must be finite and non-negative"
            )));
        }
    }
    for (name, value) in [
        (
            "acceptance_rate_threshold",
            manifest.diffusion.acceptance_rate_threshold,
        ),
        (
            "confidence_threshold",
            manifest.diffusion.confidence_threshold,
        ),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "diffusion.{name} must be finite and in [0, 1]"
            )));
        }
    }
    for (name, value) in [
        ("temperature_start", manifest.diffusion.temperature_start),
        ("temperature_end", manifest.diffusion.temperature_end),
    ] {
        if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "diffusion.{name} must be finite and greater than zero"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_mla_moe_manifest(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    let is_glm4_moe_lite = manifest.model_family == "glm4_moe_lite";
    let is_deepseek_v3 = matches!(
        manifest.model_family.as_str(),
        "deepseek_v3" | "deepseek_v32"
    );
    if !is_glm4_moe_lite && !is_deepseek_v3 {
        return Err(MlxRunnerError::UnsupportedFeature(
            "MLA tensor roles are supported only for glm4_moe_lite or DeepSeek V3 manifests"
                .to_string(),
        ));
    }
    if !manifest.mla_attention.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} requires mla_attention metadata",
            manifest.model_family
        )));
    }
    if is_glm4_moe_lite && !manifest.glm_router.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router metadata".to_string(),
        ));
    }
    if !manifest.moe.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} requires moe metadata",
            manifest.model_family
        )));
    }

    let first_dense_layers = if is_glm4_moe_lite {
        manifest.glm_router.first_dense_layer_count.ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(
                "glm4_moe_lite requires glm_router.first_dense_layer_count".to_string(),
            )
        })?
    } else {
        manifest.moe.first_dense_layers.unwrap_or(0)
    };
    // `GlmRouterConfig::from_manifest` `.expect()`s these three fields once the router
    // is considered enabled (`is_enabled()` returns true if *any* field is set), and
    // `glm_router_apply_group_selection` follows up with runtime `assert!`s on the
    // group invariants. Surface every panic-source as a typed manifest error here.
    if is_glm4_moe_lite && manifest.glm_router.routed_scaling_factor.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires glm_router.routed_scaling_factor".to_string(),
        ));
    }
    let routed_scaling_factor = if is_glm4_moe_lite {
        manifest.glm_router.routed_scaling_factor.unwrap_or(1.0)
    } else {
        manifest.moe.routed_scaling_factor.unwrap_or(1.0)
    };
    if !routed_scaling_factor.is_finite() || routed_scaling_factor <= 0.0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} requires finite positive routed_scaling_factor",
            manifest.model_family
        )));
    }
    let n_group = if is_glm4_moe_lite {
        manifest.glm_router.n_group.ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(
                "glm4_moe_lite requires glm_router.n_group".to_string(),
            )
        })?
    } else {
        manifest.moe.n_group.unwrap_or(1)
    };
    let topk_group = if is_glm4_moe_lite {
        manifest.glm_router.topk_group.ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(
                "glm4_moe_lite requires glm_router.topk_group".to_string(),
            )
        })?
    } else {
        manifest.moe.topk_group.unwrap_or(1)
    };
    if n_group == 0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} n_group must be greater than zero",
            manifest.model_family
        )));
    }
    if topk_group == 0 || topk_group > n_group {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} topk_group {topk_group} must satisfy 0 < topk_group <= n_group ({n_group})",
            manifest.model_family
        )));
    }
    // `NativeMoeConfig::is_enabled` (checked above) only requires that *some*
    // MoE field is present, but `ModelConfig::from_manifest` then
    // `unwrap_or(0)`s the missing ones. With `n_group > 1`,
    // `glm_router_apply_group_selection` asserts both divisibility and
    // `experts_per_group >= 2`, so a missing `expert_count` (decoded as 0)
    // would silently slip past the divisibility check and then crash on the
    // group-size assert. Require the fields explicitly here.
    let expert_count = manifest.moe.expert_count.ok_or_else(|| {
        MlxRunnerError::UnsupportedFeature("glm4_moe_lite requires moe.expert_count".to_string())
    })?;
    if manifest.moe.experts_per_token.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "glm4_moe_lite requires moe.experts_per_token".to_string(),
        ));
    }
    if n_group > 1 {
        if expert_count % n_group != 0 {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "{} moe.expert_count {expert_count} must be divisible by n_group {n_group}",
                manifest.model_family
            )));
        }
        if expert_count / n_group < 2 {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "{} moe.expert_count {expert_count} divided by n_group {n_group} must yield at least two experts per group",
                manifest.model_family
            )));
        }
    }
    if first_dense_layers > manifest.layer_count {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} first_dense_layer_count {first_dense_layers} cannot exceed layer_count {}",
            manifest.model_family, manifest.layer_count
        )));
    }
    let has_shared_experts = if is_glm4_moe_lite {
        manifest.glm_router.has_shared_experts
    } else {
        manifest.moe.shared_expert_count.unwrap_or(0) > 0
    };
    let moe_layer_freq = manifest.moe.layer_freq.unwrap_or(1);
    if is_deepseek_v3 && moe_layer_freq == 0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "{} requires moe.layer_freq greater than zero",
            manifest.model_family
        )));
    }

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
            require_manifest_role(manifest, layer_index, role)?;
        }
        let has_kv_b = manifest.tensors.iter().any(|tensor| {
            tensor.layer_index == Some(layer_index) && tensor.role == NativeTensorRole::AttentionKvB
        });
        let has_embed_q = manifest.tensors.iter().any(|tensor| {
            tensor.layer_index == Some(layer_index)
                && tensor.role == NativeTensorRole::AttentionEmbedQ
        });
        let has_unembed_out = manifest.tensors.iter().any(|tensor| {
            tensor.layer_index == Some(layer_index)
                && tensor.role == NativeTensorRole::AttentionUnembedOut
        });
        if (has_kv_b && (has_embed_q || has_unembed_out))
            || (!has_kv_b && (!has_embed_q || !has_unembed_out))
        {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "{} layer {layer_index} must provide exactly one MLA KV-B layout",
                manifest.model_family
            )));
        }

        let is_moe_layer = if is_deepseek_v3 {
            layer_index >= first_dense_layers && layer_index.is_multiple_of(moe_layer_freq)
        } else {
            layer_index >= first_dense_layers
        };
        if !is_moe_layer {
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnGate)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnUp)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnDown)?;
        } else {
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnGateInp)?;
            require_manifest_role(
                manifest,
                layer_index,
                NativeTensorRole::FfnGateInpCorrectionBias,
            )?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnGateExps)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnUpExps)?;
            require_manifest_role(manifest, layer_index, NativeTensorRole::FfnDownExps)?;
            if has_shared_experts {
                require_manifest_role(
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnSharedExpertGate,
                )?;
                require_manifest_role(manifest, layer_index, NativeTensorRole::FfnSharedExpertUp)?;
                require_manifest_role(
                    manifest,
                    layer_index,
                    NativeTensorRole::FfnSharedExpertDown,
                )?;
            }
        }
    }

    Ok(())
}

pub(super) fn validate_llama4_manifest(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if manifest.moe.is_enabled() && manifest.moe.layer_freq == Some(0) {
        return Err(MlxRunnerError::UnsupportedFeature(
            "llama4 moe.layer_freq must be greater than zero".to_string(),
        ));
    }
    if manifest.no_rope_layer_interval > 0 {
        if manifest.attn_temperature_floor == Some(0) {
            return Err(MlxRunnerError::UnsupportedFeature(
                "llama4 attn_temperature_floor must be greater than zero".to_string(),
            ));
        }
        if manifest
            .attn_temperature_scale
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(MlxRunnerError::UnsupportedFeature(
                "llama4 attn_temperature_scale must be finite and non-negative".to_string(),
            ));
        }
    }

    let expert_count = manifest.moe.expert_count.unwrap_or(0);
    let layer_freq = manifest.moe.layer_freq.unwrap_or(1);
    for layer_index in 0..manifest.layer_count {
        let is_moe = expert_count > 0
            && layer_freq > 0
            && layer_index % layer_freq == layer_freq.saturating_sub(1);
        if is_moe {
            for role in [
                NativeTensorRole::FfnGateInp,
                NativeTensorRole::FfnSharedExpertGate,
                NativeTensorRole::FfnSharedExpertUp,
                NativeTensorRole::FfnSharedExpertDown,
            ] {
                require_manifest_role(manifest, layer_index, role)?;
            }
        } else {
            for role in [
                NativeTensorRole::FfnGate,
                NativeTensorRole::FfnUp,
                NativeTensorRole::FfnDown,
            ] {
                require_manifest_role(manifest, layer_index, role)?;
            }
        }
    }

    Ok(())
}

pub(super) fn require_manifest_role(
    manifest: &NativeModelManifest,
    layer_index: u32,
    role: NativeTensorRole,
) -> Result<(), MlxRunnerError> {
    if manifest
        .tensors
        .iter()
        .any(|tensor| tensor.layer_index == Some(layer_index) && tensor.role == role)
    {
        return Ok(());
    }

    Err(MlxRunnerError::UnsupportedFeature(format!(
        "{} layer {layer_index} is missing required tensor role {role:?}",
        manifest.model_family
    )))
}

pub(super) fn has_glm_mla_tensors(artifacts: &NativeModelArtifacts) -> bool {
    artifacts.tensor_specs().iter().any(|tensor| {
        matches!(
            tensor.role,
            NativeTensorRole::AttentionQa
                | NativeTensorRole::AttentionQaNorm
                | NativeTensorRole::AttentionQb
                | NativeTensorRole::AttentionKvA
                | NativeTensorRole::AttentionKvB
                | NativeTensorRole::AttentionKvANorm
                | NativeTensorRole::AttentionEmbedQ
                | NativeTensorRole::AttentionUnembedOut
                | NativeTensorRole::FfnGateInpCorrectionBias
        )
    })
}

pub(super) fn has_linear_attention_tensors(artifacts: &NativeModelArtifacts) -> bool {
    artifacts.tensor_specs().iter().any(|tensor| {
        matches!(
            tensor.role,
            NativeTensorRole::LinearAttentionInProjQkv
                | NativeTensorRole::LinearAttentionInProjZ
                | NativeTensorRole::LinearAttentionInProjA
                | NativeTensorRole::LinearAttentionInProjB
                | NativeTensorRole::LinearAttentionConv1d
                | NativeTensorRole::LinearAttentionDtBias
                | NativeTensorRole::LinearAttentionALog
                | NativeTensorRole::LinearAttentionNorm
                | NativeTensorRole::LinearAttentionOutProj
        )
    })
}

pub(super) fn binding_summary_from_specs(
    specs: &[ax_engine_core::NativeTensorSpec],
) -> NativeModelBindingSummary {
    let mut summary = NativeModelBindingSummary {
        bindings_prepared: true,
        buffers_bound: true,
        buffer_count: specs.len().min(u32::MAX as usize) as u32,
        buffer_bytes: 0,
        source_quantized_binding_count: 0,
        source_q4_k_binding_count: 0,
        source_q5_k_binding_count: 0,
        source_q6_k_binding_count: 0,
        source_q8_0_binding_count: 0,
    };

    for spec in specs {
        summary.buffer_bytes = summary.buffer_bytes.saturating_add(spec.length_bytes);
        if !spec.source_quantized {
            continue;
        }
        summary.source_quantized_binding_count =
            summary.source_quantized_binding_count.saturating_add(1);
        match spec.source_tensor_type.as_deref() {
            Some("q4_k") => {
                summary.source_q4_k_binding_count =
                    summary.source_q4_k_binding_count.saturating_add(1);
            }
            Some("q5_k") => {
                summary.source_q5_k_binding_count =
                    summary.source_q5_k_binding_count.saturating_add(1);
            }
            Some("q6_k") => {
                summary.source_q6_k_binding_count =
                    summary.source_q6_k_binding_count.saturating_add(1);
            }
            Some("q8_0") => {
                summary.source_q8_0_binding_count =
                    summary.source_q8_0_binding_count.saturating_add(1);
            }
            _ => {}
        }
    }

    summary
}

pub(super) fn resolve_terminal_token_ids(artifacts: &NativeModelArtifacts) -> Vec<u32> {
    let mut token_ids = BTreeSet::new();
    let mut token_strings = BTreeSet::new();
    let stop_on_pad = artifacts.manifest().model_family != "diffusion_gemma";

    for file_name in ["config.json", "tokenizer_config.json"] {
        let Some(value) = read_json_file(&artifacts.root_dir().join(file_name)) else {
            continue;
        };
        collect_token_ids(value.get("eos_token_id"), &mut token_ids);
        collect_token_ids(value.get("eos_token_ids"), &mut token_ids);
        collect_token_strings(value.get("eos_token"), &mut token_strings);
        if stop_on_pad {
            collect_token_ids(value.get("pad_token_id"), &mut token_ids);
            collect_token_strings(value.get("pad_token"), &mut token_strings);
        }
    }

    for token in COMMON_EOT_TOKEN_STRINGS {
        token_strings.insert((*token).to_string());
    }

    if !token_strings.is_empty()
        && let Some(tokenizer) = read_json_file(&artifacts.root_dir().join("tokenizer.json"))
    {
        collect_added_token_ids_for_strings(&tokenizer, &token_strings, &mut token_ids);
    }

    let vocab_size = artifacts.manifest().vocab_size;
    token_ids.retain(|token_id| *token_id < vocab_size);
    token_ids.into_iter().collect()
}

pub(super) fn read_json_file(path: &std::path::Path) -> Option<serde_json::Value> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub(super) fn collect_token_ids(value: Option<&serde_json::Value>, token_ids: &mut BTreeSet<u32>) {
    match value {
        Some(serde_json::Value::Number(number)) => {
            if let Some(token_id) = number.as_u64().and_then(|id| u32::try_from(id).ok()) {
                token_ids.insert(token_id);
            }
        }
        Some(serde_json::Value::Array(values)) => {
            for value in values {
                collect_token_ids(Some(value), token_ids);
            }
        }
        Some(serde_json::Value::Object(object)) => {
            collect_token_ids(object.get("id"), token_ids);
        }
        _ => {}
    }
}

pub(super) fn collect_token_strings(
    value: Option<&serde_json::Value>,
    token_strings: &mut BTreeSet<String>,
) {
    match value {
        Some(serde_json::Value::String(token)) => {
            token_strings.insert(token.clone());
        }
        Some(serde_json::Value::Array(values)) => {
            for value in values {
                collect_token_strings(Some(value), token_strings);
            }
        }
        Some(serde_json::Value::Object(object)) => {
            if let Some(content) = object.get("content") {
                collect_token_strings(Some(content), token_strings);
            }
        }
        _ => {}
    }
}

pub(super) fn collect_added_token_ids_for_strings(
    tokenizer: &serde_json::Value,
    token_strings: &BTreeSet<String>,
    token_ids: &mut BTreeSet<u32>,
) {
    let Some(added_tokens) = tokenizer
        .get("added_tokens")
        .and_then(|value| value.as_array())
    else {
        return;
    };

    for token in added_tokens {
        let Some(content) = token.get("content").and_then(|value| value.as_str()) else {
            continue;
        };
        if !token_strings.contains(content) {
            continue;
        }
        collect_token_ids(token.get("id"), token_ids);
    }
}

pub(super) fn validate_qwen_gated_delta_linear_attention(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if !matches!(manifest.model_family.as_str(), "qwen3_5" | "qwen3_next") {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention is currently supported only for qwen3_5/qwen3_next MLX manifests"
                .to_string(),
        ));
    }
    let cfg = &manifest.linear_attention;
    let Some(key_head_dim) = cfg.key_head_dim else {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.key_head_dim must be configured".to_string(),
        ));
    };
    if key_head_dim % 32 != 0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "linear_attention.key_head_dim {key_head_dim} must be divisible by 32 for the MLX gated-delta kernel"
        )));
    }
    if cfg.num_value_heads.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.num_value_heads must be configured".to_string(),
        ));
    }
    if cfg.num_key_heads.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.num_key_heads must be configured".to_string(),
        ));
    }
    if cfg.value_head_dim.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.value_head_dim must be configured".to_string(),
        ));
    }
    if cfg.conv_kernel_dim.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.conv_kernel_dim must be configured".to_string(),
        ));
    }
    // `resolved_full_attention_interval` falls back to QWEN3_5_DEFAULT_FULL_ATTENTION_INTERVAL
    // when the manifest omits the field, so None here means an explicit zero (or an
    // unsupported family that slipped past the model_family gate above). Reject zero
    // explicitly: `is_linear_layer` uses `is_multiple_of(interval)`, which would silently
    // treat every layer as linear when interval == 0.
    match cfg.resolved_full_attention_interval(&manifest.model_family) {
        Some(0) => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "linear_attention.full_attention_interval must be greater than zero".to_string(),
            ));
        }
        Some(_) => {}
        None => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "linear_attention.full_attention_interval must be configured".to_string(),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_gemma4_interleaved_attention(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    // Families with a runtime path for per-layer sliding/full patterns.
    // GPT-OSS uses alternating sliding-128 / full attention (mlx-lm gpt_oss);
    // Gemma4-class families use SWA interleaving (and optional KV sharing).
    if !matches!(
        manifest.model_family.as_str(),
        "gemma4" | "gemma3" | "gemma4_unified" | "diffusion_gemma" | "embeddinggemma" | "gpt_oss"
    ) {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "interleaved sliding/full attention is not implemented for {} manifests",
            manifest.model_family
        )));
    }
    if manifest.layer_types.len() != manifest.layer_count as usize {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "interleaved attention requires one layer_type per layer, got {} for {} layers",
            manifest.layer_types.len(),
            manifest.layer_count
        )));
    }

    for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
        if layer_type != "sliding_attention" && layer_type != "full_attention" {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "layer {idx} uses unsupported layer_type {layer_type:?}"
            )));
        }
    }

    let has_sliding = manifest
        .layer_types
        .iter()
        .any(|layer_type| layer_type == "sliding_attention");
    if has_sliding {
        match manifest.sliding_window_size {
            None => {
                return Err(MlxRunnerError::UnsupportedFeature(
                    "sliding_attention layers require sliding_window_size".to_string(),
                ));
            }
            Some(0) => {
                // build_layer_configs maps Some(0) to Some(0), and the cache path then
                // filters it back to None — sliding layers would silently degrade to a
                // grow-forever window. Reject up front instead of running with a layout
                // the user did not ask for.
                return Err(MlxRunnerError::UnsupportedFeature(
                    "sliding_window_size must be greater than zero".to_string(),
                ));
            }
            Some(_) => {}
        }
    }

    for (&layer, &source) in &manifest.kv_shared_source_layers {
        if layer >= manifest.layer_count || source >= manifest.layer_count || source >= layer {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "KV-shared layer {layer} has invalid source layer {source}"
            )));
        }
        let layer_type = &manifest.layer_types[layer as usize];
        let source_type = &manifest.layer_types[source as usize];
        if layer_type != source_type {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "KV-shared layer {layer} type {layer_type:?} cannot reuse source {source} type {source_type:?}"
            )));
        }
        // Chained KV sharing would panic at runtime in `MlxKVCache::peek_source_kv`
        // (the source layer never writes its own K/V, so the cached entry is None
        // and the `.expect("…source layer must appear earlier")` fires). Reject it
        // here so the manifest fails closed instead of producing a midstream panic.
        if manifest.kv_shared_source_layers.contains_key(&source) {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "KV-shared layer {layer} cannot use shared layer {source} as its source"
            )));
        }
    }

    Ok(())
}
