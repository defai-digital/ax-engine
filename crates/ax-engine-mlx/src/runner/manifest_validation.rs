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
    MlxRunnerAdmission, NativeModelArtifacts, NativeModelManifest, NativeTensorDataType,
    NativeTensorRole, NativeTensorSpec, mlx_runner_admission_for_family,
    runner::NativeModelBindingSummary,
};

use super::{COMMON_EOT_TOKEN_STRINGS, MlxRunnerError};

pub(super) fn validate_mlx_supported_manifest(
    artifacts: &NativeModelArtifacts,
) -> Result<(), MlxRunnerError> {
    let manifest = artifacts.manifest();
    validate_mlx_primary_admission(&manifest.model_family)?;
    // Family-scoped qwen4_exp contract (hyper-connection sites, QSA indexer +
    // gate-packed q_proj geometry, PLE module + n-gram shards, root mixer).
    // The family trunk (`model::qwen4_exp_forward_*`) serves admitted packs.
    if manifest.model_family == "qwen4_exp" {
        validate_qwen4_exp_manifest(manifest)?;
    }
    if manifest.model_family == "glm4_moe_lite"
        || manifest.model_family == "deepseek_v4"
        || has_glm_mla_tensors(artifacts)
    {
        validate_mla_moe_manifest(manifest)?;
    }
    if manifest.model_family != "nemotron_h"
        && (manifest.linear_attention.is_enabled() || has_linear_attention_tensors(artifacts))
    {
        // Nemotron-H reuses linear_attention dims for Mamba-2; skip Qwen gated-delta contract.
        validate_qwen_gated_delta_linear_attention(manifest)?;
    }
    if manifest.model_family == "llama4" {
        validate_llama4_manifest(manifest)?;
    }
    // Interleaved SWA validation (Gemma3/4): triggered by layer_types, KV sharing,
    // a separate global head dim, or a separate SWA rope theta. Families with
    // uniform SWA (mistral3, mixtral) use only sliding_window_size with no
    // layer_types, so they skip this gate. Nemotron-H also uses layer_types for
    // hybrid mixer kinds (mamba/attention/moe) and must not enter this path.
    if manifest.model_family != "nemotron_h"
        && (!manifest.layer_types.is_empty()
            || !manifest.kv_shared_source_layers.is_empty()
            || manifest.global_head_dim.is_some()
            || manifest.rope_theta_swa.is_some())
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

pub(super) fn validate_mlx_primary_admission(model_family: &str) -> Result<(), MlxRunnerError> {
    match mlx_runner_admission_for_family(model_family) {
        Some(MlxRunnerAdmission::Primary) => Ok(()),
        Some(MlxRunnerAdmission::AuxiliaryOnly) => {
            Err(MlxRunnerError::UnsupportedFeature(format!(
                "model_family {model_family:?} is an auxiliary-only artifact and cannot be loaded as the primary MLX runner"
            )))
        }
        None => Err(MlxRunnerError::UnsupportedFeature(format!(
            "model_family {model_family:?} is not supported by the MLX runner"
        ))),
    }
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
    if manifest.model_family == "deepseek_v4" {
        return validate_deepseek_v4_manifest(manifest);
    }
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

/// Validate the DeepSeek V4 manifest contract at load time.
///
/// Per-layer tensor-role structure is already enforced by the core manifest
/// validator (`validate_native_model_manifest`) before this runner gate runs;
/// here we only require the V4 config block to be present and consistent.
pub(super) fn validate_deepseek_v4_manifest(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if manifest.deepseek_v4.is_disabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "deepseek_v4 requires deepseek_v4 metadata in the manifest".to_string(),
        ));
    }
    if manifest.mla_attention.is_enabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "deepseek_v4 must not carry mla_attention metadata (V4 drops the V3 MLA keys)"
                .to_string(),
        ));
    }
    if manifest.moe.sigmoid_routing {
        return Err(MlxRunnerError::UnsupportedFeature(
            "deepseek_v4 must not enable moe.sigmoid_routing (routing is scoring_func-based)"
                .to_string(),
        ));
    }
    if manifest.deepseek_v4.compress_ratios.len() != manifest.layer_count as usize {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "deepseek_v4.compress_ratios must contain one entry per layer, got {} for layer_count {}",
            manifest.deepseek_v4.compress_ratios.len(),
            manifest.layer_count
        )));
    }
    if manifest
        .deepseek_v4
        .num_hash_layers
        .is_none_or(|layers| layers > manifest.layer_count)
    {
        return Err(MlxRunnerError::UnsupportedFeature(
            "deepseek_v4.num_hash_layers must be configured and <= layer_count".to_string(),
        ));
    }

    Ok(())
}

/// Validate the qwen4_exp (Qwen3.8-Flash-Next) manifest contract.
///
/// Called from `validate_mlx_supported_manifest` AFTER the fail-closed guard,
/// so it only gates the runner once the family trunk lands; unit tests
/// exercise it directly. The family has no `model.norm` and no per-layer
/// input/post layernorms — hyper-connections replace them and the loader
/// mirrors the hyper-connection norms into the shared norm slots — so the
/// generic FinalNorm/AttentionNorm requirements are exempted family-scoped in
/// core manifest validation (`validate_native_model_manifest`). This arm
/// instead requires the family tensors the loader resolves by exact
/// checkpoint name: both hyper-connection sites on every layer, the QSA
/// indexer plus gate-packed `q_proj` geometry (rows = 2 × heads × head_dim)
/// on full-attention layers, the PLE module (resident tensors, I64 hash
/// buffers, contiguous n-gram shards) on each `ple_layer_ids` entry, and the
/// root hyper-connection mixer.
pub(super) fn validate_qwen4_exp_manifest(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    let cfg = &manifest.qwen4_exp;
    if cfg.is_disabled() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "qwen4_exp requires qwen4_exp metadata in the manifest".to_string(),
        ));
    }
    // Absent fields fall back to the same reference-config defaults the
    // converter (`qwen4_exp_config`) and the typed config loader
    // (`Qwen4ExpConfig::from_manifest`) apply.
    let hc_count = u64::from(cfg.hc_count.unwrap_or(4));
    let hc_lowrank = u64::from(cfg.hc_lowrank.unwrap_or(320));
    if hc_count == 0 || hc_lowrank == 0 {
        return Err(MlxRunnerError::UnsupportedFeature(
            "qwen4_exp hc_count/hc_lowrank must be greater than zero".to_string(),
        ));
    }
    // QSA full-attention layers pack the sigmoid output gate into q_proj, so
    // attention head dims only resolve after halving the q_proj row count.
    if !manifest.attn_output_gate {
        return Err(MlxRunnerError::UnsupportedFeature(
            "qwen4_exp requires attn_output_gate (the sigmoid gate packs into self_attn.q_proj)"
                .to_string(),
        ));
    }
    // The reference architecture's gated-delta output gate is sigmoid (the
    // one activation difference from the shared stack). An absent field
    // defaults to sigmoid in the typed config; a contradictory value would
    // silently run the Qwen3.5 silu gate, so fail closed.
    if let Some(kind) = cfg.output_gate_type.as_deref()
        && !kind.eq_ignore_ascii_case("sigmoid")
    {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "qwen4_exp output_gate_type must be \"sigmoid\", got {kind:?}"
        )));
    }
    let interval = match manifest
        .linear_attention
        .resolved_full_attention_interval("qwen4_exp")
    {
        Some(0) => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "qwen4_exp linear_attention.full_attention_interval must be greater than zero"
                    .to_string(),
            ));
        }
        Some(value) => u64::from(value),
        None => {
            return Err(MlxRunnerError::UnsupportedFeature(
                "qwen4_exp requires linear_attention.full_attention_interval".to_string(),
            ));
        }
    };
    let hidden = u64::from(manifest.hidden_size);
    let hc_width = hidden.saturating_mul(hc_count);
    let indexer_head_dim = u64::from(cfg.indexer_head_dim.unwrap_or(128));
    let indexer_rows = u64::from(cfg.indexer_n_heads.unwrap_or(4))
        .saturating_add(u64::from(cfg.indexer_kv_heads.unwrap_or(1)))
        .saturating_mul(indexer_head_dim);
    let gate_packed_q_rows = u64::from(manifest.attention_head_count)
        .saturating_mul(u64::from(manifest.attention_head_dim))
        .saturating_mul(2);

    for layer_index in 0..manifest.layer_count {
        let base = qwen4_exp_layer_base(manifest, layer_index).ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(format!(
                "qwen4_exp layer {layer_index} is missing attn_hyper_connection tensors"
            ))
        })?;
        for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
            let prefix = format!("{base}.{site}");
            expect_qwen4_exp_vector(
                require_qwen4_exp_tensor(manifest, &format!("{prefix}.hc_norm.weight"))?,
                hc_width,
            )?;
            expect_qwen4_exp_matrix_rows(
                require_qwen4_exp_tensor(
                    manifest,
                    &format!("{prefix}.input_mix_weight_down.weight"),
                )?,
                hc_lowrank,
                hc_width,
            )?;
            expect_qwen4_exp_matrix_rows(
                require_qwen4_exp_tensor(
                    manifest,
                    &format!("{prefix}.input_mix_weight_up.weight"),
                )?,
                hc_width,
                hc_lowrank,
            )?;
            expect_qwen4_exp_matrix_rows(
                require_qwen4_exp_tensor(
                    manifest,
                    &format!("{prefix}.block_inject_weight.weight"),
                )?,
                hc_count,
                hc_width,
            )?;
        }

        if (u64::from(layer_index) + 1).is_multiple_of(interval) {
            // QSA (sparse full-attention) layer: indexer trio plus the
            // gate-packed q_proj row geometry.
            let indexer = format!("{base}.self_attn.indexer");
            expect_qwen4_exp_matrix_rows(
                require_qwen4_exp_tensor(manifest, &format!("{indexer}.index_qk_proj.weight"))?,
                indexer_rows,
                hidden,
            )?;
            expect_qwen4_exp_vector(
                require_qwen4_exp_tensor(manifest, &format!("{indexer}.q_layernorm.weight"))?,
                indexer_head_dim,
            )?;
            expect_qwen4_exp_vector(
                require_qwen4_exp_tensor(manifest, &format!("{indexer}.k_layernorm.weight"))?,
                indexer_head_dim,
            )?;
            let q_proj = manifest
                .tensors
                .iter()
                .find(|tensor| {
                    tensor.role == NativeTensorRole::AttentionQ
                        && tensor.layer_index == Some(layer_index)
                })
                .ok_or_else(|| {
                    MlxRunnerError::UnsupportedFeature(format!(
                        "qwen4_exp QSA layer {layer_index} is missing self_attn.q_proj"
                    ))
                })?;
            if q_proj.shape.first() != Some(&gate_packed_q_rows) {
                return Err(MlxRunnerError::UnsupportedFeature(format!(
                    "qwen4_exp QSA layer {layer_index} q_proj rows must be 2 * heads * head_dim ({gate_packed_q_rows}), got {:?}",
                    q_proj.shape
                )));
            }
        }
    }

    // PLE lives on the layers listed in `ple_layer_ids` (1-based ids).
    let ple_kernel = u64::from(cfg.ple_conv_kernel_size.unwrap_or(4));
    for &ple_id in &cfg.ple_layer_ids {
        if ple_id == 0 || ple_id > manifest.layer_count {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "qwen4_exp ple_layer_ids entry {ple_id} is out of range for layer_count {} (ids are 1-based)",
                manifest.layer_count
            )));
        }
        let layer_index = ple_id - 1;
        let base = qwen4_exp_layer_base(manifest, layer_index).ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(format!(
                "qwen4_exp PLE layer {layer_index} is missing attn_hyper_connection tensors"
            ))
        })?;
        let ple = format!("{base}.ple");
        expect_qwen4_exp_matrix_rows(
            require_qwen4_exp_tensor(manifest, &format!("{ple}.key_proj.weight"))?,
            hc_width,
            hidden,
        )?;
        expect_qwen4_exp_matrix_rows(
            require_qwen4_exp_tensor(manifest, &format!("{ple}.value_proj.weight"))?,
            hidden,
            hidden,
        )?;
        for norm in ["norm_key", "norm_query", "norm_conv"] {
            expect_qwen4_exp_vector(
                require_qwen4_exp_tensor(manifest, &format!("{ple}.{norm}.weight"))?,
                hc_width,
            )?;
        }
        let conv = require_qwen4_exp_tensor(manifest, &format!("{ple}.conv1d.weight"))?;
        if conv.shape != [hc_width, ple_kernel, 1] {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "qwen4_exp tensor {} must have shape [{hc_width}, {ple_kernel}, 1], got {:?}",
                conv.name, conv.shape
            )));
        }
        let embedding = format!("{ple}.ple_embedding");
        for buffer in [
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        ] {
            let spec = require_qwen4_exp_tensor(manifest, &format!("{embedding}.{buffer}"))?;
            if spec.dtype != NativeTensorDataType::I64 {
                return Err(MlxRunnerError::UnsupportedFeature(format!(
                    "qwen4_exp PLE hash buffer {} must have dtype i64, got {:?}",
                    spec.name, spec.dtype
                )));
            }
        }
        // The n-gram table ships as contiguous row shards from index 0; the
        // loader gathers by global row id, so gaps or a short shard set would
        // silently mis-gather.
        let marker = format!("{embedding}.ngram_embedding.shards.");
        let mut shard_indices: Vec<u32> = manifest
            .tensors
            .iter()
            .filter_map(|tensor| {
                tensor
                    .name
                    .strip_prefix(&marker)?
                    .strip_suffix(".weight")?
                    .parse::<u32>()
                    .ok()
            })
            .collect();
        shard_indices.sort_unstable();
        shard_indices.dedup();
        if shard_indices.is_empty() {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "qwen4_exp PLE layer {layer_index} is missing n-gram table shards under {marker}"
            )));
        }
        for (expected, index) in shard_indices.iter().enumerate() {
            if *index != expected as u32 {
                return Err(MlxRunnerError::UnsupportedFeature(format!(
                    "qwen4_exp PLE n-gram shards are not contiguous from 0: found index {index} at position {expected}"
                )));
            }
        }
        let split_parts = cfg.split_ngram_parts.unwrap_or(128) as usize;
        if shard_indices.len() != split_parts {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "qwen4_exp PLE n-gram table ships {} shard(s) but split_ngram_parts is {split_parts}",
                shard_indices.len()
            )));
        }
    }

    // Root-level final hyper-connection mixer (no block-injection gate).
    let mixer = [
        "language_model.model.hyper_connection_mixer",
        "model.hyper_connection_mixer",
        "model.language_model.hyper_connection_mixer",
    ]
    .into_iter()
    .find(|prefix| {
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.name == format!("{prefix}.hc_norm.weight"))
    })
    .ok_or_else(|| {
        MlxRunnerError::UnsupportedFeature(
            "qwen4_exp is missing the root model.hyper_connection_mixer tensors".to_string(),
        )
    })?;
    expect_qwen4_exp_vector(
        require_qwen4_exp_tensor(manifest, &format!("{mixer}.hc_norm.weight"))?,
        hc_width,
    )?;
    expect_qwen4_exp_matrix_rows(
        require_qwen4_exp_tensor(manifest, &format!("{mixer}.input_mix_weight_down.weight"))?,
        hc_lowrank,
        hc_width,
    )?;
    expect_qwen4_exp_matrix_rows(
        require_qwen4_exp_tensor(manifest, &format!("{mixer}.input_mix_weight_up.weight"))?,
        hc_width,
        hc_lowrank,
    )?;
    Ok(())
}

/// Candidate per-layer checkpoint prefixes for qwen4_exp family tensors.
/// Must mirror the loader's probe prefixes in weights.rs.
fn qwen4_exp_layer_base(manifest: &NativeModelManifest, layer_index: u32) -> Option<String> {
    [
        "language_model.model.layers.",
        "model.layers.",
        "model.language_model.layers.",
    ]
    .into_iter()
    .map(|prefix| format!("{prefix}{layer_index}"))
    .find(|base| {
        manifest
            .tensors
            .iter()
            .any(|tensor| tensor.name == format!("{base}.attn_hyper_connection.hc_norm.weight"))
    })
}

fn require_qwen4_exp_tensor<'a>(
    manifest: &'a NativeModelManifest,
    name: &str,
) -> Result<&'a NativeTensorSpec, MlxRunnerError> {
    manifest
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| {
            MlxRunnerError::UnsupportedFeature(format!(
                "qwen4_exp manifest is missing required tensor {name}"
            ))
        })
}

fn expect_qwen4_exp_vector(spec: &NativeTensorSpec, len: u64) -> Result<(), MlxRunnerError> {
    if spec.shape == [len] {
        return Ok(());
    }
    Err(MlxRunnerError::UnsupportedFeature(format!(
        "qwen4_exp tensor {} must have shape [{len}], got {:?}",
        spec.name, spec.shape
    )))
}

/// Row-exact matrix shape check. Quantized tensors record the packed file
/// shape (columns packed), so the column count is only checked for
/// unquantized storage; row counts stay logical in both layouts.
fn expect_qwen4_exp_matrix_rows(
    spec: &NativeTensorSpec,
    rows: u64,
    cols: u64,
) -> Result<(), MlxRunnerError> {
    let shape_ok = spec.shape.len() == 2
        && spec.shape[0] == rows
        && (spec.source_quantized || spec.shape[1] == cols);
    if shape_ok {
        return Ok(());
    }
    Err(MlxRunnerError::UnsupportedFeature(format!(
        "qwen4_exp tensor {} must have shape [{rows}, {cols}], got {:?}",
        spec.name, spec.shape
    )))
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
    if !matches!(
        manifest.model_family.as_str(),
        "qwen3_5" | "qwen3_next" | "minicpmv4_6" | "qwen4_exp"
    ) {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention is currently supported only for qwen3_5/qwen3_next/qwen4_exp/MiniCPM-V 4.6 MLX manifests".to_string(),
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
    // Muse Glimmer interleaves SWA(2048)+RoPE layers with NoPE full layers
    // through the dedicated muse_glimmer route.
    if !matches!(
        manifest.model_family.as_str(),
        "gemma4"
            | "gemma4_vl"
            | "gemma3"
            | "gemma4_unified"
            | "diffusion_gemma"
            | "embeddinggemma"
            | "gpt_oss"
            | "muse_glimmer"
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

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use std::fs;
    use std::path::PathBuf;

    use ax_engine_core::{
        AX_NATIVE_MODEL_MANIFEST_FILE, NativeDiffusionConfig, NativeLinearAttentionConfig,
        NativeMoeConfig, NativeQwen4ExpConfig, NativeRuntimeStatus, NativeTensorFormat,
        NativeTensorQuantization, WeightSanitize,
    };

    use super::*;

    fn tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    fn i64_tensor(name: &str, layer_index: Option<u32>, shape: Vec<u64>) -> NativeTensorSpec {
        NativeTensorSpec {
            dtype: NativeTensorDataType::I64,
            ..tensor(name, NativeTensorRole::Other, layer_index, shape)
        }
    }

    fn ple_shard_tensor(name: &str, layer_index: u32) -> NativeTensorSpec {
        NativeTensorSpec {
            dtype: NativeTensorDataType::U32,
            source_quantized: true,
            quantization: Some(NativeTensorQuantization {
                mode: "affine".to_string(),
                group_size: 32,
                bits: 8,
            }),
            ..tensor(name, NativeTensorRole::Other, Some(layer_index), vec![8, 4])
        }
    }

    /// Minimal valid qwen4_exp manifest: one gated-delta layer (0), one QSA
    /// layer (1), PLE on 0-indexed layer 1 (`ple_layer_ids` is 1-based), the
    /// root hyper-connection mixer, and no norm tensors anywhere — the layout
    /// real checkpoints convert to.
    fn qwen4_exp_test_manifest() -> NativeModelManifest {
        let mut tensors = vec![
            tensor(
                "language_model.model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                vec![32, 8],
            ),
            tensor(
                "language_model.lm_head.weight",
                NativeTensorRole::LmHead,
                None,
                vec![32, 8],
            ),
            tensor(
                "language_model.model.hyper_connection_mixer.hc_norm.weight",
                NativeTensorRole::Other,
                None,
                vec![32],
            ),
            tensor(
                "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
                NativeTensorRole::Other,
                None,
                vec![2, 32],
            ),
            tensor(
                "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
                NativeTensorRole::Other,
                None,
                vec![32, 2],
            ),
        ];
        for layer in 0..2u32 {
            for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
                let base = format!("language_model.model.layers.{layer}.{site}");
                tensors.extend([
                    tensor(
                        &format!("{base}.hc_norm.weight"),
                        NativeTensorRole::Other,
                        Some(layer),
                        vec![32],
                    ),
                    tensor(
                        &format!("{base}.input_mix_weight_down.weight"),
                        NativeTensorRole::Other,
                        Some(layer),
                        vec![2, 32],
                    ),
                    tensor(
                        &format!("{base}.input_mix_weight_up.weight"),
                        NativeTensorRole::Other,
                        Some(layer),
                        vec![32, 2],
                    ),
                    tensor(
                        &format!("{base}.block_inject_weight.weight"),
                        NativeTensorRole::Other,
                        Some(layer),
                        vec![4, 32],
                    ),
                ]);
            }
            let mlp = format!("language_model.model.layers.{layer}.mlp");
            tensors.extend([
                tensor(
                    &format!("{mlp}.gate.weight"),
                    NativeTensorRole::FfnGateInp,
                    Some(layer),
                    vec![4, 8],
                ),
                tensor(
                    &format!("{mlp}.switch_mlp.gate_proj.weight"),
                    NativeTensorRole::FfnGateExps,
                    Some(layer),
                    vec![4, 8, 8],
                ),
                tensor(
                    &format!("{mlp}.switch_mlp.up_proj.weight"),
                    NativeTensorRole::FfnUpExps,
                    Some(layer),
                    vec![4, 8, 8],
                ),
                tensor(
                    &format!("{mlp}.switch_mlp.down_proj.weight"),
                    NativeTensorRole::FfnDownExps,
                    Some(layer),
                    vec![4, 8, 8],
                ),
                tensor(
                    &format!("{mlp}.shared_expert_gate.weight"),
                    NativeTensorRole::FfnSharedExpertGateInp,
                    Some(layer),
                    vec![1, 8],
                ),
                tensor(
                    &format!("{mlp}.shared_expert.gate_proj.weight"),
                    NativeTensorRole::FfnSharedExpertGate,
                    Some(layer),
                    vec![8, 8],
                ),
                tensor(
                    &format!("{mlp}.shared_expert.up_proj.weight"),
                    NativeTensorRole::FfnSharedExpertUp,
                    Some(layer),
                    vec![8, 8],
                ),
                tensor(
                    &format!("{mlp}.shared_expert.down_proj.weight"),
                    NativeTensorRole::FfnSharedExpertDown,
                    Some(layer),
                    vec![8, 8],
                ),
            ]);
        }
        let linear = "language_model.model.layers.0.linear_attn";
        tensors.extend([
            tensor(
                &format!("{linear}.in_proj_qkv.weight"),
                NativeTensorRole::LinearAttentionInProjQkv,
                Some(0),
                vec![128, 8],
            ),
            tensor(
                &format!("{linear}.in_proj_z.weight"),
                NativeTensorRole::LinearAttentionInProjZ,
                Some(0),
                vec![64, 8],
            ),
            tensor(
                &format!("{linear}.in_proj_a.weight"),
                NativeTensorRole::LinearAttentionInProjA,
                Some(0),
                vec![2, 8],
            ),
            tensor(
                &format!("{linear}.in_proj_b.weight"),
                NativeTensorRole::LinearAttentionInProjB,
                Some(0),
                vec![2, 8],
            ),
            tensor(
                &format!("{linear}.conv1d.weight"),
                NativeTensorRole::LinearAttentionConv1d,
                Some(0),
                vec![128, 1, 4],
            ),
            tensor(
                &format!("{linear}.dt_bias"),
                NativeTensorRole::LinearAttentionDtBias,
                Some(0),
                vec![2],
            ),
            tensor(
                &format!("{linear}.A_log"),
                NativeTensorRole::LinearAttentionALog,
                Some(0),
                vec![2],
            ),
            tensor(
                &format!("{linear}.norm.weight"),
                NativeTensorRole::LinearAttentionNorm,
                Some(0),
                vec![32],
            ),
            tensor(
                &format!("{linear}.out_proj.weight"),
                NativeTensorRole::LinearAttentionOutProj,
                Some(0),
                vec![8, 64],
            ),
        ]);
        let attn = "language_model.model.layers.1.self_attn";
        tensors.extend([
            tensor(
                &format!("{attn}.q_proj.weight"),
                NativeTensorRole::AttentionQ,
                Some(1),
                vec![32, 8],
            ),
            tensor(
                &format!("{attn}.k_proj.weight"),
                NativeTensorRole::AttentionK,
                Some(1),
                vec![8, 8],
            ),
            tensor(
                &format!("{attn}.v_proj.weight"),
                NativeTensorRole::AttentionV,
                Some(1),
                vec![8, 8],
            ),
            tensor(
                &format!("{attn}.o_proj.weight"),
                NativeTensorRole::AttentionO,
                Some(1),
                vec![8, 16],
            ),
            tensor(
                &format!("{attn}.q_norm.weight"),
                NativeTensorRole::AttentionQNorm,
                Some(1),
                vec![8],
            ),
            tensor(
                &format!("{attn}.k_norm.weight"),
                NativeTensorRole::AttentionKNorm,
                Some(1),
                vec![8],
            ),
            tensor(
                &format!("{attn}.indexer.index_qk_proj.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![20, 8],
            ),
            tensor(
                &format!("{attn}.indexer.q_layernorm.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![4],
            ),
            tensor(
                &format!("{attn}.indexer.k_layernorm.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![4],
            ),
        ]);
        let ple = "language_model.model.layers.1.ple";
        tensors.extend([
            tensor(
                &format!("{ple}.key_proj.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![32, 8],
            ),
            tensor(
                &format!("{ple}.value_proj.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![8, 8],
            ),
            tensor(
                &format!("{ple}.norm_key.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![32],
            ),
            tensor(
                &format!("{ple}.norm_query.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![32],
            ),
            tensor(
                &format!("{ple}.norm_conv.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![32],
            ),
            tensor(
                &format!("{ple}.conv1d.weight"),
                NativeTensorRole::Other,
                Some(1),
                vec![32, 4, 1],
            ),
            i64_tensor(
                &format!("{ple}.ple_embedding.layer_multipliers"),
                Some(1),
                vec![3],
            ),
            i64_tensor(
                &format!("{ple}.ple_embedding.ngram_heads_vocab_sizes"),
                Some(1),
                vec![16],
            ),
            i64_tensor(
                &format!("{ple}.ple_embedding.ngram_heads_offsets"),
                Some(1),
                vec![16],
            ),
        ]);
        for shard in 0..4u32 {
            tensors.push(ple_shard_tensor(
                &format!("{ple}.ple_embedding.ngram_embedding.shards.{shard}.weight"),
                1,
            ));
        }

        NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen4_exp".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 2,
            hidden_size: 8,
            intermediate_size: 0,
            attention_head_count: 2,
            attention_head_dim: 8,
            kv_head_count: 1,
            vocab_size: 32,
            tie_word_embeddings: false,
            rope_theta: Some(10_000_000),
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
            partial_rotary_factor: Some(0.25),
            rms_norm_eps: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            global_kv_head_count: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            final_logits_scale: None,
            attention_scale_multiplier: None,
            post_norm_eps: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: true,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig {
                full_attention_interval: Some(2),
                num_value_heads: Some(2),
                num_key_heads: Some(1),
                key_head_dim: Some(32),
                value_head_dim: Some(32),
                conv_kernel_dim: Some(4),
            },
            mla_attention: Default::default(),
            moe: NativeMoeConfig {
                expert_count: Some(4),
                experts_per_token: Some(2),
                expert_intermediate_size: Some(8),
                ..Default::default()
            },
            glm_router: Default::default(),
            deepseek_v4: Default::default(),
            qwen4_exp: NativeQwen4ExpConfig {
                hc_count: Some(4),
                hc_lowrank: Some(2),
                indexer_head_dim: Some(4),
                indexer_kv_heads: Some(1),
                indexer_n_heads: Some(4),
                split_ngram_parts: Some(4),
                ple_conv_kernel_size: Some(4),
                ple_embed_dim: Some(16),
                ple_layer_ids: vec![2],
                ..Default::default()
            },
            weight_sanitize: WeightSanitize::None,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: NativeDiffusionConfig::default(),
            dropped_tensors: Default::default(),
            kv_cache_quantization: None,
            tensors,
        }
    }

    fn write_artifacts(manifest: NativeModelManifest) -> NativeModelArtifacts {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ax-mlx-qwen4-exp-manifest-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("fixture directory should create");
        fs::write(dir.join("model.safetensors"), vec![0_u8; 4096]).expect("weights should write");
        fs::write(
            dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
        NativeModelArtifacts::from_dir(&dir).expect("fixture manifest should validate")
    }

    #[test]
    fn qwen4_exp_arm_accepts_minimal_valid_manifest() {
        let manifest = qwen4_exp_test_manifest();
        validate_qwen4_exp_manifest(&manifest).expect("minimal qwen4_exp manifest should pass");
        // The norm-free fixture must also clear core manifest validation
        // (the family-scoped FinalNorm/AttentionNorm exemptions).
        write_artifacts(manifest);
    }

    #[test]
    fn qwen4_exp_arm_requires_indexer_on_qsa_layers() {
        let mut manifest = qwen4_exp_test_manifest();
        manifest
            .tensors
            .retain(|spec| !spec.name.contains(".indexer."));

        let error = validate_qwen4_exp_manifest(&manifest)
            .expect_err("a QSA layer without its indexer must fail");

        assert!(
            error.to_string().contains("indexer.index_qk_proj"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn qwen4_exp_arm_requires_ple_on_ple_layer() {
        let mut missing_resident = qwen4_exp_test_manifest();
        missing_resident.tensors.retain(|spec| {
            !(spec.name.contains(".ple.") && !spec.name.contains(".ple_embedding."))
        });
        let error = validate_qwen4_exp_manifest(&missing_resident)
            .expect_err("a PLE layer without resident PLE tensors must fail");
        assert!(
            error.to_string().contains("ple.key_proj"),
            "unexpected error: {error}"
        );

        let mut missing_shards = qwen4_exp_test_manifest();
        missing_shards
            .tensors
            .retain(|spec| !spec.name.contains(".ngram_embedding.shards."));
        let error = validate_qwen4_exp_manifest(&missing_shards)
            .expect_err("a PLE layer without n-gram shards must fail");
        assert!(
            error.to_string().contains("n-gram table shards"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn qwen4_exp_family_arm_is_the_live_runner_gate() {
        // Phase 1 flipped the fail-closed guard: a manifest that satisfies the
        // family arm is admitted by the runner entry, and the arm (not a
        // blanket rejection) is now the gate for malformed manifests.
        let artifacts = write_artifacts(qwen4_exp_test_manifest());
        validate_mlx_supported_manifest(&artifacts)
            .expect("qwen4_exp must be admitted now that the family trunk has landed");

        let mut malformed = qwen4_exp_test_manifest();
        malformed
            .tensors
            .retain(|spec| !spec.name.contains(".indexer."));
        let artifacts = write_artifacts(malformed);
        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("a QSA layer without its indexer must fail the family arm");
        assert!(
            error.to_string().contains("indexer.index_qk_proj"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn qwen4_exp_arm_rejects_genuinely_malformed_manifests() {
        for (label, mutate) in [
            (
                "missing family metadata",
                Box::new(|m: &mut NativeModelManifest| {
                    m.qwen4_exp = NativeQwen4ExpConfig::default();
                }) as Box<dyn Fn(&mut NativeModelManifest)>,
            ),
            (
                "attn_output_gate disabled",
                Box::new(|m: &mut NativeModelManifest| m.attn_output_gate = false),
            ),
            (
                "q_proj rows not gate-packed",
                Box::new(|m: &mut NativeModelManifest| {
                    for spec in &mut m.tensors {
                        if spec.role == NativeTensorRole::AttentionQ {
                            spec.shape = vec![16, 8];
                        }
                    }
                }),
            ),
            (
                "ple_layer_ids out of range",
                Box::new(|m: &mut NativeModelManifest| m.qwen4_exp.ple_layer_ids = vec![3]),
            ),
            (
                "I64 hash buffer with float dtype",
                Box::new(|m: &mut NativeModelManifest| {
                    for spec in &mut m.tensors {
                        if spec.name.ends_with("ple_embedding.layer_multipliers") {
                            spec.dtype = NativeTensorDataType::F32;
                        }
                    }
                }),
            ),
            (
                "missing root hyper-connection mixer",
                Box::new(|m: &mut NativeModelManifest| {
                    m.tensors
                        .retain(|spec| !spec.name.contains(".hyper_connection_mixer."));
                }),
            ),
        ] {
            let mut manifest = qwen4_exp_test_manifest();
            mutate(&mut manifest);
            let error = validate_qwen4_exp_manifest(&manifest)
                .expect_err(&format!("{label} should fail closed"));
            assert!(
                error.to_string().contains("qwen4_exp"),
                "{label}: unexpected error message: {error}"
            );
        }
    }

    #[test]
    fn qwen4_exp_arm_defaults_output_gate_to_sigmoid_and_rejects_contradictions() {
        // Absent field: valid (the typed config defaults it to sigmoid).
        let manifest = qwen4_exp_test_manifest();
        validate_qwen4_exp_manifest(&manifest)
            .expect("absent output_gate_type defaults to sigmoid and stays valid");

        // Explicit sigmoid (any case): valid.
        let mut manifest = qwen4_exp_test_manifest();
        manifest.qwen4_exp.output_gate_type = Some("SIGMOID".to_string());
        validate_qwen4_exp_manifest(&manifest).expect("case-insensitive sigmoid passes");

        // A contradictory value fails closed — it would otherwise silently
        // take the Qwen3.5 silu gate in the shared gated-delta stack.
        let mut manifest = qwen4_exp_test_manifest();
        manifest.qwen4_exp.output_gate_type = Some("silu".to_string());
        let error = validate_qwen4_exp_manifest(&manifest)
            .expect_err("a silu output gate must fail closed");
        assert!(
            error.to_string().contains("output_gate_type"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn qwen4_exp_arm_is_scoped_to_the_family() {
        // A manifest without the qwen4_exp block never passes the arm, and
        // the arm's requirements never leak into other families (the guard
        // in `validate_mlx_supported_manifest` is family-string gated too).
        let mut manifest = qwen4_exp_test_manifest();
        manifest.model_family = "qwen3_5".to_string();
        manifest.qwen4_exp = NativeQwen4ExpConfig::default();

        let error = validate_qwen4_exp_manifest(&manifest)
            .expect_err("non-qwen4_exp manifests must not pass the family arm");

        assert!(
            error.to_string().contains("requires qwen4_exp metadata"),
            "unexpected error: {error}"
        );
    }
}
