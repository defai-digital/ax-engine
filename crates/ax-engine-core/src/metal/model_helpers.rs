use super::*;

#[cfg(target_os = "macos")]
pub(super) fn native_model_rms_norm_epsilon(artifacts: &NativeModelArtifacts) -> f32 {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("qwen") || family.starts_with("gemma") {
        1e-6
    } else {
        1e-5
    }
}

#[cfg(target_os = "macos")]
pub(super) fn native_model_rms_norm_weight_offset(artifacts: &NativeModelArtifacts) -> f32 {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("gemma") {
        1.0
    } else {
        0.0
    }
}

#[cfg(target_os = "macos")]
pub(super) fn native_model_embedding_scale(artifacts: &NativeModelArtifacts) -> f32 {
    if let Some(scale) = artifacts.manifest().hidden_states_scale {
        return scale;
    }
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("gemma") {
        (artifacts.manifest().hidden_size as f32).sqrt()
    } else {
        1.0
    }
}

#[cfg(target_os = "macos")]
pub(super) fn native_model_ffn_activation(artifacts: &NativeModelArtifacts) -> ModelFfnActivation {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("gemma") {
        ModelFfnActivation::GeluApprox
    } else {
        ModelFfnActivation::Silu
    }
}

#[cfg(target_os = "macos")]
pub(super) fn native_model_rope_theta(artifacts: &NativeModelArtifacts) -> f32 {
    artifacts
        .manifest()
        .rope_theta
        .map(|rope_theta| rope_theta as f32)
        .unwrap_or(PHASE1_MODEL_STAGE_ROPE_FREQ_BASE)
}

/// Per-layer RoPE frequency base. For ISWA models (Gemma4), SWA layers have a smaller
/// head_dim than the manifest's full-attention head_dim; they use the SWA rope theta
/// from `rope_theta_swa`. Reference: llama.cpp llama-model.cpp `get_rope_freq_base`.
#[cfg(target_os = "macos")]
pub(super) fn effective_rope_theta(artifacts: &NativeModelArtifacts, head_dim: usize) -> f32 {
    let manifest = artifacts.manifest();
    let manifest_head_dim = manifest.attention_head_dim as usize;
    if manifest_head_dim > 0 && head_dim < manifest_head_dim {
        manifest
            .rope_theta_swa
            .map(|t| t as f32)
            .unwrap_or(PHASE1_MODEL_STAGE_ROPE_FREQ_BASE)
    } else {
        native_model_rope_theta(artifacts)
    }
}

/// Per-layer rotary dimension. For ISWA models (Gemma4), SWA layers use a smaller
/// head_dim and rotate all of it (partial_rotary_factor = 1.0). The global rotary_dim
/// (derived from partial_rotary_factor applied to the full-attention head_dim) would be
/// wrong for SWA layers. Reference: mlx-lm gemma4_text.py sliding_attention config.
#[cfg(target_os = "macos")]
pub(super) fn effective_rotary_dim(artifacts: &NativeModelArtifacts, head_dim: usize) -> usize {
    let global = artifacts.rotary_dim();
    let manifest_head_dim = artifacts.manifest().attention_head_dim as usize;
    if manifest_head_dim > 0 && head_dim < manifest_head_dim {
        head_dim
    } else {
        global
    }
}

#[cfg(target_os = "macos")]
pub(super) fn native_model_reference_attention_config(
    artifacts: &NativeModelArtifacts,
    head_dim: usize,
) -> ReferenceAttentionConfig {
    let softmax_scale = artifacts
        .manifest()
        .query_pre_attn_scalar
        .map(|scalar| 1.0 / (scalar as f32).sqrt())
        .or_else(|| {
            ReferenceAttentionConfig::from_head_dim(head_dim).map(|config| config.softmax_scale)
        })
        .unwrap_or(1.0);
    let softcap = artifacts
        .manifest()
        .attention_logit_softcap
        .map(|softcap| softcap as f32);
    ReferenceAttentionConfig {
        softmax_scale,
        softcap,
    }
}
