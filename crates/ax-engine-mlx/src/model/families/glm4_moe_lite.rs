use mlx_sys::{MlxArray, add, rms_norm};
use std::time::Instant;

use super::super::ModelConfig;
use super::super::profile::{
    DecodeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    prefill_profile_enabled,
};
use super::super::shared::{
    ffn_swiglu, glm_mla_attention_forward, moe_experts_forward, moe_router_glm, rms_norm_opt,
    shared_expert_forward,
};
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full layer forward for GLM4MoELite layers (MLA attention + GLM MoE or dense FFN).
///
/// glm_mla_attention_forward includes its own per-stage profiling for the
/// attention portion; this function adds the generic FFN and residual stages.
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    let seq = hidden.shape()[1] as usize;
    let profile_decode_layer = seq == 1 && decode_profile_enabled();
    let profile_prefill_layer = seq > 1 && prefill_profile_enabled();
    let profile_forward_layer = profile_decode_layer || profile_prefill_layer;

    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    // glm_mla_attention_forward includes its own per-stage profiling.
    let attn_proj = glm_mla_attention_forward(cfg, w, &normed, cache, layer_idx, token_offset);
    let attn_proj = if let Some(post_norm) = &w.attn_post_norm {
        rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
    } else {
        attn_proj
    };

    let residual_norm_started = profile_forward_layer.then(Instant::now);
    let hidden = add(hidden, &attn_proj, None);
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    if let Some(started) = residual_norm_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttnResidualNorm,
            started,
            &[&normed2],
        );
    }

    let ffn_started = profile_forward_layer.then(Instant::now);
    let ffn_out = if w.router_proj.is_some() {
        // GLM MoE layer.
        let (top_k_indices, top_k_weights) = moe_router_glm(cfg, w, &normed2);
        let mut out = moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights);
        if w.shared_gate_proj.is_some() {
            out = add(&out, &shared_expert_forward(cfg, w, &normed2), None);
        }
        rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
    } else {
        // Dense FFN (first_dense_layer_count layers).
        ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm.as_ref())
    };
    if let Some(started) = ffn_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttnFfn,
            started,
            &[&ffn_out],
        );
    }

    let residual_gate_started = profile_forward_layer.then(Instant::now);
    let out = add(&hidden, &ffn_out, None);
    if let Some(started) = residual_gate_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttnResidualGate,
            started,
            &[&out],
        );
    }
    out
}
