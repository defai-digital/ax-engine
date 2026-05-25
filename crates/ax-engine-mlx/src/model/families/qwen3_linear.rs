use mlx_sys::{MlxArray, add, rms_norm};
use std::time::Instant;

use super::super::ModelConfig;
use super::super::profile::{
    DecodeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    prefill_profile_enabled,
};
use super::super::shared::{
    ffn_swiglu, linear_attention_forward, moe_experts_forward, moe_router_qwen3, rms_norm_opt,
    shared_expert_forward,
};
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full layer forward for Qwen3.5/Qwen3Next linear-attention layers.
///
/// These layers use the gated-delta recurrent kernel instead of SDPA. Dense FFN
/// covers the common case (e.g. Qwen3.5 9B). MoE-only variants such as
/// Qwen3.6 35B A3B pair linear attention with sparse FFN (router + experts +
/// optional shared expert), so the FFN dispatch mirrors `standard::layer_forward`.
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
) -> MlxArray {
    let seq = hidden.shape()[1] as usize;
    let profile_decode_layer = seq == 1 && decode_profile_enabled();
    let profile_prefill_layer = seq > 1 && prefill_profile_enabled();
    let profile_forward_layer = profile_decode_layer || profile_prefill_layer;

    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    // linear_attention_forward includes its own per-layer profiling.
    let attn_proj = linear_attention_forward(cfg, w, &normed, cache, layer_idx);

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
    let out = if w.router_proj.is_some() {
        let (top_k_indices, top_k_weights) = moe_router_qwen3(cfg, w, &normed2);
        let mut moe_out = moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights);
        if w.shared_gate_proj.is_some() {
            moe_out = add(&moe_out, &shared_expert_forward(cfg, w, &normed2), None);
        }
        moe_out
    } else {
        ffn_swiglu(cfg, w, &normed2, None)
    };
    let ffn_out = rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps);
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
