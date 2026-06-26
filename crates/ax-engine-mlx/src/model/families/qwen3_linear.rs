use mlx_sys::{MlxArray, MlxVectorArray, add, rms_norm, slice};
use std::time::Instant;

use super::super::ModelConfig;
use super::super::profile::{
    DecodeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    prefill_profile_enabled,
};
use super::super::shared::{
    ffn_swiglu, flatten_compiled_moe_inputs, linear_attention_forward, moe_experts_forward,
    moe_experts_forward_with_cloned_weights, moe_experts_forward_with_shared, moe_router_qwen3,
    rms_norm_opt, shared_expert_forward,
};
use crate::fastpath;
use crate::kv_cache::MlxKVCache;
use crate::per_layer_compile::apply_layer_moe_decode;
use crate::weights::LayerWeights;

/// Full layer forward for Qwen3.5/Qwen3Next linear-attention layers.
///
/// These layers use the gated-delta recurrent kernel instead of SDPA. Dense FFN
/// covers the common case (e.g. Qwen3.5 9B). MoE-only variants such as
/// Qwen3.6 35B A3B pair linear attention with sparse FFN (router + experts +
/// optional shared expert), so the FFN dispatch mirrors `standard::layer_forward`.
///
/// `last_position_only`: when `true` and `seq > 1`, slice `hidden` to the last
/// position after the attention-residual add, so the FFN / MoE steps run on
/// `[1, 1, hidden]` instead of `[1, seq, hidden]`. The linear-attention state
/// (conv1d + recurrent) is already written to `cache` inside
/// `linear_attention_forward` before this slice, so the optimization is safe.
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    last_position_only: bool,
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
    // Last-position-only: after the attention-residual add, the linear-attention
    // state has been committed to `cache`. The FFN is position-wise, so slicing
    // to the last position is safe and avoids redundant compute on preceding
    // positions whose output will be discarded by the post-loop slice.
    let last_only_active = last_position_only && seq > 1;
    let hidden = if last_only_active {
        let last = (seq - 1) as i32;
        let hs = cfg.hidden_size as i32;
        slice(&hidden, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None)
    } else {
        hidden
    };
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
        let router_started = profile_forward_layer.then(Instant::now);
        let (top_k_indices, top_k_weights) = moe_router_qwen3(cfg, w, &normed2);
        if let Some(started) = router_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::MoeRouter,
                started,
                &[&top_k_indices, &top_k_weights],
            );
        }
        let shared_started = profile_forward_layer.then(Instant::now);
        let shared_out = if w.shared_gate_proj.is_some() {
            Some(shared_expert_forward(cfg, w, &normed2))
        } else {
            None
        };
        if let Some(started) = shared_started {
            if let Some(shared) = &shared_out {
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::MoeSharedExpert,
                    started,
                    &[shared],
                );
            } else {
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::MoeSharedExpert,
                    started,
                    &[],
                );
            }
        }
        // Try compiled MoE decode closure (gated by AX_MLX_MOE_LAYER_COMPILE).
        // The entire MoE expert forward is compiled into a single graph,
        // collapsing ~10 dispatches per layer into one. Every MLX array the
        // graph depends on (expert weights + optional shared-expert output) is
        // threaded through as an explicit input: MLX-C 0.6.0 forbids compiling
        // a function with uncaptured inputs, so capturing the weight tensors in
        // the closure aborts on the first decode. Only `cfg` (no MoE-relevant
        // MlxArray fields) and the Copy index schema are captured.
        // Guard the flag first to avoid building the input vector on every
        // decode step when the compile path is disabled.
        let compiled_result = if seq == 1 && fastpath::moe_layer_compile_enabled() {
            let cfg_clone = cfg.clone();
            let (inputs, schema) = flatten_compiled_moe_inputs(
                &normed2,
                &top_k_indices,
                &top_k_weights,
                w.gate_up_exps_packed.as_ref(),
                w.gate_exps.as_ref(),
                w.up_exps.as_ref(),
                w.down_exps.as_ref(),
                shared_out.as_ref(),
            );
            let input_refs: Vec<&MlxArray> = inputs.iter().collect();
            apply_layer_moe_decode(layer_idx, &input_refs, move |inputs: &MlxVectorArray| {
                let (x, indices, weights, gate_up, gate, up, down, shared) = schema.rebuild(inputs);
                vec![moe_experts_forward_with_cloned_weights(
                    &cfg_clone, &x, &indices, &weights, gate_up, gate, up, down, shared, None,
                )]
            })
        } else {
            None
        };

        if let Some(result) = compiled_result.and_then(|result| result.into_iter().next()) {
            result
        } else if let Some(shared) = &shared_out {
            moe_experts_forward_with_shared(
                cfg,
                w,
                &normed2,
                &top_k_indices,
                &top_k_weights,
                shared,
            )
        } else {
            moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights)
        }
    } else {
        ffn_swiglu(cfg, w, &normed2, None, layer_idx)
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
