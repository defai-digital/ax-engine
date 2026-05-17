use mlx_sys::{MlxArray, add, rms_norm};

use super::super::ModelConfig;
use super::super::shared::{
    glm_mla_attention_forward, moe_experts_forward, moe_router_deepseek_v3, qw,
    shared_expert_forward,
};
use super::super::turboquant_context::TurboQuantModelDecodeContext;
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    _turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    // 1. Attention norm + MLA forward (same as GLM4MoELite).
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let attn_out = glm_mla_attention_forward(cfg, w, &normed, cache, layer_idx, token_offset);
    let post_attn = add(hidden, &attn_out, None);

    // 2. FFN norm.
    let normed2 = rms_norm(&post_attn, Some(&w.ffn_norm), cfg.rms_norm_eps, None);

    // 3. FFN: MoE for eligible layers, dense SwiGLU for the rest.
    let ffn_out = if cfg.is_deepseek_moe_layer(layer_idx) {
        deepseek_moe_forward(cfg, w, &normed2)
    } else {
        deepseek_dense_ffn(w, &normed2)
    };

    add(&post_attn, &ffn_out, None)
}

/// Dense SwiGLU FFN for non-MoE layers.
fn deepseek_dense_ffn(w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let gate = qw(x, w.gate_proj.as_ref().expect("dense gate_proj"));
    let up = qw(x, w.up_proj.as_ref().expect("dense up_proj"));
    let act = mlx_sys::multiply(&mlx_sys::ops::silu(&gate, None), &up, None);
    qw(&act, w.down_proj.as_ref().expect("dense down_proj"))
}

/// DeepSeek V3 MoE: sigmoid routing with optional correction bias + shared experts.
fn deepseek_moe_forward(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let (indices, scores) = moe_router_deepseek_v3(cfg, w, x);
    let expert_out = moe_experts_forward(cfg, w, x, &indices, &scores);

    if cfg.moe_shared_expert_count > 0 {
        let shared_out = shared_expert_forward(cfg, w, x);
        add(&expert_out, &shared_out, None)
    } else {
        expert_out
    }
}
