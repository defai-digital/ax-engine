use mlx_sys::{
    MlxArray, add, astype, expand_dims_axes, multiply, rms_norm, rope, swiglu_oai, transpose,
};

use super::super::ModelConfig;
use super::super::config::layer_params;
use super::super::shared::{
    attention_mask_array, attention_with_sinks, moe_router_gpt_oss, prepare_value_bhsd_from_proj,
    qk_norm_bhsd_from_proj, qw, qw_gather, squeeze_switch_singleton,
};
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full layer forward for GPT-OSS (20B / 120B).
///
/// Architecture (aligned with mlx-lm `gpt_oss.py` and llama.cpp `openai-moe.cpp`):
/// - MoE decoder with per-head attention sinks
/// - Router: top-k on logits then softmax on selected experts only
/// - Experts: **split** gate/up/down via `gather_qmm` (mlx-lm SwitchGLU layout)
/// - Activation: OpenAI SwiGLU (`swiglu_oai`: clip + alpha=1.702 + (up+1))
/// - Alternating full / sliding-128 attention, GQA, YaRN RoPE
///
/// Weight layouts (both keep experts packed — no load-time BF16 expand):
/// 1. openai/gpt-oss: fused `*_blocks` sanitized → split gate/up/down at load
/// 2. mlx-community MXFP4-Q4: already split `experts.{gate,up,down}_proj.weight`
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, layer_rope_freqs, sliding_window, _, _) =
        layer_params(cfg, layer_idx);

    let seq = hidden.shape()[1] as usize;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;

    // 1. Attention norm + QKV projection (with bias).
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    let q_raw = qw(
        &normed,
        w.q_proj.as_ref().expect("gpt_oss layer must have q_proj"),
    );
    let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
    let (rope_base, rope_freqs_ref) = rope_freqs
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));

    let q = qk_norm_bhsd_from_proj(
        &q_raw,
        w.q_norm.as_ref(),
        n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    // YaRN mscale (mlx-lm YarnRoPE): scale rotated dims before rope.
    let q = if (cfg.rope_mscale - 1.0).abs() > 1e-6 {
        multiply(
            &q,
            &mlx_sys::ops::cached_scalar(cfg.rope_mscale, q.dtype()),
            None,
        )
    } else {
        q
    };
    let q_rope = rope(
        &q,
        rope_dims as i32,
        false,
        rope_base,
        1.0,
        token_offset as i32,
        rope_freqs_ref,
        None,
    );

    let k_raw = qw(
        &normed,
        w.k_proj.as_ref().expect("gpt_oss layer must have k_proj"),
    );
    let k = qk_norm_bhsd_from_proj(
        &k_raw,
        w.k_norm.as_ref(),
        n_kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let k = if (cfg.rope_mscale - 1.0).abs() > 1e-6 {
        multiply(
            &k,
            &mlx_sys::ops::cached_scalar(cfg.rope_mscale, k.dtype()),
            None,
        )
    } else {
        k
    };
    let k_rope = rope(
        &k,
        rope_dims as i32,
        false,
        rope_base,
        1.0,
        token_offset as i32,
        rope_freqs_ref,
        None,
    );

    let v_raw = qw(
        &normed,
        w.v_proj.as_ref().expect("gpt_oss layer must have v_proj"),
    );
    let v =
        prepare_value_bhsd_from_proj(&v_raw, false, n_kv_heads, head_dim, seq, cfg.rms_norm_eps);

    // 2. KV cache update.
    let (cached_k, cached_v) = if seq == 1 {
        cache.append_with_retained_window(layer_idx, k_rope, v, sliding_window)
    } else {
        cache.append(layer_idx, k_rope, v)
    };

    // 3. Attention with sinks (mlx-lm: scaled_dot_product_attention(..., sinks=)).
    let query_scale = 1.0 / (head_dim as f32).sqrt();
    let mask = attention_mask_array(seq, cached_k.shape()[2] as usize, sliding_window);

    let sinks = w
        .attn_sink
        .as_ref()
        .expect("gpt_oss layer must have attn_sink");
    let attn_out = attention_with_sinks(
        &q_rope,
        &cached_k,
        &cached_v,
        sinks,
        query_scale,
        seq,
        &mask,
    );

    let batch = attn_out.shape()[0];
    let attn_flat = transpose(&attn_out, &[0, 2, 1, 3], None);
    let attn_flat = mlx_sys::reshape(
        &attn_flat,
        &[batch, seq as i32, (n_heads * head_dim) as i32],
        None,
    );

    let o_proj = qw(
        &attn_flat,
        w.o_proj.as_ref().expect("gpt_oss layer must have o_proj"),
    );
    let hidden = add(hidden, &o_proj, None);

    // 5. FFN: RMSNorm → MoE router → expert forward → residual.
    let ffn_normed = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let (top_k_indices, top_k_weights) = moe_router_gpt_oss(cfg, w, &ffn_normed);
    let ffn_out = gpt_oss_moe_experts_forward(cfg, w, &ffn_normed, &top_k_indices, &top_k_weights);
    add(&hidden, &ffn_out, None)
}

/// GPT-OSS MoE experts — mlx-lm SwitchGLU layout (split gate/up/down + OAI SwiGLU).
fn gpt_oss_moe_experts_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
) -> MlxArray {
    let gate_exps = w
        .gate_exps
        .as_ref()
        .expect("gpt_oss MoE requires gate_exps (split experts after sanitize)");
    let up_exps = w
        .up_exps
        .as_ref()
        .expect("gpt_oss MoE requires up_exps (split experts after sanitize)");
    let down_exps = w
        .down_exps
        .as_ref()
        .expect("gpt_oss MoE requires down_exps");

    // mlx-lm SwitchGLU: x = expand_dims(x, (-2, -3))
    let x_exp = expand_dims_axes(x, &[-2, -3], None);
    let batch = x.shape()[0];
    let seq_len = x.shape()[1];
    let top_k = top_k_indices.shape().last().copied().unwrap_or(0);

    // x_up = up_proj(x, indices); x_gate = gate_proj(x, indices)
    let up = squeeze_switch_singleton(&qw_gather(&x_exp, up_exps, top_k_indices, false));
    let gate = squeeze_switch_singleton(&qw_gather(&x_exp, gate_exps, top_k_indices, false));

    // activation(x_up, x_gate) → swiglu_oai(up, gate)  [mlx-lm gpt_oss.SwiGLU]
    let h = swiglu_oai(&up, &gate, None);

    let down_out = squeeze_switch_singleton(&qw_gather(&h, down_exps, top_k_indices, false));
    let down_out = mlx_sys::reshape(
        &down_out,
        &[batch, seq_len, top_k, cfg.hidden_size as i32],
        None,
    );

    // Weighted sum: x * expand_dims(expert_weights, -1); sum over top-k.
    let weights = mlx_sys::reshape(
        &astype(top_k_weights, x.dtype(), None),
        &[batch, seq_len, top_k, 1],
        None,
    );
    let weighted = mlx_sys::multiply(&down_out, &weights, None);
    let summed = mlx_sys::sum_axis(&weighted, 2, false, None);
    mlx_sys::reshape(&summed, &[batch, seq_len, cfg.hidden_size as i32], None)
}
