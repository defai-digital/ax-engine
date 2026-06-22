use mlx_sys::{
    MlxArray, add, astype, expand_dims_axes, gather_mm, rms_norm, rope, silu_mul, slice, transpose,
};

use super::super::ModelConfig;
use super::super::config::layer_params;
use super::super::shared::{
    attention_mask_array, attention_with_sinks, moe_router_gpt_oss, prepare_value_bhsd_from_proj,
    qk_norm_bhsd_from_proj, qw,
};
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full layer forward for GPT-OSS (20B / 120B).
///
/// Architecture: MoE decoder with per-head attention sinks, softmax-after-topk
/// routing, alternating full/sliding-128 attention, GQA (64q/8k heads),
/// SwiGLU activation, and MXFP4 MoE expert weights.
///
/// Attention uses a custom sink-aware path (not SDPA) because the learned
/// per-head sink bias must be injected before softmax.
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

    // Q projection → QK norm → RoPE.
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

    // K projection → QK norm → RoPE.
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

    // V projection → prepare BHSD view.
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

    // 3. Attention with sinks.
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

    // Flatten BHSd → BSHD.
    let batch = attn_out.shape()[0];
    let attn_flat = transpose(&attn_out, &[0, 2, 1, 3], None);
    let attn_flat = mlx_sys::reshape(
        &attn_flat,
        &[batch, seq as i32, (n_heads * head_dim) as i32],
        None,
    );

    // 4. Output projection → residual.
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

/// GPT-OSS MoE expert forward using dequantized BF16 MXFP4 weights.
///
/// Uses `gather_mm` (non-quantized matmul) with the pre-dequantized expert
/// weights stored in `w.mxfp4_gate_up_exps` and `w.mxfp4_down_exps`.
fn gpt_oss_moe_experts_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
) -> MlxArray {
    let gate_up_exps = w
        .mxfp4_gate_up_exps
        .as_ref()
        .expect("gpt_oss MoE layer must have mxfp4_gate_up_exps");
    let down_exps = w
        .mxfp4_down_exps
        .as_ref()
        .expect("gpt_oss MoE layer must have mxfp4_down_exps");

    // [batch, seq, hidden] → [batch, seq, 1, 1, hidden]
    let x_exp = expand_dims_axes(x, &[-2, -3], None);
    let batch = x_exp.shape()[0];
    let seq_len = x_exp.shape()[1];
    let top_k = top_k_indices.shape().last().copied().unwrap_or(0);

    // Gather expert indices for gate_up: [batch, seq, top_k, 1, 1]
    let gate_up_indices = mlx_sys::reshape(top_k_indices, &[batch, seq_len, top_k, 1, 1], None);

    // gate_up = x @ gate_up_exps[indices].T → [batch, seq, top_k, 1, 2*intermediate]
    let gate_up = gather_mm(&x_exp, gate_up_exps, &gate_up_indices, false, None);

    // Split gate/up and apply SwiGLU activation.
    let intermediate = cfg.moe_expert_intermediate_size as i32;
    let gate = slice(
        &gate_up,
        &[0, 0, 0, 0, 0],
        &[
            gate_up.shape()[0],
            gate_up.shape()[1],
            gate_up.shape()[2],
            gate_up.shape()[3],
            intermediate,
        ],
        &[1, 1, 1, 1, 1],
        None,
    );
    let up = slice(
        &gate_up,
        &[0, 0, 0, 0, intermediate],
        &[
            gate_up.shape()[0],
            gate_up.shape()[1],
            gate_up.shape()[2],
            gate_up.shape()[3],
            intermediate * 2,
        ],
        &[1, 1, 1, 1, 1],
        None,
    );
    let h = silu_mul(&gate, &up, None);

    // Down projection.
    let down_indices = gate_up_indices;
    let down_out = gather_mm(&h, down_exps, &down_indices, false, None);

    // Squeeze singleton dims and weight by router scores.
    // down_out: [batch, seq, top_k, 1, hidden] → [batch, seq, top_k, hidden]
    let down_out = mlx_sys::reshape(
        &down_out,
        &[batch, seq_len, top_k, cfg.hidden_size as i32],
        None,
    );

    // Weighted sum over top-k experts.
    // top_k_weights: [batch, seq, top_k] → [batch, seq, top_k, 1]
    let weights = mlx_sys::reshape(
        &astype(top_k_weights, x.dtype(), None),
        &[batch, seq_len, top_k, 1],
        None,
    );
    let weighted = mlx_sys::multiply(&down_out, &weights, None);
    let summed = mlx_sys::sum_axis(&weighted, 2, false, None);

    // Squeeze the top_k dimension → [batch, seq, hidden]
    mlx_sys::reshape(&summed, &[batch, seq_len, cfg.hidden_size as i32], None)
}
