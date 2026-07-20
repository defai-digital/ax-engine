//! Nemotron-H hybrid (`nemotron_h`): residual mixers per layer.
//!
//! Pattern-driven layer kinds (`mamba` / `attention` / `moe` / `mlp`) from
//! `hybrid_override_pattern`. Each layer is a single residual block:
//! `x + mixer(rms_norm(x))` — not an attn+FFN sandwich.
//!
//! Mamba-2 follows mlx-lm `nemotron_h.NemotronHMamba2Mixer` + `ssm.ssm_update`.
//! MoE uses DeepSeek-style sigmoid routing with ReLU² experts (fc1/fc2 only).

use mlx_sys::ops::silu;
use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, conv1d, exp, expand_dims, log1p, maximum,
    multiply, negative, pad, reshape, rms_norm, slice, slice_last_dim, sum_axis, transpose, zeros,
};

use super::super::ModelConfig;
use super::super::shared::{
    attention_mask_array, flatten_attention_output_bhsd, full_precision_attention,
    moe_experts_forward_with_shared, moe_router_deepseek_v3, qw,
};
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full residual mixer forward for one Nemotron-H layer.
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    let kind = layer_kind(layer_idx, w);
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let delta = match kind {
        NemotronLayerKind::Mamba => mamba2_forward(cfg, w, &normed, cache, layer_idx),
        NemotronLayerKind::Attention => {
            attention_forward(cfg, w, &normed, cache, layer_idx, token_offset)
        }
        NemotronLayerKind::Moe => moe_forward(cfg, w, &normed),
        NemotronLayerKind::Mlp => relu2_mlp_forward(w, &normed),
    };
    add(hidden, &delta, None)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NemotronLayerKind {
    Mamba,
    Attention,
    Moe,
    Mlp,
}

fn layer_kind(layer_idx: usize, w: &LayerWeights) -> NemotronLayerKind {
    if w.linear_attn.is_some() {
        return NemotronLayerKind::Mamba;
    }
    if w.q_proj.is_some() && w.o_proj.is_some() {
        return NemotronLayerKind::Attention;
    }
    if w.router_proj.is_some() || w.up_exps.is_some() {
        return NemotronLayerKind::Moe;
    }
    if w.up_proj.is_some() && w.down_proj.is_some() {
        return NemotronLayerKind::Mlp;
    }
    panic!("nemotron_h layer {layer_idx} has no recognisable mixer weights");
}

// ---------------------------------------------------------------------------
// Attention (*): no-RoPE GQA
// ---------------------------------------------------------------------------

fn attention_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
) -> MlxArray {
    let q_proj = w
        .q_proj
        .as_ref()
        .expect("nemotron attention layer must have q_proj");
    let k_proj = w
        .k_proj
        .as_ref()
        .expect("nemotron attention layer must have k_proj");
    let v_proj = w
        .v_proj
        .as_ref()
        .expect("nemotron attention layer must have v_proj");
    let o_proj = w
        .o_proj
        .as_ref()
        .expect("nemotron attention layer must have o_proj");

    let batch = x.shape()[0] as i32;
    let seq = x.shape()[1] as i32;
    let n_heads = cfg.n_heads as i32;
    let n_kv = cfg.n_kv_heads as i32;
    let head_dim = cfg.head_dim as i32;

    let q = qw(x, q_proj);
    let k = qw(x, k_proj);
    let v = qw(x, v_proj);

    // [B, L, H, D] → [B, H, L, D]
    let q = transpose(
        &reshape(&q, &[batch, seq, n_heads, head_dim], None),
        &[0, 2, 1, 3],
        None,
    );
    let k = transpose(
        &reshape(&k, &[batch, seq, n_kv, head_dim], None),
        &[0, 2, 1, 3],
        None,
    );
    let v = transpose(
        &reshape(&v, &[batch, seq, n_kv, head_dim], None),
        &[0, 2, 1, 3],
        None,
    );

    // No RoPE — matches mlx-lm NemotronHAttention.
    let (cached_k, cached_v) = cache.append(layer_idx, k, v);
    let key_len = token_offset + seq as usize;
    let mask = attention_mask_array(seq as usize, key_len, None);
    let scale = cfg.query_scale;
    let attn = full_precision_attention(&q, &cached_k, &cached_v, scale, seq as usize, &mask);
    let flat = flatten_attention_output_bhsd(&attn, seq as usize, cfg.n_heads, cfg.head_dim);
    qw(&flat, o_proj)
}

// ---------------------------------------------------------------------------
// MoE (E): sigmoid router + ReLU² experts + shared expert
// ---------------------------------------------------------------------------

fn moe_forward(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let (indices, scores) = moe_router_deepseek_v3(cfg, w, x);
    let shared = if w.shared_up_proj.is_some() && w.shared_down_proj.is_some() {
        Some(relu2_shared_expert(w, x))
    } else {
        None
    };
    if let Some(shared) = shared.as_ref() {
        moe_experts_forward_with_shared(cfg, w, x, &indices, &scores, shared)
    } else {
        // Reuse with_shared path with a zero tensor when no shared expert.
        let shape = x.shape();
        let zero = zeros(&shape, x.dtype(), None);
        moe_experts_forward_with_shared(cfg, w, x, &indices, &scores, &zero)
    }
}

fn relu2_shared_expert(w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let up = qw(
        x,
        w.shared_up_proj
            .as_ref()
            .expect("shared expert up projection"),
    );
    let act = relu2(&up);
    qw(
        &act,
        w.shared_down_proj
            .as_ref()
            .expect("shared expert down projection"),
    )
}

fn relu2_mlp_forward(w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let up = qw(x, w.up_proj.as_ref().expect("mlp up projection"));
    let act = relu2(&up);
    qw(&act, w.down_proj.as_ref().expect("mlp down projection"))
}

/// `nn.ReLU2(x) = relu(x)^2` (mlx-lm / NVIDIA Nemotron).
fn relu2(x: &MlxArray) -> MlxArray {
    let zero = zeros(&[], x.dtype(), None);
    let relu = maximum(x, &zero, None);
    multiply(&relu, &relu, None)
}

// ---------------------------------------------------------------------------
// Mamba-2 (M)
// ---------------------------------------------------------------------------

fn mamba2_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
) -> MlxArray {
    let linear_cfg = cfg
        .linear_attention
        .as_ref()
        .expect("nemotron mamba layer requires linear_attention config");
    let la = w
        .linear_attn
        .as_ref()
        .expect("nemotron mamba layer requires linear_attn weights");
    let in_proj = la
        .in_proj_qkvz
        .as_ref()
        .expect("nemotron mamba requires packed in_proj (LinearAttentionInProjQkvz)");

    let batch = x.shape()[0] as usize;
    let seq = x.shape()[1] as usize;
    let num_heads = linear_cfg.num_value_heads;
    let head_dim = linear_cfg.value_head_dim;
    let n_groups = linear_cfg.num_key_heads;
    let ssm_state = linear_cfg.key_head_dim;
    let intermediate = num_heads * head_dim;
    let conv_dim = intermediate + 2 * n_groups * ssm_state;

    let projected = qw(x, in_proj);
    let gate = slice_last_dim(&projected, 0, intermediate as i32, None);
    let conv_input = slice_last_dim(
        &projected,
        intermediate as i32,
        (intermediate + conv_dim) as i32,
        None,
    );
    let dt = slice_last_dim(
        &projected,
        (intermediate + conv_dim) as i32,
        (intermediate + conv_dim + num_heads) as i32,
        None,
    );

    let (conv_state, recurrent_state) = cache.linear_state(layer_idx);
    let (conv_out, new_conv_state) = mamba_conv1d(
        &conv_input,
        la,
        conv_state,
        conv_dim,
        linear_cfg.conv_kernel_dim,
    );

    let hidden_ssm = slice_last_dim(&conv_out, 0, intermediate as i32, None);
    let b_in = slice_last_dim(
        &conv_out,
        intermediate as i32,
        (intermediate + n_groups * ssm_state) as i32,
        None,
    );
    let c_in = slice_last_dim(
        &conv_out,
        (intermediate + n_groups * ssm_state) as i32,
        (intermediate + 2 * n_groups * ssm_state) as i32,
        None,
    );

    let x_heads = reshape(
        &hidden_ssm,
        &[batch as i32, seq as i32, num_heads as i32, head_dim as i32],
        None,
    );
    let b = reshape(
        &b_in,
        &[batch as i32, seq as i32, n_groups as i32, ssm_state as i32],
        None,
    );
    let c = reshape(
        &c_in,
        &[batch as i32, seq as i32, n_groups as i32, ssm_state as i32],
        None,
    );

    let d = la.d.clone().unwrap_or_else(|| {
        // ones([H]) via exp(zeros)
        exp(&zeros(&[num_heads as i32], MlxDtype::Float32, None), None)
    });

    let state = recurrent_state.cloned().unwrap_or_else(|| {
        zeros(
            &[
                batch as i32,
                num_heads as i32,
                head_dim as i32,
                ssm_state as i32,
            ],
            MlxDtype::Float32,
            None,
        )
    });

    let (y, new_state) = ssm_update(
        &x_heads,
        &la.a_log,
        &b,
        &c,
        &d,
        &dt,
        &la.dt_bias,
        &state,
        n_groups,
    );
    cache.set_linear_state(layer_idx, new_conv_state, new_state);

    let y_flat = reshape(&y, &[batch as i32, seq as i32, intermediate as i32], None);
    let y_gated = mamba_rms_norm_gated(&y_flat, &gate, &la.norm, cfg.rms_norm_eps, n_groups);
    qw(&y_gated, &la.out_proj)
}

fn mamba_conv1d(
    conv_input: &MlxArray,
    la: &crate::weights::LinearAttentionWeights,
    conv_state: Option<&MlxArray>,
    conv_dim: usize,
    kernel: usize,
) -> (MlxArray, MlxArray) {
    let batch = conv_input.shape()[0] as i32;
    let conv_dim_i = conv_dim as i32;
    let k = kernel as i32;
    let n_keep = k - 1;

    let padded = if let Some(state) = conv_state {
        concatenate(&[state, conv_input], 1, None)
    } else {
        let zero = zeros(&[], conv_input.dtype(), None);
        pad(conv_input, &[1], &[n_keep], &[0], &zero, None)
    };
    let mut out = conv1d(
        &padded,
        &la.conv1d_dense,
        /*stride*/ 1,
        /*padding*/ 0,
        /*dilation*/ 1,
        /*groups*/ conv_dim as i32,
        None,
    );
    if let Some(bias) = la.conv1d_bias.as_ref() {
        let b = reshape(bias, &[1, 1, conv_dim_i], None);
        out = add(&out, &b, None);
    }
    let out = silu(&out, None);
    let t = padded.shape()[1] as i32;
    let new_state = slice(
        &padded,
        &[0, t - n_keep, 0],
        &[batch, t, conv_dim_i],
        &[1, 1, 1],
        None,
    );
    (out, new_state)
}

/// Stable softplus: `max(x, 0) + log1p(exp(-abs(x)))`.
fn softplus(x: &MlxArray) -> MlxArray {
    let zero = zeros(&[], x.dtype(), None);
    let abs_x = maximum(x, &negative(x, None), None);
    let neg_abs = negative(&abs_x, None);
    let pos = maximum(x, &zero, None);
    add(&pos, &log1p(&exp(&neg_abs, None), None), None)
}

/// Sequential Mamba-2 SSM update (correct for prefill and decode).
///
/// - x: [B, L, H, Dh]
/// - A_log: [H]
/// - B, C: [B, L, G, Ds]
/// - D: [H]
/// - dt: [B, L, H]
/// - dt_bias: [H]
/// - state: [B, H, Dh, Ds]
fn ssm_update(
    x: &MlxArray,
    a_log: &MlxArray,
    b: &MlxArray,
    c: &MlxArray,
    d: &MlxArray,
    dt: &MlxArray,
    dt_bias: &MlxArray,
    state: &MlxArray,
    n_groups: usize,
) -> (MlxArray, MlxArray) {
    let batch = x.shape()[0] as usize;
    let seq = x.shape()[1] as usize;
    let num_heads = x.shape()[2] as usize;
    let head_dim = x.shape()[3] as usize;
    let ssm_state = b.shape()[3] as usize;
    let heads_per_group = num_heads / n_groups;

    let dt_f = astype(dt, MlxDtype::Float32, None);
    let dt_bias_f = astype(dt_bias, MlxDtype::Float32, None);
    let dt_b = reshape(&dt_bias_f, &[1, 1, num_heads as i32], None);
    let dt_act = softplus(&add(&dt_f, &dt_b, None));

    let mut state = state.clone();
    let mut ys = Vec::with_capacity(seq);
    let a_neg = negative(&exp(a_log, None), None); // A = -exp(A_log)

    for t in 0..seq {
        let x_t = slice(
            x,
            &[0, t as i32, 0, 0],
            &[
                batch as i32,
                t as i32 + 1,
                num_heads as i32,
                head_dim as i32,
            ],
            &[1, 1, 1, 1],
            None,
        );
        let x_t = reshape(
            &x_t,
            &[batch as i32, num_heads as i32, head_dim as i32],
            None,
        );
        let dt_t = slice(
            &dt_act,
            &[0, t as i32, 0],
            &[batch as i32, t as i32 + 1, num_heads as i32],
            &[1, 1, 1],
            None,
        );
        let dt_t = reshape(&dt_t, &[batch as i32, num_heads as i32, 1, 1], None);
        let b_t = slice(
            b,
            &[0, t as i32, 0, 0],
            &[
                batch as i32,
                t as i32 + 1,
                n_groups as i32,
                ssm_state as i32,
            ],
            &[1, 1, 1, 1],
            None,
        );
        let c_t = slice(
            c,
            &[0, t as i32, 0, 0],
            &[
                batch as i32,
                t as i32 + 1,
                n_groups as i32,
                ssm_state as i32,
            ],
            &[1, 1, 1, 1],
            None,
        );
        let b_t = expand_b_c_to_heads(&b_t, batch, n_groups, heads_per_group, ssm_state);
        let c_t = expand_b_c_to_heads(&c_t, batch, n_groups, heads_per_group, ssm_state);

        let a = reshape(&a_neg, &[1, num_heads as i32, 1, 1], None);
        let d_a = exp(&multiply(&a, &dt_t, None), None);
        let x_dt = multiply(
            &reshape(
                &x_t,
                &[batch as i32, num_heads as i32, head_dim as i32, 1],
                None,
            ),
            &dt_t,
            None,
        );
        let b_e = reshape(
            &b_t,
            &[batch as i32, num_heads as i32, 1, ssm_state as i32],
            None,
        );
        let delta_state = multiply(&x_dt, &b_e, None);
        state = add(&multiply(&d_a, &state, None), &delta_state, None);

        let c_e = reshape(
            &c_t,
            &[batch as i32, num_heads as i32, 1, ssm_state as i32],
            None,
        );
        let y = sum_axis(&multiply(&state, &c_e, None), 3, false, None);
        let d_e = reshape(d, &[1, num_heads as i32, 1], None);
        let y = add(&y, &multiply(&d_e, &x_t, None), None);
        let y = reshape(
            &y,
            &[batch as i32, 1, num_heads as i32, head_dim as i32],
            None,
        );
        ys.push(astype(&y, x.dtype(), None));
    }

    let y = if ys.len() == 1 {
        ys.pop().expect("one step")
    } else {
        let refs: Vec<&MlxArray> = ys.iter().collect();
        concatenate(&refs, 1, None)
    };
    (y, state)
}

fn expand_b_c_to_heads(
    x: &MlxArray,
    batch: usize,
    n_groups: usize,
    heads_per_group: usize,
    ssm_state: usize,
) -> MlxArray {
    let x = reshape(x, &[batch as i32, n_groups as i32, ssm_state as i32], None);
    let x = expand_dims(&x, 2, None); // [B, G, 1, Ds]
    let x = mlx_sys::ops::repeat_axis(&x, heads_per_group as i32, 2, None);
    reshape(
        &x,
        &[
            batch as i32,
            (n_groups * heads_per_group) as i32,
            ssm_state as i32,
        ],
        None,
    )
}

fn mamba_rms_norm_gated(
    x: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
    n_groups: usize,
) -> MlxArray {
    let gated = multiply(&silu(gate, None), x, None);
    let batch = gated.shape()[0];
    let seq = gated.shape()[1];
    let inter = gated.shape()[2] as usize;
    let group_size = inter / n_groups;
    let reshaped = reshape(
        &gated,
        &[batch, seq, n_groups as i32, group_size as i32],
        None,
    );
    let normed = rms_norm(&reshaped, None, eps, None);
    let flat = reshape(&normed, &[batch, seq, inter as i32], None);
    multiply(&flat, &reshape(weight, &[1, 1, inter as i32], None), None)
}
