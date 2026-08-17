use mlx_sys::{MlxArray, add, multiply, reshape, rms_norm, rope, transpose};

use super::super::ModelConfig;
use super::super::config::layer_params;
use super::super::shared::{
    attention_mask_array, full_precision_attention, prepare_value_bhsd, qw,
};
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Full layer forward for Muse Glimmer (Meta dense 30B image-text agent,
/// text tower).
///
/// Reference: mlxcel `muse_glimmer_layers.rs`. Key properties:
/// - **iRoPE by layer type**: `sliding_attention` layers rotate at the
///   uniform theta (500k) with `traditional=false` over the full head_dim;
///   `full_attention` layers are NoPE (`rope_dims == 0` in `LayerConfig`).
/// - **Weightless QK RMSNorm** (eps = `rms_norm_eps`) applied before RoPE.
/// - **SDPA scale** = `head_dim^-0.5 * qk_scale_factor`, folded into
///   `cfg.query_scale` at config build.
/// - **Sigmoid attention output gate**: a separate `gate_proj` on the
///   normed layer input, multiplied per head/dim into the attention output
///   before `o_proj`.
/// - **Gemma-style sandwich norms** with alternating eps: the input and
///   pre-FFN norms use `rms_norm_eps` (1e-6); the post-attention and
///   post-FFN norms use `post_norm_eps` (1e-8). All four norm weights are
///   `(1 + w)`-lifted at load (`WeightSanitize::HfLayerNormsOnly`).
/// - **Dense SiLU FFN** (SwiGLU) on every layer; no MoE.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    shared_mask: Option<&Option<MlxArray>>,
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, _rope_freqs, sliding_window, _, _) =
        layer_params(cfg, layer_idx);
    let use_rope = rope_dims > 0;
    let seq = hidden.shape()[1] as usize;

    // 1. Input norm (centered `(1+w)` lifted at load; eps = rms_norm_eps).
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    // 2. Q/K/V projections (separate; H*D != hidden_size, so head geometry
    //    comes from the projection widths, not `hidden_size / n_heads`).
    let q_raw = qw(
        &normed,
        w.q_proj
            .as_ref()
            .expect("muse_glimmer layer must have q_proj"),
    );
    let k_raw = qw(
        &normed,
        w.k_proj
            .as_ref()
            .expect("muse_glimmer layer must have k_proj"),
    );
    let v_raw = qw(
        &normed,
        w.v_proj
            .as_ref()
            .expect("muse_glimmer layer must have v_proj"),
    );
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");

    let q = reshape(
        &q_raw,
        &[1, seq as i32, cfg.n_heads as i32, head_dim as i32],
        None,
    );
    let k = reshape(
        &k_raw,
        &[1, seq as i32, kv_heads as i32, head_dim as i32],
        None,
    );
    let v = reshape(
        &v_raw,
        &[1, seq as i32, kv_heads as i32, head_dim as i32],
        None,
    );

    // 3. Weightless QK RMSNorm over head_dim (reference applies it before
    //    the transpose; the last axis is head_dim in both layouts, so norm
    //    and transpose commute).
    let q = rms_norm(&q, None, cfg.rms_norm_eps, None);
    let k = rms_norm(&k, None, cfg.rms_norm_eps, None);

    let q = transpose(&q, &[0, 2, 1, 3], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = prepare_value_bhsd(v, false, kv_heads, head_dim, seq, cfg.rms_norm_eps);

    // 4. RoPE on sliding layers only, AFTER the QK norm (reference order:
    //    norm → transpose → rope). Full-attention layers are NoPE.
    let (q, k) = if use_rope {
        (
            rope(
                &q,
                rope_dims as i32,
                false, // split-half convention (mlx default), not traditional
                Some(rope_theta),
                1.0,
                token_offset as i32,
                None,
                None,
            ),
            rope(
                &k,
                rope_dims as i32,
                false,
                Some(rope_theta),
                1.0,
                token_offset as i32,
                None,
                None,
            ),
        )
    } else {
        (q, k)
    };

    // 5. KV cache append.
    let (ck, cv) = if seq == 1 {
        cache.append_with_retained_window(layer_idx, k, v, sliding_window)
    } else {
        cache.append(layer_idx, k, v)
    };

    // 6. SDPA mask (window 2048 on sliding layers; None within the window).
    let key_len = ck.shape()[2] as usize;
    let local_mask: Option<MlxArray>;
    let mask_opt: &Option<MlxArray> = if let Some(m) = shared_mask {
        m
    } else {
        local_mask = attention_mask_array(seq, key_len, sliding_window);
        &local_mask
    };

    // 7. SDPA with the folded qk_scale_factor query scale.
    let attn_sdpa = full_precision_attention(&q, &ck, &cv, cfg.query_scale, seq, mask_opt);

    // 8. Back to [1, seq, H, D] and apply the sigmoid output gate on the
    //    un-flattened view (per head/dim), gate input = the normed x.
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);
    let gate_raw = qw(
        &normed,
        w.attn_out_gate
            .as_ref()
            .expect("muse_glimmer layer must have attn_out_gate"),
    );
    let gate = reshape(
        &gate_raw,
        &[1, seq as i32, cfg.n_heads as i32, head_dim as i32],
        None,
    );
    let gate = mlx_sys::ops::sigmoid(&gate, None);
    let gated = multiply(&attn_out, &gate, None);

    let attn_flat = reshape(
        &gated,
        &[1, seq as i32, (cfg.n_heads * head_dim) as i32],
        None,
    );
    let attn_proj = qw(
        &attn_flat,
        w.o_proj
            .as_ref()
            .expect("muse_glimmer layer must have o_proj"),
    );

    // 9. Post-attention sandwich norm (eps = post_norm_eps), then residual.
    let attn_normed = rms_norm(
        &attn_proj,
        Some(
            w.attn_post_norm
                .as_ref()
                .expect("muse_glimmer layer must have attn_post_norm"),
        ),
        cfg.post_norm_eps,
        None,
    );
    let post_attn = add(hidden, &attn_normed, None);

    // 10. Pre-FFN norm (eps = rms_norm_eps) → SiLU SwiGLU FFN →
    //     post-FFN sandwich norm (eps = post_norm_eps) → residual.
    let ffn_in = rms_norm(&post_attn, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let gate_ffn = qw(
        &ffn_in,
        w.gate_proj
            .as_ref()
            .expect("muse_glimmer layer must have gate_proj"),
    );
    let up = qw(
        &ffn_in,
        w.up_proj
            .as_ref()
            .expect("muse_glimmer layer must have up_proj"),
    );
    let act = multiply(&mlx_sys::ops::silu(&gate_ffn, None), &up, None);
    let ffn_out = qw(
        &act,
        w.down_proj
            .as_ref()
            .expect("muse_glimmer layer must have down_proj"),
    );
    let ffn_normed = rms_norm(
        &ffn_out,
        Some(
            w.ffn_post_norm
                .as_ref()
                .expect("muse_glimmer layer must have ffn_post_norm"),
        ),
        cfg.post_norm_eps,
        None,
    );
    add(&post_attn, &ffn_normed, None)
}
