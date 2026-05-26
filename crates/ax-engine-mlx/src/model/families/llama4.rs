use mlx_sys::{
    MlxArray, MlxDtype, add, arange, argpartition_axis, astype, broadcast_to, divide, floor, log,
    multiply, negative, reshape, rms_norm, rope, slice_last_dim, take_along_axis, transpose,
};

use super::super::ModelConfig;
use super::super::config::layer_params;
use super::super::shared::{
    attention_mask_array, attention_output_projection, full_precision_attention,
    moe_experts_forward, prepare_value_bhsd, qkv_project, qw, shared_expert_forward,
};
use super::super::turboquant_context::TurboQuantModelDecodeContext;
use crate::kv_cache::MlxKVCache;
use crate::weights::LayerWeights;

/// Returns true when this layer uses RoPE, false for iRoPE (no-rope) layers.
fn layer_uses_rope(cfg: &ModelConfig, layer_idx: usize) -> bool {
    cfg.no_rope_layer_interval == 0 || !(layer_idx + 1).is_multiple_of(cfg.no_rope_layer_interval)
}

/// Returns true when this layer is a MoE layer (same interval as no-rope).
fn layer_is_moe(cfg: &ModelConfig, layer_idx: usize) -> bool {
    !layer_uses_rope(cfg, layer_idx) && cfg.moe_expert_count > 0
}

/// Full layer forward for LLaMA-4 (Scout / Maverick).
///
/// Key differences from LLaMA-3 / standard families:
/// - **iRoPE**: every N-th layer (`no_rope_layer_interval`) has NO RoPE applied.
/// - **QK norm without weight**: rope layers apply `rms_norm(q/k, weight=None, eps=1e-6)`.
/// - **Temperature scaling**: no-rope layers scale queries by a position-dependent
///   temperature: `q *= log(floor(pos / floor_scale) + 1) * attn_scale + 1`.
/// - **Interleaved MoE**: no-rope layers are MoE; rope layers are dense SwiGLU FFN.
/// - **Dense FFN size**: rope layers use `intermediate_size_mlp` when set.
/// - **Traditional RoPE**: uses `traditional=true` (interleaved halves, not split-half).
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    shared_mask: Option<&Option<MlxArray>>,
    _turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let use_rope = layer_uses_rope(cfg, layer_idx);
    let is_moe = layer_is_moe(cfg, layer_idx);
    let (head_dim, rope_theta, rope_dims, _rope_freqs, sliding_window, _, _) =
        layer_params(cfg, layer_idx);

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let seq = hidden.shape()[1] as usize;

    // 2. QKV projection.
    let (q_raw, k_raw, v_raw, _attn_gate) = qkv_project(cfg, w, &normed, head_dim);

    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");

    // Reshape to BSHD for norm/RoPE.
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

    // 3. QK norm without weight (all rope layers, regardless of iRoPE interval config).
    //    LLaMA4 applies no-weight RMSNorm on every rope layer.
    let (q, k) = if use_rope {
        (
            rms_norm(&q, None, 1e-6, None),
            rms_norm(&k, None, 1e-6, None),
        )
    } else {
        (q, k)
    };

    // 4. Transpose to BHSD for RoPE and KV cache.
    let q = transpose(&q, &[0, 2, 1, 3], None);
    let k = transpose(&k, &[0, 2, 1, 3], None);
    let v = prepare_value_bhsd(v, false, kv_heads, head_dim, seq, cfg.rms_norm_eps);

    // 5. RoPE (traditional=true for LLaMA4, rope layers only).
    let (rope_base, rope_freqs_ref) = cfg
        .rope_freqs
        .as_ref()
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));

    let q_out = if use_rope {
        rope(
            &q,
            rope_dims as i32,
            true, // traditional RoPE for LLaMA4
            rope_base,
            1.0,
            token_offset as i32,
            rope_freqs_ref,
            None,
        )
    } else {
        q
    };
    let k_out = if use_rope {
        rope(
            &k,
            rope_dims as i32,
            true, // traditional RoPE for LLaMA4
            rope_base,
            1.0,
            token_offset as i32,
            rope_freqs_ref,
            None,
        )
    } else {
        k
    };

    // 6. Temperature scaling for no-rope layers.
    //    scale_i = log(floor(pos / floor_scale) + 1) * attn_scale + 1
    //    where pos ∈ [token_offset+1 .. token_offset+seq+1].
    //    For decode (seq=1, token_offset=0) this is log(floor(1/8192)+1)*0.1+1 = 1.0, a no-op.
    let q_final = if !use_rope && cfg.attn_temperature_scale > 0.0 {
        let positions = arange(
            (token_offset + 1) as f64,
            (token_offset + seq + 1) as f64,
            1.0,
            MlxDtype::Float32,
            None,
        );
        let floor_scale_arr =
            mlx_sys::ops::cached_scalar(cfg.attn_temperature_floor, MlxDtype::Float32);
        let floored = floor(&divide(&positions, &floor_scale_arr, None), None);
        let one = mlx_sys::ops::cached_scalar(1.0, MlxDtype::Float32);
        let log_part = log(&add(&floored, &one, None), None);
        let attn_scale_arr =
            mlx_sys::ops::cached_scalar(cfg.attn_temperature_scale, MlxDtype::Float32);
        let attn_scales = add(&multiply(&log_part, &attn_scale_arr, None), &one, None);
        // scales: [seq] → [1, 1, seq, 1] to broadcast over [B, H, seq, D].
        let scales = reshape(&attn_scales, &[1, 1, seq as i32, 1], None);
        let q_dtype = q_out.dtype();
        astype(&multiply(&q_out, &scales, None), q_dtype, None)
    } else {
        q_out
    };

    // 7. KV cache append.
    let (ck, cv) = if seq == 1 {
        cache.append_with_retained_window(layer_idx, k_out, v, sliding_window)
    } else {
        cache.append(layer_idx, k_out, v)
    };

    // 8. SDPA mask.
    let key_len = ck.shape()[2] as usize;
    let local_mask: Option<MlxArray>;
    let mask_opt: &Option<MlxArray> = if let Some(m) = shared_mask {
        m
    } else {
        local_mask = attention_mask_array(seq, key_len, sliding_window);
        &local_mask
    };

    // 9. Scaled dot-product attention.
    let attn_sdpa = full_precision_attention(&q_final, &ck, &cv, cfg.query_scale, seq, mask_opt);

    // 10. Transpose back: [1, n_heads, seq, head_dim] → [1, seq, n_heads, head_dim].
    let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);

    // 11. Reshape to [1, seq, hidden].
    let attn_flat = reshape(
        &attn_out,
        &[1, seq as i32, (cfg.n_heads * head_dim) as i32],
        None,
    );

    // 12. Output projection (no attn_gate in LLaMA4).
    let attn_proj = attention_output_projection(
        &attn_flat,
        None, // LLaMA4 has no attn_output_gate
        w.o_proj.as_ref().expect("LLaMA4 layer must have o_proj"),
    );

    // 13. Residual add.
    let post_attn = add(hidden, &attn_proj, None);

    // 14. Pre-FFN norm.
    let normed2 = rms_norm(&post_attn, Some(&w.ffn_norm), cfg.rms_norm_eps, None);

    // 15. FFN: MoE for no-rope layers, dense SwiGLU for rope layers.
    let ffn_out = if is_moe {
        llama4_moe_forward(cfg, w, &normed2)
    } else {
        llama4_dense_ffn(w, &normed2)
    };

    // 16. Residual add.
    add(&post_attn, &ffn_out, None)
}

/// Dense SwiGLU FFN for LLaMA4 rope layers.
///
/// Uses the standard gate/up/down weights. LLaMA4 uses SiLU activation (SwiGLU),
/// matching LLaMA3. The `intermediate_size_mlp` field controls the projection
/// size for dense layers; the weights are already the correct shape at load time.
fn llama4_dense_ffn(w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let gate = qw(
        x,
        w.gate_proj
            .as_ref()
            .expect("LLaMA4 dense FFN layer must have gate_proj"),
    );
    let up = qw(
        x,
        w.up_proj
            .as_ref()
            .expect("LLaMA4 dense FFN layer must have up_proj"),
    );
    let act = multiply(&mlx_sys::ops::silu(&gate, None), &up, None);
    qw(
        &act,
        w.down_proj
            .as_ref()
            .expect("LLaMA4 dense FFN layer must have down_proj"),
    )
}

/// LLaMA-4 MoE: top-1 sigmoid routing + shared expert.
///
/// Router uses sigmoid (not softmax) scores. Top-1 selection via argpartition.
/// The expert output is summed with a shared expert applied to the same input.
fn llama4_moe_forward(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let router = w
        .router_proj
        .as_ref()
        .expect("LLaMA4 MoE layer must have router_proj");
    let logits = qw(x, router);

    // Top-1 by argpartition on negated logits (select the maximum logit).
    let neg_logits = negative(&logits, None);
    let last_axis = neg_logits.ndim() as i32 - 1;
    let indices_full = argpartition_axis(&neg_logits, 0, last_axis, None);
    // Slice out the first (minimum-of-negated = maximum) index.
    let indices = slice_last_dim(&indices_full, 0, 1, None);

    // Gather raw logit for the selected expert and apply sigmoid.
    let scores_raw = take_along_axis(&logits, &indices, last_axis, None);
    let scores = astype(
        &mlx_sys::ops::sigmoid(&astype(&scores_raw, MlxDtype::Float32, None), None),
        x.dtype(),
        None,
    );

    // LLaMA4 uses input-weighted routing: expert receives (score * x), not score * expert(x).
    // silu(W_gate * (s*x)) ≠ s * silu(W_gate * x), so input- and output-weighting differ.
    let x_scaled = multiply(x, &scores, None);
    let ones_scores = broadcast_to(
        &mlx_sys::ops::cached_scalar(1.0_f32, scores.dtype()),
        &scores.shape(),
        None,
    );
    let expert_out = moe_experts_forward(cfg, w, &x_scaled, &indices, &ones_scores);

    // Shared expert: applied to the pre-normed input and added directly.
    let shared_out = shared_expert_forward(cfg, w, x);

    add(&expert_out, &shared_out, None)
}
