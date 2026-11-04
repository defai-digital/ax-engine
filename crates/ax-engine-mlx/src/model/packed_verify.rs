// SPDX-License-Identifier: Apache-2.0

//! Row-packed verify forward for several sequences at once (batched DFlash2).
//!
//! Every lane's verify rows are concatenated into one `[1, ΣW, hidden]`
//! activation. Position-wise work — norms, every projection, the gated norm,
//! the FFN and the LM head — runs once over the packed rows, so the weights
//! are read once for all lanes. Only the sequence-dependent cores split by
//! lane: for full attention, QK norm + RoPE at the lane's offset, the KV write
//! and SDPA on the lane's own cache; for gated delta, the conv and recurrence
//! on the lane's own state (with its verify prefix checkpoint / tape /
//! projection stash). Each lane's cache therefore ends exactly where a
//! single-lane verify of its rows would leave it, and the runner commits each
//! lane with the existing adopt / replay path.
//!
//! Scope: the Qwen 3.5 / 3.8 dense hybrid text model (full attention with an
//! output gate plus gated-delta layers, dense SwiGLU FFN, no sliding window,
//! KV sharing, per-layer inputs, MoE or post-norms). [`supported`] checks it.

use mlx_sys::{MlxArray, add, concatenate, rms_norm, slice};

use crate::kv_cache::{MlxAttentionKv, MlxKVCache};
use crate::model::config::layer_params;
use crate::model::layer_capture;
use crate::model::shared::attention::{
    build_layer_masks_for_forward, direct_qk_norm_rope_route_enabled_for_family,
    flatten_attention_output_bhsd, full_precision_attention_with_window,
    prepare_value_bhsd_from_proj, qk_norm_rope_bhsd_from_proj_with_route,
};
use crate::model::shared::linear_attention::{
    linear_attention_core, linear_attention_gate_out, linear_attention_inputs,
};
use crate::model::shared::mlp::{
    attention_output_projection_with_post_norm_policy, ffn_swiglu_plus_residual, qkv_project,
};
use crate::model::shared::utils::ProjectionBatchPolicy;
use crate::model::{ModelConfig, embed_tokens_arr, lm_head_verify_window_projection};
use crate::weights::ModelWeights;

/// One lane of a packed verify: its cache (a clone the caller owns, with the
/// linear prefix capture already begun), the position of its first row and
/// how many rows it verifies.
pub(crate) struct PackedLane<'a> {
    pub(crate) cache: &'a mut MlxKVCache,
    pub(crate) token_offset: usize,
    pub(crate) rows: usize,
}

/// Whether [`packed_verify_forward`] supports this model.
pub(crate) fn supported(cfg: &ModelConfig, weights: &ModelWeights) -> bool {
    if cfg.linear_attention.is_none()
        || cfg.model_family == "qwen4_exp"
        || cfg.deepseek_v4.is_some()
        || cfg.moe_expert_count != 0
        || weights.per_layer_embed.is_some()
        || cfg.hidden_states_scale.is_some()
    {
        return false;
    }
    weights.layers.iter().enumerate().all(|(layer_idx, w)| {
        let (_, _, _, _, sliding_window, kv_source, _) = layer_params(cfg, layer_idx);
        sliding_window.is_none()
            && kv_source.is_none()
            && w.per_layer_gate.is_none()
            && w.router_proj.is_none()
            && w.ffn_post_norm.is_none()
            && w.layer_scalar.is_none()
            && w.unlimited_ocr2.is_none()
            && w.glm_mla_attn.is_none()
            && (cfg.is_linear_attention_layer(layer_idx) || w.o_proj.is_some())
    })
}

fn rows_of(x: &MlxArray, start: usize, rows: usize) -> MlxArray {
    let shape = x.shape();
    let mut begin = vec![0_i32; shape.len()];
    let mut end = shape.clone();
    begin[1] = start as i32;
    end[1] = (start + rows) as i32;
    let strides = vec![1_i32; shape.len()];
    slice(x, &begin, &end, &strides, None)
}

fn join_rows(parts: &[MlxArray]) -> MlxArray {
    if parts.len() == 1 {
        return parts[0].clone();
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    concatenate(&refs, 1, None)
}

/// Builds (lazily) the verify forward of `ids` — every lane's rows in lane
/// order — and returns `[ΣW, vocab]` logits and the final-norm hidden rows
/// `[1, ΣW, hidden]`. The hidden states of the layers
/// a [`layer_capture`] guard names are observed over the packed rows; the
/// caller slices them per lane. Each lane's cache gains its rows' K/V and
/// linear state but is not advanced.
pub(crate) fn packed_verify_forward(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    ids: &MlxArray,
    lanes: &mut [PackedLane<'_>],
) -> (MlxArray, MlxArray) {
    let total: usize = lanes.iter().map(|lane| lane.rows).sum();
    assert_eq!(ids.shape(), vec![total as i32], "one id per packed row");
    let starts: Vec<usize> = lanes
        .iter()
        .scan(0usize, |acc, lane| {
            let start = *acc;
            *acc += lane.rows;
            Some(start)
        })
        .collect();

    let mut hidden = embed_tokens_arr(ids, &weights.token_embedding, cfg.hidden_size);
    hidden = mlx_sys::astype(
        &hidden,
        crate::fastpath::activation_dtype_for_family(&cfg.model_family),
        None,
    );
    hidden = crate::model::shared::utils::maybe_weightless_embed_norm(cfg, hidden);

    let layer_count = weights.layers.len();
    let masks: Vec<Vec<Option<MlxArray>>> = lanes
        .iter()
        .map(|lane| {
            build_layer_masks_for_forward(
                cfg,
                layer_count,
                lane.rows,
                lane.token_offset + lane.rows,
                lane.cache,
            )
        })
        .collect();
    let linear_cfg = cfg.linear_attention.as_ref();

    for (layer_idx, w) in weights.layers.iter().enumerate() {
        let attn_proj = if cfg.is_linear_attention_layer(layer_idx) {
            let linear_cfg = linear_cfg.expect("hybrid model has a linear-attention config");
            let linear_w = w
                .linear_attn
                .as_ref()
                .expect("linear-attention layer has linear weights");
            let normed = rms_norm(&hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
            let (qkv, z, a, b) =
                linear_attention_inputs(cfg, linear_cfg, linear_w, &normed, total as i32, false);
            let mut outs = Vec::with_capacity(lanes.len());
            for (lane, &start) in lanes.iter_mut().zip(&starts) {
                let qkv_l = rows_of(&qkv, start, lane.rows);
                let a_l = rows_of(&a, start, lane.rows);
                let b_l = rows_of(&b, start, lane.rows);
                if lane.cache.linear_prefix_capture_after().is_some() {
                    lane.cache.set_linear_mtp_projection_stash(
                        layer_idx,
                        qkv_l.clone(),
                        a_l.clone(),
                        b_l.clone(),
                    );
                }
                outs.push(linear_attention_core(
                    cfg, linear_cfg, linear_w, &qkv_l, &a_l, &b_l, lane.cache, layer_idx, false,
                ));
            }
            let out = join_rows(&outs);
            linear_attention_gate_out(
                cfg,
                linear_w,
                &out,
                &z,
                total as i32,
                linear_cfg.value_dim() as i32,
                layer_idx,
                false,
            )
        } else {
            let (head_dim, rope_theta, rope_dims, layer_rope_freqs, _, _, v_norm_no_scale) =
                layer_params(cfg, layer_idx);
            let normed = rms_norm(&hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
            let (q_raw, k_raw, v_raw, attn_gate) = qkv_project(cfg, w, &normed, head_dim);
            let kv_heads = (k_raw.shape()[2] as usize)
                .checked_div(head_dim)
                .expect("k projection output must divide by head_dim");
            let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
            let (rope_base, rope_freqs_ref) = rope_freqs
                .map(|f| (None, Some(f)))
                .unwrap_or((Some(rope_theta), None));
            let direct_q = direct_qk_norm_rope_route_enabled_for_family(
                cfg.model_family.as_str(),
                w.q_norm.as_ref(),
            );
            let direct_k = direct_qk_norm_rope_route_enabled_for_family(
                cfg.model_family.as_str(),
                w.k_norm.as_ref(),
            );
            let mut outs = Vec::with_capacity(lanes.len());
            for ((lane, &start), lane_masks) in lanes.iter_mut().zip(&starts).zip(&masks) {
                let rows = lane.rows;
                let v = prepare_value_bhsd_from_proj(
                    &rows_of(&v_raw, start, rows),
                    v_norm_no_scale,
                    kv_heads,
                    head_dim,
                    rows,
                    cfg.rms_norm_eps,
                );
                let q_rope = qk_norm_rope_bhsd_from_proj_with_route(
                    &rows_of(&q_raw, start, rows),
                    w.q_norm.as_ref(),
                    cfg.n_heads,
                    head_dim,
                    rows,
                    cfg.rms_norm_eps,
                    rope_dims,
                    rope_base,
                    lane.token_offset,
                    rope_freqs_ref,
                    direct_q,
                );
                let k_rope = qk_norm_rope_bhsd_from_proj_with_route(
                    &rows_of(&k_raw, start, rows),
                    w.k_norm.as_ref(),
                    kv_heads,
                    head_dim,
                    rows,
                    cfg.rms_norm_eps,
                    rope_dims,
                    rope_base,
                    lane.token_offset,
                    rope_freqs_ref,
                    direct_k,
                );
                let attention_kv = lane
                    .cache
                    .append_with_retained_window_for_attention(layer_idx, k_rope, v, None);
                let (k, v) = match attention_kv {
                    MlxAttentionKv::Dense { k, v } => (k, v),
                    MlxAttentionKv::Paged(view) => view.materialize(),
                };
                let attn = full_precision_attention_with_window(
                    &q_rope,
                    &k,
                    &v,
                    cfg.query_scale,
                    rows,
                    &lane_masks[layer_idx],
                    None,
                    None,
                );
                outs.push(flatten_attention_output_bhsd(
                    &attn,
                    rows,
                    cfg.n_heads,
                    head_dim,
                ));
            }
            let attn_flat = join_rows(&outs);
            attention_output_projection_with_post_norm_policy(
                &attn_flat,
                attn_gate.as_ref(),
                w.o_proj
                    .as_ref()
                    .expect("full-attention layer must have o_proj"),
                w.attn_post_norm.as_ref(),
                cfg.rms_norm_eps,
                ProjectionBatchPolicy::Shared,
            )
        };
        let residual = add(&hidden, &attn_proj, None);
        let normed2 = rms_norm(&residual, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
        hidden = ffn_swiglu_plus_residual(cfg, w, &normed2, None, layer_idx, &residual);
        layer_capture::observe(layer_idx, &hidden);
    }

    let normed = rms_norm(&hidden, Some(weights.final_norm()), cfg.rms_norm_eps, None);
    let logits = lm_head_verify_window_projection(
        &normed,
        &weights.lm_head,
        &cfg.model_family,
        total as i32,
        cfg.hidden_size as i32,
    );
    let logits = mlx_sys::reshape(&logits, &[total as i32, cfg.vocab_size as i32], None);
    (logits, normed)
}
