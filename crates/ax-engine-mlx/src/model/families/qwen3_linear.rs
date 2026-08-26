use mlx_sys::{MlxArray, MlxVectorArray, add, add_rms_norm_pair, rms_norm, slice};
use std::time::Instant;

use super::super::ModelConfig;
use super::super::profile::{
    DecodeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    prefill_profile_enabled,
};
use super::super::shared::{
    LinearAttentionVerifyOutput, ffn_swiglu, ffn_swiglu_plus_residual, flatten_compiled_moe_inputs,
    linear_attention_forward, linear_attention_forward_verify_functional, moe_experts_forward,
    moe_experts_forward_with_cloned_weights, moe_experts_forward_with_shared, moe_router_qwen3,
    qwen_compiled_split_verify_ffn_plus_residual, rms_norm_opt, shared_expert_forward,
};
use crate::fastpath;
use crate::kv_cache::MlxKVCache;
use crate::per_layer_compile::apply_layer_moe_decode;
use crate::weights::LayerWeights;
use std::cell::RefCell;

thread_local! {
    static PENDING_PREFILL_FFN: RefCell<Option<MlxArray>> = const { RefCell::new(None) };
}

/// Drop a leftover deferred FFN so a new generate forward starts clean.
pub(crate) fn clear_qwen_prefill_pending_ffn() {
    PENDING_PREFILL_FFN.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn take_qwen_prefill_pending_ffn() -> Option<MlxArray> {
    PENDING_PREFILL_FFN.with(|slot| slot.borrow_mut().take())
}

pub(crate) struct QwenLinearVerifyLayerOutput {
    pub hidden: MlxArray,
    pub state: LinearAttentionVerifyOutput,
}

/// Pure short-verifier layer used by the enclosing whole-model MLX closure.
///
/// This follows the ordinary dense Qwen3.5 layer arithmetic but returns every
/// gated-delta state leaf instead of mutating a request cache. MoE and
/// post-normalized FFNs are intentionally rejected by the caller's eligibility
/// gate; keeping this function dense-only makes fallback unambiguous.
pub(crate) fn layer_forward_verify_functional(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    layer_idx: usize,
    conv_state: &MlxArray,
    recurrent_state: &MlxArray,
) -> Option<QwenLinearVerifyLayerOutput> {
    if w.router_proj.is_some() || w.ffn_post_norm.is_some() || cfg.uses_geglu {
        return None;
    }
    let seq = hidden.shape().get(1).copied()?;
    if !(2..=4).contains(&seq) {
        return None;
    }

    let fuse_la_norm = fastpath::should_qwen_la_norm_qkvz_fuse(&cfg.model_family, seq);
    let fold_exact_attn_norm = !fuse_la_norm && fastpath::qwen_linear_mtp_exact_enabled();
    if fuse_la_norm {
        crate::model::shared::set_qwen_la_norm_qkvz_fuse_weights(Some((
            w.attn_norm.clone(),
            cfg.rms_norm_eps,
        )));
    } else if fold_exact_attn_norm {
        crate::model::shared::set_qwen_la_exact_attn_norm(Some((
            w.attn_norm.clone(),
            cfg.rms_norm_eps,
        )));
    }
    let normed = if fuse_la_norm || fold_exact_attn_norm {
        hidden.clone()
    } else {
        rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None)
    };
    let state = linear_attention_forward_verify_functional(
        cfg,
        w,
        &normed,
        layer_idx,
        conv_state,
        recurrent_state,
    );
    if fuse_la_norm {
        crate::model::shared::set_qwen_la_norm_qkvz_fuse_weights(None);
    } else if fold_exact_attn_norm {
        crate::model::shared::set_qwen_la_exact_attn_norm(None);
    }
    let state = state?;

    let (residual, normed2) = qwen_linear_attn_residual_ffn_norm(
        hidden,
        &state.output,
        &w.ffn_norm,
        cfg.rms_norm_eps,
        false,
    );
    let hidden = ffn_swiglu_plus_residual(cfg, w, &normed2, None, layer_idx, &residual);
    Some(QwenLinearVerifyLayerOutput { hidden, state })
}

fn stash_qwen_prefill_pending_ffn(ffn: MlxArray) {
    PENDING_PREFILL_FFN.with(|slot| {
        *slot.borrow_mut() = Some(ffn);
    });
}

/// Fuse a deferred post-FFN residual into this layer's attn RMSNorm.
///
/// Matches `add(hidden, ffn)` then `rms_norm(..., attn_norm)`.
pub(crate) fn apply_qwen_prefill_pending_ffn(
    hidden: &MlxArray,
    ffn: &MlxArray,
    attn_norm: &MlxArray,
    eps: f32,
) -> (MlxArray, MlxArray) {
    add_rms_norm_pair(hidden, ffn, attn_norm, eps, None)
}

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
///
/// `skip_post_attention_ffn`: when `true`, return after the attention residual
/// (no FFN). Use only for the last layer of a cache-only prefill
/// (`FinalLogitsMode::Skip`) where the residual is discarded.
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    last_position_only: bool,
    skip_post_attention_ffn: bool,
) -> MlxArray {
    let seq = hidden.shape()[1] as usize;
    if let Some(compiled) = super::super::whole_verify::try_compiled_qwen_linear_verify_layer(
        cfg, w, hidden, cache, layer_idx,
    ) {
        return compiled;
    }
    crate::model::shared::set_qwen_prefill_dequant_dense_family(matches!(
        cfg.model_family.to_ascii_lowercase().as_str(),
        "qwen3_5" | "qwen3_next"
    ));
    let profile_decode_layer = seq == 1 && decode_profile_enabled();
    let profile_prefill_layer = seq > 1 && prefill_profile_enabled();
    let profile_forward_layer = profile_decode_layer || profile_prefill_layer;

    let fuse_la_norm = fastpath::should_qwen_la_norm_qkvz_fuse(&cfg.model_family, seq as i32);
    // Exact S=2..=4: skip the outer attn RMS so it compiles into the pre-Metal
    // fused QKVZ+BA qmm+unpack closure. Not the portable output gate.
    let fold_exact_attn_norm = !fuse_la_norm
        && fastpath::qwen_linear_mtp_exact_enabled()
        && (2..=4).contains(&(seq as i32));
    if fuse_la_norm {
        crate::model::shared::set_qwen_la_norm_qkvz_fuse_weights(Some((
            w.attn_norm.clone(),
            cfg.rms_norm_eps,
        )));
    } else if fold_exact_attn_norm {
        crate::model::shared::set_qwen_la_exact_attn_norm(Some((
            w.attn_norm.clone(),
            cfg.rms_norm_eps,
        )));
    }
    let (hidden_owned, normed) = if fuse_la_norm || fold_exact_attn_norm {
        (hidden.clone(), hidden.clone())
    } else if fastpath::should_qwen_prefill_interlayer_add_rms(&cfg.model_family, seq as i32)
        && let Some(ffn) = take_qwen_prefill_pending_ffn()
    {
        apply_qwen_prefill_pending_ffn(hidden, &ffn, &w.attn_norm, cfg.rms_norm_eps)
    } else {
        let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
        (hidden.clone(), normed)
    };
    let hidden = &hidden_owned;
    // linear_attention_forward includes its own per-layer profiling.
    let skip_unused_la_out = fastpath::should_qwen_prefill_skip_unused_la_out(
        &cfg.model_family,
        skip_post_attention_ffn,
        seq as i32,
    );
    let last_token_out_proj = fastpath::should_qwen_prefill_last_token_o_proj(
        &cfg.model_family,
        last_position_only,
        seq as i32,
    );
    let attn_proj = linear_attention_forward(
        cfg,
        w,
        &normed,
        cache,
        layer_idx,
        skip_unused_la_out,
        last_token_out_proj,
    );
    if fuse_la_norm {
        crate::model::shared::set_qwen_la_norm_qkvz_fuse_weights(None);
    } else if fold_exact_attn_norm {
        crate::model::shared::set_qwen_la_exact_attn_norm(None);
    }
    let residual_norm_started = profile_forward_layer.then(Instant::now);

    // Cache-only terminal layer: linear state already in cache; residual discarded.
    if skip_post_attention_ffn {
        if skip_unused_la_out {
            return hidden.clone();
        }
        let hidden = add(hidden, &attn_proj, None);
        if let Some(started) = residual_norm_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttnResidualNorm,
                started,
                &[&hidden],
            );
        }
        return hidden;
    }

    // Last-position-only: after the attention-residual add, the linear-attention
    // state has been committed to `cache`. The FFN is position-wise, so slicing
    // to the last position is safe and avoids redundant compute on preceding
    // positions whose output will be discarded by the post-loop slice.
    let last_only_active = last_position_only && seq > 1;
    let should_defer_this_ffn = w.router_proj.is_none()
        && fastpath::should_defer_qwen_prefill_ffn_residual(
            &cfg.model_family,
            seq as i32,
            layer_idx,
            cfg.is_linear_attention_layer(layer_idx.saturating_add(1)),
            skip_post_attention_ffn,
        );

    // Exact S=2..=4: one compiled residual-add + pre-FFN RMS + FFN + residual.
    // Portable attention RMS+SiLU stays outside. Compiling out_proj into this
    // closure (bbcc72ad) reproduced factory `f4b5490d` and is unhooked.
    // Last-only / deferred FFN keep the split path.
    if !last_only_active
        && !should_defer_this_ffn
        && let Some(out) =
            qwen_compiled_split_verify_ffn_plus_residual(cfg, w, hidden, &attn_proj, layer_idx)
    {
        if let Some(started) = residual_norm_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttnResidualNorm,
                started,
                &[&out],
            );
        }
        return out;
    }
    let (hidden, normed2) = qwen_linear_attn_residual_ffn_norm(
        hidden,
        &attn_proj,
        &w.ffn_norm,
        cfg.rms_norm_eps,
        last_only_active,
    );
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
        // decode step when the compile path is disabled. SSD-streamed expert
        // layers are excluded: their expert weights resolve at forward time.
        let compiled_result =
            if seq == 1 && w.expert_stream.is_none() && fastpath::moe_layer_compile_enabled() {
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
                apply_layer_moe_decode(
                    cfg.compile_cache_identity,
                    layer_idx,
                    &input_refs,
                    move |inputs: &MlxVectorArray| {
                        let (x, indices, weights, gate_up, gate, up, down, shared) =
                            schema.rebuild(inputs);
                        vec![moe_experts_forward_with_cloned_weights(
                            &cfg_clone, &x, &indices, &weights, gate_up, gate, up, down, shared,
                            None,
                        )]
                    },
                )
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
    } else if w.ffn_post_norm.is_none() && !should_defer_this_ffn && w.router_proj.is_none() {
        ffn_swiglu_plus_residual(cfg, w, &normed2, None, layer_idx, &hidden)
    } else {
        ffn_swiglu(cfg, w, &normed2, None, layer_idx)
    };
    let fused_residual =
        w.router_proj.is_none() && w.ffn_post_norm.is_none() && !should_defer_this_ffn;
    let ffn_out = if fused_residual {
        out
    } else {
        rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
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

    if should_defer_this_ffn {
        stash_qwen_prefill_pending_ffn(ffn_out);
        return hidden;
    }

    let residual_gate_started = profile_forward_layer.then(Instant::now);
    let out = if fused_residual {
        ffn_out
    } else {
        add(&hidden, &ffn_out, None)
    };
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

fn slice_bsh_last_token(x: &MlxArray) -> MlxArray {
    let shape = x.shape();
    let last = shape[1] - 1;
    slice(
        x,
        &[0, last, 0],
        &[shape[0], last + 1, shape[2]],
        &[1, 1, 1],
        None,
    )
}

/// Residual add + pre-FFN RMSNorm for a Qwen linear-attention layer.
///
/// Prefill (not last-only) uses the same `add_rms_norm_pair` fuse as
/// full-attn `standard::layer_forward`. Last-only slices after the add
/// so the FFN sees `[1, 1, H]`. Kill-switch keeps the split ops.
fn qwen_linear_attn_residual_ffn_norm(
    hidden: &MlxArray,
    attn_proj: &MlxArray,
    ffn_norm: &MlxArray,
    eps: f32,
    last_only: bool,
) -> (MlxArray, MlxArray) {
    if last_only {
        let hidden_for_add = if attn_proj.shape().get(1).copied().unwrap_or(1) == 1
            && hidden.shape().get(1).copied().unwrap_or(1) > 1
        {
            slice_bsh_last_token(hidden)
        } else {
            hidden.clone()
        };
        let residual = add(&hidden_for_add, attn_proj, None);
        let sliced = if residual.shape().get(1).copied().unwrap_or(1) > 1 {
            slice_bsh_last_token(&residual)
        } else {
            residual
        };
        let normed = rms_norm(&sliced, Some(ffn_norm), eps, None);
        return (sliced, normed);
    }
    if fastpath::qwen_linear_add_rms_norm_enabled() {
        return add_rms_norm_pair(hidden, attn_proj, ffn_norm, eps, None);
    }
    let residual = add(hidden, attn_proj, None);
    let normed = rms_norm(&residual, Some(ffn_norm), eps, None);
    (residual, normed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{MlxDtype, eval};

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn qwen_linear_attn_residual_ffn_norm_matches_split_add_rms() {
        let hidden_data: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.015625).collect();
        let attn_data: Vec<f32> = (0..256).map(|i| ((i as f32) - 64.0) * -0.0078125).collect();
        let norm_data: Vec<f32> = (0..32).map(|i| 0.75 + (i as f32) * 0.01).collect();
        let hidden = array_f32(&hidden_data, &[1, 8, 32]);
        let attn = array_f32(&attn_data, &[1, 8, 32]);
        let norm_w = array_f32(&norm_data, &[32]);
        let (residual, normed) =
            qwen_linear_attn_residual_ffn_norm(&hidden, &attn, &norm_w, 1e-6, false);
        let split_residual = add(&hidden, &attn, None);
        let split_normed = rms_norm(&split_residual, Some(&norm_w), 1e-6, None);
        eval(&[&residual, &normed, &split_residual, &split_normed]);
        assert_eq!(residual.shape(), split_residual.shape());
        assert_eq!(normed.shape(), split_normed.shape());
        for (a, b) in residual
            .data_f32()
            .iter()
            .zip(split_residual.data_f32().iter())
        {
            assert!(
                (a - b).abs() < 1e-5,
                "fused residual must match add: {a} vs {b}"
            );
        }
        for (a, b) in normed.data_f32().iter().zip(split_normed.data_f32().iter()) {
            assert!(
                (a - b).abs() < 2e-3 || (a - b).abs() / (b.abs().max(1e-6)) < 2e-3,
                "fused pre-FFN RMSNorm must match split: {a} vs {b}"
            );
        }
        let (last_residual, last_normed) =
            qwen_linear_attn_residual_ffn_norm(&hidden, &attn, &norm_w, 1e-6, true);
        eval(&[&last_residual, &last_normed]);
        assert_eq!(last_residual.shape(), vec![1, 1, 32]);
        assert_eq!(last_normed.shape(), vec![1, 1, 32]);

        let attn_last = slice_bsh_last_token(&attn);
        let (last_from_sliced, _) =
            qwen_linear_attn_residual_ffn_norm(&hidden, &attn_last, &norm_w, 1e-6, true);
        eval(&[&last_from_sliced]);
        assert_eq!(last_from_sliced.shape(), vec![1, 1, 32]);
        for (a, b) in last_residual
            .data_f32()
            .iter()
            .zip(last_from_sliced.data_f32().iter())
        {
            assert!(
                (a - b).abs() < 1e-5,
                "last-token o_proj residual must match full-seq last row: {a} vs {b}"
            );
        }
    }

    #[test]
    fn apply_qwen_prefill_pending_ffn_matches_add_then_rms() {
        let hidden_data: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.015625).collect();
        let ffn_data: Vec<f32> = (0..256).map(|i| ((i as f32) - 32.0) * 0.01171875).collect();
        let norm_data: Vec<f32> = (0..32).map(|i| 0.8 + (i as f32) * 0.008).collect();
        let hidden = array_f32(&hidden_data, &[1, 8, 32]);
        let ffn = array_f32(&ffn_data, &[1, 8, 32]);
        let norm_w = array_f32(&norm_data, &[32]);
        let (fused_hidden, fused_norm) =
            apply_qwen_prefill_pending_ffn(&hidden, &ffn, &norm_w, 1e-6);
        let split_hidden = add(&hidden, &ffn, None);
        let split_norm = rms_norm(&split_hidden, Some(&norm_w), 1e-6, None);
        eval(&[&fused_hidden, &fused_norm, &split_hidden, &split_norm]);
        assert_eq!(fused_hidden.shape(), split_hidden.shape());
        assert_eq!(fused_norm.shape(), split_norm.shape());
        for (a, b) in fused_hidden
            .data_f32()
            .iter()
            .zip(split_hidden.data_f32().iter())
        {
            assert!((a - b).abs() < 1e-5, "fused residual {a} vs add {b}");
        }
        for (a, b) in fused_norm
            .data_f32()
            .iter()
            .zip(split_norm.data_f32().iter())
        {
            assert!(
                (a - b).abs() < 2e-3 || (a - b).abs() / (b.abs().max(1e-6)) < 2e-3,
                "fused attn RMSNorm {a} vs split {b}"
            );
        }
        clear_qwen_prefill_pending_ffn();
        stash_qwen_prefill_pending_ffn(ffn);
        assert!(take_qwen_prefill_pending_ffn().is_some());
        assert!(take_qwen_prefill_pending_ffn().is_none());
    }
}
