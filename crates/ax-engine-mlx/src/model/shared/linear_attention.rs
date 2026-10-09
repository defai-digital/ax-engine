use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, MlxVectorArray, async_eval, concatenate, contiguous,
    eval, qwen_linear_attention_inputs_packed, qwen_linear_attention_inputs_packed_compiled,
    qwen_linear_attention_post_input, qwen_linear_attention_post_input_compiled, reshape, rms_norm,
    rms_norm_quantized_matmul, slice, slice_last_dim, zeros,
};
use std::time::Instant;

use super::super::config::{LinearAttentionConfig, ModelConfig};
use super::super::profile::{
    LinearAttentionProfileStage, linear_attention_profile_enabled,
    linear_attention_profile_eval_elapsed, record_linear_attention_decode_post_input_metal_attempt,
    record_linear_attention_decode_post_input_metal_fallback,
    record_linear_attention_decode_post_input_metal_hit,
    record_linear_attention_decode_post_input_metal_profile_blocked,
    record_linear_attention_direct_cpp_inputs_attempt,
    record_linear_attention_direct_cpp_inputs_fallback,
    record_linear_attention_direct_cpp_inputs_hit,
    record_linear_attention_direct_cpp_inputs_profile_blocked,
    record_linear_attention_direct_cpp_post_input_attempt,
    record_linear_attention_direct_cpp_post_input_fallback,
    record_linear_attention_direct_cpp_post_input_hit,
    record_linear_attention_direct_cpp_post_input_profile_blocked,
    record_linear_attention_profile_layer,
};
use super::utils::qw;
use crate::batched_linear_state::BatchedLinearState;
use crate::fastpath;
use crate::kv_cache::MlxKVCache;
use crate::linear_attention_ops::{
    gated_delta_fused_verify_from_qkv, gated_delta_fused_verify_no_checkpoint_from_qkv,
    gated_delta_kernel, gated_delta_kernel_verify_no_checkpoint,
    gated_delta_kernel_with_prefix_checkpoint, gated_delta_kernel_with_tape,
    linear_attention_conv1d, linear_attention_decode_post_input_metal,
    normalize_linear_attention_qk, replay_gated_delta_tape, rms_norm_gated_with_full_gate_policy,
    slice_seq_row_4d, split_linear_attention_qkv,
};
use crate::weights::{
    LayerWeights, LinearAttentionWeights, QuantizedWeight, SHARED_VERIFY_COMPILE_LAYER,
    compile_quant_contract_salt,
};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static LA_NORM_QKVZ_FUSE: RefCell<Option<(MlxArray, f32)>> = const { RefCell::new(None) };
    static LA_EXACT_ATTN_NORM: RefCell<Option<(MlxArray, f32)>> = const { RefCell::new(None) };
    static LA_PRE_GATE_Z: RefCell<Option<MlxArray>> = const { RefCell::new(None) };
}

/// Bind `attn_norm` so packed QKVZ/BA can fuse RMSNorm into the qmm.
pub(crate) fn set_qwen_la_norm_qkvz_fuse_weights(norm: Option<(MlxArray, f32)>) {
    LA_NORM_QKVZ_FUSE.with(|slot| {
        *slot.borrow_mut() = norm;
    });
}

fn qwen_la_norm_qkvz_fuse_weights() -> Option<(MlxArray, f32)> {
    LA_NORM_QKVZ_FUSE.with(|slot| slot.borrow().clone())
}

/// Bind `attn_norm` so exact S=2..=4 can compile RMS + fused QKVZ+BA qmm +
/// unpack as one pre-Metal closure. Not the affine `rms_norm_quantized_matmul`
/// fuse and not the portable RMS+SiLU output gate.
pub(crate) fn set_qwen_la_exact_attn_norm(norm: Option<(MlxArray, f32)>) {
    LA_EXACT_ATTN_NORM.with(|slot| {
        *slot.borrow_mut() = norm;
    });
}

fn qwen_la_exact_attn_norm() -> Option<(MlxArray, f32)> {
    LA_EXACT_ATTN_NORM.with(|slot| slot.borrow().clone())
}

/// Apply the bound exact `attn_norm` when the compile path that folds it
/// is not taken. The layer shell skips the outer RMS whenever the TLS is
/// set; dropping it here would project raw residual.
fn apply_bound_exact_attn_norm(x: &MlxArray) -> MlxArray {
    match qwen_la_exact_attn_norm() {
        Some((norm_w, eps)) => rms_norm(x, Some(&norm_w), eps, None),
        None => x.clone(),
    }
}

pub(crate) fn qw_rms_norm_qmm(
    x: &MlxArray,
    norm_w: &MlxArray,
    eps: f32,
    proj: &QuantizedWeight,
) -> MlxArray {
    // The fused C++ helper infers the mode from the bias channel: affine
    // with group biases, scales-only MXFP4 without. MXFP8/NVFP4 keep the
    // mode-aware `qw` path (no fused-path evidence for them).
    match &proj.scales {
        Some(scales)
            if matches!(
                proj.mlx_quantization_mode(),
                MlxQuantizationMode::Affine | MlxQuantizationMode::Mxfp4
            ) =>
        {
            rms_norm_quantized_matmul(
                x,
                norm_w,
                eps,
                &proj.weight,
                scales,
                proj.biases.as_ref(),
                proj.group_size,
                proj.bits,
                None,
            )
        }
        _ => qw(&rms_norm(x, Some(norm_w), eps, None), proj),
    }
}

type InitialRecurrentZerosCache = Option<((i32, i32, i32), MlxArray)>;

thread_local! {
    static INITIAL_RECURRENT_ZEROS: RefCell<InitialRecurrentZerosCache> =
        const { RefCell::new(None) };
    static PREFILL_LA_CONTIG_W: RefCell<HashMap<usize, QuantizedWeight>> =
        RefCell::new(HashMap::new());
}

/// Initial gated-delta recurrent state (`mx.zeros(..., float32)`).
///
/// When [`fastpath::should_reuse_la_initial_state_zeros`] is on, reuse one
/// template per thread for the common Qwen hybrid shape so p2048 chunk 1
/// does not allocate 48 identical zeros tensors.
fn initial_recurrent_state_zeros(linear_cfg: &LinearAttentionConfig) -> MlxArray {
    let dims = (
        linear_cfg.num_value_heads as i32,
        linear_cfg.value_head_dim as i32,
        linear_cfg.key_head_dim as i32,
    );
    let shape = [1, dims.0, dims.1, dims.2];
    if !fastpath::should_reuse_la_initial_state_zeros() {
        return zeros(&shape, MlxDtype::Float32, None);
    }
    INITIAL_RECURRENT_ZEROS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some((cached_dims, arr)) = slot.as_ref()
            && *cached_dims == dims
        {
            return arr.clone();
        }
        let arr = zeros(&shape, MlxDtype::Float32, None);
        *slot = Some((dims, arr.clone()));
        arr
    })
}

/// Functional state returned by one short Qwen target-verifier
/// linear-attention layer.
///
/// The whole-verifier compiler cannot mutate [`MlxKVCache`] while tracing.
/// Every recurrent/cache leaf therefore crosses the closure boundary as an
/// explicit array. A compact delta tape and the projections needed to rebuild
/// convolution/K are returned instead of a second full recurrent checkpoint.
pub(crate) struct LinearAttentionVerifyOutput {
    pub output: MlxArray,
    pub conv_state: MlxArray,
    pub recurrent_state: MlxArray,
    pub qkv: MlxArray,
    pub a: MlxArray,
    pub tape: MlxArray,
}

/// Pure S=2..=4 counterpart of [`linear_attention_forward`].
///
/// This deliberately calls the same projection, fused conv/QK-normalization,
/// gated-delta, portable gate, and output-projection helpers as the ordinary
/// relaxed target verifier. It only replaces cache mutation with explicit
/// state inputs/outputs, making the graph legal to enclose in `mlx_compile`.
pub(crate) fn linear_attention_forward_verify_functional(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    layer_idx: usize,
    conv_state: &MlxArray,
    recurrent_state: &MlxArray,
) -> Option<LinearAttentionVerifyOutput> {
    let linear_cfg = cfg.linear_attention.as_ref()?;
    let linear_w = w.linear_attn.as_ref()?;
    let seq = x.shape().get(1).copied()?;
    if !fastpath::qwen_linear_mtp_verify_seq_contains(seq as i64) {
        return None;
    }

    let (qkv, z, a, b) = linear_attention_inputs(cfg, linear_cfg, linear_w, x, seq, false);
    let (q, k, v, new_conv_state, _) =
        linear_attention_post_input(cfg, linear_cfg, linear_w, &qkv, Some(conv_state), false);
    let (out, new_recurrent_state, tape) = gated_delta_kernel_with_tape(
        &q,
        &k,
        &v,
        &linear_w.a_log,
        &a,
        &linear_w.dt_bias,
        &b,
        recurrent_state,
    )?;
    let value_dim = linear_cfg.value_dim() as i32;
    let output = if let Some(fused) =
        try_qwen_la_out_proj_silu_mul_qmm(cfg, &out, &z, linear_w, seq, value_dim, layer_idx)
    {
        fused
    } else {
        let gated = rms_norm_gated_with_full_gate_policy(
            &out,
            &z,
            &linear_w.norm,
            cfg.rms_norm_eps,
            if fastpath::qwen_linear_mtp_exact_enabled() {
                false
            } else {
                linear_attention_full_gate_metal_allowed(cfg, linear_w, layer_idx)
            },
        );
        let flat = reshape(&gated, &[1, seq, value_dim], None);
        qw(&flat, &linear_w.out_proj)
    };

    Some(LinearAttentionVerifyOutput {
        output,
        conv_state: new_conv_state,
        recurrent_state: new_recurrent_state,
        qkv,
        a,
        tape,
    })
}

pub(crate) fn linear_attention_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    skip_out_proj: bool,
    last_token_out_proj: bool,
) -> MlxArray {
    linear_attention_forward_inner(
        cfg,
        w,
        x,
        cache,
        layer_idx,
        skip_out_proj,
        last_token_out_proj,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn linear_attention_forward_inner(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    skip_out_proj: bool,
    last_token_out_proj: bool,
    stop_before_out_proj: bool,
    stop_before_gate: bool,
) -> MlxArray {
    let linear_cfg = cfg
        .linear_attention
        .as_ref()
        .expect("linear attention layer requires linear_attention config");
    let linear_w = w
        .linear_attn
        .as_ref()
        .expect("linear attention layer requires linear attention weights");
    let seq = x.shape()[1];

    // Try whole-layer Metal kernel for decode (single-token step).
    // Falls back to the standard multi-dispatch path on failure or when
    // the fastpath flag is disabled.
    if seq == 1
        && let Some(out) = try_linear_attention_whole_layer_metal(cfg, w, x, cache, layer_idx)
    {
        return out;
    }

    let profile_enabled = linear_attention_profile_enabled();
    if profile_enabled {
        record_linear_attention_profile_layer(seq);
    }

    let profile_started = Instant::now();
    let (qkv, z, a, b) =
        linear_attention_inputs(cfg, linear_cfg, linear_w, x, seq, profile_enabled);
    if cache.linear_prefix_capture_after().is_some() {
        // oMLX's Qwen verifier avoids a partial-accept backbone replay by
        // retaining these already-computed projections and replaying only the
        // gated-delta conv/recurrent update. Keep the stash tied to the same
        // transient capture lifetime as AX's existing prefix checkpoint.
        cache.set_linear_mtp_projection_stash(layer_idx, qkv.clone(), a.clone(), b.clone());
    }
    qwen_prefill_maybe_async_la_outputs(&qkv, &z, &a, &b, seq);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Projection,
        profile_started,
        &[&qkv, &z, &a, &b],
    );

    let (conv_state, recurrent_state) = cache.linear_state(layer_idx);
    let prefix_capture_after = cache
        .linear_prefix_capture_after()
        .filter(|after| *after < seq as usize);
    // `a_log` and `dt_bias` are pre-cast to f32 at weight-load time (see
    // `load_linear_attention_weights` in `weights.rs`). mlx_lm preserves A_log
    // as float32 and computes g in float32 precision; doing the cast per
    // forward-pass-per-layer was ~24 wasted astype dispatches per decode step
    // on a 12-layer hybrid model.
    let profile_started = Instant::now();
    let a_log_f32 = linear_w.a_log.clone();
    let dt_bias_f32 = linear_w.dt_bias.clone();
    // State is always float32: mlx_lm initialises state as mx.zeros(..., dtype=mx.float32).
    let state = recurrent_state
        .cloned()
        .unwrap_or_else(|| initial_recurrent_state_zeros(linear_cfg));
    // The short relaxed verifier can keep conv/QK intermediates in one Metal
    // dispatch. Skip-checkpoint uses a lockstep fused kernel without the
    // row-0 recurrent/conv-prefix buffers; complete-miss then keep=1
    // projected-replays. Tape capture still excludes fusion. Unsupported
    // shapes fall through to the two-kernel composition below.
    let fused_eligible = prefix_capture_after == Some(1)
        && !profile_enabled
        && fastpath::qwen_linear_mtp_target_verify_enabled()
        && fastpath::mtp_fused_gated_delta_verify_enabled()
        && !fastpath::mtp_linear_tape_capture_enabled();
    let skip_prefix_checkpoint = fastpath::mtp_skip_prefix_checkpoint_enabled();
    let fused_verify = fused_eligible
        .then(|| {
            if skip_prefix_checkpoint {
                gated_delta_fused_verify_no_checkpoint_from_qkv(
                    linear_cfg,
                    &qkv,
                    &linear_w.conv1d_dense,
                    conv_state,
                    &a_log_f32,
                    &a,
                    &dt_bias_f32,
                    &b,
                    &state,
                    linear_cfg.q_scale,
                    linear_cfg.k_scale,
                    cfg.rms_norm_eps,
                )
                .map(|(out, new_state, new_conv)| (out, new_state, None, new_conv, None))
            } else {
                gated_delta_fused_verify_from_qkv(
                    linear_cfg,
                    &qkv,
                    &linear_w.conv1d_dense,
                    conv_state,
                    &a_log_f32,
                    &a,
                    &dt_bias_f32,
                    &b,
                    &state,
                    linear_cfg.q_scale,
                    linear_cfg.k_scale,
                    cfg.rms_norm_eps,
                )
                .map(|(out, new_state, prefix_state, new_conv, prefix_conv)| {
                    (
                        out,
                        new_state,
                        Some(prefix_state),
                        new_conv,
                        Some(prefix_conv),
                    )
                })
            }
        })
        .flatten();

    // g and beta are computed inside the Metal kernels instead of as separate
    // lazy MLX ops, eliminating ~8 dispatches per layer.
    let (
        out,
        new_conv_state,
        new_recurrent_state,
        prefix_conv_state,
        prefix_recurrent_state,
        mtp_tape,
    ) = if let Some((out, new_state, prefix_state, new_conv, prefix_conv)) = fused_verify {
        (out, new_conv, new_state, prefix_conv, prefix_state, None)
    } else {
        let (q, k, v, new_conv_state, metal_prefix_conv) = linear_attention_post_input(
            cfg,
            linear_cfg,
            linear_w,
            &qkv,
            conv_state,
            profile_enabled,
        );
        let prefix_conv_state = match (prefix_capture_after, metal_prefix_conv) {
            (Some(1), Some(prefix)) => Some(prefix),
            (Some(after), _) => {
                linear_attention_conv_prefix_state(linear_cfg, &qkv, conv_state, after)
            }
            (None, _) => None,
        };
        let mut mtp_tape = None;
        let (out, new_recurrent_state, prefix_recurrent_state) = if prefix_capture_after.is_some()
            && fastpath::qwen_linear_mtp_target_verify_enabled()
            && fastpath::mtp_skip_prefix_checkpoint_enabled()
        {
            let (out, new_state) = gated_delta_kernel_verify_no_checkpoint(
                &q,
                &k,
                &v,
                &a_log_f32,
                &a,
                &dt_bias_f32,
                &b,
                &state,
            )
            .unwrap_or_else(|| {
                gated_delta_kernel(&q, &k, &v, &a_log_f32, &a, &dt_bias_f32, &b, &state)
            });
            (out, new_state, None)
        } else if prefix_capture_after.is_some()
            && fastpath::qwen_linear_mtp_target_verify_enabled()
            && fastpath::mtp_linear_tape_capture_enabled()
            && let Some((out, new_state, tape)) =
                gated_delta_kernel_with_tape(&q, &k, &v, &a_log_f32, &a, &dt_bias_f32, &b, &state)
        {
            mtp_tape = Some(tape);
            (out, new_state, None)
        } else if let Some(after) = prefix_capture_after {
            let (out, new_state, prefix_state) = gated_delta_kernel_with_prefix_checkpoint(
                &q,
                &k,
                &v,
                &a_log_f32,
                &a,
                &dt_bias_f32,
                &b,
                &state,
                after,
            );
            (out, new_state, Some(prefix_state))
        } else {
            let (out, new_state) =
                gated_delta_kernel(&q, &k, &v, &a_log_f32, &a, &dt_bias_f32, &b, &state);
            (out, new_state, None)
        };
        if prefix_capture_after.is_some()
            && fastpath::qwen_linear_mtp_target_verify_enabled()
            && fastpath::mtp_reuse_processed_gdn_enabled()
        {
            cache.set_linear_mtp_processed_stash(layer_idx, q.clone(), k.clone(), v.clone());
        }
        (
            out,
            new_conv_state,
            new_recurrent_state,
            prefix_conv_state,
            prefix_recurrent_state,
            mtp_tape,
        )
    };
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Recurrent,
        profile_started,
        &[&out, &new_recurrent_state],
    );
    cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);
    if let Some(tape) = mtp_tape {
        cache.set_linear_mtp_tape_stash(layer_idx, qkv.clone(), a.clone(), tape);
    }
    if let (Some(conv_state), Some(recurrent_state)) = (prefix_conv_state, prefix_recurrent_state) {
        cache.set_linear_prefix_checkpoint(layer_idx, conv_state, recurrent_state);
    }
    qwen_prefill_maybe_async_gd(&out, seq);
    qwen_prefill_maybe_eval_gd(&out, seq);
    let out = qwen_prefill_maybe_contiguous_gd(out, seq);
    if let Some(skipped) = qwen_prefill_maybe_skip_unused_la_out(x, skip_out_proj) {
        return skipped;
    }

    let profile_started = Instant::now();
    let value_dim = linear_cfg.value_dim() as i32;
    let (out, z, seq) = match qwen_prefill_maybe_last_token_la_out(&out, &z, last_token_out_proj) {
        Some((out, z, seq)) => (out, z, seq),
        None => (out, z, seq),
    };
    if stop_before_gate {
        LA_PRE_GATE_Z.with(|slot| {
            *slot.borrow_mut() = Some(z);
        });
        return out;
    }
    let out = if let Some(fused) =
        try_qwen_la_out_proj_silu_mul_qmm(cfg, &out, &z, linear_w, seq, value_dim, layer_idx)
    {
        fused
    } else {
        let out = rms_norm_gated_with_full_gate_policy(
            &out,
            &z,
            &linear_w.norm,
            cfg.rms_norm_eps,
            if fastpath::qwen_linear_mtp_exact_enabled() {
                // Exact S=2 fused Metal on early layers (`dced27d4`) kept
                // MTP-off `39a36e3f` but ON became `f4b5490d`. Stay portable.
                false
            } else {
                linear_attention_full_gate_metal_allowed(cfg, linear_w, layer_idx)
            },
        );
        let flat = if fastpath::should_skip_unused_la_out_reshape(&out.shape(), seq, value_dim) {
            out
        } else {
            reshape(&out, &[1, seq, value_dim], None)
        };
        if stop_before_out_proj {
            flat
        } else {
            qw(&flat, &linear_w.out_proj)
        }
    };
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Output,
        profile_started,
        &[&out],
    );
    out
}

fn mtp_projection_prefix(projected: &MlxArray, keep: i32) -> Option<MlxArray> {
    let shape = projected.shape();
    if shape.len() < 2 || shape[0] != 1 || keep <= 0 || keep > shape[1] {
        return None;
    }
    let starts = vec![0_i32; shape.len()];
    let mut ends = shape;
    ends[1] = keep;
    let strides = vec![1_i32; ends.len()];
    Some(slice(projected, &starts, &ends, &strides, None))
}

/// Rebuild only one Qwen gated-delta layer's conv/recurrent state after an
/// MTP partial accept.
///
/// The verify forward already paid for the QKV/A/B projections and retained
/// them in `verify_cache`. Replaying their accepted prefix from the unchanged
/// pre-verify state is equivalent to oMLX 0.6.2's `mtp_partial_rollback`: full
/// attention layers can trim their KV window, while linear layers avoid a
/// second transformer-backbone forward.
pub(crate) fn replay_linear_attention_mtp_prefix(
    cfg: &ModelConfig,
    w: &LayerWeights,
    source_cache: &MlxKVCache,
    verify_cache: &mut MlxKVCache,
    layer_idx: usize,
    keep: usize,
) -> bool {
    let Some(linear_cfg) = cfg.linear_attention.as_ref() else {
        return false;
    };
    let Some(linear_w) = w.linear_attn.as_ref() else {
        return false;
    };
    let Ok(keep) = i32::try_from(keep) else {
        return false;
    };
    if let Some((qkv, a, tape)) = verify_cache.linear_mtp_tape_stash(layer_idx) {
        let Some(qkv) = mtp_projection_prefix(&qkv, keep) else {
            return false;
        };
        let Some(a_prefix) = mtp_projection_prefix(&a, keep) else {
            return false;
        };
        let Some(tape_prefix) = mtp_projection_prefix(&tape, keep) else {
            return false;
        };
        let (source_conv, source_recurrent) = source_cache.linear_state(layer_idx);
        let (_, k, _, new_conv_state, _) =
            linear_attention_post_input(cfg, linear_cfg, linear_w, &qkv, source_conv, false);
        let recurrent_state = source_recurrent
            .cloned()
            .unwrap_or_else(|| initial_recurrent_state_zeros(linear_cfg));
        let Some(new_recurrent_state) = replay_gated_delta_tape(
            &k,
            &linear_w.a_log,
            &a_prefix,
            &linear_w.dt_bias,
            &tape_prefix,
            &recurrent_state,
            keep as usize,
        ) else {
            return false;
        };
        verify_cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);
        return true;
    }

    let Some((qkv, a, b)) = verify_cache.linear_mtp_projection_stash(layer_idx) else {
        return false;
    };
    let processed = fastpath::mtp_reuse_processed_gdn_enabled()
        .then(|| verify_cache.linear_mtp_processed_stash(layer_idx))
        .flatten();
    let (source_conv, source_recurrent) = source_cache.linear_state(layer_idx);
    let reused_conv_state = processed.as_ref().and_then(|_| {
        linear_attention_conv_prefix_state(linear_cfg, &qkv, source_conv, keep as usize)
    });
    let Some(qkv) = mtp_projection_prefix(&qkv, keep) else {
        return false;
    };
    let Some(a) = mtp_projection_prefix(&a, keep) else {
        return false;
    };
    let Some(b) = mtp_projection_prefix(&b, keep) else {
        return false;
    };

    if let (Some((q, k, v)), Some(new_conv_state)) = (processed, reused_conv_state)
        && let (Some(q), Some(k), Some(v)) = (
            mtp_projection_prefix(&q, keep),
            mtp_projection_prefix(&k, keep),
            mtp_projection_prefix(&v, keep),
        )
    {
        let recurrent_state = source_recurrent
            .cloned()
            .unwrap_or_else(|| initial_recurrent_state_zeros(linear_cfg));
        let (_, new_recurrent_state) = gated_delta_kernel(
            &q,
            &k,
            &v,
            &linear_w.a_log,
            &a,
            &linear_w.dt_bias,
            &b,
            &recurrent_state,
        );
        verify_cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);
        return true;
    }

    let (q, k, v, new_conv_state, _) =
        linear_attention_post_input(cfg, linear_cfg, linear_w, &qkv, source_conv, false);
    let recurrent_state = source_recurrent
        .cloned()
        .unwrap_or_else(|| initial_recurrent_state_zeros(linear_cfg));
    let (_, new_recurrent_state) = gated_delta_kernel(
        &q,
        &k,
        &v,
        &linear_w.a_log,
        &a,
        &linear_w.dt_bias,
        &b,
        &recurrent_state,
    );
    verify_cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);
    true
}

/// Exact S=2..=4: run the MTP-off S=1 Metal gate + S=1 o_proj per row.
///
/// Factory trial-2 `0c6b1484` kept `39a36e3f`, but `--full` regressed
/// general-long 1.038 → 1.010 (two S=1 Metal+o_proj is slower than fused
/// portable S=2). Unhooked from the verify path.
#[cfg_attr(not(test), allow(dead_code))]
fn exact_verify_s1_metal_gate_o_proj(
    hidden: &MlxArray,
    gate: &MlxArray,
    linear_w: &LinearAttentionWeights,
    eps: f32,
    value_dim: i32,
    seq: i32,
    allow_full_gate_metal: bool,
) -> Option<MlxArray> {
    if !fastpath::qwen_linear_mtp_exact_enabled()
        || !fastpath::qwen_linear_mtp_verify_seq_contains(seq as i64)
    {
        return None;
    }
    if hidden.shape().len() != 4 || gate.shape().len() != 4 {
        return None;
    }
    let _mtp_off_gate = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let mut rows: Vec<MlxArray> = Vec::with_capacity(seq as usize);
    for t in 0..seq {
        let h = slice_seq_row_4d(hidden, t);
        let g = slice_seq_row_4d(gate, t);
        let gated = rms_norm_gated_with_full_gate_policy(
            &h,
            &g,
            &linear_w.norm,
            eps,
            allow_full_gate_metal,
        );
        let flat = reshape(&gated, &[1, 1, value_dim], None);
        rows.push(qw(&flat, &linear_w.out_proj));
    }
    let refs: Vec<&MlxArray> = rows.iter().collect();
    Some(concatenate(&refs, 1, None))
}

#[allow(clippy::too_many_arguments)]
fn try_qwen_la_out_proj_silu_mul_qmm(
    cfg: &ModelConfig,
    hidden: &MlxArray,
    gate: &MlxArray,
    linear_w: &LinearAttentionWeights,
    seq: i32,
    value_dim: i32,
    layer_idx: usize,
) -> Option<MlxArray> {
    if !fastpath::should_qwen_la_out_proj_silu_mul_qmm(&cfg.model_family, seq) {
        return None;
    }
    qwen_la_gated_out_projection(
        hidden,
        gate,
        &linear_w.norm,
        &linear_w.out_proj,
        cfg.rms_norm_eps,
        seq,
        value_dim,
        !fastpath::qwen_linear_mtp_exact_enabled()
            && linear_attention_full_gate_metal_allowed(cfg, linear_w, layer_idx),
    )
}

#[allow(clippy::too_many_arguments)]
fn qwen_la_gated_out_projection(
    hidden: &MlxArray,
    gate: &MlxArray,
    norm: &MlxArray,
    out_proj: &QuantizedWeight,
    eps: f32,
    seq: i32,
    value_dim: i32,
    allow_full_gate_metal: bool,
) -> Option<MlxArray> {
    if !out_proj.is_fused_qmm_quantized() {
        return None;
    }
    // Share both dtype boundaries and layer-specific Metal admission with
    // ordinary target execution; a portable-only gate still changes rounding.
    let gated =
        rms_norm_gated_with_full_gate_policy(hidden, gate, norm, eps, allow_full_gate_metal);
    Some(qw(&reshape(&gated, &[1, seq, value_dim], None), out_proj))
}

fn linear_attention_conv_prefix_state(
    cfg: &LinearAttentionConfig,
    qkv: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    after_tokens: usize,
) -> Option<MlxArray> {
    let cached = cached_conv_state?;
    let shape = qkv.shape();
    if shape.len() != 3 || after_tokens == 0 || after_tokens >= shape[1] as usize {
        return None;
    }
    let batch = shape[0];
    let conv_dim = cfg.conv_dim() as i32;
    let tail_len = cfg.conv_kernel_dim as i32 - 1;
    if shape[2] != conv_dim || cached.shape() != vec![batch, tail_len, conv_dim] {
        return None;
    }
    if tail_len == 0 {
        return Some(zeros(&[batch, 0, conv_dim], qkv.dtype(), None));
    }
    let after = after_tokens as i32;
    let prefix = slice(qkv, &[0, 0, 0], &[batch, after, conv_dim], &[1, 1, 1], None);
    let combined = concatenate(&[cached, &prefix], 1, None);
    let total = tail_len + after;
    Some(slice(
        &combined,
        &[0, total - tail_len, 0],
        &[batch, total, conv_dim],
        &[1, 1, 1],
        None,
    ))
}

/// Batched (leading dim `B`) linear-attention decode forward — Phase 3.7.
///
/// Mirrors [`linear_attention_forward`] for a `[B, 1, hidden]` cohort, reading
/// and writing per-row conv1d + recurrent state from a [`BatchedLinearState`]
/// instead of the single-request [`MlxKVCache`]. `x` is the already
/// attn-normed input (the caller applies `attn_norm`, exactly as the single-row
/// path splits that step into the layer shell).
///
/// Correctness contract (oracle-tested): **row `r` of the output is
/// byte-identical to a single-sequence decode of row `r`** through the portable
/// composition. This path deliberately uses the portable projection + conv1d +
/// qk-norm ops (all batch-general: they derive `batch` from `shape[0]`) rather
/// than the batch=1-shaped Metal/direct-C++ decode fast paths, and the portable
/// gated RMSNorm. The gated-delta recurrent kernel is already batch-native
/// (dispatches over `batch * num_value_heads`, state `[B, Hv, Dv, Dk]`), so it
/// is shared with the single-row path unchanged. Decode-only (`seq == 1`).
pub(crate) fn linear_attention_forward_batched(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    lin_state: &mut BatchedLinearState,
    linear_state_idx: usize,
) -> MlxArray {
    let linear_cfg = cfg
        .linear_attention
        .as_ref()
        .expect("linear attention layer requires linear_attention config");
    let linear_w = w
        .linear_attn
        .as_ref()
        .expect("linear attention layer requires linear attention weights");
    let batch = x.shape()[0];
    let seq = x.shape()[1];
    debug_assert_eq!(seq, 1, "batched linear-attention forward is decode-only");

    let (qkv, z, a, b) = linear_attention_inputs_batched(linear_cfg, linear_w, x, batch, seq);

    // Snapshot this layer's current per-row state (cloned so the store can be
    // reborrowed mutably for the write-back below). `None` on the first step.
    let (conv_state, recurrent_state) = match lin_state.layer_state(linear_state_idx) {
        Some((conv, rec)) => (Some(conv.clone()), Some(rec.clone())),
        None => (None, None),
    };
    let (q, k, v, new_conv_state) = linear_attention_post_input_batched(
        linear_cfg,
        linear_w,
        &qkv,
        conv_state.as_ref(),
        cfg.rms_norm_eps,
    );

    // `a_log` / `dt_bias` are pre-cast to f32 at load time (see single-row path).
    let a_log_f32 = linear_w.a_log.clone();
    let dt_bias_f32 = linear_w.dt_bias.clone();
    let state = recurrent_state.unwrap_or_else(|| {
        zeros(
            &[
                batch,
                linear_cfg.num_value_heads as i32,
                linear_cfg.value_head_dim as i32,
                linear_cfg.key_head_dim as i32,
            ],
            MlxDtype::Float32,
            None,
        )
    });
    let (out, new_recurrent_state) =
        gated_delta_kernel(&q, &k, &v, &a_log_f32, &a, &dt_bias_f32, &b, &state);
    lin_state.update_layer(linear_state_idx, new_conv_state, new_recurrent_state);

    // Portable gated RMSNorm (allow_full_gate_metal = false): batch-general.
    let out =
        rms_norm_gated_with_full_gate_policy(&out, &z, &linear_w.norm, cfg.rms_norm_eps, false);
    let flat = reshape(&out, &[batch, seq, linear_cfg.value_dim() as i32], None);
    qw(&flat, &linear_w.out_proj)
}

/// Batched projection stage for [`linear_attention_forward_batched`] — the
/// portable mirror of [`linear_attention_inputs`]'s composition, with the
/// leading dim parameterised to `batch` instead of hardcoded `1`. Skips the
/// direct-C++ packed shim (whose shape filter assumes batch=1) so the graph is
/// provably batch-general.
fn linear_attention_inputs_batched(
    cfg: &LinearAttentionConfig,
    w: &LinearAttentionWeights,
    x: &MlxArray,
    batch: i32,
    seq: i32,
) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
    if let (Some(qkvz_w), Some(ba_w)) = (&w.in_proj_qkvz, &w.in_proj_ba) {
        let mixed_qkvz = qw(x, qkvz_w);
        let value_heads_per_key = cfg.num_value_heads / cfg.num_key_heads;
        let value_dim_per_key = value_heads_per_key * cfg.value_head_dim;
        let qkvz_per_key = cfg.key_head_dim * 2 + value_dim_per_key * 2;
        let mixed_qkvz = reshape(
            &mixed_qkvz,
            &[batch, seq, cfg.num_key_heads as i32, qkvz_per_key as i32],
            None,
        );
        let q = slice_last_dim(&mixed_qkvz, 0, cfg.key_head_dim as i32, None);
        let k = slice_last_dim(
            &mixed_qkvz,
            cfg.key_head_dim as i32,
            (cfg.key_head_dim * 2) as i32,
            None,
        );
        let v = slice_last_dim(
            &mixed_qkvz,
            (cfg.key_head_dim * 2) as i32,
            (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
            None,
        );
        let z = slice_last_dim(
            &mixed_qkvz,
            (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
            qkvz_per_key as i32,
            None,
        );
        let qkv = concatenate(
            &[
                &reshape(&q, &[batch, seq, cfg.key_dim() as i32], None),
                &reshape(&k, &[batch, seq, cfg.key_dim() as i32], None),
                &reshape(&v, &[batch, seq, cfg.value_dim() as i32], None),
            ],
            2,
            None,
        );
        let z = reshape(
            &z,
            &[
                batch,
                seq,
                cfg.num_value_heads as i32,
                cfg.value_head_dim as i32,
            ],
            None,
        );
        let mixed_ba = qw(x, ba_w);
        let ba = reshape(
            &mixed_ba,
            &[
                batch,
                seq,
                cfg.num_key_heads as i32,
                (value_heads_per_key * 2) as i32,
            ],
            None,
        );
        let b = reshape(
            &slice_last_dim(&ba, 0, value_heads_per_key as i32, None),
            &[batch, seq, cfg.num_value_heads as i32],
            None,
        );
        let a = reshape(
            &slice_last_dim(
                &ba,
                value_heads_per_key as i32,
                (value_heads_per_key * 2) as i32,
                None,
            ),
            &[batch, seq, cfg.num_value_heads as i32],
            None,
        );
        return (qkv, z, a, b);
    }

    // Split (non-packed) projections — same portable ops, batch leading dim.
    let qkv = qw(
        x,
        w.in_proj_qkv
            .as_ref()
            .expect("split linear attention must have qkv projection"),
    );
    let z = reshape(
        &qw(
            x,
            w.in_proj_z
                .as_ref()
                .expect("split linear attention must have z projection"),
        ),
        &[
            batch,
            seq,
            cfg.num_value_heads as i32,
            cfg.value_head_dim as i32,
        ],
        None,
    );
    let a = qw(
        x,
        w.in_proj_a
            .as_ref()
            .expect("split linear attention must have a projection"),
    );
    let b = qw(
        x,
        w.in_proj_b
            .as_ref()
            .expect("split linear attention must have b projection"),
    );
    (qkv, z, a, b)
}

/// Batched conv1d + split + qk-norm — the portable branch of
/// [`linear_attention_post_input`], which is already batch-general (the conv1d,
/// split and normalize helpers derive `batch` from `shape[0]`).
fn linear_attention_post_input_batched(
    cfg: &LinearAttentionConfig,
    w: &LinearAttentionWeights,
    qkv: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    eps: f32,
) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
    let (conv_out, new_conv_state) =
        linear_attention_conv1d(cfg, qkv, &w.conv1d_dense, cached_conv_state);
    let split = split_linear_attention_qkv(cfg, &conv_out);
    let (q, k) = normalize_linear_attention_qk(cfg, &split.q, &split.k, eps);
    (q, k, split.v, new_conv_state)
}

/// Run the linear-attention post-input chain (conv1d + SiLU + split + per-head
/// reshape + qk RMSNorm + scale) as either:
/// (a) one direct-C++ FFI round-trip via `qwen_linear_attention_post_input` when
///     the env flag is set AND per-layer linear-attention profiling is off, or
/// (b) the portable Rust composition that mirrors mlx_lm's reference.
///
/// `profile_enabled` blocks the shim because the shim does not surface
/// `LinearAttentionProfileStage::Conv` / `QkNorm` per-stage eval barriers; the
/// portable path is preserved exactly so profiling-driven decode artifacts
/// remain fair against any future Rust-side optimisation.
fn linear_attention_post_input(
    cfg: &ModelConfig,
    linear_cfg: &LinearAttentionConfig,
    linear_w: &crate::weights::LinearAttentionWeights,
    qkv: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    profile_enabled: bool,
) -> (MlxArray, MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    let qwen_default_enabled = qwen_linear_attention_direct_cpp_default_family(cfg)
        && fastpath::qwen_direct_cpp_linear_attention_post_input_enabled();
    let seq = qkv.shape().get(1).copied().unwrap_or_default();
    let qkv_storage = if fastpath::should_qwen_la_contiguous_qkv(seq) {
        Some(contiguous(qkv, None))
    } else {
        None
    };
    let qkv = qkv_storage.as_ref().unwrap_or(qkv);
    let speculative_multi_token = fastpath::qwen_linear_mtp_verify_seq_contains(seq as i64)
        && fastpath::qwen_linear_mtp_verify_fast_kernels_enabled();
    let prefill_metal = seq > 1
        && seq <= crate::linear_attention_ops::GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32
        && fastpath::qwen_linear_attention_prefill_post_input_metal_enabled();
    if (seq == 1 || speculative_multi_token)
        && fastpath::qwen_linear_attention_decode_post_input_metal_enabled()
        || prefill_metal
    {
        record_linear_attention_decode_post_input_metal_attempt();
        if profile_enabled {
            record_linear_attention_decode_post_input_metal_profile_blocked();
            record_linear_attention_decode_post_input_metal_fallback();
        } else if let Some((q, k, v, new_state, prefix_conv)) =
            linear_attention_decode_post_input_metal(
                linear_cfg,
                qkv,
                &linear_w.conv1d_dense,
                cached_conv_state,
                linear_cfg.q_scale,
                linear_cfg.k_scale,
                cfg.rms_norm_eps,
            )
        {
            record_linear_attention_decode_post_input_metal_hit();
            return (q, k, v, new_state, Some(prefix_conv));
        } else {
            record_linear_attention_decode_post_input_metal_fallback();
        }
    }
    if fastpath::direct_cpp_linear_attention_post_input_enabled() || qwen_default_enabled {
        record_linear_attention_direct_cpp_post_input_attempt();
        if profile_enabled {
            record_linear_attention_direct_cpp_post_input_profile_blocked();
            record_linear_attention_direct_cpp_post_input_fallback();
        } else if fastpath::should_qwen_la_post_input_compile(seq)
            && let Some(state) = cached_conv_state.cloned().or_else(|| {
                let batch = qkv.shape().first().copied().unwrap_or(1);
                let tail = (linear_cfg.conv_kernel_dim as i32 - 1).max(0);
                Some(zeros(
                    &[batch, tail, linear_cfg.conv_dim() as i32],
                    qkv.dtype(),
                    None,
                ))
            })
            && let Some(outputs) = qwen_linear_attention_post_input_compiled(
                qkv,
                &linear_w.conv1d_dense,
                &state,
                linear_cfg.num_key_heads as i32,
                linear_cfg.key_head_dim as i32,
                linear_cfg.num_value_heads as i32,
                linear_cfg.value_head_dim as i32,
                linear_cfg.conv_kernel_dim as i32,
                linear_cfg.q_scale,
                linear_cfg.k_scale,
                cfg.rms_norm_eps,
                None,
            )
        {
            record_linear_attention_direct_cpp_post_input_hit();
            return (outputs.0, outputs.1, outputs.2, outputs.3, None);
        } else if let Some(outputs) = qwen_linear_attention_post_input(
            qkv,
            &linear_w.conv1d_dense,
            cached_conv_state,
            linear_cfg.num_key_heads as i32,
            linear_cfg.key_head_dim as i32,
            linear_cfg.num_value_heads as i32,
            linear_cfg.value_head_dim as i32,
            linear_cfg.conv_kernel_dim as i32,
            linear_cfg.q_scale,
            linear_cfg.k_scale,
            cfg.rms_norm_eps,
            None,
        ) {
            record_linear_attention_direct_cpp_post_input_hit();
            return (outputs.0, outputs.1, outputs.2, outputs.3, None);
        } else {
            record_linear_attention_direct_cpp_post_input_fallback();
        }
    }

    // Portable composition — exact mirror of the C++ shim, used when the flag
    // is off, when profiling is on, or when the shim rejected the shapes.
    let profile_started = Instant::now();
    let (conv_out, new_conv_state) =
        linear_attention_conv1d(linear_cfg, qkv, &linear_w.conv1d_dense, cached_conv_state);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Conv,
        profile_started,
        &[&conv_out, &new_conv_state],
    );
    let split = split_linear_attention_qkv(linear_cfg, &conv_out);
    let profile_started = Instant::now();
    let (q, k) = normalize_linear_attention_qk(linear_cfg, &split.q, &split.k, cfg.rms_norm_eps);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::QkNorm,
        profile_started,
        &[&q, &k],
    );
    (q, k, split.v, new_conv_state, None)
}

/// Submit packed LA projections before conv/GatedDelta is attached.
fn qwen_prefill_maybe_async_la_outputs(
    qkv: &MlxArray,
    z: &MlxArray,
    a: &MlxArray,
    b: &MlxArray,
    seq: i32,
) {
    qwen_prefill_maybe_async_la_outputs_for(
        qkv,
        z,
        a,
        b,
        fastpath::qwen_prefill_async_la_outputs_enabled(),
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_async_la_outputs`].
pub(crate) fn qwen_prefill_maybe_async_la_outputs_for(
    qkv: &MlxArray,
    z: &MlxArray,
    a: &MlxArray,
    b: &MlxArray,
    enabled: bool,
    seq: i32,
) {
    if fastpath::should_qwen_prefill_async_la_outputs_for(enabled, seq) {
        mlx_sys::async_eval(&[qkv, z, a, b]);
    }
}

/// Pack GatedDelta output so rms_norm_gated + out_proj see a contiguous view.
fn qwen_prefill_maybe_contiguous_gd(gd_out: MlxArray, seq: i32) -> MlxArray {
    qwen_prefill_maybe_contiguous_gd_for(
        gd_out,
        fastpath::qwen_prefill_contiguous_gd_enabled(),
        seq,
    )
}

/// Pure helper for [`qwen_prefill_maybe_contiguous_gd`].
pub(crate) fn qwen_prefill_maybe_contiguous_gd_for(
    gd_out: MlxArray,
    enabled: bool,
    seq: i32,
) -> MlxArray {
    if fastpath::should_qwen_prefill_contiguous_gd_for(enabled, seq) {
        contiguous(&gd_out, None)
    } else {
        gd_out
    }
}

/// Materialize GatedDelta output once before rms_norm_gated + out_proj.
fn qwen_prefill_maybe_eval_gd(gd_out: &MlxArray, seq: i32) {
    qwen_prefill_maybe_eval_gd_for(gd_out, fastpath::qwen_prefill_eval_gd_enabled(), seq);
}

/// Pure helper for [`qwen_prefill_maybe_eval_gd`].
pub(crate) fn qwen_prefill_maybe_eval_gd_for(gd_out: &MlxArray, enabled: bool, seq: i32) {
    if fastpath::should_qwen_prefill_eval_gd_for(enabled, seq) {
        eval(&[gd_out]);
    }
}

/// Submit GatedDelta output before rms_norm_gated + out_proj is encoded.
fn qwen_prefill_maybe_async_gd(gd_out: &MlxArray, seq: i32) {
    qwen_prefill_maybe_async_gd_for(gd_out, fastpath::qwen_prefill_async_gd_enabled(), seq);
}

/// Pure helper for [`qwen_prefill_maybe_async_gd`].
pub(crate) fn qwen_prefill_maybe_async_gd_for(gd_out: &MlxArray, enabled: bool, seq: i32) {
    if fastpath::should_qwen_prefill_async_gd_for(enabled, seq) {
        async_eval(&[gd_out]);
    }
}

/// After conv/recurrent state is written, skip unused LA out_proj.
pub(crate) fn qwen_prefill_maybe_skip_unused_la_out(
    x: &MlxArray,
    skip_out_proj: bool,
) -> Option<MlxArray> {
    skip_out_proj.then(|| x.clone())
}

/// After conv/recurrent state is written, slice LA output + gate to the last
/// token so last-only generate prefill runs rms_norm_gated + out_proj at S=1.
pub(crate) fn qwen_prefill_maybe_last_token_la_out(
    out: &MlxArray,
    z: &MlxArray,
    last_token_out_proj: bool,
) -> Option<(MlxArray, MlxArray, i32)> {
    if !last_token_out_proj {
        return None;
    }
    let seq = out.shape().get(1).copied().unwrap_or(1);
    if seq <= 1 {
        return None;
    }
    Some((slice_seq_axis1(out), slice_seq_axis1(z), 1))
}

fn slice_seq_axis1(x: &MlxArray) -> MlxArray {
    let shape = x.shape();
    let last = shape[1] - 1;
    let mut start = vec![0i32; shape.len()];
    let mut stop = shape.clone();
    start[1] = last;
    stop[1] = last + 1;
    let strides = vec![1i32; shape.len()];
    slice(x, &start, &stop, &strides, None)
}

/// Cache a contiguous overlay of one LA quantized projection.
pub(crate) fn cached_prefill_la_contiguous_weight(src: &QuantizedWeight) -> QuantizedWeight {
    let key = src as *const QuantizedWeight as usize;
    PREFILL_LA_CONTIG_W.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return existing.clone();
        }
        let made = crate::weights::contiguous_affine_weight(src);
        cache.borrow_mut().insert(key, made.clone());
        made
    })
}

/// Materialize the Qwen linear-attention activation once before QKVZ/BA qmm.
fn qwen_prefill_maybe_eval_la_input(x: &MlxArray, seq: i32) {
    qwen_prefill_maybe_eval_la_input_for(x, fastpath::qwen_prefill_eval_la_input_enabled(), seq);
}

/// Pure helper for [`qwen_prefill_maybe_eval_la_input`].
pub(crate) fn qwen_prefill_maybe_eval_la_input_for(x: &MlxArray, enabled: bool, seq: i32) {
    if fastpath::should_qwen_prefill_eval_la_input_for(enabled, seq) {
        mlx_sys::eval(&[x]);
    }
}

pub(crate) fn linear_attention_inputs(
    model_cfg: &ModelConfig,
    cfg: &LinearAttentionConfig,
    w: &crate::weights::LinearAttentionWeights,
    x: &MlxArray,
    seq: i32,
    profile_enabled: bool,
) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
    qwen_prefill_maybe_eval_la_input(x, seq);
    let x_contig;
    let x = if fastpath::should_qwen_prefill_contiguous_la_input(&model_cfg.model_family, seq) {
        x_contig = contiguous(x, None);
        &x_contig
    } else {
        x
    };
    if let (Some(qkvz_w), Some(ba_w)) = (&w.in_proj_qkvz, &w.in_proj_ba) {
        let (qkvz_w, ba_w) = if fastpath::should_qwen_la_prefill_q2(seq) {
            match (w.prefill_q2_qkvz.as_ref(), w.prefill_q2_ba.as_ref()) {
                (Some(q2_qkvz), Some(q2_ba)) => (q2_qkvz, q2_ba),
                _ => (qkvz_w, ba_w),
            }
        } else {
            (qkvz_w, ba_w)
        };
        let contig_qkvz;
        let contig_ba;
        let (qkvz_w, ba_w) = if fastpath::should_qwen_prefill_contiguous_la_weights(seq) {
            contig_qkvz = cached_prefill_la_contiguous_weight(qkvz_w);
            contig_ba = cached_prefill_la_contiguous_weight(ba_w);
            (&contig_qkvz, &contig_ba)
        } else {
            (qkvz_w, ba_w)
        };
        let fuse_norm = qwen_la_norm_qkvz_fuse_weights();
        // Exact S=2..=4: one MXFP4 qmm for QKVZ+BA instead of two. S=1
        // decode keeps the split qmm pair. Isolated BA is ~10ms of LA.
        if fuse_norm.is_none()
            && !profile_enabled
            && fastpath::qwen_linear_mtp_exact_enabled()
            && fastpath::qwen_linear_mtp_verify_seq_contains(seq as i64)
            && qkvz_w.matching_mxfp4_quant(ba_w)
            && let Some(outputs) = linear_attention_inputs_fused_qmm(
                model_cfg.compile_cache_identity,
                cfg,
                x,
                qkvz_w,
                ba_w,
                w.fused_qkvz_ba.as_ref(),
            )
        {
            return outputs;
        }
        // Compile-fold of attn_norm missed (or this is the split-qmm path).
        // The layer shell already skipped the outer RMS.
        let x_exact_norm;
        let x = if fuse_norm.is_none() && qwen_la_exact_attn_norm().is_some() {
            x_exact_norm = apply_bound_exact_attn_norm(x);
            &x_exact_norm
        } else {
            x
        };
        if fuse_norm.is_none()
            && !fastpath::qwen_linear_mtp_exact_for_seq(seq)
            && !profile_enabled
            && should_fuse_qkvz_ba_qmm(qkvz_w, ba_w, seq)
            && let Some(outputs) = linear_attention_inputs_fused_qmm(
                model_cfg.compile_cache_identity,
                cfg,
                x,
                qkvz_w,
                ba_w,
                w.fused_qkvz_ba.as_ref(),
            )
        {
            return outputs;
        }
        let qwen_default_enabled = qwen_linear_attention_direct_cpp_default_family(model_cfg)
            && fastpath::qwen_direct_cpp_linear_attention_inputs_enabled()
            && !fastpath::qwen_linear_mtp_exact_for_seq(seq);
        if fuse_norm.is_none()
            && !fastpath::qwen_linear_mtp_exact_for_seq(seq)
            && (fastpath::direct_cpp_linear_attention_inputs_enabled() || qwen_default_enabled)
        {
            record_linear_attention_direct_cpp_inputs_attempt();
            if profile_enabled {
                record_linear_attention_direct_cpp_inputs_profile_blocked();
                record_linear_attention_direct_cpp_inputs_fallback();
            } else if let Some(outputs) =
                linear_attention_inputs_packed_direct(cfg, x, qkvz_w, ba_w)
            {
                record_linear_attention_direct_cpp_inputs_hit();
                return outputs;
            } else {
                record_linear_attention_direct_cpp_inputs_fallback();
            }
        }

        let profile_started = Instant::now();
        let mixed_qkvz = if let Some((norm_w, eps)) = &fuse_norm {
            qw_rms_norm_qmm(x, norm_w, *eps, qkvz_w)
        } else {
            qw(x, qkvz_w)
        };
        linear_attention_profile_eval_elapsed(
            profile_enabled,
            LinearAttentionProfileStage::ProjectionQkvz,
            profile_started,
            &[&mixed_qkvz],
        );
        let value_heads_per_key = cfg.num_value_heads / cfg.num_key_heads;
        let value_dim_per_key = value_heads_per_key * cfg.value_head_dim;
        let qkvz_per_key = cfg.key_head_dim * 2 + value_dim_per_key * 2;
        let mixed_qkvz = reshape(
            &mixed_qkvz,
            &[1, seq, cfg.num_key_heads as i32, qkvz_per_key as i32],
            None,
        );
        let q = slice_last_dim(&mixed_qkvz, 0, cfg.key_head_dim as i32, None);
        let k = slice_last_dim(
            &mixed_qkvz,
            cfg.key_head_dim as i32,
            (cfg.key_head_dim * 2) as i32,
            None,
        );
        let v = slice_last_dim(
            &mixed_qkvz,
            (cfg.key_head_dim * 2) as i32,
            (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
            None,
        );
        let z = slice_last_dim(
            &mixed_qkvz,
            (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
            qkvz_per_key as i32,
            None,
        );
        let qkv = concatenate(
            &[
                &reshape(&q, &[1, seq, cfg.key_dim() as i32], None),
                &reshape(&k, &[1, seq, cfg.key_dim() as i32], None),
                &reshape(&v, &[1, seq, cfg.value_dim() as i32], None),
            ],
            2,
            None,
        );
        let z = reshape(
            &z,
            &[
                1,
                seq,
                cfg.num_value_heads as i32,
                cfg.value_head_dim as i32,
            ],
            None,
        );

        let profile_started = Instant::now();
        let mixed_ba = if let Some((norm_w, eps)) = &fuse_norm {
            qw_rms_norm_qmm(x, norm_w, *eps, ba_w)
        } else {
            qw(x, ba_w)
        };
        linear_attention_profile_eval_elapsed(
            profile_enabled,
            LinearAttentionProfileStage::ProjectionBa,
            profile_started,
            &[&mixed_ba],
        );
        let ba = reshape(
            &mixed_ba,
            &[
                1,
                seq,
                cfg.num_key_heads as i32,
                (value_heads_per_key * 2) as i32,
            ],
            None,
        );
        let b = reshape(
            &slice_last_dim(&ba, 0, value_heads_per_key as i32, None),
            &[1, seq, cfg.num_value_heads as i32],
            None,
        );
        let a = reshape(
            &slice_last_dim(
                &ba,
                value_heads_per_key as i32,
                (value_heads_per_key * 2) as i32,
                None,
            ),
            &[1, seq, cfg.num_value_heads as i32],
            None,
        );
        return (qkv, z, a, b);
    }

    let profile_started = Instant::now();
    let qkv = qw(
        x,
        w.in_proj_qkv
            .as_ref()
            .expect("split linear attention must have qkv projection"),
    );
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::ProjectionQkv,
        profile_started,
        &[&qkv],
    );
    let profile_started = Instant::now();
    let z = reshape(
        &qw(
            x,
            w.in_proj_z
                .as_ref()
                .expect("split linear attention must have z projection"),
        ),
        &[
            1,
            seq,
            cfg.num_value_heads as i32,
            cfg.value_head_dim as i32,
        ],
        None,
    );
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::ProjectionZ,
        profile_started,
        &[&z],
    );
    let profile_started = Instant::now();
    let a = qw(
        x,
        w.in_proj_a
            .as_ref()
            .expect("split linear attention must have a projection"),
    );
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::ProjectionA,
        profile_started,
        &[&a],
    );
    let profile_started = Instant::now();
    let b = qw(
        x,
        w.in_proj_b
            .as_ref()
            .expect("split linear attention must have b projection"),
    );
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::ProjectionB,
        profile_started,
        &[&b],
    );
    (qkv, z, a, b)
}

fn qwen_linear_attention_direct_cpp_default_family(cfg: &ModelConfig) -> bool {
    matches!(cfg.model_family.as_str(), "qwen3_5" | "qwen3_next")
}

fn linear_attention_full_gate_metal_allowed(
    cfg: &ModelConfig,
    w: &LinearAttentionWeights,
    layer_idx: usize,
) -> bool {
    // Qwen3.6 27B 5-bit is token-exact against mlx_lm only when later
    // linear-attention gated norms keep MLX's rms_norm node and use the
    // narrower gate Metal node. The early layers retain the full fused kernel:
    // disabling it globally regresses other correctness prompts.
    if qwen_linear_attention_direct_cpp_default_family(cfg)
        && !linear_attention_full_gate_metal_allowed_for_bits(
            cfg.model_family.as_str(),
            w.out_proj.scales.is_some(),
            w.out_proj.bits,
            layer_idx,
        )
    {
        return false;
    }
    true
}

/// Exact S=2..=4 fused Metal is limited to the same early-layer window
/// as 5-bit Qwen (`layer_idx < 16`). Later layers stay portable.
/// Factory `dced27d4` still flipped ON to `f4b5490d`; call site stays off.
#[allow(dead_code)]
const EXACT_S2_FULL_GATE_METAL_LAYER_LIMIT: usize = 16;

#[allow(dead_code)]
fn exact_s2_full_gate_metal_allowed(seq: i32, layer_idx: usize, family_allow: bool) -> bool {
    family_allow
        && fastpath::qwen_linear_mtp_verify_seq_contains(seq as i64)
        && layer_idx < EXACT_S2_FULL_GATE_METAL_LAYER_LIMIT
}

fn linear_attention_full_gate_metal_allowed_for_bits(
    model_family: &str,
    quantized: bool,
    bits: i32,
    layer_idx: usize,
) -> bool {
    if matches!(model_family, "qwen3_5" | "qwen3_next") && quantized && bits == 5 {
        return layer_idx < 16;
    }
    true
}

/// Mixed qkvz/ba quant is prefill-only. Decode keeps the matching-bits pack.
pub(crate) const fn linear_attention_prefill_allows_mixed_pack(
    seq: i32,
    mixed_quant: bool,
) -> bool {
    !mixed_quant || seq > 1
}

fn should_fuse_qkvz_ba_qmm(qkvz_w: &QuantizedWeight, ba_w: &QuantizedWeight, seq: i32) -> bool {
    // MXFP4 packs materialize `fused_qkvz_ba` at load just like affine; the
    // fused qmm underneath (`qw`) is mode-aware, so prefill may take the
    // single-qmm route for both contracts.
    fastpath::should_qwen_la_fused_qkvz_ba_qmm(
        seq,
        qkvz_w.matching_affine_quant(ba_w) || qkvz_w.matching_mxfp4_quant(ba_w),
    )
}

fn packed_la_outputs_match_cfg(
    qkv: &MlxArray,
    z: &MlxArray,
    a: &MlxArray,
    b: &MlxArray,
    x: &MlxArray,
    cfg: &LinearAttentionConfig,
) -> bool {
    let seq = x.shape()[1];
    qkv.shape() == vec![1, seq, cfg.conv_dim() as i32]
        && z.shape()
            == vec![
                1,
                seq,
                cfg.num_value_heads as i32,
                cfg.value_head_dim as i32,
            ]
        && a.shape() == vec![1, seq, cfg.num_value_heads as i32]
        && b.shape() == vec![1, seq, cfg.num_value_heads as i32]
}

fn packed_qkvz_ba_widths(cfg: &LinearAttentionConfig) -> (i32, i32) {
    let value_heads_per_key = cfg.num_value_heads / cfg.num_key_heads;
    let value_dim_per_key = value_heads_per_key * cfg.value_head_dim;
    let qkvz_per_key = cfg.key_head_dim * 2 + value_dim_per_key * 2;
    (
        (cfg.num_key_heads * qkvz_per_key) as i32,
        (cfg.num_key_heads * value_heads_per_key * 2) as i32,
    )
}

fn split_packed_qkvz_ba_projection(
    cfg: &LinearAttentionConfig,
    mixed_qkvz: &MlxArray,
    mixed_ba: &MlxArray,
    batch: i32,
    seq: i32,
) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
    let value_heads_per_key = cfg.num_value_heads / cfg.num_key_heads;
    let value_dim_per_key = value_heads_per_key * cfg.value_head_dim;
    let qkvz_per_key = cfg.key_head_dim * 2 + value_dim_per_key * 2;
    let mixed_qkvz = reshape(
        mixed_qkvz,
        &[batch, seq, cfg.num_key_heads as i32, qkvz_per_key as i32],
        None,
    );
    let q = slice_last_dim(&mixed_qkvz, 0, cfg.key_head_dim as i32, None);
    let k = slice_last_dim(
        &mixed_qkvz,
        cfg.key_head_dim as i32,
        (cfg.key_head_dim * 2) as i32,
        None,
    );
    let v = slice_last_dim(
        &mixed_qkvz,
        (cfg.key_head_dim * 2) as i32,
        (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
        None,
    );
    let z = slice_last_dim(
        &mixed_qkvz,
        (cfg.key_head_dim * 2 + value_dim_per_key) as i32,
        qkvz_per_key as i32,
        None,
    );
    let qkv = concatenate(
        &[
            &reshape(&q, &[batch, seq, cfg.key_dim() as i32], None),
            &reshape(&k, &[batch, seq, cfg.key_dim() as i32], None),
            &reshape(&v, &[batch, seq, cfg.value_dim() as i32], None),
        ],
        2,
        None,
    );
    let z = reshape(
        &z,
        &[
            batch,
            seq,
            cfg.num_value_heads as i32,
            cfg.value_head_dim as i32,
        ],
        None,
    );
    let ba = reshape(
        mixed_ba,
        &[
            batch,
            seq,
            cfg.num_key_heads as i32,
            (value_heads_per_key * 2) as i32,
        ],
        None,
    );
    let b = reshape(
        &slice_last_dim(&ba, 0, value_heads_per_key as i32, None),
        &[batch, seq, cfg.num_value_heads as i32],
        None,
    );
    let a = reshape(
        &slice_last_dim(
            &ba,
            value_heads_per_key as i32,
            (value_heads_per_key * 2) as i32,
            None,
        ),
        &[batch, seq, cfg.num_value_heads as i32],
        None,
    );
    (qkv, z, a, b)
}

fn linear_attention_inputs_fused_qmm(
    model_identity: u64,
    cfg: &LinearAttentionConfig,
    x: &MlxArray,
    qkvz_w: &QuantizedWeight,
    ba_w: &QuantizedWeight,
    load_fused: Option<&QuantizedWeight>,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    let owned = if load_fused.is_none() {
        qkvz_w.concat_output_rows(ba_w)
    } else {
        None
    };
    let fused = load_fused.or(owned.as_ref())?;
    let batch = x.shape().first().copied().unwrap_or(1);
    let seq = x.shape().get(1).copied()?;
    if let Some(compiled) =
        compiled_fused_qkvz_ba_qmm_unpack(model_identity, cfg, x, fused, batch, seq)
    {
        return Some(compiled);
    }
    let x = apply_bound_exact_attn_norm(x);
    let mixed = qw(&x, fused);
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(cfg);
    let last = *mixed.shape().last()?;
    if last != qkvz_out + ba_out {
        return None;
    }
    let mixed_qkvz = slice_last_dim(&mixed, 0, qkvz_out, None);
    let mixed_ba = slice_last_dim(&mixed, qkvz_out, qkvz_out + ba_out, None);
    if let Some(compiled) = compiled_split_packed_qkvz_ba_projection(
        model_identity,
        cfg,
        &mixed_qkvz,
        &mixed_ba,
        batch,
        seq,
    ) {
        return Some(compiled);
    }
    Some(split_packed_qkvz_ba_projection(
        cfg,
        &mixed_qkvz,
        &mixed_ba,
        batch,
        seq,
    ))
}

/// Compile identity for exact S=2..=4 fused QKVZ+BA qmm + unpack.
/// One graph is shared across every linear-attention layer in one model.
const EXACT_LA_FUSED_QMM_UNPACK_COMPILE_ID: u64 = 0x5155_4D4D_554E_5032;
/// Distinct cache key when `attn_norm` is compiled into the same closure.
const EXACT_LA_RMS_QMM_UNPACK_COMPILE_ID: u64 = 0x5155_524D_5351_4D32;

fn compiled_fused_qkvz_ba_qmm_unpack(
    model_identity: u64,
    cfg: &LinearAttentionConfig,
    x: &MlxArray,
    fused: &QuantizedWeight,
    batch: i32,
    seq: i32,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    if !fastpath::qwen_linear_mtp_exact_enabled()
        || batch != 1
        || !fastpath::qwen_linear_mtp_verify_seq_contains(seq as i64)
    {
        return None;
    }
    // The closure below rebuilds the weight with `biases: None`, which is
    // only correct for scales-only contracts. An affine fused pack must not
    // reach it, or its group biases would be silently dropped.
    if fused.biases.is_some() {
        return None;
    }
    let scales = fused.scales.as_ref()?;
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(cfg);
    let leading = i64::from(batch).checked_mul(i64::from(seq))?;
    let cfg = cfg.clone();
    let group_size = fused.group_size;
    let bits = fused.bits;
    let mode = fused.mode.clone();
    let attn_norm = qwen_la_exact_attn_norm();
    let quant_salt = compile_quant_contract_salt(&[fused]);
    let (compile_id, input_store, fold_rms, rms_eps) = if let Some((norm_w, eps)) = attn_norm {
        (
            EXACT_LA_RMS_QMM_UNPACK_COMPILE_ID ^ quant_salt,
            vec![x.clone(), norm_w, fused.weight.clone(), scales.clone()],
            true,
            eps,
        )
    } else {
        (
            EXACT_LA_FUSED_QMM_UNPACK_COMPILE_ID ^ quant_salt,
            vec![x.clone(), fused.weight.clone(), scales.clone()],
            false,
            0.0,
        )
    };
    let input_refs: Vec<&MlxArray> = input_store.iter().collect();
    crate::per_layer_compile::apply_layer_dense_ffn_prefill_min(
        model_identity ^ compile_id,
        SHARED_VERIFY_COMPILE_LAYER,
        leading,
        2,
        &input_refs,
        move |inputs: &MlxVectorArray| {
            let x = if fold_rms {
                rms_norm(&inputs.get(0), Some(&inputs.get(1)), rms_eps, None)
            } else {
                inputs.get(0)
            };
            let weight_idx = if fold_rms { 2 } else { 1 };
            let fused = QuantizedWeight {
                weight: inputs.get(weight_idx),
                scales: Some(inputs.get(weight_idx + 1)),
                biases: None,
                group_size,
                bits,
                mode: mode.clone(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            };
            let mixed = qw(&x, &fused);
            let mixed_qkvz = slice_last_dim(&mixed, 0, qkvz_out, None);
            let mixed_ba = slice_last_dim(&mixed, qkvz_out, qkvz_out + ba_out, None);
            let (qkv, z, a, b) =
                split_packed_qkvz_ba_projection(&cfg, &mixed_qkvz, &mixed_ba, 1, seq);
            vec![qkv, z, a, b]
        },
    )
    .and_then(|mut outs| {
        if outs.len() != 4 {
            return None;
        }
        let b = outs.pop()?;
        let a = outs.pop()?;
        let z = outs.pop()?;
        let qkv = outs.pop()?;
        Some((qkv, z, a, b))
    })
}

/// Compile identity for the exact S=2..=4 QKVZ/BA unpack glue. One graph
/// is shared across every linear-attention layer in one model.
const EXACT_LA_UNPACK_COMPILE_ID: u64 = 0x5155_4E50_4143_4B32;

/// Shape-compile the reshape/slice/concat unpack after fused QKVZ+BA qmm.
///
/// Sits between two graph-breaks (the fused qmm and Metal post-input). Does
/// not touch the portable RMS+SiLU gate. Falls back to the imperative unpack
/// when exact MTP is off or compile fails.
fn compiled_split_packed_qkvz_ba_projection(
    model_identity: u64,
    cfg: &LinearAttentionConfig,
    mixed_qkvz: &MlxArray,
    mixed_ba: &MlxArray,
    batch: i32,
    seq: i32,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    if !fastpath::qwen_linear_mtp_exact_enabled()
        || batch != 1
        || !fastpath::qwen_linear_mtp_verify_seq_contains(seq as i64)
    {
        return None;
    }
    let leading = i64::from(batch).checked_mul(i64::from(seq))?;
    let cfg = cfg.clone();
    let inputs = [mixed_qkvz, mixed_ba];
    crate::per_layer_compile::apply_layer_dense_ffn_prefill_min(
        model_identity ^ EXACT_LA_UNPACK_COMPILE_ID,
        SHARED_VERIFY_COMPILE_LAYER,
        leading,
        2,
        &inputs,
        move |inputs: &MlxVectorArray| {
            let (qkv, z, a, b) =
                split_packed_qkvz_ba_projection(&cfg, &inputs.get(0), &inputs.get(1), 1, seq);
            vec![qkv, z, a, b]
        },
    )
    .and_then(|mut outs| {
        if outs.len() != 4 {
            return None;
        }
        let b = outs.pop()?;
        let a = outs.pop()?;
        let z = outs.pop()?;
        let qkv = outs.pop()?;
        Some((qkv, z, a, b))
    })
}

fn linear_attention_inputs_packed_direct(
    cfg: &LinearAttentionConfig,
    x: &MlxArray,
    qkvz_w: &crate::weights::QuantizedWeight,
    ba_w: &crate::weights::QuantizedWeight,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    // The packed C++ helper infers each projection's mode from its bias
    // channel: affine with group biases, scales-only MXFP4 without.
    if !qkvz_w.is_fused_qmm_quantized() || !ba_w.is_fused_qmm_quantized() {
        return None;
    }
    let qkvz_quantized = qkvz_w.scales.is_some();
    let ba_quantized = ba_w.scales.is_some();
    let mixed_quant = qkvz_quantized
        && ba_quantized
        && (qkvz_w.group_size != ba_w.group_size || qkvz_w.bits != ba_w.bits);
    // Decode (seq==1) keeps matching-bits packing only: mixed 4/6-bit on
    // AXQ 27B measured 29.84 vs 30.14 tok/s. Prefill (seq>1) takes the
    // extra 31/48 packed hits — those layers are the prefill dispatch miss.
    let seq = x.shape().get(1).copied().unwrap_or(1);
    if !linear_attention_prefill_allows_mixed_pack(seq, mixed_quant) {
        return None;
    }
    let group_size = if qkvz_quantized {
        qkvz_w.group_size
    } else {
        ba_w.group_size
    };
    let bits = if qkvz_quantized {
        qkvz_w.bits
    } else {
        ba_w.bits
    };
    let ba_group_size = if mixed_quant {
        ba_w.group_size
    } else {
        group_size
    };
    let ba_bits = if mixed_quant { ba_w.bits } else { bits };

    if fastpath::should_qwen_packed_la_inputs_compile(seq)
        && let (Some(qkvz_scales), Some(ba_scales)) =
            (qkvz_w.scales.as_ref(), ba_w.scales.as_ref())
        // Both projections carry group biases (affine) or neither does
        // (scales-only MXFP4); mixed contracts fail closed in the shim.
        && qkvz_w.biases.is_some() == ba_w.biases.is_some()
        && let Some(compiled) = qwen_linear_attention_inputs_packed_compiled(
            x,
            &qkvz_w.weight,
            qkvz_scales,
            qkvz_w.biases.as_ref(),
            &ba_w.weight,
            ba_scales,
            ba_w.biases.as_ref(),
            cfg.num_key_heads as i32,
            cfg.num_value_heads as i32,
            cfg.key_head_dim as i32,
            cfg.value_head_dim as i32,
            group_size,
            bits,
            ba_group_size,
            ba_bits,
            None,
        )
        .filter(|(qkv, z, a, b)| packed_la_outputs_match_cfg(qkv, z, a, b, x, cfg))
    {
        return Some(compiled);
    }

    qwen_linear_attention_inputs_packed(
        x,
        &qkvz_w.weight,
        qkvz_w.scales.as_ref(),
        qkvz_w.biases.as_ref(),
        &ba_w.weight,
        ba_w.scales.as_ref(),
        ba_w.biases.as_ref(),
        cfg.num_key_heads as i32,
        cfg.num_value_heads as i32,
        cfg.key_head_dim as i32,
        cfg.value_head_dim as i32,
        group_size,
        bits,
        ba_group_size,
        ba_bits,
        None,
    )
    .filter(|(qkv, z, a, b)| packed_la_outputs_match_cfg(qkv, z, a, b, x, cfg))
}

// ---------------------------------------------------------------------------
// Tier 3A: Whole-layer linear-attention decode (compositional Metal path).
//
// When `AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL` is on, decode runs the
// existing Metal-accelerated gated-delta + gate pipeline under one outer
// entry. A true single-dispatch mega-kernel that also fuses quantized
// projections remains residual (hardware/kernel engineering).
// ---------------------------------------------------------------------------

/// Attempt whole-layer Metal dispatch for linear-attention decode.
///
/// Returns `Some(output)` if the compositional Metal path succeeds, `None` to
/// fall back to the standard multi-dispatch path.
///
/// Gated by `AX_MLX_LINEAR_ATTENTION_WHOLE_LAYER_METAL` (default OFF).
pub(crate) fn try_linear_attention_whole_layer_metal(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
) -> Option<MlxArray> {
    if !fastpath::linear_attention_whole_layer_metal_enabled() {
        return None;
    }
    // Compositional whole-layer decode entry: run the Metal-accelerated
    // gated-delta pipeline (existing qwen35_gated_delta_decode_v1 + Metal
    // conv/gate helpers) under one outer barrier. A single-dispatch mega
    // kernel that also fuses quantized projections remains residual.
    if x.shape().get(1).copied().unwrap_or(0) != 1 {
        return None;
    }
    let linear_cfg = cfg.linear_attention.as_ref()?;
    let linear_w = w.linear_attn.as_ref()?;
    let seq = x.shape()[1];
    let (qkv, z, a, b) = linear_attention_inputs(cfg, linear_cfg, linear_w, x, seq, false);
    let (conv_state, recurrent_state) = cache.linear_state(layer_idx);
    let (q, k, v, new_conv_state, _prefix_conv) =
        linear_attention_post_input(cfg, linear_cfg, linear_w, &qkv, conv_state, false);
    let a_log_f32 = linear_w.a_log.clone();
    let dt_bias_f32 = linear_w.dt_bias.clone();
    let state = recurrent_state.cloned().unwrap_or_else(|| {
        zeros(
            &[
                1,
                linear_cfg.num_value_heads as i32,
                linear_cfg.value_head_dim as i32,
                linear_cfg.key_head_dim as i32,
            ],
            MlxDtype::Float32,
            None,
        )
    });
    let (out, new_recurrent_state) =
        gated_delta_kernel(&q, &k, &v, &a_log_f32, &a, &dt_bias_f32, &b, &state);
    cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);
    let out = rms_norm_gated_with_full_gate_policy(
        &out,
        &z,
        &linear_w.norm,
        cfg.rms_norm_eps,
        linear_attention_full_gate_metal_allowed(cfg, linear_w, layer_idx),
    );
    let flat = reshape(&out, &[1, seq, linear_cfg.value_dim() as i32], None);
    Some(qw(&flat, &linear_w.out_proj))
}

#[cfg(test)]
mod tests;
