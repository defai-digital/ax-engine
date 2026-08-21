use mlx_sys::{
    MlxArray, MlxDtype, MlxQuantizationMode, MlxVectorArray, async_eval, concatenate, contiguous,
    eval, qwen_linear_attention_inputs_packed, qwen_linear_attention_inputs_packed_compiled,
    qwen_linear_attention_post_input, qwen_linear_attention_post_input_compiled, reshape, rms_norm,
    rms_norm_quantized_matmul, silu_mul_quantized_matmul, slice, slice_last_dim, zeros,
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
    gated_delta_kernel, gated_delta_kernel_with_prefix_checkpoint, linear_attention_conv1d,
    linear_attention_decode_post_input_metal, normalize_linear_attention_qk,
    rms_norm_gated_with_full_gate_policy, slice_seq_row_4d, split_linear_attention_qkv,
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

/// Same as [`linear_attention_forward`] but stops after the portable RMS+SiLU
/// gate, returning `[1, seq, value_dim]` so the caller can compile `out_proj`
/// together with residual+FFN.
///
/// Unhooked from the factory verify path: folding `out_proj` into the
/// residual+FFN compile reproduced `f4b5490d`.
#[allow(dead_code)]
pub(crate) fn linear_attention_forward_pre_out_proj(
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
        true,
        false,
    )
}

/// Run LA through GatedDelta and last-token slice; return `(gd_out, z)`
/// so exact S=2 can compile portable gate + o_proj + residual + FFN as
/// one closure (one eval of hidden+gd+z).
/// Factory `19bc8f95` ON=`f4b5490d`; unhooked.
#[allow(dead_code)]
pub(crate) fn linear_attention_forward_pre_gate(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    skip_out_proj: bool,
    last_token_out_proj: bool,
) -> (MlxArray, MlxArray) {
    let gd = linear_attention_forward_inner(
        cfg,
        w,
        x,
        cache,
        layer_idx,
        skip_out_proj,
        last_token_out_proj,
        false,
        true,
    );
    let z = LA_PRE_GATE_Z
        .with(|slot| slot.borrow_mut().take())
        .expect("pre-gate forward must stash z");
    (gd, z)
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
    let (q, k, v, new_conv_state, metal_prefix_conv) =
        linear_attention_post_input(cfg, linear_cfg, linear_w, &qkv, conv_state, profile_enabled);
    let prefix_conv_state = match (prefix_capture_after, metal_prefix_conv) {
        (Some(1), Some(prefix)) => Some(prefix),
        (Some(after), _) => linear_attention_conv_prefix_state(linear_cfg, &qkv, conv_state, after),
        (None, _) => None,
    };
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
    // g and beta are computed inside the Metal kernel (fused) instead of as separate
    // lazy MLX ops, eliminating ~8 kernel dispatches per layer.
    let (out, new_recurrent_state, prefix_recurrent_state) =
        if let Some(after) = prefix_capture_after {
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
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Recurrent,
        profile_started,
        &[&out, &new_recurrent_state],
    );
    cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);
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
        try_qwen_la_out_proj_silu_mul_qmm(cfg, &out, &z, linear_w, seq, value_dim)
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
    let Some((qkv, a, b)) = verify_cache.linear_mtp_projection_stash(layer_idx) else {
        return false;
    };
    let Some(qkv) = mtp_projection_prefix(&qkv, keep) else {
        return false;
    };
    let Some(a) = mtp_projection_prefix(&a, keep) else {
        return false;
    };
    let Some(b) = mtp_projection_prefix(&b, keep) else {
        return false;
    };

    let (source_conv, source_recurrent) = source_cache.linear_state(layer_idx);
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
    if !fastpath::qwen_linear_mtp_exact_enabled() || !(2..=4).contains(&seq) {
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

fn try_qwen_la_out_proj_silu_mul_qmm(
    cfg: &ModelConfig,
    hidden: &MlxArray,
    gate: &MlxArray,
    linear_w: &LinearAttentionWeights,
    seq: i32,
    value_dim: i32,
) -> Option<MlxArray> {
    if !fastpath::should_qwen_la_out_proj_silu_mul_qmm(&cfg.model_family, seq) {
        return None;
    }
    if !linear_w.out_proj.is_fused_qmm_quantized() {
        return None;
    }
    let scales = linear_w.out_proj.scales.as_ref()?;
    let normed = rms_norm(hidden, Some(&linear_w.norm), cfg.rms_norm_eps, None);
    let flat_n = reshape(&normed, &[1, seq, value_dim], None);
    let flat_z = reshape(gate, &[1, seq, value_dim], None);
    silu_mul_quantized_matmul(
        &flat_z,
        &flat_n,
        &linear_w.out_proj.weight,
        scales,
        linear_w.out_proj.biases.as_ref(),
        linear_w.out_proj.group_size,
        linear_w.out_proj.bits,
        None,
    )
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
    let speculative_multi_token =
        (2..=4).contains(&seq) && fastpath::qwen_linear_mtp_exact_enabled();
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
            && (2..=4).contains(&seq)
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
    family_allow && (2..=4).contains(&seq) && layer_idx < EXACT_S2_FULL_GATE_METAL_LAYER_LIMIT
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
    if !fastpath::qwen_linear_mtp_exact_enabled() || batch != 1 || !(2..=4).contains(&seq) {
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
    if !fastpath::qwen_linear_mtp_exact_enabled() || batch != 1 || !(2..=4).contains(&seq) {
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
mod tests {
    use super::*;

    const TEST_COMPILE_IDENTITY: u64 = 0x5445_5354_4C41_4348;

    #[test]
    fn qwen_five_bit_full_gate_policy_keeps_only_early_layers() {
        assert!(linear_attention_full_gate_metal_allowed_for_bits(
            "qwen3_5", true, 5, 15
        ));
        assert!(!linear_attention_full_gate_metal_allowed_for_bits(
            "qwen3_5", true, 5, 16
        ));
        assert!(linear_attention_full_gate_metal_allowed_for_bits(
            "qwen3_5", true, 4, 63
        ));
        assert!(linear_attention_full_gate_metal_allowed_for_bits(
            "glm4_moe_lite",
            true,
            5,
            63
        ));
        assert!(linear_attention_full_gate_metal_allowed_for_bits(
            "qwen3_5", false, 5, 63
        ));
    }

    #[test]
    fn exact_s2_full_gate_metal_follows_early_layer_policy() {
        assert!(exact_s2_full_gate_metal_allowed(2, 0, true));
        assert!(exact_s2_full_gate_metal_allowed(2, 15, true));
        assert!(!exact_s2_full_gate_metal_allowed(2, 16, true));
        assert!(!exact_s2_full_gate_metal_allowed(1, 0, true));
        assert!(!exact_s2_full_gate_metal_allowed(2, 0, false));
        assert!(exact_s2_full_gate_metal_allowed(4, 7, true));
    }

    #[test]
    fn exact_s2_s1_metal_gate_o_proj_matches_two_s1_rows() {
        let seq = 2i32;
        let hv = 2i32;
        let dv = 32i32;
        let value_dim = hv * dv;
        let n = (seq * hv * dv) as usize;
        let hidden_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
        let gate_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
        let norm_data: Vec<f32> = (0..dv as usize)
            .map(|i| 0.75 + (i as f32) * 0.004)
            .collect();
        let o_data: Vec<f32> = (0..(value_dim * value_dim) as usize)
            .map(|i| ((i as f32) - 20.0) * 0.004)
            .collect();
        let to_bf16 = |data: &[f32], shape: &[i32]| {
            mlx_sys::astype(
                &MlxArray::from_raw_data(
                    data.as_ptr() as *const u8,
                    std::mem::size_of_val(data),
                    shape,
                    MlxDtype::Float32,
                ),
                MlxDtype::Bfloat16,
                None,
            )
        };
        let hidden = to_bf16(&hidden_data, &[1, seq, hv, dv]);
        let gate = to_bf16(&gate_data, &[1, seq, hv, dv]);
        let norm = to_bf16(&norm_data, &[dv]);
        let o_w = MlxArray::from_raw_data(
            o_data.as_ptr() as *const u8,
            std::mem::size_of_val(o_data.as_slice()),
            &[value_dim, value_dim],
            MlxDtype::Float32,
        );
        let q = mlx_sys::quantize(
            &o_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        let out_proj = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let dummy = zeros(&[1], MlxDtype::Float32, None);
        let linear_w = LinearAttentionWeights {
            in_proj_qkv: None,
            in_proj_z: None,
            in_proj_a: None,
            in_proj_b: None,
            in_proj_qkvz: None,
            in_proj_ba: None,
            fused_qkvz_ba: None,
            prefill_q2_qkvz: None,
            prefill_q2_ba: None,
            conv1d_dense: dummy.clone(),
            conv1d_bias: None,
            dt_bias: dummy.clone(),
            a_log: dummy,
            d: None,
            norm: norm.clone(),
            out_proj: out_proj.clone(),
        };
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let shipped = exact_verify_s1_metal_gate_o_proj(
            &hidden, &gate, &linear_w, 1e-6, value_dim, seq, true,
        )
        .expect("exact S=2 per-row S=1 Metal gate+o_proj must engage");
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        let r0 = {
            let h = slice_seq_row_4d(&hidden, 0);
            let g = slice_seq_row_4d(&gate, 0);
            let gated = rms_norm_gated_with_full_gate_policy(&h, &g, &norm, 1e-6, true);
            qw(&reshape(&gated, &[1, 1, value_dim], None), &out_proj)
        };
        let r1 = {
            let h = slice_seq_row_4d(&hidden, 1);
            let g = slice_seq_row_4d(&gate, 1);
            let gated = rms_norm_gated_with_full_gate_policy(&h, &g, &norm, 1e-6, true);
            qw(&reshape(&gated, &[1, 1, value_dim], None), &out_proj)
        };
        let expected = concatenate(&[&r0, &r1], 1, None);
        eval(&[&shipped, &expected]);
        let a = mlx_sys::astype(&shipped, MlxDtype::Float32, None);
        let b = mlx_sys::astype(&expected, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let mut max_abs = 0.0f32;
        for (l, r) in a.data_f32().iter().zip(b.data_f32().iter()) {
            max_abs = max_abs.max((l - r).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "per-row S=1 Metal gate+o_proj must match two decode rows, max_abs={max_abs}"
        );
        let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(
            exact_verify_s1_metal_gate_o_proj(
                &hidden, &gate, &linear_w, 1e-6, value_dim, seq, true
            )
            .is_none(),
            "per-row S=1 path must stay off when exact MTP is scoped off"
        );
    }

    #[test]
    fn mixed_pack_is_prefill_only() {
        assert!(linear_attention_prefill_allows_mixed_pack(128, true));
        assert!(linear_attention_prefill_allows_mixed_pack(512, true));
        assert!(linear_attention_prefill_allows_mixed_pack(2048, true));
        assert!(!linear_attention_prefill_allows_mixed_pack(1, true));
        assert!(linear_attention_prefill_allows_mixed_pack(1, false));
        assert!(linear_attention_prefill_allows_mixed_pack(128, false));
    }

    #[test]
    fn qwen_prefill_maybe_skip_unused_la_out_returns_input_when_set() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let x = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 32, 1],
            MlxDtype::Float32,
        );
        let skipped = qwen_prefill_maybe_skip_unused_la_out(&x, true)
            .expect("skip must return the unused residual placeholder");
        mlx_sys::eval(&[&skipped]);
        assert_eq!(skipped.shape(), x.shape());
        assert!(
            skipped.data_f32().iter().all(|v| v.is_finite()),
            "skipped unused LA out must leave a finite placeholder"
        );
        assert!(qwen_prefill_maybe_skip_unused_la_out(&x, false).is_none());
        assert!(
            fastpath::should_qwen_prefill_skip_unused_la_out_for(true, "qwen3_5", true, 1024),
            "shipped unused-LA-out skip must accept the p2048 cache-only last layer"
        );
    }

    #[test]
    fn qwen_prefill_maybe_last_token_la_out_slices_seq_when_set() {
        let out_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let z_data: Vec<f32> = (0..16).map(|i| (i as f32) + 100.0).collect();
        let out = MlxArray::from_raw_data(
            out_data.as_ptr() as *const u8,
            std::mem::size_of_val(out_data.as_slice()),
            &[1, 4, 2, 2],
            MlxDtype::Float32,
        );
        let z = MlxArray::from_raw_data(
            z_data.as_ptr() as *const u8,
            std::mem::size_of_val(z_data.as_slice()),
            &[1, 4, 2, 2],
            MlxDtype::Float32,
        );
        let (sliced_out, sliced_z, seq) = qwen_prefill_maybe_last_token_la_out(&out, &z, true)
            .expect("last-token LA out must slice when set");
        mlx_sys::eval(&[&sliced_out, &sliced_z]);
        assert_eq!(seq, 1);
        assert_eq!(sliced_out.shape(), vec![1, 1, 2, 2]);
        assert_eq!(sliced_z.shape(), vec![1, 1, 2, 2]);
        assert_eq!(sliced_out.data_f32(), vec![12.0, 13.0, 14.0, 15.0]);
        assert_eq!(sliced_z.data_f32(), vec![112.0, 113.0, 114.0, 115.0]);
        assert!(qwen_prefill_maybe_last_token_la_out(&out, &z, false).is_none());
        assert!(
            fastpath::should_qwen_prefill_last_token_o_proj_for(true, "qwen3_5", true, 1024),
            "shipped last-token o_proj must accept the p2048 generate last layer"
        );
    }

    #[test]
    fn qwen_prefill_maybe_async_gd_submits_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let gd = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 4, 2, 4],
            MlxDtype::Float32,
        );
        qwen_prefill_maybe_async_gd_for(&gd, true, 1024);
        mlx_sys::eval(&[&gd]);
        assert_eq!(gd.shape(), vec![1, 4, 2, 4]);
        assert!(
            gd.data_f32().iter().all(|v| v.is_finite()),
            "async GD must leave a finite materialized tensor"
        );
        assert!(
            fastpath::should_qwen_prefill_async_gd_for(true, 1024),
            "shipped async-GD gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_async_gd_for(&gd, false, 1024);
        qwen_prefill_maybe_async_gd_for(&gd, true, 512);
    }

    #[test]
    fn qwen_prefill_maybe_eval_gd_materializes_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let gd = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 4, 2, 4],
            MlxDtype::Float32,
        );
        qwen_prefill_maybe_eval_gd_for(&gd, true, 1024);
        mlx_sys::eval(&[&gd]);
        assert_eq!(gd.shape(), vec![1, 4, 2, 4]);
        assert!(
            gd.data_f32().iter().all(|v| v.is_finite()),
            "eval GD must leave a finite materialized tensor"
        );
        assert!(
            fastpath::should_qwen_prefill_eval_gd_for(true, 1024),
            "shipped eval-GD gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_eval_gd_for(&gd, false, 1024);
        qwen_prefill_maybe_eval_gd_for(&gd, true, 512);
    }

    #[test]
    fn qwen_prefill_maybe_contiguous_gd_packs_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let gd = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 4, 2, 4],
            MlxDtype::Float32,
        );
        let packed = qwen_prefill_maybe_contiguous_gd_for(gd, true, 1024);
        mlx_sys::eval(&[&packed]);
        assert_eq!(packed.shape(), vec![1, 4, 2, 4]);
        assert!(
            packed.data_f32().iter().all(|v| v.is_finite()),
            "contiguous GD must leave a finite packed tensor"
        );
        assert!(
            fastpath::should_qwen_prefill_contiguous_gd_for(true, 1024),
            "shipped contiguous-GD gate must accept the p2048 chunk length"
        );
        let data2: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let gd2 = MlxArray::from_raw_data(
            data2.as_ptr() as *const u8,
            std::mem::size_of_val(data2.as_slice()),
            &[1, 4, 2, 4],
            MlxDtype::Float32,
        );
        let kept = qwen_prefill_maybe_contiguous_gd_for(gd2, false, 1024);
        mlx_sys::eval(&[&kept]);
        assert_eq!(kept.shape(), vec![1, 4, 2, 4]);
    }

    #[test]
    fn cached_prefill_la_contiguous_weight_keeps_bits_and_qws() {
        let hidden_data: Vec<f32> = (0..32 * 32)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let proj_data: Vec<f32> = (0..64 * 32)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let hidden = MlxArray::from_raw_data(
            hidden_data.as_ptr() as *const u8,
            std::mem::size_of_val(hidden_data.as_slice()),
            &[1, 32, 32],
            MlxDtype::Float32,
        );
        let proj_w = MlxArray::from_raw_data(
            proj_data.as_ptr() as *const u8,
            std::mem::size_of_val(proj_data.as_slice()),
            &[64, 32],
            MlxDtype::Float32,
        );
        let dq = mlx_sys::quantize(
            &proj_w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let src = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let contig = cached_prefill_la_contiguous_weight(&src);
        assert_eq!(contig.bits, 4);
        assert_eq!(contig.group_size, 32);
        let again = cached_prefill_la_contiguous_weight(&src);
        assert_eq!(again.bits, contig.bits);
        let out = qw(&hidden, &contig);
        mlx_sys::eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 32, 64]);
        assert!(
            out.data_f32().iter().all(|v| v.is_finite()),
            "contiguous LA-weight qmm must produce finite values"
        );
        assert!(
            fastpath::should_qwen_prefill_contiguous_la_weights_for(true, 1024),
            "shipped LA contiguous-weight gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn qwen_prefill_maybe_eval_la_input_materializes_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let x = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data.as_slice()),
            &[1, 32, 1],
            MlxDtype::Float32,
        );
        qwen_prefill_maybe_eval_la_input_for(&x, true, 1024);
        mlx_sys::eval(&[&x]);
        assert_eq!(x.shape(), vec![1, 32, 1]);
        assert!(
            x.data_f32().iter().all(|v| v.is_finite()),
            "eval-la-input must leave a finite materialized activation"
        );
        assert!(
            fastpath::should_qwen_prefill_eval_la_input_for(true, 1024),
            "shipped LA input-eval gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_eval_la_input_for(&x, false, 1024);
        qwen_prefill_maybe_eval_la_input_for(&x, true, 512);
    }

    #[test]
    fn qwen_prefill_maybe_async_la_outputs_submits_at_min_seq() {
        let qkv_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let z_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125 + 0.1).collect();
        let a_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125 + 0.2).collect();
        let b_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125 + 0.3).collect();
        let from = |data: &[f32]| {
            MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
                &[1, 32, 1],
                MlxDtype::Float32,
            )
        };
        let qkv = from(&qkv_data);
        let z = from(&z_data);
        let a = from(&a_data);
        let b = from(&b_data);
        qwen_prefill_maybe_async_la_outputs_for(&qkv, &z, &a, &b, true, 1024);
        mlx_sys::eval(&[&qkv, &z, &a, &b]);
        assert_eq!(qkv.shape(), vec![1, 32, 1]);
        assert!(
            qkv.data_f32()
                .iter()
                .chain(z.data_f32().iter())
                .chain(a.data_f32().iter())
                .chain(b.data_f32().iter())
                .all(|v| v.is_finite()),
            "async LA outputs must leave finite materialized tensors"
        );
        assert!(
            fastpath::should_qwen_prefill_async_la_outputs_for(true, 1024),
            "shipped async LA-outputs gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_async_la_outputs_for(&qkv, &z, &a, &b, false, 1024);
        qwen_prefill_maybe_async_la_outputs_for(&qkv, &z, &a, &b, true, 512);
    }

    #[test]
    fn packed_la_inputs_compile_matches_imperative_at_min_seq() {
        let seq = fastpath::QWEN_PACKED_LA_INPUTS_COMPILE_MIN_SEQ;
        let hidden = 32_i32;
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 4,
            num_key_heads: 2,
            key_head_dim: 4,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            q_scale: 0.25,
            k_scale: 0.5,
        };
        let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
        let x_data: Vec<f32> = (0..(seq * hidden))
            .map(|i| ((i as f32) - 31.0) * 0.015625)
            .collect();
        let qkvz_data: Vec<f32> = (0..(qkvz_out * hidden))
            .map(|i| ((i as f32) - 400.0) * 0.0005)
            .collect();
        let ba_data: Vec<f32> = (0..(ba_out * hidden))
            .map(|i| ((i as f32) - 80.0) * 0.001)
            .collect();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[1, seq, hidden],
            MlxDtype::Float32,
        );
        let qkvz_w = MlxArray::from_raw_data(
            qkvz_data.as_ptr() as *const u8,
            std::mem::size_of_val(qkvz_data.as_slice()),
            &[qkvz_out, hidden],
            MlxDtype::Float32,
        );
        let ba_w = MlxArray::from_raw_data(
            ba_data.as_ptr() as *const u8,
            std::mem::size_of_val(ba_data.as_slice()),
            &[ba_out, hidden],
            MlxDtype::Float32,
        );
        let qkvz_q = mlx_sys::quantize(
            &qkvz_w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let ba_q = mlx_sys::quantize(
            &ba_w,
            Some(32),
            Some(6),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let qkvz_qw = affine_quant_weight(
            qkvz_q[0].clone(),
            qkvz_q[1].clone(),
            qkvz_q[2].clone(),
            4,
            32,
        );
        let ba_qw = affine_quant_weight(ba_q[0].clone(), ba_q[1].clone(), ba_q[2].clone(), 6, 32);
        assert!(
            fastpath::should_qwen_packed_la_inputs_compile_for(true, seq),
            "shipped compile gate must accept the p2048 chunk length"
        );
        let (compiled_qkv, compiled_z, compiled_a, compiled_b) =
            qwen_linear_attention_inputs_packed_compiled(
                &x,
                &qkvz_qw.weight,
                qkvz_qw.scales.as_ref().expect("qkvz scales"),
                qkvz_qw.biases.as_ref(),
                &ba_qw.weight,
                ba_qw.scales.as_ref().expect("ba scales"),
                ba_qw.biases.as_ref(),
                cfg.num_key_heads as i32,
                cfg.num_value_heads as i32,
                cfg.key_head_dim as i32,
                cfg.value_head_dim as i32,
                32,
                4,
                32,
                6,
                None,
            )
            .expect("compiled packed LA inputs must engage at seq>=1024");
        let (imp_qkv, imp_z, imp_a, imp_b) = qwen_linear_attention_inputs_packed(
            &x,
            &qkvz_qw.weight,
            qkvz_qw.scales.as_ref(),
            qkvz_qw.biases.as_ref(),
            &ba_qw.weight,
            ba_qw.scales.as_ref(),
            ba_qw.biases.as_ref(),
            cfg.num_key_heads as i32,
            cfg.num_value_heads as i32,
            cfg.key_head_dim as i32,
            cfg.value_head_dim as i32,
            32,
            4,
            32,
            6,
            None,
        )
        .expect("imperative packed LA inputs must stay the fallback");
        mlx_sys::eval(&[
            &compiled_qkv,
            &compiled_z,
            &compiled_a,
            &compiled_b,
            &imp_qkv,
            &imp_z,
            &imp_a,
            &imp_b,
        ]);
        for (got, want, name) in [
            (&compiled_qkv, &imp_qkv, "qkv"),
            (&compiled_z, &imp_z, "z"),
            (&compiled_a, &imp_a, "a"),
            (&compiled_b, &imp_b, "b"),
        ] {
            assert_eq!(got.shape(), want.shape(), "{name} shape");
            let g = got.data_f32();
            let w = want.data_f32();
            assert_eq!(g.len(), w.len(), "{name} len");
            for i in 0..g.len() {
                let err = (g[i] - w[i]).abs();
                assert!(
                    err < 2.0e-4,
                    "{name}[{i}] compiled {} imperative {} err {err}",
                    g[i],
                    w[i]
                );
            }
        }
    }

    #[test]
    fn contiguous_packed_qkv_post_input_matches_view() {
        let seq = 2_i32;
        let num_key_heads = 2_i32;
        let key_head_dim = 4_i32;
        let num_value_heads = 4_i32;
        let value_head_dim = 3_i32;
        let conv_kernel_dim = 4_i32;
        let conv_dim = num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim;
        let tail = conv_kernel_dim - 1;
        let qkv_data: Vec<f32> = (0..(seq * conv_dim))
            .map(|i| ((i as f32) - 16.0) * 0.0625)
            .collect();
        let conv_data: Vec<f32> = (0..(conv_dim * conv_kernel_dim))
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let state_data: Vec<f32> = (0..(tail * conv_dim))
            .map(|i| ((i as f32) - 8.0) * 0.03125)
            .collect();
        let dense = MlxArray::from_raw_data(
            qkv_data.as_ptr() as *const u8,
            std::mem::size_of_val(qkv_data.as_slice()),
            &[1, seq, conv_dim],
            MlxDtype::Float32,
        );
        let half = conv_dim / 2;
        let view = concatenate(
            &[
                &slice_last_dim(&dense, 0, half, None),
                &slice_last_dim(&dense, half, conv_dim, None),
            ],
            2,
            None,
        );
        let compact = contiguous(&view, None);
        let conv_weight = MlxArray::from_raw_data(
            conv_data.as_ptr() as *const u8,
            std::mem::size_of_val(conv_data.as_slice()),
            &[conv_dim, conv_kernel_dim, 1],
            MlxDtype::Float32,
        );
        let state = MlxArray::from_raw_data(
            state_data.as_ptr() as *const u8,
            std::mem::size_of_val(state_data.as_slice()),
            &[1, tail, conv_dim],
            MlxDtype::Float32,
        );
        let (q_s, k_s, v_s, st_s) = qwen_linear_attention_post_input(
            &view,
            &conv_weight,
            Some(&state),
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            conv_kernel_dim,
            0.5,
            0.5,
            1.0e-6,
            None,
        )
        .expect("view qkv post-input");
        let (q_c, k_c, v_c, st_c) = qwen_linear_attention_post_input(
            &compact,
            &conv_weight,
            Some(&state),
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            conv_kernel_dim,
            0.5,
            0.5,
            1.0e-6,
            None,
        )
        .expect("contiguous qkv post-input");
        mlx_sys::eval(&[&q_s, &k_s, &v_s, &st_s, &q_c, &k_c, &v_c, &st_c]);
        assert_eq!(q_s.shape(), q_c.shape());
        assert_eq!(k_s.shape(), k_c.shape());
        assert_eq!(v_s.shape(), v_c.shape());
        assert_eq!(st_s.shape(), st_c.shape());
        for (got, want, name) in [
            (&q_c, &q_s, "q"),
            (&k_c, &k_s, "k"),
            (&v_c, &v_s, "v"),
            (&st_c, &st_s, "state"),
        ] {
            let g = contiguous(got, None);
            let w = contiguous(want, None);
            mlx_sys::eval(&[&g, &w]);
            let gd = g.data_f32();
            let wd = w.data_f32();
            assert_eq!(gd.len(), wd.len(), "{name} len");
            for i in 0..gd.len() {
                let err = (gd[i] - wd[i]).abs();
                assert!(
                    err < 2.0e-4,
                    "{name}[{i}] contiguous {} view {} err {err}",
                    gd[i],
                    wd[i]
                );
            }
        }
        assert!(
            fastpath::should_qwen_la_contiguous_qkv_for(true, 1024),
            "shipped contiguous gate must accept the p2048 chunk length"
        );
    }

    fn affine_quant_weight(
        weight: MlxArray,
        scales: MlxArray,
        biases: MlxArray,
        bits: i32,
        group_size: i32,
    ) -> QuantizedWeight {
        QuantizedWeight {
            weight,
            scales: Some(scales),
            biases: Some(biases),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        }
    }

    #[test]
    fn matching_affine_quant_rejects_mxfp4_even_when_bits_match() {
        let w = MlxArray::from_raw_data(
            [0.1f32; 64].as_ptr() as *const u8,
            64 * std::mem::size_of::<f32>(),
            &[2, 32],
            MlxDtype::Float32,
        );
        let affine = mlx_sys::quantize(
            &w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let mxfp4 = mlx_sys::quantize(
            &w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        let affine_qw = QuantizedWeight {
            weight: affine[0].clone(),
            scales: Some(affine[1].clone()),
            biases: Some(affine[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let mxfp4_qw = QuantizedWeight {
            weight: mxfp4[0].clone(),
            scales: Some(mxfp4[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        assert!(affine_qw.is_affine_quantized());
        assert!(!mxfp4_qw.is_affine_quantized());
        assert!(mxfp4_qw.is_mxfp4_quantized());
        assert!(!affine_qw.is_mxfp4_quantized());
        assert!(affine_qw.is_fused_qmm_quantized());
        assert!(mxfp4_qw.is_fused_qmm_quantized());
        assert!(affine_qw.matching_affine_quant(&affine_qw));
        assert!(!mxfp4_qw.matching_affine_quant(&mxfp4_qw));
        assert!(!affine_qw.matching_affine_quant(&mxfp4_qw));
        assert!(mxfp4_qw.matching_mxfp4_quant(&mxfp4_qw));
        assert!(!affine_qw.matching_mxfp4_quant(&mxfp4_qw));
        assert!(mxfp4_qw.concat_output_rows(&mxfp4_qw).is_some());
        let mut mislabeled = mxfp4_qw.clone();
        mislabeled.mode = "affine".to_string();
        assert!(!mislabeled.is_affine_quantized());
        assert!(
            mislabeled.is_fused_qmm_quantized(),
            "mislabeled 4/32 no-bias resolves to MXFP4 and stays fused-eligible"
        );
        assert!(matches!(
            mislabeled.mlx_quantization_mode(),
            mlx_sys::MlxQuantizationMode::Mxfp4
        ));
        assert_ne!(
            affine_qw.compile_contract_word(),
            mxfp4_qw.compile_contract_word()
        );
        assert_eq!(
            mxfp4_qw.compile_contract_word(),
            mislabeled.compile_contract_word(),
            "mislabeled affine 4/32 no-bias must share the MXFP4 compile contract"
        );
        assert_eq!(
            compile_quant_contract_salt(&[&mxfp4_qw, &mxfp4_qw]),
            compile_quant_contract_salt(&[&mislabeled, &mislabeled])
        );
        let mut with_linear_bias = mxfp4_qw.clone();
        with_linear_bias.linear_bias = Some(mlx_sys::zeros(&[2], MlxDtype::Float32, None));
        assert_ne!(
            mxfp4_qw.compile_contract_word(),
            with_linear_bias.compile_contract_word(),
            "dense linear-bias presence changes the compiled input layout"
        );
        assert_ne!(
            compile_quant_contract_salt(&[&mxfp4_qw]),
            compile_quant_contract_salt(&[&with_linear_bias])
        );
    }

    #[test]
    fn matching_affine_quant_rejects_mixed_bits() {
        let w = mlx_sys::zeros(&[4, 8], MlxDtype::Uint32, None);
        let s = mlx_sys::zeros(&[4, 1], MlxDtype::Float32, None);
        let b = mlx_sys::zeros(&[4, 1], MlxDtype::Float32, None);
        let q4 = affine_quant_weight(w.clone(), s.clone(), b.clone(), 4, 32);
        let q6 = affine_quant_weight(w, s, b, 6, 32);
        assert!(q4.matching_affine_quant(&q4));
        assert!(!q4.matching_affine_quant(&q6));
        assert!(!should_fuse_qkvz_ba_qmm(&q4, &q6, 1024));
        assert!(
            !fastpath::should_qwen_la_fused_qkvz_ba_qmm_for(true, 1, true),
            "decode must not take the fused prefill qmm"
        );
    }

    #[test]
    fn exact_s2_mxfp4_fused_qkvz_ba_matches_split_qw() {
        let hidden = 32i32;
        let qkvz_out = 64i32;
        let ba_out = 16i32;
        let seq = 2i32;
        let mk = |rows: i32, seed: f32| {
            let n = (rows * hidden) as usize;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) - seed) * 0.015625).collect();
            let w = MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data.as_slice()),
                &[rows, hidden],
                MlxDtype::Float32,
            );
            let q = mlx_sys::quantize(
                &w,
                Some(32),
                Some(4),
                mlx_sys::MlxQuantizationMode::Mxfp4,
                None,
                None,
            );
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            }
        };
        let qkvz = mk(qkvz_out, 8.0);
        let ba = mk(ba_out, 3.0);
        assert!(qkvz.matching_mxfp4_quant(&ba));
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.03125)
            .collect();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[1, seq, hidden],
            MlxDtype::Float32,
        );
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let fused_w = qkvz.concat_output_rows(&ba).expect("mxfp4 concat");
        let fused = qw(&x, &fused_w);
        let split_q = qw(&x, &qkvz);
        let split_b = qw(&x, &ba);
        let expected = concatenate(&[&split_q, &split_b], 2, None);
        eval(&[&fused, &expected]);
        let a = fused.data_f32();
        let b = expected.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "fused MXFP4 QKVZ+BA qmm must match split qw, max_abs={max_abs}"
        );
    }

    #[test]
    fn exact_s2_compiled_qkvz_ba_unpack_matches_imperative() {
        let seq = 2i32;
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 4,
            num_key_heads: 2,
            key_head_dim: 4,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            q_scale: 0.25,
            k_scale: 0.5,
        };
        let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
        let qkvz_data: Vec<f32> = (0..(seq * qkvz_out) as usize)
            .map(|i| ((i as f32) - 12.0) * 0.03125)
            .collect();
        let ba_data: Vec<f32> = (0..(seq * ba_out) as usize)
            .map(|i| ((i as f32) - 4.0) * 0.0625)
            .collect();
        let mixed_qkvz = MlxArray::from_raw_data(
            qkvz_data.as_ptr() as *const u8,
            std::mem::size_of_val(qkvz_data.as_slice()),
            &[1, seq, qkvz_out],
            MlxDtype::Float32,
        );
        let mixed_ba = MlxArray::from_raw_data(
            ba_data.as_ptr() as *const u8,
            std::mem::size_of_val(ba_data.as_slice()),
            &[1, seq, ba_out],
            MlxDtype::Float32,
        );
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let compiled = compiled_split_packed_qkvz_ba_projection(
            TEST_COMPILE_IDENTITY,
            &cfg,
            &mixed_qkvz,
            &mixed_ba,
            1,
            seq,
        )
        .expect("exact S=2 unpack compile must engage");
        let imperative = split_packed_qkvz_ba_projection(&cfg, &mixed_qkvz, &mixed_ba, 1, seq);
        eval(&[
            &compiled.0,
            &compiled.1,
            &compiled.2,
            &compiled.3,
            &imperative.0,
            &imperative.1,
            &imperative.2,
            &imperative.3,
        ]);
        for (got, want, name) in [
            (&compiled.0, &imperative.0, "qkv"),
            (&compiled.1, &imperative.1, "z"),
            (&compiled.2, &imperative.2, "a"),
            (&compiled.3, &imperative.3, "b"),
        ] {
            assert_eq!(got.shape(), want.shape(), "{name} shape");
            let g = got.data_f32();
            let w = want.data_f32();
            assert_eq!(g.len(), w.len(), "{name} len");
            for i in 0..g.len() {
                let err = (g[i] - w[i]).abs();
                assert!(
                    err < 1.0e-6,
                    "{name}[{i}] compiled {} imperative {} err {err}",
                    g[i],
                    w[i]
                );
            }
        }
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(
            compiled_split_packed_qkvz_ba_projection(
                TEST_COMPILE_IDENTITY,
                &cfg,
                &mixed_qkvz,
                &mixed_ba,
                1,
                seq,
            )
            .is_none(),
            "unpack compile must stay off when exact MTP is scoped off"
        );
    }

    #[test]
    fn exact_s2_compiled_rms_qmm_unpack_matches_rms_then_fused() {
        let seq = 2i32;
        let hidden = 32i32;
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 4,
            num_key_heads: 2,
            key_head_dim: 8,
            value_head_dim: 8,
            conv_kernel_dim: 4,
            q_scale: 0.125,
            k_scale: 0.35355338,
        };
        let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
        let mk = |rows: i32, seed: f32| {
            let n = (rows * hidden) as usize;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) - seed) * 0.015625).collect();
            let w = MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data.as_slice()),
                &[rows, hidden],
                MlxDtype::Float32,
            );
            let q = mlx_sys::quantize(
                &w,
                Some(32),
                Some(4),
                mlx_sys::MlxQuantizationMode::Mxfp4,
                None,
                None,
            );
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            }
        };
        let qkvz = mk(qkvz_out, 8.0);
        let ba = mk(ba_out, 3.0);
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.03125)
            .collect();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[1, seq, hidden],
            MlxDtype::Float32,
        );
        let norm_data: Vec<f32> = (0..hidden as usize)
            .map(|i| 0.8 + (i as f32) * 0.004)
            .collect();
        let norm_w = MlxArray::from_raw_data(
            norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(norm_data.as_slice()),
            &[hidden],
            MlxDtype::Float32,
        );
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        set_qwen_la_exact_attn_norm(Some((norm_w.clone(), 1e-6)));
        let compiled =
            linear_attention_inputs_fused_qmm(TEST_COMPILE_IDENTITY, &cfg, &x, &qkvz, &ba, None)
                .expect("exact S=2 rms+qmm+unpack compile must engage");
        set_qwen_la_exact_attn_norm(None);
        let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
        let imperative = linear_attention_inputs_fused_qmm(
            TEST_COMPILE_IDENTITY,
            &cfg,
            &normed,
            &qkvz,
            &ba,
            None,
        )
        .expect("exact S=2 qmm+unpack compile must engage");
        eval(&[
            &compiled.0,
            &compiled.1,
            &compiled.2,
            &compiled.3,
            &imperative.0,
            &imperative.1,
            &imperative.2,
            &imperative.3,
        ]);
        for (got, want, name) in [
            (&compiled.0, &imperative.0, "qkv"),
            (&compiled.1, &imperative.1, "z"),
            (&compiled.2, &imperative.2, "a"),
            (&compiled.3, &imperative.3, "b"),
        ] {
            assert_eq!(got.shape(), want.shape(), "{name} shape");
            let g = got.data_f32();
            let w = want.data_f32();
            for i in 0..g.len() {
                let err = (g[i] - w[i]).abs();
                assert!(
                    err < 1.0e-5,
                    "{name}[{i}] rms-folded {} split {} err {err}",
                    g[i],
                    w[i]
                );
            }
        }
    }

    #[test]
    fn exact_attn_norm_fallback_applies_rms_when_compile_does_not() {
        // batch=2 makes compiled_fused_qkvz_ba_qmm_unpack return None.
        // The layer shell still skips outer RMS whenever the TLS is set.
        let seq = 2i32;
        let batch = 2i32;
        let hidden = 32i32;
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 4,
            num_key_heads: 2,
            key_head_dim: 8,
            value_head_dim: 8,
            conv_kernel_dim: 4,
            q_scale: 0.125,
            k_scale: 0.35355338,
        };
        let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
        let mk = |rows: i32, seed: f32| {
            let n = (rows * hidden) as usize;
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) - seed) * 0.015625).collect();
            let w = MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data.as_slice()),
                &[rows, hidden],
                MlxDtype::Float32,
            );
            let q = mlx_sys::quantize(
                &w,
                Some(32),
                Some(4),
                MlxQuantizationMode::Mxfp4,
                None,
                None,
            );
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q2_weight: None,
                decode_q2_scales: None,
                decode_q2_biases: None,
            }
        };
        let qkvz = mk(qkvz_out, 8.0);
        let ba = mk(ba_out, 3.0);
        let x_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.03125)
            .collect();
        let x = MlxArray::from_raw_data(
            x_data.as_ptr() as *const u8,
            std::mem::size_of_val(x_data.as_slice()),
            &[batch, seq, hidden],
            MlxDtype::Float32,
        );
        let norm_data: Vec<f32> = (0..hidden as usize)
            .map(|i| 0.8 + (i as f32) * 0.004)
            .collect();
        let norm_w = MlxArray::from_raw_data(
            norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(norm_data.as_slice()),
            &[hidden],
            MlxDtype::Float32,
        );
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        set_qwen_la_exact_attn_norm(Some((norm_w.clone(), 1e-6)));
        let got =
            linear_attention_inputs_fused_qmm(TEST_COMPILE_IDENTITY, &cfg, &x, &qkvz, &ba, None)
                .expect("fallback fused qmm must still run");
        set_qwen_la_exact_attn_norm(None);
        let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
        let want = linear_attention_inputs_fused_qmm(
            TEST_COMPILE_IDENTITY,
            &cfg,
            &normed,
            &qkvz,
            &ba,
            None,
        )
        .expect("normed fused qmm");
        eval(&[
            &got.0, &got.1, &got.2, &got.3, &want.0, &want.1, &want.2, &want.3,
        ]);
        let g = got.0.data_f32();
        let w = want.0.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..g.len() {
            max_abs = max_abs.max((g[i] - w[i]).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "compile-miss fallback must still apply bound attn_norm, max_abs={max_abs}"
        );
    }

    #[test]
    fn initial_recurrent_state_zeros_reuses_shape() {
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 4,
            num_key_heads: 2,
            key_head_dim: 8,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            q_scale: 0.125,
            k_scale: 0.35355338,
        };
        let a = initial_recurrent_state_zeros(&cfg);
        let b = initial_recurrent_state_zeros(&cfg);
        assert_eq!(a.shape(), vec![1, 4, 4, 8]);
        assert_eq!(b.shape(), a.shape());
        assert_eq!(a.dtype(), MlxDtype::Float32);
    }

    #[test]
    fn fused_qkvz_ba_qmm_matches_split_two_qmm() {
        let seq = 2_i32;
        let hidden = 32_i32;
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 4,
            num_key_heads: 2,
            key_head_dim: 4,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            q_scale: 0.25,
            k_scale: 0.5,
        };
        let (qkvz_out, ba_out) = packed_qkvz_ba_widths(&cfg);
        let x_data: Vec<f32> = (0..(seq * hidden))
            .map(|i| ((i as f32) - 31.0) * 0.03125)
            .collect();
        let qkvz_data: Vec<f32> = (0..(qkvz_out * hidden))
            .map(|i| ((i as f32) - 400.0) * 0.0005)
            .collect();
        let ba_data: Vec<f32> = (0..(ba_out * hidden))
            .map(|i| ((i as f32) - 80.0) * 0.001)
            .collect();
        let from_f32 = |data: &[f32], shape: &[i32]| {
            MlxArray::from_raw_data(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
                shape,
                MlxDtype::Float32,
            )
        };
        let x = from_f32(&x_data, &[1, seq, hidden]);
        let qkvz_q = mlx_sys::quantize(
            &from_f32(&qkvz_data, &[qkvz_out, hidden]),
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let ba_q = mlx_sys::quantize(
            &from_f32(&ba_data, &[ba_out, hidden]),
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let qkvz_w = affine_quant_weight(
            qkvz_q[0].clone(),
            qkvz_q[1].clone(),
            qkvz_q[2].clone(),
            4,
            32,
        );
        let ba_w = affine_quant_weight(ba_q[0].clone(), ba_q[1].clone(), ba_q[2].clone(), 4, 32);
        assert!(
            fastpath::should_qwen_la_fused_qkvz_ba_qmm_for(
                true,
                seq,
                qkvz_w.matching_affine_quant(&ba_w)
            ),
            "matching 4-bit qkvz/ba must be eligible when the fuse flag is on"
        );

        let mut packed = LinearAttentionWeights {
            in_proj_qkv: None,
            in_proj_z: None,
            in_proj_a: None,
            in_proj_b: None,
            in_proj_qkvz: Some(qkvz_w.clone()),
            in_proj_ba: Some(ba_w.clone()),
            fused_qkvz_ba: None,
            prefill_q2_qkvz: None,
            prefill_q2_ba: None,
            conv1d_dense: zeros(&[1, 1, 1], MlxDtype::Float32, None),
            conv1d_bias: None,
            dt_bias: zeros(&[1], MlxDtype::Float32, None),
            a_log: zeros(&[1], MlxDtype::Float32, None),
            d: None,
            norm: zeros(&[1], MlxDtype::Float32, None),
            out_proj: qkvz_w.clone(),
        };
        packed.prepare_fused_qkvz_ba_prefill();
        packed.prepare_prefill_q2_projections();
        let q2 = packed
            .prefill_q2_qkvz
            .as_ref()
            .expect("4-bit qkvz must grow a 2-bit prefill overlay");
        let b2 = packed
            .prefill_q2_ba
            .as_ref()
            .expect("4-bit ba must grow a 2-bit prefill overlay");
        assert_eq!(q2.bits, crate::weights::PREFILL_LA_Q2_BITS);
        assert_eq!(b2.bits, crate::weights::PREFILL_LA_Q2_BITS);
        assert_eq!(q2.group_size, crate::weights::PREFILL_LA_Q2_GROUP_SIZE);
        assert!(
            fastpath::should_qwen_la_prefill_q2_for(true, 1024),
            "shipped q2 gate must accept the p2048 chunk length"
        );
        let (q2_qkv, q2_z, q2_a, q2_b) = linear_attention_inputs_packed_direct(&cfg, &x, q2, b2)
            .expect("2-bit packed LA inputs must engage");
        mlx_sys::eval(&[&q2_qkv, &q2_z, &q2_a, &q2_b]);
        assert_eq!(q2_qkv.shape()[1], seq);
        assert!(
            packed.fused_qkvz_ba.is_some(),
            "load-time matching-bit concat must populate fused_qkvz_ba"
        );
        let (fused_qkv, fused_z, fused_a, fused_b) = linear_attention_inputs_fused_qmm(
            TEST_COMPILE_IDENTITY,
            &cfg,
            &x,
            &qkvz_w,
            &ba_w,
            packed.fused_qkvz_ba.as_ref(),
        )
        .expect("matching 4-bit qkvz/ba should fuse");
        let split_qkvz = qw(&x, &qkvz_w);
        let split_ba = qw(&x, &ba_w);
        let (split_qkv, split_z, split_a, split_b) =
            split_packed_qkvz_ba_projection(&cfg, &split_qkvz, &split_ba, 1, seq);
        mlx_sys::eval(&[
            &fused_qkv, &fused_z, &fused_a, &fused_b, &split_qkv, &split_z, &split_a, &split_b,
        ]);
        for (got, want, name) in [
            (&fused_qkv, &split_qkv, "qkv"),
            (&fused_z, &split_z, "z"),
            (&fused_a, &split_a, "a"),
            (&fused_b, &split_b, "b"),
        ] {
            assert_eq!(got.shape(), want.shape(), "{name} shape");
            let g = got.data_f32();
            let w = want.data_f32();
            assert_eq!(g.len(), w.len(), "{name} len");
            for i in 0..g.len() {
                let err = (g[i] - w[i]).abs();
                assert!(
                    err < 2.0e-5,
                    "{name}[{i}] fused {} split {} err {err}",
                    g[i],
                    w[i]
                );
            }
        }
    }

    #[test]
    fn qw_rms_norm_qmm_matches_rms_then_qw() {
        let hidden_data: Vec<f32> = (0..8 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let proj_data: Vec<f32> = (0..96 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0004)
            .collect();
        let x = MlxArray::from_raw_data(
            hidden_data.as_ptr() as *const u8,
            std::mem::size_of_val(hidden_data.as_slice()),
            &[1, 8, 64],
            MlxDtype::Float32,
        );
        let proj_w = MlxArray::from_raw_data(
            proj_data.as_ptr() as *const u8,
            std::mem::size_of_val(proj_data.as_slice()),
            &[96, 64],
            MlxDtype::Float32,
        );
        let dq = mlx_sys::quantize(
            &proj_w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let proj = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let norm_data = vec![1.0f32; 64];
        let norm_w = MlxArray::from_raw_data(
            norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(norm_data.as_slice()),
            &[64],
            MlxDtype::Float32,
        );
        let fused = qw_rms_norm_qmm(&x, &norm_w, 1e-6, &proj);
        let portable = qw(&rms_norm(&x, Some(&norm_w), 1e-6, None), &proj);
        eval(&[&fused, &portable]);
        assert_eq!(fused.shape(), portable.shape());
        let g = fused.data_f32();
        let w = portable.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..g.len() {
            max_abs = max_abs.max((g[i] - w[i]).abs());
        }
        assert!(
            max_abs < 3.0e-2,
            "LA norm+qmm fuse must match rms then qw, max_abs={max_abs}"
        );
        assert!(
            fastpath::should_qwen_la_norm_qkvz_fuse_for(true, "qwen3_5", 1024),
            "shipped LA norm fuse gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn qw_rms_norm_qmm_mxfp4_matches_rms_then_qw() {
        let hidden_data: Vec<f32> = (0..8 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let proj_data: Vec<f32> = (0..96 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0004)
            .collect();
        let x = MlxArray::from_raw_data(
            hidden_data.as_ptr() as *const u8,
            std::mem::size_of_val(hidden_data.as_slice()),
            &[1, 8, 64],
            MlxDtype::Float32,
        );
        let proj_w = MlxArray::from_raw_data(
            proj_data.as_ptr() as *const u8,
            std::mem::size_of_val(proj_data.as_slice()),
            &[96, 64],
            MlxDtype::Float32,
        );
        let dq = mlx_sys::quantize(
            &proj_w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        assert_eq!(dq.len(), 2, "mxfp4 quant returns [packed, scales]");
        let proj = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let norm_data = vec![1.0f32; 64];
        let norm_w = MlxArray::from_raw_data(
            norm_data.as_ptr() as *const u8,
            std::mem::size_of_val(norm_data.as_slice()),
            &[64],
            MlxDtype::Float32,
        );
        let fused = qw_rms_norm_qmm(&x, &norm_w, 1e-6, &proj);
        let portable = qw(&rms_norm(&x, Some(&norm_w), 1e-6, None), &proj);
        eval(&[&fused, &portable]);
        assert_eq!(fused.shape(), portable.shape());
        let g = fused.data_f32();
        let w = portable.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..g.len() {
            max_abs = max_abs.max((g[i] - w[i]).abs());
        }
        assert!(
            max_abs < 3.0e-2,
            "MXFP4 LA norm+qmm must use portable qw, max_abs={max_abs}"
        );
    }

    #[test]
    fn qwen_prefill_contiguous_la_input_qw_matches_view() {
        let full: Vec<f32> = (0..16 * 64)
            .map(|i| ((i as f32) - 512.0) * 0.0009765625)
            .collect();
        let proj_data: Vec<f32> = (0..96 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0004)
            .collect();
        let wide = MlxArray::from_raw_data(
            full.as_ptr() as *const u8,
            std::mem::size_of_val(full.as_slice()),
            &[1, 16, 64],
            MlxDtype::Float32,
        );
        let view = slice(&wide, &[0, 4, 0], &[1, 12, 64], &[1, 1, 1], None);
        let proj_w = MlxArray::from_raw_data(
            proj_data.as_ptr() as *const u8,
            std::mem::size_of_val(proj_data.as_slice()),
            &[96, 64],
            MlxDtype::Float32,
        );
        let dq = mlx_sys::quantize(
            &proj_w,
            Some(32),
            Some(4),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
            None,
        );
        let proj = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q2_weight: None,
            decode_q2_scales: None,
            decode_q2_biases: None,
        };
        let packed = qw(&contiguous(&view, None), &proj);
        let portable = qw(&view, &proj);
        eval(&[&packed, &portable]);
        assert_eq!(packed.shape(), portable.shape());
        let g = packed.data_f32();
        let w = portable.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..g.len() {
            max_abs = max_abs.max((g[i] - w[i]).abs());
        }
        assert!(
            max_abs < 3.0e-2,
            "contiguous LA input qmm must match view qmm, max_abs={max_abs}"
        );
        assert!(
            fastpath::should_qwen_prefill_contiguous_la_input_for(true, "qwen3_5", 1024),
            "shipped LA input contiguous gate must accept the p2048 chunk length"
        );
    }
}
