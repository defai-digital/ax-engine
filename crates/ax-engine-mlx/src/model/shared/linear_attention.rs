use mlx_sys::{
    MlxArray, MlxDtype, concatenate, contiguous, qwen_linear_attention_inputs_packed,
    qwen_linear_attention_inputs_packed_compiled, qwen_linear_attention_post_input,
    qwen_linear_attention_post_input_compiled, reshape, rms_norm, silu_mul_quantized_matmul, slice,
    slice_last_dim, zeros,
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
    rms_norm_gated_with_full_gate_policy, split_linear_attention_qkv,
};
use crate::weights::{LayerWeights, LinearAttentionWeights, QuantizedWeight};
use std::cell::RefCell;

thread_local! {
    static INITIAL_RECURRENT_ZEROS: RefCell<Option<((i32, i32, i32), MlxArray)>> =
        const { RefCell::new(None) };
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
    let prefix_conv_state = prefix_capture_after
        .and_then(|after| linear_attention_conv_prefix_state(linear_cfg, &qkv, conv_state, after));
    let (q, k, v, new_conv_state) =
        linear_attention_post_input(cfg, linear_cfg, linear_w, &qkv, conv_state, profile_enabled);
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

    let profile_started = Instant::now();
    let value_dim = linear_cfg.value_dim() as i32;
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
            linear_attention_full_gate_metal_allowed(cfg, linear_w, layer_idx),
        );
        let flat = if fastpath::should_skip_unused_la_out_reshape(&out.shape(), seq, value_dim) {
            out
        } else {
            reshape(&out, &[1, seq, value_dim], None)
        };
        qw(&flat, &linear_w.out_proj)
    };
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Output,
        profile_started,
        &[&out],
    );
    out
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
) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
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
        } else if let Some(outputs) = linear_attention_decode_post_input_metal(
            linear_cfg,
            qkv,
            &linear_w.conv1d_dense,
            cached_conv_state,
            linear_cfg.q_scale,
            linear_cfg.k_scale,
            cfg.rms_norm_eps,
        ) {
            record_linear_attention_decode_post_input_metal_hit();
            return outputs;
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
            return outputs;
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
            return outputs;
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
    (q, k, split.v, new_conv_state)
}

pub(crate) fn linear_attention_inputs(
    model_cfg: &ModelConfig,
    cfg: &LinearAttentionConfig,
    w: &crate::weights::LinearAttentionWeights,
    x: &MlxArray,
    seq: i32,
    profile_enabled: bool,
) -> (MlxArray, MlxArray, MlxArray, MlxArray) {
    if let (Some(qkvz_w), Some(ba_w)) = (&w.in_proj_qkvz, &w.in_proj_ba) {
        let (qkvz_w, ba_w) = if fastpath::should_qwen_la_prefill_q2(seq) {
            match (w.prefill_q2_qkvz.as_ref(), w.prefill_q2_ba.as_ref()) {
                (Some(q2_qkvz), Some(q2_ba)) => (q2_qkvz, q2_ba),
                _ => (qkvz_w, ba_w),
            }
        } else {
            (qkvz_w, ba_w)
        };
        if !fastpath::qwen_linear_mtp_exact_enabled()
            && !profile_enabled
            && should_fuse_qkvz_ba_qmm(qkvz_w, ba_w, seq)
            && let Some(outputs) =
                linear_attention_inputs_fused_qmm(cfg, x, qkvz_w, ba_w, w.fused_qkvz_ba.as_ref())
        {
            return outputs;
        }
        let qwen_default_enabled = qwen_linear_attention_direct_cpp_default_family(model_cfg)
            && fastpath::qwen_direct_cpp_linear_attention_inputs_enabled()
            && !fastpath::qwen_linear_mtp_exact_enabled();
        if !fastpath::qwen_linear_mtp_exact_enabled()
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
        let mixed_qkvz = qw(x, qkvz_w);
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
        let mixed_ba = qw(x, ba_w);
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
    fastpath::should_qwen_la_fused_qkvz_ba_qmm(seq, qkvz_w.matching_affine_quant(ba_w))
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
    let mixed = qw(x, fused);
    let (qkvz_out, ba_out) = packed_qkvz_ba_widths(cfg);
    let last = *mixed.shape().last()?;
    if last != qkvz_out + ba_out {
        return None;
    }
    let batch = x.shape().first().copied().unwrap_or(1);
    let seq = x.shape().get(1).copied()?;
    let mixed_qkvz = slice_last_dim(&mixed, 0, qkvz_out, None);
    let mixed_ba = slice_last_dim(&mixed, qkvz_out, qkvz_out + ba_out, None);
    Some(split_packed_qkvz_ba_projection(
        cfg,
        &mixed_qkvz,
        &mixed_ba,
        batch,
        seq,
    ))
}

fn linear_attention_inputs_packed_direct(
    cfg: &LinearAttentionConfig,
    x: &MlxArray,
    qkvz_w: &crate::weights::QuantizedWeight,
    ba_w: &crate::weights::QuantizedWeight,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
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
        && let (Some(qkvz_scales), Some(qkvz_biases), Some(ba_scales), Some(ba_biases)) = (
            qkvz_w.scales.as_ref(),
            qkvz_w.biases.as_ref(),
            ba_w.scales.as_ref(),
            ba_w.biases.as_ref(),
        )
        && let Some(compiled) = qwen_linear_attention_inputs_packed_compiled(
            x,
            &qkvz_w.weight,
            qkvz_scales,
            qkvz_biases,
            &ba_w.weight,
            ba_scales,
            ba_biases,
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
    let (q, k, v, new_conv_state) =
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
    fn mixed_pack_is_prefill_only() {
        assert!(linear_attention_prefill_allows_mixed_pack(128, true));
        assert!(linear_attention_prefill_allows_mixed_pack(512, true));
        assert!(linear_attention_prefill_allows_mixed_pack(2048, true));
        assert!(!linear_attention_prefill_allows_mixed_pack(1, true));
        assert!(linear_attention_prefill_allows_mixed_pack(1, false));
        assert!(linear_attention_prefill_allows_mixed_pack(128, false));
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
                qkvz_qw.biases.as_ref().expect("qkvz biases"),
                &ba_qw.weight,
                ba_qw.scales.as_ref().expect("ba scales"),
                ba_qw.biases.as_ref().expect("ba biases"),
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
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        }
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
}
