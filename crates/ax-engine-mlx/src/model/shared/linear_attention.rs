use mlx_sys::{
    MlxArray, MlxDtype, concatenate, qwen_linear_attention_inputs_packed,
    qwen_linear_attention_post_input, reshape, slice_last_dim, zeros,
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
    gated_delta_kernel, linear_attention_conv1d, linear_attention_decode_post_input_metal,
    normalize_linear_attention_qk, rms_norm_gated_with_full_gate_policy,
    split_linear_attention_qkv,
};
use crate::weights::{LayerWeights, LinearAttentionWeights};

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
    // g and beta are computed inside the Metal kernel (fused) instead of as separate
    // lazy MLX ops, eliminating ~8 kernel dispatches per layer.
    let (out, new_recurrent_state) =
        gated_delta_kernel(&q, &k, &v, &a_log_f32, &a, &dt_bias_f32, &b, &state);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Recurrent,
        profile_started,
        &[&out, &new_recurrent_state],
    );
    cache.set_linear_state(layer_idx, new_conv_state, new_recurrent_state);

    let profile_started = Instant::now();
    let out = rms_norm_gated_with_full_gate_policy(
        &out,
        &z,
        &linear_w.norm,
        cfg.rms_norm_eps,
        linear_attention_full_gate_metal_allowed(cfg, linear_w, layer_idx),
    );
    let flat = reshape(&out, &[1, seq, linear_cfg.value_dim() as i32], None);
    let out = qw(&flat, &linear_w.out_proj);
    linear_attention_profile_eval_elapsed(
        profile_enabled,
        LinearAttentionProfileStage::Output,
        profile_started,
        &[&out],
    );
    out
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
    let batch = x.shape()[0];
    let seq = x.shape()[1];
    debug_assert_eq!(seq, 1, "batched linear-attention forward is decode-only");

    let (qkv, z, a, b) = linear_attention_inputs_batched(linear_cfg, linear_w, x, batch, seq);

    // Snapshot this layer's current per-row state (cloned so the store can be
    // reborrowed mutably for the write-back below). `None` on the first step.
    let (conv_state, recurrent_state) = match lin_state.layer_state(layer_idx) {
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
    lin_state.update_layer(layer_idx, new_conv_state, new_recurrent_state);

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
    if seq == 1 && fastpath::qwen_linear_attention_decode_post_input_metal_enabled() {
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
        let qwen_default_enabled = qwen_linear_attention_direct_cpp_default_family(model_cfg)
            && fastpath::qwen_direct_cpp_linear_attention_inputs_enabled();
        if fastpath::direct_cpp_linear_attention_inputs_enabled() || qwen_default_enabled {
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

fn linear_attention_inputs_packed_direct(
    cfg: &LinearAttentionConfig,
    x: &MlxArray,
    qkvz_w: &crate::weights::QuantizedWeight,
    ba_w: &crate::weights::QuantizedWeight,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    let qkvz_quantized = qkvz_w.scales.is_some();
    let ba_quantized = ba_w.scales.is_some();
    if qkvz_quantized
        && ba_quantized
        && (qkvz_w.group_size != ba_w.group_size || qkvz_w.bits != ba_w.bits)
    {
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
        None,
    )
    .filter(|(qkv, z, a, b)| {
        qkv.shape() == vec![1, x.shape()[1], cfg.conv_dim() as i32]
            && z.shape()
                == vec![
                    1,
                    x.shape()[1],
                    cfg.num_value_heads as i32,
                    cfg.value_head_dim as i32,
                ]
            && a.shape() == vec![1, x.shape()[1], cfg.num_value_heads as i32]
            && b.shape() == vec![1, x.shape()[1], cfg.num_value_heads as i32]
    })
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
}
