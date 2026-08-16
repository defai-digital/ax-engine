use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxClosure, MlxDtype, MlxMetalKernel,
    MlxVectorArray, astype, concatenate, contiguous, conv1d, multiply, reshape, rms_norm, slice,
    slice_last_dim, zeros,
};
#[cfg(test)]
use mlx_sys::{add, exp, less, log1p, negative, sigmoid, where_cond};

use crate::attention_mask::scalar_i32;
use crate::fastpath;
use crate::model::LinearAttentionConfig;

/// Split Qwen3.5 gated-delta conv output into shaped q/k/v tensors.
pub struct LinearAttentionQkv {
    pub q: MlxArray,
    pub k: MlxArray,
    pub v: MlxArray,
}

static GATED_DELTA_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_PREFILL_STREAMING_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_DECODE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static DECODE_POST_INPUT_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static RMS_NORM_GATE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static RMS_NORM_FULL_GATE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
type GatedDeltaPrefillCompileKey = (i32, i32, i32, i32, i32, i32, ThreadId);
type GatedDeltaPrefillCompileCache =
    Mutex<HashMap<GatedDeltaPrefillCompileKey, Option<MlxClosure>>>;
static GATED_DELTA_PREFILL_COMPILE_CACHE: OnceLock<GatedDeltaPrefillCompileCache> = OnceLock::new();
pub(crate) const GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY: usize = 512;
pub(crate) const GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY: usize = 1024;
pub(crate) const GATED_DELTA_THREADGROUP_CACHE_CAPACITY: usize = 2048;
/// Chunkwise prefill tile: 256-token no-copy views (not the closed 512 TG tile).
pub(crate) const GATED_DELTA_CHUNKWISE_TILE: usize = 256;

/// Runner prefill-chunk cap for linear-attention families.
///
/// Production clamp is 1024. 1536 remasured community p2048 ~899 vs 908.5.
/// One 2048 FFN + tile-512 remasured 889.96 vs 891.02 (2026-08-13).
/// Streaming / `AX_MLX_QWEN_PREFILL_SINGLE_2048=1` still take 2048.
pub(crate) fn linear_attention_prefill_chunk_cap(streaming: bool) -> usize {
    if streaming || fastpath::qwen_prefill_single_2048_enabled() {
        GATED_DELTA_THREADGROUP_CACHE_CAPACITY
    } else if fastpath::qwen_prefill_chunk_1536_enabled() {
        1536
    } else if fastpath::qwen_prefill_chunk_1280_enabled() {
        1280
    } else {
        GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY
    }
}

/// seq>512 is eligible for the 512 TG tile path (p2048's 1024 chunks).
pub(crate) fn gated_delta_prefill_tile_512_seq_eligible(seq: i32) -> bool {
    seq > GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32
}

/// compute_g from mlx-lm/mlx-swift-lm:
/// `exp(-exp(A_log.float32) * softplus(a + dt_bias))`.
///
/// Production prefill/decode fold this into Metal; this MLX-ops form is the
/// unit/oracle reference for softplus and dtype contracts.
#[cfg(test)]
pub(crate) fn compute_gated_delta_g(
    a_log: &MlxArray,
    a: &MlxArray,
    dt_bias: &MlxArray,
) -> MlxArray {
    let a_log_f32 = astype(a_log, MlxDtype::Float32, None);
    let decay_rate = exp(&a_log_f32, None);
    let a_plus_bias = add(a, dt_bias, None);
    let threshold = scalar_f32_as(20.0, a_plus_bias.dtype());
    let exp_branch = log1p(&exp(&a_plus_bias, None), None);
    let softplus = where_cond(
        &less(&threshold, &a_plus_bias, None),
        &a_plus_bias,
        &exp_branch,
        None,
    );
    let decay = multiply(&decay_rate, &softplus, None);
    let g = exp(&negative(&decay, None), None);
    // Keep g in float32 for the recurrent state update (matches mlx_lm).
    astype(&g, MlxDtype::Float32, None)
}

/// `beta = sigmoid(b)` with the bf16/activation rounding contract used by the
/// fused Metal kernel: compute in float, cast through the activation dtype,
/// then promote back to float32 for the recurrent update.
///
/// Kept for unit/oracle comparisons; production prefill fuses beta inside the
/// streaming Metal kernel.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn compute_gated_delta_beta(b_raw: &MlxArray) -> MlxArray {
    let beta = sigmoid(b_raw, None);
    let beta_act = astype(&beta, b_raw.dtype(), None);
    astype(&beta_act, MlxDtype::Float32, None)
}

/// Compile the linear-attention Metal kernel specializations a decode
/// step will hit, at production head dimensions, before the first
/// request arrives. MLX builds each MSL→pipeline specialization lazily
/// on the first eval that materializes it, so without this the compile
/// stall lands inside the first request's latency. Shapes mirror the
/// decode path exactly (batch 1, seq 1, cfg head dims) because template
/// specialization is shape-keyed — warming a different shape compiles a
/// different pipeline. Best-effort by contract: the caller logs and
/// continues on `Err`.
pub fn warm_gated_delta_decode_kernels(cfg: &LinearAttentionConfig) -> Result<(), String> {
    // Mirror the fused kernel's own precondition: configs below it never
    // dispatch the custom kernel in production either, so there is
    // nothing to warm.
    if !cfg.key_head_dim.is_multiple_of(32) {
        return Ok(());
    }
    let key_heads = cfg.num_key_heads as i32;
    let key_dim = cfg.key_head_dim as i32;
    let value_heads = cfg.num_value_heads as i32;
    let value_dim = cfg.value_head_dim as i32;

    let q = zeros(&[1, 1, key_heads, key_dim], MlxDtype::Float32, None);
    let k = zeros(&[1, 1, key_heads, key_dim], MlxDtype::Float32, None);
    let v = zeros(&[1, 1, value_heads, value_dim], MlxDtype::Float32, None);
    let a_log = zeros(&[value_heads], MlxDtype::Float32, None);
    let a_raw = zeros(&[1, 1, value_heads], MlxDtype::Float32, None);
    let dt_bias = zeros(&[value_heads], MlxDtype::Float32, None);
    let b_raw = zeros(&[1, 1, value_heads], MlxDtype::Float32, None);
    let state = zeros(
        &[1, value_heads, value_dim, key_dim],
        MlxDtype::Float32,
        None,
    );
    let (y, new_state) = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    mlx_sys::try_eval(&[&y, &new_state])
        .map_err(|error| format!("gated-delta decode warm-up failed: {error}"))?;

    // The per-token conv1d + tail update is the other custom path a
    // decode step touches every token.
    let qkv = zeros(&[1, 1, cfg.conv_dim() as i32], MlxDtype::Float32, None);
    let conv_weight = zeros(
        &[cfg.conv_dim() as i32, cfg.conv_kernel_dim as i32, 1],
        MlxDtype::Float32,
        None,
    );
    let (conv_out, conv_tail) = linear_attention_conv1d(cfg, &qkv, &conv_weight, None);
    mlx_sys::try_eval(&[&conv_out, &conv_tail])
        .map_err(|error| format!("linear-attention conv1d warm-up failed: {error}"))?;
    Ok(())
}

/// Apply Qwen3.5's depthwise conv over `[cached_tail, qkv]`.
///
/// Inputs/outputs follow mlx-lm and mlx-swift-lm:
/// - `qkv`: `[1, seq, conv_dim]`
/// - `cached_conv_state`: `[1, conv_kernel_dim - 1, conv_dim]`
/// - `conv_weight`: `[conv_dim, conv_kernel_dim, 1]`
/// - returns `(silu(conv1d(...)), new_tail)`
pub fn linear_attention_conv1d(
    cfg: &LinearAttentionConfig,
    qkv: &MlxArray,
    conv_weight: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let shape = qkv.shape();
    let batch = shape[0];
    let conv_dim = cfg.conv_dim() as i32;
    let tail_len = cfg.conv_kernel_dim as i32 - 1;
    let dtype = qkv.dtype();

    let conv_state = cached_conv_state
        .cloned()
        .unwrap_or_else(|| zeros(&[batch, tail_len, conv_dim], dtype, None));
    let conv_input = concatenate(&[&conv_state, qkv], 1, None);
    let total = conv_input.shape()[1];
    let new_state = slice(
        &conv_input,
        &[0, total - tail_len, 0],
        &[batch, total, conv_dim],
        &[1, 1, 1],
        None,
    );
    let conv_out = conv1d(&conv_input, conv_weight, 1, 0, 1, conv_dim, None);
    (mlx_sys::ops::silu(&conv_out, None), new_state)
}

#[allow(clippy::too_many_arguments)]
pub fn linear_attention_decode_post_input_metal(
    cfg: &LinearAttentionConfig,
    qkv: &MlxArray,
    conv_weight: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    q_scale: f32,
    k_scale: f32,
    eps: f32,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    let qkv_shape = qkv.shape();
    if qkv_shape.len() != 3 {
        return None;
    }
    let seq = qkv_shape[1];
    if seq < 1 || seq > GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32 {
        return None;
    }
    if cfg.key_head_dim != cfg.value_head_dim {
        return None;
    }
    if !cfg.key_head_dim.is_power_of_two() || cfg.key_head_dim > 256 {
        return None;
    }
    if cfg.conv_kernel_dim < 1 {
        return None;
    }
    let batch = qkv_shape[0];
    let conv_dim = cfg.conv_dim() as i32;
    let tail_len = cfg.conv_kernel_dim as i32 - 1;
    if qkv_shape[2] != conv_dim {
        return None;
    }
    let zero_state;
    let conv_state = if let Some(state) = cached_conv_state {
        if state.shape() != vec![batch, tail_len, conv_dim] {
            return None;
        }
        state
    } else {
        // First prefill chunk has no cached conv state. Zeros match the
        // portable conv1d cold start so Metal can engage on chunk 1.
        zero_state = zeros(&[batch, tail_len, conv_dim], qkv.dtype(), None);
        &zero_state
    };
    if conv_weight.shape() != vec![conv_dim, cfg.conv_kernel_dim as i32, 1] {
        return None;
    }

    let kernel = DECODE_POST_INPUT_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_linear_attention_decode_post_input_v2",
            &[
                "qkv",
                "conv_weight",
                "conv_state",
                "q_scale",
                "k_scale",
                "eps",
            ],
            &["q", "k", "v", "new_conv_state"],
            DECODE_POST_INPUT_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let q_scale_arr = scalar_f32_as(q_scale, MlxDtype::Float32);
    let k_scale_arr = scalar_f32_as(k_scale, MlxDtype::Float32);
    let eps_arr = scalar_f32_as(eps, MlxDtype::Float32);
    let groups = (cfg.num_key_heads * 2 + cfg.num_value_heads) as i32;
    let head_dim = cfg.key_head_dim as i32;
    let outputs = kernel.apply_with_template(
        &[
            qkv,
            conv_weight,
            conv_state,
            &q_scale_arr,
            &k_scale_arr,
            &eps_arr,
        ],
        &[
            KernelOutputSpec {
                shape: vec![batch, seq, cfg.num_key_heads as i32, head_dim],
                dtype: qkv.dtype(),
            },
            KernelOutputSpec {
                shape: vec![batch, seq, cfg.num_key_heads as i32, head_dim],
                dtype: qkv.dtype(),
            },
            KernelOutputSpec {
                shape: vec![batch, seq, cfg.num_value_heads as i32, head_dim],
                dtype: qkv.dtype(),
            },
            KernelOutputSpec {
                shape: vec![batch, tail_len, conv_dim],
                dtype: qkv.dtype(),
            },
        ],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: qkv.dtype(),
            },
            KernelTemplateArg::Int {
                name: "Hk",
                value: cfg.num_key_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "Hv",
                value: cfg.num_value_heads as i32,
            },
            KernelTemplateArg::Int {
                name: "HeadDim",
                value: head_dim,
            },
            KernelTemplateArg::Int {
                name: "ConvKernelDim",
                value: cfg.conv_kernel_dim as i32,
            },
            KernelTemplateArg::Int {
                name: "Seq",
                value: seq,
            },
        ],
        (head_dim, 1, batch * groups),
        (head_dim, 1, 1),
        None,
    );

    let mut outputs = outputs.into_iter();
    Some((
        outputs.next()?,
        outputs.next()?,
        outputs.next()?,
        outputs.next()?,
    ))
}

pub fn split_linear_attention_qkv(
    cfg: &LinearAttentionConfig,
    conv_out: &MlxArray,
) -> LinearAttentionQkv {
    let shape = conv_out.shape();
    let batch = shape[0];
    let seq = shape[1];
    let key_dim = cfg.key_dim() as i32;
    let value_dim = cfg.value_dim() as i32;

    let q = slice_last_dim(conv_out, 0, key_dim, None);
    let k = slice_last_dim(conv_out, key_dim, 2 * key_dim, None);
    let v = slice_last_dim(conv_out, 2 * key_dim, 2 * key_dim + value_dim, None);

    LinearAttentionQkv {
        q: reshape(
            &q,
            &[
                batch,
                seq,
                cfg.num_key_heads as i32,
                cfg.key_head_dim as i32,
            ],
            None,
        ),
        k: reshape(
            &k,
            &[
                batch,
                seq,
                cfg.num_key_heads as i32,
                cfg.key_head_dim as i32,
            ],
            None,
        ),
        v: reshape(
            &v,
            &[
                batch,
                seq,
                cfg.num_value_heads as i32,
                cfg.value_head_dim as i32,
            ],
            None,
        ),
    }
}

/// Qwen3.5 gated-delta Q/K no-scale RMSNorm and scaling.
pub fn normalize_linear_attention_qk(
    cfg: &LinearAttentionConfig,
    q: &MlxArray,
    k: &MlxArray,
    eps: f32,
) -> (MlxArray, MlxArray) {
    let (q_scale, k_scale) = (cfg.q_scale, cfg.k_scale);
    let q_normed = rms_norm(q, None, eps, None);
    let k_normed = rms_norm(k, None, eps, None);
    let q_scale = scalar_f32_as(q_scale, q.dtype());
    let k_scale = scalar_f32_as(k_scale, k.dtype());
    (
        multiply(&q_normed, &q_scale, None),
        multiply(&k_normed, &k_scale, None),
    )
}

pub(crate) fn linear_attention_qk_scale(key_head_dim: usize) -> (f32, f32) {
    // mlx-lm/Swift: q *= inv_scale², k *= inv_scale  (inv_scale = Dk^(-0.5))
    let inv_scale = (key_head_dim as f32).powf(-0.5);
    (inv_scale * inv_scale, inv_scale)
}

#[allow(clippy::too_many_arguments)]
/// Run Qwen3.5's gated-delta recurrent update with the MLX Metal kernel.
///
/// `g = exp(-exp(a_log) * softplus(a_raw + dt_bias))` and `beta = sigmoid(b_raw)` are
/// computed inside the Metal kernel rather than as separate MLX ops, eliminating 8 lazy
/// graph nodes per GatedDeltaNet layer (~216 kernel dispatches/step for Qwen3.5 9B).
///
/// Shapes match mlx-lm/mlx-swift-lm:
/// - `q`, `k`: `[B, T, Hk, Dk]` — activation dtype (InT)
/// - `v`: `[B, T, Hv, Dv]` — activation dtype (InT)
/// - `a_log`: `[Hv]` — float32 (StT); the `A_log` model weight
/// - `a_raw`: `[B, T, Hv]` — activation dtype (InT)
/// - `dt_bias`: `[Hv]` — float32 (StT)
/// - `b_raw`: `[B, T, Hv]` — activation dtype (InT)
/// - `state`: `[B, Hv, Dv, Dk]` — float32 (StT)
/// - returns `(y: [B, T, Hv, Dv], state: [B, Hv, Dv, Dk])`
pub fn gated_delta_kernel(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
) -> (MlxArray, MlxArray) {
    gated_delta_kernel_impl(q, k, v, a_log, a_raw, dt_bias, b_raw, state)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel_with_prefix_checkpoint(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
    checkpoint_after: usize,
) -> (MlxArray, MlxArray, MlxArray) {
    assert_eq!(
        checkpoint_after, 1,
        "lazy gated-delta checkpoint currently supports the first token only"
    );
    let q_shape = q.shape();
    let v_shape = v.shape();
    assert!(
        q_shape[1] > 1,
        "gated-delta prefix checkpoint requires a multi-token sequence"
    );
    let batch = q_shape[0];
    let num_key_heads = q_shape[2];
    let key_head_dim = q_shape[3];
    let num_value_heads = v_shape[2];
    let value_head_dim = v_shape[3];
    assert_eq!(
        batch, 1,
        "lazy gated-delta checkpoint currently supports decode batch 1 only"
    );
    // The decode kernel's pointer arithmetic deliberately addresses the first
    // sequence row and its output spec fixes T=1. The speculative verifier's
    // q/k/v/a/b tensors are contiguous, so passing the full T>1 buffers avoids
    // five slice+contiguous materialisations per linear-attention layer while
    // preserving exactly the same row-0 arithmetic.
    let (_, checkpoint) = gated_delta_decode_kernel(
        q,
        k,
        v,
        a_log,
        a_raw,
        dt_bias,
        b_raw,
        state,
        batch,
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        state.shape(),
    );
    let (output, final_state) = gated_delta_kernel(q, k, v, a_log, a_raw, dt_bias, b_raw, state);
    (output, final_state, checkpoint)
}

/// Run GatedDelta as sequential `tile`-length TG kernels, carrying state.
///
/// Production uses `tile = 512` (default-ON) so a 1024-token chunk keeps
/// the winning short TG specialization. The 2048 TG tier lost ~15% vs 512;
/// tiling 1024 as two 512 kernels is the unused recurrent A/B. `tile = 1024`
/// remains the fallback when the 512 tile flag is off.
#[allow(clippy::too_many_arguments)]
fn gated_delta_prefill_tiled(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
    tile: i32,
) -> (MlxArray, MlxArray) {
    assert!(tile > 0, "gated_delta prefill tile must be positive");
    let q_shape = q.shape();
    let v_shape = v.shape();
    let batch = q_shape[0];
    let seq = q_shape[1];
    let num_key_heads = q_shape[2];
    let key_head_dim = q_shape[3];
    let num_value_heads = v_shape[2];
    let value_head_dim = v_shape[3];
    let mut state_cur = state.clone();
    let mut ys: Vec<MlxArray> = Vec::new();
    let mut start = 0i32;
    while start < seq {
        let end = (start + tile).min(seq);
        let q_t = contiguous(
            &slice(
                q,
                &[0, start, 0, 0],
                &[batch, end, num_key_heads, key_head_dim],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        );
        let k_t = contiguous(
            &slice(
                k,
                &[0, start, 0, 0],
                &[batch, end, num_key_heads, key_head_dim],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        );
        let v_t = contiguous(
            &slice(
                v,
                &[0, start, 0, 0],
                &[batch, end, num_value_heads, value_head_dim],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        );
        let a_t = contiguous(
            &slice(
                a_raw,
                &[0, start, 0],
                &[batch, end, num_value_heads],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let b_t = contiguous(
            &slice(
                b_raw,
                &[0, start, 0],
                &[batch, end, num_value_heads],
                &[1, 1, 1],
                None,
            ),
            None,
        );
        let (y_t, next_state) =
            gated_delta_kernel_impl(&q_t, &k_t, &v_t, a_log, &a_t, dt_bias, &b_t, &state_cur);
        ys.push(y_t);
        state_cur = next_state;
        start = end;
    }
    let refs: Vec<&MlxArray> = ys.iter().collect();
    (concatenate(&refs, 1, None), state_cur)
}

/// GatedDelta prefill as no-copy 256-token chunks.
///
/// Distinct from [`gated_delta_prefill_tiled`]: B=1 production slices stay
/// views (no `contiguous` copy of q/k/v/a/b per tile). Tile length is 256,
/// not the closed 512 TG specialization. State still carries sequentially —
/// GatedDelta's rank-1 map is not a scalar decay, so independent chunks
/// would be numerically wrong.
#[allow(clippy::too_many_arguments)]
fn gated_delta_prefill_chunkwise(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
    tile: i32,
) -> (MlxArray, MlxArray) {
    assert!(
        tile > 0,
        "gated_delta prefill chunkwise tile must be positive"
    );
    let q_shape = q.shape();
    let v_shape = v.shape();
    let batch = q_shape[0];
    let seq = q_shape[1];
    let num_key_heads = q_shape[2];
    let key_head_dim = q_shape[3];
    let num_value_heads = v_shape[2];
    let value_head_dim = v_shape[3];
    let skip_copy = batch == 1;
    let mut state_cur = state.clone();
    let mut ys: Vec<MlxArray> = Vec::new();
    let mut start = 0i32;
    while start < seq {
        let end = (start + tile).min(seq);
        let q_view = slice(
            q,
            &[0, start, 0, 0],
            &[batch, end, num_key_heads, key_head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let k_view = slice(
            k,
            &[0, start, 0, 0],
            &[batch, end, num_key_heads, key_head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let v_view = slice(
            v,
            &[0, start, 0, 0],
            &[batch, end, num_value_heads, value_head_dim],
            &[1, 1, 1, 1],
            None,
        );
        let a_view = slice(
            a_raw,
            &[0, start, 0],
            &[batch, end, num_value_heads],
            &[1, 1, 1],
            None,
        );
        let b_view = slice(
            b_raw,
            &[0, start, 0],
            &[batch, end, num_value_heads],
            &[1, 1, 1],
            None,
        );
        let (q_t, k_t, v_t, a_t, b_t) = if skip_copy {
            (q_view, k_view, v_view, a_view, b_view)
        } else {
            (
                contiguous(&q_view, None),
                contiguous(&k_view, None),
                contiguous(&v_view, None),
                contiguous(&a_view, None),
                contiguous(&b_view, None),
            )
        };
        let (y_t, next_state) =
            gated_delta_kernel_impl(&q_t, &k_t, &v_t, a_log, &a_t, dt_bias, &b_t, &state_cur);
        ys.push(y_t);
        state_cur = next_state;
        start = end;
    }
    let refs: Vec<&MlxArray> = ys.iter().collect();
    (concatenate(&refs, 1, None), state_cur)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_kernel_impl(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
) -> (MlxArray, MlxArray) {
    let q_shape = q.shape();
    let v_shape = v.shape();
    let state_shape = state.shape();
    let batch = q_shape[0];
    let seq = q_shape[1];
    let num_key_heads = q_shape[2];
    let key_head_dim = q_shape[3];
    let num_value_heads = v_shape[2];
    let value_head_dim = v_shape[3];
    let q_c;
    let k_c;
    let v_c;
    let a_c;
    let b_c;
    let (q, k, v, a_raw, b_raw) = if fastpath::should_qwen_gated_delta_prefill_contiguous(seq) {
        q_c = contiguous(q, None);
        k_c = contiguous(k, None);
        v_c = contiguous(v, None);
        a_c = contiguous(a_raw, None);
        b_c = contiguous(b_raw, None);
        (&q_c, &k_c, &v_c, &a_c, &b_c)
    } else {
        (q, k, v, a_raw, b_raw)
    };
    if seq == 1 && fastpath::qwen_gated_delta_decode_metal_enabled() {
        return gated_delta_decode_kernel(
            q,
            k,
            v,
            a_log,
            a_raw,
            dt_bias,
            b_raw,
            state,
            batch,
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            state_shape,
        );
    }
    // Multi-token prefill hybrid:
    // - default: 512 TG oneshot, or tile at 512 when seq>512 so p2048's
    //   two 1024 chunks keep the winning short specialization.
    // - opt-in streaming (seq>512): no CacheCapacity TG array.
    // - tile-at-1024 only when the 512 tile flag is off (seq>1024).
    if seq > GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32
        && fastpath::qwen_gated_delta_prefill_streaming_enabled()
    {
        return gated_delta_prefill_streaming_kernel(
            q,
            k,
            v,
            a_log,
            a_raw,
            dt_bias,
            b_raw,
            state,
            batch,
            seq,
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            state_shape,
        );
    }
    if gated_delta_prefill_tile_512_seq_eligible(seq)
        && fastpath::qwen_gated_delta_prefill_tile_512_enabled()
    {
        return gated_delta_prefill_tiled(
            q,
            k,
            v,
            a_log,
            a_raw,
            dt_bias,
            b_raw,
            state,
            GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32,
        );
    }
    if seq > GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32 {
        return gated_delta_prefill_tiled(
            q,
            k,
            v,
            a_log,
            a_raw,
            dt_bias,
            b_raw,
            state,
            GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32,
        );
    }
    if fastpath::should_qwen_gd_prefill_chunkwise(seq) {
        return gated_delta_prefill_chunkwise(
            q,
            k,
            v,
            a_log,
            a_raw,
            dt_bias,
            b_raw,
            state,
            GATED_DELTA_CHUNKWISE_TILE as i32,
        );
    }
    let seq_i32 = scalar_i32(seq);
    assert!(
        seq <= GATED_DELTA_THREADGROUP_CACHE_CAPACITY as i32,
        "gated_delta_kernel t_len ({seq}) exceeds threadgroup cache capacity ({GATED_DELTA_THREADGROUP_CACHE_CAPACITY})"
    );
    // Three-tier CacheCapacity. The 2048 tier loses ~15% per-token vs 512 on
    // Qwen 3.6 27B (Hv=48); seq>1024 tiles at 1024 instead of this branch.
    let cache_capacity = if seq <= GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32 {
        GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32
    } else if seq <= GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32 {
        GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32
    } else {
        GATED_DELTA_THREADGROUP_CACHE_CAPACITY as i32
    };
    // The Metal kernel uses `constexpr int n_per_t = Dk / 32` (integer division over
    // 32 SIMD lanes).  If key_head_dim is not divisible by 32, the remainder is silently
    // dropped and the state update is mathematically wrong.
    assert!(
        key_head_dim % 32 == 0,
        "gated_delta_kernel requires key_head_dim divisible by 32 (got {key_head_dim})"
    );
    // The kernel GQA mapping is `hk_idx = hv_idx / (Hv / Hk)` (integer division).
    // If num_value_heads is not a multiple of num_key_heads the mapping truncates
    // silently and every affected value head reads the wrong key/query slice.
    assert!(
        num_key_heads > 0 && num_value_heads % num_key_heads == 0,
        "gated_delta_kernel requires num_value_heads to be a multiple of num_key_heads \
         (got {num_value_heads} value heads, {num_key_heads} key heads)"
    );

    let kernel = GATED_DELTA_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "qwen35_gated_delta_v3",
            &[
                "q", "k", "v", "a_log", "a_raw", "dt_bias", "b_raw", "state_in", "seq_len",
            ],
            &["y", "state_out"],
            GATED_DELTA_KERNEL_SOURCE,
            "",
            true,
        )
    });
    if let Some(compiled) = try_compiled_gated_delta_oneshot(
        q,
        k,
        v,
        a_log,
        a_raw,
        dt_bias,
        b_raw,
        state,
        &seq_i32,
        batch,
        seq,
        num_key_heads,
        key_head_dim,
        num_value_heads,
        value_head_dim,
        cache_capacity,
        &state_shape,
    ) {
        return compiled;
    }
    let outputs = kernel.apply_with_template(
        &[q, k, v, a_log, a_raw, dt_bias, b_raw, state, &seq_i32],
        &[
            KernelOutputSpec {
                shape: vec![batch, seq, num_value_heads, value_head_dim],
                dtype: q.dtype(),
            },
            KernelOutputSpec {
                shape: state_shape,
                dtype: state.dtype(),
            },
        ],
        &[
            KernelTemplateArg::Dtype {
                name: "InT",
                dtype: q.dtype(),
            },
            KernelTemplateArg::Dtype {
                name: "StT",
                dtype: state.dtype(),
            },
            KernelTemplateArg::Int {
                name: "Dk",
                value: key_head_dim,
            },
            KernelTemplateArg::Int {
                name: "Dv",
                value: value_head_dim,
            },
            KernelTemplateArg::Int {
                name: "Hk",
                value: num_key_heads,
            },
            KernelTemplateArg::Int {
                name: "Hv",
                value: num_value_heads,
            },
            KernelTemplateArg::Int {
                name: "CacheCapacity",
                value: cache_capacity,
            },
        ],
        (32, value_head_dim, batch * num_value_heads),
        (32, 4, 1),
        None,
    );

    let mut outputs = outputs.into_iter();
    (
        outputs.next().expect("gated delta y output"),
        outputs.next().expect("gated delta state output"),
    )
}

#[allow(clippy::too_many_arguments)]
fn try_compiled_gated_delta_oneshot(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
    seq_i32: &MlxArray,
    batch: i32,
    seq: i32,
    num_key_heads: i32,
    key_head_dim: i32,
    num_value_heads: i32,
    value_head_dim: i32,
    cache_capacity: i32,
    state_shape: &[i32],
) -> Option<(MlxArray, MlxArray)> {
    if !fastpath::should_qwen_compiled_gated_delta_prefill(seq) {
        return None;
    }
    let kernel = GATED_DELTA_KERNEL.get()?;
    let q_dtype = q.dtype();
    let state_dtype = state.dtype();
    let y_shape = vec![batch, seq, num_value_heads, value_head_dim];
    let state_shape = state_shape.to_vec();
    let key = (
        seq,
        key_head_dim,
        value_head_dim,
        num_key_heads,
        num_value_heads,
        cache_capacity,
        std::thread::current().id(),
    );
    let cache = GATED_DELTA_PREFILL_COMPILE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().ok()?;
    let slot = guard.entry(key).or_insert_with(|| {
        let y_shape_c = y_shape.clone();
        let state_shape_c = state_shape.clone();
        let body = move |inputs: &MlxVectorArray| {
            let q = inputs.get(0);
            let k = inputs.get(1);
            let v = inputs.get(2);
            let a_log = inputs.get(3);
            let a_raw = inputs.get(4);
            let dt_bias = inputs.get(5);
            let b_raw = inputs.get(6);
            let state = inputs.get(7);
            let seq_len = inputs.get(8);
            kernel.apply_with_template(
                &[
                    &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, &seq_len,
                ],
                &[
                    KernelOutputSpec {
                        shape: y_shape_c.clone(),
                        dtype: q_dtype,
                    },
                    KernelOutputSpec {
                        shape: state_shape_c.clone(),
                        dtype: state_dtype,
                    },
                ],
                &[
                    KernelTemplateArg::Dtype {
                        name: "InT",
                        dtype: q_dtype,
                    },
                    KernelTemplateArg::Dtype {
                        name: "StT",
                        dtype: state_dtype,
                    },
                    KernelTemplateArg::Int {
                        name: "Dk",
                        value: key_head_dim,
                    },
                    KernelTemplateArg::Int {
                        name: "Dv",
                        value: value_head_dim,
                    },
                    KernelTemplateArg::Int {
                        name: "Hk",
                        value: num_key_heads,
                    },
                    KernelTemplateArg::Int {
                        name: "Hv",
                        value: num_value_heads,
                    },
                    KernelTemplateArg::Int {
                        name: "CacheCapacity",
                        value: cache_capacity,
                    },
                ],
                (32, value_head_dim, batch * num_value_heads),
                (32, 4, 1),
                None,
            )
        };
        MlxClosure::new_dyn(body).compile(false).ok()
    });
    let closure = slot.as_ref()?;
    let outputs = closure
        .try_apply(&[q, k, v, a_log, a_raw, dt_bias, b_raw, state, seq_i32])
        .ok()?;
    if outputs.len() != 2 {
        return None;
    }
    let mut outputs = outputs.into_iter();
    Some((outputs.next()?, outputs.next()?))
}

/// Multi-token GatedDelta prefill without a CacheCapacity-sized TG cache.
/// Fuses g/beta each timestep (same math as the decode kernel) so long
/// prompts keep high SM occupancy and skip the separate MLX precompute graph.
#[allow(clippy::too_many_arguments)]
fn gated_delta_prefill_streaming_kernel(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
    batch: i32,
    seq: i32,
    num_key_heads: i32,
    key_head_dim: i32,
    num_value_heads: i32,
    value_head_dim: i32,
    state_shape: Vec<i32>,
) -> (MlxArray, MlxArray) {
    assert!(
        key_head_dim % 32 == 0,
        "gated_delta_kernel requires key_head_dim divisible by 32 (got {key_head_dim})"
    );
    assert!(
        num_key_heads > 0 && num_value_heads % num_key_heads == 0,
        "gated_delta_kernel requires num_value_heads to be a multiple of num_key_heads \
         (got {num_value_heads} value heads, {num_key_heads} key heads)"
    );

    let seq_i32 = scalar_i32(seq);

    let kernel = GATED_DELTA_PREFILL_STREAMING_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "qwen35_gated_delta_prefill_streaming_v2",
            &[
                "q", "k", "v", "a_log", "a_raw", "dt_bias", "b_raw", "state_in", "seq_len",
            ],
            &["y", "state_out"],
            GATED_DELTA_PREFILL_STREAMING_KERNEL_SOURCE,
            "",
            true,
        )
    });

    let outputs = kernel.apply_with_template(
        &[q, k, v, a_log, a_raw, dt_bias, b_raw, state, &seq_i32],
        &[
            KernelOutputSpec {
                shape: vec![batch, seq, num_value_heads, value_head_dim],
                dtype: q.dtype(),
            },
            KernelOutputSpec {
                shape: state_shape,
                dtype: state.dtype(),
            },
        ],
        &[
            KernelTemplateArg::Dtype {
                name: "InT",
                dtype: q.dtype(),
            },
            KernelTemplateArg::Dtype {
                name: "StT",
                dtype: state.dtype(),
            },
            KernelTemplateArg::Int {
                name: "Dk",
                value: key_head_dim,
            },
            KernelTemplateArg::Int {
                name: "Dv",
                value: value_head_dim,
            },
            KernelTemplateArg::Int {
                name: "Hk",
                value: num_key_heads,
            },
            KernelTemplateArg::Int {
                name: "Hv",
                value: num_value_heads,
            },
        ],
        (32, value_head_dim, batch * num_value_heads),
        (32, 4, 1),
        None,
    );

    let mut outputs = outputs.into_iter();
    (
        outputs
            .next()
            .expect("gated delta streaming prefill y output"),
        outputs
            .next()
            .expect("gated delta streaming prefill state output"),
    )
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_decode_kernel(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
    batch: i32,
    num_key_heads: i32,
    key_head_dim: i32,
    num_value_heads: i32,
    value_head_dim: i32,
    state_shape: Vec<i32>,
) -> (MlxArray, MlxArray) {
    assert!(
        key_head_dim % 32 == 0,
        "gated_delta_kernel requires key_head_dim divisible by 32 (got {key_head_dim})"
    );
    assert!(
        num_key_heads > 0 && num_value_heads % num_key_heads == 0,
        "gated_delta_kernel requires num_value_heads to be a multiple of num_key_heads \
         (got {num_value_heads} value heads, {num_key_heads} key heads)"
    );

    let kernel = GATED_DELTA_DECODE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "qwen35_gated_delta_decode_v1",
            &[
                "q", "k", "v", "a_log", "a_raw", "dt_bias", "b_raw", "state_in",
            ],
            &["y", "state_out"],
            GATED_DELTA_DECODE_KERNEL_SOURCE,
            "",
            true,
        )
    });

    let outputs = kernel.apply_with_template(
        &[q, k, v, a_log, a_raw, dt_bias, b_raw, state],
        &[
            KernelOutputSpec {
                shape: vec![batch, 1, num_value_heads, value_head_dim],
                dtype: q.dtype(),
            },
            KernelOutputSpec {
                shape: state_shape,
                dtype: state.dtype(),
            },
        ],
        &[
            KernelTemplateArg::Dtype {
                name: "InT",
                dtype: q.dtype(),
            },
            KernelTemplateArg::Dtype {
                name: "StT",
                dtype: state.dtype(),
            },
            KernelTemplateArg::Int {
                name: "Dk",
                value: key_head_dim,
            },
            KernelTemplateArg::Int {
                name: "Dv",
                value: value_head_dim,
            },
            KernelTemplateArg::Int {
                name: "Hk",
                value: num_key_heads,
            },
            KernelTemplateArg::Int {
                name: "Hv",
                value: num_value_heads,
            },
        ],
        (32, value_head_dim, batch * num_value_heads),
        (32, 4, 1),
        None,
    );

    let mut outputs = outputs.into_iter();
    (
        outputs.next().expect("gated delta decode y output"),
        outputs.next().expect("gated delta decode state output"),
    )
}

/// Qwen3Next/Qwen3.5 gated RMSNorm: `silu(gate.float32) * rms_norm(x).float32`.
#[cfg(test)]
fn rms_norm_gated(
    hidden_states: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
) -> MlxArray {
    rms_norm_gated_with_full_gate_policy(hidden_states, gate, weight, eps, true)
}

pub fn rms_norm_gated_with_full_gate_policy(
    hidden_states: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
    allow_full_gate_metal: bool,
) -> MlxArray {
    if allow_full_gate_metal
        && !skip_rms_norm_gate_metal_for_exact_verify()
        && let Some(gated) = rms_norm_full_gate_metal(hidden_states, gate, weight, eps)
    {
        return gated;
    }
    let normed = rms_norm(hidden_states, Some(weight), eps, None);
    if let Some(gated) = rms_norm_gate_metal(&normed, gate, hidden_states.dtype()) {
        return gated;
    }
    portable_rms_norm_gated(hidden_states, gate, weight, eps)
}

/// Uncompiled exact-identity RMSNorm + f32 SiLU*norm graph.
///
/// Metal fused/elementwise gates are not sequence-equivalent under exact
/// MTP-on. This is the matching portable chain; compile only fuses it.
fn portable_rms_norm_gated(
    hidden_states: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
) -> MlxArray {
    let normed = rms_norm(hidden_states, Some(weight), eps, None);
    let gate_f32 = astype(gate, MlxDtype::Float32, None);
    let normed_f32 = astype(&normed, MlxDtype::Float32, None);
    let gated = multiply(&mlx_sys::ops::silu(&gate_f32, None), &normed_f32, None);
    astype(&gated, hidden_states.dtype(), None)
}

fn skip_rms_norm_gate_metal_for_exact_verify() -> bool {
    // Factory MXFP4: Metal gate on MTP-on (S=1 or S=2) flips token 41 vs
    // MTP-off. Portable silu*norm for the whole exact request matches.
    fastpath::qwen_linear_mtp_exact_enabled()
}

fn rms_norm_gate_metal(
    normed: &MlxArray,
    gate: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if !fastpath::linear_attention_rms_norm_gate_metal_enabled()
        || skip_rms_norm_gate_metal_for_exact_verify()
    {
        return None;
    }
    rms_norm_gate_metal_impl(normed, gate, output_dtype)
}

fn rms_norm_full_gate_metal(
    hidden_states: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
) -> Option<MlxArray> {
    if !fastpath::linear_attention_rms_norm_gate_metal_enabled()
        || skip_rms_norm_gate_metal_for_exact_verify()
    {
        return None;
    }
    rms_norm_full_gate_metal_impl(hidden_states, gate, weight, eps)
}

fn rms_norm_full_gate_metal_impl(
    hidden_states: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
) -> Option<MlxArray> {
    if !matches!(
        hidden_states.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        gate.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        weight.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let shape = hidden_states.shape();
    if shape != gate.shape() {
        return None;
    }
    let head_dim = *shape.last()?;
    if !(1..=256).contains(&head_dim) {
        return None;
    }
    if weight.shape() != vec![head_dim] {
        return None;
    }
    let element_count = shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if element_count % i64::from(head_dim) != 0 {
        return None;
    }
    let row_count = i32::try_from(element_count / i64::from(head_dim)).ok()?;

    let eps_arr = scalar_f32_as(eps, MlxDtype::Float32);
    let kernel = RMS_NORM_FULL_GATE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_linear_attention_rms_norm_full_gate_v1",
            &["hidden", "gate", "weight", "eps"],
            &["out"],
            RMS_NORM_FULL_GATE_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[hidden_states, gate, weight, &eps_arr],
        &[KernelOutputSpec {
            shape,
            dtype: hidden_states.dtype(),
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: hidden_states.dtype(),
            },
            KernelTemplateArg::Int {
                name: "HeadDim",
                value: head_dim,
            },
        ],
        (256, 1, row_count),
        (256, 1, 1),
        None,
    );
    outputs.pop()
}

fn rms_norm_gate_metal_impl(
    normed: &MlxArray,
    gate: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if !matches!(
        normed.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        gate.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        output_dtype,
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let shape = normed.shape();
    if shape != gate.shape() {
        return None;
    }
    let element_count = shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = RMS_NORM_GATE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_linear_attention_rms_norm_gate_v1",
            &["normed", "gate"],
            &["out"],
            RMS_NORM_GATE_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[normed, gate],
        &[KernelOutputSpec {
            shape,
            dtype: output_dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: output_dtype,
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
        ],
        (element_count, 1, 1),
        (256, 1, 1),
        None,
    );
    outputs.pop()
}

fn scalar_f32_as(value: f32, dtype: MlxDtype) -> MlxArray {
    let scalar = MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1],
        MlxDtype::Float32,
    );
    astype(&scalar, dtype, None)
}

const RMS_NORM_GATE_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    float gate_v = static_cast<float>(gate[idx]);
    float normed_v = static_cast<float>(normed[idx]);
    // gate_v / (1 + exp(-gate_v)) = gate_v * sigmoid(gate_v) = silu(gate_v)
    float activated = gate_v / (1.0f + exp(-gate_v));
    out[idx] = static_cast<T>(activated * normed_v);
"#;

const RMS_NORM_FULL_GATE_KERNEL_SOURCE: &str = r#"
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_grid.z;
    const uint base = row * HeadDim;

    threadgroup float squares[256];
    float x = 0.0f;
    if (lane < HeadDim) {
        x = static_cast<float>(hidden[base + lane]);
        squares[lane] = x * x;
    } else {
        squares[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            squares[lane] += squares[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < HeadDim) {
        float inv_rms = rsqrt(squares[0] / static_cast<float>(HeadDim) + eps[0]);
        float normed = x * inv_rms * static_cast<float>(weight[lane]);
        float gate_v = static_cast<float>(gate[base + lane]);
        // gate_v / (1 + exp(-gate_v)) = gate_v * sigmoid(gate_v) = silu(gate_v)
        float activated = gate_v / (1.0f + exp(-gate_v));
        out[base + lane] = static_cast<T>(activated * normed);
    }
"#;

const DECODE_POST_INPUT_KERNEL_SOURCE: &str = r#"
    constexpr int KeyDim = Hk * HeadDim;
    constexpr int ValueDim = Hv * HeadDim;
    constexpr int ConvDim = 2 * KeyDim + ValueDim;
    constexpr int TailLen = ConvKernelDim - 1;
    constexpr int Groups = 2 * Hk + Hv;

    const int lane = thread_position_in_threadgroup.x;
    const int z = thread_position_in_grid.z;
    const int batch_idx = z / Groups;
    const int group_idx = z - batch_idx * Groups;

    threadgroup float squares[256];

    int channel = 0;
    bool is_q = group_idx < Hk;
    bool is_k = group_idx >= Hk && group_idx < 2 * Hk;
    if (is_q) {
      channel = group_idx * HeadDim + lane;
    } else if (is_k) {
      channel = KeyDim + (group_idx - Hk) * HeadDim + lane;
    } else {
      channel = 2 * KeyDim + (group_idx - 2 * Hk) * HeadDim + lane;
    }

    auto qkv_b = qkv + batch_idx * Seq * ConvDim;
    auto state_b = conv_state + batch_idx * TailLen * ConvDim;
    auto new_state_b = new_conv_state + batch_idx * TailLen * ConvDim;

    float tail[ConvKernelDim];
    for (int t = 0; t < TailLen; ++t) {
      tail[t] = static_cast<float>(state_b[t * ConvDim + channel]);
    }

    for (int token = 0; token < Seq; ++token) {
      auto qkv_t = qkv_b + token * ConvDim;
      float acc = static_cast<float>(qkv_t[channel]) *
          static_cast<float>(conv_weight[channel * ConvKernelDim + TailLen]);
      for (int t = 0; t < TailLen; ++t) {
        acc += tail[t] *
            static_cast<float>(conv_weight[channel * ConvKernelDim + t]);
      }
      float activated = acc / (1.0f + exp(-acc));

      if (is_q || is_k) {
        squares[lane] = activated * activated;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int stride = HeadDim >> 1; stride > 0; stride >>= 1) {
          if (lane < stride) {
            squares[lane] += squares[lane + stride];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float norm_scale =
            rsqrt(squares[0] / static_cast<float>(HeadDim) + eps[0]);
        if (is_q) {
          int head = group_idx;
          q[((batch_idx * Seq + token) * Hk + head) * HeadDim + lane] =
              static_cast<T>(activated * norm_scale * q_scale[0]);
        } else {
          int head = group_idx - Hk;
          k[((batch_idx * Seq + token) * Hk + head) * HeadDim + lane] =
              static_cast<T>(activated * norm_scale * k_scale[0]);
        }
      } else {
        int head = group_idx - 2 * Hk;
        v[((batch_idx * Seq + token) * Hv + head) * HeadDim + lane] =
            static_cast<T>(activated);
      }

      if ((is_q || is_k) && token + 1 < Seq) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (int t = 0; t < TailLen - 1; ++t) {
        tail[t] = tail[t + 1];
      }
      if (TailLen > 0) {
        tail[TailLen - 1] = static_cast<float>(qkv_t[channel]);
      }
    }

    for (int t = 0; t < TailLen; ++t) {
      new_state_b[t * ConvDim + channel] = static_cast<T>(tail[t]);
    }
"#;

// Prefill streaming: fuse g/beta each timestep (like decode) without a
// CacheCapacity-sized threadgroup array. One leader thread computes the
// shared (hv, t) gates into two scalars; the rest of the TG waits on a
// barrier. Occupancy stays high for long prompts and there is no separate
// MLX precompute graph for g/beta.
const GATED_DELTA_PREFILL_STREAMING_KERNEL_SOURCE: &str = r#"
    const int t_len = seq_len[0];
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk] InT
    auto q_ = q + b_idx * t_len * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * t_len * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv] InT
    auto v_ = v + b_idx * t_len * Hv * Dv + hv_idx * Dv;
    y += b_idx * t_len * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // a_log: [Hv] StT (float32); dt_bias: [Hv] StT (float32)
    const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
    const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
    auto a_base = a_raw + b_idx * t_len * Hv;
    auto b_base = b_raw + b_idx * t_len * Hv;

    // state_in, state_out: [B, Hv, Dv, Dk] StT
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    const int s_base = n_per_t * dk_idx;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    threadgroup float g_t;
    threadgroup float beta_t;

    for (int t = 0; t < t_len; ++t) {
      if (thread_index_in_threadgroup == 0) {
        float a_plus_dt = static_cast<float>(a_base[t * Hv + hv_idx]) + dt_bias_v;
        float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
        g_t = exp(-exp_a_log * sp);
        float b_val = static_cast<float>(b_base[t * Hv + hv_idx]);
        // Preserve bf16/activation rounding of sigmoid(b) (mlx_lm + legacy kernel).
        beta_t = static_cast<float>(static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const float v_t = static_cast<float>(v_[dv_idx]);

      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] * g_t;
        kv_mem += state[i] * static_cast<float>(k_[s_base + i]);
      }
      kv_mem = simd_sum(kv_mem);

      const float delta = (v_t - kv_mem) * beta_t;

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] + static_cast<float>(k_[s_base + i]) * delta;
        out += state[i] * static_cast<float>(q_[s_base + i]);
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }

      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

const GATED_DELTA_KERNEL_SOURCE: &str = r#"
    const int t_len = seq_len[0];
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk] InT
    auto q_ = q + b_idx * t_len * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * t_len * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv] InT
    auto v_ = v + b_idx * t_len * Hv * Dv + hv_idx * Dv;
    y += b_idx * t_len * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // a_log: [Hv] StT (float32); dt_bias: [Hv] StT (float32)
    // exp(A_log[hv]) is invariant across all timesteps for this thread.
    const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
    const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);

    // Precompute g_t and beta_t for all timesteps cooperatively across the
    // threadgroup (32x4x1 = 128 threads). All threads share the same hv_idx
    // so they would otherwise recompute identical transcendental values in
    // every iteration of the hot loop — 127/128 redundant calls eliminated.
    //
    // CacheCapacity is specialized from Rust into three tiers (legacy path):
    //   512  — short prompts
    //   1024 — medium prompts
    //   2048 — long prompts
    // Prefer the streaming prefill kernel (no TG cache) in production.
    threadgroup float g_t_cache[CacheCapacity];
    threadgroup float beta_t_cache[CacheCapacity];

    auto a_base = a_raw + b_idx * t_len * Hv;
    auto b_base = b_raw + b_idx * t_len * Hv;
    const uint tid = thread_index_in_threadgroup;
    for (uint fill_t = tid; fill_t < (uint)t_len; fill_t += 128) {
      float a_plus_dt = static_cast<float>(a_base[fill_t * Hv + hv_idx]) + dt_bias_v;
      float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
      g_t_cache[fill_t] = exp(-exp_a_log * sp);
      float b_val = static_cast<float>(b_base[fill_t * Hv + hv_idx]);
      // mlx_lm computes `beta = sigmoid(b)` as a separate MLX op. For bf16
      // activations that op returns bf16, then the Metal recurrent kernel reads
      // the rounded value. Preserve that contract here even though the fused
      // kernel computes beta internally in float.
      beta_t_cache[fill_t] =
          static_cast<float>(static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // state_in, state_out: [B, Hv, Dv, Dk] StT
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    // s_base is invariant across both the t-loop and the inner i-loops.
    const int s_base = n_per_t * dk_idx;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    for (int t = 0; t < t_len; ++t) {
      const float g_t = g_t_cache[t];
      const float beta_t = beta_t_cache[t];
      const float v_t = static_cast<float>(v_[dv_idx]);

      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] * g_t;
        kv_mem += state[i] * static_cast<float>(k_[s_base + i]);
      }
      kv_mem = simd_sum(kv_mem);

      const float delta = (v_t - kv_mem) * beta_t;

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] + static_cast<float>(k_[s_base + i]) * delta;
        out += state[i] * static_cast<float>(q_[s_base + i]);
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }

      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

const GATED_DELTA_DECODE_KERNEL_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, 1, Hk, Dk] InT
    auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;

    // v, y: [B, 1, Hv, Dv] InT
    auto v_ = v + b_idx * Hv * Dv + hv_idx * Dv;
    y += b_idx * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    threadgroup float g_t;
    threadgroup float beta_t;
    if (thread_index_in_threadgroup == 0) {
      const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
      const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
      float a_plus_dt = static_cast<float>(a_raw[b_idx * Hv + hv_idx]) + dt_bias_v;
      float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
      g_t = exp(-exp_a_log * sp);
      float b_val = static_cast<float>(b_raw[b_idx * Hv + hv_idx]);
      beta_t = static_cast<float>(static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // state_in, state_out: [B, Hv, Dv, Dk] StT
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    const int s_base = n_per_t * dk_idx;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    const float v_t = static_cast<float>(v_[dv_idx]);

    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = state[i] * g_t;
      kv_mem += state[i] * static_cast<float>(k_[s_base + i]);
    }
    kv_mem = simd_sum(kv_mem);

    const float delta = (v_t - kv_mem) * beta_t;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = state[i] + static_cast<float>(k_[s_base + i]) * delta;
      out += state[i] * static_cast<float>(q_[s_base + i]);
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LinearAttentionConfig {
        let (q_scale, k_scale) = linear_attention_qk_scale(4);
        LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 2,
            num_key_heads: 1,
            key_head_dim: 4,
            value_head_dim: 3,
            conv_kernel_dim: 4,
            q_scale,
            k_scale,
        }
    }

    fn f32_array(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn stable_softplus(value: f32) -> f32 {
        if value > 20.0 {
            value
        } else {
            (1.0 + value.exp()).ln()
        }
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label} length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        );

        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= tolerance,
                "{label}[{idx}] mismatch: actual={actual}, expected={expected}, diff={diff}, tolerance={tolerance}"
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_delta_cpu_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        a_log: &[f32],
        a_raw: &[f32],
        dt_bias: &[f32],
        b_raw: &[f32],
        initial_state: &[f32],
        seq: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut state = initial_state.to_vec();
        let mut y = vec![0.0; seq * value_head_dim];
        let decay_rate = a_log[0].exp();

        for t in 0..seq {
            let g = (-decay_rate * stable_softplus(a_raw[t] + dt_bias[0])).exp();
            let beta = sigmoid(b_raw[t]);
            for dv in 0..value_head_dim {
                let state_offset = dv * key_head_dim;
                let mut kv_mem = 0.0;
                for dk in 0..key_head_dim {
                    let state_idx = state_offset + dk;
                    state[state_idx] *= g;
                    kv_mem += state[state_idx] * k[t * key_head_dim + dk];
                }

                let delta = (v[t * value_head_dim + dv] - kv_mem) * beta;
                let mut out = 0.0;
                for dk in 0..key_head_dim {
                    let state_idx = state_offset + dk;
                    state[state_idx] += k[t * key_head_dim + dk] * delta;
                    out += state[state_idx] * q[t * key_head_dim + dk];
                }
                y[t * value_head_dim + dv] = out;
            }
        }

        (y, state)
    }

    #[test]
    fn warm_gated_delta_decode_kernels_compiles_decode_specializations() {
        // Exercises the exact load-time warm path — decode-shape gated
        // delta (seq=1 specialization) plus conv1d — at kernel-eligible
        // head dims (key_head_dim divisible by 32, like every production
        // linear-attention config). Must succeed on any Metal host that
        // can serve the model at all.
        let (q_scale, k_scale) = linear_attention_qk_scale(32);
        let production_like = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 2,
            num_key_heads: 1,
            key_head_dim: 32,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            q_scale,
            k_scale,
        };
        warm_gated_delta_decode_kernels(&production_like)
            .expect("warm-up must compile decode kernels");

        // Sub-threshold dims never dispatch the custom kernel; the warm
        // path must be a no-op success, not a panic.
        warm_gated_delta_decode_kernels(&cfg()).expect("ineligible dims must be a no-op");
    }

    #[test]
    fn compute_gated_delta_g_preserves_shape_and_float32_dtype() {
        let cfg = cfg();
        let a_log = zeros(&[cfg.num_value_heads as i32], MlxDtype::Float32, None);
        let a = zeros(
            &[1, 5, cfg.num_value_heads as i32],
            MlxDtype::Bfloat16,
            None,
        );
        let dt_bias = zeros(&[cfg.num_value_heads as i32], MlxDtype::Bfloat16, None);

        let g = compute_gated_delta_g(&a_log, &a, &dt_bias);

        assert_eq!(g.shape(), vec![1, 5, 2]);
        // Streaming prefill keeps g in float32 for the recurrent state update
        // (matches mlx_lm's compute_g → float32 contract).
        assert_eq!(g.dtype(), MlxDtype::Float32);
    }

    #[test]
    fn compute_gated_delta_g_uses_stable_softplus_for_large_positive_values() {
        let a_log = f32_array(&[0.0], &[1]);
        let a = f32_array(&[25.0], &[1, 1, 1]);
        let dt_bias = f32_array(&[0.0], &[1]);

        let g = compute_gated_delta_g(&a_log, &a, &dt_bias);
        mlx_sys::eval(&[&g]);

        let actual = g.data_f32()[0];
        let expected = (-25.0_f32).exp();
        assert!(actual.is_finite(), "g should stay finite, got {actual}");
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn linear_attention_conv1d_returns_prompt_output_and_tail() {
        let cfg = cfg();
        let qkv = zeros(&[1, 5, cfg.conv_dim() as i32], MlxDtype::Float32, None);
        let weight = zeros(
            &[cfg.conv_dim() as i32, cfg.conv_kernel_dim as i32, 1_i32],
            MlxDtype::Float32,
            None,
        );

        let (conv_out, new_state) = linear_attention_conv1d(&cfg, &qkv, &weight, None);

        assert_eq!(conv_out.shape(), vec![1, 5, 14]);
        assert_eq!(new_state.shape(), vec![1, 3, 14]);
    }

    #[test]
    fn split_linear_attention_qkv_matches_config_dims() {
        let cfg = cfg();
        let conv_out = zeros(&[1, 5, cfg.conv_dim() as i32], MlxDtype::Float32, None);

        let qkv = split_linear_attention_qkv(&cfg, &conv_out);

        assert_eq!(qkv.q.shape(), vec![1, 5, 1, 4]);
        assert_eq!(qkv.k.shape(), vec![1, 5, 1, 4]);
        assert_eq!(qkv.v.shape(), vec![1, 5, 2, 3]);
    }

    #[test]
    fn gated_delta_kernel_reports_reference_shapes() {
        // B=1, T=2, Hk=1, Dk=32, Hv=1, Dv=4
        let q = zeros(&[1, 2, 1, 32], MlxDtype::Float32, None);
        let k = zeros(&[1, 2, 1, 32], MlxDtype::Float32, None);
        let v = zeros(&[1, 2, 1, 4], MlxDtype::Float32, None);
        // a_log, dt_bias: [Hv] float32 (StT)
        let a_log = zeros(&[1], MlxDtype::Float32, None);
        let a_raw = zeros(&[1, 2, 1], MlxDtype::Float32, None);
        let dt_bias = zeros(&[1], MlxDtype::Float32, None);
        let b_raw = zeros(&[1, 2, 1], MlxDtype::Float32, None);
        let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

        let (y, new_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);

        assert_eq!(y.shape(), vec![1, 2, 1, 4]);
        assert_eq!(new_state.shape(), vec![1, 1, 4, 32]);
    }

    #[test]
    fn gated_delta_prefill_short_seq_matches_cpu_on_default_path() {
        // Hybrid default: seq <= 512 uses the legacy short TG-cache kernel.
        const SEQ: usize = 8;
        const KEY_HEAD_DIM: usize = 32;
        const VALUE_HEAD_DIM: usize = 4;
        let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
            .collect();
        let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
            .collect();
        let v_data: Vec<f32> = (0..SEQ * VALUE_HEAD_DIM)
            .map(|idx| ((idx % 3) as f32 - 1.0) * 0.04)
            .collect();
        let a_log_data = vec![-0.2];
        let a_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let dt_bias_data = vec![0.05];
        let b_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect();
        let (expected_y, expected_state) = gated_delta_cpu_reference(
            &q_data,
            &k_data,
            &v_data,
            &a_log_data,
            &a_raw_data,
            &dt_bias_data,
            &b_raw_data,
            &state_data,
            SEQ,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
        );
        let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
        let a_log = f32_array(&a_log_data, &[1]);
        let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
        let dt_bias = f32_array(&dt_bias_data, &[1]);
        let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
        let state = f32_array(
            &state_data,
            &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
        );
        let (y, new_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
        mlx_sys::eval(&[&y, &new_state]);
        assert_close("y", y.data_f32(), &expected_y, 1e-5);
        assert_close("state", new_state.data_f32(), &expected_state, 1e-5);
    }

    #[test]
    fn gated_delta_kernel_accepts_medium_prefill_specialization() {
        // seq=1024 is the production p2048 chunk. Default-ON tile-512 splits
        // this into two 512 TG kernels; this test drives that shipped path.
        let seq = (GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY) as i32;
        let q = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
        let k = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
        let v = zeros(&[1, seq, 1, 4], MlxDtype::Float32, None);
        let a_log = zeros(&[1], MlxDtype::Float32, None);
        let a_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
        let dt_bias = zeros(&[1], MlxDtype::Float32, None);
        let b_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
        let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

        let (y, new_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
        mlx_sys::eval(&[&y, &new_state]);

        assert_eq!(y.shape(), vec![1, seq, 1, 4]);
        assert_eq!(new_state.shape(), vec![1, 1, 4, 32]);
    }

    #[test]
    fn linear_attention_prefill_chunk_cap_follows_streaming() {
        assert_eq!(
            linear_attention_prefill_chunk_cap(true),
            GATED_DELTA_THREADGROUP_CACHE_CAPACITY
        );
        // Default-OFF after the 1280 wash: non-streaming cap is 1024.
        assert_eq!(
            linear_attention_prefill_chunk_cap(false),
            GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY
        );
        assert!(
            !fastpath::qwen_prefill_chunk_1280_enabled(),
            "closed 1280 chunk stays opt-in"
        );
    }

    #[test]
    fn gated_delta_prefill_tile_512_is_seq_gated() {
        assert!(!gated_delta_prefill_tile_512_seq_eligible(1));
        assert!(!gated_delta_prefill_tile_512_seq_eligible(
            GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32
        ));
        assert!(gated_delta_prefill_tile_512_seq_eligible(
            GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY as i32 + 1
        ));
        assert!(gated_delta_prefill_tile_512_seq_eligible(
            GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY as i32
        ));
    }

    #[test]
    fn gated_delta_prefill_tiled_matches_oneshot_short_seq() {
        const SEQ: usize = 16;
        const TILE: i32 = 8;
        const KEY_HEAD_DIM: usize = 32;
        const VALUE_HEAD_DIM: usize = 4;
        let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
            .collect();
        let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
            .collect();
        let v_data: Vec<f32> = (0..SEQ * VALUE_HEAD_DIM)
            .map(|idx| ((idx % 3) as f32 - 1.0) * 0.04)
            .collect();
        let a_log_data = vec![-0.2];
        let a_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let dt_bias_data = vec![0.05];
        let b_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect();
        let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
        let a_log = f32_array(&a_log_data, &[1]);
        let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
        let dt_bias = f32_array(&dt_bias_data, &[1]);
        let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
        let state = f32_array(
            &state_data,
            &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
        );
        let (want_y, want_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
        let (got_y, got_state) =
            gated_delta_prefill_tiled(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, TILE);
        mlx_sys::eval(&[&got_y, &got_state, &want_y, &want_state]);
        assert_close("y", got_y.data_f32(), want_y.data_f32(), 1e-5);
        assert_close("state", got_state.data_f32(), want_state.data_f32(), 1e-5);
    }

    #[test]
    fn gated_delta_prefill_chunkwise_matches_oneshot() {
        const SEQ: usize = 16;
        const TILE: i32 = 8;
        const KEY_HEAD_DIM: usize = 32;
        const VALUE_HEAD_DIM: usize = 4;
        let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
            .collect();
        let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
            .collect();
        let v_data: Vec<f32> = (0..SEQ * VALUE_HEAD_DIM)
            .map(|idx| ((idx % 3) as f32 - 1.0) * 0.04)
            .collect();
        let a_log_data = vec![-0.2];
        let a_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let dt_bias_data = vec![0.05];
        let b_raw_data: Vec<f32> = (0..SEQ).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect();
        let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
        let a_log = f32_array(&a_log_data, &[1]);
        let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
        let dt_bias = f32_array(&dt_bias_data, &[1]);
        let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
        let state = f32_array(
            &state_data,
            &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
        );
        let (want_y, want_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
        let (got_y, got_state) = gated_delta_prefill_chunkwise(
            &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, TILE,
        );
        mlx_sys::eval(&[&got_y, &got_state, &want_y, &want_state]);
        assert_close("y", got_y.data_f32(), want_y.data_f32(), 1e-5);
        assert_close("state", got_state.data_f32(), want_state.data_f32(), 1e-5);
        assert!(
            fastpath::should_qwen_gd_prefill_chunkwise_for(true, 1024),
            "shipped chunkwise gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn gated_delta_kernel_accepts_long_prefill_specialization() {
        let seq = (GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY + 1) as i32;
        let q = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
        let k = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
        let v = zeros(&[1, seq, 1, 4], MlxDtype::Float32, None);
        let a_log = zeros(&[1], MlxDtype::Float32, None);
        let a_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
        let dt_bias = zeros(&[1], MlxDtype::Float32, None);
        let b_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
        let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

        let (y, new_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
        mlx_sys::eval(&[&y, &new_state]);

        assert_eq!(y.shape(), vec![1, seq, 1, 4]);
        assert_eq!(new_state.shape(), vec![1, 1, 4, 32]);
    }

    #[test]
    fn gated_delta_kernel_matches_cpu_reference_for_small_sequence() {
        const SEQ: usize = 2;
        const KEY_HEAD_DIM: usize = 32;
        const VALUE_HEAD_DIM: usize = 4;

        let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
            .collect();
        let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
            .collect();
        let v_data = vec![0.10, -0.05, 0.07, 0.03, -0.02, 0.04, 0.08, -0.06];
        let a_log_data = vec![-0.2];
        let a_raw_data = vec![0.1, -0.15];
        let dt_bias_data = vec![0.05];
        let b_raw_data = vec![0.25, -0.1];
        let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect();
        let (expected_y, expected_state) = gated_delta_cpu_reference(
            &q_data,
            &k_data,
            &v_data,
            &a_log_data,
            &a_raw_data,
            &dt_bias_data,
            &b_raw_data,
            &state_data,
            SEQ,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
        );

        let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
        let a_log = f32_array(&a_log_data, &[1]);
        let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
        let dt_bias = f32_array(&dt_bias_data, &[1]);
        let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
        let state = f32_array(
            &state_data,
            &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
        );

        let (y, new_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
        mlx_sys::eval(&[&y, &new_state]);

        assert_close("y", y.data_f32(), &expected_y, 1e-6);
        assert_close("state", new_state.data_f32(), &expected_state, 1e-6);
    }

    #[test]
    fn gated_delta_prefix_checkpoint_matches_decode_kernel_row0() {
        const SEQ: i32 = 2;
        const KEY_HEAD_DIM: i32 = 32;
        const VALUE_HEAD_DIM: i32 = 4;
        let q_data: Vec<f32> = (0..(SEQ * KEY_HEAD_DIM) as usize)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
            .collect();
        let k_data: Vec<f32> = (0..(SEQ * KEY_HEAD_DIM) as usize)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
            .collect();
        let v_data = vec![0.10, -0.05, 0.07, 0.03, -0.02, 0.04, 0.08, -0.06];
        let a_log = f32_array(&[-0.2], &[1]);
        let a_raw = f32_array(&[0.1, -0.15], &[1, SEQ, 1]);
        let dt_bias = f32_array(&[0.05], &[1]);
        let b_raw = f32_array(&[0.25, -0.1], &[1, SEQ, 1]);
        let q = f32_array(&q_data, &[1, SEQ, 1, KEY_HEAD_DIM]);
        let k = f32_array(&k_data, &[1, SEQ, 1, KEY_HEAD_DIM]);
        let v = f32_array(&v_data, &[1, SEQ, 1, VALUE_HEAD_DIM]);
        let state = f32_array(
            &(0..(VALUE_HEAD_DIM * KEY_HEAD_DIM) as usize)
                .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
                .collect::<Vec<_>>(),
            &[1, 1, VALUE_HEAD_DIM, KEY_HEAD_DIM],
        );

        let (_y_ck, _final_ck, prefix_ck) = gated_delta_kernel_with_prefix_checkpoint(
            &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, 1,
        );
        let q0 = contiguous(
            &slice(
                &q,
                &[0, 0, 0, 0],
                &[1, 1, 1, KEY_HEAD_DIM],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        );
        let k0 = contiguous(
            &slice(
                &k,
                &[0, 0, 0, 0],
                &[1, 1, 1, KEY_HEAD_DIM],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        );
        let v0 = contiguous(
            &slice(
                &v,
                &[0, 0, 0, 0],
                &[1, 1, 1, VALUE_HEAD_DIM],
                &[1, 1, 1, 1],
                None,
            ),
            None,
        );
        let a0 = contiguous(
            &slice(&a_raw, &[0, 0, 0], &[1, 1, 1], &[1, 1, 1], None),
            None,
        );
        let b0 = contiguous(
            &slice(&b_raw, &[0, 0, 0], &[1, 1, 1], &[1, 1, 1], None),
            None,
        );
        let (_y0, state1) = gated_delta_kernel(&q0, &k0, &v0, &a_log, &a0, &dt_bias, &b0, &state);
        mlx_sys::eval(&[&prefix_ck, &state1]);
        assert_eq!(prefix_ck.data_f32(), state1.data_f32());
    }

    #[test]
    fn gated_delta_decode_kernel_matches_cpu_reference_for_single_token() {
        const SEQ: usize = 1;
        const KEY_HEAD_DIM: usize = 32;
        const VALUE_HEAD_DIM: usize = 4;

        let q_data: Vec<f32> = (0..KEY_HEAD_DIM)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
            .collect();
        let k_data: Vec<f32> = (0..KEY_HEAD_DIM)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
            .collect();
        let v_data = vec![0.10, -0.05, 0.07, 0.03];
        let a_log_data = vec![-0.2];
        let a_raw_data = vec![0.1];
        let dt_bias_data = vec![0.05];
        let b_raw_data = vec![0.25];
        let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect();
        let (expected_y, expected_state) = gated_delta_cpu_reference(
            &q_data,
            &k_data,
            &v_data,
            &a_log_data,
            &a_raw_data,
            &dt_bias_data,
            &b_raw_data,
            &state_data,
            SEQ,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
        );

        let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
        let a_log = f32_array(&a_log_data, &[1]);
        let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
        let dt_bias = f32_array(&dt_bias_data, &[1]);
        let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
        let state = f32_array(
            &state_data,
            &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
        );

        let (y, new_state) =
            gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
        mlx_sys::eval(&[&y, &new_state]);

        assert_close("decode_y", y.data_f32(), &expected_y, 1e-6);
        assert_close("decode_state", new_state.data_f32(), &expected_state, 1e-6);
    }

    #[test]
    fn gated_delta_prefix_checkpoint_matches_first_singleton_state() {
        const SEQ: usize = 2;
        const KEY_HEAD_DIM: usize = 32;
        const VALUE_HEAD_DIM: usize = 4;
        let q_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.03)
            .collect();
        let k_data: Vec<f32> = (0..SEQ * KEY_HEAD_DIM)
            .map(|idx| ((idx % 5) as f32 - 2.0) * 0.02)
            .collect();
        let v_data = vec![0.10, -0.05, 0.07, 0.03, -0.02, 0.04, 0.08, -0.06];
        let a_log_data = vec![-0.2];
        let a_raw_data = vec![0.1, -0.15];
        let dt_bias_data = vec![0.05];
        let b_raw_data = vec![0.25, -0.1];
        let state_data: Vec<f32> = (0..VALUE_HEAD_DIM * KEY_HEAD_DIM)
            .map(|idx| ((idx % 11) as f32 - 5.0) * 0.005)
            .collect();
        let (_, expected_checkpoint) = gated_delta_cpu_reference(
            &q_data[..KEY_HEAD_DIM],
            &k_data[..KEY_HEAD_DIM],
            &v_data[..VALUE_HEAD_DIM],
            &a_log_data,
            &a_raw_data[..1],
            &dt_bias_data,
            &b_raw_data[..1],
            &state_data,
            1,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
        );
        let q = f32_array(&q_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let k = f32_array(&k_data, &[1, SEQ as i32, 1, KEY_HEAD_DIM as i32]);
        let v = f32_array(&v_data, &[1, SEQ as i32, 1, VALUE_HEAD_DIM as i32]);
        let a_log = f32_array(&a_log_data, &[1]);
        let a_raw = f32_array(&a_raw_data, &[1, SEQ as i32, 1]);
        let dt_bias = f32_array(&dt_bias_data, &[1]);
        let b_raw = f32_array(&b_raw_data, &[1, SEQ as i32, 1]);
        let state = f32_array(
            &state_data,
            &[1, 1, VALUE_HEAD_DIM as i32, KEY_HEAD_DIM as i32],
        );

        let (output, final_state, checkpoint) = gated_delta_kernel_with_prefix_checkpoint(
            &q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state, 1,
        );
        mlx_sys::eval(&[&output, &final_state, &checkpoint]);

        assert_close(
            "prefix_checkpoint",
            checkpoint.data_f32(),
            &expected_checkpoint,
            1e-6,
        );
    }

    #[test]
    fn normalize_linear_attention_qk_preserves_reference_shapes() {
        let (q_scale, k_scale) = linear_attention_qk_scale(32);
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 1,
            num_key_heads: 1,
            key_head_dim: 32,
            value_head_dim: 4,
            conv_kernel_dim: 4,
            q_scale,
            k_scale,
        };
        let q = zeros(&[1, 2, 1, 32], MlxDtype::Bfloat16, None);
        let k = zeros(&[1, 2, 1, 32], MlxDtype::Bfloat16, None);

        let (q, k) = normalize_linear_attention_qk(&cfg, &q, &k, 1e-6);

        assert_eq!(q.shape(), vec![1, 2, 1, 32]);
        assert_eq!(k.shape(), vec![1, 2, 1, 32]);
        assert_eq!(q.dtype(), MlxDtype::Bfloat16);
        assert_eq!(k.dtype(), MlxDtype::Bfloat16);
    }

    #[test]
    fn decode_post_input_metal_matches_portable_composition_for_short_sequences() {
        let (q_scale, k_scale) = linear_attention_qk_scale(32);
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 2,
            num_key_heads: 1,
            key_head_dim: 32,
            value_head_dim: 32,
            conv_kernel_dim: 4,
            q_scale,
            k_scale,
        };
        let conv_dim = cfg.conv_dim();
        let state_data: Vec<f32> = (0..3 * conv_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
            .collect();
        let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
            .collect();
        let weight = f32_array(
            &weight_data,
            &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
        );

        for seq in 1..=4 {
            let qkv_data: Vec<f32> = (0..seq * conv_dim)
                .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
                .collect();
            let qkv = f32_array(&qkv_data, &[1, seq as i32, conv_dim as i32]);
            let state = f32_array(&state_data, &[1, 3, conv_dim as i32]);
            let (conv_out, portable_state) =
                linear_attention_conv1d(&cfg, &qkv, &weight, Some(&state));
            let split = split_linear_attention_qkv(&cfg, &conv_out);
            let (portable_q, portable_k) =
                normalize_linear_attention_qk(&cfg, &split.q, &split.k, 1e-6);
            let (metal_q, metal_k, metal_v, metal_state) =
                linear_attention_decode_post_input_metal(
                    &cfg,
                    &qkv,
                    &weight,
                    Some(&state),
                    q_scale,
                    k_scale,
                    1e-6,
                )
                .expect("decode post-input Metal path should accept Qwen-like shape");
            let portable_q = mlx_sys::contiguous(&portable_q, None);
            let portable_k = mlx_sys::contiguous(&portable_k, None);
            let portable_v = mlx_sys::contiguous(&split.v, None);
            let portable_state = mlx_sys::contiguous(&portable_state, None);
            let metal_q = mlx_sys::contiguous(&metal_q, None);
            let metal_k = mlx_sys::contiguous(&metal_k, None);
            let metal_v = mlx_sys::contiguous(&metal_v, None);
            let metal_state = mlx_sys::contiguous(&metal_state, None);
            mlx_sys::eval(&[
                &portable_q,
                &portable_k,
                &portable_v,
                &portable_state,
                &metal_q,
                &metal_k,
                &metal_v,
                &metal_state,
            ]);

            assert_close(
                &format!("decode_post_input_q_seq{seq}"),
                metal_q.data_f32(),
                portable_q.data_f32(),
                1e-5,
            );
            assert_close(
                &format!("decode_post_input_k_seq{seq}"),
                metal_k.data_f32(),
                portable_k.data_f32(),
                1e-5,
            );
            assert_close(
                &format!("decode_post_input_v_seq{seq}"),
                metal_v.data_f32(),
                portable_v.data_f32(),
                1e-5,
            );
            assert_close(
                &format!("decode_post_input_state_seq{seq}"),
                metal_state.data_f32(),
                portable_state.data_f32(),
                1e-6,
            );
        }
    }

    #[test]
    fn prefill_post_input_metal_matches_portable_seq8_cold_start() {
        // p2048 first chunk has no cached conv state. Metal must match
        // portable conv1d cold start at a multi-token prefill shape.
        let (q_scale, k_scale) = linear_attention_qk_scale(32);
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 2,
            num_key_heads: 1,
            key_head_dim: 32,
            value_head_dim: 32,
            conv_kernel_dim: 4,
            q_scale,
            k_scale,
        };
        let conv_dim = cfg.conv_dim();
        let seq = 8_i32;
        let qkv_data: Vec<f32> = (0..seq as usize * conv_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
            .collect();
        let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
            .collect();
        let qkv = f32_array(&qkv_data, &[1, seq, conv_dim as i32]);
        let weight = f32_array(
            &weight_data,
            &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
        );
        let (conv_out, portable_state) = linear_attention_conv1d(&cfg, &qkv, &weight, None);
        let split = split_linear_attention_qkv(&cfg, &conv_out);
        let (portable_q, portable_k) =
            normalize_linear_attention_qk(&cfg, &split.q, &split.k, 1e-6);
        let (metal_q, metal_k, metal_v, metal_state) = linear_attention_decode_post_input_metal(
            &cfg, &qkv, &weight, None, q_scale, k_scale, 1e-6,
        )
        .expect("prefill post-input Metal path should accept seq=8 cold start");
        let portable_q = mlx_sys::contiguous(&portable_q, None);
        let portable_k = mlx_sys::contiguous(&portable_k, None);
        let portable_v = mlx_sys::contiguous(&split.v, None);
        let portable_state = mlx_sys::contiguous(&portable_state, None);
        let metal_q = mlx_sys::contiguous(&metal_q, None);
        let metal_k = mlx_sys::contiguous(&metal_k, None);
        let metal_v = mlx_sys::contiguous(&metal_v, None);
        let metal_state = mlx_sys::contiguous(&metal_state, None);
        mlx_sys::eval(&[
            &portable_q,
            &portable_k,
            &portable_v,
            &portable_state,
            &metal_q,
            &metal_k,
            &metal_v,
            &metal_state,
        ]);
        assert_close(
            "prefill_post_input_q",
            metal_q.data_f32(),
            portable_q.data_f32(),
            2e-5,
        );
        assert_close(
            "prefill_post_input_k",
            metal_k.data_f32(),
            portable_k.data_f32(),
            2e-5,
        );
        assert_close(
            "prefill_post_input_v",
            metal_v.data_f32(),
            portable_v.data_f32(),
            2e-5,
        );
        assert_close(
            "prefill_post_input_state",
            metal_state.data_f32(),
            portable_state.data_f32(),
            1e-6,
        );
    }

    #[test]
    fn normalize_linear_attention_qk_q_uses_inv_scale_squared() {
        // mlx-lm/Swift: q_scale = Dk^(-1), k_scale = Dk^(-0.5)
        let (q_scale, k_scale) = linear_attention_qk_scale(4);

        assert!((q_scale - 0.25).abs() < f32::EPSILON, "q_scale={q_scale}");
        assert!((k_scale - 0.5).abs() < f32::EPSILON, "k_scale={k_scale}");
    }

    #[test]
    fn rms_norm_gate_metal_matches_direct_chain_for_bf16() {
        let normed_data: Vec<f32> = (0..16)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.125)
            .collect();
        let gate_data: Vec<f32> = (0..16).map(|idx| ((idx % 5) as f32 - 2.0) * 0.25).collect();
        let normed = astype(
            &f32_array(&normed_data, &[1, 2, 2, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let gate = astype(
            &f32_array(&gate_data, &[1, 2, 2, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let direct = astype(
            &multiply(
                &mlx_sys::ops::silu(&astype(&gate, MlxDtype::Float32, None), None),
                &astype(&normed, MlxDtype::Float32, None),
                None,
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let metal = rms_norm_gate_metal_impl(&normed, &gate, MlxDtype::Bfloat16)
            .expect("bf16 linear-attention RMSNorm gate Metal fast path");
        let direct = astype(&direct, MlxDtype::Float32, None);
        let metal = astype(&metal, MlxDtype::Float32, None);
        mlx_sys::eval(&[&direct, &metal]);

        assert_close("rms_norm_gate", metal.data_f32(), direct.data_f32(), 2.0e-2);
    }

    #[test]
    fn rms_norm_full_gate_metal_matches_direct_chain_for_bf16() {
        let hidden_data: Vec<f32> = (0..16)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.125)
            .collect();
        let gate_data: Vec<f32> = (0..16).map(|idx| ((idx % 5) as f32 - 2.0) * 0.25).collect();
        let weight_data = vec![0.8_f32, 1.0, 1.2, 1.4];
        let hidden = astype(
            &f32_array(&hidden_data, &[1, 2, 2, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let gate = astype(
            &f32_array(&gate_data, &[1, 2, 2, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let weight = astype(&f32_array(&weight_data, &[4]), MlxDtype::Bfloat16, None);

        let normed = rms_norm(&hidden, Some(&weight), 1e-6, None);
        let direct = astype(
            &multiply(
                &mlx_sys::ops::silu(&astype(&gate, MlxDtype::Float32, None), None),
                &astype(&normed, MlxDtype::Float32, None),
                None,
            ),
            MlxDtype::Bfloat16,
            None,
        );
        let metal = rms_norm_full_gate_metal_impl(&hidden, &gate, &weight, 1e-6)
            .expect("bf16 linear-attention full RMSNorm gate Metal fast path");
        let direct = astype(&direct, MlxDtype::Float32, None);
        let metal = astype(&metal, MlxDtype::Float32, None);
        mlx_sys::eval(&[&direct, &metal]);

        assert_close(
            "rms_norm_full_gate",
            metal.data_f32(),
            direct.data_f32(),
            2.0e-2,
        );
    }

    #[test]
    fn rms_norm_gate_metal_rejects_shape_mismatch() {
        let normed = zeros(&[1, 2, 2, 4], MlxDtype::Bfloat16, None);
        let gate = zeros(&[1, 2, 1, 4], MlxDtype::Bfloat16, None);

        assert!(rms_norm_gate_metal_impl(&normed, &gate, MlxDtype::Bfloat16).is_none());
    }

    #[test]
    fn rms_norm_gated_preserves_hidden_shape_and_dtype() {
        let hidden = zeros(&[1, 5, 2, 3], MlxDtype::Bfloat16, None);
        let gate = zeros(&[1, 5, 2, 3], MlxDtype::Bfloat16, None);
        let weight = zeros(&[3], MlxDtype::Bfloat16, None);

        let out = rms_norm_gated(&hidden, &gate, &weight, 1e-6);

        assert_eq!(out.shape(), vec![1, 5, 2, 3]);
        assert_eq!(out.dtype(), MlxDtype::Bfloat16);
    }

    #[test]
    fn exact_profile_skips_rms_norm_gate_metal() {
        {
            let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
            assert!(!super::skip_rms_norm_gate_metal_for_exact_verify());
        }
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        assert!(super::skip_rms_norm_gate_metal_for_exact_verify());
    }
}
