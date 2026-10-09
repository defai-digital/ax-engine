use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxClosure, MlxDtype, MlxMetalKernel,
    MlxVectorArray, astype, concatenate, contiguous, conv1d, multiply, reshape, rms_norm,
    rms_norm_silu_mul_normed, silu_mul, slice, slice_last_dim, zeros,
};
#[cfg(test)]
use mlx_sys::{add, exp, less, log1p, negative, where_cond};

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
static GATED_DELTA_DECODE_SEQ_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_DECODE_SEQ_NO_CHECKPOINT_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_DECODE_SEQ_TAPE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_DECODE_SEQ_TAPE_REPLAY_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_FUSED_VERIFY_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_FUSED_VERIFY_NO_CHECKPOINT_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GATED_DELTA_FUSED_VERIFY_TRACE_ONCE: OnceLock<()> = OnceLock::new();
static GATED_DELTA_FUSED_VERIFY_NO_CHECKPOINT_TRACE_ONCE: OnceLock<()> = OnceLock::new();
static DECODE_POST_INPUT_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static DECODE_POST_INPUT_SIMD32_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static RMS_NORM_GATE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static RMS_NORM_FULL_GATE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Value rows processed by one short gated-delta verifier threadgroup.
///
/// Four is the historical AX launch geometry. Eight matches the better-filled
/// short-window layout used by the peer runtime while leaving each SIMD
/// group's arithmetic independent and unchanged. Keep this as an A/B knob
/// until the matched hardware gate selects a default.
fn gated_delta_verify_threadgroup_y(value_head_dim: i32) -> i32 {
    let requested = fastpath::gated_delta_verify_threadgroup_y_env();
    if value_head_dim % requested == 0 {
        requested
    } else {
        4
    }
}
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
/// Dense-hybrid production clamp is 1024. 1536 remasured community p2048
/// ~899 vs 908.5. One 2048 FFN + tile-512 remasured 889.96 vs 891.02
/// (2026-08-13). Streaming / `AX_MLX_QWEN_PREFILL_SINGLE_2048=1` still
/// take 2048.
///
/// MoE hybrids (Qwen 3.6 35B-A3B class) default to one 2048 chunk:
/// per-chunk MoE argsort/gather/dispatch overhead scales with chunk count,
/// and the `df-macbookpro-m5` A/B (2026-08-17, 35B AXQ 6bit, p2048,
/// reps 5) measured **+11.9%** prefill (2813.59 → 3149.26 tok/s) with
/// decode flat (131.86 vs 131.56). Dense hybrids measured wash and keep
/// the 1024 TG tile. Kill switch: `AX_MLX_QWEN_MOE_PREFILL_SINGLE_2048=0`.
pub(crate) fn linear_attention_prefill_chunk_cap(streaming: bool, moe: bool) -> usize {
    if streaming || fastpath::qwen_prefill_single_2048_enabled() {
        GATED_DELTA_THREADGROUP_CACHE_CAPACITY
    } else if fastpath::qwen_prefill_chunk_1536_enabled() {
        1536
    } else if fastpath::qwen_prefill_chunk_1280_enabled() {
        1280
    } else if moe && fastpath::qwen_moe_prefill_single_2048_enabled() {
        GATED_DELTA_THREADGROUP_CACHE_CAPACITY
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

/// Compile the linear-attention Metal kernel specializations a decode
/// step will hit, at production head dimensions, before the first
/// request arrives. MLX builds each MSL→pipeline specialization lazily
/// on the first eval that materializes it, so without this the compile
/// stall lands inside the first request's latency. Shapes mirror the
/// decode path exactly (batch 1, seq 1, cfg head dims) because template
/// specialization is shape-keyed — warming a different shape compiles a
/// different pipeline. Best-effort by contract: the caller logs and
/// continues on `Err`.
pub(crate) fn warm_gated_delta_decode_kernels(cfg: &LinearAttentionConfig) -> Result<(), String> {
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

    // Depth-1 exact verify is T=2. Compile the sequential-decode
    // specialization so the first factory step is not a kernel JIT.
    let q2 = zeros(&[1, 2, key_heads, key_dim], MlxDtype::Float32, None);
    let k2 = zeros(&[1, 2, key_heads, key_dim], MlxDtype::Float32, None);
    let v2 = zeros(&[1, 2, value_heads, value_dim], MlxDtype::Float32, None);
    let a2 = zeros(&[1, 2, value_heads], MlxDtype::Float32, None);
    let b2 = zeros(&[1, 2, value_heads], MlxDtype::Float32, None);
    if let Some((y2, state2, ck2)) = gated_delta_decode_seq_kernel(
        &q2,
        &k2,
        &v2,
        &a_log,
        &a2,
        &dt_bias,
        &b2,
        &state,
        1,
        2,
        key_heads,
        key_dim,
        value_heads,
        value_dim,
        &state.shape(),
    ) {
        mlx_sys::try_eval(&[&y2, &state2, &ck2])
            .map_err(|error| format!("gated-delta decode-seq warm-up failed: {error}"))?;
    }

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
    let (conv_out, new_state) =
        linear_attention_conv1d_pre_activation(cfg, qkv, conv_weight, cached_conv_state);
    (mlx_sys::ops::silu(&conv_out, None), new_state)
}

pub(crate) fn linear_attention_conv1d_pre_activation(
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
    (conv_out, new_state)
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
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray, MlxArray)> {
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

    let head_dim = cfg.key_head_dim as i32;
    let simd32 = seq <= fastpath::qwen_linear_mtp_max_verify_seq()
        && head_dim % 32 == 0
        && fastpath::qwen_linear_mtp_target_verify_enabled()
        && fastpath::mtp_gdn_prework_simd32_enabled();
    let kernel = if simd32 {
        DECODE_POST_INPUT_SIMD32_KERNEL.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_qwen_linear_attention_decode_post_input_simd32_v1",
                &[
                    "qkv",
                    "conv_weight",
                    "conv_state",
                    "q_scale",
                    "k_scale",
                    "eps",
                ],
                &["q", "k", "v", "new_conv_state", "prefix_conv_state"],
                DECODE_POST_INPUT_SIMD32_KERNEL_SOURCE,
                POST_INPUT_ROUNDING_HEADER,
                true,
            )
        })
    } else {
        DECODE_POST_INPUT_KERNEL.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_qwen_linear_attention_decode_post_input_v3",
                &[
                    "qkv",
                    "conv_weight",
                    "conv_state",
                    "q_scale",
                    "k_scale",
                    "eps",
                ],
                &["q", "k", "v", "new_conv_state", "prefix_conv_state"],
                DECODE_POST_INPUT_KERNEL_SOURCE,
                POST_INPUT_ROUNDING_HEADER,
                true,
            )
        })
    };
    let q_scale_arr = scalar_f32_as(q_scale, MlxDtype::Float32);
    let k_scale_arr = scalar_f32_as(k_scale, MlxDtype::Float32);
    let eps_arr = scalar_f32_as(eps, MlxDtype::Float32);
    let groups = (cfg.num_key_heads * 2 + cfg.num_value_heads) as i32;
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
        (if simd32 { 32 } else { head_dim }, 1, batch * groups),
        (if simd32 { 32 } else { head_dim }, 1, 1),
        None,
    );

    let mut outputs = outputs.into_iter();
    Some((
        outputs.next()?,
        outputs.next()?,
        outputs.next()?,
        outputs.next()?,
        outputs.next()?,
    ))
}

/// Short Qwen target-verifier kernel that keeps conv/QK intermediates inside
/// one Metal dispatch and returns both final and row-0 rollback state.
///
/// The verifier is deliberately narrow: B=1-style `Seq=2..=4`, equal
/// power-of-two Q/K/V head dimensions, and value rows evenly tiled by eight
/// SIMD groups. Unsupported inputs return `None` and retain the established
/// post-input + gated-delta composition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_fused_verify_from_qkv(
    cfg: &LinearAttentionConfig,
    qkv: &MlxArray,
    conv_weight: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    recurrent_state: &MlxArray,
    q_scale: f32,
    k_scale: f32,
    eps: f32,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray, MlxArray)> {
    const TGY: i32 = 8;
    let qkv_shape = qkv.shape();
    if qkv_shape.len() != 3
        || !fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(qkv_shape[1]))
    {
        return None;
    }
    let batch = qkv_shape[0];
    let seq = qkv_shape[1];
    let key_heads = cfg.num_key_heads as i32;
    let value_heads = cfg.num_value_heads as i32;
    let key_dim = cfg.key_head_dim as i32;
    let value_dim = cfg.value_head_dim as i32;
    if key_heads <= 0
        || value_heads <= 0
        || value_heads % key_heads != 0
        || key_dim != value_dim
        || !(32..=256).contains(&key_dim)
        || !(key_dim as u32).is_power_of_two()
        || key_dim % 32 != 0
        || value_dim % TGY != 0
        || cfg.conv_kernel_dim < 2
    {
        return None;
    }
    let conv_dim = cfg.conv_dim() as i32;
    let tail_len = cfg.conv_kernel_dim as i32 - 1;
    if qkv_shape[2] != conv_dim
        || conv_weight.shape() != vec![conv_dim, cfg.conv_kernel_dim as i32, 1]
        || a_log.shape() != vec![value_heads]
        || dt_bias.shape() != vec![value_heads]
        || a_raw.shape() != vec![batch, seq, value_heads]
        || b_raw.shape() != vec![batch, seq, value_heads]
        || recurrent_state.shape() != vec![batch, value_heads, value_dim, key_dim]
    {
        return None;
    }
    let zero_conv_state;
    let conv_state = if let Some(state) = cached_conv_state {
        if state.shape() != vec![batch, tail_len, conv_dim] {
            return None;
        }
        state
    } else {
        zero_conv_state = zeros(&[batch, tail_len, conv_dim], qkv.dtype(), None);
        &zero_conv_state
    };

    let kernel = GATED_DELTA_FUSED_VERIFY_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_gated_delta_fused_verify_v1",
            &[
                "qkv",
                "conv_weight",
                "conv_state",
                "a_log",
                "a_raw",
                "dt_bias",
                "b_raw",
                "state_in",
                "q_scale",
                "k_scale",
                "eps",
            ],
            &[
                "y",
                "state_out",
                "checkpoint",
                "new_conv_state",
                "prefix_conv_state",
            ],
            GATED_DELTA_FUSED_VERIFY_KERNEL_SOURCE,
            POST_INPUT_ROUNDING_HEADER,
            true,
        )
    });
    let q_scale_arr = scalar_f32_as(q_scale, MlxDtype::Float32);
    let k_scale_arr = scalar_f32_as(k_scale, MlxDtype::Float32);
    let eps_arr = scalar_f32_as(eps, MlxDtype::Float32);
    let mut outputs = kernel
        .try_apply_with_template(
            &[
                qkv,
                conv_weight,
                conv_state,
                a_log,
                a_raw,
                dt_bias,
                b_raw,
                recurrent_state,
                &q_scale_arr,
                &k_scale_arr,
                &eps_arr,
            ],
            &[
                KernelOutputSpec {
                    shape: vec![batch, seq, value_heads, value_dim],
                    dtype: qkv.dtype(),
                },
                KernelOutputSpec {
                    shape: recurrent_state.shape(),
                    dtype: recurrent_state.dtype(),
                },
                KernelOutputSpec {
                    shape: recurrent_state.shape(),
                    dtype: recurrent_state.dtype(),
                },
                KernelOutputSpec {
                    shape: vec![batch, tail_len, conv_dim],
                    dtype: qkv.dtype(),
                },
                KernelOutputSpec {
                    shape: vec![batch, tail_len, conv_dim],
                    dtype: qkv.dtype(),
                },
            ],
            &[
                KernelTemplateArg::Dtype {
                    name: "InT",
                    dtype: qkv.dtype(),
                },
                KernelTemplateArg::Dtype {
                    name: "StT",
                    dtype: recurrent_state.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "Dk",
                    value: key_dim,
                },
                KernelTemplateArg::Int {
                    name: "Dv",
                    value: value_dim,
                },
                KernelTemplateArg::Int {
                    name: "Hk",
                    value: key_heads,
                },
                KernelTemplateArg::Int {
                    name: "Hv",
                    value: value_heads,
                },
                KernelTemplateArg::Int {
                    name: "SeqLen",
                    value: seq,
                },
                KernelTemplateArg::Int {
                    name: "ConvKernelDim",
                    value: cfg.conv_kernel_dim as i32,
                },
                KernelTemplateArg::Int {
                    name: "Tgy",
                    value: TGY,
                },
            ],
            (32, value_dim, batch * value_heads),
            (32, TGY, 1),
            None,
        )
        .ok()?;
    if outputs.len() != 5 {
        return None;
    }
    let prefix_conv_state = outputs.pop()?;
    let new_conv_state = outputs.pop()?;
    let checkpoint = outputs.pop()?;
    let state_out = outputs.pop()?;
    let y = outputs.pop()?;
    if std::env::var_os("AX_MLX_MTP_FUSED_GDN_VERIFY_TRACE").is_some()
        && GATED_DELTA_FUSED_VERIFY_TRACE_ONCE.set(()).is_ok()
    {
        eprintln!("AX_MTP_FUSED_GDN_VERIFY engaged seq={seq}");
    }
    Some((y, state_out, checkpoint, new_conv_state, prefix_conv_state))
}

/// Fused verifier without the row-0 recurrent checkpoint or conv-prefix
/// buffers. Arithmetic stays lockstep with
/// [`gated_delta_fused_verify_from_qkv`]; complete-miss rollback uses keep=1
/// projected replay instead of a host-side checkpoint swap.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_fused_verify_no_checkpoint_from_qkv(
    cfg: &LinearAttentionConfig,
    qkv: &MlxArray,
    conv_weight: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    recurrent_state: &MlxArray,
    q_scale: f32,
    k_scale: f32,
    eps: f32,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    const TGY: i32 = 8;
    let qkv_shape = qkv.shape();
    if qkv_shape.len() != 3
        || !fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(qkv_shape[1]))
    {
        return None;
    }
    let batch = qkv_shape[0];
    let seq = qkv_shape[1];
    let key_heads = cfg.num_key_heads as i32;
    let value_heads = cfg.num_value_heads as i32;
    let key_dim = cfg.key_head_dim as i32;
    let value_dim = cfg.value_head_dim as i32;
    if key_heads <= 0
        || value_heads <= 0
        || value_heads % key_heads != 0
        || key_dim != value_dim
        || !(32..=256).contains(&key_dim)
        || !(key_dim as u32).is_power_of_two()
        || key_dim % 32 != 0
        || value_dim % TGY != 0
        || cfg.conv_kernel_dim < 2
    {
        return None;
    }
    let conv_dim = cfg.conv_dim() as i32;
    let tail_len = cfg.conv_kernel_dim as i32 - 1;
    if qkv_shape[2] != conv_dim
        || conv_weight.shape() != vec![conv_dim, cfg.conv_kernel_dim as i32, 1]
        || a_log.shape() != vec![value_heads]
        || dt_bias.shape() != vec![value_heads]
        || a_raw.shape() != vec![batch, seq, value_heads]
        || b_raw.shape() != vec![batch, seq, value_heads]
        || recurrent_state.shape() != vec![batch, value_heads, value_dim, key_dim]
    {
        return None;
    }
    let zero_conv_state;
    let conv_state = if let Some(state) = cached_conv_state {
        if state.shape() != vec![batch, tail_len, conv_dim] {
            return None;
        }
        state
    } else {
        zero_conv_state = zeros(&[batch, tail_len, conv_dim], qkv.dtype(), None);
        &zero_conv_state
    };

    let kernel = GATED_DELTA_FUSED_VERIFY_NO_CHECKPOINT_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_gated_delta_fused_verify_no_checkpoint_v1",
            &[
                "qkv",
                "conv_weight",
                "conv_state",
                "a_log",
                "a_raw",
                "dt_bias",
                "b_raw",
                "state_in",
                "q_scale",
                "k_scale",
                "eps",
            ],
            &["y", "state_out", "new_conv_state"],
            GATED_DELTA_FUSED_VERIFY_NO_CHECKPOINT_KERNEL_SOURCE,
            POST_INPUT_ROUNDING_HEADER,
            true,
        )
    });
    let q_scale_arr = scalar_f32_as(q_scale, MlxDtype::Float32);
    let k_scale_arr = scalar_f32_as(k_scale, MlxDtype::Float32);
    let eps_arr = scalar_f32_as(eps, MlxDtype::Float32);
    let mut outputs = kernel
        .try_apply_with_template(
            &[
                qkv,
                conv_weight,
                conv_state,
                a_log,
                a_raw,
                dt_bias,
                b_raw,
                recurrent_state,
                &q_scale_arr,
                &k_scale_arr,
                &eps_arr,
            ],
            &[
                KernelOutputSpec {
                    shape: vec![batch, seq, value_heads, value_dim],
                    dtype: qkv.dtype(),
                },
                KernelOutputSpec {
                    shape: recurrent_state.shape(),
                    dtype: recurrent_state.dtype(),
                },
                KernelOutputSpec {
                    shape: vec![batch, tail_len, conv_dim],
                    dtype: qkv.dtype(),
                },
            ],
            &[
                KernelTemplateArg::Dtype {
                    name: "InT",
                    dtype: qkv.dtype(),
                },
                KernelTemplateArg::Dtype {
                    name: "StT",
                    dtype: recurrent_state.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "Dk",
                    value: key_dim,
                },
                KernelTemplateArg::Int {
                    name: "Dv",
                    value: value_dim,
                },
                KernelTemplateArg::Int {
                    name: "Hk",
                    value: key_heads,
                },
                KernelTemplateArg::Int {
                    name: "Hv",
                    value: value_heads,
                },
                KernelTemplateArg::Int {
                    name: "SeqLen",
                    value: seq,
                },
                KernelTemplateArg::Int {
                    name: "ConvKernelDim",
                    value: cfg.conv_kernel_dim as i32,
                },
                KernelTemplateArg::Int {
                    name: "Tgy",
                    value: TGY,
                },
            ],
            (32, value_dim, batch * value_heads),
            (32, TGY, 1),
            None,
        )
        .ok()?;
    if outputs.len() != 3 {
        return None;
    }
    let new_conv_state = outputs.pop()?;
    let state_out = outputs.pop()?;
    let y = outputs.pop()?;
    if std::env::var_os("AX_MLX_MTP_FUSED_GDN_VERIFY_TRACE").is_some()
        && GATED_DELTA_FUSED_VERIFY_NO_CHECKPOINT_TRACE_ONCE
            .set(())
            .is_ok()
    {
        eprintln!("AX_MTP_FUSED_GDN_VERIFY_NO_CHECKPOINT engaged seq={seq}");
    }
    Some((y, state_out, new_conv_state))
}

pub(crate) fn split_linear_attention_qkv(
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
pub(crate) fn normalize_linear_attention_qk(
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
    if fastpath::qwen_gated_delta_prefill_mlx_enabled()
        && fastpath::qwen_gated_delta_prefill_mlx_seq_eligible(q.shape()[1])
        && let Some(result) = crate::mlx_gated_delta::try_mlx_gated_delta_prefill(
            q, k, v, a_log, a_raw, dt_bias, b_raw, state,
        )
    {
        return result;
    }
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
    // Depth-1/2/3 exact verify is T=2..=4. One sequential-decode kernel
    // runs the same per-token math as singleton `qwen35_gated_delta_decode_v1`
    // (and writes the row-0 checkpoint) so S=2 is one dispatch instead of
    // two decode launches plus five slice+contiguous copies + concat.
    let seq = q_shape[1];
    let state_shape = state.shape();
    if fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(seq))
        && let Some((y, state_out, checkpoint)) = gated_delta_decode_seq_kernel(
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
            &state_shape,
        )
    {
        return (y, state_out, checkpoint);
    }
    let decode_row = |q_t: &MlxArray,
                      k_t: &MlxArray,
                      v_t: &MlxArray,
                      a_t: &MlxArray,
                      b_t: &MlxArray,
                      state_t: &MlxArray| {
        gated_delta_decode_kernel(
            q_t,
            k_t,
            v_t,
            a_log,
            a_t,
            dt_bias,
            b_t,
            state_t,
            batch,
            num_key_heads,
            key_head_dim,
            num_value_heads,
            value_head_dim,
            state_shape.clone(),
        )
    };
    // Row 0: the decode kernel reads only the first sequence row, so the full
    // T>1 buffers avoid five slice+contiguous ops on the committed token.
    let (y0, checkpoint) = decode_row(q, k, v, a_raw, b_raw, state);
    let mut ys = Vec::with_capacity(seq as usize);
    ys.push(y0);
    let mut state_cur = checkpoint.clone();
    for t in 1..seq {
        let q_t = slice_seq_row_4d(q, t);
        let k_t = slice_seq_row_4d(k, t);
        let v_t = slice_seq_row_4d(v, t);
        let a_t = slice_seq_row_3d(a_raw, t);
        let b_t = slice_seq_row_3d(b_raw, t);
        let (y_t, next_state) = decode_row(&q_t, &k_t, &v_t, &a_t, &b_t, &state_cur);
        ys.push(y_t);
        state_cur = next_state;
    }
    let refs: Vec<&MlxArray> = ys.iter().collect();
    (concatenate(&refs, 1, None), state_cur, checkpoint)
}

/// Sequential T=2..=4 verifier update without a recurrent checkpoint output.
///
/// This is the replay-on-rejection morphology used by oMLX: the common full
/// accept evaluates only the verifier output and final state, while a reject
/// rebuilds the kept prefix from the pre-forward state and retained
/// projections. It intentionally shares the per-token arithmetic and launch
/// geometry of [`gated_delta_kernel_with_prefix_checkpoint`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel_verify_no_checkpoint(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
) -> Option<(MlxArray, MlxArray)> {
    let q_shape = q.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || v_shape.len() != 4 || q_shape[0] != 1 {
        return None;
    }
    let seq = q_shape[1];
    let num_key_heads = q_shape[2];
    let key_head_dim = q_shape[3];
    let num_value_heads = v_shape[2];
    if !fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(seq))
        || key_head_dim <= 0
        || key_head_dim % 32 != 0
        || num_key_heads <= 0
        || num_value_heads % num_key_heads != 0
    {
        return None;
    }

    let q_c = contiguous(q, None);
    let k_c = contiguous(k, None);
    let v_c = contiguous(v, None);
    let a_c = contiguous(a_raw, None);
    let b_c = contiguous(b_raw, None);
    let state_c = contiguous(state, None);
    let batch = q_shape[0];
    let value_head_dim = v_shape[3];
    let kernel = GATED_DELTA_DECODE_SEQ_NO_CHECKPOINT_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "qwen35_gated_delta_decode_seq_no_checkpoint_v1",
            &[
                "q", "k", "v", "a_log", "a_raw", "dt_bias", "b_raw", "state_in",
            ],
            &["y", "state_out"],
            GATED_DELTA_DECODE_SEQ_NO_CHECKPOINT_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let state_shape = state.shape();
    let mut outputs = kernel
        .try_apply_with_template(
            &[&q_c, &k_c, &v_c, a_log, &a_c, dt_bias, &b_c, &state_c],
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
                    name: "SeqLen",
                    value: seq,
                },
            ],
            (32, value_head_dim, batch * num_value_heads),
            (32, gated_delta_verify_threadgroup_y(value_head_dim), 1),
            None,
        )
        .ok()?;
    let state_out = outputs.pop()?;
    let y = outputs.pop()?;
    Some((y, state_out))
}

pub(crate) fn slice_seq_row_4d(x: &MlxArray, t: i32) -> MlxArray {
    let shape = x.shape();
    contiguous(
        &slice(
            x,
            &[0, t, 0, 0],
            &[shape[0], t + 1, shape[2], shape[3]],
            &[1, 1, 1, 1],
            None,
        ),
        None,
    )
}

fn slice_seq_row_3d(x: &MlxArray, t: i32) -> MlxArray {
    let shape = x.shape();
    contiguous(
        &slice(
            x,
            &[0, t, 0],
            &[shape[0], t + 1, shape[2]],
            &[1, 1, 1],
            None,
        ),
        None,
    )
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
pub(crate) fn gated_delta_kernel_impl(
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

/// Sequential T=2..=4 decode: same per-token math as
/// [`gated_delta_decode_kernel`], one dispatch, checkpoint after row 0.
#[allow(clippy::too_many_arguments)]
fn gated_delta_decode_seq_kernel(
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
    state_shape: &[i32],
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    if !fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(seq)) {
        return None;
    }
    if key_head_dim <= 0 || key_head_dim % 32 != 0 {
        return None;
    }
    if num_key_heads <= 0 || num_value_heads % num_key_heads != 0 {
        return None;
    }

    let q_c = contiguous(q, None);
    let k_c = contiguous(k, None);
    let v_c = contiguous(v, None);
    let a_c = contiguous(a_raw, None);
    let b_c = contiguous(b_raw, None);
    let state_c = contiguous(state, None);
    let kernel = GATED_DELTA_DECODE_SEQ_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "qwen35_gated_delta_decode_seq_v1",
            &[
                "q", "k", "v", "a_log", "a_raw", "dt_bias", "b_raw", "state_in",
            ],
            &["y", "state_out", "checkpoint"],
            GATED_DELTA_DECODE_SEQ_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel
        .try_apply_with_template(
            &[&q_c, &k_c, &v_c, a_log, &a_c, dt_bias, &b_c, &state_c],
            &[
                KernelOutputSpec {
                    shape: vec![batch, seq, num_value_heads, value_head_dim],
                    dtype: q.dtype(),
                },
                KernelOutputSpec {
                    shape: state_shape.to_vec(),
                    dtype: state.dtype(),
                },
                KernelOutputSpec {
                    shape: state_shape.to_vec(),
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
                    name: "SeqLen",
                    value: seq,
                },
            ],
            (32, value_head_dim, batch * num_value_heads),
            (32, gated_delta_verify_threadgroup_y(value_head_dim), 1),
            None,
        )
        .ok()?;
    let checkpoint = outputs.pop()?;
    let state_out = outputs.pop()?;
    let y = outputs.pop()?;
    Some((y, state_out, checkpoint))
}

/// Sequential T=2..=4 gated-delta update that records only the scalar delta
/// needed to reconstruct an accepted prefix.
///
/// A recurrent checkpoint is `[B, Hv, Dv, Dk]` float32; the tape is only
/// `[B, T, Hv, Dv]` float32. The forward arithmetic is deliberately identical
/// to [`gated_delta_decode_seq_kernel`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_kernel_with_tape(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    b_raw: &MlxArray,
    state: &MlxArray,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    let q_shape = q.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || v_shape.len() != 4 || q_shape[0] != 1 {
        return None;
    }
    let batch = q_shape[0];
    let seq = q_shape[1];
    let num_key_heads = q_shape[2];
    let key_head_dim = q_shape[3];
    let num_value_heads = v_shape[2];
    let value_head_dim = v_shape[3];
    if !fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(seq))
        || key_head_dim <= 0
        || key_head_dim % 32 != 0
        || num_key_heads <= 0
        || num_value_heads % num_key_heads != 0
    {
        return None;
    }

    let q_c = contiguous(q, None);
    let k_c = contiguous(k, None);
    let v_c = contiguous(v, None);
    let a_c = contiguous(a_raw, None);
    let b_c = contiguous(b_raw, None);
    let state_c = contiguous(state, None);

    let kernel = GATED_DELTA_DECODE_SEQ_TAPE_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "qwen35_gated_delta_decode_seq_tape_v1",
            &[
                "q", "k", "v", "a_log", "a_raw", "dt_bias", "b_raw", "state_in",
            ],
            &["y", "state_out", "tape"],
            GATED_DELTA_DECODE_SEQ_TAPE_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let state_shape = state.shape();
    let mut outputs = kernel
        .try_apply_with_template(
            &[&q_c, &k_c, &v_c, a_log, &a_c, dt_bias, &b_c, &state_c],
            &[
                KernelOutputSpec {
                    shape: vec![batch, seq, num_value_heads, value_head_dim],
                    dtype: q.dtype(),
                },
                KernelOutputSpec {
                    shape: state_shape,
                    dtype: state.dtype(),
                },
                KernelOutputSpec {
                    shape: vec![batch, seq, num_value_heads, value_head_dim],
                    dtype: MlxDtype::Float32,
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
                    name: "SeqLen",
                    value: seq,
                },
            ],
            (32, value_head_dim, batch * num_value_heads),
            (32, gated_delta_verify_threadgroup_y(value_head_dim), 1),
            None,
        )
        .ok()?;
    let tape = outputs.pop()?;
    let state_out = outputs.pop()?;
    let y = outputs.pop()?;
    Some((y, state_out, tape))
}

/// Reconstruct the recurrent state after `steps` verifier rows from the
/// pre-verify state and a delta tape produced by
/// [`gated_delta_kernel_with_tape`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay_gated_delta_tape(
    k: &MlxArray,
    a_log: &MlxArray,
    a_raw: &MlxArray,
    dt_bias: &MlxArray,
    tape: &MlxArray,
    state: &MlxArray,
    steps: usize,
) -> Option<MlxArray> {
    let k_shape = k.shape();
    let tape_shape = tape.shape();
    if k_shape.len() != 4 || tape_shape.len() != 4 || k_shape[0] != 1 {
        return None;
    }
    let batch = k_shape[0];
    let seq = k_shape[1];
    let num_key_heads = k_shape[2];
    let key_head_dim = k_shape[3];
    let num_value_heads = tape_shape[2];
    let value_head_dim = tape_shape[3];
    let steps = i32::try_from(steps).ok()?;
    if !(1..=4).contains(&seq)
        || steps <= 0
        || steps > seq
        || key_head_dim <= 0
        || key_head_dim % 32 != 0
        || num_key_heads <= 0
        || num_value_heads % num_key_heads != 0
        || tape_shape[0] != batch
        || tape_shape[1] != seq
    {
        return None;
    }

    let k_c = contiguous(k, None);
    let a_c = contiguous(a_raw, None);
    let tape_c = contiguous(tape, None);
    let state_c = contiguous(state, None);
    let kernel = GATED_DELTA_DECODE_SEQ_TAPE_REPLAY_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "qwen35_gated_delta_decode_seq_tape_replay_v1",
            &["k", "a_log", "a_raw", "dt_bias", "tape", "state_in"],
            &["state_out"],
            GATED_DELTA_DECODE_SEQ_TAPE_REPLAY_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel
        .try_apply_with_template(
            &[&k_c, a_log, &a_c, dt_bias, &tape_c, &state_c],
            &[KernelOutputSpec {
                shape: state.shape(),
                dtype: state.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "InT",
                    dtype: k.dtype(),
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
                    name: "SeqLen",
                    value: seq,
                },
                KernelTemplateArg::Int {
                    name: "Steps",
                    value: steps,
                },
            ],
            (32, value_head_dim, batch * num_value_heads),
            (32, gated_delta_verify_threadgroup_y(value_head_dim), 1),
            None,
        )
        .ok()?;
    outputs.pop()
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
    let exact = skip_rms_norm_gate_metal_for_exact_verify();
    if exact {
        // Factory `dced27d4`: fused Metal on early exact S=2 layers
        // (allow=true) → ON `f4b5490d`. Exact stays portable.
        let _ = allow_full_gate_metal;
        return rms_norm_silu_mul_normed(hidden_states, gate, weight, eps, None);
    }
    if allow_full_gate_metal
        && let Some(gated) = rms_norm_full_gate_metal(hidden_states, gate, weight, eps)
    {
        return gated;
    }
    let normed = rms_norm(hidden_states, Some(weight), eps, None);
    if let Some(gated) = rms_norm_gate_metal(&normed, gate, hidden_states.dtype()) {
        return gated;
    }
    portable_silu_mul_normed(&normed, gate, hidden_states.dtype())
}

/// Uncompiled exact-identity RMSNorm + f32 SiLU*norm graph.
///
/// Metal fused/elementwise gates are not sequence-equivalent under exact
/// MTP-on. This is the matching portable chain; compile only fuses it.
#[cfg(test)]
fn portable_rms_norm_gated(
    hidden_states: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
) -> MlxArray {
    let normed = rms_norm(hidden_states, Some(weight), eps, None);
    portable_silu_mul_normed(&normed, gate, hidden_states.dtype())
}

/// `silu(gate.float32) * normed.float32` then cast back. `silu_mul` is
/// bit-exact vs `silu`+`multiply` on f32 (mlx-sys unit).
fn portable_silu_mul_normed(
    normed: &MlxArray,
    gate: &MlxArray,
    output_dtype: MlxDtype,
) -> MlxArray {
    let gate_f32 = astype(gate, MlxDtype::Float32, None);
    let normed_f32 = astype(normed, MlxDtype::Float32, None);
    let gated = silu_mul(&gate_f32, &normed_f32, None);
    astype(&gated, output_dtype, None)
}

#[cfg(test)]
type SiluMulNormedCompileKey = (Vec<i32>, MlxDtype, MlxDtype, MlxDtype, ThreadId);
#[cfg(test)]
type SiluMulNormedCompileCache = Mutex<HashMap<SiluMulNormedCompileKey, Option<MlxClosure>>>;
#[cfg(test)]
static SILU_MUL_NORMED_COMPILE_CACHE: OnceLock<SiluMulNormedCompileCache> = OnceLock::new();

#[cfg(test)]
fn try_compiled_silu_mul_normed(
    normed: &MlxArray,
    gate: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if normed.shape() != gate.shape() {
        return None;
    }
    let seq = normed.shape().get(1).copied().unwrap_or(0);
    if !fastpath::qwen_linear_mtp_verify_seq_contains(i64::from(seq)) {
        return None;
    }
    let key = (
        normed.shape(),
        normed.dtype(),
        gate.dtype(),
        output_dtype,
        std::thread::current().id(),
    );
    let cache = SILU_MUL_NORMED_COMPILE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().ok()?;
    let slot = guard.entry(key).or_insert_with(|| {
        MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
            let normed = inputs.get(0);
            let gate = inputs.get(1);
            vec![portable_silu_mul_normed(&normed, &gate, output_dtype)]
        })
        .compile(false)
        .ok()
    });
    let closure = slot.as_ref()?;
    let mut outputs = closure.try_apply(&[normed, gate]).ok()?;
    if outputs.len() != 1 {
        return None;
    }
    outputs.pop()
}

fn skip_rms_norm_gate_metal_for_exact_verify() -> bool {
    // Factory MXFP4: any Metal gate on MTP-on (fused or elementwise) flips
    // token 41 (`f4b5490d`) vs MTP-off. Portable silu*norm matches.
    // `AX_MLX_EXACT_RMS_GATE_METAL=1` re-arms the Metal gate for A/B.
    fastpath::qwen_linear_mtp_exact_enabled() && !fastpath::exact_rms_gate_metal_enabled()
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
    // Keep the historical Metal silu (`x/(1+exp(-x))`). Changing it to
    // `x*sigmoid(x)` flips MTP-off trial-2 to `f4b5490d`.
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
        float activated = gate_v / (1.0f + exp(-gate_v));
        out[base + lane] = static_cast<T>(activated * normed);
    }
"#;

// Short-verifier specialization adapted from the Apache-2.0 oMLX GDN
// prework layout. One SIMD lane owns adjacent channels, so Q/K normalization
// needs one SIMD reduction and no threadgroup-memory barrier tree.
// Preserve the stored activation boundaries of conv1d -> sigmoid -> multiply
// and RMSNorm -> scale even when these operations share a Metal dispatch.
const POST_INPUT_ROUNDING_HEADER: &str = r#"
    template <typename T>
    inline float ax_post_input_silu(float accumulator) {
      const T x = static_cast<T>(accumulator);
      const T exponential = static_cast<T>(metal::precise::exp(metal::abs(static_cast<float>(x))));
      const T denominator = static_cast<T>(1) + exponential;
      const T low = static_cast<T>(metal::precise::divide(1.0f, static_cast<float>(denominator)));
      const T gate = x >= static_cast<T>(0) ? static_cast<T>(1) - low : low;
      return static_cast<float>(static_cast<T>(x * gate));
    }

    // MLX 0.32.3 computes the FP16 sigmoid's precise exponential and
    // reciprocal in FP32, then stores the sigmoid before the multiply.
    template <>
    inline float ax_post_input_silu<half>(float accumulator) {
      const half x = static_cast<half>(accumulator);
      const float exponential = metal::precise::exp(metal::abs(static_cast<float>(x)));
      const float low = metal::precise::divide(1.0f, 1.0f + exponential);
      const half gate = static_cast<half>(x >= static_cast<half>(0) ? 1.0f - low : low);
      return static_cast<float>(static_cast<half>(x * gate));
    }

    template <typename T>
    inline T ax_post_input_scaled_norm(float value, float inverse_rms, float scale) {
      const T normalized = static_cast<T>(value * inverse_rms);
      const T typed_scale = static_cast<T>(scale);
      return static_cast<T>(static_cast<float>(normalized) * static_cast<float>(typed_scale));
    }
"#;

const DECODE_POST_INPUT_SIMD32_KERNEL_SOURCE: &str = r#"
    constexpr int KeyDim = Hk * HeadDim;
    constexpr int ValueDim = Hv * HeadDim;
    constexpr int ConvDim = 2 * KeyDim + ValueDim;
    constexpr int TailLen = ConvKernelDim - 1;
    constexpr int Groups = 2 * Hk + Hv;
    constexpr int ValuesPerLane = HeadDim / 32;

    const int lane = thread_position_in_threadgroup.x;
    const int z = thread_position_in_grid.z;
    const int batch_idx = z / Groups;
    const int group_idx = z - batch_idx * Groups;
    const bool is_q = group_idx < Hk;
    const bool is_k = group_idx >= Hk && group_idx < 2 * Hk;
    const int head = is_q ? group_idx
        : (is_k ? group_idx - Hk : group_idx - 2 * Hk);
    const int channel_base = is_q ? head * HeadDim
        : (is_k ? KeyDim + head * HeadDim : 2 * KeyDim + head * HeadDim);

    auto qkv_b = qkv + batch_idx * Seq * ConvDim;
    auto state_b = conv_state + batch_idx * TailLen * ConvDim;
    auto new_state_b = new_conv_state + batch_idx * TailLen * ConvDim;
    auto prefix_state_b = prefix_conv_state + batch_idx * TailLen * ConvDim;

    float tails[ValuesPerLane][ConvKernelDim];
    for (int i = 0; i < ValuesPerLane; ++i) {
      const int channel = channel_base + lane * ValuesPerLane + i;
      for (int t = 0; t < TailLen; ++t) {
        tails[i][t] = static_cast<float>(state_b[t * ConvDim + channel]);
      }
    }

    for (int token = 0; token < Seq; ++token) {
      auto qkv_t = qkv_b + token * ConvDim;
      float activated[ValuesPerLane];
      float sumsq = 0.0f;
      for (int i = 0; i < ValuesPerLane; ++i) {
        const int channel = channel_base + lane * ValuesPerLane + i;
        float acc = static_cast<float>(qkv_t[channel]) *
            static_cast<float>(conv_weight[channel * ConvKernelDim + TailLen]);
        for (int t = 0; t < TailLen; ++t) {
          acc += tails[i][t] *
              static_cast<float>(conv_weight[channel * ConvKernelDim + t]);
        }
        const float value = ax_post_input_silu<T>(acc);
        activated[i] = value;
        sumsq += value * value;
      }

      if (is_q || is_k) {
        const float norm_scale = rsqrt(
            simd_sum(sumsq) / static_cast<float>(HeadDim) + eps[0]);
        for (int i = 0; i < ValuesPerLane; ++i) {
          const int d = lane * ValuesPerLane + i;
          const int out_idx = ((batch_idx * Seq + token) * Hk + head) * HeadDim + d;
          const float scale = is_q ? q_scale[0] : k_scale[0];
          if (is_q) {
            q[out_idx] = ax_post_input_scaled_norm<T>(activated[i], norm_scale, scale);
          } else {
            k[out_idx] = ax_post_input_scaled_norm<T>(activated[i], norm_scale, scale);
          }
        }
      } else {
        for (int i = 0; i < ValuesPerLane; ++i) {
          const int d = lane * ValuesPerLane + i;
          const int out_idx = ((batch_idx * Seq + token) * Hv + head) * HeadDim + d;
          v[out_idx] = static_cast<T>(activated[i]);
        }
      }

      for (int i = 0; i < ValuesPerLane; ++i) {
        const int channel = channel_base + lane * ValuesPerLane + i;
        for (int t = 0; t < TailLen - 1; ++t) {
          tails[i][t] = tails[i][t + 1];
        }
        if (TailLen > 0) {
          tails[i][TailLen - 1] = static_cast<float>(qkv_t[channel]);
        }
        if (token == 0) {
          for (int t = 0; t < TailLen; ++t) {
            prefix_state_b[t * ConvDim + channel] = static_cast<T>(tails[i][t]);
          }
        }
      }
    }

    for (int i = 0; i < ValuesPerLane; ++i) {
      const int channel = channel_base + lane * ValuesPerLane + i;
      for (int t = 0; t < TailLen; ++t) {
        new_state_b[t * ConvDim + channel] = static_cast<T>(tails[i][t]);
      }
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
    auto prefix_state_b = prefix_conv_state + batch_idx * TailLen * ConvDim;

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
      float activated = ax_post_input_silu<T>(acc);

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
              ax_post_input_scaled_norm<T>(activated, norm_scale, q_scale[0]);
        } else {
          int head = group_idx - Hk;
          k[((batch_idx * Seq + token) * Hk + head) * HeadDim + lane] =
              ax_post_input_scaled_norm<T>(activated, norm_scale, k_scale[0]);
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
      // After token 0 the rolling tail is the conv checkpoint used by
      // exact MTP adopt (committed token only).
      if (token == 0) {
        for (int t = 0; t < TailLen; ++t) {
          prefix_state_b[t * ConvDim + channel] = static_cast<T>(tail[t]);
        }
      }
    }

    for (int t = 0; t < TailLen; ++t) {
      new_state_b[t * ConvDim + channel] = static_cast<T>(tail[t]);
    }
"#;

// Short verifier: keep depthwise-conv activations, normalized Q/K, and V in
// threadgroup memory until the gated-delta recurrence consumes them. One
// eight-SIMD-group tile owns eight value rows. Q/K work is shared inside the
// tile; immutable per-token slots avoid the loop-carried shared-scalar race.
const GATED_DELTA_FUSED_VERIFY_KERNEL_SOURCE: &str = r#"
    constexpr int KeyDim = Hk * Dk;
    constexpr int ValueDim = Hv * Dv;
    constexpr int ConvDim = 2 * KeyDim + ValueDim;
    constexpr int TailLen = ConvKernelDim - 1;
    constexpr int NPerLane = Dk / 32;
    constexpr int ValuesPerKeyHead = SeqLen * Dk;

    const int lane = thread_position_in_threadgroup.x;
    const int local_y = thread_position_in_threadgroup.y;
    const int value_tile = threadgroup_position_in_grid.y;
    const int dv_idx = value_tile * Tgy + local_y;
    const int n = thread_position_in_grid.z;
    const int batch_idx = n / Hv;
    const int hv_idx = n % Hv;
    const int values_per_key = Hv / Hk;
    const int hk_idx = hv_idx / values_per_key;
    const int s_base = lane * NPerLane;

    auto qkv_b = qkv + batch_idx * SeqLen * ConvDim;
    auto conv_state_b = conv_state + batch_idx * TailLen * ConvDim;
    auto new_conv_b = new_conv_state + batch_idx * TailLen * ConvDim;
    auto prefix_conv_b = prefix_conv_state + batch_idx * TailLen * ConvDim;

    threadgroup InT q_values[ValuesPerKeyHead];
    threadgroup InT k_values[ValuesPerKeyHead];
    threadgroup InT v_values[SeqLen * Tgy];
    threadgroup float g_values[SeqLen];
    threadgroup float beta_values[SeqLen];

    if (thread_index_in_threadgroup == 0) {
      const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
      const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
      for (int token = 0; token < SeqLen; ++token) {
        float a_plus_dt =
            static_cast<float>(a_raw[(batch_idx * SeqLen + token) * Hv + hv_idx]) +
            dt_bias_v;
        float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
        g_values[token] = exp(-exp_a_log * sp);
        float b_val =
            static_cast<float>(b_raw[(batch_idx * SeqLen + token) * Hv + hv_idx]);
        beta_values[token] = static_cast<float>(
            static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
      }
    }

    // The first SIMD group computes the mapped key head once for this value
    // tile. Each lane owns NPerLane adjacent dimensions and the SIMD reduction
    // publishes the two RMS denominators without a divergent barrier.
    if (local_y == 0) {
      float q_tail[NPerLane][TailLen];
      float k_tail[NPerLane][TailLen];
      for (int i = 0; i < NPerLane; ++i) {
        const int d = s_base + i;
        const int q_channel = hk_idx * Dk + d;
        const int k_channel = KeyDim + hk_idx * Dk + d;
        for (int t = 0; t < TailLen; ++t) {
          q_tail[i][t] =
              static_cast<float>(conv_state_b[t * ConvDim + q_channel]);
          k_tail[i][t] =
              static_cast<float>(conv_state_b[t * ConvDim + k_channel]);
        }
      }

      for (int token = 0; token < SeqLen; ++token) {
        float q_raw[NPerLane];
        float k_raw[NPerLane];
        float q_square_sum = 0.0f;
        float k_square_sum = 0.0f;
        auto qkv_t = qkv_b + token * ConvDim;
        for (int i = 0; i < NPerLane; ++i) {
          const int d = s_base + i;
          const int q_channel = hk_idx * Dk + d;
          const int k_channel = KeyDim + hk_idx * Dk + d;
          float q_acc = static_cast<float>(qkv_t[q_channel]) *
              static_cast<float>(
                  conv_weight[q_channel * ConvKernelDim + TailLen]);
          float k_acc = static_cast<float>(qkv_t[k_channel]) *
              static_cast<float>(
                  conv_weight[k_channel * ConvKernelDim + TailLen]);
          for (int t = 0; t < TailLen; ++t) {
            q_acc += q_tail[i][t] * static_cast<float>(
                conv_weight[q_channel * ConvKernelDim + t]);
            k_acc += k_tail[i][t] * static_cast<float>(
                conv_weight[k_channel * ConvKernelDim + t]);
          }
          q_raw[i] = ax_post_input_silu<InT>(q_acc);
          k_raw[i] = ax_post_input_silu<InT>(k_acc);
          q_square_sum += q_raw[i] * q_raw[i];
          k_square_sum += k_raw[i] * k_raw[i];
        }
        q_square_sum = simd_sum(q_square_sum);
        k_square_sum = simd_sum(k_square_sum);
        const float q_norm =
            rsqrt(q_square_sum / static_cast<float>(Dk) + eps[0]);
        const float k_norm =
            rsqrt(k_square_sum / static_cast<float>(Dk) + eps[0]);
        for (int i = 0; i < NPerLane; ++i) {
          const int d = s_base + i;
          q_values[token * Dk + d] =
              ax_post_input_scaled_norm<InT>(q_raw[i], q_norm, q_scale[0]);
          k_values[token * Dk + d] =
              ax_post_input_scaled_norm<InT>(k_raw[i], k_norm, k_scale[0]);

          const int q_channel = hk_idx * Dk + d;
          const int k_channel = KeyDim + hk_idx * Dk + d;
          for (int t = 0; t < TailLen - 1; ++t) {
            q_tail[i][t] = q_tail[i][t + 1];
            k_tail[i][t] = k_tail[i][t + 1];
          }
          q_tail[i][TailLen - 1] = static_cast<float>(qkv_t[q_channel]);
          k_tail[i][TailLen - 1] = static_cast<float>(qkv_t[k_channel]);
          if (token == 0 && value_tile == 0 && hv_idx % values_per_key == 0) {
            for (int t = 0; t < TailLen; ++t) {
              prefix_conv_b[t * ConvDim + q_channel] =
                  static_cast<InT>(q_tail[i][t]);
              prefix_conv_b[t * ConvDim + k_channel] =
                  static_cast<InT>(k_tail[i][t]);
            }
          }
        }
      }

      if (value_tile == 0 && hv_idx % values_per_key == 0) {
        for (int i = 0; i < NPerLane; ++i) {
          const int d = s_base + i;
          const int q_channel = hk_idx * Dk + d;
          const int k_channel = KeyDim + hk_idx * Dk + d;
          for (int t = 0; t < TailLen; ++t) {
            new_conv_b[t * ConvDim + q_channel] =
                static_cast<InT>(q_tail[i][t]);
            new_conv_b[t * ConvDim + k_channel] =
                static_cast<InT>(k_tail[i][t]);
          }
        }
      }
    }

    // One lane per SIMD group owns the value channel's tiny conv tail.
    if (lane == 0) {
      const int v_channel = 2 * KeyDim + hv_idx * Dv + dv_idx;
      float v_tail[TailLen];
      for (int t = 0; t < TailLen; ++t) {
        v_tail[t] = static_cast<float>(conv_state_b[t * ConvDim + v_channel]);
      }
      for (int token = 0; token < SeqLen; ++token) {
        auto qkv_t = qkv_b + token * ConvDim;
        float v_acc = static_cast<float>(qkv_t[v_channel]) *
            static_cast<float>(
                conv_weight[v_channel * ConvKernelDim + TailLen]);
        for (int t = 0; t < TailLen; ++t) {
          v_acc += v_tail[t] * static_cast<float>(
              conv_weight[v_channel * ConvKernelDim + t]);
        }
        v_values[token * Tgy + local_y] =
            static_cast<InT>(ax_post_input_silu<InT>(v_acc));
        for (int t = 0; t < TailLen - 1; ++t) {
          v_tail[t] = v_tail[t + 1];
        }
        v_tail[TailLen - 1] = static_cast<float>(qkv_t[v_channel]);
        if (token == 0) {
          for (int t = 0; t < TailLen; ++t) {
            prefix_conv_b[t * ConvDim + v_channel] =
                static_cast<InT>(v_tail[t]);
          }
        }
      }
      for (int t = 0; t < TailLen; ++t) {
        new_conv_b[t * ConvDim + v_channel] = static_cast<InT>(v_tail[t]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    auto ck_state = checkpoint + (n * Dv + dv_idx) * Dk;
    float state[NPerLane];
    for (int i = 0; i < NPerLane; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    for (int token = 0; token < SeqLen; ++token) {
      float kv_mem = 0.0f;
      for (int i = 0; i < NPerLane; ++i) {
        const float k = static_cast<float>(k_values[token * Dk + s_base + i]);
        state[i] *= g_values[token];
        kv_mem += state[i] * k;
      }
      kv_mem = simd_sum(kv_mem);
      const float delta =
          (static_cast<float>(v_values[token * Tgy + local_y]) - kv_mem) *
          beta_values[token];
      float out = 0.0f;
      for (int i = 0; i < NPerLane; ++i) {
        const float k = static_cast<float>(k_values[token * Dk + s_base + i]);
        const float q = static_cast<float>(q_values[token * Dk + s_base + i]);
        state[i] += k * delta;
        out += state[i] * q;
      }
      out = simd_sum(out);
      if (lane == 0) {
        y[((batch_idx * SeqLen + token) * Hv + hv_idx) * Dv + dv_idx] =
            static_cast<InT>(out);
      }
      if (token == 0) {
        for (int i = 0; i < NPerLane; ++i) {
          ck_state[s_base + i] = static_cast<StT>(state[i]);
        }
      }
    }
    for (int i = 0; i < NPerLane; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

/// Lockstep with [`GATED_DELTA_FUSED_VERIFY_KERNEL_SOURCE`]: same per-token
/// math, no row-0 recurrent checkpoint and no conv-prefix buffers.
const GATED_DELTA_FUSED_VERIFY_NO_CHECKPOINT_KERNEL_SOURCE: &str = r#"
    constexpr int KeyDim = Hk * Dk;
    constexpr int ValueDim = Hv * Dv;
    constexpr int ConvDim = 2 * KeyDim + ValueDim;
    constexpr int TailLen = ConvKernelDim - 1;
    constexpr int NPerLane = Dk / 32;
    constexpr int ValuesPerKeyHead = SeqLen * Dk;

    const int lane = thread_position_in_threadgroup.x;
    const int local_y = thread_position_in_threadgroup.y;
    const int value_tile = threadgroup_position_in_grid.y;
    const int dv_idx = value_tile * Tgy + local_y;
    const int n = thread_position_in_grid.z;
    const int batch_idx = n / Hv;
    const int hv_idx = n % Hv;
    const int values_per_key = Hv / Hk;
    const int hk_idx = hv_idx / values_per_key;
    const int s_base = lane * NPerLane;

    auto qkv_b = qkv + batch_idx * SeqLen * ConvDim;
    auto conv_state_b = conv_state + batch_idx * TailLen * ConvDim;
    auto new_conv_b = new_conv_state + batch_idx * TailLen * ConvDim;

    threadgroup InT q_values[ValuesPerKeyHead];
    threadgroup InT k_values[ValuesPerKeyHead];
    threadgroup InT v_values[SeqLen * Tgy];
    threadgroup float g_values[SeqLen];
    threadgroup float beta_values[SeqLen];

    if (thread_index_in_threadgroup == 0) {
      const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
      const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
      for (int token = 0; token < SeqLen; ++token) {
        float a_plus_dt =
            static_cast<float>(a_raw[(batch_idx * SeqLen + token) * Hv + hv_idx]) +
            dt_bias_v;
        float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
        g_values[token] = exp(-exp_a_log * sp);
        float b_val =
            static_cast<float>(b_raw[(batch_idx * SeqLen + token) * Hv + hv_idx]);
        beta_values[token] = static_cast<float>(
            static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
      }
    }

    // The first SIMD group computes the mapped key head once for this value
    // tile. Each lane owns NPerLane adjacent dimensions and the SIMD reduction
    // publishes the two RMS denominators without a divergent barrier.
    if (local_y == 0) {
      float q_tail[NPerLane][TailLen];
      float k_tail[NPerLane][TailLen];
      for (int i = 0; i < NPerLane; ++i) {
        const int d = s_base + i;
        const int q_channel = hk_idx * Dk + d;
        const int k_channel = KeyDim + hk_idx * Dk + d;
        for (int t = 0; t < TailLen; ++t) {
          q_tail[i][t] =
              static_cast<float>(conv_state_b[t * ConvDim + q_channel]);
          k_tail[i][t] =
              static_cast<float>(conv_state_b[t * ConvDim + k_channel]);
        }
      }

      for (int token = 0; token < SeqLen; ++token) {
        float q_raw[NPerLane];
        float k_raw[NPerLane];
        float q_square_sum = 0.0f;
        float k_square_sum = 0.0f;
        auto qkv_t = qkv_b + token * ConvDim;
        for (int i = 0; i < NPerLane; ++i) {
          const int d = s_base + i;
          const int q_channel = hk_idx * Dk + d;
          const int k_channel = KeyDim + hk_idx * Dk + d;
          float q_acc = static_cast<float>(qkv_t[q_channel]) *
              static_cast<float>(
                  conv_weight[q_channel * ConvKernelDim + TailLen]);
          float k_acc = static_cast<float>(qkv_t[k_channel]) *
              static_cast<float>(
                  conv_weight[k_channel * ConvKernelDim + TailLen]);
          for (int t = 0; t < TailLen; ++t) {
            q_acc += q_tail[i][t] * static_cast<float>(
                conv_weight[q_channel * ConvKernelDim + t]);
            k_acc += k_tail[i][t] * static_cast<float>(
                conv_weight[k_channel * ConvKernelDim + t]);
          }
          q_raw[i] = ax_post_input_silu<InT>(q_acc);
          k_raw[i] = ax_post_input_silu<InT>(k_acc);
          q_square_sum += q_raw[i] * q_raw[i];
          k_square_sum += k_raw[i] * k_raw[i];
        }
        q_square_sum = simd_sum(q_square_sum);
        k_square_sum = simd_sum(k_square_sum);
        const float q_norm =
            rsqrt(q_square_sum / static_cast<float>(Dk) + eps[0]);
        const float k_norm =
            rsqrt(k_square_sum / static_cast<float>(Dk) + eps[0]);
        for (int i = 0; i < NPerLane; ++i) {
          const int d = s_base + i;
          q_values[token * Dk + d] =
              ax_post_input_scaled_norm<InT>(q_raw[i], q_norm, q_scale[0]);
          k_values[token * Dk + d] =
              ax_post_input_scaled_norm<InT>(k_raw[i], k_norm, k_scale[0]);

          const int q_channel = hk_idx * Dk + d;
          const int k_channel = KeyDim + hk_idx * Dk + d;
          for (int t = 0; t < TailLen - 1; ++t) {
            q_tail[i][t] = q_tail[i][t + 1];
            k_tail[i][t] = k_tail[i][t + 1];
          }
          q_tail[i][TailLen - 1] = static_cast<float>(qkv_t[q_channel]);
          k_tail[i][TailLen - 1] = static_cast<float>(qkv_t[k_channel]);
        }
      }

      if (value_tile == 0 && hv_idx % values_per_key == 0) {
        for (int i = 0; i < NPerLane; ++i) {
          const int d = s_base + i;
          const int q_channel = hk_idx * Dk + d;
          const int k_channel = KeyDim + hk_idx * Dk + d;
          for (int t = 0; t < TailLen; ++t) {
            new_conv_b[t * ConvDim + q_channel] =
                static_cast<InT>(q_tail[i][t]);
            new_conv_b[t * ConvDim + k_channel] =
                static_cast<InT>(k_tail[i][t]);
          }
        }
      }
    }

    // One lane per SIMD group owns the value channel's tiny conv tail.
    if (lane == 0) {
      const int v_channel = 2 * KeyDim + hv_idx * Dv + dv_idx;
      float v_tail[TailLen];
      for (int t = 0; t < TailLen; ++t) {
        v_tail[t] = static_cast<float>(conv_state_b[t * ConvDim + v_channel]);
      }
      for (int token = 0; token < SeqLen; ++token) {
        auto qkv_t = qkv_b + token * ConvDim;
        float v_acc = static_cast<float>(qkv_t[v_channel]) *
            static_cast<float>(
                conv_weight[v_channel * ConvKernelDim + TailLen]);
        for (int t = 0; t < TailLen; ++t) {
          v_acc += v_tail[t] * static_cast<float>(
              conv_weight[v_channel * ConvKernelDim + t]);
        }
        v_values[token * Tgy + local_y] =
            static_cast<InT>(ax_post_input_silu<InT>(v_acc));
        for (int t = 0; t < TailLen - 1; ++t) {
          v_tail[t] = v_tail[t + 1];
        }
        v_tail[TailLen - 1] = static_cast<float>(qkv_t[v_channel]);
      }
      for (int t = 0; t < TailLen; ++t) {
        new_conv_b[t * ConvDim + v_channel] = static_cast<InT>(v_tail[t]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    float state[NPerLane];
    for (int i = 0; i < NPerLane; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    for (int token = 0; token < SeqLen; ++token) {
      float kv_mem = 0.0f;
      for (int i = 0; i < NPerLane; ++i) {
        const float k = static_cast<float>(k_values[token * Dk + s_base + i]);
        state[i] *= g_values[token];
        kv_mem += state[i] * k;
      }
      kv_mem = simd_sum(kv_mem);
      const float delta =
          (static_cast<float>(v_values[token * Tgy + local_y]) - kv_mem) *
          beta_values[token];
      float out = 0.0f;
      for (int i = 0; i < NPerLane; ++i) {
        const float k = static_cast<float>(k_values[token * Dk + s_base + i]);
        const float q = static_cast<float>(q_values[token * Dk + s_base + i]);
        state[i] += k * delta;
        out += state[i] * q;
      }
      out = simd_sum(out);
      if (lane == 0) {
        y[((batch_idx * SeqLen + token) * Hv + hv_idx) * Dv + dv_idx] =
            static_cast<InT>(out);
      }
    }
    for (int i = 0; i < NPerLane; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
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

      // g_t/beta_t are single threadgroup-shared scalars reused every
      // iteration (unlike the cached-array kernel below). Without this
      // barrier a fast SIMD-group can race into the next iteration's
      // leader-thread write while a slower SIMD-group is still reading
      // this iteration's values, corrupting its state update.
      threadgroup_barrier(mem_flags::mem_threadgroup);
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

/// Same per-token body as `qwen35_gated_delta_decode_v1`, looped over
/// `SeqLen` (2..=4) with T-aware strides. Checkpoint is state after t=0.
const GATED_DELTA_DECODE_SEQ_KERNEL_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    const int s_base = n_per_t * dk_idx;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    auto ck_state = checkpoint + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    threadgroup float g_values[SeqLen];
    threadgroup float beta_values[SeqLen];
    if (thread_index_in_threadgroup == 0) {
      const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
      const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
      for (int t = 0; t < SeqLen; ++t) {
        float a_plus_dt =
            static_cast<float>(a_raw[(b_idx * SeqLen + t) * Hv + hv_idx]) + dt_bias_v;
        float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
        g_values[t] = exp(-exp_a_log * sp);
        float b_val = static_cast<float>(b_raw[(b_idx * SeqLen + t) * Hv + hv_idx]);
        beta_values[t] = static_cast<float>(
            static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < SeqLen; ++t) {
      const float g_t = g_values[t];
      const float beta_t = beta_values[t];

      auto q_ = q + ((b_idx * SeqLen + t) * Hk + hk_idx) * Dk;
      auto k_ = k + ((b_idx * SeqLen + t) * Hk + hk_idx) * Dk;
      auto v_ = v + ((b_idx * SeqLen + t) * Hv + hv_idx) * Dv;
      auto y_ = y + ((b_idx * SeqLen + t) * Hv + hv_idx) * Dv;
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
        y_[dv_idx] = static_cast<InT>(out);
      }

      if (t == 0) {
        for (int i = 0; i < n_per_t; ++i) {
          ck_state[s_base + i] = static_cast<StT>(state[i]);
        }
      }
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

/// Two-output counterpart of [`GATED_DELTA_DECODE_SEQ_KERNEL_SOURCE`].
/// Keep the arithmetic in lockstep; only the row-0 checkpoint write is absent.
const GATED_DELTA_DECODE_SEQ_NO_CHECKPOINT_KERNEL_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    const int s_base = n_per_t * dk_idx;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    threadgroup float g_values[SeqLen];
    threadgroup float beta_values[SeqLen];
    if (thread_index_in_threadgroup == 0) {
      const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
      const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
      for (int t = 0; t < SeqLen; ++t) {
        float a_plus_dt =
            static_cast<float>(a_raw[(b_idx * SeqLen + t) * Hv + hv_idx]) + dt_bias_v;
        float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
        g_values[t] = exp(-exp_a_log * sp);
        float b_val = static_cast<float>(b_raw[(b_idx * SeqLen + t) * Hv + hv_idx]);
        beta_values[t] =
            static_cast<float>(static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < SeqLen; ++t) {
      const float g_t = g_values[t];
      const float beta_t = beta_values[t];

      auto q_ = q + ((b_idx * SeqLen + t) * Hk + hk_idx) * Dk;
      auto k_ = k + ((b_idx * SeqLen + t) * Hk + hk_idx) * Dk;
      auto v_ = v + ((b_idx * SeqLen + t) * Hv + hv_idx) * Dv;
      auto y_ = y + ((b_idx * SeqLen + t) * Hv + hv_idx) * Dv;
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
        y_[dv_idx] = static_cast<InT>(out);
      }
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

/// Tape form of [`GATED_DELTA_DECODE_SEQ_KERNEL_SOURCE`]. Keep the arithmetic
/// in lockstep; only the third output differs.
const GATED_DELTA_DECODE_SEQ_TAPE_KERNEL_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    const int s_base = n_per_t * dk_idx;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    threadgroup float g_values[SeqLen];
    threadgroup float beta_values[SeqLen];
    if (thread_index_in_threadgroup == 0) {
      const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
      const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
      for (int t = 0; t < SeqLen; ++t) {
        float a_plus_dt =
            static_cast<float>(a_raw[(b_idx * SeqLen + t) * Hv + hv_idx]) + dt_bias_v;
        float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
        g_values[t] = exp(-exp_a_log * sp);
        float b_val = static_cast<float>(b_raw[(b_idx * SeqLen + t) * Hv + hv_idx]);
        beta_values[t] =
            static_cast<float>(static_cast<InT>(1.0f / (1.0f + exp(-b_val))));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < SeqLen; ++t) {
      const float g_t = g_values[t];
      const float beta_t = beta_values[t];

      auto q_ = q + ((b_idx * SeqLen + t) * Hk + hk_idx) * Dk;
      auto k_ = k + ((b_idx * SeqLen + t) * Hk + hk_idx) * Dk;
      auto v_ = v + ((b_idx * SeqLen + t) * Hv + hv_idx) * Dv;
      auto y_ = y + ((b_idx * SeqLen + t) * Hv + hv_idx) * Dv;
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
        y_[dv_idx] = static_cast<InT>(out);
        tape[((b_idx * SeqLen + t) * Hv + hv_idx) * Dv + dv_idx] = delta;
      }
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

const GATED_DELTA_DECODE_SEQ_TAPE_REPLAY_KERNEL_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    const int s_base = n_per_t * dk_idx;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    threadgroup float g_values[Steps];
    if (thread_index_in_threadgroup == 0) {
      const float exp_a_log = exp(static_cast<float>(a_log[hv_idx]));
      const float dt_bias_v = static_cast<float>(dt_bias[hv_idx]);
      for (int t = 0; t < Steps; ++t) {
        float a_plus_dt =
            static_cast<float>(a_raw[(b_idx * SeqLen + t) * Hv + hv_idx]) + dt_bias_v;
        float sp = a_plus_dt > 20.0f ? a_plus_dt : log1p(exp(a_plus_dt));
        g_values[t] = exp(-exp_a_log * sp);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < Steps; ++t) {
      const float g_t = g_values[t];

      auto k_ = k + ((b_idx * SeqLen + t) * Hk + hk_idx) * Dk;
      const float delta =
          tape[((b_idx * SeqLen + t) * Hv + hv_idx) * Dv + dv_idx];
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] * g_t;
        state[i] = state[i] + static_cast<float>(k_[s_base + i]) * delta;
      }
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

#[cfg(test)]
mod tests;
