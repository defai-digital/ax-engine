use std::sync::OnceLock;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, astype, concatenate,
    conv1d, multiply, reshape, rms_norm, slice, slice_last_dim, zeros,
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
static GATED_DELTA_DECODE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static DECODE_POST_INPUT_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static RMS_NORM_GATE_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
pub(crate) const GATED_DELTA_SHORT_THREADGROUP_CACHE_CAPACITY: usize = 512;
pub(crate) const GATED_DELTA_MEDIUM_THREADGROUP_CACHE_CAPACITY: usize = 1024;
pub(crate) const GATED_DELTA_THREADGROUP_CACHE_CAPACITY: usize = 2048;

/// compute_g from mlx-lm/mlx-swift-lm:
/// `exp(-exp(A_log.float32) * softplus(a + dt_bias))`.
///
/// This function is retained for testing; the production path folds this
/// computation into `gated_delta_kernel` to avoid the 7-node lazy graph.
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
    astype(&g, a.dtype(), None)
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
pub(crate) fn linear_attention_decode_post_input_metal(
    cfg: &LinearAttentionConfig,
    qkv: &MlxArray,
    conv_weight: &MlxArray,
    cached_conv_state: Option<&MlxArray>,
    q_scale: f32,
    k_scale: f32,
    eps: f32,
) -> Option<(MlxArray, MlxArray, MlxArray, MlxArray)> {
    if !fastpath::qwen_linear_attention_decode_post_input_metal_enabled() {
        return None;
    }
    let conv_state = cached_conv_state?;
    let qkv_shape = qkv.shape();
    if qkv_shape.len() != 3 || qkv_shape[1] != 1 {
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
    if conv_state.shape() != vec![batch, tail_len, conv_dim] {
        return None;
    }
    if conv_weight.shape() != vec![conv_dim, cfg.conv_kernel_dim as i32, 1] {
        return None;
    }

    let kernel = DECODE_POST_INPUT_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_linear_attention_decode_post_input_v1",
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
                shape: vec![batch, 1, cfg.num_key_heads as i32, head_dim],
                dtype: qkv.dtype(),
            },
            KernelOutputSpec {
                shape: vec![batch, 1, cfg.num_key_heads as i32, head_dim],
                dtype: qkv.dtype(),
            },
            KernelOutputSpec {
                shape: vec![batch, 1, cfg.num_value_heads as i32, head_dim],
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
    let q_shape = q.shape();
    let v_shape = v.shape();
    let state_shape = state.shape();
    let batch = q_shape[0];
    let seq = q_shape[1];
    let num_key_heads = q_shape[2];
    let key_head_dim = q_shape[3];
    let num_value_heads = v_shape[2];
    let value_head_dim = v_shape[3];
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
    let seq_i32 = scalar_i32(seq);
    assert!(
        seq <= GATED_DELTA_THREADGROUP_CACHE_CAPACITY as i32,
        "gated_delta_kernel t_len ({seq}) exceeds threadgroup cache capacity ({GATED_DELTA_THREADGROUP_CACHE_CAPACITY})"
    );
    // Three-tier specialization. The 2048 specialization was measured to lose
    // ~15% per-token throughput in the gated-delta recurrent loop on Qwen 3.6
    // 27B (Hv=48) vs the 512 specialization, because doubling `CacheCapacity`
    // doubles the per-threadgroup `g_t_cache`/`beta_t_cache` allocation and
    // halves SM occupancy. The 1024 tier recovers most of that occupancy while
    // still amortizing dispatch over a 1024-token chunk; `runner.rs` caps the
    // linear-attention prefill chunk to 1024 so the long tier is reserved for
    // exceptional callers that explicitly opt in to a larger chunk.
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
pub fn rms_norm_gated(
    hidden_states: &MlxArray,
    gate: &MlxArray,
    weight: &MlxArray,
    eps: f32,
) -> MlxArray {
    let normed = rms_norm(hidden_states, Some(weight), eps, None);
    if let Some(gated) = rms_norm_gate_metal(&normed, gate, hidden_states.dtype()) {
        return gated;
    }
    let gate_f32 = astype(gate, MlxDtype::Float32, None);
    let normed_f32 = astype(&normed, MlxDtype::Float32, None);
    let gated = multiply(&mlx_sys::ops::silu(&gate_f32, None), &normed_f32, None);
    astype(&gated, hidden_states.dtype(), None)
}

fn rms_norm_gate_metal(
    normed: &MlxArray,
    gate: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if !fastpath::linear_attention_rms_norm_gate_metal_enabled() {
        return None;
    }
    rms_norm_gate_metal_impl(normed, gate, output_dtype)
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
    float activated = gate_v / (1.0f + exp(-gate_v));
    out[idx] = static_cast<T>(activated * normed_v);
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

    auto qkv_b = qkv + batch_idx * ConvDim;
    auto state_b = conv_state + batch_idx * TailLen * ConvDim;
    auto new_state_b = new_conv_state + batch_idx * TailLen * ConvDim;

    float acc = static_cast<float>(qkv_b[channel]) *
        static_cast<float>(conv_weight[channel * ConvKernelDim + TailLen]);
    for (int t = 0; t < TailLen; ++t) {
      acc += static_cast<float>(state_b[t * ConvDim + channel]) *
          static_cast<float>(conv_weight[channel * ConvKernelDim + t]);
    }
    float activated = acc / (1.0f + exp(-acc));

    for (int t = 0; t < TailLen - 1; ++t) {
      new_state_b[t * ConvDim + channel] = state_b[(t + 1) * ConvDim + channel];
    }
    if (TailLen > 0) {
      new_state_b[(TailLen - 1) * ConvDim + channel] =
          static_cast<T>(qkv_b[channel]);
    }

    if (is_q || is_k) {
      squares[lane] = activated * activated;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int stride = HeadDim >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
          squares[lane] += squares[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      float norm_scale = rsqrt(squares[0] / static_cast<float>(HeadDim) + eps[0]);
      if (is_q) {
        int head = group_idx;
        q[(batch_idx * Hk + head) * HeadDim + lane] =
            static_cast<T>(activated * norm_scale * q_scale[0]);
      } else {
        int head = group_idx - Hk;
        k[(batch_idx * Hk + head) * HeadDim + lane] =
            static_cast<T>(activated * norm_scale * k_scale[0]);
      }
    } else {
      int head = group_idx - 2 * Hk;
      v[(batch_idx * Hv + head) * HeadDim + lane] = static_cast<T>(activated);
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
    // CacheCapacity is specialized from Rust into three tiers:
    //   512  — decode/short prompts (smallest threadgroup allocation),
    //   1024 — default long-prefill tier (matches the runner's prefill-chunk
    //          cap for linear-attention models; recovers SM occupancy vs the
    //          2048 tier on M5 Max),
    //   2048 — opt-in for callers that override `--prefill-chunk` to 2048+.
    // Each tier doubles the per-threadgroup `g_t_cache`/`beta_t_cache`
    // footprint, so smaller is faster when the prompt fits.
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
    fn compute_gated_delta_g_preserves_activation_shape_and_dtype() {
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
        assert_eq!(g.dtype(), MlxDtype::Bfloat16);
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
    #[should_panic(expected = "exceeds threadgroup cache capacity")]
    fn gated_delta_kernel_rejects_seq_above_threadgroup_cache_capacity() {
        let seq = (GATED_DELTA_THREADGROUP_CACHE_CAPACITY + 1) as i32;
        let q = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
        let k = zeros(&[1, seq, 1, 32], MlxDtype::Float32, None);
        let v = zeros(&[1, seq, 1, 4], MlxDtype::Float32, None);
        let a_log = zeros(&[1], MlxDtype::Float32, None);
        let a_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
        let dt_bias = zeros(&[1], MlxDtype::Float32, None);
        let b_raw = zeros(&[1, seq, 1], MlxDtype::Float32, None);
        let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

        let _ = gated_delta_kernel(&q, &k, &v, &a_log, &a_raw, &dt_bias, &b_raw, &state);
    }

    #[test]
    fn gated_delta_kernel_accepts_medium_prefill_specialization() {
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
    fn decode_post_input_metal_matches_portable_composition() {
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
        let qkv_data: Vec<f32> = (0..conv_dim)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.01)
            .collect();
        let state_data: Vec<f32> = (0..3 * conv_dim)
            .map(|idx| ((idx % 13) as f32 - 6.0) * 0.005)
            .collect();
        let weight_data: Vec<f32> = (0..conv_dim * cfg.conv_kernel_dim)
            .map(|idx| ((idx % 7) as f32 - 3.0) * 0.02)
            .collect();
        let qkv = f32_array(&qkv_data, &[1, 1, conv_dim as i32]);
        let state = f32_array(&state_data, &[1, 3, conv_dim as i32]);
        let weight = f32_array(
            &weight_data,
            &[conv_dim as i32, cfg.conv_kernel_dim as i32, 1],
        );

        let (conv_out, portable_state) = linear_attention_conv1d(&cfg, &qkv, &weight, Some(&state));
        let split = split_linear_attention_qkv(&cfg, &conv_out);
        let (portable_q, portable_k) =
            normalize_linear_attention_qk(&cfg, &split.q, &split.k, 1e-6);
        let (metal_q, metal_k, metal_v, metal_state) = linear_attention_decode_post_input_metal(
            &cfg,
            &qkv,
            &weight,
            Some(&state),
            q_scale,
            k_scale,
            1e-6,
        )
        .expect("decode post-input Metal path should accept Qwen-like shape");
        mlx_sys::eval(&[
            &portable_q,
            &portable_k,
            &split.v,
            &portable_state,
            &metal_q,
            &metal_k,
            &metal_v,
            &metal_state,
        ]);

        assert_close(
            "decode_post_input_q",
            metal_q.data_f32(),
            portable_q.data_f32(),
            1e-5,
        );
        assert_close(
            "decode_post_input_k",
            metal_k.data_f32(),
            portable_k.data_f32(),
            1e-5,
        );
        assert_close(
            "decode_post_input_v",
            metal_v.data_f32(),
            split.v.data_f32(),
            1e-5,
        );
        assert_close(
            "decode_post_input_state",
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
}
