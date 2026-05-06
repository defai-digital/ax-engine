use std::sync::OnceLock;

use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, add, astype,
    concatenate, conv1d, exp, log1p, multiply, negative, reshape, rms_norm, slice, slice_last_dim,
    where_cond, zeros,
};

use crate::model::LinearAttentionConfig;

/// Split Qwen3.5 gated-delta conv output into shaped q/k/v tensors.
pub struct LinearAttentionQkv {
    pub q: MlxArray,
    pub k: MlxArray,
    pub v: MlxArray,
}

static GATED_DELTA_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// compute_g from mlx-lm/mlx-swift-lm:
/// `exp(-exp(A_log.float32) * softplus(a + dt_bias))`.
pub fn compute_gated_delta_g(a_log: &MlxArray, a: &MlxArray, dt_bias: &MlxArray) -> MlxArray {
    let a_log_f32 = astype(a_log, MlxDtype::Float32, None);
    let decay_rate = exp(&a_log_f32, None);
    let a_plus_bias = add(a, dt_bias, None);
    let softplus = log1p(&exp(&a_plus_bias, None), None);
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
    mask: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let shape = qkv.shape();
    let batch = shape[0];
    let seq = shape[1];
    let conv_dim = cfg.conv_dim() as i32;
    let tail_len = cfg.conv_kernel_dim as i32 - 1;
    let dtype = qkv.dtype();

    let qkv = if let Some(mask) = mask {
        let mask = mlx_sys::expand_dims(mask, -1, None);
        let zeros = zeros(&[batch, seq, conv_dim], dtype, None);
        where_cond(&mask, qkv, &zeros, None)
    } else {
        qkv.clone()
    };

    let conv_state = cached_conv_state
        .cloned()
        .unwrap_or_else(|| zeros(&[batch, tail_len, conv_dim], dtype, None));
    let conv_input = concatenate(&[&conv_state, &qkv], 1, None);
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
) -> (MlxArray, MlxArray) {
    let (q_scale, k_scale) = linear_attention_qk_scale(cfg.key_head_dim);
    let q_normed = rms_norm(q, None, 1e-6, None);
    let k_normed = rms_norm(k, None, 1e-6, None);
    let q_scale = scalar_f32_as(q_scale, q.dtype());
    let k_scale = scalar_f32_as(k_scale, k.dtype());
    (
        multiply(&q_normed, &q_scale, None),
        multiply(&k_normed, &k_scale, None),
    )
}

fn linear_attention_qk_scale(key_head_dim: usize) -> (f32, f32) {
    let inv_scale = (key_head_dim as f32).powf(-0.5);
    (inv_scale, inv_scale)
}

/// Run Qwen3.5's gated-delta recurrent update with the MLX Metal kernel.
///
/// Shapes match mlx-lm/mlx-swift-lm:
/// - `q`, `k`: `[B, T, Hk, Dk]`
/// - `v`: `[B, T, Hv, Dv]`
/// - `g`, `beta`: `[B, T, Hv]`
/// - `state`: `[B, Hv, Dv, Dk]`
/// - returns `(y: [B, T, Hv, Dv], state: [B, Hv, Dv, Dk])`
pub fn gated_delta_kernel(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    g: &MlxArray,
    beta: &MlxArray,
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
    let seq_i32 = scalar_i32(seq);
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
            "qwen35_gated_delta_step",
            &["q", "k", "v", "g", "beta", "state_in", "seq_len"],
            &["y", "state_out"],
            GATED_DELTA_KERNEL_SOURCE,
            "",
            true,
        )
    });

    let outputs = kernel.apply_with_template(
        &[q, k, v, g, beta, state, &seq_i32],
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
        outputs.next().expect("gated delta y output"),
        outputs.next().expect("gated delta state output"),
    )
}

/// Qwen3Next/Qwen3.5 gated RMSNorm: `silu(gate.float32) * rms_norm(x).float32`.
pub fn rms_norm_gated(hidden_states: &MlxArray, gate: &MlxArray, weight: &MlxArray) -> MlxArray {
    let normed = rms_norm(hidden_states, Some(weight), 1e-6, None);
    let gate_f32 = astype(gate, MlxDtype::Float32, None);
    let normed_f32 = astype(&normed, MlxDtype::Float32, None);
    let gated = multiply(&mlx_sys::ops::silu(&gate_f32, None), &normed_f32, None);
    astype(&gated, hidden_states.dtype(), None)
}

fn scalar_i32(value: i32) -> MlxArray {
    MlxArray::from_raw_data(
        &value as *const i32 as *const u8,
        std::mem::size_of::<i32>(),
        &[1],
        MlxDtype::Int32,
    )
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

const GATED_DELTA_KERNEL_SOURCE: &str = r#"
    const int t_len = seq_len[0];
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + b_idx * t_len * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * t_len * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * t_len * Hv * Dv + hv_idx * Dv;
    y += b_idx * t_len * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // g, beta: [B, T, Hv]
    auto g_ = g + b_idx * t_len * Hv;
    auto beta_ = beta + b_idx * t_len * Hv;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    // s_base is invariant across both the t-loop and the inner i-loops.
    const int s_base = n_per_t * dk_idx;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[s_base + i]);
    }

    for (int t = 0; t < t_len; ++t) {
      // Load per-timestep scalars once from global memory.  g_[hv_idx] and
      // beta_[hv_idx] are read n_per_t and 1 times respectively in the
      // original loop body; caching them avoids repeated address computations
      // and makes the float promotion explicit (InT may be BF16).
      const float g_t    = static_cast<float>(g_[hv_idx]);
      const float beta_t = static_cast<float>(beta_[hv_idx]);
      const float v_t    = static_cast<float>(v_[dv_idx]);

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
      g_ += Hv;
      beta_ += Hv;
    }

    for (int i = 0; i < n_per_t; ++i) {
      o_state[s_base + i] = static_cast<StT>(state[i]);
    }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LinearAttentionConfig {
        LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 2,
            num_key_heads: 1,
            key_head_dim: 4,
            value_head_dim: 3,
            conv_kernel_dim: 4,
        }
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
    fn linear_attention_conv1d_returns_prompt_output_and_tail() {
        let cfg = cfg();
        let qkv = zeros(&[1, 5, cfg.conv_dim() as i32], MlxDtype::Float32, None);
        let weight = zeros(
            &[cfg.conv_dim() as i32, cfg.conv_kernel_dim as i32, 1_i32],
            MlxDtype::Float32,
            None,
        );

        let (conv_out, new_state) = linear_attention_conv1d(&cfg, &qkv, &weight, None, None);

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
        let q = zeros(&[1, 2, 1, 32], MlxDtype::Float32, None);
        let k = zeros(&[1, 2, 1, 32], MlxDtype::Float32, None);
        let v = zeros(&[1, 2, 1, 4], MlxDtype::Float32, None);
        let g = zeros(&[1, 2, 1], MlxDtype::Float32, None);
        let beta = zeros(&[1, 2, 1], MlxDtype::Float32, None);
        let state = zeros(&[1, 1, 4, 32], MlxDtype::Float32, None);

        let (y, new_state) = gated_delta_kernel(&q, &k, &v, &g, &beta, &state);

        assert_eq!(y.shape(), vec![1, 2, 1, 4]);
        assert_eq!(new_state.shape(), vec![1, 1, 4, 32]);
    }

    #[test]
    fn normalize_linear_attention_qk_preserves_reference_shapes() {
        let cfg = LinearAttentionConfig {
            full_attention_interval: 4,
            num_value_heads: 1,
            num_key_heads: 1,
            key_head_dim: 32,
            value_head_dim: 4,
            conv_kernel_dim: 4,
        };
        let q = zeros(&[1, 2, 1, 32], MlxDtype::Bfloat16, None);
        let k = zeros(&[1, 2, 1, 32], MlxDtype::Bfloat16, None);

        let (q, k) = normalize_linear_attention_qk(&cfg, &q, &k);

        assert_eq!(q.shape(), vec![1, 2, 1, 32]);
        assert_eq!(k.shape(), vec![1, 2, 1, 32]);
        assert_eq!(q.dtype(), MlxDtype::Bfloat16);
        assert_eq!(k.dtype(), MlxDtype::Bfloat16);
    }

    #[test]
    fn normalize_linear_attention_qk_scales_q_and_k_symmetrically() {
        let (q_scale, k_scale) = linear_attention_qk_scale(4);

        assert!((q_scale - 0.5).abs() < f32::EPSILON);
        assert!((k_scale - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn rms_norm_gated_preserves_hidden_shape_and_dtype() {
        let hidden = zeros(&[1, 5, 2, 3], MlxDtype::Bfloat16, None);
        let gate = zeros(&[1, 5, 2, 3], MlxDtype::Bfloat16, None);
        let weight = zeros(&[3], MlxDtype::Bfloat16, None);

        let out = rms_norm_gated(&hidden, &gate, &weight);

        assert_eq!(out.shape(), vec![1, 5, 2, 3]);
        assert_eq!(out.dtype(), MlxDtype::Bfloat16);
    }
}
