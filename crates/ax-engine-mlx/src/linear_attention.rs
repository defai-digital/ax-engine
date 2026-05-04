use mlx_sys::{
    MlxArray, MlxDtype, add, astype, concatenate, conv1d, exp, log1p, multiply, negative, reshape,
    slice, slice_last_dim, where_cond, zeros,
};

use crate::model::LinearAttentionConfig;

/// Split Qwen3.5 gated-delta conv output into shaped q/k/v tensors.
pub struct LinearAttentionQkv {
    pub q: MlxArray,
    pub k: MlxArray,
    pub v: MlxArray,
}

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
}
