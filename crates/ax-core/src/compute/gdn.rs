//! Shared Gated Delta Net (GDN) compute helpers for Qwen3.5-style recurrent layers.
//!
//! This module mirrors the architectural split used by upstream implementations:
//! the model path prepares projections and per-layer parameters, while the
//! backend owns the recurrent update primitive. Today the default backend path
//! is CPU scalar code; the Metal backend can override these entry points with
//! sequence-aware kernels later without changing model code again.

/// Shape/configuration for one Qwen3.5 recurrent block.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35RecurrentConfig {
    pub conv_cache_len: usize,
    pub conv_dim: usize,
    pub group_count: usize,
    pub state_size: usize,
    pub time_step_rank: usize,
    pub rms_norm_eps: f32,
}

impl Qwen35RecurrentConfig {
    pub fn key_dim(self) -> usize {
        self.group_count * self.state_size
    }

    pub fn value_dim(self) -> usize {
        self.time_step_rank * self.state_size
    }
}

pub fn sigmoid_in_place(buf: &mut [f32]) {
    for v in buf {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

pub fn l2_norm_heads(buf: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
    for head in buf.chunks_mut(head_dim).take(n_heads) {
        let sum_sq = head.iter().map(|v| v * v).sum::<f32>();
        let inv = 1.0 / (sum_sq + eps).sqrt();
        for v in head {
            *v *= inv;
        }
    }
}

pub fn repeat_heads(
    input: &[f32],
    n_src_heads: usize,
    n_dst_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    if n_src_heads == n_dst_heads {
        return input.to_vec();
    }
    assert!(n_dst_heads.is_multiple_of(n_src_heads));
    let repeat = n_dst_heads / n_src_heads;
    let mut out = vec![0.0f32; n_dst_heads * head_dim];
    for src in 0..n_src_heads {
        let src_slice = &input[src * head_dim..(src + 1) * head_dim];
        for rep in 0..repeat {
            let dst = src * repeat + rep;
            out[dst * head_dim..(dst + 1) * head_dim].copy_from_slice(src_slice);
        }
    }
    out
}

pub fn depthwise_conv1d_step(
    conv_state: &mut [f32],
    input: &[f32],
    kernel: &[f32],
    conv_cache_len: usize,
    conv_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(conv_state.len(), conv_cache_len * conv_dim);
    assert_eq!(input.len(), conv_dim);
    assert_eq!(kernel.len(), (conv_cache_len + 1) * conv_dim);
    assert_eq!(out.len(), conv_dim);

    for c in 0..conv_dim {
        let mut acc = input[c] * kernel[conv_cache_len * conv_dim + c];
        for t in 0..conv_cache_len {
            acc += conv_state[t * conv_dim + c] * kernel[t * conv_dim + c];
        }
        out[c] = acc / (1.0 + (-acc).exp());
    }

    if conv_cache_len > 0 {
        if conv_cache_len > 1 {
            conv_state.copy_within(conv_dim.., 0);
        }
        let start = (conv_cache_len - 1) * conv_dim;
        conv_state[start..start + conv_dim].copy_from_slice(input);
    }
}

pub fn depthwise_conv1d_sequence(
    input_batch: &[f32],
    kernel: &[f32],
    conv_state: &mut [f32],
    output_batch: &mut [f32],
    n_tokens: usize,
    conv_cache_len: usize,
    conv_dim: usize,
) {
    assert_eq!(input_batch.len(), n_tokens * conv_dim);
    assert_eq!(output_batch.len(), n_tokens * conv_dim);
    for token_idx in 0..n_tokens {
        let start = token_idx * conv_dim;
        let end = start + conv_dim;
        depthwise_conv1d_step(
            conv_state,
            &input_batch[start..end],
            kernel,
            conv_cache_len,
            conv_dim,
            &mut output_batch[start..end],
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(k.len(), n_heads * head_dim);
    assert_eq!(v.len(), n_heads * head_dim);
    assert_eq!(gate.len(), n_heads);
    assert_eq!(beta.len(), n_heads);
    assert_eq!(state.len(), n_heads * head_dim * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let kh = &k[h * head_dim..(h + 1) * head_dim];
        let vh = &v[h * head_dim..(h + 1) * head_dim];
        let state_h = &mut state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];
        let decay = gate[h].exp();
        for s in state_h.iter_mut() {
            *s *= decay;
        }

        let mut delta = vec![0.0f32; head_dim];
        for value_idx in 0..head_dim {
            let mut kv_mem = 0.0f32;
            for key_idx in 0..head_dim {
                kv_mem += state_h[key_idx * head_dim + value_idx] * kh[key_idx];
            }
            delta[value_idx] = (vh[value_idx] - kv_mem) * beta[h];
        }

        for key_idx in 0..head_dim {
            let k_val = kh[key_idx];
            for value_idx in 0..head_dim {
                state_h[key_idx * head_dim + value_idx] += k_val * delta[value_idx];
            }
        }

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for value_idx in 0..head_dim {
            let mut sum = 0.0f32;
            for key_idx in 0..head_dim {
                sum += state_h[key_idx * head_dim + value_idx] * qh[key_idx];
            }
            out_h[value_idx] = sum * scale;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_sequence(
    q_batch: &[f32],
    k_batch: &[f32],
    v_batch: &[f32],
    gate_batch: &[f32],
    beta_batch: &[f32],
    state: &mut [f32],
    output_batch: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    let value_dim = n_heads * head_dim;
    assert_eq!(q_batch.len(), n_tokens * value_dim);
    assert_eq!(k_batch.len(), n_tokens * value_dim);
    assert_eq!(v_batch.len(), n_tokens * value_dim);
    assert_eq!(gate_batch.len(), n_tokens * n_heads);
    assert_eq!(beta_batch.len(), n_tokens * n_heads);
    assert_eq!(output_batch.len(), n_tokens * value_dim);

    for token_idx in 0..n_tokens {
        let q_start = token_idx * value_dim;
        let q_end = q_start + value_dim;
        let gb_start = token_idx * n_heads;
        let gb_end = gb_start + n_heads;
        gated_delta_rule_step(
            &q_batch[q_start..q_end],
            &k_batch[q_start..q_end],
            &v_batch[q_start..q_end],
            &gate_batch[gb_start..gb_end],
            &beta_batch[gb_start..gb_end],
            state,
            n_heads,
            head_dim,
            &mut output_batch[q_start..q_end],
        );
    }
}

pub fn prepare_alpha_beta(
    alpha_batch: &mut [f32],
    beta_batch: &mut [f32],
    dt_bias: &[f32],
    a: &[f32],
) {
    assert_eq!(alpha_batch.len(), beta_batch.len());
    assert_eq!(dt_bias.len(), a.len());
    assert!(alpha_batch.len().is_multiple_of(dt_bias.len()));
    let n_tokens = alpha_batch.len() / dt_bias.len();
    for token_idx in 0..n_tokens {
        let start = token_idx * dt_bias.len();
        let end = start + dt_bias.len();
        sigmoid_in_place(&mut beta_batch[start..end]);
        for ((alpha, &bias), &a_val) in alpha_batch[start..end]
            .iter_mut()
            .zip(dt_bias.iter())
            .zip(a.iter())
        {
            *alpha = (1.0 + (*alpha + bias).exp()).ln() * a_val;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qwen35_recurrent_sequence(
    qkv_batch: &[f32],
    beta_batch: &mut [f32],
    alpha_batch: &mut [f32],
    dt_bias: &[f32],
    a: &[f32],
    conv_kernel: &[f32],
    conv_state: &mut [f32],
    recurrent_state: &mut [f32],
    output_batch: &mut [f32],
    n_tokens: usize,
    cfg: Qwen35RecurrentConfig,
) {
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let mut conv_out_batch = vec![0.0f32; n_tokens * cfg.conv_dim];
    let mut q_batch = vec![0.0f32; n_tokens * value_dim];
    let mut k_batch = vec![0.0f32; n_tokens * value_dim];
    let mut v_batch = vec![0.0f32; n_tokens * value_dim];

    prepare_alpha_beta(alpha_batch, beta_batch, dt_bias, a);
    depthwise_conv1d_sequence(
        qkv_batch,
        conv_kernel,
        conv_state,
        &mut conv_out_batch,
        n_tokens,
        cfg.conv_cache_len,
        cfg.conv_dim,
    );

    for token_idx in 0..n_tokens {
        let conv_start = token_idx * cfg.conv_dim;
        let conv_end = conv_start + cfg.conv_dim;
        let conv_out = &conv_out_batch[conv_start..conv_end];
        let mut q_lin = conv_out[..key_dim].to_vec();
        let mut k_lin = conv_out[key_dim..2 * key_dim].to_vec();
        let v_lin = &conv_out[2 * key_dim..2 * key_dim + value_dim];

        l2_norm_heads(
            &mut q_lin,
            cfg.group_count,
            cfg.state_size,
            cfg.rms_norm_eps,
        );
        l2_norm_heads(
            &mut k_lin,
            cfg.group_count,
            cfg.state_size,
            cfg.rms_norm_eps,
        );

        let q_rep = repeat_heads(&q_lin, cfg.group_count, cfg.time_step_rank, cfg.state_size);
        let k_rep = repeat_heads(&k_lin, cfg.group_count, cfg.time_step_rank, cfg.state_size);
        let out_start = token_idx * value_dim;
        let out_end = out_start + value_dim;
        q_batch[out_start..out_end].copy_from_slice(&q_rep);
        k_batch[out_start..out_end].copy_from_slice(&k_rep);
        v_batch[out_start..out_end].copy_from_slice(v_lin);
    }

    gated_delta_rule_sequence(
        &q_batch,
        &k_batch,
        &v_batch,
        alpha_batch,
        beta_batch,
        recurrent_state,
        output_batch,
        n_tokens,
        cfg.time_step_rank,
        cfg.state_size,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_heads_expands_groups() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let repeated = repeat_heads(&input, 2, 4, 2);
        assert_eq!(repeated, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_qwen35_recurrent_sequence_shapes() {
        let cfg = Qwen35RecurrentConfig {
            conv_cache_len: 2,
            conv_dim: 6,
            group_count: 1,
            state_size: 2,
            time_step_rank: 1,
            rms_norm_eps: 1e-5,
        };
        let n_tokens = 2;
        let mut beta = vec![0.1, 0.2];
        let mut alpha = vec![0.3, 0.4];
        let qkv = vec![
            0.5, 0.1, -0.2, 0.7, 0.3, 0.4, //
            -0.1, 0.2, 0.6, 0.8, -0.4, 0.9,
        ];
        let dt_bias = vec![0.05];
        let a = vec![0.7];
        let kernel = vec![
            0.1, 0.2, 0.1, 0.2, 0.1, 0.2, //
            0.2, 0.1, 0.2, 0.1, 0.2, 0.1, //
            0.3, 0.4, 0.3, 0.4, 0.3, 0.4,
        ];
        let mut conv_state = vec![0.0; cfg.conv_cache_len * cfg.conv_dim];
        let mut recurrent_state = vec![0.0; cfg.time_step_rank * cfg.state_size * cfg.state_size];
        let mut out = vec![0.0; n_tokens * cfg.value_dim()];

        qwen35_recurrent_sequence(
            &qkv,
            &mut beta,
            &mut alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut conv_state,
            &mut recurrent_state,
            &mut out,
            n_tokens,
            cfg,
        );

        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(conv_state.iter().all(|v| v.is_finite()));
        assert!(recurrent_state.iter().all(|v| v.is_finite()));
    }
}
