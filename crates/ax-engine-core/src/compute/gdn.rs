//! Shared Gated Delta Net (GDN) compute helpers for Qwen3.5-style recurrent layers.
//!
//! This module mirrors the architectural split used by upstream implementations:
//! the model path prepares projections and per-layer parameters, while the
//! backend owns the recurrent update primitive. Today the default backend path
//! is CPU scalar code; the Metal backend can override these entry points with
//! sequence-aware kernels later without changing model code again.

#[cfg(target_arch = "aarch64")]
use std::os::raw::c_int;

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn vvexpf(y: *mut f32, x: *const f32, n: *const c_int);
}

#[cfg(target_arch = "aarch64")]
const GDN_VFORCE_CHUNK: usize = 256;

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
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let mut neg = [0.0f32; GDN_VFORCE_CHUNK];
        let mut exp_neg = [0.0f32; GDN_VFORCE_CHUNK];

        for chunk in buf.chunks_mut(GDN_VFORCE_CHUNK) {
            let len = chunk.len();
            for i in 0..len {
                neg[i] = -chunk[i];
            }

            let n = len as c_int;
            unsafe {
                vvexpf(exp_neg.as_mut_ptr(), neg.as_ptr(), &n);
            }

            let simd_chunks = len / 4;
            unsafe {
                let ones = vdupq_n_f32(1.0);
                for i in 0..simd_chunks {
                    let offset = i * 4;
                    let expv = vld1q_f32(exp_neg.as_ptr().add(offset));
                    let denom = vaddq_f32(ones, expv);
                    let out = vdivq_f32(ones, denom);
                    vst1q_f32(chunk.as_mut_ptr().add(offset), out);
                }
            }

            for i in simd_chunks * 4..len {
                chunk[i] = 1.0 / (1.0 + exp_neg[i]);
            }
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for v in buf {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }
}

pub fn l2_norm_heads(buf: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        for head in buf.chunks_mut(head_dim).take(n_heads) {
            let len = head.len();
            let chunks = len / 4;
            let mut sum_sq = unsafe { vdupq_n_f32(0.0) };

            unsafe {
                for i in 0..chunks {
                    let v = vld1q_f32(head.as_ptr().add(i * 4));
                    sum_sq = vfmaq_f32(sum_sq, v, v);
                }
            }

            let mut total = unsafe { vaddvq_f32(sum_sq) };
            for &v in &head[chunks * 4..] {
                total += v * v;
            }

            let inv = 1.0 / (total + eps).sqrt();
            let inv_v = unsafe { vdupq_n_f32(inv) };
            unsafe {
                for i in 0..chunks {
                    let offset = i * 4;
                    let v = vld1q_f32(head.as_ptr().add(offset));
                    let out = vmulq_f32(v, inv_v);
                    vst1q_f32(head.as_mut_ptr().add(offset), out);
                }
            }
            for v in &mut head[chunks * 4..] {
                *v *= inv;
            }
        }
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for head in buf.chunks_mut(head_dim).take(n_heads) {
            let sum_sq = head.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (sum_sq + eps).sqrt();
            for v in head {
                *v *= inv;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn scale_in_place(buf: &mut [f32], scalar: f32) {
    use std::arch::aarch64::*;

    let len = buf.len();
    let chunks = len / 4;
    unsafe {
        let sv = vdupq_n_f32(scalar);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(buf.as_ptr().add(offset));
            let out = vmulq_f32(v, sv);
            vst1q_f32(buf.as_mut_ptr().add(offset), out);
        }
    }
    for v in &mut buf[chunks * 4..] {
        *v *= scalar;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn scale_in_place(buf: &mut [f32], scalar: f32) {
    for v in buf {
        *v *= scalar;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn scaled_add_in_place(dst: &mut [f32], scalar: f32, src: &[f32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let chunks = len / 4;
    unsafe {
        let sv = vdupq_n_f32(scalar);
        for i in 0..chunks {
            let offset = i * 4;
            let dv = vld1q_f32(dst.as_ptr().add(offset));
            let srcv = vld1q_f32(src.as_ptr().add(offset));
            let out = vfmaq_f32(dv, sv, srcv);
            vst1q_f32(dst.as_mut_ptr().add(offset), out);
        }
    }
    for (d, &s) in dst[chunks * 4..].iter_mut().zip(src[chunks * 4..].iter()) {
        *d += scalar * s;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn scaled_add_in_place(dst: &mut [f32], scalar: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += scalar * s;
    }
}

pub fn repeat_heads(
    input: &[f32],
    n_src_heads: usize,
    n_dst_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_dst_heads * head_dim];
    repeat_heads_into(&mut out, input, n_src_heads, n_dst_heads, head_dim);
    out
}

pub fn repeat_heads_into(
    dst: &mut [f32],
    input: &[f32],
    n_src_heads: usize,
    n_dst_heads: usize,
    head_dim: usize,
) {
    assert_eq!(
        input.len(),
        n_src_heads * head_dim,
        "repeat_heads input length mismatch"
    );
    assert_eq!(
        dst.len(),
        n_dst_heads * head_dim,
        "repeat_heads output length mismatch"
    );
    if n_src_heads == n_dst_heads {
        dst.copy_from_slice(input);
        return;
    }
    assert!(n_dst_heads.is_multiple_of(n_src_heads));
    let repeat = n_dst_heads / n_src_heads;
    for src in 0..n_src_heads {
        let src_slice = &input[src * head_dim..(src + 1) * head_dim];
        for rep in 0..repeat {
            let dst_head = src * repeat + rep;
            dst[dst_head * head_dim..(dst_head + 1) * head_dim].copy_from_slice(src_slice);
        }
    }
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
        out[c] = acc;
    }

    crate::compute::silu::silu(out);

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
    let mut delta = vec![0.0f32; head_dim];
    for h in 0..n_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let kh = &k[h * head_dim..(h + 1) * head_dim];
        let vh = &v[h * head_dim..(h + 1) * head_dim];
        let state_h = &mut state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];
        let decay = gate[h].exp();
        scale_in_place(state_h, decay);

        for value_idx in 0..head_dim {
            let mut kv_mem = 0.0f32;
            for key_idx in 0..head_dim {
                kv_mem += state_h[key_idx * head_dim + value_idx] * kh[key_idx];
            }
            delta[value_idx] = (vh[value_idx] - kv_mem) * beta[h];
        }

        for (row, &k_val) in state_h.chunks_exact_mut(head_dim).zip(kh.iter()) {
            scaled_add_in_place(row, k_val, &delta);
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
            let x = *alpha + bias;
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            *alpha = sp * a_val;
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

    let mut q_lin = vec![0.0f32; key_dim];
    let mut k_lin = vec![0.0f32; key_dim];
    let mut q_rep = vec![0.0f32; value_dim];
    let mut k_rep = vec![0.0f32; value_dim];

    for token_idx in 0..n_tokens {
        let conv_start = token_idx * cfg.conv_dim;
        let conv_end = conv_start + cfg.conv_dim;
        let conv_out = &conv_out_batch[conv_start..conv_end];
        q_lin.copy_from_slice(&conv_out[..key_dim]);
        k_lin.copy_from_slice(&conv_out[key_dim..2 * key_dim]);
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

        repeat_heads_into(
            &mut q_rep,
            &q_lin,
            cfg.group_count,
            cfg.time_step_rank,
            cfg.state_size,
        );
        repeat_heads_into(
            &mut k_rep,
            &k_lin,
            cfg.group_count,
            cfg.time_step_rank,
            cfg.state_size,
        );
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

    fn sigmoid_in_place_scalar(buf: &mut [f32]) {
        for v in buf {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    fn l2_norm_heads_scalar(buf: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
        for head in buf.chunks_mut(head_dim).take(n_heads) {
            let sum_sq = head.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (sum_sq + eps).sqrt();
            for v in head {
                *v *= inv;
            }
        }
    }

    fn scaled_add_in_place_scalar(dst: &mut [f32], scalar: f32, src: &[f32]) {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += scalar * s;
        }
    }

    #[test]
    fn test_repeat_heads_expands_groups() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let repeated = repeat_heads(&input, 2, 4, 2);
        assert_eq!(repeated, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_repeat_heads_into_matches_allocating_version() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut actual = vec![0.0f32; 8];
        repeat_heads_into(&mut actual, &input, 2, 4, 2);
        assert_eq!(actual, repeat_heads(&input, 2, 4, 2));
    }

    #[test]
    fn test_sigmoid_in_place_matches_scalar() {
        let mut actual = [-8.0, -2.5, -0.5, 0.0, 0.5, 2.5, 8.0];
        let mut expected = actual;
        sigmoid_in_place(&mut actual);
        sigmoid_in_place_scalar(&mut expected);
        for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((act - exp).abs() < 1e-6, "buf[{i}]: {act} != {exp}");
        }
    }

    #[test]
    fn test_l2_norm_heads_matches_scalar() {
        let mut actual = [
            1.0, -2.0, 3.0, -4.0, 0.5, 0.25, -0.75, 1.5, 2.0, -1.0, 0.25, -0.5,
        ];
        let mut expected = actual;
        l2_norm_heads(&mut actual, 3, 4, 1e-5);
        l2_norm_heads_scalar(&mut expected, 3, 4, 1e-5);
        for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((act - exp).abs() < 1e-5, "buf[{i}]: {act} != {exp}");
        }
    }

    #[test]
    fn test_scaled_add_in_place_matches_scalar() {
        let mut actual = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let mut expected = actual;
        let src = [-0.5, 0.25, 1.5, -2.0, 0.75, 3.0];
        let scalar = 1.75;
        scaled_add_in_place(&mut actual, scalar, &src);
        scaled_add_in_place_scalar(&mut expected, scalar, &src);
        for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((act - exp).abs() < 1e-6, "buf[{i}]: {act} != {exp}");
        }
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
