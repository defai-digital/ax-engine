//! Numerically stable softmax.
//!
//! softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//!
//! The max-subtraction prevents overflow in exp().

/// In-place softmax over a slice.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        softmax_neon(x);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        softmax_scalar(x);
    }
}

#[cfg(target_arch = "aarch64")]
fn softmax_neon(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let n = x.len();
    let chunks = n / 4;

    // Pass 1: vectorized max
    let mut max_v = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    unsafe {
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            max_v = vmaxq_f32(max_v, v);
        }
    }
    let mut max = unsafe { vmaxvq_f32(max_v) };
    for &v in &x[chunks * 4..] {
        max = max.max(v);
    }

    // Pass 2: exp(x - max) and accumulate sum
    // (exp must remain scalar for correctness; the subtraction and sum are vectorized)
    let max_v = unsafe { vdupq_n_f32(max) };
    let mut sum = 0.0f32;
    unsafe {
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            let shifted = vsubq_f32(v, max_v);
            // Must compute exp element-wise for correctness
            let mut exp_vals = [0.0f32; 4];
            vst1q_f32(exp_vals.as_mut_ptr(), shifted);
            exp_vals[0] = exp_vals[0].exp();
            exp_vals[1] = exp_vals[1].exp();
            exp_vals[2] = exp_vals[2].exp();
            exp_vals[3] = exp_vals[3].exp();
            let exp_v = vld1q_f32(exp_vals.as_ptr());
            vst1q_f32(x.as_mut_ptr().add(i * 4), exp_v);
            sum += vaddvq_f32(exp_v);
        }
    }
    for v in &mut x[chunks * 4..] {
        *v = (*v - max).exp();
        sum += *v;
    }

    // Guard against degenerate input
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / n as f32;
        x.fill(uniform);
        return;
    }

    // Pass 3: vectorized normalize
    let inv_sum_v = unsafe { vdupq_n_f32(1.0 / sum) };
    unsafe {
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            vst1q_f32(x.as_mut_ptr().add(i * 4), vmulq_f32(v, inv_sum_v));
        }
    }
    let inv_sum = 1.0 / sum;
    for v in &mut x[chunks * 4..] {
        *v *= inv_sum;
    }
}

#[allow(dead_code)]
fn softmax_scalar(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / x.len() as f32;
        x.fill(uniform);
        return;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Softmax writing to a separate output buffer.
pub fn softmax_out(x: &[f32], out: &mut [f32]) {
    assert!(out.len() >= x.len());
    if x.is_empty() {
        return;
    }

    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = (v - max).exp();
        sum += *o;
    }

    // Guard against sum==0 (all -inf inputs)
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / x.len() as f32;
        out[..x.len()].fill(uniform);
        return;
    }
    let inv_sum = 1.0 / sum;
    for o in out[..x.len()].iter_mut() {
        *o *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_uniform() {
        // Equal inputs → uniform distribution
        let mut x = [1.0, 1.0, 1.0, 1.0];
        softmax(&mut x);
        for &v in &x {
            assert!((v - 0.25).abs() < 1e-6, "expected 0.25, got {v}");
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn test_softmax_ordering() {
        // Larger input → larger probability
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_softmax_large_values() {
        // Should not overflow — max subtraction handles this
        let mut x = [1000.0, 1001.0, 1002.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        for &v in &x {
            assert!(v.is_finite(), "got non-finite: {v}");
        }
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut x = [-1.0, -2.0, -3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[0] > x[1]);
        assert!(x[1] > x[2]);
    }

    #[test]
    fn test_softmax_single() {
        let mut x = [42.0];
        softmax(&mut x);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_empty() {
        let mut x: [f32; 0] = [];
        softmax(&mut x); // should not panic
    }

    #[test]
    fn test_softmax_known_values() {
        // softmax([0, 1]) = [1/(1+e), e/(1+e)] ≈ [0.2689, 0.7311]
        let mut x = [0.0, 1.0];
        softmax(&mut x);
        let e = std::f32::consts::E;
        let expected_0 = 1.0 / (1.0 + e);
        let expected_1 = e / (1.0 + e);
        assert!((x[0] - expected_0).abs() < 1e-5);
        assert!((x[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_all_neg_inf() {
        // After aggressive top-k, all logits may be -inf.
        // softmax should not produce NaN — return uniform.
        let mut x = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        softmax(&mut x);
        for &v in &x {
            assert!(v.is_finite(), "got non-finite: {v}");
        }
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn test_softmax_out_all_neg_inf() {
        let x = [f32::NEG_INFINITY; 4];
        let mut out = [0.0f32; 4];
        softmax_out(&x, &mut out);
        for &v in &out {
            assert!(v.is_finite(), "got non-finite: {v}");
        }
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn test_softmax_out() {
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        softmax_out(&x, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
