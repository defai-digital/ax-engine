//! RMS normalization as used in LLaMA.
//!
//! RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
//!
//! Applied element-wise: each element is scaled by the reciprocal of
//! the RMS of the input vector, then multiplied by the learned weight.

/// In-place RMS normalization.
///
/// `x`: input/output vector (length n)
/// `weight`: learned per-element scale (length n)
/// `eps`: small constant for numerical stability (typically 1e-5 or 1e-6)
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    assert_eq!(
        n,
        weight.len(),
        "rms_norm: x.len()={} != weight.len()={}",
        n,
        weight.len()
    );
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        rms_norm_neon(x, weight, eps);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rms_norm_scalar(x, weight, eps);
    }
}

#[cfg(target_arch = "aarch64")]
fn rms_norm_neon(x: &mut [f32], weight: &[f32], eps: f32) {
    use std::arch::aarch64::*;

    let n = x.len();

    // Vectorized sum of squares: process 4 floats per iteration
    let mut sum_sq = unsafe { vdupq_n_f32(0.0) };
    let chunks = n / 4;
    unsafe {
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            sum_sq = vfmaq_f32(sum_sq, v, v);
        }
    }

    // Horizontal sum of the 4-lane accumulator + scalar remainder
    let mut total = unsafe { vaddvq_f32(sum_sq) };
    for &v in &x[chunks * 4..] {
        total += v * v;
    }

    let inv_rms = 1.0 / (total / n as f32 + eps).sqrt();

    // Vectorized normalize + weight: x[i] = x[i] * inv_rms * weight[i]
    let inv_rms_v = unsafe { vdupq_n_f32(inv_rms) };
    unsafe {
        for i in 0..chunks {
            let xv = vld1q_f32(x.as_ptr().add(i * 4));
            let wv = vld1q_f32(weight.as_ptr().add(i * 4));
            let normed = vmulq_f32(vmulq_f32(xv, inv_rms_v), wv);
            vst1q_f32(x.as_mut_ptr().add(i * 4), normed);
        }
    }
    for (xi, &wi) in x[chunks * 4..].iter_mut().zip(weight[chunks * 4..].iter()) {
        *xi = *xi * inv_rms * wi;
    }
}

#[allow(dead_code)]
fn rms_norm_scalar(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for (xi, &wi) in x.iter_mut().zip(weight.iter()) {
        *xi = *xi * inv_rms * wi;
    }
}

/// RMS normalization writing to a separate output buffer.
///
/// `x`: input vector (length n, not modified)
/// `weight`: learned per-element scale (length n)
/// `out`: output buffer (length n)
/// `eps`: numerical stability constant
pub fn rms_norm_out(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let n = x.len();
    assert_eq!(n, weight.len());
    assert!(out.len() >= n);
    if x.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let chunks = n / 4;

        // Vectorized sum of squares
        let mut sum_sq = unsafe { vdupq_n_f32(0.0) };
        unsafe {
            for i in 0..chunks {
                let v = vld1q_f32(x.as_ptr().add(i * 4));
                sum_sq = vfmaq_f32(sum_sq, v, v);
            }
        }
        let mut total = unsafe { vaddvq_f32(sum_sq) };
        for &v in &x[chunks * 4..] {
            total += v * v;
        }

        let inv_rms = 1.0 / (total / n as f32 + eps).sqrt();
        let inv_rms_v = unsafe { vdupq_n_f32(inv_rms) };
        unsafe {
            for i in 0..chunks {
                let xv = vld1q_f32(x.as_ptr().add(i * 4));
                let wv = vld1q_f32(weight.as_ptr().add(i * 4));
                let normed = vmulq_f32(vmulq_f32(xv, inv_rms_v), wv);
                vst1q_f32(out.as_mut_ptr().add(i * 4), normed);
            }
        }
        let tail = chunks * 4;
        for ((&xi, &wi), oi) in x[tail..]
            .iter()
            .zip(weight[tail..].iter())
            .zip(out[tail..].iter_mut())
        {
            *oi = xi * inv_rms * wi;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum_sq = 0.0f32;
        for &v in x.iter() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] * inv_rms * weight[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_ones() {
        // x = [1,1,1,1], weight = [1,1,1,1]
        // mean(x^2) = 1, rms = sqrt(1 + eps) ≈ 1
        // result ≈ [1,1,1,1] / 1 = [1,1,1,1]
        let mut x = [1.0, 1.0, 1.0, 1.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        rms_norm(&mut x, &w, 1e-5);
        for &v in &x {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_scaling() {
        // x = [2,2,2,2], weight = [1,1,1,1]
        // mean(x^2) = 4, rms = sqrt(4 + eps) ≈ 2
        // result ≈ [2/2, 2/2, 2/2, 2/2] = [1,1,1,1]
        let mut x = [2.0, 2.0, 2.0, 2.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        rms_norm(&mut x, &w, 1e-5);
        for &v in &x {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rms_norm_weight() {
        // x = [1,1], weight = [2,3]
        // rms = sqrt(1 + eps) ≈ 1
        // result ≈ [1*2, 1*3] = [2, 3]
        let mut x = [1.0, 1.0];
        let w = [2.0, 3.0];
        rms_norm(&mut x, &w, 1e-5);
        assert!((x[0] - 2.0).abs() < 0.01);
        assert!((x[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_rms_norm_mixed() {
        // x = [3,4], weight = [1,1]
        // mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5) ≈ 3.5355
        // result = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        let mut x = [3.0, 4.0];
        let w = [1.0, 1.0];
        rms_norm(&mut x, &w, 0.0);
        let rms = (12.5f32).sqrt();
        assert!((x[0] - 3.0 / rms).abs() < 1e-4);
        assert!((x[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_eps_protects_zero() {
        // x = [0,0], should not NaN thanks to eps
        let mut x = [0.0, 0.0];
        let w = [1.0, 1.0];
        rms_norm(&mut x, &w, 1e-5);
        for &v in &x {
            assert!(v.is_finite(), "expected finite, got {v}");
            assert!(v.abs() < 0.01);
        }
    }

    #[test]
    fn test_rms_norm_out() {
        let x = [3.0, 4.0];
        let w = [1.0, 1.0];
        let mut out = [0.0f32; 2];
        rms_norm_out(&x, &w, &mut out, 0.0);
        let rms = (12.5f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
        assert!((out[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_empty_is_noop() {
        let mut x = [];
        let w = [];
        rms_norm(&mut x, &w, 1e-5);
        assert!(x.is_empty());
    }

    #[test]
    fn test_rms_norm_out_empty_is_noop() {
        let x = [];
        let w = [];
        let mut out = [];
        rms_norm_out(&x, &w, &mut out, 1e-5);
        assert!(out.is_empty());
    }
}
