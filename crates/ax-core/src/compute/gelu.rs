//! GELU (Gaussian Error Linear Unit) activation.
//!
//! GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//!
//! Used in Gemma3's feed-forward network instead of SiLU.

#[cfg(target_arch = "aarch64")]
use std::os::raw::c_int;

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn vvexpf(y: *mut f32, x: *const f32, n: *const c_int);
}

/// Constant: sqrt(2/pi) ≈ 0.7978845608
const SQRT_2_PI: f32 = 0.797_884_6;
const GELU_CUBIC_COEFF: f32 = 0.044715;

#[cfg(target_arch = "aarch64")]
const GELU_VFORCE_CHUNK: usize = 256;

/// In-place GELU activation (tanh approximation).
pub fn gelu(x: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        gelu_vforce(x);
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        gelu_scalar(x);
    }
}

/// GELU followed by element-wise multiply: x = GELU(x) * y
///
/// This is the GeGLU pattern used in Gemma3:
///   gate = GELU(W_gate * input), up = W_up * input,
///   then output = gate * up.
pub fn gelu_elementwise_mul(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "aarch64")]
    {
        gelu_elementwise_mul_vforce(x, y);
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        gelu_elementwise_mul_scalar(x, y);
    }
}

#[allow(dead_code)]
fn gelu_scalar(x: &mut [f32]) {
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        let inner = SQRT_2_PI * (*v + GELU_CUBIC_COEFF * x3);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}

#[allow(dead_code)]
fn gelu_elementwise_mul_scalar(x: &mut [f32], y: &[f32]) {
    for (xi, &yi) in x.iter_mut().zip(y.iter()) {
        let x3 = *xi * *xi * *xi;
        let inner = SQRT_2_PI * (*xi + GELU_CUBIC_COEFF * x3);
        *xi = 0.5 * *xi * (1.0 + inner.tanh()) * yi;
    }
}

#[cfg(target_arch = "aarch64")]
fn gelu_vforce(x: &mut [f32]) {
    use std::arch::aarch64::*;

    // GELU approximation:
    //   0.5 * x * (1 + tanh(inner))
    // where inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    // Rewrite:
    //   0.5 * (1 + tanh(inner)) = 1 / (1 + exp(-2 * inner))
    // so GELU(x) = x / (1 + exp(-2 * inner))
    let mut neg_two_inner = [0.0f32; GELU_VFORCE_CHUNK];
    let mut exp_neg_two_inner = [0.0f32; GELU_VFORCE_CHUNK];

    for chunk in x.chunks_mut(GELU_VFORCE_CHUNK) {
        let len = chunk.len();
        let simd_chunks = len / 4;

        unsafe {
            let cubic_coeff = vdupq_n_f32(GELU_CUBIC_COEFF);
            let neg_two_sqrt_2_pi = vdupq_n_f32(-2.0 * SQRT_2_PI);

            for i in 0..simd_chunks {
                let offset = i * 4;
                let xv = vld1q_f32(chunk.as_ptr().add(offset));
                let x2 = vmulq_f32(xv, xv);
                let x3 = vmulq_f32(x2, xv);
                let inner_poly = vaddq_f32(xv, vmulq_f32(cubic_coeff, x3));
                let neg_scaled = vmulq_f32(neg_two_sqrt_2_pi, inner_poly);
                vst1q_f32(neg_two_inner.as_mut_ptr().add(offset), neg_scaled);
            }
        }

        for i in simd_chunks * 4..len {
            let xv = chunk[i];
            let x3 = xv * xv * xv;
            let inner = SQRT_2_PI * (xv + GELU_CUBIC_COEFF * x3);
            neg_two_inner[i] = -2.0 * inner;
        }

        let n = len as c_int;
        unsafe {
            vvexpf(exp_neg_two_inner.as_mut_ptr(), neg_two_inner.as_ptr(), &n);
        }

        unsafe {
            let ones = vdupq_n_f32(1.0);
            for i in 0..simd_chunks {
                let offset = i * 4;
                let xv = vld1q_f32(chunk.as_ptr().add(offset));
                let expv = vld1q_f32(exp_neg_two_inner.as_ptr().add(offset));
                let denom = vaddq_f32(ones, expv);
                let out = vdivq_f32(xv, denom);
                vst1q_f32(chunk.as_mut_ptr().add(offset), out);
            }
        }

        for i in simd_chunks * 4..len {
            chunk[i] = chunk[i] / (1.0 + exp_neg_two_inner[i]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn gelu_elementwise_mul_vforce(x: &mut [f32], y: &[f32]) {
    use std::arch::aarch64::*;

    let mut neg_two_inner = [0.0f32; GELU_VFORCE_CHUNK];
    let mut exp_neg_two_inner = [0.0f32; GELU_VFORCE_CHUNK];

    for (x_chunk, y_chunk) in x
        .chunks_mut(GELU_VFORCE_CHUNK)
        .zip(y.chunks(GELU_VFORCE_CHUNK))
    {
        let len = x_chunk.len();
        let simd_chunks = len / 4;

        unsafe {
            let cubic_coeff = vdupq_n_f32(GELU_CUBIC_COEFF);
            let neg_two_sqrt_2_pi = vdupq_n_f32(-2.0 * SQRT_2_PI);

            for i in 0..simd_chunks {
                let offset = i * 4;
                let xv = vld1q_f32(x_chunk.as_ptr().add(offset));
                let x2 = vmulq_f32(xv, xv);
                let x3 = vmulq_f32(x2, xv);
                let inner_poly = vaddq_f32(xv, vmulq_f32(cubic_coeff, x3));
                let neg_scaled = vmulq_f32(neg_two_sqrt_2_pi, inner_poly);
                vst1q_f32(neg_two_inner.as_mut_ptr().add(offset), neg_scaled);
            }
        }

        for i in simd_chunks * 4..len {
            let xv = x_chunk[i];
            let x3 = xv * xv * xv;
            let inner = SQRT_2_PI * (xv + GELU_CUBIC_COEFF * x3);
            neg_two_inner[i] = -2.0 * inner;
        }

        let n = len as c_int;
        unsafe {
            vvexpf(exp_neg_two_inner.as_mut_ptr(), neg_two_inner.as_ptr(), &n);
        }

        unsafe {
            let ones = vdupq_n_f32(1.0);
            for i in 0..simd_chunks {
                let offset = i * 4;
                let xv = vld1q_f32(x_chunk.as_ptr().add(offset));
                let yv = vld1q_f32(y_chunk.as_ptr().add(offset));
                let expv = vld1q_f32(exp_neg_two_inner.as_ptr().add(offset));
                let denom = vaddq_f32(ones, expv);
                let activated = vdivq_f32(xv, denom);
                let out = vmulq_f32(activated, yv);
                vst1q_f32(x_chunk.as_mut_ptr().add(offset), out);
            }
        }

        for i in simd_chunks * 4..len {
            x_chunk[i] = x_chunk[i] / (1.0 + exp_neg_two_inner[i]) * y_chunk[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_zero() {
        let mut x = [0.0];
        gelu(&mut x);
        assert!(x[0].abs() < 1e-6, "GELU(0) should be 0, got {}", x[0]);
    }

    #[test]
    fn test_gelu_large_positive() {
        // GELU(x) ≈ x for large positive x
        let mut x = [10.0];
        gelu(&mut x);
        assert!(
            (x[0] - 10.0).abs() < 0.01,
            "GELU(10) should ≈ 10, got {}",
            x[0]
        );
    }

    #[test]
    fn test_gelu_large_negative() {
        // GELU(x) ≈ 0 for large negative x
        let mut x = [-10.0];
        gelu(&mut x);
        assert!(x[0].abs() < 0.01, "GELU(-10) should ≈ 0, got {}", x[0]);
    }

    #[test]
    fn test_gelu_one() {
        // GELU(1) ≈ 0.8412
        let mut x = [1.0];
        gelu(&mut x);
        assert!(
            (x[0] - 0.8412).abs() < 0.01,
            "GELU(1) should ≈ 0.8412, got {}",
            x[0]
        );
    }

    #[test]
    fn test_gelu_negative_one() {
        // GELU(-1) ≈ -0.1588
        let mut x = [-1.0];
        gelu(&mut x);
        assert!(
            (x[0] - (-0.1588)).abs() < 0.01,
            "GELU(-1) should ≈ -0.1588, got {}",
            x[0]
        );
    }

    #[test]
    fn test_gelu_batch() {
        let mut x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let original = x;
        gelu(&mut x);
        for (i, (&out, &inp)) in x.iter().zip(original.iter()).enumerate() {
            let x3 = inp * inp * inp;
            let inner = super::SQRT_2_PI * (inp + 0.044715 * x3);
            let expected = 0.5 * inp * (1.0 + inner.tanh());
            assert!((out - expected).abs() < 1e-5, "x[{i}]: {out} != {expected}");
        }
    }

    #[test]
    fn test_gelu_elementwise_mul() {
        let mut gate = [1.0, -1.0, 2.0];
        let up = [2.0, 3.0, 0.5];
        let original_gate = gate;

        gelu_elementwise_mul(&mut gate, &up);

        for (i, ((&out, &g), &u)) in gate
            .iter()
            .zip(original_gate.iter())
            .zip(up.iter())
            .enumerate()
        {
            let x3 = g * g * g;
            let inner = super::SQRT_2_PI * (g + 0.044715 * x3);
            let expected = 0.5 * g * (1.0 + inner.tanh()) * u;
            assert!((out - expected).abs() < 1e-5, "x[{i}]: {out} != {expected}");
        }
    }

    #[test]
    fn test_gelu_monotonic_positive() {
        // GELU should be monotonically increasing for x > ~-0.17
        let mut x = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
        gelu(&mut x);
        for i in 1..x.len() {
            assert!(
                x[i] > x[i - 1],
                "GELU should be increasing for positive x: x[{}]={} <= x[{}]={}",
                i,
                x[i],
                i - 1,
                x[i - 1]
            );
        }
    }
}
