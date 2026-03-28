//! SiLU (Sigmoid Linear Unit) activation, also known as Swish.
//!
//! SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//!
//! Used in LLaMA's feed-forward network (SwiGLU variant):
//!   FFN(x) = SiLU(W_gate * x) * (W_up * x)

#[cfg(target_arch = "aarch64")]
use std::os::raw::c_int;

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn vvexpf(y: *mut f32, x: *const f32, n: *const c_int);
}

#[cfg(target_arch = "aarch64")]
const VFORCE_CHUNK: usize = 256;

/// In-place SiLU activation.
pub fn silu(x: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        silu_vforce(x);
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        silu_scalar(x);
    }
}

/// SiLU followed by element-wise multiply: x = SiLU(x) * y
///
/// This is the SwiGLU pattern: gate = SiLU(W_gate * input), up = W_up * input,
/// then output = gate * up.
pub fn silu_elementwise_mul(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "aarch64")]
    {
        silu_elementwise_mul_vforce(x, y);
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        silu_elementwise_mul_scalar(x, y);
    }
}

#[allow(dead_code)]
fn silu_scalar(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

#[allow(dead_code)]
fn silu_elementwise_mul_scalar(x: &mut [f32], y: &[f32]) {
    for (xi, &yi) in x.iter_mut().zip(y.iter()) {
        *xi = *xi / (1.0 + (-*xi).exp()) * yi;
    }
}

#[cfg(target_arch = "aarch64")]
fn silu_vforce(x: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut neg = [0.0f32; VFORCE_CHUNK];
    let mut exp_neg = [0.0f32; VFORCE_CHUNK];

    for chunk in x.chunks_mut(VFORCE_CHUNK) {
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
                let xv = vld1q_f32(chunk.as_ptr().add(offset));
                let expv = vld1q_f32(exp_neg.as_ptr().add(offset));
                let denom = vaddq_f32(ones, expv);
                let out = vdivq_f32(xv, denom);
                vst1q_f32(chunk.as_mut_ptr().add(offset), out);
            }
        }

        for i in simd_chunks * 4..len {
            chunk[i] = chunk[i] / (1.0 + exp_neg[i]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn silu_elementwise_mul_vforce(x: &mut [f32], y: &[f32]) {
    use std::arch::aarch64::*;

    let mut neg = [0.0f32; VFORCE_CHUNK];
    let mut exp_neg = [0.0f32; VFORCE_CHUNK];

    for (x_chunk, y_chunk) in x.chunks_mut(VFORCE_CHUNK).zip(y.chunks(VFORCE_CHUNK)) {
        let len = x_chunk.len();
        for i in 0..len {
            neg[i] = -x_chunk[i];
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
                let xv = vld1q_f32(x_chunk.as_ptr().add(offset));
                let yv = vld1q_f32(y_chunk.as_ptr().add(offset));
                let expv = vld1q_f32(exp_neg.as_ptr().add(offset));
                let denom = vaddq_f32(ones, expv);
                let activated = vdivq_f32(xv, denom);
                let out = vmulq_f32(activated, yv);
                vst1q_f32(x_chunk.as_mut_ptr().add(offset), out);
            }
        }

        for i in simd_chunks * 4..len {
            x_chunk[i] = x_chunk[i] / (1.0 + exp_neg[i]) * y_chunk[i];
        }
    }
}

/// In-place element-wise multiply: a *= b (NEON-vectorized on aarch64).
pub fn elementwise_mul(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = a.len();
        let chunks = n / 4;
        unsafe {
            for i in 0..chunks {
                let av = vld1q_f32(a.as_ptr().add(i * 4));
                let bv = vld1q_f32(b.as_ptr().add(i * 4));
                vst1q_f32(a.as_mut_ptr().add(i * 4), vmulq_f32(av, bv));
            }
        }
        for (ai, &bi) in a[chunks * 4..].iter_mut().zip(b[chunks * 4..].iter()) {
            *ai *= bi;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (ai, &bi) in a.iter_mut().zip(b.iter()) {
            *ai *= bi;
        }
    }
}

/// In-place element-wise add: a += b (NEON-vectorized on aarch64).
pub fn elementwise_add(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = a.len();
        let chunks = n / 4;
        unsafe {
            for i in 0..chunks {
                let av = vld1q_f32(a.as_ptr().add(i * 4));
                let bv = vld1q_f32(b.as_ptr().add(i * 4));
                vst1q_f32(a.as_mut_ptr().add(i * 4), vaddq_f32(av, bv));
            }
        }
        for (ai, &bi) in a[chunks * 4..].iter_mut().zip(b[chunks * 4..].iter()) {
            *ai += bi;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for (ai, &bi) in a.iter_mut().zip(b.iter()) {
            *ai += bi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_zero() {
        // SiLU(0) = 0 / (1 + 1) = 0
        let mut x = [0.0];
        silu(&mut x);
        assert!(x[0].abs() < 1e-6);
    }

    #[test]
    fn test_silu_positive() {
        // SiLU(x) → x for large positive x (sigmoid → 1)
        let mut x = [10.0];
        silu(&mut x);
        assert!((x[0] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_silu_negative() {
        // SiLU(x) → 0 for large negative x (sigmoid → 0)
        let mut x = [-10.0];
        silu(&mut x);
        assert!(x[0].abs() < 0.001);
    }

    #[test]
    fn test_silu_one() {
        // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        let mut x = [1.0];
        silu(&mut x);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((x[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_silu_batch() {
        let mut x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let original = x;
        silu(&mut x);
        for (i, (&out, &inp)) in x.iter().zip(original.iter()).enumerate() {
            let expected = inp / (1.0 + (-inp).exp());
            assert!((out - expected).abs() < 1e-5, "x[{i}]: {out} != {expected}");
        }
    }

    #[test]
    fn test_silu_elementwise_mul() {
        let mut gate = [1.0, -1.0, 2.0];
        let up = [2.0, 3.0, 0.5];
        let expected: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u): (&f32, &f32)| g / (1.0 + (-g).exp()) * u)
            .collect();
        silu_elementwise_mul(&mut gate, &up);
        for (i, (&out, &exp)) in gate.iter().zip(expected.iter()).enumerate() {
            assert!((out - exp).abs() < 1e-5, "x[{i}]: {out} != {exp}");
        }
    }

    #[test]
    fn test_elementwise_mul() {
        let mut a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        elementwise_mul(&mut a, &b);
        assert_eq!(a, [4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_elementwise_add() {
        let mut a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        elementwise_add(&mut a, &b);
        assert_eq!(a, [5.0, 7.0, 9.0]);
    }
}
