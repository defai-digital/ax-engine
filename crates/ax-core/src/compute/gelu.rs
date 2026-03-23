//! GELU (Gaussian Error Linear Unit) activation.
//!
//! GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//!
//! Used in Gemma3's feed-forward network instead of SiLU.

/// Constant: sqrt(2/pi) ≈ 0.7978845608
const SQRT_2_PI: f32 = 0.797_884_6;

/// In-place GELU activation (tanh approximation).
pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        let inner = SQRT_2_PI * (*v + 0.044715 * x3);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}

/// GELU followed by element-wise multiply: x = GELU(x) * y
///
/// This is the GeGLU pattern used in Gemma3:
///   gate = GELU(W_gate * input), up = W_up * input,
///   then output = gate * up.
pub fn gelu_elementwise_mul(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (xi, &yi) in x.iter_mut().zip(y.iter()) {
        let x3 = *xi * *xi * *xi;
        let inner = SQRT_2_PI * (*xi + 0.044715 * x3);
        *xi = 0.5 * *xi * (1.0 + inner.tanh()) * yi;
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
