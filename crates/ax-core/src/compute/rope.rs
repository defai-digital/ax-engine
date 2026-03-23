//! Rotary Position Embedding (RoPE) for LLaMA.
//!
//! RoPE encodes position information by rotating pairs of dimensions
//! in the query and key vectors using sinusoidal functions.
//!
//! For each pair (x_2i, x_{2i+1}) at position p:
//!   freq = 1 / (base ^ (2i / dim))
//!   theta = p * freq
//!   x_2i'    = x_2i * cos(theta) - x_{2i+1} * sin(theta)
//!   x_{2i+1}' = x_2i * sin(theta) + x_{2i+1} * cos(theta)

/// Apply RoPE to a single head's query and key vectors.
///
/// `q`: query vector for one head (length = head_dim)
/// `k`: key vector for one head (length = head_dim)
/// `head_dim`: dimension per head (must be even)
/// `position`: token position in the sequence
/// `freq_base`: base frequency (typically 10000.0 for LLaMA)
pub fn apply_rope(q: &mut [f32], k: &mut [f32], head_dim: usize, position: usize, freq_base: f32) {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k.len(), head_dim);
    assert!(
        head_dim.is_multiple_of(2),
        "RoPE head_dim must be even, got {head_dim}"
    );

    let half_dim = head_dim / 2;
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / head_dim as f32;

    for i in 0..half_dim {
        // exp(-ln(base) * 2i/dim) is faster than 1/base.powf(2i/dim)
        let freq = (neg_log_base * i as f32 * dim_inv).exp();
        let theta = position as f32 * freq;
        let (sin_t, cos_t) = theta.sin_cos();

        // Rotate query pair
        let q0 = q[2 * i];
        let q1 = q[2 * i + 1];
        q[2 * i] = q0 * cos_t - q1 * sin_t;
        q[2 * i + 1] = q0 * sin_t + q1 * cos_t;

        // Rotate key pair
        let k0 = k[2 * i];
        let k1 = k[2 * i + 1];
        k[2 * i] = k0 * cos_t - k1 * sin_t;
        k[2 * i + 1] = k0 * sin_t + k1 * cos_t;
    }
}

/// Apply RoPE to query only (used when key is not needed, e.g. cached).
pub fn apply_rope_q(q: &mut [f32], head_dim: usize, position: usize, freq_base: f32) {
    assert_eq!(q.len(), head_dim);
    assert!(head_dim.is_multiple_of(2));

    let half_dim = head_dim / 2;
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / head_dim as f32;

    for i in 0..half_dim {
        let freq = (neg_log_base * i as f32 * dim_inv).exp();
        let theta = position as f32 * freq;
        let (sin_t, cos_t) = theta.sin_cos();

        let q0 = q[2 * i];
        let q1 = q[2 * i + 1];
        q[2 * i] = q0 * cos_t - q1 * sin_t;
        q[2 * i + 1] = q0 * sin_t + q1 * cos_t;
    }
}

/// Apply RoPE to multiple heads at once (contiguous in memory).
///
/// `qkv`: interleaved q and k vectors for n_heads, layout:
///   [q_head0, q_head1, ..., k_head0, k_head1, ...]
/// `n_heads`: number of query heads
/// `n_kv_heads`: number of key/value heads (for GQA, may be < n_heads)
/// `head_dim`: dimension per head
/// `position`: token position
/// `freq_base`: RoPE base frequency
pub fn apply_rope_multi_head(
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: usize,
    freq_base: f32,
) {
    apply_rope_multi_head_scaled(
        q,
        k,
        n_heads,
        n_kv_heads,
        head_dim,
        position as f32,
        freq_base,
    );
}

/// Apply RoPE with a float position (supports fractional positions from linear scaling).
pub fn apply_rope_multi_head_scaled(
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: f32,
    freq_base: f32,
) {
    assert!(q.len() >= n_heads * head_dim);
    assert!(k.len() >= n_kv_heads * head_dim);

    // Precompute cos/sin table for this position (stack-allocated for typical head_dim <= 256)
    let half_dim = head_dim / 2;
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / head_dim as f32;

    // Stack buffer for common head_dim sizes (up to 256 → 128 pairs → 1 KiB)
    const STACK_MAX: usize = 128;
    let mut cos_stack = [0.0f32; STACK_MAX];
    let mut sin_stack = [0.0f32; STACK_MAX];
    let mut cos_heap;
    let mut sin_heap;
    let (cos_table, sin_table): (&[f32], &[f32]) = if half_dim <= STACK_MAX {
        for i in 0..half_dim {
            let freq = (neg_log_base * i as f32 * dim_inv).exp();
            let theta = position * freq;
            (sin_stack[i], cos_stack[i]) = theta.sin_cos();
        }
        (&cos_stack[..half_dim], &sin_stack[..half_dim])
    } else {
        cos_heap = vec![0.0f32; half_dim];
        sin_heap = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            let freq = (neg_log_base * i as f32 * dim_inv).exp();
            let theta = position * freq;
            (sin_heap[i], cos_heap[i]) = theta.sin_cos();
        }
        (&cos_heap[..], &sin_heap[..])
    };

    // Rotate all query heads
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in 0..half_dim {
            let q0 = q[offset + 2 * i];
            let q1 = q[offset + 2 * i + 1];
            q[offset + 2 * i] = q0 * cos_table[i] - q1 * sin_table[i];
            q[offset + 2 * i + 1] = q0 * sin_table[i] + q1 * cos_table[i];
        }
    }

    // Rotate all key heads
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        for i in 0..half_dim {
            let k0 = k[offset + 2 * i];
            let k1 = k[offset + 2 * i + 1];
            k[offset + 2 * i] = k0 * cos_table[i] - k1 * sin_table[i];
            k[offset + 2 * i + 1] = k0 * sin_table[i] + k1 * cos_table[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FREQ_BASE: f32 = 10000.0;

    #[test]
    fn test_rope_position_zero() {
        // At position 0, all thetas are 0, cos=1, sin=0 → no rotation
        let mut q = [1.0, 2.0, 3.0, 4.0];
        let mut k = [5.0, 6.0, 7.0, 8.0];
        let orig_q = q;
        let orig_k = k;
        apply_rope(&mut q, &mut k, 4, 0, FREQ_BASE);
        for i in 0..4 {
            assert!(
                (q[i] - orig_q[i]).abs() < 1e-6,
                "q[{i}]: {} != {}",
                q[i],
                orig_q[i]
            );
            assert!(
                (k[i] - orig_k[i]).abs() < 1e-6,
                "k[{i}]: {} != {}",
                k[i],
                orig_k[i]
            );
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        // Rotation preserves vector magnitude for each pair
        let mut q = [3.0, 4.0, 1.0, 0.0];
        let mut k = [0.0, 1.0, 1.0, 1.0];

        let q_norms_before: Vec<f32> = (0..2)
            .map(|i| f32::sqrt(q[2 * i] * q[2 * i] + q[2 * i + 1] * q[2 * i + 1]))
            .collect();
        let k_norms_before: Vec<f32> = (0..2)
            .map(|i| f32::sqrt(k[2 * i] * k[2 * i] + k[2 * i + 1] * k[2 * i + 1]))
            .collect();

        apply_rope(&mut q, &mut k, 4, 42, FREQ_BASE);

        for i in 0..2 {
            let q_norm = (q[2 * i] * q[2 * i] + q[2 * i + 1] * q[2 * i + 1]).sqrt();
            let k_norm = (k[2 * i] * k[2 * i] + k[2 * i + 1] * k[2 * i + 1]).sqrt();
            assert!(
                (q_norm - q_norms_before[i]).abs() < 1e-4,
                "q pair {i}: norm {q_norm} != {}",
                q_norms_before[i]
            );
            assert!(
                (k_norm - k_norms_before[i]).abs() < 1e-4,
                "k pair {i}: norm {k_norm} != {}",
                k_norms_before[i]
            );
        }
    }

    #[test]
    fn test_rope_known_rotation() {
        // For dim=2 (single pair), freq = 1/10000^0 = 1, theta = position
        // At position = pi/2: cos=0, sin=1
        // (1, 0) → (1*0 - 0*1, 1*1 + 0*0) = (0, 1)
        let pos_quarter_turn = (std::f32::consts::FRAC_PI_2) as usize; // 1 (rounded)
        // Use exact float position via freq_base = 1/(pi/2) trick:
        // theta = position * (1 / base^(0/2)) = position * 1
        // We want theta = pi/2, so position = pi/2 ≈ 1.5708 → not integer
        // Instead: set freq_base to make theta exact at position=1
        // freq = 1/base^0 = 1 always for first pair, so theta = position
        // Use position = 1, so theta = 1 rad
        let mut q = [1.0, 0.0];
        let mut k = [1.0, 0.0];
        apply_rope(&mut q, &mut k, 2, 1, FREQ_BASE);
        // theta = 1.0 (rad), cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert!((q[0] - cos1).abs() < 1e-5, "q[0]={}, expected {cos1}", q[0]);
        assert!((q[1] - sin1).abs() < 1e-5, "q[1]={}, expected {sin1}", q[1]);

        // Verify position=1 doesn't affect the test
        let _ = pos_quarter_turn;
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let mut q1 = [1.0, 0.0, 0.0, 1.0];
        let mut k1 = [1.0, 0.0, 0.0, 1.0];
        let mut q2 = [1.0, 0.0, 0.0, 1.0];
        let mut k2 = [1.0, 0.0, 0.0, 1.0];

        apply_rope(&mut q1, &mut k1, 4, 5, FREQ_BASE);
        apply_rope(&mut q2, &mut k2, 4, 10, FREQ_BASE);

        // Different positions should produce different rotations
        let diff: f32 = q1.iter().zip(q2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "positions 5 and 10 should differ");
    }

    #[test]
    fn test_rope_multi_head() {
        let head_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 2;
        let mut q = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut k = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

        // Single-head reference
        let mut q_ref = [1.0, 0.0, 0.0, 1.0];
        let mut k_ref = [1.0, 0.0, 0.0, 1.0];
        apply_rope(&mut q_ref, &mut k_ref, head_dim, 7, FREQ_BASE);

        apply_rope_multi_head(&mut q, &mut k, n_heads, n_kv_heads, head_dim, 7, FREQ_BASE);

        // Both heads should match the single-head reference
        for i in 0..head_dim {
            assert!(
                (q[i] - q_ref[i]).abs() < 1e-5,
                "head0 q[{i}]: {} != {}",
                q[i],
                q_ref[i]
            );
            assert!(
                (q[head_dim + i] - q_ref[i]).abs() < 1e-5,
                "head1 q[{i}]: {} != {}",
                q[head_dim + i],
                q_ref[i]
            );
        }
    }

    #[test]
    fn test_rope_q_only() {
        let mut q1 = [1.0, 2.0, 3.0, 4.0];
        let mut q2 = [1.0, 2.0, 3.0, 4.0];
        let mut k_dummy = [0.0, 0.0, 0.0, 0.0];
        apply_rope(&mut q1, &mut k_dummy, 4, 5, FREQ_BASE);
        apply_rope_q(&mut q2, 4, 5, FREQ_BASE);
        for i in 0..4 {
            assert!(
                (q1[i] - q2[i]).abs() < 1e-6,
                "q[{i}]: {} != {}",
                q1[i],
                q2[i]
            );
        }
    }
}
