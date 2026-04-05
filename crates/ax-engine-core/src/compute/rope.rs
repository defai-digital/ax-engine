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

const STACK_TABLE_MAX: usize = 128;

#[inline]
fn fill_rope_tables(
    position: f32,
    head_dim: usize,
    freq_base: f32,
    cos_table: &mut [f32],
    sin_table: &mut [f32],
) {
    let half_dim = head_dim / 2;
    assert_eq!(cos_table.len(), half_dim);
    assert_eq!(sin_table.len(), half_dim);

    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / head_dim as f32;

    for i in 0..half_dim {
        // exp(-ln(base) * 2i/dim) is faster than 1/base.powf(2i/dim)
        let freq = (neg_log_base * i as f32 * dim_inv).exp();
        let theta = position * freq;
        (sin_table[i], cos_table[i]) = theta.sin_cos();
    }
}

#[inline]
fn with_rope_tables<R>(
    head_dim: usize,
    position: f32,
    freq_base: f32,
    f: impl FnOnce(&[f32], &[f32]) -> R,
) -> R {
    let half_dim = head_dim / 2;
    let mut cos_stack = [0.0f32; STACK_TABLE_MAX];
    let mut sin_stack = [0.0f32; STACK_TABLE_MAX];

    if half_dim <= STACK_TABLE_MAX {
        let cos_table = &mut cos_stack[..half_dim];
        let sin_table = &mut sin_stack[..half_dim];
        fill_rope_tables(position, head_dim, freq_base, cos_table, sin_table);
        f(cos_table, sin_table)
    } else {
        let mut cos_heap = vec![0.0f32; half_dim];
        let mut sin_heap = vec![0.0f32; half_dim];
        fill_rope_tables(position, head_dim, freq_base, &mut cos_heap, &mut sin_heap);
        f(&cos_heap, &sin_heap)
    }
}

#[inline]
fn apply_rope_pairs_scalar(buf: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    debug_assert_eq!(buf.len(), cos_table.len() * 2);
    debug_assert_eq!(cos_table.len(), sin_table.len());

    for i in 0..cos_table.len() {
        let cos_t = cos_table[i];
        let sin_t = sin_table[i];
        let idx = 2 * i;
        let x0 = buf[idx];
        let x1 = buf[idx + 1];
        buf[idx] = x0 * cos_t - x1 * sin_t;
        buf[idx + 1] = x0 * sin_t + x1 * cos_t;
    }
}

#[inline]
fn apply_rope_pairs_split_half_scalar(buf: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    debug_assert_eq!(cos_table.len(), sin_table.len());
    debug_assert_eq!(buf.len(), cos_table.len() * 2);

    let half = cos_table.len();
    let (lo, hi) = buf.split_at_mut(half);
    for i in 0..half {
        let cos_t = cos_table[i];
        let sin_t = sin_table[i];
        let x0 = lo[i];
        let x1 = hi[i];
        lo[i] = x0 * cos_t - x1 * sin_t;
        hi[i] = x0 * sin_t + x1 * cos_t;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn apply_rope_pairs_in_place(buf: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(buf.len(), cos_table.len() * 2);
    debug_assert_eq!(cos_table.len(), sin_table.len());

    let n_pairs = cos_table.len();
    let chunks = n_pairs / 4;

    unsafe {
        for chunk in 0..chunks {
            let pair_offset = chunk * 4;
            let buf_offset = pair_offset * 2;

            let in_lo = vld1q_f32(buf.as_ptr().add(buf_offset));
            let in_hi = vld1q_f32(buf.as_ptr().add(buf_offset + 4));
            let even = vuzp1q_f32(in_lo, in_hi);
            let odd = vuzp2q_f32(in_lo, in_hi);
            let cos_v = vld1q_f32(cos_table.as_ptr().add(pair_offset));
            let sin_v = vld1q_f32(sin_table.as_ptr().add(pair_offset));

            let rot_even = vsubq_f32(vmulq_f32(even, cos_v), vmulq_f32(odd, sin_v));
            let rot_odd = vaddq_f32(vmulq_f32(even, sin_v), vmulq_f32(odd, cos_v));
            let out_lo = vzip1q_f32(rot_even, rot_odd);
            let out_hi = vzip2q_f32(rot_even, rot_odd);

            vst1q_f32(buf.as_mut_ptr().add(buf_offset), out_lo);
            vst1q_f32(buf.as_mut_ptr().add(buf_offset + 4), out_hi);
        }
    }

    let tail_pairs = chunks * 4;
    apply_rope_pairs_scalar(
        &mut buf[tail_pairs * 2..],
        &cos_table[tail_pairs..],
        &sin_table[tail_pairs..],
    );
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn apply_rope_pairs_in_place(buf: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    apply_rope_pairs_scalar(buf, cos_table, sin_table);
}

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

    with_rope_tables(
        head_dim,
        position as f32,
        freq_base,
        |cos_table, sin_table| {
            apply_rope_pairs_in_place(q, cos_table, sin_table);
            apply_rope_pairs_in_place(k, cos_table, sin_table);
        },
    );
}

/// Apply RoPE to query only (used when key is not needed, e.g. cached).
pub fn apply_rope_q(q: &mut [f32], head_dim: usize, position: usize, freq_base: f32) {
    assert_eq!(q.len(), head_dim);
    assert!(head_dim.is_multiple_of(2));

    with_rope_tables(
        head_dim,
        position as f32,
        freq_base,
        |cos_table, sin_table| {
            apply_rope_pairs_in_place(q, cos_table, sin_table);
        },
    );
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

    with_rope_tables(head_dim, position, freq_base, |cos_table, sin_table| {
        for q_head in q[..n_heads * head_dim].chunks_exact_mut(head_dim) {
            apply_rope_pairs_in_place(q_head, cos_table, sin_table);
        }

        for k_head in k[..n_kv_heads * head_dim].chunks_exact_mut(head_dim) {
            apply_rope_pairs_in_place(k_head, cos_table, sin_table);
        }
    });
}

/// Apply RoPE to the first `rotary_dim` values of each head.
///
/// Models like Qwen3.5 expose `n_rot < head_dim` and only rotate a prefix of
/// each head while preserving the trailing channels.
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_multi_head_partial_scaled(
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: f32,
    freq_base: f32,
) {
    assert!(q.len() >= n_heads * head_dim);
    assert!(k.len() >= n_kv_heads * head_dim);
    assert!(rotary_dim <= head_dim);
    assert!(rotary_dim.is_multiple_of(2));

    if rotary_dim == 0 {
        return;
    }

    with_rope_tables(rotary_dim, position, freq_base, |cos_table, sin_table| {
        for q_head in q[..n_heads * head_dim].chunks_exact_mut(head_dim) {
            apply_rope_pairs_in_place(&mut q_head[..rotary_dim], cos_table, sin_table);
        }

        for k_head in k[..n_kv_heads * head_dim].chunks_exact_mut(head_dim) {
            apply_rope_pairs_in_place(&mut k_head[..rotary_dim], cos_table, sin_table);
        }
    });
}

/// Apply RoPE (NeoX pair layout) to the first `rotary_dim` values of each head.
///
/// NeoX-style rotary uses split-half pairs `(x[i], x[i + rotary_dim/2])` instead
/// of adjacent pairs `(x[2i], x[2i+1])`.
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_multi_head_neox_partial_scaled(
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: f32,
    freq_base: f32,
) {
    assert!(q.len() >= n_heads * head_dim);
    assert!(k.len() >= n_kv_heads * head_dim);
    assert!(rotary_dim <= head_dim);
    assert!(rotary_dim.is_multiple_of(2));

    if rotary_dim == 0 {
        return;
    }

    with_rope_tables(rotary_dim, position, freq_base, |cos_table, sin_table| {
        for q_head in q[..n_heads * head_dim].chunks_exact_mut(head_dim) {
            apply_rope_pairs_split_half_scalar(&mut q_head[..rotary_dim], cos_table, sin_table);
        }

        for k_head in k[..n_kv_heads * head_dim].chunks_exact_mut(head_dim) {
            apply_rope_pairs_split_half_scalar(&mut k_head[..rotary_dim], cos_table, sin_table);
        }
    });
}

/// Apply NeoX-style RoPE with explicit per-dimension frequency factors (proportional RoPE).
///
/// Used by Gemma4 global layers where `rope_freqs.weight` provides per-pair
/// frequency multipliers. Dimensions with large freq_factors (~1e30) effectively
/// lose position information. Uses split-half pairs: `(x[i], x[i + half_dim])`.
///
/// `freq_factors`: [head_dim/2] frequency multipliers per pair
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_neox_with_freq_factors(
    q: &mut [f32],
    k: &mut [f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    position: f32,
    freq_base: f32,
    freq_factors: &[f32],
) {
    assert!(q.len() >= n_heads * head_dim);
    assert!(k.len() >= n_kv_heads * head_dim);
    assert!(head_dim.is_multiple_of(2));
    let half_dim = head_dim / 2;
    assert_eq!(freq_factors.len(), half_dim);

    // Build cos/sin tables incorporating freq_factors
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / head_dim as f32;

    let mut cos_table = vec![0.0f32; half_dim];
    let mut sin_table = vec![0.0f32; half_dim];
    for i in 0..half_dim {
        let base_freq = (neg_log_base * i as f32 * dim_inv).exp();
        let freq = base_freq * freq_factors[i];
        let theta = position * freq;
        (sin_table[i], cos_table[i]) = theta.sin_cos();
    }

    for q_head in q[..n_heads * head_dim].chunks_exact_mut(head_dim) {
        apply_rope_pairs_split_half_scalar(q_head, &cos_table, &sin_table);
    }

    for k_head in k[..n_kv_heads * head_dim].chunks_exact_mut(head_dim) {
        apply_rope_pairs_split_half_scalar(k_head, &cos_table, &sin_table);
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

    #[test]
    fn test_rope_pairs_in_place_matches_scalar_large_head_dim() {
        let head_dim = 16;
        let mut actual = [
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
        ];
        let mut expected = actual;

        with_rope_tables(head_dim, 7.25, FREQ_BASE, |cos_table, sin_table| {
            apply_rope_pairs_in_place(&mut actual, cos_table, sin_table);
            apply_rope_pairs_scalar(&mut expected, cos_table, sin_table);
        });

        for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((act - exp).abs() < 1e-5, "buf[{i}]: {act} != {exp}");
        }
    }

    #[test]
    fn test_rope_multi_head_scaled_matches_scalar_reference_large_head_dim() {
        let head_dim = 16;
        let n_heads = 3;
        let n_kv_heads = 2;
        let position = 11.5;
        let mut q = vec![0.0f32; n_heads * head_dim];
        let mut k = vec![0.0f32; n_kv_heads * head_dim];

        for (i, v) in q.iter_mut().enumerate() {
            *v = i as f32 * 0.25 - 3.0;
        }
        for (i, v) in k.iter_mut().enumerate() {
            *v = i as f32 * -0.125 + 1.0;
        }

        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        with_rope_tables(head_dim, position, FREQ_BASE, |cos_table, sin_table| {
            for q_head in q_ref.chunks_exact_mut(head_dim) {
                apply_rope_pairs_scalar(q_head, cos_table, sin_table);
            }
            for k_head in k_ref.chunks_exact_mut(head_dim) {
                apply_rope_pairs_scalar(k_head, cos_table, sin_table);
            }
        });

        apply_rope_multi_head_scaled(
            &mut q, &mut k, n_heads, n_kv_heads, head_dim, position, FREQ_BASE,
        );

        for (i, (&act, &exp)) in q.iter().zip(q_ref.iter()).enumerate() {
            assert!((act - exp).abs() < 1e-5, "q[{i}]: {act} != {exp}");
        }
        for (i, (&act, &exp)) in k.iter().zip(k_ref.iter()).enumerate() {
            assert!((act - exp).abs() < 1e-5, "k[{i}]: {act} != {exp}");
        }
    }

    #[test]
    fn test_rope_multi_head_partial_scaled_rotates_prefix_only() {
        let head_dim = 8usize;
        let rotary_dim = 4usize;
        let n_heads = 1usize;
        let n_kv_heads = 1usize;
        let position = 3.5f32;

        let mut q = [1.0f32, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0];
        let mut k = [4.0f32, 3.0, 2.0, 1.0, 20.0, 21.0, 22.0, 23.0];
        let q_orig = q;
        let k_orig = k;

        apply_rope_multi_head_partial_scaled(
            &mut q, &mut k, n_heads, n_kv_heads, head_dim, rotary_dim, position, FREQ_BASE,
        );

        let mut q_ref_prefix = q_orig[..rotary_dim].to_vec();
        let mut k_ref_prefix = k_orig[..rotary_dim].to_vec();
        apply_rope_multi_head_scaled(
            &mut q_ref_prefix,
            &mut k_ref_prefix,
            n_heads,
            n_kv_heads,
            rotary_dim,
            position,
            FREQ_BASE,
        );

        for i in 0..rotary_dim {
            assert!((q[i] - q_ref_prefix[i]).abs() < 1e-6, "q[{i}]");
            assert!((k[i] - k_ref_prefix[i]).abs() < 1e-6, "k[{i}]");
        }
        for i in rotary_dim..head_dim {
            assert!((q[i] - q_orig[i]).abs() < 1e-6, "q tail[{i}]");
            assert!((k[i] - k_orig[i]).abs() < 1e-6, "k tail[{i}]");
        }
    }

    #[test]
    fn test_rope_multi_head_neox_partial_scaled_rotates_split_halves() {
        let n_heads = 1usize;
        let n_kv_heads = 1usize;
        let head_dim = 8usize;
        let rotary_dim = 4usize;
        let position = 9.0f32;

        let mut q = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut k = vec![5.0f32, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0];
        let mut expected_q = q.clone();
        let mut expected_k = k.clone();

        apply_rope_multi_head_neox_partial_scaled(
            &mut q, &mut k, n_heads, n_kv_heads, head_dim, rotary_dim, position, FREQ_BASE,
        );

        let mut cos_table = vec![0.0f32; rotary_dim / 2];
        let mut sin_table = vec![0.0f32; rotary_dim / 2];
        super::fill_rope_tables(
            position,
            rotary_dim,
            FREQ_BASE,
            &mut cos_table,
            &mut sin_table,
        );
        apply_rope_pairs_split_half_scalar(&mut expected_q[..rotary_dim], &cos_table, &sin_table);
        apply_rope_pairs_split_half_scalar(&mut expected_k[..rotary_dim], &cos_table, &sin_table);

        for (actual, expected) in q.iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in k.iter().zip(expected_k.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // Tail (non-rotary) channels remain unchanged.
        assert_eq!(&q[rotary_dim..], &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(&k[rotary_dim..], &[50.0, 60.0, 70.0, 80.0]);
    }
}
