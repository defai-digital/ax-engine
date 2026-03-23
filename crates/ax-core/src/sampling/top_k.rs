//! Top-k filtering for logits.
//!
//! Keeps only the k highest logits and sets the rest to -inf.
//! This restricts sampling to the k most likely tokens.

/// Apply top-k filtering to logits (in-place).
///
/// Keeps the `k` largest logit values, sets all others to `-inf`.
/// If `k <= 0` or `k >= logits.len()`, no filtering is applied.
pub fn apply_top_k(logits: &mut [f32], k: i32) {
    let mut scratch = Vec::new();
    apply_top_k_with_scratch(logits, k, 1, &mut scratch);
}

/// Apply top-k filtering using a caller-provided scratch buffer.
///
/// `scratch` is cleared and resized in-place — pass a reusable Vec to
/// avoid per-call allocation.
pub(crate) fn apply_top_k_with_scratch(
    logits: &mut [f32],
    k: i32,
    min_keep: usize,
    scratch: &mut Vec<usize>,
) {
    let n = logits.len();
    if k <= 0 || k as usize >= n {
        return;
    }
    let k = (k as usize).max(min_keep.min(n));

    scratch.clear();
    // Reserve without zeroing; extend fills exactly n elements.
    if scratch.capacity() < n {
        scratch.reserve(n - scratch.capacity());
    }
    scratch.extend(0..n);
    scratch.select_nth_unstable_by(k, |&a, &b| logits[b].total_cmp(&logits[a]));

    for &i in &scratch[k..] {
        logits[i] = f32::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_basic() {
        let mut logits = [1.0, 4.0, 2.0, 5.0, 3.0];
        apply_top_k(&mut logits, 2);
        // Top 2: indices 3 (5.0) and 1 (4.0) should survive
        assert_eq!(logits[3], 5.0);
        assert_eq!(logits[1], 4.0);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_one() {
        let mut logits = [3.0, 1.0, 2.0];
        apply_top_k(&mut logits, 1);
        assert_eq!(logits[0], 3.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_no_filter() {
        // k >= len → no filtering
        let mut logits = [1.0, 2.0, 3.0];
        let original = logits;
        apply_top_k(&mut logits, 5);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_zero() {
        // k = 0 → no filtering
        let mut logits = [1.0, 2.0, 3.0];
        let original = logits;
        apply_top_k(&mut logits, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_negative() {
        // k < 0 → no filtering
        let mut logits = [1.0, 2.0, 3.0];
        let original = logits;
        apply_top_k(&mut logits, -1);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_preserves_count() {
        let mut logits = [5.0, 3.0, 1.0, 4.0, 2.0];
        apply_top_k(&mut logits, 3);
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_count, 3);
    }

    #[test]
    fn test_top_k_honors_min_keep() {
        let mut logits = [5.0, 4.0, 3.0, 2.0];
        let mut scratch = Vec::new();
        apply_top_k_with_scratch(&mut logits, 1, 2, &mut scratch);
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_count, 2);
    }
}
