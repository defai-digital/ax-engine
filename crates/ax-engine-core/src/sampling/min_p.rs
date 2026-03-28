//! Minimum-probability filtering for logits.
//!
//! Keeps tokens whose probability is at least `p * max_prob`, where `max_prob`
//! is the probability of the most likely surviving token.

use crate::compute::softmax;

/// Apply min-p filtering to logits (in-place).
///
/// Converts finite logits to probabilities via softmax, keeps tokens whose
/// probability is at least `p * max_prob`, and masks the rest to `-inf`.
///
/// If `p <= 0.0`, no filtering is applied. At least one token always survives.
pub fn apply_min_p(logits: &mut [f32], p: f32) {
    let mut scratch_probs = Vec::new();
    let mut scratch_indices = Vec::new();
    apply_min_p_with_scratch(logits, p, 1, &mut scratch_probs, &mut scratch_indices);
}

/// Apply min-p filtering using caller-provided scratch buffers.
pub(crate) fn apply_min_p_with_scratch(
    logits: &mut [f32],
    p: f32,
    min_keep: usize,
    scratch_probs: &mut Vec<f32>,
    scratch_indices: &mut Vec<usize>,
) {
    if p <= 0.0 {
        return;
    }
    let n = logits.len();
    if n == 0 {
        return;
    }

    scratch_indices.clear();
    if scratch_indices.capacity() < n {
        scratch_indices.reserve(n - scratch_indices.capacity());
    }
    scratch_indices.extend(
        logits
            .iter()
            .enumerate()
            .filter_map(|(idx, &logit)| logit.is_finite().then_some(idx)),
    );
    if scratch_indices.is_empty() {
        return;
    }

    scratch_indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    scratch_probs.clear();
    if scratch_probs.capacity() < scratch_indices.len() {
        scratch_probs.reserve(scratch_indices.len() - scratch_probs.capacity());
    }
    scratch_probs.extend(scratch_indices.iter().map(|&idx| logits[idx]));
    softmax::softmax(scratch_probs);

    let max_prob = scratch_probs[0];
    let threshold = p * max_prob;
    let mut cutoff = 1usize;
    while cutoff < scratch_probs.len() && scratch_probs[cutoff] >= threshold {
        cutoff += 1;
    }
    cutoff = cutoff.max(min_keep.min(scratch_probs.len()));

    for &idx in &scratch_indices[cutoff..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_p_disabled_is_noop() {
        let mut logits = [1.0, 2.0, 3.0];
        let original = logits;
        apply_min_p(&mut logits, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_min_p_keeps_tokens_above_relative_threshold() {
        let mut logits = [5.0, 4.0, 0.5];
        apply_min_p(&mut logits, 0.2);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_p_always_keeps_top_token() {
        let mut logits = [1.0, 0.0, -1.0];
        apply_min_p(&mut logits, 1.5);
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_p_ignores_non_finite_candidates() {
        let mut logits = [2.0, 1.0, f32::NEG_INFINITY];
        apply_min_p(&mut logits, 0.8);
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_min_p_empty_is_noop() {
        let mut logits: [f32; 0] = [];
        apply_min_p(&mut logits, 0.5);
    }

    #[test]
    fn test_min_p_honors_min_keep() {
        let mut logits = [5.0, 4.0, 1.0, 0.5];
        let mut scratch_probs = Vec::new();
        let mut scratch_indices = Vec::new();
        apply_min_p_with_scratch(
            &mut logits,
            0.8,
            2,
            &mut scratch_probs,
            &mut scratch_indices,
        );
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_count, 2);
    }
}
