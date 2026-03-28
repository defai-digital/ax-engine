//! Top-p (nucleus) filtering for logits.
//!
//! After converting logits to probabilities via softmax, keeps the smallest
//! set of tokens whose cumulative probability exceeds p. All other tokens
//! have their logits set to -inf.

use crate::compute::softmax;

/// Apply top-p (nucleus) filtering to logits (in-place).
///
/// Converts logits to probabilities, sorts descending, accumulates until
/// the cumulative probability exceeds `p`, then sets excluded logits to `-inf`.
///
/// If `p >= 1.0`, no filtering is applied. If `p <= 0.0`, only the top token survives.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    let mut scratch_probs = Vec::new();
    let mut scratch_indices = Vec::new();
    apply_top_p_with_scratch(logits, p, 1, &mut scratch_probs, &mut scratch_indices);
}

/// Apply top-p filtering using caller-provided scratch buffers.
///
/// Pass reusable Vecs to avoid per-call heap allocation.
pub(crate) fn apply_top_p_with_scratch(
    logits: &mut [f32],
    p: f32,
    min_keep: usize,
    scratch_probs: &mut Vec<f32>,
    scratch_indices: &mut Vec<usize>,
) {
    if p >= 1.0 {
        return;
    }
    let n = logits.len();
    if n == 0 {
        return;
    }

    // Reuse indices buffer, but only keep finite candidates. This makes the
    // common top-k -> top-p path scale with the surviving candidate count
    // instead of the full vocabulary size.
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
        // Preserve previous behavior for the all -inf case: leave logits alone
        // and let the later softmax provide its uniform fallback.
        return;
    }

    scratch_indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    // Reuse probs buffer in the sorted-candidate order.
    scratch_probs.clear();
    if scratch_probs.capacity() < scratch_indices.len() {
        scratch_probs.reserve(scratch_indices.len() - scratch_probs.capacity());
    }
    scratch_probs.extend(scratch_indices.iter().map(|&idx| logits[idx]));
    softmax::softmax(scratch_probs);

    let mut cumsum = 0.0f32;
    let mut cutoff = scratch_indices.len();
    for (rank, &prob) in scratch_probs.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff = rank + 1;
            break;
        }
    }
    cutoff = cutoff.max(min_keep.min(scratch_indices.len())).max(1);

    for &idx in &scratch_indices[cutoff..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_p_no_filter() {
        // p = 1.0 → keep all
        let mut logits = [1.0, 2.0, 3.0];
        let original = logits;
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_greedy() {
        // p very small → only top token survives
        let mut logits = [1.0, 5.0, 2.0];
        apply_top_p(&mut logits, 0.01);
        // Index 1 has highest logit → should survive
        assert!(logits[1].is_finite());
        // At least one finite value
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count >= 1);
    }

    #[test]
    fn test_top_p_keeps_top_tokens() {
        // Logits heavily skewed: softmax([10, 1, 1]) ≈ [0.9998, 0.0001, 0.0001]
        // p = 0.9 → only top token needed
        let mut logits = [10.0, 1.0, 1.0];
        apply_top_p(&mut logits, 0.9);
        assert!(logits[0].is_finite());
        // The top token's probability alone exceeds 0.9
    }

    #[test]
    fn test_top_p_uniform() {
        // Uniform logits: probs = [0.25, 0.25, 0.25, 0.25]
        // p = 0.6 → need at least 3 tokens (0.25 + 0.25 + 0.25 = 0.75 > 0.6)
        let mut logits = [0.0, 0.0, 0.0, 0.0];
        apply_top_p(&mut logits, 0.6);
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count >= 3, "expected >=3 tokens, got {finite_count}");
    }

    #[test]
    fn test_top_p_always_keeps_one() {
        // Even with p=0, at least one token survives
        let mut logits = [1.0, 2.0, 3.0];
        apply_top_p(&mut logits, 0.0);
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count >= 1);
    }

    #[test]
    fn test_top_p_empty() {
        let mut logits: [f32; 0] = [];
        apply_top_p(&mut logits, 0.5); // should not panic
    }

    #[test]
    fn test_top_p_only_considers_finite_candidates() {
        let mut logits = [2.0, 1.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        apply_top_p(&mut logits, 0.7);
        assert!(logits[0].is_finite());
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_all_neg_inf_is_noop() {
        let mut logits = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        apply_top_p(&mut logits, 0.5);
        assert!(logits.iter().all(|v| *v == f32::NEG_INFINITY));
    }

    #[test]
    fn test_top_p_honors_min_keep() {
        let mut logits = [5.0, 4.0, 3.0, 2.0];
        let mut scratch_probs = Vec::new();
        let mut scratch_indices = Vec::new();
        apply_top_p_with_scratch(
            &mut logits,
            0.01,
            3,
            &mut scratch_probs,
            &mut scratch_indices,
        );
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_count, 3);
    }
}
