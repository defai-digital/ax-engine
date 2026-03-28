//! Repetition penalty for logits.
//!
//! Penalizes tokens that have appeared recently in the generated sequence.
//! Matches llama.cpp behavior: if logit > 0, divide by penalty; if < 0, multiply.

/// Apply repetition penalty to logits (in-place).
///
/// For each token in `recent_tokens`, if its logit is positive it is
/// divided by `penalty`, if negative it is multiplied. This reduces
/// the probability of recently generated tokens.
///
/// `penalty` of 1.0 means no penalty. Typical values: 1.0-1.3.
pub fn apply_repetition_penalty(logits: &mut [f32], recent_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 || recent_tokens.is_empty() {
        return;
    }

    let n = logits.len();
    for &token in recent_tokens {
        let idx = token as usize;
        if idx >= n {
            continue;
        }
        if logits[idx] > 0.0 {
            logits[idx] /= penalty;
        } else {
            logits[idx] *= penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_penalty_positive() {
        let mut logits = [0.0, 2.0, 4.0, 1.0];
        apply_repetition_penalty(&mut logits, &[1, 2], 2.0);
        assert_eq!(logits[0], 0.0); // untouched
        assert_eq!(logits[1], 1.0); // 2.0 / 2.0
        assert_eq!(logits[2], 2.0); // 4.0 / 2.0
        assert_eq!(logits[3], 1.0); // untouched
    }

    #[test]
    fn test_repetition_penalty_negative() {
        let mut logits = [0.0, -2.0, -4.0, 1.0];
        apply_repetition_penalty(&mut logits, &[1, 2], 2.0);
        assert_eq!(logits[1], -4.0); // -2.0 * 2.0
        assert_eq!(logits[2], -8.0); // -4.0 * 2.0
    }

    #[test]
    fn test_repetition_penalty_no_penalty() {
        let mut logits = [1.0, 2.0, 3.0];
        let original = logits;
        apply_repetition_penalty(&mut logits, &[0, 1], 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repetition_penalty_empty_tokens() {
        let mut logits = [1.0, 2.0, 3.0];
        let original = logits;
        apply_repetition_penalty(&mut logits, &[], 2.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repetition_penalty_out_of_range() {
        // Token ID beyond logits range should be silently ignored
        let mut logits = [1.0, 2.0];
        let original = logits;
        apply_repetition_penalty(&mut logits, &[5, 100], 2.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repetition_penalty_zero_logit() {
        // Zero logit is unchanged (neither positive nor negative path)
        let mut logits = [0.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert_eq!(logits[0], 0.0);
    }
}
