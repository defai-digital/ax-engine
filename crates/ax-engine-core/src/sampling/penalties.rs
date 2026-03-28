//! OpenAI-style presence and frequency penalties.
//!
//! These penalties operate on the generated token history:
//! - presence penalty: subtract once if a token has appeared at least once
//! - frequency penalty: subtract once per prior occurrence

use std::collections::HashMap;

/// Apply OpenAI-style presence and frequency penalties to logits in-place.
///
/// Tokens that appear in `recent_tokens` have their logit reduced by:
///
/// `presence_penalty + frequency_penalty * count`
///
/// where `count` is the number of occurrences in `recent_tokens`.
pub fn apply_presence_frequency_penalties(
    logits: &mut [f32],
    recent_tokens: &[u32],
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    let mut counts = HashMap::new();
    apply_presence_frequency_penalties_with_counts(
        logits,
        recent_tokens,
        presence_penalty,
        frequency_penalty,
        &mut counts,
    );
}

pub(crate) fn apply_presence_frequency_penalties_with_counts(
    logits: &mut [f32],
    recent_tokens: &[u32],
    presence_penalty: f32,
    frequency_penalty: f32,
    counts: &mut HashMap<u32, u32>,
) {
    if recent_tokens.is_empty() || (presence_penalty == 0.0 && frequency_penalty == 0.0) {
        return;
    }

    counts.clear();
    for &token in recent_tokens {
        *counts.entry(token).or_insert(0) += 1;
    }

    for (&token, &count) in counts.iter() {
        let Some(logit) = logits.get_mut(token as usize) else {
            continue;
        };

        if presence_penalty != 0.0 {
            *logit -= presence_penalty;
        }
        if frequency_penalty != 0.0 {
            *logit -= frequency_penalty * count as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presence_penalty_applies_once_per_seen_token() {
        let mut logits = [5.0, 4.0, 3.0];
        apply_presence_frequency_penalties(&mut logits, &[0, 0, 2], 0.5, 0.0);
        assert_eq!(logits, [4.5, 4.0, 2.5]);
    }

    #[test]
    fn test_frequency_penalty_scales_with_occurrences() {
        let mut logits = [5.0, 4.0, 3.0];
        apply_presence_frequency_penalties(&mut logits, &[0, 0, 2], 0.0, 0.25);
        assert_eq!(logits, [4.5, 4.0, 2.75]);
    }

    #[test]
    fn test_presence_and_frequency_penalties_combine() {
        let mut logits = [5.0, 4.0, 3.0];
        apply_presence_frequency_penalties(&mut logits, &[0, 0, 2], 0.5, 0.25);
        assert_eq!(logits, [4.0, 4.0, 2.25]);
    }
}
