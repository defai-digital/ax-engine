//! Hard token allowlist masking.

/// Mask all logits except the explicitly allowed token IDs.
///
/// If `allowed_tokens` is empty, no masking is applied. Out-of-range token IDs
/// are ignored. If none of the provided token IDs exist in `logits`, the logits
/// are left unchanged so callers can decide how to handle invalid configuration.
pub fn apply_allowed_token_mask(logits: &mut [f32], allowed_tokens: &[u32]) {
    if allowed_tokens.is_empty() {
        return;
    }

    let mut has_valid = false;
    for &token in allowed_tokens {
        if (token as usize) < logits.len() {
            has_valid = true;
            break;
        }
    }
    if !has_valid {
        return;
    }

    let mut mask = vec![false; logits.len()];
    for &token in allowed_tokens {
        if let Some(slot) = mask.get_mut(token as usize) {
            *slot = true;
        }
    }

    for (idx, logit) in logits.iter_mut().enumerate() {
        if !mask[idx] {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_allowed_token_mask_keeps_only_allowed_tokens() {
        let mut logits = [5.0, 4.0, 3.0, 2.0];
        apply_allowed_token_mask(&mut logits, &[1, 3]);

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 4.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 2.0);
    }

    #[test]
    fn test_apply_allowed_token_mask_ignores_invalid_allowlist() {
        let mut logits = [5.0, 4.0];
        let original = logits;
        apply_allowed_token_mask(&mut logits, &[99]);
        assert_eq!(logits, original);
    }
}
