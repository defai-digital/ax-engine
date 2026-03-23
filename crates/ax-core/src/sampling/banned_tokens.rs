//! Hard token ban masking.

/// Mask explicitly banned token IDs by setting their logits to negative infinity.
pub fn apply_banned_token_mask(logits: &mut [f32], banned_tokens: &[u32]) {
    if banned_tokens.is_empty() {
        return;
    }

    for &token in banned_tokens {
        if let Some(logit) = logits.get_mut(token as usize) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_banned_token_mask_masks_only_banned_tokens() {
        let mut logits = [5.0, 4.0, 3.0];
        apply_banned_token_mask(&mut logits, &[1]);

        assert_eq!(logits[0], 5.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn test_apply_banned_token_mask_ignores_out_of_range_tokens() {
        let mut logits = [5.0, 4.0];
        let original = logits;
        apply_banned_token_mask(&mut logits, &[99]);
        assert_eq!(logits, original);
    }
}
