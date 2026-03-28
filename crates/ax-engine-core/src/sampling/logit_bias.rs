//! Additive token-specific logit bias.

use super::LogitBias;

/// Apply additive logit bias in-place.
///
/// Each configured bias is added to the matching token logit before the rest
/// of the sampling pipeline runs.
pub fn apply_logit_bias(logits: &mut [f32], biases: &[LogitBias]) {
    if biases.is_empty() {
        return;
    }

    for bias in biases {
        let Some(logit) = logits.get_mut(bias.token as usize) else {
            continue;
        };
        *logit += bias.bias;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_logit_bias_adds_bias_to_target_tokens() {
        let mut logits = [1.0, 2.0, 3.0];
        let biases = [
            LogitBias {
                token: 0,
                bias: 0.5,
            },
            LogitBias {
                token: 2,
                bias: -1.0,
            },
        ];

        apply_logit_bias(&mut logits, &biases);

        assert_eq!(logits, [1.5, 2.0, 2.0]);
    }

    #[test]
    fn test_apply_logit_bias_ignores_out_of_range_tokens() {
        let mut logits = [1.0, 2.0];
        let biases = [LogitBias {
            token: 99,
            bias: 5.0,
        }];

        apply_logit_bias(&mut logits, &biases);

        assert_eq!(logits, [1.0, 2.0]);
    }
}
