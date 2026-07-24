//! N-gram loop detection for generation stop (WS-C2 / R-C2).
//!
//! Detects a repeated token pattern of length `p ∈ [min_pattern, max_pattern]`
//! occurring `min_count` consecutive times ending at the cursor. When detected,
//! generation ends with [`crate::sampling::StopReason::LoopDetected`].
//!
//! This is distinct from [`crate::sampling::SamplingParams::no_repeat_ngram_size`],
//! which bans logits for completing an n-gram; loop detection **stops** after a
//! collapse has already been emitted.

/// Configuration for sliding-window n-gram loop detection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoopDetectionConfig {
    pub min_pattern_size: usize,
    pub max_pattern_size: usize,
    pub min_count: usize,
}

impl LoopDetectionConfig {
    /// Gemma 4 family defaults (mlxcel / vLLM conservative ladder).
    pub const GEMMA4_DEFAULT: Self = Self {
        min_pattern_size: 1,
        max_pattern_size: 20,
        min_count: 4,
    };

    /// Disabled detector (no pattern size).
    pub const DISABLED: Self = Self {
        min_pattern_size: 1,
        max_pattern_size: 0,
        min_count: 4,
    };

    pub fn is_enabled(self) -> bool {
        self.max_pattern_size > 0
            && self.min_count >= 2
            && self.min_pattern_size >= 1
            && self.min_pattern_size <= self.max_pattern_size
    }

    /// Build from request override fields. `max_pattern_size == 0` disables.
    pub fn from_request(
        min_pattern_size: Option<u32>,
        max_pattern_size: Option<u32>,
        min_count: Option<u32>,
        default: Self,
    ) -> Self {
        let max = max_pattern_size.map(|v| v as usize);
        if max == Some(0) {
            return Self::DISABLED;
        }
        Self {
            min_pattern_size: min_pattern_size
                .map(|v| v as usize)
                .unwrap_or(default.min_pattern_size)
                .max(1),
            max_pattern_size: max.unwrap_or(default.max_pattern_size),
            min_count: min_count
                .map(|v| v as usize)
                .unwrap_or(default.min_count)
                .max(2),
        }
    }
}

/// Returns true when `tokens` ends with a repeated pattern under `config`.
///
/// Only tokens already generated (including the latest) are inspected; prompt
/// tokens should not be passed unless the product wants prompt-aware detection.
pub fn detects_loop(tokens: &[u32], config: LoopDetectionConfig) -> bool {
    if !config.is_enabled() || tokens.is_empty() {
        return false;
    }
    let max_p = config.max_pattern_size.min(tokens.len() / config.min_count);
    if max_p < config.min_pattern_size {
        return false;
    }
    // Prefer longer patterns first so a multi-token collapse is not reported as
    // a shorter sub-pattern when both would match.
    for pattern_len in (config.min_pattern_size..=max_p).rev() {
        let need = pattern_len.saturating_mul(config.min_count);
        if tokens.len() < need {
            continue;
        }
        let window = &tokens[tokens.len() - need..];
        let pattern = &window[..pattern_len];
        let mut matched = true;
        for rep in 1..config.min_count {
            let start = rep * pattern_len;
            if &window[start..start + pattern_len] != pattern {
                matched = false;
                break;
            }
        }
        if matched {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_never_detects() {
        let tokens = vec![1, 1, 1, 1, 1, 1];
        assert!(!detects_loop(&tokens, LoopDetectionConfig::DISABLED));
        assert!(!detects_loop(
            &tokens,
            LoopDetectionConfig {
                max_pattern_size: 0,
                ..LoopDetectionConfig::GEMMA4_DEFAULT
            }
        ));
    }

    #[test]
    fn single_token_repeat_at_count() {
        let cfg = LoopDetectionConfig::GEMMA4_DEFAULT;
        // 3 repeats — not enough
        assert!(!detects_loop(&[7, 7, 7], cfg));
        // 4 repeats — detect
        assert!(detects_loop(&[7, 7, 7, 7], cfg));
        // longer history ending in 4
        assert!(detects_loop(&[1, 2, 3, 7, 7, 7, 7], cfg));
    }

    #[test]
    fn multi_token_pattern() {
        let cfg = LoopDetectionConfig::GEMMA4_DEFAULT;
        // pattern [1,2] × 4
        let tokens = [9, 1, 2, 1, 2, 1, 2, 1, 2];
        assert!(detects_loop(&tokens, cfg));
        // only 3 repeats
        assert!(!detects_loop(&[1, 2, 1, 2, 1, 2], cfg));
    }

    #[test]
    fn boundary_max_pattern() {
        let cfg = LoopDetectionConfig {
            min_pattern_size: 1,
            max_pattern_size: 2,
            min_count: 3,
        };
        // length-3 pattern should not be considered
        let long_pat = [1, 2, 3, 1, 2, 3, 1, 2, 3];
        assert!(!detects_loop(&long_pat, cfg));
        // length-2 does
        assert!(detects_loop(&[1, 2, 1, 2, 1, 2], cfg));
    }

    #[test]
    fn from_request_max_zero_disables() {
        let cfg = LoopDetectionConfig::from_request(
            None,
            Some(0),
            None,
            LoopDetectionConfig::GEMMA4_DEFAULT,
        );
        assert!(!cfg.is_enabled());
    }

    #[test]
    fn empty_tokens() {
        assert!(!detects_loop(&[], LoopDetectionConfig::GEMMA4_DEFAULT));
    }
}
