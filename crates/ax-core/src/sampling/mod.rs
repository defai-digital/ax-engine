//! Token sampling from logits.
//!
//! Sampling pipeline (matches llama.cpp order):
//!   1. Repetition penalty
//!   2. Temperature scaling
//!   3. Top-k filtering
//!   4. Top-p (nucleus) filtering
//!   5. Softmax to probabilities
//!   6. Weighted random selection (or argmax if temperature = 0)

pub mod grammar;
pub mod repetition;
pub mod temperature;
pub mod top_k;
pub mod top_p;

use crate::compute::softmax;

/// Sampling configuration matching llama.cpp defaults.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: u64::MAX, // random
        }
    }
}

/// Sampler state with RNG and reusable scratch buffers.
pub struct Sampler {
    config: SamplingConfig,
    rng_state: u64,
    /// Scratch buffer for top-p probability copy — avoids per-token heap allocation.
    scratch_probs: Vec<f32>,
    /// Scratch buffer for top-k / top-p index sort — avoids per-token heap allocation.
    scratch_indices: Vec<usize>,
}

impl Sampler {
    /// Create a new sampler.
    pub fn new(config: SamplingConfig) -> Self {
        let seed = if config.seed == u64::MAX {
            // Use system time as seed
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        } else {
            config.seed
        };
        Self {
            config,
            rng_state: seed,
            scratch_probs: Vec::new(),
            scratch_indices: Vec::new(),
        }
    }

    /// Sample a token from logits, applying the full sampling pipeline.
    ///
    /// `logits` is modified in-place (penalties, temperature, filtering applied).
    /// `recent_tokens` is the recent token history for repetition penalty.
    ///
    /// Reuses internal scratch buffers to avoid per-token heap allocation.
    pub fn sample(&mut self, logits: &mut [f32], recent_tokens: &[u32]) -> u32 {
        assert!(!logits.is_empty(), "cannot sample from empty logits");

        // 1. Repetition penalty
        if self.config.repeat_penalty != 1.0 && !recent_tokens.is_empty() {
            let window = if self.config.repeat_last_n > 0 {
                let start = recent_tokens
                    .len()
                    .saturating_sub(self.config.repeat_last_n as usize);
                &recent_tokens[start..]
            } else {
                recent_tokens
            };
            repetition::apply_repetition_penalty(logits, window, self.config.repeat_penalty);
        }

        // 2. Greedy (temperature = 0): just return argmax
        if self.config.temperature == 0.0 {
            return argmax(logits);
        }

        // 3. Temperature scaling
        temperature::apply_temperature(logits, self.config.temperature);

        // 4. Top-k filtering (reuses scratch_indices)
        top_k::apply_top_k_with_scratch(logits, self.config.top_k, &mut self.scratch_indices);

        // 5. Top-p filtering (reuses scratch_probs + scratch_indices)
        top_p::apply_top_p_with_scratch(
            logits,
            self.config.top_p,
            &mut self.scratch_probs,
            &mut self.scratch_indices,
        );

        // 6. Softmax + categorical sampling over the surviving candidate set.
        sample_filtered_logits_with_scratch(
            logits,
            &mut self.scratch_probs,
            &mut self.scratch_indices,
            &mut self.rng_state,
        )
    }

    /// Sample a uniform random value in [0, 1) using the internal RNG.
    ///
    /// Used by speculative decoding for accept/reject decisions.
    pub fn sample_uniform(&mut self) -> f32 {
        let r = xorshift64(&mut self.rng_state);
        (r >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Sample a token directly from a pre-computed probability distribution.
    ///
    /// Unlike `sample`, this does NOT apply temperature / top-k / top-p.
    /// Useful when a probability distribution has already been computed (e.g.
    /// for speculative decoding correction tokens sampled from `max(0, p_target - p_draft)`).
    pub fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        sample_categorical(probs, &mut self.rng_state)
    }

    /// Get the sampling config.
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }
}

/// Sample a token from logits using the configured sampling chain.
///
/// This is the stateless version — pass your own rng state.
pub fn sample_token(logits: &mut [f32], config: &SamplingConfig, recent_tokens: &[u32]) -> u32 {
    let mut rng = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    sample_token_with_rng(logits, config, recent_tokens, &mut rng)
}

fn sample_token_with_rng(
    logits: &mut [f32],
    config: &SamplingConfig,
    recent_tokens: &[u32],
    rng: &mut u64,
) -> u32 {
    assert!(!logits.is_empty(), "cannot sample from empty logits");

    // 1. Repetition penalty
    if config.repeat_penalty != 1.0 && !recent_tokens.is_empty() {
        let window = if config.repeat_last_n > 0 {
            let start = recent_tokens
                .len()
                .saturating_sub(config.repeat_last_n as usize);
            &recent_tokens[start..]
        } else {
            recent_tokens
        };
        repetition::apply_repetition_penalty(logits, window, config.repeat_penalty);
    }

    // 2. Greedy (temperature = 0): just return argmax
    if config.temperature == 0.0 {
        return argmax(logits);
    }

    // 3. Temperature scaling
    temperature::apply_temperature(logits, config.temperature);

    // 4. Top-k filtering
    top_k::apply_top_k(logits, config.top_k);

    // 5. Top-p filtering
    top_p::apply_top_p(logits, config.top_p);

    // 6. Softmax + categorical sampling over the surviving candidate set.
    let mut scratch_probs = Vec::new();
    let mut scratch_indices = Vec::new();
    sample_filtered_logits_with_scratch(logits, &mut scratch_probs, &mut scratch_indices, rng)
}

/// Return the index of the maximum value (greedy / argmax decoding).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Sample from a categorical probability distribution using xorshift64.
fn sample_categorical(probs: &[f32], rng: &mut u64) -> u32 {
    let r = xorshift64(rng);
    // Convert to [0, 1) float
    let threshold = (r >> 11) as f64 / (1u64 << 53) as f64;

    let mut cumsum = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p as f64;
        if cumsum > threshold {
            return i as u32;
        }
    }
    // Fallback: return last token (rounding errors)
    (probs.len() - 1) as u32
}

fn sample_filtered_logits_with_scratch(
    logits: &mut [f32],
    scratch_probs: &mut Vec<f32>,
    scratch_indices: &mut Vec<usize>,
    rng: &mut u64,
) -> u32 {
    scratch_indices.clear();
    if scratch_indices.capacity() < logits.len() {
        scratch_indices.reserve(logits.len() - scratch_indices.capacity());
    }
    scratch_indices.extend(
        logits
            .iter()
            .enumerate()
            .filter_map(|(idx, &logit)| logit.is_finite().then_some(idx)),
    );

    if scratch_indices.is_empty() {
        softmax::softmax(logits);
        return sample_categorical(logits, rng);
    }

    scratch_probs.clear();
    if scratch_probs.capacity() < scratch_indices.len() {
        scratch_probs.reserve(scratch_indices.len() - scratch_probs.capacity());
    }
    scratch_probs.extend(scratch_indices.iter().map(|&idx| logits[idx]));
    softmax::softmax(scratch_probs);

    let r = xorshift64(rng);
    let threshold = (r >> 11) as f64 / (1u64 << 53) as f64;

    let mut cumsum = 0.0f64;
    for (rank, &p) in scratch_probs.iter().enumerate() {
        cumsum += p as f64;
        if cumsum > threshold {
            return scratch_indices[rank] as u32;
        }
    }

    *scratch_indices.last().unwrap() as u32
}

/// Xorshift64 PRNG — fast, decent quality, no external deps.
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1; // avoid zero fixpoint
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 5.0]), 2);
    }

    #[test]
    fn test_greedy_sampling() {
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let mut logits = [1.0, 5.0, 2.0, 4.0];
        let token = sample_token(&mut logits, &config, &[]);
        assert_eq!(token, 1); // index of 5.0
    }

    #[test]
    fn test_greedy_with_repetition_penalty() {
        let config = SamplingConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 10,
            ..Default::default()
        };
        // Without penalty, token 1 (logit 5.0) wins
        // With penalty on token 1: logit becomes 5.0/2.0 = 2.5
        // Token 3 (logit 4.0) should now win
        let mut logits = [1.0, 5.0, 2.0, 4.0];
        let token = sample_token(&mut logits, &config, &[1]);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_sample_deterministic() {
        // Same seed should produce same result
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 42,
            ..Default::default()
        };
        let mut sampler1 = Sampler::new(config.clone());
        let mut sampler2 = Sampler::new(config);

        let mut logits1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut logits2 = [1.0, 2.0, 3.0, 4.0, 5.0];

        let t1 = sampler1.sample(&mut logits1, &[]);
        let t2 = sampler2.sample(&mut logits2, &[]);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_sample_produces_valid_token() {
        let config = SamplingConfig {
            temperature: 0.8,
            top_k: 3,
            top_p: 0.9,
            seed: 123,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let mut logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let token = sampler.sample(&mut logits, &[]);
        assert!(token < 5, "token {token} out of range");
    }

    #[test]
    fn test_xorshift_not_stuck() {
        let mut state = 42u64;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            seen.insert(xorshift64(&mut state));
        }
        assert!(
            seen.len() > 90,
            "RNG seems stuck: only {} unique values",
            seen.len()
        );
    }

    #[test]
    fn test_categorical_deterministic() {
        // Probability 1.0 on index 2 should always return 2
        let probs = [0.0, 0.0, 1.0, 0.0];
        let mut rng = 42u64;
        for _ in 0..10 {
            assert_eq!(sample_categorical(&probs, &mut rng), 2);
        }
    }

    #[test]
    fn test_categorical_distribution() {
        // Heavily skewed: ~99% on index 0
        let probs = [0.99, 0.005, 0.005];
        let mut rng = 1u64;
        let mut counts = [0u32; 3];
        for _ in 0..1000 {
            let idx = sample_categorical(&probs, &mut rng);
            counts[idx as usize] += 1;
        }
        // Index 0 should get the vast majority
        assert!(counts[0] > 900, "expected >900, got {}", counts[0]);
    }

    #[test]
    fn test_full_pipeline() {
        let config = SamplingConfig {
            temperature: 0.5,
            top_k: 3,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_last_n: 5,
            seed: 999,
        };
        let mut sampler = Sampler::new(config);

        let mut logits = vec![0.0; 100];
        logits[42] = 10.0;
        logits[7] = 8.0;
        logits[99] = 6.0;

        // Token 42 has highest logit, low temperature → should be strongly preferred
        let mut got_42 = 0;
        for _ in 0..20 {
            let mut l = logits.clone();
            let t = sampler.sample(&mut l, &[]);
            if t == 42 {
                got_42 += 1;
            }
        }
        assert!(
            got_42 > 10,
            "expected token 42 often, got it {got_42}/20 times"
        );
    }

    #[test]
    fn test_sample_all_neg_inf_returns_valid_token() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 123,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let mut logits = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let token = sampler.sample(&mut logits, &[]);
        assert!(token < logits.len() as u32);
    }

    #[test]
    fn test_sample_prefiltered_logits_stays_within_finite_candidates() {
        let mut logits = [5.0, 4.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mut scratch_probs = Vec::new();
        let mut scratch_indices = Vec::new();
        let mut rng = 42u64;

        for _ in 0..20 {
            let token = sample_filtered_logits_with_scratch(
                &mut logits,
                &mut scratch_probs,
                &mut scratch_indices,
                &mut rng,
            );
            assert!(token == 0 || token == 1, "unexpected token {token}");
        }
    }
}
