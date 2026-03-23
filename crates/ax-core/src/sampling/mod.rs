//! Token sampling from logits.
//!
//! Sampling pipeline (matches llama.cpp order):
//!   1. Logit bias
//!   2. Allowed-token masking
//!   3. Banned-token masking
//!   4. Repetition penalty
//!   5. Presence/frequency penalties
//!   6. Temperature scaling
//!   7. Top-k filtering
//!   8. Top-p (nucleus) filtering
//!   9. Min-p filtering
//!   10. Softmax to probabilities
//!   11. Weighted random selection (or argmax if temperature = 0)

pub mod allowed_tokens;
pub mod banned_tokens;
pub mod grammar;
pub mod logit_bias;
pub mod min_p;
pub mod penalties;
pub mod repetition;
pub mod temperature;
pub mod top_k;
pub mod top_p;

use std::collections::HashMap;

use crate::compute::softmax;

/// Sampling configuration matching llama.cpp defaults.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub logit_bias: Vec<LogitBias>,
    pub allowed_token_ids: Vec<u32>,
    pub banned_token_ids: Vec<u32>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub min_keep: usize,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            logit_bias: Vec::new(),
            allowed_token_ids: Vec::new(),
            banned_token_ids: Vec::new(),
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            min_keep: 1,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 64,
            seed: u64::MAX, // random
        }
    }
}

/// Additive bias for a specific token before sampling.
#[derive(Debug, Clone, PartialEq)]
pub struct LogitBias {
    pub token: u32,
    pub bias: f32,
}

/// Logprob metadata for a candidate token in the final sampling distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
}

/// Sampling result with the chosen token and optional top-logprobs metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledTokenInfo {
    pub token: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<TokenLogprob>,
}

/// Sampler state with RNG and reusable scratch buffers.
pub struct Sampler {
    config: SamplingConfig,
    rng_state: u64,
    /// Scratch buffer for top-p probability copy — avoids per-token heap allocation.
    scratch_probs: Vec<f32>,
    /// Scratch buffer for top-k / top-p index sort — avoids per-token heap allocation.
    scratch_indices: Vec<usize>,
    /// Scratch map for presence/frequency penalties.
    scratch_counts: HashMap<u32, u32>,
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
            scratch_counts: HashMap::new(),
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

        apply_penalties_and_filters_with_scratch(
            logits,
            &self.config,
            recent_tokens,
            &mut self.scratch_counts,
            &mut self.scratch_probs,
            &mut self.scratch_indices,
        );

        if self.config.temperature == 0.0 {
            return argmax(logits);
        }

        sample_filtered_logits_with_scratch(
            logits,
            &mut self.scratch_probs,
            &mut self.scratch_indices,
            &mut self.rng_state,
        )
    }

    /// Sample a token and return logprob metadata from the final distribution.
    ///
    /// `top_logprobs` controls how many of the most likely candidates to return
    /// from the filtered distribution.
    pub fn sample_with_logprobs(
        &mut self,
        logits: &mut [f32],
        recent_tokens: &[u32],
        top_logprobs: usize,
    ) -> SampledTokenInfo {
        assert!(!logits.is_empty(), "cannot sample from empty logits");

        apply_penalties_and_filters_with_scratch(
            logits,
            &self.config,
            recent_tokens,
            &mut self.scratch_counts,
            &mut self.scratch_probs,
            &mut self.scratch_indices,
        );

        if self.config.temperature == 0.0 {
            return greedy_token_info_with_scratch(
                logits,
                top_logprobs,
                &mut self.scratch_probs,
                &mut self.scratch_indices,
            );
        }

        sample_filtered_logits_info_with_scratch(
            logits,
            top_logprobs,
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

/// Sample a token and return logprob metadata from the final distribution.
pub fn sample_token_with_logprobs(
    logits: &mut [f32],
    config: &SamplingConfig,
    recent_tokens: &[u32],
    top_logprobs: usize,
) -> SampledTokenInfo {
    let mut rng = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);
    sample_token_with_rng_and_logprobs(logits, config, recent_tokens, top_logprobs, &mut rng)
}

fn sample_token_with_rng(
    logits: &mut [f32],
    config: &SamplingConfig,
    recent_tokens: &[u32],
    rng: &mut u64,
) -> u32 {
    assert!(!logits.is_empty(), "cannot sample from empty logits");

    let mut scratch_counts = HashMap::new();
    let mut scratch_probs = Vec::new();
    let mut scratch_indices = Vec::new();
    apply_penalties_and_filters_with_scratch(
        logits,
        config,
        recent_tokens,
        &mut scratch_counts,
        &mut scratch_probs,
        &mut scratch_indices,
    );

    if config.temperature == 0.0 {
        return argmax(logits);
    }

    sample_filtered_logits_with_scratch(logits, &mut scratch_probs, &mut scratch_indices, rng)
}

fn sample_token_with_rng_and_logprobs(
    logits: &mut [f32],
    config: &SamplingConfig,
    recent_tokens: &[u32],
    top_logprobs: usize,
    rng: &mut u64,
) -> SampledTokenInfo {
    assert!(!logits.is_empty(), "cannot sample from empty logits");

    let mut scratch_counts = HashMap::new();
    let mut scratch_probs = Vec::new();
    let mut scratch_indices = Vec::new();
    apply_penalties_and_filters_with_scratch(
        logits,
        config,
        recent_tokens,
        &mut scratch_counts,
        &mut scratch_probs,
        &mut scratch_indices,
    );

    if config.temperature == 0.0 {
        return greedy_token_info_with_scratch(
            logits,
            top_logprobs,
            &mut scratch_probs,
            &mut scratch_indices,
        );
    }

    sample_filtered_logits_info_with_scratch(
        logits,
        top_logprobs,
        &mut scratch_probs,
        &mut scratch_indices,
        rng,
    )
}

fn apply_penalties_and_filters_with_scratch(
    logits: &mut [f32],
    config: &SamplingConfig,
    recent_tokens: &[u32],
    scratch_counts: &mut HashMap<u32, u32>,
    scratch_probs: &mut Vec<f32>,
    scratch_indices: &mut Vec<usize>,
) {
    logit_bias::apply_logit_bias(logits, &config.logit_bias);
    allowed_tokens::apply_allowed_token_mask(logits, &config.allowed_token_ids);
    banned_tokens::apply_banned_token_mask(logits, &config.banned_token_ids);

    if config.repeat_penalty != 1.0 && !recent_tokens.is_empty() && config.repeat_last_n != 0 {
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

    penalties::apply_presence_frequency_penalties_with_counts(
        logits,
        recent_tokens,
        config.presence_penalty,
        config.frequency_penalty,
        scratch_counts,
    );

    if config.temperature == 0.0 {
        return;
    }

    temperature::apply_temperature(logits, config.temperature);
    top_k::apply_top_k_with_scratch(logits, config.top_k, config.min_keep, scratch_indices);
    top_p::apply_top_p_with_scratch(
        logits,
        config.top_p,
        config.min_keep,
        scratch_probs,
        scratch_indices,
    );
    min_p::apply_min_p_with_scratch(
        logits,
        config.min_p,
        config.min_keep,
        scratch_probs,
        scratch_indices,
    );
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

fn sample_filtered_logits_info_with_scratch(
    logits: &mut [f32],
    top_logprobs: usize,
    scratch_probs: &mut Vec<f32>,
    scratch_indices: &mut Vec<usize>,
    rng: &mut u64,
) -> SampledTokenInfo {
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
        let token = sample_categorical(logits, rng);
        let logprob = logits[token as usize].ln();
        let top_logprobs = collect_top_logprobs_from_probs(logits, top_logprobs, scratch_indices);
        return SampledTokenInfo {
            token,
            logprob,
            top_logprobs,
        };
    }

    scratch_probs.clear();
    if scratch_probs.capacity() < scratch_indices.len() {
        scratch_probs.reserve(scratch_indices.len() - scratch_probs.capacity());
    }
    scratch_probs.extend(scratch_indices.iter().map(|&idx| logits[idx]));
    softmax::softmax(scratch_probs);

    let top_logprobs =
        collect_top_logprobs_from_sorted_candidates(scratch_indices, scratch_probs, top_logprobs);

    let r = xorshift64(rng);
    let threshold = (r >> 11) as f64 / (1u64 << 53) as f64;

    let mut cumsum = 0.0f64;
    for (rank, &prob) in scratch_probs.iter().enumerate() {
        cumsum += prob as f64;
        if cumsum > threshold {
            return SampledTokenInfo {
                token: scratch_indices[rank] as u32,
                logprob: prob.ln(),
                top_logprobs,
            };
        }
    }

    let last_rank = scratch_indices.len() - 1;
    SampledTokenInfo {
        token: scratch_indices[last_rank] as u32,
        logprob: scratch_probs[last_rank].ln(),
        top_logprobs,
    }
}

fn greedy_token_info_with_scratch(
    logits: &[f32],
    top_logprobs: usize,
    scratch_probs: &mut Vec<f32>,
    scratch_indices: &mut Vec<usize>,
) -> SampledTokenInfo {
    let token = argmax(logits);

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
        let uniform_logprob = -(logits.len() as f32).ln();
        return SampledTokenInfo {
            token,
            logprob: uniform_logprob,
            top_logprobs: Vec::new(),
        };
    }

    scratch_indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    scratch_probs.clear();
    if scratch_probs.capacity() < scratch_indices.len() {
        scratch_probs.reserve(scratch_indices.len() - scratch_probs.capacity());
    }
    scratch_probs.extend(scratch_indices.iter().map(|&idx| logits[idx]));
    softmax::softmax(scratch_probs);

    let sampled_rank = scratch_indices
        .iter()
        .position(|&idx| idx == token as usize)
        .expect("argmax token must exist in finite candidate set");

    SampledTokenInfo {
        token,
        logprob: scratch_probs[sampled_rank].ln(),
        top_logprobs: collect_top_logprobs_from_sorted_candidates(
            scratch_indices,
            scratch_probs,
            top_logprobs,
        ),
    }
}

fn collect_top_logprobs_from_probs(
    probs: &[f32],
    top_logprobs: usize,
    scratch_indices: &mut Vec<usize>,
) -> Vec<TokenLogprob> {
    if top_logprobs == 0 {
        return Vec::new();
    }

    scratch_indices.clear();
    if scratch_indices.capacity() < probs.len() {
        scratch_indices.reserve(probs.len() - scratch_indices.capacity());
    }
    scratch_indices.extend(0..probs.len());
    scratch_indices.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

    collect_top_logprobs_from_sorted_candidates(
        scratch_indices,
        probs,
        top_logprobs.min(scratch_indices.len()),
    )
}

fn collect_top_logprobs_from_sorted_candidates(
    sorted_indices: &[usize],
    sorted_probs: &[f32],
    top_logprobs: usize,
) -> Vec<TokenLogprob> {
    if top_logprobs == 0 {
        return Vec::new();
    }

    sorted_indices
        .iter()
        .zip(sorted_probs.iter())
        .take(top_logprobs)
        .map(|(&idx, &prob)| TokenLogprob {
            token: idx as u32,
            logprob: prob.ln(),
        })
        .collect()
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
    fn test_greedy_with_presence_and_frequency_penalties() {
        let config = SamplingConfig {
            temperature: 0.0,
            presence_penalty: 0.4,
            frequency_penalty: 0.6,
            ..Default::default()
        };
        let mut logits = [5.0, 4.0, 2.0];
        let token = sample_token(&mut logits, &config, &[0, 0]);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_greedy_with_logit_bias_changes_choice() {
        let config = SamplingConfig {
            temperature: 0.0,
            logit_bias: vec![LogitBias {
                token: 2,
                bias: 3.0,
            }],
            ..Default::default()
        };
        let mut logits = [1.0, 4.0, 2.0];
        let token = sample_token(&mut logits, &config, &[]);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_logit_bias_applies_before_repetition_penalty() {
        let config = SamplingConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: -1,
            logit_bias: vec![LogitBias {
                token: 1,
                bias: 2.0,
            }],
            ..Default::default()
        };
        let mut logits = [1.0, 3.0];
        let token = sample_token(&mut logits, &config, &[1]);
        assert_eq!(token, 1);
        assert_eq!(logits[1], 2.5);
    }

    #[test]
    fn test_allowed_token_ids_restrict_choice() {
        let config = SamplingConfig {
            temperature: 0.0,
            allowed_token_ids: vec![2],
            ..Default::default()
        };
        let mut logits = [5.0, 4.0, 1.0];
        let token = sample_token(&mut logits, &config, &[]);
        assert_eq!(token, 2);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn test_allowed_token_ids_dominate_logit_bias_for_disallowed_tokens() {
        let config = SamplingConfig {
            temperature: 0.0,
            allowed_token_ids: vec![1],
            logit_bias: vec![LogitBias {
                token: 2,
                bias: 100.0,
            }],
            ..Default::default()
        };
        let mut logits = [1.0, 2.0, 3.0];
        let token = sample_token(&mut logits, &config, &[]);
        assert_eq!(token, 1);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_banned_token_ids_block_choice() {
        let config = SamplingConfig {
            temperature: 0.0,
            banned_token_ids: vec![0],
            ..Default::default()
        };
        let mut logits = [5.0, 4.0, 3.0];
        let token = sample_token(&mut logits, &config, &[]);
        assert_eq!(token, 1);
        assert_eq!(logits[0], f32::NEG_INFINITY);
    }

    #[test]
    fn test_banned_token_ids_override_allowlist() {
        let config = SamplingConfig {
            temperature: 0.0,
            allowed_token_ids: vec![0, 1],
            banned_token_ids: vec![0],
            ..Default::default()
        };
        let mut logits = [5.0, 4.0, 3.0];
        let token = sample_token(&mut logits, &config, &[]);
        assert_eq!(token, 1);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
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
            min_p: 0.0,
            seed: 123,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let mut logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let token = sampler.sample(&mut logits, &[]);
        assert!(token < 5, "token {token} out of range");
    }

    #[test]
    fn test_sample_with_min_p_filters_tail() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.2,
            min_keep: 1,
            seed: 7,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let mut logits = [5.0, 4.0, 0.5];

        for _ in 0..20 {
            let token = sampler.sample(&mut logits, &[]);
            assert!(token == 0 || token == 1, "unexpected token {token}");
            logits = [5.0, 4.0, 0.5];
        }
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
            logit_bias: Vec::new(),
            allowed_token_ids: Vec::new(),
            banned_token_ids: Vec::new(),
            temperature: 0.5,
            top_k: 3,
            top_p: 0.95,
            min_p: 0.0,
            min_keep: 1,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
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

    #[test]
    fn test_sample_with_logprobs_returns_sample_metadata() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            min_keep: 1,
            seed: 123,
            ..Default::default()
        };
        let mut sampler = Sampler::new(config);
        let mut logits = [5.0, 2.0, 1.0];

        let info = sampler.sample_with_logprobs(&mut logits, &[], 2);
        assert!(info.token < 3);
        assert!(info.logprob.is_finite());
        assert_eq!(info.top_logprobs.len(), 2);
        assert_eq!(info.top_logprobs[0].token, 0);
        assert!(info.top_logprobs[0].logprob >= info.top_logprobs[1].logprob);
    }

    #[test]
    fn test_greedy_logprobs_uses_post_penalty_distribution() {
        let config = SamplingConfig {
            temperature: 0.0,
            presence_penalty: 0.5,
            frequency_penalty: 0.5,
            ..Default::default()
        };
        let mut logits = [5.0, 4.0, 1.0];

        let info = sample_token_with_logprobs(&mut logits, &config, &[0, 0], 2);
        assert_eq!(info.token, 1);
        assert_eq!(info.top_logprobs[0].token, 1);
        assert!(info.logprob.is_finite());
    }

    #[test]
    fn test_repeat_last_n_limits_repetition_window() {
        let config = SamplingConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 1,
            ..Default::default()
        };
        let mut logits = [1.0, 5.0, 4.0];
        let token = sample_token(&mut logits, &config, &[1, 2]);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_repeat_last_n_zero_disables_repetition_penalty() {
        let config = SamplingConfig {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 0,
            ..Default::default()
        };
        let mut logits = [1.0, 5.0, 4.0];
        let token = sample_token(&mut logits, &config, &[1]);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_min_keep_preserves_candidate_floor() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 0.01,
            min_p: 0.9,
            min_keep: 2,
            seed: 9,
            ..Default::default()
        };
        let mut logits = [5.0, 4.0, 1.0];
        let mut sampler = Sampler::new(config);

        for _ in 0..20 {
            let token = sampler.sample(&mut logits, &[]);
            assert!(token == 0 || token == 1, "unexpected token {token}");
            logits = [5.0, 4.0, 1.0];
        }
    }
}
