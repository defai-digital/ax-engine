use std::collections::{BTreeMap, HashSet};

use mlx_sys::{MlxArray, eval_first_u32, multiply, random_categorical};

/// Minimal xorshift64 PRNG — no external dependency.
///
/// Used for per-request temperature sampling.  Each request gets its own
/// independent RNG seeded from the request ID so deterministic seeds produce
/// reproducible outputs.
#[derive(Clone, Copy)]
pub struct Xorshift64(pub u64);

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        // Seed must be non-zero; mix with a prime to avoid bad seeds like 0.
        let s = seed.wrapping_add(0x9e3779b97f4a7c15);
        Self(if s == 0 { 1 } else { s })
    }

    /// Generate next random u64.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 * (1.0 / (1u64 << 53) as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlxSamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub repetition_context_size: Option<u32>,
}

impl MlxSamplingParams {
    pub const fn new(temperature: f32, top_p: f32, top_k: u32) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty: 1.0,
            repetition_context_size: None,
        }
    }

    pub const fn greedy() -> Self {
        Self::new(0.0, 1.0, 0)
    }

    pub const fn with_repetition_penalty(
        mut self,
        repetition_penalty: f32,
        repetition_context_size: Option<u32>,
    ) -> Self {
        self.repetition_penalty = repetition_penalty;
        self.repetition_context_size = repetition_context_size;
        self
    }

    pub fn uses_repetition_penalty(self) -> bool {
        self.repetition_penalty.is_finite()
            && self.repetition_penalty > 0.0
            && self.repetition_penalty != 1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlxSamplingRequest<'a> {
    pub params: MlxSamplingParams,
    pub repetition_tokens: &'a [u32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenDistribution {
    entries: Vec<(u32, f32)>,
}

impl TokenDistribution {
    pub fn new(entries: Vec<(u32, f32)>) -> Option<Self> {
        if entries.is_empty() {
            return None;
        }
        let sum: f32 = entries.iter().map(|(_, p)| *p).sum();
        if sum <= 0.0 || !sum.is_finite() {
            return None;
        }
        Some(Self {
            entries: entries
                .into_iter()
                .filter_map(|(token, prob)| {
                    let normalized = prob / sum;
                    (normalized > 0.0 && normalized.is_finite()).then_some((token, normalized))
                })
                .collect(),
        })
        .filter(|distribution| !distribution.entries.is_empty())
    }

    pub fn entries(&self) -> &[(u32, f32)] {
        &self.entries
    }

    pub fn probability(&self, token: u32) -> f32 {
        self.entries
            .iter()
            .find(|(candidate, _)| *candidate == token)
            .map_or(0.0, |(_, prob)| *prob)
    }

    pub fn sample_with_logprob(&self, rng: &mut Xorshift64) -> Option<(u32, f32)> {
        let token = sample_from_token_distribution(self, rng)?;
        let prob = self.probability(token);
        Some((token, prob.max(1e-37_f32).ln().max(-30.0)))
    }
}

impl<'a> MlxSamplingRequest<'a> {
    pub const fn new(params: MlxSamplingParams, repetition_tokens: &'a [u32]) -> Self {
        Self {
            params,
            repetition_tokens,
        }
    }
}

impl Default for MlxSamplingParams {
    fn default() -> Self {
        Self::greedy()
    }
}

/// Sample one token index from logits with temperature scaling.
///
/// When `temperature` is 0.0 or logits is empty, falls back to argmax.
/// Uses a numerically stable softmax (shift by max before exp).
///
/// Does not require GPU — caller must have already eval'd logits to CPU.
pub fn sample_categorical(
    logits: &[f32],
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut adjusted_logits_buf: Vec<f32>;
    let logits = if sampling.uses_repetition_penalty() && !repetition_tokens.is_empty() {
        adjusted_logits_buf = logits.to_vec();
        logits_with_repetition_penalty_in_place(
            &mut adjusted_logits_buf,
            sampling.repetition_penalty,
            recent_repetition_tokens(repetition_tokens, sampling.repetition_context_size),
        );
        adjusted_logits_buf.as_slice()
    } else {
        logits
    };
    if sampling.temperature <= 0.0 {
        return argmax_f32(logits);
    }

    let inv_temp = 1.0 / sampling.temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Fast path: no top-k/top-p filtering. Avoids index-tracking allocation
    // and the extra O(vocab) filtered_sum pass — the common case for plain
    // temperature sampling.
    if sampling.top_k == 0 && sampling.top_p >= 1.0 {
        let probs: Vec<f32> = logits
            .iter()
            .map(|&l| {
                let p = ((l - max_l) * inv_temp).exp();
                if p.is_finite() { p } else { 0.0 }
            })
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum == 0.0 || !sum.is_finite() {
            return argmax_f32(logits);
        }
        let threshold = rng.next_f32() * sum;
        let mut cumsum = 0.0f32;
        for (i, p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= threshold {
                return i as u32;
            }
        }
        return probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Filtered path: top-k or top-p active — track indices for truncation.
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| {
            let prob = ((logit - max_l) * inv_temp).exp();
            (idx, if prob.is_finite() { prob } else { 0.0 })
        })
        .collect();
    let sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if sum == 0.0 || !sum.is_finite() {
        return argmax_f32(logits);
    }

    apply_top_k_top_p(&mut candidates, sampling.top_k, sampling.top_p);
    let filtered_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if filtered_sum == 0.0 || !filtered_sum.is_finite() {
        return argmax_f32(logits);
    }

    let threshold = rng.next_f32() * filtered_sum;
    let mut cumsum = 0.0f32;
    for (i, p) in candidates.iter() {
        cumsum += p;
        if cumsum >= threshold {
            return *i as u32;
        }
    }
    candidates
        .iter()
        .copied()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Sample from a pre-filtered set of `(token_id, logit)` candidates.
///
/// This preserves the same temperature/top-p behavior as `sample_categorical`
/// for callers that have already selected the top-k candidates on GPU and only
/// want to transfer that small candidate set to CPU.
pub fn sample_indexed_categorical(
    logits: &[f32],
    indices: &[u32],
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
) -> Option<u32> {
    if logits.is_empty() || logits.len() != indices.len() {
        return None;
    }

    let best = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| indices[i])
        .unwrap_or(0);
    if sampling.temperature <= 0.0 {
        return Some(best);
    }

    let inv_temp = 1.0 / sampling.temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .zip(indices.iter())
        .map(|(&logit, &idx)| {
            let prob = ((logit - max_l) * inv_temp).exp();
            (idx as usize, if prob.is_finite() { prob } else { 0.0 })
        })
        .collect();
    let sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if sum == 0.0 || !sum.is_finite() {
        return Some(best);
    }

    apply_top_k_top_p(&mut candidates, sampling.top_k, sampling.top_p);
    let filtered_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if filtered_sum == 0.0 || !filtered_sum.is_finite() {
        return Some(best);
    }

    let threshold = rng.next_f32() * filtered_sum;
    let mut cumsum = 0.0f32;
    for (idx, prob) in candidates.iter() {
        cumsum += *prob;
        if cumsum >= threshold {
            return Some(*idx as u32);
        }
    }
    candidates
        .iter()
        .copied()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
}

/// Sample from a pre-filtered top-k candidate set and return `(token, log_prob)`.
///
/// Combines sampling and log-probability calculation in a single pass over the
/// candidate set, which is more efficient than calling `sample_indexed_categorical`
/// and `indexed_token_logprob` separately.  Used by MTP stochastic draft sampling
/// so the rejection-sampling acceptance check has an accurate `q_draft(d)`.
///
/// Returns `None` when the candidate set is empty or malformed.
pub fn sample_indexed_categorical_with_logprob(
    logits: &[f32],
    indices: &[u32],
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
) -> Option<(u32, f32)> {
    let distribution = indexed_token_distribution(logits, indices, sampling)?;
    distribution.sample_with_logprob(rng)
}

pub fn sample_indexed_categorical_with_logprob_and_distribution(
    logits: &[f32],
    indices: &[u32],
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
) -> Option<(u32, f32, TokenDistribution)> {
    let distribution = indexed_token_distribution(logits, indices, sampling)?;
    let (token, log_prob) = distribution.sample_with_logprob(rng)?;
    Some((token, log_prob, distribution))
}

pub fn indexed_token_distribution(
    logits: &[f32],
    indices: &[u32],
    sampling: MlxSamplingParams,
) -> Option<TokenDistribution> {
    if logits.is_empty() || logits.len() != indices.len() {
        return None;
    }
    let best_i = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    if sampling.temperature <= 0.0 {
        return TokenDistribution::new(vec![(indices[best_i], 1.0)]);
    }

    let inv_temp = 1.0 / sampling.temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .zip(indices.iter())
        .map(|(&logit, &idx)| {
            let p = ((logit - max_l) * inv_temp).exp();
            (idx as usize, if p.is_finite() { p } else { 0.0 })
        })
        .collect();

    let total: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if total == 0.0 || !total.is_finite() {
        return TokenDistribution::new(vec![(indices[best_i], 1.0)]);
    }

    // Apply the sampler's top-k/top-p over the caller-provided candidate set.
    // The set is usually already narrowed on GPU for bandwidth, but it can be
    // much larger than the sampler's requested top-k.
    apply_top_k_top_p(&mut candidates, sampling.top_k, sampling.top_p);
    let filtered_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if filtered_sum == 0.0 || !filtered_sum.is_finite() {
        return TokenDistribution::new(vec![(indices[best_i], 1.0)]);
    }

    TokenDistribution::new(
        candidates
            .into_iter()
            .map(|(token, prob)| (token as u32, prob))
            .collect(),
    )
}

/// Return the log-probability of `token` within a pre-filtered candidate set.
///
/// The candidate set is treated like the post-top-k set used by
/// `sample_indexed_categorical`; top-p is applied over that set before
/// normalisation.
pub fn indexed_token_logprob(
    logits: &[f32],
    indices: &[u32],
    token: u32,
    sampling: MlxSamplingParams,
) -> Option<f32> {
    if logits.is_empty() || logits.len() != indices.len() {
        return None;
    }
    if sampling.temperature <= 0.0 {
        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| indices[i])?;
        return if best == token { Some(0.0) } else { None };
    }

    let inv_temp = 1.0 / sampling.temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .zip(indices.iter())
        .map(|(&logit, &idx)| {
            let prob = ((logit - max_l) * inv_temp).exp();
            (idx as usize, if prob.is_finite() { prob } else { 0.0 })
        })
        .collect();

    apply_top_k_top_p(&mut candidates, sampling.top_k, sampling.top_p);
    let filtered_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if filtered_sum == 0.0 || !filtered_sum.is_finite() {
        return None;
    }

    candidates
        .iter()
        .find(|(idx, _)| *idx == token as usize)
        .map(|(_, prob)| (*prob / filtered_sum).ln().max(-30.0))
}

/// Sample one token with temperature / top-k / top-p filtering; also return its
/// log-probability under the filtered distribution.
///
/// Used for MTP draft tokens so the caller can perform rejection-sampling
/// acceptance: `accept_prob = min(1, p_target(d) / p_draft(d))`.
///
/// Returns `(token, log_prob)`.  When `temperature <= 0.0` (greedy), returns
/// `(argmax, 0.0)` — the convention that log_prob=0 signals a point-mass draft
/// that should fall back to greedy acceptance rather than rejection sampling.
pub fn sample_categorical_with_logprob(
    logits: &[f32],
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
) -> (u32, f32) {
    let Some(distribution) = token_distribution(logits, sampling) else {
        return (argmax_f32(logits), 0.0);
    };
    distribution
        .sample_with_logprob(rng)
        .unwrap_or((argmax_f32(logits), 0.0))
}

pub fn sample_categorical_with_logprob_and_distribution(
    logits: &[f32],
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
) -> (u32, f32, Option<TokenDistribution>) {
    let Some(distribution) = token_distribution(logits, sampling) else {
        return (argmax_f32(logits), 0.0, None);
    };
    let (token, log_prob) = distribution
        .sample_with_logprob(rng)
        .unwrap_or((argmax_f32(logits), 0.0));
    (token, log_prob, Some(distribution))
}

pub fn token_distribution(
    logits: &[f32],
    sampling: MlxSamplingParams,
) -> Option<TokenDistribution> {
    if logits.is_empty() {
        return None;
    }
    if sampling.temperature <= 0.0 {
        return TokenDistribution::new(vec![(argmax_f32(logits), 1.0)]);
    }

    let inv_temp = 1.0 / sampling.temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| {
            let p = ((logit - max_l) * inv_temp).exp();
            (idx, if p.is_finite() { p } else { 0.0 })
        })
        .collect();

    let total_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if total_sum == 0.0 || !total_sum.is_finite() {
        return TokenDistribution::new(vec![(argmax_f32(logits), 1.0)]);
    }

    if sampling.top_k > 0 || sampling.top_p < 1.0 {
        apply_top_k_top_p(&mut candidates, sampling.top_k, sampling.top_p);
    }

    let filtered_sum: f32 = candidates.iter().map(|(_, p)| *p).sum();
    if filtered_sum == 0.0 || !filtered_sum.is_finite() {
        return TokenDistribution::new(vec![(argmax_f32(logits), 1.0)]);
    }

    TokenDistribution::new(
        candidates
            .into_iter()
            .map(|(token, prob)| (token as u32, prob))
            .collect(),
    )
}

pub fn sample_from_token_distribution(
    distribution: &TokenDistribution,
    rng: &mut Xorshift64,
) -> Option<u32> {
    let total: f32 = distribution.entries.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 || !total.is_finite() {
        return None;
    }
    let threshold = rng.next_f32() * total;
    let mut cumsum = 0.0f32;
    for (token, prob) in distribution.entries.iter().copied() {
        cumsum += prob;
        if cumsum >= threshold {
            return Some(token);
        }
    }
    distribution
        .entries
        .iter()
        .copied()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(token, _)| token)
}

pub fn sample_residual_token_distribution(
    target: &TokenDistribution,
    draft: &TokenDistribution,
    rng: &mut Xorshift64,
) -> Option<u32> {
    let mut residual = BTreeMap::<u32, f32>::new();
    for (token, prob) in target.entries() {
        residual.insert(*token, *prob);
    }
    for (token, prob) in draft.entries() {
        let entry = residual.entry(*token).or_insert(0.0);
        *entry = (*entry - *prob).max(0.0);
    }
    let residual_entries: Vec<(u32, f32)> = residual
        .into_iter()
        .filter(|(_, prob)| *prob > 0.0 && prob.is_finite())
        .collect();
    let distribution = TokenDistribution::new(residual_entries).unwrap_or_else(|| target.clone());
    sample_from_token_distribution(&distribution, rng)
}

/// Compute the full-vocabulary log-probability of `token` under temperature scaling.
///
/// MTP rejection sampling requires `p_draft` and `p_target` to be in the same
/// normalization domain (full-vocab softmax).  Using a top-k/top-p filtered
/// log-prob inflates `p_draft` relative to `p_target`, causing systematic
/// over-rejection.  This function computes the correct full-vocab log-prob
/// regardless of what sampler filters were used to choose the token.
pub fn full_vocab_token_logprob(logits: &[f32], token: u32, temperature: f32) -> f32 {
    let token_idx = token as usize;
    if logits.is_empty() || token_idx >= logits.len() {
        return -30.0;
    }
    if temperature <= 0.0 {
        return if argmax_f32(logits) == token {
            0.0
        } else {
            f32::NEG_INFINITY
        };
    }
    let inv_temp = 1.0 / temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let unnorm = ((logits[token_idx] - max_l) * inv_temp).exp();
    let sum: f32 = logits
        .iter()
        .map(|&l| {
            let p = ((l - max_l) * inv_temp).exp();
            if p.is_finite() { p } else { 0.0 }
        })
        .sum();
    if sum <= 0.0 || !sum.is_finite() {
        return -30.0;
    }
    (unnorm / sum).max(1e-37_f32).ln().max(-30.0)
}

/// GPU-side categorical sampling from logits.
///
/// Returns the sampled token ID. Uses MLX's `random_categorical` which runs
/// entirely on GPU — no logits data transfer to CPU.
///
/// **Constraints:** only valid when:
/// - `temperature > 0.0`
/// - No repetition penalty (`sampling.uses_repetition_penalty() == false`)
/// - No top-k/top-p filtering (`sampling.top_k == 0 && sampling.top_p >= 1.0`)
///
/// When any of these constraints are violated, fall back to `sample_categorical`.
pub fn sample_categorical_gpu(logits: &MlxArray, temperature: f32) -> u32 {
    // Scale logits by 1/temperature on GPU, then sample.
    let inv_temp = 1.0 / temperature;
    let inv_temp_arr = MlxArray::from_f32(inv_temp);
    let scaled = multiply(logits, &inv_temp_arr, None);
    let token_arr = random_categorical(&scaled, None);
    eval_first_u32(&token_arr)
}

fn recent_repetition_tokens(tokens: &[u32], context_size: Option<u32>) -> &[u32] {
    let keep_len = context_size
        .map(|size| size as usize)
        .unwrap_or(tokens.len())
        .min(tokens.len());
    &tokens[tokens.len() - keep_len..]
}

fn logits_with_repetition_penalty_in_place(
    logits: &mut [f32],
    repetition_penalty: f32,
    repetition_tokens: &[u32],
) {
    let mut seen_tokens = HashSet::with_capacity(repetition_tokens.len());
    for &token in repetition_tokens {
        if !seen_tokens.insert(token) {
            continue;
        }
        let Some(logit) = logits.get_mut(token as usize) else {
            continue;
        };
        if *logit < 0.0 {
            *logit *= repetition_penalty;
        } else {
            *logit /= repetition_penalty;
        }
    }
}

fn apply_top_k_top_p(candidates: &mut Vec<(usize, f32)>, top_k: u32, top_p: f32) {
    let filters_enabled = top_k > 0 || top_p < 1.0;
    if !filters_enabled {
        return;
    }

    candidates.sort_by(|(left_idx, left_prob), (right_idx, right_prob)| {
        right_prob
            .partial_cmp(left_prob)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_idx.cmp(right_idx))
    });

    if top_p.is_finite() && top_p > 0.0 && top_p < 1.0 {
        let total_mass: f32 = candidates.iter().map(|(_, p)| *p).sum();
        let cutoff = top_p * total_mass;
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for (_, prob) in candidates.iter() {
            cumulative += *prob;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        candidates.truncate(keep.max(1));
    }

    if top_k > 0 {
        candidates.truncate((top_k as usize).max(1));
    }
}

fn argmax_f32(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_with_zero_temperature_is_argmax() {
        let logits = vec![0.1_f32, 5.0, 1.0, 2.0];
        let mut rng = Xorshift64::new(42);
        assert_eq!(
            sample_categorical(&logits, MlxSamplingParams::greedy(), &[], &mut rng),
            1
        );
    }

    #[test]
    fn categorical_with_high_temperature_samples_nondeterministically() {
        let logits = vec![1.0_f32; 10]; // uniform distribution
        let mut rng = Xorshift64::new(1);
        let mut seen: std::collections::HashSet<u32> = Default::default();
        for _ in 0..200 {
            seen.insert(sample_categorical(
                &logits,
                MlxSamplingParams::new(1.0, 1.0, 0),
                &[],
                &mut rng,
            ));
        }
        // With 200 samples from 10 uniform classes, we expect most to appear.
        assert!(
            seen.len() >= 5,
            "sampling should cover multiple tokens, got {:?}",
            seen
        );
    }

    #[test]
    fn categorical_with_peaked_distribution_samples_mode() {
        // logit[3] >> others — mode should dominate even with temperature=1.0
        let mut logits = vec![0.0_f32; 10];
        logits[3] = 20.0;
        let mut rng = Xorshift64::new(7);
        for _ in 0..50 {
            assert_eq!(
                sample_categorical(&logits, MlxSamplingParams::new(1.0, 1.0, 0), &[], &mut rng),
                3
            );
        }
    }

    #[test]
    fn categorical_top_k_limits_candidates() {
        let logits = vec![1.0_f32; 6];
        let mut rng = Xorshift64::new(11);
        let sampling = MlxSamplingParams::new(1.0, 1.0, 2);

        for _ in 0..100 {
            let tok = sample_categorical(&logits, sampling, &[], &mut rng);
            assert!(tok < 2, "top_k=2 should exclude token {tok}");
        }
    }

    #[test]
    fn categorical_top_p_limits_candidates() {
        let logits = vec![1.0_f32; 6];
        let mut rng = Xorshift64::new(17);
        let sampling = MlxSamplingParams::new(1.0, 0.5, 0);

        for _ in 0..100 {
            let tok = sample_categorical(&logits, sampling, &[], &mut rng);
            assert!(tok < 3, "top_p=0.5 should exclude token {tok}");
        }
    }

    #[test]
    fn indexed_categorical_samples_from_original_token_ids() {
        let logits = vec![0.0_f32, 4.0, 1.0];
        let indices = vec![42_u32, 99, 7];
        let mut rng = Xorshift64::new(23);

        for _ in 0..20 {
            assert_eq!(
                sample_indexed_categorical(
                    &logits,
                    &indices,
                    MlxSamplingParams::new(0.01, 1.0, 0),
                    &mut rng
                ),
                Some(99)
            );
        }
    }

    #[test]
    fn indexed_categorical_applies_top_p_over_candidate_set() {
        let logits = vec![1.0_f32; 4];
        let indices = vec![10_u32, 11, 12, 13];
        let mut rng = Xorshift64::new(29);
        let sampling = MlxSamplingParams::new(1.0, 0.5, 0);

        for _ in 0..100 {
            let tok = sample_indexed_categorical(&logits, &indices, sampling, &mut rng).unwrap();
            assert!(
                tok == 10 || tok == 11,
                "top_p=0.5 should keep first half of equal-prob candidates, got {tok}"
            );
        }
    }

    #[test]
    fn indexed_categorical_with_logprob_applies_top_k() {
        let logits = vec![4.0_f32, 3.0, 2.0, 1.0];
        let indices = vec![10_u32, 11, 12, 13];
        let mut rng = Xorshift64::new(31);
        let sampling = MlxSamplingParams::new(1.0, 1.0, 2);

        for _ in 0..50 {
            let (tok, log_prob) =
                sample_indexed_categorical_with_logprob(&logits, &indices, sampling, &mut rng)
                    .unwrap();
            assert!(tok == 10 || tok == 11, "top_k=2 should exclude token {tok}");
            assert!(log_prob.is_finite());
        }
    }

    #[test]
    fn categorical_applies_top_p_before_top_k() {
        let logits = vec![0.6_f32.ln(), 0.2_f32.ln(), 0.1_f32.ln(), 0.1_f32.ln()];
        let distribution =
            token_distribution(&logits, MlxSamplingParams::new(1.0, 0.7, 2)).expect("distribution");

        assert!(distribution.probability(0) > 0.0);
        assert!(distribution.probability(1) > 0.0);
        assert_eq!(distribution.probability(2), 0.0);
        assert_eq!(distribution.probability(3), 0.0);
    }

    #[test]
    fn indexed_token_logprob_uses_filtered_distribution() {
        let logits = vec![1.0_f32; 4];
        let indices = vec![10_u32, 11, 12, 13];
        let sampling = MlxSamplingParams::new(1.0, 0.5, 0);

        assert!(
            (indexed_token_logprob(&logits, &indices, 10, sampling).unwrap() - 0.5_f32.ln()).abs()
                < 1e-6
        );
        assert_eq!(indexed_token_logprob(&logits, &indices, 12, sampling), None);
    }

    #[test]
    fn residual_distribution_removes_draft_mass() {
        let target = TokenDistribution::new(vec![(1, 0.6), (2, 0.3), (3, 0.1)]).unwrap();
        let draft = TokenDistribution::new(vec![(1, 0.6), (2, 0.4)]).unwrap();
        let mut rng = Xorshift64::new(41);

        for _ in 0..50 {
            assert_eq!(
                sample_residual_token_distribution(&target, &draft, &mut rng),
                Some(3)
            );
        }
    }

    #[test]
    fn xorshift_produces_distinct_values() {
        let mut rng = Xorshift64::new(1);
        let vals: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        let unique: std::collections::HashSet<u64> = vals.iter().cloned().collect();
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn categorical_falls_back_to_argmax_on_nan_logits() {
        // All-NaN logits produce a non-finite sum; must not panic and must return a valid index.
        let logits = vec![f32::NAN, f32::NAN, f32::NAN];
        let mut rng = Xorshift64::new(99);
        let tok = sample_categorical(&logits, MlxSamplingParams::new(1.0, 1.0, 0), &[], &mut rng);
        assert!((tok as usize) < logits.len());
    }

    #[test]
    fn categorical_falls_back_to_argmax_on_all_neg_inf() {
        // exp(−∞) = 0 for every entry → sum == 0 → fall back to argmax (index 0).
        let logits = vec![f32::NEG_INFINITY; 4];
        let mut rng = Xorshift64::new(3);
        let tok = sample_categorical(&logits, MlxSamplingParams::new(1.0, 1.0, 0), &[], &mut rng);
        assert!((tok as usize) < logits.len());
    }

    #[test]
    fn repetition_penalty_demotes_recent_tokens_before_argmax() {
        let logits = vec![1.0_f32, 1.8, 0.9];
        let mut rng = Xorshift64::new(1);
        let sampling = MlxSamplingParams::greedy().with_repetition_penalty(2.0, None);

        assert_eq!(sample_categorical(&logits, sampling, &[1], &mut rng), 0);
    }

    #[test]
    fn repetition_penalty_respects_context_window() {
        let logits = vec![1.0_f32, 1.8, 0.9];
        let mut rng = Xorshift64::new(1);
        let sampling = MlxSamplingParams::greedy().with_repetition_penalty(2.0, Some(1));

        assert_eq!(sample_categorical(&logits, sampling, &[1, 2], &mut rng), 1);
    }
}
