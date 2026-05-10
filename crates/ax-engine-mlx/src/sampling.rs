use std::collections::BTreeSet;

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
    let adjusted_logits;
    let logits = if sampling.uses_repetition_penalty() && !repetition_tokens.is_empty() {
        adjusted_logits = logits_with_repetition_penalty(
            logits,
            sampling.repetition_penalty,
            recent_repetition_tokens(repetition_tokens, sampling.repetition_context_size),
        );
        adjusted_logits.as_slice()
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

    apply_top_k_top_p(&mut candidates, sampling.top_k, sampling.top_p, sum);
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

fn recent_repetition_tokens(tokens: &[u32], context_size: Option<u32>) -> &[u32] {
    let keep_len = context_size
        .map(|size| size as usize)
        .unwrap_or(tokens.len())
        .min(tokens.len());
    &tokens[tokens.len() - keep_len..]
}

fn logits_with_repetition_penalty(
    logits: &[f32],
    repetition_penalty: f32,
    repetition_tokens: &[u32],
) -> Vec<f32> {
    let mut adjusted = logits.to_vec();
    let mut seen_tokens = BTreeSet::new();
    for token in repetition_tokens {
        if !seen_tokens.insert(*token) {
            continue;
        }
        let Some(logit) = adjusted.get_mut(*token as usize) else {
            continue;
        };
        if *logit < 0.0 {
            *logit *= repetition_penalty;
        } else {
            *logit /= repetition_penalty;
        }
    }
    adjusted
}

fn apply_top_k_top_p(candidates: &mut Vec<(usize, f32)>, top_k: u32, top_p: f32, total_mass: f32) {
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

    if top_k > 0 {
        candidates.truncate((top_k as usize).max(1));
    }

    if top_p.is_finite() && top_p < 1.0 {
        let cutoff = top_p.max(0.0) * total_mass;
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
