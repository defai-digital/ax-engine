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

/// Sample one token index from logits with temperature scaling.
///
/// When `temperature` is 0.0 or logits is empty, falls back to argmax.
/// Uses a numerically stable softmax (shift by max before exp).
///
/// Does not require GPU — caller must have already eval'd logits to CPU.
pub fn sample_categorical(logits: &[f32], temperature: f32, rng: &mut Xorshift64) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 {
        return argmax_f32(logits);
    }

    let inv_temp = 1.0 / temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp((logit - max) / temperature) for numerical stability.
    let probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_l) * inv_temp).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return argmax_f32(logits);
    }

    // Normalize and sample.
    let threshold = rng.next_f32() * sum;
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= threshold {
            return i as u32;
        }
    }
    // Rounding fallback: return the highest-probability token.
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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
        assert_eq!(sample_categorical(&logits, 0.0, &mut rng), 1);
    }

    #[test]
    fn categorical_with_high_temperature_samples_nondeterministically() {
        let logits = vec![1.0_f32; 10]; // uniform distribution
        let mut rng = Xorshift64::new(1);
        let mut seen: std::collections::HashSet<u32> = Default::default();
        for _ in 0..200 {
            seen.insert(sample_categorical(&logits, 1.0, &mut rng));
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
            assert_eq!(sample_categorical(&logits, 1.0, &mut rng), 3);
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
        let tok = sample_categorical(&logits, 1.0, &mut rng);
        assert!((tok as usize) < logits.len());
    }

    #[test]
    fn categorical_falls_back_to_argmax_on_all_neg_inf() {
        // exp(−∞) = 0 for every entry → sum == 0 → fall back to argmax (index 0).
        let logits = vec![f32::NEG_INFINITY; 4];
        let mut rng = Xorshift64::new(3);
        let tok = sample_categorical(&logits, 1.0, &mut rng);
        assert!((tok as usize) < logits.len());
    }
}
