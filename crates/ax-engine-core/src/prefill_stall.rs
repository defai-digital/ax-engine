//! Time-targeted prefill while decodes run (opt-in, default off).
//!
//! A long prompt that arrives while other requests decode stalls every
//! decoding request for the length of each prefill step. With a stall target
//! set, the engine sizes each text prefill chunk so one step takes about the
//! target, from the measured prefill cost per token (floor
//! [`MIN_PREFILL_CAP_TOKENS`]), and charges half of every such step as decode
//! debt that decode-only steps repay before the next prefill chunk runs.

/// Smallest prefill chunk the policy hands the scheduler.
pub const MIN_PREFILL_CAP_TOKENS: u32 = 256;

/// Smoothing weight of the newest prefill-only step in the cost estimate.
const COST_EWMA_WEIGHT: f64 = 0.3;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrefillStallPolicy {
    target_us: u64,
    us_per_prefill_token: Option<f64>,
    /// Smoothed wall time of a decode-only step, used to price the prefill
    /// part of a mixed step (prefill rarely runs alone under decode load).
    decode_step_us: Option<f64>,
    decode_debt_us: u64,
}

impl PrefillStallPolicy {
    /// `target_ms == 0` disables the policy.
    pub fn new(target_ms: u32) -> Self {
        Self {
            target_us: u64::from(target_ms) * 1_000,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.target_us > 0
    }

    /// Scheduler cap for text prefills in a step that also decodes: `None`
    /// when off, `Some(0)` while decode debt is outstanding.
    pub fn cap(&self) -> Option<u32> {
        if !self.enabled() {
            return None;
        }
        if self.decode_debt_us > 0 {
            return Some(0);
        }
        let tokens = match self.us_per_prefill_token {
            Some(cost) if cost > 0.0 => (self.target_us as f64 / cost).floor(),
            _ => f64::from(MIN_PREFILL_CAP_TOKENS),
        };
        Some((tokens.min(f64::from(u32::MAX)) as u32).max(MIN_PREFILL_CAP_TOKENS))
    }

    /// Account one executed step: its prefill tokens, whether it decoded,
    /// and the runner wall time.
    pub fn record_step(&mut self, prefill_tokens: u32, decoded: bool, runner_time_us: u64) {
        if !self.enabled() {
            return;
        }
        match (decoded, prefill_tokens > 0) {
            (true, true) => {
                self.decode_debt_us = self.decode_debt_us.saturating_add(runner_time_us / 2);
                // The prefill share of a mixed step: its wall time less one
                // decode-only step.
                if let Some(decode_us) = self.decode_step_us {
                    let prefill_us = runner_time_us as f64 - decode_us;
                    if prefill_us > 0.0 {
                        self.observe_prefill_cost(prefill_us / f64::from(prefill_tokens));
                    }
                }
            }
            (true, false) => {
                self.decode_debt_us = self.decode_debt_us.saturating_sub(runner_time_us);
                let step = runner_time_us as f64;
                self.decode_step_us = Some(match self.decode_step_us {
                    Some(previous) => previous + COST_EWMA_WEIGHT * (step - previous),
                    None => step,
                });
            }
            (false, has_prefill) => {
                // Nobody is decoding, so nobody is owed decode time.
                self.decode_debt_us = 0;
                if has_prefill && runner_time_us > 0 {
                    self.observe_prefill_cost(runner_time_us as f64 / f64::from(prefill_tokens));
                }
            }
        }
    }

    fn observe_prefill_cost(&mut self, cost: f64) {
        self.us_per_prefill_token = Some(match self.us_per_prefill_token {
            Some(previous) => previous + COST_EWMA_WEIGHT * (cost - previous),
            None => cost,
        });
    }

    pub fn decode_debt_us(&self) -> u64 {
        self.decode_debt_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_policy_never_caps() {
        let mut policy = PrefillStallPolicy::new(0);
        policy.record_step(4096, false, 1_000_000);
        policy.record_step(512, true, 500_000);
        assert_eq!(policy.cap(), None);
        assert_eq!(policy.decode_debt_us(), 0);
    }

    #[test]
    fn cap_follows_measured_prefill_cost_with_a_floor() {
        let mut policy = PrefillStallPolicy::new(500);
        // Unmeasured: the floor.
        assert_eq!(policy.cap(), Some(MIN_PREFILL_CAP_TOKENS));
        // 2048 tokens in 1 s -> ~488 us/token -> 500 ms is ~1024 tokens.
        policy.record_step(2048, false, 1_000_000);
        assert_eq!(policy.cap(), Some(1024));
        // A very slow model never drops below the floor.
        let mut slow = PrefillStallPolicy::new(500);
        slow.record_step(16, false, 1_000_000);
        assert_eq!(slow.cap(), Some(MIN_PREFILL_CAP_TOKENS));
    }

    #[test]
    fn mixed_steps_accrue_debt_that_decode_steps_repay() {
        let mut policy = PrefillStallPolicy::new(500);
        policy.record_step(2048, false, 1_000_000);
        // A 400 ms mixed step owes 200 ms of decode-only time.
        policy.record_step(1024, true, 400_000);
        assert_eq!(policy.decode_debt_us(), 200_000);
        assert_eq!(policy.cap(), Some(0));
        policy.record_step(0, true, 150_000);
        assert_eq!(policy.cap(), Some(0));
        policy.record_step(0, true, 60_000);
        assert_eq!(policy.decode_debt_us(), 0);
        assert_eq!(policy.cap(), Some(1024));
        // A mixed step is priced net of a decode-only step: (400 - 60) ms for
        // 1024 tokens is about 332 us/token.
        let mut mixed = PrefillStallPolicy::new(500);
        mixed.record_step(0, true, 60_000);
        mixed.record_step(1024, true, 400_000);
        assert_eq!(mixed.us_per_prefill_token.map(f64::round), Some(332.0));
        // Debt vanishes once nothing decodes.
        policy.record_step(1024, true, 400_000);
        policy.record_step(512, false, 250_000);
        assert_eq!(policy.decode_debt_us(), 0);
    }
}
