//! Request-local profitability policy for Qwen linear-attention MTP.
//!
//! Acceptance is necessary but not sufficient for speculative decoding to
//! win: a depth-one round can accept every draft and still cost more than two
//! direct target steps.  This module compares complete MTP-round wall time
//! with real single-token target probes and latches direct decode when the
//! measured cost per emitted token is no better than the target baseline.

use std::sync::OnceLock;

const MAX_PROBE_ROUNDS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MtpProfitabilityConfig {
    pub(super) enabled: bool,
    pub(super) probe_rounds: u32,
    pub(super) probe_spacing: u32,
    pub(super) warmup_mtp_rounds: u32,
    pub(super) min_mtp_rounds: u32,
    pub(super) decline_ratio: f64,
}

impl Default for MtpProfitabilityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            probe_rounds: 2,
            probe_spacing: 4,
            warmup_mtp_rounds: 4,
            min_mtp_rounds: 8,
            decline_ratio: 1.0,
        }
    }
}

impl MtpProfitabilityConfig {
    /// Pure depth-one MTP rounds needed before every configured probe and the
    /// post-warmup sample window can be complete.
    const fn required_observation_rounds(self) -> u32 {
        let sampled_rounds = self.warmup_mtp_rounds.saturating_add(self.min_mtp_rounds);
        let probe_rounds = self
            .probe_rounds
            .saturating_sub(1)
            .saturating_mul(self.probe_spacing)
            .saturating_add(1);
        if sampled_rounds > probe_rounds {
            sampled_rounds
        } else {
            probe_rounds
        }
    }

    /// Whether a request can reach a profitability decision before its normal
    /// short-tail MTP cutoff. The prefill output and empty-draft bootstrap use
    /// two tokens before the first measurable round; each depth-one round can
    /// then emit at most two tokens.
    pub(super) const fn has_observation_budget(
        self,
        max_output_tokens: u32,
        min_remaining_tokens: u32,
    ) -> bool {
        let rounds = self.required_observation_rounds();
        let tokens_before_final_round =
            2_u32.saturating_add(rounds.saturating_sub(1).saturating_mul(2));
        let required_tail = if min_remaining_tokens > 0 {
            min_remaining_tokens
        } else {
            1
        };
        max_output_tokens >= tokens_before_final_round.saturating_add(required_tail)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MtpProfitabilityEligibility {
    pub(super) mtp_requested: bool,
    pub(super) exact_qwen_linear: bool,
    pub(super) has_linear_attention: bool,
    pub(super) has_qwen_mtp: bool,
    pub(super) depth_one: bool,
    pub(super) dense_lm_head: bool,
    pub(super) greedy: bool,
    pub(super) skip_state_disabled: bool,
    pub(super) optimistic_disabled: bool,
    /// `AX_MLX_MTP_BYPASS_THRESHOLD=0` is the existing force-MTP contract.
    pub(super) automatic_bypass_allowed: bool,
    /// Calibration must be able to finish before the short-tail MTP cutoff.
    pub(super) output_budget_sufficient: bool,
}

impl MtpProfitabilityEligibility {
    pub(super) const fn eligible(self) -> bool {
        self.mtp_requested
            && self.exact_qwen_linear
            && self.has_linear_attention
            && self.has_qwen_mtp
            && self.depth_one
            && self.dense_lm_head
            && self.greedy
            && self.skip_state_disabled
            && self.optimistic_disabled
            && self.automatic_bypass_allowed
            && self.output_budget_sufficient
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MtpProfitabilitySnapshot {
    pub(super) eligible: bool,
    pub(super) probe_steps: u32,
    pub(super) probe_wall_us: u32,
    pub(super) direct_reference_wall_us: u32,
    pub(super) mtp_rounds_seen: u32,
    pub(super) mtp_warmup_rounds: u32,
    pub(super) mtp_rounds: u32,
    pub(super) mtp_round_wall_us: u32,
    pub(super) mtp_emitted_tokens: u32,
    pub(super) baseline_equivalent_wall_us: u32,
    pub(super) estimated_speedup_x1000: u32,
    pub(super) bypassed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MtpProfitabilityState {
    config: MtpProfitabilityConfig,
    eligible: bool,
    probe_wall_us: [u32; MAX_PROBE_ROUNDS as usize],
    probe_steps: u32,
    mtp_rounds_seen: u32,
    mtp_rounds: u32,
    mtp_round_wall_us: u64,
    mtp_emitted_tokens: u64,
    bypassed: bool,
}

impl Default for MtpProfitabilityState {
    fn default() -> Self {
        Self {
            config: MtpProfitabilityConfig::default(),
            eligible: false,
            probe_wall_us: [0; MAX_PROBE_ROUNDS as usize],
            probe_steps: 0,
            mtp_rounds_seen: 0,
            mtp_rounds: 0,
            mtp_round_wall_us: 0,
            mtp_emitted_tokens: 0,
            bypassed: false,
        }
    }
}

impl MtpProfitabilityState {
    pub(super) fn reset(&mut self, eligible: bool, config: MtpProfitabilityConfig) {
        *self = Self {
            config,
            eligible: eligible && config.enabled,
            ..Self::default()
        };
    }

    /// Whether a cloned-cache direct target probe should run before this MTP
    /// round. The second probe is deliberately separated by several MTP
    /// rounds so the reference is not entirely a cold-start measurement.
    pub(super) fn should_probe_now(&self) -> bool {
        if !self.eligible || self.bypassed || self.probe_steps >= self.config.probe_rounds {
            return false;
        }
        self.probe_steps == 0
            || self.mtp_rounds_seen >= self.probe_steps.saturating_mul(self.config.probe_spacing)
    }

    pub(super) fn record_direct_probe(&mut self, wall_us: u32) {
        if !self.should_probe_now() {
            return;
        }
        let index = self.probe_steps as usize;
        self.probe_wall_us[index] = wall_us.max(1);
        self.probe_steps = self.probe_steps.saturating_add(1);
    }

    /// Record one complete, pure-MTP round. Returns true exactly when the
    /// request should latch onto direct decode.
    pub(super) fn record_mtp_round(&mut self, wall_us: u32, emitted_tokens: usize) -> bool {
        if !self.eligible || self.bypassed || emitted_tokens == 0 {
            return false;
        }
        self.mtp_rounds_seen = self.mtp_rounds_seen.saturating_add(1);
        if self.mtp_rounds_seen <= self.config.warmup_mtp_rounds {
            return false;
        }
        self.mtp_rounds = self.mtp_rounds.saturating_add(1);
        self.mtp_round_wall_us = self
            .mtp_round_wall_us
            .saturating_add(u64::from(wall_us.max(1)));
        self.mtp_emitted_tokens = self
            .mtp_emitted_tokens
            .saturating_add(emitted_tokens as u64);

        if self.probe_steps < self.config.probe_rounds
            || self.mtp_rounds < self.config.min_mtp_rounds
        {
            return false;
        }
        let Some(speedup) = self.estimated_speedup() else {
            return false;
        };
        if speedup <= self.config.decline_ratio {
            self.bypassed = true;
            return true;
        }
        false
    }

    fn direct_reference_wall_us(&self) -> Option<u32> {
        let len = self.probe_steps.min(self.config.probe_rounds) as usize;
        if len == 0 {
            return None;
        }
        let mut samples = self.probe_wall_us;
        samples[..len].sort_unstable();
        // Lower median is conservative with the default two probes: a cold
        // first target step cannot make profitable MTP look slower.
        Some(samples[(len - 1) / 2])
    }

    fn baseline_equivalent_wall_us(&self) -> Option<u64> {
        self.direct_reference_wall_us()
            .map(|reference| u64::from(reference).saturating_mul(self.mtp_emitted_tokens))
    }

    fn estimated_speedup(&self) -> Option<f64> {
        let baseline = self.baseline_equivalent_wall_us()?;
        (self.mtp_round_wall_us > 0).then_some(baseline as f64 / self.mtp_round_wall_us as f64)
    }

    pub(super) fn snapshot(&self) -> MtpProfitabilitySnapshot {
        let probe_wall_us = self.probe_wall_us[..self.probe_steps as usize]
            .iter()
            .fold(0_u32, |sum, value| sum.saturating_add(*value));
        MtpProfitabilitySnapshot {
            eligible: self.eligible,
            probe_steps: self.probe_steps,
            probe_wall_us,
            direct_reference_wall_us: self.direct_reference_wall_us().unwrap_or(0),
            mtp_rounds_seen: self.mtp_rounds_seen,
            mtp_warmup_rounds: self.mtp_rounds_seen.min(self.config.warmup_mtp_rounds),
            mtp_rounds: self.mtp_rounds,
            mtp_round_wall_us: saturating_u32(self.mtp_round_wall_us),
            mtp_emitted_tokens: saturating_u32(self.mtp_emitted_tokens),
            baseline_equivalent_wall_us: saturating_u32(
                self.baseline_equivalent_wall_us().unwrap_or(0),
            ),
            estimated_speedup_x1000: self
                .estimated_speedup()
                .map(|ratio| (ratio * 1000.0).round().clamp(0.0, u32::MAX as f64) as u32)
                .unwrap_or(0),
            bypassed: self.bypassed,
        }
    }
}

fn saturating_u32(value: u64) -> u32 {
    value.min(u64::from(u32::MAX)) as u32
}

fn parse_bool_default_on(raw: Option<String>) -> bool {
    raw.map(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
    .unwrap_or(true)
}

fn env_u32(name: &str, default: u32, min: u32, max: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn env_f64(name: &str, default: f64, min: f64, max: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
        .clamp(min, max)
}

/// Process-wide policy configuration. `AX_MLX_MTP_PROFITABILITY_GATE=0`
/// preserves forced-MTP A/B runs without introducing probe steps.
pub(super) fn mtp_profitability_config_from_env() -> MtpProfitabilityConfig {
    static CACHED: OnceLock<MtpProfitabilityConfig> = OnceLock::new();
    *CACHED.get_or_init(|| MtpProfitabilityConfig {
        enabled: parse_bool_default_on(std::env::var("AX_MLX_MTP_PROFITABILITY_GATE").ok()),
        probe_rounds: env_u32(
            "AX_MLX_MTP_PROFITABILITY_PROBE_ROUNDS",
            2,
            1,
            MAX_PROBE_ROUNDS,
        ),
        probe_spacing: env_u32("AX_MLX_MTP_PROFITABILITY_PROBE_SPACING", 4, 1, 64),
        warmup_mtp_rounds: env_u32("AX_MLX_MTP_PROFITABILITY_WARMUP_ROUNDS", 4, 0, 64),
        min_mtp_rounds: env_u32("AX_MLX_MTP_PROFITABILITY_MIN_ROUNDS", 8, 1, 256),
        decline_ratio: env_f64("AX_MLX_MTP_PROFITABILITY_DECLINE_RATIO", 1.0, 0.5, 2.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MtpProfitabilityConfig {
        MtpProfitabilityConfig::default()
    }

    #[test]
    fn profile_requires_every_exact_greedy_dense_depth_one_guard() {
        let eligible = MtpProfitabilityEligibility {
            mtp_requested: true,
            exact_qwen_linear: true,
            has_linear_attention: true,
            has_qwen_mtp: true,
            depth_one: true,
            dense_lm_head: true,
            greedy: true,
            skip_state_disabled: true,
            optimistic_disabled: true,
            automatic_bypass_allowed: true,
            output_budget_sufficient: true,
        };
        assert!(eligible.eligible());
        assert!(
            !MtpProfitabilityEligibility {
                mtp_requested: false,
                ..eligible
            }
            .eligible()
        );
        assert!(
            !MtpProfitabilityEligibility {
                greedy: false,
                ..eligible
            }
            .eligible()
        );
        assert!(
            !MtpProfitabilityEligibility {
                dense_lm_head: false,
                ..eligible
            }
            .eligible()
        );
        assert!(
            !MtpProfitabilityEligibility {
                automatic_bypass_allowed: false,
                ..eligible
            }
            .eligible()
        );
        assert!(
            !MtpProfitabilityEligibility {
                output_budget_sufficient: false,
                ..eligible
            }
            .eligible()
        );
    }

    #[test]
    fn observation_budget_accounts_for_bootstrap_rounds_and_tail_cutoff() {
        let cfg = config();
        assert_eq!(cfg.required_observation_rounds(), 12);
        assert!(!cfg.has_observation_budget(39, 16));
        assert!(cfg.has_observation_budget(40, 16));
        assert!(!cfg.has_observation_budget(24, 0));
        assert!(cfg.has_observation_budget(25, 0));

        let spaced = MtpProfitabilityConfig {
            probe_rounds: 4,
            probe_spacing: 8,
            warmup_mtp_rounds: 0,
            min_mtp_rounds: 1,
            ..cfg
        };
        assert_eq!(spaced.required_observation_rounds(), 25);
    }

    #[test]
    fn probes_are_interleaved_with_real_mtp_rounds() {
        let mut state = MtpProfitabilityState::default();
        state.reset(true, config());
        assert!(state.should_probe_now());
        state.record_direct_probe(30_000);
        assert!(!state.should_probe_now());
        for _ in 0..3 {
            assert!(!state.record_mtp_round(55_000, 2));
            assert!(!state.should_probe_now());
        }
        assert!(!state.record_mtp_round(55_000, 2));
        assert!(state.should_probe_now());
        state.record_direct_probe(29_000);
        assert!(!state.should_probe_now());
        assert_eq!(state.snapshot().direct_reference_wall_us, 29_000);
    }

    #[test]
    fn slower_cost_per_token_latches_direct_after_warmup() {
        let mut state = MtpProfitabilityState::default();
        state.reset(true, config());
        state.record_direct_probe(30_000);
        for _ in 0..4 {
            assert!(!state.record_mtp_round(500_000, 2));
        }
        state.record_direct_probe(30_000);
        for round in 0..8 {
            let bypassed = state.record_mtp_round(62_500, 2);
            assert_eq!(bypassed, round == 7);
        }
        let snapshot = state.snapshot();
        assert!(snapshot.bypassed);
        assert_eq!(snapshot.mtp_rounds_seen, 12);
        assert_eq!(snapshot.mtp_warmup_rounds, 4);
        assert_eq!(snapshot.mtp_rounds, 8);
        assert_eq!(snapshot.mtp_round_wall_us, 500_000);
        assert_eq!(snapshot.estimated_speedup_x1000, 960);
    }

    #[test]
    fn profitable_profile_stays_on() {
        let mut state = MtpProfitabilityState::default();
        state.reset(true, config());
        state.record_direct_probe(30_000);
        for _ in 0..4 {
            assert!(!state.record_mtp_round(500_000, 2));
        }
        state.record_direct_probe(30_000);
        for _ in 0..8 {
            assert!(!state.record_mtp_round(52_356, 2));
        }
        let snapshot = state.snapshot();
        assert!(!snapshot.bypassed);
        assert_eq!(snapshot.mtp_rounds_seen, 12);
        assert_eq!(snapshot.mtp_warmup_rounds, 4);
        assert_eq!(snapshot.mtp_rounds, 8);
        assert_eq!(snapshot.mtp_round_wall_us, 418_848);
        assert_eq!(snapshot.estimated_speedup_x1000, 1_146);
    }

    #[test]
    fn disabled_policy_never_probes_or_bypasses() {
        let mut state = MtpProfitabilityState::default();
        state.reset(
            true,
            MtpProfitabilityConfig {
                enabled: false,
                ..config()
            },
        );
        assert!(!state.should_probe_now());
        for _ in 0..16 {
            assert!(!state.record_mtp_round(100_000, 1));
        }
        assert_eq!(state.snapshot(), MtpProfitabilitySnapshot::default());
    }
}
