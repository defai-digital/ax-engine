//! Measured cost-model draft-depth controller for the Qwen linear MTP path.
//!
//! Supersedes the streak controller (`mtp_next_adaptive_depth`) when
//! `AX_MLX_MTP_COST_MODEL_DEPTH` is on: every decision picks the draft depth
//! with the best measured expected accepted tokens per cycle, compares it
//! against a measured direct single-token baseline from the existing probe,
//! and parks speculation for the rest of the request when every speculative
//! depth loses to direct decode. It only chooses draft depth and routes to
//! direct; the verifier still decides every emitted token.
//!
//! # Deferred considerations
//!
//! Reviewed and deliberately not adopted while the controller ships opt-in
//! (default OFF) pending hardware A/B evidence:
//! - Dropping the park latch: a losing request would keep paying the verify
//!   forward for the whole generation; the latch is the measured escape.
//! - In-band direct-window baseline: measuring the direct step inside the
//!   decode loop changes scheduling for every request, opt-in or not.
//! - Linear `t(d) = a + b*d` cost model: the measured per-depth EMAs plus a
//!   marginal slope fit sparse, non-monotone measurements better.
//! - Scalar geometric acceptance: per-position EMAs show the real cascade
//!   shape (position 1..d are not one scalar), and the borrow rule already
//!   covers unobserved positions.
//! - Removing staleness probing: without re-probes the EMAs freeze on an
//!   early context and the depth decision stops tracking the request.

use super::*;
use std::sync::OnceLock;
use std::time::Instant;

/// Per-depth slots. The widest configurable throughput draft depth is seven
/// (`AX_MLX_QWEN_LINEAR_THROUGHPUT_MTP_DEPTH`).
pub(super) const MTP_COST_MAX_DEPTH: usize = 7;

/// Weight a cycle-wall sample above twice the current EMA moves at.
const SPIKE_DAMPING_WEIGHT: f32 = 0.25;

/// Exact cumulative mean for a draft position's first samples, then EMA.
const POSITION_CUMULATIVE_SAMPLES: u16 = 4;

/// Direct single-token probes retained for the warmup-context reference.
const DIRECT_PROBE_SLOTS: usize = 2;

/// Steady-state cycles that must pass before a direct probe may arm park.
///
/// Warmup-context probes measure an artificially cheap direct step (short KV
/// clone, cold pipeline), so the park baseline needs a probe taken once the
/// request has settled. Warmup probes still feed the display/scoring reference.
pub(super) const PARK_BASELINE_PROBE_DELAY_CYCLES: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MtpCostDepthConfig {
    /// `AX_MLX_MTP_COST_MODEL_DEPTH` — v1 opt-in, off by default.
    pub(super) enabled: bool,
    /// `AX_MLX_MTP_BYPASS_THRESHOLD > 0`: the force-MTP harness contract
    /// (`=0`) freezes the controller at the width, skips probes, and never
    /// parks.
    pub(super) automatic_bypass_allowed: bool,
    pub(super) accept_alpha: f32,
    pub(super) time_tau_ms: f32,
    /// Depth-switch hysteresis. Apple Silicon wall-clock noise exceeds the
    /// original 3% margin, so the default is 10%.
    pub(super) hysteresis: f32,
    pub(super) probe_period_cycles: u32,
    pub(super) probe_len: u32,
    pub(super) probe_duty: f32,
    pub(super) probe_margin: f32,
    pub(super) park_margin: f32,
    pub(super) park_streak: u32,
    pub(super) probe_rounds: u32,
}

impl Default for MtpCostDepthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            automatic_bypass_allowed: true,
            accept_alpha: 0.05,
            time_tau_ms: 400.0,
            hysteresis: 1.10,
            probe_period_cycles: 64,
            probe_len: 4,
            probe_duty: 0.15,
            probe_margin: 1.15,
            park_margin: 1.0,
            park_streak: 16,
            probe_rounds: 2,
        }
    }
}

/// Raw `AX_MLX_MTP_COST_*` values (`None` = unset) for
/// [`MtpCostDepthConfig::config_from`].
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MtpCostDepthEnvRaw<'a> {
    pub(super) accept_alpha: Option<&'a str>,
    pub(super) time_tau_ms: Option<&'a str>,
    pub(super) hysteresis: Option<&'a str>,
    pub(super) probe_period_cycles: Option<&'a str>,
    pub(super) probe_len: Option<&'a str>,
    pub(super) probe_duty: Option<&'a str>,
    pub(super) probe_margin: Option<&'a str>,
    pub(super) park_margin: Option<&'a str>,
    pub(super) park_streak: Option<&'a str>,
    pub(super) probe_rounds: Option<&'a str>,
}

impl MtpCostDepthConfig {
    /// Pure constructor: an unset or unparseable value keeps the documented
    /// default, a parsed value is clamped into its documented range. The
    /// documented open lower bounds (`accept_alpha`, `probe_duty`) clamp to
    /// their nearest inclusive value.
    pub(super) fn config_from(
        enabled: bool,
        automatic_bypass_allowed: bool,
        raw: MtpCostDepthEnvRaw<'_>,
    ) -> Self {
        let defaults = Self::default();
        Self {
            enabled,
            automatic_bypass_allowed,
            accept_alpha: parse_f32(raw.accept_alpha, defaults.accept_alpha, 0.001, 0.5),
            time_tau_ms: parse_f32(raw.time_tau_ms, defaults.time_tau_ms, 50.0, 10_000.0),
            hysteresis: parse_f32(raw.hysteresis, defaults.hysteresis, 1.0, 1.5),
            probe_period_cycles: parse_u32(
                raw.probe_period_cycles,
                defaults.probe_period_cycles,
                8,
                4096,
            ),
            probe_len: parse_u32(raw.probe_len, defaults.probe_len, 1, 16),
            probe_duty: parse_f32(raw.probe_duty, defaults.probe_duty, 0.0, 0.5),
            probe_margin: parse_f32(raw.probe_margin, defaults.probe_margin, 1.0, 2.0),
            park_margin: parse_f32(raw.park_margin, defaults.park_margin, 0.8, 2.0),
            park_streak: parse_u32(raw.park_streak, defaults.park_streak, 2, 1024),
            probe_rounds: parse_u32(raw.probe_rounds, defaults.probe_rounds, 1, 4),
        }
    }
}

fn parse_f32(raw: Option<&str>, default: f32, min: f32, max: f32) -> f32 {
    raw.and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn parse_u32(raw: Option<&str>, default: u32, min: u32, max: u32) -> u32 {
    raw.and_then(|value| value.trim().parse::<u32>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

/// Process-wide controller configuration (mirrors the profitability gate's
/// cached reader). `AX_MLX_MTP_COST_MODEL_DEPTH` is the v1 opt-in.
pub(super) fn mtp_cost_depth_config_from_env() -> MtpCostDepthConfig {
    static CACHED: OnceLock<MtpCostDepthConfig> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let accept_alpha = std::env::var("AX_MLX_MTP_COST_ACCEPT_ALPHA").ok();
        let time_tau_ms = std::env::var("AX_MLX_MTP_COST_TIME_TAU_MS").ok();
        let hysteresis = std::env::var("AX_MLX_MTP_COST_HYSTERESIS").ok();
        let probe_period_cycles = std::env::var("AX_MLX_MTP_COST_PROBE_PERIOD_CYCLES").ok();
        let probe_len = std::env::var("AX_MLX_MTP_COST_PROBE_LEN").ok();
        let probe_duty = std::env::var("AX_MLX_MTP_COST_PROBE_DUTY").ok();
        let probe_margin = std::env::var("AX_MLX_MTP_COST_PROBE_MARGIN").ok();
        let park_margin = std::env::var("AX_MLX_MTP_COST_PARK_MARGIN").ok();
        let park_streak = std::env::var("AX_MLX_MTP_COST_PARK_STREAK").ok();
        let probe_rounds = std::env::var("AX_MLX_MTP_COST_PROBE_ROUNDS").ok();
        MtpCostDepthConfig::config_from(
            crate::fastpath::mtp_cost_model_depth_enabled(),
            mtp_bypass_threshold() > 0.0,
            MtpCostDepthEnvRaw {
                accept_alpha: accept_alpha.as_deref(),
                time_tau_ms: time_tau_ms.as_deref(),
                hysteresis: hysteresis.as_deref(),
                probe_period_cycles: probe_period_cycles.as_deref(),
                probe_len: probe_len.as_deref(),
                probe_duty: probe_duty.as_deref(),
                probe_margin: probe_margin.as_deref(),
                park_margin: park_margin.as_deref(),
                park_streak: park_streak.as_deref(),
                probe_rounds: probe_rounds.as_deref(),
            },
        )
    })
}

/// Effective controller width: the verify window the head actually serves,
/// capped by the policy's maximum depth.
pub(super) const fn mtp_cost_depth_width(verify_drafts: usize, max_depth: usize) -> usize {
    if verify_drafts < max_depth {
        verify_drafts
    } else {
        max_depth
    }
}

/// Per-request snapshot for route telemetry. Per-depth entries are indexed by
/// depth - 1; `0` means "not measured yet".
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct MtpCostDepthSnapshot {
    pub(super) width: u32,
    pub(super) depth_current: u32,
    pub(super) park_events: u32,
    pub(super) probe_steps: u32,
    pub(super) direct_reference_wall_us: u32,
    pub(super) t_depth_us: [u32; MTP_COST_MAX_DEPTH],
    pub(super) p_depth_x1000: [u32; MTP_COST_MAX_DEPTH],
}

/// Request-local cost-model depth controller. Small and `Copy` so the async
/// committed fold can preview its next decision without disturbing the live
/// controller (the fold drafts before `record_step` runs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MtpCostDepthController {
    config: MtpCostDepthConfig,
    width: usize,
    /// Conditional acceptance per draft position (EMA of 1.0/0.0 samples).
    p_accept: [Option<f32>; MTP_COST_MAX_DEPTH],
    p_samples: [u16; MTP_COST_MAX_DEPTH],
    /// Completed-cycle wall per used depth.
    t_cycle_us: [Option<f32>; MTP_COST_MAX_DEPTH],
    t_age_cycles: [u32; MTP_COST_MAX_DEPTH],
    direct_probe_us: [u32; DIRECT_PROBE_SLOTS],
    direct_probes: u32,
    /// Conservative direct reference measured once the request settled; park
    /// only arms against this (warmup-context probes measure a cheap step).
    park_baseline_us: Option<u32>,
    /// A direct probe ran inside the currently open cycle window. The window
    /// is then not a clean depth-cost sample and its decision must not count
    /// toward the park streak.
    cycle_probe: bool,
    current: usize,
    cycle_open: Option<Instant>,
    warmup_remaining: usize,
    probe_depth: usize,
    probe_burst_remaining: u32,
    cycles: u32,
    probe_cycles: u32,
    since_probe_cycles: u32,
    park_streak: u32,
    parked: bool,
}

impl Default for MtpCostDepthController {
    fn default() -> Self {
        Self {
            config: MtpCostDepthConfig::default(),
            width: 0,
            p_accept: [None; MTP_COST_MAX_DEPTH],
            p_samples: [0; MTP_COST_MAX_DEPTH],
            t_cycle_us: [None; MTP_COST_MAX_DEPTH],
            t_age_cycles: [0; MTP_COST_MAX_DEPTH],
            direct_probe_us: [0; DIRECT_PROBE_SLOTS],
            direct_probes: 0,
            park_baseline_us: None,
            cycle_probe: false,
            current: 0,
            cycle_open: None,
            warmup_remaining: 0,
            probe_depth: 0,
            probe_burst_remaining: 0,
            cycles: 0,
            probe_cycles: 0,
            since_probe_cycles: 0,
            park_streak: 0,
            parked: false,
        }
    }
}

impl MtpCostDepthController {
    /// Per-generation reset. The controller is request-local: a park latch
    /// never crosses a generation boundary.
    pub(super) fn reset(&mut self, enabled: bool, config: MtpCostDepthConfig, width: usize) {
        let width = width.min(MTP_COST_MAX_DEPTH);
        *self = Self {
            config: MtpCostDepthConfig {
                enabled: enabled && config.enabled,
                ..config
            },
            width,
            warmup_remaining: width,
            ..Self::default()
        };
    }

    /// Whether this request owns its depth decisions.
    pub(super) fn enabled(&self) -> bool {
        self.config.enabled && self.width >= 1
    }

    /// Set once every speculative depth lost to the measured direct baseline
    /// for `park_streak` consecutive decisions.
    pub(super) fn parked(&self) -> bool {
        self.parked
    }

    /// Depth decided for the round currently in flight. The cost-sample
    /// attribution needs it (and tests drive the same invariant: the round
    /// verifies the draft generated at this depth).
    #[cfg(test)]
    pub(super) fn current_depth(&self) -> usize {
        self.current
    }

    /// Whether the direct reference probe should run before this round.
    ///
    /// Two tiers: the `probe_rounds` warmup-context probes requested right
    /// after the sweep (scoring/display reference), then one settling probe
    /// that arms the park baseline. Never derived from MTP-side estimates.
    pub(super) fn wants_direct_probe(&self) -> bool {
        self.enabled()
            && self.config.automatic_bypass_allowed
            && !self.parked
            && self.warmup_remaining == 0
            && (self.direct_probes < self.config.probe_rounds
                || (self.park_baseline_us.is_none()
                    && self.cycles >= PARK_BASELINE_PROBE_DELAY_CYCLES))
    }

    /// Record one measured direct singleton step.
    ///
    /// The warmup-context probes keep the fastest reference for display and
    /// scoring; a probe taken once the request has settled arms the park
    /// baseline, which speculation must beat for park to latch.
    pub(super) fn record_direct_probe(&mut self, wall_us: u32) {
        let wall_us = wall_us.max(1);
        if self.direct_probes >= self.config.probe_rounds
            && self.cycles >= PARK_BASELINE_PROBE_DELAY_CYCLES
        {
            self.park_baseline_us = Some(
                self.park_baseline_us
                    .map_or(wall_us, |current| current.min(wall_us)),
            );
        }
        let index = self.direct_probes as usize;
        if index < self.direct_probe_us.len() {
            self.direct_probe_us[index] = wall_us;
        }
        self.direct_probes = self.direct_probes.saturating_add(1);
        self.cycle_probe = true;
    }

    /// Close the previous cycle, record its measurements, and return the
    /// draft depth for the next cycle. `0` means the request is parked on
    /// direct decode.
    ///
    /// The decision deliberately reads only measurements recorded by *earlier*
    /// decisions; this cycle's own wall and acceptance sample are recorded
    /// after it. The async committed fold previews the same call on a copy at a
    /// different wall time within the same step, so sharing one input state is
    /// what keeps the preview and the committed decision identical (a
    /// freshly-sampled wall differs between the two call sites by a few
    /// microseconds and can flip a score near-tie).
    ///
    /// `wall_valid` is the runner's "exactly one decode step ran since the
    /// previous decision" signal: a fallback, think-window, single-decode, or
    /// n-gram step in between makes the closed window span foreign work.
    pub(super) fn observe_and_decide(
        &mut self,
        used: usize,
        accepted: usize,
        pure_mtp_round: bool,
        wall_valid: bool,
    ) -> usize {
        // Window conditions are read from the state the previous decision left
        // behind: the draft verified in the closing window was generated at
        // `current`, so a verified length that differs from it (draft gating or
        // a depth change) is not a clean sample of either depth.
        let window_depth = self.current;
        let warmup_cycle = self.warmup_remaining > 0;
        let probe_cycle = self.cycle_probe;
        let homogeneous = wall_valid && !probe_cycle;
        let steady_depth_switch = !warmup_cycle && used != window_depth;
        let depth = self.decide_depth(homogeneous);
        self.record_measurements(
            used,
            accepted,
            pure_mtp_round,
            homogeneous,
            steady_depth_switch,
        );
        depth
    }

    fn decide_depth(&mut self, homogeneous: bool) -> usize {
        if !self.enabled() || self.parked {
            return 0;
        }
        if !self.config.automatic_bypass_allowed {
            // Force-MTP harness contract (`AX_MLX_MTP_BYPASS_THRESHOLD=0`):
            // freeze at the widest depth, never probe, never park. MTP-P
            // evidence must bind to a fixed mechanism, not to a controller
            // trajectory over depths.
            self.current = self.width;
            return self.width;
        }
        if self.update_park_streak(homogeneous) {
            return 0;
        }
        if self.warmup_remaining > 0 {
            let depth = self.warmup_remaining.min(self.width);
            self.warmup_remaining -= 1;
            self.current = depth;
            return depth;
        }
        if self.probe_burst_remaining > 0 {
            self.probe_cycles = self.probe_cycles.saturating_add(1);
            self.probe_burst_remaining -= 1;
            self.current = if self.probe_burst_remaining == 0 {
                self.best_with_hysteresis()
            } else {
                self.probe_depth
            };
            return self.current;
        }
        let candidate = self.best_with_hysteresis();
        if self.staleness_probe_due() {
            self.probe_depth = self.staleness_probe_target(candidate);
            self.probe_burst_remaining = self.config.probe_len.max(1);
            self.since_probe_cycles = 0;
            self.current = self.probe_depth;
            return self.probe_depth;
        }
        self.current = candidate;
        candidate
    }

    /// Route-telemetry snapshot of the controller's live estimates.
    pub(super) fn snapshot(&self) -> MtpCostDepthSnapshot {
        let width = self.width.min(MTP_COST_MAX_DEPTH);
        let mut t_depth_us = [0u32; MTP_COST_MAX_DEPTH];
        let mut p_depth_x1000 = [0u32; MTP_COST_MAX_DEPTH];
        for depth in 1..=width {
            t_depth_us[depth - 1] = self.t_cycle_us[depth - 1]
                .map(|cost| cost.round().clamp(0.0, u32::MAX as f32) as u32)
                .unwrap_or(0);
            p_depth_x1000[depth - 1] = self.p_accept[depth - 1]
                .map(|p| (p.clamp(0.0, 1.0) * 1000.0).round() as u32)
                .unwrap_or(0);
        }
        MtpCostDepthSnapshot {
            width: width as u32,
            depth_current: if self.parked { 0 } else { self.current as u32 },
            park_events: u32::from(self.parked),
            probe_steps: self.direct_probes,
            direct_reference_wall_us: self.direct_reference_us().unwrap_or(0),
            t_depth_us,
            p_depth_x1000,
        }
    }

    fn warmup_done(&self) -> bool {
        self.warmup_remaining == 0
    }

    fn record_measurements(
        &mut self,
        used: usize,
        accepted: usize,
        pure_mtp_round: bool,
        homogeneous: bool,
        steady_depth_switch: bool,
    ) {
        // The probe flag describes the window that just closed; clear it
        // whether or not that window qualified as a cost sample.
        self.cycle_probe = false;
        for depth in 0..self.width {
            self.t_age_cycles[depth] = self.t_age_cycles[depth].saturating_add(1);
        }
        self.cycles = self.cycles.saturating_add(1);
        self.since_probe_cycles = self.since_probe_cycles.saturating_add(1);
        if let Some(started) = self.cycle_open {
            let wall_us = elapsed_us(started).max(1);
            // Only a homogeneous window is a depth-cost sample. A round with
            // no draft has no depth to price; a foreign step (or a direct
            // probe, whose own span cannot be subtracted exactly — the probe's
            // cache clone is untimed) spans work that is not this depth's;
            // and a steady-state draft generated at a different depth than the
            // round verified mixes two depths. Warmup keeps such samples: one
            // noisy sample per sweep depth is acceptable for the initial
            // ranking, and the sweep is the only place every depth is visited.
            if used >= 1 && used <= self.width && homogeneous && !steady_depth_switch {
                self.update_cycle_wall(used, wall_us);
            }
        }
        self.cycle_open = Some(Instant::now());
        if pure_mtp_round {
            // Acceptance is measured entirely inside the round (the verifier's
            // own prefix decision) and stays valid even when the window between
            // decisions is not: contamination is a wall property.
            self.record_acceptance(used, accepted);
        }
    }

    /// One acceptance observation per position up to the first rejection:
    /// positions past it were cascade-rejected by that same window and carry
    /// no evidence about the MTP head's own yield.
    fn record_acceptance(&mut self, used: usize, accepted: usize) {
        for position in 0..used.min(self.width) {
            let observation = if position < accepted { 1.0 } else { 0.0 };
            self.update_position_acceptance(position, observation);
            if position >= accepted {
                break;
            }
        }
    }

    fn update_position_acceptance(&mut self, position: usize, observation: f32) {
        let samples = self.p_samples[position];
        let value = match self.p_accept[position] {
            None => observation,
            Some(current) if samples < POSITION_CUMULATIVE_SAMPLES => {
                (current * f32::from(samples) + observation) / f32::from(samples.saturating_add(1))
            }
            Some(current) => {
                (1.0 - self.config.accept_alpha) * current + self.config.accept_alpha * observation
            }
        };
        self.p_accept[position] = Some(value.clamp(0.0, 1.0));
        self.p_samples[position] = samples.saturating_add(1);
    }

    /// Wall-horizon EMA: the update weight follows the sample's own wall, so
    /// long noisy cycles move faster than short ones. A sample above twice the
    /// current EMA is damped to the spike weight so one scheduling outlier
    /// cannot reprice a depth.
    fn update_cycle_wall(&mut self, depth: usize, wall_us: u32) {
        let sample = wall_us as f32;
        let index = depth - 1;
        let Some(current) = self.t_cycle_us[index] else {
            self.t_cycle_us[index] = Some(sample);
            self.t_age_cycles[index] = 0;
            return;
        };
        let tau_us = self.config.time_tau_ms.max(1.0) * 1000.0;
        let mut alpha = 1.0 - (-sample / tau_us).exp();
        if sample > 2.0 * current {
            alpha = alpha.min(SPIKE_DAMPING_WEIGHT);
        }
        let alpha = alpha.clamp(0.0, 1.0);
        self.t_cycle_us[index] = Some((1.0 - alpha) * current + alpha * sample);
        self.t_age_cycles[index] = 0;
    }

    /// Effective conditional acceptance at one draft position. Deeper
    /// positions with no observations borrow the nearest observed shallower
    /// position.
    fn effective_p_accept(&self, position: usize) -> Option<f32> {
        for candidate in (0..=position.min(MTP_COST_MAX_DEPTH - 1)).rev() {
            if let Some(value) = self.p_accept[candidate] {
                return Some(value);
            }
        }
        None
    }

    /// Expected tokens emitted per cycle at depth `d`: the committed tail
    /// token plus every draft position that survives the accept cascade.
    fn expected_tokens(&self, depth: usize) -> Option<f32> {
        let mut survival = self.effective_p_accept(0)?;
        let mut total = 1.0 + survival;
        for position in 1..depth {
            survival *= self.effective_p_accept(position)?;
            total += survival;
        }
        Some(total)
    }

    /// Cycle cost at depth `d`: the measured EMA, else the nearest measured
    /// depth plus the measured marginal slope between the two measured depths
    /// closest to `d`. No measurement at all leaves the cost unknown.
    fn estimated_cycle_us(&self, depth: usize) -> Option<f32> {
        if depth == 0 || depth > self.width {
            return None;
        }
        if let Some(measured) = self.t_cycle_us[depth - 1] {
            return Some(measured);
        }
        let mut measured = [(0usize, 0.0_f32); MTP_COST_MAX_DEPTH];
        let mut count = 0usize;
        for candidate in 1..=self.width {
            if let Some(cost) = self.t_cycle_us[candidate - 1] {
                measured[count] = (candidate, cost);
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let mut nearest = measured[0].0;
        for (candidate, _) in &measured[1..count] {
            let distance = candidate.abs_diff(depth);
            let nearest_distance = nearest.abs_diff(depth);
            if distance < nearest_distance || (distance == nearest_distance && *candidate < nearest)
            {
                nearest = *candidate;
            }
        }
        let mut slope = 0.0_f32;
        if count >= 2 {
            let mut pair = (0usize, 1usize);
            let mut best_distance = usize::MAX;
            for index in 0..count - 1 {
                let (lower, _) = measured[index];
                let (upper, _) = measured[index + 1];
                let distance = if depth < lower {
                    lower - depth
                } else {
                    depth.saturating_sub(upper)
                };
                if distance < best_distance {
                    best_distance = distance;
                    pair = (index, index + 1);
                }
            }
            let (lower_depth, lower_cost) = measured[pair.0];
            let (upper_depth, upper_cost) = measured[pair.1];
            slope = (upper_cost - lower_cost) / (upper_depth - lower_depth) as f32;
        }
        let base = self.t_cycle_us[nearest - 1]?;
        Some((base + slope * (depth as f32 - nearest as f32)).max(1.0))
    }

    /// Expected accepted tokens per cycle at this depth. `None` when the depth
    /// has neither acceptance evidence nor a cost estimate.
    fn score(&self, depth: usize) -> Option<f32> {
        let expected = self.expected_tokens(depth)?;
        let cost = self.estimated_cycle_us(depth)?;
        (cost > 0.0).then_some(expected / cost)
    }

    fn best_scored(&self) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for depth in 1..=self.width {
            let Some(score) = self.score(depth) else {
                continue;
            };
            if best.is_none_or(|(_, best_score)| score > best_score) {
                best = Some((depth, score));
            }
        }
        best
    }

    fn best_scoring_depth(&self) -> Option<usize> {
        self.best_scored().map(|(depth, _)| depth)
    }

    /// Switch away from `current` only when the challenger clears the
    /// configured hysteresis margin; ties resolve shallower.
    fn best_with_hysteresis(&self) -> usize {
        let Some(challenger) = self.best_scoring_depth() else {
            return self.current.clamp(1, self.width.max(1));
        };
        if challenger == self.current {
            return challenger;
        }
        let current_score = if self.current >= 1 && self.current <= self.width {
            self.score(self.current)
        } else {
            None
        };
        match (self.score(challenger), current_score) {
            (Some(challenger_score), Some(current_score))
                if challenger_score <= current_score * self.config.hysteresis =>
            {
                self.current
            }
            _ => challenger,
        }
    }

    fn staleness_probe_due(&self) -> bool {
        self.since_probe_cycles >= self.config.probe_period_cycles
            && (self.probe_cycles as f32) < self.config.probe_duty * (self.cycles as f32)
    }

    /// Probe the strongest rival whose score is within `probe_margin` of the
    /// candidate's, else the most-stale depth (unmeasured depths first, ties
    /// shallower).
    fn staleness_probe_target(&self, candidate: usize) -> usize {
        if let Some(candidate_score) = self.score(candidate) {
            let mut best: Option<(usize, f32)> = None;
            for depth in 1..=self.width {
                if depth == candidate {
                    continue;
                }
                let Some(score) = self.score(depth) else {
                    continue;
                };
                if score * self.config.probe_margin < candidate_score {
                    continue;
                }
                if best.is_none_or(|(_, best_score)| score > best_score) {
                    best = Some((depth, score));
                }
            }
            if let Some((depth, _)) = best {
                return depth;
            }
        }
        let mut target = candidate.clamp(1, self.width.max(1));
        let mut target_staleness = 0u32;
        let mut first = true;
        for depth in 1..=self.width {
            let staleness = if self.t_cycle_us[depth - 1].is_none() {
                u32::MAX
            } else {
                self.t_age_cycles[depth - 1]
            };
            if first || staleness > target_staleness {
                target = depth;
                target_staleness = staleness;
                first = false;
            }
        }
        target
    }

    fn direct_reference_us(&self) -> Option<u32> {
        let count = (self.direct_probes as usize).min(DIRECT_PROBE_SLOTS);
        if count == 0 {
            return None;
        }
        self.direct_probe_us[..count].iter().copied().min()
    }

    /// Park bookkeeping.
    ///
    /// Only homogeneous production cycles count: a window that spanned a
    /// foreign step or a direct probe neither extends nor resets the streak.
    /// Park never arms on an estimated baseline, and only post-warmup probes
    /// (see [`PARK_BASELINE_PROBE_DELAY_CYCLES`]) arm the comparison.
    fn update_park_streak(&mut self, homogeneous: bool) -> bool {
        if !homogeneous {
            return false;
        }
        if !self.warmup_done() || self.park_baseline_us.is_none() {
            self.park_streak = 0;
            return false;
        }
        let Some(direct_us) = self.park_baseline_us else {
            self.park_streak = 0;
            return false;
        };
        let direct_score = 1.0 / direct_us as f32;
        let Some((_, best_score)) = self.best_scored() else {
            self.park_streak = 0;
            return false;
        };
        if best_score < direct_score * self.config.park_margin {
            self.park_streak = self.park_streak.saturating_add(1);
        } else {
            self.park_streak = 0;
        }
        if self.park_streak >= self.config.park_streak {
            self.parked = true;
            return true;
        }
        false
    }
}

/// Resolve one cycle's draft depth: the cost controller when it owns the
/// decision, the legacy controller's result otherwise. An explicit fixed
/// draft depth (`AX_MLX_MTP_FIXED_DRAFT_DEPTH`) preempts both.
pub(super) fn mtp_cost_depth_cycle_depth(
    controller: &mut MtpCostDepthController,
    legacy_depth: usize,
    used: usize,
    accepted: usize,
    pure_mtp_round: bool,
    wall_valid: bool,
    fixed_depth: Option<usize>,
) -> usize {
    if fixed_depth.is_none() && controller.enabled() {
        controller.observe_and_decide(used, accepted, pure_mtp_round, wall_valid)
    } else {
        legacy_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MtpCostDepthConfig {
        MtpCostDepthConfig {
            enabled: true,
            ..MtpCostDepthConfig::default()
        }
    }

    fn fresh_controller(width: usize) -> MtpCostDepthController {
        let mut controller = MtpCostDepthController::default();
        controller.reset(true, config(), width);
        controller
    }

    #[test]
    fn position_zero_miss_leaves_deeper_positions_unobserved() {
        let mut controller = fresh_controller(3);
        controller.observe_and_decide(3, 0, true, true);
        assert_eq!(controller.p_accept[0], Some(0.0));
        assert_eq!(controller.p_accept[1], None);
        assert_eq!(controller.p_accept[2], None);
        // Deeper scoring borrows the observed position.
        assert_eq!(controller.effective_p_accept(2), Some(0.0));
        assert_eq!(controller.expected_tokens(3), Some(1.0));

        let mut controller = fresh_controller(3);
        controller.observe_and_decide(3, 2, true, true);
        assert_eq!(controller.p_accept[0], Some(1.0));
        assert_eq!(controller.p_accept[1], Some(1.0));
        assert_eq!(controller.p_accept[2], Some(0.0));
        assert_eq!(controller.p_samples[2], 1);

        let mut controller = fresh_controller(3);
        controller.observe_and_decide(3, 3, true, true);
        assert_eq!(controller.p_accept[0], Some(1.0));
        assert_eq!(controller.p_accept[1], Some(1.0));
        assert_eq!(controller.p_accept[2], Some(1.0));
    }

    #[test]
    fn position_acceptance_uses_cumulative_mean_then_ema() {
        let mut controller = fresh_controller(1);
        for observation in [1.0_f32, 1.0, 0.0] {
            controller.update_position_acceptance(0, observation);
        }
        assert_eq!(controller.p_accept[0], Some(2.0 / 3.0));
        controller.update_position_acceptance(0, 1.0);
        assert_eq!(controller.p_accept[0], Some(0.75));
        controller.update_position_acceptance(0, 0.0);
        let after_ema = controller.p_accept[0];
        assert!((after_ema.unwrap() - 0.7125).abs() < 1e-6);

        // A cold first observation cannot dominate the admission window:
        // three of four samples accept, and the exact mean reads 0.75 where
        // an EMA seeded from the first sample would read ~0.14.
        let mut controller = fresh_controller(1);
        for observation in [0.0_f32, 1.0, 1.0, 1.0] {
            controller.update_position_acceptance(0, observation);
        }
        assert_eq!(controller.p_accept[0], Some(0.75));
    }

    #[test]
    fn expected_tokens_matches_closed_form() {
        let mut controller = fresh_controller(3);
        controller.p_accept = [Some(0.9), Some(0.8), Some(0.7), None, None, None, None];
        let first = controller.expected_tokens(1).unwrap();
        let second = controller.expected_tokens(2).unwrap();
        let third = controller.expected_tokens(3).unwrap();
        assert!((first - 1.9).abs() < 1e-6);
        assert!((second - 2.62).abs() < 1e-5);
        assert!((third - 3.124).abs() < 1e-5);
    }

    #[test]
    fn score_prefers_shallow_when_deep_acceptance_cannot_pay() {
        let mut controller = fresh_controller(3);
        controller.p_accept = [Some(0.9), Some(0.1), Some(0.1), None, None, None, None];
        controller.t_cycle_us = [
            Some(100.0),
            Some(300.0),
            Some(500.0),
            None,
            None,
            None,
            None,
        ];
        assert_eq!(controller.best_scoring_depth(), Some(1));

        let mut controller = fresh_controller(3);
        controller.p_accept = [Some(0.9), Some(0.9), Some(0.9), None, None, None, None];
        controller.t_cycle_us = [
            Some(100.0),
            Some(120.0),
            Some(140.0),
            None,
            None,
            None,
            None,
        ];
        assert_eq!(controller.best_scoring_depth(), Some(3));
    }

    #[test]
    fn hysteresis_keeps_current_until_the_challenger_clears_it() {
        // score(1) = 1.9/100, score(2) = 2.71/140: inside the 1.10 margin.
        let mut controller = fresh_controller(3);
        controller.p_accept = [Some(0.9), Some(0.9), Some(0.9), None, None, None, None];
        controller.t_cycle_us = [
            Some(100.0),
            Some(140.0),
            Some(400.0),
            None,
            None,
            None,
            None,
        ];
        controller.current = 1;
        assert_eq!(controller.best_scoring_depth(), Some(2));
        assert_eq!(controller.best_with_hysteresis(), 1);

        // 2.71/130 is a 9.7% gain: still short of the 10% margin (a 1.03
        // margin would have switched, which is why the default moved).
        controller.t_cycle_us = [
            Some(100.0),
            Some(130.0),
            Some(400.0),
            None,
            None,
            None,
            None,
        ];
        assert_eq!(controller.best_with_hysteresis(), 1);

        // score(2) = 2.71/120 (12.9% better) clears the margin.
        controller.t_cycle_us = [
            Some(100.0),
            Some(120.0),
            Some(400.0),
            None,
            None,
            None,
            None,
        ];
        assert_eq!(controller.best_with_hysteresis(), 2);

        // Equal scores keep the shallower depth.
        let mut controller = fresh_controller(3);
        controller.p_accept = [Some(0.0), None, None, None, None, None, None];
        controller.t_cycle_us = [
            Some(100.0),
            Some(100.0),
            Some(100.0),
            None,
            None,
            None,
            None,
        ];
        assert_eq!(controller.best_scoring_depth(), Some(1));
        controller.current = 2;
        assert_eq!(controller.best_with_hysteresis(), 2);
    }

    #[test]
    fn warmup_sweeps_the_width_then_requests_probes() {
        let mut controller = fresh_controller(3);
        assert_eq!(controller.observe_and_decide(3, 0, true, true), 3);
        assert_eq!(controller.observe_and_decide(3, 0, true, true), 2);
        assert_eq!(controller.observe_and_decide(3, 0, true, true), 1);
        assert!(!controller.parked());
        assert_eq!(controller.park_streak, 0);
        assert!(controller.wants_direct_probe());
        controller.record_direct_probe(1_000);
        controller.record_direct_probe(1_200);
        assert!(!controller.wants_direct_probe());
        assert_eq!(controller.direct_reference_us(), Some(1_000));

        // Terrible scores cannot park while the sweep is still running.
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                park_streak: 2,
                ..config()
            },
            3,
        );
        controller.p_accept = [Some(0.0); MTP_COST_MAX_DEPTH];
        controller.t_cycle_us = [Some(500_000.0); MTP_COST_MAX_DEPTH];
        controller.record_direct_probe(1);
        controller.record_direct_probe(1);
        for expected in [3, 2, 1] {
            assert_eq!(controller.observe_and_decide(3, 0, true, true), expected);
            assert_eq!(controller.park_streak, 0);
            assert!(!controller.parked());
        }
    }

    /// Drive the controller through the warmup sweep, the warmup-context
    /// probes, and the settling delay, then record the probe that arms the
    /// park baseline.
    fn drive_to_armed_park_baseline(controller: &mut MtpCostDepthController) {
        for _ in 0..(PARK_BASELINE_PROBE_DELAY_CYCLES + 8) {
            if controller.wants_direct_probe() {
                controller.record_direct_probe(1_000);
            }
            if controller.park_baseline_us.is_some() {
                // Close the arming probe's own window: that decision is not a
                // park credit (probe cycles never count), so the streak starts
                // clean on the caller's next decision.
                assert!(controller.observe_and_decide(0, 0, false, true) >= 1);
                return;
            }
            assert!(controller.observe_and_decide(0, 0, false, true) >= 1);
        }
        assert!(
            controller.park_baseline_us.is_some(),
            "the settling probe never armed the park baseline"
        );
    }

    #[test]
    fn park_requires_a_post_warmup_baseline_probe() {
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                park_streak: 2,
                ..config()
            },
            2,
        );
        // Warmup sweep and the two warmup-context probes: those measure a
        // cheap direct step and must not arm park.
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 2);
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 1);
        assert!(controller.wants_direct_probe());
        controller.record_direct_probe(1);
        controller.record_direct_probe(1);
        assert_eq!(controller.park_baseline_us, None);
        assert_eq!(controller.direct_reference_us(), Some(1));
        // Terrible scores with only the warmup baseline: no park, however long
        // the losing streak runs.
        controller.p_accept = [Some(0.0); MTP_COST_MAX_DEPTH];
        controller.t_cycle_us = [Some(500_000.0); MTP_COST_MAX_DEPTH];
        drive_to_armed_park_baseline(&mut controller);
        // The settling probe is armed but the streak starts from this decision.
        assert!(!controller.parked());
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 1);
        assert!(!controller.parked());
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 0);
        assert!(controller.parked());
    }

    #[test]
    fn park_requires_a_measured_direct_baseline() {
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                park_streak: 1,
                ..config()
            },
            2,
        );
        for _ in 0..100 {
            assert!(
                controller.observe_and_decide(2, 0, true, true) >= 1,
                "no measured direct baseline means no park"
            );
        }
        assert!(controller.wants_direct_probe());
        assert_eq!(controller.park_baseline_us, None);
        assert_eq!(controller.park_streak, 0);
        assert!(!controller.parked());
    }

    #[test]
    fn contaminated_windows_are_not_depth_cost_samples() {
        let mut controller = fresh_controller(2);
        // Close two windows so the depth slots are alive.
        controller.observe_and_decide(0, 0, false, true);
        controller.observe_and_decide(0, 0, false, true);

        // Clean window at the decided depth: recorded.
        controller.current = 1;
        controller.t_cycle_us[0] = Some(500_000.0);
        controller.observe_and_decide(1, 0, false, true);
        assert_ne!(controller.t_cycle_us[0], Some(500_000.0));

        // A foreign step in the window (fallback / think / n-gram): discarded.
        controller.current = 1;
        controller.t_cycle_us[0] = Some(500_000.0);
        controller.observe_and_decide(1, 0, false, false);
        assert_eq!(controller.t_cycle_us[0], Some(500_000.0));

        // A direct probe inside the window: discarded. Its own span cannot be
        // subtracted exactly (the probe's cache clone is untimed).
        controller.record_direct_probe(1_000);
        controller.current = 1;
        controller.t_cycle_us[0] = Some(500_000.0);
        controller.observe_and_decide(1, 0, false, true);
        assert_eq!(controller.t_cycle_us[0], Some(500_000.0));

        // Steady-state depth switch: the window's draft was generated at a
        // different depth than the round verified, so it prices neither.
        controller.current = 2;
        controller.t_cycle_us[0] = Some(500_000.0);
        controller.observe_and_decide(1, 0, false, true);
        assert_eq!(controller.t_cycle_us[0], Some(500_000.0));

        // Warmup keeps the sample (attributed to the verified depth): one
        // noisy sample per sweep depth is acceptable for the initial ranking.
        let mut controller = fresh_controller(2);
        controller.observe_and_decide(0, 0, false, true);
        controller.observe_and_decide(1, 0, false, true);
        assert!(controller.t_cycle_us[0].is_some());
    }

    #[test]
    fn contaminated_windows_do_not_count_toward_park() {
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                park_streak: 3,
                ..config()
            },
            2,
        );
        drive_to_armed_park_baseline(&mut controller);
        // Every speculative depth emits one token per half second against a
        // 1 ms direct step: losing, so a clean window advances the streak.
        controller.p_accept = [Some(0.0); MTP_COST_MAX_DEPTH];
        controller.t_cycle_us = [Some(500_000.0); MTP_COST_MAX_DEPTH];

        // Window spanned a foreign step.
        assert_eq!(controller.observe_and_decide(0, 0, false, false), 1);
        assert_eq!(controller.park_streak, 0);
        // Window contained the probe.
        controller.record_direct_probe(1_000);
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 1);
        assert_eq!(controller.park_streak, 0);
        // Clean windows count, and a contaminated one in between neither
        // extends nor resets the streak.
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 1);
        assert_eq!(controller.park_streak, 1);
        assert_eq!(controller.observe_and_decide(0, 0, false, false), 1);
        assert_eq!(controller.park_streak, 1);
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 1);
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 0);
        assert!(controller.parked());
    }

    #[test]
    fn park_latches_after_the_streak_and_freezes_force_mtp() {
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                park_streak: 2,
                ..config()
            },
            3,
        );
        drive_to_armed_park_baseline(&mut controller);
        controller.p_accept = [Some(0.0); MTP_COST_MAX_DEPTH];
        controller.t_cycle_us = [Some(500_000.0); MTP_COST_MAX_DEPTH];
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 1);
        assert!(!controller.parked());
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 0);
        assert!(controller.parked());
        // Latched for the rest of the request.
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 0);

        // AX_MLX_MTP_BYPASS_THRESHOLD=0 (force MTP) freezes the depth at the
        // width, skips probes, and never parks: MTP-P evidence must bind to a
        // fixed mechanism, not a controller trajectory.
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                park_streak: 2,
                automatic_bypass_allowed: false,
                ..config()
            },
            3,
        );
        controller.p_accept = [Some(0.0); MTP_COST_MAX_DEPTH];
        controller.t_cycle_us = [Some(500_000.0); MTP_COST_MAX_DEPTH];
        for _ in 0..50 {
            assert_eq!(controller.observe_and_decide(0, 0, false, true), 3);
        }
        assert!(!controller.parked());
        assert!(!controller.wants_direct_probe());
        assert_eq!(controller.park_streak, 0);
        assert_eq!(controller.warmup_remaining, 3, "the freeze skips the sweep");
    }

    #[test]
    fn reset_clears_the_park_latch_and_measurements() {
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                park_streak: 1,
                ..config()
            },
            2,
        );
        drive_to_armed_park_baseline(&mut controller);
        controller.p_accept = [Some(0.0); MTP_COST_MAX_DEPTH];
        controller.t_cycle_us = [Some(500_000.0); MTP_COST_MAX_DEPTH];
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 0);
        assert!(controller.parked());

        controller.reset(true, config(), 3);
        assert!(!controller.parked());
        assert_eq!(controller.park_streak, 0);
        assert_eq!(controller.p_accept, [None; MTP_COST_MAX_DEPTH]);
        assert_eq!(controller.t_cycle_us, [None; MTP_COST_MAX_DEPTH]);
        assert_eq!(controller.direct_probes, 0);
        assert_eq!(controller.park_baseline_us, None);
        assert_eq!(controller.observe_and_decide(0, 0, false, true), 3);
    }

    #[test]
    fn staleness_probe_targets_a_rival_then_the_most_stale_depth() {
        // Candidate 3 (score 0.0344) with rival 2 (0.0301, within 1.15x).
        let mut controller = fresh_controller(3);
        controller.p_accept = [Some(0.9), Some(0.9), Some(0.9), None, None, None, None];
        controller.t_cycle_us = [Some(200.0), Some(90.0), Some(100.0), None, None, None, None];
        assert_eq!(controller.best_scoring_depth(), Some(3));
        assert_eq!(controller.staleness_probe_target(3), 2);

        // No rival within the margin: fall back to the most-stale depth.
        controller.t_cycle_us = [
            Some(100.0),
            Some(200.0),
            Some(300.0),
            None,
            None,
            None,
            None,
        ];
        controller.t_age_cycles = [1, 7, 3, 0, 0, 0, 0];
        assert_eq!(controller.best_scoring_depth(), Some(1));
        assert_eq!(controller.staleness_probe_target(1), 2);

        // Never-measured depths are the stalest of all.
        controller.t_cycle_us = [Some(100.0), Some(200.0), None, None, None, None, None];
        assert_eq!(controller.staleness_probe_target(1), 3);

        // Duty bound: probe cycles stay within the configured share.
        let mut controller = MtpCostDepthController::default();
        controller.reset(
            true,
            MtpCostDepthConfig {
                probe_period_cycles: 1,
                probe_len: 2,
                probe_duty: 0.5,
                ..config()
            },
            2,
        );
        controller.p_accept = [Some(0.0); MTP_COST_MAX_DEPTH];
        controller.t_cycle_us = [Some(1_000.0); MTP_COST_MAX_DEPTH];
        controller.observe_and_decide(0, 0, false, true);
        controller.observe_and_decide(0, 0, false, true);
        for _ in 0..40 {
            assert!((1..=2).contains(&controller.observe_and_decide(0, 0, false, true)));
        }
        assert!(controller.probe_cycles > 0);
        assert!(
            controller.probe_cycles * 2 <= controller.cycles + 2 * controller.config.probe_len,
            "duty bound exceeded: {} probe cycles over {} cycles",
            controller.probe_cycles,
            controller.cycles
        );
    }

    #[test]
    fn spike_damping_limits_an_outlier_cycle_wall() {
        let mut controller = fresh_controller(1);
        controller.update_cycle_wall(1, 1_000);
        assert_eq!(controller.t_cycle_us[0], Some(1_000.0));
        // A 500 ms sample against a 1 ms EMA (> 2x) moves at 0.25 weight, not
        // the wall-horizon alpha (~0.71 at tau = 400 ms).
        controller.update_cycle_wall(1, 500_000);
        let damped = controller.t_cycle_us[0].unwrap();
        assert!((damped - 125_750.0).abs() < 1.0);

        let mut controller = fresh_controller(1);
        controller.update_cycle_wall(1, 100_000);
        controller.update_cycle_wall(1, 120_000);
        assert!((controller.t_cycle_us[0].unwrap() - 105_184.0).abs() < 5.0);
    }

    #[test]
    fn unmeasured_depths_use_measured_slope_and_unknown_costs_stay_excluded() {
        let mut controller = fresh_controller(3);
        assert_eq!(controller.estimated_cycle_us(1), None);
        assert_eq!(controller.score(1), None);
        controller.p_accept = [Some(0.5), None, None, None, None, None, None];
        assert_eq!(controller.score(1), None);

        controller.t_cycle_us = [Some(100.0), Some(140.0), None, None, None, None, None];
        assert_eq!(controller.estimated_cycle_us(1), Some(100.0));
        assert_eq!(controller.estimated_cycle_us(2), Some(140.0));
        // Nearest measured depth 2 plus the measured marginal slope (+40/step).
        assert_eq!(controller.estimated_cycle_us(3), Some(180.0));

        // A bracketed depth interpolates between the measured pair.
        controller.t_cycle_us = [Some(100.0), None, Some(200.0), None, None, None, None];
        assert_eq!(controller.estimated_cycle_us(2), Some(150.0));
    }

    #[test]
    fn env_config_defaults_and_clamps() {
        let defaults = MtpCostDepthConfig::config_from(false, true, MtpCostDepthEnvRaw::default());
        assert_eq!(defaults, MtpCostDepthConfig::default());
        assert!(!defaults.enabled);

        let clamped = MtpCostDepthConfig::config_from(
            true,
            false,
            MtpCostDepthEnvRaw {
                accept_alpha: Some("9.0"),
                time_tau_ms: Some("10"),
                hysteresis: Some("0.5"),
                probe_period_cycles: Some("1"),
                probe_len: Some("99"),
                probe_duty: Some("1.0"),
                probe_margin: Some("0.1"),
                park_margin: Some("5.0"),
                park_streak: Some("0"),
                probe_rounds: Some("9"),
            },
        );
        assert!(clamped.enabled);
        assert!(!clamped.automatic_bypass_allowed);
        assert_eq!(clamped.accept_alpha, 0.5);
        assert_eq!(clamped.time_tau_ms, 50.0);
        assert_eq!(clamped.hysteresis, 1.0);
        assert_eq!(clamped.probe_period_cycles, 8);
        assert_eq!(clamped.probe_len, 16);
        assert_eq!(clamped.probe_duty, 0.5);
        assert_eq!(clamped.probe_margin, 1.0);
        assert_eq!(clamped.park_margin, 2.0);
        assert_eq!(clamped.park_streak, 2);
        assert_eq!(clamped.probe_rounds, 4);

        let lower = MtpCostDepthConfig::config_from(
            false,
            true,
            MtpCostDepthEnvRaw {
                accept_alpha: Some("0.0"),
                time_tau_ms: Some("1e9"),
                hysteresis: Some("nan"),
                probe_period_cycles: Some("not-a-number"),
                probe_len: Some("-3"),
                probe_duty: Some("-1"),
                probe_margin: Some("1e9"),
                park_margin: Some("0.0"),
                park_streak: Some("4096"),
                probe_rounds: Some("0"),
            },
        );
        assert_eq!(lower.accept_alpha, 0.001);
        assert_eq!(lower.time_tau_ms, 10_000.0);
        assert_eq!(lower.hysteresis, 1.10);
        assert_eq!(lower.probe_period_cycles, 64);
        assert_eq!(lower.probe_len, 4);
        assert_eq!(lower.probe_duty, 0.0);
        assert_eq!(lower.probe_margin, 2.0);
        assert_eq!(lower.park_margin, 0.8);
        assert_eq!(lower.park_streak, 1024);
        assert_eq!(lower.probe_rounds, 1);
    }

    #[test]
    fn mixed_rounds_update_cost_without_touching_acceptance() {
        let mut controller = fresh_controller(3);
        controller.observe_and_decide(3, 1, false, true);
        controller.observe_and_decide(3, 1, false, true);
        assert_eq!(controller.p_accept, [None; MTP_COST_MAX_DEPTH]);
        assert_eq!(controller.p_samples, [0; MTP_COST_MAX_DEPTH]);
        assert!(controller.t_cycle_us[2].is_some());

        let mut controller = fresh_controller(3);
        controller.observe_and_decide(3, 1, true, true);
        controller.observe_and_decide(3, 1, true, true);
        assert_eq!(controller.p_accept[0], Some(1.0));
        assert_eq!(controller.p_accept[1], Some(0.0));
        assert_eq!(controller.p_accept[2], None);
    }
}
