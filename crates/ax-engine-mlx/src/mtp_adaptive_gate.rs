//! Online adaptive MTP draft confidence gate (default OFF).
//!
//! Design: `docs/designs/mtp-embed-perf-sprint-2026-07-16.md`.
//! Enable with `AX_MLX_MTP_ADAPTIVE_GATE=1`. Adaptive applies only under
//! low-temperature `auto` speculation profile; coding/agentic/chatbot pins
//! and high-T diversity remain hard pins.

use std::sync::OnceLock;

use crate::mtp::DEFAULT_MTP_DRAFT_MIN_CONFIDENCE;
use crate::speculation_profile::{
    ResolutionSource, SpeculationProfile, speculation_profile_from_env,
};

const HEAD_CONF_ALPHA: f32 = 0.05;
const RECOMPUTE_ALPHA: f32 = 0.05;
const DRAFT_LEN_ALPHA: f32 = 0.05;
const DEFAULT_GATE_MIN: f32 = 0.80;
const DEFAULT_GATE_MAX: f32 = 0.95;
const DEFAULT_RESIDUAL_WINDOW: u32 = 64;
const DEFAULT_RESIDUAL_MAX: f32 = 0.02;
const DEFAULT_SNAP_MIN_SAMPLES: u32 = 1;

/// Per-request adaptive gate state. Not shared across requests.
#[derive(Clone, Debug)]
pub struct MtpAdaptiveGateState {
    pub gate: f32,
    pub head_conf_ewma: f32,
    pub head_conf_samples: u32,
    pub recompute_ewma: f32,
    pub recompute_samples: u32,
    pub gated_draft_len_ewma: f32,
    pub draft_len_samples: u32,
    pub steps_since_residual: u32,
    pub frozen: bool,
}

impl MtpAdaptiveGateState {
    pub fn new(initial_gate: f32) -> Self {
        Self {
            gate: initial_gate.clamp(gate_min(), gate_max()),
            head_conf_ewma: 0.0,
            head_conf_samples: 0,
            recompute_ewma: 0.0,
            recompute_samples: 0,
            gated_draft_len_ewma: 0.0,
            draft_len_samples: 0,
            steps_since_residual: 0,
            frozen: false,
        }
    }
}

/// Signals observed after one MTP decode step (verify/accept complete).
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveStepSignals {
    pub pre_gate_mean_conf: f32,
    pub gated_draft_len: usize,
    pub recomputed: bool,
    pub mtp_only_accept_rate_ewma: f32,
    pub mtp_only_accept_rate_ewma_samples: u32,
    pub mtp_bypassed: bool,
    pub adaptive_depth: usize,
    pub auto_optimistic_active: bool,
}

#[derive(Clone, Debug)]
pub struct NextGateConfig {
    pub residual_window: u32,
    pub residual_enabled: bool,
    pub residual_max: f32,
    pub k_accept: f32,
    pub k_recomp: f32,
    pub snap_min_samples: u32,
    pub bins: Vec<(f32, f32)>,
}

impl Default for NextGateConfig {
    fn default() -> Self {
        Self {
            residual_window: residual_window_from_env(),
            residual_enabled: residual_enabled_from_env(),
            residual_max: DEFAULT_RESIDUAL_MAX,
            k_accept: 0.05,
            k_recomp: 0.05,
            snap_min_samples: DEFAULT_SNAP_MIN_SAMPLES,
            bins: default_provisional_bins().to_vec(),
        }
    }
}

fn truthy_env(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            matches!(t.as_str(), "1" | "true" | "on" | "yes")
        }
        Err(_) => false,
    }
}

/// Process flag: adaptive controller enabled (default OFF).
pub fn adaptive_gate_enabled_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| truthy_env("AX_MLX_MTP_ADAPTIVE_GATE"))
}

fn residual_enabled_from_env() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| truthy_env("AX_MLX_MTP_ADAPTIVE_GATE_RESIDUAL"))
}

fn parse_f32_env(name: &str, default: f32, lo: f32, hi: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(lo, hi))
        .unwrap_or(default)
}

fn gate_min() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED
        .get_or_init(|| parse_f32_env("AX_MLX_MTP_ADAPTIVE_GATE_MIN", DEFAULT_GATE_MIN, 0.5, 0.99))
}

fn gate_max() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_f32_env("AX_MLX_MTP_ADAPTIVE_GATE_MAX", DEFAULT_GATE_MAX, 0.5, 0.99).max(gate_min())
    })
}

fn residual_window_from_env() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_ADAPTIVE_GATE_WINDOW")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(|v| v.clamp(8, 512))
            .unwrap_or(DEFAULT_RESIDUAL_WINDOW)
    })
}

/// PROVISIONAL prior bins — replace via `AX_MLX_MTP_ADAPTIVE_GATE_PRIOR_BINS`
/// after PR0 calibration (`thr:prior,...` descending thresholds).
pub fn default_provisional_bins() -> &'static [(f32, f32)] {
    &[(0.95, 0.90), (0.90, 0.88), (0.85, 0.85), (0.00, 0.80)]
}

fn bins_from_env() -> Vec<(f32, f32)> {
    static CACHED: OnceLock<Vec<(f32, f32)>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            if let Ok(raw) = std::env::var("AX_MLX_MTP_ADAPTIVE_GATE_PRIOR_BINS") {
                let mut bins = Vec::new();
                for part in raw.split(',') {
                    let mut it = part.trim().split(':');
                    let (Some(thr), Some(prior)) = (it.next(), it.next()) else {
                        continue;
                    };
                    if let (Ok(t), Ok(p)) = (thr.parse::<f32>(), prior.parse::<f32>()) {
                        if t.is_finite() && p.is_finite() {
                            bins.push((t, p.clamp(gate_min(), gate_max())));
                        }
                    }
                }
                if !bins.is_empty() {
                    bins.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    return bins;
                }
            }
            default_provisional_bins().to_vec()
        })
        .clone()
}

pub fn prior_from_head_conf(mean_conf: f32, bins: &[(f32, f32)]) -> f32 {
    let lo = gate_min();
    let hi = gate_max();
    for &(threshold, prior) in bins {
        if mean_conf >= threshold {
            return prior.clamp(lo, hi);
        }
    }
    lo
}

fn ewma(prev: f32, samples: u32, obs: f32, alpha: f32) -> (f32, u32) {
    if samples == 0 {
        (obs, 1)
    } else {
        (
            (1.0 - alpha) * prev + alpha * obs,
            samples.saturating_add(1),
        )
    }
}

/// Update state after one decode step; returns gate for the *next* draft.
pub fn observe_step(
    state: &mut MtpAdaptiveGateState,
    sig: AdaptiveStepSignals,
    cfg: &NextGateConfig,
) -> f32 {
    if sig.mtp_bypassed || sig.adaptive_depth == 0 || sig.auto_optimistic_active {
        state.frozen = true;
        return state.gate;
    }
    state.frozen = false;

    let (hc, n_hc) = ewma(
        state.head_conf_ewma,
        state.head_conf_samples,
        sig.pre_gate_mean_conf.clamp(0.0, 1.0),
        HEAD_CONF_ALPHA,
    );
    state.head_conf_ewma = hc;
    state.head_conf_samples = n_hc;

    let (rc, n_rc) = ewma(
        state.recompute_ewma,
        state.recompute_samples,
        if sig.recomputed { 1.0 } else { 0.0 },
        RECOMPUTE_ALPHA,
    );
    state.recompute_ewma = rc;
    state.recompute_samples = n_rc;

    let (dl, n_dl) = ewma(
        state.gated_draft_len_ewma,
        state.draft_len_samples,
        sig.gated_draft_len as f32,
        DRAFT_LEN_ALPHA,
    );
    state.gated_draft_len_ewma = dl;
    state.draft_len_samples = n_dl;

    if state.head_conf_samples < cfg.snap_min_samples.max(1) {
        return state.gate;
    }

    let prior = prior_from_head_conf(state.head_conf_ewma, &cfg.bins);
    let lo = gate_min();
    let hi = gate_max();

    if !cfg.residual_enabled {
        state.gate = prior;
        return state.gate;
    }

    state.steps_since_residual = state.steps_since_residual.saturating_add(1);
    let rw = cfg.residual_window.max(1);
    if state.steps_since_residual < rw || sig.mtp_only_accept_rate_ewma_samples < rw {
        state.gate = prior;
        return state.gate;
    }
    state.steps_since_residual = 0;

    // Residual around prior (not free-walk): loosen when accept high & recompute low.
    let accept = sig.mtp_only_accept_rate_ewma.clamp(0.0, 1.0);
    let residual = cfg.k_accept * (accept - 0.97) - cfg.k_recomp * state.recompute_ewma;
    let residual = residual.clamp(-cfg.residual_max, cfg.residual_max);
    state.gate = (prior + residual).clamp(prior - cfg.residual_max, prior + cfg.residual_max);
    state.gate = state.gate.clamp(lo, hi);
    state.gate
}

/// True when this request may allocate adaptive state.
pub fn adaptive_eligible(
    adaptive_enabled: bool,
    profile: SpeculationProfile,
    temperature: Option<f32>,
) -> bool {
    adaptive_enabled
        && matches!(profile, SpeculationProfile::Auto)
        && !SpeculationProfile::is_diversity_temperature(temperature)
}

/// Allocate initial adaptive state at generation start when eligible.
pub fn maybe_init_state(
    adaptive_enabled: bool,
    profile: SpeculationProfile,
    temperature: Option<f32>,
) -> Option<MtpAdaptiveGateState> {
    if adaptive_eligible(adaptive_enabled, profile, temperature) {
        Some(MtpAdaptiveGateState::new(DEFAULT_MTP_DRAFT_MIN_CONFIDENCE))
    } else {
        None
    }
}

/// Full resolution with optimistic override + adaptive (design A.2b).
pub fn resolve_mtp_gate(
    profile: SpeculationProfile,
    temperature: Option<f32>,
    adaptive_enabled: bool,
    adaptive: Option<&MtpAdaptiveGateState>,
    optimistic_override: Option<f32>,
    explicit_env: Option<f32>,
) -> (f32, ResolutionSource) {
    if let Some(g) = optimistic_override {
        return (g, ResolutionSource::Optimistic);
    }
    if let Some(g) = explicit_env {
        return (g, ResolutionSource::Explicit);
    }
    // Non-auto profiles are hard pins (ADR-022).
    if !matches!(profile, SpeculationProfile::Auto) {
        if let Some(g) = profile.qwen_gate(temperature) {
            return (g, ResolutionSource::Profile);
        }
    }
    // High-T auto diversity pin BEFORE adaptive.
    if matches!(profile, SpeculationProfile::Auto)
        && SpeculationProfile::is_diversity_temperature(temperature)
    {
        if let Some(g) = profile.qwen_gate(temperature) {
            return (g, ResolutionSource::Profile);
        }
    }
    // Low-T auto + adaptive: always use st.gate when present (including frozen).
    if adaptive_enabled {
        if let Some(st) = adaptive {
            return (st.gate, ResolutionSource::Adaptive);
        }
    }
    if let Some(g) = profile.qwen_gate(temperature) {
        return (g, ResolutionSource::Profile);
    }
    (DEFAULT_MTP_DRAFT_MIN_CONFIDENCE, ResolutionSource::Default)
}

/// Convenience: resolve using process env profile + adaptive flag + optional state.
pub fn resolve_mtp_gate_from_env(
    temperature: Option<f32>,
    adaptive: Option<&MtpAdaptiveGateState>,
    optimistic_override: Option<f32>,
) -> (f32, ResolutionSource) {
    let profile = speculation_profile_from_env();
    let explicit = crate::mtp::mtp_draft_min_confidence_env_value();
    resolve_mtp_gate(
        profile,
        temperature,
        adaptive_gate_enabled_from_env(),
        adaptive,
        optimistic_override,
        explicit,
    )
}

/// Config snapshot for observe_step (env-backed).
pub fn next_gate_config_from_env() -> NextGateConfig {
    NextGateConfig {
        residual_window: residual_window_from_env(),
        residual_enabled: residual_enabled_from_env(),
        residual_max: DEFAULT_RESIDUAL_MAX,
        k_accept: 0.05,
        k_recomp: 0.05,
        snap_min_samples: DEFAULT_SNAP_MIN_SAMPLES,
        bins: bins_from_env(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_bins_pick_by_threshold() {
        let bins = default_provisional_bins();
        assert!((prior_from_head_conf(0.96, bins) - 0.90).abs() < 1e-5);
        assert!((prior_from_head_conf(0.86, bins) - 0.85).abs() < 1e-5);
        assert!((prior_from_head_conf(0.50, bins) - 0.80).abs() < 1e-5);
    }

    #[test]
    fn observe_step_snaps_after_first_sample() {
        let mut st = MtpAdaptiveGateState::new(0.90);
        let cfg = NextGateConfig {
            residual_enabled: false,
            bins: default_provisional_bins().to_vec(),
            ..NextGateConfig::default()
        };
        let sig = AdaptiveStepSignals {
            pre_gate_mean_conf: 0.70,
            gated_draft_len: 1,
            recomputed: false,
            mtp_only_accept_rate_ewma: 0.95,
            mtp_only_accept_rate_ewma_samples: 10,
            mtp_bypassed: false,
            adaptive_depth: 2,
            auto_optimistic_active: false,
        };
        let g = observe_step(&mut st, sig, &cfg);
        assert!(
            (g - 0.80).abs() < 1e-5,
            "low conf → long_code prior 0.80, got {g}"
        );
        assert_eq!(st.head_conf_samples, 1);
    }

    #[test]
    fn freeze_holds_last_gate_not_default() {
        let mut st = MtpAdaptiveGateState::new(0.90);
        let cfg = NextGateConfig {
            residual_enabled: false,
            bins: default_provisional_bins().to_vec(),
            ..NextGateConfig::default()
        };
        let mut sig = AdaptiveStepSignals {
            pre_gate_mean_conf: 0.70,
            gated_draft_len: 1,
            recomputed: false,
            mtp_only_accept_rate_ewma: 0.95,
            mtp_only_accept_rate_ewma_samples: 10,
            mtp_bypassed: false,
            adaptive_depth: 2,
            auto_optimistic_active: false,
        };
        let _ = observe_step(&mut st, sig, &cfg);
        let frozen_gate = st.gate;
        sig.auto_optimistic_active = true;
        let g = observe_step(&mut st, sig, &cfg);
        assert!(st.frozen);
        assert!((g - frozen_gate).abs() < 1e-6);
        let (resolved, src) = resolve_mtp_gate(
            SpeculationProfile::Auto,
            Some(0.0),
            true,
            Some(&st),
            None,
            None,
        );
        assert_eq!(src, ResolutionSource::Adaptive);
        assert!((resolved - frozen_gate).abs() < 1e-6);
    }

    #[test]
    fn high_t_auto_pins_chatbot_before_adaptive() {
        let st = MtpAdaptiveGateState::new(0.80);
        let (g, src) = resolve_mtp_gate(
            SpeculationProfile::Auto,
            Some(0.7),
            true,
            Some(&st),
            None,
            None,
        );
        assert_eq!(src, ResolutionSource::Profile);
        assert!((g - 0.99).abs() < 1e-5);
    }

    #[test]
    fn coding_profile_is_hard_pin() {
        let st = MtpAdaptiveGateState::new(0.80);
        let (g, src) = resolve_mtp_gate(
            SpeculationProfile::Coding,
            Some(0.0),
            true,
            Some(&st),
            None,
            None,
        );
        assert_eq!(src, ResolutionSource::Profile);
        assert!((g - 0.85).abs() < 1e-5);
    }

    #[test]
    fn optimistic_override_wins() {
        let (g, src) = resolve_mtp_gate(
            SpeculationProfile::Auto,
            Some(0.0),
            true,
            None,
            Some(0.0),
            Some(0.5),
        );
        assert_eq!(src, ResolutionSource::Optimistic);
        assert_eq!(g, 0.0);
    }

    #[test]
    fn adaptive_eligible_only_low_t_auto() {
        assert!(adaptive_eligible(true, SpeculationProfile::Auto, Some(0.0)));
        assert!(!adaptive_eligible(
            true,
            SpeculationProfile::Auto,
            Some(0.6)
        ));
        assert!(!adaptive_eligible(
            true,
            SpeculationProfile::Coding,
            Some(0.0)
        ));
        assert!(!adaptive_eligible(
            false,
            SpeculationProfile::Auto,
            Some(0.0)
        ));
    }
}
