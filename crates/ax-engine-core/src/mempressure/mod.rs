//! Memory pressure observation (invariant I-4).
//!
//! This module records a graded soft/hard pressure signal from host RSS and
//! device-resident-set inputs. It is **observe-mode only**: a
//! [`PressureObservation`] is emitted alongside existing KV-pressure scheduler
//! decisions, but no admission policy is changed here. Enforcement is gated on
//! a separate ADR per `.internal/adr/ADR-007-engine-serving-invariants.md`.
//!
//! Thresholds default to the values agreed in the PRD (`Soft` at 75% of the
//! configured budget, `Hard` at 90%). Callers may construct custom thresholds
//! when a deployment has unusually small or large unified memory.

pub mod device;
pub mod host;

use serde::Serialize;

pub use device::DeviceResidentSnapshot;
pub use host::HostRssSnapshot;

/// Graded memory pressure level visible to scheduler telemetry.
///
/// `Normal` -> spare capacity, no action needed.
/// `Soft`   -> approaching the configured budget; treat as advisory.
/// `Hard`   -> at or beyond the budget; future ADR may admit admission caps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureLevel {
    #[default]
    Normal,
    Soft,
    Hard,
}

impl PressureLevel {
    /// Stable telemetry key. Recorded directly into bench artifacts and
    /// `RouteMetadata` decisions where applicable.
    pub const fn as_str(self) -> &'static str {
        match self {
            PressureLevel::Normal => "normal",
            PressureLevel::Soft => "soft",
            PressureLevel::Hard => "hard",
        }
    }

    /// Integer encoding suitable for `RouteMetadata::crossover_decisions`
    /// (`u64` key/value telemetry surface).
    pub const fn as_u64(self) -> u64 {
        match self {
            PressureLevel::Normal => 0,
            PressureLevel::Soft => 1,
            PressureLevel::Hard => 2,
        }
    }

    /// Numerically dominant of two levels. Used when host and device probes
    /// disagree — the more severe wins.
    pub fn max(self, other: PressureLevel) -> PressureLevel {
        if self.as_u64() >= other.as_u64() {
            self
        } else {
            other
        }
    }
}

/// Threshold pair used by the classifier. Values are fractions of the budget
/// (0.0–1.0); the classifier saturates if a caller supplies inputs outside
/// `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureThresholds {
    pub soft_fraction: f64,
    pub hard_fraction: f64,
}

impl PressureThresholds {
    /// Defaults agreed in `.internal/prd/engine-serving-invariants.md` §9 risks.
    pub const DEFAULT: PressureThresholds = PressureThresholds {
        soft_fraction: 0.75,
        hard_fraction: 0.90,
    };

    /// Construct custom thresholds. The constructor sorts the pair so `soft`
    /// is always ≤ `hard`; this avoids silently-inverted policy on a caller
    /// mistake. Returns `None` if either input is NaN.
    pub fn new(soft: f64, hard: f64) -> Option<Self> {
        if soft.is_nan() || hard.is_nan() {
            return None;
        }
        let (soft, hard) = if soft <= hard {
            (soft, hard)
        } else {
            (hard, soft)
        };
        Some(Self {
            soft_fraction: soft.clamp(0.0, 1.0),
            hard_fraction: hard.clamp(0.0, 1.0),
        })
    }

    fn classify(self, fraction: f64) -> PressureLevel {
        if fraction.is_nan() {
            return PressureLevel::Normal;
        }
        let fraction = fraction.clamp(0.0, 1.0);
        if fraction >= self.hard_fraction {
            PressureLevel::Hard
        } else if fraction >= self.soft_fraction {
            PressureLevel::Soft
        } else {
            PressureLevel::Normal
        }
    }
}

impl Default for PressureThresholds {
    fn default() -> Self {
        PressureThresholds::DEFAULT
    }
}

/// Single observation snapshot.
///
/// All numeric fields use `Option<u64>` so the absence of a probe (for
/// example, no device resident-set on a host without an attached Metal device)
/// is distinguishable from a probe that returned zero.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PressureObservation {
    pub host_used_bytes: Option<u64>,
    pub host_budget_bytes: Option<u64>,
    pub host_level: PressureLevel,
    pub device_used_bytes: Option<u64>,
    pub device_budget_bytes: Option<u64>,
    pub device_level: PressureLevel,
    pub combined_level: PressureLevel,
}

impl PressureObservation {
    /// Classify a `(host, device)` input pair against the supplied thresholds.
    pub fn from_snapshots(
        host: Option<HostRssSnapshot>,
        device: Option<DeviceResidentSnapshot>,
        thresholds: PressureThresholds,
    ) -> Self {
        let (host_used, host_budget, host_level) = match host {
            Some(s) => {
                let level = thresholds.classify(s.utilization_fraction());
                (Some(s.used_bytes), Some(s.budget_bytes), level)
            }
            None => (None, None, PressureLevel::Normal),
        };
        let (device_used, device_budget, device_level) = match device {
            Some(s) => {
                let level = thresholds.classify(s.utilization_fraction());
                (Some(s.used_bytes), Some(s.budget_bytes), level)
            }
            None => (None, None, PressureLevel::Normal),
        };
        Self {
            host_used_bytes: host_used,
            host_budget_bytes: host_budget,
            host_level,
            device_used_bytes: device_used,
            device_budget_bytes: device_budget,
            device_level,
            combined_level: host_level.max(device_level),
        }
    }

    /// Convenience constructor used by tests and by callers that want a
    /// "no probes available" observation (e.g., on platforms where neither
    /// host nor device resident-set is currently instrumented).
    pub fn empty() -> Self {
        Self {
            host_used_bytes: None,
            host_budget_bytes: None,
            host_level: PressureLevel::Normal,
            device_used_bytes: None,
            device_budget_bytes: None,
            device_level: PressureLevel::Normal,
            combined_level: PressureLevel::Normal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_thresholds_match_prd() {
        let t = PressureThresholds::default();
        assert!((t.soft_fraction - 0.75).abs() < f64::EPSILON);
        assert!((t.hard_fraction - 0.90).abs() < f64::EPSILON);
    }

    #[test]
    fn new_sorts_inverted_thresholds() {
        let t = PressureThresholds::new(0.9, 0.5).expect("non-nan");
        assert!(t.soft_fraction <= t.hard_fraction);
    }

    #[test]
    fn new_clamps_to_unit_interval() {
        let t = PressureThresholds::new(-0.1, 1.5).expect("non-nan");
        assert_eq!(t.soft_fraction, 0.0);
        assert_eq!(t.hard_fraction, 1.0);
    }

    #[test]
    fn new_rejects_nan() {
        assert!(PressureThresholds::new(f64::NAN, 0.5).is_none());
        assert!(PressureThresholds::new(0.5, f64::NAN).is_none());
    }

    #[test]
    fn classify_default_thresholds() {
        let t = PressureThresholds::default();
        assert_eq!(t.classify(0.0), PressureLevel::Normal);
        assert_eq!(t.classify(0.5), PressureLevel::Normal);
        assert_eq!(t.classify(0.74999), PressureLevel::Normal);
        assert_eq!(t.classify(0.75), PressureLevel::Soft);
        assert_eq!(t.classify(0.85), PressureLevel::Soft);
        assert_eq!(t.classify(0.90), PressureLevel::Hard);
        assert_eq!(t.classify(1.0), PressureLevel::Hard);
    }

    #[test]
    fn classify_handles_nan_as_normal() {
        let t = PressureThresholds::default();
        assert_eq!(t.classify(f64::NAN), PressureLevel::Normal);
    }

    #[test]
    fn level_max_picks_more_severe() {
        assert_eq!(
            PressureLevel::Normal.max(PressureLevel::Soft),
            PressureLevel::Soft
        );
        assert_eq!(
            PressureLevel::Soft.max(PressureLevel::Hard),
            PressureLevel::Hard
        );
        assert_eq!(
            PressureLevel::Hard.max(PressureLevel::Normal),
            PressureLevel::Hard
        );
    }

    #[test]
    fn observation_combines_host_and_device() {
        let host = HostRssSnapshot {
            used_bytes: 6_000,
            budget_bytes: 10_000,
        };
        let device = DeviceResidentSnapshot {
            used_bytes: 9_500,
            budget_bytes: 10_000,
        };
        let obs = PressureObservation::from_snapshots(
            Some(host),
            Some(device),
            PressureThresholds::default(),
        );
        assert_eq!(obs.host_level, PressureLevel::Normal);
        assert_eq!(obs.device_level, PressureLevel::Hard);
        assert_eq!(obs.combined_level, PressureLevel::Hard);
        assert_eq!(obs.host_used_bytes, Some(6_000));
        assert_eq!(obs.device_used_bytes, Some(9_500));
    }

    #[test]
    fn observation_absent_probes_are_normal() {
        let obs = PressureObservation::from_snapshots(None, None, PressureThresholds::default());
        assert_eq!(obs.combined_level, PressureLevel::Normal);
        assert!(obs.host_used_bytes.is_none());
        assert!(obs.device_used_bytes.is_none());
    }

    #[test]
    fn empty_observation_is_normal_with_no_inputs() {
        let obs = PressureObservation::empty();
        assert_eq!(obs.combined_level, PressureLevel::Normal);
        assert_eq!(obs.host_level, PressureLevel::Normal);
        assert_eq!(obs.device_level, PressureLevel::Normal);
    }

    #[test]
    fn level_string_and_u64_encoding_are_stable() {
        assert_eq!(PressureLevel::Normal.as_str(), "normal");
        assert_eq!(PressureLevel::Soft.as_str(), "soft");
        assert_eq!(PressureLevel::Hard.as_str(), "hard");
        assert_eq!(PressureLevel::Normal.as_u64(), 0);
        assert_eq!(PressureLevel::Soft.as_u64(), 1);
        assert_eq!(PressureLevel::Hard.as_u64(), 2);
    }
}
