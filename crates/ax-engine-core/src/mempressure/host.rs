//! Host RSS pressure input.
//!
//! `HostRssSnapshot` is a passive value type: it carries a `(used, budget)`
//! pair sampled by an external probe. The classifier in
//! [`super::PressureThresholds`] converts that pair into a [`super::PressureLevel`].
//!
//! Real-OS host RSS probing (mach/sysctl on macOS, `/proc` on Linux) is
//! deliberately out of scope here — that work belongs to the bench harness
//! (`crates/ax-engine-bench/src/harness/pressure_observer.rs`), so that core
//! stays free of platform-conditional FFI.

use serde::Serialize;

/// Sampled host resident-set bytes against the configured budget.
///
/// `used_bytes` saturates at `u64::MAX`. `budget_bytes` is the value the
/// caller wants pressure measured against (often the deployment's configured
/// per-process memory ceiling, not the physical machine total).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HostRssSnapshot {
    pub used_bytes: u64,
    pub budget_bytes: u64,
}

impl HostRssSnapshot {
    /// `used / budget` as a fraction in `[0.0, 1.0]`. A zero budget returns
    /// `1.0` (treat "no budget allocated" as fully-loaded so a misconfigured
    /// deployment does not silently report `Normal`).
    pub fn utilization_fraction(self) -> f64 {
        if self.budget_bytes == 0 {
            return 1.0;
        }
        let used = self.used_bytes as f64;
        let budget = self.budget_bytes as f64;
        (used / budget).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utilization_basic() {
        let s = HostRssSnapshot {
            used_bytes: 500,
            budget_bytes: 1_000,
        };
        assert!((s.utilization_fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn utilization_saturates_above_unity() {
        let s = HostRssSnapshot {
            used_bytes: 2_000,
            budget_bytes: 1_000,
        };
        assert_eq!(s.utilization_fraction(), 1.0);
    }

    #[test]
    fn zero_budget_reports_full_load() {
        let s = HostRssSnapshot {
            used_bytes: 0,
            budget_bytes: 0,
        };
        assert_eq!(s.utilization_fraction(), 1.0);
    }

    #[test]
    fn zero_used_reports_zero_fraction() {
        let s = HostRssSnapshot {
            used_bytes: 0,
            budget_bytes: 4096,
        };
        assert_eq!(s.utilization_fraction(), 0.0);
    }
}
