//! Device-resident-set pressure input.
//!
//! Mirrors [`super::host::HostRssSnapshot`] for the GPU / Metal device's
//! resident bytes. Apple Silicon's unified memory architecture means host and
//! device share the same physical pool, but the budgets a deployment cares
//! about are usually distinct (per-process RSS vs. recommended working-set
//! size from `MTLDevice.recommendedMaxWorkingSetSize`).

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceResidentSnapshot {
    pub used_bytes: u64,
    pub budget_bytes: u64,
}

impl DeviceResidentSnapshot {
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
        let s = DeviceResidentSnapshot {
            used_bytes: 8_000,
            budget_bytes: 10_000,
        };
        assert!((s.utilization_fraction() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn utilization_clamps_above_unity() {
        let s = DeviceResidentSnapshot {
            used_bytes: 16_000,
            budget_bytes: 10_000,
        };
        assert_eq!(s.utilization_fraction(), 1.0);
    }

    #[test]
    fn zero_budget_reports_full_load() {
        let s = DeviceResidentSnapshot {
            used_bytes: 0,
            budget_bytes: 0,
        };
        assert_eq!(s.utilization_fraction(), 1.0);
    }
}
