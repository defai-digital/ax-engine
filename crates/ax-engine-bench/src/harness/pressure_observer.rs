//! Bench-harness adapter for invariant I-4 memory pressure observation.
//!
//! Wraps the platform-neutral [`ax_engine_core::mempressure`] classifier and
//! records the resulting [`PressureObservation`] onto a [`WorkloadReport`].
//! This is observe-mode: probes feed the reporter, the reporter prints
//! telemetry, no admission policy is altered.
//!
//! Real-OS probing (mach_task_basic_info, sysctl hw.memsize, Metal device
//! recommendedMaxWorkingSetSize) is intentionally pluggable behind the
//! [`PressureProbes`] trait so unit tests can drive synthetic snapshots
//! without touching the host or device.

use ax_engine_core::{
    DeviceResidentSnapshot, HostRssSnapshot, PressureLevel, PressureObservation, PressureThresholds,
};

use super::WorkloadReport;

/// Pluggable probe surface. Tests inject a deterministic implementation;
/// production code wires a platform-specific probe in a follow-up patch.
pub(crate) trait PressureProbes {
    fn host_rss(&self) -> Option<HostRssSnapshot>;
    fn device_resident(&self) -> Option<DeviceResidentSnapshot>;
}

/// Static snapshot pair, useful for tests and for replay fixtures that load
/// a recorded probe pair from disk.
#[derive(Debug, Clone, Default)]
pub(crate) struct StaticProbes {
    pub host: Option<HostRssSnapshot>,
    pub device: Option<DeviceResidentSnapshot>,
}

impl PressureProbes for StaticProbes {
    fn host_rss(&self) -> Option<HostRssSnapshot> {
        self.host
    }
    fn device_resident(&self) -> Option<DeviceResidentSnapshot> {
        self.device
    }
}

/// Capture a single pressure observation and attach it to a workload report.
///
/// Decisions written:
///
/// - `ax_mempressure_host_level`     -> 0/1/2 (Normal/Soft/Hard)
/// - `ax_mempressure_device_level`   -> 0/1/2
/// - `ax_mempressure_combined_level` -> 0/1/2
///
/// Optional byte counters (`*_used_bytes`, `*_budget_bytes`) are added only
/// when the corresponding probe returned a snapshot; an absent probe leaves
/// the keys out of the decisions vector so downstream tooling can tell apart
/// "probe unavailable" from "probe sampled zero".
pub(crate) fn observe_and_record(
    probes: &dyn PressureProbes,
    thresholds: PressureThresholds,
    report: &mut WorkloadReport,
) -> PressureObservation {
    let host = probes.host_rss();
    let device = probes.device_resident();
    let observation = PressureObservation::from_snapshots(host, device, thresholds);
    record_decisions(&observation, report);
    observation
}

fn record_decisions(observation: &PressureObservation, report: &mut WorkloadReport) {
    report.add_decision("ax_mempressure_host_level", observation.host_level.as_u64());
    report.add_decision(
        "ax_mempressure_device_level",
        observation.device_level.as_u64(),
    );
    report.add_decision(
        "ax_mempressure_combined_level",
        observation.combined_level.as_u64(),
    );
    if let (Some(used), Some(budget)) = (observation.host_used_bytes, observation.host_budget_bytes)
    {
        report.add_decision("ax_mempressure_host_used_bytes", used);
        report.add_decision("ax_mempressure_host_budget_bytes", budget);
    }
    if let (Some(used), Some(budget)) = (
        observation.device_used_bytes,
        observation.device_budget_bytes,
    ) {
        report.add_decision("ax_mempressure_device_used_bytes", used);
        report.add_decision("ax_mempressure_device_budget_bytes", budget);
    }
    report.add_note(format!(
        "mempressure observe-mode: host={}, device={}, combined={}",
        observation.host_level.as_str(),
        observation.device_level.as_str(),
        observation.combined_level.as_str()
    ));
}

/// Convenience: an `empty` observation that callers can drop into a report
/// when no probes are wired (for example, when the fixture runs on a CI host
/// without Metal). Records `Normal` for all three levels.
#[allow(dead_code)]
pub(crate) fn record_empty(report: &mut WorkloadReport) -> PressureObservation {
    let observation = PressureObservation::empty();
    record_decisions(&observation, report);
    observation
}

/// Default thresholds re-exported so fixtures can construct an observation
/// without depending on `ax_engine_core::PressureThresholds` directly.
#[allow(dead_code)]
pub(crate) fn default_thresholds() -> PressureThresholds {
    PressureThresholds::default()
}

/// Helper used by tests and by the run-serving-stress driver: classify a raw
/// `(host, device)` pair without going through a `PressureProbes` instance.
/// Returns the combined level only.
#[allow(dead_code)]
pub(crate) fn classify_pair(
    host: Option<HostRssSnapshot>,
    device: Option<DeviceResidentSnapshot>,
    thresholds: PressureThresholds,
) -> PressureLevel {
    PressureObservation::from_snapshots(host, device, thresholds).combined_level
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn count_decision(value: &Value, key: &str) -> Option<u64> {
        let decisions = value["decisions"].as_array()?;
        for entry in decisions {
            if entry["key"].as_str() == Some(key) {
                return entry["value"].as_u64();
            }
        }
        None
    }

    #[test]
    fn observe_and_record_writes_decisions_into_report() {
        let probes = StaticProbes {
            host: Some(HostRssSnapshot {
                used_bytes: 800,
                budget_bytes: 1_000,
            }),
            device: Some(DeviceResidentSnapshot {
                used_bytes: 950,
                budget_bytes: 1_000,
            }),
        };
        let mut report = WorkloadReport::new("test_pressure");
        let obs = observe_and_record(&probes, PressureThresholds::default(), &mut report);

        assert_eq!(obs.host_level, PressureLevel::Soft);
        assert_eq!(obs.device_level, PressureLevel::Hard);
        assert_eq!(obs.combined_level, PressureLevel::Hard);

        let value = report.to_json();
        assert_eq!(count_decision(&value, "ax_mempressure_host_level"), Some(1));
        assert_eq!(
            count_decision(&value, "ax_mempressure_device_level"),
            Some(2)
        );
        assert_eq!(
            count_decision(&value, "ax_mempressure_combined_level"),
            Some(2)
        );
        assert_eq!(
            count_decision(&value, "ax_mempressure_host_used_bytes"),
            Some(800)
        );
        assert_eq!(
            count_decision(&value, "ax_mempressure_device_budget_bytes"),
            Some(1_000)
        );
    }

    #[test]
    fn absent_probes_record_only_levels() {
        let probes = StaticProbes {
            host: None,
            device: None,
        };
        let mut report = WorkloadReport::new("test_empty");
        let obs = observe_and_record(&probes, PressureThresholds::default(), &mut report);
        assert_eq!(obs.combined_level, PressureLevel::Normal);

        let value = report.to_json();
        // Levels always present, even when probes are absent: this is a contract
        // surface for downstream tooling that filters on level keys.
        assert_eq!(
            count_decision(&value, "ax_mempressure_combined_level"),
            Some(0)
        );
        // Byte counters must be absent so an analyzer can distinguish absent
        // probe from zero-byte usage.
        assert!(count_decision(&value, "ax_mempressure_host_used_bytes").is_none());
        assert!(count_decision(&value, "ax_mempressure_device_used_bytes").is_none());
    }

    #[test]
    fn record_empty_adds_normal_levels() {
        let mut report = WorkloadReport::new("test_record_empty");
        let obs = record_empty(&mut report);
        assert_eq!(obs.combined_level, PressureLevel::Normal);
        let value = report.to_json();
        assert_eq!(
            count_decision(&value, "ax_mempressure_combined_level"),
            Some(0)
        );
    }

    #[test]
    fn classify_pair_picks_more_severe_side() {
        let host = HostRssSnapshot {
            used_bytes: 100,
            budget_bytes: 1_000,
        };
        let device = DeviceResidentSnapshot {
            used_bytes: 920,
            budget_bytes: 1_000,
        };
        let level = classify_pair(Some(host), Some(device), PressureThresholds::default());
        assert_eq!(level, PressureLevel::Hard);
    }

    #[test]
    fn note_carries_observation_summary() {
        let probes = StaticProbes {
            host: Some(HostRssSnapshot {
                used_bytes: 0,
                budget_bytes: 1_000,
            }),
            device: None,
        };
        let mut report = WorkloadReport::new("note_test");
        let _ = observe_and_record(&probes, PressureThresholds::default(), &mut report);
        let value = report.to_json();
        let notes = value["notes"].as_array().expect("notes is array");
        assert!(
            notes.iter().any(|n| n
                .as_str()
                .is_some_and(|s| s.contains("mempressure observe-mode"))),
            "expected pressure observe note in report"
        );
    }
}
