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
use mlx_sys::{device_active_bytes, device_recommended_working_set_bytes, host_resident_bytes};

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

/// Real platform probes for invariant I-4. Reads the current process RSS
/// via `mlx_sys::host_resident_bytes` (POSIX `getrusage`) and the device
/// working-set via `mlx_get_active_memory` + Metal
/// `recommendedMaxWorkingSetSize`.
///
/// `host_budget_bytes` and `device_budget_bytes` are stored at construction
/// so the classifier can compare against a stable budget for the entire
/// fixture run. The host budget defaults to the device budget (Apple
/// Silicon unified memory) and can be overridden via the constructor when
/// a deployment has a separate per-process RSS ceiling.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlatformProbes {
    host_budget_bytes: u64,
    device_budget_bytes: u64,
}

impl PlatformProbes {
    /// Construct a probe pair using Metal's recommended working-set size
    /// as both budgets. Returns `None` when the Metal budget is
    /// unavailable (CPU-only hosts, non-macOS targets without an MLX
    /// runtime backing), so the caller can fall back to
    /// [`StaticProbes::default()`].
    ///
    /// On a fresh process that has not yet touched MLX, the Metal
    /// device-info pathway returns 0 for `recommendedMaxWorkingSetSize`
    /// (the platform driver lazy-initializes on first GPU touch). To
    /// avoid mis-reporting the budget as unavailable for session-free
    /// fixtures that nonetheless run alongside Metal-using ones, this
    /// constructor first creates and drops a default GPU stream, which
    /// is sufficient to force MLX's Metal context to come up. If Metal
    /// is genuinely unavailable (no GPU, CPU-only target), the stream
    /// creation is harmless and the probe returns None as before.
    pub fn from_metal_runtime() -> Option<Self> {
        // Drop the stream immediately; we don't need to keep it around,
        // we just need MLX to have queried the device once.
        let _bootstrap = mlx_sys::MlxStream::default_gpu();
        let device_budget_bytes = device_recommended_working_set_bytes()?;
        Some(Self {
            host_budget_bytes: device_budget_bytes,
            device_budget_bytes,
        })
    }

    /// Construct a probe pair with an explicit host budget. Useful when a
    /// deployment caps per-process RSS below the device working-set size.
    #[allow(dead_code)]
    pub fn with_host_budget(host_budget_bytes: u64, device_budget_bytes: u64) -> Self {
        Self {
            host_budget_bytes,
            device_budget_bytes,
        }
    }
}

impl PressureProbes for PlatformProbes {
    fn host_rss(&self) -> Option<HostRssSnapshot> {
        host_resident_bytes().map(|used_bytes| HostRssSnapshot {
            used_bytes,
            budget_bytes: self.host_budget_bytes,
        })
    }

    fn device_resident(&self) -> Option<DeviceResidentSnapshot> {
        device_active_bytes().map(|used_bytes| DeviceResidentSnapshot {
            used_bytes,
            budget_bytes: self.device_budget_bytes,
        })
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
    fn platform_probes_from_metal_runtime_provides_host_and_device_on_apple_silicon() {
        // On a Metal-capable host the platform probes must produce a
        // non-empty observation: a positive RSS via getrusage and a
        // non-zero working-set budget via Metal recommendedMaxWorkingSetSize.
        // On CPU-only / non-macOS hosts the constructor returns None; the
        // bench dispatch then falls back to StaticProbes::default(). This
        // test asserts the happy path *when* PlatformProbes constructs.
        if let Some(probes) = PlatformProbes::from_metal_runtime() {
            let host = probes.host_rss().expect("host RSS available on Unix");
            assert!(host.used_bytes > 0, "host RSS must be positive");
            assert!(host.budget_bytes > 0, "host budget must be positive");

            // device_active_bytes() may return None if MLX has not yet
            // touched the GPU; we accept both states, but the budget side
            // of the snapshot must reflect what the constructor captured.
            if let Some(dev) = probes.device_resident() {
                assert!(dev.budget_bytes > 0, "device budget must be positive");
            }
        }
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
