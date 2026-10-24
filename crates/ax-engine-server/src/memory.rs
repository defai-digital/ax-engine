//! `GET /v1/memory`: process physical footprint, the priced plan's wired-limit
//! numbers, and, when the native MLX runtime is linked, MLX allocator gauges.
//! No model is loaded.

use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_core::memory_budget::WiredMemoryReport;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde_json::{Value, json};

use crate::app_state::AppState;

pub(crate) async fn memory_sample(
    State(state): State<AppState>,
) -> Result<Json<Value>, StatusCode> {
    if !state.env.api_polish {
        return Err(StatusCode::NOT_FOUND);
    }
    Ok(Json(memory_report()))
}

pub(crate) fn memory_report() -> Value {
    let (footprint, lifetime_peak) = physical_footprint();
    let mut report = json!({
        "phys_footprint_bytes": footprint,
        "lifetime_max_phys_footprint_bytes": lifetime_peak,
        "sampled_at": sampled_at_rfc3339(),
    });
    if let Some(mlx) = mlx_memory() {
        if let Some(object) = report.as_object_mut() {
            object.insert("mlx".to_string(), mlx);
        }
    }
    // The memory plan publishes its wired-limit view from the model worker;
    // a process that never loaded a native model has none.
    if let Some(plan) = wired_memory_plan_json(ax_engine_core::memory_budget::wired_memory_report())
    {
        if let Some(object) = report.as_object_mut() {
            object.insert("memory_plan".to_string(), plan);
        }
    }
    report
}

/// `/v1/memory` view of the priced plan's wired-limit numbers.
///
/// `wired_limit_bytes` is the limit the runner actually set and
/// `wired_shortfall_bytes` is how far the priced working set exceeds it, so an
/// operator can tell a healthy resident session from one macOS is paging. The
/// remedy is the `sysctl` that closes the gap; it needs root and resets on
/// reboot, which is why it is reported rather than applied.
///
/// `live_sequence_cap` is the concurrency the plan proved: when the configured
/// count did not fit under the ceiling the engine serves fewer requests at once
/// (the rest wait in the scheduler) instead of paging the experts, so an
/// operator who sees lower parallelism than `--max-concurrent-requests` asked
/// for can read the reason here. `live_sequences` is the count the plan charged
/// before that reduction. Both are null when the configured count fits: the
/// plan publishes a cap only when it actually limits admission.
fn wired_memory_plan_json(report: Option<WiredMemoryReport>) -> Option<Value> {
    let report = report?;
    let remedy = report
        .remedy_mebibytes
        .map(|mebibytes| format!("sudo sysctl iogpu.wired_limit_mb={mebibytes}"));
    Some(json!({
        "wired_limit_bytes": report.wired_limit_bytes,
        "wired_shortfall_bytes": report.wired_shortfall_bytes,
        "working_set_bytes": report.working_set_bytes,
        "recommended_wired_limit_mebibytes": report.remedy_mebibytes,
        "remedy": remedy,
        "live_sequences": report.charged_live_sequences,
        "live_sequence_cap": report.live_sequence_cap,
    }))
}

fn mlx_memory() -> Option<Value> {
    #[cfg(feature = "mlx-native-server")]
    {
        Some(json!({
            "active_bytes": mlx_sys::device_active_bytes(),
            "cache_bytes": mlx_sys::device_cache_bytes(),
            "peak_bytes": mlx_sys::device_peak_bytes(),
        }))
    }
    #[cfg(not(feature = "mlx-native-server"))]
    {
        None
    }
}

/// macOS physical footprint from `mlx_sys::process_memory_footprint`.
/// Delegated builds have no `mlx-sys` linkage and report nulls.
fn physical_footprint() -> (Option<u64>, Option<u64>) {
    #[cfg(feature = "mlx-native-server")]
    {
        match mlx_sys::process_memory_footprint() {
            Some(footprint) => (
                Some(footprint.phys_footprint_bytes),
                Some(footprint.lifetime_max_phys_footprint_bytes),
            ),
            None => (None, None),
        }
    }
    #[cfg(not(feature = "mlx-native-server"))]
    {
        (None, None)
    }
}

fn sampled_at_rfc3339() -> String {
    let Ok(elapsed) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "1970-01-01T00:00:00.000000000Z".to_string();
    };
    rfc3339_from_unix(elapsed.as_secs(), elapsed.subsec_nanos())
}

fn rfc3339_from_unix(secs: u64, nanos: u32) -> String {
    let days = (secs / 86_400) as i64;
    let tod = secs % 86_400;
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{nanos:09}Z")
}

/// Howard Hinnant's `civil_from_days`. `days` is days since 1970-01-01.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_part = (5 * doy + 2) / 153;
    let day = doy - (153 * month_part + 2) / 5 + 1;
    let month = if month_part < 10 {
        month_part + 3
    } else {
        month_part - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    (year as i32, month as u32, day as u32)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn assert_paired_footprint(footprint: &Value, peak: &Value) {
        let paired = match (footprint, peak) {
            (Value::Null, Value::Null) => true,
            (Value::Number(footprint), Value::Number(peak)) => {
                matches!(
                    (footprint.as_u64(), peak.as_u64()),
                    (Some(footprint), Some(peak)) if peak >= footprint
                )
            }
            _ => false,
        };
        assert!(
            paired,
            "footprint and lifetime peak must both be null or both be u64 with peak >= footprint: {footprint:?} {peak:?}"
        );
    }

    #[test]
    fn rfc3339_formats_known_unix_timestamps() {
        assert_eq!(rfc3339_from_unix(0, 0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(
            rfc3339_from_unix(1_609_459_200, 0),
            "2021-01-01T00:00:00.000000000Z"
        );
    }

    #[test]
    fn wired_memory_plan_reports_the_limit_shortfall_and_remedy() {
        let report = WiredMemoryReport {
            wired_limit_bytes: Some(103_903_852_953),
            working_set_bytes: Some(121_000_000_000),
            wired_shortfall_bytes: Some(17_096_147_047),
            remedy_mebibytes: Some(119_000),
            charged_live_sequences: Some(8),
            live_sequence_cap: Some(5),
        };
        let plan = wired_memory_plan_json(Some(report)).expect("plan block");
        assert_eq!(plan["wired_limit_bytes"], 103_903_852_953u64);
        assert_eq!(plan["wired_shortfall_bytes"], 17_096_147_047u64);
        assert_eq!(plan["working_set_bytes"], 121_000_000_000u64);
        assert_eq!(plan["recommended_wired_limit_mebibytes"], 119_000u64);
        assert_eq!(plan["remedy"], "sudo sysctl iogpu.wired_limit_mb=119000");
        // The concurrency the plan proved travels with the wired numbers: an
        // operator who sees fewer parallel requests than requested reads why.
        assert_eq!(plan["live_sequences"], 8);
        assert_eq!(plan["live_sequence_cap"], 5);
        // No published report means no block at all.
        assert!(wired_memory_plan_json(None).is_none());
        // A report with no remedy still reports both wired fields.
        let quiet = wired_memory_plan_json(Some(WiredMemoryReport {
            wired_limit_bytes: Some(1),
            working_set_bytes: Some(2),
            wired_shortfall_bytes: Some(1),
            remedy_mebibytes: None,
            ..WiredMemoryReport::default()
        }))
        .expect("plan block");
        assert_eq!(quiet["wired_limit_bytes"], 1);
        assert_eq!(quiet["wired_shortfall_bytes"], 1);
        assert!(quiet["remedy"].is_null());
        // A plan that never priced live sequences reports nulls, not zeroes.
        assert!(quiet["live_sequence_cap"].is_null());
        assert!(quiet["live_sequences"].is_null());
    }

    #[test]
    fn memory_report_carries_the_published_wired_memory_plan() {
        // Publishing is process-wide by design (the wired limit is too), so a
        // later sample must expose exactly what the model worker reported.
        ax_engine_core::memory_budget::publish_wired_memory_report(WiredMemoryReport {
            wired_limit_bytes: Some(96_800_000_000),
            working_set_bytes: Some(121_000_000_000),
            wired_shortfall_bytes: Some(24_200_000_000),
            remedy_mebibytes: Some(119_000),
            charged_live_sequences: Some(8),
            live_sequence_cap: Some(6),
        });
        let report = memory_report();
        assert_eq!(
            report["memory_plan"]["wired_limit_bytes"],
            96_800_000_000u64
        );
        assert_eq!(
            report["memory_plan"]["wired_shortfall_bytes"],
            24_200_000_000u64
        );
        assert_eq!(report["memory_plan"]["live_sequence_cap"], 6);
    }

    #[test]
    fn memory_report_includes_sampled_at_and_footprint_fields() {
        let report = memory_report();
        let sampled_at = report["sampled_at"].as_str().expect("sampled_at");
        assert!(
            sampled_at.contains('T') && sampled_at.ends_with('Z'),
            "{sampled_at}"
        );
        assert_paired_footprint(
            &report["phys_footprint_bytes"],
            &report["lifetime_max_phys_footprint_bytes"],
        );
        #[cfg(feature = "mlx-native-server")]
        {
            assert!(report.get("mlx").is_some());
        }
        #[cfg(not(feature = "mlx-native-server"))]
        {
            assert!(report.get("mlx").is_none());
        }
    }
}
