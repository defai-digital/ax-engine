//! `GET /v1/memory`: process physical footprint and, when the native MLX
//! runtime is linked, MLX allocator gauges. No model is loaded.

use std::time::{SystemTime, UNIX_EPOCH};

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
    report
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

    #[test]
    fn rfc3339_formats_known_unix_timestamps() {
        assert_eq!(rfc3339_from_unix(0, 0), "1970-01-01T00:00:00.000000000Z");
        assert_eq!(
            rfc3339_from_unix(1_609_459_200, 0),
            "2021-01-01T00:00:00.000000000Z"
        );
    }

    #[test]
    fn memory_report_includes_sampled_at_and_footprint_fields() {
        let report = memory_report();
        let sampled_at = report["sampled_at"].as_str().expect("sampled_at");
        assert!(
            sampled_at.contains('T') && sampled_at.ends_with('Z'),
            "{sampled_at}"
        );
        let footprint = report["phys_footprint_bytes"].as_u64();
        let peak = report["lifetime_max_phys_footprint_bytes"].as_u64();
        if let (Some(footprint), Some(peak)) = (footprint, peak) {
            assert!(peak >= footprint);
        }
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
