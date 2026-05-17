//! Latency and throughput collection for serving-shaped workload fixtures.
//!
//! See `.internal/prd/engine-serving-invariants.md` (Phase 1). This module owns
//! the percentile math and report shape that workload fixtures emit. It is
//! intentionally independent of `crates/ax-engine-bench/src/stats.rs`, whose
//! percentile helpers are gated behind `#[cfg(test)]` and not part of the
//! production bench surface.

use std::cmp::Ordering;
use std::time::Duration;

use serde::Serialize;
use serde_json::{Map, Value, json};

/// A growable bucket of latency samples in microseconds.
#[derive(Debug, Clone)]
pub struct LatencySamples {
    name: String,
    values_us: Vec<u64>,
}

impl LatencySamples {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values_us: Vec::new(),
        }
    }

    /// Pre-allocated constructor used by Phase 5 fixtures that know their
    /// sample count up front (concurrent_short_inserts, partial_prefix_hit).
    #[allow(dead_code)]
    pub fn with_capacity(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            values_us: Vec::with_capacity(capacity),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn record_us(&mut self, value: u64) {
        self.values_us.push(value);
    }

    pub fn record_duration(&mut self, duration: Duration) {
        let micros = duration.as_micros().min(u128::from(u64::MAX)) as u64;
        self.record_us(micros);
    }

    /// Sample count. Not consumed by the current Phase 1 driver; used by
    /// PRD Phase 5 aggregation and by unit tests.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.values_us.len()
    }

    /// Mirror of [`Self::len`] for ergonomic emptiness checks.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values_us.is_empty()
    }

    pub fn min_us(&self) -> Option<u64> {
        self.values_us.iter().copied().min()
    }

    pub fn max_us(&self) -> Option<u64> {
        self.values_us.iter().copied().max()
    }

    pub fn mean_us(&self) -> Option<f64> {
        if self.values_us.is_empty() {
            return None;
        }
        let sum: u128 = self.values_us.iter().map(|v| u128::from(*v)).sum();
        Some(sum as f64 / self.values_us.len() as f64)
    }

    /// Nearest-rank percentile in microseconds. `quantile` is clamped to `[0.0, 1.0]`.
    pub fn percentile_us(&self, quantile: f64) -> Option<u64> {
        if self.values_us.is_empty() {
            return None;
        }
        let mut sorted = self.values_us.clone();
        sorted.sort_unstable();
        let quantile = quantile.clamp(0.0, 1.0);
        let last = sorted.len() - 1;
        let index = ((last as f64) * quantile).round() as usize;
        sorted.get(index).copied()
    }

    pub fn p50_us(&self) -> Option<u64> {
        self.percentile_us(0.50)
    }

    pub fn p95_us(&self) -> Option<u64> {
        self.percentile_us(0.95)
    }

    pub fn p99_us(&self) -> Option<u64> {
        self.percentile_us(0.99)
    }

    /// Serialize a summary view (counts + key percentiles) suitable for artifact JSON.
    /// The raw sample vector is omitted from the summary to keep artifact size bounded.
    pub fn to_summary_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("name".to_string(), Value::String(self.name.clone()));
        map.insert("count".to_string(), json!(self.values_us.len()));
        map.insert("min_us".to_string(), to_json_opt_u64(self.min_us()));
        map.insert("max_us".to_string(), to_json_opt_u64(self.max_us()));
        map.insert("mean_us".to_string(), to_json_opt_f64(self.mean_us()));
        map.insert("p50_us".to_string(), to_json_opt_u64(self.p50_us()));
        map.insert("p95_us".to_string(), to_json_opt_u64(self.p95_us()));
        map.insert("p99_us".to_string(), to_json_opt_u64(self.p99_us()));
        Value::Object(map)
    }

    #[cfg(test)]
    pub(crate) fn raw_values(&self) -> &[u64] {
        &self.values_us
    }
}

fn to_json_opt_u64(value: Option<u64>) -> Value {
    value.map(|v| json!(v)).unwrap_or(Value::Null)
}

fn to_json_opt_f64(value: Option<f64>) -> Value {
    value.map(|v| json!(v)).unwrap_or(Value::Null)
}

/// Stable identity for the latency channel a sample belongs to.
///
/// Workload fixtures emit named latency buckets; this enum names the contracts
/// PRD §7.2 requires every fixture to report on. Fixtures may add extra
/// auxiliary buckets via `WorkloadReport::add_extra_samples`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyChannel {
    /// Time-to-first-token for foreground decode requests.
    ForegroundTtft,
    /// Inter-token latency for foreground decode requests under load.
    ForegroundItl,
    /// Wall time from cancel signal to fully reclaimed scheduler slot.
    CancellationTail,
    /// TTFT degradation for short requests inserted during long-request load.
    ConcurrentShortInsertDelta,
}

impl LatencyChannel {
    pub fn channel_name(self) -> &'static str {
        match self {
            LatencyChannel::ForegroundTtft => "foreground_ttft",
            LatencyChannel::ForegroundItl => "foreground_itl",
            LatencyChannel::CancellationTail => "cancellation_tail",
            LatencyChannel::ConcurrentShortInsertDelta => "concurrent_short_insert_delta",
        }
    }
}

/// Single fixture's measurement bundle.
#[derive(Debug, Clone)]
pub struct WorkloadReport {
    workload: String,
    started_at_unix_secs: u64,
    elapsed_us: u64,
    pub foreground_ttft: LatencySamples,
    pub foreground_itl: LatencySamples,
    pub cancellation_tail: LatencySamples,
    pub concurrent_short_insert_delta: LatencySamples,
    extra_samples: Vec<LatencySamples>,
    post_restart_cache: PostRestartCacheCounts,
    notes: Vec<String>,
    decisions: Vec<(String, u64)>,
}

impl WorkloadReport {
    pub fn new(workload: impl Into<String>) -> Self {
        let started_at_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let workload = workload.into();
        Self {
            foreground_ttft: LatencySamples::new(LatencyChannel::ForegroundTtft.channel_name()),
            foreground_itl: LatencySamples::new(LatencyChannel::ForegroundItl.channel_name()),
            cancellation_tail: LatencySamples::new(LatencyChannel::CancellationTail.channel_name()),
            concurrent_short_insert_delta: LatencySamples::new(
                LatencyChannel::ConcurrentShortInsertDelta.channel_name(),
            ),
            workload,
            started_at_unix_secs,
            elapsed_us: 0,
            extra_samples: Vec::new(),
            post_restart_cache: PostRestartCacheCounts::default(),
            notes: Vec::new(),
            decisions: Vec::new(),
        }
    }

    /// Read accessor; consumed by Phase 5 aggregation and tests.
    #[allow(dead_code)]
    pub fn workload(&self) -> &str {
        &self.workload
    }

    pub fn record_elapsed(&mut self, elapsed: Duration) {
        self.elapsed_us = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
    }

    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    pub fn add_decision(&mut self, key: impl Into<String>, value: u64) {
        self.decisions.push((key.into(), value));
    }

    /// Attach an auxiliary latency channel to the report. Used by Phase 5
    /// fixtures (e.g. partial_prefix_hit) that report bespoke samples
    /// alongside the standard TTFT/ITL/cancellation/short-insert channels.
    #[allow(dead_code)]
    pub fn add_extra_samples(&mut self, samples: LatencySamples) {
        self.extra_samples.push(samples);
    }

    /// Mutable view into post-restart cache counters. Used by Phase 2
    /// `post_restart_cache_safety` fixture.
    #[allow(dead_code)]
    pub fn post_restart_cache_mut(&mut self) -> &mut PostRestartCacheCounts {
        &mut self.post_restart_cache
    }

    pub fn to_json(&self) -> Value {
        let mut extras = Map::new();
        for sample in &self.extra_samples {
            extras.insert(sample.name().to_string(), sample.to_summary_json());
        }
        let decisions: Vec<Value> = self
            .decisions
            .iter()
            .map(|(k, v)| json!({"key": k, "value": v}))
            .collect();
        json!({
            "schema": "ax.serving_workload.report.v1",
            "workload": self.workload,
            "started_at_unix_secs": self.started_at_unix_secs,
            "elapsed_us": self.elapsed_us,
            "foreground_ttft": self.foreground_ttft.to_summary_json(),
            "foreground_itl": self.foreground_itl.to_summary_json(),
            "cancellation_tail": self.cancellation_tail.to_summary_json(),
            "concurrent_short_insert_delta": self.concurrent_short_insert_delta.to_summary_json(),
            "post_restart_cache": self.post_restart_cache.to_json(),
            "extras": Value::Object(extras),
            "decisions": decisions,
            "notes": self.notes,
        })
    }
}

/// Counts emitted by post-restart cache safety fixtures.
///
/// See PRD §7.2: rejection reasons are tracked discretely so the artifact
/// distinguishes a legitimate miss from a fail-closed mismatch rejection.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PostRestartCacheCounts {
    pub hits: u64,
    pub misses: u64,
    pub rejected_model_mismatch: u64,
    pub rejected_policy_mismatch: u64,
    pub rejected_layout_mismatch: u64,
    pub rejected_block_size_mismatch: u64,
    pub rejected_token_payload_mismatch: u64,
    pub rejected_format_version_mismatch: u64,
    pub rejected_dtype_mismatch: u64,
    pub rejected_other: u64,
}

impl PostRestartCacheCounts {
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Sum of all rejection counters. Used by PRD Phase 2 post-restart cache
    /// safety fixtures and by unit tests; not consumed by the current
    /// `long_prefill_vs_decode` driver.
    #[allow(dead_code)]
    pub fn total_rejections(&self) -> u64 {
        self.rejected_model_mismatch
            + self.rejected_policy_mismatch
            + self.rejected_layout_mismatch
            + self.rejected_block_size_mismatch
            + self.rejected_token_payload_mismatch
            + self.rejected_format_version_mismatch
            + self.rejected_dtype_mismatch
            + self.rejected_other
    }
}

/// Comparator helper used by tests to sort latency buckets predictably.
#[cfg(test)]
pub(crate) fn cmp_u64(a: &u64, b: &u64) -> Ordering {
    a.cmp(b)
}

// Keep a non-test reference to silence unused import warnings on Ordering when
// the cfg(test) helper is not compiled in.
#[allow(dead_code)]
fn _ordering_marker() -> Ordering {
    Ordering::Equal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_samples_report_no_percentile() {
        let s = LatencySamples::new("empty");
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.min_us(), None);
        assert_eq!(s.max_us(), None);
        assert_eq!(s.mean_us(), None);
        assert_eq!(s.p50_us(), None);
        assert_eq!(s.p95_us(), None);
        assert_eq!(s.p99_us(), None);
    }

    #[test]
    fn single_sample_returns_itself_for_all_percentiles() {
        let mut s = LatencySamples::new("single");
        s.record_us(1_234);
        assert_eq!(s.len(), 1);
        assert_eq!(s.min_us(), Some(1_234));
        assert_eq!(s.max_us(), Some(1_234));
        assert_eq!(s.p50_us(), Some(1_234));
        assert_eq!(s.p95_us(), Some(1_234));
        assert_eq!(s.p99_us(), Some(1_234));
    }

    #[test]
    fn percentile_nearest_rank_against_known_distribution() {
        // 100 samples: 1..=100us. p50 nearest-rank lands at index round(99 * 0.5)
        // = 50 (zero-based) = value 51us. p95 -> index 94 -> value 95us.
        let mut s = LatencySamples::new("ramp");
        for i in 1..=100 {
            s.record_us(i);
        }
        assert_eq!(s.min_us(), Some(1));
        assert_eq!(s.max_us(), Some(100));
        assert_eq!(s.p50_us(), Some(51));
        assert_eq!(s.p95_us(), Some(95));
        assert_eq!(s.p99_us(), Some(99));
    }

    #[test]
    fn percentile_clamps_quantile_out_of_range() {
        let mut s = LatencySamples::new("clamp");
        for i in [10u64, 20, 30, 40, 50] {
            s.record_us(i);
        }
        assert_eq!(s.percentile_us(-1.0), Some(10));
        assert_eq!(s.percentile_us(2.0), Some(50));
    }

    #[test]
    fn percentile_ignores_input_order() {
        let mut a = LatencySamples::new("a");
        let mut b = LatencySamples::new("b");
        for v in [50u64, 10, 30, 40, 20] {
            a.record_us(v);
        }
        for v in [10u64, 20, 30, 40, 50] {
            b.record_us(v);
        }
        assert_eq!(a.p50_us(), b.p50_us());
        assert_eq!(a.p95_us(), b.p95_us());
        assert_eq!(a.p99_us(), b.p99_us());

        // Sanity: raw vector preserves insertion order, not sorted order.
        let mut raw = a.raw_values().to_vec();
        let initial = raw.clone();
        raw.sort_by(cmp_u64);
        assert_ne!(initial, raw);
    }

    #[test]
    fn mean_matches_arithmetic_mean() {
        let mut s = LatencySamples::new("mean");
        for v in [10u64, 20, 30, 40, 50] {
            s.record_us(v);
        }
        assert_eq!(s.mean_us(), Some(30.0));
    }

    #[test]
    fn duration_recording_saturates_at_u64_max() {
        let mut s = LatencySamples::new("duration");
        s.record_duration(Duration::from_secs(1));
        s.record_duration(Duration::from_micros(500));
        let mut sorted = s.raw_values().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![500, 1_000_000]);
    }

    #[test]
    fn summary_json_shape_is_stable() {
        let mut s = LatencySamples::new("summary");
        for v in 1..=10 {
            s.record_us(v);
        }
        let summary = s.to_summary_json();
        let obj = summary.as_object().expect("summary is object");
        for key in [
            "name", "count", "min_us", "max_us", "mean_us", "p50_us", "p95_us", "p99_us",
        ] {
            assert!(obj.contains_key(key), "summary missing key: {key}");
        }
        assert_eq!(obj["count"], json!(10));
        assert_eq!(obj["min_us"], json!(1));
        assert_eq!(obj["max_us"], json!(10));
    }

    #[test]
    fn workload_report_round_trips_to_json() {
        let mut report = WorkloadReport::new("test_fixture");
        report.foreground_ttft.record_us(100);
        report.foreground_ttft.record_us(200);
        report.foreground_itl.record_us(50);
        report.cancellation_tail.record_us(1_000);
        report.add_note("scaffold");
        report.add_decision("ax_pressure_level", 1);
        report.record_elapsed(Duration::from_millis(42));

        let value = report.to_json();
        assert_eq!(value["schema"], json!("ax.serving_workload.report.v1"));
        assert_eq!(value["workload"], json!("test_fixture"));
        assert_eq!(value["foreground_ttft"]["count"], json!(2));
        assert_eq!(value["foreground_itl"]["count"], json!(1));
        assert_eq!(value["cancellation_tail"]["count"], json!(1));
        assert_eq!(value["notes"][0], json!("scaffold"));
        assert_eq!(value["decisions"][0]["key"], json!("ax_pressure_level"));
        assert_eq!(value["decisions"][0]["value"], json!(1));
        assert!(value["elapsed_us"].as_u64().unwrap() >= 42_000);
    }

    #[test]
    fn post_restart_cache_counts_total_rejections() {
        let counts = PostRestartCacheCounts {
            hits: 9,
            misses: 7,
            rejected_model_mismatch: 2,
            rejected_layout_mismatch: 1,
            rejected_format_version_mismatch: 4,
            ..PostRestartCacheCounts::default()
        };
        assert_eq!(counts.total_rejections(), 7);
    }

    #[test]
    fn extra_samples_appear_in_json_under_extras() {
        let mut report = WorkloadReport::new("extras_fixture");
        let mut aux = LatencySamples::new("aux_channel");
        aux.record_us(100);
        aux.record_us(200);
        report.add_extra_samples(aux);

        let value = report.to_json();
        let extras = value["extras"].as_object().expect("extras is object");
        assert!(extras.contains_key("aux_channel"));
        assert_eq!(extras["aux_channel"]["count"], json!(2));
    }
}
