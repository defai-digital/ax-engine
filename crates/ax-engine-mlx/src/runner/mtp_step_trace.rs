//! Env-gated per-cycle MTP trace (`AX_MLX_MTP_STEP_TRACE=<path>`).
//!
//! Appends one JSON line per verified speculative cycle: the verify window,
//! the verifier's accepted prefix, the draft head's per-position log-probs
//! (present only when the draft path computed them, i.e. a non-zero draft
//! confidence gate), the context length, and a process-monotonic timestamp.
//! It feeds the offline verify-prefix simulator
//! (`scripts/simulate_mtp_verify_prefix.py`), which prices truncation
//! policies against measured cycle costs before any runtime policy changes.
//!
//! Capture-only diagnostic: one buffered line write per cycle, no extra
//! device sync (it reads only host values the cycle already produced). Unset
//! (the default) costs one cached `OnceLock` read per cycle.

use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// One verified cycle. Field names are the JSON keys.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct MtpStepTraceRecord<'a> {
    /// Microseconds since the trace sink opened.
    pub(super) t_us: u64,
    /// Committed context length (target cache tokens) before this cycle.
    pub(super) ctx: usize,
    /// Tokens generated for this request before this cycle.
    pub(super) generated: usize,
    /// Draft depth the controller requested for this cycle's draft.
    pub(super) depth: usize,
    /// Draft tokens submitted to the verifier.
    pub(super) used: usize,
    /// Leading draft tokens the verifier accepted.
    pub(super) accepted: usize,
    /// Every submitted draft came from the MTP head (no n-gram stacking).
    pub(super) pure_mtp: bool,
    /// Sampling temperature of the request (0 = greedy).
    pub(super) temperature: f32,
    pub(super) verify_forward_us: u32,
    pub(super) verify_eval_us: u32,
    pub(super) draft_us: u32,
    /// Draft head log-probs aligned with the submitted drafts (may be empty).
    pub(super) draft_log_probs: &'a [f32],
}

/// Serialize one record as a single JSON line (no trailing newline).
/// Non-finite log-probs are written as `null` so the line stays valid JSON.
pub(super) fn format_trace_line(record: &MtpStepTraceRecord<'_>) -> String {
    let mut line = String::with_capacity(256);
    line.push_str(&format!(
        "{{\"t_us\":{},\"ctx\":{},\"generated\":{},\"depth\":{},\"used\":{},\"accepted\":{},\
         \"pure_mtp\":{},\"temperature\":{},\"verify_forward_us\":{},\"verify_eval_us\":{},\
         \"draft_us\":{},\"draft_log_probs\":[",
        record.t_us,
        record.ctx,
        record.generated,
        record.depth,
        record.used,
        record.accepted,
        record.pure_mtp,
        json_f32(record.temperature),
        record.verify_forward_us,
        record.verify_eval_us,
        record.draft_us,
    ));
    for (index, value) in record.draft_log_probs.iter().enumerate() {
        if index > 0 {
            line.push(',');
        }
        line.push_str(&json_f32(*value));
    }
    line.push_str("]}");
    line
}

fn json_f32(value: f32) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_string()
    }
}

struct TraceSink {
    file: Mutex<std::fs::File>,
    opened: Instant,
}

fn trace_sink() -> Option<&'static TraceSink> {
    static SINK: OnceLock<Option<TraceSink>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = std::env::var("AX_MLX_MTP_STEP_TRACE").ok()?;
        let path = path.trim();
        if path.is_empty() {
            return None;
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(file) => Some(TraceSink {
                file: Mutex::new(file),
                opened: Instant::now(),
            }),
            Err(error) => {
                eprintln!("AX_MLX_MTP_STEP_TRACE: cannot open {path}: {error}");
                None
            }
        }
    })
    .as_ref()
}

/// Whether the trace sink is active (cheap after the first call).
pub(super) fn mtp_step_trace_enabled() -> bool {
    trace_sink().is_some()
}

/// Microseconds since the sink opened; 0 when tracing is off.
pub(super) fn mtp_step_trace_now_us() -> u64 {
    trace_sink().map_or(0, |sink| {
        u64::try_from(sink.opened.elapsed().as_micros()).unwrap_or(u64::MAX)
    })
}

/// Append one record. A write error is reported once per failure and never
/// affects decoding.
pub(super) fn record_mtp_step_trace(record: &MtpStepTraceRecord<'_>) {
    let Some(sink) = trace_sink() else {
        return;
    };
    let mut line = format_trace_line(record);
    line.push('\n');
    if let Ok(mut file) = sink.file.lock()
        && let Err(error) = file.write_all(line.as_bytes())
    {
        eprintln!("AX_MLX_MTP_STEP_TRACE: write failed: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(log_probs: &[f32]) -> MtpStepTraceRecord<'_> {
        MtpStepTraceRecord {
            t_us: 1234,
            ctx: 512,
            generated: 7,
            depth: 3,
            used: 3,
            accepted: 2,
            pure_mtp: true,
            temperature: 0.0,
            verify_forward_us: 100,
            verify_eval_us: 40_000,
            draft_us: 900,
            draft_log_probs: log_probs,
        }
    }

    #[test]
    fn trace_line_is_valid_json_with_all_fields() {
        let line = format_trace_line(&record(&[-0.01, -0.5, -2.25]));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["t_us"], 1234);
        assert_eq!(value["ctx"], 512);
        assert_eq!(value["generated"], 7);
        assert_eq!(value["depth"], 3);
        assert_eq!(value["used"], 3);
        assert_eq!(value["accepted"], 2);
        assert_eq!(value["pure_mtp"], true);
        assert_eq!(value["verify_eval_us"], 40_000);
        let log_probs = value["draft_log_probs"].as_array().expect("array");
        assert_eq!(log_probs.len(), 3);
        assert!((log_probs[2].as_f64().expect("number") + 2.25).abs() < 1e-6);
    }

    #[test]
    fn trace_line_writes_non_finite_log_probs_as_null() {
        let line = format_trace_line(&record(&[f32::NEG_INFINITY, f32::NAN]));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        let log_probs = value["draft_log_probs"].as_array().expect("array");
        assert!(log_probs.iter().all(serde_json::Value::is_null));
    }

    #[test]
    fn trace_line_handles_empty_log_probs() {
        let line = format_trace_line(&record(&[]));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert!(
            value["draft_log_probs"]
                .as_array()
                .expect("array")
                .is_empty()
        );
        assert!(!line.contains('\n'));
    }
}
