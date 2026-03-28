//! llama.cpp baseline comparison.
//!
//! Runs llama.cpp with identical parameters and compares results
//! against AX Engine benchmarks.

use std::time::Duration;

use crate::perf::BenchResult;

/// Benchmark matrix entry: model + config for comparison.
#[derive(Debug, Clone)]
pub struct MatrixEntry {
    /// Label for this entry.
    pub label: String,
    /// Model path.
    pub model_path: String,
    /// Prompt size (tokens).
    pub prompt_tokens: usize,
    /// Decode count (tokens).
    pub decode_tokens: usize,
    /// Context length.
    pub context_length: usize,
}

/// Baseline numbers from llama.cpp for comparison.
#[derive(Debug, Clone)]
pub struct BaselineNumbers {
    pub label: String,
    pub prefill_tok_per_sec: f64,
    pub decode_tok_per_sec: f64,
    pub p95_latency: Duration,
}

/// Comparison result between AX Engine and baseline.
#[derive(Debug)]
pub struct ComparisonResult {
    pub label: String,
    pub ax_result: BenchResult,
    pub baseline: BaselineNumbers,
    /// AX prefill tok/s as fraction of baseline (1.0 = parity).
    pub prefill_ratio: f64,
    /// AX decode tok/s as fraction of baseline.
    pub decode_ratio: f64,
    /// Whether this entry meets the >= 95% target.
    pub meets_target: bool,
}

/// Standard benchmark matrix from the implementation plan.
pub fn standard_matrix() -> Vec<MatrixEntry> {
    vec![
        MatrixEntry {
            label: "7B/Q4_0/4k".into(),
            model_path: String::new(), // set by caller
            prompt_tokens: 512,
            decode_tokens: 128,
            context_length: 4096,
        },
        MatrixEntry {
            label: "20B/Q4_K/8k".into(),
            model_path: String::new(),
            prompt_tokens: 1024,
            decode_tokens: 256,
            context_length: 8192,
        },
        MatrixEntry {
            label: "70B/Q4_K/32k".into(),
            model_path: String::new(),
            prompt_tokens: 2048,
            decode_tokens: 512,
            context_length: 32768,
        },
    ]
}

/// Compare AX Engine results against baseline numbers.
pub fn compare(ax: &BenchResult, baseline: &BaselineNumbers) -> ComparisonResult {
    let prefill_ratio = if baseline.prefill_tok_per_sec > 0.0 {
        ax.prefill_tok_per_sec / baseline.prefill_tok_per_sec
    } else {
        0.0
    };
    let decode_ratio = if baseline.decode_tok_per_sec > 0.0 {
        ax.decode_tok_per_sec / baseline.decode_tok_per_sec
    } else {
        0.0
    };

    // Pass if both prefill and decode are >= 95% of baseline
    let meets_target = prefill_ratio >= 0.95 && decode_ratio >= 0.95;

    ComparisonResult {
        label: baseline.label.clone(),
        ax_result: BenchResult {
            model: ax.model.clone(),
            prompt_tokens: ax.prompt_tokens,
            decode_tokens: ax.decode_tokens,
            prefill_tok_per_sec: ax.prefill_tok_per_sec,
            prefill_tok_per_sec_median: ax.prefill_tok_per_sec_median,
            decode_tok_per_sec: ax.decode_tok_per_sec,
            decode_tok_per_sec_median: ax.decode_tok_per_sec_median,
            p50_latency: ax.p50_latency,
            p95_latency: ax.p95_latency,
            p99_latency: ax.p99_latency,
            prefill_command_buffers: ax.prefill_command_buffers,
            prefill_buffer_barriers: ax.prefill_buffer_barriers,
            decode_command_buffers: ax.decode_command_buffers,
            decode_buffer_barriers: ax.decode_buffer_barriers,
            decode_command_buffers_per_tok: ax.decode_command_buffers_per_tok,
            decode_buffer_barriers_per_tok: ax.decode_buffer_barriers_per_tok,
            decode_command_buffers_per_tok_max: ax.decode_command_buffers_per_tok_max,
            decode_buffer_barriers_per_tok_max: ax.decode_buffer_barriers_per_tok_max,
            decode_intent: ax.decode_intent.clone(),
            decode_mode: ax.decode_mode.clone(),
            prefill_plan: ax.prefill_plan.clone(),
            q5k_prefill_mode: ax.q5k_prefill_mode.clone(),
            decode_plan: ax.decode_plan.clone(),
            support_note: ax.support_note.clone(),
            decode_fallback_reason: ax.decode_fallback_reason.clone(),
            kernel_profile_path: ax.kernel_profile_path.clone(),
            kv_f16: ax.kv_f16,
            deterministic: ax.deterministic,
            samples: ax.samples,
            cooldown_ms: ax.cooldown_ms,
        },
        baseline: baseline.clone(),
        prefill_ratio,
        decode_ratio,
        meets_target,
    }
}

impl ComparisonResult {
    /// Print a comparison summary line.
    pub fn print_row(&self) {
        let status = if self.meets_target { "PASS" } else { "FAIL" };
        eprintln!(
            "{:<20} prefill {:.0}% | decode {:.0}% | [{}]",
            self.label,
            self.prefill_ratio * 100.0,
            self.decode_ratio * 100.0,
            status,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_bench() -> BenchResult {
        BenchResult {
            model: "test".into(),
            prompt_tokens: 512,
            decode_tokens: 128,
            prefill_tok_per_sec: 950.0,
            prefill_tok_per_sec_median: 950.0,
            decode_tok_per_sec: 48.0,
            decode_tok_per_sec_median: 48.0,
            p50_latency: Duration::from_millis(20),
            p95_latency: Duration::from_millis(25),
            p99_latency: Duration::from_millis(30),
            prefill_command_buffers: 0.0,
            prefill_buffer_barriers: 0.0,
            decode_command_buffers: 0.0,
            decode_buffer_barriers: 0.0,
            decode_command_buffers_per_tok: 0.0,
            decode_buffer_barriers_per_tok: 0.0,
            decode_command_buffers_per_tok_max: 0.0,
            decode_buffer_barriers_per_tok_max: 0.0,
            decode_intent: "latency".into(),
            decode_mode: "single_cb".into(),
            prefill_plan: "mode=gpu_batch".into(),
            q5k_prefill_mode: None,
            decode_plan: "sync=single_cb scratch=gpu_shared".into(),
            support_note: None,
            decode_fallback_reason: None,
            kernel_profile_path: None,
            deterministic: false,
            samples: 1,
            kv_f16: true,
            cooldown_ms: 0,
        }
    }

    #[test]
    fn test_compare_passes() {
        let ax = dummy_bench();
        let baseline = BaselineNumbers {
            label: "7B/Q4_0/4k".into(),
            prefill_tok_per_sec: 1000.0,
            decode_tok_per_sec: 50.0,
            p95_latency: Duration::from_millis(25),
        };
        let result = compare(&ax, &baseline);
        // 950/1000 = 0.95, 48/50 = 0.96 → both >= 0.95
        assert!(
            result.meets_target,
            "prefill={:.2}, decode={:.2}",
            result.prefill_ratio, result.decode_ratio
        );
    }

    #[test]
    fn test_compare_fails() {
        let mut ax = dummy_bench();
        ax.prefill_tok_per_sec = 900.0; // 90% of baseline
        let baseline = BaselineNumbers {
            label: "7B/Q4_0/4k".into(),
            prefill_tok_per_sec: 1000.0,
            decode_tok_per_sec: 50.0,
            p95_latency: Duration::from_millis(25),
        };
        let result = compare(&ax, &baseline);
        assert!(!result.meets_target);
    }

    #[test]
    fn test_standard_matrix_has_entries() {
        let matrix = standard_matrix();
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].label, "7B/Q4_0/4k");
        assert_eq!(matrix[1].label, "20B/Q4_K/8k");
        assert_eq!(matrix[2].label, "70B/Q4_K/32k");
    }
}
