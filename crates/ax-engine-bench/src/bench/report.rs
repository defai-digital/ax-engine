//! Benchmark results formatting and output.

use crate::baseline::ComparisonResult;
use crate::perf::BenchResult;

/// Format a single benchmark result as a table row.
pub fn format_bench_row(result: &BenchResult) -> String {
    format!(
        "{:<30} {:>8} tok {:>10.1} tok/s {:>8} tok {:>10.1} tok/s  {:<10} {:<10}",
        result.model,
        result.prompt_tokens,
        result.prefill_tok_per_sec_median,
        result.decode_tokens,
        result.decode_tok_per_sec_median,
        result.decode_intent,
        result.decode_mode,
    )
}

/// Print a comparison table.
pub fn print_comparison_table(results: &[ComparisonResult]) {
    eprintln!();
    eprintln!("=== Performance Comparison vs llama.cpp ===");
    eprintln!(
        "{:<20} {:>12} {:>12}   Status",
        "Config", "Prefill", "Decode"
    );
    eprintln!("{}", "-".repeat(60));
    for r in results {
        r.print_row();
    }

    let all_pass = results.iter().all(|r| r.meets_target);
    eprintln!("{}", "-".repeat(60));
    if all_pass {
        eprintln!("Overall: PASSED (all entries >= 95% of baseline)");
    } else {
        let failed = results.iter().filter(|r| !r.meets_target).count();
        eprintln!("Overall: FAILED ({failed} entries below 95% threshold)");
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_format_bench_row() {
        let result = BenchResult {
            model: "test-model".into(),
            prompt_tokens: 512,
            effective_prefill_tokens: 512,
            decode_tokens: 128,
            prefill_tok_per_sec: 1000.0,
            prefill_tok_per_sec_median: 1000.0,
            effective_prefill_tok_per_sec: 1000.0,
            effective_prefill_tok_per_sec_median: 1000.0,
            decode_tok_per_sec: 50.0,
            decode_tok_per_sec_median: 50.0,
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
            prefill_mode: "gpu_batch".into(),
            prefill_route_family: "dense_gpu_batch".into(),
            prefill_route_detail: "generic_gpu_batch".into(),
            prefill_attention_route: None,
            prefill_qkv_plan: None,
            prefill_split_rope_append: None,
            q5k_prefill_mode: None,
            decode_plan: "sync=single_cb scratch=gpu_shared".into(),
            support_note: None,
            decode_fallback_reason: None,
            kernel_profile_path: None,
            qwen35_shared_timeline_slots: 1,
            qwen35_shared_timeline_source_slot: None,
            kv_f16: true,
            deterministic: false,
            samples: 1,
            cooldown_ms: 0,
        };
        let row = format_bench_row(&result);
        assert!(row.contains("test-model"));
        assert!(row.contains("1000.0"));
        assert!(row.contains("50.0"));
    }
}
