//! Long-run soak test runner.
//!
//! Runs continuous inference for a specified duration and checks:
//! - RSS drift < threshold (default 5%)
//! - P95 latency drift < threshold (default 5%)
//! - No crashes or hangs
//!
//! Supports durations from quick smoke tests (5 min) to full nightly
//! validation (24h). Results are serializable to JSON for CI integration.

use std::path::Path;
use std::time::{Duration, Instant};

use ax_core::gguf::MappedModel;
use ax_core::metrics::{LatencyHistogram, current_rss_bytes};
use ax_core::model::{
    DecodeControl, DecodeIntent, DecodeRunConfig, LlamaModel, ModelConfig, WeightStore, run_decode,
};
use ax_core::sampling::{Sampler, SamplingConfig};
use ax_core::tokenizer::Tokenizer;
use serde::Serialize;

/// Soak test configuration.
pub struct SoakConfig {
    /// Model path (GGUF file).
    pub model_path: String,
    /// Total test duration.
    pub duration: Duration,
    /// Maximum acceptable RSS drift (fraction, e.g. 0.05 = 5%).
    pub max_rss_drift: f64,
    /// Maximum acceptable P95 latency drift (fraction).
    pub max_p95_drift: f64,
    /// Tokens to generate per iteration.
    pub tokens_per_iter: usize,
    /// Interval between drift checks.
    pub check_interval: Duration,
}

impl Default for SoakConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            duration: Duration::from_secs(8 * 3600), // 8 hours
            max_rss_drift: 0.05,
            max_p95_drift: 0.05,
            tokens_per_iter: 128,
            check_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl SoakConfig {
    /// 24-hour nightly soak test configuration.
    pub fn nightly() -> Self {
        Self {
            duration: Duration::from_secs(24 * 3600),
            check_interval: Duration::from_secs(600), // 10 minutes
            ..Default::default()
        }
    }

    /// Quick smoke test (5 minutes) for CI pre-merge.
    pub fn smoke() -> Self {
        Self {
            duration: Duration::from_secs(300),
            check_interval: Duration::from_secs(60),
            tokens_per_iter: 64,
            ..Default::default()
        }
    }
}

/// Result of a soak test run.
#[derive(Debug, Serialize)]
pub struct SoakResult {
    /// Total tokens generated.
    pub total_tokens: u64,
    /// Total iterations completed.
    pub iterations: u64,
    /// Total wall-clock duration in seconds.
    pub elapsed_secs: f64,
    /// Baseline RSS (bytes, measured at start).
    pub baseline_rss: u64,
    /// Final RSS (bytes).
    pub final_rss: u64,
    /// RSS drift fraction (final / baseline - 1.0).
    pub rss_drift: f64,
    /// Baseline P95 latency in milliseconds.
    pub baseline_p95_ms: f64,
    /// Final P95 latency in milliseconds.
    pub final_p95_ms: f64,
    /// P95 drift fraction.
    pub p95_drift: f64,
    /// Whether the test passed all criteria.
    pub passed: bool,
    /// Failure reasons (empty if passed).
    pub failures: Vec<String>,
    /// Average decode throughput (tokens/second).
    pub avg_tok_per_sec: f64,
}

/// Run a soak test with the given configuration.
///
/// Returns `SoakResult` with drift measurements and pass/fail status.
pub fn run_soak_test(config: &SoakConfig) -> anyhow::Result<SoakResult> {
    run_soak_test_with_backend(
        config,
        ax_core::backend::create_backend(ax_core::backend::BackendConfig::default())?,
    )
}

pub fn run_soak_test_with_backend(
    config: &SoakConfig,
    backend: Box<dyn ax_core::backend::Backend>,
) -> anyhow::Result<SoakResult> {
    // Load model
    let mapped = MappedModel::open(Path::new(&config.model_path))?;
    crate::init_kernel_profile(&config.model_path, &mapped);
    let model_config = ModelConfig::from_gguf(&mapped.header)?;
    let tokenizer = Tokenizer::from_gguf(&mapped.header)?;
    let model = LlamaModel::with_backend(model_config.clone(), backend);
    let weights = WeightStore::new(&mapped);

    let vocab_size = model_config.vocab_size as usize;
    let sampling = SamplingConfig::default();
    let mut sampler = Sampler::new(sampling);

    eprintln!(
        "Soak test: {} layers, {:.0}MB, target {}h",
        model_config.n_layers,
        mapped.total_tensor_bytes() as f64 / 1024.0 / 1024.0,
        config.duration.as_secs() / 3600,
    );

    // Warmup: run one iteration to stabilize RSS
    let warmup_tokens = run_one_iteration(
        &model,
        &weights,
        &tokenizer,
        &mut sampler,
        vocab_size,
        config.tokens_per_iter,
    )?;
    eprintln!("Warmup: {warmup_tokens} tokens generated");

    // Measure baseline
    let baseline_rss = current_rss_bytes();
    let mut baseline_latency = LatencyHistogram::new();
    run_timed_iteration(
        &model,
        &weights,
        &tokenizer,
        &mut sampler,
        vocab_size,
        config.tokens_per_iter,
        &mut baseline_latency,
    )?;
    let baseline_p95 = baseline_latency.p95().unwrap_or(Duration::ZERO);

    eprintln!(
        "Baseline: RSS {:.1}MB, P95 {:.2}ms",
        baseline_rss as f64 / 1024.0 / 1024.0,
        baseline_p95.as_secs_f64() * 1000.0,
    );

    // Main soak loop
    let start = Instant::now();
    let mut total_tokens = warmup_tokens + config.tokens_per_iter as u64;
    let mut iterations = 2u64; // warmup + baseline
    let mut last_check = Instant::now();
    let mut current_latency = LatencyHistogram::new();
    let mut failures = Vec::new();

    while start.elapsed() < config.duration {
        run_timed_iteration(
            &model,
            &weights,
            &tokenizer,
            &mut sampler,
            vocab_size,
            config.tokens_per_iter,
            &mut current_latency,
        )?;

        total_tokens += config.tokens_per_iter as u64;
        iterations += 1;

        // Periodic drift check
        if last_check.elapsed() >= config.check_interval {
            let now_rss = current_rss_bytes();
            let rss_drift = if baseline_rss > 0 {
                (now_rss as f64 / baseline_rss as f64) - 1.0
            } else {
                0.0
            };

            let now_p95 = current_latency.p95().unwrap_or(Duration::ZERO);
            let p95_drift = if baseline_p95 > Duration::ZERO {
                (now_p95.as_secs_f64() / baseline_p95.as_secs_f64()) - 1.0
            } else {
                0.0
            };

            let elapsed_h = start.elapsed().as_secs_f64() / 3600.0;
            eprintln!(
                "[{elapsed_h:.1}h] iter={iterations}, tok={total_tokens}, RSS drift {rss_drift:+.1}%, P95 drift {p95_drift:+.1}%",
                rss_drift = rss_drift * 100.0,
                p95_drift = p95_drift * 100.0,
            );

            if rss_drift > config.max_rss_drift {
                failures.push(format!(
                    "RSS drift {:.1}% exceeds {:.0}% threshold at {elapsed_h:.1}h",
                    rss_drift * 100.0,
                    config.max_rss_drift * 100.0,
                ));
            }
            if p95_drift > config.max_p95_drift {
                failures.push(format!(
                    "P95 drift {:.1}% exceeds {:.0}% threshold at {elapsed_h:.1}h",
                    p95_drift * 100.0,
                    config.max_p95_drift * 100.0,
                ));
            }

            // Reset latency window for next interval
            current_latency.clear();
            last_check = Instant::now();
        }
    }

    let final_rss = current_rss_bytes();
    let rss_drift = if baseline_rss > 0 {
        (final_rss as f64 / baseline_rss as f64) - 1.0
    } else {
        0.0
    };
    let final_p95 = current_latency
        .p95()
        .or(baseline_latency.p95())
        .unwrap_or(Duration::ZERO);
    let p95_drift = if baseline_p95 > Duration::ZERO {
        (final_p95.as_secs_f64() / baseline_p95.as_secs_f64()) - 1.0
    } else {
        0.0
    };

    let passed = failures.is_empty();
    let elapsed = start.elapsed();
    let avg_tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        total_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    Ok(SoakResult {
        total_tokens,
        iterations,
        elapsed_secs: elapsed.as_secs_f64(),
        baseline_rss,
        final_rss,
        rss_drift,
        baseline_p95_ms: baseline_p95.as_secs_f64() * 1000.0,
        final_p95_ms: final_p95.as_secs_f64() * 1000.0,
        p95_drift,
        passed,
        failures,
        avg_tok_per_sec,
    })
}

/// Run one inference iteration (prefill + decode), returns tokens generated.
fn run_one_iteration(
    model: &LlamaModel,
    weights: &WeightStore,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    vocab_size: usize,
    max_tokens: usize,
) -> anyhow::Result<u64> {
    let mut kv = model.create_model_kv();
    let mut logits = vec![0.0f32; vocab_size];

    // Use a simple prompt
    let prompt_tokens = tokenizer.encode("The meaning of life is", true);

    model.forward_batch(&prompt_tokens, &mut kv, weights, &mut logits)?;

    let mut history = prompt_tokens;
    let first_token = sampler.sample(&mut logits, &history);
    let position = history.len();
    let outcome = run_decode(
        model,
        weights,
        tokenizer,
        &mut kv,
        sampler,
        &mut history,
        first_token,
        None,
        position,
        max_tokens,
        DecodeRunConfig {
            intent: DecodeIntent::Throughput,
            allow_pipelined: true,
            top_logprobs: 0,
        },
        |_tok, _info| Ok(DecodeControl::Continue),
    )?;

    Ok(outcome.generated_tokens)
}

/// Run one iteration with per-token latency tracking.
fn run_timed_iteration(
    model: &LlamaModel,
    weights: &WeightStore,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    vocab_size: usize,
    max_tokens: usize,
    latency: &mut LatencyHistogram,
) -> anyhow::Result<()> {
    let mut kv = model.create_model_kv();
    let mut logits = vec![0.0f32; vocab_size];

    let prompt_tokens = tokenizer.encode("The meaning of life is", true);

    model.forward_batch(&prompt_tokens, &mut kv, weights, &mut logits)?;

    let mut history = prompt_tokens;
    let first_token = sampler.sample(&mut logits, &history);
    let position = history.len();
    let outcome = run_decode(
        model,
        weights,
        tokenizer,
        &mut kv,
        sampler,
        &mut history,
        first_token,
        None,
        position,
        max_tokens,
        DecodeRunConfig {
            intent: DecodeIntent::Latency,
            allow_pipelined: true,
            top_logprobs: 0,
        },
        |_tok, _info| Ok(DecodeControl::Continue),
    )?;
    for sample in outcome.latencies {
        latency.record(sample);
    }

    Ok(())
}

impl SoakResult {
    /// Print a human-readable summary.
    pub fn print_summary(&self) {
        let hours = self.elapsed_secs / 3600.0;
        eprintln!();
        eprintln!("=== Soak Test Results ===");
        eprintln!("Duration:    {hours:.2}h");
        eprintln!("Iterations:  {}", self.iterations);
        eprintln!("Tokens:      {}", self.total_tokens);
        eprintln!("Throughput:  {:.1} tok/s", self.avg_tok_per_sec);
        eprintln!(
            "RSS:         {:.1}MB → {:.1}MB (drift {:+.1}%)",
            self.baseline_rss as f64 / 1024.0 / 1024.0,
            self.final_rss as f64 / 1024.0 / 1024.0,
            self.rss_drift * 100.0,
        );
        eprintln!(
            "P95 latency: {:.2}ms → {:.2}ms (drift {:+.1}%)",
            self.baseline_p95_ms,
            self.final_p95_ms,
            self.p95_drift * 100.0,
        );
        if self.passed {
            eprintln!("Result:      PASSED");
        } else {
            eprintln!("Result:      FAILED");
            for f in &self.failures {
                eprintln!("  - {f}");
            }
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soak_config_defaults() {
        let c = SoakConfig::default();
        assert_eq!(c.duration, Duration::from_secs(8 * 3600));
        assert!((c.max_rss_drift - 0.05).abs() < f64::EPSILON);
        assert!((c.max_p95_drift - 0.05).abs() < f64::EPSILON);
        assert_eq!(c.tokens_per_iter, 128);
    }

    #[test]
    fn test_soak_config_nightly() {
        let c = SoakConfig::nightly();
        assert_eq!(c.duration, Duration::from_secs(24 * 3600));
        assert_eq!(c.check_interval, Duration::from_secs(600));
    }

    #[test]
    fn test_soak_config_smoke() {
        let c = SoakConfig::smoke();
        assert_eq!(c.duration, Duration::from_secs(300));
        assert_eq!(c.tokens_per_iter, 64);
    }

    #[test]
    fn test_soak_result_passed() {
        let r = SoakResult {
            total_tokens: 1000,
            iterations: 10,
            elapsed_secs: 100.0,
            baseline_rss: 100_000_000,
            final_rss: 102_000_000,
            rss_drift: 0.02,
            baseline_p95_ms: 50.0,
            final_p95_ms: 51.0,
            p95_drift: 0.02,
            passed: true,
            failures: vec![],
            avg_tok_per_sec: 10.0,
        };
        assert!(r.passed);
        assert!(r.failures.is_empty());
    }

    #[test]
    fn test_soak_result_failed() {
        let r = SoakResult {
            total_tokens: 1000,
            iterations: 10,
            elapsed_secs: 100.0,
            baseline_rss: 100_000_000,
            final_rss: 120_000_000,
            rss_drift: 0.20,
            baseline_p95_ms: 50.0,
            final_p95_ms: 50.0,
            p95_drift: 0.0,
            passed: false,
            failures: vec!["RSS drift 20.0% exceeds 5% threshold".into()],
            avg_tok_per_sec: 10.0,
        };
        assert!(!r.passed);
        assert_eq!(r.failures.len(), 1);
    }

    #[test]
    fn test_soak_result_to_json() {
        let r = SoakResult {
            total_tokens: 5000,
            iterations: 50,
            elapsed_secs: 3600.0,
            baseline_rss: 100_000_000,
            final_rss: 103_000_000,
            rss_drift: 0.03,
            baseline_p95_ms: 20.5,
            final_p95_ms: 21.0,
            p95_drift: 0.024,
            passed: true,
            failures: vec![],
            avg_tok_per_sec: 1.39,
        };
        let json = r.to_json().unwrap();
        assert!(json.contains("\"passed\": true"));
        assert!(json.contains("\"total_tokens\": 5000"));
        assert!(json.contains("\"avg_tok_per_sec\""));
    }
}
