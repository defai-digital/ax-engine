//! ax-bench CLI — benchmark, profiler, and soak test runner for AX Engine.
//!
//! Usage:
//!   ax-bench soak --model <path> [--duration 24h] [--json]
//!   ax-bench bench --model <path> [--prompt-tokens 512] [--decode-tokens 128]
//!   ax-bench profile --model <path> [--profile-tokens 64] [--json]

use std::process;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};

use ax_bench::perf::{self, BenchConfig, SpecBenchConfig};
use ax_bench::profile::{self, ProfileConfig};
use ax_bench::soak::{self, SoakConfig};
use ax_core::model::DecodeIntent;

#[derive(Parser)]
#[command(name = "ax-bench", about = "AX Engine benchmark and soak test runner")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum BenchIntentArg {
    Throughput,
    Latency,
}

impl From<BenchIntentArg> for DecodeIntent {
    fn from(value: BenchIntentArg) -> Self {
        match value {
            BenchIntentArg::Throughput => DecodeIntent::Throughput,
            BenchIntentArg::Latency => DecodeIntent::Latency,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Run a long-duration soak test (stability validation).
    Soak {
        /// Path to GGUF model file.
        #[arg(long)]
        model: String,

        /// Test duration (e.g. "5m", "8h", "24h").
        #[arg(long, default_value = "8h")]
        duration: String,

        /// Maximum acceptable RSS drift percentage.
        #[arg(long, default_value = "5")]
        max_rss_drift: f64,

        /// Maximum acceptable P95 latency drift percentage.
        #[arg(long, default_value = "5")]
        max_p95_drift: f64,

        /// Tokens to generate per iteration.
        #[arg(long, default_value = "128")]
        tokens_per_iter: usize,

        /// Interval between drift checks (e.g. "5m", "10m").
        #[arg(long, default_value = "5m")]
        check_interval: String,

        /// Use nightly preset (24h, 10m checks).
        #[arg(long)]
        nightly: bool,

        /// Use smoke preset (5m quick test).
        #[arg(long)]
        smoke: bool,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Write JSON results to file.
        #[arg(long)]
        json_output: Option<String>,
    },

    /// Profile the decode hot path (per-operation timing breakdown).
    Profile {
        /// Path to GGUF model file.
        #[arg(long)]
        model: String,

        /// Number of warmup tokens before profiling.
        #[arg(long, default_value = "16")]
        warmup_tokens: usize,

        /// Number of tokens to profile.
        #[arg(long, default_value = "64")]
        profile_tokens: usize,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Write JSON results to file.
        #[arg(long)]
        json_output: Option<String>,

        /// Optional baseline profile JSON file to compare against.
        #[arg(long)]
        baseline_json: Option<String>,

        /// Number of top regression kernels to print when baseline is provided.
        #[arg(long, default_value = "3")]
        top_regressions: usize,

        /// Apply llama.cpp-aligned AX runtime preset (if vars are unset).
        #[arg(long)]
        llama_parity_preset: bool,
    },

    /// Run a performance benchmark.
    Bench {
        /// Path to GGUF model file.
        #[arg(long)]
        model: String,

        /// Number of prompt tokens for prefill.
        #[arg(long, default_value = "512")]
        prompt_tokens: usize,

        /// Number of tokens to decode.
        #[arg(long, default_value = "128")]
        decode_tokens: usize,

        /// Number of warmup iterations.
        #[arg(long, default_value = "2")]
        warmup_iters: usize,

        /// Number of measurement iterations.
        #[arg(long, default_value = "5")]
        measure_iters: usize,

        /// Deterministic mode: repeated samples with cooldown, report medians.
        #[arg(long)]
        deterministic: bool,

        /// Number of repeated samples in deterministic mode.
        #[arg(long, default_value = "1")]
        samples: usize,

        /// Cooldown between measured iterations in deterministic mode (ms).
        #[arg(long, default_value = "0")]
        cooldown_ms: u64,

        /// Apply llama.cpp-aligned AX runtime preset (if vars are unset).
        #[arg(long)]
        llama_parity_preset: bool,

        /// Benchmark intent. Throughput mode uses the production fast path;
        /// latency mode keeps per-token latency measurement meaningful.
        #[arg(long, value_enum, default_value_t = BenchIntentArg::Throughput)]
        intent: BenchIntentArg,
    },

    /// Run a speculative-decoding performance benchmark.
    Speculative {
        /// Path to target GGUF model file.
        #[arg(long)]
        model: String,

        /// Path to draft GGUF model file.
        #[arg(long = "draft-model")]
        draft_model: String,

        /// Number of prompt tokens for prefill.
        #[arg(long, default_value = "512")]
        prompt_tokens: usize,

        /// Number of tokens to decode.
        #[arg(long, default_value = "128")]
        decode_tokens: usize,

        /// Number of warmup iterations.
        #[arg(long, default_value = "2")]
        warmup_iters: usize,

        /// Number of measurement iterations.
        #[arg(long, default_value = "5")]
        measure_iters: usize,

        /// Deterministic mode: repeated samples with cooldown, report medians.
        #[arg(long)]
        deterministic: bool,

        /// Number of repeated samples in deterministic mode.
        #[arg(long, default_value = "1")]
        samples: usize,

        /// Cooldown between measured iterations in deterministic mode (ms).
        #[arg(long, default_value = "0")]
        cooldown_ms: u64,

        /// Speculative lookahead K.
        #[arg(long = "speculative-k", default_value = "4")]
        speculative_k: usize,

        /// Apply llama.cpp-aligned AX runtime preset (if vars are unset).
        #[arg(long)]
        llama_parity_preset: bool,
    },
}

fn set_env_default(key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        // SAFETY: Process-global environment mutation is intentional here for this
        // one-shot CLI process before model/runner initialization.
        unsafe { std::env::set_var(key, value) };
    }
}

fn apply_llama_parity_preset() {
    set_env_default("AX_METAL_BATCH_F16_IO", "1");
    set_env_default("AX_METAL_F16_KV_CACHE", "on");
    set_env_default("AX_METAL_PREFILL_FA2_MODE", "auto");
    set_env_default("AX_METAL_PREFILL_FA2_HD128_MODE", "auto");
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Soak {
            model,
            duration,
            max_rss_drift,
            max_p95_drift,
            tokens_per_iter,
            check_interval,
            nightly,
            smoke,
            json,
            json_output,
        } => {
            let backend =
                ax_core::backend::create_backend(ax_core::backend::BackendConfig::default())
                    .unwrap_or_else(|e| {
                        eprintln!("Soak test error: {e}");
                        process::exit(2);
                    });
            ax_core::scheduler::init_global_threadpool();
            let mut config = if nightly {
                SoakConfig::nightly()
            } else if smoke {
                SoakConfig::smoke()
            } else {
                SoakConfig {
                    duration: parse_duration(&duration).unwrap_or_else(|e| {
                        eprintln!("Invalid --duration value '{duration}': {e}");
                        process::exit(2);
                    }),
                    max_rss_drift: max_rss_drift / 100.0,
                    max_p95_drift: max_p95_drift / 100.0,
                    tokens_per_iter,
                    check_interval: parse_duration(&check_interval).unwrap_or_else(|e| {
                        eprintln!("Invalid --check-interval value '{check_interval}': {e}");
                        process::exit(2);
                    }),
                    ..Default::default()
                }
            };
            config.model_path = model;

            eprintln!(
                "Starting soak test: duration={:.1}h, max_rss_drift={:.0}%, max_p95_drift={:.0}%",
                config.duration.as_secs_f64() / 3600.0,
                config.max_rss_drift * 100.0,
                config.max_p95_drift * 100.0,
            );

            match soak::run_soak_test_with_backend(&config, backend) {
                Ok(result) => {
                    result.print_summary();

                    if json {
                        match result.to_json() {
                            Ok(j) => println!("{j}"),
                            Err(e) => eprintln!("JSON serialization error: {e}"),
                        }
                    }

                    if let Some(path) = json_output {
                        match result.to_json() {
                            Ok(j) => {
                                if let Err(e) = std::fs::write(&path, j) {
                                    eprintln!("Failed to write JSON output to {path}: {e}");
                                } else {
                                    eprintln!("Results written to {path}");
                                }
                            }
                            Err(e) => eprintln!("JSON serialization error: {e}"),
                        }
                    }

                    if !result.passed {
                        process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Soak test error: {e}");
                    process::exit(2);
                }
            }
        }

        Command::Profile {
            model,
            warmup_tokens,
            profile_tokens,
            json,
            json_output,
            baseline_json,
            top_regressions,
            llama_parity_preset,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            let backend =
                ax_core::backend::create_backend(ax_core::backend::BackendConfig::default())
                    .unwrap_or_else(|e| {
                        eprintln!("Profile error: {e}");
                        process::exit(2);
                    });
            ax_core::scheduler::init_global_threadpool();
            let config = ProfileConfig {
                model_path: model,
                warmup_tokens,
                profile_tokens,
            };

            match profile::run_profile_with_backend(&config, backend) {
                Ok(result) => {
                    result.print_summary();

                    if json {
                        match result.to_json() {
                            Ok(j) => println!("{j}"),
                            Err(e) => eprintln!("JSON serialization error: {e}"),
                        }
                    }

                    if let Some(path) = json_output {
                        match result.to_json() {
                            Ok(j) => {
                                if let Err(e) = std::fs::write(&path, j) {
                                    eprintln!("Failed to write JSON output to {path}: {e}");
                                } else {
                                    eprintln!("Results written to {path}");
                                }
                            }
                            Err(e) => eprintln!("JSON serialization error: {e}"),
                        }
                    }

                    if let Some(path) = baseline_json {
                        match std::fs::read_to_string(&path) {
                            Ok(s) => match profile::ProfileResult::from_json(&s) {
                                Ok(baseline) => {
                                    result.print_regression_summary(
                                        &baseline,
                                        top_regressions.max(1),
                                    );
                                }
                                Err(e) => {
                                    eprintln!("Failed to parse baseline JSON at {path}: {e}");
                                    process::exit(2);
                                }
                            },
                            Err(e) => {
                                eprintln!("Failed to read baseline JSON at {path}: {e}");
                                process::exit(2);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Profile error: {e}");
                    process::exit(2);
                }
            }
        }

        Command::Bench {
            model,
            prompt_tokens,
            decode_tokens,
            warmup_iters,
            measure_iters,
            deterministic,
            samples,
            cooldown_ms,
            llama_parity_preset,
            intent,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            let backend =
                ax_core::backend::create_backend(ax_core::backend::BackendConfig::default())
                    .unwrap_or_else(|e| {
                        eprintln!("Benchmark error: {e}");
                        process::exit(2);
                    });
            ax_core::scheduler::init_global_threadpool();
            let config = BenchConfig {
                model_path: model,
                prompt_tokens,
                decode_tokens,
                warmup_iters,
                measure_iters,
                deterministic,
                samples,
                cooldown_ms,
                intent: intent.into(),
            };

            match perf::run_benchmark_with_backend(&config, backend) {
                Ok(result) => {
                    result.print_summary();
                }
                Err(e) => {
                    eprintln!("Benchmark error: {e}");
                    process::exit(2);
                }
            }
        }

        Command::Speculative {
            model,
            draft_model,
            prompt_tokens,
            decode_tokens,
            warmup_iters,
            measure_iters,
            deterministic,
            samples,
            cooldown_ms,
            speculative_k,
            llama_parity_preset,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            let backend =
                ax_core::backend::create_backend(ax_core::backend::BackendConfig::default())
                    .unwrap_or_else(|e| {
                        eprintln!("Speculative benchmark error: {e}");
                        process::exit(2);
                    });
            ax_core::scheduler::init_global_threadpool();
            let config = SpecBenchConfig {
                model_path: model,
                draft_model_path: draft_model,
                prompt_tokens,
                decode_tokens,
                warmup_iters,
                measure_iters,
                deterministic,
                samples,
                cooldown_ms,
                speculative_k,
            };

            match perf::run_speculative_benchmark_with_backend(&config, backend) {
                Ok(result) => {
                    result.print_summary();
                }
                Err(e) => {
                    eprintln!("Speculative benchmark error: {e}");
                    process::exit(2);
                }
            }
        }
    }
}

/// Parse a human-friendly duration string (e.g. "5m", "8h", "24h", "300s").
fn parse_duration(s: &str) -> anyhow::Result<Duration> {
    let s = s.trim();
    let parse_secs = |raw: &str, unit: &str, mul: u64| -> anyhow::Result<Duration> {
        let base = raw
            .parse::<u64>()
            .map_err(|_| anyhow::anyhow!("invalid duration '{}{}'", raw, unit))?;
        let secs = base
            .checked_mul(mul)
            .ok_or_else(|| anyhow::anyhow!("duration too large: '{}{}'", raw, unit))?;
        Ok(Duration::from_secs(secs))
    };
    if let Some(hours) = s.strip_suffix('h') {
        parse_secs(hours, "h", 3600)
    } else if let Some(mins) = s.strip_suffix('m') {
        parse_secs(mins, "m", 60)
    } else if let Some(secs) = s.strip_suffix('s') {
        parse_secs(secs, "s", 1)
    } else {
        // Default: assume seconds
        parse_secs(s, "", 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(
            parse_duration("24h").unwrap(),
            Duration::from_secs(24 * 3600)
        );
        assert_eq!(parse_duration("8h").unwrap(), Duration::from_secs(8 * 3600));
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(parse_duration("10m").unwrap(), Duration::from_secs(600));
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("300s").unwrap(), Duration::from_secs(300));
    }

    #[test]
    fn test_parse_duration_bare_number() {
        assert_eq!(parse_duration("60").unwrap(), Duration::from_secs(60));
    }

    #[test]
    fn test_parse_duration_invalid() {
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("1x").is_err());
    }
}
