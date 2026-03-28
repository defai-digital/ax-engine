//! ax-engine-bench CLI — benchmark, profiler, and soak test runner for AX Engine.
//!
//! Usage:
//!   ax-engine-bench soak --model <path> [--duration 24h] [--json] [--json-output <path>]
//!   ax-engine-bench bench --model <path> [--prompt-tokens 512] [--decode-tokens 128] [--json] [--json-output <path>]
//!   ax-engine-bench prefill-profile --model <path> [--prompt-tokens 512] [--json] [--json-output <path>]
//!   ax-engine-bench profile --model <path> [--profile-tokens 64] [--json] [--json-output <path>]
//!   ax-engine-bench speculative --model <target> --draft-model <draft> [--json] [--json-output <path>]
//!   ax-engine-bench parity --model <path> [--prompt "Hello"] [--decode-tokens 8]
//!   ax-engine-bench parity --model <path> [--prompt "Hello"] --speculative-verify-k 2
//!   ax-engine-bench microbench [--suite gpu] [--json] [--profile-output <path>]

use std::process;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};

use ax_engine_bench::microbench::{
    self, MicrobenchConfig, MicrobenchProfileExportAction, MicrobenchProfileExportStatus,
    MicrobenchSuite,
};
use ax_engine_bench::parity::{self, ParityConfig, ParityMode};
use ax_engine_bench::perf::{self, BenchConfig, SpecBenchConfig};
use ax_engine_bench::prefill_profile::{self, PrefillProfileConfig};
use ax_engine_bench::profile::{self, ProfileConfig};
use ax_engine_bench::soak::{self, SoakConfig};
use ax_engine_core::backend::BackendConfig;
use ax_engine_core::model::DecodeIntent;

#[derive(Parser)]
#[command(
    name = "ax-engine-bench",
    about = "AX Engine benchmark and soak test runner"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum BenchIntentArg {
    Throughput,
    Latency,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum MicrobenchSuiteArg {
    Cpu,
    Gpu,
    Uma,
    Sync,
    All,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ParityBackendArg {
    Hybrid,
    Metal,
    HybridCpuDecode,
}

impl From<BenchIntentArg> for DecodeIntent {
    fn from(value: BenchIntentArg) -> Self {
        match value {
            BenchIntentArg::Throughput => DecodeIntent::Throughput,
            BenchIntentArg::Latency => DecodeIntent::Latency,
        }
    }
}

impl From<MicrobenchSuiteArg> for MicrobenchSuite {
    fn from(value: MicrobenchSuiteArg) -> Self {
        match value {
            MicrobenchSuiteArg::Cpu => MicrobenchSuite::Cpu,
            MicrobenchSuiteArg::Gpu => MicrobenchSuite::Gpu,
            MicrobenchSuiteArg::Uma => MicrobenchSuite::Uma,
            MicrobenchSuiteArg::Sync => MicrobenchSuite::Sync,
            MicrobenchSuiteArg::All => MicrobenchSuite::All,
        }
    }
}

impl From<ParityBackendArg> for BackendConfig {
    fn from(value: ParityBackendArg) -> Self {
        match value {
            ParityBackendArg::Hybrid => BackendConfig::Hybrid,
            ParityBackendArg::Metal => BackendConfig::Metal,
            ParityBackendArg::HybridCpuDecode => BackendConfig::HybridCpuDecode,
        }
    }
}

fn format_export_status_blockers(status: &MicrobenchProfileExportStatus) -> Vec<String> {
    microbench::format_export_status_blockers(status)
}

fn print_export_status_blockers(status: &MicrobenchProfileExportStatus) {
    for line in format_export_status_blockers(status) {
        eprintln!("  {line}");
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

        /// Override kernel profile path for this run.
        #[arg(long)]
        kernel_profile_path: Option<String>,
    },

    /// Profile the batched-prefill path (coarse timing breakdown).
    PrefillProfile {
        /// Path to GGUF model file.
        #[arg(long)]
        model: String,

        /// Number of prompt tokens to process.
        #[arg(long, default_value = "512")]
        prompt_tokens: usize,

        /// Number of unprofiled warmup prefills before measurement.
        #[arg(long, default_value = "1")]
        warmup_iters: usize,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Write JSON results to file.
        #[arg(long)]
        json_output: Option<String>,

        /// Apply llama.cpp-aligned AX runtime preset (if vars are unset).
        #[arg(long)]
        llama_parity_preset: bool,

        /// Override kernel profile path for this run.
        #[arg(long)]
        kernel_profile_path: Option<String>,
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

        /// Override kernel profile path for this run.
        #[arg(long)]
        kernel_profile_path: Option<String>,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Write JSON results to file.
        #[arg(long)]
        json_output: Option<String>,
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

        /// Override kernel profile path for this run.
        #[arg(long)]
        kernel_profile_path: Option<String>,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Write JSON results to file.
        #[arg(long)]
        json_output: Option<String>,
    },

    /// Run hardware-oriented microbenchmarks.
    Microbench {
        /// Which suite to run.
        #[arg(long, value_enum, default_value_t = MicrobenchSuiteArg::All)]
        suite: MicrobenchSuiteArg,

        /// Number of measurement iterations per subtest.
        #[arg(long, default_value = "10")]
        iterations: usize,

        /// Number of times to rerun the whole selected suite before aggregating.
        #[arg(long, default_value = "1")]
        suite_runs: usize,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Write a suggested kernel profile JSON derived from microbench winners.
        #[arg(long)]
        profile_output: Option<String>,

        /// Allow writing a suggested profile even when the confidence gate blocks export.
        #[arg(long)]
        allow_low_confidence_export: bool,
    },

    /// Compare CPU reference logits against another backend on the same token path.
    Parity {
        /// Path to GGUF model file.
        #[arg(long)]
        model: String,

        /// Prompt text used to seed the shared context.
        #[arg(long, default_value = "Hello")]
        prompt: String,

        /// Number of greedy decode steps to compare after prefill.
        #[arg(long, default_value = "8")]
        decode_tokens: usize,

        /// Compare the speculative verification batch shape
        /// `forward_batch_all_logits([last_token] + draft_tokens)` with K drafts.
        #[arg(long)]
        speculative_verify_k: Option<usize>,

        /// Backend to compare against CPU.
        #[arg(long, value_enum, default_value_t = ParityBackendArg::Hybrid)]
        compare_backend: ParityBackendArg,

        /// Number of top tokens to print per step.
        #[arg(long, default_value = "5")]
        top_tokens: usize,

        /// Absolute logit tolerance used for divergence reporting.
        #[arg(long, default_value = "0.001")]
        max_abs_tolerance: f32,

        /// Apply llama.cpp-aligned AX runtime preset (if vars are unset).
        #[arg(long)]
        llama_parity_preset: bool,

        /// Override kernel profile path for this run.
        #[arg(long)]
        kernel_profile_path: Option<String>,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Write JSON results to file.
        #[arg(long)]
        json_output: Option<String>,
    },
}

fn set_env_default(key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        // SAFETY: Process-global environment mutation is intentional here for this
        // one-shot CLI process before model/runner initialization.
        unsafe { std::env::set_var(key, value) };
    }
}

fn apply_kernel_profile_override(path: Option<&str>) {
    if let Some(path) = path {
        // SAFETY: Process-global environment mutation is intentional here for this
        // one-shot CLI process before backend/model initialization.
        unsafe { std::env::set_var("AX_KERNEL_PROFILE_PATH", path) };
    }
}

fn apply_llama_parity_preset() {
    set_env_default("AX_METAL_F16_KV_CACHE", "on");
}

fn create_runtime_backend(context: &str) -> Box<dyn ax_engine_core::backend::Backend> {
    ax_engine_core::backend::create_backend(
        ax_engine_core::backend::resolve_backend_config_from_env(),
    )
    .unwrap_or_else(|e| {
        eprintln!("{context} error: {e}");
        process::exit(2);
    })
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
            let backend = create_runtime_backend("Soak test");
            ax_engine_core::scheduler::init_global_threadpool();
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
            kernel_profile_path,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            apply_kernel_profile_override(kernel_profile_path.as_deref());
            let backend = create_runtime_backend("Profile");
            ax_engine_core::scheduler::init_global_threadpool();
            let config = ProfileConfig {
                model_path: model,
                warmup_tokens,
                profile_tokens,
                kernel_profile_path,
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

        Command::PrefillProfile {
            model,
            prompt_tokens,
            warmup_iters,
            json,
            json_output,
            llama_parity_preset,
            kernel_profile_path,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            apply_kernel_profile_override(kernel_profile_path.as_deref());
            let backend = create_runtime_backend("Prefill profile");
            ax_engine_core::scheduler::init_global_threadpool();
            let config = PrefillProfileConfig {
                model_path: model,
                prompt_tokens,
                warmup_iters,
                kernel_profile_path,
            };

            match prefill_profile::run_prefill_profile_with_backend(&config, backend) {
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
                }
                Err(e) => {
                    eprintln!("Prefill profile error: {e}");
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
            kernel_profile_path,
            json,
            json_output,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            apply_kernel_profile_override(kernel_profile_path.as_deref());
            let backend = create_runtime_backend("Benchmark");
            ax_engine_core::scheduler::init_global_threadpool();
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
                kernel_profile_path,
            };

            match perf::run_benchmark_with_backend(&config, backend) {
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
            kernel_profile_path,
            json,
            json_output,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            apply_kernel_profile_override(kernel_profile_path.as_deref());
            let backend = create_runtime_backend("Speculative benchmark");
            ax_engine_core::scheduler::init_global_threadpool();
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
                kernel_profile_path,
            };

            match perf::run_speculative_benchmark_with_backend(&config, backend) {
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
                }
                Err(e) => {
                    eprintln!("Speculative benchmark error: {e}");
                    process::exit(2);
                }
            }
        }

        Command::Microbench {
            suite,
            iterations,
            suite_runs,
            json,
            profile_output,
            allow_low_confidence_export,
        } => {
            let config = MicrobenchConfig {
                suite: suite.into(),
                iterations: iterations.max(1),
                suite_runs: suite_runs.max(1),
            };

            match microbench::run_microbench(&config) {
                Ok(result) => {
                    let output_path = profile_output.clone();
                    let export_decision = result.suggested_kernel_profile_export_decision();
                    let requested_profile_output = output_path.is_some();
                    let export_action = microbench::determine_profile_export_action(
                        requested_profile_output,
                        allow_low_confidence_export,
                        &export_decision,
                    );
                    let mut wrote_profile = false;

                    if let Some(path) = output_path.clone()
                        && matches!(export_action, MicrobenchProfileExportAction::Write { .. })
                    {
                        match result.suggested_kernel_profile_json() {
                            Ok(j) => {
                                if let Err(e) = std::fs::write(&path, j) {
                                    eprintln!(
                                        "Failed to write suggested kernel profile to {path}: {e}"
                                    );
                                    process::exit(2);
                                } else {
                                    wrote_profile = true;
                                    eprintln!("Suggested kernel profile written to {path}");
                                }
                            }
                            Err(e) => {
                                eprintln!("Kernel profile serialization error: {e}");
                                process::exit(2);
                            }
                        }
                    }

                    let report =
                        result.with_export_outcome(output_path, export_action, wrote_profile);
                    let export_status = report
                        .export_status()
                        .expect("microbench export status should be present after outcome");
                    if export_status.override_used {
                        eprintln!(
                            "Bypassing suggested kernel profile confidence gate due to --allow-low-confidence-export"
                        );
                        print_export_status_blockers(export_status);
                    }
                    if json {
                        match report.to_json() {
                            Ok(j) => println!("{j}"),
                            Err(e) => {
                                eprintln!("JSON serialization error: {e}");
                                process::exit(2);
                            }
                        }
                    } else {
                        report.print_summary();
                    }

                    if export_status.exit_code != 0 {
                        eprintln!("Suggested kernel profile export blocked by confidence gate:");
                        print_export_status_blockers(export_status);
                        process::exit(export_status.exit_code);
                    }
                }
                Err(e) => {
                    eprintln!("Microbench error: {e}");
                    process::exit(2);
                }
            }
        }

        Command::Parity {
            model,
            prompt,
            decode_tokens,
            speculative_verify_k,
            compare_backend,
            top_tokens,
            max_abs_tolerance,
            llama_parity_preset,
            kernel_profile_path,
            json,
            json_output,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            apply_kernel_profile_override(kernel_profile_path.as_deref());
            let config = ParityConfig {
                model_path: model,
                prompt,
                decode_tokens,
                compare_backend: compare_backend.into(),
                top_tokens,
                max_abs_tolerance,
                mode: speculative_verify_k
                    .map(|draft_tokens| ParityMode::SpeculativeVerify { draft_tokens })
                    .unwrap_or(ParityMode::Decode),
            };

            match parity::run_parity(&config) {
                Ok(result) => {
                    result.print_summary();

                    if json {
                        match result.to_json() {
                            Ok(j) => println!("{j}"),
                            Err(e) => {
                                eprintln!("JSON serialization error: {e}");
                                process::exit(2);
                            }
                        }
                    }

                    if let Some(path) = json_output {
                        match result.to_json() {
                            Ok(j) => {
                                if let Err(e) = std::fs::write(&path, j) {
                                    eprintln!("Failed to write JSON output to {path}: {e}");
                                    process::exit(2);
                                } else {
                                    eprintln!("Results written to {path}");
                                }
                            }
                            Err(e) => {
                                eprintln!("JSON serialization error: {e}");
                                process::exit(2);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Parity probe error: {e}");
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
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use ax_engine_bench::microbench::{
        MicrobenchProfileExportBlocker, MicrobenchProfileExportBlockerKind,
        MicrobenchProfileExportState,
    };

    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("ax-bench env test lock")
    }

    struct EnvVarRestore {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match &self.previous {
                Some(prev) => unsafe { std::env::set_var(self.key, prev) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    fn with_cleared_env_vars<T>(keys: &[&'static str], f: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        let _restore: Vec<_> = keys
            .iter()
            .map(|&key| {
                let previous = std::env::var_os(key);
                unsafe { std::env::remove_var(key) };
                EnvVarRestore { key, previous }
            })
            .collect();
        f()
    }

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

    #[test]
    fn test_determine_profile_export_action_blocks_by_default() {
        let decision = microbench::MicrobenchProfileExportDecision {
            allowed: false,
            blocker_details: vec![],
            blockers: vec!["blocked".to_string()],
        };
        assert_eq!(
            microbench::determine_profile_export_action(true, false, &decision),
            MicrobenchProfileExportAction::Blocked
        );
    }

    #[test]
    fn test_determine_profile_export_action_allows_override() {
        let decision = microbench::MicrobenchProfileExportDecision {
            allowed: false,
            blocker_details: vec![],
            blockers: vec!["blocked".to_string()],
        };
        assert_eq!(
            microbench::determine_profile_export_action(true, true, &decision),
            MicrobenchProfileExportAction::Write {
                override_used: true
            }
        );
    }

    #[test]
    fn test_determine_profile_export_action_skips_when_not_requested() {
        let decision = microbench::MicrobenchProfileExportDecision {
            allowed: true,
            blocker_details: vec![],
            blockers: vec![],
        };
        assert_eq!(
            microbench::determine_profile_export_action(false, false, &decision),
            MicrobenchProfileExportAction::NotRequested
        );
    }

    #[test]
    fn test_format_export_status_blockers_uses_structured_threshold_details() {
        let status = MicrobenchProfileExportStatus {
            state: MicrobenchProfileExportState::Blocked,
            exit_code: 1,
            requested: true,
            gate_allowed: false,
            override_used: false,
            wrote_profile: false,
            blocker_details: vec![MicrobenchProfileExportBlocker {
                kind: MicrobenchProfileExportBlockerKind::BelowThreshold,
                reason: "decode_matvec.q4_k avg speedup 1.080x below export threshold 1.100x"
                    .to_string(),
                rule: Some("decode_matvec.q4_k".to_string()),
                avg_speedup: Some(1.08),
                required_speedup: Some(1.10),
                forced: false,
            }],
            blockers: vec!["legacy blocker".to_string()],
            output_path: Some("/tmp/profile.json".to_string()),
        };

        let lines = format_export_status_blockers(&status);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("kind=below_threshold"));
        assert!(lines[0].contains("rule=decode_matvec.q4_k"));
        assert!(lines[0].contains("avg=1.080x"));
        assert!(lines[0].contains("required=1.100x"));
        assert!(!lines[0].contains("legacy blocker"));
    }

    #[test]
    fn test_format_export_status_blockers_uses_structured_forced_details() {
        let status = MicrobenchProfileExportStatus {
            state: MicrobenchProfileExportState::Blocked,
            exit_code: 1,
            requested: true,
            gate_allowed: false,
            override_used: false,
            wrote_profile: false,
            blocker_details: vec![MicrobenchProfileExportBlocker {
                kind: MicrobenchProfileExportBlockerKind::Forced,
                reason: "forced export block via AX_BENCH_MICROBENCH_FORCE_EXPORT_BLOCK=1"
                    .to_string(),
                rule: None,
                avg_speedup: None,
                required_speedup: None,
                forced: true,
            }],
            blockers: vec!["legacy forced blocker".to_string()],
            output_path: None,
        };

        let lines = format_export_status_blockers(&status);
        assert_eq!(
            lines,
            vec![
                "forced export block via AX_BENCH_MICROBENCH_FORCE_EXPORT_BLOCK=1 kind=forced"
                    .to_string()
            ]
        );
    }

    #[test]
    fn test_llama_parity_preset_does_not_override_prefill_routing_modes() {
        with_cleared_env_vars(
            &[
                "AX_METAL_F16_KV_CACHE",
                "AX_METAL_PREFILL_FA2_MODE",
                "AX_METAL_PREFILL_FA2_HD128_MODE",
            ],
            || {
                apply_llama_parity_preset();
                assert_eq!(
                    std::env::var("AX_METAL_F16_KV_CACHE").ok().as_deref(),
                    Some("on")
                );
                assert!(std::env::var_os("AX_METAL_BATCH_F16_IO").is_none());
                assert!(std::env::var_os("AX_METAL_PREFILL_FA2_MODE").is_none());
                assert!(std::env::var_os("AX_METAL_PREFILL_FA2_HD128_MODE").is_none());
            },
        );
    }
}
