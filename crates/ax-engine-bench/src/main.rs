//! ax-engine-bench CLI — benchmark, profiler, and soak test runner for AX Engine.
//!
//! Usage:
//!   ax-engine-bench soak --model <path> [--duration 24h] [--json] [--json-output <path>]
//!   ax-engine-bench bench --model <path> [--prompt-tokens 512] [--decode-tokens 128] [--json] [--json-output <path>]
//!   ax-engine-bench prefill-profile --model <path> [--prompt-tokens 512] [--json] [--json-output <path>]
//!   ax-engine-bench prefill-route-compare --model <path> [--prompt-tokens 512] [--samples 3] [--json]
//!   ax-engine-bench prefill-gap --model <path> [--baseline-json <path> | --baseline-prefill-tok-s <n>] [--json]
//!   ax-engine-bench profile --model <path> [--profile-tokens 64] [--json] [--json-output <path>]
//!   ax-engine-bench speculative --model <target> --draft-model <draft> [--json] [--json-output <path>]
//!   ax-engine-bench parity --model <path> [--prompt "Hello"] [--decode-tokens 8]
//!   ax-engine-bench parity --model <path> [--prompt "Hello"] --speculative-verify-k 2
//!   ax-engine-bench microbench [--suite gpu] [--json] [--profile-output <path>]

use std::process::{self, Command as ProcessCommand};
use std::time::Duration;

use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use ax_engine_bench::arch_config::{
    Qwen35PrefillAlphaBetaStorageModeArg, Qwen35PrefillRecurrentStateModeArg,
    Qwen35SpecVerifyBranchArg, qwen35_prefill_alpha_beta_storage_mode_env,
    qwen35_prefill_force_backend_state_batch_env, qwen35_prefill_recurrent_state_mode_env,
    qwen35_spec_verify_branch_env,
};
use ax_engine_bench::microbench::{
    self, MicrobenchConfig, MicrobenchProfileExportAction, MicrobenchProfileExportStatus,
    MicrobenchSuite,
};
use ax_engine_bench::parity::{self, ParityConfig, ParityMode};
use ax_engine_bench::perf::{self, BenchConfig, SpecBenchConfig};
use ax_engine_bench::prefill_gap::{self, PrefillGapConfig};
use ax_engine_bench::prefill_profile::{
    self, LocalPrefillHd128Route, PrefillProfileConfig, PrefillProfileResult,
};
use ax_engine_bench::profile::{self, ProfileConfig};
use ax_engine_bench::soak::{self, SoakConfig};
use ax_engine_core::backend::BackendConfig;
use ax_engine_core::model::DecodeIntent;

#[cfg(test)]
mod tests;

#[derive(Parser)]
#[command(
    name = "ax-engine-bench",
    about = "AX Engine benchmark and soak test runner"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
enum BenchIntentArg {
    Throughput,
    Latency,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
enum MicrobenchSuiteArg {
    Cpu,
    Gpu,
    Uma,
    Sync,
    All,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
enum ParityBackendArg {
    Hybrid,
    Metal,
    HybridCpuDecode,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
enum LocalPrefillHd128RouteArg {
    Auto,
    AxBc64,
    Fa2SimdHd128,
    Fa2HalfHd128,
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

impl From<LocalPrefillHd128RouteArg> for LocalPrefillHd128Route {
    fn from(value: LocalPrefillHd128RouteArg) -> Self {
        match value {
            LocalPrefillHd128RouteArg::Auto => LocalPrefillHd128Route::Auto,
            LocalPrefillHd128RouteArg::AxBc64 => LocalPrefillHd128Route::AxBc64,
            LocalPrefillHd128RouteArg::Fa2SimdHd128 => LocalPrefillHd128Route::Fa2SimdHd128,
            LocalPrefillHd128RouteArg::Fa2HalfHd128 => LocalPrefillHd128Route::Fa2HalfHd128,
        }
    }
}

impl LocalPrefillHd128RouteArg {
    fn cli_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::AxBc64 => "ax-bc64",
            Self::Fa2SimdHd128 => "fa2-simd-hd128",
            Self::Fa2HalfHd128 => "fa2-half-hd128",
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

fn parse_non_zero_usize(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid integer '{value}': {err}"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_string());
    }
    Ok(parsed)
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
        #[arg(long, default_value = "128", value_parser = parse_non_zero_usize)]
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

        /// Fan the same prompt out across this many Qwen3.5 recurrent slots on
        /// a shared attention timeline during prefill.
        #[arg(long, default_value = "1")]
        qwen35_shared_timeline_slots: usize,

        /// Optional source recurrent slot for Qwen3.5 shared-timeline prefill.
        #[arg(long)]
        qwen35_shared_timeline_source_slot: Option<usize>,

        /// Experimental Qwen3.5 recurrent state mode for GPU prefill handoff.
        #[arg(long, value_enum, default_value = "auto")]
        qwen35_recurrent_state_mode: Qwen35PrefillRecurrentStateModeArg,

        /// Experimental Qwen3.5 alpha/beta scratch storage mode for recurrent handoff.
        #[arg(long, value_enum, default_value = "auto")]
        qwen35_alpha_beta_storage_mode: Qwen35PrefillAlphaBetaStorageModeArg,

        /// Prime Qwen3.5 recurrent Metal slot buffers before timed prefill.
        #[arg(long)]
        qwen35_prime_slot_buffers: bool,

        /// Run one unmeasured prefill on the same KV before timing the measured prefill.
        #[arg(long)]
        qwen35_prewarm_prefill_same_kv: bool,

        /// Force Qwen3.5 recurrent prefill to bypass model-side GPU QKV handoff and use backend state batch.
        #[arg(long)]
        qwen35_force_backend_state_batch: bool,

        /// Force a specific local HD128 attention prefill route for this run.
        #[arg(long, value_enum, default_value = "auto")]
        local_hd128_route: LocalPrefillHd128RouteArg,

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

    /// Compare local HD128 prefill attention routes via fresh subprocess runs.
    PrefillRouteCompare {
        /// Path to GGUF model file.
        #[arg(long)]
        model: String,

        /// Number of prompt tokens to process.
        #[arg(long, default_value = "512")]
        prompt_tokens: usize,

        /// Number of unprofiled warmup prefills before each measured run.
        #[arg(long, default_value = "1")]
        warmup_iters: usize,

        /// Number of measured runs to execute per route.
        #[arg(long, default_value = "3")]
        samples: usize,

        /// Restrict comparison to specific routes. Repeat the flag to add more.
        #[arg(long, value_enum)]
        route: Vec<LocalPrefillHd128RouteArg>,

        /// Fan the same prompt out across this many Qwen3.5 recurrent slots on
        /// a shared attention timeline during prefill.
        #[arg(long, default_value = "1")]
        qwen35_shared_timeline_slots: usize,

        /// Optional source recurrent slot for Qwen3.5 shared-timeline prefill.
        #[arg(long)]
        qwen35_shared_timeline_source_slot: Option<usize>,

        /// Experimental Qwen3.5 recurrent state mode for GPU prefill handoff.
        #[arg(long, value_enum, default_value = "auto")]
        qwen35_recurrent_state_mode: Qwen35PrefillRecurrentStateModeArg,

        /// Experimental Qwen3.5 alpha/beta scratch storage mode for recurrent handoff.
        #[arg(long, value_enum, default_value = "auto")]
        qwen35_alpha_beta_storage_mode: Qwen35PrefillAlphaBetaStorageModeArg,

        /// Prime Qwen3.5 recurrent Metal slot buffers before timed prefill.
        #[arg(long)]
        qwen35_prime_slot_buffers: bool,

        /// Run one unmeasured prefill on the same KV before timing the measured prefill.
        #[arg(long)]
        qwen35_prewarm_prefill_same_kv: bool,

        /// Force Qwen3.5 recurrent prefill to bypass model-side GPU QKV handoff and use backend state batch.
        #[arg(long)]
        qwen35_force_backend_state_batch: bool,

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

    /// Attribute AX prefill gap versus a baseline artifact or inline baseline.
    PrefillGap {
        /// Path to GGUF model file.
        #[arg(long)]
        model: String,

        /// Number of prompt tokens to process.
        #[arg(long, default_value = "512")]
        prompt_tokens: usize,

        /// Number of unprofiled warmup prefills before measurement.
        #[arg(long, default_value = "1")]
        warmup_iters: usize,

        /// Fan the same prompt out across this many Qwen3.5 recurrent slots on
        /// a shared attention timeline during prefill.
        #[arg(long, default_value = "1")]
        qwen35_shared_timeline_slots: usize,

        /// Optional source recurrent slot for Qwen3.5 shared-timeline prefill.
        #[arg(long)]
        qwen35_shared_timeline_source_slot: Option<usize>,

        /// Experimental Qwen3.5 recurrent state mode for GPU prefill handoff.
        #[arg(long, value_enum, default_value = "auto")]
        qwen35_recurrent_state_mode: Qwen35PrefillRecurrentStateModeArg,

        /// Experimental Qwen3.5 alpha/beta scratch storage mode for recurrent handoff.
        #[arg(long, value_enum, default_value = "auto")]
        qwen35_alpha_beta_storage_mode: Qwen35PrefillAlphaBetaStorageModeArg,

        /// Prime Qwen3.5 recurrent Metal slot buffers before timed prefill.
        #[arg(long)]
        qwen35_prime_slot_buffers: bool,

        /// Run one unmeasured prefill on the same KV before timing the measured prefill.
        #[arg(long)]
        qwen35_prewarm_prefill_same_kv: bool,

        /// Force Qwen3.5 recurrent prefill to bypass model-side GPU QKV handoff and use backend state batch.
        #[arg(long)]
        qwen35_force_backend_state_batch: bool,

        /// Baseline JSON artifact with prefill tok/s fields.
        #[arg(long, conflicts_with = "baseline_prefill_tok_s")]
        baseline_json: Option<String>,

        /// Inline baseline prefill throughput in tok/s.
        #[arg(long, conflicts_with = "baseline_json")]
        baseline_prefill_tok_s: Option<f64>,

        /// Optional display label for the baseline.
        #[arg(long)]
        baseline_label: Option<String>,

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

        /// Fan the same prompt out across this many Qwen3.5 recurrent slots on
        /// a shared attention timeline during prefill.
        #[arg(long, default_value = "1")]
        qwen35_shared_timeline_slots: usize,

        /// Optional source recurrent slot for Qwen3.5 shared-timeline prefill.
        #[arg(long)]
        qwen35_shared_timeline_source_slot: Option<usize>,

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

        /// Force Qwen3.5 speculative target verify branch routing.
        #[arg(long, value_enum, default_value = "auto")]
        qwen35_spec_verify_branch: Qwen35SpecVerifyBranchArg,

        /// Run branch `on` and `off` back-to-back and print a delta summary.
        #[arg(long)]
        qwen35_spec_verify_compare: bool,

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

fn with_env_var_override<T>(key: &'static str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
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

    let previous = std::env::var_os(key);
    let _restore = EnvVarRestore { key, previous };
    match value {
        Some(value) => unsafe { std::env::set_var(key, value) },
        None => unsafe { std::env::remove_var(key) },
    }
    f()
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

fn with_env_var_overrides<T>(
    overrides: &[(&'static str, Option<&str>)],
    f: impl FnOnce() -> T,
) -> T {
    struct EnvVarsRestore {
        previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl Drop for EnvVarsRestore {
        fn drop(&mut self) {
            for (key, previous) in self.previous.iter().rev() {
                match previous {
                    Some(prev) => unsafe { std::env::set_var(key, prev) },
                    None => unsafe { std::env::remove_var(key) },
                }
            }
        }
    }

    let previous = overrides
        .iter()
        .map(|(key, _)| (*key, std::env::var_os(key)))
        .collect();
    let _restore = EnvVarsRestore { previous };

    for (key, value) in overrides {
        match value {
            Some(value) => unsafe { std::env::set_var(key, value) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    f()
}

fn local_hd128_prefill_route_env_overrides(
    route: LocalPrefillHd128Route,
) -> Vec<(&'static str, Option<&'static str>)> {
    let mut overrides = vec![
        ("AX_METAL_PREFILL_FA2_MODE", None),
        ("AX_METAL_PREFILL_FA2", None),
        ("AX_METAL_PREFILL_FA2_HD128_MODE", None),
        ("AX_METAL_PREFILL_FA2_HD128", None),
        ("AX_METAL_PREFILL_BC64_MODE", None),
        ("AX_METAL_PREFILL_BC64", None),
        ("AX_METAL_PREFILL_BC64_MIN_TOKENS", None),
        ("AX_METAL_PREFILL_FA2_SIMD", None),
        ("AX_METAL_PREFILL_FA2_HALF", None),
        ("AX_METAL_PREFILL_FA2V2", None),
        ("AX_METAL_PREFILL_AX_SMEM", None),
        ("AX_METAL_PREFILL_AX_SMEM_F16", None),
    ];

    match route {
        LocalPrefillHd128Route::Auto => {}
        LocalPrefillHd128Route::AxBc64 => {
            overrides.extend_from_slice(&[
                ("AX_METAL_PREFILL_FA2_HD128_MODE", Some("off")),
                ("AX_METAL_PREFILL_BC64_MODE", Some("on")),
                ("AX_METAL_PREFILL_FA2_SIMD", Some("off")),
                ("AX_METAL_PREFILL_FA2_HALF", Some("off")),
                ("AX_METAL_PREFILL_FA2V2", Some("off")),
            ]);
        }
        LocalPrefillHd128Route::Fa2SimdHd128 => {
            overrides.extend_from_slice(&[
                ("AX_METAL_PREFILL_FA2_HD128_MODE", Some("on")),
                ("AX_METAL_PREFILL_BC64_MODE", Some("off")),
                ("AX_METAL_PREFILL_FA2_SIMD", Some("on")),
                ("AX_METAL_PREFILL_FA2_HALF", Some("off")),
                ("AX_METAL_PREFILL_FA2V2", Some("off")),
            ]);
        }
        LocalPrefillHd128Route::Fa2HalfHd128 => {
            overrides.extend_from_slice(&[
                ("AX_METAL_PREFILL_FA2_HD128_MODE", Some("on")),
                ("AX_METAL_PREFILL_BC64_MODE", Some("off")),
                ("AX_METAL_PREFILL_FA2_SIMD", Some("on")),
                ("AX_METAL_PREFILL_FA2_HALF", Some("on")),
                ("AX_METAL_PREFILL_FA2V2", Some("off")),
            ]);
        }
    }

    overrides
}

fn with_prefill_profile_env_overrides<T>(
    qwen35_recurrent_state_mode: Qwen35PrefillRecurrentStateModeArg,
    qwen35_alpha_beta_storage_mode: Qwen35PrefillAlphaBetaStorageModeArg,
    qwen35_force_backend_state_batch: bool,
    local_hd128_route: LocalPrefillHd128Route,
    f: impl FnOnce() -> T,
) -> T {
    let mut overrides = vec![
        (
            "AX_QWEN35_PREFILL_RECURRENT_STATE_MODE",
            qwen35_prefill_recurrent_state_mode_env(qwen35_recurrent_state_mode),
        ),
        (
            "AX_QWEN35_PREFILL_ALPHA_BETA_STORAGE_MODE",
            qwen35_prefill_alpha_beta_storage_mode_env(qwen35_alpha_beta_storage_mode),
        ),
        (
            "AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH",
            qwen35_prefill_force_backend_state_batch_env(qwen35_force_backend_state_batch),
        ),
    ];
    overrides.extend(local_hd128_prefill_route_env_overrides(local_hd128_route));
    with_env_var_overrides(&overrides, f)
}

#[derive(Debug, Serialize)]
struct PrefillRouteCompareVariant {
    route: LocalPrefillHd128Route,
    expected_attention_route_prefix: Option<String>,
    observed_attention_routes: Vec<String>,
    matched_expected_route: bool,
    sample_results: Vec<PrefillProfileResult>,
    median_tok_per_sec: f64,
    mean_tok_per_sec: f64,
    median_effective_tok_per_sec: f64,
    mean_effective_tok_per_sec: f64,
    median_total_ms: f64,
}

#[derive(Debug, Serialize)]
struct PrefillRouteCompareReport {
    model: String,
    prompt_tokens: usize,
    warmup_iters: usize,
    samples_per_route: usize,
    kernel_profile_path: Option<String>,
    qwen35_shared_timeline_slots: usize,
    qwen35_shared_timeline_source_slot: Option<usize>,
    qwen35_recurrent_state_mode: Qwen35PrefillRecurrentStateModeArg,
    qwen35_alpha_beta_storage_mode: Qwen35PrefillAlphaBetaStorageModeArg,
    qwen35_prime_slot_buffers: bool,
    qwen35_prewarm_prefill_same_kv: bool,
    qwen35_force_backend_state_batch: bool,
    variants: Vec<PrefillRouteCompareVariant>,
    winner_by_median_effective_tok_per_sec: Option<LocalPrefillHd128Route>,
}

impl PrefillRouteCompareReport {
    fn print_summary(&self) {
        eprintln!();
        eprintln!("=== Prefill Route Compare ===");
        eprintln!("Model:       {}", self.model);
        eprintln!("Prompt:      {} tokens", self.prompt_tokens);
        eprintln!("Samples:     {} per route", self.samples_per_route);
        if let Some(path) = &self.kernel_profile_path {
            eprintln!("KernelProf:  {path}");
        }
        eprintln!();
        eprintln!(
            "{:<18} {:>12} {:>12} {:>10} {:>7}  Observed",
            "Route", "Median tok/s", "Mean tok/s", "Median ms", "Match"
        );
        for variant in &self.variants {
            let observed = if variant.observed_attention_routes.is_empty() {
                "-".to_string()
            } else {
                variant.observed_attention_routes.join(",")
            };
            eprintln!(
                "{:<18} {:>12.1} {:>12.1} {:>10.1} {:>7}  {}",
                variant.route.label(),
                variant.median_effective_tok_per_sec,
                variant.mean_effective_tok_per_sec,
                variant.median_total_ms,
                if variant.matched_expected_route {
                    "yes"
                } else {
                    "no"
                },
                observed,
            );
        }
        if let Some(winner) = self.winner_by_median_effective_tok_per_sec {
            eprintln!();
            eprintln!("Winner:      {}", winner.label());
        }
    }
}

fn prefill_profile_effective_tok_per_sec(result: &PrefillProfileResult) -> f64 {
    if result.effective_tok_per_sec > 0.0 {
        result.effective_tok_per_sec
    } else {
        result.tok_per_sec * result.qwen35_shared_timeline_slots.max(1) as f64
    }
}

fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_profile_subprocess(
    model: &str,
    prompt_tokens: usize,
    warmup_iters: usize,
    local_hd128_route: LocalPrefillHd128RouteArg,
    qwen35_shared_timeline_slots: usize,
    qwen35_shared_timeline_source_slot: Option<usize>,
    qwen35_recurrent_state_mode: Qwen35PrefillRecurrentStateModeArg,
    qwen35_alpha_beta_storage_mode: Qwen35PrefillAlphaBetaStorageModeArg,
    qwen35_prime_slot_buffers: bool,
    qwen35_prewarm_prefill_same_kv: bool,
    qwen35_force_backend_state_batch: bool,
    llama_parity_preset: bool,
    kernel_profile_path: Option<&str>,
) -> anyhow::Result<PrefillProfileResult> {
    let current_exe = std::env::current_exe().context("resolve current bench executable")?;
    let mut cmd = ProcessCommand::new(current_exe);
    cmd.arg("prefill-profile")
        .arg("--model")
        .arg(model)
        .arg("--prompt-tokens")
        .arg(prompt_tokens.to_string())
        .arg("--warmup-iters")
        .arg(warmup_iters.to_string())
        .arg("--local-hd128-route")
        .arg(local_hd128_route.cli_value())
        .arg("--qwen35-shared-timeline-slots")
        .arg(qwen35_shared_timeline_slots.to_string())
        .arg("--qwen35-recurrent-state-mode")
        .arg(qwen35_recurrent_state_mode.cli_value())
        .arg("--qwen35-alpha-beta-storage-mode")
        .arg(qwen35_alpha_beta_storage_mode.cli_value())
        .arg("--json");

    if let Some(source_slot) = qwen35_shared_timeline_source_slot {
        cmd.arg("--qwen35-shared-timeline-source-slot")
            .arg(source_slot.to_string());
    }
    if qwen35_prime_slot_buffers {
        cmd.arg("--qwen35-prime-slot-buffers");
    }
    if qwen35_prewarm_prefill_same_kv {
        cmd.arg("--qwen35-prewarm-prefill-same-kv");
    }
    if qwen35_force_backend_state_batch {
        cmd.arg("--qwen35-force-backend-state-batch");
    }
    if llama_parity_preset {
        cmd.arg("--llama-parity-preset");
    }
    if let Some(path) = kernel_profile_path {
        cmd.arg("--kernel-profile-path").arg(path);
    }

    let output = cmd.output().context("run prefill-profile subprocess")?;
    if !output.status.success() {
        return Err(anyhow!(
            "prefill-profile subprocess failed for route {} (status={}):\nstdout:\n{}\nstderr:\n{}",
            local_hd128_route.cli_value(),
            output
                .status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string()),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    serde_json::from_slice::<PrefillProfileResult>(&output.stdout).with_context(|| {
        format!(
            "parse prefill-profile JSON for route {}",
            local_hd128_route.cli_value()
        )
    })
}

fn build_prefill_route_compare_variant(
    route: LocalPrefillHd128Route,
    sample_results: Vec<PrefillProfileResult>,
) -> PrefillRouteCompareVariant {
    let tok_per_sec: Vec<f64> = sample_results.iter().map(|r| r.tok_per_sec).collect();
    let effective_tok_per_sec: Vec<f64> = sample_results
        .iter()
        .map(prefill_profile_effective_tok_per_sec)
        .collect();
    let total_ms: Vec<f64> = sample_results.iter().map(|r| r.total_ms).collect();
    let expected_prefix = route.expected_attention_route_prefix().map(str::to_owned);
    let mut observed_attention_routes: Vec<String> = sample_results
        .iter()
        .filter_map(|r| r.prefill_attention_route.clone())
        .collect();
    observed_attention_routes.sort();
    observed_attention_routes.dedup();
    let matched_expected_route = expected_prefix.as_ref().is_none_or(|prefix| {
        sample_results.iter().all(|result| {
            result
                .prefill_attention_route
                .as_deref()
                .is_some_and(|route| route.starts_with(prefix))
        })
    });

    PrefillRouteCompareVariant {
        route,
        expected_attention_route_prefix: expected_prefix,
        observed_attention_routes,
        matched_expected_route,
        sample_results,
        median_tok_per_sec: median_f64(&tok_per_sec),
        mean_tok_per_sec: mean_f64(&tok_per_sec),
        median_effective_tok_per_sec: median_f64(&effective_tok_per_sec),
        mean_effective_tok_per_sec: mean_f64(&effective_tok_per_sec),
        median_total_ms: median_f64(&total_ms),
    }
}

fn default_prefill_route_compare_routes() -> Vec<LocalPrefillHd128RouteArg> {
    vec![
        LocalPrefillHd128RouteArg::Auto,
        LocalPrefillHd128RouteArg::AxBc64,
        LocalPrefillHd128RouteArg::Fa2SimdHd128,
        LocalPrefillHd128RouteArg::Fa2HalfHd128,
    ]
}

#[derive(Debug, Serialize)]
struct SpecVerifyCompareDelta {
    verify_ms_per_step: f64,
    verify_ms_per_step_pct: f64,
    verify_prepare_ms_per_step: f64,
    verify_forward_ms_per_step: f64,
    verify_cleanup_ms_per_step: f64,
    decode_tok_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct SpecVerifyCompareReport {
    deterministic: bool,
    samples: usize,
    measure_iters: usize,
    cooldown_ms: u64,
    branch_on: perf::SpecBenchResult,
    branch_off: perf::SpecBenchResult,
    delta_on_minus_off: SpecVerifyCompareDelta,
}

fn build_spec_verify_compare_report(
    branch_on: perf::SpecBenchResult,
    branch_off: perf::SpecBenchResult,
    measure_iters: usize,
) -> SpecVerifyCompareReport {
    let delta_pct = |on: f64, off: f64| {
        if off.abs() <= f64::EPSILON {
            0.0
        } else {
            ((on - off) / off) * 100.0
        }
    };

    let delta = SpecVerifyCompareDelta {
        verify_ms_per_step: branch_on.verify_ms_per_step - branch_off.verify_ms_per_step,
        verify_ms_per_step_pct: delta_pct(
            branch_on.verify_ms_per_step,
            branch_off.verify_ms_per_step,
        ),
        verify_prepare_ms_per_step: branch_on.verify_prepare_ms_per_step
            - branch_off.verify_prepare_ms_per_step,
        verify_forward_ms_per_step: branch_on.verify_forward_ms_per_step
            - branch_off.verify_forward_ms_per_step,
        verify_cleanup_ms_per_step: branch_on.verify_cleanup_ms_per_step
            - branch_off.verify_cleanup_ms_per_step,
        decode_tok_per_sec: branch_on.decode_tok_per_sec - branch_off.decode_tok_per_sec,
    };

    SpecVerifyCompareReport {
        deterministic: branch_on.deterministic || branch_off.deterministic,
        samples: branch_on.samples.max(branch_off.samples),
        measure_iters,
        cooldown_ms: branch_on.cooldown_ms.max(branch_off.cooldown_ms),
        branch_on,
        branch_off,
        delta_on_minus_off: delta,
    }
}

fn print_spec_verify_compare(report: &SpecVerifyCompareReport) {
    let branch_on = &report.branch_on;
    let branch_off = &report.branch_off;
    let delta = &report.delta_on_minus_off;

    eprintln!();
    eprintln!("=== Qwen3.5 Verify Compare ===");
    if report.deterministic {
        eprintln!(
            "Mode:        deterministic (samples={}, measure-iters={}, cooldown={}ms)",
            report.samples, report.measure_iters, report.cooldown_ms,
        );
    } else {
        eprintln!(
            "Mode:        standard (measure-iters={})",
            report.measure_iters
        );
    }
    eprintln!(
        "Verify:      on {:.2} ms | off {:.2} ms | delta {:+.2} ms ({:+.1}%)",
        branch_on.verify_ms_per_step,
        branch_off.verify_ms_per_step,
        delta.verify_ms_per_step,
        delta.verify_ms_per_step_pct,
    );
    eprintln!(
        "Prep:        on {:.2} ms | off {:.2} ms | delta {:+.2} ms",
        branch_on.verify_prepare_ms_per_step,
        branch_off.verify_prepare_ms_per_step,
        delta.verify_prepare_ms_per_step,
    );
    eprintln!(
        "Forward:     on {:.2} ms | off {:.2} ms | delta {:+.2} ms",
        branch_on.verify_forward_ms_per_step,
        branch_off.verify_forward_ms_per_step,
        delta.verify_forward_ms_per_step,
    );
    eprintln!(
        "Cleanup:     on {:.2} ms | off {:.2} ms | delta {:+.2} ms",
        branch_on.verify_cleanup_ms_per_step,
        branch_off.verify_cleanup_ms_per_step,
        delta.verify_cleanup_ms_per_step,
    );
    eprintln!(
        "Decode:      on {:.2} tok/s | off {:.2} tok/s | delta {:+.2} tok/s",
        branch_on.decode_tok_per_sec, branch_off.decode_tok_per_sec, delta.decode_tok_per_sec,
    );
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
            qwen35_shared_timeline_slots,
            qwen35_shared_timeline_source_slot,
            qwen35_recurrent_state_mode,
            qwen35_alpha_beta_storage_mode,
            qwen35_prime_slot_buffers,
            qwen35_prewarm_prefill_same_kv,
            qwen35_force_backend_state_batch,
            local_hd128_route,
            json,
            json_output,
            llama_parity_preset,
            kernel_profile_path,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            apply_kernel_profile_override(kernel_profile_path.as_deref());
            let config = PrefillProfileConfig {
                model_path: model,
                prompt_tokens,
                warmup_iters,
                qwen35_shared_timeline_slots,
                qwen35_shared_timeline_source_slot,
                kernel_profile_path,
                qwen35_recurrent_state_mode: qwen35_recurrent_state_mode.into(),
                qwen35_alpha_beta_storage_mode: qwen35_alpha_beta_storage_mode.into(),
                qwen35_prime_slot_buffers,
                qwen35_prewarm_prefill_same_kv,
                qwen35_force_backend_state_batch,
                local_hd128_route: local_hd128_route.into(),
            };

            match with_prefill_profile_env_overrides(
                qwen35_recurrent_state_mode,
                qwen35_alpha_beta_storage_mode,
                qwen35_force_backend_state_batch,
                local_hd128_route.into(),
                || {
                    let backend = create_runtime_backend("Prefill profile");
                    ax_engine_core::scheduler::init_global_threadpool();
                    prefill_profile::run_prefill_profile_with_backend(&config, backend)
                },
            ) {
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
                    eprintln!("Prefill profile error: {e}");
                    process::exit(2);
                }
            }
        }

        Command::PrefillRouteCompare {
            model,
            prompt_tokens,
            warmup_iters,
            samples,
            route,
            qwen35_shared_timeline_slots,
            qwen35_shared_timeline_source_slot,
            qwen35_recurrent_state_mode,
            qwen35_alpha_beta_storage_mode,
            qwen35_prime_slot_buffers,
            qwen35_prewarm_prefill_same_kv,
            qwen35_force_backend_state_batch,
            json,
            json_output,
            llama_parity_preset,
            kernel_profile_path,
        } => {
            let routes = if route.is_empty() {
                default_prefill_route_compare_routes()
            } else {
                route
            };

            let variants_result = routes
                .iter()
                .map(|route| {
                    let route_label: LocalPrefillHd128Route = (*route).into();
                    eprintln!(
                        "Route compare: {} ({} samples)",
                        route_label.label(),
                        samples
                    );
                    let sample_results = (0..samples)
                        .map(|sample_idx| {
                            eprintln!("  sample {}/{}", sample_idx + 1, samples);
                            run_prefill_profile_subprocess(
                                &model,
                                prompt_tokens,
                                warmup_iters,
                                *route,
                                qwen35_shared_timeline_slots,
                                qwen35_shared_timeline_source_slot,
                                qwen35_recurrent_state_mode,
                                qwen35_alpha_beta_storage_mode,
                                qwen35_prime_slot_buffers,
                                qwen35_prewarm_prefill_same_kv,
                                qwen35_force_backend_state_batch,
                                llama_parity_preset,
                                kernel_profile_path.as_deref(),
                            )
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    Ok(build_prefill_route_compare_variant(
                        route_label,
                        sample_results,
                    ))
                })
                .collect::<anyhow::Result<Vec<_>>>();

            match variants_result {
                Ok(variants) => {
                    let winner_by_median_effective_tok_per_sec = variants
                        .iter()
                        .max_by(|a, b| {
                            a.median_effective_tok_per_sec
                                .total_cmp(&b.median_effective_tok_per_sec)
                        })
                        .map(|variant| variant.route);
                    let report = PrefillRouteCompareReport {
                        model,
                        prompt_tokens,
                        warmup_iters,
                        samples_per_route: samples,
                        kernel_profile_path,
                        qwen35_shared_timeline_slots,
                        qwen35_shared_timeline_source_slot,
                        qwen35_recurrent_state_mode,
                        qwen35_alpha_beta_storage_mode,
                        qwen35_prime_slot_buffers,
                        qwen35_prewarm_prefill_same_kv,
                        qwen35_force_backend_state_batch,
                        variants,
                        winner_by_median_effective_tok_per_sec,
                    };

                    report.print_summary();

                    if json {
                        match serde_json::to_string_pretty(&report) {
                            Ok(j) => println!("{j}"),
                            Err(e) => {
                                eprintln!("JSON serialization error: {e}");
                                process::exit(2);
                            }
                        }
                    }

                    if let Some(path) = json_output {
                        match serde_json::to_string_pretty(&report) {
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
                    eprintln!("Prefill route compare error: {e:?}");
                    process::exit(2);
                }
            }
        }

        Command::PrefillGap {
            model,
            prompt_tokens,
            warmup_iters,
            qwen35_shared_timeline_slots,
            qwen35_shared_timeline_source_slot,
            qwen35_recurrent_state_mode,
            qwen35_alpha_beta_storage_mode,
            qwen35_prime_slot_buffers,
            qwen35_prewarm_prefill_same_kv,
            qwen35_force_backend_state_batch,
            baseline_json,
            baseline_prefill_tok_s,
            baseline_label,
            json,
            json_output,
            llama_parity_preset,
            kernel_profile_path,
        } => {
            if llama_parity_preset {
                apply_llama_parity_preset();
            }
            apply_kernel_profile_override(kernel_profile_path.as_deref());
            let backend = create_runtime_backend("Prefill gap");
            ax_engine_core::scheduler::init_global_threadpool();
            let config = PrefillGapConfig {
                profile: PrefillProfileConfig {
                    model_path: model,
                    prompt_tokens,
                    warmup_iters,
                    qwen35_shared_timeline_slots,
                    qwen35_shared_timeline_source_slot,
                    kernel_profile_path: kernel_profile_path.clone(),
                    qwen35_recurrent_state_mode: qwen35_recurrent_state_mode.into(),
                    qwen35_alpha_beta_storage_mode: qwen35_alpha_beta_storage_mode.into(),
                    qwen35_prime_slot_buffers,
                    qwen35_prewarm_prefill_same_kv,
                    qwen35_force_backend_state_batch,
                    local_hd128_route: LocalPrefillHd128Route::Auto,
                },
                baseline_json,
                baseline_prefill_tok_per_sec: baseline_prefill_tok_s,
                baseline_label,
            };

            match with_env_var_override(
                "AX_QWEN35_PREFILL_RECURRENT_STATE_MODE",
                qwen35_prefill_recurrent_state_mode_env(qwen35_recurrent_state_mode),
                || {
                    with_env_var_override(
                        "AX_QWEN35_PREFILL_ALPHA_BETA_STORAGE_MODE",
                        qwen35_prefill_alpha_beta_storage_mode_env(qwen35_alpha_beta_storage_mode),
                        || {
                            with_env_var_override(
                                "AX_QWEN35_PREFILL_FORCE_BACKEND_STATE_BATCH",
                                qwen35_prefill_force_backend_state_batch_env(
                                    qwen35_force_backend_state_batch,
                                ),
                                || prefill_gap::run_prefill_gap_with_backend(&config, backend),
                            )
                        },
                    )
                },
            ) {
                Ok(report) => {
                    report.print_summary();

                    if json {
                        match report.to_json() {
                            Ok(j) => println!("{j}"),
                            Err(e) => {
                                eprintln!("JSON serialization error: {e}");
                                process::exit(2);
                            }
                        }
                    }

                    if let Some(path) = json_output {
                        match report.to_json() {
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
                    eprintln!("Prefill gap error: {e}");
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
            qwen35_shared_timeline_slots,
            qwen35_shared_timeline_source_slot,
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
                qwen35_shared_timeline_slots,
                qwen35_shared_timeline_source_slot,
                kernel_profile_path,
            };

            match perf::run_benchmark_with_backend(&config, backend) {
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
            qwen35_spec_verify_branch,
            qwen35_spec_verify_compare,
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

            if qwen35_spec_verify_compare {
                let branch_on =
                    with_env_var_override("AX_QWEN35_SPEC_VERIFY_BRANCH", Some("on"), || {
                        perf::run_speculative_benchmark_with_backend(
                            &config,
                            create_runtime_backend("Speculative benchmark"),
                        )
                    });
                let branch_off =
                    with_env_var_override("AX_QWEN35_SPEC_VERIFY_BRANCH", Some("off"), || {
                        perf::run_speculative_benchmark_with_backend(
                            &config,
                            create_runtime_backend("Speculative benchmark"),
                        )
                    });

                match (branch_on, branch_off) {
                    (Ok(branch_on), Ok(branch_off)) => {
                        let report = build_spec_verify_compare_report(
                            branch_on,
                            branch_off,
                            config.measure_iters.max(1),
                        );
                        report.branch_on.print_summary();
                        report.branch_off.print_summary();
                        print_spec_verify_compare(&report);

                        if json {
                            match serde_json::to_string_pretty(&report) {
                                Ok(j) => println!("{j}"),
                                Err(e) => {
                                    eprintln!("JSON serialization error: {e}");
                                    process::exit(2);
                                }
                            }
                        }

                        if let Some(path) = json_output {
                            match serde_json::to_string_pretty(&report) {
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
                    (Err(e), _) | (_, Err(e)) => {
                        eprintln!("Speculative benchmark error: {e}");
                        process::exit(2);
                    }
                }
                return;
            }

            match with_env_var_override(
                "AX_QWEN35_SPEC_VERIFY_BRANCH",
                qwen35_spec_verify_branch_env(qwen35_spec_verify_branch),
                || perf::run_speculative_benchmark_with_backend(&config, backend),
            ) {
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
