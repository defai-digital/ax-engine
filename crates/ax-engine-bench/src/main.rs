#![recursion_limit = "512"]
#![allow(clippy::collapsible_if, clippy::single_match, dead_code)]

use ax_engine_core::{
    CacheGroupId, EngineCore, MetalBuildDoctorReport, MetalBuildHostReport, MetalBuildStatus,
    MetalBuildToolStatus, MetalBuildToolchainReport, MetalDispatchTrace, MetalKernelBuildRequest,
    NativeModelArtifacts, NativeModelBindingSummary, RequestId, RequestSnapshot, RequestSubmission,
    RouteMetadata, SamplingParams, SequenceNo, StepId, StopReason, build_phase1_kernel_artifacts,
};
use ax_engine_sdk::{
    BackendPolicy, EngineSession, EngineSessionConfig, EngineStepReport, GenerateFinishReason,
    GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateSampling, GenerateStatus,
    GenerateStreamEvent, HostReport, LlamaCppConfig, MetalToolchainReport,
    NativeModelArtifactsSource, NativeModelReport, NativeRuntimeArtifactsSource,
    NativeRuntimeReport, PreviewBackendRequest, ResolutionPolicy, ResolvedBackend,
    ResolvedSessionConfigRequest, RuntimeReport, SelectedBackend, SessionRequestReport,
    SessionRequestState, SupportTier, ToolStatusReport, current_host_report,
    current_metal_toolchain_report, preview_support_tier_from_label,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::process::ExitCode;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing_subscriber::EnvFilter;

const NATIVE_DENSE_DEQUANTIZED_EXPORT_NOTE: &str =
    "source_quantization_dequantized_for_dense_native_export";
const NATIVE_DENSE_DEQUANTIZED_SOURCE_BLOCKER: &str =
    "source_quantization_dequantized_dense_native_artifact";

fn main() -> ExitCode {
    init_tracing();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(message)) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
        Err(CliError::Runtime(message)) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
        Err(CliError::Contract(message)) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
        Err(CliError::Correctness(message)) => {
            eprintln!("{message}");
            ExitCode::from(3)
        }
        Err(CliError::Performance(message)) => {
            eprintln!("{message}");
            ExitCode::from(4)
        }
    }
}

fn init_tracing() {
    let filter = env::var("AX_BENCH_LOG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("RUST_LOG")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });

    let Some(filter) = filter else {
        return;
    };
    let Ok(env_filter) = EnvFilter::try_new(filter) else {
        return;
    };

    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_ansi(false)
        .compact()
        .try_init();
}

fn run() -> Result<(), CliError> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return Err(CliError::Usage(usage()));
    };

    let remaining: Vec<String> = args.collect();
    match command.as_str() {
        "generate" => handle_generate(&remaining),
        "stream" => handle_stream(&remaining),
        "scenario" => handle_scenario(&remaining),
        "replay" => handle_replay(&remaining),
        "autotune" => handle_autotune(&remaining),
        "compare" => handle_compare(&remaining),
        "matrix-compare" => handle_matrix_compare(&remaining),
        "baseline" => handle_baseline(&remaining),
        "matrix" => handle_matrix(&remaining),
        "doctor" => handle_doctor(&remaining),
        "metal-build" => handle_metal_build(&remaining),
        "help" | "--help" | "-h" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(CliError::Usage(format!(
            "unknown subcommand: {other}\n\n{}",
            usage()
        ))),
    }
}

fn handle_scenario(args: &[String]) -> Result<(), CliError> {
    let manifest = required_flag(args, "--manifest")?;
    let output_root = required_flag(args, "--output-root")?;

    require_existing_file(&manifest)?;
    ensure_output_root(&output_root)?;
    let manifest_data = load_manifest(&manifest)?;
    let started_at_unix_s = unix_timestamp_secs()?;
    let execution = match validate_manifest(&manifest_data, ManifestClass::Scenario)
        .and_then(|_| execute_manifest_runtime(&manifest_data))
    {
        Ok(execution) => execution,
        Err(CliError::Contract(message)) => {
            let artifact_dir = write_contract_failure_artifacts(
                "scenario",
                &manifest,
                &manifest_data,
                &output_root,
                started_at_unix_s,
                &message,
            )?;
            println!(
                "ax-engine-bench scenario\nmanifest={}\noutput_root={}\nresult_dir={}\nstatus=contract_failure",
                manifest.display(),
                output_root.display(),
                artifact_dir.display(),
            );
            return Err(CliError::Contract(format!(
                "{message}\nartifact_dir={}",
                artifact_dir.display()
            )));
        }
        Err(error) => return Err(error),
    };
    let artifact_dir = write_execution_artifacts(
        "scenario",
        &manifest,
        &manifest_data,
        &output_root,
        started_at_unix_s,
        &execution,
    )?;

    println!(
        "ax-engine-bench scenario\nmanifest={}\noutput_root={}\nresult_dir={}\nstatus={}",
        manifest.display(),
        output_root.display(),
        artifact_dir.display(),
        execution.status_label(),
    );

    enforce_runtime_gates(&execution)
}

fn handle_replay(args: &[String]) -> Result<(), CliError> {
    let manifest = required_flag(args, "--manifest")?;
    let output_root = required_flag(args, "--output-root")?;

    require_existing_file(&manifest)?;
    ensure_output_root(&output_root)?;
    let manifest_data = load_manifest(&manifest)?;
    let started_at_unix_s = unix_timestamp_secs()?;
    let execution = match validate_manifest(&manifest_data, ManifestClass::Replay)
        .and_then(|_| execute_manifest_runtime(&manifest_data))
    {
        Ok(execution) => execution,
        Err(CliError::Contract(message)) => {
            let artifact_dir = write_contract_failure_artifacts(
                "replay",
                &manifest,
                &manifest_data,
                &output_root,
                started_at_unix_s,
                &message,
            )?;
            println!(
                "ax-engine-bench replay\nmanifest={}\noutput_root={}\nresult_dir={}\nstatus=contract_failure",
                manifest.display(),
                output_root.display(),
                artifact_dir.display(),
            );
            return Err(CliError::Contract(format!(
                "{message}\nartifact_dir={}",
                artifact_dir.display()
            )));
        }
        Err(error) => return Err(error),
    };
    let artifact_dir = write_execution_artifacts(
        "replay",
        &manifest,
        &manifest_data,
        &output_root,
        started_at_unix_s,
        &execution,
    )?;

    println!(
        "ax-engine-bench replay\nmanifest={}\noutput_root={}\nresult_dir={}\nstatus={}",
        manifest.display(),
        output_root.display(),
        artifact_dir.display(),
        execution.status_label(),
    );

    enforce_runtime_gates(&execution)
}

fn handle_autotune(args: &[String]) -> Result<(), CliError> {
    let _ = parse_autotune_args(args)?;
    Err(CliError::Contract(
        "autotune is not supported; use MLX mode or llama.cpp".to_string(),
    ))
}

#[derive(Clone, Debug, PartialEq)]
struct AutotuneArgs {
    manifest_path: PathBuf,
    output_root: PathBuf,
    iterations: usize,
    exploration_weight: f64,
    max_batch_token_options: Option<Vec<u32>>,
    kv_total_block_options: Option<Vec<Option<u32>>>,
    prefix_cache_options: Option<Vec<bool>>,
    disable_history: bool,
}

fn parse_autotune_args(args: &[String]) -> Result<AutotuneArgs, CliError> {
    let manifest_path = required_flag(args, "--manifest")?;
    let output_root = required_flag(args, "--output-root")?;
    let mut iterations = 8_usize;
    let mut exploration_weight = 0.5_f64;
    let mut max_batch_token_options = None;
    let mut kv_total_block_options = None;
    let mut prefix_cache_options = None;
    let mut disable_history = false;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--manifest" | "--output-root" => {
                let _ = next_flag_value(&mut iter, arg)?;
            }
            "--iterations" => {
                iterations = parse_flag_value::<usize>(
                    next_flag_value(&mut iter, "--iterations")?,
                    "--iterations",
                )?;
            }
            "--exploration-weight" => {
                exploration_weight = parse_flag_value::<f64>(
                    next_flag_value(&mut iter, "--exploration-weight")?,
                    "--exploration-weight",
                )?;
            }
            "--max-batch-token-options" => {
                max_batch_token_options = Some(parse_u32_list(
                    next_flag_value(&mut iter, "--max-batch-token-options")?,
                    "--max-batch-token-options",
                )?);
            }
            "--kv-total-block-options" => {
                kv_total_block_options = Some(parse_optional_u32_list(
                    next_flag_value(&mut iter, "--kv-total-block-options")?,
                    "--kv-total-block-options",
                )?);
            }
            "--prefix-cache-options" => {
                prefix_cache_options = Some(parse_bool_list(
                    next_flag_value(&mut iter, "--prefix-cache-options")?,
                    "--prefix-cache-options",
                )?);
            }
            "--disable-history" => {
                disable_history = true;
            }
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag for autotune: {other}\n\n{}",
                    usage()
                )));
            }
        }
    }

    if iterations == 0 {
        return Err(CliError::Usage(
            "--iterations must be greater than zero".to_string(),
        ));
    }
    if !exploration_weight.is_finite() || exploration_weight < 0.0 {
        return Err(CliError::Usage(
            "--exploration-weight must be a finite non-negative number".to_string(),
        ));
    }

    Ok(AutotuneArgs {
        manifest_path,
        output_root,
        iterations,
        exploration_weight,
        max_batch_token_options,
        kv_total_block_options,
        prefix_cache_options,
        disable_history,
    })
}

fn handle_compare(args: &[String]) -> Result<(), CliError> {
    let baseline = required_flag(args, "--baseline")?;
    let candidate = required_flag(args, "--candidate")?;
    let output_root = required_flag(args, "--output-root")?;

    require_existing_path(&baseline)?;
    require_existing_path(&candidate)?;
    reject_contract_failure_artifact_dir(&baseline, "baseline")?;
    reject_contract_failure_artifact_dir(&candidate, "candidate")?;
    ensure_output_root(&output_root)?;
    let comparison_dir = write_compare_artifacts(&baseline, &candidate, &output_root)?;

    println!(
        "ax-engine-bench compare\nbaseline={}\ncandidate={}\noutput_root={}\nresult_dir={}",
        baseline.display(),
        candidate.display(),
        output_root.display(),
        comparison_dir.display()
    );
    Ok(())
}

fn handle_baseline(args: &[String]) -> Result<(), CliError> {
    let source = required_flag(args, "--source")?;
    let name = required_string_flag(args, "--name")?;
    let output_root = required_flag(args, "--output-root")?;

    require_existing_dir(&source)?;
    reject_contract_failure_artifact_dir(&source, "source")?;
    ensure_output_root(&output_root)?;
    let baseline_dir = write_trusted_baseline_artifacts(&source, &name, &output_root)?;

    println!(
        "ax-engine-bench baseline\nsource={}\nname={}\noutput_root={}\nresult_dir={}",
        source.display(),
        name,
        output_root.display(),
        baseline_dir.display()
    );
    Ok(())
}

fn handle_matrix_compare(args: &[String]) -> Result<(), CliError> {
    let baseline = required_flag(args, "--baseline")?;
    let candidate = required_flag(args, "--candidate")?;
    let output_root = required_flag(args, "--output-root")?;

    require_existing_dir(&baseline)?;
    require_existing_dir(&candidate)?;
    ensure_output_root(&output_root)?;
    let comparison_dir = write_matrix_compare_artifacts(&baseline, &candidate, &output_root)?;

    println!(
        "ax-engine-bench matrix-compare\nbaseline={}\ncandidate={}\noutput_root={}\nresult_dir={}",
        baseline.display(),
        candidate.display(),
        output_root.display(),
        comparison_dir.display()
    );
    Ok(())
}

fn handle_matrix(args: &[String]) -> Result<(), CliError> {
    let manifest = required_flag(args, "--manifest")?;
    let output_root = required_flag(args, "--output-root")?;

    require_existing_file(&manifest)?;
    ensure_output_root(&output_root)?;
    let matrix_manifest = load_matrix_manifest(&manifest)?;
    validate_matrix_manifest(&matrix_manifest)?;
    let execution = execute_matrix_manifest(&manifest, &matrix_manifest, &output_root)?;

    println!(
        "ax-engine-bench matrix\nmanifest={}\noutput_root={}\nresult_dir={}\nstatus={}",
        manifest.display(),
        output_root.display(),
        execution.result_dir.display(),
        execution.overall_status
    );
    enforce_matrix_gates(&execution)
}

fn handle_doctor(args: &[String]) -> Result<(), CliError> {
    let doctor_args = parse_doctor_args(args)?;
    let report = build_doctor_report_for_model(
        current_host_report(),
        current_metal_toolchain_report(),
        doctor_args.mlx_model_artifacts_dir.as_deref(),
    );

    if doctor_args.json {
        let json = serde_json::to_string_pretty(&report).map_err(|error| {
            CliError::Runtime(format!("failed to serialize doctor report: {error}"))
        })?;
        println!("{json}");
    } else {
        println!("{}", render_doctor_report(&report));
    }

    Ok(())
}

fn handle_metal_build(args: &[String]) -> Result<(), CliError> {
    let metal_build_args = parse_metal_build_args(args)?;
    let doctor = build_doctor_report(current_host_report(), current_metal_toolchain_report());
    let artifacts = build_phase1_kernel_artifacts(&MetalKernelBuildRequest {
        manifest_path: metal_build_args.manifest_path.clone(),
        output_dir: metal_build_args.output_dir.clone(),
        doctor: metal_build_doctor_report(&doctor),
        toolchain_path_override: None,
    })
    .map_err(map_metal_build_error)?;

    println!(
        "ax-metal-build\nmanifest={}\noutput_dir={}\nstatus={}\nreused_existing_artifacts={}",
        artifacts.manifest_path.display(),
        artifacts.output_dir.display(),
        metal_build_status_label(artifacts.build_status()),
        artifacts.reused_existing_artifacts(),
    );
    if let Some(reason) = artifacts.build_report.reason.as_deref() {
        println!("reason={reason}");
    }

    if artifacts.build_status() == MetalBuildStatus::FailedCompile {
        return Err(CliError::Runtime(
            artifacts
                .build_report
                .reason
                .unwrap_or_else(|| "metal kernel build failed".to_string()),
        ));
    }

    Ok(())
}

fn handle_generate(args: &[String]) -> Result<(), CliError> {
    let inference_args = parse_inference_args(args, "generate")?;
    let response = run_inference_generate(&inference_args)?;
    print!(
        "{}",
        render_generate_response(&response, inference_args.json)?
    );
    Ok(())
}

fn handle_stream(args: &[String]) -> Result<(), CliError> {
    let inference_args = parse_inference_args(args, "stream")?;
    let mut session = build_inference_session(&inference_args)?;
    let stream = session
        .stream_generate(inference_args.generate_request())
        .map_err(|error| CliError::Runtime(format!("stream request failed: {error}")))?;

    let mut stdout = io::stdout();
    for event in stream {
        let event =
            event.map_err(|error| CliError::Runtime(format!("stream request failed: {error}")))?;
        stdout
            .write_all(render_stream_event(&event, inference_args.json)?.as_bytes())
            .map_err(|error| {
                CliError::Runtime(format!("failed to write stream output: {error}"))
            })?;
        stdout.flush().map_err(|error| {
            CliError::Runtime(format!("failed to flush stream output: {error}"))
        })?;
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct InferenceArgs {
    model_id: String,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    sampling: GenerateSampling,
    metadata: Option<String>,
    deterministic: bool,
    mlx: bool,
    support_tier: SupportTier,
    llama_cli_path: PathBuf,
    llama_model_path: Option<PathBuf>,
    llama_server_url: Option<String>,
    mlx_lm_server_url: Option<String>,
    mlx_model_artifacts_dir: Option<PathBuf>,
    json: bool,
}

impl Default for InferenceArgs {
    fn default() -> Self {
        Self {
            model_id: "qwen3_dense".to_string(),
            input_tokens: Vec::new(),
            input_text: None,
            max_output_tokens: 32,
            sampling: GenerateSampling::default(),
            metadata: None,
            deterministic: true,
            mlx: false,
            support_tier: SupportTier::LlamaCpp,
            llama_cli_path: PathBuf::from("llama-cli"),
            llama_model_path: None,
            llama_server_url: None,
            mlx_lm_server_url: None,
            mlx_model_artifacts_dir: None,
            json: false,
        }
    }
}

impl InferenceArgs {
    fn generate_request(&self) -> GenerateRequest {
        GenerateRequest {
            model_id: self.model_id.clone(),
            input_tokens: self.input_tokens.clone(),
            input_text: self.input_text.clone(),
            max_output_tokens: self.max_output_tokens,
            sampling: self.sampling.clone(),
            stop_sequences: Vec::new(),
            metadata: self.metadata.clone(),
        }
    }
}

fn parse_inference_args(args: &[String], command: &str) -> Result<InferenceArgs, CliError> {
    let mut parsed = InferenceArgs::default();
    let mut support_tier_label = "llama_cpp".to_string();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model-id" => parsed.model_id = next_flag_value(&mut iter, "--model-id")?.to_string(),
            "--prompt" => {
                parsed.input_text = Some(next_flag_value(&mut iter, "--prompt")?.to_string())
            }
            "--tokens" => {
                parsed.input_tokens = parse_token_list(next_flag_value(&mut iter, "--tokens")?)?
            }
            "--max-output-tokens" => {
                parsed.max_output_tokens = parse_flag_value::<u32>(
                    next_flag_value(&mut iter, "--max-output-tokens")?,
                    "--max-output-tokens",
                )?
            }
            "--temperature" => {
                parsed.sampling.temperature = parse_flag_value::<f32>(
                    next_flag_value(&mut iter, "--temperature")?,
                    "--temperature",
                )?
            }
            "--top-p" => {
                parsed.sampling.top_p =
                    parse_flag_value::<f32>(next_flag_value(&mut iter, "--top-p")?, "--top-p")?
            }
            "--top-k" => {
                parsed.sampling.top_k =
                    parse_flag_value::<u32>(next_flag_value(&mut iter, "--top-k")?, "--top-k")?
            }
            "--repetition-penalty" => {
                parsed.sampling.repetition_penalty = parse_flag_value::<f32>(
                    next_flag_value(&mut iter, "--repetition-penalty")?,
                    "--repetition-penalty",
                )?
            }
            "--seed" => {
                parsed.sampling.seed =
                    parse_flag_value::<u64>(next_flag_value(&mut iter, "--seed")?, "--seed")?
            }
            "--metadata" => {
                parsed.metadata = Some(next_flag_value(&mut iter, "--metadata")?.to_string())
            }
            "--deterministic" => {
                parsed.deterministic = parse_flag_value::<bool>(
                    next_flag_value(&mut iter, "--deterministic")?,
                    "--deterministic",
                )?
            }
            "--mlx" => parsed.mlx = true,
            "--support-tier" => {
                support_tier_label = next_flag_value(&mut iter, "--support-tier")?.to_string()
            }
            "--llama-cli-path" => {
                parsed.llama_cli_path =
                    PathBuf::from(next_flag_value(&mut iter, "--llama-cli-path")?)
            }
            "--llama-model-path" => {
                parsed.llama_model_path = Some(PathBuf::from(next_flag_value(
                    &mut iter,
                    "--llama-model-path",
                )?))
            }
            "--llama-server-url" => {
                parsed.llama_server_url =
                    Some(next_flag_value(&mut iter, "--llama-server-url")?.to_string())
            }
            "--mlx-lm-server-url" => {
                parsed.mlx_lm_server_url =
                    Some(next_flag_value(&mut iter, "--mlx-lm-server-url")?.to_string())
            }
            "--mlx-model-artifacts-dir" => {
                parsed.mlx_model_artifacts_dir = Some(PathBuf::from(next_flag_value(
                    &mut iter,
                    "--mlx-model-artifacts-dir",
                )?))
            }
            "--json" => parsed.json = true,
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag for {command}: {other}\n\n{}",
                    usage()
                )));
            }
        }
    }

    if parsed.input_text.is_some() && !parsed.input_tokens.is_empty() {
        return Err(CliError::Usage(format!(
            "{command} accepts either --prompt or --tokens, not both"
        )));
    }

    if parsed.input_text.is_none() && parsed.input_tokens.is_empty() {
        return Err(CliError::Usage(format!(
            "{command} requires either --prompt or --tokens"
        )));
    }

    parsed.support_tier = preview_support_tier_from_label(&support_tier_label)
        .map_err(|error| CliError::Usage(format!("invalid --support-tier: {error}")))?;
    parsed.sampling.deterministic = Some(parsed.deterministic);

    Ok(parsed)
}

fn next_flag_value<'a>(
    iter: &mut std::slice::Iter<'a, String>,
    name: &str,
) -> Result<&'a str, CliError> {
    iter.next()
        .map(|value| value.as_str())
        .ok_or_else(|| CliError::Usage(format!("missing value for required flag {name}")))
}

fn parse_flag_value<T>(value: &str, name: &str) -> Result<T, CliError>
where
    T: std::str::FromStr,
{
    value.parse::<T>().map_err(|_| {
        CliError::Usage(format!(
            "invalid value for {name}: expected {}, got {value}",
            std::any::type_name::<T>()
        ))
    })
}

fn parse_u32_list(value: &str, name: &str) -> Result<Vec<u32>, CliError> {
    let parts = split_list_parts(value);
    if parts.is_empty() {
        return Err(CliError::Usage(format!(
            "{name} expects a comma- or space-separated list"
        )));
    }
    Ok(unique_sorted_u32(
        parts
            .into_iter()
            .map(|part| parse_flag_value::<u32>(part, name))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

fn parse_optional_u32_list(value: &str, name: &str) -> Result<Vec<Option<u32>>, CliError> {
    let parts = split_list_parts(value);
    if parts.is_empty() {
        return Err(CliError::Usage(format!(
            "{name} expects a comma- or space-separated list"
        )));
    }
    Ok(unique_sorted_option_u32(
        parts
            .into_iter()
            .map(|part| {
                if part.eq_ignore_ascii_case("none") {
                    Ok(None)
                } else {
                    Ok(Some(parse_flag_value::<u32>(part, name)?))
                }
            })
            .collect::<Result<Vec<_>, CliError>>()?,
    ))
}

fn parse_bool_list(value: &str, name: &str) -> Result<Vec<bool>, CliError> {
    let parts = split_list_parts(value);
    if parts.is_empty() {
        return Err(CliError::Usage(format!(
            "{name} expects a comma- or space-separated list"
        )));
    }
    Ok(unique_sorted_bool(
        parts
            .into_iter()
            .map(|part| parse_flag_value::<bool>(part, name))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

fn split_list_parts(value: &str) -> Vec<&str> {
    value
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect()
}

fn parse_token_list(value: &str) -> Result<Vec<u32>, CliError> {
    split_list_parts(value)
        .into_iter()
        .map(|part| {
            part.parse::<u32>().map_err(|_| {
                CliError::Usage(format!(
                    "invalid token id {part}; --tokens expects a comma- or space-separated u32 list"
                ))
            })
        })
        .collect()
}

fn build_inference_session(args: &InferenceArgs) -> Result<EngineSession, CliError> {
    let backend_request = if args.mlx {
        PreviewBackendRequest::shipping_mlx()
    } else if args.support_tier == SupportTier::LlamaCpp {
        PreviewBackendRequest::shipping_default_llama_cpp(
            args.llama_cli_path.clone(),
            args.llama_model_path.clone(),
            args.llama_server_url.clone(),
        )
    } else if args.support_tier == SupportTier::MlxLmDelegated {
        PreviewBackendRequest {
            support_tier: SupportTier::MlxLmDelegated,
            mlx_lm_server_url: args.mlx_lm_server_url.clone(),
            ..PreviewBackendRequest::default()
        }
    } else {
        return Err(CliError::Usage(
            "non-MLX inference routes to explicit delegated backends: llama_cpp or mlx_lm_delegated; pass --mlx for AX-owned MLX inference"
                .to_string(),
        ));
    };

    let mlx_model_artifacts_dir = if args.mlx {
        args.mlx_model_artifacts_dir
            .clone()
            .or_else(|| args.llama_model_path.clone())
    } else {
        args.mlx_model_artifacts_dir.clone()
    };

    let config =
        EngineSessionConfig::from_preview_request(ax_engine_sdk::PreviewSessionConfigRequest {
            cache_group_id: CacheGroupId(0),
            block_size_tokens: 16,
            total_blocks: 1024,
            deterministic: args.deterministic,
            max_batch_tokens: 2048,
            mlx_runtime_artifacts_dir: None,
            backend_request,
            mlx_model_artifacts_dir,
            mlx_disable_ngram_acceleration: false,
            mlx_kv_compression: ax_engine_sdk::MlxKvCompressionConfig::disabled(),
        })
        .map_err(|error| CliError::Usage(format!("invalid inference configuration: {error}")))?;

    EngineSession::new(config)
        .map_err(|error| CliError::Runtime(format!("failed to start AX Engine session: {error}")))
}

fn run_inference_generate(args: &InferenceArgs) -> Result<GenerateResponse, CliError> {
    let mut session = build_inference_session(args)?;
    session
        .generate(args.generate_request())
        .map_err(|error| CliError::Runtime(format!("generate request failed: {error}")))
}

#[cfg_attr(not(test), allow(dead_code))]
fn collect_inference_stream_events(
    args: &InferenceArgs,
) -> Result<Vec<GenerateStreamEvent>, CliError> {
    let mut session = build_inference_session(args)?;
    session
        .stream_generate(args.generate_request())
        .map_err(|error| CliError::Runtime(format!("stream request failed: {error}")))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| CliError::Runtime(format!("stream request failed: {error}")))
}

fn render_generate_response(
    response: &GenerateResponse,
    json_output: bool,
) -> Result<String, CliError> {
    if json_output {
        return serde_json::to_string_pretty(response)
            .map(|json| format!("{json}\n"))
            .map_err(|error| {
                CliError::Runtime(format!("failed to serialize generate response: {error}"))
            });
    }

    if let Some(output_text) = response.output_text.as_deref() {
        let mut rendered = output_text.to_string();
        if !rendered.ends_with('\n') {
            rendered.push('\n');
        }
        rendered.push_str(&format_generate_metadata_suffix(response));
        return Ok(rendered);
    }

    let rendered_tokens = response
        .output_tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let mut rendered = format!("{rendered_tokens}\n");
    rendered.push_str(&format_generate_metadata_suffix(response));
    Ok(rendered)
}

fn render_stream_event(event: &GenerateStreamEvent, json_output: bool) -> Result<String, CliError> {
    if json_output {
        return serde_json::to_string(event)
            .map(|json| format!("{json}\n"))
            .map_err(|error| {
                CliError::Runtime(format!("failed to serialize stream event: {error}"))
            });
    }

    let rendered = match event {
        GenerateStreamEvent::Request(payload) => format!(
            "request id={} backend={} support_tier={} state={} execution_plan={}\n",
            payload.request.request_id,
            selected_backend_label(payload.runtime.selected_backend),
            support_tier_label(payload.runtime.support_tier),
            request_state_label(payload.request.state),
            optional_route_label(payload.request.route.execution_plan.as_deref()),
        ),
        GenerateStreamEvent::Step(payload) => {
            let finish_reason = payload
                .request
                .finish_reason
                .map(generate_finish_reason_label)
                .unwrap_or("none");
            let delta_text = payload.delta_text.as_deref().unwrap_or("");
            format!(
                "step id={} state={} execution_plan={} delta_tokens={:?} delta_token_logprobs={:?} delta_text={delta_text:?} total_output_tokens={} finish_reason={finish_reason}\n",
                payload.request.request_id,
                request_state_label(payload.request.state),
                optional_route_label(payload.request.route.execution_plan.as_deref()),
                payload.delta_tokens,
                payload.delta_token_logprobs,
                payload.request.output_tokens.len(),
            )
        }
        GenerateStreamEvent::Response(payload) => {
            let finish_reason = payload
                .response
                .finish_reason
                .map(generate_finish_reason_label)
                .unwrap_or("none");
            if let Some(output_text) = payload.response.output_text.as_deref() {
                format!(
                    "response id={} status={} finish_reason={} execution_plan={} output_text={output_text:?} output_token_logprobs={:?}\n",
                    payload.response.request_id,
                    generate_status_label(payload.response.status),
                    finish_reason,
                    optional_route_label(payload.response.route.execution_plan.as_deref()),
                    payload.response.output_token_logprobs,
                )
            } else {
                format!(
                    "response id={} status={} finish_reason={} execution_plan={} output_tokens={:?} output_token_logprobs={:?}\n",
                    payload.response.request_id,
                    generate_status_label(payload.response.status),
                    finish_reason,
                    optional_route_label(payload.response.route.execution_plan.as_deref()),
                    payload.response.output_tokens,
                    payload.response.output_token_logprobs,
                )
            }
        }
    };

    Ok(rendered)
}

fn format_generate_metadata_suffix(response: &GenerateResponse) -> String {
    let finish_reason = response
        .finish_reason
        .map(generate_finish_reason_label)
        .unwrap_or("none");
    let execution_plan = optional_route_label(response.route.execution_plan.as_deref());
    format!(
        "request_id={}\nstatus={}\nfinish_reason={}\nexecution_plan={}\noutput_token_logprobs={:?}\n",
        response.request_id,
        generate_status_label(response.status),
        finish_reason,
        execution_plan,
        response.output_token_logprobs,
    )
}

fn selected_backend_label(backend: SelectedBackend) -> &'static str {
    match backend {
        SelectedBackend::Mlx => "mlx",
        SelectedBackend::MlxLmDelegated => "mlx_lm_delegated",
        SelectedBackend::LlamaCpp => "llama_cpp",
    }
}

fn support_tier_label(support_tier: SupportTier) -> &'static str {
    match support_tier {
        SupportTier::MlxCertified => "mlx_certified",
        SupportTier::MlxPreview => "mlx_preview",
        SupportTier::MlxLmDelegated => "mlx_lm_delegated",
        SupportTier::LlamaCpp => "llama_cpp",
        SupportTier::Unsupported => "unsupported",
    }
}

fn request_state_label(state: SessionRequestState) -> &'static str {
    match state {
        SessionRequestState::Waiting => "waiting",
        SessionRequestState::Runnable => "runnable",
        SessionRequestState::Running => "running",
        SessionRequestState::BlockedOnMemory => "blocked_on_memory",
        SessionRequestState::Finished => "finished",
        SessionRequestState::Cancelled => "cancelled",
        SessionRequestState::Failed => "failed",
    }
}

fn generate_status_label(status: GenerateStatus) -> &'static str {
    match status {
        GenerateStatus::Pending => "pending",
        GenerateStatus::Finished => "finished",
        GenerateStatus::Cancelled => "cancelled",
        GenerateStatus::Failed => "failed",
    }
}

fn generate_finish_reason_label(finish_reason: GenerateFinishReason) -> &'static str {
    match finish_reason {
        GenerateFinishReason::Stop => "stop",
        GenerateFinishReason::MaxOutputTokens => "max_output_tokens",
        GenerateFinishReason::Cancelled => "cancelled",
        GenerateFinishReason::Error => "error",
    }
}

fn optional_route_label(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}

fn optional_u32_label(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn stop_reason_from_generate_finish_reason(
    finish_reason: Option<GenerateFinishReason>,
) -> Option<StopReason> {
    match finish_reason {
        Some(GenerateFinishReason::Stop) => Some(StopReason::EosToken),
        Some(GenerateFinishReason::MaxOutputTokens) => Some(StopReason::MaxOutputTokens),
        Some(GenerateFinishReason::Cancelled) => Some(StopReason::Cancelled),
        Some(GenerateFinishReason::Error) => Some(StopReason::Error),
        None => None,
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DoctorArgs {
    json: bool,
    mlx_model_artifacts_dir: Option<PathBuf>,
}

fn parse_doctor_args(args: &[String]) -> Result<DoctorArgs, CliError> {
    let mut doctor_args = DoctorArgs::default();
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--json" => doctor_args.json = true,
            "--mlx-model-artifacts-dir" => {
                let Some(value) = iter.next() else {
                    return Err(CliError::Usage(
                        "missing value for flag --mlx-model-artifacts-dir".to_string(),
                    ));
                };
                doctor_args.mlx_model_artifacts_dir = Some(PathBuf::from(value));
            }
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag for doctor: {other}\n\n{}",
                    usage()
                )));
            }
        }
    }

    Ok(doctor_args)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MetalBuildArgs {
    manifest_path: PathBuf,
    output_dir: PathBuf,
}

fn parse_metal_build_args(args: &[String]) -> Result<MetalBuildArgs, CliError> {
    let current_dir = env::current_dir().map_err(|error| {
        CliError::Runtime(format!(
            "failed to resolve current working directory: {error}"
        ))
    })?;
    let mut manifest_path = env::var_os("AX_METAL_MANIFEST_PATH").map(PathBuf::from);
    let mut output_dir = env::var_os("AX_METAL_OUTPUT_DIR").map(PathBuf::from);

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--manifest" => {
                let Some(value) = iter.next() else {
                    return Err(CliError::Usage(
                        "missing value for required flag --manifest".to_string(),
                    ));
                };
                manifest_path = Some(PathBuf::from(value));
            }
            "--output-dir" => {
                let Some(value) = iter.next() else {
                    return Err(CliError::Usage(
                        "missing value for required flag --output-dir".to_string(),
                    ));
                };
                output_dir = Some(PathBuf::from(value));
            }
            other => {
                return Err(CliError::Usage(format!(
                    "unknown flag for metal-build: {other}\n\n{}",
                    usage()
                )));
            }
        }
    }

    let manifest_path = absolutize_path(
        manifest_path.unwrap_or_else(|| current_dir.join("metal/phase1-kernels.json")),
        &current_dir,
    );
    let output_dir = absolutize_path(
        output_dir.unwrap_or_else(|| current_dir.join("build/metal")),
        &current_dir,
    );

    Ok(MetalBuildArgs {
        manifest_path,
        output_dir,
    })
}

fn required_flag(args: &[String], name: &str) -> Result<PathBuf, CliError> {
    Ok(PathBuf::from(required_string_flag(args, name)?))
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DoctorStatus {
    Ready,
    BringupOnly,
    NotReady,
}

impl DoctorStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::BringupOnly => "bringup_only",
            Self::NotReady => "not_ready",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct DoctorReport {
    schema_version: String,
    mlx_target: String,
    status: DoctorStatus,
    mlx_runtime_ready: bool,
    bringup_allowed: bool,
    host: HostReport,
    metal_toolchain: MetalToolchainReport,
    issues: Vec<String>,
    notes: Vec<String>,
    performance_advice: Vec<DoctorAdvice>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum DoctorAdviceSeverity {
    Info,
    Warning,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct DoctorAdvice {
    id: String,
    severity: DoctorAdviceSeverity,
    summary: String,
    detail: String,
}

impl DoctorAdvice {
    fn info(id: &str, summary: &str, detail: &str) -> Self {
        Self::new(id, DoctorAdviceSeverity::Info, summary, detail)
    }

    fn warning(id: &str, summary: &str, detail: &str) -> Self {
        Self::new(id, DoctorAdviceSeverity::Warning, summary, detail)
    }

    fn new(id: &str, severity: DoctorAdviceSeverity, summary: &str, detail: &str) -> Self {
        Self {
            id: id.to_string(),
            severity,
            summary: summary.to_string(),
            detail: detail.to_string(),
        }
    }
}

impl DoctorAdviceSeverity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
        }
    }
}

fn metal_build_doctor_report(report: &DoctorReport) -> MetalBuildDoctorReport {
    MetalBuildDoctorReport {
        status: report.status.as_str().to_string(),
        bringup_allowed: report.bringup_allowed,
        mlx_runtime_ready: report.mlx_runtime_ready,
        metal_toolchain_fully_available: report.metal_toolchain.fully_available,
        host: MetalBuildHostReport {
            os: report.host.os.clone(),
            arch: report.host.arch.clone(),
            detected_soc: report.host.detected_soc.clone(),
            supported_mlx_runtime: report.host.supported_mlx_runtime,
            unsupported_host_override_active: report.host.unsupported_host_override_active,
        },
        metal_toolchain: MetalBuildToolchainReport {
            fully_available: report.metal_toolchain.fully_available,
            metal: metal_build_tool_status(&report.metal_toolchain.metal),
            metallib: metal_build_tool_status(&report.metal_toolchain.metallib),
            metal_ar: metal_build_tool_status(&report.metal_toolchain.metal_ar),
        },
    }
}

fn metal_build_tool_status(tool: &ToolStatusReport) -> MetalBuildToolStatus {
    MetalBuildToolStatus {
        available: tool.available,
        version: tool.version.clone(),
    }
}

fn metal_build_status_label(status: MetalBuildStatus) -> &'static str {
    match status {
        MetalBuildStatus::Unknown => "unknown",
        MetalBuildStatus::Compiled => "compiled",
        MetalBuildStatus::SkippedToolchainUnavailable => "skipped_toolchain_unavailable",
        MetalBuildStatus::SkippedNotReady => "skipped_not_ready",
        MetalBuildStatus::FailedCompile => "failed_compile",
    }
}

fn absolutize_path(path: PathBuf, current_dir: &Path) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        current_dir.join(path)
    }
}

fn map_metal_build_error(error: ax_engine_core::MetalRuntimeError) -> CliError {
    match error {
        ax_engine_core::MetalRuntimeError::InvalidManifest { .. }
        | ax_engine_core::MetalRuntimeError::InvalidBuildReport { .. }
        | ax_engine_core::MetalRuntimeError::MissingBuildArtifact { .. } => {
            CliError::Contract(error.to_string())
        }
        _ => CliError::Runtime(error.to_string()),
    }
}

fn build_doctor_report(host: HostReport, metal_toolchain: MetalToolchainReport) -> DoctorReport {
    build_doctor_report_with_mlx_model_artifacts(host, metal_toolchain, None)
}

fn build_doctor_report_for_model(
    host: HostReport,
    metal_toolchain: MetalToolchainReport,
    mlx_model_artifacts_dir: Option<&Path>,
) -> DoctorReport {
    build_doctor_report_with_mlx_model_artifacts(host, metal_toolchain, mlx_model_artifacts_dir)
}

fn build_doctor_report_with_mlx_model_artifacts(
    host: HostReport,
    metal_toolchain: MetalToolchainReport,
    mlx_model_artifacts_dir: Option<&Path>,
) -> DoctorReport {
    let mlx_runtime_ready = host.supported_mlx_runtime && metal_toolchain.fully_available;
    let bringup_allowed = metal_toolchain.fully_available
        && (host.supported_mlx_runtime || host.unsupported_host_override_active);
    let status = if mlx_runtime_ready {
        DoctorStatus::Ready
    } else if bringup_allowed {
        DoctorStatus::BringupOnly
    } else {
        DoctorStatus::NotReady
    };

    DoctorReport {
        schema_version: "ax.engine_bench.doctor.v1".to_string(),
        mlx_target: "apple_m4_or_newer_macos_aarch64".to_string(),
        status,
        mlx_runtime_ready,
        bringup_allowed,
        host: host.clone(),
        metal_toolchain: metal_toolchain.clone(),
        issues: doctor_issues(&host, &metal_toolchain),
        notes: doctor_notes(&host),
        performance_advice: doctor_performance_advice(&host, mlx_model_artifacts_dir),
    }
}

fn doctor_issues(host: &HostReport, metal_toolchain: &MetalToolchainReport) -> Vec<String> {
    let mut issues = Vec::new();

    if !host.supported_mlx_runtime {
        let detected_host = if host.os != "macos" || host.arch != "aarch64" {
            format!("{}/{}", host.os, host.arch)
        } else {
            host.detected_soc
                .clone()
                .unwrap_or_else(|| "unknown Apple Silicon".to_string())
        };
        issues.push(format!(
            "AX Engine MLX Metal runtime requires macOS/aarch64 on Apple M4 or newer; detected {detected_host}"
        ));
    }

    if host.unsupported_host_override_active {
        issues.push(
            "AX_ALLOW_UNSUPPORTED_HOST is active; this machine is bring-up only and not a supported MLX runtime host"
                .to_string(),
        );
    }

    let missing_tools = missing_metal_tools(metal_toolchain);
    if !missing_tools.is_empty() {
        issues.push(format!(
            "Metal toolchain is incomplete; missing {}",
            missing_tools.join(", ")
        ));
    }

    issues
}

fn doctor_notes(host: &HostReport) -> Vec<String> {
    let mut notes = vec!["llama.cpp backends do not widen supported host scope".to_string()];
    if host.unsupported_host_override_active {
        notes.push(
            "AX_ALLOW_UNSUPPORTED_HOST only unlocks development or CI bring-up and does not make benchmark or runtime results supported"
                .to_string(),
        );
    }
    notes
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DoctorModelArtifactsHint {
    model_type: Option<String>,
    quantization: Option<DoctorQuantizationHint>,
    path_label: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DoctorQuantizationHint {
    mode: String,
    group_size: u32,
    bits: u32,
}

fn doctor_performance_advice(
    host: &HostReport,
    mlx_model_artifacts_dir: Option<&Path>,
) -> Vec<DoctorAdvice> {
    let mut advice = vec![
        DoctorAdvice::info(
            "ngram_acceleration_default_on",
            "N-gram acceleration is enabled by default for the repo-owned MLX runtime.",
            "Use --disable-ngram-acceleration only for direct A/B comparison rows; do not add a separate --ngram-accel enable flag.",
        ),
        DoctorAdvice::info(
            "mlx_throughput_harness",
            "Use the MLX inference-stack harness for throughput claims.",
            "Run scripts/bench_mlx_inference_stack.py with --ax-compare-policies so AX rows are paired with matching mlx_lm baseline rows.",
        ),
        DoctorAdvice::info(
            "single_request_benchmark_shape",
            "Treat batch=1 as the supported MLX performance shape today.",
            "The repo-owned MLX runner is optimized for single-request execution; multi-item batching remains a separate scheduler/runtime milestone.",
        ),
        DoctorAdvice::warning(
            "swiftlm_is_baseline_only",
            "Do not treat mlx-swift-lm as an AX prefill/decode hybrid path.",
            "mlx-swift-lm is admitted as a named benchmark baseline adapter, not as a supported runtime path that can prefill before AX decode.",
        ),
    ];

    if !host.supported_mlx_runtime {
        advice.push(DoctorAdvice::warning(
            "unsupported_host_benchmark_scope",
            "Do not publish MLX throughput claims from an unsupported host.",
            "Unsupported-host runs are useful for bring-up only; use a supported Apple Silicon host before comparing N-gram or quantization policy.",
        ));
    }

    let Some(model_dir) = mlx_model_artifacts_dir else {
        advice.push(DoctorAdvice::info(
            "model_artifacts_not_selected",
            "Pass --mlx-model-artifacts-dir for model-specific quantization advice.",
            "Without model artifacts doctor can only report runtime-level guidance, not whether this checkpoint should be compared against another quantization.",
        ));
        return advice;
    };

    match inspect_doctor_model_artifacts(model_dir) {
        Ok(model_hint) => advice.extend(doctor_model_performance_advice(&model_hint)),
        Err(message) => advice.push(DoctorAdvice::warning(
            "model_artifacts_unreadable",
            "Model-specific performance advice is unavailable.",
            &message,
        )),
    }

    advice
}

fn inspect_doctor_model_artifacts(path: &Path) -> Result<DoctorModelArtifactsHint, String> {
    if !path.exists() {
        return Err(format!(
            "model artifacts path does not exist: {}",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!(
            "model artifacts path is not a directory: {}",
            path.display()
        ));
    }

    let config_path = path.join("config.json");
    if !config_path.is_file() {
        return Err(format!(
            "model artifacts path is missing config.json: {}",
            path.display()
        ));
    }

    let manifest_path = path.join("model-manifest.json");
    if !manifest_path.is_file() {
        return Err(format!(
            "model artifacts path is missing model-manifest.json: {}; run `cargo run -p ax-engine-core --bin generate-manifest -- {}` before using this snapshot as AX MLX artifacts",
            path.display(),
            path.display()
        ));
    }

    let config = load_json_value(&config_path).map_err(|error| error.to_string())?;
    Ok(DoctorModelArtifactsHint {
        model_type: doctor_config_string(&config, "model_type").map(str::to_string),
        quantization: doctor_config_quantization(&config),
        path_label: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("")
            .to_ascii_lowercase(),
    })
}

fn doctor_model_performance_advice(hint: &DoctorModelArtifactsHint) -> Vec<DoctorAdvice> {
    let mut advice = Vec::new();
    let model_type = hint.model_type.as_deref().unwrap_or("unknown");
    let quantization = hint.quantization.as_ref();

    match model_type {
        "gemma4" => {
            if quantization.map(|q| q.bits) == Some(4) {
                advice.push(DoctorAdvice::info(
                    "gemma4_4bit_first",
                    "Gemma 4 4-bit is the first throughput candidate.",
                    "Current Gemma 4 decode rows show 4-bit as the fastest direct and N-gram policy for the checked-in E2B comparison; verify quality before moving up in bits.",
                ));
            } else {
                advice.push(DoctorAdvice::info(
                    "gemma4_quantization_compare",
                    "Benchmark this Gemma 4 quantization against 4-bit before calling it faster.",
                    "Higher-bit Gemma 4 checkpoints can improve quality, but current decode rows do not support a blanket speed claim over 4-bit.",
                ));
            }
        }
        "qwen3_next" | "qwen3_6" | "qwen3.6" => {
            if quantization.map(|q| q.bits) == Some(4) || hint.path_label.contains("ud-mlx") {
                advice.push(DoctorAdvice::warning(
                    "qwen36_quantization_compare",
                    "Do not assume Qwen 3.6 4-bit is the fastest checkpoint.",
                    "Current Qwen 3.6 35B rows show the MLX 5-bit checkpoint ahead of the UD-MLX 4-bit checkpoint on decode throughput; compare both on the target prompt mix.",
                ));
            } else if quantization.map(|q| q.bits) == Some(5) {
                advice.push(DoctorAdvice::info(
                    "qwen36_5bit_throughput_candidate",
                    "Qwen 3.6 5-bit is a strong throughput candidate.",
                    "Current Qwen 3.6 35B rows show 5-bit leading decode throughput, but memory pressure and quality targets still need workload-specific validation.",
                ));
            }
            advice.push(DoctorAdvice::info(
                "qwen_gated_delta_prefill_scope",
                "Keep Qwen 3.6 prefill/decode comparisons inside the MLX inference-stack harness.",
                "Qwen gated-delta prefill remains a known architecture-sensitive path; do not substitute a SwiftLM prefill plus AX decode claim without a new runtime contract.",
            ));
        }
        "qwen3_5" | "qwen3_5_moe" | "qwen3_5_text" => advice.push(DoctorAdvice::info(
            "qwen_gated_delta_prefill_scope",
            "Keep Qwen gated-delta prefill/decode comparisons inside the MLX inference-stack harness.",
            "Qwen gated-delta prefill remains architecture-sensitive; use paired baseline rows before changing runtime policy.",
        )),
        _ => advice.push(DoctorAdvice::info(
            "model_specific_policy_unknown",
            "No model-family-specific performance policy is available.",
            "Use the MLX inference-stack harness to establish direct and N-gram rows before making quantization or acceleration recommendations.",
        )),
    }

    if quantization.is_none() {
        advice.push(DoctorAdvice::info(
            "quantization_metadata_missing",
            "Quantization metadata was not found in config.json.",
            "Doctor cannot rank quantization choices without a quantization or quantization_config block.",
        ));
    }

    advice
}

fn doctor_config_string<'a>(config: &'a Value, field: &str) -> Option<&'a str> {
    config
        .get(field)
        .and_then(Value::as_str)
        .or_else(|| config.get("text_config")?.get(field)?.as_str())
}

fn doctor_config_quantization(config: &Value) -> Option<DoctorQuantizationHint> {
    let obj = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
        .or_else(|| config.get("text_config")?.get("quantization"))
        .or_else(|| config.get("text_config")?.get("quantization_config"))?;
    Some(DoctorQuantizationHint {
        mode: obj
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("affine")
            .to_string(),
        group_size: obj.get("group_size").and_then(Value::as_u64).unwrap_or(64) as u32,
        bits: obj.get("bits").and_then(Value::as_u64).unwrap_or(4) as u32,
    })
}

fn missing_metal_tools(metal_toolchain: &MetalToolchainReport) -> Vec<&'static str> {
    let mut missing = Vec::new();

    if !metal_toolchain.metal.available {
        missing.push("xcrun metal");
    }
    if !metal_toolchain.metallib.available {
        missing.push("xcrun metallib");
    }
    missing
}

fn tool_version_text(tool: &ToolStatusReport) -> &str {
    tool.version.as_deref().unwrap_or("unknown")
}

fn render_doctor_report(report: &DoctorReport) -> String {
    let mut lines = vec![
        "AX Engine v4 doctor".to_string(),
        format!("schema_version={}", report.schema_version),
        format!("target={}", report.mlx_target),
        format!("status={}", report.status.as_str()),
        format!("mlx_runtime_ready={}", report.mlx_runtime_ready),
        format!("bringup_allowed={}", report.bringup_allowed),
        format!("host.os={}", report.host.os),
        format!("host.arch={}", report.host.arch),
        format!(
            "host.detected_soc={}",
            report.host.detected_soc.as_deref().unwrap_or("unknown")
        ),
        format!(
            "host.supported_mlx_runtime={}",
            report.host.supported_mlx_runtime
        ),
        format!(
            "host.unsupported_host_override_active={}",
            report.host.unsupported_host_override_active
        ),
        format!(
            "metal_toolchain.fully_available={}",
            report.metal_toolchain.fully_available
        ),
    ];

    lines.extend([
        format!(
            "metal.available={} ({})",
            report.metal_toolchain.metal.available,
            tool_version_text(&report.metal_toolchain.metal)
        ),
        format!(
            "metallib.available={} ({})",
            report.metal_toolchain.metallib.available,
            tool_version_text(&report.metal_toolchain.metallib)
        ),
        format!(
            "metal_ar.available={} ({})",
            report.metal_toolchain.metal_ar.available,
            tool_version_text(&report.metal_toolchain.metal_ar)
        ),
        "issues:".to_string(),
    ]);

    if report.issues.is_empty() {
        lines.push("  - none".to_string());
    } else {
        lines.extend(report.issues.iter().map(|issue| format!("  - {issue}")));
    }

    lines.push("notes:".to_string());
    lines.extend(report.notes.iter().map(|note| format!("  - {note}")));
    lines.push("performance_advice:".to_string());
    lines.extend(report.performance_advice.iter().map(|advice| {
        format!(
            "  - [{}] {}: {} ({})",
            advice.severity.as_str(),
            advice.id,
            advice.summary,
            advice.detail
        )
    }));

    lines.join("\n")
}

fn required_string_flag(args: &[String], name: &str) -> Result<String, CliError> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            let Some(value) = iter.next() else {
                return Err(CliError::Usage(format!(
                    "missing value for required flag {name}"
                )));
            };
            return Ok(value.clone());
        }
    }

    Err(CliError::Usage(format!("missing required flag {name}")))
}

fn require_existing_file(path: &Path) -> Result<(), CliError> {
    if !path.is_file() {
        return Err(CliError::Contract(format!(
            "manifest file does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

fn require_existing_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::Contract(format!(
            "required path does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

fn require_existing_dir(path: &Path) -> Result<(), CliError> {
    if !path.is_dir() {
        return Err(CliError::Contract(format!(
            "required directory does not exist or is not a directory: {}",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_output_root(path: &Path) -> Result<(), CliError> {
    if path.exists() && !path.is_dir() {
        return Err(CliError::Contract(format!(
            "output root exists but is not a directory: {}",
            path.display()
        )));
    }

    if !path.exists() {
        fs::create_dir_all(path).map_err(|error| {
            CliError::Runtime(format!(
                "failed to create output root {}: {error}",
                path.display()
            ))
        })?;
    }

    Ok(())
}

fn load_manifest(path: &Path) -> Result<BenchmarkManifest, CliError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read manifest {}: {error}",
            path.display()
        ))
    })?;

    let mut manifest: BenchmarkManifest = serde_json::from_str(&raw).map_err(|error| {
        CliError::Contract(format!(
            "failed to parse manifest {} as JSON: {error}",
            path.display()
        ))
    })?;
    resolve_manifest_runtime_paths(&mut manifest, path)?;
    Ok(manifest)
}

fn load_matrix_manifest(path: &Path) -> Result<BenchmarkMatrixManifest, CliError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read matrix manifest {}: {error}",
            path.display()
        ))
    })?;

    serde_json::from_str(&raw).map_err(|error| {
        CliError::Contract(format!(
            "failed to parse matrix manifest {} as JSON: {error}",
            path.display()
        ))
    })
}

fn resolve_manifest_runtime_paths(
    manifest: &mut BenchmarkManifest,
    manifest_path: &Path,
) -> Result<(), CliError> {
    manifest.runtime.mlx_model_artifacts_dir = manifest
        .runtime
        .mlx_model_artifacts_dir
        .as_deref()
        .map(|path| resolve_manifest_path(path, manifest_path))
        .transpose()?;
    Ok(())
}

fn resolve_manifest_path(path: &Path, manifest_path: &Path) -> Result<PathBuf, CliError> {
    let expanded = expand_manifest_path_env(path)?;
    if expanded.is_absolute() {
        return Ok(normalize_path_lexically(expanded));
    }

    let manifest_dir = manifest_path.parent().ok_or_else(|| {
        CliError::Contract(format!(
            "manifest {} has no parent directory for relative path resolution",
            manifest_path.display()
        ))
    })?;
    Ok(normalize_path_lexically(manifest_dir.join(expanded)))
}

fn expand_manifest_path_env(path: &Path) -> Result<PathBuf, CliError> {
    let raw = path.to_string_lossy();
    if let Some(variable) = raw
        .strip_prefix("${")
        .and_then(|value| value.strip_suffix('}'))
    {
        return env::var_os(variable).map(PathBuf::from).ok_or_else(|| {
            CliError::Contract(format!(
                "manifest path references unset environment variable {variable}"
            ))
        });
    }

    if let Some(stripped) = raw.strip_prefix('$') {
        let variable_len = stripped
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .count();
        if variable_len > 0 {
            let (variable, remainder) = stripped.split_at(variable_len);
            let value = env::var_os(variable).ok_or_else(|| {
                CliError::Contract(format!(
                    "manifest path references unset environment variable {variable}"
                ))
            })?;
            let mut resolved = PathBuf::from(value);
            if !remainder.is_empty() {
                resolved.push(remainder.trim_start_matches(['/', '\\']));
            }
            return Ok(resolved);
        }
    }

    Ok(path.to_path_buf())
}

fn normalize_path_lexically(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    normalized.pop();
                } else if !normalized.has_root() {
                    normalized.push(component.as_os_str());
                }
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

fn validate_manifest(
    manifest: &BenchmarkManifest,
    expected_class: ManifestClass,
) -> Result<(), CliError> {
    if manifest.class != expected_class {
        return Err(CliError::Contract(format!(
            "manifest class mismatch: expected {}, got {}",
            expected_class.as_str(),
            manifest.class.as_str()
        )));
    }

    match expected_class {
        ManifestClass::Scenario => {
            let shape = manifest.shape.as_ref().ok_or_else(|| {
                CliError::Contract("scenario manifest must contain a shape object".to_string())
            })?;
            if shape.concurrency == 0 {
                return Err(CliError::Contract(
                    "scenario manifest shape.concurrency must be greater than zero".to_string(),
                ));
            }
        }
        ManifestClass::Replay => {
            if manifest.events.is_empty() {
                return Err(CliError::Contract(
                    "replay manifest events array must not be empty".to_string(),
                ));
            }
        }
    }

    Ok(())
}

fn validate_matrix_manifest(matrix_manifest: &BenchmarkMatrixManifest) -> Result<(), CliError> {
    if matrix_manifest.class != MatrixClass::ScenarioMatrix {
        return Err(CliError::Contract(format!(
            "unsupported matrix class: expected {}, got {}",
            MatrixClass::ScenarioMatrix.as_str(),
            matrix_manifest.class.as_str()
        )));
    }

    if matrix_manifest.members.is_empty() {
        return Err(CliError::Contract(
            "matrix manifest members array must not be empty".to_string(),
        ));
    }

    Ok(())
}

fn execute_manifest_runtime(manifest: &BenchmarkManifest) -> Result<RuntimeResult, CliError> {
    execute_manifest_runtime_with_test_options(manifest, false)
}

fn execute_manifest_runtime_with_test_options(
    manifest: &BenchmarkManifest,
    preserve_native_artifacts_for_test: bool,
) -> Result<RuntimeResult, CliError> {
    let runtime = normalize_runtime_for_test_execution(
        runtime_config_from_manifest(manifest)?,
        preserve_native_artifacts_for_test,
    );
    let first = execute_manifest_with_runtime(manifest, runtime.clone())?;
    let correctness = evaluate_correctness(manifest, &first)?;
    let determinism = if manifest_expect_deterministic(manifest)? {
        let second = execute_manifest_with_runtime(manifest, runtime.clone())?;
        if deterministic_runtime_digest(&first.digest)
            == deterministic_runtime_digest(&second.digest)
        {
            GateStatus::pass()
        } else {
            GateStatus::fail("deterministic rerun produced a different runtime digest")
        }
    } else {
        GateStatus::pass()
    };

    Ok(RuntimeResult {
        tool_mode: runtime.tool_mode(),
        runtime,
        observation: first,
        correctness,
        determinism,
    })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AutotuneCandidateConfig {
    max_batch_tokens: u32,
    kv_total_blocks: Option<u32>,
    prefix_cache: bool,
}

impl AutotuneCandidateConfig {
    fn label(&self) -> String {
        format!(
            "max_batch_tokens={}; kv_total_blocks={}; prefix_cache={}",
            self.max_batch_tokens,
            optional_u32_label(self.kv_total_blocks),
            self.prefix_cache
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
struct AutotuneSearchSpace {
    max_batch_token_options: Vec<u32>,
    kv_total_block_options: Vec<Option<u32>>,
    prefix_cache_options: Vec<bool>,
}

#[derive(Clone, Debug, Default)]
struct AutotuneWarmStartHistory {
    trials: Vec<AutotuneTrialRecord>,
    source_dirs: Vec<PathBuf>,
}

#[derive(Clone, Debug)]
struct AutotuneHistoryIndexEntry {
    result_dir: PathBuf,
    manifest_id: String,
    selected_backend: String,
    trials: Vec<Value>,
}

#[derive(Clone, Debug)]
struct AutotuneHistoryIndexSummary {
    manifest_id: String,
    selected_backend: String,
    source_dirs: Vec<PathBuf>,
    best_trials: Vec<Value>,
}

#[derive(Clone, Debug)]
struct AutotuneSelectionDiagnostics {
    strategy: String,
    predicted_mean: Option<f64>,
    uncertainty: Option<f64>,
    acquisition: Option<f64>,
    good_density: Option<f64>,
    bad_density: Option<f64>,
    density_ratio: Option<f64>,
    novelty_bonus: Option<f64>,
}

#[derive(Clone, Debug)]
struct AutotuneCandidateSelection {
    candidate_index: usize,
    diagnostics: AutotuneSelectionDiagnostics,
}

#[derive(Clone, Debug, Default)]
struct AutotuneTrialMetrics {
    ttft_ms: Option<u64>,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    prefix_hit_rate: f64,
    prefix_native_dispatch_count: u32,
    prefix_cpu_reference_dispatch_count: u32,
    prefix_cpu_projection_row_count: u32,
    prefix_cpu_rms_norm_element_count: u32,
    prefix_cpu_ffn_activation_element_count: u32,
    prefix_cpu_residual_add_element_count: u32,
    prefix_cpu_scale_element_count: u32,
    direct_decode_batched_group_fallback_count: u32,
    direct_decode_cpu_projection_row_count: u32,
    direct_decode_cpu_rms_norm_element_count: u32,
    direct_decode_cpu_ffn_activation_element_count: u32,
    direct_decode_cpu_residual_add_element_count: u32,
    direct_decode_cpu_scale_element_count: u32,
    mlx_metal_hot_path_cpu_fallback_free: bool,
    real_model_forward: bool,
    model_bound_ffn_decode: bool,
}

#[derive(Clone, Debug)]
struct AutotuneTrialRecord {
    trial_index: usize,
    candidate: AutotuneCandidateConfig,
    selection: AutotuneSelectionDiagnostics,
    probe: Option<AutotuneProbeObservation>,
    score: f64,
    status: String,
    result_dir: Option<PathBuf>,
    error: Option<String>,
    metrics: Option<AutotuneTrialMetrics>,
}

impl AutotuneTrialRecord {
    fn label(&self) -> String {
        format!("trial-{:03}", self.trial_index + 1)
    }
}

#[derive(Clone, Debug)]
struct AutotuneProbeObservation {
    shape: ScenarioShape,
    status: String,
    score: f64,
    skipped_full_trial: bool,
    skip_reason: Option<String>,
    result_dir: Option<PathBuf>,
    baseline: Option<AutotuneProbeBaseline>,
    metrics: AutotuneTrialMetrics,
}

#[derive(Clone, Debug)]
struct AutotuneProbeBaseline {
    observed_trial_count: usize,
    incumbent_score: Option<f64>,
    incumbent_decode_tok_s: Option<f64>,
    score_floor: Option<f64>,
    decode_tok_s_floor: Option<f64>,
    fallback_free_decode_tok_s_floor: Option<f64>,
    ttft_ceiling_ms: Option<u64>,
}

const AUTOTUNE_SCHEMA_VERSION: &str = "ax.engine_bench.autotune.v1";
const AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION: &str = "ax.engine_bench.autotune_history_index.v1";
const AUTOTUNE_FAILURE_SCORE: f64 = -1_000_000.0;

fn resolve_autotune_search_space(
    manifest: &BenchmarkManifest,
    args: &AutotuneArgs,
) -> AutotuneSearchSpace {
    let base_max_batch_tokens = manifest.runtime.max_batch_tokens.max(1);
    let max_batch_token_options = args.max_batch_token_options.clone().unwrap_or_else(|| {
        unique_sorted_u32(vec![
            (base_max_batch_tokens / 2).max(1),
            base_max_batch_tokens,
            base_max_batch_tokens.saturating_mul(2).max(1),
        ])
    });
    let kv_total_block_options = args.kv_total_block_options.clone().unwrap_or_else(|| {
        if let Some(base_kv_total_blocks) = manifest.runtime.kv_total_blocks {
            unique_sorted_option_u32(vec![
                Some((base_kv_total_blocks / 2).max(1)),
                Some(base_kv_total_blocks.max(1)),
                Some(base_kv_total_blocks.saturating_mul(2).max(1)),
            ])
        } else {
            unique_sorted_option_u32(vec![None, Some(32), Some(64)])
        }
    });
    let prefix_cache_options = args.prefix_cache_options.clone().unwrap_or_else(|| {
        if manifest.runtime.flags.prefix_cache {
            vec![true, false]
        } else {
            vec![false, true]
        }
    });

    AutotuneSearchSpace {
        max_batch_token_options,
        kv_total_block_options,
        prefix_cache_options,
    }
}

fn autotune_candidate_configs(
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> Vec<AutotuneCandidateConfig> {
    let base_candidate = AutotuneCandidateConfig {
        max_batch_tokens: manifest.runtime.max_batch_tokens.max(1),
        kv_total_blocks: manifest.runtime.kv_total_blocks,
        prefix_cache: manifest.runtime.flags.prefix_cache,
    };

    let mut candidates = Vec::new();
    for max_batch_tokens in &search_space.max_batch_token_options {
        for kv_total_blocks in &search_space.kv_total_block_options {
            for prefix_cache in &search_space.prefix_cache_options {
                candidates.push(AutotuneCandidateConfig {
                    max_batch_tokens: *max_batch_tokens,
                    kv_total_blocks: *kv_total_blocks,
                    prefix_cache: *prefix_cache,
                });
            }
        }
    }

    candidates.sort_by(|left, right| {
        (
            left != &base_candidate,
            left.max_batch_tokens,
            left.kv_total_blocks.unwrap_or(0),
            left.prefix_cache,
        )
            .cmp(&(
                right != &base_candidate,
                right.max_batch_tokens,
                right.kv_total_blocks.unwrap_or(0),
                right.prefix_cache,
            ))
    });
    candidates
}

fn select_next_autotune_candidate(
    candidates: &[AutotuneCandidateConfig],
    search_space: &AutotuneSearchSpace,
    trials: &[AutotuneTrialRecord],
    exploration_weight: f64,
) -> AutotuneCandidateSelection {
    if trials.is_empty() {
        return AutotuneCandidateSelection {
            candidate_index: 0,
            diagnostics: AutotuneSelectionDiagnostics {
                strategy: "base_config_seed".to_string(),
                predicted_mean: None,
                uncertainty: None,
                acquisition: None,
                good_density: None,
                bad_density: None,
                density_ratio: None,
                novelty_bonus: None,
            },
        };
    }
    if trials.len() == 1 {
        return select_next_autotune_candidate_by_coverage(candidates, trials);
    }

    select_next_autotune_candidate_by_tpe(candidates, search_space, trials, exploration_weight)
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn execute_autotune_trial(
    manifest_path: &Path,
    base_manifest: &BenchmarkManifest,
    trials_dir: &Path,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
    selection: AutotuneSelectionDiagnostics,
    observed_trials: &[AutotuneTrialRecord],
    started_at_unix_s: u64,
) -> Result<AutotuneTrialRecord, CliError> {
    let manifest = autotune_trial_manifest(base_manifest, trial_index, candidate);
    let probe = execute_autotune_probe_trial(
        manifest_path,
        base_manifest,
        trials_dir,
        trial_index,
        candidate,
        observed_trials,
        started_at_unix_s,
    )?;
    if probe.as_ref().is_some_and(|probe| probe.skipped_full_trial) {
        let probe = probe.expect("checked above");
        return Ok(AutotuneTrialRecord {
            trial_index,
            candidate: candidate.clone(),
            selection,
            probe: Some(probe.clone()),
            score: probe.score,
            status: "early_stopped".to_string(),
            result_dir: probe.result_dir.clone(),
            error: probe.skip_reason.clone(),
            metrics: Some(probe.metrics.clone()),
        });
    }
    match execute_manifest_runtime(&manifest) {
        Ok(execution) => {
            let result_dir = write_execution_artifacts(
                "autotune",
                manifest_path,
                &manifest,
                trials_dir,
                started_at_unix_s,
                &execution,
            )?;
            let metrics = autotune_trial_metrics(&execution);
            let score = autotune_score(&execution, &metrics);
            Ok(AutotuneTrialRecord {
                trial_index,
                candidate: candidate.clone(),
                selection,
                probe,
                score,
                status: execution.status_label().to_string(),
                result_dir: Some(result_dir),
                error: None,
                metrics: Some(metrics),
            })
        }
        Err(CliError::Contract(message)) => {
            let result_dir = write_contract_failure_artifacts(
                "autotune",
                manifest_path,
                &manifest,
                trials_dir,
                started_at_unix_s,
                &message,
            )?;
            Ok(AutotuneTrialRecord {
                trial_index,
                candidate: candidate.clone(),
                selection,
                probe,
                score: AUTOTUNE_FAILURE_SCORE,
                status: "contract_failure".to_string(),
                result_dir: Some(result_dir),
                error: Some(message),
                metrics: None,
            })
        }
        Err(error) => Ok(AutotuneTrialRecord {
            trial_index,
            candidate: candidate.clone(),
            selection,
            probe,
            score: AUTOTUNE_FAILURE_SCORE,
            status: "runtime_error".to_string(),
            result_dir: None,
            error: Some(error.to_string()),
            metrics: None,
        }),
    }
}

#[allow(dead_code)]
fn execute_autotune_probe_trial(
    manifest_path: &Path,
    base_manifest: &BenchmarkManifest,
    trials_dir: &Path,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
    observed_trials: &[AutotuneTrialRecord],
    started_at_unix_s: u64,
) -> Result<Option<AutotuneProbeObservation>, CliError> {
    let Some(probe_manifest) = autotune_probe_manifest(base_manifest, trial_index, candidate)
    else {
        return Ok(None);
    };
    let baseline = autotune_probe_baseline(observed_trials);
    let probe_shape = probe_manifest.shape.clone().ok_or_else(|| {
        CliError::Contract("autotune probe manifest missing scenario shape".to_string())
    })?;
    let probes_dir = trials_dir.join("probes");
    fs::create_dir_all(&probes_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create autotune probe directory {}: {error}",
            probes_dir.display()
        ))
    })?;

    match execute_manifest_runtime(&probe_manifest) {
        Ok(execution) => {
            let result_dir = write_execution_artifacts(
                "autotune-probe",
                manifest_path,
                &probe_manifest,
                &probes_dir,
                started_at_unix_s,
                &execution,
            )?;
            let metrics = autotune_trial_metrics(&execution);
            let score = autotune_score(&execution, &metrics);
            let skip_reason =
                autotune_probe_skip_reason(&execution, &metrics, baseline.as_ref(), score);
            Ok(Some(AutotuneProbeObservation {
                shape: probe_shape,
                status: execution.status_label().to_string(),
                score,
                skipped_full_trial: skip_reason.is_some(),
                skip_reason,
                result_dir: Some(result_dir),
                baseline,
                metrics,
            }))
        }
        Err(CliError::Contract(message)) => {
            let result_dir = write_contract_failure_artifacts(
                "autotune-probe",
                manifest_path,
                &probe_manifest,
                &probes_dir,
                started_at_unix_s,
                &message,
            )?;
            Ok(Some(AutotuneProbeObservation {
                shape: probe_shape,
                status: "contract_failure".to_string(),
                score: AUTOTUNE_FAILURE_SCORE,
                skipped_full_trial: true,
                skip_reason: Some(format!("probe contract failure: {message}")),
                result_dir: Some(result_dir),
                baseline,
                metrics: AutotuneTrialMetrics::default(),
            }))
        }
        Err(error) => Ok(Some(AutotuneProbeObservation {
            shape: probe_shape,
            status: "runtime_error".to_string(),
            score: AUTOTUNE_FAILURE_SCORE,
            skipped_full_trial: true,
            skip_reason: Some(format!("probe runtime error: {error}")),
            result_dir: None,
            baseline,
            metrics: AutotuneTrialMetrics::default(),
        })),
    }
}

fn autotune_probe_manifest(
    base_manifest: &BenchmarkManifest,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
) -> Option<BenchmarkManifest> {
    let mut manifest = autotune_trial_manifest(base_manifest, trial_index, candidate);
    let shape = manifest.shape.as_mut()?;
    shape.input_tokens_target = shape.input_tokens_target.clamp(1, 128);
    shape.output_tokens_target = shape.output_tokens_target.clamp(1, 8);
    shape.concurrency = shape.concurrency.clamp(1, 2);
    manifest.id = format!("{}-probe", manifest.id);
    manifest.checks.expect_deterministic = false;
    manifest.notes = Some(format!(
        "autotune probe {} using max_batch_tokens={}, kv_total_blocks={}, prefix_cache={}",
        trial_index + 1,
        candidate.max_batch_tokens,
        optional_u32_label(candidate.kv_total_blocks),
        candidate.prefix_cache
    ));
    Some(manifest)
}

fn autotune_probe_skip_reason(
    execution: &RuntimeResult,
    metrics: &AutotuneTrialMetrics,
    baseline: Option<&AutotuneProbeBaseline>,
    probe_score: f64,
) -> Option<String> {
    if !execution.correctness.passed {
        return Some(gate_failure_reason(
            &execution.correctness,
            "probe correctness gate failed",
        ));
    }
    if !execution.determinism.passed {
        return Some(gate_failure_reason(
            &execution.determinism,
            "probe determinism gate failed",
        ));
    }
    if !metrics.real_model_forward {
        return Some("probe did not reach real-model forward coverage".to_string());
    }
    if metrics.decode_tok_s <= 0.0 {
        return Some("probe decode throughput was zero".to_string());
    }
    let has_prefix_cpu_fallback = metrics.prefix_cpu_reference_dispatch_count > 0;
    let has_decode_cpu_fallback = metrics.direct_decode_batched_group_fallback_count > 0
        || metrics.direct_decode_cpu_projection_row_count > 0
        || metrics.direct_decode_cpu_scale_element_count > 0;
    let has_hot_path_cpu_fallback = has_prefix_cpu_fallback || has_decode_cpu_fallback;
    let has_incumbent_cpu_fallback = has_prefix_cpu_fallback
        || metrics.direct_decode_batched_group_fallback_count > 0
        || metrics.direct_decode_cpu_scale_element_count > 0;
    if has_prefix_cpu_fallback && has_decode_cpu_fallback {
        return Some("probe hit CPU fallback in both prefix and decode hot paths".to_string());
    }

    let baseline = baseline?;
    if !metrics.mlx_metal_hot_path_cpu_fallback_free
        && baseline
            .fallback_free_decode_tok_s_floor
            .is_some_and(|floor| floor > 0.0 && metrics.decode_tok_s < floor * 0.75)
    {
        return Some(format!(
            "probe decode tok/s {:.2} stayed below fallback-free floor {:.2} with CPU fallback",
            metrics.decode_tok_s,
            baseline
                .fallback_free_decode_tok_s_floor
                .unwrap_or_default()
        ));
    }
    if baseline
        .score_floor
        .is_some_and(|floor| floor > 0.0 && probe_score < floor * 0.7 && has_hot_path_cpu_fallback)
    {
        return Some(format!(
            "probe score {:.3} stayed below adaptive floor {:.3}",
            probe_score,
            baseline.score_floor.unwrap_or_default()
        ));
    }
    if baseline.ttft_ceiling_ms.is_some_and(|ceiling| {
        metrics.ttft_ms.is_some_and(|ttft_ms| {
            ttft_ms > ceiling.saturating_mul(2)
                && baseline
                    .decode_tok_s_floor
                    .is_some_and(|floor| floor > 0.0 && metrics.decode_tok_s < floor)
        })
    }) {
        return Some(format!(
            "probe TTFT {}ms exceeded adaptive ceiling {}ms while decode tok/s stayed low",
            metrics.ttft_ms.unwrap_or_default(),
            baseline.ttft_ceiling_ms.unwrap_or_default()
        ));
    }
    if !metrics.mlx_metal_hot_path_cpu_fallback_free
        && baseline
            .incumbent_decode_tok_s
            .is_some_and(|decode| decode > 0.0 && metrics.decode_tok_s < decode * 0.6)
    {
        return Some(format!(
            "probe decode tok/s {:.2} fell well below incumbent {:.2} with extra CPU fallback",
            metrics.decode_tok_s,
            baseline.incumbent_decode_tok_s.unwrap_or_default()
        ));
    }
    if baseline.incumbent_score.is_some_and(|score| {
        score > 0.0 && probe_score < score * 0.55 && has_incumbent_cpu_fallback
    }) {
        return Some(format!(
            "probe score {:.3} stayed far below incumbent {:.3}",
            probe_score,
            baseline.incumbent_score.unwrap_or_default()
        ));
    }

    None
}

fn autotune_probe_baseline(
    observed_trials: &[AutotuneTrialRecord],
) -> Option<AutotuneProbeBaseline> {
    let incumbent = best_autotune_trial(observed_trials.iter())?;
    let score_floor = percentile_f64(
        &observed_trials
            .iter()
            .map(|trial| trial.score)
            .collect::<Vec<_>>(),
        0.35,
    );
    let decode_tok_s_floor = percentile_f64(
        &observed_trials
            .iter()
            .filter_map(|trial| trial.metrics.as_ref().map(|metrics| metrics.decode_tok_s))
            .collect::<Vec<_>>(),
        0.35,
    );
    let fallback_free_decode_tok_s_floor = percentile_f64(
        &observed_trials
            .iter()
            .filter_map(|trial| {
                trial.metrics.as_ref().and_then(|metrics| {
                    metrics
                        .mlx_metal_hot_path_cpu_fallback_free
                        .then_some(metrics.decode_tok_s)
                })
            })
            .collect::<Vec<_>>(),
        0.5,
    );
    let ttft_ceiling_ms = percentile_u64(
        &observed_trials
            .iter()
            .filter_map(|trial| trial.metrics.as_ref().and_then(|metrics| metrics.ttft_ms))
            .collect::<Vec<_>>(),
        0.75,
    );

    Some(AutotuneProbeBaseline {
        observed_trial_count: observed_trials.len(),
        incumbent_score: Some(incumbent.score),
        incumbent_decode_tok_s: incumbent
            .metrics
            .as_ref()
            .map(|metrics| metrics.decode_tok_s),
        score_floor,
        decode_tok_s_floor,
        fallback_free_decode_tok_s_floor,
        ttft_ceiling_ms,
    })
}

fn percentile_f64(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = quantile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values.get(index).copied()
}

fn percentile_u64(values: &[u64], quantile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    let quantile = quantile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values.get(index).copied()
}

fn select_next_autotune_candidate_by_coverage(
    candidates: &[AutotuneCandidateConfig],
    trials: &[AutotuneTrialRecord],
) -> AutotuneCandidateSelection {
    let tried = collect_tried_autotune_candidates(trials);
    let mut best_index = None;
    let mut best_novelty = f64::NEG_INFINITY;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if tried.contains(candidate) {
            continue;
        }
        let novelty = autotune_candidate_novelty(candidate, trials);
        if novelty > best_novelty {
            best_novelty = novelty;
            best_index = Some(candidate_index);
        }
    }

    AutotuneCandidateSelection {
        candidate_index: best_index.unwrap_or(0),
        diagnostics: AutotuneSelectionDiagnostics {
            strategy: "coverage_explore".to_string(),
            predicted_mean: None,
            uncertainty: Some(best_novelty.max(0.0)),
            acquisition: Some(best_novelty.max(0.0)),
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: Some(best_novelty.max(0.0)),
        },
    }
}

fn select_next_autotune_candidate_by_tpe(
    candidates: &[AutotuneCandidateConfig],
    search_space: &AutotuneSearchSpace,
    trials: &[AutotuneTrialRecord],
    exploration_weight: f64,
) -> AutotuneCandidateSelection {
    let tried = collect_tried_autotune_candidates(trials);
    let (good_trials, bad_trials) = split_autotune_trials_by_score(trials);

    let mut best_index = None;
    let mut best_diagnostics = None;
    let mut best_acquisition = f64::NEG_INFINITY;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if tried.contains(candidate) {
            continue;
        }

        let good_density =
            autotune_candidate_density(candidate, search_space, &good_trials).max(f64::EPSILON);
        let bad_density =
            autotune_candidate_density(candidate, search_space, &bad_trials).max(f64::EPSILON);
        let density_ratio = good_density / bad_density.max(f64::EPSILON);
        let novelty = autotune_candidate_novelty(candidate, trials);
        let posterior_good = good_density / (good_density + bad_density).max(f64::EPSILON);
        let acquisition = density_ratio.ln() + exploration_weight * novelty;

        if acquisition > best_acquisition {
            best_acquisition = acquisition;
            best_index = Some(candidate_index);
            best_diagnostics = Some(AutotuneSelectionDiagnostics {
                strategy: "tpe_ratio".to_string(),
                predicted_mean: Some(posterior_good),
                uncertainty: Some(novelty),
                acquisition: Some(acquisition),
                good_density: Some(good_density),
                bad_density: Some(bad_density),
                density_ratio: Some(density_ratio),
                novelty_bonus: Some(novelty),
            });
        }
    }

    AutotuneCandidateSelection {
        candidate_index: best_index.unwrap_or(0),
        diagnostics: best_diagnostics.unwrap_or(AutotuneSelectionDiagnostics {
            strategy: "tpe_ratio".to_string(),
            predicted_mean: None,
            uncertainty: Some(0.0),
            acquisition: Some(f64::NEG_INFINITY),
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: Some(0.0),
        }),
    }
}

fn collect_tried_autotune_candidates(
    trials: &[AutotuneTrialRecord],
) -> BTreeSet<AutotuneCandidateConfig> {
    trials
        .iter()
        .map(|trial| trial.candidate.clone())
        .collect::<BTreeSet<_>>()
}

fn split_autotune_trials_by_score(
    trials: &[AutotuneTrialRecord],
) -> (Vec<&AutotuneTrialRecord>, Vec<&AutotuneTrialRecord>) {
    let mut ordered = trials.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let good_count = ((ordered.len() as f64) * 0.35).ceil() as usize;
    let good_count = good_count
        .max(1)
        .min(ordered.len().saturating_sub(1).max(1));
    let good_trials = ordered.iter().take(good_count).copied().collect::<Vec<_>>();
    let bad_trials = ordered.iter().skip(good_count).copied().collect::<Vec<_>>();
    let bad_trials = if bad_trials.is_empty() {
        good_trials.clone()
    } else {
        bad_trials
    };
    (good_trials, bad_trials)
}

fn autotune_candidate_density(
    candidate: &AutotuneCandidateConfig,
    search_space: &AutotuneSearchSpace,
    trials: &[&AutotuneTrialRecord],
) -> f64 {
    smoothed_probability(
        candidate.max_batch_tokens,
        search_space.max_batch_token_options.len(),
        trials,
        |config| config.max_batch_tokens,
    ) * smoothed_probability(
        candidate.kv_total_blocks,
        search_space.kv_total_block_options.len(),
        trials,
        |config| config.kv_total_blocks,
    ) * smoothed_probability(
        candidate.prefix_cache,
        search_space.prefix_cache_options.len(),
        trials,
        |config| config.prefix_cache,
    )
}

fn autotune_candidate_novelty(
    candidate: &AutotuneCandidateConfig,
    trials: &[AutotuneTrialRecord],
) -> f64 {
    let max_batch_rarity = rarity_score(trials, candidate, |config| config.max_batch_tokens);
    let kv_total_blocks_rarity = rarity_score(trials, candidate, |config| config.kv_total_blocks);
    let prefix_cache_rarity = rarity_score(trials, candidate, |config| config.prefix_cache);
    (max_batch_rarity + kv_total_blocks_rarity + prefix_cache_rarity) / 3.0
}

fn rarity_score<T>(
    trials: &[AutotuneTrialRecord],
    candidate: &AutotuneCandidateConfig,
    accessor: impl Fn(&AutotuneCandidateConfig) -> T,
) -> f64
where
    T: PartialEq,
{
    1.0 / (1.0 + matching_trial_count(trials, candidate, accessor) as f64)
}

fn matching_trial_count<T>(
    trials: &[AutotuneTrialRecord],
    candidate: &AutotuneCandidateConfig,
    accessor: impl Fn(&AutotuneCandidateConfig) -> T,
) -> usize
where
    T: PartialEq,
{
    let expected = accessor(candidate);
    trials
        .iter()
        .filter(|trial| accessor(&trial.candidate) == expected)
        .count()
}

fn smoothed_probability<T>(
    value: T,
    option_count: usize,
    trials: &[&AutotuneTrialRecord],
    accessor: impl Fn(&AutotuneCandidateConfig) -> T,
) -> f64
where
    T: Copy + PartialEq,
{
    let matching = trials
        .iter()
        .filter(|trial| accessor(&trial.candidate) == value)
        .count() as f64;
    (matching + 1.0) / (trials.len() as f64 + option_count as f64)
}

fn load_autotune_warm_start_history(
    output_root: &Path,
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> Result<AutotuneWarmStartHistory, CliError> {
    if !output_root.is_dir() {
        return Ok(AutotuneWarmStartHistory::default());
    }

    if let Some(history) =
        load_autotune_warm_start_history_from_index(output_root, manifest, search_space)?
    {
        return Ok(history);
    }

    let entries = collect_autotune_history_index_entries_from_output_root(output_root)?;
    Ok(autotune_warm_start_history_from_index_entries(
        &entries,
        manifest,
        search_space,
    ))
}

fn autotune_history_index_path(output_root: &Path) -> PathBuf {
    output_root.join("autotune-history-index.json")
}

fn load_autotune_warm_start_history_from_index(
    output_root: &Path,
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> Result<Option<AutotuneWarmStartHistory>, CliError> {
    let Some(index_json) = load_autotune_history_index_json(output_root)? else {
        return Ok(None);
    };

    if let Some(summary) = load_matching_autotune_history_index_summary(&index_json, manifest)? {
        return Ok(Some(autotune_warm_start_history_from_index_summary(
            &summary,
            search_space,
        )));
    }

    let Some(entries) = load_autotune_history_index_entries_from_json(&index_json)? else {
        return Ok(None);
    };
    Ok(Some(autotune_warm_start_history_from_index_entries(
        &entries,
        manifest,
        search_space,
    )))
}

fn load_autotune_history_index_json(output_root: &Path) -> Result<Option<Value>, CliError> {
    let index_path = autotune_history_index_path(output_root);
    let Some(index_json) = load_optional_json_value(&index_path)? else {
        return Ok(None);
    };
    if index_json.get("schema_version").and_then(Value::as_str)
        != Some(AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION)
    {
        return Ok(None);
    }
    Ok(Some(index_json))
}

fn load_matching_autotune_history_index_summary(
    index_json: &Value,
    manifest: &BenchmarkManifest,
) -> Result<Option<AutotuneHistoryIndexSummary>, CliError> {
    let Some(summaries_json) = index_json.get("summaries").and_then(Value::as_array) else {
        return Ok(None);
    };
    let summaries = summaries_json
        .iter()
        .filter_map(autotune_history_index_summary_from_json)
        .collect::<Vec<_>>();
    if summaries.is_empty() && !summaries_json.is_empty() {
        return Ok(None);
    }
    Ok(summaries.into_iter().find(|summary| {
        summary.manifest_id == manifest.id
            && summary.selected_backend == selected_backend_label(manifest.runtime.selected_backend)
    }))
}

fn load_autotune_history_index_entries_from_json(
    index_json: &Value,
) -> Result<Option<Vec<AutotuneHistoryIndexEntry>>, CliError> {
    let Some(entries_json) = index_json.get("entries").and_then(Value::as_array) else {
        return Ok(None);
    };
    let entries = entries_json
        .iter()
        .filter_map(autotune_history_index_entry_from_json)
        .collect::<Vec<_>>();
    if entries.is_empty() && !entries_json.is_empty() {
        return Ok(None);
    }
    Ok(Some(entries))
}

fn collect_autotune_history_index_entries_from_output_root(
    output_root: &Path,
) -> Result<Vec<AutotuneHistoryIndexEntry>, CliError> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(output_root).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read autotune output root {}: {error}",
            output_root.display()
        ))
    })? {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let autotune_json_path = path.join("autotune.json");
        let history_json = match load_optional_json_value(&autotune_json_path) {
            Ok(Some(json)) => json,
            Ok(None) | Err(_) => continue,
        };
        let Some(entry) = autotune_history_index_entry_from_result_json(&path, &history_json)
        else {
            continue;
        };
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.result_dir.cmp(&right.result_dir));
    entries.dedup_by(|left, right| left.result_dir == right.result_dir);
    Ok(entries)
}

fn autotune_history_index_entry_from_result_json(
    result_dir: &Path,
    history_json: &Value,
) -> Option<AutotuneHistoryIndexEntry> {
    if history_json.get("schema_version").and_then(Value::as_str) != Some(AUTOTUNE_SCHEMA_VERSION) {
        return None;
    }
    Some(AutotuneHistoryIndexEntry {
        result_dir: result_dir.to_path_buf(),
        manifest_id: history_json.get("manifest_id")?.as_str()?.to_string(),
        selected_backend: history_json.get("selected_backend")?.as_str()?.to_string(),
        trials: history_json.get("trials")?.as_array()?.clone(),
    })
}

fn autotune_history_index_entry_from_json(entry_json: &Value) -> Option<AutotuneHistoryIndexEntry> {
    Some(AutotuneHistoryIndexEntry {
        result_dir: PathBuf::from(entry_json.get("result_dir")?.as_str()?),
        manifest_id: entry_json.get("manifest_id")?.as_str()?.to_string(),
        selected_backend: entry_json.get("selected_backend")?.as_str()?.to_string(),
        trials: entry_json.get("trials")?.as_array()?.clone(),
    })
}

fn autotune_history_index_summary_from_json(
    summary_json: &Value,
) -> Option<AutotuneHistoryIndexSummary> {
    let source_dirs_json = summary_json.get("source_dirs")?.as_array()?;
    let best_trials_json = summary_json.get("best_trials")?.as_array()?;
    let source_dirs = source_dirs_json
        .iter()
        .filter_map(Value::as_str)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if source_dirs.is_empty() && !source_dirs_json.is_empty() {
        return None;
    }
    Some(AutotuneHistoryIndexSummary {
        manifest_id: summary_json.get("manifest_id")?.as_str()?.to_string(),
        selected_backend: summary_json.get("selected_backend")?.as_str()?.to_string(),
        source_dirs,
        best_trials: best_trials_json.clone(),
    })
}

fn autotune_warm_start_history_from_index_entries(
    entries: &[AutotuneHistoryIndexEntry],
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
) -> AutotuneWarmStartHistory {
    let mut source_dirs = Vec::new();
    let mut best_trials_by_candidate =
        BTreeMap::<AutotuneCandidateConfig, AutotuneTrialRecord>::new();
    for entry in entries {
        if entry.manifest_id != manifest.id {
            continue;
        }
        if entry.selected_backend != selected_backend_label(manifest.runtime.selected_backend) {
            continue;
        }
        let mut loaded_any = false;
        for (trial_index, trial_json) in entry.trials.iter().enumerate() {
            let Some(trial) =
                autotune_trial_record_from_history_json(trial_index, trial_json, search_space)
            else {
                continue;
            };
            loaded_any = true;
            match best_trials_by_candidate.get(&trial.candidate) {
                Some(existing) if existing.score >= trial.score => {}
                _ => {
                    best_trials_by_candidate.insert(trial.candidate.clone(), trial);
                }
            }
        }
        if loaded_any {
            source_dirs.push(entry.result_dir.clone());
        }
    }

    source_dirs.sort();
    source_dirs.dedup();
    let trials = best_trials_by_candidate.into_values().collect::<Vec<_>>();
    AutotuneWarmStartHistory {
        trials,
        source_dirs,
    }
}

fn autotune_warm_start_history_from_index_summary(
    summary: &AutotuneHistoryIndexSummary,
    search_space: &AutotuneSearchSpace,
) -> AutotuneWarmStartHistory {
    let trials = summary
        .best_trials
        .iter()
        .enumerate()
        .filter_map(|(trial_index, trial_json)| {
            autotune_trial_record_from_history_json(trial_index, trial_json, search_space)
        })
        .collect::<Vec<_>>();
    AutotuneWarmStartHistory {
        trials,
        source_dirs: summary.source_dirs.clone(),
    }
}

fn write_autotune_history_index_incremental(
    output_root: &Path,
    result_dir: &Path,
    result_json: &Value,
) -> Result<(), CliError> {
    let Some(current_entry) =
        autotune_history_index_entry_from_result_json(result_dir, result_json)
    else {
        return write_autotune_history_index(output_root);
    };
    let mut entries = match load_autotune_history_index_json(output_root)? {
        Some(index_json) => match load_autotune_history_index_entries_from_json(&index_json)? {
            Some(entries) => entries,
            None => collect_autotune_history_index_entries_from_output_root(output_root)?,
        },
        None => collect_autotune_history_index_entries_from_output_root(output_root)?,
    };
    upsert_autotune_history_index_entry(&mut entries, current_entry);
    write_autotune_history_index_entries(output_root, &entries)
}

fn write_autotune_history_index(output_root: &Path) -> Result<(), CliError> {
    let entries = collect_autotune_history_index_entries_from_output_root(output_root)?;
    write_autotune_history_index_entries(output_root, &entries)
}

fn write_autotune_history_index_entries(
    output_root: &Path,
    entries: &[AutotuneHistoryIndexEntry],
) -> Result<(), CliError> {
    let summaries = summarize_autotune_history_index_entries(entries);
    let index_json = json!({
        "schema_version": AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION,
        "entry_count": entries.len(),
        "summary_count": summaries.len(),
        "entries": entries.iter().map(|entry| json!({
            "result_dir": entry.result_dir.display().to_string(),
            "manifest_id": entry.manifest_id.clone(),
            "selected_backend": entry.selected_backend.clone(),
            "trials": entry.trials.clone(),
        })).collect::<Vec<_>>(),
        "summaries": summaries.iter().map(|summary| json!({
            "manifest_id": summary.manifest_id.clone(),
            "selected_backend": summary.selected_backend.clone(),
            "source_dirs": summary
                .source_dirs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>(),
            "best_trials": summary.best_trials.clone(),
        })).collect::<Vec<_>>(),
    });
    write_json_file(&autotune_history_index_path(output_root), &index_json)
}

fn upsert_autotune_history_index_entry(
    entries: &mut Vec<AutotuneHistoryIndexEntry>,
    entry: AutotuneHistoryIndexEntry,
) {
    if let Some(existing) = entries
        .iter_mut()
        .find(|existing| existing.result_dir == entry.result_dir)
    {
        *existing = entry;
    } else {
        entries.push(entry);
    }
    entries.sort_by(|left, right| left.result_dir.cmp(&right.result_dir));
    entries.dedup_by(|left, right| left.result_dir == right.result_dir);
}

fn summarize_autotune_history_index_entries(
    entries: &[AutotuneHistoryIndexEntry],
) -> Vec<AutotuneHistoryIndexSummary> {
    let mut grouped_best_trials = BTreeMap::<
        (String, String),
        (
            BTreeSet<PathBuf>,
            BTreeMap<AutotuneCandidateConfig, (f64, Value)>,
        ),
    >::new();
    for entry in entries {
        let grouped = grouped_best_trials
            .entry((entry.manifest_id.clone(), entry.selected_backend.clone()))
            .or_insert_with(|| (BTreeSet::new(), BTreeMap::new()));
        grouped.0.insert(entry.result_dir.clone());
        for trial_json in &entry.trials {
            let Some(candidate) = autotune_candidate_config_from_history_json(trial_json) else {
                continue;
            };
            let Some(score) = trial_json.get("score").and_then(Value::as_f64) else {
                continue;
            };
            match grouped.1.get(&candidate) {
                Some((existing_score, _)) if *existing_score >= score => {}
                _ => {
                    grouped.1.insert(candidate, (score, trial_json.clone()));
                }
            }
        }
    }

    grouped_best_trials
        .into_iter()
        .map(
            |((manifest_id, selected_backend), (source_dirs, best_trials))| {
                AutotuneHistoryIndexSummary {
                    manifest_id,
                    selected_backend,
                    source_dirs: source_dirs.into_iter().collect(),
                    best_trials: best_trials
                        .into_values()
                        .map(|(_, trial_json)| trial_json)
                        .collect(),
                }
            },
        )
        .collect()
}

fn autotune_trial_record_from_history_json(
    trial_index: usize,
    trial_json: &Value,
    search_space: &AutotuneSearchSpace,
) -> Option<AutotuneTrialRecord> {
    let candidate = autotune_candidate_config_from_history_json(trial_json)?;
    if !search_space
        .max_batch_token_options
        .contains(&candidate.max_batch_tokens)
        || !search_space
            .kv_total_block_options
            .contains(&candidate.kv_total_blocks)
        || !search_space
            .prefix_cache_options
            .contains(&candidate.prefix_cache)
    {
        return None;
    }

    Some(AutotuneTrialRecord {
        trial_index,
        candidate,
        selection: autotune_selection_from_history_json(trial_json.get("selection")),
        probe: None,
        score: trial_json.get("score")?.as_f64()?,
        status: trial_json
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("historical")
            .to_string(),
        result_dir: trial_json
            .get("result_dir")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        error: trial_json
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_string),
        metrics: autotune_metrics_from_history_json(trial_json.get("metrics")),
    })
}

fn autotune_candidate_config_from_history_json(
    trial_json: &Value,
) -> Option<AutotuneCandidateConfig> {
    let config = trial_json.get("config")?;
    let max_batch_tokens = config.get("max_batch_tokens")?.as_u64()? as u32;
    let kv_total_blocks = match config.get("kv_total_blocks") {
        Some(Value::Null) | None => None,
        Some(value) => Some(value.as_u64()? as u32),
    };
    let prefix_cache = config.get("prefix_cache")?.as_bool()?;
    Some(AutotuneCandidateConfig {
        max_batch_tokens,
        kv_total_blocks,
        prefix_cache,
    })
}

fn autotune_selection_from_history_json(
    selection_json: Option<&Value>,
) -> AutotuneSelectionDiagnostics {
    let Some(selection_json) = selection_json else {
        return AutotuneSelectionDiagnostics {
            strategy: "historical_import".to_string(),
            predicted_mean: None,
            uncertainty: None,
            acquisition: None,
            good_density: None,
            bad_density: None,
            density_ratio: None,
            novelty_bonus: None,
        };
    };
    AutotuneSelectionDiagnostics {
        strategy: selection_json
            .get("strategy")
            .and_then(Value::as_str)
            .unwrap_or("historical_import")
            .to_string(),
        predicted_mean: selection_json.get("predicted_mean").and_then(Value::as_f64),
        uncertainty: selection_json.get("uncertainty").and_then(Value::as_f64),
        acquisition: selection_json.get("acquisition").and_then(Value::as_f64),
        good_density: selection_json.get("good_density").and_then(Value::as_f64),
        bad_density: selection_json.get("bad_density").and_then(Value::as_f64),
        density_ratio: selection_json.get("density_ratio").and_then(Value::as_f64),
        novelty_bonus: selection_json.get("novelty_bonus").and_then(Value::as_f64),
    }
}

fn autotune_metrics_from_history_json(
    metrics_json: Option<&Value>,
) -> Option<AutotuneTrialMetrics> {
    let metrics_json = metrics_json?;
    Some(AutotuneTrialMetrics {
        ttft_ms: metrics_json.get("ttft_ms").and_then(Value::as_u64),
        prefill_tok_s: metrics_json.get("prefill_tok_s").and_then(Value::as_f64)?,
        decode_tok_s: metrics_json.get("decode_tok_s").and_then(Value::as_f64)?,
        prefix_hit_rate: metrics_json
            .get("prefix_hit_rate")
            .and_then(Value::as_f64)?,
        prefix_native_dispatch_count: metrics_json
            .get("prefix_native_dispatch_count")
            .and_then(Value::as_u64)? as u32,
        prefix_cpu_reference_dispatch_count: metrics_json
            .get("prefix_cpu_reference_dispatch_count")
            .and_then(Value::as_u64)? as u32,
        prefix_cpu_projection_row_count: metrics_json
            .get("prefix_cpu_projection_row_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_rms_norm_element_count: metrics_json
            .get("prefix_cpu_rms_norm_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_ffn_activation_element_count: metrics_json
            .get("prefix_cpu_ffn_activation_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_residual_add_element_count: metrics_json
            .get("prefix_cpu_residual_add_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        prefix_cpu_scale_element_count: metrics_json
            .get("prefix_cpu_scale_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_batched_group_fallback_count: metrics_json
            .get("direct_decode_batched_group_fallback_count")
            .and_then(Value::as_u64)? as u32,
        direct_decode_cpu_projection_row_count: metrics_json
            .get("direct_decode_cpu_projection_row_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_rms_norm_element_count: metrics_json
            .get("direct_decode_cpu_rms_norm_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_ffn_activation_element_count: metrics_json
            .get("direct_decode_cpu_ffn_activation_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_residual_add_element_count: metrics_json
            .get("direct_decode_cpu_residual_add_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        direct_decode_cpu_scale_element_count: metrics_json
            .get("direct_decode_cpu_scale_element_count")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        mlx_metal_hot_path_cpu_fallback_free: metrics_json
            .get("mlx_metal_hot_path_cpu_fallback_free")
            .and_then(Value::as_bool)?,
        real_model_forward: metrics_json
            .get("real_model_forward")
            .and_then(Value::as_bool)?,
        model_bound_ffn_decode: metrics_json
            .get("model_bound_ffn_decode")
            .and_then(Value::as_bool)?,
    })
}

fn autotune_trial_manifest(
    base_manifest: &BenchmarkManifest,
    trial_index: usize,
    candidate: &AutotuneCandidateConfig,
) -> BenchmarkManifest {
    let mut manifest = base_manifest.clone();
    manifest.id = format!("{}-autotune-{:03}", base_manifest.id, trial_index + 1);
    manifest.runtime.max_batch_tokens = candidate.max_batch_tokens;
    manifest.runtime.kv_total_blocks = candidate.kv_total_blocks;
    manifest.runtime.flags.prefix_cache = candidate.prefix_cache;
    manifest.notes = Some(format!(
        "autotune trial {} using {}",
        trial_index + 1,
        candidate.label()
    ));
    manifest
}

#[allow(dead_code)]
fn autotune_trial_metrics(execution: &RuntimeResult) -> AutotuneTrialMetrics {
    let route = &execution.observation.route_metadata;
    AutotuneTrialMetrics {
        ttft_ms: execution.observation.ttft_ms,
        prefill_tok_s: execution.observation.prefill_tok_s(),
        decode_tok_s: execution.observation.decode_tok_s(),
        prefix_hit_rate: execution.observation.prefix_hit_rate(),
        prefix_native_dispatch_count: route_prefix_native_dispatch_count(route),
        prefix_cpu_reference_dispatch_count: route_prefix_cpu_reference_dispatch_count(route),
        prefix_cpu_projection_row_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_projection_row_count",
        ),
        prefix_cpu_rms_norm_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_rms_norm_element_count",
        ),
        prefix_cpu_ffn_activation_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_ffn_activation_element_count",
        ),
        prefix_cpu_residual_add_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_residual_add_element_count",
        ),
        prefix_cpu_scale_element_count: route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_scale_element_count",
        ),
        direct_decode_batched_group_fallback_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_group_fallback_count",
        ),
        direct_decode_cpu_projection_row_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_projection_row_count",
        ),
        direct_decode_cpu_rms_norm_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
        ),
        direct_decode_cpu_ffn_activation_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
        ),
        direct_decode_cpu_residual_add_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_residual_add_element_count",
        ),
        direct_decode_cpu_scale_element_count: route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_scale_element_count",
        ),
        mlx_metal_hot_path_cpu_fallback_free: route_prefix_cpu_reference_dispatch_count(route) == 0
            && route_decision_value(route, "metal_dispatch_prefix_cpu_projection_row_count") == 0
            && route_decision_value(route, "metal_dispatch_prefix_cpu_rms_norm_element_count") == 0
            && route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_ffn_activation_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_residual_add_element_count",
            ) == 0
            && route_decision_value(route, "metal_dispatch_prefix_cpu_scale_element_count") == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_projection_row_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_residual_add_element_count",
            ) == 0
            && route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_scale_element_count",
            ) == 0,
        real_model_forward: route_decision_flag(route, "metal_dispatch_real_model_forward"),
        model_bound_ffn_decode: route_decision_flag(
            route,
            "metal_dispatch_direct_decode_model_bound_ffn",
        ),
    }
}

#[allow(dead_code)]
fn autotune_score(execution: &RuntimeResult, metrics: &AutotuneTrialMetrics) -> f64 {
    let throughput_reward = metrics.decode_tok_s + (metrics.prefill_tok_s * 0.25);
    let prefix_reward = metrics.prefix_hit_rate * 2.0;
    let native_dispatch_reward = f64::from(metrics.prefix_native_dispatch_count) * 50.0;
    let ttft_penalty = metrics.ttft_ms.unwrap_or_default() as f64 * 0.25;
    let cpu_fallback_penalty = f64::from(metrics.prefix_cpu_reference_dispatch_count) * 1500.0
        + f64::from(metrics.direct_decode_batched_group_fallback_count) * 750.0
        + f64::from(metrics.prefix_cpu_projection_row_count) * 0.05
        + f64::from(metrics.prefix_cpu_rms_norm_element_count) * 0.001
        + f64::from(metrics.prefix_cpu_ffn_activation_element_count) * 0.001
        + f64::from(metrics.prefix_cpu_residual_add_element_count) * 0.001
        + f64::from(metrics.prefix_cpu_scale_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_projection_row_count) * 0.05
        + f64::from(metrics.direct_decode_cpu_rms_norm_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_ffn_activation_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_residual_add_element_count) * 0.001
        + f64::from(metrics.direct_decode_cpu_scale_element_count) * 0.001;
    let readiness_reward = if metrics.mlx_metal_hot_path_cpu_fallback_free {
        500.0
    } else {
        0.0
    } + if metrics.real_model_forward {
        250.0
    } else {
        -250.0
    } + if metrics.model_bound_ffn_decode {
        100.0
    } else {
        -100.0
    };
    let gate_penalty = if execution.correctness.passed && execution.determinism.passed {
        0.0
    } else {
        100_000.0
    };

    throughput_reward + prefix_reward + native_dispatch_reward + readiness_reward
        - ttft_penalty
        - cpu_fallback_penalty
        - gate_penalty
}

fn unique_sorted_u32(values: Vec<u32>) -> Vec<u32> {
    let mut values = values;
    values.sort_unstable();
    values.dedup();
    values
}

fn unique_sorted_option_u32(values: Vec<Option<u32>>) -> Vec<Option<u32>> {
    let mut values = values;
    values.sort_by_key(|value| value.unwrap_or(0));
    values.dedup();
    values
}

fn unique_sorted_bool(values: Vec<bool>) -> Vec<bool> {
    let mut values = values;
    values.sort_unstable();
    values.dedup();
    values
}

fn best_autotune_trial<'a>(
    trials: impl IntoIterator<Item = &'a AutotuneTrialRecord>,
) -> Option<&'a AutotuneTrialRecord> {
    trials.into_iter().max_by(|left, right| {
        left.score
            .partial_cmp(&right.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn best_autotune_trial_with_history<'a>(
    trials: &'a [AutotuneTrialRecord],
    warm_start_history: &'a AutotuneWarmStartHistory,
) -> Option<&'a AutotuneTrialRecord> {
    best_autotune_trial(trials.iter().chain(warm_start_history.trials.iter()))
}

fn autotune_probe_baseline_json(baseline: &AutotuneProbeBaseline) -> Value {
    json!({
        "observed_trial_count": baseline.observed_trial_count,
        "incumbent_score": baseline.incumbent_score,
        "incumbent_decode_tok_s": baseline.incumbent_decode_tok_s,
        "score_floor": baseline.score_floor,
        "decode_tok_s_floor": baseline.decode_tok_s_floor,
        "fallback_free_decode_tok_s_floor": baseline.fallback_free_decode_tok_s_floor,
        "ttft_ceiling_ms": baseline.ttft_ceiling_ms,
    })
}

fn autotune_probe_json(probe: &AutotuneProbeObservation) -> Value {
    json!({
        "status": probe.status,
        "score": probe.score,
        "skipped_full_trial": probe.skipped_full_trial,
        "skip_reason": probe.skip_reason,
        "result_dir": probe.result_dir.as_ref().map(|path| path.display().to_string()),
        "baseline": probe.baseline.as_ref().map(autotune_probe_baseline_json),
        "shape": {
            "input_tokens_target": probe.shape.input_tokens_target,
            "output_tokens_target": probe.shape.output_tokens_target,
            "concurrency": probe.shape.concurrency,
        },
    })
}

fn autotune_candidate_config_json(candidate: &AutotuneCandidateConfig) -> Value {
    json!({
        "max_batch_tokens": candidate.max_batch_tokens,
        "kv_total_blocks": candidate.kv_total_blocks,
        "prefix_cache": candidate.prefix_cache,
    })
}

fn autotune_selection_json(selection: &AutotuneSelectionDiagnostics) -> Value {
    json!({
        "strategy": selection.strategy,
        "predicted_mean": selection.predicted_mean,
        "uncertainty": selection.uncertainty,
        "acquisition": selection.acquisition,
        "good_density": selection.good_density,
        "bad_density": selection.bad_density,
        "density_ratio": selection.density_ratio,
        "novelty_bonus": selection.novelty_bonus,
    })
}

fn autotune_trial_base_json(trial: &AutotuneTrialRecord) -> serde_json::Map<String, Value> {
    let mut object = serde_json::Map::new();
    object.insert("label".to_string(), json!(trial.label()));
    object.insert("score".to_string(), json!(trial.score));
    object.insert("status".to_string(), json!(trial.status));
    object.insert(
        "config".to_string(),
        autotune_candidate_config_json(&trial.candidate),
    );
    object.insert(
        "selection".to_string(),
        autotune_selection_json(&trial.selection),
    );
    object.insert(
        "result_dir".to_string(),
        json!(
            trial
                .result_dir
                .as_ref()
                .map(|path| path.display().to_string())
        ),
    );
    object.insert(
        "probe".to_string(),
        json!(trial.probe.as_ref().map(autotune_probe_json)),
    );
    object
}

fn autotune_probe_run_label(probe: Option<&AutotuneProbeObservation>) -> &'static str {
    match probe {
        Some(probe) if probe.skipped_full_trial => "early-stop",
        Some(_) => "full-run",
        None => "none",
    }
}

fn format_optional_f64_3(value: Option<f64>, fallback: &str) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| fallback.to_string())
}

fn format_optional_to_string<T: ToString>(value: Option<T>, fallback: &str) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| fallback.to_string())
}

fn join_display(values: impl IntoIterator<Item = String>) -> String {
    values.into_iter().collect::<Vec<_>>().join(", ")
}

#[allow(dead_code)]
fn build_autotune_result_json(
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    args: &AutotuneArgs,
    search_space: &AutotuneSearchSpace,
    warm_start_history: &AutotuneWarmStartHistory,
    trials: &[AutotuneTrialRecord],
) -> Value {
    let best_trial = best_autotune_trial_with_history(trials, warm_start_history);
    json!({
        "schema_version": AUTOTUNE_SCHEMA_VERSION,
        "manifest_path": manifest_path.display().to_string(),
        "manifest_id": manifest.id,
        "selected_backend": manifest.runtime.selected_backend,
        "iterations_requested": args.iterations,
        "iterations_completed": trials.len(),
        "history_only": trials.is_empty() && !warm_start_history.trials.is_empty(),
        "exploration_weight": args.exploration_weight,
        "warm_start_history": {
            "enabled": !args.disable_history,
            "loaded_trial_count": warm_start_history.trials.len(),
            "loaded_result_count": warm_start_history.source_dirs.len(),
            "source_dirs": warm_start_history
                .source_dirs
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>(),
        },
        "search_space": {
            "max_batch_tokens": search_space.max_batch_token_options,
            "kv_total_blocks": search_space.kv_total_block_options,
            "prefix_cache": search_space.prefix_cache_options,
            "candidate_count": search_space.max_batch_token_options.len()
                * search_space.kv_total_block_options.len()
                * search_space.prefix_cache_options.len(),
        },
        "best_trial": best_trial.map(|trial| Value::Object(autotune_trial_base_json(trial))),
        "trials": trials.iter().map(|trial| {
            let mut object = autotune_trial_base_json(trial);
            object.insert("error".to_string(), json!(trial.error));
            object.insert("metrics".to_string(), json!(trial.metrics.as_ref().map(|metrics| json!({
                "ttft_ms": metrics.ttft_ms,
                "prefill_tok_s": metrics.prefill_tok_s,
                "decode_tok_s": metrics.decode_tok_s,
                "prefix_hit_rate": metrics.prefix_hit_rate,
                "prefix_native_dispatch_count": metrics.prefix_native_dispatch_count,
                "prefix_cpu_reference_dispatch_count": metrics.prefix_cpu_reference_dispatch_count,
                "prefix_cpu_projection_row_count": metrics.prefix_cpu_projection_row_count,
                "prefix_cpu_rms_norm_element_count": metrics.prefix_cpu_rms_norm_element_count,
                "prefix_cpu_ffn_activation_element_count": metrics.prefix_cpu_ffn_activation_element_count,
                "prefix_cpu_residual_add_element_count": metrics.prefix_cpu_residual_add_element_count,
                "prefix_cpu_scale_element_count": metrics.prefix_cpu_scale_element_count,
                "direct_decode_batched_group_fallback_count": metrics.direct_decode_batched_group_fallback_count,
                "direct_decode_cpu_projection_row_count": metrics.direct_decode_cpu_projection_row_count,
                "direct_decode_cpu_rms_norm_element_count": metrics.direct_decode_cpu_rms_norm_element_count,
                "direct_decode_cpu_ffn_activation_element_count": metrics.direct_decode_cpu_ffn_activation_element_count,
                "direct_decode_cpu_residual_add_element_count": metrics.direct_decode_cpu_residual_add_element_count,
                "direct_decode_cpu_scale_element_count": metrics.direct_decode_cpu_scale_element_count,
                "mlx_metal_hot_path_cpu_fallback_free": metrics.mlx_metal_hot_path_cpu_fallback_free,
                "real_model_forward": metrics.real_model_forward,
                "model_bound_ffn_decode": metrics.model_bound_ffn_decode,
            }))));
            Value::Object(object)
        }).collect::<Vec<_>>(),
    })
}

#[allow(dead_code)]
fn build_autotune_summary_markdown(
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    search_space: &AutotuneSearchSpace,
    warm_start_history: &AutotuneWarmStartHistory,
    trials: &[AutotuneTrialRecord],
) -> String {
    let mut lines = vec![
        "# Benchmark Autotune".to_string(),
        String::new(),
        format!("- manifest: `{}`", manifest_path.display()),
        format!("- manifest_id: `{}`", manifest.id),
        format!(
            "- selected_backend: `{}`",
            selected_backend_label(manifest.runtime.selected_backend)
        ),
        format!("- trials_completed: `{}`", trials.len()),
        format!(
            "- warm_start_history.loaded_trials: `{}`",
            warm_start_history.trials.len()
        ),
        format!(
            "- warm_start_history.loaded_results: `{}`",
            warm_start_history.source_dirs.len()
        ),
        format!(
            "- history_only: `{}`",
            trials.is_empty() && !warm_start_history.trials.is_empty()
        ),
        format!(
            "- search_space.max_batch_tokens: `{}`",
            join_display(
                search_space
                    .max_batch_token_options
                    .iter()
                    .map(|value| value.to_string()),
            )
        ),
        format!(
            "- search_space.kv_total_blocks: `{}`",
            join_display(
                search_space
                    .kv_total_block_options
                    .iter()
                    .map(|value| optional_u32_label(*value)),
            )
        ),
        format!(
            "- search_space.prefix_cache: `{}`",
            join_display(
                search_space
                    .prefix_cache_options
                    .iter()
                    .map(|value| value.to_string()),
            )
        ),
    ];
    if let Some(best_trial) = best_autotune_trial_with_history(trials, warm_start_history) {
        lines.push(format!("- best_trial: `{}`", best_trial.label()));
        lines.push(format!("- best_score: `{:.3}`", best_trial.score));
        lines.push(format!("- best_config: `{}`", best_trial.candidate.label()));
    }
    lines.push(String::new());
    lines.push("| Trial | Score | Status | Probe | Strategy | Ratio | Acquisition | Max Batch | KV Blocks | Prefix Cache | Decode tok/s | TTFT ms | Prefix CPU Fallbacks |".to_string());
    lines.push(
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"
            .to_string(),
    );
    for trial in trials {
        let metrics = trial.metrics.as_ref();
        lines.push(format!(
            "| {} | {:.3} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} | {} |",
            trial.label(),
            trial.score,
            trial.status,
            autotune_probe_run_label(trial.probe.as_ref()),
            trial.selection.strategy,
            format_optional_f64_3(trial.selection.density_ratio, "n/a"),
            format_optional_f64_3(trial.selection.acquisition, "seed"),
            trial.candidate.max_batch_tokens,
            optional_u32_label(trial.candidate.kv_total_blocks),
            trial.candidate.prefix_cache,
            metrics
                .map(|metrics| metrics.decode_tok_s)
                .unwrap_or_default(),
            format_optional_to_string(metrics.and_then(|metrics| metrics.ttft_ms), "n/a"),
            format_optional_to_string(
                metrics.map(|metrics| metrics.prefix_cpu_reference_dispatch_count),
                "n/a",
            ),
        ));
    }
    lines.join("\n")
}

fn execute_manifest_with_runtime(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    if runtime.uses_mlx_runtime() {
        execute_manifest_once(manifest, runtime)
    } else {
        execute_manifest_llama_cpp_once(manifest, runtime)
    }
}

fn execute_manifest_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    match manifest.class {
        ManifestClass::Scenario => execute_scenario_once(manifest, runtime),
        ManifestClass::Replay => execute_replay_once(manifest, runtime),
    }
}

fn execute_scenario_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    let specs = scenario_specs_from_manifest(manifest)?;
    run_scenario_workload(runtime, specs)
}

fn scenario_specs_from_manifest(
    manifest: &BenchmarkManifest,
) -> Result<Vec<SyntheticRequestSpec>, CliError> {
    let shape = manifest.shape.as_ref().ok_or_else(|| {
        CliError::Contract("scenario manifest must contain a shape object".to_string())
    })?;
    if shape.input_tokens_target == 0 {
        return Err(CliError::Contract(
            "scenario manifest shape.input_tokens_target must be greater than zero".to_string(),
        ));
    }
    if shape.output_tokens_target == 0 {
        return Err(CliError::Contract(
            "scenario manifest shape.output_tokens_target must be greater than zero".to_string(),
        ));
    }
    let shared_prefix = manifest.scenario == "shared_prefix";

    let mut specs = Vec::new();
    for ordinal in 0..shape.concurrency {
        let request_id = RequestId(u64::from(ordinal) + 1);
        let prefix_group = if shared_prefix {
            Some("scenario-shared")
        } else {
            None
        };
        specs.push(SyntheticRequestSpec {
            external_id: format!("req-{}", ordinal + 1),
            request_id,
            arrival_sequence: SequenceNo(u64::from(ordinal) + 1),
            model_family: manifest.model.family.clone(),
            prompt_token_target: shape.input_tokens_target,
            input_tokens: synthetic_prompt_tokens(
                shape.input_tokens_target,
                Some("scenario"),
                prefix_group,
                None,
                ordinal,
            ),
            input_text: None,
            max_output_tokens: shape.output_tokens_target,
            sampling_params: sampling_from_manifest(manifest)?,
            metadata: prefix_group.map(str::to_string),
        });
    }

    Ok(specs)
}

fn execute_manifest_llama_cpp_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    validate_llama_cpp_benchmark_runtime(manifest, &runtime)?;

    match runtime.backend_adapter.as_ref() {
        Some(adapter) if adapter.supports_stepwise_benchmark() => match manifest.class {
            ManifestClass::Scenario => execute_llama_cpp_scenario_once(manifest, runtime),
            ManifestClass::Replay => execute_llama_cpp_replay_once(manifest, runtime),
        },
        Some(adapter) if adapter.supports_blocking_benchmark() => match manifest.class {
            ManifestClass::Scenario => execute_llama_cpp_blocking_scenario_once(manifest, runtime),
            ManifestClass::Replay => Err(CliError::Contract(
                "blocking llama.cpp benchmark adapters currently support scenario manifests only"
                    .to_string(),
            )),
        },
        _ => Err(CliError::Contract(
            "llama.cpp benchmark execution requires a supported backend adapter".to_string(),
        )),
    }
}

fn execute_llama_cpp_scenario_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    let specs = scenario_specs_from_manifest(manifest)?;
    if specs.is_empty() {
        return Err(CliError::Contract(
            "scenario manifest produced no requests".to_string(),
        ));
    }
    run_llama_cpp_scenario_workload(manifest, runtime, specs)
}

fn execute_llama_cpp_blocking_scenario_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    let specs = scenario_specs_from_manifest(manifest)?
        .into_iter()
        .map(|spec| {
            let prompt_text = synthetic_prompt_text(
                spec.prompt_token_target,
                Some(&spec.external_id),
                spec.metadata.as_deref(),
                None,
                spec.request_id.0 as u32,
            );
            spec.with_input_text(prompt_text)
        })
        .collect::<Vec<_>>();
    if specs.is_empty() {
        return Err(CliError::Contract(
            "scenario manifest produced no requests".to_string(),
        ));
    }
    run_llama_cpp_blocking_scenario_workload(manifest, runtime, specs)
}

fn execute_llama_cpp_replay_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    run_llama_cpp_replay_workload(manifest, runtime, replay_events_from_manifest(manifest)?)
}

fn run_llama_cpp_scenario_workload(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
    specs: Vec<SyntheticRequestSpec>,
) -> Result<RuntimeObservation, CliError> {
    let started = Instant::now();
    let prefill_tokens = specs
        .iter()
        .map(|spec| spec.input_tokens.len() as u64)
        .sum();
    let mut observation = RuntimeObservation {
        prefill_tokens,
        total_scheduled_tokens: prefill_tokens,
        ..RuntimeObservation::default()
    };
    let request_ids = specs.iter().map(|spec| spec.request_id).collect::<Vec<_>>();
    let max_steps = specs
        .iter()
        .map(|spec| u64::from(spec.max_output_tokens).saturating_add(256))
        .sum::<u64>()
        .max(256);
    let mut session = build_session(&runtime, &specs)?;
    observation.runtime_report = Some(session.runtime_report());

    for spec in &specs {
        session
            .submit_generate_with_request_id(
                spec.request_id.0,
                generate_request_from_spec(spec, manifest),
            )
            .map_err(|error| {
                CliError::Runtime(format!(
                    "llama.cpp benchmark request failed through the SDK contract: {error}"
                ))
            })?;
    }

    loop {
        if !llama_cpp_session_has_live_requests(&session, &request_ids)? {
            let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
            let final_reports = specs
                .iter()
                .map(|spec| {
                    let report = session.request_report(spec.request_id.0).ok_or_else(|| {
                        CliError::Runtime(format!(
                            "missing llama.cpp request {}",
                            spec.request_id.0
                        ))
                    })?;
                    Ok((spec.clone(), report))
                })
                .collect::<Result<Vec<_>, CliError>>()?;
            observation.finalize_llama_cpp(final_reports, elapsed_ms.max(1));
            return Ok(observation);
        }

        let reports_before = llama_cpp_reports_for_session(&session, &request_ids)?;
        let step = session.step_report().map_err(|error| {
            CliError::Runtime(format!(
                "llama.cpp benchmark step failed through the SDK contract: {error}"
            ))
        })?;
        let current_time_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64 + 1;
        let reports_after = llama_cpp_reports_for_session(&session, &request_ids)?;
        observation.observe_llama_cpp_session_step(
            &reports_before,
            &reports_after,
            &step,
            current_time_ms,
        );
        let progress_made = step.scheduled_requests > 0
            || step.scheduled_tokens > 0
            || llama_cpp_reports_changed(&reports_before, &reports_after);

        if observation.step_count > max_steps {
            return Err(CliError::Runtime(
                "llama.cpp scenario exceeded delegated step guard".to_string(),
            ));
        }

        if !progress_made {
            return Err(CliError::Runtime(
                "llama.cpp scenario stalled with a live delegated request".to_string(),
            ));
        }
    }
}

fn run_llama_cpp_blocking_scenario_workload(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
    specs: Vec<SyntheticRequestSpec>,
) -> Result<RuntimeObservation, CliError> {
    let started = Instant::now();
    let mut observation = RuntimeObservation::default();
    let mut session = build_session(&runtime, &specs)?;
    observation.runtime_report = Some(session.runtime_report());

    let mut total_proxy_ms = 0u64;
    let mut final_reports = Vec::new();
    for spec in specs {
        let request_started = Instant::now();
        let response = session
            .generate_with_request_id(
                spec.request_id.0,
                generate_request_from_spec(&spec, manifest),
            )
            .map_err(|error| {
                CliError::Runtime(format!(
                    "blocking llama.cpp benchmark request failed through the SDK contract: {error}"
                ))
            })?;
        let elapsed_ms = request_started
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
            + 1;
        let prompt_token_count = response
            .known_prompt_token_count()
            .unwrap_or(spec.prompt_token_target);
        let known_output_token_count = response.known_output_token_count();
        total_proxy_ms = total_proxy_ms.saturating_add(elapsed_ms);
        observation.step_count += 1;
        observation.total_selected_requests += 1;
        observation.prefill_tokens += u64::from(prompt_token_count);
        observation.total_scheduled_tokens += u64::from(prompt_token_count);
        observation.prefill_steps += elapsed_ms;
        if observation.ttft_ms.is_none() {
            observation.ttft_ms = Some(elapsed_ms);
        }

        let output_tokens = if response.output_tokens.is_empty() {
            if known_output_token_count.is_some() {
                Vec::new()
            } else {
                response
                    .output_text
                    .as_deref()
                    .map(|text| synthetic_text_output_tokens(text, spec.max_output_tokens))
                    .unwrap_or_default()
            }
        } else {
            response.output_tokens.clone()
        };
        let output_token_count = known_output_token_count.unwrap_or(output_tokens.len() as u32);
        observation.decode_tokens += u64::from(output_token_count);
        observation.total_scheduled_tokens += u64::from(output_token_count);
        observation.decode_steps += elapsed_ms;

        let route_metadata = route_metadata_from_generate_route(&response.route);
        observation.merge_route_metadata(&route_metadata);

        let final_state = match response.status {
            GenerateStatus::Pending => SessionRequestState::Running,
            GenerateStatus::Finished => SessionRequestState::Finished,
            GenerateStatus::Cancelled => SessionRequestState::Cancelled,
            GenerateStatus::Failed => SessionRequestState::Failed,
        };
        let max_output_tokens = spec.max_output_tokens;
        let output_token_logprobs = if response.output_tokens.is_empty() {
            Vec::new()
        } else {
            response.output_token_logprobs.clone()
        };
        final_reports.push((
            spec,
            SessionRequestReport {
                request_id: response.request_id,
                model_id: response.model_id,
                state: final_state,
                prompt_tokens: response.prompt_tokens,
                processed_prompt_tokens: prompt_token_count,
                output_tokens,
                output_token_logprobs,
                prompt_len: prompt_token_count,
                output_len: output_token_count,
                max_output_tokens,
                cancel_requested: false,
                execution_plan_ref: response.route.execution_plan.clone(),
                route: response.route,
                finish_reason: response.finish_reason,
                terminal_stop_reason: stop_reason_from_generate_finish_reason(
                    response.finish_reason,
                ),
                last_error: None,
            },
        ));
    }

    observation.e2e_latency_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64 + 1;
    observation.finalize_llama_cpp(final_reports, observation.e2e_latency_ms.max(1));
    observation.prefill_steps = observation.prefill_steps.max(total_proxy_ms.max(1));
    observation.decode_steps = observation.decode_steps.max(total_proxy_ms.max(1));
    Ok(observation)
}

fn run_llama_cpp_replay_workload(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
    events: Vec<ReplayEvent>,
) -> Result<RuntimeObservation, CliError> {
    let specs = events
        .iter()
        .filter_map(|event| match event {
            ReplayEvent::Submit { spec, .. } => Some(spec.clone()),
            ReplayEvent::Cancel { .. } => None,
        })
        .collect::<Vec<_>>();
    let mut session = build_session(&runtime, &specs)?;
    let spec_by_request = specs
        .iter()
        .cloned()
        .map(|spec| (spec.request_id, spec))
        .collect::<BTreeMap<_, _>>();
    let mut submitted_request_ids = Vec::new();

    let mut observation = RuntimeObservation {
        runtime_report: Some(session.runtime_report()),
        ..RuntimeObservation::default()
    };
    let mut cancelled_requests = BTreeSet::new();
    let mut event_index = 0usize;
    let mut current_time_ms = 0u64;
    let max_steps = specs
        .iter()
        .map(|spec| u64::from(spec.max_output_tokens).saturating_add(256))
        .sum::<u64>()
        .max(256);

    while event_index < events.len()
        || llama_cpp_session_has_live_requests(&session, &submitted_request_ids)?
    {
        while let Some(event) = events.get(event_index) {
            if event.t_ms() > current_time_ms {
                break;
            }

            match event {
                ReplayEvent::Submit { spec, .. } => {
                    if submitted_request_ids.contains(&spec.request_id) {
                        return Err(CliError::Runtime(format!(
                            "llama.cpp replay request {:?} was submitted more than once",
                            spec.request_id
                        )));
                    }

                    session
                        .submit_generate_with_request_id(
                            spec.request_id.0,
                            generate_request_from_spec(spec, manifest),
                        )
                        .map_err(|error| {
                            CliError::Runtime(format!(
                                "failed to submit llama.cpp replay benchmark request: {error}"
                            ))
                        })?;
                    submitted_request_ids.push(spec.request_id);
                    observation.prefill_tokens += spec.input_tokens.len() as u64;
                    observation.total_scheduled_tokens += spec.input_tokens.len() as u64;

                    let report = session.request_report(spec.request_id.0).ok_or_else(|| {
                        CliError::Runtime(format!(
                            "missing llama.cpp request {}",
                            spec.request_id.0
                        ))
                    })?;
                    observation
                        .merge_route_metadata(&route_metadata_from_generate_route(&report.route));
                }
                ReplayEvent::Cancel { request_id, .. } => {
                    if !submitted_request_ids.contains(request_id) {
                        return Err(CliError::Runtime(format!(
                            "llama.cpp replay cancel arrived before submit for {:?}",
                            request_id
                        )));
                    }

                    cancelled_requests.insert(*request_id);
                    session.cancel_request(request_id.0).map_err(|error| {
                        CliError::Runtime(format!(
                            "failed to cancel llama.cpp replay benchmark request: {error}"
                        ))
                    })?;
                    let report = session.request_report(request_id.0).ok_or_else(|| {
                        CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
                    })?;
                    observation
                        .merge_route_metadata(&route_metadata_from_generate_route(&report.route));
                }
            }

            event_index += 1;
        }

        if !llama_cpp_session_has_live_requests(&session, &submitted_request_ids)? {
            if let Some(next_event) = events.get(event_index) {
                current_time_ms = next_event.t_ms();
                continue;
            }
            break;
        }

        let tick_time_ms = current_time_ms.saturating_add(1);
        let reports_before = llama_cpp_reports_for_session(&session, &submitted_request_ids)?;
        let step = session.step_report().map_err(|error| {
            CliError::Runtime(format!(
                "llama.cpp replay step failed through the SDK contract: {error}"
            ))
        })?;
        let reports_after = llama_cpp_reports_for_session(&session, &submitted_request_ids)?;
        observation.observe_llama_cpp_session_step(
            &reports_before,
            &reports_after,
            &step,
            tick_time_ms,
        );
        let progress_made = step.scheduled_requests > 0
            || step.scheduled_tokens > 0
            || llama_cpp_reports_changed(&reports_before, &reports_after);

        current_time_ms = tick_time_ms;

        if observation.step_count > max_steps {
            return Err(CliError::Runtime(
                "llama.cpp replay exceeded delegated step guard".to_string(),
            ));
        }

        if !progress_made {
            return Err(CliError::Runtime(
                "llama.cpp replay stalled with a live delegated request".to_string(),
            ));
        }
    }

    observation.cancelled_requests = cancelled_requests;
    let final_reports = submitted_request_ids
        .iter()
        .map(|request_id| {
            let spec = spec_by_request.get(request_id).cloned().ok_or_else(|| {
                CliError::Runtime(format!("missing llama.cpp replay spec {:?}", request_id))
            })?;
            let report = session.request_report(request_id.0).ok_or_else(|| {
                CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
            })?;
            Ok((spec, report))
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    observation.finalize_llama_cpp(final_reports, current_time_ms.max(1));
    Ok(observation)
}

fn execute_replay_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    run_replay_workload(runtime, replay_events_from_manifest(manifest)?)
}

fn elapsed_ms_since(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64 + 1
}

fn run_scenario_workload(
    runtime: RuntimeConfig,
    specs: Vec<SyntheticRequestSpec>,
) -> Result<RuntimeObservation, CliError> {
    let started = Instant::now();
    let max_steps = specs
        .iter()
        .map(|spec| u64::from(spec.max_output_tokens).saturating_add(256))
        .sum::<u64>()
        .max(256);
    let mut session = build_session(&runtime, &specs)?;
    let mut submitted_request_ids = Vec::new();
    let mut external_ids = BTreeMap::new();
    let shared_prefix_staging = should_stage_shared_prefix_scenario(&specs);
    let mut pending_specs = Vec::new();

    for (index, spec) in specs.into_iter().enumerate() {
        external_ids.insert(spec.request_id, spec.external_id.clone());
        if shared_prefix_staging && index > 0 {
            pending_specs.push(spec);
            continue;
        }
        submit_scenario_spec(&mut session, &mut submitted_request_ids, spec)?;
    }

    let mut observation = RuntimeObservation {
        runtime_report: Some(session.runtime_report()),
        ..RuntimeObservation::default()
    };
    let mut current_time_ms = 0u64;

    while !pending_specs.is_empty() || has_live_requests(session.core(), &submitted_request_ids)? {
        if current_time_ms > 0 && !pending_specs.is_empty() {
            for spec in pending_specs.drain(..) {
                submit_scenario_spec(&mut session, &mut submitted_request_ids, spec)?;
            }
        }

        if submitted_request_ids.is_empty()
            || !has_live_requests(session.core(), &submitted_request_ids)?
        {
            break;
        }

        let outcome = session
            .step()
            .map_err(|error| CliError::Runtime(format!("engine session step failed: {error}")))?;
        current_time_ms = elapsed_ms_since(started);
        observation.observe_step(session.core(), &outcome, current_time_ms);

        if observation.step_count > max_steps {
            return Err(CliError::Runtime(
                "scenario workload exceeded engine step guard".to_string(),
            ));
        }

        if outcome.runner_output.is_none()
            && outcome.schedule_plan.memory_blocked_requests.is_empty()
            && outcome.schedule_plan.selected_requests.is_empty()
            && has_live_requests(session.core(), &submitted_request_ids)?
        {
            return Err(CliError::Runtime(
                "scenario workload stalled with live requests".to_string(),
            ));
        }
    }

    observation.finalize(
        session.core(),
        &submitted_request_ids,
        external_ids,
        elapsed_ms_since(started),
        runtime.block_size_tokens,
    )?;
    Ok(observation)
}

fn should_stage_shared_prefix_scenario(specs: &[SyntheticRequestSpec]) -> bool {
    specs.len() > 1
        && specs
            .iter()
            .all(|spec| spec.metadata.as_deref() == Some("scenario-shared"))
}

fn submit_scenario_spec(
    session: &mut EngineSession,
    submitted_request_ids: &mut Vec<RequestId>,
    spec: SyntheticRequestSpec,
) -> Result<(), CliError> {
    submitted_request_ids.push(spec.request_id);
    session.submit(spec.into_submission()).map_err(|error| {
        CliError::Runtime(format!("failed to submit benchmark request: {error}"))
    })?;
    Ok(())
}

fn run_replay_workload(
    runtime: RuntimeConfig,
    events: Vec<ReplayEvent>,
) -> Result<RuntimeObservation, CliError> {
    let specs = events
        .iter()
        .filter_map(|event| match event {
            ReplayEvent::Submit { spec, .. } => Some(spec.clone()),
            ReplayEvent::Cancel { .. } => None,
        })
        .collect::<Vec<_>>();
    let mut session = build_session(&runtime, &specs)?;
    let mut request_ids = Vec::new();
    let mut external_ids = BTreeMap::new();
    let mut cancelled_requests = BTreeSet::new();
    let mut observation = RuntimeObservation {
        runtime_report: Some(session.runtime_report()),
        ..RuntimeObservation::default()
    };
    let mut event_index = 0usize;
    let mut current_time_ms = 0u64;
    let max_steps = 10_000u64;

    while event_index < events.len() || has_live_requests(session.core(), &request_ids)? {
        while let Some(event) = events.get(event_index) {
            if event.t_ms() > current_time_ms {
                break;
            }

            match event {
                ReplayEvent::Submit { spec, .. } => {
                    request_ids.push(spec.request_id);
                    external_ids.insert(spec.request_id, spec.external_id.clone());
                    session
                        .submit(spec.clone().into_submission())
                        .map_err(|error| {
                            CliError::Runtime(format!(
                                "failed to submit replay benchmark request: {error}"
                            ))
                        })?;
                }
                ReplayEvent::Cancel { request_id, .. } => {
                    cancelled_requests.insert(*request_id);
                    session.cancel(*request_id).map_err(|error| {
                        CliError::Runtime(format!(
                            "failed to cancel replay benchmark request: {error}"
                        ))
                    })?;
                }
            }

            event_index += 1;
        }

        if !has_live_requests(session.core(), &request_ids)? {
            if let Some(next_event) = events.get(event_index) {
                current_time_ms = next_event.t_ms();
                continue;
            }
            break;
        }

        let outcome = session
            .step()
            .map_err(|error| CliError::Runtime(format!("engine session step failed: {error}")))?;
        current_time_ms += 1;
        observation.observe_step(session.core(), &outcome, current_time_ms);

        if current_time_ms > max_steps {
            return Err(CliError::Runtime(
                "replay workload exceeded engine step guard".to_string(),
            ));
        }

        if outcome.runner_output.is_none()
            && outcome.schedule_plan.memory_blocked_requests.is_empty()
            && outcome.schedule_plan.selected_requests.is_empty()
            && has_live_requests(session.core(), &request_ids)?
        {
            return Err(CliError::Runtime(
                "replay workload stalled with live requests".to_string(),
            ));
        }
    }

    observation.cancelled_requests = cancelled_requests;
    observation.finalize(
        session.core(),
        &request_ids,
        external_ids,
        current_time_ms,
        runtime.block_size_tokens,
    )?;
    Ok(observation)
}

fn session_config_from_runtime(
    runtime: &RuntimeConfig,
    specs: &[SyntheticRequestSpec],
) -> EngineSessionConfig {
    let estimated_total_blocks = specs
        .iter()
        .map(|spec| {
            u64::from(spec.prompt_token_target + spec.max_output_tokens)
                .div_ceil(u64::from(runtime.block_size_tokens))
        })
        .sum::<u64>()
        .max(1) as u32;
    let kv_total_blocks = runtime
        .kv_total_blocks
        .unwrap_or_else(|| estimated_total_blocks.saturating_add(8));

    let llama_backend = if runtime.resolved_backend.selected_backend.is_mlx() {
        None
    } else {
        runtime
            .backend_adapter
            .as_ref()
            .map(BackendAdapterManifest::to_sdk_config)
    };
    let mlx_runtime_artifacts_dir = if runtime.resolved_backend.selected_backend.is_mlx() {
        EngineSessionConfig::default_mlx_runtime_artifacts_dir().or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|dir| dir.join("build/metal"))
                .filter(|dir| dir.join("build_report.json").is_file())
        })
    } else {
        None
    };
    let mlx_runtime_artifacts_source = mlx_runtime_artifacts_dir
        .as_ref()
        .map(|_| NativeRuntimeArtifactsSource::RepoAutoDetect);
    EngineSessionConfig::from_resolved_request(ResolvedSessionConfigRequest {
        cache_group_id: CacheGroupId(0),
        block_size_tokens: runtime.block_size_tokens,
        total_blocks: kv_total_blocks,
        deterministic: runtime.deterministic,
        max_batch_tokens: runtime.max_batch_tokens,
        backend_policy: runtime.backend_policy,
        resolved_backend: runtime.resolved_backend.clone(),
        llama_backend,
        mlx_lm_backend: None,
        mlx_runtime_artifacts_dir,
        mlx_runtime_artifacts_source,
        mlx_model_artifacts_dir: runtime.mlx_model_artifacts_dir.clone(),
        mlx_model_artifacts_source: runtime.mlx_model_artifacts_source,
        mlx_disable_ngram_acceleration: false,
        mlx_kv_compression: ax_engine_sdk::MlxKvCompressionConfig::disabled(),
    })
}

fn build_session(
    runtime: &RuntimeConfig,
    specs: &[SyntheticRequestSpec],
) -> Result<EngineSession, CliError> {
    let session_config = session_config_from_runtime(runtime, specs);

    EngineSession::new(session_config).map_err(|error| {
        CliError::Contract(format!(
            "failed to create benchmark SDK session from manifest runtime: {error}"
        ))
    })
}

fn evaluate_correctness(
    manifest: &BenchmarkManifest,
    observation: &RuntimeObservation,
) -> Result<GateStatus, CliError> {
    if observation
        .final_requests
        .iter()
        .any(|request| request.state == "Failed")
    {
        return Ok(GateStatus::fail(
            "one or more requests reached Failed in engine bring-up execution",
        ));
    }

    if observation
        .final_requests
        .iter()
        .any(|request| request.state != "Finished" && request.state != "Cancelled")
    {
        return Ok(GateStatus::fail(
            "engine execution completed with non-terminal live requests",
        ));
    }

    let require_prefix_reuse = manifest.checks.require_prefix_reuse;
    if require_prefix_reuse && observation.prefix_hit_rate() <= f64::EPSILON {
        return Ok(GateStatus::fail(
            "manifest requires prefix reuse but runtime reported zero prefix hits",
        ));
    }

    if observation
        .cancelled_requests
        .iter()
        .any(|request_id| !observation.request_state(*request_id, "Cancelled"))
    {
        return Ok(GateStatus::fail(
            "cancelled replay request did not finish in Cancelled state",
        ));
    }

    if manifest.checks.require_no_allocator_churn_failure && observation.churn_status() == "fail" {
        return Ok(GateStatus::fail(
            "manifest requires no allocator churn failure but one or more requests reached Failed state",
        ));
    }

    Ok(GateStatus::pass())
}

fn enforce_runtime_gates(execution: &RuntimeResult) -> Result<(), CliError> {
    if !execution.correctness.passed {
        return Err(CliError::Correctness(gate_failure_reason(
            &execution.correctness,
            "correctness gate failed",
        )));
    }

    if !execution.determinism.passed {
        return Err(CliError::Correctness(gate_failure_reason(
            &execution.determinism,
            "determinism gate failed",
        )));
    }

    Ok(())
}

fn gate_failure_reason(status: &GateStatus, fallback: &str) -> String {
    status
        .reason
        .clone()
        .unwrap_or_else(|| fallback.to_string())
}

fn write_execution_artifacts(
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    output_root: &Path,
    started_at_unix_s: u64,
    execution: &RuntimeResult,
) -> Result<PathBuf, CliError> {
    let run_id = format!(
        "{}-{}",
        unix_timestamp_secs()?,
        sanitize_component(&manifest.id)
    );
    let result_dir = output_root.join(run_id.clone());

    fs::create_dir_all(&result_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create result directory {}: {error}",
            result_dir.display()
        ))
    })?;

    write_json_file(&result_dir.join("manifest.json"), manifest)?;
    write_json_file(
        &result_dir.join("environment.json"),
        &build_environment_json(
            &run_id,
            command,
            manifest_path,
            output_root,
            started_at_unix_s,
            execution,
        )?,
    )?;
    write_json_file(
        &result_dir.join("metrics.json"),
        &build_metrics_json(&run_id, execution),
    )?;
    write_json_file(
        &result_dir.join("routes.json"),
        &build_routes_json(&run_id, execution),
    )?;
    write_json_file(
        &result_dir.join("trace.json"),
        &build_trace_json(&run_id, execution),
    )?;
    fs::write(
        result_dir.join("summary.md"),
        build_execution_summary_markdown(&run_id, command, manifest_path, execution),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}

fn write_contract_failure_artifacts(
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    output_root: &Path,
    started_at_unix_s: u64,
    message: &str,
) -> Result<PathBuf, CliError> {
    let run_id = format!(
        "{}-{}-contract-failure",
        unix_timestamp_secs()?,
        sanitize_component(&manifest.id)
    );
    let result_dir = output_root.join(run_id.clone());

    fs::create_dir_all(&result_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create contract-failure directory {}: {error}",
            result_dir.display()
        ))
    })?;

    write_json_file(&result_dir.join("manifest.json"), manifest)?;
    write_json_file(
        &result_dir.join("contract_failure.json"),
        &build_contract_failure_json(
            &run_id,
            command,
            manifest_path,
            manifest,
            output_root,
            started_at_unix_s,
            message,
        )?,
    )?;
    fs::write(
        result_dir.join("summary.md"),
        build_contract_failure_summary_markdown(&run_id, command, manifest_path, manifest, message),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write contract-failure summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}

fn write_compare_artifacts(
    baseline_dir: &Path,
    candidate_dir: &Path,
    output_root: &Path,
) -> Result<PathBuf, CliError> {
    let baseline_manifest = load_json_value(&baseline_dir.join("manifest.json"))?;
    let candidate_manifest = load_json_value(&candidate_dir.join("manifest.json"))?;
    validate_comparable_manifests(&baseline_manifest, &candidate_manifest)?;
    let baseline_environment = load_json_value(&baseline_dir.join("environment.json"))?;
    let candidate_environment = load_json_value(&candidate_dir.join("environment.json"))?;
    validate_comparable_environments(&baseline_environment, &candidate_environment)?;
    let trusted_baseline = load_optional_json_value(&baseline_dir.join("trusted_baseline.json"))?;

    let baseline_metrics = load_json_value(&baseline_dir.join("metrics.json"))?;
    let candidate_metrics = load_json_value(&candidate_dir.join("metrics.json"))?;

    let manifest_id = baseline_manifest
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("compare");
    let compare_id = format!(
        "{}-compare-{}",
        unix_timestamp_secs()?,
        sanitize_component(manifest_id)
    );
    let result_dir = output_root.join(compare_id);

    fs::create_dir_all(&result_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create compare directory {}: {error}",
            result_dir.display()
        ))
    })?;

    let regression_json = build_regression_json(
        &baseline_metrics,
        &candidate_metrics,
        &baseline_environment,
        &candidate_environment,
        trusted_baseline.as_ref(),
    )?;
    write_json_file(&result_dir.join("baseline.json"), &baseline_metrics)?;
    write_json_file(&result_dir.join("candidate.json"), &candidate_metrics)?;
    write_json_file(&result_dir.join("regression.json"), &regression_json)?;
    fs::write(
        result_dir.join("comparison.md"),
        build_compare_summary_markdown(
            &baseline_metrics,
            &candidate_metrics,
            &regression_json,
            trusted_baseline.as_ref(),
        ),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write comparison.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}

fn write_trusted_baseline_artifacts(
    source_dir: &Path,
    name: &str,
    output_root: &Path,
) -> Result<PathBuf, CliError> {
    let manifest = load_json_value(&source_dir.join("manifest.json"))?;
    let environment = load_json_value(&source_dir.join("environment.json"))?;
    let metrics = load_json_value(&source_dir.join("metrics.json"))?;
    let summary_path = source_dir.join("summary.md");
    if !summary_path.is_file() {
        return Err(CliError::Contract(format!(
            "benchmark artifact missing summary.md: {}",
            source_dir.display()
        )));
    }

    let slug = sanitize_component(name.trim())
        .trim_matches('-')
        .to_string();
    if slug.is_empty() {
        return Err(CliError::Usage(
            "baseline name must contain at least one alphanumeric character".to_string(),
        ));
    }

    let baseline_dir = output_root.join(&slug);
    if baseline_dir.exists() {
        return Err(CliError::Contract(format!(
            "trusted baseline already exists and will not be overwritten: {}",
            baseline_dir.display()
        )));
    }
    fs::create_dir_all(&baseline_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create trusted baseline directory {}: {error}",
            baseline_dir.display()
        ))
    })?;

    let trusted_baseline =
        build_trusted_baseline_json(name, &slug, source_dir, &manifest, &environment, &metrics)?;
    write_json_file(
        &baseline_dir.join("trusted_baseline.json"),
        &trusted_baseline,
    )?;
    fs::write(
        baseline_dir.join("trusted_baseline.md"),
        build_trusted_baseline_summary_markdown(&trusted_baseline),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write trusted_baseline.md in {}: {error}",
            baseline_dir.display()
        ))
    })?;

    copy_required_artifact_file(source_dir, &baseline_dir, "manifest.json")?;
    copy_required_artifact_file(source_dir, &baseline_dir, "environment.json")?;
    copy_required_artifact_file(source_dir, &baseline_dir, "metrics.json")?;
    copy_required_artifact_file(source_dir, &baseline_dir, "summary.md")?;
    copy_optional_artifact_file(source_dir, &baseline_dir, "routes.json")?;
    copy_optional_artifact_file(source_dir, &baseline_dir, "trace.json")?;

    Ok(baseline_dir)
}

fn write_matrix_compare_artifacts(
    baseline_dir: &Path,
    candidate_dir: &Path,
    output_root: &Path,
) -> Result<PathBuf, CliError> {
    let baseline_matrix = load_json_value(&baseline_dir.join("matrix.json"))?;
    let candidate_matrix = load_json_value(&candidate_dir.join("matrix.json"))?;
    validate_comparable_matrix_results(&baseline_matrix, &candidate_matrix)?;

    let matrix_id = baseline_matrix
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("matrix-compare");
    let result_dir = output_root.join(format!(
        "{}-matrix-compare-{}",
        unix_timestamp_secs()?,
        sanitize_component(matrix_id)
    ));
    let cases_dir = result_dir.join("cases");
    fs::create_dir_all(&cases_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create matrix compare directory {}: {error}",
            result_dir.display()
        ))
    })?;

    let baseline_members = matrix_members_by_manifest_id(&baseline_matrix)?;
    let candidate_members = matrix_members_by_manifest_id(&candidate_matrix)?;
    let mut member_results = Vec::with_capacity(baseline_members.len());

    for (manifest_id, baseline_member) in baseline_members {
        let candidate_member = candidate_members.get(&manifest_id).ok_or_else(|| {
            CliError::Contract(format!(
                "candidate matrix missing member manifest_id={manifest_id}"
            ))
        })?;
        let baseline_result_dir = PathBuf::from(
            baseline_member
                .get("result_dir")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CliError::Contract(format!(
                        "baseline matrix member {manifest_id} missing result_dir"
                    ))
                })?,
        );
        let candidate_result_dir = PathBuf::from(
            candidate_member
                .get("result_dir")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CliError::Contract(format!(
                        "candidate matrix member {manifest_id} missing result_dir"
                    ))
                })?,
        );
        require_existing_dir(&baseline_result_dir)?;
        require_existing_dir(&candidate_result_dir)?;
        let member_compare_root = cases_dir.join(sanitize_component(&manifest_id));
        fs::create_dir_all(&member_compare_root).map_err(|error| {
            CliError::Runtime(format!(
                "failed to create matrix member compare directory {}: {error}",
                member_compare_root.display()
            ))
        })?;
        let member_compare_dir = write_compare_artifacts(
            &baseline_result_dir,
            &candidate_result_dir,
            &member_compare_root,
        )?;
        let regression = load_json_value(&member_compare_dir.join("regression.json"))?;
        member_results.push(build_matrix_compare_member_result(
            baseline_member,
            &manifest_id,
            &member_compare_dir,
            &regression,
        )?);
    }

    write_json_file(&result_dir.join("baseline_matrix.json"), &baseline_matrix)?;
    write_json_file(&result_dir.join("candidate_matrix.json"), &candidate_matrix)?;
    write_json_file(
        &result_dir.join("matrix_regression.json"),
        &build_matrix_regression_json(
            matrix_id,
            baseline_dir,
            candidate_dir,
            &baseline_matrix,
            &candidate_matrix,
            &member_results,
        ),
    )?;
    fs::write(
        result_dir.join("summary.md"),
        build_matrix_compare_summary_markdown(
            matrix_id,
            &baseline_matrix,
            &candidate_matrix,
            &member_results,
        ),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write matrix compare summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(result_dir)
}

#[derive(Clone, Debug)]
struct MatrixMemberResult {
    label: String,
    manifest_id: String,
    manifest_path: PathBuf,
    scenario: String,
    model_family: String,
    status: String,
    tool_mode: String,
    selected_backend: String,
    support_tier: String,
    resolution_policy: String,
    result_dir: PathBuf,
    correctness_passed: Option<bool>,
    determinism_passed: Option<bool>,
    ttft_ms: Option<f64>,
    decode_tok_s: Option<f64>,
    prefix_hit_rate: Option<f64>,
    failure_code: Option<String>,
    failure_message: Option<String>,
}

#[derive(Clone, Debug)]
struct MatrixExecutionResult {
    result_dir: PathBuf,
    overall_status: String,
    members: Vec<MatrixMemberResult>,
}

#[derive(Clone, Debug)]
struct MatrixCompareMemberResult {
    label: String,
    manifest_id: String,
    compare_result_dir: PathBuf,
    compare_mode: String,
    tool_mode: String,
    execution_semantics: String,
    selected_backend: String,
    support_tier: String,
    resolution_policy: String,
    ttft_ms_pct: f64,
    decode_tok_s_pct: f64,
    memory_peak_mb_pct: f64,
    prefix_hit_rate_pct: f64,
}

fn execute_matrix_manifest(
    matrix_manifest_path: &Path,
    matrix_manifest: &BenchmarkMatrixManifest,
    output_root: &Path,
) -> Result<MatrixExecutionResult, CliError> {
    let run_id = format!(
        "{}-matrix-{}",
        unix_timestamp_secs()?,
        sanitize_component(&matrix_manifest.id)
    );
    let result_dir = output_root.join(&run_id);
    let cases_dir = result_dir.join("cases");
    fs::create_dir_all(&cases_dir).map_err(|error| {
        CliError::Runtime(format!(
            "failed to create matrix result directory {}: {error}",
            result_dir.display()
        ))
    })?;

    let mut members = Vec::with_capacity(matrix_manifest.members.len());
    for member in &matrix_manifest.members {
        let manifest_path = resolve_matrix_member_manifest_path(matrix_manifest_path, member);
        require_existing_file(&manifest_path)?;
        let manifest = load_manifest(&manifest_path)?;
        validate_manifest(&manifest, ManifestClass::Scenario)?;
        let started_at_unix_s = unix_timestamp_secs()?;
        let member_output_root = cases_dir.join(sanitize_component(&manifest.id));
        fs::create_dir_all(&member_output_root).map_err(|error| {
            CliError::Runtime(format!(
                "failed to create matrix member output root {}: {error}",
                member_output_root.display()
            ))
        })?;

        let member_result = match execute_manifest_runtime(&manifest) {
            Ok(execution) => {
                let artifact_dir = write_execution_artifacts(
                    "scenario",
                    &manifest_path,
                    &manifest,
                    &member_output_root,
                    started_at_unix_s,
                    &execution,
                )?;
                MatrixMemberResult {
                    label: member.label.clone().unwrap_or_else(|| manifest.id.clone()),
                    manifest_id: manifest.id.clone(),
                    manifest_path: manifest_path.clone(),
                    scenario: manifest.scenario.clone(),
                    model_family: manifest.model.family.clone(),
                    status: execution.status_label().to_string(),
                    tool_mode: execution.tool_mode.to_string(),
                    selected_backend: json_string_label(
                        execution.runtime.resolved_backend.selected_backend,
                    ),
                    support_tier: json_string_label(
                        execution.runtime.resolved_backend.support_tier,
                    ),
                    resolution_policy: json_string_label(
                        execution.runtime.backend_policy.resolution_policy,
                    ),
                    result_dir: artifact_dir,
                    correctness_passed: Some(execution.correctness.passed),
                    determinism_passed: Some(execution.determinism.passed),
                    ttft_ms: Some(execution.observation.ttft_ms.unwrap_or_default() as f64),
                    decode_tok_s: Some(execution.observation.decode_tok_s()),
                    prefix_hit_rate: Some(execution.observation.prefix_hit_rate()),
                    failure_code: None,
                    failure_message: execution
                        .correctness
                        .reason
                        .clone()
                        .or(execution.determinism.reason.clone()),
                }
            }
            Err(CliError::Contract(message)) => {
                let artifact_dir = write_contract_failure_artifacts(
                    "scenario",
                    &manifest_path,
                    &manifest,
                    &member_output_root,
                    started_at_unix_s,
                    &message,
                )?;
                let failure = classify_contract_failure(&manifest, &message);
                MatrixMemberResult {
                    label: member.label.clone().unwrap_or_else(|| manifest.id.clone()),
                    manifest_id: manifest.id.clone(),
                    manifest_path: manifest_path.clone(),
                    scenario: manifest.scenario.clone(),
                    model_family: manifest.model.family.clone(),
                    status: "contract_failure".to_string(),
                    tool_mode: contract_failure_tool_mode(&manifest).to_string(),
                    selected_backend: json_string_label(manifest.runtime.selected_backend),
                    support_tier: json_string_label(manifest.runtime.support_tier),
                    resolution_policy: json_string_label(manifest.runtime.resolution_policy),
                    result_dir: artifact_dir,
                    correctness_passed: None,
                    determinism_passed: None,
                    ttft_ms: None,
                    decode_tok_s: None,
                    prefix_hit_rate: None,
                    failure_code: Some(failure.code.to_string()),
                    failure_message: Some(message),
                }
            }
            Err(error) => return Err(error),
        };
        members.push(member_result);
    }

    let overall_status = matrix_overall_status(&members).to_string();
    write_json_file(&result_dir.join("matrix_manifest.json"), matrix_manifest)?;
    write_json_file(
        &result_dir.join("matrix.json"),
        &build_matrix_json(
            &run_id,
            matrix_manifest_path,
            matrix_manifest,
            &result_dir,
            &members,
        )?,
    )?;
    fs::write(
        result_dir.join("summary.md"),
        build_matrix_summary_markdown(matrix_manifest, &members, &overall_status),
    )
    .map_err(|error| {
        CliError::Runtime(format!(
            "failed to write matrix summary.md in {}: {error}",
            result_dir.display()
        ))
    })?;

    Ok(MatrixExecutionResult {
        result_dir,
        overall_status,
        members,
    })
}

fn resolve_matrix_member_manifest_path(
    matrix_manifest_path: &Path,
    member: &BenchmarkMatrixMember,
) -> PathBuf {
    let manifest_path = PathBuf::from(&member.manifest);
    if manifest_path.is_absolute() {
        manifest_path
    } else {
        matrix_manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(manifest_path)
    }
}

fn matrix_overall_status(members: &[MatrixMemberResult]) -> &'static str {
    if members
        .iter()
        .any(|member| member.status == "contract_failure")
    {
        "contract_failure"
    } else if members.iter().any(|member| member.status != "ok") {
        "completed_with_failures"
    } else {
        "ok"
    }
}

fn build_matrix_json(
    run_id: &str,
    matrix_manifest_path: &Path,
    matrix_manifest: &BenchmarkMatrixManifest,
    result_dir: &Path,
    members: &[MatrixMemberResult],
) -> Result<Value, CliError> {
    let contract_failure_count = members
        .iter()
        .filter(|member| member.status == "contract_failure")
        .count();
    let completed_with_failures_count = members
        .iter()
        .filter(|member| member.status == "completed_with_failures")
        .count();
    let ok_count = members
        .iter()
        .filter(|member| member.status == "ok")
        .count();
    Ok(json!({
        "schema_version": "ax.engine_bench.matrix_result.v1",
        "run_id": run_id,
        "id": matrix_manifest.id,
        "class": matrix_manifest.class,
        "status": matrix_overall_status(members),
        "matrix_manifest_path": matrix_manifest_path.display().to_string(),
        "matrix_manifest_fingerprint_fnv1a64": file_fingerprint_fnv1a64(matrix_manifest_path)?,
        "result_dir": result_dir.display().to_string(),
        "summary": {
            "member_count": members.len(),
            "ok_count": ok_count,
            "completed_with_failures_count": completed_with_failures_count,
            "contract_failure_count": contract_failure_count
        },
        "members": members.iter().map(|member| {
            let mut value = json!({
                "label": member.label,
                "manifest_id": member.manifest_id,
                "manifest_path": member.manifest_path.display().to_string(),
                "scenario": member.scenario,
                "model_family": member.model_family,
                "status": member.status,
                "tool_mode": member.tool_mode,
                "selected_backend": member.selected_backend,
                "support_tier": member.support_tier,
                "resolution_policy": member.resolution_policy,
                "result_dir": member.result_dir.display().to_string()
            });
            if let Some(value_ttft_ms) = member.ttft_ms {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("ttft_ms".to_string(), json!(value_ttft_ms));
            }
            if let Some(value_decode_tok_s) = member.decode_tok_s {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("decode_tok_s".to_string(), json!(value_decode_tok_s));
            }
            if let Some(value_prefix_hit_rate) = member.prefix_hit_rate {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("prefix_hit_rate".to_string(), json!(value_prefix_hit_rate));
            }
            if let Some(correctness_passed) = member.correctness_passed {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("correctness_passed".to_string(), json!(correctness_passed));
            }
            if let Some(determinism_passed) = member.determinism_passed {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("determinism_passed".to_string(), json!(determinism_passed));
            }
            if let Some(failure_code) = &member.failure_code {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("failure_code".to_string(), json!(failure_code));
            }
            if let Some(failure_message) = &member.failure_message {
                value
                    .as_object_mut()
                    .expect("matrix member JSON should serialize as object")
                    .insert("failure_message".to_string(), json!(failure_message));
            }
            value
        }).collect::<Vec<_>>()
    }))
}

fn build_matrix_summary_markdown(
    matrix_manifest: &BenchmarkMatrixManifest,
    members: &[MatrixMemberResult],
    overall_status: &str,
) -> String {
    let mut lines = vec![
        "# Benchmark Matrix".to_string(),
        String::new(),
        format!("- id: `{}`", matrix_manifest.id),
        format!("- status: `{overall_status}`"),
        format!("- member_count: `{}`", members.len()),
        String::new(),
        "| Label | Scenario | Model | Status | TTFT ms | Decode tok/s | Prefix hit rate |"
            .to_string(),
        "| --- | --- | --- | --- | ---: | ---: | ---: |".to_string(),
    ];

    for member in members {
        let ttft_ms = member
            .ttft_ms
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let decode_tok_s = member
            .decode_tok_s
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        let prefix_hit_rate = member
            .prefix_hit_rate
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "-".to_string());
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            member.label,
            member.scenario,
            member.model_family,
            member.status,
            ttft_ms,
            decode_tok_s,
            prefix_hit_rate
        ));
        lines.push(format!("result_dir: `{}`", member.result_dir.display()));
        if let Some(failure_code) = &member.failure_code {
            lines.push(format!("failure_code: `{failure_code}`"));
        }
    }

    lines.push(String::new());
    lines.push(
        "This summary is the frozen scenario matrix roll-up for Tier 2 dense-path benchmarking."
            .to_string(),
    );
    lines.join("\n")
}

fn validate_comparable_matrix_results(baseline: &Value, candidate: &Value) -> Result<(), CliError> {
    validate_matching_json_field(baseline, candidate, &["schema_version"])?;
    validate_matching_json_field(baseline, candidate, &["id"])?;
    validate_matching_json_field(baseline, candidate, &["class"])?;
    validate_matching_json_field(
        baseline,
        candidate,
        &["matrix_manifest_fingerprint_fnv1a64"],
    )?;

    let baseline_status = baseline
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let candidate_status = candidate
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if baseline_status != "ok" {
        return Err(CliError::Contract(format!(
            "matrix compare requires successful matrix execution artifacts; baseline status is {baseline_status}"
        )));
    }
    if candidate_status != "ok" {
        return Err(CliError::Contract(format!(
            "matrix compare requires successful matrix execution artifacts; candidate status is {candidate_status}"
        )));
    }

    let baseline_members = matrix_members_by_manifest_id(baseline)?;
    let candidate_members = matrix_members_by_manifest_id(candidate)?;
    if baseline_members.len() != candidate_members.len() {
        return Err(CliError::Contract(format!(
            "matrix compare member-count mismatch: baseline={}, candidate={}",
            baseline_members.len(),
            candidate_members.len()
        )));
    }
    for manifest_id in baseline_members.keys() {
        if !candidate_members.contains_key(manifest_id) {
            return Err(CliError::Contract(format!(
                "candidate matrix missing member manifest_id={manifest_id}"
            )));
        }
    }

    Ok(())
}

fn matrix_members_by_manifest_id(matrix: &Value) -> Result<BTreeMap<String, &Value>, CliError> {
    let members = matrix
        .get("members")
        .and_then(Value::as_array)
        .ok_or_else(|| CliError::Contract("matrix result missing members array".to_string()))?;
    let mut by_manifest_id = BTreeMap::new();
    for member in members {
        let manifest_id = member
            .get("manifest_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                CliError::Contract("matrix result member missing manifest_id".to_string())
            })?
            .to_string();
        if by_manifest_id.insert(manifest_id.clone(), member).is_some() {
            return Err(CliError::Contract(format!(
                "matrix result contains duplicate manifest_id={manifest_id}"
            )));
        }
    }
    Ok(by_manifest_id)
}

fn build_matrix_compare_member_result(
    baseline_member: &Value,
    manifest_id: &str,
    compare_result_dir: &Path,
    regression: &Value,
) -> Result<MatrixCompareMemberResult, CliError> {
    Ok(MatrixCompareMemberResult {
        label: baseline_member
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or(manifest_id)
            .to_string(),
        manifest_id: manifest_id.to_string(),
        compare_result_dir: compare_result_dir.to_path_buf(),
        compare_mode: explicit_or_inferred_compare_mode(regression),
        tool_mode: explicit_or_inferred_tool_mode(
            regression,
            &["summary", "tool_mode"],
            "engine_bringup_runtime",
        ),
        execution_semantics: nested_value(regression, &["summary", "execution_semantics"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        selected_backend: nested_value(regression, &["summary", "selected_backend"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        support_tier: nested_value(regression, &["summary", "support_tier"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        resolution_policy: nested_value(regression, &["summary", "resolution_policy"])
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        ttft_ms_pct: nested_value(regression, &["comparison", "ttft_ms_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing ttft_ms_pct"
                ))
            })?,
        decode_tok_s_pct: nested_value(regression, &["comparison", "decode_tok_s_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing decode_tok_s_pct"
                ))
            })?,
        memory_peak_mb_pct: nested_value(regression, &["comparison", "memory_peak_mb_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing memory_peak_mb_pct"
                ))
            })?,
        prefix_hit_rate_pct: nested_value(regression, &["comparison", "prefix_hit_rate_pct"])
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                CliError::Contract(format!(
                    "matrix compare member {manifest_id} missing prefix_hit_rate_pct"
                ))
            })?,
    })
}

fn build_matrix_regression_json(
    matrix_id: &str,
    baseline_dir: &Path,
    candidate_dir: &Path,
    baseline_matrix: &Value,
    candidate_matrix: &Value,
    members: &[MatrixCompareMemberResult],
) -> Value {
    json!({
        "schema_version": "ax.engine_bench.matrix_regression.v1",
        "id": matrix_id,
        "baseline_matrix_run_id": baseline_matrix.get("run_id").cloned().unwrap_or(Value::String("baseline".to_string())),
        "candidate_matrix_run_id": candidate_matrix.get("run_id").cloned().unwrap_or(Value::String("candidate".to_string())),
        "baseline_dir": baseline_dir.display().to_string(),
        "candidate_dir": candidate_dir.display().to_string(),
        "summary": {
            "member_count": members.len(),
            "requires_human_review": true
        },
        "members": members.iter().map(|member| json!({
            "label": member.label,
            "manifest_id": member.manifest_id,
            "compare_result_dir": member.compare_result_dir.display().to_string(),
            "compare_mode": member.compare_mode,
            "tool_mode": member.tool_mode,
            "execution_semantics": member.execution_semantics,
            "selected_backend": member.selected_backend,
            "support_tier": member.support_tier,
            "resolution_policy": member.resolution_policy,
            "comparison": {
                "ttft_ms_pct": member.ttft_ms_pct,
                "decode_tok_s_pct": member.decode_tok_s_pct,
                "memory_peak_mb_pct": member.memory_peak_mb_pct,
                "prefix_hit_rate_pct": member.prefix_hit_rate_pct
            }
        })).collect::<Vec<_>>()
    })
}

fn build_matrix_compare_summary_markdown(
    matrix_id: &str,
    baseline_matrix: &Value,
    candidate_matrix: &Value,
    members: &[MatrixCompareMemberResult],
) -> String {
    let baseline_run_id = baseline_matrix
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("baseline");
    let candidate_run_id = candidate_matrix
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("candidate");
    let mut lines = vec![
        "# Benchmark Matrix Compare".to_string(),
        String::new(),
        format!("- id: `{matrix_id}`"),
        format!("- baseline_matrix_run_id: `{baseline_run_id}`"),
        format!("- candidate_matrix_run_id: `{candidate_run_id}`"),
        format!("- member_count: `{}`", members.len()),
        String::new(),
        "| Label | Mode | TTFT % | Decode tok/s % | Memory peak MB % | Prefix hit rate % |"
            .to_string(),
        "| --- | --- | ---: | ---: | ---: | ---: |".to_string(),
    ];

    for member in members {
        lines.push(format!(
            "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} |",
            member.label,
            member.compare_mode,
            member.ttft_ms_pct,
            member.decode_tok_s_pct,
            member.memory_peak_mb_pct,
            member.prefix_hit_rate_pct
        ));
        lines.push(format!(
            "compare_result_dir: `{}`",
            member.compare_result_dir.display()
        ));
    }

    lines.push(String::new());
    lines.push(
        "This summary is the frozen matrix regression roll-up. Read per-member compare artifacts before drawing performance conclusions."
            .to_string(),
    );
    lines.join("\n")
}

fn enforce_matrix_gates(execution: &MatrixExecutionResult) -> Result<(), CliError> {
    if execution.overall_status == "contract_failure" {
        let failed_members = execution
            .members
            .iter()
            .filter(|member| member.status == "contract_failure")
            .map(|member| member.manifest_id.clone())
            .collect::<Vec<_>>();
        return Err(CliError::Contract(format!(
            "matrix completed with contract failures in {} member(s): {}",
            failed_members.len(),
            failed_members.join(", ")
        )));
    }

    if execution.overall_status == "completed_with_failures" {
        let failed_members = execution
            .members
            .iter()
            .filter(|member| member.status != "ok")
            .map(|member| member.manifest_id.clone())
            .collect::<Vec<_>>();
        return Err(CliError::Correctness(format!(
            "matrix completed with failed correctness or determinism gates in {} member(s): {}",
            failed_members.len(),
            failed_members.join(", ")
        )));
    }

    Ok(())
}

fn build_trusted_baseline_json(
    name: &str,
    slug: &str,
    source_dir: &Path,
    manifest: &Value,
    environment: &Value,
    metrics: &Value,
) -> Result<Value, CliError> {
    let created_at_unix_s = unix_timestamp_secs()?;
    let source_run_id = metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_id = manifest
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_class = manifest
        .get("class")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let scenario = manifest
        .get("scenario")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_family = nested_value(manifest, &["model", "family"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let tool_mode =
        explicit_or_inferred_tool_mode(environment, &["software", "tool_mode"], "unknown");
    let selected_backend = nested_value(environment, &["runtime", "selected_backend"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = nested_value(environment, &["runtime", "support_tier"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = nested_value(environment, &["runtime", "resolution_policy"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_path = nested_value(environment, &["route", "prefix_cache_path"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_evidence = nested_value(environment, &["route", "prefix_cache_evidence"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_reuse_provenance = nested_value(environment, &["route", "prefix_reuse_provenance"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let route_json = environment
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let execution_semantics = route_execution_semantics_from_environment_json(environment);
    let metal_numeric_scaffold_only = route_flag_from_json(
        &route_json,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let metal_real_model_forward = route_flag_from_json(
        &route_json,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let metal_complete_model_forward_supported = route_flag_from_json(
        &route_json,
        "metal_complete_model_forward_supported",
        &[
            "metal_dispatch_complete_model_forward_supported",
            "metal_dispatch_runtime_complete_model_forward_supported",
        ],
    );
    let metal_model_conditioned_inputs = route_flag_from_json(
        &route_json,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let mlx_metal_prefix_dispatch_count = route_count_from_json(
        &route_json,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        &route_json,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let metal_direct_decode_tokens = route_flag_from_json(
        &route_json,
        "metal_direct_decode_tokens",
        &["metal_dispatch_direct_decode_tokens"],
    );
    let metal_direct_decode_model_bound_ffn = route_flag_from_json(
        &route_json,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let metal_direct_decode_checksum_lo =
        nested_value(&route_json, &["metal_direct_decode_checksum_lo"])
            .cloned()
            .unwrap_or_else(|| json!(0));
    let metal_real_model_tensor_inputs = route_flag_from_json(
        &route_json,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let metal_model_artifacts_validated = route_flag_from_json(
        &route_json,
        "metal_model_artifacts_validated",
        &["metal_dispatch_model_artifacts_validated"],
    );
    let mlx_metal_projection_f32_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_f32_binding_count",
        &["metal_dispatch_native_projection_f32_binding_count"],
    );
    let mlx_metal_projection_f16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_f16_binding_count",
        &["metal_dispatch_native_projection_f16_binding_count"],
    );
    let mlx_metal_projection_bf16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_bf16_binding_count",
        &["metal_dispatch_native_projection_bf16_binding_count"],
    );
    let mlx_metal_projection_unsupported_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_unsupported_binding_count",
        &["metal_dispatch_native_projection_unsupported_binding_count"],
    );
    let mlx_metal_projection_source_quantized_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_projection_source_quantized_binding_count",
        &["metal_dispatch_native_projection_source_quantized_binding_count"],
    );
    let mlx_metal_rms_norm_f32_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_f32_binding_count",
        &["metal_dispatch_native_rms_norm_f32_binding_count"],
    );
    let mlx_metal_rms_norm_f16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_f16_binding_count",
        &["metal_dispatch_native_rms_norm_f16_binding_count"],
    );
    let mlx_metal_rms_norm_bf16_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_bf16_binding_count",
        &["metal_dispatch_native_rms_norm_bf16_binding_count"],
    );
    let mlx_metal_rms_norm_unsupported_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_unsupported_binding_count",
        &["metal_dispatch_native_rms_norm_unsupported_binding_count"],
    );
    let mlx_metal_rms_norm_source_quantized_binding_count = route_count_from_json(
        &route_json,
        "mlx_metal_rms_norm_source_quantized_binding_count",
        &["metal_dispatch_native_rms_norm_source_quantized_binding_count"],
    );
    let backend_reported_cached_prompt_tokens = nested_value(
        environment,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .cloned();
    let machine = environment
        .get("machine")
        .cloned()
        .unwrap_or_else(|| json!({ "device": "unknown" }));
    let software = environment
        .get("software")
        .cloned()
        .unwrap_or_else(|| json!({ "tool_mode": tool_mode.clone() }));

    let mut baseline = json!({
        "schema_version": "ax.engine_bench.trusted_baseline.v1",
        "name": name,
        "slug": slug,
        "created_at_unix_s": created_at_unix_s,
        "source_result_dir": source_dir.display().to_string(),
        "source_run_id": source_run_id,
        "manifest": {
            "id": manifest_id,
            "class": manifest_class,
            "scenario": scenario,
            "model_family": model_family
        },
        "runtime": {
            "tool_mode": tool_mode,
            "selected_backend": selected_backend,
            "support_tier": support_tier,
            "resolution_policy": resolution_policy
        },
        "route": {
            "execution_semantics": execution_semantics,
            "prefix_cache_path": prefix_cache_path,
            "prefix_cache_evidence": prefix_cache_evidence,
            "prefix_reuse_provenance": prefix_reuse_provenance,
            "metal_numeric_scaffold_only": metal_numeric_scaffold_only,
            "metal_complete_model_forward_supported": metal_complete_model_forward_supported,
            "metal_real_model_forward": metal_real_model_forward,
            "metal_model_conditioned_inputs": metal_model_conditioned_inputs,
            "mlx_metal_prefix_dispatch_count": mlx_metal_prefix_dispatch_count,
            "metal_prefix_cpu_reference_dispatch_count": metal_prefix_cpu_reference_dispatch_count,
            "metal_direct_decode_tokens": metal_direct_decode_tokens,
            "metal_direct_decode_model_bound_ffn": metal_direct_decode_model_bound_ffn,
            "metal_direct_decode_checksum_lo": metal_direct_decode_checksum_lo,
            "metal_real_model_tensor_inputs": metal_real_model_tensor_inputs,
            "metal_model_artifacts_validated": metal_model_artifacts_validated,
            "mlx_metal_projection_f32_binding_count": mlx_metal_projection_f32_binding_count,
            "mlx_metal_projection_f16_binding_count": mlx_metal_projection_f16_binding_count,
            "mlx_metal_projection_bf16_binding_count": mlx_metal_projection_bf16_binding_count,
            "mlx_metal_projection_unsupported_binding_count": mlx_metal_projection_unsupported_binding_count,
            "mlx_metal_projection_source_quantized_binding_count": mlx_metal_projection_source_quantized_binding_count,
            "mlx_metal_rms_norm_f32_binding_count": mlx_metal_rms_norm_f32_binding_count,
            "mlx_metal_rms_norm_f16_binding_count": mlx_metal_rms_norm_f16_binding_count,
            "mlx_metal_rms_norm_bf16_binding_count": mlx_metal_rms_norm_bf16_binding_count,
            "mlx_metal_rms_norm_unsupported_binding_count": mlx_metal_rms_norm_unsupported_binding_count,
            "mlx_metal_rms_norm_source_quantized_binding_count": mlx_metal_rms_norm_source_quantized_binding_count
        },
        "machine": machine,
        "software": software,
        "metrics": {
            "ttft_ms": metric_number(metrics, "ttft_ms")?,
            "decode_tok_s": metric_number(metrics, "decode_tok_s")?,
            "memory_peak_mb": metric_number(metrics, "memory_peak_mb")?,
            "prefix_hit_rate": metric_number(metrics, "prefix_hit_rate")?
        }
    });

    if let Some(value) = backend_reported_cached_prompt_tokens {
        baseline
            .get_mut("route")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline route should serialize as object")
            .insert("backend_reported_cached_prompt_tokens".to_string(), value);
    }

    if let Some(adapter) = nested_value(environment, &["runtime", "backend_adapter"]).cloned() {
        baseline
            .get_mut("runtime")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline runtime should serialize as object")
            .insert("backend_adapter".to_string(), adapter);
    }

    if let Some(mlx_runtime) = nested_value(environment, &["runtime", "mlx_runtime"]).cloned() {
        baseline
            .get_mut("runtime")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline runtime should serialize as object")
            .insert("mlx_runtime".to_string(), mlx_runtime);
    }

    if let Some(mlx_model) = nested_value(environment, &["runtime", "mlx_model"]).cloned() {
        baseline
            .get_mut("runtime")
            .and_then(Value::as_object_mut)
            .expect("trusted baseline runtime should serialize as object")
            .insert("mlx_model".to_string(), mlx_model);
    }

    Ok(baseline)
}

fn build_trusted_baseline_summary_markdown(trusted_baseline: &Value) -> String {
    let name = trusted_baseline
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let source_run_id = trusted_baseline
        .get("source_run_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let source_result_dir = trusted_baseline
        .get("source_result_dir")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_id = nested_value(trusted_baseline, &["manifest", "id"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let manifest_class = nested_value(trusted_baseline, &["manifest", "class"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let scenario = nested_value(trusted_baseline, &["manifest", "scenario"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let model_family = nested_value(trusted_baseline, &["manifest", "model_family"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let tool_mode = nested_value(trusted_baseline, &["runtime", "tool_mode"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let selected_backend = nested_value(trusted_baseline, &["runtime", "selected_backend"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = nested_value(trusted_baseline, &["runtime", "support_tier"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = nested_value(trusted_baseline, &["runtime", "resolution_policy"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let backend_adapter = nested_value(trusted_baseline, &["runtime", "backend_adapter"])
        .map(json_value_label)
        .unwrap_or_else(|| "none".to_string());
    let prefix_cache_path = nested_value(trusted_baseline, &["route", "prefix_cache_path"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_evidence = nested_value(trusted_baseline, &["route", "prefix_cache_evidence"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_reuse_provenance =
        nested_value(trusted_baseline, &["route", "prefix_reuse_provenance"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let execution_semantics = nested_value(trusted_baseline, &["route", "execution_semantics"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let metal_numeric_scaffold_only =
        nested_value(trusted_baseline, &["route", "metal_numeric_scaffold_only"])
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let metal_complete_model_forward_supported = nested_value(
        trusted_baseline,
        &["route", "metal_complete_model_forward_supported"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_real_model_forward =
        nested_value(trusted_baseline, &["route", "metal_real_model_forward"])
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let metal_model_conditioned_inputs = nested_value(
        trusted_baseline,
        &["route", "metal_model_conditioned_inputs"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let mlx_metal_prefix_layers_attention = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_prefix_layers_attention"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_prefix_layers_cpu_reference = nested_value(
        trusted_baseline,
        &["route", "metal_prefix_layers_cpu_reference"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let mlx_metal_prefix_dispatch_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_prefix_dispatch_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_prefix_cpu_reference_dispatch_count = nested_value(
        trusted_baseline,
        &["route", "metal_prefix_cpu_reference_dispatch_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_tokens =
        nested_value(trusted_baseline, &["route", "metal_direct_decode_tokens"])
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let metal_direct_decode_model_bound_ffn = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_model_bound_ffn"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_direct_decode_checksum_lo = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_checksum_lo"],
    )
    .map(json_value_label)
    .unwrap_or_else(|| "0".to_string());
    let metal_real_model_tensor_inputs = nested_value(
        trusted_baseline,
        &["route", "metal_real_model_tensor_inputs"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let metal_model_artifacts_validated = nested_value(
        trusted_baseline,
        &["route", "metal_model_artifacts_validated"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let mlx_metal_projection_f32_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_f32_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_projection_f16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_f16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_projection_bf16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_bf16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_projection_unsupported_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_projection_unsupported_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_f32_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_f32_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_f16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_f16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_bf16_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_bf16_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let mlx_metal_rms_norm_unsupported_binding_count = nested_value(
        trusted_baseline,
        &["route", "mlx_metal_rms_norm_unsupported_binding_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_logits_group_count = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batched_logits_group_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_logits_token_count = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batched_logits_token_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_count = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batched_group_fallback_count"],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_token_count = nested_value(
        trusted_baseline,
        &[
            "route",
            "metal_direct_decode_batched_group_fallback_token_count",
        ],
    )
    .and_then(Value::as_u64)
    .unwrap_or(0);
    let metal_direct_decode_batching_opportunity_observed = nested_value(
        trusted_baseline,
        &["route", "metal_direct_decode_batching_opportunity_observed"],
    )
    .map(json_value_label)
    .unwrap_or_else(|| "unknown".to_string());
    let backend_reported_cached_prompt_tokens = nested_value(
        trusted_baseline,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .map(json_value_label)
    .unwrap_or_else(|| "none".to_string());
    let route_json = trusted_baseline
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mlx_kv_cache_lines = mlx_kv_cache_route_markdown_lines(&route_json, "");
    let ttft_ms = nested_value(trusted_baseline, &["metrics", "ttft_ms"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let decode_tok_s = nested_value(trusted_baseline, &["metrics", "decode_tok_s"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let memory_peak_mb = nested_value(trusted_baseline, &["metrics", "memory_peak_mb"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let prefix_hit_rate = nested_value(trusted_baseline, &["metrics", "prefix_hit_rate"])
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    format!(
        "# Trusted Benchmark Baseline\n\n- name: `{name}`\n- source_run_id: `{source_run_id}`\n- source_result_dir: `{source_result_dir}`\n- manifest_id: `{manifest_id}`\n- manifest_class: `{manifest_class}`\n- scenario: `{scenario}`\n- model_family: `{model_family}`\n- tool_mode: `{tool_mode}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{backend_adapter}`\n- execution_semantics: `{execution_semantics}`\n- metal_numeric_scaffold_only: `{metal_numeric_scaffold_only}`\n- metal_complete_model_forward_supported: `{metal_complete_model_forward_supported}`\n- metal_model_conditioned_inputs: `{metal_model_conditioned_inputs}`\n- mlx_metal_prefix_layers_attention: `{mlx_metal_prefix_layers_attention}`\n- metal_prefix_layers_cpu_reference: `{metal_prefix_layers_cpu_reference}`\n- mlx_metal_prefix_dispatch_count: `{mlx_metal_prefix_dispatch_count}`\n- metal_prefix_cpu_reference_dispatch_count: `{metal_prefix_cpu_reference_dispatch_count}`\n- metal_direct_decode_tokens: `{metal_direct_decode_tokens}`\n- metal_direct_decode_batching_opportunity_observed: `{metal_direct_decode_batching_opportunity_observed}`\n- metal_direct_decode_model_bound_ffn: `{metal_direct_decode_model_bound_ffn}`\n- metal_direct_decode_checksum_lo: `{metal_direct_decode_checksum_lo}`\n- metal_direct_decode_batched_logits_group_count: `{metal_direct_decode_batched_logits_group_count}`\n- metal_direct_decode_batched_logits_token_count: `{metal_direct_decode_batched_logits_token_count}`\n- metal_direct_decode_batched_group_fallback_count: `{metal_direct_decode_batched_group_fallback_count}`\n- metal_direct_decode_batched_group_fallback_token_count: `{metal_direct_decode_batched_group_fallback_token_count}`\n- metal_real_model_tensor_inputs: `{metal_real_model_tensor_inputs}`\n- metal_real_model_forward: `{metal_real_model_forward}`\n- metal_model_artifacts_validated: `{metal_model_artifacts_validated}`\n- mlx_metal_projection_f32_binding_count: `{mlx_metal_projection_f32_binding_count}`\n- mlx_metal_projection_f16_binding_count: `{mlx_metal_projection_f16_binding_count}`\n- mlx_metal_projection_bf16_binding_count: `{mlx_metal_projection_bf16_binding_count}`\n- mlx_metal_projection_unsupported_binding_count: `{mlx_metal_projection_unsupported_binding_count}`\n- mlx_metal_rms_norm_f32_binding_count: `{mlx_metal_rms_norm_f32_binding_count}`\n- mlx_metal_rms_norm_f16_binding_count: `{mlx_metal_rms_norm_f16_binding_count}`\n- mlx_metal_rms_norm_bf16_binding_count: `{mlx_metal_rms_norm_bf16_binding_count}`\n- mlx_metal_rms_norm_unsupported_binding_count: `{mlx_metal_rms_norm_unsupported_binding_count}`\n{mlx_kv_cache_lines}- prefix_cache_path: `{prefix_cache_path}`\n- prefix_cache_evidence: `{prefix_cache_evidence}`\n- prefix_reuse_provenance: `{prefix_reuse_provenance}`\n- backend_reported_cached_prompt_tokens: `{backend_reported_cached_prompt_tokens}`\n- ttft_ms: `{ttft_ms:.2}`\n- decode_tok_s: `{decode_tok_s:.2}`\n- memory_peak_mb: `{memory_peak_mb:.2}`\n- prefix_hit_rate: `{prefix_hit_rate:.2}`\n\nThis directory is a named trusted baseline snapshot. Compare future benchmark runs against this directory instead of overwriting it in place.\n"
    )
}

fn copy_required_artifact_file(
    source_dir: &Path,
    destination_dir: &Path,
    name: &str,
) -> Result<(), CliError> {
    let source = source_dir.join(name);
    if !source.is_file() {
        return Err(CliError::Contract(format!(
            "benchmark artifact missing {name}: {}",
            source_dir.display()
        )));
    }
    fs::copy(&source, destination_dir.join(name)).map_err(|error| {
        CliError::Runtime(format!(
            "failed to copy {} into {}: {error}",
            source.display(),
            destination_dir.display()
        ))
    })?;
    Ok(())
}

fn copy_optional_artifact_file(
    source_dir: &Path,
    destination_dir: &Path,
    name: &str,
) -> Result<(), CliError> {
    let source = source_dir.join(name);
    if !source.is_file() {
        return Ok(());
    }
    fs::copy(&source, destination_dir.join(name)).map_err(|error| {
        CliError::Runtime(format!(
            "failed to copy {} into {}: {error}",
            source.display(),
            destination_dir.display()
        ))
    })?;
    Ok(())
}

fn reject_contract_failure_artifact_dir(path: &Path, label: &str) -> Result<(), CliError> {
    let failure_path = path.join("contract_failure.json");
    if !failure_path.is_file() {
        return Ok(());
    }

    let failure = load_json_value(&failure_path)?;
    let code = nested_value(&failure, &["failure", "code"])
        .and_then(Value::as_str)
        .unwrap_or("contract_validation_failed");
    let message = nested_value(&failure, &["failure", "message"])
        .and_then(Value::as_str)
        .unwrap_or("contract failure artifact present");

    Err(CliError::Contract(format!(
        "compare requires successful execution artifacts; {label} points to a contract-failure result at {} ({code}): {message}",
        path.display()
    )))
}

fn validate_comparable_manifests(baseline: &Value, candidate: &Value) -> Result<(), CliError> {
    let fields = [
        ["schema_version"].as_slice(),
        ["id"].as_slice(),
        ["class"].as_slice(),
        ["scenario"].as_slice(),
        ["model"].as_slice(),
        ["runtime"].as_slice(),
        ["sampling"].as_slice(),
        ["checks"].as_slice(),
    ];

    for field in fields {
        validate_matching_json_field(baseline, candidate, field)?;
    }

    validate_matching_optional_json_field(baseline, candidate, &["source"])?;

    match nested_string(baseline, &["class"])? {
        "scenario" => validate_matching_json_field(baseline, candidate, &["shape"])?,
        "replay" => validate_matching_json_field(baseline, candidate, &["events"])?,
        other => {
            return Err(CliError::Contract(format!(
                "unsupported manifest class for compare validation: {other}"
            )));
        }
    }

    Ok(())
}

fn validate_comparable_environments(baseline: &Value, candidate: &Value) -> Result<(), CliError> {
    let fields = [
        ["schema_version"].as_slice(),
        ["machine", "arch"].as_slice(),
        ["machine", "system_model"].as_slice(),
        ["machine", "soc"].as_slice(),
        ["machine", "memory_bytes"].as_slice(),
        ["software", "os_family"].as_slice(),
        ["software", "os_version"].as_slice(),
        ["software", "os_build"].as_slice(),
        ["software", "kernel_release"].as_slice(),
        ["software", "metal_driver"].as_slice(),
        ["software", "tool_mode"].as_slice(),
        ["runtime", "selected_backend"].as_slice(),
        ["runtime", "support_tier"].as_slice(),
        ["runtime", "resolution_policy"].as_slice(),
        ["runtime", "host", "os"].as_slice(),
        ["runtime", "host", "arch"].as_slice(),
        ["runtime", "host", "supported_mlx_runtime"].as_slice(),
        ["runtime", "host", "unsupported_host_override_active"].as_slice(),
        ["runtime", "metal_toolchain", "fully_available"].as_slice(),
        ["runtime", "metal_toolchain", "metal", "available"].as_slice(),
        ["runtime", "metal_toolchain", "metallib", "available"].as_slice(),
        ["runtime", "metal_toolchain", "metal_ar", "available"].as_slice(),
        ["route", "prefix_cache_path"].as_slice(),
        ["route", "prefix_cache_evidence"].as_slice(),
        ["route", "prefix_reuse_provenance"].as_slice(),
        ["benchmark", "schema_family"].as_slice(),
        ["benchmark", "subcommand"].as_slice(),
    ];

    for field in fields {
        validate_matching_json_field(baseline, candidate, field)?;
    }

    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "host", "detected_soc"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "metal_toolchain", "metal", "version"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "metal_toolchain", "metallib", "version"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "metal_toolchain", "metal_ar", "version"],
    )?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "fallback_reason"])?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "backend_adapter"])?;
    validate_matching_optional_json_field(baseline, candidate, &["runtime", "llama_cpp_preset"])?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_runtime", "runner"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_runtime", "artifacts_source"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "artifacts_source"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "model_family"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "tensor_format"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "layer_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "tensor_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "tie_word_embeddings"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "bindings_prepared"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "buffers_bound"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "buffer_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["runtime", "mlx_model", "buffer_bytes"],
    )?;
    validate_matching_optional_json_field(baseline, candidate, &["route", "execution_semantics"])?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_numeric_scaffold_only"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_complete_model_forward_supported"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_real_model_forward"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_model_conditioned_inputs"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_prefix_layers_attention"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_prefix_layers_cpu_reference"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_prefix_dispatch_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_prefix_cpu_reference_dispatch_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_direct_decode_tokens"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_direct_decode_model_bound_ffn"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_real_model_tensor_inputs"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "metal_model_artifacts_validated"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_f32_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_f16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_bf16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_projection_unsupported_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_f32_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_f16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_bf16_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "mlx_metal_rms_norm_unsupported_binding_count"],
    )?;
    validate_matching_optional_json_field(
        baseline,
        candidate,
        &["route", "backend_reported_cached_prompt_tokens"],
    )?;

    Ok(())
}

fn validate_matching_json_field(
    baseline: &Value,
    candidate: &Value,
    field: &[&str],
) -> Result<(), CliError> {
    let baseline_value = nested_value(baseline, field)
        .ok_or_else(|| CliError::Contract(format!("baseline missing {}", field.join("."))))?;
    let candidate_value = nested_value(candidate, field)
        .ok_or_else(|| CliError::Contract(format!("candidate missing {}", field.join("."))))?;
    validate_matching_values(field, baseline_value, candidate_value)
}

fn validate_matching_optional_json_field(
    baseline: &Value,
    candidate: &Value,
    field: &[&str],
) -> Result<(), CliError> {
    match (
        nested_value(baseline, field),
        nested_value(candidate, field),
    ) {
        (None, None) => Ok(()),
        (Some(baseline_value), Some(candidate_value)) => {
            validate_matching_values(field, baseline_value, candidate_value)
        }
        (Some(_), None) => Err(CliError::Contract(format!(
            "candidate missing {}",
            field.join(".")
        ))),
        (None, Some(_)) => Err(CliError::Contract(format!(
            "baseline missing {}",
            field.join(".")
        ))),
    }
}

fn validate_matching_values(
    field: &[&str],
    baseline_value: &Value,
    candidate_value: &Value,
) -> Result<(), CliError> {
    if baseline_value != candidate_value {
        return Err(CliError::Contract(format!(
            "benchmark contract mismatch for {}: baseline={}, candidate={}",
            field.join("."),
            json_value_label(baseline_value),
            json_value_label(candidate_value)
        )));
    }

    Ok(())
}

fn json_value_label(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("{value:?}"))
}

fn inferred_tool_mode_from_runtime_json(runtime: &Value) -> Option<&'static str> {
    match nested_value(runtime, &["mlx_runtime", "runner"]).and_then(Value::as_str) {
        Some("metal_bringup") => return Some("engine_bringup_runtime"),
        _ => {}
    }

    match nested_value(runtime, &["backend_adapter", "kind"]).and_then(Value::as_str) {
        Some("llama_cpp_server_completion") => return Some("llama_cpp_stepwise_runtime"),
        Some(_) => return Some("llama_cpp_blocking_runtime"),
        None => {}
    }

    match runtime.get("selected_backend").and_then(Value::as_str) {
        Some("mlx") => None,
        Some(_) => Some("llama_cpp_blocking_runtime"),
        None => None,
    }
}

fn explicit_or_inferred_tool_mode(
    artifact: &Value,
    explicit_path: &[&str],
    fallback: &'static str,
) -> String {
    nested_value(artifact, explicit_path)
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            artifact
                .get("runtime")
                .and_then(inferred_tool_mode_from_runtime_json)
                .map(str::to_string)
        })
        .unwrap_or_else(|| fallback.to_string())
}

fn explicit_or_inferred_compare_mode(regression: &Value) -> String {
    nested_value(regression, &["summary", "result"])
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            let tool_mode =
                explicit_or_inferred_tool_mode(regression, &["summary", "tool_mode"], "unknown");
            let execution_semantics = nested_value(regression, &["summary", "execution_semantics"])
                .and_then(Value::as_str)
                .unwrap_or("unknown");

            if tool_mode != "unknown" {
                return Some(compare_result_label(&tool_mode, execution_semantics).to_string());
            }

            regression
                .get("runtime")
                .and_then(inferred_tool_mode_from_runtime_json)
                .map(|tool_mode| compare_result_label(tool_mode, "unknown"))
                .map(str::to_string)
        })
        .unwrap_or_else(|| "engine_bringup_compare".to_string())
}

fn string_at_path_or_unknown(json: &Value, path: &[&str]) -> String {
    nested_value(json, path)
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string()
}

fn compare_result_label(tool_mode: &str, execution_semantics: &str) -> &'static str {
    match tool_mode {
        "llama_cpp_stepwise_runtime" => "llama_cpp_stepwise_compare",
        "llama_cpp_blocking_runtime" => "llama_cpp_blocking_compare",
        _ => match execution_semantics {
            "metal_real_model_forward" => "metal_real_model_forward_compare",
            "metal_multilayer_mixed_prefix_attention" => {
                "metal_multilayer_mixed_prefix_attention_compare"
            }
            "metal_multilayer_native_prefix_attention" => {
                "metal_multilayer_native_prefix_attention_compare"
            }
            "metal_multilayer_model_incomplete" => "metal_multilayer_model_incomplete_compare",
            "metal_model_bound_ffn_decode" => "metal_model_bound_ffn_decode_compare",
            "metal_real_model_tensor_inputs" => "metal_real_model_tensor_inputs_compare",
            "metal_model_conditioned_numeric_scaffold" => {
                "metal_model_conditioned_scaffold_compare"
            }
            "metal_numeric_scaffold_only" => "metal_numeric_scaffold_compare",
            _ => "engine_bringup_compare",
        },
    }
}

fn compare_summary_note(tool_mode: &str, execution_semantics: &str) -> &'static str {
    match tool_mode {
        "llama_cpp_blocking_runtime" => {
            "\nThis comparison reflects the current llama.cpp llama.cpp benchmark path. TTFT, prefill throughput, and decode throughput should be read as delegated llama.cpp wall-time proxies rather than MLX-mode scheduler measurements.\n"
        }
        "llama_cpp_stepwise_runtime" => {
            "\nThis comparison reflects the delegated llama.cpp benchmark path over the stepwise `llama.cpp /completion` adapter. Throughput and prefix-reuse deltas should be read as delegated request-cadence and backend-managed prompt-cache evidence, not as AX Engine scheduler, KV, or runner evidence.\n"
        }
        _ if execution_semantics == "metal_real_model_forward" => {
            "\nThis comparison reflects Metal runs that explicitly marked real model forward execution. Compare only against the same execution semantics and validated model provenance.\n"
        }
        _ if execution_semantics == "metal_multilayer_mixed_prefix_attention" => {
            "\nThis comparison reflects multi-layer Metal runs where some prefix-layer attention already executed through MLX Metal dispatch while part of the prefix path still fell back to CPU reference. Read it as partial MLX dense-forward progress, not as full MLX prefix coverage or release-grade inference.\n"
        }
        _ if execution_semantics == "metal_multilayer_native_prefix_attention" => {
            "\nThis comparison reflects Metal runs where validated multi-layer model artifacts were present and at least part of prefix-layer attention already executed through MLX Metal dispatch, but the full dense forward path still remained incomplete. Treat these runs as stronger bring-up evidence than CPU-reference multilayer staging, but not as release-grade MLX inference.\n"
        }
        _ if execution_semantics == "metal_multilayer_model_incomplete" => {
            "\nThis comparison reflects Metal runs that loaded validated multi-layer model artifacts, but did not report a complete real model forward path. Treat these runs as partial MLX Metal bring-up milestones rather than final dense inference performance.\n"
        }
        _ if execution_semantics == "metal_model_bound_ffn_decode" => {
            "\nThis comparison reflects Metal runs that consumed real model tensor inputs and reported model-bound FFN direct decode continuation. It is stronger evidence than tensor-input bring-up alone, but it is still not a full release-grade native forward path.\n"
        }
        _ if execution_semantics == "metal_real_model_tensor_inputs" => {
            "\nThis comparison reflects Metal runs with real model tensor inputs present, but without a reported real model forward pass. Treat deltas as model-bound bring-up progress rather than final inference performance.\n"
        }
        _ if execution_semantics == "metal_model_conditioned_numeric_scaffold" => {
            "\nThis comparison reflects Metal runs with model-conditioned staged inputs that still remain numeric scaffold bring-up paths. Do not use these deltas as release-grade inference conclusions.\n"
        }
        _ if execution_semantics == "metal_numeric_scaffold_only" => {
            "\nThis comparison reflects Metal numeric scaffold bring-up runs. The command queue, kernels, and runtime may be real, but the workload is still not a full model forward.\n"
        }
        _ => {
            "\nThis comparison reflects the current engine bring-up benchmark path and still requires human review before performance conclusions are used for release decisions.\n"
        }
    }
}

fn build_environment_json(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    output_root: &Path,
    started_at_unix_s: u64,
    execution: &RuntimeResult,
) -> Result<Value, CliError> {
    let completed_at_unix_s = unix_timestamp_secs().unwrap_or(started_at_unix_s);
    let manifest_fingerprint = file_fingerprint_fnv1a64(manifest_path)?;
    let cwd = env::current_dir().map_err(|error| {
        CliError::Runtime(format!(
            "failed to resolve current working directory for benchmark provenance: {error}"
        ))
    })?;
    let hostname = env::var("HOSTNAME")
        .or_else(|_| env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let logical_cpu_count = thread::available_parallelism()
        .ok()
        .map(|parallelism| parallelism.get());
    let system_model = detect_system_model().unwrap_or_else(|| "unknown".to_string());
    let soc = detect_soc().unwrap_or_else(|| "unknown".to_string());
    let memory_bytes = detect_memory_bytes();
    let memory_gb = memory_bytes.map(bytes_to_gib);
    let os_version = detect_os_version().unwrap_or_else(|| "unknown".to_string());
    let os_build = detect_os_build().unwrap_or_else(|| "unknown".to_string());
    let kernel_release = detect_kernel_release().unwrap_or_else(|| "unknown".to_string());
    let metal_driver =
        env::var("AX_METAL_DRIVER").unwrap_or_else(|_| default_metal_driver().to_string());
    let device = if system_model != "unknown" {
        system_model.clone()
    } else {
        hostname.clone()
    };
    let mut route_json = serialize_route_metadata(&execution.observation.route_metadata);
    annotate_route_json_with_decode_batching_opportunity(
        &mut route_json,
        &execution.observation.step_trace,
    );

    Ok(json!({
        "schema_version": "ax.engine_bench.environment.v1",
        "run_id": run_id,
        "engine_commit": env::var("AX_ENGINE_COMMIT").unwrap_or_else(|_| "unknown".to_string()),
        "machine": {
            "device": device,
            "hostname": hostname,
            "system_model": system_model,
            "soc": soc,
            "memory_gb": memory_gb,
            "memory_bytes": memory_bytes,
            "arch": env::consts::ARCH,
            "logical_cpu_count": logical_cpu_count
        },
        "software": {
            "os_family": env::consts::OS,
            "os_version": os_version,
            "os_build": os_build,
            "kernel_release": kernel_release,
            "metal_driver": metal_driver,
            "tool_mode": execution.tool_mode,
            "ax_engine_bench_version": env!("CARGO_PKG_VERSION")
        },
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "route": route_json,
        "benchmark": {
            "started_at": format!("unix:{started_at_unix_s}"),
            "started_at_unix_s": started_at_unix_s,
            "completed_at_unix_s": completed_at_unix_s,
            "command": format!(
                "ax-engine-bench {} --manifest {} --output-root {}",
                command,
                manifest_path.display(),
                output_root.display()
            ),
            "subcommand": command,
            "schema_family": "ax.engine_bench.v1",
            "manifest_path": manifest_path.display().to_string(),
            "manifest_fingerprint_fnv1a64": manifest_fingerprint,
            "output_root": output_root.display().to_string(),
            "cwd": cwd.display().to_string()
        }
    }))
}

fn detect_system_model() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "hw.model"]),
        _ => None,
    }
}

fn detect_soc() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"]),
        _ => None,
    }
}

fn detect_memory_bytes() -> Option<u64> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse::<u64>().ok()),
        _ => None,
    }
}

fn detect_os_version() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-productVersion"]),
        _ => None,
    }
}

fn detect_os_build() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-buildVersion"]),
        _ => None,
    }
}

fn detect_kernel_release() -> Option<String> {
    command_stdout("uname", &["-r"])
}

fn default_metal_driver() -> &'static str {
    match env::consts::OS {
        "macos" => "system-default",
        _ => "unavailable",
    }
}

fn bytes_to_gib(bytes: u64) -> u64 {
    bytes / (1024 * 1024 * 1024)
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn file_fingerprint_fnv1a64(path: &Path) -> Result<String, CliError> {
    let bytes = fs::read(path).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read {} for benchmark provenance fingerprint: {error}",
            path.display()
        ))
    })?;
    Ok(format!("{:016x}", fnv1a64(&bytes)))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS_64: u64 = 0xcbf29ce484222325;
    const FNV_PRIME_64: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS_64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME_64);
    }
    hash
}

fn build_metrics_json(run_id: &str, execution: &RuntimeResult) -> Value {
    let mut metrics_json = json!({
        "schema_version": "ax.engine_bench.metrics.v1",
        "run_id": run_id,
        "status": execution.status_label(),
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "metrics": {
            "ttft_ms": execution.observation.ttft_ms.unwrap_or_default() as f64,
            "prefill_tok_s": execution.observation.prefill_tok_s(),
            "decode_tok_s": execution.observation.decode_tok_s(),
            "e2e_latency_ms": execution.observation.e2e_latency_ms as f64,
            "memory_peak_mb": execution.observation.memory_peak_mb,
            "cpu_time_per_token_us": execution.observation.cpu_time_per_token_us(),
            "runner_time_per_token_us": execution.observation.runner_time_per_token_us(),
            "runner_time_share_pct": execution.observation.runner_time_share_pct(),
            "prefix_hit_rate": execution.observation.prefix_hit_rate(),
            "kv_block_usage": execution.observation.kv_peak_blocks as f64,
            "evictions": execution.observation.evictions as f64
        },
        "correctness": {
            "passed": execution.correctness.passed,
            "reason": execution.correctness.reason
        },
        "determinism": {
            "passed": execution.determinism.passed,
            "reason": execution.determinism.reason
        },
        "step_count": execution.observation.step_count,
        "scheduled_tokens_per_step": execution.observation.scheduled_tokens_per_step(),
        "memory_blocked_steps": execution.observation.memory_blocked_steps,
        "memory_blocked_request_events": execution.observation.memory_blocked_request_events,
        "replay_status": execution.observation.replay_status(),
        "churn_status": execution.observation.churn_status()
    });

    if execution.runtime.llama_cpp_preset.is_some() {
        metrics_json
            .get_mut("metrics")
            .and_then(Value::as_object_mut)
            .expect("metrics artifact should contain an object metrics field")
            .insert(
                "delegated_llama_cpp".to_string(),
                json!({
                    "kv_usage_blocks": execution.observation.kv_peak_blocks,
                    "requests_processing_events": execution.observation.llama_cpp_processing_request_events,
                    "requests_deferred_events": execution.observation.llama_cpp_deferred_request_events,
                    "backend_reported_cached_prompt_tokens": backend_reported_cached_prompt_tokens(
                        &execution.observation.route_metadata
                    ).unwrap_or(0),
                    "cache_reuse_observed": backend_reported_cached_prompt_tokens(
                        &execution.observation.route_metadata
                    ).is_some()
                }),
            );
    }

    metrics_json
}

fn build_routes_json(run_id: &str, execution: &RuntimeResult) -> Value {
    let mut route_json = serialize_route_metadata(&execution.observation.route_metadata);
    annotate_route_json_with_decode_batching_opportunity(
        &mut route_json,
        &execution.observation.step_trace,
    );
    json!({
        "schema_version": "ax.engine_bench.routes.v1",
        "run_id": run_id,
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "route": route_json,
        "mode": execution.tool_mode
    })
}

fn build_trace_json(run_id: &str, execution: &RuntimeResult) -> Value {
    json!({
        "schema_version": "ax.engine_bench.trace.v1",
        "run_id": run_id,
        "mode": execution.tool_mode,
        "runtime": serialize_runtime_metadata(
            &execution.runtime,
            execution.observation.runtime_report.as_ref(),
        ),
        "steps": execution
            .observation
            .step_trace
            .iter()
            .map(StepTraceEntry::json)
            .collect::<Vec<_>>(),
        "observation": execution.observation.digest
    })
}

fn build_execution_summary_markdown(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    execution: &RuntimeResult,
) -> String {
    let correctness = if execution.correctness.passed {
        "pass"
    } else {
        "fail"
    };
    let determinism = if execution.determinism.passed {
        "pass"
    } else {
        "fail"
    };
    let fallback_reason = execution
        .runtime
        .resolved_backend
        .fallback_reason
        .as_deref()
        .unwrap_or("none");
    let selected_backend = json_string_label(execution.runtime.resolved_backend.selected_backend);
    let support_tier = json_string_label(execution.runtime.resolved_backend.support_tier);
    let resolution_policy = json_string_label(execution.runtime.backend_policy.resolution_policy);
    let backend_adapter = execution
        .runtime
        .backend_adapter
        .as_ref()
        .map(|adapter| serde_json::to_string(adapter).unwrap_or_else(|_| "unknown".to_string()))
        .unwrap_or_else(|| "none".to_string());
    let prefix_reuse_provenance =
        prefix_reuse_provenance_label(&execution.observation.route_metadata);
    let execution_semantics =
        route_execution_semantics_label(&execution.observation.route_metadata);
    let metal_numeric_scaffold_only = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_numeric_scaffold_only",
    );
    let metal_complete_model_forward_supported = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_complete_model_forward_supported",
    ) || route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_runtime_complete_model_forward_supported",
    );
    let metal_real_model_forward = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_real_model_forward",
    );
    let metal_model_conditioned_inputs = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_model_conditioned_inputs",
    ) || route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_runtime_model_conditioned_inputs",
    );
    let mlx_metal_prefix_layers_attention = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_layers_native_attention",
    );
    let metal_prefix_layers_cpu_reference = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_layers_cpu_reference",
    );
    let mlx_metal_prefix_dispatch_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_dispatch_count",
    );
    let metal_prefix_cpu_reference_dispatch_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_reference_dispatch_count",
    );
    let mlx_metal_prefix_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_projection_row_count",
    );
    let metal_prefix_cpu_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_projection_row_count",
    );
    let mlx_metal_prefix_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_rms_norm_element_count",
    );
    let metal_prefix_cpu_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_rms_norm_element_count",
    );
    let mlx_metal_prefix_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_ffn_activation_element_count",
    );
    let metal_prefix_cpu_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_ffn_activation_element_count",
    );
    let mlx_metal_prefix_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_residual_add_element_count",
    );
    let metal_prefix_cpu_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_residual_add_element_count",
    );
    let mlx_metal_prefix_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_native_scale_element_count",
    );
    let metal_prefix_cpu_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_prefix_cpu_scale_element_count",
    );
    let metal_direct_decode_tokens = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_tokens",
    );
    let metal_direct_decode_model_bound_ffn = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_model_bound_ffn",
    );
    let metal_direct_decode_checksum_lo = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_checksum_lo",
    );
    let metal_real_model_tensor_inputs = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_real_model_tensor_inputs",
    ) || route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_runtime_real_model_tensor_inputs",
    );
    let metal_model_artifacts_validated = route_decision_flag(
        &execution.observation.route_metadata,
        "metal_dispatch_model_artifacts_validated",
    );
    let mlx_metal_projection_f32_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_f32_binding_count",
    );
    let mlx_metal_projection_f16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_f16_binding_count",
    );
    let mlx_metal_projection_bf16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_bf16_binding_count",
    );
    let mlx_metal_projection_unsupported_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_unsupported_binding_count",
    );
    let mlx_metal_projection_source_quantized_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_projection_source_quantized_binding_count",
    );
    let mlx_metal_rms_norm_f32_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_f32_binding_count",
    );
    let mlx_metal_rms_norm_f16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_f16_binding_count",
    );
    let mlx_metal_rms_norm_bf16_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_bf16_binding_count",
    );
    let mlx_metal_rms_norm_unsupported_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_unsupported_binding_count",
    );
    let mlx_metal_rms_norm_source_quantized_binding_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_native_rms_norm_source_quantized_binding_count",
    );
    let mlx_metal_direct_decode_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_projection_row_count",
    );
    let metal_direct_decode_cpu_projection_row_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_projection_row_count",
    );
    let mlx_metal_direct_decode_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_rms_norm_element_count",
    );
    let metal_direct_decode_cpu_rms_norm_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
    );
    let mlx_metal_direct_decode_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_ffn_activation_element_count",
    );
    let metal_direct_decode_cpu_ffn_activation_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
    );
    let mlx_metal_direct_decode_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_residual_add_element_count",
    );
    let metal_direct_decode_cpu_residual_add_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_residual_add_element_count",
    );
    let mlx_metal_direct_decode_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_native_scale_element_count",
    );
    let metal_direct_decode_cpu_scale_element_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_cpu_scale_element_count",
    );
    let metal_direct_decode_batched_logits_group_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_logits_group_count",
    );
    let metal_direct_decode_batched_logits_token_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_logits_token_count",
    );
    let metal_direct_decode_batched_group_fallback_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_group_fallback_count",
    );
    let metal_direct_decode_batched_group_fallback_token_count = route_decision_value(
        &execution.observation.route_metadata,
        "metal_dispatch_direct_decode_batched_group_fallback_token_count",
    );
    let backend_reported_cached_prompt_tokens =
        backend_reported_cached_prompt_tokens(&execution.observation.route_metadata)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
    let llama_cpp_preset = execution
        .runtime
        .llama_cpp_preset
        .as_ref()
        .map(|preset| serde_json::to_string(preset).unwrap_or_else(|_| "unknown".to_string()))
        .unwrap_or_else(|| "none".to_string());
    let llama_cpp_processing_request_events =
        execution.observation.llama_cpp_processing_request_events;
    let llama_cpp_deferred_request_events = execution.observation.llama_cpp_deferred_request_events;
    let prefix_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_projection_row_count,
        metal_prefix_cpu_projection_row_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_rms_norm_element_count,
        metal_prefix_cpu_rms_norm_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_ffn_activation_element_count,
        metal_prefix_cpu_ffn_activation_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_residual_add_element_count,
        metal_prefix_cpu_residual_add_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let prefix_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_prefix_scale_element_count,
        metal_prefix_cpu_scale_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_projection_row_count,
        metal_direct_decode_cpu_projection_row_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_rms_norm_element_count,
        metal_direct_decode_cpu_rms_norm_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_ffn_activation_element_count,
        metal_direct_decode_cpu_ffn_activation_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_residual_add_element_count,
        metal_direct_decode_cpu_residual_add_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let direct_decode_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        mlx_metal_direct_decode_scale_element_count,
        metal_direct_decode_cpu_scale_element_count,
    )
    .map(|value| format!("{:.2}%", value * 100.0))
    .unwrap_or_else(|| "n/a".to_string());
    let metal_direct_decode_batching_opportunity_observed =
        direct_decode_batching_opportunity_observed(&execution.observation.step_trace);
    let native_dense_dequantized_source = execution
        .runtime
        .native_model_report()
        .is_some_and(|report| native_model_report_has_dense_dequantized_source(&report));
    let mlx_metal_readiness = mlx_metal_readiness(MlxMetalReadinessInputs {
        tool_mode: execution.tool_mode,
        selected_backend: &selected_backend,
        native_dense_dequantized_source,
        native_quantized_projection_binding_count:
            mlx_metal_projection_source_quantized_binding_count,
        direct_decode_batching_opportunity_observed:
            metal_direct_decode_batching_opportunity_observed,
        metal_complete_model_forward_supported,
        metal_real_model_forward,
        metal_model_artifacts_validated,
        mlx_metal_prefix_layers_attention,
        metal_prefix_layers_cpu_reference,
        metal_prefix_cpu_reference_dispatch_count,
        mlx_metal_prefix_projection_row_count,
        metal_prefix_cpu_projection_row_count,
        mlx_metal_prefix_rms_norm_element_count,
        metal_prefix_cpu_rms_norm_element_count,
        mlx_metal_prefix_ffn_activation_element_count,
        metal_prefix_cpu_ffn_activation_element_count,
        mlx_metal_prefix_residual_add_element_count,
        metal_prefix_cpu_residual_add_element_count,
        mlx_metal_prefix_scale_element_count,
        metal_prefix_cpu_scale_element_count,
        metal_direct_decode_tokens,
        metal_direct_decode_model_bound_ffn,
        metal_direct_decode_batched_logits_group_count,
        metal_direct_decode_batched_logits_token_count,
        metal_direct_decode_batched_group_fallback_count,
        metal_direct_decode_batched_group_fallback_token_count,
        mlx_metal_direct_decode_projection_row_count,
        metal_direct_decode_cpu_projection_row_count,
        mlx_metal_direct_decode_rms_norm_element_count,
        metal_direct_decode_cpu_rms_norm_element_count,
        mlx_metal_direct_decode_ffn_activation_element_count,
        metal_direct_decode_cpu_ffn_activation_element_count,
        mlx_metal_direct_decode_residual_add_element_count,
        metal_direct_decode_cpu_residual_add_element_count,
        mlx_metal_direct_decode_scale_element_count,
        metal_direct_decode_cpu_scale_element_count,
    });
    let mlx_metal_readiness_status = mlx_metal_readiness.status;
    let mlx_metal_hot_path_cpu_fallback_free = mlx_metal_readiness.hot_path_cpu_fallback_free;
    let mlx_metal_batched_direct_decode_logits_ready =
        mlx_metal_readiness.batched_direct_decode_logits_ready;
    let mlx_metal_prefix_min_dispatch_share =
        mlx_metal_dispatch_share_label(mlx_metal_readiness.prefix_min_mlx_metal_dispatch_share);
    let mlx_metal_direct_decode_min_dispatch_share = mlx_metal_dispatch_share_label(
        mlx_metal_readiness.direct_decode_min_mlx_metal_dispatch_share,
    );
    let mlx_metal_readiness_blockers = mlx_metal_readiness.blockers_label();
    let summary_note = if execution.tool_mode == "llama_cpp_blocking_runtime" {
        "\nLlama.cpp benchmarking currently runs through the supported llama.cpp route. TTFT, prefill throughput, and decode throughput should be read as llama.cpp wall-time proxies rather than MLX-mode stepwise measurements.\n"
    } else if execution.tool_mode == "llama_cpp_stepwise_runtime" {
        "\nLlama.cpp benchmarking currently runs through the delegated SDK stepwise lifecycle over `llama.cpp /completion`. TTFT and decode throughput reflect delegated step cadence, and any reported prefix reuse is backend-managed prompt-cache evidence such as `delegated_prompt_cache`, not AX Engine scheduler / KV reuse. Prefill throughput, KV metrics, and runner metrics therefore remain exploratory.\n"
    } else if execution_semantics == "metal_real_model_forward" {
        "\nThis run reports a Metal path that marked real model forward execution. Compare it only against runs with the same execution semantics and validated model provenance.\n"
    } else if execution_semantics == "metal_multilayer_mixed_prefix_attention" {
        "\nThis run loaded validated multi-layer model artifacts and executed part of prefix-layer attention through MLX Metal dispatch, but some multilayer prefix work still fell back to CPU reference. Read it as partial MLX dense-forward progress, not as full MLX prefix coverage or final inference evidence.\n"
    } else if execution_semantics == "metal_multilayer_native_prefix_attention" {
        "\nThis run loaded validated multi-layer model artifacts and already executed at least part of prefix-layer attention through MLX Metal dispatch, but it still did not report a complete dense real-model forward. Read it as a stronger MLX Metal bring-up milestone than CPU-reference multilayer staging, not as final inference evidence.\n"
    } else if execution_semantics == "metal_multilayer_model_incomplete" {
        "\nThis run loaded validated multi-layer model artifacts into the Metal path, but it did not report a complete real model forward. Read it as partial MLX Metal bring-up progress, not final dense inference evidence.\n"
    } else if execution_semantics == "metal_model_bound_ffn_decode" {
        "\nThis run executed Metal dispatch with real model tensor inputs and reported model-bound FFN direct decode continuation. It is a stronger MLX Metal bring-up milestone than tensor-input-only evidence, but it still does not prove a full release-grade dense forward path.\n"
    } else if execution_semantics == "metal_real_model_tensor_inputs" {
        "\nThis run executed Metal dispatch with real model tensor inputs present, but it did not mark a real model forward pass. Treat it as model-bound bring-up evidence rather than final inference performance.\n"
    } else if execution_semantics == "metal_model_conditioned_numeric_scaffold" {
        "\nThis run executed Metal dispatch with model-conditioned staged inputs, but it still remains a numeric scaffold path rather than a full model forward. Do not compare it directly with release-grade inference results.\n"
    } else if execution_semantics == "metal_numeric_scaffold_only" {
        "\nThis run submitted real Metal command buffers, but the data path remained numeric scaffold only. It validates kernel plumbing and runtime bring-up, not model-bound inference quality or throughput.\n"
    } else {
        "\nThis run executed the current engine bring-up path through the preview SDK session, request manager, scheduler, KV manager, deterministic runner, and deterministic sampler.\n"
    };

    format!(
        "# Benchmark Run\n\n- run_id: `{run_id}`\n- command: `ax-engine-bench {command}`\n- manifest: `{}`\n- status: `{}`\n- tool_mode: `{}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{backend_adapter}`\n- llama_cpp_preset: `{llama_cpp_preset}`\n- fallback_reason: `{fallback_reason}`\n- execution_semantics: `{execution_semantics}`\n- mlx_metal_readiness: `{mlx_metal_readiness_status}`\n- mlx_metal_hot_path_cpu_fallback_free: `{mlx_metal_hot_path_cpu_fallback_free}`\n- mlx_metal_batched_direct_decode_logits_ready: `{mlx_metal_batched_direct_decode_logits_ready}`\n- mlx_metal_prefix_min_dispatch_share: `{mlx_metal_prefix_min_dispatch_share}`\n- mlx_metal_direct_decode_min_dispatch_share: `{mlx_metal_direct_decode_min_dispatch_share}`\n- mlx_metal_readiness_blockers: `{mlx_metal_readiness_blockers}`\n- metal_numeric_scaffold_only: `{metal_numeric_scaffold_only}`\n- metal_complete_model_forward_supported: `{metal_complete_model_forward_supported}`\n- metal_model_conditioned_inputs: `{metal_model_conditioned_inputs}`\n- mlx_metal_prefix_layers_attention: `{mlx_metal_prefix_layers_attention}`\n- metal_prefix_layers_cpu_reference: `{metal_prefix_layers_cpu_reference}`\n- mlx_metal_prefix_dispatch_count: `{mlx_metal_prefix_dispatch_count}`\n- metal_prefix_cpu_reference_dispatch_count: `{metal_prefix_cpu_reference_dispatch_count}`\n- mlx_metal_prefix_projection_row_count: `{mlx_metal_prefix_projection_row_count}`\n- metal_prefix_cpu_projection_row_count: `{metal_prefix_cpu_projection_row_count}`\n- metal_prefix_projection_mlx_metal_dispatch_share: `{prefix_projection_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_rms_norm_element_count: `{mlx_metal_prefix_rms_norm_element_count}`\n- metal_prefix_cpu_rms_norm_element_count: `{metal_prefix_cpu_rms_norm_element_count}`\n- metal_prefix_rms_norm_mlx_metal_dispatch_share: `{prefix_rms_norm_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_ffn_activation_element_count: `{mlx_metal_prefix_ffn_activation_element_count}`\n- metal_prefix_cpu_ffn_activation_element_count: `{metal_prefix_cpu_ffn_activation_element_count}`\n- metal_prefix_ffn_activation_mlx_metal_dispatch_share: `{prefix_ffn_activation_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_residual_add_element_count: `{mlx_metal_prefix_residual_add_element_count}`\n- metal_prefix_cpu_residual_add_element_count: `{metal_prefix_cpu_residual_add_element_count}`\n- metal_prefix_residual_add_mlx_metal_dispatch_share: `{prefix_residual_add_mlx_metal_dispatch_share}`\n- mlx_metal_prefix_scale_element_count: `{mlx_metal_prefix_scale_element_count}`\n- metal_prefix_cpu_scale_element_count: `{metal_prefix_cpu_scale_element_count}`\n- metal_prefix_scale_mlx_metal_dispatch_share: `{prefix_scale_mlx_metal_dispatch_share}`\n- metal_direct_decode_tokens: `{metal_direct_decode_tokens}`\n- metal_direct_decode_batching_opportunity_observed: `{metal_direct_decode_batching_opportunity_observed}`\n- metal_direct_decode_model_bound_ffn: `{metal_direct_decode_model_bound_ffn}`\n- metal_direct_decode_checksum_lo: `{metal_direct_decode_checksum_lo}`\n- metal_direct_decode_batched_logits_group_count: `{metal_direct_decode_batched_logits_group_count}`\n- metal_direct_decode_batched_logits_token_count: `{metal_direct_decode_batched_logits_token_count}`\n- metal_direct_decode_batched_group_fallback_count: `{metal_direct_decode_batched_group_fallback_count}`\n- metal_direct_decode_batched_group_fallback_token_count: `{metal_direct_decode_batched_group_fallback_token_count}`\n- metal_real_model_tensor_inputs: `{metal_real_model_tensor_inputs}`\n- metal_real_model_forward: `{metal_real_model_forward}`\n- metal_model_artifacts_validated: `{metal_model_artifacts_validated}`\n- mlx_metal_projection_f32_binding_count: `{mlx_metal_projection_f32_binding_count}`\n- mlx_metal_projection_f16_binding_count: `{mlx_metal_projection_f16_binding_count}`\n- mlx_metal_projection_bf16_binding_count: `{mlx_metal_projection_bf16_binding_count}`\n- mlx_metal_projection_unsupported_binding_count: `{mlx_metal_projection_unsupported_binding_count}`\n- mlx_metal_projection_source_quantized_binding_count: `{mlx_metal_projection_source_quantized_binding_count}`\n- mlx_metal_rms_norm_f32_binding_count: `{mlx_metal_rms_norm_f32_binding_count}`\n- mlx_metal_rms_norm_f16_binding_count: `{mlx_metal_rms_norm_f16_binding_count}`\n- mlx_metal_rms_norm_bf16_binding_count: `{mlx_metal_rms_norm_bf16_binding_count}`\n- mlx_metal_rms_norm_unsupported_binding_count: `{mlx_metal_rms_norm_unsupported_binding_count}`\n- mlx_metal_rms_norm_source_quantized_binding_count: `{mlx_metal_rms_norm_source_quantized_binding_count}`\n- mlx_metal_direct_decode_projection_row_count: `{mlx_metal_direct_decode_projection_row_count}`\n- metal_direct_decode_cpu_projection_row_count: `{metal_direct_decode_cpu_projection_row_count}`\n- metal_direct_decode_projection_mlx_metal_dispatch_share: `{direct_decode_projection_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_rms_norm_element_count: `{mlx_metal_direct_decode_rms_norm_element_count}`\n- metal_direct_decode_cpu_rms_norm_element_count: `{metal_direct_decode_cpu_rms_norm_element_count}`\n- metal_direct_decode_rms_norm_mlx_metal_dispatch_share: `{direct_decode_rms_norm_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_ffn_activation_element_count: `{mlx_metal_direct_decode_ffn_activation_element_count}`\n- metal_direct_decode_cpu_ffn_activation_element_count: `{metal_direct_decode_cpu_ffn_activation_element_count}`\n- metal_direct_decode_ffn_activation_mlx_metal_dispatch_share: `{direct_decode_ffn_activation_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_residual_add_element_count: `{mlx_metal_direct_decode_residual_add_element_count}`\n- metal_direct_decode_cpu_residual_add_element_count: `{metal_direct_decode_cpu_residual_add_element_count}`\n- metal_direct_decode_residual_add_mlx_metal_dispatch_share: `{direct_decode_residual_add_mlx_metal_dispatch_share}`\n- mlx_metal_direct_decode_scale_element_count: `{mlx_metal_direct_decode_scale_element_count}`\n- metal_direct_decode_cpu_scale_element_count: `{metal_direct_decode_cpu_scale_element_count}`\n- metal_direct_decode_scale_mlx_metal_dispatch_share: `{direct_decode_scale_mlx_metal_dispatch_share}`\n- correctness: `{}`\n- determinism: `{}`\n- ttft_ms: `{:.2}`\n- prefill_tok_s: `{:.2}`\n- decode_tok_s: `{:.2}`\n- e2e_latency_ms: `{:.2}`\n- cpu_time_per_token_us: `{:.2}`\n- runner_time_per_token_us: `{:.2}`\n- runner_time_share_pct: `{:.2}`\n- prefix_hit_rate: `{:.2}`\n- prefix_reuse_provenance: `{prefix_reuse_provenance}`\n- backend_reported_cached_prompt_tokens: `{backend_reported_cached_prompt_tokens}`\n- llama_cpp_processing_request_events: `{llama_cpp_processing_request_events}`\n- llama_cpp_deferred_request_events: `{llama_cpp_deferred_request_events}`\n- kv_peak_blocks: `{}`\n{}",
        manifest_path.display(),
        execution.status_label(),
        execution.tool_mode,
        correctness,
        determinism,
        execution.observation.ttft_ms.unwrap_or_default() as f64,
        execution.observation.prefill_tok_s(),
        execution.observation.decode_tok_s(),
        execution.observation.e2e_latency_ms as f64,
        execution.observation.cpu_time_per_token_us(),
        execution.observation.runner_time_per_token_us(),
        execution.observation.runner_time_share_pct(),
        execution.observation.prefix_hit_rate(),
        execution.observation.kv_peak_blocks,
        summary_note
    )
}

fn build_contract_failure_json(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    output_root: &Path,
    started_at_unix_s: u64,
    message: &str,
) -> Result<Value, CliError> {
    let completed_at_unix_s = unix_timestamp_secs().unwrap_or(started_at_unix_s);
    let cwd = env::current_dir().map_err(|error| {
        CliError::Runtime(format!(
            "failed to resolve current working directory for contract failure artifact: {error}"
        ))
    })?;
    let failure = classify_contract_failure(manifest, message);

    Ok(json!({
        "schema_version": "ax.engine_bench.contract_failure.v1",
        "run_id": run_id,
        "status": "contract_failure",
        "command": command,
        "tool_mode": contract_failure_tool_mode(manifest),
        "manifest": {
            "id": manifest.id,
            "class": manifest.class,
            "scenario": manifest.scenario
        },
        "runtime": serialize_contract_failure_runtime(manifest),
        "failure": {
            "class": "contract",
            "code": failure.code,
            "message": message,
            "recommended_action": failure.recommended_action
        },
        "benchmark": {
            "started_at_unix_s": started_at_unix_s,
            "completed_at_unix_s": completed_at_unix_s,
            "manifest_path": manifest_path.display().to_string(),
            "output_root": output_root.display().to_string(),
            "cwd": cwd.display().to_string()
        }
    }))
}

fn build_contract_failure_summary_markdown(
    run_id: &str,
    command: &str,
    manifest_path: &Path,
    manifest: &BenchmarkManifest,
    message: &str,
) -> String {
    let failure = classify_contract_failure(manifest, message);
    let runtime = serialize_contract_failure_runtime(manifest);
    let selected_backend = nested_value(&runtime, &["selected_backend"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = nested_value(&runtime, &["support_tier"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = nested_value(&runtime, &["resolution_policy"])
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let backend_adapter = nested_value(&runtime, &["backend_adapter"])
        .cloned()
        .unwrap_or(Value::String("none".to_string()));
    format!(
        "# Benchmark Contract Failure\n\n- run_id: `{run_id}`\n- command: `ax-engine-bench {command}`\n- manifest: `{}`\n- status: `contract_failure`\n- code: `{}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{}`\n- scenario: `{}`\n\nFailure reason:\n\n{}\n\nRecommended action:\n\n{}\n",
        manifest_path.display(),
        failure.code,
        json_value_label(&backend_adapter),
        manifest.scenario,
        message,
        failure.recommended_action
    )
}

fn serialize_contract_failure_runtime(manifest: &BenchmarkManifest) -> Value {
    match runtime_config_from_manifest(manifest) {
        Ok(runtime) => serialize_runtime_metadata(&runtime, None),
        Err(_) => serde_json::to_value(&manifest.runtime)
            .expect("runtime manifest should serialize for contract failure artifact"),
    }
}

struct ContractFailureClassification {
    code: &'static str,
    recommended_action: &'static str,
}

fn classify_contract_failure(
    manifest: &BenchmarkManifest,
    message: &str,
) -> ContractFailureClassification {
    if message.contains("runtime.backend_adapter is required") {
        return ContractFailureClassification {
            code: "llama_backend_adapter_required",
            recommended_action: "Set runtime.backend_adapter to a supported llama.cpp adapter before running delegated benchmark manifests.",
        };
    }

    if message.contains("runtime.backend_adapter kind does not match selected_backend") {
        return ContractFailureClassification {
            code: "backend_adapter_kind_mismatch",
            recommended_action: "Align runtime.selected_backend and runtime.backend_adapter.kind so they describe the same delegated backend.",
        };
    }

    if message.contains("runtime.backend_adapter must be omitted when selected_backend is mlx") {
        return ContractFailureClassification {
            code: "mlx_backend_adapter_forbidden",
            recommended_action: "Remove runtime.backend_adapter when selected_backend is mlx.",
        };
    }

    if message.contains("invalid benchmark runtime backend-resolution contract") {
        return ContractFailureClassification {
            code: "runtime_backend_resolution_contract_invalid",
            recommended_action: "Fix selected_backend, support_tier, resolution_policy, and fallback_reason so they satisfy the benchmark runtime contract.",
        };
    }

    if manifest.runtime.support_tier == SupportTier::LlamaCpp
        && matches!(
            manifest.runtime.backend_adapter,
            Some(BackendAdapterManifest::LlamaCppCli { .. })
        )
    {
        return ContractFailureClassification {
            code: "llama_cpp_cli_token_workload_unsupported",
            recommended_action: "Use runtime.backend_adapter.kind=llama_cpp_server_completion for token-based llama.cpp benchmark manifests.",
        };
    }

    if message
        .contains("blocking llama.cpp benchmark adapters currently support scenario manifests only")
    {
        return ContractFailureClassification {
            code: "llama_cpp_blocking_replay_unsupported",
            recommended_action: "Use the llama.cpp server adapter for non-MLX llama.cpp workloads, or switch to an MLX-mode manifest for ax-engine MLX inference.",
        };
    }

    if message
        .contains("blocking llama.cpp benchmark adapters currently require shape.concurrency=1")
    {
        return ContractFailureClassification {
            code: "llama_cpp_blocking_concurrency_unsupported",
            recommended_action: "Reduce shape.concurrency to 1 for blocking llama.cpp benchmark adapters, or use the stepwise llama.cpp server adapter for multi-request delegated workloads.",
        };
    }

    if message.contains("blocking llama.cpp benchmark adapters do not support require_prefix_reuse")
    {
        return ContractFailureClassification {
            code: "llama_cpp_blocking_prefix_reuse_unsupported",
            recommended_action: "Disable require_prefix_reuse for blocking delegated adapters, or use the stepwise llama.cpp server adapter for delegated prompt-cache benchmark coverage.",
        };
    }

    if message.contains("scenario manifest must contain a shape object") {
        return ContractFailureClassification {
            code: "scenario_shape_required",
            recommended_action: "Add a valid shape object to the scenario manifest before rerunning ax-engine-bench scenario.",
        };
    }

    if message.contains("submit event missing request_id")
        || message.contains("cancel event missing request_id")
        || message.contains("submit event missing prompt_ref")
    {
        return ContractFailureClassification {
            code: "replay_event_schema_invalid",
            recommended_action: "Repair the replay events so submit/cancel entries include the required request identifiers and prompt references.",
        };
    }

    let _ = message;
    ContractFailureClassification {
        code: "contract_validation_failed",
        recommended_action: "Review the manifest/runtime contract and rerun with a supported workload shape or runtime configuration.",
    }
}

fn contract_failure_tool_mode(manifest: &BenchmarkManifest) -> &'static str {
    if let Ok(runtime) = runtime_config_from_manifest(manifest) {
        runtime.tool_mode()
    } else if matches!(
        manifest.runtime.backend_adapter.as_ref(),
        Some(adapter) if adapter.supports_stepwise_benchmark()
    ) {
        "llama_cpp_stepwise_runtime"
    } else {
        "llama_cpp_blocking_runtime"
    }
}

fn build_regression_json(
    baseline_metrics: &Value,
    candidate_metrics: &Value,
    baseline_environment: &Value,
    candidate_environment: &Value,
    trusted_baseline: Option<&Value>,
) -> Result<Value, CliError> {
    let baseline_run_id = baseline_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("baseline");
    let candidate_run_id = candidate_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("candidate");
    let baseline_prefix_cache_path =
        nested_value(baseline_environment, &["route", "prefix_cache_path"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let candidate_prefix_cache_path =
        nested_value(candidate_environment, &["route", "prefix_cache_path"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let baseline_prefix_cache_evidence =
        nested_value(baseline_environment, &["route", "prefix_cache_evidence"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let candidate_prefix_cache_evidence =
        nested_value(candidate_environment, &["route", "prefix_cache_evidence"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let baseline_prefix_reuse_provenance =
        nested_value(baseline_environment, &["route", "prefix_reuse_provenance"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let candidate_prefix_reuse_provenance =
        nested_value(candidate_environment, &["route", "prefix_reuse_provenance"])
            .and_then(Value::as_str)
            .unwrap_or("unknown");
    let baseline_route = baseline_environment
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let candidate_route = candidate_environment
        .get("route")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let baseline_execution_semantics =
        route_execution_semantics_from_environment_json(baseline_environment);
    let candidate_execution_semantics =
        route_execution_semantics_from_environment_json(candidate_environment);
    let baseline_metal_numeric_scaffold_only = route_flag_from_json(
        &baseline_route,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let candidate_metal_numeric_scaffold_only = route_flag_from_json(
        &candidate_route,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let baseline_metal_real_model_forward = route_flag_from_json(
        &baseline_route,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let candidate_metal_real_model_forward = route_flag_from_json(
        &candidate_route,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let baseline_metal_complete_model_forward_supported = route_flag_from_json(
        &baseline_route,
        "metal_complete_model_forward_supported",
        &[
            "metal_dispatch_complete_model_forward_supported",
            "metal_dispatch_runtime_complete_model_forward_supported",
        ],
    );
    let candidate_metal_complete_model_forward_supported = route_flag_from_json(
        &candidate_route,
        "metal_complete_model_forward_supported",
        &[
            "metal_dispatch_complete_model_forward_supported",
            "metal_dispatch_runtime_complete_model_forward_supported",
        ],
    );
    let baseline_metal_model_conditioned_inputs = route_flag_from_json(
        &baseline_route,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let candidate_metal_model_conditioned_inputs = route_flag_from_json(
        &candidate_route,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let baseline_mlx_metal_prefix_layers_attention = route_flag_from_json(
        &baseline_route,
        "mlx_metal_prefix_layers_attention",
        &["metal_dispatch_prefix_layers_native_attention"],
    );
    let candidate_mlx_metal_prefix_layers_attention = route_flag_from_json(
        &candidate_route,
        "mlx_metal_prefix_layers_attention",
        &["metal_dispatch_prefix_layers_native_attention"],
    );
    let baseline_metal_prefix_layers_cpu_reference = route_flag_from_json(
        &baseline_route,
        "metal_prefix_layers_cpu_reference",
        &["metal_dispatch_prefix_layers_cpu_reference"],
    );
    let candidate_metal_prefix_layers_cpu_reference = route_flag_from_json(
        &candidate_route,
        "metal_prefix_layers_cpu_reference",
        &["metal_dispatch_prefix_layers_cpu_reference"],
    );
    let baseline_mlx_metal_prefix_dispatch_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let candidate_mlx_metal_prefix_dispatch_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let baseline_metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        &baseline_route,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let candidate_metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        &candidate_route,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let baseline_mlx_metal_projection_f32_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_f32_binding_count",
        &["metal_dispatch_native_projection_f32_binding_count"],
    );
    let candidate_mlx_metal_projection_f32_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_f32_binding_count",
        &["metal_dispatch_native_projection_f32_binding_count"],
    );
    let baseline_mlx_metal_projection_f16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_f16_binding_count",
        &["metal_dispatch_native_projection_f16_binding_count"],
    );
    let candidate_mlx_metal_projection_f16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_f16_binding_count",
        &["metal_dispatch_native_projection_f16_binding_count"],
    );
    let baseline_mlx_metal_projection_bf16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_bf16_binding_count",
        &["metal_dispatch_native_projection_bf16_binding_count"],
    );
    let candidate_mlx_metal_projection_bf16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_bf16_binding_count",
        &["metal_dispatch_native_projection_bf16_binding_count"],
    );
    let baseline_mlx_metal_projection_unsupported_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_projection_unsupported_binding_count",
        &["metal_dispatch_native_projection_unsupported_binding_count"],
    );
    let candidate_mlx_metal_projection_unsupported_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_projection_unsupported_binding_count",
        &["metal_dispatch_native_projection_unsupported_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_f32_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_f32_binding_count",
        &["metal_dispatch_native_rms_norm_f32_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_f32_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_f32_binding_count",
        &["metal_dispatch_native_rms_norm_f32_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_f16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_f16_binding_count",
        &["metal_dispatch_native_rms_norm_f16_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_f16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_f16_binding_count",
        &["metal_dispatch_native_rms_norm_f16_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_bf16_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_bf16_binding_count",
        &["metal_dispatch_native_rms_norm_bf16_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_bf16_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_bf16_binding_count",
        &["metal_dispatch_native_rms_norm_bf16_binding_count"],
    );
    let baseline_mlx_metal_rms_norm_unsupported_binding_count = route_count_from_json(
        &baseline_route,
        "mlx_metal_rms_norm_unsupported_binding_count",
        &["metal_dispatch_native_rms_norm_unsupported_binding_count"],
    );
    let candidate_mlx_metal_rms_norm_unsupported_binding_count = route_count_from_json(
        &candidate_route,
        "mlx_metal_rms_norm_unsupported_binding_count",
        &["metal_dispatch_native_rms_norm_unsupported_binding_count"],
    );
    let baseline_metal_direct_decode_tokens = route_flag_from_json(
        &baseline_route,
        "metal_direct_decode_tokens",
        &["metal_dispatch_direct_decode_tokens"],
    );
    let candidate_metal_direct_decode_tokens = route_flag_from_json(
        &candidate_route,
        "metal_direct_decode_tokens",
        &["metal_dispatch_direct_decode_tokens"],
    );
    let baseline_metal_direct_decode_batching_opportunity_observed = route_flag_from_json(
        &baseline_route,
        "metal_direct_decode_batching_opportunity_observed",
        &["metal_direct_decode_batching_opportunity_observed"],
    );
    let candidate_metal_direct_decode_batching_opportunity_observed = route_flag_from_json(
        &candidate_route,
        "metal_direct_decode_batching_opportunity_observed",
        &["metal_direct_decode_batching_opportunity_observed"],
    );
    let baseline_metal_direct_decode_model_bound_ffn = route_flag_from_json(
        &baseline_route,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let candidate_metal_direct_decode_model_bound_ffn = route_flag_from_json(
        &candidate_route,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let baseline_metal_direct_decode_batched_logits_group_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_logits_group_count",
        &["metal_dispatch_direct_decode_batched_logits_group_count"],
    );
    let candidate_metal_direct_decode_batched_logits_group_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_logits_group_count",
        &["metal_dispatch_direct_decode_batched_logits_group_count"],
    );
    let baseline_metal_direct_decode_batched_logits_token_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_logits_token_count",
        &["metal_dispatch_direct_decode_batched_logits_token_count"],
    );
    let candidate_metal_direct_decode_batched_logits_token_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_logits_token_count",
        &["metal_dispatch_direct_decode_batched_logits_token_count"],
    );
    let baseline_metal_direct_decode_batched_group_fallback_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_group_fallback_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_count"],
    );
    let candidate_metal_direct_decode_batched_group_fallback_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_group_fallback_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_count"],
    );
    let baseline_metal_direct_decode_batched_group_fallback_token_count = route_count_from_json(
        &baseline_route,
        "metal_direct_decode_batched_group_fallback_token_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_token_count"],
    );
    let candidate_metal_direct_decode_batched_group_fallback_token_count = route_count_from_json(
        &candidate_route,
        "metal_direct_decode_batched_group_fallback_token_count",
        &["metal_dispatch_direct_decode_batched_group_fallback_token_count"],
    );
    let baseline_metal_real_model_tensor_inputs = route_flag_from_json(
        &baseline_route,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let candidate_metal_real_model_tensor_inputs = route_flag_from_json(
        &candidate_route,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let baseline_metal_model_artifacts_validated = route_flag_from_json(
        &baseline_route,
        "metal_model_artifacts_validated",
        &["metal_dispatch_model_artifacts_validated"],
    );
    let candidate_metal_model_artifacts_validated = route_flag_from_json(
        &candidate_route,
        "metal_model_artifacts_validated",
        &["metal_dispatch_model_artifacts_validated"],
    );
    let baseline_backend_reported_cached_prompt_tokens = nested_value(
        baseline_environment,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .cloned();
    let candidate_backend_reported_cached_prompt_tokens = nested_value(
        candidate_environment,
        &["route", "backend_reported_cached_prompt_tokens"],
    )
    .cloned();
    let baseline_tool_mode =
        string_at_path_or_unknown(baseline_environment, &["software", "tool_mode"]);
    let candidate_tool_mode =
        string_at_path_or_unknown(candidate_environment, &["software", "tool_mode"]);
    let baseline_selected_backend =
        string_at_path_or_unknown(baseline_environment, &["runtime", "selected_backend"]);
    let candidate_selected_backend =
        string_at_path_or_unknown(candidate_environment, &["runtime", "selected_backend"]);
    let baseline_support_tier =
        string_at_path_or_unknown(baseline_environment, &["runtime", "support_tier"]);
    let candidate_support_tier =
        string_at_path_or_unknown(candidate_environment, &["runtime", "support_tier"]);
    let baseline_resolution_policy =
        string_at_path_or_unknown(baseline_environment, &["runtime", "resolution_policy"]);
    let candidate_resolution_policy =
        string_at_path_or_unknown(candidate_environment, &["runtime", "resolution_policy"]);
    let baseline_backend_adapter =
        nested_value(baseline_environment, &["runtime", "backend_adapter"]).cloned();
    let candidate_backend_adapter =
        nested_value(candidate_environment, &["runtime", "backend_adapter"]).cloned();
    let baseline_llama_cpp_preset =
        nested_value(baseline_environment, &["runtime", "llama_cpp_preset"]).cloned();
    let candidate_llama_cpp_preset =
        nested_value(candidate_environment, &["runtime", "llama_cpp_preset"]).cloned();
    let baseline_mlx_dense_dequantized_source = nested_value(baseline_environment, &["runtime"])
        .is_some_and(native_dense_dequantized_source_from_runtime_json);
    let candidate_mlx_dense_dequantized_source = nested_value(candidate_environment, &["runtime"])
        .is_some_and(native_dense_dequantized_source_from_runtime_json);
    let baseline_mlx_metal_readiness = mlx_metal_readiness_from_route_json(
        &baseline_tool_mode,
        &baseline_selected_backend,
        &baseline_route,
        baseline_mlx_dense_dequantized_source,
    );
    let candidate_mlx_metal_readiness = mlx_metal_readiness_from_route_json(
        &candidate_tool_mode,
        &candidate_selected_backend,
        &candidate_route,
        candidate_mlx_dense_dequantized_source,
    );
    let compare_result = compare_result_label(&baseline_tool_mode, &baseline_execution_semantics);
    let mut runtime = json!({
        "tool_mode": baseline_tool_mode,
        "selected_backend": baseline_selected_backend,
        "support_tier": baseline_support_tier,
        "resolution_policy": baseline_resolution_policy
    });
    if let Some(backend_adapter) = baseline_backend_adapter.clone() {
        runtime
            .as_object_mut()
            .expect("regression runtime should serialize as object")
            .insert("backend_adapter".to_string(), backend_adapter);
    }
    if let Some(llama_cpp_preset) = baseline_llama_cpp_preset.clone() {
        runtime
            .as_object_mut()
            .expect("regression runtime should serialize as object")
            .insert("llama_cpp_preset".to_string(), llama_cpp_preset);
    }
    let mut contract_runtime = json!({
        "selected_backend": {
            "baseline": baseline_selected_backend,
            "candidate": candidate_selected_backend
        },
        "support_tier": {
            "baseline": baseline_support_tier,
            "candidate": candidate_support_tier
        },
        "resolution_policy": {
            "baseline": baseline_resolution_policy,
            "candidate": candidate_resolution_policy
        }
    });
    if baseline_backend_adapter.is_some() || candidate_backend_adapter.is_some() {
        contract_runtime
            .as_object_mut()
            .expect("regression contract runtime should serialize as object")
            .insert(
                "backend_adapter".to_string(),
                json!({
                    "baseline": baseline_backend_adapter,
                    "candidate": candidate_backend_adapter
                }),
            );
    }
    if baseline_llama_cpp_preset.is_some() || candidate_llama_cpp_preset.is_some() {
        contract_runtime
            .as_object_mut()
            .expect("regression contract runtime should serialize as object")
            .insert(
                "llama_cpp_preset".to_string(),
                json!({
                    "baseline": baseline_llama_cpp_preset,
                    "candidate": candidate_llama_cpp_preset
                }),
            );
    }

    let mut regression = json!({
        "schema_version": "ax.engine_bench.regression.v1",
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "runtime": runtime,
        "comparison": {
            "ttft_ms_pct": percentage_delta(metric_number(baseline_metrics, "ttft_ms")?, metric_number(candidate_metrics, "ttft_ms")?),
            "decode_tok_s_pct": percentage_delta(metric_number(baseline_metrics, "decode_tok_s")?, metric_number(candidate_metrics, "decode_tok_s")?),
            "memory_peak_mb_pct": percentage_delta(metric_number(baseline_metrics, "memory_peak_mb")?, metric_number(candidate_metrics, "memory_peak_mb")?),
            "prefix_hit_rate_pct": percentage_delta(metric_number(baseline_metrics, "prefix_hit_rate")?, metric_number(candidate_metrics, "prefix_hit_rate")?)
        },
        "gates": {
            "correctness": "inspect",
            "determinism": "inspect",
            "benchmark_contract": "pass",
            "performance": "inspect"
        },
        "summary": {
            "result": compare_result,
            "requires_human_review": true,
            "tool_mode": baseline_tool_mode,
            "selected_backend": baseline_selected_backend,
            "support_tier": baseline_support_tier,
            "resolution_policy": baseline_resolution_policy,
            "execution_semantics": baseline_execution_semantics,
            "metal_numeric_scaffold_only": baseline_metal_numeric_scaffold_only,
            "metal_complete_model_forward_supported": baseline_metal_complete_model_forward_supported,
            "metal_real_model_forward": baseline_metal_real_model_forward,
            "metal_model_conditioned_inputs": baseline_metal_model_conditioned_inputs,
            "mlx_metal_prefix_layers_attention": baseline_mlx_metal_prefix_layers_attention,
            "metal_prefix_layers_cpu_reference": baseline_metal_prefix_layers_cpu_reference,
            "mlx_metal_prefix_dispatch_count": baseline_mlx_metal_prefix_dispatch_count,
            "metal_prefix_cpu_reference_dispatch_count": baseline_metal_prefix_cpu_reference_dispatch_count,
            "mlx_metal_projection_f32_binding_count": baseline_mlx_metal_projection_f32_binding_count,
            "mlx_metal_projection_f16_binding_count": baseline_mlx_metal_projection_f16_binding_count,
            "mlx_metal_projection_bf16_binding_count": baseline_mlx_metal_projection_bf16_binding_count,
            "mlx_metal_projection_unsupported_binding_count": baseline_mlx_metal_projection_unsupported_binding_count,
            "metal_direct_decode_tokens": baseline_metal_direct_decode_tokens,
            "metal_direct_decode_batching_opportunity_observed": baseline_metal_direct_decode_batching_opportunity_observed,
            "metal_direct_decode_model_bound_ffn": baseline_metal_direct_decode_model_bound_ffn,
            "metal_direct_decode_batched_logits_group_count": baseline_metal_direct_decode_batched_logits_group_count,
            "metal_direct_decode_batched_logits_token_count": baseline_metal_direct_decode_batched_logits_token_count,
            "metal_direct_decode_batched_group_fallback_count": baseline_metal_direct_decode_batched_group_fallback_count,
            "metal_direct_decode_batched_group_fallback_token_count": baseline_metal_direct_decode_batched_group_fallback_token_count,
            "metal_real_model_tensor_inputs": baseline_metal_real_model_tensor_inputs,
            "metal_model_artifacts_validated": baseline_metal_model_artifacts_validated,
            "mlx_metal_rms_norm_f32_binding_count": baseline_mlx_metal_rms_norm_f32_binding_count,
            "mlx_metal_rms_norm_f16_binding_count": baseline_mlx_metal_rms_norm_f16_binding_count,
            "mlx_metal_rms_norm_bf16_binding_count": baseline_mlx_metal_rms_norm_bf16_binding_count,
            "mlx_metal_rms_norm_unsupported_binding_count": baseline_mlx_metal_rms_norm_unsupported_binding_count,
            "mlx_metal_readiness": baseline_mlx_metal_readiness.status,
            "mlx_metal_hot_path_cpu_fallback_free": baseline_mlx_metal_readiness.hot_path_cpu_fallback_free,
            "mlx_metal_batched_direct_decode_logits_ready": baseline_mlx_metal_readiness.batched_direct_decode_logits_ready,
            "mlx_metal_prefix_min_dispatch_share": baseline_mlx_metal_readiness.prefix_min_mlx_metal_dispatch_share,
            "mlx_metal_direct_decode_min_dispatch_share": baseline_mlx_metal_readiness.direct_decode_min_mlx_metal_dispatch_share,
            "mlx_metal_readiness_blockers": baseline_mlx_metal_readiness.blockers,
            "prefix_cache_path": baseline_prefix_cache_path,
            "prefix_cache_evidence": baseline_prefix_cache_evidence,
            "prefix_reuse_provenance": baseline_prefix_reuse_provenance
        },
        "contract": {
            "tool_mode": {
                "baseline": baseline_tool_mode,
                "candidate": candidate_tool_mode
            },
            "runtime": contract_runtime,
            "execution_semantics": {
                "baseline": baseline_execution_semantics,
                "candidate": candidate_execution_semantics
            },
            "metal_numeric_scaffold_only": {
                "baseline": baseline_metal_numeric_scaffold_only,
                "candidate": candidate_metal_numeric_scaffold_only
            },
            "metal_complete_model_forward_supported": {
                "baseline": baseline_metal_complete_model_forward_supported,
                "candidate": candidate_metal_complete_model_forward_supported
            },
            "metal_real_model_forward": {
                "baseline": baseline_metal_real_model_forward,
                "candidate": candidate_metal_real_model_forward
            },
            "metal_model_conditioned_inputs": {
                "baseline": baseline_metal_model_conditioned_inputs,
                "candidate": candidate_metal_model_conditioned_inputs
            },
            "mlx_metal_prefix_layers_attention": {
                "baseline": baseline_mlx_metal_prefix_layers_attention,
                "candidate": candidate_mlx_metal_prefix_layers_attention
            },
            "metal_prefix_layers_cpu_reference": {
                "baseline": baseline_metal_prefix_layers_cpu_reference,
                "candidate": candidate_metal_prefix_layers_cpu_reference
            },
            "mlx_metal_prefix_dispatch_count": {
                "baseline": baseline_mlx_metal_prefix_dispatch_count,
                "candidate": candidate_mlx_metal_prefix_dispatch_count
            },
            "metal_prefix_cpu_reference_dispatch_count": {
                "baseline": baseline_metal_prefix_cpu_reference_dispatch_count,
                "candidate": candidate_metal_prefix_cpu_reference_dispatch_count
            },
            "mlx_metal_projection_f32_binding_count": {
                "baseline": baseline_mlx_metal_projection_f32_binding_count,
                "candidate": candidate_mlx_metal_projection_f32_binding_count
            },
            "mlx_metal_projection_f16_binding_count": {
                "baseline": baseline_mlx_metal_projection_f16_binding_count,
                "candidate": candidate_mlx_metal_projection_f16_binding_count
            },
            "mlx_metal_projection_bf16_binding_count": {
                "baseline": baseline_mlx_metal_projection_bf16_binding_count,
                "candidate": candidate_mlx_metal_projection_bf16_binding_count
            },
            "mlx_metal_projection_unsupported_binding_count": {
                "baseline": baseline_mlx_metal_projection_unsupported_binding_count,
                "candidate": candidate_mlx_metal_projection_unsupported_binding_count
            },
            "metal_direct_decode_tokens": {
                "baseline": baseline_metal_direct_decode_tokens,
                "candidate": candidate_metal_direct_decode_tokens
            },
            "metal_direct_decode_batching_opportunity_observed": {
                "baseline": baseline_metal_direct_decode_batching_opportunity_observed,
                "candidate": candidate_metal_direct_decode_batching_opportunity_observed
            },
            "metal_direct_decode_model_bound_ffn": {
                "baseline": baseline_metal_direct_decode_model_bound_ffn,
                "candidate": candidate_metal_direct_decode_model_bound_ffn
            },
            "metal_direct_decode_batched_logits_group_count": {
                "baseline": baseline_metal_direct_decode_batched_logits_group_count,
                "candidate": candidate_metal_direct_decode_batched_logits_group_count
            },
            "metal_direct_decode_batched_logits_token_count": {
                "baseline": baseline_metal_direct_decode_batched_logits_token_count,
                "candidate": candidate_metal_direct_decode_batched_logits_token_count
            },
            "metal_direct_decode_batched_group_fallback_count": {
                "baseline": baseline_metal_direct_decode_batched_group_fallback_count,
                "candidate": candidate_metal_direct_decode_batched_group_fallback_count
            },
            "metal_direct_decode_batched_group_fallback_token_count": {
                "baseline": baseline_metal_direct_decode_batched_group_fallback_token_count,
                "candidate": candidate_metal_direct_decode_batched_group_fallback_token_count
            },
            "metal_real_model_tensor_inputs": {
                "baseline": baseline_metal_real_model_tensor_inputs,
                "candidate": candidate_metal_real_model_tensor_inputs
            },
            "metal_model_artifacts_validated": {
                "baseline": baseline_metal_model_artifacts_validated,
                "candidate": candidate_metal_model_artifacts_validated
            },
            "mlx_metal_rms_norm_f32_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_f32_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_f32_binding_count
            },
            "mlx_metal_rms_norm_f16_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_f16_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_f16_binding_count
            },
            "mlx_metal_rms_norm_bf16_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_bf16_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_bf16_binding_count
            },
            "mlx_metal_rms_norm_unsupported_binding_count": {
                "baseline": baseline_mlx_metal_rms_norm_unsupported_binding_count,
                "candidate": candidate_mlx_metal_rms_norm_unsupported_binding_count
            },
            "prefix_cache_path": {
                "baseline": baseline_prefix_cache_path,
                "candidate": candidate_prefix_cache_path
            },
            "prefix_cache_evidence": {
                "baseline": baseline_prefix_cache_evidence,
                "candidate": candidate_prefix_cache_evidence
            },
            "prefix_reuse_provenance": {
                "baseline": baseline_prefix_reuse_provenance,
                "candidate": candidate_prefix_reuse_provenance
            }
        }
    });

    regression
        .as_object_mut()
        .expect("regression artifact should serialize as object")
        .insert(
            "readiness".to_string(),
            json!({
                "baseline": baseline_mlx_metal_readiness.to_json(),
                "candidate": candidate_mlx_metal_readiness.to_json()
            }),
        );

    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        let baseline_value = route_counter_from_json(&baseline_route, key);
        let candidate_value = route_counter_from_json(&candidate_route, key);
        regression
            .get_mut("summary")
            .and_then(Value::as_object_mut)
            .expect("regression summary should serialize as object")
            .insert(key.to_string(), json!(baseline_value));
        regression
            .get_mut("contract")
            .and_then(Value::as_object_mut)
            .expect("regression contract should serialize as object")
            .insert(
                key.to_string(),
                json!({
                    "baseline": baseline_value,
                    "candidate": candidate_value
                }),
            );
    }

    if let Some(value) = baseline_backend_reported_cached_prompt_tokens.clone() {
        regression
            .get_mut("summary")
            .and_then(Value::as_object_mut)
            .expect("regression summary should serialize as object")
            .insert("backend_reported_cached_prompt_tokens".to_string(), value);
    }

    if baseline_backend_reported_cached_prompt_tokens.is_some()
        || candidate_backend_reported_cached_prompt_tokens.is_some()
    {
        regression
            .get_mut("contract")
            .and_then(Value::as_object_mut)
            .expect("regression contract should serialize as object")
            .insert(
                "backend_reported_cached_prompt_tokens".to_string(),
                json!({
                    "baseline": baseline_backend_reported_cached_prompt_tokens,
                    "candidate": candidate_backend_reported_cached_prompt_tokens
                }),
            );
    }

    if let Some(trusted_baseline) = trusted_baseline {
        regression
            .as_object_mut()
            .expect("regression artifact should serialize as object")
            .insert(
                "trusted_baseline".to_string(),
                json!({
                    "name": trusted_baseline.get("name").cloned().unwrap_or(Value::String("unknown".to_string())),
                    "slug": trusted_baseline.get("slug").cloned().unwrap_or(Value::String("unknown".to_string())),
                    "created_at_unix_s": trusted_baseline
                        .get("created_at_unix_s")
                        .cloned()
                        .unwrap_or(Value::Null),
                    "source_result_dir": trusted_baseline
                        .get("source_result_dir")
                        .cloned()
                        .unwrap_or(Value::String("unknown".to_string())),
                    "source_run_id": trusted_baseline
                        .get("source_run_id")
                        .cloned()
                        .unwrap_or(Value::String("unknown".to_string()))
                }),
            );
    }

    Ok(regression)
}

fn build_compare_summary_markdown(
    baseline_metrics: &Value,
    candidate_metrics: &Value,
    regression_json: &Value,
    trusted_baseline: Option<&Value>,
) -> String {
    let baseline_id = baseline_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("baseline");
    let candidate_id = candidate_metrics
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or("candidate");
    let ttft = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("ttft_ms_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let decode = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("decode_tok_s_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let memory = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("memory_peak_mb_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let prefix = regression_json
        .get("comparison")
        .and_then(|comparison| comparison.get("prefix_hit_rate_pct"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let prefix_reuse_provenance = regression_json
        .get("summary")
        .and_then(|summary| summary.get("prefix_reuse_provenance"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_path = regression_json
        .get("summary")
        .and_then(|summary| summary.get("prefix_cache_path"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let prefix_cache_evidence = regression_json
        .get("summary")
        .and_then(|summary| summary.get("prefix_cache_evidence"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let execution_semantics = regression_json
        .get("summary")
        .and_then(|summary| summary.get("execution_semantics"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let metal_numeric_scaffold_only = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_numeric_scaffold_only"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_complete_model_forward_supported = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_complete_model_forward_supported"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_real_model_forward = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_real_model_forward"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_model_conditioned_inputs = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_model_conditioned_inputs"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mlx_metal_prefix_layers_attention = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_prefix_layers_attention"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_prefix_layers_cpu_reference = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_prefix_layers_cpu_reference"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mlx_metal_prefix_dispatch_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_prefix_dispatch_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_prefix_cpu_reference_dispatch_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_prefix_cpu_reference_dispatch_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_f32_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_f32_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_f16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_f16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_bf16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_bf16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_projection_unsupported_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_projection_unsupported_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_tokens = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_tokens"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_direct_decode_batching_opportunity_observed = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batching_opportunity_observed"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_direct_decode_model_bound_ffn = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_model_bound_ffn"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_direct_decode_batched_logits_group_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_logits_group_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_batched_logits_token_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_logits_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_group_fallback_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_direct_decode_batched_group_fallback_token_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_direct_decode_batched_group_fallback_token_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let metal_real_model_tensor_inputs = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_real_model_tensor_inputs"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let metal_model_artifacts_validated = regression_json
        .get("summary")
        .and_then(|summary| summary.get("metal_model_artifacts_validated"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mlx_metal_rms_norm_f32_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_f32_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_rms_norm_f16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_f16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_rms_norm_bf16_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_bf16_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mlx_metal_rms_norm_unsupported_binding_count = regression_json
        .get("summary")
        .and_then(|summary| summary.get("mlx_metal_rms_norm_unsupported_binding_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let baseline_mlx_metal_readiness = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let baseline_mlx_metal_hot_path_cpu_fallback_free = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("hot_path_cpu_fallback_free"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let baseline_mlx_metal_batched_direct_decode_logits_ready = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("batched_direct_decode_logits_ready"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let baseline_mlx_metal_prefix_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("prefix_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let baseline_mlx_metal_direct_decode_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("direct_decode_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let baseline_mlx_metal_readiness_blockers = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("baseline"))
        .and_then(|value| value.get("blockers"))
        .map(json_value_label)
        .unwrap_or_else(|| "[]".to_string());
    let candidate_mlx_metal_readiness = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let candidate_mlx_metal_hot_path_cpu_fallback_free = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("hot_path_cpu_fallback_free"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let candidate_mlx_metal_batched_direct_decode_logits_ready = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("batched_direct_decode_logits_ready"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let candidate_mlx_metal_prefix_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("prefix_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let candidate_mlx_metal_direct_decode_min_dispatch_share = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("direct_decode_min_mlx_metal_dispatch_share"))
        .and_then(Value::as_f64);
    let candidate_mlx_metal_readiness_blockers = regression_json
        .get("readiness")
        .and_then(|readiness| readiness.get("candidate"))
        .and_then(|value| value.get("blockers"))
        .map(json_value_label)
        .unwrap_or_else(|| "[]".to_string());
    let backend_reported_cached_prompt_tokens = regression_json
        .get("summary")
        .and_then(|summary| summary.get("backend_reported_cached_prompt_tokens"))
        .map(json_value_label)
        .unwrap_or_else(|| "none".to_string());
    let mlx_kv_cache_lines = mlx_kv_cache_regression_markdown_lines(regression_json);
    let compare_mode = explicit_or_inferred_compare_mode(regression_json);
    let tool_mode = explicit_or_inferred_tool_mode(
        regression_json,
        &["summary", "tool_mode"],
        "engine_bringup_runtime",
    );
    let selected_backend = regression_json
        .get("summary")
        .and_then(|summary| summary.get("selected_backend"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let support_tier = regression_json
        .get("summary")
        .and_then(|summary| summary.get("support_tier"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let resolution_policy = regression_json
        .get("summary")
        .and_then(|summary| summary.get("resolution_policy"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let backend_adapter = regression_json
        .get("runtime")
        .and_then(|runtime| runtime.get("backend_adapter"))
        .map(json_value_label)
        .unwrap_or_else(|| "none".to_string());
    let trusted_baseline_line = trusted_baseline
        .and_then(|baseline| baseline.get("name"))
        .and_then(Value::as_str)
        .map(|name| format!("- trusted_baseline: `{name}`\n"))
        .unwrap_or_default();
    let summary_note = compare_summary_note(&tool_mode, execution_semantics);
    let baseline_mlx_metal_prefix_min_dispatch_share =
        mlx_metal_dispatch_share_label(baseline_mlx_metal_prefix_min_dispatch_share);
    let baseline_mlx_metal_direct_decode_min_dispatch_share =
        mlx_metal_dispatch_share_label(baseline_mlx_metal_direct_decode_min_dispatch_share);
    let candidate_mlx_metal_prefix_min_dispatch_share =
        mlx_metal_dispatch_share_label(candidate_mlx_metal_prefix_min_dispatch_share);
    let candidate_mlx_metal_direct_decode_min_dispatch_share =
        mlx_metal_dispatch_share_label(candidate_mlx_metal_direct_decode_min_dispatch_share);

    format!(
        "# Benchmark Compare\n\n- baseline: `{baseline_id}`\n{trusted_baseline_line}- candidate: `{candidate_id}`\n- mode: `{compare_mode}`\n- tool_mode: `{tool_mode}`\n- selected_backend: `{selected_backend}`\n- support_tier: `{support_tier}`\n- resolution_policy: `{resolution_policy}`\n- backend_adapter: `{backend_adapter}`\n- execution_semantics: `{execution_semantics}`\n- baseline_mlx_metal_readiness: `{baseline_mlx_metal_readiness}`\n- baseline_mlx_metal_hot_path_cpu_fallback_free: `{baseline_mlx_metal_hot_path_cpu_fallback_free}`\n- baseline_mlx_metal_batched_direct_decode_logits_ready: `{baseline_mlx_metal_batched_direct_decode_logits_ready}`\n- baseline_prefix_min_mlx_metal_dispatch_share: `{baseline_mlx_metal_prefix_min_dispatch_share}`\n- baseline_direct_decode_min_mlx_metal_dispatch_share: `{baseline_mlx_metal_direct_decode_min_dispatch_share}`\n- baseline_mlx_metal_readiness_blockers: `{baseline_mlx_metal_readiness_blockers}`\n- candidate_mlx_metal_readiness: `{candidate_mlx_metal_readiness}`\n- candidate_mlx_metal_hot_path_cpu_fallback_free: `{candidate_mlx_metal_hot_path_cpu_fallback_free}`\n- candidate_mlx_metal_batched_direct_decode_logits_ready: `{candidate_mlx_metal_batched_direct_decode_logits_ready}`\n- candidate_prefix_min_mlx_metal_dispatch_share: `{candidate_mlx_metal_prefix_min_dispatch_share}`\n- candidate_direct_decode_min_mlx_metal_dispatch_share: `{candidate_mlx_metal_direct_decode_min_dispatch_share}`\n- candidate_mlx_metal_readiness_blockers: `{candidate_mlx_metal_readiness_blockers}`\n- metal_numeric_scaffold_only: `{metal_numeric_scaffold_only}`\n- metal_complete_model_forward_supported: `{metal_complete_model_forward_supported}`\n- metal_model_conditioned_inputs: `{metal_model_conditioned_inputs}`\n- mlx_metal_prefix_layers_attention: `{mlx_metal_prefix_layers_attention}`\n- metal_prefix_layers_cpu_reference: `{metal_prefix_layers_cpu_reference}`\n- mlx_metal_prefix_dispatch_count: `{mlx_metal_prefix_dispatch_count}`\n- metal_prefix_cpu_reference_dispatch_count: `{metal_prefix_cpu_reference_dispatch_count}`\n- mlx_metal_projection_f32_binding_count: `{mlx_metal_projection_f32_binding_count}`\n- mlx_metal_projection_f16_binding_count: `{mlx_metal_projection_f16_binding_count}`\n- mlx_metal_projection_bf16_binding_count: `{mlx_metal_projection_bf16_binding_count}`\n- mlx_metal_projection_unsupported_binding_count: `{mlx_metal_projection_unsupported_binding_count}`\n- metal_direct_decode_tokens: `{metal_direct_decode_tokens}`\n- metal_direct_decode_batching_opportunity_observed: `{metal_direct_decode_batching_opportunity_observed}`\n- metal_direct_decode_model_bound_ffn: `{metal_direct_decode_model_bound_ffn}`\n- metal_direct_decode_batched_logits_group_count: `{metal_direct_decode_batched_logits_group_count}`\n- metal_direct_decode_batched_logits_token_count: `{metal_direct_decode_batched_logits_token_count}`\n- metal_direct_decode_batched_group_fallback_count: `{metal_direct_decode_batched_group_fallback_count}`\n- metal_direct_decode_batched_group_fallback_token_count: `{metal_direct_decode_batched_group_fallback_token_count}`\n- metal_real_model_tensor_inputs: `{metal_real_model_tensor_inputs}`\n- metal_real_model_forward: `{metal_real_model_forward}`\n- metal_model_artifacts_validated: `{metal_model_artifacts_validated}`\n- mlx_metal_rms_norm_f32_binding_count: `{mlx_metal_rms_norm_f32_binding_count}`\n- mlx_metal_rms_norm_f16_binding_count: `{mlx_metal_rms_norm_f16_binding_count}`\n- mlx_metal_rms_norm_bf16_binding_count: `{mlx_metal_rms_norm_bf16_binding_count}`\n- mlx_metal_rms_norm_unsupported_binding_count: `{mlx_metal_rms_norm_unsupported_binding_count}`\n{mlx_kv_cache_lines}- prefix_cache_path: `{prefix_cache_path}`\n- prefix_cache_evidence: `{prefix_cache_evidence}`\n- prefix_reuse_provenance: `{prefix_reuse_provenance}`\n- backend_reported_cached_prompt_tokens: `{backend_reported_cached_prompt_tokens}`\n\n| Metric | Delta % |\n| --- | ---: |\n| TTFT | {ttft:.2} |\n| Decode tok/s | {decode:.2} |\n| Memory peak MB | {memory:.2} |\n| Prefix hit rate | {prefix:.2} |\n{}",
        summary_note
    )
}

fn runtime_config_from_manifest(manifest: &BenchmarkManifest) -> Result<RuntimeConfig, CliError> {
    let backend_policy = BackendPolicy::new(manifest.runtime.resolution_policy);
    let resolved_backend = ResolvedBackend::new(
        manifest.runtime.selected_backend,
        manifest.runtime.support_tier,
        manifest.runtime.fallback_reason.clone(),
    );
    resolved_backend
        .validate_against(&backend_policy)
        .map_err(|error| {
            CliError::Contract(format!(
                "invalid benchmark runtime backend-resolution contract: {error}"
            ))
        })?;

    if resolved_backend.selected_backend == SelectedBackend::Mlx {
        if let Some(backend_adapter) = manifest.runtime.backend_adapter.as_ref() {
            return Err(CliError::Contract(format!(
                "runtime.backend_adapter must be omitted when selected_backend is mlx; found adapter={:?}",
                backend_adapter.selected_backend()
            )));
        }
        if manifest.runtime.llama_cpp_preset.is_some() {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset must be omitted when selected_backend is mlx".to_string(),
            ));
        }
    } else {
        let backend_adapter = manifest.runtime.backend_adapter.as_ref().ok_or_else(|| {
            CliError::Contract(
                "runtime.backend_adapter is required when selected_backend is llama_cpp"
                    .to_string(),
            )
        })?;
        if backend_adapter.selected_backend() != resolved_backend.selected_backend {
            return Err(CliError::Contract(format!(
                "runtime.backend_adapter kind does not match selected_backend: adapter={:?}, selected_backend={:?}",
                backend_adapter.selected_backend(),
                resolved_backend.selected_backend
            )));
        }
    }
    let llama_cpp_preset = if resolved_backend.selected_backend == SelectedBackend::LlamaCpp {
        let preset = manifest
            .runtime
            .llama_cpp_preset
            .clone()
            .unwrap_or_default();
        validate_llama_cpp_preset(&preset, manifest)?;
        Some(preset)
    } else {
        None
    };

    let default_session_config = EngineSessionConfig::default();
    let mlx_model_artifacts_dir = manifest
        .runtime
        .mlx_model_artifacts_dir
        .clone()
        .or(default_session_config.mlx_model_artifacts_dir.clone());
    let mlx_model_artifacts_source = manifest
        .runtime
        .mlx_model_artifacts_dir
        .as_ref()
        .map(|_| NativeModelArtifactsSource::ExplicitConfig)
        .or(default_session_config.mlx_model_artifacts_source);

    Ok(RuntimeConfig {
        deterministic: manifest.runtime.deterministic,
        max_batch_tokens: manifest.runtime.max_batch_tokens,
        block_size_tokens: 16,
        kv_total_blocks: manifest.runtime.kv_total_blocks,
        flags: manifest.runtime.flags.clone(),
        llama_cpp_preset,
        backend_policy,
        resolved_backend,
        backend_adapter: manifest.runtime.backend_adapter.clone(),
        mlx_model_artifacts_dir,
        mlx_model_artifacts_source,
    })
}

fn validate_llama_cpp_preset(
    preset: &LlamaCppPresetManifest,
    manifest: &BenchmarkManifest,
) -> Result<(), CliError> {
    if preset.parallel_slots == 0 {
        return Err(CliError::Contract(
            "runtime.llama_cpp_preset.parallel_slots must be greater than zero".to_string(),
        ));
    }
    if let Some(logical_batch_size) = preset.logical_batch_size {
        if logical_batch_size == 0 {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset.logical_batch_size must be greater than zero".to_string(),
            ));
        }
    }
    if let Some(physical_batch_size) = preset.physical_batch_size {
        if physical_batch_size == 0 {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset.physical_batch_size must be greater than zero"
                    .to_string(),
            ));
        }
    }
    if let (Some(logical), Some(physical)) = (preset.logical_batch_size, preset.physical_batch_size)
    {
        if physical > logical {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset.physical_batch_size must not exceed logical_batch_size"
                    .to_string(),
            ));
        }
    }
    if manifest.checks.require_prefix_reuse && !preset.cache_prompt {
        return Err(CliError::Contract(
            "runtime.llama_cpp_preset.cache_prompt must be true when checks.require_prefix_reuse is true"
                .to_string(),
        ));
    }
    Ok(())
}

fn normalize_runtime_for_test_execution(
    runtime: RuntimeConfig,
    _preserve_native_artifacts_for_test: bool,
) -> RuntimeConfig {
    #[cfg(test)]
    {
        // Unit tests should remain stable across local machines and must not
        // silently flip from deterministic placeholder execution into repo
        // auto-detected Metal bring-up just because native artifacts happen to
        // exist in the checkout.
        if runtime.uses_mlx_runtime() && !_preserve_native_artifacts_for_test {
            return RuntimeConfig {
                mlx_model_artifacts_dir: None,
                mlx_model_artifacts_source: None,
                ..runtime
            };
        }
    }

    runtime
}

fn sampling_from_manifest(manifest: &BenchmarkManifest) -> Result<SamplingParams, CliError> {
    Ok(SamplingParams {
        temperature: manifest.sampling.temperature,
        top_p: manifest.sampling.top_p,
        top_k: manifest.sampling.top_k,
        repetition_penalty: 1.0,
        seed: manifest.sampling.seed,
        deterministic: manifest.runtime.deterministic,
    })
}

fn generate_request_from_spec(
    spec: &SyntheticRequestSpec,
    manifest: &BenchmarkManifest,
) -> GenerateRequest {
    GenerateRequest {
        model_id: spec.model_family.clone(),
        input_tokens: spec.input_tokens.clone(),
        input_text: spec.input_text.clone(),
        max_output_tokens: spec.max_output_tokens,
        sampling: GenerateSampling {
            temperature: manifest.sampling.temperature,
            top_p: manifest.sampling.top_p,
            top_k: manifest.sampling.top_k,
            min_p: None,
            repetition_penalty: 1.0,
            repetition_context_size: None,
            seed: manifest.sampling.seed,
            deterministic: Some(manifest.runtime.deterministic),
        },
        stop_sequences: Vec::new(),
        metadata: spec.metadata.clone(),
    }
}

fn manifest_expect_deterministic(manifest: &BenchmarkManifest) -> Result<bool, CliError> {
    Ok(manifest.checks.expect_deterministic)
}

fn deterministic_runtime_digest(value: &Value) -> Value {
    json!({
        "requests": value.get("requests").cloned().unwrap_or(Value::Null)
    })
}

fn route_metadata_from_generate_route(route: &GenerateRouteReport) -> RouteMetadata {
    RouteMetadata {
        execution_plan: route.execution_plan.clone(),
        attention_route: route.attention_route.clone(),
        kv_mode: route.kv_mode.clone(),
        prefix_cache_path: route.prefix_cache_path.clone(),
        barrier_mode: route.barrier_mode.clone(),
        crossover_decisions: route
            .crossover_decisions
            .iter()
            .map(|(key, value)| (key.clone(), *value))
            .collect(),
    }
}

fn route_decision_value(route: &RouteMetadata, key: &str) -> u32 {
    route
        .crossover_decisions
        .iter()
        .find(|(decision, _)| decision == key)
        .map(|(_, value)| *value)
        .unwrap_or(0)
}

fn route_decision_flag(route: &RouteMetadata, key: &str) -> bool {
    route_decision_value(route, key) > 0
}

fn mlx_metal_coverage_ratio(native: u32, cpu: u32) -> Option<f64> {
    let total = native.checked_add(cpu)?;
    (total > 0).then_some(native as f64 / total as f64)
}

#[derive(Debug, Clone, PartialEq)]
struct MlxMetalReadiness {
    status: &'static str,
    hot_path_cpu_fallback_free: bool,
    batched_direct_decode_logits_ready: bool,
    prefix_min_mlx_metal_dispatch_share: Option<f64>,
    direct_decode_min_mlx_metal_dispatch_share: Option<f64>,
    blockers: Vec<&'static str>,
}

impl MlxMetalReadiness {
    fn to_json(&self) -> Value {
        json!({
            "status": self.status,
            "hot_path_cpu_fallback_free": self.hot_path_cpu_fallback_free,
            "batched_direct_decode_logits_ready": self.batched_direct_decode_logits_ready,
            "prefix_min_mlx_metal_dispatch_share": self.prefix_min_mlx_metal_dispatch_share,
            "direct_decode_min_mlx_metal_dispatch_share": self.direct_decode_min_mlx_metal_dispatch_share,
            "blockers": self.blockers
        })
    }

    fn blockers_label(&self) -> String {
        if self.blockers.is_empty() {
            "none".to_string()
        } else {
            self.blockers.join(",")
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MlxMetalReadinessInputs<'a> {
    tool_mode: &'a str,
    selected_backend: &'a str,
    native_dense_dequantized_source: bool,
    native_quantized_projection_binding_count: u32,
    direct_decode_batching_opportunity_observed: bool,
    metal_complete_model_forward_supported: bool,
    metal_real_model_forward: bool,
    metal_model_artifacts_validated: bool,
    mlx_metal_prefix_layers_attention: bool,
    metal_prefix_layers_cpu_reference: bool,
    metal_prefix_cpu_reference_dispatch_count: u32,
    mlx_metal_prefix_projection_row_count: u32,
    metal_prefix_cpu_projection_row_count: u32,
    mlx_metal_prefix_rms_norm_element_count: u32,
    metal_prefix_cpu_rms_norm_element_count: u32,
    mlx_metal_prefix_ffn_activation_element_count: u32,
    metal_prefix_cpu_ffn_activation_element_count: u32,
    mlx_metal_prefix_residual_add_element_count: u32,
    metal_prefix_cpu_residual_add_element_count: u32,
    mlx_metal_prefix_scale_element_count: u32,
    metal_prefix_cpu_scale_element_count: u32,
    metal_direct_decode_tokens: bool,
    metal_direct_decode_model_bound_ffn: bool,
    metal_direct_decode_batched_logits_group_count: u32,
    metal_direct_decode_batched_logits_token_count: u32,
    metal_direct_decode_batched_group_fallback_count: u32,
    metal_direct_decode_batched_group_fallback_token_count: u32,
    mlx_metal_direct_decode_projection_row_count: u32,
    metal_direct_decode_cpu_projection_row_count: u32,
    mlx_metal_direct_decode_rms_norm_element_count: u32,
    metal_direct_decode_cpu_rms_norm_element_count: u32,
    mlx_metal_direct_decode_ffn_activation_element_count: u32,
    metal_direct_decode_cpu_ffn_activation_element_count: u32,
    mlx_metal_direct_decode_residual_add_element_count: u32,
    metal_direct_decode_cpu_residual_add_element_count: u32,
    mlx_metal_direct_decode_scale_element_count: u32,
    metal_direct_decode_cpu_scale_element_count: u32,
}

fn min_mlx_metal_dispatch_share(shares: &[Option<f64>]) -> Option<f64> {
    shares.iter().flatten().copied().reduce(f64::min)
}

fn mlx_metal_readiness(inputs: MlxMetalReadinessInputs<'_>) -> MlxMetalReadiness {
    if inputs.tool_mode.starts_with("llama_cpp_") || inputs.selected_backend != "mlx" {
        return MlxMetalReadiness {
            status: "delegated_runtime",
            hot_path_cpu_fallback_free: false,
            batched_direct_decode_logits_ready: false,
            prefix_min_mlx_metal_dispatch_share: None,
            direct_decode_min_mlx_metal_dispatch_share: None,
            blockers: vec!["delegated_runtime_not_mlx"],
        };
    }

    let prefix_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_projection_row_count,
        inputs.metal_prefix_cpu_projection_row_count,
    );
    let prefix_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_rms_norm_element_count,
        inputs.metal_prefix_cpu_rms_norm_element_count,
    );
    let prefix_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_ffn_activation_element_count,
        inputs.metal_prefix_cpu_ffn_activation_element_count,
    );
    let prefix_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_residual_add_element_count,
        inputs.metal_prefix_cpu_residual_add_element_count,
    );
    let prefix_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_prefix_scale_element_count,
        inputs.metal_prefix_cpu_scale_element_count,
    );
    let direct_decode_projection_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_projection_row_count,
        inputs.metal_direct_decode_cpu_projection_row_count,
    );
    let direct_decode_rms_norm_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_rms_norm_element_count,
        inputs.metal_direct_decode_cpu_rms_norm_element_count,
    );
    let direct_decode_ffn_activation_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_ffn_activation_element_count,
        inputs.metal_direct_decode_cpu_ffn_activation_element_count,
    );
    let direct_decode_residual_add_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_residual_add_element_count,
        inputs.metal_direct_decode_cpu_residual_add_element_count,
    );
    let direct_decode_scale_mlx_metal_dispatch_share = mlx_metal_coverage_ratio(
        inputs.mlx_metal_direct_decode_scale_element_count,
        inputs.metal_direct_decode_cpu_scale_element_count,
    );
    let prefix_min_mlx_metal_dispatch_share = min_mlx_metal_dispatch_share(&[
        prefix_projection_mlx_metal_dispatch_share,
        prefix_rms_norm_mlx_metal_dispatch_share,
        prefix_ffn_activation_mlx_metal_dispatch_share,
        prefix_residual_add_mlx_metal_dispatch_share,
        prefix_scale_mlx_metal_dispatch_share,
    ]);
    let direct_decode_min_mlx_metal_dispatch_share = min_mlx_metal_dispatch_share(&[
        direct_decode_projection_mlx_metal_dispatch_share,
        direct_decode_rms_norm_mlx_metal_dispatch_share,
        direct_decode_ffn_activation_mlx_metal_dispatch_share,
        direct_decode_residual_add_mlx_metal_dispatch_share,
        direct_decode_scale_mlx_metal_dispatch_share,
    ]);
    let hot_path_cpu_fallback_free = inputs.metal_prefix_cpu_reference_dispatch_count == 0
        && inputs.metal_prefix_cpu_projection_row_count == 0
        && inputs.metal_prefix_cpu_rms_norm_element_count == 0
        && inputs.metal_prefix_cpu_ffn_activation_element_count == 0
        && inputs.metal_prefix_cpu_residual_add_element_count == 0
        && inputs.metal_prefix_cpu_scale_element_count == 0
        && inputs.metal_direct_decode_cpu_projection_row_count == 0
        && inputs.metal_direct_decode_cpu_rms_norm_element_count == 0
        && inputs.metal_direct_decode_cpu_ffn_activation_element_count == 0
        && inputs.metal_direct_decode_cpu_residual_add_element_count == 0
        && inputs.metal_direct_decode_cpu_scale_element_count == 0;
    let batched_direct_decode_logits_ready = !inputs.direct_decode_batching_opportunity_observed
        || (inputs.metal_direct_decode_tokens
            && inputs.metal_direct_decode_batched_logits_group_count > 0
            && inputs.metal_direct_decode_batched_logits_token_count > 0
            && inputs.metal_direct_decode_batched_group_fallback_count == 0
            && inputs.metal_direct_decode_batched_group_fallback_token_count == 0);

    let mut blockers = Vec::new();
    if !inputs.metal_model_artifacts_validated {
        blockers.push("model_artifacts_unvalidated");
    }
    if inputs.native_dense_dequantized_source {
        blockers.push(NATIVE_DENSE_DEQUANTIZED_SOURCE_BLOCKER);
    }
    if inputs.native_quantized_projection_binding_count > 0 {
        blockers.push("native_quantized_projection_kernel_missing");
    }
    if !inputs.metal_complete_model_forward_supported {
        blockers.push("complete_model_forward_not_reported");
    }
    if !inputs.metal_real_model_forward {
        blockers.push("real_model_forward_not_reported");
    }
    if !inputs.mlx_metal_prefix_layers_attention {
        blockers.push("prefix_native_attention_not_observed");
    }
    if inputs.metal_prefix_layers_cpu_reference
        || inputs.metal_prefix_cpu_reference_dispatch_count > 0
    {
        blockers.push("prefix_attention_cpu_reference_remaining");
    }
    if inputs.metal_prefix_cpu_projection_row_count > 0 {
        blockers.push("prefix_projection_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_rms_norm_element_count > 0 {
        blockers.push("prefix_rms_norm_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_ffn_activation_element_count > 0 {
        blockers.push("prefix_ffn_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_residual_add_element_count > 0 {
        blockers.push("prefix_residual_add_cpu_fallback_remaining");
    }
    if inputs.metal_prefix_cpu_scale_element_count > 0 {
        blockers.push("prefix_scale_cpu_fallback_remaining");
    }
    if !inputs.metal_direct_decode_tokens {
        blockers.push("direct_decode_tokens_not_observed");
    } else {
        if !inputs.metal_direct_decode_model_bound_ffn {
            blockers.push("direct_decode_model_bound_ffn_not_reported");
        }
        if inputs.direct_decode_batching_opportunity_observed && !batched_direct_decode_logits_ready
        {
            blockers.push("batched_direct_decode_logits_not_observed");
        }
        if inputs.metal_direct_decode_batched_group_fallback_count > 0
            || inputs.metal_direct_decode_batched_group_fallback_token_count > 0
        {
            blockers.push("batched_direct_decode_group_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_projection_row_count > 0 {
            blockers.push("direct_decode_projection_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_rms_norm_element_count > 0 {
            blockers.push("direct_decode_rms_norm_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_ffn_activation_element_count > 0 {
            blockers.push("direct_decode_ffn_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_residual_add_element_count > 0 {
            blockers.push("direct_decode_residual_add_cpu_fallback_remaining");
        }
        if inputs.metal_direct_decode_cpu_scale_element_count > 0 {
            blockers.push("direct_decode_scale_cpu_fallback_remaining");
        }
    }

    MlxMetalReadiness {
        status: if blockers.is_empty() {
            "ready"
        } else {
            "coverage_gap"
        },
        hot_path_cpu_fallback_free,
        batched_direct_decode_logits_ready,
        prefix_min_mlx_metal_dispatch_share,
        direct_decode_min_mlx_metal_dispatch_share,
        blockers,
    }
}

fn mlx_metal_readiness_from_route_json(
    tool_mode: &str,
    selected_backend: &str,
    route_json: &Value,
    native_dense_dequantized_source: bool,
) -> MlxMetalReadiness {
    mlx_metal_readiness(MlxMetalReadinessInputs {
        tool_mode,
        selected_backend,
        native_dense_dequantized_source,
        native_quantized_projection_binding_count: route_count_from_json(
            route_json,
            "mlx_metal_projection_source_quantized_binding_count",
            &["metal_dispatch_native_projection_source_quantized_binding_count"],
        ),
        direct_decode_batching_opportunity_observed: route_flag_from_json(
            route_json,
            "metal_direct_decode_batching_opportunity_observed",
            &["metal_direct_decode_batching_opportunity_observed"],
        ) || route_count_from_json(
            route_json,
            "metal_direct_decode_batched_logits_group_count",
            &["metal_dispatch_direct_decode_batched_logits_group_count"],
        ) > 0
            || route_count_from_json(
                route_json,
                "metal_direct_decode_batched_group_fallback_count",
                &["metal_dispatch_direct_decode_batched_group_fallback_count"],
            ) > 0,
        metal_complete_model_forward_supported: route_flag_from_json(
            route_json,
            "metal_complete_model_forward_supported",
            &[
                "metal_dispatch_complete_model_forward_supported",
                "metal_dispatch_runtime_complete_model_forward_supported",
            ],
        ),
        metal_real_model_forward: route_flag_from_json(
            route_json,
            "metal_real_model_forward",
            &["metal_dispatch_real_model_forward"],
        ),
        metal_model_artifacts_validated: route_flag_from_json(
            route_json,
            "metal_model_artifacts_validated",
            &["metal_dispatch_model_artifacts_validated"],
        ),
        mlx_metal_prefix_layers_attention: route_flag_from_json(
            route_json,
            "mlx_metal_prefix_layers_attention",
            &["metal_dispatch_prefix_layers_native_attention"],
        ),
        metal_prefix_layers_cpu_reference: route_flag_from_json(
            route_json,
            "metal_prefix_layers_cpu_reference",
            &["metal_dispatch_prefix_layers_cpu_reference"],
        ),
        metal_prefix_cpu_reference_dispatch_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_reference_dispatch_count",
            &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
        ),
        mlx_metal_prefix_projection_row_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_projection_row_count",
            &["metal_dispatch_prefix_native_projection_row_count"],
        ),
        metal_prefix_cpu_projection_row_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_projection_row_count",
            &["metal_dispatch_prefix_cpu_projection_row_count"],
        ),
        mlx_metal_prefix_rms_norm_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_rms_norm_element_count",
            &["metal_dispatch_prefix_native_rms_norm_element_count"],
        ),
        metal_prefix_cpu_rms_norm_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_rms_norm_element_count",
            &["metal_dispatch_prefix_cpu_rms_norm_element_count"],
        ),
        mlx_metal_prefix_ffn_activation_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_ffn_activation_element_count",
            &["metal_dispatch_prefix_native_ffn_activation_element_count"],
        ),
        metal_prefix_cpu_ffn_activation_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_ffn_activation_element_count",
            &["metal_dispatch_prefix_cpu_ffn_activation_element_count"],
        ),
        mlx_metal_prefix_residual_add_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_residual_add_element_count",
            &["metal_dispatch_prefix_native_residual_add_element_count"],
        ),
        metal_prefix_cpu_residual_add_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_residual_add_element_count",
            &["metal_dispatch_prefix_cpu_residual_add_element_count"],
        ),
        mlx_metal_prefix_scale_element_count: route_count_from_json(
            route_json,
            "mlx_metal_prefix_scale_element_count",
            &["metal_dispatch_prefix_native_scale_element_count"],
        ),
        metal_prefix_cpu_scale_element_count: route_count_from_json(
            route_json,
            "metal_prefix_cpu_scale_element_count",
            &["metal_dispatch_prefix_cpu_scale_element_count"],
        ),
        metal_direct_decode_tokens: route_flag_from_json(
            route_json,
            "metal_direct_decode_tokens",
            &["metal_dispatch_direct_decode_tokens"],
        ),
        metal_direct_decode_model_bound_ffn: route_flag_from_json(
            route_json,
            "metal_direct_decode_model_bound_ffn",
            &["metal_dispatch_direct_decode_model_bound_ffn"],
        ),
        metal_direct_decode_batched_logits_group_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_logits_group_count",
            &["metal_dispatch_direct_decode_batched_logits_group_count"],
        ),
        metal_direct_decode_batched_logits_token_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_logits_token_count",
            &["metal_dispatch_direct_decode_batched_logits_token_count"],
        ),
        metal_direct_decode_batched_group_fallback_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_group_fallback_count",
            &["metal_dispatch_direct_decode_batched_group_fallback_count"],
        ),
        metal_direct_decode_batched_group_fallback_token_count: route_count_from_json(
            route_json,
            "metal_direct_decode_batched_group_fallback_token_count",
            &["metal_dispatch_direct_decode_batched_group_fallback_token_count"],
        ),
        mlx_metal_direct_decode_projection_row_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_projection_row_count",
            &["metal_dispatch_direct_decode_native_projection_row_count"],
        ),
        metal_direct_decode_cpu_projection_row_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_projection_row_count",
            &["metal_dispatch_direct_decode_cpu_projection_row_count"],
        ),
        mlx_metal_direct_decode_rms_norm_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_rms_norm_element_count",
            &["metal_dispatch_direct_decode_native_rms_norm_element_count"],
        ),
        metal_direct_decode_cpu_rms_norm_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_rms_norm_element_count",
            &["metal_dispatch_direct_decode_cpu_rms_norm_element_count"],
        ),
        mlx_metal_direct_decode_ffn_activation_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_ffn_activation_element_count",
            &["metal_dispatch_direct_decode_native_ffn_activation_element_count"],
        ),
        metal_direct_decode_cpu_ffn_activation_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_ffn_activation_element_count",
            &["metal_dispatch_direct_decode_cpu_ffn_activation_element_count"],
        ),
        mlx_metal_direct_decode_residual_add_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_residual_add_element_count",
            &["metal_dispatch_direct_decode_native_residual_add_element_count"],
        ),
        metal_direct_decode_cpu_residual_add_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_residual_add_element_count",
            &["metal_dispatch_direct_decode_cpu_residual_add_element_count"],
        ),
        mlx_metal_direct_decode_scale_element_count: route_count_from_json(
            route_json,
            "mlx_metal_direct_decode_scale_element_count",
            &["metal_dispatch_direct_decode_native_scale_element_count"],
        ),
        metal_direct_decode_cpu_scale_element_count: route_count_from_json(
            route_json,
            "metal_direct_decode_cpu_scale_element_count",
            &["metal_dispatch_direct_decode_cpu_scale_element_count"],
        ),
    })
}

fn native_model_report_has_dense_dequantized_source(report: &NativeModelReport) -> bool {
    serde_json::to_value(report)
        .ok()
        .is_some_and(|native_model| {
            native_dense_dequantized_source_from_native_model_json(&native_model)
        })
}

fn native_dense_dequantized_source_from_runtime_json(runtime_json: &Value) -> bool {
    nested_value(runtime_json, &["mlx_model"])
        .is_some_and(native_dense_dequantized_source_from_native_model_json)
}

fn native_dense_dequantized_source_from_native_model_json(native_model: &Value) -> bool {
    let contains_quantized_tensors = nested_value(
        native_model,
        &["source_quantization", "contains_quantized_tensors"],
    )
    .and_then(Value::as_bool)
    .unwrap_or(false);
    let has_dense_dequantized_note = nested_value(native_model, &["runtime_status", "notes"])
        .and_then(Value::as_array)
        .is_some_and(|notes| {
            notes
                .iter()
                .any(|note| note.as_str() == Some(NATIVE_DENSE_DEQUANTIZED_EXPORT_NOTE))
        });

    contains_quantized_tensors && has_dense_dequantized_note
}

fn mlx_metal_dispatch_share_label(share: Option<f64>) -> String {
    share
        .map(|value| format!("{:.2}%", value * 100.0))
        .unwrap_or_else(|| "n/a".to_string())
}

fn route_model_layer_count(route: &RouteMetadata) -> u32 {
    route_decision_value(route, "metal_dispatch_model_layer_count")
}

fn route_prefix_native_dispatch_count(route: &RouteMetadata) -> u32 {
    route_decision_value(route, "metal_dispatch_prefix_native_dispatch_count")
}

fn route_prefix_cpu_reference_dispatch_count(route: &RouteMetadata) -> u32 {
    route_decision_value(route, "metal_dispatch_prefix_cpu_reference_dispatch_count")
}

fn route_execution_semantics_label(route: &RouteMetadata) -> &'static str {
    let metal_real_model_forward = route_decision_flag(route, "metal_dispatch_real_model_forward");
    let metal_direct_decode_model_bound_ffn =
        route_decision_flag(route, "metal_dispatch_direct_decode_model_bound_ffn");
    let metal_real_model_tensor_inputs =
        route_decision_flag(route, "metal_dispatch_real_model_tensor_inputs")
            || route_decision_flag(route, "metal_dispatch_runtime_real_model_tensor_inputs");
    let metal_model_conditioned_inputs =
        route_decision_flag(route, "metal_dispatch_model_conditioned_inputs")
            || route_decision_flag(route, "metal_dispatch_runtime_model_conditioned_inputs");
    let mlx_metal_prefix_layers_attention =
        route_decision_flag(route, "metal_dispatch_prefix_layers_native_attention");
    let metal_prefix_layers_cpu_reference =
        route_decision_flag(route, "metal_dispatch_prefix_layers_cpu_reference");
    let mlx_metal_prefix_dispatch_count = route_prefix_native_dispatch_count(route);
    let metal_prefix_cpu_reference_dispatch_count =
        route_prefix_cpu_reference_dispatch_count(route);
    let metal_multilayer_mixed_prefix_attention = mlx_metal_prefix_dispatch_count > 0
        && metal_prefix_cpu_reference_dispatch_count > 0
        || (mlx_metal_prefix_layers_attention && metal_prefix_layers_cpu_reference);
    let metal_numeric_scaffold_only =
        route_decision_flag(route, "metal_dispatch_numeric_scaffold_only");
    let metal_model_layer_count = route_model_layer_count(route);
    let metal_multilayer_model_incomplete =
        metal_model_layer_count > 1 && !metal_real_model_forward;
    let delegated_cached_tokens = route_decision_flag(route, "delegated_cached_tokens")
        || route.prefix_cache_path.as_deref() == Some("delegated_prompt_cache");

    if metal_real_model_forward {
        "metal_real_model_forward"
    } else if metal_multilayer_model_incomplete && metal_multilayer_mixed_prefix_attention {
        "metal_multilayer_mixed_prefix_attention"
    } else if metal_multilayer_model_incomplete && mlx_metal_prefix_layers_attention {
        "metal_multilayer_native_prefix_attention"
    } else if metal_multilayer_model_incomplete {
        "metal_multilayer_model_incomplete"
    } else if metal_real_model_tensor_inputs && metal_direct_decode_model_bound_ffn {
        "metal_model_bound_ffn_decode"
    } else if metal_real_model_tensor_inputs {
        "metal_real_model_tensor_inputs"
    } else if metal_model_conditioned_inputs {
        "metal_model_conditioned_numeric_scaffold"
    } else if metal_numeric_scaffold_only {
        "metal_numeric_scaffold_only"
    } else if delegated_cached_tokens {
        "delegated_backend_runtime"
    } else {
        "none_observed"
    }
}

fn route_flag_from_json(route_json: &Value, field: &str, decision_keys: &[&str]) -> bool {
    if let Some(value) = nested_value(route_json, &[field]).and_then(Value::as_bool) {
        return value;
    }

    decision_keys.iter().any(|key| {
        nested_value(route_json, &["crossover_decisions", key])
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
    })
}

fn route_count_from_json(route_json: &Value, field: &str, decision_keys: &[&str]) -> u32 {
    if let Some(value) = nested_value(route_json, &[field]).and_then(Value::as_u64) {
        return value.min(u64::from(u32::MAX)) as u32;
    }

    decision_keys
        .iter()
        .find_map(|key| {
            nested_value(route_json, &["crossover_decisions", key])
                .and_then(Value::as_u64)
                .map(|value| value.min(u64::from(u32::MAX)) as u32)
        })
        .unwrap_or(0)
}

fn route_counter_from_json(route_json: &Value, key: &str) -> u32 {
    route_count_from_json(route_json, key, &[key])
}

fn mlx_kv_cache_route_markdown_lines(route_json: &Value, prefix: &str) -> String {
    let mut lines = String::new();
    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        lines.push_str(&format!(
            "- {prefix}{key}: `{}`\n",
            route_counter_from_json(route_json, key)
        ));
    }
    lines
}

fn mlx_kv_cache_regression_markdown_lines(regression_json: &Value) -> String {
    let mut lines = String::new();
    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        let baseline = nested_value(regression_json, &["contract", key, "baseline"])
            .or_else(|| nested_value(regression_json, &["summary", key]))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let candidate = nested_value(regression_json, &["contract", key, "candidate"])
            .and_then(Value::as_u64)
            .unwrap_or(0);
        lines.push_str(&format!("- baseline_{key}: `{baseline}`\n"));
        lines.push_str(&format!("- candidate_{key}: `{candidate}`\n"));
    }
    lines
}

fn route_execution_semantics_from_json(route_json: &Value) -> String {
    if let Some(value) = nested_value(route_json, &["execution_semantics"]).and_then(Value::as_str)
    {
        return value.to_string();
    }

    let metal_real_model_forward = route_flag_from_json(
        route_json,
        "metal_real_model_forward",
        &["metal_dispatch_real_model_forward"],
    );
    let metal_direct_decode_model_bound_ffn = route_flag_from_json(
        route_json,
        "metal_direct_decode_model_bound_ffn",
        &["metal_dispatch_direct_decode_model_bound_ffn"],
    );
    let metal_real_model_tensor_inputs = route_flag_from_json(
        route_json,
        "metal_real_model_tensor_inputs",
        &[
            "metal_dispatch_real_model_tensor_inputs",
            "metal_dispatch_runtime_real_model_tensor_inputs",
        ],
    );
    let metal_model_conditioned_inputs = route_flag_from_json(
        route_json,
        "metal_model_conditioned_inputs",
        &[
            "metal_dispatch_model_conditioned_inputs",
            "metal_dispatch_runtime_model_conditioned_inputs",
        ],
    );
    let mlx_metal_prefix_layers_attention = route_flag_from_json(
        route_json,
        "mlx_metal_prefix_layers_attention",
        &["metal_dispatch_prefix_layers_native_attention"],
    );
    let metal_prefix_layers_cpu_reference = route_flag_from_json(
        route_json,
        "metal_prefix_layers_cpu_reference",
        &["metal_dispatch_prefix_layers_cpu_reference"],
    );
    let mlx_metal_prefix_dispatch_count = route_count_from_json(
        route_json,
        "mlx_metal_prefix_dispatch_count",
        &["metal_dispatch_prefix_native_dispatch_count"],
    );
    let metal_prefix_cpu_reference_dispatch_count = route_count_from_json(
        route_json,
        "metal_prefix_cpu_reference_dispatch_count",
        &["metal_dispatch_prefix_cpu_reference_dispatch_count"],
    );
    let metal_multilayer_mixed_prefix_attention = mlx_metal_prefix_dispatch_count > 0
        && metal_prefix_cpu_reference_dispatch_count > 0
        || (mlx_metal_prefix_layers_attention && metal_prefix_layers_cpu_reference);
    let metal_numeric_scaffold_only = route_flag_from_json(
        route_json,
        "metal_numeric_scaffold_only",
        &["metal_dispatch_numeric_scaffold_only"],
    );
    let metal_model_layer_count = nested_value(route_json, &["metal_model_layer_count"])
        .and_then(Value::as_u64)
        .map(|value| value.min(u64::from(u32::MAX)) as u32)
        .unwrap_or_else(|| {
            nested_value(
                route_json,
                &["crossover_decisions", "metal_dispatch_model_layer_count"],
            )
            .and_then(Value::as_u64)
            .map(|value| value.min(u64::from(u32::MAX)) as u32)
            .unwrap_or(0)
        });
    let metal_multilayer_model_incomplete =
        metal_model_layer_count > 1 && !metal_real_model_forward;
    let delegated_backend_runtime = nested_value(route_json, &["prefix_cache_path"])
        .and_then(Value::as_str)
        == Some("delegated_prompt_cache")
        || nested_value(
            route_json,
            &["crossover_decisions", "delegated_cached_tokens"],
        )
        .and_then(Value::as_u64)
        .unwrap_or(0)
            > 0;

    if metal_real_model_forward {
        "metal_real_model_forward".to_string()
    } else if metal_multilayer_model_incomplete && metal_multilayer_mixed_prefix_attention {
        "metal_multilayer_mixed_prefix_attention".to_string()
    } else if metal_multilayer_model_incomplete && mlx_metal_prefix_layers_attention {
        "metal_multilayer_native_prefix_attention".to_string()
    } else if metal_multilayer_model_incomplete {
        "metal_multilayer_model_incomplete".to_string()
    } else if metal_real_model_tensor_inputs && metal_direct_decode_model_bound_ffn {
        "metal_model_bound_ffn_decode".to_string()
    } else if metal_real_model_tensor_inputs {
        "metal_real_model_tensor_inputs".to_string()
    } else if metal_model_conditioned_inputs {
        "metal_model_conditioned_numeric_scaffold".to_string()
    } else if metal_numeric_scaffold_only {
        "metal_numeric_scaffold_only".to_string()
    } else if delegated_backend_runtime {
        "delegated_backend_runtime".to_string()
    } else {
        "none_observed".to_string()
    }
}

fn route_execution_semantics_from_environment_json(environment: &Value) -> String {
    let route_json = environment.get("route").unwrap_or(&Value::Null);
    let semantics = route_execution_semantics_from_json(route_json);

    if semantics != "none_observed" {
        return semantics;
    }

    semantics
}

fn serialize_route_metadata(route: &RouteMetadata) -> Value {
    let mut crossover_decisions = Map::new();
    for (key, value) in &route.crossover_decisions {
        crossover_decisions.insert(key.clone(), json!(value));
    }

    let mut route_json = Map::new();
    route_json.insert("execution_plan".to_string(), json!(route.execution_plan));
    route_json.insert(
        "execution_semantics".to_string(),
        json!(route_execution_semantics_label(route)),
    );
    route_json.insert("attention_route".to_string(), json!(route.attention_route));
    route_json.insert("kv_mode".to_string(), json!(route.kv_mode));
    route_json.insert(
        "prefix_cache_path".to_string(),
        json!(route.prefix_cache_path),
    );
    route_json.insert(
        "prefix_cache_evidence".to_string(),
        json!(prefix_cache_evidence_label(route)),
    );
    route_json.insert(
        "prefix_reuse_provenance".to_string(),
        json!(prefix_reuse_provenance_label(route)),
    );
    route_json.insert("barrier_mode".to_string(), json!(route.barrier_mode));
    route_json.insert(
        "metal_numeric_scaffold_only".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_numeric_scaffold_only"
        )),
    );
    route_json.insert(
        "metal_model_layer_count".to_string(),
        json!(route_model_layer_count(route)),
    );
    for key in ax_engine_core::ROUTE_DECISION_AX_MLX_KV_KEYS {
        if let Some(value) = crossover_decisions.get(key).cloned() {
            route_json.insert(key.to_string(), value);
        }
    }
    route_json.insert(
        "metal_complete_model_forward_supported".to_string(),
        json!(
            route_decision_flag(route, "metal_dispatch_complete_model_forward_supported")
                || route_decision_flag(
                    route,
                    "metal_dispatch_runtime_complete_model_forward_supported"
                )
        ),
    );
    route_json.insert(
        "metal_multilayer_model_incomplete".to_string(),
        json!(
            route_model_layer_count(route) > 1
                && !route_decision_flag(route, "metal_dispatch_real_model_forward")
                && !(route_decision_flag(route, "metal_dispatch_complete_model_forward_supported")
                    || route_decision_flag(
                        route,
                        "metal_dispatch_runtime_complete_model_forward_supported"
                    ))
        ),
    );
    route_json.insert(
        "metal_real_model_forward".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_real_model_forward"
        )),
    );
    route_json.insert(
        "metal_model_conditioned_inputs".to_string(),
        json!(
            route_decision_flag(route, "metal_dispatch_model_conditioned_inputs")
                || route_decision_flag(route, "metal_dispatch_runtime_model_conditioned_inputs")
        ),
    );
    route_json.insert(
        "mlx_metal_prefix_layers_attention".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_prefix_layers_native_attention"
        )),
    );
    route_json.insert(
        "metal_prefix_layers_cpu_reference".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_prefix_layers_cpu_reference"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_dispatch_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_dispatch_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_reference_dispatch_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_reference_dispatch_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_projection_row_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_projection_row_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_residual_add_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_residual_add_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_prefix_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_native_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_cpu_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_prefix_cpu_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_prefix_projection_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(route, "metal_dispatch_prefix_native_projection_row_count"),
            route_decision_value(route, "metal_dispatch_prefix_cpu_projection_row_count"),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_rms_norm_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(route, "metal_dispatch_prefix_native_rms_norm_element_count"),
            route_decision_value(route, "metal_dispatch_prefix_cpu_rms_norm_element_count"),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_ffn_activation_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_prefix_native_ffn_activation_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_ffn_activation_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_residual_add_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_prefix_native_residual_add_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_prefix_cpu_residual_add_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_prefix_scale_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(route, "metal_dispatch_prefix_native_scale_element_count"),
            route_decision_value(route, "metal_dispatch_prefix_cpu_scale_element_count"),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "mlx_metal_projection_f32_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_f32_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_projection_f16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_f16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_projection_bf16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_bf16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_projection_unsupported_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_projection_unsupported_binding_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_tokens".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_direct_decode_tokens"
        )),
    );
    route_json.insert(
        "metal_direct_decode_model_bound_ffn".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_direct_decode_model_bound_ffn"
        )),
    );
    route_json.insert(
        "metal_direct_decode_checksum_lo".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_checksum_lo"
        )),
    );
    route_json.insert(
        "metal_real_model_tensor_inputs".to_string(),
        json!(
            route_decision_flag(route, "metal_dispatch_real_model_tensor_inputs")
                || route_decision_flag(route, "metal_dispatch_runtime_real_model_tensor_inputs")
        ),
    );
    route_json.insert(
        "metal_model_artifacts_validated".to_string(),
        json!(route_decision_flag(
            route,
            "metal_dispatch_model_artifacts_validated"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_f32_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_f32_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_f16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_f16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_bf16_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_bf16_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_rms_norm_unsupported_binding_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_native_rms_norm_unsupported_binding_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_projection_row_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_projection_row_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_projection_row_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_rms_norm_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_rms_norm_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_ffn_activation_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_ffn_activation_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_residual_add_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_residual_add_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_residual_add_element_count"
        )),
    );
    route_json.insert(
        "mlx_metal_direct_decode_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_native_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_cpu_scale_element_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_cpu_scale_element_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_logits_group_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_logits_group_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_logits_token_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_logits_token_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_group_fallback_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_group_fallback_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_batched_group_fallback_token_count".to_string(),
        json!(route_decision_value(
            route,
            "metal_dispatch_direct_decode_batched_group_fallback_token_count"
        )),
    );
    route_json.insert(
        "metal_direct_decode_projection_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_projection_row_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_projection_row_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_rms_norm_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_rms_norm_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_rms_norm_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_ffn_activation_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_ffn_activation_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_ffn_activation_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_residual_add_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_residual_add_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_residual_add_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "metal_direct_decode_scale_mlx_metal_dispatch_share".to_string(),
        mlx_metal_coverage_ratio(
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_native_scale_element_count",
            ),
            route_decision_value(
                route,
                "metal_dispatch_direct_decode_cpu_scale_element_count",
            ),
        )
        .map_or(Value::Null, Value::from),
    );
    route_json.insert(
        "crossover_decisions".to_string(),
        Value::Object(crossover_decisions),
    );

    if let Some(value) = backend_reported_cached_prompt_tokens(route) {
        route_json.insert(
            "backend_reported_cached_prompt_tokens".to_string(),
            json!(value),
        );
    }

    Value::Object(route_json)
}

fn direct_decode_batching_opportunity_observed(step_trace: &[StepTraceEntry]) -> bool {
    step_trace.iter().any(|step| {
        step.items
            .iter()
            .filter(|item| item.mode == ax_engine_core::ExecutionMode::Decode)
            .take(2)
            .count()
            > 1
    })
}

fn annotate_route_json_with_decode_batching_opportunity(
    route_json: &mut Value,
    step_trace: &[StepTraceEntry],
) {
    let Some(route_object) = route_json.as_object_mut() else {
        return;
    };
    route_object.insert(
        "metal_direct_decode_batching_opportunity_observed".to_string(),
        json!(direct_decode_batching_opportunity_observed(step_trace)),
    );
}

fn prefix_cache_evidence_label(route: &RouteMetadata) -> String {
    match route.prefix_cache_path.as_deref() {
        Some("live_request_share") => "engine_live_request_scheduler_reuse".to_string(),
        Some("retained_prompt_prefix_cache") => "engine_retained_block_cache_metadata".to_string(),
        Some("mixed_live_and_retained") => "engine_mixed_live_and_retained_reuse".to_string(),
        Some("delegated_prompt_cache") => "backend_reported_cached_prompt_tokens".to_string(),
        Some("mixed_prefix_cache_paths") => "mixed_prefix_cache_evidence".to_string(),
        Some("metadata_lookup") | None => "none_observed".to_string(),
        Some(other) => format!("route_label:{other}"),
    }
}

fn prefix_reuse_provenance_label(route: &RouteMetadata) -> String {
    match route.prefix_cache_path.as_deref() {
        Some("live_request_share") => "mlx_live_request_share".to_string(),
        Some("retained_prompt_prefix_cache") => "mlx_retained_prompt_prefix_cache".to_string(),
        Some("mixed_live_and_retained") => "mlx_mixed_live_and_retained".to_string(),
        Some("delegated_prompt_cache") => "delegated_backend_prompt_cache".to_string(),
        Some("mixed_prefix_cache_paths") => "mixed_prefix_reuse_provenance".to_string(),
        Some("metadata_lookup") | None => "none_observed".to_string(),
        Some(other) => format!("route_label:{other}"),
    }
}

fn backend_reported_cached_prompt_tokens(route: &RouteMetadata) -> Option<u32> {
    route
        .crossover_decisions
        .iter()
        .find(|(key, _)| key == "delegated_cached_tokens")
        .map(|(_, value)| *value)
        .filter(|value| *value > 0)
}

fn serialize_runtime_metadata(
    runtime: &RuntimeConfig,
    actual_runtime: Option<&RuntimeReport>,
) -> Value {
    let mut runtime_json = json!({
        "selected_backend": runtime.resolved_backend.selected_backend,
        "support_tier": runtime.resolved_backend.support_tier,
        "resolution_policy": runtime.backend_policy.resolution_policy,
        "deterministic": runtime.deterministic,
        "max_batch_tokens": runtime.max_batch_tokens,
        "block_size_tokens": runtime.block_size_tokens,
        "flags": {
            "prefix_cache": runtime.flags.prefix_cache
        },
        "host": current_host_report(),
        "metal_toolchain": current_metal_toolchain_report()
    });

    if let Some(preset) = runtime.llama_cpp_preset.as_ref() {
        runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object")
            .insert(
                "llama_cpp_preset".to_string(),
                serde_json::to_value(preset).expect("llama.cpp preset should serialize"),
            );
    }

    if let Some(actual_runtime) = actual_runtime {
        let runtime_object = runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object");
        runtime_object.insert(
            "selected_backend".to_string(),
            serde_json::to_value(actual_runtime.selected_backend)
                .expect("runtime selected_backend should serialize"),
        );
        runtime_object.insert(
            "support_tier".to_string(),
            serde_json::to_value(actual_runtime.support_tier)
                .expect("runtime support_tier should serialize"),
        );
        runtime_object.insert(
            "resolution_policy".to_string(),
            serde_json::to_value(actual_runtime.resolution_policy)
                .expect("runtime resolution_policy should serialize"),
        );
        runtime_object.insert(
            "capabilities".to_string(),
            serde_json::to_value(&actual_runtime.capabilities)
                .expect("runtime capabilities should serialize"),
        );
        runtime_object.insert(
            "host".to_string(),
            serde_json::to_value(&actual_runtime.host).expect("runtime host should serialize"),
        );
        runtime_object.insert(
            "metal_toolchain".to_string(),
            serde_json::to_value(&actual_runtime.metal_toolchain)
                .expect("runtime metal_toolchain should serialize"),
        );
        if let Some(fallback_reason) = actual_runtime.fallback_reason.as_ref() {
            runtime_object.insert("fallback_reason".to_string(), json!(fallback_reason));
        } else {
            runtime_object.remove("fallback_reason");
        }
        if let Some(native_runtime) = actual_runtime.mlx_runtime.as_ref() {
            runtime_object.insert(
                "mlx_runtime".to_string(),
                serde_json::to_value(native_runtime).expect("MLX runtime should serialize"),
            );
        }
        if let Some(native_model) = actual_runtime.mlx_model.as_ref() {
            runtime_object.insert(
                "mlx_model".to_string(),
                serde_json::to_value(native_model).expect("MLX model should serialize"),
            );
        }
    }

    if let Some(kv_total_blocks) = runtime.kv_total_blocks {
        runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object")
            .insert("kv_total_blocks".to_string(), json!(kv_total_blocks));
    }

    if let Some(backend_adapter) = runtime_backend_adapter_for_report(runtime, actual_runtime) {
        runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object")
            .insert(
                "backend_adapter".to_string(),
                serde_json::to_value(backend_adapter).expect("backend adapter should serialize"),
            );
    }

    if actual_runtime.is_none() {
        if let Some(fallback_reason) = runtime.resolved_backend.fallback_reason.as_deref() {
            runtime_json
                .as_object_mut()
                .expect("runtime metadata should serialize as object")
                .insert("fallback_reason".to_string(), json!(fallback_reason));
        }
    }

    if actual_runtime.is_none() {
        if let Some(native_runtime) = runtime.mlx_runtime_report() {
            runtime_json
                .as_object_mut()
                .expect("runtime metadata should serialize as object")
                .insert(
                    "mlx_runtime".to_string(),
                    serde_json::to_value(native_runtime).expect("MLX runtime should serialize"),
                );
        }
    }

    if actual_runtime.is_none() {
        if let Some(native_model) = runtime.native_model_report() {
            runtime_json
                .as_object_mut()
                .expect("runtime metadata should serialize as object")
                .insert(
                    "mlx_model".to_string(),
                    serde_json::to_value(native_model).expect("MLX model should serialize"),
                );
        }
    }

    runtime_json
}

fn runtime_backend_adapter_for_report<'a>(
    runtime: &'a RuntimeConfig,
    actual_runtime: Option<&RuntimeReport>,
) -> Option<&'a BackendAdapterManifest> {
    let selected_backend = actual_runtime
        .map(|report| report.selected_backend)
        .unwrap_or(runtime.resolved_backend.selected_backend);

    runtime
        .backend_adapter
        .as_ref()
        .filter(|adapter| adapter.selected_backend() == selected_backend)
}

fn json_string_label<T: Serialize>(value: T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

fn synthetic_prompt_tokens(
    target_len: u32,
    prompt_ref: Option<&str>,
    prefix_group: Option<&str>,
    body_group: Option<&str>,
    ordinal: u32,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(target_len as usize);
    let shared_prefix_len = prefix_group.map_or(0, |_| target_len.min(64));
    let shared_seed = prefix_group.map(stable_hash32).unwrap_or(17);
    let body_seed = body_group.map(stable_hash32).unwrap_or_else(|| {
        stable_hash32(prompt_ref.unwrap_or("prompt")).wrapping_add(ordinal * 131)
    });

    for index in 0..shared_prefix_len {
        tokens.push(shared_seed.wrapping_add(index + 1));
    }

    for index in shared_prefix_len..target_len {
        tokens.push(body_seed.wrapping_add(index + 1));
    }

    tokens
}

fn synthetic_prompt_text(
    target_len: u32,
    prompt_ref: Option<&str>,
    prefix_group: Option<&str>,
    body_group: Option<&str>,
    ordinal: u32,
) -> String {
    let shared_prefix_len = prefix_group.map_or(0, |_| target_len.min(64));
    let shared_seed = prefix_group.map(stable_hash32).unwrap_or(17);
    let body_seed = body_group.map(stable_hash32).unwrap_or_else(|| {
        stable_hash32(prompt_ref.unwrap_or("prompt")).wrapping_add(ordinal * 131)
    });

    let mut parts = Vec::with_capacity(target_len as usize);
    for index in 0..shared_prefix_len {
        parts.push(format!("shared{}_{}", shared_seed, index + 1));
    }

    for index in shared_prefix_len..target_len {
        parts.push(format!("body{}_{}", body_seed, index + 1));
    }

    parts.join(" ")
}

fn synthetic_text_output_tokens(text: &str, target_len: u32) -> Vec<u32> {
    synthetic_prompt_tokens(target_len, Some(text), None, None, 0)
}

fn replay_prompt_target(prompt_ref: &str) -> u32 {
    if prompt_ref.contains("long") {
        1024
    } else if prompt_ref.contains("variant") {
        320
    } else {
        256
    }
}

fn replay_events_from_manifest(manifest: &BenchmarkManifest) -> Result<Vec<ReplayEvent>, CliError> {
    let events = if manifest.events.is_empty() {
        return Err(CliError::Contract(
            "replay manifest must contain an events array".to_string(),
        ));
    } else {
        &manifest.events
    };

    let mut replay_events = Vec::with_capacity(events.len());
    let mut next_request_id = 1u64;
    let mut request_ids_by_external = BTreeMap::new();

    for event in events {
        match event.kind {
            ReplayEventKind::Submit => {
                let external_id = event
                    .request_id
                    .as_deref()
                    .ok_or_else(|| {
                        CliError::Contract("submit event missing request_id".to_string())
                    })?
                    .to_string();
                let request_id = RequestId(next_request_id);
                next_request_id += 1;
                request_ids_by_external.insert(external_id.clone(), request_id);

                let prompt_ref = event.prompt_ref.as_deref().ok_or_else(|| {
                    CliError::Contract("submit event missing prompt_ref".to_string())
                })?;
                let prefix_group = event.prefix_group.as_deref();
                let body_group = event.body_group.as_deref();
                let prompt_len = replay_prompt_target(prompt_ref);

                replay_events.push(ReplayEvent::Submit {
                    t_ms: event.t_ms,
                    spec: SyntheticRequestSpec {
                        external_id,
                        request_id,
                        arrival_sequence: SequenceNo(request_id.0),
                        model_family: manifest.model.family.clone(),
                        prompt_token_target: prompt_len,
                        input_tokens: synthetic_prompt_tokens(
                            prompt_len,
                            Some(prompt_ref),
                            prefix_group,
                            body_group,
                            request_id.0 as u32,
                        ),
                        input_text: None,
                        max_output_tokens: event.output_tokens_target.ok_or_else(|| {
                            CliError::Contract(
                                "submit event missing output_tokens_target".to_string(),
                            )
                        })?,
                        sampling_params: sampling_from_manifest(manifest)?,
                        metadata: prefix_group.map(str::to_string),
                    },
                });
            }
            ReplayEventKind::Cancel => {
                let external_id = event.request_id.as_deref().ok_or_else(|| {
                    CliError::Contract("cancel event missing request_id".to_string())
                })?;
                let request_id = *request_ids_by_external.get(external_id).ok_or_else(|| {
                    CliError::Contract(format!(
                        "cancel event references request_id before submit: {external_id}"
                    ))
                })?;
                replay_events.push(ReplayEvent::Cancel {
                    t_ms: event.t_ms,
                    request_id,
                });
            }
        }
    }

    for window in replay_events.windows(2) {
        if window[1].t_ms() < window[0].t_ms() {
            return Err(CliError::Contract(format!(
                "replay events must be sorted by t_ms: found t_ms={} after t_ms={}",
                window[1].t_ms(),
                window[0].t_ms()
            )));
        }
    }

    Ok(replay_events)
}

fn validate_llama_cpp_benchmark_runtime(
    manifest: &BenchmarkManifest,
    runtime: &RuntimeConfig,
) -> Result<(), CliError> {
    match runtime.backend_adapter.as_ref() {
        Some(adapter) if adapter.supports_stepwise_benchmark() => Ok(()),
        Some(adapter) if adapter.supports_blocking_benchmark() => {
            if manifest.class != ManifestClass::Scenario {
                return Err(CliError::Contract(
                    "blocking llama.cpp benchmark adapters currently support scenario manifests only"
                        .to_string(),
                ));
            }
            let shape = manifest.shape.as_ref().ok_or_else(|| {
                CliError::Contract("scenario manifest must contain a shape object".to_string())
            })?;
            if shape.concurrency != 1 {
                return Err(CliError::Contract(
                    "blocking llama.cpp benchmark adapters currently require shape.concurrency=1"
                        .to_string(),
                ));
            }
            if manifest.checks.require_prefix_reuse {
                return Err(CliError::Contract(
                    "blocking llama.cpp benchmark adapters do not support require_prefix_reuse"
                        .to_string(),
                ));
            }
            Ok(())
        }
        Some(BackendAdapterManifest::LlamaCppCli { .. }) => Err(CliError::Contract(
            "ax-engine-bench llama.cpp execution does not support CLI adapters; use a server-backed llama.cpp adapter instead".to_string(),
        )),
        Some(_) => Err(CliError::Contract(
            "llama.cpp benchmark execution requires a supported backend adapter".to_string(),
        )),
        None => Err(CliError::Contract(
            "llama.cpp benchmark execution requires runtime.backend_adapter".to_string(),
        )),
    }
}

fn stable_hash32(input: &str) -> u32 {
    let mut hash = 2_166_136_261u32;
    for byte in input.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

fn request_snapshot_for_bench(
    engine: &EngineCore,
    request_id: RequestId,
) -> Result<RequestSnapshot, CliError> {
    engine
        .request_manager()
        .snapshot(request_id)
        .ok_or_else(|| CliError::Runtime(format!("missing request snapshot {:?}", request_id)))
}

fn has_live_requests(engine: &EngineCore, request_ids: &[RequestId]) -> Result<bool, CliError> {
    for request_id in request_ids {
        let snapshot = request_snapshot_for_bench(engine, *request_id)?;
        if !snapshot.state.is_terminal() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn llama_cpp_reports_for_session(
    session: &EngineSession,
    request_ids: &[RequestId],
) -> Result<BTreeMap<RequestId, SessionRequestReport>, CliError> {
    let mut reports = BTreeMap::new();

    for request_id in request_ids {
        let report = session.request_report(request_id.0).ok_or_else(|| {
            CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
        })?;
        reports.insert(*request_id, report);
    }

    Ok(reports)
}

fn llama_cpp_reports_changed(
    before: &BTreeMap<RequestId, SessionRequestReport>,
    after: &BTreeMap<RequestId, SessionRequestReport>,
) -> bool {
    before != after
}

fn llama_cpp_session_has_live_requests(
    session: &EngineSession,
    request_ids: &[RequestId],
) -> Result<bool, CliError> {
    for request_id in request_ids {
        let report = session.request_report(request_id.0).ok_or_else(|| {
            CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
        })?;
        if !llama_cpp_request_is_terminal(report.state) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn load_json_value(path: &Path) -> Result<Value, CliError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        CliError::Runtime(format!("failed to read {}: {error}", path.display()))
    })?;

    serde_json::from_str(&raw).map_err(|error| {
        CliError::Contract(format!(
            "failed to parse {} as JSON: {error}",
            path.display()
        ))
    })
}

fn load_optional_json_value(path: &Path) -> Result<Option<Value>, CliError> {
    if !path.is_file() {
        return Ok(None);
    }
    load_json_value(path).map(Some)
}

fn nested_value<'a>(json: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = json;
    for component in path {
        current = current.get(*component)?;
    }
    Some(current)
}

fn nested_string<'a>(json: &'a Value, path: &[&str]) -> Result<&'a str, CliError> {
    nested_value(json, path)
        .and_then(Value::as_str)
        .ok_or_else(|| CliError::Contract(format!("missing string field {}", path.join("."))))
}

fn metric_number(metrics_json: &Value, key: &str) -> Result<f64, CliError> {
    metrics_json
        .get("metrics")
        .and_then(|metrics| metrics.get(key))
        .and_then(Value::as_f64)
        .ok_or_else(|| CliError::Contract(format!("metrics artifact missing numeric field {key}")))
}

fn percentage_delta(baseline: f64, candidate: f64) -> f64 {
    if baseline.abs() < f64::EPSILON {
        if candidate.abs() < f64::EPSILON {
            0.0
        } else {
            f64::INFINITY.copysign(candidate)
        }
    } else {
        ((candidate - baseline) / baseline) * 100.0
    }
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), CliError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        CliError::Runtime(format!(
            "failed to serialize JSON for {}: {error}",
            path.display()
        ))
    })?;
    fs::write(path, bytes)
        .map_err(|error| CliError::Runtime(format!("failed to write {}: {error}", path.display())))
}

fn unix_timestamp_secs() -> Result<u64, CliError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| CliError::Runtime(format!("system clock error: {error}")))
}

fn sanitize_component(input: &str) -> String {
    input
        .chars()
        .map(|char| match char {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => char,
            _ => '-',
        })
        .collect()
}

fn usage() -> String {
    let text = r#"AX Engine v4 benchmark CLI

Usage:
  ax-engine-bench generate [--model-id <id>] (--prompt <text> | --tokens <ids>) [--max-output-tokens <n>] [--mlx] [--support-tier <tier>] [--llama-cli-path <path>] [--llama-model-path <path>] [--llama-server-url <url>] [--mlx-lm-server-url <url>] [--mlx-model-artifacts-dir <path>] [--json]
  ax-engine-bench stream [--model-id <id>] (--prompt <text> | --tokens <ids>) [--max-output-tokens <n>] [--mlx] [--support-tier <tier>] [--llama-cli-path <path>] [--llama-model-path <path>] [--llama-server-url <url>] [--mlx-lm-server-url <url>] [--mlx-model-artifacts-dir <path>] [--json]
  ax-engine-bench scenario --manifest <path> --output-root <path>
  ax-engine-bench replay --manifest <path> --output-root <path>
  ax-engine-bench autotune --manifest <path> --output-root <path> [--iterations <n>] [--exploration-weight <value>] [--max-batch-token-options <list>] [--kv-total-block-options <list>] [--prefix-cache-options <list>] [--disable-history]
  ax-engine-bench compare --baseline <path> --candidate <path> --output-root <path>
  ax-engine-bench matrix-compare --baseline <path> --candidate <path> --output-root <path>
  ax-engine-bench baseline --source <path> --name <name> --output-root <path>
  ax-engine-bench matrix --manifest <path> --output-root <path>
  ax-engine-bench doctor [--json] [--mlx-model-artifacts-dir <path>]
  ax-engine-bench metal-build [--manifest <path>] [--output-dir <path>]
"#;

    text.to_string()
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ManifestClass {
    Scenario,
    Replay,
}

impl ManifestClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::Scenario => "scenario",
            Self::Replay => "replay",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BenchmarkManifest {
    schema_version: String,
    id: String,
    class: ManifestClass,
    scenario: String,
    model: ModelManifest,
    runtime: RuntimeManifest,
    sampling: SamplingManifest,
    #[serde(default)]
    shape: Option<ScenarioShape>,
    #[serde(default)]
    events: Vec<ReplayEventManifest>,
    #[serde(default)]
    source: Option<ManifestSource>,
    #[serde(default)]
    checks: ManifestChecks,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MatrixClass {
    ScenarioMatrix,
}

impl MatrixClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::ScenarioMatrix => "scenario_matrix",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BenchmarkMatrixManifest {
    schema_version: String,
    id: String,
    class: MatrixClass,
    members: Vec<BenchmarkMatrixMember>,
    #[serde(default)]
    notes: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BenchmarkMatrixMember {
    manifest: String,
    #[serde(default)]
    label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelManifest {
    family: String,
    revision: String,
    quant: String,
    tokenizer_revision: String,
    chat_template_revision: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RuntimeManifest {
    selected_backend: SelectedBackend,
    support_tier: SupportTier,
    resolution_policy: ResolutionPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend_adapter: Option<BackendAdapterManifest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    llama_cpp_preset: Option<LlamaCppPresetManifest>,
    #[serde(default = "default_true")]
    deterministic: bool,
    #[serde(default)]
    max_batch_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kv_total_blocks: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mlx_model_artifacts_dir: Option<PathBuf>,
    #[serde(default)]
    flags: RuntimeFlags,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct LlamaCppPresetManifest {
    #[serde(default = "default_llama_cpp_preset_name")]
    name: String,
    #[serde(default = "default_llama_cpp_parallel_slots")]
    parallel_slots: u32,
    #[serde(default)]
    continuous_batching: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logical_batch_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    physical_batch_size: Option<u32>,
    #[serde(default)]
    cache_prompt: bool,
    #[serde(default)]
    slot_save_path: Option<String>,
    #[serde(default)]
    slot_restore_path: Option<String>,
    #[serde(default = "default_llama_cpp_speculative_decode_mode")]
    speculative_decode_mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metrics_endpoint: Option<String>,
}

impl Default for LlamaCppPresetManifest {
    fn default() -> Self {
        Self {
            name: default_llama_cpp_preset_name(),
            parallel_slots: default_llama_cpp_parallel_slots(),
            continuous_batching: true,
            logical_batch_size: Some(2048),
            physical_batch_size: Some(512),
            cache_prompt: true,
            slot_save_path: None,
            slot_restore_path: None,
            speculative_decode_mode: default_llama_cpp_speculative_decode_mode(),
            metrics_endpoint: Some("server:/metrics".to_string()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BackendAdapterManifest {
    LlamaCppCli {
        cli_path: PathBuf,
        model_path: PathBuf,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        extra_args: Vec<String>,
    },
    LlamaCppServerCompletion {
        server_url: String,
    },
}

impl BackendAdapterManifest {
    fn selected_backend(&self) -> SelectedBackend {
        match self {
            Self::LlamaCppCli { .. } | Self::LlamaCppServerCompletion { .. } => {
                SelectedBackend::LlamaCpp
            }
        }
    }

    fn to_sdk_config(&self) -> LlamaCppConfig {
        match self {
            Self::LlamaCppCli {
                cli_path,
                model_path,
                extra_args,
            } => {
                let mut config = ax_engine_sdk::LlamaCppCliConfig::new(cli_path, model_path);
                config.extra_args = extra_args.clone();
                LlamaCppConfig::Cli(config)
            }
            Self::LlamaCppServerCompletion { server_url } => {
                LlamaCppConfig::server_completion(server_url.clone())
            }
        }
    }

    fn supports_stepwise_benchmark(&self) -> bool {
        matches!(self, Self::LlamaCppServerCompletion { .. })
    }

    fn supports_blocking_benchmark(&self) -> bool {
        matches!(self, Self::LlamaCppServerCompletion { .. })
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct RuntimeFlags {
    #[serde(default)]
    prefix_cache: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SamplingManifest {
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    top_k: u32,
    #[serde(default)]
    seed: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ScenarioShape {
    input_tokens_target: u32,
    output_tokens_target: u32,
    concurrency: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ManifestSource {
    #[serde(default)]
    dataset_id: Option<String>,
    #[serde(default)]
    prompt_set: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct ManifestChecks {
    #[serde(default)]
    expect_deterministic: bool,
    #[serde(default)]
    require_prefix_reuse: bool,
    #[serde(default)]
    require_no_allocator_churn_failure: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReplayEventKind {
    Submit,
    Cancel,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReplayEventManifest {
    t_ms: u64,
    #[serde(rename = "type")]
    kind: ReplayEventKind,
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    prompt_ref: Option<String>,
    #[serde(default)]
    output_tokens_target: Option<u32>,
    #[serde(default)]
    prefix_group: Option<String>,
    #[serde(default)]
    body_group: Option<String>,
}

const fn default_true() -> bool {
    true
}

const fn default_top_p() -> f32 {
    1.0
}

fn default_llama_cpp_preset_name() -> String {
    "safe_stepwise_server".to_string()
}

const fn default_llama_cpp_parallel_slots() -> u32 {
    1
}

fn default_llama_cpp_speculative_decode_mode() -> String {
    "disabled".to_string()
}

#[derive(Clone, Debug)]
struct RuntimeConfig {
    deterministic: bool,
    max_batch_tokens: u32,
    block_size_tokens: u32,
    kv_total_blocks: Option<u32>,
    flags: RuntimeFlags,
    llama_cpp_preset: Option<LlamaCppPresetManifest>,
    backend_policy: BackendPolicy,
    resolved_backend: ResolvedBackend,
    backend_adapter: Option<BackendAdapterManifest>,
    mlx_model_artifacts_dir: Option<PathBuf>,
    mlx_model_artifacts_source: Option<NativeModelArtifactsSource>,
}

impl RuntimeConfig {
    fn uses_mlx_runtime(&self) -> bool {
        self.resolved_backend.selected_backend.is_mlx()
    }

    fn uses_metal_bringup_runtime(&self) -> bool {
        false
    }

    fn mlx_runtime_report(&self) -> Option<NativeRuntimeReport> {
        if !self.uses_mlx_runtime() {
            return None;
        }

        None
    }

    fn native_model_report(&self) -> Option<NativeModelReport> {
        if !self.uses_mlx_runtime() {
            return None;
        }

        let model_dir = self.mlx_model_artifacts_dir.as_ref()?;
        if model_dir.is_file() {
            return None;
        }
        let summary = NativeModelArtifacts::from_dir(model_dir).ok()?.summary();
        let source = self
            .mlx_model_artifacts_source
            .unwrap_or(NativeModelArtifactsSource::ExplicitConfig);
        let binding = self.resolved_native_model_binding_summary();

        Some(NativeModelReport::from_summary(source, summary, binding))
    }

    fn resolved_native_model_binding_summary(&self) -> Option<NativeModelBindingSummary> {
        None
    }

    fn tool_mode(&self) -> &'static str {
        if self.uses_metal_bringup_runtime() {
            "engine_bringup_runtime"
        } else if self.uses_mlx_runtime() {
            "mlx_runtime"
        } else if self
            .backend_adapter
            .as_ref()
            .is_some_and(BackendAdapterManifest::supports_stepwise_benchmark)
        {
            "llama_cpp_stepwise_runtime"
        } else {
            "llama_cpp_blocking_runtime"
        }
    }
}

#[derive(Clone, Debug)]
struct SyntheticRequestSpec {
    external_id: String,
    request_id: RequestId,
    arrival_sequence: SequenceNo,
    model_family: String,
    prompt_token_target: u32,
    input_tokens: Vec<u32>,
    input_text: Option<String>,
    max_output_tokens: u32,
    sampling_params: SamplingParams,
    metadata: Option<String>,
}

impl SyntheticRequestSpec {
    fn into_submission(self) -> RequestSubmission {
        RequestSubmission {
            request_id: self.request_id,
            model_id: ax_engine_core::ModelId(self.model_family),
            input_tokens: self.input_tokens,
            sampling_params: self.sampling_params,
            max_output_tokens: self.max_output_tokens,
            arrival_sequence: self.arrival_sequence,
            metadata: self.metadata,
        }
    }

    fn with_input_text(mut self, input_text: String) -> Self {
        self.input_tokens.clear();
        self.input_text = Some(input_text);
        self
    }
}

#[derive(Clone, Debug)]
enum ReplayEvent {
    Submit {
        t_ms: u64,
        spec: SyntheticRequestSpec,
    },
    Cancel {
        t_ms: u64,
        request_id: RequestId,
    },
}

impl ReplayEvent {
    fn t_ms(&self) -> u64 {
        match self {
            Self::Submit { t_ms, .. } | Self::Cancel { t_ms, .. } => *t_ms,
        }
    }
}

#[derive(Clone, Debug)]
struct GateStatus {
    passed: bool,
    reason: Option<String>,
}

impl GateStatus {
    fn pass() -> Self {
        Self {
            passed: true,
            reason: None,
        }
    }

    fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Clone, Debug)]
struct RuntimeResult {
    tool_mode: &'static str,
    runtime: RuntimeConfig,
    observation: RuntimeObservation,
    correctness: GateStatus,
    determinism: GateStatus,
}

impl RuntimeResult {
    fn status_label(&self) -> &'static str {
        if self.correctness.passed && self.determinism.passed {
            "ok"
        } else {
            "completed_with_failures"
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RuntimeObservation {
    step_count: u64,
    e2e_latency_ms: u64,
    ttft_ms: Option<u64>,
    prefill_tokens: u64,
    decode_tokens: u64,
    prefill_steps: u64,
    decode_steps: u64,
    total_scheduled_tokens: u64,
    total_selected_requests: u64,
    prefix_hits: u64,
    memory_blocked_steps: u64,
    memory_blocked_request_events: u64,
    total_cpu_time_us: u64,
    total_runner_time_us: u64,
    total_prefill_runner_time_us: u64,
    total_decode_runner_time_us: u64,
    kv_peak_blocks: u32,
    memory_peak_mb: f64,
    evictions: u64,
    cleanup_count: usize,
    llama_cpp_processing_request_events: u64,
    llama_cpp_deferred_request_events: u64,
    route_metadata: RouteMetadata,
    step_trace: Vec<StepTraceEntry>,
    final_requests: Vec<FinalRequestState>,
    cancelled_requests: BTreeSet<RequestId>,
    digest: Value,
    execution_plans_seen: BTreeSet<String>,
    attention_routes_seen: BTreeSet<String>,
    kv_modes_seen: BTreeSet<String>,
    barrier_modes_seen: BTreeSet<String>,
    runtime_report: Option<RuntimeReport>,
}

fn proportional_time_us(total_us: u64, part_tokens: u64, total_tokens: u64) -> u64 {
    if total_us == 0 || part_tokens == 0 || total_tokens == 0 {
        return 0;
    }
    let value =
        u128::from(total_us).saturating_mul(u128::from(part_tokens)) / u128::from(total_tokens);
    value.min(u128::from(u64::MAX)) as u64
}

fn tokens_per_second_from_micros(tokens: u64, elapsed_us: u64) -> f64 {
    if tokens == 0 || elapsed_us == 0 {
        0.0
    } else {
        (tokens as f64 * 1_000_000.0) / elapsed_us as f64
    }
}

impl RuntimeObservation {
    fn observe_step(
        &mut self,
        engine: &EngineCore,
        outcome: &ax_engine_core::EngineStepOutcome,
        current_time_ms: u64,
    ) {
        self.step_count += 1;
        self.total_selected_requests += outcome.schedule_plan.selected_requests.len() as u64;
        self.total_scheduled_tokens += u64::from(outcome.metrics.scheduled_tokens);
        self.prefix_hits += u64::from(outcome.metrics.prefix_hits);
        if !outcome.schedule_plan.memory_blocked_requests.is_empty() {
            self.memory_blocked_steps += 1;
            self.memory_blocked_request_events +=
                outcome.schedule_plan.memory_blocked_requests.len() as u64;
        }
        self.total_cpu_time_us += outcome.metrics.cpu_time_us;
        self.total_runner_time_us += outcome.metrics.runner_time_us;
        self.evictions = self
            .evictions
            .saturating_add(u64::from(outcome.metrics.evictions));
        self.kv_peak_blocks = self
            .kv_peak_blocks
            .max(engine.kv_manager().used_block_count());
        self.cleanup_count += outcome.cleanup_results.len();

        if self.ttft_ms.is_none() && outcome.metrics.ttft_events > 0 {
            self.ttft_ms = Some(current_time_ms);
        }

        if let Some(batch) = &outcome.schedule_plan.execution_batch {
            let mut saw_prefill = false;
            let mut saw_decode = false;
            let mut step_prefill_tokens = 0u64;
            let mut step_decode_tokens = 0u64;
            for item in &batch.items {
                match item.mode {
                    ax_engine_core::ExecutionMode::Prefill => {
                        let scheduled_tokens = u64::from(item.scheduled_token_count);
                        self.prefill_tokens += scheduled_tokens;
                        step_prefill_tokens += scheduled_tokens;
                        saw_prefill = true;
                    }
                    ax_engine_core::ExecutionMode::Decode => {
                        let scheduled_tokens = u64::from(item.scheduled_token_count);
                        self.decode_tokens += scheduled_tokens;
                        step_decode_tokens += scheduled_tokens;
                        saw_decode = true;
                    }
                }
            }
            let step_phase_tokens = step_prefill_tokens.saturating_add(step_decode_tokens);
            if step_phase_tokens > 0 {
                self.total_prefill_runner_time_us = self
                    .total_prefill_runner_time_us
                    .saturating_add(proportional_time_us(
                        outcome.metrics.runner_time_us,
                        step_prefill_tokens,
                        step_phase_tokens,
                    ));
                self.total_decode_runner_time_us =
                    self.total_decode_runner_time_us
                        .saturating_add(proportional_time_us(
                            outcome.metrics.runner_time_us,
                            step_decode_tokens,
                            step_phase_tokens,
                        ));
            }
            if saw_prefill {
                self.prefill_steps += 1;
            }
            if saw_decode {
                self.decode_steps += 1;
            }
        }

        if let Some(runner_output) = &outcome.runner_output {
            self.merge_route_metadata(&runner_output.route_metadata);
        }

        self.step_trace
            .push(StepTraceEntry::capture(engine, outcome, current_time_ms));
    }

    fn observe_llama_cpp_session_step(
        &mut self,
        reports_before: &BTreeMap<RequestId, SessionRequestReport>,
        reports_after: &BTreeMap<RequestId, SessionRequestReport>,
        step: &EngineStepReport,
        current_time_ms: u64,
    ) {
        self.step_count += 1;
        self.total_selected_requests += u64::from(step.scheduled_requests);
        self.total_scheduled_tokens += u64::from(step.scheduled_tokens);
        self.prefix_hits += u64::from(step.prefix_hits);
        self.decode_tokens += u64::from(step.scheduled_tokens);
        if step.scheduled_tokens > 0 {
            self.decode_steps += 1;
        }
        self.total_cpu_time_us += step.cpu_time_us;
        self.total_runner_time_us += step.runner_time_us;
        self.evictions = self.evictions.saturating_add(u64::from(step.evictions));
        self.kv_peak_blocks = self.kv_peak_blocks.max(step.kv_usage_blocks);

        if self.ttft_ms.is_none() && step.ttft_events > 0 {
            self.ttft_ms = Some(current_time_ms);
        }

        let mut selected_request_ids = Vec::new();
        let mut items = Vec::new();
        let mut step_route_metadata = RouteMetadata::empty();
        let mut saw_prefill_progress = false;
        let live_request_ids = reports_after
            .iter()
            .filter_map(|(request_id, report)| {
                (!llama_cpp_request_is_terminal(report.state)).then_some(*request_id)
            })
            .collect::<BTreeSet<_>>();

        for (request_id, report_after) in reports_after {
            let report_before = reports_before.get(request_id);
            let output_delta = report_before
                .map(|report| report_after.output_len.saturating_sub(report.output_len))
                .unwrap_or(report_after.output_len);
            let prompt_delta = report_before
                .map(|report| {
                    report_after
                        .processed_prompt_tokens
                        .saturating_sub(report.processed_prompt_tokens)
                })
                .unwrap_or(report_after.processed_prompt_tokens);
            let state_changed = report_before
                .map(|report| report.state != report_after.state)
                .unwrap_or(true);

            if output_delta == 0 && prompt_delta == 0 && !state_changed {
                continue;
            }

            selected_request_ids.push(*request_id);
            if prompt_delta > 0 {
                saw_prefill_progress = true;
            }
            if output_delta > 0 {
                items.push(StepTraceItem::capture_llama_cpp_decode(
                    *request_id,
                    report_after.output_len,
                    output_delta,
                ));
            }

            let route_metadata = route_metadata_from_generate_route(&report_after.route);
            self.merge_route_metadata(&route_metadata);
            merge_step_route_metadata(&mut step_route_metadata, &route_metadata);
        }

        if saw_prefill_progress {
            self.prefill_steps += 1;
        }
        let selected_request_set = selected_request_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let deferred_request_ids = live_request_ids
            .difference(&selected_request_set)
            .copied()
            .collect::<Vec<_>>();
        self.llama_cpp_processing_request_events = self
            .llama_cpp_processing_request_events
            .saturating_add(live_request_ids.len() as u64);
        self.llama_cpp_deferred_request_events = self
            .llama_cpp_deferred_request_events
            .saturating_add(deferred_request_ids.len() as u64);

        self.step_trace
            .push(StepTraceEntry::capture_llama_cpp_shared(
                selected_request_ids,
                deferred_request_ids,
                step,
                current_time_ms,
                step_route_metadata,
                items,
            ));
    }

    fn finalize(
        &mut self,
        engine: &EngineCore,
        request_ids: &[RequestId],
        external_ids: BTreeMap<RequestId, String>,
        current_time_ms: u64,
        block_size_tokens: u32,
    ) -> Result<(), CliError> {
        self.e2e_latency_ms = current_time_ms;
        self.memory_peak_mb =
            (f64::from(self.kv_peak_blocks) * f64::from(block_size_tokens) * 16.0) / 1024.0;

        let mut final_requests = Vec::new();
        for request_id in request_ids {
            let snapshot = request_snapshot_for_bench(engine, *request_id)?;
            final_requests.push(FinalRequestState {
                external_id: external_ids
                    .get(request_id)
                    .cloned()
                    .unwrap_or_else(|| format!("req-{}", request_id.0)),
                request_id: *request_id,
                state: format!("{:?}", snapshot.state),
                processed_prompt_tokens: snapshot.processed_prompt_tokens,
                generated_tokens: snapshot.generated_tokens,
                cancel_requested: snapshot.cancel_requested,
                last_error: snapshot.last_error,
            });
        }
        final_requests.sort_by_key(|request| request.request_id);
        self.digest = json!({
            "step_count": self.step_count,
            "ttft_ms": Value::Null,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "prefix_hits": self.prefix_hits,
            "memory_blocked_steps": self.memory_blocked_steps,
            "memory_blocked_request_events": self.memory_blocked_request_events,
            "cleanup_count": self.cleanup_count,
            "llama_cpp_processing_request_events": self.llama_cpp_processing_request_events,
            "llama_cpp_deferred_request_events": self.llama_cpp_deferred_request_events,
            "route": serialize_route_metadata(&self.route_metadata),
            "requests": final_requests.iter().map(FinalRequestState::digest_json).collect::<Vec<_>>()
        });
        self.final_requests = final_requests;
        Ok(())
    }

    fn finalize_llama_cpp(
        &mut self,
        final_reports: Vec<(SyntheticRequestSpec, SessionRequestReport)>,
        elapsed_ms: u64,
    ) {
        self.e2e_latency_ms = elapsed_ms;
        self.memory_peak_mb = 0.0;
        self.prefill_tokens = final_reports
            .iter()
            .map(|(_, report)| u64::from(report.prompt_len))
            .sum();
        let delegated_cached_tokens_total = final_reports
            .iter()
            .map(|(_, report)| delegated_cached_tokens_from_generate_route(&report.route))
            .sum::<u32>();
        if delegated_cached_tokens_total > 0 {
            if self.route_metadata.prefix_cache_path.is_none() {
                self.route_metadata.prefix_cache_path = Some("delegated_prompt_cache".to_string());
            }
            upsert_route_decision(
                &mut self.route_metadata.crossover_decisions,
                "delegated_cached_tokens",
                delegated_cached_tokens_total,
            );
        }
        let mut final_requests = final_reports
            .into_iter()
            .map(|(spec, report)| FinalRequestState {
                external_id: spec.external_id,
                request_id: spec.request_id,
                state: llama_cpp_final_request_state_label(report.state).to_string(),
                processed_prompt_tokens: report.processed_prompt_tokens,
                generated_tokens: report.output_tokens.clone(),
                cancel_requested: report.cancel_requested,
                last_error: report.last_error.clone(),
            })
            .collect::<Vec<_>>();
        final_requests.sort_by_key(|request| request.request_id);
        self.final_requests = final_requests;
        self.digest = json!({
            "step_count": self.step_count,
            "ttft_ms": Value::Null,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "prefix_hits": self.prefix_hits,
            "memory_blocked_steps": self.memory_blocked_steps,
            "memory_blocked_request_events": self.memory_blocked_request_events,
            "cleanup_count": self.cleanup_count,
            "llama_cpp_processing_request_events": self.llama_cpp_processing_request_events,
            "llama_cpp_deferred_request_events": self.llama_cpp_deferred_request_events,
            "route": serialize_route_metadata(&self.route_metadata),
            "requests": self
                .final_requests
                .iter()
                .map(FinalRequestState::digest_json)
                .collect::<Vec<_>>()
        });
    }

    fn prefix_hit_rate(&self) -> f64 {
        if self.total_selected_requests == 0 {
            0.0
        } else {
            (self.prefix_hits as f64 / self.total_selected_requests as f64) * 100.0
        }
    }

    /// Prefill throughput from runner execution time when available.
    /// Legacy synthetic/llama.cpp paths fall back to the old step proxy.
    fn prefill_tok_s(&self) -> f64 {
        if self.prefill_tokens == 0 {
            0.0
        } else if self.total_prefill_runner_time_us > 0 {
            tokens_per_second_from_micros(self.prefill_tokens, self.total_prefill_runner_time_us)
        } else if self.prefill_steps == 0 {
            0.0
        } else {
            (self.prefill_tokens as f64 * 1000.0) / self.prefill_steps as f64
        }
    }

    /// Decode throughput from runner execution time when available.
    /// Returns 0.0 when the runtime did not report decode runner time.
    fn decode_tok_s(&self) -> f64 {
        if self.decode_tokens == 0 {
            0.0
        } else if self.total_decode_runner_time_us > 0 {
            tokens_per_second_from_micros(self.decode_tokens, self.total_decode_runner_time_us)
        } else {
            0.0
        }
    }

    fn cpu_time_per_token_us(&self) -> f64 {
        if self.total_scheduled_tokens == 0 {
            0.0
        } else {
            self.total_cpu_time_us as f64 / self.total_scheduled_tokens as f64
        }
    }

    fn runner_time_per_token_us(&self) -> f64 {
        if self.total_scheduled_tokens == 0 {
            0.0
        } else {
            self.total_runner_time_us as f64 / self.total_scheduled_tokens as f64
        }
    }

    fn runner_time_share_pct(&self) -> f64 {
        if self.total_cpu_time_us == 0 {
            0.0
        } else {
            (self.total_runner_time_us as f64 / self.total_cpu_time_us as f64) * 100.0
        }
    }

    fn scheduled_tokens_per_step(&self) -> f64 {
        if self.step_count == 0 {
            0.0
        } else {
            self.total_scheduled_tokens as f64 / self.step_count as f64
        }
    }

    fn request_state(&self, request_id: RequestId, expected_state: &str) -> bool {
        self.final_requests
            .iter()
            .find(|request| request.request_id == request_id)
            .map(|request| request.state == expected_state)
            .unwrap_or(false)
    }

    fn replay_status(&self) -> &'static str {
        if self.cancelled_requests.is_empty() {
            "not_applicable"
        } else if self
            .cancelled_requests
            .iter()
            .all(|request_id| self.request_state(*request_id, "Cancelled"))
        {
            "pass"
        } else {
            "fail"
        }
    }

    fn churn_status(&self) -> &'static str {
        if self
            .final_requests
            .iter()
            .any(|request| request.state == "Failed")
        {
            "fail"
        } else {
            "pass"
        }
    }

    fn merge_route_metadata(&mut self, route_metadata: &RouteMetadata) {
        record_route_variant(
            &mut self.execution_plans_seen,
            route_metadata.execution_plan.as_deref(),
        );
        record_route_variant(
            &mut self.attention_routes_seen,
            route_metadata.attention_route.as_deref(),
        );
        record_route_variant(&mut self.kv_modes_seen, route_metadata.kv_mode.as_deref());
        record_route_variant(
            &mut self.barrier_modes_seen,
            route_metadata.barrier_mode.as_deref(),
        );

        self.route_metadata.execution_plan =
            aggregate_route_variant(&self.execution_plans_seen, "mixed_step_plans");
        self.route_metadata.attention_route =
            aggregate_route_variant(&self.attention_routes_seen, "mixed_attention_routes");
        self.route_metadata.kv_mode =
            aggregate_route_variant(&self.kv_modes_seen, "mixed_kv_modes");
        self.route_metadata.barrier_mode =
            aggregate_route_variant(&self.barrier_modes_seen, "mixed_barrier_modes");

        let mut cumulative_decisions = self
            .route_metadata
            .crossover_decisions
            .iter()
            .cloned()
            .collect::<BTreeMap<_, _>>();
        for (key, value) in &route_metadata.crossover_decisions {
            let entry = cumulative_decisions.entry(key.clone()).or_insert(0);
            *entry = entry.saturating_add(*value);
        }
        cumulative_decisions.insert(
            "execution_plan_variants".into(),
            self.execution_plans_seen.len() as u32,
        );
        cumulative_decisions.insert(
            "attention_route_variants".into(),
            self.attention_routes_seen.len() as u32,
        );
        cumulative_decisions.insert("kv_mode_variants".into(), self.kv_modes_seen.len() as u32);
        cumulative_decisions.insert(
            "barrier_mode_variants".into(),
            self.barrier_modes_seen.len() as u32,
        );
        self.route_metadata.crossover_decisions = cumulative_decisions.into_iter().collect();

        let live_share_hits = self
            .route_metadata
            .crossover_decisions
            .iter()
            .find(|(key, _)| key == "live_share_hits")
            .map(|(_, value)| *value)
            .unwrap_or(0);
        let retained_cache_hits = self
            .route_metadata
            .crossover_decisions
            .iter()
            .find(|(key, _)| key == "retained_cache_hits")
            .map(|(_, value)| *value)
            .unwrap_or(0);

        self.route_metadata.prefix_cache_path = Some(
            match (live_share_hits > 0, retained_cache_hits > 0) {
                (true, true) => "mixed_live_and_retained",
                (true, false) => "live_request_share",
                (false, true) => "retained_prompt_prefix_cache",
                (false, false) => route_metadata
                    .prefix_cache_path
                    .as_deref()
                    .unwrap_or("metadata_lookup"),
            }
            .to_string(),
        );
    }
}

fn llama_cpp_request_is_terminal(state: SessionRequestState) -> bool {
    matches!(
        state,
        SessionRequestState::Finished
            | SessionRequestState::Cancelled
            | SessionRequestState::Failed
    )
}

fn llama_cpp_final_request_state_label(state: SessionRequestState) -> &'static str {
    match state {
        SessionRequestState::Finished => "Finished",
        SessionRequestState::Cancelled => "Cancelled",
        SessionRequestState::Failed => "Failed",
        SessionRequestState::Waiting
        | SessionRequestState::Runnable
        | SessionRequestState::Running
        | SessionRequestState::BlockedOnMemory => "Running",
    }
}

impl Default for RuntimeObservation {
    fn default() -> Self {
        Self {
            step_count: 0,
            e2e_latency_ms: 0,
            ttft_ms: None,
            prefill_tokens: 0,
            decode_tokens: 0,
            prefill_steps: 0,
            decode_steps: 0,
            total_scheduled_tokens: 0,
            total_selected_requests: 0,
            prefix_hits: 0,
            memory_blocked_steps: 0,
            memory_blocked_request_events: 0,
            total_cpu_time_us: 0,
            total_runner_time_us: 0,
            total_prefill_runner_time_us: 0,
            total_decode_runner_time_us: 0,
            kv_peak_blocks: 0,
            memory_peak_mb: 0.0,
            evictions: 0,
            cleanup_count: 0,
            llama_cpp_processing_request_events: 0,
            llama_cpp_deferred_request_events: 0,
            route_metadata: RouteMetadata::empty(),
            step_trace: Vec::new(),
            final_requests: Vec::new(),
            cancelled_requests: BTreeSet::new(),
            digest: Value::Null,
            execution_plans_seen: BTreeSet::new(),
            attention_routes_seen: BTreeSet::new(),
            kv_modes_seen: BTreeSet::new(),
            barrier_modes_seen: BTreeSet::new(),
            runtime_report: None,
        }
    }
}

fn record_route_variant(seen: &mut BTreeSet<String>, value: Option<&str>) {
    if let Some(value) = value {
        seen.insert(value.to_string());
    }
}

fn aggregate_route_variant(seen: &BTreeSet<String>, mixed_label: &str) -> Option<String> {
    match seen.len() {
        0 => None,
        1 => seen.iter().next().cloned(),
        _ => Some(mixed_label.to_string()),
    }
}

fn merge_route_variant(
    current: &Option<String>,
    next: &Option<String>,
    mixed_label: &str,
) -> Option<String> {
    match (current.as_deref(), next.as_deref()) {
        (None, None) => None,
        (Some(current), None) | (None, Some(current)) => Some(current.to_string()),
        (Some(current), Some(next)) if current == next => Some(current.to_string()),
        _ => Some(mixed_label.to_string()),
    }
}

fn merge_step_route_metadata(target: &mut RouteMetadata, next: &RouteMetadata) {
    target.execution_plan = merge_route_variant(
        &target.execution_plan,
        &next.execution_plan,
        "mixed_step_plans",
    );
    target.attention_route = merge_route_variant(
        &target.attention_route,
        &next.attention_route,
        "mixed_attention_routes",
    );
    target.kv_mode = merge_route_variant(&target.kv_mode, &next.kv_mode, "mixed_kv_modes");
    target.prefix_cache_path = merge_route_variant(
        &target.prefix_cache_path,
        &next.prefix_cache_path,
        "mixed_prefix_cache_paths",
    );
    target.barrier_mode = merge_route_variant(
        &target.barrier_mode,
        &next.barrier_mode,
        "mixed_barrier_modes",
    );

    let mut decisions = target
        .crossover_decisions
        .iter()
        .cloned()
        .collect::<BTreeMap<_, _>>();
    for (key, value) in &next.crossover_decisions {
        let entry = decisions.entry(key.clone()).or_insert(0);
        *entry = entry.saturating_add(*value);
    }
    target.crossover_decisions = decisions.into_iter().collect();
}

fn delegated_cached_tokens_from_generate_route(route: &GenerateRouteReport) -> u32 {
    route
        .crossover_decisions
        .iter()
        .find(|(key, _)| key.as_str() == "delegated_cached_tokens")
        .map(|(_, value)| *value)
        .unwrap_or(0)
}

fn upsert_route_decision(decisions: &mut Vec<(String, u32)>, key: &str, value: u32) {
    if let Some((_, existing)) = decisions.iter_mut().find(|(decision, _)| decision == key) {
        *existing = value;
    } else {
        decisions.push((key.to_string(), value));
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MetalDispatchValidationStepTrace {
    expected_key_cache_checksum: u64,
    expected_attention_output_checksum: u64,
    expected_gather_output_checksum: u64,
    expected_copy_output_checksum: u64,
    attention_max_abs_diff_microunits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MetalDispatchNumericStepTrace {
    key_cache_checksum: u64,
    attention_output_checksum: u64,
    gather_output_checksum: u64,
    copy_output_checksum: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    validation: Option<MetalDispatchValidationStepTrace>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MetalDispatchKernelStepTrace {
    function_name: String,
    element_count: u32,
    threads_per_grid_width: u64,
    threads_per_threadgroup_width: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MetalDispatchStepTrace {
    command_queue_label: String,
    command_buffer_label: String,
    command_buffer_status: String,
    runtime_device_name: String,
    runtime_required_pipeline_count: u32,
    runtime_max_thread_execution_width: u64,
    runtime_model_conditioned_inputs: bool,
    runtime_real_model_tensor_inputs: bool,
    runtime_complete_model_forward_supported: bool,
    runtime_model_bindings_prepared: bool,
    runtime_model_buffers_bound: bool,
    runtime_model_buffer_count: u32,
    runtime_model_buffer_bytes: u64,
    runtime_model_family: Option<String>,
    execution_direct_decode_token_count: u32,
    execution_direct_decode_checksum_lo: u32,
    execution_logits_output_count: u32,
    execution_remaining_logits_handle_count: u32,
    execution_model_bound_ffn_decode: bool,
    execution_real_model_forward_completed: bool,
    execution_prefix_native_dispatch_count: u32,
    execution_prefix_cpu_reference_dispatch_count: u32,
    execution_qkv_projection_token_count: u32,
    execution_layer_continuation_token_count: u32,
    execution_logits_projection_token_count: u32,
    execution_logits_vocab_scan_row_count: u32,
    execution_prefix_native_projection_row_count: u32,
    execution_prefix_cpu_projection_row_count: u32,
    execution_prefix_native_rms_norm_element_count: u32,
    execution_prefix_cpu_rms_norm_element_count: u32,
    execution_prefix_native_ffn_activation_element_count: u32,
    execution_prefix_cpu_ffn_activation_element_count: u32,
    execution_prefix_native_residual_add_element_count: u32,
    execution_prefix_cpu_residual_add_element_count: u32,
    execution_prefix_native_scale_element_count: u32,
    execution_prefix_cpu_scale_element_count: u32,
    execution_direct_decode_native_projection_row_count: u32,
    execution_direct_decode_cpu_projection_row_count: u32,
    execution_direct_decode_native_rms_norm_element_count: u32,
    execution_direct_decode_cpu_rms_norm_element_count: u32,
    execution_direct_decode_native_ffn_activation_element_count: u32,
    execution_direct_decode_cpu_ffn_activation_element_count: u32,
    execution_direct_decode_native_residual_add_element_count: u32,
    execution_direct_decode_cpu_residual_add_element_count: u32,
    execution_direct_decode_native_scale_element_count: u32,
    execution_direct_decode_cpu_scale_element_count: u32,
    execution_direct_decode_batched_logits_group_count: u32,
    execution_direct_decode_batched_logits_token_count: u32,
    execution_direct_decode_batched_group_fallback_count: u32,
    execution_direct_decode_batched_group_fallback_token_count: u32,
    binary_archive_state: String,
    binary_archive_attached_pipeline_count: u32,
    binary_archive_serialized: bool,
    arena_token_capacity: u32,
    arena_slot_capacity: u32,
    arena_attention_ref_capacity: u32,
    arena_gather_ref_capacity: u32,
    arena_gather_output_capacity: u32,
    arena_copy_pair_capacity: u32,
    arena_sequence_capacity: u32,
    arena_reused_existing: bool,
    arena_grew_existing: bool,
    kernels: Vec<MetalDispatchKernelStepTrace>,
    numeric: MetalDispatchNumericStepTrace,
}

impl MetalDispatchStepTrace {
    fn from_trace(trace: &MetalDispatchTrace) -> Self {
        Self {
            command_queue_label: trace.command_queue_label.clone(),
            command_buffer_label: trace.command_buffer_label.clone(),
            command_buffer_status: json_string_label(trace.command_buffer_status),
            runtime_device_name: trace.runtime.device_name.clone(),
            runtime_required_pipeline_count: trace.runtime.required_pipeline_count,
            runtime_max_thread_execution_width: trace.runtime.max_thread_execution_width,
            runtime_model_conditioned_inputs: trace.runtime.model_conditioned_inputs,
            runtime_real_model_tensor_inputs: trace.runtime.real_model_tensor_inputs,
            runtime_complete_model_forward_supported: trace
                .runtime
                .complete_model_forward_supported,
            runtime_model_bindings_prepared: trace.runtime.model_bindings_prepared,
            runtime_model_buffers_bound: trace.runtime.model_buffers_bound,
            runtime_model_buffer_count: trace.runtime.model_buffer_count,
            runtime_model_buffer_bytes: trace.runtime.model_buffer_bytes,
            runtime_model_family: trace
                .runtime
                .model
                .as_ref()
                .map(|model| model.model_family.clone()),
            execution_direct_decode_token_count: trace.execution.direct_decode_token_count,
            execution_direct_decode_checksum_lo: trace.execution.direct_decode_checksum_lo,
            execution_logits_output_count: trace.execution.logits_output_count,
            execution_remaining_logits_handle_count: trace.execution.remaining_logits_handle_count,
            execution_model_bound_ffn_decode: trace.execution.model_bound_ffn_decode,
            execution_real_model_forward_completed: trace.execution.real_model_forward_completed,
            execution_prefix_native_dispatch_count: trace.execution.prefix_native_dispatch_count,
            execution_prefix_cpu_reference_dispatch_count: trace
                .execution
                .prefix_cpu_reference_dispatch_count,
            execution_qkv_projection_token_count: trace.execution.qkv_projection_token_count,
            execution_layer_continuation_token_count: trace
                .execution
                .layer_continuation_token_count,
            execution_logits_projection_token_count: trace.execution.logits_projection_token_count,
            execution_logits_vocab_scan_row_count: trace.execution.logits_vocab_scan_row_count,
            execution_prefix_native_projection_row_count: trace
                .execution
                .prefix_native_projection_row_count,
            execution_prefix_cpu_projection_row_count: trace
                .execution
                .prefix_cpu_projection_row_count,
            execution_prefix_native_rms_norm_element_count: trace
                .execution
                .prefix_native_rms_norm_element_count,
            execution_prefix_cpu_rms_norm_element_count: trace
                .execution
                .prefix_cpu_rms_norm_element_count,
            execution_prefix_native_ffn_activation_element_count: trace
                .execution
                .prefix_native_ffn_activation_element_count,
            execution_prefix_cpu_ffn_activation_element_count: trace
                .execution
                .prefix_cpu_ffn_activation_element_count,
            execution_prefix_native_residual_add_element_count: trace
                .execution
                .prefix_native_residual_add_element_count,
            execution_prefix_cpu_residual_add_element_count: trace
                .execution
                .prefix_cpu_residual_add_element_count,
            execution_prefix_native_scale_element_count: trace
                .execution
                .prefix_native_scale_element_count,
            execution_prefix_cpu_scale_element_count: trace
                .execution
                .prefix_cpu_scale_element_count,
            execution_direct_decode_native_projection_row_count: trace
                .execution
                .direct_decode_native_projection_row_count,
            execution_direct_decode_cpu_projection_row_count: trace
                .execution
                .direct_decode_cpu_projection_row_count,
            execution_direct_decode_native_rms_norm_element_count: trace
                .execution
                .direct_decode_native_rms_norm_element_count,
            execution_direct_decode_cpu_rms_norm_element_count: trace
                .execution
                .direct_decode_cpu_rms_norm_element_count,
            execution_direct_decode_native_ffn_activation_element_count: trace
                .execution
                .direct_decode_native_ffn_activation_element_count,
            execution_direct_decode_cpu_ffn_activation_element_count: trace
                .execution
                .direct_decode_cpu_ffn_activation_element_count,
            execution_direct_decode_native_residual_add_element_count: trace
                .execution
                .direct_decode_native_residual_add_element_count,
            execution_direct_decode_cpu_residual_add_element_count: trace
                .execution
                .direct_decode_cpu_residual_add_element_count,
            execution_direct_decode_native_scale_element_count: trace
                .execution
                .direct_decode_native_scale_element_count,
            execution_direct_decode_cpu_scale_element_count: trace
                .execution
                .direct_decode_cpu_scale_element_count,
            execution_direct_decode_batched_logits_group_count: trace
                .execution
                .direct_decode_batched_logits_group_count,
            execution_direct_decode_batched_logits_token_count: trace
                .execution
                .direct_decode_batched_logits_token_count,
            execution_direct_decode_batched_group_fallback_count: trace
                .execution
                .direct_decode_batched_group_fallback_count,
            execution_direct_decode_batched_group_fallback_token_count: trace
                .execution
                .direct_decode_batched_group_fallback_token_count,
            binary_archive_state: json_string_label(trace.runtime.binary_archive.state),
            binary_archive_attached_pipeline_count: trace
                .runtime
                .binary_archive
                .attached_pipeline_count,
            binary_archive_serialized: trace.runtime.binary_archive.serialized,
            arena_token_capacity: trace.arena.token_capacity,
            arena_slot_capacity: trace.arena.slot_capacity,
            arena_attention_ref_capacity: trace.arena.attention_ref_capacity,
            arena_gather_ref_capacity: trace.arena.gather_ref_capacity,
            arena_gather_output_capacity: trace.arena.gather_output_capacity,
            arena_copy_pair_capacity: trace.arena.copy_pair_capacity,
            arena_sequence_capacity: trace.arena.sequence_capacity,
            arena_reused_existing: trace.arena.reused_existing,
            arena_grew_existing: trace.arena.grew_existing,
            kernels: trace
                .kernels
                .iter()
                .map(|kernel| MetalDispatchKernelStepTrace {
                    function_name: kernel.function_name.clone(),
                    element_count: kernel.element_count,
                    threads_per_grid_width: kernel.threads_per_grid.width,
                    threads_per_threadgroup_width: kernel.threads_per_threadgroup.width,
                })
                .collect(),
            numeric: MetalDispatchNumericStepTrace {
                key_cache_checksum: trace.numeric.key_cache_checksum,
                attention_output_checksum: trace.numeric.attention_output_checksum,
                gather_output_checksum: trace.numeric.gather_output_checksum,
                copy_output_checksum: trace.numeric.copy_output_checksum,
                validation: trace.numeric.validation.as_ref().map(|validation| {
                    MetalDispatchValidationStepTrace {
                        expected_key_cache_checksum: validation.expected_key_cache_checksum,
                        expected_attention_output_checksum: validation
                            .expected_attention_output_checksum,
                        expected_gather_output_checksum: validation.expected_gather_output_checksum,
                        expected_copy_output_checksum: validation.expected_copy_output_checksum,
                        attention_max_abs_diff_microunits: validation
                            .attention_max_abs_diff_microunits,
                    }
                }),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StepTraceEntry {
    t_ms: u64,
    step_id: Option<StepId>,
    admitted_request_ids: Vec<RequestId>,
    selected_request_ids: Vec<RequestId>,
    deferred_request_ids: Vec<RequestId>,
    memory_blocked_request_ids: Vec<RequestId>,
    cleanup_request_ids: Vec<RequestId>,
    scheduled_tokens: u32,
    prefix_hits: u32,
    cpu_time_us: u64,
    runner_time_us: u64,
    kv_usage_blocks: u32,
    evictions: u32,
    runner_executed: bool,
    route_metadata: RouteMetadata,
    metal_dispatch: Option<MetalDispatchStepTrace>,
    items: Vec<StepTraceItem>,
}

impl StepTraceEntry {
    fn capture(
        engine: &EngineCore,
        outcome: &ax_engine_core::EngineStepOutcome,
        current_time_ms: u64,
    ) -> Self {
        let (route_metadata, items) = if let Some(batch) = &outcome.schedule_plan.execution_batch {
            (
                batch.route_metadata.clone(),
                batch.items.iter().map(StepTraceItem::capture).collect(),
            )
        } else {
            (RouteMetadata::empty(), Vec::new())
        };

        Self {
            t_ms: current_time_ms,
            step_id: outcome.metrics.step_id,
            admitted_request_ids: outcome.admitted_requests.clone(),
            selected_request_ids: outcome.schedule_plan.selected_requests.clone(),
            deferred_request_ids: outcome.schedule_plan.deferred_requests.clone(),
            memory_blocked_request_ids: outcome.schedule_plan.memory_blocked_requests.clone(),
            cleanup_request_ids: outcome
                .cleanup_results
                .iter()
                .map(|result| result.request_id)
                .collect(),
            scheduled_tokens: outcome.metrics.scheduled_tokens,
            prefix_hits: outcome.metrics.prefix_hits,
            cpu_time_us: outcome.metrics.cpu_time_us,
            runner_time_us: outcome.metrics.runner_time_us,
            kv_usage_blocks: engine.kv_manager().used_block_count(),
            evictions: outcome.metrics.evictions,
            runner_executed: outcome.runner_output.is_some(),
            route_metadata,
            metal_dispatch: outcome
                .runner_output
                .as_ref()
                .and_then(|_| engine.last_metal_dispatch())
                .map(|trace| MetalDispatchStepTrace::from_trace(&trace)),
            items,
        }
    }

    fn json(&self) -> Value {
        let mut value = json!({
            "t_ms": self.t_ms,
            "step_id": self.step_id.map(|step_id| step_id.0),
            "admitted_request_ids": self.admitted_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "selected_request_ids": self.selected_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "deferred_request_ids": self.deferred_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "memory_blocked_request_ids": self.memory_blocked_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "cleanup_request_ids": self.cleanup_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "scheduled_tokens": self.scheduled_tokens,
            "prefix_hits": self.prefix_hits,
            "cpu_time_us": self.cpu_time_us,
            "runner_time_us": self.runner_time_us,
            "kv_usage_blocks": self.kv_usage_blocks,
            "evictions": self.evictions,
            "runner_executed": self.runner_executed,
            "route": serialize_route_metadata(&self.route_metadata),
            "items": self.items.iter().map(StepTraceItem::json).collect::<Vec<_>>()
        });
        if let Some(metal_dispatch) = self.metal_dispatch.as_ref() {
            value
                .as_object_mut()
                .expect("step trace json should serialize as object")
                .insert(
                    "metal_dispatch".to_string(),
                    serde_json::to_value(metal_dispatch)
                        .expect("metal dispatch summary should serialize"),
                );
        }
        value
    }

    fn capture_llama_cpp_shared(
        selected_request_ids: Vec<RequestId>,
        deferred_request_ids: Vec<RequestId>,
        step: &EngineStepReport,
        current_time_ms: u64,
        route_metadata: RouteMetadata,
        items: Vec<StepTraceItem>,
    ) -> Self {
        Self {
            t_ms: current_time_ms,
            step_id: step.step_id.map(StepId),
            admitted_request_ids: Vec::new(),
            selected_request_ids,
            deferred_request_ids,
            memory_blocked_request_ids: Vec::new(),
            cleanup_request_ids: Vec::new(),
            scheduled_tokens: step.scheduled_tokens,
            prefix_hits: step.prefix_hits,
            cpu_time_us: step.cpu_time_us,
            runner_time_us: step.runner_time_us,
            kv_usage_blocks: step.kv_usage_blocks,
            evictions: step.evictions,
            runner_executed: false,
            route_metadata,
            metal_dispatch: None,
            items,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StepTraceItem {
    request_id: RequestId,
    mode: ax_engine_core::ExecutionMode,
    position_start: u32,
    position_end_exclusive: u32,
    scheduled_token_count: u32,
    input_token_count: usize,
    prefix_tokens_reused: u32,
    prefix_blocks_reused: u32,
}

impl StepTraceItem {
    fn capture(item: &ax_engine_core::ExecutionItem) -> Self {
        Self {
            request_id: item.request_id,
            mode: item.mode,
            position_start: item.position_range.start,
            position_end_exclusive: item.position_range.end_exclusive,
            scheduled_token_count: item.scheduled_token_count,
            input_token_count: item.input_token_slice.len(),
            prefix_tokens_reused: item.prefix_tokens_reused,
            prefix_blocks_reused: item.prefix_blocks_reused,
        }
    }

    fn json(&self) -> Value {
        json!({
            "request_id": self.request_id.0,
            "mode": match self.mode {
                ax_engine_core::ExecutionMode::Prefill => "Prefill",
                ax_engine_core::ExecutionMode::Decode => "Decode",
            },
            "position_start": self.position_start,
            "position_end_exclusive": self.position_end_exclusive,
            "scheduled_token_count": self.scheduled_token_count,
            "input_token_count": self.input_token_count,
            "prefix_tokens_reused": self.prefix_tokens_reused,
            "prefix_blocks_reused": self.prefix_blocks_reused
        })
    }

    fn capture_llama_cpp_decode(
        request_id: RequestId,
        output_len: u32,
        scheduled_tokens: u32,
    ) -> Self {
        let position_start = output_len.saturating_sub(scheduled_tokens);
        Self {
            request_id,
            mode: ax_engine_core::ExecutionMode::Decode,
            position_start,
            position_end_exclusive: output_len,
            scheduled_token_count: scheduled_tokens,
            input_token_count: 0,
            prefix_tokens_reused: 0,
            prefix_blocks_reused: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FinalRequestState {
    external_id: String,
    request_id: RequestId,
    state: String,
    processed_prompt_tokens: u32,
    generated_tokens: Vec<u32>,
    cancel_requested: bool,
    last_error: Option<String>,
}

impl FinalRequestState {
    fn digest_json(&self) -> Value {
        json!({
            "external_id": self.external_id,
            "request_id": self.request_id.0,
            "state": self.state,
            "processed_prompt_tokens": self.processed_prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "cancel_requested": self.cancel_requested,
            "last_error": self.last_error
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Error)]
enum CliError {
    #[error("{0}")]
    Usage(String),
    #[error("{0}")]
    Runtime(String),
    #[error("{0}")]
    Contract(String),
    #[error("{0}")]
    Correctness(String),
    #[error("{0}")]
    Performance(String),
}

impl From<ax_engine_core::EngineCoreError> for CliError {
    fn from(value: ax_engine_core::EngineCoreError) -> Self {
        Self::Runtime(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_route_metadata(
        execution_plan: &str,
        attention_route: &str,
        kv_mode: &str,
        barrier_mode: &str,
    ) -> RouteMetadata {
        RouteMetadata {
            execution_plan: Some(execution_plan.into()),
            attention_route: Some(attention_route.into()),
            kv_mode: Some(kv_mode.into()),
            prefix_cache_path: Some("metadata_lookup".into()),
            barrier_mode: Some(barrier_mode.into()),
            crossover_decisions: Vec::new(),
        }
    }

    fn sample_metal_dispatch_trace() -> MetalDispatchTrace {
        MetalDispatchTrace {
            command_queue_label: "ax.queue".to_string(),
            command_buffer_label: "ax.buffer".to_string(),
            command_buffer_status: ax_engine_core::MetalCommandBufferStatus::Completed,
            runtime: ax_engine_core::metal::MetalDispatchRuntimeInfo {
                device_name: "Apple M4 Max".to_string(),
                required_pipeline_count: 4,
                max_thread_execution_width: 64,
                binary_archive: ax_engine_core::MetalBinaryArchiveInfo {
                    path: PathBuf::from("/tmp/ax_phase1_dense_path.binary_archive.metallib"),
                    state: ax_engine_core::MetalBinaryArchiveState::Loaded,
                    attached_pipeline_count: 4,
                    serialized: true,
                    note: None,
                },
                command_queue_ready: true,
                model_conditioned_inputs: false,
                real_model_tensor_inputs: false,
                complete_model_forward_supported: false,
                model_bindings_prepared: false,
                model_buffers_bound: false,
                model_buffer_count: 0,
                model_buffer_bytes: 0,
                native_dense_kernel_coverage:
                    ax_engine_core::metal::MetalNativeDenseKernelCoverage::default(),
                model: None,
            },
            workload: ax_engine_core::MetalDispatchWorkload {
                scheduled_requests: 1,
                prefill_requests: 1,
                decode_requests: 0,
                scheduled_tokens: 2,
                scheduled_token_ids: vec![10, 11],
                scheduled_positions: vec![0, 1],
                resolved_blocks: 1,
                token_elements: 2,
                block_elements: 16,
                scratch_elements: 32,
                kv_slot_capacity: 16,
                kv_block_capacity: 1,
                numeric_layout: ax_engine_core::MetalDispatchNumericLayout::default(),
                kv_metadata: ax_engine_core::metal::MetalDispatchKvMetadata {
                    block_size_tokens: 16,
                    slot_mapping: vec![0, 1],
                    attention_block_table: vec![0, 1],
                    gather_block_table: vec![0],
                    gather_block_table_stride: 1,
                    copy_block_mapping: vec![[0, 0]],
                    seq_lens: vec![2],
                    cu_seq_lens: vec![0, 2],
                    scheduled_cu_seq_lens: vec![0, 2],
                },
            },
            arena: ax_engine_core::MetalDispatchArenaInfo {
                token_capacity: 8,
                slot_capacity: 64,
                attention_ref_capacity: 8,
                gather_ref_capacity: 8,
                gather_output_capacity: 8,
                copy_pair_capacity: 4,
                sequence_capacity: 4,
                reused_existing: true,
                grew_existing: false,
            },
            execution: ax_engine_core::metal::MetalDispatchExecutionInfo {
                direct_decode_token_count: 1,
                direct_decode_checksum_lo: 0x1234,
                logits_output_count: 1,
                remaining_logits_handle_count: 0,
                model_bound_ffn_decode: true,
                real_model_forward_completed: true,
                prefix_native_dispatch_count: 35,
                prefix_cpu_reference_dispatch_count: 1,
                qkv_projection_token_count: 72,
                layer_continuation_token_count: 37,
                logits_projection_token_count: 1,
                logits_vocab_scan_row_count: 151936,
                prefix_native_projection_row_count: 4096,
                prefix_cpu_projection_row_count: 512,
                prefix_native_rms_norm_element_count: 2048,
                prefix_cpu_rms_norm_element_count: 128,
                prefix_native_ffn_activation_element_count: 1024,
                prefix_cpu_ffn_activation_element_count: 64,
                prefix_native_residual_add_element_count: 4096,
                prefix_cpu_residual_add_element_count: 0,
                prefix_native_scale_element_count: 2048,
                prefix_cpu_scale_element_count: 0,
                direct_decode_native_projection_row_count: 151936,
                direct_decode_cpu_projection_row_count: 3072,
                direct_decode_native_rms_norm_element_count: 6144,
                direct_decode_cpu_rms_norm_element_count: 0,
                direct_decode_native_ffn_activation_element_count: 1536,
                direct_decode_cpu_ffn_activation_element_count: 0,
                direct_decode_native_residual_add_element_count: 3072,
                direct_decode_cpu_residual_add_element_count: 0,
                direct_decode_native_scale_element_count: 1536,
                direct_decode_cpu_scale_element_count: 0,
                direct_decode_batched_logits_group_count: 1,
                direct_decode_batched_logits_token_count: 2,
                direct_decode_batched_group_fallback_count: 0,
                direct_decode_batched_group_fallback_token_count: 0,
            },
            kernels: vec![
                ax_engine_core::MetalDispatchKernelTrace {
                    function_name: "reshape_and_cache".to_string(),
                    element_count: 32,
                    threads_per_grid: ax_engine_core::MetalThreadgroupSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                    threads_per_threadgroup: ax_engine_core::MetalThreadgroupSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                },
                ax_engine_core::MetalDispatchKernelTrace {
                    function_name: "paged_decode_attention".to_string(),
                    element_count: 16,
                    threads_per_grid: ax_engine_core::MetalThreadgroupSize {
                        width: 16,
                        height: 1,
                        depth: 1,
                    },
                    threads_per_threadgroup: ax_engine_core::MetalThreadgroupSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                },
            ],
            numeric: ax_engine_core::metal::MetalDispatchNumericTrace {
                attention_output_bits: vec![0],
                key_cache_checksum: 0x11,
                attention_output_checksum: 0x22,
                gather_output_checksum: 0x33,
                copy_output_checksum: 0x44,
                validation: Some(ax_engine_core::metal::MetalNumericValidationSummary {
                    expected_key_cache_checksum: 0x11,
                    expected_attention_output_checksum: 0x22,
                    expected_gather_output_checksum: 0x33,
                    expected_copy_output_checksum: 0x44,
                    attention_max_abs_diff_microunits: 0,
                }),
            },
        }
    }

    #[test]
    fn step_trace_json_includes_metal_dispatch_summary() {
        let entry = StepTraceEntry {
            t_ms: 12,
            step_id: Some(StepId(3)),
            admitted_request_ids: vec![RequestId(1)],
            selected_request_ids: vec![RequestId(1)],
            deferred_request_ids: Vec::new(),
            memory_blocked_request_ids: Vec::new(),
            cleanup_request_ids: Vec::new(),
            scheduled_tokens: 2,
            prefix_hits: 0,
            cpu_time_us: 123,
            runner_time_us: 45,
            kv_usage_blocks: 1,
            evictions: 2,
            runner_executed: true,
            route_metadata: RouteMetadata::empty(),
            metal_dispatch: Some(MetalDispatchStepTrace::from_trace(
                &sample_metal_dispatch_trace(),
            )),
            items: Vec::new(),
        };

        let json = entry.json();

        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("command_buffer_status"))
                .and_then(Value::as_str),
            Some("completed")
        );
        assert_eq!(json.get("evictions").and_then(Value::as_u64), Some(2));
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("runtime_required_pipeline_count"))
                .and_then(Value::as_u64),
            Some(4)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("runtime_complete_model_forward_supported"))
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("runtime_model_buffers_bound"))
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_direct_decode_token_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_real_model_forward_completed"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_native_dispatch_count"))
                .and_then(Value::as_u64),
            Some(35)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_cpu_reference_dispatch_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_qkv_projection_token_count"))
                .and_then(Value::as_u64),
            Some(72)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_layer_continuation_token_count"))
                .and_then(Value::as_u64),
            Some(37)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_logits_projection_token_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_logits_vocab_scan_row_count"))
                .and_then(Value::as_u64),
            Some(151936)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_native_projection_row_count"))
                .and_then(Value::as_u64),
            Some(4096)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_cpu_projection_row_count"))
                .and_then(Value::as_u64),
            Some(512)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_native_rms_norm_element_count"))
                .and_then(Value::as_u64),
            Some(2048)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_cpu_rms_norm_element_count"))
                .and_then(Value::as_u64),
            Some(128)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_native_residual_add_element_count"))
                .and_then(Value::as_u64),
            Some(4096)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_cpu_residual_add_element_count"))
                .and_then(Value::as_u64),
            Some(0)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_native_scale_element_count"))
                .and_then(Value::as_u64),
            Some(2048)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_prefix_cpu_scale_element_count"))
                .and_then(Value::as_u64),
            Some(0)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_direct_decode_native_projection_row_count"))
                .and_then(Value::as_u64),
            Some(151936)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_direct_decode_cpu_projection_row_count"))
                .and_then(Value::as_u64),
            Some(3072)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| {
                    trace.get("execution_direct_decode_native_rms_norm_element_count")
                })
                .and_then(Value::as_u64),
            Some(6144)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_direct_decode_cpu_rms_norm_element_count"))
                .and_then(Value::as_u64),
            Some(0)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| {
                    trace.get("execution_direct_decode_native_residual_add_element_count")
                })
                .and_then(Value::as_u64),
            Some(3072)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| {
                    trace.get("execution_direct_decode_cpu_residual_add_element_count")
                })
                .and_then(Value::as_u64),
            Some(0)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| {
                    trace.get("execution_direct_decode_native_scale_element_count")
                })
                .and_then(Value::as_u64),
            Some(1536)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| { trace.get("execution_direct_decode_cpu_scale_element_count") })
                .and_then(Value::as_u64),
            Some(0)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_direct_decode_batched_logits_group_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("execution_direct_decode_batched_logits_token_count"))
                .and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("binary_archive_state"))
                .and_then(Value::as_str),
            Some("loaded")
        );
        assert_eq!(
            json.get("metal_dispatch")
                .and_then(|trace| trace.get("numeric"))
                .and_then(|numeric| numeric.get("validation"))
                .and_then(|validation| validation.get("attention_max_abs_diff_microunits"))
                .and_then(Value::as_u64),
            Some(0)
        );
    }

    #[test]
    fn metrics_json_surfaces_observed_evictions() {
        let execution = RuntimeResult {
            tool_mode: "mlx_runtime",
            runtime: RuntimeConfig {
                deterministic: true,
                max_batch_tokens: 8,
                block_size_tokens: 4,
                kv_total_blocks: Some(2),
                flags: RuntimeFlags::default(),
                llama_cpp_preset: None,
                backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
                resolved_backend: ResolvedBackend::new(
                    SelectedBackend::Mlx,
                    SupportTier::MlxPreview,
                    None,
                ),
                backend_adapter: None,
                mlx_model_artifacts_dir: None,
                mlx_model_artifacts_source: None,
            },
            observation: RuntimeObservation {
                evictions: 7,
                ..RuntimeObservation::default()
            },
            correctness: GateStatus::pass(),
            determinism: GateStatus::pass(),
        };

        let metrics = build_metrics_json("run-evictions", &execution);

        assert_eq!(
            nested_value(&metrics, &["metrics", "evictions"]).and_then(Value::as_f64),
            Some(7.0)
        );
        assert!(
            nested_value(&metrics, &["metrics", "delegated_llama_cpp"]).is_none(),
            "MLX metrics must not publish delegated llama.cpp counters"
        );
    }

    #[test]
    fn metrics_json_nests_delegated_llama_cpp_metrics_under_metrics() {
        let mut route = RouteMetadata::empty();
        route.prefix_cache_path = Some("delegated_prompt_cache".to_string());
        route
            .crossover_decisions
            .push(("delegated_cached_tokens".to_string(), 64));
        let execution = RuntimeResult {
            tool_mode: "llama_cpp_stepwise_runtime",
            runtime: runtime_config_from_manifest(&llama_cpp_scenario_manifest(
                "http://127.0.0.1:8081",
            ))
            .expect("llama.cpp runtime should load"),
            observation: RuntimeObservation {
                kv_peak_blocks: 3,
                llama_cpp_processing_request_events: 2,
                llama_cpp_deferred_request_events: 1,
                route_metadata: route,
                ..RuntimeObservation::default()
            },
            correctness: GateStatus::pass(),
            determinism: GateStatus::pass(),
        };

        let metrics = build_metrics_json("run-llama", &execution);
        let delegated = nested_value(&metrics, &["metrics", "delegated_llama_cpp"])
            .expect("delegated llama.cpp metrics should live under metrics");

        assert!(
            metrics.get("delegated_llama_cpp").is_none(),
            "delegated metrics must not drift to the root artifact object"
        );
        assert_eq!(
            delegated
                .get("backend_reported_cached_prompt_tokens")
                .and_then(Value::as_u64),
            Some(64)
        );
        assert_eq!(
            delegated
                .get("requests_processing_events")
                .and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            delegated
                .get("requests_deferred_events")
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            delegated
                .get("cache_reuse_observed")
                .and_then(Value::as_bool),
            Some(true)
        );
    }

    fn repo_manifest_path(relative: &str) -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root should be two levels above CARGO_MANIFEST_DIR")
            .join(relative)
            .to_string_lossy()
            .into_owned()
    }

    fn load_test_manifest(path: &str) -> BenchmarkManifest {
        let mut manifest = load_manifest(Path::new(path)).expect("manifest should load");
        if manifest.runtime.selected_backend == SelectedBackend::Mlx {
            manifest.runtime.deterministic = false;
            manifest.runtime.mlx_model_artifacts_dir = Some(write_valid_native_model_fixture());
        }
        manifest
    }

    fn load_test_manifest_json(path: &str) -> Value {
        load_json_value(Path::new(path)).expect("manifest JSON should load")
    }

    fn write_test_manifest_with_mlx_backend(root: &Path, source_path: &str) -> PathBuf {
        let mut manifest_json = load_test_manifest_json(source_path);
        if manifest_json
            .get("runtime")
            .and_then(|runtime| runtime.get("selected_backend"))
            .and_then(Value::as_str)
            == Some("mlx")
        {
            let runtime = manifest_json
                .get_mut("runtime")
                .and_then(Value::as_object_mut)
                .expect("test manifest runtime should be an object");
            runtime.insert("deterministic".to_string(), json!(false));
            runtime.insert(
                "mlx_model_artifacts_dir".to_string(),
                json!(write_valid_native_model_fixture()),
            );
        }

        let file_name = Path::new(source_path)
            .file_name()
            .expect("source manifest should have filename");
        let manifest_path = root.join(file_name);
        write_json_file(&manifest_path, &manifest_json).expect("test manifest should write");
        manifest_path
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ax-engine-bench-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn write_doctor_model_manifest(model_dir: &Path) {
        fs::write(model_dir.join("model-manifest.json"), "{}")
            .expect("model manifest should write");
    }

    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn compiled_repo_metal_build_dir() -> Option<PathBuf> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .to_path_buf();
        let build_dir = repo_root.join("build/metal");
        let build_report = load_json_value(&build_dir.join("build_report.json")).ok()?;
        (build_report.get("status").and_then(Value::as_str) == Some("compiled"))
            .then_some(build_dir)
    }

    fn native_model_tensor(
        name: &str,
        role: ax_engine_core::NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> ax_engine_core::NativeTensorSpec {
        ax_engine_core::NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: ax_engine_core::NativeTensorDataType::F16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    fn write_valid_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-runtime-metadata");
        fs::create_dir_all(&root_dir).expect("native model fixture directory should create");
        fs::write(root_dir.join("model.safetensors"), vec![0_u8; 4096])
            .expect("native model weights should write");

        let manifest = ax_engine_core::NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: ax_engine_core::NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
            layer_count: 1,
            hidden_size: 2048,
            intermediate_size: 11008,
            attention_head_count: 16,
            attention_head_dim: 128,
            kv_head_count: 8,
            vocab_size: 151936,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: std::collections::BTreeMap::new(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: ax_engine_core::NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: ax_engine_core::NativeMoeConfig::default(),
            glm_router: Default::default(),
            tensors: vec![
                native_model_tensor(
                    "model.embed_tokens.weight",
                    ax_engine_core::NativeTensorRole::TokenEmbedding,
                    None,
                    vec![151_936, 2_048],
                ),
                native_model_tensor(
                    "model.norm.weight",
                    ax_engine_core::NativeTensorRole::FinalNorm,
                    None,
                    vec![2_048],
                ),
                native_model_tensor(
                    "lm_head.weight",
                    ax_engine_core::NativeTensorRole::LmHead,
                    None,
                    vec![151_936, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.input_layernorm.weight",
                    ax_engine_core::NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![2_048],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.qkv_proj.weight",
                    ax_engine_core::NativeTensorRole::AttentionQkvPacked,
                    Some(0),
                    vec![4_096, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    ax_engine_core::NativeTensorRole::AttentionO,
                    Some(0),
                    vec![2_048, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.post_attention_layernorm.weight",
                    ax_engine_core::NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![2_048],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.gate_up_proj.weight",
                    ax_engine_core::NativeTensorRole::FfnGateUpPacked,
                    Some(0),
                    vec![8_192, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    ax_engine_core::NativeTensorRole::FfnDown,
                    Some(0),
                    vec![2_048, 4_096],
                ),
            ],
        };

        fs::write(
            root_dir.join(ax_engine_core::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("native model manifest should serialize"),
        )
        .expect("native model manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn native_model_tensor_with_file(
        name: &str,
        role: ax_engine_core::NativeTensorRole,
        layer_index: Option<u32>,
        shape: &[u64],
        file: &str,
        length_bytes: u64,
    ) -> ax_engine_core::NativeTensorSpec {
        ax_engine_core::NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: ax_engine_core::NativeTensorDataType::F32,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape: shape.to_vec(),
            file: PathBuf::from(file),
            offset_bytes: 0,
            length_bytes,
        }
    }

    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn write_f32_tensor_file(root_dir: &Path, file_name: &str, values: &[f32]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(root_dir.join(file_name), bytes).expect("tensor bytes should write");
    }

    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn write_projection_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-projection");
        fs::create_dir_all(&root_dir).expect("projection fixture directory should create");

        let embedding = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ];
        let ones = vec![1.0_f32; 8];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let double_identity = identity.iter().map(|value| value * 2.0).collect::<Vec<_>>();
        let triple_identity = identity.iter().map(|value| value * 3.0).collect::<Vec<_>>();
        let zero_matrix = vec![0.0_f32; 64];

        write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
        write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "lm_head.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "attn_q.bin", &identity);
        write_f32_tensor_file(&root_dir, "attn_k.bin", &double_identity);
        write_f32_tensor_file(&root_dir, "attn_v.bin", &triple_identity);
        write_f32_tensor_file(&root_dir, "attn_o.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_matrix);

        let matrix_bytes = (64 * std::mem::size_of::<f32>()) as u64;
        let vector_bytes = (8 * std::mem::size_of::<f32>()) as u64;
        let embedding_bytes = (embedding.len() * std::mem::size_of::<f32>()) as u64;
        let manifest = ax_engine_core::NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: ax_engine_core::NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
            layer_count: 1,
            hidden_size: 8,
            intermediate_size: 24,
            attention_head_count: 2,
            attention_head_dim: 4,
            kv_head_count: 2,
            vocab_size: 5,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: std::collections::BTreeMap::new(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: ax_engine_core::NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: ax_engine_core::NativeMoeConfig::default(),
            glm_router: Default::default(),
            tensors: vec![
                native_model_tensor_with_file(
                    "model.embed_tokens.weight",
                    ax_engine_core::NativeTensorRole::TokenEmbedding,
                    None,
                    &[5, 8],
                    "embed.bin",
                    embedding_bytes,
                ),
                native_model_tensor_with_file(
                    "model.norm.weight",
                    ax_engine_core::NativeTensorRole::FinalNorm,
                    None,
                    &[8],
                    "final_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "lm_head.weight",
                    ax_engine_core::NativeTensorRole::LmHead,
                    None,
                    &[5, 8],
                    "lm_head.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.input_layernorm.weight",
                    ax_engine_core::NativeTensorRole::AttentionNorm,
                    Some(0),
                    &[8],
                    "attn_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.q_proj.weight",
                    ax_engine_core::NativeTensorRole::AttentionQ,
                    Some(0),
                    &[8, 8],
                    "attn_q.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.k_proj.weight",
                    ax_engine_core::NativeTensorRole::AttentionK,
                    Some(0),
                    &[8, 8],
                    "attn_k.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.v_proj.weight",
                    ax_engine_core::NativeTensorRole::AttentionV,
                    Some(0),
                    &[8, 8],
                    "attn_v.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.o_proj.weight",
                    ax_engine_core::NativeTensorRole::AttentionO,
                    Some(0),
                    &[8, 8],
                    "attn_o.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.post_attention_layernorm.weight",
                    ax_engine_core::NativeTensorRole::FfnNorm,
                    Some(0),
                    &[8],
                    "ffn_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.gate_proj.weight",
                    ax_engine_core::NativeTensorRole::FfnGate,
                    Some(0),
                    &[8, 8],
                    "ffn_gate.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.up_proj.weight",
                    ax_engine_core::NativeTensorRole::FfnUp,
                    Some(0),
                    &[8, 8],
                    "ffn_up.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.down_proj.weight",
                    ax_engine_core::NativeTensorRole::FfnDown,
                    Some(0),
                    &[8, 8],
                    "ffn_down.bin",
                    matrix_bytes,
                ),
            ],
        };

        fs::write(
            root_dir.join(ax_engine_core::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)
                .expect("projection native model manifest should serialize"),
        )
        .expect("projection native model manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn write_multilayer_direct_decode_native_model_fixture() -> PathBuf {
        let root_dir = write_projection_native_model_fixture();
        let zero_matrix = vec![0.0_f32; 64];
        let mut lm_head = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ];
        lm_head.resize(64, 0.0);

        write_f32_tensor_file(&root_dir, "attn_q_l1.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "attn_k_l1.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "attn_v_l1.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "lm_head.bin", &lm_head);

        let manifest_path = root_dir.join(ax_engine_core::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("manifest should read");
        let mut manifest =
            serde_json::from_slice::<ax_engine_core::NativeModelManifest>(&manifest_bytes)
                .expect("manifest should parse");
        let matrix_bytes = (64 * std::mem::size_of::<f32>()) as u64;
        let vector_bytes = (8 * std::mem::size_of::<f32>()) as u64;
        manifest.layer_count = 2;
        manifest.tensors.extend([
            native_model_tensor_with_file(
                "model.layers.1.input_layernorm.weight",
                ax_engine_core::NativeTensorRole::AttentionNorm,
                Some(1),
                &[8],
                "attn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.q_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionQ,
                Some(1),
                &[8, 8],
                "attn_q_l1.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.k_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionK,
                Some(1),
                &[8, 8],
                "attn_k_l1.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.v_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionV,
                Some(1),
                &[8, 8],
                "attn_v_l1.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.o_proj.weight",
                ax_engine_core::NativeTensorRole::AttentionO,
                Some(1),
                &[8, 8],
                "attn_o.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.post_attention_layernorm.weight",
                ax_engine_core::NativeTensorRole::FfnNorm,
                Some(1),
                &[8],
                "ffn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.mlp.gate_proj.weight",
                ax_engine_core::NativeTensorRole::FfnGate,
                Some(1),
                &[8, 8],
                "ffn_gate.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.mlp.up_proj.weight",
                ax_engine_core::NativeTensorRole::FfnUp,
                Some(1),
                &[8, 8],
                "ffn_up.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.mlp.down_proj.weight",
                ax_engine_core::NativeTensorRole::FfnDown,
                Some(1),
                &[8, 8],
                "ffn_down.bin",
                matrix_bytes,
            ),
        ]);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");

        root_dir
    }

    fn native_environment_fixture() -> Value {
        json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "device": "Mac17,6",
                "hostname": "fixture-host",
                "system_model": "Mac17,6",
                "soc": "Apple M4 Max",
                "memory_gb": 128,
                "memory_bytes": 137438953472_u64,
                "logical_cpu_count": 16
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime",
                "ax_engine_bench_version": "0.1.0"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        })
    }

    fn mlx_runtime_fixture() -> Value {
        json!({
            "selected_backend": "mlx",
            "support_tier": "mlx_preview",
            "resolution_policy": "mlx_only",
            "mlx_runtime": {
                "runner": "metal_bringup",
                "artifacts_source": "explicit_config"
            },
            "host": runtime_host_fixture(),
            "metal_toolchain": runtime_metal_toolchain_fixture()
        })
    }

    #[test]
    fn mlx_runtime_tool_mode_inference_recognizes_metal_bringup() {
        let runtime = mlx_runtime_fixture();
        assert_eq!(
            inferred_tool_mode_from_runtime_json(&runtime),
            Some("engine_bringup_runtime")
        );
    }

    #[test]
    fn compare_result_label_distinguishes_metal_execution_semantics() {
        assert_eq!(
            compare_result_label("engine_bringup_runtime", "metal_numeric_scaffold_only"),
            "metal_numeric_scaffold_compare"
        );
        assert_eq!(
            compare_result_label(
                "engine_bringup_runtime",
                "metal_model_conditioned_numeric_scaffold"
            ),
            "metal_model_conditioned_scaffold_compare"
        );
        assert_eq!(
            compare_result_label("engine_bringup_runtime", "metal_real_model_forward"),
            "metal_real_model_forward_compare"
        );
        assert_eq!(
            compare_result_label(
                "engine_bringup_runtime",
                "metal_multilayer_mixed_prefix_attention"
            ),
            "metal_multilayer_mixed_prefix_attention_compare"
        );
        assert_eq!(
            compare_result_label(
                "engine_bringup_runtime",
                "metal_multilayer_native_prefix_attention"
            ),
            "metal_multilayer_native_prefix_attention_compare"
        );
        assert_eq!(
            compare_result_label(
                "engine_bringup_runtime",
                "metal_multilayer_model_incomplete"
            ),
            "metal_multilayer_model_incomplete_compare"
        );
        assert_eq!(
            compare_result_label("engine_bringup_runtime", "metal_model_bound_ffn_decode"),
            "metal_model_bound_ffn_decode_compare"
        );
        assert_eq!(
            compare_result_label("mlx_runtime", "metal_real_model_forward"),
            "metal_real_model_forward_compare"
        );
    }

    #[test]
    fn route_execution_semantics_marks_model_conditioned_numeric_scaffold_without_runtime_tensors()
    {
        let mut route = RouteMetadata::empty();
        route
            .crossover_decisions
            .push(("metal_dispatch_numeric_scaffold_only".to_string(), 1));
        route
            .crossover_decisions
            .push(("metal_dispatch_model_conditioned_inputs".to_string(), 1));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_model_conditioned_numeric_scaffold"
        );
    }

    #[test]
    fn route_execution_semantics_prefers_runtime_real_model_tensor_inputs() {
        let mut route = RouteMetadata::empty();
        route
            .crossover_decisions
            .push(("metal_dispatch_model_conditioned_inputs".to_string(), 1));
        route.crossover_decisions.push((
            "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
            1,
        ));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_real_model_tensor_inputs"
        );
    }

    #[test]
    fn route_execution_semantics_marks_multilayer_model_incomplete() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
            1,
        ));
        route
            .crossover_decisions
            .push(("metal_dispatch_model_layer_count".to_string(), 36));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_multilayer_model_incomplete"
        );
    }

    #[test]
    fn route_execution_semantics_keeps_multilayer_incomplete_without_real_forward_marker() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_runtime_complete_model_forward_supported".to_string(),
            1,
        ));
        route
            .crossover_decisions
            .push(("metal_dispatch_model_layer_count".to_string(), 36));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_multilayer_model_incomplete"
        );
    }

    #[test]
    fn route_execution_semantics_marks_multilayer_native_prefix_attention() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_layers_native_attention".to_string(),
            1,
        ));
        route
            .crossover_decisions
            .push(("metal_dispatch_model_layer_count".to_string(), 36));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_multilayer_native_prefix_attention"
        );
    }

    #[test]
    fn route_execution_semantics_marks_multilayer_mixed_prefix_attention() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_layers_native_attention".to_string(),
            1,
        ));
        route
            .crossover_decisions
            .push(("metal_dispatch_prefix_layers_cpu_reference".to_string(), 1));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_native_dispatch_count".to_string(),
            35,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_cpu_reference_dispatch_count".to_string(),
            1,
        ));
        route
            .crossover_decisions
            .push(("metal_dispatch_model_layer_count".to_string(), 36));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_multilayer_mixed_prefix_attention"
        );
    }

    #[test]
    fn route_execution_semantics_prefers_explicit_real_model_forward_marker() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_model_bound_ffn".to_string(),
            1,
        ));
        route
            .crossover_decisions
            .push(("metal_dispatch_real_model_forward".to_string(), 1));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_real_model_forward"
        );
    }

    #[test]
    fn route_execution_semantics_marks_model_bound_ffn_decode() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_model_bound_ffn".to_string(),
            1,
        ));

        assert_eq!(
            route_execution_semantics_label(&route),
            "metal_model_bound_ffn_decode"
        );
    }

    #[test]
    fn route_execution_semantics_from_json_marks_multilayer_model_incomplete() {
        let route_json = json!({
            "metal_real_model_tensor_inputs": true,
            "metal_model_layer_count": 28
        });

        assert_eq!(
            route_execution_semantics_from_json(&route_json),
            "metal_multilayer_model_incomplete"
        );
    }

    #[test]
    fn route_execution_semantics_from_json_marks_multilayer_native_prefix_attention() {
        let route_json = json!({
            "metal_real_model_tensor_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_model_layer_count": 28
        });

        assert_eq!(
            route_execution_semantics_from_json(&route_json),
            "metal_multilayer_native_prefix_attention"
        );
    }

    #[test]
    fn route_execution_semantics_from_json_prefers_explicit_real_model_forward_marker() {
        let route_json = json!({
            "metal_real_model_tensor_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_model_layer_count": 28,
            "metal_real_model_forward": true
        });

        assert_eq!(
            route_execution_semantics_from_json(&route_json),
            "metal_real_model_forward"
        );
    }

    #[test]
    fn route_execution_semantics_from_json_marks_multilayer_mixed_prefix_attention() {
        let route_json = json!({
            "metal_real_model_tensor_inputs": true,
            "mlx_metal_prefix_layers_attention": true,
            "metal_prefix_layers_cpu_reference": true,
            "mlx_metal_prefix_dispatch_count": 35,
            "metal_prefix_cpu_reference_dispatch_count": 1,
            "metal_model_layer_count": 28
        });

        assert_eq!(
            route_execution_semantics_from_json(&route_json),
            "metal_multilayer_mixed_prefix_attention"
        );
    }

    #[test]
    fn serialize_route_metadata_surfaces_runtime_complete_model_forward_support() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            "metal_dispatch_runtime_complete_model_forward_supported".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_native_dispatch_count".to_string(),
            35,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_cpu_reference_dispatch_count".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_native_projection_row_count".to_string(),
            4096,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_cpu_projection_row_count".to_string(),
            512,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_native_rms_norm_element_count".to_string(),
            2048,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_cpu_rms_norm_element_count".to_string(),
            128,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_native_ffn_activation_element_count".to_string(),
            1024,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_cpu_ffn_activation_element_count".to_string(),
            64,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_native_residual_add_element_count".to_string(),
            4096,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_prefix_cpu_residual_add_element_count".to_string(),
            0,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_native_projection_f16_binding_count".to_string(),
            28,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_native_rms_norm_bf16_binding_count".to_string(),
            28,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_native_projection_row_count".to_string(),
            151936,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_cpu_projection_row_count".to_string(),
            3072,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_native_rms_norm_element_count".to_string(),
            6144,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_cpu_rms_norm_element_count".to_string(),
            0,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_native_ffn_activation_element_count".to_string(),
            1536,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_cpu_ffn_activation_element_count".to_string(),
            0,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_native_residual_add_element_count".to_string(),
            3072,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_cpu_residual_add_element_count".to_string(),
            0,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_batched_logits_group_count".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_batched_logits_token_count".to_string(),
            2,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_batched_group_fallback_count".to_string(),
            1,
        ));
        route.crossover_decisions.push((
            "metal_dispatch_direct_decode_batched_group_fallback_token_count".to_string(),
            2,
        ));

        let route_json = serialize_route_metadata(&route);

        assert_eq!(
            route_json.get("metal_complete_model_forward_supported"),
            Some(&Value::Bool(true))
        );
        assert_eq!(
            route_json.get("mlx_metal_prefix_dispatch_count"),
            Some(&Value::from(35))
        );
        assert_eq!(
            route_json.get("metal_prefix_cpu_reference_dispatch_count"),
            Some(&Value::from(1))
        );
        assert_eq!(
            route_json.get("mlx_metal_prefix_projection_row_count"),
            Some(&Value::from(4096))
        );
        assert_eq!(
            route_json.get("metal_prefix_cpu_projection_row_count"),
            Some(&Value::from(512))
        );
        assert_eq!(
            route_json.get("mlx_metal_prefix_rms_norm_element_count"),
            Some(&Value::from(2048))
        );
        assert_eq!(
            route_json.get("metal_prefix_cpu_rms_norm_element_count"),
            Some(&Value::from(128))
        );
        assert_eq!(
            route_json.get("mlx_metal_prefix_ffn_activation_element_count"),
            Some(&Value::from(1024))
        );
        assert_eq!(
            route_json.get("metal_prefix_cpu_ffn_activation_element_count"),
            Some(&Value::from(64))
        );
        assert_eq!(
            route_json.get("mlx_metal_prefix_residual_add_element_count"),
            Some(&Value::from(4096))
        );
        assert_eq!(
            route_json.get("metal_prefix_cpu_residual_add_element_count"),
            Some(&Value::from(0))
        );
        assert_eq!(
            route_json.get("metal_prefix_projection_mlx_metal_dispatch_share"),
            Some(&Value::from(4096.0_f64 / 4608.0_f64))
        );
        assert_eq!(
            route_json.get("metal_prefix_rms_norm_mlx_metal_dispatch_share"),
            Some(&Value::from(2048.0_f64 / 2176.0_f64))
        );
        assert_eq!(
            route_json.get("metal_prefix_ffn_activation_mlx_metal_dispatch_share"),
            Some(&Value::from(1024.0_f64 / 1088.0_f64))
        );
        assert_eq!(
            route_json.get("metal_prefix_residual_add_mlx_metal_dispatch_share"),
            Some(&Value::from(1.0_f64))
        );
        assert_eq!(
            route_json.get("mlx_metal_projection_f16_binding_count"),
            Some(&Value::from(28))
        );
        assert_eq!(
            route_json.get("mlx_metal_rms_norm_bf16_binding_count"),
            Some(&Value::from(28))
        );
        assert_eq!(
            route_json.get("mlx_metal_direct_decode_projection_row_count"),
            Some(&Value::from(151936))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_cpu_projection_row_count"),
            Some(&Value::from(3072))
        );
        assert_eq!(
            route_json.get("mlx_metal_direct_decode_rms_norm_element_count"),
            Some(&Value::from(6144))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_cpu_rms_norm_element_count"),
            Some(&Value::from(0))
        );
        assert_eq!(
            route_json.get("mlx_metal_direct_decode_ffn_activation_element_count"),
            Some(&Value::from(1536))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_cpu_ffn_activation_element_count"),
            Some(&Value::from(0))
        );
        assert_eq!(
            route_json.get("mlx_metal_direct_decode_residual_add_element_count"),
            Some(&Value::from(3072))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_cpu_residual_add_element_count"),
            Some(&Value::from(0))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_batched_logits_group_count"),
            Some(&Value::from(1))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_batched_logits_token_count"),
            Some(&Value::from(2))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_batched_group_fallback_count"),
            Some(&Value::from(1))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_batched_group_fallback_token_count"),
            Some(&Value::from(2))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_projection_mlx_metal_dispatch_share"),
            Some(&Value::from(151936.0_f64 / 155008.0_f64))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_rms_norm_mlx_metal_dispatch_share"),
            Some(&Value::from(1.0_f64))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_ffn_activation_mlx_metal_dispatch_share"),
            Some(&Value::from(1.0_f64))
        );
        assert_eq!(
            route_json.get("metal_direct_decode_residual_add_mlx_metal_dispatch_share"),
            Some(&Value::from(1.0_f64))
        );
    }

    #[test]
    fn serialize_route_metadata_preserves_mlx_kv_cache_counters() {
        let mut route = RouteMetadata::empty();
        route.crossover_decisions.push((
            ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB.to_string(),
            24,
        ));
        route.crossover_decisions.push((
            ax_engine_core::ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT.to_string(),
            3,
        ));
        route.crossover_decisions.push((
            ax_engine_core::ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB.to_string(),
            8,
        ));

        let route_json = serialize_route_metadata(&route);
        let decisions = route_json
            .get("crossover_decisions")
            .and_then(Value::as_object)
            .expect("route decisions should serialize as an object");

        assert_eq!(
            decisions.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB),
            Some(&Value::from(24))
        );
        assert_eq!(
            decisions.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
            Some(&Value::from(3))
        );
        assert_eq!(
            decisions
                .get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB),
            Some(&Value::from(8))
        );
        assert_eq!(
            route_json.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB),
            Some(&Value::from(24))
        );
        assert_eq!(
            route_json.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT),
            Some(&Value::from(3))
        );
        assert_eq!(
            route_json
                .get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB),
            Some(&Value::from(8))
        );
    }

    #[test]
    fn trusted_baseline_json_infers_tool_mode_from_runtime_when_software_tool_mode_is_missing() {
        let manifest = json!({
            "id": "chat_qwen_short",
            "class": "scenario",
            "scenario": "chat",
            "model": {
                "family": "qwen3_dense"
            }
        });
        let mut environment = native_environment_fixture();
        environment
            .get_mut("software")
            .and_then(Value::as_object_mut)
            .expect("fixture software should serialize as object")
            .remove("tool_mode");
        let metrics = json!({
            "run_id": "run-a",
            "metrics": {
                "ttft_ms": 10.0,
                "decode_tok_s": 120.0,
                "memory_peak_mb": 512.0,
                "prefix_hit_rate": 0.0
            }
        });

        let baseline = build_trusted_baseline_json(
            "Dense Qwen M4 Baseline",
            "Dense-Qwen-M4-Baseline",
            Path::new("/tmp/ax-engine-bench-baseline"),
            &manifest,
            &environment,
            &metrics,
        )
        .expect("trusted baseline JSON should build");

        assert_eq!(
            nested_value(&baseline, &["runtime", "tool_mode"]).and_then(Value::as_str),
            Some("engine_bringup_runtime")
        );
        assert_eq!(
            nested_value(&baseline, &["route", "execution_semantics"]).and_then(Value::as_str),
            Some("none_observed")
        );
    }

    #[test]
    fn trusted_baseline_json_preserves_native_model_provenance() {
        let manifest = json!({
            "id": "chat_qwen_short",
            "class": "scenario",
            "scenario": "chat",
            "model": {
                "family": "qwen3_dense"
            }
        });
        let mut environment = native_environment_fixture();
        environment["runtime"]["mlx_model"] = json!({
            "artifacts_source": "explicit_config",
            "model_family": "qwen3_dense",
            "tensor_format": "safetensors",
            "layer_count": 36,
            "tensor_count": 402,
            "tie_word_embeddings": false,
            "bindings_prepared": false,
            "buffers_bound": false,
            "buffer_count": 0,
            "buffer_bytes": 0
        });
        let metrics = json!({
            "run_id": "run-native-model",
            "metrics": {
                "ttft_ms": 10.0,
                "decode_tok_s": 120.0,
                "memory_peak_mb": 512.0,
                "prefix_hit_rate": 0.0
            }
        });

        let baseline = build_trusted_baseline_json(
            "Dense Qwen Native Model Baseline",
            "Dense-Qwen-Native-Model-Baseline",
            Path::new("/tmp/ax-engine-bench-baseline"),
            &manifest,
            &environment,
            &metrics,
        )
        .expect("trusted baseline JSON should build");

        assert_eq!(
            nested_value(&baseline, &["runtime", "mlx_model", "artifacts_source"])
                .and_then(Value::as_str),
            Some("explicit_config")
        );
        assert_eq!(
            nested_value(&baseline, &["runtime", "mlx_model", "tensor_format"])
                .and_then(Value::as_str),
            Some("safetensors")
        );
        assert_eq!(
            nested_value(&baseline, &["runtime", "mlx_model", "tensor_count"])
                .and_then(Value::as_u64),
            Some(402)
        );
    }

    #[test]
    fn matrix_compare_member_result_infers_deterministic_native_tool_mode_from_runtime() {
        let baseline_member = json!({
            "label": "Chat Qwen Short"
        });
        let regression = json!({
            "runtime": mlx_runtime_fixture(),
            "summary": {
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only"
            },
            "comparison": {
                "ttft_ms_pct": -5.0,
                "decode_tok_s_pct": 8.0,
                "memory_peak_mb_pct": 0.0,
                "prefix_hit_rate_pct": 0.0
            }
        });

        let member = build_matrix_compare_member_result(
            &baseline_member,
            "chat_qwen_short",
            Path::new("/tmp/ax-engine-bench-compare-member"),
            &regression,
        )
        .expect("member result should build");

        assert_eq!(member.tool_mode, "engine_bringup_runtime");
        assert_eq!(member.compare_mode, "engine_bringup_compare");
        assert_eq!(member.execution_semantics, "unknown");
    }

    #[test]
    fn serialize_runtime_metadata_includes_native_model_provenance() {
        let model_dir = write_valid_native_model_fixture();
        let runtime = RuntimeConfig {
            deterministic: false,
            max_batch_tokens: 128,
            block_size_tokens: 16,
            kv_total_blocks: Some(32),
            flags: RuntimeFlags::default(),
            llama_cpp_preset: None,
            backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
            resolved_backend: ResolvedBackend::new(
                SelectedBackend::Mlx,
                SupportTier::MlxPreview,
                None,
            ),
            backend_adapter: None,
            mlx_model_artifacts_dir: Some(model_dir.clone()),
            mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
        };

        let runtime_json = serialize_runtime_metadata(&runtime, None);

        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("artifacts_source"))
                .and_then(Value::as_str),
            Some("explicit_config")
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("model_family"))
                .and_then(Value::as_str),
            Some("qwen3_dense")
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("layer_count"))
                .and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("tensor_count"))
                .and_then(Value::as_u64),
            Some(9)
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("bindings_prepared"))
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("buffers_bound"))
                .and_then(Value::as_bool),
            Some(false)
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn serialize_runtime_metadata_keeps_native_model_summary_when_runtime_binding_fails() {
        let model_dir = write_valid_native_model_fixture();
        let _missing_build_dir = unique_test_dir("missing-native-runtime-build");
        let runtime = RuntimeConfig {
            deterministic: false,
            max_batch_tokens: 128,
            block_size_tokens: 16,
            kv_total_blocks: Some(32),
            flags: RuntimeFlags::default(),
            llama_cpp_preset: None,
            backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
            resolved_backend: ResolvedBackend::new(
                SelectedBackend::Mlx,
                SupportTier::MlxPreview,
                None,
            ),
            backend_adapter: None,
            mlx_model_artifacts_dir: Some(model_dir.clone()),
            mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
        };

        let runtime_json = serialize_runtime_metadata(&runtime, None);

        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("model_family"))
                .and_then(Value::as_str),
            Some("qwen3_dense")
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("bindings_prepared"))
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("buffers_bound"))
                .and_then(Value::as_bool),
            Some(false)
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn serialize_runtime_metadata_omits_config_side_model_summary_for_file_path() {
        let runtime = RuntimeConfig {
            deterministic: false,
            max_batch_tokens: 128,
            block_size_tokens: 16,
            kv_total_blocks: Some(32),
            flags: RuntimeFlags::default(),
            llama_cpp_preset: None,
            backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
            resolved_backend: ResolvedBackend::new(
                SelectedBackend::Mlx,
                SupportTier::MlxPreview,
                None,
            ),
            backend_adapter: None,
            mlx_model_artifacts_dir: Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf")),
            mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
        };

        let runtime_json = serialize_runtime_metadata(&runtime, None);

        assert_eq!(
            runtime_json
                .get("mlx_runtime")
                .and_then(|value| value.get("runner"))
                .and_then(Value::as_str),
            None
        );
        assert_eq!(runtime_json.get("mlx_model"), None);
    }

    #[test]
    fn serialize_runtime_metadata_prefers_actual_runtime_report_over_config_side_metadata() {
        let model_dir = write_valid_native_model_fixture();
        let runtime = RuntimeConfig {
            deterministic: false,
            max_batch_tokens: 128,
            block_size_tokens: 16,
            kv_total_blocks: Some(32),
            flags: RuntimeFlags::default(),
            llama_cpp_preset: None,
            backend_policy: BackendPolicy::new(ResolutionPolicy::MlxOnly),
            resolved_backend: ResolvedBackend::new(
                SelectedBackend::Mlx,
                SupportTier::MlxPreview,
                None,
            ),
            backend_adapter: None,
            mlx_model_artifacts_dir: Some(model_dir.clone()),
            mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
        };
        let actual_runtime = RuntimeReport {
            selected_backend: SelectedBackend::Mlx,
            support_tier: SupportTier::MlxCertified,
            resolution_policy: ResolutionPolicy::MlxOnly,
            capabilities: ResolvedBackend::mlx_certified().capabilities,
            fallback_reason: None,
            host: HostReport {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                detected_soc: Some("Apple M4 Max".to_string()),
                supported_mlx_runtime: true,
                unsupported_host_override_active: false,
            },
            metal_toolchain: MetalToolchainReport {
                fully_available: true,
                metal: ToolStatusReport {
                    available: true,
                    version: Some("metal actual".to_string()),
                },
                metallib: ToolStatusReport {
                    available: true,
                    version: Some("metallib actual".to_string()),
                },
                metal_ar: ToolStatusReport {
                    available: true,
                    version: Some("metal-ar actual".to_string()),
                },
            },
            mlx_runtime: Some(NativeRuntimeReport::metal_bringup(
                NativeRuntimeArtifactsSource::ExplicitConfig,
            )),
            mlx_model: Some(NativeModelReport {
                artifacts_source: NativeModelArtifactsSource::ExplicitEnv,
                model_family: "actual_bound_model".to_string(),
                tensor_format: ax_engine_core::NativeTensorFormat::Safetensors,
                source_quantization: None,
                runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
                layer_count: 42,
                tensor_count: 420,
                tie_word_embeddings: true,
                bindings_prepared: true,
                buffers_bound: true,
                buffer_count: 12,
                buffer_bytes: 4096,
                source_quantized_binding_count: 0,
                source_q4_k_binding_count: 0,
                source_q5_k_binding_count: 0,
                source_q6_k_binding_count: 0,
                source_q8_0_binding_count: 0,
            }),
        };

        let runtime_json = serialize_runtime_metadata(&runtime, Some(&actual_runtime));

        assert_eq!(
            runtime_json.get("support_tier").and_then(Value::as_str),
            Some("mlx_certified")
        );
        assert_eq!(
            runtime_json
                .get("capabilities")
                .and_then(|value| value.get("long_context_validation"))
                .and_then(Value::as_str),
            Some("supported")
        );
        assert_eq!(
            runtime_json
                .get("metal_toolchain")
                .and_then(|value| value.get("metal"))
                .and_then(|value| value.get("version"))
                .and_then(Value::as_str),
            Some("metal actual")
        );
        assert_eq!(
            runtime_json
                .get("mlx_runtime")
                .and_then(|value| value.get("runner"))
                .and_then(Value::as_str),
            Some("metal_bringup")
        );
        assert_eq!(
            runtime_json
                .get("mlx_runtime")
                .and_then(|value| value.get("artifacts_source"))
                .and_then(Value::as_str),
            Some("explicit_config")
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("artifacts_source"))
                .and_then(Value::as_str),
            Some("explicit_env")
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("model_family"))
                .and_then(Value::as_str),
            Some("actual_bound_model")
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("bindings_prepared"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            runtime_json
                .get("mlx_model")
                .and_then(|value| value.get("buffers_bound"))
                .and_then(Value::as_bool),
            Some(true)
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    fn llama_runtime_fixture(server_url: &str) -> Value {
        json!({
            "selected_backend": "llama_cpp",
            "support_tier": "llama_cpp",
            "resolution_policy": "allow_llama_cpp",
            "backend_adapter": {
                "kind": "llama_cpp_server_completion",
                "server_url": server_url
            },
            "host": runtime_host_fixture(),
            "metal_toolchain": runtime_metal_toolchain_fixture()
        })
    }

    fn runtime_host_fixture() -> Value {
        json!({
            "os": "macos",
            "arch": "aarch64",
            "detected_soc": "Apple M4 Max",
            "supported_mlx_runtime": true,
            "unsupported_host_override_active": false
        })
    }

    fn runtime_metal_toolchain_fixture() -> Value {
        json!({
            "fully_available": true,
            "metal": {
                "available": true,
                "version": "Apple metal version 36000.4"
            },
            "metallib": {
                "available": true,
                "version": "Apple metallib version 36000.4"
            },
            "metal_ar": {
                "available": true,
                "version": "Apple metal-ar version 36000.4"
            }
        })
    }

    fn doctor_host_fixture(
        supported_mlx_runtime: bool,
        unsupported_host_override_active: bool,
        detected_soc: Option<&str>,
    ) -> HostReport {
        HostReport {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            detected_soc: detected_soc.map(str::to_string),
            supported_mlx_runtime,
            unsupported_host_override_active,
        }
    }

    fn doctor_tool_status_fixture(available: bool, version: Option<&str>) -> ToolStatusReport {
        ToolStatusReport {
            available,
            version: version.map(str::to_string),
        }
    }

    fn doctor_metal_toolchain_fixture(
        metal_available: bool,
        metallib_available: bool,
        metal_ar_available: bool,
    ) -> MetalToolchainReport {
        MetalToolchainReport {
            fully_available: metal_available && metallib_available,
            metal: doctor_tool_status_fixture(metal_available, Some("Apple metal version 36000.4")),
            metallib: doctor_tool_status_fixture(
                metallib_available,
                Some("Apple metallib version 36000.4"),
            ),
            metal_ar: doctor_tool_status_fixture(
                metal_ar_available,
                Some("Apple metal-ar version 36000.4"),
            ),
        }
    }

    #[test]
    fn doctor_report_marks_supported_m4_host_ready() {
        let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, true, true);

        let report = build_doctor_report(host, toolchain);

        assert_eq!(report.status, DoctorStatus::Ready);
        assert!(report.mlx_runtime_ready);
        assert!(report.bringup_allowed);
        assert!(report.issues.is_empty());
        assert_eq!(
            report.notes,
            vec!["llama.cpp backends do not widen supported host scope".to_string()]
        );
        assert!(
            report
                .performance_advice
                .iter()
                .any(|advice| advice.id == "ngram_acceleration_default_on")
        );
        assert!(
            report
                .performance_advice
                .iter()
                .any(|advice| advice.id == "swiftlm_is_baseline_only")
        );
    }

    #[test]
    fn doctor_report_marks_override_host_as_bringup_only() {
        let host = doctor_host_fixture(false, true, Some("Apple M3 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, true, true);

        let report = build_doctor_report(host, toolchain);

        assert_eq!(report.status, DoctorStatus::BringupOnly);
        assert!(!report.mlx_runtime_ready);
        assert!(report.bringup_allowed);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("Apple M3 Max"))
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("AX_ALLOW_UNSUPPORTED_HOST"))
        );
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("development or CI bring-up"))
        );
    }

    #[test]
    fn doctor_report_marks_missing_metal_toolchain_not_ready() {
        let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, false, false);

        let report = build_doctor_report(host, toolchain);

        assert_eq!(report.status, DoctorStatus::NotReady);
        assert!(!report.mlx_runtime_ready);
        assert!(!report.bringup_allowed);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.contains("xcrun metallib"))
        );
        assert!(
            !report
                .issues
                .iter()
                .any(|issue| issue.contains("xcrun metal-ar"))
        );
    }

    #[test]
    fn doctor_report_accepts_command_line_tools_without_metal_ar() {
        let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, true, false);

        let report = build_doctor_report(host, toolchain);

        assert_eq!(report.status, DoctorStatus::Ready);
        assert!(report.mlx_runtime_ready);
        assert!(report.bringup_allowed);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn render_doctor_report_includes_status_and_issue_sections() {
        let host = doctor_host_fixture(false, true, Some("Apple M3 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, false, true);
        let report = build_doctor_report(host, toolchain);

        let text = render_doctor_report(&report);

        assert!(text.contains("AX Engine v4 doctor"));
        assert!(text.contains("status=not_ready"));
        assert!(text.contains("host.detected_soc=Apple M3 Max"));
        assert!(text.contains("issues:"));
        assert!(text.contains("notes:"));
        assert!(text.contains("performance_advice:"));
        assert!(text.contains("ngram_acceleration_default_on"));
        assert!(text.contains("llama.cpp backends do not widen supported host scope"));
    }

    #[test]
    fn doctor_report_adds_qwen36_quantization_advice_from_model_config() {
        let root = unique_test_dir("doctor-qwen36");
        let model_dir = root.join("Qwen3.6-35B-A3B-UD-MLX-4bit");
        fs::create_dir_all(&model_dir).expect("model dir should create");
        fs::write(
            model_dir.join("config.json"),
            r#"{
                "model_type": "qwen3_next",
                "quantization": {
                    "mode": "affine",
                    "group_size": 64,
                    "bits": 4
                }
            }"#,
        )
        .expect("config should write");
        write_doctor_model_manifest(&model_dir);
        let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, true, true);

        let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

        assert!(
            report
                .performance_advice
                .iter()
                .any(|advice| advice.id == "qwen36_quantization_compare"
                    && advice.severity == DoctorAdviceSeverity::Warning)
        );
        assert!(
            report
                .performance_advice
                .iter()
                .any(|advice| advice.id == "qwen_gated_delta_prefill_scope")
        );

        fs::remove_dir_all(root).expect("test dir should clean up");
    }

    #[test]
    fn doctor_report_adds_gemma4_quantization_advice_from_model_config() {
        let root = unique_test_dir("doctor-gemma4");
        let model_dir = root.join("gemma-4-e2b-it-4bit");
        fs::create_dir_all(&model_dir).expect("model dir should create");
        fs::write(
            model_dir.join("config.json"),
            r#"{
                "model_type": "gemma4",
                "quantization_config": {
                    "mode": "affine",
                    "group_size": 64,
                    "bits": 4
                }
            }"#,
        )
        .expect("config should write");
        write_doctor_model_manifest(&model_dir);
        let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, true, true);

        let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

        assert!(
            report
                .performance_advice
                .iter()
                .any(|advice| advice.id == "gemma4_4bit_first")
        );

        fs::remove_dir_all(root).expect("test dir should clean up");
    }

    #[test]
    fn doctor_report_points_plain_mlx_snapshot_to_manifest_generation() {
        let root = unique_test_dir("doctor-missing-manifest");
        let model_dir = root.join("plain-mlx-snapshot");
        fs::create_dir_all(&model_dir).expect("model dir should create");
        fs::write(
            model_dir.join("config.json"),
            r#"{"model_type":"qwen3","quantization":{"mode":"affine","group_size":64,"bits":4}}"#,
        )
        .expect("config should write");
        let host = doctor_host_fixture(true, false, Some("Apple M4 Max"));
        let toolchain = doctor_metal_toolchain_fixture(true, true, true);

        let report = build_doctor_report_for_model(host, toolchain, Some(&model_dir));

        assert!(report.performance_advice.iter().any(|advice| {
            advice.id == "model_artifacts_unreadable"
                && advice.detail.contains("model-manifest.json")
                && advice.detail.contains("generate-manifest")
        }));

        fs::remove_dir_all(root).expect("test dir should clean up");
    }

    #[test]
    fn parse_doctor_args_accepts_json_and_rejects_unknown_flags() {
        let args = vec![
            "--json".to_string(),
            "--mlx-model-artifacts-dir".to_string(),
            "/tmp/google_gemma-4-26b-it-q4_k_m.gguf".to_string(),
        ];
        let parsed = parse_doctor_args(&args).expect("doctor args should parse");
        assert!(parsed.json);
        assert_eq!(
            parsed.mlx_model_artifacts_dir,
            Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"))
        );

        let error = parse_doctor_args(&["--bogus".to_string()])
            .expect_err("doctor args should reject unknown flags");
        let CliError::Usage(message) = error else {
            panic!("doctor args should surface a usage error");
        };
        assert!(message.contains("unknown flag for doctor"));
        assert!(
            message.contains("ax-engine-bench doctor [--json] [--mlx-model-artifacts-dir <path>]")
        );
    }

    #[test]
    fn parse_autotune_args_accepts_defaults_and_explicit_overrides() {
        let parsed = parse_autotune_args(&[
            "--manifest".to_string(),
            "/tmp/manifest.json".to_string(),
            "--output-root".to_string(),
            "/tmp/out".to_string(),
            "--iterations".to_string(),
            "5".to_string(),
            "--exploration-weight".to_string(),
            "1.25".to_string(),
            "--max-batch-token-options".to_string(),
            "64,128,256".to_string(),
            "--kv-total-block-options".to_string(),
            "none,32,64".to_string(),
            "--prefix-cache-options".to_string(),
            "true,false".to_string(),
        ])
        .expect("autotune args should parse");

        assert_eq!(
            parsed,
            AutotuneArgs {
                manifest_path: PathBuf::from("/tmp/manifest.json"),
                output_root: PathBuf::from("/tmp/out"),
                iterations: 5,
                exploration_weight: 1.25,
                max_batch_token_options: Some(vec![64, 128, 256]),
                kv_total_block_options: Some(vec![None, Some(32), Some(64)]),
                prefix_cache_options: Some(vec![false, true]),
                disable_history: false,
            }
        );
    }

    #[test]
    fn parse_autotune_args_accepts_disable_history_flag() {
        let parsed = parse_autotune_args(&[
            "--manifest".to_string(),
            "/tmp/manifest.json".to_string(),
            "--output-root".to_string(),
            "/tmp/out".to_string(),
            "--disable-history".to_string(),
        ])
        .expect("autotune args should parse");

        assert!(parsed.disable_history);
    }

    #[test]
    fn autotune_candidate_configs_keep_manifest_runtime_as_first_trial() {
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );

        let search_space = resolve_autotune_search_space(
            &manifest,
            &AutotuneArgs {
                manifest_path: PathBuf::from("/tmp/manifest.json"),
                output_root: PathBuf::from("/tmp/out"),
                iterations: 8,
                exploration_weight: 0.5,
                max_batch_token_options: None,
                kv_total_block_options: None,
                prefix_cache_options: None,
                disable_history: false,
            },
        );
        let candidates = autotune_candidate_configs(&manifest, &search_space);

        assert!(!candidates.is_empty());
        assert_eq!(
            candidates[0],
            AutotuneCandidateConfig {
                max_batch_tokens: manifest.runtime.max_batch_tokens,
                kv_total_blocks: manifest.runtime.kv_total_blocks,
                prefix_cache: manifest.runtime.flags.prefix_cache,
            }
        );
    }

    #[test]
    fn autotune_candidate_configs_respect_explicit_search_space_overrides() {
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        let search_space = resolve_autotune_search_space(
            &manifest,
            &AutotuneArgs {
                manifest_path: PathBuf::from("/tmp/manifest.json"),
                output_root: PathBuf::from("/tmp/out"),
                iterations: 4,
                exploration_weight: 0.5,
                max_batch_token_options: Some(vec![64, 256]),
                kv_total_block_options: Some(vec![None, Some(48)]),
                prefix_cache_options: Some(vec![true]),
                disable_history: false,
            },
        );

        let candidates = autotune_candidate_configs(&manifest, &search_space);

        assert_eq!(search_space.max_batch_token_options, vec![64, 256]);
        assert_eq!(search_space.kv_total_block_options, vec![None, Some(48)]);
        assert_eq!(search_space.prefix_cache_options, vec![true]);
        assert_eq!(candidates.len(), 4);
        assert!(candidates.iter().all(|candidate| candidate.prefix_cache));
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.max_batch_tokens == 64
                    || candidate.max_batch_tokens == 256)
        );
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.kv_total_blocks.is_none()
                    || candidate.kv_total_blocks == Some(48))
        );
    }

    #[test]
    fn autotune_selector_picks_untried_candidate_after_base_trial() {
        let candidates = vec![
            AutotuneCandidateConfig {
                max_batch_tokens: 128,
                kv_total_blocks: Some(32),
                prefix_cache: false,
            },
            AutotuneCandidateConfig {
                max_batch_tokens: 256,
                kv_total_blocks: Some(32),
                prefix_cache: false,
            },
            AutotuneCandidateConfig {
                max_batch_tokens: 128,
                kv_total_blocks: Some(64),
                prefix_cache: true,
            },
        ];
        let search_space = AutotuneSearchSpace {
            max_batch_token_options: vec![128, 256],
            kv_total_block_options: vec![Some(32), Some(64)],
            prefix_cache_options: vec![false, true],
        };
        let trials = vec![AutotuneTrialRecord {
            trial_index: 0,
            candidate: candidates[0].clone(),
            selection: AutotuneSelectionDiagnostics {
                strategy: "base_config_seed".to_string(),
                predicted_mean: None,
                uncertainty: None,
                acquisition: None,
                good_density: None,
                bad_density: None,
                density_ratio: None,
                novelty_bonus: None,
            },
            probe: None,
            score: 42.0,
            status: "ok".to_string(),
            result_dir: None,
            error: None,
            metrics: None,
        }];

        let selected = select_next_autotune_candidate(&candidates, &search_space, &trials, 0.5);

        assert!(selected.candidate_index == 1 || selected.candidate_index == 2);
        assert_ne!(selected.candidate_index, 0);
        assert_eq!(selected.diagnostics.strategy, "coverage_explore");
        assert!(selected.diagnostics.predicted_mean.is_none());
        assert!(selected.diagnostics.uncertainty.is_some());
        assert!(selected.diagnostics.acquisition.is_some());
        assert!(selected.diagnostics.novelty_bonus.is_some());
    }

    #[test]
    fn autotune_selector_uses_tpe_ratio_after_multiple_trials() {
        let candidates = vec![
            AutotuneCandidateConfig {
                max_batch_tokens: 128,
                kv_total_blocks: Some(32),
                prefix_cache: false,
            },
            AutotuneCandidateConfig {
                max_batch_tokens: 256,
                kv_total_blocks: Some(32),
                prefix_cache: false,
            },
            AutotuneCandidateConfig {
                max_batch_tokens: 256,
                kv_total_blocks: Some(64),
                prefix_cache: true,
            },
        ];
        let search_space = AutotuneSearchSpace {
            max_batch_token_options: vec![128, 256],
            kv_total_block_options: vec![Some(32), Some(64)],
            prefix_cache_options: vec![false, true],
        };
        let trials = vec![
            AutotuneTrialRecord {
                trial_index: 0,
                candidate: candidates[0].clone(),
                selection: AutotuneSelectionDiagnostics {
                    strategy: "base_config_seed".to_string(),
                    predicted_mean: None,
                    uncertainty: None,
                    acquisition: None,
                    good_density: None,
                    bad_density: None,
                    density_ratio: None,
                    novelty_bonus: None,
                },
                probe: None,
                score: 10.0,
                status: "ok".to_string(),
                result_dir: None,
                error: None,
                metrics: None,
            },
            AutotuneTrialRecord {
                trial_index: 1,
                candidate: candidates[1].clone(),
                selection: AutotuneSelectionDiagnostics {
                    strategy: "coverage_explore".to_string(),
                    predicted_mean: None,
                    uncertainty: Some(0.5),
                    acquisition: Some(0.5),
                    good_density: None,
                    bad_density: None,
                    density_ratio: None,
                    novelty_bonus: Some(0.5),
                },
                probe: None,
                score: 100.0,
                status: "ok".to_string(),
                result_dir: None,
                error: None,
                metrics: None,
            },
        ];

        let selected = select_next_autotune_candidate(&candidates, &search_space, &trials, 0.5);

        assert_eq!(selected.candidate_index, 2);
        assert_eq!(selected.diagnostics.strategy, "tpe_ratio");
        assert!(selected.diagnostics.predicted_mean.is_some());
        assert!(selected.diagnostics.good_density.is_some());
        assert!(selected.diagnostics.bad_density.is_some());
        assert!(selected.diagnostics.density_ratio.is_some());
        assert!(selected.diagnostics.novelty_bonus.is_some());
    }

    #[test]
    fn autotune_probe_manifest_reduces_shape_and_disables_determinism_check() {
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );

        let probe_manifest = autotune_probe_manifest(
            &manifest,
            0,
            &AutotuneCandidateConfig {
                max_batch_tokens: manifest.runtime.max_batch_tokens,
                kv_total_blocks: manifest.runtime.kv_total_blocks,
                prefix_cache: manifest.runtime.flags.prefix_cache,
            },
        )
        .expect("probe manifest should build");

        let original_shape = manifest.shape.expect("scenario shape should exist");
        let probe_shape = probe_manifest.shape.expect("probe shape should exist");
        assert!(probe_manifest.id.ends_with("-probe"));
        assert!(!probe_manifest.checks.expect_deterministic);
        assert!(probe_shape.input_tokens_target <= original_shape.input_tokens_target);
        assert!(probe_shape.output_tokens_target <= original_shape.output_tokens_target);
        assert!(probe_shape.concurrency <= original_shape.concurrency);
        assert!(probe_shape.output_tokens_target >= 1);
    }

    #[test]
    fn autotune_probe_skip_reason_flags_far_worse_cpu_fallback_probe() {
        let incumbent_metrics = AutotuneTrialMetrics {
            decode_tok_s: 100.0,
            mlx_metal_hot_path_cpu_fallback_free: true,
            real_model_forward: true,
            ..AutotuneTrialMetrics::default()
        };
        let observed_trials = vec![AutotuneTrialRecord {
            trial_index: 0,
            candidate: AutotuneCandidateConfig {
                max_batch_tokens: 128,
                kv_total_blocks: Some(32),
                prefix_cache: false,
            },
            selection: AutotuneSelectionDiagnostics {
                strategy: "base_config_seed".to_string(),
                predicted_mean: None,
                uncertainty: None,
                acquisition: None,
                good_density: None,
                bad_density: None,
                density_ratio: None,
                novelty_bonus: None,
            },
            probe: None,
            score: 1000.0,
            status: "ok".to_string(),
            result_dir: None,
            error: None,
            metrics: Some(incumbent_metrics),
        }];
        let execution = RuntimeResult {
            tool_mode: "engine_bringup_runtime",
            runtime: runtime_config_from_manifest(&load_test_manifest(
                repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
            ))
            .expect("runtime config should build"),
            observation: RuntimeObservation::default(),
            correctness: GateStatus::pass(),
            determinism: GateStatus::pass(),
        };
        let probe_metrics = AutotuneTrialMetrics {
            decode_tok_s: 40.0,
            prefix_cpu_reference_dispatch_count: 1,
            direct_decode_cpu_projection_row_count: 8,
            mlx_metal_hot_path_cpu_fallback_free: false,
            real_model_forward: true,
            ..AutotuneTrialMetrics::default()
        };
        let baseline = autotune_probe_baseline(&observed_trials);

        let reason =
            autotune_probe_skip_reason(&execution, &probe_metrics, baseline.as_ref(), 400.0);

        assert!(reason.is_some());
        let reason = reason.expect("skip reason should exist");
        assert!(
            reason.contains("CPU fallback")
                || reason.contains("fell well below incumbent")
                || reason.contains("adaptive floor"),
            "{reason}"
        );
    }

    #[test]
    fn autotune_probe_baseline_uses_observed_trial_distribution() {
        let observed_trials = vec![
            AutotuneTrialRecord {
                trial_index: 0,
                candidate: AutotuneCandidateConfig {
                    max_batch_tokens: 128,
                    kv_total_blocks: Some(32),
                    prefix_cache: false,
                },
                selection: AutotuneSelectionDiagnostics {
                    strategy: "base_config_seed".to_string(),
                    predicted_mean: None,
                    uncertainty: None,
                    acquisition: None,
                    good_density: None,
                    bad_density: None,
                    density_ratio: None,
                    novelty_bonus: None,
                },
                probe: None,
                score: 80.0,
                status: "ok".to_string(),
                result_dir: None,
                error: None,
                metrics: Some(AutotuneTrialMetrics {
                    decode_tok_s: 90.0,
                    mlx_metal_hot_path_cpu_fallback_free: true,
                    ttft_ms: Some(12),
                    ..AutotuneTrialMetrics::default()
                }),
            },
            AutotuneTrialRecord {
                trial_index: 1,
                candidate: AutotuneCandidateConfig {
                    max_batch_tokens: 256,
                    kv_total_blocks: Some(64),
                    prefix_cache: true,
                },
                selection: AutotuneSelectionDiagnostics {
                    strategy: "tpe_ratio".to_string(),
                    predicted_mean: Some(0.9),
                    uncertainty: Some(0.1),
                    acquisition: Some(1.0),
                    good_density: Some(0.6),
                    bad_density: Some(0.2),
                    density_ratio: Some(3.0),
                    novelty_bonus: Some(0.2),
                },
                probe: None,
                score: 120.0,
                status: "ok".to_string(),
                result_dir: None,
                error: None,
                metrics: Some(AutotuneTrialMetrics {
                    decode_tok_s: 110.0,
                    mlx_metal_hot_path_cpu_fallback_free: true,
                    ttft_ms: Some(10),
                    ..AutotuneTrialMetrics::default()
                }),
            },
        ];

        let baseline = autotune_probe_baseline(&observed_trials).expect("baseline should exist");

        assert_eq!(baseline.observed_trial_count, 2);
        assert_eq!(baseline.incumbent_score, Some(120.0));
        assert_eq!(baseline.incumbent_decode_tok_s, Some(110.0));
        assert!(baseline.score_floor.is_some());
        assert!(baseline.decode_tok_s_floor.is_some());
        assert!(baseline.fallback_free_decode_tok_s_floor.is_some());
        assert!(baseline.ttft_ceiling_ms.is_some());
    }

    #[test]
    fn usage_mentions_generate_and_stream_commands() {
        let text = usage();
        assert!(text.contains("ax-engine-bench generate"));
        assert!(text.contains("ax-engine-bench stream"));
        assert!(text.contains("ax-engine-bench autotune"));
    }

    #[test]
    fn parse_inference_args_accepts_llama_cpp_alias() {
        let args = vec![
            "--prompt".to_string(),
            "hello cli".to_string(),
            "--support-tier".to_string(),
            "llama_cpp".to_string(),
            "--llama-server-url".to_string(),
            "http://127.0.0.1:8081".to_string(),
            "--json".to_string(),
        ];

        let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");

        assert_eq!(parsed.input_text.as_deref(), Some("hello cli"));
        assert_eq!(parsed.support_tier, SupportTier::LlamaCpp);
        assert_eq!(
            parsed.llama_server_url.as_deref(),
            Some("http://127.0.0.1:8081")
        );
        assert!(parsed.json);
    }

    #[test]
    fn parse_inference_args_accepts_mlx_lm_delegated_server_url() {
        let args = vec![
            "--prompt".to_string(),
            "hello mlx-lm".to_string(),
            "--support-tier".to_string(),
            "mlx_lm_delegated".to_string(),
            "--mlx-lm-server-url".to_string(),
            "http://127.0.0.1:8090".to_string(),
        ];

        let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");

        assert_eq!(parsed.input_text.as_deref(), Some("hello mlx-lm"));
        assert_eq!(parsed.support_tier, SupportTier::MlxLmDelegated);
        assert_eq!(
            parsed.mlx_lm_server_url.as_deref(),
            Some("http://127.0.0.1:8090")
        );
    }

    #[test]
    fn parse_inference_args_accepts_explicit_mlx_model_artifacts_dir() {
        let args = vec![
            "--tokens".to_string(),
            "1,2,3".to_string(),
            "--mlx-model-artifacts-dir".to_string(),
            "/tmp/ax-model".to_string(),
        ];

        let parsed = parse_inference_args(&args, "generate").expect("inference args should parse");

        assert_eq!(
            parsed.mlx_model_artifacts_dir,
            Some(PathBuf::from("/tmp/ax-model"))
        );
    }

    fn native_metrics_fixture(run_id: &str, ttft_ms: f64, decode_tok_s: f64) -> Value {
        json!({
            "schema_version": "ax.engine_bench.metrics.v1",
            "run_id": run_id,
            "status": "pass",
            "runtime": mlx_runtime_fixture(),
            "metrics": {
                "ttft_ms": ttft_ms,
                "prefill_tok_s": 2048.0,
                "decode_tok_s": decode_tok_s,
                "e2e_latency_ms": 12.0,
                "memory_peak_mb": 512.0,
                "cpu_time_per_token_us": 10.0,
                "runner_time_per_token_us": 8.0,
                "runner_time_share_pct": 70.0,
                "prefix_hit_rate": 0.0,
                "kv_block_usage": 24.0,
                "evictions": 0.0
            },
            "correctness": {
                "passed": true,
                "reason": null
            },
            "determinism": {
                "passed": true,
                "reason": null
            },
            "step_count": 4,
            "scheduled_tokens_per_step": 64.0,
            "memory_blocked_steps": 0,
            "memory_blocked_request_events": 0,
            "replay_status": "not_applicable",
            "churn_status": "pass"
        })
    }

    fn write_execution_artifact_fixture(
        root: &Path,
        run_dir_name: &str,
        run_id: &str,
        ttft_ms: f64,
        decode_tok_s: f64,
    ) -> PathBuf {
        let run_dir = root.join(run_dir_name);
        fs::create_dir_all(&run_dir).expect("fixture run dir should create");
        write_json_file(
            &run_dir.join("manifest.json"),
            &load_test_manifest_json(
                repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
            ),
        )
        .expect("fixture manifest should write");
        write_json_file(
            &run_dir.join("environment.json"),
            &native_environment_fixture(),
        )
        .expect("fixture environment should write");
        write_json_file(
            &run_dir.join("metrics.json"),
            &native_metrics_fixture(run_id, ttft_ms, decode_tok_s),
        )
        .expect("fixture metrics should write");
        write_json_file(
            &run_dir.join("routes.json"),
            &json!({
                "schema_version": "ax.engine_bench.routes.v1",
                "run_id": run_id,
                "runtime": {
                    "selected_backend": "mlx",
                    "support_tier": "mlx_preview",
                    "resolution_policy": "mlx_only"
                },
                "route": {
                    "execution_plan": "mixed_step_plans",
                    "attention_route": "mixed_attention_routes",
                    "kv_mode": "paged_metadata",
                    "barrier_mode": "serial",
                    "prefix_cache_path": "metadata_lookup",
                    "prefix_cache_evidence": "none_observed",
                    "prefix_reuse_provenance": "none_observed"
                },
                "mode": "engine_bringup_runtime"
            }),
        )
        .expect("fixture routes should write");
        write_json_file(
            &run_dir.join("trace.json"),
            &json!({
                "schema_version": "ax.engine_bench.trace.v1",
                "run_id": run_id,
                "mode": "engine_bringup_runtime",
                "runtime": {
                    "selected_backend": "mlx",
                    "support_tier": "mlx_preview",
                    "resolution_policy": "mlx_only"
                },
                "steps": []
            }),
        )
        .expect("fixture trace should write");
        fs::write(
            run_dir.join("summary.md"),
            format!("# Benchmark Run\n\n- run_id: `{run_id}`\n- selected_backend: `mlx`\n"),
        )
        .expect("fixture summary should write");
        run_dir
    }

    fn write_matrix_manifest_fixture(root: &Path, members: &[Value]) -> PathBuf {
        let manifest_path = root.join("matrix.json");
        write_json_file(
            &manifest_path,
            &json!({
                "schema_version": "ax.engine_bench.matrix.v1",
                "id": "test_mlx_mlx_matrix",
                "class": "scenario_matrix",
                "members": members
            }),
        )
        .expect("matrix manifest should write");
        manifest_path
    }

    fn write_matrix_result_fixture(
        root: &Path,
        dir_name: &str,
        run_id: &str,
        members: &[Value],
    ) -> PathBuf {
        let result_dir = root.join(dir_name);
        fs::create_dir_all(&result_dir).expect("matrix result dir should create");
        write_json_file(
            &result_dir.join("matrix.json"),
            &json!({
                "schema_version": "ax.engine_bench.matrix_result.v1",
                "run_id": run_id,
                "id": "test_mlx_mlx_matrix",
                "class": "scenario_matrix",
                "status": "ok",
                "matrix_manifest_path": "/tmp/test-matrix.json",
                "matrix_manifest_fingerprint_fnv1a64": "abc123def4567890",
                "result_dir": result_dir.display().to_string(),
                "summary": {
                    "member_count": members.len(),
                    "ok_count": members.len(),
                    "completed_with_failures_count": 0,
                    "contract_failure_count": 0
                },
                "members": members
            }),
        )
        .expect("matrix result JSON should write");
        fs::write(
            result_dir.join("summary.md"),
            format!("# Benchmark Matrix\n\n- run_id: `{run_id}`\n"),
        )
        .expect("matrix summary should write");
        result_dir
    }

    fn route_decision(observation: &RuntimeObservation, key: &str) -> u32 {
        observation
            .route_metadata
            .crossover_decisions
            .iter()
            .find(|(decision, _)| decision == key)
            .map(|(_, value)| *value)
            .unwrap_or(0)
    }

    fn llama_cpp_scenario_manifest(server_url: &str) -> BenchmarkManifest {
        serde_json::from_value(json!({
            "schema_version": "ax.engine_bench.manifest.v1",
            "id": "llama_cpp_chat_qwen_short",
            "class": "scenario",
            "scenario": "chat",
            "model": {
                "family": "qwen3_dense",
                "revision": "phase1-llama-cpp",
                "quant": "q4_k_m",
                "tokenizer_revision": "qwen-tokenizer-v1",
                "chat_template_revision": "chatml-v1"
            },
            "runtime": {
                "selected_backend": "llama_cpp",
                "support_tier": "llama_cpp",
                "resolution_policy": "allow_llama_cpp",
                "fallback_reason": "benchmark requested delegated runtime",
                "backend_adapter": {
                    "kind": "llama_cpp_server_completion",
                    "server_url": server_url
                },
                "deterministic": true,
                "max_batch_tokens": 2048,
                "flags": {
                    "prefix_cache": false
                }
            },
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 1234
            },
            "shape": {
                "input_tokens_target": 32,
                "output_tokens_target": 4,
                "concurrency": 1
            },
            "checks": {
                "expect_deterministic": true,
                "require_prefix_reuse": false
            }
        }))
        .expect("llama.cpp manifest should parse")
    }

    fn llama_cpp_shared_prefix_scenario_manifest(server_url: &str) -> BenchmarkManifest {
        let mut manifest = llama_cpp_scenario_manifest(server_url);
        manifest.id = "llama_cpp_shared_prefix_qwen_short".to_string();
        manifest.scenario = "shared_prefix".to_string();
        manifest
            .shape
            .as_mut()
            .expect("shape should exist")
            .input_tokens_target = 128;
        manifest
            .shape
            .as_mut()
            .expect("shape should exist")
            .concurrency = 2;
        manifest.checks.require_prefix_reuse = true;
        manifest
    }

    fn llama_cpp_server_scenario_manifest(server_url: &str) -> BenchmarkManifest {
        let mut manifest = llama_cpp_scenario_manifest(server_url);
        manifest.id = "llama_cpp_chat_qwen_short".to_string();
        manifest.runtime.selected_backend = SelectedBackend::LlamaCpp;
        manifest.runtime.backend_adapter = Some(BackendAdapterManifest::LlamaCppServerCompletion {
            server_url: server_url.to_string(),
        });
        manifest.checks.expect_deterministic = false;
        manifest
    }

    fn write_named_autotune_history_fixture(
        output_root: &Path,
        dir_name: &str,
        manifest: &BenchmarkManifest,
        candidate: &AutotuneCandidateConfig,
        score: f64,
    ) -> PathBuf {
        let result_dir = output_root.join(dir_name);
        fs::create_dir_all(&result_dir).expect("historical autotune dir should create");
        write_json_file(
            &result_dir.join("autotune.json"),
            &json!({
                "schema_version": AUTOTUNE_SCHEMA_VERSION,
                "manifest_path": "/tmp/historical-manifest.json",
                "manifest_id": manifest.id,
                "selected_backend": selected_backend_label(manifest.runtime.selected_backend),
                "iterations_requested": 1,
                "iterations_completed": 1,
                "exploration_weight": 0.5,
                "warm_start_history": {
                    "enabled": true,
                    "loaded_trial_count": 0,
                    "loaded_result_count": 0,
                    "source_dirs": []
                },
                "search_space": {
                    "max_batch_tokens": [candidate.max_batch_tokens],
                    "kv_total_blocks": [candidate.kv_total_blocks],
                    "prefix_cache": [candidate.prefix_cache],
                    "candidate_count": 1
                },
                "trials": [{
                    "label": "trial-001",
                    "score": score,
                    "status": "ok",
                    "config": {
                        "max_batch_tokens": candidate.max_batch_tokens,
                        "kv_total_blocks": candidate.kv_total_blocks,
                        "prefix_cache": candidate.prefix_cache
                    },
                    "selection": {
                        "strategy": "base_config_seed",
                        "predicted_mean": null,
                        "uncertainty": null,
                        "acquisition": null,
                        "good_density": null,
                        "bad_density": null,
                        "density_ratio": null,
                        "novelty_bonus": null
                    },
                    "result_dir": null,
                    "error": null,
                    "metrics": {
                        "ttft_ms": 5,
                        "prefill_tok_s": 1000.0,
                        "decode_tok_s": 500.0,
                        "prefix_hit_rate": 0.0,
                        "prefix_native_dispatch_count": 0,
                        "prefix_cpu_reference_dispatch_count": 0,
                        "direct_decode_batched_group_fallback_count": 0,
                        "mlx_metal_hot_path_cpu_fallback_free": true,
                        "real_model_forward": true,
                        "model_bound_ffn_decode": true
                    }
                }]
            }),
        )
        .expect("historical autotune JSON should write");
        result_dir
    }

    fn write_autotune_history_fixture(
        output_root: &Path,
        manifest: &BenchmarkManifest,
        candidate: &AutotuneCandidateConfig,
        score: f64,
    ) -> PathBuf {
        write_named_autotune_history_fixture(
            output_root,
            "historical-autotune",
            manifest,
            candidate,
            score,
        )
    }

    fn find_fresh_autotune_result_dir(output_root: &Path, excluded_dirs: &[&Path]) -> PathBuf {
        fs::read_dir(output_root)
            .expect("autotune output root should be readable")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .find(|path| excluded_dirs.iter().all(|excluded| path != *excluded))
            .expect("fresh autotune result dir should exist")
    }

    #[test]
    fn load_autotune_warm_start_history_prefers_index_summary_when_present() {
        let root = unique_test_dir("autotune-history-index");
        let output_root = root.join("out");
        fs::create_dir_all(&output_root).expect("autotune output root should create");
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        let search_space = resolve_autotune_search_space(
            &manifest,
            &AutotuneArgs {
                manifest_path: PathBuf::from("/tmp/manifest.json"),
                output_root: output_root.clone(),
                iterations: 1,
                exploration_weight: 0.5,
                max_batch_token_options: None,
                kv_total_block_options: None,
                prefix_cache_options: None,
                disable_history: false,
            },
        );
        let indexed_result_dir = output_root.join("indexed-result");
        write_json_file(
            &autotune_history_index_path(&output_root),
            &json!({
                "schema_version": AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION,
                "entry_count": 1,
                "summary_count": 1,
                "entries": [{
                    "result_dir": output_root.join("stale-entry").display().to_string(),
                    "manifest_id": "wrong-manifest",
                    "selected_backend": "mlx",
                    "trials": []
                }],
                "summaries": [{
                    "manifest_id": manifest.id,
                    "selected_backend": selected_backend_label(manifest.runtime.selected_backend),
                    "source_dirs": [indexed_result_dir.display().to_string()],
                    "best_trials": [{
                        "label": "trial-001",
                        "score": 77.0,
                        "status": "ok",
                        "config": {
                            "max_batch_tokens": manifest.runtime.max_batch_tokens,
                            "kv_total_blocks": manifest.runtime.kv_total_blocks,
                            "prefix_cache": manifest.runtime.flags.prefix_cache
                        },
                        "selection": {
                            "strategy": "base_config_seed",
                            "predicted_mean": null,
                            "uncertainty": null,
                            "acquisition": null,
                            "good_density": null,
                            "bad_density": null,
                            "density_ratio": null,
                            "novelty_bonus": null
                        },
                        "result_dir": indexed_result_dir.display().to_string(),
                        "error": null,
                        "metrics": {
                            "ttft_ms": 5,
                            "prefill_tok_s": 1000.0,
                            "decode_tok_s": 500.0,
                            "prefix_hit_rate": 0.0,
                            "prefix_native_dispatch_count": 0,
                            "prefix_cpu_reference_dispatch_count": 0,
                            "direct_decode_batched_group_fallback_count": 0,
                            "mlx_metal_hot_path_cpu_fallback_free": true,
                            "real_model_forward": true,
                            "model_bound_ffn_decode": true
                        }
                    }]
                }]
            }),
        )
        .expect("history index should write");

        let history = load_autotune_warm_start_history(&output_root, &manifest, &search_space)
            .expect("warm-start history should load from index");

        assert_eq!(history.trials.len(), 1);
        assert_eq!(history.trials[0].score, 77.0);
        assert_eq!(history.source_dirs, vec![indexed_result_dir]);
    }

    #[test]
    fn load_autotune_warm_start_history_falls_back_to_index_entries_when_summary_missing() {
        let root = unique_test_dir("autotune-history-index-entries");
        let output_root = root.join("out");
        fs::create_dir_all(&output_root).expect("autotune output root should create");
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        let search_space = resolve_autotune_search_space(
            &manifest,
            &AutotuneArgs {
                manifest_path: PathBuf::from("/tmp/manifest.json"),
                output_root: output_root.clone(),
                iterations: 1,
                exploration_weight: 0.5,
                max_batch_token_options: None,
                kv_total_block_options: None,
                prefix_cache_options: None,
                disable_history: false,
            },
        );
        let indexed_result_dir = output_root.join("indexed-entry-result");
        write_json_file(
            &autotune_history_index_path(&output_root),
            &json!({
                "schema_version": AUTOTUNE_HISTORY_INDEX_SCHEMA_VERSION,
                "entry_count": 1,
                "entries": [{
                    "result_dir": indexed_result_dir.display().to_string(),
                    "manifest_id": manifest.id,
                    "selected_backend": selected_backend_label(manifest.runtime.selected_backend),
                    "trials": [{
                        "label": "trial-001",
                        "score": 66.0,
                        "status": "ok",
                        "config": {
                            "max_batch_tokens": manifest.runtime.max_batch_tokens,
                            "kv_total_blocks": manifest.runtime.kv_total_blocks,
                            "prefix_cache": manifest.runtime.flags.prefix_cache
                        },
                        "selection": {
                            "strategy": "base_config_seed",
                            "predicted_mean": null,
                            "uncertainty": null,
                            "acquisition": null,
                            "good_density": null,
                            "bad_density": null,
                            "density_ratio": null,
                            "novelty_bonus": null
                        },
                        "result_dir": indexed_result_dir.display().to_string(),
                        "error": null,
                        "metrics": {
                            "ttft_ms": 5,
                            "prefill_tok_s": 1000.0,
                            "decode_tok_s": 500.0,
                            "prefix_hit_rate": 0.0,
                            "prefix_native_dispatch_count": 0,
                            "prefix_cpu_reference_dispatch_count": 0,
                            "direct_decode_batched_group_fallback_count": 0,
                            "mlx_metal_hot_path_cpu_fallback_free": true,
                            "real_model_forward": true,
                            "model_bound_ffn_decode": true
                        }
                    }]
                }]
            }),
        )
        .expect("history index should write");

        let history = load_autotune_warm_start_history(&output_root, &manifest, &search_space)
            .expect("warm-start history should load from entry fallback");

        assert_eq!(history.trials.len(), 1);
        assert_eq!(history.trials[0].score, 66.0);
        assert_eq!(history.source_dirs, vec![indexed_result_dir]);
    }

    #[test]
    fn load_autotune_warm_start_history_falls_back_to_directory_scan_when_index_missing() {
        let root = unique_test_dir("autotune-history-scan");
        let output_root = root.join("out");
        fs::create_dir_all(&output_root).expect("autotune output root should create");
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        let search_space = resolve_autotune_search_space(
            &manifest,
            &AutotuneArgs {
                manifest_path: PathBuf::from("/tmp/manifest.json"),
                output_root: output_root.clone(),
                iterations: 1,
                exploration_weight: 0.5,
                max_batch_token_options: None,
                kv_total_block_options: None,
                prefix_cache_options: None,
                disable_history: false,
            },
        );
        let result_dir = write_autotune_history_fixture(
            &output_root,
            &manifest,
            &AutotuneCandidateConfig {
                max_batch_tokens: manifest.runtime.max_batch_tokens,
                kv_total_blocks: manifest.runtime.kv_total_blocks,
                prefix_cache: manifest.runtime.flags.prefix_cache,
            },
            55.0,
        );

        let history = load_autotune_warm_start_history(&output_root, &manifest, &search_space)
            .expect("warm-start history should load from directory scan");

        assert_eq!(history.trials.len(), 1);
        assert_eq!(history.trials[0].score, 55.0);
        assert_eq!(history.source_dirs, vec![result_dir]);
    }

    #[test]
    fn write_autotune_history_index_incremental_merges_new_result_into_existing_index() {
        let root = unique_test_dir("autotune-history-incremental");
        let output_root = root.join("out");
        fs::create_dir_all(&output_root).expect("autotune output root should create");
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        let historical_dir = write_named_autotune_history_fixture(
            &output_root,
            "historical-autotune-a",
            &manifest,
            &AutotuneCandidateConfig {
                max_batch_tokens: manifest.runtime.max_batch_tokens,
                kv_total_blocks: manifest.runtime.kv_total_blocks,
                prefix_cache: manifest.runtime.flags.prefix_cache,
            },
            55.0,
        );
        write_autotune_history_index(&output_root).expect("history index should build");
        let new_dir = write_named_autotune_history_fixture(
            &output_root,
            "historical-autotune-b",
            &manifest,
            &AutotuneCandidateConfig {
                max_batch_tokens: manifest.runtime.max_batch_tokens.saturating_mul(2),
                kv_total_blocks: manifest.runtime.kv_total_blocks,
                prefix_cache: !manifest.runtime.flags.prefix_cache,
            },
            88.0,
        );
        let new_result_json =
            load_json_value(&new_dir.join("autotune.json")).expect("new autotune JSON should load");

        write_autotune_history_index_incremental(&output_root, &new_dir, &new_result_json)
            .expect("incremental history index should update");

        let history_index = load_json_value(&autotune_history_index_path(&output_root))
            .expect("history index should load");
        assert_eq!(
            history_index.get("entry_count").and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            history_index.get("summary_count").and_then(Value::as_u64),
            Some(1)
        );
        let source_dirs = history_index
            .get("summaries")
            .and_then(Value::as_array)
            .and_then(|summaries| summaries.first())
            .and_then(|summary| summary.get("source_dirs"))
            .and_then(Value::as_array)
            .expect("summary source dirs should exist")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();
        let historical_dir_string = historical_dir.display().to_string();
        let new_dir_string = new_dir.display().to_string();
        assert!(source_dirs.contains(&historical_dir_string.as_str()));
        assert!(source_dirs.contains(&new_dir_string.as_str()));
    }

    fn llama_cpp_replay_manifest(server_url: &str) -> BenchmarkManifest {
        serde_json::from_value(json!({
            "schema_version": "ax.engine_bench.manifest.v1",
            "id": "llama_cpp_replay_dual_cancel",
            "class": "replay",
            "scenario": "replay",
            "model": {
                "family": "qwen3_dense",
                "revision": "phase1-llama-cpp",
                "quant": "q4_k_m",
                "tokenizer_revision": "qwen-tokenizer-v1",
                "chat_template_revision": "chatml-v1"
            },
            "runtime": {
                "selected_backend": "llama_cpp",
                "support_tier": "llama_cpp",
                "resolution_policy": "allow_llama_cpp",
                "fallback_reason": "benchmark requested delegated runtime",
                "backend_adapter": {
                    "kind": "llama_cpp_server_completion",
                    "server_url": server_url
                },
                "deterministic": true,
                "max_batch_tokens": 2048,
                "flags": {
                    "prefix_cache": false
                }
            },
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 1234
            },
            "events": [
                {
                    "t_ms": 0,
                    "type": "submit",
                    "request_id": "req-1",
                    "prompt_ref": "prompts/replay_req_a.txt",
                    "output_tokens_target": 2
                },
                {
                    "t_ms": 0,
                    "type": "submit",
                    "request_id": "req-2",
                    "prompt_ref": "prompts/replay_req_b.txt",
                    "output_tokens_target": 2
                },
                {
                    "t_ms": 1,
                    "type": "cancel",
                    "request_id": "req-2"
                }
            ],
            "checks": {
                "expect_deterministic": true,
                "require_prefix_reuse": false
            }
        }))
        .expect("llama.cpp replay manifest should parse")
    }

    fn llama_cpp_server_replay_manifest(server_url: &str) -> BenchmarkManifest {
        let mut manifest = llama_cpp_replay_manifest(server_url);
        manifest.id = "llama_cpp_replay_dual_llama_cpp".to_string();
        manifest.runtime.selected_backend = SelectedBackend::LlamaCpp;
        manifest.runtime.backend_adapter = Some(BackendAdapterManifest::LlamaCppServerCompletion {
            server_url: server_url.to_string(),
        });
        manifest
    }

    fn llama_cpp_cli_manifest() -> BenchmarkManifest {
        serde_json::from_value(json!({
            "schema_version": "ax.engine_bench.manifest.v1",
            "id": "llama_cpp_cli_forbidden",
            "class": "scenario",
            "scenario": "chat",
            "model": {
                "family": "qwen3_dense",
                "revision": "phase1-llama-cpp",
                "quant": "q4_k_m",
                "tokenizer_revision": "qwen-tokenizer-v1",
                "chat_template_revision": "chatml-v1"
            },
            "runtime": {
                "selected_backend": "llama_cpp",
                "support_tier": "llama_cpp",
                "resolution_policy": "allow_llama_cpp",
                "backend_adapter": {
                    "kind": "llama_cpp_cli",
                    "cli_path": "llama-cli",
                    "model_path": "/tmp/model.gguf"
                },
                "deterministic": true,
                "max_batch_tokens": 2048,
                "flags": {
                    "prefix_cache": false
                }
            },
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 1234
            },
            "shape": {
                "input_tokens_target": 32,
                "output_tokens_target": 4,
                "concurrency": 1
            },
            "checks": {
                "expect_deterministic": true,
                "require_prefix_reuse": false
            }
        }))
        .expect("llama.cpp CLI manifest should parse")
    }

    fn spawn_scripted_llama_cpp_completion_stream_server(
        expected_requests: usize,
        response_for_request: impl Fn(usize, &Value) -> Vec<Value> + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            for request_index in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("request should arrive");
                let request = read_http_request(&mut stream);
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4)
                    .expect("request should include header terminator");
                let body =
                    String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
                let payload: Value =
                    serde_json::from_str(&body).expect("request body should be json");
                let chunks = response_for_request(request_index, &payload);

                let mut body = String::new();
                for chunk in &chunks {
                    body.push_str("data: ");
                    body.push_str(
                        &serde_json::to_string(chunk).expect("chunk payload should serialize"),
                    );
                    body.push_str("\n\n");
                }
                body.push_str("data: [DONE]\n\n");

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("response should write");
            }
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_single_json_completion_server(
        expected_path: &'static str,
        response_body: String,
        assert_request: impl Fn(&Value) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("request should arrive");
            let request = read_http_request(&mut stream);
            let request_text = String::from_utf8(request.clone()).expect("request should be utf8");
            assert!(
                request_text.starts_with(&format!("POST {expected_path} HTTP/1.1")),
                "unexpected request path: {request_text}"
            );

            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
                .expect("request should include header terminator");
            let body =
                String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
            let payload: Value = serde_json::from_str(&body).expect("request body should be json");
            assert_request(&payload);

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream
                .write_all(response.as_bytes())
                .expect("response should write");
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_llama_cpp_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(&Value) + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        spawn_scripted_llama_cpp_completion_stream_server(expected_requests, move |_, payload| {
            assert_request(payload);
            chunks.clone()
        })
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let mut header_end = None;
        let mut content_length = None;

        loop {
            let bytes_read = stream.read(&mut buffer).expect("request should read");
            assert!(
                bytes_read > 0,
                "client closed connection before request completed"
            );
            request.extend_from_slice(&buffer[..bytes_read]);

            if header_end.is_none() {
                header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4);
                if let Some(end) = header_end {
                    let headers =
                        String::from_utf8(request[..end].to_vec()).expect("headers should be utf8");
                    content_length = Some(parse_content_length(&headers));
                }
            }

            if let (Some(end), Some(length)) = (header_end, content_length) {
                if request.len() >= end + length {
                    request.truncate(end + length);
                    return request;
                }
            }
        }
    }

    fn parse_content_length(headers: &str) -> usize {
        headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    Some(
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("content-length should parse"),
                    )
                } else {
                    None
                }
            })
            .expect("content-length header should exist")
    }

    #[test]
    fn route_aggregation_marks_mixed_step_labels_when_runner_routes_differ() {
        let mut observation = RuntimeObservation::default();

        observation.merge_route_metadata(&make_route_metadata(
            "phase1.qwen3_dense.dense_prefill",
            "qwen3_dense_prefill",
            "paged_metadata",
            "serial",
        ));
        observation.merge_route_metadata(&make_route_metadata(
            "phase1.qwen3_dense.paged_decode",
            "qwen3_dense_paged_decode",
            "paged_metadata",
            "serial",
        ));

        assert_eq!(
            observation.route_metadata.execution_plan.as_deref(),
            Some("mixed_step_plans")
        );
        assert_eq!(
            observation.route_metadata.attention_route.as_deref(),
            Some("mixed_attention_routes")
        );
        assert_eq!(
            observation.route_metadata.kv_mode.as_deref(),
            Some("paged_metadata")
        );
        assert_eq!(
            observation.route_metadata.barrier_mode.as_deref(),
            Some("serial")
        );
        assert_eq!(route_decision(&observation, "execution_plan_variants"), 2);
        assert_eq!(route_decision(&observation, "attention_route_variants"), 2);
        assert_eq!(route_decision(&observation, "kv_mode_variants"), 1);
        assert_eq!(route_decision(&observation, "barrier_mode_variants"), 1);
    }

    #[test]
    fn route_aggregation_saturates_crossover_decisions_instead_of_overflowing() {
        let mut observation = RuntimeObservation::default();
        let mut first = make_route_metadata(
            "phase1.qwen3_dense.dense_prefill",
            "qwen3_dense_prefill",
            "paged_metadata",
            "serial",
        );
        first
            .crossover_decisions
            .push(("live_share_hits".to_string(), u32::MAX));
        observation.merge_route_metadata(&first);

        let mut second = make_route_metadata(
            "phase1.qwen3_dense.paged_decode",
            "qwen3_dense_paged_decode",
            "paged_metadata",
            "serial",
        );
        second
            .crossover_decisions
            .push(("live_share_hits".to_string(), 1));
        observation.merge_route_metadata(&second);

        assert_eq!(route_decision(&observation, "live_share_hits"), u32::MAX);
    }

    #[test]
    fn deterministic_digest_ignores_route_telemetry_counters() {
        let first = json!({
            "step_count": 2,
            "route": {
                "crossover_decisions": {
                    "ax_mlx_decode_steps": 4,
                    "ax_mlx_decode_wall_us": 100,
                    "ax_mlx_prefill_wall_us": 50
                }
            },
            "requests": [{"request_id": 1, "generated_tokens": [1, 2, 3]}]
        });
        let second = json!({
            "step_count": 2,
            "route": {
                "crossover_decisions": {
                    "ax_mlx_decode_steps": 4,
                    "ax_mlx_decode_wall_us": 999,
                    "ax_mlx_prefill_wall_us": 777
                }
            },
            "requests": [{"request_id": 1, "generated_tokens": [1, 2, 3]}]
        });

        assert_eq!(
            deterministic_runtime_digest(&first),
            deterministic_runtime_digest(&second)
        );
    }

    #[test]
    fn deterministic_digest_keeps_generated_token_differences() {
        let first = json!({
            "route": {"crossover_decisions": {"ax_mlx_decode_wall_us": 100}},
            "requests": [{"request_id": 1, "generated_tokens": [1, 2, 3]}]
        });
        let second = json!({
            "route": {"crossover_decisions": {"ax_mlx_decode_wall_us": 999}},
            "requests": [{"request_id": 1, "generated_tokens": [1, 2, 4]}]
        });

        assert_ne!(
            deterministic_runtime_digest(&first),
            deterministic_runtime_digest(&second)
        );
    }

    #[test]
    fn decode_tok_s_returns_zero_without_runner_timing() {
        let observation = RuntimeObservation {
            decode_tokens: 4,
            decode_steps: 2,
            total_decode_runner_time_us: 0,
            ..RuntimeObservation::default()
        };

        assert_eq!(observation.decode_tok_s(), 0.0);
    }

    #[test]
    fn decode_tok_s_uses_runner_timing_when_available() {
        let observation = RuntimeObservation {
            decode_tokens: 4,
            decode_steps: 2,
            total_decode_runner_time_us: 2_000,
            ..RuntimeObservation::default()
        };

        assert_eq!(observation.decode_tok_s(), 2_000.0);
    }

    #[test]
    fn llama_cpp_scenario_executes_through_server_completion_adapter() {
        let placeholder_manifest = llama_cpp_scenario_manifest("http://127.0.0.1:1");
        let expected_prompt = scenario_specs_from_manifest(&placeholder_manifest)
            .expect("scenario specs should build")[0]
            .input_tokens
            .clone();
        let expected_prompt_for_request = expected_prompt.clone();
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "content": "llama benchmark",
                    "tokens": [91, 92],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " output",
                    "tokens": [93, 94],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            move |payload| {
                assert_eq!(
                    payload.get("prompt"),
                    Some(&json!(expected_prompt_for_request))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("n_predict"), Some(&Value::Number(4.into())));
            },
        );
        let manifest = llama_cpp_scenario_manifest(&server_url);

        let result =
            execute_manifest_runtime(&manifest).expect("llama.cpp scenario should execute");
        server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
        assert!(
            result.correctness.passed,
            "correctness reason: {:?}",
            result.correctness.reason
        );
        assert!(result.determinism.passed);
        assert_eq!(
            result.observation.final_requests[0].generated_tokens,
            vec![91, 92, 93, 94]
        );
        assert_eq!(
            result.observation.route_metadata.execution_plan.as_deref(),
            Some("llama_cpp.server_completion_stream")
        );
        assert_eq!(result.observation.replay_status(), "not_applicable");
        assert!(result.observation.e2e_latency_ms > 0);
        assert_eq!(result.observation.step_count, 2);
        assert_eq!(result.observation.decode_tokens, 4);
        assert_eq!(
            result.observation.prefill_tokens,
            expected_prompt.len() as u64
        );
        assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));
        assert_eq!(result.observation.step_trace.len(), 2);
        assert_eq!(result.observation.step_trace[0].scheduled_tokens, 2);

        let environment = build_environment_json(
            "llama-run",
            "scenario",
            &{
                let manifest_path =
                    std::env::temp_dir().join("ax-engine-bench-llama-cpp-scenario-manifest.json");
                write_json_file(&manifest_path, &manifest)
                    .expect("llama.cpp manifest should write");
                manifest_path
            },
            Path::new("/tmp/ax-engine-bench-llama-cpp"),
            123,
            &result,
        )
        .expect("environment artifact should build");
        assert_eq!(
            environment
                .get("software")
                .and_then(|software| software.get("tool_mode"))
                .and_then(Value::as_str),
            Some("llama_cpp_stepwise_runtime")
        );
        assert_eq!(
            environment
                .get("runtime")
                .and_then(|runtime| runtime.get("backend_adapter"))
                .and_then(|adapter| adapter.get("kind"))
                .and_then(Value::as_str),
            Some("llama_cpp_server_completion")
        );
    }

    #[test]
    fn llama_cpp_server_scenario_executes_through_blocking_runtime() {
        let (server_url, server_handle) =
            spawn_scripted_llama_cpp_completion_stream_server(1, |_request_index, payload| {
                assert_eq!(
                    payload
                        .get("prompt")
                        .and_then(Value::as_array)
                        .map(Vec::len),
                    Some(32)
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                vec![json!({
                    "content": "llama benchmark output",
                    "tokens": [11, 12, 13, 14],
                    "stop": true,
                    "stop_type": "limit"
                })]
            });
        let manifest = llama_cpp_server_scenario_manifest(&server_url);

        let result = execute_manifest_runtime(&manifest)
            .expect("streaming llama.cpp scenario should execute");
        server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
        assert!(
            result.correctness.passed,
            "correctness reason: {:?}, route decisions: {:?}",
            result.correctness.reason, result.observation.route_metadata.crossover_decisions
        );
        assert_eq!(result.observation.step_count, 1);
        assert_eq!(result.observation.prefill_tokens, 32);
        assert_eq!(result.observation.decode_tokens, 4);
        assert_eq!(result.observation.step_trace.len(), 1);
        assert_eq!(
            result.observation.route_metadata.execution_plan.as_deref(),
            Some("llama_cpp.server_completion_stream")
        );
    }

    #[test]
    fn generate_command_executes_server_backed_openai_compatible_prompt() {
        let (server_url, server_handle) = spawn_single_json_completion_server(
            "/completion",
            json!({
                "content": "llama::hello cli",
                "tokens": [4, 5, 6],
                "tokens_evaluated": 2,
                "tokens_cached": 0,
                "stop": true,
                "stop_type": "limit"
            })
            .to_string(),
            |payload| {
                assert_eq!(payload["prompt"], Value::String("hello cli".to_string()));
                assert_eq!(payload["stream"], Value::Bool(false));
            },
        );
        let args = parse_inference_args(
            &[
                "--prompt".to_string(),
                "hello cli".to_string(),
                "--support-tier".to_string(),
                "llama_cpp".to_string(),
                "--llama-server-url".to_string(),
                server_url.clone(),
            ],
            "generate",
        )
        .expect("generate args should parse");

        let response =
            run_inference_generate(&args).expect("llama.cpp generate command should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(response.output_text.as_deref(), Some("llama::hello cli"));
        assert_eq!(response.runtime.selected_backend, SelectedBackend::LlamaCpp);
        let rendered = render_generate_response(&response, false).expect("response should render");
        assert!(rendered.starts_with("llama::hello cli\n"));
        assert!(rendered.contains("status=finished"));
        assert!(rendered.contains("finish_reason=max_output_tokens"));
        assert!(rendered.contains("execution_plan=llama_cpp.server_completion"));
    }

    #[test]
    fn stream_command_collects_server_backed_llama_cpp_events() {
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            1,
            vec![
                json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );
        let args = parse_inference_args(
            &[
                "--tokens".to_string(),
                "1,2,3".to_string(),
                "--support-tier".to_string(),
                "llama_cpp".to_string(),
                "--llama-server-url".to_string(),
                server_url.clone(),
            ],
            "stream",
        )
        .expect("stream args should parse");

        let events = collect_inference_stream_events(&args)
            .expect("llama.cpp stream command should succeed");
        server_handle.join().expect("server thread should finish");

        assert_eq!(events.len(), 4);
        assert_eq!(events[0].event_name(), "request");
        assert_eq!(events[1].event_name(), "step");
        assert_eq!(events[2].event_name(), "step");
        assert_eq!(events[3].event_name(), "response");

        let step_rendered = render_stream_event(&events[2], false).expect("event should render");
        assert!(step_rendered.contains("delta_token_logprobs=[None]"));
        assert!(step_rendered.contains("delta_text=\" world\""));
        assert!(step_rendered.contains("finish_reason=max_output_tokens"));
        let rendered = render_stream_event(&events[3], false).expect("event should render");
        assert!(rendered.contains("finish_reason=max_output_tokens"));
        assert!(rendered.contains("output_token_logprobs=[None, None]"));
        assert!(rendered.contains("output_text=\"hello world\""));
    }

    #[test]
    fn llama_cpp_server_replay_is_rejected_for_blocking_adapter() {
        let manifest = llama_cpp_server_replay_manifest("http://127.0.0.1:8081");

        let error = execute_manifest_runtime(&manifest)
            .expect_err("llama.cpp replay without a server should fail closed");
        assert!(format!("{error}").contains("llama.cpp"));
    }

    #[test]
    fn llama_cpp_server_shared_prefix_is_rejected_for_blocking_adapter() {
        let mut manifest = llama_cpp_server_scenario_manifest("http://127.0.0.1:8081");
        manifest.scenario = "shared_prefix".to_string();
        manifest.checks.require_prefix_reuse = true;

        let error = execute_manifest_runtime(&manifest)
            .expect_err("llama.cpp shared-prefix without a server should fail closed");
        assert!(format!("{error}").contains("llama.cpp"));
    }

    #[test]
    fn llama_cpp_replay_executes_through_shared_sdk_session() {
        let placeholder_manifest = llama_cpp_replay_manifest("http://127.0.0.1:1");
        let replay_events =
            replay_events_from_manifest(&placeholder_manifest).expect("replay events should build");
        let expected_specs = replay_events
            .iter()
            .filter_map(|event| match event {
                ReplayEvent::Submit { spec, .. } => Some(spec.clone()),
                ReplayEvent::Cancel { .. } => None,
            })
            .collect::<Vec<_>>();
        let expected_prompt_total = expected_specs
            .iter()
            .map(|spec| spec.input_tokens.len() as u64)
            .sum::<u64>();
        let expected_prompts = expected_specs
            .iter()
            .map(|spec| spec.input_tokens.clone())
            .collect::<Vec<_>>();
        let expected_prompts_for_request = expected_prompts.clone();
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            expected_prompts.len() * 2,
            vec![
                serde_json::json!({
                    "content": "llama replay",
                    "tokens": [111],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " done",
                    "tokens": [112],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            move |payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    expected_prompts_for_request
                        .iter()
                        .any(|candidate| prompt == &json!(candidate))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("n_predict"), Some(&Value::Number(2.into())));
            },
        );
        let manifest = llama_cpp_replay_manifest(&server_url);

        let result = execute_manifest_runtime(&manifest).expect("llama.cpp replay should execute");
        server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
        assert!(
            result.correctness.passed,
            "correctness reason: {:?}",
            result.correctness.reason
        );
        assert!(result.determinism.passed);
        assert_eq!(result.observation.final_requests.len(), 2);
        assert_eq!(
            result
                .observation
                .final_requests
                .iter()
                .map(|request| request.state.as_str())
                .collect::<Vec<_>>(),
            vec!["Finished", "Cancelled"]
        );
        assert_eq!(
            result
                .observation
                .final_requests
                .iter()
                .map(|request| request.generated_tokens.clone())
                .collect::<Vec<_>>(),
            vec![vec![111, 112], vec![111]]
        );
        assert_eq!(
            result
                .observation
                .final_requests
                .iter()
                .map(|request| request.cancel_requested)
                .collect::<Vec<_>>(),
            vec![false, true]
        );
        assert_eq!(
            result.observation.route_metadata.execution_plan.as_deref(),
            Some("llama_cpp.server_completion_stream")
        );
        assert_eq!(result.observation.replay_status(), "pass");
        assert_eq!(result.observation.step_count, 2);
        assert_eq!(result.observation.step_trace.len(), 2);
        assert_eq!(result.observation.decode_tokens, 3);
        assert_eq!(result.observation.prefill_tokens, expected_prompt_total);
        assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));
    }

    #[test]
    fn llama_cpp_concurrent_scenario_executes_through_shared_sdk_session() {
        let mut manifest = llama_cpp_scenario_manifest("http://127.0.0.1:8081");
        manifest
            .shape
            .as_mut()
            .expect("shape should exist")
            .concurrency = 2;
        manifest.scenario = "concurrent".to_string();
        let expected_specs =
            scenario_specs_from_manifest(&manifest).expect("scenario specs should build");
        let expected_prompt_total = expected_specs
            .iter()
            .map(|spec| spec.input_tokens.len() as u64)
            .sum::<u64>();
        let expected_prompts = expected_specs
            .iter()
            .map(|spec| spec.input_tokens.clone())
            .collect::<Vec<_>>();
        let expected_prompts_for_request = expected_prompts.clone();
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
            expected_prompts.len() * 2,
            vec![
                serde_json::json!({
                    "content": "llama concurrent",
                    "tokens": [101, 102],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " output",
                    "tokens": [103, 104],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            move |payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    expected_prompts_for_request
                        .iter()
                        .any(|candidate| prompt == &json!(candidate))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
            },
        );
        manifest.runtime.backend_adapter =
            Some(BackendAdapterManifest::LlamaCppServerCompletion { server_url });

        let result = execute_manifest_runtime(&manifest)
            .expect("llama.cpp concurrent benchmark should execute");
        server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
        assert_eq!(result.observation.final_requests.len(), 2);
        assert_eq!(result.observation.step_count, 2);
        assert_eq!(result.observation.step_trace.len(), 2);
        assert_eq!(result.observation.decode_tokens, 8);
        assert_eq!(result.observation.prefill_tokens, expected_prompt_total);
        assert_eq!(
            result
                .observation
                .final_requests
                .iter()
                .map(|request| request.generated_tokens.clone())
                .collect::<Vec<_>>(),
            vec![vec![101, 102, 103, 104], vec![101, 102, 103, 104]]
        );
        assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));
    }

    #[test]
    fn llama_cpp_shared_prefix_scenario_executes_with_delegated_prompt_cache_reuse() {
        let mut manifest = llama_cpp_shared_prefix_scenario_manifest("http://127.0.0.1:8081");
        let expected_specs =
            scenario_specs_from_manifest(&manifest).expect("scenario specs should build");
        assert_eq!(expected_specs.len(), 2);
        assert_eq!(
            expected_specs[0].input_tokens[..64],
            expected_specs[1].input_tokens[..64]
        );
        assert_ne!(
            expected_specs[0].input_tokens[64..],
            expected_specs[1].input_tokens[64..]
        );

        let expected_prompt_total = expected_specs
            .iter()
            .map(|spec| spec.input_tokens.len() as u64)
            .sum::<u64>();
        let expected_prompts = expected_specs
            .iter()
            .map(|spec| spec.input_tokens.clone())
            .collect::<Vec<_>>();
        let expected_prompts_for_request = expected_prompts.clone();
        let (server_url, server_handle) = spawn_scripted_llama_cpp_completion_stream_server(
            expected_prompts.len() * 2,
            move |request_index, payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    expected_prompts_for_request
                        .iter()
                        .any(|candidate| prompt == &json!(candidate))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

                let cache_tokens = if request_index % 2 == 0 { 0 } else { 64 };
                vec![
                    serde_json::json!({
                        "content": "",
                        "tokens": [],
                        "stop": false,
                        "prompt_progress": {
                            "total": 128,
                            "cache": cache_tokens,
                            "processed": 128_u32.saturating_sub(cache_tokens),
                            "time_ms": 1.0
                        }
                    }),
                    serde_json::json!({
                        "content": "shared-prefix",
                        "tokens": [141, 142, 143, 144],
                        "stop": true,
                        "stop_type": "limit"
                    }),
                ]
            },
        );
        manifest.runtime.backend_adapter =
            Some(BackendAdapterManifest::LlamaCppServerCompletion { server_url });

        let result = execute_manifest_runtime(&manifest)
            .expect("llama.cpp shared-prefix scenario should execute");
        server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert_eq!(result.tool_mode, "llama_cpp_stepwise_runtime");
        assert!(result.correctness.passed);
        assert!(result.determinism.passed);
        assert_eq!(result.observation.final_requests.len(), 2);
        assert_eq!(result.observation.step_count, 2);
        assert_eq!(result.observation.step_trace.len(), 2);
        assert_eq!(result.observation.prefill_tokens, expected_prompt_total);
        assert_eq!(result.observation.decode_tokens, 8);
        assert!(result.observation.prefix_hit_rate() > 0.0);
        assert_eq!(
            result
                .observation
                .route_metadata
                .prefix_cache_path
                .as_deref(),
            Some("delegated_prompt_cache")
        );
        assert_eq!(
            route_decision(&result.observation, "delegated_cached_tokens"),
            64
        );
        assert_eq!(result.observation.digest.get("ttft_ms"), Some(&Value::Null));

        let expected_prompts_for_artifacts = expected_prompts.clone();
        let (artifact_server_url, artifact_server_handle) =
            spawn_scripted_llama_cpp_completion_stream_server(
                expected_prompts.len() * 2,
                move |request_index, payload| {
                    let prompt = payload.get("prompt").expect("prompt should be present");
                    assert!(
                        expected_prompts_for_artifacts
                            .iter()
                            .any(|candidate| prompt == &json!(candidate))
                    );
                    assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                    assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                    assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

                    let cache_tokens = if request_index % 2 == 0 { 0 } else { 64 };
                    vec![
                        serde_json::json!({
                            "content": "",
                            "tokens": [],
                            "stop": false,
                            "prompt_progress": {
                                "total": 128,
                                "cache": cache_tokens,
                                "processed": 128_u32.saturating_sub(cache_tokens),
                                "time_ms": 1.0
                            }
                        }),
                        serde_json::json!({
                            "content": "shared-prefix",
                            "tokens": [141, 142, 143, 144],
                            "stop": true,
                            "stop_type": "limit"
                        }),
                    ]
                },
            );
        let root = unique_test_dir("llama-cpp-shared-prefix-success");
        let output_root = root.join("results");
        fs::create_dir_all(&output_root).expect("output root should create");
        manifest.runtime.backend_adapter = Some(BackendAdapterManifest::LlamaCppServerCompletion {
            server_url: artifact_server_url,
        });
        let manifest_path = root.join("llama-cpp-shared-prefix.json");
        write_json_file(&manifest_path, &manifest).expect("manifest should write");
        let args = vec![
            "--manifest".to_string(),
            manifest_path.display().to_string(),
            "--output-root".to_string(),
            output_root.display().to_string(),
        ];
        handle_scenario(&args).expect("shared-prefix scenario should write success artifacts");
        artifact_server_handle
            .join()
            .expect("artifact server thread should finish");
        let runs = fs::read_dir(&output_root)
            .expect("output root should be readable")
            .map(|entry| entry.expect("directory entry should read").path())
            .collect::<Vec<_>>();
        assert_eq!(runs.len(), 1);
        let run_dir = &runs[0];
        assert!(run_dir.join("environment.json").is_file());
        assert!(run_dir.join("metrics.json").is_file());
        assert!(run_dir.join("routes.json").is_file());
        assert!(!run_dir.join("contract_failure.json").exists());

        let environment = load_json_value(&run_dir.join("environment.json"))
            .expect("environment artifact should load");
        assert_eq!(
            environment
                .get("route")
                .and_then(|route| route.get("prefix_cache_path"))
                .and_then(Value::as_str),
            Some("delegated_prompt_cache")
        );
        assert_eq!(
            environment
                .get("route")
                .and_then(|route| route.get("prefix_reuse_provenance"))
                .and_then(Value::as_str),
            Some("delegated_backend_prompt_cache")
        );
        assert_eq!(
            environment
                .get("route")
                .and_then(|route| route.get("backend_reported_cached_prompt_tokens"))
                .and_then(Value::as_u64),
            Some(64)
        );

        fs::remove_dir_all(&root).expect("test directory should clean up");
    }

    #[test]
    fn contract_failure_json_classifies_missing_backend_adapter() {
        let mut manifest = llama_cpp_scenario_manifest("http://127.0.0.1:8081");
        manifest.runtime.backend_adapter = None;
        let artifact = build_contract_failure_json(
            "run-1",
            "scenario",
            Path::new("/tmp/manifest.json"),
            &manifest,
            Path::new("/tmp/results"),
            123,
            "runtime.backend_adapter is required when selected_backend is llama_cpp",
        )
        .expect("contract failure artifact should build");

        assert_eq!(
            artifact.get("tool_mode").and_then(Value::as_str),
            Some("llama_cpp_blocking_runtime")
        );
        assert_eq!(
            artifact
                .get("failure")
                .and_then(|failure| failure.get("code"))
                .and_then(Value::as_str),
            Some("llama_backend_adapter_required")
        );
    }

    #[test]
    fn build_session_preserves_llama_runtime_fields_via_sdk_factory() {
        let server_url = "http://127.0.0.1:8081";
        let mut manifest = llama_cpp_scenario_manifest(server_url);
        manifest.runtime.deterministic = false;
        manifest.runtime.max_batch_tokens = 4096;
        manifest.runtime.kv_total_blocks = Some(42);
        let expected_runtime = manifest.runtime.clone();

        let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
        let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");

        let config = session_config_from_runtime(&runtime, &specs);

        assert_eq!(config.kv_config.cache_group_id, CacheGroupId(0));
        assert_eq!(config.kv_config.block_size_tokens, 16);
        assert_eq!(config.kv_config.total_blocks, 42);
        assert!(!config.deterministic);
        assert_eq!(config.max_batch_tokens, 4096);
        assert_eq!(
            config.backend_policy,
            BackendPolicy::new(expected_runtime.resolution_policy)
        );
        assert_eq!(
            config.resolved_backend,
            ResolvedBackend::new(
                expected_runtime.selected_backend,
                expected_runtime.support_tier,
                expected_runtime.fallback_reason.clone(),
            )
        );
        assert_eq!(
            config.llama_backend,
            expected_runtime
                .backend_adapter
                .as_ref()
                .map(BackendAdapterManifest::to_sdk_config)
        );
    }

    #[test]
    fn build_session_preserves_mlx_runtime_artifact_defaults_via_sdk_factory() {
        let manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
        let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");
        let expected_default = EngineSessionConfig::default();

        let config = session_config_from_runtime(&runtime, &specs);

        assert_eq!(
            config.mlx_runtime_artifacts_dir,
            expected_default.mlx_runtime_artifacts_dir
        );
        assert_eq!(
            config.mlx_runtime_artifacts_source,
            expected_default.mlx_runtime_artifacts_source()
        );
        assert_eq!(
            config.mlx_model_artifacts_dir,
            manifest.runtime.mlx_model_artifacts_dir
        );
        assert_eq!(
            config.mlx_model_artifacts_source,
            Some(NativeModelArtifactsSource::ExplicitConfig)
        );
        assert_eq!(config.backend_policy, runtime.backend_policy);
        assert_eq!(config.resolved_backend, runtime.resolved_backend);
        assert_eq!(config.llama_backend, None);
    }

    #[test]
    fn build_session_preserves_explicit_mlx_model_artifact_manifest_override() {
        let mut manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        manifest.runtime.mlx_model_artifacts_dir = Some(PathBuf::from("/tmp/ax-mlx-model"));

        let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
        let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");

        let config = session_config_from_runtime(&runtime, &specs);

        let expected_mlx_runtime_artifacts_dir =
            EngineSessionConfig::default_mlx_runtime_artifacts_dir().or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|dir| dir.join("build/metal"))
                    .filter(|dir| dir.join("build_report.json").is_file())
            });
        assert_eq!(
            config.mlx_runtime_artifacts_dir,
            expected_mlx_runtime_artifacts_dir
        );
        assert_eq!(
            config.mlx_runtime_artifacts_source,
            config
                .mlx_runtime_artifacts_dir
                .as_ref()
                .map(|_| NativeRuntimeArtifactsSource::RepoAutoDetect)
        );
        assert_eq!(
            config.mlx_model_artifacts_dir,
            Some(PathBuf::from("/tmp/ax-mlx-model"))
        );
        assert_eq!(
            config.mlx_model_artifacts_source,
            Some(NativeModelArtifactsSource::ExplicitConfig)
        );
    }

    #[test]
    fn build_session_preserves_gguf_model_override_as_explicit_source() {
        let mut manifest = load_test_manifest(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        manifest.runtime.mlx_model_artifacts_dir =
            Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"));

        let runtime = runtime_config_from_manifest(&manifest).expect("runtime config should load");
        let specs = scenario_specs_from_manifest(&manifest).expect("scenario specs should build");

        let config = session_config_from_runtime(&runtime, &specs);

        assert_eq!(
            config.mlx_model_artifacts_dir,
            Some(PathBuf::from("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"))
        );
        assert_eq!(
            config.mlx_model_artifacts_source,
            Some(NativeModelArtifactsSource::ExplicitConfig)
        );
    }

    #[test]
    fn load_manifest_resolves_native_artifact_paths_relative_to_manifest_dir() {
        let root = unique_test_dir("manifest-relative-mlx-paths");
        let manifest_dir = root.join("benchmarks/manifests/scenario");
        fs::create_dir_all(&manifest_dir).expect("manifest dir should create");
        let manifest_path = manifest_dir.join("relative-native.json");
        fs::write(
            &manifest_path,
            r#"{
  "schema_version": "ax.engine_bench.manifest.v1",
  "id": "relative_native",
  "class": "scenario",
  "scenario": "chat",
  "model": {
    "family": "qwen3_dense",
    "revision": "phase1-canonical",
    "quant": "q4_k_m",
    "tokenizer_revision": "qwen-tokenizer-v1",
    "chat_template_revision": "chatml-v1"
  },
  "runtime": {
    "selected_backend": "mlx",
    "support_tier": "mlx_preview",
    "resolution_policy": "mlx_only",
    "deterministic": true,
    "max_batch_tokens": 2048,
    "mlx_model_artifacts_dir": "../../../models/qwen-mlx",
    "flags": {
      "prefix_cache": false
    }
  },
  "sampling": {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 1234
  },
  "shape": {
    "input_tokens_target": 64,
    "output_tokens_target": 32,
    "concurrency": 1
  }
}"#,
        )
        .expect("manifest should write");

        let manifest = load_manifest(&manifest_path).expect("manifest should load");

        assert_eq!(
            manifest.runtime.mlx_model_artifacts_dir,
            Some(root.join("models/qwen-mlx"))
        );

        fs::remove_dir_all(&root).expect("test dir should clean up");
    }

    #[test]
    fn load_manifest_expands_native_artifact_env_paths() {
        let Some(home_dir) = env::var_os("HOME").map(PathBuf::from) else {
            return;
        };
        let root = unique_test_dir("manifest-env-mlx-paths");
        fs::create_dir_all(&root).expect("test dir should create");
        let manifest_path = root.join("env-native.json");
        fs::write(
            &manifest_path,
            r#"{
  "schema_version": "ax.engine_bench.manifest.v1",
  "id": "env_mlx",
  "class": "scenario",
  "scenario": "chat",
  "model": {
    "family": "qwen3_dense",
    "revision": "phase1-canonical",
    "quant": "q4_k_m",
    "tokenizer_revision": "qwen-tokenizer-v1",
    "chat_template_revision": "chatml-v1"
  },
  "runtime": {
    "selected_backend": "mlx",
    "support_tier": "mlx_preview",
    "resolution_policy": "mlx_only",
    "deterministic": true,
    "max_batch_tokens": 2048,
    "mlx_model_artifacts_dir": "$HOME/ax-mlx-model",
    "flags": {
      "prefix_cache": false
    }
  },
  "sampling": {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 1234
  },
  "shape": {
    "input_tokens_target": 64,
    "output_tokens_target": 32,
    "concurrency": 1
  }
}"#,
        )
        .expect("manifest should write");

        let manifest = load_manifest(&manifest_path).expect("manifest should load");

        assert_eq!(
            manifest.runtime.mlx_model_artifacts_dir,
            Some(home_dir.join("ax-mlx-model"))
        );

        fs::remove_dir_all(&root).expect("test dir should clean up");
    }

    #[test]
    fn llama_cpp_replay_executes_with_delegated_prompt_cache_reuse() {
        let mut manifest = load_test_manifest(
            repo_manifest_path(
                "benchmarks/manifests/replay/llama_cpp_prompt_cache_reuse_dual.json",
            )
            .as_str(),
        );

        let replay_events =
            replay_events_from_manifest(&manifest).expect("replay events should build");
        let expected_prompts = replay_events
            .iter()
            .filter_map(|event| match event {
                ReplayEvent::Submit { spec, .. } => Some(spec.input_tokens.clone()),
                ReplayEvent::Cancel { .. } => None,
            })
            .collect::<Vec<_>>();
        let expected_prompts_for_request = expected_prompts.clone();
        let (server_url, server_handle) =
            spawn_scripted_llama_cpp_completion_stream_server(4, move |request_index, payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    expected_prompts_for_request
                        .iter()
                        .any(|candidate| prompt == &json!(candidate))
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

                if request_index % 2 == 0 {
                    vec![
                        serde_json::json!({
                            "content": "",
                            "tokens": [],
                            "stop": false,
                            "prompt_progress": {
                                "total": 32,
                                "cache": 0,
                                "processed": 32,
                                "time_ms": 1.0
                            }
                        }),
                        serde_json::json!({
                            "content": "seed",
                            "tokens": [121],
                            "stop": true,
                            "stop_type": "limit"
                        }),
                    ]
                } else {
                    vec![
                        serde_json::json!({
                            "content": "",
                            "tokens": [],
                            "stop": false,
                            "prompt_progress": {
                                "total": 32,
                                "cache": 24,
                                "processed": 24,
                                "time_ms": 1.0
                            }
                        }),
                        serde_json::json!({
                            "content": "reuse",
                            "tokens": [122],
                            "stop": true,
                            "stop_type": "limit"
                        }),
                    ]
                }
            });
        manifest.runtime.backend_adapter =
            Some(BackendAdapterManifest::LlamaCppServerCompletion { server_url });

        let result = execute_manifest_runtime(&manifest)
            .expect("llama.cpp prefix-reuse replay should execute");
        server_handle
            .join()
            .expect("llama.cpp server thread should finish");

        assert!(
            result.correctness.passed,
            "correctness reason: {:?}",
            result.correctness.reason
        );
        assert!(result.determinism.passed);
        assert!(result.observation.prefix_hit_rate() > 0.0);
        assert_eq!(
            result
                .observation
                .route_metadata
                .prefix_cache_path
                .as_deref(),
            Some("delegated_prompt_cache")
        );
        assert!(route_decision(&result.observation, "delegated_cached_tokens") > 0);
    }

    #[test]
    fn llama_cpp_benchmark_rejects_cli_adapter_for_token_workloads() {
        let manifest = llama_cpp_cli_manifest();

        let error = execute_manifest_runtime(&manifest)
            .expect_err("llama.cpp CLI benchmark should fail closed");
        let CliError::Contract(message) = error else {
            panic!("llama.cpp CLI benchmark should return a contract error");
        };

        assert!(message.contains("llama.cpp"));
    }

    #[test]
    fn regression_artifact_records_prefix_reuse_contract() {
        let baseline_metrics = json!({
            "run_id": "baseline-run",
            "metrics": {
                "ttft_ms": 10.0,
                "decode_tok_s": 100.0,
                "memory_peak_mb": 512.0,
                "prefix_hit_rate": 50.0
            }
        });
        let candidate_metrics = json!({
            "run_id": "candidate-run",
            "metrics": {
                "ttft_ms": 8.0,
                "decode_tok_s": 110.0,
                "memory_peak_mb": 500.0,
                "prefix_hit_rate": 55.0
            }
        });
        let baseline_environment = json!({
            "software": {
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 96
            }
        });
        let candidate_environment = json!({
            "software": {
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 96
            }
        });

        let regression = build_regression_json(
            &baseline_metrics,
            &candidate_metrics,
            &baseline_environment,
            &candidate_environment,
            None,
        )
        .expect("regression artifact should build");

        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("result"))
                .and_then(Value::as_str),
            Some("llama_cpp_stepwise_compare")
        );
        assert_eq!(
            regression
                .get("runtime")
                .and_then(|runtime| runtime.get("tool_mode"))
                .and_then(Value::as_str),
            Some("llama_cpp_stepwise_runtime")
        );
        assert_eq!(
            regression
                .get("runtime")
                .and_then(|runtime| runtime.get("selected_backend"))
                .and_then(Value::as_str),
            Some("llama_cpp")
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("execution_semantics"))
                .and_then(Value::as_str),
            Some("delegated_backend_runtime")
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("prefix_cache_path"))
                .and_then(Value::as_str),
            Some("delegated_prompt_cache")
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("prefix_cache_evidence"))
                .and_then(Value::as_str),
            Some("backend_reported_cached_prompt_tokens")
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("prefix_reuse_provenance"))
                .and_then(Value::as_str),
            Some("delegated_backend_prompt_cache")
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("backend_reported_cached_prompt_tokens"))
                .and_then(Value::as_u64),
            Some(96)
        );
        assert_eq!(
            regression
                .get("contract")
                .and_then(|contract| contract.get("tool_mode"))
                .and_then(|field| field.get("baseline"))
                .and_then(Value::as_str),
            Some("llama_cpp_stepwise_runtime")
        );
        assert_eq!(
            regression
                .get("contract")
                .and_then(|contract| contract.get("runtime"))
                .and_then(|runtime| runtime.get("selected_backend"))
                .and_then(|field| field.get("baseline"))
                .and_then(Value::as_str),
            Some("llama_cpp")
        );
        assert_eq!(
            regression
                .get("contract")
                .and_then(|contract| contract.get("prefix_cache_evidence"))
                .and_then(|field| field.get("baseline"))
                .and_then(Value::as_str),
            Some("backend_reported_cached_prompt_tokens")
        );
        assert_eq!(
            regression
                .get("contract")
                .and_then(|contract| contract.get("prefix_reuse_provenance"))
                .and_then(|field| field.get("baseline"))
                .and_then(Value::as_str),
            Some("delegated_backend_prompt_cache")
        );
        assert_eq!(
            regression
                .get("contract")
                .and_then(|contract| contract.get("backend_reported_cached_prompt_tokens"))
                .and_then(|field| field.get("baseline"))
                .and_then(Value::as_u64),
            Some(96)
        );
        assert_eq!(
            regression
                .get("readiness")
                .and_then(|readiness| readiness.get("baseline"))
                .and_then(|entry| entry.get("status"))
                .and_then(Value::as_str),
            Some("delegated_runtime")
        );
    }

    #[test]
    fn compare_summary_uses_regression_prefix_reuse_provenance() {
        let baseline_metrics = json!({
            "run_id": "baseline-run"
        });
        let candidate_metrics = json!({
            "run_id": "candidate-run"
        });
        let regression = json!({
            "runtime": {
                "tool_mode": "llama_cpp_stepwise_runtime",
                "backend_adapter": {
                    "kind": "llama_cpp_server_completion",
                    "server_url": "http://127.0.0.1:8081"
                }
            },
            "comparison": {
                "ttft_ms_pct": -20.0,
                "decode_tok_s_pct": 10.0,
                "memory_peak_mb_pct": -2.0,
                "prefix_hit_rate_pct": 5.0
            },
            "summary": {
                "result": "llama_cpp_stepwise_compare",
                "tool_mode": "llama_cpp_stepwise_runtime",
                "selected_backend": "llama_cpp",
                "support_tier": "llama_cpp",
                "resolution_policy": "allow_llama_cpp",
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 144
            },
            "readiness": {
                "baseline": {
                    "status": "delegated_runtime",
                    "hot_path_cpu_fallback_free": false,
                    "batched_direct_decode_logits_ready": false,
                    "prefix_min_mlx_metal_dispatch_share": null,
                    "direct_decode_min_mlx_metal_dispatch_share": null,
                    "blockers": ["delegated_runtime_not_mlx"]
                },
                "candidate": {
                    "status": "delegated_runtime",
                    "hot_path_cpu_fallback_free": false,
                    "batched_direct_decode_logits_ready": false,
                    "prefix_min_mlx_metal_dispatch_share": null,
                    "direct_decode_min_mlx_metal_dispatch_share": null,
                    "blockers": ["delegated_runtime_not_mlx"]
                }
            }
        });

        let summary = build_compare_summary_markdown(
            &baseline_metrics,
            &candidate_metrics,
            &regression,
            None,
        );

        assert!(summary.contains("delegated_backend_prompt_cache"));
        assert!(summary.contains("llama_cpp_stepwise_compare"));
        assert!(summary.contains("llama_cpp_stepwise_runtime"));
        assert!(summary.contains("selected_backend: `llama_cpp`"));
        assert!(summary.contains("backend_adapter: `{\"kind\":\"llama_cpp_server_completion\",\"server_url\":\"http://127.0.0.1:8081\"}`"));
        assert!(summary.contains("execution_semantics: `unknown`"));
        assert!(summary.contains("prefix_cache_evidence: `backend_reported_cached_prompt_tokens`"));
        assert!(summary.contains("backend_reported_cached_prompt_tokens: `144`"));
        assert!(summary.contains("baseline_mlx_metal_readiness: `delegated_runtime`"));
        assert!(
            summary.contains(
                "candidate_mlx_metal_readiness_blockers: `[\"delegated_runtime_not_mlx\"]`"
            )
        );
        assert!(!summary.contains("prefix_reuse_provenance: `unknown`"));
        assert!(!summary.contains("mode: `engine_bringup`"));
    }

    #[test]
    fn regression_artifact_records_mlx_metal_readiness() {
        let baseline_metrics = json!({
            "run_id": "baseline-run",
            "metrics": {
                "ttft_ms": 10.0,
                "decode_tok_s": 100.0,
                "memory_peak_mb": 512.0,
                "prefix_hit_rate": 50.0
            }
        });
        let candidate_metrics = json!({
            "run_id": "candidate-run",
            "metrics": {
                "ttft_ms": 9.0,
                "decode_tok_s": 110.0,
                "memory_peak_mb": 500.0,
                "prefix_hit_rate": 55.0
            }
        });
        let baseline_environment = json!({
            "software": {
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_forward",
                "metal_direct_decode_batching_opportunity_observed": true,
                "metal_complete_model_forward_supported": true,
                "metal_real_model_forward": true,
                "metal_model_artifacts_validated": true,
                "mlx_metal_prefix_layers_attention": true,
                "metal_prefix_layers_cpu_reference": false,
                "metal_prefix_cpu_reference_dispatch_count": 0,
                "mlx_metal_prefix_projection_row_count": 2048,
                "metal_prefix_cpu_projection_row_count": 0,
                "mlx_metal_prefix_rms_norm_element_count": 1024,
                "metal_prefix_cpu_rms_norm_element_count": 0,
                "mlx_metal_prefix_ffn_activation_element_count": 1024,
                "metal_prefix_cpu_ffn_activation_element_count": 0,
                "metal_direct_decode_tokens": true,
                "metal_direct_decode_model_bound_ffn": true,
                "metal_direct_decode_batched_logits_group_count": 2,
                "metal_direct_decode_batched_logits_token_count": 2,
                "metal_direct_decode_batched_group_fallback_count": 0,
                "metal_direct_decode_batched_group_fallback_token_count": 0,
                "mlx_metal_direct_decode_projection_row_count": 4096,
                "metal_direct_decode_cpu_projection_row_count": 0,
                "mlx_metal_direct_decode_rms_norm_element_count": 1024,
                "metal_direct_decode_cpu_rms_norm_element_count": 0,
                "mlx_metal_direct_decode_ffn_activation_element_count": 1024,
                "metal_direct_decode_cpu_ffn_activation_element_count": 0,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed",
                "ax_mlx_kv_capacity_kib": 32,
                "ax_mlx_kv_sliding_reclaimable_capacity_kib": 8,
                "ax_mlx_kv_growth_count": 1
            }
        });
        let candidate_environment = json!({
            "software": {
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_forward",
                "metal_direct_decode_batching_opportunity_observed": true,
                "metal_complete_model_forward_supported": true,
                "metal_real_model_forward": true,
                "metal_model_artifacts_validated": true,
                "mlx_metal_prefix_layers_attention": true,
                "metal_prefix_layers_cpu_reference": false,
                "metal_prefix_cpu_reference_dispatch_count": 0,
                "mlx_metal_prefix_projection_row_count": 2048,
                "metal_prefix_cpu_projection_row_count": 0,
                "mlx_metal_prefix_rms_norm_element_count": 1024,
                "metal_prefix_cpu_rms_norm_element_count": 0,
                "mlx_metal_prefix_ffn_activation_element_count": 1024,
                "metal_prefix_cpu_ffn_activation_element_count": 0,
                "metal_direct_decode_tokens": true,
                "metal_direct_decode_model_bound_ffn": true,
                "metal_direct_decode_batched_logits_group_count": 1,
                "metal_direct_decode_batched_logits_token_count": 2,
                "metal_direct_decode_batched_group_fallback_count": 1,
                "metal_direct_decode_batched_group_fallback_token_count": 2,
                "mlx_metal_direct_decode_projection_row_count": 3072,
                "metal_direct_decode_cpu_projection_row_count": 0,
                "mlx_metal_direct_decode_rms_norm_element_count": 1024,
                "metal_direct_decode_cpu_rms_norm_element_count": 0,
                "mlx_metal_direct_decode_ffn_activation_element_count": 1024,
                "metal_direct_decode_cpu_ffn_activation_element_count": 0,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed",
                "ax_mlx_kv_capacity_kib": 48,
                "ax_mlx_kv_sliding_reclaimable_capacity_kib": 12,
                "ax_mlx_kv_growth_count": 2
            }
        });

        let regression = build_regression_json(
            &baseline_metrics,
            &candidate_metrics,
            &baseline_environment,
            &candidate_environment,
            None,
        )
        .expect("regression artifact should build");

        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("metal_direct_decode_batching_opportunity_observed"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("mlx_metal_readiness"))
                .and_then(Value::as_str),
            Some("ready")
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("mlx_metal_hot_path_cpu_fallback_free"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| summary.get("mlx_metal_batched_direct_decode_logits_ready"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            regression
                .get("readiness")
                .and_then(|readiness| readiness.get("candidate"))
                .and_then(|entry| entry.get("status"))
                .and_then(Value::as_str),
            Some("coverage_gap")
        );
        assert_eq!(
            regression
                .get("readiness")
                .and_then(|readiness| readiness.get("candidate"))
                .and_then(|entry| entry.get("hot_path_cpu_fallback_free"))
                .and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            regression
                .get("readiness")
                .and_then(|readiness| readiness.get("candidate"))
                .and_then(|entry| entry.get("blockers"))
                .and_then(Value::as_array)
                .map(|entries| { entries.iter().filter_map(Value::as_str).collect::<Vec<_>>() }),
            Some(vec![
                "batched_direct_decode_logits_not_observed",
                "batched_direct_decode_group_fallback_remaining"
            ])
        );
        assert_eq!(
            regression
                .get("summary")
                .and_then(|summary| {
                    summary.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB)
                })
                .and_then(Value::as_u64),
            Some(32)
        );
        assert_eq!(
            regression
                .get("contract")
                .and_then(|contract| {
                    contract.get(ax_engine_core::ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB)
                })
                .and_then(|field| field.get("candidate"))
                .and_then(Value::as_u64),
            Some(48)
        );

        let summary = build_compare_summary_markdown(
            &baseline_metrics,
            &candidate_metrics,
            &regression,
            None,
        );
        assert!(summary.contains("metal_direct_decode_batching_opportunity_observed: `true`"));
        assert!(summary.contains("baseline_ax_mlx_kv_capacity_kib: `32`"));
        assert!(summary.contains("candidate_ax_mlx_kv_capacity_kib: `48`"));
        assert!(summary.contains("baseline_ax_mlx_kv_sliding_reclaimable_capacity_kib: `8`"));
        assert!(summary.contains("candidate_ax_mlx_kv_sliding_reclaimable_capacity_kib: `12`"));
    }

    #[test]
    fn mlx_metal_readiness_allows_single_request_decode_without_batching_opportunity() {
        let inputs = MlxMetalReadinessInputs {
            tool_mode: "engine_bringup_runtime",
            selected_backend: "mlx",
            native_dense_dequantized_source: false,
            native_quantized_projection_binding_count: 0,
            direct_decode_batching_opportunity_observed: false,
            metal_complete_model_forward_supported: true,
            metal_real_model_forward: true,
            metal_model_artifacts_validated: true,
            mlx_metal_prefix_layers_attention: true,
            metal_prefix_layers_cpu_reference: false,
            metal_prefix_cpu_reference_dispatch_count: 0,
            mlx_metal_prefix_projection_row_count: 2048,
            metal_prefix_cpu_projection_row_count: 0,
            mlx_metal_prefix_rms_norm_element_count: 1024,
            metal_prefix_cpu_rms_norm_element_count: 0,
            mlx_metal_prefix_ffn_activation_element_count: 1024,
            metal_prefix_cpu_ffn_activation_element_count: 0,
            mlx_metal_prefix_residual_add_element_count: 1024,
            metal_prefix_cpu_residual_add_element_count: 0,
            mlx_metal_prefix_scale_element_count: 1024,
            metal_prefix_cpu_scale_element_count: 0,
            metal_direct_decode_tokens: true,
            metal_direct_decode_model_bound_ffn: true,
            metal_direct_decode_batched_logits_group_count: 0,
            metal_direct_decode_batched_logits_token_count: 0,
            metal_direct_decode_batched_group_fallback_count: 0,
            metal_direct_decode_batched_group_fallback_token_count: 0,
            mlx_metal_direct_decode_projection_row_count: 4096,
            metal_direct_decode_cpu_projection_row_count: 0,
            mlx_metal_direct_decode_rms_norm_element_count: 1024,
            metal_direct_decode_cpu_rms_norm_element_count: 0,
            mlx_metal_direct_decode_ffn_activation_element_count: 1024,
            metal_direct_decode_cpu_ffn_activation_element_count: 0,
            mlx_metal_direct_decode_residual_add_element_count: 1024,
            metal_direct_decode_cpu_residual_add_element_count: 0,
            mlx_metal_direct_decode_scale_element_count: 1024,
            metal_direct_decode_cpu_scale_element_count: 0,
        };
        let readiness = mlx_metal_readiness(inputs);

        assert_eq!(readiness.status, "ready");
        assert!(readiness.hot_path_cpu_fallback_free);
        assert!(readiness.batched_direct_decode_logits_ready);
        assert!(readiness.blockers.is_empty());

        let dequantized_readiness = mlx_metal_readiness(MlxMetalReadinessInputs {
            native_dense_dequantized_source: true,
            ..inputs
        });
        assert_eq!(dequantized_readiness.status, "coverage_gap");
        assert_eq!(
            dequantized_readiness.blockers,
            vec![NATIVE_DENSE_DEQUANTIZED_SOURCE_BLOCKER]
        );

        let quantized_projection_readiness = mlx_metal_readiness(MlxMetalReadinessInputs {
            native_quantized_projection_binding_count: 1,
            ..inputs
        });
        assert_eq!(quantized_projection_readiness.status, "coverage_gap");
        assert_eq!(
            quantized_projection_readiness.blockers,
            vec!["native_quantized_projection_kernel_missing"]
        );
    }

    #[test]
    fn annotate_route_json_with_decode_batching_opportunity_marks_multi_decode_steps() {
        let mut route_json = serialize_route_metadata(&RouteMetadata::empty());
        let step_trace = vec![StepTraceEntry {
            t_ms: 0,
            step_id: Some(StepId(1)),
            admitted_request_ids: Vec::new(),
            selected_request_ids: vec![RequestId(1), RequestId(2)],
            deferred_request_ids: Vec::new(),
            memory_blocked_request_ids: Vec::new(),
            cleanup_request_ids: Vec::new(),
            scheduled_tokens: 2,
            prefix_hits: 0,
            cpu_time_us: 0,
            runner_time_us: 0,
            kv_usage_blocks: 0,
            evictions: 0,
            runner_executed: true,
            route_metadata: RouteMetadata::empty(),
            metal_dispatch: None,
            items: vec![
                StepTraceItem {
                    request_id: RequestId(1),
                    mode: ax_engine_core::ExecutionMode::Decode,
                    position_start: 0,
                    position_end_exclusive: 1,
                    scheduled_token_count: 1,
                    input_token_count: 1,
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
                StepTraceItem {
                    request_id: RequestId(2),
                    mode: ax_engine_core::ExecutionMode::Decode,
                    position_start: 0,
                    position_end_exclusive: 1,
                    scheduled_token_count: 1,
                    input_token_count: 1,
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                },
            ],
        }];

        annotate_route_json_with_decode_batching_opportunity(&mut route_json, &step_trace);

        assert_eq!(
            route_json.get("metal_direct_decode_batching_opportunity_observed"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn trusted_baseline_summary_surfaces_mlx_kv_cache_counters() {
        let trusted_baseline = json!({
            "name": "MLX KV Baseline",
            "source_run_id": "run-kv",
            "source_result_dir": "/tmp/run-kv",
            "manifest": {
                "id": "chat_qwen_short",
                "class": "scenario",
                "scenario": "chat",
                "model_family": "qwen3_dense"
            },
            "runtime": {
                "tool_mode": "engine_bringup_runtime",
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only"
            },
            "route": {
                "execution_semantics": "metal_real_model_forward",
                "ax_mlx_kv_capacity_kib": 64,
                "ax_mlx_kv_sliding_reclaimable_capacity_kib": 8,
                "ax_mlx_kv_growth_count": 4
            },
            "metrics": {
                "ttft_ms": 10.0,
                "decode_tok_s": 100.0,
                "memory_peak_mb": 1024.0,
                "prefix_hit_rate": 0.0
            }
        });

        let summary = build_trusted_baseline_summary_markdown(&trusted_baseline);

        assert!(summary.contains("ax_mlx_kv_capacity_kib: `64`"));
        assert!(summary.contains("ax_mlx_kv_sliding_reclaimable_capacity_kib: `8`"));
        assert!(summary.contains("ax_mlx_kv_growth_count: `4`"));
    }

    #[test]
    fn handle_baseline_snapshots_named_artifact_and_prevents_overwrite() {
        let root = unique_test_dir("trusted-baseline");
        let source_root = root.join("results");
        let baseline_root = root.join("baselines");
        fs::create_dir_all(&source_root).expect("source root should create");
        fs::create_dir_all(&baseline_root).expect("baseline root should create");

        let run_dir = write_execution_artifact_fixture(
            &source_root,
            "1760-chat-qwen-short",
            "run-a",
            10.0,
            120.0,
        );
        let args = vec![
            "--source".to_string(),
            run_dir.display().to_string(),
            "--name".to_string(),
            "Dense Qwen M4 Baseline".to_string(),
            "--output-root".to_string(),
            baseline_root.display().to_string(),
        ];

        handle_baseline(&args).expect("baseline command should succeed");

        let trusted_baseline_dir = baseline_root.join("Dense-Qwen-M4-Baseline");
        assert!(trusted_baseline_dir.is_dir());
        assert!(trusted_baseline_dir.join("manifest.json").is_file());
        assert!(trusted_baseline_dir.join("environment.json").is_file());
        assert!(trusted_baseline_dir.join("metrics.json").is_file());
        assert!(trusted_baseline_dir.join("summary.md").is_file());
        assert!(trusted_baseline_dir.join("routes.json").is_file());
        assert!(trusted_baseline_dir.join("trace.json").is_file());
        assert!(trusted_baseline_dir.join("trusted_baseline.md").is_file());

        let metadata = load_json_value(&trusted_baseline_dir.join("trusted_baseline.json"))
            .expect("trusted baseline metadata should load");
        assert_eq!(
            metadata.get("name").and_then(Value::as_str),
            Some("Dense Qwen M4 Baseline")
        );
        assert_eq!(
            metadata.get("slug").and_then(Value::as_str),
            Some("Dense-Qwen-M4-Baseline")
        );
        assert_eq!(
            metadata.get("source_run_id").and_then(Value::as_str),
            Some("run-a")
        );
        assert_eq!(
            nested_value(&metadata, &["manifest", "id"]).and_then(Value::as_str),
            Some("chat_qwen_short")
        );

        let error = handle_baseline(&args).expect_err("baseline overwrite should fail closed");
        let CliError::Contract(message) = error else {
            panic!("baseline overwrite should return a contract error");
        };
        assert!(message.contains("trusted baseline already exists"));

        fs::remove_dir_all(&root).expect("test directory should clean up");
    }

    #[test]
    fn compare_artifacts_include_trusted_baseline_identity() {
        let root = unique_test_dir("compare-trusted-baseline");
        let runs_root = root.join("runs");
        let compare_root = root.join("compare");
        let baseline_root = root.join("baselines");
        fs::create_dir_all(&runs_root).expect("runs root should create");
        fs::create_dir_all(&compare_root).expect("compare root should create");
        fs::create_dir_all(&baseline_root).expect("baseline root should create");

        let baseline_run = write_execution_artifact_fixture(
            &runs_root,
            "baseline-run",
            "baseline-run",
            10.0,
            120.0,
        );
        let candidate_run = write_execution_artifact_fixture(
            &runs_root,
            "candidate-run",
            "candidate-run",
            9.0,
            132.0,
        );

        let baseline_args = vec![
            "--source".to_string(),
            baseline_run.display().to_string(),
            "--name".to_string(),
            "Dense Qwen Trusted".to_string(),
            "--output-root".to_string(),
            baseline_root.display().to_string(),
        ];
        handle_baseline(&baseline_args).expect("trusted baseline should snapshot");
        let trusted_baseline_dir = baseline_root.join("Dense-Qwen-Trusted");

        let compare_dir =
            write_compare_artifacts(&trusted_baseline_dir, &candidate_run, &compare_root)
                .expect("compare should succeed against trusted baseline");
        let regression = load_json_value(&compare_dir.join("regression.json"))
            .expect("regression artifact should load");
        let summary = fs::read_to_string(compare_dir.join("comparison.md"))
            .expect("comparison summary should load");

        assert_eq!(
            nested_value(&regression, &["trusted_baseline", "name"]).and_then(Value::as_str),
            Some("Dense Qwen Trusted")
        );
        assert_eq!(
            nested_value(&regression, &["trusted_baseline", "source_run_id"])
                .and_then(Value::as_str),
            Some("baseline-run")
        );
        assert!(summary.contains("trusted_baseline: `Dense Qwen Trusted`"));

        fs::remove_dir_all(&root).expect("test directory should clean up");
    }

    #[test]
    fn handle_matrix_compare_writes_rollup_and_member_compares() {
        let root = unique_test_dir("matrix-compare");
        let runs_root = root.join("runs");
        let matrix_root = root.join("matrix");
        let compare_root = root.join("compare");
        fs::create_dir_all(&runs_root).expect("runs root should create");
        fs::create_dir_all(&matrix_root).expect("matrix root should create");
        fs::create_dir_all(&compare_root).expect("compare root should create");

        let baseline_chat = write_execution_artifact_fixture(
            &runs_root,
            "baseline-chat",
            "baseline-chat",
            10.0,
            120.0,
        );
        let candidate_chat = write_execution_artifact_fixture(
            &runs_root,
            "candidate-chat",
            "candidate-chat",
            9.0,
            132.0,
        );
        let baseline_concurrent = write_execution_artifact_fixture(
            &runs_root,
            "baseline-concurrent",
            "baseline-concurrent",
            12.0,
            100.0,
        );
        let candidate_concurrent = write_execution_artifact_fixture(
            &runs_root,
            "candidate-concurrent",
            "candidate-concurrent",
            11.0,
            108.0,
        );

        let baseline_matrix = write_matrix_result_fixture(
            &matrix_root,
            "baseline-matrix",
            "baseline-matrix-run",
            &[
                json!({
                    "label": "Chat Qwen Short",
                    "manifest_id": "chat_qwen_short",
                    "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
                    "scenario": "chat",
                    "model_family": "qwen3_dense",
                    "status": "ok",
                    "tool_mode": "engine_bringup_runtime",
                    "selected_backend": "mlx",
                    "support_tier": "mlx_preview",
                    "resolution_policy": "mlx_only",
                    "result_dir": baseline_chat.display().to_string(),
                    "ttft_ms": 10.0,
                    "decode_tok_s": 120.0,
                    "prefix_hit_rate": 0.0
                }),
                json!({
                    "label": "Concurrent Qwen Dual",
                    "manifest_id": "concurrent_qwen_dual",
                    "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/concurrent_qwen_dual.json").as_str(),
                    "scenario": "concurrent",
                    "model_family": "qwen3_dense",
                    "status": "ok",
                    "tool_mode": "engine_bringup_runtime",
                    "selected_backend": "mlx",
                    "support_tier": "mlx_preview",
                    "resolution_policy": "mlx_only",
                    "result_dir": baseline_concurrent.display().to_string(),
                    "ttft_ms": 12.0,
                    "decode_tok_s": 100.0,
                    "prefix_hit_rate": 0.0
                }),
            ],
        );
        let candidate_matrix = write_matrix_result_fixture(
            &matrix_root,
            "candidate-matrix",
            "candidate-matrix-run",
            &[
                json!({
                    "label": "Chat Qwen Short",
                    "manifest_id": "chat_qwen_short",
                    "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
                    "scenario": "chat",
                    "model_family": "qwen3_dense",
                    "status": "ok",
                    "tool_mode": "engine_bringup_runtime",
                    "selected_backend": "mlx",
                    "support_tier": "mlx_preview",
                    "resolution_policy": "mlx_only",
                    "result_dir": candidate_chat.display().to_string(),
                    "ttft_ms": 9.0,
                    "decode_tok_s": 132.0,
                    "prefix_hit_rate": 0.0
                }),
                json!({
                    "label": "Concurrent Qwen Dual",
                    "manifest_id": "concurrent_qwen_dual",
                    "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/concurrent_qwen_dual.json").as_str(),
                    "scenario": "concurrent",
                    "model_family": "qwen3_dense",
                    "status": "ok",
                    "tool_mode": "engine_bringup_runtime",
                    "selected_backend": "mlx",
                    "support_tier": "mlx_preview",
                    "resolution_policy": "mlx_only",
                    "result_dir": candidate_concurrent.display().to_string(),
                    "ttft_ms": 11.0,
                    "decode_tok_s": 108.0,
                    "prefix_hit_rate": 0.0
                }),
            ],
        );

        let args = vec![
            "--baseline".to_string(),
            baseline_matrix.display().to_string(),
            "--candidate".to_string(),
            candidate_matrix.display().to_string(),
            "--output-root".to_string(),
            compare_root.display().to_string(),
        ];

        handle_matrix_compare(&args).expect("matrix compare should succeed");

        let run_dir = fs::read_dir(&compare_root)
            .expect("compare root should be readable")
            .map(|entry| entry.expect("directory entry should read").path())
            .next()
            .expect("matrix compare result should exist");
        let matrix_regression = load_json_value(&run_dir.join("matrix_regression.json"))
            .expect("matrix regression should load");
        let summary = fs::read_to_string(run_dir.join("summary.md"))
            .expect("matrix compare summary should load");

        assert_eq!(
            matrix_regression.get("id").and_then(Value::as_str),
            Some("test_mlx_mlx_matrix")
        );
        assert_eq!(
            nested_value(&matrix_regression, &["summary", "member_count"]).and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            matrix_regression
                .get("members")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
        assert!(summary.contains("Benchmark Matrix Compare"));
        assert!(summary.contains("Chat Qwen Short"));
        assert!(summary.contains("Concurrent Qwen Dual"));

        fs::remove_dir_all(&root).expect("test directory should clean up");
    }

    #[test]
    fn handle_matrix_compare_requires_matrix_result_directories() {
        let root = unique_test_dir("matrix-compare-invalid-path");
        let runs_root = root.join("runs");
        let matrix_root = root.join("matrix");
        let compare_root = root.join("compare");
        fs::create_dir_all(&runs_root).expect("runs root should create");
        fs::create_dir_all(&matrix_root).expect("matrix root should create");
        fs::create_dir_all(&compare_root).expect("compare root should create");

        let baseline_chat = write_execution_artifact_fixture(
            &runs_root,
            "baseline-chat",
            "baseline-chat",
            10.0,
            120.0,
        );
        let candidate_chat = write_execution_artifact_fixture(
            &runs_root,
            "candidate-chat",
            "candidate-chat",
            9.0,
            132.0,
        );

        let baseline_matrix = write_matrix_result_fixture(
            &matrix_root,
            "baseline-matrix",
            "baseline-matrix-run",
            &[json!({
                "label": "Chat Qwen Short",
                "manifest_id": "chat_qwen_short",
                "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
                "scenario": "chat",
                "model_family": "qwen3_dense",
                "status": "ok",
                "tool_mode": "engine_bringup_runtime",
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only",
                "result_dir": baseline_chat.display().to_string(),
                "ttft_ms": 10.0,
                "decode_tok_s": 120.0,
                "prefix_hit_rate": 0.0
            })],
        );
        let candidate_matrix = write_matrix_result_fixture(
            &matrix_root,
            "candidate-matrix",
            "candidate-matrix-run",
            &[json!({
                "label": "Chat Qwen Short",
                "manifest_id": "chat_qwen_short",
                "manifest_path": repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
                "scenario": "chat",
                "model_family": "qwen3_dense",
                "status": "ok",
                "tool_mode": "engine_bringup_runtime",
                "selected_backend": "mlx",
                "support_tier": "mlx_preview",
                "resolution_policy": "mlx_only",
                "result_dir": candidate_chat.display().to_string(),
                "ttft_ms": 9.0,
                "decode_tok_s": 132.0,
                "prefix_hit_rate": 0.0
            })],
        );

        let args = vec![
            "--baseline".to_string(),
            baseline_matrix.join("matrix.json").display().to_string(),
            "--candidate".to_string(),
            candidate_matrix.join("matrix.json").display().to_string(),
            "--output-root".to_string(),
            compare_root.display().to_string(),
        ];

        let error =
            handle_matrix_compare(&args).expect_err("matrix compare should reject file paths");
        let CliError::Contract(message) = error else {
            panic!("matrix compare should return a contract error for non-directory inputs");
        };

        assert!(message.contains("required directory does not exist or is not a directory"));

        fs::remove_dir_all(&root).expect("test directory should clean up");
    }

    #[test]
    fn handle_compare_rejects_contract_failure_artifact_directories() {
        let root = unique_test_dir("compare-contract-failure");
        let output_root = root.join("results");
        fs::create_dir_all(&output_root).expect("output root should create");

        let mut manifest = llama_cpp_scenario_manifest("http://127.0.0.1:8081");
        manifest.runtime.backend_adapter = None;
        let manifest_path = root.join("llama-cpp-missing-adapter.json");
        write_json_file(&manifest_path, &manifest).expect("manifest should write");

        let scenario_args = vec![
            "--manifest".to_string(),
            manifest_path.display().to_string(),
            "--output-root".to_string(),
            output_root.display().to_string(),
        ];
        let _ = handle_scenario(&scenario_args)
            .expect_err("shared-prefix scenario should emit contract failure artifacts");

        let run_dir = fs::read_dir(&output_root)
            .expect("output root should be readable")
            .map(|entry| entry.expect("directory entry should read").path())
            .next()
            .expect("contract failure run should exist");

        let compare_output = root.join("compare-results");
        fs::create_dir_all(&compare_output).expect("compare output should create");
        let compare_args = vec![
            "--baseline".to_string(),
            run_dir.display().to_string(),
            "--candidate".to_string(),
            run_dir.display().to_string(),
            "--output-root".to_string(),
            compare_output.display().to_string(),
        ];

        let error =
            handle_compare(&compare_args).expect_err("compare should reject contract-failure runs");
        let CliError::Contract(message) = error else {
            panic!("compare should return a contract error for contract-failure artifacts");
        };

        assert!(message.contains("contract-failure result"));
        assert!(message.contains("llama_backend_adapter_required"));

        fs::remove_dir_all(&root).expect("test directory should clean up");
    }

    #[test]
    fn compare_validation_rejects_runtime_drift() {
        let baseline = load_test_manifest_json(
            repo_manifest_path("benchmarks/manifests/scenario/chat_qwen_short.json").as_str(),
        );
        let mut candidate = baseline.clone();
        candidate["runtime"]["max_batch_tokens"] = json!(4096);

        let error = validate_comparable_manifests(&baseline, &candidate)
            .expect_err("compare validation should reject runtime drift");
        let CliError::Contract(message) = error else {
            panic!("compare validation should return a contract error");
        };

        assert!(message.contains("runtime"));
    }

    #[test]
    fn compare_validation_rejects_replay_event_drift() {
        let baseline = load_test_manifest_json(
            repo_manifest_path("benchmarks/manifests/replay/full_prefix_to_decode_branch.json")
                .as_str(),
        );
        let mut candidate = baseline.clone();
        candidate["events"][1]["t_ms"] = json!(2);

        let error = validate_comparable_manifests(&baseline, &candidate)
            .expect_err("compare validation should reject replay event drift");
        let CliError::Contract(message) = error else {
            panic!("compare validation should return a contract error");
        };

        assert!(message.contains("events"));
    }

    #[test]
    fn compare_validation_rejects_environment_arch_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "x86_64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject environment drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("machine.arch"));
    }

    #[test]
    fn compare_validation_rejects_environment_kernel_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.5.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject kernel drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("software.kernel_release"));
    }

    #[test]
    fn compare_validation_rejects_metal_toolchain_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let mut candidate_runtime = mlx_runtime_fixture();
        candidate_runtime["metal_toolchain"]["metal"]["available"] = json!(false);
        candidate_runtime["metal_toolchain"]["fully_available"] = json!(false);
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": candidate_runtime,
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject metal toolchain drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("runtime.metal_toolchain"));
    }

    #[test]
    fn compare_validation_rejects_environment_backend_adapter_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 64
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8082"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 64
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject backend adapter drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("runtime.backend_adapter"));
    }

    #[test]
    fn compare_validation_rejects_environment_native_model_drift() {
        let mut baseline_runtime = mlx_runtime_fixture();
        baseline_runtime["mlx_model"] = json!({
            "artifacts_source": "explicit_config",
            "model_family": "qwen3_dense",
            "tensor_format": "safetensors",
            "layer_count": 36,
            "tensor_count": 402,
            "tie_word_embeddings": false,
            "bindings_prepared": false,
            "buffers_bound": false,
            "buffer_count": 0,
            "buffer_bytes": 0
        });
        let mut candidate_runtime = baseline_runtime.clone();
        candidate_runtime["mlx_model"]["tensor_count"] = json!(401);

        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": baseline_runtime,
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "mlx_runtime"
            },
            "runtime": candidate_runtime,
            "route": {
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject native model drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("runtime.mlx_model.tensor_count"));
    }

    #[test]
    fn compare_validation_rejects_route_execution_semantics_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_numeric_scaffold_only",
                "metal_numeric_scaffold_only": true,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": false,
                "metal_real_model_tensor_inputs": false,
                "metal_model_artifacts_validated": false,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_forward",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": true,
                "metal_model_conditioned_inputs": true,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject route execution semantics drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.execution_semantics"));
    }

    #[test]
    fn compare_validation_rejects_complete_model_forward_support_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_tensor_inputs",
                "metal_numeric_scaffold_only": false,
                "metal_complete_model_forward_supported": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_tensor_inputs",
                "metal_numeric_scaffold_only": false,
                "metal_complete_model_forward_supported": true,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject complete-model-forward support drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.metal_complete_model_forward_supported"));
    }

    #[test]
    fn compare_validation_rejects_route_direct_decode_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_tensor_inputs",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "metal_direct_decode_tokens": false,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_tensor_inputs",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "metal_direct_decode_tokens": true,
                "metal_direct_decode_checksum_lo": 42,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject route direct decode drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.metal_direct_decode_tokens"));
    }

    #[test]
    fn compare_validation_rejects_route_prefix_attention_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_multilayer_native_prefix_attention",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "mlx_metal_prefix_layers_attention": true,
                "metal_prefix_layers_cpu_reference": false,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_multilayer_native_prefix_attention",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "mlx_metal_prefix_layers_attention": false,
                "metal_prefix_layers_cpu_reference": true,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject route prefix-attention drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.mlx_metal_prefix_layers_attention"));
    }

    #[test]
    fn compare_validation_rejects_route_prefix_dispatch_count_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_multilayer_mixed_prefix_attention",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "mlx_metal_prefix_layers_attention": true,
                "metal_prefix_layers_cpu_reference": true,
                "mlx_metal_prefix_dispatch_count": 35,
                "metal_prefix_cpu_reference_dispatch_count": 1,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_multilayer_mixed_prefix_attention",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "mlx_metal_prefix_layers_attention": true,
                "metal_prefix_layers_cpu_reference": true,
                "mlx_metal_prefix_dispatch_count": 36,
                "metal_prefix_cpu_reference_dispatch_count": 0,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject route prefix dispatch-count drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.mlx_metal_prefix_dispatch_count"));
    }

    #[test]
    fn compare_validation_rejects_route_native_dense_projection_coverage_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_forward",
                "metal_numeric_scaffold_only": false,
                "metal_complete_model_forward_supported": true,
                "metal_real_model_forward": true,
                "metal_model_conditioned_inputs": true,
                "mlx_metal_prefix_layers_attention": true,
                "metal_prefix_layers_cpu_reference": false,
                "mlx_metal_prefix_dispatch_count": 35,
                "metal_prefix_cpu_reference_dispatch_count": 0,
                "metal_direct_decode_tokens": true,
                "metal_direct_decode_model_bound_ffn": true,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "mlx_metal_projection_f32_binding_count": 0,
                "mlx_metal_projection_f16_binding_count": 28,
                "mlx_metal_projection_bf16_binding_count": 0,
                "mlx_metal_projection_unsupported_binding_count": 0,
                "mlx_metal_rms_norm_f32_binding_count": 0,
                "mlx_metal_rms_norm_f16_binding_count": 28,
                "mlx_metal_rms_norm_bf16_binding_count": 0,
                "mlx_metal_rms_norm_unsupported_binding_count": 0,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_forward",
                "metal_numeric_scaffold_only": false,
                "metal_complete_model_forward_supported": true,
                "metal_real_model_forward": true,
                "metal_model_conditioned_inputs": true,
                "mlx_metal_prefix_layers_attention": true,
                "metal_prefix_layers_cpu_reference": false,
                "mlx_metal_prefix_dispatch_count": 35,
                "metal_prefix_cpu_reference_dispatch_count": 0,
                "metal_direct_decode_tokens": true,
                "metal_direct_decode_model_bound_ffn": true,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "mlx_metal_projection_f32_binding_count": 0,
                "mlx_metal_projection_f16_binding_count": 27,
                "mlx_metal_projection_bf16_binding_count": 0,
                "mlx_metal_projection_unsupported_binding_count": 1,
                "mlx_metal_rms_norm_f32_binding_count": 0,
                "mlx_metal_rms_norm_f16_binding_count": 28,
                "mlx_metal_rms_norm_bf16_binding_count": 0,
                "mlx_metal_rms_norm_unsupported_binding_count": 0,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject native dense projection coverage drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.mlx_metal_projection_f16_binding_count"));
    }

    #[test]
    fn compare_validation_rejects_route_direct_decode_ffn_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_tensor_inputs",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "metal_direct_decode_tokens": true,
                "metal_direct_decode_model_bound_ffn": false,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "engine_bringup_runtime"
            },
            "runtime": mlx_runtime_fixture(),
            "route": {
                "execution_semantics": "metal_real_model_tensor_inputs",
                "metal_numeric_scaffold_only": false,
                "metal_real_model_forward": false,
                "metal_model_conditioned_inputs": true,
                "metal_direct_decode_tokens": true,
                "metal_direct_decode_model_bound_ffn": true,
                "metal_real_model_tensor_inputs": true,
                "metal_model_artifacts_validated": true,
                "prefix_cache_path": "metadata_lookup",
                "prefix_cache_evidence": "none_observed",
                "prefix_reuse_provenance": "none_observed"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "scenario"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject route direct decode ffn drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.metal_direct_decode_model_bound_ffn"));
    }

    #[test]
    fn compare_validation_rejects_prefix_reuse_provenance_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 64
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "replay"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "live_request_share",
                "prefix_cache_evidence": "engine_live_request_scheduler_reuse",
                "prefix_reuse_provenance": "mlx_live_request_share"
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "replay"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject prefix reuse provenance drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(
            message.contains("route.prefix_cache_path")
                || message.contains("route.prefix_reuse_provenance")
        );
    }

    #[test]
    fn compare_validation_rejects_prefix_cache_evidence_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 64
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "replay"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "engine_live_request_scheduler_reuse",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 64
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "replay"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject prefix-cache evidence drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.prefix_cache_evidence"));
    }

    #[test]
    fn compare_validation_rejects_backend_reported_cached_prompt_tokens_drift() {
        let baseline = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 64
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "replay"
            }
        });
        let candidate = json!({
            "schema_version": "ax.engine_bench.environment.v1",
            "machine": {
                "arch": "aarch64",
                "system_model": "Mac17,6",
                "soc": "Apple M5 Max",
                "memory_bytes": 137438953472_u64
            },
            "software": {
                "os_family": "macos",
                "os_version": "26.4.1",
                "os_build": "25E253",
                "kernel_release": "25.4.0",
                "metal_driver": "system-default",
                "tool_mode": "llama_cpp_stepwise_runtime"
            },
            "runtime": llama_runtime_fixture("http://127.0.0.1:8081"),
            "route": {
                "prefix_cache_path": "delegated_prompt_cache",
                "prefix_cache_evidence": "backend_reported_cached_prompt_tokens",
                "prefix_reuse_provenance": "delegated_backend_prompt_cache",
                "backend_reported_cached_prompt_tokens": 32
            },
            "benchmark": {
                "schema_family": "ax.engine_bench.v1",
                "subcommand": "replay"
            }
        });

        let error = validate_comparable_environments(&baseline, &candidate)
            .expect_err("compare validation should reject delegated cached-token drift");
        let CliError::Contract(message) = error else {
            panic!("environment validation should return a contract error");
        };

        assert!(message.contains("route.backend_reported_cached_prompt_tokens"));
    }
}
