mod artifacts;
#[cfg(test)]
mod autotune;
mod baseline;
mod config;
mod execution;
mod format;
mod matrix;
mod trace;
mod util;

use super::*;
pub(crate) use artifacts::*;
#[cfg(test)]
pub(crate) use autotune::*;
pub(crate) use baseline::*;
pub(crate) use config::*;
pub(crate) use execution::*;
pub(crate) use format::*;
pub(crate) use matrix::*;
pub(crate) use trace::*;
pub(crate) use util::*;

pub(crate) fn run() -> Result<(), CliError> {
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
        "generate-manifest" => handle_generate_manifest(&remaining),
        "metal-build" => handle_metal_build(&remaining),
        "serving-stress" => handle_serving_stress(&remaining),
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

pub(crate) fn handle_scenario(args: &[String]) -> Result<(), CliError> {
    let manifest = required_flag(args, "--manifest")?;
    let output_root = required_flag(args, "--output-root")?;
    let json = has_flag(args, "--json");
    let write_trace = !has_flag(args, "--no-trace");

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
            let summary = BenchmarkArtifactSummary::manifest_run(
                "scenario",
                &manifest,
                &output_root,
                &artifact_dir,
                "contract_failure",
            );
            print_benchmark_artifact_summary(&summary, json)?;
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
        write_trace,
    )?;

    let summary = BenchmarkArtifactSummary::manifest_run(
        "scenario",
        &manifest,
        &output_root,
        &artifact_dir,
        execution.status_label(),
    );
    print_benchmark_artifact_summary(&summary, json)?;

    enforce_runtime_gates(&execution)
}

pub(crate) fn handle_replay(args: &[String]) -> Result<(), CliError> {
    let manifest = required_flag(args, "--manifest")?;
    let output_root = required_flag(args, "--output-root")?;
    let json = has_flag(args, "--json");
    let write_trace = !has_flag(args, "--no-trace");

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
            let summary = BenchmarkArtifactSummary::manifest_run(
                "replay",
                &manifest,
                &output_root,
                &artifact_dir,
                "contract_failure",
            );
            print_benchmark_artifact_summary(&summary, json)?;
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
        write_trace,
    )?;

    let summary = BenchmarkArtifactSummary::manifest_run(
        "replay",
        &manifest,
        &output_root,
        &artifact_dir,
        execution.status_label(),
    );
    print_benchmark_artifact_summary(&summary, json)?;

    enforce_runtime_gates(&execution)
}

pub(crate) fn handle_autotune(args: &[String]) -> Result<(), CliError> {
    let _ = parse_autotune_args(args)?;
    Err(CliError::Contract(
        "autotune is not yet available; use explicit scenario or matrix benchmark runs".to_string(),
    ))
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AutotuneArgs {
    pub(crate) manifest_path: PathBuf,
    pub(crate) output_root: PathBuf,
    pub(crate) iterations: usize,
    pub(crate) exploration_weight: f64,
    pub(crate) max_batch_token_options: Option<Vec<u32>>,
    pub(crate) kv_total_block_options: Option<Vec<Option<u32>>>,
    pub(crate) prefix_cache_options: Option<Vec<bool>>,
    pub(crate) disable_history: bool,
}

pub(crate) fn parse_autotune_args(args: &[String]) -> Result<AutotuneArgs, CliError> {
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

pub(crate) fn handle_compare(args: &[String]) -> Result<(), CliError> {
    let baseline = required_flag(args, "--baseline")?;
    let candidate = required_flag(args, "--candidate")?;
    let output_root = required_flag(args, "--output-root")?;
    let json = has_flag(args, "--json");

    require_existing_path(&baseline)?;
    require_existing_path(&candidate)?;
    reject_contract_failure_artifact_dir(&baseline, "baseline")?;
    reject_contract_failure_artifact_dir(&candidate, "candidate")?;
    ensure_output_root(&output_root)?;
    let comparison_dir = write_compare_artifacts(&baseline, &candidate, &output_root)?;

    let summary = BenchmarkArtifactSummary::comparison(
        "compare",
        &baseline,
        &candidate,
        &output_root,
        &comparison_dir,
    );
    print_benchmark_artifact_summary(&summary, json)?;
    Ok(())
}

pub(crate) fn handle_baseline(args: &[String]) -> Result<(), CliError> {
    let source = required_flag(args, "--source")?;
    let name = required_string_flag(args, "--name")?;
    let output_root = required_flag(args, "--output-root")?;
    let json = has_flag(args, "--json");

    require_existing_dir(&source)?;
    reject_contract_failure_artifact_dir(&source, "source")?;
    ensure_output_root(&output_root)?;
    let baseline_dir = write_trusted_baseline_artifacts(&source, &name, &output_root)?;

    let summary = BenchmarkArtifactSummary::baseline(&source, &name, &output_root, &baseline_dir);
    print_benchmark_artifact_summary(&summary, json)?;
    Ok(())
}

pub(crate) fn handle_matrix_compare(args: &[String]) -> Result<(), CliError> {
    let baseline = required_flag(args, "--baseline")?;
    let candidate = required_flag(args, "--candidate")?;
    let output_root = required_flag(args, "--output-root")?;
    let json = has_flag(args, "--json");

    require_existing_dir(&baseline)?;
    require_existing_dir(&candidate)?;
    ensure_output_root(&output_root)?;
    let comparison_dir = write_matrix_compare_artifacts(&baseline, &candidate, &output_root)?;

    let summary = BenchmarkArtifactSummary::comparison(
        "matrix-compare",
        &baseline,
        &candidate,
        &output_root,
        &comparison_dir,
    );
    print_benchmark_artifact_summary(&summary, json)?;
    Ok(())
}

pub(crate) fn handle_matrix(args: &[String]) -> Result<(), CliError> {
    let manifest = required_flag(args, "--manifest")?;
    let output_root = required_flag(args, "--output-root")?;
    let json = has_flag(args, "--json");
    let write_trace = !has_flag(args, "--no-trace");

    require_existing_file(&manifest)?;
    ensure_output_root(&output_root)?;
    let matrix_manifest = load_matrix_manifest(&manifest)?;
    validate_matrix_manifest(&matrix_manifest)?;
    let execution =
        execute_matrix_manifest(&manifest, &matrix_manifest, &output_root, write_trace)?;

    let summary = BenchmarkArtifactSummary::manifest_run(
        "matrix",
        &manifest,
        &output_root,
        &execution.result_dir,
        &execution.overall_status.to_string(),
    );
    print_benchmark_artifact_summary(&summary, json)?;
    enforce_matrix_gates(&execution)
}

pub(crate) fn handle_doctor(args: &[String]) -> Result<(), CliError> {
    let doctor_args = parse_doctor_args(args)?;
    let current_dir = std::env::current_dir().ok();
    let runtime_assets = detect_runtime_assets_report(current_dir.as_deref());
    let mut report = build_doctor_report_for_model_and_runtime(
        current_host_report(),
        current_metal_toolchain_report(),
        runtime_assets,
        doctor_args.mlx_model_artifacts_dir.as_deref(),
    );
    report.workflow = detect_doctor_workflow_report()?;

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

pub(crate) fn handle_metal_build(args: &[String]) -> Result<(), CliError> {
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

pub(crate) fn handle_generate(args: &[String]) -> Result<(), CliError> {
    let inference_args = parse_inference_args(args, "generate")?;
    let response = run_inference_generate(&inference_args)?;
    print!(
        "{}",
        render_generate_response(&response, inference_args.json)?
    );
    Ok(())
}

pub(crate) fn handle_serving_stress(args: &[String]) -> Result<(), CliError> {
    use crate::harness::pressure_observer::{PlatformProbes, StaticProbes, observe_and_record};
    use crate::workloads::Workload;
    use crate::workloads::cancellation_during_prefill::CancellationDuringPrefill;
    use crate::workloads::concurrent_short_inserts::ConcurrentShortInserts;
    use crate::workloads::long_prefill_vs_decode::LongPrefillVsDecode;
    use crate::workloads::partial_prefix_hit::PartialPrefixHit;
    use crate::workloads::post_restart_cache_safety::PostRestartCacheSafety;
    use crate::workloads::tool_output_repetition::ToolOutputRepetition;
    use crate::workloads::{WorkloadContext, WorkloadOutcome};
    use ax_engine_core::PressureThresholds;

    let workload_name = optional_named_flag(args, "--workload")
        .unwrap_or_else(|| "long_prefill_vs_decode".to_string());
    let cli_artifacts_dir =
        optional_named_flag(args, "--mlx-model-artifacts-dir").map(PathBuf::from);
    let cli_model_id = optional_named_flag(args, "--model-id");
    let prefill_tokens = parse_optional_u32_flag(args, "--prefill-tokens")?;
    let decode_tokens = parse_optional_u32_flag(args, "--decode-tokens")?;
    let concurrent_short_requests = parse_optional_u32_flag(args, "--concurrent-short-requests")?;
    let short_prefix_tokens = parse_optional_u32_flag(args, "--short-prefix-tokens")?;
    let seed = parse_optional_u64_flag(args, "--seed")?.unwrap_or(0);
    let output_path = optional_named_flag(args, "--output-path").map(PathBuf::from);
    let json = has_flag(args, "--json");

    let ctx_artifacts = cli_artifacts_dir.filter(|p| p.exists()).or_else(|| {
        std::env::var_os("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR")
            .map(PathBuf::from)
            .filter(|p| p.exists())
    });
    let ctx = WorkloadContext {
        mlx_model_artifacts_dir: ctx_artifacts,
        seed,
    };

    let supported_workloads = "long_prefill_vs_decode, partial_prefix_hit, tool_output_repetition, \
         cancellation_during_prefill, post_restart_cache_safety, concurrent_short_inserts";

    let mut outcome = match workload_name.as_str() {
        "long_prefill_vs_decode" => {
            let mut fixture = LongPrefillVsDecode::default();
            if let Some(model_id) = cli_model_id {
                fixture.model_id = model_id;
            }
            if let Some(value) = prefill_tokens {
                fixture.prefill_tokens = value;
            }
            if let Some(value) = decode_tokens {
                fixture.decode_tokens = value;
            }
            if let Some(value) = concurrent_short_requests {
                fixture.concurrent_short_requests = value;
            }
            if let Some(value) = short_prefix_tokens {
                fixture.short_prefix_tokens = value;
            }
            fixture.run(&ctx)
        }
        "partial_prefix_hit" => {
            let mut fixture = PartialPrefixHit::default();
            if let Some(model_id) = cli_model_id {
                fixture.model_id = model_id;
            }
            if let Some(value) = decode_tokens {
                fixture.decode_tokens = value;
            }
            fixture.run(&ctx)
        }
        "tool_output_repetition" => {
            let mut fixture = ToolOutputRepetition::default();
            if let Some(model_id) = cli_model_id {
                fixture.model_id = model_id;
            }
            if let Some(value) = decode_tokens {
                fixture.decode_tokens = value;
            }
            fixture.run(&ctx)
        }
        "cancellation_during_prefill" => {
            let mut fixture = CancellationDuringPrefill::default();
            if let Some(model_id) = cli_model_id {
                fixture.model_id = model_id;
            }
            if let Some(value) = prefill_tokens {
                fixture.prefill_tokens = value;
            }
            if let Some(value) = decode_tokens {
                fixture.decode_tokens = value;
            }
            fixture.run(&ctx)
        }
        "post_restart_cache_safety" => {
            let fixture = PostRestartCacheSafety::default();
            fixture.run(&ctx)
        }
        "concurrent_short_inserts" => {
            let mut fixture = ConcurrentShortInserts::default();
            if let Some(model_id) = cli_model_id {
                fixture.model_id = model_id;
            }
            if let Some(value) = prefill_tokens {
                fixture.long_prefill_tokens = value;
            }
            if let Some(value) = decode_tokens {
                fixture.long_decode_tokens = value;
            }
            if let Some(value) = short_prefix_tokens {
                fixture.short_prefix_tokens = value;
            }
            if let Some(value) = concurrent_short_requests {
                fixture.short_request_count = value;
            }
            fixture.run(&ctx)
        }
        other => {
            return Err(CliError::Usage(format!(
                "unknown workload: {other}\n\nsupported workloads: {supported_workloads}"
            )));
        }
    };

    // I-4 observe-mode: record a real pressure snapshot onto every
    // Completed report. PlatformProbes uses POSIX getrusage for host RSS
    // and mlx_get_active_memory + Metal recommendedMaxWorkingSetSize for
    // the device. On hosts where Metal is unavailable (CPU-only CI,
    // non-macOS targets) we fall back to StaticProbes::default() so the
    // level decisions still appear with Normal values; absence of the
    // byte counters then signals "probe unavailable" to downstream
    // tooling (PRD §7.2).
    if let WorkloadOutcome::Completed { report } = &mut outcome {
        match PlatformProbes::from_metal_runtime() {
            Some(probes) => {
                let _ = observe_and_record(&probes, PressureThresholds::default(), report);
            }
            None => {
                let probes = StaticProbes::default();
                let _ = observe_and_record(&probes, PressureThresholds::default(), report);
            }
        }
    }

    let outcome_json = outcome.to_json();
    let rendered = if json {
        outcome_json.to_string()
    } else {
        serde_json::to_string_pretty(&outcome_json)
            .map_err(|error| CliError::Runtime(format!("failed to render outcome JSON: {error}")))?
    };
    println!("{rendered}");

    if let Some(path) = output_path {
        fs::write(&path, rendered.as_bytes()).map_err(|error| {
            CliError::Runtime(format!(
                "failed to write outcome to {}: {error}",
                path.display()
            ))
        })?;
    }

    match outcome {
        WorkloadOutcome::Skipped { .. } | WorkloadOutcome::Completed { .. } => Ok(()),
        WorkloadOutcome::Failed { error, .. } => {
            Err(CliError::Runtime(format!("serving-stress failed: {error}")))
        }
    }
}

pub(crate) fn handle_stream(args: &[String]) -> Result<(), CliError> {
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

pub(crate) fn load_manifest(path: &Path) -> Result<BenchmarkManifest, CliError> {
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

pub(crate) fn load_matrix_manifest(path: &Path) -> Result<BenchmarkMatrixManifest, CliError> {
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

pub(crate) fn resolve_manifest_runtime_paths(
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

pub(crate) fn resolve_manifest_path(
    path: &Path,
    manifest_path: &Path,
) -> Result<PathBuf, CliError> {
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

pub(crate) fn validate_manifest(
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

pub(crate) fn validate_matrix_manifest(
    matrix_manifest: &BenchmarkMatrixManifest,
) -> Result<(), CliError> {
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

pub(crate) fn execute_manifest_runtime(
    manifest: &BenchmarkManifest,
) -> Result<RuntimeResult, CliError> {
    execute_manifest_runtime_with_test_options(manifest, false)
}

pub(crate) fn execute_manifest_runtime_with_test_options(
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
