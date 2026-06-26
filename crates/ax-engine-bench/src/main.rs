#![recursion_limit = "512"]
#![allow(clippy::collapsible_if, clippy::single_match)]

mod args;
mod artifact_files;
mod artifact_summary;
mod cli;
mod compare_json;
mod determinism;
mod doctor;
mod doctor_workflow;
mod environment_probe;
mod error;
mod generate_manifest;
mod harness;
mod inference_args;
mod inference_render;
mod json_io;
mod labels;
mod logging;
mod metal_build;
mod path_utils;
mod route_json;
mod route_metadata;
mod route_readiness;
mod stats;
mod synthetic;
mod token_sources;
mod workloads;

use ax_engine_core::{
    CacheGroupId, EngineCore, MetalBuildStatus, MetalDispatchTrace, MetalKernelBuildRequest,
    NativeModelArtifacts, NativeModelBindingSummary, RequestId, RequestSnapshot, RequestSubmission,
    RouteMetadata, SamplingParams, SequenceNo, StepId, build_phase1_kernel_artifacts,
    upsert_route_decision,
};
use ax_engine_sdk::{
    BackendPolicy, EngineSession, EngineSessionConfig, EngineStepReport, GenerateRequest,
    GenerateResponse, GenerateRouteReport, GenerateSampling, GenerateStatus, GenerateStreamEvent,
    LlamaCppConfig, NativeModelArtifactsSource, NativeModelReport, NativeRuntimeArtifactsSource,
    NativeRuntimeReport, ResolutionPolicy, ResolvedBackend, ResolvedSessionConfigRequest,
    RuntimeReport, SelectedBackend, SessionRequestReport, SessionRequestState, SupportTier,
    current_host_report, current_metal_toolchain_report,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use std::time::Instant;

use crate::args::{
    ensure_output_root, has_flag, next_flag_value, parse_bool_list, parse_flag_value,
    parse_optional_u32_list, parse_u32_list, require_existing_dir, require_existing_file,
    require_existing_path, required_flag, required_string_flag,
};
use crate::artifact_files::{
    copy_optional_artifact_file, copy_required_artifact_file, create_unique_result_dir,
    reject_contract_failure_artifact_dir, sanitize_component, unix_timestamp_secs, write_json_file,
};
use crate::artifact_summary::{BenchmarkArtifactSummary, print_benchmark_artifact_summary};
use crate::cli::usage;
use crate::compare_json::{
    explicit_or_inferred_compare_mode, explicit_or_inferred_tool_mode,
    runtime_tuning_informational_diff_json, string_at_path_or_unknown,
};
use crate::determinism::deterministic_runtime_digest;
use crate::doctor::{
    build_doctor_report, build_doctor_report_for_model, metal_build_doctor_report,
    parse_doctor_args, render_doctor_report,
};
use crate::doctor_workflow::detect_doctor_workflow_report;
use crate::environment_probe::{
    bytes_to_gib, default_metal_driver, detect_kernel_release, detect_memory_bytes,
    detect_os_build, detect_os_version, detect_soc, detect_system_model, file_fingerprint_fnv1a64,
};
use crate::error::CliError;
use crate::generate_manifest::handle_generate_manifest;
use crate::inference_args::{InferenceArgs, build_inference_session, parse_inference_args};
use crate::inference_render::{render_generate_response, render_stream_event};
use crate::json_io::{
    json_string_label, json_value_label, load_json_value, load_optional_json_value, metric_number,
    nested_string, nested_value, validate_matching_json_field,
    validate_matching_optional_json_field,
};
use crate::labels::{
    compare_result_label, compare_summary_note, llama_cpp_final_request_state_label,
    llama_cpp_request_is_terminal, stop_reason_from_generate_finish_reason,
};
use crate::logging::init_tracing;
use crate::metal_build::{map_metal_build_error, metal_build_status_label, parse_metal_build_args};
use crate::path_utils::{expand_manifest_path_env, normalize_path_lexically};
use crate::route_json::{
    mlx_kv_cache_regression_markdown_lines, mlx_kv_cache_route_markdown_lines,
    route_count_from_json, route_counter_from_json,
    route_execution_semantics_from_environment_json, route_flag_from_json,
};
use crate::route_metadata::{
    backend_reported_cached_prompt_tokens, mlx_metal_coverage_ratio,
    mlx_metal_dispatch_share_label, prefix_reuse_provenance_label, route_decision_flag,
    route_decision_value, route_execution_semantics_label, route_metadata_from_generate_route,
    serialize_route_metadata,
};
use crate::route_readiness::{
    MlxMetalReadinessInputs, mlx_metal_readiness, mlx_metal_readiness_from_route_json,
    native_dense_dequantized_source_from_runtime_json,
    native_model_report_has_dense_dequantized_source,
};
use crate::stats::{
    elapsed_ms_since, percentage_delta, proportional_time_us, tokens_per_second_from_micros,
};
use crate::synthetic::{
    replay_prompt_target, synthetic_prompt_text, synthetic_prompt_tokens,
    synthetic_text_output_tokens,
};
use crate::token_sources::{output_token_count_source, prompt_token_count_source};

#[cfg(test)]
use crate::args::{unique_sorted_option_u32, unique_sorted_u32};
#[cfg(test)]
use crate::artifact_files::unique_run_suffix;
#[cfg(test)]
use crate::artifact_summary::render_benchmark_artifact_summary;
#[cfg(test)]
use crate::compare_json::inferred_tool_mode_from_runtime_json;
#[cfg(test)]
use crate::doctor::{DoctorAdviceSeverity, DoctorModelArtifactsStatus, DoctorStatus};
#[cfg(test)]
use crate::doctor_workflow::command_text;
#[cfg(test)]
use crate::doctor_workflow::{DoctorWorkflowMode, doctor_workflow_report_for_cwd};
#[cfg(test)]
use crate::generate_manifest::{
    GENERATE_MANIFEST_SCHEMA_VERSION, GenerateManifestStatus, GenerateManifestSummary,
    GenerateManifestValidationSummary, parse_generate_manifest_args,
};
#[cfg(test)]
use crate::labels::optional_u32_label;
#[cfg(test)]
use crate::labels::selected_backend_label;
#[cfg(test)]
use crate::path_utils::path_string;
#[cfg(test)]
use crate::route_json::route_execution_semantics_from_json;
#[cfg(test)]
use crate::route_metadata::{
    route_prefix_cpu_reference_dispatch_count, route_prefix_native_dispatch_count,
};
#[cfg(test)]
use crate::route_readiness::NATIVE_DENSE_DEQUANTIZED_SOURCE_BLOCKER;
#[cfg(test)]
use crate::stats::{percentile_f64, percentile_u64};
#[cfg(test)]
use ax_engine_sdk::{HostReport, MetalToolchainReport, ToolStatusReport};

mod commands;
#[cfg(test)]
mod tests;

fn main() -> ExitCode {
    init_tracing();

    match commands::run() {
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
