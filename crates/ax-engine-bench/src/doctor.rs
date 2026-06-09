use ax_engine_core::{
    MetalBuildDoctorReport, MetalBuildHostReport, MetalBuildToolStatus, MetalBuildToolchainReport,
};
use ax_engine_sdk::{HostReport, MetalToolchainReport, ToolStatusReport};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

use crate::cli::usage;
use crate::doctor_workflow::{DoctorWorkflowReport, command_text, workflow_mode_label};
use crate::error::CliError;
use crate::json_io::load_json_value;
use crate::path_utils::path_string;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DoctorArgs {
    pub(crate) json: bool,
    pub(crate) mlx_model_artifacts_dir: Option<PathBuf>,
}

pub(crate) fn parse_doctor_args(args: &[String]) -> Result<DoctorArgs, CliError> {
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DoctorStatus {
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

    fn human_label(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::BringupOnly => "bring-up only",
            Self::NotReady => "not ready",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DoctorReport {
    pub(crate) schema_version: String,
    pub(crate) mlx_target: String,
    pub(crate) status: DoctorStatus,
    pub(crate) mlx_runtime_ready: bool,
    pub(crate) bringup_allowed: bool,
    pub(crate) workflow: DoctorWorkflowReport,
    pub(crate) model_artifacts: DoctorModelArtifactsReport,
    pub(crate) host: HostReport,
    pub(crate) metal_toolchain: MetalToolchainReport,
    pub(crate) issues: Vec<String>,
    pub(crate) notes: Vec<String>,
    pub(crate) performance_advice: Vec<DoctorAdvice>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DoctorModelArtifactsStatus {
    NotSelected,
    Ready,
    NotReady,
}

impl DoctorModelArtifactsStatus {
    fn human_label(self) -> &'static str {
        match self {
            Self::NotSelected => "not selected",
            Self::Ready => "ready",
            Self::NotReady => "not ready",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DoctorModelArtifactsReport {
    pub(crate) selected: bool,
    pub(crate) status: DoctorModelArtifactsStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) path: Option<String>,
    pub(crate) exists: bool,
    pub(crate) is_dir: bool,
    pub(crate) config_present: bool,
    pub(crate) manifest_present: bool,
    pub(crate) safetensors_present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) quantization: Option<DoctorQuantizationHint>,
    pub(crate) issues: Vec<String>,
}

impl DoctorModelArtifactsReport {
    fn not_selected() -> Self {
        Self {
            selected: false,
            status: DoctorModelArtifactsStatus::NotSelected,
            path: None,
            exists: false,
            is_dir: false,
            config_present: false,
            manifest_present: false,
            safetensors_present: false,
            model_type: None,
            quantization: None,
            issues: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DoctorAdviceSeverity {
    Info,
    Warning,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DoctorAdvice {
    pub(crate) id: String,
    pub(crate) severity: DoctorAdviceSeverity,
    pub(crate) summary: String,
    pub(crate) detail: String,
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

pub(crate) fn metal_build_doctor_report(report: &DoctorReport) -> MetalBuildDoctorReport {
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

pub(crate) fn build_doctor_report(
    host: HostReport,
    metal_toolchain: MetalToolchainReport,
) -> DoctorReport {
    build_doctor_report_with_mlx_model_artifacts(host, metal_toolchain, None)
}

pub(crate) fn build_doctor_report_for_model(
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
        mlx_target: "apple_m2_or_newer_macos_aarch64".to_string(),
        status,
        mlx_runtime_ready,
        bringup_allowed,
        workflow: DoctorWorkflowReport::unknown(),
        model_artifacts: doctor_model_artifacts_report(mlx_model_artifacts_dir),
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
            "AX Engine MLX Metal runtime requires macOS/aarch64 on Apple M2 Max or newer with 32 GB RAM minimum; detected {detected_host}"
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

fn doctor_model_artifacts_report(
    mlx_model_artifacts_dir: Option<&Path>,
) -> DoctorModelArtifactsReport {
    let Some(path) = mlx_model_artifacts_dir else {
        return DoctorModelArtifactsReport::not_selected();
    };

    let exists = path.exists();
    let is_dir = path.is_dir();
    let config_path = path.join("config.json");
    let manifest_path = path.join("model-manifest.json");
    let config_present = config_path.is_file();
    let manifest_present = manifest_path.is_file();
    let mut issues = Vec::new();
    let mut model_type = None;
    let mut quantization = None;
    let mut safetensors_present = false;

    if !exists {
        issues.push(format!(
            "model artifacts path does not exist: {}",
            path.display()
        ));
    } else if !is_dir {
        issues.push(format!(
            "model artifacts path is not a directory: {}",
            path.display()
        ));
    } else {
        if !config_present {
            issues.push("missing config.json".to_string());
        }
        if !manifest_present {
            issues.push("missing model-manifest.json".to_string());
        }

        match dir_contains_safetensors(path) {
            Ok(present) => {
                safetensors_present = present;
                if !present {
                    issues.push("missing safetensors file".to_string());
                }
            }
            Err(message) => issues.push(message),
        }

        if config_present {
            match load_json_value(&config_path) {
                Ok(config) => {
                    model_type = doctor_config_string(&config, "model_type").map(str::to_string);
                    quantization = doctor_config_quantization(&config);
                    if model_type.is_none() {
                        issues.push("missing model_type in config.json".to_string());
                    }
                }
                Err(error) => issues.push(format!("config.json is not readable JSON: {error}")),
            }
        }
    }

    let status = if issues.is_empty() {
        DoctorModelArtifactsStatus::Ready
    } else {
        DoctorModelArtifactsStatus::NotReady
    };

    DoctorModelArtifactsReport {
        selected: true,
        status,
        path: Some(path_string(path)),
        exists,
        is_dir,
        config_present,
        manifest_present,
        safetensors_present,
        model_type,
        quantization,
        issues,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DoctorModelArtifactsHint {
    model_type: Option<String>,
    quantization: Option<DoctorQuantizationHint>,
    path_label: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DoctorQuantizationHint {
    pub(crate) mode: String,
    pub(crate) group_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bits: Option<u32>,
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

fn dir_contains_safetensors(path: &Path) -> Result<bool, String> {
    let entries = fs::read_dir(path).map_err(|error| {
        format!(
            "failed to read model artifacts directory {}: {error}",
            path.display()
        )
    })?;
    Ok(entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
    }))
}

fn doctor_model_performance_advice(hint: &DoctorModelArtifactsHint) -> Vec<DoctorAdvice> {
    let mut advice = Vec::new();
    let model_type = hint.model_type.as_deref().unwrap_or("unknown");
    let quantization = hint.quantization.as_ref();

    match model_type {
        "gemma4" => {
            if quantization.and_then(|q| q.bits) == Some(4) {
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
            if quantization.and_then(|q| q.bits) == Some(4) {
                advice.push(DoctorAdvice::warning(
                    "qwen36_quantization_compare",
                    "Do not assume Qwen 3.6 4-bit is the fastest checkpoint.",
                    "Current Qwen 3.6 comparison coverage keeps 35B A3B to 4-bit and sweeps 27B at 4/5/6/8-bit; compare the target checkpoint on the target prompt mix.",
                ));
            } else if quantization.and_then(|q| q.bits) == Some(5) {
                advice.push(DoctorAdvice::info(
                    "qwen36_5bit_throughput_candidate",
                    "Qwen 3.6 5-bit is a strong throughput candidate.",
                    "Current Qwen 3.6 27B sweep coverage includes 5-bit, but memory pressure and quality targets still need workload-specific validation.",
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
    } else if quantization.and_then(|q| q.bits).is_none() {
        advice.push(DoctorAdvice::info(
            "quantization_bits_missing",
            "Quantization metadata did not include a bits field.",
            "Doctor will not infer 4-bit or 5-bit policy advice without explicit quantization bits.",
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
        bits: obj
            .get("bits")
            .and_then(Value::as_u64)
            .map(|bits| bits.min(u64::from(u32::MAX)) as u32),
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

fn tool_version_summary(tool: &ToolStatusReport) -> &str {
    tool.version
        .as_deref()
        .and_then(|version| version.lines().next())
        .unwrap_or("unknown")
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn ready_not_ready(value: bool) -> &'static str {
    if value { "ready" } else { "not ready" }
}

fn available_missing(value: bool) -> &'static str {
    if value { "available" } else { "missing" }
}

fn render_bullets(lines: &mut Vec<String>, items: &[String]) {
    if items.is_empty() {
        lines.push("  - none".to_string());
    } else {
        lines.extend(items.iter().map(|item| format!("  - {item}")));
    }
}

fn render_advice_group(
    lines: &mut Vec<String>,
    title: &str,
    advice: &[DoctorAdvice],
    severity: DoctorAdviceSeverity,
) {
    let matching: Vec<&DoctorAdvice> = advice
        .iter()
        .filter(|item| item.severity == severity)
        .collect();
    if matching.is_empty() {
        return;
    }

    lines.push(format!("{title}:"));
    for item in matching {
        lines.push(format!("  - {}: {}", item.id, item.summary));
        lines.push(format!("    {}", item.detail));
    }
}

pub(crate) fn render_doctor_report(report: &DoctorReport) -> String {
    let mut lines = vec![
        "AX Engine v6 doctor".to_string(),
        format!("Status: {}", report.status.human_label()),
        format!("Schema: {}", report.schema_version),
        String::new(),
        "Summary:".to_string(),
        format!(
            "  - MLX runtime: {}",
            ready_not_ready(report.mlx_runtime_ready)
        ),
        format!("  - Bring-up allowed: {}", yes_no(report.bringup_allowed)),
        format!("  - Target: {}", report.mlx_target),
        format!(
            "  - Host: {} ({}/{})",
            report.host.detected_soc.as_deref().unwrap_or("unknown"),
            report.host.os,
            report.host.arch
        ),
        format!(
            "  - Metal toolchain: {}",
            ready_not_ready(report.metal_toolchain.fully_available)
        ),
        String::new(),
        "Workflow:".to_string(),
        format!(
            "  - Mode: {}",
            workflow_mode_label(report.workflow.mode).replace('_', " ")
        ),
        format!("  - Current directory: {}", report.workflow.cwd),
        format!(
            "  - Source checkout: {}",
            report.workflow.source_root.as_deref().unwrap_or("none")
        ),
        format!(
            "  - Machine-readable doctor: {}",
            command_text(&report.workflow.doctor)
        ),
        format!("  - Server: {}", command_text(&report.workflow.server)),
        format!(
            "  - Generate manifest: {}",
            command_text(&report.workflow.generate_manifest)
        ),
        format!(
            "  - Benchmark: {}",
            command_text(&report.workflow.benchmark)
        ),
        format!(
            "  - Download model: {}",
            report
                .workflow
                .download_model
                .as_ref()
                .map(command_text)
                .unwrap_or_else(|| "none".to_string())
        ),
        String::new(),
        "Model artifacts:".to_string(),
        format!(
            "  - Status: {}",
            report.model_artifacts.status.human_label()
        ),
        format!(
            "  - Path: {}",
            report.model_artifacts.path.as_deref().unwrap_or("none")
        ),
        format!(
            "  - config.json: {}",
            yes_no(report.model_artifacts.config_present)
        ),
        format!(
            "  - model-manifest.json: {}",
            yes_no(report.model_artifacts.manifest_present)
        ),
        format!(
            "  - safetensors: {}",
            yes_no(report.model_artifacts.safetensors_present)
        ),
        format!(
            "  - Model type: {}",
            report
                .model_artifacts
                .model_type
                .as_deref()
                .unwrap_or("unknown")
        ),
    ];

    if !report.model_artifacts.selected {
        lines.push(
            "  - Next: pass --mlx-model-artifacts-dir <model-dir> for model-specific checks"
                .to_string(),
        );
    }

    lines.extend([
        String::new(),
        "Host:".to_string(),
        format!(
            "  - Supported MLX runtime host: {}",
            yes_no(report.host.supported_mlx_runtime)
        ),
        format!(
            "  - Unsupported-host override active: {}",
            yes_no(report.host.unsupported_host_override_active)
        ),
        String::new(),
        "Metal toolchain:".to_string(),
        format!(
            "  - metal: {} - {}",
            available_missing(report.metal_toolchain.metal.available),
            tool_version_summary(&report.metal_toolchain.metal)
        ),
        format!(
            "  - metallib: {} - {}",
            available_missing(report.metal_toolchain.metallib.available),
            tool_version_summary(&report.metal_toolchain.metallib)
        ),
        format!(
            "  - metal-ar: {} - {}",
            available_missing(report.metal_toolchain.metal_ar.available),
            tool_version_summary(&report.metal_toolchain.metal_ar)
        ),
        String::new(),
        "Issues:".to_string(),
    ]);

    render_bullets(&mut lines, &report.issues);

    lines.push(String::new());
    lines.push("Model artifact issues:".to_string());
    render_bullets(&mut lines, &report.model_artifacts.issues);

    lines.push(String::new());
    lines.push("Notes:".to_string());
    render_bullets(&mut lines, &report.notes);

    lines.push(String::new());
    lines.push("Performance advice:".to_string());
    render_advice_group(
        &mut lines,
        "Warnings",
        &report.performance_advice,
        DoctorAdviceSeverity::Warning,
    );
    render_advice_group(
        &mut lines,
        "Info",
        &report.performance_advice,
        DoctorAdviceSeverity::Info,
    );

    lines.join("\n")
}
