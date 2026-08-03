use ax_engine_core::{
    MetalBuildDoctorReport, MetalBuildHostReport, MetalBuildToolStatus, MetalBuildToolchainReport,
    MetalKernelAssets, sha256_hex,
};
use ax_engine_sdk::{HostReport, MetalToolchainReport, ToolStatusReport};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct DoctorReport {
    pub(crate) schema_version: String,
    pub(crate) mlx_target: String,
    pub(crate) status: DoctorStatus,
    pub(crate) mlx_runtime_ready: bool,
    pub(crate) bringup_allowed: bool,
    pub(crate) workflow: DoctorWorkflowReport,
    pub(crate) runtime_assets: DoctorRuntimeAssetsReport,
    pub(crate) model_artifacts: DoctorModelArtifactsReport,
    pub(crate) host: HostReport,
    pub(crate) metal_toolchain: MetalToolchainReport,
    pub(crate) issues: Vec<String>,
    pub(crate) notes: Vec<String>,
    pub(crate) performance_advice: Vec<DoctorAdvice>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DoctorRuntimeAssetsStatus {
    NotFound,
    Ready,
    NotReady,
}

impl DoctorRuntimeAssetsStatus {
    fn human_label(self) -> &'static str {
        match self {
            Self::NotFound => "not found",
            Self::Ready => "ready",
            Self::NotReady => "not ready",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DoctorRuntimeAssetsReport {
    pub(crate) status: DoctorRuntimeAssetsStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) issue: Option<String>,
}

impl DoctorRuntimeAssetsReport {
    fn not_found() -> Self {
        Self {
            status: DoctorRuntimeAssetsStatus::NotFound,
            path: None,
            source: None,
            issue: None,
        }
    }

    fn is_ready(&self) -> bool {
        self.status == DoctorRuntimeAssetsStatus::Ready
    }
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) axquant: Option<DoctorAxquantReport>,
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
            axquant: None,
            issues: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct DoctorAxquantReport {
    pub(crate) metadata_valid: bool,
    pub(crate) lineage_valid: bool,
    pub(crate) release_quality_evidence: bool,
    pub(crate) provenance_complete: bool,
    pub(crate) manifest_schema_version: Option<String>,
    pub(crate) evidence_kind: Option<String>,
    pub(crate) source_model_id: Option<String>,
    pub(crate) source_revision: Option<String>,
    pub(crate) plan_sha256: Option<String>,
    pub(crate) plan_present: bool,
    pub(crate) quantizer_execution_present: bool,
    pub(crate) runtime_metadata_present: bool,
    pub(crate) target_bpw: Option<f64>,
    pub(crate) effective_bpw: Option<f64>,
    pub(crate) measured_total_bpw: Option<f64>,
    pub(crate) measured_main_bpw: Option<f64>,
    pub(crate) precision_bits: Vec<u32>,
    pub(crate) mixed_precision: bool,
    pub(crate) quantized_module_count: usize,
    pub(crate) failed_module_count: usize,
    pub(crate) fallback_module_count: usize,
    pub(crate) issues: Vec<String>,
    /// Non-fatal concerns (e.g. bit widths the runtime cannot load
    /// unconditionally). Warnings never affect `metadata_valid`.
    pub(crate) warnings: Vec<String>,
}

impl DoctorAxquantReport {
    fn empty() -> Self {
        Self {
            metadata_valid: false,
            lineage_valid: false,
            release_quality_evidence: false,
            provenance_complete: false,
            manifest_schema_version: None,
            evidence_kind: None,
            source_model_id: None,
            source_revision: None,
            plan_sha256: None,
            plan_present: false,
            quantizer_execution_present: false,
            runtime_metadata_present: false,
            target_bpw: None,
            effective_bpw: None,
            measured_total_bpw: None,
            measured_main_bpw: None,
            precision_bits: Vec::new(),
            mixed_precision: false,
            quantized_module_count: 0,
            failed_module_count: 0,
            fallback_module_count: 0,
            issues: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AxquantModuleSpec {
    bits: u32,
    method: String,
    group_size: Option<u64>,
}

/// Map an AXQuant assignment bit width to a runtime-support warning when the
/// runtime cannot load it unconditionally. BF16 (16-bit) tensors and the
/// supported affine set load without any gate, so they produce no warning.
fn axquant_bits_runtime_support_warning(context: &str, bits: u32) -> Option<String> {
    if ax_engine_core::SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS.contains(&bits) || bits == 16 {
        return None;
    }
    let detail = if ax_engine_core::EXPERIMENTAL_MLX_AFFINE_QUANTIZATION_BITS.contains(&bits) {
        format!(
            "the runtime requires {}=1 to load it",
            ax_engine_core::AX_ENGINE_3BIT_EXPERIMENTAL_ENV
        )
    } else if ax_engine_core::EXPERIMENTAL_2BIT_MLX_AFFINE_QUANTIZATION_BITS.contains(&bits) {
        format!(
            "the runtime requires {}=1 to load it",
            ax_engine_core::AX_ENGINE_2BIT_EXPERIMENTAL_ENV
        )
    } else {
        format!(
            "the runtime cannot load it (supported affine bits: {:?})",
            ax_engine_core::SUPPORTED_MLX_AFFINE_QUANTIZATION_BITS
        )
    };
    Some(format!(
        "{context} uses {bits}-bit affine quantization; {detail}"
    ))
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
    build_doctor_report_with_runtime_assets(
        host,
        metal_toolchain,
        DoctorRuntimeAssetsReport::not_found(),
        None,
    )
}

#[cfg(test)]
pub(crate) fn build_doctor_report_for_model(
    host: HostReport,
    metal_toolchain: MetalToolchainReport,
    mlx_model_artifacts_dir: Option<&Path>,
) -> DoctorReport {
    build_doctor_report_with_runtime_assets(
        host,
        metal_toolchain,
        DoctorRuntimeAssetsReport::not_found(),
        mlx_model_artifacts_dir,
    )
}

pub(crate) fn build_doctor_report_for_model_and_runtime(
    host: HostReport,
    metal_toolchain: MetalToolchainReport,
    runtime_assets: DoctorRuntimeAssetsReport,
    mlx_model_artifacts_dir: Option<&Path>,
) -> DoctorReport {
    build_doctor_report_with_runtime_assets(
        host,
        metal_toolchain,
        runtime_assets,
        mlx_model_artifacts_dir,
    )
}

fn build_doctor_report_with_runtime_assets(
    host: HostReport,
    metal_toolchain: MetalToolchainReport,
    runtime_assets: DoctorRuntimeAssetsReport,
    mlx_model_artifacts_dir: Option<&Path>,
) -> DoctorReport {
    let runtime_available = runtime_assets.is_ready() || metal_toolchain.fully_available;
    let mlx_runtime_ready = host.supported_mlx_runtime && runtime_available;
    let bringup_allowed =
        runtime_available && (host.supported_mlx_runtime || host.unsupported_host_override_active);
    let status = if mlx_runtime_ready {
        DoctorStatus::Ready
    } else if bringup_allowed {
        DoctorStatus::BringupOnly
    } else {
        DoctorStatus::NotReady
    };
    let model_artifacts = doctor_model_artifacts_report(mlx_model_artifacts_dir);
    let performance_advice = doctor_performance_advice(&host, &model_artifacts);

    DoctorReport {
        schema_version: "ax.engine_bench.doctor.v1".to_string(),
        mlx_target: "apple_m2_or_newer_macos_aarch64".to_string(),
        status,
        mlx_runtime_ready,
        bringup_allowed,
        workflow: DoctorWorkflowReport::unknown(),
        runtime_assets: runtime_assets.clone(),
        model_artifacts,
        host: host.clone(),
        metal_toolchain: metal_toolchain.clone(),
        issues: doctor_issues(&host, &metal_toolchain, &runtime_assets),
        notes: doctor_notes(&host, &metal_toolchain, &runtime_assets),
        performance_advice,
    }
}

pub(crate) fn detect_runtime_assets_report(
    current_dir: Option<&Path>,
) -> DoctorRuntimeAssetsReport {
    if let Some(path) = env::var_os("AX_ENGINE_METAL_BUILD_DIR").map(PathBuf::from) {
        return runtime_assets_report_for_dir(&path, "explicit_env");
    }

    current_dir
        .and_then(detect_repo_runtime_assets_from)
        .unwrap_or_else(DoctorRuntimeAssetsReport::not_found)
}

fn detect_repo_runtime_assets_from(start_dir: &Path) -> Option<DoctorRuntimeAssetsReport> {
    // Mirror ax-engine-sdk's `detect_repo_owned_mlx_runtime_artifacts_dir_from`:
    // keep walking ancestors past a candidate that doesn't fully validate,
    // since a nested checkout (e.g. a bundled wheel's asset dir) can have a
    // partial match (manifest without a build report) below the real repo
    // root that does validate. Only stop early on a `Ready` match; otherwise
    // remember the first diagnostic to report if nothing up the chain works.
    let mut first_diagnostic: Option<DoctorRuntimeAssetsReport> = None;
    for candidate_root in start_dir.ancestors().take(20) {
        let manifest_path = candidate_root.join("metal/phase1-kernels.json");
        let build_dir = candidate_root.join("build/metal");
        let build_report_path = build_dir.join("build_report.json");

        if !manifest_path.is_file() && !build_report_path.is_file() {
            continue;
        }

        if !build_report_path.is_file() {
            first_diagnostic.get_or_insert(DoctorRuntimeAssetsReport {
                status: DoctorRuntimeAssetsStatus::NotFound,
                path: Some(path_string(&build_dir)),
                source: Some("repo_auto_detect".to_string()),
                issue: Some("build/metal/build_report.json is missing".to_string()),
            });
            continue;
        }

        let report = runtime_assets_report_for_dir(&build_dir, "repo_auto_detect");
        if report.status == DoctorRuntimeAssetsStatus::Ready {
            return Some(report);
        }
        first_diagnostic.get_or_insert(report);
    }

    first_diagnostic
}

fn runtime_assets_report_for_dir(path: &Path, source: &str) -> DoctorRuntimeAssetsReport {
    match MetalKernelAssets::from_build_dir(path) {
        Ok(_) => DoctorRuntimeAssetsReport {
            status: DoctorRuntimeAssetsStatus::Ready,
            path: Some(path_string(path)),
            source: Some(source.to_string()),
            issue: None,
        },
        Err(error) => DoctorRuntimeAssetsReport {
            status: DoctorRuntimeAssetsStatus::NotReady,
            path: Some(path_string(path)),
            source: Some(source.to_string()),
            issue: Some(error.to_string()),
        },
    }
}

fn doctor_issues(
    host: &HostReport,
    metal_toolchain: &MetalToolchainReport,
    runtime_assets: &DoctorRuntimeAssetsReport,
) -> Vec<String> {
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
    if !runtime_assets.is_ready() && !missing_tools.is_empty() {
        issues.push(format!(
            "Runtime assets are not ready and Metal toolchain is incomplete; missing {}",
            missing_tools.join(", ")
        ));
    }
    if runtime_assets.status == DoctorRuntimeAssetsStatus::NotReady
        && !metal_toolchain.fully_available
    {
        let path = runtime_assets.path.as_deref().unwrap_or("unknown");
        let issue = runtime_assets.issue.as_deref().unwrap_or("unknown error");
        issues.push(format!("Runtime assets are not ready at {path}: {issue}"));
    }

    issues
}

fn doctor_notes(
    host: &HostReport,
    metal_toolchain: &MetalToolchainReport,
    runtime_assets: &DoctorRuntimeAssetsReport,
) -> Vec<String> {
    let mut notes = vec!["llama.cpp backends do not widen supported host scope".to_string()];
    if host.unsupported_host_override_active {
        notes.push(
            "AX_ALLOW_UNSUPPORTED_HOST only unlocks development or CI bring-up and does not make benchmark or runtime results supported"
                .to_string(),
        );
    }
    let missing_tools = missing_metal_tools(metal_toolchain);
    if runtime_assets.is_ready() && !missing_tools.is_empty() {
        notes.push(format!(
            "Developer Metal toolchain is incomplete; missing {}. Bundled runtime assets are ready, so this only blocks rebuilding Metal kernels.",
            missing_tools.join(", ")
        ));
    }
    if runtime_assets.status == DoctorRuntimeAssetsStatus::NotReady
        && metal_toolchain.fully_available
    {
        let path = runtime_assets.path.as_deref().unwrap_or("unknown");
        notes.push(format!(
            "Runtime assets at {path} are stale or invalid, but the Metal toolchain is available; run ax-engine-bench metal-build before relying on repo-owned compiled kernel assets."
        ));
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
    let mut axquant = None;
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
        } else if let Err(error) = ax_engine_core::NativeModelArtifacts::from_dir(path) {
            issues.push(format!("native model artifacts are invalid: {error}"));
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

        axquant = doctor_axquant_report(path);
        if let Some(report) = &axquant {
            issues.extend(
                report
                    .issues
                    .iter()
                    .map(|issue| format!("AXQuant metadata: {issue}")),
            );
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
        axquant,
        issues,
    }
}

fn doctor_axquant_report(path: &Path) -> Option<DoctorAxquantReport> {
    let manifest_path = path.join("axquant_manifest.json");
    if !manifest_path.is_file() {
        return None;
    }

    let mut report = DoctorAxquantReport::empty();
    let manifest = match load_json_value(&manifest_path) {
        Ok(value) => value,
        Err(error) => {
            report.issues.push(format!(
                "axquant_manifest.json is not readable JSON: {error}"
            ));
            return Some(report);
        }
    };

    report.manifest_schema_version = json_string(&manifest, "schema_version");
    if report.manifest_schema_version.as_deref() != Some("axquant.artifact.v2") {
        report.issues.push(format!(
            "unsupported artifact schema {}",
            report
                .manifest_schema_version
                .as_deref()
                .unwrap_or("missing")
        ));
    }
    if json_string(&manifest, "format").as_deref() != Some("mlx") {
        report
            .issues
            .push("AXQuant artifact format must be mlx".to_string());
    }
    if json_string(&manifest, "quantizer").as_deref() != Some("axquant") {
        report
            .issues
            .push("AXQuant artifact quantizer must be axquant".to_string());
    }

    report.measured_total_bpw = json_positive_f64(&manifest, "measured_total_bpw");
    report.measured_main_bpw = json_positive_f64(&manifest, "measured_main_bpw");
    if report.measured_total_bpw.is_none() {
        report
            .issues
            .push("manifest is missing a positive measured_total_bpw".to_string());
    }
    if report.measured_main_bpw.is_none() {
        report
            .issues
            .push("manifest is missing a positive measured_main_bpw".to_string());
    }
    let manifest_effective_bpw = json_positive_f64(&manifest, "effective_bpw");
    if manifest_effective_bpw.is_none() {
        report
            .issues
            .push("manifest is missing a positive effective_bpw".to_string());
    }

    report.source_model_id = manifest
        .get("source_model")
        .and_then(|value| json_string(value, "model_id"));
    report.source_revision = manifest
        .get("source_model")
        .and_then(|value| json_string(value, "revision"));
    report.provenance_complete = report
        .source_model_id
        .as_deref()
        .is_some_and(|value| !value.is_empty())
        && report
            .source_revision
            .as_deref()
            .is_some_and(|value| !value.is_empty());

    report.plan_sha256 = json_string(&manifest, "plan_sha256");
    if !report.plan_sha256.as_deref().is_some_and(is_sha256_hex) {
        report
            .issues
            .push("manifest plan_sha256 is missing or invalid".to_string());
    }

    let plan_path = path.join("axquant_plan.json");
    report.plan_present = plan_path.is_file();
    let plan = if report.plan_present {
        validate_axquant_metadata_file(path, "axquant_plan.json", &manifest, &mut report.issues);
        match load_json_value(&plan_path) {
            Ok(value) => Some(value),
            Err(error) => {
                report
                    .issues
                    .push(format!("axquant_plan.json is not readable JSON: {error}"));
                None
            }
        }
    } else {
        report.issues.push("missing axquant_plan.json".to_string());
        None
    };

    let mut plan_digest_valid = false;
    let mut planned_quantized_modules = BTreeMap::new();
    if let Some(plan) = &plan {
        if json_string(plan, "schema_version").as_deref() != Some("axquant.plan.v1") {
            report
                .issues
                .push("unsupported or missing AXQuant plan schema".to_string());
        }
        if json_string(plan, "quantizer").as_deref() != Some("axquant") {
            report
                .issues
                .push("AXQuant plan quantizer must be axquant".to_string());
        }
        report.evidence_kind = json_string(plan, "evidence_kind");
        match report.evidence_kind.as_deref() {
            Some("measured" | "imported") => report.release_quality_evidence = true,
            Some("measured_development" | "architecture_prior") => {}
            Some(value) => report
                .issues
                .push(format!("unsupported AXQuant evidence_kind {value}")),
            None => report
                .issues
                .push("AXQuant plan is missing evidence_kind".to_string()),
        }
        report.target_bpw = json_positive_f64(plan, "target_bpw");
        report.effective_bpw = json_positive_f64(plan, "effective_bpw");
        if report.target_bpw.is_none() || report.effective_bpw.is_none() {
            report
                .issues
                .push("AXQuant plan is missing positive BPW values".to_string());
        }
        if let (Some(manifest_bpw), Some(plan_bpw)) = (manifest_effective_bpw, report.effective_bpw)
            && manifest_bpw != plan_bpw
        {
            report
                .issues
                .push("artifact and plan effective_bpw values differ".to_string());
        }

        match axquant_stable_sha256(plan) {
            Some(actual_sha256) => {
                plan_digest_valid = report.plan_sha256.as_deref() == Some(&actual_sha256);
                if !plan_digest_valid {
                    report.issues.push(format!(
                        "AXQuant plan content digest {actual_sha256} does not match manifest plan_sha256 {}",
                        report.plan_sha256.as_deref().unwrap_or("missing")
                    ));
                }
            }
            None => report
                .issues
                .push("AXQuant plan cannot be canonicalized for lineage validation".to_string()),
        }

        let mut precision_bits = BTreeSet::new();
        let mut unsupported_plan_bits = BTreeSet::new();
        if let Some(assignments) = plan.get("assignments").and_then(Value::as_array) {
            let mut invalid_assignments = 0_usize;
            let mut duplicate_modules = 0_usize;
            let mut seen_modules = BTreeSet::new();
            for assignment in assignments {
                let Some(bits) = assignment
                    .get("bits")
                    .and_then(Value::as_u64)
                    .and_then(|bits| u32::try_from(bits).ok())
                    .filter(|bits| (2..=16).contains(bits))
                else {
                    invalid_assignments += 1;
                    continue;
                };
                precision_bits.insert(bits);
                unsupported_plan_bits.insert(bits);

                let module_path = assignment
                    .get("module_path")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty());
                let method = assignment
                    .get("method")
                    .and_then(Value::as_str)
                    .filter(|value| matches!(*value, "affine" | "awq" | "dwq" | "gptq" | "bf16"));
                let group_size = json_optional_positive_u64(assignment, "group_size");
                let (Some(module_path), Some(method), Some(group_size)) =
                    (module_path, method, group_size)
                else {
                    invalid_assignments += 1;
                    continue;
                };
                if (bits < 16 && group_size.is_none())
                    || (bits == 16 && group_size.is_some())
                    || (bits < 16 && method == "bf16")
                    || (bits == 16 && method != "bf16")
                {
                    invalid_assignments += 1;
                    continue;
                }
                if !seen_modules.insert(module_path) {
                    duplicate_modules += 1;
                }

                if bits < 16 {
                    planned_quantized_modules.insert(
                        module_path.to_string(),
                        AxquantModuleSpec {
                            bits,
                            method: method.to_string(),
                            group_size,
                        },
                    );
                }
            }
            if invalid_assignments > 0 {
                report.issues.push(format!(
                    "AXQuant plan has {invalid_assignments} assignments with invalid module, method, group-size, or bit metadata"
                ));
            }
            if duplicate_modules > 0 {
                report.issues.push(format!(
                    "AXQuant plan has {duplicate_modules} duplicate module assignments"
                ));
            }
        } else {
            report
                .issues
                .push("AXQuant plan is missing assignments".to_string());
        }
        report.precision_bits = precision_bits.into_iter().collect();
        report.mixed_precision = report.precision_bits.len() > 1;
        report.warnings.extend(
            unsupported_plan_bits
                .into_iter()
                .filter_map(|bits| axquant_bits_runtime_support_warning("AXQuant plan", bits)),
        );
        if report.precision_bits.is_empty() {
            report
                .issues
                .push("AXQuant plan has no precision assignments".to_string());
        }
        if planned_quantized_modules.is_empty() {
            report
                .issues
                .push("AXQuant plan has no quantized module assignments".to_string());
        }

        let plan_source = plan.get("source_model");
        let plan_model_id = plan_source.and_then(|value| json_string(value, "model_id"));
        let plan_revision = plan_source.and_then(|value| json_string(value, "revision"));
        if plan_model_id != report.source_model_id || plan_revision != report.source_revision {
            report
                .issues
                .push("artifact and plan source-model provenance differ".to_string());
        }
    }

    let execution_path = path.join("axquant_quantizer_execution.json");
    report.quantizer_execution_present = execution_path.is_file();
    if report.quantizer_execution_present {
        validate_axquant_metadata_file(
            path,
            "axquant_quantizer_execution.json",
            &manifest,
            &mut report.issues,
        );
        match load_json_value(&execution_path) {
            Ok(execution) => {
                if json_string(&execution, "schema_version").as_deref()
                    != Some("axquant.quantizer-execution.v1")
                {
                    report
                        .issues
                        .push("unsupported or missing quantizer-execution schema".to_string());
                }
                let execution_plan_sha256 = json_string(&execution, "plan_sha256");
                let execution_digest_matches =
                    execution_plan_sha256.is_some() && execution_plan_sha256 == report.plan_sha256;
                report.lineage_valid = plan_digest_valid && execution_digest_matches;
                if !execution_digest_matches {
                    report.issues.push(
                        "artifact and quantizer execution bind different plan digests".to_string(),
                    );
                }
                if let Some(records) = execution.get("records").and_then(Value::as_array) {
                    report.quantized_module_count = records.len();
                    report.failed_module_count = records
                        .iter()
                        .filter(|record| {
                            record.get("success").and_then(Value::as_bool) != Some(true)
                        })
                        .count();
                    report.fallback_module_count = records
                        .iter()
                        .filter(|record| {
                            record.get("fallback").and_then(Value::as_bool) == Some(true)
                        })
                        .count();
                    if records.is_empty() {
                        report
                            .issues
                            .push("quantizer execution contains no module records".to_string());
                    }
                    if report.failed_module_count > 0 {
                        report.issues.push(format!(
                            "quantizer execution has {} failed module records",
                            report.failed_module_count
                        ));
                    }
                    if report.fallback_module_count > 0 {
                        report.issues.push(format!(
                            "quantizer execution has {} fallback module records",
                            report.fallback_module_count
                        ));
                    }
                    validate_axquant_execution_coverage(
                        records,
                        &planned_quantized_modules,
                        &mut report.issues,
                        &mut report.warnings,
                    );
                } else {
                    report
                        .issues
                        .push("quantizer execution is missing records".to_string());
                }
            }
            Err(error) => report.issues.push(format!(
                "axquant_quantizer_execution.json is not readable JSON: {error}"
            )),
        }
    } else {
        report
            .issues
            .push("missing axquant_quantizer_execution.json".to_string());
    }

    let runtime_path = path.join("axquant_runtime.json");
    report.runtime_metadata_present = runtime_path.is_file();
    if report.runtime_metadata_present {
        validate_axquant_metadata_file(path, "axquant_runtime.json", &manifest, &mut report.issues);
        match load_json_value(&runtime_path) {
            Ok(runtime) => {
                if json_string(&runtime, "schema_version").as_deref() != Some("axquant.runtime.v1")
                {
                    report
                        .issues
                        .push("unsupported or missing AXQuant runtime schema".to_string());
                }
                if manifest.get("runtime") != Some(&runtime) {
                    report
                        .issues
                        .push("artifact and axquant_runtime.json metadata differ".to_string());
                }
            }
            Err(error) => report.issues.push(format!(
                "axquant_runtime.json is not readable JSON: {error}"
            )),
        }
    } else {
        report
            .issues
            .push("missing axquant_runtime.json".to_string());
    }

    report.metadata_valid = report.issues.is_empty();
    Some(report)
}

fn validate_axquant_metadata_file(
    root: &Path,
    file_name: &str,
    manifest: &Value,
    issues: &mut Vec<String>,
) {
    let Some(files) = manifest.get("files").and_then(Value::as_array) else {
        issues.push("manifest is missing file bindings".to_string());
        return;
    };
    let mut matching_entries = files
        .iter()
        .filter(|entry| entry.get("path").and_then(Value::as_str) == Some(file_name));
    let entry = matching_entries.next();
    if matching_entries.next().is_some() {
        issues.push(format!("manifest binds {file_name} more than once"));
        return;
    }
    let recorded_sha256 = entry
        .and_then(|entry| entry.get("sha256"))
        .and_then(Value::as_str);
    let Some(recorded_sha256) = recorded_sha256 else {
        issues.push(format!("manifest does not bind {file_name}"));
        return;
    };
    if !is_sha256_hex(recorded_sha256) {
        issues.push(format!("manifest has an invalid SHA-256 for {file_name}"));
        return;
    }

    let recorded_size = entry
        .and_then(|entry| entry.get("size_bytes"))
        .and_then(Value::as_u64);
    if recorded_size.is_none() {
        issues.push(format!("manifest has no valid size for {file_name}"));
    }

    match fs::read(root.join(file_name)) {
        Ok(bytes) => {
            if recorded_size != u64::try_from(bytes.len()).ok() {
                issues.push(format!("{file_name} size differs from the manifest"));
            }
            if sha256_hex(&bytes) != recorded_sha256 {
                issues.push(format!("{file_name} SHA-256 differs from the manifest"));
            }
        }
        Err(error) => issues.push(format!("failed to read {file_name}: {error}")),
    }
}

fn validate_axquant_execution_coverage(
    records: &[Value],
    planned: &BTreeMap<String, AxquantModuleSpec>,
    issues: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    let mut seen = BTreeSet::new();
    let mut unsupported_bits = BTreeSet::new();
    let mut malformed = 0_usize;
    let mut duplicates = 0_usize;
    let mut unexpected = 0_usize;
    let mut mismatched = 0_usize;

    for record in records {
        let module_path = record
            .get("module_path")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        let bits = record
            .get("bits")
            .and_then(Value::as_u64)
            .and_then(|bits| u32::try_from(bits).ok())
            .filter(|bits| (2..=16).contains(bits));
        let method = record
            .get("method")
            .and_then(Value::as_str)
            .filter(|value| matches!(*value, "affine" | "awq" | "dwq" | "gptq" | "bf16"));
        let group_size = json_optional_positive_u64(record, "group_size");
        let success = record.get("success").and_then(Value::as_bool);
        let fallback = record.get("fallback").and_then(Value::as_bool);
        let (Some(module_path), Some(bits), Some(method), Some(group_size)) =
            (module_path, bits, method, group_size)
        else {
            malformed += 1;
            continue;
        };
        unsupported_bits.insert(bits);
        if success.is_none()
            || fallback.is_none()
            || (bits < 16 && group_size.is_none())
            || (bits == 16 && group_size.is_some())
        {
            malformed += 1;
            continue;
        }

        if !seen.insert(module_path.to_string()) {
            duplicates += 1;
            continue;
        }
        let Some(expected) = planned.get(module_path) else {
            unexpected += 1;
            continue;
        };
        if expected.bits != bits || expected.method != method || expected.group_size != group_size {
            mismatched += 1;
        }
    }

    let missing = planned
        .keys()
        .filter(|module| !seen.contains(*module))
        .count();
    if malformed > 0 {
        issues.push(format!(
            "quantizer execution has {malformed} malformed module records"
        ));
    }
    if duplicates > 0 {
        issues.push(format!(
            "quantizer execution has {duplicates} duplicate module records"
        ));
    }
    if unexpected > 0 {
        issues.push(format!(
            "quantizer execution has {unexpected} modules absent from the quantization plan"
        ));
    }
    if mismatched > 0 {
        issues.push(format!(
            "quantizer execution has {mismatched} modules whose method, bits, or group size differ from the plan"
        ));
    }
    if missing > 0 {
        issues.push(format!(
            "quantizer execution is missing {missing} planned quantized modules"
        ));
    }
    warnings.extend(
        unsupported_bits
            .into_iter()
            .filter_map(|bits| axquant_bits_runtime_support_warning("quantizer execution", bits)),
    );
}

fn json_optional_positive_u64(value: &Value, field: &str) -> Option<Option<u64>> {
    match value.get(field) {
        None | Some(Value::Null) => Some(None),
        Some(value) => value.as_u64().filter(|number| *number > 0).map(Some),
    }
}

pub(crate) fn axquant_stable_sha256(value: &Value) -> Option<String> {
    let mut canonical = String::new();
    write_axquant_canonical_json(value, &mut canonical)?;
    Some(sha256_hex(canonical.as_bytes()))
}

fn write_axquant_canonical_json(value: &Value, output: &mut String) -> Option<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&python_json_number(value)?),
        Value::String(value) => output.push_str(&serde_json::to_string(value).ok()?),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_axquant_canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| *key);
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).ok()?);
                output.push(':');
                write_axquant_canonical_json(value, output)?;
            }
            output.push('}');
        }
    }
    Some(())
}

fn python_json_number(number: &serde_json::Number) -> Option<String> {
    let source = number.to_string();
    if !source.contains(['.', 'e', 'E']) {
        return Some(if source == "-0" {
            "0".to_string()
        } else {
            source
        });
    }
    if let Some(value) = number.as_i64() {
        return Some(value.to_string());
    }
    if let Some(value) = number.as_u64() {
        return Some(value.to_string());
    }

    let value = number.as_f64()?;
    if !value.is_finite() {
        return None;
    }
    let magnitude = value.abs();
    let rendered = if value != 0.0 && !(0.0001..1.0e16).contains(&magnitude) {
        format!("{value:e}")
    } else {
        serde_json::Number::from_f64(value)?.to_string()
    };
    let Some((mantissa, exponent)) = rendered.split_once('e') else {
        return Some(rendered);
    };
    let exponent = exponent.parse::<i32>().ok()?;
    Some(format!("{mantissa}e{exponent:+03}"))
}

fn json_string(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_string)
}

fn json_positive_f64(value: &Value, field: &str) -> Option<f64> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug, PartialEq)]
struct DoctorModelArtifactsHint {
    model_type: Option<String>,
    quantization: Option<DoctorQuantizationHint>,
    axquant: Option<DoctorAxquantReport>,
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
    model_artifacts: &DoctorModelArtifactsReport,
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

    if !model_artifacts.selected {
        advice.push(DoctorAdvice::info(
            "model_artifacts_not_selected",
            "Pass --mlx-model-artifacts-dir for model-specific quantization advice.",
            "Without model artifacts doctor can only report runtime-level guidance, not whether this checkpoint should be compared against another quantization.",
        ));
        return advice;
    }

    match inspect_doctor_model_artifacts(model_artifacts) {
        Ok(model_hint) => advice.extend(doctor_model_performance_advice(&model_hint)),
        Err(message) => advice.push(DoctorAdvice::warning(
            "model_artifacts_unreadable",
            "Model-specific performance advice is unavailable.",
            &message,
        )),
    }

    advice
}

fn inspect_doctor_model_artifacts(
    report: &DoctorModelArtifactsReport,
) -> Result<DoctorModelArtifactsHint, String> {
    let path = report.path.as_deref().unwrap_or("unknown");
    if !report.exists {
        return Err(format!("model artifacts path does not exist: {path}"));
    }
    if !report.is_dir {
        return Err(format!("model artifacts path is not a directory: {path}"));
    }

    if !report.config_present {
        return Err(format!(
            "model artifacts path is missing config.json: {path}"
        ));
    }

    if !report.manifest_present {
        return Err(format!(
            "model artifacts path is missing model-manifest.json: {path}; run `cargo run -p ax-engine-core --bin generate-manifest -- {path}` before using this snapshot as AX MLX artifacts"
        ));
    }

    Ok(DoctorModelArtifactsHint {
        model_type: report.model_type.clone(),
        quantization: report.quantization.clone(),
        axquant: report.axquant.clone(),
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
        let path = entry.path();
        path.is_file()
            && path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
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

    if let Some(axquant) = &hint.axquant {
        if axquant.metadata_valid {
            let precisions = axquant
                .precision_bits
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("/");
            let measured_bpw = axquant
                .measured_total_bpw
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "unknown".to_string());
            advice.push(DoctorAdvice::info(
                "axquant_artifact_detected",
                "AXQuant metadata and quantizer lineage are valid.",
                &format!(
                    "This is an AXQuant {}precision artifact at measured {measured_bpw} BPW with {precisions}-bit assignments and {} successful quantized modules; use these values instead of the config's global storage default.",
                    if axquant.mixed_precision { "mixed-" } else { "single-" },
                    axquant.quantized_module_count,
                ),
            ));
        } else {
            advice.push(DoctorAdvice::warning(
                "axquant_metadata_invalid",
                "AXQuant metadata is incomplete or inconsistent.",
                "Resolve the model-artifact issues before treating this checkpoint as an AXQuant artifact or collecting benchmark evidence.",
            ));
        }
        if !axquant.release_quality_evidence {
            advice.push(DoctorAdvice::warning(
                "axquant_development_evidence",
                "This AXQuant plan is not release-quality measured evidence.",
                "The runtime may execute the artifact, but certification and publication require an evidence_kind of measured plus the bound complete-candidate validation chain.",
            ));
        }
        if !axquant.provenance_complete {
            advice.push(DoctorAdvice::warning(
                "axquant_provenance_incomplete",
                "AXQuant source-model provenance is incomplete.",
                "Record both the source model ID and pinned revision before using this artifact in reproducible benchmark or release evidence.",
            ));
        }
        for warning in &axquant.warnings {
            advice.push(DoctorAdvice::warning(
                "axquant_bits_runtime_support",
                "AXQuant assignments use bit widths the runtime cannot load unconditionally.",
                warning,
            ));
        }
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
        format!(
            "  - Runtime assets: {}",
            report.runtime_assets.status.human_label()
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

    lines.push(format!(
        "  - AXQuant metadata: {}",
        yes_no(report.model_artifacts.axquant.is_some())
    ));
    if let Some(axquant) = &report.model_artifacts.axquant {
        lines.extend([
            format!(
                "  - AXQuant metadata valid: {}",
                yes_no(axquant.metadata_valid)
            ),
            format!(
                "  - AXQuant lineage valid: {}",
                yes_no(axquant.lineage_valid)
            ),
            format!(
                "  - AXQuant evidence: {}",
                axquant.evidence_kind.as_deref().unwrap_or("unknown")
            ),
            format!(
                "  - AXQuant source: {}@{}",
                axquant.source_model_id.as_deref().unwrap_or("unknown"),
                axquant.source_revision.as_deref().unwrap_or("unknown")
            ),
            format!(
                "  - AXQuant measured BPW: {}",
                axquant
                    .measured_total_bpw
                    .map(|value| format!("{value:.4}"))
                    .unwrap_or_else(|| "unknown".to_string())
            ),
            format!(
                "  - AXQuant precision bits: {}",
                if axquant.precision_bits.is_empty() {
                    "unknown".to_string()
                } else {
                    axquant
                        .precision_bits
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join("/")
                }
            ),
            format!(
                "  - AXQuant quantized modules: {} (failed {}, fallback {})",
                axquant.quantized_module_count,
                axquant.failed_module_count,
                axquant.fallback_module_count
            ),
        ]);
    }

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
        "Runtime assets:".to_string(),
        format!("  - Status: {}", report.runtime_assets.status.human_label()),
        format!(
            "  - Source: {}",
            report.runtime_assets.source.as_deref().unwrap_or("none")
        ),
        format!(
            "  - Path: {}",
            report.runtime_assets.path.as_deref().unwrap_or("none")
        ),
        format!(
            "  - Issue: {}",
            report.runtime_assets.issue.as_deref().unwrap_or("none")
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
