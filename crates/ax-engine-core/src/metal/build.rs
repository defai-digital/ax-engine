use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    MetalKernelSpec, MetalKernelTier, MetalRuntimeError, PHASE1_DEFAULT_BLOCK_SIZE_TOKENS,
    PHASE1_DEFERRED_METAL_KERNELS, PHASE1_METAL_BLOCK_SIZE_ALIGNMENT_TOKENS,
    PHASE1_METAL_BUILD_GATE, PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION,
    PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION, PHASE1_METAL_LANGUAGE_STANDARD,
    PHASE1_METAL_LIBRARY_NAME, PHASE1_METAL_NATIVE_TARGET, PHASE1_OPTIONAL_METAL_KERNELS,
    PHASE1_REQUIRED_METAL_KERNELS, PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS,
    REQUIRED_TOOLCHAIN_REQUIREMENTS,
};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalKernelManifest {
    pub schema_version: String,
    pub native_target: String,
    pub metal_language_standard: String,
    pub library_name: String,
    pub default_block_size_tokens: u32,
    pub supported_block_size_tokens: Vec<u32>,
    pub source_file: PathBuf,
    pub toolchain_requirements: Vec<String>,
    pub build_gate: String,
    pub kernels: Vec<MetalKernelSpec>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalBuildStatus {
    Unknown,
    Compiled,
    SkippedToolchainUnavailable,
    SkippedNotReady,
    FailedCompile,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalBuildHostReport {
    pub os: String,
    pub arch: String,
    #[serde(default)]
    pub detected_soc: Option<String>,
    #[serde(default)]
    pub supported_native_runtime: bool,
    #[serde(default)]
    pub unsupported_host_override_active: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalBuildToolStatus {
    #[serde(default)]
    pub available: bool,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalBuildToolchainReport {
    #[serde(default)]
    pub fully_available: bool,
    #[serde(default)]
    pub metal: MetalBuildToolStatus,
    #[serde(default)]
    pub metallib: MetalBuildToolStatus,
    #[serde(default)]
    pub metal_ar: MetalBuildToolStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalBuildDoctorReport {
    pub status: String,
    pub bringup_allowed: bool,
    pub native_runtime_ready: bool,
    pub metal_toolchain_fully_available: bool,
    pub host: MetalBuildHostReport,
    pub metal_toolchain: MetalBuildToolchainReport,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalBuildOutputs {
    pub air: Option<PathBuf>,
    pub metalar: Option<PathBuf>,
    pub metallib: Option<PathBuf>,
    pub air_sha256: Option<String>,
    pub metalar_sha256: Option<String>,
    pub metallib_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalBuildReport {
    pub schema_version: String,
    pub manifest_path: PathBuf,
    pub source_file: PathBuf,
    pub native_target: String,
    pub metal_language_standard: String,
    pub library_name: String,
    pub default_block_size_tokens: u32,
    pub supported_block_size_tokens: Vec<u32>,
    pub toolchain_requirements: Vec<String>,
    pub doctor: MetalBuildDoctorReport,
    pub kernels: Vec<MetalKernelSpec>,
    pub source_sha256: String,
    pub outputs: MetalBuildOutputs,
    pub compile_commands: Vec<Vec<String>>,
    pub status: MetalBuildStatus,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalKernelBuildRequest {
    pub manifest_path: PathBuf,
    pub output_dir: PathBuf,
    pub doctor: MetalBuildDoctorReport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalKernelBuildArtifacts {
    pub manifest_path: PathBuf,
    pub output_dir: PathBuf,
    pub doctor_path: PathBuf,
    pub build_report_path: PathBuf,
    pub summary_path: PathBuf,
    pub build_report: MetalBuildReport,
    pub reused_existing_artifacts: bool,
}

impl MetalKernelBuildArtifacts {
    pub fn build_status(&self) -> MetalBuildStatus {
        self.build_report.status
    }

    pub fn reused_existing_artifacts(&self) -> bool {
        self.reused_existing_artifacts
    }
}

pub fn build_phase1_kernel_artifacts(
    request: &MetalKernelBuildRequest,
) -> Result<MetalKernelBuildArtifacts, MetalRuntimeError> {
    fs::create_dir_all(&request.output_dir).map_err(|source| {
        MetalRuntimeError::WriteBuildArtifact {
            path: request.output_dir.clone(),
            source,
        }
    })?;

    let manifest: MetalKernelManifest = read_json_file(&request.manifest_path)?;
    validate_phase1_manifest(&manifest)?;

    let workspace_root = workspace_root_from_manifest_path(&request.manifest_path)?;
    let resolved_source_file = workspace_root.join(&manifest.source_file);
    let source_bytes = read_non_empty_file(&resolved_source_file)?;
    let source_text =
        std::str::from_utf8(&source_bytes).map_err(|_| MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "source_file must be valid UTF-8 text: {}",
                resolved_source_file.display()
            ),
        })?;

    let manifest_kernel_names = manifest
        .kernels
        .iter()
        .map(|kernel| kernel.name.clone())
        .collect::<BTreeSet<_>>();
    let declared_kernel_names = extract_declared_kernel_names(source_text);
    let missing_source_kernels = manifest_kernel_names
        .difference(&declared_kernel_names)
        .cloned()
        .collect::<Vec<_>>();
    let extra_source_kernels = declared_kernel_names
        .difference(&manifest_kernel_names)
        .cloned()
        .collect::<Vec<_>>();

    let air_path = request
        .output_dir
        .join(format!("{}.air", manifest.library_name));
    let metalar_path = request
        .output_dir
        .join(format!("{}.metalar", manifest.library_name));
    let metallib_path = request
        .output_dir
        .join(format!("{}.metallib", manifest.library_name));
    let build_report_path = request.output_dir.join("build_report.json");
    let doctor_path = request.output_dir.join("doctor.json");
    let summary_path = request.output_dir.join("summary.md");

    let compile_air_cmd = vec![
        "xcrun".to_string(),
        "--sdk".to_string(),
        "macosx".to_string(),
        "metal".to_string(),
        format!("-std={}", manifest.metal_language_standard),
        "-O3".to_string(),
        "-Wall".to_string(),
        "-Wextra".to_string(),
        "-c".to_string(),
        resolved_source_file.display().to_string(),
        "-o".to_string(),
        air_path.display().to_string(),
    ];
    let compile_metalar_cmd = vec![
        "xcrun".to_string(),
        "--sdk".to_string(),
        "macosx".to_string(),
        "metal-ar".to_string(),
        "-q".to_string(),
        metalar_path.display().to_string(),
        air_path.display().to_string(),
    ];
    let compile_metallib_cmd = vec![
        "xcrun".to_string(),
        "--sdk".to_string(),
        "macosx".to_string(),
        "metallib".to_string(),
        metalar_path.display().to_string(),
        "-o".to_string(),
        metallib_path.display().to_string(),
    ];

    if request.doctor.metal_toolchain_fully_available && request.doctor.bringup_allowed {
        if let Some(artifacts) = try_reuse_compiled_phase1_kernel_artifacts(
            request,
            &build_report_path,
            &doctor_path,
            &summary_path,
        )? {
            return Ok(artifacts);
        }
    }

    cleanup_partial_build_outputs([
        air_path.as_path(),
        metalar_path.as_path(),
        metallib_path.as_path(),
    ]);

    let mut build_report = MetalBuildReport {
        schema_version: PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION.to_string(),
        manifest_path: request.manifest_path.clone(),
        source_file: resolved_source_file.clone(),
        native_target: manifest.native_target.clone(),
        metal_language_standard: manifest.metal_language_standard.clone(),
        library_name: manifest.library_name.clone(),
        default_block_size_tokens: manifest.default_block_size_tokens,
        supported_block_size_tokens: manifest.supported_block_size_tokens.clone(),
        toolchain_requirements: manifest.toolchain_requirements.clone(),
        doctor: request.doctor.clone(),
        kernels: manifest.kernels.clone(),
        source_sha256: sha256_hex(&source_bytes),
        outputs: MetalBuildOutputs {
            air: None,
            metalar: None,
            metallib: None,
            air_sha256: None,
            metalar_sha256: None,
            metallib_sha256: None,
        },
        compile_commands: Vec::new(),
        status: MetalBuildStatus::Unknown,
        reason: None,
    };

    if !missing_source_kernels.is_empty() || !extra_source_kernels.is_empty() {
        build_report.status = MetalBuildStatus::FailedCompile;
        build_report.reason = Some(format!(
            "source kernel declarations do not match manifest; missing={missing_source_kernels:?}; extra={extra_source_kernels:?}"
        ));
    } else if !request.doctor.metal_toolchain_fully_available {
        build_report.status = MetalBuildStatus::SkippedToolchainUnavailable;
        build_report.reason = Some("Metal toolchain is incomplete on this machine".to_string());
    } else if !request.doctor.bringup_allowed {
        build_report.status = MetalBuildStatus::SkippedNotReady;
        build_report.reason =
            Some("AX native bring-up is not allowed on this machine without override".to_string());
    } else {
        build_report.compile_commands = vec![
            compile_air_cmd.clone(),
            compile_metalar_cmd.clone(),
            compile_metallib_cmd.clone(),
        ];

        let compile_result = run_toolchain_command(&compile_air_cmd, &workspace_root)
            .and_then(|_| run_toolchain_command(&compile_metalar_cmd, &workspace_root))
            .and_then(|_| run_toolchain_command(&compile_metallib_cmd, &workspace_root));

        match compile_result {
            Ok(()) => {
                build_report.status = MetalBuildStatus::Compiled;
                build_report.outputs.air = Some(air_path.clone());
                build_report.outputs.metalar = Some(metalar_path.clone());
                build_report.outputs.metallib = Some(metallib_path.clone());
                build_report.outputs.air_sha256 = Some(file_sha256(&air_path)?);
                build_report.outputs.metalar_sha256 = Some(file_sha256(&metalar_path)?);
                build_report.outputs.metallib_sha256 = Some(file_sha256(&metallib_path)?);
            }
            Err(reason) => {
                cleanup_partial_build_outputs([
                    air_path.as_path(),
                    metalar_path.as_path(),
                    metallib_path.as_path(),
                ]);
                build_report.status = MetalBuildStatus::FailedCompile;
                build_report.reason = Some(reason);
            }
        }
    }

    if build_report.status == MetalBuildStatus::Unknown {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "metal kernel builder must never emit status=unknown".to_string(),
        });
    }

    write_build_artifact_reports(
        &doctor_path,
        &build_report_path,
        &summary_path,
        &request.doctor,
        &build_report,
    )?;

    Ok(MetalKernelBuildArtifacts {
        manifest_path: request.manifest_path.clone(),
        output_dir: request.output_dir.clone(),
        doctor_path,
        build_report_path,
        summary_path,
        build_report,
        reused_existing_artifacts: false,
    })
}

fn try_reuse_compiled_phase1_kernel_artifacts(
    request: &MetalKernelBuildRequest,
    build_report_path: &Path,
    doctor_path: &Path,
    summary_path: &Path,
) -> Result<Option<MetalKernelBuildArtifacts>, MetalRuntimeError> {
    let Ok(assets) = MetalKernelAssets::from_build_dir(&request.output_dir) else {
        return Ok(None);
    };
    if assets.build_status() != MetalBuildStatus::Compiled {
        return Ok(None);
    }
    if assets.build_report().manifest_path != request.manifest_path {
        return Ok(None);
    }

    let mut build_report = assets.build_report().clone();
    build_report.doctor = request.doctor.clone();
    write_build_artifact_reports(
        doctor_path,
        build_report_path,
        summary_path,
        &request.doctor,
        &build_report,
    )?;

    Ok(Some(MetalKernelBuildArtifacts {
        manifest_path: request.manifest_path.clone(),
        output_dir: request.output_dir.clone(),
        doctor_path: doctor_path.to_path_buf(),
        build_report_path: build_report_path.to_path_buf(),
        summary_path: summary_path.to_path_buf(),
        build_report,
        reused_existing_artifacts: true,
    }))
}

fn write_build_artifact_reports(
    doctor_path: &Path,
    build_report_path: &Path,
    summary_path: &Path,
    doctor: &MetalBuildDoctorReport,
    build_report: &MetalBuildReport,
) -> Result<(), MetalRuntimeError> {
    write_json_pretty_file(doctor_path, doctor)?;
    write_json_pretty_file(build_report_path, build_report)?;
    fs::write(summary_path, build_summary_markdown(build_report)).map_err(|source| {
        MetalRuntimeError::WriteBuildArtifact {
            path: summary_path.to_path_buf(),
            source,
        }
    })?;
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalKernelAssets {
    build_dir: PathBuf,
    workspace_root: PathBuf,
    manifest: MetalKernelManifest,
    build_report: MetalBuildReport,
    resolved_source_file: PathBuf,
}

impl MetalKernelAssets {
    pub fn from_build_dir(path: impl AsRef<Path>) -> Result<Self, MetalRuntimeError> {
        let build_dir = path.as_ref().to_path_buf();
        let build_report_path = build_dir.join("build_report.json");
        let build_report: MetalBuildReport = read_json_file(&build_report_path)?;
        validate_build_report_shape(&build_report)?;

        let manifest: MetalKernelManifest = read_json_file(&build_report.manifest_path)?;
        validate_phase1_manifest(&manifest)?;

        let workspace_root = workspace_root_from_manifest_path(&build_report.manifest_path)?;
        let resolved_source_file = workspace_root.join(&manifest.source_file);
        validate_assets(
            &manifest,
            &build_report,
            &resolved_source_file,
            &build_report.manifest_path,
        )?;

        Ok(Self {
            build_dir,
            workspace_root,
            manifest,
            build_report,
            resolved_source_file,
        })
    }

    pub fn build_dir(&self) -> &Path {
        &self.build_dir
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn manifest(&self) -> &MetalKernelManifest {
        &self.manifest
    }

    pub fn default_block_size_tokens(&self) -> u32 {
        self.manifest.default_block_size_tokens
    }

    pub fn supported_block_size_tokens(&self) -> &[u32] {
        &self.manifest.supported_block_size_tokens
    }

    pub fn supports_block_size_tokens(&self, block_size_tokens: u32) -> bool {
        self.manifest
            .supported_block_size_tokens
            .contains(&block_size_tokens)
    }

    pub fn validate_block_size_tokens(
        &self,
        block_size_tokens: u32,
    ) -> Result<(), MetalRuntimeError> {
        validate_supported_block_size_tokens(
            block_size_tokens,
            self.default_block_size_tokens(),
            self.supported_block_size_tokens(),
        )
    }

    pub fn build_report(&self) -> &MetalBuildReport {
        &self.build_report
    }

    pub fn resolved_source_file(&self) -> &Path {
        &self.resolved_source_file
    }

    pub fn build_status(&self) -> MetalBuildStatus {
        self.build_report.status
    }

    pub fn kernel(&self, name: &str) -> Option<&MetalKernelSpec> {
        self.manifest
            .kernels
            .iter()
            .find(|kernel| kernel.name == name)
    }

    pub fn required_kernel(&self, name: &str) -> Result<&MetalKernelSpec, MetalRuntimeError> {
        let Some(kernel) = self.kernel(name) else {
            return Err(MetalRuntimeError::UnknownKernel {
                kernel_name: name.to_string(),
            });
        };

        if kernel.tier != MetalKernelTier::Required {
            return Err(MetalRuntimeError::InvalidManifest {
                message: format!("kernel {name} must be tier=required"),
            });
        }

        Ok(kernel)
    }

    pub fn compiled_metallib_path(&self) -> Option<&Path> {
        self.build_report.outputs.metallib.as_deref()
    }

    pub fn compiled_metallib_bytes(&self) -> Result<Vec<u8>, MetalRuntimeError> {
        let Some(path) = self.compiled_metallib_path() else {
            return Err(MetalRuntimeError::BuildNotCompiled {
                status: self.build_report.status,
            });
        };

        read_non_empty_file(path)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalKernelBinary {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
}

pub(super) fn read_json_file<T>(path: &Path) -> Result<T, MetalRuntimeError>
where
    T: for<'de> Deserialize<'de>,
{
    let bytes = fs::read(path).map_err(|source| MetalRuntimeError::ReadJson {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| MetalRuntimeError::ParseJson {
        path: path.to_path_buf(),
        source,
    })
}

fn write_json_pretty_file<T>(path: &Path, value: &T) -> Result<(), MetalRuntimeError>
where
    T: Serialize,
{
    let json = serde_json::to_vec_pretty(value).map_err(|source| MetalRuntimeError::ParseJson {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, json).map_err(|source| MetalRuntimeError::WriteBuildArtifact {
        path: path.to_path_buf(),
        source,
    })
}

fn build_summary_markdown(report: &MetalBuildReport) -> String {
    let outputs = &report.outputs;
    let mut lines = vec![
        "# AX Metal Kernel Build".to_string(),
        String::new(),
        format!("- status: `{}`", metal_build_status_label(report.status)),
        format!("- library: `{}`", report.library_name),
        format!("- manifest: `{}`", report.manifest_path.display()),
        format!("- source: `{}`", report.source_file.display()),
        format!("- doctor_status: `{}`", report.doctor.status),
        format!(
            "- default_block_size_tokens: `{}`",
            report.default_block_size_tokens
        ),
        format!(
            "- supported_block_size_tokens: `{}`",
            report
                .supported_block_size_tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        format!(
            "- bringup_allowed: `{}`",
            bool_label(report.doctor.bringup_allowed)
        ),
        format!(
            "- metal_toolchain_fully_available: `{}`",
            bool_label(report.doctor.metal_toolchain_fully_available)
        ),
        format!(
            "- kernels: `{}`",
            report
                .kernels
                .iter()
                .map(|kernel| kernel.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ];

    if let Some(reason) = report.reason.as_deref() {
        lines.push(format!("- reason: `{reason}`"));
    }
    if let Some(path) = outputs.air.as_deref() {
        lines.push(format!("- air: `{}`", path.display()));
    }
    if let Some(path) = outputs.metalar.as_deref() {
        lines.push(format!("- metalar: `{}`", path.display()));
    }
    if let Some(path) = outputs.metallib.as_deref() {
        lines.push(format!("- metallib: `{}`", path.display()));
    }

    lines.join("\n") + "\n"
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

fn bool_label(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn cleanup_partial_build_outputs<'a>(paths: impl IntoIterator<Item = &'a Path>) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn run_toolchain_command(command: &[String], cwd: &Path) -> Result<(), String> {
    let Some((program, args)) = command.split_first() else {
        return Err("toolchain command must not be empty".to_string());
    };

    let mut attempt_failures = Vec::new();
    for developer_dir in xcrun_developer_dir_candidates(program) {
        let mut process = Command::new(program);
        process.args(args).current_dir(cwd);
        if let Some(developer_dir) = developer_dir.as_deref() {
            process.env("DEVELOPER_DIR", developer_dir);
        }

        let output = match process.output() {
            Ok(output) => output,
            Err(source) => {
                attempt_failures.push(format!(
                    "{}: failed to spawn {}: {source}",
                    toolchain_attempt_label(developer_dir.as_deref()),
                    shell_join_command(command)
                ));
                continue;
            }
        };

        if output.status.success() {
            return Ok(());
        }

        let status = output.status.code().map_or_else(
            || "terminated by signal".to_string(),
            |code| format!("exit code {code}"),
        );
        let stdout = summarize_command_stream(&output.stdout);
        let stderr = summarize_command_stream(&output.stderr);
        let mut reason = format!(
            "{}: command failed with {status}: {}",
            toolchain_attempt_label(developer_dir.as_deref()),
            shell_join_command(command)
        );
        if let Some(stderr) = stderr {
            reason.push_str(&format!("; stderr={stderr}"));
        }
        if let Some(stdout) = stdout {
            reason.push_str(&format!("; stdout={stdout}"));
        }
        attempt_failures.push(reason);
    }

    if attempt_failures.is_empty() {
        return Err(format!("failed to spawn {}", shell_join_command(command)));
    }

    Err(attempt_failures.join(" | "))
}

fn xcrun_developer_dir_candidates(program: &str) -> Vec<Option<String>> {
    if program != "xcrun" {
        return vec![None];
    }

    let mut candidates = Vec::new();
    let mut seen = BTreeSet::new();
    for candidate in [
        std::env::var("DEVELOPER_DIR").ok(),
        Some("/Applications/Xcode.app/Contents/Developer".to_string())
            .filter(|path| Path::new(path).is_dir()),
        command_stdout("xcode-select", &["-p"]).filter(|path| Path::new(path).is_dir()),
    ] {
        let Some(candidate) = candidate else {
            continue;
        };
        if seen.insert(candidate.clone()) {
            candidates.push(Some(candidate));
        }
    }
    candidates.push(None);
    candidates
}

fn toolchain_attempt_label(developer_dir: Option<&str>) -> String {
    match developer_dir {
        Some(path) => format!("DEVELOPER_DIR={path}"),
        None => "DEVELOPER_DIR=<inherited>".to_string(),
    }
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn shell_join_command(command: &[String]) -> String {
    command
        .iter()
        .map(|part| {
            if part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"/._-=:".contains(&byte))
            {
                part.clone()
            } else {
                format!("{part:?}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn summarize_command_stream(bytes: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(bytes);
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    if trimmed.is_empty() {
        return None;
    }

    const MAX_LEN: usize = 240;
    if trimmed.len() <= MAX_LEN {
        return Some(trimmed.to_string());
    }

    let mut truncated = trimmed.chars().take(MAX_LEN).collect::<String>();
    truncated.push_str("...");
    Some(truncated)
}

fn validate_phase1_manifest(manifest: &MetalKernelManifest) -> Result<(), MetalRuntimeError> {
    if manifest.schema_version != PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION {
        return Err(MetalRuntimeError::InvalidManifest {
            message: format!(
                "schema_version must be {PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION}, got {}",
                manifest.schema_version
            ),
        });
    }
    if manifest.native_target != PHASE1_METAL_NATIVE_TARGET {
        return Err(MetalRuntimeError::InvalidManifest {
            message: format!(
                "native_target must be {PHASE1_METAL_NATIVE_TARGET}, got {}",
                manifest.native_target
            ),
        });
    }
    if manifest.metal_language_standard != PHASE1_METAL_LANGUAGE_STANDARD {
        return Err(MetalRuntimeError::InvalidManifest {
            message: format!(
                "metal_language_standard must be {PHASE1_METAL_LANGUAGE_STANDARD}, got {}",
                manifest.metal_language_standard
            ),
        });
    }
    if manifest.library_name != PHASE1_METAL_LIBRARY_NAME {
        return Err(MetalRuntimeError::InvalidManifest {
            message: format!(
                "library_name must be {PHASE1_METAL_LIBRARY_NAME}, got {}",
                manifest.library_name
            ),
        });
    }
    validate_phase1_block_size_policy(
        manifest.default_block_size_tokens,
        &manifest.supported_block_size_tokens,
    )
    .map_err(|message| MetalRuntimeError::InvalidManifest { message })?;
    if manifest.build_gate != PHASE1_METAL_BUILD_GATE {
        return Err(MetalRuntimeError::InvalidManifest {
            message: format!(
                "build_gate must be {PHASE1_METAL_BUILD_GATE}, got {}",
                manifest.build_gate
            ),
        });
    }
    let toolchain_requirements = manifest
        .toolchain_requirements
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if toolchain_requirements != REQUIRED_TOOLCHAIN_REQUIREMENTS {
        return Err(MetalRuntimeError::InvalidManifest {
            message: format!(
                "toolchain_requirements must be {:?}, got {:?}",
                REQUIRED_TOOLCHAIN_REQUIREMENTS, manifest.toolchain_requirements
            ),
        });
    }

    let mut seen_names = BTreeSet::new();
    for kernel in &manifest.kernels {
        if !seen_names.insert(kernel.name.clone()) {
            return Err(MetalRuntimeError::InvalidManifest {
                message: format!("duplicate kernel entry: {}", kernel.name),
            });
        }
    }

    for kernel_name in PHASE1_REQUIRED_METAL_KERNELS {
        let Some(kernel) = manifest
            .kernels
            .iter()
            .find(|kernel| kernel.name == *kernel_name)
        else {
            return Err(MetalRuntimeError::InvalidManifest {
                message: format!("missing required kernel {kernel_name}"),
            });
        };
        if kernel.tier != MetalKernelTier::Required {
            return Err(MetalRuntimeError::InvalidManifest {
                message: format!("kernel {kernel_name} must be tier=required"),
            });
        }
    }

    for kernel_name in PHASE1_OPTIONAL_METAL_KERNELS {
        if let Some(kernel) = manifest
            .kernels
            .iter()
            .find(|kernel| kernel.name == *kernel_name)
        {
            if kernel.tier != MetalKernelTier::Optional {
                return Err(MetalRuntimeError::InvalidManifest {
                    message: format!("kernel {kernel_name} must be tier=optional"),
                });
            }
        }
    }

    for kernel_name in PHASE1_DEFERRED_METAL_KERNELS {
        let Some(kernel) = manifest
            .kernels
            .iter()
            .find(|kernel| kernel.name == *kernel_name)
        else {
            return Err(MetalRuntimeError::InvalidManifest {
                message: format!("missing deferred kernel {kernel_name}"),
            });
        };
        if kernel.tier != MetalKernelTier::Deferred {
            return Err(MetalRuntimeError::InvalidManifest {
                message: format!("kernel {kernel_name} must be tier=deferred"),
            });
        }
    }

    Ok(())
}

fn validate_build_report_shape(report: &MetalBuildReport) -> Result<(), MetalRuntimeError> {
    if report.schema_version != PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "schema_version must be {PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION}, got {}",
                report.schema_version
            ),
        });
    }
    validate_phase1_block_size_policy(
        report.default_block_size_tokens,
        &report.supported_block_size_tokens,
    )
    .map_err(|message| MetalRuntimeError::InvalidBuildReport { message })?;

    if report.status == MetalBuildStatus::Unknown {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "status unknown is not allowed in validated build artifacts".to_string(),
        });
    }

    Ok(())
}

fn validate_assets(
    manifest: &MetalKernelManifest,
    build_report: &MetalBuildReport,
    resolved_source_file: &Path,
    manifest_path: &Path,
) -> Result<(), MetalRuntimeError> {
    if build_report.manifest_path != manifest_path {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "manifest_path mismatch: report={}, expected={}",
                build_report.manifest_path.display(),
                manifest_path.display()
            ),
        });
    }
    if build_report.native_target != manifest.native_target {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "native_target mismatch: report={}, manifest={}",
                build_report.native_target, manifest.native_target
            ),
        });
    }
    if build_report.metal_language_standard != manifest.metal_language_standard {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "metal_language_standard mismatch: report={}, manifest={}",
                build_report.metal_language_standard, manifest.metal_language_standard
            ),
        });
    }
    if build_report.library_name != manifest.library_name {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "library_name mismatch: report={}, manifest={}",
                build_report.library_name, manifest.library_name
            ),
        });
    }
    if build_report.default_block_size_tokens != manifest.default_block_size_tokens {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "default_block_size_tokens mismatch: report={}, manifest={}",
                build_report.default_block_size_tokens, manifest.default_block_size_tokens
            ),
        });
    }
    if build_report.supported_block_size_tokens != manifest.supported_block_size_tokens {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "supported_block_size_tokens mismatch between build report and manifest"
                .to_string(),
        });
    }
    if build_report.toolchain_requirements != manifest.toolchain_requirements {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "toolchain_requirements mismatch between build report and manifest"
                .to_string(),
        });
    }
    if build_report.kernels != manifest.kernels {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "kernel inventory mismatch between build report and manifest".to_string(),
        });
    }
    if build_report.source_file != resolved_source_file {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "source_file mismatch: report={}, resolved={}",
                build_report.source_file.display(),
                resolved_source_file.display()
            ),
        });
    }
    if !resolved_source_file.is_file() {
        return Err(MetalRuntimeError::MissingBuildArtifact {
            path: resolved_source_file.to_path_buf(),
        });
    }
    validate_source_kernel_entry_points(manifest, resolved_source_file)?;

    let source_sha256 = file_sha256(resolved_source_file)?;
    if source_sha256 != build_report.source_sha256 {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "source_sha256 mismatch for {}",
                resolved_source_file.display()
            ),
        });
    }

    match build_report.status {
        MetalBuildStatus::Compiled => validate_compiled_outputs(&build_report.outputs),
        MetalBuildStatus::Unknown => Err(MetalRuntimeError::InvalidBuildReport {
            message: "status unknown is not allowed in validated build artifacts".to_string(),
        }),
        MetalBuildStatus::SkippedToolchainUnavailable
        | MetalBuildStatus::SkippedNotReady
        | MetalBuildStatus::FailedCompile => validate_non_compiled_outputs(build_report),
    }
}

fn validate_compiled_outputs(outputs: &MetalBuildOutputs) -> Result<(), MetalRuntimeError> {
    let Some(air_path) = outputs.air.as_deref() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "compiled build report must include outputs.air".to_string(),
        });
    };
    let Some(metalar_path) = outputs.metalar.as_deref() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "compiled build report must include outputs.metalar".to_string(),
        });
    };
    let Some(metallib_path) = outputs.metallib.as_deref() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "compiled build report must include outputs.metallib".to_string(),
        });
    };
    let Some(air_sha256) = outputs.air_sha256.as_deref() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "compiled build report must include outputs.air_sha256".to_string(),
        });
    };
    let Some(metalar_sha256) = outputs.metalar_sha256.as_deref() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "compiled build report must include outputs.metalar_sha256".to_string(),
        });
    };
    let Some(metallib_sha256) = outputs.metallib_sha256.as_deref() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: "compiled build report must include outputs.metallib_sha256".to_string(),
        });
    };

    if file_sha256(air_path)? != air_sha256 {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!("air_sha256 mismatch for {}", air_path.display()),
        });
    }
    if file_sha256(metalar_path)? != metalar_sha256 {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!("metalar_sha256 mismatch for {}", metalar_path.display()),
        });
    }
    if file_sha256(metallib_path)? != metallib_sha256 {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!("metallib_sha256 mismatch for {}", metallib_path.display()),
        });
    }

    Ok(())
}

fn validate_non_compiled_outputs(build_report: &MetalBuildReport) -> Result<(), MetalRuntimeError> {
    if build_report.reason.as_deref().is_none_or(str::is_empty) {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!("status {:?} must include a reason", build_report.status),
        });
    }
    if build_report.outputs.air.is_some()
        || build_report.outputs.metalar.is_some()
        || build_report.outputs.metallib.is_some()
        || build_report.outputs.air_sha256.is_some()
        || build_report.outputs.metalar_sha256.is_some()
        || build_report.outputs.metallib_sha256.is_some()
    {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "status {:?} must not include compiled output paths or hashes",
                build_report.status
            ),
        });
    }
    Ok(())
}

fn validate_source_kernel_entry_points(
    manifest: &MetalKernelManifest,
    resolved_source_file: &Path,
) -> Result<(), MetalRuntimeError> {
    let source_bytes = read_non_empty_file(resolved_source_file)?;
    let source_text =
        std::str::from_utf8(&source_bytes).map_err(|_| MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "source_file must be valid UTF-8 text: {}",
                resolved_source_file.display()
            ),
        })?;
    let declared_kernel_names = extract_declared_kernel_names(source_text);
    let manifest_kernel_names = manifest
        .kernels
        .iter()
        .map(|kernel| kernel.name.clone())
        .collect::<BTreeSet<_>>();

    if declared_kernel_names != manifest_kernel_names {
        let missing = manifest_kernel_names
            .difference(&declared_kernel_names)
            .cloned()
            .collect::<Vec<_>>();
        let extra = declared_kernel_names
            .difference(&manifest_kernel_names)
            .cloned()
            .collect::<Vec<_>>();
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "source kernel declarations do not match manifest for {}: missing={missing:?}, extra={extra:?}",
                resolved_source_file.display()
            ),
        });
    }

    Ok(())
}

fn validate_phase1_block_size_policy(
    default_block_size_tokens: u32,
    supported_block_size_tokens: &[u32],
) -> Result<(), String> {
    if supported_block_size_tokens.is_empty() {
        return Err("supported_block_size_tokens must not be empty".to_string());
    }
    if supported_block_size_tokens
        .windows(2)
        .any(|window| window[0] >= window[1])
    {
        return Err(
            "supported_block_size_tokens must be strictly increasing without duplicates"
                .to_string(),
        );
    }
    for &block_size_tokens in supported_block_size_tokens {
        if block_size_tokens == 0 {
            return Err("supported_block_size_tokens must not include zero".to_string());
        }
        if block_size_tokens % PHASE1_METAL_BLOCK_SIZE_ALIGNMENT_TOKENS != 0 {
            return Err(format!(
                "supported_block_size_tokens must be multiples of {}, got {}",
                PHASE1_METAL_BLOCK_SIZE_ALIGNMENT_TOKENS, block_size_tokens
            ));
        }
    }
    if !supported_block_size_tokens.contains(&default_block_size_tokens) {
        return Err(format!(
            "default_block_size_tokens {} must be listed in supported_block_size_tokens",
            default_block_size_tokens
        ));
    }
    if default_block_size_tokens != PHASE1_DEFAULT_BLOCK_SIZE_TOKENS {
        return Err(format!(
            "default_block_size_tokens must be {}, got {}",
            PHASE1_DEFAULT_BLOCK_SIZE_TOKENS, default_block_size_tokens
        ));
    }
    if supported_block_size_tokens != PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS {
        return Err(format!(
            "supported_block_size_tokens must be {:?}, got {:?}",
            PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS, supported_block_size_tokens
        ));
    }

    Ok(())
}

fn validate_supported_block_size_tokens(
    block_size_tokens: u32,
    default_block_size_tokens: u32,
    supported_block_size_tokens: &[u32],
) -> Result<(), MetalRuntimeError> {
    if supported_block_size_tokens.contains(&block_size_tokens) {
        return Ok(());
    }

    Err(MetalRuntimeError::UnsupportedNativeBlockSize {
        block_size_tokens,
        default_block_size_tokens,
        supported_block_size_tokens: supported_block_size_tokens.to_vec(),
    })
}

fn extract_declared_kernel_names(source_text: &str) -> BTreeSet<String> {
    let stripped = strip_metal_comments(source_text);
    let bytes = stripped.as_bytes();
    let mut names = BTreeSet::new();
    let mut index = 0;

    while index < bytes.len() {
        if !starts_word(bytes, index, b"kernel") {
            index += 1;
            continue;
        }

        let mut cursor = skip_ascii_whitespace(bytes, index + "kernel".len());
        if !starts_word(bytes, cursor, b"void") {
            index += "kernel".len();
            continue;
        }

        cursor = skip_ascii_whitespace(bytes, cursor + "void".len());
        let name_start = cursor;
        while cursor < bytes.len() && is_ident_char(bytes[cursor]) {
            cursor += 1;
        }
        if name_start == cursor {
            index += "kernel".len();
            continue;
        }

        let name_end = cursor;
        cursor = skip_ascii_whitespace(bytes, cursor);
        if bytes.get(cursor) == Some(&b'(') {
            names.insert(stripped[name_start..name_end].to_string());
        }

        index = cursor.saturating_add(1);
    }

    names
}

fn strip_metal_comments(source_text: &str) -> String {
    let bytes = source_text.as_bytes();
    let mut stripped = String::with_capacity(source_text.len());
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] == b'/' && index + 1 < bytes.len() {
            if bytes[index + 1] == b'/' {
                index += 2;
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
                continue;
            }

            if bytes[index + 1] == b'*' {
                index += 2;
                while index + 1 < bytes.len() && !(bytes[index] == b'*' && bytes[index + 1] == b'/')
                {
                    index += 1;
                }
                index = (index + 2).min(bytes.len());
                continue;
            }
        }

        stripped.push(bytes[index] as char);
        index += 1;
    }

    stripped
}

fn starts_word(bytes: &[u8], index: usize, word: &[u8]) -> bool {
    let Some(candidate) = bytes.get(index..index + word.len()) else {
        return false;
    };
    if candidate != word {
        return false;
    }

    let before_is_word = index > 0 && is_ident_char(bytes[index - 1]);
    let after_index = index + word.len();
    let after_is_word = after_index < bytes.len() && is_ident_char(bytes[after_index]);
    !before_is_word && !after_is_word
}

fn skip_ascii_whitespace(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}

fn is_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn workspace_root_from_manifest_path(manifest_path: &Path) -> Result<PathBuf, MetalRuntimeError> {
    let Some(metal_dir) = manifest_path.parent() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "manifest_path has no parent directory: {}",
                manifest_path.display()
            ),
        });
    };
    if metal_dir.file_name().and_then(|name| name.to_str()) != Some("metal") {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "manifest_path must live under a metal/ directory, got {}",
                manifest_path.display()
            ),
        });
    }
    let Some(workspace_root) = metal_dir.parent() else {
        return Err(MetalRuntimeError::InvalidBuildReport {
            message: format!(
                "manifest_path is missing a workspace root parent: {}",
                manifest_path.display()
            ),
        });
    };
    Ok(workspace_root.to_path_buf())
}

pub(super) fn load_compiled_metallib_binary(
    assets: &MetalKernelAssets,
) -> Result<MetalKernelBinary, MetalRuntimeError> {
    if assets.build_status() != MetalBuildStatus::Compiled {
        return Err(MetalRuntimeError::BuildNotCompiled {
            status: assets.build_status(),
        });
    }

    let metallib_path =
        assets
            .compiled_metallib_path()
            .ok_or(MetalRuntimeError::InvalidBuildReport {
                message: "compiled build report must include outputs.metallib".to_string(),
            })?;

    Ok(MetalKernelBinary {
        path: metallib_path.to_path_buf(),
        bytes: read_non_empty_file(metallib_path)?,
    })
}

pub(super) fn resolve_required_kernel_names(
    assets: &MetalKernelAssets,
) -> Result<Vec<String>, MetalRuntimeError> {
    let mut resolved_kernel_names = Vec::with_capacity(PHASE1_REQUIRED_METAL_KERNELS.len());
    for kernel_name in PHASE1_REQUIRED_METAL_KERNELS {
        assets.required_kernel(kernel_name)?;
        resolved_kernel_names.push((*kernel_name).to_string());
    }
    Ok(resolved_kernel_names)
}

fn manifest_kernel_names(manifest: &MetalKernelManifest) -> BTreeSet<String> {
    manifest
        .kernels
        .iter()
        .map(|kernel| kernel.name.clone())
        .collect()
}

pub(super) fn validate_compiled_kernel_inventory(
    manifest: &MetalKernelManifest,
    compiled_kernel_names: &[String],
    metallib_path: &Path,
) -> Result<(), MetalRuntimeError> {
    let manifest_kernel_names = manifest_kernel_names(manifest);
    let compiled_kernel_names = compiled_kernel_names
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();

    if compiled_kernel_names != manifest_kernel_names {
        let missing = manifest_kernel_names
            .difference(&compiled_kernel_names)
            .cloned()
            .collect::<Vec<_>>();
        let extra = compiled_kernel_names
            .difference(&manifest_kernel_names)
            .cloned()
            .collect::<Vec<_>>();
        return Err(MetalRuntimeError::CompiledKernelInventoryMismatch {
            path: metallib_path.to_path_buf(),
            missing,
            extra,
        });
    }

    Ok(())
}

fn read_non_empty_file(path: &Path) -> Result<Vec<u8>, MetalRuntimeError> {
    let bytes = fs::read(path).map_err(|_| MetalRuntimeError::MissingBuildArtifact {
        path: path.to_path_buf(),
    })?;
    if bytes.is_empty() {
        return Err(MetalRuntimeError::MissingBuildArtifact {
            path: path.to_path_buf(),
        });
    }
    Ok(bytes)
}

fn file_sha256(path: &Path) -> Result<String, MetalRuntimeError> {
    let bytes = read_non_empty_file(path)?;
    Ok(sha256_hex(&bytes))
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(digest.len() * 2);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}
