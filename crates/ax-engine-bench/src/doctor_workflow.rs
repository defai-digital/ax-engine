use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};

use crate::error::CliError;
use crate::path_utils::path_string;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DoctorWorkflowMode {
    Unknown,
    SourceCheckout,
    InstalledTools,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DoctorWorkflowReport {
    pub(crate) mode: DoctorWorkflowMode,
    pub(crate) cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_root: Option<String>,
    pub(crate) doctor: DoctorWorkflowCommand,
    pub(crate) server: DoctorWorkflowCommand,
    pub(crate) generate_manifest: DoctorWorkflowCommand,
    pub(crate) benchmark: DoctorWorkflowCommand,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) download_model: Option<DoctorWorkflowCommand>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DoctorWorkflowCommand {
    pub(crate) argv: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cwd: Option<String>,
}

impl DoctorWorkflowCommand {
    fn new(argv: &[&str], cwd: Option<&Path>) -> Self {
        Self {
            argv: argv.iter().map(|arg| (*arg).to_string()).collect(),
            cwd: cwd.map(path_string),
        }
    }
}

impl DoctorWorkflowReport {
    pub(crate) fn unknown() -> Self {
        Self {
            mode: DoctorWorkflowMode::Unknown,
            cwd: String::new(),
            source_root: None,
            doctor: DoctorWorkflowCommand::new(&[], None),
            server: DoctorWorkflowCommand::new(&[], None),
            generate_manifest: DoctorWorkflowCommand::new(&[], None),
            benchmark: DoctorWorkflowCommand::new(&[], None),
            download_model: None,
        }
    }
}

pub(crate) fn detect_doctor_workflow_report() -> Result<DoctorWorkflowReport, CliError> {
    let cwd = env::current_dir().map_err(|error| {
        CliError::Runtime(format!(
            "failed to resolve current working directory for workflow discovery: {error}"
        ))
    })?;
    Ok(doctor_workflow_report_for_cwd(&cwd))
}

pub(crate) fn doctor_workflow_report_for_cwd(cwd: &Path) -> DoctorWorkflowReport {
    if let Some(source_root) = find_source_checkout_root(cwd) {
        return DoctorWorkflowReport {
            mode: DoctorWorkflowMode::SourceCheckout,
            cwd: path_string(cwd),
            source_root: Some(path_string(&source_root)),
            doctor: DoctorWorkflowCommand::new(
                &[
                    "cargo",
                    "run",
                    "-p",
                    "ax-engine-bench",
                    "--",
                    "doctor",
                    "--json",
                ],
                Some(&source_root),
            ),
            server: DoctorWorkflowCommand::new(
                &["cargo", "run", "-p", "ax-engine-server", "--"],
                Some(&source_root),
            ),
            generate_manifest: DoctorWorkflowCommand::new(
                &[
                    "cargo",
                    "run",
                    "-p",
                    "ax-engine-bench",
                    "--",
                    "generate-manifest",
                    "<model-dir>",
                    "--json",
                ],
                Some(&source_root),
            ),
            benchmark: DoctorWorkflowCommand::new(
                &[
                    "cargo",
                    "run",
                    "-p",
                    "ax-engine-bench",
                    "--",
                    "scenario",
                    "--manifest",
                    "<manifest>",
                    "--output-root",
                    "<output-root>",
                    "--json",
                ],
                Some(&source_root),
            ),
            download_model: Some(DoctorWorkflowCommand::new(
                &[
                    "python3",
                    "scripts/download_model.py",
                    "<repo-id>",
                    "--json",
                ],
                Some(&source_root),
            )),
        };
    }

    DoctorWorkflowReport {
        mode: DoctorWorkflowMode::InstalledTools,
        cwd: path_string(cwd),
        source_root: None,
        doctor: DoctorWorkflowCommand::new(&["ax-engine-bench", "doctor", "--json"], None),
        server: DoctorWorkflowCommand::new(&["ax-engine-server"], None),
        generate_manifest: DoctorWorkflowCommand::new(
            &[
                "ax-engine-bench",
                "generate-manifest",
                "<model-dir>",
                "--json",
            ],
            None,
        ),
        benchmark: DoctorWorkflowCommand::new(
            &[
                "ax-engine-bench",
                "scenario",
                "--manifest",
                "<manifest>",
                "--output-root",
                "<output-root>",
                "--json",
            ],
            None,
        ),
        download_model: None,
    }
}

fn find_source_checkout_root(cwd: &Path) -> Option<PathBuf> {
    cwd.ancestors()
        .find(|candidate| is_source_checkout_root(candidate))
        .map(Path::to_path_buf)
}

fn is_source_checkout_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("scripts/download_model.py").is_file()
        && path.join("crates/ax-engine-server/Cargo.toml").is_file()
        && path.join("crates/ax-engine-bench/Cargo.toml").is_file()
}

pub(crate) fn workflow_mode_label(mode: DoctorWorkflowMode) -> &'static str {
    match mode {
        DoctorWorkflowMode::Unknown => "unknown",
        DoctorWorkflowMode::SourceCheckout => "source_checkout",
        DoctorWorkflowMode::InstalledTools => "installed_tools",
    }
}

pub(crate) fn command_text(command: &DoctorWorkflowCommand) -> String {
    if command.argv.is_empty() {
        return "none".to_string();
    }
    let argv = command.argv.join(" ");
    if let Some(cwd) = command.cwd.as_deref() {
        format!("{argv} [in: {cwd}]")
    } else {
        argv
    }
}
