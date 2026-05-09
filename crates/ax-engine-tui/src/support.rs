use crate::app::{AppState, LoadState};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const SUPPORT_BUNDLE_SCHEMA_VERSION: &str = "ax.engine_manager.support_bundle.v1";

pub fn write_support_bundle(root: &Path, state: &AppState) -> Result<PathBuf, SupportBundleError> {
    fs::create_dir_all(root).map_err(|source| SupportBundleError::Io {
        path: root.display().to_string(),
        source,
    })?;
    let path = root.join("support-bundle.json");
    let text = serde_json::to_string_pretty(&support_bundle_json(state))
        .map_err(SupportBundleError::Json)?;
    fs::write(&path, format!("{text}\n")).map_err(|source| SupportBundleError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(path)
}

fn support_bundle_json(state: &AppState) -> Value {
    json!({
        "schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
        "redaction": {
            "model_weights_copied": false,
            "environment_copied": false,
            "raw_logs_copied": false,
            "notes": [
                "This bundle records contract status only.",
                "Model weights, environment variables, API keys, and raw process logs are not copied."
            ]
        },
        "doctor": doctor_summary(state),
        "server": server_summary(state),
        "benchmark": benchmark_summary(state),
        "artifacts": artifacts_summary(state),
    })
}

fn doctor_summary(state: &AppState) -> Value {
    match &state.doctor {
        LoadState::Ready(report) => json!({
            "state": "ready",
            "status": report.status,
            "workflow_mode": report.workflow.mode,
            "mlx_runtime_ready": report.mlx_runtime_ready,
            "bringup_allowed": report.bringup_allowed,
            "model_artifacts": {
                "selected": report.model_artifacts.selected,
                "status": report.model_artifacts.status,
                "path_present": report.model_artifacts.path.is_some(),
                "exists": report.model_artifacts.exists,
                "is_dir": report.model_artifacts.is_dir,
                "config_present": report.model_artifacts.config_present,
                "manifest_present": report.model_artifacts.manifest_present,
                "safetensors_present": report.model_artifacts.safetensors_present,
                "model_type": report.model_artifacts.model_type,
                "quantization": report.model_artifacts.quantization.as_ref().map(|quantization| {
                    json!({
                        "mode": quantization.mode,
                        "group_size": quantization.group_size,
                        "bits": quantization.bits,
                    })
                }),
                "issues": report.model_artifacts.issues,
            },
            "issues": report.issues,
            "notes": report.notes,
        }),
        LoadState::Unavailable(message) => json!({"state": "unavailable", "reason": message}),
        LoadState::NotLoaded(message) => json!({"state": "not_loaded", "reason": message}),
    }
}

fn server_summary(state: &AppState) -> Value {
    json!({
        "configured": state.server.base_url.is_some(),
        "health": load_state_label(&state.server.health),
        "runtime": load_state_label(&state.server.runtime),
        "models": load_state_label(&state.server.models),
    })
}

fn benchmark_summary(state: &AppState) -> Value {
    match &state.benchmark_summary {
        LoadState::Ready(summary) => json!({
            "state": "ready",
            "schema_version": summary.schema_version,
            "command": summary.command,
            "status": summary.status,
            "has_result_dir": !summary.result_dir.is_empty(),
            "manifest_present": summary.manifest.is_some(),
            "baseline_present": summary.baseline.is_some(),
            "candidate_present": summary.candidate.is_some(),
        }),
        LoadState::Unavailable(message) => json!({"state": "unavailable", "reason": message}),
        LoadState::NotLoaded(message) => json!({"state": "not_loaded", "reason": message}),
    }
}

fn artifacts_summary(state: &AppState) -> Value {
    match &state.artifacts {
        LoadState::Ready(entries) => json!({
            "state": "ready",
            "count": entries.len(),
            "entries": entries.iter().map(|entry| {
                json!({
                    "name": entry.name,
                    "status": entry.status,
                    "summary_present": entry.summary_present,
                    "contract_failure_present": entry.contract_failure_present,
                })
            }).collect::<Vec<_>>(),
        }),
        LoadState::Unavailable(message) => json!({"state": "unavailable", "reason": message}),
        LoadState::NotLoaded(message) => json!({"state": "not_loaded", "reason": message}),
    }
}

fn load_state_label<T>(state: &LoadState<T>) -> &'static str {
    match state {
        LoadState::Ready(_) => "ready",
        LoadState::Unavailable(_) => "unavailable",
        LoadState::NotLoaded(_) => "not_loaded",
    }
}

#[derive(Debug, Error)]
pub enum SupportBundleError {
    #[error("support bundle IO failed for {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("support bundle JSON failed: {0}")]
    Json(serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{
        DoctorReport, ModelArtifactsReport, QuantizationReport, WorkflowCommand, WorkflowReport,
    };
    use tempfile::tempdir;

    #[test]
    fn support_bundle_records_status_without_copying_sensitive_payloads() {
        let root = tempdir().expect("tempdir should create");
        let mut state = AppState::empty();
        state.doctor = LoadState::Ready(DoctorReport {
            schema_version: "ax.engine_bench.doctor.v1".to_string(),
            status: "ready".to_string(),
            mlx_runtime_ready: true,
            bringup_allowed: true,
            workflow: WorkflowReport {
                mode: "source_checkout".to_string(),
                cwd: "/repo".to_string(),
                source_root: Some("/repo".to_string()),
                doctor: command("doctor"),
                server: command("server"),
                generate_manifest: command("generate-manifest"),
                benchmark: command("scenario"),
                download_model: Some(command("download")),
            },
            model_artifacts: ModelArtifactsReport {
                selected: true,
                status: "ready".to_string(),
                path: Some("/private/model/path".to_string()),
                exists: true,
                is_dir: true,
                config_present: true,
                manifest_present: true,
                safetensors_present: true,
                model_type: Some("qwen3".to_string()),
                quantization: Some(QuantizationReport {
                    mode: "affine".to_string(),
                    group_size: 64,
                    bits: 4,
                }),
                issues: Vec::new(),
            },
            issues: Vec::new(),
            notes: vec!["ready".to_string()],
            performance_advice: Vec::new(),
        });

        let path = write_support_bundle(root.path(), &state).expect("bundle should write");
        let text = fs::read_to_string(&path).expect("bundle should read");

        assert!(text.contains(SUPPORT_BUNDLE_SCHEMA_VERSION));
        assert!(text.contains("\"model_weights_copied\": false"));
        assert!(text.contains("\"path_present\": true"));
        assert!(!text.contains("/private/model/path"));
        assert!(!text.contains("API_KEY"));
        assert!(!root.path().join("model.safetensors").exists());
    }

    fn command(name: &str) -> WorkflowCommand {
        WorkflowCommand {
            argv: vec![name.to_string()],
            cwd: Some("/repo".to_string()),
        }
    }
}
