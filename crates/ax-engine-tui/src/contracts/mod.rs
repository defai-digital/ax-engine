use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {label}: {source}")]
    Parse {
        label: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported {label} schema_version: expected {expected}, got {actual}")]
    Schema {
        label: String,
        expected: &'static str,
        actual: String,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct DoctorReport {
    pub schema_version: String,
    pub status: String,
    pub mlx_runtime_ready: bool,
    pub bringup_allowed: bool,
    pub workflow: WorkflowReport,
    pub model_artifacts: ModelArtifactsReport,
    #[serde(default)]
    pub issues: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub performance_advice: Vec<DoctorAdvice>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct WorkflowReport {
    pub mode: String,
    pub cwd: String,
    pub source_root: Option<String>,
    pub doctor: WorkflowCommand,
    pub server: WorkflowCommand,
    pub generate_manifest: WorkflowCommand,
    pub benchmark: WorkflowCommand,
    pub download_model: Option<WorkflowCommand>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct WorkflowCommand {
    pub argv: Vec<String>,
    pub cwd: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ModelArtifactsReport {
    pub selected: bool,
    pub status: String,
    pub path: Option<String>,
    pub exists: bool,
    pub is_dir: bool,
    pub config_present: bool,
    pub manifest_present: bool,
    pub safetensors_present: bool,
    pub model_type: Option<String>,
    pub quantization: Option<QuantizationReport>,
    #[serde(default)]
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct QuantizationReport {
    pub mode: String,
    pub group_size: u32,
    pub bits: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct DoctorAdvice {
    pub id: String,
    pub severity: String,
    pub summary: String,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct HealthReport {
    pub status: String,
    pub service: String,
    pub model_id: String,
    pub runtime: Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct RuntimeInfoReport {
    pub service: String,
    pub model_id: String,
    pub deterministic: bool,
    pub max_batch_tokens: u32,
    pub block_size_tokens: u32,
    pub runtime: Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ModelsReport {
    pub object: String,
    #[serde(default)]
    pub data: Vec<ModelCard>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub runtime: Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct BenchmarkArtifactSummary {
    pub schema_version: String,
    pub command: String,
    pub result_dir: String,
    pub status: String,
    pub manifest: Option<String>,
    pub baseline: Option<String>,
    pub candidate: Option<String>,
    pub source: Option<String>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactEntry {
    pub path: PathBuf,
    pub name: String,
    pub status: String,
    pub summary_present: bool,
    pub contract_failure_present: bool,
}

pub fn parse_doctor_json(input: &str) -> Result<DoctorReport, ContractError> {
    let report: DoctorReport = parse_json("doctor", input)?;
    ensure_schema(
        "doctor",
        &report.schema_version,
        "ax.engine_bench.doctor.v1",
    )?;
    Ok(report)
}

pub fn parse_health_json(input: &str) -> Result<HealthReport, ContractError> {
    parse_json("server health", input)
}

pub fn parse_runtime_info_json(input: &str) -> Result<RuntimeInfoReport, ContractError> {
    parse_json("server runtime", input)
}

pub fn parse_models_json(input: &str) -> Result<ModelsReport, ContractError> {
    parse_json("server models", input)
}

pub fn parse_benchmark_artifact_json(
    input: &str,
) -> Result<BenchmarkArtifactSummary, ContractError> {
    let summary: BenchmarkArtifactSummary = parse_json("benchmark artifact", input)?;
    ensure_schema(
        "benchmark artifact",
        &summary.schema_version,
        "ax.benchmark_artifact.v1",
    )?;
    Ok(summary)
}

pub fn read_doctor_json(path: &Path) -> Result<DoctorReport, ContractError> {
    parse_doctor_json(&read_to_string(path)?)
}

pub fn read_benchmark_artifact_json(
    path: &Path,
) -> Result<BenchmarkArtifactSummary, ContractError> {
    parse_benchmark_artifact_json(&read_to_string(path)?)
}

pub fn scan_artifacts(root: &Path) -> Result<Vec<ArtifactEntry>, ContractError> {
    let entries = fs::read_dir(root).map_err(|source| ContractError::Read {
        path: root.display().to_string(),
        source,
    })?;
    let mut artifacts = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let summary_present = path.join("summary.md").is_file();
        let contract_failure_present = path.join("contract_failure.json").is_file();
        let metrics_present = path.join("metrics.json").is_file();
        if !summary_present && !contract_failure_present && !metrics_present {
            continue;
        }
        let status = if contract_failure_present {
            "contract_failure"
        } else if metrics_present {
            "run"
        } else {
            "artifact"
        };
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();
        artifacts.push(ArtifactEntry {
            path,
            name,
            status: status.to_string(),
            summary_present,
            contract_failure_present,
        });
    }
    artifacts.sort_by(|left, right| right.name.cmp(&left.name));
    Ok(artifacts)
}

fn parse_json<T>(label: &str, input: &str) -> Result<T, ContractError>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(input).map_err(|source| ContractError::Parse {
        label: label.to_string(),
        source,
    })
}

fn read_to_string(path: &Path) -> Result<String, ContractError> {
    fs::read_to_string(path).map_err(|source| ContractError::Read {
        path: path.display().to_string(),
        source,
    })
}

fn ensure_schema(label: &str, actual: &str, expected: &'static str) -> Result<(), ContractError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ContractError::Schema {
            label: label.to_string(),
            expected,
            actual: actual.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCTOR_JSON: &str = r#"{
      "schema_version": "ax.engine_bench.doctor.v1",
      "status": "ready",
      "mlx_runtime_ready": true,
      "bringup_allowed": true,
      "workflow": {
        "mode": "source_checkout",
        "cwd": "/repo",
        "source_root": "/repo",
        "doctor": {"argv": ["cargo", "run", "-p", "ax-engine-bench", "--", "doctor", "--json"], "cwd": "/repo"},
        "server": {"argv": ["cargo", "run", "-p", "ax-engine-server", "--"], "cwd": "/repo"},
        "generate_manifest": {"argv": ["cargo", "run", "-p", "ax-engine-bench", "--", "generate-manifest", "<model-dir>", "--json"], "cwd": "/repo"},
        "benchmark": {"argv": ["cargo", "run", "-p", "ax-engine-bench", "--", "scenario", "--manifest", "<manifest>", "--output-root", "<output-root>", "--json"], "cwd": "/repo"},
        "download_model": {"argv": ["python3", "scripts/download_model.py", "<repo-id>", "--json"], "cwd": "/repo"}
      },
      "model_artifacts": {
        "selected": true,
        "status": "ready",
        "path": "/models/qwen",
        "exists": true,
        "is_dir": true,
        "config_present": true,
        "manifest_present": true,
        "safetensors_present": true,
        "model_type": "qwen3",
        "quantization": {"mode": "affine", "group_size": 64, "bits": 4},
        "issues": []
      },
      "issues": [],
      "notes": [],
      "performance_advice": []
    }"#;

    #[test]
    fn parses_doctor_contract() {
        let report = parse_doctor_json(DOCTOR_JSON).expect("doctor should parse");

        assert_eq!(report.status, "ready");
        assert_eq!(report.workflow.mode, "source_checkout");
        assert_eq!(report.model_artifacts.model_type.as_deref(), Some("qwen3"));
        assert_eq!(
            report
                .model_artifacts
                .quantization
                .as_ref()
                .map(|quantization| quantization.bits),
            Some(4)
        );
    }

    #[test]
    fn rejects_wrong_doctor_schema() {
        let error = parse_doctor_json(
            r#"{
              "schema_version": "wrong",
              "status": "ready",
              "mlx_runtime_ready": true,
              "bringup_allowed": true,
              "workflow": {"mode":"unknown","cwd":"","doctor":{"argv":[]},"server":{"argv":[]},"generate_manifest":{"argv":[]},"benchmark":{"argv":[]}},
              "model_artifacts": {"selected":false,"status":"not_selected","exists":false,"is_dir":false,"config_present":false,"manifest_present":false,"safetensors_present":false,"issues":[]}
            }"#,
        )
        .expect_err("wrong schema should fail");

        assert!(matches!(error, ContractError::Schema { .. }));
    }

    #[test]
    fn parses_server_contracts() {
        let health = parse_health_json(
            r#"{"status":"ok","service":"ax-engine-server","model_id":"qwen3_dense","runtime":{"selected_backend":"llama_cpp"}}"#,
        )
        .expect("health should parse");
        assert_eq!(health.status, "ok");

        let models = parse_models_json(
            r#"{"object":"list","data":[{"id":"qwen3_dense","object":"model","owned_by":"ax-engine-v4","runtime":{"selected_backend":"llama_cpp"}}]}"#,
        )
        .expect("models should parse");
        assert_eq!(models.data[0].id, "qwen3_dense");
    }
}
