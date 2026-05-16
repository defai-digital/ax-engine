use serde::Serialize;
use std::path::Path;

use crate::error::CliError;
use crate::path_utils::path_string;

const BENCHMARK_ARTIFACT_SCHEMA_VERSION: &str = "ax.benchmark_artifact.v1";

#[derive(Debug, Serialize)]
pub(crate) struct BenchmarkArtifactSummary {
    schema_version: &'static str,
    command: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    manifest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    baseline: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    output_root: String,
    result_dir: String,
    status: String,
}

impl BenchmarkArtifactSummary {
    pub(crate) fn manifest_run(
        command: &'static str,
        manifest: &Path,
        output_root: &Path,
        result_dir: &Path,
        status: &str,
    ) -> Self {
        Self {
            schema_version: BENCHMARK_ARTIFACT_SCHEMA_VERSION,
            command,
            manifest: Some(path_string(manifest)),
            baseline: None,
            candidate: None,
            source: None,
            name: None,
            output_root: path_string(output_root),
            result_dir: path_string(result_dir),
            status: status.to_string(),
        }
    }

    pub(crate) fn comparison(
        command: &'static str,
        baseline: &Path,
        candidate: &Path,
        output_root: &Path,
        result_dir: &Path,
    ) -> Self {
        Self {
            schema_version: BENCHMARK_ARTIFACT_SCHEMA_VERSION,
            command,
            manifest: None,
            baseline: Some(path_string(baseline)),
            candidate: Some(path_string(candidate)),
            source: None,
            name: None,
            output_root: path_string(output_root),
            result_dir: path_string(result_dir),
            status: "written".to_string(),
        }
    }

    pub(crate) fn baseline(
        source: &Path,
        name: &str,
        output_root: &Path,
        result_dir: &Path,
    ) -> Self {
        Self {
            schema_version: BENCHMARK_ARTIFACT_SCHEMA_VERSION,
            command: "baseline",
            manifest: None,
            baseline: None,
            candidate: None,
            source: Some(path_string(source)),
            name: Some(name.to_string()),
            output_root: path_string(output_root),
            result_dir: path_string(result_dir),
            status: "written".to_string(),
        }
    }
}

pub(crate) fn print_benchmark_artifact_summary(
    summary: &BenchmarkArtifactSummary,
    json: bool,
) -> Result<(), CliError> {
    if json {
        let json = serde_json::to_string_pretty(summary).map_err(|error| {
            CliError::Runtime(format!(
                "failed to serialize benchmark artifact summary: {error}"
            ))
        })?;
        println!("{json}");
    } else {
        println!("{}", render_benchmark_artifact_summary(summary));
    }
    Ok(())
}

pub(crate) fn render_benchmark_artifact_summary(summary: &BenchmarkArtifactSummary) -> String {
    let mut lines = vec![format!("ax-engine-bench {}", summary.command)];
    if let Some(manifest) = summary.manifest.as_deref() {
        lines.push(format!("manifest={manifest}"));
    }
    if let Some(baseline) = summary.baseline.as_deref() {
        lines.push(format!("baseline={baseline}"));
    }
    if let Some(candidate) = summary.candidate.as_deref() {
        lines.push(format!("candidate={candidate}"));
    }
    if let Some(source) = summary.source.as_deref() {
        lines.push(format!("source={source}"));
    }
    if let Some(name) = summary.name.as_deref() {
        lines.push(format!("name={name}"));
    }
    lines.push(format!("output_root={}", summary.output_root));
    lines.push(format!("result_dir={}", summary.result_dir));
    if summary.status != "written" {
        lines.push(format!("status={}", summary.status));
    }
    lines.join("\n")
}
