use crate::contracts::{DoctorReport, WorkflowCommand};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JobKind {
    DownloadModel,
    GenerateManifest,
    ServerLaunch,
    ServerSmoke,
    BenchmarkScenario,
}

impl JobKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DownloadModel => "download_model",
            Self::GenerateManifest => "generate_manifest",
            Self::ServerLaunch => "server_launch",
            Self::ServerSmoke => "server_smoke",
            Self::BenchmarkScenario => "benchmark_scenario",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceClass {
    Readiness,
    RouteContract,
    WorkloadContract,
}

impl EvidenceClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Readiness => "readiness",
            Self::RouteContract => "route_contract",
            Self::WorkloadContract => "workload_contract",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandInvocation {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
}

impl CommandInvocation {
    pub fn new(program: impl Into<String>, args: Vec<String>, cwd: Option<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args,
            cwd,
        }
    }
}

impl TryFrom<&WorkflowCommand> for CommandInvocation {
    type Error = JobPlanError;

    fn try_from(command: &WorkflowCommand) -> Result<Self, Self::Error> {
        let Some((program, args)) = command.argv.split_first() else {
            return Err(JobPlanError::EmptyCommand);
        };
        Ok(Self {
            program: program.clone(),
            args: args.to_vec(),
            cwd: command.cwd.as_ref().map(PathBuf::from),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobSpec {
    pub id: String,
    pub label: String,
    pub kind: JobKind,
    pub evidence_class: EvidenceClass,
    pub command: CommandInvocation,
    pub owns_process: bool,
    pub artifact_path: Option<PathBuf>,
}

impl JobSpec {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        kind: JobKind,
        evidence_class: EvidenceClass,
        command: CommandInvocation,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind,
            evidence_class,
            command,
            owns_process: true,
            artifact_path: None,
        }
    }

    pub fn with_artifact_path(mut self, artifact_path: impl Into<PathBuf>) -> Self {
        self.artifact_path = Some(artifact_path.into());
        self
    }

    pub fn requires_artifact_path(&self) -> bool {
        matches!(self.kind, JobKind::BenchmarkScenario)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobPlan {
    pub jobs: Vec<JobSpec>,
}

impl JobPlan {
    pub fn from_doctor(report: &DoctorReport) -> Result<Self, JobPlanError> {
        let mut jobs = Vec::new();

        if let Some(command) = report.workflow.download_model.as_ref() {
            jobs.push(JobSpec::new(
                "download-model",
                "Download model",
                JobKind::DownloadModel,
                EvidenceClass::Readiness,
                CommandInvocation::try_from(command)?,
            ));
        }

        jobs.push(JobSpec::new(
            "generate-manifest",
            "Generate model manifest",
            JobKind::GenerateManifest,
            EvidenceClass::Readiness,
            CommandInvocation::try_from(&report.workflow.generate_manifest)?,
        ));

        jobs.push(JobSpec::new(
            "server-launch",
            "Start local server",
            JobKind::ServerLaunch,
            EvidenceClass::RouteContract,
            CommandInvocation::try_from(&report.workflow.server)?,
        ));

        if let Some(source_root) = report.workflow.source_root.as_ref() {
            jobs.push(JobSpec::new(
                "server-smoke",
                "Run server smoke check",
                JobKind::ServerSmoke,
                EvidenceClass::RouteContract,
                CommandInvocation::new(
                    "bash",
                    vec!["scripts/check-server-preview.sh".to_string()],
                    Some(PathBuf::from(source_root)),
                ),
            ));
        }

        jobs.push(JobSpec::new(
            "benchmark-scenario",
            "Run benchmark scenario",
            JobKind::BenchmarkScenario,
            EvidenceClass::WorkloadContract,
            CommandInvocation::try_from(&report.workflow.benchmark)?,
        ));

        Ok(Self { jobs })
    }

    pub fn by_kind(&self, kind: JobKind) -> Option<&JobSpec> {
        self.jobs.iter().find(|job| job.kind == kind)
    }
}

#[derive(Debug, Error)]
pub enum JobPlanError {
    #[error("workflow command argv must include a program")]
    EmptyCommand,
}

#[derive(Debug, Error)]
pub enum JobDisplayError {
    #[error("benchmark job {job_id} cannot display an official result without an artifact path")]
    MissingBenchmarkArtifactPath { job_id: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobDisplaySummary {
    pub job_id: String,
    pub evidence_class: EvidenceClass,
    pub status: String,
    pub artifact_path: Option<PathBuf>,
}

impl JobDisplaySummary {
    pub fn from_completed_job(
        spec: &JobSpec,
        status: impl Into<String>,
    ) -> Result<Self, JobDisplayError> {
        if spec.requires_artifact_path() && spec.artifact_path.is_none() {
            return Err(JobDisplayError::MissingBenchmarkArtifactPath {
                job_id: spec.id.clone(),
            });
        }
        Ok(Self {
            job_id: spec.id.clone(),
            evidence_class: spec.evidence_class.clone(),
            status: status.into(),
            artifact_path: spec.artifact_path.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::parse_doctor_json;

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
    fn builds_explicit_phase2_job_plan_from_doctor_workflow() {
        let doctor = parse_doctor_json(DOCTOR_JSON).expect("doctor should parse");
        let plan = JobPlan::from_doctor(&doctor).expect("plan should build");

        assert_eq!(plan.jobs.len(), 5);
        assert_eq!(
            plan.by_kind(JobKind::DownloadModel)
                .expect("download job")
                .evidence_class,
            EvidenceClass::Readiness
        );
        assert_eq!(
            plan.by_kind(JobKind::ServerLaunch)
                .expect("server job")
                .evidence_class,
            EvidenceClass::RouteContract
        );
        assert_eq!(
            plan.by_kind(JobKind::BenchmarkScenario)
                .expect("benchmark job")
                .evidence_class,
            EvidenceClass::WorkloadContract
        );
    }

    #[test]
    fn benchmark_display_requires_artifact_path() {
        let command = CommandInvocation::new("echo", vec!["ok".to_string()], None);
        let benchmark = JobSpec::new(
            "benchmark",
            "Benchmark",
            JobKind::BenchmarkScenario,
            EvidenceClass::WorkloadContract,
            command,
        );

        assert!(JobDisplaySummary::from_completed_job(&benchmark, "succeeded").is_err());
        assert!(
            JobDisplaySummary::from_completed_job(
                &benchmark.with_artifact_path("/tmp/artifact.json"),
                "succeeded"
            )
            .is_ok()
        );
    }
}
