//! Agent-workload-shaped fixtures for serving-invariant stress tests.
//!
//! Each fixture under this module implements the [`Workload`] trait so it can
//! be driven uniformly by `scripts/run-serving-stress.sh` (added in PRD
//! Phase 5). Fixtures must skip cleanly when the MLX model artifact directory
//! is not available, mirroring the existing bench convention for
//! `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`.

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::harness::WorkloadReport;

pub(crate) mod long_prefill_vs_decode;

/// Environment available to a workload fixture at invocation time.
///
/// Fixtures consult `mlx_model_artifacts_dir` first: when absent, they must
/// return [`WorkloadOutcome::Skipped`] without attempting any inference work.
#[derive(Debug, Clone)]
pub(crate) struct WorkloadContext {
    pub mlx_model_artifacts_dir: Option<PathBuf>,
    pub seed: u64,
}

impl WorkloadContext {
    #[cfg(test)]
    pub fn synthetic() -> Self {
        Self {
            mlx_model_artifacts_dir: None,
            seed: 0,
        }
    }
}

/// Result of running a single fixture.
#[derive(Debug, Clone)]
pub(crate) enum WorkloadOutcome {
    Skipped {
        reason: String,
    },
    Completed {
        report: WorkloadReport,
    },
    #[allow(dead_code)]
    Failed {
        error: String,
        partial: Option<WorkloadReport>,
    },
}

impl WorkloadOutcome {
    /// Stable status label used by tests and (future) Phase 5 aggregation.
    /// Not consumed by the current CLI handler, which serializes the full
    /// JSON envelope.
    #[allow(dead_code)]
    pub fn name(&self) -> &'static str {
        match self {
            WorkloadOutcome::Skipped { .. } => "skipped",
            WorkloadOutcome::Completed { .. } => "completed",
            WorkloadOutcome::Failed { .. } => "failed",
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            WorkloadOutcome::Skipped { reason } => json!({
                "status": "skipped",
                "reason": reason,
            }),
            WorkloadOutcome::Completed { report } => json!({
                "status": "completed",
                "report": report.to_json(),
            }),
            WorkloadOutcome::Failed { error, partial } => json!({
                "status": "failed",
                "error": error,
                "partial": partial.as_ref().map(|r| r.to_json()).unwrap_or(Value::Null),
            }),
        }
    }
}

/// Common interface implemented by every workload fixture.
pub(crate) trait Workload {
    fn name(&self) -> &'static str;
    fn run(&self, ctx: &WorkloadContext) -> WorkloadOutcome;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_context_has_no_artifacts() {
        let ctx = WorkloadContext::synthetic();
        assert!(ctx.mlx_model_artifacts_dir.is_none());
        assert_eq!(ctx.seed, 0);
    }

    #[test]
    fn outcome_skipped_json_shape() {
        let outcome = WorkloadOutcome::Skipped {
            reason: "no artifacts".into(),
        };
        let value = outcome.to_json();
        assert_eq!(value["status"], json!("skipped"));
        assert_eq!(value["reason"], json!("no artifacts"));
    }

    #[test]
    fn outcome_completed_json_carries_report_schema() {
        let report = WorkloadReport::new("dummy");
        let outcome = WorkloadOutcome::Completed { report };
        let value = outcome.to_json();
        assert_eq!(value["status"], json!("completed"));
        assert_eq!(
            value["report"]["schema"],
            json!("ax.serving_workload.report.v1")
        );
    }
}
