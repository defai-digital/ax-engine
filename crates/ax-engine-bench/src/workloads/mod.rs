//! Agent-workload-shaped fixtures for serving-invariant stress tests.
//!
//! Each fixture under this module implements the [`Workload`] trait so it can
//! be driven uniformly by `scripts/run-serving-stress.sh` (added in PRD
//! Phase 5). Fixtures must skip cleanly when the MLX model artifact directory
//! is not available, mirroring the existing bench convention for
//! `AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ax_engine_sdk::{GenerateSampling, GenerateStreamStepEvent};
use serde_json::{Value, json};

use crate::harness::WorkloadReport;
use crate::inference_args::InferenceArgs;

pub(crate) mod cancellation_during_prefill;
pub(crate) mod concurrent_short_inserts;
pub(crate) mod long_prefill_vs_decode;
pub(crate) mod partial_prefix_hit;
pub(crate) mod post_restart_cache_safety;
pub(crate) mod tool_output_repetition;

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
    pub(crate) fn synthetic() -> Self {
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
    Failed {
        error: String,
        partial: Option<WorkloadReport>,
    },
}

impl WorkloadOutcome {
    /// Stable status label used by tests and (future) Phase 5 aggregation.
    /// Not consumed by the current CLI handler, which serializes the full
    /// JSON envelope.
    #[cfg(test)]
    pub(crate) fn name(&self) -> &'static str {
        match self {
            WorkloadOutcome::Skipped { .. } => "skipped",
            WorkloadOutcome::Completed { .. } => "completed",
            WorkloadOutcome::Failed { .. } => "failed",
        }
    }

    pub(crate) fn to_json(&self) -> Value {
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

/// Shared deterministic MLX inference args for the serving-stress fixtures.
pub(crate) fn workload_inference_args(model_id: &str, dir: &Path) -> InferenceArgs {
    InferenceArgs {
        model_id: model_id.to_string(),
        mlx: true,
        mlx_model_artifacts_dir: Some(dir.to_path_buf()),
        deterministic: true,
        sampling: GenerateSampling::default(),
        ..InferenceArgs::default()
    }
}

/// True when a stream `Step` event carried model output.
///
/// Chunked prefill emits `Step` events with empty `delta_tokens` (the SDK's own
/// `ttft_step` also keys on a non-empty delta), and delegated adapters report
/// text deltas instead of token deltas. Timing chains must key on this signal,
/// not on the bare `Step` event.
pub(crate) fn step_produced_output(step: &GenerateStreamStepEvent) -> bool {
    !step.delta_tokens.is_empty()
        || step
            .delta_text
            .as_deref()
            .is_some_and(|text| !text.is_empty())
}

/// One sample from [`OutputTiming`]: the first output-bearing step measures
/// TTFT, later ones measure inter-token latency.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum OutputSample {
    TimeToFirstOutput(Duration),
    InterToken(Duration),
}

/// TTFT / inter-token timing chain for one streamed request.
///
/// Only output-bearing `Step` events advance the chain: an empty prefill-chunk
/// step must neither record TTFT nor anchor the first ITL gap (that gap would
/// then measure prefill chunking, not decode cadence).
#[derive(Debug, Default)]
pub(crate) struct OutputTiming {
    last_output_at: Option<Instant>,
}

impl OutputTiming {
    /// Feed one `Step` event observed at `now`. Returns `None` when the event
    /// produced no output; otherwise the sample for the caller's channel.
    pub(crate) fn observe(
        &mut self,
        submitted_at: Instant,
        produced_output: bool,
        now: Instant,
    ) -> Option<OutputSample> {
        if !produced_output {
            return None;
        }
        let sample = match self.last_output_at {
            None => OutputSample::TimeToFirstOutput(
                now.checked_duration_since(submitted_at)
                    .unwrap_or(Duration::ZERO),
            ),
            Some(previous) => OutputSample::InterToken(
                now.checked_duration_since(previous)
                    .unwrap_or(Duration::ZERO),
            ),
        };
        self.last_output_at = Some(now);
        Some(sample)
    }
}

#[cfg(test)]
mod tests {
    use ax_engine_sdk::{
        EngineStepReport, GenerateRouteReport, SessionRequestReport, SessionRequestState,
    };

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

    fn step_event(delta_tokens: Vec<u32>, delta_text: Option<&str>) -> GenerateStreamStepEvent {
        GenerateStreamStepEvent {
            request: SessionRequestReport {
                request_id: 1,
                model_id: "test".to_string(),
                state: SessionRequestState::Running,
                prompt_tokens: Vec::new(),
                processed_prompt_tokens: 0,
                output_tokens: Vec::new(),
                output_token_logprobs: Vec::new(),
                prompt_len: 0,
                output_len: 0,
                max_output_tokens: 0,
                cancel_requested: false,
                execution_plan_ref: None,
                route: GenerateRouteReport::default(),
                finish_reason: None,
                terminal_stop_reason: None,
                last_error: None,
            },
            step: EngineStepReport::default(),
            delta_tokens,
            delta_token_logprobs: Vec::new(),
            delta_text: delta_text.map(str::to_string),
        }
    }

    #[test]
    fn step_produced_output_ignores_empty_prefill_chunk_steps() {
        assert!(!step_produced_output(&step_event(Vec::new(), None)));
        assert!(!step_produced_output(&step_event(Vec::new(), Some(""))));
        assert!(step_produced_output(&step_event(vec![7], None)));
        // Delegated adapters report text deltas with empty token deltas.
        assert!(step_produced_output(&step_event(Vec::new(), Some("hi"))));
    }

    #[test]
    fn output_timing_skips_empty_steps_before_first_token() {
        // Chunked prefill emits Step events with no delta_tokens before the
        // first real token. TTFT must land on the first output-bearing step,
        // and the first ITL gap must anchor there too — anchoring on an empty
        // chunk makes the "inter-token" sample a prefill-chunk gap.
        let submitted_at = Instant::now();
        let mut timing = OutputTiming::default();
        let first_chunk = submitted_at + Duration::from_millis(10);
        let second_chunk = submitted_at + Duration::from_millis(20);
        let first_token = submitted_at + Duration::from_millis(40);
        let second_token = submitted_at + Duration::from_millis(55);

        assert_eq!(timing.observe(submitted_at, false, first_chunk), None);
        assert_eq!(timing.observe(submitted_at, false, second_chunk), None);
        assert_eq!(
            timing.observe(submitted_at, true, first_token),
            Some(OutputSample::TimeToFirstOutput(Duration::from_millis(40)))
        );
        assert_eq!(
            timing.observe(submitted_at, true, second_token),
            Some(OutputSample::InterToken(Duration::from_millis(15)))
        );
        // Empty steps between tokens do not perturb the chain either.
        assert_eq!(
            timing.observe(submitted_at, false, second_token + Duration::from_millis(5)),
            None
        );
    }
}
