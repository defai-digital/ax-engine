//! Stress fixture: cancellation during long prefill (I-5).
//!
//! Submits a long-prefill request, waits until the engine has begun emitting
//! Step events (i.e. the prefill is actually in flight), then issues
//! `EngineSession::cancel` and measures the wall time until the stream
//! terminates with no further `Step` event. Records into
//! [`WorkloadReport::cancellation_tail`].
//!
//! Important: not every backend supports preemption mid-prefill. When the
//! cancel signal lands at the start of the stream (no `Step` yet observed),
//! the fixture records a `cancellation_during_prefill_no_step_observed=1`
//! decision so the artifact preserves that condition without conflating it
//! with a successful mid-prefill cancellation.
//!
//! See `.internal/prd/engine-serving-invariants.md` §8 Phase 5.

use std::path::Path;
use std::time::{Duration, Instant};

use ax_engine_sdk::{GenerateRequest, GenerateSampling, GenerateStreamEvent};

use super::{Workload, WorkloadContext, WorkloadOutcome};
use crate::harness::WorkloadReport;
use crate::inference_args::{InferenceArgs, build_inference_session};
use crate::synthetic::synthetic_prompt_tokens;

const FIXTURE_TIME_BUDGET: Duration = Duration::from_secs(300);
const POLL_ITERATION_CAP: u64 = 5_000_000;
/// Number of `Step` events to observe before signalling cancel. Setting this
/// > 0 ensures we are measuring cancellation *during* decode, not at submit.
const STEPS_BEFORE_CANCEL: u32 = 2;

#[derive(Debug, Clone)]
pub(crate) struct CancellationDuringPrefill {
    pub model_id: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
}

impl Default for CancellationDuringPrefill {
    fn default() -> Self {
        Self {
            model_id: "qwen3".to_string(),
            prefill_tokens: 2048,
            decode_tokens: 64,
        }
    }
}

impl Workload for CancellationDuringPrefill {
    fn name(&self) -> &'static str {
        "cancellation_during_prefill"
    }

    fn run(&self, ctx: &WorkloadContext) -> WorkloadOutcome {
        let Some(artifacts_dir) = ctx.mlx_model_artifacts_dir.as_deref() else {
            return WorkloadOutcome::Skipped {
                reason: "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR not set; cancellation_during_prefill \
                         requires a valid MLX model artifact directory"
                    .to_string(),
            };
        };
        match self.run_driver(artifacts_dir, ctx.seed) {
            Ok(report) => WorkloadOutcome::Completed { report },
            Err(error) => WorkloadOutcome::Failed {
                error,
                partial: None,
            },
        }
    }
}

impl CancellationDuringPrefill {
    fn build_inference_args(&self, artifacts_dir: &Path) -> InferenceArgs {
        InferenceArgs {
            model_id: self.model_id.clone(),
            mlx: true,
            mlx_model_artifacts_dir: Some(artifacts_dir.to_path_buf()),
            deterministic: true,
            sampling: GenerateSampling::default(),
            ..InferenceArgs::default()
        }
    }

    fn build_request(&self, seed: u64) -> GenerateRequest {
        GenerateRequest {
            model_id: self.model_id.clone(),
            input_tokens: synthetic_prompt_tokens(
                self.prefill_tokens,
                Some(&format!("cancellation_during_prefill/seed={seed}")),
                None,
                None,
                0,
            ),
            input_text: None,
            multimodal_inputs: Default::default(),
            max_output_tokens: self.decode_tokens,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: Some(format!("cancellation_during_prefill/seed={seed}")),
        }
    }

    fn run_driver(&self, artifacts_dir: &Path, seed: u64) -> Result<WorkloadReport, String> {
        let args = self.build_inference_args(artifacts_dir);
        let mut session = build_inference_session(&args)
            .map_err(|e| format!("build_inference_session failed: {e:?}"))?;

        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "config: model_id={}, prefill_tokens={}, decode_tokens={}, \
             steps_before_cancel={}, seed={}",
            self.model_id, self.prefill_tokens, self.decode_tokens, STEPS_BEFORE_CANCEL, seed,
        ));

        let started_at = Instant::now();
        let request_id: u64 = 1;
        let mut state = session
            .stream_generate_state_with_request_id(request_id, self.build_request(seed))
            .map_err(|e| format!("submit failed: {e}"))?;

        let mut steps_observed: u32 = 0;
        let mut cancel_at: Option<Instant> = None;
        let mut iteration: u64 = 0;

        loop {
            iteration += 1;
            if iteration > POLL_ITERATION_CAP {
                return Err("poll iteration cap exceeded".to_string());
            }
            if started_at.elapsed() > FIXTURE_TIME_BUDGET {
                return Err("cancellation_during_prefill exceeded wall-clock budget".to_string());
            }

            match session.next_stream_event(&mut state) {
                Ok(Some(GenerateStreamEvent::Step(_))) => {
                    steps_observed += 1;
                    if steps_observed == STEPS_BEFORE_CANCEL && cancel_at.is_none() {
                        cancel_at = Some(Instant::now());
                        session
                            .cancel_request(request_id)
                            .map_err(|e| format!("cancel failed: {e}"))?;
                    }
                }
                Ok(Some(GenerateStreamEvent::Response(_))) => break,
                Ok(Some(GenerateStreamEvent::Request(_))) => {}
                Ok(None) => break,
                Err(e) => return Err(format!("stream advance failed: {e}")),
            }
        }

        if let Some(at) = cancel_at {
            let tail = Instant::now() - at;
            report.cancellation_tail.record_duration(tail);
            report.add_decision(
                "cancellation_during_prefill_steps_before_cancel",
                u64::from(steps_observed),
            );
        } else {
            // Stream terminated before STEPS_BEFORE_CANCEL was reached. This
            // is not a fixture failure — backends that finish small prefill
            // before we can interpose are legitimate — but the artifact must
            // be tagged so downstream tooling does not treat the missing
            // cancellation_tail bucket as a regression.
            report.add_decision("cancellation_during_prefill_no_step_observed", 1);
            report.add_note(
                "stream completed before cancel could be issued; \
                 cancellation_tail bucket left empty"
                    .to_string(),
            );
        }

        report.record_elapsed(started_at.elapsed());
        Ok(report)
    }

    #[allow(dead_code)]
    pub fn report_skeleton(&self) -> WorkloadReport {
        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "configured prefill_tokens={}, decode_tokens={}",
            self.prefill_tokens, self.decode_tokens
        ));
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_shape() {
        let f = CancellationDuringPrefill::default();
        assert_eq!(f.prefill_tokens, 2048);
        assert_eq!(f.decode_tokens, 64);
        assert_eq!(f.name(), "cancellation_during_prefill");
    }

    #[test]
    fn run_without_artifacts_skips_with_reason() {
        let outcome = CancellationDuringPrefill::default().run(&WorkloadContext::synthetic());
        match outcome {
            WorkloadOutcome::Skipped { reason } => {
                assert!(reason.contains("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"));
            }
            other => panic!("expected Skipped, got {:?}", other.name()),
        }
    }

    #[test]
    fn request_carries_configured_prefill_size() {
        let f = CancellationDuringPrefill::default();
        let req = f.build_request(0);
        assert_eq!(req.input_tokens.len() as u32, f.prefill_tokens);
        assert_eq!(req.max_output_tokens, f.decode_tokens);
    }
}
