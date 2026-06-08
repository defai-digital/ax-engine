//! Stress fixture: repeated tool-output-shaped prompts (I-5, I-6 input).
//!
//! Coding agents frequently re-issue near-identical tool-output prompts back
//! into the model. This fixture exercises that pattern by submitting the
//! same short prompt N times in sequence. It is the natural exerciser of
//! the n-gram acceleration path: repeated prompts produce repeated
//! continuation tokens, and the engine's n-gram drafter should land
//! progressively higher acceptance counts.
//!
//! The fixture records:
//! - aggregate ITL across all decode steps as `foreground_itl`
//! - per-iteration TTFT as `foreground_ttft`
//! - n-gram draft/accept counters via route decisions
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

#[derive(Debug, Clone)]
pub(crate) struct ToolOutputRepetition {
    pub model_id: String,
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub iterations: u32,
}

impl Default for ToolOutputRepetition {
    fn default() -> Self {
        Self {
            model_id: "qwen3".to_string(),
            prompt_tokens: 128,
            decode_tokens: 32,
            iterations: 4,
        }
    }
}

impl Workload for ToolOutputRepetition {
    fn name(&self) -> &'static str {
        "tool_output_repetition"
    }

    fn run(&self, ctx: &WorkloadContext) -> WorkloadOutcome {
        let Some(artifacts_dir) = ctx.mlx_model_artifacts_dir.as_deref() else {
            return WorkloadOutcome::Skipped {
                reason: "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR not set; tool_output_repetition \
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

impl ToolOutputRepetition {
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

    fn build_request(&self, seed: u64, iteration: u32) -> GenerateRequest {
        // Every iteration uses the same prompt_ref and ordinal, so
        // synthetic_prompt_tokens returns identical tokens iteration over
        // iteration. This is the explicit exerciser for the n-gram cache.
        GenerateRequest {
            model_id: self.model_id.clone(),
            input_tokens: synthetic_prompt_tokens(
                self.prompt_tokens,
                Some(&format!("tool_output_repetition/seed={seed}")),
                None,
                None,
                0,
            ),
            input_text: None,
            multimodal_inputs: Default::default(),
            max_output_tokens: self.decode_tokens,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: Some(format!(
                "tool_output_repetition/iter={iteration}/seed={seed}"
            )),
        }
    }

    fn run_driver(&self, artifacts_dir: &Path, seed: u64) -> Result<WorkloadReport, String> {
        let args = self.build_inference_args(artifacts_dir);
        let mut session = build_inference_session(&args)
            .map_err(|e| format!("build_inference_session failed: {e:?}"))?;

        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "config: model_id={}, prompt_tokens={}, decode_tokens={}, iterations={}, seed={}",
            self.model_id, self.prompt_tokens, self.decode_tokens, self.iterations, seed,
        ));

        let started_at = Instant::now();
        let mut accumulated_accepted: u64 = 0;
        let mut accumulated_attempts: u64 = 0;

        for iteration in 0..self.iterations {
            if started_at.elapsed() > FIXTURE_TIME_BUDGET {
                return Err(format!(
                    "tool_output_repetition exceeded wall-clock budget at iteration {iteration}"
                ));
            }
            let request = self.build_request(seed, iteration);
            let request_id = u64::from(iteration) + 1;
            let submit_at = Instant::now();
            let mut state = session
                .stream_generate_state_with_request_id(request_id, request)
                .map_err(|e| format!("submit iteration {iteration} failed: {e}"))?;

            let mut first_step_at: Option<Instant> = None;
            let mut last_step_at: Option<Instant> = None;
            let mut iter_count: u64 = 0;
            loop {
                iter_count += 1;
                if iter_count > POLL_ITERATION_CAP {
                    return Err(format!("iteration {iteration} exceeded poll cap"));
                }
                match session.next_stream_event(&mut state) {
                    Ok(Some(GenerateStreamEvent::Step(_))) => {
                        let now = Instant::now();
                        if first_step_at.is_none() {
                            first_step_at = Some(now);
                            report.foreground_ttft.record_duration(now - submit_at);
                        } else if let Some(prev) = last_step_at {
                            report.foreground_itl.record_duration(now - prev);
                        }
                        last_step_at = Some(now);
                    }
                    Ok(Some(GenerateStreamEvent::Response(response))) => {
                        // Aggregate n-gram counters across iterations so the
                        // artifact captures the cumulative acceleration effect.
                        if let Some(attempts) =
                            response.response.route.decision("ax_ngram_draft_attempts")
                        {
                            accumulated_attempts =
                                accumulated_attempts.saturating_add(u64::from(attempts));
                        }
                        if let Some(accepted) =
                            response.response.route.decision("ax_ngram_accepted_tokens")
                        {
                            accumulated_accepted =
                                accumulated_accepted.saturating_add(u64::from(accepted));
                        }
                        break;
                    }
                    Ok(Some(GenerateStreamEvent::Request(_))) => {}
                    Ok(None) => break,
                    Err(e) => {
                        return Err(format!("iteration {iteration} stream advance failed: {e}"));
                    }
                }
            }
        }

        report.add_decision(
            "tool_output_repetition_ngram_draft_attempts_total",
            accumulated_attempts,
        );
        report.add_decision(
            "tool_output_repetition_ngram_accepted_tokens_total",
            accumulated_accepted,
        );
        report.record_elapsed(started_at.elapsed());
        Ok(report)
    }

    #[allow(dead_code)]
    pub fn report_skeleton(&self) -> WorkloadReport {
        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "configured prompt_tokens={}, decode_tokens={}, iterations={}",
            self.prompt_tokens, self.decode_tokens, self.iterations
        ));
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_shape() {
        let f = ToolOutputRepetition::default();
        assert_eq!(f.iterations, 4);
        assert_eq!(f.name(), "tool_output_repetition");
    }

    #[test]
    fn run_without_artifacts_skips_with_reason() {
        let outcome = ToolOutputRepetition::default().run(&WorkloadContext::synthetic());
        match outcome {
            WorkloadOutcome::Skipped { reason } => {
                assert!(reason.contains("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"));
            }
            other => panic!("expected Skipped, got {:?}", other.name()),
        }
    }

    #[test]
    fn repeated_iterations_produce_identical_tokens() {
        // Critical correctness property: this fixture's whole point is to
        // submit the *same* prompt repeatedly so the n-gram drafter has
        // observable repetition to learn from. If iteration changed the token
        // stream, the fixture would stop measuring what its name claims.
        let f = ToolOutputRepetition::default();
        let a = f.build_request(0, 0);
        let b = f.build_request(0, 3);
        assert_eq!(a.input_tokens, b.input_tokens);
        // Metadata still differs so artifact provenance distinguishes runs.
        assert_ne!(a.metadata, b.metadata);
    }

    #[test]
    fn seed_differentiates_token_streams_across_runs() {
        let f = ToolOutputRepetition::default();
        let a = f.build_request(0, 0);
        let b = f.build_request(1, 0);
        assert_ne!(a.input_tokens, b.input_tokens);
    }
}
