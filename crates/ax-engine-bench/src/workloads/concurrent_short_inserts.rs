//! Stress fixture: short-request TTFT under concurrent long-request load
//! (I-1 / I-5).
//!
//! Submits a long-running prefill+decode pair, then injects N short requests
//! sequentially after the long request is in flight. Records each short
//! request's TTFT on the [`ConcurrentShortInsertDelta`] channel as the
//! degradation versus a near-zero baseline (the first short request's TTFT
//! is captured before further short submissions, then subsequent inserts
//! report `(this_ttft - baseline_short_ttft)`).
//!
//! [`ConcurrentShortInsertDelta`]: crate::harness::metrics::LatencyChannel::ConcurrentShortInsertDelta
//!
//! See `.internal/prd/engine-serving-invariants.md` §8 Phase 5.

use std::path::Path;
use std::time::{Duration, Instant};

use ax_engine_sdk::{GenerateRequest, GenerateSampling, GenerateStreamEvent, GenerateStreamState};

use super::{Workload, WorkloadContext, WorkloadOutcome};
use crate::harness::WorkloadReport;
use crate::inference_args::{InferenceArgs, build_inference_session};
use crate::synthetic::synthetic_prompt_tokens;

const FIXTURE_TIME_BUDGET: Duration = Duration::from_secs(300);
const POLL_ITERATION_CAP: u64 = 10_000_000;

#[derive(Debug, Clone)]
pub(crate) struct ConcurrentShortInserts {
    pub model_id: String,
    pub long_prefill_tokens: u32,
    pub long_decode_tokens: u32,
    pub short_prefix_tokens: u32,
    pub short_decode_tokens: u32,
    pub short_request_count: u32,
}

impl Default for ConcurrentShortInserts {
    fn default() -> Self {
        Self {
            model_id: "qwen3".to_string(),
            long_prefill_tokens: 2048,
            long_decode_tokens: 256,
            short_prefix_tokens: 32,
            short_decode_tokens: 8,
            short_request_count: 4,
        }
    }
}

impl Workload for ConcurrentShortInserts {
    fn name(&self) -> &'static str {
        "concurrent_short_inserts"
    }

    fn run(&self, ctx: &WorkloadContext) -> WorkloadOutcome {
        let Some(artifacts_dir) = ctx.mlx_model_artifacts_dir.as_deref() else {
            return WorkloadOutcome::Skipped {
                reason: "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR not set; concurrent_short_inserts \
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

impl ConcurrentShortInserts {
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

    fn build_long_request(&self, seed: u64) -> GenerateRequest {
        GenerateRequest {
            model_id: self.model_id.clone(),
            input_tokens: synthetic_prompt_tokens(
                self.long_prefill_tokens,
                Some(&format!("concurrent_short_inserts/long/seed={seed}")),
                None,
                0,
                None,
                0,
            ),
            input_text: None,
            multimodal_inputs: Default::default(),
            max_output_tokens: self.long_decode_tokens,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: Some(format!("concurrent_short_inserts/long/seed={seed}")),
        }
    }

    fn build_short_request(&self, seed: u64, ordinal: u32) -> GenerateRequest {
        GenerateRequest {
            model_id: self.model_id.clone(),
            input_tokens: synthetic_prompt_tokens(
                self.short_prefix_tokens,
                Some(&format!(
                    "concurrent_short_inserts/short_{ordinal}/seed={seed}"
                )),
                None,
                0,
                None,
                ordinal,
            ),
            input_text: None,
            multimodal_inputs: Default::default(),
            max_output_tokens: self.short_decode_tokens,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: Some(format!(
                "concurrent_short_inserts/short_{ordinal}/seed={seed}"
            )),
        }
    }

    fn run_driver(&self, artifacts_dir: &Path, seed: u64) -> Result<WorkloadReport, String> {
        let args = self.build_inference_args(artifacts_dir);
        let mut session = build_inference_session(&args)
            .map_err(|e| format!("build_inference_session failed: {e:?}"))?;

        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "config: long_prefill_tokens={}, long_decode_tokens={}, \
             short_prefix_tokens={}, short_decode_tokens={}, short_request_count={}, seed={}",
            self.long_prefill_tokens,
            self.long_decode_tokens,
            self.short_prefix_tokens,
            self.short_decode_tokens,
            self.short_request_count,
            seed,
        ));

        let started_at = Instant::now();
        let mut long_state = session
            .stream_generate_state_with_request_id(1, self.build_long_request(seed))
            .map_err(|e| format!("submit long failed: {e}"))?;

        // Drive the long request until its first Step event lands. After
        // that point we know the engine has prefill work in flight and the
        // short inserts will compete for batching attention.
        wait_for_first_step(&mut session, &mut long_state)?;

        let mut baseline_short_ttft_us: Option<u64> = None;
        for ordinal in 0..self.short_request_count {
            if started_at.elapsed() > FIXTURE_TIME_BUDGET {
                return Err(format!(
                    "concurrent_short_inserts exceeded wall-clock budget at ordinal {ordinal}"
                ));
            }
            let short_request_id = u64::from(ordinal) + 2;
            let submit_at = Instant::now();
            let mut short_state = session
                .stream_generate_state_with_request_id(
                    short_request_id,
                    self.build_short_request(seed, ordinal),
                )
                .map_err(|e| format!("submit short_{ordinal} failed: {e}"))?;

            let first_step_at = drive_to_first_step(&mut session, &mut short_state)
                .map_err(|e| format!("drive short_{ordinal} to first step: {e}"))?;
            let ttft_us = (first_step_at - submit_at)
                .as_micros()
                .min(u128::from(u64::MAX)) as u64;
            report.foreground_ttft.record_us(ttft_us);

            match baseline_short_ttft_us {
                None => {
                    baseline_short_ttft_us = Some(ttft_us);
                }
                Some(baseline) => {
                    let delta = ttft_us.saturating_sub(baseline);
                    report.concurrent_short_insert_delta.record_us(delta);
                }
            }
            // Drain the short request to completion so the next iteration
            // sees a clean scheduler state for its insert.
            drain_to_response(&mut session, &mut short_state)
                .map_err(|e| format!("drain short_{ordinal}: {e}"))?;
        }

        // Drain the long request so the artifact captures its decisions.
        drain_to_response(&mut session, &mut long_state).map_err(|e| format!("drain long: {e}"))?;

        report.record_elapsed(started_at.elapsed());
        if let Some(baseline) = baseline_short_ttft_us {
            report.add_decision("concurrent_short_inserts_baseline_short_ttft_us", baseline);
        }
        Ok(report)
    }

    #[allow(dead_code)]
    pub fn report_skeleton(&self) -> WorkloadReport {
        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "configured short_request_count={}, long_prefill_tokens={}",
            self.short_request_count, self.long_prefill_tokens
        ));
        report
    }
}

fn wait_for_first_step(
    session: &mut ax_engine_sdk::EngineSession,
    state: &mut GenerateStreamState,
) -> Result<(), String> {
    let mut iteration: u64 = 0;
    loop {
        iteration += 1;
        if iteration > POLL_ITERATION_CAP {
            return Err("wait_for_first_step exceeded poll cap".to_string());
        }
        match session.next_stream_event(state) {
            Ok(Some(GenerateStreamEvent::Step(_))) => return Ok(()),
            Ok(Some(GenerateStreamEvent::Response(_))) => return Ok(()),
            Ok(Some(GenerateStreamEvent::Request(_))) => continue,
            Ok(None) => return Ok(()),
            Err(e) => return Err(format!("stream advance: {e}")),
        }
    }
}

fn drive_to_first_step(
    session: &mut ax_engine_sdk::EngineSession,
    state: &mut GenerateStreamState,
) -> Result<Instant, String> {
    let mut iteration: u64 = 0;
    loop {
        iteration += 1;
        if iteration > POLL_ITERATION_CAP {
            return Err("drive_to_first_step exceeded poll cap".to_string());
        }
        match session.next_stream_event(state) {
            Ok(Some(GenerateStreamEvent::Step(_))) => return Ok(Instant::now()),
            Ok(Some(GenerateStreamEvent::Response(_))) => {
                // Completed without a Step event — treat completion timestamp
                // as the first-step proxy so the channel still gets a sample.
                return Ok(Instant::now());
            }
            Ok(Some(GenerateStreamEvent::Request(_))) => continue,
            Ok(None) => return Err("stream ended before first step".to_string()),
            Err(e) => return Err(format!("stream advance: {e}")),
        }
    }
}

fn drain_to_response(
    session: &mut ax_engine_sdk::EngineSession,
    state: &mut GenerateStreamState,
) -> Result<(), String> {
    let mut iteration: u64 = 0;
    loop {
        iteration += 1;
        if iteration > POLL_ITERATION_CAP {
            return Err("drain_to_response exceeded poll cap".to_string());
        }
        match session.next_stream_event(state) {
            Ok(Some(GenerateStreamEvent::Response(_))) => return Ok(()),
            Ok(Some(GenerateStreamEvent::Step(_) | GenerateStreamEvent::Request(_))) => continue,
            Ok(None) => return Ok(()),
            Err(e) => return Err(format!("stream advance: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_shape() {
        let f = ConcurrentShortInserts::default();
        assert_eq!(f.name(), "concurrent_short_inserts");
        assert_eq!(f.short_request_count, 4);
    }

    #[test]
    fn run_without_artifacts_skips_with_reason() {
        let outcome = ConcurrentShortInserts::default().run(&WorkloadContext::synthetic());
        match outcome {
            WorkloadOutcome::Skipped { reason } => {
                assert!(reason.contains("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"));
            }
            other => panic!("expected Skipped, got {:?}", other.name()),
        }
    }

    #[test]
    fn short_requests_differ_by_ordinal() {
        let f = ConcurrentShortInserts::default();
        let a = f.build_short_request(0, 0);
        let b = f.build_short_request(0, 1);
        assert_ne!(a.input_tokens, b.input_tokens);
        assert_ne!(a.metadata, b.metadata);
    }

    #[test]
    fn long_request_input_matches_prefill_tokens() {
        let f = ConcurrentShortInserts::default();
        let req = f.build_long_request(0);
        assert_eq!(req.input_tokens.len() as u32, f.long_prefill_tokens);
    }
}
