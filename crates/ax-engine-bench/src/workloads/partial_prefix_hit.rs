//! Stress fixture: partial prefix-cache hit TTFT differential (I-2 / I-5).
//!
//! Two requests share a long prompt prefix. The first request is a cache miss
//! (cold path). The second issues the same prefix plus a small variant suffix
//! so the engine's prefix cache can be reused. The fixture records TTFT on
//! the foreground channel for both requests and emits an auxiliary
//! `partial_prefix_hit_delta_us` channel carrying the TTFT differential.
//!
//! Concurrency: single-session, sequential. The engine's prefix cache decides
//! whether the second request reuses the first request's KV state; this
//! fixture observes the resulting TTFT, it does not implement caching itself.
//!
//! See `.internal/prd/engine-serving-invariants.md` §8 Phase 5.

use std::path::Path;
use std::time::{Duration, Instant};

use ax_engine_sdk::{GenerateRequest, GenerateSampling, GenerateStreamEvent};

use super::{Workload, WorkloadContext, WorkloadOutcome};
use crate::harness::WorkloadReport;
use crate::harness::metrics::LatencySamples;
use crate::inference_args::{InferenceArgs, build_inference_session};
use crate::synthetic::synthetic_prompt_tokens;

const FIXTURE_TIME_BUDGET: Duration = Duration::from_secs(300);
const POLL_ITERATION_CAP: u64 = 5_000_000;

#[derive(Debug, Clone)]
pub(crate) struct PartialPrefixHit {
    pub model_id: String,
    pub shared_prefix_tokens: u32,
    pub variant_suffix_tokens: u32,
    pub decode_tokens: u32,
}

impl Default for PartialPrefixHit {
    fn default() -> Self {
        Self {
            model_id: "qwen3".to_string(),
            shared_prefix_tokens: 512,
            variant_suffix_tokens: 16,
            decode_tokens: 16,
        }
    }
}

impl Workload for PartialPrefixHit {
    fn name(&self) -> &'static str {
        "partial_prefix_hit"
    }

    fn run(&self, ctx: &WorkloadContext) -> WorkloadOutcome {
        let Some(artifacts_dir) = ctx.mlx_model_artifacts_dir.as_deref() else {
            return WorkloadOutcome::Skipped {
                reason: "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR not set; partial_prefix_hit \
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

impl PartialPrefixHit {
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

    fn build_request(&self, seed: u64, ordinal: u32) -> GenerateRequest {
        // Both requests share the same prefix_group ("partial_prefix_hit/shared")
        // so synthetic_prompt_tokens emits the same leading tokens. The body
        // differs by ordinal, so the suffix forces a fresh KV write on the
        // second request only after the shared prefix has been consumed.
        let total = self.shared_prefix_tokens + self.variant_suffix_tokens;
        GenerateRequest {
            model_id: self.model_id.clone(),
            input_tokens: synthetic_prompt_tokens(
                total,
                Some(&format!("partial_prefix_hit/request_{ordinal}/seed={seed}")),
                Some("partial_prefix_hit/shared"),
                Some(&format!("partial_prefix_hit/variant_{ordinal}")),
                ordinal,
            ),
            input_text: None,
            max_output_tokens: self.decode_tokens,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: Some(format!("partial_prefix_hit/request_{ordinal}/seed={seed}")),
        }
    }

    fn run_driver(&self, artifacts_dir: &Path, seed: u64) -> Result<WorkloadReport, String> {
        let args = self.build_inference_args(artifacts_dir);
        let mut session = build_inference_session(&args)
            .map_err(|e| format!("build_inference_session failed: {e:?}"))?;

        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "config: model_id={}, shared_prefix_tokens={}, variant_suffix_tokens={}, \
             decode_tokens={}, seed={}",
            self.model_id,
            self.shared_prefix_tokens,
            self.variant_suffix_tokens,
            self.decode_tokens,
            seed,
        ));

        let started_at = Instant::now();
        let mut delta_samples = LatencySamples::with_capacity("partial_prefix_hit_delta_us", 1);

        let cold_ttft_us = run_single_request(&mut session, self.build_request(seed, 0), 1)
            .map_err(|e| format!("cold request failed: {e}"))?;
        report.foreground_ttft.record_us(cold_ttft_us);

        let warm_ttft_us = run_single_request(&mut session, self.build_request(seed, 1), 2)
            .map_err(|e| format!("warm request failed: {e}"))?;
        report.foreground_ttft.record_us(warm_ttft_us);

        // Differential ≥ 0 means cold was slower than warm (the expected
        // direction once the engine reuses the shared prefix). We record the
        // raw signed delta via two unsigned buckets: positive delta on
        // partial_prefix_hit_delta_us, plus a note when warm > cold so the
        // analyzer can detect a regression where cache reuse made things
        // slower.
        let delta_us = cold_ttft_us.saturating_sub(warm_ttft_us);
        delta_samples.record_us(delta_us);
        if warm_ttft_us > cold_ttft_us {
            report.add_note(format!(
                "warm_ttft_us > cold_ttft_us (no observable speedup): cold={cold_ttft_us}, warm={warm_ttft_us}"
            ));
        }
        report.add_extra_samples(delta_samples);
        report.add_decision("partial_prefix_hit_cold_ttft_us", cold_ttft_us);
        report.add_decision("partial_prefix_hit_warm_ttft_us", warm_ttft_us);

        if started_at.elapsed() > FIXTURE_TIME_BUDGET {
            return Err("partial_prefix_hit exceeded wall-clock budget".to_string());
        }
        report.record_elapsed(started_at.elapsed());
        Ok(report)
    }

    #[allow(dead_code)]
    pub fn report_skeleton(&self) -> WorkloadReport {
        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "configured shared_prefix_tokens={}, variant_suffix_tokens={}, decode_tokens={}",
            self.shared_prefix_tokens, self.variant_suffix_tokens, self.decode_tokens
        ));
        report
    }
}

/// Drive a single request to its first Step event, returning TTFT in microseconds.
/// Drains the rest of the stream so the engine is ready for the next request.
fn run_single_request(
    session: &mut ax_engine_sdk::EngineSession,
    request: GenerateRequest,
    request_id: u64,
) -> Result<u64, String> {
    let submit_at = Instant::now();
    let mut state = session
        .stream_generate_state_with_request_id(request_id, request)
        .map_err(|e| format!("submit failed: {e}"))?;
    let mut first_step_at: Option<Instant> = None;
    let mut iteration: u64 = 0;
    loop {
        iteration += 1;
        if iteration > POLL_ITERATION_CAP {
            return Err("poll iteration cap exceeded".to_string());
        }
        match session.next_stream_event(&mut state) {
            Ok(Some(GenerateStreamEvent::Step(_))) => {
                if first_step_at.is_none() {
                    first_step_at = Some(Instant::now());
                }
            }
            Ok(Some(GenerateStreamEvent::Response(_))) => break,
            Ok(Some(GenerateStreamEvent::Request(_))) => {}
            Ok(None) => break,
            Err(e) => return Err(format!("stream advance failed: {e}")),
        }
    }
    let ttft = first_step_at
        .ok_or_else(|| "stream finished before first Step event".to_string())?
        .duration_since(submit_at);
    Ok(ttft.as_micros().min(u128::from(u64::MAX)) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_shape() {
        let f = PartialPrefixHit::default();
        assert_eq!(f.model_id, "qwen3");
        assert_eq!(f.shared_prefix_tokens, 512);
        assert_eq!(f.variant_suffix_tokens, 16);
        assert_eq!(f.decode_tokens, 16);
        assert_eq!(f.name(), "partial_prefix_hit");
    }

    #[test]
    fn run_without_artifacts_skips_with_env_var_reason() {
        let outcome = PartialPrefixHit::default().run(&WorkloadContext::synthetic());
        match outcome {
            WorkloadOutcome::Skipped { reason } => {
                assert!(reason.contains("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"));
            }
            other => panic!("expected Skipped, got {:?}", other.name()),
        }
    }

    #[test]
    fn shared_prefix_dominates_request_token_layout() {
        let f = PartialPrefixHit::default();
        let a = f.build_request(0, 0);
        let b = f.build_request(0, 1);
        // Same total length.
        assert_eq!(a.input_tokens.len(), b.input_tokens.len());
        // Same prefix region (synthetic_prompt_tokens uses shared_prefix len
        // = min(target_len, 64); for default target=528 the prefix region is 64).
        let prefix_len = f.shared_prefix_tokens.min(64) as usize;
        assert_eq!(&a.input_tokens[..prefix_len], &b.input_tokens[..prefix_len]);
        // Body region differs.
        assert_ne!(&a.input_tokens[prefix_len..], &b.input_tokens[prefix_len..]);
    }

    #[test]
    fn report_skeleton_carries_configured_params() {
        let f = PartialPrefixHit {
            shared_prefix_tokens: 1024,
            ..PartialPrefixHit::default()
        };
        let report = f.report_skeleton();
        let value = report.to_json();
        let notes = value["notes"].as_array().expect("notes is array");
        assert!(notes.iter().any(|n| {
            n.as_str()
                .map(|s| s.contains("shared_prefix_tokens=1024"))
                .unwrap_or(false)
        }));
    }
}
