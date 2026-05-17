//! Stress fixture: foreground decode latency under concurrent long-prefill load.
//!
//! Invariant I-1 (ADR-007): a long-prefill request entering the queue must not
//! degrade the inter-token latency or TTFT of in-flight decode requests beyond
//! a documented bound. This fixture is the measurement vehicle for that
//! invariant.
//!
//! Concurrency model: AX Engine's `EngineSession::next_stream_event` takes
//! `&mut self` per call, so multi-request driving is single-threaded
//! round-robin polling over multiple `GenerateStreamState` handles. The
//! engine's internal scheduler decides how to batch prefill and decode work
//! across submitted requests; this fixture observes the resulting latency
//! distribution, it does not implement batching itself.
//!
//! See `.internal/prd/engine-serving-invariants.md` §8 Phase 1 and
//! `.internal/adr/ADR-007-engine-serving-invariants.md` I-1.

use std::path::Path;
use std::time::{Duration, Instant};

use ax_engine_sdk::{GenerateRequest, GenerateSampling, GenerateStreamEvent, GenerateStreamState};

use super::{Workload, WorkloadContext, WorkloadOutcome};
use crate::harness::WorkloadReport;
use crate::inference_args::{InferenceArgs, build_inference_session};
use crate::synthetic::synthetic_prompt_tokens;

/// Wall-clock budget that bounds a single fixture run. Exceeding this aborts
/// the driver and produces a `Failed` outcome carrying the partial report.
const FIXTURE_TIME_BUDGET: Duration = Duration::from_secs(600);

/// Iteration cap on the polling loop to prevent runaway behavior if the engine
/// reports neither `Step` nor `Response` events for an active state.
const POLL_ITERATION_CAP: u64 = 50_000_000;

#[derive(Debug, Clone)]
pub(crate) struct LongPrefillVsDecode {
    pub model_id: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub concurrent_short_requests: u32,
    pub short_prefix_tokens: u32,
}

impl Default for LongPrefillVsDecode {
    fn default() -> Self {
        Self {
            // Defaults sized to surface real prefill cost without exceeding a
            // typical artifact's modeled context window. Baseline runs may
            // override these via the CLI subcommand.
            model_id: "qwen3".to_string(),
            prefill_tokens: 4096,
            decode_tokens: 128,
            concurrent_short_requests: 4,
            short_prefix_tokens: 64,
        }
    }
}

impl Workload for LongPrefillVsDecode {
    fn name(&self) -> &'static str {
        "long_prefill_vs_decode"
    }

    fn run(&self, ctx: &WorkloadContext) -> WorkloadOutcome {
        let Some(artifacts_dir) = ctx.mlx_model_artifacts_dir.as_deref() else {
            return WorkloadOutcome::Skipped {
                reason: "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR not set; long_prefill_vs_decode \
                         requires a valid MLX model artifact directory"
                    .to_string(),
            };
        };

        match self.run_concurrent_driver(artifacts_dir, ctx.seed) {
            Ok(report) => WorkloadOutcome::Completed { report },
            Err(FixtureError {
                message,
                partial_report,
            }) => WorkloadOutcome::Failed {
                error: message,
                partial: partial_report.map(|boxed| *boxed),
            },
        }
    }
}

#[derive(Debug)]
struct FixtureError {
    message: String,
    // Boxed to keep `Result<WorkloadReport, FixtureError>` from triggering
    // `clippy::result_large_err`. `WorkloadReport` carries several
    // `LatencySamples` plus a notes/decisions vector, which makes the
    // unboxed Err variant noticeably larger than Ok.
    partial_report: Option<Box<WorkloadReport>>,
}

impl FixtureError {
    fn bare(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            partial_report: None,
        }
    }

    fn with_partial(message: impl Into<String>, report: WorkloadReport) -> Self {
        Self {
            message: message.into(),
            partial_report: Some(Box::new(report)),
        }
    }
}

impl LongPrefillVsDecode {
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

    fn build_long_prefill_request(&self, seed: u64) -> GenerateRequest {
        GenerateRequest {
            model_id: self.model_id.clone(),
            // The long-prefill request keeps its decode budget tiny so the
            // measurement focuses on prefill cost, not on its own decode
            // contention with the foreground requests.
            input_tokens: synthetic_prompt_tokens(
                self.prefill_tokens,
                Some(&format!("long_prefill_vs_decode/long/seed={seed}")),
                None,
                None,
                0,
            ),
            input_text: None,
            max_output_tokens: 4,
            sampling: GenerateSampling::default(),
            stop_sequences: Vec::new(),
            metadata: Some(format!("long_prefill_vs_decode/long_prefill/seed={seed}")),
        }
    }

    fn build_short_decode_requests(&self, seed: u64) -> Vec<GenerateRequest> {
        (0..self.concurrent_short_requests as usize)
            .map(|i| GenerateRequest {
                model_id: self.model_id.clone(),
                input_tokens: synthetic_prompt_tokens(
                    self.short_prefix_tokens,
                    Some(&format!("long_prefill_vs_decode/short_{i}/seed={seed}")),
                    None,
                    None,
                    i as u32,
                ),
                input_text: None,
                max_output_tokens: self.decode_tokens,
                sampling: GenerateSampling::default(),
                stop_sequences: Vec::new(),
                metadata: Some(format!(
                    "long_prefill_vs_decode/short_decode_{i}/seed={seed}"
                )),
            })
            .collect()
    }

    fn run_concurrent_driver(
        &self,
        artifacts_dir: &Path,
        seed: u64,
    ) -> Result<WorkloadReport, FixtureError> {
        let inference_args = self.build_inference_args(artifacts_dir);
        let mut session = build_inference_session(&inference_args).map_err(|error| {
            FixtureError::bare(format!(
                "build_inference_session failed for long_prefill_vs_decode: {error:?}"
            ))
        })?;

        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "config: model_id={}, prefill_tokens={}, decode_tokens={}, \
             concurrent_short_requests={}, short_prefix_tokens={}, seed={}",
            self.model_id,
            self.prefill_tokens,
            self.decode_tokens,
            self.concurrent_short_requests,
            self.short_prefix_tokens,
            seed,
        ));

        let started_at = Instant::now();

        // Submit the long-prefill request first so the scheduler sees prefill
        // work in flight when the short decode requests arrive.
        let long_request = self.build_long_prefill_request(seed);
        let long_submit_at = Instant::now();
        let mut long_state = session
            .stream_generate_state_with_request_id(1, long_request)
            .map_err(|error| {
                FixtureError::with_partial(
                    format!("submit long_prefill stream state failed: {error}"),
                    report.clone(),
                )
            })?;

        // Submit foreground decode requests immediately after, capturing
        // per-request submission instants for accurate TTFT.
        let short_requests = self.build_short_decode_requests(seed);
        let mut short_handles: Vec<ShortHandle> = Vec::with_capacity(short_requests.len());
        for (i, request) in short_requests.into_iter().enumerate() {
            let request_id = (i as u64).saturating_add(2);
            let submit_at = Instant::now();
            let state = session
                .stream_generate_state_with_request_id(request_id, request)
                .map_err(|error| {
                    FixtureError::with_partial(
                        format!("submit short_decode_{i} stream state failed: {error}"),
                        report.clone(),
                    )
                })?;
            short_handles.push(ShortHandle {
                request_id,
                state,
                submit_at,
                last_step_at: None,
                active: true,
                first_step_seen: false,
            });
        }

        let mut long_active = true;
        let mut iteration: u64 = 0;
        let mut last_step_long: Option<Instant> = None;

        loop {
            if !long_active && short_handles.iter().all(|h| !h.active) {
                break;
            }
            if started_at.elapsed() > FIXTURE_TIME_BUDGET {
                report.add_note(format!(
                    "aborted after {:?}: exceeded FIXTURE_TIME_BUDGET",
                    started_at.elapsed()
                ));
                return Err(FixtureError::with_partial(
                    "long_prefill_vs_decode exceeded wall-clock budget".to_string(),
                    finalize_report(report, started_at, &short_handles, long_submit_at),
                ));
            }
            iteration = iteration.saturating_add(1);
            if iteration > POLL_ITERATION_CAP {
                report.add_note(format!(
                    "aborted after {} iterations: exceeded POLL_ITERATION_CAP",
                    iteration
                ));
                return Err(FixtureError::with_partial(
                    "long_prefill_vs_decode exceeded poll iteration cap".to_string(),
                    finalize_report(report, started_at, &short_handles, long_submit_at),
                ));
            }

            // Drive the long-prefill request once per loop iteration.
            if long_active {
                match session.next_stream_event(&mut long_state) {
                    Ok(Some(GenerateStreamEvent::Step(_))) => {
                        last_step_long = Some(Instant::now());
                    }
                    Ok(Some(GenerateStreamEvent::Response(response))) => {
                        long_active = false;
                        capture_route_decisions(&mut report, &response, "long_prefill");
                    }
                    Ok(Some(GenerateStreamEvent::Request(_))) => {
                        // Initial request event carries no per-token timing.
                    }
                    Ok(None) => {
                        long_active = false;
                    }
                    Err(error) => {
                        return Err(FixtureError::with_partial(
                            format!("long_prefill stream advance failed: {error}"),
                            finalize_report(report, started_at, &short_handles, long_submit_at),
                        ));
                    }
                }
            }

            // Drive each foreground short-decode request once.
            for handle in short_handles.iter_mut() {
                if !handle.active {
                    continue;
                }
                match session.next_stream_event(&mut handle.state) {
                    Ok(Some(GenerateStreamEvent::Step(_))) => {
                        let now = Instant::now();
                        if !handle.first_step_seen {
                            let ttft = now - handle.submit_at;
                            report.foreground_ttft.record_duration(ttft);
                            handle.first_step_seen = true;
                        } else if let Some(prev) = handle.last_step_at {
                            let itl = now - prev;
                            report.foreground_itl.record_duration(itl);
                        }
                        handle.last_step_at = Some(now);
                    }
                    Ok(Some(GenerateStreamEvent::Response(response))) => {
                        handle.active = false;
                        capture_route_decisions(
                            &mut report,
                            &response,
                            &format!("short_decode_{}", handle.request_id),
                        );
                    }
                    Ok(Some(GenerateStreamEvent::Request(_))) => {
                        // Initial request event; no timing yet.
                    }
                    Ok(None) => {
                        handle.active = false;
                    }
                    Err(error) => {
                        return Err(FixtureError::with_partial(
                            format!(
                                "short_decode (request_id={}) stream advance failed: {error}",
                                handle.request_id
                            ),
                            finalize_report(report, started_at, &short_handles, long_submit_at),
                        ));
                    }
                }
            }
        }

        // Suppress unused-variable warning when long-prefill produces no Step events.
        let _ = last_step_long;

        Ok(finalize_report(
            report,
            started_at,
            &short_handles,
            long_submit_at,
        ))
    }

    /// Convenience helper for downstream callers that want a populated report
    /// shape even when the fixture itself short-circuits. The CLI handler
    /// does not use this; tests and Phase 1c baseline tooling may.
    #[allow(dead_code)]
    pub fn report_skeleton(&self) -> WorkloadReport {
        let mut report = WorkloadReport::new(self.name());
        report.add_note(format!(
            "configured prefill_tokens={}, decode_tokens={}, concurrent_short_requests={}",
            self.prefill_tokens, self.decode_tokens, self.concurrent_short_requests,
        ));
        report
    }
}

struct ShortHandle {
    request_id: u64,
    state: GenerateStreamState,
    submit_at: Instant,
    last_step_at: Option<Instant>,
    active: bool,
    first_step_seen: bool,
}

fn finalize_report(
    mut report: WorkloadReport,
    started_at: Instant,
    short_handles: &[ShortHandle],
    long_submit_at: Instant,
) -> WorkloadReport {
    report.record_elapsed(started_at.elapsed());
    report.add_note(format!(
        "long_prefill_request submit_to_now_us={}, short_decode_request_count={}",
        instant_delta_us(long_submit_at, Instant::now()),
        short_handles.len()
    ));
    report
}

fn instant_delta_us(from: Instant, to: Instant) -> u64 {
    to.checked_duration_since(from)
        .unwrap_or(Duration::ZERO)
        .as_micros()
        .min(u128::from(u64::MAX)) as u64
}

fn capture_route_decisions(
    report: &mut WorkloadReport,
    response: &ax_engine_sdk::GenerateStreamResponseEvent,
    label: &str,
) {
    // Record a small set of route decisions per terminal response. These keys
    // are stable contract surface in `RouteMetadata.crossover_decisions` and
    // are used by downstream baseline tooling to correlate latency with
    // batching/KV behavior.
    for key in [
        "ax_mlx_kv_logical_tokens",
        "ax_mlx_kv_capacity_tokens",
        "ax_mlx_kv_full_attention_layers",
        "ax_mlx_kv_sliding_window_layers",
        "ax_ngram_draft_attempts",
        "ax_ngram_accepted_tokens",
        "ax_ngram_no_draft_steps",
    ] {
        if let Some(value) = response.response.route.decision(key) {
            report.add_decision(format!("{label}/{key}"), u64::from(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn defaults_match_phase_one_documented_shape() {
        let fixture = LongPrefillVsDecode::default();
        assert_eq!(fixture.model_id, "qwen3");
        assert_eq!(fixture.prefill_tokens, 4096);
        assert_eq!(fixture.decode_tokens, 128);
        assert_eq!(fixture.concurrent_short_requests, 4);
        assert_eq!(fixture.short_prefix_tokens, 64);
        assert_eq!(fixture.name(), "long_prefill_vs_decode");
    }

    #[test]
    fn run_without_artifacts_returns_skipped_with_env_var_reason() {
        let fixture = LongPrefillVsDecode::default();
        let ctx = WorkloadContext::synthetic();
        let outcome = fixture.run(&ctx);
        match outcome {
            WorkloadOutcome::Skipped { reason } => {
                assert!(
                    reason.contains("AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR"),
                    "expected skip reason to mention the env var, got: {reason}"
                );
            }
            other => panic!("expected Skipped, got {:?}", other.name()),
        }
    }

    #[test]
    fn run_with_empty_artifacts_dir_reports_failed_not_completed() {
        // The driver path is real: when artifacts_dir points to an empty
        // directory, `build_inference_session` should fail. The fixture must
        // surface that as `Failed`, not silently `Completed`. This test
        // exercises the driver entry without requiring a real MLX model.
        let fixture = LongPrefillVsDecode {
            prefill_tokens: 16,
            decode_tokens: 4,
            concurrent_short_requests: 1,
            short_prefix_tokens: 4,
            ..LongPrefillVsDecode::default()
        };
        let tmp = tempdir_for_test();
        let ctx = WorkloadContext {
            mlx_model_artifacts_dir: Some(tmp.clone()),
            seed: 0,
        };
        let outcome = fixture.run(&ctx);
        match outcome {
            WorkloadOutcome::Failed { error, .. } => {
                assert!(
                    error.contains("build_inference_session")
                        || error.contains("session")
                        || error.contains("artifact")
                        || error.contains("MLX"),
                    "expected failure to surface a session/artifact error, got: {error}"
                );
            }
            WorkloadOutcome::Skipped { reason } => {
                panic!(
                    "expected Failed (driver attempted real session build), got Skipped: {reason}"
                );
            }
            WorkloadOutcome::Completed { .. } => {
                panic!("did not expect Completed without a real MLX model");
            }
        }
    }

    #[test]
    fn report_skeleton_carries_configured_params_in_notes() {
        let fixture = LongPrefillVsDecode {
            prefill_tokens: 1024,
            decode_tokens: 32,
            concurrent_short_requests: 2,
            ..LongPrefillVsDecode::default()
        };
        let report = fixture.report_skeleton();
        let value = report.to_json();
        let notes = value["notes"].as_array().expect("notes is array");
        assert!(
            notes.iter().any(|n| n
                .as_str()
                .map(|s| s.contains("prefill_tokens=1024"))
                .unwrap_or(false)),
            "expected report skeleton notes to capture configured prefill_tokens"
        );
    }

    #[test]
    fn build_long_prefill_request_has_capped_decode_budget() {
        // The long-prefill request's own decode budget must stay small so the
        // measurement focuses on prefill cost, not on the long request's
        // ongoing decode contention. This guards against accidental edits
        // that would conflate the measurement target.
        let fixture = LongPrefillVsDecode::default();
        let request = fixture.build_long_prefill_request(0);
        assert!(
            request.max_output_tokens <= 16,
            "long_prefill request decode budget too large: {}",
            request.max_output_tokens
        );
        assert!(
            request.input_tokens.len() as u32 == fixture.prefill_tokens,
            "long_prefill input_tokens does not match configured prefill_tokens"
        );
    }

    #[test]
    fn build_short_decode_requests_match_concurrent_count() {
        let fixture = LongPrefillVsDecode {
            concurrent_short_requests: 3,
            decode_tokens: 16,
            short_prefix_tokens: 8,
            ..LongPrefillVsDecode::default()
        };
        let requests = fixture.build_short_decode_requests(0);
        assert_eq!(requests.len(), 3);
        for request in &requests {
            assert_eq!(request.max_output_tokens, 16);
            assert_eq!(request.input_tokens.len(), 8);
        }
    }

    #[test]
    fn seed_changes_synthetic_prompt_tokens_across_runs() {
        // The fixture must consume `WorkloadContext.seed` so two runs with
        // different seeds produce different synthetic prompts. This guards
        // against the seed being plumbed into config but ignored when
        // building the actual stream requests.
        let fixture = LongPrefillVsDecode::default();
        let req_a = fixture.build_long_prefill_request(0);
        let req_b = fixture.build_long_prefill_request(1);
        assert_eq!(
            req_a.input_tokens.len(),
            req_b.input_tokens.len(),
            "seed should not change token count"
        );
        assert_ne!(
            req_a.input_tokens, req_b.input_tokens,
            "seed must influence synthetic prompt content"
        );
    }

    fn tempdir_for_test() -> PathBuf {
        // The bench crate does not depend on `tempfile`; use a deterministic
        // path under the process-scoped temp dir for this single test. The
        // directory is created empty so `build_inference_session` rejects it.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ax-engine-bench-long-prefill-vs-decode-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&path).expect("create empty temp dir for test");
        path
    }
}
