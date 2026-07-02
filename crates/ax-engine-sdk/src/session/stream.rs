use crate::backend::{RuntimeReport, SelectedBackend};
use crate::generate::{
    GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateStreamEvent,
    GenerateStreamRequestEvent, GenerateStreamResponseEvent, GenerateStreamStepEvent,
};
use crate::llama_cpp::{LlamaCppPromptProgress, LlamaCppStreamChunk, LlamaCppStreamHandle};
use crate::mlx_lm::{MlxLmStreamChunkResult, MlxLmStreamHandle, finish_reason_from_mlx_lm};
use crate::request::{EngineStepReport, SessionRequestReport, SessionRequestState};

use super::{
    EngineSession, EngineSessionError, MLX_LM_STREAM_EXECUTION_PLAN,
    NATIVE_STREAM_STEP_GUARD_BUFFER, initial_stream_request_report,
};

#[derive(Debug)]
pub struct GenerateStream<'a> {
    pub(super) session: &'a mut EngineSession,
    pub(super) state: GenerateStreamState,
}

#[derive(Debug)]
pub enum GenerateStreamState {
    Native(Box<NativeGenerateStreamState>),
    LlamaCpp(Box<LlamaCppGenerateStreamState>),
    MlxLm(Box<MlxLmGenerateStreamState>),
}

#[derive(Debug)]
pub struct NativeGenerateStreamState {
    pub(super) request_id: u64,
    pub(super) runtime: RuntimeReport,
    pub(super) current_report: SessionRequestReport,
    pub(super) emitted_output_len: usize,
    pub(super) max_steps: u64,
    pub(super) step_count: u64,
    pub(super) ttft_step: Option<u64>,
    pub(super) phase: GenerateStreamPhase,
}

#[derive(Debug)]
pub struct LlamaCppGenerateStreamState {
    pub(super) request_id: u64,
    pub(super) runtime: RuntimeReport,
    pub(super) current_report: SessionRequestReport,
    pub(super) prompt_text: Option<String>,
    pub(super) output_text: String,
    pub(super) prompt_token_count: Option<u32>,
    pub(super) output_token_count: Option<u32>,
    pub(super) cached_prompt_tokens_observed: u32,
    pub(super) prefix_hit_recorded: bool,
    pub(super) step_count: u64,
    pub(super) ttft_step: Option<u64>,
    pub(super) terminal_chunk_seen: bool,
    pub(super) stream: LlamaCppStreamHandle,
    pub(super) phase: GenerateStreamPhase,
}

#[derive(Debug)]
pub struct MlxLmGenerateStreamState {
    pub(super) request_id: u64,
    pub(super) runtime: RuntimeReport,
    pub(super) current_report: SessionRequestReport,
    pub(super) prompt_text: Option<String>,
    pub(super) output_text: String,
    pub(super) prompt_token_count: Option<u32>,
    pub(super) output_token_count: Option<u32>,
    pub(super) step_count: u64,
    pub(super) ttft_step: Option<u64>,
    pub(super) stream: MlxLmStreamHandle,
    pub(super) phase: GenerateStreamPhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GenerateStreamPhase {
    Request,
    Step,
    Done,
}

pub(super) fn next_llama_cpp_stream_event(
    state: &mut LlamaCppGenerateStreamState,
    selected_backend: SelectedBackend,
) -> Result<Option<GenerateStreamEvent>, EngineSessionError> {
    match state.phase {
        GenerateStreamPhase::Request => {
            state.phase = GenerateStreamPhase::Step;
            Ok(Some(GenerateStreamEvent::Request(
                GenerateStreamRequestEvent {
                    request: state.current_report.clone(),
                    runtime: state.runtime.clone(),
                },
            )))
        }
        GenerateStreamPhase::Step => {
            if state.terminal_chunk_seen {
                match state.stream.next_chunk()? {
                    Some(chunk) => return Ok(Some(state.step_event_from_chunk(chunk))),
                    None => {
                        state.phase = GenerateStreamPhase::Done;
                        return Ok(Some(GenerateStreamEvent::Response(
                            GenerateStreamResponseEvent {
                                response: delegated_stream_response(
                                    state.request_id,
                                    &state.current_report,
                                    &state.runtime,
                                    state.prompt_text.clone(),
                                    Some(state.output_text.clone()),
                                    state.prompt_token_count,
                                    state.output_token_count,
                                    state.step_count,
                                    state.ttft_step,
                                ),
                            },
                        )));
                    }
                }
            }

            let chunk = state.stream.next_chunk()?.ok_or(
                EngineSessionError::LlamaCppStreamEndedBeforeStop {
                    request_id: state.request_id,
                    selected_backend,
                },
            )?;

            Ok(Some(state.step_event_from_chunk(chunk)))
        }
        GenerateStreamPhase::Done => Ok(None),
    }
}

impl<'a> GenerateStream<'a> {
    pub(super) fn new(session: &'a mut EngineSession, state: GenerateStreamState) -> Self {
        Self { session, state }
    }

    pub fn next_event(&mut self) -> Result<Option<GenerateStreamEvent>, EngineSessionError> {
        self.session.next_stream_event(&mut self.state)
    }

    pub fn into_response(mut self) -> Result<GenerateResponse, EngineSessionError> {
        let mut observed_event_count = 0_u64;
        while let Some(event) = self.next_event()? {
            observed_event_count = observed_event_count.saturating_add(1);
            if let GenerateStreamEvent::Response(event) = event {
                return Ok(event.response);
            }
        }

        Err(EngineSessionError::StreamEndedWithoutResponse {
            request_id: self.state.request_id(),
            observed_event_count,
        })
    }
}

impl Iterator for GenerateStream<'_> {
    type Item = Result<GenerateStreamEvent, EngineSessionError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_event() {
            Ok(Some(event)) => Some(Ok(event)),
            Ok(None) => None,
            Err(error) => {
                self.state.finish();
                Some(Err(error))
            }
        }
    }
}

impl GenerateStreamState {
    pub(super) fn new_native(
        request_id: u64,
        runtime: RuntimeReport,
        current_report: SessionRequestReport,
    ) -> Self {
        let max_steps = u64::from(current_report.prompt_len)
            + u64::from(current_report.max_output_tokens)
            + NATIVE_STREAM_STEP_GUARD_BUFFER;

        Self::Native(Box::new(NativeGenerateStreamState {
            request_id,
            runtime,
            emitted_output_len: current_report.output_tokens.len(),
            current_report,
            max_steps,
            step_count: 0,
            ttft_step: None,
            phase: GenerateStreamPhase::Request,
        }))
    }

    /// Request id of the in-flight generation this stream tracks. Public so
    /// bindings that abandon a stream mid-generation (e.g. the Python
    /// iterator's Drop) can cancel the underlying request.
    pub fn request_id(&self) -> u64 {
        match self {
            Self::Native(state) => state.request_id,
            Self::LlamaCpp(state) => state.request_id,
            Self::MlxLm(state) => state.request_id,
        }
    }

    fn finish(&mut self) {
        match self {
            Self::Native(state) => state.phase = GenerateStreamPhase::Done,
            Self::LlamaCpp(state) => state.phase = GenerateStreamPhase::Done,
            Self::MlxLm(state) => state.phase = GenerateStreamPhase::Done,
        }
    }
}

pub(super) struct LlamaCppChunkApplyResult {
    pub(super) step: EngineStepReport,
    pub(super) delta_tokens: Vec<u32>,
    pub(super) delta_text: String,
    pub(super) request: SessionRequestReport,
    pub(super) stop: bool,
}

impl LlamaCppGenerateStreamState {
    pub(super) fn new(
        request_id: u64,
        runtime: RuntimeReport,
        mut current_report: SessionRequestReport,
        prompt_text: Option<String>,
        stream: LlamaCppStreamHandle,
    ) -> Self {
        current_report.prompt_len = current_report.prompt_tokens.len() as u32;

        Self {
            request_id,
            runtime,
            current_report,
            prompt_text,
            output_text: String::new(),
            prompt_token_count: None,
            output_token_count: None,
            cached_prompt_tokens_observed: 0,
            prefix_hit_recorded: false,
            step_count: 0,
            ttft_step: None,
            terminal_chunk_seen: false,
            stream,
            phase: GenerateStreamPhase::Request,
        }
    }

    pub(super) fn step_event_from_chunk(
        &mut self,
        chunk: LlamaCppStreamChunk,
    ) -> GenerateStreamEvent {
        let applied = apply_llama_cpp_stream_chunk(
            &mut self.current_report,
            &mut self.prompt_token_count,
            &mut self.output_token_count,
            &mut self.cached_prompt_tokens_observed,
            &mut self.prefix_hit_recorded,
            &mut self.step_count,
            &mut self.ttft_step,
            chunk,
        );
        self.output_text.push_str(&applied.delta_text);
        if applied.stop {
            self.terminal_chunk_seen = true;
        }

        let delta_token_logprobs = vec![None; applied.delta_tokens.len()];

        GenerateStreamEvent::Step(GenerateStreamStepEvent {
            request: applied.request,
            step: applied.step,
            delta_tokens: applied.delta_tokens,
            delta_token_logprobs,
            delta_text: if applied.delta_text.is_empty() {
                None
            } else {
                Some(applied.delta_text)
            },
        })
    }
}

impl MlxLmGenerateStreamState {
    pub(super) fn new(
        request_id: u64,
        runtime: RuntimeReport,
        mut current_report: SessionRequestReport,
        prompt_text: Option<String>,
        stream: MlxLmStreamHandle,
    ) -> Self {
        current_report.prompt_len = current_report.prompt_tokens.len() as u32;
        Self {
            request_id,
            runtime,
            current_report,
            prompt_text,
            output_text: String::new(),
            prompt_token_count: None,
            output_token_count: None,
            step_count: 0,
            ttft_step: None,
            stream,
            phase: GenerateStreamPhase::Request,
        }
    }

    pub(super) fn step_event_from_chunk(
        &mut self,
        chunk: MlxLmStreamChunkResult,
    ) -> GenerateStreamEvent {
        self.step_count += 1;
        let delta_text = chunk.text;
        let is_terminal = chunk.finish_reason.is_some();

        let ttft_events = if self.ttft_step.is_none() && !delta_text.is_empty() {
            self.ttft_step = Some(self.step_count);
            1
        } else {
            0
        };

        self.output_text.push_str(&delta_text);

        let finish_reason = finish_reason_from_mlx_lm(chunk.finish_reason.as_deref());
        if finish_reason.is_some() {
            self.current_report.finish_reason = finish_reason;
            self.current_report.terminal_stop_reason =
                terminal_stop_reason_from_finish_reason(finish_reason);
        }
        if let Some(pt) = chunk.prompt_token_count {
            self.prompt_token_count = Some(pt);
        }
        if let Some(ct) = chunk.output_token_count {
            self.output_token_count = Some(ct);
        }
        self.current_report.state = if is_terminal {
            SessionRequestState::Finished
        } else if !delta_text.is_empty() {
            SessionRequestState::Running
        } else {
            self.current_report.state
        };

        GenerateStreamEvent::Step(GenerateStreamStepEvent {
            request: self.current_report.clone(),
            step: EngineStepReport {
                step_id: None,
                scheduled_requests: u32::from(!delta_text.is_empty() || is_terminal),
                scheduled_tokens: 0,
                ttft_events,
                prefix_hits: 0,
                kv_usage_blocks: 0,
                evictions: 0,
                preempted_requests: 0,
                preempted_tokens: 0,
                cpu_time_us: 0,
                runner_time_us: 0,
                route: None,
                metal_dispatch: None,
            },
            delta_tokens: Vec::new(),
            delta_token_logprobs: Vec::new(),
            delta_text: if delta_text.is_empty() {
                None
            } else {
                Some(delta_text)
            },
        })
    }
}

pub(super) fn next_mlx_lm_stream_event(
    state: &mut MlxLmGenerateStreamState,
) -> Result<Option<GenerateStreamEvent>, EngineSessionError> {
    match state.phase {
        GenerateStreamPhase::Request => {
            state.phase = GenerateStreamPhase::Step;
            Ok(Some(GenerateStreamEvent::Request(
                GenerateStreamRequestEvent {
                    request: state.current_report.clone(),
                    runtime: state.runtime.clone(),
                },
            )))
        }
        GenerateStreamPhase::Step => match state.stream.next_chunk()? {
            Some(chunk) => Ok(Some(state.step_event_from_chunk(chunk))),
            None => {
                state.phase = GenerateStreamPhase::Done;
                Ok(Some(GenerateStreamEvent::Response(
                    GenerateStreamResponseEvent {
                        response: delegated_stream_response(
                            state.request_id,
                            &state.current_report,
                            &state.runtime,
                            state.prompt_text.clone(),
                            Some(state.output_text.clone()),
                            state.prompt_token_count,
                            state.output_token_count,
                            state.step_count,
                            state.ttft_step,
                        ),
                    },
                )))
            }
        },
        GenerateStreamPhase::Done => Ok(None),
    }
}

fn delegated_stream_response(
    request_id: u64,
    report: &SessionRequestReport,
    runtime: &RuntimeReport,
    prompt_text: Option<String>,
    output_text: Option<String>,
    prompt_token_count: Option<u32>,
    output_token_count: Option<u32>,
    step_count: u64,
    ttft_step: Option<u64>,
) -> GenerateResponse {
    GenerateResponse {
        request_id,
        model_id: report.model_id.clone(),
        prompt_tokens: report.prompt_tokens.clone(),
        prompt_text,
        output_tokens: report.output_tokens.clone(),
        output_token_logprobs: report.output_token_logprobs.clone(),
        output_text,
        prompt_token_count,
        output_token_count,
        status: crate::generate::GenerateStatus::Finished,
        finish_reason: report.finish_reason,
        step_count,
        ttft_step,
        route: report.route.clone(),
        runtime: runtime.clone(),
    }
}

pub(super) fn build_mlx_lm_stream_state(
    request_id: u64,
    request: GenerateRequest,
    runtime: RuntimeReport,
    stream: MlxLmStreamHandle,
) -> GenerateStreamState {
    let route = GenerateRouteReport::with_execution_plan(MLX_LM_STREAM_EXECUTION_PLAN);
    let current_report = initial_stream_request_report(
        request_id,
        request.model_id,
        request.input_tokens,
        request.max_output_tokens,
        route,
    );

    GenerateStreamState::MlxLm(Box::new(MlxLmGenerateStreamState::new(
        request_id,
        runtime,
        current_report,
        request.input_text,
        stream,
    )))
}

pub(super) fn apply_llama_cpp_stream_chunk(
    report: &mut SessionRequestReport,
    prompt_token_count: &mut Option<u32>,
    output_token_count: &mut Option<u32>,
    cached_prompt_tokens_observed: &mut u32,
    prefix_hit_recorded: &mut bool,
    step_count: &mut u64,
    ttft_step: &mut Option<u64>,
    chunk: LlamaCppStreamChunk,
) -> LlamaCppChunkApplyResult {
    *step_count += 1;
    let prefix_hits = apply_llama_cpp_prompt_progress(
        report,
        chunk.prompt_progress.as_ref(),
        cached_prompt_tokens_observed,
        prefix_hit_recorded,
    );
    apply_llama_cpp_usage_counts(report, prompt_token_count, output_token_count, &chunk);

    let delta_tokens = chunk.tokens;
    let delta_text = chunk.content;
    let was_terminal = is_terminal_request_state(report.state);
    let request_selected = chunk.prompt_progress.is_some()
        || !delta_tokens.is_empty()
        || !delta_text.is_empty()
        || chunk.stop;
    let ttft_events = if ttft_step.is_none() && (!delta_tokens.is_empty() || !delta_text.is_empty())
    {
        *ttft_step = Some(*step_count);
        1
    } else {
        0
    };

    report.output_tokens.extend(delta_tokens.iter().copied());
    report
        .output_token_logprobs
        .extend(std::iter::repeat_n(None, delta_tokens.len()));
    report.output_len = report.output_len.max(report.output_tokens.len() as u32);
    let finish_reason = finish_reason_from_stop_type(chunk.stop, chunk.stop_type.as_deref());
    if finish_reason.is_some() {
        report.finish_reason = finish_reason;
        report.terminal_stop_reason = terminal_stop_reason_from_finish_reason(finish_reason);
    }
    report.state = if chunk.stop || was_terminal {
        SessionRequestState::Finished
    } else if request_selected {
        SessionRequestState::Running
    } else {
        report.state
    };

    LlamaCppChunkApplyResult {
        step: EngineStepReport {
            step_id: None,
            scheduled_requests: u32::from(request_selected),
            scheduled_tokens: delta_tokens.len() as u32,
            ttft_events,
            prefix_hits,
            kv_usage_blocks: 0,
            evictions: 0,
            preempted_requests: 0,
            preempted_tokens: 0,
            cpu_time_us: 0,
            runner_time_us: 0,
            route: Some(report.route.clone()),
            metal_dispatch: None,
        },
        delta_tokens,
        delta_text,
        request: report.clone(),
        stop: chunk.stop,
    }
}

pub(super) fn slice_output_token_logprobs(
    report: &SessionRequestReport,
    emitted_output_len: usize,
    delta_token_count: usize,
) -> Result<Vec<Option<f32>>, EngineSessionError> {
    if report.output_token_logprobs.len() < emitted_output_len {
        return Err(EngineSessionError::RequestReportInvariantViolation {
            request_id: report.request_id,
            message: "output token logprobs shorter than emitted output length",
        });
    }

    let delta_logprobs = report.output_token_logprobs[emitted_output_len..].to_vec();
    if delta_logprobs.len() != delta_token_count {
        return Err(EngineSessionError::RequestReportInvariantViolation {
            request_id: report.request_id,
            message: "output token logprobs length diverged from output token delta",
        });
    }

    Ok(delta_logprobs)
}

pub(super) fn is_terminal_request_state(state: SessionRequestState) -> bool {
    matches!(
        state,
        SessionRequestState::Finished
            | SessionRequestState::Cancelled
            | SessionRequestState::Failed
    )
}

pub(super) fn apply_llama_cpp_prompt_progress(
    report: &mut SessionRequestReport,
    prompt_progress: Option<&LlamaCppPromptProgress>,
    cached_prompt_tokens_observed: &mut u32,
    prefix_hit_recorded: &mut bool,
) -> u32 {
    if let Some(progress) = prompt_progress {
        if progress.total > 0 {
            report.prompt_len = progress.total;
        }
        report.processed_prompt_tokens = report.processed_prompt_tokens.max(progress.processed);

        if progress.cache > *cached_prompt_tokens_observed {
            *cached_prompt_tokens_observed = progress.cache;
        }
    } else if report.prompt_len > 0 && report.processed_prompt_tokens == 0 {
        report.processed_prompt_tokens = report.prompt_len;
    }

    if *cached_prompt_tokens_observed > 0 {
        report.route.prefix_cache_path = Some("delegated_prompt_cache".to_string());
        report.route.crossover_decisions.insert(
            "delegated_cached_tokens".to_string(),
            *cached_prompt_tokens_observed,
        );
    }

    if !*prefix_hit_recorded && *cached_prompt_tokens_observed > 0 {
        *prefix_hit_recorded = true;
        1
    } else {
        0
    }
}

pub(super) fn apply_llama_cpp_usage_counts(
    report: &mut SessionRequestReport,
    prompt_token_count: &mut Option<u32>,
    output_token_count: &mut Option<u32>,
    chunk: &LlamaCppStreamChunk,
) {
    if let Some(count) = chunk.prompt_token_count {
        *prompt_token_count = Some(count);
        if report.prompt_len < count {
            report.prompt_len = count;
        }
        if report.processed_prompt_tokens < count {
            report.processed_prompt_tokens = count;
        }
    }

    if let Some(count) = chunk.output_token_count {
        *output_token_count = Some(count);
        if report.output_len < count {
            report.output_len = count;
        }
    }
}

pub(super) fn finish_reason_from_stop_type(
    stop: bool,
    stop_type: Option<&str>,
) -> Option<crate::generate::GenerateFinishReason> {
    if !stop {
        return None;
    }
    match stop_type {
        // llama.cpp server completion format
        Some("limit") => Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
        Some("eos") => Some(crate::generate::GenerateFinishReason::Stop),
        // OpenAI-compatible format (vLLM, mistral.rs, mlx)
        Some("length") => Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
        Some("stop") => Some(crate::generate::GenerateFinishReason::Stop),
        Some("content_filter") => Some(crate::generate::GenerateFinishReason::ContentFilter),
        None => Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
        Some(unknown) => {
            tracing::warn!(
                stop_type = unknown,
                "llama.cpp stream returned unknown stop_type; reporting error finish reason"
            );
            Some(crate::generate::GenerateFinishReason::Error)
        }
    }
}

pub(super) fn terminal_stop_reason_from_finish_reason(
    finish_reason: Option<crate::generate::GenerateFinishReason>,
) -> Option<ax_engine_core::StopReason> {
    match finish_reason {
        Some(crate::generate::GenerateFinishReason::Stop) => {
            Some(ax_engine_core::StopReason::EosToken)
        }
        Some(crate::generate::GenerateFinishReason::MaxOutputTokens) => {
            Some(ax_engine_core::StopReason::MaxOutputTokens)
        }
        Some(crate::generate::GenerateFinishReason::ContentFilter) => {
            Some(ax_engine_core::StopReason::Error)
        }
        Some(crate::generate::GenerateFinishReason::Cancelled) => {
            Some(ax_engine_core::StopReason::Cancelled)
        }
        Some(crate::generate::GenerateFinishReason::Error) => {
            Some(ax_engine_core::StopReason::Error)
        }
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::GenerateFinishReason;

    fn sample_finished_request_report(request_id: u64) -> SessionRequestReport {
        SessionRequestReport {
            request_id,
            model_id: "qwen3".to_string(),
            state: SessionRequestState::Finished,
            prompt_tokens: vec![1, 2, 3],
            processed_prompt_tokens: 3,
            output_tokens: vec![4, 5],
            output_token_logprobs: vec![Some(-0.25), Some(-0.5)],
            prompt_len: 3,
            output_len: 2,
            max_output_tokens: 2,
            cancel_requested: false,
            execution_plan_ref: None,
            route: GenerateRouteReport::default(),
            finish_reason: Some(GenerateFinishReason::Stop),
            terminal_stop_reason: None,
            last_error: None,
        }
    }

    #[test]
    fn slice_output_token_logprobs_fails_closed_on_length_mismatch() {
        let mut report = sample_finished_request_report(41);
        report.output_token_logprobs.pop();

        let error = slice_output_token_logprobs(&report, 0, 2)
            .expect_err("mismatched logprob lengths should fail closed");

        assert!(matches!(
            error,
            EngineSessionError::RequestReportInvariantViolation { request_id: 41, .. }
        ));
    }

    #[test]
    fn llama_cpp_stream_finish_reason_preserves_content_filter() {
        assert_eq!(
            finish_reason_from_stop_type(true, Some("content_filter")),
            Some(GenerateFinishReason::ContentFilter)
        );
        assert_eq!(
            finish_reason_from_stop_type(true, Some("backend_error")),
            Some(GenerateFinishReason::Error)
        );
        assert_eq!(
            terminal_stop_reason_from_finish_reason(Some(GenerateFinishReason::ContentFilter)),
            Some(ax_engine_core::StopReason::Error)
        );
    }
}
