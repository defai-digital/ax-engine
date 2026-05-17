use crate::backend::SelectedBackend;
use crate::llama_cpp::{LlamaCppStreamChunk, LlamaCppStreamHandle};
use crate::request::{EngineStepReport, SessionRequestReport, SessionRequestState};

use super::errors::EngineSessionError;
use super::stream::{
    apply_llama_cpp_prompt_progress, apply_llama_cpp_stream_chunk, apply_llama_cpp_usage_counts,
};

#[derive(Debug)]
pub(super) enum LlamaCppLifecycleRequestSlot {
    Active(Box<LlamaCppLifecycleRequest>),
    Terminal(Box<SessionRequestReport>),
}

#[derive(Debug)]
pub(super) struct LlamaCppLifecycleRequest {
    pub(super) request_id: u64,
    pub(super) current_report: SessionRequestReport,
    pub(super) prompt_token_count: Option<u32>,
    pub(super) output_token_count: Option<u32>,
    pub(super) cached_prompt_tokens_observed: u32,
    pub(super) prefix_hit_recorded: bool,
    pub(super) step_count: u64,
    pub(super) ttft_step: Option<u64>,
    pub(super) stream: LlamaCppStreamHandle,
}

impl LlamaCppLifecycleRequestSlot {
    pub(super) fn report(&self) -> SessionRequestReport {
        match self {
            Self::Active(request) => request.current_report.clone(),
            Self::Terminal(report) => report.as_ref().clone(),
        }
    }
}

impl LlamaCppLifecycleRequest {
    pub(super) fn new(
        request_id: u64,
        mut current_report: SessionRequestReport,
        stream: LlamaCppStreamHandle,
    ) -> Self {
        current_report.prompt_len = current_report.prompt_tokens.len() as u32;

        Self {
            request_id,
            current_report,
            prompt_token_count: None,
            output_token_count: None,
            cached_prompt_tokens_observed: 0,
            prefix_hit_recorded: false,
            step_count: 0,
            ttft_step: None,
            stream,
        }
    }

    pub(super) fn step_report(
        &mut self,
        selected_backend: SelectedBackend,
    ) -> Result<EngineStepReport, EngineSessionError> {
        let chunk =
            self.stream
                .next_chunk()?
                .ok_or(EngineSessionError::LlamaCppStreamEndedBeforeStop {
                    request_id: self.request_id,
                    selected_backend,
                })?;
        Ok(self.apply_chunk(chunk))
    }

    /// Drain any remaining stream chunks after the stop signal to capture
    /// trailing usage data. Some OpenAI-compatible servers (including MLX)
    /// send a final chunk with `usage` but empty `choices` after the stop
    /// chunk. Without draining, that usage data is lost when the slot
    /// transitions to Terminal.
    pub(super) fn drain_trailing_usage(&mut self) {
        while let Ok(Some(chunk)) = self.stream.next_chunk() {
            apply_llama_cpp_usage_counts(
                &mut self.current_report,
                &mut self.prompt_token_count,
                &mut self.output_token_count,
                &chunk,
            );
            apply_llama_cpp_prompt_progress(
                &mut self.current_report,
                chunk.prompt_progress.as_ref(),
                &mut self.cached_prompt_tokens_observed,
                &mut self.prefix_hit_recorded,
            );
        }
    }

    pub(super) fn cancel(&mut self) -> SessionRequestReport {
        self.current_report.state = SessionRequestState::Cancelled;
        self.current_report.cancel_requested = true;
        self.current_report.finish_reason = Some(crate::generate::GenerateFinishReason::Cancelled);
        self.current_report.terminal_stop_reason = Some(ax_engine_core::StopReason::Cancelled);
        self.current_report.clone()
    }

    fn apply_chunk(&mut self, chunk: LlamaCppStreamChunk) -> EngineStepReport {
        apply_llama_cpp_stream_chunk(
            &mut self.current_report,
            &mut self.prompt_token_count,
            &mut self.output_token_count,
            &mut self.cached_prompt_tokens_observed,
            &mut self.prefix_hit_recorded,
            &mut self.step_count,
            &mut self.ttft_step,
            chunk,
        )
        .step
    }
}
