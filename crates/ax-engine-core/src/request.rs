use thiserror::Error;

use crate::execution_plan::ExecutionPlanBinding;
use crate::ids::{ModelId, RequestId, SequenceNo};
use crate::kv::BlockTable;
use crate::sampling::{SamplingParams, StopReason};
use crate::scheduler::RouteMetadata;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestState {
    Waiting,
    Runnable,
    Running,
    BlockedOnMemory,
    Finished,
    Cancelled,
    Failed,
}

impl RequestState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestSubmission {
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub input_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub max_output_tokens: u32,
    pub arrival_sequence: SequenceNo,
    pub metadata: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub arrival_sequence: SequenceNo,
    pub state: RequestState,
    pub prompt_tokens: Vec<u32>,
    pub processed_prompt_tokens: u32,
    pub generated_tokens: Vec<u32>,
    pub generated_token_logprobs: Vec<Option<f32>>,
    pub max_output_tokens: u32,
    pub sampling_params: SamplingParams,
    pub execution_plan_ref: Option<String>,
    pub route_metadata_hint: RouteMetadata,
    pub block_table: BlockTable,
    pub cancel_requested: bool,
    pub terminal_stop_reason: Option<StopReason>,
    pub last_error: Option<String>,
    pub metrics_snapshot: Option<String>,
}

impl RequestRecord {
    pub fn new(submission: RequestSubmission, block_table: BlockTable) -> Self {
        Self {
            request_id: submission.request_id,
            model_id: submission.model_id,
            arrival_sequence: submission.arrival_sequence,
            state: RequestState::Waiting,
            prompt_tokens: submission.input_tokens,
            processed_prompt_tokens: 0,
            generated_tokens: Vec::new(),
            generated_token_logprobs: Vec::new(),
            max_output_tokens: submission.max_output_tokens,
            sampling_params: submission.sampling_params,
            execution_plan_ref: None,
            route_metadata_hint: RouteMetadata::empty(),
            block_table,
            cancel_requested: false,
            terminal_stop_reason: None,
            last_error: None,
            metrics_snapshot: None,
        }
    }

    pub fn snapshot(&self) -> RequestSnapshot {
        RequestSnapshot {
            request_id: self.request_id,
            model_id: self.model_id.clone(),
            arrival_sequence: self.arrival_sequence,
            state: self.state,
            prompt_tokens: self.prompt_tokens.clone(),
            processed_prompt_tokens: self.processed_prompt_tokens,
            generated_tokens: self.generated_tokens.clone(),
            generated_token_logprobs: self.generated_token_logprobs.clone(),
            prompt_len: self.prompt_tokens.len() as u32,
            generated_len: self.generated_tokens.len() as u32,
            max_output_tokens: self.max_output_tokens,
            cancel_requested: self.cancel_requested,
            execution_plan_ref: self.execution_plan_ref.clone(),
            route_metadata_hint: self.route_metadata_hint.clone(),
            terminal_stop_reason: self.terminal_stop_reason,
            last_error: self.last_error.clone(),
        }
    }

    pub fn set_execution_plan_binding(&mut self, binding: Option<ExecutionPlanBinding>) {
        if let Some(binding) = binding {
            self.execution_plan_ref = Some(binding.execution_plan_ref);
            self.route_metadata_hint = binding.route_metadata;
        } else {
            self.execution_plan_ref = None;
            self.route_metadata_hint = RouteMetadata::empty();
        }
    }

    pub fn mark_runnable(&mut self) -> Result<(), StateTransitionError> {
        self.transition_to(RequestState::Runnable)
    }

    pub fn start_running(&mut self) -> Result<(), StateTransitionError> {
        self.transition_to(RequestState::Running)
    }

    pub fn mark_blocked_on_memory(&mut self) -> Result<(), StateTransitionError> {
        self.transition_to(RequestState::BlockedOnMemory)
    }

    pub fn unblock_memory(&mut self) -> Result<(), StateTransitionError> {
        self.transition_to(RequestState::Runnable)
    }

    pub fn finish_with_reason(
        &mut self,
        terminal_stop_reason: Option<StopReason>,
    ) -> Result<(), StateTransitionError> {
        self.terminal_stop_reason = terminal_stop_reason;
        self.transition_to(RequestState::Finished)
    }

    pub fn fail(&mut self, error: impl Into<String>) -> Result<(), StateTransitionError> {
        self.last_error = Some(error.into());
        self.transition_to(RequestState::Failed)
    }

    pub fn request_cancel(&mut self) -> Result<(), StateTransitionError> {
        self.cancel_requested = true;
        match self.state {
            RequestState::Waiting | RequestState::Runnable | RequestState::BlockedOnMemory => {
                self.terminal_stop_reason = Some(StopReason::Cancelled);
                self.transition_to(RequestState::Cancelled)
            }
            RequestState::Running => Ok(()),
            RequestState::Finished | RequestState::Cancelled | RequestState::Failed => Ok(()),
        }
    }

    pub fn resolve_running_step(
        &mut self,
        completed_normally: bool,
    ) -> Result<(), StateTransitionError> {
        match self.state {
            RequestState::Running => {
                if self.cancel_requested {
                    self.terminal_stop_reason = Some(StopReason::Cancelled);
                    self.transition_to(RequestState::Cancelled)
                } else if completed_normally {
                    self.transition_to(RequestState::Runnable)
                } else {
                    self.transition_to(RequestState::Failed)
                }
            }
            other if self.cancel_requested => Err(StateTransitionError::InvalidTransition {
                from: other,
                to: RequestState::Cancelled,
            }),
            other if completed_normally => Err(StateTransitionError::InvalidTransition {
                from: other,
                to: RequestState::Runnable,
            }),
            other => Err(StateTransitionError::InvalidTransition {
                from: other,
                to: RequestState::Failed,
            }),
        }
    }

    pub fn cleanup_request_id(&self) -> Option<RequestId> {
        if self.state.is_terminal() {
            Some(self.request_id)
        } else {
            None
        }
    }

    fn transition_to(&mut self, next: RequestState) -> Result<(), StateTransitionError> {
        if self.state == next {
            return Ok(());
        }

        let allowed = matches!(
            (self.state, next),
            (RequestState::Waiting, RequestState::Runnable)
                | (RequestState::Waiting, RequestState::Cancelled)
                | (RequestState::Waiting, RequestState::Failed)
                | (RequestState::Runnable, RequestState::Running)
                | (RequestState::Runnable, RequestState::BlockedOnMemory)
                | (RequestState::Runnable, RequestState::Cancelled)
                | (RequestState::Runnable, RequestState::Failed)
                | (RequestState::BlockedOnMemory, RequestState::Runnable)
                | (RequestState::BlockedOnMemory, RequestState::Cancelled)
                | (RequestState::BlockedOnMemory, RequestState::Failed)
                | (RequestState::Running, RequestState::Runnable)
                | (RequestState::Running, RequestState::Finished)
                | (RequestState::Running, RequestState::Cancelled)
                | (RequestState::Running, RequestState::Failed)
        );

        if !allowed {
            return Err(StateTransitionError::InvalidTransition {
                from: self.state,
                to: next,
            });
        }

        self.state = next;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestSnapshot {
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub arrival_sequence: SequenceNo,
    pub state: RequestState,
    pub prompt_tokens: Vec<u32>,
    pub processed_prompt_tokens: u32,
    pub generated_tokens: Vec<u32>,
    pub generated_token_logprobs: Vec<Option<f32>>,
    pub prompt_len: u32,
    pub generated_len: u32,
    pub max_output_tokens: u32,
    pub cancel_requested: bool,
    pub execution_plan_ref: Option<String>,
    pub route_metadata_hint: RouteMetadata,
    pub terminal_stop_reason: Option<StopReason>,
    pub last_error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum StateTransitionError {
    #[error("invalid request state transition: {from:?} -> {to:?}")]
    InvalidTransition {
        from: RequestState,
        to: RequestState,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::ExecutionPlanBinding;
    use crate::ids::{CacheGroupId, SequenceNo};

    fn make_record() -> RequestRecord {
        let submission = RequestSubmission {
            request_id: RequestId(1),
            model_id: ModelId("qwen3".into()),
            input_tokens: vec![1, 2, 3],
            sampling_params: SamplingParams::default(),
            max_output_tokens: 32,
            arrival_sequence: SequenceNo(1),
            metadata: None,
        };

        RequestRecord::new(submission, BlockTable::empty(CacheGroupId(0)))
    }

    #[test]
    fn follows_normal_lifecycle() {
        let mut record = make_record();

        record.mark_runnable().unwrap();
        record.start_running().unwrap();
        record.resolve_running_step(true).unwrap();
        record.start_running().unwrap();
        record
            .finish_with_reason(Some(StopReason::MaxOutputTokens))
            .unwrap();

        assert_eq!(record.state, RequestState::Finished);
        assert_eq!(
            record.terminal_stop_reason,
            Some(StopReason::MaxOutputTokens)
        );
        assert_eq!(record.cleanup_request_id(), Some(RequestId(1)));
    }

    #[test]
    fn blocks_invalid_transition() {
        let mut record = make_record();

        let error = record.start_running().unwrap_err();
        assert_eq!(
            error,
            StateTransitionError::InvalidTransition {
                from: RequestState::Waiting,
                to: RequestState::Running,
            }
        );
    }

    #[test]
    fn defers_cancellation_for_running_requests() {
        let mut record = make_record();

        record.mark_runnable().unwrap();
        record.start_running().unwrap();
        record.request_cancel().unwrap();

        assert!(record.cancel_requested);
        assert_eq!(record.state, RequestState::Running);

        record.resolve_running_step(true).unwrap();
        assert_eq!(record.state, RequestState::Cancelled);
    }

    #[test]
    fn blocked_on_memory_transitions_to_runnable_on_unblock() {
        let mut record = make_record();

        record.mark_runnable().unwrap();
        record.mark_blocked_on_memory().unwrap();
        assert_eq!(record.state, RequestState::BlockedOnMemory);

        record.unblock_memory().unwrap();
        assert_eq!(record.state, RequestState::Runnable);
    }

    #[test]
    fn fail_records_error_and_transitions_to_failed() {
        let mut record = make_record();

        record.mark_runnable().unwrap();
        record.start_running().unwrap();
        record.fail("test failure reason").unwrap();

        assert_eq!(record.state, RequestState::Failed);
        assert_eq!(record.last_error.as_deref(), Some("test failure reason"));
        assert!(record.state.is_terminal());
    }

    #[test]
    fn cancel_on_already_terminal_is_idempotent() {
        let mut record = make_record();

        record.mark_runnable().unwrap();
        record.start_running().unwrap();
        record
            .finish_with_reason(Some(StopReason::MaxOutputTokens))
            .unwrap();

        record.request_cancel().unwrap();

        assert_eq!(record.state, RequestState::Finished);
    }

    #[test]
    fn cancel_from_blocked_on_memory_transitions_directly() {
        let mut record = make_record();

        record.mark_runnable().unwrap();
        record.mark_blocked_on_memory().unwrap();
        record.request_cancel().unwrap();

        assert_eq!(record.state, RequestState::Cancelled);
    }

    #[test]
    fn resolve_running_step_with_failure_transitions_to_failed() {
        let mut record = make_record();

        record.mark_runnable().unwrap();
        record.start_running().unwrap();
        record.resolve_running_step(false).unwrap();

        assert_eq!(record.state, RequestState::Failed);
    }

    #[test]
    fn cleanup_snapshot_returns_none_for_non_terminal() {
        let record = make_record();

        assert!(record.cleanup_request_id().is_none());
    }

    #[test]
    fn cleanup_request_id_returns_request_for_terminal_record() {
        let mut record = make_record();
        record.request_cancel().unwrap();

        assert_eq!(record.cleanup_request_id(), Some(RequestId(1)));
    }

    #[test]
    fn resolve_running_step_reports_failed_target_for_non_running_failure() {
        let mut record = make_record();

        let error = record
            .resolve_running_step(false)
            .expect_err("non-running resolve should fail");

        assert_eq!(
            error,
            StateTransitionError::InvalidTransition {
                from: RequestState::Waiting,
                to: RequestState::Failed,
            }
        );
    }

    #[test]
    fn stores_route_hint_when_execution_plan_binding_is_applied() {
        let mut record = make_record();

        record.set_execution_plan_binding(Some(ExecutionPlanBinding {
            execution_plan_ref: "phase1.qwen3.dense_prefill".into(),
            route_metadata: RouteMetadata {
                execution_plan: Some("phase1.qwen3.dense_prefill".into()),
                attention_route: Some("qwen3_prefill".into()),
                kv_mode: Some("paged_metadata".into()),
                prefix_cache_path: None,
                barrier_mode: Some("serial".into()),
                crossover_decisions: Vec::new(),
            },
        }));

        assert_eq!(
            record.execution_plan_ref.as_deref(),
            Some("phase1.qwen3.dense_prefill")
        );
        assert_eq!(
            record.route_metadata_hint.attention_route.as_deref(),
            Some("qwen3_prefill")
        );
    }
}
