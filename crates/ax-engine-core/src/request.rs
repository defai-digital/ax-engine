use thiserror::Error;

use crate::execution_plan::ExecutionPlanBinding;
use crate::gemma4_unified::Gemma4UnifiedRuntimeInputs;
use crate::ids::{ModelId, RequestId, SequenceNo};
use crate::kv::BlockTable;
use crate::sampling::{SamplingParams, StopReason};
use crate::scheduler::RouteMetadata;
use serde::{Deserialize, Serialize};

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
    pub multimodal_inputs: RequestMultimodalInputs,
    pub sampling_params: SamplingParams,
    pub max_output_tokens: u32,
    pub arrival_sequence: SequenceNo,
    pub metadata: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RequestMultimodalInputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemma4_unified: Option<Gemma4UnifiedRuntimeInputs>,
}

impl RequestMultimodalInputs {
    pub fn is_empty(&self) -> bool {
        self.gemma4_unified
            .as_ref()
            .is_none_or(|inputs| inputs.is_empty())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequestWorkloadHints {
    pub tool_call: bool,
    pub structured_output: bool,
}

impl RequestWorkloadHints {
    pub fn from_metadata(metadata: Option<&str>) -> Self {
        let Some(metadata) = metadata.map(str::trim).filter(|value| !value.is_empty()) else {
            return Self::default();
        };

        if let Ok(value) = serde_json::from_str::<serde_json::Value>(metadata) {
            let mut hints = Self::default();
            hints.merge_json(&value);
            return hints;
        }

        let mut hints = Self::default();
        hints.merge_text(metadata);
        hints
    }

    fn merge_json(&mut self, value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(object) => {
                for (key, value) in object {
                    self.merge_json_key_value(key, value);
                }
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    self.merge_json(value);
                }
            }
            serde_json::Value::String(value) => self.merge_text(value),
            _ => {}
        }
    }

    fn merge_json_key_value(&mut self, key: &str, value: &serde_json::Value) {
        let key = normalize_workload_hint_token(key);
        let truthy = json_hint_truthy(value);
        let structured_truthy = if key == "response_format" {
            json_response_format_is_structured(value)
        } else {
            truthy
        };

        if truthy
            && matches!(
                key.as_str(),
                "tool_call"
                    | "tool_call_mode"
                    | "ax_speculative_tool_call"
                    | "openai_tools"
                    | "tools"
                    | "tool_choice"
            )
        {
            self.tool_call = true;
        }
        if structured_truthy
            && matches!(
                key.as_str(),
                "structured_output"
                    | "structured_output_mode"
                    | "ax_speculative_structured_output"
                    | "json_mode"
                    | "json_object"
                    | "strict_json"
                    | "response_format"
                    | "json_schema"
            )
        {
            self.structured_output = true;
        }

        if matches!(key.as_str(), "workload" | "ax_workload" | "mode" | "task")
            || matches!(
                key.as_str(),
                "response_format" | "tools" | "tool_choice" | "json_schema"
            )
        {
            self.merge_json(value);
        }
    }

    fn merge_text(&mut self, value: &str) {
        let value = normalize_workload_hint_token(value);
        if value.contains("tool_call")
            || value.contains("toolcall")
            || value.contains("ax_speculative_tool_call")
        {
            self.tool_call = true;
        }
        if value.contains("structured_output")
            || value.contains("json_mode")
            || value.contains("json_object")
            || value.contains("strict_json")
            || value.contains("response_format")
            || value.contains("json_schema")
        {
            self.structured_output = true;
        }
    }
}

fn normalize_workload_hint_token(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn json_hint_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => value.as_u64().unwrap_or(1) != 0,
        serde_json::Value::String(value) => {
            let value = normalize_workload_hint_token(value);
            !matches!(value.as_str(), "" | "false" | "none" | "null" | "off" | "0")
        }
        serde_json::Value::Array(values) => !values.is_empty(),
        serde_json::Value::Object(object) => !object.is_empty(),
    }
}

fn json_response_format_is_structured(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::String(value) => {
            let value = normalize_workload_hint_token(value);
            !matches!(value.as_str(), "" | "text" | "none" | "false" | "off" | "0")
        }
        serde_json::Value::Object(object) => object
            .get("type")
            .and_then(serde_json::Value::as_str)
            .map(|value| normalize_workload_hint_token(value) != "text")
            .unwrap_or(!object.is_empty()),
        value => json_hint_truthy(value),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    pub request_id: RequestId,
    pub model_id: ModelId,
    pub arrival_sequence: SequenceNo,
    pub state: RequestState,
    pub prompt_tokens: Vec<u32>,
    pub multimodal_inputs: RequestMultimodalInputs,
    pub processed_prompt_tokens: u32,
    pub generated_tokens: Vec<u32>,
    pub generated_token_logprobs: Vec<Option<f32>>,
    pub max_output_tokens: u32,
    pub sampling_params: SamplingParams,
    pub workload_hints: RequestWorkloadHints,
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
            multimodal_inputs: submission.multimodal_inputs,
            processed_prompt_tokens: 0,
            generated_tokens: Vec::new(),
            generated_token_logprobs: Vec::new(),
            max_output_tokens: submission.max_output_tokens,
            sampling_params: submission.sampling_params,
            workload_hints: RequestWorkloadHints::from_metadata(submission.metadata.as_deref()),
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
            has_multimodal_inputs: !self.multimodal_inputs.is_empty(),
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
    /// Multimodal prefill is atomic: the MLX runner rejects any prefill item
    /// that does not complete the prompt (media soft-token spans cannot be
    /// split across execution items), so the scheduler must defer rather than
    /// chunk prefill work for requests that carry multimodal inputs.
    pub has_multimodal_inputs: bool,
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
            multimodal_inputs: Default::default(),
            sampling_params: SamplingParams::default(),
            max_output_tokens: 32,
            arrival_sequence: SequenceNo(1),
            metadata: None,
        };

        RequestRecord::new(submission, BlockTable::empty(CacheGroupId(0)))
    }

    #[test]
    fn parses_request_workload_hints_from_metadata_json() {
        let hints = RequestWorkloadHints::from_metadata(Some(
            r#"{"tool_call": true, "response_format": {"type": "json_object"}}"#,
        ));
        assert!(hints.tool_call);
        assert!(hints.structured_output);

        let hints = RequestWorkloadHints::from_metadata(Some(
            r#"{"tool_call": false, "structured_output": false}"#,
        ));
        assert_eq!(hints, RequestWorkloadHints::default());

        let hints =
            RequestWorkloadHints::from_metadata(Some(r#"{"response_format":{"type":"text"}}"#));
        assert_eq!(hints, RequestWorkloadHints::default());
    }

    #[test]
    fn parses_request_workload_hints_from_metadata_text() {
        let hints =
            RequestWorkloadHints::from_metadata(Some("workload=tool-call; mode=strict-json"));
        assert!(hints.tool_call);
        assert!(hints.structured_output);
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
