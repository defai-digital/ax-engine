use std::collections::{BTreeMap, BTreeSet, VecDeque};

use thiserror::Error;

use crate::execution_plan::ExecutionPlanBinding;
use crate::ids::{CacheGroupId, RequestId};
use crate::kv::BlockTable;
use crate::request::{
    RequestRecord, RequestSnapshot, RequestState, RequestSubmission, StateTransitionError,
};
use crate::runner::RunnerOutput;
use crate::sampling::{SampledToken, StopReason};
use crate::scheduler::SchedulePlan;

const MAX_TERMINAL_SNAPSHOT_RETENTION: usize = 1024;

#[derive(Debug)]
pub struct RequestManager {
    cache_group_id: CacheGroupId,
    records: BTreeMap<RequestId, RequestRecord>,
    terminal_snapshots: BTreeMap<RequestId, RequestSnapshot>,
    terminal_snapshot_order: VecDeque<RequestId>,
}

impl RequestManager {
    pub fn new(cache_group_id: CacheGroupId) -> Self {
        Self {
            cache_group_id,
            records: BTreeMap::new(),
            terminal_snapshots: BTreeMap::new(),
            terminal_snapshot_order: VecDeque::new(),
        }
    }

    pub fn cache_group_id(&self) -> CacheGroupId {
        self.cache_group_id
    }

    pub fn submit(
        &mut self,
        submission: RequestSubmission,
    ) -> Result<RequestId, RequestManagerError> {
        let request_id = submission.request_id;

        if self.records.contains_key(&request_id)
            || self.terminal_snapshots.contains_key(&request_id)
        {
            return Err(RequestManagerError::DuplicateRequest(request_id));
        }

        let record = RequestRecord::new(submission, BlockTable::empty(self.cache_group_id));
        self.records.insert(request_id, record);
        Ok(request_id)
    }

    pub fn admit_waiting(&mut self) -> Result<Vec<RequestId>, RequestManagerError> {
        let request_ids = self.sorted_request_ids(|record| record.state == RequestState::Waiting);

        for request_id in &request_ids {
            self.transition_request(*request_id, RequestRecord::mark_runnable)?;
        }

        Ok(request_ids)
    }

    pub fn retry_memory_blocked(&mut self) -> Result<Vec<RequestId>, RequestManagerError> {
        let request_ids =
            self.sorted_request_ids(|record| record.state == RequestState::BlockedOnMemory);

        for request_id in &request_ids {
            self.transition_request(*request_id, RequestRecord::unblock_memory)?;
        }

        Ok(request_ids)
    }

    pub fn cancel(&mut self, request_id: RequestId) -> Result<(), RequestManagerError> {
        self.transition_request(request_id, RequestRecord::request_cancel)
    }

    pub fn record(&self, request_id: RequestId) -> Option<&RequestRecord> {
        self.records.get(&request_id)
    }

    pub fn snapshot(&self, request_id: RequestId) -> Option<RequestSnapshot> {
        self.record(request_id)
            .map(RequestRecord::snapshot)
            .or_else(|| self.terminal_snapshots.get(&request_id).cloned())
    }

    pub fn sync_block_table(
        &mut self,
        request_id: RequestId,
        block_table: BlockTable,
    ) -> Result<(), RequestManagerError> {
        let record = self
            .records
            .get_mut(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        record.block_table = block_table;
        Ok(())
    }

    pub fn set_execution_plan_binding(
        &mut self,
        request_id: RequestId,
        binding: Option<ExecutionPlanBinding>,
    ) -> Result<(), RequestManagerError> {
        let record = self
            .records
            .get_mut(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        record.set_execution_plan_binding(binding);
        Ok(())
    }

    pub fn apply_prefix_reuse(
        &mut self,
        request_id: RequestId,
        matched_prompt_tokens: u32,
    ) -> Result<(), RequestManagerError> {
        self.validate_prefix_reuse(request_id, matched_prompt_tokens)?;
        let record = self
            .records
            .get_mut(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        record.processed_prompt_tokens = matched_prompt_tokens;
        Ok(())
    }

    pub fn rollback_prefix_reuse(
        &mut self,
        request_id: RequestId,
        matched_prompt_tokens: u32,
    ) -> Result<(), RequestManagerError> {
        let record = self
            .records
            .get_mut(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        if record.processed_prompt_tokens != matched_prompt_tokens {
            return Err(RequestManagerError::ProgressInvariantViolation {
                request_id,
                message: "prefix reuse rollback target no longer matches processed prompt tokens",
            });
        }
        record.processed_prompt_tokens = 0;
        record.block_table = BlockTable::empty(self.cache_group_id);
        Ok(())
    }

    pub fn validate_prefix_reuse(
        &self,
        request_id: RequestId,
        matched_prompt_tokens: u32,
    ) -> Result<(), RequestManagerError> {
        let record = self
            .records
            .get(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        if matched_prompt_tokens > record.prompt_tokens.len() as u32 {
            return Err(RequestManagerError::ProgressInvariantViolation {
                request_id,
                message: "prefix reuse advanced past prompt length",
            });
        }
        if matched_prompt_tokens < record.processed_prompt_tokens {
            return Err(RequestManagerError::ProgressInvariantViolation {
                request_id,
                message: "prefix reuse moved processed prompt tokens backwards",
            });
        }

        Ok(())
    }

    pub fn snapshots(&self) -> Vec<RequestSnapshot> {
        let mut snapshots = self
            .records
            .values()
            .map(RequestRecord::snapshot)
            .collect::<Vec<_>>();
        snapshots.sort_by_key(|snapshot| (snapshot.arrival_sequence, snapshot.request_id));
        snapshots
    }

    pub fn apply_schedule_plan(
        &mut self,
        schedule_plan: &SchedulePlan,
    ) -> Result<(), RequestManagerError> {
        self.validate_schedule_plan(schedule_plan)?;

        for request_id in &schedule_plan.selected_requests {
            self.transition_request(*request_id, RequestRecord::start_running)?;
        }

        for request_id in &schedule_plan.memory_blocked_requests {
            self.transition_request(*request_id, RequestRecord::mark_blocked_on_memory)?;
        }

        Ok(())
    }

    pub fn collect_terminal_cleanup(&self) -> Vec<RequestId> {
        let request_ids = self.sorted_request_ids(|record| record.state.is_terminal());
        request_ids
            .into_iter()
            .filter_map(|request_id| {
                self.records
                    .get(&request_id)
                    .and_then(RequestRecord::cleanup_request_id)
            })
            .collect()
    }

    pub fn mark_terminal_cleaned(
        &mut self,
        request_id: RequestId,
    ) -> Result<(), RequestManagerError> {
        let is_terminal = self
            .records
            .get(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?
            .state
            .is_terminal();
        if !is_terminal {
            return Err(RequestManagerError::ProgressInvariantViolation {
                request_id,
                message: "cannot mark cleanup complete for non-terminal request",
            });
        }
        let record = self
            .records
            .remove(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        self.terminal_snapshots
            .insert(request_id, record.snapshot());
        self.terminal_snapshot_order.push_back(request_id);
        self.prune_terminal_snapshots();
        Ok(())
    }

    pub fn apply_execution_results(
        &mut self,
        runner_output: &RunnerOutput,
        sampled_tokens: &[SampledToken],
        sampled_request_ids: &[RequestId],
    ) -> Result<RunnerApplySummary, RequestManagerError> {
        let mut sampled_by_request = BTreeMap::new();
        for sampled_token in sampled_tokens {
            if sampled_by_request
                .insert(sampled_token.request_id, sampled_token.clone())
                .is_some()
            {
                return Err(RequestManagerError::DuplicateSampledToken(
                    sampled_token.request_id,
                ));
            }
        }

        let sampled_requests = sampled_request_ids.iter().copied().collect::<BTreeSet<_>>();

        let mut seen = BTreeSet::new();
        let mut summary = RunnerApplySummary::default();

        for update in &runner_output.request_updates {
            if !seen.insert(update.request_id) {
                return Err(RequestManagerError::DuplicateRunnerUpdate(
                    update.request_id,
                ));
            }

            let record = self
                .records
                .get_mut(&update.request_id)
                .ok_or(RequestManagerError::UnknownRequest(update.request_id))?;

            if record.state != RequestState::Running {
                return Err(RequestManagerError::ProgressInvariantViolation {
                    request_id: update.request_id,
                    message: "runner updates require request to be Running",
                });
            }

            if let Some(error) = &update.error {
                record.fail(error.clone()).map_err(|source| {
                    RequestManagerError::InvalidStateTransition {
                        request_id: update.request_id,
                        source,
                    }
                })?;
                continue;
            }

            if matches!(update.stop_reason, Some(StopReason::Error)) {
                record
                    .fail("runner reported stop_reason=Error")
                    .map_err(|source| RequestManagerError::InvalidStateTransition {
                        request_id: update.request_id,
                        source,
                    })?;
                continue;
            }

            let sampled_stop_reason = if sampled_requests.contains(&update.request_id) {
                let sampled_token = sampled_by_request.remove(&update.request_id).ok_or(
                    RequestManagerError::ProgressInvariantViolation {
                        request_id: update.request_id,
                        message: "sampleable update missing sampled token",
                    },
                )?;
                if matches!(sampled_token.stop_reason, Some(StopReason::Error)) {
                    record
                        .fail("sampler reported stop_reason=Error")
                        .map_err(|source| RequestManagerError::InvalidStateTransition {
                            request_id: update.request_id,
                            source,
                        })?;
                    continue;
                }

                if record.processed_prompt_tokens < record.prompt_tokens.len() as u32 {
                    let next_processed_prompt_tokens = record
                        .processed_prompt_tokens
                        .checked_add(update.tokens_executed)
                        .ok_or(RequestManagerError::ProgressInvariantViolation {
                            request_id: update.request_id,
                            message: "processed prompt token counter overflowed",
                        })?;
                    if next_processed_prompt_tokens > record.prompt_tokens.len() as u32 {
                        return Err(RequestManagerError::ProgressInvariantViolation {
                            request_id: update.request_id,
                            message: "runner advanced prompt past available prompt tokens",
                        });
                    }
                    if next_processed_prompt_tokens != record.prompt_tokens.len() as u32 {
                        return Err(RequestManagerError::ProgressInvariantViolation {
                            request_id: update.request_id,
                            message: "prefill request may only sample when the scheduled step completes the prompt",
                        });
                    }
                    record.processed_prompt_tokens = next_processed_prompt_tokens;
                }

                if record.generated_tokens.is_empty() {
                    summary.ttft_events += 1;
                }
                let stop = sampled_token.stop_reason;
                record.generated_tokens.push(sampled_token.token_id);
                record.generated_token_logprobs.push(sampled_token.logprob);
                stop
            } else {
                let next_processed_prompt_tokens = record
                    .processed_prompt_tokens
                    .checked_add(update.tokens_executed)
                    .ok_or(RequestManagerError::ProgressInvariantViolation {
                        request_id: update.request_id,
                        message: "processed prompt token counter overflowed",
                    })?;

                if next_processed_prompt_tokens > record.prompt_tokens.len() as u32 {
                    return Err(RequestManagerError::ProgressInvariantViolation {
                        request_id: update.request_id,
                        message: "runner advanced prompt past available prompt tokens",
                    });
                }

                record.processed_prompt_tokens = next_processed_prompt_tokens;
                None
            };

            let stop_reason = sampled_stop_reason.or(update.stop_reason);
            let reached_max_output_tokens =
                record.generated_tokens.len() as u32 >= record.max_output_tokens;
            let terminal_stop_reason = stop_reason
                .or_else(|| reached_max_output_tokens.then_some(StopReason::MaxOutputTokens));
            let should_finish = reached_max_output_tokens
                || matches!(
                    stop_reason,
                    Some(StopReason::EosToken | StopReason::MaxOutputTokens)
                );

            if record.cancel_requested {
                record.resolve_running_step(true).map_err(|source| {
                    RequestManagerError::InvalidStateTransition {
                        request_id: update.request_id,
                        source,
                    }
                })?;
            } else if should_finish {
                record
                    .finish_with_reason(terminal_stop_reason)
                    .map_err(|source| RequestManagerError::InvalidStateTransition {
                        request_id: update.request_id,
                        source,
                    })?;
            } else {
                record.resolve_running_step(true).map_err(|source| {
                    RequestManagerError::InvalidStateTransition {
                        request_id: update.request_id,
                        source,
                    }
                })?;
            }
        }

        if let Some((&request_id, _)) = sampled_by_request.iter().next() {
            return Err(RequestManagerError::ProgressInvariantViolation {
                request_id,
                message: "sampled token was provided for a request without runner logits",
            });
        }

        Ok(summary)
    }

    fn validate_schedule_plan(
        &self,
        schedule_plan: &SchedulePlan,
    ) -> Result<(), RequestManagerError> {
        let mut seen = BTreeSet::new();

        for request_id in schedule_plan
            .selected_requests
            .iter()
            .chain(schedule_plan.deferred_requests.iter())
            .chain(schedule_plan.memory_blocked_requests.iter())
        {
            if !self.records.contains_key(request_id) {
                return Err(RequestManagerError::UnknownRequest(*request_id));
            }

            if !seen.insert(*request_id) {
                return Err(RequestManagerError::DuplicatePlanRequest(*request_id));
            }
        }

        for (request_id, record) in &self.records {
            if record.state == RequestState::Runnable && !seen.contains(request_id) {
                return Err(RequestManagerError::ProgressInvariantViolation {
                    request_id: *request_id,
                    message: "runnable request missing from schedule plan",
                });
            }
        }

        Ok(())
    }

    fn sorted_request_ids(&self, predicate: impl Fn(&RequestRecord) -> bool) -> Vec<RequestId> {
        let mut request_ids = self
            .records
            .values()
            .filter(|record| predicate(record))
            .map(|record| (record.arrival_sequence, record.request_id))
            .collect::<Vec<_>>();
        request_ids.sort();
        request_ids
            .into_iter()
            .map(|(_, request_id)| request_id)
            .collect()
    }

    fn prune_terminal_snapshots(&mut self) {
        while self.terminal_snapshot_order.len() > MAX_TERMINAL_SNAPSHOT_RETENTION {
            let Some(evicted_request_id) = self.terminal_snapshot_order.pop_front() else {
                break;
            };
            self.terminal_snapshots.remove(&evicted_request_id);
        }
    }

    fn transition_request(
        &mut self,
        request_id: RequestId,
        transition: impl FnOnce(&mut RequestRecord) -> Result<(), StateTransitionError>,
    ) -> Result<(), RequestManagerError> {
        let record = self
            .records
            .get_mut(&request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        transition(record)
            .map_err(|source| RequestManagerError::InvalidStateTransition { request_id, source })
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RequestManagerError {
    #[error("duplicate request registration: {0:?}")]
    DuplicateRequest(RequestId),
    #[error("unknown request: {0:?}")]
    UnknownRequest(RequestId),
    #[error("request appears multiple times in schedule plan: {0:?}")]
    DuplicatePlanRequest(RequestId),
    #[error("request appears multiple times in runner output: {0:?}")]
    DuplicateRunnerUpdate(RequestId),
    #[error("request appears multiple times in sampled token set: {0:?}")]
    DuplicateSampledToken(RequestId),
    #[error("request {request_id:?} violated progress invariant: {message}")]
    ProgressInvariantViolation {
        request_id: RequestId,
        message: &'static str,
    },
    #[error("request {request_id:?} failed transition: {source}")]
    InvalidStateTransition {
        request_id: RequestId,
        #[source]
        source: StateTransitionError,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RunnerApplySummary {
    pub ttft_events: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{ModelId, SequenceNo, StepId};
    use crate::runner::{ExecutionStatus, KvWriteSummary};
    use crate::sampling::SamplingParams;

    fn make_submission(
        request_id: u64,
        arrival_sequence: u64,
        model_id: &str,
    ) -> RequestSubmission {
        RequestSubmission {
            request_id: RequestId(request_id),
            model_id: ModelId(model_id.into()),
            input_tokens: vec![10, 11, 12],
            sampling_params: SamplingParams::default(),
            max_output_tokens: 16,
            arrival_sequence: SequenceNo(arrival_sequence),
            metadata: None,
        }
    }

    #[test]
    fn admits_waiting_requests_in_arrival_order() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(20, 2, "qwen3")).unwrap();
        manager.submit(make_submission(10, 1, "qwen3")).unwrap();

        let admitted = manager.admit_waiting().unwrap();

        assert_eq!(admitted, vec![RequestId(10), RequestId(20)]);
        assert_eq!(
            manager.snapshot(RequestId(10)).unwrap().state,
            RequestState::Runnable
        );
        assert_eq!(
            manager.snapshot(RequestId(20)).unwrap().state,
            RequestState::Runnable
        );
    }

    #[test]
    fn applies_schedule_plan_state_transitions() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.submit(make_submission(2, 2, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();

        let schedule_plan = SchedulePlan {
            step_id: StepId(3),
            selected_requests: vec![RequestId(1)],
            deferred_requests: vec![],
            memory_blocked_requests: vec![RequestId(2)],
            execution_batch: None,
        };

        manager.apply_schedule_plan(&schedule_plan).unwrap();

        assert_eq!(
            manager.snapshot(RequestId(1)).unwrap().state,
            RequestState::Running
        );
        assert_eq!(
            manager.snapshot(RequestId(2)).unwrap().state,
            RequestState::BlockedOnMemory
        );
    }

    #[test]
    fn collects_terminal_cleanup_once() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.cancel(RequestId(1)).unwrap();

        let first = manager.collect_terminal_cleanup();
        manager.mark_terminal_cleaned(RequestId(1)).unwrap();
        let second = manager.collect_terminal_cleanup();

        assert_eq!(first.len(), 1);
        assert_eq!(first[0], RequestId(1));
        assert!(second.is_empty());
    }

    #[test]
    fn mark_terminal_cleaned_moves_request_to_retained_snapshot_store() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.cancel(RequestId(1)).unwrap();
        manager.mark_terminal_cleaned(RequestId(1)).unwrap();

        assert!(manager.record(RequestId(1)).is_none());
        assert_eq!(
            manager.snapshot(RequestId(1)).unwrap().state,
            RequestState::Cancelled
        );
    }

    #[test]
    fn retained_terminal_snapshots_are_pruned_after_retention_limit() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        for request_id in 1..=(MAX_TERMINAL_SNAPSHOT_RETENTION as u64 + 1) {
            manager
                .submit(make_submission(request_id, request_id, "qwen3"))
                .unwrap();
            manager.cancel(RequestId(request_id)).unwrap();
            manager
                .mark_terminal_cleaned(RequestId(request_id))
                .unwrap();
        }

        assert!(manager.snapshot(RequestId(1)).is_none());
        assert_eq!(
            manager
                .snapshot(RequestId(MAX_TERMINAL_SNAPSHOT_RETENTION as u64 + 1))
                .unwrap()
                .state,
            RequestState::Cancelled
        );
        assert_eq!(
            manager.terminal_snapshot_order.len(),
            MAX_TERMINAL_SNAPSHOT_RETENTION
        );
    }

    #[test]
    fn applies_prefix_reuse_to_processed_prompt_position() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(5, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        manager.apply_prefix_reuse(RequestId(5), 3).unwrap();

        assert_eq!(
            manager
                .snapshot(RequestId(5))
                .unwrap()
                .processed_prompt_tokens,
            3
        );
    }

    #[test]
    fn rollback_prefix_reuse_restores_empty_request_state() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(5, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        manager.apply_prefix_reuse(RequestId(5), 3).unwrap();

        manager.rollback_prefix_reuse(RequestId(5), 3).unwrap();

        let snapshot = manager.snapshot(RequestId(5)).unwrap();
        assert_eq!(snapshot.processed_prompt_tokens, 0);
        assert!(
            manager
                .record(RequestId(5))
                .unwrap()
                .block_table
                .block_ids
                .is_empty()
        );
    }

    #[test]
    fn applies_prefill_runner_output_and_returns_request_to_runnable() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        manager
            .apply_schedule_plan(&SchedulePlan {
                step_id: StepId(1),
                selected_requests: vec![RequestId(1)],
                deferred_requests: vec![],
                memory_blocked_requests: vec![],
                execution_batch: None,
            })
            .unwrap();

        let summary = manager
            .apply_execution_results(
                &RunnerOutput {
                    step_id: StepId(1),
                    request_updates: vec![crate::runner::RequestExecutionUpdate {
                        request_id: RequestId(1),
                        tokens_executed: 3,
                        output_token: None,
                        stop_reason: None,
                        error: None,
                    }],
                    logits_handles: vec![],
                    logits_outputs: vec![],
                    kv_write_summary: KvWriteSummary {
                        tokens_written: 3,
                        blocks_touched: 1,
                    },
                    route_metadata: crate::scheduler::RouteMetadata::empty(),
                    execution_status: ExecutionStatus::Success,
                },
                &[],
                &[],
            )
            .unwrap();

        assert_eq!(summary.ttft_events, 0);
        let snapshot = manager.snapshot(RequestId(1)).unwrap();
        assert_eq!(snapshot.state, RequestState::Runnable);
        assert_eq!(snapshot.processed_prompt_tokens, 3);
    }

    #[test]
    fn applies_decode_runner_output_and_counts_ttft() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        {
            let record = manager.records.get_mut(&RequestId(1)).unwrap();
            record.processed_prompt_tokens = 3;
            record.mark_runnable().unwrap();
            record.start_running().unwrap();
        }

        let summary = manager
            .apply_execution_results(
                &RunnerOutput {
                    step_id: StepId(2),
                    request_updates: vec![crate::runner::RequestExecutionUpdate {
                        request_id: RequestId(1),
                        tokens_executed: 1,
                        output_token: None,
                        stop_reason: None,
                        error: None,
                    }],
                    logits_handles: vec![RequestId(1)],
                    logits_outputs: vec![],
                    kv_write_summary: KvWriteSummary {
                        tokens_written: 1,
                        blocks_touched: 1,
                    },
                    route_metadata: crate::scheduler::RouteMetadata::empty(),
                    execution_status: ExecutionStatus::Success,
                },
                &[crate::sampling::SampledToken {
                    request_id: RequestId(1),
                    token_id: 99,
                    stop_reason: None,
                    logprob: Some(0.0),
                }],
                &[RequestId(1)],
            )
            .unwrap();

        assert_eq!(summary.ttft_events, 1);
        let snapshot = manager.snapshot(RequestId(1)).unwrap();
        assert_eq!(snapshot.state, RequestState::Runnable);
        assert_eq!(snapshot.generated_tokens, vec![99]);
        assert_eq!(snapshot.generated_token_logprobs, vec![Some(0.0)]);
    }

    #[test]
    fn sampler_error_transitions_request_to_failed_without_fake_token() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        {
            let record = manager.records.get_mut(&RequestId(1)).unwrap();
            record.processed_prompt_tokens = 3;
            record.mark_runnable().unwrap();
            record.start_running().unwrap();
        }

        let summary = manager
            .apply_execution_results(
                &RunnerOutput {
                    step_id: StepId(2),
                    request_updates: vec![crate::runner::RequestExecutionUpdate {
                        request_id: RequestId(1),
                        tokens_executed: 1,
                        output_token: None,
                        stop_reason: None,
                        error: None,
                    }],
                    logits_handles: vec![RequestId(1)],
                    logits_outputs: vec![],
                    kv_write_summary: KvWriteSummary {
                        tokens_written: 1,
                        blocks_touched: 1,
                    },
                    route_metadata: crate::scheduler::RouteMetadata::empty(),
                    execution_status: ExecutionStatus::Success,
                },
                &[crate::sampling::SampledToken {
                    request_id: RequestId(1),
                    token_id: 100,
                    stop_reason: Some(StopReason::Error),
                    logprob: None,
                }],
                &[RequestId(1)],
            )
            .unwrap();

        assert_eq!(summary.ttft_events, 0);
        let snapshot = manager.snapshot(RequestId(1)).unwrap();
        assert_eq!(snapshot.state, RequestState::Failed);
        assert!(snapshot.generated_tokens.is_empty());
        assert!(snapshot.generated_token_logprobs.is_empty());
    }

    #[test]
    fn rejects_duplicate_request_submission() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        let error = manager.submit(make_submission(1, 2, "qwen3")).unwrap_err();

        assert_eq!(error, RequestManagerError::DuplicateRequest(RequestId(1)));
    }

    #[test]
    fn retry_memory_blocked_moves_blocked_requests_to_runnable() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        manager
            .apply_schedule_plan(&SchedulePlan {
                step_id: StepId(1),
                selected_requests: vec![],
                deferred_requests: vec![],
                memory_blocked_requests: vec![RequestId(1)],
                execution_batch: None,
            })
            .unwrap();

        assert_eq!(
            manager.snapshot(RequestId(1)).unwrap().state,
            RequestState::BlockedOnMemory
        );

        let retried = manager.retry_memory_blocked().unwrap();

        assert_eq!(retried, vec![RequestId(1)]);
        assert_eq!(
            manager.snapshot(RequestId(1)).unwrap().state,
            RequestState::Runnable
        );
    }

    #[test]
    fn prefix_reuse_rejects_advancement_past_prompt_length() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();

        let error = manager.apply_prefix_reuse(RequestId(1), 99).unwrap_err();

        match error {
            RequestManagerError::ProgressInvariantViolation { request_id, .. } => {
                assert_eq!(request_id, RequestId(1));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn prefix_reuse_rejects_backward_movement() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        manager.apply_prefix_reuse(RequestId(1), 2).unwrap();

        let error = manager.apply_prefix_reuse(RequestId(1), 1).unwrap_err();

        match error {
            RequestManagerError::ProgressInvariantViolation { request_id, .. } => {
                assert_eq!(request_id, RequestId(1));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn schedule_plan_rejects_unknown_request() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        let error = manager
            .apply_schedule_plan(&SchedulePlan {
                step_id: StepId(1),
                selected_requests: vec![RequestId(99)],
                deferred_requests: vec![],
                memory_blocked_requests: vec![],
                execution_batch: None,
            })
            .unwrap_err();

        assert_eq!(error, RequestManagerError::UnknownRequest(RequestId(99)));
    }

    #[test]
    fn schedule_plan_rejects_missing_runnable_request() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.submit(make_submission(2, 2, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();

        let error = manager
            .apply_schedule_plan(&SchedulePlan {
                step_id: StepId(1),
                selected_requests: vec![RequestId(1)],
                deferred_requests: vec![],
                memory_blocked_requests: vec![],
                execution_batch: None,
            })
            .unwrap_err();

        assert_eq!(
            error,
            RequestManagerError::ProgressInvariantViolation {
                request_id: RequestId(2),
                message: "runnable request missing from schedule plan",
            }
        );
    }

    #[test]
    fn runner_error_transitions_request_to_failed() {
        let mut manager = RequestManager::new(CacheGroupId(7));

        manager.submit(make_submission(1, 1, "qwen3")).unwrap();
        manager.admit_waiting().unwrap();
        manager
            .apply_schedule_plan(&SchedulePlan {
                step_id: StepId(1),
                selected_requests: vec![RequestId(1)],
                deferred_requests: vec![],
                memory_blocked_requests: vec![],
                execution_batch: None,
            })
            .unwrap();

        let summary = manager
            .apply_execution_results(
                &RunnerOutput {
                    step_id: StepId(1),
                    request_updates: vec![crate::runner::RequestExecutionUpdate {
                        request_id: RequestId(1),
                        tokens_executed: 0,
                        output_token: None,
                        stop_reason: None,
                        error: Some("simulated runner failure".into()),
                    }],
                    logits_handles: vec![],
                    logits_outputs: vec![],
                    kv_write_summary: KvWriteSummary {
                        tokens_written: 0,
                        blocks_touched: 0,
                    },
                    route_metadata: crate::scheduler::RouteMetadata::empty(),
                    execution_status: ExecutionStatus::Success,
                },
                &[],
                &[],
            )
            .unwrap();

        assert_eq!(summary.ttft_events, 0);
        assert_eq!(
            manager.snapshot(RequestId(1)).unwrap().state,
            RequestState::Failed
        );
    }
}
