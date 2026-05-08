use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use crate::execution_plan::{DeterministicExecutionPlanResolver, ExecutionPlanResolver};
use crate::ids::{CacheGroupId, RequestId, StepId};
use crate::kv::{
    AllocationStatus, FreeResult, KvManager, KvManagerConfig, KvManagerError, PrefixLookupResult,
};
use crate::metal::{MetalBringupRunner, MetalBringupSampler, MetalRuntimeError};
use crate::request::RequestSnapshot;
use crate::request::RequestSubmission;
use crate::request_manager::{RequestManager, RequestManagerError, RunnerApplySummary};
#[cfg(test)]
use crate::runner::RequestLogitsOutput;
use crate::runner::{
    DeterministicRunner, ExecutionRunner, ResolvedBlockTable, RunnerInput, RunnerOutput,
    RunnerRequestContext,
};
use crate::sampling::{
    DeterministicSampler, SampledToken, SamplerInput, SamplerRequest, TokenSampler,
    sampling_params_allow_deterministic_argmax_fast_path,
};
use crate::scheduler::{
    ExecutionBatch, ExecutionItem, ExecutionMode, SchedulePlan, Scheduler, SchedulerInput,
};
use thiserror::Error;
use tracing::{debug, debug_span, field, trace};

#[derive(Clone, Debug, PartialEq)]
pub enum EngineEvent {
    Submit(RequestId),
    Cancel(RequestId),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepMetrics {
    pub step_id: Option<StepId>,
    pub scheduled_requests: u32,
    pub scheduled_tokens: u32,
    pub ttft_events: u32,
    pub prefix_hits: u32,
    pub kv_usage_blocks: u32,
    pub evictions: u32,
    pub cpu_time_us: u64,
    pub runner_time_us: u64,
}

#[derive(Debug)]
pub struct EngineCore {
    request_manager: RequestManager,
    kv_manager: KvManager,
    scheduler: Scheduler,
    execution_plan_resolver: Box<dyn ExecutionPlanResolver>,
    runner: Box<dyn ExecutionRunner>,
    sampler: Box<dyn TokenSampler>,
    next_step_id: u64,
}

impl EngineCore {
    pub fn new(cache_group_id: CacheGroupId) -> Self {
        Self::with_kv_config(KvManagerConfig::new(cache_group_id, 16, 1024))
    }

    pub fn with_kv_config(kv_config: KvManagerConfig) -> Self {
        Self::with_runtime_components(kv_config, DeterministicRunner, DeterministicSampler)
    }

    pub fn with_metal_bringup_runner(
        kv_config: KvManagerConfig,
        build_dir: impl AsRef<Path>,
    ) -> Result<Self, MetalRuntimeError> {
        Self::with_metal_bringup_runner_and_model_artifacts(kv_config, build_dir, None)
    }

    pub fn with_metal_bringup_runner_and_model_artifacts(
        kv_config: KvManagerConfig,
        build_dir: impl AsRef<Path>,
        model_artifacts_dir: Option<&Path>,
    ) -> Result<Self, MetalRuntimeError> {
        let build_dir = build_dir.as_ref();
        let runner =
            MetalBringupRunner::from_build_dir_and_model_artifacts(build_dir, model_artifacts_dir)?;
        let sampler = MetalBringupSampler::from_build_dir(build_dir)?;
        runner
            .bringup()
            .assets()
            .validate_block_size_tokens(kv_config.block_size_tokens)?;
        Ok(Self::with_runtime_components(kv_config, runner, sampler))
    }

    pub fn with_runtime_components<R, S>(kv_config: KvManagerConfig, runner: R, sampler: S) -> Self
    where
        R: ExecutionRunner + 'static,
        S: TokenSampler + 'static,
    {
        Self::with_runtime_components_and_planner(
            kv_config,
            DeterministicExecutionPlanResolver,
            runner,
            sampler,
        )
    }

    pub fn with_runtime_components_and_planner<P, R, S>(
        kv_config: KvManagerConfig,
        execution_plan_resolver: P,
        runner: R,
        sampler: S,
    ) -> Self
    where
        P: ExecutionPlanResolver + 'static,
        R: ExecutionRunner + 'static,
        S: TokenSampler + 'static,
    {
        Self {
            request_manager: RequestManager::new(kv_config.cache_group_id),
            kv_manager: KvManager::new(kv_config),
            scheduler: Scheduler::new(),
            execution_plan_resolver: Box::new(execution_plan_resolver),
            runner: Box::new(runner),
            sampler: Box::new(sampler),
            next_step_id: 0,
        }
    }

    pub fn request_manager(&self) -> &RequestManager {
        &self.request_manager
    }

    pub fn kv_manager(&self) -> &KvManager {
        &self.kv_manager
    }

    pub fn last_metal_dispatch(&self) -> Option<crate::metal::MetalDispatchTrace> {
        self.runner.metal_dispatch_trace()
    }

    pub fn native_model_artifacts_summary(
        &self,
    ) -> Option<crate::model::NativeModelArtifactsSummary> {
        self.runner.native_model_artifacts_summary()
    }

    pub fn native_model_binding_summary(&self) -> Option<crate::runner::NativeModelBindingSummary> {
        self.runner.native_model_binding_summary()
    }

    pub fn embed(
        &self,
        token_ids: &[u32],
        pooling: crate::runner::EmbeddingPooling,
        normalize: bool,
    ) -> Result<Vec<f32>, &'static str> {
        self.runner.embed(token_ids, pooling, normalize)
    }

    pub fn embed_batch(
        &self,
        batch: &[Vec<u32>],
        pooling: crate::runner::EmbeddingPooling,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>, &'static str> {
        self.runner.embed_batch(batch, pooling, normalize)
    }

    pub fn submit(&mut self, submission: RequestSubmission) -> Result<RequestId, EngineCoreError> {
        validate_submission(&submission)?;
        let request_id = submission.request_id;
        self.kv_manager
            .register_request(request_id, submission.input_tokens.clone())?;
        if let Err(error) = self.request_manager.submit(submission) {
            let _ = self.kv_manager.free(request_id);
            return Err(error.into());
        }
        Ok(request_id)
    }

    pub fn cancel(&mut self, request_id: RequestId) -> Result<(), EngineCoreError> {
        self.request_manager.cancel(request_id)?;
        let _ = self.drain_terminal_cleanup()?;
        Ok(())
    }

    pub fn step(
        &mut self,
        global_token_budget: u32,
        deterministic_mode: bool,
    ) -> Result<EngineStepOutcome, EngineCoreError> {
        let step_started = Instant::now();
        let step_span = debug_span!(
            "engine.step",
            step_id = field::Empty,
            global_token_budget,
            deterministic_mode
        );
        let _step_entered = step_span.enter();
        let _ = self.kv_manager.take_recent_evictions();

        let mut cleanup_results = self.drain_terminal_cleanup()?;
        let retried_memory_blocked = self.request_manager.retry_memory_blocked()?;
        let admitted_requests = self.request_manager.admit_waiting()?;
        self.refresh_execution_plan_refs()?;
        let step_id = self.allocate_step_id()?;
        step_span.record("step_id", step_id.0);
        trace!(
            cleanup_results = cleanup_results.len(),
            retried_memory_blocked = retried_memory_blocked.len(),
            admitted_requests = admitted_requests.len(),
            "prepared step state"
        );

        let schedule_plan = self.scheduler.plan(&SchedulerInput {
            step_id,
            request_snapshots: self.request_manager.snapshots(),
            memory_pressure: self.kv_manager.memory_pressure(),
            global_token_budget,
        });
        trace!(
            scheduled_requests = schedule_plan.selected_requests.len(),
            deferred_requests = schedule_plan.deferred_requests.len(),
            memory_blocked_requests = schedule_plan.memory_blocked_requests.len(),
            has_execution_batch = schedule_plan.execution_batch.is_some(),
            "scheduler produced initial plan"
        );
        let (mut schedule_plan, prefix_hits) =
            self.resolve_kv_schedule_plan(schedule_plan, global_token_budget)?;
        trace!(
            scheduled_requests = schedule_plan.selected_requests.len(),
            deferred_requests = schedule_plan.deferred_requests.len(),
            memory_blocked_requests = schedule_plan.memory_blocked_requests.len(),
            has_execution_batch = schedule_plan.execution_batch.is_some(),
            prefix_hits,
            "resolved schedule plan against KV state"
        );
        self.request_manager.apply_schedule_plan(&schedule_plan)?;

        let (runner_output, sampled_tokens, runner_summary, runner_time_us) =
            self.dispatch_runner(&mut schedule_plan)?;
        cleanup_results.extend(self.drain_terminal_cleanup()?);

        let scheduled_tokens = schedule_plan
            .execution_batch
            .as_ref()
            .map(|batch| batch.total_scheduled_tokens)
            .unwrap_or(0);

        let cpu_time_us = step_started.elapsed().as_micros() as u64;
        let metrics = StepMetrics {
            step_id: Some(step_id),
            scheduled_requests: schedule_plan.selected_requests.len() as u32,
            scheduled_tokens,
            ttft_events: runner_summary.ttft_events,
            prefix_hits,
            kv_usage_blocks: self.kv_manager.used_block_count(),
            evictions: self.kv_manager.take_recent_evictions(),
            cpu_time_us,
            runner_time_us,
        };
        debug!(
            scheduled_requests = metrics.scheduled_requests,
            scheduled_tokens = metrics.scheduled_tokens,
            ttft_events = metrics.ttft_events,
            prefix_hits = metrics.prefix_hits,
            kv_usage_blocks = metrics.kv_usage_blocks,
            evictions = metrics.evictions,
            cpu_time_us = metrics.cpu_time_us,
            runner_time_us = metrics.runner_time_us,
            cleanup_results = cleanup_results.len(),
            runner_executed = runner_output.is_some(),
            "completed engine step"
        );

        Ok(EngineStepOutcome {
            admitted_requests,
            cleanup_results,
            metrics,
            schedule_plan,
            runner_output,
            sampled_tokens,
        })
    }

    fn allocate_step_id(&mut self) -> Result<StepId, EngineCoreError> {
        let step_id = StepId(self.next_step_id);
        self.next_step_id = self
            .next_step_id
            .checked_add(1)
            .ok_or(EngineCoreError::StepIdOverflow)?;
        Ok(step_id)
    }

    fn resolve_kv_schedule_plan(
        &mut self,
        schedule_plan: SchedulePlan,
        global_token_budget: u32,
    ) -> Result<(SchedulePlan, u32), EngineCoreError> {
        let (prefix_reuse, schedule_plan) =
            self.apply_prefix_reuse(schedule_plan, global_token_budget)?;
        let prefix_hits = prefix_reuse.len() as u32;
        let Some(mut execution_batch) = schedule_plan.execution_batch else {
            return Ok((schedule_plan, prefix_hits));
        };

        let mut selected_requests = Vec::new();
        let mut memory_blocked_requests = schedule_plan.memory_blocked_requests;
        let mut allocated_items = Vec::new();
        let batch_items = std::mem::take(&mut execution_batch.items);

        for item in batch_items {
            let rid = item.request_id;
            let allocation_plan = self.kv_manager.allocate(rid, item.scheduled_token_count)?;
            trace!(
                request_id = rid.0,
                scheduled_token_count = item.scheduled_token_count,
                allocation_status = ?allocation_plan.allocation_status,
                "resolved KV allocation for scheduled request"
            );
            match allocation_plan.allocation_status {
                AllocationStatus::Allocated => {
                    self.sync_request_block_table(rid)?;
                    selected_requests.push(rid);
                    let request_snapshot = self.request_manager.snapshot(rid);
                    allocated_items.push(Self::annotate_prefix_reuse(
                        item,
                        prefix_reuse.get(&rid),
                        request_snapshot.as_ref(),
                    ));
                }
                AllocationStatus::InsufficientCapacity | AllocationStatus::Deferred => {
                    if let Some(lookup) = prefix_reuse.get(&rid) {
                        self.kv_manager.rollback_prefix_share(rid, lookup)?;
                        self.request_manager
                            .rollback_prefix_reuse(rid, lookup.matched_token_count)?;
                        self.sync_request_block_table(rid)?;
                    }
                    memory_blocked_requests.push(rid);
                }
            }
        }

        let execution_batch =
            Self::rebuild_execution_batch(execution_batch, allocated_items, &prefix_reuse);

        let resolved_plan = SchedulePlan {
            step_id: schedule_plan.step_id,
            selected_requests,
            deferred_requests: schedule_plan.deferred_requests,
            memory_blocked_requests,
            execution_batch,
        };

        if resolved_plan.execution_batch.is_none()
            && !resolved_plan.memory_blocked_requests.is_empty()
            && !resolved_plan.deferred_requests.is_empty()
        {
            debug!(
                step_id = resolved_plan.step_id.0,
                blocked_requests = resolved_plan.memory_blocked_requests.len(),
                deferred_requests = resolved_plan.deferred_requests.len(),
                "retrying scheduler after KV capacity blocked the initial batch"
            );
            let blocked_request_ids = resolved_plan
                .memory_blocked_requests
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let fallback_schedule_plan = self.scheduler.plan(&SchedulerInput {
                step_id: resolved_plan.step_id,
                request_snapshots: self
                    .request_manager
                    .snapshots()
                    .into_iter()
                    .filter(|snapshot| !blocked_request_ids.contains(&snapshot.request_id))
                    .collect(),
                memory_pressure: self.kv_manager.memory_pressure(),
                global_token_budget,
            });
            let (fallback_plan, fallback_prefix_hits) =
                self.resolve_kv_schedule_plan(fallback_schedule_plan, global_token_budget)?;

            let selected_requests = fallback_plan.selected_requests;
            let memory_blocked_requests = resolved_plan
                .memory_blocked_requests
                .iter()
                .chain(fallback_plan.memory_blocked_requests.iter())
                .copied()
                .collect::<BTreeSet<_>>();
            let deferred_requests = resolved_plan
                .deferred_requests
                .iter()
                .chain(fallback_plan.deferred_requests.iter())
                .copied()
                .filter(|request_id| {
                    !selected_requests.contains(request_id)
                        && !memory_blocked_requests.contains(request_id)
                })
                .collect::<BTreeSet<_>>();

            return Ok((
                SchedulePlan {
                    step_id: fallback_plan.step_id,
                    selected_requests,
                    deferred_requests: deferred_requests.into_iter().collect(),
                    memory_blocked_requests: memory_blocked_requests.into_iter().collect(),
                    execution_batch: fallback_plan.execution_batch,
                },
                prefix_hits + fallback_prefix_hits,
            ));
        }

        Ok((resolved_plan, prefix_hits))
    }

    fn apply_prefix_reuse(
        &mut self,
        schedule_plan: SchedulePlan,
        global_token_budget: u32,
    ) -> Result<(BTreeMap<RequestId, PrefixLookupResult>, SchedulePlan), EngineCoreError> {
        let Some(execution_batch) = &schedule_plan.execution_batch else {
            return Ok((BTreeMap::new(), schedule_plan));
        };

        let mut pending_prefix_reuse = Vec::new();
        for item in &execution_batch.items {
            if item.mode != ExecutionMode::Prefill {
                continue;
            }

            let lookup = self.lookup_prefix(item.request_id)?;
            if !lookup.hit {
                continue;
            }

            debug!(
                request_id = item.request_id.0,
                matched_tokens = lookup.matched_token_count,
                matched_blocks = lookup.matched_blocks.len(),
                retained_cache_hit = lookup.uses_retained_cache(),
                "reusing prompt prefix before scheduling runner work"
            );
            self.request_manager
                .validate_prefix_reuse(item.request_id, lookup.matched_token_count)?;
            self.kv_manager
                .validate_prefix_share(item.request_id, &lookup)?;
            pending_prefix_reuse.push((item.request_id, lookup));
        }

        let mut prefix_reuse = BTreeMap::new();
        for (request_id, lookup) in pending_prefix_reuse {
            self.kv_manager.share_prefix(request_id, &lookup)?;
            self.sync_request_block_table(request_id)?;
            self.request_manager
                .apply_prefix_reuse(request_id, lookup.matched_token_count)?;
            prefix_reuse.insert(request_id, lookup);
        }

        if prefix_reuse.is_empty() {
            return Ok((prefix_reuse, schedule_plan));
        }

        self.refresh_execution_plan_refs()?;

        Ok((
            prefix_reuse,
            self.scheduler.plan(&SchedulerInput {
                step_id: schedule_plan.step_id,
                request_snapshots: self.request_manager.snapshots(),
                memory_pressure: self.kv_manager.memory_pressure(),
                global_token_budget,
            }),
        ))
    }

    fn dispatch_runner(
        &mut self,
        schedule_plan: &mut SchedulePlan,
    ) -> Result<
        (
            Option<RunnerOutput>,
            Vec<SampledToken>,
            RunnerApplySummary,
            u64,
        ),
        EngineCoreError,
    > {
        let Some(execution_batch) = schedule_plan.execution_batch.clone() else {
            return Ok((None, Vec::new(), RunnerApplySummary::default(), 0));
        };

        trace!(
            step_id = execution_batch.step_id.0,
            scheduled_requests = execution_batch.items.len(),
            scheduled_tokens = execution_batch.total_scheduled_tokens,
            "dispatching runner"
        );
        let runner_input = self.build_runner_input(execution_batch)?;
        let runner_started = Instant::now();
        let runner_output = self.runner.run(runner_input);
        let runner_time_us = runner_started.elapsed().as_micros() as u64;
        self.validate_runner_output(schedule_plan, &runner_output)?;
        let sampled_tokens = self.sample_runner_output(schedule_plan, &runner_output)?;
        let sampled_request_ids = sampled_tokens
            .iter()
            .map(|sampled| sampled.request_id)
            .collect::<Vec<_>>();
        let runner_summary = self.request_manager.apply_execution_results(
            &runner_output,
            &sampled_tokens,
            &sampled_request_ids,
        )?;

        if let Some(batch) = schedule_plan.execution_batch.as_mut() {
            batch.route_metadata = runner_output.route_metadata.clone();
        }

        debug!(
            step_id = runner_output.step_id.0,
            execution_status = ?runner_output.execution_status,
            sampled_tokens = sampled_tokens.len(),
            ttft_events = runner_summary.ttft_events,
            runner_time_us,
            "runner dispatch completed"
        );
        Ok((
            Some(runner_output),
            sampled_tokens,
            runner_summary,
            runner_time_us,
        ))
    }

    fn build_runner_input(
        &self,
        execution_batch: ExecutionBatch,
    ) -> Result<RunnerInput, EngineCoreError> {
        let mut block_tables = Vec::with_capacity(execution_batch.items.len());
        let mut seen_request_ids = BTreeSet::new();
        let mut request_contexts = Vec::with_capacity(execution_batch.items.len());

        for item in &execution_batch.items {
            block_tables.push(ResolvedBlockTable {
                request_id: item.request_id,
                block_table: self.kv_manager.block_table(item.request_id)?,
            });
            if seen_request_ids.insert(item.request_id) {
                let record = self
                    .request_manager
                    .record(item.request_id)
                    .ok_or(RequestManagerError::UnknownRequest(item.request_id))?;
                request_contexts.push(RunnerRequestContext {
                    request_id: item.request_id,
                    prompt_len: record.prompt_tokens.len() as u32,
                    processed_prompt_tokens: record.processed_prompt_tokens,
                    generated_len: record.generated_tokens.len() as u32,
                    max_output_tokens: record.max_output_tokens,
                    deterministic_argmax_sampling:
                        sampling_params_allow_deterministic_argmax_fast_path(
                            &record.sampling_params,
                        ),
                    temperature: record.sampling_params.temperature,
                });
            }
        }

        Ok(RunnerInput {
            block_size_tokens: self.kv_manager.config().block_size_tokens,
            execution_batch,
            block_tables,
            request_contexts,
        })
    }

    fn sample_runner_output(
        &self,
        schedule_plan: &SchedulePlan,
        runner_output: &RunnerOutput,
    ) -> Result<Vec<SampledToken>, EngineCoreError> {
        let Some(execution_batch) = &schedule_plan.execution_batch else {
            return Ok(Vec::new());
        };
        let sample_request_ids = self.sample_request_ids(execution_batch)?;

        let mut runner_tokens = Vec::new();
        let mut requests = Vec::new();
        let update_by_request = runner_output
            .request_updates
            .iter()
            .map(|update| (update.request_id, update))
            .collect::<BTreeMap<_, _>>();
        let logits_output_by_request = runner_output
            .logits_outputs
            .iter()
            .map(|output| (output.request_id, &output.logits))
            .collect::<BTreeMap<_, _>>();
        let logits_handles = runner_output
            .logits_handles
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();

        for request_id in sample_request_ids {
            let update = update_by_request.get(&request_id).ok_or(
                EngineCoreError::RunnerContractViolation {
                    step_id: runner_output.step_id,
                    message: "runner omitted update for sampleable request",
                },
            )?;
            if update.error.is_some()
                || matches!(update.stop_reason, Some(crate::sampling::StopReason::Error))
            {
                continue;
            }

            if let Some(token_id) = update.output_token {
                runner_tokens.push(SampledToken {
                    request_id,
                    token_id,
                    stop_reason: update.stop_reason,
                    logprob: None,
                });
                continue;
            }

            let logits = logits_output_by_request
                .get(&request_id)
                .map(|values| (*values).clone());
            let item = execution_batch
                .items
                .iter()
                .find(|item| item.request_id == request_id)
                .ok_or(EngineCoreError::RunnerContractViolation {
                    step_id: runner_output.step_id,
                    message: "runner logits handle missing from execution batch",
                })?;
            let decode_request = item.mode == ExecutionMode::Decode;
            if logits.is_none() && !logits_handles.contains(&request_id) {
                if !decode_request {
                    continue;
                }
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: runner_output.step_id,
                    message: "decode request requires logits payload, logits handle, or output token",
                });
            }

            let record = self
                .request_manager
                .record(request_id)
                .ok_or(RequestManagerError::UnknownRequest(request_id))?;
            let previous_token = item.input_token_slice.last().copied().ok_or(
                RequestManagerError::ProgressInvariantViolation {
                    request_id,
                    message: "decode item missing previous token context",
                },
            )?;

            requests.push(SamplerRequest {
                request_id,
                previous_token,
                logits,
                generated_len: record.generated_tokens.len() as u32,
                max_output_tokens: record.max_output_tokens,
                sampling_params: record.sampling_params.clone(),
            });
        }

        if requests.is_empty() {
            return Ok(runner_tokens);
        }

        let mut sampled_tokens = self.sampler.sample(SamplerInput { requests });
        runner_tokens.append(&mut sampled_tokens);
        Ok(runner_tokens)
    }

    fn validate_runner_output(
        &self,
        schedule_plan: &SchedulePlan,
        runner_output: &RunnerOutput,
    ) -> Result<(), EngineCoreError> {
        let Some(execution_batch) = &schedule_plan.execution_batch else {
            return Err(EngineCoreError::RunnerContractViolation {
                step_id: runner_output.step_id,
                message: "runner output produced without an execution batch",
            });
        };

        if runner_output.step_id != execution_batch.step_id {
            return Err(EngineCoreError::RunnerContractViolation {
                step_id: execution_batch.step_id,
                message: "runner output step_id does not match scheduled batch",
            });
        }
        let prefill_completion_request_ids =
            self.prefill_completion_request_ids(execution_batch)?;

        let batch_request_ids = execution_batch
            .items
            .iter()
            .map(|item| item.request_id)
            .collect::<BTreeSet<_>>();
        let decode_request_ids = execution_batch
            .items
            .iter()
            .filter(|item| item.mode == ExecutionMode::Decode)
            .map(|item| item.request_id)
            .collect::<BTreeSet<_>>();
        let mut seen_updates = BTreeSet::new();
        for update in &runner_output.request_updates {
            if !batch_request_ids.contains(&update.request_id) {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner emitted update for request outside execution batch",
                });
            }
            if !seen_updates.insert(update.request_id) {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner emitted duplicate request update",
                });
            }

            let item = execution_batch
                .items
                .iter()
                .find(|item| item.request_id == update.request_id)
                .expect("validated request_id should exist in execution batch");

            if item.mode == ExecutionMode::Prefill
                && update.output_token.is_some()
                && !prefill_completion_request_ids.contains(&update.request_id)
            {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "prefill updates may only emit output tokens when completing the prompt",
                });
            }

            if update.error.is_none()
                && !matches!(update.stop_reason, Some(crate::sampling::StopReason::Error))
                && update.tokens_executed != item.scheduled_token_count
            {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner tokens_executed must match scheduled_token_count for non-failed updates",
                });
            }
        }

        if seen_updates != batch_request_ids {
            return Err(EngineCoreError::RunnerContractViolation {
                step_id: execution_batch.step_id,
                message: "runner omitted update for one or more scheduled requests",
            });
        }

        let mut seen_logits = BTreeSet::new();
        for request_id in &runner_output.logits_handles {
            if !decode_request_ids.contains(request_id) {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner emitted logits handle for non-decode request",
                });
            }
            if !seen_logits.insert(*request_id) {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner emitted duplicate logits handle",
                });
            }
        }
        let mut seen_logits_outputs = BTreeSet::new();
        for output in &runner_output.logits_outputs {
            if !decode_request_ids.contains(&output.request_id)
                && !prefill_completion_request_ids.contains(&output.request_id)
            {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner emitted logits payload for non-sampleable request",
                });
            }
            if output.logits.is_empty() || output.logits.iter().any(|value| !value.is_finite()) {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner emitted invalid logits payload",
                });
            }
            if !seen_logits_outputs.insert(output.request_id) {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "runner emitted duplicate logits payload",
                });
            }
        }
        if seen_logits
            .iter()
            .any(|request_id| seen_logits_outputs.contains(request_id))
        {
            return Err(EngineCoreError::RunnerContractViolation {
                step_id: execution_batch.step_id,
                message: "runner emitted duplicate decode result source",
            });
        }

        if runner_output.execution_status == crate::runner::ExecutionStatus::Failed {
            if !runner_output.logits_handles.is_empty() || !runner_output.logits_outputs.is_empty()
            {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "failed runner output must not provide sampler inputs",
                });
            }

            for update in &runner_output.request_updates {
                if update.output_token.is_some()
                    || (update.error.is_none()
                        && !matches!(update.stop_reason, Some(crate::sampling::StopReason::Error)))
                {
                    return Err(EngineCoreError::RunnerContractViolation {
                        step_id: execution_batch.step_id,
                        message: "failed runner output must resolve each scheduled request as failed",
                    });
                }
            }
            return Ok(());
        }

        for request_id in decode_request_ids {
            let update = runner_output
                .request_updates
                .iter()
                .find(|update| update.request_id == request_id)
                .expect("validated request_id should exist in runner output");
            let failed = update.error.is_some()
                || matches!(update.stop_reason, Some(crate::sampling::StopReason::Error));
            let decode_result_sources = u32::from(update.output_token.is_some())
                + u32::from(seen_logits.contains(&request_id))
                + u32::from(seen_logits_outputs.contains(&request_id));
            if failed && decode_result_sources > 0 {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "failed decode update must not provide decode result sources",
                });
            }
            if !failed && decode_result_sources == 0 {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "decode update must provide logits payload, logits handle, or output token",
                });
            }
            if decode_result_sources > 1 {
                return Err(EngineCoreError::RunnerContractViolation {
                    step_id: execution_batch.step_id,
                    message: "decode update must not provide multiple decode result sources",
                });
            }
        }

        Ok(())
    }

    fn prefill_completion_request_ids(
        &self,
        execution_batch: &ExecutionBatch,
    ) -> Result<BTreeSet<RequestId>, EngineCoreError> {
        let mut request_ids = BTreeSet::new();
        for item in &execution_batch.items {
            if item.mode != ExecutionMode::Prefill {
                continue;
            }
            let record = self
                .request_manager
                .record(item.request_id)
                .ok_or(RequestManagerError::UnknownRequest(item.request_id))?;
            let prompt_len = record.prompt_tokens.len() as u32;
            let completes_prompt = record
                .processed_prompt_tokens
                .checked_add(item.scheduled_token_count)
                .is_some_and(|next| next == prompt_len);
            if completes_prompt && record.generated_tokens.is_empty() {
                request_ids.insert(item.request_id);
            }
        }
        Ok(request_ids)
    }

    fn sample_request_ids(
        &self,
        execution_batch: &ExecutionBatch,
    ) -> Result<Vec<RequestId>, EngineCoreError> {
        let prefill_completion_request_ids =
            self.prefill_completion_request_ids(execution_batch)?;
        Ok(execution_batch
            .items
            .iter()
            .filter(|item| {
                item.mode == ExecutionMode::Decode
                    || prefill_completion_request_ids.contains(&item.request_id)
            })
            .map(|item| item.request_id)
            .collect())
    }

    fn lookup_prefix(&self, request_id: RequestId) -> Result<PrefixLookupResult, EngineCoreError> {
        let snapshot = self
            .request_manager
            .snapshot(request_id)
            .ok_or(RequestManagerError::UnknownRequest(request_id))?;
        self.kv_manager
            .lookup_prefix(request_id, &snapshot.prompt_tokens)
            .map_err(Into::into)
    }

    fn sync_request_block_table(&mut self, request_id: RequestId) -> Result<(), EngineCoreError> {
        let block_table = self.kv_manager.block_table_snapshot(request_id)?;
        self.request_manager
            .sync_block_table(request_id, block_table)
            .map_err(Into::into)
    }

    fn refresh_execution_plan_refs(&mut self) -> Result<(), EngineCoreError> {
        for snapshot in self.request_manager.snapshots() {
            let execution_plan_binding = self.execution_plan_resolver.resolve(&snapshot);
            self.request_manager
                .set_execution_plan_binding(snapshot.request_id, execution_plan_binding)?;
        }
        Ok(())
    }

    fn annotate_prefix_reuse(
        mut item: ExecutionItem,
        lookup: Option<&PrefixLookupResult>,
        snapshot: Option<&RequestSnapshot>,
    ) -> ExecutionItem {
        item.reused_prefix_token_slice =
            Self::native_prefix_warmup_token_slice(&item, lookup, snapshot);
        if let Some(lookup) = lookup {
            item.prefix_tokens_reused = lookup.matched_token_count;
            item.prefix_blocks_reused = lookup.matched_blocks.len() as u32;
        }
        item
    }

    fn native_prefix_warmup_token_slice(
        item: &ExecutionItem,
        lookup: Option<&PrefixLookupResult>,
        snapshot: Option<&RequestSnapshot>,
    ) -> Vec<u32> {
        let Some(snapshot) = snapshot else {
            return Vec::new();
        };

        if let Some(lookup) = lookup {
            return snapshot
                .prompt_tokens
                .get(..lookup.matched_token_count as usize)
                .unwrap_or(snapshot.prompt_tokens.as_slice())
                .to_vec();
        }

        if item.mode != ExecutionMode::Decode {
            return Vec::new();
        }

        let processed_prompt_tokens =
            (snapshot.processed_prompt_tokens as usize).min(snapshot.prompt_tokens.len());
        let mut warmup_tokens = snapshot.prompt_tokens[..processed_prompt_tokens].to_vec();
        warmup_tokens.extend(snapshot.generated_tokens.iter().copied());
        warmup_tokens
    }

    fn rebuild_execution_batch(
        mut execution_batch: ExecutionBatch,
        allocated_items: Vec<ExecutionItem>,
        prefix_reuse: &BTreeMap<RequestId, PrefixLookupResult>,
    ) -> Option<ExecutionBatch> {
        if allocated_items.is_empty() {
            return None;
        }

        let allocated_request_ids = allocated_items
            .iter()
            .map(|item| item.request_id)
            .collect::<BTreeSet<_>>();
        let total_reused_blocks = prefix_reuse
            .values()
            .map(|lookup| lookup.matched_blocks.len() as u32)
            .sum::<u32>();
        let total_reused_tokens = prefix_reuse
            .values()
            .map(|lookup| lookup.matched_token_count)
            .sum::<u32>();
        let branch_prefill_requests = allocated_items
            .iter()
            .filter(|item| item.prefix_tokens_reused > 0 && item.mode == ExecutionMode::Prefill)
            .count() as u32;
        let branch_decode_requests = allocated_items
            .iter()
            .filter(|item| item.prefix_tokens_reused > 0 && item.mode == ExecutionMode::Decode)
            .count() as u32;
        let branch_prefill_tail_tokens = allocated_items
            .iter()
            .filter(|item| item.prefix_tokens_reused > 0 && item.mode == ExecutionMode::Prefill)
            .map(|item| item.scheduled_token_count)
            .sum::<u32>();
        let branch_decode_tokens = allocated_items
            .iter()
            .filter(|item| item.prefix_tokens_reused > 0 && item.mode == ExecutionMode::Decode)
            .map(|item| item.scheduled_token_count)
            .sum::<u32>();
        let max_reused_blocks_per_request = prefix_reuse
            .values()
            .map(|lookup| lookup.matched_blocks.len() as u32)
            .max()
            .unwrap_or(0);
        let retained_cache_hits = prefix_reuse
            .values()
            .filter(|lookup| lookup.uses_retained_cache())
            .count() as u32;
        let live_share_hits = prefix_reuse.len() as u32 - retained_cache_hits;
        let blocked_prefix_reuse_requests = prefix_reuse
            .keys()
            .filter(|request_id| !allocated_request_ids.contains(request_id))
            .count() as u32;
        let blocked_prefix_reuse_blocks = prefix_reuse
            .iter()
            .filter(|(request_id, _)| !allocated_request_ids.contains(request_id))
            .map(|(_, lookup)| lookup.matched_blocks.len() as u32)
            .sum::<u32>();
        let blocked_prefix_reuse_tokens = prefix_reuse
            .iter()
            .filter(|(request_id, _)| !allocated_request_ids.contains(request_id))
            .map(|(_, lookup)| lookup.matched_token_count)
            .sum::<u32>();

        execution_batch.total_scheduled_tokens = allocated_items
            .iter()
            .map(|item| item.scheduled_token_count)
            .sum();
        execution_batch.route_metadata.prefix_cache_path = Some(
            match (live_share_hits > 0, retained_cache_hits > 0) {
                (true, true) => "mixed_live_and_retained",
                (true, false) => "live_request_share",
                (false, true) => "retained_prompt_prefix_cache",
                (false, false) => "metadata_lookup",
            }
            .into(),
        );
        execution_batch.route_metadata.crossover_decisions.extend([
            ("prefix_reused_requests".into(), prefix_reuse.len() as u32),
            ("live_share_hits".into(), live_share_hits),
            ("retained_cache_hits".into(), retained_cache_hits),
            ("prefix_reused_blocks".into(), total_reused_blocks),
            ("prefix_reused_tokens".into(), total_reused_tokens),
            (
                "blocked_prefix_reuse_requests".into(),
                blocked_prefix_reuse_requests,
            ),
            (
                "blocked_prefix_reuse_blocks".into(),
                blocked_prefix_reuse_blocks,
            ),
            (
                "blocked_prefix_reuse_tokens".into(),
                blocked_prefix_reuse_tokens,
            ),
            ("branch_prefill_requests".into(), branch_prefill_requests),
            ("branch_decode_requests".into(), branch_decode_requests),
            (
                "branch_prefill_tail_tokens".into(),
                branch_prefill_tail_tokens,
            ),
            ("branch_decode_tokens".into(), branch_decode_tokens),
            (
                "max_prefix_blocks_reused_per_request".into(),
                max_reused_blocks_per_request,
            ),
        ]);
        execution_batch.items = allocated_items;
        Some(execution_batch)
    }

    fn drain_terminal_cleanup(&mut self) -> Result<Vec<FreeResult>, EngineCoreError> {
        let pending = self.request_manager.collect_terminal_cleanup();
        let mut free_results = Vec::with_capacity(pending.len());
        for request_id in pending {
            let free_result = self.kv_manager.free(request_id)?;
            self.runner.release_request_state(request_id);
            self.request_manager.mark_terminal_cleaned(request_id)?;
            free_results.push(free_result);
        }
        Ok(free_results)
    }
}

fn validate_submission(submission: &RequestSubmission) -> Result<(), EngineCoreError> {
    if submission.input_tokens.is_empty() {
        return Err(EngineCoreError::InvalidRequestSubmission {
            request_id: submission.request_id,
            message: "input_tokens must not be empty",
        });
    }
    if submission.max_output_tokens == 0 {
        return Err(EngineCoreError::InvalidRequestSubmission {
            request_id: submission.request_id,
            message: "max_output_tokens must be greater than 0",
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct EngineStepOutcome {
    pub admitted_requests: Vec<RequestId>,
    pub cleanup_results: Vec<FreeResult>,
    pub metrics: StepMetrics,
    pub schedule_plan: SchedulePlan,
    pub runner_output: Option<RunnerOutput>,
    pub sampled_tokens: Vec<SampledToken>,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum EngineCoreError {
    #[error("invalid request submission {request_id:?}: {message}")]
    InvalidRequestSubmission {
        request_id: RequestId,
        message: &'static str,
    },
    #[error("request manager error: {0}")]
    RequestManager(#[from] RequestManagerError),
    #[error("KV manager error: {0}")]
    KvManager(#[from] KvManagerError),
    #[error("runner contract violation at {step_id:?}: {message}")]
    RunnerContractViolation {
        step_id: StepId,
        message: &'static str,
    },
    #[error("step ID counter overflowed")]
    StepIdOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{ModelId, SequenceNo};
    use crate::runner::{RunnerInput, RunnerOutput};
    use crate::sampling::StopReason;
    use std::thread;
    use std::time::Duration;

    #[derive(Debug)]
    struct DelayedRunner {
        delay_ms: u64,
    }

    impl ExecutionRunner for DelayedRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            thread::sleep(Duration::from_millis(self.delay_ms));
            DeterministicRunner.run(input)
        }
    }

    #[derive(Clone, Debug)]
    struct FixedTokenSampler {
        token_id: u32,
        stop_reason: Option<StopReason>,
    }

    impl TokenSampler for FixedTokenSampler {
        fn sample(&self, input: SamplerInput) -> Vec<SampledToken> {
            input
                .requests
                .into_iter()
                .map(|request| SampledToken {
                    request_id: request.request_id,
                    token_id: self.token_id,
                    stop_reason: self.stop_reason,
                    logprob: Some(0.0),
                })
                .collect()
        }
    }

    #[derive(Debug)]
    struct PanicSampler;

    impl TokenSampler for PanicSampler {
        fn sample(&self, _input: SamplerInput) -> Vec<SampledToken> {
            panic!("sampler should not be called when runner provides output tokens")
        }
    }

    #[derive(Debug)]
    struct LogitsPayloadRunner {
        logits: Vec<f32>,
    }

    impl ExecutionRunner for LogitsPayloadRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| crate::runner::RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: None,
                    stop_reason: None,
                    error: None,
                })
                .collect::<Vec<_>>();
            let logits_outputs = input
                .execution_batch
                .items
                .iter()
                .filter(|item| item.mode == ExecutionMode::Decode)
                .map(|item| RequestLogitsOutput {
                    request_id: item.request_id,
                    logits: self.logits.clone(),
                })
                .collect::<Vec<_>>();

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles: Vec::new(),
                logits_outputs,
                kv_write_summary: crate::runner::KvWriteSummary {
                    tokens_written: input.execution_batch.total_scheduled_tokens,
                    blocks_touched: input.block_tables.len() as u32,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: crate::runner::ExecutionStatus::Success,
            }
        }
    }

    #[derive(Debug)]
    struct DuplicateDecodeSourceRunner;

    impl ExecutionRunner for DuplicateDecodeSourceRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| crate::runner::RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: None,
                    stop_reason: None,
                    error: None,
                })
                .collect::<Vec<_>>();
            let decode_request_id = input
                .execution_batch
                .items
                .iter()
                .find(|item| item.mode == ExecutionMode::Decode)
                .map(|item| item.request_id);

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles: decode_request_id.into_iter().collect(),
                logits_outputs: decode_request_id
                    .into_iter()
                    .map(|request_id| RequestLogitsOutput {
                        request_id,
                        logits: vec![0.1, 0.9, 0.2],
                    })
                    .collect(),
                kv_write_summary: crate::runner::KvWriteSummary {
                    tokens_written: input.execution_batch.total_scheduled_tokens,
                    blocks_touched: input.block_tables.len() as u32,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: crate::runner::ExecutionStatus::Success,
            }
        }
    }

    #[derive(Debug)]
    struct MissingUpdateRunner;

    impl ExecutionRunner for MissingUpdateRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates: Vec::new(),
                logits_handles: Vec::new(),
                logits_outputs: Vec::new(),
                kv_write_summary: crate::runner::KvWriteSummary {
                    tokens_written: 0,
                    blocks_touched: 0,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: crate::runner::ExecutionStatus::Success,
            }
        }
    }

    #[derive(Debug)]
    struct WrongStepIdRunner;

    impl ExecutionRunner for WrongStepIdRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let mut output = DeterministicRunner.run(input);
            output.step_id = StepId(99999);
            output
        }
    }

    #[derive(Debug)]
    struct FailedBatchRunner;

    impl ExecutionRunner for FailedBatchRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| crate::runner::RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: 0,
                    output_token: None,
                    stop_reason: Some(crate::sampling::StopReason::Error),
                    error: Some("simulated batch failure".into()),
                })
                .collect();

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles: Vec::new(),
                logits_outputs: Vec::new(),
                kv_write_summary: crate::runner::KvWriteSummary {
                    tokens_written: 0,
                    blocks_touched: 0,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: crate::runner::ExecutionStatus::Failed,
            }
        }
    }

    #[derive(Debug)]
    struct PrefillWithOutputTokenRunner;

    impl ExecutionRunner for PrefillWithOutputTokenRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| crate::runner::RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: Some(42),
                    stop_reason: None,
                    error: None,
                })
                .collect();

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles: Vec::new(),
                logits_outputs: Vec::new(),
                kv_write_summary: crate::runner::KvWriteSummary {
                    tokens_written: input.execution_batch.total_scheduled_tokens,
                    blocks_touched: 0,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: crate::runner::ExecutionStatus::Success,
            }
        }
    }

    #[derive(Debug)]
    struct WrongTokenCountRunner;

    impl ExecutionRunner for WrongTokenCountRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| crate::runner::RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count.saturating_sub(1),
                    output_token: None,
                    stop_reason: None,
                    error: None,
                })
                .collect();

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles: Vec::new(),
                logits_outputs: Vec::new(),
                kv_write_summary: crate::runner::KvWriteSummary {
                    tokens_written: 0,
                    blocks_touched: 0,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: crate::runner::ExecutionStatus::Success,
            }
        }
    }

    #[derive(Debug)]
    struct DirectDecodeTokenRunner {
        token_id: u32,
        stop_reason: Option<StopReason>,
    }

    impl ExecutionRunner for DirectDecodeTokenRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| crate::runner::RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: (item.mode == ExecutionMode::Decode).then_some(self.token_id),
                    stop_reason: (item.mode == ExecutionMode::Decode)
                        .then_some(self.stop_reason)
                        .flatten(),
                    error: None,
                })
                .collect();

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles: Vec::new(),
                logits_outputs: Vec::new(),
                kv_write_summary: crate::runner::KvWriteSummary {
                    tokens_written: input.execution_batch.total_scheduled_tokens,
                    blocks_touched: input.block_tables.len() as u32,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: crate::runner::ExecutionStatus::Success,
            }
        }
    }
    use crate::request::RequestState;
    use crate::sampling::SamplingParams;

    fn make_submission(
        request_id: u64,
        arrival_sequence: u64,
        max_output_tokens: u32,
    ) -> RequestSubmission {
        make_submission_with_prompt(
            request_id,
            arrival_sequence,
            vec![1, 2, 3, 4],
            max_output_tokens,
        )
    }

    fn make_submission_with_prompt(
        request_id: u64,
        arrival_sequence: u64,
        input_tokens: Vec<u32>,
        max_output_tokens: u32,
    ) -> RequestSubmission {
        RequestSubmission {
            request_id: RequestId(request_id),
            model_id: ModelId("qwen3".into()),
            input_tokens,
            sampling_params: SamplingParams::default(),
            max_output_tokens,
            arrival_sequence: SequenceNo(arrival_sequence),
            metadata: None,
        }
    }

    #[test]
    fn submit_rejects_empty_input_tokens() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        let error = engine
            .submit(make_submission_with_prompt(5, 1, Vec::new(), 2))
            .unwrap_err();

        assert_eq!(
            error,
            EngineCoreError::InvalidRequestSubmission {
                request_id: RequestId(5),
                message: "input_tokens must not be empty",
            }
        );
        assert!(engine.request_manager().snapshot(RequestId(5)).is_none());
    }

    #[test]
    fn submit_rejects_zero_max_output_tokens() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        let error = engine.submit(make_submission(5, 1, 0)).unwrap_err();

        assert_eq!(
            error,
            EngineCoreError::InvalidRequestSubmission {
                request_id: RequestId(5),
                message: "max_output_tokens must be greater than 0",
            }
        );
        assert!(engine.request_manager().snapshot(RequestId(5)).is_none());
    }

    #[test]
    fn step_executes_prefill_and_returns_request_to_runnable() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine.submit(make_submission(5, 1, 2)).unwrap();

        let outcome = engine.step(3, true).unwrap();

        assert_eq!(outcome.admitted_requests, vec![RequestId(5)]);
        assert_eq!(outcome.schedule_plan.selected_requests, vec![RequestId(5)]);
        assert_eq!(outcome.metrics.scheduled_requests, 1);
        assert_eq!(outcome.metrics.scheduled_tokens, 3);
        assert_eq!(outcome.metrics.kv_usage_blocks, 1);
        assert!(outcome.runner_output.is_some());

        let snapshot = engine.request_manager().snapshot(RequestId(5)).unwrap();
        assert_eq!(snapshot.state, RequestState::Runnable);
        assert_eq!(snapshot.processed_prompt_tokens, 3);
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.execution_plan_ref.as_deref()),
            Some("phase1.qwen3.dense_prefill")
        );
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.route_metadata.attention_route.as_deref()),
            Some("qwen3_prefill")
        );
    }

    #[test]
    fn step_reports_measured_runner_and_cpu_time() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            DelayedRunner { delay_ms: 2 },
            DeterministicSampler,
        );

        engine.submit(make_submission(8, 1, 1)).unwrap();

        let outcome = engine.step(3, true).unwrap();

        assert!(outcome.metrics.runner_time_us >= 1_000);
        assert!(outcome.metrics.cpu_time_us >= outcome.metrics.runner_time_us);
    }

    #[test]
    fn step_rejects_runner_output_that_omits_scheduled_updates() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            MissingUpdateRunner,
            DeterministicSampler,
        );

        engine.submit(make_submission(10, 1, 1)).unwrap();

        let error = engine
            .step(3, true)
            .expect_err("runner output without updates should fail closed");

        let EngineCoreError::RunnerContractViolation { message, .. } = error else {
            panic!("expected runner contract violation");
        };
        assert_eq!(
            message,
            "runner omitted update for one or more scheduled requests"
        );
    }

    #[test]
    fn step_rejects_runner_output_with_wrong_tokens_executed() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            WrongTokenCountRunner,
            DeterministicSampler,
        );

        engine.submit(make_submission(10, 1, 1)).unwrap();

        let error = engine
            .step(4, true)
            .expect_err("runner output with wrong tokens_executed should fail closed");

        let EngineCoreError::RunnerContractViolation { message, .. } = error else {
            panic!("expected runner contract violation");
        };
        assert_eq!(
            message,
            "runner tokens_executed must match scheduled_token_count for non-failed updates"
        );
    }

    #[test]
    fn step_keeps_requests_runnable_when_budget_is_zero() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine.submit(make_submission(6, 1, 2)).unwrap();

        let outcome = engine.step(0, true).unwrap();

        assert!(outcome.schedule_plan.selected_requests.is_empty());
        assert_eq!(outcome.schedule_plan.deferred_requests, vec![RequestId(6)]);
        assert!(outcome.runner_output.is_none());
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(6))
                .unwrap()
                .state,
            RequestState::Runnable
        );
    }

    #[test]
    fn request_finishes_after_prefill_and_decode_steps() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine.submit(make_submission(7, 1, 1)).unwrap();

        let prefill = engine.step(4, true).unwrap();
        assert_eq!(prefill.metrics.scheduled_tokens, 4);
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(7))
                .unwrap()
                .state,
            RequestState::Runnable
        );

        let decode = engine.step(1, true).unwrap();

        assert_eq!(decode.metrics.ttft_events, 1);
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(7))
                .unwrap()
                .state,
            RequestState::Finished
        );
        assert_eq!(engine.kv_manager().used_block_count(), 1);
        assert_eq!(decode.cleanup_results.len(), 1);
        assert_eq!(decode.cleanup_results[0].request_id, RequestId(7));
    }

    #[test]
    fn engine_accepts_custom_sampler_for_decode_progress() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            DeterministicRunner,
            FixedTokenSampler {
                token_id: 42,
                stop_reason: Some(StopReason::MaxOutputTokens),
            },
        );

        engine.submit(make_submission(9, 1, 1)).unwrap();
        engine.step(4, true).unwrap();

        let decode = engine.step(1, true).unwrap();
        let snapshot = engine.request_manager().snapshot(RequestId(9)).unwrap();

        assert_eq!(decode.metrics.ttft_events, 1);
        assert_eq!(snapshot.generated_tokens, vec![42]);
        assert_eq!(snapshot.state, RequestState::Finished);
    }

    #[test]
    fn engine_routes_runner_logits_payload_into_sampler() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            LogitsPayloadRunner {
                logits: vec![0.2, -1.0, 3.5, 0.7],
            },
            DeterministicSampler,
        );

        engine.submit(make_submission(10, 1, 1)).unwrap();
        engine.step(4, true).unwrap();

        let decode = engine.step(1, true).unwrap();
        let snapshot = engine.request_manager().snapshot(RequestId(10)).unwrap();

        assert_eq!(decode.metrics.ttft_events, 1);
        assert_eq!(snapshot.generated_tokens, vec![2]);
        assert_eq!(snapshot.generated_token_logprobs.len(), 1);
        assert!(
            snapshot.generated_token_logprobs[0]
                .is_some_and(|logprob| logprob.is_finite() && logprob < 0.0)
        );
        assert_eq!(snapshot.state, RequestState::Finished);
        assert_eq!(
            decode
                .runner_output
                .as_ref()
                .map(|output| output.logits_outputs.len()),
            Some(1)
        );
    }

    #[test]
    fn engine_accepts_runner_provided_decode_tokens_without_sampler() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            DirectDecodeTokenRunner {
                token_id: 77,
                stop_reason: Some(StopReason::MaxOutputTokens),
            },
            PanicSampler,
        );

        engine.submit(make_submission(11, 1, 1)).unwrap();
        engine.step(4, true).unwrap();

        let outcome = engine.step(1, true).unwrap();
        let snapshot = engine.request_manager().snapshot(RequestId(11)).unwrap();

        assert_eq!(outcome.metrics.ttft_events, 1);
        assert_eq!(snapshot.generated_tokens, vec![77]);
        assert_eq!(snapshot.state, RequestState::Finished);
        assert_eq!(
            outcome
                .runner_output
                .as_ref()
                .and_then(|output| output.route_metadata.attention_route.as_deref()),
            Some("qwen3_paged_decode")
        );
    }

    #[test]
    fn step_rejects_runner_output_with_duplicate_decode_result_sources() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            DuplicateDecodeSourceRunner,
            DeterministicSampler,
        );

        engine.submit(make_submission(12, 1, 1)).unwrap();
        engine.step(4, true).unwrap();

        let error = engine
            .step(1, true)
            .expect_err("duplicate decode result sources should fail closed");

        let EngineCoreError::RunnerContractViolation { message, .. } = error else {
            panic!("expected runner contract violation");
        };
        assert_eq!(message, "runner emitted duplicate decode result source");
    }

    #[test]
    fn step_allows_prefill_candidate_to_join_live_decode_after_prefix_reuse() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine.submit(make_submission(21, 1, 4)).unwrap();
        engine.step(4, true).unwrap();

        engine.submit(make_submission(22, 2, 4)).unwrap();

        let outcome = engine.step(8, true).unwrap();

        assert_eq!(
            outcome.schedule_plan.selected_requests,
            vec![RequestId(21), RequestId(22)]
        );
        assert!(outcome.schedule_plan.deferred_requests.is_empty());
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.execution_plan_ref.as_deref()),
            Some("phase1.qwen3.paged_decode")
        );
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.route_metadata.attention_route.as_deref()),
            Some("qwen3_paged_decode")
        );
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.route_metadata.prefix_cache_path.as_deref()),
            Some("live_request_share")
        );
    }

    #[test]
    fn step_marks_unallocatable_requests_blocked_on_memory() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 1));

        engine.submit(make_submission(1, 1, 1)).unwrap();
        engine.submit(make_submission(2, 2, 1)).unwrap();

        let outcome = engine.step(8, true).unwrap();

        assert_eq!(outcome.schedule_plan.selected_requests, vec![RequestId(1)]);
        assert_eq!(
            outcome.schedule_plan.memory_blocked_requests,
            vec![RequestId(2)]
        );
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(2))
                .unwrap()
                .state,
            RequestState::BlockedOnMemory
        );
    }

    #[test]
    fn step_reports_step_id_overflow_without_panicking() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 1));
        engine.next_step_id = u64::MAX;

        let error = engine.step(1, true).unwrap_err();

        assert_eq!(error, EngineCoreError::StepIdOverflow);
    }

    #[test]
    fn blocked_request_retries_after_capacity_is_freed() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 8, 1));

        engine.submit(make_submission(1, 1, 1)).unwrap();
        engine.submit(make_submission(2, 2, 1)).unwrap();

        engine.step(8, true).unwrap();
        engine.step(1, true).unwrap();

        let outcome = engine.step(4, true).unwrap();

        assert_eq!(outcome.schedule_plan.selected_requests, vec![RequestId(2)]);
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(2))
                .unwrap()
                .processed_prompt_tokens,
            4
        );
    }

    #[test]
    fn blocked_full_prefix_reuse_remains_visible_in_route_metadata() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 3));
        let shared_prompt = vec![10, 11, 12, 13, 14, 15, 16, 17];

        engine
            .submit(make_submission_with_prompt(1, 1, shared_prompt.clone(), 1))
            .unwrap();
        engine.step(8, true).unwrap();

        engine
            .submit(make_submission_with_prompt(2, 2, shared_prompt, 1))
            .unwrap();

        let outcome = engine.step(4, true).unwrap();
        let route_metadata = outcome
            .schedule_plan
            .execution_batch
            .as_ref()
            .map(|batch| &batch.route_metadata)
            .expect("decode step should still execute while reused request is blocked");

        assert_eq!(outcome.schedule_plan.selected_requests, vec![RequestId(1)]);
        assert_eq!(
            outcome.schedule_plan.memory_blocked_requests,
            vec![RequestId(2)]
        );
        assert_eq!(
            route_metadata.prefix_cache_path.as_deref(),
            Some("live_request_share")
        );
        assert_eq!(
            route_metadata
                .crossover_decisions
                .iter()
                .find(|(key, _)| key == "prefix_reused_tokens")
                .map(|(_, value)| *value),
            Some(8)
        );
        assert_eq!(
            route_metadata
                .crossover_decisions
                .iter()
                .find(|(key, _)| key == "prefix_reused_blocks")
                .map(|(_, value)| *value),
            Some(2)
        );
        assert_eq!(
            route_metadata
                .crossover_decisions
                .iter()
                .find(|(key, _)| key == "blocked_prefix_reuse_requests")
                .map(|(_, value)| *value),
            Some(1)
        );
        assert_eq!(
            route_metadata
                .crossover_decisions
                .iter()
                .find(|(key, _)| key == "blocked_prefix_reuse_blocks")
                .map(|(_, value)| *value),
            Some(2)
        );
        assert_eq!(
            route_metadata
                .crossover_decisions
                .iter()
                .find(|(key, _)| key == "blocked_prefix_reuse_tokens")
                .map(|(_, value)| *value),
            Some(8)
        );
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(2))
                .unwrap()
                .state,
            RequestState::BlockedOnMemory
        );
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(2))
                .unwrap()
                .processed_prompt_tokens,
            0
        );
    }

    #[test]
    fn engine_eviction_preserves_shorter_retained_prefix_for_future_request() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 2));

        engine
            .submit(make_submission_with_prompt(
                1,
                1,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                1,
            ))
            .unwrap();
        engine.step(8, true).unwrap();
        engine.cancel(RequestId(1)).unwrap();

        engine
            .submit(make_submission_with_prompt(2, 2, vec![9, 10, 11, 12], 1))
            .unwrap();
        let eviction = engine.step(4, true).unwrap();

        assert_eq!(eviction.metrics.evictions, 1);
        assert_eq!(engine.kv_manager().used_block_count(), 2);

        engine
            .submit(make_submission_with_prompt(3, 3, vec![1, 2, 3, 4, 99], 1))
            .unwrap();

        let lookup = engine
            .kv_manager()
            .lookup_prefix(RequestId(3), &[1, 2, 3, 4, 99])
            .unwrap();

        assert_eq!(lookup.matched_token_count, 4);
        assert!(lookup.uses_retained_cache());
    }

    #[test]
    fn blocked_prefix_reuse_rolls_back_refs_so_capacity_can_recover() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 3));
        let shared_prefix = vec![10, 11, 12, 13, 14, 15, 16, 17];

        engine
            .submit(make_submission_with_prompt(1, 1, shared_prefix.clone(), 2))
            .unwrap();
        engine.step(8, true).unwrap();

        let mut extended_prompt = shared_prefix;
        extended_prompt.extend([90, 91, 92, 93, 94]);
        engine
            .submit(make_submission_with_prompt(2, 2, extended_prompt, 1))
            .unwrap();

        let blocked = engine.step(5, true).unwrap();
        assert_eq!(
            blocked.schedule_plan.memory_blocked_requests,
            vec![RequestId(2)]
        );
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(2))
                .unwrap()
                .processed_prompt_tokens,
            0
        );

        engine.step(1, true).unwrap();
        let recovered = engine.step(4, true).unwrap();

        assert_eq!(
            recovered.schedule_plan.selected_requests,
            vec![RequestId(2)]
        );
    }

    #[test]
    fn step_reuses_full_block_prefix_for_later_request() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));
        let shared_prefix = vec![10, 11, 12, 13, 14, 15, 16, 17];

        engine
            .submit(make_submission_with_prompt(1, 1, shared_prefix.clone(), 2))
            .unwrap();
        engine.step(8, true).unwrap();

        let mut extended_prompt = shared_prefix;
        extended_prompt.extend([90, 91, 92, 93]);
        engine
            .submit(make_submission_with_prompt(2, 2, extended_prompt, 1))
            .unwrap();

        let outcome = engine.step(5, true).unwrap();
        let target_item = outcome
            .schedule_plan
            .execution_batch
            .as_ref()
            .and_then(|batch| {
                batch
                    .items
                    .iter()
                    .find(|item| item.request_id == RequestId(2))
            })
            .expect("target request should be scheduled after prefix reuse");

        assert_eq!(outcome.metrics.prefix_hits, 1);
        assert_eq!(target_item.prefix_tokens_reused, 8);
        assert_eq!(target_item.prefix_blocks_reused, 2);
        assert_eq!(
            target_item.reused_prefix_token_slice,
            vec![10, 11, 12, 13, 14, 15, 16, 17]
        );
        assert_eq!(target_item.scheduled_token_count, 4);
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.route_metadata.prefix_cache_path.as_deref()),
            Some("live_request_share")
        );
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| {
                    batch
                        .route_metadata
                        .crossover_decisions
                        .iter()
                        .find(|(key, _)| key == "branch_prefill_tail_tokens")
                        .map(|(_, value)| *value)
                }),
            Some(4)
        );
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(2))
                .unwrap()
                .processed_prompt_tokens,
            12
        );
    }

    #[test]
    fn step_moves_fully_reused_prompt_directly_to_decode() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));
        let shared_prompt = vec![10, 11, 12, 13, 14, 15, 16, 17];

        engine
            .submit(make_submission_with_prompt(1, 1, shared_prompt.clone(), 4))
            .unwrap();
        engine.step(8, true).unwrap();

        engine
            .submit(make_submission_with_prompt(2, 2, shared_prompt, 4))
            .unwrap();

        let outcome = engine.step(4, true).unwrap();
        let target_item = outcome
            .schedule_plan
            .execution_batch
            .as_ref()
            .and_then(|batch| {
                batch
                    .items
                    .iter()
                    .find(|item| item.request_id == RequestId(2))
            })
            .expect("target request should be scheduled after full prefix reuse");

        assert_eq!(outcome.metrics.prefix_hits, 1);
        assert_eq!(target_item.mode, ExecutionMode::Decode);
        assert_eq!(target_item.prefix_tokens_reused, 8);
        assert_eq!(target_item.prefix_blocks_reused, 2);
        assert_eq!(
            target_item.reused_prefix_token_slice,
            vec![10, 11, 12, 13, 14, 15, 16, 17]
        );
        assert_eq!(target_item.scheduled_token_count, 1);
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.execution_plan_ref.as_deref()),
            Some("phase1.qwen3.paged_decode")
        );
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| batch.route_metadata.attention_route.as_deref()),
            Some("qwen3_paged_decode")
        );
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| {
                    batch
                        .route_metadata
                        .crossover_decisions
                        .iter()
                        .find(|(key, _)| key == "branch_decode_requests")
                        .map(|(_, value)| *value)
                }),
            Some(1)
        );
        assert_eq!(
            outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .and_then(|batch| {
                    batch
                        .route_metadata
                        .crossover_decisions
                        .iter()
                        .find(|(key, _)| key == "branch_decode_tokens")
                        .map(|(_, value)| *value)
                }),
            Some(1)
        );
    }

    #[test]
    fn cancel_during_running_resolves_on_next_step() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine.submit(make_submission(1, 1, 4)).unwrap();
        engine.step(4, true).unwrap();

        let snapshot = engine.request_manager().snapshot(RequestId(1)).unwrap();
        assert_eq!(snapshot.state, RequestState::Runnable);

        engine.submit(make_submission(2, 2, 4)).unwrap();

        engine.step(2, true).unwrap();

        engine.cancel(RequestId(2)).unwrap();

        let snapshot = engine.request_manager().snapshot(RequestId(2)).unwrap();
        assert!(snapshot.state.is_terminal());
        assert_eq!(snapshot.state, RequestState::Cancelled);
    }

    #[test]
    fn multiple_requests_progress_across_successive_steps() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 16));

        engine.submit(make_submission(1, 1, 2)).unwrap();
        engine.submit(make_submission(2, 2, 2)).unwrap();

        for _ in 0..20 {
            let s1 = engine.request_manager().snapshot(RequestId(1)).unwrap();
            let s2 = engine.request_manager().snapshot(RequestId(2)).unwrap();
            if s1.state.is_terminal() && s2.state.is_terminal() {
                break;
            }
            engine.step(8, true).unwrap();
        }

        let s1 = engine.request_manager().snapshot(RequestId(1)).unwrap();
        let s2 = engine.request_manager().snapshot(RequestId(2)).unwrap();
        assert_eq!(s1.state, RequestState::Finished);
        assert_eq!(s2.state, RequestState::Finished);
        assert_eq!(s1.generated_len, 2);
        assert_eq!(s2.generated_len, 2);
    }

    #[test]
    fn submit_duplicate_request_returns_error() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine.submit(make_submission(1, 1, 2)).unwrap();
        let error = engine.submit(make_submission(1, 2, 2));

        assert!(error.is_err());
    }

    #[test]
    fn cancel_unknown_request_returns_error() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        let error = engine.cancel(RequestId(99));

        assert!(error.is_err());
    }

    #[test]
    fn step_rejects_runner_output_with_wrong_step_id() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            WrongStepIdRunner,
            DeterministicSampler,
        );

        engine.submit(make_submission(1, 1, 1)).unwrap();

        let error = engine
            .step(4, true)
            .expect_err("wrong step_id should fail closed");

        let EngineCoreError::RunnerContractViolation { message, .. } = error else {
            panic!("expected runner contract violation");
        };
        assert_eq!(
            message,
            "runner output step_id does not match scheduled batch"
        );
    }

    #[test]
    fn step_rejects_non_completing_prefill_update_with_output_token() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            PrefillWithOutputTokenRunner,
            DeterministicSampler,
        );

        engine.submit(make_submission(1, 1, 1)).unwrap();

        let error = engine
            .step(2, true)
            .expect_err("prefill with output_token should fail closed");

        let EngineCoreError::RunnerContractViolation { message, .. } = error else {
            panic!("expected runner contract violation");
        };
        assert_eq!(
            message,
            "prefill updates may only emit output tokens when completing the prompt"
        );
    }

    #[test]
    fn step_accepts_prefill_completion_update_with_output_token() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            PrefillWithOutputTokenRunner,
            DeterministicSampler,
        );

        engine.submit(make_submission(1, 1, 1)).unwrap();

        let outcome = engine
            .step(4, true)
            .expect("prefill-completion output_token should be accepted");
        let snapshot = engine
            .request_manager()
            .snapshot(RequestId(1))
            .expect("request should remain queryable after first sampled token");

        assert_eq!(outcome.sampled_tokens.len(), 1);
        assert_eq!(outcome.sampled_tokens[0].request_id, RequestId(1));
        assert_eq!(outcome.sampled_tokens[0].token_id, 42);
        assert_eq!(snapshot.processed_prompt_tokens, 4);
        assert_eq!(snapshot.generated_tokens, vec![42]);
    }

    #[test]
    fn failed_batch_runner_transitions_all_requests_to_failed() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            FailedBatchRunner,
            DeterministicSampler,
        );

        engine.submit(make_submission(1, 1, 2)).unwrap();
        engine.submit(make_submission(2, 2, 2)).unwrap();
        engine.step(8, true).unwrap();

        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(1))
                .unwrap()
                .state,
            RequestState::Failed
        );
        assert_eq!(
            engine
                .request_manager()
                .snapshot(RequestId(2))
                .unwrap()
                .state,
            RequestState::Failed
        );
    }

    #[test]
    fn fallback_replan_schedules_decode_when_prefill_is_blocked() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 2));

        engine
            .submit(make_submission_with_prompt(1, 1, vec![1, 2, 3, 4], 2))
            .unwrap();
        engine.step(4, true).unwrap();
        engine.step(1, true).unwrap();

        engine
            .submit(make_submission_with_prompt(
                2,
                2,
                vec![5, 6, 7, 8, 9, 10, 11, 12],
                2,
            ))
            .unwrap();

        let outcome = engine.step(8, true).unwrap();

        let scheduled = &outcome.schedule_plan.selected_requests;
        let blocked = &outcome.schedule_plan.memory_blocked_requests;

        assert!(
            scheduled.contains(&RequestId(1)) || blocked.contains(&RequestId(2)),
            "engine should either schedule request 1 (decode) or block request 2 (needs more blocks)"
        );
    }

    #[test]
    fn execution_plan_resolver_produces_model_specific_route_for_gemma() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine
            .submit(RequestSubmission {
                request_id: RequestId(1),
                model_id: ModelId("gemma-4-27b-it".into()),
                input_tokens: vec![1, 2, 3, 4],
                sampling_params: SamplingParams::default(),
                max_output_tokens: 2,
                arrival_sequence: SequenceNo(1),
                metadata: None,
            })
            .unwrap();

        let outcome = engine.step(4, true).unwrap();

        let batch = outcome
            .schedule_plan
            .execution_batch
            .as_ref()
            .expect("batch should exist");
        assert_eq!(
            batch.execution_plan_ref.as_deref(),
            Some("phase1.gemma_4_27b_it.dense_prefill")
        );
        assert_eq!(
            batch.route_metadata.attention_route.as_deref(),
            Some("gemma_4_27b_it_prefill")
        );
    }

    #[test]
    fn route_metadata_propagates_through_full_prefill_decode_cycle() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 8));

        engine.submit(make_submission(1, 1, 1)).unwrap();

        let prefill = engine.step(4, true).unwrap();
        let prefill_batch = prefill
            .schedule_plan
            .execution_batch
            .as_ref()
            .expect("prefill batch");
        assert_eq!(
            prefill_batch.execution_plan_ref.as_deref(),
            Some("phase1.qwen3.dense_prefill")
        );
        assert_eq!(
            prefill_batch.route_metadata.kv_mode.as_deref(),
            Some("paged_metadata")
        );
        assert_eq!(
            prefill_batch.route_metadata.barrier_mode.as_deref(),
            Some("serial")
        );

        let runner_route = prefill
            .runner_output
            .as_ref()
            .expect("runner output should exist");
        assert_eq!(
            runner_route.route_metadata.execution_plan.as_deref(),
            Some("phase1.qwen3.dense_prefill")
        );

        let decode = engine.step(1, true).unwrap();
        let decode_batch = decode
            .schedule_plan
            .execution_batch
            .as_ref()
            .expect("decode batch");
        assert_eq!(
            decode_batch.execution_plan_ref.as_deref(),
            Some("phase1.qwen3.paged_decode")
        );
        assert_eq!(
            decode_batch.route_metadata.attention_route.as_deref(),
            Some("qwen3_paged_decode")
        );
    }

    #[test]
    fn decode_batches_carry_processed_context_for_native_prefix_warmup() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            DeterministicRunner,
            FixedTokenSampler {
                token_id: 42,
                stop_reason: None,
            },
        );

        engine.submit(make_submission(1, 1, 2)).unwrap();
        engine.step(4, true).unwrap();

        let decode1 = engine.step(1, true).unwrap();
        let decode1_item = decode1
            .schedule_plan
            .execution_batch
            .as_ref()
            .and_then(|batch| {
                batch
                    .items
                    .iter()
                    .find(|item| item.request_id == RequestId(1))
            })
            .expect("first decode item should be scheduled");
        assert_eq!(decode1_item.mode, ExecutionMode::Decode);
        assert_eq!(decode1_item.reused_prefix_token_slice, vec![1, 2, 3, 4]);

        let decode2 = engine.step(1, true).unwrap();
        let decode2_item = decode2
            .schedule_plan
            .execution_batch
            .as_ref()
            .and_then(|batch| {
                batch
                    .items
                    .iter()
                    .find(|item| item.request_id == RequestId(1))
            })
            .expect("second decode item should be scheduled");
        assert_eq!(decode2_item.mode, ExecutionMode::Decode);
        assert_eq!(decode2_item.input_token_slice, vec![42]);
        assert_eq!(decode2_item.reused_prefix_token_slice, vec![1, 2, 3, 4, 42]);
    }

    #[test]
    fn different_model_requests_are_batched_separately() {
        let mut engine = EngineCore::with_kv_config(KvManagerConfig::new(CacheGroupId(2), 4, 16));

        engine
            .submit(RequestSubmission {
                request_id: RequestId(1),
                model_id: ModelId("qwen3".into()),
                input_tokens: vec![1, 2, 3, 4],
                sampling_params: SamplingParams::default(),
                max_output_tokens: 1,
                arrival_sequence: SequenceNo(1),
                metadata: None,
            })
            .unwrap();
        engine
            .submit(RequestSubmission {
                request_id: RequestId(2),
                model_id: ModelId("gemma".into()),
                input_tokens: vec![5, 6, 7, 8],
                sampling_params: SamplingParams::default(),
                max_output_tokens: 1,
                arrival_sequence: SequenceNo(2),
                metadata: None,
            })
            .unwrap();

        let step1 = engine.step(8, true).unwrap();
        assert_eq!(step1.schedule_plan.selected_requests, vec![RequestId(1)]);
        assert_eq!(step1.schedule_plan.deferred_requests, vec![RequestId(2)]);
        assert_eq!(
            step1
                .schedule_plan
                .execution_batch
                .as_ref()
                .unwrap()
                .model_id,
            "qwen3"
        );

        // step2: qwen3 request 1 does decode (still earliest arrival), gemma deferred again
        let step2 = engine.step(8, true).unwrap();
        assert_eq!(step2.schedule_plan.selected_requests, vec![RequestId(1)]);

        // step3: qwen3 request 1 finished and cleaned up, now gemma request 2 gets scheduled
        let step3 = engine.step(8, true).unwrap();
        assert!(
            step3
                .schedule_plan
                .selected_requests
                .contains(&RequestId(2))
        );
        assert_eq!(
            step3
                .schedule_plan
                .execution_batch
                .as_ref()
                .unwrap()
                .model_id,
            "gemma"
        );
    }

    #[test]
    fn eos_token_finishes_request_before_max_output() {
        let mut engine = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(2), 4, 8),
            DeterministicRunner,
            FixedTokenSampler {
                token_id: 0,
                stop_reason: Some(StopReason::EosToken),
            },
        );

        engine.submit(make_submission(1, 1, 100)).unwrap();
        engine.step(4, true).unwrap();
        engine.step(1, true).unwrap();

        let snapshot = engine.request_manager().snapshot(RequestId(1)).unwrap();
        assert_eq!(snapshot.state, RequestState::Finished);
        assert_eq!(snapshot.generated_len, 1);
        assert_eq!(snapshot.terminal_stop_reason, Some(StopReason::EosToken));
    }
}
