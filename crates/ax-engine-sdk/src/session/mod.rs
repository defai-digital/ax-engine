use std::collections::{BTreeMap, VecDeque};

use ax_engine_core::{
    EmbeddingPooling, EngineCore, EngineStepOutcome, ModelId, RequestId, RequestSubmission,
    SequenceNo,
};

use crate::backend::{ResolvedBackend, RuntimeReport, SelectedBackend};
use crate::generate::{
    GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateStreamEvent,
    GenerateStreamRequestEvent, GenerateStreamResponseEvent, GenerateStreamStepEvent,
};
use crate::llama_cpp::LlamaCppStreamHandle;
use crate::mlx_lm::start_streaming_generate as start_mlx_lm_streaming_generate;
use crate::request::{
    EngineStepReport, MetalDispatchStepReport, SessionRequestReport, SessionRequestState,
};

mod artifacts;
mod config;
mod delegated;
mod errors;
mod llama_lifecycle;
mod native;
mod routes;
mod stream;

use artifacts::resolve_native_model_report;
pub use config::{
    EngineSessionConfig, PreviewSessionConfigError, PreviewSessionConfigRequest,
    ResolvedSessionConfigRequest,
};
use delegated::{
    run_delegated_generate_prevalidated, run_delegated_generate_with_config,
    start_llama_cpp_stream_prevalidated,
};
pub use errors::EngineSessionError;
use llama_lifecycle::{LlamaCppLifecycleRequest, LlamaCppLifecycleRequestSlot};
use native::build_native_core;
#[cfg(feature = "mlx-native")]
use native::build_native_core_with_mlx_prefix_cache;
use routes::{apply_native_step_route_to_report, llama_cpp_stream_route, merge_native_route_into};
pub use stream::{GenerateStream, GenerateStreamState};
use stream::{
    GenerateStreamPhase, LlamaCppGenerateStreamState, NativeGenerateStreamState,
    build_mlx_lm_stream_state, is_terminal_request_state, next_llama_cpp_stream_event,
    next_mlx_lm_stream_event, slice_output_token_logprobs,
};

const LLAMA_CPP_STREAM_EXECUTION_PLAN: &str = "llama_cpp.server_completion_stream";
const MLX_LM_STREAM_EXECUTION_PLAN: &str = "mlx_lm_delegated.server_completion_stream";
const MAX_LLAMA_CPP_TERMINAL_REQUESTS: usize = 1024;
const MAX_NATIVE_ROUTE_REPORTS: usize = 1024;
// Native streaming advances through prompt prefill, decode, and occasional scheduler
// bookkeeping steps; keep the guard explicit so it is not an unexplained literal.
const NATIVE_STREAM_STEP_GUARD_BUFFER: u64 = 256;

/// Stateless generation helper for delegated backends.
///
/// For native MLX, blocking generation still constructs a full `EngineSession`
/// per call so the model and KV state are not reused. Use `EngineSession`
/// directly for repeated MLX requests.
#[derive(Clone, Debug)]
pub struct StatelessGenerateContext {
    config: EngineSessionConfig,
    delegated_runtime: Option<RuntimeReport>,
}

impl StatelessGenerateContext {
    pub fn new(config: EngineSessionConfig) -> Result<Self, EngineSessionError> {
        let delegated_runtime = if config.resolved_backend.selected_backend.is_mlx() {
            None
        } else {
            config.validate()?;
            Some(config.runtime_report())
        };

        Ok(Self {
            config,
            delegated_runtime,
        })
    }

    pub fn config(&self) -> &EngineSessionConfig {
        &self.config
    }

    pub fn supports_stateless_streaming(&self) -> bool {
        matches!(
            self.config.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp | SelectedBackend::MlxLmDelegated
        )
    }

    pub fn generate_with_request_id(
        &self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, EngineSessionError> {
        if self.config.resolved_backend.selected_backend.is_mlx() {
            let mut session = EngineSession::new(self.config.clone())?;
            return session.generate_with_request_id(request_id, request);
        }

        EngineSession::validate_generate_request(request_id, &request)?;
        let runtime =
            self.delegated_runtime
                .as_ref()
                .ok_or(EngineSessionError::MissingDelegatedRuntime {
                    selected_backend: self.config.resolved_backend.selected_backend,
                })?;
        run_delegated_generate_prevalidated(&self.config, runtime, request_id, &request)
    }

    pub fn stream_state_with_request_id(
        &self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateStreamState, EngineSessionError> {
        if self.config.resolved_backend.selected_backend.is_mlx() {
            return Err(
                EngineSessionError::NativeBackendStatelessStreamNotSupported {
                    selected_backend: self.config.resolved_backend.selected_backend,
                },
            );
        }

        EngineSession::validate_generate_request(request_id, &request)?;
        let runtime =
            self.delegated_runtime
                .as_ref()
                .ok_or(EngineSessionError::MissingDelegatedRuntime {
                    selected_backend: self.config.resolved_backend.selected_backend,
                })?;

        match self.config.resolved_backend.selected_backend {
            SelectedBackend::LlamaCpp => {
                let (runtime, stream, _route_backend) = start_llama_cpp_stream_prevalidated(
                    &self.config,
                    runtime,
                    request_id,
                    &request,
                )?;
                Ok(build_llama_cpp_stream_state(
                    request_id, request, runtime, stream,
                ))
            }
            SelectedBackend::MlxLmDelegated => {
                let mlx_lm_backend = self
                    .config
                    .mlx_lm_backend
                    .as_ref()
                    .ok_or(EngineSessionError::MissingMlxLmConfig)?;
                let stream = start_mlx_lm_streaming_generate(runtime, mlx_lm_backend, &request)
                    .map_err(EngineSessionError::from)?;
                Ok(build_mlx_lm_stream_state(
                    request_id,
                    request,
                    runtime.clone(),
                    stream,
                ))
            }
            SelectedBackend::Mlx => unreachable!("is_mlx() was already checked"),
        }
    }

    pub fn next_stream_event(
        &self,
        state: &mut GenerateStreamState,
    ) -> Result<Option<GenerateStreamEvent>, EngineSessionError> {
        match state {
            GenerateStreamState::LlamaCpp(state) => next_llama_cpp_stream_event(
                state.as_mut(),
                self.config.resolved_backend.selected_backend,
            ),
            GenerateStreamState::MlxLm(state) => next_mlx_lm_stream_event(state.as_mut()),
            GenerateStreamState::Native(_) => Err(
                EngineSessionError::NativeBackendStatelessStreamNotSupported {
                    selected_backend: self.config.resolved_backend.selected_backend,
                },
            ),
        }
    }
}

#[derive(Debug)]
pub struct EngineSession {
    core: EngineCore,
    config: EngineSessionConfig,
    runtime: RuntimeReport,
    next_request_id: u64,
    native_request_routes: BTreeMap<u64, GenerateRouteReport>,
    native_route_report_order: VecDeque<u64>,
    llama_requests: BTreeMap<u64, LlamaCppLifecycleRequestSlot>,
    llama_terminal_request_order: VecDeque<u64>,
}

impl EngineSession {
    fn uses_mlx_runtime(&self) -> bool {
        self.config.resolved_backend.selected_backend.is_mlx()
    }

    fn llama_lifecycle_unsupported_error(&self, operation: &'static str) -> EngineSessionError {
        EngineSessionError::LlamaCppDoesNotSupportLifecycle {
            selected_backend: self.config.resolved_backend.selected_backend,
            operation,
        }
    }

    fn validate_generate_request(
        request_id: u64,
        request: &GenerateRequest,
    ) -> Result<(), EngineSessionError> {
        if request_id == 0 {
            return Err(EngineSessionError::InvalidRequestId);
        }
        if request.max_output_tokens == 0 {
            return Err(EngineSessionError::InvalidMaxOutputTokens);
        }
        let has_input_text = request
            .input_text
            .as_ref()
            .is_some_and(|input_text| !input_text.is_empty());
        if request.input_tokens.is_empty() && !has_input_text {
            return Err(EngineSessionError::EmptyInputTokens);
        }

        Ok(())
    }

    fn advance_request_id(&mut self, request_id: u64) {
        self.next_request_id = self.next_request_id.max(request_id.saturating_add(1));
    }

    fn llama_active_request_ids(&self) -> Vec<u64> {
        self.llama_requests
            .iter()
            .filter_map(|(request_id, slot)| match slot {
                LlamaCppLifecycleRequestSlot::Active(_) => Some(*request_id),
                LlamaCppLifecycleRequestSlot::Terminal(_) => None,
            })
            .collect()
    }

    fn store_terminal_llama_report(&mut self, request_id: u64, report: SessionRequestReport) {
        let already_terminal = matches!(
            self.llama_requests.get(&request_id),
            Some(LlamaCppLifecycleRequestSlot::Terminal(_))
        );
        self.llama_requests.insert(
            request_id,
            LlamaCppLifecycleRequestSlot::Terminal(Box::new(report)),
        );
        if !already_terminal {
            self.llama_terminal_request_order.push_back(request_id);
        }
        self.prune_terminal_llama_requests();
    }

    fn prune_terminal_llama_requests(&mut self) {
        while self.llama_terminal_request_order.len() > MAX_LLAMA_CPP_TERMINAL_REQUESTS {
            let Some(evicted_request_id) = self.llama_terminal_request_order.pop_front() else {
                break;
            };
            if matches!(
                self.llama_requests.get(&evicted_request_id),
                Some(LlamaCppLifecycleRequestSlot::Terminal(_))
            ) {
                self.llama_requests.remove(&evicted_request_id);
            }
        }
    }

    fn store_native_request_route(&mut self, request_id: u64, route: GenerateRouteReport) {
        if let Some(existing) = self.native_request_routes.get_mut(&request_id) {
            merge_native_route_into(existing, route);
        } else {
            self.native_route_report_order.push_back(request_id);
            self.native_request_routes.insert(request_id, route);
        }

        while self.native_route_report_order.len() > MAX_NATIVE_ROUTE_REPORTS {
            let Some(evicted_request_id) = self.native_route_report_order.pop_front() else {
                break;
            };
            self.native_request_routes.remove(&evicted_request_id);
        }
    }

    fn llama_cpp_submit_generate_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<u64, EngineSessionError> {
        Self::validate_generate_request(request_id, &request)?;
        self.advance_request_id(request_id);
        let (_runtime, stream, _route_backend) =
            self.llama_cpp_stream_start(request_id, &request)?;
        let route = llama_cpp_stream_route();
        let current_report = SessionRequestReport {
            request_id,
            model_id: request.model_id,
            state: SessionRequestState::Waiting,
            prompt_tokens: request.input_tokens,
            processed_prompt_tokens: 0,
            output_tokens: Vec::new(),
            output_token_logprobs: Vec::new(),
            prompt_len: 0,
            output_len: 0,
            max_output_tokens: request.max_output_tokens,
            cancel_requested: false,
            execution_plan_ref: route.execution_plan.clone(),
            route,
            finish_reason: None,
            terminal_stop_reason: None,
            last_error: None,
        };

        self.llama_requests.insert(
            request_id,
            LlamaCppLifecycleRequestSlot::Active(Box::new(LlamaCppLifecycleRequest::new(
                request_id,
                current_report,
                stream,
            ))),
        );
        Ok(request_id)
    }

    fn llama_cpp_stream_state_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateStreamState, EngineSessionError> {
        Self::validate_generate_request(request_id, &request)?;
        self.advance_request_id(request_id);

        let (runtime, stream, _route_backend) =
            self.llama_cpp_stream_start(request_id, &request)?;
        Ok(build_llama_cpp_stream_state(
            request_id, request, runtime, stream,
        ))
    }

    fn llama_cpp_stream_start(
        &self,
        request_id: u64,
        request: &GenerateRequest,
    ) -> Result<(RuntimeReport, LlamaCppStreamHandle, SelectedBackend), EngineSessionError> {
        let runtime = self.runtime_report();
        start_llama_cpp_stream_prevalidated(&self.config, &runtime, request_id, request)
    }

    pub fn new(config: EngineSessionConfig) -> Result<Self, EngineSessionError> {
        config.validate()?;
        let core = build_native_core(&config)?;
        Self::from_validated_config_and_core(config, core)
    }

    #[cfg(feature = "mlx-native")]
    pub fn new_with_shared_mlx_prefix_cache(
        config: EngineSessionConfig,
        prefix_cache_store: ax_engine_mlx::MlxPrefixCacheStore,
    ) -> Result<Self, EngineSessionError> {
        config.validate()?;
        let core = build_native_core_with_mlx_prefix_cache(&config, Some(prefix_cache_store))?;
        Self::from_validated_config_and_core(config, core)
    }

    fn from_validated_config_and_core(
        config: EngineSessionConfig,
        core: EngineCore,
    ) -> Result<Self, EngineSessionError> {
        let runtime = config
            .runtime_report()
            .with_mlx_model(resolve_native_model_report(&config, &core));
        Ok(Self {
            core,
            config,
            runtime,
            next_request_id: 1,
            native_request_routes: BTreeMap::new(),
            native_route_report_order: VecDeque::new(),
            llama_requests: BTreeMap::new(),
            llama_terminal_request_order: VecDeque::new(),
        })
    }

    pub fn generate_stateless_with_request_id(
        config: EngineSessionConfig,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, EngineSessionError> {
        if config.resolved_backend.selected_backend.is_mlx() {
            let mut session = Self::new(config)?;
            return session.generate_with_request_id(request_id, request);
        }

        Self::generate_stateless_with_config(&config, request_id, request)
    }

    pub fn generate_stateless_with_config(
        config: &EngineSessionConfig,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, EngineSessionError> {
        if config.resolved_backend.selected_backend.is_mlx() {
            let mut session = Self::new(config.clone())?;
            return session.generate_with_request_id(request_id, request);
        }

        Self::validate_generate_request(request_id, &request)?;
        config.validate()?;
        run_delegated_generate_with_config(config, request_id, &request)
    }

    pub fn config(&self) -> &EngineSessionConfig {
        &self.config
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        &self.config.resolved_backend
    }

    pub fn runtime_report(&self) -> RuntimeReport {
        self.runtime.clone()
    }

    pub fn core(&self) -> &EngineCore {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut EngineCore {
        &mut self.core
    }

    pub fn submit(
        &mut self,
        submission: RequestSubmission,
    ) -> Result<RequestId, EngineSessionError> {
        self.core
            .submit(submission)
            .map_err(EngineSessionError::from)
    }

    pub fn cancel(&mut self, request_id: RequestId) -> Result<(), EngineSessionError> {
        self.core
            .cancel(request_id)
            .map_err(EngineSessionError::from)
    }

    pub fn cancel_request(&mut self, request_id: u64) -> Result<(), EngineSessionError> {
        if !self.uses_mlx_runtime() {
            let terminal_report = {
                let Some(slot) = self.llama_requests.get_mut(&request_id) else {
                    return Err(EngineSessionError::MissingRequestSnapshot { request_id });
                };
                match slot {
                    LlamaCppLifecycleRequestSlot::Active(request) => Some(request.cancel()),
                    LlamaCppLifecycleRequestSlot::Terminal(_) => None,
                }
            };
            if let Some(report) = terminal_report {
                self.store_terminal_llama_report(request_id, report);
            }
            return Ok(());
        }
        self.cancel(RequestId(request_id))
    }

    pub fn step(&mut self) -> Result<EngineStepOutcome, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return Err(self.llama_lifecycle_unsupported_error("step"));
        }
        self.core
            .step(self.config.max_batch_tokens, self.config.deterministic)
            .map_err(EngineSessionError::from)
    }

    pub fn step_report(&mut self) -> Result<EngineStepReport, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            let active_request_ids = self.llama_active_request_ids();
            if active_request_ids.is_empty() {
                return Ok(EngineStepReport::default());
            }
            let selected_backend = self.config.resolved_backend.selected_backend;
            let mut aggregate = EngineStepReport::default();
            let mut terminal_reports = Vec::new();

            for request_id in active_request_ids {
                let step = {
                    let slot = self
                        .llama_requests
                        .get_mut(&request_id)
                        .ok_or(EngineSessionError::MissingRequestSnapshot { request_id })?;
                    let LlamaCppLifecycleRequestSlot::Active(request) = slot else {
                        continue;
                    };
                    let step = request.step_report(selected_backend)?;
                    if is_terminal_request_state(request.current_report.state) {
                        request.drain_trailing_usage();
                        terminal_reports.push((request_id, request.current_report.clone()));
                    }
                    step
                };
                aggregate.accumulate(step);
            }

            for (request_id, report) in terminal_reports {
                self.store_terminal_llama_report(request_id, report);
            }

            return Ok(aggregate);
        }

        let outcome = self.step()?;
        let metal_dispatch = outcome
            .runner_output
            .as_ref()
            .and_then(|_| self.core.last_metal_dispatch())
            .map(|trace| MetalDispatchStepReport::from_trace(&trace));
        let report = EngineStepReport::from_native_outcome(&outcome, metal_dispatch);
        if let Some(route) = report.route.as_ref() {
            let request_ids = outcome
                .schedule_plan
                .execution_batch
                .as_ref()
                .map(|batch| {
                    batch
                        .items
                        .iter()
                        .map(|item| item.request_id.0)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            for request_id in request_ids {
                self.store_native_request_route(request_id, route.clone());
            }
        }
        Ok(report)
    }

    pub fn request_report(&self, request_id: u64) -> Option<SessionRequestReport> {
        if !self.uses_mlx_runtime() {
            return self
                .llama_requests
                .get(&request_id)
                .map(LlamaCppLifecycleRequestSlot::report);
        }
        let mut report: SessionRequestReport = self
            .core
            .request_manager()
            .snapshot(RequestId(request_id))
            .map(Into::into)?;
        if let Some(route) = self.native_request_routes.get(&request_id) {
            report.route = route.clone();
        }
        Some(report)
    }

    pub fn stream_request(
        &mut self,
        request_id: u64,
    ) -> Result<GenerateStream<'_>, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return Err(self.llama_lifecycle_unsupported_error("stream_request"));
        }
        Ok(GenerateStream::new(self, self.stream_state(request_id)?))
    }

    pub fn submit_generate(&mut self, request: GenerateRequest) -> Result<u64, EngineSessionError> {
        let request_id = self.next_request_id;
        self.submit_generate_with_request_id(request_id, request)
    }

    pub fn submit_generate_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<u64, EngineSessionError> {
        Self::validate_generate_request(request_id, &request)?;
        if !self.uses_mlx_runtime() {
            return match self.config.resolved_backend.selected_backend {
                SelectedBackend::LlamaCpp => {
                    self.llama_cpp_submit_generate_with_request_id(request_id, request)
                }
                SelectedBackend::MlxLmDelegated => {
                    Err(EngineSessionError::MlxLmDoesNotSupportLifecycle {
                        operation: "submit_generate",
                    })
                }
                SelectedBackend::Mlx => unreachable!("uses_mlx_runtime was already checked"),
            };
        }
        if request.input_text.is_some() {
            return Err(EngineSessionError::MlxBackendRequiresTokenizedInput);
        }

        let request_id = RequestId(request_id);
        self.advance_request_id(request_id.0);

        let submission = RequestSubmission {
            request_id,
            model_id: ModelId(request.model_id),
            input_tokens: request.input_tokens,
            sampling_params: request.sampling.into_core(self.config.deterministic),
            max_output_tokens: request.max_output_tokens,
            arrival_sequence: SequenceNo(request_id.0),
            metadata: request.metadata,
        };

        self.submit(submission)?;
        Ok(request_id.0)
    }

    pub fn stream_generate(
        &mut self,
        request: GenerateRequest,
    ) -> Result<GenerateStream<'_>, EngineSessionError> {
        self.stream_generate_with_request_id(self.next_request_id, request)
    }

    pub fn stream_generate_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateStream<'_>, EngineSessionError> {
        let state = self.stream_generate_state_with_request_id(request_id, request)?;
        Ok(GenerateStream::new(self, state))
    }

    pub fn run_to_completion(
        &mut self,
        request_id: u64,
    ) -> Result<GenerateResponse, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return Err(self.llama_lifecycle_unsupported_error("run_to_completion"));
        }
        self.stream_request(request_id)?.into_response()
    }

    pub fn generate(
        &mut self,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, EngineSessionError> {
        self.generate_with_request_id(self.next_request_id, request)
    }

    pub fn generate_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return run_delegated_generate_with_config(&self.config, request_id, &request);
        }
        let request_id = self.submit_generate_with_request_id(request_id, request)?;
        self.run_to_completion(request_id)
    }

    /// Compute a dense embedding for `token_ids` using the active MLX model.
    ///
    /// When `normalize` is `true` the returned vector is L2-normalized to unit
    /// length, which is required for cosine / dot-product similarity and is the
    /// standard expectation of all major embedding models.
    ///
    /// Only supported when the session is using an MLX-native backend; returns
    /// `EngineSessionError::EmbeddingNotSupported` otherwise.
    pub fn embed(
        &self,
        token_ids: &[u32],
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<Vec<f32>, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return Err(EngineSessionError::EmbeddingNotSupported);
        }
        self.core
            .embed(token_ids, pooling, normalize)
            .map_err(|message| EngineSessionError::EmbeddingFailed { message })
    }

    pub fn embed_batch(
        &self,
        batch: &[Vec<u32>],
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return Err(EngineSessionError::EmbeddingNotSupported);
        }
        self.core
            .embed_batch(batch, pooling, normalize)
            .map_err(|message| EngineSessionError::EmbeddingFailed { message })
    }

    /// Batched embedding returning one contiguous row-major
    /// `[batch_size, hidden_size]` buffer instead of `Vec<Vec<f32>>`.
    /// Saves `B - 1` heap allocations per call and lets downstream code
    /// (numpy, faiss, HNSW indices) treat the result as a zero-copy view
    /// over a single `&[f32]`.
    pub fn embed_batch_flat(
        &self,
        batch: &[Vec<u32>],
        pooling: EmbeddingPooling,
        normalize: bool,
    ) -> Result<ax_engine_core::EmbeddingMatrix, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return Err(EngineSessionError::EmbeddingNotSupported);
        }
        self.core
            .embed_batch_flat(batch, pooling, normalize)
            .map_err(|message| EngineSessionError::EmbeddingFailed { message })
    }

    pub fn stream_state(&self, request_id: u64) -> Result<GenerateStreamState, EngineSessionError> {
        let current_report = self
            .request_report(request_id)
            .ok_or(EngineSessionError::MissingRequestSnapshot { request_id })?;
        let runtime = self.runtime_report();

        Ok(GenerateStreamState::new_native(
            request_id,
            runtime,
            current_report,
        ))
    }

    pub fn stream_generate_state(
        &mut self,
        request: GenerateRequest,
    ) -> Result<GenerateStreamState, EngineSessionError> {
        self.stream_generate_state_with_request_id(self.next_request_id, request)
    }

    pub fn stream_generate_state_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateStreamState, EngineSessionError> {
        if !self.uses_mlx_runtime() {
            return match self.config.resolved_backend.selected_backend {
                SelectedBackend::LlamaCpp => {
                    self.llama_cpp_stream_state_with_request_id(request_id, request)
                }
                SelectedBackend::MlxLmDelegated => {
                    let mlx_lm_backend = self
                        .config
                        .mlx_lm_backend
                        .as_ref()
                        .ok_or(EngineSessionError::MissingMlxLmConfig)?;
                    let runtime = self.config.runtime_report();
                    let stream =
                        start_mlx_lm_streaming_generate(&runtime, mlx_lm_backend, &request)
                            .map_err(EngineSessionError::from)?;
                    Ok(build_mlx_lm_stream_state(
                        request_id, request, runtime, stream,
                    ))
                }
                SelectedBackend::Mlx => unreachable!("uses_mlx_runtime was already checked"),
            };
        }

        let request_id = self.submit_generate_with_request_id(request_id, request)?;
        self.stream_state(request_id)
    }

    pub fn next_stream_event(
        &mut self,
        state: &mut GenerateStreamState,
    ) -> Result<Option<GenerateStreamEvent>, EngineSessionError> {
        match state {
            GenerateStreamState::Native(state) => self.next_native_stream_event(state.as_mut()),
            GenerateStreamState::LlamaCpp(state) => next_llama_cpp_stream_event(
                state.as_mut(),
                self.config.resolved_backend.selected_backend,
            ),
            GenerateStreamState::MlxLm(state) => next_mlx_lm_stream_event(state.as_mut()),
        }
    }

    fn next_native_stream_event(
        &mut self,
        state: &mut NativeGenerateStreamState,
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
                if is_terminal_request_state(state.current_report.state) {
                    state.phase = GenerateStreamPhase::Done;
                    return Ok(Some(GenerateStreamEvent::Response(
                        GenerateStreamResponseEvent {
                            response: GenerateResponse::from_report(
                                state.current_report.clone(),
                                state.step_count,
                                state.ttft_step,
                                state.runtime.clone(),
                            ),
                        },
                    )));
                }

                let step = self.step_report()?;
                state.step_count += 1;

                if state.ttft_step.is_none() && step.ttft_events > 0 {
                    state.ttft_step = Some(state.step_count);
                }

                if state.step_count >= state.max_steps {
                    return Err(EngineSessionError::RequestDidNotTerminate {
                        request_id: state.request_id,
                        max_steps: state.max_steps,
                    });
                }

                let mut next_report = self.request_report(state.request_id).ok_or(
                    EngineSessionError::MissingRequestSnapshot {
                        request_id: state.request_id,
                    },
                )?;
                apply_native_step_route_to_report(&mut next_report, &step);
                if state.emitted_output_len > next_report.output_tokens.len() {
                    return Err(EngineSessionError::RequestReportInvariantViolation {
                        request_id: state.request_id,
                        message: "output tokens shrunk between stream snapshots",
                    });
                }
                let delta_tokens = next_report.output_tokens[state.emitted_output_len..].to_vec();
                let delta_token_logprobs = slice_output_token_logprobs(
                    &next_report,
                    state.emitted_output_len,
                    delta_tokens.len(),
                )?;
                state.emitted_output_len = next_report.output_tokens.len();
                state.current_report = next_report.clone();

                Ok(Some(GenerateStreamEvent::Step(GenerateStreamStepEvent {
                    request: next_report,
                    step,
                    delta_tokens,
                    delta_token_logprobs,
                    delta_text: None,
                })))
            }
            GenerateStreamPhase::Done => Ok(None),
        }
    }
}

fn build_llama_cpp_stream_state(
    request_id: u64,
    request: GenerateRequest,
    runtime: RuntimeReport,
    stream: LlamaCppStreamHandle,
) -> GenerateStreamState {
    let route = llama_cpp_stream_route();
    let current_report = initial_stream_request_report(
        request_id,
        request.model_id,
        request.input_tokens,
        request.max_output_tokens,
        route,
    );

    GenerateStreamState::LlamaCpp(Box::new(LlamaCppGenerateStreamState::new(
        request_id,
        runtime,
        current_report,
        request.input_text,
        stream,
    )))
}

fn initial_stream_request_report(
    request_id: u64,
    model_id: String,
    input_tokens: Vec<u32>,
    max_output_tokens: u32,
    route: GenerateRouteReport,
) -> SessionRequestReport {
    SessionRequestReport {
        request_id,
        model_id,
        state: SessionRequestState::Waiting,
        prompt_tokens: input_tokens,
        processed_prompt_tokens: 0,
        output_tokens: Vec::new(),
        output_token_logprobs: Vec::new(),
        prompt_len: 0,
        output_len: 0,
        max_output_tokens,
        cancel_requested: false,
        execution_plan_ref: route.execution_plan.clone(),
        route,
        finish_reason: None,
        terminal_stop_reason: None,
        last_error: None,
    }
}

#[cfg(test)]
mod tests;
