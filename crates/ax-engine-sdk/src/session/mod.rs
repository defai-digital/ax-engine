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
// GenerateRouteReport::default used by compact stream progress reports.

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
    EngineSessionConfig, MlxMtpPolicy, PreviewSessionConfigError, PreviewSessionConfigRequest,
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
use native::{build_native_core_with_mlx_shares, load_native_whisper_model};
use routes::{
    apply_native_step_route_to_report, llama_cpp_stream_route, merge_native_route_into,
    native_step_needs_route_capture, route_has_decode_path_work,
};
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

/// Text and detected/resolved language returned by a native speech model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeechTranscription {
    pub text: String,
    pub language: Option<String>,
}

/// Stateless generation helper for delegated backends.
///
/// For native MLX, blocking generation still constructs a full
/// `EngineSession` per call so model and KV state stay request-local, while
/// exact prefix snapshots are shared through this context.
#[derive(Clone, Debug)]
pub struct StatelessGenerateContext {
    config: EngineSessionConfig,
    delegated_runtime: Option<RuntimeReport>,
    #[cfg(feature = "mlx-native")]
    native_mlx_prefix_cache: Option<ax_engine_mlx::MlxPrefixCacheStore>,
    /// Cross-request shared weights cell (Option A of the session/weight-reuse
    /// design), populated by the first per-request session build. `Some` only
    /// when `AX_ENGINE_SHARED_WEIGHTS` is truthy; clones share the same cell,
    /// and the weights drop with this context when a model hot-swap replaces
    /// it.
    #[cfg(feature = "mlx-native")]
    native_mlx_shared_weights: Option<ax_engine_mlx::MlxSharedWeightsCell>,
}

/// `AX_ENGINE_SHARED_WEIGHTS` opt-in: share loaded model weights across the
/// per-request native sessions built from one `StatelessGenerateContext`.
/// Default OFF (fully request-local weights). Opt-in semantics: empty, `0`,
/// and `false` all leave it disabled.
#[cfg(feature = "mlx-native")]
fn shared_weights_enabled_from_env() -> bool {
    std::env::var("AX_ENGINE_SHARED_WEIGHTS").is_ok_and(|raw| {
        let trimmed = raw.trim();
        !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
    })
}

impl StatelessGenerateContext {
    pub fn new(config: EngineSessionConfig) -> Result<Self, EngineSessionError> {
        let delegated_runtime = if config.resolved_backend.selected_backend.is_mlx() {
            None
        } else {
            config.validate()?;
            Some(config.runtime_report())
        };

        #[cfg(feature = "mlx-native")]
        let native_mlx_prefix_cache = config
            .resolved_backend
            .selected_backend
            .is_mlx()
            .then(ax_engine_mlx::MlxPrefixCacheStore::from_env);

        #[cfg(feature = "mlx-native")]
        let native_mlx_shared_weights = (config.resolved_backend.selected_backend.is_mlx()
            && shared_weights_enabled_from_env())
        .then(ax_engine_mlx::MlxSharedWeightsCell::new);

        Ok(Self {
            config,
            delegated_runtime,
            #[cfg(feature = "mlx-native")]
            native_mlx_prefix_cache,
            #[cfg(feature = "mlx-native")]
            native_mlx_shared_weights,
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
            let mut session = self.build_stateful_session()?;
            return session.generate_with_request_id(request_id, request);
        }

        EngineSession::validate_generate_request_for_backend(
            self.config.resolved_backend.selected_backend,
            self.config.max_batch_tokens,
            request_id,
            &request,
        )?;
        let runtime =
            self.delegated_runtime
                .as_ref()
                .ok_or(EngineSessionError::MissingDelegatedRuntime {
                    selected_backend: self.config.resolved_backend.selected_backend,
                })?;
        run_delegated_generate_prevalidated(&self.config, runtime, request_id, &request)
    }

    /// Build a full `EngineSession` for routes that cannot be served by a
    /// delegated stateless context. Native MLX sessions reuse this context's
    /// prefix-cache store; request KV state remains private to the new session.
    pub fn build_stateful_session(&self) -> Result<EngineSession, EngineSessionError> {
        if self.config.resolved_backend.selected_backend.is_mlx() {
            #[cfg(feature = "mlx-native")]
            if self.native_mlx_prefix_cache.is_some() || self.native_mlx_shared_weights.is_some() {
                return EngineSession::new_with_shared_mlx_runtime(
                    self.config.clone(),
                    self.native_mlx_prefix_cache.clone(),
                    self.native_mlx_shared_weights.as_ref(),
                );
            }
        }

        EngineSession::new(self.config.clone())
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

        EngineSession::validate_generate_request_for_backend(
            self.config.resolved_backend.selected_backend,
            self.config.max_batch_tokens,
            request_id,
            &request,
        )?;
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
    #[cfg(feature = "mlx-native")]
    whisper: Option<ax_engine_mlx::WhisperModel>,
    next_request_id: u64,
    native_request_routes: BTreeMap<u64, GenerateRouteReport>,
    native_route_report_order: VecDeque<u64>,
    llama_requests: BTreeMap<u64, LlamaCppLifecycleRequestSlot>,
    llama_terminal_request_order: VecDeque<u64>,
}

impl EngineSession {
    /// Clear process-global native compiled closures after all work for the
    /// current model has drained and before constructing a replacement model.
    pub fn clear_native_model_compile_caches() {
        #[cfg(feature = "mlx-native")]
        ax_engine_mlx::clear_process_caches();
    }

    /// Flag whether a sibling model's work is active in this process so
    /// multi-token prefill can use ring-rotated sliding KV (S1 dual-model
    /// contract win) while exclusive single-model sessions keep ordered
    /// prefill (see `AX_MLX_ROTATING_SLIDING_PREFILL` in ax-engine-mlx
    /// fastpath docs for the trade).
    pub fn set_native_sibling_prefill_rotation(enabled: bool) {
        #[cfg(feature = "mlx-native")]
        ax_engine_mlx::fastpath::set_sibling_prefill_rotation(enabled);
        #[cfg(not(feature = "mlx-native"))]
        let _ = enabled;
    }

    /// Update bounded fair-prefill scheduling for this live session.
    ///
    /// Servers use this when a second model becomes resident: a long prefill
    /// must be split into bounded engine turns so a sibling model's streaming
    /// decode can reacquire the shared device between chunks.
    pub fn set_multi_prefill_fair(
        &mut self,
        enabled: bool,
        max_tokens_per_request_per_step: u32,
        max_inflight_prefill_requests: u32,
    ) {
        self.core.set_multi_prefill_fair(
            enabled,
            max_tokens_per_request_per_step,
            max_inflight_prefill_requests,
        );
        self.config.multi_prefill_fair = enabled;
        self.config.max_prefill_tokens_per_request_per_step = max_tokens_per_request_per_step;
        self.config.max_inflight_prefill_requests = max_inflight_prefill_requests;
    }

    /// The operator-declared MLX prefill chunk (`--prefill-chunk`), the
    /// ring-safe upper bound for a single multi-token forward.
    pub fn mlx_prefill_chunk_limit(&self) -> Option<usize> {
        self.config.mlx_prefill_chunk
    }

    /// Return the live fair-prefill policy.
    ///
    /// The server uses this to change only the token quantum that actually
    /// differs when multi-model latency isolation moves between its
    /// throughput and sibling-active modes.
    pub fn multi_prefill_policy(&self) -> (bool, u32, u32) {
        (
            self.config.multi_prefill_fair,
            self.config.max_prefill_tokens_per_request_per_step,
            self.config.max_inflight_prefill_requests,
        )
    }

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
        if request.sampling.no_repeat_ngram_size > 0
            && (request.sampling.ngram_window == 0
                || request.sampling.no_repeat_ngram_size > request.sampling.ngram_window)
        {
            return Err(EngineSessionError::InvalidNoRepeatNgram {
                no_repeat_ngram_size: request.sampling.no_repeat_ngram_size,
                ngram_window: request.sampling.ngram_window,
            });
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

    fn validate_generate_request_for_backend(
        selected_backend: SelectedBackend,
        max_batch_tokens: u32,
        request_id: u64,
        request: &GenerateRequest,
    ) -> Result<(), EngineSessionError> {
        Self::validate_generate_request(request_id, request)?;
        if !selected_backend.is_mlx() && !request.multimodal_inputs.is_empty() {
            return Err(EngineSessionError::MultimodalInputsRequireNativeMlx { selected_backend });
        }
        if selected_backend.is_mlx() && !request.multimodal_inputs.is_empty() {
            // Multimodal prefill is atomic (the runner requires the complete
            // prompt in one execution item), so a prompt longer than the
            // per-step token budget could never be scheduled. Reject it here
            // with an actionable error instead of deferring it forever.
            if request.input_tokens.len() > max_batch_tokens as usize {
                return Err(EngineSessionError::MultimodalPromptExceedsMaxBatchTokens {
                    prompt_len: request.input_tokens.len() as u32,
                    max_batch_tokens,
                });
            }
            if !request.input_tokens.is_empty() {
                request
                    .multimodal_inputs
                    .validate_for_prompt_tokens(&request.input_tokens)?;
            }
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
        Self::validate_generate_request_for_backend(
            self.config.resolved_backend.selected_backend,
            self.config.max_batch_tokens,
            request_id,
            &request,
        )?;
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
        Self::validate_generate_request_for_backend(
            self.config.resolved_backend.selected_backend,
            self.config.max_batch_tokens,
            request_id,
            &request,
        )?;
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

    /// Deterministic native session for binding-crate unit tests (no MLX weights).
    ///
    /// Uses `DeterministicRunner` so stream submit/step/cancel can be exercised
    /// without model artifacts. Selected backend remains MLX-native so the
    /// native stream / stepwise paths are the ones under test.
    #[doc(hidden)]
    pub fn new_deterministic_native_for_tests() -> Self {
        use ax_engine_core::{DeterministicRunner, DeterministicSampler};

        let config = EngineSessionConfig {
            // Clear auto-detected artifact dirs so tests never touch real models.
            mlx_runtime_artifacts_dir: None,
            mlx_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        };
        let core = EngineCore::with_runtime_components(
            config.kv_config,
            DeterministicRunner,
            DeterministicSampler,
        );
        Self {
            core,
            runtime: config.runtime_report(),
            config,
            #[cfg(feature = "mlx-native")]
            whisper: None,
            next_request_id: 1,
            native_request_routes: BTreeMap::new(),
            native_route_report_order: VecDeque::new(),
            llama_requests: BTreeMap::new(),
            llama_terminal_request_order: VecDeque::new(),
        }
    }

    #[cfg(feature = "mlx-native")]
    pub fn new_with_shared_mlx_prefix_cache(
        config: EngineSessionConfig,
        prefix_cache_store: ax_engine_mlx::MlxPrefixCacheStore,
    ) -> Result<Self, EngineSessionError> {
        Self::new_with_shared_mlx_runtime(config, Some(prefix_cache_store), None)
    }

    /// Build a session that reuses cross-session native-MLX state: an optional
    /// prefix snapshot store and an optional shared-weights cell (see
    /// `MlxSharedWeightsCell`). Request KV state remains private to the
    /// session either way.
    #[cfg(feature = "mlx-native")]
    pub fn new_with_shared_mlx_runtime(
        config: EngineSessionConfig,
        prefix_cache_store: Option<ax_engine_mlx::MlxPrefixCacheStore>,
        shared_weights: Option<&ax_engine_mlx::MlxSharedWeightsCell>,
    ) -> Result<Self, EngineSessionError> {
        config.validate()?;
        let core = build_native_core_with_mlx_shares(&config, prefix_cache_store, shared_weights)?;
        Self::from_validated_config_and_core(config, core)
    }

    fn from_validated_config_and_core(
        config: EngineSessionConfig,
        core: EngineCore,
    ) -> Result<Self, EngineSessionError> {
        #[cfg(feature = "mlx-native")]
        let whisper = load_native_whisper_model(&config)?;
        let runtime = config
            .runtime_report()
            .with_mlx_model(resolve_native_model_report(&config, &core));
        Ok(Self {
            core,
            config,
            runtime,
            #[cfg(feature = "mlx-native")]
            whisper,
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

        Self::validate_generate_request_for_backend(
            config.resolved_backend.selected_backend,
            config.max_batch_tokens,
            request_id,
            &request,
        )?;
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
        self.step_report_with_request_ids()
            .map(|(report, _)| report)
    }

    /// Advance the session once and return the report with its selected request IDs.
    pub fn step_report_with_request_ids(
        &mut self,
    ) -> Result<(EngineStepReport, Vec<u64>), EngineSessionError> {
        if !self.uses_mlx_runtime() {
            let active_request_ids = self.llama_active_request_ids();
            if active_request_ids.is_empty() {
                return Ok((EngineStepReport::default(), Vec::new()));
            }
            let selected_backend = self.config.resolved_backend.selected_backend;
            let mut aggregate = EngineStepReport::default();
            let mut selected_request_ids = Vec::new();

            for request_id in active_request_ids {
                // Persist a terminal transition immediately, in the same
                // loop iteration that computed it, rather than deferring to
                // a flush pass after the whole loop. A later request's
                // `step_report` error (`?` below) used to abort the
                // function before that deferred flush ran, leaving an
                // already-finished request's slot stuck `Active` — on the
                // next poll it would be re-queried, its stream already
                // fully consumed, and `step_report` would raise
                // `LlamaCppStreamEndedBeforeStop` for a request that had in
                // fact already completed successfully, permanently.
                let terminal = {
                    let slot = self
                        .llama_requests
                        .get_mut(&request_id)
                        .ok_or(EngineSessionError::MissingRequestSnapshot { request_id })?;
                    let LlamaCppLifecycleRequestSlot::Active(request) = slot else {
                        continue;
                    };
                    let step = request.step_report(selected_backend)?;
                    if step.scheduled_requests > 0 {
                        selected_request_ids.push(request_id);
                    }
                    aggregate.accumulate(step);
                    if is_terminal_request_state(request.current_report.state) {
                        request.drain_trailing_usage();
                        Some((request_id, request.current_report.clone()))
                    } else {
                        None
                    }
                };
                if let Some((request_id, report)) = terminal {
                    self.store_terminal_llama_report(request_id, report);
                }
            }

            return Ok((aggregate, selected_request_ids));
        }

        let outcome = self.step()?;
        let selected_request_ids: Vec<u64> = outcome
            .schedule_plan
            .selected_requests
            .iter()
            .map(|request_id| request_id.0)
            .collect();
        // Metal dispatch traces are opt-in (AX_ENGINE_DISPATCH_TRACE) and are
        // usually None on the pure MLX runner. Skip the mutex/clone when absent.
        let metal_dispatch = outcome
            .runner_output
            .as_ref()
            .and_then(|_| self.core.last_metal_dispatch())
            .map(|trace| MetalDispatchStepReport::from_trace(&trace));
        // Decode steps re-emit large crossover maps every token. Building the
        // full GenerateRouteReport (String clones + BTreeMap) per token was a
        // multi-ms tax on M5 Max Qwen3.5-9B SSE. Skip intermediate pure
        // single-token decode conversion only after a stored route already
        // carries real decode-path counters. Always capture prefill, first
        // decode after prefill (bootstrap-only → pipeline/single), multi-token
        // steps, and the terminal step so cumulative pipeline_steps reach the
        // harness (2026-07-26 Gemma@2048 froze prefill-only bootstrap).
        let has_terminal_update = outcome.runner_output.as_ref().is_some_and(|output| {
            output
                .request_updates
                .iter()
                .any(|update| update.stop_reason.is_some())
        });
        let any_request_missing_route = selected_request_ids
            .iter()
            .any(|request_id| !self.native_request_routes.contains_key(request_id));
        let any_stored_route_lacks_decode_work = selected_request_ids.iter().any(|request_id| {
            self.native_request_routes
                .get(request_id)
                .is_some_and(|route| !route_has_decode_path_work(route))
        });
        let needs_route = native_step_needs_route_capture(
            outcome.metrics.scheduled_tokens,
            outcome.metrics.ttft_events,
            has_terminal_update,
            any_request_missing_route,
            any_stored_route_lacks_decode_work,
        );
        let report = if needs_route {
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
                // Always merge when we paid to materialise the route: decode
                // counters are max-merged into any prefill-only store.
                for request_id in request_ids {
                    self.store_native_request_route(request_id, route.clone());
                }
            }
            report
        } else {
            EngineStepReport::from_native_outcome_without_route(&outcome, metal_dispatch)
        };
        Ok((report, selected_request_ids))
    }

    /// True when this session has any stepwise (`submit_generate`/`step`)
    /// request that has not yet reached a terminal state. Callers that are
    /// about to discard this session (e.g. a model hot-swap) should check
    /// this first: request state lives entirely inside the `EngineSession`
    /// instance, with no cross-session registry, so replacing the session
    /// while a request is non-terminal silently orphans it — the client's
    /// next `/v1/requests/:id` or `/v1/step` call finds nothing and gets a
    /// bare "not found" instead of a real terminal state, and the request's
    /// GPU/KV resources are only reclaimed once the old session's last `Arc`
    /// reference is dropped.
    pub fn has_active_stepwise_requests(&self) -> bool {
        if !self.uses_mlx_runtime() {
            return self
                .llama_requests
                .values()
                .any(|slot| matches!(slot, LlamaCppLifecycleRequestSlot::Active(_)));
        }
        self.core
            .request_manager()
            .records_iter()
            .any(|record| !record.state.is_terminal())
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
        #[cfg(feature = "mlx-native")]
        if self.whisper.is_some() {
            return Err(EngineSessionError::WhisperTextGenerationUnsupported);
        }
        Self::validate_generate_request_for_backend(
            self.config.resolved_backend.selected_backend,
            self.config.max_batch_tokens,
            request_id,
            &request,
        )?;
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
            multimodal_inputs: request.multimodal_inputs,
            sampling_params: request.sampling.into_core(self.config.deterministic),
            max_output_tokens: request.max_output_tokens,
            arrival_sequence: SequenceNo(request_id.0),
            metadata: request.metadata,
        };

        self.submit(submission)?;
        Ok(request_id.0)
    }

    /// Transcribe or translate a mono, 16 kHz waveform with the loaded native
    /// Whisper runtime.
    pub fn transcribe_audio(
        &self,
        samples_16k: &[f32],
        language: Option<&str>,
        translate: bool,
    ) -> Result<SpeechTranscription, EngineSessionError> {
        #[cfg(feature = "mlx-native")]
        {
            let whisper = self
                .whisper
                .as_ref()
                .ok_or(EngineSessionError::WhisperUnavailable)?;
            whisper
                .transcribe(samples_16k, language, translate)
                .map(|result| SpeechTranscription {
                    text: result.text,
                    language: result.language,
                })
                .map_err(|error| match error {
                    ax_engine_mlx::WhisperError::Language(language) => {
                        EngineSessionError::WhisperInvalidLanguage { language }
                    }
                    other => EngineSessionError::WhisperFailed {
                        message: other.to_string(),
                    },
                })
        }
        #[cfg(not(feature = "mlx-native"))]
        {
            let _ = (samples_16k, language, translate);
            Err(EngineSessionError::WhisperUnavailable)
        }
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
            Self::validate_generate_request_for_backend(
                self.config.resolved_backend.selected_backend,
                self.config.max_batch_tokens,
                request_id,
                &request,
            )?;
            // Delegated generations have no engine-side submission to advance
            // the id, so advance here — otherwise every repeated `generate()`
            // on a delegated backend reuses request_id 1.
            self.advance_request_id(request_id);
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
            Self::validate_generate_request_for_backend(
                self.config.resolved_backend.selected_backend,
                self.config.max_batch_tokens,
                request_id,
                &request,
            )?;
            return match self.config.resolved_backend.selected_backend {
                SelectedBackend::LlamaCpp => {
                    self.llama_cpp_stream_state_with_request_id(request_id, request)
                }
                SelectedBackend::MlxLmDelegated => {
                    // No engine-side submission advances the id on this path;
                    // advance here so concurrent mlx-lm streams get distinct
                    // request ids (the llama.cpp branch already does).
                    self.advance_request_id(request_id);
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

    /// Advance a native stream with an engine step already executed by a shared worker.
    pub fn next_native_stream_event_after_step(
        &mut self,
        state: &mut GenerateStreamState,
        step: EngineStepReport,
    ) -> Result<GenerateStreamEvent, EngineSessionError> {
        let request_id = state.request_id();
        let GenerateStreamState::Native(state) = state else {
            return Err(EngineSessionError::RequestReportInvariantViolation {
                request_id,
                message: "externally stepped advancement requires a native MLX stream",
            });
        };
        if state.phase != GenerateStreamPhase::Step
            || is_terminal_request_state(state.current_report.state)
        {
            return Err(EngineSessionError::RequestReportInvariantViolation {
                request_id: state.request_id,
                message: "native stream cannot consume an engine step in its current phase",
            });
        }
        self.apply_native_stream_step(state.as_mut(), step)
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
                    let final_report = self.request_report(state.request_id).ok_or(
                        EngineSessionError::MissingRequestSnapshot {
                            request_id: state.request_id,
                        },
                    )?;
                    return Ok(Some(GenerateStreamEvent::Response(
                        GenerateStreamResponseEvent {
                            response: GenerateResponse::from_report(
                                final_report,
                                state.step_count,
                                state.ttft_step,
                                state.runtime.clone(),
                            ),
                        },
                    )));
                }

                // First output token is emitted alone for TTFT; subsequent
                // tokens are coalesced into multi-token step events so the
                // SSE path is not paced by one channel/HTTP frame per token
                // (M5 Max Qwen3.5-9B: native ~106 tok/s vs 1-token SSE ~81).
                // Batch size stays small enough for the flip stream-gap cap
                // (≤50 ms ≈ 4 tokens at ~110 tok/s).
                let step = self.step_report()?;
                let mut event = self.apply_native_stream_step(state, step)?;
                if state.ttft_step.is_some()
                    && !is_terminal_request_state(state.current_report.state)
                {
                    const STREAM_TOKEN_BATCH: usize = 1;
                    let mut batch_tokens = match &event {
                        GenerateStreamEvent::Step(step) => step.delta_tokens.clone(),
                        _ => Vec::new(),
                    };
                    let mut batch_logprobs = match &event {
                        GenerateStreamEvent::Step(step) => step.delta_token_logprobs.clone(),
                        _ => Vec::new(),
                    };
                    let mut last_step = match &event {
                        GenerateStreamEvent::Step(step) => step.step.clone(),
                        _ => {
                            return Ok(Some(event));
                        }
                    };
                    while batch_tokens.len() < STREAM_TOKEN_BATCH
                        && !is_terminal_request_state(state.current_report.state)
                    {
                        let step = self.step_report()?;
                        let next = self.apply_native_stream_step(state, step)?;
                        let GenerateStreamEvent::Step(next_step) = next else {
                            break;
                        };
                        batch_tokens.extend(next_step.delta_tokens);
                        batch_logprobs.extend(next_step.delta_token_logprobs);
                        last_step = next_step.step;
                        if is_terminal_request_state(state.current_report.state) {
                            break;
                        }
                    }
                    event = GenerateStreamEvent::Step(GenerateStreamStepEvent {
                        request: state.current_report.clone(),
                        step: last_step,
                        delta_tokens: batch_tokens,
                        delta_token_logprobs: batch_logprobs,
                        delta_text: None,
                    });
                }
                Ok(Some(event))
            }
            GenerateStreamPhase::Done => Ok(None),
        }
    }

    fn apply_native_stream_step(
        &self,
        state: &mut NativeGenerateStreamState,
        step: EngineStepReport,
    ) -> Result<GenerateStreamEvent, EngineSessionError> {
        state.step_count += 1;
        if state.step_count >= state.max_steps {
            return Err(EngineSessionError::RequestDidNotTerminate {
                request_id: state.request_id,
                max_steps: state.max_steps,
            });
        }

        // Fast path: borrow the live record and copy only the new token slice.
        // Full `request_report()` clones prompt + full output history + route
        // map on every token; that alone was multi-ms/token on M5 Max for
        // OpenAI SSE vs the same worker's non-stream consumer.
        let progress = self
            .stream_step_progress(state.request_id, state.emitted_output_len)
            .ok_or(EngineSessionError::MissingRequestSnapshot {
                request_id: state.request_id,
            })?;

        if state.ttft_step.is_none()
            && state.emitted_output_len == 0
            && !progress.delta_tokens.is_empty()
        {
            state.ttft_step = Some(state.step_count);
        }
        state.emitted_output_len = progress.output_len as usize;

        // Update the in-place progress report with scalars only. Skip route /
        // string field churn on intermediate steps.
        state.current_report.state = progress.state;
        state.current_report.processed_prompt_tokens = progress.processed_prompt_tokens;
        state.current_report.prompt_len = progress.prompt_len;
        state.current_report.output_len = progress.output_len;
        state.current_report.max_output_tokens = progress.max_output_tokens;
        state.current_report.cancel_requested = progress.cancel_requested;
        state.current_report.finish_reason = progress.finish_reason;
        state.current_report.terminal_stop_reason = progress.terminal_stop_reason;
        if progress.terminal {
            // Restore full token histories once for the terminal Step contract;
            // the following Response event also reloads this final snapshot.
            let mut final_report = self.request_report(state.request_id).ok_or(
                EngineSessionError::MissingRequestSnapshot {
                    request_id: state.request_id,
                },
            )?;
            apply_native_step_route_to_report(&mut final_report, &step);
            if let Some(route) = self.native_request_routes.get(&state.request_id) {
                merge_native_route_into(&mut final_report.route, route.clone());
            }
            if let Some(err) = progress.last_error {
                final_report.last_error = Some(err);
            }
            state.current_report = final_report;
        }

        // Intermediate step events do not need the heavy EngineStepReport
        // (route map / metal dispatch). Keep scalars only via Default unless
        // this is a terminal transition.
        let step_for_event = if progress.terminal {
            step
        } else {
            EngineStepReport {
                step_id: step.step_id,
                scheduled_requests: step.scheduled_requests,
                scheduled_tokens: step.scheduled_tokens,
                ttft_events: step.ttft_events,
                prefix_hits: step.prefix_hits,
                kv_usage_blocks: step.kv_usage_blocks,
                waiting_requests: step.waiting_requests,
                evictions: step.evictions,
                preempted_requests: step.preempted_requests,
                preempted_tokens: step.preempted_tokens,
                cpu_time_us: step.cpu_time_us,
                runner_time_us: step.runner_time_us,
                route: None,
                metal_dispatch: None,
            }
        };

        Ok(GenerateStreamEvent::Step(GenerateStreamStepEvent {
            request: state.current_report.clone(),
            step: step_for_event,
            delta_tokens: progress.delta_tokens,
            delta_token_logprobs: progress.delta_token_logprobs,
            delta_text: None,
        }))
    }

    /// Progress + delta tokens for a native stream step without cloning the
    /// full prompt/output history.
    fn stream_step_progress(
        &self,
        request_id: u64,
        emitted_output_len: usize,
    ) -> Option<StreamStepProgress> {
        if !self.uses_mlx_runtime() {
            let report = self.request_report(request_id)?;
            if emitted_output_len > report.output_tokens.len() {
                return None;
            }
            let delta_tokens = report.output_tokens[emitted_output_len..].to_vec();
            let delta_token_logprobs =
                slice_output_token_logprobs(&report, emitted_output_len, delta_tokens.len())
                    .ok()?;
            let terminal = is_terminal_request_state(report.state);
            return Some(StreamStepProgress {
                delta_tokens,
                delta_token_logprobs,
                state: report.state,
                processed_prompt_tokens: report.processed_prompt_tokens,
                prompt_len: report.prompt_len,
                output_len: report.output_len,
                max_output_tokens: report.max_output_tokens,
                cancel_requested: report.cancel_requested,
                finish_reason: report.finish_reason,
                terminal_stop_reason: report.terminal_stop_reason,
                last_error: report.last_error,
                terminal,
            });
        }

        // Prefer the live record (no full-history clone). After the finishing
        // step the core may already have moved the request into
        // `terminal_snapshots`, so fall back to `request_report()` there.
        if let Some(record) = self.core.request_manager().record(RequestId(request_id)) {
            let generated = &record.generated_tokens;
            if emitted_output_len > generated.len() {
                return None;
            }
            let delta_tokens = generated[emitted_output_len..].to_vec();
            let delta_token_logprobs = if record.generated_token_logprobs.is_empty() {
                vec![None; delta_tokens.len()]
            } else if record.generated_token_logprobs.len() == generated.len() {
                record.generated_token_logprobs[emitted_output_len..].to_vec()
            } else {
                let report = self.request_report(request_id)?;
                return slice_output_token_logprobs(
                    &report,
                    emitted_output_len,
                    delta_tokens.len(),
                )
                .ok()
                .map(|delta_token_logprobs| {
                    let terminal = is_terminal_request_state(report.state);
                    StreamStepProgress {
                        delta_tokens,
                        delta_token_logprobs,
                        state: report.state,
                        processed_prompt_tokens: report.processed_prompt_tokens,
                        prompt_len: report.prompt_len,
                        output_len: report.output_len,
                        max_output_tokens: report.max_output_tokens,
                        cancel_requested: report.cancel_requested,
                        finish_reason: report.finish_reason,
                        terminal_stop_reason: report.terminal_stop_reason,
                        last_error: report.last_error,
                        terminal,
                    }
                });
            };
            let state = SessionRequestState::from(record.state);
            let terminal = is_terminal_request_state(state);
            return Some(StreamStepProgress {
                delta_tokens,
                delta_token_logprobs,
                state,
                processed_prompt_tokens: record.processed_prompt_tokens,
                prompt_len: record.prompt_tokens.len() as u32,
                output_len: generated.len() as u32,
                max_output_tokens: record.max_output_tokens,
                cancel_requested: record.cancel_requested,
                finish_reason: crate::generate::GenerateFinishReason::from_request_state(
                    record.state,
                    record.terminal_stop_reason,
                ),
                terminal_stop_reason: record.terminal_stop_reason,
                last_error: record.last_error.clone(),
                terminal,
            });
        }

        let report = self.request_report(request_id)?;
        if emitted_output_len > report.output_tokens.len() {
            return None;
        }
        let delta_tokens = report.output_tokens[emitted_output_len..].to_vec();
        let delta_token_logprobs =
            slice_output_token_logprobs(&report, emitted_output_len, delta_tokens.len()).ok()?;
        let terminal = is_terminal_request_state(report.state);
        Some(StreamStepProgress {
            delta_tokens,
            delta_token_logprobs,
            state: report.state,
            processed_prompt_tokens: report.processed_prompt_tokens,
            prompt_len: report.prompt_len,
            output_len: report.output_len,
            max_output_tokens: report.max_output_tokens,
            cancel_requested: report.cancel_requested,
            finish_reason: report.finish_reason,
            terminal_stop_reason: report.terminal_stop_reason,
            last_error: report.last_error,
            terminal,
        })
    }
}

struct StreamStepProgress {
    delta_tokens: Vec<u32>,
    delta_token_logprobs: Vec<Option<f32>>,
    state: SessionRequestState,
    processed_prompt_tokens: u32,
    prompt_len: u32,
    output_len: u32,
    max_output_tokens: u32,
    cancel_requested: bool,
    finish_reason: Option<crate::generate::GenerateFinishReason>,
    terminal_stop_reason: Option<ax_engine_core::StopReason>,
    last_error: Option<String>,
    terminal: bool,
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
