use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::path::{Path, PathBuf};

use ax_engine_core::{
    CacheGroupId, DeterministicRunner, DeterministicSampler, EngineCore, EngineCoreError,
    EngineStepOutcome, ExecutionRunner, KvManagerConfig, MetalKernelAssets, MetalRuntimeError,
    ModelId, RequestId, RequestSubmission, RouteMetadata, RunnerInput, RunnerOutput, SequenceNo,
};
use thiserror::Error;

use crate::backend::{
    resolve_preview_backend, BackendContractError, BackendPolicy, CapabilityReport,
    NativeModelArtifactsSource, NativeModelReport, NativeRunnerKind, NativeRuntimeArtifactsSource,
    NativeRuntimeReport, PreviewBackendRequest, PreviewBackendResolutionError, ResolvedBackend,
    RuntimeReport, SelectedBackend, SupportTier,
};
use crate::compat::{
    run_blocking_generate, start_streaming_generate, CompatibilityBackendConfig,
    CompatibilityBackendError, CompatibilityPromptProgress, CompatibilityStreamChunk,
    CompatibilityStreamHandle,
};
use crate::generate::{
    GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateStreamEvent,
    GenerateStreamRequestEvent, GenerateStreamResponseEvent, GenerateStreamStepEvent,
};
use crate::host;
use crate::request::{
    EngineStepReport, MetalDispatchStepReport, SessionRequestReport, SessionRequestState,
};

const COMPATIBILITY_STREAM_EXECUTION_PLAN: &str =
    "compatibility.llama_cpp.server_completion_stream";
const NATIVE_METAL_BUILD_DIR_ENV: &str = "AX_ENGINE_METAL_BUILD_DIR";
const NATIVE_MODEL_DIR_ENV: &str = "AX_ENGINE_NATIVE_MODEL_DIR";
const NATIVE_PLACEHOLDER_EXECUTION_DECISION: &str = "native_placeholder_execution";
const NATIVE_PLACEHOLDER_EXPLICIT_OPT_IN_DECISION: &str = "native_placeholder_explicit_opt_in";
const NATIVE_RUNTIME_RUNNER_DETERMINISTIC_DECISION: &str = "native_runtime_runner_deterministic";
const NATIVE_RUNTIME_RUNNER_METAL_BRINGUP_DECISION: &str = "native_runtime_runner_metal_bringup";
const NATIVE_RUNTIME_ARTIFACTS_VALIDATED_DECISION: &str = "native_runtime_artifacts_validated";
const MAX_COMPATIBILITY_TERMINAL_REQUESTS: usize = 1024;

#[derive(Clone, Debug)]
pub struct EngineSessionConfig {
    pub kv_config: KvManagerConfig,
    pub deterministic: bool,
    pub allow_deterministic_native_fallback: bool,
    pub max_batch_tokens: u32,
    pub backend_policy: BackendPolicy,
    pub resolved_backend: ResolvedBackend,
    pub compatibility_backend: Option<CompatibilityBackendConfig>,
    pub native_runtime_artifacts_dir: Option<PathBuf>,
    pub native_runtime_artifacts_source: Option<NativeRuntimeArtifactsSource>,
    pub native_model_artifacts_dir: Option<PathBuf>,
    pub native_model_artifacts_source: Option<NativeModelArtifactsSource>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewSessionConfigRequest {
    pub cache_group_id: CacheGroupId,
    pub block_size_tokens: u32,
    pub total_blocks: u32,
    pub deterministic: bool,
    pub allow_deterministic_native_fallback: bool,
    pub max_batch_tokens: u32,
    pub backend_request: PreviewBackendRequest,
    pub native_runtime_artifacts_dir: Option<PathBuf>,
    pub native_model_artifacts_dir: Option<PathBuf>,
}

impl Default for PreviewSessionConfigRequest {
    fn default() -> Self {
        Self {
            cache_group_id: CacheGroupId(0),
            block_size_tokens: 16,
            total_blocks: 1024,
            deterministic: true,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 2048,
            backend_request: PreviewBackendRequest::default(),
            native_runtime_artifacts_dir: None,
            native_model_artifacts_dir: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedSessionConfigRequest {
    pub cache_group_id: CacheGroupId,
    pub block_size_tokens: u32,
    pub total_blocks: u32,
    pub deterministic: bool,
    pub allow_deterministic_native_fallback: bool,
    pub max_batch_tokens: u32,
    pub backend_policy: BackendPolicy,
    pub resolved_backend: ResolvedBackend,
    pub compatibility_backend: Option<CompatibilityBackendConfig>,
    pub native_runtime_artifacts_dir: Option<PathBuf>,
    pub native_runtime_artifacts_source: Option<NativeRuntimeArtifactsSource>,
    pub native_model_artifacts_dir: Option<PathBuf>,
    pub native_model_artifacts_source: Option<NativeModelArtifactsSource>,
}

impl Default for ResolvedSessionConfigRequest {
    fn default() -> Self {
        let default = EngineSessionConfig::default();
        Self {
            cache_group_id: default.kv_config.cache_group_id,
            block_size_tokens: default.kv_config.block_size_tokens,
            total_blocks: default.kv_config.total_blocks,
            deterministic: default.deterministic,
            allow_deterministic_native_fallback: default.allow_deterministic_native_fallback,
            max_batch_tokens: default.max_batch_tokens,
            backend_policy: default.backend_policy,
            resolved_backend: default.resolved_backend,
            compatibility_backend: default.compatibility_backend,
            native_runtime_artifacts_dir: default.native_runtime_artifacts_dir,
            native_runtime_artifacts_source: default.native_runtime_artifacts_source,
            native_model_artifacts_dir: default.native_model_artifacts_dir,
            native_model_artifacts_source: default.native_model_artifacts_source,
        }
    }
}

#[derive(Debug, Error)]
pub enum PreviewSessionConfigError {
    #[error(transparent)]
    BackendResolution(#[from] PreviewBackendResolutionError),
}

impl Default for EngineSessionConfig {
    fn default() -> Self {
        let native_runtime_artifacts = Self::default_native_runtime_artifacts_selection();
        let native_model_artifacts = Self::default_native_model_artifacts_selection();
        Self {
            kv_config: KvManagerConfig::new(CacheGroupId(0), 16, 1024),
            deterministic: true,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 2048,
            backend_policy: BackendPolicy::strict_native(),
            resolved_backend: ResolvedBackend::native_preview(),
            compatibility_backend: None,
            native_runtime_artifacts_dir: native_runtime_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone()),
            native_runtime_artifacts_source: native_runtime_artifacts
                .map(|selection| selection.source),
            native_model_artifacts_dir: native_model_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone()),
            native_model_artifacts_source: native_model_artifacts.map(|selection| selection.source),
        }
    }
}

impl EngineSessionConfig {
    pub fn from_preview_request(
        request: PreviewSessionConfigRequest,
    ) -> Result<Self, PreviewSessionConfigError> {
        let resolution = resolve_preview_backend(request.backend_request)?;
        let default = Self::default();
        let native_runtime_artifacts =
            request
                .native_runtime_artifacts_dir
                .map(|dir| NativeRuntimeArtifactsSelection {
                    dir,
                    source: NativeRuntimeArtifactsSource::ExplicitConfig,
                });
        let native_model_artifacts =
            request
                .native_model_artifacts_dir
                .map(|dir| NativeModelArtifactsSelection {
                    dir,
                    source: NativeModelArtifactsSource::ExplicitConfig,
                });

        Ok(Self {
            kv_config: KvManagerConfig::new(
                request.cache_group_id,
                request.block_size_tokens,
                request.total_blocks,
            ),
            deterministic: request.deterministic,
            allow_deterministic_native_fallback: request.allow_deterministic_native_fallback,
            max_batch_tokens: request.max_batch_tokens,
            backend_policy: resolution.backend_policy,
            resolved_backend: resolution.resolved_backend,
            compatibility_backend: resolution.compatibility_backend,
            native_runtime_artifacts_dir: native_runtime_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone())
                .or(default.native_runtime_artifacts_dir),
            native_runtime_artifacts_source: native_runtime_artifacts
                .map(|selection| selection.source)
                .or(default.native_runtime_artifacts_source),
            native_model_artifacts_dir: native_model_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone())
                .or(default.native_model_artifacts_dir),
            native_model_artifacts_source: native_model_artifacts
                .map(|selection| selection.source)
                .or(default.native_model_artifacts_source),
        })
    }

    pub fn default_native_runtime_artifacts_dir() -> Option<PathBuf> {
        Self::default_native_runtime_artifacts_selection().map(|selection| selection.dir)
    }

    pub fn default_native_runtime_artifacts_source() -> Option<NativeRuntimeArtifactsSource> {
        Self::default_native_runtime_artifacts_selection().map(|selection| selection.source)
    }

    pub fn default_native_model_artifacts_dir() -> Option<PathBuf> {
        Self::default_native_model_artifacts_selection().map(|selection| selection.dir)
    }

    pub fn default_native_model_artifacts_source() -> Option<NativeModelArtifactsSource> {
        Self::default_native_model_artifacts_selection().map(|selection| selection.source)
    }

    pub fn from_resolved_request(request: ResolvedSessionConfigRequest) -> Self {
        Self {
            kv_config: KvManagerConfig::new(
                request.cache_group_id,
                request.block_size_tokens,
                request.total_blocks,
            ),
            deterministic: request.deterministic,
            allow_deterministic_native_fallback: request.allow_deterministic_native_fallback,
            max_batch_tokens: request.max_batch_tokens,
            backend_policy: request.backend_policy,
            resolved_backend: request.resolved_backend,
            compatibility_backend: request.compatibility_backend,
            native_runtime_artifacts_dir: request.native_runtime_artifacts_dir,
            native_runtime_artifacts_source: request.native_runtime_artifacts_source,
            native_model_artifacts_dir: request.native_model_artifacts_dir,
            native_model_artifacts_source: request.native_model_artifacts_source,
        }
    }

    fn default_native_runtime_artifacts_selection() -> Option<NativeRuntimeArtifactsSelection> {
        let explicit_dir = env::var_os(NATIVE_METAL_BUILD_DIR_ENV).map(PathBuf::from);
        let current_dir = env::current_dir().ok();
        resolve_default_native_runtime_artifacts_selection(explicit_dir, current_dir.as_deref())
    }

    fn default_native_model_artifacts_selection() -> Option<NativeModelArtifactsSelection> {
        env::var_os(NATIVE_MODEL_DIR_ENV)
            .map(PathBuf::from)
            .map(|dir| NativeModelArtifactsSelection {
                dir,
                source: NativeModelArtifactsSource::ExplicitEnv,
            })
    }

    pub fn validate(&self) -> Result<(), EngineSessionError> {
        self.resolved_backend
            .validate_against(&self.backend_policy)?;

        if self.max_batch_tokens == 0 {
            return Err(EngineSessionError::InvalidMaxBatchTokens);
        }

        if self.resolved_backend.support_tier == SupportTier::Unsupported {
            return Err(EngineSessionError::UnsupportedSupportTier);
        }

        if let Err(detected_host) = host::validate_local_host() {
            return Err(EngineSessionError::UnsupportedHostHardware { detected_host });
        }

        if self.resolved_backend.selected_backend == SelectedBackend::AxNative
            && self.native_runtime_artifacts_dir().is_none()
            && !self.allow_deterministic_native_fallback
        {
            return Err(EngineSessionError::NativeRuntimeArtifactsRequired);
        }

        if self.resolved_backend.selected_backend != SelectedBackend::AxNative {
            let compatibility_backend = self.compatibility_backend.as_ref().ok_or(
                EngineSessionError::MissingCompatibilityBackendConfig {
                    selected_backend: self.resolved_backend.selected_backend,
                },
            )?;
            if compatibility_backend.selected_backend() != self.resolved_backend.selected_backend {
                return Err(EngineSessionError::CompatibilityBackendConfigMismatch {
                    configured_backend: compatibility_backend.selected_backend(),
                    selected_backend: self.resolved_backend.selected_backend,
                });
            }
        }

        Ok(())
    }

    pub fn runtime_report(&self) -> RuntimeReport {
        let mut runtime =
            RuntimeReport::from_resolution(&self.backend_policy, &self.resolved_backend)
                .with_native_runtime(self.native_runtime_report());
        if let Some(compatibility_backend) = self.compatibility_backend.as_ref() {
            runtime.capabilities =
                CapabilityReport::for_compatibility_backend(compatibility_backend);
        }
        runtime
    }

    pub fn native_runtime_artifacts_dir(&self) -> Option<&Path> {
        self.native_runtime_artifacts_dir.as_deref()
    }

    pub fn native_runtime_artifacts_source(&self) -> Option<NativeRuntimeArtifactsSource> {
        self.native_runtime_artifacts_source
    }

    pub fn native_model_artifacts_dir(&self) -> Option<&Path> {
        self.native_model_artifacts_dir.as_deref()
    }

    pub fn native_model_artifacts_source(&self) -> Option<NativeModelArtifactsSource> {
        self.native_model_artifacts_source
    }

    pub const fn allow_deterministic_native_fallback(&self) -> bool {
        self.allow_deterministic_native_fallback
    }

    fn native_runtime_report(&self) -> Option<NativeRuntimeReport> {
        if self.resolved_backend.selected_backend != SelectedBackend::AxNative {
            return None;
        }

        if self.native_runtime_artifacts_dir().is_some() {
            Some(NativeRuntimeReport::metal_bringup(
                self.native_runtime_artifacts_source()
                    .unwrap_or(NativeRuntimeArtifactsSource::ExplicitConfig),
            ))
        } else if self.allow_deterministic_native_fallback() {
            Some(NativeRuntimeReport::deterministic())
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeRuntimeArtifactsSelection {
    dir: PathBuf,
    source: NativeRuntimeArtifactsSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeModelArtifactsSelection {
    dir: PathBuf,
    source: NativeModelArtifactsSource,
}

fn resolve_default_native_runtime_artifacts_selection(
    explicit_dir: Option<PathBuf>,
    current_dir: Option<&Path>,
) -> Option<NativeRuntimeArtifactsSelection> {
    explicit_dir
        .map(|dir| NativeRuntimeArtifactsSelection {
            dir,
            source: NativeRuntimeArtifactsSource::ExplicitEnv,
        })
        .or_else(|| current_dir.and_then(detect_repo_owned_native_runtime_artifacts_dir_from))
}

fn detect_repo_owned_native_runtime_artifacts_dir_from(
    start_dir: &Path,
) -> Option<NativeRuntimeArtifactsSelection> {
    for candidate_root in start_dir.ancestors() {
        let manifest_path = candidate_root.join("metal/phase1-kernels.json");
        let build_dir = candidate_root.join("build/metal");
        let build_report_path = build_dir.join("build_report.json");

        if !manifest_path.is_file() || !build_report_path.is_file() {
            continue;
        }

        // Repo auto-detect should stay conservative: only opt into the Metal
        // bring-up path when the checked-in asset contract validates end to end.
        if MetalKernelAssets::from_build_dir(&build_dir).is_ok() {
            return Some(NativeRuntimeArtifactsSelection {
                dir: build_dir,
                source: NativeRuntimeArtifactsSource::RepoAutoDetect,
            });
        }
    }

    None
}
#[derive(Debug)]
pub struct EngineSession {
    core: EngineCore,
    config: EngineSessionConfig,
    runtime: RuntimeReport,
    next_request_id: u64,
    compatibility_requests: BTreeMap<u64, CompatibilityLifecycleRequestSlot>,
    compatibility_terminal_request_order: VecDeque<u64>,
}

#[derive(Debug)]
pub struct GenerateStream<'a> {
    session: &'a mut EngineSession,
    state: GenerateStreamState,
}

#[derive(Debug)]
pub enum GenerateStreamState {
    Native(Box<NativeGenerateStreamState>),
    Compatibility(Box<CompatibilityGenerateStreamState>),
}

#[derive(Debug)]
pub struct NativeGenerateStreamState {
    request_id: u64,
    runtime: RuntimeReport,
    current_report: SessionRequestReport,
    emitted_output_len: usize,
    max_steps: u64,
    step_count: u64,
    ttft_step: Option<u64>,
    phase: GenerateStreamPhase,
}

#[derive(Debug)]
pub struct CompatibilityGenerateStreamState {
    request_id: u64,
    runtime: RuntimeReport,
    current_report: SessionRequestReport,
    prompt_text: Option<String>,
    output_text: String,
    prompt_token_count: Option<u32>,
    output_token_count: Option<u32>,
    cached_prompt_tokens_observed: u32,
    prefix_hit_recorded: bool,
    step_count: u64,
    ttft_step: Option<u64>,
    terminal_chunk_seen: bool,
    stream: CompatibilityStreamHandle,
    phase: GenerateStreamPhase,
}

#[derive(Debug)]
enum CompatibilityLifecycleRequestSlot {
    Active(Box<CompatibilityLifecycleRequest>),
    Terminal(Box<SessionRequestReport>),
}

#[derive(Debug)]
struct CompatibilityLifecycleRequest {
    request_id: u64,
    current_report: SessionRequestReport,
    prompt_token_count: Option<u32>,
    output_token_count: Option<u32>,
    cached_prompt_tokens_observed: u32,
    prefix_hit_recorded: bool,
    step_count: u64,
    ttft_step: Option<u64>,
    stream: CompatibilityStreamHandle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerateStreamPhase {
    Request,
    Step,
    Done,
}

#[derive(Clone, Debug, Default)]
struct NativePlaceholderRunner;

impl ExecutionRunner for NativePlaceholderRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let mut output = DeterministicRunner.run(input);
        annotate_native_placeholder_route_metadata(&mut output.route_metadata);
        output
    }
}

fn annotate_native_placeholder_route_metadata(route_metadata: &mut RouteMetadata) {
    upsert_route_decision(
        &mut route_metadata.crossover_decisions,
        NATIVE_PLACEHOLDER_EXECUTION_DECISION,
        1,
    );
    upsert_route_decision(
        &mut route_metadata.crossover_decisions,
        NATIVE_PLACEHOLDER_EXPLICIT_OPT_IN_DECISION,
        1,
    );
    upsert_route_decision(
        &mut route_metadata.crossover_decisions,
        NATIVE_RUNTIME_RUNNER_DETERMINISTIC_DECISION,
        1,
    );
    upsert_route_decision(
        &mut route_metadata.crossover_decisions,
        NATIVE_RUNTIME_RUNNER_METAL_BRINGUP_DECISION,
        0,
    );
    upsert_route_decision(
        &mut route_metadata.crossover_decisions,
        NATIVE_RUNTIME_ARTIFACTS_VALIDATED_DECISION,
        0,
    );
}

fn annotate_native_placeholder_route_report(route: &mut crate::generate::GenerateRouteReport) {
    route
        .crossover_decisions
        .insert(NATIVE_PLACEHOLDER_EXECUTION_DECISION.to_string(), 1);
    route
        .crossover_decisions
        .insert(NATIVE_PLACEHOLDER_EXPLICIT_OPT_IN_DECISION.to_string(), 1);
    route
        .crossover_decisions
        .insert(NATIVE_RUNTIME_RUNNER_DETERMINISTIC_DECISION.to_string(), 1);
    route
        .crossover_decisions
        .insert(NATIVE_RUNTIME_RUNNER_METAL_BRINGUP_DECISION.to_string(), 0);
    route
        .crossover_decisions
        .insert(NATIVE_RUNTIME_ARTIFACTS_VALIDATED_DECISION.to_string(), 0);
}

fn upsert_route_decision(decisions: &mut Vec<(String, u32)>, key: &str, value: u32) {
    if let Some((_, existing_value)) = decisions
        .iter_mut()
        .find(|(existing_key, _)| existing_key == key)
    {
        *existing_value = value;
    } else {
        decisions.push((key.to_string(), value));
    }
}

impl EngineSession {
    fn uses_native_runtime(&self) -> bool {
        self.config.resolved_backend.selected_backend == SelectedBackend::AxNative
    }

    fn uses_native_placeholder_runtime(&self) -> bool {
        self.uses_native_runtime()
            && matches!(
                self.runtime
                    .native_runtime
                    .as_ref()
                    .map(|report| report.runner),
                Some(NativeRunnerKind::Deterministic)
            )
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

    fn compatibility_active_request_ids(&self) -> Vec<u64> {
        self.compatibility_requests
            .iter()
            .filter_map(|(request_id, slot)| match slot {
                CompatibilityLifecycleRequestSlot::Active(_) => Some(*request_id),
                CompatibilityLifecycleRequestSlot::Terminal(_) => None,
            })
            .collect()
    }

    fn store_terminal_compatibility_report(
        &mut self,
        request_id: u64,
        report: SessionRequestReport,
    ) {
        let already_terminal = matches!(
            self.compatibility_requests.get(&request_id),
            Some(CompatibilityLifecycleRequestSlot::Terminal(_))
        );
        self.compatibility_requests.insert(
            request_id,
            CompatibilityLifecycleRequestSlot::Terminal(Box::new(report)),
        );
        if !already_terminal {
            self.compatibility_terminal_request_order
                .push_back(request_id);
        }
        self.prune_terminal_compatibility_requests();
    }

    fn prune_terminal_compatibility_requests(&mut self) {
        while self.compatibility_terminal_request_order.len() > MAX_COMPATIBILITY_TERMINAL_REQUESTS
        {
            let Some(evicted_request_id) = self.compatibility_terminal_request_order.pop_front()
            else {
                break;
            };
            if matches!(
                self.compatibility_requests.get(&evicted_request_id),
                Some(CompatibilityLifecycleRequestSlot::Terminal(_))
            ) {
                self.compatibility_requests.remove(&evicted_request_id);
            }
        }
    }

    fn compatibility_backend_config(
        &self,
    ) -> Result<&CompatibilityBackendConfig, EngineSessionError> {
        self.config.compatibility_backend.as_ref().ok_or(
            EngineSessionError::MissingCompatibilityBackendConfig {
                selected_backend: self.config.resolved_backend.selected_backend,
            },
        )
    }

    fn compatibility_generate_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateResponse, EngineSessionError> {
        Self::validate_generate_request(request_id, &request)?;
        self.advance_request_id(request_id);
        let runtime = self.runtime_report();
        let compatibility_backend = self.compatibility_backend_config()?;

        run_blocking_generate(request_id, &runtime, compatibility_backend, &request)
            .map_err(EngineSessionError::from)
    }

    fn compatibility_submit_generate_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<u64, EngineSessionError> {
        Self::validate_generate_request(request_id, &request)?;
        self.advance_request_id(request_id);
        let runtime = self.runtime_report();
        let stream =
            start_streaming_generate(&runtime, self.compatibility_backend_config()?, &request)
                .map_err(EngineSessionError::from)?;
        let route = compatibility_stream_route(self.config.resolved_backend.selected_backend);
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

        self.compatibility_requests.insert(
            request_id,
            CompatibilityLifecycleRequestSlot::Active(Box::new(
                CompatibilityLifecycleRequest::new(request_id, current_report, stream),
            )),
        );
        Ok(request_id)
    }

    fn compatibility_stream_state_with_request_id(
        &mut self,
        request_id: u64,
        request: GenerateRequest,
    ) -> Result<GenerateStreamState, EngineSessionError> {
        Self::validate_generate_request(request_id, &request)?;
        self.advance_request_id(request_id);

        let runtime = self.runtime_report();
        let stream =
            start_streaming_generate(&runtime, self.compatibility_backend_config()?, &request)
                .map_err(EngineSessionError::from)?;
        let route = compatibility_stream_route(self.config.resolved_backend.selected_backend);
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

        Ok(GenerateStreamState::Compatibility(Box::new(
            CompatibilityGenerateStreamState::new(
                request_id,
                runtime,
                current_report,
                request.input_text,
                stream,
            ),
        )))
    }

    pub fn new(config: EngineSessionConfig) -> Result<Self, EngineSessionError> {
        config.validate()?;
        let core = build_native_core(&config)?;
        let runtime = config
            .runtime_report()
            .with_native_model(resolve_native_model_report(&config, &core));
        Ok(Self {
            core,
            config,
            runtime,
            next_request_id: 1,
            compatibility_requests: BTreeMap::new(),
            compatibility_terminal_request_order: VecDeque::new(),
        })
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
        if !self.uses_native_runtime() {
            let terminal_report = {
                let Some(slot) = self.compatibility_requests.get_mut(&request_id) else {
                    return Err(EngineSessionError::MissingRequestSnapshot { request_id });
                };
                match slot {
                    CompatibilityLifecycleRequestSlot::Active(request) => Some(request.cancel()),
                    CompatibilityLifecycleRequestSlot::Terminal(_) => None,
                }
            };
            if let Some(report) = terminal_report {
                self.store_terminal_compatibility_report(request_id, report);
            }
            return Ok(());
        }
        self.cancel(RequestId(request_id))
    }

    pub fn step(&mut self) -> Result<EngineStepOutcome, EngineSessionError> {
        if !self.uses_native_runtime() {
            return Err(
                EngineSessionError::CompatibilityBackendDoesNotSupportLifecycle {
                    selected_backend: self.config.resolved_backend.selected_backend,
                    operation: "step",
                },
            );
        }
        self.core
            .step(self.config.max_batch_tokens, self.config.deterministic)
            .map_err(EngineSessionError::from)
    }

    pub fn step_report(&mut self) -> Result<EngineStepReport, EngineSessionError> {
        if !self.uses_native_runtime() {
            let active_request_ids = self.compatibility_active_request_ids();
            if active_request_ids.is_empty() {
                return Ok(EngineStepReport::default());
            }
            let selected_backend = self.config.resolved_backend.selected_backend;
            let mut aggregate = EngineStepReport::default();
            let mut terminal_reports = Vec::new();

            for request_id in active_request_ids {
                let step = {
                    let slot = self
                        .compatibility_requests
                        .get_mut(&request_id)
                        .ok_or(EngineSessionError::MissingRequestSnapshot { request_id })?;
                    let CompatibilityLifecycleRequestSlot::Active(request) = slot else {
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
                self.store_terminal_compatibility_report(request_id, report);
            }

            return Ok(aggregate);
        }

        self.step().map(|outcome| {
            let metal_dispatch = outcome
                .runner_output
                .as_ref()
                .and_then(|_| self.core.last_metal_dispatch())
                .map(|trace| MetalDispatchStepReport::from_trace(&trace));
            EngineStepReport::from_native_outcome(&outcome, metal_dispatch)
        })
    }

    pub fn request_report(&self, request_id: u64) -> Option<SessionRequestReport> {
        if !self.uses_native_runtime() {
            return self
                .compatibility_requests
                .get(&request_id)
                .map(CompatibilityLifecycleRequestSlot::report);
        }
        let mut report: SessionRequestReport = self
            .core
            .request_manager()
            .snapshot(RequestId(request_id))
            .map(Into::into)?;
        if self.uses_native_placeholder_runtime() {
            annotate_native_placeholder_route_report(&mut report.route);
        }
        Some(report)
    }

    pub fn stream_request(
        &mut self,
        request_id: u64,
    ) -> Result<GenerateStream<'_>, EngineSessionError> {
        if !self.uses_native_runtime() {
            return Err(
                EngineSessionError::CompatibilityBackendDoesNotSupportLifecycle {
                    selected_backend: self.config.resolved_backend.selected_backend,
                    operation: "stream_request",
                },
            );
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
        if !self.uses_native_runtime() {
            return self.compatibility_submit_generate_with_request_id(request_id, request);
        }
        if request.input_text.is_some() {
            return Err(EngineSessionError::NativeBackendRequiresTokenizedInput);
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
        if !self.uses_native_runtime() {
            return Err(
                EngineSessionError::CompatibilityBackendDoesNotSupportLifecycle {
                    selected_backend: self.config.resolved_backend.selected_backend,
                    operation: "run_to_completion",
                },
            );
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
        if !self.uses_native_runtime() {
            return self.compatibility_generate_with_request_id(request_id, request);
        }
        let request_id = self.submit_generate_with_request_id(request_id, request)?;
        self.run_to_completion(request_id)
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
        if !self.uses_native_runtime() {
            return self.compatibility_stream_state_with_request_id(request_id, request);
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
            GenerateStreamState::Compatibility(state) => {
                self.next_compatibility_stream_event(state.as_mut())
            }
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

                if state.step_count > state.max_steps {
                    return Err(EngineSessionError::RequestDidNotTerminate {
                        request_id: state.request_id,
                        max_steps: state.max_steps,
                    });
                }

                let next_report = self.request_report(state.request_id).ok_or(
                    EngineSessionError::MissingRequestSnapshot {
                        request_id: state.request_id,
                    },
                )?;
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

    fn next_compatibility_stream_event(
        &mut self,
        state: &mut CompatibilityGenerateStreamState,
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
                                    response: GenerateResponse {
                                        request_id: state.request_id,
                                        model_id: state.current_report.model_id.clone(),
                                        prompt_tokens: state.current_report.prompt_tokens.clone(),
                                        prompt_text: state.prompt_text.clone(),
                                        output_tokens: state.current_report.output_tokens.clone(),
                                        output_token_logprobs: state
                                            .current_report
                                            .output_token_logprobs
                                            .clone(),
                                        output_text: Some(state.output_text.clone()),
                                        prompt_token_count: state.prompt_token_count,
                                        output_token_count: state.output_token_count,
                                        status: crate::generate::GenerateStatus::Finished,
                                        finish_reason: state.current_report.finish_reason,
                                        step_count: state.step_count,
                                        ttft_step: state.ttft_step,
                                        route: state.current_report.route.clone(),
                                        runtime: state.runtime.clone(),
                                    },
                                },
                            )));
                        }
                    }
                }

                let chunk = state.stream.next_chunk()?.ok_or(
                    EngineSessionError::CompatibilityStreamEndedBeforeStop {
                        request_id: state.request_id,
                        selected_backend: self.config.resolved_backend.selected_backend,
                    },
                )?;

                Ok(Some(state.step_event_from_chunk(chunk)))
            }
            GenerateStreamPhase::Done => Ok(None),
        }
    }
}

impl<'a> GenerateStream<'a> {
    fn new(session: &'a mut EngineSession, state: GenerateStreamState) -> Self {
        Self { session, state }
    }

    pub fn next_event(&mut self) -> Result<Option<GenerateStreamEvent>, EngineSessionError> {
        self.session.next_stream_event(&mut self.state)
    }

    pub fn into_response(mut self) -> Result<GenerateResponse, EngineSessionError> {
        while let Some(event) = self.next_event()? {
            if let GenerateStreamEvent::Response(event) = event {
                return Ok(event.response);
            }
        }

        Err(EngineSessionError::MissingRequestSnapshot {
            request_id: self.state.request_id(),
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
    fn new_native(
        request_id: u64,
        runtime: RuntimeReport,
        current_report: SessionRequestReport,
    ) -> Self {
        let max_steps = u64::from(current_report.prompt_len)
            + u64::from(current_report.max_output_tokens)
            + 256;

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

    fn request_id(&self) -> u64 {
        match self {
            Self::Native(state) => state.request_id,
            Self::Compatibility(state) => state.request_id,
        }
    }

    fn finish(&mut self) {
        match self {
            Self::Native(state) => state.phase = GenerateStreamPhase::Done,
            Self::Compatibility(state) => state.phase = GenerateStreamPhase::Done,
        }
    }
}

impl CompatibilityLifecycleRequestSlot {
    fn report(&self) -> SessionRequestReport {
        match self {
            Self::Active(request) => request.current_report.clone(),
            Self::Terminal(report) => report.as_ref().clone(),
        }
    }
}

impl CompatibilityLifecycleRequest {
    fn new(
        request_id: u64,
        mut current_report: SessionRequestReport,
        stream: CompatibilityStreamHandle,
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

    fn step_report(
        &mut self,
        selected_backend: SelectedBackend,
    ) -> Result<EngineStepReport, EngineSessionError> {
        let chunk = self.stream.next_chunk()?.ok_or(
            EngineSessionError::CompatibilityStreamEndedBeforeStop {
                request_id: self.request_id,
                selected_backend,
            },
        )?;
        Ok(self.apply_chunk(chunk))
    }

    /// Drain any remaining stream chunks after the stop signal to capture
    /// trailing usage data. Some OpenAI-compatible servers (including MLX)
    /// send a final chunk with `usage` but empty `choices` after the stop
    /// chunk. Without draining, that usage data is lost when the slot
    /// transitions to Terminal.
    fn drain_trailing_usage(&mut self) {
        loop {
            match self.stream.next_chunk() {
                Ok(Some(chunk)) => {
                    apply_compatibility_usage_counts(
                        &mut self.current_report,
                        &mut self.prompt_token_count,
                        &mut self.output_token_count,
                        &chunk,
                    );
                    apply_compatibility_prompt_progress(
                        &mut self.current_report,
                        chunk.prompt_progress.as_ref(),
                        &mut self.cached_prompt_tokens_observed,
                        &mut self.prefix_hit_recorded,
                    );
                }
                Ok(None) | Err(_) => break,
            }
        }
    }

    fn cancel(&mut self) -> SessionRequestReport {
        self.current_report.state = SessionRequestState::Cancelled;
        self.current_report.cancel_requested = true;
        self.current_report.finish_reason = Some(crate::generate::GenerateFinishReason::Cancelled);
        self.current_report.terminal_stop_reason = Some(ax_engine_core::StopReason::Cancelled);
        self.current_report.clone()
    }

    fn apply_chunk(&mut self, chunk: CompatibilityStreamChunk) -> EngineStepReport {
        self.step_count += 1;
        let prefix_hits = apply_compatibility_prompt_progress(
            &mut self.current_report,
            chunk.prompt_progress.as_ref(),
            &mut self.cached_prompt_tokens_observed,
            &mut self.prefix_hit_recorded,
        );
        apply_compatibility_usage_counts(
            &mut self.current_report,
            &mut self.prompt_token_count,
            &mut self.output_token_count,
            &chunk,
        );

        let delta_tokens = chunk.tokens;
        let delta_text = chunk.content;
        let was_terminal = is_terminal_request_state(self.current_report.state);
        let request_selected = chunk.prompt_progress.is_some()
            || !delta_tokens.is_empty()
            || !delta_text.is_empty()
            || chunk.stop;
        let ttft_events =
            if self.ttft_step.is_none() && (!delta_tokens.is_empty() || !delta_text.is_empty()) {
                self.ttft_step = Some(self.step_count);
                1
            } else {
                0
            };

        self.current_report
            .output_tokens
            .extend(delta_tokens.iter().copied());
        self.current_report
            .output_token_logprobs
            .extend(std::iter::repeat_n(None, delta_tokens.len()));
        self.current_report.output_len = self
            .current_report
            .output_len
            .max(self.current_report.output_tokens.len() as u32);
        let finish_reason = finish_reason_from_stop_type(chunk.stop_type.as_deref());
        if finish_reason.is_some() {
            self.current_report.finish_reason = finish_reason;
            self.current_report.terminal_stop_reason =
                terminal_stop_reason_from_finish_reason(finish_reason);
        }
        self.current_report.state = if chunk.stop || was_terminal {
            SessionRequestState::Finished
        } else if request_selected {
            SessionRequestState::Running
        } else {
            self.current_report.state
        };

        EngineStepReport {
            step_id: None,
            scheduled_requests: u32::from(request_selected),
            scheduled_tokens: delta_tokens.len() as u32,
            ttft_events,
            prefix_hits,
            kv_usage_blocks: 0,
            evictions: 0,
            cpu_time_us: 0,
            runner_time_us: 0,
            route: None,
            metal_dispatch: None,
        }
    }
}

impl CompatibilityGenerateStreamState {
    fn new(
        request_id: u64,
        runtime: RuntimeReport,
        mut current_report: SessionRequestReport,
        prompt_text: Option<String>,
        stream: CompatibilityStreamHandle,
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

    fn step_event_from_chunk(&mut self, chunk: CompatibilityStreamChunk) -> GenerateStreamEvent {
        self.step_count += 1;
        let prefix_hits = apply_compatibility_prompt_progress(
            &mut self.current_report,
            chunk.prompt_progress.as_ref(),
            &mut self.cached_prompt_tokens_observed,
            &mut self.prefix_hit_recorded,
        );
        apply_compatibility_usage_counts(
            &mut self.current_report,
            &mut self.prompt_token_count,
            &mut self.output_token_count,
            &chunk,
        );

        let delta_tokens = chunk.tokens;
        let delta_text = chunk.content;
        let was_terminal = is_terminal_request_state(self.current_report.state);
        let request_selected = chunk.prompt_progress.is_some()
            || !delta_tokens.is_empty()
            || !delta_text.is_empty()
            || chunk.stop;
        let ttft_events =
            if self.ttft_step.is_none() && (!delta_tokens.is_empty() || !delta_text.is_empty()) {
                self.ttft_step = Some(self.step_count);
                1
            } else {
                0
            };

        self.current_report
            .output_tokens
            .extend(delta_tokens.iter().copied());
        self.current_report
            .output_token_logprobs
            .extend(std::iter::repeat_n(None, delta_tokens.len()));
        self.current_report.output_len = self
            .current_report
            .output_len
            .max(self.current_report.output_tokens.len() as u32);
        self.output_text.push_str(&delta_text);
        let finish_reason = finish_reason_from_stop_type(chunk.stop_type.as_deref());
        if finish_reason.is_some() {
            self.current_report.finish_reason = finish_reason;
            self.current_report.terminal_stop_reason =
                terminal_stop_reason_from_finish_reason(finish_reason);
        }
        self.current_report.state = if chunk.stop || was_terminal {
            SessionRequestState::Finished
        } else if request_selected {
            SessionRequestState::Running
        } else {
            self.current_report.state
        };

        if chunk.stop {
            self.terminal_chunk_seen = true;
        }

        let delta_token_logprobs = vec![None; delta_tokens.len()];

        GenerateStreamEvent::Step(GenerateStreamStepEvent {
            request: self.current_report.clone(),
            step: EngineStepReport {
                step_id: None,
                scheduled_requests: u32::from(request_selected),
                scheduled_tokens: delta_tokens.len() as u32,
                ttft_events,
                prefix_hits,
                kv_usage_blocks: 0,
                evictions: 0,
                cpu_time_us: 0,
                runner_time_us: 0,
                route: None,
                metal_dispatch: None,
            },
            delta_tokens,
            delta_token_logprobs,
            delta_text: if delta_text.is_empty() {
                None
            } else {
                Some(delta_text)
            },
        })
    }
}

fn slice_output_token_logprobs(
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

fn is_terminal_request_state(state: SessionRequestState) -> bool {
    matches!(
        state,
        SessionRequestState::Finished
            | SessionRequestState::Cancelled
            | SessionRequestState::Failed
    )
}

fn compatibility_stream_route(selected_backend: SelectedBackend) -> GenerateRouteReport {
    let execution_plan = match selected_backend {
        SelectedBackend::LlamaCpp => COMPATIBILITY_STREAM_EXECUTION_PLAN,
        SelectedBackend::Vllm => "compatibility.vllm.server_completions_stream",
        SelectedBackend::MistralRs => "compatibility.mistral_rs.server_completions_stream",
        SelectedBackend::Mlx => "compatibility.mlx.server_completions_stream",
        _ => COMPATIBILITY_STREAM_EXECUTION_PLAN,
    };

    GenerateRouteReport {
        execution_plan: Some(execution_plan.to_string()),
        attention_route: None,
        kv_mode: None,
        prefix_cache_path: None,
        barrier_mode: None,
        crossover_decisions: Default::default(),
    }
}

fn apply_compatibility_prompt_progress(
    report: &mut SessionRequestReport,
    prompt_progress: Option<&CompatibilityPromptProgress>,
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

fn apply_compatibility_usage_counts(
    report: &mut SessionRequestReport,
    prompt_token_count: &mut Option<u32>,
    output_token_count: &mut Option<u32>,
    chunk: &CompatibilityStreamChunk,
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

fn finish_reason_from_stop_type(
    stop_type: Option<&str>,
) -> Option<crate::generate::GenerateFinishReason> {
    match stop_type {
        // llama.cpp server completion format
        Some("limit") => Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
        Some("eos") => Some(crate::generate::GenerateFinishReason::Stop),
        // OpenAI-compatible format (vLLM, mistral.rs, mlx)
        Some("length") => Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
        Some("stop") => Some(crate::generate::GenerateFinishReason::Stop),
        None => None,
        // Unknown stop types (e.g. "content_filter"): default to MaxOutputTokens so the
        // request is marked complete with a non-null finish reason.
        Some(_) => Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
    }
}

fn terminal_stop_reason_from_finish_reason(
    finish_reason: Option<crate::generate::GenerateFinishReason>,
) -> Option<ax_engine_core::StopReason> {
    match finish_reason {
        Some(crate::generate::GenerateFinishReason::Stop) => {
            Some(ax_engine_core::StopReason::EosToken)
        }
        Some(crate::generate::GenerateFinishReason::MaxOutputTokens) => {
            Some(ax_engine_core::StopReason::MaxOutputTokens)
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

#[derive(Debug, Error)]
pub enum EngineSessionError {
    #[error(transparent)]
    BackendContract(#[from] BackendContractError),
    #[error(transparent)]
    Compatibility(#[from] CompatibilityBackendError),
    #[error("max_batch_tokens must be greater than zero")]
    InvalidMaxBatchTokens,
    #[error("generate request requires input_tokens or input_text")]
    EmptyInputTokens,
    #[error("generate request max_output_tokens must be greater than zero")]
    InvalidMaxOutputTokens,
    #[error("native preview session only accepts pre-tokenized input_tokens; input_text requires a compatibility backend")]
    NativeBackendRequiresTokenizedInput,
    #[error(
        "ax_native requires validated Metal runtime artifacts; deterministic native fallback is internal-only and must be explicitly enabled"
    )]
    NativeRuntimeArtifactsRequired,
    #[error("request_id must be greater than zero")]
    InvalidRequestId,
    #[error("unsupported support tier cannot start an engine session")]
    UnsupportedSupportTier,
    #[error(
        "AX Engine v4 requires Apple M4-or-newer CPU/GPU; detected unsupported host {detected_host}. Set AX_ALLOW_UNSUPPORTED_HOST=1 only for internal development or CI bring-up."
    )]
    UnsupportedHostHardware { detected_host: String },
    #[error("compatibility backend {selected_backend:?} requires compatibility_backend config")]
    MissingCompatibilityBackendConfig { selected_backend: SelectedBackend },
    #[error(
        "compatibility backend config targets {configured_backend:?}, but session resolved {selected_backend:?}"
    )]
    CompatibilityBackendConfigMismatch {
        configured_backend: SelectedBackend,
        selected_backend: SelectedBackend,
    },
    #[error(
        "compatibility backend {selected_backend:?} does not support {operation} in the preview SDK session yet"
    )]
    CompatibilityBackendDoesNotSupportLifecycle {
        selected_backend: SelectedBackend,
        operation: &'static str,
    },
    #[error(
        "compatibility stream for request {request_id} on backend {selected_backend:?} ended before a terminal stop marker"
    )]
    CompatibilityStreamEndedBeforeStop {
        request_id: u64,
        selected_backend: SelectedBackend,
    },
    #[error("request {request_id} is missing from session state")]
    MissingRequestSnapshot { request_id: u64 },
    #[error("request {request_id} did not terminate within {max_steps} steps")]
    RequestDidNotTerminate { request_id: u64, max_steps: u64 },
    #[error("request {request_id} violated session report invariants: {message}")]
    RequestReportInvariantViolation {
        request_id: u64,
        message: &'static str,
    },
    #[error(transparent)]
    Core(#[from] EngineCoreError),
    #[error(transparent)]
    MetalRuntime(#[from] MetalRuntimeError),
}

fn build_native_core(config: &EngineSessionConfig) -> Result<EngineCore, EngineSessionError> {
    if config.resolved_backend.selected_backend != SelectedBackend::AxNative {
        return Ok(EngineCore::with_kv_config(config.kv_config));
    }

    if let Some(build_dir) = config.native_runtime_artifacts_dir() {
        return EngineCore::with_metal_bringup_runner_and_model_artifacts(
            config.kv_config,
            build_dir,
            config.native_model_artifacts_dir(),
        )
        .map_err(EngineSessionError::from);
    }

    if config.allow_deterministic_native_fallback() {
        Ok(EngineCore::with_runtime_components(
            config.kv_config,
            NativePlaceholderRunner,
            DeterministicSampler,
        ))
    } else {
        Err(EngineSessionError::NativeRuntimeArtifactsRequired)
    }
}

fn resolve_native_model_report(
    config: &EngineSessionConfig,
    core: &EngineCore,
) -> Option<NativeModelReport> {
    let source = config.native_model_artifacts_source()?;
    let summary = core.native_model_artifacts_summary()?;
    let binding = core.native_model_binding_summary();
    Some(NativeModelReport::from_summary(source, summary, binding))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use ax_engine_core::metal::{
        MetalBuildOutputs, MetalDispatchExecutionInfo, MetalDispatchKvMetadata,
        MetalDispatchNumericTrace, MetalDispatchRuntimeInfo, MetalNumericValidationSummary,
        PHASE1_DEFAULT_BLOCK_SIZE_TOKENS, PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS,
    };
    use ax_engine_core::{
        DeterministicSampler, ExecutionRunner, ExecutionStatus, KvWriteSummary,
        MetalBinaryArchiveInfo, MetalBinaryArchiveState, MetalBuildDoctorReport,
        MetalBuildHostReport, MetalBuildReport, MetalBuildStatus, MetalBuildToolStatus,
        MetalBuildToolchainReport, MetalCommandBufferStatus, MetalDispatchArenaInfo,
        MetalDispatchKernelTrace, MetalDispatchTrace, MetalDispatchWorkload, MetalKernelManifest,
        MetalKernelSpec, MetalKernelTier, MetalThreadgroupSize, ModelId, NativeModelArtifacts,
        NativeModelArtifactsSummary, NativeModelManifest, NativeTensorDataType, NativeTensorFormat,
        NativeTensorRole, NativeTensorSpec, RequestExecutionUpdate, RequestState, RunnerInput,
        RunnerOutput, SamplingParams, SequenceNo, AX_NATIVE_MODEL_MANIFEST_FILE,
        AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, PHASE1_METAL_BUILD_GATE,
        PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION, PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION,
        PHASE1_METAL_LANGUAGE_STANDARD, PHASE1_METAL_LIBRARY_NAME, PHASE1_METAL_NATIVE_TARGET,
    };
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::generate::{GenerateFinishReason, GenerateStatus};

    fn sample_submission() -> RequestSubmission {
        RequestSubmission {
            request_id: RequestId(1),
            model_id: ModelId("qwen3_dense".into()),
            input_tokens: vec![1, 2, 3, 4],
            sampling_params: SamplingParams::default(),
            max_output_tokens: 2,
            arrival_sequence: SequenceNo(1),
            metadata: None,
        }
    }

    fn native_placeholder_session_config() -> EngineSessionConfig {
        EngineSessionConfig {
            allow_deterministic_native_fallback: true,
            // Explicitly clear auto-detected repo artifacts so the test
            // exercises the deterministic placeholder path regardless of
            // working directory.
            native_runtime_artifacts_dir: None,
            native_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        }
    }

    fn sample_terminal_compatibility_report(request_id: u64) -> SessionRequestReport {
        SessionRequestReport {
            request_id,
            model_id: "qwen3_dense".to_string(),
            state: SessionRequestState::Finished,
            prompt_tokens: vec![1, 2, 3],
            processed_prompt_tokens: 3,
            output_tokens: vec![4, 5],
            output_token_logprobs: vec![Some(-0.25), Some(-0.5)],
            prompt_len: 3,
            output_len: 2,
            max_output_tokens: 2,
            cancel_requested: false,
            execution_plan_ref: Some(COMPATIBILITY_STREAM_EXECUTION_PLAN.to_string()),
            route: compatibility_stream_route(SelectedBackend::LlamaCpp),
            finish_reason: Some(GenerateFinishReason::Stop),
            terminal_stop_reason: None,
            last_error: None,
        }
    }

    #[derive(Clone, Debug)]
    struct TraceReportingRunner {
        trace: MetalDispatchTrace,
    }

    #[derive(Clone, Debug)]
    struct NativeModelReportingRunner {
        summary: NativeModelArtifactsSummary,
        binding: Option<ax_engine_core::NativeModelBindingSummary>,
    }

    impl ExecutionRunner for TraceReportingRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: None,
                    stop_reason: None,
                    error: None,
                })
                .collect::<Vec<_>>();
            let mut route_metadata = input.execution_batch.route_metadata.clone();
            route_metadata.attention_route = Some("mock_native_attention".to_string());
            route_metadata
                .crossover_decisions
                .push(("metal_dispatch_completed".to_string(), 1));

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles: Vec::new(),
                logits_outputs: Vec::new(),
                kv_write_summary: KvWriteSummary {
                    tokens_written: input.execution_batch.total_scheduled_tokens,
                    blocks_touched: input
                        .block_tables
                        .iter()
                        .map(|resolved| resolved.block_table.block_ids.len() as u32)
                        .sum(),
                },
                route_metadata,
                execution_status: ExecutionStatus::Success,
            }
        }

        fn metal_dispatch_trace(&self) -> Option<MetalDispatchTrace> {
            Some(self.trace.clone())
        }
    }

    impl ExecutionRunner for NativeModelReportingRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let blocks_touched = input
                .block_tables
                .iter()
                .map(|resolved| resolved.block_table.block_ids.len() as u32)
                .sum();
            let logits_handles = input
                .execution_batch
                .items
                .iter()
                .filter(|item| matches!(item.mode, ax_engine_core::ExecutionMode::Decode))
                .map(|item| item.request_id)
                .collect();
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: None,
                    stop_reason: None,
                    error: None,
                })
                .collect();

            RunnerOutput {
                step_id: input.execution_batch.step_id,
                request_updates,
                logits_handles,
                logits_outputs: Vec::new(),
                kv_write_summary: KvWriteSummary {
                    tokens_written: input.execution_batch.total_scheduled_tokens,
                    blocks_touched,
                },
                route_metadata: input.execution_batch.route_metadata.clone(),
                execution_status: ExecutionStatus::Success,
            }
        }

        fn native_model_artifacts_summary(&self) -> Option<NativeModelArtifactsSummary> {
            Some(self.summary.clone())
        }

        fn native_model_binding_summary(
            &self,
        ) -> Option<ax_engine_core::NativeModelBindingSummary> {
            self.binding
        }
    }

    fn sample_metal_dispatch_trace() -> MetalDispatchTrace {
        MetalDispatchTrace {
            command_queue_label: "ax.queue".to_string(),
            command_buffer_label: "ax.buffer".to_string(),
            command_buffer_status: MetalCommandBufferStatus::Completed,
            runtime: MetalDispatchRuntimeInfo {
                device_name: "Apple M4 Max".to_string(),
                required_pipeline_count: 4,
                max_thread_execution_width: 64,
                binary_archive: MetalBinaryArchiveInfo {
                    path: std::path::PathBuf::from(
                        "/tmp/ax_phase1_dense_path.binary_archive.metallib",
                    ),
                    state: MetalBinaryArchiveState::Loaded,
                    attached_pipeline_count: 4,
                    serialized: true,
                    note: None,
                },
                command_queue_ready: true,
                model_conditioned_inputs: true,
                real_model_tensor_inputs: true,
                complete_model_forward_supported: true,
                model_bindings_prepared: true,
                model_buffers_bound: true,
                model_buffer_count: 12,
                model_buffer_bytes: 4096,
                native_dense_kernel_coverage: Default::default(),
                model: Some(NativeModelArtifactsSummary {
                    model_family: "qwen3_dense".to_string(),
                    tensor_format: NativeTensorFormat::Safetensors,
                    layer_count: 36,
                    tensor_count: 512,
                    tie_word_embeddings: false,
                }),
            },
            workload: MetalDispatchWorkload {
                scheduled_requests: 1,
                prefill_requests: 1,
                decode_requests: 0,
                scheduled_tokens: 2,
                scheduled_token_ids: vec![10, 11],
                scheduled_positions: vec![0, 1],
                resolved_blocks: 1,
                token_elements: 2,
                block_elements: 16,
                scratch_elements: 32,
                kv_slot_capacity: 16,
                kv_block_capacity: 1,
                numeric_layout: ax_engine_core::MetalDispatchNumericLayout::default(),
                kv_metadata: MetalDispatchKvMetadata {
                    block_size_tokens: 16,
                    slot_mapping: vec![0, 1],
                    attention_block_table: vec![0, 1],
                    gather_block_table: vec![0],
                    gather_block_table_stride: 1,
                    copy_block_mapping: vec![[0, 0]],
                    seq_lens: vec![2],
                    cu_seq_lens: vec![0, 2],
                    scheduled_cu_seq_lens: vec![0, 2],
                },
            },
            arena: MetalDispatchArenaInfo {
                token_capacity: 8,
                slot_capacity: 64,
                attention_ref_capacity: 8,
                gather_ref_capacity: 8,
                gather_output_capacity: 8,
                copy_pair_capacity: 4,
                sequence_capacity: 4,
                reused_existing: true,
                grew_existing: false,
            },
            execution: MetalDispatchExecutionInfo {
                direct_decode_token_count: 1,
                direct_decode_checksum_lo: 0x1234,
                logits_output_count: 1,
                remaining_logits_handle_count: 0,
                model_bound_ffn_decode: true,
                real_model_forward_completed: true,
                prefix_native_dispatch_count: 35,
                prefix_cpu_reference_dispatch_count: 1,
                qkv_projection_token_count: 72,
                layer_continuation_token_count: 37,
                logits_projection_token_count: 1,
                logits_vocab_scan_row_count: 151936,
                direct_decode_native_projection_row_count: 0,
                direct_decode_cpu_projection_row_count: 0,
                direct_decode_native_rms_norm_element_count: 0,
                direct_decode_cpu_rms_norm_element_count: 0,
                prefix_native_projection_row_count: 0,
                prefix_cpu_projection_row_count: 0,
                prefix_native_rms_norm_element_count: 0,
                prefix_cpu_rms_norm_element_count: 0,
                direct_decode_native_ffn_activation_element_count: 0,
                direct_decode_cpu_ffn_activation_element_count: 0,
                prefix_native_ffn_activation_element_count: 0,
                prefix_cpu_ffn_activation_element_count: 0,
                direct_decode_batched_logits_group_count: 0,
                direct_decode_batched_logits_token_count: 0,
                direct_decode_batched_group_fallback_count: 0,
                direct_decode_batched_group_fallback_token_count: 0,
            },
            kernels: vec![
                MetalDispatchKernelTrace {
                    function_name: "reshape_and_cache".to_string(),
                    element_count: 32,
                    threads_per_grid: MetalThreadgroupSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                    threads_per_threadgroup: MetalThreadgroupSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                },
                MetalDispatchKernelTrace {
                    function_name: "paged_decode_attention".to_string(),
                    element_count: 16,
                    threads_per_grid: MetalThreadgroupSize {
                        width: 16,
                        height: 1,
                        depth: 1,
                    },
                    threads_per_threadgroup: MetalThreadgroupSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                },
            ],
            numeric: MetalDispatchNumericTrace {
                attention_output_bits: vec![0],
                key_cache_checksum: 0x11,
                attention_output_checksum: 0x22,
                gather_output_checksum: 0x33,
                copy_output_checksum: 0x44,
                validation: Some(MetalNumericValidationSummary {
                    expected_key_cache_checksum: 0x11,
                    expected_attention_output_checksum: 0x22,
                    expected_gather_output_checksum: 0x33,
                    expected_copy_output_checksum: 0x44,
                    attention_max_abs_diff_microunits: 0,
                }),
            },
        }
    }

    fn unique_test_dir(label: &str) -> std::path::PathBuf {
        static NEXT_SUFFIX: AtomicU64 = AtomicU64::new(0);
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let suffix = NEXT_SUFFIX.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ax-engine-session-{label}-{unique}-{suffix}"))
    }

    fn write_repo_owned_native_runtime_status_fixture(
        status: &str,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        let repo_root = unique_test_dir("metal-runtime-fixture");
        let manifest_path = repo_root.join("metal/phase1-kernels.json");
        let build_dir = repo_root.join("build/metal");
        let build_report_path = build_dir.join("build_report.json");

        fs::create_dir_all(
            manifest_path
                .parent()
                .expect("manifest parent should exist for fixture"),
        )
        .expect("fixture manifest directory should create");
        fs::create_dir_all(&build_dir).expect("fixture build directory should create");
        fs::write(&manifest_path, "{}").expect("fixture manifest should write");
        fs::write(
            &build_report_path,
            serde_json::json!({ "status": status }).to_string(),
        )
        .expect("fixture build report should write");

        (repo_root, build_dir)
    }

    fn write_valid_repo_owned_native_runtime_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
        let repo_root = unique_test_dir("metal-runtime-valid-fixture");
        let manifest_path = repo_root.join("metal/phase1-kernels.json");
        let source_path = repo_root.join("metal/kernels/phase1_dense_path.metal");
        let build_dir = repo_root.join("build/metal");
        let air_path = build_dir.join("ax_phase1_dense_path.air");
        let metalar_path = build_dir.join("ax_phase1_dense_path.metalar");
        let metallib_path = build_dir.join("ax_phase1_dense_path.metallib");

        fs::create_dir_all(
            manifest_path
                .parent()
                .expect("manifest parent should exist for fixture"),
        )
        .expect("fixture manifest directory should create");
        fs::create_dir_all(
            source_path
                .parent()
                .expect("source parent should exist for fixture"),
        )
        .expect("fixture source directory should create");
        fs::create_dir_all(&build_dir).expect("fixture build directory should create");

        let source_text = phase1_source_text();
        fs::write(&source_path, source_text.as_bytes()).expect("fixture source should write");
        fs::write(&air_path, b"fake-air").expect("fixture air should write");
        fs::write(&metalar_path, b"fake-metalar").expect("fixture metalar should write");
        fs::write(&metallib_path, b"fake-metallib").expect("fixture metallib should write");

        let manifest = MetalKernelManifest {
            schema_version: PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION.to_string(),
            native_target: PHASE1_METAL_NATIVE_TARGET.to_string(),
            metal_language_standard: PHASE1_METAL_LANGUAGE_STANDARD.to_string(),
            library_name: PHASE1_METAL_LIBRARY_NAME.to_string(),
            default_block_size_tokens: PHASE1_DEFAULT_BLOCK_SIZE_TOKENS,
            supported_block_size_tokens: PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS.to_vec(),
            source_file: std::path::PathBuf::from("metal/kernels/phase1_dense_path.metal"),
            toolchain_requirements: ["xcrun metal", "xcrun metallib", "xcrun metal-ar"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            build_gate: PHASE1_METAL_BUILD_GATE.to_string(),
            kernels: phase1_kernel_specs(),
        };
        write_json_file(&manifest_path, &manifest);

        let build_report = MetalBuildReport {
            schema_version: PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION.to_string(),
            manifest_path: manifest_path.clone(),
            source_file: source_path.clone(),
            native_target: manifest.native_target.clone(),
            metal_language_standard: manifest.metal_language_standard.clone(),
            library_name: manifest.library_name.clone(),
            default_block_size_tokens: manifest.default_block_size_tokens,
            supported_block_size_tokens: manifest.supported_block_size_tokens.clone(),
            toolchain_requirements: manifest.toolchain_requirements.clone(),
            doctor: sample_build_doctor(true, true),
            kernels: manifest.kernels.clone(),
            source_sha256: sha256_hex(source_text.as_bytes()),
            outputs: MetalBuildOutputs {
                air: Some(air_path.clone()),
                metalar: Some(metalar_path.clone()),
                metallib: Some(metallib_path.clone()),
                air_sha256: Some(sha256_hex(b"fake-air")),
                metalar_sha256: Some(sha256_hex(b"fake-metalar")),
                metallib_sha256: Some(sha256_hex(b"fake-metallib")),
            },
            compile_commands: vec![
                vec!["xcrun".to_string(), "metal".to_string()],
                vec!["xcrun".to_string(), "metal-ar".to_string()],
                vec!["xcrun".to_string(), "metallib".to_string()],
            ],
            status: MetalBuildStatus::Compiled,
            reason: None,
        };
        write_json_file(&build_dir.join("build_report.json"), &build_report);

        (repo_root, build_dir)
    }

    fn unique_missing_metal_build_dir() -> std::path::PathBuf {
        unique_test_dir("missing-metal-build")
    }

    fn write_valid_native_model_fixture() -> std::path::PathBuf {
        let root_dir = unique_test_dir("native-model-valid-fixture");
        fs::create_dir_all(&root_dir).expect("native model fixture directory should create");
        fs::write(root_dir.join("model.safetensors"), vec![0_u8; 4096])
            .expect("native model weights should write");

        let manifest = NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            layer_count: 1,
            hidden_size: 2048,
            attention_head_count: 16,
            attention_head_dim: 128,
            kv_head_count: 8,
            vocab_size: 151936,
            tie_word_embeddings: false,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            tensors: vec![
                native_model_tensor(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    vec![151_936, 2_048],
                ),
                native_model_tensor(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    vec![2_048],
                ),
                native_model_tensor(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    vec![151_936, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![2_048],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.qkv_proj.weight",
                    NativeTensorRole::AttentionQkvPacked,
                    Some(0),
                    vec![4_096, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    vec![2_048, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![2_048],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(0),
                    vec![8_192, 2_048],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    vec![2_048, 4_096],
                ),
            ],
        };
        write_json_file(&root_dir.join(AX_NATIVE_MODEL_MANIFEST_FILE), &manifest);

        root_dir
    }

    fn native_model_tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F16,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    fn fake_compat_script() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-session-compat-{unique}.py"));
        let script = r#"#!/usr/bin/env python3
from __future__ import annotations

import sys

args = sys.argv[1:]
prompt = args[args.index("--prompt") + 1]
sys.stdout.write(f"session::{prompt}")
"#;

        fs::write(&path, script).expect("fake script should be written");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let mut permissions = fs::metadata(&path)
                .expect("script metadata should exist")
                .permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&path, permissions).expect("script should be executable");
        }
        path
    }

    #[test]
    fn native_session_uses_sdk_owned_metal_artifact_selection() {
        let missing_build_dir = unique_missing_metal_build_dir();

        let error = EngineSession::new(EngineSessionConfig {
            native_runtime_artifacts_dir: Some(missing_build_dir),
            native_runtime_artifacts_source: Some(NativeRuntimeArtifactsSource::ExplicitConfig),
            ..EngineSessionConfig::default()
        })
        .expect_err("explicit native artifact dir should route through Metal bring-up");

        let EngineSessionError::MetalRuntime(MetalRuntimeError::ReadJson { .. }) = error else {
            panic!("expected metal runtime read-json error");
        };
    }

    #[test]
    fn native_session_requires_validated_artifacts_without_explicit_placeholder_opt_in() {
        let error = EngineSession::new(EngineSessionConfig {
            // Explicitly clear auto-detected repo artifacts so the test
            // exercises the "no artifacts" path regardless of working directory.
            native_runtime_artifacts_dir: None,
            native_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        })
        .expect_err("native session should fail closed");

        assert!(matches!(
            error,
            EngineSessionError::NativeRuntimeArtifactsRequired
        ));
    }

    #[test]
    fn default_native_runtime_artifacts_dir_prefers_explicit_env_path() {
        let (repo_root, _build_dir) = write_valid_repo_owned_native_runtime_fixture();
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");
        let explicit_dir = unique_test_dir("explicit-metal-build-dir");

        let detected = resolve_default_native_runtime_artifacts_selection(
            Some(explicit_dir.clone()),
            Some(&nested_dir),
        );

        assert_eq!(
            detected.as_ref().map(|selection| selection.dir.as_path()),
            Some(explicit_dir.as_path())
        );
        assert_eq!(
            detected.as_ref().map(|selection| selection.source),
            Some(NativeRuntimeArtifactsSource::ExplicitEnv)
        );
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn default_native_runtime_artifacts_dir_detects_validated_repo_fixture_from_nested_dir() {
        let (repo_root, build_dir) = write_valid_repo_owned_native_runtime_fixture();
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");

        let detected = resolve_default_native_runtime_artifacts_selection(None, Some(&nested_dir));

        assert_eq!(
            detected.as_ref().map(|selection| selection.dir.as_path()),
            Some(build_dir.as_path())
        );
        assert_eq!(
            detected.as_ref().map(|selection| selection.source),
            Some(NativeRuntimeArtifactsSource::RepoAutoDetect)
        );
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn default_native_runtime_artifacts_dir_ignores_non_compiled_repo_fixture() {
        let (repo_root, _) =
            write_repo_owned_native_runtime_status_fixture("skipped_toolchain_unavailable");
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");

        let detected = resolve_default_native_runtime_artifacts_selection(None, Some(&nested_dir));

        assert_eq!(detected, None);
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn default_native_runtime_artifacts_dir_ignores_invalid_compiled_repo_fixture() {
        let (repo_root, _) = write_repo_owned_native_runtime_status_fixture("compiled");
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");

        let detected = resolve_default_native_runtime_artifacts_selection(None, Some(&nested_dir));

        assert_eq!(detected, None);
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn native_runtime_report_marks_deterministic_runner_when_placeholder_fallback_is_explicit() {
        let report = EngineSessionConfig {
            allow_deterministic_native_fallback: true,
            native_runtime_artifacts_dir: None,
            native_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        }
        .runtime_report();

        assert_eq!(
            report.native_runtime,
            Some(NativeRuntimeReport::deterministic())
        );
    }

    #[test]
    fn native_runtime_report_omits_runner_without_artifacts_when_placeholder_fallback_is_disabled()
    {
        let report = EngineSessionConfig {
            native_runtime_artifacts_dir: None,
            native_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        }
        .runtime_report();

        assert_eq!(report.native_runtime, None);
    }

    fn phase1_source_text() -> &'static str {
        r#"
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
kernel void kv_scale_update() {}
"#
    }

    fn phase1_kernel_specs() -> Vec<MetalKernelSpec> {
        vec![
            MetalKernelSpec {
                name: "reshape_and_cache".to_string(),
                tier: MetalKernelTier::Required,
                purpose: "paged KV writes".to_string(),
            },
            MetalKernelSpec {
                name: "paged_decode_attention".to_string(),
                tier: MetalKernelTier::Required,
                purpose: "decode attention".to_string(),
            },
            MetalKernelSpec {
                name: "gather_kv_cache".to_string(),
                tier: MetalKernelTier::Required,
                purpose: "KV gather".to_string(),
            },
            MetalKernelSpec {
                name: "copy_blocks".to_string(),
                tier: MetalKernelTier::Required,
                purpose: "block copy".to_string(),
            },
            MetalKernelSpec {
                name: "swap_blocks".to_string(),
                tier: MetalKernelTier::Deferred,
                purpose: "future block swap".to_string(),
            },
            MetalKernelSpec {
                name: "kv_scale_update".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "quantized KV scaling".to_string(),
            },
        ]
    }

    fn write_json_file<T: serde::Serialize>(path: &Path, value: &T) {
        let json = serde_json::to_vec_pretty(value).expect("json should serialize");
        fs::write(path, json).expect("json file should write");
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn sample_build_doctor(
        metal_toolchain_fully_available: bool,
        bringup_allowed: bool,
    ) -> MetalBuildDoctorReport {
        MetalBuildDoctorReport {
            status: if metal_toolchain_fully_available && bringup_allowed {
                "ready".to_string()
            } else if bringup_allowed {
                "bringup_only".to_string()
            } else {
                "not_ready".to_string()
            },
            bringup_allowed,
            native_runtime_ready: metal_toolchain_fully_available && bringup_allowed,
            metal_toolchain_fully_available,
            host: MetalBuildHostReport {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                detected_soc: Some("Apple M4 Max".to_string()),
                supported_native_runtime: true,
                unsupported_host_override_active: false,
            },
            metal_toolchain: MetalBuildToolchainReport {
                fully_available: metal_toolchain_fully_available,
                metal: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("Apple metal version 36000.4".to_string()),
                },
                metallib: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("Apple metallib version 36000.4".to_string()),
                },
                metal_ar: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("Apple metal-ar version 36000.4".to_string()),
                },
            },
        }
    }

    #[test]
    fn native_runtime_report_marks_explicit_config_metal_runner() {
        let report = EngineSessionConfig {
            native_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
            native_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        }
        .runtime_report();

        assert_eq!(
            report.native_runtime,
            Some(NativeRuntimeReport::metal_bringup(
                NativeRuntimeArtifactsSource::ExplicitConfig,
            ))
        );
    }

    #[test]
    fn build_native_core_rejects_invalid_native_model_artifacts_before_metal_bringup() {
        let (_repo_root, build_dir) = write_valid_repo_owned_native_runtime_fixture();
        let missing_model_dir = unique_test_dir("missing-native-model");
        let config = EngineSessionConfig {
            native_runtime_artifacts_dir: Some(build_dir.clone()),
            native_runtime_artifacts_source: Some(NativeRuntimeArtifactsSource::ExplicitConfig),
            native_model_artifacts_dir: Some(missing_model_dir),
            native_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
            ..EngineSessionConfig::default()
        };

        let error =
            build_native_core(&config).expect_err("invalid native model manifest should fail");

        match error {
            EngineSessionError::MetalRuntime(MetalRuntimeError::NativeModel(_)) => {}
            other => panic!("expected native model validation error, got {other:?}"),
        }

        let _ = fs::remove_dir_all(
            build_dir
                .parent()
                .and_then(Path::parent)
                .unwrap_or(&build_dir),
        );
    }

    #[test]
    fn resolve_native_model_report_uses_runner_owned_validated_summary() {
        let config = EngineSessionConfig {
            native_model_artifacts_dir: Some(PathBuf::from("/tmp/ax-model")),
            native_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
            ..EngineSessionConfig::default()
        };
        let model_dir = write_valid_native_model_fixture();
        let summary = NativeModelArtifacts::from_dir(&model_dir)
            .expect("native model fixture should validate")
            .summary();
        let core = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(0), 16, 64),
            NativeModelReportingRunner {
                summary,
                binding: Some(ax_engine_core::NativeModelBindingSummary {
                    bindings_prepared: true,
                    buffers_bound: true,
                    buffer_count: 9,
                    buffer_bytes: 288,
                }),
            },
            DeterministicSampler,
        );

        let report =
            resolve_native_model_report(&config, &core).expect("native model report should build");

        assert_eq!(
            report,
            NativeModelReport {
                artifacts_source: NativeModelArtifactsSource::ExplicitConfig,
                model_family: "qwen3_dense".to_string(),
                tensor_format: NativeTensorFormat::Safetensors,
                layer_count: 1,
                tensor_count: 9,
                tie_word_embeddings: false,
                bindings_prepared: true,
                buffers_bound: true,
                buffer_count: 9,
                buffer_bytes: 288,
            }
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn preview_session_config_factory_builds_native_preview_defaults() {
        let config =
            EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest::default())
                .expect("preview config factory should build native defaults");

        assert_eq!(
            config.kv_config,
            KvManagerConfig::new(CacheGroupId(0), 16, 1024)
        );
        assert!(config.deterministic);
        assert_eq!(config.max_batch_tokens, 2048);
        assert_eq!(config.backend_policy, BackendPolicy::strict_native());
        assert_eq!(config.resolved_backend, ResolvedBackend::native_preview());
        assert!(config.compatibility_backend.is_none());
    }

    #[test]
    fn preview_session_config_factory_builds_compatibility_server_backend() {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::Compatibility,
                compat_server_url: Some("http://127.0.0.1:8080".to_string()),
                ..PreviewBackendRequest::default()
            },
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview config factory should build compatibility config");

        assert_eq!(config.backend_policy, BackendPolicy::allow_compat());
        assert_eq!(
            config.resolved_backend.support_tier,
            SupportTier::Compatibility
        );
        assert_eq!(
            config.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert!(config.compatibility_backend.is_some());
    }

    #[test]
    fn runtime_report_marks_mlx_cli_fallback_as_non_streaming() {
        let report = EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::Mlx,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::Mlx(
                crate::compat::MlxConfig::cli("python3", "/tmp/mlx-model"),
            )),
            ..EngineSessionConfig::default()
        }
        .runtime_report();

        assert_eq!(report.selected_backend, SelectedBackend::Mlx);
        assert!(!report.capabilities.token_streaming);
        assert!(report.capabilities.text_generation);
    }

    #[test]
    fn runtime_report_marks_mlx_server_route_as_streaming_capable() {
        let report = EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::Mlx,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::Mlx(
                crate::compat::MlxConfig::server_completions("http://127.0.0.1:8082"),
            )),
            ..EngineSessionConfig::default()
        }
        .runtime_report();

        assert_eq!(report.selected_backend, SelectedBackend::Mlx);
        assert!(report.capabilities.token_streaming);
        assert!(report.capabilities.text_generation);
    }

    #[test]
    fn preview_session_config_factory_preserves_explicit_native_artifact_dirs() {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            native_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
            native_model_artifacts_dir: Some(Path::new("/tmp/ax-model").to_path_buf()),
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview config factory should preserve explicit native artifact config");

        assert_eq!(
            config.native_runtime_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-metal"))
        );
        assert_eq!(
            config.native_runtime_artifacts_source,
            Some(NativeRuntimeArtifactsSource::ExplicitConfig)
        );
        assert_eq!(
            config.native_model_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-model"))
        );
        assert_eq!(
            config.native_model_artifacts_source,
            Some(NativeModelArtifactsSource::ExplicitConfig)
        );
    }

    #[test]
    fn resolved_session_config_factory_preserves_supplied_runtime_fields() {
        let config = EngineSessionConfig::from_resolved_request(ResolvedSessionConfigRequest {
            cache_group_id: CacheGroupId(7),
            block_size_tokens: 32,
            total_blocks: 2048,
            deterministic: false,
            allow_deterministic_native_fallback: false,
            max_batch_tokens: 4096,
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::LlamaCpp(
                crate::compat::LlamaCppConfig::server_completion(
                    "http://127.0.0.1:8080".to_string(),
                ),
            )),
            native_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
            native_runtime_artifacts_source: Some(NativeRuntimeArtifactsSource::ExplicitConfig),
            native_model_artifacts_dir: Some(Path::new("/tmp/ax-model").to_path_buf()),
            native_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
        });

        assert_eq!(
            config.kv_config,
            KvManagerConfig::new(CacheGroupId(7), 32, 2048)
        );
        assert!(!config.deterministic);
        assert_eq!(config.max_batch_tokens, 4096);
        assert_eq!(config.backend_policy, BackendPolicy::allow_compat());
        assert_eq!(
            config.resolved_backend,
            ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            )
        );
        assert!(config.compatibility_backend.is_some());
        assert_eq!(
            config.native_runtime_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-metal"))
        );
        assert_eq!(
            config.native_runtime_artifacts_source,
            Some(NativeRuntimeArtifactsSource::ExplicitConfig)
        );
        assert_eq!(
            config.native_model_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-model"))
        );
        assert_eq!(
            config.native_model_artifacts_source,
            Some(NativeModelArtifactsSource::ExplicitConfig)
        );
    }

    fn compatibility_server_session(server_url: String) -> EngineSession {
        let config = EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::LlamaCpp(
                crate::compat::LlamaCppConfig::server_completion(server_url),
            )),
            ..EngineSessionConfig::default()
        };
        EngineSession::new(config).expect("compatibility session should build")
    }

    fn spawn_scripted_compat_completion_stream_server(
        expected_requests: usize,
        response_for_request: impl Fn(usize, &Value) -> Vec<Value> + Send + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            for request_index in 0..expected_requests {
                let (mut stream, _) = listener.accept().expect("request should arrive");
                let request = read_http_request(&mut stream);
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4)
                    .expect("request should include header terminator");
                let body =
                    String::from_utf8(request[header_end..].to_vec()).expect("body should be utf8");
                let payload: Value =
                    serde_json::from_str(&body).expect("request body should be json");

                let chunks = response_for_request(request_index, &payload);
                let mut body = String::new();
                for chunk in &chunks {
                    body.push_str("data: ");
                    body.push_str(
                        &serde_json::to_string(chunk).expect("chunk payload should serialize"),
                    );
                    body.push_str("\n\n");
                }
                body.push_str("data: [DONE]\n\n");

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("response should write");
            }
        });

        (format!("http://{address}"), handle)
    }

    fn spawn_compat_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(Value) + Send + Sync + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        spawn_scripted_compat_completion_stream_server(expected_requests, move |_, payload| {
            assert_request(payload.clone());
            chunks.clone()
        })
    }

    // Same SSE wire format as spawn_compat_completion_stream_server; the difference is that the
    // caller supplies OpenAI-format JSON payloads ({"choices": [...]}) rather than llama.cpp-native
    // ones ({"content": ..., "stop": ...}).
    fn spawn_openai_compat_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(Value) + Send + Sync + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        spawn_scripted_compat_completion_stream_server(expected_requests, move |_, payload| {
            assert_request(payload.clone());
            chunks.clone()
        })
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let mut header_end = None;
        let mut content_length = None;

        loop {
            let bytes_read = stream.read(&mut buffer).expect("request should read");
            assert!(
                bytes_read > 0,
                "client closed connection before request completed"
            );
            request.extend_from_slice(&buffer[..bytes_read]);

            if header_end.is_none() {
                header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|index| index + 4);
                if let Some(end) = header_end {
                    let headers =
                        String::from_utf8(request[..end].to_vec()).expect("headers should be utf8");
                    content_length = Some(parse_content_length(&headers));
                }
            }

            if let (Some(end), Some(length)) = (header_end, content_length) {
                if request.len() >= end + length {
                    request.truncate(end + length);
                    return request;
                }
            }
        }
    }

    fn parse_content_length(headers: &str) -> usize {
        headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    Some(
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("content-length should parse"),
                    )
                } else {
                    None
                }
            })
            .expect("content-length header should exist")
    }

    #[test]
    fn rejects_compatibility_backend_without_config() {
        let config = EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            ),
            ..EngineSessionConfig::default()
        };

        let error =
            EngineSession::new(config).expect_err("compatibility backend should require config");

        match error {
            EngineSessionError::MissingCompatibilityBackendConfig { selected_backend } => {
                assert_eq!(selected_backend, SelectedBackend::LlamaCpp);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn compatibility_generate_uses_blocking_adapter_and_text_fields() {
        let script_path = fake_compat_script();
        let model_path = std::env::temp_dir().join("ax-engine-session-fake-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::LlamaCpp(
                crate::compat::LlamaCppConfig::new(script_path, &model_path),
            )),
            ..EngineSessionConfig::default()
        })
        .expect("compatibility session should build");

        let response = session
            .generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("hello from compatibility".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: Some("compat".to_string()),
            })
            .expect("compatibility generate should succeed");

        assert_eq!(response.request_id, 1);
        assert_eq!(
            response.prompt_text.as_deref(),
            Some("hello from compatibility")
        );
        assert_eq!(
            response.output_text.as_deref(),
            Some("session::hello from compatibility")
        );
        assert_eq!(response.output_tokens, Vec::<u32>::new());
        assert_eq!(response.runtime.selected_backend, SelectedBackend::LlamaCpp);
    }

    #[test]
    fn compatibility_generate_with_request_id_preserves_explicit_request_id() {
        let script_path = fake_compat_script();
        let model_path = std::env::temp_dir().join("ax-engine-session-explicit-id.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::LlamaCpp(
                crate::compat::LlamaCppConfig::new(script_path, &model_path),
            )),
            ..EngineSessionConfig::default()
        })
        .expect("compatibility session should build");

        let response = session
            .generate_with_request_id(
                42,
                GenerateRequest {
                    model_id: "qwen3_dense".to_string(),
                    input_tokens: Vec::new(),
                    input_text: Some("hello explicit id".to_string()),
                    max_output_tokens: 2,
                    sampling: Default::default(),
                    metadata: None,
                },
            )
            .expect("compatibility generate should succeed");

        assert_eq!(response.request_id, 42);
    }

    #[test]
    fn compatibility_stream_generate_supports_server_completion_adapter() {
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&serde_json::json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
            },
        );
        let mut session = compatibility_server_session(server_url);

        let events = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("compatibility stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("compatibility stream should complete");

        server_handle
            .join()
            .expect("compatibility server thread should finish");

        assert!(matches!(
            events.first(),
            Some(GenerateStreamEvent::Request(_))
        ));
        assert_eq!(events.len(), 4);

        let GenerateStreamEvent::Step(first_step) = &events[1] else {
            panic!("second event should be first step");
        };
        assert_eq!(first_step.request.state, SessionRequestState::Running);
        assert_eq!(first_step.delta_tokens, vec![4]);
        assert_eq!(first_step.delta_token_logprobs, vec![None]);
        assert_eq!(first_step.step.ttft_events, 1);

        let GenerateStreamEvent::Step(final_step) = &events[2] else {
            panic!("third event should be terminal step");
        };
        assert_eq!(final_step.request.state, SessionRequestState::Finished);
        assert_eq!(final_step.delta_tokens, vec![5]);
        assert_eq!(final_step.delta_token_logprobs, vec![None]);

        let GenerateStreamEvent::Response(response_event) = &events[3] else {
            panic!("final event should be response");
        };
        assert_eq!(response_event.response.request_id, 1);
        assert_eq!(response_event.response.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(response_event.response.output_tokens, vec![4, 5]);
        assert_eq!(
            response_event.response.output_token_logprobs,
            vec![None, None]
        );
        assert_eq!(
            response_event.response.output_text.as_deref(),
            Some("hello world")
        );
        assert_eq!(
            response_event.response.route.execution_plan.as_deref(),
            Some(COMPATIBILITY_STREAM_EXECUTION_PLAN)
        );
        assert_eq!(response_event.response.step_count, 2);
        assert_eq!(response_event.response.ttft_step, Some(1));
        assert_eq!(
            response_event.response.runtime.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert_eq!(
            response_event.response.finish_reason,
            Some(crate::generate::GenerateFinishReason::MaxOutputTokens)
        );
    }

    #[test]
    fn compatibility_stream_generate_maps_openai_finish_reason_stop() {
        // Verifies that the OpenAI-format finish_reason codes ("stop", "length") are
        // correctly mapped in the streaming path, not just the llama.cpp-native ones
        // ("eos", "limit").
        use crate::compat::OpenAiCompatibleServerConfig;

        let (server_url, server_handle) = spawn_openai_compat_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "choices": [{"text": "hello", "finish_reason": null}]
                }),
                serde_json::json!({
                    "choices": [{"text": " world", "finish_reason": "stop"}]
                }),
            ],
            |payload| {
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
            },
        );

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::Vllm,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::Vllm(
                OpenAiCompatibleServerConfig::new(server_url),
            )),
            ..EngineSessionConfig::default()
        })
        .expect("vllm compatibility session should build");

        let events = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("hello vllm stream".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("vllm compatibility stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("vllm compatibility stream should complete");

        server_handle
            .join()
            .expect("vllm server thread should finish");

        let GenerateStreamEvent::Response(response_event) = events.last().unwrap() else {
            panic!("last event should be response");
        };
        assert_eq!(
            response_event.response.finish_reason,
            Some(crate::generate::GenerateFinishReason::Stop),
            "OpenAI 'stop' finish_reason should map to GenerateFinishReason::Stop"
        );
    }

    #[test]
    fn mlx_stream_generate_sends_mlx_specific_sampling_fields() {
        let (server_url, server_handle) = spawn_openai_compat_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "choices": [{"text": "hello", "finish_reason": null}]
                }),
                serde_json::json!({
                    "choices": [{"text": " mlx", "finish_reason": "stop"}]
                }),
                serde_json::json!({
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                        "prompt_tokens_details": {
                            "cached_tokens": 3
                        }
                    }
                }),
            ],
            |payload| {
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("top_k"), Some(&Value::Number(3.into())));
                let repetition_penalty = payload
                    .get("repetition_penalty")
                    .and_then(Value::as_f64)
                    .expect("mlx payload should include repetition_penalty");
                assert!((repetition_penalty - 1.2).abs() < 1e-6);
                assert_eq!(
                    payload.get("stream_options"),
                    Some(&serde_json::json!({ "include_usage": true }))
                );
            },
        );

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::Mlx,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::Mlx(
                crate::compat::MlxConfig::server_completions(server_url),
            )),
            ..EngineSessionConfig::default()
        })
        .expect("mlx compatibility session should build");

        let events = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("hello mlx stream".to_string()),
                max_output_tokens: 2,
                sampling: crate::generate::GenerateSampling {
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 3,
                    repetition_penalty: 1.2,
                    seed: 11,
                    deterministic: Some(true),
                },
                metadata: None,
            })
            .expect("mlx compatibility stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("mlx compatibility stream should complete");

        server_handle
            .join()
            .expect("mlx server thread should finish");

        let GenerateStreamEvent::Response(response_event) = events.last().unwrap() else {
            panic!("last event should be response");
        };
        assert_eq!(
            response_event.response.finish_reason,
            Some(crate::generate::GenerateFinishReason::Stop)
        );
        assert_eq!(response_event.response.prompt_token_count, Some(5));
        assert_eq!(response_event.response.output_token_count, Some(2));
        assert_eq!(
            response_event.response.route.prefix_cache_path.as_deref(),
            Some("delegated_prompt_cache")
        );
        assert_eq!(
            response_event
                .response
                .route
                .crossover_decisions
                .get("delegated_cached_tokens"),
            Some(&3)
        );
    }

    #[test]
    fn compatibility_stream_generate_maps_openai_finish_reason_length() {
        use crate::compat::OpenAiCompatibleServerConfig;

        let (server_url, server_handle) = spawn_openai_compat_completion_stream_server(
            1,
            vec![serde_json::json!({
                "choices": [{"text": "token", "finish_reason": "length"}]
            })],
            |_| {},
        );

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::Vllm,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::Vllm(
                OpenAiCompatibleServerConfig::new(server_url),
            )),
            ..EngineSessionConfig::default()
        })
        .expect("vllm compatibility session should build");

        let events = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("hello vllm length".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("vllm compatibility stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("vllm compatibility stream should complete");

        server_handle
            .join()
            .expect("vllm server thread should finish");

        let GenerateStreamEvent::Response(response_event) = events.last().unwrap() else {
            panic!("last event should be response");
        };
        assert_eq!(
            response_event.response.finish_reason,
            Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
            "OpenAI 'length' finish_reason should map to GenerateFinishReason::MaxOutputTokens"
        );
    }

    #[test]
    fn compatibility_stream_generate_rejects_cli_fallback_adapter() {
        let script_path = fake_compat_script();
        let model_path =
            std::env::temp_dir().join("ax-engine-session-stream-cli-fallback-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::LlamaCpp(
                crate::compat::LlamaCppConfig::new(script_path, &model_path),
            )),
            ..EngineSessionConfig::default()
        })
        .expect("compatibility session should build");

        let error = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("cli fallback stream".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect_err("cli fallback streaming should fail closed");

        assert!(matches!(
            error,
            EngineSessionError::Compatibility(CompatibilityBackendError::StreamingNotSupported {
                selected_backend: SelectedBackend::LlamaCpp,
            })
        ));
    }

    #[test]
    fn compatibility_cli_submit_generate_fails_closed() {
        let script_path = fake_compat_script();
        let model_path = std::env::temp_dir().join("ax-engine-session-fake-model-lifecycle.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_compat(),
            resolved_backend: ResolvedBackend::compatibility(
                SelectedBackend::LlamaCpp,
                "native preview not ready for this model",
            ),
            compatibility_backend: Some(CompatibilityBackendConfig::LlamaCpp(
                crate::compat::LlamaCppConfig::new(script_path, &model_path),
            )),
            ..EngineSessionConfig::default()
        })
        .expect("compatibility session should build");

        let error = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("unsupported lifecycle".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect_err("compatibility CLI submit should fail closed");

        match error {
            EngineSessionError::Compatibility(
                CompatibilityBackendError::StreamingNotSupported { selected_backend },
            ) => {
                assert_eq!(selected_backend, SelectedBackend::LlamaCpp);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn compatibility_stepwise_lifecycle_supports_server_completion_adapter() {
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
            1,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                assert_eq!(payload.get("prompt"), Some(&serde_json::json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
            },
        );
        let mut session = compatibility_server_session(server_url);

        let request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("compatibility submit should succeed");

        let initial = session
            .request_report(request_id)
            .expect("compatibility request should exist");
        assert_eq!(initial.state, SessionRequestState::Waiting);

        let first_step = session.step_report().expect("first step should succeed");
        assert_eq!(first_step.scheduled_requests, 1);
        assert_eq!(first_step.scheduled_tokens, 1);
        assert_eq!(first_step.ttft_events, 1);

        let running = session
            .request_report(request_id)
            .expect("running compatibility request should exist");
        assert_eq!(running.state, SessionRequestState::Running);
        assert_eq!(running.output_tokens, vec![4]);

        let second_step = session.step_report().expect("second step should succeed");
        assert_eq!(second_step.scheduled_requests, 1);
        assert_eq!(second_step.scheduled_tokens, 1);

        let terminal = session
            .request_report(request_id)
            .expect("terminal compatibility request should exist");
        assert_eq!(terminal.state, SessionRequestState::Finished);
        assert_eq!(terminal.output_tokens, vec![4, 5]);
        assert_eq!(
            terminal.execution_plan_ref.as_deref(),
            Some(COMPATIBILITY_STREAM_EXECUTION_PLAN)
        );

        server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn compatibility_stepwise_lifecycle_reports_delegated_prompt_cache_hits() {
        let (server_url, server_handle) =
            spawn_scripted_compat_completion_stream_server(1, |_, payload| {
                assert_eq!(payload.get("prompt"), Some(&serde_json::json!([1, 2, 3])));
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));

                vec![
                    serde_json::json!({
                        "content": "",
                        "tokens": [],
                        "stop": false,
                        "prompt_progress": {
                            "total": 3,
                            "cache": 2,
                            "processed": 2,
                            "time_ms": 1.0
                        }
                    }),
                    serde_json::json!({
                        "content": " cache",
                        "tokens": [4],
                        "stop": false
                    }),
                    serde_json::json!({
                        "content": " hit",
                        "tokens": [5],
                        "stop": true,
                        "stop_type": "limit"
                    }),
                ]
            });
        let mut session = compatibility_server_session(server_url);

        let request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("compatibility submit should succeed");

        let progress_step = session.step_report().expect("progress step should succeed");
        assert_eq!(progress_step.scheduled_requests, 1);
        assert_eq!(progress_step.scheduled_tokens, 0);
        assert_eq!(progress_step.prefix_hits, 1);

        let progress_report = session
            .request_report(request_id)
            .expect("compatibility request should exist after progress");
        assert_eq!(progress_report.processed_prompt_tokens, 2);
        assert_eq!(progress_report.prompt_len, 3);
        assert_eq!(
            progress_report.route.prefix_cache_path.as_deref(),
            Some("delegated_prompt_cache")
        );
        assert_eq!(
            progress_report
                .route
                .crossover_decisions
                .get("delegated_cached_tokens"),
            Some(&2)
        );

        let first_decode_step = session
            .step_report()
            .expect("first decode step should succeed");
        assert_eq!(first_decode_step.scheduled_requests, 1);
        assert_eq!(first_decode_step.scheduled_tokens, 1);
        assert_eq!(first_decode_step.prefix_hits, 0);

        let second_decode_step = session
            .step_report()
            .expect("second decode step should succeed");
        assert_eq!(second_decode_step.scheduled_requests, 1);
        assert_eq!(second_decode_step.scheduled_tokens, 1);

        let terminal = session
            .request_report(request_id)
            .expect("terminal compatibility request should exist");
        assert_eq!(terminal.state, SessionRequestState::Finished);
        assert_eq!(terminal.output_tokens, vec![4, 5]);
        assert_eq!(
            terminal.route.prefix_cache_path.as_deref(),
            Some("delegated_prompt_cache")
        );

        server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn compatibility_stepwise_lifecycle_advances_multiple_active_requests() {
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    prompt == &serde_json::json!([1, 2, 3])
                        || prompt == &serde_json::json!([7, 8, 9])
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
            },
        );
        let mut session = compatibility_server_session(server_url);

        let first_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("first compatibility submit should succeed");

        let second_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![7, 8, 9],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("second compatibility submit should also succeed");

        let first_step = session
            .step_report()
            .expect("first aggregated step should succeed");
        assert_eq!(first_step.scheduled_requests, 2);
        assert_eq!(first_step.scheduled_tokens, 2);
        assert_eq!(first_step.ttft_events, 2);

        let first_running = session
            .request_report(first_request_id)
            .expect("first compatibility request should exist");
        assert_eq!(first_running.state, SessionRequestState::Running);
        assert_eq!(first_running.output_tokens, vec![4]);
        let second_running = session
            .request_report(second_request_id)
            .expect("second compatibility request should exist");
        assert_eq!(second_running.state, SessionRequestState::Running);
        assert_eq!(second_running.output_tokens, vec![4]);

        let second_step = session
            .step_report()
            .expect("second aggregated step should succeed");
        assert_eq!(second_step.scheduled_requests, 2);
        assert_eq!(second_step.scheduled_tokens, 2);

        let first_terminal = session
            .request_report(first_request_id)
            .expect("first terminal request should exist");
        assert_eq!(first_terminal.state, SessionRequestState::Finished);
        assert_eq!(first_terminal.output_tokens, vec![4, 5]);
        let second_terminal = session
            .request_report(second_request_id)
            .expect("second terminal request should exist");
        assert_eq!(second_terminal.state, SessionRequestState::Finished);
        assert_eq!(second_terminal.output_tokens, vec![4, 5]);

        server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn compatibility_cancelled_request_does_not_block_other_active_requests() {
        let (server_url, server_handle) = spawn_compat_completion_stream_server(
            2,
            vec![
                serde_json::json!({
                    "content": "hello",
                    "tokens": [4],
                    "stop": false
                }),
                serde_json::json!({
                    "content": " world",
                    "tokens": [5],
                    "stop": true,
                    "stop_type": "limit"
                }),
            ],
            |payload| {
                let prompt = payload.get("prompt").expect("prompt should be present");
                assert!(
                    prompt == &serde_json::json!([1, 2, 3])
                        || prompt == &serde_json::json!([7, 8, 9])
                );
                assert_eq!(payload.get("stream"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_tokens"), Some(&Value::Bool(true)));
                assert_eq!(payload.get("return_progress"), Some(&Value::Bool(true)));
            },
        );
        let mut session = compatibility_server_session(server_url);

        let first_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("first compatibility submit should succeed");
        let second_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![7, 8, 9],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect("second compatibility submit should succeed");

        session
            .cancel_request(first_request_id)
            .expect("compatibility cancel should succeed");
        let cancelled = session
            .request_report(first_request_id)
            .expect("cancelled compatibility request should exist");
        assert_eq!(cancelled.state, SessionRequestState::Cancelled);
        assert!(cancelled.cancel_requested);

        let first_step = session
            .step_report()
            .expect("aggregated step after cancel should succeed");
        assert_eq!(first_step.scheduled_requests, 1);
        assert_eq!(first_step.scheduled_tokens, 1);
        assert_eq!(first_step.ttft_events, 1);

        let running = session
            .request_report(second_request_id)
            .expect("remaining compatibility request should exist");
        assert_eq!(running.state, SessionRequestState::Running);
        assert_eq!(running.output_tokens, vec![4]);

        let second_step = session
            .step_report()
            .expect("terminal step after cancel should succeed");
        assert_eq!(second_step.scheduled_requests, 1);
        assert_eq!(second_step.scheduled_tokens, 1);

        let terminal = session
            .request_report(second_request_id)
            .expect("terminal compatibility request should exist");
        assert_eq!(terminal.state, SessionRequestState::Finished);
        assert_eq!(terminal.output_tokens, vec![4, 5]);

        server_handle
            .join()
            .expect("compatibility server thread should finish");
    }

    #[test]
    fn native_submit_rejects_input_text() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");

        let error = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("native should reject text".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect_err("native preview should reject input_text");

        assert!(matches!(
            error,
            EngineSessionError::NativeBackendRequiresTokenizedInput
        ));
    }

    #[test]
    fn generate_rejects_empty_input_text() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");

        let error = session
            .generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some(String::new()),
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: None,
            })
            .expect_err("empty input_text should be rejected");

        assert!(matches!(error, EngineSessionError::EmptyInputTokens));
    }

    #[test]
    fn steps_native_request_through_preview_session() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");
        session
            .submit(sample_submission())
            .expect("submission should succeed");

        for _ in 0..8 {
            let snapshot = session
                .core()
                .request_manager()
                .snapshot(RequestId(1))
                .expect("request should exist");
            if snapshot.state.is_terminal() {
                break;
            }

            session.step().expect("step should succeed");
        }

        let snapshot = session
            .core()
            .request_manager()
            .snapshot(RequestId(1))
            .expect("request should still exist");
        assert_eq!(snapshot.state, RequestState::Finished);
    }

    #[test]
    fn generate_runs_request_to_terminal_response() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");

        let response = session
            .generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![10, 11, 12],
                input_text: None,
                max_output_tokens: 3,
                sampling: Default::default(),
                metadata: Some("server-preview".to_string()),
            })
            .expect("generate request should complete");

        assert_eq!(response.request_id, 1);
        assert_eq!(response.model_id, "qwen3_dense");
        assert_eq!(response.prompt_tokens, vec![10, 11, 12]);
        assert_eq!(response.output_tokens.len(), 3);
        assert_eq!(response.status, GenerateStatus::Finished);
        assert!(response.step_count > 0);
        assert_eq!(response.runtime.selected_backend, SelectedBackend::AxNative);
        assert_eq!(response.runtime.support_tier, SupportTier::NativePreview);
        assert_eq!(
            response.runtime.resolution_policy,
            crate::backend::ResolutionPolicy::StrictNative
        );
    }

    #[test]
    fn stepwise_request_report_reaches_terminal_state() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");
        let request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![7, 8, 9],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: Some("stepwise".to_string()),
            })
            .expect("submit should succeed");

        let initial = session
            .request_report(request_id)
            .expect("request report should exist after submit");
        assert_eq!(initial.state, crate::request::SessionRequestState::Waiting);
        assert!(initial.output_tokens.is_empty());

        for _ in 0..8 {
            let report = session
                .request_report(request_id)
                .expect("request report should still exist");
            if matches!(
                report.state,
                crate::request::SessionRequestState::Finished
                    | crate::request::SessionRequestState::Cancelled
                    | crate::request::SessionRequestState::Failed
            ) {
                break;
            }

            let step = session.step_report().expect("step should succeed");
            assert!(step.step_id.is_some());
        }

        let final_report = session
            .request_report(request_id)
            .expect("terminal request report should exist");
        assert_eq!(
            final_report.state,
            crate::request::SessionRequestState::Finished
        );
        assert_eq!(final_report.output_len, 2);
        assert_eq!(final_report.output_tokens.len(), 2);
    }

    #[test]
    fn native_placeholder_runtime_marks_request_step_and_response_routes_explicitly() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");
        let request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![3, 4, 5],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: Some("placeholder-route".to_string()),
            })
            .expect("submit should succeed");

        let initial_report = session
            .request_report(request_id)
            .expect("initial request report should exist");
        assert_eq!(
            initial_report
                .route
                .crossover_decisions
                .get(NATIVE_PLACEHOLDER_EXECUTION_DECISION),
            Some(&1)
        );
        assert_eq!(
            initial_report
                .route
                .crossover_decisions
                .get(NATIVE_RUNTIME_RUNNER_DETERMINISTIC_DECISION),
            Some(&1)
        );
        assert_eq!(
            initial_report
                .route
                .crossover_decisions
                .get(NATIVE_RUNTIME_RUNNER_METAL_BRINGUP_DECISION),
            Some(&0)
        );
        assert_eq!(
            initial_report
                .route
                .crossover_decisions
                .get(NATIVE_RUNTIME_ARTIFACTS_VALIDATED_DECISION),
            Some(&0)
        );

        let first_step = session.step_report().expect("step should succeed");
        assert_eq!(
            first_step.route.as_ref().and_then(|route| route
                .crossover_decisions
                .get(NATIVE_PLACEHOLDER_EXECUTION_DECISION)),
            Some(&1)
        );
        assert_eq!(
            first_step.route.as_ref().and_then(|route| route
                .crossover_decisions
                .get(NATIVE_RUNTIME_RUNNER_DETERMINISTIC_DECISION)),
            Some(&1)
        );
        assert_eq!(
            first_step.route.as_ref().and_then(|route| route
                .crossover_decisions
                .get(NATIVE_RUNTIME_RUNNER_METAL_BRINGUP_DECISION)),
            Some(&0)
        );

        let response = session
            .run_to_completion(request_id)
            .expect("request should complete");
        assert_eq!(
            response
                .route
                .crossover_decisions
                .get(NATIVE_PLACEHOLDER_EXECUTION_DECISION),
            Some(&1)
        );
        assert_eq!(
            response
                .route
                .crossover_decisions
                .get(NATIVE_PLACEHOLDER_EXPLICIT_OPT_IN_DECISION),
            Some(&1)
        );
        assert_eq!(
            response
                .route
                .crossover_decisions
                .get(NATIVE_RUNTIME_RUNNER_DETERMINISTIC_DECISION),
            Some(&1)
        );
        assert_eq!(
            response
                .route
                .crossover_decisions
                .get(NATIVE_RUNTIME_RUNNER_METAL_BRINGUP_DECISION),
            Some(&0)
        );
        assert_eq!(
            response
                .route
                .crossover_decisions
                .get(NATIVE_RUNTIME_ARTIFACTS_VALIDATED_DECISION),
            Some(&0)
        );
    }

    #[test]
    fn native_step_report_surfaces_route_and_metal_dispatch_summary() {
        let core = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(0), 16, 64),
            TraceReportingRunner {
                trace: sample_metal_dispatch_trace(),
            },
            DeterministicSampler,
        );
        let config = native_placeholder_session_config();
        let mut session = EngineSession {
            core,
            runtime: config.runtime_report(),
            config,
            next_request_id: 2,
            compatibility_requests: BTreeMap::new(),
            compatibility_terminal_request_order: VecDeque::new(),
        };

        session
            .submit(sample_submission())
            .expect("submission should succeed");

        let step = session.step_report().expect("step should succeed");

        assert_eq!(step.scheduled_requests, 1);
        assert_eq!(
            step.route
                .as_ref()
                .and_then(|route| route.attention_route.as_deref()),
            Some("mock_native_attention")
        );
        assert_eq!(
            step.route
                .as_ref()
                .and_then(|route| route.crossover_decisions.get("metal_dispatch_completed")),
            Some(&1)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.runtime_required_pipeline_count),
            Some(4)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.binary_archive_state),
            Some(MetalBinaryArchiveState::Loaded)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .and_then(|dispatch| dispatch.numeric.validation.as_ref())
                .map(|validation| validation.attention_max_abs_diff_microunits),
            Some(0)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.runtime_model_conditioned_inputs),
            Some(true)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.runtime_real_model_tensor_inputs),
            Some(true)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.runtime_complete_model_forward_supported),
            Some(true)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.runtime_model_buffers_bound),
            Some(true)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.runtime_model_buffer_count),
            Some(12)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .and_then(|dispatch| dispatch.runtime_model_family.as_deref()),
            Some("qwen3_dense")
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_direct_decode_token_count),
            Some(1)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_logits_output_count),
            Some(1)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_remaining_logits_handle_count),
            Some(0)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_model_bound_ffn_decode),
            Some(true)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_real_model_forward_completed),
            Some(true)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_prefix_native_dispatch_count),
            Some(35)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_prefix_cpu_reference_dispatch_count),
            Some(1)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_qkv_projection_token_count),
            Some(72)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_layer_continuation_token_count),
            Some(37)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_logits_projection_token_count),
            Some(1)
        );
        assert_eq!(
            step.metal_dispatch
                .as_ref()
                .map(|dispatch| dispatch.execution_logits_vocab_scan_row_count),
            Some(151936)
        );
    }

    #[test]
    fn slice_output_token_logprobs_fails_closed_on_length_mismatch() {
        let mut report = sample_terminal_compatibility_report(41);
        report.output_token_logprobs.pop();

        let error = slice_output_token_logprobs(&report, 0, 2)
            .expect_err("mismatched logprob lengths should fail closed");

        assert!(matches!(
            error,
            EngineSessionError::RequestReportInvariantViolation { request_id: 41, .. }
        ));
    }

    #[test]
    fn compatibility_terminal_requests_are_pruned_after_retention_limit() {
        let mut session = compatibility_server_session("http://127.0.0.1:1".to_string());

        for request_id in 1..=(MAX_COMPATIBILITY_TERMINAL_REQUESTS as u64 + 1) {
            session.store_terminal_compatibility_report(
                request_id,
                sample_terminal_compatibility_report(request_id),
            );
        }

        assert!(session.request_report(1).is_none());
        assert!(session
            .request_report(MAX_COMPATIBILITY_TERMINAL_REQUESTS as u64 + 1)
            .is_some());
        assert_eq!(
            session.compatibility_terminal_request_order.len(),
            MAX_COMPATIBILITY_TERMINAL_REQUESTS
        );
    }

    #[test]
    fn submit_generate_with_explicit_request_id_uses_caller_supplied_id() {
        let mut session =
            EngineSession::new(native_placeholder_session_config()).expect("session should build");

        let request_id = session
            .submit_generate_with_request_id(
                41,
                GenerateRequest {
                    model_id: "qwen3_dense".to_string(),
                    input_tokens: vec![1, 2, 3],
                    input_text: None,
                    max_output_tokens: 2,
                    sampling: Default::default(),
                    metadata: Some("explicit-id".to_string()),
                },
            )
            .expect("submission should succeed");

        assert_eq!(request_id, 41);
        let report = session
            .request_report(request_id)
            .expect("request report should exist");
        assert_eq!(report.request_id, 41);
    }

    #[test]
    fn stream_generate_emits_request_step_and_response_events() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");

        let events = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![4, 5, 6],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                metadata: Some("stream".to_string()),
            })
            .expect("stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("stream should complete");

        assert!(matches!(
            events.first(),
            Some(GenerateStreamEvent::Request(_))
        ));
        assert!(events
            .iter()
            .any(|event| matches!(event, GenerateStreamEvent::Step(_))));

        let step_events = events
            .iter()
            .filter(|event| matches!(event, GenerateStreamEvent::Step(_)))
            .count();
        assert!(step_events > 0);
        for step_event in events.iter().filter_map(|event| match event {
            GenerateStreamEvent::Step(step_event) => Some(step_event),
            _ => None,
        }) {
            assert_eq!(
                step_event.delta_tokens.len(),
                step_event.delta_token_logprobs.len()
            );
        }

        let last_event = events.last().expect("response event should exist");
        let GenerateStreamEvent::Response(response_event) = last_event else {
            panic!("final stream event should be response");
        };

        assert_eq!(response_event.response.request_id, 1);
        assert_eq!(response_event.response.prompt_tokens, vec![4, 5, 6]);
        assert_eq!(response_event.response.output_tokens.len(), 2);
        assert_eq!(
            response_event.response.output_tokens.len(),
            response_event.response.output_token_logprobs.len()
        );
        assert_eq!(response_event.response.status, GenerateStatus::Finished);
        assert_eq!(response_event.response.step_count, step_events as u64);
    }

    #[test]
    fn cancelled_request_can_complete_through_terminal_report() {
        let mut session = EngineSession::new(native_placeholder_session_config())
            .expect("native preview session should build");
        let request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 4,
                sampling: Default::default(),
                metadata: Some("cancel".to_string()),
            })
            .expect("submit should succeed");

        session
            .cancel_request(request_id)
            .expect("cancel should succeed");

        let report = session
            .request_report(request_id)
            .expect("cancelled request should still have a report");
        assert_eq!(report.state, crate::request::SessionRequestState::Cancelled);
        assert!(report.cancel_requested);

        let response = session
            .run_to_completion(request_id)
            .expect("terminal cancelled request should convert into response");
        assert_eq!(response.status, GenerateStatus::Cancelled);
        assert_eq!(
            response.finish_reason,
            Some(crate::generate::GenerateFinishReason::Cancelled)
        );
    }
}
