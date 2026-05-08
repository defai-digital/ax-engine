use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use ax_engine_core::{
    CacheGroupId, EmbeddingPooling, EngineCore, EngineCoreError, EngineStepOutcome,
    KvManagerConfig, MetalKernelAssets, MetalRuntimeError, MlxKvCompressionConfig, ModelId,
    RequestId, RequestSubmission, SequenceNo,
};
use thiserror::Error;

use crate::backend::{
    BackendContractError, BackendPolicy, CapabilityReport, NativeModelArtifactsSource,
    NativeModelReport, NativeRuntimeArtifactsSource, NativeRuntimeReport, PreviewBackendRequest,
    PreviewBackendResolutionError, ResolvedBackend, RuntimeReport, SelectedBackend, SupportTier,
    resolve_preview_backend,
};
use crate::generate::{
    GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateStreamEvent,
    GenerateStreamRequestEvent, GenerateStreamResponseEvent, GenerateStreamStepEvent,
};
use crate::host;
use crate::llama_cpp::{
    LlamaCppBackendError, LlamaCppConfig, LlamaCppPromptProgress, LlamaCppStreamChunk,
    LlamaCppStreamHandle, run_blocking_generate, start_streaming_generate,
};
use crate::mlx_lm::{
    MlxLmBackendError, MlxLmConfig, MlxLmStreamChunkResult, MlxLmStreamHandle,
    finish_reason_from_mlx_lm, run_blocking_generate as run_mlx_lm_generate,
    start_streaming_generate as start_mlx_lm_streaming_generate,
};
use crate::request::{
    EngineStepReport, MetalDispatchStepReport, SessionRequestReport, SessionRequestState,
};

const LLAMA_CPP_STREAM_EXECUTION_PLAN: &str = "llama_cpp.server_completion_stream";
const MLX_LM_STREAM_EXECUTION_PLAN: &str = "mlx_lm_delegated.server_completion_stream";
const NATIVE_METAL_BUILD_DIR_ENV: &str = "AX_ENGINE_METAL_BUILD_DIR";
const NATIVE_MODEL_DIR_ENV: &str = "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR";
const MAX_LLAMA_CPP_TERMINAL_REQUESTS: usize = 1024;
const MAX_NATIVE_ROUTE_REPORTS: usize = 1024;

#[derive(Clone, Debug)]
pub struct EngineSessionConfig {
    pub kv_config: KvManagerConfig,
    pub deterministic: bool,
    pub max_batch_tokens: u32,
    pub backend_policy: BackendPolicy,
    pub resolved_backend: ResolvedBackend,
    pub llama_backend: Option<LlamaCppConfig>,
    pub mlx_lm_backend: Option<MlxLmConfig>,
    pub mlx_runtime_artifacts_dir: Option<PathBuf>,
    pub mlx_runtime_artifacts_source: Option<NativeRuntimeArtifactsSource>,
    pub mlx_model_artifacts_dir: Option<PathBuf>,
    pub mlx_model_artifacts_source: Option<NativeModelArtifactsSource>,
    /// When true, MLX runner disables n-gram acceleration and uses the direct path.
    pub mlx_disable_ngram_acceleration: bool,
    /// Optional MLX KV compression policy. Disabled by default.
    pub mlx_kv_compression: MlxKvCompressionConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreviewSessionConfigRequest {
    pub cache_group_id: CacheGroupId,
    pub block_size_tokens: u32,
    pub total_blocks: u32,
    pub deterministic: bool,
    pub max_batch_tokens: u32,
    pub backend_request: PreviewBackendRequest,
    pub mlx_runtime_artifacts_dir: Option<PathBuf>,
    pub mlx_model_artifacts_dir: Option<PathBuf>,
    /// When true, MLX runner disables n-gram acceleration and uses the direct path.
    pub mlx_disable_ngram_acceleration: bool,
    /// Optional MLX KV compression policy. Disabled by default.
    pub mlx_kv_compression: MlxKvCompressionConfig,
}

impl Default for PreviewSessionConfigRequest {
    fn default() -> Self {
        Self {
            cache_group_id: CacheGroupId(0),
            block_size_tokens: 16,
            total_blocks: 1024,
            deterministic: true,
            max_batch_tokens: 2048,
            backend_request: PreviewBackendRequest::default(),
            mlx_runtime_artifacts_dir: None,
            mlx_model_artifacts_dir: None,
            mlx_disable_ngram_acceleration: false,
            mlx_kv_compression: MlxKvCompressionConfig::disabled(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedSessionConfigRequest {
    pub cache_group_id: CacheGroupId,
    pub block_size_tokens: u32,
    pub total_blocks: u32,
    pub deterministic: bool,
    pub max_batch_tokens: u32,
    pub backend_policy: BackendPolicy,
    pub resolved_backend: ResolvedBackend,
    pub llama_backend: Option<LlamaCppConfig>,
    pub mlx_lm_backend: Option<MlxLmConfig>,
    pub mlx_runtime_artifacts_dir: Option<PathBuf>,
    pub mlx_runtime_artifacts_source: Option<NativeRuntimeArtifactsSource>,
    pub mlx_model_artifacts_dir: Option<PathBuf>,
    pub mlx_model_artifacts_source: Option<NativeModelArtifactsSource>,
    pub mlx_disable_ngram_acceleration: bool,
    pub mlx_kv_compression: MlxKvCompressionConfig,
}

impl Default for ResolvedSessionConfigRequest {
    fn default() -> Self {
        let default = EngineSessionConfig::default();
        Self {
            cache_group_id: default.kv_config.cache_group_id,
            block_size_tokens: default.kv_config.block_size_tokens,
            total_blocks: default.kv_config.total_blocks,
            deterministic: default.deterministic,
            max_batch_tokens: default.max_batch_tokens,
            backend_policy: default.backend_policy,
            resolved_backend: default.resolved_backend,
            llama_backend: default.llama_backend,
            mlx_lm_backend: default.mlx_lm_backend,
            mlx_runtime_artifacts_dir: default.mlx_runtime_artifacts_dir,
            mlx_runtime_artifacts_source: default.mlx_runtime_artifacts_source,
            mlx_model_artifacts_dir: default.mlx_model_artifacts_dir,
            mlx_model_artifacts_source: default.mlx_model_artifacts_source,
            mlx_disable_ngram_acceleration: default.mlx_disable_ngram_acceleration,
            mlx_kv_compression: default.mlx_kv_compression,
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
        let mlx_runtime_artifacts = Self::default_mlx_runtime_artifacts_selection();
        let mlx_model_artifacts = Self::default_mlx_model_artifacts_selection();
        Self {
            kv_config: KvManagerConfig::new(CacheGroupId(0), 16, 1024),
            deterministic: true,
            max_batch_tokens: 2048,
            backend_policy: BackendPolicy::mlx_only(),
            resolved_backend: ResolvedBackend::mlx_preview(),
            llama_backend: None,
            mlx_lm_backend: None,
            mlx_runtime_artifacts_dir: mlx_runtime_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone()),
            mlx_runtime_artifacts_source: mlx_runtime_artifacts.map(|selection| selection.source),
            mlx_model_artifacts_dir: mlx_model_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone()),
            mlx_model_artifacts_source: mlx_model_artifacts.map(|selection| selection.source),
            mlx_disable_ngram_acceleration: false,
            mlx_kv_compression: MlxKvCompressionConfig::disabled(),
        }
    }
}

impl EngineSessionConfig {
    pub fn mlx_model_artifacts_dir(&self) -> Option<&Path> {
        self.mlx_model_artifacts_dir.as_deref()
    }

    pub fn with_mlx_model_artifacts_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.mlx_model_artifacts_dir = Some(path.into());
        self.mlx_model_artifacts_source = Some(NativeModelArtifactsSource::ExplicitConfig);
        self
    }

    /// Sets the KV compression policy. Use `MlxKvCompressionConfig::turboquant_shadow()` for
    /// accounting-only TurboQuant, or `MlxKvCompressionConfig::turboquant_fused_experimental()`
    /// for the experimental fused-decode path.
    pub fn with_kv_compression(mut self, config: MlxKvCompressionConfig) -> Self {
        self.mlx_kv_compression = config;
        self
    }

    /// Enables TurboQuant shadow mode: KV compression accounting runs alongside full-precision
    /// inference. Output tokens and logits are not affected. Safe for production monitoring.
    pub fn with_turboquant_shadow(mut self) -> Self {
        self.mlx_kv_compression = MlxKvCompressionConfig::turboquant_shadow();
        self
    }

    /// Enables TurboQuant fused-decode mode: compressed KV paths are used when all quality
    /// gates pass; falls back to full precision otherwise. **Experimental — not production-ready.**
    pub fn with_turboquant_fused_experimental(mut self) -> Self {
        self.mlx_kv_compression = MlxKvCompressionConfig::turboquant_fused_experimental();
        self
    }

    /// Disables n-gram speculation, forcing the direct single-token decode path.
    /// Useful for benchmarking baseline throughput or diagnosing speculation overhead.
    pub fn without_ngram_acceleration(mut self) -> Self {
        self.mlx_disable_ngram_acceleration = true;
        self
    }

    pub fn from_preview_request(
        request: PreviewSessionConfigRequest,
    ) -> Result<Self, PreviewSessionConfigError> {
        let resolution = resolve_preview_backend(request.backend_request)?;
        let default = Self::default();
        let mlx_runtime_artifacts =
            request
                .mlx_runtime_artifacts_dir
                .map(|dir| MlxRuntimeArtifactsSelection {
                    dir,
                    source: NativeRuntimeArtifactsSource::ExplicitConfig,
                });
        let mlx_model_artifacts =
            request
                .mlx_model_artifacts_dir
                .map(|dir| MlxModelArtifactsSelection {
                    source: NativeModelArtifactsSource::ExplicitConfig,
                    dir,
                });

        Ok(Self {
            kv_config: KvManagerConfig::new(
                request.cache_group_id,
                request.block_size_tokens,
                request.total_blocks,
            ),
            deterministic: request.deterministic,
            max_batch_tokens: request.max_batch_tokens,
            backend_policy: resolution.backend_policy,
            resolved_backend: resolution.resolved_backend,
            llama_backend: resolution.llama_backend,
            mlx_lm_backend: resolution.mlx_lm_backend,
            mlx_runtime_artifacts_dir: mlx_runtime_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone())
                .or(default.mlx_runtime_artifacts_dir),
            mlx_runtime_artifacts_source: mlx_runtime_artifacts
                .map(|selection| selection.source)
                .or(default.mlx_runtime_artifacts_source),
            mlx_model_artifacts_dir: mlx_model_artifacts
                .as_ref()
                .map(|selection| selection.dir.clone())
                .or(default.mlx_model_artifacts_dir),
            mlx_model_artifacts_source: mlx_model_artifacts
                .map(|selection| selection.source)
                .or(default.mlx_model_artifacts_source),
            mlx_disable_ngram_acceleration: request.mlx_disable_ngram_acceleration,
            mlx_kv_compression: request.mlx_kv_compression,
        })
    }

    pub fn default_mlx_runtime_artifacts_dir() -> Option<PathBuf> {
        Self::default_mlx_runtime_artifacts_selection().map(|selection| selection.dir)
    }

    pub fn default_mlx_runtime_artifacts_source() -> Option<NativeRuntimeArtifactsSource> {
        Self::default_mlx_runtime_artifacts_selection().map(|selection| selection.source)
    }

    pub fn default_mlx_model_artifacts_dir() -> Option<PathBuf> {
        Self::default_mlx_model_artifacts_selection().map(|selection| selection.dir)
    }

    pub fn default_mlx_model_artifacts_source() -> Option<NativeModelArtifactsSource> {
        Self::default_mlx_model_artifacts_selection().map(|selection| selection.source)
    }

    pub fn from_resolved_request(request: ResolvedSessionConfigRequest) -> Self {
        Self {
            kv_config: KvManagerConfig::new(
                request.cache_group_id,
                request.block_size_tokens,
                request.total_blocks,
            ),
            deterministic: request.deterministic,
            max_batch_tokens: request.max_batch_tokens,
            backend_policy: request.backend_policy,
            resolved_backend: request.resolved_backend,
            llama_backend: request.llama_backend,
            mlx_lm_backend: request.mlx_lm_backend,
            mlx_runtime_artifacts_dir: request.mlx_runtime_artifacts_dir,
            mlx_runtime_artifacts_source: request.mlx_runtime_artifacts_source,
            mlx_model_artifacts_dir: request.mlx_model_artifacts_dir,
            mlx_model_artifacts_source: request.mlx_model_artifacts_source,
            mlx_disable_ngram_acceleration: request.mlx_disable_ngram_acceleration,
            mlx_kv_compression: request.mlx_kv_compression,
        }
    }

    fn default_mlx_runtime_artifacts_selection() -> Option<MlxRuntimeArtifactsSelection> {
        let explicit_dir = env::var_os(NATIVE_METAL_BUILD_DIR_ENV).map(PathBuf::from);
        let current_dir = env::current_dir().ok();
        resolve_default_mlx_runtime_artifacts_selection(explicit_dir, current_dir.as_deref())
    }

    fn default_mlx_model_artifacts_selection() -> Option<MlxModelArtifactsSelection> {
        env::var_os(NATIVE_MODEL_DIR_ENV)
            .map(PathBuf::from)
            .map(|dir| MlxModelArtifactsSelection {
                source: NativeModelArtifactsSource::ExplicitEnv,
                dir,
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

        match self.resolved_backend.selected_backend {
            SelectedBackend::Mlx => {}
            SelectedBackend::LlamaCpp => {
                self.llama_backend
                    .as_ref()
                    .ok_or(EngineSessionError::MissingLlamaCppConfig {
                        selected_backend: self.resolved_backend.selected_backend,
                    })?;
            }
            SelectedBackend::MlxLmDelegated => {
                self.mlx_lm_backend
                    .as_ref()
                    .ok_or(EngineSessionError::MissingMlxLmConfig)?;
            }
        }

        Ok(())
    }

    pub fn runtime_report(&self) -> RuntimeReport {
        let mut runtime =
            RuntimeReport::from_resolution(&self.backend_policy, &self.resolved_backend)
                .with_mlx_runtime(self.mlx_runtime_report());
        if let Some(llama_backend) = self.llama_backend.as_ref() {
            runtime.capabilities = CapabilityReport::for_llama_cpp_backend(llama_backend);
        }
        if let Some(mlx_lm_backend) = self.mlx_lm_backend.as_ref() {
            runtime.capabilities = CapabilityReport::for_mlx_lm_backend(mlx_lm_backend);
        }
        runtime
    }

    pub fn mlx_runtime_artifacts_dir(&self) -> Option<&Path> {
        self.mlx_runtime_artifacts_dir.as_deref()
    }

    pub fn mlx_runtime_artifacts_source(&self) -> Option<NativeRuntimeArtifactsSource> {
        self.mlx_runtime_artifacts_source
    }

    pub fn llama_runtime_report(
        &self,
        llama_backend: &LlamaCppConfig,
        fallback_reason: impl Into<String>,
    ) -> RuntimeReport {
        let resolved_backend =
            ResolvedBackend::llama_cpp(SelectedBackend::LlamaCpp, fallback_reason);
        let mut runtime = RuntimeReport::from_resolution(&self.backend_policy, &resolved_backend);
        runtime.capabilities = CapabilityReport::for_llama_cpp_backend(llama_backend);
        runtime
    }
    fn mlx_runtime_report(&self) -> Option<NativeRuntimeReport> {
        if self.resolved_backend.selected_backend != SelectedBackend::Mlx {
            return None;
        }

        if self.mlx_runtime_artifacts_dir().is_some() {
            Some(NativeRuntimeReport::metal_bringup(
                self.mlx_runtime_artifacts_source()
                    .unwrap_or(NativeRuntimeArtifactsSource::ExplicitConfig),
            ))
        } else {
            None
        }
    }
}

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
                let (runtime, stream, route_backend) = start_llama_cpp_stream_prevalidated(
                    &self.config,
                    runtime,
                    request_id,
                    &request,
                )?;
                Ok(build_llama_cpp_stream_state(
                    request_id,
                    request,
                    runtime,
                    stream,
                    route_backend,
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct MlxRuntimeArtifactsSelection {
    dir: PathBuf,
    source: NativeRuntimeArtifactsSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MlxModelArtifactsSelection {
    dir: PathBuf,
    source: NativeModelArtifactsSource,
}

#[derive(Clone, Debug)]
struct PreparedEngineSessionConfig {
    config: EngineSessionConfig,
    ephemeral_mlx_model_artifacts_dir: Option<PathBuf>,
}

fn resolve_default_mlx_runtime_artifacts_selection(
    explicit_dir: Option<PathBuf>,
    current_dir: Option<&Path>,
) -> Option<MlxRuntimeArtifactsSelection> {
    explicit_dir
        .map(|dir| MlxRuntimeArtifactsSelection {
            dir,
            source: NativeRuntimeArtifactsSource::ExplicitEnv,
        })
        .or_else(|| current_dir.and_then(detect_repo_owned_mlx_runtime_artifacts_dir_from))
}

fn detect_repo_owned_mlx_runtime_artifacts_dir_from(
    start_dir: &Path,
) -> Option<MlxRuntimeArtifactsSelection> {
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
            return Some(MlxRuntimeArtifactsSelection {
                dir: build_dir,
                source: NativeRuntimeArtifactsSource::RepoAutoDetect,
            });
        }
    }

    None
}

fn prepare_engine_session_config(
    config: EngineSessionConfig,
) -> Result<PreparedEngineSessionConfig, EngineSessionError> {
    Ok(PreparedEngineSessionConfig {
        config,
        ephemeral_mlx_model_artifacts_dir: None,
    })
}

fn cleanup_ephemeral_mlx_model_artifacts_dir(path: Option<&Path>) {
    if let Some(path) = path {
        let _ = fs::remove_dir_all(path);
    }
}

#[derive(Debug)]
pub struct EngineSession {
    core: EngineCore,
    config: EngineSessionConfig,
    runtime: RuntimeReport,
    next_request_id: u64,
    ephemeral_mlx_model_artifacts_dir: Option<PathBuf>,
    native_request_routes: BTreeMap<u64, GenerateRouteReport>,
    native_route_report_order: VecDeque<u64>,
    llama_requests: BTreeMap<u64, LlamaCppLifecycleRequestSlot>,
    llama_terminal_request_order: VecDeque<u64>,
}

#[derive(Debug)]
pub struct GenerateStream<'a> {
    session: &'a mut EngineSession,
    state: GenerateStreamState,
}

#[derive(Debug)]
pub enum GenerateStreamState {
    Native(Box<NativeGenerateStreamState>),
    LlamaCpp(Box<LlamaCppGenerateStreamState>),
    MlxLm(Box<MlxLmGenerateStreamState>),
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
pub struct LlamaCppGenerateStreamState {
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
    stream: LlamaCppStreamHandle,
    phase: GenerateStreamPhase,
}

#[derive(Debug)]
pub struct MlxLmGenerateStreamState {
    request_id: u64,
    runtime: RuntimeReport,
    current_report: SessionRequestReport,
    prompt_text: Option<String>,
    output_text: String,
    prompt_token_count: Option<u32>,
    output_token_count: Option<u32>,
    step_count: u64,
    ttft_step: Option<u64>,
    stream: MlxLmStreamHandle,
    phase: GenerateStreamPhase,
}

#[derive(Debug)]
enum LlamaCppLifecycleRequestSlot {
    Active(Box<LlamaCppLifecycleRequest>),
    Terminal(Box<SessionRequestReport>),
}

#[derive(Debug)]
struct LlamaCppLifecycleRequest {
    request_id: u64,
    current_report: SessionRequestReport,
    prompt_token_count: Option<u32>,
    output_token_count: Option<u32>,
    cached_prompt_tokens_observed: u32,
    prefix_hit_recorded: bool,
    step_count: u64,
    ttft_step: Option<u64>,
    stream: LlamaCppStreamHandle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GenerateStreamPhase {
    Request,
    Step,
    Done,
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
        if !self.native_request_routes.contains_key(&request_id) {
            self.native_route_report_order.push_back(request_id);
        }
        self.native_request_routes.insert(request_id, route);

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
        let (_runtime, stream, route_backend) =
            self.llama_cpp_stream_start(request_id, &request)?;
        let route = llama_cpp_stream_route(route_backend);
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

        let (runtime, stream, route_backend) = self.llama_cpp_stream_start(request_id, &request)?;
        Ok(build_llama_cpp_stream_state(
            request_id,
            request,
            runtime,
            stream,
            route_backend,
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
        let PreparedEngineSessionConfig {
            config,
            ephemeral_mlx_model_artifacts_dir,
        } = prepare_engine_session_config(config)?;
        if let Err(error) = config.validate() {
            cleanup_ephemeral_mlx_model_artifacts_dir(ephemeral_mlx_model_artifacts_dir.as_deref());
            return Err(error);
        }
        let core = match build_native_core(&config) {
            Ok(core) => core,
            Err(error) => {
                cleanup_ephemeral_mlx_model_artifacts_dir(
                    ephemeral_mlx_model_artifacts_dir.as_deref(),
                );
                return Err(error);
            }
        };
        let runtime = config
            .runtime_report()
            .with_mlx_model(resolve_native_model_report(&config, &core));
        Ok(Self {
            core,
            config,
            runtime,
            next_request_id: 1,
            ephemeral_mlx_model_artifacts_dir,
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

fn next_llama_cpp_stream_event(
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
    fn new(session: &'a mut EngineSession, state: GenerateStreamState) -> Self {
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

impl LlamaCppLifecycleRequestSlot {
    fn report(&self) -> SessionRequestReport {
        match self {
            Self::Active(request) => request.current_report.clone(),
            Self::Terminal(report) => report.as_ref().clone(),
        }
    }
}

struct LlamaCppChunkApplyResult {
    step: EngineStepReport,
    delta_tokens: Vec<u32>,
    delta_text: String,
    request: SessionRequestReport,
    stop: bool,
}

impl LlamaCppLifecycleRequest {
    fn new(
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

    fn step_report(
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
    fn drain_trailing_usage(&mut self) {
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

    fn cancel(&mut self) -> SessionRequestReport {
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

impl LlamaCppGenerateStreamState {
    fn new(
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

    fn step_event_from_chunk(&mut self, chunk: LlamaCppStreamChunk) -> GenerateStreamEvent {
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
    fn new(
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

    fn step_event_from_chunk(&mut self, chunk: MlxLmStreamChunkResult) -> GenerateStreamEvent {
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

fn next_mlx_lm_stream_event(
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
                        response: crate::generate::GenerateResponse {
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
                )))
            }
        },
        GenerateStreamPhase::Done => Ok(None),
    }
}

fn build_mlx_lm_stream_state(
    request_id: u64,
    request: GenerateRequest,
    runtime: RuntimeReport,
    stream: MlxLmStreamHandle,
) -> GenerateStreamState {
    let route = crate::generate::GenerateRouteReport {
        execution_plan: Some(MLX_LM_STREAM_EXECUTION_PLAN.to_string()),
        attention_route: None,
        kv_mode: None,
        prefix_cache_path: None,
        barrier_mode: None,
        crossover_decisions: Default::default(),
    };
    let current_report = SessionRequestReport {
        request_id,
        model_id: request.model_id.clone(),
        state: SessionRequestState::Waiting,
        prompt_tokens: request.input_tokens.clone(),
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

    GenerateStreamState::MlxLm(Box::new(MlxLmGenerateStreamState::new(
        request_id,
        runtime,
        current_report,
        request.input_text,
        stream,
    )))
}

fn apply_llama_cpp_stream_chunk(
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

fn llama_cpp_stream_route(_selected_backend: SelectedBackend) -> GenerateRouteReport {
    let execution_plan = LLAMA_CPP_STREAM_EXECUTION_PLAN;

    GenerateRouteReport {
        execution_plan: Some(execution_plan.to_string()),
        attention_route: None,
        kv_mode: None,
        prefix_cache_path: None,
        barrier_mode: None,
        crossover_decisions: Default::default(),
    }
}

fn apply_native_step_route_to_report(report: &mut SessionRequestReport, step: &EngineStepReport) {
    if let Some(route) = step.route.as_ref() {
        report.route = route.clone();
    }
}

fn apply_llama_cpp_prompt_progress(
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

fn apply_llama_cpp_usage_counts(
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

fn finish_reason_from_stop_type(
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
        None => Some(crate::generate::GenerateFinishReason::MaxOutputTokens),
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

impl Drop for EngineSession {
    fn drop(&mut self) {
        cleanup_ephemeral_mlx_model_artifacts_dir(
            self.ephemeral_mlx_model_artifacts_dir.as_deref(),
        );
    }
}

#[derive(Debug, Error)]
pub enum EngineSessionError {
    #[error(transparent)]
    BackendContract(#[from] BackendContractError),
    #[error(transparent)]
    LlamaCpp(#[from] LlamaCppBackendError),
    #[error(transparent)]
    MlxLm(#[from] MlxLmBackendError),
    #[error("max_batch_tokens must be greater than zero")]
    InvalidMaxBatchTokens,
    #[error("generate request requires input_tokens or input_text")]
    EmptyInputTokens,
    #[error("generate request max_output_tokens must be greater than zero")]
    InvalidMaxOutputTokens,
    #[error(
        "MLX preview session only accepts pre-tokenized input_tokens; input_text requires a llama.cpp backend"
    )]
    MlxBackendRequiresTokenizedInput,
    #[error(
        "MLX mode requires validated Metal runtime artifacts; deterministic fallback is internal-only and must be explicitly enabled"
    )]
    MlxRuntimeArtifactsRequired,
    #[error(
        "MLX runtime is not available; enable MLX mode support or configure a llama.cpp backend"
    )]
    MlxRuntimeUnavailable,
    #[error("request_id must be greater than zero")]
    InvalidRequestId,
    #[error("unsupported support tier cannot start an engine session")]
    UnsupportedSupportTier,
    #[error(
        "AX Engine v4 requires Apple M4-or-newer CPU/GPU; detected unsupported host {detected_host}. Set AX_ALLOW_UNSUPPORTED_HOST=1 only for internal development or CI bring-up."
    )]
    UnsupportedHostHardware { detected_host: String },
    #[error("llama.cpp backend {selected_backend:?} requires llama_backend config")]
    MissingLlamaCppConfig { selected_backend: SelectedBackend },
    #[error("mlx-lm delegated backend requires mlx_lm_backend config")]
    MissingMlxLmConfig,
    #[error("delegated backend {selected_backend:?} is missing runtime report")]
    MissingDelegatedRuntime { selected_backend: SelectedBackend },
    #[error(
        "llama.cpp backend {selected_backend:?} does not support {operation} in the preview SDK session yet"
    )]
    LlamaCppDoesNotSupportLifecycle {
        selected_backend: SelectedBackend,
        operation: &'static str,
    },
    #[error("mlx-lm delegated backend does not support streaming in this preview contract")]
    MlxLmDoesNotSupportStreaming,
    #[error("mlx-lm delegated backend does not support {operation} in the preview SDK lifecycle")]
    MlxLmDoesNotSupportLifecycle { operation: &'static str },
    #[error(
        "stateless streaming is not supported for the native MLX backend ({selected_backend:?}); \
         use EngineSession streaming methods instead"
    )]
    NativeBackendStatelessStreamNotSupported { selected_backend: SelectedBackend },
    #[error(
        "llama.cpp stream for request {request_id} on backend {selected_backend:?} ended before a terminal stop marker"
    )]
    LlamaCppStreamEndedBeforeStop {
        request_id: u64,
        selected_backend: SelectedBackend,
    },
    #[error("request {request_id} is missing from session state")]
    MissingRequestSnapshot { request_id: u64 },
    #[error(
        "stream for request {request_id} ended after {observed_event_count} events without a final response"
    )]
    StreamEndedWithoutResponse {
        request_id: u64,
        observed_event_count: u64,
    },
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
    #[error("embedding is not supported by the active backend")]
    EmbeddingNotSupported,
    #[error("embedding failed: {message}")]
    EmbeddingFailed { message: &'static str },
}

fn build_native_core(config: &EngineSessionConfig) -> Result<EngineCore, EngineSessionError> {
    #[cfg(feature = "mlx-native")]
    if config.resolved_backend.selected_backend == SelectedBackend::Mlx {
        return build_mlx_core(config);
    }

    if !config.resolved_backend.selected_backend.is_mlx() {
        return Ok(EngineCore::with_kv_config(config.kv_config));
    }

    Err(EngineSessionError::MlxRuntimeUnavailable)
}

#[cfg(feature = "mlx-native")]
fn build_mlx_core(config: &EngineSessionConfig) -> Result<EngineCore, EngineSessionError> {
    use ax_engine_core::{DeterministicSampler, NativeModelArtifacts};
    use ax_engine_mlx::{MlxRunner, generate::DEFAULT_PREFILL_CHUNK};

    let model_dir = config
        .mlx_model_artifacts_dir()
        .ok_or(EngineSessionError::MlxRuntimeArtifactsRequired)?;

    let artifacts = NativeModelArtifacts::from_dir(model_dir)
        .map_err(|e| EngineSessionError::MetalRuntime(e.into()))?;

    let runner = MlxRunner::from_artifacts(
        &artifacts,
        DEFAULT_PREFILL_CHUNK,
        config.mlx_disable_ngram_acceleration,
        config.mlx_kv_compression,
    )
    .map_err(|e| {
        EngineSessionError::MetalRuntime(ax_engine_core::MetalRuntimeError::Generic(e.to_string()))
    })?;

    Ok(EngineCore::with_runtime_components(
        config.kv_config,
        runner,
        DeterministicSampler,
    ))
}

fn build_llama_cpp_stream_state(
    request_id: u64,
    request: GenerateRequest,
    runtime: RuntimeReport,
    stream: LlamaCppStreamHandle,
    route_backend: SelectedBackend,
) -> GenerateStreamState {
    let route = llama_cpp_stream_route(route_backend);
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

    GenerateStreamState::LlamaCpp(Box::new(LlamaCppGenerateStreamState::new(
        request_id,
        runtime,
        current_report,
        request.input_text,
        stream,
    )))
}

fn run_delegated_generate_with_config(
    config: &EngineSessionConfig,
    request_id: u64,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    let runtime = config.runtime_report();
    run_delegated_generate_prevalidated(config, &runtime, request_id, request)
}

fn run_delegated_generate_prevalidated(
    config: &EngineSessionConfig,
    runtime: &RuntimeReport,
    request_id: u64,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    match config.resolved_backend.selected_backend {
        SelectedBackend::LlamaCpp => {
            run_llama_cpp_generate_prevalidated(config, runtime, request_id, request)
        }
        SelectedBackend::MlxLmDelegated => {
            let mlx_lm_backend = config
                .mlx_lm_backend
                .as_ref()
                .ok_or(EngineSessionError::MissingMlxLmConfig)?;
            run_mlx_lm_generate(request_id, runtime, mlx_lm_backend, request)
                .map_err(EngineSessionError::from)
        }
        SelectedBackend::Mlx => {
            let mut session = EngineSession::new(config.clone())?;
            session.generate_with_request_id(request_id, request.clone())
        }
    }
}

fn run_llama_cpp_generate_prevalidated(
    config: &EngineSessionConfig,
    runtime: &RuntimeReport,
    request_id: u64,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    let llama_backend =
        config
            .llama_backend
            .as_ref()
            .ok_or(EngineSessionError::MissingLlamaCppConfig {
                selected_backend: config.resolved_backend.selected_backend,
            })?;

    run_blocking_generate(request_id, runtime, llama_backend, request)
        .map_err(EngineSessionError::from)
}

fn start_llama_cpp_stream_prevalidated(
    config: &EngineSessionConfig,
    runtime: &RuntimeReport,
    _request_id: u64,
    request: &GenerateRequest,
) -> Result<(RuntimeReport, LlamaCppStreamHandle, SelectedBackend), EngineSessionError> {
    let llama_backend =
        config
            .llama_backend
            .as_ref()
            .ok_or(EngineSessionError::MissingLlamaCppConfig {
                selected_backend: config.resolved_backend.selected_backend,
            })?;

    start_streaming_generate(runtime, llama_backend, request)
        .map(|stream| {
            (
                runtime.clone(),
                stream,
                config.resolved_backend.selected_backend,
            )
        })
        .map_err(EngineSessionError::from)
}

fn resolve_native_model_report(
    config: &EngineSessionConfig,
    core: &EngineCore,
) -> Option<NativeModelReport> {
    let source = config.mlx_model_artifacts_source?;
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
        AX_NATIVE_MODEL_MANIFEST_FILE, AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
        DeterministicSampler, ExecutionRunner, ExecutionStatus, KvWriteSummary,
        MetalBinaryArchiveInfo, MetalBinaryArchiveState, MetalBuildDoctorReport,
        MetalBuildHostReport, MetalBuildReport, MetalBuildStatus, MetalBuildToolStatus,
        MetalBuildToolchainReport, MetalCommandBufferStatus, MetalDispatchArenaInfo,
        MetalDispatchKernelTrace, MetalDispatchTrace, MetalDispatchWorkload, MetalKernelManifest,
        MetalKernelSpec, MetalKernelTier, MetalThreadgroupSize, ModelId,
        NativeLinearAttentionConfig, NativeModelArtifacts, NativeModelArtifactsSummary,
        NativeModelManifest, NativeTensorDataType, NativeTensorFormat, NativeTensorRole,
        NativeTensorSpec, PHASE1_METAL_BUILD_GATE, PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION,
        PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION, PHASE1_METAL_LANGUAGE_STANDARD,
        PHASE1_METAL_LIBRARY_NAME, PHASE1_MLX_METAL_TARGET, RequestExecutionUpdate, RunnerInput,
        RunnerOutput, SamplingParams, SequenceNo,
    };
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::generate::GenerateFinishReason;

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

    fn mlx_test_session_config() -> EngineSessionConfig {
        EngineSessionConfig {
            // Explicitly clear auto-detected repo artifacts because this test
            // injects a custom in-memory runner instead of building MLX artifacts.
            mlx_runtime_artifacts_dir: None,
            mlx_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        }
    }

    fn sample_terminal_llama_report(request_id: u64) -> SessionRequestReport {
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
            execution_plan_ref: Some(LLAMA_CPP_STREAM_EXECUTION_PLAN.to_string()),
            route: llama_cpp_stream_route(SelectedBackend::LlamaCpp),
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
    struct TerminalRouteReportingRunner {
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

    impl ExecutionRunner for TerminalRouteReportingRunner {
        fn run(&self, input: RunnerInput) -> RunnerOutput {
            let request_updates = input
                .execution_batch
                .items
                .iter()
                .map(|item| RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: item.scheduled_token_count,
                    output_token: Some(42),
                    stop_reason: Some(ax_engine_core::StopReason::MaxOutputTokens),
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
                    source_quantization: None,
                    runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
                    layer_count: 36,
                    tensor_count: 512,
                    tie_word_embeddings: false,
                    is_moe: false,
                    is_hybrid_attention: false,
                    hybrid_full_attention_interval: None,
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
                direct_decode_native_residual_add_element_count: 0,
                direct_decode_cpu_residual_add_element_count: 0,
                prefix_native_residual_add_element_count: 0,
                prefix_cpu_residual_add_element_count: 0,
                prefix_native_scale_element_count: 0,
                prefix_cpu_scale_element_count: 0,
                direct_decode_native_scale_element_count: 0,
                direct_decode_cpu_scale_element_count: 0,
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

    fn write_json_file<T: serde::Serialize>(path: &Path, value: &T) {
        let parent = path.parent().expect("test JSON path should have a parent");
        fs::create_dir_all(parent).expect("test JSON parent should create");
        let body = serde_json::to_string_pretty(value).expect("test JSON should serialize");
        fs::write(path, body).expect("test JSON should write");
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn phase1_source_text() -> &'static str {
        "#include <metal_stdlib>\nusing namespace metal;\n"
    }

    fn phase1_kernel_specs() -> Vec<MetalKernelSpec> {
        vec![MetalKernelSpec {
            name: "ax_phase1_dense_path".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "test fixture".to_string(),
        }]
    }

    fn sample_build_doctor(
        bringup_allowed: bool,
        metal_toolchain_fully_available: bool,
    ) -> MetalBuildDoctorReport {
        MetalBuildDoctorReport {
            status: "pass".to_string(),
            bringup_allowed,
            mlx_runtime_ready: bringup_allowed && metal_toolchain_fully_available,
            metal_toolchain_fully_available,
            host: MetalBuildHostReport {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                detected_soc: Some("Apple M-series".to_string()),
                supported_mlx_runtime: true,
                unsupported_host_override_active: false,
            },
            metal_toolchain: MetalBuildToolchainReport {
                fully_available: metal_toolchain_fully_available,
                metal: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("test-metal".to_string()),
                },
                metallib: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("test-metallib".to_string()),
                },
                metal_ar: MetalBuildToolStatus {
                    available: metal_toolchain_fully_available,
                    version: Some("test-metal-ar".to_string()),
                },
            },
        }
    }

    fn read_http_request(stream: &mut impl Read) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer).expect("request should read");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            if let Some(header_end) = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
            {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("Content-Length:")
                            .or_else(|| line.strip_prefix("content-length:"))
                            .and_then(|value| value.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if request.len() >= header_end + content_length {
                    break;
                }
            }
        }
        request
    }

    fn write_repo_owned_mlx_runtime_status_fixture(
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

    fn write_valid_repo_owned_mlx_runtime_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
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
            mlx_target: PHASE1_MLX_METAL_TARGET.to_string(),
            metal_language_standard: PHASE1_METAL_LANGUAGE_STANDARD.to_string(),
            library_name: PHASE1_METAL_LIBRARY_NAME.to_string(),
            default_block_size_tokens: PHASE1_DEFAULT_BLOCK_SIZE_TOKENS,
            supported_block_size_tokens: PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS.to_vec(),
            source_file: std::path::PathBuf::from("metal/kernels/phase1_dense_path.metal"),
            toolchain_requirements: ["xcrun metal", "xcrun metallib"]
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
            mlx_target: manifest.mlx_target.clone(),
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

    fn write_valid_native_model_fixture_into(root_dir: &Path) {
        fs::create_dir_all(root_dir).expect("native model fixture directory should create");
        fs::write(root_dir.join("model.safetensors"), vec![0_u8; 4096])
            .expect("native model weights should write");
        let manifest = NativeModelManifest {
            schema_version: AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
            layer_count: 1,
            hidden_size: 2048,
            intermediate_size: 11008,
            attention_head_count: 16,
            attention_head_dim: 128,
            kv_head_count: 8,
            vocab_size: 151936,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: std::collections::BTreeMap::new(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            mla_attention: Default::default(),
            moe: ax_engine_core::NativeMoeConfig::default(),
            glm_router: Default::default(),
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
    }

    fn write_valid_native_model_fixture() -> std::path::PathBuf {
        let root_dir = unique_test_dir("native-model-valid-fixture");
        write_valid_native_model_fixture_into(&root_dir);
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
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    fn fake_llama_cpp_script() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ax-engine-session-llama-cpp-{unique}.py"));
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
    fn default_mlx_runtime_artifacts_dir_prefers_explicit_env_path() {
        let (repo_root, _build_dir) = write_valid_repo_owned_mlx_runtime_fixture();
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");
        let explicit_dir = unique_test_dir("explicit-metal-build-dir");

        let detected = resolve_default_mlx_runtime_artifacts_selection(
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
    fn default_mlx_runtime_artifacts_dir_ignores_non_compiled_repo_fixture() {
        let (repo_root, _) =
            write_repo_owned_mlx_runtime_status_fixture("skipped_toolchain_unavailable");
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");

        let detected = resolve_default_mlx_runtime_artifacts_selection(None, Some(&nested_dir));

        assert_eq!(detected, None);
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn default_mlx_runtime_artifacts_dir_ignores_invalid_compiled_repo_fixture() {
        let (repo_root, _) = write_repo_owned_mlx_runtime_status_fixture("compiled");
        let nested_dir = repo_root.join("crates/ax-engine-sdk/src");
        fs::create_dir_all(&nested_dir).expect("fixture nested directory should create");

        let detected = resolve_default_mlx_runtime_artifacts_selection(None, Some(&nested_dir));

        assert_eq!(detected, None);
        let _ = fs::remove_dir_all(repo_root);
    }

    #[test]
    fn mlx_runtime_report_marks_explicit_config_metal_runner() {
        let report = EngineSessionConfig {
            mlx_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
            mlx_runtime_artifacts_source: None,
            ..EngineSessionConfig::default()
        }
        .runtime_report();

        assert_eq!(
            report.mlx_runtime,
            Some(NativeRuntimeReport::metal_bringup(
                NativeRuntimeArtifactsSource::ExplicitConfig,
            ))
        );
    }

    #[test]
    fn resolve_native_model_report_uses_runner_owned_validated_summary() {
        let config = EngineSessionConfig {
            mlx_model_artifacts_dir: Some(PathBuf::from("/tmp/ax-model")),
            mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
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
                    source_quantized_binding_count: 3,
                    source_q4_k_binding_count: 1,
                    source_q5_k_binding_count: 1,
                    source_q6_k_binding_count: 1,
                    source_q8_0_binding_count: 0,
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
                source_quantization: None,
                runtime_status: ax_engine_core::model::NativeRuntimeStatus::default(),
                layer_count: 1,
                tensor_count: 9,
                tie_word_embeddings: false,
                is_moe: false,
                is_hybrid_attention: false,
                hybrid_full_attention_interval: None,
                bindings_prepared: true,
                buffers_bound: true,
                buffer_count: 9,
                buffer_bytes: 288,
                source_quantized_binding_count: 3,
                source_q4_k_binding_count: 1,
                source_q5_k_binding_count: 1,
                source_q6_k_binding_count: 1,
                source_q8_0_binding_count: 0,
            }
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn preview_session_config_factory_builds_mlx_preview_defaults() {
        let config =
            EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest::default())
                .expect("preview config factory should build MLX defaults");

        assert_eq!(
            config.kv_config,
            KvManagerConfig::new(CacheGroupId(0), 16, 1024)
        );
        assert!(config.deterministic);
        assert_eq!(config.max_batch_tokens, 2048);
        assert_eq!(config.backend_policy, BackendPolicy::mlx_only());
        assert_eq!(config.resolved_backend, ResolvedBackend::mlx_preview());
        assert!(config.llama_backend.is_none());
    }

    #[test]
    fn preview_session_config_factory_builds_llama_cpp_server_backend() {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::LlamaCpp,
                llama_server_url: Some("http://127.0.0.1:8080".to_string()),
                ..PreviewBackendRequest::default()
            },
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview config factory should build llama.cpp config");

        assert_eq!(config.backend_policy, BackendPolicy::allow_llama_cpp());
        assert_eq!(config.resolved_backend.support_tier, SupportTier::LlamaCpp);
        assert_eq!(
            config.resolved_backend.selected_backend,
            SelectedBackend::LlamaCpp
        );
        assert!(config.llama_backend.is_some());
    }

    #[test]
    fn preview_session_config_factory_builds_mlx_lm_delegated_backend() {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            backend_request: PreviewBackendRequest {
                support_tier: SupportTier::MlxLmDelegated,
                mlx_lm_server_url: Some("http://127.0.0.1:8090".to_string()),
                ..PreviewBackendRequest::default()
            },
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview config factory should build mlx-lm delegated config");

        assert_eq!(
            config.backend_policy,
            BackendPolicy::allow_mlx_lm_delegated()
        );
        assert_eq!(
            config.resolved_backend.selected_backend,
            SelectedBackend::MlxLmDelegated
        );
        assert_eq!(
            config.resolved_backend.support_tier,
            SupportTier::MlxLmDelegated
        );
        assert!(config.llama_backend.is_none());
        assert_eq!(
            config.mlx_lm_backend,
            Some(MlxLmConfig::server_completion("http://127.0.0.1:8090"))
        );

        let runtime = config.runtime_report();
        assert_eq!(runtime.selected_backend, SelectedBackend::MlxLmDelegated);
        assert_eq!(runtime.support_tier, SupportTier::MlxLmDelegated);
        assert!(runtime.capabilities.text_generation);
        assert!(runtime.capabilities.token_streaming);
        assert!(runtime.mlx_runtime.is_none());
    }

    #[test]
    fn preview_session_config_factory_preserves_explicit_native_artifact_dirs() {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            mlx_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
            mlx_model_artifacts_dir: Some(Path::new("/tmp/ax-model").to_path_buf()),
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview config factory should preserve explicit native artifact config");

        assert_eq!(
            config.mlx_runtime_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-metal"))
        );
        assert_eq!(
            config.mlx_runtime_artifacts_source,
            Some(NativeRuntimeArtifactsSource::ExplicitConfig)
        );
        assert_eq!(
            config.mlx_model_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-model"))
        );
        assert_eq!(
            config.mlx_model_artifacts_source,
            Some(NativeModelArtifactsSource::ExplicitConfig)
        );
    }

    #[test]
    fn preview_session_config_factory_preserves_explicit_gguf_model_path_source() {
        let config = EngineSessionConfig::from_preview_request(PreviewSessionConfigRequest {
            mlx_model_artifacts_dir: Some(
                Path::new("/tmp/google_gemma-4-26b-it-q4_k_m.gguf").to_path_buf(),
            ),
            ..PreviewSessionConfigRequest::default()
        })
        .expect("preview config factory should preserve explicit gguf model config");

        assert_eq!(
            config.mlx_model_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/google_gemma-4-26b-it-q4_k_m.gguf"))
        );
        assert_eq!(
            config.mlx_model_artifacts_source,
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
            max_batch_tokens: 4096,
            backend_policy: BackendPolicy::allow_llama_cpp(),
            resolved_backend: ResolvedBackend::llama_cpp(
                SelectedBackend::LlamaCpp,
                "MLX preview not ready for this model",
            ),
            llama_backend: Some(crate::llama_cpp::LlamaCppConfig::server_completion(
                "http://127.0.0.1:8080".to_string(),
            )),
            mlx_lm_backend: None,
            mlx_runtime_artifacts_dir: Some(Path::new("/tmp/ax-metal").to_path_buf()),
            mlx_runtime_artifacts_source: Some(NativeRuntimeArtifactsSource::ExplicitConfig),
            mlx_model_artifacts_dir: Some(Path::new("/tmp/ax-model").to_path_buf()),
            mlx_model_artifacts_source: Some(NativeModelArtifactsSource::ExplicitConfig),
            mlx_disable_ngram_acceleration: true,
            mlx_kv_compression: MlxKvCompressionConfig::turboquant_shadow(),
        });

        assert_eq!(
            config.kv_config,
            KvManagerConfig::new(CacheGroupId(7), 32, 2048)
        );
        assert!(!config.deterministic);
        assert_eq!(config.max_batch_tokens, 4096);
        assert_eq!(config.backend_policy, BackendPolicy::allow_llama_cpp());
        assert_eq!(
            config.resolved_backend,
            ResolvedBackend::llama_cpp(
                SelectedBackend::LlamaCpp,
                "MLX preview not ready for this model",
            )
        );
        assert!(config.llama_backend.is_some());
        assert_eq!(
            config.mlx_runtime_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-metal"))
        );
        assert_eq!(
            config.mlx_runtime_artifacts_source,
            Some(NativeRuntimeArtifactsSource::ExplicitConfig)
        );
        assert_eq!(
            config.mlx_model_artifacts_dir.as_deref(),
            Some(Path::new("/tmp/ax-model"))
        );
        assert_eq!(
            config.mlx_model_artifacts_source,
            Some(NativeModelArtifactsSource::ExplicitConfig)
        );
        assert!(config.mlx_disable_ngram_acceleration);
        assert_eq!(
            config.mlx_kv_compression,
            MlxKvCompressionConfig::turboquant_shadow()
        );
    }

    fn llama_cpp_server_session(server_url: String) -> EngineSession {
        let config = EngineSessionConfig {
            backend_policy: BackendPolicy::allow_llama_cpp(),
            resolved_backend: ResolvedBackend::llama_cpp(
                SelectedBackend::LlamaCpp,
                "MLX preview not ready for this model",
            ),
            llama_backend: Some(crate::llama_cpp::LlamaCppConfig::server_completion(
                server_url,
            )),
            ..EngineSessionConfig::default()
        };
        EngineSession::new(config).expect("llama.cpp session should build")
    }

    fn spawn_scripted_llama_cpp_completion_stream_server(
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

    fn spawn_llama_cpp_completion_stream_server(
        expected_requests: usize,
        chunks: Vec<Value>,
        assert_request: impl Fn(Value) + Send + Sync + 'static,
    ) -> (String, thread::JoinHandle<()>) {
        spawn_scripted_llama_cpp_completion_stream_server(expected_requests, move |_, payload| {
            assert_request(payload.clone());
            chunks.clone()
        })
    }

    // Same SSE wire format as spawn_llama_cpp_completion_stream_server; the difference is that the
    // caller supplies OpenAI-format JSON payloads ({"choices": [...]}) rather than llama.cpp-native
    // ones ({"content": ..., "stop": ...}).
    #[test]
    fn llama_cpp_stream_generate_supports_server_completion_adapter() {
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
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
        let mut session = llama_cpp_server_session(server_url);

        let events = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect("llama.cpp stream should start")
            .collect::<Result<Vec<_>, _>>()
            .expect("llama.cpp stream should complete");

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");

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
            Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
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
    fn llama_cpp_stream_generate_rejects_cli_fallback_adapter() {
        let script_path = fake_llama_cpp_script();
        let model_path =
            std::env::temp_dir().join("ax-engine-session-stream-cli-fallback-model.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_llama_cpp(),
            resolved_backend: ResolvedBackend::llama_cpp(
                SelectedBackend::LlamaCpp,
                "MLX preview not ready for this model",
            ),
            llama_backend: Some(crate::llama_cpp::LlamaCppConfig::new(
                script_path,
                &model_path,
            )),
            ..EngineSessionConfig::default()
        })
        .expect("llama.cpp session should build");

        let error = session
            .stream_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("cli fallback stream".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect_err("cli fallback streaming should fail closed");

        assert!(matches!(
            error,
            EngineSessionError::LlamaCpp(LlamaCppBackendError::StreamingNotSupported {
                selected_backend: SelectedBackend::LlamaCpp,
            })
        ));
    }

    #[test]
    fn llama_cpp_cli_submit_generate_fails_closed() {
        let script_path = fake_llama_cpp_script();
        let model_path = std::env::temp_dir().join("ax-engine-session-fake-model-lifecycle.gguf");
        fs::write(&model_path, "fake gguf").expect("fake model should be written");

        let mut session = EngineSession::new(EngineSessionConfig {
            backend_policy: BackendPolicy::allow_llama_cpp(),
            resolved_backend: ResolvedBackend::llama_cpp(
                SelectedBackend::LlamaCpp,
                "MLX preview not ready for this model",
            ),
            llama_backend: Some(crate::llama_cpp::LlamaCppConfig::new(
                script_path,
                &model_path,
            )),
            ..EngineSessionConfig::default()
        })
        .expect("llama.cpp session should build");

        let error = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: Vec::new(),
                input_text: Some("unsupported lifecycle".to_string()),
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect_err("llama.cpp CLI submit should fail closed");

        match error {
            EngineSessionError::LlamaCpp(LlamaCppBackendError::StreamingNotSupported {
                selected_backend,
            }) => {
                assert_eq!(selected_backend, SelectedBackend::LlamaCpp);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn llama_cpp_stepwise_lifecycle_supports_server_completion_adapter() {
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
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
        let mut session = llama_cpp_server_session(server_url);

        let request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect("llama.cpp submit should succeed");

        let initial = session
            .request_report(request_id)
            .expect("llama.cpp request should exist");
        assert_eq!(initial.state, SessionRequestState::Waiting);

        let first_step = session.step_report().expect("first step should succeed");
        assert_eq!(first_step.scheduled_requests, 1);
        assert_eq!(first_step.scheduled_tokens, 1);
        assert_eq!(first_step.ttft_events, 1);
        assert_eq!(
            first_step
                .route
                .as_ref()
                .and_then(|route| route.execution_plan.as_deref()),
            Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
        );

        let running = session
            .request_report(request_id)
            .expect("running llama.cpp request should exist");
        assert_eq!(running.state, SessionRequestState::Running);
        assert_eq!(running.output_tokens, vec![4]);

        let second_step = session.step_report().expect("second step should succeed");
        assert_eq!(second_step.scheduled_requests, 1);
        assert_eq!(second_step.scheduled_tokens, 1);
        assert_eq!(
            second_step
                .route
                .as_ref()
                .and_then(|route| route.execution_plan.as_deref()),
            Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
        );

        let terminal = session
            .request_report(request_id)
            .expect("terminal llama.cpp request should exist");
        assert_eq!(terminal.state, SessionRequestState::Finished);
        assert_eq!(terminal.output_tokens, vec![4, 5]);
        assert_eq!(
            terminal.execution_plan_ref.as_deref(),
            Some(LLAMA_CPP_STREAM_EXECUTION_PLAN)
        );

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn llama_cpp_stepwise_lifecycle_reports_delegated_prompt_cache_hits() {
        let (server_url, server_handle) =
            spawn_scripted_llama_cpp_completion_stream_server(1, |_, payload| {
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
        let mut session = llama_cpp_server_session(server_url);

        let request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect("llama.cpp submit should succeed");

        let progress_step = session.step_report().expect("progress step should succeed");
        assert_eq!(progress_step.scheduled_requests, 1);
        assert_eq!(progress_step.scheduled_tokens, 0);
        assert_eq!(progress_step.prefix_hits, 1);

        let progress_report = session
            .request_report(request_id)
            .expect("llama.cpp request should exist after progress");
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
            .expect("terminal llama.cpp request should exist");
        assert_eq!(terminal.state, SessionRequestState::Finished);
        assert_eq!(terminal.output_tokens, vec![4, 5]);
        assert_eq!(
            terminal.route.prefix_cache_path.as_deref(),
            Some("delegated_prompt_cache")
        );

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn llama_cpp_stepwise_lifecycle_advances_multiple_active_requests() {
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
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
        let mut session = llama_cpp_server_session(server_url);

        let first_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect("first llama.cpp submit should succeed");

        let second_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![7, 8, 9],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect("second llama.cpp submit should also succeed");

        let first_step = session
            .step_report()
            .expect("first aggregated step should succeed");
        assert_eq!(first_step.scheduled_requests, 2);
        assert_eq!(first_step.scheduled_tokens, 2);
        assert_eq!(first_step.ttft_events, 2);

        let first_running = session
            .request_report(first_request_id)
            .expect("first llama.cpp request should exist");
        assert_eq!(first_running.state, SessionRequestState::Running);
        assert_eq!(first_running.output_tokens, vec![4]);
        let second_running = session
            .request_report(second_request_id)
            .expect("second llama.cpp request should exist");
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
            .expect("llama.cpp server thread should finish");
    }

    #[test]
    fn llama_cpp_cancelled_request_does_not_block_other_active_requests() {
        let (server_url, server_handle) = spawn_llama_cpp_completion_stream_server(
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
        let mut session = llama_cpp_server_session(server_url);

        let first_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![1, 2, 3],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect("first llama.cpp submit should succeed");
        let second_request_id = session
            .submit_generate(GenerateRequest {
                model_id: "qwen3_dense".to_string(),
                input_tokens: vec![7, 8, 9],
                input_text: None,
                max_output_tokens: 2,
                sampling: Default::default(),
                stop_sequences: Vec::new(),
                metadata: None,
            })
            .expect("second llama.cpp submit should succeed");

        session
            .cancel_request(first_request_id)
            .expect("llama.cpp cancel should succeed");
        let cancelled = session
            .request_report(first_request_id)
            .expect("cancelled llama.cpp request should exist");
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
            .expect("remaining llama.cpp request should exist");
        assert_eq!(running.state, SessionRequestState::Running);
        assert_eq!(running.output_tokens, vec![4]);

        let second_step = session
            .step_report()
            .expect("terminal step after cancel should succeed");
        assert_eq!(second_step.scheduled_requests, 1);
        assert_eq!(second_step.scheduled_tokens, 1);

        let terminal = session
            .request_report(second_request_id)
            .expect("terminal llama.cpp request should exist");
        assert_eq!(terminal.state, SessionRequestState::Finished);
        assert_eq!(terminal.output_tokens, vec![4, 5]);

        server_handle
            .join()
            .expect("llama.cpp server thread should finish");
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
        let config = mlx_test_session_config();
        let mut session = EngineSession {
            core,
            runtime: config.runtime_report(),
            config,
            next_request_id: 2,
            ephemeral_mlx_model_artifacts_dir: None,
            native_request_routes: BTreeMap::new(),
            native_route_report_order: VecDeque::new(),
            llama_requests: BTreeMap::new(),
            llama_terminal_request_order: VecDeque::new(),
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
        let request_report = session
            .request_report(1)
            .expect("native request report should exist after step");
        assert_eq!(
            request_report.route.attention_route.as_deref(),
            Some("mock_native_attention")
        );
        assert_eq!(
            request_report
                .route
                .crossover_decisions
                .get("metal_dispatch_completed"),
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
    fn native_generate_response_surfaces_runner_route_metadata() {
        let core = EngineCore::with_runtime_components(
            KvManagerConfig::new(CacheGroupId(0), 16, 64),
            TerminalRouteReportingRunner {
                trace: sample_metal_dispatch_trace(),
            },
            DeterministicSampler,
        );
        let config = mlx_test_session_config();
        let mut session = EngineSession {
            core,
            runtime: config.runtime_report(),
            config,
            next_request_id: 2,
            ephemeral_mlx_model_artifacts_dir: None,
            native_request_routes: BTreeMap::new(),
            native_route_report_order: VecDeque::new(),
            llama_requests: BTreeMap::new(),
            llama_terminal_request_order: VecDeque::new(),
        };

        let response = session
            .generate_with_request_id(
                41,
                GenerateRequest {
                    model_id: "qwen3_dense".to_string(),
                    input_tokens: vec![1, 2, 3, 4],
                    input_text: None,
                    max_output_tokens: 1,
                    sampling: Default::default(),
                    stop_sequences: Vec::new(),
                    metadata: None,
                },
            )
            .expect("native blocking generate should succeed");

        assert_eq!(response.status, crate::generate::GenerateStatus::Finished);
        assert_eq!(response.step_count, 1);
        assert_eq!(
            response.route.attention_route.as_deref(),
            Some("mock_native_attention")
        );
        assert_eq!(
            response
                .route
                .crossover_decisions
                .get("metal_dispatch_completed"),
            Some(&1)
        );
    }

    #[test]
    fn slice_output_token_logprobs_fails_closed_on_length_mismatch() {
        let mut report = sample_terminal_llama_report(41);
        report.output_token_logprobs.pop();

        let error = slice_output_token_logprobs(&report, 0, 2)
            .expect_err("mismatched logprob lengths should fail closed");

        assert!(matches!(
            error,
            EngineSessionError::RequestReportInvariantViolation { request_id: 41, .. }
        ));
    }

    #[test]
    fn llama_cpp_terminal_requests_are_pruned_after_retention_limit() {
        let mut session = llama_cpp_server_session("http://127.0.0.1:1".to_string());

        for request_id in 1..=(MAX_LLAMA_CPP_TERMINAL_REQUESTS as u64 + 1) {
            session
                .store_terminal_llama_report(request_id, sample_terminal_llama_report(request_id));
        }

        assert!(session.request_report(1).is_none());
        assert!(
            session
                .request_report(MAX_LLAMA_CPP_TERMINAL_REQUESTS as u64 + 1)
                .is_some()
        );
        assert_eq!(
            session.llama_terminal_request_order.len(),
            MAX_LLAMA_CPP_TERMINAL_REQUESTS
        );
    }
}
