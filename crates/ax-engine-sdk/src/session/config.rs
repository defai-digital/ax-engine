use std::env;
use std::path::{Path, PathBuf};

use ax_engine_core::{CacheGroupId, KvManagerConfig, KvManagerError};
use thiserror::Error;

use crate::backend::{
    BackendPolicy, CapabilityReport, NativeModelArtifactsSource, NativeRuntimeArtifactsSource,
    NativeRuntimeReport, PreviewBackendRequest, PreviewBackendResolutionError, ResolvedBackend,
    RuntimeReport, SelectedBackend, SupportTier, resolve_preview_backend,
};
use crate::host;
use crate::llama_cpp::LlamaCppConfig;
use crate::mlx_lm::MlxLmConfig;

use super::artifacts::{
    MlxModelArtifactsSelection, MlxRuntimeArtifactsSelection,
    resolve_default_mlx_runtime_artifacts_selection,
};
use super::errors::EngineSessionError;

const NATIVE_METAL_BUILD_DIR_ENV: &str = "AX_ENGINE_METAL_BUILD_DIR";
const NATIVE_MODEL_DIR_ENV: &str = "AX_ENGINE_MLX_MODEL_ARTIFACTS_DIR";

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
    /// When true, MLX MTP decode keeps MTP enabled but disables the n-gram-first
    /// draft source inside the MTP verify loop.
    pub mlx_mtp_disable_ngram_stacking: bool,
    /// Optional speculation-profile preset name (ADR-022): one of
    /// `auto`/`coding`/`agentic`/`chatbot`. `None` falls back to the
    /// `AX_MLX_SPECULATION_PROFILE` env / built-in `auto`.
    pub mlx_speculation_profile: Option<String>,
    /// Override the MLX runner's prefill chunk size. `None` keeps the
    /// runner's `DEFAULT_PREFILL_CHUNK` (2048), matching mlx_lm's default
    /// `prefill_step_size`. Setting this lets callers (benchmark harness,
    /// server CLI) align AX's prefill chunk geometry with upstream MLX for
    /// long-prompt comparisons. MLA models clamp through their own
    /// prefix-restore chunk policy.
    pub mlx_prefill_chunk: Option<usize>,
    /// Fair multi-prefill progress (design Track B / PR3). Default **OFF**.
    /// When true, text prefills are interleave-capped under residual budget.
    /// Does **not** enable GPU continuous batching or claim `partial_overlap`.
    pub multi_prefill_fair: bool,
    /// Per-request text prefill cap when fair mode is on. `0` means use
    /// `block_size_tokens` (see `EngineCore::set_multi_prefill_fair`).
    pub max_prefill_tokens_per_request_per_step: u32,
    /// Max concurrent text prefills admitted per step when fair mode is on.
    /// `0` means unlimited (subject to free-block headroom).
    pub max_inflight_prefill_requests: u32,
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
    /// When true, MLX MTP decode keeps MTP enabled but disables the n-gram-first
    /// draft source inside the MTP verify loop.
    pub mlx_mtp_disable_ngram_stacking: bool,
    /// Optional speculation-profile preset name (ADR-022): one of
    /// `auto`/`coding`/`agentic`/`chatbot`. `None` falls back to the
    /// `AX_MLX_SPECULATION_PROFILE` env / built-in `auto`.
    pub mlx_speculation_profile: Option<String>,
    /// Optional MLX prefill chunk override. `None` keeps the runner's
    /// `DEFAULT_PREFILL_CHUNK`. The bench harness uses this to match the
    /// `--prefill-step-size` mlx_lm and mlx-swift-lm receive so the three
    /// runtimes are compared on identical chunk geometry at long prompts.
    pub mlx_prefill_chunk: Option<usize>,
    /// Fair multi-prefill progress (default OFF). See `EngineSessionConfig`.
    pub multi_prefill_fair: bool,
    pub max_prefill_tokens_per_request_per_step: u32,
    pub max_inflight_prefill_requests: u32,
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
            mlx_mtp_disable_ngram_stacking: true,
            mlx_speculation_profile: None,
            mlx_prefill_chunk: None,
            multi_prefill_fair: false,
            max_prefill_tokens_per_request_per_step: 0,
            max_inflight_prefill_requests: 0,
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
    pub mlx_mtp_disable_ngram_stacking: bool,
    pub mlx_speculation_profile: Option<String>,
    pub mlx_prefill_chunk: Option<usize>,
    pub multi_prefill_fair: bool,
    pub max_prefill_tokens_per_request_per_step: u32,
    pub max_inflight_prefill_requests: u32,
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
            mlx_mtp_disable_ngram_stacking: default.mlx_mtp_disable_ngram_stacking,
            mlx_speculation_profile: default.mlx_speculation_profile.clone(),
            mlx_prefill_chunk: default.mlx_prefill_chunk,
            multi_prefill_fair: default.multi_prefill_fair,
            max_prefill_tokens_per_request_per_step: default
                .max_prefill_tokens_per_request_per_step,
            max_inflight_prefill_requests: default.max_inflight_prefill_requests,
        }
    }
}

#[derive(Debug, Error)]
pub enum PreviewSessionConfigError {
    #[error(transparent)]
    BackendResolution(#[from] PreviewBackendResolutionError),
    #[error(transparent)]
    KvConfig(#[from] KvManagerError),
}

impl Default for EngineSessionConfig {
    fn default() -> Self {
        let mlx_runtime_artifacts = Self::default_mlx_runtime_artifacts_selection();
        let mlx_model_artifacts = Self::default_mlx_model_artifacts_selection();
        Self {
            kv_config: KvManagerConfig::validated(CacheGroupId(0), 16, 1024),
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
            mlx_mtp_disable_ngram_stacking: true,
            mlx_speculation_profile: None,
            mlx_prefill_chunk: None,
            multi_prefill_fair: false,
            max_prefill_tokens_per_request_per_step: 0,
            max_inflight_prefill_requests: 0,
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

    /// Disables n-gram speculation, forcing the direct single-token decode path.
    /// Useful for benchmarking baseline throughput or diagnosing speculation overhead.
    pub fn without_ngram_acceleration(mut self) -> Self {
        self.mlx_disable_ngram_acceleration = true;
        self
    }

    /// Keeps MTP speculation enabled but disables the n-gram-first draft source
    /// inside the MTP verify loop. This is the default after the Gemma 4 12B
    /// Phase 4 sweep; call [`Self::with_mtp_ngram_stacking`] to opt in.
    pub fn without_mtp_ngram_stacking(mut self) -> Self {
        self.mlx_mtp_disable_ngram_stacking = true;
        self
    }

    /// Enables n-gram-first drafting inside the MTP verify loop. This is an
    /// opt-in for workloads where benchmark evidence shows stacking beats pure
    /// MTP.
    pub fn with_mtp_ngram_stacking(mut self) -> Self {
        self.mlx_mtp_disable_ngram_stacking = false;
        self
    }

    /// Opt into fair multi-prefill progress under residual budget (default OFF).
    ///
    /// `max_tokens_per_request_per_step == 0` uses `block_size_tokens`.
    /// `max_inflight_prefill_requests == 0` means unlimited (subject to free-block
    /// headroom). Multimodal prefills remain atomic.
    pub fn with_multi_prefill_fair(
        mut self,
        max_tokens_per_request_per_step: u32,
        max_inflight_prefill_requests: u32,
    ) -> Self {
        self.multi_prefill_fair = true;
        self.max_prefill_tokens_per_request_per_step = max_tokens_per_request_per_step;
        self.max_inflight_prefill_requests = max_inflight_prefill_requests;
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

        let kv_config = KvManagerConfig::new(
            request.cache_group_id,
            request.block_size_tokens,
            request.total_blocks,
        )?;

        Ok(Self {
            kv_config,
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
            mlx_mtp_disable_ngram_stacking: request.mlx_mtp_disable_ngram_stacking,
            mlx_speculation_profile: request.mlx_speculation_profile,
            mlx_prefill_chunk: request.mlx_prefill_chunk,
            multi_prefill_fair: request.multi_prefill_fair,
            max_prefill_tokens_per_request_per_step: request
                .max_prefill_tokens_per_request_per_step,
            max_inflight_prefill_requests: request.max_inflight_prefill_requests,
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
            kv_config: KvManagerConfig::validated(
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
            mlx_mtp_disable_ngram_stacking: request.mlx_mtp_disable_ngram_stacking,
            mlx_speculation_profile: request.mlx_speculation_profile,
            mlx_prefill_chunk: request.mlx_prefill_chunk,
            multi_prefill_fair: request.multi_prefill_fair,
            max_prefill_tokens_per_request_per_step: request
                .max_prefill_tokens_per_request_per_step,
            max_inflight_prefill_requests: request.max_inflight_prefill_requests,
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
