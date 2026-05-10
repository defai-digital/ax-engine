#![allow(
    clippy::collapsible_if,
    clippy::needless_update,
    clippy::too_many_arguments,
    clippy::unnecessary_lazy_evaluations
)]

pub use ax_engine_core::{
    CacheGroupId, EmbeddingPooling, KvManagerConfig, MlxKvCompressionConfig, MlxKvCompressionMode,
    MlxTurboQuantPreset,
};

pub mod backend;
mod delegated_http;
pub mod generate;
mod host;
pub mod llama_cpp;
pub mod mlx_lm;
pub mod request;
pub mod session;

pub use backend::{
    BackendContractError, BackendPolicy, CapabilityLevel, CapabilityReport, HostReport,
    MetalToolchainReport, NativeModelArtifactsSource, NativeModelReport, NativeRunnerKind,
    NativeRuntimeArtifactsSource, NativeRuntimeReport, PreviewBackendMode, PreviewBackendRequest,
    PreviewBackendResolution, PreviewBackendResolutionError, ResolutionPolicy, ResolvedBackend,
    RuntimeReport, SelectedBackend, SupportTier, ToolStatusReport, current_host_report,
    current_metal_toolchain_report, preview_support_tier_from_label, resolve_preview_backend,
};
pub use delegated_http::{
    DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS, DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS,
    DelegatedHttpTimeouts,
};
pub use generate::{
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateSampling,
    GenerateStatus, GenerateStreamEvent, GenerateStreamRequestEvent, GenerateStreamResponseEvent,
    GenerateStreamStepEvent,
};
pub use llama_cpp::{
    LlamaCppBackendError, LlamaCppCliConfig, LlamaCppConfig, LlamaCppServerCompletionConfig,
};
pub use mlx_lm::{
    MlxLmBackendError, MlxLmChatGenerateRequest, MlxLmChatMessage, MlxLmConfig,
    MlxLmServerCompletionConfig, MlxLmStreamChunkResult, MlxLmStreamHandle,
    finish_reason_from_mlx_lm, run_blocking_chat_generate, start_streaming_chat_generate,
};
pub use request::{
    EngineStepReport, MetalDispatchKernelStepReport, MetalDispatchNumericStepReport,
    MetalDispatchStepReport, MetalDispatchValidationStepReport, SessionRequestReport,
    SessionRequestState,
};
pub use session::{
    EngineSession, EngineSessionConfig, EngineSessionError, GenerateStream, GenerateStreamState,
    PreviewSessionConfigError, PreviewSessionConfigRequest, ResolvedSessionConfigRequest,
    StatelessGenerateContext,
};
