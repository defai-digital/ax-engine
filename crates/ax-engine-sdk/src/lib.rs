#![allow(
    clippy::collapsible_if,
    clippy::needless_update,
    clippy::too_many_arguments,
    clippy::unnecessary_lazy_evaluations
)]

//! Rust SDK facade for AX Engine runtimes.
//!
//! Most callers should build an [`EngineSessionConfig`] and use
//! [`EngineSession`] or [`StatelessGenerateContext`] from the crate root. The
//! root re-exports are the compatibility surface used by the server, Python
//! bindings, CLI tools, and downstream integrations.
//!
//! Backend-specific modules such as [`llama_cpp`] and [`mlx_lm`] remain public
//! for advanced configuration and delegated runtime wiring. Prefer the session
//! facade unless the integration needs backend-native request or stream types.
//! The internal `session/*` module tree owns lifecycle details while preserving
//! the existing crate-root exports.

pub use ax_engine_core::gemma4_unified::{
    Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedModality,
    Gemma4UnifiedRuntimeInputs, Gemma4UnifiedSoftTokenRange, Gemma4UnifiedTokenSpan,
    Gemma4UnifiedVideoRuntimeInput,
};
pub use ax_engine_core::{
    CacheGroupId, EmbeddingMatrix, EmbeddingPooling, KvCompressionConfig, KvCompressionMode,
    KvManagerConfig, RequestMultimodalInputs, RequestWorkloadHints, TurboQuantPreset,
};
#[allow(deprecated)]
pub use ax_engine_core::{MlxKvCompressionConfig, MlxKvCompressionMode, MlxTurboQuantPreset};
#[cfg(feature = "mlx-native")]
pub use ax_engine_mlx::MlxPrefixCacheStore;

pub mod backend;
mod delegated_http;
pub mod generate;
mod host;
pub mod llama_cpp;
pub mod mlx_lm;
pub mod request;
pub mod session;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;
#[cfg(feature = "tokenizer")]
pub use tokenizer::{EngineTokenizer, EngineTokenizerError};

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
    LlamaCppBackendError, LlamaCppChatGenerateRequest, LlamaCppChatMessage, LlamaCppCliConfig,
    LlamaCppConfig, LlamaCppServerCompletionConfig, LlamaCppStreamChunk, LlamaCppStreamHandle,
    run_blocking_chat_generate as run_blocking_llama_cpp_chat_generate,
    start_streaming_chat_generate as start_streaming_llama_cpp_chat_generate,
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
