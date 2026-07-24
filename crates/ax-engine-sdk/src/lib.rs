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
    Gemma4UnifiedAudioInput, Gemma4UnifiedAudioProcessor, Gemma4UnifiedAudioRuntimeInput,
    Gemma4UnifiedError, Gemma4UnifiedImageInput, Gemma4UnifiedImageRuntimeInput,
    Gemma4UnifiedModality, Gemma4UnifiedProcessorConfig, Gemma4UnifiedRuntimeInputError,
    Gemma4UnifiedRuntimeInputs, Gemma4UnifiedSoftTokenRange, Gemma4UnifiedTokenSpan,
    Gemma4UnifiedVideoInput, Gemma4UnifiedVideoRuntimeInput, Gemma4UnifiedVisionProcessor,
};
pub use ax_engine_core::qwen3_vl::{
    Qwen3VlImageRuntimeInput, Qwen3VlRuntimeInputError, Qwen3VlRuntimeInputs,
};
pub use ax_engine_core::{
    CacheGroupId, EmbeddingMatrix, EmbeddingPooling, KvManagerConfig, NativeDiffusionConfig,
    NativeLinearAttentionConfig, NativeMlaAttentionConfig, RequestMultimodalInputError,
    RequestMultimodalInputs, RequestWorkloadHints, UNLIMITED_OCR_BASE_SOFT_TOKEN_COUNT,
    UnlimitedOcrImageRuntimeInput, UnlimitedOcrRuntimeInputError, UnlimitedOcrRuntimeInputs,
};
#[cfg(feature = "mlx-native")]
pub use ax_engine_mlx::MlxPrefixCacheStore;

pub mod backend;
mod delegated_http;
pub mod delegated_openai;
pub mod edge_llm;
pub mod generate;
mod host;
pub mod llama_cpp;
pub mod mlx_lm;
pub mod request;
pub mod session;
#[cfg(feature = "tokenizer")]
pub mod tokenizer;
pub mod vllm;
#[cfg(feature = "tokenizer")]
pub use tokenizer::{EngineTokenizer, EngineTokenizerError};

pub use backend::{
    BackendContractError, BackendPolicy, CapabilityLevel, CapabilityReport, DelegatedReadiness,
    DelegatedRuntimeReport, HostReport, MetalToolchainReport, NativeModelArtifactsSource,
    NativeModelReport, NativeRunnerKind, NativeRuntimeArtifactsSource, NativeRuntimeReport,
    NativeRuntimeStatus, NativeSourceQuantization, PreviewBackendMode, PreviewBackendRequest,
    PreviewBackendResolution, PreviewBackendResolutionError, RedactedEndpoint, ResolutionPolicy,
    ResolvedBackend, RuntimeReport, SelectedBackend, SupportTier, ToolStatusReport,
    current_host_report, current_metal_toolchain_report, preview_support_tier_from_label,
    resolve_preview_backend,
};
pub use delegated_http::{
    DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS, DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS,
    DEFAULT_DELEGATED_HTTP_MAX_ERROR_BODY_BYTES, DelegatedBearerCredential,
    DelegatedHttpConfigError, DelegatedHttpHeaders, DelegatedHttpRequestOptions,
    DelegatedHttpRetryPolicy, DelegatedHttpTimeouts, DelegatedProxyPolicy, DelegatedRedirectPolicy,
    DelegatedTlsPolicy,
};
pub use delegated_openai::{
    DEFAULT_MAX_DELEGATED_DATA_URI_ENCODED_BYTES, DEFAULT_MAX_DELEGATED_IMAGE_BYTES,
    DEFAULT_MAX_DELEGATED_IMAGE_PIXELS, DEFAULT_MAX_DELEGATED_IMAGES,
    DEFAULT_MAX_DELEGATED_SSE_FRAME_BYTES, DEFAULT_MAX_DELEGATED_TOTAL_PIXELS,
    DelegatedChatContent, DelegatedChatContentPart, DelegatedChatMessage, DelegatedChatRole,
    DelegatedImageLimits, DelegatedImageUrl, DelegatedOpenAiSseError, DelegatedOpenAiStreamChunk,
    DelegatedOpenAiStreamHandle, DelegatedOpenAiValidationError, ValidatedDataUri,
    validate_delegated_image_budget,
};
pub use edge_llm::{
    EdgeLlmBackendError, EdgeLlmChatContent, EdgeLlmChatContentPart, EdgeLlmChatGenerateRequest,
    EdgeLlmChatMessage, EdgeLlmConfig, EdgeLlmImageUrl, EdgeLlmServerCompletionConfig,
    EdgeLlmStagedImage, EdgeLlmStreamChunkResult, EdgeLlmStreamHandle, finish_reason_from_edge_llm,
    run_blocking_chat_generate as run_blocking_edge_llm_chat_generate,
    run_blocking_generate as run_blocking_edge_llm_generate,
    start_streaming_chat_generate as start_streaming_edge_llm_chat_generate,
    start_streaming_generate as start_streaming_edge_llm_generate,
};
pub use generate::{
    GenerateFinishReason, GenerateMtpReport, GeneratePerformanceReport, GenerateRequest,
    GenerateResponse, GenerateRouteReport, GenerateSampling, GenerateStatus, GenerateStreamEvent,
    GenerateStreamRequestEvent, GenerateStreamResponseEvent, GenerateStreamStepEvent,
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
    MlxMtpPolicy, PreviewSessionConfigError, PreviewSessionConfigRequest,
    ResolvedSessionConfigRequest, StatelessGenerateContext,
};
pub use vllm::{
    NormalizedDelegatedBaseUrl, OrderedFloat, UNLIMITED_OCR_DEFAULT_CONTEXT_LENGTH,
    UNLIMITED_OCR_DEFAULT_MAX_OUTPUT_TOKENS, VllmBackendError, VllmChatGenerateRequest, VllmConfig,
    VllmConfigError, VllmModelProfile, VllmReadiness, VllmReadinessReport, VllmRequestExtensions,
    VllmRequestValidationError, VllmServerCompletionConfig, VllmStreamChunkResult,
    VllmStreamHandle, VllmXargs, check_readiness as check_vllm_readiness, finish_reason_from_vllm,
    run_blocking_chat_generate as run_blocking_vllm_chat_generate,
    run_blocking_generate as run_blocking_vllm_generate,
    start_streaming_chat_generate as start_streaming_vllm_chat_generate,
    start_streaming_generate as start_streaming_vllm_generate,
};
