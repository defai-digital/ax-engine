pub use ax_engine_core::{CacheGroupId, KvManagerConfig};

pub mod backend;
pub mod compat;
pub mod generate;
mod host;
pub mod request;
pub mod session;

pub use backend::{
    current_host_report, current_metal_toolchain_report, initial_native_mode_model_promotion,
    is_initial_native_mode_model_id, native_mode_model_requirement_message,
    preview_support_tier_from_label, resolve_preview_backend, BackendContractError, BackendPolicy,
    CapabilityLevel, CapabilityReport, CompatibilityBackendKind, HostReport, MetalToolchainReport,
    NativeModeModelPromotion, NativeModelArtifactsSource, NativeModelReport, NativeRunnerKind,
    NativeRuntimeArtifactsSource, NativeRuntimeReport, PreviewBackendMode, PreviewBackendRequest,
    PreviewBackendResolution, PreviewBackendResolutionError, ResolutionPolicy, ResolvedBackend,
    RuntimeReport, SelectedBackend, SupportTier, ToolStatusReport, INITIAL_NATIVE_MODE_MODEL_ID,
    NATIVE_MODE_MODEL_PROMOTIONS,
};
pub use compat::{
    CompatibilityBackendConfig, CompatibilityBackendError, LlamaCppCliConfig, LlamaCppConfig,
    LlamaCppServerCompletionConfig, MlxCliConfig, MlxConfig, OpenAiCompatibleServerConfig,
};
pub use generate::{
    GenerateFinishReason, GenerateRequest, GenerateResponse, GenerateRouteReport, GenerateSampling,
    GenerateStatus, GenerateStreamEvent, GenerateStreamRequestEvent, GenerateStreamResponseEvent,
    GenerateStreamStepEvent,
};
pub use request::{
    EngineStepReport, MetalDispatchKernelStepReport, MetalDispatchNumericStepReport,
    MetalDispatchStepReport, MetalDispatchValidationStepReport, SessionRequestReport,
    SessionRequestState,
};
pub use session::{
    classify_native_gguf_export_failure_message, is_gguf_path, EngineSession, EngineSessionConfig,
    EngineSessionError, GenerateStream, GenerateStreamState, NativeGgufExportFailureKind,
    PreviewSessionConfigError, PreviewSessionConfigRequest, ResolvedSessionConfigRequest,
    StatelessGenerateContext,
};
