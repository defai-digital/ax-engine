#![allow(clippy::collapsible_if)]

pub mod convert;
pub mod engine;
pub mod execution_plan;
pub mod gguf;
pub mod ids;
pub mod kv;
pub mod metal;
pub mod model;
pub mod request;
pub mod request_manager;
pub mod runner;
pub mod sampling;
pub mod scheduler;

pub use engine::{EngineCore, EngineCoreError, EngineEvent, EngineStepOutcome, StepMetrics};
pub use execution_plan::{
    DeterministicExecutionPlanResolver, ExecutionPlanBinding, ExecutionPlanResolver,
};
pub use ids::{BlockId, CacheGroupId, ModelId, RequestId, SequenceNo, StepId};
pub use kv::{
    AllocationPlan, AllocationStatus, AppendMode, BlockTable, BlockTableView, FreeResult,
    KvManager, KvManagerConfig, KvManagerError, PrefixLookupResult,
};
pub use metal::{
    MetalAssetValidator, MetalBinaryArchiveInfo, MetalBinaryArchiveState, MetalBringupRunner,
    MetalBringupSampler, MetalBuildDoctorReport, MetalBuildHostReport, MetalBuildReport,
    MetalBuildStatus, MetalBuildToolStatus, MetalBuildToolchainReport, MetalCommandBufferStatus,
    MetalComputePipelineInfo, MetalDeviceInfo, MetalDispatchArenaInfo, MetalDispatchKernelTrace,
    MetalDispatchNumericLayout, MetalDispatchTrace, MetalDispatchWorkload, MetalKernelAssets,
    MetalKernelBinary, MetalKernelBuildArtifacts, MetalKernelBuildRequest, MetalKernelManifest,
    MetalKernelSpec, MetalKernelTier, MetalRuntimeBringup, MetalRuntimeBringupReport,
    MetalRuntimeError, MetalThreadgroupSize, PHASE1_METAL_BUILD_GATE,
    PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION, PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION,
    PHASE1_METAL_LANGUAGE_STANDARD, PHASE1_METAL_LIBRARY_NAME, PHASE1_MLX_METAL_TARGET,
    PHASE1_OPTIONAL_METAL_KERNELS, PHASE1_REQUIRED_METAL_KERNELS, build_phase1_kernel_artifacts,
};
pub use model::{
    AX_NATIVE_MODEL_MANIFEST_FILE, AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION,
    NativeLinearAttentionConfig, NativeModelArtifacts, NativeModelArtifactsSummary,
    NativeModelError, NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus,
    NativeTensorDataType, NativeTensorFormat, NativeTensorQuantization, NativeTensorRole,
    NativeTensorSpec,
};
pub use request::{
    RequestRecord, RequestSnapshot, RequestState, RequestSubmission, StateTransitionError,
};
pub use request_manager::{RequestManager, RequestManagerError};
pub use runner::{
    DeterministicRunner, EmbeddingPooling, ExecutionRunner, ExecutionStatus, KvWriteSummary,
    MlxKvCompressionConfig, MlxKvCompressionMode, MlxTurboQuantPreset, NativeModelBindingSummary,
    RequestExecutionUpdate, RequestLogitsOutput, ResolvedBlockTable, RunnerInput, RunnerOutput,
};
pub use sampling::{
    DeterministicSampler, SampledToken, SamplerInput, SamplerRequest, SamplingParams, StopReason,
    TokenSampler,
};
pub use scheduler::{
    ExecutionBatch, ExecutionItem, ExecutionMode, PositionRange,
    ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB, ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_CANDIDATE_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_DECODE_PATH,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ELIGIBLE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_COMPRESSED_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ESTIMATED_SAVED_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FULL_PRECISION_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_ATTEMPTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_CANDIDATES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACK_REASON,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_FALLBACKS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_FUSED_DECODE_SUCCESSES,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_HOT_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEY_BITS, ROUTE_DECISION_AX_MLX_KV_COMPRESSION_KEYS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRESET,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_BLOCKERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_PRODUCTION_READY,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RATIO_MILLI,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_ROUTE_METADATA_SCHEMA,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_KIB,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_TOKEN_LAYERS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_RUNTIME_STORAGE_WRITTEN_SLOTS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_CALLS,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_SHADOW_SYNC_WALL_US,
    ROUTE_DECISION_AX_MLX_KV_COMPRESSION_STATUS, ROUTE_DECISION_AX_MLX_KV_COMPRESSION_VALUE_BITS,
    ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS, ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT,
    ROUTE_DECISION_AX_MLX_KV_KEYS, ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS, ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB,
    ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS, ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS, RouteMetadata, SchedulePlan, Scheduler,
    SchedulerInput,
};
