#![allow(clippy::collapsible_if)]

pub mod architecture;
pub mod architecture_registry;
pub mod convert;
pub mod engine;
pub mod execution_plan;
pub mod gemma4_unified;
pub mod generation;
pub mod gguf;
pub mod ids;
pub mod kv;
pub mod mempressure;
pub mod metal;
pub mod model;
pub mod multimodal_adapter;
pub mod request;
pub mod request_manager;
pub mod runner;
pub mod sampling;
pub mod scheduler;

pub use architecture::{
    ArchitectureSpec, AttentionKind, CacheKind, FfnKind, LayerSpec, StructuralCapabilities,
};
pub use architecture_registry::{
    ARCHITECTURE_REGISTRY, ArchitectureRegistration, LayerForwardRoute,
    default_generation_for_family, lookup_architecture, resolve_layer_forward_route,
};
pub use engine::{EngineCore, EngineCoreError, EngineEvent, EngineStepOutcome, StepMetrics};
pub use execution_plan::{
    DeterministicExecutionPlanResolver, ExecutionPlanBinding, ExecutionPlanResolver,
};
pub use gemma4_unified::Gemma4UnifiedRuntimeInputError;
pub use generation::{
    FirstVisibleEventKind, GenerationKind, GenerationProgress, GenerationStrategyDescriptor,
    WorkUnitKind,
};
pub use ids::{BlockId, CacheGroupId, ModelId, RequestId, SequenceNo, StepId};
pub use kv::{
    AllocationPlan, AllocationStatus, AppendMode, BlockTable, BlockTableView, FreeResult,
    KvManager, KvManagerConfig, KvManagerError, PrefixLookupResult,
};
pub use mempressure::{
    DeviceResidentSnapshot, HostRssSnapshot, PressureLevel, PressureObservation, PressureThresholds,
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
    AX_ENGINE_3BIT_EXPERIMENTAL_ENV, AX_NATIVE_MODEL_MANIFEST_FILE,
    AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION, NativeDiffusionConfig, NativeLinearAttentionConfig,
    NativeMlaAttentionConfig, NativeModelArtifacts, NativeModelArtifactsSummary, NativeModelError,
    NativeModelManifest, NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType,
    NativeTensorFormat, NativeTensorQuantization, NativeTensorRole, NativeTensorSpec,
    WeightSanitize,
};
pub use multimodal_adapter::{MultimodalPrefillAdapter, PrefillModality};
pub use request::{
    RequestMultimodalInputs, RequestRecord, RequestSnapshot, RequestState, RequestSubmission,
    RequestWorkloadHints, StateTransitionError,
};
pub use request_manager::{RequestManager, RequestManagerError};
pub use runner::{
    DeterministicRunner, DiffusionScheduleUpdate, EmbeddingMatrix, EmbeddingPooling,
    ExecutionRunner, ExecutionStatus, KvWriteSummary, NativeModelBindingSummary,
    RequestExecutionUpdate, RequestLogitsOutput, ResolvedBlockTable, RunnerInput, RunnerOutput,
    RunnerRequestMultimodalInput,
};
#[allow(deprecated)]
pub use sampling::{
    DeterministicSampler, SampledToken, SamplerInput, SamplerRequest, SamplingParams, StopReason,
    TokenSampler,
};
pub use scheduler::{
    ExecutionBatch, ExecutionItem, ExecutionMode, PositionRange,
    ROUTE_DECISION_AX_MLX_GENERATION_KIND, ROUTE_DECISION_AX_MLX_GENERATION_WORK_UNIT,
    ROUTE_DECISION_AX_MLX_KV_CAPACITY_KIB, ROUTE_DECISION_AX_MLX_KV_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_FULL_ATTENTION_LAYERS, ROUTE_DECISION_AX_MLX_KV_GROWTH_COUNT,
    ROUTE_DECISION_AX_MLX_KV_KEYS, ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_KIB,
    ROUTE_DECISION_AX_MLX_KV_LINEAR_STATE_LAYERS, ROUTE_DECISION_AX_MLX_KV_LOGICAL_KIB,
    ROUTE_DECISION_AX_MLX_KV_LOGICAL_TOKENS, ROUTE_DECISION_AX_MLX_KV_REQUEST_SNAPSHOTS,
    ROUTE_DECISION_AX_MLX_KV_ROTATED_RING_LAYERS, ROUTE_DECISION_AX_MLX_KV_ROTATING_RING_SLACK,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_KIB,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RECLAIMABLE_CAPACITY_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_RETAINED_TOKENS,
    ROUTE_DECISION_AX_MLX_KV_SLIDING_WINDOW_LAYERS, ROUTE_DECISION_AX_MLX_LAYER_FORWARD_ROUTE,
    ROUTE_DECISION_AX_MLX_MODEL_KEYS, ROUTE_DECISION_AX_MLX_MODEL_MLA_KV_LATENT_DIM,
    ROUTE_DECISION_AX_MLX_MODEL_MOE_ACTIVE_EXPERTS, RouteMetadata, SchedulePlan, Scheduler,
    SchedulerInput, plan_work_unit_for_snapshot, upsert_route_decision,
    work_unit_for_execution_mode,
};
