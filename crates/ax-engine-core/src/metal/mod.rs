// The Metal bring-up layer mirrors kernel binding shapes and model execution
// contracts. Keeping these signatures explicit is safer than hiding them behind
// broad parameter bags while the native runtime is still stabilizing.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[cfg(target_os = "macos")]
use metal::{
    BinaryArchive, Buffer, CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Device,
    Library, MTLCommandBufferStatus, MTLPixelFormat, MTLRegion, MTLResourceOptions, MTLSize,
    MTLTextureType, TextureDescriptor,
};
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
use serde::{Deserialize, Serialize};
#[cfg(target_os = "macos")]
use std::mem::{size_of, size_of_val};
use thiserror::Error;

use crate::model::{
    NativeModelArtifacts, NativeModelArtifactsSummary, NativeModelError, NativeTensorDataType,
    NativeTensorRole, NativeTensorSpec,
};
use crate::runner::{
    successful_runner_output_from_input, ExecutionRunner, ExecutionStatus, KvWriteSummary,
    NativeModelBindingSummary, RequestExecutionUpdate, RequestLogitsOutput, RunnerInput,
    RunnerOutput,
};
use crate::sampling::{
    sample_argmax_with_logprob, SampledToken, SamplerInput, SamplerRequest, StopReason,
    TokenSampler,
};
use crate::scheduler::{ExecutionItem, ExecutionMode, RouteMetadata};

pub(crate) mod build;
pub use build::*;

pub const PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION: &str = "ax.metal.kernel_manifest.v1";
pub const PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION: &str = "ax.metal.build_report.v1";
pub const PHASE1_METAL_NATIVE_TARGET: &str = "apple_m4_or_newer_macos_aarch64";
pub const PHASE1_METAL_LANGUAGE_STANDARD: &str = "metal3.1";
pub const PHASE1_METAL_LIBRARY_NAME: &str = "ax_phase1_dense_path";
pub const PHASE1_METAL_BUILD_GATE: &str = "bringup_allowed";
pub const PHASE1_METAL_BLOCK_SIZE_ALIGNMENT_TOKENS: u32 = 16;
pub const PHASE1_DEFAULT_BLOCK_SIZE_TOKENS: u32 = 16;
pub const PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS: &[u32] = &[PHASE1_DEFAULT_BLOCK_SIZE_TOKENS];
pub const PHASE1_NUMERIC_HEAD_COUNT: u32 = 2;
pub const PHASE1_NUMERIC_HEAD_DIM: u32 = 4;
pub const PHASE1_NUMERIC_HEAD_SIZE: u32 = PHASE1_NUMERIC_HEAD_COUNT * PHASE1_NUMERIC_HEAD_DIM;
pub const PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD: u32 = 3;
pub const PHASE1_REQUIRED_METAL_KERNELS: &[&str] = &[
    "reshape_and_cache",
    "paged_decode_attention",
    "gather_kv_cache",
    "copy_blocks",
];
pub const PHASE1_DEFERRED_METAL_KERNELS: &[&str] = &["swap_blocks"];
pub const PHASE1_OPTIONAL_METAL_KERNELS: &[&str] = &[
    "kv_scale_update",
    "vector_add_f32",
    "row_scale_f32",
    "row_vector_scale_f32",
    "gather_embedding_rows_f32",
    "gather_embedding_rows_f16",
    "gather_embedding_rows_bf16",
    "decode_logits_projection_f32",
    "decode_logits_projection_f16",
    "decode_logits_projection_bf16",
    "decode_logits_projection_batched_f32",
    "decode_logits_projection_batched_f16",
    "decode_logits_projection_batched_bf16",
    "logits_argmax_f32",
    "logits_argmax_batched_f32",
    "rms_norm_f32",
    "rms_norm_f16",
    "rms_norm_bf16",
    "rms_norm_batched_f32",
    "rms_norm_batched_f16",
    "rms_norm_batched_bf16",
    "ffn_gate_silu_product_f32",
    "ffn_gate_gelu_approx_product_f32",
    "sample_argmax_logprob_f32",
    "sample_argmax_logprob_batched_f32",
    "apply_rope_f32",
    "apply_rope_batched_f32",
    "expand_grouped_kv_heads_f32",
    "linear_attention_conv1d_f32",
    "linear_attention_conv1d_f16",
    "linear_attention_conv1d_bf16",
    "linear_attention_gate_silu_f32",
    "attention_output_gate_sigmoid_product_f32",
    "linear_attention_beta_sigmoid_f32",
    "linear_attention_decay_f32",
    "linear_gated_delta_step_f32",
];
const PHASE1_MODEL_STAGE_ROPE_FREQ_BASE: f32 = 10_000.0;
pub(super) const REQUIRED_TOOLCHAIN_REQUIREMENTS: &[&str] =
    &["xcrun metal", "xcrun metallib", "xcrun metal-ar"];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalDispatchNumericLayout {
    pub head_count: u32,
    pub head_dim: u32,
}

impl MetalDispatchNumericLayout {
    pub const fn new(head_count: u32, head_dim: u32) -> Self {
        Self {
            head_count,
            head_dim,
        }
    }

    pub const fn phase1_default() -> Self {
        Self::new(PHASE1_NUMERIC_HEAD_COUNT, PHASE1_NUMERIC_HEAD_DIM)
    }

    pub fn head_size(self) -> u32 {
        self.head_count.saturating_mul(self.head_dim)
    }

    fn is_valid(self) -> bool {
        self.head_count > 0 && self.head_dim > 0
    }
}

impl Default for MetalDispatchNumericLayout {
    fn default() -> Self {
        Self::phase1_default()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalKernelTier {
    Required,
    Deferred,
    Optional,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalKernelSpec {
    pub name: String,
    pub tier: MetalKernelTier,
    pub purpose: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalThreadgroupSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub registry_id: u64,
    pub low_power: bool,
    pub headless: bool,
    pub removable: bool,
    pub max_threadgroup_memory_length: u64,
    pub max_threads_per_threadgroup: MetalThreadgroupSize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalComputePipelineInfo {
    pub function_name: String,
    pub thread_execution_width: u64,
    pub max_total_threads_per_threadgroup: u64,
    pub static_threadgroup_memory_length: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalBinaryArchiveState {
    Disabled,
    Created,
    Loaded,
    Recreated,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalBinaryArchiveInfo {
    pub path: PathBuf,
    pub state: MetalBinaryArchiveState,
    pub attached_pipeline_count: u32,
    pub serialized: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalRuntimeBringupReport {
    pub library_name: String,
    pub metallib_path: PathBuf,
    pub device: MetalDeviceInfo,
    pub compiled_kernel_names: Vec<String>,
    pub required_pipelines: Vec<MetalComputePipelineInfo>,
    pub deferred_kernel_names: Vec<String>,
    pub optional_kernel_names: Vec<String>,
    pub binary_archive: MetalBinaryArchiveInfo,
    pub command_queue_ready: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<NativeModelArtifactsSummary>,
}

pub struct MetalRuntimeBringup {
    assets: MetalKernelAssets,
    metallib: MetalKernelBinary,
    report: MetalRuntimeBringupReport,
    #[cfg(target_os = "macos")]
    state: MetalRuntimeState,
}

#[cfg(target_os = "macos")]
struct MetalRuntimeState {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    required_pipelines: Vec<MetalPipelineHandle>,
    required_pipeline_lookup: BTreeMap<String, usize>,
    optional_pipelines: Vec<MetalPipelineHandle>,
    optional_kernel_dispatch_plan: MetalOptionalKernelDispatchPlan,
    optional_kernel_feedback: Mutex<MetalOptionalKernelFeedbackState>,
    dispatch_arena: Mutex<Option<MetalDispatchArena>>,
    fused_layer_arena: Mutex<Option<FusedLayerArena>>,
}

/// Pre-allocated GPU buffers for a single-token fused layer forward pass.
/// Reused across layers since only one layer is processed at a time.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
struct FusedLayerArena {
    hidden: Buffer,
    normed: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    residual: Buffer,
    attn_projected: Buffer,
    hidden_dim: u32,
    intermediate_dim: u32,
}

#[cfg(target_os = "macos")]
impl FusedLayerArena {
    fn new(device: &Device, hidden_dim: u32, intermediate_dim: u32) -> Self {
        Self {
            hidden: new_zeroed_shared_buffer::<f32>(device, hidden_dim),
            normed: new_zeroed_shared_buffer::<f32>(device, hidden_dim),
            gate: new_zeroed_shared_buffer::<f32>(device, intermediate_dim),
            up: new_zeroed_shared_buffer::<f32>(device, intermediate_dim),
            down: new_zeroed_shared_buffer::<f32>(device, hidden_dim),
            residual: new_zeroed_shared_buffer::<f32>(device, hidden_dim),
            attn_projected: new_zeroed_shared_buffer::<f32>(device, hidden_dim),
            hidden_dim,
            intermediate_dim,
        }
    }

    fn fits(&self, hidden_dim: u32, intermediate_dim: u32) -> bool {
        self.hidden_dim >= hidden_dim && self.intermediate_dim >= intermediate_dim
    }
}

#[cfg(target_os = "macos")]
struct MetalPipelineHandle {
    function_name: String,
    pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MetalOptionalKernelDispatchPlan {
    vector_add_f32: Option<usize>,
    row_scale_f32: Option<usize>,
    row_vector_scale_f32: Option<usize>,
    projection_f32: Option<usize>,
    projection_f16: Option<usize>,
    projection_bf16: Option<usize>,
    batched_projection_f32: Option<usize>,
    batched_projection_f16: Option<usize>,
    batched_projection_bf16: Option<usize>,
    embedding_gather_f32: Option<usize>,
    embedding_gather_f16: Option<usize>,
    embedding_gather_bf16: Option<usize>,
    rms_norm_f32: Option<usize>,
    rms_norm_f16: Option<usize>,
    rms_norm_bf16: Option<usize>,
    batched_rms_norm_f32: Option<usize>,
    batched_rms_norm_f16: Option<usize>,
    batched_rms_norm_bf16: Option<usize>,
    logits_argmax_f32: Option<usize>,
    logits_argmax_batched_f32: Option<usize>,
    sample_argmax_logprob_f32: Option<usize>,
    sample_argmax_logprob_batched_f32: Option<usize>,
    apply_rope_f32: Option<usize>,
    apply_rope_batched_f32: Option<usize>,
    expand_grouped_kv_heads_f32: Option<usize>,
    ffn_gate_silu_product_f32: Option<usize>,
    ffn_gate_gelu_approx_product_f32: Option<usize>,
    linear_attention_conv1d_f32: Option<usize>,
    linear_attention_conv1d_f16: Option<usize>,
    linear_attention_conv1d_bf16: Option<usize>,
    linear_attention_gate_silu_f32: Option<usize>,
    attention_output_gate_sigmoid_product_f32: Option<usize>,
    linear_attention_beta_sigmoid_f32: Option<usize>,
    linear_attention_decay_f32: Option<usize>,
    linear_gated_delta_step_f32: Option<usize>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MetalOptionalKernelFeedbackState {
    consecutive_failures_by_kernel: BTreeMap<MetalOptionalKernelFeedbackKey, u32>,
    disabled_kernels: BTreeSet<MetalOptionalKernelFeedbackKey>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum MetalOptionalKernelFeedbackKey {
    Kernel(&'static str),
    SamplerBatchedGroup {
        group_size: usize,
        logits_width: usize,
    },
    BatchedProjection {
        kernel_name: &'static str,
        row_count: usize,
        output_dim: usize,
        input_width: usize,
        hidden_stride: usize,
        matrix_cols: usize,
    },
    Projection {
        kernel_name: &'static str,
        output_dim: usize,
        input_width: usize,
        matrix_cols: usize,
    },
    Sampler {
        kernel_name: &'static str,
        logits_width: usize,
    },
    BatchedLogitsArgmax {
        kernel_name: &'static str,
        row_count: usize,
        vocab_rows: usize,
    },
    BatchedSampler {
        kernel_name: &'static str,
        row_count: usize,
        logits_width: usize,
    },
    LogitsArgmax {
        kernel_name: &'static str,
        vocab_rows: usize,
    },
    BatchedFfnGateProduct {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    FfnGateProduct {
        kernel_name: &'static str,
        value_count: usize,
    },
    Rope {
        kernel_name: &'static str,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_style: ModelStageRopeStyle,
    },
    EmbeddingGather {
        kernel_name: &'static str,
        token_count: usize,
        embedding_rows: usize,
        hidden_dim: usize,
    },
    BatchedGroupedKvExpand {
        kernel_name: &'static str,
        token_count: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    GroupedKvExpand {
        kernel_name: &'static str,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    VectorAdd {
        kernel_name: &'static str,
        element_count: usize,
    },
    BatchedRowScale {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    BatchedRowVectorScale {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    BatchedRope {
        kernel_name: &'static str,
        token_count: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_style: ModelStageRopeStyle,
    },
    RmsNorm {
        kernel_name: &'static str,
        value_count: usize,
    },
    BatchedRmsNorm {
        kernel_name: &'static str,
        row_count: usize,
        row_width: usize,
    },
    DirectDecodeBatchedGroup {
        group_size: usize,
        dims: ModelBoundDecodeDims,
    },
    PrefixAttentionBatchedGroup {
        scheduled_requests: u32,
        prefill_requests: u32,
        decode_requests: u32,
        scheduled_tokens: u32,
        gather_tokens: u32,
        block_size_tokens: u32,
        head_count: u32,
        head_dim: u32,
    },
    LinearAttentionConv1d {
        batch_size: usize,
        dtype: NativeTensorDataType,
        dims: ResolvedLinearAttentionDims,
    },
    LinearGatedDelta {
        batch_size: usize,
        dims: ResolvedLinearAttentionDims,
    },
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn optional_kernel_name_feedback_key(kernel_name: &'static str) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Kernel(kernel_name)
}

#[cfg(target_os = "macos")]
impl MetalOptionalKernelDispatchPlan {
    fn vector_add_kernel(self) -> Option<(&'static str, usize)> {
        self.vector_add_f32.map(|index| ("vector_add_f32", index))
    }

    fn row_scale_kernel(self) -> Option<(&'static str, usize)> {
        self.row_scale_f32.map(|index| ("row_scale_f32", index))
    }

    fn row_vector_scale_kernel(self) -> Option<(&'static str, usize)> {
        self.row_vector_scale_f32
            .map(|index| ("row_vector_scale_f32", index))
    }

    fn projection_kernel(self, dtype: NativeTensorDataType) -> Option<(&'static str, usize)> {
        match dtype {
            NativeTensorDataType::F32 => self
                .projection_f32
                .map(|index| ("decode_logits_projection_f32", index)),
            NativeTensorDataType::F16 => self
                .projection_f16
                .map(|index| ("decode_logits_projection_f16", index)),
            NativeTensorDataType::Bf16 => self
                .projection_bf16
                .map(|index| ("decode_logits_projection_bf16", index)),
            _ => None,
        }
    }

    fn batched_projection_kernel(
        self,
        dtype: NativeTensorDataType,
    ) -> Option<(&'static str, usize)> {
        match dtype {
            NativeTensorDataType::F32 => self
                .batched_projection_f32
                .map(|index| ("decode_logits_projection_batched_f32", index)),
            NativeTensorDataType::F16 => self
                .batched_projection_f16
                .map(|index| ("decode_logits_projection_batched_f16", index)),
            NativeTensorDataType::Bf16 => self
                .batched_projection_bf16
                .map(|index| ("decode_logits_projection_batched_bf16", index)),
            _ => None,
        }
    }

    fn embedding_gather_kernel(self, dtype: NativeTensorDataType) -> Option<(&'static str, usize)> {
        match dtype {
            NativeTensorDataType::F32 => self
                .embedding_gather_f32
                .map(|index| ("gather_embedding_rows_f32", index)),
            NativeTensorDataType::F16 => self
                .embedding_gather_f16
                .map(|index| ("gather_embedding_rows_f16", index)),
            NativeTensorDataType::Bf16 => self
                .embedding_gather_bf16
                .map(|index| ("gather_embedding_rows_bf16", index)),
            _ => None,
        }
    }

    fn rms_norm_kernel(self, dtype: NativeTensorDataType) -> Option<(&'static str, usize)> {
        match dtype {
            NativeTensorDataType::F32 => self.rms_norm_f32.map(|index| ("rms_norm_f32", index)),
            NativeTensorDataType::F16 => self.rms_norm_f16.map(|index| ("rms_norm_f16", index)),
            NativeTensorDataType::Bf16 => self.rms_norm_bf16.map(|index| ("rms_norm_bf16", index)),
            _ => None,
        }
    }

    fn batched_rms_norm_kernel(self, dtype: NativeTensorDataType) -> Option<(&'static str, usize)> {
        match dtype {
            NativeTensorDataType::F32 => self
                .batched_rms_norm_f32
                .map(|index| ("rms_norm_batched_f32", index)),
            NativeTensorDataType::F16 => self
                .batched_rms_norm_f16
                .map(|index| ("rms_norm_batched_f16", index)),
            NativeTensorDataType::Bf16 => self
                .batched_rms_norm_bf16
                .map(|index| ("rms_norm_batched_bf16", index)),
            _ => None,
        }
    }

    fn ffn_gate_product_kernel(
        self,
        activation: ModelFfnActivation,
    ) -> Option<(&'static str, usize)> {
        match activation {
            ModelFfnActivation::Silu => self
                .ffn_gate_silu_product_f32
                .map(|index| ("ffn_gate_silu_product_f32", index)),
            ModelFfnActivation::GeluApprox => self
                .ffn_gate_gelu_approx_product_f32
                .map(|index| ("ffn_gate_gelu_approx_product_f32", index)),
        }
    }

    fn linear_gated_delta_step_kernel(self) -> Option<(&'static str, usize)> {
        self.linear_gated_delta_step_f32
            .map(|index| ("linear_gated_delta_step_f32", index))
    }

    fn linear_attention_gate_kernel(self) -> Option<(&'static str, usize)> {
        self.linear_attention_gate_silu_f32
            .map(|index| ("linear_attention_gate_silu_f32", index))
    }

    fn attention_output_gate_kernel(self) -> Option<(&'static str, usize)> {
        self.attention_output_gate_sigmoid_product_f32
            .map(|index| ("attention_output_gate_sigmoid_product_f32", index))
    }

    fn linear_attention_beta_kernel(self) -> Option<(&'static str, usize)> {
        self.linear_attention_beta_sigmoid_f32
            .map(|index| ("linear_attention_beta_sigmoid_f32", index))
    }

    fn linear_attention_decay_kernel(self) -> Option<(&'static str, usize)> {
        self.linear_attention_decay_f32
            .map(|index| ("linear_attention_decay_f32", index))
    }

    fn linear_attention_conv1d_kernel(
        self,
        dtype: NativeTensorDataType,
    ) -> Option<(&'static str, usize)> {
        match dtype {
            NativeTensorDataType::F32 => self
                .linear_attention_conv1d_f32
                .map(|index| ("linear_attention_conv1d_f32", index)),
            NativeTensorDataType::F16 => self
                .linear_attention_conv1d_f16
                .map(|index| ("linear_attention_conv1d_f16", index)),
            NativeTensorDataType::Bf16 => self
                .linear_attention_conv1d_bf16
                .map(|index| ("linear_attention_conv1d_bf16", index)),
            _ => None,
        }
    }
}

#[cfg(target_os = "macos")]
struct BinaryArchiveSession {
    archive: Option<BinaryArchive>,
    info: MetalBinaryArchiveInfo,
}

impl MetalRuntimeBringup {
    pub fn from_build_dir(path: impl AsRef<Path>) -> Result<Self, MetalRuntimeError> {
        Self::from_assets(MetalKernelAssets::from_build_dir(path)?)
    }

    pub fn from_assets(assets: MetalKernelAssets) -> Result<Self, MetalRuntimeError> {
        let metallib = load_compiled_metallib_binary(&assets)?;
        let required_kernel_names = resolve_required_kernel_names(&assets)?;

        #[cfg(target_os = "macos")]
        {
            load_macos_runtime_bringup(assets, metallib, required_kernel_names)
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (assets, metallib, required_kernel_names);
            Err(MetalRuntimeError::UnsupportedPlatform {
                host_os: std::env::consts::OS,
            })
        }
    }

    pub fn assets(&self) -> &MetalKernelAssets {
        &self.assets
    }

    pub fn metallib(&self) -> &MetalKernelBinary {
        &self.metallib
    }

    pub fn report(&self) -> &MetalRuntimeBringupReport {
        &self.report
    }

    fn dispatch_numeric_workload_with_attention_config(
        &self,
        workload: &MetalDispatchWorkload,
        staged_inputs: &MetalDispatchStagedInputs,
        attention_config: Option<ReferenceAttentionConfig>,
    ) -> Result<MetalDispatchTrace, MetalRuntimeError> {
        self.dispatch_numeric_workload_with_arena_mode(
            workload,
            staged_inputs,
            MetalDispatchArenaMode::Persistent,
            attention_config,
        )
    }

    fn dispatch_numeric_workload_ephemeral_with_attention_config(
        &self,
        workload: &MetalDispatchWorkload,
        staged_inputs: &MetalDispatchStagedInputs,
        attention_config: Option<ReferenceAttentionConfig>,
    ) -> Result<MetalDispatchTrace, MetalRuntimeError> {
        self.dispatch_numeric_workload_with_arena_mode(
            workload,
            staged_inputs,
            MetalDispatchArenaMode::Ephemeral,
            attention_config,
        )
    }

    #[cfg(target_os = "macos")]
    fn dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
        &self,
        workload: &MetalDispatchWorkload,
        staged_inputs: &MetalDispatchStagedInputs,
        attention_config: Option<ReferenceAttentionConfig>,
    ) -> Result<(MetalDispatchTrace, MetalDispatchKvCacheSnapshot), MetalRuntimeError> {
        let workload = workload.with_numeric_layout(staged_inputs.layout);
        self.assets
            .validate_block_size_tokens(workload.kv_metadata.block_size_tokens)?;
        let kernels = build_dispatch_traces(&workload, &self.report.required_pipelines);
        let synthetic_seed =
            self_contained_owned_cache_seed_from_staged_inputs(&workload, staged_inputs);
        dispatch_numeric_workload_macos_with_cache_seed(
            self,
            &workload,
            staged_inputs,
            kernels,
            MetalDispatchArenaMode::Ephemeral,
            synthetic_seed.as_ref().map(|seed| seed.as_seed()),
            attention_config,
        )
    }

    #[cfg(target_os = "macos")]
    fn dispatch_numeric_workload_ephemeral_seeded_with_attention_config(
        &self,
        workload: &MetalDispatchWorkload,
        staged_inputs: &MetalDispatchStagedInputs,
        cache_seed: MetalDispatchKvCacheSeed<'_>,
        attention_config: Option<ReferenceAttentionConfig>,
    ) -> Result<(MetalDispatchTrace, MetalDispatchKvCacheSnapshot), MetalRuntimeError> {
        let workload = workload.with_numeric_layout(staged_inputs.layout);
        self.assets
            .validate_block_size_tokens(workload.kv_metadata.block_size_tokens)?;
        let kernels = build_dispatch_traces(&workload, &self.report.required_pipelines);
        dispatch_numeric_workload_macos_with_cache_seed(
            self,
            &workload,
            staged_inputs,
            kernels,
            MetalDispatchArenaMode::Ephemeral,
            Some(cache_seed),
            attention_config,
        )
    }

    fn dispatch_numeric_workload_with_arena_mode(
        &self,
        workload: &MetalDispatchWorkload,
        staged_inputs: &MetalDispatchStagedInputs,
        arena_mode: MetalDispatchArenaMode,
        attention_config: Option<ReferenceAttentionConfig>,
    ) -> Result<MetalDispatchTrace, MetalRuntimeError> {
        let workload = workload.with_numeric_layout(staged_inputs.layout);
        self.assets
            .validate_block_size_tokens(workload.kv_metadata.block_size_tokens)?;
        let kernels = build_dispatch_traces(&workload, &self.report.required_pipelines);

        #[cfg(target_os = "macos")]
        {
            dispatch_numeric_workload_macos(
                self,
                &workload,
                staged_inputs,
                kernels,
                arena_mode,
                attention_config,
            )
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = (workload, staged_inputs);
            let _ = (kernels, arena_mode, attention_config);
            Err(MetalRuntimeError::UnsupportedPlatform {
                host_os: std::env::consts::OS,
            })
        }
    }
}

impl fmt::Debug for MetalRuntimeBringup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("MetalRuntimeBringup");
        debug
            .field("assets", &self.assets)
            .field("metallib", &self.metallib)
            .field("report", &self.report);
        #[cfg(target_os = "macos")]
        {
            let compiled_function_count = self.state.library.function_names().len();
            let required_pipeline_names = self
                .state
                .required_pipelines
                .iter()
                .map(|pipeline| pipeline.function_name.clone())
                .collect::<Vec<_>>();
            let optional_pipeline_names = self
                .state
                .optional_pipelines
                .iter()
                .map(|pipeline| pipeline.function_name.clone())
                .collect::<Vec<_>>();
            let max_thread_execution_width = self
                .state
                .required_pipelines
                .iter()
                .chain(self.state.optional_pipelines.iter())
                .map(|pipeline| pipeline.pipeline.thread_execution_width())
                .max()
                .unwrap_or(0);
            let command_queue_ready = {
                let _ = &self.state.command_queue;
                true
            };
            let dispatch_arena = self
                .state
                .dispatch_arena
                .lock()
                .expect("metal dispatch arena mutex should not be poisoned")
                .as_ref()
                .map(|arena| arena.requirements.info(true, false));

            debug
                .field("runtime_device_name", &self.state.device.name())
                .field("runtime_compiled_function_count", &compiled_function_count)
                .field("runtime_required_pipeline_names", &required_pipeline_names)
                .field("runtime_optional_pipeline_names", &optional_pipeline_names)
                .field("runtime_binary_archive", &self.report.binary_archive)
                .field("runtime_dispatch_arena", &dispatch_arena)
                .field(
                    "runtime_max_thread_execution_width",
                    &max_thread_execution_width,
                )
                .field("runtime_command_queue_ready", &command_queue_ready);
        }
        debug.finish()
    }
}

#[derive(Debug)]
pub struct MetalBringupSampler {
    bringup: MetalRuntimeBringup,
}

impl MetalBringupSampler {
    pub fn from_build_dir(path: impl AsRef<Path>) -> Result<Self, MetalRuntimeError> {
        Ok(Self {
            bringup: MetalRuntimeBringup::from_build_dir(path)?,
        })
    }

    pub fn from_assets(assets: MetalKernelAssets) -> Result<Self, MetalRuntimeError> {
        Ok(Self {
            bringup: MetalRuntimeBringup::from_assets(assets)?,
        })
    }

    #[cfg(target_os = "macos")]
    fn sample_argmax_logprob_native_retry_worthwhile(&self, logits_width: usize) -> bool {
        self.bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_f32
            .is_some_and(|_| {
                optional_kernel_allowed(
                    &self.bringup,
                    &sampler_feedback_key("sample_argmax_logprob_f32", logits_width),
                )
            })
    }

    #[cfg(target_os = "macos")]
    fn sample_argmax_logprob_with_optional_native_path(
        &self,
        logits: &[f32],
    ) -> Option<(u32, f32)> {
        let kernel_name = "sample_argmax_logprob_f32";
        if logits.is_empty() {
            return None;
        }
        let pipeline_index = self
            .bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_f32?;
        let feedback_key = sampler_feedback_key(kernel_name, logits.len());
        if !optional_kernel_allowed(&self.bringup, &feedback_key) {
            return None;
        }

        let output = find_optional_pipeline_handle_by_index(
            &self.bringup.state,
            &self.bringup.metallib.path,
            kernel_name,
            pipeline_index,
        )
        .ok()
        .and_then(|pipeline| {
            autoreleasepool(|| {
                let logits_buffer = new_shared_buffer_with_data(&self.bringup.state.device, logits);
                let argmax_buffer = new_zeroed_shared_buffer::<u32>(&self.bringup.state.device, 1);
                let logprob_buffer = new_zeroed_shared_buffer::<f32>(&self.bringup.state.device, 1);

                let command_buffer = self.bringup.state.command_queue.new_command_buffer();
                command_buffer.set_label("ax.phase1.sample_argmax_logprob");
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_label("ax.phase1.sample_argmax_logprob.compute");

                encoder.set_compute_pipeline_state(&pipeline.pipeline);
                encoder.set_buffer(0, Some(&logits_buffer), 0);
                encoder.set_buffer(1, Some(&argmax_buffer), 0);
                encoder.set_buffer(2, Some(&logprob_buffer), 0);
                set_logits_argmax_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(logits.len()),
                );
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let command_buffer_status = command_buffer_status(command_buffer.status());
                if command_buffer_status != MetalCommandBufferStatus::Completed {
                    return None;
                }

                let token_id = read_shared_u32_buffer_prefix(&argmax_buffer, 1)
                    .into_iter()
                    .next()?;
                let logprob = read_shared_buffer_prefix(&logprob_buffer, 1)
                    .into_iter()
                    .next()?;
                logprob.is_finite().then_some((token_id, logprob))
            })
        });
        record_optional_kernel_result(&self.bringup, &feedback_key, output.is_some());
        output
    }

    #[cfg(target_os = "macos")]
    fn sample_argmax_logprob_batched_with_optional_native_path(
        &self,
        logits_rows: &[&[f32]],
    ) -> Option<Vec<(u32, f32)>> {
        let kernel_name = "sample_argmax_logprob_batched_f32";
        if logits_rows.is_empty() {
            return Some(Vec::new());
        }
        let pipeline_index = self
            .bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_batched_f32?;
        let vocab_rows = logits_rows.first()?.len();
        if vocab_rows == 0 || logits_rows.iter().any(|row| row.len() != vocab_rows) {
            return None;
        }

        let token_count = logits_rows.len();
        let feedback_key = batched_sampler_feedback_key(kernel_name, token_count, vocab_rows);
        if !optional_kernel_allowed(&self.bringup, &feedback_key) {
            return None;
        }
        let logits_element_count = token_count.checked_mul(vocab_rows)?;
        let mut flattened_logits = Vec::with_capacity(logits_element_count);
        for row in logits_rows {
            flattened_logits.extend_from_slice(row);
        }

        let output = find_optional_pipeline_handle_by_index(
            &self.bringup.state,
            &self.bringup.metallib.path,
            kernel_name,
            pipeline_index,
        )
        .ok()
        .and_then(|pipeline| {
            autoreleasepool(|| {
                let logits_buffer =
                    new_shared_buffer_with_data(&self.bringup.state.device, &flattened_logits);
                let argmax_buffer = new_zeroed_shared_buffer::<u32>(
                    &self.bringup.state.device,
                    saturating_usize_to_u32(token_count),
                );
                let logprob_buffer = new_zeroed_shared_buffer::<f32>(
                    &self.bringup.state.device,
                    saturating_usize_to_u32(token_count),
                );

                let command_buffer = self.bringup.state.command_queue.new_command_buffer();
                command_buffer.set_label("ax.phase1.sample_argmax_logprob_batched");
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_label("ax.phase1.sample_argmax_logprob_batched.compute");

                encoder.set_compute_pipeline_state(&pipeline.pipeline);
                encoder.set_buffer(0, Some(&logits_buffer), 0);
                encoder.set_buffer(1, Some(&argmax_buffer), 0);
                encoder.set_buffer(2, Some(&logprob_buffer), 0);
                set_batched_logits_argmax_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(token_count),
                    saturating_usize_to_u32(vocab_rows),
                );
                encoder.dispatch_threads(
                    MTLSize::new(token_count.max(1) as u64, 1, 1),
                    MTLSize::new(
                        pipeline
                            .pipeline
                            .thread_execution_width()
                            .max(1)
                            .min(token_count.max(1) as u64),
                        1,
                        1,
                    ),
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let command_buffer_status = command_buffer_status(command_buffer.status());
                if command_buffer_status != MetalCommandBufferStatus::Completed {
                    return None;
                }

                let token_ids = read_shared_u32_buffer_prefix(
                    &argmax_buffer,
                    saturating_usize_to_u32(token_count),
                );
                let logprobs = read_shared_buffer_prefix(
                    &logprob_buffer,
                    saturating_usize_to_u32(token_count),
                );
                if token_ids.len() != token_count
                    || logprobs.len() != token_count
                    || logprobs.iter().any(|value| !value.is_finite())
                {
                    return None;
                }

                Some(token_ids.into_iter().zip(logprobs).collect())
            })
        });
        record_optional_kernel_result(&self.bringup, &feedback_key, output.is_some());
        output
    }
}

impl TokenSampler for MetalBringupSampler {
    fn sample(&self, input: SamplerInput) -> Vec<SampledToken> {
        let requests = input.requests;
        let mut sampled_from_logits = vec![None; requests.len()];

        #[cfg(target_os = "macos")]
        {
            for (logits_width, indices) in
                grouped_sampler_request_indices_by_logits_width(&requests)
            {
                let allow_single_native =
                    self.sample_argmax_logprob_native_retry_worthwhile(logits_width);
                if let Some(group_results) = collect_grouped_sampler_results_with_item_fallback(
                    &indices,
                    &mut |group_indices| {
                        if group_indices.len() < 2 {
                            return None;
                        }
                        let feedback_key =
                            sampler_batched_group_feedback_key(group_indices.len(), logits_width);
                        if !optional_kernel_allowed(&self.bringup, &feedback_key) {
                            return None;
                        }
                        let logits_rows = group_indices
                            .iter()
                            .map(|index| {
                                requests
                                    .get(*index)
                                    .and_then(|request| request.logits.as_deref())
                                    .expect("grouped sampler request should retain logits")
                            })
                            .collect::<Vec<_>>();
                        let output = self
                            .sample_argmax_logprob_batched_with_optional_native_path(&logits_rows);
                        let (output, success) =
                            validate_batched_sampler_group_output(output, group_indices.len());
                        record_optional_kernel_result(&self.bringup, &feedback_key, success);
                        output
                    },
                    &mut |request_index| {
                        requests.get(request_index).and_then(|request| {
                            request.logits.as_ref().and_then(|logits| {
                                allow_single_native
                                    .then(|| {
                                        self.sample_argmax_logprob_with_optional_native_path(logits)
                                    })
                                    .flatten()
                                    .or_else(|| sample_argmax_with_logprob(logits))
                            })
                        })
                    },
                ) {
                    for (request_index, result) in group_results {
                        sampled_from_logits[request_index] = Some(result);
                    }
                }
            }
        }

        requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| {
                let sampled_from_logits = sampled_from_logits[index].or_else(|| {
                    request.logits.as_ref().and_then(|logits| {
                        #[cfg(target_os = "macos")]
                        {
                            self.sample_argmax_logprob_native_retry_worthwhile(logits.len())
                                .then(|| {
                                    self.sample_argmax_logprob_with_optional_native_path(logits)
                                })
                                .flatten()
                                .or_else(|| sample_argmax_with_logprob(logits))
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            sample_argmax_with_logprob(logits)
                        }
                    })
                });
                let token_id = sampled_from_logits
                    .map(|(token_id, _)| token_id)
                    .unwrap_or_else(|| request.previous_token.saturating_add(1));
                let logprob = sampled_from_logits
                    .map(|(_, logprob)| logprob)
                    .or(Some(0.0));
                let stop_reason =
                    if request.generated_len.saturating_add(1) >= request.max_output_tokens {
                        Some(StopReason::MaxOutputTokens)
                    } else {
                        None
                    };

                SampledToken {
                    request_id: request.request_id,
                    token_id,
                    stop_reason,
                    logprob,
                }
            })
            .collect()
    }
}

fn grouped_sampler_request_indices_by_logits_width(
    requests: &[SamplerRequest],
) -> BTreeMap<usize, Vec<usize>> {
    let mut grouped_request_indices = BTreeMap::<usize, Vec<usize>>::new();
    for (index, request) in requests.iter().enumerate() {
        if let Some(logits) = request.logits.as_ref() {
            grouped_request_indices
                .entry(logits.len())
                .or_default()
                .push(index);
        }
    }
    grouped_request_indices
}

#[cfg(target_os = "macos")]
fn collect_grouped_sampler_results_with_item_fallback<T>(
    indices: &[usize],
    process_group: &mut impl FnMut(&[usize]) -> Option<Vec<T>>,
    process_item: &mut impl FnMut(usize) -> Option<T>,
) -> Option<Vec<(usize, T)>> {
    if indices.is_empty() {
        return Some(Vec::new());
    }
    if indices.len() > 1 {
        if let Some(results) = process_group(indices) {
            if results.len() != indices.len() {
                return None;
            }
            return Some(indices.iter().copied().zip(results).collect());
        }
        let split_index = indices.len() / 2;
        let mut left_results = collect_grouped_sampler_results_with_item_fallback(
            &indices[..split_index],
            process_group,
            process_item,
        )?;
        let right_results = collect_grouped_sampler_results_with_item_fallback(
            &indices[split_index..],
            process_group,
            process_item,
        )?;
        left_results.extend(right_results);
        return Some(left_results);
    }

    let request_index = *indices.first()?;
    Some(vec![(request_index, process_item(request_index)?)])
}

#[cfg(target_os = "macos")]
fn sampler_batched_group_feedback_key(
    group_size: usize,
    logits_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::SamplerBatchedGroup {
        group_size,
        logits_width,
    }
}

#[cfg(target_os = "macos")]
fn batched_projection_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    output_dim: usize,
    input_width: usize,
    hidden_stride: usize,
    matrix_cols: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedProjection {
        kernel_name,
        row_count,
        output_dim,
        input_width,
        hidden_stride,
        matrix_cols,
    }
}

#[cfg(target_os = "macos")]
fn projection_feedback_key(
    kernel_name: &'static str,
    output_dim: usize,
    input_width: usize,
    matrix_cols: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Projection {
        kernel_name,
        output_dim,
        input_width,
        matrix_cols,
    }
}

#[cfg(target_os = "macos")]
fn sampler_feedback_key(
    kernel_name: &'static str,
    logits_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Sampler {
        kernel_name,
        logits_width,
    }
}

#[cfg(target_os = "macos")]
fn batched_logits_argmax_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    vocab_rows: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedLogitsArgmax {
        kernel_name,
        row_count,
        vocab_rows,
    }
}

#[cfg(target_os = "macos")]
fn batched_sampler_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    logits_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedSampler {
        kernel_name,
        row_count,
        logits_width,
    }
}

#[cfg(target_os = "macos")]
fn logits_argmax_feedback_key(
    kernel_name: &'static str,
    vocab_rows: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::LogitsArgmax {
        kernel_name,
        vocab_rows,
    }
}

#[cfg(target_os = "macos")]
fn batched_ffn_gate_product_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedFfnGateProduct {
        kernel_name,
        row_count,
        row_width,
    }
}

#[cfg(target_os = "macos")]
fn ffn_gate_product_feedback_key(
    kernel_name: &'static str,
    value_count: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::FfnGateProduct {
        kernel_name,
        value_count,
    }
}

#[cfg(target_os = "macos")]
fn rope_feedback_key(
    kernel_name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_style: ModelStageRopeStyle,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::Rope {
        kernel_name,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    }
}

#[cfg(target_os = "macos")]
fn embedding_gather_feedback_key(
    kernel_name: &'static str,
    token_count: usize,
    embedding_rows: usize,
    hidden_dim: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::EmbeddingGather {
        kernel_name,
        token_count,
        embedding_rows,
        hidden_dim,
    }
}

#[cfg(target_os = "macos")]
fn batched_grouped_kv_expand_feedback_key(
    kernel_name: &'static str,
    token_count: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedGroupedKvExpand {
        kernel_name,
        token_count,
        q_heads,
        kv_heads,
        head_dim,
    }
}

#[cfg(target_os = "macos")]
fn grouped_kv_expand_feedback_key(
    kernel_name: &'static str,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::GroupedKvExpand {
        kernel_name,
        q_heads,
        kv_heads,
        head_dim,
    }
}

#[cfg(target_os = "macos")]
fn vector_add_feedback_key(
    kernel_name: &'static str,
    element_count: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::VectorAdd {
        kernel_name,
        element_count,
    }
}

#[cfg(target_os = "macos")]
fn batched_row_scale_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRowScale {
        kernel_name,
        row_count,
        row_width,
    }
}

#[cfg(target_os = "macos")]
fn batched_row_vector_scale_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRowVectorScale {
        kernel_name,
        row_count,
        row_width,
    }
}

#[cfg(target_os = "macos")]
fn batched_rope_feedback_key(
    kernel_name: &'static str,
    token_count: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_style: ModelStageRopeStyle,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRope {
        kernel_name,
        token_count,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    }
}

#[cfg(target_os = "macos")]
fn rms_norm_feedback_key(
    kernel_name: &'static str,
    value_count: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::RmsNorm {
        kernel_name,
        value_count,
    }
}

#[cfg(target_os = "macos")]
fn batched_rms_norm_feedback_key(
    kernel_name: &'static str,
    row_count: usize,
    row_width: usize,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::BatchedRmsNorm {
        kernel_name,
        row_count,
        row_width,
    }
}

#[cfg(target_os = "macos")]
fn batched_rms_norm_feedback_binding(
    bringup: &MetalRuntimeBringup,
    weight_binding: &MetalNativeTensorBufferBinding,
    row_count: usize,
    row_width: usize,
) -> Option<(&'static str, usize, MetalOptionalKernelFeedbackKey)> {
    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_rms_norm_kernel(weight_binding.native_dtype)?;
    Some((
        kernel_name,
        pipeline_index,
        batched_rms_norm_feedback_key(kernel_name, row_count, row_width),
    ))
}

#[cfg(target_os = "macos")]
fn batched_projection_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_projection_kernel(binding.native_dtype)
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_rms_norm_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    weight_binding: &MetalNativeTensorBufferBinding,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_rms_norm_kernel(weight_binding.native_dtype)
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_rms_norm_without_weights_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_rms_norm_kernel(NativeTensorDataType::F32)
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_vector_add_split_retry_worthwhile(bringup: Option<&MetalRuntimeBringup>) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .vector_add_kernel()
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_ffn_gate_product_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    activation: ModelFfnActivation,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .ffn_gate_product_kernel(activation)
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_attention_output_gate_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .attention_output_gate_kernel()
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_linear_attention_gate_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_gate_kernel()
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_linear_attention_beta_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_beta_kernel()
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_linear_attention_decay_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_decay_kernel()
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_linear_attention_conv_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    dtype: NativeTensorDataType,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_attention_conv1d_kernel(dtype)
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_linear_attention_recurrent_split_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .linear_gated_delta_step_kernel()
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_rope_split_retry_worthwhile(bringup: Option<&MetalRuntimeBringup>) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .apply_rope_batched_f32
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn batched_grouped_kv_expand_split_retry_worthwhile(bringup: Option<&MetalRuntimeBringup>) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .expand_grouped_kv_heads_f32
            .is_some()
    })
}

#[cfg(target_os = "macos")]
fn single_projection_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
    output_dim: usize,
    input_width: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        tensor_matrix_dimensions(&binding.meta.spec).is_some_and(|(_, cols)| {
            bringup
                .state
                .optional_kernel_dispatch_plan
                .projection_kernel(binding.native_dtype)
                .is_some_and(|(kernel_name, _)| {
                    optional_kernel_allowed(
                        bringup,
                        &projection_feedback_key(kernel_name, output_dim, input_width, cols),
                    )
                })
        })
    })
}

#[cfg(target_os = "macos")]
fn single_rms_norm_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    weight_binding: &MetalNativeTensorBufferBinding,
    value_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .rms_norm_kernel(weight_binding.native_dtype)
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(bringup, &rms_norm_feedback_key(kernel_name, value_count))
            })
    })
}

#[cfg(target_os = "macos")]
fn single_rms_norm_without_weights_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    value_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .rms_norm_kernel(NativeTensorDataType::F32)
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(bringup, &rms_norm_feedback_key(kernel_name, value_count))
            })
    })
}

#[cfg(target_os = "macos")]
fn single_vector_add_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    element_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .vector_add_kernel()
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(
                    bringup,
                    &vector_add_feedback_key(kernel_name, element_count),
                )
            })
    })
}

#[cfg(target_os = "macos")]
fn single_ffn_gate_product_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    activation: ModelFfnActivation,
    value_count: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .ffn_gate_product_kernel(activation)
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(
                    bringup,
                    &ffn_gate_product_feedback_key(kernel_name, value_count),
                )
            })
    })
}

#[cfg(target_os = "macos")]
fn single_rope_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    stage_dims: ModelStageDims,
    rotary_dim: usize,
    rope_style: ModelStageRopeStyle,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .apply_rope_f32
            .is_some_and(|_| {
                optional_kernel_allowed(
                    bringup,
                    &rope_feedback_key(
                        "apply_rope_f32",
                        stage_dims.q_heads,
                        stage_dims.kv_heads,
                        stage_dims.head_dim,
                        rotary_dim,
                        rope_style,
                    ),
                )
            })
    })
}

#[cfg(target_os = "macos")]
fn single_decode_logits_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden_width: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        tensor_matrix_dimensions(&decode_projection.meta.spec).is_some_and(
            |(vocab_rows, projection_cols)| {
                let input_width = hidden_width.min(projection_cols);
                if vocab_rows == 0 || input_width == 0 {
                    return false;
                }
                bringup
                    .state
                    .optional_kernel_dispatch_plan
                    .projection_kernel(decode_projection.native_dtype)
                    .is_some_and(|(projection_kernel_name, _)| {
                        bringup
                            .state
                            .optional_kernel_dispatch_plan
                            .logits_argmax_f32
                            .is_some_and(|_| {
                                optional_kernel_allowed(
                                    bringup,
                                    &projection_feedback_key(
                                        projection_kernel_name,
                                        vocab_rows,
                                        input_width,
                                        projection_cols,
                                    ),
                                ) && optional_kernel_allowed(
                                    bringup,
                                    &logits_argmax_feedback_key("logits_argmax_f32", vocab_rows),
                                )
                            })
                    })
            },
        )
    })
}

#[cfg(target_os = "macos")]
fn single_attention_output_gate_retry_worthwhile(
    bringup: Option<&MetalRuntimeBringup>,
    row_width: usize,
) -> bool {
    bringup.is_some_and(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .attention_output_gate_kernel()
            .is_some_and(|(kernel_name, _)| {
                optional_kernel_allowed(
                    bringup,
                    &batched_ffn_gate_product_feedback_key(kernel_name, 1, row_width),
                )
            })
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
struct SingleFfnGateUpProjectionNativeRetryPolicy<'a> {
    gate_projection: Option<&'a MetalRuntimeBringup>,
    up_projection: Option<&'a MetalRuntimeBringup>,
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn single_ffn_gate_up_projection_retry_policy<'a>(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input_width: usize,
    bringup: Option<&'a MetalRuntimeBringup>,
) -> Option<SingleFfnGateUpProjectionNativeRetryPolicy<'a>> {
    let policy = match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let native_bringup =
                single_projection_retry_worthwhile(bringup, packed, intermediate_dim, input_width)
                    .then_some(bringup)
                    .flatten();
            SingleFfnGateUpProjectionNativeRetryPolicy {
                gate_projection: native_bringup,
                up_projection: native_bringup,
            }
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            SingleFfnGateUpProjectionNativeRetryPolicy {
                gate_projection: single_projection_retry_worthwhile(
                    bringup,
                    gate_binding,
                    intermediate_dim,
                    input_width,
                )
                .then_some(bringup)
                .flatten(),
                up_projection: single_projection_retry_worthwhile(
                    bringup,
                    up_binding,
                    intermediate_dim,
                    input_width,
                )
                .then_some(bringup)
                .flatten(),
            }
        }
    };
    Some(policy)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Default)]
struct DirectDecodeSingleNativeRetryPolicy<'a> {
    attention_output_gate: Option<&'a MetalRuntimeBringup>,
    attention_o_projection: Option<&'a MetalRuntimeBringup>,
    attention_residual_add: Option<&'a MetalRuntimeBringup>,
    final_norm: Option<&'a MetalRuntimeBringup>,
    logits_projection: Option<&'a MetalRuntimeBringup>,
}

#[cfg(target_os = "macos")]
fn direct_decode_single_native_retry_policy<'a>(
    artifacts: &NativeModelArtifacts,
    attention_o: &MetalNativeTensorBufferBinding,
    final_norm: &MetalNativeTensorBufferBinding,
    decode_projection: &MetalNativeTensorBufferBinding,
    dims: ModelBoundDecodeDims,
    bringup: Option<&'a MetalRuntimeBringup>,
) -> Option<DirectDecodeSingleNativeRetryPolicy<'a>> {
    let hidden_vector_add = single_vector_add_retry_worthwhile(bringup, dims.hidden_dim)
        .then_some(bringup)
        .flatten();
    Some(DirectDecodeSingleNativeRetryPolicy {
        attention_output_gate: artifacts
            .manifest()
            .attn_output_gate
            .then(|| {
                single_attention_output_gate_retry_worthwhile(bringup, dims.input_width)
                    .then_some(bringup)
                    .flatten()
            })
            .flatten(),
        attention_o_projection: single_projection_retry_worthwhile(
            bringup,
            attention_o,
            dims.hidden_dim,
            dims.input_width,
        )
        .then_some(bringup)
        .flatten(),
        attention_residual_add: hidden_vector_add,
        final_norm: single_rms_norm_retry_worthwhile(bringup, final_norm, dims.hidden_dim)
            .then_some(bringup)
            .flatten(),
        logits_projection: single_decode_logits_retry_worthwhile(
            bringup,
            decode_projection,
            dims.hidden_dim,
        )
        .then_some(bringup)
        .flatten(),
    })
}

#[cfg(target_os = "macos")]
fn ffn_gate_product_feedback_binding(
    bringup: &MetalRuntimeBringup,
    activation: ModelFfnActivation,
    row_count: usize,
    row_width: usize,
) -> Option<(&'static str, usize, MetalOptionalKernelFeedbackKey)> {
    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .ffn_gate_product_kernel(activation)?;
    Some((
        kernel_name,
        pipeline_index,
        batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width),
    ))
}

#[cfg(target_os = "macos")]
fn validate_batched_sampler_group_output<T>(
    output: Option<Vec<T>>,
    expected_len: usize,
) -> (Option<Vec<T>>, bool) {
    match output {
        Some(results) if results.len() == expected_len => (Some(results), true),
        Some(_) | None => (None, false),
    }
}

#[cfg(target_os = "macos")]
fn validate_model_bound_direct_decode_group_output(
    output: Option<ModelBoundDirectDecodeResult>,
    expected_request_ids: &[crate::ids::RequestId],
) -> (Option<ModelBoundDirectDecodeResult>, bool) {
    let expected_ids = expected_request_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let Some(result) = output else {
        return (None, false);
    };
    let token_ids = result
        .tokens
        .iter()
        .map(|(request_id, _)| *request_id)
        .collect::<BTreeSet<_>>();
    let logits_request_ids = result
        .logits_outputs
        .iter()
        .map(|output| output.request_id)
        .collect::<BTreeSet<_>>();
    let has_complete_logits_payload = result.logits_outputs.is_empty()
        || (result.logits_outputs.len() == expected_request_ids.len()
            && logits_request_ids == expected_ids);
    let success = result.tokens.len() == expected_request_ids.len()
        && token_ids == expected_ids
        && has_complete_logits_payload;
    if success {
        (Some(result), true)
    } else {
        (None, false)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalCommandBufferStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchKvMetadata {
    pub block_size_tokens: u32,
    pub slot_mapping: Vec<u32>,
    pub attention_block_table: Vec<u32>,
    pub gather_block_table: Vec<u32>,
    pub gather_block_table_stride: u32,
    pub copy_block_mapping: Vec<[u32; 2]>,
    pub seq_lens: Vec<u32>,
    pub cu_seq_lens: Vec<u32>,
    pub scheduled_cu_seq_lens: Vec<u32>,
}

impl MetalDispatchKvMetadata {
    fn gather_token_count(&self) -> u32 {
        self.cu_seq_lens.last().copied().unwrap_or(0).max(1)
    }

    fn slot_capacity(&self) -> u32 {
        let max_direct_slot = self
            .slot_mapping
            .iter()
            .chain(self.attention_block_table.iter())
            .copied()
            .max()
            .unwrap_or(0);

        max_direct_slot
            .max(self.block_capacity().saturating_sub(1))
            .saturating_add(1)
            .max(1)
    }

    fn block_capacity(&self) -> u32 {
        let block_span = self.block_size_tokens.saturating_sub(1);
        let max_gather_slot = self
            .gather_block_table
            .iter()
            .copied()
            .map(|base| base.saturating_add(block_span))
            .max()
            .unwrap_or(0);
        let max_copy_slot = self
            .copy_block_mapping
            .iter()
            .flat_map(|pair| {
                [
                    pair[0].saturating_add(block_span),
                    pair[1].saturating_add(block_span),
                ]
            })
            .max()
            .unwrap_or(0);

        max_gather_slot.max(max_copy_slot).saturating_add(1).max(1)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchWorkload {
    pub scheduled_requests: u32,
    pub prefill_requests: u32,
    pub decode_requests: u32,
    pub scheduled_tokens: u32,
    pub scheduled_token_ids: Vec<u32>,
    pub scheduled_positions: Vec<u32>,
    pub resolved_blocks: u32,
    pub token_elements: u32,
    pub block_elements: u32,
    pub scratch_elements: u32,
    pub kv_slot_capacity: u32,
    pub kv_block_capacity: u32,
    #[serde(default)]
    pub numeric_layout: MetalDispatchNumericLayout,
    pub kv_metadata: MetalDispatchKvMetadata,
}

impl MetalDispatchWorkload {
    pub fn from_runner_input(input: &RunnerInput) -> Result<Self, MetalRuntimeError> {
        let kv_metadata = build_dispatch_kv_metadata(input)?;
        let scheduled_requests = input.execution_batch.items.len() as u32;
        let decode_requests = input
            .execution_batch
            .items
            .iter()
            .filter(|item| item.mode == ExecutionMode::Decode)
            .count() as u32;
        let prefill_requests = scheduled_requests.saturating_sub(decode_requests);
        let scheduled_tokens = input.execution_batch.total_scheduled_tokens;
        let mut scheduled_token_ids = Vec::with_capacity(scheduled_tokens as usize);
        let mut scheduled_positions = Vec::with_capacity(scheduled_tokens as usize);
        for item in &input.execution_batch.items {
            if item.input_token_slice.len() != item.scheduled_token_count as usize {
                return Err(MetalRuntimeError::InvalidDispatchInput {
                    message: format!(
                        "request {} scheduled_token_count={} does not match input_token_slice length={}",
                        item.request_id.0,
                        item.scheduled_token_count,
                        item.input_token_slice.len()
                    ),
                });
            }
            scheduled_token_ids.extend(item.input_token_slice.iter().copied());
            for offset in 0..item.scheduled_token_count {
                scheduled_positions.push(item.position_range.start.saturating_add(offset));
            }
        }
        if scheduled_token_ids.len() as u32 != scheduled_tokens {
            return Err(MetalRuntimeError::InvalidDispatchInput {
                message: format!(
                    "execution_batch total_scheduled_tokens={} does not match flattened scheduled token count={}",
                    scheduled_tokens,
                    scheduled_token_ids.len()
                ),
            });
        }
        let resolved_blocks = input
            .block_tables
            .iter()
            .map(|resolved| resolved.block_table.block_ids.len() as u32)
            .sum::<u32>();
        let token_elements = scheduled_tokens.max(1);
        let block_elements = resolved_blocks.max(1);
        let scratch_elements = token_elements.max(block_elements);
        let kv_slot_capacity = kv_metadata.slot_capacity();
        let kv_block_capacity = kv_metadata.block_capacity();
        if scheduled_token_ids.is_empty() {
            scheduled_token_ids.push(0);
            scheduled_positions.push(0);
        }

        Ok(Self {
            scheduled_requests,
            prefill_requests,
            decode_requests,
            scheduled_tokens,
            scheduled_token_ids,
            scheduled_positions,
            resolved_blocks,
            token_elements,
            block_elements,
            scratch_elements,
            kv_slot_capacity,
            kv_block_capacity,
            numeric_layout: MetalDispatchNumericLayout::default(),
            kv_metadata,
        })
    }

    fn with_numeric_layout(&self, numeric_layout: MetalDispatchNumericLayout) -> Self {
        let mut workload = self.clone();
        workload.numeric_layout = if numeric_layout.is_valid() {
            numeric_layout
        } else {
            MetalDispatchNumericLayout::default()
        };
        workload
    }

    fn scheduled_numeric_elements(&self) -> u32 {
        self.token_elements
            .saturating_mul(self.numeric_layout.head_size())
    }

    fn slot_numeric_capacity(&self) -> u32 {
        self.kv_slot_capacity
            .saturating_mul(self.numeric_layout.head_size())
    }

    fn gather_numeric_elements(&self) -> u32 {
        self.kv_metadata
            .gather_token_count()
            .saturating_mul(self.numeric_layout.head_size())
    }

    fn attention_numeric_elements(&self) -> u32 {
        self.token_elements
            .saturating_mul(self.numeric_layout.head_size())
    }

    fn block_numeric_elements(&self) -> u32 {
        self.kv_metadata
            .block_size_tokens
            .max(1)
            .saturating_mul(self.numeric_layout.head_size())
    }

    fn copy_numeric_elements(&self) -> u32 {
        (self.kv_metadata.copy_block_mapping.len() as u32)
            .max(1)
            .saturating_mul(self.block_numeric_elements())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchKernelTrace {
    pub function_name: String,
    pub element_count: u32,
    pub threads_per_grid: MetalThreadgroupSize,
    pub threads_per_threadgroup: MetalThreadgroupSize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchArenaInfo {
    pub token_capacity: u32,
    pub slot_capacity: u32,
    pub attention_ref_capacity: u32,
    pub gather_ref_capacity: u32,
    pub gather_output_capacity: u32,
    pub copy_pair_capacity: u32,
    pub sequence_capacity: u32,
    pub reused_existing: bool,
    pub grew_existing: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchTrace {
    pub command_queue_label: String,
    pub command_buffer_label: String,
    pub command_buffer_status: MetalCommandBufferStatus,
    pub runtime: MetalDispatchRuntimeInfo,
    pub workload: MetalDispatchWorkload,
    pub arena: MetalDispatchArenaInfo,
    #[serde(default)]
    pub execution: MetalDispatchExecutionInfo,
    pub kernels: Vec<MetalDispatchKernelTrace>,
    pub numeric: MetalDispatchNumericTrace,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct MetalDispatchExecutionInfo {
    #[serde(default)]
    pub direct_decode_token_count: u32,
    #[serde(default)]
    pub direct_decode_checksum_lo: u32,
    #[serde(default)]
    pub logits_output_count: u32,
    #[serde(default)]
    pub remaining_logits_handle_count: u32,
    #[serde(default)]
    pub model_bound_ffn_decode: bool,
    #[serde(default)]
    pub real_model_forward_completed: bool,
    #[serde(default)]
    pub prefix_native_dispatch_count: u32,
    #[serde(default)]
    pub prefix_cpu_reference_dispatch_count: u32,
    #[serde(default)]
    pub qkv_projection_token_count: u32,
    #[serde(default)]
    pub layer_continuation_token_count: u32,
    #[serde(default)]
    pub logits_projection_token_count: u32,
    #[serde(default)]
    pub logits_vocab_scan_row_count: u32,
    #[serde(default)]
    pub prefix_native_projection_row_count: u32,
    #[serde(default)]
    pub prefix_cpu_projection_row_count: u32,
    #[serde(default)]
    pub prefix_native_rms_norm_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_rms_norm_element_count: u32,
    #[serde(default)]
    pub prefix_native_ffn_activation_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_ffn_activation_element_count: u32,
    #[serde(default)]
    pub prefix_native_residual_add_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_residual_add_element_count: u32,
    #[serde(default)]
    pub prefix_native_scale_element_count: u32,
    #[serde(default)]
    pub prefix_cpu_scale_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_projection_row_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_projection_row_count: u32,
    #[serde(default)]
    pub direct_decode_native_rms_norm_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_rms_norm_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_ffn_activation_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_ffn_activation_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_residual_add_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_residual_add_element_count: u32,
    #[serde(default)]
    pub direct_decode_native_scale_element_count: u32,
    #[serde(default)]
    pub direct_decode_cpu_scale_element_count: u32,
    #[serde(default)]
    pub direct_decode_batched_logits_group_count: u32,
    #[serde(default)]
    pub direct_decode_batched_logits_token_count: u32,
    #[serde(default)]
    pub direct_decode_batched_group_fallback_count: u32,
    #[serde(default)]
    pub direct_decode_batched_group_fallback_token_count: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct MetalNativeDenseKernelCoverage {
    #[serde(default)]
    pub projection_f32_binding_count: u32,
    #[serde(default)]
    pub projection_f16_binding_count: u32,
    #[serde(default)]
    pub projection_bf16_binding_count: u32,
    #[serde(default)]
    pub projection_unsupported_binding_count: u32,
    #[serde(default)]
    pub projection_source_quantized_binding_count: u32,
    #[serde(default)]
    pub rms_norm_f32_binding_count: u32,
    #[serde(default)]
    pub rms_norm_f16_binding_count: u32,
    #[serde(default)]
    pub rms_norm_bf16_binding_count: u32,
    #[serde(default)]
    pub rms_norm_unsupported_binding_count: u32,
    #[serde(default)]
    pub rms_norm_source_quantized_binding_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchRuntimeInfo {
    pub device_name: String,
    pub required_pipeline_count: u32,
    pub max_thread_execution_width: u64,
    pub binary_archive: MetalBinaryArchiveInfo,
    pub command_queue_ready: bool,
    #[serde(default)]
    pub model_conditioned_inputs: bool,
    #[serde(default)]
    pub real_model_tensor_inputs: bool,
    #[serde(default)]
    pub complete_model_forward_supported: bool,
    #[serde(default)]
    pub model_bindings_prepared: bool,
    #[serde(default)]
    pub model_buffers_bound: bool,
    #[serde(default)]
    pub model_buffer_count: u32,
    #[serde(default)]
    pub model_buffer_bytes: u64,
    #[serde(default)]
    pub native_dense_kernel_coverage: MetalNativeDenseKernelCoverage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<NativeModelArtifactsSummary>,
}

impl MetalDispatchRuntimeInfo {
    fn from_bringup_report(report: &MetalRuntimeBringupReport) -> Self {
        Self {
            device_name: report.device.name.clone(),
            required_pipeline_count: report.required_pipelines.len() as u32,
            max_thread_execution_width: report
                .required_pipelines
                .iter()
                .map(|pipeline| pipeline.thread_execution_width)
                .max()
                .unwrap_or(0),
            binary_archive: report.binary_archive.clone(),
            command_queue_ready: report.command_queue_ready,
            model_conditioned_inputs: false,
            real_model_tensor_inputs: false,
            complete_model_forward_supported: false,
            model_bindings_prepared: false,
            model_buffers_bound: false,
            model_buffer_count: 0,
            model_buffer_bytes: 0,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: report.model.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetalStagedInputSource {
    SyntheticTokenIds,
    ModelConditionedMiniProjection,
    ModelConditionedCpuPrefixAttention,
    ModelConditionedNativePrefixAttention,
    ModelConditionedMixedPrefixAttention,
}

#[derive(Clone, Debug)]
struct MetalDispatchStagedInputs {
    key: Vec<f32>,
    value: Vec<f32>,
    query: Vec<f32>,
    layout: MetalDispatchNumericLayout,
    source: MetalStagedInputSource,
    prefix_attention_tally: PrefixAttentionExecutionTally,
    #[cfg(target_os = "macos")]
    final_layer_hidden_state_cache: Option<ModelFinalLayerHiddenStateCache>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
struct ModelFinalLayerHiddenStateCache {
    hidden_states: Vec<Vec<f32>>,
    final_layer_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MetalNativeTensorBinding {
    spec: NativeTensorSpec,
    resolved_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::large_enum_variant)]
enum MetalAttentionQkvBindings {
    Packed(MetalNativeTensorBinding),
    Split {
        q: MetalNativeTensorBinding,
        k: MetalNativeTensorBinding,
        v: Option<MetalNativeTensorBinding>,
        value_from_key: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MetalFfnGateUpBindings {
    Packed(MetalNativeTensorBinding),
    Split {
        gate: MetalNativeTensorBinding,
        up: MetalNativeTensorBinding,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MetalMoeExpertGateUpBindings {
    Packed(MetalNativeTensorBinding),
    Split {
        gate: MetalNativeTensorBinding,
        up: MetalNativeTensorBinding,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MetalMoeBindings {
    router: MetalNativeTensorBinding,
    router_scale: Option<MetalNativeTensorBinding>,
    expert_gate_up: MetalMoeExpertGateUpBindings,
    expert_down: MetalNativeTensorBinding,
    expert_down_scale: Option<MetalNativeTensorBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MetalLinearAttentionBindings {
    in_proj_qkv: MetalNativeTensorBinding,
    in_proj_z: MetalNativeTensorBinding,
    in_proj_a: MetalNativeTensorBinding,
    in_proj_b: MetalNativeTensorBinding,
    conv1d: MetalNativeTensorBinding,
    dt_bias: MetalNativeTensorBinding,
    a_log: MetalNativeTensorBinding,
    norm: MetalNativeTensorBinding,
    out_proj: MetalNativeTensorBinding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MetalNativeLayerBindings {
    attention_norm: MetalNativeTensorBinding,
    attention_post_norm: Option<MetalNativeTensorBinding>,
    attention_q_norm: Option<MetalNativeTensorBinding>,
    attention_k_norm: Option<MetalNativeTensorBinding>,
    attention_v_norm_no_scale: bool,
    attention_qkv: Option<MetalAttentionQkvBindings>,
    attention_o: Option<MetalNativeTensorBinding>,
    linear_attention: Option<MetalLinearAttentionBindings>,
    ffn_norm: MetalNativeTensorBinding,
    ffn_norm_2: Option<MetalNativeTensorBinding>,
    ffn_post_norm: Option<MetalNativeTensorBinding>,
    ffn_post_norm_1: Option<MetalNativeTensorBinding>,
    ffn_post_norm_2: Option<MetalNativeTensorBinding>,
    ffn_gate_up: MetalFfnGateUpBindings,
    ffn_down: MetalNativeTensorBinding,
    moe: Option<MetalMoeBindings>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MetalNativeModelBindings {
    token_embedding: MetalNativeTensorBinding,
    final_norm: MetalNativeTensorBinding,
    lm_head: Option<MetalNativeTensorBinding>,
    layers: Vec<MetalNativeLayerBindings>,
}

impl MetalNativeTensorBinding {
    fn from_spec(artifacts: &NativeModelArtifacts, spec: &NativeTensorSpec) -> Self {
        Self {
            spec: spec.clone(),
            resolved_path: artifacts.resolve_tensor_path(spec),
        }
    }
}

fn record_projection_binding_coverage(
    coverage: &mut MetalNativeDenseKernelCoverage,
    binding: &MetalNativeTensorBinding,
) {
    if binding.spec.source_quantized {
        coverage.projection_source_quantized_binding_count = coverage
            .projection_source_quantized_binding_count
            .saturating_add(1);
    }
    match native_dense_effective_dtype(binding.spec.dtype) {
        NativeTensorDataType::F32 => {
            coverage.projection_f32_binding_count =
                coverage.projection_f32_binding_count.saturating_add(1);
        }
        NativeTensorDataType::F16 => {
            coverage.projection_f16_binding_count =
                coverage.projection_f16_binding_count.saturating_add(1);
        }
        NativeTensorDataType::Bf16 => {
            coverage.projection_bf16_binding_count =
                coverage.projection_bf16_binding_count.saturating_add(1);
        }
        NativeTensorDataType::I8 | NativeTensorDataType::U8 => unreachable!(),
    }
}

fn record_rms_norm_binding_coverage(
    coverage: &mut MetalNativeDenseKernelCoverage,
    binding: &MetalNativeTensorBinding,
) {
    if binding.spec.source_quantized {
        coverage.rms_norm_source_quantized_binding_count = coverage
            .rms_norm_source_quantized_binding_count
            .saturating_add(1);
    }
    match native_dense_effective_dtype(binding.spec.dtype) {
        NativeTensorDataType::F32 => {
            coverage.rms_norm_f32_binding_count =
                coverage.rms_norm_f32_binding_count.saturating_add(1);
        }
        NativeTensorDataType::F16 => {
            coverage.rms_norm_f16_binding_count =
                coverage.rms_norm_f16_binding_count.saturating_add(1);
        }
        NativeTensorDataType::Bf16 => {
            coverage.rms_norm_bf16_binding_count =
                coverage.rms_norm_bf16_binding_count.saturating_add(1);
        }
        NativeTensorDataType::I8 | NativeTensorDataType::U8 => unreachable!(),
    }
}

fn record_source_quantized_binding_summary(
    summary: &mut NativeModelBindingSummary,
    binding: &MetalNativeTensorBinding,
) {
    if !binding.spec.source_quantized {
        return;
    }

    summary.source_quantized_binding_count =
        summary.source_quantized_binding_count.saturating_add(1);
    match binding.spec.source_tensor_type.as_deref() {
        Some("q4_k") => {
            summary.source_q4_k_binding_count = summary.source_q4_k_binding_count.saturating_add(1);
        }
        Some("q5_k") => {
            summary.source_q5_k_binding_count = summary.source_q5_k_binding_count.saturating_add(1);
        }
        Some("q6_k") => {
            summary.source_q6_k_binding_count = summary.source_q6_k_binding_count.saturating_add(1);
        }
        Some("q8_0") => {
            summary.source_q8_0_binding_count = summary.source_q8_0_binding_count.saturating_add(1);
        }
        _ => {}
    }
}

fn native_dense_kernel_coverage_for_model_bindings(
    bindings: &MetalNativeModelBindings,
) -> MetalNativeDenseKernelCoverage {
    let mut coverage = MetalNativeDenseKernelCoverage::default();

    if let Some(lm_head) = &bindings.lm_head {
        record_projection_binding_coverage(&mut coverage, lm_head);
    }
    record_rms_norm_binding_coverage(&mut coverage, &bindings.final_norm);

    for layer in &bindings.layers {
        record_rms_norm_binding_coverage(&mut coverage, &layer.attention_norm);
        if let Some(binding) = &layer.attention_post_norm {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        if let Some(binding) = &layer.attention_q_norm {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        if let Some(binding) = &layer.attention_k_norm {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        if let Some(attention_qkv) = &layer.attention_qkv {
            match attention_qkv {
                MetalAttentionQkvBindings::Packed(binding) => {
                    record_projection_binding_coverage(&mut coverage, binding);
                }
                MetalAttentionQkvBindings::Split { q, k, v, .. } => {
                    record_projection_binding_coverage(&mut coverage, q);
                    record_projection_binding_coverage(&mut coverage, k);
                    if let Some(binding) = v {
                        record_projection_binding_coverage(&mut coverage, binding);
                    }
                }
            }
        }
        if let Some(attention_o) = &layer.attention_o {
            record_projection_binding_coverage(&mut coverage, attention_o);
        }
        if let Some(linear_attention) = &layer.linear_attention {
            record_projection_binding_coverage(&mut coverage, &linear_attention.in_proj_qkv);
            record_projection_binding_coverage(&mut coverage, &linear_attention.in_proj_z);
            record_projection_binding_coverage(&mut coverage, &linear_attention.in_proj_a);
            record_projection_binding_coverage(&mut coverage, &linear_attention.in_proj_b);
            record_rms_norm_binding_coverage(&mut coverage, &linear_attention.norm);
            record_projection_binding_coverage(&mut coverage, &linear_attention.out_proj);
        }
        record_rms_norm_binding_coverage(&mut coverage, &layer.ffn_norm);
        if let Some(binding) = &layer.ffn_norm_2 {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        if let Some(binding) = &layer.ffn_post_norm {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        if let Some(binding) = &layer.ffn_post_norm_1 {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        if let Some(binding) = &layer.ffn_post_norm_2 {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        match &layer.ffn_gate_up {
            MetalFfnGateUpBindings::Packed(binding) => {
                record_projection_binding_coverage(&mut coverage, binding);
            }
            MetalFfnGateUpBindings::Split { gate, up } => {
                record_projection_binding_coverage(&mut coverage, gate);
                record_projection_binding_coverage(&mut coverage, up);
            }
        }
        record_projection_binding_coverage(&mut coverage, &layer.ffn_down);
        if let Some(moe) = &layer.moe {
            record_projection_binding_coverage(&mut coverage, &moe.router);
            if let Some(binding) = &moe.router_scale {
                record_rms_norm_binding_coverage(&mut coverage, binding);
            }
            match &moe.expert_gate_up {
                MetalMoeExpertGateUpBindings::Packed(binding) => {
                    record_projection_binding_coverage(&mut coverage, binding);
                }
                MetalMoeExpertGateUpBindings::Split { gate, up } => {
                    record_projection_binding_coverage(&mut coverage, gate);
                    record_projection_binding_coverage(&mut coverage, up);
                }
            }
            record_projection_binding_coverage(&mut coverage, &moe.expert_down);
        }
    }

    coverage
}

impl MetalNativeModelBindings {
    fn from_artifacts(artifacts: &NativeModelArtifacts) -> Result<Self, NativeModelError> {
        let token_embedding = required_global_tensor_binding(
            artifacts,
            NativeTensorRole::TokenEmbedding,
            "token_embedding",
        )?;
        let final_norm =
            required_global_tensor_binding(artifacts, NativeTensorRole::FinalNorm, "final_norm")?;
        let lm_head = artifacts
            .global_tensor(NativeTensorRole::LmHead)
            .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec));

        let mut layers = Vec::with_capacity(artifacts.manifest().layer_count as usize);
        for layer_index in 0..artifacts.manifest().layer_count {
            layers.push(MetalNativeLayerBindings {
                attention_norm: required_layer_tensor_binding(
                    artifacts,
                    layer_index,
                    NativeTensorRole::AttentionNorm,
                    "attention_norm",
                )?,
                attention_post_norm: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::AttentionPostNorm)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                attention_q_norm: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::AttentionQNorm)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                attention_k_norm: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::AttentionKNorm)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                attention_v_norm_no_scale: artifacts
                    .layer_uses_attention_v_norm_no_scale(layer_index),
                attention_qkv: attention_qkv_bindings(artifacts, layer_index).ok(),
                attention_o: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::AttentionO)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                linear_attention: linear_attention_bindings(artifacts, layer_index)?,
                ffn_norm: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::FfnNorm)
                    .or_else(|| {
                        artifacts.layer_tensor(layer_index, NativeTensorRole::AttentionPostNorm)
                    })
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec))
                    .ok_or_else(|| NativeModelError::InvalidManifest {
                        message: format!(
                            "layer {} is missing ffn_norm or attention_post_norm",
                            layer_index
                        ),
                    })?,
                ffn_norm_2: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::FfnNorm2)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                ffn_post_norm: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::FfnPostNorm)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                ffn_post_norm_1: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::FfnPostNorm1)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                ffn_post_norm_2: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::FfnPostNorm2)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                ffn_gate_up: ffn_gate_up_bindings(artifacts, layer_index)?,
                ffn_down: required_layer_tensor_binding(
                    artifacts,
                    layer_index,
                    NativeTensorRole::FfnDown,
                    "ffn_down",
                )?,
                moe: moe_bindings(artifacts, layer_index)?,
            });
        }

        Ok(Self {
            token_embedding,
            final_norm,
            lm_head,
            layers,
        })
    }

    fn flattened_tensor_bindings(&self) -> Vec<MetalNativeTensorBinding> {
        let mut bindings = Vec::new();
        bindings.push(self.token_embedding.clone());
        bindings.push(self.final_norm.clone());
        if let Some(lm_head) = &self.lm_head {
            bindings.push(lm_head.clone());
        }

        for layer in &self.layers {
            bindings.push(layer.attention_norm.clone());
            if let Some(attention_post_norm) = &layer.attention_post_norm {
                bindings.push(attention_post_norm.clone());
            }
            if let Some(q_norm) = &layer.attention_q_norm {
                bindings.push(q_norm.clone());
            }
            if let Some(k_norm) = &layer.attention_k_norm {
                bindings.push(k_norm.clone());
            }
            if let Some(attention_qkv) = &layer.attention_qkv {
                match attention_qkv {
                    MetalAttentionQkvBindings::Packed(binding) => bindings.push(binding.clone()),
                    MetalAttentionQkvBindings::Split { q, k, v, .. } => {
                        bindings.push(q.clone());
                        bindings.push(k.clone());
                        if let Some(binding) = v {
                            bindings.push(binding.clone());
                        }
                    }
                }
            }
            if let Some(attention_o) = &layer.attention_o {
                bindings.push(attention_o.clone());
            }
            if let Some(linear_attention) = &layer.linear_attention {
                bindings.push(linear_attention.in_proj_qkv.clone());
                bindings.push(linear_attention.in_proj_z.clone());
                bindings.push(linear_attention.in_proj_a.clone());
                bindings.push(linear_attention.in_proj_b.clone());
                bindings.push(linear_attention.conv1d.clone());
                bindings.push(linear_attention.dt_bias.clone());
                bindings.push(linear_attention.a_log.clone());
                bindings.push(linear_attention.norm.clone());
                bindings.push(linear_attention.out_proj.clone());
            }
            bindings.push(layer.ffn_norm.clone());
            if let Some(ffn_norm_2) = &layer.ffn_norm_2 {
                bindings.push(ffn_norm_2.clone());
            }
            if let Some(ffn_post_norm) = &layer.ffn_post_norm {
                bindings.push(ffn_post_norm.clone());
            }
            if let Some(ffn_post_norm_1) = &layer.ffn_post_norm_1 {
                bindings.push(ffn_post_norm_1.clone());
            }
            if let Some(ffn_post_norm_2) = &layer.ffn_post_norm_2 {
                bindings.push(ffn_post_norm_2.clone());
            }
            match &layer.ffn_gate_up {
                MetalFfnGateUpBindings::Packed(binding) => bindings.push(binding.clone()),
                MetalFfnGateUpBindings::Split { gate, up } => {
                    bindings.push(gate.clone());
                    bindings.push(up.clone());
                }
            }
            bindings.push(layer.ffn_down.clone());
            if let Some(moe) = &layer.moe {
                bindings.push(moe.router.clone());
                if let Some(router_scale) = &moe.router_scale {
                    bindings.push(router_scale.clone());
                }
                match &moe.expert_gate_up {
                    MetalMoeExpertGateUpBindings::Packed(binding) => bindings.push(binding.clone()),
                    MetalMoeExpertGateUpBindings::Split { gate, up } => {
                        bindings.push(gate.clone());
                        bindings.push(up.clone());
                    }
                }
                bindings.push(moe.expert_down.clone());
                if let Some(expert_down_scale) = &moe.expert_down_scale {
                    bindings.push(expert_down_scale.clone());
                }
            }
        }

        bindings
    }
}

#[cfg(target_os = "macos")]
struct MetalNativeTensorBufferBinding {
    meta: MetalNativeTensorBinding,
    bytes: Vec<u8>,
    native_dtype: NativeTensorDataType,
    native_buffer: Buffer,
}

#[cfg(target_os = "macos")]
struct MetalNativeModelBufferBindings {
    tensors: Vec<MetalNativeTensorBufferBinding>,
    total_bytes: u64,
}

#[cfg(target_os = "macos")]
impl MetalNativeModelBufferBindings {
    fn from_model_bindings(
        device: &Device,
        bindings: &MetalNativeModelBindings,
    ) -> Result<Self, MetalRuntimeError> {
        let flattened = bindings.flattened_tensor_bindings();
        let mut tensors = Vec::with_capacity(flattened.len());
        let mut total_bytes = 0_u64;

        for binding in flattened {
            let bytes = read_native_tensor_bytes(&binding)?;
            let (native_dtype, native_bytes) = native_dense_shadow_bytes(&binding.spec, &bytes)
                .ok_or_else(|| MetalRuntimeError::InvalidDispatchInput {
                    message: format!(
                        "native tensor {} could not be promoted into a GPU-dense buffer",
                        binding.spec.name
                    ),
                })?;
            total_bytes = total_bytes.saturating_add(bytes.len() as u64);
            tensors.push(MetalNativeTensorBufferBinding {
                meta: binding,
                bytes,
                native_dtype,
                native_buffer: new_shared_buffer_with_data(device, native_bytes.as_slice()),
            });
        }

        Ok(Self {
            tensors,
            total_bytes,
        })
    }

    fn stats(&self) -> NativeModelBindingSummary {
        let mut summary = NativeModelBindingSummary {
            bindings_prepared: true,
            buffers_bound: !self.tensors.is_empty(),
            buffer_count: self.tensors.len() as u32,
            buffer_bytes: self.total_bytes,
            ..NativeModelBindingSummary::default()
        };
        for tensor in &self.tensors {
            record_source_quantized_binding_summary(&mut summary, &tensor.meta);
        }
        summary
    }

    fn binding_for<'a>(
        &'a self,
        binding: &MetalNativeTensorBinding,
    ) -> Option<&'a MetalNativeTensorBufferBinding> {
        self.tensors
            .iter()
            .find(|candidate| candidate.meta.spec.name == binding.spec.name)
    }
}

#[cfg(target_os = "macos")]
fn read_native_tensor_bytes(
    binding: &MetalNativeTensorBinding,
) -> Result<Vec<u8>, MetalRuntimeError> {
    let length = usize::try_from(binding.spec.length_bytes).map_err(|_| {
        MetalRuntimeError::NativeTensorTooLarge {
            path: binding.resolved_path.clone(),
            length_bytes: binding.spec.length_bytes,
        }
    })?;
    let mut file = fs::File::open(&binding.resolved_path).map_err(|source| {
        MetalRuntimeError::ReadNativeTensorRange {
            path: binding.resolved_path.clone(),
            offset_bytes: binding.spec.offset_bytes,
            length_bytes: binding.spec.length_bytes,
            source,
        }
    })?;
    file.seek(SeekFrom::Start(binding.spec.offset_bytes))
        .map_err(|source| MetalRuntimeError::ReadNativeTensorRange {
            path: binding.resolved_path.clone(),
            offset_bytes: binding.spec.offset_bytes,
            length_bytes: binding.spec.length_bytes,
            source,
        })?;
    let mut bytes = vec![0_u8; length];
    file.read_exact(&mut bytes)
        .map_err(|source| MetalRuntimeError::ReadNativeTensorRange {
            path: binding.resolved_path.clone(),
            offset_bytes: binding.spec.offset_bytes,
            length_bytes: binding.spec.length_bytes,
            source,
        })?;
    Ok(bytes)
}

fn required_global_tensor_binding(
    artifacts: &NativeModelArtifacts,
    role: NativeTensorRole,
    role_label: &str,
) -> Result<MetalNativeTensorBinding, NativeModelError> {
    let spec = artifacts
        .global_tensor(role)
        .ok_or_else(|| NativeModelError::InvalidManifest {
            message: format!("missing global tensor binding for {}", role_label),
        })?;
    Ok(MetalNativeTensorBinding::from_spec(artifacts, spec))
}

fn required_layer_tensor_binding(
    artifacts: &NativeModelArtifacts,
    layer_index: u32,
    role: NativeTensorRole,
    role_label: &str,
) -> Result<MetalNativeTensorBinding, NativeModelError> {
    let spec = artifacts.layer_tensor(layer_index, role).ok_or_else(|| {
        NativeModelError::InvalidManifest {
            message: format!(
                "missing layer tensor binding for layer {} {}",
                layer_index, role_label
            ),
        }
    })?;
    Ok(MetalNativeTensorBinding::from_spec(artifacts, spec))
}

fn attention_qkv_bindings(
    artifacts: &NativeModelArtifacts,
    layer_index: u32,
) -> Result<MetalAttentionQkvBindings, NativeModelError> {
    if let Some(spec) = artifacts.layer_tensor(layer_index, NativeTensorRole::AttentionQkvPacked) {
        return Ok(MetalAttentionQkvBindings::Packed(
            MetalNativeTensorBinding::from_spec(artifacts, spec),
        ));
    }

    let value_from_key = artifacts.layer_uses_attention_value_from_key(layer_index);
    Ok(MetalAttentionQkvBindings::Split {
        q: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::AttentionQ,
            "attention_q",
        )?,
        k: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::AttentionK,
            "attention_k",
        )?,
        v: artifacts
            .layer_tensor(layer_index, NativeTensorRole::AttentionV)
            .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
        value_from_key,
    })
}

fn ffn_gate_up_bindings(
    artifacts: &NativeModelArtifacts,
    layer_index: u32,
) -> Result<MetalFfnGateUpBindings, NativeModelError> {
    if let Some(spec) = artifacts.layer_tensor(layer_index, NativeTensorRole::FfnGateUpPacked) {
        return Ok(MetalFfnGateUpBindings::Packed(
            MetalNativeTensorBinding::from_spec(artifacts, spec),
        ));
    }

    Ok(MetalFfnGateUpBindings::Split {
        gate: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::FfnGate,
            "ffn_gate",
        )?,
        up: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::FfnUp,
            "ffn_up",
        )?,
    })
}

fn moe_bindings(
    artifacts: &NativeModelArtifacts,
    layer_index: u32,
) -> Result<Option<MetalMoeBindings>, NativeModelError> {
    let Some(router_spec) = artifacts.layer_tensor(layer_index, NativeTensorRole::FfnGateInp)
    else {
        return Ok(None);
    };
    let expert_gate_up = if let Some(spec) =
        artifacts.layer_tensor(layer_index, NativeTensorRole::FfnGateUpExpsPacked)
    {
        MetalMoeExpertGateUpBindings::Packed(MetalNativeTensorBinding::from_spec(artifacts, spec))
    } else {
        MetalMoeExpertGateUpBindings::Split {
            gate: required_layer_tensor_binding(
                artifacts,
                layer_index,
                NativeTensorRole::FfnGateExps,
                "ffn_gate_exps",
            )?,
            up: required_layer_tensor_binding(
                artifacts,
                layer_index,
                NativeTensorRole::FfnUpExps,
                "ffn_up_exps",
            )?,
        }
    };
    Ok(Some(MetalMoeBindings {
        router: MetalNativeTensorBinding::from_spec(artifacts, router_spec),
        router_scale: artifacts
            .layer_tensor(layer_index, NativeTensorRole::FfnGateInpScale)
            .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
        expert_gate_up,
        expert_down: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::FfnDownExps,
            "ffn_down_exps",
        )?,
        expert_down_scale: artifacts
            .layer_tensor(layer_index, NativeTensorRole::FfnDownExpsScale)
            .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
    }))
}

fn linear_attention_bindings(
    artifacts: &NativeModelArtifacts,
    layer_index: u32,
) -> Result<Option<MetalLinearAttentionBindings>, NativeModelError> {
    let Some(_) = artifacts.layer_tensor(layer_index, NativeTensorRole::LinearAttentionInProjQkv)
    else {
        return Ok(None);
    };

    Ok(Some(MetalLinearAttentionBindings {
        in_proj_qkv: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionInProjQkv,
            "linear_attention_in_proj_qkv",
        )?,
        in_proj_z: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionInProjZ,
            "linear_attention_in_proj_z",
        )?,
        in_proj_a: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionInProjA,
            "linear_attention_in_proj_a",
        )?,
        in_proj_b: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionInProjB,
            "linear_attention_in_proj_b",
        )?,
        conv1d: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionConv1d,
            "linear_attention_conv1d",
        )?,
        dt_bias: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionDtBias,
            "linear_attention_dt_bias",
        )?,
        a_log: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionALog,
            "linear_attention_a_log",
        )?,
        norm: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionNorm,
            "linear_attention_norm",
        )?,
        out_proj: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::LinearAttentionOutProj,
            "linear_attention_out_proj",
        )?,
    }))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalDispatchNumericTrace {
    pub attention_output_bits: Vec<u32>,
    pub key_cache_checksum: u64,
    pub attention_output_checksum: u64,
    pub gather_output_checksum: u64,
    pub copy_output_checksum: u64,
    pub validation: Option<MetalNumericValidationSummary>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MetalNumericValidationSummary {
    pub expected_key_cache_checksum: u64,
    pub expected_attention_output_checksum: u64,
    pub expected_gather_output_checksum: u64,
    pub expected_copy_output_checksum: u64,
    pub attention_max_abs_diff_microunits: u32,
}

#[derive(Clone, Copy)]
struct MetalDispatchKvCacheSeed<'a> {
    key_cache: &'a [f32],
    value_cache: &'a [f32],
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct OwnedMetalDispatchKvCacheSeed {
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl OwnedMetalDispatchKvCacheSeed {
    fn as_seed(&self) -> MetalDispatchKvCacheSeed<'_> {
        MetalDispatchKvCacheSeed {
            key_cache: &self.key_cache,
            value_cache: &self.value_cache,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct MetalDispatchKvCacheSnapshot {
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl MetalDispatchKvCacheSnapshot {
    fn from_reference_for_workload(
        workload: &MetalDispatchWorkload,
        reference: &ReferenceNumericPath,
    ) -> Self {
        let mut snapshot = Self {
            key_cache: reference.key_cache.clone(),
            value_cache: reference.value_cache.clone(),
        };
        merge_copy_targets_into_cache_snapshot(
            workload,
            &mut snapshot.key_cache,
            &mut snapshot.value_cache,
            &reference.copy_key,
            &reference.copy_value,
        );
        snapshot
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default, PartialEq)]
struct ModelBoundDirectDecodeResult {
    tokens: Vec<(crate::ids::RequestId, u32)>,
    logits_outputs: Vec<RequestLogitsOutput>,
    model_bound_ffn_decode: bool,
    native_logits_projection_decode: bool,
    execution_tally: PrefixAttentionExecutionTally,
    native_dense_tally: DirectDecodeNativeDenseTally,
}

#[derive(Clone, Debug, Default)]
struct MetalBringupExecutionFlags {
    model_bound_ffn_decode: bool,
    native_logits_projection_decode: bool,
    complete_model_forward_supported: bool,
    real_model_forward: bool,
    prefix_attention_tally: PrefixAttentionExecutionTally,
    direct_decode_native_dense_tally: DirectDecodeNativeDenseTally,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DirectDecodeNativeDenseTally {
    native_projection_rows: u32,
    cpu_projection_rows: u32,
    native_rms_norm_elements: u32,
    cpu_rms_norm_elements: u32,
    native_ffn_activation_elements: u32,
    cpu_ffn_activation_elements: u32,
    native_residual_add_elements: u32,
    cpu_residual_add_elements: u32,
    native_scale_elements: u32,
    cpu_scale_elements: u32,
    native_batched_logits_group_count: u32,
    native_batched_logits_token_count: u32,
    batched_group_fallback_count: u32,
    batched_group_fallback_token_count: u32,
}

#[cfg(target_os = "macos")]
impl DirectDecodeNativeDenseTally {
    fn merge(mut self, other: Self) -> Self {
        self.native_projection_rows = self
            .native_projection_rows
            .saturating_add(other.native_projection_rows);
        self.cpu_projection_rows = self
            .cpu_projection_rows
            .saturating_add(other.cpu_projection_rows);
        self.native_rms_norm_elements = self
            .native_rms_norm_elements
            .saturating_add(other.native_rms_norm_elements);
        self.cpu_rms_norm_elements = self
            .cpu_rms_norm_elements
            .saturating_add(other.cpu_rms_norm_elements);
        self.native_ffn_activation_elements = self
            .native_ffn_activation_elements
            .saturating_add(other.native_ffn_activation_elements);
        self.cpu_ffn_activation_elements = self
            .cpu_ffn_activation_elements
            .saturating_add(other.cpu_ffn_activation_elements);
        self.native_residual_add_elements = self
            .native_residual_add_elements
            .saturating_add(other.native_residual_add_elements);
        self.cpu_residual_add_elements = self
            .cpu_residual_add_elements
            .saturating_add(other.cpu_residual_add_elements);
        self.native_scale_elements = self
            .native_scale_elements
            .saturating_add(other.native_scale_elements);
        self.cpu_scale_elements = self
            .cpu_scale_elements
            .saturating_add(other.cpu_scale_elements);
        self.native_batched_logits_group_count = self
            .native_batched_logits_group_count
            .saturating_add(other.native_batched_logits_group_count);
        self.native_batched_logits_token_count = self
            .native_batched_logits_token_count
            .saturating_add(other.native_batched_logits_token_count);
        self.batched_group_fallback_count = self
            .batched_group_fallback_count
            .saturating_add(other.batched_group_fallback_count);
        self.batched_group_fallback_token_count = self
            .batched_group_fallback_token_count
            .saturating_add(other.batched_group_fallback_token_count);
        self
    }

    fn record_projection_rows(mut self, row_count: usize, used_native: bool) -> Self {
        let row_count = saturating_usize_to_u32(row_count);
        if used_native {
            self.native_projection_rows = self.native_projection_rows.saturating_add(row_count);
        } else {
            self.cpu_projection_rows = self.cpu_projection_rows.saturating_add(row_count);
        }
        self
    }

    fn record_rms_norm_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_rms_norm_elements =
                self.native_rms_norm_elements.saturating_add(element_count);
        } else {
            self.cpu_rms_norm_elements = self.cpu_rms_norm_elements.saturating_add(element_count);
        }
        self
    }

    fn record_ffn_activation_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_ffn_activation_elements = self
                .native_ffn_activation_elements
                .saturating_add(element_count);
        } else {
            self.cpu_ffn_activation_elements = self
                .cpu_ffn_activation_elements
                .saturating_add(element_count);
        }
        self
    }

    fn record_residual_add_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_residual_add_elements = self
                .native_residual_add_elements
                .saturating_add(element_count);
        } else {
            self.cpu_residual_add_elements =
                self.cpu_residual_add_elements.saturating_add(element_count);
        }
        self
    }

    fn record_scale_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_scale_elements = self.native_scale_elements.saturating_add(element_count);
        } else {
            self.cpu_scale_elements = self.cpu_scale_elements.saturating_add(element_count);
        }
        self
    }

    fn record_batched_logits_group(mut self, token_count: usize) -> Self {
        if token_count > 1 {
            self.native_batched_logits_group_count =
                self.native_batched_logits_group_count.saturating_add(1);
            self.native_batched_logits_token_count = self
                .native_batched_logits_token_count
                .saturating_add(saturating_usize_to_u32(token_count));
        }
        self
    }

    fn record_batched_group_fallback(mut self, token_count: usize) -> Self {
        if token_count > 1 {
            self.batched_group_fallback_count = self.batched_group_fallback_count.saturating_add(1);
            self.batched_group_fallback_token_count = self
                .batched_group_fallback_token_count
                .saturating_add(saturating_usize_to_u32(token_count));
        }
        self
    }

    fn scale_for_selected_items(self, selected_count: usize, total_count: usize) -> Self {
        if selected_count == 0 || total_count == 0 {
            return Self::default();
        }
        if selected_count >= total_count {
            return self;
        }

        let scale = |value: u32| -> u32 {
            ((u64::from(value) * selected_count as u64) / total_count as u64)
                .try_into()
                .unwrap_or(u32::MAX)
        };

        Self {
            native_projection_rows: scale(self.native_projection_rows),
            cpu_projection_rows: scale(self.cpu_projection_rows),
            native_rms_norm_elements: scale(self.native_rms_norm_elements),
            cpu_rms_norm_elements: scale(self.cpu_rms_norm_elements),
            native_ffn_activation_elements: scale(self.native_ffn_activation_elements),
            cpu_ffn_activation_elements: scale(self.cpu_ffn_activation_elements),
            native_residual_add_elements: scale(self.native_residual_add_elements),
            cpu_residual_add_elements: scale(self.cpu_residual_add_elements),
            native_scale_elements: scale(self.native_scale_elements),
            cpu_scale_elements: scale(self.cpu_scale_elements),
            native_batched_logits_group_count: u32::from(
                self.native_batched_logits_group_count > 0 && selected_count > 1,
            ),
            native_batched_logits_token_count: if self.native_batched_logits_group_count > 0
                && selected_count > 1
            {
                saturating_usize_to_u32(selected_count)
            } else {
                0
            },
            batched_group_fallback_count: u32::from(
                self.batched_group_fallback_count > 0 && selected_count > 1,
            ),
            batched_group_fallback_token_count: if self.batched_group_fallback_count > 0
                && selected_count > 1
            {
                saturating_usize_to_u32(selected_count)
            } else {
                0
            },
        }
    }
}

#[cfg(target_os = "macos")]
fn prefix_attention_tally_from_native_dense_tally(
    tally: DirectDecodeNativeDenseTally,
) -> PrefixAttentionExecutionTally {
    PrefixAttentionExecutionTally::default()
        .record_projection_rows(tally.native_projection_rows as usize, true)
        .record_projection_rows(tally.cpu_projection_rows as usize, false)
        .record_rms_norm_elements(tally.native_rms_norm_elements as usize, true)
        .record_rms_norm_elements(tally.cpu_rms_norm_elements as usize, false)
        .record_ffn_activation_elements(tally.native_ffn_activation_elements as usize, true)
        .record_ffn_activation_elements(tally.cpu_ffn_activation_elements as usize, false)
        .record_residual_add_elements(tally.native_residual_add_elements as usize, true)
        .record_residual_add_elements(tally.cpu_residual_add_elements as usize, false)
        .record_scale_elements(tally.native_scale_elements as usize, true)
        .record_scale_elements(tally.cpu_scale_elements as usize, false)
}

#[cfg(target_os = "macos")]
fn direct_decode_native_dense_tally_from_prefix_attention_tally(
    tally: PrefixAttentionExecutionTally,
) -> DirectDecodeNativeDenseTally {
    DirectDecodeNativeDenseTally::default()
        .record_projection_rows(tally.native_projection_row_count() as usize, true)
        .record_projection_rows(tally.cpu_projection_row_count() as usize, false)
        .record_rms_norm_elements(tally.native_rms_norm_element_count() as usize, true)
        .record_rms_norm_elements(tally.cpu_rms_norm_element_count() as usize, false)
        .record_ffn_activation_elements(tally.native_ffn_activation_element_count() as usize, true)
        .record_ffn_activation_elements(tally.cpu_ffn_activation_element_count() as usize, false)
        .record_residual_add_elements(tally.native_residual_add_element_count() as usize, true)
        .record_residual_add_elements(tally.cpu_residual_add_element_count() as usize, false)
        .record_scale_elements(tally.native_scale_element_count() as usize, true)
        .record_scale_elements(tally.cpu_scale_element_count() as usize, false)
}

#[derive(Clone, Debug, Default)]
struct MetalPersistentLayerKvCache {
    block_size_tokens: u32,
    numeric_layout: MetalDispatchNumericLayout,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    initialized_slots: Vec<bool>,
}

impl MetalPersistentLayerKvCache {
    fn ensure_capacity(&mut self, workload: &MetalDispatchWorkload) {
        let required_block_size = workload.kv_metadata.block_size_tokens.max(1);
        let required_slot_capacity = workload.kv_slot_capacity.max(1) as usize;
        let required_numeric_capacity = workload.slot_numeric_capacity() as usize;

        if self.block_size_tokens != 0
            && (self.block_size_tokens != required_block_size
                || self.numeric_layout != workload.numeric_layout)
        {
            self.key_cache.clear();
            self.value_cache.clear();
            self.initialized_slots.clear();
        }

        self.block_size_tokens = required_block_size;
        self.numeric_layout = workload.numeric_layout;
        if self.key_cache.len() < required_numeric_capacity {
            self.key_cache.resize(required_numeric_capacity, 0.0);
        }
        if self.value_cache.len() < required_numeric_capacity {
            self.value_cache.resize(required_numeric_capacity, 0.0);
        }
        if self.initialized_slots.len() < required_slot_capacity {
            self.initialized_slots.resize(required_slot_capacity, false);
        }
    }

    fn seed_for_workload<'a>(
        &'a mut self,
        workload: &MetalDispatchWorkload,
    ) -> MetalDispatchKvCacheSeed<'a> {
        self.ensure_capacity(workload);
        let required_numeric_capacity = workload.slot_numeric_capacity() as usize;
        MetalDispatchKvCacheSeed {
            key_cache: &self.key_cache[..required_numeric_capacity],
            value_cache: &self.value_cache[..required_numeric_capacity],
        }
    }

    fn slot_initialized(&self, slot: u32) -> bool {
        self.initialized_slots
            .get(slot as usize)
            .copied()
            .unwrap_or(false)
    }

    fn apply_snapshot(
        &mut self,
        workload: &MetalDispatchWorkload,
        snapshot: &MetalDispatchKvCacheSnapshot,
    ) {
        self.ensure_capacity(workload);
        let required_numeric_capacity = workload.slot_numeric_capacity() as usize;
        if snapshot.key_cache.len() < required_numeric_capacity
            || snapshot.value_cache.len() < required_numeric_capacity
        {
            return;
        }

        self.key_cache[..required_numeric_capacity]
            .copy_from_slice(&snapshot.key_cache[..required_numeric_capacity]);
        self.value_cache[..required_numeric_capacity]
            .copy_from_slice(&snapshot.value_cache[..required_numeric_capacity]);
        for &slot in &workload.kv_metadata.slot_mapping {
            if let Some(initialized) = self.initialized_slots.get_mut(slot as usize) {
                *initialized = true;
            }
        }
        propagate_initialized_copy_targets(workload, &mut self.initialized_slots);
    }
}

pub struct MetalBringupRunner {
    bringup: MetalRuntimeBringup,
    model_artifacts: Option<NativeModelArtifacts>,
    model_bindings: Option<MetalNativeModelBindings>,
    #[cfg(target_os = "macos")]
    model_buffers: Option<MetalNativeModelBufferBindings>,
    #[cfg(target_os = "macos")]
    prefix_layer_caches: Vec<Mutex<MetalPersistentLayerKvCache>>,
    #[cfg(target_os = "macos")]
    linear_request_states: Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>,
    last_dispatch: Mutex<Option<MetalDispatchTrace>>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default, PartialEq)]
struct MetalLinearRequestState {
    layers: BTreeMap<u32, MetalLinearLayerState>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default, PartialEq)]
struct MetalLinearLayerState {
    processed_tokens: usize,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl fmt::Debug for MetalBringupRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let buffer_stats = self.model_buffer_stats();
        f.debug_struct("MetalBringupRunner")
            .field("bringup_report", self.bringup.report())
            .field("model_artifacts", &self.model_artifacts_summary())
            .field("model_bindings_prepared", &self.model_bindings.is_some())
            .field("model_buffers_bound", &buffer_stats.buffers_bound)
            .field("model_buffer_count", &buffer_stats.buffer_count)
            .field("model_buffer_bytes", &buffer_stats.buffer_bytes)
            .field(
                "prefix_layer_cache_count",
                &self
                    .model_bindings
                    .as_ref()
                    .map(|bindings| bindings.layers.len().saturating_sub(1))
                    .unwrap_or(0),
            )
            .finish()
    }
}

impl MetalBringupRunner {
    pub fn from_build_dir(path: impl AsRef<Path>) -> Result<Self, MetalRuntimeError> {
        Self::from_build_dir_and_model_artifacts(path, None)
    }

    pub fn from_build_dir_and_model_artifacts(
        path: impl AsRef<Path>,
        model_artifacts_dir: Option<&Path>,
    ) -> Result<Self, MetalRuntimeError> {
        let model_artifacts = model_artifacts_dir
            .map(NativeModelArtifacts::from_dir)
            .transpose()?;
        Self::from_assets_and_model_artifacts(
            MetalKernelAssets::from_build_dir(path)?,
            model_artifacts,
        )
    }

    pub fn from_assets(assets: MetalKernelAssets) -> Result<Self, MetalRuntimeError> {
        Self::from_assets_and_model_artifacts(assets, None)
    }

    pub fn from_assets_and_model_artifacts(
        assets: MetalKernelAssets,
        model_artifacts: Option<NativeModelArtifacts>,
    ) -> Result<Self, MetalRuntimeError> {
        let bringup = MetalRuntimeBringup::from_assets(assets)?;
        let model_bindings = model_artifacts
            .as_ref()
            .map(MetalNativeModelBindings::from_artifacts)
            .transpose()?;
        #[cfg(target_os = "macos")]
        let model_buffers = model_bindings
            .as_ref()
            .map(|bindings| {
                MetalNativeModelBufferBindings::from_model_bindings(&bringup.state.device, bindings)
            })
            .transpose()?;
        #[cfg(target_os = "macos")]
        let prefix_layer_caches = model_bindings
            .as_ref()
            .map(|bindings| {
                (0..bindings.layers.len().saturating_sub(1))
                    .map(|_| Mutex::new(MetalPersistentLayerKvCache::default()))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            bringup,
            model_artifacts,
            model_bindings,
            #[cfg(target_os = "macos")]
            model_buffers,
            #[cfg(target_os = "macos")]
            prefix_layer_caches,
            #[cfg(target_os = "macos")]
            linear_request_states: Mutex::new(BTreeMap::new()),
            last_dispatch: Mutex::new(None),
        })
    }

    pub fn bringup(&self) -> &MetalRuntimeBringup {
        &self.bringup
    }

    pub fn last_dispatch(&self) -> Option<MetalDispatchTrace> {
        self.last_dispatch
            .lock()
            .expect("metal bring-up runner dispatch mutex should not be poisoned")
            .clone()
    }

    pub fn model_artifacts(&self) -> Option<&NativeModelArtifacts> {
        self.model_artifacts.as_ref()
    }

    #[cfg(target_os = "macos")]
    fn prefix_layer_caches(&self) -> Option<&[Mutex<MetalPersistentLayerKvCache>]> {
        (!self.prefix_layer_caches.is_empty()).then_some(self.prefix_layer_caches.as_slice())
    }

    fn staged_inputs_for_workload(
        &self,
        input: &RunnerInput,
        workload: &MetalDispatchWorkload,
    ) -> Result<MetalDispatchStagedInputs, MetalRuntimeError> {
        #[cfg(target_os = "macos")]
        {
            resolve_runtime_staged_inputs(
                input,
                workload,
                self.model_artifacts.as_ref(),
                self.model_bindings.as_ref(),
                self.model_buffers.as_ref(),
                self.prefix_layer_caches(),
                Some(&self.linear_request_states),
                Some(&self.bringup),
            )
        }

        #[cfg(not(target_os = "macos"))]
        {
            if self.model_artifacts.is_some() || self.model_bindings.is_some() {
                return Err(MetalRuntimeError::InvalidDispatchInput {
                    message: "native model-conditioned staged inputs require macOS Metal bindings"
                        .to_string(),
                });
            }

            Ok(synthetic_staged_inputs(workload))
        }
    }

    fn dispatch_runtime_info(&self) -> MetalDispatchRuntimeInfo {
        let mut runtime = MetalDispatchRuntimeInfo::from_bringup_report(self.bringup.report());
        let buffer_stats = self.model_buffer_stats();
        self.populate_runtime_model_state(&mut runtime, None, &buffer_stats);
        runtime
    }

    fn dispatch_runtime_info_for_source(
        &self,
        source: MetalStagedInputSource,
    ) -> MetalDispatchRuntimeInfo {
        let mut runtime = MetalDispatchRuntimeInfo::from_bringup_report(self.bringup.report());
        let buffer_stats = self.model_buffer_stats();
        self.populate_runtime_model_state(&mut runtime, Some(source), &buffer_stats);
        runtime
    }

    fn populate_runtime_model_state(
        &self,
        runtime: &mut MetalDispatchRuntimeInfo,
        source: Option<MetalStagedInputSource>,
        buffer_stats: &NativeModelBindingSummary,
    ) {
        let model_conditioned_inputs = matches!(
            source,
            Some(
                MetalStagedInputSource::ModelConditionedMiniProjection
                    | MetalStagedInputSource::ModelConditionedCpuPrefixAttention
                    | MetalStagedInputSource::ModelConditionedNativePrefixAttention
                    | MetalStagedInputSource::ModelConditionedMixedPrefixAttention
            )
        );
        runtime.model_conditioned_inputs = model_conditioned_inputs;
        runtime.real_model_tensor_inputs = model_conditioned_inputs && buffer_stats.buffers_bound;
        runtime.model_bindings_prepared = self.model_bindings.is_some();
        runtime.model_buffers_bound = buffer_stats.buffers_bound;
        runtime.model_buffer_count = buffer_stats.buffer_count;
        runtime.model_buffer_bytes = buffer_stats.buffer_bytes;
        runtime.native_dense_kernel_coverage = self
            .model_bindings
            .as_ref()
            .map(native_dense_kernel_coverage_for_model_bindings)
            .unwrap_or_default();
        runtime.model = self.model_artifacts_summary();
        runtime.complete_model_forward_supported =
            complete_model_forward_support_for_source(runtime.model.as_ref(), source);
    }

    fn model_artifacts_summary(&self) -> Option<NativeModelArtifactsSummary> {
        self.model_artifacts
            .as_ref()
            .map(NativeModelArtifacts::summary)
    }

    fn model_buffer_stats(&self) -> NativeModelBindingSummary {
        #[cfg(target_os = "macos")]
        {
            self.model_buffers
                .as_ref()
                .map(MetalNativeModelBufferBindings::stats)
                .unwrap_or_else(|| {
                    let mut summary = NativeModelBindingSummary {
                        bindings_prepared: self.model_bindings.is_some(),
                        ..NativeModelBindingSummary::default()
                    };
                    if let Some(bindings) = &self.model_bindings {
                        for binding in bindings.flattened_tensor_bindings() {
                            record_source_quantized_binding_summary(&mut summary, &binding);
                        }
                    }
                    summary
                })
        }

        #[cfg(not(target_os = "macos"))]
        {
            NativeModelBindingSummary {
                bindings_prepared: self.model_bindings.is_some(),
                ..NativeModelBindingSummary::default()
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn try_model_bound_direct_decode_tokens(
        &self,
        input: &RunnerInput,
        attention_output_bits: &[u32],
        staged_inputs: &MetalDispatchStagedInputs,
        _workload: &MetalDispatchWorkload,
        runtime: &MetalDispatchRuntimeInfo,
    ) -> ModelBoundDirectDecodeResult {
        if !runtime.real_model_tensor_inputs {
            return ModelBoundDirectDecodeResult::default();
        }

        derive_model_bound_direct_decode_result(
            input,
            attention_output_bits,
            self.model_artifacts.as_ref(),
            self.model_bindings.as_ref(),
            self.model_buffers.as_ref(),
            staged_inputs.final_layer_hidden_state_cache.as_ref(),
            self.prefix_layer_caches(),
            Some(&self.linear_request_states),
            Some(&self.bringup),
        )
    }

    #[cfg(not(target_os = "macos"))]
    fn try_model_bound_direct_decode_tokens(
        &self,
        _input: &RunnerInput,
        _attention_output_bits: &[u32],
        _staged_inputs: &MetalDispatchStagedInputs,
        _workload: &MetalDispatchWorkload,
        _runtime: &MetalDispatchRuntimeInfo,
    ) -> ModelBoundDirectDecodeResult {
        ModelBoundDirectDecodeResult::default()
    }
}

impl ExecutionRunner for MetalBringupRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        if let Err(error) = self
            .bringup
            .assets()
            .validate_block_size_tokens(input.block_size_tokens)
        {
            let runtime = self.dispatch_runtime_info();
            return failed_runner_output_from_input(
                &input,
                error.to_string(),
                &fallback_failed_workload(&input),
                Some(&runtime),
            );
        }
        let workload = match MetalDispatchWorkload::from_runner_input(&input) {
            Ok(workload) => workload,
            Err(error) => {
                let runtime = self.dispatch_runtime_info();
                return failed_runner_output_from_input(
                    &input,
                    error.to_string(),
                    &fallback_failed_workload(&input),
                    Some(&runtime),
                );
            }
        };
        let staged_inputs = match self.staged_inputs_for_workload(&input, &workload) {
            Ok(staged_inputs) => staged_inputs,
            Err(error) => {
                let runtime = self.dispatch_runtime_info();
                return failed_runner_output_from_input(
                    &input,
                    error.to_string(),
                    &workload,
                    Some(&runtime),
                );
            }
        };
        let attention_config = self.model_artifacts.as_ref().map(|artifacts| {
            native_model_reference_attention_config(
                artifacts,
                staged_inputs.layout.head_dim as usize,
            )
        });
        match self
            .bringup
            .dispatch_numeric_workload_with_attention_config(
                &workload,
                &staged_inputs,
                attention_config,
            ) {
            Ok(mut trace) => {
                let runtime = self.dispatch_runtime_info_for_source(staged_inputs.source);
                trace.runtime.model_conditioned_inputs = runtime.model_conditioned_inputs;
                trace.runtime.real_model_tensor_inputs = runtime.real_model_tensor_inputs;
                trace.runtime.complete_model_forward_supported =
                    runtime.complete_model_forward_supported;
                trace.runtime.model_bindings_prepared = runtime.model_bindings_prepared;
                trace.runtime.model_buffers_bound = runtime.model_buffers_bound;
                trace.runtime.model_buffer_count = runtime.model_buffer_count;
                trace.runtime.model_buffer_bytes = runtime.model_buffer_bytes;
                trace.runtime.native_dense_kernel_coverage = runtime.native_dense_kernel_coverage;
                trace.runtime.model = runtime.model;

                let direct_decode_result = self.try_model_bound_direct_decode_tokens(
                    &input,
                    &trace.numeric.attention_output_bits,
                    &staged_inputs,
                    &workload,
                    &trace.runtime,
                );
                let mut output = successful_runner_output_from_input(&input);
                let resolved_sample_request_ids =
                    apply_deterministic_sample_tokens_to_runner_output(
                        &input,
                        &mut output,
                        &direct_decode_result.tokens,
                    );
                let direct_decode_logits_outputs = direct_decode_result
                    .logits_outputs
                    .into_iter()
                    .filter(|output| !resolved_sample_request_ids.contains(&output.request_id))
                    .collect::<Vec<_>>();
                let direct_decode_tokens = filter_model_bound_tokens_by_mode(
                    &input,
                    &direct_decode_result.tokens,
                    ExecutionMode::Decode,
                );
                apply_owned_direct_decode_logits_to_runner_output(
                    &mut output,
                    direct_decode_logits_outputs,
                );
                let complete_model_forward_supported =
                    runtime_reports_complete_model_forward(&trace.runtime);
                let real_model_forward = completed_real_model_forward_step(
                    &input,
                    &output,
                    &trace.runtime,
                    &direct_decode_tokens,
                );
                let execution_tally = staged_inputs
                    .prefix_attention_tally
                    .merge(direct_decode_result.execution_tally);
                trace.execution = metal_dispatch_execution_info(
                    &output,
                    &direct_decode_tokens,
                    direct_decode_result.model_bound_ffn_decode,
                    real_model_forward,
                    execution_tally,
                    direct_decode_result.native_dense_tally,
                );
                *self
                    .last_dispatch
                    .lock()
                    .expect("metal bring-up runner dispatch mutex should not be poisoned") =
                    Some(trace.clone());
                annotate_successful_dispatch(&mut output.route_metadata, &trace);
                annotate_bringup_execution_flags(
                    &mut output.route_metadata,
                    &trace.runtime,
                    &direct_decode_tokens,
                    MetalBringupExecutionFlags {
                        model_bound_ffn_decode: direct_decode_result.model_bound_ffn_decode,
                        native_logits_projection_decode: direct_decode_result
                            .native_logits_projection_decode,
                        complete_model_forward_supported,
                        real_model_forward,
                        prefix_attention_tally: execution_tally,
                        direct_decode_native_dense_tally: direct_decode_result.native_dense_tally,
                    },
                );
                annotate_staged_input_source(&mut output.route_metadata, staged_inputs.source);
                output
            }
            Err(error) => {
                *self
                    .last_dispatch
                    .lock()
                    .expect("metal bring-up runner dispatch mutex should not be poisoned") = None;
                let runtime = self.dispatch_runtime_info_for_source(staged_inputs.source);
                failed_runner_output_from_input(
                    &input,
                    error.to_string(),
                    &workload,
                    Some(&runtime),
                )
            }
        }
    }

    fn metal_dispatch_trace(&self) -> Option<MetalDispatchTrace> {
        self.last_dispatch()
    }

    fn release_request_state(&self, request_id: crate::ids::RequestId) {
        #[cfg(target_os = "macos")]
        if let Ok(mut states) = self.linear_request_states.lock() {
            states.remove(&request_id);
        }
    }

    fn native_model_artifacts_summary(&self) -> Option<NativeModelArtifactsSummary> {
        self.model_artifacts_summary()
    }

    fn native_model_binding_summary(&self) -> Option<NativeModelBindingSummary> {
        self.model_artifacts.as_ref()?;
        Some(self.model_buffer_stats())
    }
}

/// Contract-only validator for compiled Metal assets.
///
/// This surface intentionally validates manifests, metallib loading, and
/// supported block sizes without presenting itself as an execution backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalAssetValidator {
    assets: MetalKernelAssets,
    metallib: MetalKernelBinary,
    resolved_kernel_names: Vec<String>,
}

impl MetalAssetValidator {
    pub fn from_build_dir(path: impl AsRef<Path>) -> Result<Self, MetalRuntimeError> {
        Self::from_assets(MetalKernelAssets::from_build_dir(path)?)
    }

    pub fn from_assets(assets: MetalKernelAssets) -> Result<Self, MetalRuntimeError> {
        let metallib = load_compiled_metallib_binary(&assets)?;
        let resolved_kernel_names = resolve_required_kernel_names(&assets)?;

        Ok(Self {
            assets,
            metallib,
            resolved_kernel_names,
        })
    }

    pub fn assets(&self) -> &MetalKernelAssets {
        &self.assets
    }

    pub fn metallib(&self) -> &MetalKernelBinary {
        &self.metallib
    }

    pub fn resolved_kernel_names(&self) -> &[String] {
        &self.resolved_kernel_names
    }

    pub fn validate_block_size_tokens(
        &self,
        block_size_tokens: u32,
    ) -> Result<(), MetalRuntimeError> {
        self.assets.validate_block_size_tokens(block_size_tokens)
    }
}

#[derive(Debug, Error)]
pub enum MetalRuntimeError {
    #[error(transparent)]
    NativeModel(#[from] NativeModelError),
    #[error("failed to read JSON file {path}: {source}")]
    ReadJson {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse JSON file {path}: {source}")]
    ParseJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to write build artifact {path}: {source}")]
    WriteBuildArtifact {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("metal kernel manifest is invalid: {message}")]
    InvalidManifest { message: String },
    #[error("metal build report is invalid: {message}")]
    InvalidBuildReport { message: String },
    #[error("metal build artifact is missing or empty: {path}")]
    MissingBuildArtifact { path: PathBuf },
    #[error("metal build report is not compiled; status={status:?}")]
    BuildNotCompiled { status: MetalBuildStatus },
    #[error("metal kernel {kernel_name} is not declared in the manifest")]
    UnknownKernel { kernel_name: String },
    #[error("metal runtime bring-up is only available on macOS; host_os={host_os}")]
    UnsupportedPlatform { host_os: &'static str },
    #[error("metal runtime bring-up could not find a system default MTLDevice")]
    NoSystemDevice,
    #[error("failed to load compiled metallib {path} into the Metal runtime: {message}")]
    LoadCompiledLibrary { path: PathBuf, message: String },
    #[error(
        "compiled metallib {path} function inventory does not match manifest: missing={missing:?}, extra={extra:?}"
    )]
    CompiledKernelInventoryMismatch {
        path: PathBuf,
        missing: Vec<String>,
        extra: Vec<String>,
    },
    #[error("failed to resolve Metal function {function_name} from {path}: {message}")]
    ResolveCompiledKernel {
        path: PathBuf,
        function_name: String,
        message: String,
    },
    #[error(
        "failed to build compute pipeline for {function_name} on device {device_name}: {message}"
    )]
    CreateComputePipeline {
        function_name: String,
        device_name: String,
        message: String,
    },
    #[error("metal dispatch input is invalid: {message}")]
    InvalidDispatchInput { message: String },
    #[error("metal numeric reference validation failed for {stage}: {message}")]
    NumericValidationMismatch {
        stage: &'static str,
        message: String,
    },
    #[error(
        "phase1 native Metal path only supports block_size_tokens {supported_block_size_tokens:?} (default {default_block_size_tokens}); got {block_size_tokens}"
    )]
    UnsupportedNativeBlockSize {
        block_size_tokens: u32,
        default_block_size_tokens: u32,
        supported_block_size_tokens: Vec<u32>,
    },
    #[error("Metal command buffer did not complete successfully; final_status={status:?}")]
    CommandBufferNotCompleted { status: MetalCommandBufferStatus },
    #[error(
        "failed to read native tensor bytes from {path} at offset {offset_bytes} length {length_bytes}: {source}"
    )]
    ReadNativeTensorRange {
        path: PathBuf,
        offset_bytes: u64,
        length_bytes: u64,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "native tensor {path} length_bytes {length_bytes} exceeds addressable buffer size on this host"
    )]
    NativeTensorTooLarge { path: PathBuf, length_bytes: u64 },
}

fn annotate_successful_dispatch(route_metadata: &mut RouteMetadata, trace: &MetalDispatchTrace) {
    route_metadata
        .crossover_decisions
        .push(("metal_dispatch_completed".to_string(), 1));
    annotate_runtime_summary(route_metadata, &trace.runtime);
    route_metadata.crossover_decisions.push((
        "metal_dispatch_kernel_count".to_string(),
        trace.kernels.len() as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_scratch_elements".to_string(),
        trace.workload.scratch_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefill_requests".to_string(),
        trace.workload.prefill_requests,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_decode_requests".to_string(),
        trace.workload.decode_requests,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_slot_mapping_entries".to_string(),
        trace.workload.kv_metadata.slot_mapping.len() as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_attention_block_refs".to_string(),
        trace.workload.kv_metadata.attention_block_table.len() as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_gather_block_refs".to_string(),
        trace.workload.kv_metadata.gather_block_table.len() as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_scheduled_cu_seq_refs".to_string(),
        trace.workload.kv_metadata.scheduled_cu_seq_lens.len() as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_copy_pairs".to_string(),
        trace.workload.kv_metadata.copy_block_mapping.len() as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_copy_workload_elements".to_string(),
        trace.workload.copy_numeric_elements(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_slot_capacity".to_string(),
        trace.arena.slot_capacity,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_arena_token_capacity".to_string(),
        trace.arena.token_capacity,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_arena_sequence_capacity".to_string(),
        trace.arena.sequence_capacity,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_arena_gather_output_capacity".to_string(),
        trace.arena.gather_output_capacity,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_arena_reused".to_string(),
        u32::from(trace.arena.reused_existing),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_arena_grew".to_string(),
        u32::from(trace.arena.grew_existing),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_key_cache_checksum_lo".to_string(),
        trace.numeric.key_cache_checksum as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_attention_checksum_lo".to_string(),
        trace.numeric.attention_output_checksum as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_gather_checksum_lo".to_string(),
        trace.numeric.gather_output_checksum as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_copy_checksum_lo".to_string(),
        trace.numeric.copy_output_checksum as u32,
    ));
    route_metadata
        .crossover_decisions
        .push(("metal_dispatch_attention_uses_gathered_kv".to_string(), 1));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_numeric_head_size".to_string(),
        trace.workload.numeric_layout.head_size(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_numeric_head_count".to_string(),
        trace.workload.numeric_layout.head_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_numeric_head_dim".to_string(),
        trace.workload.numeric_layout.head_dim,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_attention_output_elements".to_string(),
        trace.workload.attention_numeric_elements(),
    ));
    if let Some(validation) = &trace.numeric.validation {
        route_metadata
            .crossover_decisions
            .push(("metal_dispatch_numeric_reference_validated".to_string(), 1));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_expected_key_cache_checksum_lo".to_string(),
            validation.expected_key_cache_checksum as u32,
        ));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_expected_attention_checksum_lo".to_string(),
            validation.expected_attention_output_checksum as u32,
        ));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_expected_gather_checksum_lo".to_string(),
            validation.expected_gather_output_checksum as u32,
        ));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_expected_copy_checksum_lo".to_string(),
            validation.expected_copy_output_checksum as u32,
        ));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_attention_max_abs_diff_microunits".to_string(),
            validation.attention_max_abs_diff_microunits,
        ));
    }
}

fn annotate_failed_dispatch(
    route_metadata: &mut RouteMetadata,
    workload: &MetalDispatchWorkload,
    runtime: Option<&MetalDispatchRuntimeInfo>,
) {
    route_metadata
        .crossover_decisions
        .push(("metal_dispatch_failed".to_string(), 1));
    if let Some(runtime) = runtime {
        annotate_runtime_summary(route_metadata, runtime);
    }
    route_metadata.crossover_decisions.push((
        "metal_dispatch_scratch_elements".to_string(),
        workload.scratch_elements,
    ));
}

fn annotate_runtime_summary(
    route_metadata: &mut RouteMetadata,
    runtime: &MetalDispatchRuntimeInfo,
) {
    route_metadata.crossover_decisions.push((
        "metal_dispatch_runtime_required_pipelines".to_string(),
        runtime.required_pipeline_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_runtime_max_thread_execution_width".to_string(),
        saturating_u64_to_u32(runtime.max_thread_execution_width),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_runtime_command_queue_ready".to_string(),
        u32::from(runtime.command_queue_ready),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_runtime_model_conditioned_inputs".to_string(),
        u32::from(runtime.model_conditioned_inputs),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_runtime_real_model_tensor_inputs".to_string(),
        u32::from(runtime.real_model_tensor_inputs),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_runtime_complete_model_forward_supported".to_string(),
        u32::from(runtime.complete_model_forward_supported),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_model_bindings_prepared".to_string(),
        u32::from(runtime.model_bindings_prepared),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_model_buffers_bound".to_string(),
        u32::from(runtime.model_buffers_bound),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_model_buffer_count".to_string(),
        runtime.model_buffer_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_model_buffer_bytes_lo".to_string(),
        saturating_u64_to_u32(runtime.model_buffer_bytes),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_projection_f32_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .projection_f32_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_projection_f16_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .projection_f16_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_projection_bf16_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .projection_bf16_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_projection_unsupported_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .projection_unsupported_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_projection_source_quantized_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .projection_source_quantized_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_rms_norm_f32_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .rms_norm_f32_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_rms_norm_f16_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .rms_norm_f16_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_rms_norm_bf16_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .rms_norm_bf16_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_rms_norm_unsupported_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .rms_norm_unsupported_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_native_rms_norm_source_quantized_binding_count".to_string(),
        runtime
            .native_dense_kernel_coverage
            .rms_norm_source_quantized_binding_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_binary_archive_state".to_string(),
        binary_archive_state_code(runtime.binary_archive.state),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_binary_archive_attached_pipelines".to_string(),
        runtime.binary_archive.attached_pipeline_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_binary_archive_serialized".to_string(),
        u32::from(runtime.binary_archive.serialized),
    ));
    if let Some(model) = &runtime.model {
        route_metadata
            .crossover_decisions
            .push(("metal_dispatch_model_artifacts_validated".to_string(), 1));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_multilayer_model_artifacts".to_string(),
            u32::from(model.layer_count > 1),
        ));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_model_layer_count".to_string(),
            model.layer_count,
        ));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_model_tensor_count".to_string(),
            model.tensor_count,
        ));
        route_metadata.crossover_decisions.push((
            "metal_dispatch_model_tied_word_embeddings".to_string(),
            u32::from(model.tie_word_embeddings),
        ));
    }
}

fn saturating_u64_to_u32(value: u64) -> u32 {
    value.min(u64::from(u32::MAX)) as u32
}

fn saturating_usize_to_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

fn binary_archive_state_code(state: MetalBinaryArchiveState) -> u32 {
    match state {
        MetalBinaryArchiveState::Disabled => 0,
        MetalBinaryArchiveState::Created => 1,
        MetalBinaryArchiveState::Loaded => 2,
        MetalBinaryArchiveState::Recreated => 3,
    }
}

fn fallback_failed_workload(input: &RunnerInput) -> MetalDispatchWorkload {
    MetalDispatchWorkload {
        scheduled_requests: input.execution_batch.items.len() as u32,
        prefill_requests: input
            .execution_batch
            .items
            .iter()
            .filter(|item| item.mode == ExecutionMode::Prefill)
            .count() as u32,
        decode_requests: input
            .execution_batch
            .items
            .iter()
            .filter(|item| item.mode == ExecutionMode::Decode)
            .count() as u32,
        scheduled_tokens: input.execution_batch.total_scheduled_tokens,
        resolved_blocks: input
            .block_tables
            .iter()
            .map(|resolved| resolved.block_table.block_ids.len() as u32)
            .sum::<u32>(),
        scheduled_token_ids: vec![0],
        scheduled_positions: vec![0],
        token_elements: input.execution_batch.total_scheduled_tokens.max(1),
        block_elements: input
            .block_tables
            .iter()
            .map(|resolved| resolved.block_table.block_ids.len() as u32)
            .sum::<u32>()
            .max(1),
        scratch_elements: input.execution_batch.total_scheduled_tokens.max(1).max(
            input
                .block_tables
                .iter()
                .map(|resolved| resolved.block_table.block_ids.len() as u32)
                .sum::<u32>()
                .max(1),
        ),
        kv_slot_capacity: 1,
        kv_block_capacity: 1,
        numeric_layout: MetalDispatchNumericLayout::default(),
        kv_metadata: MetalDispatchKvMetadata {
            block_size_tokens: input.block_size_tokens.max(1),
            slot_mapping: vec![0],
            attention_block_table: vec![0],
            gather_block_table: vec![0],
            gather_block_table_stride: 1,
            copy_block_mapping: vec![[0, 0]],
            seq_lens: vec![1],
            cu_seq_lens: vec![0, 1],
            scheduled_cu_seq_lens: vec![0, 1],
        },
    }
}

fn build_dispatch_kv_metadata(
    input: &RunnerInput,
) -> Result<MetalDispatchKvMetadata, MetalRuntimeError> {
    if input.block_size_tokens == 0 {
        return Err(MetalRuntimeError::InvalidDispatchInput {
            message: "block_size_tokens must be greater than zero".to_string(),
        });
    }

    let block_size_tokens = input.block_size_tokens;
    let block_table_by_request = input
        .block_tables
        .iter()
        .map(|resolved| (resolved.request_id, &resolved.block_table))
        .collect::<BTreeMap<_, _>>();

    let mut slot_mapping = Vec::new();
    let mut attention_block_table = Vec::new();
    let mut gather_block_rows = Vec::new();
    let mut copy_block_mapping = Vec::new();
    let mut seq_lens = Vec::with_capacity(input.execution_batch.items.len());
    let mut cu_seq_lens = vec![0_u32];
    let mut scheduled_cu_seq_lens = vec![0_u32];

    for item in &input.execution_batch.items {
        let scheduled_span = item
            .position_range
            .end_exclusive
            .checked_sub(item.position_range.start)
            .ok_or(MetalRuntimeError::InvalidDispatchInput {
                message: format!(
                    "request {} has an inverted position_range {:?}",
                    item.request_id.0, item.position_range
                ),
            })?;
        if scheduled_span != item.scheduled_token_count {
            return Err(MetalRuntimeError::InvalidDispatchInput {
                message: format!(
                    "request {} scheduled_token_count={} does not match position_range width={scheduled_span}",
                    item.request_id.0, item.scheduled_token_count
                ),
            });
        }

        let block_table = block_table_by_request.get(&item.block_table_ref).ok_or(
            MetalRuntimeError::InvalidDispatchInput {
                message: format!(
                    "runner input missing block table for request {}",
                    item.block_table_ref.0
                ),
            },
        )?;

        seq_lens.push(item.position_range.end_exclusive.max(1));
        let next_cu_seq_len = cu_seq_lens
            .last()
            .copied()
            .unwrap_or(0)
            .saturating_add(item.position_range.end_exclusive.max(1));
        cu_seq_lens.push(next_cu_seq_len);
        let next_scheduled_cu_seq_len = scheduled_cu_seq_lens
            .last()
            .copied()
            .unwrap_or(0)
            .saturating_add(item.scheduled_token_count);
        scheduled_cu_seq_lens.push(next_scheduled_cu_seq_len);

        let mut gather_row = Vec::with_capacity(block_table.block_ids.len().max(1));
        for &block_id in &block_table.block_ids {
            let block_base = block_id.0 * block_size_tokens;
            gather_row.push(block_base);
        }
        if gather_row.is_empty() {
            gather_row.push(0);
        }
        gather_block_rows.push(gather_row);

        for position in item.position_range.start..item.position_range.end_exclusive {
            let block_index = position / block_size_tokens;
            let block_offset = position % block_size_tokens;
            let Some(block_id) = block_table.block_ids.get(block_index as usize) else {
                return Err(MetalRuntimeError::InvalidDispatchInput {
                    message: format!(
                        "request {} position {} requires block index {} but block table only has {} entries",
                        item.request_id.0,
                        position,
                        block_index,
                        block_table.block_ids.len()
                    ),
                });
            };
            let slot = block_id.0 * block_size_tokens + block_offset;
            slot_mapping.push(slot);
            attention_block_table.push(slot);
        }
    }

    if slot_mapping.is_empty() {
        slot_mapping.push(0);
    }
    if attention_block_table.is_empty() {
        attention_block_table.push(0);
    }
    if copy_block_mapping.is_empty() {
        copy_block_mapping.push([0, 0]);
    }
    if seq_lens.is_empty() {
        seq_lens.push(1);
    }
    if cu_seq_lens.len() == 1 {
        cu_seq_lens.push(1);
    }
    if scheduled_cu_seq_lens.len() == 1 {
        scheduled_cu_seq_lens.push(1);
    }

    let gather_block_table_stride = gather_block_rows
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
        .max(1) as u32;
    let mut gather_block_table =
        Vec::with_capacity(gather_block_rows.len().max(1) * gather_block_table_stride as usize);
    if gather_block_rows.is_empty() {
        gather_block_table.push(0);
    } else {
        for row in gather_block_rows {
            gather_block_table.extend(row.iter().copied());
            gather_block_table.resize(
                gather_block_table.len()
                    + (gather_block_table_stride as usize).saturating_sub(row.len()),
                0,
            );
        }
    }

    Ok(MetalDispatchKvMetadata {
        block_size_tokens,
        slot_mapping,
        attention_block_table,
        gather_block_table,
        gather_block_table_stride,
        copy_block_mapping,
        seq_lens,
        cu_seq_lens,
        scheduled_cu_seq_lens,
    })
}

fn failed_runner_output_from_input(
    input: &RunnerInput,
    error_message: String,
    workload: &MetalDispatchWorkload,
    runtime: Option<&MetalDispatchRuntimeInfo>,
) -> RunnerOutput {
    let mut route_metadata = input.execution_batch.route_metadata.clone();
    annotate_failed_dispatch(&mut route_metadata, workload, runtime);

    RunnerOutput {
        step_id: input.execution_batch.step_id,
        request_updates: input
            .execution_batch
            .items
            .iter()
            .map(|item| RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: 0,
                output_token: None,
                stop_reason: Some(StopReason::Error),
                error: Some(error_message.clone()),
            })
            .collect(),
        logits_handles: Vec::new(),
        logits_outputs: Vec::new(),
        kv_write_summary: KvWriteSummary {
            tokens_written: 0,
            blocks_touched: 0,
        },
        route_metadata,
        execution_status: ExecutionStatus::Failed,
    }
}

#[cfg(test)]
fn derive_metal_decode_tokens_from_attention_bits(
    input: &RunnerInput,
    attention_output_bits: &[u32],
) -> Vec<(crate::ids::RequestId, u32)> {
    let Some(token_width) = attention_output_token_width(
        attention_output_bits.len(),
        input.execution_batch.total_scheduled_tokens,
    ) else {
        return Vec::new();
    };
    let mut attention_index = 0_usize;
    let mut direct_decode_tokens = Vec::new();

    for item in &input.execution_batch.items {
        let token_base = attention_index.saturating_mul(token_width);
        if item.mode == ExecutionMode::Decode {
            let token_end = token_base.saturating_add(token_width);
            if let Some(bits) = attention_output_bits.get(token_base..token_end) {
                direct_decode_tokens
                    .push((item.request_id, decode_token_from_attention_bits(bits)));
            }
        }
        attention_index = attention_index.saturating_add(item.scheduled_token_count as usize);
    }

    direct_decode_tokens
}

#[cfg(test)]
fn decode_token_from_attention_bits(bits: &[u32]) -> u32 {
    let values = bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 1;
    }

    let projected = values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32;
    projected.round().clamp(1.0, u32::MAX as f32) as u32
}

#[cfg(target_os = "macos")]
fn prefill_completion_request_ids(input: &RunnerInput) -> BTreeSet<crate::ids::RequestId> {
    input
        .execution_batch
        .items
        .iter()
        .filter(|item| item.mode == ExecutionMode::Prefill)
        .filter_map(|item| {
            let context = input.request_context(item.request_id)?;
            let completes_prompt = context
                .processed_prompt_tokens
                .checked_add(item.scheduled_token_count)
                .is_some_and(|next| next == context.prompt_len);
            (completes_prompt && context.generated_len == 0).then_some(item.request_id)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn sampleable_request_ids(input: &RunnerInput) -> BTreeSet<crate::ids::RequestId> {
    let mut request_ids = input
        .execution_batch
        .items
        .iter()
        .filter(|item| item.mode == ExecutionMode::Decode)
        .map(|item| item.request_id)
        .collect::<BTreeSet<_>>();
    request_ids.extend(prefill_completion_request_ids(input));
    request_ids
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn derive_model_bound_direct_decode_result(
    input: &RunnerInput,
    attention_output_bits: &[u32],
    artifacts: Option<&NativeModelArtifacts>,
    bindings: Option<&MetalNativeModelBindings>,
    buffers: Option<&MetalNativeModelBufferBindings>,
    final_layer_hidden_state_cache: Option<&ModelFinalLayerHiddenStateCache>,
    prefix_layer_caches: Option<&[Mutex<MetalPersistentLayerKvCache>]>,
    linear_request_states: Option<&Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>>,
    bringup: Option<&MetalRuntimeBringup>,
) -> ModelBoundDirectDecodeResult {
    let (Some(artifacts), Some(bindings), Some(buffers)) = (artifacts, bindings, buffers) else {
        return ModelBoundDirectDecodeResult::default();
    };
    let Ok(workload) = MetalDispatchWorkload::from_runner_input(input) else {
        return ModelBoundDirectDecodeResult::default();
    };
    let sampleable_request_ids = sampleable_request_ids(input);
    // Hybrid linear-attention models still need a fresh per-step forward pass,
    // but if the caller already staged that exact pass for the current step we
    // should reuse it here rather than recomputing and double-advancing the
    // recurrent state.
    let owned_hidden_states;
    let (hidden_states, final_layer_index) = if let Some(cache) = final_layer_hidden_state_cache {
        (cache.hidden_states.as_slice(), cache.final_layer_index)
    } else {
        let Some((hidden_states, final_layer_index, _)) = model_hidden_states_before_final_layer(
            artifacts,
            bindings,
            buffers,
            input,
            &workload,
            bringup,
            prefix_layer_caches,
            linear_request_states,
        ) else {
            return ModelBoundDirectDecodeResult::default();
        };
        owned_hidden_states = hidden_states;
        (owned_hidden_states.as_slice(), final_layer_index)
    };

    derive_model_bound_direct_decode_result_from_hidden_states(
        input,
        attention_output_bits,
        artifacts,
        bindings,
        buffers,
        &sampleable_request_ids,
        &workload,
        hidden_states,
        final_layer_index,
        bringup,
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn derive_model_bound_direct_decode_result_from_hidden_states(
    input: &RunnerInput,
    attention_output_bits: &[u32],
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    sampleable_request_ids: &BTreeSet<crate::ids::RequestId>,
    workload: &MetalDispatchWorkload,
    hidden_states: &[Vec<f32>],
    final_layer_index: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> ModelBoundDirectDecodeResult {
    if let Some(result) = derive_model_bound_direct_decode_result_from_hidden_states_batched(
        input,
        attention_output_bits,
        artifacts,
        bindings,
        buffers,
        sampleable_request_ids,
        workload,
        hidden_states,
        final_layer_index,
        bringup,
    ) {
        return result;
    }

    let Some(final_layer) = bindings.layers.get(final_layer_index) else {
        return ModelBoundDirectDecodeResult::default();
    };
    let Some(token_width) =
        attention_output_token_width(attention_output_bits.len(), workload.scheduled_tokens)
    else {
        return ModelBoundDirectDecodeResult::default();
    };

    let mut attention_index = 0_usize;
    let mut direct_decode_tokens = Vec::new();
    let mut logits_outputs = Vec::new();
    let mut model_bound_ffn_decode = false;
    let mut native_logits_projection_decode = false;
    let mut execution_tally = PrefixAttentionExecutionTally::default();
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default();

    for item in &input.execution_batch.items {
        let token_base = attention_index.saturating_mul(token_width);
        if sampleable_request_ids.contains(&item.request_id) {
            let decode_item = item.mode == ExecutionMode::Decode;
            let token_end = token_base.saturating_add(token_width);
            let hidden_index = attention_index
                .saturating_add(item.scheduled_token_count as usize)
                .saturating_sub(1);
            if let Some(bits) = attention_output_bits.get(token_base..token_end) {
                if request_uses_deterministic_argmax_sampling(input, item.request_id) {
                    if let Some((
                        token_id,
                        used_model_bound_ffn,
                        vocab_rows_scanned,
                        used_native_logits_projection,
                        decode_native_dense_tally,
                    )) = hidden_states.get(hidden_index).and_then(|hidden_state| {
                        decode_token_from_model_attention_output_with_metadata(
                            artifacts,
                            bindings,
                            buffers,
                            final_layer,
                            hidden_state,
                            bits,
                            bringup,
                        )
                    }) {
                        direct_decode_tokens.push((item.request_id, token_id));
                        execution_tally = execution_tally
                            .record_layer_continuation_tokens(1)
                            .record_logits_projection(1, vocab_rows_scanned);
                        if decode_item {
                            model_bound_ffn_decode |= used_model_bound_ffn;
                            native_logits_projection_decode |= used_native_logits_projection;
                            native_dense_tally =
                                native_dense_tally.merge(decode_native_dense_tally);
                        }
                    }
                } else if let Some((
                    logits,
                    token_id,
                    used_model_bound_ffn,
                    vocab_rows_scanned,
                    used_native_logits_projection,
                    decode_native_dense_tally,
                )) = hidden_states.get(hidden_index).and_then(|hidden_state| {
                    decode_logits_from_model_attention_output_with_metadata(
                        artifacts,
                        bindings,
                        buffers,
                        final_layer,
                        hidden_state,
                        bits,
                        bringup,
                    )
                }) {
                    direct_decode_tokens.push((item.request_id, token_id));
                    logits_outputs.push(RequestLogitsOutput {
                        request_id: item.request_id,
                        logits,
                    });
                    execution_tally = execution_tally
                        .record_layer_continuation_tokens(1)
                        .record_logits_projection(1, vocab_rows_scanned);
                    if decode_item {
                        model_bound_ffn_decode |= used_model_bound_ffn;
                        native_logits_projection_decode |= used_native_logits_projection;
                        native_dense_tally = native_dense_tally.merge(decode_native_dense_tally);
                    }
                }
            }
        }
        attention_index = attention_index.saturating_add(item.scheduled_token_count as usize);
    }

    ModelBoundDirectDecodeResult {
        tokens: direct_decode_tokens,
        logits_outputs,
        model_bound_ffn_decode,
        native_logits_projection_decode,
        execution_tally,
        native_dense_tally,
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct PreparedDirectDecodeItem {
    request_id: crate::ids::RequestId,
    mode: ExecutionMode,
    dims: ModelBoundDecodeDims,
    hidden: Vec<f32>,
    attention_input: Vec<f32>,
    deterministic_argmax_sampling: bool,
}

#[cfg(target_os = "macos")]
fn request_uses_deterministic_argmax_sampling(
    input: &RunnerInput,
    request_id: crate::ids::RequestId,
) -> bool {
    input
        .request_context(request_id)
        .is_some_and(|context| context.deterministic_argmax_sampling)
}

#[cfg(target_os = "macos")]
fn decode_prepared_item_count(prepared: &[PreparedDirectDecodeItem]) -> usize {
    prepared
        .iter()
        .filter(|item| item.mode == ExecutionMode::Decode)
        .count()
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn derive_model_bound_direct_decode_result_from_hidden_states_batched(
    input: &RunnerInput,
    attention_output_bits: &[u32],
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    sampleable_request_ids: &BTreeSet<crate::ids::RequestId>,
    workload: &MetalDispatchWorkload,
    hidden_states: &[Vec<f32>],
    final_layer_index: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<ModelBoundDirectDecodeResult> {
    let final_layer = bindings.layers.get(final_layer_index)?;
    let attention_o = buffers.binding_for(final_layer.attention_o.as_ref()?)?;
    let ffn_norm = buffers.binding_for(&final_layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&final_layer.ffn_down)?;
    let final_norm = buffers.binding_for(&bindings.final_norm)?;
    let decode_projection = resolved_decode_projection_binding(artifacts, bindings, buffers)?;
    let token_width =
        attention_output_token_width(attention_output_bits.len(), workload.scheduled_tokens)?;

    let mut attention_index = 0_usize;
    let mut prepared = Vec::new();

    for item in &input.execution_batch.items {
        let token_base = attention_index.checked_mul(token_width)?;
        if sampleable_request_ids.contains(&item.request_id) {
            let token_end = token_base.checked_add(token_width)?;
            let hidden_index = attention_index
                .checked_add(item.scheduled_token_count as usize)?
                .checked_sub(1)?;
            let bits = attention_output_bits.get(token_base..token_end)?;
            let attention_output = decode_attention_output_values(bits)?;
            let final_layer_hidden_state = hidden_states.get(hidden_index)?;
            let dims = resolved_model_decode_dims(
                artifacts,
                final_layer_hidden_state.len(),
                attention_o,
                ffn_norm,
                &final_layer.ffn_gate_up,
                buffers,
                ffn_down,
                final_norm,
                decode_projection,
                attention_output.len(),
            )?;
            let residual = final_layer_hidden_state.get(..dims.hidden_dim)?.to_vec();
            prepared.push(PreparedDirectDecodeItem {
                request_id: item.request_id,
                mode: item.mode,
                dims,
                hidden: residual,
                attention_input: attention_output.get(..dims.input_width)?.to_vec(),
                deterministic_argmax_sampling: request_uses_deterministic_argmax_sampling(
                    input,
                    item.request_id,
                ),
            });
        }
        attention_index = attention_index.checked_add(item.scheduled_token_count as usize)?;
    }

    if prepared.is_empty() {
        return Some(ModelBoundDirectDecodeResult::default());
    }
    if prepared.len() <= 1 {
        return None;
    }

    let prepared_request_order = prepared
        .iter()
        .map(|item| item.request_id)
        .collect::<Vec<_>>();
    let mut grouped_prepared =
        BTreeMap::<(ModelBoundDecodeDims, bool), Vec<PreparedDirectDecodeItem>>::new();
    for item in prepared {
        grouped_prepared
            .entry((item.dims, item.deterministic_argmax_sampling))
            .or_default()
            .push(item);
    }
    let mut partitioned_groups = Vec::new();
    for group in grouped_prepared.into_values() {
        let mut group_allowed = |candidate: &[PreparedDirectDecodeItem]| {
            if candidate.len() <= 1 {
                return true;
            }
            let Some(bringup) = bringup else {
                return true;
            };
            let Some(dims) = candidate.first().map(|item| item.dims) else {
                return false;
            };
            let feedback_key = direct_decode_batched_group_feedback_key(candidate.len(), dims);
            optional_kernel_allowed(bringup, &feedback_key)
        };
        partitioned_groups.extend(partition_prepared_direct_decode_group_by_batched_viability(
            group,
            &mut group_allowed,
        ));
    }
    let mut tokens_by_request = BTreeMap::new();
    let mut logits_by_request = BTreeMap::new();
    let mut native_logits_projection_decode = false;
    let mut model_bound_ffn_decode = false;
    let mut execution_tally = PrefixAttentionExecutionTally::default();
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default();

    let grouped_results = collect_model_bound_direct_decode_group_results_with_item_fallback(
        partitioned_groups,
        |group| {
            let expected_request_ids = group.iter().map(|item| item.request_id).collect::<Vec<_>>();
            let feedback_key = bringup.and_then(|bringup| {
                let dims = group.first()?.dims;
                direct_decode_group_feedback_key_for_group(group.len(), dims)
                    .map(|feedback_key| (bringup, feedback_key))
            });
            if let Some((bringup, feedback_key)) = feedback_key.as_ref() {
                if !optional_kernel_allowed(bringup, feedback_key) {
                    return None;
                }
            }
            let output = derive_model_bound_direct_decode_result_from_prepared_group(
                artifacts,
                final_layer,
                buffers,
                attention_o,
                final_norm,
                decode_projection,
                group,
                bringup,
            );
            let (output, success) =
                validate_model_bound_direct_decode_group_output(output, &expected_request_ids);
            if let Some((bringup, feedback_key)) = feedback_key.as_ref() {
                record_optional_kernel_result(bringup, feedback_key, success);
            }
            output
        },
    )?;

    for group_result in grouped_results {
        native_logits_projection_decode |= group_result.native_logits_projection_decode;
        model_bound_ffn_decode |= group_result.model_bound_ffn_decode;
        execution_tally = execution_tally.merge(group_result.execution_tally);
        native_dense_tally = native_dense_tally.merge(group_result.native_dense_tally);
        for (request_id, token_id) in group_result.tokens {
            tokens_by_request.insert(request_id, token_id);
        }
        for output in group_result.logits_outputs {
            logits_by_request.insert(output.request_id, output.logits);
        }
    }

    let mut tokens = Vec::with_capacity(prepared_request_order.len());
    let mut logits_outputs = Vec::with_capacity(prepared_request_order.len());
    for request_id in prepared_request_order {
        if let Some(token_id) = tokens_by_request.remove(&request_id) {
            tokens.push((request_id, token_id));
        }
        if let Some(logits) = logits_by_request.remove(&request_id) {
            logits_outputs.push(RequestLogitsOutput { request_id, logits });
        }
    }

    Some(ModelBoundDirectDecodeResult {
        tokens,
        logits_outputs,
        model_bound_ffn_decode,
        native_logits_projection_decode,
        execution_tally,
        native_dense_tally,
    })
}

#[cfg(target_os = "macos")]
fn partition_prepared_direct_decode_group_by_batched_viability(
    group: Vec<PreparedDirectDecodeItem>,
    group_allowed: &mut impl for<'a> FnMut(&'a [PreparedDirectDecodeItem]) -> bool,
) -> Vec<Vec<PreparedDirectDecodeItem>> {
    let mut stack = vec![group];
    let mut partitions = Vec::new();
    while let Some(mut current_group) = stack.pop() {
        if current_group.len() <= 1 || group_allowed(&current_group) {
            partitions.push(current_group);
            continue;
        }
        let split_index = current_group.len() / 2;
        let right_group = current_group.split_off(split_index);
        stack.push(right_group);
        stack.push(current_group);
    }
    partitions
}

#[cfg(target_os = "macos")]
fn collect_model_bound_direct_decode_group_results_with_item_fallback(
    groups: impl IntoIterator<Item = Vec<PreparedDirectDecodeItem>>,
    mut process_group: impl FnMut(Vec<PreparedDirectDecodeItem>) -> Option<ModelBoundDirectDecodeResult>,
) -> Option<Vec<ModelBoundDirectDecodeResult>> {
    let mut collected = Vec::new();
    for group in groups {
        collected.extend(collect_model_bound_direct_decode_group_results_recursive(
            group,
            &mut process_group,
        )?);
    }
    Some(collected)
}

#[cfg(target_os = "macos")]
fn collect_model_bound_direct_decode_group_results_recursive(
    group: Vec<PreparedDirectDecodeItem>,
    process_group: &mut impl FnMut(
        Vec<PreparedDirectDecodeItem>,
    ) -> Option<ModelBoundDirectDecodeResult>,
) -> Option<Vec<ModelBoundDirectDecodeResult>> {
    if let Some(group_result) = process_group(group.clone()) {
        return Some(vec![group_result]);
    }
    if group.len() <= 1 {
        return None;
    }

    let fallback_tally = DirectDecodeNativeDenseTally::default()
        .record_batched_group_fallback(decode_prepared_item_count(&group));
    let split_index = group.len() / 2;
    let mut left_group = group;
    let right_group = left_group.split_off(split_index);
    let mut collected =
        collect_model_bound_direct_decode_group_results_recursive(left_group, process_group)?;
    let mut right_results =
        collect_model_bound_direct_decode_group_results_recursive(right_group, process_group)?;
    if let Some(first_result) = collected.first_mut() {
        first_result.native_dense_tally = first_result.native_dense_tally.merge(fallback_tally);
    } else if let Some(first_right_result) = right_results.first_mut() {
        first_right_result.native_dense_tally =
            first_right_result.native_dense_tally.merge(fallback_tally);
    }
    collected.extend(right_results);
    Some(collected)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn derive_model_bound_direct_decode_result_from_prepared_group(
    artifacts: &NativeModelArtifacts,
    final_layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    attention_o: &MetalNativeTensorBufferBinding,
    final_norm: &MetalNativeTensorBufferBinding,
    decode_projection: &MetalNativeTensorBufferBinding,
    mut prepared: Vec<PreparedDirectDecodeItem>,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<ModelBoundDirectDecodeResult> {
    if prepared.is_empty() {
        return Some(ModelBoundDirectDecodeResult::default());
    }

    let dims = prepared.first()?.dims;
    let decode_item_count = decode_prepared_item_count(&prepared);
    let prepared_item_count = prepared.len();
    let direct_decode_group_tally = |tally: DirectDecodeNativeDenseTally| {
        tally.scale_for_selected_items(decode_item_count, prepared_item_count)
    };
    if prepared.iter().any(|item| {
        item.dims != dims
            || item.hidden.len() < dims.hidden_dim
            || item.attention_input.len() < dims.input_width
    }) {
        return None;
    }

    let mut attention_input_rows = prepared
        .iter()
        .map(|item| item.attention_input.clone())
        .collect::<Vec<_>>();
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default();
    if artifacts.manifest().attn_output_gate {
        let hidden_rows = prepared
            .iter()
            .map(|item| item.hidden.clone())
            .collect::<Vec<_>>();
        let gate =
            compute_attn_output_gate(artifacts, final_layer, buffers, &hidden_rows, bringup)?;
        native_dense_tally = native_dense_tally.merge(direct_decode_group_tally(
            direct_decode_native_dense_tally_from_prefix_attention_tally(
                apply_attention_output_gate_in_place_with_tally(
                    &mut attention_input_rows,
                    &gate,
                    dims.input_width,
                    bringup,
                )?,
            ),
        ));
    }
    let (attention_hidden_rows, attention_o_tally) = project_batched_matrix_rows_with_tally(
        attention_o,
        0,
        dims.hidden_dim,
        &attention_input_rows,
        dims.input_width,
        bringup,
    )?;
    native_dense_tally = native_dense_tally.merge(direct_decode_group_tally(attention_o_tally));
    let attention_hidden_rows_len = attention_hidden_rows.len();
    let mut hidden_rows = prepared
        .iter()
        .map(|item| item.hidden.clone())
        .collect::<Vec<_>>();
    if hidden_rows.len() != attention_hidden_rows_len {
        return None;
    }
    let attention_residual_add_tally =
        add_batched_rows_in_place_with_direct_decode_tally_and_result_dtype(
            &mut hidden_rows,
            &attention_hidden_rows,
            dims.hidden_dim,
            attention_o.native_dtype,
            bringup,
        )?;
    native_dense_tally =
        native_dense_tally.merge(direct_decode_group_tally(attention_residual_add_tally));
    for (item, hidden) in prepared.iter_mut().zip(hidden_rows) {
        item.hidden = hidden;
    }

    let hidden_rows = prepared
        .iter()
        .map(|item| item.hidden.clone())
        .collect::<Vec<_>>();
    let (continued_hidden_rows, continued_tally, continued_nontrivial) =
        apply_ffn_continuation_rows_with_tally(
            artifacts,
            final_layer,
            buffers,
            &hidden_rows,
            bringup,
        )?;
    native_dense_tally = native_dense_tally.merge(direct_decode_group_tally(continued_tally));
    let model_bound_ffn_decode = prepared
        .iter()
        .zip(continued_hidden_rows.iter())
        .any(|(item, _)| item.mode == ExecutionMode::Decode && continued_nontrivial);
    for (item, hidden) in prepared.iter_mut().zip(continued_hidden_rows) {
        item.hidden = hidden;
    }

    let mut final_hidden_rows = prepared
        .iter()
        .map(|item| item.hidden.clone())
        .collect::<Vec<_>>();
    native_dense_tally = native_dense_tally.merge(direct_decode_group_tally(
        direct_decode_native_dense_tally_from_prefix_attention_tally(
            apply_batched_row_rms_norm_with_binding_in_place_with_tally(
                &mut final_hidden_rows,
                dims.hidden_dim,
                final_norm,
                native_model_rms_norm_epsilon(artifacts),
                native_model_rms_norm_weight_offset(artifacts),
                bringup,
            )?,
        ),
    ));

    let token_count = final_hidden_rows.len();
    let deterministic_argmax_only = prepared
        .iter()
        .all(|item| item.deterministic_argmax_sampling);
    let mut batched_logits_results = if deterministic_argmax_only {
        None
    } else if token_count > 1 {
        project_batched_decode_logits_with_optional_native_path(
            bringup,
            decode_projection,
            &final_hidden_rows,
            dims.hidden_dim,
        )
        .map(|results| results.into_iter())
    } else {
        None
    };
    let mut batched_token_results = if deterministic_argmax_only && token_count > 1 {
        project_batched_decode_tokens_with_optional_native_path(
            bringup,
            decode_projection,
            &final_hidden_rows,
            dims.hidden_dim,
        )
        .map(|results| results.into_iter())
    } else {
        None
    };
    if batched_logits_results.is_some() || batched_token_results.is_some() {
        native_dense_tally = native_dense_tally.record_batched_logits_group(decode_item_count);
    }
    let single_logits_bringup =
        single_decode_logits_retry_worthwhile(bringup, decode_projection, dims.hidden_dim)
            .then_some(bringup)
            .flatten();
    let mut tokens = Vec::with_capacity(prepared.len());
    let mut logits_outputs = Vec::with_capacity(prepared.len());
    let mut native_logits_projection_decode = false;
    let mut execution_tally = PrefixAttentionExecutionTally::default();

    for (item, hidden) in prepared.into_iter().zip(final_hidden_rows) {
        if deterministic_argmax_only {
            let (token_id, used_native_logits_projection) =
                if let Some(batched_token_results) = batched_token_results.as_mut() {
                    let token_id = batched_token_results.next()?;
                    (token_id, true)
                } else if let Some(token_id) = project_decode_token_with_optional_native_path(
                    single_logits_bringup,
                    decode_projection,
                    &hidden,
                ) {
                    (token_id, true)
                } else {
                    (
                        project_decode_token_cpu(decode_projection, item.dims.hidden_dim, &hidden)?,
                        false,
                    )
                };
            if item.mode == ExecutionMode::Decode {
                native_dense_tally = native_dense_tally
                    .record_projection_rows(item.dims.vocab_rows, used_native_logits_projection);
                native_logits_projection_decode |= used_native_logits_projection;
            }
            execution_tally = execution_tally
                .record_layer_continuation_tokens(1)
                .record_logits_projection(
                    1,
                    u32::try_from(item.dims.vocab_rows).unwrap_or(u32::MAX),
                );
            tokens.push((item.request_id, token_id));
        } else {
            let (logits, token_id, used_native_logits_projection) =
                if let Some(batched_logits_results) = batched_logits_results.as_mut() {
                    let (logits, token_id) = batched_logits_results.next()?;
                    (logits, token_id, true)
                } else if let Some((logits, token_id)) =
                    project_decode_logits_with_optional_native_path(
                        single_logits_bringup,
                        decode_projection,
                        &hidden,
                    )
                {
                    (logits, token_id, true)
                } else {
                    let (logits, token_id) = project_decode_logits_cpu(
                        decode_projection,
                        item.dims.hidden_dim,
                        &hidden,
                    )?;
                    (logits, token_id, false)
                };
            if item.mode == ExecutionMode::Decode {
                native_dense_tally = native_dense_tally
                    .record_projection_rows(item.dims.vocab_rows, used_native_logits_projection);
                native_logits_projection_decode |= used_native_logits_projection;
            }
            execution_tally = execution_tally
                .record_layer_continuation_tokens(1)
                .record_logits_projection(
                    1,
                    u32::try_from(item.dims.vocab_rows).unwrap_or(u32::MAX),
                );
            tokens.push((item.request_id, token_id));
            logits_outputs.push(RequestLogitsOutput {
                request_id: item.request_id,
                logits,
            });
        }
    }

    Some(ModelBoundDirectDecodeResult {
        tokens,
        logits_outputs,
        model_bound_ffn_decode,
        native_logits_projection_decode,
        execution_tally,
        native_dense_tally,
    })
}

#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn derive_model_bound_decode_tokens(
    input: &RunnerInput,
    attention_output_bits: &[u32],
    artifacts: Option<&NativeModelArtifacts>,
    bindings: Option<&MetalNativeModelBindings>,
    buffers: Option<&MetalNativeModelBufferBindings>,
    final_layer_hidden_state_cache: Option<&ModelFinalLayerHiddenStateCache>,
    prefix_layer_caches: Option<&[Mutex<MetalPersistentLayerKvCache>]>,
    bringup: Option<&MetalRuntimeBringup>,
) -> Vec<(crate::ids::RequestId, u32)> {
    derive_model_bound_direct_decode_result(
        input,
        attention_output_bits,
        artifacts,
        bindings,
        buffers,
        final_layer_hidden_state_cache,
        prefix_layer_caches,
        None,
        bringup,
    )
    .tokens
    .into_iter()
    .filter(|(request_id, _)| {
        input
            .execution_batch
            .items
            .iter()
            .any(|item| item.request_id == *request_id && item.mode == ExecutionMode::Decode)
    })
    .collect()
}

fn attention_output_token_width(
    attention_output_len: usize,
    scheduled_tokens: u32,
) -> Option<usize> {
    let token_count = scheduled_tokens.max(1) as usize;
    let token_width = attention_output_len.checked_div(token_count)?;
    (token_width > 0 && token_width.checked_mul(token_count)? == attention_output_len)
        .then_some(token_width)
}

#[cfg(target_os = "macos")]
fn decode_logits_from_model_attention_output_with_metadata(
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    layer: &MetalNativeLayerBindings,
    final_layer_hidden_state: &[f32],
    attention_output_bits: &[u32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, u32, bool, u32, bool, DirectDecodeNativeDenseTally)> {
    let attention_o = buffers.binding_for(layer.attention_o.as_ref()?)?;
    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;
    let final_norm = buffers.binding_for(&bindings.final_norm)?;
    let decode_projection = resolved_decode_projection_binding(artifacts, bindings, buffers)?;
    let attention_output = decode_attention_output_values(attention_output_bits)?;
    let dims = resolved_model_decode_dims(
        artifacts,
        final_layer_hidden_state.len(),
        attention_o,
        ffn_norm,
        &layer.ffn_gate_up,
        buffers,
        ffn_down,
        final_norm,
        decode_projection,
        attention_output.len(),
    )?;
    let retry_policy = direct_decode_single_native_retry_policy(
        artifacts,
        attention_o,
        final_norm,
        decode_projection,
        dims,
        bringup,
    )?;
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default();
    let residual = final_layer_hidden_state.get(..dims.hidden_dim)?.to_vec();
    let mut attention_input = attention_output.get(..dims.input_width)?.to_vec();
    if artifacts.manifest().attn_output_gate {
        let gate = compute_attn_output_gate(
            artifacts,
            layer,
            buffers,
            std::slice::from_ref(&residual),
            bringup,
        )?;
        let mut attention_input_rows = vec![attention_input];
        native_dense_tally = native_dense_tally.merge(
            direct_decode_native_dense_tally_from_prefix_attention_tally(
                apply_attention_output_gate_in_place_with_tally(
                    &mut attention_input_rows,
                    &gate,
                    dims.input_width,
                    retry_policy.attention_output_gate,
                )?,
            ),
        );
        attention_input = attention_input_rows.pop()?;
    }
    let (mut hidden, used_native_attention_o_projection) = project_matrix_rows_with_path(
        attention_o,
        0,
        dims.hidden_dim,
        &attention_input,
        retry_policy.attention_o_projection,
    )?;
    native_dense_tally = native_dense_tally
        .record_projection_rows(dims.hidden_dim, used_native_attention_o_projection);
    let used_native_attention_residual_add = add_in_place_with_path_and_result_dtype(
        &mut hidden,
        &residual,
        attention_o.native_dtype,
        retry_policy.attention_residual_add,
    )?;
    native_dense_tally = native_dense_tally
        .record_residual_add_elements(hidden.len(), used_native_attention_residual_add);
    let (mut continued_hidden_rows, continued_tally, used_model_bound_ffn) =
        apply_ffn_continuation_rows_with_tally(
            artifacts,
            layer,
            buffers,
            &[hidden.clone()],
            bringup,
        )?;
    hidden = continued_hidden_rows.pop()?;
    native_dense_tally = native_dense_tally.merge(continued_tally);

    let used_native_final_norm = apply_rms_norm_with_binding_in_place_with_path(
        &mut hidden,
        final_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        retry_policy.final_norm,
    )?;
    native_dense_tally =
        native_dense_tally.record_rms_norm_elements(hidden.len(), used_native_final_norm);

    let (logits, token_id, used_native_logits_projection) = if let Some((logits, token_id)) =
        project_decode_logits_with_optional_native_path(
            retry_policy.logits_projection,
            decode_projection,
            &hidden,
        ) {
        (logits, token_id, true)
    } else {
        let (logits, token_id) =
            project_decode_logits_cpu(decode_projection, dims.hidden_dim, &hidden)?;
        (logits, token_id, false)
    };
    native_dense_tally =
        native_dense_tally.record_projection_rows(dims.vocab_rows, used_native_logits_projection);

    Some((
        logits,
        token_id,
        used_model_bound_ffn,
        u32::try_from(dims.vocab_rows).unwrap_or(u32::MAX),
        used_native_logits_projection,
        native_dense_tally,
    ))
}

#[cfg(target_os = "macos")]
fn decode_token_from_model_attention_output_with_metadata(
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    layer: &MetalNativeLayerBindings,
    final_layer_hidden_state: &[f32],
    attention_output_bits: &[u32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(u32, bool, u32, bool, DirectDecodeNativeDenseTally)> {
    let attention_o = buffers.binding_for(layer.attention_o.as_ref()?)?;
    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;
    let final_norm = buffers.binding_for(&bindings.final_norm)?;
    let decode_projection = resolved_decode_projection_binding(artifacts, bindings, buffers)?;
    let attention_output = decode_attention_output_values(attention_output_bits)?;
    let dims = resolved_model_decode_dims(
        artifacts,
        final_layer_hidden_state.len(),
        attention_o,
        ffn_norm,
        &layer.ffn_gate_up,
        buffers,
        ffn_down,
        final_norm,
        decode_projection,
        attention_output.len(),
    )?;
    let retry_policy = direct_decode_single_native_retry_policy(
        artifacts,
        attention_o,
        final_norm,
        decode_projection,
        dims,
        bringup,
    )?;
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default();
    let residual = final_layer_hidden_state.get(..dims.hidden_dim)?.to_vec();
    let mut attention_input = attention_output.get(..dims.input_width)?.to_vec();
    if artifacts.manifest().attn_output_gate {
        let gate = compute_attn_output_gate(
            artifacts,
            layer,
            buffers,
            std::slice::from_ref(&residual),
            bringup,
        )?;
        let mut attention_input_rows = vec![attention_input];
        native_dense_tally = native_dense_tally.merge(
            direct_decode_native_dense_tally_from_prefix_attention_tally(
                apply_attention_output_gate_in_place_with_tally(
                    &mut attention_input_rows,
                    &gate,
                    dims.input_width,
                    retry_policy.attention_output_gate,
                )?,
            ),
        );
        attention_input = attention_input_rows.pop()?;
    }
    let (mut hidden, used_native_attention_o_projection) = project_matrix_rows_with_path(
        attention_o,
        0,
        dims.hidden_dim,
        &attention_input,
        retry_policy.attention_o_projection,
    )?;
    native_dense_tally = native_dense_tally
        .record_projection_rows(dims.hidden_dim, used_native_attention_o_projection);
    let used_native_attention_residual_add = add_in_place_with_path_and_result_dtype(
        &mut hidden,
        &residual,
        attention_o.native_dtype,
        retry_policy.attention_residual_add,
    )?;
    native_dense_tally = native_dense_tally
        .record_residual_add_elements(hidden.len(), used_native_attention_residual_add);
    let (mut continued_hidden_rows, continued_tally, used_model_bound_ffn) =
        apply_ffn_continuation_rows_with_tally(
            artifacts,
            layer,
            buffers,
            &[hidden.clone()],
            bringup,
        )?;
    hidden = continued_hidden_rows.pop()?;
    native_dense_tally = native_dense_tally.merge(continued_tally);

    let used_native_final_norm = apply_rms_norm_with_binding_in_place_with_path(
        &mut hidden,
        final_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        retry_policy.final_norm,
    )?;
    native_dense_tally =
        native_dense_tally.record_rms_norm_elements(hidden.len(), used_native_final_norm);

    let (token_id, used_native_logits_projection) = if let Some(token_id) =
        project_decode_token_with_optional_native_path(
            retry_policy.logits_projection,
            decode_projection,
            &hidden,
        ) {
        (token_id, true)
    } else {
        (
            project_decode_token_cpu(decode_projection, dims.hidden_dim, &hidden)?,
            false,
        )
    };
    native_dense_tally =
        native_dense_tally.record_projection_rows(dims.vocab_rows, used_native_logits_projection);

    Some((
        token_id,
        used_model_bound_ffn,
        u32::try_from(dims.vocab_rows).unwrap_or(u32::MAX),
        used_native_logits_projection,
        native_dense_tally,
    ))
}

#[cfg(target_os = "macos")]
fn project_decode_logits_cpu(
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden_dim: usize,
    hidden: &[f32],
) -> Option<(Vec<f32>, u32)> {
    let (vocab_rows, _) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let mut logits = Vec::with_capacity(vocab_rows);
    let mut best_token = None;
    let mut best_score = f32::NEG_INFINITY;
    for token_row in 0..vocab_rows {
        let weights = tensor_matrix_row_prefix_f32(decode_projection, token_row, hidden_dim)?;
        let score = dot_product(&weights, hidden.get(..hidden_dim)?);
        if !score.is_finite() {
            return None;
        }
        logits.push(score);
        if score > best_score || best_token.is_none() {
            best_score = score;
            best_token = Some(token_row as u32);
        }
    }

    best_token.map(|token_id| (logits, token_id))
}

#[cfg(target_os = "macos")]
fn project_decode_token_cpu(
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden_dim: usize,
    hidden: &[f32],
) -> Option<u32> {
    let (vocab_rows, _) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let mut best_token = None;
    let mut best_score = f32::NEG_INFINITY;
    for token_row in 0..vocab_rows {
        let weights = tensor_matrix_row_prefix_f32(decode_projection, token_row, hidden_dim)?;
        let score = dot_product(&weights, hidden.get(..hidden_dim)?);
        if !score.is_finite() {
            return None;
        }
        if score > best_score || best_token.is_none() {
            best_score = score;
            best_token = Some(token_row as u32);
        }
    }

    best_token
}

#[cfg(target_os = "macos")]
fn project_decode_token_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden: &[f32],
) -> Option<u32> {
    let bringup = bringup?;
    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel(decode_projection.native_dtype)?;
    let argmax_kernel_name = "logits_argmax_f32";
    let argmax_pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .logits_argmax_f32?;
    let (vocab_rows, projection_cols) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let input_width = hidden.len().min(projection_cols);
    if vocab_rows == 0 || input_width == 0 {
        return None;
    }
    let projection_feedback_key = projection_feedback_key(
        projection_kernel_name,
        vocab_rows,
        input_width,
        projection_cols,
    );
    let argmax_feedback_key = logits_argmax_feedback_key(argmax_kernel_name, vocab_rows);
    if !optional_kernel_allowed(bringup, &projection_feedback_key)
        || !optional_kernel_allowed(bringup, &argmax_feedback_key)
    {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .zip(
        find_optional_pipeline_handle_by_index(
            &bringup.state,
            &bringup.metallib.path,
            argmax_kernel_name,
            argmax_pipeline_index,
        )
        .ok(),
    )
    .and_then(|(projection_pipeline, argmax_pipeline)| {
        autoreleasepool(|| {
            let hidden_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &hidden[..input_width]);
            let logits_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(vocab_rows),
            );
            let argmax_buffer = new_zeroed_shared_buffer::<u32>(&bringup.state.device, 1);

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.decode_logits_argmax");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.decode_logits_argmax.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&decode_projection.native_buffer), 0);
            encoder.set_buffer(2, Some(&logits_buffer), 0);
            set_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(vocab_rows),
                saturating_usize_to_u32(projection_cols),
                saturating_usize_to_u32(input_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(vocab_rows.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(vocab_rows.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.set_compute_pipeline_state(&argmax_pipeline.pipeline);
            encoder.set_buffer(0, Some(&logits_buffer), 0);
            encoder.set_buffer(1, Some(&argmax_buffer), 0);
            set_logits_argmax_dispatch_params(encoder, 2, saturating_usize_to_u32(vocab_rows));
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed
            {
                return None;
            }

            read_shared_u32_buffer_prefix(&argmax_buffer, 1)
                .into_iter()
                .next()
        })
    });
    let success = output.is_some();
    record_optional_kernel_result(bringup, &projection_feedback_key, success);
    record_optional_kernel_result(bringup, &argmax_feedback_key, success);
    output
}

#[cfg(target_os = "macos")]
fn project_decode_logits_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden: &[f32],
) -> Option<(Vec<f32>, u32)> {
    let bringup = bringup?;
    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel(decode_projection.native_dtype)?;
    let argmax_kernel_name = "logits_argmax_f32";
    let argmax_pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .logits_argmax_f32?;
    let (vocab_rows, projection_cols) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let input_width = hidden.len().min(projection_cols);
    if vocab_rows == 0 || input_width == 0 {
        return None;
    }
    let projection_feedback_key = projection_feedback_key(
        projection_kernel_name,
        vocab_rows,
        input_width,
        projection_cols,
    );
    let argmax_feedback_key = logits_argmax_feedback_key(argmax_kernel_name, vocab_rows);
    if !optional_kernel_allowed(bringup, &projection_feedback_key)
        || !optional_kernel_allowed(bringup, &argmax_feedback_key)
    {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .zip(
        find_optional_pipeline_handle_by_index(
            &bringup.state,
            &bringup.metallib.path,
            argmax_kernel_name,
            argmax_pipeline_index,
        )
        .ok(),
    )
    .and_then(|(projection_pipeline, argmax_pipeline)| {
        autoreleasepool(|| {
            let hidden_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &hidden[..input_width]);
            let logits_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(vocab_rows),
            );
            let argmax_buffer = new_zeroed_shared_buffer::<u32>(&bringup.state.device, 1);

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.decode_logits_projection");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.decode_logits_projection.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&decode_projection.native_buffer), 0);
            encoder.set_buffer(2, Some(&logits_buffer), 0);
            set_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(vocab_rows),
                saturating_usize_to_u32(projection_cols),
                saturating_usize_to_u32(input_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(vocab_rows.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(vocab_rows.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.set_compute_pipeline_state(&argmax_pipeline.pipeline);
            encoder.set_buffer(0, Some(&logits_buffer), 0);
            encoder.set_buffer(1, Some(&argmax_buffer), 0);
            set_logits_argmax_dispatch_params(encoder, 2, saturating_usize_to_u32(vocab_rows));
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let logits =
                read_shared_buffer_prefix(&logits_buffer, saturating_usize_to_u32(vocab_rows));
            if logits.len() != vocab_rows || logits.iter().any(|value| !value.is_finite()) {
                return None;
            }
            let best_token = read_shared_u32_buffer_prefix(&argmax_buffer, 1)
                .into_iter()
                .next()?;
            Some((logits, best_token))
        })
    });
    let success = output.is_some();
    record_optional_kernel_result(bringup, &projection_feedback_key, success);
    record_optional_kernel_result(bringup, &argmax_feedback_key, success);
    output
}

#[cfg(target_os = "macos")]
fn project_batched_decode_logits_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden_rows: &[Vec<f32>],
    hidden_width: usize,
) -> Option<Vec<(Vec<f32>, u32)>> {
    let bringup = bringup?;
    if hidden_rows.is_empty() || hidden_width == 0 {
        return None;
    }
    if hidden_rows.iter().any(|row| row.len() < hidden_width) {
        return None;
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel(decode_projection.native_dtype)?;
    let argmax_kernel_name = "logits_argmax_batched_f32";
    let argmax_pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .logits_argmax_batched_f32?;
    let (vocab_rows, projection_cols) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let input_width = hidden_width.min(projection_cols);
    if vocab_rows == 0 || input_width == 0 {
        return None;
    }

    let token_count = hidden_rows.len();
    let serialized_hidden_stride = input_width;
    let projection_feedback_key = batched_projection_feedback_key(
        projection_kernel_name,
        token_count,
        vocab_rows,
        input_width,
        serialized_hidden_stride,
        projection_cols,
    );
    let argmax_feedback_key =
        batched_logits_argmax_feedback_key(argmax_kernel_name, token_count, vocab_rows);
    if !optional_kernel_allowed(bringup, &projection_feedback_key)
        || !optional_kernel_allowed(bringup, &argmax_feedback_key)
    {
        return None;
    }

    let logits_element_count = token_count.checked_mul(vocab_rows)?;
    let mut flattened_hidden =
        Vec::with_capacity(token_count.checked_mul(serialized_hidden_stride)?);
    for row in hidden_rows {
        flattened_hidden.extend_from_slice(row.get(..serialized_hidden_stride)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .zip(
        find_optional_pipeline_handle_by_index(
            &bringup.state,
            &bringup.metallib.path,
            argmax_kernel_name,
            argmax_pipeline_index,
        )
        .ok(),
    )
    .and_then(|(projection_pipeline, argmax_pipeline)| {
        autoreleasepool(|| {
            let hidden_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &flattened_hidden);
            let logits_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(logits_element_count),
            );
            let argmax_buffer = new_zeroed_shared_buffer::<u32>(
                &bringup.state.device,
                saturating_usize_to_u32(token_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.decode_logits_projection_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.decode_logits_projection_batched.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&decode_projection.native_buffer), 0);
            encoder.set_buffer(2, Some(&logits_buffer), 0);
            set_batched_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(token_count),
                saturating_usize_to_u32(vocab_rows),
                saturating_usize_to_u32(projection_cols),
                saturating_usize_to_u32(input_width),
                saturating_usize_to_u32(serialized_hidden_stride),
            );
            encoder.dispatch_threads(
                MTLSize::new(logits_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(logits_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.set_compute_pipeline_state(&argmax_pipeline.pipeline);
            encoder.set_buffer(0, Some(&logits_buffer), 0);
            encoder.set_buffer(1, Some(&argmax_buffer), 0);
            set_batched_logits_argmax_dispatch_params(
                encoder,
                2,
                saturating_usize_to_u32(token_count),
                saturating_usize_to_u32(vocab_rows),
            );
            encoder.dispatch_threads(
                MTLSize::new(token_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    argmax_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(token_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let logits = read_shared_buffer_prefix(
                &logits_buffer,
                saturating_usize_to_u32(logits_element_count),
            );
            if logits.len() != logits_element_count || logits.iter().any(|value| !value.is_finite())
            {
                return None;
            }
            let best_tokens =
                read_shared_u32_buffer_prefix(&argmax_buffer, saturating_usize_to_u32(token_count));
            if best_tokens.len() != token_count {
                return None;
            }

            Some(
                logits
                    .chunks_exact(vocab_rows)
                    .zip(best_tokens)
                    .map(|(chunk, best_token)| (chunk.to_vec(), best_token))
                    .collect(),
            )
        })
    });
    let success = output.is_some();
    record_optional_kernel_result(bringup, &projection_feedback_key, success);
    record_optional_kernel_result(bringup, &argmax_feedback_key, success);
    output
}

#[cfg(target_os = "macos")]
fn project_batched_decode_tokens_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden_rows: &[Vec<f32>],
    hidden_width: usize,
) -> Option<Vec<u32>> {
    let bringup = bringup?;
    if hidden_rows.is_empty() || hidden_width == 0 {
        return None;
    }
    if hidden_rows.iter().any(|row| row.len() < hidden_width) {
        return None;
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel(decode_projection.native_dtype)?;
    let argmax_kernel_name = "logits_argmax_batched_f32";
    let argmax_pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .logits_argmax_batched_f32?;
    let (vocab_rows, projection_cols) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let input_width = hidden_width.min(projection_cols);
    if vocab_rows == 0 || input_width == 0 {
        return None;
    }

    let token_count = hidden_rows.len();
    let serialized_hidden_stride = input_width;
    let projection_feedback_key = batched_projection_feedback_key(
        projection_kernel_name,
        token_count,
        vocab_rows,
        input_width,
        serialized_hidden_stride,
        projection_cols,
    );
    let argmax_feedback_key =
        batched_logits_argmax_feedback_key(argmax_kernel_name, token_count, vocab_rows);
    if !optional_kernel_allowed(bringup, &projection_feedback_key)
        || !optional_kernel_allowed(bringup, &argmax_feedback_key)
    {
        return None;
    }

    let logits_element_count = token_count.checked_mul(vocab_rows)?;
    let mut flattened_hidden =
        Vec::with_capacity(token_count.checked_mul(serialized_hidden_stride)?);
    for row in hidden_rows {
        flattened_hidden.extend_from_slice(row.get(..serialized_hidden_stride)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .zip(
        find_optional_pipeline_handle_by_index(
            &bringup.state,
            &bringup.metallib.path,
            argmax_kernel_name,
            argmax_pipeline_index,
        )
        .ok(),
    )
    .and_then(|(projection_pipeline, argmax_pipeline)| {
        autoreleasepool(|| {
            let hidden_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &flattened_hidden);
            let logits_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(logits_element_count),
            );
            let argmax_buffer = new_zeroed_shared_buffer::<u32>(
                &bringup.state.device,
                saturating_usize_to_u32(token_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.decode_logits_argmax_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.decode_logits_argmax_batched.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&decode_projection.native_buffer), 0);
            encoder.set_buffer(2, Some(&logits_buffer), 0);
            set_batched_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(token_count),
                saturating_usize_to_u32(vocab_rows),
                saturating_usize_to_u32(projection_cols),
                saturating_usize_to_u32(input_width),
                saturating_usize_to_u32(serialized_hidden_stride),
            );
            encoder.dispatch_threads(
                MTLSize::new(logits_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(logits_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.set_compute_pipeline_state(&argmax_pipeline.pipeline);
            encoder.set_buffer(0, Some(&logits_buffer), 0);
            encoder.set_buffer(1, Some(&argmax_buffer), 0);
            set_batched_logits_argmax_dispatch_params(
                encoder,
                2,
                saturating_usize_to_u32(token_count),
                saturating_usize_to_u32(vocab_rows),
            );
            encoder.dispatch_threads(
                MTLSize::new(token_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    argmax_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(token_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed
            {
                return None;
            }

            let best_tokens =
                read_shared_u32_buffer_prefix(&argmax_buffer, saturating_usize_to_u32(token_count));
            (best_tokens.len() == token_count).then_some(best_tokens)
        })
    });
    let success = output.is_some();
    record_optional_kernel_result(bringup, &projection_feedback_key, success);
    record_optional_kernel_result(bringup, &argmax_feedback_key, success);
    output
}

#[cfg(target_os = "macos")]
fn has_nontrivial_ffn_contribution(values: &[f32]) -> bool {
    values.iter().any(|value| value.abs() > 1e-6)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ModelBoundDecodeDims {
    input_width: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    vocab_rows: usize,
}

#[cfg(target_os = "macos")]
fn direct_decode_batched_group_feedback_key(
    group_size: usize,
    dims: ModelBoundDecodeDims,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::DirectDecodeBatchedGroup { group_size, dims }
}

#[cfg(target_os = "macos")]
fn direct_decode_group_feedback_key_for_group(
    group_size: usize,
    dims: ModelBoundDecodeDims,
) -> Option<MetalOptionalKernelFeedbackKey> {
    (group_size > 1).then(|| direct_decode_batched_group_feedback_key(group_size, dims))
}

#[cfg(target_os = "macos")]
fn prefix_attention_group_feedback_key(
    workload: &MetalDispatchWorkload,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::PrefixAttentionBatchedGroup {
        scheduled_requests: workload.scheduled_requests,
        prefill_requests: workload.prefill_requests,
        decode_requests: workload.decode_requests,
        scheduled_tokens: workload.scheduled_tokens,
        gather_tokens: workload.kv_metadata.gather_token_count(),
        block_size_tokens: workload.kv_metadata.block_size_tokens,
        head_count: workload.numeric_layout.head_count,
        head_dim: workload.numeric_layout.head_dim,
    }
}

#[cfg(target_os = "macos")]
fn prefix_attention_group_feedback_key_for_item_count(
    item_count: usize,
    workload: &MetalDispatchWorkload,
) -> Option<MetalOptionalKernelFeedbackKey> {
    (item_count > 1).then(|| prefix_attention_group_feedback_key(workload))
}

#[cfg(target_os = "macos")]
fn resolved_decode_projection_binding<'a>(
    artifacts: &NativeModelArtifacts,
    bindings: &'a MetalNativeModelBindings,
    buffers: &'a MetalNativeModelBufferBindings,
) -> Option<&'a MetalNativeTensorBufferBinding> {
    if let Some(binding) = bindings.lm_head.as_ref() {
        return buffers.binding_for(binding);
    }

    if artifacts.manifest().tie_word_embeddings {
        return buffers.binding_for(&bindings.token_embedding);
    }

    None
}

#[cfg(target_os = "macos")]
fn decode_attention_output_values(bits: &[u32]) -> Option<Vec<f32>> {
    let values = bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>();
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(values)
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn resolved_model_decode_dims(
    artifacts: &NativeModelArtifacts,
    hidden_width: usize,
    attention_o: &MetalNativeTensorBufferBinding,
    ffn_norm: &MetalNativeTensorBufferBinding,
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    ffn_down: &MetalNativeTensorBufferBinding,
    final_norm: &MetalNativeTensorBufferBinding,
    decode_projection: &MetalNativeTensorBufferBinding,
    attention_output_width: usize,
) -> Option<ModelBoundDecodeDims> {
    let (attention_o_rows, attention_o_cols) = tensor_matrix_dimensions(&attention_o.meta.spec)?;
    let ffn_norm_len = tensor_element_count(&ffn_norm.meta.spec)?;
    let ffn_gate_up_input_cols = ffn_gate_up_input_cols(ffn_gate_up, buffers)?;
    let (ffn_down_rows, ffn_down_cols) = tensor_matrix_dimensions(&ffn_down.meta.spec)?;
    let final_norm_len = tensor_element_count(&final_norm.meta.spec)?;
    let (decode_rows, decode_cols) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let input_width = attention_output_width.min(attention_o_cols);
    let hidden_dim = (artifacts.manifest().hidden_size as usize)
        .min(hidden_width)
        .min(attention_o_rows)
        .min(ffn_norm_len)
        .min(ffn_gate_up_input_cols)
        .min(ffn_down_rows)
        .min(final_norm_len)
        .min(decode_cols);
    let intermediate_dim = resolved_ffn_intermediate_dim(ffn_gate_up, buffers, ffn_down_cols)?;
    let vocab_rows = decode_rows.min(artifacts.manifest().vocab_size as usize);

    (input_width > 0 && hidden_dim > 0 && intermediate_dim > 0 && vocab_rows > 0).then_some(
        ModelBoundDecodeDims {
            input_width,
            hidden_dim,
            intermediate_dim,
            vocab_rows,
        },
    )
}

#[cfg(target_os = "macos")]
fn ffn_gate_up_input_cols(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
) -> Option<usize> {
    match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            Some(tensor_matrix_dimensions(&packed.meta.spec)?.1)
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            Some(
                tensor_matrix_dimensions(&gate_binding.meta.spec)?
                    .1
                    .min(tensor_matrix_dimensions(&up_binding.meta.spec)?.1),
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn resolved_ffn_intermediate_dim(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    ffn_down_cols: usize,
) -> Option<usize> {
    let intermediate_dim = match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (rows, _) = tensor_matrix_dimensions(&packed.meta.spec)?;
            rows / 2
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            tensor_matrix_dimensions(&gate_binding.meta.spec)?
                .0
                .min(tensor_matrix_dimensions(&up_binding.meta.spec)?.0)
        }
    };

    let resolved = intermediate_dim.min(ffn_down_cols);
    (resolved > 0).then_some(resolved)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn project_ffn_gate_up(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let gate = project_matrix_rows(packed, 0, intermediate_dim, input, bringup)?;
            let up =
                project_matrix_rows(packed, intermediate_dim, intermediate_dim, input, bringup)?;
            Some((gate, up))
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            Some((
                project_matrix_rows(gate_binding, 0, intermediate_dim, input, bringup)?,
                project_matrix_rows(up_binding, 0, intermediate_dim, input, bringup)?,
            ))
        }
    }
}

#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
fn project_ffn_gate_up_with_coverage(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, Vec<f32>, DirectDecodeNativeDenseTally)> {
    let retry_policy = single_ffn_gate_up_projection_retry_policy(
        ffn_gate_up,
        buffers,
        intermediate_dim,
        input.len(),
        bringup,
    )?;
    project_ffn_gate_up_with_coverage_and_retry_policy(
        ffn_gate_up,
        buffers,
        intermediate_dim,
        input,
        retry_policy,
    )
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn project_ffn_gate_up_with_coverage_and_retry_policy(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input: &[f32],
    retry_policy: SingleFfnGateUpProjectionNativeRetryPolicy<'_>,
) -> Option<(Vec<f32>, Vec<f32>, DirectDecodeNativeDenseTally)> {
    match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (gate, used_native_gate) = project_matrix_rows_with_path(
                packed,
                0,
                intermediate_dim,
                input,
                retry_policy.gate_projection,
            )?;
            let (up, used_native_up) = project_matrix_rows_with_path(
                packed,
                intermediate_dim,
                intermediate_dim,
                input,
                retry_policy.up_projection,
            )?;
            let tally = DirectDecodeNativeDenseTally::default()
                .record_projection_rows(intermediate_dim, used_native_gate)
                .record_projection_rows(intermediate_dim, used_native_up);
            Some((gate, up, tally))
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            let (gate, used_native_gate) = project_matrix_rows_with_path(
                gate_binding,
                0,
                intermediate_dim,
                input,
                retry_policy.gate_projection,
            )?;
            let (up, used_native_up) = project_matrix_rows_with_path(
                up_binding,
                0,
                intermediate_dim,
                input,
                retry_policy.up_projection,
            )?;
            let tally = DirectDecodeNativeDenseTally::default()
                .record_projection_rows(intermediate_dim, used_native_gate)
                .record_projection_rows(intermediate_dim, used_native_up);
            Some((gate, up, tally))
        }
    }
}

#[cfg(target_os = "macos")]
type BatchedFfnGateUpProjection = (Vec<Vec<f32>>, Vec<Vec<f32>>, DirectDecodeNativeDenseTally);

#[cfg(target_os = "macos")]
fn project_batched_ffn_gate_up_with_tally(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<BatchedFfnGateUpProjection> {
    if intermediate_dim == 0 {
        return Some((
            vec![Vec::new(); input_rows.len()],
            vec![Vec::new(); input_rows.len()],
            DirectDecodeNativeDenseTally::default(),
        ));
    }
    if input_rows.is_empty() || input_rows.iter().any(|row| row.len() < input_width) {
        return None;
    }

    // Try multi-projection batch: gate + up in a single command buffer.
    #[cfg(target_os = "macos")]
    if let Some(result) = project_batched_ffn_gate_up_multi_dispatch(
        ffn_gate_up,
        buffers,
        intermediate_dim,
        input_rows,
        input_width,
        bringup,
    ) {
        return Some(result);
    }

    // Fallback: individual dispatches per projection.
    match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (gate_rows, gate_tally) = project_batched_matrix_rows_with_tally(
                packed,
                0,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (up_rows, up_tally) = project_batched_matrix_rows_with_tally(
                packed,
                intermediate_dim,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            Some((gate_rows, up_rows, gate_tally.merge(up_tally)))
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            let (gate_rows, gate_tally) = project_batched_matrix_rows_with_tally(
                gate_binding,
                0,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (up_rows, up_tally) = project_batched_matrix_rows_with_tally(
                up_binding,
                0,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            Some((gate_rows, up_rows, gate_tally.merge(up_tally)))
        }
    }
}

/// Attempts to dispatch FFN gate and up projections in a single Metal command buffer.
/// Returns `None` to signal the caller should fall back to individual dispatches.
#[cfg(target_os = "macos")]
fn project_batched_ffn_gate_up_multi_dispatch(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<BatchedFfnGateUpProjection> {
    let bringup = bringup?;

    let (gate_binding, gate_row_offset, up_binding, up_row_offset) = match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            (packed, 0_usize, packed, intermediate_dim)
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_b = buffers.binding_for(gate)?;
            let up_b = buffers.binding_for(up)?;
            (gate_b, 0_usize, up_b, 0_usize)
        }
    };

    let tasks = [
        MultiProjectionTask {
            binding: gate_binding,
            row_offset: gate_row_offset,
            output_dim: intermediate_dim,
        },
        MultiProjectionTask {
            binding: up_binding,
            row_offset: up_row_offset,
            output_dim: intermediate_dim,
        },
    ];

    let total_projection_rows = intermediate_dim.checked_mul(2)?;

    if input_rows.len() == 1 {
        let input = input_rows.first()?.get(..input_width)?;
        let outputs = project_multi_matrix_rows_with_optional_native_path(
            bringup,
            &tasks,
            input,
            input_width,
        )?;
        if outputs.len() != 2 {
            return None;
        }
        let mut iter = outputs.into_iter();
        let gate_rows = vec![iter.next()?];
        let up_rows = vec![iter.next()?];
        let tally = DirectDecodeNativeDenseTally::default()
            .record_projection_rows(total_projection_rows, true);
        Some((gate_rows, up_rows, tally))
    } else {
        let outputs = project_multi_batched_matrix_rows_with_optional_native_path(
            bringup,
            &tasks,
            input_rows,
            input_width,
        )?;
        if outputs.len() != 2 {
            return None;
        }
        let mut iter = outputs.into_iter();
        let gate_rows = iter.next()?;
        let up_rows = iter.next()?;
        let total_rows = input_rows.len().checked_mul(total_projection_rows)?;
        let tally =
            DirectDecodeNativeDenseTally::default().record_projection_rows(total_rows, true);
        Some((gate_rows, up_rows, tally))
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResolvedLayerMoeDims {
    hidden_dim: usize,
    expert_count: usize,
    experts_per_token: usize,
    expert_intermediate_dim: usize,
}

#[cfg(target_os = "macos")]
fn resolved_layer_moe_dims(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    hidden_width: usize,
) -> Option<ResolvedLayerMoeDims> {
    let moe = layer.moe.as_ref()?;
    let router = buffers.binding_for(&moe.router)?;
    let (router_expert_count, router_hidden_dim) = tensor_matrix_dimensions(&router.meta.spec)?;
    let expert_count = artifacts
        .moe_config()?
        .expert_count
        .map(|count| count as usize)
        .unwrap_or(router_expert_count)
        .min(router_expert_count);
    let experts_per_token = artifacts
        .moe_config()?
        .experts_per_token
        .map(|count| count as usize)
        .unwrap_or(1)
        .min(expert_count);
    if expert_count == 0 || experts_per_token == 0 {
        return None;
    }

    let (expert_gate_up_count, expert_rows, expert_input_dim) = match &moe.expert_gate_up {
        MetalMoeExpertGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (count, rows, cols) = tensor_3d_dimensions(&packed.meta.spec)?;
            (count, rows / 2, cols)
        }
        MetalMoeExpertGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            let (gate_count, gate_rows, gate_cols) = tensor_3d_dimensions(&gate_binding.meta.spec)?;
            let (up_count, up_rows, up_cols) = tensor_3d_dimensions(&up_binding.meta.spec)?;
            (
                gate_count.min(up_count),
                gate_rows.min(up_rows),
                gate_cols.min(up_cols),
            )
        }
    };
    let expert_down = buffers.binding_for(&moe.expert_down)?;
    let (expert_down_count, expert_down_hidden_dim, expert_down_cols) =
        tensor_3d_dimensions(&expert_down.meta.spec)?;
    let norm2_len = match &layer.ffn_norm_2 {
        Some(binding) => tensor_element_count(&buffers.binding_for(binding)?.meta.spec)?,
        None => tensor_element_count(&layer.ffn_norm.spec)?,
    };
    let hidden_dim = hidden_width
        .min(artifacts.manifest().hidden_size as usize)
        .min(router_hidden_dim)
        .min(expert_input_dim)
        .min(expert_down_hidden_dim)
        .min(norm2_len);
    let expert_intermediate_dim = artifacts
        .moe_config()?
        .expert_intermediate_size
        .map(|size| size as usize)
        .unwrap_or(expert_rows)
        .min(expert_rows)
        .min(expert_down_cols);
    let resolved_expert_count = expert_count
        .min(expert_gate_up_count)
        .min(expert_down_count);

    (hidden_dim > 0
        && expert_intermediate_dim > 0
        && resolved_expert_count > 0
        && experts_per_token > 0)
        .then_some(ResolvedLayerMoeDims {
            hidden_dim,
            expert_count: resolved_expert_count,
            experts_per_token: experts_per_token.min(resolved_expert_count),
            expert_intermediate_dim,
        })
}

#[cfg(target_os = "macos")]
fn apply_dense_ffn_branch_rows_with_tally(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    input_rows: &[Vec<f32>],
    hidden_dim: usize,
    bringup: Option<&MetalRuntimeBringup>,
    post_norm: Option<&MetalNativeTensorBufferBinding>,
) -> Option<(Vec<Vec<f32>>, DirectDecodeNativeDenseTally, bool)> {
    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;
    let intermediate_dim = resolved_ffn_intermediate_dim(
        &layer.ffn_gate_up,
        buffers,
        tensor_matrix_dimensions(&ffn_down.meta.spec)?.1,
    )?;
    let mut ffn_hidden_rows = input_rows
        .iter()
        .map(|row| row.get(..hidden_dim).map(|slice| slice.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    let mut tally = direct_decode_native_dense_tally_from_prefix_attention_tally(
        apply_batched_row_rms_norm_with_binding_in_place_with_tally(
            &mut ffn_hidden_rows,
            hidden_dim,
            ffn_norm,
            native_model_rms_norm_epsilon(artifacts),
            native_model_rms_norm_weight_offset(artifacts),
            bringup,
        )?,
    );
    let (mut gate_rows, up_rows, gate_up_tally) = project_batched_ffn_gate_up_with_tally(
        &layer.ffn_gate_up,
        buffers,
        intermediate_dim,
        &ffn_hidden_rows,
        hidden_dim,
        bringup,
    )?;
    tally = tally.merge(gate_up_tally);
    tally = tally.merge(apply_batched_model_gate_up_product_in_place_with_tally(
        artifacts,
        &mut gate_rows,
        &up_rows,
        intermediate_dim,
        bringup,
    )?);
    let (mut ffn_output_rows, ffn_down_tally) = project_batched_matrix_rows_with_tally(
        ffn_down,
        0,
        hidden_dim,
        &gate_rows,
        intermediate_dim,
        bringup,
    )?;
    tally = tally.merge(ffn_down_tally);
    if let Some(post_norm) = post_norm {
        tally = tally.merge(
            direct_decode_native_dense_tally_from_prefix_attention_tally(
                apply_batched_row_rms_norm_with_binding_in_place_with_tally(
                    &mut ffn_output_rows,
                    hidden_dim,
                    post_norm,
                    native_model_rms_norm_epsilon(artifacts),
                    native_model_rms_norm_weight_offset(artifacts),
                    bringup,
                )?,
            ),
        );
    }
    let nontrivial = ffn_output_rows
        .iter()
        .any(|row| has_nontrivial_ffn_contribution(row));
    Some((ffn_output_rows, tally, nontrivial))
}

#[cfg(target_os = "macos")]
fn prepare_moe_router_inputs_with_tally(
    artifacts: &NativeModelArtifacts,
    input_rows: &[Vec<f32>],
    hidden_dim: usize,
    router_scale: Option<&MetalNativeTensorBufferBinding>,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, DirectDecodeNativeDenseTally)> {
    let mut router_inputs = input_rows
        .iter()
        .map(|row| row.get(..hidden_dim).map(|slice| slice.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    let tally = direct_decode_native_dense_tally_from_prefix_attention_tally(
        apply_batched_row_rms_norm_without_weights_in_place_with_tally(
            &mut router_inputs,
            hidden_dim,
            native_model_rms_norm_epsilon(artifacts),
            bringup,
        )?,
    );
    let root_size = (hidden_dim as f32).sqrt().recip();
    if !root_size.is_finite() {
        return None;
    }
    let column_scales = match router_scale {
        Some(binding) => tensor_prefix_f32(binding, hidden_dim)?
            .into_iter()
            .map(|value| value * root_size)
            .collect::<Vec<_>>(),
        None => vec![root_size; hidden_dim],
    };
    let used_native_scale = apply_batched_row_vector_scale_in_place_with_path(
        &mut router_inputs,
        hidden_dim,
        &column_scales,
        bringup,
    )?;
    let scale_element_count = router_inputs.len().checked_mul(hidden_dim)?;
    Some((
        router_inputs,
        tally.record_scale_elements(scale_element_count, used_native_scale),
    ))
}

#[cfg(target_os = "macos")]
fn select_top_k_moe_experts(logits: &[f32], selection_count: usize) -> Option<Vec<(usize, f32)>> {
    if logits.is_empty() || selection_count == 0 {
        return None;
    }
    let selection_count = selection_count.min(logits.len());
    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    if ranked.iter().any(|(_, value)| !value.is_finite()) {
        return None;
    }
    ranked.sort_unstable_by(|(_, left), (_, right)| left.total_cmp(right));
    let top = ranked.split_off(ranked.len().saturating_sub(selection_count));
    let max_logit = top
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return None;
    }
    let normalizer = top
        .iter()
        .map(|(_, value)| (*value - max_logit).exp())
        .sum::<f32>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return None;
    }
    Some(
        top.into_iter()
            .map(|(expert_index, value)| (expert_index, (value - max_logit).exp() / normalizer))
            .collect(),
    )
}

#[cfg(target_os = "macos")]
fn project_moe_expert_matrix_rows_cpu(
    binding: &MetalNativeTensorBufferBinding,
    expert_index: usize,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    let (expert_count, rows, cols) = tensor_3d_dimensions(&binding.meta.spec)?;
    if expert_index >= expert_count
        || row_offset.checked_add(output_dim)? > rows
        || input.len() > cols
    {
        return None;
    }

    let mut output = Vec::with_capacity(output_dim);
    for row in row_offset..row_offset + output_dim {
        let weights = tensor_3d_matrix_row_prefix_f32(binding, expert_index, row, input.len())?;
        output.push(dot_product(&weights, input));
    }
    round_slice_to_native_dtype(&mut output, binding.native_dtype);
    Some(output)
}

#[cfg(target_os = "macos")]
fn project_moe_expert_matrix_rows_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
    expert_index: usize,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    let (expert_count, rows_per_expert, cols) = tensor_3d_dimensions(&binding.meta.spec)?;
    if expert_index >= expert_count
        || row_offset.checked_add(output_dim)? > rows_per_expert
        || input.len() > cols
    {
        return None;
    }
    if output_dim == 0 {
        return Some(Vec::new());
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel(binding.native_dtype)?;
    let feedback_key =
        projection_feedback_key(projection_kernel_name, output_dim, input.len(), cols);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let row_byte_offset = expert_index
        .checked_mul(rows_per_expert)?
        .checked_add(row_offset)?
        .checked_mul(cols)?
        .checked_mul(native_dtype_size_bytes(binding.native_dtype))?;

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .and_then(|projection_pipeline| {
        autoreleasepool(|| {
            let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, input);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_dim),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.project_moe_expert_rows");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.project_moe_expert_rows.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input.len()),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_dim.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_dim.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(output_dim));
            (output.len() == output_dim && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn project_moe_expert_matrix_rows_with_path(
    binding: &MetalNativeTensorBufferBinding,
    expert_index: usize,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, bool)> {
    let (expert_count, rows_per_expert, cols) = tensor_3d_dimensions(&binding.meta.spec)?;
    if expert_index >= expert_count
        || row_offset.checked_add(output_dim)? > rows_per_expert
        || input.len() > cols
    {
        return None;
    }

    if let Some(output) = project_moe_expert_matrix_rows_with_optional_native_path(
        bringup,
        binding,
        expert_index,
        row_offset,
        output_dim,
        input,
    ) {
        return Some((output, true));
    }

    let output =
        project_moe_expert_matrix_rows_cpu(binding, expert_index, row_offset, output_dim, input)?;
    Some((output, false))
}

#[cfg(target_os = "macos")]
fn project_batched_moe_expert_matrix_rows_with_tally(
    binding: &MetalNativeTensorBufferBinding,
    expert_index: usize,
    row_offset: usize,
    output_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, DirectDecodeNativeDenseTally)> {
    if input_rows.is_empty() {
        return Some((Vec::new(), DirectDecodeNativeDenseTally::default()));
    }
    let (expert_count, rows_per_expert, cols) = tensor_3d_dimensions(&binding.meta.spec)?;
    if expert_index >= expert_count
        || row_offset.checked_add(output_dim)? > rows_per_expert
        || input_width > cols
        || input_rows.iter().any(|row| row.len() < input_width)
    {
        return None;
    }

    let single_native_bringup = bringup.filter(|bringup| {
        bringup
            .state
            .optional_kernel_dispatch_plan
            .projection_kernel(binding.native_dtype)
            .is_some()
    });
    if input_rows.len() == 1 && single_native_bringup.is_some() {
        let (projected, used_native) = project_moe_expert_matrix_rows_with_path(
            binding,
            expert_index,
            row_offset,
            output_dim,
            input_rows.first()?.get(..input_width)?,
            single_native_bringup,
        )?;
        return Some((
            vec![projected],
            DirectDecodeNativeDenseTally::default().record_projection_rows(output_dim, used_native),
        ));
    }

    if let Some(bringup) = bringup {
        let (projection_kernel_name, projection_pipeline_index) = bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_projection_kernel(binding.native_dtype)?;
        let row_count = input_rows.len();
        let feedback_key = batched_projection_feedback_key(
            projection_kernel_name,
            row_count,
            output_dim,
            input_width,
            input_width,
            cols,
        );
        if optional_kernel_allowed(bringup, &feedback_key) {
            let flattened_row_offset = expert_index
                .checked_mul(rows_per_expert)?
                .checked_add(row_offset)?;
            let row_byte_offset = flattened_row_offset
                .checked_mul(cols)?
                .checked_mul(native_dtype_size_bytes(binding.native_dtype))?;
            let output_element_count = row_count.checked_mul(output_dim)?;
            let mut flattened_input = Vec::with_capacity(row_count.checked_mul(input_width)?);
            for row in input_rows {
                flattened_input.extend_from_slice(row.get(..input_width)?);
            }

            let output = find_optional_pipeline_handle_by_index(
                &bringup.state,
                &bringup.metallib.path,
                projection_kernel_name,
                projection_pipeline_index,
            )
            .ok()
            .and_then(|projection_pipeline| {
                autoreleasepool(|| {
                    let hidden_buffer =
                        new_shared_buffer_with_data(&bringup.state.device, &flattened_input);
                    let output_buffer = new_zeroed_shared_buffer::<f32>(
                        &bringup.state.device,
                        saturating_usize_to_u32(output_element_count),
                    );

                    let command_buffer = bringup.state.command_queue.new_command_buffer();
                    command_buffer.set_label("ax.phase1.project_moe_expert_rows_batched");
                    let encoder = command_buffer.new_compute_command_encoder();
                    encoder.set_label("ax.phase1.project_moe_expert_rows_batched.compute");

                    encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
                    encoder.set_buffer(0, Some(&hidden_buffer), 0);
                    encoder.set_buffer(1, Some(&binding.native_buffer), row_byte_offset as u64);
                    encoder.set_buffer(2, Some(&output_buffer), 0);
                    set_batched_logits_projection_dispatch_params(
                        encoder,
                        3,
                        saturating_usize_to_u32(row_count),
                        saturating_usize_to_u32(output_dim),
                        saturating_usize_to_u32(cols),
                        saturating_usize_to_u32(input_width),
                        saturating_usize_to_u32(input_width),
                    );
                    encoder.dispatch_threads(
                        MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                        MTLSize::new(
                            projection_pipeline
                                .pipeline
                                .thread_execution_width()
                                .max(1)
                                .min(output_element_count.max(1) as u64),
                            1,
                            1,
                        ),
                    );

                    encoder.end_encoding();
                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    let command_buffer_status = command_buffer_status(command_buffer.status());
                    if command_buffer_status != MetalCommandBufferStatus::Completed {
                        return None;
                    }

                    let output = read_shared_buffer_prefix(
                        &output_buffer,
                        saturating_usize_to_u32(output_element_count),
                    );
                    if output.len() != output_element_count
                        || output.iter().any(|value| !value.is_finite())
                    {
                        return None;
                    }
                    Some(
                        output
                            .chunks_exact(output_dim)
                            .map(|chunk| chunk.to_vec())
                            .collect::<Vec<_>>(),
                    )
                })
            });
            record_optional_kernel_result(bringup, &feedback_key, output.is_some());
            if let Some(output_rows) = output {
                return Some((
                    output_rows,
                    DirectDecodeNativeDenseTally::default()
                        .record_projection_rows(input_rows.len().checked_mul(output_dim)?, true),
                ));
            }
        }
    }

    if input_rows.len() > 1 && batched_projection_split_retry_worthwhile(bringup, binding) {
        let split_index = input_rows.len() / 2;
        let (left_rows, right_rows) = input_rows.split_at(split_index);
        let (mut left_projected_rows, left_tally) =
            project_batched_moe_expert_matrix_rows_with_tally(
                binding,
                expert_index,
                row_offset,
                output_dim,
                left_rows,
                input_width,
                bringup,
            )?;
        let (right_projected_rows, right_tally) =
            project_batched_moe_expert_matrix_rows_with_tally(
                binding,
                expert_index,
                row_offset,
                output_dim,
                right_rows,
                input_width,
                bringup,
            )?;
        left_projected_rows.extend(right_projected_rows);
        return Some((left_projected_rows, left_tally.merge(right_tally)));
    }

    let mut projected_rows = Vec::with_capacity(input_rows.len());
    let mut tally = DirectDecodeNativeDenseTally::default();
    for row in input_rows {
        let (projected, used_native) = project_moe_expert_matrix_rows_with_path(
            binding,
            expert_index,
            row_offset,
            output_dim,
            row.get(..input_width)?,
            single_native_bringup,
        )?;
        tally = tally.record_projection_rows(output_dim, used_native);
        projected_rows.push(projected);
    }
    Some((projected_rows, tally))
}

#[cfg(target_os = "macos")]
struct MoeExpertMultiProjectionTask<'a> {
    binding: &'a MetalNativeTensorBufferBinding,
    expert_index: usize,
    row_offset: usize,
    output_dim: usize,
}

#[cfg(target_os = "macos")]
fn project_batched_moe_expert_gate_up_multi_dispatch(
    expert_gate_up: &MetalMoeExpertGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    expert_index: usize,
    intermediate_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<BatchedFfnGateUpProjection> {
    let bringup = bringup?;
    let (gate_binding, gate_row_offset, up_binding, up_row_offset) = match expert_gate_up {
        MetalMoeExpertGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            (packed, 0_usize, packed, intermediate_dim)
        }
        MetalMoeExpertGateUpBindings::Split { gate, up } => {
            let gate_b = buffers.binding_for(gate)?;
            let up_b = buffers.binding_for(up)?;
            (gate_b, 0_usize, up_b, 0_usize)
        }
    };
    let tasks = [
        MoeExpertMultiProjectionTask {
            binding: gate_binding,
            expert_index,
            row_offset: gate_row_offset,
            output_dim: intermediate_dim,
        },
        MoeExpertMultiProjectionTask {
            binding: up_binding,
            expert_index,
            row_offset: up_row_offset,
            output_dim: intermediate_dim,
        },
    ];

    let row_count = input_rows.len();
    let first_dtype = tasks[0].binding.native_dtype;
    if row_count == 0 || input_rows.iter().any(|row| row.len() < input_width) {
        return None;
    }
    if tasks
        .iter()
        .any(|task| task.binding.native_dtype != first_dtype)
    {
        return None;
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel(first_dtype)?;
    for task in &tasks {
        let (expert_count, rows_per_expert, cols) = tensor_3d_dimensions(&task.binding.meta.spec)?;
        if task.expert_index >= expert_count
            || task.row_offset.checked_add(task.output_dim)? > rows_per_expert
            || input_width > cols
        {
            return None;
        }
        let feedback_key = batched_projection_feedback_key(
            projection_kernel_name,
            row_count,
            task.output_dim,
            input_width,
            input_width,
            cols,
        );
        if !optional_kernel_allowed(bringup, &feedback_key) {
            return None;
        }
    }

    let projection_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()?;
    let mut flattened_input = Vec::with_capacity(row_count.checked_mul(input_width)?);
    for row in input_rows {
        flattened_input.extend_from_slice(row.get(..input_width)?);
    }

    autoreleasepool(|| {
        let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_input);
        let output_buffers: Vec<_> = tasks
            .iter()
            .map(|task| {
                let output_element_count = row_count.checked_mul(task.output_dim).unwrap_or(0);
                new_zeroed_shared_buffer::<f32>(
                    &bringup.state.device,
                    saturating_usize_to_u32(output_element_count),
                )
            })
            .collect();

        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.project_moe_expert_gate_up_batched.multi");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("ax.phase1.project_moe_expert_gate_up_batched.multi.compute");

        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let (_, rows_per_expert, cols) = tensor_3d_dimensions(&task.binding.meta.spec)?;
            let flattened_row_offset = task
                .expert_index
                .checked_mul(rows_per_expert)?
                .checked_add(task.row_offset)?;
            let row_byte_offset = flattened_row_offset
                .checked_mul(cols)?
                .checked_mul(native_dtype_size_bytes(task.binding.native_dtype))?;
            let output_element_count = row_count.checked_mul(task.output_dim)?;

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&task.binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(output_buffer), 0);
            set_batched_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(row_count),
                saturating_usize_to_u32(task.output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input_width),
                saturating_usize_to_u32(input_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer_status(command_buffer.status());
        if status != MetalCommandBufferStatus::Completed {
            return None;
        }

        let mut outputs = Vec::with_capacity(tasks.len());
        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let output_element_count = row_count.checked_mul(task.output_dim)?;
            let output = read_shared_buffer_prefix(
                output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if output.len() != output_element_count || output.iter().any(|value| !value.is_finite())
            {
                return None;
            }
            outputs.push(
                output
                    .chunks_exact(task.output_dim)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            );
        }

        let total_rows = row_count.checked_mul(intermediate_dim.checked_mul(2)?)?;
        let mut iter = outputs.into_iter();
        Some((
            iter.next()?,
            iter.next()?,
            DirectDecodeNativeDenseTally::default().record_projection_rows(total_rows, true),
        ))
    })
}

#[cfg(target_os = "macos")]
fn project_batched_moe_expert_gate_up_with_tally(
    expert_gate_up: &MetalMoeExpertGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    expert_index: usize,
    intermediate_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<BatchedFfnGateUpProjection> {
    if intermediate_dim == 0 {
        return Some((
            vec![Vec::new(); input_rows.len()],
            vec![Vec::new(); input_rows.len()],
            DirectDecodeNativeDenseTally::default(),
        ));
    }
    if input_rows.is_empty() || input_rows.iter().any(|row| row.len() < input_width) {
        return None;
    }

    if let Some(result) = project_batched_moe_expert_gate_up_multi_dispatch(
        expert_gate_up,
        buffers,
        expert_index,
        intermediate_dim,
        input_rows,
        input_width,
        bringup,
    ) {
        return Some(result);
    }

    match expert_gate_up {
        MetalMoeExpertGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (gate_rows, gate_tally) = project_batched_moe_expert_matrix_rows_with_tally(
                packed,
                expert_index,
                0,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (up_rows, up_tally) = project_batched_moe_expert_matrix_rows_with_tally(
                packed,
                expert_index,
                intermediate_dim,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            Some((gate_rows, up_rows, gate_tally.merge(up_tally)))
        }
        MetalMoeExpertGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            let (gate_rows, gate_tally) = project_batched_moe_expert_matrix_rows_with_tally(
                gate_binding,
                expert_index,
                0,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (up_rows, up_tally) = project_batched_moe_expert_matrix_rows_with_tally(
                up_binding,
                expert_index,
                0,
                intermediate_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            Some((gate_rows, up_rows, gate_tally.merge(up_tally)))
        }
    }
}

#[cfg(target_os = "macos")]
fn apply_moe_ffn_branch_rows_with_tally(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    input_rows: &[Vec<f32>],
    hidden_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, DirectDecodeNativeDenseTally, bool)> {
    let moe = layer.moe.as_ref()?;
    let dims = resolved_layer_moe_dims(artifacts, layer, buffers, hidden_width)?;
    let router = buffers.binding_for(&moe.router)?;
    let router_scale = match &moe.router_scale {
        Some(binding) => Some(buffers.binding_for(binding)?),
        None => None,
    };
    let expert_down = buffers.binding_for(&moe.expert_down)?;
    let expert_down_scale = match &moe.expert_down_scale {
        Some(binding) => Some(buffers.binding_for(binding)?),
        None => None,
    };
    let expert_norm = match &layer.ffn_norm_2 {
        Some(binding) => buffers.binding_for(binding)?,
        None => buffers.binding_for(&layer.ffn_norm)?,
    };
    let expert_post_norm = match &layer.ffn_post_norm_2 {
        Some(binding) => Some(buffers.binding_for(binding)?),
        None => None,
    };

    let (router_inputs, mut tally) = prepare_moe_router_inputs_with_tally(
        artifacts,
        input_rows,
        dims.hidden_dim,
        router_scale,
        bringup,
    )?;
    let (router_logits_rows, router_tally) = project_batched_matrix_rows_with_tally(
        router,
        0,
        dims.expert_count,
        &router_inputs,
        dims.hidden_dim,
        bringup,
    )?;
    tally = tally.merge(router_tally);

    let mut expert_inputs = input_rows
        .iter()
        .map(|row| row.get(..dims.hidden_dim).map(|slice| slice.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    tally = tally.merge(
        direct_decode_native_dense_tally_from_prefix_attention_tally(
            apply_batched_row_rms_norm_with_binding_in_place_with_tally(
                &mut expert_inputs,
                dims.hidden_dim,
                expert_norm,
                native_model_rms_norm_epsilon(artifacts),
                native_model_rms_norm_weight_offset(artifacts),
                bringup,
            )?,
        ),
    );

    let mut output_rows = vec![vec![0.0_f32; dims.hidden_dim]; input_rows.len()];
    let mut nontrivial = false;

    let mut assignments_by_expert = BTreeMap::<usize, Vec<(usize, f32)>>::new();
    for (token_index, logits) in router_logits_rows.iter().enumerate() {
        let selections = select_top_k_moe_experts(logits, dims.experts_per_token)?;
        for (expert_index, expert_weight) in selections {
            assignments_by_expert
                .entry(expert_index)
                .or_default()
                .push((token_index, expert_weight));
        }
    }

    for (expert_index, assignments) in assignments_by_expert {
        let grouped_inputs = assignments
            .iter()
            .map(|(token_index, _)| {
                expert_inputs
                    .get(*token_index)?
                    .get(..dims.hidden_dim)
                    .map(|slice| slice.to_vec())
            })
            .collect::<Option<Vec<_>>>()?;
        let (mut gate_rows, up_rows, expert_gate_up_tally) =
            project_batched_moe_expert_gate_up_with_tally(
                &moe.expert_gate_up,
                buffers,
                expert_index,
                dims.expert_intermediate_dim,
                &grouped_inputs,
                dims.hidden_dim,
                bringup,
            )?;
        tally = tally.merge(expert_gate_up_tally);
        tally = tally.merge(apply_batched_model_gate_up_product_in_place_with_tally(
            artifacts,
            &mut gate_rows,
            &up_rows,
            dims.expert_intermediate_dim,
            bringup,
        )?);
        let (expert_output_rows, expert_down_tally) =
            project_batched_moe_expert_matrix_rows_with_tally(
                expert_down,
                expert_index,
                0,
                dims.hidden_dim,
                &gate_rows,
                dims.expert_intermediate_dim,
                bringup,
            )?;
        tally = tally.merge(expert_down_tally);
        let expert_scale = match expert_down_scale {
            Some(scale_binding) => Some(tensor_scalar_f32(scale_binding, expert_index)?),
            None => None,
        };
        let mut selected_output_rows = assignments
            .iter()
            .map(|(token_index, _)| {
                output_rows
                    .get(*token_index)?
                    .get(..dims.hidden_dim)
                    .map(|slice| slice.to_vec())
            })
            .collect::<Option<Vec<_>>>()?;
        let combined_scales = assignments
            .iter()
            .map(|(_, expert_weight)| expert_scale.unwrap_or(1.0) * *expert_weight)
            .collect::<Vec<_>>();
        let mut scaled_delta_rows = expert_output_rows;
        let used_native_row_scale = apply_batched_row_scale_in_place_with_path(
            &mut scaled_delta_rows,
            dims.hidden_dim,
            &combined_scales,
            bringup,
        )?;
        tally = tally.record_scale_elements(
            scaled_delta_rows.len().checked_mul(dims.hidden_dim)?,
            used_native_row_scale,
        );
        for expert_output in &scaled_delta_rows {
            nontrivial |= has_nontrivial_ffn_contribution(expert_output);
        }
        tally = tally.merge(
            add_batched_rows_in_place_with_direct_decode_tally_and_result_dtype(
                &mut selected_output_rows,
                &scaled_delta_rows,
                dims.hidden_dim,
                expert_down.native_dtype,
                bringup,
            )?,
        );
        for ((token_index, _), updated_row) in assignments.into_iter().zip(selected_output_rows) {
            output_rows
                .get_mut(token_index)?
                .get_mut(..dims.hidden_dim)?
                .copy_from_slice(&updated_row);
        }
    }

    if let Some(post_norm) = expert_post_norm {
        tally = tally.merge(
            direct_decode_native_dense_tally_from_prefix_attention_tally(
                apply_batched_row_rms_norm_with_binding_in_place_with_tally(
                    &mut output_rows,
                    dims.hidden_dim,
                    post_norm,
                    native_model_rms_norm_epsilon(artifacts),
                    native_model_rms_norm_weight_offset(artifacts),
                    bringup,
                )?,
            ),
        );
    }

    Some((output_rows, tally, nontrivial))
}

#[cfg(target_os = "macos")]
fn apply_ffn_continuation_rows_with_tally(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    input_rows: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, DirectDecodeNativeDenseTally, bool)> {
    if input_rows.is_empty() {
        return Some((Vec::new(), DirectDecodeNativeDenseTally::default(), false));
    }

    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;
    let hidden_width = input_rows.iter().map(Vec::len).min()?;
    let ffn_norm_len = tensor_element_count(&ffn_norm.meta.spec)?;
    let dense_input_cols = ffn_gate_up_input_cols(&layer.ffn_gate_up, buffers)?;
    let dense_output_rows = tensor_matrix_dimensions(&ffn_down.meta.spec)?.0;
    let mut hidden_dim = hidden_width
        .min(artifacts.manifest().hidden_size as usize)
        .min(ffn_norm_len)
        .min(dense_input_cols)
        .min(dense_output_rows);
    if let Some(moe_dims) = resolved_layer_moe_dims(artifacts, layer, buffers, hidden_width) {
        hidden_dim = hidden_dim.min(moe_dims.hidden_dim);
    }
    if hidden_dim == 0 {
        return None;
    }

    // Fused single-token path: fuses norm → gate_proj → up_proj → activation →
    // down_proj → residual_add in ONE command buffer. Only for non-MoE, non-post_norm
    // layers (the same gates used by fused_layer_continuation_on_gpu).
    if input_rows.len() == 1 && layer.moe.is_none() && layer.ffn_post_norm.is_none() {
        if let Some(bringup) = bringup {
            let intermediate_dim =
                resolved_ffn_intermediate_dim(&layer.ffn_gate_up, buffers, tensor_matrix_dimensions(&ffn_down.meta.spec)?.1)?;
            if let Some(hidden_input) = input_rows.first().and_then(|r| r.get(..hidden_dim)) {
                if let Some((result, fused_tally)) =
                    fused_ffn_only_on_gpu(bringup, artifacts, layer, buffers, hidden_input, hidden_dim, intermediate_dim)
                {
                    let nontrivial = result.iter().any(|v| v.abs() > f32::EPSILON);
                    return Some((vec![result], fused_tally, nontrivial));
                }
            }
        }
    }

    let dense_post_norm = if layer.moe.is_some() {
        match &layer.ffn_post_norm_1 {
            Some(binding) => Some(buffers.binding_for(binding)?),
            None => None,
        }
    } else {
        None
    };
    let (mut branch_rows, mut tally, mut any_nontrivial) = apply_dense_ffn_branch_rows_with_tally(
        artifacts,
        layer,
        buffers,
        input_rows,
        hidden_dim,
        bringup,
        dense_post_norm,
    )?;

    if layer.moe.is_some() {
        let (moe_rows, moe_tally, moe_nontrivial) = apply_moe_ffn_branch_rows_with_tally(
            artifacts, layer, buffers, input_rows, hidden_dim, bringup,
        )?;
        tally = tally.merge(moe_tally);
        tally = tally.merge(
            add_batched_rows_in_place_with_direct_decode_tally_and_result_dtype(
                &mut branch_rows,
                &moe_rows,
                hidden_dim,
                ffn_down.native_dtype,
                bringup,
            )?,
        );
        any_nontrivial |= moe_nontrivial;
    }

    if let Some(post_norm) = &layer.ffn_post_norm {
        let post_norm = buffers.binding_for(post_norm)?;
        tally = tally.merge(
            direct_decode_native_dense_tally_from_prefix_attention_tally(
                apply_batched_row_rms_norm_with_binding_in_place_with_tally(
                    &mut branch_rows,
                    hidden_dim,
                    post_norm,
                    native_model_rms_norm_epsilon(artifacts),
                    native_model_rms_norm_weight_offset(artifacts),
                    bringup,
                )?,
            ),
        );
    }

    any_nontrivial |= branch_rows
        .iter()
        .any(|row| has_nontrivial_ffn_contribution(row));
    let mut next_hidden_rows = input_rows
        .iter()
        .map(|row| row.get(..hidden_dim).map(|slice| slice.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    tally = tally.merge(
        add_batched_rows_in_place_with_direct_decode_tally_and_result_dtype(
            &mut next_hidden_rows,
            &branch_rows,
            hidden_dim,
            ffn_down.native_dtype,
            bringup,
        )?,
    );

    Some((next_hidden_rows, tally, any_nontrivial))
}

#[allow(dead_code)]
fn apply_direct_decode_logits_to_runner_output(
    output: &mut RunnerOutput,
    logits_outputs: &[RequestLogitsOutput],
) {
    apply_owned_direct_decode_logits_to_runner_output(output, logits_outputs.to_vec());
}

fn apply_owned_direct_decode_logits_to_runner_output(
    output: &mut RunnerOutput,
    logits_outputs: Vec<RequestLogitsOutput>,
) {
    if logits_outputs.is_empty() {
        return;
    }

    let direct_decode_request_ids = logits_outputs
        .iter()
        .map(|output| output.request_id)
        .collect::<BTreeSet<_>>();
    output
        .logits_handles
        .retain(|request_id| !direct_decode_request_ids.contains(request_id));
    output
        .logits_outputs
        .retain(|output| !direct_decode_request_ids.contains(&output.request_id));
    output.logits_outputs.extend(logits_outputs);
}

fn apply_deterministic_sample_tokens_to_runner_output(
    input: &RunnerInput,
    output: &mut RunnerOutput,
    sample_tokens: &[(crate::ids::RequestId, u32)],
) -> BTreeSet<crate::ids::RequestId> {
    if sample_tokens.is_empty() {
        return BTreeSet::new();
    }

    let mut resolved_request_ids = BTreeSet::new();
    for (request_id, token_id) in sample_tokens {
        let Some(context) = input.request_context(*request_id) else {
            continue;
        };
        if !context.deterministic_argmax_sampling {
            continue;
        }
        let Some(update) = output
            .request_updates
            .iter_mut()
            .find(|update| update.request_id == *request_id)
        else {
            continue;
        };

        update.output_token = Some(*token_id);
        update.stop_reason = if context.generated_len.saturating_add(1) >= context.max_output_tokens
        {
            Some(StopReason::MaxOutputTokens)
        } else {
            None
        };
        resolved_request_ids.insert(*request_id);
    }

    if !resolved_request_ids.is_empty() {
        output
            .logits_handles
            .retain(|request_id| !resolved_request_ids.contains(request_id));
        output
            .logits_outputs
            .retain(|output| !resolved_request_ids.contains(&output.request_id));
    }

    resolved_request_ids
}

fn filter_model_bound_tokens_by_mode(
    input: &RunnerInput,
    tokens: &[(crate::ids::RequestId, u32)],
    mode: ExecutionMode,
) -> Vec<(crate::ids::RequestId, u32)> {
    let request_modes = input
        .execution_batch
        .items
        .iter()
        .map(|item| (item.request_id, item.mode))
        .collect::<BTreeMap<_, _>>();
    tokens
        .iter()
        .copied()
        .filter(|(request_id, _)| request_modes.get(request_id).copied() == Some(mode))
        .collect()
}

fn completed_real_model_forward_step(
    input: &RunnerInput,
    output: &RunnerOutput,
    runtime: &MetalDispatchRuntimeInfo,
    direct_decode_tokens: &[(crate::ids::RequestId, u32)],
) -> bool {
    if !runtime.real_model_tensor_inputs
        || !runtime_reports_complete_model_forward(runtime)
        || !output.logits_handles.is_empty()
    {
        return false;
    }

    let decode_request_ids = input
        .execution_batch
        .items
        .iter()
        .filter(|item| item.mode == ExecutionMode::Decode)
        .map(|item| item.request_id)
        .collect::<BTreeSet<_>>();
    let prefill_request_ids = input
        .execution_batch
        .items
        .iter()
        .filter(|item| item.mode != ExecutionMode::Decode)
        .map(|item| item.request_id)
        .collect::<BTreeSet<_>>();
    let prefill_completion_request_ids = prefill_completion_request_ids(input);
    let updates_by_request_id = output
        .request_updates
        .iter()
        .map(|update| (update.request_id, update))
        .collect::<BTreeMap<_, _>>();
    let logits_output_request_ids = output
        .logits_outputs
        .iter()
        .map(|output| output.request_id)
        .collect::<BTreeSet<_>>();

    if prefill_request_ids.iter().any(|request_id| {
        !prefill_completion_request_ids.contains(request_id)
            && updates_by_request_id
                .get(request_id)
                .and_then(|update| update.output_token)
                .is_some()
    }) {
        return false;
    }

    let resolved_request_ids = direct_decode_tokens
        .iter()
        .map(|(request_id, _)| *request_id)
        .collect::<BTreeSet<_>>();
    if resolved_request_ids
        .iter()
        .any(|request_id| !decode_request_ids.contains(request_id))
    {
        return false;
    }

    decode_request_ids.iter().all(|request_id| {
        updates_by_request_id
            .get(request_id)
            .and_then(|update| update.output_token)
            .is_some()
            || logits_output_request_ids.contains(request_id)
    }) && prefill_completion_request_ids.iter().all(|request_id| {
        updates_by_request_id
            .get(request_id)
            .is_some_and(|update| update.output_token.is_some())
            || logits_output_request_ids.contains(request_id)
    })
}

fn complete_model_forward_support_for_source(
    model: Option<&NativeModelArtifactsSummary>,
    source: Option<MetalStagedInputSource>,
) -> bool {
    let Some(model) = model else {
        return false;
    };
    let Some(source) = source else {
        return false;
    };
    let model_conditioned_source = matches!(
        source,
        MetalStagedInputSource::ModelConditionedMiniProjection
            | MetalStagedInputSource::ModelConditionedCpuPrefixAttention
            | MetalStagedInputSource::ModelConditionedNativePrefixAttention
            | MetalStagedInputSource::ModelConditionedMixedPrefixAttention
    );
    if !model_conditioned_source {
        return false;
    }
    if model.layer_count <= 1 {
        return true;
    }

    matches!(
        source,
        MetalStagedInputSource::ModelConditionedCpuPrefixAttention
            | MetalStagedInputSource::ModelConditionedNativePrefixAttention
            | MetalStagedInputSource::ModelConditionedMixedPrefixAttention
    )
}

fn runtime_reports_complete_model_forward(runtime: &MetalDispatchRuntimeInfo) -> bool {
    runtime.complete_model_forward_supported
}

fn annotate_bringup_execution_flags(
    route_metadata: &mut RouteMetadata,
    runtime: &MetalDispatchRuntimeInfo,
    direct_decode_tokens: &[(crate::ids::RequestId, u32)],
    flags: MetalBringupExecutionFlags,
) {
    route_metadata.crossover_decisions.push((
        "metal_dispatch_numeric_scaffold_only".to_string(),
        u32::from(!runtime.real_model_tensor_inputs),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_complete_model_forward_supported".to_string(),
        u32::from(flags.complete_model_forward_supported),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_real_model_forward".to_string(),
        u32::from(flags.real_model_forward),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_native_dispatch_count".to_string(),
        flags.prefix_attention_tally.native_dispatch_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_reference_dispatch_count".to_string(),
        flags.prefix_attention_tally.cpu_reference_dispatch_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_qkv_projection_token_count".to_string(),
        flags.prefix_attention_tally.qkv_projection_token_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_layer_continuation_token_count".to_string(),
        flags
            .prefix_attention_tally
            .layer_continuation_token_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_logits_projection_token_count".to_string(),
        flags.prefix_attention_tally.logits_projection_token_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_logits_vocab_scan_row_count".to_string(),
        flags.prefix_attention_tally.logits_vocab_scan_row_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_native_projection_row_count".to_string(),
        flags.prefix_attention_tally.native_projection_row_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_projection_row_count".to_string(),
        flags.prefix_attention_tally.cpu_projection_row_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_native_rms_norm_element_count".to_string(),
        flags.prefix_attention_tally.native_rms_norm_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_rms_norm_element_count".to_string(),
        flags.prefix_attention_tally.cpu_rms_norm_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_native_ffn_activation_element_count".to_string(),
        flags
            .prefix_attention_tally
            .native_ffn_activation_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_ffn_activation_element_count".to_string(),
        flags
            .prefix_attention_tally
            .cpu_ffn_activation_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_native_residual_add_element_count".to_string(),
        flags
            .prefix_attention_tally
            .native_residual_add_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_residual_add_element_count".to_string(),
        flags
            .prefix_attention_tally
            .cpu_residual_add_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_native_scale_element_count".to_string(),
        flags.prefix_attention_tally.native_scale_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_cpu_scale_element_count".to_string(),
        flags.prefix_attention_tally.cpu_scale_element_count(),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_logits_projection".to_string(),
        u32::from(flags.native_logits_projection_decode),
    ));
    annotate_direct_decode_tokens(
        route_metadata,
        direct_decode_tokens,
        flags.model_bound_ffn_decode,
        flags.direct_decode_native_dense_tally,
    );
}

fn annotate_staged_input_source(
    route_metadata: &mut RouteMetadata,
    source: MetalStagedInputSource,
) {
    let (model_conditioned, token_only, native_prefix_attention, cpu_reference_prefix_attention) =
        match source {
            MetalStagedInputSource::SyntheticTokenIds => (0, 1, 0, 0),
            MetalStagedInputSource::ModelConditionedMiniProjection => (1, 0, 0, 0),
            MetalStagedInputSource::ModelConditionedCpuPrefixAttention => (1, 0, 0, 1),
            MetalStagedInputSource::ModelConditionedNativePrefixAttention => (1, 0, 1, 0),
            MetalStagedInputSource::ModelConditionedMixedPrefixAttention => (1, 0, 1, 1),
        };

    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_layers_native_attention".to_string(),
        native_prefix_attention,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_prefix_layers_cpu_reference".to_string(),
        cpu_reference_prefix_attention,
    ));

    route_metadata.crossover_decisions.push((
        "metal_dispatch_model_conditioned_inputs".to_string(),
        model_conditioned,
    ));
    route_metadata
        .crossover_decisions
        .push(("metal_dispatch_token_only_inputs".to_string(), token_only));
}

fn annotate_direct_decode_tokens(
    route_metadata: &mut RouteMetadata,
    direct_decode_tokens: &[(crate::ids::RequestId, u32)],
    model_bound_ffn_decode: bool,
    direct_decode_native_dense_tally: DirectDecodeNativeDenseTally,
) {
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_tokens".to_string(),
        direct_decode_tokens.len() as u32,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_checksum_lo".to_string(),
        saturating_u64_to_u32(checksum_direct_decode_tokens(direct_decode_tokens)),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_model_bound_ffn".to_string(),
        u32::from(model_bound_ffn_decode),
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_projection_row_count".to_string(),
        direct_decode_native_dense_tally.native_projection_rows,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_projection_row_count".to_string(),
        direct_decode_native_dense_tally.cpu_projection_rows,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_rms_norm_element_count".to_string(),
        direct_decode_native_dense_tally.native_rms_norm_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_rms_norm_element_count".to_string(),
        direct_decode_native_dense_tally.cpu_rms_norm_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_ffn_activation_element_count".to_string(),
        direct_decode_native_dense_tally.native_ffn_activation_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_ffn_activation_element_count".to_string(),
        direct_decode_native_dense_tally.cpu_ffn_activation_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_residual_add_element_count".to_string(),
        direct_decode_native_dense_tally.native_residual_add_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_residual_add_element_count".to_string(),
        direct_decode_native_dense_tally.cpu_residual_add_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_native_scale_element_count".to_string(),
        direct_decode_native_dense_tally.native_scale_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_cpu_scale_element_count".to_string(),
        direct_decode_native_dense_tally.cpu_scale_elements,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_logits_group_count".to_string(),
        direct_decode_native_dense_tally.native_batched_logits_group_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_logits_token_count".to_string(),
        direct_decode_native_dense_tally.native_batched_logits_token_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_group_fallback_count".to_string(),
        direct_decode_native_dense_tally.batched_group_fallback_count,
    ));
    route_metadata.crossover_decisions.push((
        "metal_dispatch_direct_decode_batched_group_fallback_token_count".to_string(),
        direct_decode_native_dense_tally.batched_group_fallback_token_count,
    ));
}

fn checksum_direct_decode_tokens(direct_decode_tokens: &[(crate::ids::RequestId, u32)]) -> u64 {
    checksum_words(
        direct_decode_tokens
            .iter()
            .flat_map(|(request_id, token_id)| {
                [
                    *token_id,
                    *token_id ^ request_id.0 as u32,
                    (request_id.0 >> 32) as u32,
                ]
            }),
    )
}

fn metal_dispatch_execution_info(
    output: &RunnerOutput,
    direct_decode_tokens: &[(crate::ids::RequestId, u32)],
    model_bound_ffn_decode: bool,
    real_model_forward_completed: bool,
    prefix_attention_tally: PrefixAttentionExecutionTally,
    direct_decode_native_dense_tally: DirectDecodeNativeDenseTally,
) -> MetalDispatchExecutionInfo {
    MetalDispatchExecutionInfo {
        direct_decode_token_count: direct_decode_tokens.len() as u32,
        direct_decode_checksum_lo: saturating_u64_to_u32(checksum_direct_decode_tokens(
            direct_decode_tokens,
        )),
        logits_output_count: output.logits_outputs.len() as u32,
        remaining_logits_handle_count: output.logits_handles.len() as u32,
        model_bound_ffn_decode,
        real_model_forward_completed,
        prefix_native_dispatch_count: prefix_attention_tally.native_dispatch_count(),
        prefix_cpu_reference_dispatch_count: prefix_attention_tally.cpu_reference_dispatch_count(),
        qkv_projection_token_count: prefix_attention_tally.qkv_projection_token_count(),
        layer_continuation_token_count: prefix_attention_tally.layer_continuation_token_count(),
        logits_projection_token_count: prefix_attention_tally.logits_projection_token_count(),
        logits_vocab_scan_row_count: prefix_attention_tally.logits_vocab_scan_row_count(),
        prefix_native_projection_row_count: prefix_attention_tally.native_projection_row_count(),
        prefix_cpu_projection_row_count: prefix_attention_tally.cpu_projection_row_count(),
        prefix_native_rms_norm_element_count: prefix_attention_tally
            .native_rms_norm_element_count(),
        prefix_cpu_rms_norm_element_count: prefix_attention_tally.cpu_rms_norm_element_count(),
        prefix_native_ffn_activation_element_count: prefix_attention_tally
            .native_ffn_activation_element_count(),
        prefix_cpu_ffn_activation_element_count: prefix_attention_tally
            .cpu_ffn_activation_element_count(),
        prefix_native_residual_add_element_count: prefix_attention_tally
            .native_residual_add_element_count(),
        prefix_cpu_residual_add_element_count: prefix_attention_tally
            .cpu_residual_add_element_count(),
        prefix_native_scale_element_count: prefix_attention_tally.native_scale_element_count(),
        prefix_cpu_scale_element_count: prefix_attention_tally.cpu_scale_element_count(),
        direct_decode_native_projection_row_count: direct_decode_native_dense_tally
            .native_projection_rows,
        direct_decode_cpu_projection_row_count: direct_decode_native_dense_tally
            .cpu_projection_rows,
        direct_decode_native_rms_norm_element_count: direct_decode_native_dense_tally
            .native_rms_norm_elements,
        direct_decode_cpu_rms_norm_element_count: direct_decode_native_dense_tally
            .cpu_rms_norm_elements,
        direct_decode_native_ffn_activation_element_count: direct_decode_native_dense_tally
            .native_ffn_activation_elements,
        direct_decode_cpu_ffn_activation_element_count: direct_decode_native_dense_tally
            .cpu_ffn_activation_elements,
        direct_decode_native_residual_add_element_count: direct_decode_native_dense_tally
            .native_residual_add_elements,
        direct_decode_cpu_residual_add_element_count: direct_decode_native_dense_tally
            .cpu_residual_add_elements,
        direct_decode_native_scale_element_count: direct_decode_native_dense_tally
            .native_scale_elements,
        direct_decode_cpu_scale_element_count: direct_decode_native_dense_tally.cpu_scale_elements,
        direct_decode_batched_logits_group_count: direct_decode_native_dense_tally
            .native_batched_logits_group_count,
        direct_decode_batched_logits_token_count: direct_decode_native_dense_tally
            .native_batched_logits_token_count,
        direct_decode_batched_group_fallback_count: direct_decode_native_dense_tally
            .batched_group_fallback_count,
        direct_decode_batched_group_fallback_token_count: direct_decode_native_dense_tally
            .batched_group_fallback_token_count,
    }
}

#[cfg(target_os = "macos")]
fn load_macos_runtime_bringup(
    assets: MetalKernelAssets,
    metallib: MetalKernelBinary,
    required_kernel_names: Vec<String>,
) -> Result<MetalRuntimeBringup, MetalRuntimeError> {
    let device = Device::system_default().ok_or(MetalRuntimeError::NoSystemDevice)?;
    let command_queue = device.new_command_queue();
    command_queue.set_label("ax.phase1.command_queue");
    let library = device
        .new_library_with_data(&metallib.bytes)
        .map_err(|message| MetalRuntimeError::LoadCompiledLibrary {
            path: metallib.path.clone(),
            message,
        })?;

    let mut compiled_kernel_names = library.function_names();
    compiled_kernel_names.sort();
    validate_compiled_kernel_inventory(assets.manifest(), &compiled_kernel_names, &metallib.path)?;

    let device_info = device_info(&device);
    let mut binary_archive_session =
        prepare_binary_archive(&device, binary_archive_path(&metallib.path, &assets));
    let mut required_pipelines = Vec::with_capacity(required_kernel_names.len());
    let mut required_pipeline_handles = Vec::with_capacity(required_kernel_names.len());
    for kernel_name in &required_kernel_names {
        let prepared = prepare_compute_pipeline(
            &device,
            &library,
            &device_info.name,
            &metallib.path,
            kernel_name,
            &mut binary_archive_session,
        )?;
        required_pipelines.push(prepared.info);
        required_pipeline_handles.push(prepared.handle);
    }
    let required_pipeline_lookup = pipeline_lookup_index(&required_kernel_names);
    let optional_kernel_names = PHASE1_OPTIONAL_METAL_KERNELS
        .iter()
        .filter(|kernel_name| assets.kernel(kernel_name).is_some())
        .map(|kernel_name| (*kernel_name).to_string())
        .collect::<Vec<_>>();
    let mut optional_pipeline_handles = Vec::with_capacity(optional_kernel_names.len());
    for kernel_name in &optional_kernel_names {
        let prepared = prepare_compute_pipeline(
            &device,
            &library,
            &device_info.name,
            &metallib.path,
            kernel_name,
            &mut binary_archive_session,
        )?;
        optional_pipeline_handles.push(prepared.handle);
    }
    let optional_pipeline_lookup = pipeline_lookup_index(&optional_kernel_names);
    let optional_kernel_dispatch_plan =
        build_optional_kernel_dispatch_plan(&optional_pipeline_lookup);
    let binary_archive = finalize_binary_archive(binary_archive_session);
    let deferred_kernel_names = PHASE1_DEFERRED_METAL_KERNELS
        .iter()
        .filter(|kernel_name| assets.kernel(kernel_name).is_some())
        .map(|kernel_name| (*kernel_name).to_string())
        .collect::<Vec<_>>();

    let report = MetalRuntimeBringupReport {
        library_name: assets.manifest().library_name.clone(),
        metallib_path: metallib.path.clone(),
        device: device_info,
        compiled_kernel_names,
        required_pipelines,
        deferred_kernel_names,
        optional_kernel_names,
        binary_archive,
        command_queue_ready: true,
        model: None,
    };

    Ok(MetalRuntimeBringup {
        assets,
        metallib,
        report,
        state: MetalRuntimeState {
            device,
            command_queue,
            library,
            required_pipelines: required_pipeline_handles,
            required_pipeline_lookup,
            optional_pipelines: optional_pipeline_handles,
            optional_kernel_dispatch_plan,
            optional_kernel_feedback: Mutex::new(MetalOptionalKernelFeedbackState::default()),
            dispatch_arena: Mutex::new(None),
            fused_layer_arena: Mutex::new(None),
        },
    })
}

#[cfg(target_os = "macos")]
struct PreparedComputePipeline {
    info: MetalComputePipelineInfo,
    handle: MetalPipelineHandle,
}

#[cfg(target_os = "macos")]
fn prepare_compute_pipeline(
    device: &Device,
    library: &Library,
    device_name: &str,
    metallib_path: &Path,
    function_name: &str,
    binary_archive_session: &mut BinaryArchiveSession,
) -> Result<PreparedComputePipeline, MetalRuntimeError> {
    let function = library
        .get_function(function_name, None)
        .map_err(|message| MetalRuntimeError::ResolveCompiledKernel {
            path: metallib_path.to_path_buf(),
            function_name: function_name.to_string(),
            message,
        })?;
    let pipeline = build_compute_pipeline_state(
        device,
        &function,
        function_name,
        device_name,
        binary_archive_session,
    )
    .map_err(|message| MetalRuntimeError::CreateComputePipeline {
        function_name: function_name.to_string(),
        device_name: device_name.to_string(),
        message,
    })?;

    Ok(PreparedComputePipeline {
        info: MetalComputePipelineInfo {
            function_name: function_name.to_string(),
            thread_execution_width: pipeline.thread_execution_width(),
            max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup(),
            static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
        },
        handle: MetalPipelineHandle {
            function_name: function_name.to_string(),
            pipeline,
        },
    })
}

#[cfg(target_os = "macos")]
fn build_compute_pipeline_state(
    device: &Device,
    function: &metal::FunctionRef,
    function_name: &str,
    device_name: &str,
    binary_archive_session: &mut BinaryArchiveSession,
) -> Result<ComputePipelineState, String> {
    let descriptor = ComputePipelineDescriptor::new();
    descriptor.set_label(&format!("ax.{function_name}.compute_pipeline"));
    descriptor.set_compute_function(Some(function));
    descriptor.set_thread_group_size_is_multiple_of_thread_execution_width(true);

    if let Some(archive) = binary_archive_session.archive.as_ref() {
        descriptor.set_binary_archives(&[archive]);
        match archive.add_compute_pipeline_functions_with_descriptor(&descriptor) {
            Ok(_) => match device.new_compute_pipeline_state(&descriptor) {
                Ok(pipeline) => {
                    binary_archive_session.info.attached_pipeline_count += 1;
                    return Ok(pipeline);
                }
                Err(message) => {
                    append_binary_archive_note(
                        &mut binary_archive_session.info,
                        format!(
                            "archive-backed pipeline build failed for {function_name} on {device_name}; retrying without archive: {message}"
                        ),
                    );
                    descriptor.set_binary_archives(&[]);
                }
            },
            Err(message) => {
                append_binary_archive_note(
                    &mut binary_archive_session.info,
                    format!(
                        "archive attach failed for {function_name} on {device_name}; falling back to uncached pipeline build: {message}"
                    ),
                );
                descriptor.set_binary_archives(&[]);
            }
        }
    }

    device.new_compute_pipeline_state(&descriptor)
}

#[cfg(target_os = "macos")]
fn binary_archive_path(metallib_path: &Path, assets: &MetalKernelAssets) -> PathBuf {
    metallib_path
        .parent()
        .unwrap_or_else(|| assets.build_dir())
        .join(format!(
            "{}.binary_archive.metallib",
            assets.manifest().library_name
        ))
}

#[cfg(target_os = "macos")]
fn prepare_binary_archive(device: &Device, archive_path: PathBuf) -> BinaryArchiveSession {
    let _ = device;
    let mut info = MetalBinaryArchiveInfo {
        path: archive_path.clone(),
        state: MetalBinaryArchiveState::Disabled,
        attached_pipeline_count: 0,
        serialized: false,
        note: None,
    };
    append_binary_archive_note(
        &mut info,
        "binary archive disabled while Metal session initialization is stabilized".to_string(),
    );
    BinaryArchiveSession {
        archive: None,
        info,
    }
}

#[cfg(target_os = "macos")]
fn finalize_binary_archive(mut session: BinaryArchiveSession) -> MetalBinaryArchiveInfo {
    if let Some(archive) = session.archive.as_ref() {
        if session.info.attached_pipeline_count > 0 {
            let _ = archive;
            append_binary_archive_note(
                &mut session.info,
                "binary archive serialization skipped while session initialization is stabilized"
                    .to_string(),
            );
        } else {
            append_binary_archive_note(
                &mut session.info,
                "binary archive was available but no pipelines were attached".to_string(),
            );
        }
    }

    session.info
}

#[cfg(target_os = "macos")]
fn append_binary_archive_note(info: &mut MetalBinaryArchiveInfo, message: String) {
    match info.note.as_mut() {
        Some(note) => {
            note.push_str("; ");
            note.push_str(&message);
        }
        None => {
            info.note = Some(message);
        }
    }
}

#[cfg(target_os = "macos")]
fn device_info(device: &Device) -> MetalDeviceInfo {
    let max_threads = device.max_threads_per_threadgroup();
    MetalDeviceInfo {
        name: device.name().to_string(),
        registry_id: device.registry_id(),
        low_power: device.is_low_power(),
        headless: device.is_headless(),
        removable: device.is_removable(),
        max_threadgroup_memory_length: device.max_threadgroup_memory_length(),
        max_threads_per_threadgroup: MetalThreadgroupSize {
            width: max_threads.width,
            height: max_threads.height,
            depth: max_threads.depth,
        },
    }
}

fn build_dispatch_traces(
    workload: &MetalDispatchWorkload,
    required_pipelines: &[MetalComputePipelineInfo],
) -> Vec<MetalDispatchKernelTrace> {
    let mut ordered_pipelines = required_pipelines.iter().collect::<Vec<_>>();
    ordered_pipelines.sort_by_key(|pipeline| dispatch_execution_order(&pipeline.function_name));
    ordered_pipelines
        .into_iter()
        .map(|pipeline| build_dispatch_trace(workload, pipeline))
        .collect()
}

fn build_dispatch_trace(
    workload: &MetalDispatchWorkload,
    pipeline: &MetalComputePipelineInfo,
) -> MetalDispatchKernelTrace {
    let element_count = dispatch_element_count_for_kernel(&pipeline.function_name, workload);
    let threads_per_threadgroup_width = pipeline
        .thread_execution_width
        .max(1)
        .min(pipeline.max_total_threads_per_threadgroup.max(1))
        .min(u64::from(element_count).max(1));

    MetalDispatchKernelTrace {
        function_name: pipeline.function_name.clone(),
        element_count,
        threads_per_grid: MetalThreadgroupSize {
            width: u64::from(element_count).max(1),
            height: 1,
            depth: 1,
        },
        threads_per_threadgroup: MetalThreadgroupSize {
            width: threads_per_threadgroup_width,
            height: 1,
            depth: 1,
        },
    }
}

fn dispatch_element_count_for_kernel(function_name: &str, workload: &MetalDispatchWorkload) -> u32 {
    match function_name {
        "reshape_and_cache" => workload.scheduled_numeric_elements(),
        "paged_decode_attention" => workload.attention_numeric_elements(),
        "gather_kv_cache" => workload.gather_numeric_elements(),
        "copy_blocks" => workload.copy_numeric_elements(),
        _ => workload.scratch_elements,
    }
}

fn dispatch_execution_order(function_name: &str) -> u8 {
    match function_name {
        "reshape_and_cache" => 0,
        "gather_kv_cache" => 1,
        "paged_decode_attention" => 2,
        "copy_blocks" => 3,
        _ => u8::MAX,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetalDispatchArenaMode {
    Persistent,
    Ephemeral,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ReferenceAttentionConfig {
    softmax_scale: f32,
    softcap: Option<f32>,
}

impl ReferenceAttentionConfig {
    fn from_head_dim(head_dim: usize) -> Option<Self> {
        (head_dim > 0).then_some(Self {
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            softcap: None,
        })
    }
}

#[cfg(target_os = "macos")]
fn dispatch_numeric_workload_macos(
    bringup: &MetalRuntimeBringup,
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    kernels: Vec<MetalDispatchKernelTrace>,
    arena_mode: MetalDispatchArenaMode,
    attention_config: Option<ReferenceAttentionConfig>,
) -> Result<MetalDispatchTrace, MetalRuntimeError> {
    dispatch_numeric_workload_macos_with_cache_seed(
        bringup,
        workload,
        staged_inputs,
        kernels,
        arena_mode,
        None,
        attention_config,
    )
    .map(|(trace, _)| trace)
}

#[cfg(target_os = "macos")]
fn dispatch_numeric_workload_macos_with_cache_seed(
    bringup: &MetalRuntimeBringup,
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    kernels: Vec<MetalDispatchKernelTrace>,
    arena_mode: MetalDispatchArenaMode,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
    attention_config: Option<ReferenceAttentionConfig>,
) -> Result<(MetalDispatchTrace, MetalDispatchKvCacheSnapshot), MetalRuntimeError> {
    let ordered_pipelines = kernels
        .iter()
        .map(|trace| {
            find_required_pipeline_handle(
                &bringup.state,
                &bringup.metallib.path,
                &trace.function_name,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transformed_cache_seed = cache_seed.and_then(|seed| {
        owned_cache_seed_with_staged_inputs_and_copy_targets(workload, staged_inputs, Some(seed))
    });
    let validation_cache_seed = transformed_cache_seed.as_ref().map(|seed| seed.as_seed());

    autoreleasepool(|| {
        let (mut ephemeral_arena, arena_info) = match arena_mode {
            MetalDispatchArenaMode::Persistent => (None, None),
            MetalDispatchArenaMode::Ephemeral => {
                let requirements = MetalDispatchArenaRequirements::from_workload(workload);
                let arena_info = requirements.info(false, false);
                (
                    Some(MetalDispatchArena::new(&bringup.state.device, requirements)),
                    Some(arena_info),
                )
            }
        };
        let mut persistent_guard = match arena_mode {
            MetalDispatchArenaMode::Persistent => Some(
                bringup
                    .state
                    .dispatch_arena
                    .lock()
                    .expect("metal dispatch arena mutex should not be poisoned"),
            ),
            MetalDispatchArenaMode::Ephemeral => None,
        };
        let (arena, arena_info) = match arena_mode {
            MetalDispatchArenaMode::Persistent => ensure_dispatch_arena(
                &bringup.state.device,
                &bringup.state.command_queue,
                persistent_guard
                    .as_mut()
                    .expect("persistent arena guard should exist"),
                workload,
            ),
            MetalDispatchArenaMode::Ephemeral => (
                ephemeral_arena
                    .as_mut()
                    .expect("ephemeral arena should be initialized"),
                arena_info.expect("ephemeral arena info should exist"),
            ),
        };
        let persistent_validation_seed = match arena_mode {
            MetalDispatchArenaMode::Persistent => cache_seed_from_arena(workload, arena),
            MetalDispatchArenaMode::Ephemeral => None,
        };
        if let Some(cache_seed) = validation_cache_seed {
            arena.write_cache_seed(&bringup.state.device, cache_seed);
        }
        arena.write_workload(&bringup.state.device, workload, staged_inputs);
        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label(match arena_mode {
            MetalDispatchArenaMode::Persistent => "ax.phase1.numeric_dispatch",
            MetalDispatchArenaMode::Ephemeral => "ax.phase1.numeric_dispatch.ephemeral",
        });
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label(match arena_mode {
            MetalDispatchArenaMode::Persistent => "ax.phase1.numeric_compute",
            MetalDispatchArenaMode::Ephemeral => "ax.phase1.numeric_compute.ephemeral",
        });

        for (trace, pipeline) in kernels.iter().zip(&ordered_pipelines) {
            encode_numeric_kernel(encoder, pipeline, trace, workload, arena);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let command_buffer_status = command_buffer_status(command_buffer.status());
        if command_buffer_status != MetalCommandBufferStatus::Completed {
            return Err(MetalRuntimeError::CommandBufferNotCompleted {
                status: command_buffer_status,
            });
        }
        let cache_snapshot = capture_numeric_cache_snapshot(workload, arena);
        let mut numeric = capture_numeric_trace(workload, arena);
        let validation = validate_numeric_trace_against_inputs_and_cache_seed(
            workload,
            staged_inputs,
            validation_cache_seed.or_else(|| {
                persistent_validation_seed
                    .as_ref()
                    .map(|seed| seed.as_seed())
            }),
            attention_config,
            &numeric,
        )?;
        numeric.validation = Some(validation);
        Ok((
            MetalDispatchTrace {
                command_queue_label: bringup.state.command_queue.label().to_string(),
                command_buffer_label: command_buffer.label().to_string(),
                command_buffer_status,
                runtime: MetalDispatchRuntimeInfo::from_bringup_report(&bringup.report),
                workload: workload.clone(),
                arena: arena_info,
                execution: MetalDispatchExecutionInfo::default(),
                kernels,
                numeric,
            },
            cache_snapshot,
        ))
    })
}

#[cfg(target_os = "macos")]
fn cache_seed_from_arena(
    workload: &MetalDispatchWorkload,
    arena: &MetalDispatchArena,
) -> Option<OwnedMetalDispatchKvCacheSeed> {
    let element_count = workload.slot_numeric_capacity();
    Some(OwnedMetalDispatchKvCacheSeed {
        key_cache: read_shared_buffer_prefix(&arena.key_cache, element_count),
        value_cache: read_shared_buffer_prefix(&arena.value_cache, element_count),
    })
}

#[cfg(target_os = "macos")]
fn merge_copy_targets_into_cache_snapshot(
    workload: &MetalDispatchWorkload,
    key_cache: &mut [f32],
    value_cache: &mut [f32],
    copy_key: &[f32],
    copy_value: &[f32],
) {
    let head_size = workload.numeric_layout.head_size() as usize;
    let block_width = workload.block_numeric_elements() as usize;
    if head_size == 0 || block_width == 0 {
        return;
    }

    for &[_, target_base] in &workload.kv_metadata.copy_block_mapping {
        let target_slot = target_base as usize;
        let target_numeric_base = match target_slot.checked_mul(head_size) {
            Some(base) => base,
            None => continue,
        };
        let target_numeric_end = match target_numeric_base.checked_add(block_width) {
            Some(end) => end,
            None => continue,
        };
        let Some(copy_key_block) = copy_key.get(target_numeric_base..target_numeric_end) else {
            continue;
        };
        let Some(copy_value_block) = copy_value.get(target_numeric_base..target_numeric_end) else {
            continue;
        };
        let Some(cache_key_block) = key_cache.get_mut(target_numeric_base..target_numeric_end)
        else {
            continue;
        };
        let Some(cache_value_block) = value_cache.get_mut(target_numeric_base..target_numeric_end)
        else {
            continue;
        };
        cache_key_block.copy_from_slice(copy_key_block);
        cache_value_block.copy_from_slice(copy_value_block);
    }

    propagate_copy_targets_in_cache(workload, key_cache, value_cache);
}

#[cfg(target_os = "macos")]
fn materialize_copy_targets_into_owned_cache_seed(
    workload: &MetalDispatchWorkload,
    cache_seed: MetalDispatchKvCacheSeed<'_>,
) -> OwnedMetalDispatchKvCacheSeed {
    let mut owned_seed = OwnedMetalDispatchKvCacheSeed {
        key_cache: cache_seed.key_cache.to_vec(),
        value_cache: cache_seed.value_cache.to_vec(),
    };
    let head_size = workload.numeric_layout.head_size() as usize;
    let block_width = workload.block_numeric_elements() as usize;
    if head_size == 0 || block_width == 0 {
        return owned_seed;
    }

    for &[source_base, target_base] in &workload.kv_metadata.copy_block_mapping {
        let Some(source_numeric_base) = (source_base as usize).checked_mul(head_size) else {
            continue;
        };
        let Some(target_numeric_base) = (target_base as usize).checked_mul(head_size) else {
            continue;
        };
        let Some(source_numeric_end) = source_numeric_base.checked_add(block_width) else {
            continue;
        };
        let Some(target_numeric_end) = target_numeric_base.checked_add(block_width) else {
            continue;
        };
        let Some(source_key_block) = owned_seed
            .key_cache
            .get(source_numeric_base..source_numeric_end)
            .map(|block| block.to_vec())
        else {
            continue;
        };
        let Some(source_value_block) = owned_seed
            .value_cache
            .get(source_numeric_base..source_numeric_end)
            .map(|block| block.to_vec())
        else {
            continue;
        };
        let Some(target_key_block) = owned_seed
            .key_cache
            .get_mut(target_numeric_base..target_numeric_end)
        else {
            continue;
        };
        let Some(target_value_block) = owned_seed
            .value_cache
            .get_mut(target_numeric_base..target_numeric_end)
        else {
            continue;
        };
        target_key_block.copy_from_slice(&source_key_block);
        target_value_block.copy_from_slice(&source_value_block);
    }

    propagate_copy_targets_in_cache(
        workload,
        &mut owned_seed.key_cache,
        &mut owned_seed.value_cache,
    );

    owned_seed
}

#[cfg(target_os = "macos")]
fn owned_cache_seed_with_staged_inputs_and_copy_targets(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
) -> Option<OwnedMetalDispatchKvCacheSeed> {
    let head_size = workload.numeric_layout.head_size() as usize;
    let scheduled_tokens = workload.token_elements as usize;
    let slot_numeric_capacity = workload.slot_numeric_capacity() as usize;
    if head_size == 0
        || staged_inputs.key.len() < scheduled_tokens.checked_mul(head_size)?
        || staged_inputs.value.len() < scheduled_tokens.checked_mul(head_size)?
    {
        return None;
    }

    let mut owned_seed = cache_seed
        .map(|seed| OwnedMetalDispatchKvCacheSeed {
            key_cache: seed.key_cache.to_vec(),
            value_cache: seed.value_cache.to_vec(),
        })
        .unwrap_or_else(|| OwnedMetalDispatchKvCacheSeed {
            key_cache: vec![0.0; slot_numeric_capacity],
            value_cache: vec![0.0; slot_numeric_capacity],
        });
    for token_id in 0..scheduled_tokens {
        let slot = *workload.kv_metadata.slot_mapping.get(token_id)? as usize;
        let source_base = token_id.checked_mul(head_size)?;
        let target_base = slot.checked_mul(head_size)?;
        let source_end = source_base.checked_add(head_size)?;
        let target_end = target_base.checked_add(head_size)?;
        let source_key = staged_inputs.key.get(source_base..source_end)?;
        let source_value = staged_inputs.value.get(source_base..source_end)?;
        let target_key = owned_seed.key_cache.get_mut(target_base..target_end)?;
        let target_value = owned_seed.value_cache.get_mut(target_base..target_end)?;
        target_key.copy_from_slice(source_key);
        target_value.copy_from_slice(source_value);
    }

    Some(materialize_copy_targets_into_owned_cache_seed(
        workload,
        owned_seed.as_seed(),
    ))
}

#[cfg(target_os = "macos")]
fn self_contained_owned_cache_seed_from_staged_inputs(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
) -> Option<OwnedMetalDispatchKvCacheSeed> {
    owned_cache_seed_with_staged_inputs_and_copy_targets(workload, staged_inputs, None)
}

#[cfg(target_os = "macos")]
fn propagate_copy_targets_in_cache(
    workload: &MetalDispatchWorkload,
    key_cache: &mut [f32],
    value_cache: &mut [f32],
) {
    let head_size = workload.numeric_layout.head_size() as usize;
    let block_width = workload.block_numeric_elements() as usize;
    if head_size == 0 || block_width == 0 {
        return;
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &[source_base, target_base] in &workload.kv_metadata.copy_block_mapping {
            let Some(source_numeric_base) = (source_base as usize).checked_mul(head_size) else {
                continue;
            };
            let Some(target_numeric_base) = (target_base as usize).checked_mul(head_size) else {
                continue;
            };
            let Some(source_numeric_end) = source_numeric_base.checked_add(block_width) else {
                continue;
            };
            let Some(target_numeric_end) = target_numeric_base.checked_add(block_width) else {
                continue;
            };
            let Some(source_key_block) = key_cache
                .get(source_numeric_base..source_numeric_end)
                .map(|block| block.to_vec())
            else {
                continue;
            };
            let Some(source_value_block) = value_cache
                .get(source_numeric_base..source_numeric_end)
                .map(|block| block.to_vec())
            else {
                continue;
            };
            let Some(target_key_block) = key_cache.get_mut(target_numeric_base..target_numeric_end)
            else {
                continue;
            };
            let Some(target_value_block) =
                value_cache.get_mut(target_numeric_base..target_numeric_end)
            else {
                continue;
            };
            if target_key_block != source_key_block.as_slice()
                || target_value_block != source_value_block.as_slice()
            {
                target_key_block.copy_from_slice(&source_key_block);
                target_value_block.copy_from_slice(&source_value_block);
                changed = true;
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn propagate_initialized_copy_targets(
    workload: &MetalDispatchWorkload,
    initialized_slots: &mut [bool],
) {
    let block_width = workload.kv_metadata.block_size_tokens.max(1);
    let mut changed = true;
    while changed {
        changed = false;
        for &[source_base, target_base] in &workload.kv_metadata.copy_block_mapping {
            for offset in 0..block_width {
                let source_slot = source_base.saturating_add(offset);
                let target_slot = target_base.saturating_add(offset);
                let source_is_initialized = initialized_slots
                    .get(source_slot as usize)
                    .copied()
                    .unwrap_or(false);
                if !source_is_initialized {
                    continue;
                }
                let Some(target_initialized) = initialized_slots.get_mut(target_slot as usize)
                else {
                    continue;
                };
                if !*target_initialized {
                    *target_initialized = true;
                    changed = true;
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn supported_prefix_slots(
    workload: &MetalDispatchWorkload,
    mut slot_initialized: impl FnMut(u32) -> bool,
) -> Vec<bool> {
    let slot_capacity = workload
        .kv_slot_capacity
        .max(workload.kv_metadata.slot_capacity())
        .max(1) as usize;
    let mut supported_slots = vec![false; slot_capacity];
    for &slot in &workload.kv_metadata.slot_mapping {
        if let Some(entry) = supported_slots.get_mut(slot as usize) {
            *entry = true;
        }
    }
    for slot in 0..slot_capacity as u32 {
        if slot_initialized(slot) {
            if let Some(entry) = supported_slots.get_mut(slot as usize) {
                *entry = true;
            }
        }
    }

    propagate_initialized_copy_targets(workload, &mut supported_slots);

    supported_slots
}

#[cfg(target_os = "macos")]
fn find_required_pipeline_handle<'a>(
    state: &'a MetalRuntimeState,
    metallib_path: &Path,
    function_name: &str,
) -> Result<&'a MetalPipelineHandle, MetalRuntimeError> {
    find_pipeline_handle(
        &state.required_pipelines,
        &state.required_pipeline_lookup,
        metallib_path,
        function_name,
        "required",
    )
}

#[cfg(target_os = "macos")]
fn find_optional_pipeline_handle_by_index<'a>(
    state: &'a MetalRuntimeState,
    metallib_path: &Path,
    function_name: &str,
    pipeline_index: usize,
) -> Result<&'a MetalPipelineHandle, MetalRuntimeError> {
    state
        .optional_pipelines
        .get(pipeline_index)
        .ok_or(MetalRuntimeError::ResolveCompiledKernel {
            path: metallib_path.to_path_buf(),
            function_name: function_name.to_string(),
            message: "optional pipeline handle index is missing from runtime state".to_string(),
        })
}

#[cfg(target_os = "macos")]
fn find_pipeline_handle<'a>(
    pipelines: &'a [MetalPipelineHandle],
    lookup: &BTreeMap<String, usize>,
    metallib_path: &Path,
    function_name: &str,
    pipeline_kind: &str,
) -> Result<&'a MetalPipelineHandle, MetalRuntimeError> {
    lookup
        .get(function_name)
        .and_then(|index| pipelines.get(*index))
        .ok_or(MetalRuntimeError::ResolveCompiledKernel {
            path: metallib_path.to_path_buf(),
            function_name: function_name.to_string(),
            message: format!("{pipeline_kind} pipeline handle is missing from runtime state"),
        })
}

fn pipeline_lookup_index(function_names: &[String]) -> BTreeMap<String, usize> {
    let mut lookup = BTreeMap::new();
    for (index, function_name) in function_names.iter().enumerate() {
        lookup.entry(function_name.clone()).or_insert(index);
    }
    lookup
}

#[cfg(target_os = "macos")]
fn build_optional_kernel_dispatch_plan(
    optional_pipeline_lookup: &BTreeMap<String, usize>,
) -> MetalOptionalKernelDispatchPlan {
    let index = |name: &str| optional_pipeline_lookup.get(name).copied();
    MetalOptionalKernelDispatchPlan {
        vector_add_f32: index("vector_add_f32"),
        row_scale_f32: index("row_scale_f32"),
        row_vector_scale_f32: index("row_vector_scale_f32"),
        projection_f32: index("decode_logits_projection_f32"),
        projection_f16: index("decode_logits_projection_f16"),
        projection_bf16: index("decode_logits_projection_bf16"),
        batched_projection_f32: index("decode_logits_projection_batched_f32"),
        batched_projection_f16: index("decode_logits_projection_batched_f16"),
        batched_projection_bf16: index("decode_logits_projection_batched_bf16"),
        embedding_gather_f32: index("gather_embedding_rows_f32"),
        embedding_gather_f16: index("gather_embedding_rows_f16"),
        embedding_gather_bf16: index("gather_embedding_rows_bf16"),
        rms_norm_f32: index("rms_norm_f32"),
        rms_norm_f16: index("rms_norm_f16"),
        rms_norm_bf16: index("rms_norm_bf16"),
        batched_rms_norm_f32: index("rms_norm_batched_f32"),
        batched_rms_norm_f16: index("rms_norm_batched_f16"),
        batched_rms_norm_bf16: index("rms_norm_batched_bf16"),
        logits_argmax_f32: index("logits_argmax_f32"),
        logits_argmax_batched_f32: index("logits_argmax_batched_f32"),
        sample_argmax_logprob_f32: index("sample_argmax_logprob_f32"),
        sample_argmax_logprob_batched_f32: index("sample_argmax_logprob_batched_f32"),
        apply_rope_f32: index("apply_rope_f32"),
        apply_rope_batched_f32: index("apply_rope_batched_f32"),
        expand_grouped_kv_heads_f32: index("expand_grouped_kv_heads_f32"),
        ffn_gate_silu_product_f32: index("ffn_gate_silu_product_f32"),
        ffn_gate_gelu_approx_product_f32: index("ffn_gate_gelu_approx_product_f32"),
        linear_attention_conv1d_f32: index("linear_attention_conv1d_f32"),
        linear_attention_conv1d_f16: index("linear_attention_conv1d_f16"),
        linear_attention_conv1d_bf16: index("linear_attention_conv1d_bf16"),
        linear_attention_gate_silu_f32: index("linear_attention_gate_silu_f32"),
        attention_output_gate_sigmoid_product_f32: index(
            "attention_output_gate_sigmoid_product_f32",
        ),
        linear_attention_beta_sigmoid_f32: index("linear_attention_beta_sigmoid_f32"),
        linear_attention_decay_f32: index("linear_attention_decay_f32"),
        linear_gated_delta_step_f32: index("linear_gated_delta_step_f32"),
    }
}

#[cfg(target_os = "macos")]
fn optional_kernel_allowed(
    bringup: &MetalRuntimeBringup,
    kernel_key: &MetalOptionalKernelFeedbackKey,
) -> bool {
    let Ok(feedback) = bringup.state.optional_kernel_feedback.lock() else {
        return true;
    };
    optional_kernel_allowed_in_feedback_state(&feedback, kernel_key)
}

#[cfg(target_os = "macos")]
fn optional_kernel_allowed_in_feedback_state(
    feedback: &MetalOptionalKernelFeedbackState,
    kernel_key: &MetalOptionalKernelFeedbackKey,
) -> bool {
    !feedback.disabled_kernels.contains(kernel_key)
}

#[cfg(target_os = "macos")]
fn record_optional_kernel_result(
    bringup: &MetalRuntimeBringup,
    kernel_key: &MetalOptionalKernelFeedbackKey,
    success: bool,
) {
    let Ok(mut feedback) = bringup.state.optional_kernel_feedback.lock() else {
        return;
    };
    record_optional_kernel_feedback_state(&mut feedback, kernel_key, success);
}

#[cfg(target_os = "macos")]
fn record_optional_kernel_feedback_state(
    feedback: &mut MetalOptionalKernelFeedbackState,
    kernel_key: &MetalOptionalKernelFeedbackKey,
    success: bool,
) {
    if success {
        feedback.consecutive_failures_by_kernel.remove(kernel_key);
        feedback.disabled_kernels.remove(kernel_key);
        return;
    }
    let consecutive_failures = feedback
        .consecutive_failures_by_kernel
        .entry(*kernel_key)
        .or_insert(0);
    *consecutive_failures = consecutive_failures.saturating_add(1);
    if *consecutive_failures >= PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        feedback.disabled_kernels.insert(*kernel_key);
    }
}

#[cfg(target_os = "macos")]
fn command_buffer_status(status: MTLCommandBufferStatus) -> MetalCommandBufferStatus {
    match status {
        MTLCommandBufferStatus::NotEnqueued => MetalCommandBufferStatus::NotEnqueued,
        MTLCommandBufferStatus::Enqueued => MetalCommandBufferStatus::Enqueued,
        MTLCommandBufferStatus::Committed => MetalCommandBufferStatus::Committed,
        MTLCommandBufferStatus::Scheduled => MetalCommandBufferStatus::Scheduled,
        MTLCommandBufferStatus::Completed => MetalCommandBufferStatus::Completed,
        MTLCommandBufferStatus::Error => MetalCommandBufferStatus::Error,
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct CacheDispatchParams {
    element_count: u32,
    head_size: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct AttentionDispatchParams {
    element_count: u32,
    num_seqs: u32,
    head_count: u32,
    head_dim: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct GatherDispatchParams {
    element_count: u32,
    num_seqs: u32,
    block_size_tokens: u32,
    block_table_stride: u32,
    head_size: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct CopyBlockDispatchParams {
    num_pairs: u32,
    numel_per_block_key: u32,
    numel_per_block_value: u32,
    head_size: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct LogitsProjectionDispatchParams {
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedLogitsProjectionDispatchParams {
    token_count: u32,
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
    hidden_stride: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct LogitsArgmaxDispatchParams {
    element_count: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedLogitsArgmaxDispatchParams {
    token_count: u32,
    vocab_rows: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct RmsNormDispatchParams {
    element_count: u32,
    epsilon: f32,
    weight_offset: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedRmsNormDispatchParams {
    head_count: u32,
    head_dim: u32,
    epsilon: f32,
    weight_offset: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct FfnGateProductDispatchParams {
    element_count: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct VectorAddDispatchParams {
    element_count: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedRowScaleDispatchParams {
    row_count: u32,
    row_width: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedRowVectorScaleDispatchParams {
    row_count: u32,
    row_width: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct LinearAttentionConvDispatchParams {
    batch_size: u32,
    conv_dim: u32,
    conv_kernel_dim: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct LinearGatedDeltaDispatchParams {
    batch_size: u32,
    num_key_heads: u32,
    num_value_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    repeat_factor: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct ModelStageRopeDispatchParams {
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    rotary_dim: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct BatchedModelStageRopeDispatchParams {
    token_count: u32,
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    freq_base: f32,
    rotary_dim: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct GroupedKvExpandDispatchParams {
    output_element_count: u32,
    kv_head_count: u32,
    heads_per_kv: u32,
    head_dim: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct EmbeddingGatherDispatchParams {
    token_count: u32,
    embedding_rows: u32,
    hidden_dim: u32,
    scale: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MetalDispatchArenaRequirements {
    block_size_tokens: u32,
    head_size: u32,
    token_capacity: u32,
    slot_capacity: u32,
    attention_ref_capacity: u32,
    gather_ref_capacity: u32,
    gather_output_capacity: u32,
    copy_pair_capacity: u32,
    sequence_capacity: u32,
}

#[cfg(target_os = "macos")]
impl MetalDispatchArenaRequirements {
    fn from_workload(workload: &MetalDispatchWorkload) -> Self {
        Self {
            block_size_tokens: workload.kv_metadata.block_size_tokens.max(1),
            head_size: workload.numeric_layout.head_size().max(1),
            token_capacity: workload
                .token_elements
                .max(workload.kv_metadata.slot_mapping.len() as u32)
                .max(workload.kv_metadata.attention_block_table.len() as u32)
                .max(1),
            slot_capacity: workload.kv_slot_capacity.max(1),
            attention_ref_capacity: workload.kv_metadata.attention_block_table.len() as u32,
            gather_ref_capacity: (workload.kv_metadata.gather_block_table.len() as u32).max(1),
            gather_output_capacity: workload.kv_metadata.gather_token_count(),
            copy_pair_capacity: (workload.kv_metadata.copy_block_mapping.len() as u32).max(1),
            sequence_capacity: (workload.kv_metadata.seq_lens.len() as u32).max(1),
        }
    }

    fn supports(self, required: Self) -> bool {
        self.block_size_tokens == required.block_size_tokens
            && self.head_size == required.head_size
            && self.token_capacity >= required.token_capacity
            && self.slot_capacity >= required.slot_capacity
            && self.attention_ref_capacity >= required.attention_ref_capacity
            && self.gather_ref_capacity >= required.gather_ref_capacity
            && self.gather_output_capacity >= required.gather_output_capacity
            && self.copy_pair_capacity >= required.copy_pair_capacity
            && self.sequence_capacity >= required.sequence_capacity
    }

    fn info(self, reused_existing: bool, grew_existing: bool) -> MetalDispatchArenaInfo {
        MetalDispatchArenaInfo {
            token_capacity: self.token_capacity,
            slot_capacity: self.slot_capacity,
            attention_ref_capacity: self.attention_ref_capacity,
            gather_ref_capacity: self.gather_ref_capacity,
            gather_output_capacity: self.gather_output_capacity,
            copy_pair_capacity: self.copy_pair_capacity,
            sequence_capacity: self.sequence_capacity,
            reused_existing,
            grew_existing,
        }
    }
}

#[cfg(target_os = "macos")]
struct MetalDispatchArena {
    requirements: MetalDispatchArenaRequirements,
    reshape_key: Buffer,
    reshape_value: Buffer,
    key_cache: Buffer,
    value_cache: Buffer,
    reshape_slot_mapping: Buffer,
    attention_query: Buffer,
    attention_output: Buffer,
    kv_block_table: Buffer,
    cu_seq_lens: Buffer,
    scheduled_cu_seq_lens: Buffer,
    kv_key_gathered: Buffer,
    kv_value_gathered: Buffer,
    copy_block_mapping: Buffer,
    copy_key_target: Buffer,
    copy_value_target: Buffer,
}

#[cfg(target_os = "macos")]
impl MetalDispatchArena {
    fn new(device: &Device, requirements: MetalDispatchArenaRequirements) -> Self {
        let key_cache = new_zeroed_shared_buffer::<f32>(
            device,
            requirements
                .slot_capacity
                .saturating_mul(requirements.head_size),
        );
        let value_cache = new_zeroed_shared_buffer::<f32>(
            device,
            requirements
                .slot_capacity
                .saturating_mul(requirements.head_size),
        );

        Self {
            requirements,
            reshape_key: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .token_capacity
                    .saturating_mul(requirements.head_size),
            ),
            reshape_value: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .token_capacity
                    .saturating_mul(requirements.head_size),
            ),
            key_cache,
            value_cache,
            reshape_slot_mapping: new_zeroed_shared_buffer::<u32>(
                device,
                requirements.token_capacity,
            ),
            attention_query: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .token_capacity
                    .saturating_mul(requirements.head_size),
            ),
            attention_output: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .token_capacity
                    .saturating_mul(requirements.head_size),
            ),
            kv_block_table: new_zeroed_shared_buffer::<u32>(
                device,
                requirements.gather_ref_capacity.max(1),
            ),
            cu_seq_lens: new_zeroed_shared_buffer::<u32>(
                device,
                requirements.sequence_capacity + 1,
            ),
            scheduled_cu_seq_lens: new_zeroed_shared_buffer::<u32>(
                device,
                requirements.sequence_capacity + 1,
            ),
            kv_key_gathered: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .gather_output_capacity
                    .max(1)
                    .saturating_mul(requirements.head_size),
            ),
            kv_value_gathered: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .gather_output_capacity
                    .max(1)
                    .saturating_mul(requirements.head_size),
            ),
            copy_block_mapping: new_zeroed_shared_buffer::<[u32; 2]>(
                device,
                requirements.copy_pair_capacity.max(1),
            ),
            copy_key_target: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .slot_capacity
                    .saturating_mul(requirements.head_size),
            ),
            copy_value_target: new_zeroed_shared_buffer::<f32>(
                device,
                requirements
                    .slot_capacity
                    .saturating_mul(requirements.head_size),
            ),
        }
    }

    fn with_preserved_cache(
        device: &Device,
        command_queue: &CommandQueue,
        existing: &MetalDispatchArena,
        requirements: MetalDispatchArenaRequirements,
    ) -> Self {
        let grown = Self::new(device, requirements);
        let preserved_slot_count = existing
            .requirements
            .slot_capacity
            .min(requirements.slot_capacity);
        copy_shared_buffer_prefix::<f32>(
            command_queue,
            &existing.key_cache,
            &grown.key_cache,
            preserved_slot_count.saturating_mul(requirements.head_size),
        );
        copy_shared_buffer_prefix::<f32>(
            command_queue,
            &existing.value_cache,
            &grown.value_cache,
            preserved_slot_count.saturating_mul(requirements.head_size),
        );
        grown
    }

    fn write_workload(
        &mut self,
        device: &Device,
        workload: &MetalDispatchWorkload,
        staged_inputs: &MetalDispatchStagedInputs,
    ) {
        self.reshape_key = new_shared_buffer_with_data(device, &staged_inputs.key);
        self.reshape_value = new_shared_buffer_with_data(device, &staged_inputs.value);
        self.reshape_slot_mapping =
            new_shared_buffer_with_data(device, &workload.kv_metadata.slot_mapping);
        self.attention_query = new_shared_buffer_with_data(device, &staged_inputs.query);
        self.kv_block_table =
            new_shared_buffer_with_data(device, &workload.kv_metadata.gather_block_table);
        self.cu_seq_lens = new_shared_buffer_with_data(device, &workload.kv_metadata.cu_seq_lens);
        self.scheduled_cu_seq_lens =
            new_shared_buffer_with_data(device, &workload.kv_metadata.scheduled_cu_seq_lens);
        self.copy_block_mapping =
            new_shared_buffer_with_data(device, &workload.kv_metadata.copy_block_mapping);
        self.copy_key_target = new_zeroed_shared_buffer::<f32>(
            device,
            self.requirements
                .slot_capacity
                .saturating_mul(self.requirements.head_size),
        );
        self.copy_value_target = new_zeroed_shared_buffer::<f32>(
            device,
            self.requirements
                .slot_capacity
                .saturating_mul(self.requirements.head_size),
        );
    }

    fn write_cache_seed(&mut self, device: &Device, cache_seed: MetalDispatchKvCacheSeed<'_>) {
        self.key_cache = new_shared_buffer_with_data(device, cache_seed.key_cache);
        self.value_cache = new_shared_buffer_with_data(device, cache_seed.value_cache);
    }
}

fn synthetic_staged_inputs(workload: &MetalDispatchWorkload) -> MetalDispatchStagedInputs {
    MetalDispatchStagedInputs {
        key: staged_key_values(workload),
        value: staged_value_values(workload),
        query: staged_query_values(workload),
        layout: MetalDispatchNumericLayout::default(),
        source: MetalStagedInputSource::SyntheticTokenIds,
        prefix_attention_tally: PrefixAttentionExecutionTally::default(),
        #[cfg(target_os = "macos")]
        final_layer_hidden_state_cache: None,
    }
}

#[cfg(target_os = "macos")]
fn resolve_runtime_staged_inputs(
    input: &RunnerInput,
    workload: &MetalDispatchWorkload,
    artifacts: Option<&NativeModelArtifacts>,
    bindings: Option<&MetalNativeModelBindings>,
    buffers: Option<&MetalNativeModelBufferBindings>,
    prefix_layer_caches: Option<&[Mutex<MetalPersistentLayerKvCache>]>,
    linear_request_states: Option<&Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>>,
    bringup: Option<&MetalRuntimeBringup>,
) -> Result<MetalDispatchStagedInputs, MetalRuntimeError> {
    match (artifacts, bindings, buffers) {
        (None, None, None) => Ok(synthetic_staged_inputs(workload)),
        (Some(artifacts), Some(bindings), Some(buffers)) => {
            model_conditioned_staged_inputs(
                artifacts,
                bindings,
                buffers,
                input,
                workload,
                prefix_layer_caches,
                linear_request_states,
                bringup,
            )
            .ok_or_else(|| MetalRuntimeError::InvalidDispatchInput {
                message: "validated native model bindings were present, but staged QKV inputs could not be derived for the scheduled workload".to_string(),
            })
        }
        _ => Err(MetalRuntimeError::InvalidDispatchInput {
            message:
                "native model state is partially initialized; staged inputs require artifacts, bindings, and buffers together"
                    .to_string(),
        }),
    }
}

fn staged_key_values(workload: &MetalDispatchWorkload) -> Vec<f32> {
    stage_numeric_values(&workload.scheduled_token_ids, 0.5, 0.25, 0.03125, 0.0)
}

fn staged_value_values(workload: &MetalDispatchWorkload) -> Vec<f32> {
    stage_numeric_values(&workload.scheduled_token_ids, 1.0, 0.0625, 0.015625, 0.0)
}

fn staged_query_values(workload: &MetalDispatchWorkload) -> Vec<f32> {
    stage_numeric_values(&workload.scheduled_token_ids, 0.5, 0.25, 0.03125, 0.015625)
}

fn stage_numeric_values(
    token_ids: &[u32],
    token_scale: f32,
    head_scale: f32,
    lane_scale: f32,
    base_offset: f32,
) -> Vec<f32> {
    let mut staged = Vec::with_capacity(
        token_ids
            .len()
            .saturating_mul(PHASE1_NUMERIC_HEAD_SIZE as usize),
    );

    for token_id in token_ids {
        for head in 0..PHASE1_NUMERIC_HEAD_COUNT {
            for lane in 0..PHASE1_NUMERIC_HEAD_DIM {
                staged.push(
                    *token_id as f32 * token_scale
                        + head as f32 * head_scale
                        + lane as f32 * lane_scale
                        + base_offset,
                );
            }
        }
    }

    staged
}

#[cfg(target_os = "macos")]
fn model_conditioned_staged_inputs(
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    input: &RunnerInput,
    workload: &MetalDispatchWorkload,
    prefix_layer_caches: Option<&[Mutex<MetalPersistentLayerKvCache>]>,
    linear_request_states: Option<&Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>>,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<MetalDispatchStagedInputs> {
    let (hidden_states, final_layer_index, prefix_attention_tally) =
        model_hidden_states_before_final_layer(
            artifacts,
            bindings,
            buffers,
            input,
            workload,
            bringup,
            prefix_layer_caches,
            linear_request_states,
        )?;
    let final_layer = bindings.layers.get(final_layer_index)?;
    let mut staged = stage_model_layer_qkv_inputs(
        artifacts,
        final_layer,
        buffers,
        workload,
        &hidden_states,
        bringup,
    )?;
    let final_layer_projection_tokens = u32::try_from(hidden_states.len()).unwrap_or(u32::MAX);
    let prefix_attention_tally =
        prefix_attention_tally.record_qkv_projection_tokens(final_layer_projection_tokens);
    staged.source = prefix_attention_tally.staged_input_source();
    staged.prefix_attention_tally = prefix_attention_tally;
    staged.final_layer_hidden_state_cache = Some(ModelFinalLayerHiddenStateCache {
        hidden_states,
        final_layer_index,
    });
    Some(staged)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelStageDims {
    input_dim: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ModelStageRopeStyle {
    None,
    Neox,
    Interleaved,
}

#[cfg(target_os = "macos")]
impl ModelStageDims {
    fn numeric_layout(self) -> Option<MetalDispatchNumericLayout> {
        Some(MetalDispatchNumericLayout::new(
            u32::try_from(self.q_heads).ok()?,
            u32::try_from(self.head_dim).ok()?,
        ))
    }

    fn kv_head_size(self) -> usize {
        self.kv_heads.saturating_mul(self.head_dim)
    }
}

#[cfg(target_os = "macos")]
fn resolved_model_stage_dims_for_input_width(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    attention_norm: &MetalNativeTensorBufferBinding,
    buffers: &MetalNativeModelBufferBindings,
    input_width: usize,
) -> Option<ModelStageDims> {
    let norm_len = tensor_element_count(&attention_norm.meta.spec)?;
    let mut input_dim = (artifacts.manifest().hidden_size as usize)
        .min(input_width)
        .min(norm_len);

    let (q_heads, kv_heads, head_dim) = match layer.attention_qkv.as_ref()? {
        MetalAttentionQkvBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (_, cols) = tensor_matrix_dimensions(&packed.meta.spec)?;
            input_dim = input_dim.min(cols);
            (
                artifacts.manifest().attention_head_count as usize,
                artifacts.manifest().kv_head_count as usize,
                artifacts.manifest().attention_head_dim as usize,
            )
        }
        MetalAttentionQkvBindings::Split { q, k, v, .. } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let (q_rows, q_cols) = tensor_matrix_dimensions(&q_binding.meta.spec)?;
            let (k_rows, k_cols) = tensor_matrix_dimensions(&k_binding.meta.spec)?;
            input_dim = input_dim.min(q_cols).min(k_cols);
            if let Some(v_binding) = v.as_ref().and_then(|binding| buffers.binding_for(binding)) {
                input_dim = input_dim.min(tensor_matrix_dimensions(&v_binding.meta.spec)?.1);
            }

            let head_dim = layer
                .attention_q_norm
                .as_ref()
                .and_then(|binding| buffers.binding_for(binding))
                .and_then(|binding| tensor_element_count(&binding.meta.spec))
                .or_else(|| {
                    layer
                        .attention_k_norm
                        .as_ref()
                        .and_then(|binding| buffers.binding_for(binding))
                        .and_then(|binding| tensor_element_count(&binding.meta.spec))
                })
                .unwrap_or(artifacts.manifest().attention_head_dim as usize);
            // When attn_output_gate is enabled, q_proj encodes both Q and gate,
            // so the effective row count for head derivation is halved.
            let effective_q_rows = if artifacts.manifest().attn_output_gate {
                q_rows / 2
            } else {
                q_rows
            };
            if head_dim == 0
                || !effective_q_rows.is_multiple_of(head_dim)
                || !k_rows.is_multiple_of(head_dim)
            {
                return None;
            }
            let q_heads = effective_q_rows / head_dim;
            let kv_heads = k_rows / head_dim;
            (q_heads, kv_heads, head_dim)
        }
    };
    let rope_head_dim_supported = model_stage_rope_style(artifacts) == ModelStageRopeStyle::None
        || head_dim.is_multiple_of(2);

    (input_dim > 0
        && q_heads > 0
        && kv_heads > 0
        && head_dim > 0
        && q_heads >= kv_heads
        && q_heads.is_multiple_of(kv_heads)
        && rope_head_dim_supported)
        .then_some(ModelStageDims {
            input_dim,
            q_heads,
            kv_heads,
            head_dim,
        })
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn project_attention_qkv(
    artifacts: &NativeModelArtifacts,
    attention_qkv: &MetalAttentionQkvBindings,
    buffers: &MetalNativeModelBufferBindings,
    input: &[f32],
    stage_dims: ModelStageDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    project_attention_qkv_with_dims_and_tally(
        artifacts,
        attention_qkv,
        buffers,
        input,
        stage_dims,
        bringup,
    )
    .map(|(query, key, value, _)| (query, key, value))
}

#[cfg(target_os = "macos")]
fn apply_model_stage_rope_cpu(
    artifacts: &NativeModelArtifacts,
    query: &mut [f32],
    key: &mut [f32],
    position: f32,
    stage_dims: ModelStageDims,
) {
    let rope_style = model_stage_rope_style(artifacts);
    if rope_style == ModelStageRopeStyle::None {
        return;
    }

    let rotary_dim = artifacts.rotary_dim();
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return;
    }

    let query_len = stage_dims.q_heads.saturating_mul(stage_dims.head_dim);
    let key_len = stage_dims.kv_heads.saturating_mul(stage_dims.head_dim);
    if query.len() < query_len || key.len() < key_len {
        return;
    }

    let (cos_table, sin_table) =
        build_model_stage_rope_tables(rotary_dim, position, native_model_rope_theta(artifacts));
    for head in query[..query_len].chunks_exact_mut(stage_dims.head_dim) {
        apply_rope_style_in_place(&mut head[..rotary_dim], &cos_table, &sin_table, rope_style);
    }
    for head in key[..key_len].chunks_exact_mut(stage_dims.head_dim) {
        apply_rope_style_in_place(&mut head[..rotary_dim], &cos_table, &sin_table, rope_style);
    }
}

#[cfg(target_os = "macos")]
fn apply_model_stage_rope_with_path(
    artifacts: &NativeModelArtifacts,
    query: &mut [f32],
    key: &mut [f32],
    position: f32,
    stage_dims: ModelStageDims,
    bringup: Option<&MetalRuntimeBringup>,
) {
    let rope_style = model_stage_rope_style(artifacts);
    if rope_style == ModelStageRopeStyle::None {
        return;
    }

    if let Some((native_query, native_key)) = apply_model_stage_rope_with_optional_native_path(
        bringup,
        query,
        key,
        position,
        stage_dims,
        rope_style,
        native_model_rope_theta(artifacts),
        artifacts.rotary_dim(),
    ) {
        if native_query.len() == query.len() && native_key.len() == key.len() {
            query.copy_from_slice(&native_query);
            key.copy_from_slice(&native_key);
            return;
        }
    }

    apply_model_stage_rope_cpu(artifacts, query, key, position, stage_dims);
}

#[cfg(target_os = "macos")]
fn apply_batched_model_stage_rope_with_path(
    artifacts: &NativeModelArtifacts,
    query_rows: &mut [Vec<f32>],
    key_rows: &mut [Vec<f32>],
    positions: &[u32],
    stage_dims: ModelStageDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<()> {
    if query_rows.len() != key_rows.len() || query_rows.len() != positions.len() {
        return None;
    }

    let rope_style = model_stage_rope_style(artifacts);
    if rope_style == ModelStageRopeStyle::None {
        return Some(());
    }
    let rotary_dim = artifacts.rotary_dim();
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Some(());
    }

    if let Some((native_queries, native_keys)) =
        apply_batched_model_stage_rope_with_optional_native_path(
            bringup,
            query_rows,
            key_rows,
            positions,
            stage_dims,
            rope_style,
            native_model_rope_theta(artifacts),
            rotary_dim,
        )
    {
        for (query_row, native_query) in query_rows.iter_mut().zip(native_queries) {
            if query_row.len() != native_query.len() {
                return None;
            }
            query_row.copy_from_slice(&native_query);
        }
        for (key_row, native_key) in key_rows.iter_mut().zip(native_keys) {
            if key_row.len() != native_key.len() {
                return None;
            }
            key_row.copy_from_slice(&native_key);
        }
        return Some(());
    }
    if query_rows.len() > 1 && batched_rope_split_retry_worthwhile(bringup) {
        let split_index = query_rows.len() / 2;
        let (left_query_rows, right_query_rows) = query_rows.split_at_mut(split_index);
        let (left_key_rows, right_key_rows) = key_rows.split_at_mut(split_index);
        let (left_positions, right_positions) = positions.split_at(split_index);
        apply_batched_model_stage_rope_with_path(
            artifacts,
            left_query_rows,
            left_key_rows,
            left_positions,
            stage_dims,
            bringup,
        )?;
        apply_batched_model_stage_rope_with_path(
            artifacts,
            right_query_rows,
            right_key_rows,
            right_positions,
            stage_dims,
            bringup,
        )?;
        return Some(());
    }

    let single_native_bringup =
        single_rope_retry_worthwhile(bringup, stage_dims, rotary_dim, rope_style)
            .then_some(bringup)
            .flatten();
    for ((query_row, key_row), position) in query_rows
        .iter_mut()
        .zip(key_rows.iter_mut())
        .zip(positions.iter().copied())
    {
        apply_model_stage_rope_with_path(
            artifacts,
            query_row,
            key_row,
            position as f32,
            stage_dims,
            single_native_bringup,
        );
    }

    Some(())
}

#[cfg(target_os = "macos")]
fn model_stage_rope_style(artifacts: &NativeModelArtifacts) -> ModelStageRopeStyle {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("qwen35")
        || family.starts_with("qwen3_5")
        || family.starts_with("qwen3.5")
    {
        ModelStageRopeStyle::Interleaved
    } else if family.starts_with("qwen") {
        ModelStageRopeStyle::Neox
    } else if family.starts_with("gemma") {
        ModelStageRopeStyle::Interleaved
    } else {
        ModelStageRopeStyle::None
    }
}

#[cfg(target_os = "macos")]
fn model_hidden_states_before_final_layer(
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    input: &RunnerInput,
    workload: &MetalDispatchWorkload,
    bringup: Option<&MetalRuntimeBringup>,
    prefix_layer_caches: Option<&[Mutex<MetalPersistentLayerKvCache>]>,
    linear_request_states: Option<&Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>>,
) -> Option<(Vec<Vec<f32>>, usize, PrefixAttentionExecutionTally)> {
    let token_embedding = buffers.binding_for(&bindings.token_embedding)?;
    let (_, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)?;
    let hidden_dim = (artifacts.manifest().hidden_size as usize).min(embedding_cols);
    let final_layer_index = bindings.layers.len().checked_sub(1)?;
    let mut prefix_attention_tally = warm_prefix_reuse_layer_caches_before_final_layer(
        artifacts,
        bindings,
        buffers,
        token_embedding,
        input,
        hidden_dim,
        final_layer_index,
        bringup,
        prefix_layer_caches,
        linear_request_states,
    )?;
    let mut hidden_states = initial_model_hidden_states(
        token_embedding,
        workload,
        hidden_dim,
        native_model_embedding_scale(artifacts),
        bringup,
    )?;

    for (layer_index, layer) in bindings.layers[..final_layer_index].iter().enumerate() {
        let (advanced_hidden_states, layer_tally) = advance_hidden_states_through_model_layer(
            artifacts,
            layer,
            buffers,
            input,
            workload,
            &hidden_states,
            bringup,
            prefix_layer_caches.and_then(|caches| caches.get(layer_index)),
            linear_request_states,
            layer_index as u32,
        )?;
        hidden_states = advanced_hidden_states;
        prefix_attention_tally = prefix_attention_tally.merge(layer_tally);
    }

    Some((hidden_states, final_layer_index, prefix_attention_tally))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn warm_prefix_reuse_layer_caches_before_final_layer(
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    token_embedding: &MetalNativeTensorBufferBinding,
    input: &RunnerInput,
    hidden_dim: usize,
    final_layer_index: usize,
    bringup: Option<&MetalRuntimeBringup>,
    prefix_layer_caches: Option<&[Mutex<MetalPersistentLayerKvCache>]>,
    _linear_request_states: Option<
        &Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>,
    >,
) -> Option<PrefixAttentionExecutionTally> {
    if bringup.is_none() || final_layer_index == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    let Some(prefix_layer_caches) = prefix_layer_caches else {
        return Some(PrefixAttentionExecutionTally::default());
    };

    let mut warmup_tally = PrefixAttentionExecutionTally::default();
    for item in &input.execution_batch.items {
        if item.reused_prefix_token_slice.is_empty() {
            continue;
        }
        let actual_item_input = runner_input_for_execution_item(input, item)?;
        let actual_item_workload =
            MetalDispatchWorkload::from_runner_input(&actual_item_input).ok()?;
        if !prefix_reuse_warmup_needed(
            &actual_item_workload,
            prefix_layer_caches,
            final_layer_index,
        ) {
            continue;
        }

        let warmup_input = runner_input_for_prefix_reuse_warmup(input, item)?;
        let warmup_workload = MetalDispatchWorkload::from_runner_input(&warmup_input).ok()?;
        let mut warmup_hidden_states = initial_model_hidden_states(
            token_embedding,
            &warmup_workload,
            hidden_dim,
            native_model_embedding_scale(artifacts),
            bringup,
        )?;

        for (layer_index, layer) in bindings.layers[..final_layer_index].iter().enumerate() {
            let (advanced_hidden_states, layer_tally) = advance_hidden_states_through_model_layer(
                artifacts,
                layer,
                buffers,
                &warmup_input,
                &warmup_workload,
                &warmup_hidden_states,
                bringup,
                prefix_layer_caches.get(layer_index),
                None,
                layer_index as u32,
            )?;
            warmup_hidden_states = advanced_hidden_states;
            warmup_tally = warmup_tally.merge(layer_tally);
        }
    }

    Some(warmup_tally)
}

#[cfg(target_os = "macos")]
fn prefix_reuse_warmup_needed(
    workload: &MetalDispatchWorkload,
    prefix_layer_caches: &[Mutex<MetalPersistentLayerKvCache>],
    final_layer_index: usize,
) -> bool {
    prefix_layer_caches
        .iter()
        .take(final_layer_index)
        .any(|cache| !workload_supports_native_prefix_attention(workload, Some(cache)))
}

#[cfg(target_os = "macos")]
fn initial_model_hidden_states(
    token_embedding: &MetalNativeTensorBufferBinding,
    workload: &MetalDispatchWorkload,
    hidden_dim: usize,
    embedding_scale: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Vec<f32>>> {
    if let Some(hidden_states) = initial_model_hidden_states_with_optional_native_path(
        token_embedding,
        workload,
        hidden_dim,
        embedding_scale,
        bringup,
    ) {
        return Some(hidden_states);
    }

    initial_model_hidden_states_cpu(token_embedding, workload, hidden_dim, embedding_scale)
}

#[cfg(target_os = "macos")]
fn initial_model_hidden_states_cpu(
    token_embedding: &MetalNativeTensorBufferBinding,
    workload: &MetalDispatchWorkload,
    hidden_dim: usize,
    embedding_scale: f32,
) -> Option<Vec<Vec<f32>>> {
    let (embedding_rows, _) = tensor_matrix_dimensions(&token_embedding.meta.spec)?;
    workload
        .scheduled_token_ids
        .iter()
        .map(|token_id| {
            let token_row = (*token_id as usize) % embedding_rows.max(1);
            let mut hidden = tensor_matrix_row_prefix_f32(token_embedding, token_row, hidden_dim)?;
            if embedding_scale != 1.0 {
                for value in &mut hidden {
                    *value *= embedding_scale;
                }
            }
            round_slice_to_native_dtype(&mut hidden, token_embedding.native_dtype);
            Some(hidden)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn initial_model_hidden_states_with_optional_native_path(
    token_embedding: &MetalNativeTensorBufferBinding,
    workload: &MetalDispatchWorkload,
    hidden_dim: usize,
    embedding_scale: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    let (embedding_rows, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)?;
    if hidden_dim == 0
        || hidden_dim > embedding_cols
        || embedding_rows == 0
        || workload.scheduled_token_ids.is_empty()
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .embedding_gather_kernel(token_embedding.native_dtype)?;
    let token_count = workload.scheduled_token_ids.len();
    let feedback_key =
        embedding_gather_feedback_key(kernel_name, token_count, embedding_rows, hidden_dim);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }
    let output_element_count = token_count.checked_mul(hidden_dim)?;

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let token_ids_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &workload.scheduled_token_ids);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.gather_embedding_rows");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.gather_embedding_rows.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&token_ids_buffer), 0);
            encoder.set_buffer(1, Some(&token_embedding.native_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_embedding_gather_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(workload.scheduled_token_ids.len()),
                saturating_usize_to_u32(embedding_rows),
                saturating_usize_to_u32(hidden_dim),
                embedding_scale,
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let hidden_values = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if hidden_values.len() != output_element_count
                || !hidden_values.iter().all(|value| value.is_finite())
            {
                return None;
            }

            Some(
                hidden_values
                    .chunks_exact(hidden_dim)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn stage_model_layer_qkv_inputs(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    workload: &MetalDispatchWorkload,
    hidden_states: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<MetalDispatchStagedInputs> {
    if hidden_states.len() != workload.scheduled_positions.len() {
        return None;
    }

    let attention_norm = buffers.binding_for(&layer.attention_norm)?;
    let hidden_width = hidden_states.iter().map(Vec::len).min()?;
    let stage_dims = resolved_model_stage_dims_for_input_width(
        artifacts,
        layer,
        attention_norm,
        buffers,
        hidden_width,
    )?;
    let layout = stage_dims.numeric_layout()?;
    let attention_q_norm = match layer.attention_q_norm.as_ref() {
        Some(binding) => Some(buffers.binding_for(binding)?),
        None => None,
    };
    let attention_k_norm = match layer.attention_k_norm.as_ref() {
        Some(binding) => Some(buffers.binding_for(binding)?),
        None => None,
    };
    let token_head_size = layout.head_size() as usize;
    let mut key = Vec::with_capacity(hidden_states.len() * token_head_size);
    let mut value = Vec::with_capacity(hidden_states.len() * token_head_size);
    let mut query = Vec::with_capacity(hidden_states.len() * token_head_size);
    let mut normalized_hidden_states = hidden_states
        .iter()
        .map(|hidden_state| {
            hidden_state
                .get(..stage_dims.input_dim)
                .map(|slice| slice.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut prefix_attention_tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut normalized_hidden_states,
        stage_dims.input_dim,
        attention_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        bringup,
    )?;
    let (mut projected_queries, mut projected_keys, mut projected_values, qkv_tally) =
        project_batched_attention_qkv_with_dims_and_tally(
            artifacts,
            layer.attention_qkv.as_ref()?,
            buffers,
            &normalized_hidden_states,
            stage_dims.input_dim,
            stage_dims,
            bringup,
        )?;
    prefix_attention_tally = prefix_attention_tally.merge(qkv_tally);
    if let Some(binding) = attention_q_norm {
        prefix_attention_tally =
            prefix_attention_tally.merge(apply_batched_per_head_rms_norm_rows_with_tally(
                &mut projected_queries,
                stage_dims.q_heads,
                stage_dims.head_dim,
                binding,
                native_model_rms_norm_epsilon(artifacts),
                native_model_rms_norm_weight_offset(artifacts),
                bringup,
            )?);
    }
    if let Some(binding) = attention_k_norm {
        prefix_attention_tally =
            prefix_attention_tally.merge(apply_batched_per_head_rms_norm_rows_with_tally(
                &mut projected_keys,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                binding,
                native_model_rms_norm_epsilon(artifacts),
                native_model_rms_norm_weight_offset(artifacts),
                bringup,
            )?);
    }
    if layer.attention_v_norm_no_scale {
        prefix_attention_tally = prefix_attention_tally.merge(
            apply_batched_per_head_rms_norm_rows_without_weights_with_tally(
                &mut projected_values,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                native_model_rms_norm_epsilon(artifacts),
                bringup,
            )?,
        );
    }

    apply_batched_model_stage_rope_with_path(
        artifacts,
        &mut projected_queries,
        &mut projected_keys,
        &workload.scheduled_positions,
        stage_dims,
        bringup,
    )?;
    for projected_query in &projected_queries {
        query.extend(projected_query.iter().copied());
    }

    let expanded_keys = expand_batched_grouped_kv_heads_with_path(
        &projected_keys,
        stage_dims.q_heads,
        stage_dims.kv_heads,
        stage_dims.head_dim,
        bringup,
    )?;
    let expanded_values = expand_batched_grouped_kv_heads_with_path(
        &projected_values,
        stage_dims.q_heads,
        stage_dims.kv_heads,
        stage_dims.head_dim,
        bringup,
    )?;
    for expanded_key in expanded_keys {
        key.extend(expanded_key);
    }
    for expanded_value in expanded_values {
        value.extend(expanded_value);
    }

    Some(MetalDispatchStagedInputs {
        key,
        value,
        query,
        layout,
        source: MetalStagedInputSource::ModelConditionedMiniProjection,
        prefix_attention_tally,
        final_layer_hidden_state_cache: None,
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelReferenceLayerDims {
    input_width: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PrefixAttentionExecutionTally {
    native_dispatches: u32,
    cpu_reference_dispatches: u32,
    qkv_projection_tokens: u32,
    layer_continuation_tokens: u32,
    logits_projection_tokens: u32,
    logits_vocab_scan_rows: u32,
    native_projection_rows: u32,
    cpu_projection_rows: u32,
    native_rms_norm_elements: u32,
    cpu_rms_norm_elements: u32,
    native_ffn_activation_elements: u32,
    cpu_ffn_activation_elements: u32,
    native_residual_add_elements: u32,
    cpu_residual_add_elements: u32,
    native_scale_elements: u32,
    cpu_scale_elements: u32,
}

impl PrefixAttentionExecutionTally {
    fn record(mut self, used_native_dispatch: bool) -> Self {
        if used_native_dispatch {
            self.native_dispatches = self.native_dispatches.saturating_add(1);
        } else {
            self.cpu_reference_dispatches = self.cpu_reference_dispatches.saturating_add(1);
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        self.native_dispatches = self
            .native_dispatches
            .saturating_add(other.native_dispatches);
        self.cpu_reference_dispatches = self
            .cpu_reference_dispatches
            .saturating_add(other.cpu_reference_dispatches);
        self.qkv_projection_tokens = self
            .qkv_projection_tokens
            .saturating_add(other.qkv_projection_tokens);
        self.layer_continuation_tokens = self
            .layer_continuation_tokens
            .saturating_add(other.layer_continuation_tokens);
        self.logits_projection_tokens = self
            .logits_projection_tokens
            .saturating_add(other.logits_projection_tokens);
        self.logits_vocab_scan_rows = self
            .logits_vocab_scan_rows
            .saturating_add(other.logits_vocab_scan_rows);
        self.native_projection_rows = self
            .native_projection_rows
            .saturating_add(other.native_projection_rows);
        self.cpu_projection_rows = self
            .cpu_projection_rows
            .saturating_add(other.cpu_projection_rows);
        self.native_rms_norm_elements = self
            .native_rms_norm_elements
            .saturating_add(other.native_rms_norm_elements);
        self.cpu_rms_norm_elements = self
            .cpu_rms_norm_elements
            .saturating_add(other.cpu_rms_norm_elements);
        self.native_ffn_activation_elements = self
            .native_ffn_activation_elements
            .saturating_add(other.native_ffn_activation_elements);
        self.cpu_ffn_activation_elements = self
            .cpu_ffn_activation_elements
            .saturating_add(other.cpu_ffn_activation_elements);
        self.native_residual_add_elements = self
            .native_residual_add_elements
            .saturating_add(other.native_residual_add_elements);
        self.cpu_residual_add_elements = self
            .cpu_residual_add_elements
            .saturating_add(other.cpu_residual_add_elements);
        self.native_scale_elements = self
            .native_scale_elements
            .saturating_add(other.native_scale_elements);
        self.cpu_scale_elements = self
            .cpu_scale_elements
            .saturating_add(other.cpu_scale_elements);
        self
    }

    fn record_qkv_projection_tokens(mut self, token_count: u32) -> Self {
        self.qkv_projection_tokens = self.qkv_projection_tokens.saturating_add(token_count);
        self
    }

    fn record_layer_continuation_tokens(mut self, token_count: u32) -> Self {
        self.layer_continuation_tokens = self.layer_continuation_tokens.saturating_add(token_count);
        self
    }

    fn record_logits_projection(mut self, token_count: u32, vocab_rows: u32) -> Self {
        self.logits_projection_tokens = self.logits_projection_tokens.saturating_add(token_count);
        self.logits_vocab_scan_rows = self.logits_vocab_scan_rows.saturating_add(vocab_rows);
        self
    }

    fn record_projection_rows(mut self, row_count: usize, used_native: bool) -> Self {
        let row_count = saturating_usize_to_u32(row_count);
        if used_native {
            self.native_projection_rows = self.native_projection_rows.saturating_add(row_count);
        } else {
            self.cpu_projection_rows = self.cpu_projection_rows.saturating_add(row_count);
        }
        self
    }

    fn record_rms_norm_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_rms_norm_elements =
                self.native_rms_norm_elements.saturating_add(element_count);
        } else {
            self.cpu_rms_norm_elements = self.cpu_rms_norm_elements.saturating_add(element_count);
        }
        self
    }

    fn record_ffn_activation_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_ffn_activation_elements = self
                .native_ffn_activation_elements
                .saturating_add(element_count);
        } else {
            self.cpu_ffn_activation_elements = self
                .cpu_ffn_activation_elements
                .saturating_add(element_count);
        }
        self
    }

    fn record_residual_add_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_residual_add_elements = self
                .native_residual_add_elements
                .saturating_add(element_count);
        } else {
            self.cpu_residual_add_elements =
                self.cpu_residual_add_elements.saturating_add(element_count);
        }
        self
    }

    fn record_scale_elements(mut self, element_count: usize, used_native: bool) -> Self {
        let element_count = saturating_usize_to_u32(element_count);
        if used_native {
            self.native_scale_elements = self.native_scale_elements.saturating_add(element_count);
        } else {
            self.cpu_scale_elements = self.cpu_scale_elements.saturating_add(element_count);
        }
        self
    }

    fn native_dispatch_count(self) -> u32 {
        self.native_dispatches
    }

    fn cpu_reference_dispatch_count(self) -> u32 {
        self.cpu_reference_dispatches
    }

    fn qkv_projection_token_count(self) -> u32 {
        self.qkv_projection_tokens
    }

    fn layer_continuation_token_count(self) -> u32 {
        self.layer_continuation_tokens
    }

    fn logits_projection_token_count(self) -> u32 {
        self.logits_projection_tokens
    }

    fn logits_vocab_scan_row_count(self) -> u32 {
        self.logits_vocab_scan_rows
    }

    fn native_projection_row_count(self) -> u32 {
        self.native_projection_rows
    }

    fn cpu_projection_row_count(self) -> u32 {
        self.cpu_projection_rows
    }

    fn native_rms_norm_element_count(self) -> u32 {
        self.native_rms_norm_elements
    }

    fn cpu_rms_norm_element_count(self) -> u32 {
        self.cpu_rms_norm_elements
    }

    fn native_ffn_activation_element_count(self) -> u32 {
        self.native_ffn_activation_elements
    }

    fn cpu_ffn_activation_element_count(self) -> u32 {
        self.cpu_ffn_activation_elements
    }

    fn native_residual_add_element_count(self) -> u32 {
        self.native_residual_add_elements
    }

    fn cpu_residual_add_element_count(self) -> u32 {
        self.cpu_residual_add_elements
    }

    fn native_scale_element_count(self) -> u32 {
        self.native_scale_elements
    }

    fn cpu_scale_element_count(self) -> u32 {
        self.cpu_scale_elements
    }

    fn staged_input_source(self) -> MetalStagedInputSource {
        match (
            self.native_dispatches > 0,
            self.cpu_reference_dispatches > 0,
        ) {
            (false, false) => MetalStagedInputSource::ModelConditionedMiniProjection,
            (false, true) => MetalStagedInputSource::ModelConditionedCpuPrefixAttention,
            (true, false) => MetalStagedInputSource::ModelConditionedNativePrefixAttention,
            (true, true) => MetalStagedInputSource::ModelConditionedMixedPrefixAttention,
        }
    }
}

#[cfg(target_os = "macos")]
fn resolved_model_reference_layer_dims(
    attention_o: &MetalNativeTensorBufferBinding,
    ffn_norm: &MetalNativeTensorBufferBinding,
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    ffn_down: &MetalNativeTensorBufferBinding,
    hidden_width: usize,
    attention_output_width: usize,
) -> Option<ModelReferenceLayerDims> {
    let (attention_o_rows, attention_o_cols) = tensor_matrix_dimensions(&attention_o.meta.spec)?;
    let ffn_norm_len = tensor_element_count(&ffn_norm.meta.spec)?;
    let ffn_gate_up_input_cols = ffn_gate_up_input_cols(ffn_gate_up, buffers)?;
    let (ffn_down_rows, ffn_down_cols) = tensor_matrix_dimensions(&ffn_down.meta.spec)?;
    let input_width = attention_output_width.min(attention_o_cols);
    let hidden_dim = hidden_width
        .min(attention_o_rows)
        .min(ffn_norm_len)
        .min(ffn_gate_up_input_cols)
        .min(ffn_down_rows);
    let intermediate_dim = resolved_ffn_intermediate_dim(ffn_gate_up, buffers, ffn_down_cols)?;

    (input_width > 0 && hidden_dim > 0 && intermediate_dim > 0).then_some(ModelReferenceLayerDims {
        input_width,
        hidden_dim,
        intermediate_dim,
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn advance_hidden_states_through_model_layer(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    input: &RunnerInput,
    workload: &MetalDispatchWorkload,
    hidden_states: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
    linear_request_states: Option<&Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>>,
    layer_index: u32,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    if input.execution_batch.items.is_empty() {
        return Some((Vec::new(), PrefixAttentionExecutionTally::default()));
    }

    // Layers without attention tensors (e.g. Qwen3.5 linear_attention) skip the
    // attention block entirely and only run FFN. The hidden states pass through
    // unchanged for the attention residual (equivalent to zero attention output).
    let has_attention = layer.attention_qkv.is_some() && layer.attention_o.is_some();
    if let Some(linear_attention) = &layer.linear_attention {
        return advance_hidden_states_through_linear_attention_layer(
            artifacts,
            linear_attention,
            layer,
            buffers,
            input,
            workload,
            hidden_states,
            bringup,
            linear_request_states,
            layer_index,
        );
    }
    if !has_attention {
        return advance_hidden_states_ffn_only(artifacts, layer, buffers, hidden_states, bringup);
    }

    let ephemeral_layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());
    let shared_layer_cache = layer_cache.or(Some(&ephemeral_layer_cache));
    let attention_o = buffers.binding_for(layer.attention_o.as_ref()?)?;
    let staged_inputs =
        stage_model_layer_qkv_inputs(artifacts, layer, buffers, workload, hidden_states, bringup)?;
    let item_token_ranges = execution_item_token_ranges(input)?;
    let attention_head_size = staged_inputs.layout.head_size() as usize;
    let mut prefix_attention_tally = staged_inputs
        .prefix_attention_tally
        .record_qkv_projection_tokens(u32::try_from(hidden_states.len()).unwrap_or(u32::MAX));
    let (attention_output, attention_dispatch_tally) =
        collect_prefix_attention_outputs_with_item_fallback(
            artifacts,
            input,
            &staged_inputs,
            &item_token_ranges,
            0..input.execution_batch.items.len(),
            bringup,
            shared_layer_cache,
        )?;
    prefix_attention_tally = prefix_attention_tally.merge(attention_dispatch_tally);
    let token_cursor = item_token_ranges.last().map(|range| range.end).unwrap_or(0);

    if token_cursor != hidden_states.len() || token_cursor != workload.scheduled_positions.len() {
        return None;
    }

    // When attn_output_gate is enabled, compute the gate from the second half
    // of q_proj by re-projecting the normalized hidden states.
    let attn_gate = if artifacts.manifest().attn_output_gate {
        compute_attn_output_gate(artifacts, layer, buffers, hidden_states, bringup)
    } else {
        None
    };

    let (next_hidden_states, continuation_tally) =
        project_hidden_states_from_layer_attention_output(
            artifacts,
            layer,
            buffers,
            attention_o,
            hidden_states,
            &attention_output,
            attention_head_size,
            attn_gate.as_deref(),
            bringup,
        )?;
    let continuation_tokens = u32::try_from(next_hidden_states.len()).unwrap_or(u32::MAX);
    prefix_attention_tally = prefix_attention_tally
        .merge(continuation_tally)
        .record_layer_continuation_tokens(continuation_tokens);

    Some((next_hidden_states, prefix_attention_tally))
}

/// Compute the attention output gate by projecting normalized hidden states
/// through the second half of q_proj. Returns a flat buffer of gate values
/// (one `q_heads * head_dim` gate vector per token).
#[cfg(target_os = "macos")]
fn compute_attn_output_gate(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    hidden_states: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<f32>> {
    let attention_norm = buffers.binding_for(&layer.attention_norm)?;
    let stage_dims = resolved_model_stage_dims_for_input_width(
        artifacts,
        layer,
        attention_norm,
        buffers,
        hidden_states.iter().map(Vec::len).min()?,
    )?;
    let gate_row_offset = stage_dims.q_heads * stage_dims.head_dim;
    let gate_output_dim = gate_row_offset; // gate has same dim as Q

    if hidden_states.len() == 1 {
        let attention_norm_bringup =
            single_rms_norm_retry_worthwhile(bringup, attention_norm, stage_dims.input_dim)
                .then_some(bringup)
                .flatten();
        let mut normalized = hidden_states
            .first()?
            .get(..stage_dims.input_dim)
            .map(|hidden| hidden.to_vec())?;
        apply_rms_norm_with_binding_in_place_with_path(
            &mut normalized,
            attention_norm,
            native_model_rms_norm_epsilon(artifacts),
            native_model_rms_norm_weight_offset(artifacts),
            attention_norm_bringup,
        )?;
        return match layer.attention_qkv.as_ref()? {
            MetalAttentionQkvBindings::Split { q, .. } => {
                let q_binding = buffers.binding_for(q)?;
                let q_source_head_dim = if artifacts.manifest().attn_output_gate {
                    stage_dims.head_dim.checked_mul(2)?
                } else {
                    stage_dims.head_dim
                };
                let split_gate_row_offset = if artifacts.manifest().attn_output_gate {
                    stage_dims.head_dim
                } else {
                    gate_row_offset
                };
                let projection_bringup = single_projection_retry_worthwhile(
                    bringup,
                    q_binding,
                    stage_dims.head_dim,
                    stage_dims.input_dim,
                )
                .then_some(bringup)
                .flatten();
                project_matrix_head_prefix(
                    q_binding,
                    split_gate_row_offset,
                    stage_dims.q_heads,
                    stage_dims.head_dim,
                    q_source_head_dim,
                    &normalized,
                    projection_bringup,
                )
            }
            MetalAttentionQkvBindings::Packed(binding) => {
                let packed = buffers.binding_for(binding)?;
                let projection_bringup = single_projection_retry_worthwhile(
                    bringup,
                    packed,
                    gate_output_dim,
                    stage_dims.input_dim,
                )
                .then_some(bringup)
                .flatten();
                project_matrix_rows(
                    packed,
                    gate_row_offset,
                    gate_output_dim,
                    &normalized,
                    projection_bringup,
                )
            }
        };
    }

    let mut normalized = hidden_states
        .iter()
        .map(|h| h.get(..stage_dims.input_dim).map(|s| s.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut normalized,
        stage_dims.input_dim,
        attention_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        bringup,
    )?;

    let (gate_rows, _) = match layer.attention_qkv.as_ref()? {
        MetalAttentionQkvBindings::Split { q, .. } => {
            let q_binding = buffers.binding_for(q)?;
            let q_source_head_dim = if artifacts.manifest().attn_output_gate {
                stage_dims.head_dim.checked_mul(2)?
            } else {
                stage_dims.head_dim
            };
            let split_gate_row_offset = if artifacts.manifest().attn_output_gate {
                stage_dims.head_dim
            } else {
                gate_row_offset
            };
            project_batched_matrix_head_prefix_with_tally(
                q_binding,
                split_gate_row_offset,
                stage_dims.q_heads,
                stage_dims.head_dim,
                q_source_head_dim,
                &normalized,
                stage_dims.input_dim,
                bringup,
            )?
        }
        MetalAttentionQkvBindings::Packed(binding) => {
            let (gate_rows, gate_tally) = project_batched_matrix_rows_with_tally(
                buffers.binding_for(binding)?,
                gate_row_offset,
                gate_output_dim,
                &normalized,
                stage_dims.input_dim,
                bringup,
            )?;
            (
                gate_rows,
                prefix_attention_tally_from_native_dense_tally(gate_tally),
            )
        }
    };

    let mut gate_flat = Vec::with_capacity(gate_rows.len() * gate_output_dim);
    for row in &gate_rows {
        gate_flat.extend_from_slice(row.get(..gate_output_dim)?);
    }
    Some(gate_flat)
}

/// FFN-only path for layers without attention (e.g. Qwen3.5 linear_attention).
/// Applies: FFN norm → gate/up projection → activation → down projection → residual add.
#[cfg(target_os = "macos")]
fn advance_hidden_states_ffn_only(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    hidden_states: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    let (next_hidden_states, dense_tally, _) =
        apply_ffn_continuation_rows_with_tally(artifacts, layer, buffers, hidden_states, bringup)?;
    let mut tally = prefix_attention_tally_from_native_dense_tally(dense_tally);
    let continuation_tokens = u32::try_from(next_hidden_states.len()).unwrap_or(u32::MAX);
    tally = tally.record_layer_continuation_tokens(continuation_tokens);
    Some((next_hidden_states, tally))
}

/// Resolve FFN dimensions without requiring attention_o binding.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn resolved_ffn_only_layer_dims(
    ffn_norm: &MetalNativeTensorBufferBinding,
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    ffn_down: &MetalNativeTensorBufferBinding,
    hidden_width: usize,
) -> Option<(usize, usize)> {
    let ffn_norm_len = tensor_element_count(&ffn_norm.meta.spec)?;
    let ffn_gate_up_input_cols = ffn_gate_up_input_cols(ffn_gate_up, buffers)?;
    let (ffn_down_rows, ffn_down_cols) = tensor_matrix_dimensions(&ffn_down.meta.spec)?;
    let hidden_dim = hidden_width
        .min(ffn_norm_len)
        .min(ffn_gate_up_input_cols)
        .min(ffn_down_rows);
    let intermediate_dim = resolved_ffn_intermediate_dim(ffn_gate_up, buffers, ffn_down_cols)?;
    (hidden_dim > 0 && intermediate_dim > 0).then_some((hidden_dim, intermediate_dim))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ResolvedLinearAttentionDims {
    hidden_dim: usize,
    num_value_heads: usize,
    num_key_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    repeat_factor: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
struct ProjectedLinearAttentionItem {
    request_id: crate::ids::RequestId,
    token_count: usize,
    qkv_rows: Vec<Vec<f32>>,
    z_rows: Vec<Vec<f32>>,
    a_rows: Vec<Vec<f32>>,
    b_rows: Vec<Vec<f32>>,
    residual_rows: Vec<Vec<f32>>,
    state_before: Vec<f32>,
    pre_recurrent_tally: PrefixAttentionExecutionTally,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq)]
struct PreparedLinearAttentionItem {
    request_id: crate::ids::RequestId,
    token_count: usize,
    q_rows: Vec<Vec<f32>>,
    k_rows: Vec<Vec<f32>>,
    v_rows: Vec<Vec<f32>>,
    g_rows: Vec<Vec<f32>>,
    beta_rows: Vec<Vec<f32>>,
    z_rows: Vec<Vec<f32>>,
    residual_rows: Vec<Vec<f32>>,
    state_before: Vec<f32>,
    pre_recurrent_tally: PrefixAttentionExecutionTally,
}

#[cfg(target_os = "macos")]
fn resolved_linear_attention_dims(
    artifacts: &NativeModelArtifacts,
    bindings: &MetalLinearAttentionBindings,
    buffers: &MetalNativeModelBufferBindings,
    hidden_width: usize,
) -> Option<ResolvedLinearAttentionDims> {
    let config = artifacts.linear_attention_config()?;
    let num_value_heads = usize::try_from(config.num_value_heads?).ok()?;
    let num_key_heads = usize::try_from(config.num_key_heads?).ok()?;
    let key_head_dim = usize::try_from(config.key_head_dim?).ok()?;
    let value_head_dim = usize::try_from(config.value_head_dim?).ok()?;
    let conv_kernel_dim = usize::try_from(config.conv_kernel_dim?).ok()?;
    let key_dim = num_key_heads.checked_mul(key_head_dim)?;
    let value_dim = num_value_heads.checked_mul(value_head_dim)?;
    let conv_dim = key_dim.checked_mul(2)?.checked_add(value_dim)?;
    let repeat_factor = num_value_heads.checked_div(num_key_heads)?;
    if repeat_factor == 0 || !num_value_heads.is_multiple_of(num_key_heads) {
        return None;
    }

    let in_proj_qkv = buffers.binding_for(&bindings.in_proj_qkv)?;
    let in_proj_z = buffers.binding_for(&bindings.in_proj_z)?;
    let in_proj_a = buffers.binding_for(&bindings.in_proj_a)?;
    let in_proj_b = buffers.binding_for(&bindings.in_proj_b)?;
    let out_proj = buffers.binding_for(&bindings.out_proj)?;
    let (_qkv_rows, qkv_cols) = tensor_matrix_dimensions(&in_proj_qkv.meta.spec)?;
    let (_z_rows, z_cols) = tensor_matrix_dimensions(&in_proj_z.meta.spec)?;
    let (_a_rows, a_cols) = tensor_matrix_dimensions(&in_proj_a.meta.spec)?;
    let (_b_rows, b_cols) = tensor_matrix_dimensions(&in_proj_b.meta.spec)?;
    let (_out_rows, out_cols) = tensor_matrix_dimensions(&out_proj.meta.spec)?;
    let hidden_dim = hidden_width
        .min(qkv_cols)
        .min(z_cols)
        .min(a_cols)
        .min(b_cols);
    if hidden_dim == 0 || out_cols < value_dim {
        return None;
    }

    Some(ResolvedLinearAttentionDims {
        hidden_dim,
        num_value_heads,
        num_key_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel_dim,
        key_dim,
        value_dim,
        conv_dim,
        repeat_factor,
    })
}

#[cfg(target_os = "macos")]
fn advance_hidden_states_through_linear_attention_layer(
    artifacts: &NativeModelArtifacts,
    linear_attention: &MetalLinearAttentionBindings,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    input: &RunnerInput,
    _workload: &MetalDispatchWorkload,
    hidden_states: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
    linear_request_states: Option<&Mutex<BTreeMap<crate::ids::RequestId, MetalLinearRequestState>>>,
    layer_index: u32,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    if input.execution_batch.items.is_empty() {
        return Some((Vec::new(), PrefixAttentionExecutionTally::default()));
    }

    let attention_norm = buffers.binding_for(&layer.attention_norm)?;
    let in_proj_qkv = buffers.binding_for(&linear_attention.in_proj_qkv)?;
    let in_proj_z = buffers.binding_for(&linear_attention.in_proj_z)?;
    let in_proj_a = buffers.binding_for(&linear_attention.in_proj_a)?;
    let in_proj_b = buffers.binding_for(&linear_attention.in_proj_b)?;
    let conv1d = buffers.binding_for(&linear_attention.conv1d)?;
    let dt_bias = tensor_prefix_f32(
        buffers.binding_for(&linear_attention.dt_bias)?,
        usize::try_from(artifacts.linear_attention_config()?.num_value_heads?).ok()?,
    )?;
    let a_log = tensor_prefix_f32(
        buffers.binding_for(&linear_attention.a_log)?,
        usize::try_from(artifacts.linear_attention_config()?.num_value_heads?).ok()?,
    )?;
    let linear_norm = buffers.binding_for(&linear_attention.norm)?;
    let out_proj = buffers.binding_for(&linear_attention.out_proj)?;
    let item_token_ranges = execution_item_token_ranges(input)?;
    let hidden_width = hidden_states.iter().map(Vec::len).min()?;
    let dims = resolved_linear_attention_dims(artifacts, linear_attention, buffers, hidden_width)?;
    let ephemeral_request_states = Mutex::new(BTreeMap::new());
    let request_states = linear_request_states.unwrap_or(&ephemeral_request_states);
    let mut request_states = request_states
        .lock()
        .expect("linear attention request state mutex should not be poisoned");
    let mut next_hidden_states = Vec::with_capacity(hidden_states.len());
    let mut tally = PrefixAttentionExecutionTally::default();
    for segment in linear_attention_unique_request_segments(&input.execution_batch.items) {
        let (mut segment_hidden_states, segment_tally) =
            advance_hidden_states_through_linear_attention_segment(
                artifacts,
                layer,
                buffers,
                hidden_states,
                bringup,
                &mut request_states,
                &input.execution_batch.items[segment.clone()],
                layer_index,
                &item_token_ranges[segment],
                dims,
                attention_norm,
                in_proj_qkv,
                in_proj_z,
                in_proj_a,
                in_proj_b,
                conv1d,
                &dt_bias,
                &a_log,
                linear_norm,
                out_proj,
            )?;
        tally = tally.merge(segment_tally);
        next_hidden_states.append(&mut segment_hidden_states);
    }

    Some((next_hidden_states, tally))
}

#[cfg(target_os = "macos")]
fn linear_attention_unique_request_segments(items: &[ExecutionItem]) -> Vec<Range<usize>> {
    if items.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut segment_start = 0_usize;
    let mut seen = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        if !seen.insert(item.request_id) {
            segments.push(segment_start..index);
            segment_start = index;
            seen.clear();
            seen.insert(item.request_id);
        }
    }
    segments.push(segment_start..items.len());
    segments
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn advance_hidden_states_through_linear_attention_segment(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    hidden_states: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
    request_states: &mut BTreeMap<crate::ids::RequestId, MetalLinearRequestState>,
    items: &[ExecutionItem],
    layer_index: u32,
    item_token_ranges: &[Range<usize>],
    dims: ResolvedLinearAttentionDims,
    attention_norm: &MetalNativeTensorBufferBinding,
    in_proj_qkv: &MetalNativeTensorBufferBinding,
    in_proj_z: &MetalNativeTensorBufferBinding,
    in_proj_a: &MetalNativeTensorBufferBinding,
    in_proj_b: &MetalNativeTensorBufferBinding,
    conv1d: &MetalNativeTensorBufferBinding,
    dt_bias: &[f32],
    a_log: &[f32],
    linear_norm: &MetalNativeTensorBufferBinding,
    out_proj: &MetalNativeTensorBufferBinding,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    if items.len() != item_token_ranges.len() {
        return None;
    }

    let mut projected_items = Vec::with_capacity(items.len());
    for (item, token_range) in items.iter().zip(item_token_ranges.iter()) {
        let item_hidden_states = hidden_states.get(token_range.clone())?;
        let request_state = request_states.entry(item.request_id).or_default();
        let layer_state = request_state.layers.entry(layer_index).or_default();
        let projected_item = match project_linear_attention_item(
            artifacts,
            item.request_id,
            item.position_range.start as usize,
            item_hidden_states,
            layer_state,
            attention_norm,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            dims,
            bringup,
        ) {
            Some(projected_item) => projected_item,
            None => {
                tracing::debug!(
                    request_id = item.request_id.0,
                    layer_index,
                    position_start = item.position_range.start,
                    position_end = item.position_range.end_exclusive,
                    item_token_count = item.scheduled_token_count,
                    hidden_state_rows = item_hidden_states.len(),
                    processed_tokens = layer_state.processed_tokens,
                    "linear attention staged projection failed"
                );
                return None;
            }
        };
        projected_items.push(projected_item);
    }

    let batched_conv_outputs = apply_batched_linear_attention_decode_conv_with_optional_native_path(
        &projected_items,
        request_states,
        layer_index,
        conv1d,
        dims,
        bringup,
    );
    let mut prepared_items = Vec::with_capacity(projected_items.len());
    for (index, projected_item) in projected_items.into_iter().enumerate() {
        let request_state = request_states.get_mut(&projected_item.request_id)?;
        let layer_state = request_state.layers.get_mut(&layer_index)?;
        let (conv_output_rows, next_conv_state, conv_tally) = batched_conv_outputs
            .as_ref()
            .and_then(|outputs| outputs.get(index))
            .cloned()
            .flatten()
            .or_else(|| {
                apply_linear_attention_conv_rows_with_path(
                    conv1d,
                    &projected_item.qkv_rows,
                    &mut layer_state.conv_state,
                    dims,
                    bringup,
                )
                .map(|(rows, tally)| (rows, layer_state.conv_state.clone(), tally))
            })
            .or_else(|| {
                tracing::debug!(
                    request_id = projected_item.request_id.0,
                    layer_index,
                    token_count = projected_item.token_count,
                    conv_dim = dims.conv_dim,
                    "linear attention conv path failed"
                );
                None
            })?;
        layer_state.conv_state = next_conv_state;
        let prepared_item = finalize_projected_linear_attention_item(
            artifacts,
            projected_item,
            conv_output_rows,
            conv_tally,
            dt_bias,
            a_log,
            dims,
            bringup,
        )?;
        prepared_items.push(prepared_item);
    }

    let batched_recurrent_outputs =
        apply_batched_linear_attention_decode_recurrent_with_optional_native_path(
            &prepared_items,
            dims,
            bringup,
        );
    let mut next_hidden_states = Vec::with_capacity(hidden_states.len());
    let mut tally = PrefixAttentionExecutionTally::default();

    for (index, prepared_item) in prepared_items.into_iter().enumerate() {
        let request_state = request_states.get_mut(&prepared_item.request_id)?;
        let layer_state = request_state.layers.get_mut(&layer_index)?;
        let (linear_outputs, next_state, recurrent_tally) = batched_recurrent_outputs
            .as_ref()
            .and_then(|outputs| outputs.get(index))
            .cloned()
            .flatten()
            .or_else(|| {
                apply_linear_attention_recurrent_with_state(
                    &prepared_item.q_rows,
                    &prepared_item.k_rows,
                    &prepared_item.v_rows,
                    &prepared_item.g_rows,
                    &prepared_item.beta_rows,
                    &mut layer_state.ssm_state,
                    dims,
                    bringup,
                )
                .map(|(outputs, tally)| (outputs, layer_state.ssm_state.clone(), tally))
            })
            .or_else(|| {
                tracing::debug!(
                    request_id = prepared_item.request_id.0,
                    layer_index,
                    token_count = prepared_item.token_count,
                    key_head_dim = dims.key_head_dim,
                    value_head_dim = dims.value_head_dim,
                    "linear attention recurrent path failed"
                );
                None
            })?;
        layer_state.ssm_state = next_state;
        let (mut item_hidden_rows, item_tally) = finish_linear_attention_item(
            artifacts,
            prepared_item,
            linear_outputs,
            recurrent_tally,
            linear_norm,
            out_proj,
            layer,
            buffers,
            layer_state,
            dims,
            bringup,
        )?;
        tally = tally.merge(item_tally);
        next_hidden_states.append(&mut item_hidden_rows);
    }

    let continuation_tokens = u32::try_from(next_hidden_states.len()).unwrap_or(u32::MAX);
    tally = tally.record_layer_continuation_tokens(continuation_tokens);
    Some((next_hidden_states, tally))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn project_linear_attention_item(
    artifacts: &NativeModelArtifacts,
    request_id: crate::ids::RequestId,
    position_start: usize,
    hidden_states: &[Vec<f32>],
    layer_state: &mut MetalLinearLayerState,
    attention_norm: &MetalNativeTensorBufferBinding,
    in_proj_qkv: &MetalNativeTensorBufferBinding,
    in_proj_z: &MetalNativeTensorBufferBinding,
    in_proj_a: &MetalNativeTensorBufferBinding,
    in_proj_b: &MetalNativeTensorBufferBinding,
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<ProjectedLinearAttentionItem> {
    if position_start < layer_state.processed_tokens {
        if position_start == 0 {
            reset_linear_layer_state(layer_state, dims);
        } else {
            return None;
        }
    }
    if position_start != layer_state.processed_tokens {
        return None;
    }
    ensure_linear_attention_conv_state_shape(&mut layer_state.conv_state, dims)?;

    let mut normalized_hidden_states = hidden_states
        .iter()
        .map(|hidden_state| {
            hidden_state
                .get(..dims.hidden_dim)
                .map(|slice| slice.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut normalized_hidden_states,
        dims.hidden_dim,
        attention_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        bringup,
    )?;

    // Try multi-projection batch: qkv, z, a, b in a single command buffer.
    let (qkv_rows, z_rows, a_rows, b_rows, multi_tally) = if let Some(result) =
        project_linear_attention_item_multi_dispatch(
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            &normalized_hidden_states,
            dims,
            bringup,
        ) {
        result
    } else {
        // Fallback: individual dispatches.
        let (qkv_rows, qkv_tally) = project_batched_matrix_rows_with_tally(
            in_proj_qkv,
            0,
            dims.conv_dim,
            &normalized_hidden_states,
            dims.hidden_dim,
            bringup,
        )?;
        let (z_rows, z_tally) = project_batched_matrix_rows_with_tally(
            in_proj_z,
            0,
            dims.value_dim,
            &normalized_hidden_states,
            dims.hidden_dim,
            bringup,
        )?;
        let (a_rows, a_tally) = project_batched_matrix_rows_with_tally(
            in_proj_a,
            0,
            dims.num_value_heads,
            &normalized_hidden_states,
            dims.hidden_dim,
            bringup,
        )?;
        let (b_rows, b_tally) = project_batched_matrix_rows_with_tally(
            in_proj_b,
            0,
            dims.num_value_heads,
            &normalized_hidden_states,
            dims.hidden_dim,
            bringup,
        )?;
        let combined_tally = prefix_attention_tally_from_native_dense_tally(qkv_tally)
            .merge(prefix_attention_tally_from_native_dense_tally(z_tally))
            .merge(prefix_attention_tally_from_native_dense_tally(a_tally))
            .merge(prefix_attention_tally_from_native_dense_tally(b_tally));
        (qkv_rows, z_rows, a_rows, b_rows, combined_tally)
    };
    tally = tally.merge(multi_tally);

    let residual_rows = hidden_states
        .iter()
        .map(|hidden_state| {
            hidden_state
                .get(..dims.hidden_dim)
                .map(|slice| slice.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ProjectedLinearAttentionItem {
        request_id,
        token_count: hidden_states.len(),
        qkv_rows,
        z_rows,
        a_rows,
        b_rows,
        residual_rows,
        state_before: layer_state.ssm_state.clone(),
        pre_recurrent_tally: tally,
    })
}

/// Attempts to dispatch linear attention projections (qkv, z, a, b) in a single
/// Metal command buffer. Returns `None` to signal the caller should fall back to
/// individual dispatches.
#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
fn project_linear_attention_item_multi_dispatch(
    in_proj_qkv: &MetalNativeTensorBufferBinding,
    in_proj_z: &MetalNativeTensorBufferBinding,
    in_proj_a: &MetalNativeTensorBufferBinding,
    in_proj_b: &MetalNativeTensorBufferBinding,
    normalized_hidden_states: &[Vec<f32>],
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    PrefixAttentionExecutionTally,
)> {
    let bringup = bringup?;
    let tasks = [
        MultiProjectionTask {
            binding: in_proj_qkv,
            row_offset: 0,
            output_dim: dims.conv_dim,
        },
        MultiProjectionTask {
            binding: in_proj_z,
            row_offset: 0,
            output_dim: dims.value_dim,
        },
        MultiProjectionTask {
            binding: in_proj_a,
            row_offset: 0,
            output_dim: dims.num_value_heads,
        },
        MultiProjectionTask {
            binding: in_proj_b,
            row_offset: 0,
            output_dim: dims.num_value_heads,
        },
    ];

    let total_projection_rows = dims
        .conv_dim
        .checked_add(dims.value_dim)?
        .checked_add(dims.num_value_heads)?
        .checked_add(dims.num_value_heads)?;

    let outputs = if normalized_hidden_states.len() == 1 {
        let input = normalized_hidden_states.first()?.get(..dims.hidden_dim)?;
        let single_outputs = project_multi_matrix_rows_with_optional_native_path(
            bringup,
            &tasks,
            input,
            dims.hidden_dim,
        )?;
        single_outputs
            .into_iter()
            .map(|v| vec![v])
            .collect::<Vec<_>>()
    } else {
        project_multi_batched_matrix_rows_with_optional_native_path(
            bringup,
            &tasks,
            normalized_hidden_states,
            dims.hidden_dim,
        )?
    };

    if outputs.len() != 4 {
        return None;
    }
    let mut iter = outputs.into_iter();
    let qkv_rows = iter.next()?;
    let z_rows = iter.next()?;
    let a_rows = iter.next()?;
    let b_rows = iter.next()?;

    let total_rows = normalized_hidden_states
        .len()
        .checked_mul(total_projection_rows)?;
    let tally = prefix_attention_tally_from_native_dense_tally(
        DirectDecodeNativeDenseTally::default().record_projection_rows(total_rows, true),
    );
    Some((qkv_rows, z_rows, a_rows, b_rows, tally))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn finalize_projected_linear_attention_item(
    artifacts: &NativeModelArtifacts,
    projected_item: ProjectedLinearAttentionItem,
    conv_output_rows: Vec<Vec<f32>>,
    conv_tally: PrefixAttentionExecutionTally,
    dt_bias: &[f32],
    a_log: &[f32],
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PreparedLinearAttentionItem> {
    if conv_output_rows.len() != projected_item.token_count
        || conv_output_rows
            .iter()
            .any(|row| row.len() != dims.conv_dim)
    {
        return None;
    }
    let mut tally = projected_item.pre_recurrent_tally.merge(conv_tally);
    let mut q_rows = conv_output_rows
        .iter()
        .map(|row| row.get(..dims.key_dim).map(|slice| slice.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    let mut k_rows = conv_output_rows
        .iter()
        .map(|row| {
            row.get(dims.key_dim..dims.key_dim * 2)
                .map(|slice| slice.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    let v_rows = conv_output_rows
        .into_iter()
        .map(|row| {
            row.get(dims.key_dim * 2..dims.conv_dim)
                .map(|slice| slice.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    tally = tally.merge(
        apply_batched_per_head_rms_norm_rows_without_weights_with_tally(
            &mut q_rows,
            dims.num_key_heads,
            dims.key_head_dim,
            native_model_rms_norm_epsilon(artifacts),
            bringup,
        )?,
    );
    tally = tally.merge(
        apply_batched_per_head_rms_norm_rows_without_weights_with_tally(
            &mut k_rows,
            dims.num_key_heads,
            dims.key_head_dim,
            native_model_rms_norm_epsilon(artifacts),
            bringup,
        )?,
    );
    apply_linear_attention_query_key_scaling(&mut q_rows, &mut k_rows, dims)?;
    let (g_rows, decay_tally) = compute_linear_attention_decay_rows_with_tally(
        &projected_item.a_rows,
        a_log,
        dt_bias,
        bringup,
    )?;
    tally = tally.merge(decay_tally);
    let (beta_rows, beta_tally) =
        compute_linear_attention_beta_rows_with_tally(&projected_item.b_rows, bringup)?;
    tally = tally.merge(beta_tally);
    Some(PreparedLinearAttentionItem {
        request_id: projected_item.request_id,
        token_count: projected_item.token_count,
        q_rows,
        k_rows,
        v_rows,
        g_rows,
        beta_rows,
        z_rows: projected_item.z_rows,
        residual_rows: projected_item.residual_rows,
        state_before: projected_item.state_before,
        pre_recurrent_tally: tally,
    })
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_query_key_scaling(
    q_rows: &mut [Vec<f32>],
    k_rows: &mut [Vec<f32>],
    dims: ResolvedLinearAttentionDims,
) -> Option<()> {
    if q_rows.len() != k_rows.len() || dims.key_head_dim == 0 {
        return None;
    }
    let inv_scale = (dims.key_head_dim as f32).sqrt().recip();
    if !inv_scale.is_finite() {
        return None;
    }
    for row in q_rows.iter_mut() {
        if row.len() != dims.key_dim {
            return None;
        }
        for value in row.iter_mut() {
            *value *= inv_scale;
        }
    }
    for row in k_rows.iter_mut() {
        if row.len() != dims.key_dim {
            return None;
        }
    }
    Some(())
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn finish_linear_attention_item(
    artifacts: &NativeModelArtifacts,
    prepared_item: PreparedLinearAttentionItem,
    mut linear_outputs: Vec<Vec<f32>>,
    recurrent_tally: PrefixAttentionExecutionTally,
    linear_norm: &MetalNativeTensorBufferBinding,
    out_proj: &MetalNativeTensorBufferBinding,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    layer_state: &mut MetalLinearLayerState,
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    let mut tally = prepared_item.pre_recurrent_tally.merge(recurrent_tally);
    tally = tally.merge(apply_batched_per_head_rms_norm_rows_with_tally(
        &mut linear_outputs,
        dims.num_value_heads,
        dims.value_head_dim,
        linear_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        bringup,
    )?);
    let linear_gate_tally = apply_linear_attention_gate_in_place_with_tally(
        &mut linear_outputs,
        &prepared_item.z_rows,
        dims,
        bringup,
    )?;
    tally = tally.merge(linear_gate_tally);

    let (attention_hidden_rows, attention_tally) = project_batched_matrix_rows_with_tally(
        out_proj,
        0,
        dims.hidden_dim,
        &linear_outputs,
        dims.value_dim,
        bringup,
    )?;
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
        attention_tally,
    ));

    let mut hidden_after_attention = attention_hidden_rows;
    let attention_residual_tally = add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
        &mut hidden_after_attention,
        &prepared_item.residual_rows,
        dims.hidden_dim,
        out_proj.native_dtype,
        bringup,
    )?;
    tally = tally.merge(attention_residual_tally);
    let (next_hidden_states, ffn_tally, _) = apply_ffn_continuation_rows_with_tally(
        artifacts,
        layer,
        buffers,
        &hidden_after_attention,
        bringup,
    )?;
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(ffn_tally));
    layer_state.processed_tokens = layer_state
        .processed_tokens
        .checked_add(prepared_item.token_count)?;
    Some((next_hidden_states, tally))
}

#[cfg(target_os = "macos")]
fn reset_linear_layer_state(
    layer_state: &mut MetalLinearLayerState,
    dims: ResolvedLinearAttentionDims,
) {
    layer_state.processed_tokens = 0;
    layer_state.conv_state.clear();
    layer_state.conv_state.resize(
        (dims.conv_kernel_dim.saturating_sub(1)).saturating_mul(dims.conv_dim),
        0.0,
    );
    layer_state.ssm_state.clear();
    layer_state.ssm_state.resize(
        dims.num_value_heads
            .saturating_mul(dims.value_head_dim)
            .saturating_mul(dims.key_head_dim),
        0.0,
    );
}

#[cfg(target_os = "macos")]
fn ensure_linear_attention_conv_state_shape(
    conv_state: &mut Vec<f32>,
    dims: ResolvedLinearAttentionDims,
) -> Option<()> {
    let expected_state_len = dims
        .conv_kernel_dim
        .saturating_sub(1)
        .checked_mul(dims.conv_dim)?;
    if conv_state.len() != expected_state_len {
        conv_state.clear();
        conv_state.resize(expected_state_len, 0.0);
    }
    Some(())
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_conv_rows_with_path(
    conv1d: &MetalNativeTensorBufferBinding,
    qkv_rows: &[Vec<f32>],
    conv_state: &mut Vec<f32>,
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    ensure_linear_attention_conv_state_shape(conv_state, dims)?;
    let mut conv_output_rows = Vec::with_capacity(qkv_rows.len());
    let mut tally = PrefixAttentionExecutionTally::default();
    let allow_native_step =
        batched_linear_attention_conv_split_retry_worthwhile(bringup, conv1d.native_dtype);
    for qkv in qkv_rows {
        let (conv_out, next_state, step_used_native) = if allow_native_step {
            apply_linear_attention_conv_step_with_path(conv1d, qkv, conv_state, dims, bringup)?
        } else {
            let (conv_out, next_state) =
                apply_linear_attention_conv_step_cpu(conv1d, qkv, conv_state, dims)?;
            (conv_out, next_state, false)
        };
        tally = tally.record_projection_rows(dims.conv_dim, step_used_native);
        *conv_state = next_state;
        conv_output_rows.push(conv_out);
    }
    Some((conv_output_rows, tally))
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_conv_step_with_path(
    conv1d: &MetalNativeTensorBufferBinding,
    qkv: &[f32],
    conv_state: &[f32],
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, Vec<f32>, bool)> {
    if let Some((conv_out, next_state)) =
        linear_attention_conv_step_with_optional_native_path(bringup, conv1d, qkv, conv_state, dims)
    {
        return Some((conv_out, next_state, true));
    }
    let (conv_out, next_state) =
        apply_linear_attention_conv_step_cpu(conv1d, qkv, conv_state, dims)?;
    Some((conv_out, next_state, false))
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_conv_step_cpu(
    conv1d: &MetalNativeTensorBufferBinding,
    qkv: &[f32],
    conv_state: &[f32],
    dims: ResolvedLinearAttentionDims,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let expected_state_len = dims
        .conv_kernel_dim
        .saturating_sub(1)
        .checked_mul(dims.conv_dim)?;
    if qkv.len() < dims.conv_dim || conv_state.len() != expected_state_len {
        return None;
    }
    let mut conv_out = vec![0.0_f32; dims.conv_dim];
    for (channel, conv_out_channel) in conv_out.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for tap in 0..dims.conv_kernel_dim.saturating_sub(1) {
            let state_index = tap.checked_mul(dims.conv_dim)?.checked_add(channel)?;
            acc += conv_state.get(state_index).copied()?
                * linear_attention_conv_weight(conv1d, dims, channel, tap)?;
        }
        acc += qkv.get(channel).copied()?
            * linear_attention_conv_weight(conv1d, dims, channel, dims.conv_kernel_dim - 1)?;
        *conv_out_channel = silu(acc);
    }
    round_slice_to_native_dtype(&mut conv_out, conv1d.native_dtype);
    let mut next_state = conv_state.to_vec();
    if dims.conv_kernel_dim > 1 {
        for tap in 0..dims.conv_kernel_dim - 2 {
            let dst = tap.checked_mul(dims.conv_dim)?;
            let src = (tap + 1).checked_mul(dims.conv_dim)?;
            let moved = next_state.get(src..src + dims.conv_dim)?.to_vec();
            next_state
                .get_mut(dst..dst + dims.conv_dim)?
                .copy_from_slice(&moved);
        }
        let tail_base = (dims.conv_kernel_dim - 2).checked_mul(dims.conv_dim)?;
        next_state
            .get_mut(tail_base..tail_base + dims.conv_dim)?
            .copy_from_slice(qkv.get(..dims.conv_dim)?);
    }
    Some((conv_out, next_state))
}

#[cfg(target_os = "macos")]
fn linear_attention_conv_weight(
    conv1d: &MetalNativeTensorBufferBinding,
    dims: ResolvedLinearAttentionDims,
    channel: usize,
    tap: usize,
) -> Option<f32> {
    if channel >= dims.conv_dim || tap >= dims.conv_kernel_dim {
        return None;
    }
    tensor_scalar_f32(
        conv1d,
        channel
            .checked_mul(dims.conv_kernel_dim)?
            .checked_add(tap)?,
    )
}

#[cfg(target_os = "macos")]
fn linear_attention_conv_feedback_key(
    batch_size: usize,
    dtype: NativeTensorDataType,
    dims: ResolvedLinearAttentionDims,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::LinearAttentionConv1d {
        batch_size,
        dtype,
        dims,
    }
}

#[cfg(target_os = "macos")]
fn linear_attention_conv_step_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    conv1d: &MetalNativeTensorBufferBinding,
    qkv: &[f32],
    conv_state: &[f32],
    dims: ResolvedLinearAttentionDims,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (outputs, states) = linear_attention_conv_batched_single_step_with_optional_native_path(
        bringup,
        conv1d,
        &[qkv.to_vec()],
        &[conv_state.to_vec()],
        dims,
    )?;
    Some((outputs.into_iter().next()?, states.into_iter().next()?))
}

#[cfg(target_os = "macos")]
fn apply_batched_linear_attention_decode_conv_with_optional_native_path(
    projected_items: &[ProjectedLinearAttentionItem],
    request_states: &BTreeMap<crate::ids::RequestId, MetalLinearRequestState>,
    layer_index: u32,
    conv1d: &MetalNativeTensorBufferBinding,
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Option<(Vec<Vec<f32>>, Vec<f32>, PrefixAttentionExecutionTally)>>> {
    let eligible_indices =
        batched_linear_attention_decode_projected_candidate_indices(projected_items);
    if eligible_indices.len() < 2 {
        return None;
    }
    if !batched_linear_attention_conv_split_retry_worthwhile(bringup, conv1d.native_dtype) {
        return None;
    }
    let mut results = vec![None; projected_items.len()];
    let mut group_allowed = |candidate: &[usize]| {
        if candidate.len() <= 1 {
            return true;
        }
        let Some(bringup) = bringup else {
            return true;
        };
        let feedback_key =
            linear_attention_conv_feedback_key(candidate.len(), conv1d.native_dtype, dims);
        optional_kernel_allowed(bringup, &feedback_key)
    };
    let partitions = partition_linear_attention_decode_candidate_indices_by_batched_viability(
        eligible_indices,
        &mut group_allowed,
    );
    for partition in partitions {
        collect_batched_linear_attention_decode_conv_partition_results(
            projected_items,
            request_states,
            layer_index,
            conv1d,
            dims,
            bringup,
            &partition,
            &mut results,
        )?;
    }
    Some(results)
}

#[cfg(target_os = "macos")]
fn batched_linear_attention_decode_projected_candidate_indices(
    projected_items: &[ProjectedLinearAttentionItem],
) -> Vec<usize> {
    projected_items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            (item.token_count == 1 && item.qkv_rows.len() == 1).then_some(index)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn collect_batched_linear_attention_decode_conv_partition_results(
    projected_items: &[ProjectedLinearAttentionItem],
    request_states: &BTreeMap<crate::ids::RequestId, MetalLinearRequestState>,
    layer_index: u32,
    conv1d: &MetalNativeTensorBufferBinding,
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
    candidate_indices: &[usize],
    results: &mut [Option<(Vec<Vec<f32>>, Vec<f32>, PrefixAttentionExecutionTally)>],
) -> Option<()> {
    if candidate_indices.is_empty() {
        return Some(());
    }
    if candidate_indices.len() == 1 {
        return Some(());
    }

    let qkv_rows = candidate_indices
        .iter()
        .map(|index| projected_items.get(*index)?.qkv_rows.first().cloned())
        .collect::<Option<Vec<_>>>()?;
    let state_rows = candidate_indices
        .iter()
        .map(|index| {
            let request_id = projected_items.get(*index)?.request_id;
            let layer_state = request_states.get(&request_id)?.layers.get(&layer_index)?;
            Some(layer_state.conv_state.clone())
        })
        .collect::<Option<Vec<_>>>()?;
    if let Some((outputs, next_states)) =
        linear_attention_conv_batched_single_step_with_optional_native_path(
            bringup,
            conv1d,
            &qkv_rows,
            &state_rows,
            dims,
        )
    {
        if outputs.len() == candidate_indices.len() && next_states.len() == candidate_indices.len()
        {
            for ((index, output), next_state) in candidate_indices
                .iter()
                .copied()
                .zip(outputs)
                .zip(next_states)
            {
                results[index] = Some((
                    vec![output],
                    next_state,
                    PrefixAttentionExecutionTally::default()
                        .record_projection_rows(dims.conv_dim, true),
                ));
            }
            return Some(());
        }
    }

    if batched_linear_attention_conv_split_retry_worthwhile(bringup, conv1d.native_dtype) {
        let split_index = candidate_indices.len() / 2;
        collect_batched_linear_attention_decode_conv_partition_results(
            projected_items,
            request_states,
            layer_index,
            conv1d,
            dims,
            bringup,
            &candidate_indices[..split_index],
            results,
        )?;
        collect_batched_linear_attention_decode_conv_partition_results(
            projected_items,
            request_states,
            layer_index,
            conv1d,
            dims,
            bringup,
            &candidate_indices[split_index..],
            results,
        )?;
    }
    Some(())
}

#[cfg(target_os = "macos")]
fn linear_attention_conv_batched_single_step_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    conv1d: &MetalNativeTensorBufferBinding,
    qkv_rows: &[Vec<f32>],
    state_rows: &[Vec<f32>],
    dims: ResolvedLinearAttentionDims,
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let bringup = bringup?;
    let batch_size = qkv_rows.len();
    let expected_state_len = dims
        .conv_kernel_dim
        .saturating_sub(1)
        .checked_mul(dims.conv_dim)?;
    if batch_size == 0
        || state_rows.len() != batch_size
        || qkv_rows.iter().any(|row| row.len() != dims.conv_dim)
        || state_rows.iter().any(|row| row.len() != expected_state_len)
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .linear_attention_conv1d_kernel(conv1d.native_dtype)?;
    let feedback_key = linear_attention_conv_feedback_key(batch_size, conv1d.native_dtype, dims);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let flat_qkv = qkv_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let flat_state = if expected_state_len == 0 {
                vec![0.0_f32; batch_size]
            } else {
                state_rows
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect::<Vec<_>>()
            };
            let qkv_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_qkv);
            let state_in_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_state);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(batch_size.checked_mul(dims.conv_dim)?),
            );
            let state_out_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(batch_size.checked_mul(expected_state_len.max(1))?),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.linear_attention_conv1d");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.linear_attention_conv1d.compute");
            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&qkv_buffer), 0);
            encoder.set_buffer(1, Some(&conv1d.native_buffer), 0);
            encoder.set_buffer(2, Some(&state_in_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder.set_buffer(4, Some(&state_out_buffer), 0);
            set_linear_attention_conv_dispatch_params(
                encoder,
                5,
                saturating_usize_to_u32(batch_size),
                dims,
            );
            encoder.dispatch_threads(
                MTLSize::new(batch_size.checked_mul(dims.conv_dim)?.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(batch_size.checked_mul(dims.conv_dim)?.max(1) as u64),
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed
            {
                return None;
            }

            let conv_out = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(batch_size.checked_mul(dims.conv_dim)?),
            );
            let next_state = read_shared_buffer_prefix(
                &state_out_buffer,
                saturating_usize_to_u32(batch_size.checked_mul(expected_state_len)?),
            );
            if conv_out.len() != batch_size.checked_mul(dims.conv_dim)?
                || next_state.len() != batch_size.checked_mul(expected_state_len)?
                || conv_out.iter().any(|value| !value.is_finite())
                || next_state.iter().any(|value| !value.is_finite())
            {
                return None;
            }
            let output_rows = conv_out
                .chunks(dims.conv_dim)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let state_rows = next_state
                .chunks(expected_state_len.max(1))
                .take(batch_size)
                .map(|chunk| {
                    if expected_state_len == 0 {
                        Vec::new()
                    } else {
                        chunk.to_vec()
                    }
                })
                .collect::<Vec<_>>();
            (output_rows.len() == batch_size && state_rows.len() == batch_size)
                .then_some((output_rows, state_rows))
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_recurrent_with_state(
    q_rows: &[Vec<f32>],
    k_rows: &[Vec<f32>],
    v_rows: &[Vec<f32>],
    g_rows: &[Vec<f32>],
    beta_rows: &[Vec<f32>],
    state: &mut Vec<f32>,
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    let expected_state_len = dims
        .num_value_heads
        .checked_mul(dims.value_head_dim)?
        .checked_mul(dims.key_head_dim)?;
    if state.len() != expected_state_len {
        state.clear();
        state.resize(expected_state_len, 0.0);
    }

    let mut outputs = Vec::with_capacity(q_rows.len());
    let mut tally = PrefixAttentionExecutionTally::default();
    let allow_native_step = batched_linear_attention_recurrent_split_retry_worthwhile(bringup);
    if q_rows.len() == 1 && allow_native_step {
        if let Some((output, next_state)) = linear_attention_single_step_with_optional_native_path(
            bringup,
            q_rows.first()?,
            k_rows.first()?,
            v_rows.first()?,
            g_rows.first()?,
            beta_rows.first()?,
            state,
            dims,
        ) {
            *state = next_state;
            outputs.push(output);
            tally = tally.record_projection_rows(dims.value_dim, true);
            return Some((outputs, tally));
        }
    }

    for (((q_row, k_row), v_row), (a_row, b_row)) in q_rows
        .iter()
        .zip(k_rows.iter())
        .zip(v_rows.iter())
        .zip(g_rows.iter().zip(beta_rows.iter()))
    {
        if allow_native_step {
            if let Some((output, next_state)) =
                linear_attention_single_step_with_optional_native_path(
                    bringup, q_row, k_row, v_row, a_row, b_row, state, dims,
                )
            {
                *state = next_state;
                outputs.push(output);
                tally = tally.record_projection_rows(dims.value_dim, true);
                continue;
            }
        }
        outputs.push(apply_linear_attention_recurrent_step_cpu(
            q_row, k_row, v_row, a_row, b_row, state, dims,
        )?);
        tally = tally.record_projection_rows(dims.value_dim, false);
    }
    Some((outputs, tally))
}

#[cfg(target_os = "macos")]
fn apply_batched_linear_attention_decode_recurrent_with_optional_native_path(
    prepared_items: &[PreparedLinearAttentionItem],
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Option<(Vec<Vec<f32>>, Vec<f32>, PrefixAttentionExecutionTally)>>> {
    let eligible_indices = batched_linear_attention_decode_candidate_indices(prepared_items);
    if eligible_indices.len() < 2 {
        return None;
    }
    if !batched_linear_attention_recurrent_split_retry_worthwhile(bringup) {
        return None;
    }
    let mut results = vec![None; prepared_items.len()];
    let mut group_allowed = |candidate: &[usize]| {
        if candidate.len() <= 1 {
            return true;
        }
        let Some(bringup) = bringup else {
            return true;
        };
        let feedback_key = linear_gated_delta_feedback_key(candidate.len(), dims);
        optional_kernel_allowed(bringup, &feedback_key)
    };
    let partitions = partition_linear_attention_decode_candidate_indices_by_batched_viability(
        eligible_indices,
        &mut group_allowed,
    );
    for partition in partitions {
        collect_batched_linear_attention_decode_recurrent_partition_results(
            prepared_items,
            dims,
            bringup,
            &partition,
            &mut results,
        )?;
    }
    Some(results)
}

#[cfg(target_os = "macos")]
fn batched_linear_attention_decode_candidate_indices(
    prepared_items: &[PreparedLinearAttentionItem],
) -> Vec<usize> {
    prepared_items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            (item.q_rows.len() == 1
                && item.k_rows.len() == 1
                && item.v_rows.len() == 1
                && item.g_rows.len() == 1
                && item.beta_rows.len() == 1)
                .then_some(index)
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn partition_linear_attention_decode_candidate_indices_by_batched_viability(
    candidate_indices: Vec<usize>,
    group_allowed: &mut impl for<'a> FnMut(&'a [usize]) -> bool,
) -> Vec<Vec<usize>> {
    let mut stack = vec![candidate_indices];
    let mut partitions = Vec::new();
    while let Some(current_group) = stack.pop() {
        if current_group.len() <= 1 || group_allowed(&current_group) {
            partitions.push(current_group);
            continue;
        }
        let split_index = current_group.len() / 2;
        stack.push(current_group[split_index..].to_vec());
        stack.push(current_group[..split_index].to_vec());
    }
    partitions
}

#[cfg(target_os = "macos")]
fn collect_batched_linear_attention_decode_recurrent_partition_results(
    prepared_items: &[PreparedLinearAttentionItem],
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
    candidate_indices: &[usize],
    results: &mut [Option<(Vec<Vec<f32>>, Vec<f32>, PrefixAttentionExecutionTally)>],
) -> Option<()> {
    if candidate_indices.is_empty() {
        return Some(());
    }
    if candidate_indices.len() == 1 {
        return Some(());
    }

    let q_rows = candidate_indices
        .iter()
        .map(|index| prepared_items.get(*index)?.q_rows.first().cloned())
        .collect::<Option<Vec<_>>>()?;
    let k_rows = candidate_indices
        .iter()
        .map(|index| prepared_items.get(*index)?.k_rows.first().cloned())
        .collect::<Option<Vec<_>>>()?;
    let v_rows = candidate_indices
        .iter()
        .map(|index| prepared_items.get(*index)?.v_rows.first().cloned())
        .collect::<Option<Vec<_>>>()?;
    let g_rows = candidate_indices
        .iter()
        .map(|index| prepared_items.get(*index)?.g_rows.first().cloned())
        .collect::<Option<Vec<_>>>()?;
    let beta_rows = candidate_indices
        .iter()
        .map(|index| prepared_items.get(*index)?.beta_rows.first().cloned())
        .collect::<Option<Vec<_>>>()?;
    let state_rows = candidate_indices
        .iter()
        .map(|index| Some(prepared_items.get(*index)?.state_before.clone()))
        .collect::<Option<Vec<_>>>()?;

    if let Some((outputs, next_states)) =
        linear_attention_batched_single_step_with_optional_native_path(
            bringup,
            &q_rows,
            &k_rows,
            &v_rows,
            &g_rows,
            &beta_rows,
            &state_rows,
            dims,
        )
    {
        if outputs.len() == candidate_indices.len() && next_states.len() == candidate_indices.len()
        {
            for ((index, output), next_state) in candidate_indices
                .iter()
                .copied()
                .zip(outputs)
                .zip(next_states)
            {
                results[index] = Some((
                    vec![output],
                    next_state,
                    PrefixAttentionExecutionTally::default()
                        .record_projection_rows(dims.value_dim, true),
                ));
            }
            return Some(());
        }
    }

    if batched_linear_attention_recurrent_split_retry_worthwhile(bringup) {
        let split_index = candidate_indices.len() / 2;
        collect_batched_linear_attention_decode_recurrent_partition_results(
            prepared_items,
            dims,
            bringup,
            &candidate_indices[..split_index],
            results,
        )?;
        collect_batched_linear_attention_decode_recurrent_partition_results(
            prepared_items,
            dims,
            bringup,
            &candidate_indices[split_index..],
            results,
        )?;
    }
    Some(())
}

#[cfg(target_os = "macos")]
fn linear_gated_delta_feedback_key(
    batch_size: usize,
    dims: ResolvedLinearAttentionDims,
) -> MetalOptionalKernelFeedbackKey {
    MetalOptionalKernelFeedbackKey::LinearGatedDelta { batch_size, dims }
}

#[cfg(target_os = "macos")]
fn linear_attention_single_step_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    q_row: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    dims: ResolvedLinearAttentionDims,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (outputs, states) = linear_attention_batched_single_step_with_optional_native_path(
        bringup,
        &[q_row.to_vec()],
        &[k_row.to_vec()],
        &[v_row.to_vec()],
        &[g.to_vec()],
        &[beta.to_vec()],
        &[state_in.to_vec()],
        dims,
    )?;
    Some((outputs.into_iter().next()?, states.into_iter().next()?))
}

#[cfg(target_os = "macos")]
fn linear_attention_batched_single_step_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    q_rows: &[Vec<f32>],
    k_rows: &[Vec<f32>],
    v_rows: &[Vec<f32>],
    g_rows: &[Vec<f32>],
    beta_rows: &[Vec<f32>],
    state_rows: &[Vec<f32>],
    dims: ResolvedLinearAttentionDims,
) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let bringup = bringup?;
    let batch_size = q_rows.len();
    let state_len = dims
        .num_value_heads
        .checked_mul(dims.value_head_dim)?
        .checked_mul(dims.key_head_dim)?;
    if batch_size == 0
        || k_rows.len() != batch_size
        || v_rows.len() != batch_size
        || g_rows.len() != batch_size
        || beta_rows.len() != batch_size
        || state_rows.len() != batch_size
        || q_rows.iter().any(|row| row.len() != dims.key_dim)
        || k_rows.iter().any(|row| row.len() != dims.key_dim)
        || v_rows.iter().any(|row| row.len() != dims.value_dim)
        || g_rows.iter().any(|row| row.len() != dims.num_value_heads)
        || beta_rows
            .iter()
            .any(|row| row.len() != dims.num_value_heads)
        || state_rows.iter().any(|row| row.len() != state_len)
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .linear_gated_delta_step_kernel()?;
    let feedback_key = linear_gated_delta_feedback_key(batch_size, dims);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let flat_q = q_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let flat_k = k_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let flat_v = v_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let flat_g = g_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let flat_beta = beta_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let flat_state = state_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let q_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_q);
            let k_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_k);
            let v_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_v);
            let g_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_g);
            let beta_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_beta);
            let state_buffer = new_shared_buffer_with_data(&bringup.state.device, &flat_state);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(batch_size.checked_mul(dims.value_dim)?),
            );
            let state_out_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(batch_size.checked_mul(state_len)?),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.linear_gated_delta_step");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.linear_gated_delta_step.compute");
            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&q_buffer), 0);
            encoder.set_buffer(1, Some(&k_buffer), 0);
            encoder.set_buffer(2, Some(&v_buffer), 0);
            encoder.set_buffer(3, Some(&g_buffer), 0);
            encoder.set_buffer(4, Some(&beta_buffer), 0);
            encoder.set_buffer(5, Some(&state_buffer), 0);
            encoder.set_buffer(6, Some(&output_buffer), 0);
            encoder.set_buffer(7, Some(&state_out_buffer), 0);
            set_linear_gated_delta_dispatch_params(
                encoder,
                8,
                saturating_usize_to_u32(batch_size),
                dims,
            );
            encoder.dispatch_threads(
                MTLSize::new(
                    pipeline.pipeline.thread_execution_width().max(32),
                    dims.value_head_dim.max(1) as u64,
                    batch_size.checked_mul(dims.num_value_heads)?.max(1) as u64,
                ),
                MTLSize::new(
                    pipeline.pipeline.thread_execution_width().clamp(1, 64),
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed
            {
                return None;
            }

            let output = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(batch_size.checked_mul(dims.value_dim)?),
            );
            let state_out = read_shared_buffer_prefix(
                &state_out_buffer,
                saturating_usize_to_u32(batch_size.checked_mul(state_len)?),
            );
            if output.len() != batch_size.checked_mul(dims.value_dim)?
                || state_out.len() != batch_size.checked_mul(state_len)?
                || output.iter().any(|value| !value.is_finite())
                || state_out.iter().any(|value| !value.is_finite())
            {
                return None;
            }

            let output_rows = output
                .chunks(dims.value_dim)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            let state_rows = state_out
                .chunks(state_len)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>();
            (output_rows.len() == batch_size && state_rows.len() == batch_size)
                .then_some((output_rows, state_rows))
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn compute_linear_attention_beta(beta_row: &[f32]) -> Option<Vec<f32>> {
    Some(
        beta_row
            .iter()
            .map(|value| sigmoid(*value))
            .collect::<Vec<_>>(),
    )
}

#[cfg(target_os = "macos")]
fn compute_linear_attention_beta_rows_with_tally(
    beta_rows: &[Vec<f32>],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    if let Some(output) =
        compute_linear_attention_beta_rows_with_optional_native_path(bringup, beta_rows)
    {
        let element_count = beta_rows
            .len()
            .checked_mul(beta_rows.first().map(Vec::len).unwrap_or(0))
            .unwrap_or(0);
        return Some((
            output,
            PrefixAttentionExecutionTally::default()
                .record_ffn_activation_elements(element_count, true),
        ));
    }
    if beta_rows.len() > 1 && batched_linear_attention_beta_split_retry_worthwhile(bringup) {
        let split_index = beta_rows.len() / 2;
        let (left_output, left_tally) =
            compute_linear_attention_beta_rows_with_tally(&beta_rows[..split_index], bringup)?;
        let (right_output, right_tally) =
            compute_linear_attention_beta_rows_with_tally(&beta_rows[split_index..], bringup)?;
        let mut output = left_output;
        output.extend(right_output);
        return Some((output, left_tally.merge(right_tally)));
    }
    let row_width = beta_rows.first().map(Vec::len).unwrap_or(0);
    Some((
        beta_rows
            .iter()
            .map(|beta_row| compute_linear_attention_beta(beta_row))
            .collect::<Option<Vec<_>>>()?,
        PrefixAttentionExecutionTally::default()
            .record_ffn_activation_elements(beta_rows.len().checked_mul(row_width)?, false),
    ))
}

#[cfg(target_os = "macos")]
fn compute_linear_attention_beta_rows_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    beta_rows: &[Vec<f32>],
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    let row_count = beta_rows.len();
    let row_width = beta_rows.first().map(Vec::len)?;
    let element_count = row_count.checked_mul(row_width)?;
    if row_count == 0 || row_width == 0 || beta_rows.iter().any(|row| row.len() != row_width) {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .linear_attention_beta_kernel()?;
    let feedback_key = batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let flattened_beta = beta_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let beta_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_beta);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.linear_attention_beta");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.linear_attention_beta.compute");
            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&beta_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            set_ffn_gate_product_dispatch_params(
                encoder,
                2,
                saturating_usize_to_u32(element_count),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed
            {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }
            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn compute_linear_attention_decay(
    a_row: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
) -> Option<Vec<f32>> {
    if a_row.len() != a_log.len() || a_row.len() != dt_bias.len() {
        return None;
    }
    Some(
        a_row
            .iter()
            .zip(a_log.iter())
            .zip(dt_bias.iter())
            .map(|((a, a_log), dt_bias)| (-a_log.exp() * softplus(*a + *dt_bias)).exp())
            .collect::<Vec<_>>(),
    )
}

#[cfg(target_os = "macos")]
fn compute_linear_attention_decay_rows_with_tally(
    a_rows: &[Vec<f32>],
    a_log: &[f32],
    dt_bias: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    if let Some(output) = compute_linear_attention_decay_rows_with_optional_native_path(
        bringup, a_rows, a_log, dt_bias,
    ) {
        let element_count = a_rows
            .len()
            .checked_mul(a_rows.first().map(Vec::len).unwrap_or(0))
            .unwrap_or(0);
        return Some((
            output,
            PrefixAttentionExecutionTally::default()
                .record_ffn_activation_elements(element_count, true),
        ));
    }
    if a_rows.len() > 1 && batched_linear_attention_decay_split_retry_worthwhile(bringup) {
        let split_index = a_rows.len() / 2;
        let (left_output, left_tally) = compute_linear_attention_decay_rows_with_tally(
            &a_rows[..split_index],
            a_log,
            dt_bias,
            bringup,
        )?;
        let (right_output, right_tally) = compute_linear_attention_decay_rows_with_tally(
            &a_rows[split_index..],
            a_log,
            dt_bias,
            bringup,
        )?;
        let mut output = left_output;
        output.extend(right_output);
        return Some((output, left_tally.merge(right_tally)));
    }
    let row_width = a_rows.first().map(Vec::len).unwrap_or(0);
    Some((
        a_rows
            .iter()
            .map(|a_row| compute_linear_attention_decay(a_row, a_log, dt_bias))
            .collect::<Option<Vec<_>>>()?,
        PrefixAttentionExecutionTally::default()
            .record_ffn_activation_elements(a_rows.len().checked_mul(row_width)?, false),
    ))
}

#[cfg(target_os = "macos")]
fn compute_linear_attention_decay_rows_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    a_rows: &[Vec<f32>],
    a_log: &[f32],
    dt_bias: &[f32],
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    let row_count = a_rows.len();
    let row_width = a_rows.first().map(Vec::len)?;
    let element_count = row_count.checked_mul(row_width)?;
    if row_count == 0
        || row_width == 0
        || a_log.len() != row_width
        || dt_bias.len() != row_width
        || a_rows.iter().any(|row| row.len() != row_width)
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .linear_attention_decay_kernel()?;
    let feedback_key = batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let flattened_a = a_rows
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            let flattened_a_log = (0..row_count)
                .flat_map(|_| a_log.iter().copied())
                .collect::<Vec<_>>();
            let flattened_dt_bias = (0..row_count)
                .flat_map(|_| dt_bias.iter().copied())
                .collect::<Vec<_>>();
            let a_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_a);
            let a_log_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_a_log);
            let dt_bias_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &flattened_dt_bias);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.linear_attention_decay");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.linear_attention_decay.compute");
            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&a_buffer), 0);
            encoder.set_buffer(1, Some(&a_log_buffer), 0);
            encoder.set_buffer(2, Some(&dt_bias_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            set_ffn_gate_product_dispatch_params(
                encoder,
                4,
                saturating_usize_to_u32(element_count),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed
            {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }
            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_recurrent_step_cpu(
    q_row: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    dims: ResolvedLinearAttentionDims,
) -> Option<Vec<f32>> {
    let mut output = vec![0.0_f32; dims.value_dim];
    for value_head in 0..dims.num_value_heads {
        let key_head = value_head.checked_div(dims.repeat_factor)?;
        let q_head = q_row.get(
            key_head.checked_mul(dims.key_head_dim)?
                ..(key_head + 1).checked_mul(dims.key_head_dim)?,
        )?;
        let k_head = k_row.get(
            key_head.checked_mul(dims.key_head_dim)?
                ..(key_head + 1).checked_mul(dims.key_head_dim)?,
        )?;
        let v_head = v_row.get(
            value_head.checked_mul(dims.value_head_dim)?
                ..(value_head + 1).checked_mul(dims.value_head_dim)?,
        )?;
        let output_head = output.get_mut(
            value_head.checked_mul(dims.value_head_dim)?
                ..(value_head + 1).checked_mul(dims.value_head_dim)?,
        )?;
        for (value_lane, output_value) in output_head.iter_mut().enumerate() {
            let state_base = value_head
                .checked_mul(dims.value_head_dim)?
                .checked_add(value_lane)?
                .checked_mul(dims.key_head_dim)?;
            let mut kv_mem = 0.0_f32;
            for key_lane in 0..dims.key_head_dim {
                let idx = state_base.checked_add(key_lane)?;
                let decayed = state.get(idx).copied()? * g.get(value_head).copied()?;
                state.get_mut(idx).map(|slot| *slot = decayed)?;
                kv_mem += decayed * k_head.get(key_lane).copied()?;
            }
            let delta =
                (v_head.get(value_lane).copied()? - kv_mem) * beta.get(value_head).copied()?;
            let mut head_out = 0.0_f32;
            for key_lane in 0..dims.key_head_dim {
                let idx = state_base.checked_add(key_lane)?;
                let updated = state.get(idx).copied()? + k_head.get(key_lane).copied()? * delta;
                state.get_mut(idx).map(|slot| *slot = updated)?;
                head_out += updated * q_head.get(key_lane).copied()?;
            }
            *output_value = head_out;
        }
    }
    Some(output)
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_gate_in_place(
    outputs: &mut [Vec<f32>],
    z_rows: &[Vec<f32>],
    dims: ResolvedLinearAttentionDims,
) -> Option<()> {
    if outputs.len() != z_rows.len() {
        return None;
    }
    for (output_row, z_row) in outputs.iter_mut().zip(z_rows.iter()) {
        if output_row.len() < dims.value_dim || z_row.len() < dims.value_dim {
            return None;
        }
        for (output, gate) in output_row
            .get_mut(..dims.value_dim)?
            .iter_mut()
            .zip(z_row.get(..dims.value_dim)?.iter())
        {
            *output *= silu(*gate);
        }
    }
    Some(())
}

#[cfg(target_os = "macos")]
fn apply_attention_output_gate_in_place_with_tally(
    rows: &mut [Vec<f32>],
    gate: &[f32],
    row_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if row_width == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    if rows.iter().any(|row| row.len() < row_width)
        || gate.len() != rows.len().checked_mul(row_width)?
    {
        return None;
    }

    if let Some(output_rows) =
        apply_attention_output_gate_with_optional_native_path(bringup, rows, gate, row_width)
    {
        for (row, output_row) in rows.iter_mut().zip(output_rows) {
            row.get_mut(..row_width)?
                .copy_from_slice(output_row.get(..row_width)?);
        }
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_ffn_activation_elements(rows.len().checked_mul(row_width)?, true),
        );
    }
    if rows.len() > 1 && batched_attention_output_gate_split_retry_worthwhile(bringup) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at_mut(split_index);
        let gate_split = split_index.checked_mul(row_width)?;
        let (left_gate, right_gate) = gate.split_at(gate_split);
        let left_tally = apply_attention_output_gate_in_place_with_tally(
            left_rows, left_gate, row_width, bringup,
        )?;
        let right_tally = apply_attention_output_gate_in_place_with_tally(
            right_rows, right_gate, row_width, bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    for (token_index, row) in rows.iter_mut().enumerate() {
        let gate_base = token_index.checked_mul(row_width)?;
        let token_gate = gate.get(gate_base..gate_base.checked_add(row_width)?)?;
        for (value, gate_value) in row.get_mut(..row_width)?.iter_mut().zip(token_gate.iter()) {
            *value *= 1.0 / (1.0 + (-gate_value).exp());
        }
    }
    Some(
        PrefixAttentionExecutionTally::default()
            .record_ffn_activation_elements(rows.len().checked_mul(row_width)?, false),
    )
}

#[cfg(target_os = "macos")]
fn apply_attention_output_gate_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    rows: &[Vec<f32>],
    gate: &[f32],
    row_width: usize,
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    let row_count = rows.len();
    let element_count = row_count.checked_mul(row_width)?;
    if row_count == 0
        || row_width == 0
        || gate.len() != element_count
        || rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .attention_output_gate_kernel()?;
    let feedback_key = batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let mut flattened_rows = Vec::with_capacity(element_count);
            for row in rows {
                flattened_rows.extend_from_slice(row.get(..row_width)?);
            }

            let row_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_rows);
            let gate_buffer = new_shared_buffer_with_data(&bringup.state.device, gate);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.attention_output_gate");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.attention_output_gate.compute");
            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&gate_buffer), 0);
            encoder.set_buffer(1, Some(&row_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_ffn_gate_product_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(element_count),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed
            {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }
            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_gate_in_place_with_tally(
    outputs: &mut [Vec<f32>],
    z_rows: &[Vec<f32>],
    dims: ResolvedLinearAttentionDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if let Some(output) =
        apply_linear_attention_gate_with_optional_native_path(bringup, outputs, z_rows, dims)
    {
        for (row, output_row) in outputs.iter_mut().zip(output) {
            row.get_mut(..dims.value_dim)?
                .copy_from_slice(output_row.get(..dims.value_dim)?);
        }
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_ffn_activation_elements(outputs.len().checked_mul(dims.value_dim)?, true),
        );
    }
    if outputs.len() > 1 && batched_linear_attention_gate_split_retry_worthwhile(bringup) {
        let split_index = outputs.len() / 2;
        let (left_outputs, right_outputs) = outputs.split_at_mut(split_index);
        let (left_z_rows, right_z_rows) = z_rows.split_at(split_index);
        let left_tally = apply_linear_attention_gate_in_place_with_tally(
            left_outputs,
            left_z_rows,
            dims,
            bringup,
        )?;
        let right_tally = apply_linear_attention_gate_in_place_with_tally(
            right_outputs,
            right_z_rows,
            dims,
            bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }
    apply_linear_attention_gate_in_place(outputs, z_rows, dims)?;
    Some(
        PrefixAttentionExecutionTally::default()
            .record_ffn_activation_elements(outputs.len().checked_mul(dims.value_dim)?, false),
    )
}

#[cfg(target_os = "macos")]
fn apply_linear_attention_gate_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    outputs: &[Vec<f32>],
    z_rows: &[Vec<f32>],
    dims: ResolvedLinearAttentionDims,
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    if outputs.is_empty()
        || outputs.len() != z_rows.len()
        || outputs.iter().any(|row| row.len() < dims.value_dim)
        || z_rows.iter().any(|row| row.len() < dims.value_dim)
    {
        return None;
    }

    let row_count = outputs.len();
    let row_width = dims.value_dim;
    let element_count = row_count.checked_mul(row_width)?;
    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .linear_attention_gate_kernel()?;
    let feedback_key = batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let mut flattened_outputs = Vec::with_capacity(element_count);
            let mut flattened_z = Vec::with_capacity(element_count);
            for (output_row, z_row) in outputs.iter().zip(z_rows.iter()) {
                flattened_outputs.extend_from_slice(output_row.get(..row_width)?);
                flattened_z.extend_from_slice(z_row.get(..row_width)?);
            }

            let output_in_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &flattened_outputs);
            let z_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_z);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.linear_attention_gate");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.linear_attention_gate.compute");
            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&z_buffer), 0);
            encoder.set_buffer(1, Some(&output_in_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_ffn_gate_product_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(element_count),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }

            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(target_os = "macos")]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

#[cfg(target_os = "macos")]
fn execution_item_token_ranges(input: &RunnerInput) -> Option<Vec<std::ops::Range<usize>>> {
    let mut token_cursor = 0_usize;
    let mut token_ranges = Vec::with_capacity(input.execution_batch.items.len());
    for item in &input.execution_batch.items {
        let token_count = item.scheduled_token_count as usize;
        let token_end = token_cursor.checked_add(token_count)?;
        token_ranges.push(token_cursor..token_end);
        token_cursor = token_end;
    }
    (token_cursor == input.execution_batch.total_scheduled_tokens as usize).then_some(token_ranges)
}

#[cfg(target_os = "macos")]
fn collect_prefix_attention_outputs_with_item_fallback(
    artifacts: &NativeModelArtifacts,
    input: &RunnerInput,
    staged_inputs: &MetalDispatchStagedInputs,
    item_token_ranges: &[std::ops::Range<usize>],
    item_range: std::ops::Range<usize>,
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> Option<(Vec<f32>, PrefixAttentionExecutionTally)> {
    if item_range.start >= item_range.end {
        return Some((Vec::new(), PrefixAttentionExecutionTally::default()));
    }
    if let Some(partitioned) = partition_prefix_attention_item_range_by_native_group_feasibility(
        input,
        item_range.clone(),
        staged_inputs.layout,
        bringup,
        layer_cache,
    ) {
        if partitioned.len() > 1 {
            return collect_prefix_attention_outputs_for_partitioned_item_ranges(
                artifacts,
                input,
                staged_inputs,
                item_token_ranges,
                &partitioned,
                bringup,
                layer_cache,
            );
        }
    }

    let group_items = input.execution_batch.items.get(item_range.clone())?;
    let group_token_range =
        token_range_for_execution_item_range(item_token_ranges, item_range.clone())?;
    let group_input = runner_input_for_execution_items(input, group_items)?;
    let group_workload = MetalDispatchWorkload::from_runner_input(&group_input).ok()?;
    let group_staged_inputs =
        slice_staged_inputs_for_token_range(staged_inputs, group_token_range)?;
    let numeric_group_workload = group_workload.with_numeric_layout(group_staged_inputs.layout);
    let native_dispatch_available = bringup.is_some();
    let native_supported =
        native_prefix_attention_is_viable(&group_workload, native_dispatch_available, layer_cache);
    let prefix_group_feedback = bringup.and_then(|bringup| {
        prefix_attention_group_feedback_key_for_item_count(
            group_items.len(),
            &numeric_group_workload,
        )
        .map(|feedback_key| {
            let allowed = optional_kernel_allowed(bringup, &feedback_key);
            (bringup, feedback_key, allowed)
        })
    });
    let native_feedback_allowed = prefix_group_feedback
        .as_ref()
        .is_none_or(|(_, _, allowed)| *allowed);
    let native_viable = native_supported && native_feedback_allowed;
    let mut native_attempt_failed = false;
    let mut skip_native_retry = false;
    if native_viable {
        if let Some(mut native_attempt) = try_native_attention_output_from_model_layer(
            artifacts,
            &group_workload,
            &group_staged_inputs,
            bringup,
            layer_cache,
        ) {
            if let Some((bringup, feedback_key, _)) = prefix_group_feedback.as_ref() {
                record_optional_kernel_result(bringup, feedback_key, true);
            }
            commit_prefix_attention_dispatch_attempt(layer_cache, &mut native_attempt);
            return Some((
                native_attempt.attention_output,
                PrefixAttentionExecutionTally::default().record(true),
            ));
        }
        native_attempt_failed = true;
        skip_native_retry = true;
        if let Some((bringup, feedback_key, _)) = prefix_group_feedback.as_ref() {
            record_optional_kernel_result(bringup, feedback_key, false);
        }
    }
    if prefix_attention_group_should_split(
        group_items.len(),
        native_dispatch_available,
        native_viable,
        native_attempt_failed,
    ) {
        return split_prefix_attention_outputs_with_item_fallback(
            artifacts,
            input,
            staged_inputs,
            item_token_ranges,
            item_range,
            bringup,
            layer_cache,
        );
    }
    let mut attempt = resolve_attention_output_from_model_layer(
        artifacts,
        &group_workload,
        &group_staged_inputs,
        bringup,
        layer_cache,
        !skip_native_retry,
    )?;
    commit_prefix_attention_dispatch_attempt(layer_cache, &mut attempt);
    Some((
        attempt.attention_output,
        PrefixAttentionExecutionTally::default().record(attempt.used_native_dispatch),
    ))
}

#[cfg(target_os = "macos")]
fn split_prefix_attention_outputs_with_item_fallback(
    artifacts: &NativeModelArtifacts,
    input: &RunnerInput,
    staged_inputs: &MetalDispatchStagedInputs,
    item_token_ranges: &[std::ops::Range<usize>],
    item_range: std::ops::Range<usize>,
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> Option<(Vec<f32>, PrefixAttentionExecutionTally)> {
    let item_count = item_range.end.checked_sub(item_range.start)?;
    if item_count <= 1 {
        return None;
    }
    let split_index = item_range.start.checked_add(item_count / 2)?;
    let (mut left_output, left_tally) = collect_prefix_attention_outputs_with_item_fallback(
        artifacts,
        input,
        staged_inputs,
        item_token_ranges,
        item_range.start..split_index,
        bringup,
        layer_cache,
    )?;
    let (right_output, right_tally) = collect_prefix_attention_outputs_with_item_fallback(
        artifacts,
        input,
        staged_inputs,
        item_token_ranges,
        split_index..item_range.end,
        bringup,
        layer_cache,
    )?;
    left_output.extend(right_output);
    Some((left_output, left_tally.merge(right_tally)))
}

#[cfg(target_os = "macos")]
fn collect_prefix_attention_outputs_for_partitioned_item_ranges(
    artifacts: &NativeModelArtifacts,
    input: &RunnerInput,
    staged_inputs: &MetalDispatchStagedInputs,
    item_token_ranges: &[std::ops::Range<usize>],
    partitions: &[std::ops::Range<usize>],
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> Option<(Vec<f32>, PrefixAttentionExecutionTally)> {
    let mut collected_output = Vec::new();
    let mut collected_tally = PrefixAttentionExecutionTally::default();
    for partition in partitions {
        let (partition_output, partition_tally) =
            collect_prefix_attention_outputs_with_item_fallback(
                artifacts,
                input,
                staged_inputs,
                item_token_ranges,
                partition.clone(),
                bringup,
                layer_cache,
            )?;
        collected_output.extend(partition_output);
        collected_tally = collected_tally.merge(partition_tally);
    }
    Some((collected_output, collected_tally))
}

#[cfg(target_os = "macos")]
fn partition_prefix_attention_item_range_by_group_predicate(
    item_range: std::ops::Range<usize>,
    group_allowed: &mut impl FnMut(std::ops::Range<usize>) -> Option<bool>,
) -> Option<Vec<std::ops::Range<usize>>> {
    if item_range.start >= item_range.end {
        return None;
    }

    let mut partitions = Vec::new();
    let mut stack = vec![item_range.clone()];
    while let Some(current_range) = stack.pop() {
        if current_range.end.checked_sub(current_range.start)? <= 1
            || group_allowed(current_range.clone())?
        {
            partitions.push(current_range);
            continue;
        }
        let split_index = current_range.start + (current_range.end - current_range.start) / 2;
        stack.push(split_index..current_range.end);
        stack.push(current_range.start..split_index);
    }

    (partitions.len() > 1).then_some(partitions)
}

#[cfg(target_os = "macos")]
fn partition_prefix_attention_item_range_by_native_group_feasibility(
    input: &RunnerInput,
    item_range: std::ops::Range<usize>,
    layout: MetalDispatchNumericLayout,
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> Option<Vec<std::ops::Range<usize>>> {
    let native_dispatch_available = bringup.is_some();
    if !native_dispatch_available {
        return None;
    }
    let mut group_allowed = |candidate_range: std::ops::Range<usize>| {
        let group_items = input.execution_batch.items.get(candidate_range.clone())?;
        let group_input = runner_input_for_execution_items(input, group_items)?;
        let group_workload = MetalDispatchWorkload::from_runner_input(&group_input).ok()?;
        if !native_prefix_attention_is_viable(
            &group_workload,
            native_dispatch_available,
            layer_cache,
        ) {
            return Some(false);
        }
        if let Some(bringup) = bringup {
            let feedback_key =
                prefix_attention_group_feedback_key(&group_workload.with_numeric_layout(layout));
            return Some(optional_kernel_allowed(bringup, &feedback_key));
        }
        Some(true)
    };
    partition_prefix_attention_item_range_by_group_predicate(item_range, &mut group_allowed)
}

#[cfg(target_os = "macos")]
fn prefix_attention_group_should_split(
    group_item_count: usize,
    native_dispatch_available: bool,
    native_viable: bool,
    native_attempt_failed: bool,
) -> bool {
    native_dispatch_available && group_item_count > 1 && (!native_viable || native_attempt_failed)
}

#[cfg(target_os = "macos")]
fn native_prefix_attention_is_viable(
    workload: &MetalDispatchWorkload,
    native_dispatch_available: bool,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> bool {
    native_dispatch_available && workload_supports_native_prefix_attention(workload, layer_cache)
}

#[cfg(target_os = "macos")]
fn token_range_for_execution_item_range(
    item_token_ranges: &[std::ops::Range<usize>],
    item_range: std::ops::Range<usize>,
) -> Option<std::ops::Range<usize>> {
    if item_range.start >= item_range.end {
        return Some(0..0);
    }
    let first_range = item_token_ranges.get(item_range.start)?;
    let last_index = item_range.end.checked_sub(1)?;
    let last_range = item_token_ranges.get(last_index)?;
    Some(first_range.start..last_range.end)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct PrefixAttentionDispatchAttempt {
    attention_output: Vec<f32>,
    used_native_dispatch: bool,
    updated_layer_cache: Option<MetalPersistentLayerKvCache>,
}

#[cfg(target_os = "macos")]
fn try_native_attention_output_from_model_layer(
    artifacts: &NativeModelArtifacts,
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> Option<PrefixAttentionDispatchAttempt> {
    let bringup = bringup?;
    if let Some(layer_cache) = layer_cache {
        let cached_state = layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .clone();
        let trial_cache = Mutex::new(cached_state);
        let numeric_workload = workload.with_numeric_layout(staged_inputs.layout);
        let attention_config = native_model_reference_attention_config(
            artifacts,
            staged_inputs.layout.head_dim as usize,
        );
        let mut trial_cache_state = trial_cache
            .lock()
            .expect("trial prefix layer cache mutex should not be poisoned");
        let workload_is_self_contained =
            workload_supports_native_prefix_attention(&numeric_workload, None);
        let native_trace = {
            let cache_seed = trial_cache_state.seed_for_workload(&numeric_workload);
            bringup
                .dispatch_numeric_workload_ephemeral_seeded_with_attention_config(
                    &numeric_workload,
                    staged_inputs,
                    cache_seed,
                    Some(attention_config),
                )
                .ok()
        };
        if let Some((trace, cache_snapshot)) = native_trace {
            if let Some(attention_output) =
                decode_attention_output_values(&trace.numeric.attention_output_bits)
            {
                trial_cache_state.apply_snapshot(&numeric_workload, &cache_snapshot);
                drop(trial_cache_state);
                return Some(PrefixAttentionDispatchAttempt {
                    attention_output,
                    used_native_dispatch: true,
                    updated_layer_cache: Some(
                        trial_cache
                            .into_inner()
                            .expect("trial prefix layer cache mutex should not be poisoned"),
                    ),
                });
            }
        }
        if workload_is_self_contained {
            let native_trace_without_seed = bringup
                .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
                    &numeric_workload,
                    staged_inputs,
                    Some(attention_config),
                )
                .ok();
            if let Some((trace, cache_snapshot)) = native_trace_without_seed {
                if let Some(attention_output) =
                    decode_attention_output_values(&trace.numeric.attention_output_bits)
                {
                    trial_cache_state.apply_snapshot(&numeric_workload, &cache_snapshot);
                    drop(trial_cache_state);
                    return Some(PrefixAttentionDispatchAttempt {
                        attention_output,
                        used_native_dispatch: true,
                        updated_layer_cache: Some(
                            trial_cache
                                .into_inner()
                                .expect("trial prefix layer cache mutex should not be poisoned"),
                        ),
                    });
                }
            }
        }
        return None;
    }

    let numeric_workload = workload.with_numeric_layout(staged_inputs.layout);
    let attention_config =
        native_model_reference_attention_config(artifacts, staged_inputs.layout.head_dim as usize);
    let trace = bringup
        .dispatch_numeric_workload_ephemeral_with_attention_config(
            &numeric_workload,
            staged_inputs,
            Some(attention_config),
        )
        .ok()?;
    let attention_output = decode_attention_output_values(&trace.numeric.attention_output_bits)?;
    Some(PrefixAttentionDispatchAttempt {
        attention_output,
        used_native_dispatch: true,
        updated_layer_cache: None,
    })
}

#[cfg(target_os = "macos")]
fn resolve_attention_output_from_model_layer(
    artifacts: &NativeModelArtifacts,
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
    allow_native_retry: bool,
) -> Option<PrefixAttentionDispatchAttempt> {
    if let Some(layer_cache) = layer_cache {
        let cached_state = layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .clone();
        let trial_cache = Mutex::new(cached_state);
        let (attention_output, used_native_dispatch) = attention_output_from_model_layer(
            artifacts,
            workload,
            staged_inputs,
            bringup,
            Some(&trial_cache),
            allow_native_retry,
        )?;
        return Some(PrefixAttentionDispatchAttempt {
            attention_output,
            used_native_dispatch,
            updated_layer_cache: Some(
                trial_cache
                    .into_inner()
                    .expect("trial prefix layer cache mutex should not be poisoned"),
            ),
        });
    }

    let (attention_output, used_native_dispatch) = attention_output_from_model_layer(
        artifacts,
        workload,
        staged_inputs,
        bringup,
        None,
        allow_native_retry,
    )?;
    Some(PrefixAttentionDispatchAttempt {
        attention_output,
        used_native_dispatch,
        updated_layer_cache: None,
    })
}

#[cfg(target_os = "macos")]
fn commit_prefix_attention_dispatch_attempt(
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
    attempt: &mut PrefixAttentionDispatchAttempt,
) {
    let Some(updated_layer_cache) = attempt.updated_layer_cache.take() else {
        return;
    };
    let Some(layer_cache) = layer_cache else {
        return;
    };
    let mut layer_cache = layer_cache
        .lock()
        .expect("metal prefix layer cache mutex should not be poisoned");
    *layer_cache = updated_layer_cache;
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn project_hidden_states_from_layer_attention_output(
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    attention_o: &MetalNativeTensorBufferBinding,
    hidden_states: &[Vec<f32>],
    attention_output: &[f32],
    attention_head_size: usize,
    attn_output_gate: Option<&[f32]>,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    let hidden_width = hidden_states.iter().map(Vec::len).min()?;
    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;
    let dims = resolved_model_reference_layer_dims(
        attention_o,
        ffn_norm,
        &layer.ffn_gate_up,
        buffers,
        ffn_down,
        hidden_width,
        attention_head_size,
    )?;
    let mut attention_input_rows = hidden_states
        .iter()
        .enumerate()
        .map(|(token_index, _)| {
            let token_base = token_index.checked_mul(attention_head_size)?;
            let token_end = token_base.checked_add(attention_head_size)?;
            let token_attention_output = attention_output.get(token_base..token_end)?;
            Some(token_attention_output.get(..dims.input_width)?.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut tally = PrefixAttentionExecutionTally::default();

    // When attn_output_gate is enabled, apply sigmoid(gate) * attention_output
    // before the O projection. The gate has the same shape as the attention output.
    if let Some(gate) = attn_output_gate {
        tally = tally.merge(apply_attention_output_gate_in_place_with_tally(
            &mut attention_input_rows,
            gate,
            dims.input_width,
            bringup,
        )?);
    }

    // Fused single-token path: encode O-proj through FFN residual in one command buffer.
    // Supports both bf16 and f16 weight dtypes. Gate is already applied above.
    if hidden_states.len() == 1
        && layer.attention_post_norm.is_none()
        && layer.ffn_post_norm.is_none()
        && layer.moe.is_none()
        && matches!(
            attention_o.native_dtype,
            NativeTensorDataType::Bf16 | NativeTensorDataType::F16
        )
    {
        if let Some(bringup) = bringup {
            let attention_input = attention_input_rows.first()?.get(..dims.input_width)?;
            let residual_input = hidden_states.first()?.get(..dims.hidden_dim)?;
            if let Some((result, fused_tally)) = fused_layer_continuation_on_gpu(
                bringup,
                artifacts,
                layer,
                buffers,
                attention_o,
                attention_input,
                residual_input,
                dims,
            ) {
                return Some((vec![result], tally.merge(fused_tally)));
            }
        }
    }

    let (attention_hidden_rows, attention_projection_tally) =
        project_batched_matrix_rows_with_tally(
            attention_o,
            0,
            dims.hidden_dim,
            &attention_input_rows,
            dims.input_width,
            bringup,
        )?;
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
        attention_projection_tally,
    ));

    let residual_rows = hidden_states
        .iter()
        .map(|hidden_state| {
            hidden_state
                .get(..dims.hidden_dim)
                .map(|slice| slice.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    let mut hidden_after_attention = attention_hidden_rows;
    if let Some(attention_post_norm) = &layer.attention_post_norm {
        let attention_post_norm = buffers.binding_for(attention_post_norm)?;
        tally = tally.merge(apply_batched_row_rms_norm_with_binding_in_place_with_tally(
            &mut hidden_after_attention,
            dims.hidden_dim,
            attention_post_norm,
            native_model_rms_norm_epsilon(artifacts),
            native_model_rms_norm_weight_offset(artifacts),
            bringup,
        )?);
    }
    let attention_residual_tally = add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
        &mut hidden_after_attention,
        &residual_rows,
        dims.hidden_dim,
        attention_o.native_dtype,
        bringup,
    )?;
    tally = tally.merge(attention_residual_tally);
    let (next_hidden_states, ffn_tally, _) = apply_ffn_continuation_rows_with_tally(
        artifacts,
        layer,
        buffers,
        &hidden_after_attention,
        bringup,
    )?;
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(ffn_tally));
    Some((next_hidden_states, tally))
}

/// Fused single-token layer continuation: encodes O-projection through FFN residual
/// into a single Metal command buffer with pre-allocated arena buffers.
/// Returns `None` to fall back to per-operation dispatch.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn fused_layer_continuation_on_gpu(
    bringup: &MetalRuntimeBringup,
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    attention_o: &MetalNativeTensorBufferBinding,
    attention_input: &[f32],
    residual_input: &[f32],
    dims: ModelReferenceLayerDims,
) -> Option<(Vec<f32>, PrefixAttentionExecutionTally)> {
    let hidden_dim = dims.hidden_dim;
    let intermediate_dim = dims.intermediate_dim;
    if hidden_dim == 0 || intermediate_dim == 0 || attention_input.len() < dims.input_width {
        return None;
    }
    if residual_input.len() < hidden_dim {
        return None;
    }

    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;

    let plan = bringup.state.optional_kernel_dispatch_plan;

    // Resolve all pipelines needed for the fused dispatch.
    let weight_dtype = attention_o.native_dtype;
    let (proj_kernel_name, proj_pipeline_index) = plan.projection_kernel(weight_dtype)?;
    let (norm_kernel_name, norm_pipeline_index) = plan.rms_norm_kernel(weight_dtype)?;
    let (add_kernel_name, add_pipeline_index) = plan.vector_add_kernel()?;
    let activation = native_model_ffn_activation(artifacts);
    let (gate_kernel_name, gate_pipeline_index) = plan.ffn_gate_product_kernel(activation)?;

    // Check feedback for the fused layer dispatch.
    let fused_feedback_key = MetalOptionalKernelFeedbackKey::Kernel("fused_layer_continuation");
    if !optional_kernel_allowed(bringup, &fused_feedback_key) {
        return None;
    }

    let proj_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        proj_kernel_name,
        proj_pipeline_index,
    )
    .ok()?;
    let norm_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        norm_kernel_name,
        norm_pipeline_index,
    )
    .ok()?;
    let add_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        add_kernel_name,
        add_pipeline_index,
    )
    .ok()?;
    let gate_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        gate_kernel_name,
        gate_pipeline_index,
    )
    .ok()?;

    // Resolve FFN gate/up weight bindings.
    let (gate_weight, gate_row_offset, up_weight, up_row_offset) = match &layer.ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            (packed, 0_usize, packed, intermediate_dim)
        }
        MetalFfnGateUpBindings::Split { gate, up } => (
            buffers.binding_for(gate)?,
            0_usize,
            buffers.binding_for(up)?,
            0_usize,
        ),
    };

    let hidden_dim_u32 = saturating_usize_to_u32(hidden_dim);
    let intermediate_dim_u32 = saturating_usize_to_u32(intermediate_dim);
    let epsilon = native_model_rms_norm_epsilon(artifacts);
    let weight_offset = native_model_rms_norm_weight_offset(artifacts);

    // Weight matrix dimensions for projection dispatch params.
    let (attn_o_rows, attn_o_cols) = tensor_matrix_dimensions(&attention_o.meta.spec)?;
    if dims.input_width > attn_o_cols || hidden_dim > attn_o_rows {
        return None;
    }
    let (_, gate_cols) = tensor_matrix_dimensions(&gate_weight.meta.spec)?;
    if hidden_dim > gate_cols {
        return None;
    }
    let (_, up_cols) = tensor_matrix_dimensions(&up_weight.meta.spec)?;
    if hidden_dim > up_cols {
        return None;
    }
    let (_, down_cols) = tensor_matrix_dimensions(&ffn_down.meta.spec)?;
    if intermediate_dim > down_cols {
        return None;
    }

    let dtype_bytes = native_dtype_size_bytes(weight_dtype);
    let gate_row_byte_offset = gate_row_offset
        .checked_mul(gate_cols)?
        .checked_mul(dtype_bytes)?;
    let up_row_byte_offset = up_row_offset
        .checked_mul(up_cols)?
        .checked_mul(dtype_bytes)?;

    let thread_width = proj_pipeline.pipeline.thread_execution_width().max(1);

    // Ensure arena is allocated and large enough.
    let mut arena_guard = bringup
        .state
        .fused_layer_arena
        .lock()
        .expect("fused layer arena mutex should not be poisoned");
    if arena_guard
        .as_ref()
        .is_none_or(|a| !a.fits(hidden_dim_u32, intermediate_dim_u32))
    {
        *arena_guard = Some(FusedLayerArena::new(
            &bringup.state.device,
            hidden_dim_u32,
            intermediate_dim_u32,
        ));
    }
    let arena = arena_guard.as_ref()?;

    let result = autoreleasepool(|| {
        // Create input buffers from CPU data (StorageModeShared = unified memory).
        let attn_input_buffer = new_shared_buffer_with_data(
            &bringup.state.device,
            &attention_input[..dims.input_width],
        );
        let residual_buffer =
            new_shared_buffer_with_data(&bringup.state.device, &residual_input[..hidden_dim]);

        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.fused_layer_continuation");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("ax.phase1.fused_layer_continuation.compute");

        // 1. O projection: attn_input → hidden
        encoder.set_compute_pipeline_state(&proj_pipeline.pipeline);
        encoder.set_buffer(0, Some(&attn_input_buffer), 0);
        encoder.set_buffer(1, Some(&attention_o.native_buffer), 0);
        encoder.set_buffer(2, Some(&arena.hidden), 0);
        set_logits_projection_dispatch_params(
            encoder,
            3,
            hidden_dim_u32,
            saturating_usize_to_u32(attn_o_cols),
            saturating_usize_to_u32(dims.input_width),
        );
        encoder.dispatch_threads(
            MTLSize::new(hidden_dim as u64, 1, 1),
            MTLSize::new(thread_width.min(hidden_dim as u64), 1, 1),
        );

        // 2. Attention residual add: hidden += residual → hidden
        encoder.set_compute_pipeline_state(&add_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.hidden), 0);
        encoder.set_buffer(1, Some(&residual_buffer), 0);
        encoder.set_buffer(2, Some(&arena.hidden), 0);
        set_vector_add_dispatch_params(encoder, 3, hidden_dim_u32);
        encoder.dispatch_threads(
            MTLSize::new(hidden_dim as u64, 1, 1),
            MTLSize::new(
                add_pipeline
                    .pipeline
                    .thread_execution_width()
                    .max(1)
                    .min(hidden_dim as u64),
                1,
                1,
            ),
        );

        // 3. FFN RMS norm: hidden → normed
        encoder.set_compute_pipeline_state(&norm_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.hidden), 0);
        encoder.set_buffer(1, Some(&ffn_norm.native_buffer), 0);
        encoder.set_buffer(2, Some(&arena.normed), 0);
        set_rms_norm_dispatch_params(encoder, 3, hidden_dim_u32, epsilon, weight_offset);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

        // 4. Gate projection: normed → gate
        encoder.set_compute_pipeline_state(&proj_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.normed), 0);
        encoder.set_buffer(
            1,
            Some(&gate_weight.native_buffer),
            gate_row_byte_offset as u64,
        );
        encoder.set_buffer(2, Some(&arena.gate), 0);
        set_logits_projection_dispatch_params(
            encoder,
            3,
            intermediate_dim_u32,
            saturating_usize_to_u32(gate_cols),
            hidden_dim_u32,
        );
        encoder.dispatch_threads(
            MTLSize::new(intermediate_dim as u64, 1, 1),
            MTLSize::new(thread_width.min(intermediate_dim as u64), 1, 1),
        );

        // 5. Up projection: normed → up
        encoder.set_buffer(0, Some(&arena.normed), 0);
        encoder.set_buffer(1, Some(&up_weight.native_buffer), up_row_byte_offset as u64);
        encoder.set_buffer(2, Some(&arena.up), 0);
        set_logits_projection_dispatch_params(
            encoder,
            3,
            intermediate_dim_u32,
            saturating_usize_to_u32(up_cols),
            hidden_dim_u32,
        );
        encoder.dispatch_threads(
            MTLSize::new(intermediate_dim as u64, 1, 1),
            MTLSize::new(thread_width.min(intermediate_dim as u64), 1, 1),
        );

        // 6. SiLU activation: gate_silu_product(gate, up) → gate (in-place)
        encoder.set_compute_pipeline_state(&gate_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.gate), 0);
        encoder.set_buffer(1, Some(&arena.up), 0);
        encoder.set_buffer(2, Some(&arena.gate), 0);
        set_ffn_gate_product_dispatch_params(encoder, 3, intermediate_dim_u32);
        encoder.dispatch_threads(
            MTLSize::new(intermediate_dim as u64, 1, 1),
            MTLSize::new(
                gate_pipeline
                    .pipeline
                    .thread_execution_width()
                    .max(1)
                    .min(intermediate_dim as u64),
                1,
                1,
            ),
        );

        // 7. Down projection: gate → down
        encoder.set_compute_pipeline_state(&proj_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.gate), 0);
        encoder.set_buffer(1, Some(&ffn_down.native_buffer), 0);
        encoder.set_buffer(2, Some(&arena.down), 0);
        set_logits_projection_dispatch_params(
            encoder,
            3,
            hidden_dim_u32,
            saturating_usize_to_u32(down_cols),
            intermediate_dim_u32,
        );
        encoder.dispatch_threads(
            MTLSize::new(hidden_dim as u64, 1, 1),
            MTLSize::new(thread_width.min(hidden_dim as u64), 1, 1),
        );

        // 8. FFN residual add: hidden + down → hidden
        encoder.set_compute_pipeline_state(&add_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.hidden), 0);
        encoder.set_buffer(1, Some(&arena.down), 0);
        encoder.set_buffer(2, Some(&arena.hidden), 0);
        set_vector_add_dispatch_params(encoder, 3, hidden_dim_u32);
        encoder.dispatch_threads(
            MTLSize::new(hidden_dim as u64, 1, 1),
            MTLSize::new(
                add_pipeline
                    .pipeline
                    .thread_execution_width()
                    .max(1)
                    .min(hidden_dim as u64),
                1,
                1,
            ),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer_status(command_buffer.status());
        if status != MetalCommandBufferStatus::Completed {
            return None;
        }

        let output = read_shared_buffer_prefix(&arena.hidden, hidden_dim_u32);
        if output.len() != hidden_dim || output.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(output)
    });

    record_optional_kernel_result(bringup, &fused_feedback_key, result.is_some());

    let output = result?;
    let total_projection_rows = hidden_dim + intermediate_dim * 2 + hidden_dim;
    let tally = PrefixAttentionExecutionTally::default()
        .record_projection_rows(total_projection_rows, true)
        .record_rms_norm_elements(hidden_dim, true)
        .record_ffn_activation_elements(intermediate_dim, true)
        .record_residual_add_elements(hidden_dim * 2, true);
    Some((output, tally))
}

/// Fused single-token FFN-only continuation: norm → gate_proj → up_proj →
/// gate×up_activation → down_proj → residual_add, all in one command buffer.
/// Used by linear-attention layers whose post-recurrent output goes directly
/// into the FFN without an O-projection step.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn fused_ffn_only_on_gpu(
    bringup: &MetalRuntimeBringup,
    artifacts: &NativeModelArtifacts,
    layer: &MetalNativeLayerBindings,
    buffers: &MetalNativeModelBufferBindings,
    hidden_input: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Option<(Vec<f32>, DirectDecodeNativeDenseTally)> {
    if hidden_dim == 0 || intermediate_dim == 0 || hidden_input.len() < hidden_dim {
        return None;
    }

    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;

    let plan = bringup.state.optional_kernel_dispatch_plan;
    let weight_dtype = ffn_down.native_dtype;
    let (proj_kernel_name, proj_pipeline_index) = plan.projection_kernel(weight_dtype)?;
    let (norm_kernel_name, norm_pipeline_index) = plan.rms_norm_kernel(weight_dtype)?;
    let (add_kernel_name, add_pipeline_index) = plan.vector_add_kernel()?;
    let activation = native_model_ffn_activation(artifacts);
    let (gate_kernel_name, gate_pipeline_index) = plan.ffn_gate_product_kernel(activation)?;

    let fused_feedback_key = MetalOptionalKernelFeedbackKey::Kernel("fused_ffn_only");
    if !optional_kernel_allowed(bringup, &fused_feedback_key) {
        return None;
    }

    let proj_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        proj_kernel_name,
        proj_pipeline_index,
    )
    .ok()?;
    let norm_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        norm_kernel_name,
        norm_pipeline_index,
    )
    .ok()?;
    let add_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        add_kernel_name,
        add_pipeline_index,
    )
    .ok()?;
    let gate_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        gate_kernel_name,
        gate_pipeline_index,
    )
    .ok()?;

    let (gate_weight, gate_row_offset, up_weight, up_row_offset) = match &layer.ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            (packed, 0_usize, packed, intermediate_dim)
        }
        MetalFfnGateUpBindings::Split { gate, up } => (
            buffers.binding_for(gate)?,
            0_usize,
            buffers.binding_for(up)?,
            0_usize,
        ),
    };

    let hidden_dim_u32 = saturating_usize_to_u32(hidden_dim);
    let intermediate_dim_u32 = saturating_usize_to_u32(intermediate_dim);
    let epsilon = native_model_rms_norm_epsilon(artifacts);
    let weight_offset = native_model_rms_norm_weight_offset(artifacts);

    let (_, gate_cols) = tensor_matrix_dimensions(&gate_weight.meta.spec)?;
    if hidden_dim > gate_cols {
        return None;
    }
    let (_, up_cols) = tensor_matrix_dimensions(&up_weight.meta.spec)?;
    if hidden_dim > up_cols {
        return None;
    }
    let (_, down_cols) = tensor_matrix_dimensions(&ffn_down.meta.spec)?;
    if intermediate_dim > down_cols {
        return None;
    }

    let dtype_bytes = native_dtype_size_bytes(weight_dtype);
    let gate_row_byte_offset = gate_row_offset
        .checked_mul(gate_cols)?
        .checked_mul(dtype_bytes)?;
    let up_row_byte_offset = up_row_offset
        .checked_mul(up_cols)?
        .checked_mul(dtype_bytes)?;

    let thread_width = proj_pipeline.pipeline.thread_execution_width().max(1);

    let mut arena_guard = bringup
        .state
        .fused_layer_arena
        .lock()
        .expect("fused layer arena mutex should not be poisoned");
    if arena_guard
        .as_ref()
        .is_none_or(|a| !a.fits(hidden_dim_u32, intermediate_dim_u32))
    {
        *arena_guard = Some(FusedLayerArena::new(
            &bringup.state.device,
            hidden_dim_u32,
            intermediate_dim_u32,
        ));
    }
    let arena = arena_guard.as_ref()?;

    let result = autoreleasepool(|| {
        let hidden_buffer =
            new_shared_buffer_with_data(&bringup.state.device, &hidden_input[..hidden_dim]);

        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.fused_ffn_only");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("ax.phase1.fused_ffn_only.compute");

        // 1. FFN RMS norm: hidden → normed
        encoder.set_compute_pipeline_state(&norm_pipeline.pipeline);
        encoder.set_buffer(0, Some(&hidden_buffer), 0);
        encoder.set_buffer(1, Some(&ffn_norm.native_buffer), 0);
        encoder.set_buffer(2, Some(&arena.normed), 0);
        set_rms_norm_dispatch_params(encoder, 3, hidden_dim_u32, epsilon, weight_offset);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

        // 2. Gate projection: normed → gate
        encoder.set_compute_pipeline_state(&proj_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.normed), 0);
        encoder.set_buffer(
            1,
            Some(&gate_weight.native_buffer),
            gate_row_byte_offset as u64,
        );
        encoder.set_buffer(2, Some(&arena.gate), 0);
        set_logits_projection_dispatch_params(
            encoder,
            3,
            intermediate_dim_u32,
            saturating_usize_to_u32(gate_cols),
            hidden_dim_u32,
        );
        encoder.dispatch_threads(
            MTLSize::new(intermediate_dim as u64, 1, 1),
            MTLSize::new(thread_width.min(intermediate_dim as u64), 1, 1),
        );

        // 3. Up projection: normed → up
        encoder.set_buffer(0, Some(&arena.normed), 0);
        encoder.set_buffer(1, Some(&up_weight.native_buffer), up_row_byte_offset as u64);
        encoder.set_buffer(2, Some(&arena.up), 0);
        set_logits_projection_dispatch_params(
            encoder,
            3,
            intermediate_dim_u32,
            saturating_usize_to_u32(up_cols),
            hidden_dim_u32,
        );
        encoder.dispatch_threads(
            MTLSize::new(intermediate_dim as u64, 1, 1),
            MTLSize::new(thread_width.min(intermediate_dim as u64), 1, 1),
        );

        // 4. SiLU/GELU gate×up activation: gate × act(up) → gate (in-place)
        encoder.set_compute_pipeline_state(&gate_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.gate), 0);
        encoder.set_buffer(1, Some(&arena.up), 0);
        encoder.set_buffer(2, Some(&arena.gate), 0);
        set_ffn_gate_product_dispatch_params(encoder, 3, intermediate_dim_u32);
        encoder.dispatch_threads(
            MTLSize::new(intermediate_dim as u64, 1, 1),
            MTLSize::new(
                gate_pipeline
                    .pipeline
                    .thread_execution_width()
                    .max(1)
                    .min(intermediate_dim as u64),
                1,
                1,
            ),
        );

        // 5. Down projection: gate → down
        encoder.set_compute_pipeline_state(&proj_pipeline.pipeline);
        encoder.set_buffer(0, Some(&arena.gate), 0);
        encoder.set_buffer(1, Some(&ffn_down.native_buffer), 0);
        encoder.set_buffer(2, Some(&arena.down), 0);
        set_logits_projection_dispatch_params(
            encoder,
            3,
            hidden_dim_u32,
            saturating_usize_to_u32(down_cols),
            intermediate_dim_u32,
        );
        encoder.dispatch_threads(
            MTLSize::new(hidden_dim as u64, 1, 1),
            MTLSize::new(thread_width.min(hidden_dim as u64), 1, 1),
        );

        // 6. FFN residual add: hidden + down → hidden
        encoder.set_compute_pipeline_state(&add_pipeline.pipeline);
        encoder.set_buffer(0, Some(&hidden_buffer), 0);
        encoder.set_buffer(1, Some(&arena.down), 0);
        encoder.set_buffer(2, Some(&hidden_buffer), 0);
        set_vector_add_dispatch_params(encoder, 3, hidden_dim_u32);
        encoder.dispatch_threads(
            MTLSize::new(hidden_dim as u64, 1, 1),
            MTLSize::new(
                add_pipeline
                    .pipeline
                    .thread_execution_width()
                    .max(1)
                    .min(hidden_dim as u64),
                1,
                1,
            ),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        if command_buffer_status(command_buffer.status()) != MetalCommandBufferStatus::Completed {
            return None;
        }

        let output = read_shared_buffer_prefix(&hidden_buffer, hidden_dim_u32);
        if output.len() != hidden_dim || output.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(output)
    });

    record_optional_kernel_result(bringup, &fused_feedback_key, result.is_some());

    let output = result?;
    let tally = DirectDecodeNativeDenseTally::default()
        .record_projection_rows(intermediate_dim.saturating_mul(2).saturating_add(hidden_dim), true)
        .record_rms_norm_elements(hidden_dim, true)
        .record_ffn_activation_elements(intermediate_dim, true)
        .record_residual_add_elements(hidden_dim, true);
    Some((output, tally))
}

#[cfg(target_os = "macos")]
fn attention_output_from_model_layer(
    artifacts: &NativeModelArtifacts,
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
    allow_native_retry: bool,
) -> Option<(Vec<f32>, bool)> {
    let numeric_workload = workload.with_numeric_layout(staged_inputs.layout);
    let attention_config =
        native_model_reference_attention_config(artifacts, staged_inputs.layout.head_dim as usize);
    let workload_is_self_contained =
        workload_supports_native_prefix_attention(&numeric_workload, None);
    if let Some(layer_cache) = layer_cache {
        let mut layer_cache = layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned");
        let seeded_native_retry_viable = allow_native_retry
            && gathered_slots_for_workload(&numeric_workload).is_some_and(|gathered_slots| {
                gathered_slots_support_native_prefix_attention_with_cache_state(
                    &numeric_workload,
                    &gathered_slots,
                    &layer_cache,
                )
            });
        let native_trace = if seeded_native_retry_viable {
            let cache_seed = layer_cache.seed_for_workload(&numeric_workload);
            bringup.and_then(|bringup| {
                bringup
                    .dispatch_numeric_workload_ephemeral_seeded_with_attention_config(
                        &numeric_workload,
                        staged_inputs,
                        cache_seed,
                        Some(attention_config),
                    )
                    .ok()
            })
        } else {
            None
        };
        if let Some((trace, cache_snapshot)) = native_trace {
            if let Some(attention_output) =
                decode_attention_output_values(&trace.numeric.attention_output_bits)
            {
                layer_cache.apply_snapshot(&numeric_workload, &cache_snapshot);
                return Some((attention_output, true));
            }
        }
        if allow_native_retry && workload_is_self_contained {
            let native_trace_without_seed = bringup.and_then(|bringup| {
                bringup
                    .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
                        &numeric_workload,
                        staged_inputs,
                        Some(attention_config),
                    )
                    .ok()
            });
            if let Some((trace, cache_snapshot)) = native_trace_without_seed {
                if let Some(attention_output) =
                    decode_attention_output_values(&trace.numeric.attention_output_bits)
                {
                    layer_cache.apply_snapshot(&numeric_workload, &cache_snapshot);
                    return Some((attention_output, true));
                }
            }
        }

        let reference = {
            let cache_seed = layer_cache.seed_for_workload(&numeric_workload);
            reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
                &numeric_workload,
                staged_inputs,
                Some(cache_seed),
                Some(attention_config),
            )
        };
        layer_cache.apply_snapshot(
            &numeric_workload,
            &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                &numeric_workload,
                &reference,
            ),
        );
        return Some((reference.attention_output, false));
    }

    if allow_native_retry && workload_is_self_contained {
        if let Some(bringup) = bringup {
            if let Ok(trace) = bringup.dispatch_numeric_workload_ephemeral_with_attention_config(
                &numeric_workload,
                staged_inputs,
                Some(attention_config),
            ) {
                if let Some(attention_output) =
                    decode_attention_output_values(&trace.numeric.attention_output_bits)
                {
                    return Some((attention_output, true));
                }
            }
        }
    }

    let reference = reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
        &numeric_workload,
        staged_inputs,
        None,
        Some(attention_config),
    );
    Some((reference.attention_output, false))
}

#[cfg(target_os = "macos")]
fn workload_supports_native_prefix_attention(
    workload: &MetalDispatchWorkload,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> bool {
    let Some(gathered_slots) = gathered_slots_for_workload(workload) else {
        return false;
    };
    if let Some(layer_cache) = layer_cache {
        let layer_cache = layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned");
        return gathered_slots_support_native_prefix_attention_with_cache_state(
            workload,
            &gathered_slots,
            &layer_cache,
        );
    }

    gathered_slots_support_native_prefix_attention(workload, &gathered_slots, |_| false)
}

#[cfg(target_os = "macos")]
fn gathered_slots_support_native_prefix_attention_with_cache_state(
    workload: &MetalDispatchWorkload,
    gathered_slots: &[u32],
    layer_cache: &MetalPersistentLayerKvCache,
) -> bool {
    gathered_slots_support_native_prefix_attention(workload, gathered_slots, |slot| {
        layer_cache.slot_initialized(slot)
    })
}

#[cfg(target_os = "macos")]
fn gathered_slots_support_native_prefix_attention(
    workload: &MetalDispatchWorkload,
    gathered_slots: &[u32],
    slot_initialized: impl FnMut(u32) -> bool,
) -> bool {
    let supported_slots = supported_prefix_slots(workload, slot_initialized);
    gathered_slots.iter().all(|slot| {
        supported_slots
            .get(*slot as usize)
            .copied()
            .unwrap_or(false)
    })
}

#[cfg(target_os = "macos")]
fn gathered_slots_for_workload(workload: &MetalDispatchWorkload) -> Option<Vec<u32>> {
    let gather_tokens = workload.kv_metadata.gather_token_count() as usize;
    let mut slots = Vec::with_capacity(gather_tokens);

    for token_id in 0..gather_tokens {
        let batch_id = batch_id_for_token(&workload.kv_metadata.cu_seq_lens, token_id as u32);
        let batch_offset =
            token_id as u32 - workload.kv_metadata.cu_seq_lens.get(batch_id).copied()?;
        let block_index = (batch_offset / workload.kv_metadata.block_size_tokens) as usize;
        let block_offset = batch_offset % workload.kv_metadata.block_size_tokens;
        let table_index = batch_id
            .checked_mul(workload.kv_metadata.gather_block_table_stride as usize)?
            .checked_add(block_index)?;
        let block_base = workload
            .kv_metadata
            .gather_block_table
            .get(table_index)
            .copied()?;
        slots.push(block_base.saturating_add(block_offset));
    }

    Some(slots)
}

#[cfg(target_os = "macos")]
fn runner_input_for_execution_item(
    input: &RunnerInput,
    item: &crate::scheduler::ExecutionItem,
) -> Option<RunnerInput> {
    runner_input_for_execution_items(input, std::slice::from_ref(item))
}

#[cfg(target_os = "macos")]
fn runner_input_for_execution_items(
    input: &RunnerInput,
    items: &[crate::scheduler::ExecutionItem],
) -> Option<RunnerInput> {
    if items.is_empty() {
        return None;
    }

    let total_scheduled_tokens = items.iter().try_fold(0_u32, |total, item| {
        total.checked_add(item.scheduled_token_count)
    })?;
    let mut seen_block_table_refs = BTreeSet::new();
    let mut block_tables = Vec::new();
    let mut seen_request_ids = BTreeSet::new();
    let mut request_contexts = Vec::new();
    for item in items {
        if seen_block_table_refs.insert(item.block_table_ref) {
            block_tables.push(
                input
                    .block_tables
                    .iter()
                    .find(|resolved| resolved.request_id == item.block_table_ref)?
                    .clone(),
            );
        }
        if seen_request_ids.insert(item.request_id) {
            request_contexts.push(*input.request_context(item.request_id)?);
        }
    }

    Some(RunnerInput {
        block_size_tokens: input.block_size_tokens,
        execution_batch: crate::scheduler::ExecutionBatch {
            step_id: input.execution_batch.step_id,
            model_id: input.execution_batch.model_id.clone(),
            execution_plan_ref: input.execution_batch.execution_plan_ref.clone(),
            items: items.to_vec(),
            total_scheduled_tokens,
            route_metadata: input.execution_batch.route_metadata.clone(),
        },
        block_tables,
        request_contexts,
    })
}

#[cfg(target_os = "macos")]
fn slice_staged_inputs_for_token_range(
    staged_inputs: &MetalDispatchStagedInputs,
    token_range: std::ops::Range<usize>,
) -> Option<MetalDispatchStagedInputs> {
    let head_size = staged_inputs.layout.head_size() as usize;
    let numeric_start = token_range.start.checked_mul(head_size)?;
    let numeric_end = token_range.end.checked_mul(head_size)?;
    Some(MetalDispatchStagedInputs {
        key: staged_inputs.key.get(numeric_start..numeric_end)?.to_vec(),
        value: staged_inputs
            .value
            .get(numeric_start..numeric_end)?
            .to_vec(),
        query: staged_inputs
            .query
            .get(numeric_start..numeric_end)?
            .to_vec(),
        layout: staged_inputs.layout,
        source: staged_inputs.source,
        prefix_attention_tally: PrefixAttentionExecutionTally::default(),
        final_layer_hidden_state_cache: None,
    })
}

#[cfg(target_os = "macos")]
fn runner_input_for_prefix_reuse_warmup(
    input: &RunnerInput,
    item: &crate::scheduler::ExecutionItem,
) -> Option<RunnerInput> {
    if item.reused_prefix_token_slice.is_empty() {
        return None;
    }

    let prefix_token_count = u32::try_from(item.reused_prefix_token_slice.len()).ok()?;
    let mut warmup_item = item.clone();
    warmup_item.mode = ExecutionMode::Prefill;
    warmup_item.input_token_slice = item.reused_prefix_token_slice.clone();
    warmup_item.reused_prefix_token_slice.clear();
    warmup_item.position_range = crate::scheduler::PositionRange {
        start: 0,
        end_exclusive: prefix_token_count,
    };
    warmup_item.scheduled_token_count = prefix_token_count;
    warmup_item.prefix_tokens_reused = 0;
    warmup_item.prefix_blocks_reused = 0;

    runner_input_for_execution_item(input, &warmup_item)
}

#[cfg(target_os = "macos")]
fn build_model_stage_rope_tables(
    rotary_dim: usize,
    position: f32,
    freq_base: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = rotary_dim / 2;
    let neg_log_base = -(freq_base.ln());
    let dim_inv = 2.0 / rotary_dim as f32;
    let mut cos_table = Vec::with_capacity(half_dim);
    let mut sin_table = Vec::with_capacity(half_dim);

    for pair_index in 0..half_dim {
        let freq = (neg_log_base * pair_index as f32 * dim_inv).exp();
        let theta = position * freq;
        let (sin_theta, cos_theta) = theta.sin_cos();
        cos_table.push(cos_theta);
        sin_table.push(sin_theta);
    }

    (cos_table, sin_table)
}

#[cfg(target_os = "macos")]
fn apply_split_half_rope_in_place(values: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    let half_dim = cos_table.len();
    if values.len() != half_dim.saturating_mul(2) || sin_table.len() != half_dim {
        return;
    }

    let (low, high) = values.split_at_mut(half_dim);
    for index in 0..half_dim {
        let cos_theta = cos_table[index];
        let sin_theta = sin_table[index];
        let low_value = low[index];
        let high_value = high[index];
        low[index] = low_value * cos_theta - high_value * sin_theta;
        high[index] = low_value * sin_theta + high_value * cos_theta;
    }
}

#[cfg(target_os = "macos")]
fn apply_interleaved_rope_in_place(values: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    let half_dim = cos_table.len();
    if values.len() != half_dim.saturating_mul(2) || sin_table.len() != half_dim {
        return;
    }

    for index in 0..half_dim {
        let pair_base = index.saturating_mul(2);
        let cos_theta = cos_table[index];
        let sin_theta = sin_table[index];
        let even = values[pair_base];
        let odd = values[pair_base + 1];
        values[pair_base] = even * cos_theta - odd * sin_theta;
        values[pair_base + 1] = odd * cos_theta + even * sin_theta;
    }
}

#[cfg(target_os = "macos")]
fn apply_rope_style_in_place(
    values: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    rope_style: ModelStageRopeStyle,
) {
    match rope_style {
        ModelStageRopeStyle::None => {}
        ModelStageRopeStyle::Neox => apply_split_half_rope_in_place(values, cos_table, sin_table),
        ModelStageRopeStyle::Interleaved => {
            apply_interleaved_rope_in_place(values, cos_table, sin_table)
        }
    }
}

#[cfg(target_os = "macos")]
fn apply_model_stage_rope_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    query: &[f32],
    key: &[f32],
    position: f32,
    stage_dims: ModelStageDims,
    rope_style: ModelStageRopeStyle,
    freq_base: f32,
    rotary_dim: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let bringup = bringup?;
    if !position.is_finite() || !freq_base.is_finite() || freq_base <= 0.0 {
        return None;
    }

    let head_dim = stage_dims.head_dim;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return None;
    }

    let query_len = stage_dims.q_heads.checked_mul(head_dim)?;
    let key_len = stage_dims.kv_heads.checked_mul(head_dim)?;
    if query.len() != query_len || key.len() != key_len {
        return None;
    }
    let pipeline_index = bringup.state.optional_kernel_dispatch_plan.apply_rope_f32?;
    let kernel_name = "apply_rope_f32";
    let feedback_key = rope_feedback_key(
        kernel_name,
        stage_dims.q_heads,
        stage_dims.kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }
    let (cos_table, sin_table) = build_model_stage_rope_tables(rotary_dim, position, freq_base);

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let query_buffer = new_shared_buffer_with_data(&bringup.state.device, query);
            let key_buffer = new_shared_buffer_with_data(&bringup.state.device, key);
            let cos_buffer = new_shared_buffer_with_data(&bringup.state.device, &cos_table);
            let sin_buffer = new_shared_buffer_with_data(&bringup.state.device, &sin_table);

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.apply_rope");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.apply_rope.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&query_buffer), 0);
            encoder.set_buffer(1, Some(&key_buffer), 0);
            encoder.set_buffer(2, Some(&cos_buffer), 0);
            encoder.set_buffer(3, Some(&sin_buffer), 0);
            set_model_stage_rope_dispatch_params(
                encoder,
                4,
                saturating_usize_to_u32(stage_dims.q_heads),
                saturating_usize_to_u32(stage_dims.kv_heads),
                saturating_usize_to_u32(head_dim),
                rope_style_dispatch_value(rope_style),
                saturating_usize_to_u32(rotary_dim),
            );
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let native_query =
                read_shared_buffer_prefix(&query_buffer, saturating_usize_to_u32(query_len));
            let native_key =
                read_shared_buffer_prefix(&key_buffer, saturating_usize_to_u32(key_len));
            (native_query.len() == query_len
                && native_key.len() == key_len
                && native_query.iter().all(|value| value.is_finite())
                && native_key.iter().all(|value| value.is_finite()))
            .then_some((native_query, native_key))
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
type BatchedModelStageRopeRows = (Vec<Vec<f32>>, Vec<Vec<f32>>);

#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
fn apply_batched_model_stage_rope_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    query_rows: &[Vec<f32>],
    key_rows: &[Vec<f32>],
    positions: &[u32],
    stage_dims: ModelStageDims,
    rope_style: ModelStageRopeStyle,
    freq_base: f32,
    rotary_dim: usize,
) -> Option<BatchedModelStageRopeRows> {
    let bringup = bringup?;
    if !freq_base.is_finite() || freq_base <= 0.0 {
        return None;
    }

    let head_dim = stage_dims.head_dim;
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
    {
        return None;
    }
    if query_rows.is_empty()
        || query_rows.len() != key_rows.len()
        || query_rows.len() != positions.len()
    {
        return None;
    }

    let token_count = query_rows.len();
    let query_len = stage_dims.q_heads.checked_mul(head_dim)?;
    let key_len = stage_dims.kv_heads.checked_mul(head_dim)?;
    if query_rows.iter().any(|row| row.len() != query_len)
        || key_rows.iter().any(|row| row.len() != key_len)
    {
        return None;
    }
    if token_count == 1 {
        let (native_query, native_key) = apply_model_stage_rope_with_optional_native_path(
            Some(bringup),
            query_rows.first()?,
            key_rows.first()?,
            *positions.first()? as f32,
            stage_dims,
            rope_style,
            freq_base,
            rotary_dim,
        )?;
        return Some((vec![native_query], vec![native_key]));
    }
    let pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .apply_rope_batched_f32?;
    let kernel_name = "apply_rope_batched_f32";
    let feedback_key = batched_rope_feedback_key(
        kernel_name,
        token_count,
        stage_dims.q_heads,
        stage_dims.kv_heads,
        head_dim,
        rotary_dim,
        rope_style,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }
    let mut flattened_query = Vec::with_capacity(token_count.checked_mul(query_len)?);
    let mut flattened_key = Vec::with_capacity(token_count.checked_mul(key_len)?);
    for row in query_rows {
        flattened_query.extend_from_slice(row);
    }
    for row in key_rows {
        flattened_key.extend_from_slice(row);
    }
    let positions = positions
        .iter()
        .map(|position| *position as f32)
        .collect::<Vec<_>>();

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let query_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_query);
            let key_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_key);
            let positions_buffer = new_shared_buffer_with_data(&bringup.state.device, &positions);

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.apply_rope_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.apply_rope_batched.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&query_buffer), 0);
            encoder.set_buffer(1, Some(&key_buffer), 0);
            encoder.set_buffer(2, Some(&positions_buffer), 0);
            set_batched_model_stage_rope_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(token_count),
                saturating_usize_to_u32(stage_dims.q_heads),
                saturating_usize_to_u32(stage_dims.kv_heads),
                saturating_usize_to_u32(head_dim),
                rope_style_dispatch_value(rope_style),
                freq_base,
                saturating_usize_to_u32(rotary_dim),
            );
            encoder.dispatch_threads(
                MTLSize::new(token_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(token_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let native_query = read_shared_buffer_prefix(
                &query_buffer,
                saturating_usize_to_u32(token_count.checked_mul(query_len)?),
            );
            let native_key = read_shared_buffer_prefix(
                &key_buffer,
                saturating_usize_to_u32(token_count.checked_mul(key_len)?),
            );
            if native_query.len() != token_count.checked_mul(query_len)?
                || native_key.len() != token_count.checked_mul(key_len)?
                || native_query.iter().any(|value| !value.is_finite())
                || native_key.iter().any(|value| !value.is_finite())
            {
                return None;
            }

            Some((
                native_query
                    .chunks_exact(query_len)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
                native_key
                    .chunks_exact(key_len)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            ))
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn rope_style_dispatch_value(rope_style: ModelStageRopeStyle) -> u32 {
    match rope_style {
        ModelStageRopeStyle::Neox => 0,
        ModelStageRopeStyle::Interleaved => 1,
        ModelStageRopeStyle::None => 2,
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
fn project_attention_qkv_with_dims_and_tally(
    artifacts: &NativeModelArtifacts,
    attention_qkv: &MetalAttentionQkvBindings,
    buffers: &MetalNativeModelBufferBindings,
    input: &[f32],
    stage_dims: ModelStageDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, PrefixAttentionExecutionTally)> {
    let query_output_dim = stage_dims.q_heads.saturating_mul(stage_dims.head_dim);
    let kv_output_dim = stage_dims.kv_head_size();

    match attention_qkv {
        MetalAttentionQkvBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let q_rows = artifacts.manifest().attention_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let packed_q_rows = if artifacts.manifest().attn_output_gate {
                q_rows.saturating_mul(2)
            } else {
                q_rows
            };
            let k_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let v_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            if q_rows < query_output_dim
                || packed_q_rows < q_rows
                || k_rows < kv_output_dim
                || v_rows < kv_output_dim
            {
                return None;
            }

            let (query, query_tally) = project_matrix_head_prefix_with_tally(
                packed,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            let (key, key_tally) = project_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            let (value, value_tally) = project_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows + k_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
        MetalAttentionQkvBindings::Split {
            q,
            k,
            v,
            value_from_key,
        } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let q_source_head_dim = if artifacts.manifest().attn_output_gate {
                stage_dims.head_dim.checked_mul(2)?
            } else {
                stage_dims.head_dim
            };
            let (query, query_tally) = project_matrix_head_prefix_with_tally(
                q_binding,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                q_source_head_dim,
                input,
                bringup,
            )?;
            let (key, key_tally) = project_matrix_head_prefix_with_tally(
                k_binding,
                0,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                stage_dims.head_dim,
                input,
                bringup,
            )?;
            let (value, value_tally) = if *value_from_key {
                (key.clone(), PrefixAttentionExecutionTally::default())
            } else {
                let v_binding = buffers.binding_for(v.as_ref()?)?;
                project_matrix_head_prefix_with_tally(
                    v_binding,
                    0,
                    stage_dims.kv_heads,
                    stage_dims.head_dim,
                    stage_dims.head_dim,
                    input,
                    bringup,
                )?
            };
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::type_complexity)]
fn project_batched_attention_qkv_with_dims_and_tally(
    artifacts: &NativeModelArtifacts,
    attention_qkv: &MetalAttentionQkvBindings,
    buffers: &MetalNativeModelBufferBindings,
    input_rows: &[Vec<f32>],
    input_width: usize,
    stage_dims: ModelStageDims,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    PrefixAttentionExecutionTally,
)> {
    let query_output_dim = stage_dims.q_heads.saturating_mul(stage_dims.head_dim);
    let kv_output_dim = stage_dims.kv_head_size();

    match attention_qkv {
        MetalAttentionQkvBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let q_rows = artifacts.manifest().attention_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let packed_q_rows = if artifacts.manifest().attn_output_gate {
                q_rows.saturating_mul(2)
            } else {
                q_rows
            };
            let k_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let v_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            if q_rows < query_output_dim
                || packed_q_rows < q_rows
                || k_rows < kv_output_dim
                || v_rows < kv_output_dim
            {
                return None;
            }

            let (query, query_tally) = project_batched_matrix_head_prefix_with_tally(
                packed,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            let (key, key_tally) = project_batched_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            let (value, value_tally) = project_batched_matrix_head_prefix_with_tally(
                packed,
                packed_q_rows + k_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
        MetalAttentionQkvBindings::Split {
            q,
            k,
            v,
            value_from_key,
        } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let q_source_head_dim = if artifacts.manifest().attn_output_gate {
                stage_dims.head_dim.checked_mul(2)?
            } else {
                stage_dims.head_dim
            };
            let (query, query_tally) = project_batched_matrix_head_prefix_with_tally(
                q_binding,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                q_source_head_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (key, key_tally) = project_batched_matrix_head_prefix_with_tally(
                k_binding,
                0,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                stage_dims.head_dim,
                input_rows,
                input_width,
                bringup,
            )?;
            let (value, value_tally) = if *value_from_key {
                (key.clone(), PrefixAttentionExecutionTally::default())
            } else {
                let v_binding = buffers.binding_for(v.as_ref()?)?;
                project_batched_matrix_head_prefix_with_tally(
                    v_binding,
                    0,
                    stage_dims.kv_heads,
                    stage_dims.head_dim,
                    stage_dims.head_dim,
                    input_rows,
                    input_width,
                    bringup,
                )?
            };
            Some((
                query,
                key,
                value,
                query_tally.merge(key_tally).merge(value_tally),
            ))
        }
    }
}

#[cfg(target_os = "macos")]
fn expand_grouped_kv_heads_cpu(
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Option<Vec<f32>> {
    if kv_heads == 0
        || q_heads == 0
        || head_dim == 0
        || q_heads < kv_heads
        || !q_heads.is_multiple_of(kv_heads)
        || values.len() != kv_heads.checked_mul(head_dim)?
    {
        return None;
    }

    if q_heads == kv_heads {
        return Some(values.to_vec());
    }

    let heads_per_kv = q_heads / kv_heads;
    let mut expanded = Vec::with_capacity(q_heads.checked_mul(head_dim)?);
    for kv_head in 0..kv_heads {
        let base = kv_head.checked_mul(head_dim)?;
        let end = base.checked_add(head_dim)?;
        let head = values.get(base..end)?;
        for _ in 0..heads_per_kv {
            expanded.extend_from_slice(head);
        }
    }
    Some(expanded)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn expand_grouped_kv_heads_with_path(
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<f32>> {
    if let Some(expanded) = expand_grouped_kv_heads_with_optional_native_path(
        bringup, values, q_heads, kv_heads, head_dim,
    ) {
        return Some(expanded);
    }

    expand_grouped_kv_heads_cpu(values, q_heads, kv_heads, head_dim)
}

#[cfg(target_os = "macos")]
fn expand_batched_grouped_kv_heads_with_path(
    rows: &[Vec<f32>],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Vec<f32>>> {
    if rows.is_empty() {
        return Some(Vec::new());
    }
    let input_row_width = kv_heads.checked_mul(head_dim)?;
    if rows.iter().any(|row| row.len() != input_row_width) {
        return None;
    }
    if rows.len() == 1 {
        return expand_grouped_kv_heads_with_path(
            rows.first()?,
            q_heads,
            kv_heads,
            head_dim,
            bringup,
        )
        .map(|row| vec![row]);
    }

    if let Some(expanded) = expand_batched_grouped_kv_heads_with_optional_native_path(
        rows, q_heads, kv_heads, head_dim, bringup,
    ) {
        return Some(expanded);
    }
    if rows.len() > 1 && batched_grouped_kv_expand_split_retry_worthwhile(bringup) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at(split_index);
        let mut left_expanded = expand_batched_grouped_kv_heads_with_path(
            left_rows, q_heads, kv_heads, head_dim, bringup,
        )?;
        let right_expanded = expand_batched_grouped_kv_heads_with_path(
            right_rows, q_heads, kv_heads, head_dim, bringup,
        )?;
        left_expanded.extend(right_expanded);
        return Some(left_expanded);
    }

    rows.iter()
        .map(|row| expand_grouped_kv_heads_cpu(row, q_heads, kv_heads, head_dim))
        .collect()
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn expand_grouped_kv_heads_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if kv_heads == 0
        || q_heads == 0
        || head_dim == 0
        || q_heads < kv_heads
        || !q_heads.is_multiple_of(kv_heads)
        || values.len() != kv_heads.checked_mul(head_dim)?
    {
        return None;
    }

    if q_heads == kv_heads {
        return Some(values.to_vec());
    }

    let output_element_count = q_heads.checked_mul(head_dim)?;
    let heads_per_kv = q_heads / kv_heads;
    let pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .expand_grouped_kv_heads_f32?;
    let kernel_name = "expand_grouped_kv_heads_f32";
    let feedback_key = grouped_kv_expand_feedback_key(kernel_name, q_heads, kv_heads, head_dim);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.expand_grouped_kv_heads");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.expand_grouped_kv_heads.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            set_grouped_kv_expand_dispatch_params(
                encoder,
                2,
                saturating_usize_to_u32(output_element_count),
                saturating_usize_to_u32(kv_heads),
                saturating_usize_to_u32(heads_per_kv),
                saturating_usize_to_u32(head_dim),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let expanded = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            (expanded.len() == output_element_count
                && expanded.iter().all(|value| value.is_finite()))
            .then_some(expanded)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn expand_batched_grouped_kv_heads_with_optional_native_path(
    rows: &[Vec<f32>],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    if rows.is_empty() || kv_heads == 0 || q_heads == 0 || head_dim == 0 {
        return None;
    }
    if q_heads < kv_heads || !q_heads.is_multiple_of(kv_heads) {
        return None;
    }
    let input_row_width = kv_heads.checked_mul(head_dim)?;
    if rows.iter().any(|row| row.len() != input_row_width) {
        return None;
    }

    if q_heads == kv_heads {
        return Some(rows.to_vec());
    }

    let token_count = rows.len();
    let flattened_input_len = token_count.checked_mul(input_row_width)?;
    let output_row_width = q_heads.checked_mul(head_dim)?;
    let output_element_count = token_count.checked_mul(output_row_width)?;
    let heads_per_kv = q_heads / kv_heads;
    let pipeline_index = bringup
        .state
        .optional_kernel_dispatch_plan
        .expand_grouped_kv_heads_f32?;
    let kernel_name = "expand_grouped_kv_heads_f32";
    let feedback_key = batched_grouped_kv_expand_feedback_key(
        kernel_name,
        token_count,
        q_heads,
        kv_heads,
        head_dim,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let mut flattened_input = Vec::with_capacity(flattened_input_len);
    for row in rows {
        flattened_input.extend_from_slice(row);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_input);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.expand_grouped_kv_heads_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.expand_grouped_kv_heads_batched.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            set_grouped_kv_expand_dispatch_params(
                encoder,
                2,
                saturating_usize_to_u32(output_element_count),
                saturating_usize_to_u32(token_count.checked_mul(kv_heads)?),
                saturating_usize_to_u32(heads_per_kv),
                saturating_usize_to_u32(head_dim),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let expanded = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if expanded.len() != output_element_count
                || expanded.iter().any(|value| !value.is_finite())
            {
                return None;
            }

            Some(
                expanded
                    .chunks_exact(output_row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn native_model_rms_norm_epsilon(artifacts: &NativeModelArtifacts) -> f32 {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("qwen") || family.starts_with("gemma") {
        1e-6
    } else {
        1e-5
    }
}

#[cfg(target_os = "macos")]
fn native_model_rms_norm_weight_offset(artifacts: &NativeModelArtifacts) -> f32 {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("gemma") {
        1.0
    } else {
        0.0
    }
}

#[cfg(target_os = "macos")]
fn native_model_embedding_scale(artifacts: &NativeModelArtifacts) -> f32 {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("gemma") {
        (artifacts.manifest().hidden_size as f32).sqrt()
    } else {
        1.0
    }
}

#[cfg(target_os = "macos")]
fn native_model_ffn_activation(artifacts: &NativeModelArtifacts) -> ModelFfnActivation {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("gemma") {
        ModelFfnActivation::GeluApprox
    } else {
        ModelFfnActivation::Silu
    }
}

#[cfg(target_os = "macos")]
fn native_model_rope_theta(artifacts: &NativeModelArtifacts) -> f32 {
    artifacts
        .manifest()
        .rope_theta
        .map(|rope_theta| rope_theta as f32)
        .unwrap_or(PHASE1_MODEL_STAGE_ROPE_FREQ_BASE)
}

#[cfg(target_os = "macos")]
fn native_model_reference_attention_config(
    artifacts: &NativeModelArtifacts,
    head_dim: usize,
) -> ReferenceAttentionConfig {
    let softmax_scale = artifacts
        .manifest()
        .query_pre_attn_scalar
        .map(|scalar| 1.0 / (scalar as f32).sqrt())
        .or_else(|| {
            ReferenceAttentionConfig::from_head_dim(head_dim).map(|config| config.softmax_scale)
        })
        .unwrap_or(1.0);
    let softcap = artifacts
        .manifest()
        .attention_logit_softcap
        .map(|softcap| softcap as f32);
    ReferenceAttentionConfig {
        softmax_scale,
        softcap,
    }
}

#[cfg(target_os = "macos")]
fn apply_rms_norm_with_weights_in_place(
    values: &mut [f32],
    weights: &[f32],
    epsilon: f32,
    weight_offset: f32,
) -> Option<()> {
    if values.len() != weights.len() || values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0
    {
        return None;
    }

    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    if !mean_square.is_finite() {
        return None;
    }

    let denom = (mean_square + epsilon).sqrt();
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }

    for (value, weight) in values.iter_mut().zip(weights.iter().copied()) {
        let normalized = (*value / denom) * (weight + weight_offset);
        if !normalized.is_finite() {
            return None;
        }
        *value = normalized;
    }

    Some(())
}

#[cfg(target_os = "macos")]
fn apply_rms_norm_without_weights_in_place(values: &mut [f32], epsilon: f32) -> Option<()> {
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }

    let mean_square = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    if !mean_square.is_finite() {
        return None;
    }

    let denom = (mean_square + epsilon).sqrt();
    if !denom.is_finite() || denom <= 0.0 {
        return None;
    }

    for value in values.iter_mut() {
        let normalized = *value / denom;
        if !normalized.is_finite() {
            return None;
        }
        *value = normalized;
    }

    Some(())
}

#[cfg(target_os = "macos")]
fn apply_rms_norm_without_weights_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    epsilon: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .rms_norm_kernel(NativeTensorDataType::F32)?;
    let feedback_key = rms_norm_feedback_key(kernel_name, values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let zero_weights = vec![0.0_f32; values.len()];
            let weight_buffer = new_shared_buffer_with_data(&bringup.state.device, &zero_weights);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(values.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm_no_weight");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm_no_weight.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(values.len()),
                epsilon,
                1.0,
            );
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(values.len()));
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn apply_rms_norm_with_binding_in_place(
    values: &mut [f32],
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<()> {
    apply_rms_norm_with_binding_in_place_with_path(
        values,
        weight_binding,
        epsilon,
        weight_offset,
        bringup,
    )
    .map(|_| ())
}

#[cfg(target_os = "macos")]
fn apply_rms_norm_with_binding_in_place_with_path(
    values: &mut [f32],
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if let Some(output) = apply_rms_norm_with_optional_native_path(
        bringup,
        values,
        weight_binding,
        epsilon,
        weight_offset,
    ) {
        values.copy_from_slice(&output);
        return Some(true);
    }

    let weights = tensor_prefix_f32(weight_binding, values.len())?;
    apply_rms_norm_with_weights_in_place(values, &weights, epsilon, weight_offset)?;
    round_slice_to_native_dtype(values, weight_binding.native_dtype);
    Some(false)
}

#[cfg(target_os = "macos")]
fn apply_rms_norm_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    let (rms_norm_kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .rms_norm_kernel(weight_binding.native_dtype)?;
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    let feedback_key = rms_norm_feedback_key(rms_norm_kernel_name, values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let weight_len = tensor_element_count(&weight_binding.meta.spec)?;
    if values.len() > weight_len {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        rms_norm_kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(values.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_binding.native_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(values.len()),
                epsilon,
                weight_offset,
            );
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(values.len()));
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn apply_per_head_rms_norm_with_binding_in_place(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<()> {
    apply_per_head_rms_norm_with_binding_in_place_with_tally(
        values,
        head_count,
        head_dim,
        weight_binding,
        epsilon,
        weight_offset,
        bringup,
    )
    .map(|_| ())
}

#[cfg(target_os = "macos")]
fn apply_per_head_rms_norm_with_binding_in_place_with_tally(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if head_count == 0 || head_dim == 0 || values.len() != head_count.checked_mul(head_dim)? {
        return None;
    }

    if let Some(output) = apply_batched_per_head_rms_norm_with_optional_native_path(
        bringup,
        values,
        head_count,
        head_dim,
        weight_binding,
        epsilon,
        weight_offset,
    ) {
        values.copy_from_slice(&output);
        return Some(
            PrefixAttentionExecutionTally::default().record_rms_norm_elements(values.len(), true),
        );
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    let single_native_bringup = single_rms_norm_retry_worthwhile(bringup, weight_binding, head_dim)
        .then_some(bringup)
        .flatten();
    for head in values.chunks_exact_mut(head_dim) {
        let used_native = apply_rms_norm_with_binding_in_place_with_path(
            head,
            weight_binding,
            epsilon,
            weight_offset,
            single_native_bringup,
        )?;
        tally = tally.record_rms_norm_elements(head.len(), used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
fn apply_batched_row_rms_norm_with_binding_in_place_with_tally(
    rows: &mut [Vec<f32>],
    row_width: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if row_width == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    if rows.is_empty() || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    let single_native_bringup =
        single_rms_norm_retry_worthwhile(bringup, weight_binding, row_width)
            .then_some(bringup)
            .flatten();
    if rows.len() == 1 && single_native_bringup.is_some() {
        let row = rows.first_mut()?.get_mut(..row_width)?;
        let used_native = apply_rms_norm_with_binding_in_place_with_path(
            row,
            weight_binding,
            epsilon,
            weight_offset,
            single_native_bringup,
        )?;
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_rms_norm_elements(row_width, used_native),
        );
    }

    let row_count = rows.len();
    let flattened_element_count = row_count.checked_mul(row_width)?;
    if let Some(bringup) = bringup {
        let allow_batched_native =
            batched_rms_norm_feedback_binding(bringup, weight_binding, row_count, row_width)
                .is_some_and(|(_, _, feedback_key)| {
                    optional_kernel_allowed(bringup, &feedback_key)
                });
        if allow_batched_native {
            let mut flattened = Vec::with_capacity(flattened_element_count);
            for row in rows.iter() {
                flattened.extend_from_slice(row.get(..row_width)?);
            }
            if let Some(output) = apply_batched_per_head_rms_norm_with_optional_native_path(
                Some(bringup),
                &flattened,
                row_count,
                row_width,
                weight_binding,
                epsilon,
                weight_offset,
            ) {
                for (row, normalized) in rows.iter_mut().zip(output.chunks_exact(row_width)) {
                    row.get_mut(..row_width)?.copy_from_slice(normalized);
                }
                return Some(
                    PrefixAttentionExecutionTally::default()
                        .record_rms_norm_elements(flattened_element_count, true),
                );
            }
        }
    }
    if rows.len() > 1 && batched_rms_norm_split_retry_worthwhile(bringup, weight_binding) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at_mut(split_index);
        let left_tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
            left_rows,
            row_width,
            weight_binding,
            epsilon,
            weight_offset,
            bringup,
        )?;
        let right_tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
            right_rows,
            row_width,
            weight_binding,
            epsilon,
            weight_offset,
            bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    for row in rows.iter_mut() {
        let used_native = apply_rms_norm_with_binding_in_place_with_path(
            row.get_mut(..row_width)?,
            weight_binding,
            epsilon,
            weight_offset,
            single_native_bringup,
        )?;
        tally = tally.record_rms_norm_elements(row_width, used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
fn apply_batched_per_head_rms_norm_rows_with_tally(
    rows: &mut [Vec<f32>],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if head_count == 0 || head_dim == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    let row_width = head_count.checked_mul(head_dim)?;
    if rows.is_empty() || rows.iter().any(|row| row.len() != row_width) {
        return None;
    }

    let total_heads = rows.len().checked_mul(head_count)?;
    let mut per_head_rows = Vec::with_capacity(total_heads);
    for row in rows.iter() {
        for head in row.chunks_exact(head_dim) {
            per_head_rows.push(head.to_vec());
        }
    }
    let tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut per_head_rows,
        head_dim,
        weight_binding,
        epsilon,
        weight_offset,
        bringup,
    )?;

    let mut normalized_heads = per_head_rows.into_iter();
    for row in rows.iter_mut() {
        for head in row.chunks_exact_mut(head_dim) {
            let normalized = normalized_heads.next()?;
            head.copy_from_slice(&normalized);
        }
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
fn apply_batched_per_head_rms_norm_rows_without_weights_with_tally(
    rows: &mut [Vec<f32>],
    head_count: usize,
    head_dim: usize,
    epsilon: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if head_count == 0 || head_dim == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    let row_width = head_count.checked_mul(head_dim)?;
    if rows.is_empty() || rows.iter().any(|row| row.len() != row_width) {
        return None;
    }

    let total_heads = rows.len().checked_mul(head_count)?;
    let mut per_head_rows = Vec::with_capacity(total_heads);
    for row in rows.iter() {
        for head in row.chunks_exact(head_dim) {
            per_head_rows.push(head.to_vec());
        }
    }

    let tally = apply_batched_row_rms_norm_without_weights_in_place_with_tally(
        &mut per_head_rows,
        head_dim,
        epsilon,
        bringup,
    )?;

    let mut normalized_heads = per_head_rows.into_iter();
    for row in rows.iter_mut() {
        for head in row.chunks_exact_mut(head_dim) {
            let normalized = normalized_heads.next()?;
            head.copy_from_slice(&normalized);
        }
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
fn apply_batched_row_rms_norm_without_weights_in_place_with_tally(
    rows: &mut [Vec<f32>],
    row_width: usize,
    epsilon: f32,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if row_width == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    if rows.is_empty() || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    let single_native_bringup =
        single_rms_norm_without_weights_retry_worthwhile(bringup, row_width)
            .then_some(bringup)
            .flatten();
    if rows.len() == 1 && single_native_bringup.is_some() {
        let row = rows.first_mut()?.get_mut(..row_width)?;
        let used_native = if let Some(output) =
            apply_rms_norm_without_weights_with_optional_native_path(
                single_native_bringup,
                row,
                epsilon,
            ) {
            row.copy_from_slice(&output);
            true
        } else {
            apply_rms_norm_without_weights_in_place(row, epsilon)?;
            false
        };
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_rms_norm_elements(row_width, used_native),
        );
    }

    let row_count = rows.len();
    let flattened_element_count = row_count.checked_mul(row_width)?;
    if let Some(bringup) = bringup {
        let (kernel_name, _) = bringup
            .state
            .optional_kernel_dispatch_plan
            .batched_rms_norm_kernel(NativeTensorDataType::F32)?;
        let feedback_key = batched_rms_norm_feedback_key(kernel_name, row_count, row_width);
        if optional_kernel_allowed(bringup, &feedback_key) {
            let mut flattened = Vec::with_capacity(flattened_element_count);
            for row in rows.iter() {
                flattened.extend_from_slice(row.get(..row_width)?);
            }
            if let Some(output) =
                apply_batched_per_head_rms_norm_without_weights_with_optional_native_path(
                    Some(bringup),
                    &flattened,
                    row_count,
                    row_width,
                    epsilon,
                )
            {
                for (row, normalized) in rows.iter_mut().zip(output.chunks_exact(row_width)) {
                    row.get_mut(..row_width)?.copy_from_slice(normalized);
                }
                return Some(
                    PrefixAttentionExecutionTally::default()
                        .record_rms_norm_elements(flattened_element_count, true),
                );
            }
        }
    }
    if rows.len() > 1 && batched_rms_norm_without_weights_split_retry_worthwhile(bringup) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at_mut(split_index);
        let left_tally = apply_batched_row_rms_norm_without_weights_in_place_with_tally(
            left_rows, row_width, epsilon, bringup,
        )?;
        let right_tally = apply_batched_row_rms_norm_without_weights_in_place_with_tally(
            right_rows, row_width, epsilon, bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    for row in rows.iter_mut() {
        let used_native = if let Some(output) =
            apply_rms_norm_without_weights_with_optional_native_path(
                single_native_bringup,
                row.get(..row_width)?,
                epsilon,
            ) {
            row.get_mut(..row_width)?.copy_from_slice(&output);
            true
        } else {
            apply_rms_norm_without_weights_in_place(row.get_mut(..row_width)?, epsilon)?;
            false
        };
        tally = tally.record_rms_norm_elements(row_width, used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
fn apply_batched_per_head_rms_norm_without_weights_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    head_count: usize,
    head_dim: usize,
    epsilon: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if head_count == 0
        || head_dim == 0
        || values.len() != head_count.checked_mul(head_dim)?
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_rms_norm_kernel(NativeTensorDataType::F32)?;
    let feedback_key = batched_rms_norm_feedback_key(kernel_name, head_count, head_dim);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let element_count = saturating_usize_to_u32(values.len());
    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let zero_weights = vec![0.0_f32; head_dim];
            let weight_buffer = new_shared_buffer_with_data(&bringup.state.device, &zero_weights);
            let output_buffer =
                new_zeroed_shared_buffer::<f32>(&bringup.state.device, element_count.max(1));

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm_batched_no_weight");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm_batched_no_weight.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(head_count),
                saturating_usize_to_u32(head_dim),
                epsilon,
                1.0,
            );
            encoder.dispatch_threads(
                MTLSize::new(u64::from(element_count.max(1)), 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(u64::from(element_count.max(1))),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output = read_shared_buffer_prefix(&output_buffer, element_count);
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn apply_batched_per_head_rms_norm_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    head_count: usize,
    head_dim: usize,
    weight_binding: &MetalNativeTensorBufferBinding,
    epsilon: f32,
    weight_offset: f32,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if head_count == 0
        || head_dim == 0
        || values.len() != head_count.checked_mul(head_dim)?
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return None;
    }

    let weight_len = tensor_element_count(&weight_binding.meta.spec)?;
    if head_dim > weight_len {
        return None;
    }

    let (kernel_name, pipeline_index, feedback_key) =
        batched_rms_norm_feedback_binding(bringup, weight_binding, head_count, head_dim)?;
    let element_count = saturating_usize_to_u32(values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let output_buffer =
                new_zeroed_shared_buffer::<f32>(&bringup.state.device, element_count.max(1));

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.rms_norm_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.rms_norm_batched.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_binding.native_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_rms_norm_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(head_count),
                saturating_usize_to_u32(head_dim),
                epsilon,
                weight_offset,
            );
            encoder.dispatch_threads(
                MTLSize::new(u64::from(element_count.max(1)), 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(u64::from(element_count.max(1))),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output = read_shared_buffer_prefix(&output_buffer, element_count);
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelFfnActivation {
    Silu,
    GeluApprox,
}

#[cfg(target_os = "macos")]
fn add_in_place(values: &mut [f32], delta: &[f32]) {
    for (value, addition) in values.iter_mut().zip(delta.iter()) {
        *value += *addition;
    }
}

#[cfg(target_os = "macos")]
fn apply_vector_add_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    values: &[f32],
    delta: &[f32],
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if values.is_empty() || values.len() != delta.len() {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .vector_add_kernel()?;
    let feedback_key = vector_add_feedback_key(kernel_name, values.len());
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, values);
            let delta_buffer = new_shared_buffer_with_data(&bringup.state.device, delta);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(values.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.vector_add");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.vector_add.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&delta_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_vector_add_dispatch_params(encoder, 3, saturating_usize_to_u32(values.len()));
            encoder.dispatch_threads(
                MTLSize::new(values.len().max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(values.len().max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(values.len()));
            (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn apply_batched_row_scale_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    rows: &[Vec<f32>],
    row_width: usize,
    row_scales: &[f32],
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    if row_width == 0 {
        return Some(vec![Vec::new(); rows.len()]);
    }
    if rows.is_empty()
        || rows.len() != row_scales.len()
        || rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .row_scale_kernel()?;
    let feedback_key = batched_row_scale_feedback_key(kernel_name, rows.len(), row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let element_count = rows.len().checked_mul(row_width)?;
    let mut flattened_rows = Vec::with_capacity(element_count);
    for row in rows {
        flattened_rows.extend_from_slice(row.get(..row_width)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_rows);
            let scale_buffer = new_shared_buffer_with_data(&bringup.state.device, row_scales);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.row_scale");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.row_scale.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&scale_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_row_scale_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(rows.len()),
                saturating_usize_to_u32(row_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }
            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn apply_batched_row_vector_scale_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    rows: &[Vec<f32>],
    row_width: usize,
    scales: &[f32],
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    if row_width == 0 {
        return Some(vec![Vec::new(); rows.len()]);
    }
    if rows.is_empty() || scales.len() < row_width || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .row_vector_scale_kernel()?;
    let feedback_key = batched_row_vector_scale_feedback_key(kernel_name, rows.len(), row_width);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let element_count = rows.len().checked_mul(row_width)?;
    let mut flattened_rows = Vec::with_capacity(element_count);
    for row in rows {
        flattened_rows.extend_from_slice(row.get(..row_width)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let input_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_rows);
            let scale_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &scales[..row_width]);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.row_vector_scale");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.row_vector_scale.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&scale_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_row_vector_scale_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(rows.len()),
                saturating_usize_to_u32(row_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(element_count));
            if output.len() != element_count || output.iter().any(|value| !value.is_finite()) {
                return None;
            }
            Some(
                output
                    .chunks_exact(row_width)
                    .map(|chunk| chunk.to_vec())
                    .collect::<Vec<_>>(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn apply_batched_row_vector_scale_in_place_with_path(
    rows: &mut [Vec<f32>],
    row_width: usize,
    scales: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if row_width == 0 {
        return Some(false);
    }
    if rows.is_empty() || scales.len() < row_width || rows.iter().any(|row| row.len() < row_width) {
        return None;
    }

    if let Some(output_rows) =
        apply_batched_row_vector_scale_with_optional_native_path(bringup, rows, row_width, scales)
    {
        for (row, output_row) in rows.iter_mut().zip(output_rows) {
            row.get_mut(..row_width)?.copy_from_slice(&output_row);
        }
        return Some(true);
    }

    for row in rows {
        for (value, scale) in row.get_mut(..row_width)?.iter_mut().zip(scales.iter()) {
            *value *= *scale;
        }
    }
    Some(false)
}

#[cfg(target_os = "macos")]
fn apply_batched_row_scale_in_place_with_path(
    rows: &mut [Vec<f32>],
    row_width: usize,
    row_scales: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if row_width == 0 {
        return Some(false);
    }
    if rows.is_empty()
        || rows.len() != row_scales.len()
        || rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    if let Some(output_rows) =
        apply_batched_row_scale_with_optional_native_path(bringup, rows, row_width, row_scales)
    {
        for (row, output_row) in rows.iter_mut().zip(output_rows) {
            row.get_mut(..row_width)?.copy_from_slice(&output_row);
        }
        return Some(true);
    }

    for (row, scale) in rows.iter_mut().zip(row_scales.iter().copied()) {
        for value in row.get_mut(..row_width)? {
            *value *= scale;
        }
    }
    Some(false)
}

#[cfg(target_os = "macos")]
fn add_in_place_with_path_and_result_dtype(
    values: &mut [f32],
    delta: &[f32],
    result_dtype: NativeTensorDataType,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    if values.len() != delta.len() {
        return None;
    }
    if let Some(output) = apply_vector_add_with_optional_native_path(bringup, values, delta) {
        values.copy_from_slice(&output);
        return Some(true);
    }
    add_in_place(values, delta);
    round_slice_to_native_dtype(values, result_dtype);
    Some(false)
}

#[cfg(target_os = "macos")]
fn add_batched_rows_in_place_with_direct_decode_tally_and_result_dtype(
    rows: &mut [Vec<f32>],
    delta_rows: &[Vec<f32>],
    row_width: usize,
    result_dtype: NativeTensorDataType,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<DirectDecodeNativeDenseTally> {
    Some(
        direct_decode_native_dense_tally_from_prefix_attention_tally(
            add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
                rows,
                delta_rows,
                row_width,
                result_dtype,
                bringup,
            )?,
        ),
    )
}

#[cfg(target_os = "macos")]
fn add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
    rows: &mut [Vec<f32>],
    delta_rows: &[Vec<f32>],
    row_width: usize,
    result_dtype: NativeTensorDataType,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<PrefixAttentionExecutionTally> {
    if row_width == 0 {
        return Some(PrefixAttentionExecutionTally::default());
    }
    if rows.len() != delta_rows.len()
        || rows.iter().any(|row| row.len() < row_width)
        || delta_rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    let single_native_bringup = single_vector_add_retry_worthwhile(bringup, row_width)
        .then_some(bringup)
        .flatten();
    if rows.len() == 1 && single_native_bringup.is_some() {
        let used_native = add_in_place_with_path_and_result_dtype(
            rows.first_mut()?.get_mut(..row_width)?,
            delta_rows.first()?.get(..row_width)?,
            result_dtype,
            single_native_bringup,
        )?;
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_residual_add_elements(row_width, used_native),
        );
    }

    let row_count = rows.len();
    let element_count = row_count.checked_mul(row_width)?;
    let mut flattened_rows = Vec::with_capacity(element_count);
    let mut flattened_deltas = Vec::with_capacity(element_count);
    for (row, delta_row) in rows.iter().zip(delta_rows.iter()) {
        flattened_rows.extend_from_slice(row.get(..row_width)?);
        flattened_deltas.extend_from_slice(delta_row.get(..row_width)?);
    }

    if let Some(output) =
        apply_vector_add_with_optional_native_path(bringup, &flattened_rows, &flattened_deltas)
    {
        for (row, output_row) in rows.iter_mut().zip(output.chunks_exact(row_width)) {
            row.get_mut(..row_width)?.copy_from_slice(output_row);
        }
        return Some(
            PrefixAttentionExecutionTally::default()
                .record_residual_add_elements(element_count, true),
        );
    }
    if rows.len() > 1 && batched_vector_add_split_retry_worthwhile(bringup) {
        let split_index = rows.len() / 2;
        let (left_rows, right_rows) = rows.split_at_mut(split_index);
        let (left_delta_rows, right_delta_rows) = delta_rows.split_at(split_index);
        let left_tally = add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
            left_rows,
            left_delta_rows,
            row_width,
            result_dtype,
            bringup,
        )?;
        let right_tally = add_batched_rows_in_place_with_prefix_tally_and_result_dtype(
            right_rows,
            right_delta_rows,
            row_width,
            result_dtype,
            bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = PrefixAttentionExecutionTally::default();
    for (row, delta_row) in rows.iter_mut().zip(delta_rows.iter()) {
        let used_native = add_in_place_with_path_and_result_dtype(
            row.get_mut(..row_width)?,
            delta_row.get(..row_width)?,
            result_dtype,
            single_native_bringup,
        )?;
        tally = tally.record_residual_add_elements(row_width, used_native);
    }
    Some(tally)
}

#[cfg(target_os = "macos")]
fn apply_model_gate_up_product(artifacts: &NativeModelArtifacts, gate: &mut [f32], up: &[f32]) {
    let activation = native_model_ffn_activation(artifacts);
    for (gate_value, up_value) in gate.iter_mut().zip(up.iter()) {
        let activated = match activation {
            ModelFfnActivation::Silu => silu(*gate_value),
            ModelFfnActivation::GeluApprox => gelu_approx(*gate_value),
        };
        *gate_value = activated * *up_value;
    }
}

#[cfg(target_os = "macos")]
fn apply_model_gate_up_product_with_path(
    artifacts: &NativeModelArtifacts,
    gate: &mut [f32],
    up: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<bool> {
    let activation = native_model_ffn_activation(artifacts);
    if let Some(output) =
        apply_model_gate_up_product_with_optional_native_path(bringup, activation, gate, up, None)
    {
        gate.copy_from_slice(&output);
        return Some(true);
    }

    apply_model_gate_up_product(artifacts, gate, up);
    Some(false)
}

#[cfg(target_os = "macos")]
fn apply_batched_model_gate_up_product_in_place_with_tally(
    artifacts: &NativeModelArtifacts,
    gate_rows: &mut [Vec<f32>],
    up_rows: &[Vec<f32>],
    row_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<DirectDecodeNativeDenseTally> {
    if row_width == 0 {
        return Some(DirectDecodeNativeDenseTally::default());
    }
    if gate_rows.len() != up_rows.len()
        || gate_rows.iter().any(|row| row.len() < row_width)
        || up_rows.iter().any(|row| row.len() < row_width)
    {
        return None;
    }

    let row_count = gate_rows.len();
    let flattened_len = row_count.checked_mul(row_width)?;
    let activation = native_model_ffn_activation(artifacts);
    let single_native_bringup =
        single_ffn_gate_product_retry_worthwhile(bringup, activation, row_width)
            .then_some(bringup)
            .flatten();
    if gate_rows.len() == 1 && single_native_bringup.is_some() {
        let used_native = apply_model_gate_up_product_with_path(
            artifacts,
            gate_rows.first_mut()?.get_mut(..row_width)?,
            up_rows.first()?.get(..row_width)?,
            single_native_bringup,
        )?;
        return Some(
            DirectDecodeNativeDenseTally::default()
                .record_ffn_activation_elements(row_width, used_native),
        );
    }

    if let Some(bringup) = bringup {
        let allow_batched_native =
            ffn_gate_product_feedback_binding(bringup, activation, row_count, row_width)
                .is_some_and(|(_, _, feedback_key)| {
                    optional_kernel_allowed(bringup, &feedback_key)
                });
        if allow_batched_native {
            let mut flattened_gate = Vec::with_capacity(flattened_len);
            let mut flattened_up = Vec::with_capacity(flattened_len);
            for (gate_row, up_row) in gate_rows.iter().zip(up_rows.iter()) {
                flattened_gate.extend_from_slice(gate_row.get(..row_width)?);
                flattened_up.extend_from_slice(up_row.get(..row_width)?);
            }

            if let Some(output) = apply_model_gate_up_product_with_optional_native_path(
                Some(bringup),
                activation,
                &flattened_gate,
                &flattened_up,
                Some((row_count, row_width)),
            ) {
                for (gate_row, output_row) in
                    gate_rows.iter_mut().zip(output.chunks_exact(row_width))
                {
                    gate_row.get_mut(..row_width)?.copy_from_slice(output_row);
                }
                return Some(
                    DirectDecodeNativeDenseTally::default()
                        .record_ffn_activation_elements(flattened_len, true),
                );
            }
        }
    }
    if gate_rows.len() > 1 && batched_ffn_gate_product_split_retry_worthwhile(bringup, activation) {
        let split_index = gate_rows.len() / 2;
        let (left_gate_rows, right_gate_rows) = gate_rows.split_at_mut(split_index);
        let (left_up_rows, right_up_rows) = up_rows.split_at(split_index);
        let left_tally = apply_batched_model_gate_up_product_in_place_with_tally(
            artifacts,
            left_gate_rows,
            left_up_rows,
            row_width,
            bringup,
        )?;
        let right_tally = apply_batched_model_gate_up_product_in_place_with_tally(
            artifacts,
            right_gate_rows,
            right_up_rows,
            row_width,
            bringup,
        )?;
        return Some(left_tally.merge(right_tally));
    }

    let mut tally = DirectDecodeNativeDenseTally::default();
    for (gate_row, up_row) in gate_rows.iter_mut().zip(up_rows.iter()) {
        let used_native = apply_model_gate_up_product_with_path(
            artifacts,
            gate_row.get_mut(..row_width)?,
            up_row.get(..row_width)?,
            single_native_bringup,
        )?;
        tally = tally.record_ffn_activation_elements(row_width, used_native);
    }

    Some(tally)
}

#[cfg(target_os = "macos")]
fn apply_model_gate_up_product_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    activation: ModelFfnActivation,
    gate: &[f32],
    up: &[f32],
    batched_shape: Option<(usize, usize)>,
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if gate.is_empty() || gate.len() != up.len() {
        return None;
    }

    let (kernel_name, pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .ffn_gate_product_kernel(activation)?;
    let feedback_key = batched_shape
        .map(|(row_count, row_width)| {
            batched_ffn_gate_product_feedback_key(kernel_name, row_count, row_width)
        })
        .unwrap_or_else(|| ffn_gate_product_feedback_key(kernel_name, gate.len()));
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        kernel_name,
        pipeline_index,
    )
    .ok()
    .and_then(|pipeline| {
        autoreleasepool(|| {
            let gate_buffer = new_shared_buffer_with_data(&bringup.state.device, gate);
            let up_buffer = new_shared_buffer_with_data(&bringup.state.device, up);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(gate.len()),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.ffn_gate_product");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.ffn_gate_product.compute");

            encoder.set_compute_pipeline_state(&pipeline.pipeline);
            encoder.set_buffer(0, Some(&gate_buffer), 0);
            encoder.set_buffer(1, Some(&up_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_ffn_gate_product_dispatch_params(encoder, 3, saturating_usize_to_u32(gate.len()));
            encoder.dispatch_threads(
                MTLSize::new(gate.len().max(1) as u64, 1, 1),
                MTLSize::new(
                    pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(gate.len().max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(gate.len()));
            (output.len() == gate.len() && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(target_os = "macos")]
fn gelu_approx(value: f32) -> f32 {
    let cubic = value * value * value;
    let inner = (0.797_884_6_f32 * (value + 0.044_715_f32 * cubic)).tanh();
    0.5 * value * (1.0 + inner)
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn project_matrix_rows(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<f32>> {
    project_matrix_rows_with_path(binding, row_offset, output_dim, input, bringup)
        .map(|(output, _)| output)
}

#[cfg(target_os = "macos")]
fn project_matrix_rows_with_path(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, bool)> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if row_offset.checked_add(output_dim)? > rows || input.len() > cols {
        return None;
    }

    if let Some(output) = project_matrix_rows_with_optional_native_path(
        bringup, binding, row_offset, output_dim, input,
    ) {
        return Some((output, true));
    }

    let mut output = Vec::with_capacity(output_dim);
    for row in row_offset..row_offset + output_dim {
        let weights = tensor_matrix_row_prefix_f32(binding, row, input.len())?;
        output.push(dot_product(&weights, input));
    }
    round_slice_to_native_dtype(&mut output, binding.native_dtype);
    Some((output, false))
}

#[cfg(target_os = "macos")]
fn project_batched_matrix_rows_with_tally(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, DirectDecodeNativeDenseTally)> {
    if input_rows.is_empty() {
        return Some((Vec::new(), DirectDecodeNativeDenseTally::default()));
    }
    if input_rows.iter().any(|row| row.len() < input_width) {
        return None;
    }

    let single_native_bringup =
        single_projection_retry_worthwhile(bringup, binding, output_dim, input_width)
            .then_some(bringup)
            .flatten();
    if input_rows.len() == 1 && single_native_bringup.is_some() {
        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            row_offset,
            output_dim,
            input_rows.first()?.get(..input_width)?,
            single_native_bringup,
        )?;
        return Some((
            vec![projected],
            DirectDecodeNativeDenseTally::default().record_projection_rows(output_dim, used_native),
        ));
    }

    if let Some(output_rows) = project_batched_matrix_rows_with_optional_native_path(
        bringup,
        binding,
        row_offset,
        output_dim,
        input_rows,
        input_width,
    ) {
        return Some((
            output_rows,
            DirectDecodeNativeDenseTally::default()
                .record_projection_rows(input_rows.len().checked_mul(output_dim)?, true),
        ));
    }
    if input_rows.len() > 1 && batched_projection_split_retry_worthwhile(bringup, binding) {
        let split_index = input_rows.len() / 2;
        let (left_rows, right_rows) = input_rows.split_at(split_index);
        let (mut left_projected_rows, left_tally) = project_batched_matrix_rows_with_tally(
            binding,
            row_offset,
            output_dim,
            left_rows,
            input_width,
            bringup,
        )?;
        let (right_projected_rows, right_tally) = project_batched_matrix_rows_with_tally(
            binding,
            row_offset,
            output_dim,
            right_rows,
            input_width,
            bringup,
        )?;
        left_projected_rows.extend(right_projected_rows);
        return Some((left_projected_rows, left_tally.merge(right_tally)));
    }

    let mut projected_rows = Vec::with_capacity(input_rows.len());
    let mut tally = DirectDecodeNativeDenseTally::default();
    for row in input_rows {
        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            row_offset,
            output_dim,
            row.get(..input_width)?,
            single_native_bringup,
        )?;
        tally = tally.record_projection_rows(output_dim, used_native);
        projected_rows.push(projected);
    }
    Some((projected_rows, tally))
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn project_matrix_head_prefix(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    head_count: usize,
    projected_head_dim: usize,
    full_head_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<Vec<f32>> {
    project_matrix_head_prefix_with_tally(
        binding,
        row_offset,
        head_count,
        projected_head_dim,
        full_head_dim,
        input,
        bringup,
    )
    .map(|(output, _)| output)
}

#[cfg(target_os = "macos")]
fn project_matrix_head_prefix_with_tally(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    head_count: usize,
    projected_head_dim: usize,
    full_head_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, PrefixAttentionExecutionTally)> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input.len() > cols || projected_head_dim > full_head_dim {
        return None;
    }

    if projected_head_dim == full_head_dim {
        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            row_offset,
            head_count.checked_mul(projected_head_dim)?,
            input,
            bringup,
        )?;
        let tally = PrefixAttentionExecutionTally::default()
            .record_projection_rows(head_count.checked_mul(projected_head_dim)?, used_native);
        return Some((projected, tally));
    }

    let mut output = Vec::with_capacity(head_count.checked_mul(projected_head_dim)?);
    let mut tally = PrefixAttentionExecutionTally::default();
    let single_native_bringup =
        single_projection_retry_worthwhile(bringup, binding, projected_head_dim, input.len())
            .then_some(bringup)
            .flatten();
    for head in 0..head_count {
        let head_row_offset = row_offset.checked_add(head.checked_mul(full_head_dim)?)?;
        if head_row_offset.checked_add(projected_head_dim)? > rows {
            return None;
        }

        let (projected, used_native) = project_matrix_rows_with_path(
            binding,
            head_row_offset,
            projected_head_dim,
            input,
            single_native_bringup,
        )?;
        tally = tally.record_projection_rows(projected_head_dim, used_native);
        output.extend(projected);
    }
    Some((output, tally))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn project_batched_matrix_head_prefix_with_tally(
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    head_count: usize,
    projected_head_dim: usize,
    full_head_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input_width > cols || projected_head_dim > full_head_dim {
        return None;
    }
    if input_rows.is_empty() || input_rows.iter().any(|row| row.len() < input_width) {
        return None;
    }

    if projected_head_dim == full_head_dim {
        let (projected, tally) = project_batched_matrix_rows_with_tally(
            binding,
            row_offset,
            head_count.checked_mul(projected_head_dim)?,
            input_rows,
            input_width,
            bringup,
        )?;
        return Some((
            projected,
            prefix_attention_tally_from_native_dense_tally(tally),
        ));
    }

    let row_count = input_rows.len();
    let output_width = head_count.checked_mul(projected_head_dim)?;

    // Build multi-projection tasks for all heads.
    let mut head_tasks = Vec::with_capacity(head_count);
    for head in 0..head_count {
        let head_row_offset = row_offset.checked_add(head.checked_mul(full_head_dim)?)?;
        if head_row_offset.checked_add(projected_head_dim)? > rows {
            return None;
        }
        head_tasks.push(MultiProjectionTask {
            binding,
            row_offset: head_row_offset,
            output_dim: projected_head_dim,
        });
    }

    // Try multi-projection batch: all heads in a single command buffer.
    if let Some(bringup) = bringup {
        let multi_result = if row_count == 1 {
            let input = input_rows.first()?.get(..input_width)?;
            project_multi_matrix_rows_with_optional_native_path(
                bringup,
                &head_tasks,
                input,
                input_width,
            )
            .map(|outputs| outputs.into_iter().map(|v| vec![v]).collect::<Vec<_>>())
        } else {
            project_multi_batched_matrix_rows_with_optional_native_path(
                bringup,
                &head_tasks,
                input_rows,
                input_width,
            )
        };

        if let Some(head_outputs) = multi_result {
            if head_outputs.len() == head_count {
                let total_rows = row_count.checked_mul(output_width)?;
                let tally = prefix_attention_tally_from_native_dense_tally(
                    DirectDecodeNativeDenseTally::default()
                        .record_projection_rows(total_rows, true),
                );
                let mut output_rows = vec![Vec::with_capacity(output_width); row_count];
                for head_rows in head_outputs {
                    for (output_row, head_row) in output_rows.iter_mut().zip(head_rows) {
                        output_row.extend(head_row);
                    }
                }
                return Some((output_rows, tally));
            }
        }
    }

    // Fallback: individual dispatches per head.
    let mut output_rows = vec![Vec::with_capacity(output_width); row_count];
    let mut tally = PrefixAttentionExecutionTally::default();
    for head in 0..head_count {
        let head_row_offset = row_offset.checked_add(head.checked_mul(full_head_dim)?)?;
        if head_row_offset.checked_add(projected_head_dim)? > rows {
            return None;
        }

        let (projected_rows, projected_tally) = project_batched_matrix_rows_with_tally(
            binding,
            head_row_offset,
            projected_head_dim,
            input_rows,
            input_width,
            bringup,
        )?;
        tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
            projected_tally,
        ));
        for (output_row, projected_row) in output_rows.iter_mut().zip(projected_rows) {
            output_row.extend(projected_row);
        }
    }
    Some((output_rows, tally))
}

#[cfg(target_os = "macos")]
fn project_batched_matrix_rows_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input_rows: &[Vec<f32>],
    input_width: usize,
) -> Option<Vec<Vec<f32>>> {
    let bringup = bringup?;
    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel(binding.native_dtype)?;

    let (_, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input_width > cols || output_dim == 0 || input_rows.is_empty() {
        return None;
    }

    let row_count = input_rows.len();
    let hidden_stride = input_width;
    let feedback_key = batched_projection_feedback_key(
        projection_kernel_name,
        row_count,
        output_dim,
        input_width,
        hidden_stride,
        cols,
    );
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let row_byte_offset = row_offset
        .checked_mul(cols)?
        .checked_mul(native_dtype_size_bytes(binding.native_dtype))?;
    let output_element_count = row_count.checked_mul(output_dim)?;
    let mut flattened_input = Vec::with_capacity(row_count.checked_mul(hidden_stride)?);
    for row in input_rows {
        flattened_input.extend_from_slice(row.get(..input_width)?);
    }

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .and_then(|projection_pipeline| {
        autoreleasepool(|| {
            let hidden_buffer =
                new_shared_buffer_with_data(&bringup.state.device, &flattened_input);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_element_count),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.project_matrix_rows_batched");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.project_matrix_rows_batched.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_batched_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(row_count),
                saturating_usize_to_u32(output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input_width),
                saturating_usize_to_u32(hidden_stride),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output = read_shared_buffer_prefix(
                &output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if output.len() != output_element_count || output.iter().any(|value| !value.is_finite())
            {
                return None;
            }
            Some(
                output
                    .chunks_exact(output_dim)
                    .map(|chunk| chunk.to_vec())
                    .collect(),
            )
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

/// A single projection task within a multi-projection batch dispatch.
/// All tasks share the same input buffer but project through different weight matrices.
#[cfg(target_os = "macos")]
struct MultiProjectionTask<'a> {
    binding: &'a MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
}

/// Dispatches multiple independent matrix projections in a single Metal command buffer.
///
/// Each task reads from the same input row(s) but uses a different weight matrix and
/// produces an independent output. This eliminates per-projection command buffer
/// overhead when projections are data-independent (e.g. Q/K/V or FFN gate/up).
///
/// Returns `None` if any task cannot be dispatched natively (caller falls back to
/// individual dispatches). Follows the engine's feedback-key pattern: each task's
/// kernel is checked against the feedback state before attempting dispatch.
#[cfg(target_os = "macos")]
fn project_multi_matrix_rows_with_optional_native_path(
    bringup: &MetalRuntimeBringup,
    tasks: &[MultiProjectionTask<'_>],
    input: &[f32],
    input_width: usize,
) -> Option<Vec<Vec<f32>>> {
    if tasks.is_empty() {
        return Some(Vec::new());
    }

    // All tasks must use the same dtype (same projection kernel).
    let first_dtype = tasks[0].binding.native_dtype;
    if tasks.iter().any(|t| t.binding.native_dtype != first_dtype) {
        return None;
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel(first_dtype)?;

    // Validate every task and check feedback before creating any Metal resources.
    for task in tasks {
        let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
        if input_width > cols || task.output_dim == 0 {
            return None;
        }
        if task.row_offset.checked_add(task.output_dim)?
            > tensor_matrix_dimensions(&task.binding.meta.spec)?.0
        {
            return None;
        }
        let feedback_key =
            projection_feedback_key(projection_kernel_name, task.output_dim, input_width, cols);
        if !optional_kernel_allowed(bringup, &feedback_key) {
            return None;
        }
    }

    let projection_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()?;

    autoreleasepool(|| {
        let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, input);

        // Pre-allocate one output buffer per task.
        let output_buffers: Vec<_> = tasks
            .iter()
            .map(|task| {
                new_zeroed_shared_buffer::<f32>(
                    &bringup.state.device,
                    saturating_usize_to_u32(task.output_dim),
                )
            })
            .collect();

        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.project_matrix_rows.multi");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("ax.phase1.project_matrix_rows.multi.compute");

        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
            let row_byte_offset = task
                .row_offset
                .checked_mul(cols)?
                .checked_mul(native_dtype_size_bytes(task.binding.native_dtype))?;

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&task.binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(output_buffer), 0);
            set_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(task.output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input_width),
            );
            encoder.dispatch_threads(
                MTLSize::new(task.output_dim.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(task.output_dim.max(1) as u64),
                    1,
                    1,
                ),
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer_status(command_buffer.status());
        if status != MetalCommandBufferStatus::Completed {
            return None;
        }

        let mut outputs = Vec::with_capacity(tasks.len());
        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let output =
                read_shared_buffer_prefix(output_buffer, saturating_usize_to_u32(task.output_dim));
            if output.len() != task.output_dim || output.iter().any(|v| !v.is_finite()) {
                return None;
            }
            outputs.push(output);
        }

        Some(outputs)
    })
}

/// Batched variant of `project_multi_matrix_rows_with_optional_native_path` for
/// multiple input rows (e.g. prefill with N tokens). Each task projects all input
/// rows through its weight matrix in a single command buffer dispatch.
#[cfg(target_os = "macos")]
fn project_multi_batched_matrix_rows_with_optional_native_path(
    bringup: &MetalRuntimeBringup,
    tasks: &[MultiProjectionTask<'_>],
    input_rows: &[Vec<f32>],
    input_width: usize,
) -> Option<Vec<Vec<Vec<f32>>>> {
    if tasks.is_empty() {
        return Some(Vec::new());
    }
    if input_rows.is_empty() {
        return Some(tasks.iter().map(|_| Vec::new()).collect());
    }

    let row_count = input_rows.len();
    let first_dtype = tasks[0].binding.native_dtype;
    if tasks.iter().any(|t| t.binding.native_dtype != first_dtype) {
        return None;
    }

    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel(first_dtype)?;

    for task in tasks {
        let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
        if input_width > cols || task.output_dim == 0 {
            return None;
        }
        if task.row_offset.checked_add(task.output_dim)?
            > tensor_matrix_dimensions(&task.binding.meta.spec)?.0
        {
            return None;
        }
        let feedback_key = batched_projection_feedback_key(
            projection_kernel_name,
            row_count,
            task.output_dim,
            input_width,
            input_width,
            cols,
        );
        if !optional_kernel_allowed(bringup, &feedback_key) {
            return None;
        }
    }

    let projection_pipeline = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()?;

    let hidden_stride = input_width;
    let mut flattened_input = Vec::with_capacity(row_count.checked_mul(hidden_stride)?);
    for row in input_rows {
        flattened_input.extend_from_slice(row.get(..input_width)?);
    }

    autoreleasepool(|| {
        let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_input);

        let output_buffers: Vec<_> = tasks
            .iter()
            .map(|task| {
                let element_count = row_count.checked_mul(task.output_dim).unwrap_or(0);
                new_zeroed_shared_buffer::<f32>(
                    &bringup.state.device,
                    saturating_usize_to_u32(element_count),
                )
            })
            .collect();

        let command_buffer = bringup.state.command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.project_matrix_rows_batched.multi");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("ax.phase1.project_matrix_rows_batched.multi.compute");

        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let (_, cols) = tensor_matrix_dimensions(&task.binding.meta.spec)?;
            let row_byte_offset = task
                .row_offset
                .checked_mul(cols)?
                .checked_mul(native_dtype_size_bytes(task.binding.native_dtype))?;
            let output_element_count = row_count.checked_mul(task.output_dim)?;

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&task.binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(output_buffer), 0);
            set_batched_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(row_count),
                saturating_usize_to_u32(task.output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input_width),
                saturating_usize_to_u32(hidden_stride),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_element_count.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_element_count.max(1) as u64),
                    1,
                    1,
                ),
            );
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer_status(command_buffer.status());
        if status != MetalCommandBufferStatus::Completed {
            return None;
        }

        let mut outputs = Vec::with_capacity(tasks.len());
        for (task, output_buffer) in tasks.iter().zip(output_buffers.iter()) {
            let output_element_count = row_count.checked_mul(task.output_dim)?;
            let output = read_shared_buffer_prefix(
                output_buffer,
                saturating_usize_to_u32(output_element_count),
            );
            if output.len() != output_element_count || output.iter().any(|v| !v.is_finite()) {
                return None;
            }
            let chunked: Vec<Vec<f32>> = output
                .chunks_exact(task.output_dim)
                .map(|chunk| chunk.to_vec())
                .collect();
            outputs.push(chunked);
        }

        Some(outputs)
    })
}

#[cfg(target_os = "macos")]
fn project_matrix_rows_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    binding: &MetalNativeTensorBufferBinding,
    row_offset: usize,
    output_dim: usize,
    input: &[f32],
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    let (projection_kernel_name, projection_pipeline_index) = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel(binding.native_dtype)?;

    let (_, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input.len() > cols {
        return None;
    }
    if output_dim == 0 {
        return Some(Vec::new());
    }
    let feedback_key =
        projection_feedback_key(projection_kernel_name, output_dim, input.len(), cols);
    if !optional_kernel_allowed(bringup, &feedback_key) {
        return None;
    }

    let row_byte_offset = row_offset
        .checked_mul(cols)?
        .checked_mul(native_dtype_size_bytes(binding.native_dtype))?;

    let output = find_optional_pipeline_handle_by_index(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
        projection_pipeline_index,
    )
    .ok()
    .and_then(|projection_pipeline| {
        autoreleasepool(|| {
            let hidden_buffer = new_shared_buffer_with_data(&bringup.state.device, input);
            let output_buffer = new_zeroed_shared_buffer::<f32>(
                &bringup.state.device,
                saturating_usize_to_u32(output_dim),
            );

            let command_buffer = bringup.state.command_queue.new_command_buffer();
            command_buffer.set_label("ax.phase1.project_matrix_rows");
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_label("ax.phase1.project_matrix_rows.compute");

            encoder.set_compute_pipeline_state(&projection_pipeline.pipeline);
            encoder.set_buffer(0, Some(&hidden_buffer), 0);
            encoder.set_buffer(1, Some(&binding.native_buffer), row_byte_offset as u64);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            set_logits_projection_dispatch_params(
                encoder,
                3,
                saturating_usize_to_u32(output_dim),
                saturating_usize_to_u32(cols),
                saturating_usize_to_u32(input.len()),
            );
            encoder.dispatch_threads(
                MTLSize::new(output_dim.max(1) as u64, 1, 1),
                MTLSize::new(
                    projection_pipeline
                        .pipeline
                        .thread_execution_width()
                        .max(1)
                        .min(output_dim.max(1) as u64),
                    1,
                    1,
                ),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let command_buffer_status = command_buffer_status(command_buffer.status());
            if command_buffer_status != MetalCommandBufferStatus::Completed {
                return None;
            }

            let output =
                read_shared_buffer_prefix(&output_buffer, saturating_usize_to_u32(output_dim));
            (output.len() == output_dim && output.iter().all(|value| value.is_finite()))
                .then_some(output)
        })
    });
    record_optional_kernel_result(bringup, &feedback_key, output.is_some());
    output
}

#[cfg(target_os = "macos")]
fn tensor_matrix_row_prefix_f32(
    binding: &MetalNativeTensorBufferBinding,
    row: usize,
    width: usize,
) -> Option<Vec<f32>> {
    let (rows, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if row >= rows || width > cols {
        return None;
    }

    let base = row.checked_mul(cols)?;
    let mut values = Vec::with_capacity(width);
    for column in 0..width {
        values.push(tensor_scalar_f32(binding, base + column)?);
    }
    Some(values)
}

#[cfg(target_os = "macos")]
fn tensor_3d_matrix_row_prefix_f32(
    binding: &MetalNativeTensorBufferBinding,
    outer_index: usize,
    row: usize,
    width: usize,
) -> Option<Vec<f32>> {
    let (outer_dim, row_count, col_count) = tensor_3d_dimensions(&binding.meta.spec)?;
    if outer_index >= outer_dim || row >= row_count || width > col_count {
        return None;
    }

    let base = outer_index
        .checked_mul(row_count)?
        .checked_add(row)?
        .checked_mul(col_count)?;
    let mut values = Vec::with_capacity(width);
    for column in 0..width {
        values.push(tensor_scalar_f32(binding, base + column)?);
    }
    Some(values)
}

#[cfg(target_os = "macos")]
fn tensor_prefix_f32(binding: &MetalNativeTensorBufferBinding, width: usize) -> Option<Vec<f32>> {
    if width > tensor_element_count(&binding.meta.spec)? {
        return None;
    }

    let mut values = Vec::with_capacity(width);
    for index in 0..width {
        values.push(tensor_scalar_f32(binding, index)?);
    }
    Some(values)
}

#[cfg(target_os = "macos")]
fn tensor_scalar_f32(
    binding: &MetalNativeTensorBufferBinding,
    element_index: usize,
) -> Option<f32> {
    let bytes = tensor_buffer_bytes(binding)?;
    let element_size = native_dtype_size_bytes(binding.meta.spec.dtype);
    let byte_offset = element_index.checked_mul(element_size)?;
    let end = byte_offset.checked_add(element_size)?;
    decode_native_tensor_scalar(binding.meta.spec.dtype, bytes.get(byte_offset..end)?)
}

#[cfg(target_os = "macos")]
fn tensor_buffer_bytes(binding: &MetalNativeTensorBufferBinding) -> Option<&[u8]> {
    let length = usize::try_from(binding.meta.spec.length_bytes).ok()?;
    (binding.bytes.len() >= length).then_some(&binding.bytes[..length])
}

#[cfg(target_os = "macos")]
fn tensor_matrix_dimensions(spec: &NativeTensorSpec) -> Option<(usize, usize)> {
    if spec.shape.len() != 2 {
        return None;
    }

    Some((
        usize::try_from(*spec.shape.first()?).ok()?,
        usize::try_from(*spec.shape.get(1)?).ok()?,
    ))
}

#[cfg(target_os = "macos")]
fn tensor_3d_dimensions(spec: &NativeTensorSpec) -> Option<(usize, usize, usize)> {
    if spec.shape.len() != 3 {
        return None;
    }

    Some((
        usize::try_from(*spec.shape.first()?).ok()?,
        usize::try_from(*spec.shape.get(1)?).ok()?,
        usize::try_from(*spec.shape.get(2)?).ok()?,
    ))
}

#[cfg(target_os = "macos")]
fn tensor_element_count(spec: &NativeTensorSpec) -> Option<usize> {
    spec.shape.iter().try_fold(1_usize, |count, dim| {
        count.checked_mul(usize::try_from(*dim).ok()?)
    })
}

#[cfg(target_os = "macos")]
fn native_dtype_size_bytes(dtype: NativeTensorDataType) -> usize {
    match dtype {
        NativeTensorDataType::F16 | NativeTensorDataType::Bf16 => 2,
        NativeTensorDataType::F32 => 4,
        NativeTensorDataType::I8 | NativeTensorDataType::U8 => 1,
    }
}

#[cfg(target_os = "macos")]
fn native_dense_effective_dtype(dtype: NativeTensorDataType) -> NativeTensorDataType {
    match dtype {
        NativeTensorDataType::I8 | NativeTensorDataType::U8 => NativeTensorDataType::F32,
        _ => dtype,
    }
}

#[cfg(target_os = "macos")]
fn native_dense_shadow_bytes(
    spec: &NativeTensorSpec,
    source_bytes: &[u8],
) -> Option<(NativeTensorDataType, Vec<u8>)> {
    let native_dtype = native_dense_effective_dtype(spec.dtype);
    if native_dtype == spec.dtype {
        return Some((native_dtype, source_bytes.to_vec()));
    }

    let element_count = tensor_element_count(spec)?;
    let scalar_size = native_dtype_size_bytes(spec.dtype);
    let required_bytes = element_count.checked_mul(scalar_size)?;
    let source_prefix = source_bytes.get(..required_bytes)?;
    let mut promoted = Vec::with_capacity(element_count.checked_mul(size_of::<f32>())?);
    for scalar_bytes in source_prefix.chunks_exact(scalar_size) {
        promoted.extend_from_slice(
            &decode_native_tensor_scalar(spec.dtype, scalar_bytes)?.to_le_bytes(),
        );
    }
    Some((native_dtype, promoted))
}

#[cfg(target_os = "macos")]
fn decode_native_tensor_scalar(dtype: NativeTensorDataType, bytes: &[u8]) -> Option<f32> {
    match dtype {
        NativeTensorDataType::F16 => {
            let raw = u16::from_le_bytes(bytes.try_into().ok()?);
            Some(decode_f16_to_f32(raw))
        }
        NativeTensorDataType::Bf16 => {
            let raw = u16::from_le_bytes(bytes.try_into().ok()?);
            Some(f32::from_bits(u32::from(raw) << 16))
        }
        NativeTensorDataType::F32 => Some(f32::from_le_bytes(bytes.try_into().ok()?)),
        NativeTensorDataType::I8 => Some(i8::from_le_bytes([*bytes.first()?]) as f32),
        NativeTensorDataType::U8 => Some(*bytes.first()? as f32),
    }
}

#[cfg(target_os = "macos")]
fn decode_f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);

    let f32_bits = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut mantissa_shifted = mantissa;
            let mut exponent_shift = -14_i32;
            while (mantissa_shifted & 0x0400) == 0 {
                mantissa_shifted <<= 1;
                exponent_shift -= 1;
            }
            mantissa_shifted &= 0x03ff;
            let exponent_bits = ((exponent_shift + 127) as u32) << 23;
            sign | exponent_bits | (mantissa_shifted << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => {
            let exponent_bits = (u32::from(exponent) + 112) << 23;
            sign | exponent_bits | (mantissa << 13)
        }
    };

    f32::from_bits(f32_bits)
}

#[cfg(target_os = "macos")]
fn round_slice_to_native_dtype(values: &mut [f32], dtype: NativeTensorDataType) {
    for value in values.iter_mut() {
        *value = round_f32_to_native_dtype(*value, dtype);
    }
}

#[cfg(target_os = "macos")]
fn round_f32_to_native_dtype(value: f32, dtype: NativeTensorDataType) -> f32 {
    match dtype {
        NativeTensorDataType::F32 | NativeTensorDataType::I8 | NativeTensorDataType::U8 => value,
        NativeTensorDataType::Bf16 => round_f32_to_bf16(value),
        NativeTensorDataType::F16 => decode_f16_to_f32(encode_f32_to_f16_bits(value)),
    }
}

#[cfg(target_os = "macos")]
fn round_f32_to_bf16(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding_bias) & 0xffff_0000)
}

#[cfg(target_os = "macos")]
fn encode_f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ff_ff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        return sign | 0x7e00;
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }

    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden_bit = mantissa | 0x80_00_00;
        let shift = 14 - half_exponent;
        let mut half_mantissa = (mantissa_with_hidden_bit >> shift) as u16;
        let round_bit = 1_u32 << (shift - 1);
        let remainder = mantissa_with_hidden_bit & ((1_u32 << shift) - 1);
        if remainder > round_bit || (remainder == round_bit && (half_mantissa & 1) != 0) {
            half_mantissa = half_mantissa.wrapping_add(1);
        }
        return sign | half_mantissa;
    }

    let mut half_mantissa = (mantissa >> 13) as u16;
    let round_bits = mantissa & 0x1fff;
    let mut half_exponent_bits = (half_exponent as u16) << 10;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (half_mantissa & 1) != 0) {
        half_mantissa = half_mantissa.wrapping_add(1);
        if half_mantissa == 0x0400 {
            half_mantissa = 0;
            half_exponent_bits = half_exponent_bits.wrapping_add(0x0400);
            if half_exponent_bits >= 0x7c00 {
                return sign | 0x7c00;
            }
        }
    }

    sign | half_exponent_bits | half_mantissa
}

#[derive(Clone, Debug)]
struct ReferenceNumericPath {
    key_cache: Vec<f32>,
    #[cfg_attr(not(test), allow(dead_code))]
    value_cache: Vec<f32>,
    gather_key: Vec<f32>,
    gather_value: Vec<f32>,
    attention_output: Vec<f32>,
    copy_key: Vec<f32>,
    copy_value: Vec<f32>,
}

#[cfg_attr(not(test), allow(dead_code))]
fn reference_numeric_path(workload: &MetalDispatchWorkload) -> ReferenceNumericPath {
    let staged_inputs = synthetic_staged_inputs(workload);
    reference_numeric_path_with_inputs(workload, &staged_inputs)
}

fn reference_numeric_path_with_inputs(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
) -> ReferenceNumericPath {
    reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
        workload,
        staged_inputs,
        None,
        None,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
fn reference_numeric_path_with_inputs_and_cache_seed(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
) -> ReferenceNumericPath {
    reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
        workload,
        staged_inputs,
        cache_seed,
        None,
    )
}

fn reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
    attention_config: Option<ReferenceAttentionConfig>,
) -> ReferenceNumericPath {
    let head_size = workload.numeric_layout.head_size() as usize;
    let head_count = workload.numeric_layout.head_count as usize;
    let head_dim = workload.numeric_layout.head_dim as usize;
    let attention_config = attention_config.unwrap_or_else(|| {
        ReferenceAttentionConfig::from_head_dim(head_dim).unwrap_or(ReferenceAttentionConfig {
            softmax_scale: 1.0,
            softcap: None,
        })
    });
    let scheduled_tokens = workload.token_elements as usize;
    let gather_tokens = workload.kv_metadata.gather_token_count() as usize;
    let mut key_cache = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    let mut value_cache = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    if let Some(cache_seed) = cache_seed {
        key_cache.copy_from_slice(cache_seed.key_cache);
        value_cache.copy_from_slice(cache_seed.value_cache);
    }

    for token_id in 0..scheduled_tokens {
        let slot = workload.kv_metadata.slot_mapping[token_id] as usize;
        let source_base = token_id * head_size;
        let cache_base = slot * head_size;
        key_cache[cache_base..cache_base + head_size]
            .copy_from_slice(&staged_inputs.key[source_base..source_base + head_size]);
        value_cache[cache_base..cache_base + head_size]
            .copy_from_slice(&staged_inputs.value[source_base..source_base + head_size]);
    }

    let mut gather_key = vec![0.0_f32; workload.gather_numeric_elements() as usize];
    let mut gather_value = vec![0.0_f32; workload.gather_numeric_elements() as usize];
    for token_id in 0..gather_tokens {
        let batch_id = batch_id_for_token(&workload.kv_metadata.cu_seq_lens, token_id as u32);
        let batch_offset = token_id as u32 - workload.kv_metadata.cu_seq_lens[batch_id];
        let block_index = (batch_offset / workload.kv_metadata.block_size_tokens) as usize;
        let block_offset = (batch_offset % workload.kv_metadata.block_size_tokens) as usize;
        let block_base = workload.kv_metadata.gather_block_table
            [batch_id * workload.kv_metadata.gather_block_table_stride as usize + block_index]
            as usize;
        let slot = block_base + block_offset;
        let source_base = slot * head_size;
        let target_base = token_id * head_size;
        gather_key[target_base..target_base + head_size]
            .copy_from_slice(&key_cache[source_base..source_base + head_size]);
        gather_value[target_base..target_base + head_size]
            .copy_from_slice(&value_cache[source_base..source_base + head_size]);
    }

    let mut attention_output = vec![0.0_f32; workload.attention_numeric_elements() as usize];
    for token_id in 0..scheduled_tokens {
        let batch_id =
            batch_id_for_token(&workload.kv_metadata.scheduled_cu_seq_lens, token_id as u32);
        let context_begin = workload.kv_metadata.cu_seq_lens[batch_id] as usize;
        let context_end = workload.kv_metadata.cu_seq_lens[batch_id + 1] as usize;
        let query_base = token_id * head_size;

        for head in 0..head_count {
            let head_query_base = query_base + head * head_dim;
            let mut max_score = f32::NEG_INFINITY;

            for context_index in context_begin..context_end {
                let context_base = context_index * head_size + head * head_dim;
                let score = configured_attention_score(
                    &staged_inputs.query[head_query_base..head_query_base + head_dim],
                    &gather_key[context_base..context_base + head_dim],
                    attention_config,
                );
                max_score = max_score.max(score);
            }

            let mut weight_sum = 0.0_f32;
            let mut accum = vec![0.0_f32; head_dim];
            for context_index in context_begin..context_end {
                let context_base = context_index * head_size + head * head_dim;
                let score = configured_attention_score(
                    &staged_inputs.query[head_query_base..head_query_base + head_dim],
                    &gather_key[context_base..context_base + head_dim],
                    attention_config,
                );
                let weight = (score - max_score).exp();
                weight_sum += weight;
                for lane in 0..head_dim {
                    accum[lane] += weight * gather_value[context_base + lane];
                }
            }

            for lane in 0..head_dim {
                attention_output[head_query_base + lane] = accum[lane] / weight_sum.max(0.000001);
            }
        }
    }

    let mut copy_key = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    let mut copy_value = vec![0.0_f32; workload.slot_numeric_capacity() as usize];
    for pair in &workload.kv_metadata.copy_block_mapping {
        let source_base = pair[0] as usize * head_size;
        let target_base = pair[1] as usize * head_size;
        let block_width = workload.block_numeric_elements() as usize;
        copy_key[target_base..target_base + block_width]
            .copy_from_slice(&key_cache[source_base..source_base + block_width]);
        copy_value[target_base..target_base + block_width]
            .copy_from_slice(&value_cache[source_base..source_base + block_width]);
    }

    ReferenceNumericPath {
        key_cache,
        value_cache,
        gather_key,
        gather_value,
        attention_output,
        copy_key,
        copy_value,
    }
}

fn batch_id_for_token(cu_seq_lens: &[u32], token_id: u32) -> usize {
    let mut lo = 0_usize;
    let mut hi = cu_seq_lens.len().saturating_sub(1);

    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if cu_seq_lens[mid] <= token_id {
            lo = mid;
        } else {
            hi = mid.saturating_sub(1);
        }
    }

    lo.min(cu_seq_lens.len().saturating_sub(2))
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn configured_attention_score(
    query: &[f32],
    key: &[f32],
    attention_config: ReferenceAttentionConfig,
) -> f32 {
    let mut score = dot_product(query, key) * attention_config.softmax_scale;
    if let Some(softcap) = attention_config.softcap {
        score = softcap * (score / softcap).tanh();
    }
    score
}

#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
fn validate_numeric_trace_against_reference(
    workload: &MetalDispatchWorkload,
    trace: &MetalDispatchNumericTrace,
) -> Result<MetalNumericValidationSummary, MetalRuntimeError> {
    let staged_inputs = synthetic_staged_inputs(workload);
    validate_numeric_trace_against_inputs(workload, &staged_inputs, trace)
}

#[cfg(target_os = "macos")]
fn validate_numeric_trace_against_inputs(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    trace: &MetalDispatchNumericTrace,
) -> Result<MetalNumericValidationSummary, MetalRuntimeError> {
    validate_numeric_trace_against_inputs_and_cache_seed(workload, staged_inputs, None, None, trace)
}

#[cfg(target_os = "macos")]
fn validate_numeric_trace_against_inputs_and_cache_seed(
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    cache_seed: Option<MetalDispatchKvCacheSeed<'_>>,
    attention_config: Option<ReferenceAttentionConfig>,
    trace: &MetalDispatchNumericTrace,
) -> Result<MetalNumericValidationSummary, MetalRuntimeError> {
    let reference = reference_numeric_path_with_inputs_and_cache_seed_and_attention_config(
        workload,
        staged_inputs,
        cache_seed,
        attention_config,
    );

    let expected_key_cache_checksum =
        validate_numeric_checksum("key_cache", trace.key_cache_checksum, &reference.key_cache)?;
    let expected_gather_output_checksum = validate_numeric_pair_checksum(
        "gather_kv",
        trace.gather_output_checksum,
        &reference.gather_key,
        &reference.gather_value,
    )?;
    let expected_copy_output_checksum = validate_numeric_pair_checksum(
        "copy_blocks",
        trace.copy_output_checksum,
        &reference.copy_key,
        &reference.copy_value,
    )?;
    let (expected_attention_output_checksum, attention_max_abs_diff_microunits) =
        validate_attention_output_against_reference(
            &reference.attention_output,
            &trace.attention_output_bits,
            trace.attention_output_checksum,
        )?;

    Ok(MetalNumericValidationSummary {
        expected_key_cache_checksum,
        expected_attention_output_checksum,
        expected_gather_output_checksum,
        expected_copy_output_checksum,
        attention_max_abs_diff_microunits,
    })
}

#[cfg(target_os = "macos")]
fn validate_numeric_checksum(
    stage: &'static str,
    actual_checksum: u64,
    expected_values: &[f32],
) -> Result<u64, MetalRuntimeError> {
    let expected_checksum = checksum_f32_slice(expected_values);
    if actual_checksum == expected_checksum {
        return Ok(expected_checksum);
    }

    Err(MetalRuntimeError::NumericValidationMismatch {
        stage,
        message: format!(
            "checksum mismatch; actual={actual_checksum:#018x}, expected={expected_checksum:#018x}"
        ),
    })
}

#[cfg(target_os = "macos")]
fn validate_numeric_pair_checksum(
    stage: &'static str,
    actual_checksum: u64,
    expected_left: &[f32],
    expected_right: &[f32],
) -> Result<u64, MetalRuntimeError> {
    let expected_checksum = checksum_f32_pair(expected_left, expected_right);
    if actual_checksum == expected_checksum {
        return Ok(expected_checksum);
    }

    Err(MetalRuntimeError::NumericValidationMismatch {
        stage,
        message: format!(
            "checksum mismatch; actual={actual_checksum:#018x}, expected={expected_checksum:#018x}"
        ),
    })
}

#[cfg(target_os = "macos")]
fn validate_attention_output_against_reference(
    expected_values: &[f32],
    actual_bits: &[u32],
    actual_checksum: u64,
) -> Result<(u64, u32), MetalRuntimeError> {
    const ABS_TOLERANCE: f32 = 1.0e-4;
    const REL_TOLERANCE: f32 = 5.0e-4;

    let expected_checksum = checksum_f32_slice(expected_values);
    if actual_bits.len() != expected_values.len() {
        return Err(MetalRuntimeError::NumericValidationMismatch {
            stage: "attention_output",
            message: format!(
                "element count mismatch; actual={}, expected={}, actual_checksum={actual_checksum:#018x}, expected_checksum={expected_checksum:#018x}",
                actual_bits.len(),
                expected_values.len(),
            ),
        });
    }

    let mut max_abs_diff = 0.0_f32;
    for (index, (&actual_bits, &expected)) in actual_bits.iter().zip(expected_values).enumerate() {
        let actual = f32::from_bits(actual_bits);
        if !actual.is_finite() {
            return Err(MetalRuntimeError::NumericValidationMismatch {
                stage: "attention_output",
                message: format!("non-finite value at index {index}: actual={actual}"),
            });
        }

        let abs_diff = (actual - expected).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        let scale = actual.abs().max(expected.abs()).max(1.0);
        if abs_diff > ABS_TOLERANCE && abs_diff > scale * REL_TOLERANCE {
            return Err(MetalRuntimeError::NumericValidationMismatch {
                stage: "attention_output",
                message: format!(
                    "value drift at index {index}: actual={actual}, expected={expected}, abs_diff={abs_diff}, max_abs_diff={max_abs_diff}, actual_checksum={actual_checksum:#018x}, expected_checksum={expected_checksum:#018x}",
                ),
            });
        }
    }

    Ok((
        expected_checksum,
        (max_abs_diff * 1_000_000.0)
            .round()
            .clamp(0.0, u32::MAX as f32) as u32,
    ))
}

#[cfg(target_os = "macos")]
fn ensure_dispatch_arena<'a>(
    device: &Device,
    command_queue: &CommandQueue,
    arena_slot: &'a mut Option<MetalDispatchArena>,
    workload: &MetalDispatchWorkload,
) -> (&'a mut MetalDispatchArena, MetalDispatchArenaInfo) {
    let required = MetalDispatchArenaRequirements::from_workload(workload);
    let (reused_existing, grew_existing) = match arena_slot.as_ref() {
        Some(existing) if existing.requirements.supports(required) => (true, false),
        Some(existing) => {
            let grown = if existing.requirements.head_size == required.head_size {
                MetalDispatchArena::with_preserved_cache(device, command_queue, existing, required)
            } else {
                MetalDispatchArena::new(device, required)
            };
            *arena_slot = Some(grown);
            (false, true)
        }
        None => {
            *arena_slot = Some(MetalDispatchArena::new(device, required));
            (false, false)
        }
    };

    let info = arena_slot
        .as_ref()
        .expect("dispatch arena should exist after ensure")
        .requirements
        .info(reused_existing, grew_existing);
    let arena = arena_slot
        .as_mut()
        .expect("dispatch arena should exist after ensure");
    (arena, info)
}

#[cfg(target_os = "macos")]
fn new_zeroed_shared_buffer<T>(device: &Device, element_count: u32) -> Buffer {
    let byte_count = element_count.max(1) as usize * size_of::<T>();
    let zeros = vec![0_u8; byte_count];
    device.new_buffer_with_data(
        zeros.as_ptr().cast(),
        zeros.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[cfg(target_os = "macos")]
fn new_shared_buffer_with_data<T>(device: &Device, values: &[T]) -> Buffer {
    if values.is_empty() {
        let zeros = vec![0_u8; size_of::<T>().max(1)];
        return device.new_buffer_with_data(
            zeros.as_ptr().cast(),
            zeros.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
    }

    device.new_buffer_with_data(
        values.as_ptr().cast(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[cfg(target_os = "macos")]
fn copy_shared_buffer_prefix<T>(
    command_queue: &CommandQueue,
    src: &Buffer,
    dst: &Buffer,
    element_count: u32,
) {
    copy_shared_buffer_range::<T>(command_queue, src, 0, dst, 0, element_count);
}

#[cfg(target_os = "macos")]
fn copy_shared_buffer_range<T>(
    command_queue: &CommandQueue,
    src: &Buffer,
    src_element_offset: u32,
    dst: &Buffer,
    dst_element_offset: u32,
    element_count: u32,
) {
    if element_count == 0 {
        return;
    }

    let byte_count = element_count as u64 * size_of::<T>() as u64;
    let src_offset = src_element_offset as u64 * size_of::<T>() as u64;
    let dst_offset = dst_element_offset as u64 * size_of::<T>() as u64;
    autoreleasepool(|| {
        let command_buffer = command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.dispatch_arena_cache_copy");
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(src, src_offset, dst, dst_offset, byte_count);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
}

#[cfg(target_os = "macos")]
fn capture_numeric_trace(
    workload: &MetalDispatchWorkload,
    arena: &MetalDispatchArena,
) -> MetalDispatchNumericTrace {
    let key_cache = read_shared_buffer_prefix(&arena.key_cache, workload.slot_numeric_capacity());
    let attention_output = read_shared_buffer_prefix(
        &arena.attention_output,
        workload.attention_numeric_elements(),
    );
    let gather_key =
        read_shared_buffer_prefix(&arena.kv_key_gathered, workload.gather_numeric_elements());
    let gather_value =
        read_shared_buffer_prefix(&arena.kv_value_gathered, workload.gather_numeric_elements());
    let copy_key =
        read_shared_buffer_prefix(&arena.copy_key_target, workload.slot_numeric_capacity());
    let copy_value =
        read_shared_buffer_prefix(&arena.copy_value_target, workload.slot_numeric_capacity());

    MetalDispatchNumericTrace {
        attention_output_bits: attention_output
            .iter()
            .map(|value| value.to_bits())
            .collect(),
        key_cache_checksum: checksum_f32_slice(&key_cache),
        attention_output_checksum: checksum_f32_slice(&attention_output),
        gather_output_checksum: checksum_f32_pair(&gather_key, &gather_value),
        copy_output_checksum: checksum_f32_pair(&copy_key, &copy_value),
        validation: None,
    }
}

#[cfg(target_os = "macos")]
fn capture_numeric_cache_snapshot(
    workload: &MetalDispatchWorkload,
    arena: &MetalDispatchArena,
) -> MetalDispatchKvCacheSnapshot {
    let mut key_cache =
        read_shared_buffer_prefix(&arena.key_cache, workload.slot_numeric_capacity());
    let mut value_cache =
        read_shared_buffer_prefix(&arena.value_cache, workload.slot_numeric_capacity());
    let copy_key =
        read_shared_buffer_prefix(&arena.copy_key_target, workload.slot_numeric_capacity());
    let copy_value =
        read_shared_buffer_prefix(&arena.copy_value_target, workload.slot_numeric_capacity());
    merge_copy_targets_into_cache_snapshot(
        workload,
        &mut key_cache,
        &mut value_cache,
        &copy_key,
        &copy_value,
    );

    MetalDispatchKvCacheSnapshot {
        key_cache,
        value_cache,
    }
}

#[cfg(target_os = "macos")]
const SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH: u32 = 32_768;
#[cfg(target_os = "macos")]
const SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS: u32 = 4;

#[cfg(target_os = "macos")]
fn read_shared_buffer_prefix(buffer: &Buffer, element_count: u32) -> Vec<f32> {
    if element_count == 0 {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(element_count as usize);
    let mut offset = 0_u32;
    while offset < element_count {
        let max_chunk_len = element_count
            .saturating_sub(offset)
            .min(SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH);
        let chunk_len = max_chunk_len - (max_chunk_len % SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS);
        if chunk_len == 0 {
            break;
        }
        let bytes_per_row = chunk_len as u64 * size_of::<f32>() as u64;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::R32Float);
        descriptor.set_width(chunk_len as u64);
        descriptor.set_height(1);
        let buffer_offset = offset as u64 * size_of::<f32>() as u64;
        let texture = buffer.new_texture_with_descriptor(&descriptor, buffer_offset, bytes_per_row);

        if texture.width() < chunk_len as u64 {
            return Vec::new();
        }

        let mut chunk_values = vec![0.0_f32; chunk_len as usize];
        texture.get_bytes(
            chunk_values.as_mut_ptr().cast(),
            bytes_per_row,
            MTLRegion::new_2d(0, 0, chunk_len as u64, 1),
            0,
        );
        values.extend_from_slice(&chunk_values);
        offset = offset.saturating_add(chunk_len);
    }
    if offset < element_count {
        let tail_count = element_count.saturating_sub(offset);
        let tail_window = SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS.max(tail_count);
        let device = buffer.device().to_owned();
        let command_queue = device.new_command_queue();
        let staging_buffer = new_zeroed_shared_buffer::<f32>(&device, tail_window);
        copy_shared_buffer_range::<f32>(
            &command_queue,
            buffer,
            offset,
            &staging_buffer,
            0,
            tail_count,
        );
        let bytes_per_row = tail_window as u64 * size_of::<f32>() as u64;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::R32Float);
        descriptor.set_width(tail_window as u64);
        descriptor.set_height(1);
        let texture = staging_buffer.new_texture_with_descriptor(&descriptor, 0, bytes_per_row);
        if texture.width() < tail_window as u64 {
            return Vec::new();
        }
        let mut tail_values = vec![0.0_f32; tail_window as usize];
        texture.get_bytes(
            tail_values.as_mut_ptr().cast(),
            bytes_per_row,
            MTLRegion::new_2d(0, 0, tail_window as u64, 1),
            0,
        );
        values.extend_from_slice(&tail_values[..tail_count as usize]);
    }
    values
}

#[cfg(target_os = "macos")]
fn read_shared_u32_buffer_prefix(buffer: &Buffer, element_count: u32) -> Vec<u32> {
    if element_count == 0 {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(element_count as usize);
    let mut offset = 0_u32;
    while offset < element_count {
        let max_chunk_len = element_count
            .saturating_sub(offset)
            .min(SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH);
        let chunk_len = max_chunk_len - (max_chunk_len % SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS);
        if chunk_len == 0 {
            break;
        }
        let bytes_per_row = chunk_len as u64 * size_of::<u32>() as u64;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::R32Uint);
        descriptor.set_width(chunk_len as u64);
        descriptor.set_height(1);
        let buffer_offset = offset as u64 * size_of::<u32>() as u64;
        let texture = buffer.new_texture_with_descriptor(&descriptor, buffer_offset, bytes_per_row);

        if texture.width() < chunk_len as u64 {
            return Vec::new();
        }

        let mut chunk_values = vec![0_u32; chunk_len as usize];
        texture.get_bytes(
            chunk_values.as_mut_ptr().cast(),
            bytes_per_row,
            MTLRegion::new_2d(0, 0, chunk_len as u64, 1),
            0,
        );
        values.extend_from_slice(&chunk_values);
        offset = offset.saturating_add(chunk_len);
    }
    if offset < element_count {
        let tail_count = element_count.saturating_sub(offset);
        let tail_window = SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS.max(tail_count);
        let device = buffer.device().to_owned();
        let command_queue = device.new_command_queue();
        let staging_buffer = new_zeroed_shared_buffer::<u32>(&device, tail_window);
        copy_shared_buffer_range::<u32>(
            &command_queue,
            buffer,
            offset,
            &staging_buffer,
            0,
            tail_count,
        );
        let bytes_per_row = tail_window as u64 * size_of::<u32>() as u64;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::R32Uint);
        descriptor.set_width(tail_window as u64);
        descriptor.set_height(1);
        let texture = staging_buffer.new_texture_with_descriptor(&descriptor, 0, bytes_per_row);
        if texture.width() < tail_window as u64 {
            return Vec::new();
        }
        let mut tail_values = vec![0_u32; tail_window as usize];
        texture.get_bytes(
            tail_values.as_mut_ptr().cast(),
            bytes_per_row,
            MTLRegion::new_2d(0, 0, tail_window as u64, 1),
            0,
        );
        values.extend_from_slice(&tail_values[..tail_count as usize]);
    }
    values
}

fn checksum_f32_slice(values: &[f32]) -> u64 {
    checksum_words(values.iter().map(|value| value.to_bits()))
}

fn checksum_f32_pair(left: &[f32], right: &[f32]) -> u64 {
    checksum_words(
        left.iter()
            .map(|value| value.to_bits())
            .chain(right.iter().map(|value| value.to_bits())),
    )
}

fn checksum_words(words: impl IntoIterator<Item = u32>) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    words.into_iter().fold(OFFSET_BASIS, |checksum, word| {
        checksum
            .wrapping_mul(PRIME)
            .wrapping_add(u64::from(word))
            .rotate_left(7)
    })
}

#[cfg(target_os = "macos")]
fn encode_numeric_kernel(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &MetalPipelineHandle,
    trace: &MetalDispatchKernelTrace,
    workload: &MetalDispatchWorkload,
    arena: &MetalDispatchArena,
) {
    encoder.set_compute_pipeline_state(&pipeline.pipeline);

    match pipeline.function_name.as_str() {
        "reshape_and_cache" => {
            encoder.set_buffer(0, Some(&arena.reshape_key), 0);
            encoder.set_buffer(1, Some(&arena.reshape_value), 0);
            encoder.set_buffer(2, Some(&arena.key_cache), 0);
            encoder.set_buffer(3, Some(&arena.value_cache), 0);
            encoder.set_buffer(4, Some(&arena.reshape_slot_mapping), 0);
            set_cache_dispatch_params(
                encoder,
                5,
                trace.element_count,
                workload.numeric_layout.head_size(),
            );
        }
        "paged_decode_attention" => {
            encoder.set_buffer(0, Some(&arena.attention_query), 0);
            encoder.set_buffer(1, Some(&arena.kv_key_gathered), 0);
            encoder.set_buffer(2, Some(&arena.kv_value_gathered), 0);
            encoder.set_buffer(3, Some(&arena.cu_seq_lens), 0);
            encoder.set_buffer(4, Some(&arena.scheduled_cu_seq_lens), 0);
            encoder.set_buffer(5, Some(&arena.attention_output), 0);
            set_attention_dispatch_params(
                encoder,
                6,
                trace.element_count,
                workload.kv_metadata.seq_lens.len() as u32,
                workload.numeric_layout.head_count,
                workload.numeric_layout.head_dim,
            );
        }
        "gather_kv_cache" => {
            encoder.set_buffer(0, Some(&arena.key_cache), 0);
            encoder.set_buffer(1, Some(&arena.value_cache), 0);
            encoder.set_buffer(2, Some(&arena.kv_block_table), 0);
            encoder.set_buffer(3, Some(&arena.cu_seq_lens), 0);
            encoder.set_buffer(4, Some(&arena.kv_key_gathered), 0);
            encoder.set_buffer(5, Some(&arena.kv_value_gathered), 0);
            set_gather_dispatch_params(
                encoder,
                6,
                trace.element_count,
                arena.requirements.block_size_tokens,
                workload.kv_metadata.seq_lens.len() as u32,
                workload.kv_metadata.gather_block_table_stride,
                workload.numeric_layout.head_size(),
            );
        }
        "copy_blocks" => {
            encoder.set_buffer(0, Some(&arena.key_cache), 0);
            encoder.set_buffer(1, Some(&arena.value_cache), 0);
            encoder.set_buffer(2, Some(&arena.copy_key_target), 0);
            encoder.set_buffer(3, Some(&arena.copy_value_target), 0);
            encoder.set_buffer(4, Some(&arena.copy_block_mapping), 0);
            set_copy_block_dispatch_params(
                encoder,
                5,
                workload.block_numeric_elements(),
                workload.block_numeric_elements(),
                workload.kv_metadata.copy_block_mapping.len() as u32,
                workload.numeric_layout.head_size(),
            );
        }
        _ => {
            unreachable!("required pipeline inventory should only include current required kernels")
        }
    }

    encoder.dispatch_threads(
        MTLSize::new(
            trace.threads_per_grid.width,
            trace.threads_per_grid.height,
            trace.threads_per_grid.depth,
        ),
        MTLSize::new(
            trace.threads_per_threadgroup.width,
            trace.threads_per_threadgroup.height,
            trace.threads_per_threadgroup.depth,
        ),
    );
}

#[cfg(target_os = "macos")]
fn set_cache_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    head_size: u32,
) {
    let params = CacheDispatchParams {
        element_count,
        head_size,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<CacheDispatchParams>() as u64,
        (&params as *const CacheDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_attention_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    num_seqs: u32,
    head_count: u32,
    head_dim: u32,
) {
    let params = AttentionDispatchParams {
        element_count,
        num_seqs,
        head_count,
        head_dim,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<AttentionDispatchParams>() as u64,
        (&params as *const AttentionDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_gather_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    block_size_tokens: u32,
    num_seqs: u32,
    block_table_stride: u32,
    head_size: u32,
) {
    let params = GatherDispatchParams {
        element_count,
        num_seqs,
        block_size_tokens,
        block_table_stride,
        head_size,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<GatherDispatchParams>() as u64,
        (&params as *const GatherDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_copy_block_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    numel_per_block_key: u32,
    numel_per_block_value: u32,
    num_pairs: u32,
    head_size: u32,
) {
    let params = CopyBlockDispatchParams {
        num_pairs,
        numel_per_block_key,
        numel_per_block_value,
        head_size,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<CopyBlockDispatchParams>() as u64,
        (&params as *const CopyBlockDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_logits_projection_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
) {
    let params = LogitsProjectionDispatchParams {
        vocab_rows,
        projection_cols,
        input_width,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<LogitsProjectionDispatchParams>() as u64,
        (&params as *const LogitsProjectionDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_batched_logits_projection_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    vocab_rows: u32,
    projection_cols: u32,
    input_width: u32,
    hidden_stride: u32,
) {
    let params = BatchedLogitsProjectionDispatchParams {
        token_count,
        vocab_rows,
        projection_cols,
        input_width,
        hidden_stride,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<BatchedLogitsProjectionDispatchParams>() as u64,
        (&params as *const BatchedLogitsProjectionDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_logits_argmax_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
) {
    let params = LogitsArgmaxDispatchParams { element_count };
    encoder.set_bytes(
        buffer_index,
        size_of::<LogitsArgmaxDispatchParams>() as u64,
        (&params as *const LogitsArgmaxDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_batched_logits_argmax_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    vocab_rows: u32,
) {
    let params = BatchedLogitsArgmaxDispatchParams {
        token_count,
        vocab_rows,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<BatchedLogitsArgmaxDispatchParams>() as u64,
        (&params as *const BatchedLogitsArgmaxDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_rms_norm_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
    epsilon: f32,
    weight_offset: f32,
) {
    let params = RmsNormDispatchParams {
        element_count,
        epsilon,
        weight_offset,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<RmsNormDispatchParams>() as u64,
        (&params as *const RmsNormDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_batched_rms_norm_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    head_count: u32,
    head_dim: u32,
    epsilon: f32,
    weight_offset: f32,
) {
    let params = BatchedRmsNormDispatchParams {
        head_count,
        head_dim,
        epsilon,
        weight_offset,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<BatchedRmsNormDispatchParams>() as u64,
        (&params as *const BatchedRmsNormDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_ffn_gate_product_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
) {
    let params = FfnGateProductDispatchParams { element_count };
    encoder.set_bytes(
        buffer_index,
        size_of::<FfnGateProductDispatchParams>() as u64,
        (&params as *const FfnGateProductDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_vector_add_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    element_count: u32,
) {
    let params = VectorAddDispatchParams { element_count };
    encoder.set_bytes(
        buffer_index,
        size_of::<VectorAddDispatchParams>() as u64,
        (&params as *const VectorAddDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_batched_row_scale_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    row_count: u32,
    row_width: u32,
) {
    let params = BatchedRowScaleDispatchParams {
        row_count,
        row_width,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<BatchedRowScaleDispatchParams>() as u64,
        (&params as *const BatchedRowScaleDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_batched_row_vector_scale_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    row_count: u32,
    row_width: u32,
) {
    let params = BatchedRowVectorScaleDispatchParams {
        row_count,
        row_width,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<BatchedRowVectorScaleDispatchParams>() as u64,
        (&params as *const BatchedRowVectorScaleDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_linear_attention_conv_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    batch_size: u32,
    dims: ResolvedLinearAttentionDims,
) {
    let params = LinearAttentionConvDispatchParams {
        batch_size,
        conv_dim: saturating_usize_to_u32(dims.conv_dim),
        conv_kernel_dim: saturating_usize_to_u32(dims.conv_kernel_dim),
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<LinearAttentionConvDispatchParams>() as u64,
        (&params as *const LinearAttentionConvDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_linear_gated_delta_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    batch_size: u32,
    dims: ResolvedLinearAttentionDims,
) {
    let params = LinearGatedDeltaDispatchParams {
        batch_size,
        num_key_heads: saturating_usize_to_u32(dims.num_key_heads),
        num_value_heads: saturating_usize_to_u32(dims.num_value_heads),
        key_head_dim: saturating_usize_to_u32(dims.key_head_dim),
        value_head_dim: saturating_usize_to_u32(dims.value_head_dim),
        repeat_factor: saturating_usize_to_u32(dims.repeat_factor),
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<LinearGatedDeltaDispatchParams>() as u64,
        (&params as *const LinearGatedDeltaDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_model_stage_rope_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    rotary_dim: u32,
) {
    let params = ModelStageRopeDispatchParams {
        query_head_count,
        key_head_count,
        head_dim,
        rope_style,
        rotary_dim,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<ModelStageRopeDispatchParams>() as u64,
        (&params as *const ModelStageRopeDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn set_batched_model_stage_rope_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
    freq_base: f32,
    rotary_dim: u32,
) {
    let params = BatchedModelStageRopeDispatchParams {
        token_count,
        query_head_count,
        key_head_count,
        head_dim,
        rope_style,
        freq_base,
        rotary_dim,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<BatchedModelStageRopeDispatchParams>() as u64,
        (&params as *const BatchedModelStageRopeDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_grouped_kv_expand_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    output_element_count: u32,
    kv_head_count: u32,
    heads_per_kv: u32,
    head_dim: u32,
) {
    let params = GroupedKvExpandDispatchParams {
        output_element_count,
        kv_head_count,
        heads_per_kv,
        head_dim,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<GroupedKvExpandDispatchParams>() as u64,
        (&params as *const GroupedKvExpandDispatchParams).cast(),
    );
}

#[cfg(target_os = "macos")]
fn set_embedding_gather_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    token_count: u32,
    embedding_rows: u32,
    hidden_dim: u32,
    scale: f32,
) {
    let params = EmbeddingGatherDispatchParams {
        token_count,
        embedding_rows,
        hidden_dim,
        scale,
    };
    encoder.set_bytes(
        buffer_index,
        size_of::<EmbeddingGatherDispatchParams>() as u64,
        (&params as *const EmbeddingGatherDispatchParams).cast(),
    );
}

#[cfg(test)]
mod tests;
