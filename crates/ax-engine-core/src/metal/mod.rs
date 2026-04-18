use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
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
use crate::scheduler::{ExecutionMode, RouteMetadata};

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
    optional_pipeline_lookup: BTreeMap<String, usize>,
    optional_kernel_dispatch_plan: MetalOptionalKernelDispatchPlan,
    optional_kernel_feedback: Mutex<MetalOptionalKernelFeedbackState>,
    dispatch_arena: Mutex<Option<MetalDispatchArena>>,
}

#[cfg(target_os = "macos")]
struct MetalPipelineHandle {
    function_name: String,
    pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MetalOptionalKernelDispatchPlan {
    projection_f32: bool,
    projection_f16: bool,
    projection_bf16: bool,
    batched_projection_f32: bool,
    batched_projection_f16: bool,
    batched_projection_bf16: bool,
    embedding_gather_f32: bool,
    embedding_gather_f16: bool,
    embedding_gather_bf16: bool,
    rms_norm_f32: bool,
    rms_norm_f16: bool,
    rms_norm_bf16: bool,
    batched_rms_norm_f32: bool,
    batched_rms_norm_f16: bool,
    batched_rms_norm_bf16: bool,
    logits_argmax_f32: bool,
    logits_argmax_batched_f32: bool,
    sample_argmax_logprob_f32: bool,
    sample_argmax_logprob_batched_f32: bool,
    apply_rope_f32: bool,
    apply_rope_batched_f32: bool,
    expand_grouped_kv_heads_f32: bool,
    ffn_gate_silu_product_f32: bool,
    ffn_gate_gelu_approx_product_f32: bool,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MetalOptionalKernelFeedbackState {
    consecutive_failures_by_kernel: BTreeMap<String, u32>,
    disabled_kernels: BTreeSet<String>,
}

#[cfg(target_os = "macos")]
impl MetalOptionalKernelDispatchPlan {
    fn projection_kernel_name(self, dtype: NativeTensorDataType) -> Option<&'static str> {
        match dtype {
            NativeTensorDataType::F32 if self.projection_f32 => {
                Some("decode_logits_projection_f32")
            }
            NativeTensorDataType::F16 if self.projection_f16 => {
                Some("decode_logits_projection_f16")
            }
            NativeTensorDataType::Bf16 if self.projection_bf16 => {
                Some("decode_logits_projection_bf16")
            }
            _ => None,
        }
    }

    fn batched_projection_kernel_name(self, dtype: NativeTensorDataType) -> Option<&'static str> {
        match dtype {
            NativeTensorDataType::F32 if self.batched_projection_f32 => {
                Some("decode_logits_projection_batched_f32")
            }
            NativeTensorDataType::F16 if self.batched_projection_f16 => {
                Some("decode_logits_projection_batched_f16")
            }
            NativeTensorDataType::Bf16 if self.batched_projection_bf16 => {
                Some("decode_logits_projection_batched_bf16")
            }
            _ => None,
        }
    }

    fn embedding_gather_kernel_name(self, dtype: NativeTensorDataType) -> Option<&'static str> {
        match dtype {
            NativeTensorDataType::F32 if self.embedding_gather_f32 => {
                Some("gather_embedding_rows_f32")
            }
            NativeTensorDataType::F16 if self.embedding_gather_f16 => {
                Some("gather_embedding_rows_f16")
            }
            NativeTensorDataType::Bf16 if self.embedding_gather_bf16 => {
                Some("gather_embedding_rows_bf16")
            }
            _ => None,
        }
    }

    fn rms_norm_kernel_name(self, dtype: NativeTensorDataType) -> Option<&'static str> {
        match dtype {
            NativeTensorDataType::F32 if self.rms_norm_f32 => Some("rms_norm_f32"),
            NativeTensorDataType::F16 if self.rms_norm_f16 => Some("rms_norm_f16"),
            NativeTensorDataType::Bf16 if self.rms_norm_bf16 => Some("rms_norm_bf16"),
            _ => None,
        }
    }

    fn batched_rms_norm_kernel_name(self, dtype: NativeTensorDataType) -> Option<&'static str> {
        match dtype {
            NativeTensorDataType::F32 if self.batched_rms_norm_f32 => Some("rms_norm_batched_f32"),
            NativeTensorDataType::F16 if self.batched_rms_norm_f16 => Some("rms_norm_batched_f16"),
            NativeTensorDataType::Bf16 if self.batched_rms_norm_bf16 => {
                Some("rms_norm_batched_bf16")
            }
            _ => None,
        }
    }

    fn ffn_gate_product_kernel_name(self, activation: ModelFfnActivation) -> Option<&'static str> {
        match activation {
            ModelFfnActivation::Silu if self.ffn_gate_silu_product_f32 => {
                Some("ffn_gate_silu_product_f32")
            }
            ModelFfnActivation::GeluApprox if self.ffn_gate_gelu_approx_product_f32 => {
                Some("ffn_gate_gelu_approx_product_f32")
            }
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
    fn sample_argmax_logprob_with_optional_native_path(
        &self,
        logits: &[f32],
    ) -> Option<(u32, f32)> {
        let kernel_name = "sample_argmax_logprob_f32";
        if logits.is_empty() {
            return None;
        }
        if !self
            .bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_f32
        {
            return None;
        }
        if !optional_kernel_allowed(&self.bringup, kernel_name) {
            return None;
        }

        let output = find_optional_pipeline_handle(
            &self.bringup.state,
            &self.bringup.metallib.path,
            kernel_name,
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
        record_optional_kernel_result(&self.bringup, kernel_name, output.is_some());
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
        if !self
            .bringup
            .state
            .optional_kernel_dispatch_plan
            .sample_argmax_logprob_batched_f32
        {
            return None;
        }
        if !optional_kernel_allowed(&self.bringup, kernel_name) {
            return None;
        }
        let vocab_rows = logits_rows.first()?.len();
        if vocab_rows == 0 || logits_rows.iter().any(|row| row.len() != vocab_rows) {
            return None;
        }

        let token_count = logits_rows.len();
        let logits_element_count = token_count.checked_mul(vocab_rows)?;
        let mut flattened_logits = Vec::with_capacity(logits_element_count);
        for row in logits_rows {
            flattened_logits.extend_from_slice(row);
        }

        let output = find_optional_pipeline_handle(
            &self.bringup.state,
            &self.bringup.metallib.path,
            kernel_name,
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
        record_optional_kernel_result(&self.bringup, kernel_name, output.is_some());
        output
    }
}

impl TokenSampler for MetalBringupSampler {
    fn sample(&self, input: SamplerInput) -> Vec<SampledToken> {
        let requests = input.requests;
        let mut sampled_from_logits = vec![None; requests.len()];

        #[cfg(target_os = "macos")]
        {
            for indices in grouped_sampler_request_indices_by_logits_width(&requests).into_values()
            {
                if let Some(group_results) = collect_grouped_sampler_results_with_item_fallback(
                    &indices,
                    &mut |group_indices| {
                        if group_indices.len() < 2 {
                            return None;
                        }
                        let feedback_key = sampler_batched_group_feedback_key(group_indices.len());
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
                        record_optional_kernel_result(
                            &self.bringup,
                            &feedback_key,
                            output.is_some(),
                        );
                        output
                    },
                    &mut |request_index| {
                        requests.get(request_index).and_then(|request| {
                            request.logits.as_ref().and_then(|logits| {
                                self.sample_argmax_logprob_with_optional_native_path(logits)
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
                            self.sample_argmax_logprob_with_optional_native_path(logits)
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
fn sampler_batched_group_feedback_key(group_size: usize) -> String {
    format!("batched_group:sampler_argmax_logprob:{group_size}")
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
    pub rms_norm_f32_binding_count: u32,
    #[serde(default)]
    pub rms_norm_f16_binding_count: u32,
    #[serde(default)]
    pub rms_norm_bf16_binding_count: u32,
    #[serde(default)]
    pub rms_norm_unsupported_binding_count: u32,
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
        v: MetalNativeTensorBinding,
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
struct MetalNativeLayerBindings {
    attention_norm: MetalNativeTensorBinding,
    attention_q_norm: Option<MetalNativeTensorBinding>,
    attention_k_norm: Option<MetalNativeTensorBinding>,
    attention_qkv: MetalAttentionQkvBindings,
    attention_o: MetalNativeTensorBinding,
    ffn_norm: MetalNativeTensorBinding,
    ffn_gate_up: MetalFfnGateUpBindings,
    ffn_down: MetalNativeTensorBinding,
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
        if let Some(binding) = &layer.attention_q_norm {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        if let Some(binding) = &layer.attention_k_norm {
            record_rms_norm_binding_coverage(&mut coverage, binding);
        }
        match &layer.attention_qkv {
            MetalAttentionQkvBindings::Packed(binding) => {
                record_projection_binding_coverage(&mut coverage, binding);
            }
            MetalAttentionQkvBindings::Split { q, k, v } => {
                record_projection_binding_coverage(&mut coverage, q);
                record_projection_binding_coverage(&mut coverage, k);
                record_projection_binding_coverage(&mut coverage, v);
            }
        }
        record_projection_binding_coverage(&mut coverage, &layer.attention_o);
        record_rms_norm_binding_coverage(&mut coverage, &layer.ffn_norm);
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
                attention_q_norm: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::AttentionQNorm)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                attention_k_norm: artifacts
                    .layer_tensor(layer_index, NativeTensorRole::AttentionKNorm)
                    .map(|spec| MetalNativeTensorBinding::from_spec(artifacts, spec)),
                attention_qkv: attention_qkv_bindings(artifacts, layer_index)?,
                attention_o: required_layer_tensor_binding(
                    artifacts,
                    layer_index,
                    NativeTensorRole::AttentionO,
                    "attention_o",
                )?,
                ffn_norm: required_layer_tensor_binding(
                    artifacts,
                    layer_index,
                    NativeTensorRole::FfnNorm,
                    "ffn_norm",
                )?,
                ffn_gate_up: ffn_gate_up_bindings(artifacts, layer_index)?,
                ffn_down: required_layer_tensor_binding(
                    artifacts,
                    layer_index,
                    NativeTensorRole::FfnDown,
                    "ffn_down",
                )?,
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
            if let Some(q_norm) = &layer.attention_q_norm {
                bindings.push(q_norm.clone());
            }
            if let Some(k_norm) = &layer.attention_k_norm {
                bindings.push(k_norm.clone());
            }
            match &layer.attention_qkv {
                MetalAttentionQkvBindings::Packed(binding) => bindings.push(binding.clone()),
                MetalAttentionQkvBindings::Split { q, k, v } => {
                    bindings.push(q.clone());
                    bindings.push(k.clone());
                    bindings.push(v.clone());
                }
            }
            bindings.push(layer.attention_o.clone());
            bindings.push(layer.ffn_norm.clone());
            match &layer.ffn_gate_up {
                MetalFfnGateUpBindings::Packed(binding) => bindings.push(binding.clone()),
                MetalFfnGateUpBindings::Split { gate, up } => {
                    bindings.push(gate.clone());
                    bindings.push(up.clone());
                }
            }
            bindings.push(layer.ffn_down.clone());
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
        NativeModelBindingSummary {
            bindings_prepared: true,
            buffers_bound: !self.tensors.is_empty(),
            buffer_count: self.tensors.len() as u32,
            buffer_bytes: self.total_bytes,
        }
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
        v: required_layer_tensor_binding(
            artifacts,
            layer_index,
            NativeTensorRole::AttentionV,
            "attention_v",
        )?,
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
    last_dispatch: Mutex<Option<MetalDispatchTrace>>,
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
                .unwrap_or_else(|| NativeModelBindingSummary {
                    bindings_prepared: self.model_bindings.is_some(),
                    buffers_bound: false,
                    buffer_count: 0,
                    buffer_bytes: 0,
                })
        }

        #[cfg(not(target_os = "macos"))]
        {
            NativeModelBindingSummary {
                bindings_prepared: self.model_bindings.is_some(),
                buffers_bound: false,
                buffer_count: 0,
                buffer_bytes: 0,
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn try_model_bound_direct_decode_tokens(
        &self,
        input: &RunnerInput,
        attention_output_bits: &[u32],
        staged_inputs: &MetalDispatchStagedInputs,
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
            Some(&self.bringup),
        )
    }

    #[cfg(not(target_os = "macos"))]
    fn try_model_bound_direct_decode_tokens(
        &self,
        _input: &RunnerInput,
        _attention_output_bits: &[u32],
        _staged_inputs: &MetalDispatchStagedInputs,
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
                trace.runtime.model = runtime.model;

                let direct_decode_result = self.try_model_bound_direct_decode_tokens(
                    &input,
                    &trace.numeric.attention_output_bits,
                    &staged_inputs,
                    &trace.runtime,
                );
                let mut output = successful_runner_output_from_input(&input);
                apply_direct_decode_logits_to_runner_output(
                    &mut output,
                    &direct_decode_result.logits_outputs,
                );
                let complete_model_forward_supported =
                    runtime_reports_complete_model_forward(&trace.runtime);
                let real_model_forward = completed_real_model_forward_step(
                    &input,
                    &output,
                    &trace.runtime,
                    &direct_decode_result.tokens,
                );
                let execution_tally = staged_inputs
                    .prefix_attention_tally
                    .merge(direct_decode_result.execution_tally);
                trace.execution = metal_dispatch_execution_info(
                    &output,
                    &direct_decode_result.tokens,
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
                    &direct_decode_result.tokens,
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
#[allow(clippy::too_many_arguments)]
fn derive_model_bound_direct_decode_result(
    input: &RunnerInput,
    attention_output_bits: &[u32],
    artifacts: Option<&NativeModelArtifacts>,
    bindings: Option<&MetalNativeModelBindings>,
    buffers: Option<&MetalNativeModelBufferBindings>,
    final_layer_hidden_state_cache: Option<&ModelFinalLayerHiddenStateCache>,
    prefix_layer_caches: Option<&[Mutex<MetalPersistentLayerKvCache>]>,
    bringup: Option<&MetalRuntimeBringup>,
) -> ModelBoundDirectDecodeResult {
    let (Some(artifacts), Some(bindings), Some(buffers)) = (artifacts, bindings, buffers) else {
        return ModelBoundDirectDecodeResult::default();
    };
    let Ok(workload) = MetalDispatchWorkload::from_runner_input(input) else {
        return ModelBoundDirectDecodeResult::default();
    };
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
        ) else {
            return ModelBoundDirectDecodeResult::default();
        };
        return derive_model_bound_direct_decode_result_from_hidden_states(
            input,
            attention_output_bits,
            artifacts,
            bindings,
            buffers,
            &workload,
            &hidden_states,
            final_layer_index,
            bringup,
        );
    };
    derive_model_bound_direct_decode_result_from_hidden_states(
        input,
        attention_output_bits,
        artifacts,
        bindings,
        buffers,
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
        if item.mode == ExecutionMode::Decode {
            let token_end = token_base.saturating_add(token_width);
            let hidden_index = attention_index
                .saturating_add(item.scheduled_token_count as usize)
                .saturating_sub(1);
            if let Some(bits) = attention_output_bits.get(token_base..token_end) {
                if let Some((
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
                    model_bound_ffn_decode |= used_model_bound_ffn;
                    native_logits_projection_decode |= used_native_logits_projection;
                    native_dense_tally = native_dense_tally.merge(decode_native_dense_tally);
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
    dims: ModelBoundDecodeDims,
    hidden: Vec<f32>,
    attention_input: Vec<f32>,
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn derive_model_bound_direct_decode_result_from_hidden_states_batched(
    input: &RunnerInput,
    attention_output_bits: &[u32],
    artifacts: &NativeModelArtifacts,
    bindings: &MetalNativeModelBindings,
    buffers: &MetalNativeModelBufferBindings,
    workload: &MetalDispatchWorkload,
    hidden_states: &[Vec<f32>],
    final_layer_index: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<ModelBoundDirectDecodeResult> {
    let final_layer = bindings.layers.get(final_layer_index)?;
    let attention_o = buffers.binding_for(&final_layer.attention_o)?;
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
        if item.mode == ExecutionMode::Decode {
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
                dims,
                hidden: residual,
                attention_input: attention_output.get(..dims.input_width)?.to_vec(),
            });
        }
        attention_index = attention_index.checked_add(item.scheduled_token_count as usize)?;
    }

    if prepared.is_empty() {
        return Some(ModelBoundDirectDecodeResult::default());
    }

    let prepared_request_order = prepared
        .iter()
        .map(|item| item.request_id)
        .collect::<Vec<_>>();
    let mut grouped_prepared =
        BTreeMap::<ModelBoundDecodeDims, Vec<PreparedDirectDecodeItem>>::new();
    for item in prepared {
        grouped_prepared.entry(item.dims).or_default().push(item);
    }
    let mut tokens_by_request = BTreeMap::new();
    let mut logits_by_request = BTreeMap::new();
    let mut native_logits_projection_decode = false;
    let mut model_bound_ffn_decode = false;
    let mut execution_tally = PrefixAttentionExecutionTally::default();
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default();

    let grouped_results = collect_model_bound_direct_decode_group_results_with_item_fallback(
        grouped_prepared.into_values(),
        |group| {
            let feedback_key = bringup.and_then(|bringup| {
                let dims = group.first()?.dims;
                Some((
                    bringup,
                    direct_decode_batched_group_feedback_key(group.len(), dims),
                ))
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
                ffn_norm,
                ffn_down,
                final_norm,
                decode_projection,
                group,
                bringup,
            );
            if let Some((bringup, feedback_key)) = feedback_key.as_ref() {
                record_optional_kernel_result(bringup, feedback_key, output.is_some());
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

    let fallback_tally =
        DirectDecodeNativeDenseTally::default().record_batched_group_fallback(group.len());
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
    ffn_norm: &MetalNativeTensorBufferBinding,
    ffn_down: &MetalNativeTensorBufferBinding,
    final_norm: &MetalNativeTensorBufferBinding,
    decode_projection: &MetalNativeTensorBufferBinding,
    mut prepared: Vec<PreparedDirectDecodeItem>,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<ModelBoundDirectDecodeResult> {
    if prepared.is_empty() {
        return Some(ModelBoundDirectDecodeResult::default());
    }

    let dims = prepared.first()?.dims;
    if prepared.iter().any(|item| {
        item.dims != dims
            || item.hidden.len() < dims.hidden_dim
            || item.attention_input.len() < dims.input_width
    }) {
        return None;
    }

    let attention_input_rows = prepared
        .iter()
        .map(|item| item.attention_input.clone())
        .collect::<Vec<_>>();
    let (attention_hidden_rows, attention_o_tally) = project_batched_matrix_rows_with_tally(
        attention_o,
        0,
        dims.hidden_dim,
        &attention_input_rows,
        dims.input_width,
        bringup,
    )?;
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default().merge(attention_o_tally);
    for (item, attention_hidden) in prepared.iter_mut().zip(attention_hidden_rows.into_iter()) {
        add_in_place(&mut item.hidden, &attention_hidden);
    }

    let mut ffn_hidden_rows = prepared
        .iter()
        .map(|item| item.hidden.clone())
        .collect::<Vec<_>>();
    native_dense_tally = native_dense_tally.merge(
        direct_decode_native_dense_tally_from_prefix_attention_tally(
            apply_batched_row_rms_norm_with_binding_in_place_with_tally(
                &mut ffn_hidden_rows,
                dims.hidden_dim,
                ffn_norm,
                native_model_rms_norm_epsilon(artifacts),
                native_model_rms_norm_weight_offset(artifacts),
                bringup,
            )?,
        ),
    );

    let mut model_bound_ffn_decode = false;
    let (mut gate_rows, up_rows, ffn_gate_up_tally) = project_batched_ffn_gate_up_with_tally(
        &final_layer.ffn_gate_up,
        buffers,
        dims.intermediate_dim,
        &ffn_hidden_rows,
        dims.hidden_dim,
        bringup,
    )?;
    native_dense_tally = native_dense_tally.merge(ffn_gate_up_tally);
    native_dense_tally =
        native_dense_tally.merge(apply_batched_model_gate_up_product_in_place_with_tally(
            artifacts,
            &mut gate_rows,
            &up_rows,
            dims.intermediate_dim,
            bringup,
        )?);
    let (ffn_output_rows, ffn_down_tally) = project_batched_matrix_rows_with_tally(
        ffn_down,
        0,
        dims.hidden_dim,
        &gate_rows,
        dims.intermediate_dim,
        bringup,
    )?;
    native_dense_tally = native_dense_tally.merge(ffn_down_tally);
    for (item, ffn_output) in prepared.iter_mut().zip(ffn_output_rows.into_iter()) {
        model_bound_ffn_decode |= has_nontrivial_ffn_contribution(&ffn_output);
        add_in_place(&mut item.hidden, &ffn_output);
    }

    let mut final_hidden_rows = prepared
        .iter()
        .map(|item| item.hidden.clone())
        .collect::<Vec<_>>();
    native_dense_tally = native_dense_tally.merge(
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
    );

    let token_count = final_hidden_rows.len();
    let mut batched_logits_results = if token_count > 1 {
        Some(
            project_batched_decode_logits_with_optional_native_path(
                bringup,
                decode_projection,
                &final_hidden_rows,
                dims.hidden_dim,
            )?
            .into_iter(),
        )
    } else {
        None
    };
    if batched_logits_results.is_some() {
        native_dense_tally = native_dense_tally.record_batched_logits_group(token_count);
    }
    let mut tokens = Vec::with_capacity(prepared.len());
    let mut logits_outputs = Vec::with_capacity(prepared.len());
    let mut native_logits_projection_decode = false;
    let mut execution_tally = PrefixAttentionExecutionTally::default();

    for (item, hidden) in prepared.into_iter().zip(final_hidden_rows.into_iter()) {
        let (logits, token_id, used_native_logits_projection) =
            if let Some(batched_logits_results) = batched_logits_results.as_mut() {
                let (logits, token_id) = batched_logits_results.next()?;
                (logits, token_id, true)
            } else if let Some((logits, token_id)) =
                project_decode_logits_with_optional_native_path(bringup, decode_projection, &hidden)
            {
                (logits, token_id, true)
            } else if token_count == 1 {
                let (logits, token_id) =
                    project_decode_logits_cpu(decode_projection, item.dims.hidden_dim, &hidden)?;
                (logits, token_id, false)
            } else {
                return None;
            };
        native_dense_tally = native_dense_tally
            .record_projection_rows(item.dims.vocab_rows, used_native_logits_projection);
        native_logits_projection_decode |= used_native_logits_projection;
        execution_tally = execution_tally
            .record_layer_continuation_tokens(1)
            .record_logits_projection(1, u32::try_from(item.dims.vocab_rows).unwrap_or(u32::MAX));
        tokens.push((item.request_id, token_id));
        logits_outputs.push(RequestLogitsOutput {
            request_id: item.request_id,
            logits,
        });
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
        bringup,
    )
    .tokens
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
    let attention_o = buffers.binding_for(&layer.attention_o)?;
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
    let mut native_dense_tally = DirectDecodeNativeDenseTally::default();
    let residual = final_layer_hidden_state.get(..dims.hidden_dim)?.to_vec();
    let (mut hidden, used_native_attention_o_projection) = project_matrix_rows_with_path(
        attention_o,
        0,
        dims.hidden_dim,
        &attention_output[..dims.input_width],
        bringup,
    )?;
    native_dense_tally = native_dense_tally
        .record_projection_rows(dims.hidden_dim, used_native_attention_o_projection);
    add_in_place(&mut hidden, &residual);

    let mut ffn_hidden = hidden.clone();
    let used_native_ffn_norm = apply_rms_norm_with_binding_in_place_with_path(
        &mut ffn_hidden,
        ffn_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        bringup,
    )?;
    native_dense_tally =
        native_dense_tally.record_rms_norm_elements(ffn_hidden.len(), used_native_ffn_norm);
    let (mut gate, up, ffn_gate_up_tally) = project_ffn_gate_up_with_coverage(
        &layer.ffn_gate_up,
        buffers,
        dims.intermediate_dim,
        &ffn_hidden,
        bringup,
    )?;
    native_dense_tally = native_dense_tally.merge(ffn_gate_up_tally);
    let used_native_gate_up_product =
        apply_model_gate_up_product_with_path(artifacts, &mut gate, &up, bringup)?;
    native_dense_tally =
        native_dense_tally.record_ffn_activation_elements(gate.len(), used_native_gate_up_product);
    let (ffn_output, used_native_ffn_down_projection) =
        project_matrix_rows_with_path(ffn_down, 0, dims.hidden_dim, &gate, bringup)?;
    native_dense_tally =
        native_dense_tally.record_projection_rows(dims.hidden_dim, used_native_ffn_down_projection);
    let used_model_bound_ffn = has_nontrivial_ffn_contribution(&ffn_output);
    add_in_place(&mut hidden, &ffn_output);

    let used_native_final_norm = apply_rms_norm_with_binding_in_place_with_path(
        &mut hidden,
        final_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        bringup,
    )?;
    native_dense_tally =
        native_dense_tally.record_rms_norm_elements(hidden.len(), used_native_final_norm);

    let (logits, token_id, used_native_logits_projection) = if let Some((logits, token_id)) =
        project_decode_logits_with_optional_native_path(bringup, decode_projection, &hidden)
    {
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
fn project_decode_logits_with_optional_native_path(
    bringup: Option<&MetalRuntimeBringup>,
    decode_projection: &MetalNativeTensorBufferBinding,
    hidden: &[f32],
) -> Option<(Vec<f32>, u32)> {
    let bringup = bringup?;
    let projection_kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel_name(decode_projection.native_dtype)?;
    let argmax_kernel_name = "logits_argmax_f32";
    if !bringup
        .state
        .optional_kernel_dispatch_plan
        .logits_argmax_f32
    {
        return None;
    }
    if !optional_kernel_allowed(bringup, projection_kernel_name)
        || !optional_kernel_allowed(bringup, argmax_kernel_name)
    {
        return None;
    }

    let (vocab_rows, projection_cols) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let input_width = hidden.len().min(projection_cols);
    if vocab_rows == 0 || input_width == 0 {
        return None;
    }

    let output = find_optional_pipeline_handle(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
    )
    .ok()
    .zip(
        find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, argmax_kernel_name)
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
    record_optional_kernel_result(bringup, projection_kernel_name, success);
    record_optional_kernel_result(bringup, argmax_kernel_name, success);
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

    let projection_kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel_name(decode_projection.native_dtype)?;
    let argmax_kernel_name = "logits_argmax_batched_f32";
    if !bringup
        .state
        .optional_kernel_dispatch_plan
        .logits_argmax_batched_f32
    {
        return None;
    }
    if !optional_kernel_allowed(bringup, projection_kernel_name)
        || !optional_kernel_allowed(bringup, argmax_kernel_name)
    {
        return None;
    }
    let (vocab_rows, projection_cols) = tensor_matrix_dimensions(&decode_projection.meta.spec)?;
    let input_width = hidden_width.min(projection_cols);
    if vocab_rows == 0 || input_width == 0 {
        return None;
    }

    let token_count = hidden_rows.len();
    let logits_element_count = token_count.checked_mul(vocab_rows)?;
    let mut flattened_hidden = Vec::with_capacity(token_count.checked_mul(hidden_width)?);
    for row in hidden_rows {
        flattened_hidden.extend_from_slice(row.get(..hidden_width)?);
    }

    let output = find_optional_pipeline_handle(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
    )
    .ok()
    .zip(
        find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, argmax_kernel_name)
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
                saturating_usize_to_u32(hidden_width),
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
    record_optional_kernel_result(bringup, projection_kernel_name, success);
    record_optional_kernel_result(bringup, argmax_kernel_name, success);
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
) -> String {
    format!(
        "batched_group:direct_decode:{group_size}:{}:{}:{}:{}",
        dims.input_width, dims.hidden_dim, dims.intermediate_dim, dims.vocab_rows
    )
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
fn project_ffn_gate_up_with_coverage(
    ffn_gate_up: &MetalFfnGateUpBindings,
    buffers: &MetalNativeModelBufferBindings,
    intermediate_dim: usize,
    input: &[f32],
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<f32>, Vec<f32>, DirectDecodeNativeDenseTally)> {
    match ffn_gate_up {
        MetalFfnGateUpBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (gate, used_native_gate) =
                project_matrix_rows_with_path(packed, 0, intermediate_dim, input, bringup)?;
            let (up, used_native_up) = project_matrix_rows_with_path(
                packed,
                intermediate_dim,
                intermediate_dim,
                input,
                bringup,
            )?;
            let tally = DirectDecodeNativeDenseTally::default()
                .record_projection_rows(intermediate_dim, used_native_gate)
                .record_projection_rows(intermediate_dim, used_native_up);
            Some((gate, up, tally))
        }
        MetalFfnGateUpBindings::Split { gate, up } => {
            let gate_binding = buffers.binding_for(gate)?;
            let up_binding = buffers.binding_for(up)?;
            let (gate, used_native_gate) =
                project_matrix_rows_with_path(gate_binding, 0, intermediate_dim, input, bringup)?;
            let (up, used_native_up) =
                project_matrix_rows_with_path(up_binding, 0, intermediate_dim, input, bringup)?;
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

fn apply_direct_decode_logits_to_runner_output(
    output: &mut RunnerOutput,
    logits_outputs: &[RequestLogitsOutput],
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
    output.logits_outputs.extend(logits_outputs.iter().cloned());
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
        updates_by_request_id
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
        updates_by_request_id.get(request_id).is_some_and(|update| {
            update.output_token.is_some() || logits_output_request_ids.contains(request_id)
        })
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
        MetalStagedInputSource::ModelConditionedNativePrefixAttention
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
            optional_pipeline_lookup,
            optional_kernel_dispatch_plan,
            optional_kernel_feedback: Mutex::new(MetalOptionalKernelFeedbackState::default()),
            dispatch_arena: Mutex::new(None),
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
fn find_optional_pipeline_handle<'a>(
    state: &'a MetalRuntimeState,
    metallib_path: &Path,
    function_name: &str,
) -> Result<&'a MetalPipelineHandle, MetalRuntimeError> {
    find_pipeline_handle(
        &state.optional_pipelines,
        &state.optional_pipeline_lookup,
        metallib_path,
        function_name,
        "optional",
    )
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
    let has = |name: &str| optional_pipeline_lookup.contains_key(name);
    MetalOptionalKernelDispatchPlan {
        projection_f32: has("decode_logits_projection_f32"),
        projection_f16: has("decode_logits_projection_f16"),
        projection_bf16: has("decode_logits_projection_bf16"),
        batched_projection_f32: has("decode_logits_projection_batched_f32"),
        batched_projection_f16: has("decode_logits_projection_batched_f16"),
        batched_projection_bf16: has("decode_logits_projection_batched_bf16"),
        embedding_gather_f32: has("gather_embedding_rows_f32"),
        embedding_gather_f16: has("gather_embedding_rows_f16"),
        embedding_gather_bf16: has("gather_embedding_rows_bf16"),
        rms_norm_f32: has("rms_norm_f32"),
        rms_norm_f16: has("rms_norm_f16"),
        rms_norm_bf16: has("rms_norm_bf16"),
        batched_rms_norm_f32: has("rms_norm_batched_f32"),
        batched_rms_norm_f16: has("rms_norm_batched_f16"),
        batched_rms_norm_bf16: has("rms_norm_batched_bf16"),
        logits_argmax_f32: has("logits_argmax_f32"),
        logits_argmax_batched_f32: has("logits_argmax_batched_f32"),
        sample_argmax_logprob_f32: has("sample_argmax_logprob_f32"),
        sample_argmax_logprob_batched_f32: has("sample_argmax_logprob_batched_f32"),
        apply_rope_f32: has("apply_rope_f32"),
        apply_rope_batched_f32: has("apply_rope_batched_f32"),
        expand_grouped_kv_heads_f32: has("expand_grouped_kv_heads_f32"),
        ffn_gate_silu_product_f32: has("ffn_gate_silu_product_f32"),
        ffn_gate_gelu_approx_product_f32: has("ffn_gate_gelu_approx_product_f32"),
    }
}

#[cfg(target_os = "macos")]
fn optional_kernel_allowed(bringup: &MetalRuntimeBringup, kernel_name: &str) -> bool {
    let Ok(feedback) = bringup.state.optional_kernel_feedback.lock() else {
        return true;
    };
    optional_kernel_allowed_in_feedback_state(&feedback, kernel_name)
}

#[cfg(target_os = "macos")]
fn optional_kernel_allowed_in_feedback_state(
    feedback: &MetalOptionalKernelFeedbackState,
    kernel_name: &str,
) -> bool {
    !feedback.disabled_kernels.contains(kernel_name)
}

#[cfg(target_os = "macos")]
fn record_optional_kernel_result(bringup: &MetalRuntimeBringup, kernel_name: &str, success: bool) {
    let Ok(mut feedback) = bringup.state.optional_kernel_feedback.lock() else {
        return;
    };
    record_optional_kernel_feedback_state(&mut feedback, kernel_name, success);
}

#[cfg(target_os = "macos")]
fn record_optional_kernel_feedback_state(
    feedback: &mut MetalOptionalKernelFeedbackState,
    kernel_name: &str,
    success: bool,
) {
    if success {
        feedback.consecutive_failures_by_kernel.remove(kernel_name);
        feedback.disabled_kernels.remove(kernel_name);
        return;
    }
    let consecutive_failures = feedback
        .consecutive_failures_by_kernel
        .entry(kernel_name.to_string())
        .or_insert(0);
    *consecutive_failures = consecutive_failures.saturating_add(1);
    if *consecutive_failures >= PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
        feedback.disabled_kernels.insert(kernel_name.to_string());
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
struct ModelStageRopeDispatchParams {
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
    attention_norm: &MetalNativeTensorBufferBinding,
    attention_qkv: &MetalAttentionQkvBindings,
    buffers: &MetalNativeModelBufferBindings,
    input_width: usize,
) -> Option<ModelStageDims> {
    let norm_len = tensor_element_count(&attention_norm.meta.spec)?;
    let mut input_dim = (artifacts.manifest().hidden_size as usize)
        .min(input_width)
        .min(norm_len);

    match attention_qkv {
        MetalAttentionQkvBindings::Packed(binding) => {
            let packed = buffers.binding_for(binding)?;
            let (_, cols) = tensor_matrix_dimensions(&packed.meta.spec)?;
            input_dim = input_dim.min(cols);
        }
        MetalAttentionQkvBindings::Split { q, k, v } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let v_binding = buffers.binding_for(v)?;
            input_dim = input_dim
                .min(tensor_matrix_dimensions(&q_binding.meta.spec)?.1)
                .min(tensor_matrix_dimensions(&k_binding.meta.spec)?.1)
                .min(tensor_matrix_dimensions(&v_binding.meta.spec)?.1);
        }
    }

    let q_heads = artifacts.manifest().attention_head_count as usize;
    let kv_heads = artifacts.manifest().kv_head_count as usize;
    let head_dim = artifacts.manifest().attention_head_dim as usize;
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

    let rotary_dim = stage_dims.head_dim;
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
        apply_rope_style_in_place(head, &cos_table, &sin_table, rope_style);
    }
    for head in key[..key_len].chunks_exact_mut(stage_dims.head_dim) {
        apply_rope_style_in_place(head, &cos_table, &sin_table, rope_style);
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

    if let Some((native_queries, native_keys)) =
        apply_batched_model_stage_rope_with_optional_native_path(
            bringup,
            query_rows,
            key_rows,
            positions,
            stage_dims,
            rope_style,
            native_model_rope_theta(artifacts),
        )
    {
        for (query_row, native_query) in query_rows.iter_mut().zip(native_queries.into_iter()) {
            if query_row.len() != native_query.len() {
                return None;
            }
            query_row.copy_from_slice(&native_query);
        }
        for (key_row, native_key) in key_rows.iter_mut().zip(native_keys.into_iter()) {
            if key_row.len() != native_key.len() {
                return None;
            }
            key_row.copy_from_slice(&native_key);
        }
        return Some(());
    }
    if bringup.is_some() && query_rows.len() > 1 {
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
            bringup,
        );
    }

    Some(())
}

#[cfg(target_os = "macos")]
fn model_stage_rope_style(artifacts: &NativeModelArtifacts) -> ModelStageRopeStyle {
    let family = artifacts.manifest().model_family.to_ascii_lowercase();
    if family.starts_with("qwen") {
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

    let kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .embedding_gather_kernel_name(token_embedding.native_dtype)?;
    if !optional_kernel_allowed(bringup, kernel_name) {
        return None;
    }
    let output_element_count = workload.scheduled_token_ids.len().checked_mul(hidden_dim)?;

    let output = find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, kernel_name)
        .ok()
        .and_then(|pipeline| {
            autoreleasepool(|| {
                let token_ids_buffer = new_shared_buffer_with_data(
                    &bringup.state.device,
                    &workload.scheduled_token_ids,
                );
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
    record_optional_kernel_result(bringup, kernel_name, output.is_some());
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
        attention_norm,
        &layer.attention_qkv,
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
    let (mut projected_queries, mut projected_keys, projected_values, qkv_tally) =
        project_batched_attention_qkv_with_dims_and_tally(
            artifacts,
            &layer.attention_qkv,
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
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    if input.execution_batch.items.is_empty() {
        return Some((Vec::new(), PrefixAttentionExecutionTally::default()));
    }

    let ephemeral_layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());
    let shared_layer_cache = layer_cache.or(Some(&ephemeral_layer_cache));
    let attention_o = buffers.binding_for(&layer.attention_o)?;
    let ffn_norm = buffers.binding_for(&layer.ffn_norm)?;
    let ffn_down = buffers.binding_for(&layer.ffn_down)?;
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

    let (next_hidden_states, continuation_tally) =
        project_hidden_states_from_layer_attention_output(
            artifacts,
            layer,
            buffers,
            attention_o,
            ffn_norm,
            ffn_down,
            hidden_states,
            &attention_output,
            attention_head_size,
            bringup,
        )?;
    let continuation_tokens = u32::try_from(next_hidden_states.len()).unwrap_or(u32::MAX);
    prefix_attention_tally = prefix_attention_tally
        .merge(continuation_tally)
        .record_layer_continuation_tokens(continuation_tokens);

    Some((next_hidden_states, prefix_attention_tally))
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

    let group_items = input.execution_batch.items.get(item_range.clone())?;
    let group_token_range =
        token_range_for_execution_item_range(item_token_ranges, item_range.clone())?;
    let group_input = runner_input_for_execution_items(input, group_items)?;
    let group_workload = MetalDispatchWorkload::from_runner_input(&group_input).ok()?;
    let group_staged_inputs =
        slice_staged_inputs_for_token_range(staged_inputs, group_token_range)?;
    if native_prefix_attention_is_viable(&group_workload, bringup.is_some(), layer_cache) {
        if let Some(mut native_attempt) = try_native_attention_output_from_model_layer(
            artifacts,
            &group_workload,
            &group_staged_inputs,
            bringup,
            layer_cache,
        ) {
            commit_prefix_attention_dispatch_attempt(layer_cache, &mut native_attempt);
            return Some((
                native_attempt.attention_output,
                PrefixAttentionExecutionTally::default().record(true),
            ));
        }
    } else if bringup.is_some() && group_items.len() > 1 {
        let split_index = item_range.start.checked_add(group_items.len() / 2)?;
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
        return Some((left_output, left_tally.merge(right_tally)));
    }
    let mut attempt = resolve_attention_output_from_model_layer(
        artifacts,
        &group_workload,
        &group_staged_inputs,
        bringup,
        layer_cache,
    )?;
    commit_prefix_attention_dispatch_attempt(layer_cache, &mut attempt);
    Some((
        attempt.attention_output,
        PrefixAttentionExecutionTally::default().record(attempt.used_native_dispatch),
    ))
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

    let (attention_output, used_native_dispatch) =
        attention_output_from_model_layer(artifacts, workload, staged_inputs, bringup, None)?;
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
    ffn_norm: &MetalNativeTensorBufferBinding,
    ffn_down: &MetalNativeTensorBufferBinding,
    hidden_states: &[Vec<f32>],
    attention_output: &[f32],
    attention_head_size: usize,
    bringup: Option<&MetalRuntimeBringup>,
) -> Option<(Vec<Vec<f32>>, PrefixAttentionExecutionTally)> {
    let hidden_width = hidden_states.iter().map(Vec::len).min()?;
    let dims = resolved_model_reference_layer_dims(
        attention_o,
        ffn_norm,
        &layer.ffn_gate_up,
        buffers,
        ffn_down,
        hidden_width,
        attention_head_size,
    )?;
    let attention_input_rows = hidden_states
        .iter()
        .enumerate()
        .map(|(token_index, _)| {
            let token_base = token_index.checked_mul(attention_head_size)?;
            let token_end = token_base.checked_add(attention_head_size)?;
            let token_attention_output = attention_output.get(token_base..token_end)?;
            Some(token_attention_output.get(..dims.input_width)?.to_vec())
        })
        .collect::<Option<Vec<_>>>()?;
    let (attention_hidden_rows, attention_projection_tally) =
        project_batched_matrix_rows_with_tally(
            attention_o,
            0,
            dims.hidden_dim,
            &attention_input_rows,
            dims.input_width,
            bringup,
        )?;
    let mut hidden_after_attention = Vec::with_capacity(hidden_states.len());
    let mut tally = PrefixAttentionExecutionTally::default();
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
        attention_projection_tally,
    ));

    for (hidden_state, mut hidden) in hidden_states.iter().zip(attention_hidden_rows.into_iter()) {
        let residual = hidden_state.get(..dims.hidden_dim)?.to_vec();
        add_in_place(&mut hidden, &residual);
        hidden_after_attention.push(hidden);
    }

    let mut ffn_hidden_states = hidden_after_attention.clone();
    tally = tally.merge(apply_batched_row_rms_norm_with_binding_in_place_with_tally(
        &mut ffn_hidden_states,
        dims.hidden_dim,
        ffn_norm,
        native_model_rms_norm_epsilon(artifacts),
        native_model_rms_norm_weight_offset(artifacts),
        bringup,
    )?);

    let (mut gate_rows, up_rows, gate_up_tally) = project_batched_ffn_gate_up_with_tally(
        &layer.ffn_gate_up,
        buffers,
        dims.intermediate_dim,
        &ffn_hidden_states,
        dims.hidden_dim,
        bringup,
    )?;
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
        gate_up_tally,
    ));
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
        apply_batched_model_gate_up_product_in_place_with_tally(
            artifacts,
            &mut gate_rows,
            &up_rows,
            dims.intermediate_dim,
            bringup,
        )?,
    ));
    let (ffn_output_rows, ffn_down_tally) = project_batched_matrix_rows_with_tally(
        ffn_down,
        0,
        dims.hidden_dim,
        &gate_rows,
        dims.intermediate_dim,
        bringup,
    )?;
    tally = tally.merge(prefix_attention_tally_from_native_dense_tally(
        ffn_down_tally,
    ));

    let mut next_hidden_states = Vec::with_capacity(hidden_after_attention.len());
    for (mut hidden, ffn_output) in hidden_after_attention
        .into_iter()
        .zip(ffn_output_rows.into_iter())
    {
        add_in_place(&mut hidden, &ffn_output);
        next_hidden_states.push(hidden);
    }

    Some((next_hidden_states, tally))
}

#[cfg(target_os = "macos")]
fn attention_output_from_model_layer(
    artifacts: &NativeModelArtifacts,
    workload: &MetalDispatchWorkload,
    staged_inputs: &MetalDispatchStagedInputs,
    bringup: Option<&MetalRuntimeBringup>,
    layer_cache: Option<&Mutex<MetalPersistentLayerKvCache>>,
) -> Option<(Vec<f32>, bool)> {
    let numeric_workload = workload.with_numeric_layout(staged_inputs.layout);
    let attention_config =
        native_model_reference_attention_config(artifacts, staged_inputs.layout.head_dim as usize);
    if let Some(layer_cache) = layer_cache {
        let mut layer_cache = layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned");
        let workload_is_self_contained =
            workload_supports_native_prefix_attention(&numeric_workload, None);
        let native_trace = {
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
        };
        if let Some((trace, cache_snapshot)) = native_trace {
            if let Some(attention_output) =
                decode_attention_output_values(&trace.numeric.attention_output_bits)
            {
                layer_cache.apply_snapshot(&numeric_workload, &cache_snapshot);
                return Some((attention_output, true));
            }
        }
        if workload_is_self_contained {
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
        return gathered_slots_support_native_prefix_attention(workload, &gathered_slots, |slot| {
            layer_cache.slot_initialized(slot)
        });
    }

    gathered_slots_support_native_prefix_attention(workload, &gathered_slots, |_| false)
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
    if !bringup.state.optional_kernel_dispatch_plan.apply_rope_f32 {
        return None;
    }
    let kernel_name = "apply_rope_f32";
    if !optional_kernel_allowed(bringup, kernel_name) {
        return None;
    }
    let (cos_table, sin_table) = build_model_stage_rope_tables(head_dim, position, freq_base);

    let output = find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, kernel_name)
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
    record_optional_kernel_result(bringup, kernel_name, output.is_some());
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
) -> Option<BatchedModelStageRopeRows> {
    let bringup = bringup?;
    if !freq_base.is_finite() || freq_base <= 0.0 {
        return None;
    }

    let head_dim = stage_dims.head_dim;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
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
    if !bringup
        .state
        .optional_kernel_dispatch_plan
        .apply_rope_batched_f32
    {
        return None;
    }
    let kernel_name = "apply_rope_batched_f32";
    if !optional_kernel_allowed(bringup, kernel_name) {
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

    let output = find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, kernel_name)
        .ok()
        .and_then(|pipeline| {
            autoreleasepool(|| {
                let query_buffer =
                    new_shared_buffer_with_data(&bringup.state.device, &flattened_query);
                let key_buffer = new_shared_buffer_with_data(&bringup.state.device, &flattened_key);
                let positions_buffer =
                    new_shared_buffer_with_data(&bringup.state.device, &positions);

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
    record_optional_kernel_result(bringup, kernel_name, output.is_some());
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
            let k_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let v_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            if q_rows < query_output_dim || k_rows < kv_output_dim || v_rows < kv_output_dim {
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
                q_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            let (value, value_tally) = project_matrix_head_prefix_with_tally(
                packed,
                q_rows + k_rows,
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
        MetalAttentionQkvBindings::Split { q, k, v } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let v_binding = buffers.binding_for(v)?;
            let (query, query_tally) = project_matrix_head_prefix_with_tally(
                q_binding,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            let (key, key_tally) = project_matrix_head_prefix_with_tally(
                k_binding,
                0,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input,
                bringup,
            )?;
            let (value, value_tally) = project_matrix_head_prefix_with_tally(
                v_binding,
                0,
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
            let k_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            let v_rows = artifacts.manifest().kv_head_count as usize
                * artifacts.manifest().attention_head_dim as usize;
            if q_rows < query_output_dim || k_rows < kv_output_dim || v_rows < kv_output_dim {
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
                q_rows,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            let (value, value_tally) = project_batched_matrix_head_prefix_with_tally(
                packed,
                q_rows + k_rows,
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
        MetalAttentionQkvBindings::Split { q, k, v } => {
            let q_binding = buffers.binding_for(q)?;
            let k_binding = buffers.binding_for(k)?;
            let v_binding = buffers.binding_for(v)?;
            let (query, query_tally) = project_batched_matrix_head_prefix_with_tally(
                q_binding,
                0,
                stage_dims.q_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            let (key, key_tally) = project_batched_matrix_head_prefix_with_tally(
                k_binding,
                0,
                stage_dims.kv_heads,
                stage_dims.head_dim,
                artifacts.manifest().attention_head_dim as usize,
                input_rows,
                input_width,
                bringup,
            )?;
            let (value, value_tally) = project_batched_matrix_head_prefix_with_tally(
                v_binding,
                0,
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

    if let Some(expanded) = expand_batched_grouped_kv_heads_with_optional_native_path(
        rows, q_heads, kv_heads, head_dim, bringup,
    ) {
        return Some(expanded);
    }
    if bringup.is_some() && rows.len() > 1 {
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
    if !bringup
        .state
        .optional_kernel_dispatch_plan
        .expand_grouped_kv_heads_f32
    {
        return None;
    }
    let kernel_name = "expand_grouped_kv_heads_f32";
    if !optional_kernel_allowed(bringup, kernel_name) {
        return None;
    }

    let output = find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, kernel_name)
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
    record_optional_kernel_result(bringup, kernel_name, output.is_some());
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
    if !bringup
        .state
        .optional_kernel_dispatch_plan
        .expand_grouped_kv_heads_f32
    {
        return None;
    }
    let kernel_name = "expand_grouped_kv_heads_f32";
    if !optional_kernel_allowed(bringup, kernel_name) {
        return None;
    }

    let mut flattened_input = Vec::with_capacity(flattened_input_len);
    for row in rows {
        flattened_input.extend_from_slice(row);
    }

    let output = find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, kernel_name)
        .ok()
        .and_then(|pipeline| {
            autoreleasepool(|| {
                let input_buffer =
                    new_shared_buffer_with_data(&bringup.state.device, &flattened_input);
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
    record_optional_kernel_result(bringup, kernel_name, output.is_some());
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
    let rms_norm_kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .rms_norm_kernel_name(weight_binding.native_dtype)?;
    if values.is_empty() || !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    if !optional_kernel_allowed(bringup, rms_norm_kernel_name) {
        return None;
    }

    let weight_len = tensor_element_count(&weight_binding.meta.spec)?;
    if values.len() > weight_len {
        return None;
    }

    let output =
        find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, rms_norm_kernel_name)
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

                    let output = read_shared_buffer_prefix(
                        &output_buffer,
                        saturating_usize_to_u32(values.len()),
                    );
                    (output.len() == values.len() && output.iter().all(|value| value.is_finite()))
                        .then_some(output)
                })
            });
    record_optional_kernel_result(bringup, rms_norm_kernel_name, output.is_some());
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
    for head in values.chunks_exact_mut(head_dim) {
        let used_native = apply_rms_norm_with_binding_in_place_with_path(
            head,
            weight_binding,
            epsilon,
            weight_offset,
            bringup,
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

    let row_count = rows.len();
    let mut flattened = Vec::with_capacity(row_count.checked_mul(row_width)?);
    for row in rows.iter() {
        flattened.extend_from_slice(row.get(..row_width)?);
    }

    if let Some(output) = apply_batched_per_head_rms_norm_with_optional_native_path(
        bringup,
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
                .record_rms_norm_elements(flattened.len(), true),
        );
    }
    if bringup.is_some() && rows.len() > 1 {
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
            bringup,
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

    let kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_rms_norm_kernel_name(weight_binding.native_dtype)?;
    if !optional_kernel_allowed(bringup, kernel_name) {
        return None;
    }
    let element_count = saturating_usize_to_u32(values.len());

    let output = find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, kernel_name)
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
    record_optional_kernel_result(bringup, kernel_name, output.is_some());
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
        apply_model_gate_up_product_with_optional_native_path(bringup, activation, gate, up)
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
    let mut flattened_gate = Vec::with_capacity(flattened_len);
    let mut flattened_up = Vec::with_capacity(flattened_len);
    for (gate_row, up_row) in gate_rows.iter().zip(up_rows.iter()) {
        flattened_gate.extend_from_slice(gate_row.get(..row_width)?);
        flattened_up.extend_from_slice(up_row.get(..row_width)?);
    }

    let activation = native_model_ffn_activation(artifacts);
    if let Some(output) = apply_model_gate_up_product_with_optional_native_path(
        bringup,
        activation,
        &flattened_gate,
        &flattened_up,
    ) {
        for (gate_row, output_row) in gate_rows.iter_mut().zip(output.chunks_exact(row_width)) {
            gate_row.get_mut(..row_width)?.copy_from_slice(output_row);
        }
        return Some(
            DirectDecodeNativeDenseTally::default()
                .record_ffn_activation_elements(flattened_len, true),
        );
    }
    if bringup.is_some() && gate_rows.len() > 1 {
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
            bringup,
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
) -> Option<Vec<f32>> {
    let bringup = bringup?;
    if gate.is_empty() || gate.len() != up.len() {
        return None;
    }

    let kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .ffn_gate_product_kernel_name(activation)?;
    if !optional_kernel_allowed(bringup, kernel_name) {
        return None;
    }

    let output = find_optional_pipeline_handle(&bringup.state, &bringup.metallib.path, kernel_name)
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
                set_ffn_gate_product_dispatch_params(
                    encoder,
                    3,
                    saturating_usize_to_u32(gate.len()),
                );
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
    record_optional_kernel_result(bringup, kernel_name, output.is_some());
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
    if bringup.is_some() && input_rows.len() > 1 {
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
            bringup,
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
            bringup,
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
        for (output_row, projected_row) in output_rows.iter_mut().zip(projected_rows.into_iter()) {
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
    let projection_kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .batched_projection_kernel_name(binding.native_dtype)?;
    if !optional_kernel_allowed(bringup, projection_kernel_name) {
        return None;
    }

    let (_, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input_width > cols || output_dim == 0 || input_rows.is_empty() {
        return None;
    }

    let row_byte_offset = row_offset
        .checked_mul(cols)?
        .checked_mul(native_dtype_size_bytes(binding.native_dtype))?;
    let row_count = input_rows.len();
    let output_element_count = row_count.checked_mul(output_dim)?;
    let hidden_stride = input_rows.iter().map(Vec::len).max()?;
    let mut flattened_input = Vec::with_capacity(row_count.checked_mul(hidden_stride)?);
    for row in input_rows {
        let mut padded = row.get(..input_width)?.to_vec();
        padded.resize(hidden_stride, 0.0);
        flattened_input.extend_from_slice(&padded);
    }

    let output = find_optional_pipeline_handle(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
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
    record_optional_kernel_result(bringup, projection_kernel_name, output.is_some());
    output
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
    let projection_kernel_name = bringup
        .state
        .optional_kernel_dispatch_plan
        .projection_kernel_name(binding.native_dtype)?;
    if !optional_kernel_allowed(bringup, projection_kernel_name) {
        return None;
    }

    let (_, cols) = tensor_matrix_dimensions(&binding.meta.spec)?;
    if input.len() > cols {
        return None;
    }
    if output_dim == 0 {
        return Some(Vec::new());
    }

    let row_byte_offset = row_offset
        .checked_mul(cols)?
        .checked_mul(native_dtype_size_bytes(binding.native_dtype))?;

    let output = find_optional_pipeline_handle(
        &bringup.state,
        &bringup.metallib.path,
        projection_kernel_name,
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
    record_optional_kernel_result(bringup, projection_kernel_name, output.is_some());
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
        let batch_offset = token_id as u32 - workload.kv_metadata.cu_seq_lens[batch_id] as u32;
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
fn set_model_stage_rope_dispatch_params(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    query_head_count: u32,
    key_head_count: u32,
    head_dim: u32,
    rope_style: u32,
) {
    let params = ModelStageRopeDispatchParams {
        query_head_count,
        key_head_count,
        head_dim,
        rope_style,
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
) {
    let params = BatchedModelStageRopeDispatchParams {
        token_count,
        query_head_count,
        key_head_count,
        head_dim,
        rope_style,
        freq_base,
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
mod tests {
    use super::*;
    use crate::ids::{BlockId, CacheGroupId, RequestId, StepId};
    use crate::kv::BlockTableView;
    use crate::scheduler::{
        ExecutionBatch, ExecutionItem, ExecutionMode, PositionRange, RouteMetadata,
    };
    use std::ffi::OsString;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Phase1Fixture {
        root: PathBuf,
        build_dir: PathBuf,
    }

    impl Phase1Fixture {
        fn cleanup(self) {
            let _ = fs::remove_dir_all(self.root);
        }
    }

    type SimulatedNumericPath = ReferenceNumericPath;

    fn simulated_numeric_path(workload: &MetalDispatchWorkload) -> SimulatedNumericPath {
        reference_numeric_path(workload)
    }

    fn simulated_numeric_trace(reference: &SimulatedNumericPath) -> MetalDispatchNumericTrace {
        MetalDispatchNumericTrace {
            attention_output_bits: reference
                .attention_output
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            key_cache_checksum: checksum_f32_slice(&reference.key_cache),
            attention_output_checksum: checksum_f32_slice(&reference.attention_output),
            gather_output_checksum: checksum_f32_pair(
                &reference.gather_key,
                &reference.gather_value,
            ),
            copy_output_checksum: checksum_f32_pair(&reference.copy_key, &reference.copy_value),
            validation: None,
        }
    }

    #[test]
    fn pipeline_lookup_index_preserves_first_position_for_duplicate_kernel_names() {
        let lookup = pipeline_lookup_index(&[
            "rms_norm_f32".to_string(),
            "decode_logits_projection_f32".to_string(),
            "rms_norm_f32".to_string(),
        ]);

        assert_eq!(lookup.get("rms_norm_f32"), Some(&0));
        assert_eq!(lookup.get("decode_logits_projection_f32"), Some(&1));
        assert!(!lookup.contains_key("missing_kernel"));
    }

    #[test]
    fn optional_kernel_dispatch_plan_tracks_available_hot_path_kernels() {
        let lookup = pipeline_lookup_index(&[
            "decode_logits_projection_f32".to_string(),
            "decode_logits_projection_batched_f16".to_string(),
            "gather_embedding_rows_bf16".to_string(),
            "rms_norm_f32".to_string(),
            "rms_norm_batched_bf16".to_string(),
            "logits_argmax_f32".to_string(),
            "sample_argmax_logprob_f32".to_string(),
            "apply_rope_batched_f32".to_string(),
            "expand_grouped_kv_heads_f32".to_string(),
            "ffn_gate_gelu_approx_product_f32".to_string(),
        ]);

        let plan = build_optional_kernel_dispatch_plan(&lookup);

        assert_eq!(
            plan.projection_kernel_name(NativeTensorDataType::F32),
            Some("decode_logits_projection_f32")
        );
        assert_eq!(plan.projection_kernel_name(NativeTensorDataType::F16), None);
        assert_eq!(
            plan.batched_projection_kernel_name(NativeTensorDataType::F16),
            Some("decode_logits_projection_batched_f16")
        );
        assert_eq!(
            plan.embedding_gather_kernel_name(NativeTensorDataType::Bf16),
            Some("gather_embedding_rows_bf16")
        );
        assert_eq!(
            plan.rms_norm_kernel_name(NativeTensorDataType::F32),
            Some("rms_norm_f32")
        );
        assert_eq!(
            plan.batched_rms_norm_kernel_name(NativeTensorDataType::Bf16),
            Some("rms_norm_batched_bf16")
        );
        assert!(plan.logits_argmax_f32);
        assert!(!plan.logits_argmax_batched_f32);
        assert!(plan.sample_argmax_logprob_f32);
        assert!(!plan.sample_argmax_logprob_batched_f32);
        assert!(!plan.apply_rope_f32);
        assert!(plan.apply_rope_batched_f32);
        assert!(plan.expand_grouped_kv_heads_f32);
        assert_eq!(
            plan.ffn_gate_product_kernel_name(ModelFfnActivation::GeluApprox),
            Some("ffn_gate_gelu_approx_product_f32")
        );
        assert_eq!(
            plan.ffn_gate_product_kernel_name(ModelFfnActivation::Silu),
            None
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn optional_kernel_feedback_disables_kernel_after_threshold_failures() {
        let mut feedback = MetalOptionalKernelFeedbackState::default();
        let kernel_name = "decode_logits_projection_f32";

        for attempt in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
            assert!(optional_kernel_allowed_in_feedback_state(
                &feedback,
                kernel_name
            ));
            record_optional_kernel_feedback_state(&mut feedback, kernel_name, false);
            let expected_failures = attempt + 1;
            assert_eq!(
                feedback.consecutive_failures_by_kernel.get(kernel_name),
                Some(&expected_failures)
            );
        }

        assert!(!optional_kernel_allowed_in_feedback_state(
            &feedback,
            kernel_name
        ));
        assert!(feedback.disabled_kernels.contains(kernel_name));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn optional_kernel_feedback_success_resets_consecutive_failures() {
        let mut feedback = MetalOptionalKernelFeedbackState::default();
        let kernel_name = "rms_norm_f32";

        for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
            record_optional_kernel_feedback_state(&mut feedback, kernel_name, false);
        }
        assert!(!optional_kernel_allowed_in_feedback_state(
            &feedback,
            kernel_name
        ));

        record_optional_kernel_feedback_state(&mut feedback, kernel_name, true);

        assert!(optional_kernel_allowed_in_feedback_state(
            &feedback,
            kernel_name
        ));
        assert!(!feedback
            .consecutive_failures_by_kernel
            .contains_key(kernel_name));
        assert!(!feedback.disabled_kernels.contains(kernel_name));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_group_feedback_keys_disable_only_the_failing_group_shape() {
        let mut feedback = MetalOptionalKernelFeedbackState::default();
        let sampler_group_key = sampler_batched_group_feedback_key(4);
        let different_sampler_group_key = sampler_batched_group_feedback_key(2);
        let decode_dims = ModelBoundDecodeDims {
            input_width: 8,
            hidden_dim: 16,
            intermediate_dim: 32,
            vocab_rows: 64,
        };
        let decode_group_key = direct_decode_batched_group_feedback_key(4, decode_dims);

        for _ in 0..PHASE1_OPTIONAL_KERNEL_DISABLE_FAILURE_THRESHOLD {
            record_optional_kernel_feedback_state(&mut feedback, &sampler_group_key, false);
        }

        assert!(!optional_kernel_allowed_in_feedback_state(
            &feedback,
            &sampler_group_key
        ));
        assert!(optional_kernel_allowed_in_feedback_state(
            &feedback,
            &different_sampler_group_key
        ));
        assert!(optional_kernel_allowed_in_feedback_state(
            &feedback,
            &decode_group_key
        ));
    }

    #[test]
    fn grouped_sampler_request_indices_by_logits_width_preserves_request_order() {
        let requests = vec![
            SamplerRequest {
                request_id: RequestId(1),
                previous_token: 9,
                logits: Some(vec![0.1, 0.9, -0.5]),
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params: crate::sampling::SamplingParams::default(),
            },
            SamplerRequest {
                request_id: RequestId(2),
                previous_token: 19,
                logits: Some(vec![0.2, 0.3]),
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params: crate::sampling::SamplingParams::default(),
            },
            SamplerRequest {
                request_id: RequestId(3),
                previous_token: 29,
                logits: None,
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params: crate::sampling::SamplingParams::default(),
            },
            SamplerRequest {
                request_id: RequestId(4),
                previous_token: 39,
                logits: Some(vec![1.2, 0.3, -0.2]),
                generated_len: 0,
                max_output_tokens: 4,
                sampling_params: crate::sampling::SamplingParams::default(),
            },
        ];

        let grouped = grouped_sampler_request_indices_by_logits_width(&requests)
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(grouped, vec![(2, vec![1]), (3, vec![0, 3])]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn grouped_sampler_results_recursively_preserve_batched_subgroups() {
        let indices = vec![0_usize, 2, 4, 6];
        let mut attempted_group_sizes = Vec::new();

        let results = collect_grouped_sampler_results_with_item_fallback(
            &indices,
            &mut |group_indices| {
                attempted_group_sizes.push(group_indices.len());
                if group_indices.len() > 2 {
                    return None;
                }
                Some(
                    group_indices
                        .iter()
                        .map(|request_index| {
                            (
                                u32::try_from(*request_index).unwrap_or(u32::MAX),
                                -(*request_index as f32),
                            )
                        })
                        .collect(),
                )
            },
            &mut |request_index| {
                Some((
                    u32::try_from(request_index).unwrap_or(u32::MAX),
                    request_index as f32,
                ))
            },
        )
        .expect("recursive sampler subgroup fallback should succeed");

        assert_eq!(attempted_group_sizes, vec![4, 2, 2]);
        assert_eq!(
            results,
            vec![
                (0, (0, -0.0)),
                (2, (2, -2.0)),
                (4, (4, -4.0)),
                (6, (6, -6.0)),
            ]
        );
    }

    #[allow(dead_code)]
    fn write_valid_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-fixture");
        fs::create_dir_all(&root_dir).expect("native model fixture directory should create");
        fs::write(root_dir.join("model.safetensors"), vec![0_u8; 4096])
            .expect("native model weights should write");

        // Dimensions are deliberately tiny so that every tensor fits within the
        // 32-byte (= 16 × f16) limit imposed by `native_model_tensor`.
        //
        // hidden_size=2, vocab=4, q_heads=1, kv_heads=1, head_dim=2 gives:
        //   embed_tokens  [4, 2]  =  8 f16 = 16 B
        //   qkv_proj      [6, 2]  = 12 f16 = 24 B  (packed: (q+2k) heads × head_dim rows)
        //   gate_up_proj  [4, 2]  =  8 f16 = 16 B  (intermediate_size=2, gate+up packed)
        //   all 1-D norms  [2]    =  2 f16 =  4 B
        //   all 2-D mats  [2, 2] =  4 f16 =  8 B
        // Token IDs 1–4 map to rows 1, 2, 3, 0 (mod 4) — all within the 4-row embedding.
        let manifest = crate::model::NativeModelManifest {
            schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            hidden_size: 2,
            attention_head_count: 1,
            attention_head_dim: 2,
            kv_head_count: 1,
            vocab_size: 4,
            tie_word_embeddings: false,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            tensors: vec![
                native_model_tensor(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    vec![4, 2],
                ),
                native_model_tensor(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    vec![2],
                ),
                native_model_tensor("lm_head.weight", NativeTensorRole::LmHead, None, vec![4, 2]),
                native_model_tensor(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![2],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.qkv_proj.weight",
                    NativeTensorRole::AttentionQkvPacked,
                    Some(0),
                    // (q_heads + 2 * kv_heads) * head_dim = (1 + 2) * 2 = 6 rows
                    vec![6, 2],
                ),
                native_model_tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    vec![2, 2],
                ),
                native_model_tensor(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![2],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(0),
                    // 2 * intermediate_size = 2 * 2 = 4 rows
                    vec![4, 2],
                ),
                native_model_tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    vec![2, 2],
                ),
            ],
        };

        fs::write(
            root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("native model manifest should serialize"),
        )
        .expect("native model manifest should write");

        root_dir
    }

    #[allow(dead_code)]
    fn native_model_tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> crate::model::NativeTensorSpec {
        crate::model::NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: crate::model::NativeTensorDataType::F16,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    fn native_model_tensor_with_file(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: &[u64],
        file: &str,
        length_bytes: u64,
    ) -> crate::model::NativeTensorSpec {
        crate::model::NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: crate::model::NativeTensorDataType::F32,
            shape: shape.to_vec(),
            file: PathBuf::from(file),
            offset_bytes: 0,
            length_bytes,
        }
    }

    fn write_f32_tensor_file(root_dir: &Path, file_name: &str, values: &[f32]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(root_dir.join(file_name), bytes).expect("tensor bytes should write");
    }

    #[cfg(target_os = "macos")]
    fn write_projection_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-projection");
        fs::create_dir_all(&root_dir).expect("projection fixture directory should create");

        let embedding = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ];
        let ones = vec![1.0_f32; 8];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let double_identity = identity.iter().map(|value| value * 2.0).collect::<Vec<_>>();
        let triple_identity = identity.iter().map(|value| value * 3.0).collect::<Vec<_>>();
        let zero_matrix = vec![0.0_f32; 64];

        write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
        write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "lm_head.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "attn_q.bin", &identity);
        write_f32_tensor_file(&root_dir, "attn_k.bin", &double_identity);
        write_f32_tensor_file(&root_dir, "attn_v.bin", &triple_identity);
        write_f32_tensor_file(&root_dir, "attn_o.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_matrix);

        let matrix_bytes = (64 * size_of::<f32>()) as u64;
        let vector_bytes = (8 * size_of::<f32>()) as u64;
        let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
        let manifest = crate::model::NativeModelManifest {
            schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            hidden_size: 8,
            attention_head_count: 2,
            attention_head_dim: 4,
            kv_head_count: 2,
            vocab_size: 5,
            tie_word_embeddings: false,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            tensors: vec![
                native_model_tensor_with_file(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    &[5, 8],
                    "embed.bin",
                    embedding_bytes,
                ),
                native_model_tensor_with_file(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    &[8],
                    "final_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    &[5, 8],
                    "lm_head.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    &[8],
                    "attn_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.q_proj.weight",
                    NativeTensorRole::AttentionQ,
                    Some(0),
                    &[8, 8],
                    "attn_q.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.k_proj.weight",
                    NativeTensorRole::AttentionK,
                    Some(0),
                    &[8, 8],
                    "attn_k.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.v_proj.weight",
                    NativeTensorRole::AttentionV,
                    Some(0),
                    &[8, 8],
                    "attn_v.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    &[8, 8],
                    "attn_o.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    &[8],
                    "ffn_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.gate_proj.weight",
                    NativeTensorRole::FfnGate,
                    Some(0),
                    &[8, 8],
                    "ffn_gate.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.up_proj.weight",
                    NativeTensorRole::FfnUp,
                    Some(0),
                    &[8, 8],
                    "ffn_up.bin",
                    matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    &[8, 8],
                    "ffn_down.bin",
                    matrix_bytes,
                ),
            ],
        };

        fs::write(
            root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
        )
        .expect("projection manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_projection_qk_norm_native_model_fixture() -> PathBuf {
        let root_dir = write_projection_native_model_fixture();
        let q_norm = vec![2.0_f32, 1.0, 1.0, 1.0];
        let k_norm = vec![3.0_f32, 1.0, 1.0, 1.0];
        let norm_bytes = (q_norm.len() * size_of::<f32>()) as u64;
        write_f32_tensor_file(&root_dir, "attn_q_norm.bin", &q_norm);
        write_f32_tensor_file(&root_dir, "attn_k_norm.bin", &k_norm);

        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("projection manifest should parse");
        manifest.tensors.push(native_model_tensor_with_file(
            "model.layers.0.self_attn.q_norm.weight",
            NativeTensorRole::AttentionQNorm,
            Some(0),
            &[4],
            "attn_q_norm.bin",
            norm_bytes,
        ));
        manifest.tensors.push(native_model_tensor_with_file(
            "model.layers.0.self_attn.k_norm.weight",
            NativeTensorRole::AttentionKNorm,
            Some(0),
            &[4],
            "attn_k_norm.bin",
            norm_bytes,
        ));
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
        )
        .expect("projection manifest should rewrite");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_projection_custom_rope_native_model_fixture(rope_theta: u32) -> PathBuf {
        let root_dir = write_projection_native_model_fixture();
        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("projection manifest should parse");
        manifest.rope_theta = Some(rope_theta);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
        )
        .expect("projection manifest should rewrite");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_gemma_projection_native_model_fixture() -> PathBuf {
        let root_dir = write_projection_native_model_fixture();
        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("projection manifest should parse");
        manifest.model_family = "gemma2_dense".to_string();
        fs::write(
            root_dir.join("attn_norm.bin"),
            vec![0_u8; 8 * size_of::<f32>()],
        )
        .expect("gemma attention norm should write");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
        )
        .expect("projection manifest should rewrite");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_gemma_projection_custom_rope_native_model_fixture(rope_theta: u32) -> PathBuf {
        let root_dir = write_gemma_projection_native_model_fixture();
        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes =
            fs::read(&manifest_path).expect("gemma projection manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("gemma projection manifest should parse");
        manifest.rope_theta = Some(rope_theta);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest)
                .expect("gemma projection manifest should serialize"),
        )
        .expect("gemma projection manifest should rewrite");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_gemma_projection_attention_config_native_model_fixture(
        query_pre_attn_scalar: u32,
        attention_logit_softcap: u32,
    ) -> PathBuf {
        let root_dir = write_gemma_projection_native_model_fixture();
        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes =
            fs::read(&manifest_path).expect("gemma projection manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("gemma projection manifest should parse");
        manifest.query_pre_attn_scalar = Some(query_pre_attn_scalar);
        manifest.attention_logit_softcap = Some(attention_logit_softcap);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest)
                .expect("gemma projection manifest should serialize"),
        )
        .expect("gemma projection manifest should rewrite");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_grouped_projection_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-grouped-projection");
        fs::create_dir_all(&root_dir).expect("grouped projection fixture directory should create");

        let embedding = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ];
        let ones = vec![1.0_f32; 8];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let grouped_k = [
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let grouped_v = [
            3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let zero_matrix = vec![0.0_f32; 64];

        write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
        write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "lm_head.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "attn_q.bin", &identity);
        write_f32_tensor_file(&root_dir, "attn_k.bin", &grouped_k);
        write_f32_tensor_file(&root_dir, "attn_v.bin", &grouped_v);
        write_f32_tensor_file(&root_dir, "attn_o.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_matrix);

        let full_matrix_bytes = (64 * size_of::<f32>()) as u64;
        let grouped_matrix_bytes = (32 * size_of::<f32>()) as u64;
        let vector_bytes = (8 * size_of::<f32>()) as u64;
        let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
        let manifest = crate::model::NativeModelManifest {
            schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            hidden_size: 8,
            attention_head_count: 4,
            attention_head_dim: 2,
            kv_head_count: 2,
            vocab_size: 5,
            tie_word_embeddings: false,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            tensors: vec![
                native_model_tensor_with_file(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    &[5, 8],
                    "embed.bin",
                    embedding_bytes,
                ),
                native_model_tensor_with_file(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    &[8],
                    "final_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    &[5, 8],
                    "lm_head.bin",
                    full_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    &[8],
                    "attn_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.q_proj.weight",
                    NativeTensorRole::AttentionQ,
                    Some(0),
                    &[8, 8],
                    "attn_q.bin",
                    full_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.k_proj.weight",
                    NativeTensorRole::AttentionK,
                    Some(0),
                    &[4, 8],
                    "attn_k.bin",
                    grouped_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.v_proj.weight",
                    NativeTensorRole::AttentionV,
                    Some(0),
                    &[4, 8],
                    "attn_v.bin",
                    grouped_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    &[8, 8],
                    "attn_o.bin",
                    full_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    &[8],
                    "ffn_norm.bin",
                    vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.gate_proj.weight",
                    NativeTensorRole::FfnGate,
                    Some(0),
                    &[8, 8],
                    "ffn_gate.bin",
                    full_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.up_proj.weight",
                    NativeTensorRole::FfnUp,
                    Some(0),
                    &[8, 8],
                    "ffn_up.bin",
                    full_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    &[8, 8],
                    "ffn_down.bin",
                    full_matrix_bytes,
                ),
            ],
        };

        fs::write(
            root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)
                .expect("grouped projection manifest should serialize"),
        )
        .expect("grouped projection manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_multilayer_projection_native_model_fixture() -> PathBuf {
        let root_dir = write_projection_native_model_fixture();
        let zero_matrix = vec![0.0_f32; 64];
        write_f32_tensor_file(&root_dir, "attn_q_l1.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "attn_k_l1.bin", &zero_matrix);
        write_f32_tensor_file(&root_dir, "attn_v_l1.bin", &zero_matrix);

        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("projection manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("projection manifest should parse");
        let vector_bytes = (8 * size_of::<f32>()) as u64;
        let matrix_bytes = (64 * size_of::<f32>()) as u64;
        manifest.layer_count = 2;
        manifest.tensors.extend([
            native_model_tensor_with_file(
                "model.layers.1.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(1),
                &[8],
                "attn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(1),
                &[8, 8],
                "attn_q_l1.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(1),
                &[8, 8],
                "attn_k_l1.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(1),
                &[8, 8],
                "attn_v_l1.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(1),
                &[8, 8],
                "attn_o.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.post_attention_layernorm.weight",
                NativeTensorRole::FfnNorm,
                Some(1),
                &[8],
                "ffn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.mlp.gate_proj.weight",
                NativeTensorRole::FfnGate,
                Some(1),
                &[8, 8],
                "ffn_gate.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.mlp.up_proj.weight",
                NativeTensorRole::FfnUp,
                Some(1),
                &[8, 8],
                "ffn_up.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.1.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(1),
                &[8, 8],
                "ffn_down.bin",
                matrix_bytes,
            ),
        ]);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("projection manifest should serialize"),
        )
        .expect("projection manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn tail_projector_matrix(
        output_rows: usize,
        hidden_size: usize,
        tail_size: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut matrix = vec![0.0_f32; output_rows * hidden_size];
        let tail_start = hidden_size.saturating_sub(tail_size);
        for row in 0..output_rows.min(tail_size) {
            matrix[row * hidden_size + tail_start + row] = scale;
        }
        matrix
    }

    #[cfg(target_os = "macos")]
    fn write_wide_projection_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-wide-projection");
        fs::create_dir_all(&root_dir).expect("wide projection fixture directory should create");

        let hidden_size = 40_usize;
        let head_width = 8_usize;
        let vocab_size = 5_usize;
        let tail_size = 8_usize;
        let mut embedding = vec![0.0_f32; vocab_size * hidden_size];
        for token in 1..vocab_size {
            let base = token * hidden_size + (hidden_size - tail_size);
            for lane in 0..tail_size {
                embedding[base + lane] = token as f32 + lane as f32;
            }
        }
        let ones = vec![1.0_f32; hidden_size];
        let q_proj = tail_projector_matrix(head_width, hidden_size, tail_size, 1.0);
        let k_proj = tail_projector_matrix(head_width, hidden_size, tail_size, 2.0);
        let v_proj = tail_projector_matrix(head_width, hidden_size, tail_size, 3.0);
        let attention_o = vec![0.0_f32; hidden_size * head_width];
        let zero_square = vec![0.0_f32; hidden_size * hidden_size];
        let zero_lm_head = vec![0.0_f32; vocab_size * hidden_size];

        write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
        write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "lm_head.bin", &zero_lm_head);
        write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "attn_q.bin", &q_proj);
        write_f32_tensor_file(&root_dir, "attn_k.bin", &k_proj);
        write_f32_tensor_file(&root_dir, "attn_v.bin", &v_proj);
        write_f32_tensor_file(&root_dir, "attn_o.bin", &attention_o);
        write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_square);
        write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_square);
        write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_square);

        let hidden_vector_bytes = (hidden_size * size_of::<f32>()) as u64;
        let qkv_matrix_bytes = (head_width * hidden_size * size_of::<f32>()) as u64;
        let attention_o_bytes = (hidden_size * head_width * size_of::<f32>()) as u64;
        let square_matrix_bytes = (hidden_size * hidden_size * size_of::<f32>()) as u64;
        let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
        let lm_head_bytes = (zero_lm_head.len() * size_of::<f32>()) as u64;

        let manifest = crate::model::NativeModelManifest {
            schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            hidden_size: hidden_size as u32,
            attention_head_count: 2,
            attention_head_dim: 4,
            kv_head_count: 2,
            vocab_size: vocab_size as u32,
            tie_word_embeddings: false,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            tensors: vec![
                native_model_tensor_with_file(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    &[vocab_size as u64, hidden_size as u64],
                    "embed.bin",
                    embedding_bytes,
                ),
                native_model_tensor_with_file(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    &[hidden_size as u64],
                    "final_norm.bin",
                    hidden_vector_bytes,
                ),
                native_model_tensor_with_file(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    &[vocab_size as u64, hidden_size as u64],
                    "lm_head.bin",
                    lm_head_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    &[hidden_size as u64],
                    "attn_norm.bin",
                    hidden_vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.q_proj.weight",
                    NativeTensorRole::AttentionQ,
                    Some(0),
                    &[head_width as u64, hidden_size as u64],
                    "attn_q.bin",
                    qkv_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.k_proj.weight",
                    NativeTensorRole::AttentionK,
                    Some(0),
                    &[head_width as u64, hidden_size as u64],
                    "attn_k.bin",
                    qkv_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.v_proj.weight",
                    NativeTensorRole::AttentionV,
                    Some(0),
                    &[head_width as u64, hidden_size as u64],
                    "attn_v.bin",
                    qkv_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    &[hidden_size as u64, head_width as u64],
                    "attn_o.bin",
                    attention_o_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    &[hidden_size as u64],
                    "ffn_norm.bin",
                    hidden_vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.gate_proj.weight",
                    NativeTensorRole::FfnGate,
                    Some(0),
                    &[hidden_size as u64, hidden_size as u64],
                    "ffn_gate.bin",
                    square_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.up_proj.weight",
                    NativeTensorRole::FfnUp,
                    Some(0),
                    &[hidden_size as u64, hidden_size as u64],
                    "ffn_up.bin",
                    square_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    &[hidden_size as u64, hidden_size as u64],
                    "ffn_down.bin",
                    square_matrix_bytes,
                ),
            ],
        };

        fs::write(
            root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)
                .expect("wide projection manifest should serialize"),
        )
        .expect("wide projection manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_wide_direct_decode_native_model_fixture() -> PathBuf {
        let root_dir = unique_test_dir("native-model-wide-direct-decode");
        fs::create_dir_all(&root_dir).expect("wide direct decode fixture directory should create");

        let hidden_size = 40_usize;
        let head_width = 8_usize;
        let vocab_size = 5_usize;
        let tail_size = 8_usize;
        let mut embedding = vec![0.0_f32; vocab_size * hidden_size];
        let token_four_base = 4 * hidden_size + (hidden_size - tail_size);
        for lane in 0..tail_size {
            embedding[token_four_base + lane] = (lane + 1) as f32;
        }
        let ones = vec![1.0_f32; hidden_size];
        let zero_qkv = vec![0.0_f32; head_width * hidden_size];
        let zero_attention_o = vec![0.0_f32; hidden_size * head_width];
        let zero_square = vec![0.0_f32; hidden_size * hidden_size];
        let mut lm_head = vec![0.0_f32; vocab_size * hidden_size];
        let lm_head_token_two_base = 2 * hidden_size + (hidden_size - tail_size);
        for lane in 0..tail_size {
            lm_head[lm_head_token_two_base + lane] = 1.0;
        }

        write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
        write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "lm_head.bin", &lm_head);
        write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "attn_q.bin", &zero_qkv);
        write_f32_tensor_file(&root_dir, "attn_k.bin", &zero_qkv);
        write_f32_tensor_file(&root_dir, "attn_v.bin", &zero_qkv);
        write_f32_tensor_file(&root_dir, "attn_o.bin", &zero_attention_o);
        write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "ffn_gate.bin", &zero_square);
        write_f32_tensor_file(&root_dir, "ffn_up.bin", &zero_square);
        write_f32_tensor_file(&root_dir, "ffn_down.bin", &zero_square);

        let hidden_vector_bytes = (hidden_size * size_of::<f32>()) as u64;
        let qkv_matrix_bytes = (head_width * hidden_size * size_of::<f32>()) as u64;
        let attention_o_bytes = (hidden_size * head_width * size_of::<f32>()) as u64;
        let square_matrix_bytes = (hidden_size * hidden_size * size_of::<f32>()) as u64;
        let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
        let lm_head_bytes = (lm_head.len() * size_of::<f32>()) as u64;

        let manifest = crate::model::NativeModelManifest {
            schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            hidden_size: hidden_size as u32,
            attention_head_count: 2,
            attention_head_dim: 4,
            kv_head_count: 2,
            vocab_size: vocab_size as u32,
            tie_word_embeddings: false,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            tensors: vec![
                native_model_tensor_with_file(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    &[vocab_size as u64, hidden_size as u64],
                    "embed.bin",
                    embedding_bytes,
                ),
                native_model_tensor_with_file(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    &[hidden_size as u64],
                    "final_norm.bin",
                    hidden_vector_bytes,
                ),
                native_model_tensor_with_file(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    &[vocab_size as u64, hidden_size as u64],
                    "lm_head.bin",
                    lm_head_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    &[hidden_size as u64],
                    "attn_norm.bin",
                    hidden_vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.q_proj.weight",
                    NativeTensorRole::AttentionQ,
                    Some(0),
                    &[head_width as u64, hidden_size as u64],
                    "attn_q.bin",
                    qkv_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.k_proj.weight",
                    NativeTensorRole::AttentionK,
                    Some(0),
                    &[head_width as u64, hidden_size as u64],
                    "attn_k.bin",
                    qkv_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.v_proj.weight",
                    NativeTensorRole::AttentionV,
                    Some(0),
                    &[head_width as u64, hidden_size as u64],
                    "attn_v.bin",
                    qkv_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    &[hidden_size as u64, head_width as u64],
                    "attn_o.bin",
                    attention_o_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.post_attention_layernorm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    &[hidden_size as u64],
                    "ffn_norm.bin",
                    hidden_vector_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.gate_proj.weight",
                    NativeTensorRole::FfnGate,
                    Some(0),
                    &[hidden_size as u64, hidden_size as u64],
                    "ffn_gate.bin",
                    square_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.up_proj.weight",
                    NativeTensorRole::FfnUp,
                    Some(0),
                    &[hidden_size as u64, hidden_size as u64],
                    "ffn_up.bin",
                    square_matrix_bytes,
                ),
                native_model_tensor_with_file(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    &[hidden_size as u64, hidden_size as u64],
                    "ffn_down.bin",
                    square_matrix_bytes,
                ),
            ],
        };

        fs::write(
            root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)
                .expect("wide direct decode manifest should serialize"),
        )
        .expect("wide direct decode manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum DirectDecodeFixtureGateUpLayout {
        Split,
        Packed,
    }

    #[cfg(target_os = "macos")]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum DirectDecodeFixtureVariant {
        ProjectionOnly,
        FfnContinuation,
    }

    #[cfg(target_os = "macos")]
    fn write_direct_decode_native_model_fixture_with_variant(
        tie_word_embeddings: bool,
        gate_up_layout: DirectDecodeFixtureGateUpLayout,
        variant: DirectDecodeFixtureVariant,
    ) -> PathBuf {
        let root_dir = unique_test_dir("native-model-direct-decode");
        fs::create_dir_all(&root_dir).expect("decode fixture directory should create");

        let embedding = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        ];
        let ones = vec![1.0_f32; 8];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let double_identity = identity.iter().map(|value| value * 2.0).collect::<Vec<_>>();
        let triple_identity = identity.iter().map(|value| value * 3.0).collect::<Vec<_>>();
        let zero_matrix = vec![0.0_f32; 64];
        let projection_lm_head = embedding.clone();
        let continuation_lm_head = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
        ];
        let (ffn_gate, ffn_up, ffn_down, lm_head) = match variant {
            DirectDecodeFixtureVariant::ProjectionOnly => (
                zero_matrix.clone(),
                zero_matrix.clone(),
                zero_matrix.clone(),
                projection_lm_head,
            ),
            DirectDecodeFixtureVariant::FfnContinuation => {
                let mut gate = zero_matrix.clone();
                let mut up = zero_matrix.clone();
                let mut down = zero_matrix.clone();
                gate[0] = 1.0;
                up[0] = 0.5;
                down[2 * 8] = 3.0;
                (gate, up, down, continuation_lm_head)
            }
        };
        let packed_gate_up = ffn_gate
            .iter()
            .chain(ffn_up.iter())
            .copied()
            .collect::<Vec<_>>();

        write_f32_tensor_file(&root_dir, "embed.bin", &embedding);
        write_f32_tensor_file(&root_dir, "final_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "attn_norm.bin", &ones);
        write_f32_tensor_file(&root_dir, "attn_q.bin", &identity);
        write_f32_tensor_file(&root_dir, "attn_k.bin", &double_identity);
        write_f32_tensor_file(&root_dir, "attn_v.bin", &triple_identity);
        write_f32_tensor_file(&root_dir, "attn_o.bin", &identity);
        write_f32_tensor_file(&root_dir, "ffn_norm.bin", &ones);
        match gate_up_layout {
            DirectDecodeFixtureGateUpLayout::Split => {
                write_f32_tensor_file(&root_dir, "ffn_gate.bin", &ffn_gate);
                write_f32_tensor_file(&root_dir, "ffn_up.bin", &ffn_up);
            }
            DirectDecodeFixtureGateUpLayout::Packed => {
                write_f32_tensor_file(&root_dir, "ffn_gate_up.bin", &packed_gate_up);
            }
        }
        write_f32_tensor_file(&root_dir, "ffn_down.bin", &ffn_down);
        if !tie_word_embeddings {
            write_f32_tensor_file(&root_dir, "lm_head.bin", &lm_head);
        }

        let matrix_bytes = (64 * size_of::<f32>()) as u64;
        let packed_matrix_bytes = (128 * size_of::<f32>()) as u64;
        let vector_bytes = (8 * size_of::<f32>()) as u64;
        let embedding_bytes = (embedding.len() * size_of::<f32>()) as u64;
        let mut tensors = vec![
            native_model_tensor_with_file(
                "model.embed_tokens.weight",
                NativeTensorRole::TokenEmbedding,
                None,
                &[5, 8],
                "embed.bin",
                embedding_bytes,
            ),
            native_model_tensor_with_file(
                "model.norm.weight",
                NativeTensorRole::FinalNorm,
                None,
                &[8],
                "final_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.input_layernorm.weight",
                NativeTensorRole::AttentionNorm,
                Some(0),
                &[8],
                "attn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.q_proj.weight",
                NativeTensorRole::AttentionQ,
                Some(0),
                &[8, 8],
                "attn_q.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.k_proj.weight",
                NativeTensorRole::AttentionK,
                Some(0),
                &[8, 8],
                "attn_k.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.v_proj.weight",
                NativeTensorRole::AttentionV,
                Some(0),
                &[8, 8],
                "attn_v.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.self_attn.o_proj.weight",
                NativeTensorRole::AttentionO,
                Some(0),
                &[8, 8],
                "attn_o.bin",
                matrix_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.post_attention_layernorm.weight",
                NativeTensorRole::FfnNorm,
                Some(0),
                &[8],
                "ffn_norm.bin",
                vector_bytes,
            ),
            native_model_tensor_with_file(
                "model.layers.0.mlp.down_proj.weight",
                NativeTensorRole::FfnDown,
                Some(0),
                &[8, 8],
                "ffn_down.bin",
                matrix_bytes,
            ),
        ];
        match gate_up_layout {
            DirectDecodeFixtureGateUpLayout::Split => {
                tensors.push(native_model_tensor_with_file(
                    "model.layers.0.mlp.gate_proj.weight",
                    NativeTensorRole::FfnGate,
                    Some(0),
                    &[8, 8],
                    "ffn_gate.bin",
                    matrix_bytes,
                ));
                tensors.push(native_model_tensor_with_file(
                    "model.layers.0.mlp.up_proj.weight",
                    NativeTensorRole::FfnUp,
                    Some(0),
                    &[8, 8],
                    "ffn_up.bin",
                    matrix_bytes,
                ));
            }
            DirectDecodeFixtureGateUpLayout::Packed => {
                tensors.push(native_model_tensor_with_file(
                    "model.layers.0.mlp.gate_up_proj.weight",
                    NativeTensorRole::FfnGateUpPacked,
                    Some(0),
                    &[16, 8],
                    "ffn_gate_up.bin",
                    packed_matrix_bytes,
                ));
            }
        }
        if !tie_word_embeddings {
            tensors.insert(
                2,
                native_model_tensor_with_file(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    &[5, 8],
                    "lm_head.bin",
                    embedding_bytes,
                ),
            );
        }

        let manifest = crate::model::NativeModelManifest {
            schema_version: crate::model::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            hidden_size: 8,
            attention_head_count: 2,
            attention_head_dim: 4,
            kv_head_count: 2,
            vocab_size: 5,
            tie_word_embeddings,
            rope_theta: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            tensors,
        };

        fs::write(
            root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("decode manifest should serialize"),
        )
        .expect("decode manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_direct_decode_native_model_fixture(tie_word_embeddings: bool) -> PathBuf {
        write_direct_decode_native_model_fixture_with_variant(
            tie_word_embeddings,
            DirectDecodeFixtureGateUpLayout::Split,
            DirectDecodeFixtureVariant::ProjectionOnly,
        )
    }

    #[cfg(target_os = "macos")]
    fn write_gemma_direct_decode_native_model_fixture(tie_word_embeddings: bool) -> PathBuf {
        let root_dir = write_direct_decode_native_model_fixture(tie_word_embeddings);
        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("decode manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("decode manifest should parse");
        manifest.model_family = "gemma2_dense".to_string();
        fs::write(
            root_dir.join("final_norm.bin"),
            vec![0_u8; 8 * size_of::<f32>()],
        )
        .expect("gemma final norm should write");
        fs::write(
            root_dir.join("ffn_norm.bin"),
            vec![0_u8; 8 * size_of::<f32>()],
        )
        .expect("gemma ffn norm should write");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("decode manifest should serialize"),
        )
        .expect("decode manifest should rewrite");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn write_ffn_decode_native_model_fixture(
        gate_up_layout: DirectDecodeFixtureGateUpLayout,
    ) -> PathBuf {
        write_direct_decode_native_model_fixture_with_variant(
            false,
            gate_up_layout,
            DirectDecodeFixtureVariant::FfnContinuation,
        )
    }

    #[cfg(target_os = "macos")]
    fn write_multilayer_direct_decode_native_model_fixture() -> PathBuf {
        let root_dir = write_direct_decode_native_model_fixture(false);
        let manifest_path = root_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("manifest should parse");
        let layer_zero_tensors = manifest
            .tensors
            .iter()
            .filter(|tensor| tensor.layer_index == Some(0))
            .cloned()
            .collect::<Vec<_>>();

        for mut tensor in layer_zero_tensors {
            tensor.layer_index = Some(1);
            tensor.name = tensor.name.replace("layers.0", "layers.1");
            manifest.tensors.push(tensor);
        }
        manifest.layer_count = 2;

        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");

        root_dir
    }

    #[cfg(target_os = "macos")]
    fn assert_f32_slice_close(actual: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= epsilon,
                "index {index}: actual={actual} expected={expected} epsilon={epsilon}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn neox_rotate_reference(values: &[f32], position: f32) -> Vec<f32> {
        neox_rotate_reference_with_base(values, position, PHASE1_MODEL_STAGE_ROPE_FREQ_BASE)
    }

    #[cfg(target_os = "macos")]
    fn neox_rotate_reference_with_base(values: &[f32], position: f32, freq_base: f32) -> Vec<f32> {
        let half = values.len() / 2;
        let neg_log_base = -(freq_base.ln());
        let dim_inv = 2.0 / values.len() as f32;
        let mut rotated = values.to_vec();

        for index in 0..half {
            let freq = (neg_log_base * index as f32 * dim_inv).exp();
            let theta = position * freq;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let low = values[index];
            let high = values[index + half];
            rotated[index] = low * cos_theta - high * sin_theta;
            rotated[index + half] = low * sin_theta + high * cos_theta;
        }

        rotated
    }

    #[cfg(target_os = "macos")]
    fn interleaved_rotate_reference(values: &[f32], position: f32) -> Vec<f32> {
        interleaved_rotate_reference_with_base(values, position, PHASE1_MODEL_STAGE_ROPE_FREQ_BASE)
    }

    #[cfg(target_os = "macos")]
    fn interleaved_rotate_reference_with_base(
        values: &[f32],
        position: f32,
        freq_base: f32,
    ) -> Vec<f32> {
        let half = values.len() / 2;
        let neg_log_base = -(freq_base.ln());
        let dim_inv = 2.0 / values.len() as f32;
        let mut rotated = values.to_vec();

        for index in 0..half {
            let freq = (neg_log_base * index as f32 * dim_inv).exp();
            let theta = position * freq;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let pair_base = index * 2;
            let even = values[pair_base];
            let odd = values[pair_base + 1];
            rotated[pair_base] = even * cos_theta - odd * sin_theta;
            rotated[pair_base + 1] = odd * cos_theta + even * sin_theta;
        }

        rotated
    }

    #[cfg(target_os = "macos")]
    fn rms_normalize_reference(values: &[f32], epsilon: f32) -> Vec<f32> {
        let mean_square =
            values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
        let denom = (mean_square + epsilon).sqrt();
        values.iter().map(|value| value / denom).collect()
    }

    #[cfg(target_os = "macos")]
    fn scale_reference(values: &[f32], factor: f32) -> Vec<f32> {
        values.iter().map(|value| value * factor).collect()
    }

    #[cfg(target_os = "macos")]
    fn manual_attention_head_output(
        workload: &MetalDispatchWorkload,
        staged_inputs: &MetalDispatchStagedInputs,
        token_index: usize,
        head_index: usize,
        attention_config: ReferenceAttentionConfig,
    ) -> Vec<f32> {
        let reference = reference_numeric_path_with_inputs(workload, staged_inputs);
        let head_size = workload.numeric_layout.head_size() as usize;
        let head_dim = workload.numeric_layout.head_dim as usize;
        let batch_id = batch_id_for_token(
            &workload.kv_metadata.scheduled_cu_seq_lens,
            token_index as u32,
        );
        let context_begin = workload.kv_metadata.cu_seq_lens[batch_id] as usize;
        let context_end = workload.kv_metadata.cu_seq_lens[batch_id + 1] as usize;
        let query_base = token_index * head_size + head_index * head_dim;
        let query = &staged_inputs.query[query_base..query_base + head_dim];

        let mut scores = Vec::new();
        for context_index in context_begin..context_end {
            let context_base = context_index * head_size + head_index * head_dim;
            let key = &reference.gather_key[context_base..context_base + head_dim];
            scores.push(configured_attention_score(query, key, attention_config));
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut accum = vec![0.0_f32; head_dim];
        let mut weight_sum = 0.0_f32;
        for (relative_index, context_index) in (context_begin..context_end).enumerate() {
            let weight = (scores[relative_index] - max_score).exp();
            let context_base = context_index * head_size + head_index * head_dim;
            let value = &reference.gather_value[context_base..context_base + head_dim];
            weight_sum += weight;
            for lane in 0..head_dim {
                accum[lane] += weight * value[lane];
            }
        }
        for lane in &mut accum {
            *lane /= weight_sum.max(0.000001);
        }

        accum
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_gate_up_product_uses_family_specific_activation() {
        let qwen_dir = write_projection_native_model_fixture();
        let qwen_artifacts =
            NativeModelArtifacts::from_dir(&qwen_dir).expect("qwen artifacts should load");
        let gemma_dir = write_gemma_projection_native_model_fixture();
        let gemma_artifacts =
            NativeModelArtifacts::from_dir(&gemma_dir).expect("gemma artifacts should load");

        let gate = vec![-1.0_f32, 0.5, 1.0];
        let up = vec![2.0_f32, 3.0, 4.0];
        let mut qwen_gate = gate.clone();
        let mut gemma_gate = gate.clone();

        apply_model_gate_up_product(&qwen_artifacts, &mut qwen_gate, &up);
        apply_model_gate_up_product(&gemma_artifacts, &mut gemma_gate, &up);

        let expected_qwen = gate
            .iter()
            .zip(&up)
            .map(|(gate_value, up_value)| silu(*gate_value) * *up_value)
            .collect::<Vec<_>>();
        let expected_gemma = gate
            .iter()
            .zip(&up)
            .map(|(gate_value, up_value)| gelu_approx(*gate_value) * *up_value)
            .collect::<Vec<_>>();

        assert_f32_slice_close(&qwen_gate, &expected_qwen, 1e-6);
        assert_f32_slice_close(&gemma_gate, &expected_gemma, 1e-6);
        assert_ne!(qwen_gate, gemma_gate);

        let _ = fs::remove_dir_all(qwen_dir);
        let _ = fs::remove_dir_all(gemma_dir);
    }

    #[cfg(target_os = "macos")]
    fn per_head_rms_norm_reference(values: &[f32], head_dim: usize, weights: &[f32]) -> Vec<f32> {
        values
            .chunks_exact(head_dim)
            .flat_map(|head| {
                rms_normalize_reference(head, 1e-6)
                    .into_iter()
                    .zip(weights.iter().copied())
                    .map(|(value, weight)| value * weight)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_dense_shadow_bytes_promote_byte_tensors_to_f32() {
        let u8_spec = NativeTensorSpec {
            name: "u8_weight".to_string(),
            role: NativeTensorRole::AttentionO,
            layer_index: Some(0),
            dtype: NativeTensorDataType::U8,
            shape: vec![2, 2],
            file: PathBuf::from("weights.bin"),
            offset_bytes: 0,
            length_bytes: 4,
        };
        let (u8_dtype, u8_shadow) = native_dense_shadow_bytes(&u8_spec, &[1, 2, 3, 4])
            .expect("u8 weights should promote into f32 shadow bytes");

        assert_eq!(u8_dtype, NativeTensorDataType::F32);
        assert_eq!(
            u8_shadow
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_le_bytes(
                    chunk.try_into().expect("chunk width should match")
                ))
                .collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );

        let i8_spec = NativeTensorSpec {
            name: "i8_weight".to_string(),
            role: NativeTensorRole::FfnNorm,
            layer_index: Some(0),
            dtype: NativeTensorDataType::I8,
            shape: vec![4],
            file: PathBuf::from("weights.bin"),
            offset_bytes: 0,
            length_bytes: 4,
        };
        let (i8_dtype, i8_shadow) = native_dense_shadow_bytes(&i8_spec, &[255, 0, 1, 127])
            .expect("i8 weights should promote into f32 shadow bytes");

        assert_eq!(i8_dtype, NativeTensorDataType::F32);
        assert_eq!(
            i8_shadow
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_le_bytes(
                    chunk.try_into().expect("chunk width should match")
                ))
                .collect::<Vec<_>>(),
            vec![-1.0, 0.0, 1.0, 127.0]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_dense_kernel_coverage_counts_byte_tensors_as_promoted_f32_support() {
        let byte_projection = MetalNativeTensorBinding {
            spec: NativeTensorSpec {
                name: "projection".to_string(),
                role: NativeTensorRole::AttentionO,
                layer_index: Some(0),
                dtype: NativeTensorDataType::U8,
                shape: vec![2, 2],
                file: PathBuf::from("projection.bin"),
                offset_bytes: 0,
                length_bytes: 4,
            },
            resolved_path: PathBuf::from("/tmp/projection.bin"),
        };
        let byte_norm = MetalNativeTensorBinding {
            spec: NativeTensorSpec {
                name: "norm".to_string(),
                role: NativeTensorRole::FfnNorm,
                layer_index: Some(0),
                dtype: NativeTensorDataType::I8,
                shape: vec![4],
                file: PathBuf::from("norm.bin"),
                offset_bytes: 0,
                length_bytes: 4,
            },
            resolved_path: PathBuf::from("/tmp/norm.bin"),
        };
        let mut coverage = MetalNativeDenseKernelCoverage::default();

        record_projection_binding_coverage(&mut coverage, &byte_projection);
        record_rms_norm_binding_coverage(&mut coverage, &byte_norm);

        assert_eq!(coverage.projection_f32_binding_count, 1);
        assert_eq!(coverage.rms_norm_f32_binding_count, 1);
        assert_eq!(coverage.projection_unsupported_binding_count, 0);
        assert_eq!(coverage.rms_norm_unsupported_binding_count, 0);
    }

    #[test]
    fn metal_assets_load_compiled_build_and_resolve_required_kernels() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

        let assets =
            MetalKernelAssets::from_build_dir(&fixture.build_dir).expect("assets should load");

        assert_eq!(assets.build_status(), MetalBuildStatus::Compiled);
        assert_eq!(assets.manifest().library_name, PHASE1_METAL_LIBRARY_NAME);
        assert_eq!(
            assets.default_block_size_tokens(),
            PHASE1_DEFAULT_BLOCK_SIZE_TOKENS
        );
        assert_eq!(
            assets.supported_block_size_tokens(),
            PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS
        );
        assert_eq!(
            assets
                .required_kernel("reshape_and_cache")
                .expect("required kernel should resolve")
                .tier,
            MetalKernelTier::Required
        );
        assert_eq!(
            assets.kernel("kv_scale_update").map(|kernel| kernel.tier),
            Some(MetalKernelTier::Optional)
        );
        assert_eq!(
            assets.kernel("swap_blocks").map(|kernel| kernel.tier),
            Some(MetalKernelTier::Deferred)
        );
        assert!(assets.compiled_metallib_path().is_some());
        assert!(!assets
            .compiled_metallib_bytes()
            .expect("compiled metallib should load")
            .is_empty());

        fixture.cleanup();
    }

    #[test]
    fn native_model_bindings_cover_all_manifest_tensors() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");

        assert_eq!(
            bindings.flattened_tensor_bindings().len() as u32,
            artifacts.summary().tensor_count
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_model_buffers_bind_real_tensor_bytes_into_metal_shared_buffers() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");

        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let stats = buffers.stats();

        // The projection fixture uses real tensor data with varying per-tensor sizes
        // (embedding: 160 B, 3 vectors: 32 B each, 8 matrices: 256 B each = 2304 B total).
        let expected_bytes: u64 = artifacts
            .manifest()
            .tensors
            .iter()
            .map(|t| t.length_bytes)
            .sum();
        assert!(stats.buffers_bound);
        assert_eq!(stats.buffer_count, artifacts.summary().tensor_count);
        assert_eq!(stats.buffer_bytes, expected_bytes);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_bringup_runner_executes_single_layer_fixture_with_real_build_artifacts() {
        let Some(build_dir) = compiled_repo_metal_build_dir() else {
            return;
        };
        let model_dir = write_projection_native_model_fixture();
        let runner =
            MetalBringupRunner::from_build_dir_and_model_artifacts(&build_dir, Some(&model_dir))
                .expect("metal bring-up runner should initialize");

        let output = runner.run(sample_runner_input());

        assert_eq!(
            output.execution_status,
            ExecutionStatus::Success,
            "{output:?}"
        );
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_runtime_real_model_tensor_inputs"
                    && *value > 0
            ));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_runtime_complete_model_forward_supported"
                    && *value > 0
            ));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(|(key, value)| key == "metal_dispatch_real_model_forward" && *value > 0));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_native_logits_projection"
                    && *value > 0
            ));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_native_projection_row_count"
                    && *value > 0
            ));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_cpu_projection_row_count"
                    && *value == 0
            ));

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_bringup_runner_reuses_multilayer_prefix_cache_for_native_decode_continuation() {
        let Some(build_dir) = compiled_repo_metal_build_dir() else {
            return;
        };
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let runner =
            MetalBringupRunner::from_build_dir_and_model_artifacts(&build_dir, Some(&model_dir))
                .expect("metal bring-up runner should initialize");

        let prefill_output = runner.run(sample_prefill_only_runner_input());

        assert_eq!(
            prefill_output.execution_status,
            ExecutionStatus::Success,
            "{prefill_output:?}"
        );
        assert!(prefill_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_prefix_layers_native_attention" && *value > 0
            ));
        assert!(prefill_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_prefix_cpu_reference_dispatch_count"
                    && *value == 0
            ));
        assert!(prefill_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_runtime_complete_model_forward_supported"
                    && *value > 0
            ));

        let continuation_output = runner.run(sample_decode_continuation_runner_input());

        assert_eq!(
            continuation_output.execution_status,
            ExecutionStatus::Success,
            "{continuation_output:?}"
        );
        assert!(continuation_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_prefix_layers_native_attention" && *value > 0
            ));
        assert!(continuation_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_prefix_cpu_reference_dispatch_count"
                    && *value == 0
            ));
        assert!(continuation_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_runtime_complete_model_forward_supported"
                    && *value > 0
            ));
        assert!(continuation_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(|(key, value)| key == "metal_dispatch_real_model_forward" && *value > 0));
        assert!(continuation_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_native_logits_projection"
                    && *value > 0
            ));
        assert!(continuation_output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_cpu_projection_row_count"
                    && *value == 0
            ));
        assert!(continuation_output
            .logits_outputs
            .iter()
            .any(|output| output.request_id == RequestId(17) && !output.logits.is_empty()));

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_bringup_runner_batches_direct_decode_logits_for_multiple_requests() {
        let Some(build_dir) = compiled_repo_metal_build_dir() else {
            return;
        };
        let model_dir = write_direct_decode_native_model_fixture(false);
        let runner =
            MetalBringupRunner::from_build_dir_and_model_artifacts(&build_dir, Some(&model_dir))
                .expect("metal bring-up runner should initialize");

        let output = runner.run(sample_decode_only_runner_input());

        assert_eq!(
            output.execution_status,
            ExecutionStatus::Success,
            "{output:?}"
        );
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(|(key, value)| key == "metal_dispatch_real_model_forward" && *value > 0));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_batched_logits_group_count"
                    && *value > 0
            ));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_batched_logits_token_count"
                    && *value >= 2
            ));
        assert!(output
            .route_metadata
            .crossover_decisions
            .iter()
            .any(
                |(key, value)| key == "metal_dispatch_direct_decode_cpu_projection_row_count"
                    && *value == 0
            ));
        assert_eq!(output.logits_outputs.len(), 2);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_use_bound_tensor_bytes() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("projection-backed staged inputs should resolve");
        let synthetic = synthetic_staged_inputs(&workload);

        assert_eq!(
            staged.source,
            MetalStagedInputSource::ModelConditionedMiniProjection
        );
        let expected_hidden =
            rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
        assert_f32_slice_close(&staged.query[..8], &expected_hidden, 1e-6);
        assert_f32_slice_close(
            &staged.key[..8],
            &scale_reference(&expected_hidden, 2.0),
            1e-6,
        );
        assert_f32_slice_close(
            &staged.value[..8],
            &scale_reference(&expected_hidden, 3.0),
            1e-6,
        );
        assert_ne!(staged.query, synthetic.query);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_row_rms_norm_matches_per_row_reference_path() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let attention_norm = buffers
            .binding_for(&bindings.layers[0].attention_norm)
            .expect("attention norm binding should resolve");
        let row_width = tensor_element_count(&attention_norm.meta.spec)
            .expect("attention norm tensor should have a shape");
        let epsilon = native_model_rms_norm_epsilon(&artifacts);
        let weight_offset = native_model_rms_norm_weight_offset(&artifacts);
        let weights =
            tensor_prefix_f32(attention_norm, row_width).expect("attention norm weights exist");
        let mut rows = vec![
            (0..row_width)
                .map(|index| index as f32 + 1.0)
                .collect::<Vec<_>>(),
            (0..row_width)
                .map(|index| (index as f32 + 1.0) * 0.5)
                .collect::<Vec<_>>(),
        ];
        let mut expected = rows.clone();
        for row in &mut expected {
            apply_rms_norm_with_weights_in_place(row, &weights, epsilon, weight_offset)
                .expect("per-row reference rms norm should succeed");
        }

        let tally = apply_batched_row_rms_norm_with_binding_in_place_with_tally(
            &mut rows,
            row_width,
            attention_norm,
            epsilon,
            weight_offset,
            None,
        )
        .expect("batched row rms norm should succeed");

        assert_eq!(rows, expected);
        assert_eq!(tally.native_rms_norm_element_count(), 0);
        assert_eq!(
            tally.cpu_rms_norm_element_count(),
            saturating_usize_to_u32(row_width * 2)
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_apply_optional_qk_head_norms() {
        let model_dir = write_projection_qk_norm_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("qk norm staged inputs should resolve");

        let expected_hidden =
            rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
        let expected_query =
            per_head_rms_norm_reference(&expected_hidden, 4, &[2.0, 1.0, 1.0, 1.0]);
        let expected_key = per_head_rms_norm_reference(
            &scale_reference(&expected_hidden, 2.0),
            4,
            &[3.0, 1.0, 1.0, 1.0],
        );

        assert_f32_slice_close(&staged.query[..8], &expected_query, 1e-6);
        assert_f32_slice_close(&staged.key[..8], &expected_key, 1e-6);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_apply_gemma_weight_offset_norms() {
        let model_dir = write_gemma_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("gemma staged inputs should resolve");

        let expected_hidden =
            rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
        assert_f32_slice_close(&staged.query[..8], &expected_hidden, 1e-6);
        assert_f32_slice_close(
            &staged.key[..8],
            &scale_reference(&expected_hidden, 2.0),
            1e-6,
        );
        assert_f32_slice_close(
            &staged.value[..8],
            &scale_reference(&expected_hidden, 3.0),
            1e-6,
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_hidden_states_before_final_layer_apply_gemma_embedding_scale() {
        let model_dir = write_gemma_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");

        let (hidden_states, final_layer_index, _) = model_hidden_states_before_final_layer(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("gemma hidden states should resolve");

        let scale = (8.0_f32).sqrt();
        assert_eq!(final_layer_index, 0);
        assert_f32_slice_close(
            &hidden_states[0],
            &scale_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], scale),
            1e-6,
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_expand_grouped_kv_heads_to_query_layout() {
        let model_dir = write_grouped_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("grouped projection staged inputs should resolve");

        assert_eq!(staged.layout, MetalDispatchNumericLayout::new(4, 2));
        let expected_hidden =
            rms_normalize_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
        assert_f32_slice_close(&staged.query[..8], &expected_hidden, 1e-6);
        assert_f32_slice_close(
            &staged.key[..8],
            &[
                expected_hidden[0] * 2.0,
                expected_hidden[1] * 2.0,
                expected_hidden[0] * 2.0,
                expected_hidden[1] * 2.0,
                expected_hidden[2] * 2.0,
                expected_hidden[3] * 2.0,
                expected_hidden[2] * 2.0,
                expected_hidden[3] * 2.0,
            ],
            1e-6,
        );
        assert_f32_slice_close(
            &staged.value[..8],
            &[
                expected_hidden[0] * 3.0,
                expected_hidden[1] * 3.0,
                expected_hidden[0] * 3.0,
                expected_hidden[1] * 3.0,
                expected_hidden[2] * 3.0,
                expected_hidden[3] * 3.0,
                expected_hidden[2] * 3.0,
                expected_hidden[3] * 3.0,
            ],
            1e-6,
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_apply_qwen_rope_for_nonzero_positions() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("projection-backed staged inputs should resolve");
        let normalized_hidden =
            rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
        let expected_query = [
            neox_rotate_reference(&normalized_hidden[..4], 1.0),
            neox_rotate_reference(&normalized_hidden[4..8], 1.0),
        ]
        .concat();
        let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
        let expected_key = [
            neox_rotate_reference(&doubled_hidden[..4], 1.0),
            neox_rotate_reference(&doubled_hidden[4..8], 1.0),
        ]
        .concat();

        assert_eq!(workload.scheduled_positions, vec![0, 1, 2, 3]);
        assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
        assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_apply_gemma_interleaved_rope_for_nonzero_positions() {
        let model_dir = write_gemma_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("gemma projection-backed staged inputs should resolve");
        let normalized_hidden =
            rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
        let expected_query = [
            interleaved_rotate_reference(&normalized_hidden[..4], 1.0),
            interleaved_rotate_reference(&normalized_hidden[4..8], 1.0),
        ]
        .concat();
        let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
        let expected_key = [
            interleaved_rotate_reference(&doubled_hidden[..4], 1.0),
            interleaved_rotate_reference(&doubled_hidden[4..8], 1.0),
        ]
        .concat();

        assert_eq!(workload.scheduled_positions, vec![0, 1, 2, 3]);
        assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
        assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_apply_manifest_rope_theta() {
        let model_dir = write_projection_custom_rope_native_model_fixture(100);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("custom rope staged inputs should resolve");
        let normalized_hidden =
            rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
        let expected_query = [
            neox_rotate_reference_with_base(&normalized_hidden[..4], 1.0, 100.0),
            neox_rotate_reference_with_base(&normalized_hidden[4..8], 1.0, 100.0),
        ]
        .concat();
        let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
        let expected_key = [
            neox_rotate_reference_with_base(&doubled_hidden[..4], 1.0, 100.0),
            neox_rotate_reference_with_base(&doubled_hidden[4..8], 1.0, 100.0),
        ]
        .concat();

        assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
        assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_gemma_staged_inputs_apply_manifest_rope_theta() {
        let model_dir = write_gemma_projection_custom_rope_native_model_fixture(100);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("custom gemma rope staged inputs should resolve");
        let normalized_hidden =
            rms_normalize_reference(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-6);
        let expected_query = [
            interleaved_rotate_reference_with_base(&normalized_hidden[..4], 1.0, 100.0),
            interleaved_rotate_reference_with_base(&normalized_hidden[4..8], 1.0, 100.0),
        ]
        .concat();
        let doubled_hidden = scale_reference(&normalized_hidden, 2.0);
        let expected_key = [
            interleaved_rotate_reference_with_base(&doubled_hidden[..4], 1.0, 100.0),
            interleaved_rotate_reference_with_base(&doubled_hidden[4..8], 1.0, 100.0),
        ]
        .concat();

        assert_f32_slice_close(&staged.query[8..16], &expected_query, 1e-5);
        assert_f32_slice_close(&staged.key[8..16], &expected_key, 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gemma_attention_output_uses_manifest_scale_and_softcap() {
        let default_model_dir = write_gemma_projection_native_model_fixture();
        let configured_model_dir =
            write_gemma_projection_attention_config_native_model_fixture(16, 1);

        let default_artifacts = NativeModelArtifacts::from_dir(&default_model_dir)
            .expect("default gemma artifacts should load");
        let configured_artifacts = NativeModelArtifacts::from_dir(&configured_model_dir)
            .expect("configured gemma artifacts should load");
        let default_bindings = MetalNativeModelBindings::from_artifacts(&default_artifacts)
            .expect("default bindings should load");
        let configured_bindings = MetalNativeModelBindings::from_artifacts(&configured_artifacts)
            .expect("configured bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let default_buffers =
            MetalNativeModelBufferBindings::from_model_bindings(&device, &default_bindings)
                .expect("default buffers should bind");
        let configured_buffers =
            MetalNativeModelBufferBindings::from_model_bindings(&device, &configured_bindings)
                .expect("configured buffers should bind");
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let default_staged = model_conditioned_staged_inputs(
            &default_artifacts,
            &default_bindings,
            &default_buffers,
            &input,
            &workload,
            None,
            None,
        )
        .expect("default staged inputs should resolve");
        let configured_staged = model_conditioned_staged_inputs(
            &configured_artifacts,
            &configured_bindings,
            &configured_buffers,
            &input,
            &workload,
            None,
            None,
        )
        .expect("configured staged inputs should resolve");

        assert_f32_slice_close(&default_staged.query, &configured_staged.query, 1e-6);
        assert_f32_slice_close(&default_staged.key, &configured_staged.key, 1e-6);
        assert_f32_slice_close(&default_staged.value, &configured_staged.value, 1e-6);

        let (_, default_used_native) = attention_output_from_model_layer(
            &default_artifacts,
            &workload,
            &default_staged,
            None,
            None,
        )
        .expect("default attention output should resolve");
        let (configured_attention, configured_used_native) = attention_output_from_model_layer(
            &configured_artifacts,
            &workload,
            &configured_staged,
            None,
            None,
        )
        .expect("configured attention output should resolve");
        assert!(!default_used_native);
        assert!(!configured_used_native);

        let default_attention = attention_output_from_model_layer(
            &default_artifacts,
            &workload,
            &default_staged,
            None,
            None,
        )
        .expect("default attention output should resolve")
        .0;
        assert!(
            default_attention
                .iter()
                .zip(configured_attention.iter())
                .any(|(default, configured)| (default - configured).abs() > 1e-5),
            "configured Gemma attention should differ when manifest scale/softcap change"
        );

        let numeric_workload = workload.with_numeric_layout(configured_staged.layout);
        let attention_config = native_model_reference_attention_config(
            &configured_artifacts,
            configured_staged.layout.head_dim as usize,
        );
        let expected_head = manual_attention_head_output(
            &numeric_workload,
            &configured_staged,
            1,
            0,
            attention_config,
        );
        let head_size = numeric_workload.numeric_layout.head_size() as usize;
        let head_dim = numeric_workload.numeric_layout.head_dim as usize;
        let token_base = head_size;
        assert_f32_slice_close(
            &configured_attention[token_base..token_base + head_dim],
            &expected_head,
            1e-6,
        );

        let _ = fs::remove_dir_all(default_model_dir);
        let _ = fs::remove_dir_all(configured_model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_use_final_layer_qkv_for_multilayer_fixture() {
        let model_dir = write_multilayer_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("projection-backed staged inputs should resolve");

        assert_eq!(
            staged.source,
            MetalStagedInputSource::ModelConditionedCpuPrefixAttention
        );
        assert_eq!(staged.prefix_attention_tally.native_dispatch_count(), 0);
        assert_eq!(
            staged.prefix_attention_tally.cpu_reference_dispatch_count(),
            2
        );
        assert!(staged.query.iter().all(|value| value.abs() <= 1e-6));
        assert!(staged.key.iter().all(|value| value.abs() <= 1e-6));
        assert!(staged.value.iter().all(|value| value.abs() <= 1e-6));

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_match_explicit_prefix_cache_for_multilayer_fixture() {
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let prefix_layer_caches = vec![Mutex::new(MetalPersistentLayerKvCache::default())];

        let staged_without_cache = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("projection-backed staged inputs should resolve without explicit cache");
        let staged_with_cache = model_conditioned_staged_inputs(
            &artifacts,
            &bindings,
            &buffers,
            &input,
            &workload,
            Some(prefix_layer_caches.as_slice()),
            None,
        )
        .expect("projection-backed staged inputs should resolve with explicit cache");

        assert_eq!(staged_without_cache.source, staged_with_cache.source);
        assert_eq!(
            staged_without_cache.prefix_attention_tally,
            staged_with_cache.prefix_attention_tally
        );
        assert_f32_slice_close(&staged_without_cache.query, &staged_with_cache.query, 1e-6);
        assert_f32_slice_close(&staged_without_cache.key, &staged_with_cache.key, 1e-6);
        assert_f32_slice_close(&staged_without_cache.value, &staged_with_cache.value, 1e-6);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_layer_qkv_staging_matches_per_item_staging_for_mixed_batch() {
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let token_embedding = buffers
            .binding_for(&bindings.token_embedding)
            .expect("token embedding binding should resolve");
        let (_, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)
            .expect("token embedding dimensions should resolve");
        let hidden_states = initial_model_hidden_states_cpu(
            token_embedding,
            &workload,
            (artifacts.manifest().hidden_size as usize).min(embedding_cols),
            native_model_embedding_scale(&artifacts),
        )
        .expect("initial hidden states should resolve");
        let layer = bindings
            .layers
            .first()
            .expect("multilayer fixture should expose at least one layer");
        let full_staged = stage_model_layer_qkv_inputs(
            &artifacts,
            layer,
            &buffers,
            &workload,
            &hidden_states,
            None,
        )
        .expect("full staged inputs should resolve");
        let mut merged_item_tally = PrefixAttentionExecutionTally::default();
        let mut token_cursor = 0_usize;

        for item in &input.execution_batch.items {
            let token_end = token_cursor + item.scheduled_token_count as usize;
            let item_input =
                runner_input_for_execution_item(&input, item).expect("item input should resolve");
            let item_workload = MetalDispatchWorkload::from_runner_input(&item_input)
                .expect("item workload should resolve");
            let item_hidden_states = hidden_states
                .get(token_cursor..token_end)
                .expect("item hidden states should slice");
            let item_staged = stage_model_layer_qkv_inputs(
                &artifacts,
                layer,
                &buffers,
                &item_workload,
                item_hidden_states,
                None,
            )
            .expect("per-item staged inputs should resolve");
            let sliced = slice_staged_inputs_for_token_range(&full_staged, token_cursor..token_end)
                .expect("full staged inputs should slice");

            assert_eq!(sliced.layout, item_staged.layout);
            assert_f32_slice_close(&sliced.query, &item_staged.query, 1e-6);
            assert_f32_slice_close(&sliced.key, &item_staged.key, 1e-6);
            assert_f32_slice_close(&sliced.value, &item_staged.value, 1e-6);
            merged_item_tally = merged_item_tally.merge(item_staged.prefix_attention_tally);
            token_cursor = token_end;
        }

        assert_eq!(token_cursor, hidden_states.len());
        assert_eq!(
            full_staged
                .prefix_attention_tally
                .cpu_reference_dispatch_count(),
            merged_item_tally.cpu_reference_dispatch_count()
        );
        assert_eq!(
            full_staged
                .prefix_attention_tally
                .cpu_projection_row_count(),
            merged_item_tally.cpu_projection_row_count()
        );
        assert_eq!(
            full_staged
                .prefix_attention_tally
                .cpu_rms_norm_element_count(),
            merged_item_tally.cpu_rms_norm_element_count()
        );
        assert_eq!(
            full_staged
                .prefix_attention_tally
                .cpu_ffn_activation_element_count(),
            merged_item_tally.cpu_ffn_activation_element_count()
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_layer_continuation_matches_per_item_continuation_for_mixed_batch() {
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let token_embedding = buffers
            .binding_for(&bindings.token_embedding)
            .expect("token embedding binding should resolve");
        let (_, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)
            .expect("token embedding dimensions should resolve");
        let hidden_states = initial_model_hidden_states_cpu(
            token_embedding,
            &workload,
            (artifacts.manifest().hidden_size as usize).min(embedding_cols),
            native_model_embedding_scale(&artifacts),
        )
        .expect("initial hidden states should resolve");
        let layer = bindings
            .layers
            .first()
            .expect("multilayer fixture should expose at least one layer");
        let attention_o = buffers
            .binding_for(&layer.attention_o)
            .expect("attention_o binding should resolve");
        let ffn_norm = buffers
            .binding_for(&layer.ffn_norm)
            .expect("ffn_norm binding should resolve");
        let ffn_down = buffers
            .binding_for(&layer.ffn_down)
            .expect("ffn_down binding should resolve");
        let staged = stage_model_layer_qkv_inputs(
            &artifacts,
            layer,
            &buffers,
            &workload,
            &hidden_states,
            None,
        )
        .expect("full staged inputs should resolve");
        let head_size = staged.layout.head_size() as usize;
        let attention_output = (0..hidden_states.len().checked_mul(head_size).unwrap_or(0))
            .map(|index| index as f32 * 0.03125 + 0.5)
            .collect::<Vec<_>>();
        let (batched_hidden_states, batched_tally) =
            project_hidden_states_from_layer_attention_output(
                &artifacts,
                layer,
                &buffers,
                attention_o,
                ffn_norm,
                ffn_down,
                &hidden_states,
                &attention_output,
                head_size,
                None,
            )
            .expect("batched continuation should resolve");
        let mut token_cursor = 0_usize;
        let mut per_item_hidden_states = Vec::new();
        let mut per_item_tally = PrefixAttentionExecutionTally::default();

        for item in &input.execution_batch.items {
            let token_end = token_cursor + item.scheduled_token_count as usize;
            let item_hidden_states = hidden_states
                .get(token_cursor..token_end)
                .expect("item hidden states should slice");
            let attention_base = token_cursor.checked_mul(head_size).expect("attention base");
            let attention_end = token_end.checked_mul(head_size).expect("attention end");
            let item_attention_output = attention_output
                .get(attention_base..attention_end)
                .expect("item attention output should slice");
            let (item_next_hidden_states, item_tally) =
                project_hidden_states_from_layer_attention_output(
                    &artifacts,
                    layer,
                    &buffers,
                    attention_o,
                    ffn_norm,
                    ffn_down,
                    item_hidden_states,
                    item_attention_output,
                    head_size,
                    None,
                )
                .expect("per-item continuation should resolve");
            per_item_hidden_states.extend(item_next_hidden_states);
            per_item_tally = per_item_tally.merge(item_tally);
            token_cursor = token_end;
        }

        assert_eq!(token_cursor, hidden_states.len());
        assert_eq!(batched_hidden_states.len(), per_item_hidden_states.len());
        for (batched_hidden, per_item_hidden) in batched_hidden_states
            .iter()
            .zip(per_item_hidden_states.iter())
        {
            assert_f32_slice_close(batched_hidden, per_item_hidden, 1e-6);
        }
        assert_eq!(
            batched_tally.cpu_projection_row_count(),
            per_item_tally.cpu_projection_row_count()
        );
        assert_eq!(
            batched_tally.cpu_rms_norm_element_count(),
            per_item_tally.cpu_rms_norm_element_count()
        );
        assert_eq!(
            batched_tally.cpu_ffn_activation_element_count(),
            per_item_tally.cpu_ffn_activation_element_count()
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn grouped_prefix_attention_matches_per_item_outputs_for_mixed_batch() {
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let token_embedding = buffers
            .binding_for(&bindings.token_embedding)
            .expect("token embedding binding should resolve");
        let (_, embedding_cols) = tensor_matrix_dimensions(&token_embedding.meta.spec)
            .expect("token embedding dimensions should resolve");
        let hidden_states = initial_model_hidden_states_cpu(
            token_embedding,
            &workload,
            (artifacts.manifest().hidden_size as usize).min(embedding_cols),
            native_model_embedding_scale(&artifacts),
        )
        .expect("initial hidden states should resolve");
        let layer = bindings
            .layers
            .first()
            .expect("multilayer fixture should expose at least one layer");
        let staged = stage_model_layer_qkv_inputs(
            &artifacts,
            layer,
            &buffers,
            &workload,
            &hidden_states,
            None,
        )
        .expect("full staged inputs should resolve");
        let item_token_ranges =
            execution_item_token_ranges(&input).expect("item token ranges should resolve");
        let (grouped_attention_output, grouped_tally) =
            collect_prefix_attention_outputs_with_item_fallback(
                &artifacts,
                &input,
                &staged,
                &item_token_ranges,
                0..input.execution_batch.items.len(),
                None,
                None,
            )
            .expect("grouped prefix attention should resolve");
        let mut token_cursor = 0_usize;
        let mut per_item_attention_output = Vec::new();
        let mut per_item_tally = PrefixAttentionExecutionTally::default();

        for item in &input.execution_batch.items {
            let token_end = token_cursor + item.scheduled_token_count as usize;
            let item_input =
                runner_input_for_execution_item(&input, item).expect("item input should resolve");
            let item_workload = MetalDispatchWorkload::from_runner_input(&item_input)
                .expect("item workload should resolve");
            let item_staged = slice_staged_inputs_for_token_range(&staged, token_cursor..token_end)
                .expect("item staged inputs should slice");
            let (item_attention_output, used_native_dispatch) = attention_output_from_model_layer(
                &artifacts,
                &item_workload,
                &item_staged,
                None,
                None,
            )
            .expect("per-item attention output should resolve");
            assert!(!used_native_dispatch);
            per_item_attention_output.extend(item_attention_output);
            per_item_tally = per_item_tally.record(used_native_dispatch);
            token_cursor = token_end;
        }

        assert_eq!(token_cursor, hidden_states.len());
        assert_f32_slice_close(&grouped_attention_output, &per_item_attention_output, 1e-6);
        assert_eq!(grouped_tally.native_dispatch_count(), 0);
        assert_eq!(grouped_tally.cpu_reference_dispatch_count(), 1);
        assert_eq!(
            per_item_tally.cpu_reference_dispatch_count(),
            input.execution_batch.items.len() as u32
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_direct_decode_reuses_cached_multilayer_hidden_states() {
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("multilayer staged inputs should resolve");
        let hidden_state_cache = staged
            .final_layer_hidden_state_cache
            .as_ref()
            .expect("model-conditioned staged inputs should retain hidden states");
        let mut attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
        let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .enumerate()
        {
            attention_output_bits[decode_base + index] = value.to_bits();
        }

        let replayed = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );
        let reused = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            Some(hidden_state_cache),
            None,
            None,
        );

        assert_eq!(reused.tokens, replayed.tokens);
        assert_eq!(reused.logits_outputs, replayed.logits_outputs);
        assert_eq!(
            reused.model_bound_ffn_decode,
            replayed.model_bound_ffn_decode
        );
        assert_eq!(
            reused.native_logits_projection_decode,
            replayed.native_logits_projection_decode
        );
        assert_eq!(reused.execution_tally, replayed.execution_tally);
        assert_eq!(reused.native_dense_tally, replayed.native_dense_tally);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_direct_decode_batches_multiple_decode_items_without_changing_outputs() {
        let model_dir = write_direct_decode_native_model_fixture(false);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_decode_only_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let (hidden_states, final_layer_index, _) = model_hidden_states_before_final_layer(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("hidden states before final layer should resolve");
        let final_layer = bindings
            .layers
            .get(final_layer_index)
            .expect("final layer should resolve");
        let token_width = PHASE1_NUMERIC_HEAD_SIZE as usize;
        let mut attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * token_width];
        let token_patterns = [
            [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [8.0_f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        for (token_index, pattern) in token_patterns.iter().enumerate() {
            let token_base = token_index * token_width;
            for (lane, value) in pattern.iter().enumerate() {
                attention_output_bits[token_base + lane] = value.to_bits();
            }
        }

        let batched = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        let mut expected_tokens = Vec::new();
        let mut expected_logits_outputs = Vec::new();
        let mut expected_model_bound_ffn_decode = false;
        let mut expected_native_logits_projection_decode = false;
        let mut expected_execution_tally = PrefixAttentionExecutionTally::default();
        let mut expected_native_dense_tally = DirectDecodeNativeDenseTally::default();
        let mut attention_index = 0_usize;
        for item in &input.execution_batch.items {
            if item.mode == ExecutionMode::Decode {
                let token_base = attention_index * token_width;
                let token_end = token_base + token_width;
                let hidden_index = attention_index + item.scheduled_token_count as usize - 1;
                let (
                    logits,
                    token_id,
                    used_model_bound_ffn,
                    vocab_rows_scanned,
                    used_native_logits_projection,
                    decode_native_dense_tally,
                ) = decode_logits_from_model_attention_output_with_metadata(
                    &artifacts,
                    &bindings,
                    &buffers,
                    final_layer,
                    hidden_states
                        .get(hidden_index)
                        .expect("decode hidden state should resolve"),
                    &attention_output_bits[token_base..token_end],
                    None,
                )
                .expect("per-item decode logits should resolve");
                expected_tokens.push((item.request_id, token_id));
                expected_logits_outputs.push(RequestLogitsOutput {
                    request_id: item.request_id,
                    logits,
                });
                expected_model_bound_ffn_decode |= used_model_bound_ffn;
                expected_native_logits_projection_decode |= used_native_logits_projection;
                expected_execution_tally = expected_execution_tally
                    .record_layer_continuation_tokens(1)
                    .record_logits_projection(1, vocab_rows_scanned);
                expected_native_dense_tally =
                    expected_native_dense_tally.merge(decode_native_dense_tally);
            }
            attention_index += item.scheduled_token_count as usize;
        }
        if expected_native_logits_projection_decode {
            expected_native_dense_tally =
                expected_native_dense_tally.record_batched_logits_group(expected_tokens.len());
        } else if expected_tokens.len() > 1 {
            expected_native_dense_tally =
                expected_native_dense_tally.record_batched_group_fallback(expected_tokens.len());
        }

        assert_eq!(batched.tokens, expected_tokens);
        assert_eq!(batched.logits_outputs, expected_logits_outputs);
        assert_eq!(
            batched.model_bound_ffn_decode,
            expected_model_bound_ffn_decode
        );
        assert_eq!(
            batched.native_logits_projection_decode,
            expected_native_logits_projection_decode
        );
        assert_eq!(batched.execution_tally, expected_execution_tally);
        assert_eq!(batched.native_dense_tally, expected_native_dense_tally);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn grouped_direct_decode_results_fall_back_to_singletons_when_group_processing_fails() {
        let dims = ModelBoundDecodeDims {
            input_width: 8,
            hidden_dim: 8,
            intermediate_dim: 16,
            vocab_rows: 32,
        };
        let groups = vec![vec![
            PreparedDirectDecodeItem {
                request_id: RequestId(9),
                dims,
                hidden: vec![0.0; dims.hidden_dim],
                attention_input: vec![0.0; dims.input_width],
            },
            PreparedDirectDecodeItem {
                request_id: RequestId(11),
                dims,
                hidden: vec![0.0; dims.hidden_dim],
                attention_input: vec![0.0; dims.input_width],
            },
        ]];
        let mut attempted_group_sizes = Vec::new();

        let collected =
            collect_model_bound_direct_decode_group_results_with_item_fallback(groups, |group| {
                attempted_group_sizes.push(group.len());
                if group.len() > 1 {
                    return None;
                }
                let item = group.first()?;
                Some(ModelBoundDirectDecodeResult {
                    tokens: vec![(
                        item.request_id,
                        u32::try_from(item.request_id.0).unwrap_or(u32::MAX),
                    )],
                    logits_outputs: vec![RequestLogitsOutput {
                        request_id: item.request_id,
                        logits: vec![item.request_id.0 as f32],
                    }],
                    model_bound_ffn_decode: true,
                    native_logits_projection_decode: true,
                    execution_tally: PrefixAttentionExecutionTally::default()
                        .record_layer_continuation_tokens(1),
                    native_dense_tally: DirectDecodeNativeDenseTally::default()
                        .record_batched_logits_group(1),
                })
            })
            .expect("singleton fallback should recover grouped processing");

        assert_eq!(attempted_group_sizes, vec![2, 1, 1]);
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].tokens, vec![(RequestId(9), 9)]);
        assert_eq!(collected[1].tokens, vec![(RequestId(11), 11)]);
        let merged_tally = collected
            .iter()
            .fold(DirectDecodeNativeDenseTally::default(), |tally, result| {
                tally.merge(result.native_dense_tally)
            });
        assert_eq!(merged_tally.batched_group_fallback_count, 1);
        assert_eq!(merged_tally.batched_group_fallback_token_count, 2);
        assert!(collected
            .iter()
            .all(|result| result.native_logits_projection_decode));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn grouped_direct_decode_results_recursively_preserve_batched_subgroups() {
        let dims = ModelBoundDecodeDims {
            input_width: 8,
            hidden_dim: 8,
            intermediate_dim: 16,
            vocab_rows: 32,
        };
        let groups = vec![vec![
            PreparedDirectDecodeItem {
                request_id: RequestId(3),
                dims,
                hidden: vec![0.0; dims.hidden_dim],
                attention_input: vec![0.0; dims.input_width],
            },
            PreparedDirectDecodeItem {
                request_id: RequestId(5),
                dims,
                hidden: vec![0.0; dims.hidden_dim],
                attention_input: vec![0.0; dims.input_width],
            },
            PreparedDirectDecodeItem {
                request_id: RequestId(7),
                dims,
                hidden: vec![0.0; dims.hidden_dim],
                attention_input: vec![0.0; dims.input_width],
            },
            PreparedDirectDecodeItem {
                request_id: RequestId(9),
                dims,
                hidden: vec![0.0; dims.hidden_dim],
                attention_input: vec![0.0; dims.input_width],
            },
        ]];
        let mut attempted_group_sizes = Vec::new();

        let collected =
            collect_model_bound_direct_decode_group_results_with_item_fallback(groups, |group| {
                attempted_group_sizes.push(group.len());
                if group.len() > 2 {
                    return None;
                }
                let tokens = group
                    .iter()
                    .map(|item| {
                        (
                            item.request_id,
                            u32::try_from(item.request_id.0).unwrap_or(u32::MAX),
                        )
                    })
                    .collect::<Vec<_>>();
                let logits_outputs = group
                    .iter()
                    .map(|item| RequestLogitsOutput {
                        request_id: item.request_id,
                        logits: vec![item.request_id.0 as f32],
                    })
                    .collect::<Vec<_>>();
                Some(ModelBoundDirectDecodeResult {
                    tokens,
                    logits_outputs,
                    model_bound_ffn_decode: true,
                    native_logits_projection_decode: true,
                    execution_tally: PrefixAttentionExecutionTally::default()
                        .record_layer_continuation_tokens(
                            u32::try_from(group.len()).unwrap_or(u32::MAX),
                        ),
                    native_dense_tally: DirectDecodeNativeDenseTally::default()
                        .record_batched_logits_group(group.len()),
                })
            })
            .expect("recursive subgroup fallback should preserve grouped execution");

        assert_eq!(attempted_group_sizes, vec![4, 2, 2]);
        assert_eq!(collected.len(), 2);
        assert_eq!(
            collected[0].tokens,
            vec![(RequestId(3), 3), (RequestId(5), 5)]
        );
        assert_eq!(
            collected[1].tokens,
            vec![(RequestId(7), 7), (RequestId(9), 9)]
        );
        let merged_tally = collected
            .iter()
            .fold(DirectDecodeNativeDenseTally::default(), |tally, result| {
                tally.merge(result.native_dense_tally)
            });
        assert_eq!(merged_tally.native_batched_logits_group_count, 2);
        assert_eq!(merged_tally.native_batched_logits_token_count, 4);
        assert_eq!(merged_tally.batched_group_fallback_count, 1);
        assert_eq!(merged_tally.batched_group_fallback_token_count, 4);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_conditioned_staged_inputs_use_full_hidden_width_beyond_32d_cap() {
        let model_dir = write_wide_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_prefill_only_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");

        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("wide projection staged inputs should resolve");

        assert_eq!(
            staged.source,
            MetalStagedInputSource::ModelConditionedMiniProjection
        );
        let mut wide_hidden = vec![0.0_f32; 40];
        wide_hidden[32..40].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let expected_hidden = rms_normalize_reference(&wide_hidden, 1e-6);
        assert_f32_slice_close(&staged.query[..8], &expected_hidden[32..40], 1e-6);
        assert_f32_slice_close(
            &staged.key[..8],
            &scale_reference(&expected_hidden[32..40], 2.0),
            1e-6,
        );
        assert_f32_slice_close(
            &staged.value[..8],
            &scale_reference(&expected_hidden[32..40], 3.0),
            1e-6,
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_runtime_staged_inputs_fails_closed_when_model_projection_is_invalid() {
        let model_dir = write_projection_native_model_fixture();
        let manifest_path = model_dir.join(crate::model::AX_NATIVE_MODEL_MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path).expect("manifest should read");
        let mut manifest =
            serde_json::from_slice::<crate::model::NativeModelManifest>(&manifest_bytes)
                .expect("manifest should parse");
        manifest.attention_head_dim = 3;
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");

        let error = NativeModelArtifacts::from_dir(&model_dir)
            .expect_err("invalid projection manifest should fail closed at load time");
        let crate::model::NativeModelError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest");
        };
        assert!(message.contains("tensor attention_o must have shape [8, 6]"));

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn numeric_trace_validation_accepts_model_conditioned_reference_path() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let input = sample_runner_input();
        let staged = model_conditioned_staged_inputs(
            &artifacts, &bindings, &buffers, &input, &workload, None, None,
        )
        .expect("projection-backed staged inputs should resolve");
        let simulated = reference_numeric_path_with_inputs(&workload, &staged);
        let trace = simulated_numeric_trace(&simulated);

        let summary = validate_numeric_trace_against_inputs(&workload, &staged, &trace)
            .expect("model-conditioned reference trace should validate");

        assert_eq!(
            summary.expected_key_cache_checksum,
            trace.key_cache_checksum
        );
        assert_eq!(
            summary.expected_attention_output_checksum,
            trace.attention_output_checksum
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_decode_tokens_use_lm_head_projection_for_decode_items() {
        let model_dir = write_direct_decode_native_model_fixture(false);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let mut attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
        let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .enumerate()
        {
            attention_output_bits[decode_base + index] = value.to_bits();
        }

        let direct_decode = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        assert_eq!(direct_decode.tokens, vec![(RequestId(9), 4)]);
        assert_eq!(direct_decode.logits_outputs.len(), 1);
        assert_eq!(direct_decode.logits_outputs[0].request_id, RequestId(9));
        assert_eq!(
            direct_decode.logits_outputs[0]
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as u32),
            Some(4)
        );
        assert!(!direct_decode.model_bound_ffn_decode);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_decode_tokens_apply_gemma_weight_offset_norms() {
        let model_dir = write_gemma_direct_decode_native_model_fixture(false);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let mut attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
        let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .enumerate()
        {
            attention_output_bits[decode_base + index] = value.to_bits();
        }

        let direct_decode_tokens = derive_model_bound_decode_tokens(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        assert_eq!(direct_decode_tokens, vec![(RequestId(9), 4)]);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_decode_tokens_fall_back_to_tied_embeddings_when_lm_head_is_absent() {
        let model_dir = write_direct_decode_native_model_fixture(true);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let mut attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
        let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .enumerate()
        {
            attention_output_bits[decode_base + index] = value.to_bits();
        }

        let direct_decode_tokens = derive_model_bound_decode_tokens(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        assert_eq!(direct_decode_tokens, vec![(RequestId(9), 4)]);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_decode_tokens_support_multilayer_cpu_prefix_forward() {
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let mut attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
        let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .enumerate()
        {
            attention_output_bits[decode_base + index] = value.to_bits();
        }

        let direct_decode_tokens = derive_model_bound_decode_tokens(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        assert_eq!(direct_decode_tokens.len(), 1);
        assert_eq!(direct_decode_tokens[0].0, RequestId(9));
        assert!(direct_decode_tokens[0].1 > 0);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_direct_decode_matches_explicit_prefix_cache_for_multilayer_fixture() {
        let model_dir = write_multilayer_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let prefix_layer_caches = vec![Mutex::new(MetalPersistentLayerKvCache::default())];
        let mut attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];
        let decode_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        for (index, value) in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .enumerate()
        {
            attention_output_bits[decode_base + index] = value.to_bits();
        }

        let direct_decode_without_cache = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );
        let direct_decode_with_cache = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            Some(prefix_layer_caches.as_slice()),
            None,
        );

        assert_eq!(
            direct_decode_without_cache.tokens,
            direct_decode_with_cache.tokens
        );
        assert_eq!(
            direct_decode_without_cache.model_bound_ffn_decode,
            direct_decode_with_cache.model_bound_ffn_decode
        );
        assert_eq!(
            direct_decode_without_cache.logits_outputs,
            direct_decode_with_cache.logits_outputs
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_decode_tokens_apply_split_ffn_continuation_before_projection() {
        let model_dir =
            write_ffn_decode_native_model_fixture(DirectDecodeFixtureGateUpLayout::Split);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];

        let direct_decode = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        assert_eq!(direct_decode.tokens, vec![(RequestId(9), 4)]);
        assert_eq!(direct_decode.logits_outputs.len(), 1);
        assert!(direct_decode.model_bound_ffn_decode);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_decode_tokens_apply_packed_ffn_continuation_before_projection() {
        let model_dir =
            write_ffn_decode_native_model_fixture(DirectDecodeFixtureGateUpLayout::Packed);
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];

        let direct_decode = derive_model_bound_direct_decode_result(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        assert_eq!(direct_decode.tokens, vec![(RequestId(9), 4)]);
        assert_eq!(direct_decode.logits_outputs.len(), 1);
        assert!(direct_decode.model_bound_ffn_decode);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_ffn_gate_up_paths_match_rowwise_reference_for_split_and_packed_layouts() {
        for layout in [
            DirectDecodeFixtureGateUpLayout::Split,
            DirectDecodeFixtureGateUpLayout::Packed,
        ] {
            let model_dir = write_ffn_decode_native_model_fixture(layout);
            let artifacts = NativeModelArtifacts::from_dir(&model_dir)
                .expect("native model artifacts should load");
            let bindings =
                MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
            let device = Device::system_default().expect("Metal device should exist on macOS");
            let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
                .expect("native model buffers should bind");
            let layer = bindings
                .layers
                .first()
                .expect("fixture should contain one layer");
            let input_rows = vec![
                vec![1.0, 0.5, -0.5, 0.25, 0.0, 0.75, -1.0, 0.5],
                vec![0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.8, -0.7],
            ];

            let mut expected_gate_rows = Vec::new();
            let mut expected_up_rows = Vec::new();
            let mut expected_projection_tally = DirectDecodeNativeDenseTally::default();
            let mut expected_activation_tally = DirectDecodeNativeDenseTally::default();
            for input in &input_rows {
                let (mut gate, up, projection_tally) =
                    project_ffn_gate_up_with_coverage(&layer.ffn_gate_up, &buffers, 8, input, None)
                        .expect("rowwise gate/up projection should succeed");
                expected_projection_tally = expected_projection_tally.merge(projection_tally);
                let used_native_activation =
                    apply_model_gate_up_product_with_path(&artifacts, &mut gate, &up, None)
                        .expect("rowwise gate/up product should succeed");
                expected_activation_tally = expected_activation_tally
                    .record_ffn_activation_elements(gate.len(), used_native_activation);
                expected_gate_rows.push(gate);
                expected_up_rows.push(up);
            }

            let (mut actual_gate_rows, actual_up_rows, actual_projection_tally) =
                project_batched_ffn_gate_up_with_tally(
                    &layer.ffn_gate_up,
                    &buffers,
                    8,
                    &input_rows,
                    8,
                    None,
                )
                .expect("batched gate/up projection should succeed");
            let actual_activation_tally = apply_batched_model_gate_up_product_in_place_with_tally(
                &artifacts,
                &mut actual_gate_rows,
                &actual_up_rows,
                8,
                None,
            )
            .expect("batched gate/up product should succeed");

            assert_eq!(actual_projection_tally, expected_projection_tally);
            assert_eq!(actual_activation_tally, expected_activation_tally);
            for (actual, expected) in actual_up_rows.iter().zip(expected_up_rows.iter()) {
                assert_f32_slice_close(actual, expected, 1e-6);
            }
            for (actual, expected) in actual_gate_rows.iter().zip(expected_gate_rows.iter()) {
                assert_f32_slice_close(actual, expected, 1e-6);
            }

            let _ = fs::remove_dir_all(model_dir);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn model_bound_decode_tokens_use_hidden_dimensions_beyond_32d_cap() {
        let model_dir = write_wide_direct_decode_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("native model artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("native model buffers should bind");
        let input = sample_runner_input();
        let attention_output_bits =
            vec![0_u32; input.execution_batch.total_scheduled_tokens as usize * 8];

        let direct_decode_tokens = derive_model_bound_decode_tokens(
            &input,
            &attention_output_bits,
            Some(&artifacts),
            Some(&bindings),
            Some(&buffers),
            None,
            None,
            None,
        );

        assert_eq!(direct_decode_tokens, vec![(RequestId(9), 2)]);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[test]
    fn annotate_staged_input_source_marks_model_conditioned_inputs() {
        let mut route_metadata = RouteMetadata::empty();

        annotate_staged_input_source(
            &mut route_metadata,
            MetalStagedInputSource::ModelConditionedMiniProjection,
        );

        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_token_only_inputs".to_string(), 0,)));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_layers_native_attention".to_string(),
            0,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 0,)));
        assert!(!route_metadata
            .crossover_decisions
            .iter()
            .any(|(key, _)| key == "metal_dispatch_real_model_tensor_inputs"));
    }

    #[test]
    fn annotate_staged_input_source_marks_cpu_prefix_attention() {
        let mut route_metadata = RouteMetadata::empty();

        annotate_staged_input_source(
            &mut route_metadata,
            MetalStagedInputSource::ModelConditionedCpuPrefixAttention,
        );

        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_layers_native_attention".to_string(),
            0,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 1,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
    }

    #[test]
    fn annotate_staged_input_source_marks_native_prefix_attention() {
        let mut route_metadata = RouteMetadata::empty();

        annotate_staged_input_source(
            &mut route_metadata,
            MetalStagedInputSource::ModelConditionedNativePrefixAttention,
        );

        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_layers_native_attention".to_string(),
            1,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 0,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
    }

    #[test]
    fn annotate_staged_input_source_marks_mixed_prefix_attention() {
        let mut route_metadata = RouteMetadata::empty();

        annotate_staged_input_source(
            &mut route_metadata,
            MetalStagedInputSource::ModelConditionedMixedPrefixAttention,
        );

        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_layers_native_attention".to_string(),
            1,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_prefix_layers_cpu_reference".to_string(), 1,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_model_conditioned_inputs".to_string(), 1,)));
    }

    #[test]
    fn complete_model_forward_support_requires_model_conditioned_source() {
        let single_layer_model = NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 1,
            tensor_count: 9,
            tie_word_embeddings: false,
        };
        let multilayer_model = NativeModelArtifactsSummary {
            model_family: "qwen3_dense".to_string(),
            tensor_format: crate::model::NativeTensorFormat::Safetensors,
            layer_count: 2,
            tensor_count: 18,
            tie_word_embeddings: false,
        };

        assert!(!complete_model_forward_support_for_source(
            Some(&single_layer_model),
            None
        ));
        assert!(!complete_model_forward_support_for_source(
            Some(&single_layer_model),
            Some(MetalStagedInputSource::SyntheticTokenIds)
        ));
        assert!(complete_model_forward_support_for_source(
            Some(&single_layer_model),
            Some(MetalStagedInputSource::ModelConditionedMiniProjection)
        ));
        assert!(!complete_model_forward_support_for_source(
            Some(&multilayer_model),
            Some(MetalStagedInputSource::ModelConditionedMiniProjection)
        ));
        assert!(complete_model_forward_support_for_source(
            Some(&multilayer_model),
            Some(MetalStagedInputSource::ModelConditionedNativePrefixAttention)
        ));
        assert!(complete_model_forward_support_for_source(
            Some(&multilayer_model),
            Some(MetalStagedInputSource::ModelConditionedMixedPrefixAttention)
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_prefix_attention_support_requires_prefill_only_self_contained_context() {
        let prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill-only workload should resolve");
        let mixed_workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("mixed workload should resolve");

        assert!(workload_supports_native_prefix_attention(
            &prefill_workload,
            None
        ));
        assert!(!workload_supports_native_prefix_attention(
            &mixed_workload,
            None
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mixed_batch_prefill_segment_retains_native_prefix_attention_eligibility() {
        let input = sample_runner_input();
        let prefill_item = input
            .execution_batch
            .items
            .iter()
            .find(|item| item.mode == ExecutionMode::Prefill)
            .expect("mixed batch should contain a prefill item");
        let decode_item = input
            .execution_batch
            .items
            .iter()
            .find(|item| item.mode == ExecutionMode::Decode)
            .expect("mixed batch should contain a decode item");

        let prefill_input = runner_input_for_execution_item(&input, prefill_item)
            .expect("prefill item should slice");
        let decode_input =
            runner_input_for_execution_item(&input, decode_item).expect("decode item should slice");
        let prefill_workload = MetalDispatchWorkload::from_runner_input(&prefill_input)
            .expect("prefill slice workload should resolve");
        let decode_workload = MetalDispatchWorkload::from_runner_input(&decode_input)
            .expect("decode slice workload should resolve");

        assert!(workload_supports_native_prefix_attention(
            &prefill_workload,
            None
        ));
        assert!(!workload_supports_native_prefix_attention(
            &decode_workload,
            None
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_prefix_attention_viability_matches_group_support_and_cache_state() {
        let mixed_workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("mixed workload should resolve");
        let prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill-only workload should resolve");
        let decode_workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());

        assert!(!native_prefix_attention_is_viable(
            &mixed_workload,
            true,
            Some(&layer_cache)
        ));
        assert!(native_prefix_attention_is_viable(
            &prefill_workload,
            true,
            Some(&layer_cache)
        ));
        assert!(!native_prefix_attention_is_viable(
            &decode_workload,
            true,
            Some(&layer_cache)
        ));

        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .apply_snapshot(
                &prefill_workload,
                &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                    &prefill_workload,
                    &prefill_reference,
                ),
            );

        assert!(native_prefix_attention_is_viable(
            &decode_workload,
            true,
            Some(&layer_cache)
        ));
        assert!(!native_prefix_attention_is_viable(
            &prefill_workload,
            false,
            Some(&layer_cache)
        ));
    }

    #[test]
    fn seeded_reference_numeric_path_reuses_prior_layer_cache_for_decode() {
        let prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        let decode_workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        let mut layer_cache = MetalPersistentLayerKvCache::default();
        layer_cache.apply_snapshot(
            &prefill_workload,
            &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                &prefill_workload,
                &prefill_reference,
            ),
        );
        let decode_staged = synthetic_staged_inputs(&decode_workload);
        let seeded_reference = {
            let cache_seed = layer_cache.seed_for_workload(&decode_workload);
            reference_numeric_path_with_inputs_and_cache_seed(
                &decode_workload,
                &decode_staged,
                Some(cache_seed),
            )
        };
        let baseline_reference =
            reference_numeric_path_with_inputs(&decode_workload, &decode_staged);

        assert_ne!(
            checksum_f32_slice(&seeded_reference.gather_key),
            checksum_f32_slice(&baseline_reference.gather_key)
        );
        assert_ne!(
            checksum_f32_slice(&seeded_reference.attention_output),
            checksum_f32_slice(&baseline_reference.attention_output)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn initialized_layer_cache_allows_decode_native_prefix_attention() {
        let prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        let decode_workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());

        assert!(!workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));

        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .apply_snapshot(
                &prefill_workload,
                &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                    &prefill_workload,
                    &prefill_reference,
                ),
            );

        assert!(workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn prefix_support_probe_preserves_initialized_slots_across_layout_mismatch() {
        let prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve")
                .with_numeric_layout(MetalDispatchNumericLayout::new(4, 8));
        let decode_workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        let decode_numeric_workload =
            decode_workload.with_numeric_layout(MetalDispatchNumericLayout::new(4, 8));
        let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());
        layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .apply_snapshot(
                &prefill_workload,
                &MetalDispatchKvCacheSnapshot {
                    key_cache: vec![1.0; prefill_workload.slot_numeric_capacity() as usize],
                    value_cache: vec![2.0; prefill_workload.slot_numeric_capacity() as usize],
                },
            );

        assert!(workload_supports_native_prefix_attention(
            &decode_numeric_workload,
            Some(&layer_cache)
        ));
        assert!(workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));
        assert!(workload_supports_native_prefix_attention(
            &decode_numeric_workload,
            Some(&layer_cache)
        ));

        let layer_cache = layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned");
        for slot in 0..4 {
            assert!(layer_cache.slot_initialized(slot));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn prefix_reuse_warmup_is_needed_until_all_prefix_layer_caches_are_initialized() {
        let prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        let decode_workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        let prefix_layer_caches = vec![
            Mutex::new(MetalPersistentLayerKvCache::default()),
            Mutex::new(MetalPersistentLayerKvCache::default()),
        ];
        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
            &prefill_workload,
            &prefill_reference,
        );

        assert!(prefix_reuse_warmup_needed(
            &decode_workload,
            prefix_layer_caches.as_slice(),
            2,
        ));

        prefix_layer_caches[0]
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .apply_snapshot(&prefill_workload, &prefill_snapshot);

        assert!(prefix_reuse_warmup_needed(
            &decode_workload,
            prefix_layer_caches.as_slice(),
            2,
        ));

        prefix_layer_caches[1]
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .apply_snapshot(&prefill_workload, &prefill_snapshot);

        assert!(!prefix_reuse_warmup_needed(
            &decode_workload,
            prefix_layer_caches.as_slice(),
            2,
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn copied_prefix_blocks_persist_into_layer_cache_for_future_native_decode() {
        let mut prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        prefill_workload.kv_slot_capacity = 32;
        prefill_workload.kv_block_capacity = 32;
        prefill_workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
            &prefill_workload,
            &prefill_reference,
        );
        let head_size = prefill_workload.numeric_layout.head_size() as usize;
        let copied_block_base = 16_usize * head_size;
        let copied_block_end = copied_block_base + 4 * head_size;

        assert_eq!(
            &prefill_snapshot.key_cache[copied_block_base..copied_block_end],
            &prefill_reference.key_cache[..4 * head_size]
        );
        assert_eq!(
            &prefill_snapshot.value_cache[copied_block_base..copied_block_end],
            &prefill_reference.value_cache[..4 * head_size]
        );

        let decode_input = RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(7),
                model_id: "qwen3_dense".into(),
                execution_plan_ref: Some("phase1.qwen3_dense.decode_copied_prefix".into()),
                items: vec![ExecutionItem {
                    request_id: RequestId(29),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![5],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 4,
                        end_exclusive: 5,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(29),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                }],
                total_scheduled_tokens: 1,
                route_metadata: RouteMetadata::empty(),
            },
            block_tables: vec![crate::runner::ResolvedBlockTable {
                request_id: RequestId(29),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(1)],
                },
            }],
        };
        let decode_workload = MetalDispatchWorkload::from_runner_input(&decode_input)
            .expect("decode workload should resolve");
        let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());

        assert!(!workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));

        layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .apply_snapshot(&prefill_workload, &prefill_snapshot);

        assert!(workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn copied_prefix_snapshot_only_marks_offsets_backed_by_real_prefix_tokens() {
        let mut prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        prefill_workload.kv_slot_capacity = 32;
        prefill_workload.kv_block_capacity = 32;
        prefill_workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
            &prefill_workload,
            &prefill_reference,
        );
        let mut layer_cache = MetalPersistentLayerKvCache::default();

        layer_cache.apply_snapshot(&prefill_workload, &prefill_snapshot);

        for slot in 16..20 {
            assert!(layer_cache.slot_initialized(slot));
        }
        for slot in 20..32 {
            assert!(!layer_cache.slot_initialized(slot));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn copied_prefix_sources_allow_native_prefix_attention_before_target_slots_are_marked() {
        let prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        let layer_cache = Mutex::new(MetalPersistentLayerKvCache::default());
        layer_cache
            .lock()
            .expect("metal prefix layer cache mutex should not be poisoned")
            .apply_snapshot(
                &prefill_workload,
                &MetalDispatchKvCacheSnapshot::from_reference_for_workload(
                    &prefill_workload,
                    &prefill_reference,
                ),
            );

        let mut decode_workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        decode_workload.kv_slot_capacity = 32;
        decode_workload.kv_block_capacity = 32;
        decode_workload.kv_metadata.gather_block_table = vec![16];
        decode_workload.kv_metadata.gather_block_table_stride = 1;

        assert!(!workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));

        decode_workload.kv_metadata.copy_block_mapping = vec![[0, 16]];

        assert!(workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn materialized_seed_cache_applies_copy_block_mapping_to_target_slots() {
        let mut workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        workload.kv_slot_capacity = 32;
        workload.kv_block_capacity = 32;
        workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
        let slot_capacity = workload.slot_numeric_capacity() as usize;
        let head_size = workload.numeric_layout.head_size() as usize;
        let block_width = workload.block_numeric_elements() as usize;
        let mut key_cache = vec![0.0_f32; slot_capacity];
        let mut value_cache = vec![0.0_f32; slot_capacity];
        for index in 0..block_width {
            key_cache[index] = index as f32 + 1.0;
            value_cache[index] = index as f32 + 101.0;
        }

        let owned_seed = materialize_copy_targets_into_owned_cache_seed(
            &workload,
            MetalDispatchKvCacheSeed {
                key_cache: &key_cache,
                value_cache: &value_cache,
            },
        );
        let copied_block_base = 16 * head_size;
        let copied_block_end = copied_block_base + block_width;

        assert_eq!(
            &owned_seed.key_cache[copied_block_base..copied_block_end],
            &key_cache[..block_width]
        );
        assert_eq!(
            &owned_seed.value_cache[copied_block_base..copied_block_end],
            &value_cache[..block_width]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn materialized_seed_cache_applies_transitive_copy_block_mapping_regardless_of_order() {
        let mut workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        workload.kv_slot_capacity = 48;
        workload.kv_block_capacity = 48;
        workload.kv_metadata.copy_block_mapping = vec![[16, 32], [0, 16]];
        let slot_capacity = workload.slot_numeric_capacity() as usize;
        let head_size = workload.numeric_layout.head_size() as usize;
        let block_width = workload.block_numeric_elements() as usize;
        let mut key_cache = vec![0.0_f32; slot_capacity];
        let mut value_cache = vec![0.0_f32; slot_capacity];
        for index in 0..block_width {
            key_cache[index] = index as f32 + 1.0;
            value_cache[index] = index as f32 + 101.0;
        }

        let owned_seed = materialize_copy_targets_into_owned_cache_seed(
            &workload,
            MetalDispatchKvCacheSeed {
                key_cache: &key_cache,
                value_cache: &value_cache,
            },
        );
        let copied_block_16_base = 16 * head_size;
        let copied_block_32_base = 32 * head_size;
        let copied_block_16_end = copied_block_16_base + block_width;
        let copied_block_32_end = copied_block_32_base + block_width;

        assert_eq!(
            &owned_seed.key_cache[copied_block_16_base..copied_block_16_end],
            &key_cache[..block_width]
        );
        assert_eq!(
            &owned_seed.value_cache[copied_block_16_base..copied_block_16_end],
            &value_cache[..block_width]
        );
        assert_eq!(
            &owned_seed.key_cache[copied_block_32_base..copied_block_32_end],
            &key_cache[..block_width]
        );
        assert_eq!(
            &owned_seed.value_cache[copied_block_32_base..copied_block_32_end],
            &value_cache[..block_width]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn self_contained_seed_from_staged_inputs_populates_scheduled_and_copied_slots() {
        let mut workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        workload.kv_slot_capacity = 32;
        workload.kv_block_capacity = 32;
        workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
        let staged_inputs = synthetic_staged_inputs(&workload);
        let owned_seed =
            self_contained_owned_cache_seed_from_staged_inputs(&workload, &staged_inputs)
                .expect("self-contained seed should build");
        let head_size = workload.numeric_layout.head_size() as usize;
        let populated_width = workload.scheduled_numeric_elements() as usize;
        let block_width = workload.block_numeric_elements() as usize;
        let copied_block_base = 16 * head_size;
        let copied_block_end = copied_block_base + block_width;

        assert_eq!(
            &owned_seed.key_cache[..populated_width],
            &staged_inputs.key[..populated_width]
        );
        assert_eq!(
            &owned_seed.value_cache[..populated_width],
            &staged_inputs.value[..populated_width]
        );
        assert_eq!(
            &owned_seed.key_cache[copied_block_base..copied_block_base + populated_width],
            &staged_inputs.key[..populated_width]
        );
        assert_eq!(
            &owned_seed.value_cache[copied_block_base..copied_block_base + populated_width],
            &staged_inputs.value[..populated_width]
        );
        assert!(
            owned_seed.key_cache[copied_block_base + populated_width..copied_block_end]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            owned_seed.value_cache[copied_block_base + populated_width..copied_block_end]
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn composed_seed_overrides_stale_cache_with_current_staged_inputs_before_copying() {
        let mut workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        workload.kv_slot_capacity = 32;
        workload.kv_block_capacity = 32;
        workload.kv_metadata.copy_block_mapping = vec![[0, 16]];
        let staged_inputs = synthetic_staged_inputs(&workload);
        let slot_capacity = workload.slot_numeric_capacity() as usize;
        let head_size = workload.numeric_layout.head_size() as usize;
        let populated_width = workload.scheduled_numeric_elements() as usize;
        let mut stale_key_cache = vec![-1.0_f32; slot_capacity];
        let mut stale_value_cache = vec![-2.0_f32; slot_capacity];
        let copied_block_base = 16 * head_size;

        for value in &mut stale_key_cache[copied_block_base..copied_block_base + populated_width] {
            *value = -9.0;
        }
        for value in &mut stale_value_cache[copied_block_base..copied_block_base + populated_width]
        {
            *value = -10.0;
        }

        let owned_seed = owned_cache_seed_with_staged_inputs_and_copy_targets(
            &workload,
            &staged_inputs,
            Some(MetalDispatchKvCacheSeed {
                key_cache: &stale_key_cache,
                value_cache: &stale_value_cache,
            }),
        )
        .expect("composed seed should build");

        assert_eq!(
            &owned_seed.key_cache[..populated_width],
            &staged_inputs.key[..populated_width]
        );
        assert_eq!(
            &owned_seed.value_cache[..populated_width],
            &staged_inputs.value[..populated_width]
        );
        assert_eq!(
            &owned_seed.key_cache[copied_block_base..copied_block_base + populated_width],
            &staged_inputs.key[..populated_width]
        );
        assert_eq!(
            &owned_seed.value_cache[copied_block_base..copied_block_base + populated_width],
            &staged_inputs.value[..populated_width]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn reference_snapshot_and_layer_cache_materialize_transitive_copy_targets() {
        let mut prefill_workload =
            MetalDispatchWorkload::from_runner_input(&sample_prefill_only_runner_input())
                .expect("prefill workload should resolve");
        prefill_workload.kv_slot_capacity = 48;
        prefill_workload.kv_block_capacity = 48;
        prefill_workload.kv_metadata.copy_block_mapping = vec![[16, 32], [0, 16]];
        let prefill_staged = synthetic_staged_inputs(&prefill_workload);
        let prefill_reference =
            reference_numeric_path_with_inputs(&prefill_workload, &prefill_staged);
        let prefill_snapshot = MetalDispatchKvCacheSnapshot::from_reference_for_workload(
            &prefill_workload,
            &prefill_reference,
        );
        let head_size = prefill_workload.numeric_layout.head_size() as usize;
        let populated_width = prefill_workload.scheduled_numeric_elements() as usize;
        let copied_block_16_base = 16 * head_size;
        let copied_block_32_base = 32 * head_size;

        assert_eq!(
            &prefill_snapshot.key_cache
                [copied_block_16_base..copied_block_16_base + populated_width],
            &prefill_staged.key[..populated_width]
        );
        assert_eq!(
            &prefill_snapshot.value_cache
                [copied_block_16_base..copied_block_16_base + populated_width],
            &prefill_staged.value[..populated_width]
        );
        assert_eq!(
            &prefill_snapshot.key_cache
                [copied_block_32_base..copied_block_32_base + populated_width],
            &prefill_staged.key[..populated_width]
        );
        assert_eq!(
            &prefill_snapshot.value_cache
                [copied_block_32_base..copied_block_32_base + populated_width],
            &prefill_staged.value[..populated_width]
        );

        let mut layer_cache = MetalPersistentLayerKvCache::default();
        layer_cache.apply_snapshot(&prefill_workload, &prefill_snapshot);
        for slot in 16..20 {
            assert!(layer_cache.slot_initialized(slot));
        }
        for slot in 32..36 {
            assert!(layer_cache.slot_initialized(slot));
        }

        let mut decode_workload =
            MetalDispatchWorkload::from_runner_input(&sample_decode_continuation_runner_input())
                .expect("decode continuation workload should resolve");
        decode_workload.kv_slot_capacity = 48;
        decode_workload.kv_block_capacity = 48;
        decode_workload.kv_metadata.slot_mapping = vec![36];
        decode_workload.kv_metadata.attention_block_table = vec![36];
        decode_workload.kv_metadata.gather_block_table = vec![32];
        decode_workload.kv_metadata.gather_block_table_stride = 1;

        let layer_cache = Mutex::new(layer_cache);
        assert!(workload_supports_native_prefix_attention(
            &decode_workload,
            Some(&layer_cache)
        ));
    }

    #[test]
    fn annotate_bringup_execution_flags_clears_numeric_scaffold_when_runtime_uses_model_tensors() {
        let mut route_metadata = RouteMetadata::empty();
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
            model_buffer_count: 4,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage {
                projection_f32_binding_count: 1,
                projection_f16_binding_count: 2,
                projection_bf16_binding_count: 3,
                projection_unsupported_binding_count: 4,
                rms_norm_f32_binding_count: 5,
                rms_norm_f16_binding_count: 6,
                rms_norm_bf16_binding_count: 7,
                rms_norm_unsupported_binding_count: 8,
            },
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: crate::model::NativeTensorFormat::Safetensors,
                layer_count: 1,
                tensor_count: 9,
                tie_word_embeddings: false,
            }),
        };

        annotate_runtime_summary(&mut route_metadata, &runtime);
        annotate_bringup_execution_flags(
            &mut route_metadata,
            &runtime,
            &[(RequestId(9), 4)],
            MetalBringupExecutionFlags {
                model_bound_ffn_decode: false,
                native_logits_projection_decode: true,
                complete_model_forward_supported: true,
                real_model_forward: false,
                prefix_attention_tally: PrefixAttentionExecutionTally {
                    native_dispatches: 1,
                    cpu_reference_dispatches: 0,
                    qkv_projection_tokens: 3,
                    layer_continuation_tokens: 2,
                    logits_projection_tokens: 1,
                    logits_vocab_scan_rows: 5,
                    native_projection_rows: 31,
                    cpu_projection_rows: 37,
                    native_rms_norm_elements: 41,
                    cpu_rms_norm_elements: 43,
                    native_ffn_activation_elements: 47,
                    cpu_ffn_activation_elements: 53,
                },
                direct_decode_native_dense_tally: DirectDecodeNativeDenseTally {
                    native_projection_rows: 17,
                    cpu_projection_rows: 19,
                    native_rms_norm_elements: 23,
                    cpu_rms_norm_elements: 29,
                    native_ffn_activation_elements: 31,
                    cpu_ffn_activation_elements: 37,
                    native_batched_logits_group_count: 1,
                    native_batched_logits_token_count: 2,
                    batched_group_fallback_count: 0,
                    batched_group_fallback_token_count: 0,
                },
            },
        );

        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_numeric_scaffold_only".to_string(), 0,)));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_complete_model_forward_supported".to_string(),
            1,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_real_model_forward".to_string(), 0,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_direct_decode_tokens".to_string(), 1,)));
        assert!(route_metadata
            .crossover_decisions
            .iter()
            .any(|(key, value)| {
                key == "metal_dispatch_direct_decode_checksum_lo" && *value > 0
            }));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_model_bound_ffn".to_string(),
            0,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_prefix_native_dispatch_count".to_string(), 1,)));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_cpu_reference_dispatch_count".to_string(),
            0,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_qkv_projection_token_count".to_string(), 3,)));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_layer_continuation_token_count".to_string(),
            2,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_logits_projection_token_count".to_string(),
            1,
        )));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_logits_vocab_scan_row_count".to_string(), 5,)));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_native_projection_row_count".to_string(),
            31,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_cpu_projection_row_count".to_string(),
            37,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_native_rms_norm_element_count".to_string(),
            41,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_cpu_rms_norm_element_count".to_string(),
            43,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_native_ffn_activation_element_count".to_string(),
            47,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_prefix_cpu_ffn_activation_element_count".to_string(),
            53,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_native_logits_projection".to_string(),
            1,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_native_projection_row_count".to_string(),
            17,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_cpu_projection_row_count".to_string(),
            19,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_native_rms_norm_element_count".to_string(),
            23,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_cpu_rms_norm_element_count".to_string(),
            29,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_native_ffn_activation_element_count".to_string(),
            31,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_cpu_ffn_activation_element_count".to_string(),
            37,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_batched_logits_group_count".to_string(),
            1,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_batched_logits_token_count".to_string(),
            2,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_batched_group_fallback_count".to_string(),
            0,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_direct_decode_batched_group_fallback_token_count".to_string(),
            0,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_native_projection_bf16_binding_count".to_string(),
            3,
        )));
        assert!(route_metadata.crossover_decisions.contains(&(
            "metal_dispatch_native_rms_norm_unsupported_binding_count".to_string(),
            8,
        )));
    }

    #[test]
    fn completed_real_model_forward_step_marks_pure_decode_batch_with_no_remaining_logits() {
        let input = sample_decode_only_runner_input();
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
            model_buffer_count: 4,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: crate::model::NativeTensorFormat::Safetensors,
                layer_count: 1,
                tensor_count: 9,
                tie_word_embeddings: false,
            }),
        };
        let mut output = successful_runner_output_from_input(&input);
        let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];

        let direct_logits_outputs = direct_decode_tokens
            .iter()
            .map(|(request_id, token_id)| RequestLogitsOutput {
                request_id: *request_id,
                logits: vec![0.0, *token_id as f32],
            })
            .collect::<Vec<_>>();

        apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

        assert!(completed_real_model_forward_step(
            &input,
            &output,
            &runtime,
            &direct_decode_tokens,
        ));

        let mut route_metadata = RouteMetadata::empty();
        annotate_bringup_execution_flags(
            &mut route_metadata,
            &runtime,
            &direct_decode_tokens,
            MetalBringupExecutionFlags {
                model_bound_ffn_decode: true,
                native_logits_projection_decode: true,
                complete_model_forward_supported: true,
                real_model_forward: true,
                prefix_attention_tally: PrefixAttentionExecutionTally {
                    native_dispatches: 2,
                    cpu_reference_dispatches: 0,
                    qkv_projection_tokens: 4,
                    layer_continuation_tokens: 3,
                    logits_projection_tokens: 1,
                    logits_vocab_scan_rows: 7,
                    native_projection_rows: 11,
                    cpu_projection_rows: 13,
                    native_rms_norm_elements: 17,
                    cpu_rms_norm_elements: 19,
                    native_ffn_activation_elements: 23,
                    cpu_ffn_activation_elements: 29,
                },
                direct_decode_native_dense_tally: DirectDecodeNativeDenseTally::default(),
            },
        );
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_real_model_forward".to_string(), 1,)));
    }

    #[test]
    fn completed_real_model_forward_step_accepts_mixed_prefill_decode_batches_when_decode_items_resolve(
    ) {
        let input = sample_runner_input();
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
            model_buffer_count: 4,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: crate::model::NativeTensorFormat::Safetensors,
                layer_count: 1,
                tensor_count: 9,
                tie_word_embeddings: false,
            }),
        };
        let mut output = successful_runner_output_from_input(&input);
        let direct_decode_tokens = vec![(RequestId(9), 17)];

        let direct_logits_outputs = direct_decode_tokens
            .iter()
            .map(|(request_id, token_id)| RequestLogitsOutput {
                request_id: *request_id,
                logits: vec![0.0, *token_id as f32],
            })
            .collect::<Vec<_>>();

        apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

        assert!(completed_real_model_forward_step(
            &input,
            &output,
            &runtime,
            &direct_decode_tokens,
        ));
    }

    #[test]
    fn completed_real_model_forward_step_marks_prefill_only_batch_without_remaining_logits() {
        let input = sample_prefill_only_runner_input();
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
            model_buffer_count: 4,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: crate::model::NativeTensorFormat::Safetensors,
                layer_count: 1,
                tensor_count: 9,
                tie_word_embeddings: false,
            }),
        };
        let output = successful_runner_output_from_input(&input);

        assert!(completed_real_model_forward_step(
            &input,
            &output,
            &runtime,
            &[],
        ));
    }

    #[test]
    fn completed_real_model_forward_step_accepts_multilayer_runtime_when_prefix_attention_is_native(
    ) {
        let input = sample_decode_only_runner_input();
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
            model_buffer_count: 4,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: crate::model::NativeTensorFormat::Safetensors,
                layer_count: 2,
                tensor_count: 18,
                tie_word_embeddings: false,
            }),
        };
        let unresolved_output = successful_runner_output_from_input(&input);
        let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];

        assert!(!completed_real_model_forward_step(
            &input,
            &unresolved_output,
            &runtime,
            &[],
        ));

        let mut output = unresolved_output;
        let direct_logits_outputs = direct_decode_tokens
            .iter()
            .map(|(request_id, token_id)| RequestLogitsOutput {
                request_id: *request_id,
                logits: vec![0.0, *token_id as f32],
            })
            .collect::<Vec<_>>();

        apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

        assert!(completed_real_model_forward_step(
            &input,
            &output,
            &runtime,
            &direct_decode_tokens,
        ));
    }

    #[test]
    fn completed_real_model_forward_step_rejects_multilayer_runtime_when_prefix_attention_is_cpu_reference(
    ) {
        let input = sample_decode_only_runner_input();
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
                state: MetalBinaryArchiveState::Loaded,
                attached_pipeline_count: 4,
                serialized: true,
                note: None,
            },
            command_queue_ready: true,
            model_conditioned_inputs: true,
            real_model_tensor_inputs: true,
            complete_model_forward_supported: false,
            model_bindings_prepared: true,
            model_buffers_bound: true,
            model_buffer_count: 4,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: crate::model::NativeTensorFormat::Safetensors,
                layer_count: 2,
                tensor_count: 18,
                tie_word_embeddings: false,
            }),
        };
        let mut output = successful_runner_output_from_input(&input);
        let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];

        let direct_logits_outputs = direct_decode_tokens
            .iter()
            .map(|(request_id, token_id)| RequestLogitsOutput {
                request_id: *request_id,
                logits: vec![0.0, *token_id as f32],
            })
            .collect::<Vec<_>>();

        apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

        assert!(!completed_real_model_forward_step(
            &input,
            &output,
            &runtime,
            &direct_decode_tokens,
        ));
    }

    #[test]
    fn completed_real_model_forward_step_accepts_multilayer_runtime_when_prefix_attention_is_mixed()
    {
        let input = sample_decode_only_runner_input();
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax-phase1.binary_archive.metallib"),
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
            model_buffer_count: 4,
            model_buffer_bytes: 4096,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: Some(NativeModelArtifactsSummary {
                model_family: "qwen3_dense".to_string(),
                tensor_format: crate::model::NativeTensorFormat::Safetensors,
                layer_count: 2,
                tensor_count: 18,
                tie_word_embeddings: false,
            }),
        };
        let mut output = successful_runner_output_from_input(&input);
        let direct_decode_tokens = vec![(RequestId(9), 17), (RequestId(11), 23)];
        let direct_logits_outputs = direct_decode_tokens
            .iter()
            .map(|(request_id, token_id)| RequestLogitsOutput {
                request_id: *request_id,
                logits: vec![0.0, *token_id as f32],
            })
            .collect::<Vec<_>>();

        apply_direct_decode_logits_to_runner_output(&mut output, &direct_logits_outputs);

        assert!(completed_real_model_forward_step(
            &input,
            &output,
            &runtime,
            &direct_decode_tokens,
        ));
    }

    #[test]
    fn metal_assets_reject_missing_required_kernel() {
        let fixture = write_phase1_fixture(
            MetalBuildStatus::Compiled,
            Some(|manifest: &mut MetalKernelManifest| {
                manifest
                    .kernels
                    .retain(|kernel| kernel.name != "copy_blocks");
            }),
        );

        let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
            .expect_err("assets should reject missing required kernel");
        let MetalRuntimeError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("missing required kernel copy_blocks"));

        fixture.cleanup();
    }

    #[test]
    fn metal_assets_reject_missing_deferred_kernel() {
        let fixture = write_phase1_fixture(
            MetalBuildStatus::Compiled,
            Some(|manifest: &mut MetalKernelManifest| {
                manifest
                    .kernels
                    .retain(|kernel| kernel.name != "swap_blocks");
            }),
        );

        let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
            .expect_err("assets should reject missing deferred kernel inventory");
        let MetalRuntimeError::InvalidManifest { message } = error else {
            panic!("expected invalid manifest error");
        };
        assert!(message.contains("missing deferred kernel swap_blocks"));

        fixture.cleanup();
    }

    #[test]
    fn metal_assets_reject_non_phase1_block_size_policy() {
        let fixture = write_phase1_fixture(
            MetalBuildStatus::Compiled,
            Some(|manifest: &mut MetalKernelManifest| {
                manifest.supported_block_size_tokens = vec![8, 16];
            }),
        );

        let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
            .expect_err("assets should reject unsupported block size policy");
        let MetalRuntimeError::InvalidBuildReport { message } = error else {
            panic!("expected invalid build report error");
        };
        assert!(message.contains("supported_block_size_tokens must be multiples of"));

        fixture.cleanup();
    }

    #[test]
    fn metal_asset_validator_loads_metallib_and_preserves_asset_contract() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
        let validator =
            MetalAssetValidator::from_build_dir(&fixture.build_dir).expect("validator should load");

        validator
            .validate_block_size_tokens(PHASE1_DEFAULT_BLOCK_SIZE_TOKENS)
            .expect("validator should accept supported native block size");
        assert_eq!(
            validator
                .metallib()
                .path
                .file_name()
                .and_then(|name| name.to_str()),
            Some("ax_phase1_dense_path.metallib")
        );
        assert_eq!(
            validator.resolved_kernel_names().len(),
            PHASE1_REQUIRED_METAL_KERNELS.len()
        );

        fixture.cleanup();
    }

    #[test]
    fn metal_asset_validator_fails_closed_on_unsupported_native_block_size() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
        let validator =
            MetalAssetValidator::from_build_dir(&fixture.build_dir).expect("validator should load");

        let error = validator
            .validate_block_size_tokens(8)
            .expect_err("validator should reject unsupported native block size");
        let MetalRuntimeError::UnsupportedNativeBlockSize {
            block_size_tokens,
            default_block_size_tokens,
            supported_block_size_tokens,
        } = error
        else {
            panic!("expected unsupported native block size error");
        };
        assert_eq!(block_size_tokens, 8);
        assert_eq!(default_block_size_tokens, PHASE1_DEFAULT_BLOCK_SIZE_TOKENS);
        assert_eq!(
            supported_block_size_tokens,
            PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS
        );

        fixture.cleanup();
    }

    #[test]
    fn metal_dispatch_workload_tracks_tokens_blocks_and_modes() {
        let input = sample_runner_input();

        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");

        assert_eq!(workload.scheduled_requests, 2);
        assert_eq!(workload.prefill_requests, 1);
        assert_eq!(workload.decode_requests, 1);
        assert_eq!(workload.scheduled_tokens, 4);
        assert_eq!(workload.scheduled_token_ids, vec![1, 2, 3, 4]);
        assert_eq!(workload.scheduled_positions, vec![0, 1, 2, 3]);
        assert_eq!(workload.resolved_blocks, 3);
        assert_eq!(workload.token_elements, 4);
        assert_eq!(workload.block_elements, 3);
        assert_eq!(workload.scratch_elements, 4);
        assert_eq!(workload.scheduled_numeric_elements(), 32);
        assert_eq!(workload.gather_numeric_elements(), 56);
        assert_eq!(workload.slot_numeric_capacity(), 384);
        assert_eq!(workload.kv_slot_capacity, 48);
        assert_eq!(workload.kv_block_capacity, 48);
        assert_eq!(workload.kv_metadata.slot_mapping, vec![0, 1, 2, 35]);
        assert_eq!(
            workload.kv_metadata.attention_block_table,
            vec![0, 1, 2, 35]
        );
        assert_eq!(workload.kv_metadata.gather_block_table, vec![0, 16, 32, 0]);
        assert_eq!(workload.kv_metadata.gather_block_table_stride, 2);
        assert_eq!(workload.kv_metadata.copy_block_mapping, vec![[0, 0]]);
        assert_eq!(workload.kv_metadata.seq_lens, vec![3, 4]);
        assert_eq!(workload.kv_metadata.cu_seq_lens, vec![0, 3, 7]);
        assert_eq!(workload.kv_metadata.scheduled_cu_seq_lens, vec![0, 3, 4]);
    }

    #[test]
    fn metal_dispatch_trace_builder_respects_pipeline_widths_and_kernel_kinds() {
        let workload = MetalDispatchWorkload {
            scheduled_requests: 2,
            prefill_requests: 1,
            decode_requests: 1,
            scheduled_tokens: 8,
            scheduled_token_ids: vec![11, 12, 13, 14, 15, 16, 17, 18],
            scheduled_positions: vec![0, 1, 2, 3, 4, 5, 6, 7],
            resolved_blocks: 2,
            token_elements: 8,
            block_elements: 2,
            scratch_elements: 8,
            kv_slot_capacity: 8,
            kv_block_capacity: 24,
            numeric_layout: MetalDispatchNumericLayout::default(),
            kv_metadata: MetalDispatchKvMetadata {
                block_size_tokens: 16,
                slot_mapping: vec![0, 1, 2, 3, 4, 5, 6, 7],
                attention_block_table: vec![0, 1, 2, 3, 20, 21, 22, 23],
                gather_block_table: vec![0, 16],
                gather_block_table_stride: 2,
                copy_block_mapping: vec![[0, 0], [16, 16]],
                seq_lens: vec![8],
                cu_seq_lens: vec![0, 8],
                scheduled_cu_seq_lens: vec![0, 8],
            },
        };
        let traces = build_dispatch_traces(
            &workload,
            &[
                MetalComputePipelineInfo {
                    function_name: "reshape_and_cache".to_string(),
                    thread_execution_width: 32,
                    max_total_threads_per_threadgroup: 128,
                    static_threadgroup_memory_length: 0,
                },
                MetalComputePipelineInfo {
                    function_name: "copy_blocks".to_string(),
                    thread_execution_width: 64,
                    max_total_threads_per_threadgroup: 64,
                    static_threadgroup_memory_length: 0,
                },
            ],
        );

        assert_eq!(traces.len(), 2);
        assert_eq!(traces[0].function_name, "reshape_and_cache");
        assert_eq!(traces[0].element_count, 64);
        assert_eq!(traces[0].threads_per_grid.width, 64);
        assert_eq!(traces[0].threads_per_threadgroup.width, 32);
        assert_eq!(traces[1].function_name, "copy_blocks");
        assert_eq!(traces[1].element_count, workload.copy_numeric_elements());
        assert_eq!(
            traces[1].threads_per_grid.width,
            u64::from(workload.copy_numeric_elements())
        );
        assert_eq!(traces[1].threads_per_threadgroup.width, 64);
    }

    #[test]
    fn metal_dispatch_trace_builder_runs_gather_before_attention() {
        let workload =
            MetalDispatchWorkload::from_runner_input(&sample_runner_input()).expect("workload");
        let traces = build_dispatch_traces(
            &workload,
            &[
                MetalComputePipelineInfo {
                    function_name: "paged_decode_attention".to_string(),
                    thread_execution_width: 32,
                    max_total_threads_per_threadgroup: 128,
                    static_threadgroup_memory_length: 0,
                },
                MetalComputePipelineInfo {
                    function_name: "copy_blocks".to_string(),
                    thread_execution_width: 32,
                    max_total_threads_per_threadgroup: 128,
                    static_threadgroup_memory_length: 0,
                },
                MetalComputePipelineInfo {
                    function_name: "gather_kv_cache".to_string(),
                    thread_execution_width: 32,
                    max_total_threads_per_threadgroup: 128,
                    static_threadgroup_memory_length: 0,
                },
                MetalComputePipelineInfo {
                    function_name: "reshape_and_cache".to_string(),
                    thread_execution_width: 32,
                    max_total_threads_per_threadgroup: 128,
                    static_threadgroup_memory_length: 0,
                },
            ],
        );

        assert_eq!(
            traces
                .iter()
                .map(|trace| trace.function_name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "reshape_and_cache",
                "gather_kv_cache",
                "paged_decode_attention",
                "copy_blocks",
            ]
        );
        assert_eq!(traces[1].element_count, workload.gather_numeric_elements());
        assert_eq!(
            traces[2].element_count,
            workload.attention_numeric_elements()
        );
    }

    #[test]
    fn metal_dispatch_workload_rejects_request_without_required_block_table_span() {
        let mut input = sample_runner_input();
        input.block_tables[0].block_table.block_ids.clear();

        let error = MetalDispatchWorkload::from_runner_input(&input)
            .expect_err("workload should reject missing block coverage");
        let MetalRuntimeError::InvalidDispatchInput { message } = error else {
            panic!("expected invalid dispatch input");
        };
        assert!(message.contains("requires block index 0"));
    }

    #[test]
    fn failed_metal_runner_output_marks_all_requests_as_errors() {
        let input = sample_runner_input();
        let workload =
            MetalDispatchWorkload::from_runner_input(&input).expect("workload should resolve");
        let runtime = MetalDispatchRuntimeInfo {
            device_name: "Apple M4 Max".to_string(),
            required_pipeline_count: 4,
            max_thread_execution_width: 64,
            binary_archive: MetalBinaryArchiveInfo {
                path: PathBuf::from("/tmp/ax_phase1_dense_path.binary_archive.metallib"),
                state: MetalBinaryArchiveState::Loaded,
                attached_pipeline_count: 4,
                serialized: true,
                note: None,
            },
            command_queue_ready: true,
            model_conditioned_inputs: false,
            real_model_tensor_inputs: false,
            complete_model_forward_supported: false,
            model_bindings_prepared: false,
            model_buffers_bound: false,
            model_buffer_count: 0,
            model_buffer_bytes: 0,
            native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
            model: None,
        };

        let output = failed_runner_output_from_input(
            &input,
            "metal dispatch exploded".to_string(),
            &workload,
            Some(&runtime),
        );

        assert_eq!(output.execution_status, ExecutionStatus::Failed);
        assert!(output.logits_handles.is_empty());
        assert!(output.logits_outputs.is_empty());
        assert_eq!(output.kv_write_summary.tokens_written, 0);
        assert_eq!(output.kv_write_summary.blocks_touched, 0);
        assert!(output
            .route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_failed".to_string(), 1)));
        assert!(output
            .route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_runtime_required_pipelines".to_string(), 4)));
        assert!(output
            .route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_binary_archive_state".to_string(), 2)));
        assert_eq!(output.request_updates.len(), 2);
        assert!(output
            .request_updates
            .iter()
            .all(|update| update.stop_reason == Some(StopReason::Error)
                && update.error.as_deref() == Some("metal dispatch exploded")));
    }

    #[test]
    fn derive_metal_decode_tokens_uses_attention_output_bits_for_decode_items() {
        let input = sample_runner_input();
        let direct_decode_tokens = derive_metal_decode_tokens_from_attention_bits(
            &input,
            &[
                1001.0_f32.to_bits(),
                1001.0_f32.to_bits(),
                1001.0_f32.to_bits(),
                1001.0_f32.to_bits(),
                1001.0_f32.to_bits(),
                1001.0_f32.to_bits(),
                1001.0_f32.to_bits(),
                1001.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1002.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                1003.0_f32.to_bits(),
                3015.0_f32.to_bits(),
                3015.0_f32.to_bits(),
                3015.0_f32.to_bits(),
                3015.0_f32.to_bits(),
                3015.0_f32.to_bits(),
                3015.0_f32.to_bits(),
                3015.0_f32.to_bits(),
                3015.0_f32.to_bits(),
            ],
        );

        assert_eq!(direct_decode_tokens, vec![(RequestId(9), 3015)]);
    }

    #[test]
    fn apply_direct_decode_logits_clears_logits_handles_for_resolved_requests() {
        let input = sample_runner_input();
        let mut output = successful_runner_output_from_input(&input);

        apply_direct_decode_logits_to_runner_output(
            &mut output,
            &[RequestLogitsOutput {
                request_id: RequestId(9),
                logits: vec![0.1, 3015.0],
            }],
        );

        assert!(output.logits_handles.is_empty());
        assert_eq!(output.logits_outputs.len(), 1);
        assert_eq!(
            output
                .logits_outputs
                .iter()
                .find(|output| output.request_id == RequestId(9))
                .map(|output| output.logits.clone()),
            Some(vec![0.1, 3015.0])
        );
    }

    #[test]
    fn metal_dispatch_execution_info_tracks_direct_decode_resolution() {
        let input = sample_runner_input();
        let mut output = successful_runner_output_from_input(&input);
        let direct_decode_tokens = vec![(RequestId(9), 3015)];

        apply_direct_decode_logits_to_runner_output(
            &mut output,
            &[RequestLogitsOutput {
                request_id: RequestId(9),
                logits: vec![0.1, 3015.0],
            }],
        );

        let execution = metal_dispatch_execution_info(
            &output,
            &direct_decode_tokens,
            true,
            true,
            PrefixAttentionExecutionTally {
                native_dispatches: 2,
                cpu_reference_dispatches: 1,
                qkv_projection_tokens: 6,
                layer_continuation_tokens: 4,
                logits_projection_tokens: 1,
                logits_vocab_scan_rows: 9,
                native_projection_rows: 23,
                cpu_projection_rows: 29,
                native_rms_norm_elements: 31,
                cpu_rms_norm_elements: 37,
                native_ffn_activation_elements: 41,
                cpu_ffn_activation_elements: 43,
            },
            DirectDecodeNativeDenseTally {
                native_projection_rows: 13,
                cpu_projection_rows: 17,
                native_rms_norm_elements: 19,
                cpu_rms_norm_elements: 23,
                native_ffn_activation_elements: 29,
                cpu_ffn_activation_elements: 31,
                native_batched_logits_group_count: 1,
                native_batched_logits_token_count: 2,
                batched_group_fallback_count: 0,
                batched_group_fallback_token_count: 0,
            },
        );

        assert_eq!(execution.direct_decode_token_count, 1);
        assert!(execution.direct_decode_checksum_lo > 0);
        assert_eq!(execution.logits_output_count, 1);
        assert_eq!(execution.remaining_logits_handle_count, 0);
        assert!(execution.model_bound_ffn_decode);
        assert!(execution.real_model_forward_completed);
        assert_eq!(execution.prefix_native_dispatch_count, 2);
        assert_eq!(execution.prefix_cpu_reference_dispatch_count, 1);
        assert_eq!(execution.qkv_projection_token_count, 6);
        assert_eq!(execution.layer_continuation_token_count, 4);
        assert_eq!(execution.logits_projection_token_count, 1);
        assert_eq!(execution.logits_vocab_scan_row_count, 9);
        assert_eq!(execution.prefix_native_projection_row_count, 23);
        assert_eq!(execution.prefix_cpu_projection_row_count, 29);
        assert_eq!(execution.prefix_native_rms_norm_element_count, 31);
        assert_eq!(execution.prefix_cpu_rms_norm_element_count, 37);
        assert_eq!(execution.prefix_native_ffn_activation_element_count, 41);
        assert_eq!(execution.prefix_cpu_ffn_activation_element_count, 43);
        assert_eq!(execution.direct_decode_native_projection_row_count, 13);
        assert_eq!(execution.direct_decode_cpu_projection_row_count, 17);
        assert_eq!(execution.direct_decode_native_rms_norm_element_count, 19);
        assert_eq!(execution.direct_decode_cpu_rms_norm_element_count, 23);
        assert_eq!(
            execution.direct_decode_native_ffn_activation_element_count,
            29
        );
        assert_eq!(execution.direct_decode_cpu_ffn_activation_element_count, 31);
        assert_eq!(execution.direct_decode_batched_logits_group_count, 1);
        assert_eq!(execution.direct_decode_batched_logits_token_count, 2);
        assert_eq!(execution.direct_decode_batched_group_fallback_count, 0);
        assert_eq!(
            execution.direct_decode_batched_group_fallback_token_count,
            0
        );
    }

    #[test]
    fn staged_numeric_values_follow_scheduled_token_ids() {
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let staged_key = staged_key_values(&workload);
        let staged_value = staged_value_values(&workload);
        let staged_query = staged_query_values(&workload);

        assert_eq!(staged_key.len(), 32);
        assert_eq!(staged_value.len(), 32);
        assert_eq!(staged_query.len(), 32);
        assert_eq!(
            &staged_key[..8],
            &[0.5, 0.53125, 0.5625, 0.59375, 0.75, 0.78125, 0.8125, 0.84375]
        );
        assert_eq!(
            &staged_value[8..16],
            &[2.0, 2.015625, 2.03125, 2.046875, 2.0625, 2.078125, 2.09375, 2.109375,]
        );
        assert_eq!(
            &staged_query[24..32],
            &[2.015625, 2.046875, 2.078125, 2.109375, 2.265625, 2.296875, 2.328125, 2.359375,]
        );
    }

    #[test]
    fn simulated_numeric_path_keeps_vectorized_kv_and_decode_contract() {
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let simulated = simulated_numeric_path(&workload);

        let decode_slot_base = 35 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        assert_eq!(
            &simulated.key_cache[decode_slot_base..decode_slot_base + 8],
            &[2.0, 2.03125, 2.0625, 2.09375, 2.25, 2.28125, 2.3125, 2.34375]
        );
        assert_eq!(simulated.gather_key.len(), 56);
        assert_eq!(simulated.gather_value.len(), 56);
        assert_eq!(simulated.attention_output.len(), 32);
        assert!(simulated
            .attention_output
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        let decode_output_base = 3 * PHASE1_NUMERIC_HEAD_SIZE as usize;
        assert_eq!(
            decode_token_from_attention_bits(
                &simulated.attention_output[decode_output_base..decode_output_base + 8]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            ),
            4
        );
        assert_eq!(
            checksum_f32_slice(&simulated.key_cache),
            checksum_f32_slice(&simulated.copy_key)
        );
        assert_eq!(
            checksum_f32_slice(&simulated.value_cache),
            checksum_f32_slice(&simulated.copy_value)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn numeric_trace_reference_validation_accepts_reference_path() {
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let simulated = simulated_numeric_path(&workload);
        let trace = simulated_numeric_trace(&simulated);

        let summary = validate_numeric_trace_against_reference(&workload, &trace)
            .expect("reference trace should validate");
        assert_eq!(
            summary.expected_key_cache_checksum,
            trace.key_cache_checksum
        );
        assert_eq!(
            summary.expected_attention_output_checksum,
            trace.attention_output_checksum
        );
        assert_eq!(
            summary.expected_gather_output_checksum,
            trace.gather_output_checksum
        );
        assert_eq!(
            summary.expected_copy_output_checksum,
            trace.copy_output_checksum
        );
        assert_eq!(summary.attention_max_abs_diff_microunits, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn numeric_trace_reference_validation_rejects_attention_drift() {
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let simulated = simulated_numeric_path(&workload);
        let mut trace = simulated_numeric_trace(&simulated);
        trace.attention_output_bits[7] =
            (f32::from_bits(trace.attention_output_bits[7]) + 0.25).to_bits();

        let error = validate_numeric_trace_against_reference(&workload, &trace)
            .expect_err("drifted attention output should fail validation");
        let MetalRuntimeError::NumericValidationMismatch { stage, message } = error else {
            panic!("expected numeric validation mismatch");
        };
        assert_eq!(stage, "attention_output");
        assert!(message.contains("value drift"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn numeric_trace_reference_validation_rejects_copy_checksum_drift() {
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let simulated = simulated_numeric_path(&workload);
        let mut trace = simulated_numeric_trace(&simulated);
        trace.copy_output_checksum ^= 0x55;

        let error = validate_numeric_trace_against_reference(&workload, &trace)
            .expect_err("copy checksum drift should fail validation");
        let MetalRuntimeError::NumericValidationMismatch { stage, message } = error else {
            panic!("expected numeric validation mismatch");
        };
        assert_eq!(stage, "copy_blocks");
        assert!(message.contains("checksum mismatch"));
    }

    #[test]
    fn annotate_successful_dispatch_surfaces_validation_summary_metrics() {
        let input = sample_runner_input();
        let workload = MetalDispatchWorkload::from_runner_input(&input).expect("workload");
        let expected_copy_workload_elements = workload.copy_numeric_elements();
        let simulated = simulated_numeric_path(&workload);
        let mut trace = simulated_numeric_trace(&simulated);
        trace.validation = Some(
            validate_numeric_trace_against_reference(&workload, &trace)
                .expect("reference summary should exist"),
        );

        let mut route_metadata = RouteMetadata::empty();
        annotate_successful_dispatch(
            &mut route_metadata,
            &MetalDispatchTrace {
                command_queue_label: "test.queue".to_string(),
                command_buffer_label: "test.buffer".to_string(),
                command_buffer_status: MetalCommandBufferStatus::Completed,
                runtime: MetalDispatchRuntimeInfo {
                    device_name: "Apple M4 Max".to_string(),
                    required_pipeline_count: 4,
                    max_thread_execution_width: 64,
                    binary_archive: MetalBinaryArchiveInfo {
                        path: PathBuf::from("/tmp/ax_phase1_dense_path.binary_archive.metallib"),
                        state: MetalBinaryArchiveState::Loaded,
                        attached_pipeline_count: 4,
                        serialized: true,
                        note: None,
                    },
                    command_queue_ready: true,
                    model_conditioned_inputs: false,
                    real_model_tensor_inputs: false,
                    complete_model_forward_supported: false,
                    model_bindings_prepared: false,
                    model_buffers_bound: false,
                    model_buffer_count: 0,
                    model_buffer_bytes: 0,
                    native_dense_kernel_coverage: MetalNativeDenseKernelCoverage::default(),
                    model: None,
                },
                workload,
                arena: MetalDispatchArenaInfo {
                    token_capacity: 8,
                    slot_capacity: 64,
                    attention_ref_capacity: 8,
                    gather_ref_capacity: 8,
                    gather_output_capacity: 8,
                    copy_pair_capacity: 4,
                    sequence_capacity: 4,
                    reused_existing: false,
                    grew_existing: false,
                },
                execution: MetalDispatchExecutionInfo::default(),
                kernels: Vec::new(),
                numeric: trace,
            },
        );

        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_numeric_reference_validated".to_string(), 1,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_runtime_required_pipelines".to_string(), 4,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_binary_archive_state".to_string(), 2,)));
        assert!(route_metadata
            .crossover_decisions
            .contains(&("metal_dispatch_binary_archive_serialized".to_string(), 1,)));
        assert!(route_metadata
            .crossover_decisions
            .iter()
            .any(|(key, value)| {
                key == "metal_dispatch_copy_workload_elements"
                    && *value == expected_copy_workload_elements
            }));
        assert!(route_metadata
            .crossover_decisions
            .iter()
            .any(|(key, value)| {
                key == "metal_dispatch_attention_max_abs_diff_microunits" && *value == 0
            }));
    }

    #[test]
    fn metal_runtime_bringup_requires_compiled_build_report() {
        let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);

        let error = MetalRuntimeBringup::from_build_dir(&fixture.build_dir)
            .expect_err("bring-up should require compiled build artifacts");
        let MetalRuntimeError::BuildNotCompiled { status } = error else {
            panic!("expected build-not-compiled error");
        };
        assert_eq!(status, MetalBuildStatus::SkippedToolchainUnavailable);

        fixture.cleanup();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_runtime_bringup_rejects_invalid_metallib_bytes() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

        let error = MetalRuntimeBringup::from_build_dir(&fixture.build_dir)
            .expect_err("bring-up should reject fake metallib bytes");
        let MetalRuntimeError::LoadCompiledLibrary { path, message } = error else {
            panic!("expected metallib load error");
        };
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("ax_phase1_dense_path.metallib")
        );
        assert!(!message.is_empty());

        fixture.cleanup();
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn metal_runtime_bringup_requires_macos_host() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

        let error = MetalRuntimeBringup::from_build_dir(&fixture.build_dir)
            .expect_err("bring-up should reject non-macos hosts");
        let MetalRuntimeError::UnsupportedPlatform { host_os } = error else {
            panic!("expected unsupported platform error");
        };
        assert_eq!(host_os, std::env::consts::OS);

        fixture.cleanup();
    }

    #[test]
    fn metal_kernel_builder_skips_when_toolchain_is_unavailable() {
        let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
        let request = MetalKernelBuildRequest {
            manifest_path: fixture.root.join("metal/phase1-kernels.json"),
            output_dir: fixture.build_dir.clone(),
            doctor: sample_build_doctor(false, false),
        };

        let artifacts =
            build_phase1_kernel_artifacts(&request).expect("builder should emit skipped report");

        assert_eq!(
            artifacts.build_status(),
            MetalBuildStatus::SkippedToolchainUnavailable
        );
        assert!(artifacts.build_report.compile_commands.is_empty());
        assert!(artifacts.build_report.outputs.air.is_none());
        assert!(artifacts.build_report.outputs.metalar.is_none());
        assert!(artifacts.build_report.outputs.metallib.is_none());
        assert!(artifacts.doctor_path.is_file());
        assert!(artifacts.build_report_path.is_file());
        let summary =
            fs::read_to_string(&artifacts.summary_path).expect("summary file should be readable");
        assert!(summary.contains("skipped_toolchain_unavailable"));

        fixture.cleanup();
    }

    #[test]
    fn metal_kernel_builder_fails_closed_on_source_manifest_drift() {
        let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
        let source_path = fixture.root.join("metal/kernels/phase1_dense_path.metal");
        fs::write(
            &source_path,
            r#"
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
"#,
        )
        .expect("drifted source should write");

        let request = MetalKernelBuildRequest {
            manifest_path: fixture.root.join("metal/phase1-kernels.json"),
            output_dir: fixture.build_dir.clone(),
            doctor: sample_build_doctor(true, true),
        };

        let artifacts =
            build_phase1_kernel_artifacts(&request).expect("builder should emit failed report");

        assert_eq!(artifacts.build_status(), MetalBuildStatus::FailedCompile);
        assert!(artifacts.build_report.compile_commands.is_empty());
        assert!(artifacts
            .build_report
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("kv_scale_update")));
        assert!(artifacts.build_report.outputs.metallib.is_none());

        fixture.cleanup();
    }

    #[test]
    fn metal_kernel_builder_compiles_with_fake_xcrun_toolchain() {
        let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
        let bin_dir = fixture.root.join("fake-bin");
        fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
        let fake_xcrun = bin_dir.join("xcrun");
        fs::write(&fake_xcrun, fake_xcrun_script()).expect("fake xcrun should write");
        let mut permissions = fs::metadata(&fake_xcrun)
            .expect("fake xcrun metadata should load")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_xcrun, permissions).expect("fake xcrun should be executable");

        let request = MetalKernelBuildRequest {
            manifest_path: fixture.root.join("metal/phase1-kernels.json"),
            output_dir: fixture.build_dir.clone(),
            doctor: sample_build_doctor(true, true),
        };

        let _guard = env_lock().lock().expect("env lock should acquire");
        let original_path = std::env::var_os("PATH");
        let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());
        std::env::set_var("PATH", &fake_path);

        let artifacts = build_phase1_kernel_artifacts(&request)
            .expect("builder should compile through fake toolchain");

        if let Some(path) = original_path {
            std::env::set_var("PATH", path);
        } else {
            std::env::remove_var("PATH");
        }

        assert_eq!(artifacts.build_status(), MetalBuildStatus::Compiled);
        assert!(!artifacts.reused_existing_artifacts());
        assert_eq!(artifacts.build_report.compile_commands.len(), 3);
        assert!(artifacts
            .build_report
            .outputs
            .air
            .as_deref()
            .is_some_and(Path::is_file));
        assert!(artifacts
            .build_report
            .outputs
            .metalar
            .as_deref()
            .is_some_and(Path::is_file));
        assert!(artifacts
            .build_report
            .outputs
            .metallib
            .as_deref()
            .is_some_and(Path::is_file));
        assert_eq!(
            artifacts
                .build_report
                .outputs
                .metallib_sha256
                .as_deref()
                .map(str::len),
            Some(64)
        );

        fixture.cleanup();
    }

    #[test]
    fn metal_kernel_builder_reuses_valid_compiled_artifacts_without_recompiling() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
        let bin_dir = fixture.root.join("fake-bin");
        fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
        let fake_xcrun = bin_dir.join("xcrun");
        fs::write(&fake_xcrun, fake_failing_xcrun_script())
            .expect("fake failing xcrun should write");
        let mut permissions = fs::metadata(&fake_xcrun)
            .expect("fake failing xcrun metadata should load")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_xcrun, permissions)
            .expect("fake failing xcrun should be executable");

        let mut doctor = sample_build_doctor(true, true);
        doctor.metal_toolchain.metal.version = Some("Apple metal version 99999.1".to_string());

        let request = MetalKernelBuildRequest {
            manifest_path: fixture.root.join("metal/phase1-kernels.json"),
            output_dir: fixture.build_dir.clone(),
            doctor: doctor.clone(),
        };

        let _guard = env_lock().lock().expect("env lock should acquire");
        let original_path = std::env::var_os("PATH");
        let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());
        std::env::set_var("PATH", &fake_path);

        let artifacts = build_phase1_kernel_artifacts(&request)
            .expect("builder should reuse validated compiled artifacts");

        if let Some(path) = original_path {
            std::env::set_var("PATH", path);
        } else {
            std::env::remove_var("PATH");
        }

        assert_eq!(artifacts.build_status(), MetalBuildStatus::Compiled);
        assert!(artifacts.reused_existing_artifacts());
        assert_eq!(artifacts.build_report.doctor, doctor);
        assert!(artifacts
            .build_report
            .outputs
            .metallib
            .as_deref()
            .is_some_and(Path::is_file));

        fixture.cleanup();
    }

    fn write_phase1_fixture(
        status: MetalBuildStatus,
        manifest_edit: Option<fn(&mut MetalKernelManifest)>,
    ) -> Phase1Fixture {
        let root = unique_test_dir("metal-fixture");
        let metal_dir = root.join("metal");
        let kernels_dir = metal_dir.join("kernels");
        let build_dir = root.join("build").join("metal");
        fs::create_dir_all(&kernels_dir).expect("kernels directory should create");
        fs::create_dir_all(&build_dir).expect("build directory should create");

        let source_path = kernels_dir.join("phase1_dense_path.metal");
        let source_text = phase1_source_text();
        fs::write(&source_path, source_text.as_bytes()).expect("source file should write");

        let manifest_path = metal_dir.join("phase1-kernels.json");
        let mut manifest = MetalKernelManifest {
            schema_version: PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION.to_string(),
            native_target: PHASE1_METAL_NATIVE_TARGET.to_string(),
            metal_language_standard: PHASE1_METAL_LANGUAGE_STANDARD.to_string(),
            library_name: PHASE1_METAL_LIBRARY_NAME.to_string(),
            default_block_size_tokens: PHASE1_DEFAULT_BLOCK_SIZE_TOKENS,
            supported_block_size_tokens: PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS.to_vec(),
            source_file: PathBuf::from("metal/kernels/phase1_dense_path.metal"),
            toolchain_requirements: REQUIRED_TOOLCHAIN_REQUIREMENTS
                .iter()
                .map(|tool| (*tool).to_string())
                .collect(),
            build_gate: PHASE1_METAL_BUILD_GATE.to_string(),
            kernels: phase1_kernel_specs(),
        };
        if let Some(edit) = manifest_edit {
            edit(&mut manifest);
        }
        write_json_file(&manifest_path, &manifest);

        let air_path = build_dir.join("ax_phase1_dense_path.air");
        let metalar_path = build_dir.join("ax_phase1_dense_path.metalar");
        let metallib_path = build_dir.join("ax_phase1_dense_path.metallib");
        let mut outputs = MetalBuildOutputs {
            air: None,
            metalar: None,
            metallib: None,
            air_sha256: None,
            metalar_sha256: None,
            metallib_sha256: None,
        };
        let reason = match status {
            MetalBuildStatus::Compiled => {
                fs::write(&air_path, b"fake-air").expect("air file should write");
                fs::write(&metalar_path, b"fake-metalar").expect("metalar should write");
                fs::write(&metallib_path, b"fake-metallib").expect("metallib should write");
                outputs.air = Some(air_path.clone());
                outputs.metalar = Some(metalar_path.clone());
                outputs.metallib = Some(metallib_path.clone());
                outputs.air_sha256 = Some(sha256_hex(b"fake-air"));
                outputs.metalar_sha256 = Some(sha256_hex(b"fake-metalar"));
                outputs.metallib_sha256 = Some(sha256_hex(b"fake-metallib"));
                None
            }
            MetalBuildStatus::SkippedToolchainUnavailable => {
                Some("Metal toolchain is incomplete on this machine".to_string())
            }
            MetalBuildStatus::SkippedNotReady => Some(
                "AX native bring-up is not allowed on this machine without override".to_string(),
            ),
            MetalBuildStatus::FailedCompile => Some("command failed with exit code 1".to_string()),
            MetalBuildStatus::Unknown => None,
        };
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
            doctor: MetalBuildDoctorReport {
                status: match status {
                    MetalBuildStatus::Compiled => "ready".to_string(),
                    MetalBuildStatus::SkippedToolchainUnavailable => "not_ready".to_string(),
                    MetalBuildStatus::SkippedNotReady => "not_ready".to_string(),
                    MetalBuildStatus::FailedCompile => "bringup_only".to_string(),
                    MetalBuildStatus::Unknown => "not_ready".to_string(),
                },
                bringup_allowed: matches!(
                    status,
                    MetalBuildStatus::Compiled | MetalBuildStatus::FailedCompile
                ),
                native_runtime_ready: status == MetalBuildStatus::Compiled,
                metal_toolchain_fully_available: !matches!(
                    status,
                    MetalBuildStatus::SkippedToolchainUnavailable
                ),
                host: MetalBuildHostReport {
                    os: "macos".to_string(),
                    arch: "aarch64".to_string(),
                    detected_soc: Some("Apple M4 Max".to_string()),
                    supported_native_runtime: true,
                    unsupported_host_override_active: false,
                },
                metal_toolchain: MetalBuildToolchainReport {
                    fully_available: !matches!(
                        status,
                        MetalBuildStatus::SkippedToolchainUnavailable
                    ),
                    metal: MetalBuildToolStatus {
                        available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                        version: Some("Apple metal version 36000.4".to_string()),
                    },
                    metallib: MetalBuildToolStatus {
                        available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                        version: Some("Apple metallib version 36000.4".to_string()),
                    },
                    metal_ar: MetalBuildToolStatus {
                        available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                        version: Some("Apple metal-ar version 36000.4".to_string()),
                    },
                },
            },
            kernels: manifest.kernels.clone(),
            source_sha256: sha256_hex(source_text.as_bytes()),
            outputs,
            compile_commands: if status == MetalBuildStatus::Compiled {
                vec![
                    vec!["xcrun".to_string(), "metal".to_string()],
                    vec!["xcrun".to_string(), "metal-ar".to_string()],
                    vec!["xcrun".to_string(), "metallib".to_string()],
                ]
            } else {
                Vec::new()
            },
            status,
            reason,
        };
        write_json_file(&build_dir.join("build_report.json"), &build_report);

        Phase1Fixture { root, build_dir }
    }

    fn phase1_source_text() -> &'static str {
        r#"
// The parser should ignore commented kernels such as:
// kernel void commented_line_kernel() {}
/*
kernel void commented_block_kernel() {}
*/
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
kernel void kv_scale_update() {}
kernel void gather_embedding_rows_f32() {}
kernel void gather_embedding_rows_f16() {}
kernel void gather_embedding_rows_bf16() {}
kernel void decode_logits_projection_f32() {}
kernel void decode_logits_projection_f16() {}
kernel void decode_logits_projection_bf16() {}
kernel void decode_logits_projection_batched_f32() {}
kernel void decode_logits_projection_batched_f16() {}
kernel void decode_logits_projection_batched_bf16() {}
kernel void logits_argmax_f32() {}
kernel void logits_argmax_batched_f32() {}
kernel void sample_argmax_logprob_f32() {}
kernel void sample_argmax_logprob_batched_f32() {}
kernel void rms_norm_f32() {}
kernel void rms_norm_f16() {}
kernel void rms_norm_bf16() {}
kernel void rms_norm_batched_f32() {}
kernel void rms_norm_batched_f16() {}
kernel void rms_norm_batched_bf16() {}
kernel void ffn_gate_silu_product_f32() {}
kernel void ffn_gate_gelu_approx_product_f32() {}
kernel void apply_rope_f32() {}
kernel void expand_grouped_kv_heads_f32() {}
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
            MetalKernelSpec {
                name: "gather_embedding_rows_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native embedding row gather for f32 token embeddings".to_string(),
            },
            MetalKernelSpec {
                name: "gather_embedding_rows_f16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native embedding row gather for f16 token embeddings".to_string(),
            },
            MetalKernelSpec {
                name: "gather_embedding_rows_bf16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native embedding row gather for bf16 token embeddings".to_string(),
            },
            MetalKernelSpec {
                name: "decode_logits_projection_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "f32 decode logits projection".to_string(),
            },
            MetalKernelSpec {
                name: "decode_logits_projection_f16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "f16 decode logits projection".to_string(),
            },
            MetalKernelSpec {
                name: "decode_logits_projection_bf16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "bf16 decode logits projection".to_string(),
            },
            MetalKernelSpec {
                name: "decode_logits_projection_batched_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "batched f32 decode logits projection".to_string(),
            },
            MetalKernelSpec {
                name: "decode_logits_projection_batched_f16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "batched f16 decode logits projection".to_string(),
            },
            MetalKernelSpec {
                name: "decode_logits_projection_batched_bf16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "batched bf16 decode logits projection".to_string(),
            },
            MetalKernelSpec {
                name: "logits_argmax_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native top-1 logits scan".to_string(),
            },
            MetalKernelSpec {
                name: "logits_argmax_batched_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native batched top-1 logits scan".to_string(),
            },
            MetalKernelSpec {
                name: "sample_argmax_logprob_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native deterministic top-1 sampling with logprob".to_string(),
            },
            MetalKernelSpec {
                name: "sample_argmax_logprob_batched_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native batched deterministic top-1 sampling with logprob".to_string(),
            },
            MetalKernelSpec {
                name: "rms_norm_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native RMSNorm".to_string(),
            },
            MetalKernelSpec {
                name: "rms_norm_f16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native f16 RMSNorm".to_string(),
            },
            MetalKernelSpec {
                name: "rms_norm_bf16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native bf16 RMSNorm".to_string(),
            },
            MetalKernelSpec {
                name: "rms_norm_batched_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native batched f32 RMSNorm".to_string(),
            },
            MetalKernelSpec {
                name: "rms_norm_batched_f16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native batched f16 RMSNorm".to_string(),
            },
            MetalKernelSpec {
                name: "rms_norm_batched_bf16".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native batched bf16 RMSNorm".to_string(),
            },
            MetalKernelSpec {
                name: "ffn_gate_silu_product_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native SiLU gate-up activation product".to_string(),
            },
            MetalKernelSpec {
                name: "ffn_gate_gelu_approx_product_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native GELU-approx gate-up activation product".to_string(),
            },
            MetalKernelSpec {
                name: "apply_rope_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native Q/K rotary position embedding application".to_string(),
            },
            MetalKernelSpec {
                name: "expand_grouped_kv_heads_f32".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "native grouped-KV expansion into query-head layout".to_string(),
            },
        ]
    }

    fn sample_runner_input() -> RunnerInput {
        RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(3),
                model_id: "qwen3_dense".into(),
                execution_plan_ref: Some("phase1.qwen3_dense.dense_prefill".into()),
                items: vec![
                    ExecutionItem {
                        request_id: RequestId(7),
                        mode: ExecutionMode::Prefill,
                        input_token_slice: vec![1, 2, 3],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 0,
                            end_exclusive: 3,
                        },
                        scheduled_token_count: 3,
                        block_table_ref: RequestId(7),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                    ExecutionItem {
                        request_id: RequestId(9),
                        mode: ExecutionMode::Decode,
                        input_token_slice: vec![4],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 3,
                            end_exclusive: 4,
                        },
                        scheduled_token_count: 1,
                        block_table_ref: RequestId(9),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                ],
                total_scheduled_tokens: 4,
                route_metadata: RouteMetadata {
                    execution_plan: Some("phase1.qwen3_dense.dense_prefill".into()),
                    attention_route: Some("qwen3_dense_prefill".into()),
                    kv_mode: Some("paged_metadata".into()),
                    prefix_cache_path: Some("metadata_lookup".into()),
                    barrier_mode: Some("serial".into()),
                    crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
                },
            },
            block_tables: vec![
                crate::runner::ResolvedBlockTable {
                    request_id: RequestId(7),
                    block_table: BlockTableView {
                        cache_group_id: CacheGroupId(1),
                        block_ids: vec![BlockId(0), BlockId(1)],
                    },
                },
                crate::runner::ResolvedBlockTable {
                    request_id: RequestId(9),
                    block_table: BlockTableView {
                        cache_group_id: CacheGroupId(1),
                        block_ids: vec![BlockId(2)],
                    },
                },
            ],
        }
    }

    fn sample_decode_only_runner_input() -> RunnerInput {
        RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(4),
                model_id: "qwen3_dense".into(),
                execution_plan_ref: Some("phase1.qwen3_dense.decode_only".into()),
                items: vec![
                    ExecutionItem {
                        request_id: RequestId(9),
                        mode: ExecutionMode::Decode,
                        input_token_slice: vec![4],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 3,
                            end_exclusive: 4,
                        },
                        scheduled_token_count: 1,
                        block_table_ref: RequestId(9),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                    ExecutionItem {
                        request_id: RequestId(11),
                        mode: ExecutionMode::Decode,
                        input_token_slice: vec![8],
                        reused_prefix_token_slice: Vec::new(),
                        position_range: PositionRange {
                            start: 5,
                            end_exclusive: 6,
                        },
                        scheduled_token_count: 1,
                        block_table_ref: RequestId(11),
                        prefix_tokens_reused: 0,
                        prefix_blocks_reused: 0,
                    },
                ],
                total_scheduled_tokens: 2,
                route_metadata: RouteMetadata {
                    execution_plan: Some("phase1.qwen3_dense.decode_only".into()),
                    attention_route: Some("qwen3_dense_decode".into()),
                    kv_mode: Some("paged_metadata".into()),
                    prefix_cache_path: Some("metadata_lookup".into()),
                    barrier_mode: Some("serial".into()),
                    crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
                },
            },
            block_tables: vec![
                crate::runner::ResolvedBlockTable {
                    request_id: RequestId(9),
                    block_table: BlockTableView {
                        cache_group_id: CacheGroupId(1),
                        block_ids: vec![BlockId(2)],
                    },
                },
                crate::runner::ResolvedBlockTable {
                    request_id: RequestId(11),
                    block_table: BlockTableView {
                        cache_group_id: CacheGroupId(1),
                        block_ids: vec![BlockId(3)],
                    },
                },
            ],
        }
    }

    #[cfg(target_os = "macos")]
    fn compiled_repo_metal_build_dir() -> Option<PathBuf> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .to_path_buf();
        let build_dir = repo_root.join("build/metal");
        let build_report_path = build_dir.join("build_report.json");
        let build_report = fs::read_to_string(&build_report_path).ok()?;
        let parsed: serde_json::Value = serde_json::from_str(&build_report).ok()?;
        (parsed.get("status").and_then(serde_json::Value::as_str) == Some("compiled"))
            .then_some(build_dir)
    }

    fn sample_prefill_only_runner_input() -> RunnerInput {
        RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(5),
                model_id: "qwen3_dense".into(),
                execution_plan_ref: Some("phase1.qwen3_dense.prefill_only".into()),
                items: vec![ExecutionItem {
                    request_id: RequestId(17),
                    mode: ExecutionMode::Prefill,
                    input_token_slice: vec![1, 2, 3, 4],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 0,
                        end_exclusive: 4,
                    },
                    scheduled_token_count: 4,
                    block_table_ref: RequestId(17),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                }],
                total_scheduled_tokens: 4,
                route_metadata: RouteMetadata {
                    execution_plan: Some("phase1.qwen3_dense.prefill_only".into()),
                    attention_route: Some("qwen3_dense_prefill".into()),
                    kv_mode: Some("paged_metadata".into()),
                    prefix_cache_path: Some("metadata_lookup".into()),
                    barrier_mode: Some("serial".into()),
                    crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
                },
            },
            block_tables: vec![crate::runner::ResolvedBlockTable {
                request_id: RequestId(17),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(0)],
                },
            }],
        }
    }

    fn sample_decode_continuation_runner_input() -> RunnerInput {
        RunnerInput {
            block_size_tokens: 16,
            execution_batch: ExecutionBatch {
                step_id: StepId(6),
                model_id: "qwen3_dense".into(),
                execution_plan_ref: Some("phase1.qwen3_dense.decode_continuation".into()),
                items: vec![ExecutionItem {
                    request_id: RequestId(17),
                    mode: ExecutionMode::Decode,
                    input_token_slice: vec![4],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: 4,
                        end_exclusive: 5,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: RequestId(17),
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                }],
                total_scheduled_tokens: 1,
                route_metadata: RouteMetadata {
                    execution_plan: Some("phase1.qwen3_dense.decode_continuation".into()),
                    attention_route: Some("qwen3_dense_decode".into()),
                    kv_mode: Some("paged_metadata".into()),
                    prefix_cache_path: Some("metadata_lookup".into()),
                    barrier_mode: Some("serial".into()),
                    crossover_decisions: vec![("prefix_reused_requests".into(), 0)],
                },
            },
            block_tables: vec![crate::runner::ResolvedBlockTable {
                request_id: RequestId(17),
                block_table: BlockTableView {
                    cache_group_id: CacheGroupId(1),
                    block_ids: vec![BlockId(0)],
                },
            }],
        }
    }

    #[test]
    fn metal_assets_reject_unknown_build_status() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Unknown, None);

        let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
            .expect_err("assets should reject unknown build status");
        let MetalRuntimeError::InvalidBuildReport { message } = error else {
            panic!("expected invalid build report error");
        };
        assert!(message.contains("status unknown is not allowed"));

        fixture.cleanup();
    }

    #[test]
    fn successful_metal_runner_output_matches_expected_decode_sampling_contract() {
        let input = sample_runner_input();

        let output = successful_runner_output_from_input(&input);

        assert_eq!(output.execution_status, ExecutionStatus::Success);
        assert_eq!(output.request_updates.len(), 2);
        assert_eq!(output.request_updates[0].tokens_executed, 3);
        assert_eq!(output.request_updates[1].tokens_executed, 1);
        assert_eq!(output.logits_handles, vec![RequestId(9)]);
        assert!(output.logits_outputs.is_empty());
        assert_eq!(output.kv_write_summary.tokens_written, 4);
        assert_eq!(output.kv_write_summary.blocks_touched, 3);
        assert_eq!(output.route_metadata, input.execution_batch.route_metadata);
    }

    #[test]
    fn metal_assets_reject_unknown_build_status_before_manifest_resolution() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Unknown, None);
        let manifest_path = fixture.root.join("metal/phase1-kernels.json");
        fs::remove_file(&manifest_path).expect("manifest file should be removable");

        let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
            .expect_err("assets should reject unknown status before reading manifest");
        let MetalRuntimeError::InvalidBuildReport { message } = error else {
            panic!("expected invalid build report error");
        };
        assert!(message.contains("status unknown is not allowed"));

        fixture.cleanup();
    }

    #[test]
    fn metal_assets_reject_source_kernel_manifest_drift() {
        let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
        let source_path = fixture.root.join("metal/kernels/phase1_dense_path.metal");
        let drifted_source = r#"
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
"#;
        fs::write(&source_path, drifted_source).expect("source file should be updated");

        let build_report_path = fixture.build_dir.join("build_report.json");
        let mut build_report: MetalBuildReport =
            read_json_file(&build_report_path).expect("build report should load");
        build_report.source_sha256 = sha256_hex(drifted_source.as_bytes());
        write_json_file(&build_report_path, &build_report);

        let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
            .expect_err("assets should reject source/manifest drift");
        let MetalRuntimeError::InvalidBuildReport { message } = error else {
            panic!("expected invalid build report error");
        };
        assert!(message.contains("source kernel declarations do not match manifest"));
        assert!(message.contains("kv_scale_update"));

        fixture.cleanup();
    }

    fn write_json_file<T: Serialize>(path: &Path, value: &T) {
        let json = serde_json::to_vec_pretty(value).expect("json should serialize");
        fs::write(path, json).expect("json file should write");
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

    fn fake_xcrun_script() -> &'static str {
        r#"#!/bin/sh
set -eu

while [ "$#" -gt 0 ]; do
  if [ "$1" = "--sdk" ]; then
    shift 2
    continue
  fi
  tool="$1"
  shift
  break
done

case "${tool:-}" in
  metal)
    out=""
    src=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o)
          out="$2"
          shift 2
          ;;
        -c|-O3|-Wall|-Wextra)
          shift
          ;;
        -std=*)
          shift
          ;;
        *)
          src="$1"
          shift
          ;;
      esac
    done
    cp "$src" "$out"
    ;;
  metal-ar)
    archive=""
    input=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -q)
          shift
          ;;
        *)
          if [ -z "$archive" ]; then
            archive="$1"
          else
            input="$1"
          fi
          shift
          ;;
      esac
    done
    cp "$input" "$archive"
    ;;
  metallib)
    input=""
    out=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o)
          out="$2"
          shift 2
          ;;
        *)
          input="$1"
          shift
          ;;
      esac
    done
    cp "$input" "$out"
    ;;
  *)
    echo "unexpected tool" >&2
    exit 64
    ;;
esac
"#
    }

    fn fake_failing_xcrun_script() -> &'static str {
        r#"#!/bin/sh
set -eu
echo "unexpected xcrun invocation" >&2
exit 99
"#
    }

    fn env_lock() -> &'static Mutex<()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn prepend_to_path(dir: &Path, existing: Option<&OsString>) -> OsString {
        let mut paths = vec![dir.to_path_buf()];
        if let Some(existing) = existing {
            paths.extend(std::env::split_paths(existing));
        }
        std::env::join_paths(paths).expect("PATH should join")
    }

    // -------------------------------------------------------------------------
    // Tier 1: apply_rms_norm_with_weights_in_place (no Metal device required)
    // -------------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn rms_norm_with_weights_in_place_normalizes_and_scales_by_weights() {
        let mut values = vec![3.0_f32, 4.0];
        let weights = vec![2.0_f32, 0.5];
        let epsilon = 1e-6_f32;

        apply_rms_norm_with_weights_in_place(&mut values, &weights, epsilon, 0.0)
            .expect("rms norm should succeed");

        // mean_square = (9 + 16) / 2 = 12.5
        let mean_square = (9.0_f32 + 16.0) / 2.0;
        let denom = (mean_square + epsilon).sqrt();
        let expected = vec![(3.0 / denom) * 2.0, (4.0 / denom) * 0.5];
        assert_f32_slice_close(&values, &expected, 1e-5);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rms_norm_with_weights_in_place_applies_weight_offset() {
        // weight_offset=1.0 is the Gemma style: effective_weight = weight + 1.0
        let mut values = vec![1.0_f32, 0.0];
        let weights = vec![0.0_f32, 0.0];
        let epsilon = 1e-6_f32;

        apply_rms_norm_with_weights_in_place(&mut values, &weights, epsilon, 1.0)
            .expect("rms norm with weight_offset should succeed");

        // effective weight = 0 + 1.0 = 1.0 for all; mean_square = 0.5
        let mean_square = 0.5_f32;
        let denom = (mean_square + epsilon).sqrt();
        let expected = vec![1.0 / denom, 0.0 / denom];
        assert_f32_slice_close(&values, &expected, 1e-5);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rms_norm_with_weights_in_place_rejects_mismatched_lengths() {
        let mut values = vec![1.0_f32, 2.0, 3.0];
        let weights = vec![1.0_f32, 1.0]; // wrong length

        assert!(
            apply_rms_norm_with_weights_in_place(&mut values, &weights, 1e-6, 0.0).is_none(),
            "mismatched lengths should return None"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rms_norm_with_weights_in_place_rejects_nonpositive_epsilon() {
        let mut values = vec![1.0_f32, 2.0];
        let weights = vec![1.0_f32, 1.0];

        assert!(
            apply_rms_norm_with_weights_in_place(&mut values, &weights, 0.0, 0.0).is_none(),
            "zero epsilon should return None"
        );
        assert!(
            apply_rms_norm_with_weights_in_place(&mut values, &weights, -1e-6, 0.0).is_none(),
            "negative epsilon should return None"
        );
    }

    // -------------------------------------------------------------------------
    // Tier 1: expand_grouped_kv_heads_cpu (no Metal device required)
    // -------------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_grouped_kv_heads_cpu_repeats_kv_head_to_fill_query_heads() {
        // 1 kv head, 4 q heads, head_dim=3 → each q head gets the same kv head
        let kv = vec![1.0_f32, 2.0, 3.0];
        let expanded = expand_grouped_kv_heads_cpu(&kv, 4, 1, 3)
            .expect("single kv head expansion should succeed");

        assert_eq!(
            expanded,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_grouped_kv_heads_cpu_is_identity_when_q_heads_equals_kv_heads() {
        let kv = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expanded =
            expand_grouped_kv_heads_cpu(&kv, 2, 2, 3).expect("equal-head expansion should succeed");

        assert_eq!(expanded, kv);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_grouped_kv_heads_cpu_handles_multiple_kv_heads_correctly() {
        // 2 kv heads, 4 q heads, head_dim=2
        // kv: [head0: 1,2], [head1: 3,4]
        // expected q: [1,2, 1,2, 3,4, 3,4]
        let kv = vec![1.0_f32, 2.0, 3.0, 4.0];
        let expanded = expand_grouped_kv_heads_cpu(&kv, 4, 2, 2)
            .expect("multi kv head expansion should succeed");

        assert_eq!(expanded, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_grouped_kv_heads_cpu_rejects_q_heads_not_divisible_by_kv_heads() {
        let kv = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert!(
            expand_grouped_kv_heads_cpu(&kv, 3, 2, 2).is_none(),
            "3 q_heads / 2 kv_heads is not integer, should return None"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_grouped_kv_heads_cpu_rejects_zero_head_dim() {
        let kv = vec![1.0_f32, 2.0];
        assert!(
            expand_grouped_kv_heads_cpu(&kv, 2, 1, 0).is_none(),
            "zero head_dim should return None"
        );
    }

    // -------------------------------------------------------------------------
    // Tier 2: apply_model_stage_rope_cpu (needs artifacts on disk, no GPU ops)
    // -------------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn rope_cpu_is_identity_at_position_zero_for_qwen() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
        let stage_dims = ModelStageDims {
            input_dim: 8,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        let original_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original_key = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut query = original_query.clone();
        let mut key = original_key.clone();

        apply_model_stage_rope_cpu(&artifacts, &mut query, &mut key, 0.0, stage_dims);

        // At position 0: theta = 0 for all pairs → cos(0)=1, sin(0)=0 → identity
        assert_f32_slice_close(&query, &original_query, 1e-5);
        assert_f32_slice_close(&key, &original_key, 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rope_cpu_result_matches_neox_rotate_reference_for_qwen() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
        let stage_dims = ModelStageDims {
            input_dim: 8,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        let original_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let mut query = original_query.clone();
        let mut key = original_key.clone();

        apply_model_stage_rope_cpu(&artifacts, &mut query, &mut key, 2.0, stage_dims);

        // Qwen uses NeoX style: each head independently rotated
        let expected_query: Vec<f32> = neox_rotate_reference(&original_query[..4], 2.0)
            .into_iter()
            .chain(neox_rotate_reference(&original_query[4..], 2.0))
            .collect();
        let expected_key: Vec<f32> = neox_rotate_reference(&original_key[..4], 2.0)
            .into_iter()
            .chain(neox_rotate_reference(&original_key[4..], 2.0))
            .collect();

        assert_f32_slice_close(&query, &expected_query, 1e-5);
        assert_f32_slice_close(&key, &expected_key, 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rope_cpu_result_matches_interleaved_rotate_reference_for_gemma() {
        let model_dir = write_gemma_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");
        let stage_dims = ModelStageDims {
            input_dim: 8,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        let original_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let mut query = original_query.clone();
        let mut key = original_key.clone();

        apply_model_stage_rope_cpu(&artifacts, &mut query, &mut key, 1.0, stage_dims);

        // Gemma uses Interleaved style: consecutive element-pairs rotated
        let expected_query: Vec<f32> = interleaved_rotate_reference(&original_query[..4], 1.0)
            .into_iter()
            .chain(interleaved_rotate_reference(&original_query[4..], 1.0))
            .collect();
        let expected_key: Vec<f32> = interleaved_rotate_reference(&original_key[..4], 1.0)
            .into_iter()
            .chain(interleaved_rotate_reference(&original_key[4..], 1.0))
            .collect();

        assert_f32_slice_close(&query, &expected_query, 1e-5);
        assert_f32_slice_close(&key, &expected_key, 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    // -------------------------------------------------------------------------
    // Tier 2: *_with_path fallback wiring — verify CPU is used when bringup=None
    // -------------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn rope_with_path_matches_rope_cpu_when_bringup_is_none() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
        let stage_dims = ModelStageDims {
            input_dim: 8,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        let input_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

        let mut cpu_query = input_query.clone();
        let mut cpu_key = input_key.clone();
        apply_model_stage_rope_cpu(&artifacts, &mut cpu_query, &mut cpu_key, 3.0, stage_dims);

        let mut with_path_query = input_query.clone();
        let mut with_path_key = input_key.clone();
        apply_model_stage_rope_with_path(
            &artifacts,
            &mut with_path_query,
            &mut with_path_key,
            3.0,
            stage_dims,
            None, // bringup=None → forces CPU
        );

        assert_f32_slice_close(&with_path_query, &cpu_query, 1e-6);
        assert_f32_slice_close(&with_path_key, &cpu_key, 1e-6);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_rope_with_path_matches_per_row_reference_when_bringup_is_none() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
        let stage_dims = ModelStageDims {
            input_dim: 8,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        let mut batched_query = vec![
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![0.5_f32, -1.0, 2.0, -0.5, 3.0, 1.5, -2.0, 4.0],
        ];
        let mut batched_key = vec![
            vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            vec![-0.25_f32, 0.75, -1.5, 2.5, 1.0, -3.0, 0.5, 4.5],
        ];
        let positions = vec![1_u32, 3_u32];

        let mut reference_query = batched_query.clone();
        let mut reference_key = batched_key.clone();
        for ((query, key), position) in reference_query
            .iter_mut()
            .zip(reference_key.iter_mut())
            .zip(positions.iter().copied())
        {
            apply_model_stage_rope_with_path(
                &artifacts,
                query,
                key,
                position as f32,
                stage_dims,
                None,
            );
        }

        apply_batched_model_stage_rope_with_path(
            &artifacts,
            &mut batched_query,
            &mut batched_key,
            &positions,
            stage_dims,
            None,
        )
        .expect("batched rope path should succeed");

        assert_eq!(batched_query, reference_query);
        assert_eq!(batched_key, reference_key);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gate_up_product_with_path_reports_cpu_path_and_matches_cpu_when_bringup_is_none() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");

        let original_gate = vec![1.0_f32, -1.0, 2.0, 0.5];
        let up = vec![1.0_f32, 2.0, 0.5, -1.0];

        let mut cpu_gate = original_gate.clone();
        apply_model_gate_up_product(&artifacts, &mut cpu_gate, &up);

        let mut with_path_gate = original_gate.clone();
        let used_native =
            apply_model_gate_up_product_with_path(&artifacts, &mut with_path_gate, &up, None)
                .expect("gate_up_product_with_path should succeed");

        assert!(!used_native, "should report CPU path when bringup is None");
        assert_f32_slice_close(&with_path_gate, &cpu_gate, 1e-6);

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_grouped_kv_heads_with_path_falls_back_to_cpu_when_bringup_is_none() {
        // 2 kv heads, 4 q heads, head_dim=2
        let kv = vec![1.0_f32, 2.0, 3.0, 4.0];
        let with_path = expand_grouped_kv_heads_with_path(&kv, 4, 2, 2, None)
            .expect("expand_grouped_kv_heads_with_path should succeed");
        let cpu = expand_grouped_kv_heads_cpu(&kv, 4, 2, 2)
            .expect("expand_grouped_kv_heads_cpu should succeed");

        assert_eq!(with_path, cpu);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_batched_grouped_kv_heads_with_path_matches_row_wise_cpu_reference() {
        let rows = vec![
            vec![1.0_f32, 2.0, 3.0, 4.0],
            vec![5.0_f32, 6.0, 7.0, 8.0],
            vec![9.0_f32, 10.0, 11.0, 12.0],
        ];

        let with_path = expand_batched_grouped_kv_heads_with_path(&rows, 4, 2, 2, None)
            .expect("expand_batched_grouped_kv_heads_with_path should succeed");
        let cpu_reference: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                expand_grouped_kv_heads_cpu(row, 4, 2, 2)
                    .expect("expand_grouped_kv_heads_cpu should succeed")
            })
            .collect();

        assert_eq!(with_path, cpu_reference);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn project_batched_attention_qkv_with_dims_matches_per_row_reference() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        let layer = &bindings.layers[0];
        let attention_norm = buffers
            .binding_for(&layer.attention_norm)
            .expect("attention norm binding should resolve");
        let stage_dims = resolved_model_stage_dims_for_input_width(
            &artifacts,
            attention_norm,
            &layer.attention_qkv,
            &buffers,
            8,
        )
        .expect("stage dims should resolve");
        let input_rows = vec![
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![0.5_f32, -1.0, 2.0, -0.5, 3.0, 1.5, -2.0, 4.0],
            vec![3.0_f32, 0.0, -1.0, 2.0, -4.0, 1.0, 0.5, -0.25],
        ];

        let (batched_query, batched_key, batched_value, batched_tally) =
            project_batched_attention_qkv_with_dims_and_tally(
                &artifacts,
                &layer.attention_qkv,
                &buffers,
                &input_rows,
                8,
                stage_dims,
                None,
            )
            .expect("batched attention qkv projection should succeed");

        let mut reference_query = Vec::with_capacity(input_rows.len());
        let mut reference_key = Vec::with_capacity(input_rows.len());
        let mut reference_value = Vec::with_capacity(input_rows.len());
        let mut reference_tally = PrefixAttentionExecutionTally::default();
        for input in &input_rows {
            let (query, key, value, tally) = project_attention_qkv_with_dims_and_tally(
                &artifacts,
                &layer.attention_qkv,
                &buffers,
                input,
                stage_dims,
                None,
            )
            .expect("row-wise attention qkv projection should succeed");
            reference_query.push(query);
            reference_key.push(key);
            reference_value.push(value);
            reference_tally = reference_tally.merge(tally);
        }

        assert_eq!(batched_query, reference_query);
        assert_eq!(batched_key, reference_key);
        assert_eq!(batched_value, reference_value);
        assert_eq!(
            batched_tally.cpu_projection_row_count(),
            reference_tally.cpu_projection_row_count()
        );
        assert_eq!(
            batched_tally.native_projection_row_count(),
            reference_tally.native_projection_row_count()
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn read_shared_buffer_prefix_reads_large_buffers_in_texture_sized_chunks() {
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let element_count = SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH as usize + 19;
        let values = (0..element_count)
            .map(|index| index as f32 * 0.5 - 7.0)
            .collect::<Vec<_>>();
        let buffer = new_shared_buffer_with_data(&device, &values);

        let restored = read_shared_buffer_prefix(&buffer, saturating_usize_to_u32(values.len()));

        assert_eq!(restored.len(), values.len());
        assert_f32_slice_close(&restored, &values, 1e-6);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn read_shared_u32_buffer_prefix_reads_large_buffers_in_texture_sized_chunks() {
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let element_count = SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH as usize + 23;
        let values = (0..element_count)
            .map(|index| (index as u32).wrapping_mul(17))
            .collect::<Vec<_>>();
        let buffer = new_shared_buffer_with_data(&device, &values);

        let restored =
            read_shared_u32_buffer_prefix(&buffer, saturating_usize_to_u32(values.len()));

        assert_eq!(restored, values);
    }

    // -------------------------------------------------------------------------
    // Tier 3: per_head_rms_norm_with_tally (needs Metal device for buffer)
    // -------------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn per_head_rms_norm_with_tally_result_matches_per_head_reference() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        let attention_norm = buffers
            .binding_for(&bindings.layers[0].attention_norm)
            .expect("attention norm binding should resolve");
        let head_count = 2_usize;
        let head_dim = 4_usize;
        let epsilon = native_model_rms_norm_epsilon(&artifacts);
        let weight_offset = native_model_rms_norm_weight_offset(&artifacts);
        let weights = tensor_prefix_f32(attention_norm, head_dim)
            .expect("attention norm weights should exist");

        let original: Vec<f32> = (1..=8).map(|i| i as f32).collect();

        // Reference: apply rms norm to each head slice independently
        let mut expected = original.clone();
        for head in expected.chunks_exact_mut(head_dim) {
            apply_rms_norm_with_weights_in_place(head, &weights, epsilon, weight_offset)
                .expect("per-head reference norm should succeed");
        }

        let mut actual = original.clone();
        let tally = apply_per_head_rms_norm_with_binding_in_place_with_tally(
            &mut actual,
            head_count,
            head_dim,
            attention_norm,
            epsilon,
            weight_offset,
            None, // bringup=None → CPU path
        )
        .expect("per_head_rms_norm_with_tally should succeed");

        assert_f32_slice_close(&actual, &expected, 1e-6);
        assert_eq!(tally.native_rms_norm_element_count(), 0);
        assert_eq!(
            tally.cpu_rms_norm_element_count(),
            (head_count * head_dim) as u32
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn per_head_rms_norm_with_tally_normalizes_heads_independently_not_jointly() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        let attention_norm = buffers
            .binding_for(&bindings.layers[0].attention_norm)
            .expect("attention norm binding should resolve");
        // Use uniform weights=1.0 so we can reason about the norms directly
        let head_count = 2_usize;
        let head_dim = 4_usize;
        let epsilon = 1e-6_f32;

        // head 0: [1, 1, 1, 1] — small magnitude
        // head 1: [10, 10, 10, 10] — large magnitude
        // If normalized jointly the ratio would be 1:1; if independently each head
        // normalizes to roughly unit RMS
        let mut values = vec![1.0_f32, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        apply_per_head_rms_norm_with_binding_in_place_with_tally(
            &mut values,
            head_count,
            head_dim,
            attention_norm,
            epsilon,
            0.0,
            None,
        )
        .expect("per_head_rms_norm should succeed");

        // After independent per-head normalization both heads should have ~equal RMS
        // (since uniform-magnitude heads both normalize to ~1/sqrt(head_dim))
        let rms0 = (values[..head_dim].iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
        let rms1 = (values[head_dim..].iter().map(|v| v * v).sum::<f32>() / head_dim as f32).sqrt();
        // Both heads should have RMS ≈ weights / sqrt(head_dim), not wildly different
        assert!(
            (rms0 - rms1).abs() < 0.01,
            "independently normalized heads should have similar RMS; got rms0={rms0}, rms1={rms1}"
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    // -------------------------------------------------------------------------
    // Tier 3: project_decode_logits_cpu (needs Metal device for buffer)
    // -------------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn project_decode_logits_cpu_returns_correct_argmax_token() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        // The fixture lm_head is a zero matrix ([5, 8]) so all dot products are 0.
        // Token 0 (first max) is chosen as argmax.
        let lm_head_binding = bindings.lm_head.as_ref().expect("lm_head should exist");
        let lm_head = buffers
            .binding_for(lm_head_binding)
            .expect("lm_head buffer should bind");
        let hidden = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let (logits, token_id) = project_decode_logits_cpu(lm_head, 8, &hidden)
            .expect("project_decode_logits_cpu should succeed with zero lm_head");

        assert_eq!(logits.len(), 5, "vocab_size=5 → 5 logits");
        assert!(
            logits.iter().all(|v| v.abs() < 1e-6),
            "zero lm_head → all logits should be ~0"
        );
        assert_eq!(token_id, 0, "argmax of all-zero logits should be token 0");

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn project_decode_logits_cpu_selects_highest_scoring_token() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        // The fixture attention Q matrix is an identity matrix ([8, 8]).
        // Use it as a stand-in projection to get non-zero logits.
        // With identity weights, logit[i] = hidden[i].
        // If we set hidden = [0, 0, 0, 0, 0, 0, 0, 99], logit for row i = hidden[i].
        // But vocab is only 5 rows, so logits = [hidden·row0, ..., hidden·row4].
        // Row 0 of identity = [1,0,0,...], so logit[0] = hidden[0] = 0.
        // ...hidden[7] is not reached because vocab_size=5.
        // To get a clear winner, set hidden[2]=10 so logit[2]=10 is the max.
        let layer = &bindings.layers[0];
        let attn_q_binding = match &layer.attention_qkv {
            MetalAttentionQkvBindings::Split { q, .. } => q,
            MetalAttentionQkvBindings::Packed(_) => return, // fixture uses split
        };
        let attn_q = buffers
            .binding_for(attn_q_binding)
            .expect("attn_q buffer should bind");
        // hidden[2] = 10 → row 2 of identity → logit[2] = 10, all others ≤ 5
        let hidden = vec![1.0_f32, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let (logits, token_id) = project_decode_logits_cpu(attn_q, 8, &hidden)
            .expect("project_decode_logits_cpu should succeed");

        // attn_q has 8 rows (not 5), so logits.len() = 8
        // logit[2] = dot([0,0,1,0,...], hidden) = 10.0 → argmax
        assert!(logits.len() >= 3, "should have at least 3 logits");
        assert_eq!(token_id, 2, "token 2 should win with logit 10.0");
        assert_f32_slice_close(&logits[..3], &[1.0, 2.0, 10.0], 1e-5);

        let _ = fs::remove_dir_all(model_dir);
    }

    // -------------------------------------------------------------------------
    // Tier 3: initial_model_hidden_states_cpu (needs Metal device + workload)
    // -------------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn initial_model_hidden_states_cpu_returns_one_vector_per_scheduled_token() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        let token_embedding = buffers
            .binding_for(&bindings.token_embedding)
            .expect("token_embedding binding should resolve");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve from sample runner input");

        let hidden_states = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 1.0)
            .expect("initial_model_hidden_states_cpu should succeed");

        let expected_count = workload.scheduled_token_ids.len();
        assert_eq!(
            hidden_states.len(),
            expected_count,
            "should produce one hidden state per scheduled token"
        );
        assert!(
            hidden_states.iter().all(|h| h.len() == 8),
            "each hidden state should have hidden_dim=8 elements"
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn initial_model_hidden_states_cpu_gathers_correct_embedding_rows() {
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        let token_embedding = buffers
            .binding_for(&bindings.token_embedding)
            .expect("token_embedding binding should resolve");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        // The fixture embedding rows are:
        //   row 0: [0,0,0,0,0,0,0,0]
        //   row 1: [1,2,3,4,5,6,7,8]
        //   row 2: [2,3,4,5,6,7,8,9]
        //   row 3: [3,4,5,6,7,8,9,10]
        //   row 4: [4,5,6,7,8,9,10,11]
        // sample_runner_input schedules token IDs [1,2,3,4]
        let hidden_states = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 1.0)
            .expect("initial_model_hidden_states_cpu should succeed");

        assert_f32_slice_close(
            &hidden_states[0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            1e-6,
        );
        assert_f32_slice_close(
            &hidden_states[1],
            &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            1e-6,
        );

        let _ = fs::remove_dir_all(model_dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn initial_model_hidden_states_cpu_applies_embedding_scale() {
        let model_dir = write_gemma_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        let token_embedding = buffers
            .binding_for(&bindings.token_embedding)
            .expect("token_embedding binding should resolve");
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");

        let scale_1x = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 1.0)
            .expect("scale 1x should succeed");
        let scale_2x = initial_model_hidden_states_cpu(token_embedding, &workload, 8, 2.0)
            .expect("scale 2x should succeed");

        for (unscaled, scaled) in scale_1x.iter().zip(scale_2x.iter()) {
            let expected: Vec<f32> = unscaled.iter().map(|v| v * 2.0).collect();
            assert_f32_slice_close(scaled, &expected, 1e-6);
        }

        let _ = fs::remove_dir_all(model_dir);
    }

    // -------------------------------------------------------------------------
    // Tier 4: GPU vs CPU correctness — each test compiles real Metal kernels via
    // xcrun (skipped gracefully when xcrun / Metal toolchain is unavailable) and
    // then asserts that the GPU path produces the same numerical result as the
    // corresponding CPU reference implementation.
    // -------------------------------------------------------------------------

    /// Compile the real phase1 Metal kernels from the workspace source and load
    /// a `MetalRuntimeBringup`.  Returns `None` when xcrun is unavailable, when
    /// the workspace `metal/phase1-kernels.json` manifest cannot be found, or
    /// when the Metal device is absent.  The caller receives the temp output
    /// directory and is responsible for cleaning it up.
    #[cfg(target_os = "macos")]
    fn try_compile_real_bringup() -> Option<(MetalRuntimeBringup, PathBuf)> {
        // CARGO_MANIFEST_DIR points at .../crates/ax-engine-core.
        // The workspace root is two levels up.
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir.parent()?.parent()?;
        let manifest_path = workspace_root.join("metal").join("phase1-kernels.json");
        if !manifest_path.is_file() {
            return None; // manifest not found in workspace
        }

        let output_dir = unique_test_dir("real-bringup");
        let request = MetalKernelBuildRequest {
            manifest_path,
            output_dir: output_dir.clone(),
            doctor: sample_build_doctor(true, true),
        };

        let artifacts = build_phase1_kernel_artifacts(&request).ok()?;
        if artifacts.build_status() != MetalBuildStatus::Compiled {
            let _ = fs::remove_dir_all(&output_dir);
            return None; // xcrun failed or toolchain unavailable
        }

        // Wrap Metal initialization in autoreleasepool so that internally
        // autoreleased objects created by the Metal framework during pipeline
        // compilation are drained while the bringup (and all its Metal objects)
        // is still alive.  Moving bringup out of the closure ensures it outlives
        // the pool drain, preventing double-release on thread exit.
        let bringup =
            autoreleasepool(|| MetalRuntimeBringup::from_build_dir(&artifacts.output_dir).ok())?;
        Some((bringup, output_dir))
    }

    // --- GPU test: expand_grouped_kv_heads_f32 --------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn expand_grouped_kv_heads_f32_gpu_output_matches_cpu_reference() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return; // xcrun / Metal toolchain unavailable — skip
        };

        // 2 kv heads, 4 q heads, head_dim=3
        let kv = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu = expand_grouped_kv_heads_cpu(&kv, 4, 2, 3).expect("cpu expand should succeed");
        let gpu = expand_grouped_kv_heads_with_path(&kv, 4, 2, 3, Some(&bringup))
            .expect("gpu expand should succeed");

        assert_f32_slice_close(&gpu, &cpu, 1e-5);
        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: ffn_gate_silu_product_f32 ----------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn ffn_gate_silu_product_f32_gpu_output_matches_cpu_reference() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let model_dir = write_projection_native_model_fixture(); // Qwen → silu
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");

        let gate_input = vec![1.0_f32, -1.0, 2.0, 0.5, -0.5, 3.0, -3.0, 0.0];
        let up = vec![0.5_f32, 2.0, -1.0, 1.5, 1.0, 0.25, -0.5, 1.0];

        // CPU reference
        let mut cpu_gate = gate_input.clone();
        apply_model_gate_up_product(&artifacts, &mut cpu_gate, &up);

        // GPU path
        let mut gpu_gate = gate_input.clone();
        let used_native =
            apply_model_gate_up_product_with_path(&artifacts, &mut gpu_gate, &up, Some(&bringup))
                .expect("gate_up_product_with_path should succeed");

        assert!(used_native, "GPU path should be taken when bringup is Some");
        assert_f32_slice_close(&gpu_gate, &cpu_gate, 1e-5);

        let _ = fs::remove_dir_all(&model_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: ffn_gate_gelu_approx_product_f32 ---------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn ffn_gate_gelu_approx_product_f32_gpu_output_matches_cpu_reference() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let model_dir = write_gemma_projection_native_model_fixture(); // Gemma → gelu_approx
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");

        let gate_input = vec![0.5_f32, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25];
        let up = vec![1.0_f32, 1.0, 0.5, 0.5, 2.0, 2.0, -1.0, -1.0];

        let mut cpu_gate = gate_input.clone();
        apply_model_gate_up_product(&artifacts, &mut cpu_gate, &up);

        let mut gpu_gate = gate_input.clone();
        let used_native =
            apply_model_gate_up_product_with_path(&artifacts, &mut gpu_gate, &up, Some(&bringup))
                .expect("gate_up_product_with_path should succeed");

        assert!(used_native, "GPU path should be taken when bringup is Some");
        // gelu_approx uses a polynomial approximation; tolerance is slightly wider
        assert_f32_slice_close(&gpu_gate, &cpu_gate, 1e-4);

        let _ = fs::remove_dir_all(&model_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: apply_rope_f32 (NeoX style, Qwen) --------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn apply_rope_f32_gpu_output_matches_cpu_reference_qwen() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let model_dir = write_projection_native_model_fixture(); // Qwen → NeoX
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("qwen artifacts should load");
        let stage_dims = ModelStageDims {
            input_dim: 8,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        let input_query = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input_key = vec![0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];

        // CPU reference
        let mut cpu_query = input_query.clone();
        let mut cpu_key = input_key.clone();
        apply_model_stage_rope_cpu(&artifacts, &mut cpu_query, &mut cpu_key, 3.0, stage_dims);

        // GPU path (dispatches to GPU when bringup is Some)
        let mut gpu_query = input_query.clone();
        let mut gpu_key = input_key.clone();
        apply_model_stage_rope_with_path(
            &artifacts,
            &mut gpu_query,
            &mut gpu_key,
            3.0,
            stage_dims,
            Some(&bringup),
        );

        assert_f32_slice_close(&gpu_query, &cpu_query, 1e-4);
        assert_f32_slice_close(&gpu_key, &cpu_key, 1e-4);

        let _ = fs::remove_dir_all(&model_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: apply_rope_f32 (Interleaved style, Gemma) ------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn apply_rope_f32_gpu_output_matches_cpu_reference_gemma() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let model_dir = write_gemma_projection_native_model_fixture();
        let artifacts =
            NativeModelArtifacts::from_dir(&model_dir).expect("gemma artifacts should load");
        let stage_dims = ModelStageDims {
            input_dim: 8,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
        };
        let input_query = vec![1.0_f32, 0.5, -1.0, 2.0, -2.0, 0.25, 0.75, -0.5];
        let input_key = vec![0.1_f32, -0.1, 0.9, -0.9, 1.1, -1.1, 0.3, -0.3];

        let mut cpu_query = input_query.clone();
        let mut cpu_key = input_key.clone();
        apply_model_stage_rope_cpu(&artifacts, &mut cpu_query, &mut cpu_key, 1.0, stage_dims);

        let mut gpu_query = input_query.clone();
        let mut gpu_key = input_key.clone();
        apply_model_stage_rope_with_path(
            &artifacts,
            &mut gpu_query,
            &mut gpu_key,
            1.0,
            stage_dims,
            Some(&bringup),
        );

        assert_f32_slice_close(&gpu_query, &cpu_query, 1e-4);
        assert_f32_slice_close(&gpu_key, &cpu_key, 1e-4);

        let _ = fs::remove_dir_all(&model_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: rms_norm_batched_f32 (per-head norm) -----------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn rms_norm_batched_f32_gpu_output_matches_cpu_reference() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let model_dir = write_projection_native_model_fixture();
        let artifacts = NativeModelArtifacts::from_dir(&model_dir).expect("artifacts should load");
        let bindings =
            MetalNativeModelBindings::from_artifacts(&artifacts).expect("bindings should load");
        let device = Device::system_default().expect("Metal device should exist on macOS");
        let buffers = MetalNativeModelBufferBindings::from_model_bindings(&device, &bindings)
            .expect("buffers should bind");
        let attention_norm = buffers
            .binding_for(&bindings.layers[0].attention_norm)
            .expect("attention norm binding should resolve");

        let head_count = 2_usize;
        let head_dim = 4_usize;
        let epsilon = native_model_rms_norm_epsilon(&artifacts);
        let weight_offset = native_model_rms_norm_weight_offset(&artifacts);

        let original: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();

        // CPU reference: per-head norm via scalar function
        let weights = tensor_prefix_f32(attention_norm, head_dim)
            .expect("attention norm weights should read");
        let mut cpu_out = original.clone();
        for head in cpu_out.chunks_exact_mut(head_dim) {
            apply_rms_norm_with_weights_in_place(head, &weights, epsilon, weight_offset)
                .expect("cpu per-head norm should succeed");
        }

        // GPU path via batched tally function
        let mut gpu_out = original.clone();
        let tally = apply_per_head_rms_norm_with_binding_in_place_with_tally(
            &mut gpu_out,
            head_count,
            head_dim,
            attention_norm,
            epsilon,
            weight_offset,
            Some(&bringup),
        )
        .expect("per_head_rms_norm_with_tally should succeed");

        assert_f32_slice_close(&gpu_out, &cpu_out, 1e-4);
        // When the batched GPU kernel fires, native element count is non-zero
        assert!(
            tally.native_rms_norm_element_count() > 0,
            "native element count should be > 0 when GPU path was taken"
        );

        let _ = fs::remove_dir_all(&model_dir);
        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: reshape_and_cache (required kernel) ------------------------

    /// Runs the full required-kernel dispatch and verifies that `reshape_and_cache`
    /// wrote each token's key and value vectors to the correct paged KV cache slot.
    /// Failure indicates that the `reshape_and_cache` Metal kernel is producing
    /// wrong slot assignments or corrupted values (validation stage: "key_cache").
    #[cfg(target_os = "macos")]
    #[test]
    fn reshape_and_cache_gpu_writes_kv_to_correct_cache_slots() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return; // xcrun / Metal toolchain unavailable — skip
        };
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let staged_inputs = synthetic_staged_inputs(&workload);

        let (_trace, cache_snapshot) = bringup
            .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
                &workload,
                &staged_inputs,
                None,
            )
            .expect(
                "reshape_and_cache: GPU should write key/value token vectors to the \
                 correct paged KV cache slots (numeric validation stage: key_cache)",
            );

        // Compare the GPU cache snapshot against the CPU reference.  The snapshot
        // incorporates copy_blocks destinations via merge_copy_targets so we build
        // the matching CPU snapshot the same way.
        let reference = reference_numeric_path_with_inputs(&workload, &staged_inputs);
        let reference_snapshot =
            MetalDispatchKvCacheSnapshot::from_reference_for_workload(&workload, &reference);
        assert_f32_slice_close(
            &cache_snapshot.key_cache,
            &reference_snapshot.key_cache,
            1e-5,
        );
        assert_f32_slice_close(
            &cache_snapshot.value_cache,
            &reference_snapshot.value_cache,
            1e-5,
        );

        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: paged_decode_attention (required kernel) -------------------

    /// Runs the full required-kernel dispatch and verifies that `paged_decode_attention`
    /// produces attention outputs that match the CPU softmax-attention reference.
    /// Failure indicates the attention kernel has wrong softmax scaling, causal masking,
    /// or output accumulation (validation stage: "attention_output").
    #[cfg(target_os = "macos")]
    #[test]
    fn paged_decode_attention_gpu_output_matches_cpu_softmax_reference() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let staged_inputs = synthetic_staged_inputs(&workload);

        let (trace, _cache_snapshot) = bringup
            .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
                &workload,
                &staged_inputs,
                None,
            )
            .expect(
                "paged_decode_attention: GPU attention output should match CPU softmax \
                 reference (numeric validation stage: attention_output)",
            );

        let reference = reference_numeric_path_with_inputs(&workload, &staged_inputs);
        let gpu_attn: Vec<f32> = trace
            .numeric
            .attention_output_bits
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect();
        assert_f32_slice_close(&gpu_attn, &reference.attention_output, 1e-4);

        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: gather_kv_cache (required kernel) --------------------------

    /// Runs the full required-kernel dispatch and verifies that `gather_kv_cache`
    /// assembles the correct KV slices from paged block-table metadata.
    /// Failure indicates the gather kernel is reading from wrong block offsets or
    /// producing a mismatched KV layout for attention (validation stage: "gather_kv").
    #[cfg(target_os = "macos")]
    #[test]
    fn gather_kv_cache_gpu_output_passes_validation_against_cpu_reference() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let staged_inputs = synthetic_staged_inputs(&workload);

        let (trace, _cache_snapshot) = bringup
            .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
                &workload,
                &staged_inputs,
                None,
            )
            .expect(
                "gather_kv_cache: GPU gather of block-table KV slices should match CPU \
                 reference checksum (numeric validation stage: gather_kv)",
            );

        let summary = trace
            .numeric
            .validation
            .expect("validation summary should be present on successful dispatch");
        assert_eq!(
            trace.numeric.gather_output_checksum, summary.expected_gather_output_checksum,
            "gather_kv_cache: GPU gather checksum should equal CPU reference checksum"
        );

        let _ = fs::remove_dir_all(output_dir);
    }

    // --- GPU test: copy_blocks (required kernel) ------------------------------

    /// Runs the full required-kernel dispatch and verifies that `copy_blocks`
    /// moves KV block data to the correct destination slots.
    /// Failure indicates the copy kernel is writing to wrong destinations or
    /// corrupting block contents during movement (validation stage: "copy_blocks").
    #[cfg(target_os = "macos")]
    #[test]
    fn copy_blocks_gpu_output_passes_validation_against_cpu_reference() {
        let Some((bringup, output_dir)) = try_compile_real_bringup() else {
            return;
        };
        let workload = MetalDispatchWorkload::from_runner_input(&sample_runner_input())
            .expect("workload should resolve");
        let staged_inputs = synthetic_staged_inputs(&workload);

        let (trace, _cache_snapshot) = bringup
            .dispatch_numeric_workload_ephemeral_with_cache_snapshot_and_attention_config(
                &workload,
                &staged_inputs,
                None,
            )
            .expect(
                "copy_blocks: GPU block-movement primitive should write matching KV data \
                 to copy destinations (numeric validation stage: copy_blocks)",
            );

        let summary = trace
            .numeric
            .validation
            .expect("validation summary should be present on successful dispatch");
        assert_eq!(
            trace.numeric.copy_output_checksum, summary.expected_copy_output_checksum,
            "copy_blocks: GPU copy checksum should equal CPU reference checksum"
        );

        let _ = fs::remove_dir_all(output_dir);
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        static NEXT_SUFFIX: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let suffix = NEXT_SUFFIX.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ax-engine-core-{label}-{}-{nanos}-{suffix}",
            std::process::id()
        ))
    }
}
