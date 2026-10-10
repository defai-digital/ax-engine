// The Metal bring-up layer mirrors kernel binding shapes and model execution
// contracts. Keeping these signatures explicit is safer than hiding them behind
// broad parameter bags while the native runtime is still stabilizing.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]
// Non-macOS type-checking retains the public Metal report/error types used by
// the SDK contract, but cannot construct the macOS runtime paths that consume
// their private helpers. AX Engine does not ship a Linux runtime.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

#[cfg(all(test, target_os = "macos"))]
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{NativeModelArtifactsSummary, NativeModelError};

#[cfg(target_os = "macos")]
pub(crate) mod build;
mod dispatch_types;

#[cfg(target_os = "macos")]
pub use build::*;
pub use dispatch_types::*;

pub const PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION: &str = "ax.metal.kernel_manifest.v1";
pub const PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION: &str = "ax.metal.build_report.v1";
pub const PHASE1_MLX_METAL_TARGET: &str = "apple_m2_or_newer_macos_aarch64";
pub const PHASE1_METAL_LANGUAGE_STANDARD: &str = "metal3.1";
pub const PHASE1_METAL_LIBRARY_NAME: &str = "ax_phase1_dense_path";
pub const PHASE1_METAL_BUILD_GATE: &str = "bringup_allowed";
pub(crate) const PHASE1_METAL_BLOCK_SIZE_ALIGNMENT_TOKENS: u32 = 16;
pub const PHASE1_DEFAULT_BLOCK_SIZE_TOKENS: u32 = 16;
pub const PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS: &[u32] = &[PHASE1_DEFAULT_BLOCK_SIZE_TOKENS];
pub(crate) const PHASE1_NUMERIC_HEAD_COUNT: u32 = 2;
pub(crate) const PHASE1_NUMERIC_HEAD_DIM: u32 = 4;
pub(crate) const PHASE1_REQUIRED_METAL_KERNELS: &[&str] = &[
    "reshape_and_cache",
    "paged_decode_attention",
    "gather_kv_cache",
    "copy_blocks",
];
pub(crate) const PHASE1_DEFERRED_METAL_KERNELS: &[&str] = &["swap_blocks"];
pub(crate) const PHASE1_OPTIONAL_METAL_KERNELS: &[&str] = &[
    "kv_scale_update",
    "vector_add_f32",
    "row_scale_f32",
    "row_vector_scale_f32",
    "gather_embedding_rows_f32",
    "gather_embedding_rows_f16",
    "gather_embedding_rows_bf16",
    "decode_projection_q4km",
    "decode_logits_projection_f32",
    "decode_logits_projection_f16",
    "decode_logits_projection_bf16",
    "decode_logits_projection_batched_f32",
    "decode_logits_projection_batched_f16",
    "decode_logits_projection_batched_bf16",
    "decode_logits_projection_sg_f32",
    "decode_logits_projection_sg_f16",
    "decode_logits_projection_sg_bf16",
    "decode_logits_projection_batched_sg_f16",
    "decode_logits_projection_batched_sg_bf16",
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
pub(super) const REQUIRED_TOOLCHAIN_REQUIREMENTS: &[&str] = &["xcrun metal", "xcrun metallib"];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetalDispatchNumericLayout {
    pub head_count: u32,
    pub head_dim: u32,
}

impl MetalDispatchNumericLayout {
    pub(crate) const fn new(head_count: u32, head_dim: u32) -> Self {
        Self {
            head_count,
            head_dim,
        }
    }

    pub(crate) const fn phase1_default() -> Self {
        Self::new(PHASE1_NUMERIC_HEAD_COUNT, PHASE1_NUMERIC_HEAD_DIM)
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

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
#[derive(Debug, Error)]
pub enum MetalRuntimeError {
    #[error(transparent)]
    NativeModel(#[from] NativeModelError),
    #[error("runtime error: {0}")]
    Generic(String),
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
    #[error("failed to serialize JSON file {path}: {source}")]
    SerializeJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to read build artifact {path}: {source}")]
    ReadBuildArtifact {
        path: PathBuf,
        #[source]
        source: std::io::Error,
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

#[cfg(test)]
#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
