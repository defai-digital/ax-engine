use rustc_hash::FxHashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;

use super::{Backend, KvPrecisionPolicy, RuntimePolicy};
use crate::gguf::tensor::GgmlType;
use anyhow::Context;

// v2: GpuKv is owned by LlamaModel via ModelKv, not stored in MetalOps.
use crate::model::{ModelFingerprint, config::ModelConfig};
use ax_engine_metal::profile::MatvecParams;
use ax_engine_metal::{
    AttentionDispatchConfig, AttentionKernels, DequantDispatchConfig, DequantKernels,
    ElementwiseKernels, GdnKernels, KernelProfile, KvPrecisionMode, MatmulKernels,
    MatvecProfileVariant, MetalBuffer, MetalDevice,
};

struct MatvecBenchBuffers {
    a: MetalBuffer,
    x: MetalBuffer,
    y: MetalBuffer,
}

struct BatchF16InBenchBuffers {
    a: MetalBuffer,
    b: MetalBuffer,
    c: MetalBuffer,
}

#[derive(Debug, Clone, Copy)]
struct WeightedMatvecShape {
    m: u32,
    k: u32,
    weight: u32,
}

#[derive(Debug, Clone, Copy)]
struct DecodeDispatchCache {
    short_max_attend_len: u32,
    short_dequant_dispatch: DequantDispatchConfig,
    short_attention_dispatch: AttentionDispatchConfig,
    long_dequant_dispatch: DequantDispatchConfig,
    long_attention_dispatch: AttentionDispatchConfig,
}

impl DecodeDispatchCache {
    fn for_attend_len(self, attend_len: u32) -> (DequantDispatchConfig, AttentionDispatchConfig) {
        if attend_len <= self.short_max_attend_len {
            (self.short_dequant_dispatch, self.short_attention_dispatch)
        } else {
            (self.long_dequant_dispatch, self.long_attention_dispatch)
        }
    }
}

const Q4K_DECODE_MATVEC_AUTOTUNE_CANDIDATES: [MatvecParams; 3] = [
    MatvecParams {
        threadgroup_size: 128,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Base),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 2,
        variant: Some(MatvecProfileVariant::Nr2),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Ilp4),
    },
];

const Q5K_DECODE_MATVEC_AUTOTUNE_CANDIDATES: [MatvecParams; 3] = [
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Base),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Ilp4),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 2,
        variant: Some(MatvecProfileVariant::Nr2),
    },
];

const Q6K_DECODE_MATVEC_AUTOTUNE_CANDIDATES: [MatvecParams; 3] = [
    MatvecParams {
        threadgroup_size: 128,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Base),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 2,
        variant: Some(MatvecProfileVariant::Nr2),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Ilp4),
    },
];

const Q8_0_DECODE_MATVEC_AUTOTUNE_CANDIDATES: [MatvecParams; 3] = [
    MatvecParams {
        threadgroup_size: 128,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Base),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 1,
        variant: Some(MatvecProfileVariant::Ilp4),
    },
    MatvecParams {
        threadgroup_size: 64,
        rows_per_simdgroup: 2,
        variant: Some(MatvecProfileVariant::Nr2),
    },
];

fn decode_matvec_profile_key(quant: GgmlType) -> Option<&'static str> {
    match quant {
        GgmlType::Q4K => Some("q4_k"),
        GgmlType::Q5K => Some("q5_k"),
        GgmlType::Q6K => Some("q6_k"),
        GgmlType::Q8_0 => Some("q8_0"),
        _ => None,
    }
}

fn decode_matvec_default_candidates(quant: GgmlType) -> &'static [MatvecParams] {
    match quant {
        GgmlType::Q4K => &Q4K_DECODE_MATVEC_AUTOTUNE_CANDIDATES,
        GgmlType::Q5K => &Q5K_DECODE_MATVEC_AUTOTUNE_CANDIDATES,
        GgmlType::Q6K => &Q6K_DECODE_MATVEC_AUTOTUNE_CANDIDATES,
        GgmlType::Q8_0 => &Q8_0_DECODE_MATVEC_AUTOTUNE_CANDIDATES,
        _ => &[],
    }
}

fn quant_from_profile_label(label: &str) -> Option<GgmlType> {
    let normalized = label.trim().to_ascii_uppercase().replace('-', "_");
    if normalized == "Q4K" || normalized.starts_with("Q4_K") {
        return Some(GgmlType::Q4K);
    }
    if normalized == "Q5K" || normalized.starts_with("Q5_K") {
        return Some(GgmlType::Q5K);
    }
    if normalized == "Q6K" || normalized.starts_with("Q6_K") {
        return Some(GgmlType::Q6K);
    }
    if normalized == "Q8" || normalized == "Q80" || normalized.starts_with("Q8_0") {
        return Some(GgmlType::Q8_0);
    }
    None
}

fn metal_profile_llama() -> bool {
    static LLAMA: OnceLock<bool> = OnceLock::new();
    *LLAMA.get_or_init(|| match std::env::var("AX_METAL_LLAMA_MODE") {
        Ok(v) => v.trim().eq_ignore_ascii_case("llama"),
        Err(_) => false,
    })
}

/// Whether native Q8_0 batch dequant matmul kernel is enabled.
///
/// Controlled by `AX_METAL_Q8_BATCH_NATIVE`:
/// - `0` / `false` / `off` -> disabled
/// - unset / any other value -> enabled (default)
pub fn metal_q8_batch_native_enabled() -> bool {
    RuntimePolicy::resolved_defaults().q8_batch_native_enabled()
}

/// Whether native Q8_0 batch kernel should be used for a specific shape.
///
/// Enabled by default, unless `AX_METAL_Q8_BATCH_NATIVE=0` disables it, then applies
/// optional shape gates:
/// - `AX_METAL_Q8_NATIVE_M_MIN` (default: 0)
/// - `AX_METAL_Q8_NATIVE_M_MAX` (default: u32::MAX)
/// - `AX_METAL_Q8_NATIVE_K_MIN` (default: 0)
/// - `AX_METAL_Q8_NATIVE_K_MAX` (default: u32::MAX)
pub fn metal_q8_batch_native_shape_enabled(m: u32, _n: u32, k: u32) -> bool {
    RuntimePolicy::resolved_defaults().q8_batch_native_shape_enabled(m, 1, k)
}

/// Whether fused QKV prefill matmul path is enabled.
///
/// Controlled by `AX_METAL_FUSED_QKV`:
/// - unset -> enabled (default)
/// - `0` / `false` / `off` -> disabled
pub fn metal_fused_qkv_enabled() -> bool {
    RuntimePolicy::resolved_defaults().fused_qkv_prefill_enabled()
}

/// Per-architecture override for fused QKV prefill path.
///
/// Precedence:
/// 1) `AX_METAL_FUSED_QKV_<ARCH>`
/// 2) `AX_METAL_FUSED_QKV`
/// 3) built-in default (`true`)
pub fn metal_fused_qkv_enabled_for_arch(arch: &str) -> bool {
    RuntimePolicy::for_model("default", "default", arch).fused_qkv_prefill_enabled()
}

/// Per-architecture override for experimental decode-side fused QKV path.
///
/// This is intentionally opt-in only. It gates fused decode helpers that
/// combine QKV split with downstream elementwise work, and should not change
/// default routing until benchmarked.
///
/// Precedence:
/// 1) `AX_METAL_DECODE_FUSED_QKV_<ARCH>`
/// 2) `AX_METAL_DECODE_FUSED_QKV`
/// 3) built-in default (`false`)
pub fn metal_decode_fused_qkv_enabled_for_arch(arch: &str) -> bool {
    RuntimePolicy::for_model("default", "default", arch).decode_fused_qkv_enabled()
}

/// Per-architecture override for simd-sum batch kernels.
///
/// Precedence:
/// 1) `AX_METAL_BATCH_SIMD_<ARCH>`
/// 2) `AX_METAL_BATCH_SIMD` (via `ax_engine_metal::batch_simd_enabled()`)
pub fn metal_batch_simd_enabled_for_arch(arch: &str) -> bool {
    RuntimePolicy::for_model("default", "default", arch).batch_simd_enabled()
}

/// Whether runtime kernel autotuning is enabled.
///
/// Controlled by `AX_METAL_AUTOTUNE`:
/// - `1` / `true` / `on` -> enabled
/// - unset -> disabled (default)
/// - `0` / `false` / `off` -> disabled
pub fn metal_autotune_enabled() -> bool {
    RuntimePolicy::resolved_defaults().autotune_f16in_batch_route_enabled()
}

fn parse_bool_env_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" => Some(true),
        "0" | "false" | "off" => Some(false),
        _ => None,
    }
}

fn bool_env_with_default(var: &str, default: bool) -> bool {
    std::env::var(var)
        .ok()
        .and_then(|value| parse_bool_env_flag(&value))
        .unwrap_or(default)
}

/// Whether persistent load-time autotune is enabled.
///
/// Controlled by `AX_METAL_LOAD_AUTOTUNE`:
/// - unset / `1` / `true` / `on` -> enabled (default)
/// - `0` / `false` / `off` -> disabled
fn metal_load_autotune_enabled() -> bool {
    bool_env_with_default("AX_METAL_LOAD_AUTOTUNE", true)
}

/// Whether load-time tuning should ignore the cached profile and re-run probes.
///
/// Controlled by `AX_METAL_FORCE_RETUNE` or `AX_METAL_LOAD_AUTOTUNE_FORCE`.
fn metal_force_retune_enabled() -> bool {
    bool_env_with_default("AX_METAL_FORCE_RETUNE", false)
        || bool_env_with_default("AX_METAL_LOAD_AUTOTUNE_FORCE", false)
}

fn metal_tune_cache_dir() -> PathBuf {
    if let Ok(path) = std::env::var("AX_METAL_TUNE_CACHE_DIR")
        && !path.trim().is_empty()
    {
        return PathBuf::from(path);
    }

    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join("Library")
            .join("Caches")
            .join("ax-engine")
            .join("kernel-profiles");
    }

    PathBuf::from(".ax-engine-kernel-profiles")
}

fn sanitize_cache_component(value: &str) -> String {
    let sanitized: String = value
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();
    let sanitized = sanitized.trim_matches('-');
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized.to_string()
    }
}

/// Resolve whether GPU KV cache should use f16 storage for this model.
///
/// Controlled by `AX_METAL_F16_KV_CACHE`:
/// - `1` / `true` / `on`  -> force enable
/// - `0` / `false` / `off` -> force disable
/// - `auto` or unset       -> enable when context length >= 256
pub fn metal_f16_kv_cache_enabled(context_len: usize) -> bool {
    let _ = metal_profile_llama();
    RuntimePolicy::resolved_defaults().uses_f16_gpu_kv(context_len)
}

/// Whether load-time precomputed dense f16 decode weights should be used for a model.
///
/// Controlled by `AX_PRECOMPUTE_F16` and `AX_PRECOMPUTE_F16_<ARCH>`:
/// - `on`  -> enabled for all supported models
/// - `off` -> disabled for all supported models
/// - `auto` / unset -> conservative default (currently disabled)
///
/// The explicit `on` path remains useful for A/B benchmarking, but the default
/// stays conservative until a clean, uncontended release-benchmark win is
/// established for a model family.
pub fn metal_precompute_f16_enabled_for_model(config: &ModelConfig) -> bool {
    RuntimePolicy::for_model("default", "default", &config.architecture).precompute_f16_enabled()
}

fn qwen35_gate_up_scratch_dims(
    config: &ModelConfig,
    q_dim: usize,
    inter_dim: usize,
) -> (usize, usize) {
    if !matches!(config.architecture.as_str(), "qwen35" | "qwen35moe") {
        return (inter_dim, inter_dim);
    }

    let recurrent_inner_dim = config.qwen35_ssm_inner_size.unwrap_or(0) as usize;
    let gate_dim = inter_dim.max(q_dim * 2).max(recurrent_inner_dim);
    let up_dim = inter_dim.max(q_dim).max(recurrent_inner_dim);
    (gate_dim, up_dim)
}

fn create_mmap_weight_buffer_from_bytes(device: &MetalDevice, data: &[u8]) -> MetalBuffer {
    unsafe { MetalBuffer::from_bytes_no_copy(device.device(), data) }.unwrap_or_else(|err| {
        tracing::debug!(
            len = data.len(),
            ptr = data.as_ptr() as usize,
            error = %err,
            "Metal no-copy weight alias failed; falling back to copied buffer"
        );
        MetalBuffer::from_bytes(device.device(), data)
            .expect("Failed to create Metal buffer for weight tensor")
    })
}

fn create_mmap_weight_buffer_from_f32(device: &MetalDevice, data: &[f32]) -> MetalBuffer {
    unsafe { MetalBuffer::from_slice_no_copy(device.device(), data) }.unwrap_or_else(|err| {
        tracing::debug!(
            len = data.len(),
            ptr = data.as_ptr() as usize,
            error = %err,
            "Metal no-copy f32 weight alias failed; falling back to copied buffer"
        );
        MetalBuffer::from_slice(device.device(), data)
            .expect("Failed to create Metal buffer for f32 weight")
    })
}

#[derive(Debug)]
struct ResolvedKernelProfile {
    profile: KernelProfile,
    skip_runtime_batch_route_autotune: bool,
}

fn profile_source_with_autotune(seed_source: &str) -> String {
    if seed_source.is_empty() {
        "load-autotune".to_string()
    } else if seed_source.contains("load-autotune") {
        seed_source.to_string()
    } else {
        format!("{seed_source}+load-autotune")
    }
}

/// Metal compute backend — offloads matmul to GPU.
///
/// Caches Metal buffers for weight tensors (keyed by mmap pointer address)
/// so they are created once on first access and reused for all subsequent
/// tokens. Also pre-allocates reusable IO buffers to eliminate per-call
/// Metal buffer allocation overhead.
pub struct MetalBackend {
    device: MetalDevice,
    matmul_kernels: MatmulKernels,
    dequant_kernels: DequantKernels,
    attention_kernels: AttentionKernels,
    gdn_kernels: GdnKernels,
    /// Cache of Metal buffers for dequantized f32 weight tensors.
    weight_cache: Mutex<FxHashMap<usize, MetalBuffer>>,
    /// Cache of Metal buffers for raw quantized weight data (used by fused
    /// dequant batch kernels).
    quant_weight_cache: Mutex<FxHashMap<usize, MetalBuffer>>,
    /// Pre-allocated input vector buffer (grows on demand).
    input_buf: Mutex<(Option<MetalBuffer>, usize)>,
    /// Pre-allocated output vector buffer (grows on demand).
    output_buf: Mutex<(Option<MetalBuffer>, usize)>,
    /// Pre-allocated batch input buffer for fused dequant matmul (grows on demand).
    batch_input_buf: Mutex<(Option<MetalBuffer>, usize)>,
    /// Pre-allocated batch output buffer for fused dequant matmul (grows on demand).
    batch_output_buf: Mutex<(Option<MetalBuffer>, usize)>,
    /// Cached Metal buffers for per-layer Qwen3.5 conv kernels.
    qwen35_conv_kernel_cache: Mutex<FxHashMap<usize, MetalBuffer>>,
    /// Reusable recurrent scratch buffers keyed by per-slot sequence shape.
    qwen35_recurrent_scratch_buffers:
        Mutex<FxHashMap<Qwen35RecurrentScratchBufferKey, Qwen35MetalRecurrentScratchBuffers>>,
    /// Metal GPU operations for phased forward pass dispatch.
    ops: MetalOps,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen35RecurrentBatchPerfCounters {
    pub conv_ns: u64,
    pub pack_ns: u64,
    pub gated_delta_ns: u64,
    pub unpack_ns: u64,
    pub qkv_handoff_ns: u64,
    pub qkv_handoff_layers: u64,
    pub qkv_handoff_fused_tail_layers: u64,
    pub qkv_gpu_projection_layers: u64,
    pub qkv_fast_path_eligible_layers: u64,
    pub gpu_ssm_projection_layers: u64,
    pub qkv_fast_reject_state_size_layers: u64,
    pub qkv_fast_reject_group_divisibility_layers: u64,
    pub qkv_fast_reject_missing_batch_scratches_layers: u64,
    pub qkv_fast_reject_q_capacity_layers: u64,
    pub qkv_fast_reject_k_capacity_layers: u64,
    pub qkv_fast_reject_v_capacity_layers: u64,
    pub qkv_fast_reject_gate_capacity_layers: u64,
    pub qkv_fast_reject_up_capacity_layers: u64,
    pub qkv_handoff_cpu_alias_layers: u64,
    pub qkv_handoff_slot_buffer_layers: u64,
    pub qkv_handoff_backend_carryover_layers: u64,
    pub qkv_handoff_backend_zero_init_layers: u64,
    pub qkv_handoff_cpu_materialization_layers: u64,
    pub state_batch_backend_native_layers: u64,
    pub state_batch_cpu_direct_layers: u64,
    pub state_batch_cpu_direct_materialized_from_backend_layers: u64,
    pub state_batch_cpu_gathered_layers: u64,
    pub state_batch_cpu_gathered_materialized_from_backend_layers: u64,
}

#[derive(Default)]
struct Qwen35RecurrentBatchPerfState {
    conv_ns: AtomicU64,
    pack_ns: AtomicU64,
    gated_delta_ns: AtomicU64,
    unpack_ns: AtomicU64,
    qkv_handoff_ns: AtomicU64,
    qkv_handoff_layers: AtomicU64,
    qkv_handoff_fused_tail_layers: AtomicU64,
    qkv_gpu_projection_layers: AtomicU64,
    qkv_fast_path_eligible_layers: AtomicU64,
    gpu_ssm_projection_layers: AtomicU64,
    qkv_fast_reject_state_size_layers: AtomicU64,
    qkv_fast_reject_group_divisibility_layers: AtomicU64,
    qkv_fast_reject_missing_batch_scratches_layers: AtomicU64,
    qkv_fast_reject_q_capacity_layers: AtomicU64,
    qkv_fast_reject_k_capacity_layers: AtomicU64,
    qkv_fast_reject_v_capacity_layers: AtomicU64,
    qkv_fast_reject_gate_capacity_layers: AtomicU64,
    qkv_fast_reject_up_capacity_layers: AtomicU64,
    qkv_handoff_cpu_alias_layers: AtomicU64,
    qkv_handoff_slot_buffer_layers: AtomicU64,
    qkv_handoff_backend_carryover_layers: AtomicU64,
    qkv_handoff_backend_zero_init_layers: AtomicU64,
    qkv_handoff_cpu_materialization_layers: AtomicU64,
    state_batch_backend_native_layers: AtomicU64,
    state_batch_cpu_direct_layers: AtomicU64,
    state_batch_cpu_direct_materialized_from_backend_layers: AtomicU64,
    state_batch_cpu_gathered_layers: AtomicU64,
    state_batch_cpu_gathered_materialized_from_backend_layers: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Qwen35SlotBufferSyncOutcome {
    pub(crate) used_backend_carryover: bool,
    pub(crate) used_backend_zero_init: bool,
    pub(crate) used_cpu_materialization: bool,
}

impl Qwen35SlotBufferSyncOutcome {
    pub(crate) fn note_backend_carryover(&mut self) {
        self.used_backend_carryover = true;
    }

    fn note_backend_zero_init(&mut self) {
        self.used_backend_zero_init = true;
    }

    fn note_cpu_materialization(&mut self) {
        self.used_cpu_materialization = true;
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Qwen35RecurrentSlotBufferKey {
    layer_idx: usize,
    slot_idx: usize,
    conv_state_stride: usize,
    recurrent_state_stride: usize,
}

pub(crate) struct Qwen35MetalSlotBuffers {
    pub(crate) conv_state: MetalBuffer,
    pub(crate) recurrent_state: MetalBuffer,
    pub(crate) conv_synced_generation: Option<u64>,
    pub(crate) recurrent_synced_generation: Option<u64>,
    source_kv_identity: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Qwen35RecurrentScratchBufferKey {
    tokens_per_slot: usize,
    conv_dim: usize,
    time_step_rank: usize,
    state_size: usize,
}

struct Qwen35MetalRecurrentScratchBuffers {
    input: MetalBuffer,
    conv_out: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    gate: MetalBuffer,
    beta: MetalBuffer,
    output: MetalBuffer,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Qwen35QkvHandoffScratchBufferKey {
    n_tokens: usize,
    conv_dim: usize,
    inner_size: usize,
    time_step_rank: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Qwen35BatchProjectionScratchBufferKey {
    n_tokens: usize,
    output_dims: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Qwen35BatchLogitsScratchBufferKey {
    hidden_len: usize,
    logits_len: usize,
}

pub(crate) struct Qwen35MetalQkvHandoffScratchBuffers {
    pub(crate) conv_out: MetalBuffer,
    pub(crate) k: MetalBuffer,
    pub(crate) v: MetalBuffer,
    pub(crate) alpha: MetalBuffer,
    pub(crate) beta: MetalBuffer,
    pub(crate) alpha_f16: MetalBuffer,
    pub(crate) beta_f16: MetalBuffer,
}

pub(crate) struct Qwen35MetalRecurrentProjectionScratchBuffers {
    pub(crate) qkv: MetalBuffer,
    pub(crate) z: MetalBuffer,
    pub(crate) beta: MetalBuffer,
    pub(crate) alpha: MetalBuffer,
}

pub(crate) struct Qwen35MetalBatchProjectionScratchBuffers {
    pub(crate) outputs: Vec<MetalBuffer>,
}

pub(crate) struct Qwen35MetalBatchLogitsScratchBuffers {
    pub(crate) hidden: MetalBuffer,
    pub(crate) hidden_f16: MetalBuffer,
    pub(crate) logits: MetalBuffer,
}

impl Qwen35MetalSlotBuffers {
    fn new(
        device: &MetalDevice,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            conv_state: MetalBuffer::new(
                device.device(),
                conv_state_stride * std::mem::size_of::<f32>(),
            )?,
            recurrent_state: MetalBuffer::new(
                device.device(),
                recurrent_state_stride * std::mem::size_of::<f32>(),
            )?,
            conv_synced_generation: None,
            recurrent_synced_generation: None,
            source_kv_identity: None,
        })
    }
}

impl Qwen35MetalRecurrentScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35RecurrentScratchBufferKey) -> anyhow::Result<Self> {
        let value_dim = key.time_step_rank * key.state_size;
        Ok(Self {
            input: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            conv_out: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            q: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
            k: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
            v: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
            gate: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            beta: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            output: MetalBuffer::new(
                device.device(),
                key.tokens_per_slot * value_dim * std::mem::size_of::<f32>(),
            )?,
        })
    }
}

impl Qwen35MetalBatchLogitsScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35BatchLogitsScratchBufferKey) -> anyhow::Result<Self> {
        Ok(Self {
            hidden: MetalBuffer::new(device.device(), key.hidden_len * std::mem::size_of::<f32>())?,
            hidden_f16: MetalBuffer::new(
                device.device(),
                key.hidden_len * std::mem::size_of::<half::f16>(),
            )?,
            logits: MetalBuffer::new(device.device(), key.logits_len * std::mem::size_of::<f32>())?,
        })
    }
}

impl Qwen35MetalQkvHandoffScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35QkvHandoffScratchBufferKey) -> anyhow::Result<Self> {
        Ok(Self {
            conv_out: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            k: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.inner_size * std::mem::size_of::<f32>(),
            )?,
            v: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.inner_size * std::mem::size_of::<f32>(),
            )?,
            alpha: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            beta: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            alpha_f16: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<half::f16>(),
            )?,
            beta_f16: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<half::f16>(),
            )?,
        })
    }
}

impl Qwen35MetalRecurrentProjectionScratchBuffers {
    fn new(device: &MetalDevice, key: Qwen35QkvHandoffScratchBufferKey) -> anyhow::Result<Self> {
        Ok(Self {
            qkv: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.conv_dim * std::mem::size_of::<f32>(),
            )?,
            z: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.inner_size * std::mem::size_of::<f32>(),
            )?,
            beta: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
            alpha: MetalBuffer::new(
                device.device(),
                key.n_tokens * key.time_step_rank * std::mem::size_of::<f32>(),
            )?,
        })
    }
}

impl Qwen35MetalBatchProjectionScratchBuffers {
    fn new(
        device: &MetalDevice,
        key: &Qwen35BatchProjectionScratchBufferKey,
    ) -> anyhow::Result<Self> {
        let mut outputs = Vec::with_capacity(key.output_dims.len());
        for &out_dim in &key.output_dims {
            outputs.push(MetalBuffer::new(
                device.device(),
                key.n_tokens * out_dim * std::mem::size_of::<f32>(),
            )?);
        }
        Ok(Self { outputs })
    }
}

impl MetalBackend {
    /// Initialize Metal device and compile all compute kernels.
    pub fn new() -> anyhow::Result<Self> {
        let device = MetalDevice::new()?;
        let matmul_kernels = MatmulKernels::new(&device)?;
        let dequant_kernels = DequantKernels::new(&device)?;
        let attention_kernels = AttentionKernels::new(&device)?;
        let gdn_kernels = GdnKernels::new(&device)?;
        let ops = MetalOps::with_device(&device)?;
        tracing::info!("MetalBackend initialized");
        Ok(Self {
            device,
            matmul_kernels,
            dequant_kernels,
            attention_kernels,
            gdn_kernels,
            weight_cache: Mutex::new(FxHashMap::default()),
            quant_weight_cache: Mutex::new(FxHashMap::default()),
            input_buf: Mutex::new((None, 0)),
            output_buf: Mutex::new((None, 0)),
            batch_input_buf: Mutex::new((None, 0)),
            batch_output_buf: Mutex::new((None, 0)),
            qwen35_conv_kernel_cache: Mutex::new(FxHashMap::default()),
            qwen35_recurrent_scratch_buffers: Mutex::new(FxHashMap::default()),
            ops,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_causal_conv_sequence_sync(
        &self,
        input: &MetalBuffer,
        kernel: &MetalBuffer,
        conv_state: &MetalBuffer,
        output: &MetalBuffer,
        n_tokens: u32,
        conv_cache_len: u32,
        conv_dim: u32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            conv_cache_len <= 8,
            "qwen35 causal conv Metal kernel supports conv_cache_len <= 8, got {conv_cache_len}",
        );
        self.device.execute_sync(|encoder| {
            self.gdn_kernels.encode_causal_conv_sequence(
                encoder,
                input,
                kernel,
                conv_state,
                output,
                n_tokens,
                conv_cache_len,
                conv_dim,
            );
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_gated_delta_sequence_sync(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        gate: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        output: &MetalBuffer,
        n_tokens: u32,
        n_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        self.device.execute_sync(|encoder| {
            self.gdn_kernels.encode_gated_delta_sequence(
                encoder, q, k, v, gate, beta, state, output, n_tokens, n_heads, head_dim,
            );
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_prepare_multi_token_qkv_sync(
        &self,
        conv_out: &MetalBuffer,
        alpha: &[f32],
        beta: &[f32],
        q_out: &MetalBuffer,
        k_out: &MetalBuffer,
        v_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        beta_out: &MetalBuffer,
        n_tokens: u32,
        group_count: u32,
        time_step_rank: u32,
        state_size: u32,
        eps: f32,
    ) -> anyhow::Result<bool> {
        let alpha_buf = unsafe { MetalBuffer::from_slice_no_copy(self.device.device(), alpha) }
            .unwrap_or_else(|_| {
                MetalBuffer::from_slice(self.device.device(), alpha)
                    .expect("Failed to create Metal buffer for qwen35 recurrent alpha batch")
            });
        let beta_buf = unsafe { MetalBuffer::from_slice_no_copy(self.device.device(), beta) }
            .unwrap_or_else(|_| {
                MetalBuffer::from_slice(self.device.device(), beta)
                    .expect("Failed to create Metal buffer for qwen35 recurrent beta batch")
            });
        let mut encoded = false;
        self.device.execute_sync(|encoder| {
            encoded = self.gdn_kernels.encode_prepare_multi_token_qkv(
                encoder,
                conv_out,
                &alpha_buf,
                &beta_buf,
                q_out,
                k_out,
                v_out,
                gate_out,
                beta_out,
                n_tokens,
                group_count,
                time_step_rank,
                state_size,
                eps,
            );
            Ok(())
        })?;
        Ok(encoded)
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_prepare_multi_token_qkv_with_fallback(
        &self,
        conv_out: &MetalBuffer,
        alpha: &[f32],
        beta: &[f32],
        q: &mut MetalBuffer,
        k: &mut MetalBuffer,
        v: &mut MetalBuffer,
        gate: &mut MetalBuffer,
        beta_out: &mut MetalBuffer,
        tokens_per_slot: usize,
        value_dim: usize,
        group_count: usize,
        time_step_rank: usize,
        state_size: usize,
        eps: f32,
    ) -> anyhow::Result<()> {
        let mut packed = false;
        if state_size <= 256 && time_step_rank.is_multiple_of(group_count) {
            packed = self.qwen35_prepare_multi_token_qkv_sync(
                conv_out,
                alpha,
                beta,
                q,
                k,
                v,
                gate,
                beta_out,
                tokens_per_slot as u32,
                group_count as u32,
                time_step_rank as u32,
                state_size as u32,
                eps,
            )?;
        }

        if !packed {
            let conv_out_slice = unsafe { conv_out.as_slice::<f32>() };
            unsafe {
                prepare_multi_token_gdn_bh_buffers(
                    conv_out_slice,
                    alpha,
                    beta,
                    &mut q.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                    &mut k.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                    &mut v.as_mut_slice::<f32>()[..tokens_per_slot * value_dim],
                    &mut gate.as_mut_slice::<f32>()[..tokens_per_slot * time_step_rank],
                    &mut beta_out.as_mut_slice::<f32>()[..tokens_per_slot * time_step_rank],
                    tokens_per_slot,
                    group_count,
                    time_step_rank,
                    state_size,
                    eps,
                );
            }
        }
        Ok(())
    }

    fn try_clone_qwen35_recurrent_slot_from_backend_owned(
        &self,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> bool {
        if !qwen_kv.has_recurrent_slot(src_slot_idx) {
            return false;
        }
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let kv_identity = qwen_kv as *const crate::kv::Qwen35Kv as usize;
        let mut stale_layers = Vec::new();
        {
            let slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
            for layer_idx in 0..qwen_kv.layer_count() {
                if !qwen_kv.is_recurrent_layer(layer_idx) {
                    continue;
                }
                let conv_stale = qwen_kv.conv_state_cpu_stale(src_slot_idx, layer_idx);
                let recurrent_stale = qwen_kv.recurrent_state_cpu_stale(src_slot_idx, layer_idx);
                if !conv_stale && !recurrent_stale {
                    continue;
                }
                let slot_key = Qwen35RecurrentSlotBufferKey {
                    layer_idx,
                    slot_idx: src_slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                };
                let Some(slot_buffers) = slot_cache.get(&slot_key) else {
                    return false;
                };
                if slot_buffers.source_kv_identity != Some(kv_identity) {
                    return false;
                }
                let conv_generation = qwen_kv.conv_state_generation(src_slot_idx, layer_idx);
                let recurrent_generation =
                    qwen_kv.recurrent_state_generation(src_slot_idx, layer_idx);
                if conv_stale && slot_buffers.conv_synced_generation != Some(conv_generation) {
                    return false;
                }
                if recurrent_stale
                    && slot_buffers.recurrent_synced_generation != Some(recurrent_generation)
                {
                    return false;
                }
                stale_layers.push((layer_idx, conv_stale, recurrent_stale));
            }
        }

        qwen_kv.clone_recurrent_slot_preserving_ownership(src_slot_idx, dst_slot_idx);

        if stale_layers.is_empty() {
            return true;
        }

        let mut slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        for (layer_idx, conv_stale, recurrent_stale) in stale_layers {
            let src_key = Qwen35RecurrentSlotBufferKey {
                layer_idx,
                slot_idx: src_slot_idx,
                conv_state_stride,
                recurrent_state_stride,
            };
            let (src_conv_ptr, src_recurrent_ptr) = {
                let src_buffers = slot_cache
                    .get(&src_key)
                    .expect("validated qwen35 source slot buffer should exist");
                let conv =
                    conv_stale.then(|| src_buffers.conv_state.contents().as_ptr() as *const f32);
                let recurrent = recurrent_stale
                    .then(|| src_buffers.recurrent_state.contents().as_ptr() as *const f32);
                (conv, recurrent)
            };
            let dst_key = Qwen35RecurrentSlotBufferKey {
                layer_idx,
                slot_idx: dst_slot_idx,
                conv_state_stride,
                recurrent_state_stride,
            };
            let dst_buffers = slot_cache.entry(dst_key).or_insert_with(|| {
                Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                    .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
            });
            if let Some(src_conv_ptr) = src_conv_ptr {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_conv_ptr,
                        dst_buffers.conv_state.contents().as_ptr() as *mut f32,
                        conv_state_stride,
                    );
                }
                dst_buffers.conv_synced_generation =
                    Some(qwen_kv.conv_state_generation(dst_slot_idx, layer_idx));
            }
            if let Some(src_recurrent_ptr) = src_recurrent_ptr {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_recurrent_ptr,
                        dst_buffers.recurrent_state.contents().as_ptr() as *mut f32,
                        recurrent_state_stride,
                    );
                }
                dst_buffers.recurrent_synced_generation =
                    Some(qwen_kv.recurrent_state_generation(dst_slot_idx, layer_idx));
            }
            dst_buffers.source_kv_identity = Some(kv_identity);
        }

        true
    }

    /// Get or grow the reusable input buffer. Copies `data` into it.
    /// Returns a lock guard holding the buffer.
    fn prepare_input(
        &self,
        data: &[f32],
    ) -> std::sync::MutexGuard<'_, (Option<MetalBuffer>, usize)> {
        let needed = std::mem::size_of_val(data);
        let mut guard = self.input_buf.lock().unwrap();
        if guard.1 < needed {
            guard.0 = Some(
                MetalBuffer::new(self.device.device(), needed)
                    .expect("Failed to allocate input Metal buffer"),
            );
            guard.1 = needed;
        }
        // Copy input data into the pre-allocated buffer via raw pointer
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                guard.0.as_ref().unwrap().contents().as_ptr() as *mut u8,
                needed,
            );
        }
        guard
    }

    /// Get or grow the reusable output buffer.
    fn prepare_output(
        &self,
        size_bytes: usize,
    ) -> std::sync::MutexGuard<'_, (Option<MetalBuffer>, usize)> {
        let mut guard = self.output_buf.lock().unwrap();
        if guard.1 < size_bytes {
            guard.0 = Some(
                MetalBuffer::new(self.device.device(), size_bytes)
                    .expect("Failed to allocate output Metal buffer"),
            );
            guard.1 = size_bytes;
        }
        guard
    }

    fn ensure_dense_weight_buffer(
        &self,
        a_quant: &[u8],
        dtype: GgmlType,
        m: usize,
        k: usize,
    ) -> usize {
        let key = a_quant.as_ptr() as usize;
        let mut cache = self.weight_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            if dtype == GgmlType::F32 {
                create_mmap_weight_buffer_from_bytes(&self.device, a_quant)
            } else {
                let mut a_f32 = vec![0.0f32; m * k];
                crate::quant::dequantize(dtype, a_quant, &mut a_f32);
                MetalBuffer::from_slice(self.device.device(), &a_f32)
                    .expect("Failed to create Metal buffer for dequantized weight tensor")
            }
        });
        key
    }

    fn ensure_quant_weight_cached(&self, key: usize, data: &[u8]) {
        let mut cache = self.quant_weight_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, data));
    }

    fn safe_batch_dequant_matvec_dense(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        debug_assert_eq!(ops.len(), outputs.len());
        if ops.is_empty() {
            return;
        }

        let input_guard = self.prepare_input(x);
        let buf_x = input_guard.0.as_ref().unwrap();

        let mut weight_keys = Vec::with_capacity(ops.len());
        for (a_quant, dtype, m) in ops {
            weight_keys.push(self.ensure_dense_weight_buffer(a_quant, *dtype, *m, k));
        }

        let mut output_bufs: Vec<MetalBuffer> = Vec::with_capacity(ops.len());
        for (_, _, m) in ops {
            output_bufs.push(
                MetalBuffer::new(self.device.device(), m * std::mem::size_of::<f32>())
                    .expect("Failed to allocate output Metal buffer for safe batch matvec"),
            );
        }

        let cache = self.weight_cache.lock().unwrap();
        self.device
            .execute_sync(|encoder| {
                for (i, (_, _, m)) in ops.iter().enumerate() {
                    let buf_a = cache.get(&weight_keys[i]).unwrap();
                    self.matmul_kernels.encode_matvec(
                        encoder,
                        buf_a,
                        buf_x,
                        &output_bufs[i],
                        *m as u32,
                        k as u32,
                    );
                }
                Ok(())
            })
            .expect("Metal safe batch matvec dispatch failed");
        drop(cache);

        for (i, (_, _, m)) in ops.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output_bufs[i].contents().as_ptr() as *const f32,
                    outputs[i].as_mut_ptr(),
                    *m,
                );
            }
        }
    }

    fn qwen35_recurrent_slot_buffer_key(
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> Qwen35RecurrentSlotBufferKey {
        Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        }
    }

    fn ensure_qwen35_recurrent_slot_buffers(
        &self,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) -> std::sync::MutexGuard<'_, FxHashMap<Qwen35RecurrentSlotBufferKey, Qwen35MetalSlotBuffers>>
    {
        let key = Self::qwen35_recurrent_slot_buffer_key(
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        );
        let mut cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
        });
        cache
    }

    fn qwen35_recurrent_scratch_buffer_key(
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) -> Qwen35RecurrentScratchBufferKey {
        Qwen35RecurrentScratchBufferKey {
            tokens_per_slot,
            conv_dim: cfg.conv_dim,
            time_step_rank: cfg.time_step_rank,
            state_size: cfg.state_size,
        }
    }

    fn ensure_qwen35_recurrent_scratch_buffers(
        &self,
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) -> std::sync::MutexGuard<
        '_,
        FxHashMap<Qwen35RecurrentScratchBufferKey, Qwen35MetalRecurrentScratchBuffers>,
    > {
        let key = Self::qwen35_recurrent_scratch_buffer_key(tokens_per_slot, cfg);
        let mut cache = self.qwen35_recurrent_scratch_buffers.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            Qwen35MetalRecurrentScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 recurrent Metal scratch buffers")
        });
        cache
    }

    fn get_qwen35_conv_kernel_buffer(
        &self,
        kernel: &[f32],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = kernel.as_ptr() as usize;
        let mut cache = self.qwen35_conv_kernel_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            MetalBuffer::from_slice(self.device.device(), kernel)
                .expect("Failed to create Metal buffer for qwen35 conv kernel")
        });
        cache
    }

    fn qwen35_sync_slot_buffers_from_kv(
        qwen_kv: &crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
        slot_buffers: &mut Qwen35MetalSlotBuffers,
    ) {
        let kv_identity = qwen_kv as *const crate::kv::Qwen35Kv as usize;
        if slot_buffers.source_kv_identity != Some(kv_identity) {
            slot_buffers.conv_synced_generation = None;
            slot_buffers.recurrent_synced_generation = None;
        }
        let conv_generation = qwen_kv.conv_state_generation(slot_idx, layer_idx);
        // Only copy conv_state CPU→GPU if GPU doesn't have the latest.
        if !qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx) {
            // CPU has the latest version — copy to GPU.
            if slot_buffers.conv_synced_generation != Some(conv_generation) {
                unsafe {
                    slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                        .copy_from_slice(qwen_kv.conv_state_for_slot(slot_idx, layer_idx));
                }
                slot_buffers.conv_synced_generation = Some(conv_generation);
            }
        } else if slot_buffers.conv_synced_generation != Some(conv_generation) {
            // Backend-owned but GPU slot buffer has a different generation.
            // This can happen when GPU prefill sets backend-owned state and
            // then the slot/layer is reused. Accept the GPU version as
            // authoritative and update the tracked generation.
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.conv_synced_generation,
                cpu_gen = conv_generation,
                "qwen35 conv state generation mismatch — accepting GPU version"
            );
            slot_buffers.conv_synced_generation = Some(conv_generation);
        }

        let recurrent_generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
        if !qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx) {
            if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
                unsafe {
                    slot_buffers.recurrent_state.as_mut_slice::<f32>()[..recurrent_state_stride]
                        .copy_from_slice(qwen_kv.recurrent_state_for_slot(slot_idx, layer_idx));
                }
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
            }
        } else if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.recurrent_synced_generation,
                cpu_gen = recurrent_generation,
                "qwen35 recurrent state generation mismatch — accepting GPU version"
            );
            slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
        }
        slot_buffers.source_kv_identity = Some(kv_identity);
    }

    fn sync_qwen35_slot_from_backend_if_needed(
        &self,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_idx: usize,
    ) {
        let recurrent_stale = qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx);
        let conv_stale = qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx);
        if !recurrent_stale && !conv_stale {
            return;
        }

        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let slot_key = Self::qwen35_recurrent_slot_buffer_key(
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        );
        let slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        let slot_buffers = slot_cache.get(&slot_key).unwrap_or_else(|| {
            panic!(
                "qwen35 state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer is missing"
            )
        });
        let kv_identity = qwen_kv as *const crate::kv::Qwen35Kv as usize;
        assert_eq!(
            slot_buffers.source_kv_identity,
            Some(kv_identity),
            "qwen35 state for slot {slot_idx} layer {layer_idx} is backend-owned by a different KV instance"
        );
        if recurrent_stale {
            let generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
            assert_eq!(
                slot_buffers.recurrent_synced_generation,
                Some(generation),
                "qwen35 recurrent state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer generation does not match"
            );
            let recurrent_state = unsafe {
                &slot_buffers.recurrent_state.as_slice::<f32>()[..recurrent_state_stride]
            };
            qwen_kv.sync_recurrent_state_from_backend(
                slot_idx,
                layer_idx,
                recurrent_state,
                generation,
            );
        }
        if conv_stale {
            let generation = qwen_kv.conv_state_generation(slot_idx, layer_idx);
            assert_eq!(
                slot_buffers.conv_synced_generation,
                Some(generation),
                "qwen35 conv state for slot {slot_idx} layer {layer_idx} is backend-owned but Metal slot buffer generation does not match"
            );
            let conv_state =
                unsafe { &slot_buffers.conv_state.as_slice::<f32>()[..conv_state_stride] };
            qwen_kv.sync_conv_state_from_backend(slot_idx, layer_idx, conv_state, generation);
        }
    }

    fn sync_qwen35_slots_from_backend_if_needed(
        &self,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_indices: &[usize],
    ) {
        for &slot_idx in slot_indices {
            self.sync_qwen35_slot_from_backend_if_needed(qwen_kv, layer_idx, slot_idx);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence_with_slot_buffers(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        state_batch: &mut super::Qwen35RecurrentStateBatch<'_>,
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        if tokens_per_slot == 1 || cfg.conv_cache_len > 8 {
            qwen35_recurrent_sequence_with_backend(
                self,
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                state_batch,
                output_batch,
                tokens_per_slot,
                cfg,
            );
            return;
        }

        assert!(
            tokens_per_slot > 0,
            "qwen35 recurrent sequence requires tokens_per_slot > 0"
        );
        let slot_count = state_batch.slot_count();
        let total_tokens = slot_count * tokens_per_slot;
        let value_dim = cfg.value_dim();
        let conv_state_stride = state_batch.conv_state_stride();
        let recurrent_state_stride = state_batch.recurrent_state_stride();
        assert_eq!(
            qkv_batch.len(),
            total_tokens * cfg.conv_dim,
            "qwen35 recurrent qkv batch has wrong length"
        );
        assert_eq!(
            beta_batch.len(),
            total_tokens * cfg.time_step_rank,
            "qwen35 recurrent beta batch has wrong length"
        );
        assert_eq!(
            alpha_batch.len(),
            total_tokens * cfg.time_step_rank,
            "qwen35 recurrent alpha batch has wrong length"
        );
        assert_eq!(
            output_batch.len(),
            total_tokens * value_dim,
            "qwen35 recurrent output batch has wrong length"
        );
        assert_eq!(
            conv_state_stride,
            cfg.conv_cache_len * cfg.conv_dim,
            "qwen35 recurrent conv state stride mismatch"
        );
        assert_eq!(
            recurrent_state_stride,
            cfg.time_step_rank * cfg.state_size * cfg.state_size,
            "qwen35 recurrent state stride mismatch"
        );

        crate::compute::gdn::prepare_alpha_beta(alpha_batch, beta_batch, dt_bias, a);
        let kernel_key = conv_kernel.as_ptr() as usize;
        let kernel_cache = self.get_qwen35_conv_kernel_buffer(conv_kernel);
        let buf_kernel = kernel_cache
            .get(&kernel_key)
            .expect("qwen35 conv kernel Metal buffer missing after allocation");
        let scratch_key = Self::qwen35_recurrent_scratch_buffer_key(tokens_per_slot, cfg);
        let mut scratch_cache = self.ensure_qwen35_recurrent_scratch_buffers(tokens_per_slot, cfg);
        let scratch_buffers = scratch_cache
            .get_mut(&scratch_key)
            .expect("qwen35 recurrent Metal scratch buffers missing after allocation");

        for batch_idx in 0..slot_count {
            let token_start = batch_idx * tokens_per_slot;
            let token_end = token_start + tokens_per_slot;
            let qkv_start = token_start * cfg.conv_dim;
            let qkv_end = token_end * cfg.conv_dim;
            let gate_start = token_start * cfg.time_step_rank;
            let gate_end = token_end * cfg.time_step_rank;
            let out_start = token_start * value_dim;
            let out_end = token_end * value_dim;
            let (conv_state_src, recurrent_state_src) =
                state_batch.recurrent_buffers_for_slot_mut(batch_idx);
            let conv_state_buf = unsafe {
                MetalBuffer::from_mut_slice_no_copy(self.device.device(), conv_state_src)
            }
            .expect("Failed to alias qwen35 conv state batch into Metal buffer");
            let recurrent_state_buf = unsafe {
                MetalBuffer::from_mut_slice_no_copy(self.device.device(), recurrent_state_src)
            }
            .expect("Failed to alias qwen35 recurrent state batch into Metal buffer");

            unsafe {
                scratch_buffers.input.as_mut_slice::<f32>()[..qkv_end - qkv_start]
                    .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
            }
            self.qwen35_causal_conv_sequence_sync(
                &scratch_buffers.input,
                buf_kernel,
                &conv_state_buf,
                &scratch_buffers.conv_out,
                tokens_per_slot as u32,
                cfg.conv_cache_len as u32,
                cfg.conv_dim as u32,
            )
            .expect("Metal qwen35 recurrent causal conv dispatch failed");

            if tokens_per_slot == 1 {
                let conv_out = unsafe { scratch_buffers.conv_out.as_slice::<f32>() };
                unsafe {
                    let q_dst = &mut scratch_buffers.q.as_mut_slice::<f32>()[..value_dim];
                    let k_dst = &mut scratch_buffers.k.as_mut_slice::<f32>()[..value_dim];
                    let v_dst = &mut scratch_buffers.v.as_mut_slice::<f32>()[..value_dim];
                    prepare_single_token_gdn_qkv(
                        &conv_out[..cfg.conv_dim],
                        q_dst,
                        k_dst,
                        v_dst,
                        cfg.group_count,
                        cfg.time_step_rank,
                        cfg.state_size,
                        cfg.rms_norm_eps,
                    );
                    scratch_buffers.gate.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&alpha_batch[gate_start..gate_end]);
                    scratch_buffers.beta.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&beta_batch[gate_start..gate_end]);
                }
            } else {
                self.qwen35_prepare_multi_token_qkv_with_fallback(
                    &scratch_buffers.conv_out,
                    &alpha_batch[gate_start..gate_end],
                    &beta_batch[gate_start..gate_end],
                    &mut scratch_buffers.q,
                    &mut scratch_buffers.k,
                    &mut scratch_buffers.v,
                    &mut scratch_buffers.gate,
                    &mut scratch_buffers.beta,
                    tokens_per_slot,
                    value_dim,
                    cfg.group_count,
                    cfg.time_step_rank,
                    cfg.state_size,
                    cfg.rms_norm_eps,
                )
                .expect("Metal qwen35 recurrent batch pack dispatch failed");
            }
            self.qwen35_gated_delta_sequence_sync(
                &scratch_buffers.q,
                &scratch_buffers.k,
                &scratch_buffers.v,
                &scratch_buffers.gate,
                &scratch_buffers.beta,
                &recurrent_state_buf,
                &scratch_buffers.output,
                tokens_per_slot as u32,
                cfg.time_step_rank as u32,
                cfg.state_size as u32,
            )
            .expect("Metal qwen35 recurrent gated-delta dispatch failed");

            let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
            if tokens_per_slot == 1 {
                output_batch[out_start..out_end].copy_from_slice(&out_bhsv[..value_dim]);
            } else {
                let output_slice = &mut output_batch[out_start..out_end];
                if let Ok(output_buf) = unsafe {
                    MetalBuffer::from_mut_slice_no_copy(self.device.device(), output_slice)
                } {
                    self.gdn_kernels
                        .unpack_bhsk_to_token_major(
                            &self.device,
                            &scratch_buffers.output,
                            &output_buf,
                            tokens_per_slot as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                        )
                        .expect("Metal qwen35 recurrent batch unpack dispatch failed");
                } else {
                    unpack_bhsk_to_token_major(
                        out_bhsv,
                        output_slice,
                        tokens_per_slot,
                        cfg.time_step_rank,
                        cfg.state_size,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence_for_kv_with_slot_buffers(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_indices: &[usize],
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        assert!(
            tokens_per_slot > 0,
            "qwen35 recurrent sequence requires tokens_per_slot > 0"
        );
        let slot_count = slot_indices.len();
        let total_tokens = slot_count * tokens_per_slot;
        let value_dim = cfg.value_dim();
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        assert_eq!(
            qkv_batch.len(),
            total_tokens * cfg.conv_dim,
            "qwen35 recurrent qkv batch has wrong length"
        );
        assert_eq!(
            beta_batch.len(),
            total_tokens * cfg.time_step_rank,
            "qwen35 recurrent beta batch has wrong length"
        );
        assert_eq!(
            alpha_batch.len(),
            total_tokens * cfg.time_step_rank,
            "qwen35 recurrent alpha batch has wrong length"
        );
        assert_eq!(
            output_batch.len(),
            total_tokens * value_dim,
            "qwen35 recurrent output batch has wrong length"
        );

        crate::compute::gdn::prepare_alpha_beta(alpha_batch, beta_batch, dt_bias, a);
        let kernel_key = conv_kernel.as_ptr() as usize;
        let kernel_cache = self.get_qwen35_conv_kernel_buffer(conv_kernel);
        let buf_kernel = kernel_cache
            .get(&kernel_key)
            .expect("qwen35 conv kernel Metal buffer missing after allocation");
        let scratch_key = Self::qwen35_recurrent_scratch_buffer_key(tokens_per_slot, cfg);
        let mut scratch_cache = self.ensure_qwen35_recurrent_scratch_buffers(tokens_per_slot, cfg);
        let scratch_buffers = scratch_cache
            .get_mut(&scratch_key)
            .expect("qwen35 recurrent Metal scratch buffers missing after allocation");

        for (batch_idx, &slot_idx) in slot_indices.iter().enumerate() {
            let token_start = batch_idx * tokens_per_slot;
            let token_end = token_start + tokens_per_slot;
            let qkv_start = token_start * cfg.conv_dim;
            let qkv_end = token_end * cfg.conv_dim;
            let gate_start = token_start * cfg.time_step_rank;
            let gate_end = token_end * cfg.time_step_rank;
            let out_start = token_start * value_dim;
            let out_end = token_end * value_dim;

            if tokens_per_slot == 1 {
                let mut slot_cache = self.ensure_qwen35_recurrent_slot_buffers(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_key = Self::qwen35_recurrent_slot_buffer_key(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_buffers = slot_cache
                    .get_mut(&slot_key)
                    .expect("qwen35 recurrent Metal slot buffers missing after allocation");

                // Sync state CPU→GPU only when GPU is stale (first token after
                // clear/restore/snapshot). On subsequent tokens both conv and
                // recurrent state are already resident on GPU from the previous
                // decode step.
                Self::qwen35_sync_slot_buffers_from_kv(
                    qwen_kv,
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                    slot_buffers,
                );

                // GPU-resident single-token decode: conv1d + QKV prep + GDN
                // all run on GPU in one command buffer, eliminating CPU↔GPU
                // sync overhead for conv state (~4.2 MB per layer).

                // Copy input QKV to GPU scratch buffer.
                unsafe {
                    scratch_buffers.input.as_mut_slice::<f32>()[..cfg.conv_dim]
                        .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
                }

                // Copy gate/beta to GPU scratch buffers.
                unsafe {
                    scratch_buffers.gate.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&alpha_batch[gate_start..gate_end]);
                    scratch_buffers.beta.as_mut_slice::<f32>()[..cfg.time_step_rank]
                        .copy_from_slice(&beta_batch[gate_start..gate_end]);
                }

                // Single CB: conv1d → QKV prep → GDN, all GPU-resident.
                self.device
                    .execute_sync(|encoder| {
                        // 1. GPU causal conv1d (updates conv_state in-place on GPU).
                        self.gdn_kernels.encode_causal_conv_sequence(
                            encoder,
                            &scratch_buffers.input,
                            buf_kernel,
                            &slot_buffers.conv_state,
                            &scratch_buffers.conv_out,
                            tokens_per_slot as u32,
                            cfg.conv_cache_len as u32,
                            cfg.conv_dim as u32,
                        );

                        // 2. GPU QKV preparation (norm + head split).
                        self.gdn_kernels.encode_prepare_single_token_qkv(
                            encoder,
                            &scratch_buffers.conv_out,
                            &scratch_buffers.q,
                            &scratch_buffers.k,
                            &scratch_buffers.v,
                            cfg.group_count as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                            cfg.rms_norm_eps,
                        );

                        // 3. GPU gated delta net (updates recurrent_state in-place).
                        self.gdn_kernels.encode_gated_delta_sequence(
                            encoder,
                            &scratch_buffers.q,
                            &scratch_buffers.k,
                            &scratch_buffers.v,
                            &scratch_buffers.gate,
                            &scratch_buffers.beta,
                            &slot_buffers.recurrent_state,
                            &scratch_buffers.output,
                            tokens_per_slot as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                        );
                        Ok(())
                    })
                    .expect("Metal qwen35 GPU-resident recurrent decode failed");

                // Read back only the output (16 KB), not conv/recurrent state.
                let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                output_batch[out_start..out_end].copy_from_slice(&out_bhsv[..value_dim]);

                // Mark both conv and recurrent state as backend-owned
                // (GPU has the latest — no CPU copy needed until slot
                // clear/restore/snapshot).
                let conv_generation = qwen_kv.note_backend_conv_state_update(slot_idx, layer_idx);
                slot_buffers.conv_synced_generation = Some(conv_generation);
                let backend_generation =
                    qwen_kv.note_backend_recurrent_state_update(slot_idx, layer_idx);
                slot_buffers.recurrent_synced_generation = Some(backend_generation);
                slot_buffers.source_kv_identity =
                    Some(qwen_kv as *const crate::kv::Qwen35Kv as usize);
            } else {
                let mut slot_cache = self.ensure_qwen35_recurrent_slot_buffers(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_key = Self::qwen35_recurrent_slot_buffer_key(
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                );
                let slot_buffers = slot_cache
                    .get_mut(&slot_key)
                    .expect("qwen35 recurrent Metal slot buffers missing after allocation");

                // Keep the recurrent path device-primary for multi-token
                // prefill. Only reload CPU state when the host owns a fresher
                // version; otherwise keep using the cached backend buffers.
                Self::qwen35_sync_slot_buffers_from_kv(
                    qwen_kv,
                    layer_idx,
                    slot_idx,
                    conv_state_stride,
                    recurrent_state_stride,
                    slot_buffers,
                );
                unsafe {
                    scratch_buffers.input.as_mut_slice::<f32>()[..qkv_end - qkv_start]
                        .copy_from_slice(&qkv_batch[qkv_start..qkv_end]);
                }

                let conv_t = Instant::now();
                self.qwen35_causal_conv_sequence_sync(
                    &scratch_buffers.input,
                    buf_kernel,
                    &slot_buffers.conv_state,
                    &scratch_buffers.conv_out,
                    tokens_per_slot as u32,
                    cfg.conv_cache_len as u32,
                    cfg.conv_dim as u32,
                )
                .expect("Metal qwen35 recurrent causal conv dispatch failed");
                self.ops
                    .record_qwen35_recurrent_batch_conv(conv_t.elapsed());

                let pack_t = Instant::now();
                self.qwen35_prepare_multi_token_qkv_with_fallback(
                    &scratch_buffers.conv_out,
                    &alpha_batch[gate_start..gate_end],
                    &beta_batch[gate_start..gate_end],
                    &mut scratch_buffers.q,
                    &mut scratch_buffers.k,
                    &mut scratch_buffers.v,
                    &mut scratch_buffers.gate,
                    &mut scratch_buffers.beta,
                    tokens_per_slot,
                    value_dim,
                    cfg.group_count,
                    cfg.time_step_rank,
                    cfg.state_size,
                    cfg.rms_norm_eps,
                )
                .expect("Metal qwen35 recurrent batch pack dispatch failed");
                self.ops
                    .record_qwen35_recurrent_batch_pack(pack_t.elapsed());
                let gated_delta_t = Instant::now();
                self.qwen35_gated_delta_sequence_sync(
                    &scratch_buffers.q,
                    &scratch_buffers.k,
                    &scratch_buffers.v,
                    &scratch_buffers.gate,
                    &scratch_buffers.beta,
                    &slot_buffers.recurrent_state,
                    &scratch_buffers.output,
                    tokens_per_slot as u32,
                    cfg.time_step_rank as u32,
                    cfg.state_size as u32,
                )
                .expect("Metal qwen35 recurrent gated-delta dispatch failed");
                self.ops
                    .record_qwen35_recurrent_batch_gated_delta(gated_delta_t.elapsed());

                let out_bhsv = unsafe { scratch_buffers.output.as_slice::<f32>() };
                let unpack_t = Instant::now();
                let output_slice = &mut output_batch[out_start..out_end];
                if let Ok(output_buf) = unsafe {
                    MetalBuffer::from_mut_slice_no_copy(self.device.device(), output_slice)
                } {
                    self.gdn_kernels
                        .unpack_bhsk_to_token_major(
                            &self.device,
                            &scratch_buffers.output,
                            &output_buf,
                            tokens_per_slot as u32,
                            cfg.time_step_rank as u32,
                            cfg.state_size as u32,
                        )
                        .expect("Metal qwen35 recurrent batch unpack dispatch failed");
                } else {
                    unpack_bhsk_to_token_major(
                        out_bhsv,
                        output_slice,
                        tokens_per_slot,
                        cfg.time_step_rank,
                        cfg.state_size,
                    );
                }
                self.ops
                    .record_qwen35_recurrent_batch_unpack(unpack_t.elapsed());

                let conv_generation = qwen_kv.note_backend_conv_state_update(slot_idx, layer_idx);
                let recurrent_generation =
                    qwen_kv.note_backend_recurrent_state_update(slot_idx, layer_idx);
                slot_buffers.conv_synced_generation = Some(conv_generation);
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
                slot_buffers.source_kv_identity =
                    Some(qwen_kv as *const crate::kv::Qwen35Kv as usize);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen35_recurrent_sequence_with_backend(
    backend: &dyn Backend,
    qkv_batch: &[f32],
    beta_batch: &mut [f32],
    alpha_batch: &mut [f32],
    dt_bias: &[f32],
    a: &[f32],
    conv_kernel: &[f32],
    state_batch: &mut super::Qwen35RecurrentStateBatch<'_>,
    output_batch: &mut [f32],
    tokens_per_slot: usize,
    cfg: crate::compute::gdn::Qwen35RecurrentConfig,
) {
    super::qwen35_recurrent_sequence_via_backend(
        backend,
        qkv_batch,
        beta_batch,
        alpha_batch,
        dt_bias,
        a,
        conv_kernel,
        state_batch,
        output_batch,
        tokens_per_slot,
        cfg,
    );
}

impl Backend for MetalBackend {
    fn note_qwen35_prepared_state_batch_kind(
        &self,
        kind: crate::kv::qwen35_kv::Qwen35PreparedRecurrentStateBatchKind,
    ) {
        match kind {
            crate::kv::qwen35_kv::Qwen35PreparedRecurrentStateBatchKind::CpuDirect => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_direct();
            }
            crate::kv::qwen35_kv::Qwen35PreparedRecurrentStateBatchKind::CpuDirectMaterializedFromBackend => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_direct_materialized_from_backend();
            }
            crate::kv::qwen35_kv::Qwen35PreparedRecurrentStateBatchKind::CpuGathered => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_gathered();
            }
            crate::kv::qwen35_kv::Qwen35PreparedRecurrentStateBatchKind::CpuGatheredMaterializedFromBackend => {
                self.ops
                    .record_qwen35_recurrent_batch_state_batch_cpu_gathered_materialized_from_backend();
            }
        }
    }

    fn configure_for_fingerprint(&self, fingerprint: &ModelFingerprint) -> anyhow::Result<()> {
        let resolved = self.ops.resolve_kernel_profile_for_fingerprint(fingerprint);
        self.ops
            .apply_runtime_policy(RuntimePolicy::from_kernel_profile_for_arch(
                resolved.profile,
                &fingerprint.architecture,
            ));
        self.ops.set_runtime_f16in_autotune_quant(
            self.ops
                .preferred_batch_prefill_quant(fingerprint)
                .unwrap_or(GgmlType::Q4K),
        );
        self.ops.f16in_route_tuned.store(
            resolved.skip_runtime_batch_route_autotune,
            Ordering::Relaxed,
        );
        Ok(())
    }

    fn configure_for_model(
        &self,
        model_name: &str,
        quant: &str,
        architecture: &str,
    ) -> anyhow::Result<()> {
        self.ops
            .apply_runtime_policy(RuntimePolicy::for_model(model_name, quant, architecture));
        self.ops.set_runtime_f16in_autotune_quant(
            quant_from_profile_label(quant).unwrap_or(GgmlType::Q4K),
        );
        self.ops.f16in_route_tuned.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn runtime_policy(&self) -> Option<RuntimePolicy> {
        Some(self.ops.runtime_policy())
    }

    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        let buf_a = MetalBuffer::from_slice(self.device.device(), a)
            .expect("Failed to create Metal buffer for A");
        let buf_b = MetalBuffer::from_slice(self.device.device(), b)
            .expect("Failed to create Metal buffer for B");
        let buf_c = MetalBuffer::new(self.device.device(), m * n * std::mem::size_of::<f32>())
            .expect("Failed to create Metal buffer for C");

        self.matmul_kernels
            .matmul(
                &self.device,
                &buf_a,
                &buf_b,
                &buf_c,
                (m as u32, n as u32, k as u32),
            )
            .expect("Metal matmul dispatch failed");

        let result = unsafe { buf_c.as_slice::<f32>() };
        c[..m * n].copy_from_slice(&result[..m * n]);
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant_matmul(
        &self,
        a_quant: &[u8],
        dtype: GgmlType,
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        // Fused dequant+matvec for decode (N=1) with supported quant types
        if n == 1 {
            match dtype {
                GgmlType::Q8_0 => {
                    self.fused_matvec_q8_0(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q4K => {
                    self.fused_matvec_q4_k(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q5K => {
                    self.fused_matvec_q5_k(a_quant, b, c, m, k);
                    return;
                }
                GgmlType::Q6K => {
                    self.fused_matvec_q6_k(a_quant, b, c, m, k);
                    return;
                }
                _ => {} // fall through to dequant + matmul
            }
        }

        // Fallback: dequant on CPU, matmul on GPU.
        // Cache the dequantized f32 weights for reuse across tokens.
        let key = a_quant.as_ptr() as usize;
        let mut cache = self.weight_cache.lock().unwrap();
        let buf_a = cache.entry(key).or_insert_with(|| {
            let mut a_f32 = vec![0.0f32; m * k];
            crate::quant::dequantize(dtype, a_quant, &mut a_f32);
            MetalBuffer::from_slice(self.device.device(), &a_f32)
                .expect("Failed to create Metal buffer for dequantized weights")
        });

        let buf_b = MetalBuffer::from_slice(self.device.device(), b)
            .expect("Failed to create Metal buffer for B");
        let buf_c = MetalBuffer::new(self.device.device(), m * n * std::mem::size_of::<f32>())
            .expect("Failed to create Metal buffer for C");

        self.matmul_kernels
            .matmul(
                &self.device,
                buf_a,
                &buf_b,
                &buf_c,
                (m as u32, n as u32, k as u32),
            )
            .expect("Metal matmul dispatch failed");

        drop(cache);

        let result = unsafe { buf_c.as_slice::<f32>() };
        c[..m * n].copy_from_slice(&result[..m * n]);
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant_matmul_token_major(
        &self,
        a_quant: &[u8],
        dtype: GgmlType,
        input_token_major: &[f32],
        output_token_major: &mut [f32],
        n_tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) {
        if n_tokens <= 1 {
            // For single-token, use the standard decode path (n=1 fused matvec).
            self.dequant_matmul(
                a_quant,
                dtype,
                input_token_major,
                output_token_major,
                out_dim,
                n_tokens,
                in_dim,
            );
            return;
        }

        // Check whether fused dequant batch kernels support this dtype.
        let has_fused = matches!(dtype, GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K);
        if !has_fused {
            // Fall back to default trait implementation (CPU transpose path).
            let mut input_t = vec![0.0f32; n_tokens * in_dim];
            for row in 0..n_tokens {
                for col in 0..in_dim {
                    input_t[col * n_tokens + row] = input_token_major[row * in_dim + col];
                }
            }
            let mut output_mn = vec![0.0f32; out_dim * n_tokens];
            self.dequant_matmul(
                a_quant,
                dtype,
                &input_t,
                &mut output_mn,
                out_dim,
                n_tokens,
                in_dim,
            );
            for row in 0..out_dim {
                for col in 0..n_tokens {
                    output_token_major[col * out_dim + row] = output_mn[row * n_tokens + col];
                }
            }
            return;
        }

        // For multi-token (prefill), use fused dequant batch kernels that
        // natively accept token-major B[N×K] and produce C[N×M].
        // This avoids the CPU transpose + f32 dequant path.
        let weight_key = a_quant.as_ptr() as usize;
        self.ensure_quant_weight_cached(weight_key, a_quant);

        let input_bytes = n_tokens * in_dim * std::mem::size_of::<f32>();
        let output_bytes = n_tokens * out_dim * std::mem::size_of::<f32>();

        // Reuse pre-allocated batch buffers (grow on demand).
        let mut input_guard = self.batch_input_buf.lock().unwrap();
        if input_guard.1 < input_bytes {
            input_guard.0 = Some(
                MetalBuffer::new(self.device.device(), input_bytes)
                    .expect("Failed to allocate batch input buffer"),
            );
            input_guard.1 = input_bytes;
        }
        let buf_b = input_guard.0.as_mut().unwrap();
        unsafe {
            buf_b.as_mut_slice::<f32>()[..n_tokens * in_dim].copy_from_slice(input_token_major);
        }

        let mut output_guard = self.batch_output_buf.lock().unwrap();
        if output_guard.1 < output_bytes {
            output_guard.0 = Some(
                MetalBuffer::new(self.device.device(), output_bytes)
                    .expect("Failed to allocate batch output buffer"),
            );
            output_guard.1 = output_bytes;
        }
        let buf_c = output_guard.0.as_ref().unwrap();

        let m = out_dim as u32;
        let n = n_tokens as u32;
        let k = in_dim as u32;

        {
            let cache = self.quant_weight_cache.lock().unwrap();
            let weight_buf = cache
                .get(&weight_key)
                .expect("quant weight buffer missing after ensure");

            self.device
                .execute_sync(|encoder| {
                    match dtype {
                        GgmlType::Q4K => {
                            self.dequant_kernels.encode_fused_batch_q4_k(
                                encoder, weight_buf, buf_b, buf_c, m, n, k,
                            );
                        }
                        GgmlType::Q5K => {
                            self.dequant_kernels.encode_fused_batch_q5_k(
                                encoder, weight_buf, buf_b, buf_c, m, n, k,
                            );
                        }
                        GgmlType::Q6K => {
                            self.dequant_kernels.encode_fused_batch_q6_k(
                                encoder, weight_buf, buf_b, buf_c, m, n, k,
                            );
                        }
                        _ => unreachable!("checked above"),
                    }
                    Ok(())
                })
                .expect("Metal fused batch matmul dispatch failed");
        }

        let result = unsafe { buf_c.as_slice::<f32>() };
        output_token_major[..n_tokens * out_dim].copy_from_slice(&result[..n_tokens * out_dim]);
        drop(output_guard);
        drop(input_guard);
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_dequant_matvec(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        debug_assert_eq!(ops.len(), outputs.len());
        if ops.is_empty() {
            return;
        }

        let can_batch = ops.len() >= 2
            && ops.iter().all(|(_, dtype, _)| {
                matches!(
                    dtype,
                    GgmlType::F32 | GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
                )
            });

        if !can_batch {
            // Fall back to sequential dispatch
            for (i, (a_quant, dtype, m)) in ops.iter().enumerate() {
                self.dequant_matmul(a_quant, *dtype, x, outputs[i], *m, 1, k);
            }
            return;
        }

        // Prepare shared input buffer
        let input_guard = self.prepare_input(x);
        let buf_x = input_guard.0.as_ref().unwrap();

        // Prepare weight buffers (get or create cached buffers for each weight matrix)
        let mut weight_keys: Vec<usize> = Vec::with_capacity(ops.len());
        for (a_quant, _, _) in ops {
            let key = a_quant.as_ptr() as usize;
            weight_keys.push(key);
            // Ensure weight buffer exists in cache
            let mut cache = self.weight_cache.lock().unwrap();
            cache
                .entry(key)
                .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, a_quant));
        }

        // Allocate output buffers for each operation
        let mut output_bufs: Vec<MetalBuffer> = Vec::with_capacity(ops.len());
        for (_, _, m) in ops {
            let y_bytes = m * std::mem::size_of::<f32>();
            output_bufs.push(
                MetalBuffer::new(self.device.device(), y_bytes)
                    .expect("Failed to allocate output Metal buffer for batch"),
            );
        }

        // Single command buffer for all dispatches
        let cache = self.weight_cache.lock().unwrap();
        let dispatch_config = self.ops.dequant_dispatch_config();
        self.device
            .execute_sync(|encoder| {
                for (i, (_, dtype, m)) in ops.iter().enumerate() {
                    let buf_a = cache.get(&weight_keys[i]).unwrap();
                    match dtype {
                        GgmlType::F32 => {
                            self.matmul_kernels.encode_matvec(
                                encoder,
                                buf_a,
                                buf_x,
                                &output_bufs[i],
                                *m as u32,
                                k as u32,
                            );
                        }
                        GgmlType::Q8_0 => {
                            self.dequant_kernels.encode_fused_matvec_q8_0(
                                encoder,
                                buf_a,
                                buf_x,
                                &output_bufs[i],
                                *m as u32,
                                k as u32,
                            );
                        }
                        GgmlType::Q4K => {
                            self.dequant_kernels.encode_fused_matvec_q4_k_with_config(
                                encoder,
                                buf_a,
                                buf_x,
                                &output_bufs[i],
                                *m as u32,
                                k as u32,
                                dispatch_config,
                            );
                        }
                        GgmlType::Q5K => {
                            self.dequant_kernels.encode_fused_matvec_q5_k_with_config(
                                encoder,
                                buf_a,
                                buf_x,
                                &output_bufs[i],
                                *m as u32,
                                k as u32,
                                dispatch_config,
                            );
                        }
                        GgmlType::Q6K => {
                            self.dequant_kernels.encode_fused_matvec_q6_k_with_config(
                                encoder,
                                buf_a,
                                buf_x,
                                &output_bufs[i],
                                *m as u32,
                                k as u32,
                                dispatch_config,
                            );
                        }
                        _ => unreachable!("unsupported dtype in batch_dequant_matvec"),
                    }
                }
                Ok(())
            })
            .expect("Metal batch matvec dispatch failed");
        drop(cache);

        // Copy results back
        for (i, (_, _, m)) in ops.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output_bufs[i].contents().as_ptr() as *const f32,
                    outputs[i].as_mut_ptr(),
                    *m,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn safe_batch_dequant_matvec(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        self.safe_batch_dequant_matvec_dense(ops, x, k, outputs);
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        let q_size = n_tokens * n_heads * head_dim;
        let kv_size = n_tokens * n_kv_heads * head_dim;
        let o_size = q_size;

        let buf_q = MetalBuffer::from_slice(self.device.device(), &q[..q_size])
            .expect("Failed to create Metal buffer for Q");
        let buf_k = MetalBuffer::from_slice(self.device.device(), &k[..kv_size])
            .expect("Failed to create Metal buffer for K");
        let buf_v = MetalBuffer::from_slice(self.device.device(), &v[..kv_size])
            .expect("Failed to create Metal buffer for V");
        let buf_o = MetalBuffer::new(self.device.device(), o_size * std::mem::size_of::<f32>())
            .expect("Failed to create Metal buffer for O");

        self.attention_kernels
            .attention_prefill_with_config(
                &self.device,
                &buf_q,
                &buf_k,
                &buf_v,
                &buf_o,
                n_tokens as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                self.ops.attention_dispatch_config(),
            )
            .expect("Metal attention prefill dispatch failed");

        let result = unsafe { buf_o.as_slice::<f32>() };
        output[..o_size].copy_from_slice(&result[..o_size]);
    }

    fn qwen35_causal_conv_sequence(
        &self,
        input_batch: &[f32],
        kernel: &[f32],
        conv_state: &mut [f32],
        output_batch: &mut [f32],
        n_tokens: usize,
        conv_cache_len: usize,
        conv_dim: usize,
    ) {
        if n_tokens == 1 || conv_cache_len > 8 {
            crate::compute::gdn::depthwise_conv1d_sequence(
                input_batch,
                kernel,
                conv_state,
                output_batch,
                n_tokens,
                conv_cache_len,
                conv_dim,
            );
            return;
        }

        let buf_input = MetalBuffer::from_slice(self.device.device(), input_batch)
            .expect("Failed to create Metal buffer for qwen35 causal conv input");
        let buf_kernel = MetalBuffer::from_slice(self.device.device(), kernel)
            .expect("Failed to create Metal buffer for qwen35 causal conv kernel");
        let buf_state = MetalBuffer::from_slice(self.device.device(), conv_state)
            .expect("Failed to create Metal buffer for qwen35 causal conv state");
        let buf_output =
            MetalBuffer::new(self.device.device(), std::mem::size_of_val(output_batch))
                .expect("Failed to create Metal buffer for qwen35 causal conv output");

        self.qwen35_causal_conv_sequence_sync(
            &buf_input,
            &buf_kernel,
            &buf_state,
            &buf_output,
            n_tokens as u32,
            conv_cache_len as u32,
            conv_dim as u32,
        )
        .expect("Metal qwen35 causal conv dispatch failed");

        let output = unsafe { buf_output.as_slice::<f32>() };
        output_batch.copy_from_slice(&output[..output_batch.len()]);
        let state_out = unsafe { buf_state.as_slice::<f32>() };
        conv_state.copy_from_slice(&state_out[..conv_state.len()]);
    }

    fn qwen35_gated_delta_sequence(
        &self,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        gate_batch: &[f32],
        beta_batch: &[f32],
        recurrent_state: &mut [f32],
        output_batch: &mut [f32],
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        let q_bhsk = pack_token_major_to_bhsk(q_batch, n_tokens, n_heads, head_dim);
        let k_bhsk = pack_token_major_to_bhsk(k_batch, n_tokens, n_heads, head_dim);
        let v_bhsv = pack_token_major_to_bhsk(v_batch, n_tokens, n_heads, head_dim);
        let g_bhs = pack_token_major_scalars_to_bhs(gate_batch, n_tokens, n_heads);
        let beta_bhs = pack_token_major_scalars_to_bhs(beta_batch, n_tokens, n_heads);

        let buf_q = MetalBuffer::from_slice(self.device.device(), &q_bhsk)
            .expect("Failed to create Metal buffer for qwen35 q");
        let buf_k = MetalBuffer::from_slice(self.device.device(), &k_bhsk)
            .expect("Failed to create Metal buffer for qwen35 k");
        let buf_v = MetalBuffer::from_slice(self.device.device(), &v_bhsv)
            .expect("Failed to create Metal buffer for qwen35 v");
        let buf_g = MetalBuffer::from_slice(self.device.device(), &g_bhs)
            .expect("Failed to create Metal buffer for qwen35 gate");
        let buf_beta = MetalBuffer::from_slice(self.device.device(), &beta_bhs)
            .expect("Failed to create Metal buffer for qwen35 beta");
        let buf_state = MetalBuffer::from_slice(self.device.device(), recurrent_state)
            .expect("Failed to create Metal buffer for qwen35 recurrent state");
        let buf_out = MetalBuffer::new(
            self.device.device(),
            n_tokens * n_heads * head_dim * std::mem::size_of::<f32>(),
        )
        .expect("Failed to create Metal buffer for qwen35 recurrent output");

        self.qwen35_gated_delta_sequence_sync(
            &buf_q,
            &buf_k,
            &buf_v,
            &buf_g,
            &buf_beta,
            &buf_state,
            &buf_out,
            n_tokens as u32,
            n_heads as u32,
            head_dim as u32,
        )
        .expect("Metal qwen35 gated delta dispatch failed");

        let out_bhsv = unsafe { buf_out.as_slice::<f32>() };
        unpack_bhsk_to_token_major(out_bhsv, output_batch, n_tokens, n_heads, head_dim);
        let state_out = unsafe { buf_state.as_slice::<f32>() };
        recurrent_state.copy_from_slice(&state_out[..recurrent_state.len()]);
    }

    fn qwen35_recurrent_sequence(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        state_batch: &mut super::Qwen35RecurrentStateBatch<'_>,
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        self.qwen35_recurrent_sequence_with_slot_buffers(
            qkv_batch,
            beta_batch,
            alpha_batch,
            dt_bias,
            a,
            conv_kernel,
            state_batch,
            output_batch,
            tokens_per_slot,
            cfg,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence_for_kv(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_indices: &[usize],
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        qwen_kv.assert_valid_recurrent_slot_batch(slot_indices, layer_idx);
        if cfg.conv_cache_len <= 8 {
            self.ops
                .record_qwen35_recurrent_batch_state_batch_backend_native();
            self.qwen35_recurrent_sequence_for_kv_with_slot_buffers(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                qwen_kv,
                layer_idx,
                slot_indices,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        } else {
            self.sync_qwen35_slots_from_backend_if_needed(qwen_kv, layer_idx, slot_indices);
            let mut prepared_state_batch =
                qwen_kv.prepare_recurrent_state_batch(slot_indices, layer_idx);
            let mut state_batch = prepared_state_batch.state_batch();
            self.qwen35_recurrent_sequence(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                &mut state_batch,
                output_batch,
                tokens_per_slot,
                cfg,
            );
            prepared_state_batch.finish();
        }
    }

    fn sync_qwen35_kv(&self, qwen_kv: &mut crate::kv::Qwen35Kv) {
        let slot_cache = self.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let kv_identity = qwen_kv as *const crate::kv::Qwen35Kv as usize;
        for (slot_key, slot_buffers) in slot_cache.iter() {
            if !qwen_kv.has_recurrent_slot(slot_key.slot_idx) {
                continue;
            }
            if !qwen_kv.has_layer(slot_key.layer_idx) {
                continue;
            }
            if slot_key.conv_state_stride != conv_state_stride
                || slot_key.recurrent_state_stride != recurrent_state_stride
            {
                continue;
            }
            if slot_buffers.source_kv_identity != Some(kv_identity) {
                continue;
            }
            // Sync recurrent state GPU → CPU if backend-owned.
            if let Some(generation) = slot_buffers.recurrent_synced_generation
                && qwen_kv.recurrent_state_cpu_stale(slot_key.slot_idx, slot_key.layer_idx)
                && qwen_kv.recurrent_state_generation(slot_key.slot_idx, slot_key.layer_idx)
                    == generation
            {
                let recurrent_state = unsafe {
                    &slot_buffers.recurrent_state.as_slice::<f32>()
                        [..slot_key.recurrent_state_stride]
                };
                qwen_kv.sync_recurrent_state_from_backend(
                    slot_key.slot_idx,
                    slot_key.layer_idx,
                    recurrent_state,
                    generation,
                );
            }
            // Sync conv state GPU → CPU if backend-owned.
            if let Some(generation) = slot_buffers.conv_synced_generation
                && qwen_kv.conv_state_cpu_stale(slot_key.slot_idx, slot_key.layer_idx)
                && qwen_kv.conv_state_generation(slot_key.slot_idx, slot_key.layer_idx)
                    == generation
            {
                let conv_state = unsafe {
                    &slot_buffers.conv_state.as_slice::<f32>()[..slot_key.conv_state_stride]
                };
                qwen_kv.sync_conv_state_from_backend(
                    slot_key.slot_idx,
                    slot_key.layer_idx,
                    conv_state,
                    generation,
                );
            }
        }
        qwen_kv.sync_attention_cpu_from_gpu_if_needed();
    }

    fn try_clone_qwen35_recurrent_slot(
        &self,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> bool {
        self.try_clone_qwen35_recurrent_slot_from_backend_owned(qwen_kv, src_slot_idx, dst_slot_idx)
    }

    fn metal_ops(&self) -> Option<&MetalOps> {
        Some(&self.ops)
    }

    fn use_gpu_decode(&self) -> bool {
        true
    }
}

fn pack_token_major_to_bhsk(
    input: &[f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    if n_tokens == 1 {
        return input.to_vec();
    }
    let mut out = vec![0.0f32; input.len()];
    for token_idx in 0..n_tokens {
        for head in 0..n_heads {
            let src = token_idx * n_heads * head_dim + head * head_dim;
            let dst = head * n_tokens * head_dim + token_idx * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
    out
}

fn pack_token_major_scalars_to_bhs(input: &[f32], n_tokens: usize, n_heads: usize) -> Vec<f32> {
    if n_tokens == 1 {
        return input.to_vec();
    }
    let mut out = vec![0.0f32; input.len()];
    for token_idx in 0..n_tokens {
        for head in 0..n_heads {
            let src = token_idx * n_heads + head;
            let dst = head * n_tokens + token_idx;
            out[dst] = input[src];
        }
    }
    out
}

pub(crate) fn unpack_bhsk_to_token_major(
    input: &[f32],
    output: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
) {
    if n_tokens == 1 {
        output.copy_from_slice(input);
        return;
    }
    for head in 0..n_heads {
        for token_idx in 0..n_tokens {
            let src = head * n_tokens * head_dim + token_idx * head_dim;
            let dst = token_idx * n_heads * head_dim + head * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_multi_token_gdn_bh_buffers(
    conv_out_batch: &[f32],
    alpha_batch: &[f32],
    beta_batch: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    gate_out: &mut [f32],
    beta_out: &mut [f32],
    n_tokens: usize,
    group_count: usize,
    time_step_rank: usize,
    state_size: usize,
    rms_norm_eps: f32,
) {
    let key_dim = group_count * state_size;
    let value_dim = time_step_rank * state_size;
    let conv_dim = 2 * key_dim + value_dim;
    assert_eq!(
        conv_out_batch.len(),
        n_tokens * conv_dim,
        "qwen35 recurrent conv output batch has wrong length"
    );
    assert_eq!(
        alpha_batch.len(),
        n_tokens * time_step_rank,
        "qwen35 recurrent alpha batch has wrong length"
    );
    assert_eq!(
        beta_batch.len(),
        n_tokens * time_step_rank,
        "qwen35 recurrent beta batch has wrong length"
    );
    assert_eq!(
        q_out.len(),
        n_tokens * value_dim,
        "qwen35 recurrent q buffer has wrong length"
    );
    assert_eq!(
        k_out.len(),
        n_tokens * value_dim,
        "qwen35 recurrent k buffer has wrong length"
    );
    assert_eq!(
        v_out.len(),
        n_tokens * value_dim,
        "qwen35 recurrent v buffer has wrong length"
    );
    assert_eq!(
        gate_out.len(),
        n_tokens * time_step_rank,
        "qwen35 recurrent gate buffer has wrong length"
    );
    assert_eq!(
        beta_out.len(),
        n_tokens * time_step_rank,
        "qwen35 recurrent beta buffer has wrong length"
    );
    assert!(
        time_step_rank.is_multiple_of(group_count),
        "qwen35 recurrent head expansion requires dst heads to be divisible by src heads"
    );

    let repeats = time_step_rank / group_count;

    for token_idx in 0..n_tokens {
        let conv_start = token_idx * conv_dim;
        let conv_end = conv_start + conv_dim;
        let conv_token = &conv_out_batch[conv_start..conv_end];

        for src_head in 0..group_count {
            let src_start = src_head * state_size;
            let k_src_start = key_dim + src_start;
            let mut q_sum_sq = 0.0f32;
            let mut k_sum_sq = 0.0f32;
            for lane in 0..state_size {
                let q = conv_token[src_start + lane];
                let k = conv_token[k_src_start + lane];
                q_sum_sq = q.mul_add(q, q_sum_sq);
                k_sum_sq = k.mul_add(k, k_sum_sq);
            }
            let q_inv = (q_sum_sq + rms_norm_eps).sqrt().recip();
            let k_inv = (k_sum_sq + rms_norm_eps).sqrt().recip();
            for rep in 0..repeats {
                let dst_head = src_head * repeats + rep;
                let dst_start = dst_head * n_tokens * state_size + token_idx * state_size;
                for lane in 0..state_size {
                    q_out[dst_start + lane] = conv_token[src_start + lane] * q_inv;
                    k_out[dst_start + lane] = conv_token[k_src_start + lane] * k_inv;
                }
            }
        }

        for head in 0..time_step_rank {
            let src_start = 2 * key_dim + head * state_size;
            let src_end = src_start + state_size;
            let dst_start = head * n_tokens * state_size + token_idx * state_size;
            let dst_end = dst_start + state_size;
            v_out[dst_start..dst_end].copy_from_slice(&conv_token[src_start..src_end]);
            gate_out[head * n_tokens + token_idx] = alpha_batch[token_idx * time_step_rank + head];
            beta_out[head * n_tokens + token_idx] = beta_batch[token_idx * time_step_rank + head];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_single_token_gdn_qkv(
    conv_token: &[f32],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    group_count: usize,
    time_step_rank: usize,
    state_size: usize,
    rms_norm_eps: f32,
) {
    let key_dim = group_count * state_size;
    let value_dim = time_step_rank * state_size;
    assert_eq!(
        conv_token.len(),
        2 * key_dim + value_dim,
        "qwen35 single-token recurrent conv slice has wrong length"
    );
    assert_eq!(
        q_out.len(),
        value_dim,
        "qwen35 single-token recurrent q output has wrong length"
    );
    assert_eq!(
        k_out.len(),
        value_dim,
        "qwen35 single-token recurrent k output has wrong length"
    );
    assert_eq!(
        v_out.len(),
        value_dim,
        "qwen35 single-token recurrent v output has wrong length"
    );
    assert!(
        time_step_rank.is_multiple_of(group_count),
        "qwen35 recurrent head expansion requires dst heads to be divisible by src heads"
    );

    let q_src = &conv_token[..key_dim];
    let k_src = &conv_token[key_dim..2 * key_dim];
    let v_src = &conv_token[2 * key_dim..];
    let repeats = time_step_rank / group_count;

    for src_head in 0..group_count {
        let src_start = src_head * state_size;
        let src_end = src_start + state_size;
        let q_head = &q_src[src_start..src_end];
        let k_head = &k_src[src_start..src_end];
        let q_sum_sq = q_head.iter().map(|v| v * v).sum::<f32>();
        let k_sum_sq = k_head.iter().map(|v| v * v).sum::<f32>();
        let q_inv = 1.0 / (q_sum_sq + rms_norm_eps).sqrt();
        let k_inv = 1.0 / (k_sum_sq + rms_norm_eps).sqrt();

        for rep in 0..repeats {
            let dst_head = src_head * repeats + rep;
            let dst_start = dst_head * state_size;
            let dst_end = dst_start + state_size;
            let q_dst = &mut q_out[dst_start..dst_end];
            let k_dst = &mut k_out[dst_start..dst_end];
            for idx in 0..state_size {
                q_dst[idx] = q_head[idx] * q_inv;
                k_dst[idx] = k_head[idx] * k_inv;
            }
        }
    }

    v_out.copy_from_slice(v_src);
}

impl MetalBackend {
    /// Get or create a cached Metal buffer for weight data.
    ///
    /// Weight tensors from mmap'd GGUF files have stable pointer addresses,
    /// so we use the data pointer as the cache key. On first access the data
    /// is aliased via a no-copy Metal shared-mode buffer when possible;
    /// otherwise it falls back to a copied buffer. Subsequent calls return
    /// the cached buffer with zero overhead.
    fn get_weight_buffer(
        &self,
        data: &[u8],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = data.as_ptr() as usize;
        let mut cache = self.weight_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, data));
        cache
    }

    fn fused_matvec_q8_0(&self, a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
        let y_bytes = m * std::mem::size_of::<f32>();
        let input_guard = self.prepare_input(x);
        let output_guard = self.prepare_output(y_bytes);

        let key = a_quant.as_ptr() as usize;
        let cache = self.get_weight_buffer(a_quant);
        let buf_a = cache.get(&key).unwrap();

        self.dequant_kernels
            .fused_matvec_q8_0(
                &self.device,
                buf_a,
                input_guard.0.as_ref().unwrap(),
                output_guard.0.as_ref().unwrap(),
                m as u32,
                k as u32,
            )
            .expect("Metal fused Q8_0 matvec failed");

        drop(cache);

        unsafe {
            std::ptr::copy_nonoverlapping(
                output_guard.0.as_ref().unwrap().contents().as_ptr() as *const f32,
                y.as_mut_ptr(),
                m,
            );
        }
    }

    fn fused_matvec_q4_k(&self, a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
        let y_bytes = m * std::mem::size_of::<f32>();
        let input_guard = self.prepare_input(x);
        let output_guard = self.prepare_output(y_bytes);

        let key = a_quant.as_ptr() as usize;
        let cache = self.get_weight_buffer(a_quant);
        let buf_a = cache.get(&key).unwrap();

        self.dequant_kernels
            .fused_matvec_q4_k_with_config(
                &self.device,
                buf_a,
                input_guard.0.as_ref().unwrap(),
                output_guard.0.as_ref().unwrap(),
                m as u32,
                k as u32,
                self.ops.dequant_dispatch_config(),
            )
            .expect("Metal fused Q4_K matvec failed");

        drop(cache);

        // Read back output via raw pointer
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_guard.0.as_ref().unwrap().contents().as_ptr() as *const f32,
                y.as_mut_ptr(),
                m,
            );
        }
    }

    fn fused_matvec_q5_k(&self, a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
        let y_bytes = m * std::mem::size_of::<f32>();
        let input_guard = self.prepare_input(x);
        let output_guard = self.prepare_output(y_bytes);

        let key = a_quant.as_ptr() as usize;
        let cache = self.get_weight_buffer(a_quant);
        let buf_a = cache.get(&key).unwrap();

        self.dequant_kernels
            .fused_matvec_q5_k_with_config(
                &self.device,
                buf_a,
                input_guard.0.as_ref().unwrap(),
                output_guard.0.as_ref().unwrap(),
                m as u32,
                k as u32,
                self.ops.dequant_dispatch_config(),
            )
            .expect("Metal fused Q5_K matvec failed");

        drop(cache);

        unsafe {
            std::ptr::copy_nonoverlapping(
                output_guard.0.as_ref().unwrap().contents().as_ptr() as *const f32,
                y.as_mut_ptr(),
                m,
            );
        }
    }

    fn fused_matvec_q6_k(&self, a_quant: &[u8], x: &[f32], y: &mut [f32], m: usize, k: usize) {
        let y_bytes = m * std::mem::size_of::<f32>();
        let input_guard = self.prepare_input(x);
        let output_guard = self.prepare_output(y_bytes);

        let key = a_quant.as_ptr() as usize;
        let cache = self.get_weight_buffer(a_quant);
        let buf_a = cache.get(&key).unwrap();

        self.dequant_kernels
            .fused_matvec_q6_k_with_config(
                &self.device,
                buf_a,
                input_guard.0.as_ref().unwrap(),
                output_guard.0.as_ref().unwrap(),
                m as u32,
                k as u32,
                self.ops.dequant_dispatch_config(),
            )
            .expect("Metal fused Q6_K matvec failed");

        drop(cache);

        // Read back output via raw pointer
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_guard.0.as_ref().unwrap().contents().as_ptr() as *const f32,
                y.as_mut_ptr(),
                m,
            );
        }
    }
}

/// Pre-computed weight cache keys for one transformer layer.
///
/// Keys are mmap pointer addresses (`usize`), matching the key space of `f32_weight_cache`.
/// Built once on first forward call and reused on all subsequent tokens,
/// eliminating per-token `format!()` + `HashMap::get()` overhead.
pub struct CachedLayerKeys {
    pub attn_norm: usize,
    pub wq: usize,
    pub wq_dtype: GgmlType,
    pub wk: usize,
    pub wk_dtype: GgmlType,
    pub wv: usize,
    pub wv_dtype: GgmlType,
    pub wo: usize,
    pub wo_dtype: GgmlType,
    pub ffn_norm: usize,
    pub wg: usize,
    pub wg_dtype: GgmlType,
    pub wu: usize,
    pub wu_dtype: GgmlType,
    pub wd: usize,
    pub wd_dtype: GgmlType,
    // Gemma3-specific
    pub attn_q_norm: Option<usize>,
    pub attn_k_norm: Option<usize>,
    pub post_attn_norm: Option<usize>,
    pub post_ffn_norm: Option<usize>,
    // Qwen3-specific
    pub q_bias: Option<usize>,
    pub k_bias: Option<usize>,
    pub v_bias: Option<usize>,
    // StarCoder2-specific: FFN and output projection biases
    pub wo_bias: Option<usize>,
    pub gate_bias: Option<usize>,
    pub up_bias: Option<usize>,
    pub down_bias: Option<usize>,
    // MoE-specific: router and per-expert weights
    /// Router weight key (dim × n_expert). None for dense models.
    pub moe_router: Option<usize>,
    pub moe_router_dtype: Option<GgmlType>,
    /// Per-expert gate weights [n_expert], each cached separately.
    pub moe_expert_gate: Option<Vec<usize>>,
    /// Per-expert up weights [n_expert].
    pub moe_expert_up: Option<Vec<usize>>,
    /// Per-expert down weights [n_expert].
    pub moe_expert_down: Option<Vec<usize>>,
    /// Quant dtype shared by all experts in this layer.
    pub moe_expert_dtype: Option<GgmlType>,
    /// Shared expert gate/up/down keys (Qwen3 MoE specific).
    pub moe_shared_gate: Option<usize>,
    pub moe_shared_up: Option<usize>,
    pub moe_shared_down: Option<usize>,
    pub moe_shared_dtype: Option<GgmlType>,
}

/// Pre-computed weight cache keys for the full model.
pub struct CachedModelKeys {
    pub layers: Vec<CachedLayerKeys>,
    pub output_norm: usize,
    pub lm_head: usize,
    pub lm_head_dtype: GgmlType,
}

/// GPU-resident scratch buffers for phased forward pass dispatch.
///
/// These buffers live in Metal shared memory (UMA) and persist across layers.
/// The CPU reads/writes via `contents()` pointer (zero-copy on Apple Silicon).
/// GPU dispatches use the same buffers directly.
pub struct GpuScratchBuffers {
    pub hidden: MetalBuffer,
    pub norm_buf: MetalBuffer,
    /// f16 staging input for decode-side paired FFN matvecs and future fused decode paths.
    pub matmul_in_f16: MetalBuffer,
    /// Fused QKV projection output [q_dim + 2 * kv_dim] for single-token decode.
    pub qkv_buf: MetalBuffer,
    pub q_buf: MetalBuffer,
    pub k_buf: MetalBuffer,
    pub v_buf: MetalBuffer,
    pub attn_out: MetalBuffer,
    pub proj_buf: MetalBuffer,
    pub gate_buf: MetalBuffer,
    pub up_buf: MetalBuffer,
    pub down_buf: MetalBuffer,
    /// LM head output buffer [vocab_size] for folding the final matmul
    /// into the main command buffer (eliminates second GPU submission).
    pub logits_buf: MetalBuffer,
    /// Split-K decode scratch: partial weighted-V accumulators
    /// [max_chunks × n_heads × head_dim].
    pub splitk_partial_out: MetalBuffer,
    /// Split-K decode scratch: partial log-sum-exp terms [max_chunks × n_heads].
    pub splitk_partial_lse: MetalBuffer,
    /// GPU-side argmax result: winning index (1 × u32).
    pub argmax_idx: MetalBuffer,
    /// GPU-side argmax result: winning value (1 × f32).
    pub argmax_val: MetalBuffer,
}

/// GPU-resident batch scratch buffers for prefill (N-token capacity).
///
/// Only Q, K, V, attn_out, and hidden need [N × dim] capacity.
/// Per-token temporaries (norm_buf, gate, up, down, proj) reuse
/// the single-token `GpuScratchBuffers`.
pub struct GpuBatchScratchBuffers {
    pub hidden: MetalBuffer,
    /// Fused QKV projection output [N × (q_dim + 2 * kv_dim)].
    pub qkv_buf: MetalBuffer,
    pub q_buf: MetalBuffer,
    pub k_buf: MetalBuffer,
    pub v_buf: MetalBuffer,
    pub attn_out: MetalBuffer,
    /// Batch norm buffer [N × dim] for batched matmul input.
    pub norm_buf: MetalBuffer,
    /// Batch projection buffer [N × dim] for batched matmul output.
    pub proj_buf: MetalBuffer,
    /// Batch gate buffer [N × inter_dim] for FFN batched matmul.
    pub gate_buf: MetalBuffer,
    /// Batch up buffer [N × inter_dim] for FFN batched matmul.
    pub up_buf: MetalBuffer,
    /// f16 staging input for batched dequant matmul (size: N × max_input_dim).
    pub matmul_in_f16: MetalBuffer,
    /// f16 staging output buffer (kept for compatibility with staged paths).
    pub matmul_out_f16: MetalBuffer,
    pub batch_size: usize,
    // ── MoE scratch buffers (allocated when n_expert > 0) ──
    /// Norm output for MoE FFN [N × dim].
    pub moe_norm: Option<MetalBuffer>,
    /// Router logits [N × n_expert].
    pub moe_router_out: Option<MetalBuffer>,
    /// Expert accumulator [N × dim].
    pub moe_accum: Option<MetalBuffer>,
    /// Gate matmul output [N × n_expert_used × expert_inter_dim].
    pub moe_gate_out: Option<MetalBuffer>,
    /// Up matmul output [N × n_expert_used × expert_inter_dim].
    pub moe_up_out: Option<MetalBuffer>,
    /// Down matmul output [N × n_expert_used × dim].
    pub moe_down_out: Option<MetalBuffer>,
    /// Tokens per expert [n_expert] u32.
    pub moe_tpe: Option<MetalBuffer>,
    /// Routing index [n_expert × N + (n_expert + 1)] i32/u32.
    pub moe_hids: Option<MetalBuffer>,
    /// Expert IDs [N × n_expert_used] i32.
    pub moe_expert_ids: Option<MetalBuffer>,
    /// Expert weights [N × n_expert_used] f32.
    pub moe_expert_weights: Option<MetalBuffer>,
    /// Active expert list [n_expert] u32.
    pub moe_active_experts: Option<MetalBuffer>,
    /// Token index for scatter [N × n_expert_used] u32.
    pub moe_tok_idx: Option<MetalBuffer>,
}

#[derive(Clone, Copy)]
pub(crate) struct MoeBatchScratchView {
    norm: *const MetalBuffer,
    router: *const MetalBuffer,
    accum: *const MetalBuffer,
    gate_out: *const MetalBuffer,
    up_out: *const MetalBuffer,
    down_out: *const MetalBuffer,
    tpe: *const MetalBuffer,
    hids: *const MetalBuffer,
    active: *mut MetalBuffer,
    ids: *mut MetalBuffer,
    weights: *mut MetalBuffer,
    tok_idx: *mut MetalBuffer,
}

impl MoeBatchScratchView {
    pub(crate) fn from_batch_scratches(bs: &GpuBatchScratchBuffers) -> anyhow::Result<Self> {
        Ok(Self {
            norm: bs
                .moe_norm
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_norm"))?
                as *const _,
            router: bs
                .moe_router_out
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_router_out"))?
                as *const _,
            accum: bs
                .moe_accum
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_accum"))?
                as *const _,
            gate_out: bs
                .moe_gate_out
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_gate_out"))?
                as *const _,
            up_out: bs
                .moe_up_out
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_up_out"))?
                as *const _,
            down_out: bs
                .moe_down_out
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_down_out"))?
                as *const _,
            tpe: bs
                .moe_tpe
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_tpe"))?
                as *const _,
            hids: bs
                .moe_hids
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_hids"))?
                as *const _,
            active: bs.moe_active_experts.as_ref().ok_or_else(|| {
                anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_active_experts")
            })? as *const _ as *mut _,
            ids: bs
                .moe_expert_ids
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_expert_ids"))?
                as *const _ as *mut _,
            weights: bs.moe_expert_weights.as_ref().ok_or_else(|| {
                anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_expert_weights")
            })? as *const _ as *mut _,
            tok_idx: bs
                .moe_tok_idx
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing moe_tok_idx"))?
                as *const _ as *mut _,
        })
    }
}

/// Metal GPU operations for phased forward pass dispatch.
///
/// Provides elementwise kernels (RMSNorm, RoPE, GELU/SiLU, add) and
/// dequant kernels that can be encoded into shared command buffers.
/// Also manages scratch buffer allocation and f32 weight caching.
pub struct MetalOps {
    pub device: MetalDevice,
    pub elementwise: ElementwiseKernels,
    pub dequant: DequantKernels,
    pub attention: AttentionKernels,
    pub matmul: MatmulKernels,
    pub gdn: GdnKernels,
    /// Stable device capability key used for autotune cache partitioning.
    device_cache_key: String,
    /// Backend-local runtime policy for this model/backend instance.
    runtime_policy: RwLock<RuntimePolicy>,
    /// Precomputed short/long decode dispatch selections when decode regimes are enabled.
    decode_dispatch_cache: RwLock<Option<DecodeDispatchCache>>,
    /// Cache of f32 weight tensors as MetalBuffers (norm weights, bias vectors).
    f32_weight_cache: Mutex<FxHashMap<usize, MetalBuffer>>,
    /// Cache of fused QKV quantized buffers keyed by (wq_ptr, wk_ptr, wv_ptr).
    fused_qkv_weight_cache: Mutex<FxHashMap<(usize, usize, usize), MetalBuffer>>,
    /// Cache of precomputed dense f16 weights keyed by quant buffer contents ptr.
    precomputed_f16_weight_cache: Mutex<FxHashMap<usize, MetalBuffer>>,
    /// GPU scratch buffers, allocated lazily on first forward pass.
    scratches: Mutex<Option<GpuScratchBuffers>>,
    /// GPU batch scratch buffers for prefill, allocated lazily.
    batch_scratches: Mutex<Option<GpuBatchScratchBuffers>>,
    /// Reusable recurrent state buffers keyed by logical Qwen3.5 slot.
    qwen35_recurrent_slot_buffers:
        Mutex<FxHashMap<Qwen35RecurrentSlotBufferKey, Qwen35MetalSlotBuffers>>,
    /// Reusable scratch buffers for Qwen3.5 recurrent GPU QKV handoff prefill.
    qwen35_qkv_handoff_scratch_buffers:
        Mutex<FxHashMap<Qwen35QkvHandoffScratchBufferKey, Qwen35MetalQkvHandoffScratchBuffers>>,
    /// Reusable projection temp buffers for Qwen3.5 unified recurrent prefill.
    qwen35_recurrent_projection_scratch_buffers: Mutex<
        FxHashMap<Qwen35QkvHandoffScratchBufferKey, Qwen35MetalRecurrentProjectionScratchBuffers>,
    >,
    /// Reusable projection temp buffers for Qwen3.5 batch RMSNorm+projection staging.
    qwen35_batch_projection_scratch_buffers: Mutex<
        FxHashMap<Qwen35BatchProjectionScratchBufferKey, Qwen35MetalBatchProjectionScratchBuffers>,
    >,
    /// Reusable buffers for Qwen3.5 batched LM-head GPU projection.
    qwen35_batch_logits_scratch_buffers:
        Mutex<FxHashMap<Qwen35BatchLogitsScratchBufferKey, Qwen35MetalBatchLogitsScratchBuffers>>,
    /// Backend-local profiling for Qwen3.5 recurrent multi-token batch phases.
    qwen35_recurrent_batch_perf: Qwen35RecurrentBatchPerfState,
    /// Quant family used for runtime f16in batch-route autotune.
    runtime_f16in_autotune_quant: RwLock<GgmlType>,
    /// One-time runtime tuning state for f16in batch kernel routing.
    f16in_route_tuned: AtomicBool,
    /// Serializes one-time runtime f16in autotune updates under contention.
    f16in_route_tune_lock: Mutex<()>,
    /// Pre-computed weight cache keys, built once on first forward call.
    cached_model_keys: Mutex<Option<CachedModelKeys>>,
    /// Cache of raw quantized expert weight buffers for single-CB MoE dispatch.
    moe_quant_cache: Mutex<FxHashMap<usize, MetalBuffer>>,
}

/// Shared expert weight data for fused MoE dispatch.
pub struct SharedExpertWeights<'a> {
    pub gate_raw: &'a [u8],
    pub up_raw: &'a [u8],
    pub down_raw: &'a [u8],
    pub gate_inp_raw: Option<&'a [u8]>,
    pub gate_inp_dtype: Option<GgmlType>,
    pub dtype: GgmlType,
    pub inter_dim: usize,
    pub gate_inp_rows: usize,
}

pub struct SharedExpertCachedBuffers<'a> {
    pub gate: &'a MetalBuffer,
    pub up: &'a MetalBuffer,
    pub down: &'a MetalBuffer,
    pub gate_inp: Option<&'a MetalBuffer>,
    pub gate_inp_dtype: Option<GgmlType>,
    pub dtype: GgmlType,
    pub inter_dim: usize,
    pub gate_inp_rows: usize,
}

impl MetalOps {
    fn build_device_cache_key(device: &MetalDevice) -> String {
        let info = device.info();
        format!(
            "{}-apple7{}-apple9{}-metal3{}-metal4{}-uma{}",
            sanitize_cache_component(&info.name),
            u8::from(info.gpu_family.apple7),
            u8::from(info.gpu_family.apple9),
            u8::from(info.gpu_family.metal3),
            u8::from(info.gpu_family.metal4),
            u8::from(info.unified_memory),
        )
    }

    /// Create MetalOps sharing the same GPU as an existing MetalDevice.
    ///
    /// Creates a new command queue on the shared MTLDevice for independent
    /// command submission, avoiding a redundant system device enumeration.
    fn with_device(existing_device: &MetalDevice) -> anyhow::Result<Self> {
        let device = existing_device.clone_sharing_device()?;
        let elementwise = ElementwiseKernels::new(&device)?;
        let dequant = DequantKernels::new(&device)?;
        let attention = AttentionKernels::new(&device)?;
        let matmul = MatmulKernels::new(&device)?;
        let gdn = GdnKernels::new(&device)?;
        let device_cache_key = Self::build_device_cache_key(&device);
        let runtime_policy = RuntimePolicy::resolved_defaults();
        let decode_dispatch_cache = Self::build_decode_dispatch_cache(&runtime_policy);
        Ok(Self {
            device,
            elementwise,
            dequant,
            attention,
            matmul,
            gdn,
            device_cache_key,
            runtime_policy: RwLock::new(runtime_policy),
            decode_dispatch_cache: RwLock::new(decode_dispatch_cache),
            f32_weight_cache: Mutex::new(FxHashMap::default()),
            fused_qkv_weight_cache: Mutex::new(FxHashMap::default()),
            precomputed_f16_weight_cache: Mutex::new(FxHashMap::default()),
            scratches: Mutex::new(None),
            batch_scratches: Mutex::new(None),
            qwen35_recurrent_slot_buffers: Mutex::new(FxHashMap::default()),
            qwen35_qkv_handoff_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_recurrent_projection_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_batch_projection_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_batch_logits_scratch_buffers: Mutex::new(FxHashMap::default()),
            qwen35_recurrent_batch_perf: Qwen35RecurrentBatchPerfState::default(),
            runtime_f16in_autotune_quant: RwLock::new(GgmlType::Q4K),
            f16in_route_tuned: AtomicBool::new(false),
            f16in_route_tune_lock: Mutex::new(()),
            cached_model_keys: Mutex::new(None),
            moe_quant_cache: Mutex::new(FxHashMap::default()),
        })
    }

    fn build_decode_dispatch_cache(runtime_policy: &RuntimePolicy) -> Option<DecodeDispatchCache> {
        let profile = runtime_policy.kernel_profile();
        let regimes = profile.decode_regimes.as_ref()?;
        let short_profile = profile.effective_decode_profile(regimes.short_max_attend_len);
        let long_profile =
            profile.effective_decode_profile(regimes.short_max_attend_len.saturating_add(1));
        Some(DecodeDispatchCache {
            short_max_attend_len: regimes.short_max_attend_len,
            short_dequant_dispatch: DequantDispatchConfig::from_profile(&short_profile),
            short_attention_dispatch: AttentionDispatchConfig::from_profile(&short_profile),
            long_dequant_dispatch: DequantDispatchConfig::from_profile(&long_profile),
            long_attention_dispatch: AttentionDispatchConfig::from_profile(&long_profile),
        })
    }

    pub fn apply_runtime_policy(&self, runtime_policy: RuntimePolicy) {
        let decode_dispatch_cache = Self::build_decode_dispatch_cache(&runtime_policy);
        *self.runtime_policy.write().unwrap() = runtime_policy;
        *self.decode_dispatch_cache.write().unwrap() = decode_dispatch_cache;
    }

    pub fn runtime_policy(&self) -> RuntimePolicy {
        self.runtime_policy.read().unwrap().clone()
    }

    fn set_runtime_f16in_autotune_quant(&self, quant: GgmlType) {
        *self.runtime_f16in_autotune_quant.write().unwrap() = quant;
    }

    fn runtime_f16in_autotune_quant(&self) -> GgmlType {
        *self.runtime_f16in_autotune_quant.read().unwrap()
    }

    pub fn dequant_dispatch_config(&self) -> DequantDispatchConfig {
        self.runtime_policy
            .read()
            .unwrap()
            .dequant_dispatch_config()
    }

    pub fn attention_dispatch_config(&self) -> AttentionDispatchConfig {
        self.runtime_policy
            .read()
            .unwrap()
            .attention_dispatch_config()
    }

    pub fn decode_dispatch_configs_for_attend_len(
        &self,
        attend_len: u32,
    ) -> (DequantDispatchConfig, AttentionDispatchConfig) {
        if let Some(cache) = *self.decode_dispatch_cache.read().unwrap() {
            return cache.for_attend_len(attend_len);
        }
        let runtime_policy = self.runtime_policy.read().unwrap();
        (
            runtime_policy.dequant_dispatch_config(),
            runtime_policy.attention_dispatch_config(),
        )
    }

    pub fn metal_batch_f16_io_enabled(&self) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .batch_prefill_prefers_f16_io()
    }

    pub(crate) fn record_qwen35_recurrent_batch_conv(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .conv_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_pack(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .pack_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_gated_delta(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .gated_delta_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_unpack(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .unpack_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_gpu(&self, elapsed: Duration) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_ns
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_fused_tail(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_fused_tail_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_gpu_projection(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_gpu_projection_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_path_eligible(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_path_eligible_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_state_size(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_state_size_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_group_divisibility(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_group_divisibility_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_missing_batch_scratches(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_missing_batch_scratches_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_q_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_q_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_k_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_k_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_v_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_v_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_gate_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_gate_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_qkv_fast_reject_up_capacity(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_up_capacity_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_gpu_ssm_projection(&self) {
        self.qwen35_recurrent_batch_perf
            .gpu_ssm_projection_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_cpu_alias(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_alias_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_slot_buffer(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_slot_buffer_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_backend_carryover(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_carryover_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_backend_zero_init(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_zero_init_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub(crate) fn record_qwen35_recurrent_batch_qkv_handoff_cpu_materialization(&self) {
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_materialization_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_backend_native(&self) {
        self.qwen35_recurrent_batch_perf
            .state_batch_backend_native_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_direct(&self) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_direct_materialized_from_backend(
        &self,
    ) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_materialized_from_backend_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_gathered(&self) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_qwen35_recurrent_batch_state_batch_cpu_gathered_materialized_from_backend(
        &self,
    ) {
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_materialized_from_backend_layers
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset_qwen35_recurrent_batch_perf_counters(&self) {
        self.qwen35_recurrent_batch_perf
            .conv_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .pack_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .gated_delta_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .unpack_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_ns
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_fused_tail_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_gpu_projection_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_path_eligible_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .gpu_ssm_projection_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_state_size_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_group_divisibility_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_missing_batch_scratches_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_q_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_k_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_v_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_gate_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_fast_reject_up_capacity_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_alias_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_slot_buffer_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_carryover_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_backend_zero_init_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .qkv_handoff_cpu_materialization_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_backend_native_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_direct_materialized_from_backend_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_layers
            .store(0, Ordering::Relaxed);
        self.qwen35_recurrent_batch_perf
            .state_batch_cpu_gathered_materialized_from_backend_layers
            .store(0, Ordering::Relaxed);
    }

    pub fn qwen35_recurrent_batch_perf_counters(&self) -> Qwen35RecurrentBatchPerfCounters {
        Qwen35RecurrentBatchPerfCounters {
            conv_ns: self
                .qwen35_recurrent_batch_perf
                .conv_ns
                .load(Ordering::Relaxed),
            pack_ns: self
                .qwen35_recurrent_batch_perf
                .pack_ns
                .load(Ordering::Relaxed),
            gated_delta_ns: self
                .qwen35_recurrent_batch_perf
                .gated_delta_ns
                .load(Ordering::Relaxed),
            unpack_ns: self
                .qwen35_recurrent_batch_perf
                .unpack_ns
                .load(Ordering::Relaxed),
            qkv_handoff_ns: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_ns
                .load(Ordering::Relaxed),
            qkv_handoff_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_layers
                .load(Ordering::Relaxed),
            qkv_handoff_fused_tail_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_fused_tail_layers
                .load(Ordering::Relaxed),
            qkv_gpu_projection_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_gpu_projection_layers
                .load(Ordering::Relaxed),
            qkv_fast_path_eligible_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_path_eligible_layers
                .load(Ordering::Relaxed),
            gpu_ssm_projection_layers: self
                .qwen35_recurrent_batch_perf
                .gpu_ssm_projection_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_state_size_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_state_size_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_group_divisibility_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_group_divisibility_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_missing_batch_scratches_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_missing_batch_scratches_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_q_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_q_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_k_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_k_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_v_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_v_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_gate_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_gate_capacity_layers
                .load(Ordering::Relaxed),
            qkv_fast_reject_up_capacity_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_fast_reject_up_capacity_layers
                .load(Ordering::Relaxed),
            qkv_handoff_cpu_alias_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_cpu_alias_layers
                .load(Ordering::Relaxed),
            qkv_handoff_slot_buffer_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_slot_buffer_layers
                .load(Ordering::Relaxed),
            qkv_handoff_backend_carryover_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_backend_carryover_layers
                .load(Ordering::Relaxed),
            qkv_handoff_backend_zero_init_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_backend_zero_init_layers
                .load(Ordering::Relaxed),
            qkv_handoff_cpu_materialization_layers: self
                .qwen35_recurrent_batch_perf
                .qkv_handoff_cpu_materialization_layers
                .load(Ordering::Relaxed),
            state_batch_backend_native_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_backend_native_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_direct_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_direct_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_direct_materialized_from_backend_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_direct_materialized_from_backend_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_gathered_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_gathered_layers
                .load(Ordering::Relaxed),
            state_batch_cpu_gathered_materialized_from_backend_layers: self
                .qwen35_recurrent_batch_perf
                .state_batch_cpu_gathered_materialized_from_backend_layers
                .load(Ordering::Relaxed),
        }
    }

    pub fn metal_batch_f16_pair_enabled(&self) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .batch_prefill_prefers_pair_kernel()
    }

    pub fn metal_fused_qkv_enabled(&self) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .fused_qkv_prefill_enabled()
    }

    pub fn metal_decode_fused_qkv_enabled(&self) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .decode_fused_qkv_enabled()
    }

    pub fn metal_batch_simd_enabled(&self) -> bool {
        self.runtime_policy.read().unwrap().batch_simd_enabled()
    }

    pub fn metal_precompute_f16_enabled(&self) -> bool {
        self.runtime_policy.read().unwrap().precompute_f16_enabled()
    }

    pub fn metal_autotune_enabled(&self) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .autotune_f16in_batch_route_enabled()
    }

    pub fn metal_q8_batch_native_shape_enabled(&self, m: u32, n: u32, k: u32) -> bool {
        self.runtime_policy
            .read()
            .unwrap()
            .q8_batch_native_shape_enabled(m, n, k)
    }

    fn resolve_kernel_profile_for_fingerprint(
        &self,
        fingerprint: &ModelFingerprint,
    ) -> ResolvedKernelProfile {
        let seed_profile =
            KernelProfile::load(&fingerprint.model_name, &fingerprint.predominant_quant);
        let cache_path = self.kernel_profile_cache_path(fingerprint);
        let load_autotune = metal_load_autotune_enabled();
        let force_retune = metal_force_retune_enabled();
        let can_loadtime_batch_route_tune =
            self.preferred_batch_prefill_quant(fingerprint).is_some();

        if !load_autotune && !force_retune {
            tracing::debug!(
                fingerprint = %fingerprint.stable_id(),
                "Load-time kernel autotune disabled; using seeded kernel profile"
            );
            return ResolvedKernelProfile {
                profile: seed_profile,
                skip_runtime_batch_route_autotune: false,
            };
        }

        if !force_retune && let Some(profile) = KernelProfile::load_from_path(&cache_path) {
            let profile = self.reconcile_cached_profile_kv_precision(
                profile,
                &seed_profile,
                fingerprint,
                &cache_path,
            );
            tracing::info!(
                path = %cache_path.display(),
                fingerprint = %fingerprint.stable_id(),
                "Loaded cached kernel autotune profile"
            );
            return ResolvedKernelProfile {
                profile,
                skip_runtime_batch_route_autotune: can_loadtime_batch_route_tune,
            };
        }

        let tuned = self.autotune_kernel_profile(fingerprint, seed_profile);
        if let Err(error) = self.persist_kernel_profile_cache(&cache_path, &tuned) {
            tracing::warn!(
                path = %cache_path.display(),
                fingerprint = %fingerprint.stable_id(),
                error = %error,
                "Failed to persist autotuned kernel profile cache"
            );
        }

        ResolvedKernelProfile {
            profile: tuned,
            skip_runtime_batch_route_autotune: can_loadtime_batch_route_tune,
        }
    }

    fn kernel_profile_cache_path(&self, fingerprint: &ModelFingerprint) -> PathBuf {
        let file_name = format!(
            "{}-{}.json",
            sanitize_cache_component(&fingerprint.model_name),
            fingerprint.stable_id(),
        );
        metal_tune_cache_dir()
            .join(format!("ax-{}", env!("CARGO_PKG_VERSION")))
            .join(&self.device_cache_key)
            .join(sanitize_cache_component(&fingerprint.cache_namespace()))
            .join(file_name)
    }

    fn persist_kernel_profile_cache(
        &self,
        cache_path: &PathBuf,
        profile: &KernelProfile,
    ) -> anyhow::Result<()> {
        let Some(parent) = cache_path.parent() else {
            return Ok(());
        };
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let json = serde_json::to_string_pretty(profile)
            .context("failed to encode autotuned kernel profile")?;
        fs::write(cache_path, json)
            .with_context(|| format!("failed to write {}", cache_path.display()))?;
        Ok(())
    }

    fn reconcile_cached_profile_kv_precision(
        &self,
        mut cached_profile: KernelProfile,
        seed_profile: &KernelProfile,
        fingerprint: &ModelFingerprint,
        cache_path: &Path,
    ) -> KernelProfile {
        let seeded_precision = seed_profile.kv_cache.precision;
        if cached_profile.kv_cache.precision != seeded_precision {
            tracing::info!(
                path = %cache_path.display(),
                fingerprint = %fingerprint.stable_id(),
                cached = ?cached_profile.kv_cache.precision,
                seeded = ?seeded_precision,
                "Ignoring cached KV precision override; using seeded profile precision"
            );
            cached_profile.kv_cache.precision = seeded_precision;
        }
        cached_profile
    }

    fn autotune_kernel_profile(
        &self,
        fingerprint: &ModelFingerprint,
        mut profile: KernelProfile,
    ) -> KernelProfile {
        let recommended_kv_precision = self.select_kv_precision_mode(fingerprint);
        tracing::debug!(
            fingerprint = %fingerprint.stable_id(),
            recommended = ?recommended_kv_precision,
            "Load-time KV precision recommendation computed (runtime selection remains profile/env driven)"
        );
        if fingerprint.has_q4k_layer_weights {
            self.autotune_decode_matvec_quant(&mut profile, fingerprint, GgmlType::Q4K);
        }
        if fingerprint.has_q5k_layer_weights {
            self.autotune_decode_matvec_quant(&mut profile, fingerprint, GgmlType::Q5K);
        }
        if fingerprint.has_q6k_layer_weights {
            self.autotune_decode_matvec_quant(&mut profile, fingerprint, GgmlType::Q6K);
        }
        if fingerprint.has_q8_layer_weights {
            self.autotune_decode_matvec_quant(&mut profile, fingerprint, GgmlType::Q8_0);
        }
        if let Some(batch_quant) = self.preferred_batch_prefill_quant(fingerprint) {
            self.autotune_batch_prefill_route(&mut profile, fingerprint, batch_quant);
        }

        profile.model = fingerprint.model_name.clone();
        profile.source = profile_source_with_autotune(&profile.source);
        profile.generated = format!(
            "ax-engine={} device={} fingerprint={}",
            env!("CARGO_PKG_VERSION"),
            self.device_cache_key.as_str(),
            fingerprint.stable_id(),
        );
        profile
    }

    fn select_kv_precision_mode(&self, fingerprint: &ModelFingerprint) -> KvPrecisionMode {
        let info = self.device.info();
        let kv_f32_bytes =
            Self::estimated_attention_kv_bytes(fingerprint, KvPrecisionPolicy::ForceF32)
                .unwrap_or(0);
        let kv_f16_bytes =
            Self::estimated_attention_kv_bytes(fingerprint, KvPrecisionPolicy::ForceF16)
                .unwrap_or(kv_f32_bytes / 2);
        let kv_q8_bytes =
            Self::estimated_attention_kv_bytes(fingerprint, KvPrecisionPolicy::ForceQ8_0)
                .unwrap_or(u64::MAX);
        let working_set = info.max_working_set_bytes.max(1);
        let total_f16 = fingerprint.total_tensor_bytes.saturating_add(kv_f16_bytes);

        let selected = if !Self::supports_q8_kv(fingerprint) {
            if fingerprint.context_length < 256 && kv_f16_bytes < (128 << 20) {
                KvPrecisionMode::F32
            } else {
                KvPrecisionMode::F16
            }
        } else if total_f16 > working_set.saturating_mul(85) / 100
            || kv_f16_bytes >= (1 << 30)
            || fingerprint.context_length >= 16_384
        {
            KvPrecisionMode::Q8_0
        } else if fingerprint.context_length < 256 && kv_f16_bytes < (128 << 20) {
            KvPrecisionMode::F32
        } else {
            KvPrecisionMode::F16
        };

        tracing::info!(
            fingerprint = %fingerprint.stable_id(),
            kv_f32_mb = kv_f32_bytes / 1024 / 1024,
            kv_f16_mb = kv_f16_bytes / 1024 / 1024,
            kv_q8_mb = if kv_q8_bytes == u64::MAX {
                0
            } else {
                kv_q8_bytes / 1024 / 1024
            },
            working_set_mb = working_set / 1024 / 1024,
            selected = ?selected,
            "Selected load-time KV precision policy"
        );
        selected
    }

    fn supports_q8_kv(fingerprint: &ModelFingerprint) -> bool {
        let kv_stride = fingerprint.n_kv_heads.saturating_mul(fingerprint.head_dim);
        kv_stride > 0
            && kv_stride.is_multiple_of(32)
            && matches!(fingerprint.head_dim, 64 | 128 | 256)
    }

    fn estimated_attention_kv_layers(fingerprint: &ModelFingerprint) -> u64 {
        if matches!(fingerprint.architecture.as_str(), "qwen35" | "qwen35moe")
            && let Some(interval) = fingerprint.qwen35_full_attention_interval
            && interval > 0
        {
            return (fingerprint.n_layers as u64).div_ceil(interval as u64);
        }
        fingerprint.n_layers as u64
    }

    fn estimated_attention_kv_bytes(
        fingerprint: &ModelFingerprint,
        policy: KvPrecisionPolicy,
    ) -> Option<u64> {
        let kv_stride = fingerprint.n_kv_heads as usize * fingerprint.head_dim as usize;
        let context = fingerprint.context_length as usize;
        let layers = Self::estimated_attention_kv_layers(fingerprint);
        let row_bytes = match policy {
            KvPrecisionPolicy::ForceF32 => (kv_stride * std::mem::size_of::<f32>()) as u64,
            KvPrecisionPolicy::ForceF16 => (kv_stride * std::mem::size_of::<half::f16>()) as u64,
            KvPrecisionPolicy::ForceQ8_0 => {
                if kv_stride == 0 || !kv_stride.is_multiple_of(32) {
                    return None;
                }
                ((kv_stride / 32) * 34) as u64
            }
            KvPrecisionPolicy::Auto => return None,
        };

        (layers)
            .checked_mul(context as u64)?
            .checked_mul(row_bytes)?
            .checked_mul(2)
    }

    fn autotune_decode_matvec_quant(
        &self,
        profile: &mut KernelProfile,
        fingerprint: &ModelFingerprint,
        quant: GgmlType,
    ) {
        let Some((quant_key, candidates)) = Self::decode_matvec_autotune_candidates(profile, quant)
        else {
            return;
        };

        let candidates = Self::dedup_matvec_candidates(candidates);
        if candidates.is_empty() {
            return;
        }
        let shapes = Self::decode_autotune_shapes(fingerprint);
        let base_config = self.dequant_dispatch_config();
        let mut total_ns = vec![0u128; candidates.len()];
        let mut samples = vec![0u32; candidates.len()];

        for shape in &shapes {
            let m = shape.m;
            let k = shape.k;
            let weight = shape.weight;
            let Some(buffers) = self.prepare_matvec_bench_buffers(quant, m, k) else {
                continue;
            };
            for (idx, candidate) in candidates.iter().enumerate() {
                if let Some(ns) = self.bench_matvec_variant_with_buffers(
                    quant,
                    m,
                    k,
                    candidate,
                    &buffers,
                    base_config,
                ) {
                    total_ns[idx] += ns.saturating_mul(weight as u128);
                    samples[idx] = samples[idx].saturating_add(weight);
                }
            }
        }

        let mut best: Option<(usize, u128)> = None;
        for (idx, sample_count) in samples.iter().copied().enumerate() {
            if sample_count == 0 {
                continue;
            }
            let avg_ns = total_ns[idx] / sample_count as u128;
            match best {
                Some((_, best_avg_ns)) if best_avg_ns <= avg_ns => {}
                _ => best = Some((idx, avg_ns)),
            };
        }

        if let Some((best_idx, best_avg_ns)) = best {
            let best_params = candidates[best_idx].clone();
            let old_params = profile.matvec_params(quant_key);
            if old_params != best_params {
                tracing::info!(
                    quant = quant_key,
                    fingerprint = %fingerprint.stable_id(),
                    old_threadgroup_size = old_params.threadgroup_size,
                    old_rows_per_simdgroup = old_params.rows_per_simdgroup,
                    old_variant = ?old_params.variant,
                    tuned_threadgroup_size = best_params.threadgroup_size,
                    tuned_rows_per_simdgroup = best_params.rows_per_simdgroup,
                    tuned_variant = ?best_params.variant,
                    avg_ns = best_avg_ns,
                    "Autotuned decode matvec profile"
                );
            }
            profile
                .decode_matvec
                .insert(quant_key.to_string(), best_params);
        }
    }

    fn decode_matvec_autotune_candidates(
        profile: &KernelProfile,
        quant: GgmlType,
    ) -> Option<(&'static str, Vec<MatvecParams>)> {
        let quant_key = decode_matvec_profile_key(quant)?;
        let mut candidates = Vec::with_capacity(1 + decode_matvec_default_candidates(quant).len());
        candidates.push(Self::autotune_active_matvec_params(
            profile, quant_key, quant,
        ));
        candidates.extend(decode_matvec_default_candidates(quant).iter().cloned());
        Some((quant_key, candidates))
    }

    fn autotune_active_matvec_params(
        profile: &KernelProfile,
        quant_key: &str,
        quant: GgmlType,
    ) -> MatvecParams {
        let Some(override_params) = profile.decode_matvec.get(quant_key) else {
            return profile.matvec_params(quant_key);
        };

        let inferred_variant = override_params.variant.or(match quant {
            GgmlType::Q4K | GgmlType::Q6K => {
                if override_params.rows_per_simdgroup >= 2 || override_params.threadgroup_size == 64
                {
                    Some(MatvecProfileVariant::Nr2)
                } else {
                    Some(MatvecProfileVariant::Base)
                }
            }
            GgmlType::Q5K => {
                if override_params.rows_per_simdgroup >= 2 {
                    Some(MatvecProfileVariant::Nr2)
                } else {
                    Some(MatvecProfileVariant::Base)
                }
            }
            GgmlType::Q8_0 => {
                if override_params.rows_per_simdgroup >= 2 {
                    Some(MatvecProfileVariant::Nr2)
                } else if override_params.threadgroup_size == 64 {
                    Some(MatvecProfileVariant::Ilp4)
                } else {
                    Some(MatvecProfileVariant::Base)
                }
            }
            _ => None,
        });

        MatvecParams {
            threadgroup_size: override_params.threadgroup_size,
            rows_per_simdgroup: override_params.rows_per_simdgroup,
            variant: inferred_variant,
        }
    }

    fn dedup_matvec_candidates(candidates: Vec<MatvecParams>) -> Vec<MatvecParams> {
        let mut unique = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !unique
                .iter()
                .any(|existing: &MatvecParams| existing == &candidate)
            {
                unique.push(candidate);
            }
        }
        unique
    }

    fn preferred_batch_prefill_quant(&self, fingerprint: &ModelFingerprint) -> Option<GgmlType> {
        quant_from_profile_label(&fingerprint.predominant_layer_quant)
            .or_else(|| quant_from_profile_label(&fingerprint.predominant_quant))
            .or({
                if fingerprint.has_q4k_layer_weights {
                    Some(GgmlType::Q4K)
                } else if fingerprint.has_q5k_layer_weights {
                    Some(GgmlType::Q5K)
                } else if fingerprint.has_q6k_layer_weights {
                    Some(GgmlType::Q6K)
                } else if fingerprint.has_q8_layer_weights {
                    Some(GgmlType::Q8_0)
                } else {
                    None
                }
            })
    }

    fn autotune_batch_prefill_route(
        &self,
        profile: &mut KernelProfile,
        fingerprint: &ModelFingerprint,
        quant: GgmlType,
    ) {
        let Some((n_threshold, m_max)) = self.tune_f16in_batch_route(
            quant,
            fingerprint.embedding_dim,
            fingerprint.intermediate_dim,
        ) else {
            return;
        };

        let old_n_threshold = profile.batch_prefill.small_n_threshold;
        let old_m_max = profile.batch_prefill.small_m_max;
        profile.batch_prefill.small_n_threshold = n_threshold;
        profile.batch_prefill.small_m_max = m_max;

        if old_n_threshold != n_threshold || old_m_max != m_max {
            tracing::info!(
                quant = %quant,
                fingerprint = %fingerprint.stable_id(),
                old_n_threshold,
                old_m_max,
                tuned_n_threshold = n_threshold,
                tuned_m_max = m_max,
                "Autotuned batch prefill routing profile"
            );
        }
    }

    fn decode_autotune_shapes(fingerprint: &ModelFingerprint) -> Vec<WeightedMatvecShape> {
        let dim = fingerprint.embedding_dim;
        let inter = fingerprint.intermediate_dim;
        let kv_dim = fingerprint.n_kv_heads.saturating_mul(fingerprint.head_dim);
        let q_dim = fingerprint.n_heads.saturating_mul(fingerprint.head_dim);
        let mut shapes: Vec<WeightedMatvecShape> = Vec::new();
        for (m, k) in [
            (q_dim, dim),
            (kv_dim, dim),
            (kv_dim, dim),
            (dim, dim),
            (inter, dim),
            (inter, dim),
            (dim, inter),
        ] {
            if m == 0 || k == 0 {
                continue;
            }
            if let Some(existing) = shapes.iter_mut().find(|shape| shape.m == m && shape.k == k) {
                existing.weight = existing.weight.saturating_add(1);
            } else {
                shapes.push(WeightedMatvecShape { m, k, weight: 1 });
            }
        }
        shapes
    }

    fn tune_f16in_batch_route(&self, quant: GgmlType, dim: u32, inter: u32) -> Option<(u32, u32)> {
        let k = dim;
        let n_candidates = [32u32, 48u32, 64u32];
        let m_candidates = [dim, inter];
        let m_candidate_count = if dim == inter { 1 } else { 2 };
        let base_config = self.dequant_dispatch_config();
        let mut compared_any = false;
        let mut has_win = false;
        let mut max_n = 0u32;
        let mut max_m = 0u32;

        for &m in &m_candidates[..m_candidate_count] {
            for &n in &n_candidates {
                let Some(buffers) = self.prepare_f16in_bench_buffers(quant, m, n, k) else {
                    continue;
                };
                let t_large = self.bench_f16in_variant_with_buffers(
                    quant,
                    m,
                    n,
                    k,
                    false,
                    &buffers,
                    base_config,
                );
                let t_small = self.bench_f16in_variant_with_buffers(
                    quant,
                    m,
                    n,
                    k,
                    true,
                    &buffers,
                    base_config,
                );
                if let (Some(large), Some(small)) = (t_large, t_small) {
                    compared_any = true;
                    if small < (large * 98 / 100) {
                        has_win = true;
                        max_n = max_n.max(n);
                        max_m = max_m.max(m);
                    }
                }
            }
        }

        if !compared_any {
            return None;
        }

        Some(if !has_win {
            (1u32, 0u32)
        } else {
            (max_n + 1, max_m)
        })
    }

    fn prepare_matvec_bench_buffers(
        &self,
        quant: GgmlType,
        m: u32,
        k: u32,
    ) -> Option<MatvecBenchBuffers> {
        if m == 0 || k == 0 || !k.is_multiple_of(quant.block_size() as u32) {
            return None;
        }

        let blocks_per_row = k as usize / quant.block_size();
        if blocks_per_row == 0 {
            return None;
        }

        let a_bytes = (m as usize)
            .checked_mul(blocks_per_row)?
            .checked_mul(quant.bytes_per_block())?;
        let a = MetalBuffer::from_bytes(self.device.device(), &vec![0u8; a_bytes]).ok()?;
        let x = MetalBuffer::from_slice(self.device.device(), &vec![0.0f32; k as usize]).ok()?;
        let y = MetalBuffer::new(
            self.device.device(),
            m as usize * std::mem::size_of::<f32>(),
        )
        .ok()?;

        Some(MatvecBenchBuffers { a, x, y })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_matvec_with_config(
        &self,
        quant: GgmlType,
        a: &MetalBuffer,
        x: &MetalBuffer,
        y: &MetalBuffer,
        m: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) -> anyhow::Result<()> {
        match quant {
            GgmlType::Q4K => {
                self.dequant
                    .fused_matvec_q4_k_with_config(&self.device, a, x, y, m, k, config)
            }
            GgmlType::Q5K => {
                self.dequant
                    .fused_matvec_q5_k_with_config(&self.device, a, x, y, m, k, config)
            }
            GgmlType::Q6K => {
                self.dequant
                    .fused_matvec_q6_k_with_config(&self.device, a, x, y, m, k, config)
            }
            GgmlType::Q8_0 => self.device.execute_sync(|encoder| {
                self.dequant
                    .encode_fused_matvec_q8_0_with_config(encoder, a, x, y, m, k, config);
                Ok(())
            }),
            _ => unreachable!("unsupported quant for decode matvec autotune benchmark"),
        }
    }

    fn matvec_dispatch_config_for_candidate(
        quant: GgmlType,
        config_params: &MatvecParams,
        base_config: DequantDispatchConfig,
    ) -> Option<DequantDispatchConfig> {
        let mut config = base_config;
        match quant {
            GgmlType::Q4K => {
                config.q4_k_threadgroup_size = config_params.threadgroup_size as usize;
                config.q4_k_rows_per_simdgroup = config_params.rows_per_simdgroup;
                config.q4_k_variant = config_params.variant;
            }
            GgmlType::Q5K => {
                config.q5_k_rows_per_simdgroup = config_params.rows_per_simdgroup;
                config.q5_k_ilp4 =
                    matches!(config_params.variant, Some(MatvecProfileVariant::Ilp4));
                config.q5_k_variant = config_params.variant;
            }
            GgmlType::Q6K => {
                config.q6_k_threadgroup_size = config_params.threadgroup_size as usize;
                config.q6_k_rows_per_simdgroup = config_params.rows_per_simdgroup;
                config.q6_k_variant = config_params.variant;
            }
            GgmlType::Q8_0 => {
                config.q8_0_variant = config_params.variant;
            }
            _ => return None,
        }
        Some(config)
    }

    fn bench_matvec_variant_with_buffers(
        &self,
        quant: GgmlType,
        m: u32,
        k: u32,
        config_params: &MatvecParams,
        buffers: &MatvecBenchBuffers,
        base_config: DequantDispatchConfig,
    ) -> Option<u128> {
        let config = Self::matvec_dispatch_config_for_candidate(quant, config_params, base_config)?;

        if self
            .dispatch_matvec_with_config(quant, &buffers.a, &buffers.x, &buffers.y, m, k, config)
            .is_err()
        {
            return None;
        }

        let reps = 4;
        let t0 = Instant::now();
        for _ in 0..reps {
            if self
                .dispatch_matvec_with_config(
                    quant, &buffers.a, &buffers.x, &buffers.y, m, k, config,
                )
                .is_err()
            {
                return None;
            }
        }
        Some(t0.elapsed().as_nanos() / reps as u128)
    }

    fn prepare_f16in_bench_buffers(
        &self,
        quant: GgmlType,
        m: u32,
        n: u32,
        k: u32,
    ) -> Option<BatchF16InBenchBuffers> {
        if !k.is_multiple_of(quant.block_size() as u32) {
            return None;
        }

        let blocks_per_row = k as usize / quant.block_size();
        if blocks_per_row == 0 {
            return None;
        }

        let a_bytes = (m as usize)
            .checked_mul(blocks_per_row)?
            .checked_mul(quant.bytes_per_block())?;
        let b_len = (n as usize).checked_mul(k as usize)?;
        let c_bytes = (n as usize)
            .checked_mul(m as usize)?
            .checked_mul(std::mem::size_of::<f32>())?;

        let a = MetalBuffer::from_bytes(self.device.device(), &vec![0u8; a_bytes]).ok()?;
        let b =
            MetalBuffer::from_slice(self.device.device(), &vec![half::f16::from_f32(0.0); b_len])
                .ok()?;
        let c = MetalBuffer::new(self.device.device(), c_bytes).ok()?;

        Some(BatchF16InBenchBuffers { a, b, c })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_fused_batch_f16in_with_config(
        &self,
        quant: GgmlType,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        config: DequantDispatchConfig,
    ) -> anyhow::Result<()> {
        self.device.execute_sync(|encoder| {
            match quant {
                GgmlType::Q4K => self
                    .dequant
                    .encode_fused_batch_q4_k_f16in_with_config(encoder, a, b, c, m, n, k, config),
                GgmlType::Q5K => self
                    .dequant
                    .encode_fused_batch_q5_k_f16in_with_config(encoder, a, b, c, m, n, k, config),
                GgmlType::Q6K => self
                    .dequant
                    .encode_fused_batch_q6_k_f16in_with_config(encoder, a, b, c, m, n, k, config),
                GgmlType::Q8_0 => self
                    .dequant
                    .encode_fused_batch_q8_0_f16in_with_config(encoder, a, b, c, m, n, k, config),
                _ => unreachable!("unsupported quant for f16in batch autotune benchmark"),
            }
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn bench_f16in_variant_with_buffers(
        &self,
        quant: GgmlType,
        m: u32,
        n: u32,
        k: u32,
        use_small: bool,
        buffers: &BatchF16InBenchBuffers,
        base_config: DequantDispatchConfig,
    ) -> Option<u128> {
        let mut bench_config = base_config;
        if use_small {
            bench_config.batch_f16in_small_n_threshold = u32::MAX;
            bench_config.batch_f16in_small_m_max = m;
        } else {
            bench_config.batch_f16in_small_n_threshold = 1;
            bench_config.batch_f16in_small_m_max = 0;
        }

        let _ = self.encode_fused_batch_f16in_with_config(
            quant,
            &buffers.a,
            &buffers.b,
            &buffers.c,
            m,
            n,
            k,
            bench_config,
        );

        let reps = 3;
        let t0 = Instant::now();
        for _ in 0..reps {
            if self
                .encode_fused_batch_f16in_with_config(
                    quant,
                    &buffers.a,
                    &buffers.b,
                    &buffers.c,
                    m,
                    n,
                    k,
                    bench_config,
                )
                .is_err()
            {
                return None;
            }
        }
        Some(t0.elapsed().as_nanos() / reps as u128)
    }

    /// Initialize scratch buffers sized to the model config.
    /// Called lazily on first forward pass.
    pub fn init_scratches(&self, config: &ModelConfig) {
        let mut guard = self.scratches.lock().unwrap();
        if guard.is_some() {
            return;
        }

        let dim = config.embedding_dim as usize;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let head_dim = config.head_dim as usize;
        let inter_dim = config.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let (gate_scratch_dim, up_scratch_dim) =
            qwen35_gate_up_scratch_dims(config, q_dim, inter_dim);
        let max_chunks = config
            .context_length
            .div_ceil(self.attention_dispatch_config().decode_splitk_chunk_size())
            as usize;

        let alloc = |size: usize| -> MetalBuffer {
            MetalBuffer::new(self.device.device(), size * std::mem::size_of::<f32>())
                .expect("Failed to allocate GPU scratch buffer")
        };
        let alloc_f16 = |size: usize| -> MetalBuffer {
            MetalBuffer::new(
                self.device.device(),
                size * std::mem::size_of::<half::f16>(),
            )
            .expect("Failed to allocate GPU decode f16 scratch buffer")
        };
        let max_input_dim = dim.max(inter_dim);
        let vocab_size = config.vocab_size as usize;
        let scratches = GpuScratchBuffers {
            hidden: alloc(dim),
            norm_buf: alloc(dim),
            matmul_in_f16: alloc_f16(max_input_dim),
            qkv_buf: alloc(q_dim + 2 * kv_dim),
            q_buf: alloc(q_dim),
            k_buf: alloc(kv_dim),
            v_buf: alloc(kv_dim),
            attn_out: alloc(q_dim),
            proj_buf: alloc(dim),
            gate_buf: alloc(gate_scratch_dim),
            up_buf: alloc(up_scratch_dim),
            down_buf: alloc(dim),
            logits_buf: alloc(vocab_size),
            splitk_partial_out: alloc(max_chunks * n_heads * head_dim),
            splitk_partial_lse: alloc(max_chunks * n_heads),
            argmax_idx: alloc(1), // 1 × u32 (4 bytes)
            argmax_val: alloc(1), // 1 × f32 (4 bytes)
        };
        tracing::info!(
            dim,
            q_dim,
            kv_dim,
            inter_dim,
            gate_scratch_dim,
            up_scratch_dim,
            max_chunks,
            "GPU scratch buffers allocated for phased dispatch"
        );
        *guard = Some(scratches);
    }

    /// Access GPU scratch buffers (must call init_scratches first).
    pub fn scratches(&self) -> std::sync::MutexGuard<'_, Option<GpuScratchBuffers>> {
        self.scratches.lock().unwrap()
    }

    /// Check if cached model weight keys have already been built.
    pub fn has_cached_model_keys(&self) -> bool {
        self.cached_model_keys.lock().unwrap().is_some()
    }

    /// Store pre-computed model weight keys (called once on first forward pass).
    pub fn set_cached_model_keys(&self, keys: CachedModelKeys) {
        let mut guard = self.cached_model_keys.lock().unwrap();
        *guard = Some(keys);
        drop(guard);
        // First key cache is also the final weight-loading event: commit residency
        // so the OS keeps all model buffers wired for the GPU.
        self.commit_weight_residency();
    }

    /// Register all cached weight buffers in the Metal residency set and commit.
    ///
    /// Call once after `build_cached_model_keys` so the OS keeps model weights
    /// resident in GPU-accessible memory.  No-op if residency is disabled.
    pub fn commit_weight_residency(&self) {
        if !self.device.has_residency() {
            return;
        }
        for buf in self.f32_weight_cache.lock().unwrap().values() {
            self.device.register_resident_buffer(buf);
        }
        for buf in self.fused_qkv_weight_cache.lock().unwrap().values() {
            self.device.register_resident_buffer(buf);
        }
        for buf in self.precomputed_f16_weight_cache.lock().unwrap().values() {
            self.device.register_resident_buffer(buf);
        }
        self.device.commit_residency();
        self.device.log_memory_pressure();
    }

    /// Access the cached model weight keys.
    pub fn cached_model_keys(&self) -> std::sync::MutexGuard<'_, Option<CachedModelKeys>> {
        self.cached_model_keys.lock().unwrap()
    }

    /// Initialize batch scratch buffers for prefill of `n_tokens` tokens.
    /// Re-allocates if the existing batch is smaller than needed.
    pub fn init_batch_scratches(&self, config: &ModelConfig, n_tokens: usize) {
        self.maybe_autotune_f16in_batch_route(config);
        let mut guard = self.batch_scratches.lock().unwrap();
        if let Some(ref existing) = *guard
            && existing.batch_size >= n_tokens
        {
            return;
        }

        let dim = config.embedding_dim as usize;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let head_dim = config.head_dim as usize;
        let inter_dim = config.intermediate_dim as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let (gate_scratch_dim, up_scratch_dim) =
            qwen35_gate_up_scratch_dims(config, q_dim, inter_dim);

        let alloc = |size: usize| -> MetalBuffer {
            MetalBuffer::new(self.device.device(), size * std::mem::size_of::<f32>())
                .expect("Failed to allocate GPU batch scratch buffer")
        };
        let alloc_f16 = |size: usize| -> MetalBuffer {
            MetalBuffer::new(
                self.device.device(),
                size * std::mem::size_of::<half::f16>(),
            )
            .expect("Failed to allocate GPU batch f16 scratch buffer")
        };
        let max_input_dim = dim.max(inter_dim);

        let max_output_dim = dim
            .max(kv_dim)
            .max(q_dim)
            .max(gate_scratch_dim)
            .max(up_scratch_dim);

        let mut bs = GpuBatchScratchBuffers {
            hidden: alloc(n_tokens * dim),
            qkv_buf: alloc(n_tokens * (q_dim + 2 * kv_dim)),
            q_buf: alloc(n_tokens * q_dim),
            k_buf: alloc(n_tokens * kv_dim),
            v_buf: alloc(n_tokens * kv_dim),
            attn_out: alloc(n_tokens * q_dim),
            norm_buf: alloc(n_tokens * dim),
            proj_buf: alloc(n_tokens * dim),
            gate_buf: alloc(n_tokens * gate_scratch_dim),
            up_buf: alloc(n_tokens * up_scratch_dim),
            matmul_in_f16: alloc_f16(n_tokens * max_input_dim),
            matmul_out_f16: alloc_f16(n_tokens * max_output_dim),
            batch_size: n_tokens,
            moe_norm: None,
            moe_router_out: None,
            moe_accum: None,
            moe_gate_out: None,
            moe_up_out: None,
            moe_down_out: None,
            moe_tpe: None,
            moe_hids: None,
            moe_expert_ids: None,
            moe_expert_weights: None,
            moe_active_experts: None,
            moe_tok_idx: None,
        };
        // Allocate MoE scratch buffers if this is a MoE model.
        let n_expert = config.n_expert.unwrap_or(0) as usize;
        let n_expert_used = config.n_expert_used.unwrap_or(0) as usize;
        let expert_inter_dim = config.expert_intermediate_dim.unwrap_or(0) as usize;
        if n_expert > 0 && n_expert_used > 0 && expert_inter_dim > 0 {
            let f4 = std::mem::size_of::<f32>();
            let i4 = std::mem::size_of::<i32>();
            let u4 = std::mem::size_of::<u32>();
            let alloc_bytes = |size: usize| -> MetalBuffer {
                MetalBuffer::new(self.device.device(), size)
                    .expect("Failed to allocate MoE scratch buffer")
            };
            bs.moe_norm = Some(alloc_bytes(n_tokens * dim * f4));
            bs.moe_router_out = Some(alloc_bytes(n_tokens * n_expert.max(dim) * f4));
            bs.moe_accum = Some(alloc_bytes(n_tokens * dim * f4));
            bs.moe_gate_out = Some(alloc_bytes(
                n_tokens * n_expert_used * expert_inter_dim * f4,
            ));
            bs.moe_up_out = Some(alloc_bytes(
                n_tokens * n_expert_used * expert_inter_dim * f4,
            ));
            bs.moe_down_out = Some(alloc_bytes(n_tokens * n_expert_used * dim * f4));
            bs.moe_tpe = Some(alloc_bytes(n_expert * u4));
            bs.moe_hids = Some(alloc_bytes(n_expert * n_tokens * i4 + (n_expert + 1) * u4));
            bs.moe_expert_ids = Some(alloc_bytes(n_tokens * n_expert_used * i4));
            bs.moe_expert_weights = Some(alloc_bytes(n_tokens * n_expert_used * f4));
            bs.moe_active_experts = Some(alloc_bytes(n_expert * u4));
            let mut tok_buf = alloc_bytes(n_tokens * n_expert_used * u4);
            // Pre-fill tok_idx: [0,0,...,0, 1,1,...,1, ...] — same every forward.
            unsafe {
                let toks = tok_buf.as_mut_slice::<u32>();
                for (i, tok) in toks.iter_mut().enumerate() {
                    *tok = (i / n_expert_used) as u32;
                }
            }
            bs.moe_tok_idx = Some(tok_buf);
            // Pre-fill active_experts: [0, 1, 2, ..., n_expert-1] (full range).
            let mut act_buf = alloc_bytes(n_expert * u4);
            unsafe {
                let acts = act_buf.as_mut_slice::<u32>();
                for (i, act) in acts.iter_mut().enumerate() {
                    *act = i as u32;
                }
            }
            bs.moe_active_experts = Some(act_buf);
        }
        tracing::info!(
            n_tokens,
            dim,
            q_dim,
            kv_dim,
            inter_dim,
            gate_scratch_dim,
            up_scratch_dim,
            "GPU batch scratch buffers allocated for prefill"
        );
        *guard = Some(bs);
    }

    fn maybe_autotune_f16in_batch_route(&self, config: &ModelConfig) {
        if !self.metal_autotune_enabled() {
            return;
        }
        if self.f16in_route_tuned.load(Ordering::Relaxed) {
            return;
        }
        let _tune_guard = self.f16in_route_tune_lock.lock().unwrap();
        if self.f16in_route_tuned.load(Ordering::Relaxed) {
            return;
        }

        let old_policy = self.runtime_policy();
        let old_config = old_policy.dequant_dispatch_config();
        let autotune_quant = self.runtime_f16in_autotune_quant();
        let Some((n_threshold, m_max)) = self.tune_f16in_batch_route(
            autotune_quant,
            config.embedding_dim,
            config.intermediate_dim,
        ) else {
            self.f16in_route_tuned.store(true, Ordering::Relaxed);
            tracing::debug!(
                quant = %autotune_quant,
                embedding_dim = config.embedding_dim,
                intermediate_dim = config.intermediate_dim,
                "Skipping f16in batch-route autotune: no valid benchmark shapes"
            );
            return;
        };
        let mut tuned = old_config;
        tuned.batch_f16in_small_n_threshold = n_threshold;
        tuned.batch_f16in_small_m_max = m_max;
        self.apply_runtime_policy(old_policy.with_dequant_dispatch(tuned));
        self.f16in_route_tuned.store(true, Ordering::Relaxed);
        tracing::info!(
            old_n_threshold = old_config.batch_f16in_small_n_threshold,
            old_m_max = old_config.batch_f16in_small_m_max,
            tuned_n_threshold = n_threshold,
            tuned_m_max = m_max,
            quant = %autotune_quant,
            "Autotuned f16in batch kernel routing"
        );
    }

    /// Access GPU batch scratch buffers (must call init_batch_scratches first).
    pub fn batch_scratches(&self) -> std::sync::MutexGuard<'_, Option<GpuBatchScratchBuffers>> {
        self.batch_scratches.lock().unwrap()
    }

    pub(crate) fn moe_batch_scratch_view(&self) -> anyhow::Result<MoeBatchScratchView> {
        let bs_guard = self.batch_scratches.lock().unwrap();
        let bs = bs_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("moe_ffn_gpu_resident: missing batch scratches"))?;
        MoeBatchScratchView::from_batch_scratches(bs)
    }

    pub(crate) fn with_qwen35_qkv_handoff_scratch<R>(
        &self,
        n_tokens: usize,
        conv_dim: usize,
        inner_size: usize,
        time_step_rank: usize,
        f: impl FnOnce(&mut Qwen35MetalQkvHandoffScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35QkvHandoffScratchBufferKey {
            n_tokens,
            conv_dim,
            inner_size,
            time_step_rank,
        };
        let mut cache = self.qwen35_qkv_handoff_scratch_buffers.lock().unwrap();
        let scratch = cache.entry(key).or_insert_with(|| {
            Qwen35MetalQkvHandoffScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 recurrent handoff scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn with_qwen35_recurrent_projection_scratch<R>(
        &self,
        n_tokens: usize,
        conv_dim: usize,
        inner_size: usize,
        time_step_rank: usize,
        f: impl FnOnce(&mut Qwen35MetalRecurrentProjectionScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35QkvHandoffScratchBufferKey {
            n_tokens,
            conv_dim,
            inner_size,
            time_step_rank,
        };
        let mut cache = self
            .qwen35_recurrent_projection_scratch_buffers
            .lock()
            .unwrap();
        let scratch = cache.entry(key).or_insert_with(|| {
            Qwen35MetalRecurrentProjectionScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 recurrent projection scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn with_qwen35_batch_projection_scratch<R>(
        &self,
        n_tokens: usize,
        output_dims: &[usize],
        f: impl FnOnce(&mut Qwen35MetalBatchProjectionScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35BatchProjectionScratchBufferKey {
            n_tokens,
            output_dims: output_dims.to_vec(),
        };
        let mut cache = self.qwen35_batch_projection_scratch_buffers.lock().unwrap();
        let scratch = cache.entry(key.clone()).or_insert_with(|| {
            Qwen35MetalBatchProjectionScratchBuffers::new(&self.device, &key)
                .expect("Failed to allocate qwen35 batch projection scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn with_qwen35_batch_logits_scratch<R>(
        &self,
        hidden_len: usize,
        logits_len: usize,
        f: impl FnOnce(&mut Qwen35MetalBatchLogitsScratchBuffers) -> R,
    ) -> R {
        let key = Qwen35BatchLogitsScratchBufferKey {
            hidden_len,
            logits_len,
        };
        let mut cache = self.qwen35_batch_logits_scratch_buffers.lock().unwrap();
        let scratch = cache.entry(key).or_insert_with(|| {
            Qwen35MetalBatchLogitsScratchBuffers::new(&self.device, key)
                .expect("Failed to allocate qwen35 batch logits scratch buffers")
        });
        f(scratch)
    }

    pub(crate) fn sync_qwen35_slot_buffers_from_kv(
        &self,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_idx: usize,
    ) -> Qwen35SlotBufferSyncOutcome {
        // Fast path: GPU-resident buffers live in Qwen35Kv directly — no sync needed.
        if qwen_kv.has_gpu_recurrent_state() {
            let mut outcome = Qwen35SlotBufferSyncOutcome::default();
            outcome.note_backend_carryover();
            return outcome;
        }
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();
        let slot_key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        };
        let recurrent_generation = qwen_kv.recurrent_state_generation(slot_idx, layer_idx);
        let mut outcome = Qwen35SlotBufferSyncOutcome::default();
        let mut slot_cache = self.qwen35_recurrent_slot_buffers.lock().unwrap();
        let slot_buffers = slot_cache.entry(slot_key).or_insert_with(|| {
            Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
        });
        let kv_identity = qwen_kv as *const crate::kv::Qwen35Kv as usize;
        if slot_buffers.source_kv_identity != Some(kv_identity) {
            slot_buffers.conv_synced_generation = None;
            slot_buffers.recurrent_synced_generation = None;
        }
        let conv_generation = qwen_kv.conv_state_generation(slot_idx, layer_idx);
        if !qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx) {
            if slot_buffers.conv_synced_generation != Some(conv_generation) {
                if qwen_kv.conv_state_pristine_zero(slot_idx, layer_idx) {
                    unsafe {
                        slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                            .fill(0.0);
                    }
                    outcome.note_backend_zero_init();
                } else {
                    unsafe {
                        slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                            .copy_from_slice(qwen_kv.conv_state_for_slot(slot_idx, layer_idx));
                    }
                    outcome.note_cpu_materialization();
                }
                slot_buffers.conv_synced_generation = Some(conv_generation);
            }
        } else if slot_buffers.conv_synced_generation != Some(conv_generation) {
            // Backend-owned but GPU slot buffer has a different generation.
            // This can happen when GPU prefill sets backend-owned state and
            // then the slot/layer is reused. Accept the GPU version as
            // authoritative and update the tracked generation.
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.conv_synced_generation,
                cpu_gen = conv_generation,
                "qwen35 conv state generation mismatch — accepting GPU version"
            );
            slot_buffers.conv_synced_generation = Some(conv_generation);
        } else {
            outcome.note_backend_carryover();
        }

        if !qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx) {
            if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
                if qwen_kv.recurrent_state_pristine_zero(slot_idx, layer_idx) {
                    unsafe {
                        slot_buffers.recurrent_state.as_mut_slice::<f32>()
                            [..recurrent_state_stride]
                            .fill(0.0);
                    }
                    outcome.note_backend_zero_init();
                } else {
                    unsafe {
                        slot_buffers.recurrent_state.as_mut_slice::<f32>()
                            [..recurrent_state_stride]
                            .copy_from_slice(qwen_kv.recurrent_state_for_slot(slot_idx, layer_idx));
                    }
                    outcome.note_cpu_materialization();
                }
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
            }
        } else if slot_buffers.recurrent_synced_generation != Some(recurrent_generation) {
            tracing::debug!(
                slot_idx,
                layer_idx,
                gpu_gen = ?slot_buffers.recurrent_synced_generation,
                cpu_gen = recurrent_generation,
                "qwen35 recurrent state generation mismatch — accepting GPU version"
            );
            slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
        } else {
            outcome.note_backend_carryover();
        }
        slot_buffers.source_kv_identity = Some(kv_identity);
        outcome
    }

    /// Provide mutable access to GPU slot buffers for a recurrent layer.
    ///
    /// When `qwen_kv` is provided and has GPU-resident state, uses those
    /// buffers directly (no mutex, no cache lookup). Otherwise falls back
    /// to the internal slot buffer cache.
    pub(crate) fn with_qwen35_recurrent_slot_buffer_for_kv<R>(
        &self,
        qwen_kv: &crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
        f: impl FnOnce(&mut Qwen35MetalSlotBuffers) -> R,
    ) -> R {
        if let Some((conv_buf, rec_buf)) = qwen_kv.gpu_recurrent_buffers(slot_idx, layer_idx) {
            let mut wrapper = Qwen35MetalSlotBuffers {
                conv_state: conv_buf.clone(),
                recurrent_state: rec_buf.clone(),
                conv_synced_generation: None,
                recurrent_synced_generation: None,
                source_kv_identity: None,
            };
            return f(&mut wrapper);
        }
        self.with_qwen35_recurrent_slot_buffer(
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
            f,
        )
    }

    pub(crate) fn with_qwen35_recurrent_slot_buffer<R>(
        &self,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
        f: impl FnOnce(&mut Qwen35MetalSlotBuffers) -> R,
    ) -> R {
        let key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        };
        let mut cache = self.qwen35_recurrent_slot_buffers.lock().unwrap();
        let slot_buffers = cache.entry(key).or_insert_with(|| {
            Qwen35MetalSlotBuffers::new(&self.device, conv_state_stride, recurrent_state_stride)
                .expect("Failed to allocate qwen35 recurrent Metal slot buffers")
        });
        f(slot_buffers)
    }

    /// Materialize GPU-resident recurrent state for a slot/layer back to CPU.
    ///
    /// If the GPU holds the latest version of conv or recurrent state (backend-owned),
    /// copies it back to the Qwen35Kv CPU buffers so CPU-side decode can proceed.
    /// No-op if state is already CPU-fresh.
    pub(crate) fn materialize_qwen35_slot_state_to_cpu(
        &self,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_idx: usize,
        conv_state_stride: usize,
        recurrent_state_stride: usize,
    ) {
        let key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx,
            conv_state_stride,
            recurrent_state_stride,
        };
        let cache = self.qwen35_recurrent_slot_buffers.lock().unwrap();
        let Some(slot_buffers) = cache.get(&key) else {
            return; // No GPU buffers allocated for this slot/layer
        };

        if qwen_kv.conv_state_cpu_stale(slot_idx, layer_idx) {
            if let Some(generation) = slot_buffers.conv_synced_generation {
                let conv_data =
                    unsafe { &slot_buffers.conv_state.as_slice::<f32>()[..conv_state_stride] };
                qwen_kv.sync_conv_state_from_backend(slot_idx, layer_idx, conv_data, generation);
            }
        }
        if qwen_kv.recurrent_state_cpu_stale(slot_idx, layer_idx) {
            if let Some(generation) = slot_buffers.recurrent_synced_generation {
                let rec_data = unsafe {
                    &slot_buffers.recurrent_state.as_slice::<f32>()[..recurrent_state_stride]
                };
                qwen_kv
                    .sync_recurrent_state_from_backend(slot_idx, layer_idx, rec_data, generation);
            }
        }
    }

    /// Get or create a cached MetalBuffer for an f32 weight slice.
    /// Keyed by the data pointer address (stable for mmap'd weights).
    pub fn get_f32_weight_buffer(
        &self,
        data: &[f32],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = data.as_ptr() as usize;
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_f32(&self.device, data));
        cache
    }

    /// Get or create a cached MetalBuffer for raw quantized weight data.
    /// Uses the same cache as f32 weights (keyed by pointer address).
    pub fn get_quant_weight_buffer(
        &self,
        data: &[u8],
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        let key = data.as_ptr() as usize;
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, data));
        cache
    }

    /// Ensure an f32 weight slice is cached as a MetalBuffer.
    /// Returns the cache key (pointer address) for later lookup via `lock_weight_cache`.
    pub fn ensure_f32_cached(&self, data: &[f32]) -> usize {
        let key = data.as_ptr() as usize;
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_f32(&self.device, data));
        key
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_quant_batch_matmul(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
        dtype: GgmlType,
    ) -> anyhow::Result<()> {
        match dtype {
            GgmlType::F32 => self
                .matmul
                .encode_matmul(encoder, weights, input, output, m, n, k),
            GgmlType::Q4K => self
                .dequant
                .encode_fused_batch_q4_k(encoder, weights, input, output, m, n, k),
            GgmlType::Q5K => self
                .dequant
                .encode_fused_batch_q5_k(encoder, weights, input, output, m, n, k),
            GgmlType::Q6K => self
                .dequant
                .encode_fused_batch_q6_k(encoder, weights, input, output, m, n, k),
            GgmlType::Q8_0 => self
                .dequant
                .encode_fused_batch_q8_0(encoder, weights, input, output, m, n, k),
            _ => anyhow::bail!("unsupported Metal MoE batch matmul dtype: {dtype:?}"),
        }
        Ok(())
    }

    fn validate_moe_mul_mat_id_dtype(dtype: GgmlType, role: &str) -> anyhow::Result<()> {
        if matches!(dtype, GgmlType::Q4K | GgmlType::Q5K) {
            Ok(())
        } else {
            anyhow::bail!(
                "Metal MoE mul_mat_id path only supports Q4_K/Q5_K {role} weights today; got {dtype:?}",
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_moe_mul_mat_id(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        weights: &MetalBuffer,
        input: &MetalBuffer,
        tpe: &MetalBuffer,
        hids: &MetalBuffer,
        output: &MetalBuffer,
        dtype: GgmlType,
        m: u32,
        k: u32,
        n_tokens: u32,
        n_expert_used: u32,
        n_expert: u32,
        weight_stride: u32,
        active_experts: &MetalBuffer,
        n_active_experts: u32,
    ) -> anyhow::Result<()> {
        match dtype {
            GgmlType::Q4K => self.dequant.encode_moe_mul_mat_id_q4_k(
                encoder,
                weights,
                input,
                tpe,
                hids,
                output,
                m,
                k,
                n_tokens,
                n_expert_used,
                n_expert,
                weight_stride,
                active_experts,
                n_active_experts,
            ),
            GgmlType::Q5K => self.dequant.encode_moe_mul_mat_id_q5_k(
                encoder,
                weights,
                input,
                tpe,
                hids,
                output,
                m,
                k,
                n_tokens,
                n_expert_used,
                n_expert,
                weight_stride,
                active_experts,
                n_active_experts,
            ),
            _ => anyhow::bail!(
                "unsupported Metal MoE mul_mat_id dtype for routed experts: {dtype:?}"
            ),
        }
        Ok(())
    }

    fn moe_blocks_per_expert(
        stride_bytes: usize,
        dtype: GgmlType,
        role: &str,
    ) -> anyhow::Result<u32> {
        Self::validate_moe_mul_mat_id_dtype(dtype, role)?;
        let bytes_per_block = dtype.bytes_per_block();
        if !stride_bytes.is_multiple_of(bytes_per_block) {
            anyhow::bail!(
                "Metal MoE {role} stride {stride_bytes} is not aligned to {:?} block size {bytes_per_block}",
                dtype,
            );
        }
        Ok((stride_bytes / bytes_per_block) as u32)
    }

    /// Execute MoE FFN on GPU-resident hidden buffer (no upload/download).
    ///
    /// Operates directly on `hidden_gpu` (MetalBuffer from bs.hidden).
    /// Encodes: RMSNorm → router → [sync for CPU top-k] → map0 + mul_mat_id ×3 + SiLU + scatter → residual add.
    /// The hidden buffer is updated in-place.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_gpu_resident_cached(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        hidden_gpu: &MetalBuffer,
        ffn_norm_w_buf: &MetalBuffer,
        router_buf: &MetalBuffer,
        router_dtype: GgmlType,
        gate_buf: &MetalBuffer,
        gate_dtype: GgmlType,
        up_buf: &MetalBuffer,
        up_dtype: GgmlType,
        down_buf: &MetalBuffer,
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        eps: f32,
        shared_expert: Option<&SharedExpertCachedBuffers<'_>>,
        explicit_barriers: bool,
    ) -> anyhow::Result<()> {
        let scratch = self.moe_batch_scratch_view()?;
        self.encode_moe_ffn_gpu_resident_cached_with_scratch(
            encoder,
            scratch,
            hidden_gpu,
            ffn_norm_w_buf,
            router_buf,
            router_dtype,
            gate_buf,
            gate_dtype,
            up_buf,
            up_dtype,
            down_buf,
            down_dtype,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            gate_stride,
            up_stride,
            down_stride,
            eps,
            shared_expert,
            explicit_barriers,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_gpu_resident_cached_with_scratch(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        scratch: MoeBatchScratchView,
        hidden_gpu: &MetalBuffer,
        ffn_norm_w_buf: &MetalBuffer,
        router_buf: &MetalBuffer,
        router_dtype: GgmlType,
        gate_buf: &MetalBuffer,
        gate_dtype: GgmlType,
        up_buf: &MetalBuffer,
        up_dtype: GgmlType,
        down_buf: &MetalBuffer,
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        eps: f32,
        shared_expert: Option<&SharedExpertCachedBuffers<'_>>,
        explicit_barriers: bool,
    ) -> anyhow::Result<()> {
        let barrier = |enc: &ax_engine_metal::MetalEncoder| {
            if explicit_barriers {
                ax_engine_metal::barrier_buffers(enc);
            }
        };
        let blocks_per_expert_gate =
            Self::moe_blocks_per_expert(gate_stride, gate_dtype, "expert gate")?;
        let blocks_per_expert_up = Self::moe_blocks_per_expert(up_stride, up_dtype, "expert up")?;
        let blocks_per_expert_down =
            Self::moe_blocks_per_expert(down_stride, down_dtype, "expert down")?;
        let norm_gpu = unsafe { &*scratch.norm };
        let router_gpu = unsafe { &*scratch.router };
        let accum_gpu = unsafe { &mut *(scratch.accum as *mut MetalBuffer) };
        let gate_out_gpu = unsafe { &*scratch.gate_out };
        let up_out_gpu = unsafe { &*scratch.up_out };
        let down_out_gpu = unsafe { &*scratch.down_out };
        let tpe_gpu = unsafe { &*scratch.tpe };
        let hids_gpu = unsafe { &*scratch.hids };
        let active_buf = unsafe { &mut *scratch.active };
        let ids_gpu = unsafe { &mut *scratch.ids };
        let weights_gpu = unsafe { &mut *scratch.weights };
        let tok_idx_buf = unsafe { &mut *scratch.tok_idx };

        unsafe { accum_gpu.as_mut_slice::<f32>().fill(0.0) };

        let nt = n_tokens as u32;
        let ne = n_expert as u32;
        let neu = n_expert_used as u32;
        let m_inter = expert_inter_dim as u32;
        let m_dim = dim as u32;

        self.elementwise.encode_rms_norm_out_batch(
            encoder,
            hidden_gpu,
            ffn_norm_w_buf,
            norm_gpu,
            dim as u32,
            n_tokens as u32,
            eps,
        );
        barrier(encoder);

        self.encode_quant_batch_matmul(
            encoder,
            router_buf,
            norm_gpu,
            router_gpu,
            n_expert as u32,
            n_tokens as u32,
            dim as u32,
            router_dtype,
        )?;
        barrier(encoder);

        self.elementwise.encode_moe_softmax_topk(
            encoder,
            router_gpu,
            ids_gpu,
            weights_gpu,
            n_tokens as u32,
            n_expert as u32,
            n_expert_used as u32,
        );
        barrier(encoder);

        self.dequant.encode_moe_map0(
            encoder,
            ids_gpu,
            tpe_gpu,
            hids_gpu,
            n_tokens as u32,
            n_expert_used as u32,
            n_expert as u32,
        );
        barrier(encoder);

        self.encode_moe_mul_mat_id(
            encoder,
            gate_buf,
            norm_gpu,
            tpe_gpu,
            hids_gpu,
            gate_out_gpu,
            gate_dtype,
            m_inter,
            m_dim,
            nt,
            neu,
            ne,
            blocks_per_expert_gate,
            active_buf,
            ne,
        )?;
        self.encode_moe_mul_mat_id(
            encoder,
            up_buf,
            norm_gpu,
            tpe_gpu,
            hids_gpu,
            up_out_gpu,
            up_dtype,
            m_inter,
            m_dim,
            nt,
            neu,
            ne,
            blocks_per_expert_up,
            active_buf,
            ne,
        )?;
        barrier(encoder);

        self.elementwise.encode_silu_elementwise_mul_batch(
            encoder,
            gate_out_gpu,
            up_out_gpu,
            m_inter,
            nt * neu,
        );
        barrier(encoder);

        self.encode_moe_mul_mat_id(
            encoder,
            down_buf,
            gate_out_gpu,
            tpe_gpu,
            hids_gpu,
            down_out_gpu,
            down_dtype,
            m_dim,
            m_inter,
            nt,
            neu,
            ne,
            blocks_per_expert_down,
            active_buf,
            ne,
        )?;
        barrier(encoder);

        self.elementwise.encode_moe_weighted_scatter(
            encoder,
            down_out_gpu,
            tok_idx_buf,
            weights_gpu,
            accum_gpu,
            m_dim,
            nt * neu,
        );
        barrier(encoder);

        if let Some(se) = shared_expert {
            let se_inter = se.inter_dim as u32;

            self.encode_quant_batch_matmul(
                encoder,
                se.gate,
                norm_gpu,
                gate_out_gpu,
                se_inter,
                nt,
                m_dim,
                se.dtype,
            )?;
            self.encode_quant_batch_matmul(
                encoder, se.up, norm_gpu, up_out_gpu, se_inter, nt, m_dim, se.dtype,
            )?;
            barrier(encoder);

            self.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                gate_out_gpu,
                up_out_gpu,
                se_inter,
                nt,
            );
            barrier(encoder);

            self.encode_quant_batch_matmul(
                encoder,
                se.down,
                gate_out_gpu,
                down_out_gpu,
                m_dim,
                nt,
                se_inter,
                se.dtype,
            )?;
            barrier(encoder);

            if let Some(gate_inp) = se.gate_inp {
                let gate_inp_dtype = se
                    .gate_inp_dtype
                    .ok_or_else(|| anyhow::anyhow!("missing shared expert gate input dtype"))?;
                self.encode_quant_batch_matmul(
                    encoder,
                    gate_inp,
                    norm_gpu,
                    router_gpu,
                    se.gate_inp_rows as u32,
                    nt,
                    m_dim,
                    gate_inp_dtype,
                )?;
                barrier(encoder);
                if se.gate_inp_rows == dim {
                    self.elementwise.encode_sigmoid_elementwise_mul(
                        encoder,
                        router_gpu,
                        down_out_gpu,
                        m_dim * nt,
                    );
                } else {
                    anyhow::ensure!(
                        se.gate_inp_rows == 1,
                        "shared expert gate width {} is not supported by the resident Metal MoE path (expected 1 or {})",
                        se.gate_inp_rows,
                        dim,
                    );
                    self.elementwise
                        .encode_sigmoid_inplace(encoder, router_gpu, nt);
                    barrier(encoder);
                    self.gdn.encode_broadcast_mul(
                        encoder,
                        down_out_gpu,
                        router_gpu,
                        down_out_gpu,
                        m_dim,
                        m_dim * nt,
                    );
                }
                barrier(encoder);
            }

            self.elementwise.encode_elementwise_add_batch(
                encoder,
                accum_gpu,
                down_out_gpu,
                m_dim,
                nt,
            );
            barrier(encoder);
        }

        self.elementwise
            .encode_elementwise_add_batch(encoder, hidden_gpu, accum_gpu, m_dim, nt);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_ffn_gpu_resident(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        hidden_gpu: &MetalBuffer,
        ffn_norm_w_buf: &MetalBuffer,
        router_raw: &[u8],
        router_dtype: GgmlType,
        gate_exps_raw: &[u8],
        gate_dtype: GgmlType,
        up_exps_raw: &[u8],
        up_dtype: GgmlType,
        down_exps_raw: &[u8],
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        eps: f32,
        shared_expert: Option<&SharedExpertWeights<'_>>,
        explicit_barriers: bool,
    ) -> anyhow::Result<()> {
        let router_key = self.ensure_moe_quant_cached(router_raw);
        let gate_key = self.ensure_moe_quant_cached(gate_exps_raw);
        let up_key = self.ensure_moe_quant_cached(up_exps_raw);
        let down_key = self.ensure_moe_quant_cached(down_exps_raw);
        if let Some(se) = shared_expert {
            self.ensure_moe_quant_cached(se.gate_raw);
            self.ensure_moe_quant_cached(se.up_raw);
            self.ensure_moe_quant_cached(se.down_raw);
            if let Some(gate_inp_raw) = se.gate_inp_raw {
                self.ensure_moe_quant_cached(gate_inp_raw);
            }
        }

        let cache = self.moe_quant_cache.lock().unwrap();
        let router_buf = cache.get(&router_key).unwrap();
        let gate_buf = cache.get(&gate_key).unwrap();
        let up_buf = cache.get(&up_key).unwrap();
        let down_buf = cache.get(&down_key).unwrap();
        let shared_expert = shared_expert.map(|se| SharedExpertCachedBuffers {
            gate: cache.get(&(se.gate_raw.as_ptr() as usize)).unwrap(),
            up: cache.get(&(se.up_raw.as_ptr() as usize)).unwrap(),
            down: cache.get(&(se.down_raw.as_ptr() as usize)).unwrap(),
            gate_inp: se
                .gate_inp_raw
                .map(|gate_inp_raw| cache.get(&(gate_inp_raw.as_ptr() as usize)).unwrap()),
            gate_inp_dtype: se.gate_inp_dtype,
            dtype: se.dtype,
            inter_dim: se.inter_dim,
            gate_inp_rows: se.gate_inp_rows,
        });

        self.encode_moe_ffn_gpu_resident_cached(
            encoder,
            hidden_gpu,
            ffn_norm_w_buf,
            router_buf,
            router_dtype,
            gate_buf,
            gate_dtype,
            up_buf,
            up_dtype,
            down_buf,
            down_dtype,
            n_tokens,
            n_expert,
            n_expert_used,
            dim,
            expert_inter_dim,
            gate_stride,
            up_stride,
            down_stride,
            eps,
            shared_expert.as_ref(),
            explicit_barriers,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_ffn_gpu_resident(
        &self,
        hidden_gpu: &MetalBuffer,
        ffn_norm_w: &[f32],
        router_raw: &[u8],
        router_dtype: GgmlType,
        gate_exps_raw: &[u8],
        gate_dtype: GgmlType,
        up_exps_raw: &[u8],
        up_dtype: GgmlType,
        down_exps_raw: &[u8],
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        eps: f32,
        shared_expert: Option<&SharedExpertWeights<'_>>,
    ) -> anyhow::Result<()> {
        let norm_key = self.ensure_f32_cached(ffn_norm_w);
        let f32_cache = self.f32_weight_cache.lock().unwrap();
        let norm_w_buf = f32_cache.get(&norm_key).unwrap();
        self.device.execute_sync(|encoder| {
            self.encode_moe_ffn_gpu_resident(
                encoder,
                hidden_gpu,
                norm_w_buf,
                router_raw,
                router_dtype,
                gate_exps_raw,
                gate_dtype,
                up_exps_raw,
                up_dtype,
                down_exps_raw,
                down_dtype,
                n_tokens,
                n_expert,
                n_expert_used,
                dim,
                expert_inter_dim,
                gate_stride,
                up_stride,
                down_stride,
                eps,
                shared_expert,
                false,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_ffn_gpu_resident_cached(
        &self,
        hidden_gpu: &MetalBuffer,
        ffn_norm_w_buf: &MetalBuffer,
        router_buf: &MetalBuffer,
        router_dtype: GgmlType,
        gate_buf: &MetalBuffer,
        gate_dtype: GgmlType,
        up_buf: &MetalBuffer,
        up_dtype: GgmlType,
        down_buf: &MetalBuffer,
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        eps: f32,
        shared_expert: Option<&SharedExpertCachedBuffers<'_>>,
    ) -> anyhow::Result<()> {
        self.device.execute_sync(|encoder| {
            self.encode_moe_ffn_gpu_resident_cached(
                encoder,
                hidden_gpu,
                ffn_norm_w_buf,
                router_buf,
                router_dtype,
                gate_buf,
                gate_dtype,
                up_buf,
                up_dtype,
                down_buf,
                down_dtype,
                n_tokens,
                n_expert,
                n_expert_used,
                dim,
                expert_inter_dim,
                gate_stride,
                up_stride,
                down_stride,
                eps,
                shared_expert,
                false,
            )
        })
    }

    /// Execute MoE FFN via unified mul_mat_id kernel (2 dispatches per matmul).
    ///
    /// `expert_ids`: per-token expert assignments `[n_tokens × n_expert_used]` i32.
    /// `expert_weights`: per-token routing weights `[n_tokens × n_expert_used]` f32.
    /// `gate_exps_raw`/`up_exps_raw`/`down_exps_raw`: packed expert weights
    /// `[n_expert, dim, ...]` in Q4_K format.
    ///
    /// Output: `accum_out[n_tokens × dim]` f32, accumulated weighted expert outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_mul_mat_id_dispatch(
        &self,
        norm_buf: &[f32],
        accum_out: &mut [f32],
        expert_ids: &[i32],
        expert_weights_flat: &[f32],
        gate_exps_raw: &[u8],
        up_exps_raw: &[u8],
        down_exps_raw: &[u8],
        gate_dtype: GgmlType,
        up_dtype: GgmlType,
        down_dtype: GgmlType,
        n_tokens: usize,
        n_expert: usize,
        n_expert_used: usize,
        dim: usize,
        expert_inter_dim: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        shared_expert: Option<&SharedExpertWeights<'_>>,
    ) -> anyhow::Result<()> {
        let f4 = std::mem::size_of::<f32>();
        let i4 = std::mem::size_of::<i32>();
        let u4 = std::mem::size_of::<u32>();
        let dev = self.device.device();
        let blocks_per_expert_gate =
            Self::moe_blocks_per_expert(gate_stride, gate_dtype, "expert gate")?;
        let blocks_per_expert_up = Self::moe_blocks_per_expert(up_stride, up_dtype, "expert up")?;
        let blocks_per_expert_down =
            Self::moe_blocks_per_expert(down_stride, down_dtype, "expert down")?;

        // Cache full packed expert weight tensors (mmap-backed MetalBuffers).
        let gate_key = self.ensure_moe_quant_cached(gate_exps_raw);
        let up_key = self.ensure_moe_quant_cached(up_exps_raw);
        let down_key = self.ensure_moe_quant_cached(down_exps_raw);

        // Cache shared expert weights if present.
        if let Some(se) = shared_expert {
            self.ensure_moe_quant_cached(se.gate_raw);
            self.ensure_moe_quant_cached(se.up_raw);
            self.ensure_moe_quant_cached(se.down_raw);
            if let Some(gi) = se.gate_inp_raw {
                self.ensure_moe_quant_cached(gi);
            }
        }

        // Allocate GPU buffers.
        let mut input_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;
        let mut accum_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;
        let mut ids_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * i4)?;
        let mut weights_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * f4)?;
        let tpe_gpu = MetalBuffer::new(dev, n_expert * u4)?;
        // Extra space for active_experts list + atomic count (appended by map0 kernel).
        let hids_gpu = MetalBuffer::new(dev, n_expert * n_tokens * i4 + (n_expert + 1) * u4)?;
        // Intermediate: gate_up output [n_tokens * n_expert_used, expert_inter_dim]
        let gate_out_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * expert_inter_dim * f4)?;
        let up_out_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * expert_inter_dim * f4)?;
        // Down output: [n_tokens * n_expert_used, dim]
        let down_out_gpu = MetalBuffer::new(dev, n_tokens * n_expert_used * dim * f4)?;

        // Upload input data (UMA copy).
        unsafe {
            input_gpu.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(norm_buf);
            accum_gpu.as_mut_slice::<f32>()[..n_tokens * dim].fill(0.0);
            let ids_u8: &[u8] = std::slice::from_raw_parts(
                expert_ids.as_ptr() as *const u8,
                std::mem::size_of_val(expert_ids),
            );
            std::ptr::copy_nonoverlapping(
                ids_u8.as_ptr(),
                ids_gpu.as_mut_slice::<u8>().as_mut_ptr(),
                ids_u8.len(),
            );
            weights_gpu.as_mut_slice::<f32>()[..expert_weights_flat.len()]
                .copy_from_slice(expert_weights_flat);
        }

        // Pre-build token index buffer for weighted scatter (deterministic).
        let mut tok_idx_buf = MetalBuffer::new(dev, n_tokens * n_expert_used * u4)?;
        unsafe {
            let s = tok_idx_buf.as_mut_slice::<u32>();
            for (i, slot) in s.iter_mut().take(n_tokens * n_expert_used).enumerate() {
                *slot = (i / n_expert_used) as u32;
            }
        }

        let cache = self.moe_quant_cache.lock().unwrap();
        let gate_buf = cache.get(&gate_key).unwrap();
        let up_buf = cache.get(&up_key).unwrap();
        let down_buf = cache.get(&down_key).unwrap();

        let nt = n_tokens as u32;
        let ne = n_expert as u32;
        let neu = n_expert_used as u32;
        let m_inter = expert_inter_dim as u32;
        let m_dim = dim as u32;
        let k_dim = dim as u32;
        let k_inter = expert_inter_dim as u32;

        // Build active expert set on CPU from expert_ids (no GPU readback needed).
        let mut active_set = std::collections::BTreeSet::new();
        for &eid in expert_ids {
            if eid >= 0 {
                active_set.insert(eid as u32);
            }
        }
        let active_list: Vec<u32> = active_set.into_iter().collect();
        let n_active = active_list.len() as u32;

        let mut active_buf = MetalBuffer::new(dev, (n_active as usize).max(1) * u4)?;
        unsafe {
            active_buf.as_mut_slice::<u32>()[..active_list.len()].copy_from_slice(&active_list);
        }

        // Single CB: map0 + 3× mul_mat_id + SiLU + scatter.
        self.device.execute_sync(|encoder| {
            // Build routing index on GPU.
            self.dequant
                .encode_moe_map0(encoder, &ids_gpu, &tpe_gpu, &hids_gpu, nt, neu, ne);
            // Stage 2a: Gate matmul — compact grid (only active experts).
            self.encode_moe_mul_mat_id(
                encoder,
                gate_buf,
                &input_gpu,
                &tpe_gpu,
                &hids_gpu,
                &gate_out_gpu,
                gate_dtype,
                m_inter,
                k_dim,
                nt,
                neu,
                ne,
                blocks_per_expert_gate,
                &active_buf,
                n_active,
            )?;

            // Stage 2b: Up matmul.
            self.encode_moe_mul_mat_id(
                encoder,
                up_buf,
                &input_gpu,
                &tpe_gpu,
                &hids_gpu,
                &up_out_gpu,
                up_dtype,
                m_inter,
                k_dim,
                nt,
                neu,
                ne,
                blocks_per_expert_up,
                &active_buf,
                n_active,
            )?;

            // Stage 3: SiLU(gate) * up — in-place on gate_out.
            self.elementwise.encode_silu_elementwise_mul_batch(
                encoder,
                &gate_out_gpu,
                &up_out_gpu,
                m_inter,
                nt * neu,
            );

            // Stage 4: Down matmul — down_out = down_weights × gate_out (routed).
            // Note: down uses the SAME routing index (hids/tpe) but reads from
            // gate_out which is [n_tokens*n_expert_used, expert_inter_dim].
            // The down kernel needs input as [n_tokens, K] but gate_out is
            // [n_tokens*n_expert_used, expert_inter_dim] — each "token" in
            // gate_out is one (token, expert_slot) pair.
            // So we treat it as n_tokens_effective = n_tokens * n_expert_used,
            // but the down kernel routes by expert which expects the SAME hids.
            //
            // Actually, for down: the input is already per-(token, expert_slot),
            // stored flat at index hid. The down kernel should just do a regular
            // batched matmul (not routed) since the data is already separated.
            //
            // Simplification: use the per-expert dispatch for down matmul,
            // or use a non-routed batch matmul on the flat output.
            //
            // For now, use the fused batch approach for down:
            // down_out = down_weights[expert] × gate_out[hid]
            // This IS routed, using the same hids. The kernel reads input at
            // hid position which for gate_out means index hid in the flat array.
            // Since gate_out[hid] is already the correct (token, expert) data,
            // and the kernel routes by expert, this should work.
            self.encode_moe_mul_mat_id(
                encoder,
                down_buf,
                &gate_out_gpu,
                &tpe_gpu,
                &hids_gpu,
                &down_out_gpu,
                down_dtype,
                m_dim,
                k_inter,
                nt,
                neu,
                ne,
                blocks_per_expert_down,
                &active_buf,
                n_active,
            )?;

            // Stage 5: Weighted scatter (pre-built tok_idx_buf, same CB).
            self.elementwise.encode_moe_weighted_scatter(
                encoder,
                &down_out_gpu,
                &tok_idx_buf,
                &weights_gpu,
                &accum_gpu,
                m_dim,
                nt * neu,
            );

            // Stage 6: Shared expert (fused into same CB when provided).
            if let Some(se) = shared_expert {
                let se_inter = se.inter_dim as u32;
                let se_gw = cache.get(&(se.gate_raw.as_ptr() as usize)).unwrap();
                let se_uw = cache.get(&(se.up_raw.as_ptr() as usize)).unwrap();
                let se_dw = cache.get(&(se.down_raw.as_ptr() as usize)).unwrap();

                self.encode_quant_batch_matmul(
                    encoder,
                    se_gw,
                    &input_gpu,
                    &gate_out_gpu,
                    se_inter,
                    nt,
                    m_dim,
                    se.dtype,
                )?;
                self.encode_quant_batch_matmul(
                    encoder,
                    se_uw,
                    &input_gpu,
                    &up_out_gpu,
                    se_inter,
                    nt,
                    m_dim,
                    se.dtype,
                )?;
                self.elementwise.encode_silu_elementwise_mul_batch(
                    encoder,
                    &gate_out_gpu,
                    &up_out_gpu,
                    se_inter,
                    nt,
                );
                self.encode_quant_batch_matmul(
                    encoder,
                    se_dw,
                    &gate_out_gpu,
                    &down_out_gpu,
                    m_dim,
                    nt,
                    se_inter,
                    se.dtype,
                )?;
                // TODO: sigmoid gate (se.gate_inp_raw) — handled on CPU for now.
                self.elementwise.encode_elementwise_add_batch(
                    encoder,
                    &accum_gpu,
                    &down_out_gpu,
                    m_dim,
                    nt,
                );
            }

            Ok(())
        })?;

        drop(cache);

        // Download.
        unsafe {
            accum_out[..n_tokens * dim]
                .copy_from_slice(&accum_gpu.as_slice::<f32>()[..n_tokens * dim]);
        }
        Ok(())
    }

    /// Cache raw quantized weight data for single-CB MoE expert dispatch.
    /// Returns the cache key.
    pub fn ensure_moe_quant_cached(&self, data: &[u8]) -> usize {
        let key = data.as_ptr() as usize;
        let mut cache = self.moe_quant_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, data));
        key
    }

    /// Execute batched MoE expert FFN in a single command buffer.
    ///
    /// For each active expert: GPU gather → gate matmul → up matmul → SiLU →
    /// down matmul → weighted scatter. All encoded in ONE execute_sync.
    ///
    /// `expert_assignments[eid]` = `(indices: &[u32], weights: &[f32])` for each
    /// active expert. `norm_buf_gpu` is the input, `accum_gpu` is output.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn moe_expert_batch_single_cb(
        &self,
        norm_buf: &[f32],
        accum_out: &mut [f32],
        expert_data: &[(
            usize,    // n_assigned
            Vec<u32>, // token indices
            Vec<f32>, // routing weights
            &[u8],    // gate weight raw
            &[u8],    // up weight raw
            &[u8],    // down weight raw
        )],
        gate_dtype: GgmlType,
        up_dtype: GgmlType,
        down_dtype: GgmlType,
        n_tokens: usize,
        dim: usize,
        expert_inter_dim: usize,
    ) -> anyhow::Result<()> {
        let f4 = std::mem::size_of::<f32>();
        let u4 = std::mem::size_of::<u32>();
        let dev = self.device.device();

        // Pre-cache all expert weights.
        for &(_, _, _, gate_raw, up_raw, down_raw) in expert_data {
            self.ensure_moe_quant_cached(gate_raw);
            self.ensure_moe_quant_cached(up_raw);
            self.ensure_moe_quant_cached(down_raw);
        }

        // Allocate GPU buffers.
        let mut norm_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;
        let mut accum_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;
        let gather_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;
        let gate_gpu = MetalBuffer::new(dev, n_tokens * expert_inter_dim * f4)?;
        let up_gpu = MetalBuffer::new(dev, n_tokens * expert_inter_dim * f4)?;
        let down_gpu = MetalBuffer::new(dev, n_tokens * dim * f4)?;

        // Upload input, zero accumulator.
        unsafe {
            norm_gpu.as_mut_slice::<f32>()[..n_tokens * dim].copy_from_slice(norm_buf);
            accum_gpu.as_mut_slice::<f32>()[..n_tokens * dim].fill(0.0);
        }

        // Build per-expert GPU index/weight buffers.
        struct Egd {
            n: u32,
            idx_buf: MetalBuffer,
            wt_buf: MetalBuffer,
            gate_key: usize,
            up_key: usize,
            down_key: usize,
        }
        let mut egds = Vec::new();
        for &(n_assigned, ref indices, ref weights, gate_raw, up_raw, down_raw) in expert_data {
            let mut idx_buf = MetalBuffer::new(dev, n_assigned * u4)?;
            let mut wt_buf = MetalBuffer::new(dev, n_assigned * f4)?;
            unsafe {
                idx_buf.as_mut_slice::<u32>()[..n_assigned].copy_from_slice(indices);
                wt_buf.as_mut_slice::<f32>()[..n_assigned].copy_from_slice(weights);
            }
            egds.push(Egd {
                n: n_assigned as u32,
                idx_buf,
                wt_buf,
                gate_key: gate_raw.as_ptr() as usize,
                up_key: up_raw.as_ptr() as usize,
                down_key: down_raw.as_ptr() as usize,
            });
        }

        // Lock weight cache ONCE for the entire encode.
        let cache = self.moe_quant_cache.lock().unwrap();

        self.device.execute_sync(|encoder| {
            for egd in &egds {
                let n = egd.n;
                let m_inter = expert_inter_dim as u32;
                let m_dim = dim as u32;

                // Gather
                self.elementwise.encode_moe_gather(
                    encoder,
                    &norm_gpu,
                    &egd.idx_buf,
                    &gather_gpu,
                    m_dim,
                    n,
                );

                // Gate matmul
                let gw = cache.get(&egd.gate_key).unwrap();
                self.encode_quant_batch_matmul(
                    encoder,
                    gw,
                    &gather_gpu,
                    &gate_gpu,
                    m_inter,
                    n,
                    m_dim,
                    gate_dtype,
                )?;

                // Up matmul
                let uw = cache.get(&egd.up_key).unwrap();
                self.encode_quant_batch_matmul(
                    encoder,
                    uw,
                    &gather_gpu,
                    &up_gpu,
                    m_inter,
                    n,
                    m_dim,
                    up_dtype,
                )?;

                // SiLU
                self.elementwise
                    .encode_silu_elementwise_mul_batch(encoder, &gate_gpu, &up_gpu, m_inter, n);

                // Down matmul
                let dw = cache.get(&egd.down_key).unwrap();
                self.encode_quant_batch_matmul(
                    encoder, dw, &gate_gpu, &down_gpu, m_dim, n, m_inter, down_dtype,
                )?;

                // Weighted scatter
                self.elementwise.encode_moe_weighted_scatter(
                    encoder,
                    &down_gpu,
                    &egd.idx_buf,
                    &egd.wt_buf,
                    &accum_gpu,
                    m_dim,
                    n,
                );
            }
            Ok(())
        })?;

        drop(cache);

        // Download result.
        unsafe {
            accum_out[..n_tokens * dim]
                .copy_from_slice(&accum_gpu.as_slice::<f32>()[..n_tokens * dim]);
        }
        Ok(())
    }

    /// Ensure quantized weight data is cached as a MetalBuffer.
    /// Returns the cache key (pointer address) for later lookup via `lock_weight_cache`.
    pub fn ensure_quant_cached(&self, data: &[u8]) -> usize {
        let key = data.as_ptr() as usize;
        let mut cache = self.f32_weight_cache.lock().unwrap();
        cache
            .entry(key)
            .or_insert_with(|| create_mmap_weight_buffer_from_bytes(&self.device, data));
        key
    }

    /// Lock the weight buffer cache for batch access during command encoding.
    /// Use after `ensure_f32_cached`/`ensure_quant_cached` to get all buffer
    /// references at once without risk of deadlock.
    pub fn lock_weight_cache(&self) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        self.f32_weight_cache.lock().unwrap()
    }

    /// Lock the resident MoE quantized weight cache for batch/decode encoding.
    pub fn lock_moe_weight_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, FxHashMap<usize, MetalBuffer>> {
        self.moe_quant_cache.lock().unwrap()
    }

    /// Ensure fused QKV quantized data is cached as a MetalBuffer.
    ///
    /// The fused layout is row-concatenated [Wq; Wk; Wv], preserving the
    /// original quantized row encoding for each tensor.
    pub fn ensure_qkv_fused_quant_cached(&self, wq: &[u8], wk: &[u8], wv: &[u8]) {
        let key = (
            wq.as_ptr() as usize,
            wk.as_ptr() as usize,
            wv.as_ptr() as usize,
        );
        let mut cache = self.fused_qkv_weight_cache.lock().unwrap();
        cache.entry(key).or_insert_with(|| {
            let mut fused = Vec::with_capacity(wq.len() + wk.len() + wv.len());
            fused.extend_from_slice(wq);
            fused.extend_from_slice(wk);
            fused.extend_from_slice(wv);
            MetalBuffer::from_bytes(self.device.device(), &fused)
                .expect("Failed to create Metal buffer for fused QKV quantized weight")
        });
    }

    /// Lock fused QKV cache for batch command encoding.
    pub fn lock_fused_qkv_weight_cache(
        &self,
    ) -> std::sync::MutexGuard<'_, FxHashMap<(usize, usize, usize), MetalBuffer>> {
        self.fused_qkv_weight_cache.lock().unwrap()
    }

    fn quant_buf_key(weight_buf: &MetalBuffer) -> usize {
        weight_buf.contents().as_ptr() as usize
    }

    fn ensure_precomputed_q4k_f16_impl(
        &self,
        key: usize,
        quant_buf: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        if self
            .precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .contains_key(&key)
        {
            return Ok(());
        }

        let n_elems = (m as usize) * (k as usize);
        let n_blocks = n_elems / 256;
        let tmp_f32 = MetalBuffer::new(self.device.device(), n_elems * std::mem::size_of::<f32>())?;
        let dense_f16 = MetalBuffer::new(
            self.device.device(),
            n_elems * std::mem::size_of::<half::f16>(),
        )?;

        self.dequant
            .dequant_q4_k(&self.device, quant_buf, &tmp_f32, n_blocks as u32)?;
        self.device.execute_sync(|encoder| {
            self.elementwise
                .encode_cast_f32_to_f16(encoder, &tmp_f32, &dense_f16, m * k);
            Ok(())
        })?;

        self.precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .insert(key, dense_f16);
        tracing::info!(key, m, k, "Cached precomputed dense f16 weight (Q4_K)");
        Ok(())
    }

    /// Ensure a precomputed dense f16 buffer exists for a Q4_K quantized weight.
    pub fn ensure_precomputed_q4k_f16(
        &self,
        quant_buf: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let key = Self::quant_buf_key(quant_buf);
        self.ensure_precomputed_q4k_f16_impl(key, quant_buf, m, k)
    }

    /// Ensure a precomputed dense f16 buffer exists for a raw Q4_K tensor.
    pub fn ensure_precomputed_q4k_f16_from_raw(
        &self,
        quant_raw: &[u8],
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let key = self.ensure_quant_cached(quant_raw);
        let cache = self.f32_weight_cache.lock().unwrap();
        let quant_buf = cache
            .get(&key)
            .context("Missing Metal buffer for quantized Q4_K weight")?;
        let key = Self::quant_buf_key(quant_buf);
        self.ensure_precomputed_q4k_f16_impl(key, quant_buf, m, k)
    }

    fn ensure_precomputed_q6k_f16_impl(
        &self,
        key: usize,
        quant_buf: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        if self
            .precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .contains_key(&key)
        {
            return Ok(());
        }

        let n_elems = (m as usize) * (k as usize);
        let n_blocks = n_elems / 256;
        let tmp_f32 = MetalBuffer::new(self.device.device(), n_elems * std::mem::size_of::<f32>())?;
        let dense_f16 = MetalBuffer::new(
            self.device.device(),
            n_elems * std::mem::size_of::<half::f16>(),
        )?;

        self.dequant
            .dequant_q6_k(&self.device, quant_buf, &tmp_f32, n_blocks as u32)?;
        self.device.execute_sync(|encoder| {
            self.elementwise
                .encode_cast_f32_to_f16(encoder, &tmp_f32, &dense_f16, m * k);
            Ok(())
        })?;

        self.precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .insert(key, dense_f16);
        tracing::info!(key, m, k, "Cached precomputed dense f16 weight (Q6_K)");
        Ok(())
    }

    /// Ensure a precomputed dense f16 buffer exists for a raw Q6_K tensor.
    pub fn ensure_precomputed_q6k_f16_from_raw(
        &self,
        quant_raw: &[u8],
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let key = self.ensure_quant_cached(quant_raw);
        let cache = self.f32_weight_cache.lock().unwrap();
        let quant_buf = cache
            .get(&key)
            .context("Missing Metal buffer for quantized Q6_K weight")?;
        let key = Self::quant_buf_key(quant_buf);
        self.ensure_precomputed_q6k_f16_impl(key, quant_buf, m, k)
    }

    /// Ensure a precomputed dense f16 buffer exists for a raw Q8_0 tensor.
    pub fn ensure_precomputed_q8_0_f16_from_raw(
        &self,
        quant_raw: &[u8],
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let quant_key = self.ensure_quant_cached(quant_raw);
        let weight_cache = self.f32_weight_cache.lock().unwrap();
        let quant_buf = weight_cache
            .get(&quant_key)
            .context("Missing Metal buffer for quantized Q8_0 weight")?;
        let key = Self::quant_buf_key(quant_buf);
        if self
            .precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .contains_key(&key)
        {
            return Ok(());
        }

        let n_elems = (m as usize) * (k as usize);
        let mut dense_f32 = vec![0.0f32; n_elems];
        crate::quant::dequantize(GgmlType::Q8_0, quant_raw, &mut dense_f32);
        let dense_f16: Vec<half::f16> = dense_f32.into_iter().map(half::f16::from_f32).collect();
        let dense_f16_buf = MetalBuffer::from_slice(self.device.device(), &dense_f16)
            .context("Failed to create precomputed dense f16 buffer for Q8_0")?;

        self.precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .insert(key, dense_f16_buf);
        tracing::info!(key, m, k, "Cached precomputed dense f16 weight (Q8_0)");
        Ok(())
    }

    /// Encode precomputed dense-f16 batch matmul when available.
    ///
    /// Returns true if precomputed path was used.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_precomputed_batch_if_available(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        quant_buf: &MetalBuffer,
        input_f16: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        let key = Self::quant_buf_key(quant_buf);
        let cache = self.precomputed_f16_weight_cache.lock().unwrap();
        let Some(dense_f16) = cache.get(&key) else {
            return false;
        };
        self.dequant
            .encode_batch_matmul_btrans_f16_f32(encoder, dense_f16, input_f16, output, m, n, k);
        true
    }

    /// Ensure a precomputed dense f16 buffer exists for fused QKV Q4_K weight.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_precomputed_q4k_f16_fused_qkv(
        &self,
        wq_raw: &[u8],
        wk_raw: &[u8],
        wv_raw: &[u8],
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        self.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
        let fused_key = (
            wq_raw.as_ptr() as usize,
            wk_raw.as_ptr() as usize,
            wv_raw.as_ptr() as usize,
        );
        let cache = self.fused_qkv_weight_cache.lock().unwrap();
        let fused_buf = cache
            .get(&fused_key)
            .context("Missing Metal buffer for fused QKV quantized weight")?;
        let key = Self::quant_buf_key(fused_buf);
        self.ensure_precomputed_q4k_f16_impl(key, fused_buf, m, k)?;
        Ok(())
    }

    /// Ensure a precomputed dense f16 buffer exists for fused QKV Q8_0 weight.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_precomputed_q8_0_f16_fused_qkv(
        &self,
        wq_raw: &[u8],
        wk_raw: &[u8],
        wv_raw: &[u8],
        m: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        self.ensure_qkv_fused_quant_cached(wq_raw, wk_raw, wv_raw);
        let fused_key = (
            wq_raw.as_ptr() as usize,
            wk_raw.as_ptr() as usize,
            wv_raw.as_ptr() as usize,
        );
        let fused_cache = self.fused_qkv_weight_cache.lock().unwrap();
        let fused_buf = fused_cache
            .get(&fused_key)
            .context("Missing Metal buffer for fused QKV quantized weight")?;
        let key = Self::quant_buf_key(fused_buf);
        if self
            .precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .contains_key(&key)
        {
            return Ok(());
        }

        let mut fused_raw = Vec::with_capacity(wq_raw.len() + wk_raw.len() + wv_raw.len());
        fused_raw.extend_from_slice(wq_raw);
        fused_raw.extend_from_slice(wk_raw);
        fused_raw.extend_from_slice(wv_raw);
        let n_elems = (m as usize) * (k as usize);
        let mut dense_f32 = vec![0.0f32; n_elems];
        crate::quant::dequantize(GgmlType::Q8_0, &fused_raw, &mut dense_f32);
        let dense_f16: Vec<half::f16> = dense_f32.into_iter().map(half::f16::from_f32).collect();
        let dense_f16_buf = MetalBuffer::from_slice(self.device.device(), &dense_f16)
            .context("Failed to create precomputed dense f16 buffer for fused Q8_0 QKV")?;
        self.precomputed_f16_weight_cache
            .lock()
            .unwrap()
            .insert(key, dense_f16_buf);
        tracing::info!(
            key,
            m,
            k,
            "Cached precomputed dense f16 weight (fused Q8_0 QKV)"
        );
        Ok(())
    }

    /// Encode precomputed dense-f16 batch matmul for Q4_K when available.
    ///
    /// Returns true if precomputed path was used.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_precomputed_q4k_batch_if_available(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        quant_buf: &MetalBuffer,
        input_f16: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        let key = Self::quant_buf_key(quant_buf);
        let cache = self.precomputed_f16_weight_cache.lock().unwrap();
        let Some(dense_f16) = cache.get(&key) else {
            return false;
        };
        self.dequant
            .encode_batch_matmul_btrans_f16_f32(encoder, dense_f16, input_f16, output, m, n, k);
        true
    }

    /// Encode precomputed dense-f16 matvec for Q4_K when available.
    ///
    /// Returns true if precomputed path was used.
    #[allow(clippy::too_many_arguments)]
    /// Check whether a precomputed (dequantized to f16) version of this weight
    /// buffer exists in the cache, without dispatching anything.
    pub fn has_precomputed_weight(&self, quant_buf: &MetalBuffer) -> bool {
        let key = Self::quant_buf_key(quant_buf);
        let cache = self.precomputed_f16_weight_cache.lock().unwrap();
        cache.contains_key(&key)
    }

    pub fn encode_precomputed_q4k_matvec_if_available(
        &self,
        encoder: &ax_engine_metal::MetalEncoder,
        quant_buf: &MetalBuffer,
        input_f32: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
        k: u32,
    ) -> bool {
        let key = Self::quant_buf_key(quant_buf);
        let cache = self.precomputed_f16_weight_cache.lock().unwrap();
        let Some(dense_f16) = cache.get(&key) else {
            return false;
        };
        self.dequant
            .encode_fused_matvec_dense_f16(encoder, dense_f16, input_f32, output, m, k);
        true
    }

    // v2: GpuKv is owned by LlamaModel via ModelKv::Gpu. No init_gpu_kv or gpu_kv() here.
}

/// Hybrid backend: Metal GPU for all matmuls, CPU fallback for unsupported ops.
///
/// Uses Metal fused dequant+matvec kernels for both decode (N=1) and prefill (N>1).
/// On Apple Silicon UMA, GPU memory bandwidth (~200GB/s) is significantly higher
/// than CPU (~50GB/s via NEON), making GPU decode faster even for single-token matvec.
pub struct HybridBackend {
    #[allow(dead_code)]
    cpu: super::cpu::CpuBackend,
    metal: MetalBackend,
    /// Decode strategy for Hybrid backend.
    ///
    /// `true`: route decode (n=1) to CPU.
    /// `false`: route decode to Metal (default fast path).
    decode_on_cpu: bool,
}

impl HybridBackend {
    /// Initialize hybrid backend (creates Metal device + CPU fallback).
    pub fn new() -> anyhow::Result<Self> {
        let metal = MetalBackend::new()?;
        let decode_on_cpu = std::env::var("AX_HYBRID_DECODE")
            .map(|v| v.eq_ignore_ascii_case("cpu"))
            .unwrap_or(false);
        let decode_backend = if decode_on_cpu { "CPU" } else { "Metal" };
        tracing::info!("HybridBackend initialized (decode={decode_backend}, prefill=Metal)");
        Ok(Self {
            cpu: super::cpu::CpuBackend,
            metal,
            decode_on_cpu,
        })
    }
}

impl Backend for HybridBackend {
    fn configure_for_fingerprint(&self, fingerprint: &ModelFingerprint) -> anyhow::Result<()> {
        self.metal.configure_for_fingerprint(fingerprint)
    }

    fn configure_for_model(
        &self,
        model_name: &str,
        quant: &str,
        architecture: &str,
    ) -> anyhow::Result<()> {
        self.metal
            .configure_for_model(model_name, quant, architecture)
    }

    fn runtime_policy(&self) -> Option<RuntimePolicy> {
        self.metal.runtime_policy()
    }

    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        if self.decode_on_cpu && n == 1 {
            self.cpu.matmul(a, b, c, m, n, k);
        } else {
            self.metal.matmul(a, b, c, m, n, k);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant_matmul(
        &self,
        a_quant: &[u8],
        dtype: GgmlType,
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        // Decode path can be forced to CPU via env; default remains Metal.
        // Prefill path remains on Metal for throughput.
        if self.decode_on_cpu && n == 1 {
            self.cpu.dequant_matmul(a_quant, dtype, b, c, m, n, k);
        } else {
            self.metal.dequant_matmul(a_quant, dtype, b, c, m, n, k);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant_matmul_token_major(
        &self,
        a_quant: &[u8],
        dtype: GgmlType,
        input_token_major: &[f32],
        output_token_major: &mut [f32],
        n_tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) {
        // Prefill (n>1) always routes to Metal for fused dequant batch kernels.
        self.metal.dequant_matmul_token_major(
            a_quant,
            dtype,
            input_token_major,
            output_token_major,
            n_tokens,
            out_dim,
            in_dim,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_dequant_matvec(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        // Decode grouped matvecs are n=1 operations; route to CPU only when explicitly requested.
        if self.decode_on_cpu {
            self.cpu.batch_dequant_matvec(ops, x, k, outputs);
        } else {
            self.metal.batch_dequant_matvec(ops, x, k, outputs);
        }
    }

    fn safe_batch_dequant_matvec(
        &self,
        ops: &[(&[u8], GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        if self.decode_on_cpu {
            self.cpu.batch_dequant_matvec(ops, x, k, outputs);
        } else {
            self.metal.safe_batch_dequant_matvec(ops, x, k, outputs);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        self.metal
            .attention_prefill(q, k, v, output, n_tokens, n_heads, n_kv_heads, head_dim);
    }

    fn qwen35_gated_delta_sequence(
        &self,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        gate_batch: &[f32],
        beta_batch: &[f32],
        recurrent_state: &mut [f32],
        output_batch: &mut [f32],
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        if self.decode_on_cpu && n_tokens == 1 {
            self.cpu.qwen35_gated_delta_sequence(
                q_batch,
                k_batch,
                v_batch,
                gate_batch,
                beta_batch,
                recurrent_state,
                output_batch,
                n_tokens,
                n_heads,
                head_dim,
            );
        } else {
            self.metal.qwen35_gated_delta_sequence(
                q_batch,
                k_batch,
                v_batch,
                gate_batch,
                beta_batch,
                recurrent_state,
                output_batch,
                n_tokens,
                n_heads,
                head_dim,
            );
        }
    }

    fn qwen35_causal_conv_sequence(
        &self,
        input_batch: &[f32],
        kernel: &[f32],
        conv_state: &mut [f32],
        output_batch: &mut [f32],
        n_tokens: usize,
        conv_cache_len: usize,
        conv_dim: usize,
    ) {
        if n_tokens == 1 {
            self.cpu.qwen35_causal_conv_sequence(
                input_batch,
                kernel,
                conv_state,
                output_batch,
                n_tokens,
                conv_cache_len,
                conv_dim,
            );
        } else {
            self.metal.qwen35_causal_conv_sequence(
                input_batch,
                kernel,
                conv_state,
                output_batch,
                n_tokens,
                conv_cache_len,
                conv_dim,
            );
        }
    }

    fn qwen35_recurrent_sequence(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        state_batch: &mut super::Qwen35RecurrentStateBatch<'_>,
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        if self.decode_on_cpu && tokens_per_slot == 1 {
            self.cpu.qwen35_recurrent_sequence(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                state_batch,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        } else if tokens_per_slot == 1 {
            qwen35_recurrent_sequence_with_backend(
                self,
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                state_batch,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        } else {
            self.metal.qwen35_recurrent_sequence(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                state_batch,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        }
    }

    fn sync_qwen35_kv(&self, qwen_kv: &mut crate::kv::Qwen35Kv) {
        if self.decode_on_cpu {
            self.cpu.sync_qwen35_kv(qwen_kv);
        } else {
            self.metal.sync_qwen35_kv(qwen_kv);
        }
    }

    fn try_clone_qwen35_recurrent_slot(
        &self,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> bool {
        if self.decode_on_cpu {
            self.cpu
                .try_clone_qwen35_recurrent_slot(qwen_kv, src_slot_idx, dst_slot_idx)
        } else {
            self.metal
                .try_clone_qwen35_recurrent_slot_from_backend_owned(
                    qwen_kv,
                    src_slot_idx,
                    dst_slot_idx,
                )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence_for_kv(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_indices: &[usize],
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        if self.decode_on_cpu && tokens_per_slot == 1 {
            self.cpu.qwen35_recurrent_sequence_for_kv(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                qwen_kv,
                layer_idx,
                slot_indices,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        } else {
            self.metal.qwen35_recurrent_sequence_for_kv(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                qwen_kv,
                layer_idx,
                slot_indices,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        }
    }

    fn metal_ops(&self) -> Option<&MetalOps> {
        self.metal.metal_ops()
    }

    fn use_gpu_decode(&self) -> bool {
        !self.decode_on_cpu
    }
}

// ── v2 addition: HybridCpuDecodeBackend ─────────────────────────────────────
//
// In v1, HybridBackend had a `decode_on_cpu: bool` field set by the
// AX_HYBRID_DECODE env var. This caused the P1 bug: `forward_batch` checked
// `metal_ops()` (always Some for HybridBackend) but not `use_gpu_decode()`,
// so GPU batch prefill ran even when CPU decode was requested.
//
// v2 makes CPU decode a separate backend type. The forward pass checks
// `ctx.backend.use_gpu_decode()` which returns false for this type, preventing
// GPU batch prefill from running.

/// Hybrid backend with forced CPU decode.
///
/// Prefill uses Metal (N > 1). Single-token decode uses CPU (N = 1).
///
/// `use_gpu_decode()` returns `false`, which causes `forward_batch` to fall
/// through to serial prefill (writing KV to the CPU-resident `ModelKv::Cpu`).
/// This guarantees prefill and decode use the same KV storage.
pub struct HybridCpuDecodeBackend {
    cpu: super::cpu::CpuBackend,
    metal: MetalBackend,
}

impl HybridCpuDecodeBackend {
    pub fn new() -> anyhow::Result<Self> {
        tracing::info!("HybridCpuDecodeBackend initialized (prefill=Metal, decode=CPU)");
        Ok(Self {
            cpu: super::cpu::CpuBackend,
            metal: MetalBackend::new()?,
        })
    }
}

impl Backend for HybridCpuDecodeBackend {
    fn configure_for_fingerprint(&self, fingerprint: &ModelFingerprint) -> anyhow::Result<()> {
        self.metal.configure_for_fingerprint(fingerprint)
    }

    fn configure_for_model(
        &self,
        model_name: &str,
        quant: &str,
        architecture: &str,
    ) -> anyhow::Result<()> {
        self.metal
            .configure_for_model(model_name, quant, architecture)
    }

    fn runtime_policy(&self) -> Option<RuntimePolicy> {
        self.metal.runtime_policy()
    }

    fn matmul(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        if n == 1 {
            self.cpu.matmul(a, b, c, m, n, k);
        } else {
            self.metal.matmul(a, b, c, m, n, k);
        }
    }

    fn qwen35_gated_delta_sequence(
        &self,
        q_batch: &[f32],
        k_batch: &[f32],
        v_batch: &[f32],
        gate_batch: &[f32],
        beta_batch: &[f32],
        recurrent_state: &mut [f32],
        output_batch: &mut [f32],
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) {
        if n_tokens == 1 {
            self.cpu.qwen35_gated_delta_sequence(
                q_batch,
                k_batch,
                v_batch,
                gate_batch,
                beta_batch,
                recurrent_state,
                output_batch,
                n_tokens,
                n_heads,
                head_dim,
            );
        } else {
            self.metal.qwen35_gated_delta_sequence(
                q_batch,
                k_batch,
                v_batch,
                gate_batch,
                beta_batch,
                recurrent_state,
                output_batch,
                n_tokens,
                n_heads,
                head_dim,
            );
        }
    }

    fn qwen35_causal_conv_sequence(
        &self,
        input_batch: &[f32],
        kernel: &[f32],
        conv_state: &mut [f32],
        output_batch: &mut [f32],
        n_tokens: usize,
        conv_cache_len: usize,
        conv_dim: usize,
    ) {
        if n_tokens == 1 {
            self.cpu.qwen35_causal_conv_sequence(
                input_batch,
                kernel,
                conv_state,
                output_batch,
                n_tokens,
                conv_cache_len,
                conv_dim,
            );
        } else {
            self.metal.qwen35_causal_conv_sequence(
                input_batch,
                kernel,
                conv_state,
                output_batch,
                n_tokens,
                conv_cache_len,
                conv_dim,
            );
        }
    }

    fn qwen35_recurrent_sequence(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        state_batch: &mut super::Qwen35RecurrentStateBatch<'_>,
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        if tokens_per_slot == 1 {
            self.cpu.qwen35_recurrent_sequence(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                state_batch,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        } else {
            self.metal.qwen35_recurrent_sequence(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                state_batch,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_recurrent_sequence_for_kv(
        &self,
        qkv_batch: &[f32],
        beta_batch: &mut [f32],
        alpha_batch: &mut [f32],
        dt_bias: &[f32],
        a: &[f32],
        conv_kernel: &[f32],
        qwen_kv: &mut crate::kv::Qwen35Kv,
        layer_idx: usize,
        slot_indices: &[usize],
        output_batch: &mut [f32],
        tokens_per_slot: usize,
        cfg: crate::compute::gdn::Qwen35RecurrentConfig,
    ) {
        if tokens_per_slot == 1 {
            self.cpu.qwen35_recurrent_sequence_for_kv(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                qwen_kv,
                layer_idx,
                slot_indices,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        } else {
            self.metal.qwen35_recurrent_sequence_for_kv(
                qkv_batch,
                beta_batch,
                alpha_batch,
                dt_bias,
                a,
                conv_kernel,
                qwen_kv,
                layer_idx,
                slot_indices,
                output_batch,
                tokens_per_slot,
                cfg,
            );
        }
    }

    fn sync_qwen35_kv(&self, qwen_kv: &mut crate::kv::Qwen35Kv) {
        self.cpu.sync_qwen35_kv(qwen_kv);
    }

    fn try_clone_qwen35_recurrent_slot(
        &self,
        qwen_kv: &mut crate::kv::Qwen35Kv,
        src_slot_idx: usize,
        dst_slot_idx: usize,
    ) -> bool {
        self.cpu
            .try_clone_qwen35_recurrent_slot(qwen_kv, src_slot_idx, dst_slot_idx)
    }

    fn dequant_matmul(
        &self,
        a_quant: &[u8],
        dtype: crate::gguf::tensor::GgmlType,
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        if n == 1 {
            self.cpu.dequant_matmul(a_quant, dtype, b, c, m, n, k);
        } else {
            self.metal.dequant_matmul(a_quant, dtype, b, c, m, n, k);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant_matmul_token_major(
        &self,
        a_quant: &[u8],
        dtype: crate::gguf::tensor::GgmlType,
        input_token_major: &[f32],
        output_token_major: &mut [f32],
        n_tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) {
        // Prefill (n>1) routes to Metal for fused dequant batch kernels.
        self.metal.dequant_matmul_token_major(
            a_quant,
            dtype,
            input_token_major,
            output_token_major,
            n_tokens,
            out_dim,
            in_dim,
        );
    }

    fn batch_dequant_matvec(
        &self,
        ops: &[(&[u8], crate::gguf::tensor::GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        self.cpu.batch_dequant_matvec(ops, x, k, outputs);
    }

    fn safe_batch_dequant_matvec(
        &self,
        ops: &[(&[u8], crate::gguf::tensor::GgmlType, usize)],
        x: &[f32],
        k: usize,
        outputs: &mut [&mut [f32]],
    ) {
        self.cpu.batch_dequant_matvec(ops, x, k, outputs);
    }

    fn metal_ops(&self) -> Option<&MetalOps> {
        // Return Some so prefill layers can still use Metal compute (norms, RoPE, etc.)
        // even though single-token decode routes to CPU.
        self.metal.metal_ops()
    }

    /// Always false: signals forward_batch to use serial CPU prefill.
    fn use_gpu_decode(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Qwen35RecurrentStateBatch;

    fn lock_env_test() -> std::sync::MutexGuard<'static, ()> {
        static ENV_TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
            std::sync::OnceLock::new();
        ENV_TEST_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap()
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => unsafe { std::env::set_var(self.key, previous) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    fn test_fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            model_name: "Qwen3-8B".to_string(),
            architecture: "qwen3".to_string(),
            family: "qwen3".to_string(),
            size_label: "8b".to_string(),
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim: 4096,
            head_dim: 128,
            intermediate_dim: 14336,
            context_length: 4096,
            sliding_window_size: None,
            sliding_window_pattern: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            total_tensor_bytes: 0,
            predominant_quant: "Q4_K".to_string(),
            predominant_layer_quant: "Q4_K".to_string(),
            lm_head_quant: None,
            layer_quant_histogram: vec![],
            has_mixed_layer_quants: false,
            has_q4k_layer_weights: true,
            has_q5k_layer_weights: false,
            has_q6k_layer_weights: false,
            has_q8_layer_weights: false,
            has_f32_layer_weights: false,
        }
    }

    fn test_model_config(embedding_dim: u32, intermediate_dim: u32) -> ModelConfig {
        ModelConfig {
            architecture: "qwen3".into(),
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim,
            head_dim: 128,
            intermediate_dim,
            context_length: 4096,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            has_qkv_bias: true,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: None,
            n_expert_used: None,
            expert_intermediate_dim: None,
            qwen35_full_attention_interval: None,
            qwen35_ssm_conv_kernel: None,
            qwen35_ssm_inner_size: None,
            qwen35_ssm_state_size: None,
            qwen35_ssm_time_step_rank: None,
            qwen35_ssm_group_count: None,
        }
    }

    fn unique_temp_cache_dir(label: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ax-engine-metal-test-{label}-{}-{unique}",
            std::process::id()
        ))
    }

    #[test]
    fn test_validate_moe_mul_mat_id_dtype_accepts_q4k_and_q5k_rejects_q6k() {
        MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q4K, "expert").unwrap();
        MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q5K, "expert").unwrap();
        let err = MetalOps::validate_moe_mul_mat_id_dtype(GgmlType::Q6K, "expert")
            .expect_err("Q6_K routed expert weights should be rejected");
        assert!(err.to_string().contains("Q4_K/Q5_K expert"));
    }

    #[test]
    fn test_moe_blocks_per_expert_requires_aligned_q4k_stride() {
        assert_eq!(
            MetalOps::moe_blocks_per_expert(288, GgmlType::Q4K, "expert").unwrap(),
            2
        );

        let err = MetalOps::moe_blocks_per_expert(145, GgmlType::Q4K, "expert")
            .expect_err("misaligned Q4_K stride should fail");
        assert!(err.to_string().contains("stride 145"));
    }

    #[test]
    fn test_supports_q8_kv_requires_supported_alignment_and_head_dim() {
        let fingerprint = test_fingerprint();
        assert!(MetalOps::supports_q8_kv(&fingerprint));

        let mut unsupported_head_dim = fingerprint.clone();
        unsupported_head_dim.head_dim = 96;
        assert!(!MetalOps::supports_q8_kv(&unsupported_head_dim));

        let mut unsupported_stride = fingerprint;
        unsupported_stride.n_kv_heads = 0;
        assert!(!MetalOps::supports_q8_kv(&unsupported_stride));
    }

    #[test]
    fn test_quant_from_profile_label_parses_common_forms() {
        assert_eq!(quant_from_profile_label("Q4_K"), Some(GgmlType::Q4K));
        assert_eq!(quant_from_profile_label("Q4_K_M"), Some(GgmlType::Q4K));
        assert_eq!(quant_from_profile_label("q5-k"), Some(GgmlType::Q5K));
        assert_eq!(quant_from_profile_label("q5_k_s"), Some(GgmlType::Q5K));
        assert_eq!(quant_from_profile_label("q6k"), Some(GgmlType::Q6K));
        assert_eq!(quant_from_profile_label("Q8"), Some(GgmlType::Q8_0));
        assert_eq!(quant_from_profile_label("Q8_0"), Some(GgmlType::Q8_0));
        assert_eq!(quant_from_profile_label("Q8_1"), None);
        assert_eq!(quant_from_profile_label("unknown"), None);
    }

    #[test]
    fn test_preferred_batch_prefill_quant_accepts_normalized_layer_label_forms() {
        let backend = MetalBackend::new().unwrap();
        let mut fingerprint = test_fingerprint();
        fingerprint.predominant_layer_quant = "q5-k".to_string();
        fingerprint.has_q4k_layer_weights = false;
        fingerprint.has_q5k_layer_weights = false;

        assert_eq!(
            backend.ops.preferred_batch_prefill_quant(&fingerprint),
            Some(GgmlType::Q5K)
        );
    }

    #[test]
    fn test_preferred_batch_prefill_quant_falls_back_to_predominant_quant() {
        let backend = MetalBackend::new().unwrap();
        let mut fingerprint = test_fingerprint();
        fingerprint.predominant_layer_quant = "unknown".to_string();
        fingerprint.predominant_quant = "Q6K".to_string();
        fingerprint.has_q4k_layer_weights = false;
        fingerprint.has_q6k_layer_weights = false;

        assert_eq!(
            backend.ops.preferred_batch_prefill_quant(&fingerprint),
            Some(GgmlType::Q6K)
        );
    }

    #[test]
    fn test_decode_autotune_shapes_coalesces_duplicates_with_weights() {
        let fingerprint = test_fingerprint();
        let shapes = MetalOps::decode_autotune_shapes(&fingerprint);

        assert_eq!(shapes.len(), 4);
        assert_eq!(shapes.iter().map(|shape| shape.weight).sum::<u32>(), 7);

        let weight_for = |m: u32, k: u32| -> u32 {
            shapes
                .iter()
                .find(|shape| shape.m == m && shape.k == k)
                .map(|shape| shape.weight)
                .unwrap_or(0)
        };

        assert_eq!(weight_for(4096, 4096), 2);
        assert_eq!(weight_for(1024, 4096), 2);
        assert_eq!(weight_for(14336, 4096), 2);
        assert_eq!(weight_for(4096, 14336), 1);
    }

    #[test]
    fn test_decode_matvec_autotune_candidates_dedup_non_adjacent_profile_match() {
        let mut profile = KernelProfile::default();
        profile.decode_matvec.insert(
            "q4_k".to_string(),
            MatvecParams {
                threadgroup_size: 64,
                rows_per_simdgroup: 2,
                variant: Some(MatvecProfileVariant::Nr2),
            },
        );

        let (_, candidates) =
            MetalOps::decode_matvec_autotune_candidates(&profile, GgmlType::Q4K).unwrap();
        let deduped = MetalOps::dedup_matvec_candidates(candidates);

        assert_eq!(deduped.len(), 3);
        assert_eq!(
            deduped[0],
            MatvecParams {
                threadgroup_size: 64,
                rows_per_simdgroup: 2,
                variant: Some(MatvecProfileVariant::Nr2),
            }
        );
        assert!(
            deduped
                .iter()
                .any(|params| params.variant == Some(MatvecProfileVariant::Base))
        );
        assert!(
            deduped
                .iter()
                .any(|params| params.variant == Some(MatvecProfileVariant::Ilp4))
        );
    }

    #[test]
    fn test_decode_matvec_autotune_candidates_infer_q4_variant_from_legacy_override_shape() {
        let mut profile = KernelProfile::default();
        profile.decode_matvec.insert(
            "q4_k".to_string(),
            MatvecParams {
                threadgroup_size: 128,
                rows_per_simdgroup: 1,
                variant: None,
            },
        );

        let (_, candidates) =
            MetalOps::decode_matvec_autotune_candidates(&profile, GgmlType::Q4K).unwrap();
        assert_eq!(candidates[0].threadgroup_size, 128);
        assert_eq!(candidates[0].rows_per_simdgroup, 1);
        assert_eq!(candidates[0].variant, Some(MatvecProfileVariant::Base));
    }

    #[test]
    fn test_decode_matvec_autotune_candidates_infer_q8_variant_from_legacy_override_shape() {
        let mut profile = KernelProfile::default();
        profile.decode_matvec.insert(
            "q8_0".to_string(),
            MatvecParams {
                threadgroup_size: 64,
                rows_per_simdgroup: 1,
                variant: None,
            },
        );

        let (_, candidates) =
            MetalOps::decode_matvec_autotune_candidates(&profile, GgmlType::Q8_0).unwrap();
        assert_eq!(candidates[0].threadgroup_size, 64);
        assert_eq!(candidates[0].rows_per_simdgroup, 1);
        assert_eq!(candidates[0].variant, Some(MatvecProfileVariant::Ilp4));
    }

    #[test]
    fn test_tune_f16in_batch_route_returns_none_when_no_valid_shapes() {
        let backend = MetalBackend::new().unwrap();
        // Q4_K uses block size 256; k=130 yields no valid benchmark shapes.
        let tuned = backend.ops.tune_f16in_batch_route(GgmlType::Q4K, 130, 194);
        assert_eq!(tuned, None);
    }

    #[test]
    fn test_maybe_autotune_f16in_batch_route_stops_retrying_when_no_valid_shapes() {
        let _env_lock = lock_env_test();
        let _autotune = EnvVarGuard::set("AX_METAL_AUTOTUNE", "on");
        let backend = MetalBackend::new().unwrap();
        let config = test_model_config(130, 194);

        assert!(!backend.ops.f16in_route_tuned.load(Ordering::Relaxed));
        backend.ops.maybe_autotune_f16in_batch_route(&config);
        assert!(
            backend.ops.f16in_route_tuned.load(Ordering::Relaxed),
            "runtime batch-route autotune should stop retrying after deterministic no-shape result"
        );
    }

    #[test]
    fn test_estimated_attention_kv_bytes_respects_sparse_attention_layers() {
        let mut fingerprint = test_fingerprint();
        fingerprint.architecture = "qwen35".to_string();
        fingerprint.n_layers = 28;
        fingerprint.context_length = 8192;
        fingerprint.qwen35_full_attention_interval = Some(4);

        let estimated =
            MetalOps::estimated_attention_kv_bytes(&fingerprint, KvPrecisionPolicy::ForceQ8_0)
                .unwrap();
        let kv_stride = fingerprint.n_kv_heads as u64 * fingerprint.head_dim as u64;
        let row_bytes = (kv_stride / 32) * 34;
        let expected_layers = 7u64;
        let expected = expected_layers * fingerprint.context_length as u64 * row_bytes * 2;

        assert_eq!(
            MetalOps::estimated_attention_kv_layers(&fingerprint),
            expected_layers
        );
        assert_eq!(estimated, expected);
    }

    #[test]
    fn test_qwen35moe_scratch_dims_cover_q_projection_and_recurrent_inner_size() {
        let config = ModelConfig {
            architecture: "qwen35moe".into(),
            n_layers: 40,
            n_heads: 16,
            n_kv_heads: 2,
            embedding_dim: 2048,
            head_dim: 256,
            intermediate_dim: 1024,
            context_length: 128,
            vocab_size: 32_000,
            rms_norm_eps: 1e-6,
            rope_freq_base: 1_000_000.0,
            has_qkv_bias: false,
            sliding_window_size: None,
            sliding_window_pattern: None,
            gate_activation: crate::model::config::GateActivation::SiLU,
            tie_word_embeddings: false,
            logit_scale: None,
            rope_scaling: crate::model::config::RopeScaling::None,
            embed_scale: false,
            rope_freq_base_local: None,
            n_expert: Some(256),
            n_expert_used: Some(8),
            expert_intermediate_dim: Some(512),
            qwen35_full_attention_interval: Some(4),
            qwen35_ssm_conv_kernel: Some(4),
            qwen35_ssm_inner_size: Some(8192),
            qwen35_ssm_state_size: Some(128),
            qwen35_ssm_time_step_rank: Some(32),
            qwen35_ssm_group_count: Some(16),
        };

        let q_dim = config.n_heads as usize * config.head_dim as usize;
        let (gate_dim, up_dim) =
            qwen35_gate_up_scratch_dims(&config, q_dim, config.intermediate_dim as usize);

        assert_eq!(gate_dim, 8192);
        assert_eq!(up_dim, 8192);
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_metal_backend_init() {
        let _backend = MetalBackend::new().unwrap();
    }

    #[test]
    fn test_metal_backend_matmul_small() {
        let backend = MetalBackend::new().unwrap();
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        backend.matmul(&a, &b, &mut c, 2, 2, 2);
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((c[0] - 19.0).abs() < 1e-3);
        assert!((c[1] - 22.0).abs() < 1e-3);
        assert!((c[2] - 43.0).abs() < 1e-3);
        assert!((c[3] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_metal_backend_matvec() {
        let backend = MetalBackend::new().unwrap();
        // A: 3×4, x: 4×1
        #[rustfmt::skip]
        let a = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 3];
        backend.matmul(&a, &x, &mut y, 3, 1, 4);
        // row 0: 1+4+9+16 = 30
        // row 1: 5+12+21+32 = 70
        // row 2: 9+20+33+48 = 110
        assert!((y[0] - 30.0).abs() < 1e-3);
        assert!((y[1] - 70.0).abs() < 1e-3);
        assert!((y[2] - 110.0).abs() < 1e-3);
    }

    #[test]
    fn test_metal_backend_dequant_matmul_f32() {
        // dequant_matmul with F32 dtype should work (passthrough)
        let backend = MetalBackend::new().unwrap();
        let a_f32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let a_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(a_f32.as_ptr() as *const u8, std::mem::size_of_val(&a_f32))
        };
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        backend.dequant_matmul(a_bytes, GgmlType::F32, &b, &mut c, 2, 2, 2);
        // Identity multiplication: [[1,2],[3,4]] * I = [[1,2],[3,4]]
        assert!((c[0] - 1.0).abs() < 1e-3);
        assert!((c[1] - 2.0).abs() < 1e-3);
        assert!((c[2] - 3.0).abs() < 1e-3);
        assert!((c[3] - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_prepare_multi_token_gdn_bh_buffers_matches_legacy_pack_path() {
        let n_tokens = 3;
        let group_count = 2;
        let time_step_rank = 4;
        let state_size = 4;
        let key_dim = group_count * state_size;
        let value_dim = time_step_rank * state_size;
        let conv_dim = 2 * key_dim + value_dim;
        let rms_norm_eps = 1e-5;

        let conv_out: Vec<f32> = (0..n_tokens * conv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let alpha: Vec<f32> = (0..n_tokens * time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let beta: Vec<f32> = (0..n_tokens * time_step_rank)
            .map(|i| 0.2 + (i % 5) as f32 * 0.03)
            .collect();

        let mut expected_q = vec![0.0f32; n_tokens * value_dim];
        let mut expected_k = vec![0.0f32; n_tokens * value_dim];
        let mut expected_v = vec![0.0f32; n_tokens * value_dim];

        for token_idx in 0..n_tokens {
            let conv_start = token_idx * conv_dim;
            let conv_end = conv_start + conv_dim;
            let conv_token = &conv_out[conv_start..conv_end];
            let mut q_lin = conv_token[..key_dim].to_vec();
            let mut k_lin = conv_token[key_dim..2 * key_dim].to_vec();
            let v_lin = &conv_token[2 * key_dim..2 * key_dim + value_dim];

            crate::compute::gdn::l2_norm_heads(&mut q_lin, group_count, state_size, rms_norm_eps);
            crate::compute::gdn::l2_norm_heads(&mut k_lin, group_count, state_size, rms_norm_eps);

            let q_rep =
                crate::compute::gdn::repeat_heads(&q_lin, group_count, time_step_rank, state_size);
            let k_rep =
                crate::compute::gdn::repeat_heads(&k_lin, group_count, time_step_rank, state_size);
            let out_start = token_idx * value_dim;
            let out_end = out_start + value_dim;
            expected_q[out_start..out_end].copy_from_slice(&q_rep);
            expected_k[out_start..out_end].copy_from_slice(&k_rep);
            expected_v[out_start..out_end].copy_from_slice(v_lin);
        }

        let expected_q =
            pack_token_major_to_bhsk(&expected_q, n_tokens, time_step_rank, state_size);
        let expected_k =
            pack_token_major_to_bhsk(&expected_k, n_tokens, time_step_rank, state_size);
        let expected_v =
            pack_token_major_to_bhsk(&expected_v, n_tokens, time_step_rank, state_size);
        let expected_gate = pack_token_major_scalars_to_bhs(&alpha, n_tokens, time_step_rank);
        let expected_beta = pack_token_major_scalars_to_bhs(&beta, n_tokens, time_step_rank);

        let mut actual_q = vec![0.0f32; n_tokens * value_dim];
        let mut actual_k = vec![0.0f32; n_tokens * value_dim];
        let mut actual_v = vec![0.0f32; n_tokens * value_dim];
        let mut actual_gate = vec![0.0f32; n_tokens * time_step_rank];
        let mut actual_beta = vec![0.0f32; n_tokens * time_step_rank];

        prepare_multi_token_gdn_bh_buffers(
            &conv_out,
            &alpha,
            &beta,
            &mut actual_q,
            &mut actual_k,
            &mut actual_v,
            &mut actual_gate,
            &mut actual_beta,
            n_tokens,
            group_count,
            time_step_rank,
            state_size,
            rms_norm_eps,
        );

        for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_v.iter().zip(expected_v.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_gate.iter().zip(expected_gate.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_metal_gdn_prepare_multi_token_qkv_matches_cpu_pack_path() {
        let backend = MetalBackend::new().unwrap();
        let n_tokens = 3usize;
        let group_count = 2usize;
        let time_step_rank = 4usize;
        let state_size = 4usize;
        let key_dim = group_count * state_size;
        let value_dim = time_step_rank * state_size;
        let conv_dim = 2 * key_dim + value_dim;
        let rms_norm_eps = 1e-5f32;

        let conv_out: Vec<f32> = (0..n_tokens * conv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let alpha: Vec<f32> = (0..n_tokens * time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let beta: Vec<f32> = (0..n_tokens * time_step_rank)
            .map(|i| 0.2 + (i % 5) as f32 * 0.03)
            .collect();

        let mut expected_q = vec![0.0f32; n_tokens * value_dim];
        let mut expected_k = vec![0.0f32; n_tokens * value_dim];
        let mut expected_v = vec![0.0f32; n_tokens * value_dim];
        let mut expected_gate = vec![0.0f32; n_tokens * time_step_rank];
        let mut expected_beta = vec![0.0f32; n_tokens * time_step_rank];
        prepare_multi_token_gdn_bh_buffers(
            &conv_out,
            &alpha,
            &beta,
            &mut expected_q,
            &mut expected_k,
            &mut expected_v,
            &mut expected_gate,
            &mut expected_beta,
            n_tokens,
            group_count,
            time_step_rank,
            state_size,
            rms_norm_eps,
        );

        let conv_buf = MetalBuffer::from_slice(backend.device.device(), &conv_out).unwrap();
        let q_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * value_dim * size_of::<f32>(),
        )
        .unwrap();
        let k_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * value_dim * size_of::<f32>(),
        )
        .unwrap();
        let v_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * value_dim * size_of::<f32>(),
        )
        .unwrap();
        let gate_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * time_step_rank * size_of::<f32>(),
        )
        .unwrap();
        let beta_out_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * time_step_rank * size_of::<f32>(),
        )
        .unwrap();

        assert!(
            backend
                .qwen35_prepare_multi_token_qkv_sync(
                    &conv_buf,
                    &alpha,
                    &beta,
                    &q_buf,
                    &k_buf,
                    &v_buf,
                    &gate_buf,
                    &beta_out_buf,
                    n_tokens as u32,
                    group_count as u32,
                    time_step_rank as u32,
                    state_size as u32,
                    rms_norm_eps,
                )
                .unwrap()
        );

        let actual_q = unsafe { q_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
        let actual_k = unsafe { k_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
        let actual_v = unsafe { v_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
        let actual_gate =
            unsafe { gate_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };
        let actual_beta =
            unsafe { beta_out_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };

        for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_v.iter().zip(expected_v.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_gate.iter().zip(expected_gate.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_metal_gdn_prepare_multi_token_qkv_matches_cpu_pack_path_with_f16_alpha_beta_storage() {
        let backend = MetalBackend::new().unwrap();
        let n_tokens = 3usize;
        let group_count = 2usize;
        let time_step_rank = 4usize;
        let state_size = 4usize;
        let key_dim = group_count * state_size;
        let value_dim = time_step_rank * state_size;
        let conv_dim = 2 * key_dim + value_dim;
        let rms_norm_eps = 1e-5f32;

        let conv_out: Vec<f32> = (0..n_tokens * conv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
            .collect();
        let alpha: Vec<f32> = (0..n_tokens * time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let beta: Vec<f32> = (0..n_tokens * time_step_rank)
            .map(|i| 0.2 + (i % 5) as f32 * 0.03)
            .collect();
        let alpha_f16: Vec<half::f16> = alpha.iter().copied().map(half::f16::from_f32).collect();
        let beta_f16: Vec<half::f16> = beta.iter().copied().map(half::f16::from_f32).collect();

        let mut expected_q = vec![0.0f32; n_tokens * value_dim];
        let mut expected_k = vec![0.0f32; n_tokens * value_dim];
        let mut expected_v = vec![0.0f32; n_tokens * value_dim];
        let mut expected_gate = vec![0.0f32; n_tokens * time_step_rank];
        let mut expected_beta = vec![0.0f32; n_tokens * time_step_rank];
        prepare_multi_token_gdn_bh_buffers(
            &conv_out,
            &alpha,
            &beta,
            &mut expected_q,
            &mut expected_k,
            &mut expected_v,
            &mut expected_gate,
            &mut expected_beta,
            n_tokens,
            group_count,
            time_step_rank,
            state_size,
            rms_norm_eps,
        );

        let conv_buf = MetalBuffer::from_slice(backend.device.device(), &conv_out).unwrap();
        let alpha_buf = MetalBuffer::from_slice(backend.device.device(), &alpha_f16).unwrap();
        let beta_buf = MetalBuffer::from_slice(backend.device.device(), &beta_f16).unwrap();
        let q_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * value_dim * size_of::<f32>(),
        )
        .unwrap();
        let k_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * value_dim * size_of::<f32>(),
        )
        .unwrap();
        let v_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * value_dim * size_of::<f32>(),
        )
        .unwrap();
        let gate_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * time_step_rank * size_of::<f32>(),
        )
        .unwrap();
        let beta_out_buf = MetalBuffer::new(
            backend.device.device(),
            n_tokens * time_step_rank * size_of::<f32>(),
        )
        .unwrap();

        backend
            .device
            .execute_sync(|encoder| {
                assert!(
                    backend
                        .gdn_kernels
                        .encode_prepare_multi_token_qkv_alpha_beta_f16(
                            encoder,
                            &conv_buf,
                            &alpha_buf,
                            &beta_buf,
                            &q_buf,
                            &k_buf,
                            &v_buf,
                            &gate_buf,
                            &beta_out_buf,
                            n_tokens as u32,
                            group_count as u32,
                            time_step_rank as u32,
                            state_size as u32,
                            rms_norm_eps,
                        )
                );
                Ok(())
            })
            .unwrap();

        let actual_q = unsafe { q_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
        let actual_k = unsafe { k_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
        let actual_v = unsafe { v_buf.as_slice::<f32>()[..n_tokens * value_dim].to_vec() };
        let actual_gate =
            unsafe { gate_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };
        let actual_beta =
            unsafe { beta_out_buf.as_slice::<f32>()[..n_tokens * time_step_rank].to_vec() };

        for (actual, expected) in actual_q.iter().zip(expected_q.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_k.iter().zip(expected_k.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in actual_v.iter().zip(expected_v.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in actual_gate.iter().zip(expected_gate.iter()) {
            assert!((actual - expected).abs() < 5e-4);
        }
        for (actual, expected) in actual_beta.iter().zip(expected_beta.iter()) {
            assert!((actual - expected).abs() < 5e-4);
        }
    }

    #[test]
    fn test_metal_gdn_unpack_bhsk_to_token_major_matches_cpu() {
        let backend = MetalBackend::new().unwrap();
        let n_tokens = 3usize;
        let n_heads = 4usize;
        let head_dim = 4usize;
        let input: Vec<f32> = (0..n_tokens * n_heads * head_dim)
            .map(|i| i as f32 * 0.25 - 3.0)
            .collect();
        let mut expected = vec![0.0f32; input.len()];
        unpack_bhsk_to_token_major(&input, &mut expected, n_tokens, n_heads, head_dim);

        let input_buf = MetalBuffer::from_slice(backend.device.device(), &input).unwrap();
        let output_buf =
            MetalBuffer::new(backend.device.device(), input.len() * size_of::<f32>()).unwrap();
        backend
            .gdn_kernels
            .unpack_bhsk_to_token_major(
                &backend.device,
                &input_buf,
                &output_buf,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
            )
            .unwrap();
        let actual = unsafe { output_buf.as_slice::<f32>()[..input.len()].to_vec() };

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_metal_backend_batch_dequant_matvec_mixed_dtypes_share_command_buffer() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;

        let k = 32;
        // Q8_0 block: 2 bytes f16 scale + 32 i8 values = 34 bytes, 32 values
        let mut q8_block = [0u8; 34];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        q8_block[0] = d_bytes[0];
        q8_block[1] = d_bytes[1];
        // Fill i8 quantized values with 1 (signed)
        q8_block[2..34].fill(1);

        let f32_weight = [0.5f32; 32];
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                f32_weight.as_ptr() as *const u8,
                std::mem::size_of_val(&f32_weight),
            )
        };

        let x = [1.0f32; 32];
        let mut q8_out = [0.0f32; 1];
        let mut f32_out = [0.0f32; 1];
        let mut expected_q8 = [0.0f32; 1];
        let mut expected_f32 = [0.0f32; 1];

        cpu.dequant_matmul(&q8_block, GgmlType::Q8_0, &x, &mut expected_q8, 1, 1, k);
        cpu.dequant_matmul(f32_bytes, GgmlType::F32, &x, &mut expected_f32, 1, 1, k);

        backend.device.reset_perf_counters();
        backend.batch_dequant_matvec(
            &[
                (&q8_block, GgmlType::Q8_0, 1),
                (f32_bytes, GgmlType::F32, 1),
            ],
            &x,
            k,
            &mut [&mut q8_out, &mut f32_out],
        );
        let counters = backend.device.perf_counters();

        assert!((q8_out[0] - expected_q8[0]).abs() < 1e-2);
        assert!((f32_out[0] - expected_f32[0]).abs() < 1e-3);
        assert_eq!(
            counters.command_buffers, 1,
            "mixed-dtype batch matvec should use one command buffer"
        );
    }

    #[test]
    fn test_metal_backend_safe_batch_dequant_matvec_mixed_dtypes_share_command_buffer() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;

        let k = 32;
        // Q8_0 block: 2 bytes f16 scale + 32 i8 values = 34 bytes, 32 values
        let mut q8_block = [0u8; 34];
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        q8_block[0] = d_bytes[0];
        q8_block[1] = d_bytes[1];
        // Fill i8 quantized values with 1 (signed)
        q8_block[2..34].fill(1);

        let f32_weight = [0.5f32; 32];
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                f32_weight.as_ptr() as *const u8,
                std::mem::size_of_val(&f32_weight),
            )
        };

        let x = [1.0f32; 32];
        let mut q8_out = [0.0f32; 1];
        let mut f32_out = [0.0f32; 1];
        let mut expected_q8 = [0.0f32; 1];
        let mut expected_f32 = [0.0f32; 1];

        cpu.dequant_matmul(&q8_block, GgmlType::Q8_0, &x, &mut expected_q8, 1, 1, k);
        cpu.dequant_matmul(f32_bytes, GgmlType::F32, &x, &mut expected_f32, 1, 1, k);

        backend.device.reset_perf_counters();
        backend.safe_batch_dequant_matvec(
            &[
                (&q8_block, GgmlType::Q8_0, 1),
                (f32_bytes, GgmlType::F32, 1),
            ],
            &x,
            k,
            &mut [&mut q8_out, &mut f32_out],
        );
        let counters = backend.device.perf_counters();

        assert!((q8_out[0] - expected_q8[0]).abs() < 1e-2);
        assert!((f32_out[0] - expected_f32[0]).abs() < 1e-3);
        assert_eq!(
            counters.command_buffers, 1,
            "safe mixed-dtype batch matvec should use one command buffer"
        );
    }

    #[test]
    fn test_hybrid_backend_matvec() {
        let backend = HybridBackend::new().unwrap();
        // N=1 matvec through Hybrid decode path (CPU by default).
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 2.0];
        let mut y = [0.0f32; 2];
        backend.matmul(&a, &x, &mut y, 2, 1, 2);
        // row 0: 1+4 = 5, row 1: 3+8 = 11
        assert!((y[0] - 5.0).abs() < 1e-3);
        assert!((y[1] - 11.0).abs() < 1e-3);
    }

    #[test]
    fn test_hybrid_backend_routes_matmul_to_metal() {
        let backend = HybridBackend::new().unwrap();
        // N=2 (prefill-style) should use Metal path.
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0f32; 4];
        backend.matmul(&a, &b, &mut c, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-3);
        assert!((c[1] - 22.0).abs() < 1e-3);
        assert!((c[2] - 43.0).abs() < 1e-3);
        assert!((c[3] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_metal_backend_fused_q5_k_matvec() {
        let backend = MetalBackend::new().unwrap();

        let m = 4;
        let k = 512;
        let blocks_per_row = k / 256;

        let mut quant_data = Vec::new();
        for row in 0..m {
            for blk in 0..blocks_per_row {
                let mut block = vec![0u8; 176];
                let d_val = (row as f32 + 1.0) * 0.05 + blk as f32 * 0.02;
                let d_bytes = half::f16::from_f32(d_val).to_le_bytes();
                block[0] = d_bytes[0];
                block[1] = d_bytes[1];
                let dmin_val = (blk as f32) * 0.01;
                let dmin_bytes = half::f16::from_f32(dmin_val).to_le_bytes();
                block[2] = dmin_bytes[0];
                block[3] = dmin_bytes[1];
                for i in 0..8 {
                    block[4 + (i % 4)] = ((row + i) % 8 + 1) as u8;
                    block[8 + (i % 4)] = ((blk + i) % 4) as u8;
                }
                for (i, b) in block[16..48].iter_mut().enumerate() {
                    *b = ((row * 5 + blk * 3 + i) % 256) as u8;
                }
                for (i, b) in block[48..176].iter_mut().enumerate() {
                    *b = ((row * 11 + blk * 7 + i) % 256) as u8;
                }
                quant_data.extend(block);
            }
        }

        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01) - 2.56).collect();

        let mut weights = vec![0.0f32; m * k];
        crate::quant::q5_k::dequantize(&quant_data, &mut weights);
        let mut expected = vec![0.0f32; m];
        crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

        let mut result = vec![0.0f32; m];
        backend.dequant_matmul(&quant_data, GgmlType::Q5K, &x, &mut result, m, 1, k);

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < 0.5,
            "Fused Q5_K matvec mismatch: max_diff={diff}, result={result:?}, expected={expected:?}"
        );
    }

    #[test]
    fn test_metal_backend_fused_q6_k_matvec() {
        let backend = MetalBackend::new().unwrap();

        // Create Q6_K data: 4 rows × 512 cols (2 blocks per row)
        let m = 4;
        let k = 512;
        let block_bytes = 210;
        let blocks_per_row = k / 256;

        // Use a simple LCG for deterministic pseudo-random data
        let mut rng_state: u64 = 42;
        let mut next_u8 = || -> u8 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as u8
        };

        let mut quant_data = Vec::new();
        for row in 0..m {
            for _blk in 0..blocks_per_row {
                let mut block = vec![0u8; block_bytes];

                // d = varied scale
                let d_val = (row as f32 + 1.0) * 0.05;
                let d_bytes = half::f16::from_f32(d_val).to_le_bytes();
                block[208] = d_bytes[0];
                block[209] = d_bytes[1];

                // Varied scales (signed i8)
                for i in 0..16 {
                    block[192 + i] = ((i as i8 % 7) - 3) as u8; // range -3..3
                }

                // Varied ql and qh
                for b in block[..128].iter_mut() {
                    *b = next_u8();
                }
                for b in block[128..192].iter_mut() {
                    *b = next_u8();
                }

                quant_data.extend(block);
            }
        }

        // Input vector
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.01) - 2.56).collect();

        // CPU reference
        let mut weights = vec![0.0f32; m * k];
        crate::quant::q6_k::dequantize(&quant_data, &mut weights);
        let mut expected = vec![0.0f32; m];
        crate::compute::matmul::matmul_f32(&weights, &x, &mut expected, m, 1, k);

        // GPU fused dequant+matvec
        let mut result = vec![0.0f32; m];
        backend.dequant_matmul(&quant_data, GgmlType::Q6K, &x, &mut result, m, 1, k);

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < 0.5,
            "Fused Q6_K matvec mismatch: max_diff={diff}, result={result:?}, expected={expected:?}"
        );
    }

    #[test]
    fn test_metal_matches_cpu_larger() {
        let cpu = super::super::cpu::CpuBackend;
        let metal = MetalBackend::new().unwrap();

        let m = 64;
        let n = 32;
        let k = 48;

        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        let mut cpu_c = vec![0.0f32; m * n];
        let mut metal_c = vec![0.0f32; m * n];

        cpu.matmul(&a, &b, &mut cpu_c, m, n, k);
        metal.matmul(&a, &b, &mut metal_c, m, n, k);

        let diff = max_abs_diff(&cpu_c, &metal_c);
        assert!(diff < 0.1, "CPU vs Metal mismatch: max_diff={diff}");
    }

    #[test]
    fn test_metal_backend_attention_prefill() {
        let backend = MetalBackend::new().unwrap();

        // 4 tokens, 2 heads, 2 KV heads, head_dim=4
        let n_tokens = 4;
        let n_heads = 2;
        let n_kv_heads = 2;
        let head_dim = 4;
        let q_size = n_tokens * n_heads * head_dim;
        let kv_size = n_tokens * n_kv_heads * head_dim;

        let q: Vec<f32> = (0..q_size).map(|i| ((i % 5) as f32 - 2.0) * 0.2).collect();
        let k: Vec<f32> = (0..kv_size).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let v: Vec<f32> = (0..kv_size).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect();

        // CPU reference
        let cpu = super::super::cpu::CpuBackend;
        let mut expected = vec![0.0f32; q_size];
        cpu.attention_prefill(
            &q,
            &k,
            &v,
            &mut expected,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        // Metal
        let mut result = vec![0.0f32; q_size];
        backend.attention_prefill(
            &q,
            &k,
            &v,
            &mut result,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < 1e-3,
            "Metal attention vs CPU mismatch: max_diff={diff}"
        );
    }

    #[test]
    fn test_metal_backend_qwen35_causal_conv_sequence_matches_cpu() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;

        let n_tokens = 7;
        let conv_cache_len = 3;
        let conv_dim = 96;
        let kernel_size = conv_cache_len + 1;

        let input: Vec<f32> = (0..n_tokens * conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let kernel: Vec<f32> = (0..kernel_size * conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
            .collect();
        let mut cpu_state: Vec<f32> = (0..conv_cache_len * conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.02)
            .collect();
        let mut metal_state = cpu_state.clone();
        let mut expected = vec![0.0f32; n_tokens * conv_dim];
        let mut result = vec![0.0f32; n_tokens * conv_dim];

        cpu.qwen35_causal_conv_sequence(
            &input,
            &kernel,
            &mut cpu_state,
            &mut expected,
            n_tokens,
            conv_cache_len,
            conv_dim,
        );
        backend.qwen35_causal_conv_sequence(
            &input,
            &kernel,
            &mut metal_state,
            &mut result,
            n_tokens,
            conv_cache_len,
            conv_dim,
        );

        let output_diff = max_abs_diff(&result, &expected);
        let state_diff = max_abs_diff(&metal_state, &cpu_state);
        assert!(
            output_diff < 1e-5,
            "Metal qwen35 causal conv output mismatch: max_diff={output_diff}"
        );
        assert!(
            state_diff < 1e-6,
            "Metal qwen35 causal conv state mismatch: max_diff={state_diff}"
        );
    }

    #[test]
    fn test_metal_backend_qwen35_single_token_fused_gated_delta_matches_cpu() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;

        let group_count = 2usize;
        let time_step_rank = 4usize;
        let state_size = 128usize;
        let key_dim = group_count * state_size;
        let value_dim = time_step_rank * state_size;
        let conv_dim = 2 * key_dim + value_dim;
        let state_len = time_step_rank * state_size * state_size;
        let eps = 1e-5f32;

        let conv_out: Vec<f32> = (0..conv_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
            .collect();
        let gate: Vec<f32> = (0..time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
            .collect();
        let beta: Vec<f32> = (0..time_step_rank)
            .map(|i| 0.1 + (i % 5) as f32 * 0.04)
            .collect();
        let initial_state: Vec<f32> = (0..state_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let mut cpu_state = initial_state.clone();
        let mut q = vec![0.0f32; value_dim];
        let mut k = vec![0.0f32; value_dim];
        let mut v = vec![0.0f32; value_dim];
        let mut expected = vec![0.0f32; value_dim];

        prepare_single_token_gdn_qkv(
            &conv_out,
            &mut q,
            &mut k,
            &mut v,
            group_count,
            time_step_rank,
            state_size,
            eps,
        );
        cpu.qwen35_gated_delta_sequence(
            &q,
            &k,
            &v,
            &gate,
            &beta,
            &mut cpu_state,
            &mut expected,
            1,
            time_step_rank,
            state_size,
        );

        let conv_out_buf = MetalBuffer::from_slice(backend.device.device(), &conv_out).unwrap();
        let gate_buf = MetalBuffer::from_slice(backend.device.device(), &gate).unwrap();
        let beta_buf = MetalBuffer::from_slice(backend.device.device(), &beta).unwrap();
        let state_buf = MetalBuffer::from_slice(backend.device.device(), &initial_state).unwrap();
        let output_buf = MetalBuffer::new(
            backend.device.device(),
            value_dim * std::mem::size_of::<f32>(),
        )
        .unwrap();

        backend
            .device
            .execute_sync(|encoder| {
                anyhow::ensure!(
                    backend.gdn_kernels.encode_single_token_gated_delta_fused(
                        encoder,
                        &conv_out_buf,
                        &gate_buf,
                        &beta_buf,
                        &state_buf,
                        &output_buf,
                        group_count as u32,
                        time_step_rank as u32,
                        state_size as u32,
                        eps,
                    ),
                    "single-token fused gated-delta kernel should support head_dim={state_size}"
                );
                Ok(())
            })
            .unwrap();

        let result = unsafe { output_buf.as_slice::<f32>()[..value_dim].to_vec() };
        let metal_state = unsafe { state_buf.as_slice::<f32>()[..state_len].to_vec() };

        let output_diff = max_abs_diff(&result, &expected);
        let state_diff = max_abs_diff(&metal_state, &cpu_state);
        assert!(
            output_diff < 1e-4,
            "Metal qwen35 single-token fused gated delta output mismatch: max_diff={output_diff}"
        );
        assert!(
            state_diff < 1e-4,
            "Metal qwen35 single-token fused gated delta state mismatch: max_diff={state_diff}"
        );
    }

    #[test]
    fn test_metal_backend_qwen35_gated_delta_sequence_matches_cpu() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;

        let n_tokens = 3;
        let n_heads = 2;
        let head_dim = 128;
        let value_dim = n_heads * head_dim;
        let state_len = n_heads * head_dim * head_dim;

        let q_batch: Vec<f32> = (0..n_tokens * value_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
            .collect();
        let k_batch: Vec<f32> = (0..n_tokens * value_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
            .collect();
        let v_batch: Vec<f32> = (0..n_tokens * value_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let gate_batch: Vec<f32> = (0..n_tokens * n_heads)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
            .collect();
        let beta_batch: Vec<f32> = (0..n_tokens * n_heads)
            .map(|i| 0.1 + (i % 7) as f32 * 0.05)
            .collect();
        let mut cpu_state: Vec<f32> = (0..state_len)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect();
        let mut metal_state = cpu_state.clone();
        let mut expected = vec![0.0f32; n_tokens * value_dim];
        let mut result = vec![0.0f32; n_tokens * value_dim];

        cpu.qwen35_gated_delta_sequence(
            &q_batch,
            &k_batch,
            &v_batch,
            &gate_batch,
            &beta_batch,
            &mut cpu_state,
            &mut expected,
            n_tokens,
            n_heads,
            head_dim,
        );
        backend.qwen35_gated_delta_sequence(
            &q_batch,
            &k_batch,
            &v_batch,
            &gate_batch,
            &beta_batch,
            &mut metal_state,
            &mut result,
            n_tokens,
            n_heads,
            head_dim,
        );

        let output_diff = max_abs_diff(&result, &expected);
        let state_diff = max_abs_diff(&metal_state, &cpu_state);
        assert!(
            output_diff < 1e-4,
            "Metal qwen35 gated delta output mismatch: max_diff={output_diff}"
        );
        assert!(
            state_diff < 1e-4,
            "Metal qwen35 gated delta state mismatch: max_diff={state_diff}"
        );
    }

    #[test]
    fn test_metal_backend_qwen35_gated_delta_sequence_chunked_matches_cpu() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;

        let n_tokens = 96;
        let n_heads = 2;
        let head_dim = 128;
        let value_dim = n_heads * head_dim;
        let state_len = n_heads * head_dim * head_dim;

        let q_batch: Vec<f32> = (0..n_tokens * value_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.04)
            .collect();
        let k_batch: Vec<f32> = (0..n_tokens * value_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.03)
            .collect();
        let v_batch: Vec<f32> = (0..n_tokens * value_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let gate_batch: Vec<f32> = (0..n_tokens * n_heads)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.2)
            .collect();
        let beta_batch: Vec<f32> = (0..n_tokens * n_heads)
            .map(|i| 0.1 + (i % 7) as f32 * 0.05)
            .collect();
        let mut cpu_state: Vec<f32> = (0..state_len)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect();
        let mut metal_state = cpu_state.clone();
        let mut expected = vec![0.0f32; n_tokens * value_dim];
        let mut result = vec![0.0f32; n_tokens * value_dim];

        cpu.qwen35_gated_delta_sequence(
            &q_batch,
            &k_batch,
            &v_batch,
            &gate_batch,
            &beta_batch,
            &mut cpu_state,
            &mut expected,
            n_tokens,
            n_heads,
            head_dim,
        );
        backend.qwen35_gated_delta_sequence(
            &q_batch,
            &k_batch,
            &v_batch,
            &gate_batch,
            &beta_batch,
            &mut metal_state,
            &mut result,
            n_tokens,
            n_heads,
            head_dim,
        );

        let output_diff = max_abs_diff(&result, &expected);
        let state_diff = max_abs_diff(&metal_state, &cpu_state);
        assert!(
            output_diff < 1e-4,
            "Metal qwen35 chunked gated delta output mismatch: max_diff={output_diff}"
        );
        assert!(
            state_diff < 1e-4,
            "Metal qwen35 chunked gated delta state mismatch: max_diff={state_diff}"
        );
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_reuses_slot_buffers() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let slot_indices = [7usize];
        let tokens_per_slot = 2;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 2,
            conv_dim: 192,
            group_count: 1,
            state_size: 64,
            time_step_rank: 1,
            rms_norm_eps: 1e-5,
        };
        let conv_state_stride = cfg.conv_cache_len * cfg.conv_dim;
        let recurrent_state_stride = cfg.time_step_rank * cfg.state_size * cfg.state_size;
        let total_tokens = slot_indices.len() * tokens_per_slot;
        let kernel_len = (cfg.conv_cache_len + 1) * cfg.conv_dim;
        let slot_key = Qwen35RecurrentSlotBufferKey {
            layer_idx: 3,
            slot_idx: slot_indices[0],
            conv_state_stride,
            recurrent_state_stride,
        };

        let qkv: Vec<f32> = (0..total_tokens * cfg.conv_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.02)
            .collect();
        let alpha_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        let beta_input: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
            .map(|i| 0.1 + (i % 7) as f32 * 0.03)
            .collect();
        let dt_bias = vec![0.05f32; cfg.time_step_rank];
        let a = vec![0.02f32; cfg.time_step_rank];
        let kernel: Vec<f32> = (0..kernel_len)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();

        let mut cpu_conv_state = vec![0.0f32; conv_state_stride];
        let mut cpu_recurrent_state = vec![0.0f32; recurrent_state_stride];
        let mut metal_conv_state = cpu_conv_state.clone();
        let mut metal_recurrent_state = cpu_recurrent_state.clone();
        let mut cpu_alpha = alpha_input.clone();
        let mut cpu_beta = beta_input.clone();
        let mut metal_alpha = alpha_input.clone();
        let mut metal_beta = beta_input.clone();
        let mut expected = vec![0.0f32; total_tokens * cfg.value_dim()];
        let mut result = vec![0.0f32; total_tokens * cfg.value_dim()];

        {
            let mut cpu_state_batch = Qwen35RecurrentStateBatch::new(
                slot_key.layer_idx,
                &slot_indices,
                &mut cpu_conv_state,
                &mut cpu_recurrent_state,
                conv_state_stride,
                recurrent_state_stride,
            );
            cpu.qwen35_recurrent_sequence(
                &qkv,
                &mut cpu_beta,
                &mut cpu_alpha,
                &dt_bias,
                &a,
                &kernel,
                &mut cpu_state_batch,
                &mut expected,
                tokens_per_slot,
                cfg,
            );
        }
        {
            let mut metal_state_batch = Qwen35RecurrentStateBatch::new(
                slot_key.layer_idx,
                &slot_indices,
                &mut metal_conv_state,
                &mut metal_recurrent_state,
                conv_state_stride,
                recurrent_state_stride,
            );
            backend.qwen35_recurrent_sequence(
                &qkv,
                &mut metal_beta,
                &mut metal_alpha,
                &dt_bias,
                &a,
                &kernel,
                &mut metal_state_batch,
                &mut result,
                tokens_per_slot,
                cfg,
            );
        }

        assert!(max_abs_diff(&result, &expected) < 1e-4);
        assert!(max_abs_diff(&metal_alpha, &cpu_alpha) < 1e-5);
        assert!(max_abs_diff(&metal_beta, &cpu_beta) < 1e-5);
        assert!(max_abs_diff(&metal_conv_state, &cpu_conv_state) < 1e-5);
        assert!(max_abs_diff(&metal_recurrent_state, &cpu_recurrent_state) < 1e-4);

        {
            let cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
            assert_eq!(
                cache.len(),
                0,
                "state-batch Metal recurrent path should alias CPU state directly without slot buffers"
            );
        }
        let first_scratch_ptrs = {
            let cache = backend.qwen35_recurrent_scratch_buffers.lock().unwrap();
            assert_eq!(cache.len(), 1);
            let scratch_key = Qwen35RecurrentScratchBufferKey {
                tokens_per_slot,
                conv_dim: cfg.conv_dim,
                time_step_rank: cfg.time_step_rank,
                state_size: cfg.state_size,
            };
            let buffers = cache.get(&scratch_key).unwrap();
            (
                buffers.input.ptr_id(),
                buffers.conv_out.ptr_id(),
                buffers.q.ptr_id(),
                buffers.output.ptr_id(),
            )
        };

        {
            let mut metal_state_batch = Qwen35RecurrentStateBatch::new(
                slot_key.layer_idx,
                &slot_indices,
                &mut metal_conv_state,
                &mut metal_recurrent_state,
                conv_state_stride,
                recurrent_state_stride,
            );
            backend.qwen35_recurrent_sequence(
                &qkv,
                &mut metal_beta,
                &mut metal_alpha,
                &dt_bias,
                &a,
                &kernel,
                &mut metal_state_batch,
                &mut result,
                tokens_per_slot,
                cfg,
            );
        }
        let cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        assert_eq!(cache.len(), 0);
        drop(cache);

        let cache = backend.qwen35_recurrent_scratch_buffers.lock().unwrap();
        assert_eq!(cache.len(), 1);
        let scratch_key = Qwen35RecurrentScratchBufferKey {
            tokens_per_slot,
            conv_dim: cfg.conv_dim,
            time_step_rank: cfg.time_step_rank,
            state_size: cfg.state_size,
        };
        let buffers = cache.get(&scratch_key).unwrap();
        assert_eq!(buffers.input.ptr_id(), first_scratch_ptrs.0);
        assert_eq!(buffers.conv_out.ptr_id(), first_scratch_ptrs.1);
        assert_eq!(buffers.q.ptr_id(), first_scratch_ptrs.2);
        assert_eq!(buffers.output.ptr_id(), first_scratch_ptrs.3);
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_single_slot_matches_cpu() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 1usize;
        let slot_indices = [0usize];
        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        expected_kv
            .conv_state_for_slot_mut(0, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = 0.1 + i as f32 * 0.001);
        expected_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = 0.2 + i as f32 * 0.002);
        actual_kv
            .conv_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(expected_kv.conv_state_for_slot(0, layer_idx));
        actual_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(expected_kv.recurrent_state_for_slot(0, layer_idx));

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
            .collect();
        let mut expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.07)
            .collect();
        let mut expected_beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.1 + i as f32 * 0.03)
            .collect();
        let mut actual_alpha = expected_alpha.clone();
        let mut actual_beta = expected_beta.clone();
        let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
        let a = vec![0.2, 0.25, 0.3, 0.35];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let mut expected_out = vec![0.0f32; cfg.value_dim()];
        let mut actual_out = vec![0.0f32; cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut expected_beta,
            &mut expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );
        backend.device.reset_perf_counters();
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta,
            &mut actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut actual_out,
            tokens_per_slot,
            cfg,
        );
        let counters = backend.device.perf_counters();

        assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
        assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
        assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "single-token qwen35 recurrent state should stay backend-owned until an explicit sync"
        );
        backend.sync_qwen35_kv(&mut actual_kv);
        assert!(
            max_abs_diff(
                actual_kv.conv_state_for_slot(0, layer_idx),
                expected_kv.conv_state_for_slot(0, layer_idx),
            ) < 1e-5
        );
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(0, layer_idx),
                expected_kv.recurrent_state_for_slot(0, layer_idx),
            ) < 1e-4
        );
        assert_eq!(
            counters.command_buffers, 1,
            "single-token qwen35 recurrent KV path should only dispatch gated-delta on Metal"
        );
        let slot_key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx: 0,
            conv_state_stride: actual_kv.conv_cache_len() * actual_kv.conv_dim(),
            recurrent_state_stride: actual_kv.recurrent_state_len(),
        };
        let slot_cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        assert_eq!(slot_cache.len(), 1);
        let slot_buffers = slot_cache.get(&slot_key).unwrap();
        assert_eq!(
            slot_buffers.recurrent_synced_generation,
            Some(actual_kv.recurrent_state_generation(0, layer_idx))
        );
        drop(slot_cache);

        let scratch_key = Qwen35RecurrentScratchBufferKey {
            tokens_per_slot,
            conv_dim: cfg.conv_dim,
            time_step_rank: cfg.time_step_rank,
            state_size: cfg.state_size,
        };
        assert_eq!(
            backend
                .qwen35_recurrent_scratch_buffers
                .lock()
                .unwrap()
                .len(),
            1
        );
        assert!(
            backend
                .qwen35_recurrent_scratch_buffers
                .lock()
                .unwrap()
                .contains_key(&scratch_key)
        );
    }

    #[test]
    fn test_metal_ops_qwen35_recurrent_projection_scratch_reuses_buffers_by_shape() {
        let backend = MetalBackend::new().unwrap();

        let first_ptrs =
            backend
                .ops
                .with_qwen35_recurrent_projection_scratch(64, 4096, 2048, 128, |scratch| {
                    (
                        scratch.qkv.ptr_id(),
                        scratch.z.ptr_id(),
                        scratch.beta.ptr_id(),
                        scratch.alpha.ptr_id(),
                    )
                });
        let second_ptrs =
            backend
                .ops
                .with_qwen35_recurrent_projection_scratch(64, 4096, 2048, 128, |scratch| {
                    (
                        scratch.qkv.ptr_id(),
                        scratch.z.ptr_id(),
                        scratch.beta.ptr_id(),
                        scratch.alpha.ptr_id(),
                    )
                });

        assert_eq!(first_ptrs, second_ptrs);

        let third_ptrs =
            backend
                .ops
                .with_qwen35_recurrent_projection_scratch(128, 4096, 2048, 128, |scratch| {
                    (
                        scratch.qkv.ptr_id(),
                        scratch.z.ptr_id(),
                        scratch.beta.ptr_id(),
                        scratch.alpha.ptr_id(),
                    )
                });

        assert_ne!(first_ptrs, third_ptrs);
        let cache = backend
            .ops
            .qwen35_recurrent_projection_scratch_buffers
            .lock()
            .unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_metal_ops_qwen35_batch_projection_scratch_reuses_buffers_by_shape() {
        let backend = MetalBackend::new().unwrap();

        let first_ptrs =
            backend
                .ops
                .with_qwen35_batch_projection_scratch(64, &[4096, 4096, 1024], |scratch| {
                    scratch
                        .outputs
                        .iter()
                        .map(MetalBuffer::ptr_id)
                        .collect::<Vec<_>>()
                });
        let second_ptrs =
            backend
                .ops
                .with_qwen35_batch_projection_scratch(64, &[4096, 4096, 1024], |scratch| {
                    scratch
                        .outputs
                        .iter()
                        .map(MetalBuffer::ptr_id)
                        .collect::<Vec<_>>()
                });

        assert_eq!(first_ptrs, second_ptrs);

        let third_ptrs =
            backend
                .ops
                .with_qwen35_batch_projection_scratch(128, &[4096, 4096, 1024], |scratch| {
                    scratch
                        .outputs
                        .iter()
                        .map(MetalBuffer::ptr_id)
                        .collect::<Vec<_>>()
                });

        assert_ne!(first_ptrs, third_ptrs);
        let cache = backend
            .ops
            .qwen35_batch_projection_scratch_buffers
            .lock()
            .unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_metal_ops_qwen35_batch_logits_scratch_reuses_buffers_by_shape() {
        let backend = MetalBackend::new().unwrap();

        let first_ptrs = backend
            .ops
            .with_qwen35_batch_logits_scratch(4096, 32000, |scratch| {
                vec![
                    scratch.hidden.ptr_id(),
                    scratch.hidden_f16.ptr_id(),
                    scratch.logits.ptr_id(),
                ]
            });
        let second_ptrs = backend
            .ops
            .with_qwen35_batch_logits_scratch(4096, 32000, |scratch| {
                vec![
                    scratch.hidden.ptr_id(),
                    scratch.hidden_f16.ptr_id(),
                    scratch.logits.ptr_id(),
                ]
            });
        let third_ptrs = backend
            .ops
            .with_qwen35_batch_logits_scratch(8192, 32000, |scratch| {
                vec![
                    scratch.hidden.ptr_id(),
                    scratch.hidden_f16.ptr_id(),
                    scratch.logits.ptr_id(),
                ]
            });

        assert_eq!(first_ptrs, second_ptrs);
        assert_ne!(first_ptrs, third_ptrs);
        let cache = backend
            .ops
            .qwen35_batch_logits_scratch_buffers
            .lock()
            .unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_token_keeps_backend_owned_state() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 5usize;
        let slot_indices = [0usize];
        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

        expected_kv
            .conv_state_for_slot_mut(0, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = -0.05 + i as f32 * 0.002);
        expected_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = 0.07 + i as f32 * 0.0015);
        actual_kv
            .conv_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(expected_kv.conv_state_for_slot(0, layer_idx));
        actual_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(expected_kv.recurrent_state_for_slot(0, layer_idx));

        let qkv: Vec<f32> = (0..tokens_per_slot * cfg.conv_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.03)
            .collect();
        let mut expected_alpha: Vec<f32> = (0..tokens_per_slot * cfg.time_step_rank)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.07)
            .collect();
        let mut expected_beta: Vec<f32> = (0..tokens_per_slot * cfg.time_step_rank)
            .map(|i| 0.1 + (i % 11) as f32 * 0.02)
            .collect();
        let mut actual_alpha = expected_alpha.clone();
        let mut actual_beta = expected_beta.clone();
        let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
        let a = vec![0.2, 0.25, 0.3, 0.35];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let mut expected_out = vec![0.0f32; tokens_per_slot * cfg.value_dim()];
        let mut actual_out = vec![0.0f32; tokens_per_slot * cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut expected_beta,
            &mut expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta,
            &mut actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut actual_out,
            tokens_per_slot,
            cfg,
        );

        assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
        assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
        assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
        assert!(
            actual_kv.conv_state_cpu_stale(0, layer_idx),
            "multi-token qwen35 recurrent KV path should keep conv state backend-owned",
        );
        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "multi-token qwen35 recurrent KV path should keep recurrent state backend-owned",
        );
        assert!(
            !backend
                .ops
                .qwen35_recurrent_slot_buffers
                .lock()
                .unwrap()
                .is_empty(),
            "multi-token qwen35 recurrent KV path should populate cached Metal slot buffers",
        );

        backend.sync_qwen35_kv(&mut actual_kv);

        assert!(
            max_abs_diff(
                actual_kv.conv_state_for_slot(0, layer_idx),
                expected_kv.conv_state_for_slot(0, layer_idx),
            ) < 1e-5
        );
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(0, layer_idx),
                expected_kv.recurrent_state_for_slot(0, layer_idx),
            ) < 1e-4
        );
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_token_then_single_token_matches_cpu()
     {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let slot_indices = [0usize];
        let warmup_tokens_per_slot = 4usize;
        let decode_tokens_per_slot = 1usize;
        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

        expected_kv
            .conv_state_for_slot_mut(0, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = -0.03 + i as f32 * 0.0017);
        expected_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = 0.05 + i as f32 * 0.0013);
        actual_kv
            .conv_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(expected_kv.conv_state_for_slot(0, layer_idx));
        actual_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(expected_kv.recurrent_state_for_slot(0, layer_idx));

        let warmup_qkv: Vec<f32> = (0..warmup_tokens_per_slot * cfg.conv_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.028)
            .collect();
        let mut warmup_expected_alpha: Vec<f32> = (0..warmup_tokens_per_slot * cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.045)
            .collect();
        let mut warmup_expected_beta: Vec<f32> = (0..warmup_tokens_per_slot * cfg.time_step_rank)
            .map(|i| 0.09 + (i % 9) as f32 * 0.017)
            .collect();
        let mut warmup_actual_alpha = warmup_expected_alpha.clone();
        let mut warmup_actual_beta = warmup_expected_beta.clone();
        let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
        let a = vec![0.2, 0.25, 0.3, 0.35];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let mut warmup_expected_out = vec![0.0f32; warmup_tokens_per_slot * cfg.value_dim()];
        let mut warmup_actual_out = vec![0.0f32; warmup_tokens_per_slot * cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &warmup_qkv,
            &mut warmup_expected_beta,
            &mut warmup_expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut warmup_expected_out,
            warmup_tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &warmup_qkv,
            &mut warmup_actual_beta,
            &mut warmup_actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut warmup_actual_out,
            warmup_tokens_per_slot,
            cfg,
        );

        assert!(max_abs_diff(&warmup_actual_alpha, &warmup_expected_alpha) < 1e-5);
        assert!(max_abs_diff(&warmup_actual_beta, &warmup_expected_beta) < 1e-5);
        assert!(max_abs_diff(&warmup_actual_out, &warmup_expected_out) < 1e-4);
        assert!(
            actual_kv.conv_state_cpu_stale(0, layer_idx),
            "multi-token warmup should leave conv state backend-owned",
        );
        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "multi-token warmup should leave recurrent state backend-owned",
        );

        let decode_qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.031)
            .collect();
        let mut decode_expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.052)
            .collect();
        let mut decode_expected_beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.07 + i as f32 * 0.019)
            .collect();
        let mut decode_actual_alpha = decode_expected_alpha.clone();
        let mut decode_actual_beta = decode_expected_beta.clone();
        let mut decode_expected_out = vec![0.0f32; cfg.value_dim()];
        let mut decode_actual_out = vec![0.0f32; cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &decode_qkv,
            &mut decode_expected_beta,
            &mut decode_expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut decode_expected_out,
            decode_tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &decode_qkv,
            &mut decode_actual_beta,
            &mut decode_actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut decode_actual_out,
            decode_tokens_per_slot,
            cfg,
        );

        assert!(max_abs_diff(&decode_actual_alpha, &decode_expected_alpha) < 1e-5);
        assert!(max_abs_diff(&decode_actual_beta, &decode_expected_beta) < 1e-5);
        assert!(max_abs_diff(&decode_actual_out, &decode_expected_out) < 1e-4);
        backend.sync_qwen35_kv(&mut actual_kv);
        assert!(
            max_abs_diff(
                actual_kv.conv_state_for_slot(0, layer_idx),
                expected_kv.conv_state_for_slot(0, layer_idx),
            ) < 1e-5
        );
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(0, layer_idx),
                expected_kv.recurrent_state_for_slot(0, layer_idx),
            ) < 1e-4
        );
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_multi_slot_multi_token_keeps_backend_owned_state()
     {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let warmup_tokens_per_slot = 1usize;
        let tokens_per_slot = 2usize;
        let dt_bias = vec![0.03, 0.04, 0.05, 0.06];
        let a = vec![0.2, 0.25, 0.3, 0.35];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();

        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = expected_kv.allocate_recurrent_slot();
        let actual_slot1 = actual_kv.allocate_recurrent_slot();
        assert_eq!(slot1, actual_slot1);

        let slot0_qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let slot1_qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.035)
            .collect();
        let slot0_alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let slot1_alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.06)
            .collect();
        let slot0_beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let slot1_beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.11 + i as f32 * 0.015)
            .collect();

        let mut warmup_slot0_alpha_cpu = slot0_alpha.clone();
        let mut warmup_slot0_beta_cpu = slot0_beta.clone();
        let mut warmup_slot0_alpha_metal = slot0_alpha.clone();
        let mut warmup_slot0_beta_metal = slot0_beta.clone();
        let mut warmup_slot1_alpha_cpu = slot1_alpha.clone();
        let mut warmup_slot1_beta_cpu = slot1_beta.clone();
        let mut warmup_slot1_alpha_metal = slot1_alpha.clone();
        let mut warmup_slot1_beta_metal = slot1_beta.clone();
        let mut warmup_out_cpu = vec![0.0f32; cfg.value_dim()];
        let mut warmup_out_metal = vec![0.0f32; cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &slot0_qkv,
            &mut warmup_slot0_beta_cpu,
            &mut warmup_slot0_alpha_cpu,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &[0usize],
            &mut warmup_out_cpu,
            warmup_tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &slot0_qkv,
            &mut warmup_slot0_beta_metal,
            &mut warmup_slot0_alpha_metal,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &[0usize],
            &mut warmup_out_metal,
            warmup_tokens_per_slot,
            cfg,
        );

        cpu.qwen35_recurrent_sequence_for_kv(
            &slot1_qkv,
            &mut warmup_slot1_beta_cpu,
            &mut warmup_slot1_alpha_cpu,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &[slot1],
            &mut warmup_out_cpu,
            warmup_tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &slot1_qkv,
            &mut warmup_slot1_beta_metal,
            &mut warmup_slot1_alpha_metal,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &[slot1],
            &mut warmup_out_metal,
            warmup_tokens_per_slot,
            cfg,
        );

        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "slot 0 warmup should leave recurrent state backend-owned"
        );
        assert!(
            actual_kv.recurrent_state_cpu_stale(slot1, layer_idx),
            "slot 1 warmup should leave recurrent state backend-owned"
        );

        let slot_indices = [0usize, slot1];
        let total_tokens = slot_indices.len() * tokens_per_slot;
        let qkv: Vec<f32> = (0..total_tokens * cfg.conv_dim)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.025)
            .collect();
        let mut expected_alpha: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.04)
            .collect();
        let mut expected_beta: Vec<f32> = (0..total_tokens * cfg.time_step_rank)
            .map(|i| 0.09 + (i % 13) as f32 * 0.015)
            .collect();
        let mut actual_alpha = expected_alpha.clone();
        let mut actual_beta = expected_beta.clone();
        let mut expected_out = vec![0.0f32; total_tokens * cfg.value_dim()];
        let mut actual_out = vec![0.0f32; total_tokens * cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut expected_beta,
            &mut expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta,
            &mut actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut actual_out,
            tokens_per_slot,
            cfg,
        );

        assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
        assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
        assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
        assert!(
            actual_kv.conv_state_cpu_stale(0, layer_idx),
            "multi-token slot 0 path should keep conv state backend-owned",
        );
        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "multi-token slot 0 path should keep recurrent state backend-owned",
        );
        assert!(
            actual_kv.conv_state_cpu_stale(slot1, layer_idx),
            "multi-token slot 1 path should keep conv state backend-owned",
        );
        assert!(
            actual_kv.recurrent_state_cpu_stale(slot1, layer_idx),
            "multi-token slot 1 path should keep recurrent state backend-owned",
        );

        backend.sync_qwen35_kv(&mut actual_kv);

        assert!(
            max_abs_diff(
                actual_kv.conv_state_for_slot(0, layer_idx),
                expected_kv.conv_state_for_slot(0, layer_idx),
            ) < 1e-5
        );
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(0, layer_idx),
                expected_kv.recurrent_state_for_slot(0, layer_idx),
            ) < 1e-4
        );
        assert!(
            max_abs_diff(
                actual_kv.conv_state_for_slot(slot1, layer_idx),
                expected_kv.conv_state_for_slot(slot1, layer_idx),
            ) < 1e-5
        );
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(slot1, layer_idx),
                expected_kv.recurrent_state_for_slot(slot1, layer_idx),
            ) < 1e-4
        );
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_reloads_after_cpu_mutation() {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 1usize;
        let slot_indices = [0usize];
        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let expected_beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let mut actual_alpha = expected_alpha.clone();
        let mut actual_beta = expected_beta.clone();
        let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
        let a = vec![0.11, 0.13, 0.17, 0.19];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
            .collect();
        let mut warmup_out = vec![0.0f32; cfg.value_dim()];

        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta,
            &mut actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut warmup_out,
            tokens_per_slot,
            cfg,
        );
        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "single-token warmup should leave recurrent state backend-owned"
        );
        backend.sync_qwen35_kv(&mut actual_kv);
        expected_kv
            .conv_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(actual_kv.conv_state_for_slot(0, layer_idx));
        expected_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .copy_from_slice(actual_kv.recurrent_state_for_slot(0, layer_idx));

        actual_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.33);
        actual_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .fill(-0.27);
        expected_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.33);
        expected_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .fill(-0.27);

        let cpu_generation = actual_kv.recurrent_state_generation(0, layer_idx);
        let mut expected_alpha_2 = expected_alpha.clone();
        let mut expected_beta_2 = expected_beta.clone();
        let mut actual_alpha_2 = expected_alpha.clone();
        let mut actual_beta_2 = expected_beta.clone();
        let mut expected_out = vec![0.0f32; cfg.value_dim()];
        let mut actual_out = vec![0.0f32; cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut expected_beta_2,
            &mut expected_alpha_2,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta_2,
            &mut actual_alpha_2,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut actual_out,
            tokens_per_slot,
            cfg,
        );

        assert!(max_abs_diff(&actual_alpha_2, &expected_alpha_2) < 1e-5);
        assert!(max_abs_diff(&actual_beta_2, &expected_beta_2) < 1e-5);
        assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "single-token reload path should again leave recurrent state backend-owned"
        );
        backend.sync_qwen35_kv(&mut actual_kv);
        assert!(
            max_abs_diff(
                actual_kv.conv_state_for_slot(0, layer_idx),
                expected_kv.conv_state_for_slot(0, layer_idx),
            ) < 1e-5
        );
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(0, layer_idx),
                expected_kv.recurrent_state_for_slot(0, layer_idx),
            ) < 1e-4
        );

        let slot_key = Qwen35RecurrentSlotBufferKey {
            layer_idx,
            slot_idx: 0,
            conv_state_stride: actual_kv.conv_cache_len() * actual_kv.conv_dim(),
            recurrent_state_stride: actual_kv.recurrent_state_len(),
        };
        let slot_cache = backend.ops.qwen35_recurrent_slot_buffers.lock().unwrap();
        let slot_buffers = slot_cache.get(&slot_key).unwrap();
        assert!(
            actual_kv.recurrent_state_generation(0, layer_idx) > cpu_generation,
            "qwen35 single-token reload path should advance CPU state generation"
        );
        assert_eq!(
            slot_buffers.recurrent_synced_generation,
            Some(actual_kv.recurrent_state_generation(0, layer_idx))
        );
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_keeps_backend_recurrent_state_on_conv_only_cpu_mutation()
     {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 1usize;
        let slot_indices = [0usize];
        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let expected_alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let expected_beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let mut actual_alpha = expected_alpha.clone();
        let mut actual_beta = expected_beta.clone();
        let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
        let a = vec![0.11, 0.13, 0.17, 0.19];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
            .collect();
        let mut warmup_alpha_cpu = expected_alpha.clone();
        let mut warmup_beta_cpu = expected_beta.clone();
        let mut warmup_out_cpu = vec![0.0f32; cfg.value_dim()];
        let mut warmup_out_metal = vec![0.0f32; cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut warmup_beta_cpu,
            &mut warmup_alpha_cpu,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut warmup_out_cpu,
            tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta,
            &mut actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut warmup_out_metal,
            tokens_per_slot,
            cfg,
        );

        assert!(
            actual_kv.recurrent_state_cpu_stale(0, layer_idx),
            "warmup should leave recurrent state backend-owned"
        );

        expected_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);
        actual_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);

        let mut expected_alpha_2 = expected_alpha.clone();
        let mut expected_beta_2 = expected_beta.clone();
        let mut actual_alpha_2 = expected_alpha.clone();
        let mut actual_beta_2 = expected_beta.clone();
        let mut expected_out = vec![0.0f32; cfg.value_dim()];
        let mut actual_out = vec![0.0f32; cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut expected_beta_2,
            &mut expected_alpha_2,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta_2,
            &mut actual_alpha_2,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut actual_out,
            tokens_per_slot,
            cfg,
        );

        assert!(max_abs_diff(&actual_alpha_2, &expected_alpha_2) < 1e-5);
        assert!(max_abs_diff(&actual_beta_2, &expected_beta_2) < 1e-5);
        assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
        backend.sync_qwen35_kv(&mut actual_kv);
        assert!(
            max_abs_diff(
                actual_kv.conv_state_for_slot(0, layer_idx),
                expected_kv.conv_state_for_slot(0, layer_idx),
            ) < 1e-5
        );
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(0, layer_idx),
                expected_kv.recurrent_state_for_slot(0, layer_idx),
            ) < 1e-4
        );
    }

    #[test]
    fn test_metal_backend_qwen35_recurrent_sequence_for_kv_reloads_cpu_materialized_state_across_fresh_kv_instances()
     {
        let backend = MetalBackend::new().unwrap();
        let cpu = super::super::cpu::CpuBackend;
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 1usize;
        let slot_indices = [0usize];

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
        let a = vec![0.11, 0.13, 0.17, 0.19];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
            .collect();

        let mut stale_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut stale_alpha = alpha.clone();
        let mut stale_beta = beta.clone();
        let mut stale_out = vec![0.0f32; cfg.value_dim()];
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut stale_beta,
            &mut stale_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut stale_kv,
            layer_idx,
            &slot_indices,
            &mut stale_out,
            tokens_per_slot,
            cfg,
        );
        assert!(
            stale_kv.recurrent_state_cpu_stale(0, layer_idx),
            "warmup should leave recurrent state backend-owned"
        );

        let mut expected_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let mut actual_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        expected_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .fill(-0.27);
        actual_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .fill(-0.27);
        expected_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);
        actual_kv.conv_state_for_slot_mut(0, layer_idx).fill(0.41);

        let mut expected_alpha = alpha.clone();
        let mut expected_beta = beta.clone();
        let mut actual_alpha = alpha.clone();
        let mut actual_beta = beta.clone();
        let mut expected_out = vec![0.0f32; cfg.value_dim()];
        let mut actual_out = vec![0.0f32; cfg.value_dim()];

        cpu.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut expected_beta,
            &mut expected_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut expected_kv,
            layer_idx,
            &slot_indices,
            &mut expected_out,
            tokens_per_slot,
            cfg,
        );
        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut actual_beta,
            &mut actual_alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut actual_kv,
            layer_idx,
            &slot_indices,
            &mut actual_out,
            tokens_per_slot,
            cfg,
        );

        assert!(max_abs_diff(&actual_alpha, &expected_alpha) < 1e-5);
        assert!(max_abs_diff(&actual_beta, &expected_beta) < 1e-5);
        assert!(max_abs_diff(&actual_out, &expected_out) < 1e-4);
        backend.sync_qwen35_kv(&mut actual_kv);
        assert!(
            max_abs_diff(
                actual_kv.recurrent_state_for_slot(0, layer_idx),
                expected_kv.recurrent_state_for_slot(0, layer_idx),
            ) < 1e-4
        );
    }

    #[test]
    fn test_metal_backend_sync_qwen35_kv_skips_slot_buffers_for_missing_slots() {
        let backend = MetalBackend::new().unwrap();
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 1usize;
        let mut cached_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let slot1 = cached_kv.allocate_recurrent_slot();
        let slot_indices = [slot1];

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let mut beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
        let a = vec![0.11, 0.13, 0.17, 0.19];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
            .collect();
        let mut out = vec![0.0f32; cfg.value_dim()];

        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut beta,
            &mut alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut cached_kv,
            layer_idx,
            &slot_indices,
            &mut out,
            tokens_per_slot,
            cfg,
        );

        let mut fresh_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        backend.sync_qwen35_kv(&mut fresh_kv);
        assert_eq!(fresh_kv.active_slot(), 0);
        assert_eq!(fresh_kv.seq_len(), 0);
    }

    #[test]
    fn test_metal_backend_sync_qwen35_kv_skips_slot_buffers_for_missing_layers() {
        let backend = MetalBackend::new().unwrap();
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 4usize;
        let tokens_per_slot = 1usize;
        let slot_indices = [0usize];
        let mut larger_kv = crate::kv::Qwen35Kv::new(8, 1, 2, 16, 4, 4, 8, 2, 4, 2);

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let mut beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
        let a = vec![0.11, 0.13, 0.17, 0.19];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
            .collect();
        let mut out = vec![0.0f32; cfg.value_dim()];

        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut beta,
            &mut alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut larger_kv,
            layer_idx,
            &slot_indices,
            &mut out,
            tokens_per_slot,
            cfg,
        );

        let mut smaller_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        backend.sync_qwen35_kv(&mut smaller_kv);
        assert_eq!(smaller_kv.active_slot(), 0);
        assert_eq!(smaller_kv.seq_len(), 0);
    }

    #[test]
    fn test_metal_backend_sync_qwen35_kv_skips_slot_buffers_for_mismatched_shapes() {
        let backend = MetalBackend::new().unwrap();
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 1usize;
        let slot_indices = [0usize];
        let mut old_shape_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let mut beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
        let a = vec![0.11, 0.13, 0.17, 0.19];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
            .collect();
        let mut out = vec![0.0f32; cfg.value_dim()];

        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut beta,
            &mut alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut old_shape_kv,
            layer_idx,
            &slot_indices,
            &mut out,
            tokens_per_slot,
            cfg,
        );

        let mut different_shape_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 12, 3, 4, 2);
        let generation = different_shape_kv.note_backend_recurrent_state_update(0, layer_idx);
        assert_eq!(generation, 2, "fresh kv should advance to generation 2");
        assert!(different_shape_kv.recurrent_state_cpu_stale(0, layer_idx));

        backend.sync_qwen35_kv(&mut different_shape_kv);

        assert!(
            different_shape_kv.recurrent_state_cpu_stale(0, layer_idx),
            "mismatched cached Metal slot buffers must not sync into a different qwen35 shape"
        );
        assert!(
            different_shape_kv
                .recurrent_state_for_slot(0, layer_idx)
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    #[should_panic(
        expected = "cannot snapshot qwen35 recurrent slot while backend-owned recurrent state is not materialized on CPU"
    )]
    fn test_metal_backend_qwen35_snapshot_requires_sync_after_backend_owned_decode() {
        let backend = MetalBackend::new().unwrap();
        let cfg = crate::compute::gdn::Qwen35RecurrentConfig {
            conv_cache_len: 3,
            conv_dim: 16,
            group_count: 2,
            state_size: 2,
            time_step_rank: 4,
            rms_norm_eps: 1e-5,
        };
        let layer_idx = 0usize;
        let tokens_per_slot = 1usize;
        let slot_indices = [0usize];
        let mut qwen_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);

        let qkv: Vec<f32> = (0..cfg.conv_dim)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect();
        let mut alpha: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let mut beta: Vec<f32> = (0..cfg.time_step_rank)
            .map(|i| 0.08 + i as f32 * 0.02)
            .collect();
        let dt_bias = vec![0.02, 0.03, 0.04, 0.05];
        let a = vec![0.11, 0.13, 0.17, 0.19];
        let kernel: Vec<f32> = (0..(cfg.conv_cache_len + 1) * cfg.conv_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.015)
            .collect();
        let mut out = vec![0.0f32; cfg.value_dim()];

        backend.qwen35_recurrent_sequence_for_kv(
            &qkv,
            &mut beta,
            &mut alpha,
            &dt_bias,
            &a,
            &kernel,
            &mut qwen_kv,
            layer_idx,
            &slot_indices,
            &mut out,
            tokens_per_slot,
            cfg,
        );

        assert!(qwen_kv.recurrent_state_cpu_stale(0, layer_idx));
        let _ = qwen_kv.snapshot_active_slot();
    }

    #[test]
    fn test_metal_backend_try_clone_qwen35_recurrent_slot_preserves_backend_owned_state() {
        let backend = MetalBackend::new().unwrap();
        let mut qwen_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let layer_idx = 0usize;
        let dst_slot = qwen_kv.allocate_recurrent_slot();
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();

        qwen_kv.conv_state_for_slot_mut(0, layer_idx).fill(-1.0);
        qwen_kv
            .recurrent_state_for_slot_mut(0, layer_idx)
            .fill(-2.0);

        let expected_conv: Vec<f32> = (0..conv_state_stride)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let expected_recurrent: Vec<f32> = (0..recurrent_state_stride)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let conv_generation = qwen_kv.note_backend_conv_state_update(0, layer_idx);
        let recurrent_generation = qwen_kv.note_backend_recurrent_state_update(0, layer_idx);
        let kv_identity = &qwen_kv as *const crate::kv::Qwen35Kv as usize;
        backend.ops.with_qwen35_recurrent_slot_buffer(
            layer_idx,
            0,
            conv_state_stride,
            recurrent_state_stride,
            |slot_buffers| {
                unsafe {
                    slot_buffers.conv_state.as_mut_slice::<f32>()[..conv_state_stride]
                        .copy_from_slice(&expected_conv);
                    slot_buffers.recurrent_state.as_mut_slice::<f32>()[..recurrent_state_stride]
                        .copy_from_slice(&expected_recurrent);
                }
                slot_buffers.conv_synced_generation = Some(conv_generation);
                slot_buffers.recurrent_synced_generation = Some(recurrent_generation);
                slot_buffers.source_kv_identity = Some(kv_identity);
            },
        );

        assert!(qwen_kv.conv_state_cpu_stale(0, layer_idx));
        assert!(qwen_kv.recurrent_state_cpu_stale(0, layer_idx));
        assert!(backend.try_clone_qwen35_recurrent_slot_from_backend_owned(
            &mut qwen_kv,
            0,
            dst_slot
        ));
        assert!(qwen_kv.conv_state_cpu_stale(dst_slot, layer_idx));
        assert!(qwen_kv.recurrent_state_cpu_stale(dst_slot, layer_idx));
        assert_eq!(
            qwen_kv.conv_state_generation(dst_slot, layer_idx),
            conv_generation
        );
        assert_eq!(
            qwen_kv.recurrent_state_generation(dst_slot, layer_idx),
            recurrent_generation
        );

        backend.sync_qwen35_kv(&mut qwen_kv);

        assert_eq!(
            qwen_kv.conv_state_for_slot(dst_slot, layer_idx),
            expected_conv.as_slice()
        );
        assert_eq!(
            qwen_kv.recurrent_state_for_slot(dst_slot, layer_idx),
            expected_recurrent.as_slice()
        );
    }

    #[test]
    fn test_sync_qwen35_slot_buffers_from_kv_zero_inits_pristine_cpu_state_without_copy() {
        let backend = MetalBackend::new().unwrap();
        let qwen_kv = crate::kv::Qwen35Kv::new(4, 1, 2, 16, 4, 4, 8, 2, 4, 2);
        let layer_idx = 0usize;
        let conv_state_stride = qwen_kv.conv_cache_len() * qwen_kv.conv_dim();
        let recurrent_state_stride = qwen_kv.recurrent_state_len();

        let outcome = backend
            .ops
            .sync_qwen35_slot_buffers_from_kv(&qwen_kv, layer_idx, 0);
        assert!(outcome.used_backend_zero_init);
        assert!(!outcome.used_cpu_materialization);
        assert!(!outcome.used_backend_carryover);

        backend.ops.with_qwen35_recurrent_slot_buffer(
            layer_idx,
            0,
            conv_state_stride,
            recurrent_state_stride,
            |slot_buffers| {
                let conv = unsafe { slot_buffers.conv_state.as_slice::<f32>() };
                let recurrent = unsafe { slot_buffers.recurrent_state.as_slice::<f32>() };
                assert!(conv[..conv_state_stride].iter().all(|&v| v == 0.0));
                assert!(
                    recurrent[..recurrent_state_stride]
                        .iter()
                        .all(|&v| v == 0.0)
                );
                assert_eq!(
                    slot_buffers.conv_synced_generation,
                    Some(qwen_kv.conv_state_generation(0, layer_idx))
                );
                assert_eq!(
                    slot_buffers.recurrent_synced_generation,
                    Some(qwen_kv.recurrent_state_generation(0, layer_idx))
                );
            },
        );
    }

    #[test]
    fn test_configure_for_model_updates_runtime_policy() {
        let backend = MetalBackend::new().unwrap();
        let before = backend.ops.runtime_policy();

        backend
            .configure_for_model("Qwen3-8B", "Q4_K", "qwen3")
            .unwrap();

        let after = backend.ops.runtime_policy();
        let expected = RuntimePolicy::for_model("Qwen3-8B", "Q4_K", "qwen3");

        assert_eq!(
            after.dequant_dispatch_config(),
            expected.dequant_dispatch_config()
        );
        assert_eq!(
            after.attention_dispatch_config(),
            expected.attention_dispatch_config()
        );
        assert_eq!(
            after.batch_prefill_prefers_f16_io(),
            expected.batch_prefill_prefers_f16_io()
        );
        assert_eq!(
            after.batch_prefill_prefers_pair_kernel(),
            expected.batch_prefill_prefers_pair_kernel()
        );
        assert_eq!(
            after.dequant_dispatch_config(),
            expected.dequant_dispatch_config(),
            "configure_for_model should resolve the same runtime policy as direct policy loading"
        );
        assert_eq!(
            before.attention_dispatch_config(),
            RuntimePolicy::resolved_defaults().attention_dispatch_config()
        );
        assert_eq!(
            after.gpu_kv_dtype(4096),
            expected.gpu_kv_dtype(4096),
            "configure_for_model should carry KV precision policy through the backend-local runtime policy"
        );
        assert_eq!(
            after.fused_qkv_prefill_enabled(),
            expected.fused_qkv_prefill_enabled()
        );
        assert_eq!(after.batch_simd_enabled(), expected.batch_simd_enabled());
        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q4K);
    }

    #[test]
    fn test_configure_for_model_sets_runtime_f16in_autotune_quant() {
        let backend = MetalBackend::new().unwrap();
        backend
            .configure_for_model("Qwen3-8B", "q6-k", "qwen3")
            .unwrap();
        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q6K);

        backend
            .configure_for_model("Qwen3-8B", "q5_k_m", "qwen3")
            .unwrap();
        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q5K);

        backend
            .configure_for_model("Qwen3-8B", "unknown", "qwen3")
            .unwrap();
        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q4K);
    }

    #[test]
    fn test_decode_dispatch_configs_for_attend_len_uses_decode_regime_cache() {
        let backend = MetalBackend::new().unwrap();
        let profile: KernelProfile = serde_json::from_str(
            r#"{
                "model": "qwen3-8b",
                "source": "unit",
                "decode_matvec": {
                    "q4_k": {
                        "threadgroup_size": 64,
                        "rows_per_simdgroup": 2,
                        "variant": "nr2"
                    }
                },
                "attention_decode": {
                    "splitk_chunk_size": 256,
                    "splitk_threshold": 1024
                },
                "decode_regimes": {
                    "short_max_attend_len": 384,
                    "short": {
                        "decode_matvec": {
                            "q4_k": {
                                "rows_per_simdgroup": 1,
                                "variant": "base"
                            }
                        }
                    },
                    "long": {
                        "attention_decode": {
                            "splitk_threshold": 256
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        backend
            .ops
            .apply_runtime_policy(RuntimePolicy::from_kernel_profile_for_arch(
                profile.clone(),
                "qwen3",
            ));

        let (short_dequant, short_attention) =
            backend.ops.decode_dispatch_configs_for_attend_len(128);
        let short_profile = profile.effective_decode_profile(128);
        assert_eq!(
            short_dequant,
            DequantDispatchConfig::from_profile(&short_profile)
        );
        assert_eq!(
            short_attention,
            AttentionDispatchConfig::from_profile(&short_profile)
        );

        let (long_dequant, long_attention) =
            backend.ops.decode_dispatch_configs_for_attend_len(4096);
        let long_profile = profile.effective_decode_profile(4096);
        assert_eq!(
            long_dequant,
            DequantDispatchConfig::from_profile(&long_profile)
        );
        assert_eq!(
            long_attention,
            AttentionDispatchConfig::from_profile(&long_profile)
        );
    }

    #[test]
    fn test_configure_for_fingerprint_uses_seed_profile_when_load_autotune_disabled() {
        let _env_lock = lock_env_test();
        let backend = MetalBackend::new().unwrap();
        let temp_cache_dir = unique_temp_cache_dir("seed-profile");
        let _load_autotune = EnvVarGuard::set("AX_METAL_LOAD_AUTOTUNE", "off");
        let _force_retune = EnvVarGuard::set("AX_METAL_FORCE_RETUNE", "off");
        let _cache_dir = EnvVarGuard::set(
            "AX_METAL_TUNE_CACHE_DIR",
            temp_cache_dir
                .to_str()
                .unwrap_or("/tmp/ax-engine-metal-test"),
        );

        let fingerprint = ModelFingerprint {
            model_name: "Qwen3-8B".to_string(),
            architecture: "qwen3".to_string(),
            family: "qwen3".to_string(),
            size_label: "8b".to_string(),
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim: 4096,
            head_dim: 128,
            intermediate_dim: 14336,
            context_length: 4096,
            sliding_window_size: None,
            sliding_window_pattern: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            total_tensor_bytes: 0,
            predominant_quant: "Q4_K".to_string(),
            predominant_layer_quant: "Q4_K".to_string(),
            lm_head_quant: None,
            layer_quant_histogram: vec![],
            has_mixed_layer_quants: false,
            has_q4k_layer_weights: true,
            has_q5k_layer_weights: false,
            has_q6k_layer_weights: false,
            has_q8_layer_weights: false,
            has_f32_layer_weights: false,
        };

        backend.configure_for_fingerprint(&fingerprint).unwrap();

        let after = backend.ops.runtime_policy();
        let expected = RuntimePolicy::for_model("Qwen3-8B", "Q4_K", "qwen3");
        assert_eq!(
            after.dequant_dispatch_config(),
            expected.dequant_dispatch_config()
        );
        assert_eq!(
            after.attention_dispatch_config(),
            expected.attention_dispatch_config()
        );
        assert_eq!(
            after.batch_prefill_prefers_f16_io(),
            expected.batch_prefill_prefers_f16_io()
        );
        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q4K);
    }

    #[test]
    fn test_configure_for_fingerprint_sets_runtime_f16in_autotune_quant() {
        let _env_lock = lock_env_test();
        let backend = MetalBackend::new().unwrap();
        let temp_cache_dir = unique_temp_cache_dir("runtime-quant");
        let _load_autotune = EnvVarGuard::set("AX_METAL_LOAD_AUTOTUNE", "off");
        let _force_retune = EnvVarGuard::set("AX_METAL_FORCE_RETUNE", "off");
        let _cache_dir = EnvVarGuard::set(
            "AX_METAL_TUNE_CACHE_DIR",
            temp_cache_dir
                .to_str()
                .unwrap_or("/tmp/ax-engine-metal-test"),
        );

        let mut fingerprint = ModelFingerprint {
            model_name: "Qwen3-8B".to_string(),
            architecture: "qwen3".to_string(),
            family: "qwen3".to_string(),
            size_label: "8b".to_string(),
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            embedding_dim: 4096,
            head_dim: 128,
            intermediate_dim: 14336,
            context_length: 4096,
            sliding_window_size: None,
            sliding_window_pattern: None,
            n_expert: None,
            n_expert_used: None,
            qwen35_full_attention_interval: None,
            total_tensor_bytes: 0,
            predominant_quant: "Q6_K".to_string(),
            predominant_layer_quant: "Q6_K".to_string(),
            lm_head_quant: None,
            layer_quant_histogram: vec![],
            has_mixed_layer_quants: false,
            has_q4k_layer_weights: false,
            has_q5k_layer_weights: false,
            has_q6k_layer_weights: true,
            has_q8_layer_weights: false,
            has_f32_layer_weights: false,
        };
        backend.configure_for_fingerprint(&fingerprint).unwrap();
        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q6K);

        // Fallback to Q4_K when fingerprint quant labels are unknown.
        fingerprint.predominant_quant = "unknown".to_string();
        fingerprint.predominant_layer_quant = "unknown".to_string();
        fingerprint.has_q6k_layer_weights = false;
        fingerprint.has_q4k_layer_weights = false;
        backend.configure_for_fingerprint(&fingerprint).unwrap();
        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q4K);
    }

    #[test]
    fn test_configure_for_fingerprint_ignores_cached_kv_precision_override() {
        let _env_lock = lock_env_test();
        let backend = MetalBackend::new().unwrap();
        let temp_cache_dir = unique_temp_cache_dir("cached-kv");
        let _load_autotune = EnvVarGuard::set("AX_METAL_LOAD_AUTOTUNE", "on");
        let _force_retune = EnvVarGuard::set("AX_METAL_FORCE_RETUNE", "off");
        let _cache_dir = EnvVarGuard::set(
            "AX_METAL_TUNE_CACHE_DIR",
            temp_cache_dir
                .to_str()
                .unwrap_or("/tmp/ax-engine-metal-test-cache-kv"),
        );

        let fingerprint = test_fingerprint();
        let cache_path = backend.ops.kernel_profile_cache_path(&fingerprint);
        fs::create_dir_all(
            cache_path
                .parent()
                .expect("kernel profile cache path should have parent"),
        )
        .unwrap();
        let mut cached_profile = KernelProfile::default();
        cached_profile.kv_cache.precision = KvPrecisionMode::F32;
        fs::write(
            &cache_path,
            serde_json::to_string_pretty(&cached_profile).unwrap(),
        )
        .unwrap();

        backend.configure_for_fingerprint(&fingerprint).unwrap();

        let runtime_policy = backend.ops.runtime_policy();
        assert_eq!(
            runtime_policy.kv_precision_policy(),
            KvPrecisionPolicy::Auto
        );
        assert_eq!(
            runtime_policy.gpu_kv_dtype(fingerprint.context_length as usize),
            crate::kv::gpu_kv::GpuKvDtype::F16
        );
    }

    #[test]
    fn test_configure_for_fingerprint_ignores_cached_profile_when_load_autotune_disabled() {
        let _env_lock = lock_env_test();
        let backend = MetalBackend::new().unwrap();
        let temp_cache_dir = unique_temp_cache_dir("cached-profile-load-autotune-off");
        let _load_autotune = EnvVarGuard::set("AX_METAL_LOAD_AUTOTUNE", "off");
        let _force_retune = EnvVarGuard::set("AX_METAL_FORCE_RETUNE", "off");
        let _cache_dir = EnvVarGuard::set(
            "AX_METAL_TUNE_CACHE_DIR",
            temp_cache_dir
                .to_str()
                .unwrap_or("/tmp/ax-engine-metal-test-cache-load-autotune-off"),
        );

        let fingerprint = test_fingerprint();
        let cache_path = backend.ops.kernel_profile_cache_path(&fingerprint);
        fs::create_dir_all(
            cache_path
                .parent()
                .expect("kernel profile cache path should have parent"),
        )
        .unwrap();

        let mut cached_profile = KernelProfile::default();
        cached_profile.decode_matvec.insert(
            "q4_k".to_string(),
            MatvecParams {
                threadgroup_size: 64,
                rows_per_simdgroup: 1,
                variant: Some(MatvecProfileVariant::Ilp4),
            },
        );
        fs::write(
            &cache_path,
            serde_json::to_string_pretty(&cached_profile).unwrap(),
        )
        .unwrap();

        backend.configure_for_fingerprint(&fingerprint).unwrap();

        let after = backend.ops.runtime_policy();
        let expected = RuntimePolicy::for_model("Qwen3-8B", "Q4_K", "qwen3");
        assert_eq!(
            after.dequant_dispatch_config(),
            expected.dequant_dispatch_config()
        );
        assert_eq!(
            after.attention_dispatch_config(),
            expected.attention_dispatch_config()
        );
        assert!(!backend.ops.f16in_route_tuned.load(Ordering::Relaxed));
    }

    #[test]
    fn test_configure_for_fingerprint_keeps_runtime_batch_autotune_enabled_when_quant_unknown() {
        let _env_lock = lock_env_test();
        let backend = MetalBackend::new().unwrap();
        let temp_cache_dir = unique_temp_cache_dir("unknown-quant-runtime-autotune");
        let _load_autotune = EnvVarGuard::set("AX_METAL_LOAD_AUTOTUNE", "on");
        let _force_retune = EnvVarGuard::set("AX_METAL_FORCE_RETUNE", "off");
        let _cache_dir = EnvVarGuard::set(
            "AX_METAL_TUNE_CACHE_DIR",
            temp_cache_dir
                .to_str()
                .unwrap_or("/tmp/ax-engine-metal-test-unknown-quant-runtime-autotune"),
        );

        let mut fingerprint = test_fingerprint();
        fingerprint.predominant_quant = "unknown".to_string();
        fingerprint.predominant_layer_quant = "unknown".to_string();
        fingerprint.has_q4k_layer_weights = false;
        fingerprint.has_q5k_layer_weights = false;
        fingerprint.has_q6k_layer_weights = false;
        fingerprint.has_q8_layer_weights = false;

        backend.configure_for_fingerprint(&fingerprint).unwrap();

        assert_eq!(backend.ops.runtime_f16in_autotune_quant(), GgmlType::Q4K);
        assert!(
            !backend.ops.f16in_route_tuned.load(Ordering::Relaxed),
            "runtime batch-route autotune should stay enabled when load-time quant family is unknown"
        );
    }

    #[test]
    fn test_hybrid_backend_attention_prefill() {
        let backend = HybridBackend::new().unwrap();

        let n_tokens = 8;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 32;
        let q_size = n_tokens * n_heads * head_dim;
        let kv_size = n_tokens * n_kv_heads * head_dim;

        let q: Vec<f32> = (0..q_size)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        let k: Vec<f32> = (0..kv_size)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.03)
            .collect();
        let v: Vec<f32> = (0..kv_size)
            .map(|i| ((i % 9) as f32 - 4.0) * 0.08)
            .collect();

        // CPU reference
        let cpu = super::super::cpu::CpuBackend;
        let mut expected = vec![0.0f32; q_size];
        cpu.attention_prefill(
            &q,
            &k,
            &v,
            &mut expected,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        // Hybrid should route to Metal
        let mut result = vec![0.0f32; q_size];
        backend.attention_prefill(
            &q,
            &k,
            &v,
            &mut result,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
        );

        let diff = max_abs_diff(&result, &expected);
        assert!(
            diff < 1e-2,
            "Hybrid attention vs CPU mismatch: max_diff={diff}"
        );
    }
}
